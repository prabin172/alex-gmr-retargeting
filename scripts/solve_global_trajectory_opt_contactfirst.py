#!/usr/bin/env python3
"""Contact-aware GlobalOPT for the contact-first pipeline.

Post-processes a contact-first IK NPZ
(`solve_fbx_canonical_alex_contactfirst.py`) with a two-stage global trajectory
optimizer over ALL frames, made **contact-aware** so feet/hands do not slide off
their contact points (the base GlobalOPT is position-only + contact-blind, which
lets smoothing drift the end-effectors = slip).

  Stage A — closed-form per-joint tridiagonal smoothing (kills velocity spikes).
            Root DOF (qpos[0:7]) untouched. Identical to base GlobalOPT.

  Stage B — sparse global QP (OSQP) over actuated δq of all frames.
    Nothing is a hard equality — every term is a soft weighted cost or a box
    constraint, so the QP is always feasible (reach-limited pushes yield
    gracefully instead of going primal-infeasible):
      * Anchor  = median of the per-frame-IK contact-point world positions over
        each contiguous contact interval (robust to jitter, stays near the IK
        pose). One fixed anchor per interval → no slip.
      * Feet (soft pins, weights): pull foot-body position to the anchor and,
        on planted frames, keep the foot flat (up-axis→world +Z) — both as
        weighted least-squares costs, not equalities.
      * Hands (soft pins, weights): pull the palm contact site to the anchor and
        press the fist down (+X→world −Z, low weight).
      * Self-collision: always-on SLACK-based soft avoidance. Each active
        collision row gets a non-negative slack variable penalised by
        `--collision-penalty` ρ, so genuinely-close links relax via (penalised)
        slack instead of driving the solve infeasible.
      * Trust region + joint-limit box constraints bound each outer SCA step.
      * Tracking of a contacting effector's own role is down-weighted while in
        contact (the anchor governs that point) — mirrors the per-frame
        `skip_pos_roles` suppress.

Usage:
    conda run -n gmr python scripts/solve_global_trajectory_opt_contactfirst.py \\
        --ik-npz outputs/contactfirst/standup_02_contactfirst.npz \\
        --out    outputs/global_opt_contactfirst/standup_02_global_opt.npz \\
        --lambda-smooth 10.0 --lambda-track 1.0 --n-outer 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solve_banded

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"

COLL_MARGIN  = 0.02   # metres
COLL_HOPS    = 2
N_ACT        = 29     # actuated joints (Alex)
Q_ACT_SLICE  = slice(7, None)
DV_ACT_SLICE = slice(6, None)   # actuated columns in velocity space (nv=35)

# Contact effector geometry — mirrors solve_fbx_canonical_alex_contactfirst.py.
# kind: "foot" (hard) pins the body frame; "hand" (soft) pins the palm site.
CONTACT_GEOM = {
    "left_foot":  dict(kind="foot", body="LEFT_FOOT",
                       axis_local=(0.0, 0.0, 1.0),  world_dir=(0.0, 0.0, 1.0)),
    "right_foot": dict(kind="foot", body="RIGHT_FOOT",
                       axis_local=(0.0, 0.0, 1.0),  world_dir=(0.0, 0.0, 1.0)),
    "left_hand":  dict(kind="hand", body="LEFT_GRIPPER_Z_LINK",
                       site="alex_left_palm_contact_site",
                       axis_local=(1.0, 0.0, 0.0),  world_dir=(0.0, 0.0, -1.0)),
    "right_hand": dict(kind="hand", body="RIGHT_GRIPPER_Z_LINK",
                       site="alex_right_palm_contact_site",
                       axis_local=(1.0, 0.0, 0.0),  world_dir=(0.0, 0.0, -1.0)),
}
# Canonical role whose position-tracking is down-weighted while the effector is
# in contact (the anchor governs that point instead).
CONTACT_TRACK_ROLE = {
    "left_foot": "left_foot", "right_foot": "right_foot",
    "left_hand": "left_hand", "right_hand": "right_hand",
}


# ---------------------------------------------------------------------------
# Shared helpers (from base GlobalOPT)
# ---------------------------------------------------------------------------

def _within_k_hops(model, b1, b2, k):
    for b, other in [(b1, b2), (b2, b1)]:
        cur = b
        for _ in range(k):
            cur = int(model.body_parentid[cur])
            if cur == other:
                return True
            if cur == 0:
                break
    return False


def _get_joint_limits(model):
    lo = np.full(N_ACT, -1e6)
    hi = np.full(N_ACT,  1e6)
    act_idx = 0
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:     # free joint, skip
            continue
        if bool(model.jnt_limited[j]):
            lo[act_idx] = float(model.jnt_range[j, 0])
            hi[act_idx] = float(model.jnt_range[j, 1])
        act_idx += 1
        if act_idx == N_ACT:
            break
    return lo, hi


def _delta_stats(qpos):
    dq = np.abs(np.diff(qpos[:, 7:], axis=0))
    mpf = dq.max(axis=1)
    return {"max": float(mpf.max()), "p95": float(np.percentile(mpf, 95)),
            "mean": float(mpf.mean()), "n_spikes_05": int((mpf > 0.5).sum())}


def _collision_stats(model, data, qpos):
    pen = []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        mx = 0.0
        for c in range(data.ncon):
            ct = data.contact[c]
            b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
            if b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS):
                continue
            if ct.dist < 0:
                mx = max(mx, abs(float(ct.dist)))
        pen.append(mx)
    arr = np.array(pen)
    n = int((arr > 0).sum())
    return {"pct": n / len(arr) * 100, "max_pen_cm": float(arr.max()) * 100}


def _tracking_stats(qpos, target_positions, role_to_body, role_names, model, data):
    errs = []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for ri, role in enumerate(role_names):
            if role not in role_to_body:
                continue
            errs.append(float(np.linalg.norm(target_positions[t, ri] - data.xpos[role_to_body[role]])))
    arr = np.array(errs)
    return {"mean": float(arr.mean()), "max": float(arr.max())}


# ---------------------------------------------------------------------------
# Contact anchoring
# ---------------------------------------------------------------------------

def _contact_intervals(flag_col):
    """Contiguous True runs → list of (start, end) inclusive."""
    intervals = []
    t = 0
    n = len(flag_col)
    while t < n:
        if flag_col[t]:
            s = t
            while t < n and flag_col[t]:
                t += 1
            intervals.append((s, t - 1))
        else:
            t += 1
    return intervals


def _resolve_contact_geom(model, eff_names, contact_sites):
    """Resolve body/site ids for each present effector; skip unresolved."""
    resolved = {}
    for eff in eff_names:
        if eff not in CONTACT_GEOM:
            continue
        g = CONTACT_GEOM[eff]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, g["body"])
        if bid < 0:
            print(f"  [warn] body {g['body']} not found — skipping {eff}")
            continue
        sid = -1
        if g["kind"] == "hand":
            sname = contact_sites.get(eff, g.get("site"))
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                print(f"  [warn] site {sname} not found — skipping {eff}")
                continue
        resolved[eff] = dict(
            body_id=bid, site_id=sid, kind=g["kind"],
            axis_local=np.asarray(g["axis_local"]), world_dir=np.asarray(g["world_dir"]),
        )
    return resolved


def _contact_point(data, info):
    return data.site_xpos[info["site_id"]].copy() if info["kind"] == "hand" \
        else data.xpos[info["body_id"]].copy()


def _compute_anchors(model, data, qpos_ik, eff_names, flags, resolved, fps,
                     plant_speed, foot_w, hand_w, move_ratio, plant_min_run=8):
    """Per effector, per frame: contact target (T,3), weight (T,), planted flag (T,).

    Contact intervals are NOT stationary plants (a foot/hand can reposition ~30 cm
    while staying labelled in-contact). So within each interval we split into
    *stationary sub-segments* (IK contact-point speed < plant_speed) and anchor
    each to its own median (high weight, planted=True). Non-stationary contact
    frames follow the per-frame IK contact point at a low weight (just enough to
    stop smoothing from adding drift). NaN target / 0 weight = not in contact.

    A stillness sub-segment shorter than `plant_min_run` frames is NOT treated as a
    plant — it is reclassified as moving (low weight, follows IK). This debounces
    momentary speed dips (e.g. a velocity zero-crossing while a hand swings/lifts
    off during a get-up): a 1-frame 'plant' anchored to that instant otherwise sits
    metres of smoothed motion away, inflating the plant-slip metric with a phantom
    (standup_side_05 right_hand: 14.7 cm from 25 single-frame blips → 2.2 cm real).
    Mirrors the Stage-3 contact_min_run debounce, but for the stillness split."""
    T = qpos_ik.shape[0]
    pts = {eff: np.full((T, 3), np.nan) for eff in resolved}
    for t in range(T):
        data.qpos[:] = qpos_ik[t]
        mujoco.mj_forward(model, data)
        for eff, info in resolved.items():
            pts[eff][t] = _contact_point(data, info)

    tgt = {eff: np.full((T, 3), np.nan) for eff in resolved}
    wgt = {eff: np.zeros(T) for eff in resolved}
    planted = {eff: np.zeros(T, bool) for eff in resolved}

    for eff, info in resolved.items():
        col = flags[:, eff_names.index(eff)]
        w_plant = foot_w if info["kind"] == "foot" else hand_w
        w_move = w_plant * move_ratio
        p = pts[eff]
        speed = np.zeros(T)
        speed[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
        speed[0] = speed[1] if T > 1 else 0.0
        for (s, e) in _contact_intervals(col):
            still = speed[s:e + 1] < plant_speed          # (L,) within interval
            k = s
            while k <= e:
                if still[k - s]:                          # start of a stillness run
                    j = k
                    while j <= e and still[j - s]:
                        j += 1
                    if j - k >= plant_min_run:            # long enough → real plant
                        med = np.median(p[k:j], axis=0)
                        tgt[eff][k:j] = med
                        wgt[eff][k:j] = w_plant
                        planted[eff][k:j] = True
                    else:                                 # too short → treat as moving
                        tgt[eff][k:j] = p[k:j]
                        wgt[eff][k:j] = w_move
                    k = j
                else:                                     # repositioning frame
                    tgt[eff][k] = p[k]
                    wgt[eff][k] = w_move
                    k += 1
    return tgt, wgt, planted


def _contact_slip_stats(model, data, qpos, tgt, wgt, planted, resolved):
    """Drift of the contact point off its target, split planted vs moving, plus
    mean foot-flat angle over planted foot frames."""
    slip_p, slip_m, flat = [], [], []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for eff, info in resolved.items():
            a = tgt[eff][t]
            if np.isnan(a[0]):
                continue
            disp = float(np.linalg.norm(_contact_point(data, info) - a))
            (slip_p if planted[eff][t] else slip_m).append(disp)
            if info["kind"] == "foot" and planted[eff][t]:
                R = data.xmat[info["body_id"]].reshape(3, 3)
                cos = float(np.clip(np.dot(R @ info["axis_local"], info["world_dir"]), -1, 1))
                flat.append(np.degrees(np.arccos(cos)))
    return {
        "plant_slip_max_cm": float(np.max(slip_p)) * 100 if slip_p else 0.0,
        "move_slip_max_cm": float(np.max(slip_m)) * 100 if slip_m else 0.0,
        "flat_mean_deg": float(np.mean(flat)) if flat else 0.0,
    }


# ---------------------------------------------------------------------------
# Stage A — closed-form smoothing
# ---------------------------------------------------------------------------

def _banded_smoother(T, lambda_track, lambda_smooth):
    dtd_main = np.full(T, 2.0); dtd_main[0] = dtd_main[-1] = 1.0
    ab = np.zeros((3, T))
    ab[1, :] = lambda_track + lambda_smooth * dtd_main
    ab[0, 1:] = -lambda_smooth
    ab[2, :-1] = -lambda_smooth
    return ab


def _smooth_channel(sig, ab, lambda_track):
    return solve_banded((1, 1), ab, lambda_track * sig)


def stage_a(qpos_ik, lambda_track, lambda_smooth, q_lo, q_hi,
            smooth_root=True, root_lambda_smooth=None):
    """Closed-form per-channel tridiagonal smoothing.

    Actuated joints (qpos[7:]) always smoothed. With smooth_root, the free-base
    root is ALSO smoothed — position (qpos[0:3]) with the same tridiagonal solver
    and the quaternion (qpos[3:7]) via hemisphere-aligned component smoothing +
    renormalise. Without this the root passes through jumpy (per-frame IK root has
    ~3cm / 10deg per-frame pops that read as the whole body flicking)."""
    T = qpos_ik.shape[0]
    ab = _banded_smoother(T, lambda_track, lambda_smooth)
    out = qpos_ik.copy()
    for j in range(N_ACT):
        out[:, 7 + j] = np.clip(_smooth_channel(qpos_ik[:, 7 + j], ab, lambda_track),
                                q_lo[j], q_hi[j])
    if smooth_root:
        rls = lambda_smooth if root_lambda_smooth is None else root_lambda_smooth
        abr = _banded_smoother(T, lambda_track, rls)
        for j in range(3):                        # root position
            out[:, j] = _smooth_channel(qpos_ik[:, j], abr, lambda_track)
        Q = qpos_ik[:, 3:7].copy()                # root quaternion (wxyz)
        for t in range(1, T):                     # hemisphere continuity
            if np.dot(Q[t], Q[t - 1]) < 0:
                Q[t] = -Q[t]
        Qs = np.column_stack([_smooth_channel(Q[:, k], abr, lambda_track) for k in range(4)])
        n = np.linalg.norm(Qs, axis=1, keepdims=True); n[n < 1e-9] = 1.0
        out[:, 3:7] = Qs / n
    return out


# ---------------------------------------------------------------------------
# Stage B — sparse QP: tracking + smoothness + collision + contact
# ---------------------------------------------------------------------------

def _build_smoothness_hessian(T, lambda_smooth):
    N = T * N_ACT
    rows, cols, vals = [], [], []
    for t in range(T):
        start = t * N_ACT
        scale = lambda_smooth * (2.0 if 0 < t < T - 1 else 1.0)
        for j in range(N_ACT):
            rows.append(start + j); cols.append(start + j); vals.append(scale)
        if t > 0:
            prev = (t - 1) * N_ACT
            for j in range(N_ACT):
                rows.append(start + j); cols.append(prev + j); vals.append(-lambda_smooth)
                rows.append(prev + j); cols.append(start + j); vals.append(-lambda_smooth)
    return sp.csc_matrix((vals, (rows, cols)), shape=(N, N))


def _blocks_to_sparse(H_blocks, N):
    r, c, v = [], [], []
    for t, Hb in enumerate(H_blocks):
        s = t * N_ACT
        nz = np.argwhere(np.abs(Hb) > 1e-15)
        for i, j in nz:
            r.append(s + i); c.append(s + j); v.append(Hb[i, j])
    return sp.csc_matrix((v, (r, c)), shape=(N, N))


def _build_tracking(qpos_warm, target_positions, role_names, role_to_body,
                    target_weights, model, data, lambda_track,
                    downweight_roles, downweight_factor):
    """Σ_t Σ_r w_r ||J_r δq_t - e_r||²  (position). Returns (H_blocks, g_dense)."""
    T = qpos_warm.shape[0]
    N = T * N_ACT
    nv = model.nv
    H_blocks = [np.zeros((N_ACT, N_ACT)) for _ in range(T)]
    g = np.zeros(N)
    for t in range(T):
        data.qpos[:] = qpos_warm[t]
        mujoco.mj_forward(model, data)
        skip = downweight_roles[t]
        for ri, role in enumerate(role_names):
            if role not in role_to_body:
                continue
            w = lambda_track * target_weights.get(role, 1.0)
            if role in skip:
                w *= downweight_factor
            bid = role_to_body[role]
            e = target_positions[t, ri] - data.xpos[bid]
            jacp = np.zeros((3, nv))
            mujoco.mj_jac(model, data, jacp, None, data.xpos[bid], bid)
            J = jacp[:, DV_ACT_SLICE]
            H_blocks[t] += w * (J.T @ J)
            g[t * N_ACT:(t + 1) * N_ACT] += -w * (J.T @ e)
    return H_blocks, g


def _build_contact(qpos_warm, tgt, wgt, planted, resolved, model, data,
                   foot_flat_w, fist_w):
    """All-soft contact terms into H_blocks/g:
        * position: w_t ||J_pt δq - (target - p)||²   (per-frame weight from wgt)
        * foot-flat: foot up-axis → world +Z, weight foot_flat_w on planted frames
        * fist-down: gripper +X → world −Z, weight fist_w while a hand is in contact
    Soft everywhere → the QP is always feasible (reach-limited pushes yield
    gracefully instead of going infeasible)."""
    T = qpos_warm.shape[0]
    N = T * N_ACT
    nv = model.nv
    H_blocks = [np.zeros((N_ACT, N_ACT)) for _ in range(T)]
    g = np.zeros(N)

    def add_soft(J, e, w, t):
        H_blocks[t] += w * (J.T @ J)
        g[t * N_ACT:(t + 1) * N_ACT] += -w * (J.T @ e)

    for t in range(T):
        data.qpos[:] = qpos_warm[t]
        mujoco.mj_forward(model, data)
        for eff, info in resolved.items():
            a = tgt[eff][t]
            if np.isnan(a[0]):
                continue
            bid = info["body_id"]
            jacp = np.zeros((3, nv)); jacr = np.zeros((3, nv))
            if info["kind"] == "hand":
                mujoco.mj_jacSite(model, data, jacp, jacr, info["site_id"])
                p = data.site_xpos[info["site_id"]]
            else:
                mujoco.mj_jac(model, data, jacp, jacr, data.xpos[bid], bid)
                p = data.xpos[bid]
            Jp = jacp[:, DV_ACT_SLICE]; Jr = jacr[:, DV_ACT_SLICE]
            R = data.xmat[bid].reshape(3, 3)
            err_rot = np.cross(R @ info["axis_local"], info["world_dir"])

            add_soft(Jp, a - p, wgt[eff][t], t)              # position pin
            if info["kind"] == "foot":
                if planted[eff][t]:
                    add_soft(Jr, err_rot, foot_flat_w, t)     # foot-flat (planted)
            else:
                add_soft(Jr, err_rot, fist_w, t)              # fist-down
    return H_blocks, g


def _build_collision(qpos_warm, model, data, lambda_coll):
    T = qpos_warm.shape[0]
    nv = model.nv
    sqw = float(np.sqrt(lambda_coll))
    r, c, v, l, u = [], [], [], [], []
    row = 0
    for t in range(T):
        data.qpos[:] = qpos_warm[t]
        mujoco.mj_forward(model, data)
        for cc in range(data.ncon):
            ct = data.contact[cc]
            b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
            if b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS):
                continue
            pen = COLL_MARGIN - float(ct.dist)
            if pen <= 0:
                continue
            normal = ct.frame[:3].copy()
            if float(np.dot(normal, data.xpos[b1] - data.xpos[b2])) < 0:
                normal = -normal
            j1 = np.zeros((3, nv)); j2 = np.zeros((3, nv))
            mujoco.mj_jac(model, data, j1, None, ct.pos, b1)
            mujoco.mj_jac(model, data, j2, None, ct.pos, b2)
            jsep = (normal @ (j1 - j2))[DV_ACT_SLICE]
            if np.linalg.norm(jsep) < 1e-9:
                continue
            cs = t * N_ACT
            for j in range(N_ACT):
                if abs(jsep[j]) > 1e-12:
                    r.append(row); c.append(cs + j); v.append(sqw * jsep[j])
            l.append(sqw * min(pen, 0.05)); u.append(1e6); row += 1
    if row == 0:
        return None, None, None
    return sp.csc_matrix((v, (r, c)), shape=(row, T * N_ACT)), np.array(l), np.array(u)


def stage_b(qpos_warm, target_positions, role_names, role_to_body, target_weights,
            tgt, wgt, planted, resolved, downweight_roles,
            model, data, q_lo, q_hi,
            lambda_track, lambda_smooth, lambda_coll,
            foot_flat_w, fist_w, downweight_factor, n_outer, trust,
            collision_penalty=1000.0):
    T = qpos_warm.shape[0]
    N = T * N_ACT
    q_warm_act = qpos_warm[:, 7:].reshape(-1)
    print(f"  Stage B: T={T} variables={N} n_outer={n_outer} trust={trust} "
          f"soft_collision=ON penalty={collision_penalty}")

    H_smooth = _build_smoothness_hessian(T, lambda_smooth)
    A_jl = sp.eye(N, format="csc")
    qpos_cur = qpos_warm.copy()
    delta = np.zeros(N)
    jl_lo_abs = np.tile(q_lo, T) - q_warm_act
    jl_hi_abs = np.tile(q_hi, T) - q_warm_act

    # Keep-best-iterate: the SCA outer loop oscillates — an outer that starts
    # collision-free drops all collision rows and takes an unconstrained
    # tracking+smoothing step straight back into penetration. Returning the LAST
    # iterate unconditionally makes the result depend on n_outer parity (odd
    # happened to land on a resolving step, even on a bad victory-lap step).
    # Instead track the best iterate and return that — parity-immune, never
    # worse than the Stage-A warm start (which seeds it).
    #
    # Score is slip-AWARE, not penetration-only: stronger contact pins cut plant
    # slip but hold the effector where it mildly penetrates, so a pure-penetration
    # argmin would silently trade slip back. Lexicographic:
    #   1) hard = penetration beyond PEN_TOL cm — a self-collision failure that is
    #      never traded for slip;
    #   2) pen + slip — among acceptable-penetration iterates, minimise total drift.
    PEN_TOL = 1.0  # cm; sub-tol penetration is soft-collision slack noise
    def _iter_score(q):
        cs = _collision_stats(model, data, q)
        ss = _contact_slip_stats(model, data, q, tgt, wgt, planted, resolved)
        pen = cs["max_pen_cm"]; slip = ss["plant_slip_max_cm"]
        hard = max(0.0, pen - PEN_TOL)
        return (hard, pen + slip, cs["pct"], pen, slip)
    best_qpos = qpos_cur.copy()
    best_score = _iter_score(best_qpos)
    print(f"    warm: pen={best_score[3]:.2f}cm slip={best_score[4]:.1f}cm coll={best_score[2]:.1f}%")

    for outer in range(n_outer):
        t0 = time.time()
        Ht, gt = _build_tracking(qpos_cur, target_positions, role_names, role_to_body,
                                 target_weights, model, data, lambda_track,
                                 downweight_roles, downweight_factor)
        Hc, gc = _build_contact(qpos_cur, tgt, wgt, planted, resolved,
                                model, data, foot_flat_w, fist_w)
        H_task = _blocks_to_sparse([Ht[t] + Hc[t] for t in range(T)], N)
        P = 2.0 * (H_task + H_smooth)
        q_vec = gt + gc

        # Trust region: keep this iterate's δQ within `trust` of the previous one
        # (SCA stabiliser — stops the collision re-linearisation from oscillating).
        jl_lo = np.maximum(jl_lo_abs, delta - trust)
        jl_hi = np.minimum(jl_hi_abs, delta + trust)

        A_coll, l_coll, u_coll = _build_collision(qpos_cur, model, data, lambda_coll)
        n_coll_rows = 0 if A_coll is None else A_coll.shape[0]

        slack_info = ""
        # Slack-based SOFT collision (always on): augment the decision vector with
        # one slack var per collision row so the QP is always feasible.
        # Genuinely-close links relax via (penalised) slack instead of driving
        # OSQP primal-infeasible.
        m = n_coll_rows
        if m > 0:
            # P: block-diag [ 2*(H_task+H_smooth) , 0 ; 0 , 2*rho*I_m ]
            P_slack = sp.diags(np.full(m, 2.0 * collision_penalty), format="csc")
            P_aug = sp.block_diag([P, P_slack], format="csc")
            q_aug = np.concatenate([q_vec, np.zeros(m)])
            # joint-limit rows: [eye_N, 0_{N x m}]
            A_jl_aug = sp.hstack([A_jl, sp.csc_matrix((N, m))], format="csc")
            # collision rows: [sqw*jsep , +I_m] ; sqw*jsep·δq + s_i >= l_i
            A_coll_aug = sp.hstack([A_coll, sp.eye(m, format="csc")], format="csc")
            # slack non-negativity: [0_{m x N}, eye_m], 0 <= s <= 1e6
            A_slack = sp.hstack([sp.csc_matrix((m, N)), sp.eye(m, format="csc")], format="csc")
            A = sp.vstack([A_jl_aug, A_coll_aug, A_slack], format="csc")
            l = np.concatenate([jl_lo, l_coll, np.zeros(m)])
            u = np.concatenate([jl_hi, u_coll, np.full(m, 1e6)])
            P_use, q_use = P_aug, q_aug
        else:
            # no collision rows this iter → plain joint-limit QP (no slack)
            P_use, q_use = P, q_vec
            A, l, u = A_jl, jl_lo, jl_hi

        prob = osqp.OSQP()
        # max_iter scaled up from 8000: at the native 120 Hz solve the QP is ~4x
        # larger (T up to ~3800) and the x16 smoothness Hessian is stiffer, so
        # 8000 iters left the harder clips at "solved inaccurate" (step discarded
        # below -> Stage B no-op). 20000 lets them reach full "solved".
        prob.setup(P_use.tocsc(), q_use, A, l, u, warm_starting=True, verbose=False,
                   eps_abs=1e-4, eps_rel=1e-4, max_iter=20000, polish=True)
        res = prob.solve()

        # NOTE: OSQP (>=1.x) reports the inaccurate status as "solved inaccurate"
        # (space), NOT "solved_inaccurate" — accept both. An inaccurate solve is a
        # within-10x-tolerance improving step (trust-region-bounded); discarding
        # it silently no-op'd Stage B on the larger 120 Hz problems.
        pen_info = ""
        if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
            print(f"    outer {outer+1}/{n_outer}: OSQP {res.info.status} — keep previous")
        else:
            delta = res.x[:N]
            q_act = np.clip((q_warm_act + delta).reshape(T, N_ACT), q_lo, q_hi)
            qpos_cur[:, 7:] = q_act
            if n_coll_rows > 0:
                s = res.x[N:N + n_coll_rows]
                slack_info = (f" slack_max={float(np.abs(s).max()):.4f} "
                              f"slack_active={int((np.abs(s) > 1e-4).sum())}/{n_coll_rows}")
            score = _iter_score(qpos_cur)
            keep = score < best_score
            pen_info = (f" pen={score[3]:.2f}cm slip={score[4]:.1f}cm coll={score[2]:.1f}%"
                        + (" *best" if keep else ""))
            if keep:
                best_score = score
                best_qpos = qpos_cur.copy()
        print(f"    outer {outer+1}/{n_outer}: coll_rows={n_coll_rows} status={res.info.status} "
              f"|dQ|max={np.abs(delta).max():.4f} time={time.time()-t0:.1f}s" + slack_info + pen_info)
    print(f"    Stage B best: pen={best_score[3]:.2f}cm slip={best_score[4]:.1f}cm coll={best_score[2]:.1f}%")
    return best_qpos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _stats_row(label, d, c, tr, cs):
    print(f"  {label:22s} spikes={d['n_spikes_05']:3d} max_dq={d['max']:.3f} "
          f"p95_dq={d['p95']:.3f} coll={c['pct']:5.1f}% peak={c['max_pen_cm']:.1f}cm "
          f"track={tr['mean']:.4f}m plant_slip={cs['plant_slip_max_cm']:.1f}cm "
          f"flat={cs['flat_mean_deg']:.1f}deg")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ik-npz", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_DEFAULT, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lambda-track", type=float, default=1.0)
    ap.add_argument("--lambda-smooth", type=float, default=10.0)
    ap.add_argument("--lambda-coll", type=float, default=5.0)
    ap.add_argument("--no-root-smooth", action="store_true",
                    help="Do NOT smooth the free-base root (qpos[0:7]). By default the root "
                         "IS smoothed (pos + quaternion) — its per-frame IK jumps (~3cm/10deg) "
                         "otherwise read as the whole body flicking.")
    ap.add_argument("--root-smooth", type=float, default=0.0,
                    help="Root smoothing weight (0 = use --lambda-smooth). Lower = gentler root "
                         "smoothing (less risk of feet sliding as the root moves).")
    ap.add_argument("--foot-weight", type=float, default=40.0,
                    help="Soft weight pinning a PLANTED foot to its stationary anchor.")
    ap.add_argument("--hand-weight", type=float, default=8.0,
                    help="Soft weight pinning a PLANTED palm site to its anchor.")
    ap.add_argument("--move-ratio", type=float, default=0.15,
                    help="Weight factor for non-stationary (repositioning) contact frames.")
    ap.add_argument("--plant-speed", type=float, default=0.05,
                    help="IK contact-point speed (m/s) below which a contact frame is "
                         "treated as a stationary plant.")
    ap.add_argument("--plant-min-run", type=int, default=8,
                    help="Minimum length (frames) of a stillness sub-segment before it "
                         "counts as a plant; shorter dips are reclassified as moving. "
                         "Debounces momentary speed zero-crossings that otherwise create "
                         "phantom 1-frame plants (frame-count knob → scale with fps).")
    ap.add_argument("--foot-flat-weight", type=float, default=3.0,
                    help="Soft weight for foot-flat (up-axis→+Z) on planted foot frames.")
    ap.add_argument("--fist-weight", type=float, default=0.8,
                    help="Soft weight for fist-down (+X→−Z) while a hand is in contact.")
    ap.add_argument("--contact-downweight", type=float, default=0.1,
                    help="Factor applied to a contacting effector's own tracking weight.")
    ap.add_argument("--n-outer", type=int, default=0,
                    help="SCA outer iters for Stage B (contact-pin QP). Default 0 = "
                         "Stage A only, which is the robust win (spikes→0, collisions "
                         "down, tracking preserved). Stage B is EXPERIMENTAL: on the "
                         "current loosely-labelled contacts it fights non-stationary "
                         "'plants' and can regress collisions — enable + tune only once "
                         "contact detection isolates true stationary plants.")
    ap.add_argument("--trust", type=float, default=0.15,
                    help="Stage B trust-region: max change in δq per outer iter (rad).")
    ap.add_argument("--collision-penalty", type=float, default=1000.0,
                    help="Quadratic penalty weight ρ on the always-on collision slack "
                         "variables (Stage B soft self-collision).")
    args = ap.parse_args()

    z = np.load(args.ik_npz, allow_pickle=True)
    qpos_ik = np.asarray(z["qpos"], dtype=np.float64)
    target_positions = np.asarray(z["target_positions"], dtype=np.float64)
    role_names = [str(r) for r in z["role_names"]]
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    T = qpos_ik.shape[0]

    eff_names = [str(x) for x in z["contact_effector_names"]] if "contact_effector_names" in z.files else []
    flags = np.asarray(z["contact_flags"], dtype=bool) if "contact_flags" in z.files else np.zeros((T, 0), bool)
    meta = json.loads(z["metadata_json"].item()) if "metadata_json" in z.files else {}
    contact_sites = meta.get("contact_pos_sites", {})
    target_weights = meta.get("target_weights", {r: 1.0 for r in role_names})

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    role_to_body = {}
    for ri, role in enumerate(role_names):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(z["alex_body_names"][ri]))
        if bid >= 0:
            role_to_body[role] = bid
    q_lo, q_hi = _get_joint_limits(model)

    print(f"Contact-GlobalOPT  T={T}  λ_track={args.lambda_track} λ_smooth={args.lambda_smooth} "
          f"λ_coll={args.lambda_coll}  n_outer={args.n_outer}")
    print(f"Input: {args.ik_npz}")
    print(f"Effectors: {eff_names}")

    resolved = _resolve_contact_geom(model, eff_names, contact_sites)
    tgt, wgt, planted = _compute_anchors(
        model, data, qpos_ik, eff_names, flags, resolved, fps,
        args.plant_speed, args.foot_weight, args.hand_weight, args.move_ratio,
        args.plant_min_run)
    for eff in resolved:
        n = int((~np.isnan(tgt[eff][:, 0])).sum())
        npl = int(planted[eff].sum())
        print(f"  {eff:11s} contact: {n}/{T} ({n/T*100:.1f}%)  planted: {npl} ({npl/max(n,1)*100:.0f}% of contact)")

    # per-frame set of roles to down-weight (contacting effectors' own roles)
    downweight_roles = [set() for _ in range(T)]
    for eff in resolved:
        col = flags[:, eff_names.index(eff)]
        role = CONTACT_TRACK_ROLE.get(eff)
        for t in np.where(col)[0]:
            downweight_roles[t].add(role)

    def all_stats(q):
        return (_delta_stats(q), _collision_stats(model, data, q),
                _tracking_stats(q, target_positions, role_to_body, role_names, model, data),
                _contact_slip_stats(model, data, q, tgt, wgt, planted, resolved))

    print("\nComputing baseline stats...")
    s_ik = all_stats(qpos_ik)

    print("Stage A: closed-form smoothing...")
    qpos_a = stage_a(qpos_ik, args.lambda_track, args.lambda_smooth, q_lo, q_hi,
                     smooth_root=not args.no_root_smooth,
                     root_lambda_smooth=(args.root_smooth if args.root_smooth > 0 else None))
    s_a = all_stats(qpos_a)

    qpos_b = None
    if args.n_outer > 0:
        print("Stage B: contact-aware QP + SCA...")
        qpos_b = stage_b(qpos_a, target_positions, role_names, role_to_body, target_weights,
                         tgt, wgt, planted, resolved, downweight_roles,
                         model, data, q_lo, q_hi,
                         args.lambda_track, args.lambda_smooth, args.lambda_coll,
                         args.foot_flat_weight, args.fist_weight,
                         args.contact_downweight, args.n_outer, args.trust,
                         collision_penalty=args.collision_penalty)
        s_b = all_stats(qpos_b)

    print("\n" + "=" * 120)
    _stats_row("per-frame IK (warm)", *s_ik)
    _stats_row("Stage A (smoothing)", *s_a)
    if qpos_b is not None:
        _stats_row("Stage B (contact QP)", *s_b)
    print("=" * 120)

    save = {k: z[k] for k in z.files}
    save.update({
        "qpos": qpos_a if qpos_b is None else qpos_b,
        "qpos_per_frame": qpos_ik,
        "qpos_stage_a": qpos_a,
        "fps": np.float64(fps),
    })
    if qpos_b is not None:
        save["qpos_stage_b"] = qpos_b
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out), **save)
    print(f"\nSaved: {args.out}")
    print("Keys: qpos(best), qpos_per_frame, qpos_stage_a"
          + (", qpos_stage_b" if qpos_b is not None else "")
          + " (+ carried contact arrays for the renderer)")


if __name__ == "__main__":
    main()

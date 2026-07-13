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

# Sole corner sites (toe/heel × body-left/right) per foot. Driving all four
# corner Zs to a SHARED floor height in Stage B enforces on-floor + flat +
# inter-foot coplanarity from one row type — a rigid grounding shift (1 DOF)
# can't co-plant two feet that the solve left non-coplanar; this can.
SOLE_CORNER_SITES = {
    "left_foot": [
        "alex_left_sole_corner_toe_body_left_site",
        "alex_left_sole_corner_toe_body_right_site",
        "alex_left_sole_corner_heel_body_left_site",
        "alex_left_sole_corner_heel_body_right_site",
    ],
    "right_foot": [
        "alex_right_sole_corner_toe_body_left_site",
        "alex_right_sole_corner_toe_body_right_site",
        "alex_right_sole_corner_heel_body_left_site",
        "alex_right_sole_corner_heel_body_right_site",
    ],
}


# ---------------------------------------------------------------------------
# Shared helpers (from base GlobalOPT)
# ---------------------------------------------------------------------------

FLOOR_BODY_NAME = "floor_collider"
FLOOR_GEOM_NAME = "floor_collider_geom"


def _load_model_with_floor(model_path):
    """Load the robot MJCF and inject a floor PLANE geom as a mocap body — in
    memory only, never written back to the hand-maintained asset XML.

    Mocap (not a normal welded/static body): a static child of worldbody has its
    world position baked in at compile time (mj_forward does NOT re-derive it from
    a post-compile `model.geom_pos` mutation — verified empirically, geom_xpos
    stays frozen). A mocap body's position IS re-applied every mj_forward via
    `data.mocap_pos`, and — unlike a free joint — adds zero DOFs, so `model.nv`/
    `N_ACT`/`DV_ACT_SLICE` and `_get_joint_limits`'s njnt walk are unaffected.

    No contype/conaffinity wiring needed: the asset XML sets none anywhere, so
    MuJoCo defaults (1/1) apply — the floor geom collides with everything already.

    Returns (model, data, floor_geom_id, floor_mocap_id)."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    floor_body = spec.worldbody.add_body(name=FLOOR_BODY_NAME, mocap=True)
    floor_body.add_geom(name=FLOOR_GEOM_NAME, type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[0, 0, 0.01], pos=[0, 0, 0])
    model = spec.compile()
    data = mujoco.MjData(model)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    floor_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FLOOR_BODY_NAME)
    floor_mocap_id = int(model.body_mocapid[floor_bid])
    return model, data, floor_gid, floor_mocap_id


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


def _actuated_joint_indices(model, names):
    """Actuated-joint indices (0..N_ACT-1, same ordering as qpos[7:]) for the
    given joint names. Mirrors `_get_joint_limits`'s free-joint-skipping walk."""
    by_name = {}
    act_idx = 0
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            continue
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        by_name[jname] = act_idx
        act_idx += 1
        if act_idx == N_ACT:
            break
    return [by_name[n] for n in names if n in by_name]


def _delta_stats(qpos):
    dq = np.abs(np.diff(qpos[:, 7:], axis=0))
    mpf = dq.max(axis=1)
    return {"max": float(mpf.max()), "p95": float(np.percentile(mpf, 95)),
            "mean": float(mpf.mean()), "n_spikes_05": int((mpf > 0.5).sum())}


def _collision_stats(model, data, qpos, floor_gid=None, count_floor=False,
                     floor_active_frames=None):
    """Self-collision penetration, plus robot-vs-floor penetration when
    `count_floor=True` (the injected floor plane — see `_load_model_with_floor`).

    `floor_gid` (when the model has one — always, once `_load_model_with_floor`
    is used) is needed EITHER WAY: the floor body's id is never 0 (it's its own
    mocap child of worldbody, not worldbody itself), so the old `b1==0 or b2==0`
    self-collision exclusion does not catch it — a raw floor contact would
    otherwise silently leak into "self-collision" counting regardless of
    `count_floor`. So floor pairs are always recognized and either counted
    (count_floor=True, no k-hop filter — floor is never anatomically adjacent to
    a robot body) or excluded outright (count_floor=False, same as an ordinary
    self-collision-irrelevant pair).

    `floor_active_frames` (T,) bool, optional: per-frame override on top of
    `count_floor` for phase-aware clips (see `floor_phase_weight`) — a frame
    with `floor_active_frames[t] == False` never counts a floor contact even
    when `count_floor=True` (a lying-phase pelvis reading "penetration" against
    a floor_z calibrated to the standing phase is not a real violation)."""
    pen = []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        active_t = count_floor and (floor_active_frames is None or bool(floor_active_frames[t]))
        mx = 0.0
        for c in range(data.ncon):
            ct = data.contact[c]
            is_floor = floor_gid is not None and (ct.geom1 == floor_gid or ct.geom2 == floor_gid)
            if is_floor:
                if not active_t:
                    continue
            else:
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
        sole_sites = []
        if g["kind"] == "foot":
            for sname in SOLE_CORNER_SITES.get(eff, []):
                ss = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
                if ss >= 0:
                    sole_sites.append(ss)
            if len(sole_sites) != 4:
                print(f"  [warn] {eff}: {len(sole_sites)}/4 sole corner sites found "
                      f"— on-floor rows disabled for it")
                sole_sites = []
        resolved[eff] = dict(
            body_id=bid, site_id=sid, kind=g["kind"],
            axis_local=np.asarray(g["axis_local"]), world_dir=np.asarray(g["world_dir"]),
            sole_sites=sole_sites,
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


def _estimate_floor_z(model, data, qpos_warm, planted, resolved):
    """Shared floor height (world Z) that planted feet should rest on.

    Per foot: median over its planted frames of the min sole-corner Z (its typical
    ground contact in the warm start). Floor = MEDIAN across feet (= midpoint for
    two feet) so BOTH feet share the coplanarity correction — Stage B holds the
    root fixed and leg articulation alone can't pull the higher foot the full gap
    down to the lower foot's ground (reach-limited, saturates ~3 cm). Splitting it
    (each foot ~half the gap) stays within reach; a final constant grounding shift
    then plants the now-coplanar pair. Returns None if no foot has sole sites +
    plants. Single-stance (one planted foot) → that foot's own ground (no lift)."""
    per_foot = []
    for eff, info in resolved.items():
        if info["kind"] != "foot" or not info["sole_sites"]:
            continue
        pf = np.where(planted[eff])[0]
        if pf.size == 0:
            continue
        mins = []
        for t in pf:
            data.qpos[:] = qpos_warm[t]
            mujoco.mj_forward(model, data)
            mins.append(min(float(data.site_xpos[s][2]) for s in info["sole_sites"]))
        per_foot.append(float(np.median(mins)))
    return float(np.median(per_foot)) if per_foot else None


def floor_phase_weight(z_signal, planted_any, lo_pct=5, hi_pct=95):
    """Per-frame [0,1] weight gating floor-collision strength by posture phase.

    Duplicated from solve_fbx_canonical_alex_contactfirst.py (independent CLI
    scripts, no shared imports — same pattern as `_load_model_with_floor`).
    See that copy's docstring for the full rationale: a single clip-wide
    floor_z is calibrated to the standing/planted-foot stance and is not valid
    for a lying/supine/prone phase in the same clip (wiki/concepts/
    grounding.md, "Get-up floor residual is BETWEEN-PHASE"). Here `z_signal`
    is the SOLVED root qpos Z (qpos_ik[:, 2]) rather than Stage 3's pelvis
    target — Stage 4 already has the full trajectory upfront, no causality
    issue. `planted_any`: OR of all foot `planted` masks from
    `_compute_anchors`."""
    z = np.asarray(z_signal, dtype=np.float64)
    pool = z[planted_any] if np.any(planted_any) else z
    hi = float(np.percentile(pool, hi_pct))
    lo = float(np.percentile(z, lo_pct))
    if hi - lo < 1e-6:
        return np.ones_like(z)
    frac = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    return frac * frac * (3.0 - 2.0 * frac)   # smoothstep


def _detect_floor_sensitive_frames(model, data, qpos, floor_gid, floor_z,
                                   min_pen=0.015, min_run=8, pad=5,
                                   floor_active_frames=None, body_filter=None):
    """Boolean (T,) mask: SUSTAINED (>= min_run consecutive frames) floor
    penetration deeper than `min_pen` (default 1.5cm), padded `pad` frames
    either side.

    Used to locally boost Stage A's tracking weight (see `stage_a`'s
    `lambda_track_frames`) so its floor-blind smoothing doesn't erode a sharp
    upstream floor correction (Stage 3's `floor_collision_rows`) by blending it
    back toward uncorrected neighbours — measured regression on
    luigi_standProne_03: a Stage-3-fixed 2.4cm violation came out of plain
    Stage-A smoothing at 13.9cm, WORSE than the original 11.5cm baseline
    (overshoot/ringing past a narrow local fix).

    Two things this is deliberately NOT, both measured:
    1. NOT proximity — a first version flagged "within 3cm of the floor"
       (widening the floor geom's `margin` to make MuJoCo report near-misses).
       A legitimately, correctly planted foot also sits within a few cm of the
       floor for its entire stance (that's what "planted" means), so this
       flagged 100% of frames on any clip with a standing/kneeling phase,
       boosted λ_track everywhere, and defeated Stage A's actual job (spikes
       came back: 0→7).
    2. NOT bare dist<0 either — the infinite floor plane, compared against a
       large multi-body mesh across an entire clip, showed SOME (typically
       sub-cm) negative dist on effectively every frame (802/802 measured;
       median -0.55cm) even where visible worst-case penetration was fine —
       an inherent floor-vs-large-mesh artefact / floor_z estimate imprecision,
       not something worth protecting. `min_pen` filters to genuinely
       consequential violations only.
    3. NOT a HARD on/off mask either — 41.8% of frames turned out to be
       genuinely, sustainedly violating (measured: 5 runs, one 120 frames long
       during the prone phase, one 95 frames long spanning the whole
       swing-to-plant transition — this is wider than the originally-targeted
       ~12-frame peak, min_run didn't shrink it). A hard step in λ_track at
       each protected run's boundary creates a sharp weight discontinuity in
       the banded system, which itself produced a kink (spikes 0->7,
       max_dq up to 1.187 rad). Returns a CONTINUOUS weight in [0,1]
       (1.0 = fully protected, cosine-ramped to 0.0 over `pad` frames at each
       boundary) instead of a boolean mask, so the caller can blend λ_track
       smoothly rather than switch it.

    `floor_active_frames` (T,) bool, optional: phase gate (see
    `floor_phase_weight`) — a frame with `floor_active_frames[t] == False`
    never registers a floor violation here either, so Stage A doesn't get
    told to protect a lying-phase "penetration" that isn't real.

    `body_filter` (set of body ids, optional): restrict which contacting body
    counts toward `viol[t]`. Exists because a single `min_pen` doesn't
    transfer evenly across body geometry: on luigi_standProne_03, the LEFT
    swing foot's mesh contact bottoms out around ~1.1cm deep (never crossing
    the default 1.5cm) even though the sole-corner SITE this codebase's other
    floor checks use reads ~1.5cm+ -- Stage A then smooths this
    unprotected, borderline dig from ~1.5cm to ~13cm. Lowering `min_pen`
    globally to catch it ALSO widens protection onto unrelated hand-floor
    windows elsewhere in the same clip, revealing a Stage-3 wrist jump Stage A
    was separately (and correctly) smoothing away (measured: a NEW 26deg jump
    at a hand-contact frame that a global threshold change touches by
    accident). `body_filter` lets the caller run a SECOND, foot-scoped pass at
    a lower threshold and combine (np.maximum) with the default-threshold,
    unrestricted pass — tightens only where the swing-foot dig actually is."""
    T = qpos.shape[0]
    data.mocap_pos[int(model.body_mocapid[int(model.geom_bodyid[floor_gid])])] = [0.0, 0.0, floor_z]
    viol = np.zeros(T, dtype=bool)
    for t in range(T):
        if floor_active_frames is not None and not floor_active_frames[t]:
            continue
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            ct = data.contact[c]
            if not (ct.geom1 == floor_gid or ct.geom2 == floor_gid) or ct.dist >= -min_pen:
                continue
            if body_filter is not None:
                other = ct.geom2 if ct.geom1 == floor_gid else ct.geom1
                if int(model.geom_bodyid[other]) not in body_filter:
                    continue
            viol[t] = True
            break

    weight = np.zeros(T, dtype=np.float64)
    if not viol.any():
        return weight

    # Runs shorter than min_run don't count as sustained (kept as plain
    # (start, end) pairs, not a boolean mask, so each run's cosine ramp can be
    # built independently below).
    runs = []
    k = 0
    while k < T:
        if not viol[k]:
            k += 1
            continue
        j = k
        while j < T and viol[j]:
            j += 1
        if (j - k) >= min_run:
            runs.append((k, j))
        k = j

    # Raised-cosine ramp 0->1 over `pad` frames (excludes both endpoints, so
    # frame `pad` steps before the run is ~0 and the run's own first frame is
    # 1.0 — same shape family as Stage 3's ramp_envelope).
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, pad + 2))[1:-1]
    for k, j in runs:
        run_weight = np.zeros(T, dtype=np.float64)
        run_weight[k:j] = 1.0
        lo = max(0, k - pad)
        run_weight[lo:k] = ramp[-(k - lo):] if k > lo else ramp[:0]
        hi = min(T, j + pad)
        run_weight[j:hi] = ramp[::-1][:hi - j]
        weight = np.maximum(weight, run_weight)
    return weight


def _foot_floor_err_cm(model, data, qpos, planted, resolved, floor_z):
    """Max |sole-corner Z − floor_z| over planted foot frames (cm). Combined
    on-floor + coplanar residual — the quantity the floor rows drive to zero, and
    the term keep-best must credit or it discards every floor-improving iterate."""
    if floor_z is None:
        return 0.0
    errs = []
    for t in range(qpos.shape[0]):
        touched = False
        for eff, info in resolved.items():
            if info["kind"] != "foot" or not info["sole_sites"] or not planted[eff][t]:
                continue
            if not touched:
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                touched = True
            for sid in info["sole_sites"]:
                errs.append(abs(float(data.site_xpos[sid][2]) - floor_z))
    return float(np.max(errs)) * 100 if errs else 0.0


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
            d = _contact_point(data, info) - a
            # Feet: plant-slip is HORIZONTAL sliding only — vertical foot motion is
            # the deliberate on-floor/coplanar correction (floor rows), not slip, and
            # scoring it as slip would make keep-best reject the correction. Hands: 3D.
            disp = float(np.linalg.norm(d[:2] if info["kind"] == "foot" else d))
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
            smooth_root=True, root_lambda_smooth=None, lambda_track_frames=None):
    """Closed-form per-channel tridiagonal smoothing.

    Actuated joints (qpos[7:]) always smoothed. With smooth_root, the free-base
    root is ALSO smoothed — position (qpos[0:3]) with the same tridiagonal solver
    and the quaternion (qpos[3:7]) via hemisphere-aligned component smoothing +
    renormalise. Without this the root passes through jumpy (per-frame IK root has
    ~3cm / 10deg per-frame pops that read as the whole body flicking).

    `lambda_track_frames`: optional override of the scalar `lambda_track`,
    either (T,) [uniform across joints+root, e.g. locally boosted at
    floor-sensitive frames] or (T, N_ACT) [PER-JOINT — see below]. Purpose:
    this global smoothing pass is otherwise floor-blind and can erode a sharp,
    narrow correction (Stage 3's floor-avoidance term) by blending it back
    toward its uncorrected neighbours — measured on luigi_standProne_03:
    uniform λ_track let Stage A re-inflate a Stage-3-fixed 2.4cm floor
    violation to 13.9cm, WORSE than the original 11.5cm (smoothing
    overshoots/rings past a sharp local fix). The underlying banded solve
    (`_banded_smoother`/`_smooth_channel`) already broadcasts array vs scalar
    `lambda_track` with no other changes needed.

    2D (T, N_ACT) exists because a flat per-frame boost also suppressed
    smoothing on joints that have NOTHING to do with the floor violation —
    measured: WRIST_Z (wrist roll, left deliberately unconstrained by the
    fist-alignment IK term, which only pins the palm-normal axis) started
    flipping once its smoothing was locally disabled too (spikes 0->6, all
    WRIST_Z, cosmetically inert but a real regression against the "0 spikes"
    invariant). Per-joint lets the caller protect only the floor-relevant
    chain (legs, or whichever joints actually move to fix a violation) and
    leave unrelated redundant DOFs fully smoothed. Root position uses the
    per-frame MAX across joint columns (root height correlates with any
    floor-relevant protection, whichever joint it came from)."""
    T = qpos_ik.shape[0]
    if lambda_track_frames is None:
        ltrack = lambda_track
        ltrack_root = lambda_track
    elif np.ndim(lambda_track_frames) == 2:
        ltrack = lambda_track_frames               # (T, N_ACT), indexed per joint below
        ltrack_root = lambda_track_frames.max(axis=1)
    else:
        ltrack = lambda_track_frames                # (T,), uniform
        ltrack_root = lambda_track_frames
    out = qpos_ik.copy()
    for j in range(N_ACT):
        ltrack_j = ltrack[:, j] if np.ndim(ltrack) == 2 else ltrack
        ab = _banded_smoother(T, ltrack_j, lambda_smooth)
        out[:, 7 + j] = np.clip(_smooth_channel(qpos_ik[:, 7 + j], ab, ltrack_j),
                                q_lo[j], q_hi[j])
    if smooth_root:
        rls = lambda_smooth if root_lambda_smooth is None else root_lambda_smooth
        abr = _banded_smoother(T, ltrack_root, rls)
        for j in range(3):                        # root position
            out[:, j] = _smooth_channel(qpos_ik[:, j], abr, ltrack_root)
        Q = qpos_ik[:, 3:7].copy()                # root quaternion (wxyz)
        for t in range(1, T):                     # hemisphere continuity
            if np.dot(Q[t], Q[t - 1]) < 0:
                Q[t] = -Q[t]
        Qs = np.column_stack([_smooth_channel(Q[:, k], abr, ltrack_root) for k in range(4)])
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
                   foot_flat_w, fist_w, floor_z=None, floor_w=0.0):
    """All-soft contact terms into H_blocks/g:
        * position: w_t ||J_pt δq - (target - p)||²   (per-frame weight from wgt)
        * foot-flat: foot up-axis → world +Z, weight foot_flat_w on planted frames
        * fist-down: gripper +X → world −Z, weight fist_w while a hand is in contact
        * on-floor (planted feet, floor_w>0): each sole-corner Z → shared floor_z,
          weight floor_w. Enforces on-floor + flat + inter-foot coplanar at once.
          On those frames the position pin drops to X,Y only so it does not fight
          the floor rows over height.
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

            do_floor = (info["kind"] == "foot" and planted[eff][t]
                        and floor_w > 0.0 and floor_z is not None and info["sole_sites"])
            if do_floor:
                add_soft(Jp[:2], (a - p)[:2], wgt[eff][t], t)  # pin X,Y only (Z→floor)
            else:
                add_soft(Jp, a - p, wgt[eff][t], t)            # full 3D position pin

            if info["kind"] == "foot":
                if planted[eff][t]:
                    add_soft(Jr, err_rot, foot_flat_w, t)     # foot-flat (planted)
                    if do_floor:
                        for sid in info["sole_sites"]:
                            js = np.zeros((3, nv))
                            mujoco.mj_jacSite(model, data, js, None, sid)
                            Jz = js[2:3, DV_ACT_SLICE]        # ∂(corner_z)/∂δq_act
                            e = np.array([floor_z - data.site_xpos[sid][2]])
                            add_soft(Jz, e, floor_w, t)       # sole corner Z → floor
            else:
                add_soft(Jr, err_rot, fist_w, t)              # fist-down
    return H_blocks, g


def _build_collision(qpos_warm, model, data, lambda_coll, floor_gid=None, count_floor=False,
                     floor_active_frames=None):
    """Self-collision rows (existing), plus robot-vs-floor rows when
    `count_floor=True` — same contact-point/Jacobian/margin machinery, just the
    floor plane (a mocap body, zero DOF) as one side of the pair instead of
    another robot link. Floor pairs skip the k-hop adjacency filter (never
    anatomically adjacent) and the floor's own Jacobian naturally comes out zero
    (mocap has no joints), so `jsep` reduces to the robot side alone with no
    special-casing.

    `floor_gid` must be passed whenever the model has an injected floor (i.e.
    always, once `_load_model_with_floor` is used) REGARDLESS of `count_floor`:
    the floor body's id is never 0 (own mocap child of worldbody), so without
    recognizing it explicitly a raw floor contact would silently fall through
    the old `b1==0 or b2==0` exclusion and get added as a bogus self-collision
    row. `count_floor=False` recognizes-and-excludes it instead.

    `floor_active_frames` (T,) bool, optional: phase gate (see
    `floor_phase_weight`) — frame t skips floor rows entirely when
    `floor_active_frames[t] == False`, even with `count_floor=True`."""
    T = qpos_warm.shape[0]
    nv = model.nv
    sqw = float(np.sqrt(lambda_coll))
    r, c, v, l, u = [], [], [], [], []
    row = 0
    for t in range(T):
        data.qpos[:] = qpos_warm[t]
        mujoco.mj_forward(model, data)
        floor_active_t = count_floor and (floor_active_frames is None or bool(floor_active_frames[t]))
        for cc in range(data.ncon):
            ct = data.contact[cc]
            is_floor = floor_gid is not None and (ct.geom1 == floor_gid or ct.geom2 == floor_gid)
            b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
            if is_floor:
                if not floor_active_t:
                    continue
            elif b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS):
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
            collision_penalty=1000.0, floor_z=None, floor_w=0.0,
            floor_gid=None, count_floor=False, floor_active_frames=None):
    """`floor_gid`: the injected floor plane's geom id, ALWAYS passed once
    `_load_model_with_floor` is used (needed to correctly recognize-and-exclude
    floor contacts from self-collision, regardless of `count_floor` — see
    `_build_collision`/`_collision_stats` docstrings). `count_floor`: the actual
    --floor-collision on/off toggle — whether floor contacts become hard
    QP rows + count toward penetration stats. `floor_active_frames` (T,) bool,
    optional: phase gate on top of `count_floor` (see `floor_phase_weight`) —
    threaded into every `_build_collision`/`_collision_stats` call below."""
    T = qpos_warm.shape[0]
    N = T * N_ACT
    q_warm_act = qpos_warm[:, 7:].reshape(-1)
    floor_info = (f" floor_w={floor_w} floor_z={floor_z:.4f}"
                  if floor_w > 0.0 and floor_z is not None else " floor=OFF")
    floor_coll_info = f" floor_collision=ON(gid={floor_gid})" if count_floor else " floor_collision=OFF"
    print(f"  Stage B: T={T} variables={N} n_outer={n_outer} trust={trust} "
          f"soft_collision=ON penalty={collision_penalty}{floor_info}{floor_coll_info}")

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
    # Penetration gate: sub-tol self-penetration is soft-collision slack noise and
    # never traded for contact quality. Pressing floating feet onto the floor costs
    # ~1–1.5 cm of extra self-penetration (legs extend), so with floor rows OR the
    # hard floor-collision constraint ON the gate widens to 2 cm — else keep-best
    # keeps the (feet-4.7cm-apart / floor-clipping) warm start.
    floor_active = (floor_w > 0.0 and floor_z is not None) or count_floor
    PEN_TOL = 2.0 if floor_active else 1.0
    def _iter_score(q):
        cs = _collision_stats(model, data, q, floor_gid=floor_gid, count_floor=count_floor,
                              floor_active_frames=floor_active_frames)
        ss = _contact_slip_stats(model, data, q, tgt, wgt, planted, resolved)
        pen = cs["max_pen_cm"]; slip = ss["plant_slip_max_cm"]
        ferr = _foot_floor_err_cm(model, data, q, planted, resolved,
                                  floor_z if floor_w > 0.0 else None)
        hard = max(0.0, pen - PEN_TOL)
        # Primary (below the hard gate): total contact error = horizontal plant slip
        # + vertical foot-off-floor. Both are "how badly the planted contacts miss".
        return (hard, slip + ferr, pen + slip, cs["pct"], pen, slip, ferr)
    best_qpos = qpos_cur.copy()
    best_score = _iter_score(best_qpos)
    print(f"    warm: pen={best_score[4]:.2f}cm slip={best_score[5]:.1f}cm "
          f"floor_err={best_score[6]:.2f}cm coll={best_score[3]:.1f}%")

    for outer in range(n_outer):
        t0 = time.time()
        Ht, gt = _build_tracking(qpos_cur, target_positions, role_names, role_to_body,
                                 target_weights, model, data, lambda_track,
                                 downweight_roles, downweight_factor)
        Hc, gc = _build_contact(qpos_cur, tgt, wgt, planted, resolved,
                                model, data, foot_flat_w, fist_w, floor_z, floor_w)
        H_task = _blocks_to_sparse([Ht[t] + Hc[t] for t in range(T)], N)
        P = 2.0 * (H_task + H_smooth)
        q_vec = gt + gc

        # Trust region: keep this iterate's δQ within `trust` of the previous one
        # (SCA stabiliser — stops the collision re-linearisation from oscillating).
        jl_lo = np.maximum(jl_lo_abs, delta - trust)
        jl_hi = np.minimum(jl_hi_abs, delta + trust)

        A_coll, l_coll, u_coll = _build_collision(qpos_cur, model, data, lambda_coll,
                                                  floor_gid=floor_gid, count_floor=count_floor,
                                                  floor_active_frames=floor_active_frames)
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
            pen_info = (f" pen={score[4]:.2f}cm slip={score[5]:.1f}cm "
                        f"floor_err={score[6]:.2f}cm coll={score[3]:.1f}%"
                        + (" *best" if keep else ""))
            if keep:
                best_score = score
                best_qpos = qpos_cur.copy()
        print(f"    outer {outer+1}/{n_outer}: coll_rows={n_coll_rows} status={res.info.status} "
              f"|dQ|max={np.abs(delta).max():.4f} time={time.time()-t0:.1f}s" + slack_info + pen_info)
    print(f"    Stage B best: pen={best_score[4]:.2f}cm slip={best_score[5]:.1f}cm "
          f"floor_err={best_score[6]:.2f}cm coll={best_score[3]:.1f}%")
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
    ap.add_argument("--floor-weight", type=float, default=0.0,
                    help="Soft weight driving each planted foot's 4 sole-corner Zs to a "
                         "SHARED floor height (on-floor + flat + inter-foot coplanar). "
                         "0 = off (legacy). On planted frames the position pin drops to "
                         "X,Y only so it doesn't fight the floor rows. Fixes the "
                         "non-coplanar-feet gap a rigid grounding shift can't.")
    ap.add_argument("--floor-mode", choices=["estimate", "zero"], default="estimate",
                    help="Floor height for the on-floor rows. estimate = lower planted "
                         "foot's warm-start ground height (still needs a final constant "
                         "grounding shift for absolute z=0); zero = drive soles to world "
                         "z=0 directly (grounding becomes a near no-op, but legs must "
                         "reach 0 from Stage-A root height).")
    ap.add_argument("--floor-collision", choices=["on", "off"], default="on",
                    help="Hard mesh-accurate robot-vs-floor collision (a plane geom "
                         "injected in-memory, reusing the self-collision soft-slack QP "
                         "machinery). Unlike --floor-weight (a soft pin, planted feet "
                         "only), this stops ANY fullmesh geometry — swing feet, hands, "
                         "a tilted toe mid-get-up — from passing through the floor. "
                         "on = default; off to bisect/regression-check against the old "
                         "behaviour.")
    ap.add_argument("--sens-foot-min-pen", type=float, default=0.015,
                    help="Foot-scoped SECOND floor-sensitivity pass (see _detect_floor_"
                         "sensitive_frames' body_filter docstring): a lower --sens-min-pen-style "
                         "threshold applied ONLY to leg/foot body contacts, combined (max) with "
                         "the default-threshold pass. Catches a swing foot's shallower mesh-"
                         "contact depth without also widening protection onto hand-floor windows "
                         "(scoping to feet only avoided a measured new wrist jump a global "
                         "threshold change caused). Default: 0.015 (equals --sens-min-pen, i.e. "
                         "no-op/inert unless explicitly lowered).")
    ap.add_argument("--sens-min-pen", type=float, default=0.015,
                    help="_detect_floor_sensitive_frames' penetration threshold (m) for "
                         "protecting Stage A from smoothing away a Stage-3 floor correction. "
                         "Measured against MESH CONTACT depth (data.contact.dist), which reads "
                         "SHALLOWER than the sole-corner-SITE depth this codebase's other floor "
                         "checks use (e.g. eval_artifacts_corpus.py's anyPen) -- on "
                         "luigi_standProne_03's swing-foot dig, site-depth was 1.5cm but the "
                         "mesh contact never exceeded 1.1cm, so the default 1.5cm threshold "
                         "misses this window entirely (protection weight stays exactly 0 across "
                         "it) and Stage A's floor-blind smoothing drags it from a borderline "
                         "~1.5cm miss to ~13cm. Lower cautiously -- the docstring's own warning: "
                         "too low re-flags legitimately-planted stances (near-floor for their "
                         "whole stance by definition) and defeats Stage A's actual smoothing job "
                         "(measured regression: spikes 0->7). Default: 0.015 = 1.5cm.")
    ap.add_argument("--floor-phase-aware", choices=["on", "off"], default="off",
                    help="Gate --floor-collision off during a clip's low/lying-phase frames "
                         "(root-qpos-Z smoothstep between the clip's low reference and its "
                         "planted-foot/standing height — see floor_phase_weight). Off by "
                         "default: identity, byte-for-byte unchanged from before this flag "
                         "existed. Needed for clips with a genuine standing+lying phase split "
                         "(e.g. get-ups) — forcing hard floor collision through the lying "
                         "phase misreads the legitimately-low pelvis/hip as a violation "
                         "(measured on luigi_standSupine_08: RIGHT_HIP_X_LINK 14.4cm, the SCA "
                         "loop could never resolve it since it wasn't real). No-op for "
                         "single-phase clips (root Z barely varies -> all-1s weight).")
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

    model, data, floor_gid, floor_mocap_id = _load_model_with_floor(args.model)

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

    # floor_gid is passed to _collision_stats in EVERY call below, count_floor=False
    # by default — this is required (not optional) once the model has an injected
    # floor: without recognizing the floor geom explicitly, its contacts would
    # silently leak into "self-collision" counting (its body id isn't 0, so the
    # old bodyid==0 exclusion doesn't catch it). count_floor=True only once the
    # floor plane has actually been positioned + enabled for Stage B, below.
    def all_stats(q, count_floor=False, floor_active_frames=None):
        return (_delta_stats(q), _collision_stats(model, data, q, floor_gid=floor_gid,
                                                   count_floor=count_floor,
                                                   floor_active_frames=floor_active_frames),
                _tracking_stats(q, target_positions, role_to_body, role_names, model, data),
                _contact_slip_stats(model, data, q, tgt, wgt, planted, resolved))

    print("\nComputing baseline stats...")
    s_ik = all_stats(qpos_ik)

    # floor_z: needed for the soft on-floor pin (--floor-weight), the hard
    # floor-collision constraint (--floor-collision), AND (new) protecting
    # floor-sensitive frames during Stage A smoothing below — so compute it
    # up front from the WARM (Stage-3) input, before Stage A runs, rather than
    # gating behind floor_weight alone or waiting for Stage A's output.
    # _estimate_floor_z only needs planted-foot geometry, which barely moves
    # under smoothing, so qpos_ik vs qpos_a makes no meaningful difference here.
    want_floor_z = args.floor_weight > 0.0 or args.floor_collision == "on"
    floor_z = None
    if want_floor_z:
        floor_z = 0.0 if args.floor_mode == "zero" else \
            _estimate_floor_z(model, data, qpos_ik, planted, resolved)
        print(f"  floor: weight={args.floor_weight} mode={args.floor_mode} "
              f"collision={args.floor_collision} floor_z={floor_z}")

    # Phase-aware floor gating (see floor_phase_weight docstring): a single
    # clip-wide floor_z is calibrated to the standing/planted-foot stance and
    # misreads a legitimately-low lying/supine phase as penetration (measured
    # on luigi_standSupine_08: RIGHT_HIP_X_LINK 14.4cm "penetration" the QP
    # could never resolve, since it wasn't real). Opt-in, off by default —
    # identity (all-1s active mask) when off, byte-for-byte unchanged from
    # before this flag existed.
    floor_active_frames = None
    if args.floor_collision == "on" and args.floor_phase_aware == "on":
        planted_any = np.zeros(T, dtype=bool)
        for eff, info in resolved.items():
            if info["kind"] == "foot":
                planted_any |= planted[eff]
        floor_phase_w = floor_phase_weight(qpos_ik[:, 2], planted_any)
        floor_active_frames = floor_phase_w >= 0.5
        print(f"  floor-phase-aware: root_z-based weight min={floor_phase_w.min():.2f} "
              f"max={floor_phase_w.max():.2f} active={floor_active_frames.mean() * 100:.1f}% of frames")

    # Locally boost Stage A's tracking weight at floor-sensitive frames (see
    # _detect_floor_sensitive_frames docstring) so this floor-blind smoothing
    # pass doesn't erode an upstream floor correction. Boost factor: strong
    # enough that λ_track dominates λ_smooth locally (λ_smooth is typically
    # ~100-300x the default λ_track=1 at the native 120Hz rate scaling).
    #
    # Gated specifically on --floor-collision (not the broader want_floor_z,
    # which also covers the PRE-EXISTING --floor-weight on-floor pin, default
    # ON at 200 in the pipeline) — this protection is new/unvalidated and must
    # not change behaviour for the already-shipped --floor-weight-only path.
    # Joints deliberately left unconstrained by Stage 3's contact IK (fist
    # alignment only pins the palm-normal axis, not roll about it) — excluded
    # from the boost below (kept at plain λ_track, fully smoothed) since
    # they're irrelevant to floor clearance and boosting them just let their
    # own free-floating angle stop being regularized: measured spikes 0->6,
    # all WRIST_Z, when a flat (non-per-joint) boost was used instead.
    UNCONSTRAINED_ROLL_JOINTS = ["LEFT_WRIST_Z", "RIGHT_WRIST_Z",
                                 "LEFT_GRIPPER_Z", "RIGHT_GRIPPER_Z"]

    lambda_track_frames = None
    if floor_z is not None and args.floor_collision == "on":
        sens_w = _detect_floor_sensitive_frames(model, data, qpos_ik, floor_gid, floor_z,
                                                min_pen=args.sens_min_pen,
                                                floor_active_frames=floor_active_frames)
        # Foot-scoped SECOND pass at a lower threshold (see _detect_floor_
        # sensitive_frames' body_filter docstring): a swing foot's mesh
        # contact reads shallower than the sole-corner-site depth this
        # codebase's other floor checks use, so the default threshold above
        # misses it. Restricted to leg/foot bodies ONLY so it can't also
        # widen protection onto unrelated hand-floor windows (that caused a
        # measured NEW wrist jump when tried as a single global threshold
        # change). No-op (sens_w unchanged) when args.sens_foot_min_pen
        # equals args.sens_min_pen (its default).
        sens_w_foot = None
        if args.sens_foot_min_pen < args.sens_min_pen:
            LEG_FOOT_BODIES = ["LEFT_HIP_X_LINK", "LEFT_HIP_Z_LINK", "LEFT_THIGH", "LEFT_SHIN",
                               "LEFT_ANKLE_Y_LINK", "LEFT_FOOT",
                               "RIGHT_HIP_X_LINK", "RIGHT_HIP_Z_LINK", "RIGHT_THIGH", "RIGHT_SHIN",
                               "RIGHT_ANKLE_Y_LINK", "RIGHT_FOOT"]
            foot_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LEG_FOOT_BODIES}
            sens_w_foot = _detect_floor_sensitive_frames(
                model, data, qpos_ik, floor_gid, floor_z,
                min_pen=args.sens_foot_min_pen, body_filter=foot_bids,
                floor_active_frames=floor_active_frames)
        if sens_w.any() or (sens_w_foot is not None and sens_w_foot.any()):
            boost = max(args.lambda_track, args.lambda_smooth * 2.0)
            # Continuous blend, not a step: sens_w in [0,1] (1.0 = fully
            # protected core, cosine-ramped to 0 over `pad` frames at each
            # violation run's boundary) — see docstring for why a hard on/off
            # switch reintroduced spikes.
            per_frame = args.lambda_track + sens_w * (boost - args.lambda_track)
            lambda_track_frames = np.tile(per_frame[:, None], (1, N_ACT))
            excl = _actuated_joint_indices(model, UNCONSTRAINED_ROLL_JOINTS)
            lambda_track_frames[:, excl] = args.lambda_track
            # Foot-scoped boost (see body_filter docstring): applied ONLY to
            # leg/ankle JOINT COLUMNS, never wrist/shoulder/spine, regardless
            # of which frames sens_w_foot flags — this is what actually
            # prevents the cross-contamination a frame-uniform boost caused
            # (measured: a foot-scoped DETECTION alone still produced a new
            # wrist jump, because the old code applied its result to every
            # joint at the flagged frame, not just the leg).
            if sens_w_foot is not None and sens_w_foot.any():
                LEG_JOINTS = ["LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y",
                              "LEFT_ANKLE_Y", "LEFT_ANKLE_X",
                              "RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y",
                              "RIGHT_ANKLE_Y", "RIGHT_ANKLE_X"]
                leg_idx = _actuated_joint_indices(model, LEG_JOINTS)
                per_frame_foot = args.lambda_track + sens_w_foot * (boost - args.lambda_track)
                for j in leg_idx:
                    lambda_track_frames[:, j] = np.maximum(lambda_track_frames[:, j], per_frame_foot)
            sens_w_report = np.maximum(sens_w, sens_w_foot) if sens_w_foot is not None else sens_w
            print(f"  floor-sensitive frames (Stage-A protection): "
                  f"{int((sens_w_report > 0.99).sum())}/{qpos_ik.shape[0]} fully protected "
                  f"({int((sens_w_report > 0).sum())} incl. ramp), boosted λ_track={boost:.0f}")

    print("Stage A: closed-form smoothing...")
    qpos_a = stage_a(qpos_ik, args.lambda_track, args.lambda_smooth, q_lo, q_hi,
                     smooth_root=not args.no_root_smooth,
                     root_lambda_smooth=(args.root_smooth if args.root_smooth > 0 else None),
                     lambda_track_frames=lambda_track_frames)
    s_a = all_stats(qpos_a)

    qpos_b = None
    if args.n_outer > 0:
        count_floor = args.floor_collision == "on" and floor_z is not None
        if count_floor:
            # Position the injected floor plane at this clip's estimated floor
            # height, in the SAME ungrounded frame the whole solve operates in —
            # valid because Stage 4.5's rigid shift later zeroes this exact
            # reference (see plan doc / wiki concepts/globalopt.md).
            data.mocap_pos[floor_mocap_id] = [0.0, 0.0, floor_z]
        print("Stage B: contact-aware QP + SCA...")
        qpos_b = stage_b(qpos_a, target_positions, role_names, role_to_body, target_weights,
                         tgt, wgt, planted, resolved, downweight_roles,
                         model, data, q_lo, q_hi,
                         args.lambda_track, args.lambda_smooth, args.lambda_coll,
                         args.foot_flat_weight, args.fist_weight,
                         args.contact_downweight, args.n_outer, args.trust,
                         collision_penalty=args.collision_penalty,
                         floor_z=floor_z, floor_w=args.floor_weight,
                         floor_gid=floor_gid, count_floor=count_floor,
                         floor_active_frames=floor_active_frames)
        s_b = all_stats(qpos_b, count_floor=count_floor, floor_active_frames=floor_active_frames)

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

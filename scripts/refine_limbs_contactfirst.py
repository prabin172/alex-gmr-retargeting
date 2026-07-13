#!/usr/bin/env python3
"""Per-limb cleanup solver (phasic-v2 M5/P4, plan.md's core new deliverable).

Input: the grounded NPZ (or physics-plausibility NPZ if that stage ran),
carrying `qpos (T,36)`, `fps`, `contact_flags (T,4)`, `contact_effector_names`,
`role_names`/`alex_body_names`/`target_positions` (for the tracking-delta
gate). Output: same schema, `qpos` replaced by the cleaned-up trajectory.

ROOT IS FROZEN (qpos[:, 0:7] copied unchanged to every output frame, every
round). Every remaining DOF is a plain hinge joint (qpos_adr = dof_adr+1,
always, since the only non-hinge joint is the single leading free root) — no
quaternion retraction needed anywhere in this script, unlike
scripts/physics_plausibility_pass.py's tangent-space machinery. This is the
whole point of freezing the root: keeps the per-limb QPs pure Euclidean
least-squares over hinge angles.

Four independent per-limb whole-clip banded QPs (LEFT_LEG, RIGHT_LEG,
LEFT_ARM, RIGHT_ARM — see LIMB_CHAINS), solved Gauss-Seidel style (T5.4):
legs first, then arms, 2 rounds, each limb's solve re-linearized against the
CURRENT state (including corrections from limbs already processed this
round). Decision variable per limb: delta_q_limb in R^{T*k} (k=6 legs, k=7
arms), additive offset to that limb's OWN joint angles only.

Objective per limb-QP: posture regularization toward the limb's OWN prior
value at each frame (never freeze — "regularize to own prior value," the
`refine_arm_floor_transitions` lesson: freezing just moves the discontinuity
to the window boundary, see scripts/solve_fbx_canonical_alex_contactfirst.py)
+ banded smoothness (same block-tridiagonal pattern as Stage 4's
`_build_smoothness_hessian` / this repo's physics_plausibility_pass.py, sized
to the limb's own k DOFs) + a Cartesian effector-tracking ridge (keep the
limb's terminal effector close to where it already was, redundant-chain
insurance beyond the joint-space ridge alone).

Inequality rows (T5.2, T5.3), all restricted to nonzero-Jacobian columns for
THIS limb (a contact whose Jacobian w.r.t. this limb's DOFs is all-zero
contributes no row — automatically excludes frozen-body-only contacts, no
explicit body-membership filtering needed):
  - Floor non-penetration for every geom on this limb (reuses
    `_load_model_with_floor`'s in-memory floor-plane injection, same pattern
    as Stage 3/4 — never touches the hand-maintained asset XML).
  - Self-collision vs the rest of the body (frozen or otherwise), same
    k-hop-adjacency skip as Stage 3/4 (`_within_k_hops`, k=2).
  - Swing clearance (T5.3, new): when this limb's OWN contact envelope alpha
    (recomputed from the persisted bool contact_flags via
    scripts/contact_labels.py's `ramp_envelope` — the SAME continuous
    cross-fade Stage 3 uses internally, just reconstructed here since only
    the debounced bool survives to the NPZ) is near 0 AND its support point
    is within an activation band above the floor, a one-sided row keeps the
    support point at or above `--swing-clearance` (default 2cm). Weighted by
    (1-alpha) — fades OUT as contact fades IN, so it never fights touchdown
    (the missing-ramp bug class that caused the original wrist-flick
    disaster this whole redesign traces back to; mandatory here, not
    optional, per plan.md T5.3).

Keep-best-iterate across the 2 Gauss-Seidel rounds (T5.4), lexicographic
score adapted from Stage 4's `stage_b`: (hard-fail gate on floor penetration
beyond a tolerance, then tracking delta, then self-collision depth) — never
ships worse than the input.

OPTIONAL ROOT-Z DOF (hierarchical-v1 plan.md H1, `--root-z`, default off):
the one fallback plan.md's own M5 Risks section anticipated ("root frozen =>
reach saturation... fallback = allow a root-z DOF in P4 round 2"), targeting
the 7/20 whole-body-lying clips M5's pure per-limb solver cannot reach
(planLog.md M5: TORSO/PELVIS/thighs/feet all in floor contact simultaneously
-- CORE-classified penetration, architecturally outside any limb chain).
When enabled, ONLY qpos index 2 (root world Z, a plain Euclidean translation
-- root x/y/orientation stay hard-frozen, never touched) becomes an
additional 1-DOF "pseudo-limb" solved after the 4 real limbs each round,
active starting `--root-z-start-round` (default round index 1, i.e. the
SECOND round) so the per-limb solves get first crack alone. Because root-Z
translation shifts every body in the tree identically, its Jacobian column
naturally (a) produces a real floor-collision row for CORE bodies (fixing
what limb chains structurally cannot) and (b) cancels to ~0 in every
self-collision row (both sides of a self-contact move together), so it
never fights self-penetration -- no special-casing needed, reuses
`_solve_limb_qp` verbatim with a synthetic 1-DOF "resolved" struct
(`support_sites=[]`, swing-clearance skipped by construction). See
planLog.md H1 for verification numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import scipy.sparse as sp
import osqp

from contact_labels import CONTACT_EFFECTORS, debounce_flags, ramp_envelope

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"

FLOOR_BODY_NAME = "floor_collider"
FLOOR_GEOM_NAME = "floor_collider_geom"
COLL_HOPS = 2          # matches Stage 3/4's self-collision adjacency skip
COLL_MARGIN = 0.02     # matches Stage 3/4's self_collision_rows/_build_collision margin
COLL_PENALTY = 1000.0  # soft-slack quadratic penalty, matches Stage 4's collision_penalty

LIMB_CHAINS = {
    "left_leg": ["LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y", "LEFT_ANKLE_Y", "LEFT_ANKLE_X"],
    "right_leg": ["RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y", "RIGHT_ANKLE_Y", "RIGHT_ANKLE_X"],
    "left_arm": ["LEFT_SHOULDER_Y", "LEFT_SHOULDER_X", "LEFT_SHOULDER_Z", "LEFT_ELBOW_Y",
                 "LEFT_WRIST_Z", "LEFT_WRIST_X", "LEFT_GRIPPER_Z"],
    "right_arm": ["RIGHT_SHOULDER_Y", "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Z", "RIGHT_ELBOW_Y",
                  "RIGHT_WRIST_Z", "RIGHT_WRIST_X", "RIGHT_GRIPPER_Z"],
}
LIMB_ORDER = ["left_leg", "right_leg", "left_arm", "right_arm"]   # legs first, T5.4
LIMB_EFFECTOR_BODY = {
    "left_leg": "LEFT_FOOT", "right_leg": "RIGHT_FOOT",
    "left_arm": "LEFT_GRIPPER_Z_LINK", "right_arm": "RIGHT_GRIPPER_Z_LINK",
}
LIMB_CONTACT_EFF = {"left_leg": "left_foot", "right_leg": "right_foot",
                     "left_arm": "left_hand", "right_arm": "right_hand"}
# Support-point sites for the swing-clearance check (T5.3) -- mesh-accurate
# sole corners for feet (same set used throughout this codebase: Stage 3/4/
# 4.5/physics-plausibility), single palm-contact site for hands (same as
# Stage 3's CONTACT_POS).
LIMB_SUPPORT_SITES = {
    "left_leg": ["alex_left_sole_corner_toe_body_left_site", "alex_left_sole_corner_toe_body_right_site",
                 "alex_left_sole_corner_heel_body_left_site", "alex_left_sole_corner_heel_body_right_site"],
    "right_leg": ["alex_right_sole_corner_toe_body_left_site", "alex_right_sole_corner_toe_body_right_site",
                  "alex_right_sole_corner_heel_body_left_site", "alex_right_sole_corner_heel_body_right_site"],
    "left_arm": ["alex_left_palm_contact_site"],
    "right_arm": ["alex_right_palm_contact_site"],
}

# Rate-scaled to native 120 Hz, matching pipeline.md's x16 rule (same
# LAMBDA_SMOOTH the corpus batch uses for Stage 4).
LAMBDA_SMOOTH_DEFAULT = 320.0
POSTURE_REG_DEFAULT = 0.02       # matches refine_arm_floor_transitions's arm_posture_reg
TRACK_DEFAULT = 1.0
TRUST_REGION = 0.10              # rad, plan.md T5.1
PEN_TOL_CM = 0.5                 # plan.md's per-clip gate: floor pen <= 0.5cm


def _load_model_with_floor(model_path):
    """Inject a floor PLANE geom as a mocap body -- in memory only, identical
    pattern to Stage 3/4 (never touches the hand-maintained asset XML)."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    floor_body = spec.worldbody.add_body(name=FLOOR_BODY_NAME, mocap=True)
    floor_body.add_geom(name=FLOOR_GEOM_NAME, type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[0, 0, 0.01], pos=[0, 0, 0])
    model = spec.compile()
    data = mujoco.MjData(model)
    floor_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FLOOR_BODY_NAME)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    floor_mocap_id = int(model.body_mocapid[floor_bid])
    data.mocap_pos[floor_mocap_id] = [0.0, 0.0, 0.0]
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


def _joint_adr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Missing joint in Alex model: {name}")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _resolve_limbs(model):
    """Per limb: qpos addresses (k,), dof addresses (k,), effector body id,
    support-point site ids."""
    out = {}
    for limb, joints in LIMB_CHAINS.items():
        qadrs, dofadrs = [], []
        for j in joints:
            qa, da = _joint_adr(model, j)
            qadrs.append(qa); dofadrs.append(da)
        eff_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LIMB_EFFECTOR_BODY[limb])
        site_names = LIMB_SUPPORT_SITES[limb]
        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in site_names]
        site_ids = [s for s in site_ids if s >= 0]
        out[limb] = dict(qpos_adr=np.array(qadrs), dof_adr=np.array(dofadrs),
                         eff_bid=eff_bid, support_sites=site_ids, k=len(joints))
    return out


def _limb_body_ids(model, resolved_all):
    """Set of body ids belonging to ANY limb chain -- walk model.body_parentid
    from each limb's effector body up to (excluding) the first non-limb
    ancestor (PELVIS_LINK/TORSO_LINK/etc). Used to classify a floor contact
    as LIMB-caused (this pass can fix it, subject to the PEN_TOL gate) vs
    CORE-caused (torso/pelvis/head/spine -- architecturally out of scope
    since root+spine are frozen by design, T5.1's "Root FROZEN"; report only,
    don't gate on it)."""
    core_names = {"PELVIS_LINK", "TORSO_LINK", "NECK_Z_LINK", "NECK_Y_LINK", "HEAD_LINK",
                  "SPINE_Z_LINK"}
    core_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in core_names}
    core_ids = {b for b in core_ids if b >= 0}
    limb_ids = set()
    for resolved in resolved_all.values():
        cur = resolved["eff_bid"]
        while cur not in core_ids and cur != 0:
            limb_ids.add(cur)
            cur = int(model.body_parentid[cur])
    return limb_ids


def _contact_alpha(contact_flags, eff_names, eff, fps, contact_min_run=12, contact_ramp=16, contact_preroll=8):
    """Reconstruct Stage 3's continuous contact cross-fade envelope from the
    persisted (debounced-bool) contact_flags -- the continuous version is
    Stage-3-internal only, not saved to the NPZ. Same debounce/ramp/preroll
    defaults the pipeline uses at 120 Hz (pipeline.md's x4 frame-count rule)."""
    if eff not in eff_names:
        T = contact_flags.shape[0]
        return np.zeros(T)
    col = eff_names.index(eff)
    flag = contact_flags[:, col].astype(bool)
    solved = debounce_flags(flag, contact_min_run)
    return ramp_envelope(solved, contact_ramp, contact_preroll)


def _lowest_support_site(data, site_ids):
    """(site_id, z) of the currently-lowest support-point site -- local
    linearization point for the swing-clearance row (min() is nonsmooth;
    using the CURRENT worst site each re-linearization matches how the rest
    of this codebase's contact-margin code already handles similar cases)."""
    zs = [(s, float(data.site_xpos[s][2])) for s in site_ids]
    return min(zs, key=lambda x: x[1])


def _solve_limb_qp(model, data, qpos, limb, resolved, floor_gid, contact_alpha_env,
                    dt, lambda_smooth, posture_reg, track_w, swing_clearance, swing_band,
                    plant_hold_boost=50.0, qpos_ref=None, eff_ref_pos=None,
                    trust_region=TRUST_REGION):
    """Whole-clip banded QP for ONE limb. Returns delta_q (T, k) additive
    offset to this limb's OWN joint angles (qpos indices), plus diagnostic
    counts (floor rows, self-collision rows, swing-clearance rows).

    `qpos` is the CURRENT round's state -- used for LINEARIZATION (Jacobians,
    contact detection; must be the current best-known state for these to be
    accurate). `qpos_ref`/`eff_ref_pos` (default to `qpos`'s own values if
    None, for backward-compat / a first-round call) are the REGULARIZATION
    TARGET the posture and Cartesian-tracking ridges pull toward -- these
    MUST be the ORIGINAL INPUT (round 0), not `qpos` itself, or every round's
    small mistakes compound into the next round's anchor instead of being
    pulled back toward the known-good original. This was a REAL bug: without
    a fixed ref, the Gauss-Seidel loop DIVERGED on several corpus clips
    (measured on standup_02: floor pen grew 11.26->14.68->...->26.72cm over
    10 rounds, monotonically, never recovering) even with swing-clearance
    disabled entirely (ruled out as the cause by direct A/B test) -- the
    ridge terms had nothing stable to anchor to. See planLog.md M5.

    `plant_hold_boost`: the Cartesian tracking ridge's XY weight is boosted
    on frames where this limb's OWN contact alpha > 0.5 (planted) -- without
    this, a correction needed on a nearby SWING frame can bleed into an
    adjacent PLANTED frame through the smoothness coupling (measured on
    standup_01: up to 19.8cm planted-frame slip with a uniform tracking
    weight). Z is NEVER boosted (a floor-fix on the same frame must not be
    fought) -- same pattern Stage 4's on-floor rows already use ("the
    position pin drops to X,Y only," wiki/concepts/globalopt.md)."""
    T = qpos.shape[0]
    k = resolved["k"]
    qadr = resolved["qpos_adr"]
    dofadr = resolved["dof_adr"]
    eff_bid = resolved["eff_bid"]
    N = T * k
    if qpos_ref is None:
        qpos_ref = qpos
    q_target_delta = qpos_ref[:, qadr] - qpos[:, qadr]   # (T, k): where delta SHOULD end up, per frame

    # --- Objective: posture ridge (toward qpos_ref, NOT qpos) + smoothness
    # + Cartesian tracking ridge (toward eff_ref_pos, NOT qpos's own effector
    # position) ---
    H_blocks = [np.full(k, 2.0 * posture_reg) for _ in range(T)]  # diag posture ridge
    g = np.zeros(N)
    for t in range(T):
        s0 = t * k
        g[s0:s0 + k] += -2.0 * posture_reg * q_target_delta[t]

    r_sm, c_sm, v_sm = [], [], []
    for t in range(T):
        s0 = t * k
        for j in range(k):
            r_sm.append(s0 + j); c_sm.append(s0 + j); v_sm.append(H_blocks[t][j])
        scale = lambda_smooth * (2.0 if 0 < t < T - 1 else 1.0)
        for j in range(k):
            r_sm.append(s0 + j); c_sm.append(s0 + j); v_sm.append(2.0 * scale)
        if t > 0:
            prev = (t - 1) * k
            for j in range(k):
                r_sm.append(s0 + j); c_sm.append(prev + j); v_sm.append(-2.0 * lambda_smooth)
                r_sm.append(prev + j); c_sm.append(s0 + j); v_sm.append(-2.0 * lambda_smooth)
    P_base = sp.csc_matrix((v_sm, (r_sm, c_sm)), shape=(N, N))

    # Cartesian effector-tracking ridge (soft, added directly into P/g -- pulls
    # the effector toward its OWN current position, redundant-chain insurance
    # beyond the joint-space posture ridge). Boosted on PLANTED frames (see
    # plant_hold_boost docstring above) so a swing-phase correction can't
    # bleed into an adjacent plant via the smoothness coupling.
    # XY (horizontal) and Z (vertical) weighted SEPARATELY on planted frames
    # -- same pattern Stage 4's on-floor rows already use ("the position pin
    # drops to X,Y only on these frames so it doesn't fight the height row",
    # wiki/concepts/globalopt.md). A single boosted 3D ridge was tried first
    # and BLOCKED a genuinely-needed Z correction on a fully-planted frame
    # (measured on standup_01: the worst residual floor penetration, 5.8cm,
    # sat on a right_leg frame with alpha=1.0 -- a uniformly-boosted ridge or
    # a hard per-frame trust-region freeze both fight the floor-fix instead
    # of just preventing horizontal slip; see planLog.md M5). Splitting XY
    # (boosted heavily on planted frames -- this is the actual "don't slip"
    # requirement) from Z (left at the normal weight always, so floor
    # correction is never fought) resolves both simultaneously.
    if eff_ref_pos is None:
        eff_ref_pos = [None] * T   # filled with the current position per-frame below
    r_tr, c_tr, v_tr = [], [], []
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        eff_cur = data.xpos[eff_bid].copy()
        e = (eff_ref_pos[t] - eff_cur) if eff_ref_pos[t] is not None else np.zeros(3)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, eff_bid)
        J = jacp[:, dofadr]   # (3, k)
        alpha_t = float(contact_alpha_env[t])
        w_xy = track_w * (1.0 + (plant_hold_boost - 1.0) * alpha_t)
        w_z = track_w
        Hb = w_xy * (J[0:2].T @ J[0:2]) + w_z * np.outer(J[2], J[2])
        gb = w_xy * (J[0:2].T @ e[0:2]) + w_z * J[2] * e[2]
        s0 = t * k
        for a in range(k):
            for b in range(k):
                if abs(Hb[a, b]) > 1e-14:
                    r_tr.append(s0 + a); c_tr.append(s0 + b); v_tr.append(2.0 * Hb[a, b])
        g[s0:s0 + k] += -2.0 * gb
    P_track = sp.csc_matrix((v_tr, (r_tr, c_tr)), shape=(N, N)) if r_tr else sp.csc_matrix((N, N))
    P = P_base + P_track

    # --- Inequality rows: floor + self-collision + swing clearance ---
    r, c, v, l, u = [], [], [], [], []
    row = 0
    n_floor = n_self = n_swing = 0

    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        s0 = t * k

        for cc in range(data.ncon):
            ct = data.contact[cc]
            is_floor = ct.geom1 == floor_gid or ct.geom2 == floor_gid
            b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
            if not is_floor and (b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS)):
                continue
            pen = COLL_MARGIN - float(ct.dist)
            if pen <= 0:
                continue
            normal = ct.frame[:3].copy()
            if float(np.dot(normal, data.xpos[b1] - data.xpos[b2])) < 0:
                normal = -normal
            j1 = np.zeros((3, model.nv)); j2 = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, j1, None, ct.pos, b1)
            mujoco.mj_jac(model, data, j2, None, ct.pos, b2)
            jsep_full = normal @ (j1 - j2)
            row_coef = jsep_full[dofadr]   # restrict to THIS limb's dof columns
            if np.linalg.norm(row_coef) < 1e-9:
                continue   # neither side of the contact moves with this limb
            for j in range(k):
                if abs(row_coef[j]) > 1e-12:
                    r.append(row); c.append(s0 + j); v.append(row_coef[j])
            l.append(min(pen, 0.05)); u.append(1e6); row += 1
            if is_floor:
                n_floor += 1
            else:
                n_self += 1

        # Swing clearance (T5.3): only when this limb's own contact alpha is
        # near 0 and its support point is within the activation band. Skipped
        # entirely for pseudo-limbs with no support sites (root-z, H1) -- the
        # concept doesn't apply to a whole-body translation DOF.
        alpha = float(contact_alpha_env[t])
        if resolved["support_sites"] and alpha < 0.5:
            site_id, z = _lowest_support_site(data, resolved["support_sites"])
            if z < swing_band:
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, None, site_id)
                row_coef = jacp[2, dofadr]   # z-row only
                if np.linalg.norm(row_coef) > 1e-9:
                    w = np.sqrt(max(1.0 - alpha, 0.0))   # fades OUT as contact fades IN
                    for j in range(k):
                        if abs(row_coef[j]) > 1e-12:
                            r.append(row); c.append(s0 + j); v.append(w * row_coef[j])
                    l.append(w * (swing_clearance - z)); u.append(1e6); row += 1
                    n_swing += 1

    jl_lo = np.tile(np.full(k, -trust_region), T)
    jl_hi = np.tile(np.full(k, trust_region), T)
    A_jl = sp.eye(N, format="csc")

    if row == 0:
        A, l_full, u_full, P_use, q_use = A_jl, jl_lo, jl_hi, P, g
        m = 0
    else:
        m = row
        A_ineq = sp.csc_matrix((v, (r, c)), shape=(m, N))
        l = np.array(l); u = np.array(u)
        P_slack = sp.diags(np.full(m, 2.0 * COLL_PENALTY), format="csc")
        P_use = sp.block_diag([P, P_slack], format="csc")
        q_use = np.concatenate([g, np.zeros(m)])
        A_jl_aug = sp.hstack([A_jl, sp.csc_matrix((N, m))], format="csc")
        A_ineq_aug = sp.hstack([A_ineq, sp.eye(m, format="csc")], format="csc")
        A_slack = sp.hstack([sp.csc_matrix((m, N)), sp.eye(m, format="csc")], format="csc")
        A = sp.vstack([A_jl_aug, A_ineq_aug, A_slack], format="csc")
        l_full = np.concatenate([jl_lo, l, np.zeros(m)])
        u_full = np.concatenate([jl_hi, u, np.full(m, 1e6)])

    prob = osqp.OSQP()
    prob.setup(P_use.tocsc(), q_use, A, l_full, u_full, warm_starting=True, verbose=False,
               eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polish=True)
    res = prob.solve()
    if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
        return np.zeros((T, k)), n_floor, n_self, n_swing
    delta = res.x[:N].reshape(T, k)
    return delta, n_floor, n_self, n_swing


def _apply_limb_delta(qpos, resolved, delta):
    out = qpos.copy()
    qadr = resolved["qpos_adr"]
    out[:, qadr] = qpos[:, qadr] + delta
    return out


def _achieved_positions(model, data, qpos, role_to_body, role_names):
    """(T, n_roles, 3) achieved Cartesian position per tracked role -- used as
    the BEFORE reference for the tracking-delta gate (compares this pass's
    OWN input/output, not the original human target, matching
    physics_plausibility_pass.py's _tracking_delta_rms_cm convention: this is
    a cleanup pass, its job is "don't move things much," not "re-track")."""
    T = qpos.shape[0]
    out = np.zeros((T, len(role_names), 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for ri, role in enumerate(role_names):
            if role in role_to_body:
                out[t, ri] = data.xpos[role_to_body[role]]
    return out


def _ref_support_xy(model, data, qpos, resolved_all):
    """Per limb: (T, 2) support-point XY in the INPUT qpos -- the BEFORE
    reference for the plant-slip score term below."""
    T = qpos.shape[0]
    out = {}
    for limb, resolved in resolved_all.items():
        arr = np.zeros((T, 2))
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            arr[t] = np.mean([data.site_xpos[s][:2] for s in resolved["support_sites"]], axis=0)
        out[limb] = arr
    return out


def _score(model, data, qpos, floor_gid, ref_achieved, role_to_body, role_names, limb_body_ids,
           resolved_all, contact_alpha_all, ref_support_xy):
    """Lexicographic keep-best score: (hard-fail LIMB-floor-pen gate, PLANT
    SLIP, tracking delta FROM THE INPUT's own achieved positions,
    self-collision depth) -- adapted from Stage 4's stage_b.

    Floor penetration is split LIMB (this pass's own body chains -- fixable,
    gated on PEN_TOL) vs CORE (torso/pelvis/head/spine -- frozen by design,
    T5.1's "Root FROZEN", architecturally out of this pass's reach; reported,
    never gated on). Gating on the combined max would make the pass either
    fail to converge chasing something it structurally cannot fix, or (if the
    gate silently ignored it) hide a real fixable violation behind a
    pre-existing lying-phase torso/pelvis-near-floor frame -- see planLog.md
    M5 for the luigi_standProne_03 case that surfaced this (TORSO_LINK
    penetration during its prone lying phase).

    PLANT SLIP is its own score term, not folded into track_rms: without it,
    keep-best has no reason to prefer a low-slip round among rounds with
    similar floor-pen/track_rms, and CAN select a round that fixed floor
    penetration while leaving a large (10cm+) planted-frame horizontal shift
    -- the exact "foot_floor_err... essential once floor rows exist" lesson
    Stage 4 already learned (wiki/concepts/globalopt.md's keep-best section)."""
    T = qpos.shape[0]
    max_floor_pen_limb = 0.0
    max_floor_pen_core = 0.0
    max_self_pen = 0.0
    track_errs = []
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for cc in range(data.ncon):
            ct = data.contact[cc]
            is_floor = ct.geom1 == floor_gid or ct.geom2 == floor_gid
            b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
            if not is_floor and (b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS)):
                continue
            pen = -float(ct.dist)
            if pen <= 0:
                continue
            if is_floor:
                other = b2 if ct.geom1 == floor_gid else b1
                if other in limb_body_ids:
                    max_floor_pen_limb = max(max_floor_pen_limb, pen)
                else:
                    max_floor_pen_core = max(max_floor_pen_core, pen)
            else:
                max_self_pen = max(max_self_pen, pen)
        for ri, role in enumerate(role_names):
            if role not in role_to_body:
                continue
            track_errs.append(float(np.linalg.norm(ref_achieved[t, ri] - data.xpos[role_to_body[role]])))
    track_rms = float(np.sqrt(np.mean(np.square(track_errs)))) if track_errs else 0.0

    max_plant_slip = 0.0
    for limb, resolved in resolved_all.items():
        alpha = contact_alpha_all[limb]
        planted = np.where(alpha > 0.5)[0]
        if planted.size == 0:
            continue
        for t in planted:
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            xy = np.mean([data.site_xpos[s][:2] for s in resolved["support_sites"]], axis=0)
            max_plant_slip = max(max_plant_slip, float(np.linalg.norm(xy - ref_support_xy[limb][t])))

    hard = max(0.0, max_floor_pen_limb * 100.0 - PEN_TOL_CM)
    return (hard, max_plant_slip, track_rms, max_floor_pen_limb + max_self_pen, max_self_pen, max_floor_pen_core)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH_DEFAULT)
    ap.add_argument("--posture-reg", type=float, default=POSTURE_REG_DEFAULT)
    ap.add_argument("--track-weight", type=float, default=TRACK_DEFAULT)
    ap.add_argument("--plant-hold-boost", type=float, default=300.0,
                    help="Multiplier on the HORIZONTAL (XY) component of the Cartesian tracking "
                         "ridge for frames where this limb's own contact is active (>0.5) -- "
                         "keeps planted frames from slipping despite corrections on nearby swing "
                         "frames (Z is NOT boosted, so a floor-fix on the SAME frame is never "
                         "fought -- see _solve_limb_qp's docstring). Calibrated at 300 on "
                         "standup_01 (a severe case: the same leg swings 11cm into the floor AND "
                         "plants elsewhere in the clip) as a balance between floor-pen compliance "
                         "and plant slip; higher values push floor_pen up without proportionally "
                         "reducing slip further. See planLog.md M5.")
    ap.add_argument("--swing-clearance", type=float, default=0.02,
                    help="Minimum support-point height (m) for a non-contact limb near the "
                         "floor. Default 0.02 (2cm).")
    ap.add_argument("--swing-band", type=float, default=0.10,
                    help="Activation band (m): swing clearance only checked when the support "
                         "point is within this height above the floor. Default 0.10 (10cm).")
    ap.add_argument("--n-rounds", type=int, default=10,
                    help="Gauss-Seidel rounds over the 4 limbs. plan.md specified 2; empirically "
                         "insufficient to fully resolve a large pre-existing swing-foot violation "
                         "(standup_01: 2 rounds -> 1.15cm residual limb-floor-pen, still above "
                         "the 0.5cm gate; 6 rounds -> 0.00cm. kneelingFall_02 needed 7 rounds for "
                         "its deeper 15.5cm initial violation). 10 gives safety margin; runtime is "
                         "cheap (~2-16s/clip depending on length) and keep-best-iterate makes more "
                         "rounds strictly safe (never ships worse than an earlier round).")
    ap.add_argument("--root-z", action="store_true", default=False,
                    help="hierarchical-v1 H1 (plan.md): enable an additional 1-DOF root-Z "
                         "pseudo-limb (root x/y/orientation stay hard-frozen) targeting the 7/20 "
                         "whole-body-lying clips M5 could not reach (CORE-classified floor "
                         "penetration, out of any limb chain's reach). Default OFF -- verified "
                         "byte-identical no-op via two consecutive off-runs.")
    ap.add_argument("--root-z-trust-region", type=float, default=0.03,
                    help="Per-round trust region (m) for the root-Z pseudo-limb. Default 0.03 "
                         "(3cm) -- deliberately tighter than the 0.10rad limb TRUST_REGION since "
                         "this is a whole-body translation, not a single joint.")
    ap.add_argument("--root-z-start-round", type=int, default=1,
                    help="0-indexed Gauss-Seidel round at which the root-Z step activates "
                         "(default 1 = the SECOND round) -- plan.md H1: let the 4 per-limb "
                         "solves get first crack alone before introducing a whole-body shift.")
    args = ap.parse_args()

    model, data, floor_gid, floor_mocap_id = _load_model_with_floor(args.model)
    resolved_all = _resolve_limbs(model)

    resolved_root_z = None
    if args.root_z:
        pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "PELVIS_LINK")
        resolved_root_z = dict(qpos_adr=np.array([2]), dof_adr=np.array([2]),
                                eff_bid=pelvis_bid, support_sites=[], k=1)
        root_z_alpha_env = None  # set once T is known, below

    z = np.load(args.npz, allow_pickle=True)
    qpos_in = np.asarray(z["qpos"], dtype=np.float64)
    fps = float(z["fps"])
    dt = 1.0 / fps
    T = qpos_in.shape[0]
    if resolved_root_z is not None:
        root_z_alpha_env = np.zeros(T)  # no swing-clearance concept for root-z; alpha always "planted"

    eff_names = [str(x) for x in z["contact_effector_names"]] if "contact_effector_names" in z.files else []
    contact_flags = np.asarray(z["contact_flags"], dtype=bool) if "contact_flags" in z.files else np.zeros((T, 0), bool)

    role_names = [str(r) for r in z["role_names"]] if "role_names" in z.files else []
    role_to_body = {}
    if role_names and "alex_body_names" in z.files:
        for ri, role in enumerate(role_names):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(z["alex_body_names"][ri]))
            if bid >= 0:
                role_to_body[role] = bid
    print(f"[refine-limbs] {args.npz.name}  T={T} fps={fps}")

    contact_alpha_all = {
        limb: _contact_alpha(contact_flags, eff_names, LIMB_CONTACT_EFF[limb], fps)
        for limb in LIMB_ORDER
    }

    # Reference = the INPUT's OWN achieved positions (not the original human
    # target) -- this pass's job is "don't move things much while fixing
    # floor/collision," not re-tracking, matching
    # physics_plausibility_pass.py's _tracking_delta_rms_cm convention.
    ref_achieved = _achieved_positions(model, data, qpos_in, role_to_body, role_names) if role_to_body else None
    limb_body_ids = _limb_body_ids(model, resolved_all)
    ref_support_xy = _ref_support_xy(model, data, qpos_in, resolved_all)

    # Regularization targets for _solve_limb_qp's ridges (posture + Cartesian
    # tracking): the ORIGINAL input, fixed for the whole Gauss-Seidel loop --
    # see _solve_limb_qp's docstring for why this must NOT be qpos_cur.
    eff_ref_pos_all = {}
    for limb, resolved in resolved_all.items():
        arr = np.zeros((T, 3))
        for t in range(T):
            data.qpos[:] = qpos_in[t]
            mujoco.mj_forward(model, data)
            arr[t] = data.xpos[resolved["eff_bid"]]
        eff_ref_pos_all[limb] = arr

    def score_fn(q):
        if ref_achieved is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return _score(model, data, q, floor_gid, ref_achieved, role_to_body, role_names, limb_body_ids,
                     resolved_all, contact_alpha_all, ref_support_xy)

    qpos_cur = qpos_in.copy()
    best_qpos = qpos_cur.copy()
    best_score = score_fn(best_qpos)
    print(f"  warm: floor_pen(limb)={best_score[0]+PEN_TOL_CM if best_score[0]>0 else 0:.2f}cm(hard-adj) "
          f"plant_slip={best_score[1]*100:.2f}cm track={best_score[2]:.4f}m "
          f"pen+self={best_score[3]*100:.2f}cm selfpen={best_score[4]*100:.2f}cm "
          f"floor_pen(core,unfixable-by-design)={best_score[5]*100:.2f}cm")

    # PATIENCE-based reset (trust-region "reject a bad step, retry from the
    # last good point" -- but not on the FIRST failure): a round that fails
    # to beat best_score is allowed to keep accumulating for up to
    # PATIENCE consecutive failures before the loop resets qpos_cur back to
    # best_qpos. Two unconditional alternatives were tried and rejected
    # first (see planLog.md M5 for full numbers):
    #   - NO reset ever (pure accumulation): diverges on clips with a severe,
    #     widespread violation -- standup_02's pen+self grew monotonically
    #     11.26->26.72cm over 10 rounds with no recovery.
    #   - RESET on the very FIRST failure, every time: fixed the divergence
    #     but also discarded near-misses that needed one more accumulating
    #     round to close -- shovel_fronthard_02 regressed from a clean
    #     0.00cm (a round-1 result just 0.03cm short of beating warm, which
    #     round 2 would have closed by continuing to accumulate) to a
    #     stuck-at-3.16cm early exit, because resetting to best before round
    #     2 threw away round 1's near-miss and round 2 just reproduced it.
    # PATIENCE=2 lets a near-miss get one extra round to converge (fixes the
    # shovel case) while still catching genuine divergence within a couple
    # rounds (standup_02 diverges much faster than 2 rounds' grace period).
    PATIENCE = 2
    round_stats = []
    consecutive_failures = 0
    qpos_cur = best_qpos.copy()
    for rnd in range(args.n_rounds):
        if consecutive_failures >= PATIENCE:
            qpos_cur = best_qpos.copy()
            consecutive_failures = 0
        rnd_floor = rnd_self = rnd_swing = 0
        for limb in LIMB_ORDER:
            resolved = resolved_all[limb]
            delta, nf, ns, nsw = _solve_limb_qp(
                model, data, qpos_cur, limb, resolved, floor_gid, contact_alpha_all[limb],
                dt, args.lambda_smooth, args.posture_reg, args.track_weight,
                args.swing_clearance, args.swing_band, plant_hold_boost=args.plant_hold_boost)
            qpos_cur = _apply_limb_delta(qpos_cur, resolved, delta)
            rnd_floor += nf; rnd_self += ns; rnd_swing += nsw
        rnd_rootz = 0
        if resolved_root_z is not None and rnd >= args.root_z_start_round:
            # H1: whole-body root-Z shift, solved AFTER the 4 limbs each
            # round -- reuses _solve_limb_qp verbatim with the synthetic
            # 1-DOF struct. eff_ref_pos=None: no Cartesian tracking ridge (a
            # root-Z-only Jacobian has zero XY component anyway, so the
            # ridge would be Z-only and redundant with posture_reg's direct
            # pull toward qpos_ref[:,2]) -- the smaller trust region
            # (--root-z-trust-region, default 3cm/round) plus posture_reg
            # keep this step conservative by construction.
            delta_z, nf, ns, _ = _solve_limb_qp(
                model, data, qpos_cur, "root_z", resolved_root_z, floor_gid, root_z_alpha_env,
                dt, args.lambda_smooth, args.posture_reg, args.track_weight,
                swing_clearance=0.0, swing_band=0.0, plant_hold_boost=1.0,
                trust_region=args.root_z_trust_region)
            qpos_cur = _apply_limb_delta(qpos_cur, resolved_root_z, delta_z)
            rnd_floor += nf; rnd_self += ns; rnd_rootz = nf
        # clamp all actuated joints to their ranges (hinge joints: qpos == angle)
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 0:
                continue
            if bool(model.jnt_limited[j]):
                qa = int(model.jnt_qposadr[j])
                lo, hi = model.jnt_range[j]
                qpos_cur[:, qa] = np.clip(qpos_cur[:, qa], lo, hi)
        score = score_fn(qpos_cur)
        keep = score < best_score
        round_stats.append((rnd, rnd_floor, rnd_self, rnd_swing, score, keep))
        rootz_note = f" root_z_floor_rows={rnd_rootz}" if resolved_root_z is not None else ""
        print(f"  round {rnd+1}/{args.n_rounds}: floor_rows={rnd_floor} self_rows={rnd_self} "
              f"swing_rows={rnd_swing}{rootz_note}  plant_slip={score[1]*100:.2f}cm track={score[2]:.4f}m "
              f"pen+self(limb)={score[3]*100:.2f}cm selfpen={score[4]*100:.2f}cm "
              f"floor_pen(core)={score[5]*100:.2f}cm" + (" *best" if keep else ""))
        if keep:
            best_score = score
            best_qpos = qpos_cur.copy()
            consecutive_failures = 0
        else:
            consecutive_failures += 1

    # Post-hoc verification -- split limb vs core exactly like _score does
    final_score = best_score
    floor_pen_limb_cm = 0.0
    floor_pen_core_cm = 0.0
    for t in range(T):
        data.qpos[:] = best_qpos[t]
        mujoco.mj_forward(model, data)
        for cc in range(data.ncon):
            ct = data.contact[cc]
            if ct.geom1 != floor_gid and ct.geom2 != floor_gid:
                continue
            pen_cm = -float(ct.dist) * 100.0
            if pen_cm <= 0:
                continue
            other = int(model.geom_bodyid[ct.geom2 if ct.geom1 == floor_gid else ct.geom1])
            if other in limb_body_ids:
                floor_pen_limb_cm = max(floor_pen_limb_cm, pen_cm)
            else:
                floor_pen_core_cm = max(floor_pen_core_cm, pen_cm)
    core_label = "core,root-z-eligible" if resolved_root_z is not None else "core,unfixable-by-design"
    print(f"  Final: floor_pen(limb)={floor_pen_limb_cm:.2f}cm (gate <= {PEN_TOL_CM}cm) "
          f"floor_pen({core_label})={floor_pen_core_cm:.2f}cm "
          f"plant_slip={final_score[1]*100:.2f}cm track_rms={final_score[2]*100:.3f}cm "
          f"selfpen={final_score[4]*100:.2f}cm")

    dq_act = np.diff(best_qpos[:, 7:], axis=0)
    spikes = int(np.sum(np.abs(dq_act).max(axis=1) > 0.5))
    print(f"  Velocity spikes (|dq|>0.5rad/frame): {spikes}")

    # Root-frozen invariant check -- split x/y/orientation (must ALWAYS be
    # exactly 0, root-z or not) from root-z (0 unless --root-z enabled it).
    root_xyquat_diff = np.abs(np.delete(best_qpos[:, 0:7] - qpos_in[:, 0:7], 2, axis=1)).max()
    root_z_diff_cm = float(np.abs(best_qpos[:, 2] - qpos_in[:, 2]).max()) * 100.0
    print(f"  Root-frozen check (x,y,quat): max|delta|={root_xyquat_diff:.2e} (should be exactly 0.0)")
    print(f"  Root-Z delta: max={root_z_diff_cm:.2f}cm "
          f"({'root-z DOF enabled' if resolved_root_z is not None else 'disabled -- should be exactly 0.0'})")

    out = {k: z[k] for k in z.files}
    out["qpos"] = best_qpos
    out["qpos_pre_limb_refine"] = qpos_in
    out["limb_refine_meta_json"] = json.dumps({
        "swing_clearance": args.swing_clearance, "swing_band": args.swing_band,
        "n_rounds": args.n_rounds,
        "floor_pen_limb_cm": floor_pen_limb_cm, "floor_pen_core_cm": floor_pen_core_cm,
        "plant_slip_cm": final_score[1] * 100.0, "track_rms_cm": final_score[2] * 100.0,
        "selfpen_cm": final_score[4] * 100.0,
        "spikes": spikes, "root_frozen_xyquat_max_diff": float(root_xyquat_diff),
        "root_z_enabled": bool(args.root_z), "root_z_max_delta_cm": root_z_diff_cm,
        "root_z_trust_region": args.root_z_trust_region, "root_z_start_round": args.root_z_start_round,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

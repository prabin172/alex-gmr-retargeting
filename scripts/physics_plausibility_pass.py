#!/usr/bin/env python3
"""Physics plausibility pass (phasic-v2 M4/P3).

Input: a grounded NPZ (`outputs/grounded_contactfirst/<clip>_grounded.npz`,
Stage 4.5 output) or any NPZ carrying `qpos (T,36)` + `fps` (+
`contact_flags (T,4)`/`contact_effector_names` if Increment 2 is explicitly
enabled). Output: same schema, `qpos` replaced by the plausibility-corrected
trajectory (all other keys passed through unchanged).

Separate phase from GlobalOPT (Stage 4) by design (plan.md P3 vs P2): Stage 4
optimizes tracking/smoothness/contacts; this pass ONLY clips joint/root
velocity/acceleration into plausible bounds (Increment 1), independently
ablatable. Flag-gated at the pipeline level (default OFF) — see
retargetingPipeline.sh's PHYSICS_PASS knob.

Increment 2 (CoM ground-projection inside the support polygon) is BUILT but
DISABLED BY DEFAULT (`--enable-com` to turn it on; decision 2026-07-10, see
planLog.md M4/T4.2). A purely kinematic CoM/support-polygon estimate has no
mass/inertia/contact-force data behind it, so it can't distinguish a genuine
static-balance problem from a dynamic posture legitimately leaning on
momentum — measured ~40cm "violations" on luigi_standSupine_08's get-up
transition that are not real problems, just a lying-to-standing pose using
momentum rather than static balance. Correctly judging that requires actual
dynamics data (mass, inertia, contact forces), which belongs in a later
physics-aware training loop, not this kinematic pass. The code (below) is
kept, tested, and documented as groundwork should that revisit happen — see
its own docstrings for the 4 real bugs found and fixed while building it
(single-support false positives, an independent-QP vel/accel violation, a
solver-tolerance false positive, and the large-violation over-correction that
led to this disable-by-default decision).

Increment 1: joint velocity/acceleration box rows + root linear/angular
velocity/acceleration box rows (hard box constraints, see below).

Increment 2: CoM ground-projection inside the support polygon during
DOUBLE-SUPPORT still plants ONLY. Support polygon per checked frame = convex
hull of the XY sole-corner positions of BOTH feet, on frames where BOTH are
simultaneously a STILL PLANT (contact-labelled AND sole-centroid speed <
`--still-speed`, matching the corpus-wide convention — see
scripts/ground_canonical_human.py, Stage 4's `_compute_anchors`, Stage 4.5's
`_planted_foot_sole_samples`), shrunk inward by `--com-margin`. Single-foot
stances are DELIBERATELY EXCLUDED, not treated as "polygon = that one foot's
sole": tested including them and found the majority of "violations" were
single-support frames mid dynamic weight-transfer (the other foot swinging),
with excursions up to 19cm — real human locomotion doesn't balance statically
over one foot during a fast transfer, it uses momentum, and forcing that
correction blew tracking delta to 27cm RMS and reintroduced vel/accel
violations (see planLog.md M4/T4.2). Frames without a genuine double-support
stance are UNCONSTRAINED (no CoM check) — dynamic/swing/single-support phases
aren't expected to be statically balanced, consistent with the design
philosophy's explicit deferral of CoM/stability constraints to downstream
physics-RL (wiki/concepts/design-philosophy.md NEXT-4 in the older roadmap).
ONE-SIDED SOFT-SLACK rows (same pattern as
`solve_global_trajectory_opt_contactfirst.py`'s `_build_collision`): a
violated half-space gets a row `n·J_com_xy·δQ_t − s ≤ −margin−violation`,
`s≥0`, quadratic penalty on `s` — always feasible, degrades gracefully
instead of risking QP infeasibility on a frame that genuinely can't reach the
polygon within a small correction.

Math: decision variable delta_Q in R^{T*nv} is a per-frame TANGENT-SPACE
perturbation (nv=35: 6 free-root DOF + 29 actuated joints — MuJoCo's own
qvel/qacc convention, via mj_differentiatePos/mj_integratePos, so the
free-joint quaternion is handled correctly with no hand-rolled quaternion
math). To first order (valid for small corrections, which is what a
"plausibility clip" should ever need — if this pass wants to move something
by more than a few percent of a limit, the INPUT trajectory has a real
problem this pass should not paper over):

  qvel_corrected(t)  ~= qvel_raw(t)  + (dQ[t+1] - dQ[t]) / dt
  qaccel_corrected(t) ~= qaccel_raw(t) + (dQ[t+1] - 2*dQ[t] + dQ[t-1]) / dt^2

Objective: minimize ||delta_Q||^2 (least perturbation) subject to hard box
constraints on qvel_corrected and qaccel_corrected. Reuses OSQP the same way
Stage 4 does (scripts/solve_global_trajectory_opt_contactfirst.py stage_b).

Velocity/acceleration limits: the robot model (assets/alex/alex_floating_base_with_sites.xml)
has NO actuator or velocity spec at all (checked directly, read-only — no
<actuator> section, joints carry only `range`/`actuatorfrcrange`; this matches
the design philosophy of a kinematics-only pipeline, see CLAUDE.md/
design-philosophy.md: "Physics-RL absorbs dynamics errors"). So per plan.md
T4.1 ("else conservative defaults — document the source"), the defaults below
are NOT model-derived: they were picked as ~2-3x headroom over the OBSERVED
peak on 4 representative clips post-M2/M3 (standup_01, standup_natural_02,
kneelingFall_02, shovel_fronthard_02 — measured via mj_differentiatePos):
joint_vel peak 3.8-12.6 rad/s, joint_acc peak 31-125 rad/s^2, root_lin_vel
peak 0.4-0.8 m/s, root_lin_acc peak 1.3-2.4 m/s^2, root_ang_vel peak 0.65-2.0
rad/s, root_ang_acc peak 3.7-5.7 rad/s^2 (see planLog.md M4). This is a
PLAUSIBILITY check, not a hardware-accurate limit — the goal is catching
genuine insanity (a residual spike Stage 4 missed), not nailing a torque-
accurate bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import scipy.sparse as sp
from scipy.spatial import ConvexHull, QhullError
import osqp

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"

NV = 35   # 6 free-root DOF + 29 actuated (this model; asserted against model.nv at load)
ROOT_BODY_NAME = "PELVIS_LINK"   # kinematic-tree root; subtree_com here = whole-robot CoM

# Duplicated from solve_global_trajectory_opt_contactfirst.py / post_process_ground_contactfirst.py
# / solve_fbx_canonical_alex_contactfirst.py (independent CLI scripts, no shared imports for
# solver-internal constants — established convention, see _load_model_with_floor elsewhere).
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
COM_PENALTY = 1000.0   # slack quadratic penalty, matches Stage 4's collision_penalty convention

# Conservative defaults — NOT from the model (no actuator/velocity spec exists).
# See module docstring for the calibration this was based on.
JOINT_VEL_LIMIT = 25.0        # rad/s
JOINT_ACC_LIMIT = 400.0       # rad/s^2
ROOT_LIN_VEL_LIMIT = 3.0      # m/s
ROOT_LIN_ACC_LIMIT = 10.0     # m/s^2 (~1g)
ROOT_ANG_VEL_LIMIT = 6.0      # rad/s
ROOT_ANG_ACC_LIMIT = 20.0     # rad/s^2


def _qvel_series(model, qpos, dt):
    """(T-1, nv) tangent-space velocity between consecutive raw frames."""
    T = qpos.shape[0]
    out = np.zeros((T - 1, model.nv))
    for t in range(T - 1):
        v = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, v, dt, qpos[t], qpos[t + 1])
        out[t] = v
    return out


# Internal safety margin for the QP's own bound construction, root-ANGULAR
# channels only (dofs 3:6). The decision variable is a first-order tangent-
# space model; retracting it through mj_integratePos's quaternion exponential
# map and re-differentiating afterward is exact for every Euclidean DOF (root
# linear, all 29 actuated joints) but introduces a small nonlinearity residual
# for root orientation specifically (measured: up to ~1% overshoot of the true
# bound on luigi_standProne_03 — see planLog.md M4). Shrinking ONLY the
# internal angular bounds by this margin absorbs that residual so the
# POST-HOC check (against the true, unshrunk, documented limits) still
# passes. Reported/verified limits (module constants above, printed output,
# saved metadata) are always the true nominal values — this margin is purely
# an internal QP-construction detail.
_ANGULAR_MARGIN = 0.90

# Post-hoc verification tolerances. Acceleration rows carry a 1/dt^2 coefficient
# (14400 at 120 Hz) — OSQP's own solve tolerance (eps_abs=1e-5, QP-space) gets
# amplified by that factor when mapped back to physical accel units, so a
# solver-precision residual near a bound reads as an "overshoot" of up to a
# few thousandths there even though the QP itself reports "solved" cleanly.
# Measured max on the corpus: 0.0041 (physical units) on luigi_standSupine_08's
# combined CoM+vel/accel QP — real solver noise, not a modeling error (see
# planLog.md M4/T4.2). Velocity rows carry only 1/dt (120x), no comparable
# issue observed, kept tight.
VEL_CHECK_TOL = 1e-3
ACC_CHECK_TOL = 1e-2


def _vel_bounds(nv):
    lo = np.full(nv, -JOINT_VEL_LIMIT)
    hi = np.full(nv, JOINT_VEL_LIMIT)
    lo[0:3] = -ROOT_LIN_VEL_LIMIT; hi[0:3] = ROOT_LIN_VEL_LIMIT
    lo[3:6] = -ROOT_ANG_VEL_LIMIT; hi[3:6] = ROOT_ANG_VEL_LIMIT
    return lo, hi


def _vel_bounds_internal(nv):
    lo, hi = _vel_bounds(nv)
    lo[3:6] *= _ANGULAR_MARGIN; hi[3:6] *= _ANGULAR_MARGIN
    return lo, hi


def _acc_bounds(nv):
    lo = np.full(nv, -JOINT_ACC_LIMIT)
    hi = np.full(nv, JOINT_ACC_LIMIT)
    lo[0:3] = -ROOT_LIN_ACC_LIMIT; hi[0:3] = ROOT_LIN_ACC_LIMIT
    lo[3:6] = -ROOT_ANG_ACC_LIMIT; hi[3:6] = ROOT_ANG_ACC_LIMIT
    return lo, hi


def _acc_bounds_internal(nv):
    lo, hi = _acc_bounds(nv)
    lo[3:6] *= _ANGULAR_MARGIN; hi[3:6] *= _ANGULAR_MARGIN
    return lo, hi


def _vel_acc_rows(model, qpos, dt):
    """Build the velocity + acceleration box-constraint rows (as hard equality-
    free inequality rows on delta_Q) for the given qpos. Returns (r, c, v, l,
    u, n_vel_rows, n_acc_rows) — raw COO triplets + bounds, NOT yet a
    csc_matrix (caller stacks these with other rows, e.g. _build_com_qp adds
    CoM rows on top of the SAME vel/acc rows so Increment 2 structurally
    cannot reintroduce a velocity/acceleration violation Increment 1 fixed —
    see planLog.md M4/T4.2 for why this sharing is necessary: two
    INDEPENDENT least-perturbation QPs is not safe, the second can make an
    isolated large correction that blows the first's constraints (measured:
    root linear acceleration -577.7 m/s^2 vs a +-10.0 bound when tried as two
    fully separate problems)."""
    T = qpos.shape[0]
    nv = model.nv

    qvel_raw = _qvel_series(model, qpos, dt)          # (T-1, nv)
    qaccel_raw = np.diff(qvel_raw, axis=0) / dt        # (T-2, nv), centered frames 1..T-2

    vel_lo, vel_hi = _vel_bounds_internal(nv)
    acc_lo, acc_hi = _acc_bounds_internal(nv)

    # Velocity rows: one block-row per (t, dof), coefficients on dQ[t] and
    # dQ[t+1] — same adjacency pattern as Stage 4's _build_smoothness_hessian,
    # but as inequality rows (not a quadratic Hessian).
    r, c, v, l, u = [], [], [], [], []
    row = 0
    for t in range(T - 1):
        s0 = t * nv
        s1 = (t + 1) * nv
        for j in range(nv):
            r.append(row); c.append(s0 + j); v.append(-1.0 / dt)
            r.append(row); c.append(s1 + j); v.append(1.0 / dt)
            l.append(vel_lo[j] - qvel_raw[t, j])
            u.append(vel_hi[j] - qvel_raw[t, j])
            row += 1
    n_vel_rows = row

    # Acceleration rows: 3-point stencil on dQ[t-1], dQ[t], dQ[t+1] for
    # interior frames t=1..T-2 (matches qaccel_raw's indexing: qaccel_raw[t-1]
    # is centered at frame t).
    for t in range(1, T - 1):
        sm = (t - 1) * nv
        s0 = t * nv
        sp_ = (t + 1) * nv
        qa = qaccel_raw[t - 1]
        for j in range(nv):
            r.append(row); c.append(sm + j); v.append(1.0 / dt ** 2)
            r.append(row); c.append(s0 + j); v.append(-2.0 / dt ** 2)
            r.append(row); c.append(sp_ + j); v.append(1.0 / dt ** 2)
            l.append(acc_lo[j] - qa[j])
            u.append(acc_hi[j] - qa[j])
            row += 1
    n_acc_rows = row - n_vel_rows
    return r, c, v, l, u, n_vel_rows, n_acc_rows


def _build_qp(model, qpos, dt, ridge=1e-3):
    """Least-perturbation QP: minimize ||delta_Q||^2 subject to velocity and
    acceleration box constraints on the CORRECTED trajectory. Returns
    (delta_Q (T,nv), n_vel_active, n_acc_active) — the last two are diagnostic
    counts of how many rows were actually binding (near-zero for an
    already-plausible input, which is the expected common case)."""
    T = qpos.shape[0]
    nv = model.nv
    N = T * nv

    r, c, v, l, u, n_vel_rows, n_acc_rows = _vel_acc_rows(model, qpos, dt)
    row = n_vel_rows + n_acc_rows

    A = sp.csc_matrix((v, (r, c)), shape=(row, N))
    l = np.array(l); u = np.array(u)

    P = sp.diags(np.full(N, 2.0 * ridge), format="csc")
    q_vec = np.zeros(N)

    prob = osqp.OSQP()
    prob.setup(P, q_vec, A, l, u, warm_starting=True, verbose=False,
               eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polish=True)
    res = prob.solve()
    if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
        raise RuntimeError(f"physics plausibility QP failed: {res.info.status}")

    delta_Q = res.x.reshape(T, nv)

    # Diagnostic: how many rows were actually binding (near a bound) —
    # near-zero for an already-plausible input.
    Ax = A @ res.x
    n_vel_active = int(np.sum((np.abs(Ax[:n_vel_rows] - l[:n_vel_rows]) < 1e-4) |
                              (np.abs(Ax[:n_vel_rows] - u[:n_vel_rows]) < 1e-4)))
    n_acc_active = int(np.sum((np.abs(Ax[n_vel_rows:] - l[n_vel_rows:]) < 1e-4) |
                              (np.abs(Ax[n_vel_rows:] - u[n_vel_rows:]) < 1e-4)))
    return delta_Q, n_vel_active, n_acc_active


def _retract(model, qpos, delta_Q):
    T = qpos.shape[0]
    out = qpos.copy()
    for t in range(T):
        mujoco.mj_integratePos(model, out[t], delta_Q[t], 1.0)
    return out


# ---------------------------------------------------------------------------
# Increment 2: CoM support-polygon check
# ---------------------------------------------------------------------------

def _still_plant_mask(model, data, qpos, contact_flags, eff_names, fps, still_speed=0.05):
    """Per foot effector, per-frame bool: contact-labelled AND sole-centroid
    speed < still_speed. Matches the corpus-wide "still plant" convention
    (scripts/ground_canonical_human.py's still_plant_support_samples, Stage
    4's _compute_anchors, Stage 4.5's _planted_foot_sole_samples) but WITHOUT
    their min-run debounce — acceptable here since a brief single-frame
    misclassification just means the CoM check is skipped or engaged one
    frame too early/late on a plausibility pass, not a hard failure."""
    T = qpos.shape[0]
    dt = 1.0 / fps
    out = {}
    for eff, sole_names in SOLE_CORNER_SITES.items():
        if eff not in eff_names:
            continue
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in sole_names]
        ids = [i for i in ids if i >= 0]
        if not ids:
            continue
        col = eff_names.index(eff)
        labelled = contact_flags[:, col]
        centroid = np.zeros((T, 3))
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            centroid[t] = np.mean([data.site_xpos[s] for s in ids], axis=0)
        speed = np.zeros(T)
        speed[1:] = np.linalg.norm(np.diff(centroid, axis=0), axis=1) / dt
        speed[0] = speed[1] if T > 1 else 0.0
        out[eff] = labelled & (speed < still_speed)
    return out


def _sole_corner_xy(model, data, qpos_t, eff):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in SOLE_CORNER_SITES[eff]]
    ids = [i for i in ids if i >= 0]
    data.qpos[:] = qpos_t
    mujoco.mj_forward(model, data)
    return np.array([data.site_xpos[s][:2] for s in ids])


def _build_com_qp(model, data, qpos, still_mask, root_bid, margin, dt, ridge=1e-3,
                   max_correction=0.08):
    """CoM support-polygon rows (soft-slack) PLUS the SAME velocity/
    acceleration hard box rows Increment 1 enforces (via _vel_acc_rows, on
    THIS qpos — i.e. Increment 1's own output — as the baseline), solved as
    ONE combined QP. Returns (delta_Q (T,nv), n_frames_checked,
    n_frames_violated, n_rows).

    Sharing the vel/accel rows here is NOT optional: an earlier version ran
    this as an INDEPENDENT least-perturbation QP (CoM rows only) and found it
    could reintroduce a severe velocity/acceleration violation Increment 1 had
    just fixed — measured on standup_01: root linear acceleration -577.7 m/s^2
    vs a +-10.0 m/s^2 bound, because a least-|delta_Q|-norm CoM correction
    concentrated at an isolated frame (a double-support window's edge) doesn't
    "know" about the derivative constraints Increment 1 enforced. Including
    the SAME rows here as hard constraints makes that structurally impossible
    — this QP can only find a delta_Q that satisfies BOTH goals (see
    planLog.md M4/T4.2 for the full before/after).

    Support polygon per frame = convex hull of the sole-corner XY points of
    BOTH feet, only on frames where BOTH are simultaneously still-planted —
    a genuine quasi-static standing/resting stance. Single-foot-only frames
    are DELIBERATELY EXCLUDED, not just "one foot's small footprint": tested
    including them (standup_01) and found 105/122 violations came from
    single-support frames with the OTHER foot mid-swing during a dynamic
    weight-transfer, violation depth up to 19cm — a real human doesn't
    balance statically with CoM directly over one foot during a fast
    single-leg transfer, they use momentum (design philosophy: CoM/stability
    is downstream physics-RL's job, NOT this pass's, except for genuine
    static double-support imbalance). Restricting to double-support cut it to
    17/318 frames, max 4.7cm depth — a physically sane scope this pass can
    actually help with. Frames without a genuine double-support stance are
    skipped entirely (no CoM constraint, though vel/accel rows still apply
    there).

    `max_correction` (default 8cm) caps what a SINGLE frame's violation is
    allowed to be for this pass to attempt correcting it — even restricted to
    double-support, tested on luigi_standSupine_08 and found frames with a
    genuine ~40cm CoM excursion (a get-up posture leaning on momentum, both
    feet happening to be still-planted but the pose is NOT a static balance
    failure). Forcing that fully-corrected in one shot produced a 43cm
    tracking-delta outlier and a body-wide 8.7cm RMS shift — same
    over-correction risk Increment 1 already guards against ("if this pass
    wants to move something by more than a few percent of a limit, the input
    has a real problem this pass should not paper over," see module
    docstring). Frames beyond the cap are FLAGGED (returned count), not
    corrected — informational, not a silent drop."""
    T = qpos.shape[0]
    nv = model.nv
    N = T * nv

    r, c, v, l, u, n_vel_rows, n_acc_rows = _vel_acc_rows(model, qpos, dt)
    n_va_rows = n_vel_rows + n_acc_rows
    row = n_va_rows
    n_checked = 0
    n_violated = 0
    n_flagged_large = [0]

    for t in range(T):
        feet_here = [eff for eff, mask in still_mask.items() if mask[t]]
        if len(feet_here) < 2:
            continue
        n_checked += 1
        pts = np.concatenate([_sole_corner_xy(model, data, qpos[t], eff) for eff in feet_here], axis=0)
        if pts.shape[0] < 3:
            continue   # degenerate (shouldn't happen: each foot contributes 4 corners)
        try:
            hull = ConvexHull(pts)
        except QhullError:
            continue   # near-degenerate (near-collinear corners) — skip, not worth failing over

        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        com_xy = data.subtree_com[root_bid][:2].copy()
        jacp = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(model, data, jacp, root_bid)
        J_com_xy = jacp[:2, :]   # (2, nv)

        frame_violated = False
        for eq in hull.equations:            # [A, B, C], inside iff A*x+B*y+C <= 0
            n_vec = eq[:2]
            viol = float(n_vec @ com_xy + eq[2] + margin)   # > 0 -> outside the shrunk polygon
            if viol <= 0.0:
                continue
            if viol > max_correction:
                # Large violation (measured up to 40cm on luigi_standSupine_08 —
                # a genuine get-up posture leaning on momentum, not a static-
                # balance failure a small nudge should paper over). Flagged as
                # diagnostic, NOT corrected — same principle as Increment 1's
                # "if this pass wants to move something by more than a few
                # percent of a limit, the input has a real problem this pass
                # should not paper over" (see module docstring).
                n_flagged_large[0] += 1
                frame_violated = True
                continue
            frame_violated = True
            row_coef = n_vec @ J_com_xy      # (nv,) — d(violation)/d(delta_Q_t)
            cs = t * nv
            for j in range(nv):
                if abs(row_coef[j]) > 1e-12:
                    r.append(row); c.append(cs + j); v.append(row_coef[j])
            l.append(-1e6); u.append(-viol)   # row_coef.δQ_t - s <= -viol  (s subtracted below)
            row += 1
        if frame_violated:
            n_violated += 1

    m_com = row - n_va_rows   # CoM rows only (vel/accel rows carry no slack)
    if m_com == 0:
        # No CoM violations this frame set -- still need to respect vel/accel
        # (matches _build_qp's own solve, so this degenerates to that).
        return np.zeros((T, nv)), n_checked, n_violated, 0, n_flagged_large[0]

    A_body = sp.csc_matrix((v, (r, c)), shape=(row, N))   # both vel/acc AND CoM rows, N cols
    l = np.array(l); u = np.array(u)

    P_ridge = sp.diags(np.full(N, 2.0 * ridge), format="csc")
    P_slack = sp.diags(np.full(m_com, 2.0 * COM_PENALTY), format="csc")
    P = sp.block_diag([P_ridge, P_slack], format="csc")
    q_vec = np.zeros(N + m_com)

    # Vel/accel rows (first n_va_rows): HARD, zero slack columns.
    # CoM rows (remaining m_com): A_com @ delta_Q - s <= u, s >= 0 — same
    # slack-row shape as Stage 4's _build_collision (always feasible, degrades
    # gracefully instead of risking infeasibility).
    slack_col = sp.vstack([
        sp.csc_matrix((n_va_rows, m_com)),
        -sp.eye(m_com, format="csc"),
    ], format="csc")
    A_row = sp.hstack([A_body, slack_col], format="csc")
    A_slack_nonneg = sp.hstack([sp.csc_matrix((m_com, N)), sp.eye(m_com, format="csc")], format="csc")
    A = sp.vstack([A_row, A_slack_nonneg], format="csc")
    l_full = np.concatenate([l, np.zeros(m_com)])
    u_full = np.concatenate([u, np.full(m_com, 1e6)])

    prob = osqp.OSQP()
    prob.setup(P, q_vec, A, l_full, u_full, warm_starting=True, verbose=False,
               eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polish=True)
    res = prob.solve()
    if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
        raise RuntimeError(f"CoM plausibility QP failed: {res.info.status}")

    delta_Q = res.x[:N].reshape(T, nv)
    slack = res.x[N:]
    return delta_Q, n_checked, n_violated, m_com, n_flagged_large[0]


def _tracking_delta_rms_cm(model, data, qpos_before, qpos_after, target_positions,
                            role_names, role_to_body):
    """RMS displacement (cm) of every tracked body between before/after —
    the gate's "effector tracking delta <= 1cm RMS" check. Compares ACHIEVED
    positions before vs after (not vs target — this pass shouldn't move
    tracking quality at all in the common case; a large shift here means the
    QP made a real correction, worth knowing even if targets are unaffected)."""
    diffs = []
    for t in range(qpos_before.shape[0]):
        data.qpos[:] = qpos_before[t]
        mujoco.mj_forward(model, data)
        pos_before = {role: data.xpos[bid].copy() for role, bid in role_to_body.items()}
        data.qpos[:] = qpos_after[t]
        mujoco.mj_forward(model, data)
        for role, bid in role_to_body.items():
            diffs.append(float(np.linalg.norm(data.xpos[bid] - pos_before[role])))
    arr = np.asarray(diffs) * 100.0
    return float(np.sqrt(np.mean(arr ** 2))), float(arr.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="Least-perturbation weight (higher = more reluctant to move "
                         "away from the input even to satisfy a bound). Default 1e-3.")
    ap.add_argument("--enable-com", dest="enable_com", action="store_true", default=False,
                    help="Run Increment 2 (CoM support-polygon check) after Increment 1. "
                         "Default OFF (2026-07-10 decision, see planLog.md M4/T4.2: a purely "
                         "kinematic CoM/support-polygon estimate has no mass/inertia/contact-"
                         "force data behind it, so it can't distinguish a genuine static-balance "
                         "problem from a dynamic get-up posture leaning on momentum (measured: "
                         "~40cm 'violations' on luigi_standSupine_08 that are not real problems). "
                         "Better addressed later by physics-aware training with actual dynamics "
                         "data, not this pass. Code kept, tested, and documented as groundwork.")
    ap.add_argument("--no-enable-com", dest="enable_com", action="store_false",
                    help="Run Increment 1 (velocity/acceleration) only. Default.")
    ap.add_argument("--still-speed", type=float, default=0.05,
                    help="Sole-centroid speed (m/s) below which a contact-labelled foot "
                         "counts as a STILL plant for the CoM check (default 0.05, matches "
                         "the corpus-wide convention — Stage 4.5's --still-speed, Stage 2.5's "
                         "--plant-speed).")
    ap.add_argument("--com-margin", type=float, default=0.02,
                    help="Inward shrink (m) of the support polygon before the CoM check "
                         "engages — a safety buffer, not requiring the CoM exactly at the "
                         "polygon edge. Default 0.02 (2cm).")
    ap.add_argument("--com-max-correction", type=float, default=0.08,
                    help="Cap (m) on a single frame's CoM violation depth for this pass to "
                         "attempt correcting — larger violations are flagged (diagnostic) "
                         "but not corrected (a large excursion is likely a genuine dynamic "
                         "posture leaning on momentum, not a static-balance failure a small "
                         "nudge should paper over; see _build_com_qp docstring). Default 0.08 (8cm).")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    if model.nv != NV:
        raise RuntimeError(f"unexpected model.nv={model.nv}, expected {NV} — "
                           "velocity/accel bound vectors assume this model's DOF layout")

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=np.float64)
    fps = float(z["fps"])
    dt = 1.0 / fps
    T = qpos.shape[0]

    print(f"[physics-plausibility] {args.npz.name}  T={T} fps={fps}")

    delta_Q, n_vel_active, n_acc_active = _build_qp(model, qpos, dt, ridge=args.ridge)
    qpos_out = _retract(model, qpos, delta_Q)

    max_dQ = float(np.abs(delta_Q).max())
    print(f"  QP: max|delta_Q|={max_dQ:.4f}  vel-rows-active={n_vel_active}  "
          f"acc-rows-active={n_acc_active}")

    # Post-hoc verification: recompute vel/accel on the CORRECTED trajectory
    # and confirm they're within bounds (independent check, not trusting the
    # QP's own row accounting).
    qvel_after = _qvel_series(model, qpos_out, dt)
    qaccel_after = np.diff(qvel_after, axis=0) / dt
    vel_lo, vel_hi = _vel_bounds(model.nv)
    acc_lo, acc_hi = _acc_bounds(model.nv)
    vel_ok = bool(np.all(qvel_after >= vel_lo - VEL_CHECK_TOL) and np.all(qvel_after <= vel_hi + VEL_CHECK_TOL))
    acc_ok = bool(np.all(qaccel_after >= acc_lo - ACC_CHECK_TOL) and np.all(qaccel_after <= acc_hi + ACC_CHECK_TOL))
    print(f"  Post-hoc gate: velocity-within-limits={vel_ok}  acceleration-within-limits={acc_ok}")
    if not (vel_ok and acc_ok):
        print("  [WARNING] post-hoc check failed -- QP solve may be inaccurate, investigate.")

    role_names = [str(r) for r in z["role_names"]] if "role_names" in z.files else []
    role_to_body = {}
    if role_names and "alex_body_names" in z.files:
        for ri, role in enumerate(role_names):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(z["alex_body_names"][ri]))
            if bid >= 0:
                role_to_body[role] = bid
    if role_to_body:
        rms_cm, max_cm = _tracking_delta_rms_cm(model, data, qpos, qpos_out,
                                                 z["target_positions"], role_names, role_to_body)
        print(f"  Tracking delta (before vs after, all tracked bodies): "
              f"RMS={rms_cm:.3f}cm max={max_cm:.3f}cm  (gate: RMS <= 1cm)")

    # Velocity spike check on the corrected trajectory (same criterion used
    # throughout phasic-v2: max|diff(qpos[:,7:])| > 0.5 rad in a single frame).
    dq_act = np.diff(qpos_out[:, 7:], axis=0)
    spikes = int(np.sum(np.abs(dq_act).max(axis=1) > 0.5))
    print(f"  Velocity spikes (|dq|>0.5rad/frame, actuated joints): {spikes}")

    com_meta = {"enabled": False}
    if args.enable_com:
        eff_names = [str(x) for x in z["contact_effector_names"]] if "contact_effector_names" in z.files else []
        flags = np.asarray(z["contact_flags"], dtype=bool) if "contact_flags" in z.files else None
        if not eff_names or flags is None:
            print("  [CoM check] SKIPPED — input NPZ has no contact_flags/contact_effector_names")
        else:
            root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY_NAME)
            still_mask = _still_plant_mask(model, data, qpos_out, flags, eff_names, fps,
                                           still_speed=args.still_speed)
            delta_Q_com, n_checked, n_violated, n_rows, n_flagged_large = _build_com_qp(
                model, data, qpos_out, still_mask, root_bid, args.com_margin, dt,
                ridge=args.ridge, max_correction=args.com_max_correction)
            qpos_before_com = qpos_out.copy()
            if n_rows > 0:
                qpos_out = _retract(model, qpos_out, delta_Q_com)
            max_dQ_com = float(np.abs(delta_Q_com).max())
            if n_flagged_large > 0:
                print(f"  [CoM check] [WARNING] {n_flagged_large} frame(s) had a CoM "
                      f"violation > --com-max-correction ({args.com_max_correction*100:.0f}cm) "
                      f"— flagged, NOT corrected (see _build_com_qp docstring)")
            print(f"  [CoM check] still-plant frames checked={n_checked}  "
                  f"violated(pre-correction)={n_violated}  rows={n_rows}  "
                  f"max|delta_Q|={max_dQ_com:.4f}")

            # Post-hoc: recompute violation on the CORRECTED trajectory.
            still_mask_after = _still_plant_mask(model, data, qpos_out, flags, eff_names, fps,
                                                 still_speed=args.still_speed)
            n_still_violated_after = 0
            for t in range(T):
                feet_here = [e for e, m in still_mask_after.items() if m[t]]
                if len(feet_here) < 2:
                    continue
                pts = np.concatenate([_sole_corner_xy(model, data, qpos_out[t], e) for e in feet_here], axis=0)
                if pts.shape[0] < 3:
                    continue
                try:
                    hull = ConvexHull(pts)
                except QhullError:
                    continue
                data.qpos[:] = qpos_out[t]
                mujoco.mj_forward(model, data)
                com_xy = data.subtree_com[root_bid][:2]
                if np.any(hull.equations[:, :2] @ com_xy + hull.equations[:, 2] > 1e-3):
                    n_still_violated_after += 1
            # Frames flagged as too-large-to-correct (max_correction cap) are
            # EXPECTED to still show as violated here — that's by design, not
            # a gate failure. Only an UNEXPECTED residual (more violations
            # than were flagged) indicates a real problem.
            com_ok = n_still_violated_after <= n_flagged_large
            print(f"  [CoM check] Post-hoc gate: still-planted CoM outside polygon "
                  f"AFTER correction = {n_still_violated_after}/{n_checked} frames "
                  f"({n_flagged_large} expected from the max-correction cap) "
                  f"({'PASS' if com_ok else 'UNEXPECTED residual violation'})")

            # Re-verify vel/accel still hold (should always pass now — they're
            # hard rows INSIDE the same combined QP as the CoM rows, see
            # _build_com_qp's docstring; this is a canary against a future
            # edit breaking that sharing, not a live correction path).
            qvel_after2 = _qvel_series(model, qpos_out, dt)
            qaccel_after2 = np.diff(qvel_after2, axis=0) / dt
            vel_ok2 = bool(np.all(qvel_after2 >= vel_lo - VEL_CHECK_TOL) and np.all(qvel_after2 <= vel_hi + VEL_CHECK_TOL))
            acc_ok2 = bool(np.all(qaccel_after2 >= acc_lo - ACC_CHECK_TOL) and np.all(qaccel_after2 <= acc_hi + ACC_CHECK_TOL))
            print(f"  [CoM check] vel/accel bounds still hold after CoM pass: "
                  f"velocity={vel_ok2} acceleration={acc_ok2}")
            if not (vel_ok2 and acc_ok2):
                print("  [WARNING] CoM pass reintroduced a vel/accel violation despite "
                      "shared hard constraints — investigate, this should not happen.")

            com_meta = {
                "enabled": True, "still_speed": args.still_speed, "com_margin": args.com_margin,
                "com_max_correction": args.com_max_correction,
                "frames_checked": n_checked, "frames_violated_pre": n_violated,
                "rows": n_rows, "frames_flagged_large_uncorrected": n_flagged_large,
                "max_delta_Q": max_dQ_com,
                "frames_violated_post": n_still_violated_after,
                "vel_within_limits_after": vel_ok2, "acc_within_limits_after": acc_ok2,
            }

            rms_cm2, max_cm2 = None, None
            if role_to_body:
                rms_cm2, max_cm2 = _tracking_delta_rms_cm(model, data, qpos_before_com, qpos_out,
                                                           z["target_positions"], role_names, role_to_body)
                print(f"  [CoM check] Tracking delta (before vs after CoM pass): "
                      f"RMS={rms_cm2:.3f}cm max={max_cm2:.3f}cm")

            dq_act2 = np.diff(qpos_out[:, 7:], axis=0)
            spikes2 = int(np.sum(np.abs(dq_act2).max(axis=1) > 0.5))
            print(f"  [CoM check] Velocity spikes after CoM pass: {spikes2}")

    out = {k: z[k] for k in z.files}
    out["qpos"] = qpos_out
    out["qpos_pre_plausibility"] = qpos
    out["plausibility_meta_json"] = json.dumps({
        "joint_vel_limit": JOINT_VEL_LIMIT, "joint_acc_limit": JOINT_ACC_LIMIT,
        "root_lin_vel_limit": ROOT_LIN_VEL_LIMIT, "root_lin_acc_limit": ROOT_LIN_ACC_LIMIT,
        "root_ang_vel_limit": ROOT_ANG_VEL_LIMIT, "root_ang_acc_limit": ROOT_ANG_ACC_LIMIT,
        "max_delta_Q": max_dQ, "vel_rows_active": n_vel_active, "acc_rows_active": n_acc_active,
        "velocity_within_limits": vel_ok, "acceleration_within_limits": acc_ok,
        "com_check": com_meta,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

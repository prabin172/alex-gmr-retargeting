#!/usr/bin/env python3
"""S6-A2: exact per-frame floor clamp, shared by Phase A (inline, inside GMR's
solve loop -- `gmr_contact_retarget.py --floor-clamp`) and Phase B (post-hoc,
corpus-level -- `polish_median_limbwise.py`). Do not reimplement this elsewhere.

Why this exists instead of a QP inequality constraint: S6-A1 (see planLogGMR.md)
tried appending `mink.CollisionAvoidanceLimit` to GMR's own solve. Even after fixing
a real bug in how GMR passes `ik_limits` into `mink.solve_ik` (positional arg lands
on `safety_break`, not `limits` -- see `gmr_contact_retarget.py`'s
`_solve_after_targets`), the rate-limited (CBF-style) QP inequality never converges
close to zero, because GMR's per-frame solve loop exits after ~1 iteration (task
error stops improving fast). Forcing 50 extra QP solves/frame got within ~0.34cm,
never exact, and GMR's own G1 XML excludes the foot's real mesh from collision
entirely (only 4 incidental 5mm marker spheres are collidable). This module sidesteps
all of that: no QP, no rate-limit tuning, no reliance on GMR's collision geometry --
a direct damped-least-squares nudge on the limb's own joint chain, using OUR OWN
vetted mesh/cylinder geometry (the same one every eval metric in this project uses),
with a small bounded iteration to converge past the linearization error of a single
step. Exact and deterministic, not asymptotic.

Usage (both call sites import this, never re-derive the chain/DLS logic):
    from leg_floor_clamp import EFF_BODY, build_chain_dofs, clamp_limb
    chain = build_chain_dofs(model, "left_foot")
    clamp_limb(model, data, mesh_cache, "left_foot", chain, floor_margin=0.0)
"""
from __future__ import annotations

import sys
import pathlib

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from post_process_ground_contactfirst import _geom_lowest_z  # noqa: E402  (kept for parity/reference)
from solve_global_trajectory_opt_contactfirst import (  # noqa: E402
    _within_k_hops, COLL_MARGIN, COLL_HOPS)

FLOOR_GEOM_NAME = "g1_floor_geom"  # matches g1_model_setup.py -- see clamp_limb's
# avoid_self_collision docstring for why this is looked up by name, not threaded
# as a new required parameter through every existing caller.

# Matches gmr_contact_retarget.py's EFF_BODY exactly -- same effector keys project-wide.
# "waist" (S7-T6 probe only, NOT in CLAMP_TARGETS / not shipped by default): torso_link
# is the waist chain's terminal effector. pelvis is NOT reachable via this chain --
# confirmed via body_parentid, pelvis attaches directly to world (it's the free-joint
# root body here), waist_yaw_joint's parent IS pelvis. So a "waist" correction can only
# ever move torso_link (and everything above it); pelvis floor penetration is a root-
# level residual out of scope by design (root lift is perframe's job, don't duplicate).
EFF_BODY = {
    "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
    "left_hand": "left_wrist_yaw_link", "right_hand": "right_wrist_yaw_link",
    "waist": "torso_link",
}

# Unitree G1 joint names, hip-to-effector order, one hinge DOF each (no free joints
# in the chain -- confirmed via GMR's own robot_motor_names printout).
CHAIN_JOINTS = {
    "left_foot": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "right_foot": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                   "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
    "left_hand": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
                  "left_wrist_yaw_joint"],
    "right_hand": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                   "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
                   "right_wrist_yaw_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
}


# (eff-chain, watch_body) pairs covering the whole lower/upper body's floor
# clearance -- S6-A4 found the worst whole-body penetration is often NOT a
# foot/hand at all (elbow during fast arm-swing, hip_yaw during floor-contact
# clips); watching only EFF_BODY's terminal effector misses both. Every watched
# body's Jacobian is exact via its own chain regardless of whether that chain's
# terminal effector is a contact target elsewhere (S5's --effectors scope).
# Shared by Phase A (gmr_contact_retarget.py --floor-clamp, inline) and Phase B
# (polish_median_limbwise.py, post-hoc) -- do not duplicate this list.
#
# ORDER MATTERS: proximal-to-distal (hip -> knee -> ankle, shoulder -> elbow).
# Correcting an upstream body moves every downstream body on the same chain too
# (shared DOFs), so correcting distal-first (as an earlier version of this list
# did) lets a later proximal correction silently re-violate an already-fixed
# distal body -- confirmed empirically in S6-A4 (walk3_subject1's worst
# violation after clamping was STILL the ankle, because hip_yaw ran after it on
# the same chain). Proximal-first means each correction only affects itself and
# bodies below it, never undoes an earlier one.
CLAMP_TARGETS = [
    ("left_foot", "left_hip_yaw_link"), ("left_foot", "left_knee_link"),
    ("left_foot", "left_ankle_roll_link"),
    ("right_foot", "right_hip_yaw_link"), ("right_foot", "right_knee_link"),
    ("right_foot", "right_ankle_roll_link"),
    ("left_hand", "left_elbow_link"), ("left_hand", "left_wrist_yaw_link"),
    ("right_hand", "right_elbow_link"), ("right_hand", "right_wrist_yaw_link"),
]


class CorrectionRateLimiter:
    """S8-T1b (opt-in, attempt 1): temporal trust region on the APPLIED per-frame
    clamp correction. S8-T0b measured >80% of velocity spikes in both
    gmr_contact_fc and perframelimb as cause A -- the clamp's per-frame
    independent DLS correction either toggling on/off between adjacent frames or
    GROWING rapidly across consecutive frames near a singular configuration
    (obstacles6_subject5 t=2313-2316: correction -0.76 -> -17.4 -> -63.1 -> -13.9
    rad/s over 4 frames; the recurring 94.2 rad/s spikes are pi rad/frame --
    joint-RANGE-scale solution-branch flips, classic DLS singularity chatter).
    A per-iteration `max_dq` cap (S7-T3) does not fix this: 0.15 rad/iter x 10
    iters still allows 1.5 rad of correction change in one frame.

    Mechanism: maintain the previously APPLIED correction vector c_prev (per
    joint, rad). Each frame, the desired correction c_des = q_corrected - q_ref
    (q_ref = the pre-clamp pose: GMR's own solve for Phase A, the centered raw
    pose for perframelimb). Apply c_app = c_prev + clip(c_des - c_prev, -rate,
    +rate) -- the correction can only change by `rate` rad/frame per joint, so
    its contribution to joint velocity is bounded by rate*fps rad/s (0.15
    rad/frame @ 30fps = +4.5 rad/s worst case, vs the pi-rad flips it replaces).
    First frame applies c_des in full (no previous frame -> no velocity cost).
    Decay back to zero when the clamp releases is bounded by the same rate
    (symmetric -- a fast release would just re-introduce the velocity spike on
    the other side).

    OPT-IN by construction (a new object per clip, only instantiated when the
    driver's --clamp-rate-limit flag is passed): no existing variant's output
    changes unless the flag is used, per S7-T3's lesson that clamp-path
    defaults must never silently change shipped baselines."""

    def __init__(self, rate):
        self.rate = float(rate)
        self.prev = None

    def apply(self, q_ref, q_corrected):
        """q_ref/q_corrected: joint-only qpos slices (qpos[7:], 29 joints).
        Returns the rate-limited joint vector; caller clips to joint ranges."""
        c_des = np.asarray(q_corrected) - np.asarray(q_ref)
        if self.prev is None:
            self.prev = c_des.copy()
            return np.asarray(q_corrected).copy()
        c_app = self.prev + np.clip(c_des - self.prev, -self.rate, self.rate)
        self.prev = c_app
        return np.asarray(q_ref) + c_app


def joint_ranges(model):
    """(q_lo, q_hi) for the 29 actuated joints in qpos[7:] order (hinge joints
    1..njnt-1; joint 0 is the free root). Used by rate-limit callers to clip
    the limited pose back into joint range."""
    lo = model.jnt_range[1:, 0].copy()
    hi = model.jnt_range[1:, 1].copy()
    return lo, hi


def build_chain_dofs(model, eff):
    """qpos/dof addresses + joint ranges for one limb's chain, in CHAIN_JOINTS
    order. Compute once per model (not per frame) and pass the result into
    `clamp_limb` every call."""
    qadr, dadr, lo, hi = [], [], [], []
    for jname in CHAIN_JOINTS[eff]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"joint not found: {jname}"
        qadr.append(int(model.jnt_qposadr[jid]))
        dadr.append(int(model.jnt_dofadr[jid]))
        rng = model.jnt_range[jid]
        lo.append(float(rng[0]))
        hi.append(float(rng[1]))
    return (np.array(qadr), np.array(dadr), np.array(lo), np.array(hi))


def _lowest_point(model, data, mesh_cache, body_id):
    """(xyz world point, z) of a body's own lowest-point geom. Mirrors
    `_geom_lowest_z`'s per-geom-type branches but keeps the full 3D point (exact
    for mesh; approximate X/Y -- geom-center-projected -- for sphere/capsule/
    cylinder/box). The approximation only affects the DLS Jacobian's contact
    point, not correctness of the Z bound; `clamp_limb`'s bounded iteration
    self-corrects across steps regardless."""
    best_z = np.inf
    best_pt = None
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body_id:
            continue
        gtype = int(model.geom_type[g])
        pos = data.geom_xpos[g]
        mat = data.geom_xmat[g].reshape(3, 3)
        sz = model.geom_size[g]

        if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
            verts = mesh_cache.get(g)
            if verts is None:
                z, pt = float(pos[2]), pos.copy()
            else:
                world = pos + verts @ mat.T
                idx = int(np.argmin(world[:, 2]))
                z, pt = float(world[idx, 2]), world[idx].copy()
        elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            pt = pos.copy()
            pt[2] = float(pos[2] - sz[0])
            z = pt[2]
        elif gtype in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            radius, half_len = float(sz[0]), float(sz[1])
            axis = mat[:, 2]
            end1 = pos + axis * half_len
            end2 = pos - axis * half_len
            base = (end1 if end1[2] < end2[2] else end2).copy()
            base[2] -= radius
            pt, z = base, float(base[2])
        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            hx, hy, hz = float(sz[0]), float(sz[1]), float(sz[2])
            z = float(pos[2]) - abs(mat[2, 0]) * hx - abs(mat[2, 1]) * hy - abs(mat[2, 2]) * hz
            pt = pos.copy()
            pt[2] = z
        else:
            z, pt = float(pos[2]), pos.copy()

        if z < best_z:
            best_z, best_pt = z, pt
    return best_pt, best_z


def clamp_limb(model, data, mesh_cache, eff, chain_dofs, floor_margin=0.0,
                target_xy=None, max_iters=10, damping=1e-3, watch_body=None,
                max_dq=None, avoid_self_collision=False, coll_weight=0.5,
                collision_only=False, q_prev_chain=None, posture_weight=1.0,
                limit_margin=0.0, limit_weight=0.0):
    """Exact per-frame floor clamp for one limb, mutates `data.qpos` in place.

    Clearance-only mode (target_xy=None -- S6-A's default, also used for swing
    feet in S6-B): if the watched body's lowest point sits below `floor_margin`,
    nudge the limb's own joint chain (DLS, Z-only constraint row) until it
    doesn't. Bounded `max_iters` re-linearization steps (not a rate-limited QP)
    to converge past single-step linearization error -- typically 1 iteration is
    enough for corrections under ~2cm, but S6-A4 found genuine deep-crouch frames
    (hip_pitch/knee both saturated at their joint-limit boundary) need more:
    each DLS step near an active joint-limit clamp makes smaller progress since
    the remaining free DOFs carry more of the correction. Default raised
    3->10 after confirming on walk3_subject1 frame 6526 (hip_pitch/knee both at
    their exact range boundary): 3 iters left -3.48cm residual, 20 iters reached
    -0.21cm. Cheap either way (small chain, closed-form DLS, not a QP).

    Held mode (target_xy given -- S6-B's held-effector reuse): also drives XY to
    `target_xy` via the same chain (X,Y,Z rows). Orientation is NOT corrected
    here (S5's contact-layer cost already handles held-effector orientation
    upstream in Phase A's case; Phase B's polish is position-only by design,
    scope decision, not an oversight).

    `watch_body` (S6-A4 extension -- see planLogGMR.md): override which body's
    lowest point gets checked/corrected, instead of `EFF_BODY[eff]` (the chain's
    own terminal effector). Any body ON the chain works and is mathematically
    exact -- e.g. `eff="left_foot"` with `watch_body="left_knee_link"` corrects
    the KNEE's own violation via the SAME leg chain; the Jacobian at a point on
    an upstream link automatically has zero columns for downstream-only DOFs
    (ankle joints don't move the knee), so this is not an approximation. Found
    necessary in S6-A4: on walk3_subject1 the worst whole-body penetration is
    `left_elbow_link` (arm swinging low during fast walking), on ground1 it's
    `left_hip_yaw_link` -- feet/hands-only watching misses both entirely.

    `max_dq` (S7-T3 fix, OPT-IN, default None=uncapped): per-iteration
    element-wise cap on `dq` (rad). Root cause (walk1_subject1 t~5001-5006,
    `polish_median_limbwise.py --center perframe`, held-mode, right leg): full
    leg extension drives the knee to its exact lower joint limit (-0.0873 rad
    on G1) -- a near-singular configuration for this 6-DOF chain's position
    IK. With `damping=1e-3`, a residual error of ~1cm (never stale, never
    large -- confirmed via direct instrumentation, ruling out an
    onset-target-staleness theory) produced `dq` large enough to snap the
    effector body up to world Z=0.80m within the same frame's 10-iteration
    loop, then chaotically oscillate frame to frame as each fresh per-frame
    solve re-entered the same singular basin from a slightly different
    starting pose. Capping `dq` at 0.15 rad/iteration converges the same
    frames cleanly (post-fix knee angle stabilizes ~0.06-0.14 rad, never
    pinned at the limit; whole-clip max frame-to-frame body jump matches
    `gmr_raw`'s own natural swing-foot motion exactly, 16.59cm, confirming no
    residual pathology).

    MUST stay opt-in (default None), NOT a global default: tested and
    REJECTED as a default change -- Phase A (`gmr_contact_retarget.py
    --floor-clamp`, inline per-frame call, warm-started across frames by
    GMR's own solve) legitimately needs large single-iteration corrections on
    deep-crouch frames (ground1_subject1: capping by default at 0.15
    regressed joint_ok 95.3%->89.8%, floorPen 4.74->9.36cm -- truncating a
    genuinely-needed big correction, not fixing a singularity). Phase B's
    `--center median` path also never triggers the bug (verified: the
    constant Z-shift moves the trajectory off the singular knee-limit basin
    that `--center perframe`'s frame-varying, near-zero-in-this-window shift
    lands on). Only `_perframe_shift`'s call path passes `max_dq` explicitly;
    every other caller (Phase A, Phase B median) is untouched and stays
    byte-identical to pre-S7 output -- verified via direct pkl diff.

    `avoid_self_collision` (S7-T7 fix, opt-in, default False): the floor clamp
    corrects ONE limb chain's Z-clearance against the floor with zero
    awareness of every other body -- on cramped floor poses (crawl, prone) it
    can drive a corrected elbow/knee straight into the torso/head while
    satisfying its own floor constraint perfectly. Confirmed on
    ground1_subject1's own worst-penetration frame (visually: a hand-through-
    head penetration in the annotated render), and at full corpus scale: floor
    class self-collision incidence 6.34%->9.95% (+57% relative), peak depth
    5.66->7.50cm, ground1 specifically 2.57%->13.12% (~5x) after `--floor-
    clamp`, vs near-zero change on locomotion (3.85%->3.86%) -- confirming the
    mechanism, not the underlying motion, causes it.

    Fix: reuse the project's OWN trusted self-collision QP row-builder
    (`stage_b`'s collision rows in `solve_global_trajectory_opt_contactfirst.py`)
    verbatim in math -- contact normal + the RELATIVE Jacobian `j1 - j2` (both
    bodies' Jacobians at the contact point, sliced to this chain's own DOFs),
    `COLL_MARGIN`/`COLL_HOPS` exclusions (floor pairs via `FLOOR_GEOM_NAME`,
    anatomically-adjacent pairs like wrist-elbow that are SUPPOSED to be
    close) -- ported as a SEQUENTIAL second phase, run after floor/held
    convergence, NOT mixed into the same per-iteration weighted solve.

    Mixing was tried FIRST and REJECTED: stacking collision rows alongside
    floor/held rows in every DLS iteration (both weighted into one solve, the
    literal form of "relative terms compete with absolute tracking") let
    collision-avoidance destabilize floor convergence on ground1_subject1's
    full clip -- floorPen got WORSE than doing nothing at all (4.74cm->38.76cm
    at coll_weight=2.0), because Phase A is inline and warm-started across
    frames: this module's small 10-iteration-per-frame Gauss-Newton loop has
    none of stage_b's whole-trajectory QP's convergence guarantees for
    jointly conflicting objectives, so one badly-converged frame cascades
    into every later frame's warm start. Sequential phases avoid this: phase
    1 (floor/held, unchanged math, byte-identical to pre-S7-T7) always
    converges exactly as before; phase 2 (self-collision only, bounded
    `max_iters` more steps) runs after, using the SAME chain DOFs but never
    competing with phase 1 inside one linear solve. A phase-3 "floor mop-up"
    after phase 2 was also tried and also rejected (made ground1 WORSE, not
    better: 4.74cm->41.41cm) for the identical cascading-warm-start reason --
    ship phase 1 + phase 2 only, nothing after phase 2.

    `coll_weight` tuning (dev-clip sweep, 5 clips): 2.0 and 1.0 both cause a
    clip-dependent cascading floor-pen regression (1.0: fine on walk1/ground1,
    catastrophic on fallAndGetUp1, joint_ok 97.1%->66.7%). **0.5 is the
    shipped default** -- self-collision resolved to ~0% on every dev clip
    (coll_pct 13.12%->0.00% on ground1, 6.16%->0.02% on fallAndGetUp1) with
    small joint_ok cost everywhere (-0.0 to -1.6pp) and moderate floorPen cost
    on the floor-class clips (ground1 4.74->9.42cm, fallAndGetUp1
    5.70->7.77cm) -- an honest trade, not free. **Known residual (not
    resolved, logged not hidden)**: at coll_weight=0.5, fallAndGetUp1's range
    metric still spikes (6.96->39.15cm) from ONE held-right-foot frame
    (t=2251, support_z=+31.42cm) where phase 2's collision correction, run on
    the SAME chain a held-target lock depends on, disrupts that lock badly at
    that specific frame -- phase 2 has zero awareness of held targets, only
    of collision. Flagged as an open item for whoever picks this up next, not
    silently absorbed into the shipped numbers.

    `q_prev_chain` / `posture_weight` (dev probe, opt-in): null-space posture-
    continuity bias on phase 1. Root cause this targets: phase 1's DLS is a
    per-frame INDEPENDENT solve (no memory of any other frame) against a
    redundant chain (6 chain DOFs vs. a 1-row clearance-only or 3-row held
    task) -- the minimum-norm solution is a function of THIS frame's own
    starting pose only, so two frames with a near-identical target can still
    land on two different null-space solutions (observed: `sprint1_subject4`
    t~6296-6306, right_ankle_pitch flipping between its own hard joint limit
    0.5236 rad and ~-0.5 to -0.8 rad frame to frame while GMR's raw target is
    flat -- a solver branch-flip, not a real correction). Fix: bias `dq`
    toward `q_prev_chain` (caller-supplied, typically the previous frame's
    OWN post-clamp chain posture) via null-space projection --
    `dq = dq_bias + J^+ (e - J @ dq_bias)`, `dq_bias = posture_weight *
    (q_prev_chain - current chain qpos)` -- so the primary floor/held task is
    still solved exactly (up to the usual DLS damping), the previous frame's
    posture only fills in the leftover null-space freedom that would
    otherwise flip arbitrarily. Default `q_prev_chain=None` is a byte-
    identical no-op (dq_bias=0 recovers the original `J^+ e` line exactly).

    `limit_margin` / `limit_weight` (S9-T1, opt-in): a SECOND null-space bias,
    combined additively with the posture-continuity bias above before
    projection, that repels each chain DOF from its own hard joint limit
    (linear ramp starting `limit_margin` rad from `lo`/`hi`, zero outside that
    band). Targets a residual T0/S9 finding posture-continuity alone couldn't
    reach: `sprint1_subject4` t~6296-6306 `right_ankle_pitch` keeps flipping
    between its exact hard limit (0.5236 rad) and a free value even with
    posture-continuity on -- near a limit, which chain DOFs are "free" (null
    space) vs. "needed for the task" (row space) can itself flip depending on
    which side of the limit the frame started from, so a pure previous-frame
    attractor can't out-vote it; an explicit repulsive term can. Default
    `limit_weight=0.0` is a byte-identical no-op (repulsion term is zero).

    `collision_only` (S8-T1b attempt 2, opt-in): skip phase 1 entirely and run
    ONLY the phase-2 self-collision step (requires avoid_self_collision=True).
    Exists so a rate-limiting caller can bound the phase-1 floor/held
    correction (the measured source of >80% of velocity spikes, S8-T0b cause
    A) while leaving self-collision resolution un-limited (cause B was 1%/0%
    of spikes) -- attempt 1 rate-limited BOTH and thereby disabled the S7-T7
    self-collision fix (obstacles6_subject5 coll_pct 0.344%->5.946%, back at
    gmr_raw's inherited level). No existing caller passes this; default False
    is byte-identical to pre-S8 behavior.

    Returns True if any correction was applied.
    """
    if collision_only:
        assert avoid_self_collision, "collision_only requires avoid_self_collision"
    qadr, dadr, lo, hi = chain_dofs
    body_name = watch_body if watch_body is not None else EFF_BODY[eff]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    applied = [False]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    def _converge_primary(iters):
        """Floor/held convergence -- UNCHANGED math from pre-S7-T7 (byte-
        identical when avoid_self_collision=False). Self-collision is
        deliberately NOT mixed into this loop's per-iteration weighted solve
        (tried first, REJECTED: on ground1_subject1's full clip, mixing floor
        and collision rows into the same small-chain DLS every iteration let
        collision-avoidance dominate/destabilize floor convergence -- floorPen
        got WORSE than doing nothing at all, 4.74cm->38.76cm, because Phase A
        is inline/warm-started across frames, so one badly-converged frame
        cascades into every subsequent frame's warm start. This module's
        small 10-iteration Gauss-Newton loop has none of stage_b's whole-
        trajectory QP's convergence guarantees for jointly conflicting
        objectives -- unlike that QP, sequential phases are safe here. Called
        ONCE, as phase 1 -- a phase-3 "floor mop-up" (re-calling this after
        phase 2) was tried and rejected, see the caller's comment below."""
        for _ in range(iters):
            mujoco.mj_forward(model, data)
            pt, z = _lowest_point(model, data, mesh_cache, body_id)

            if target_xy is None:
                if z >= floor_margin:
                    return
                mujoco.mj_jac(model, data, jacp, jacr, pt, body_id)
                J = jacp[2:3, dadr]
                e = np.array([floor_margin - z])
            else:
                cur = data.xpos[body_id]
                # Held mode pins Z to floor_margin EXACTLY (both directions),
                # unlike clearance-only mode above which only corrects
                # downward violations. A held effector that's floating above
                # floor_margin needs pulling DOWN too -- an earlier version
                # only fired when z < floor_margin, leaving held-target Z
                # completely uncorrected whenever GMR's raw output floated
                # (which it usually does), producing a systematic +4cm bias
                # on walk1_subject1 (every held frame floating, none
                # penetrating, so the one-sided condition never triggered).
                need_z = floor_margin - z
                e = np.array([target_xy[0] - cur[0], target_xy[1] - cur[1], need_z])
                if np.all(np.abs(e) < 1e-5):
                    return
                # X,Y error is body-ORIGIN-based (target_xy comes from an
                # onset xpos[:2] lock elsewhere), so its Jacobian must be
                # queried at `cur` (body origin). Z error is LOWEST-MESH-
                # POINT-based (`z`, from _lowest_point), so its Jacobian
                # must be queried at `pt`, a DIFFERENT world point for a
                # rotated/offset foot. Using `cur`'s Jacobian for the Z row
                # (an earlier bug) described how the ORIGIN's Z moves, not
                # the actual lowest point's Z -- for a body whose lowest
                # point isn't directly below its origin, that Jacobian can
                # have the wrong sign/magnitude for the Z correction
                # entirely, causing the DLS solve to diverge instead of
                # converge (confirmed: a 1mm target on walk1_subject1 blew
                # up into a 28 degree knee correction before this fix).
                jacp_z = np.zeros((3, model.nv))
                jacr_z = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacp, jacr, cur, body_id)
                mujoco.mj_jac(model, data, jacp_z, jacr_z, pt, body_id)
                J = np.vstack([jacp[:2, dadr], jacp_z[2:3, dadr]])

            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq_bias = None
            if q_prev_chain is not None:
                dq_bias = posture_weight * (q_prev_chain - data.qpos[qadr])
            if limit_weight > 0.0:
                assert limit_margin > 0.0, "limit_weight>0 requires limit_margin>0"
                cur_q = data.qpos[qadr]
                dist_hi = hi - cur_q  # >=0 normally
                dist_lo = cur_q - lo  # >=0 normally
                # linear ramp: 0 outside the margin band, -1 (push toward lo)
                # right at hi, +1 (push toward lo... i.e. away from hi) scaled
                # by limit_weight*limit_margin (rad) at the boundary itself.
                push_from_hi = -np.clip((limit_margin - dist_hi) / limit_margin, 0.0, 1.0)
                push_from_lo = np.clip((limit_margin - dist_lo) / limit_margin, 0.0, 1.0)
                limit_bias = limit_weight * limit_margin * (push_from_hi + push_from_lo)
                dq_bias = limit_bias if dq_bias is None else dq_bias + limit_bias
            if dq_bias is not None:
                dq = dq_bias + J.T @ np.linalg.solve(JJt, e - J @ dq_bias)
            else:
                dq = J.T @ np.linalg.solve(JJt, e)
            if max_dq is not None:
                dq = np.clip(dq, -max_dq, max_dq)
            data.qpos[qadr] += dq
            data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)
            applied[0] = True

    # Phase 1: initial floor/held convergence (skipped in collision_only mode,
    # S8-T1b attempt 2 -- see docstring).
    if not collision_only:
        _converge_primary(max_iters)

    # Phase 2 (S7-T7, opt-in): self-collision-only correction, run AFTER
    # floor/held has converged (or exhausted max_iters) -- sequential, not
    # mixed, so it can never destabilize phase 1's proven convergence
    # behavior. Bounded to max_iters more steps; each step nudges away from
    # every active violating pair simultaneously (one stacked DLS solve per
    # iteration, not one row at a time) using the SAME contact-normal +
    # relative-Jacobian math as the project's own trusted stage_b QP rows.
    if avoid_self_collision:
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
        for _ in range(max_iters):
            mujoco.mj_forward(model, data)
            coll_J, coll_e = [], []
            for cc in range(data.ncon):
                ct = data.contact[cc]
                if ct.geom1 == floor_gid or ct.geom2 == floor_gid:
                    continue
                b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
                if b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS):
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
                jsep = (normal @ (j1 - j2))[dadr]
                if np.linalg.norm(jsep) < 1e-9:
                    continue
                coll_J.append(jsep)
                coll_e.append(min(pen, 0.05))
            if not coll_J:
                break
            Jc = np.array(coll_J)
            ec = np.array(coll_e)
            JJt = Jc @ Jc.T + (damping ** 2) * np.eye(Jc.shape[0])
            dq = Jc.T @ np.linalg.solve(JJt, ec) * coll_weight
            if max_dq is not None:
                dq = np.clip(dq, -max_dq, max_dq)
            data.qpos[qadr] += dq
            data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)
            applied[0] = True

        # A phase-3 "floor mop-up" (re-run _converge_primary once more here)
        # was tried and REJECTED: on ground1_subject1's full clip it made
        # things WORSE, not better (floorPen 4.74cm->41.41cm, coll% even rose
        # back to 1.33% from 0.00%) -- the mop-up's floor-only correction has
        # zero collision awareness (same blind spot as the original bug) and
        # its extra aggressive per-frame correction cascades through Phase
        # A's warm-starting just like the mixed-rows attempt did. The clean
        # 2-phase result (floorPen UNCHANGED at 4.74cm, coll% 13.12%->0.00%,
        # joint_ok 95.3%->94.9%) is better than any attempted refinement of
        # it -- ship phase 1 + phase 2 only.

    mujoco.mj_forward(model, data)
    return applied[0]

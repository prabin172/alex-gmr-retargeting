#!/usr/bin/env python3
"""Contact-first QP IK (Alex V2).

Forked from solve_fbx_canonical_alex_posori_qp_fresh_worlddelta.py. Same
damped Gauss-Newton stacked-least-squares core, but with a contact-first
priority on the end effectors:

  - Contact is detected from the *human* canonical data (marker height above
    the clip floor + low speed), per frame, for each foot and hand.
  - During foot contact: the foot is forced flat (foot up-axis -> world +Z),
    overriding the human world-delta foot orientation.
  - During hand contact: the palm/finger-front face is forced down
    (gripper +X palm-normal -> world -Z), overriding the human world-delta
    hand orientation. This is the closed-fist support surface (NOT knuckles).

Intermediate-segment orientation (upper arm, forearm, shin) is deliberately
untracked here (as in the baseline) so the limb is free to find whatever bend
the contact demands. Heading/yaw at the contact is left free (axis-alignment,
not full-orientation lock) so position tracking and the human heading still
determine it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from contact_labels import (
    CONTACT_EFFECTORS,
    detect_contacts_from_human,
    debounce_flags,
    ramp_envelope,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"

FLOOR_BODY_NAME = "floor_collider"
FLOOR_GEOM_NAME = "floor_collider_geom"


def _load_model_with_floor(model_path):
    """Load the robot MJCF and inject a floor PLANE geom as a mocap body — in
    memory only, never written back to the hand-maintained asset XML.

    Mocap (not a normal welded/static body): a static child of worldbody has its
    world position baked in at compile time (mj_forward does NOT re-derive it
    from a post-compile `model.geom_pos` mutation). A mocap body's position IS
    re-applied every mj_forward via `data.mocap_pos`, and — unlike a free joint —
    adds zero DOFs, so `nv`/joint-limit indexing is unaffected.

    No contype/conaffinity wiring needed: the asset XML sets none anywhere, so
    MuJoCo defaults (1/1) apply — the floor geom collides with everything
    already. Mirrors `_load_model_with_floor` in
    `solve_global_trajectory_opt_contactfirst.py` (Stage 4) — same technique,
    duplicated rather than shared since these are independent CLI scripts.

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

# Position-only mapping: canonical human role -> real Alex body.
# No added palm/sole/contact sites.
ROLE_TO_ALEX_BODY = {
    "pelvis": "PELVIS_LINK",
    "torso": "TORSO_LINK",
    "head": "HEAD_LINK",

    "left_hip": "LEFT_THIGH",
    "right_hip": "RIGHT_THIGH",

    "left_shoulder": "LEFT_SHOULDER_Z_LINK",
    "right_shoulder": "RIGHT_SHOULDER_Z_LINK",

    "left_knee": "LEFT_SHIN",
    "left_ankle": "LEFT_ANKLE_Y_LINK",
    "right_knee": "RIGHT_SHIN",
    "right_ankle": "RIGHT_ANKLE_Y_LINK",

    "left_elbow": "LEFT_ELBOW_Y_LINK",
    "left_wrist": "LEFT_WRIST_X_LINK",
    "right_elbow": "RIGHT_ELBOW_Y_LINK",
    "right_wrist": "RIGHT_WRIST_X_LINK",
}

TARGET_WEIGHTS = {
    "pelvis": 4.0,
    "torso": 2.0,
    "head": 2.0,

    "left_hip": 0.8,
    "right_hip": 0.8,

    "left_shoulder": 0.8,
    "right_shoulder": 0.8,

    "left_knee": 1.0,
    "left_ankle": 1.5,
    "right_knee": 1.0,
    "right_ankle": 1.5,

    "left_elbow": 1.0,
    "left_wrist": 1.5,
    "right_elbow": 1.0,
    "right_wrist": 1.5,
}


# Orientation-delta tracking.
# These are real Alex bodies, not added sites.
ORI_TO_ALEX_BODY = {
    "pelvis": "PELVIS_LINK",
    "torso": "TORSO_LINK",
    "head": "HEAD_LINK",
    "left_foot": "LEFT_FOOT",
    "right_foot": "RIGHT_FOOT",
    "left_hand": "LEFT_GRIPPER_Z_LINK",
    "right_hand": "RIGHT_GRIPPER_Z_LINK",
}

ORI_WEIGHTS = {
    "pelvis": 0.5,
    "torso": 0.25,
    "head": 0.20,
    "left_foot": 0.70,
    "right_foot": 0.70,
    "left_hand": 0.40,
    "right_hand": 0.40,
}


# ---------------------------------------------------------------------------
# Contact-first configuration
# ---------------------------------------------------------------------------
# CONTACT_EFFECTORS (imported above from contact_labels.py): `markers` are
# canonical human roles used to detect contact; `body` is the Alex body whose
# axis we align; `axis_local` is that body's local axis to drive; `world_dir` is
# where it must point during contact; `ori_role` is the world-delta orientation
# term to suppress while in contact (the contact constraint replaces it).

# Weight of the contact axis-alignment term (replaces the human ori term while
# in contact). Feet: high (foot-flat is the goal and the gate ensures it is
# reachable). Hands: low/best-effort — the palm-down face is achieved when the
# arm can reach it (lying) and yields gracefully to the natural reachable
# support face during the dynamic push, instead of fighting kinematics. The
# fist *position* pin (below) is what actually establishes the support.
CONTACT_ALIGN_WEIGHT = {
    "left_foot": 3.0,
    "right_foot": 3.0,
    "left_hand": 0.8,
    "right_hand": 0.8,
}

# Palm/fist position pin: while a hand is in contact, pin its palm contact site
# to the human hand contact location (morphology-delta target, same machinery as
# the body position targets). This is the substantive "fist support" term — the
# solver otherwise only tracks the wrist body, leaving the fist a few cm off the
# ground.
CONTACT_POS = {
    "left_hand": dict(site="alex_left_palm_contact_site", marker="left_hand_middle",
                      skip_pos_role="left_wrist"),
    "right_hand": dict(site="alex_right_palm_contact_site", marker="right_hand_middle",
                       skip_pos_role="right_wrist"),
}
CONTACT_POS_WEIGHT = 3.0

# Foot effector -> the position-target role that actually places it (the ankle
# landmark; there is no separate "foot" position role). Used by the planted-foot
# position hold to freeze the right target.
FOOT_POS_ROLE = {"left_foot": "left_ankle", "right_foot": "right_ankle"}

# Foot effector -> the knee position role above it (shank-tilt clamp).
FOOT_KNEE_ROLE = {"left_foot": "left_knee", "right_foot": "right_knee"}

# Leg-chain joints (hip -> ankle), used by refine_leg_floor_transitions to build
# a per-DOF posture_reg vector that locks everything except one leg (mirrors
# ARM_CHAIN_JOINTS below for refine_arm_floor_transitions). Names match
# refine_limbs_contactfirst.py's LIMB_CHAINS for consistency across scripts.
LEG_CHAIN_JOINTS = {
    "left_foot": ["LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y",
                  "LEFT_ANKLE_Y", "LEFT_ANKLE_X"],
    "right_foot": ["RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y",
                   "RIGHT_ANKLE_Y", "RIGHT_ANKLE_X"],
}

# Arm-chain joints (shoulder -> gripper), used ONLY by the floor-transition arm
# refinement pass (refine_arm_floor_transitions) to build a per-DOF posture_reg
# vector that locks everything except one arm during a local re-solve.
ARM_CHAIN_JOINTS = {
    "left_hand": ["LEFT_SHOULDER_Y", "LEFT_SHOULDER_X", "LEFT_SHOULDER_Z",
                  "LEFT_ELBOW_Y", "LEFT_WRIST_Z", "LEFT_WRIST_X", "LEFT_GRIPPER_Z"],
    "right_hand": ["RIGHT_SHOULDER_Y", "RIGHT_SHOULDER_X", "RIGHT_SHOULDER_Z",
                   "RIGHT_ELBOW_Y", "RIGHT_WRIST_Z", "RIGHT_WRIST_X", "RIGHT_GRIPPER_Z"],
}

# Arm-chain BODIES (wrist + gripper links) checked for floor engagement onset —
# the geometry that can independently touch the floor before the hand-contact
# detector (which is human-marker-based, not geometric) declares contact.
ARM_CHAIN_FLOOR_BODIES = {
    "left_hand": ["LEFT_WRIST_Z_LINK", "LEFT_WRIST_X_LINK", "LEFT_GRIPPER_Z_LINK"],
    "right_hand": ["RIGHT_WRIST_Z_LINK", "RIGHT_WRIST_X_LINK", "RIGHT_GRIPPER_Z_LINK"],
}

# Sole corner sites per foot — mirrors solve_global_trajectory_opt_contactfirst.py's
# SOLE_CORNER_SITES exactly (same model, same sites). Used only to measure the
# ankle-body-to-sole-plane clearance for the target-space floor estimate (see
# `main()`'s floor-placement block) — this script does not add on-floor QP rows.
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



def floor_phase_weight(z_signal, planted_any, lo_pct=5, hi_pct=95):
    """Per-frame [0,1] weight gating floor-collision strength by posture phase.

    A single clip-wide floor_z (this file's own target-space estimate, or
    Stage 4's `_estimate_floor_z`) is calibrated to the STANDING stance and is
    NOT valid for a lying/supine/prone phase in the same clip — the free-
    floating root's own vertical trajectory is not phase-invariant in this
    pipeline's local frame (documented in wiki/concepts/grounding.md, "Get-up
    floor residual is BETWEEN-PHASE": a foot plants several cm lower in the
    lying phase than in the terminal standing stance, even though the true
    physical floor is one flat plane). Forcing full-strength hard floor
    collision through a lying phase misreads the legitimately-low pelvis/hip
    as a violation (measured on luigi_standSupine_08: RIGHT_HIP_X_LINK 14.4cm
    "penetration" that the QP could never resolve, since it isn't real).

    Rather than clustering explicit phase windows, use `z_signal` (pelvis
    height — target-space here, root qpos Z in Stage 4) directly: smoothstep
    from `lo_pct` (clip-wide low reference, e.g. the lying phase) to `hi_pct`
    of the standing/planted-foot frames (the terminal stance height, matching
    what floor_z itself is calibrated to). No plants at all, or a clip with no
    real phase separation (hi≈lo, e.g. a standing-only push) -> returns all-1s,
    i.e. behaves exactly like the un-gated original (identity, no regression
    risk for single-phase clips)."""
    z = np.asarray(z_signal, dtype=np.float64)
    pool = z[planted_any] if np.any(planted_any) else z
    hi = float(np.percentile(pool, hi_pct))
    lo = float(np.percentile(z, lo_pct))
    if hi - lo < 1e-6:
        return np.ones_like(z)
    frac = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    return frac * frac * (3.0 - 2.0 * frac)   # smoothstep


def mj_name(model, objtype, idx):
    out = mujoco.mj_id2name(model, objtype, idx)
    return "" if out is None else out


def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Missing body in Alex model: {name}")
    return bid


def site_id(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Missing site in Alex model: {name}")
    return sid


def joint_info(model, name):
    """(qpos address, dof address, (lo, hi) range) of a named hinge joint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Missing joint in Alex model: {name}")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), tuple(model.jnt_range[jid])


def clamp_shank_tilt(ankle_t, knee_t, fwd_xy, pitch_range, roll_range):
    """
    Project the knee position target into the shank-tilt region reachable with a
    FLAT foot, i.e. within the ankle pitch/roll joint ranges. Retargeting-side
    feasibility fix: copying the human verbatim demands near/over-limit ankle
    angles during plants, and no solver weighting can reconcile "foot flat" with
    an infeasible knee target — so the target itself is made consistent here.

    ankle_t/knee_t: world position targets; fwd_xy: planted-foot heading (unit,
    ground plane). pitch_range: (min, max) forward-lean angle of the shank;
    roll_range: (min, max) leftward-lean angle. Returns (knee_target, clamped?).
    """
    v = knee_t - ankle_t
    L = float(np.linalg.norm(v))
    z = np.array([0.0, 0.0, 1.0])
    vz = float(v @ z)
    # Knee not meaningfully above the ankle (deep kneel / data glitch): the
    # flat-foot tilt decomposition is undefined, leave the target alone.
    if L < 1e-9 or vz < 0.2 * L:
        return knee_t, False

    f = np.array([fwd_xy[0], fwd_xy[1], 0.0])
    f /= np.linalg.norm(f)
    lat = np.cross(z, f)

    pitch = np.arctan2(float(v @ f), vz)      # shank forward lean over the foot
    roll = np.arctan2(float(v @ lat), vz)     # shank leftward lean
    cp = float(np.clip(pitch, *pitch_range))
    cr = float(np.clip(roll, *roll_range))
    if cp == pitch and cr == roll:
        return knee_t, False

    u = f * np.tan(cp) + lat * np.tan(cr) + z
    u /= np.linalg.norm(u)
    return ankle_t + L * u, True


def load_canonical(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    roles = [str(x) for x in z["roles"]]
    role_to_idx = {r: i for i, r in enumerate(roles)}
    positions = np.asarray(z["positions"], dtype=np.float64)
    fps = float(z["fps"])

    if "orientation_role_names" not in z.files or "orientation_mats" not in z.files:
        raise RuntimeError(
            f"{npz_path} must contain orientation_role_names and orientation_mats. "
            "Run scripts/build_canonical_orientation_frames_fresh.py first."
        )

    orientation_roles = [str(x) for x in z["orientation_role_names"]]
    ori_to_idx = {r: i for i, r in enumerate(orientation_roles)}
    orientation_mats = np.asarray(z["orientation_mats"], dtype=np.float64)

    # Persisted contact labels from Stage 2.5 (scripts/ground_canonical_human.py,
    # phasic-v2 M1). None/None when absent (a plain _with_orient.npz that never
    # went through grounding) -- main() falls back to on-the-fly detection.
    persisted_contacts = None
    persisted_eff_names = None
    if "contact_flags" in z.files and "contact_effector_names" in z.files:
        persisted_eff_names = [str(x) for x in z["contact_effector_names"]]
        flags = np.asarray(z["contact_flags"], dtype=bool)
        persisted_contacts = {e: flags[:, i] for i, e in enumerate(persisted_eff_names)}

    return (roles, role_to_idx, positions, fps, orientation_roles, ori_to_idx, orientation_mats,
            persisted_contacts, persisted_eff_names)


def measure_alex_pelvis_to_head(model, data, role_to_body_id):
    """Query Alex's pelvis-to-head distance at the default (all-joints-zero) pose."""
    q_rest = np.zeros(model.nq)
    q_rest[3] = 1.0
    data.qpos[:] = q_rest
    mujoco.mj_forward(model, data)
    pelvis_pos = data.xpos[role_to_body_id["pelvis"]].copy()
    head_pos = data.xpos[role_to_body_id["head"]].copy()
    return float(np.linalg.norm(head_pos - pelvis_pos))


def estimate_source_scale(first_pos, role_to_idx, alex_pelvis_to_head):
    """Global scale for root translation: ratio of Alex to human pelvis-to-head height."""
    pelvis = first_pos[role_to_idx["pelvis"]]
    head = first_pos[role_to_idx["head"]]
    human_height = np.linalg.norm(head - pelvis)
    return alex_pelvis_to_head / max(human_height, 1e-6)


def make_initial_alignment_targets(first_src_pos, role_to_idx, scale):
    """
    Initial Alex rest-alignment target.

    The first human frame defines the source rest/reference pose for this clip.
    Alex is first solved into a corresponding pose. Later frames are expressed
    as deltas from this solved Alex pose.
    """
    src_pelvis0 = first_src_pos[role_to_idx["pelvis"]]

    targets = {}
    for role in ROLE_TO_ALEX_BODY:
        if role == "pelvis":
            targets[role] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            rel0 = first_src_pos[role_to_idx[role]] - src_pelvis0
            targets[role] = scale * rel0

    return targets


def compute_per_role_scales(first_src_pos, role_to_idx, target_rest_positions, clamp=(0.4, 2.5)):
    """
    Per-role scale factors from comparing source vs Alex achieved-rest proportions.

    For each non-root role, we compare the pelvis-relative distance in the human
    rest frame to the pelvis-relative distance Alex achieved after initial IK.
    This captures limb-proportion differences (e.g. shorter arms on Alex) that a
    single global scale misses.

    The global scale is not needed here; the root translation uses its own scale.
    """
    src_pelvis = first_src_pos[role_to_idx["pelvis"]]
    alex_pelvis = target_rest_positions["pelvis"]

    scales = {}
    for role in ROLE_TO_ALEX_BODY:
        if role == "pelvis":
            scales[role] = 1.0
            continue
        src_dist = float(np.linalg.norm(first_src_pos[role_to_idx[role]] - src_pelvis))
        alex_dist = float(np.linalg.norm(target_rest_positions[role] - alex_pelvis))
        if src_dist < 1e-6:
            scales[role] = 1.0
        else:
            scales[role] = float(np.clip(alex_dist / src_dist, clamp[0], clamp[1]))

    return scales


def make_targets_for_frame(src_pos, role_to_idx, first_src_pos, target_rest_positions, root_scale, role_scales):
    """
    Rest-aligned morphology-delta target with per-role scales.

    For each role:
      target(t) = achieved_alex_rest(role)
                + root_scale * pelvis displacement from source rest
                + role_scales[role] * role motion relative to pelvis from source rest
    """
    src_pelvis0 = first_src_pos[role_to_idx["pelvis"]]
    src_pelvis = src_pos[role_to_idx["pelvis"]]
    root_delta = root_scale * (src_pelvis - src_pelvis0)

    targets = {}
    for role in ROLE_TO_ALEX_BODY:
        if role == "pelvis":
            targets[role] = target_rest_positions[role] + root_delta
        else:
            rel0 = first_src_pos[role_to_idx[role]] - src_pelvis0
            rel = src_pos[role_to_idx[role]] - src_pelvis
            s = role_scales.get(role, root_scale)
            rel_delta = s * (rel - rel0)
            targets[role] = target_rest_positions[role] + root_delta + rel_delta

    return targets


def clamp_hinge_joint_limits(model, qpos):
    """
    MuJoCo does not automatically clamp qpos during our manual least-squares IK.
    Clamp scalar hinge/slide joints that have limits. Skip free root.
    """
    for j in range(model.njnt):
        if not bool(model.jnt_limited[j]):
            continue
        jtype = int(model.jnt_type[j])
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            continue

        qadr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        qpos[qadr] = np.clip(qpos[qadr], lo, hi)


def rotmat_to_rotvec(R):
    """
    Convert rotation matrix to axis-angle vector.
    Returns world-frame small rotation vector.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        return 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ], dtype=np.float64)

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2.0 * np.sin(theta))

    return theta * axis


def _expmap(rotvec):
    """Axis-angle (world-frame small rotation vector) -> rotation matrix (Rodrigues)."""
    th = float(np.linalg.norm(rotvec))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rotvec, dtype=np.float64) / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def cap_foot_pitch(R, max_toe_down_rad, lift_frac):
    """Rotate a foot orientation target R so its forward (+X local) axis points no
    more than `max_toe_down_rad` below horizontal. `lift_frac` in [0,1] ramps the
    correction (1 = full cap, 0 = untouched); used to fade the cap in over the swing
    phase via the contact envelope. Only ever tilts the toe UP, never down. Yaw
    (foot heading) and orthonormality are preserved (verified). Used by the
    swing-foot toe-clearance path (--swing-clear); see the main loop for rationale."""
    R = np.asarray(R, dtype=np.float64)
    fwd = R @ np.array([1.0, 0.0, 0.0])
    pitch = float(np.arcsin(np.clip(-fwd[2], -1.0, 1.0)))   # +ve = toe-down
    if pitch <= max_toe_down_rad or lift_frac <= 0.0:
        return R
    delta = lift_frac * (pitch - max_toe_down_rad)
    a = np.array([-fwd[1], fwd[0], 0.0])   # horizontal axis perpendicular to fwd's vertical plane
    na = float(np.linalg.norm(a))
    a = a / na if na > 1e-6 else np.array([0.0, 1.0, 0.0])  # foot near-vertical: fall back to world +Y
    R1 = _expmap(a * delta) @ R
    R2 = _expmap(-a * delta) @ R
    p1 = float(np.arcsin(np.clip(-(R1 @ np.array([1.0, 0.0, 0.0]))[2], -1.0, 1.0)))
    p2 = float(np.arcsin(np.clip(-(R2 @ np.array([1.0, 0.0, 0.0]))[2], -1.0, 1.0)))
    return R1 if abs(p1 - max_toe_down_rad) < abs(p2 - max_toe_down_rad) else R2


def _swing_posture_reg(nv, leg_dofs, boost):
    """Per-DOF posture_reg for the swing-clear temporal-continuity term: the base
    1e-3 everywhere, plus `boost` added on the hip/knee leg DOFs only. `boost` is
    already ramped by the de-pitch strength, so at boost=0 this is uniform 1e-3
    (identical to the scalar default -> exact no-op off de-pitch frames)."""
    preg = np.full(nv, 1e-3, dtype=np.float64)
    if boost > 0.0:
        for d in leg_dofs:
            preg[d] += boost
    return preg


def body_xmat(data, body_id):
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)


def make_orientation_targets_for_frame(src_ori_mats, ori_to_idx, first_src_ori_mats, target_rest_orientations):
    """
    World orientation delta transfer, matching the older working solver:
      R_delta_world = R_source_current @ R_source_rest.T
      R_target = R_delta_world @ R_alex_rest

    Do NOT copy absolute human orientation into Alex.
    Transfer only the world-frame change from the source rest frame.
    """
    targets = {}
    for role in ORI_TO_ALEX_BODY:
        R0 = first_src_ori_mats[ori_to_idx[role]]
        Rt = src_ori_mats[ori_to_idx[role]]
        delta_world = Rt @ R0.T
        targets[role] = delta_world @ target_rest_orientations[role]
    return targets


def _within_k_hops(model: mujoco.MjModel, b1: int, b2: int, k: int) -> bool:
    """True if b1 is an ancestor of b2 (or vice versa) within k steps in the kinematic tree."""
    cur = b2
    for _ in range(k):
        cur = int(model.body_parentid[cur])
        if cur == b1:
            return True
        if cur == 0:
            break
    cur = b1
    for _ in range(k):
        cur = int(model.body_parentid[cur])
        if cur == b2:
            return True
        if cur == 0:
            break
    return False


def self_collision_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nv: int,
    weight: float,
    margin: float,
    gain: float,
    exclude_hops: int = 2,
    floor_gid: int | None = None,
) -> tuple[list, list]:
    """
    Build QP rows that push apart self-colliding body pairs.

    MuJoCo excludes direct parent-child contacts (1 hop) automatically.
    We also exclude bodies within `exclude_hops` of each other in the kinematic
    tree — this catches structural near-misses like HEAD↔TORSO (2 hops, always
    overlapping due to large geom radii, not a real cross-body collision).

    For each remaining contact with dist < margin, adds one row:
        sqrt(w) * n·(J1 - J2) @ dq = sqrt(w) * min(penetration, 0.05) * gain
    where n is the contact normal pushing b1 away from b2.

    Weight sweep on standup_side_04 (152 frames, lying-down + get-up):
      w=0  → 71.7% coll frames, peak 10.4 cm, track 0.073 m
      w=5  → 46.7% coll frames, peak  4.7 cm, track 0.076 m
      w=20 → 23.7% coll frames, peak  6.4 cm, track 0.075 m  ← sweet spot
      w=50 → 44.7% coll frames (regresses: solver fights itself at extreme weight)
      w=100→ 42.8% coll frames (same reason)
    w=20 gives the best collision reduction with negligible tracking regression.
    Above ~20 the QP becomes over-constrained and the solver can no longer route
    around collisions, so it oscillates in a stuck configuration.

    `floor_gid` must be passed whenever the model has an injected floor plane
    (i.e. always, once `_load_model_with_floor` is used), REGARDLESS of whether
    floor avoidance itself is enabled: the floor body's id is never 0 (its own
    mocap child of worldbody, not worldbody itself), so the old `b1==0 or
    b2==0` "skip floor/static contacts" check silently stopped catching it once
    a floor geom existed — a raw floor contact would get treated as a
    self-colliding ROBOT LINK and repelled accordingly, corrupting the solve
    (confirmed: with the floor left in and unexcluded, tracking error runs away
    to metres within ~200 frames on luigi_standProne_03).
    """
    rows: list[np.ndarray] = []
    rhs_vals: list[float] = []

    sqw = float(np.sqrt(weight))

    for c_idx in range(data.ncon):
        ct = data.contact[c_idx]
        if floor_gid is not None and (ct.geom1 == floor_gid or ct.geom2 == floor_gid):
            continue
        b1 = int(model.geom_bodyid[ct.geom1])
        b2 = int(model.geom_bodyid[ct.geom2])

        # Skip floor/static contacts
        if b1 == 0 or b2 == 0:
            continue

        # Skip bodies that are kinematically close (structural near-misses)
        if _within_k_hops(model, b1, b2, exclude_hops):
            continue

        penetration = margin - float(ct.dist)   # positive when too close or penetrating
        if penetration <= 0:
            continue

        # Contact normal — sign-correct so it pushes b1 away from b2
        normal = ct.frame[:3].copy()
        if float(np.dot(normal, data.xpos[b1] - data.xpos[b2])) < 0:
            normal = -normal

        # Jacobian at the contact point for each body
        jacp1 = np.zeros((3, nv), dtype=np.float64)
        jacp2 = np.zeros((3, nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp1, None, ct.pos, b1)
        mujoco.mj_jac(model, data, jacp2, None, ct.pos, b2)

        jac_sep = normal @ (jacp1 - jacp2)       # (nv,) separation Jacobian

        if np.linalg.norm(jac_sep) < 1e-9:
            continue

        target = min(penetration, 0.05) * gain   # cap to 5 cm/iter

        rows.append(sqw * jac_sep)
        rhs_vals.append(sqw * target)

    return rows, rhs_vals


def floor_collision_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nv: int,
    weight: float,
    floor_gid: int,
    margin: float = 0.0,
    gain: float = 5.0,
) -> tuple[list, list]:
    """Mesh-accurate robot-vs-floor avoidance — same shape as
    `self_collision_rows`, against the injected floor plane (see
    `_load_model_with_floor`) instead of another robot body. No k-hop exclusion
    (floor is never anatomically adjacent) and the floor's own Jacobian comes
    out zero (mocap body, no joints), so `jac_sep` reduces to the robot side
    alone with no special-casing.

    `margin=0.0` (unlike self-collision's 2cm) is deliberate: a genuinely
    planted foot/hand sits AT the floor by design (foot-flat/foot-hold/palm-pin
    put it there) — a nonzero margin would read that as "too close" and push it
    back up, fighting the contact terms. Only real penetration is corrected.

    This is what upstream-fixes the "hands forced into the ground during the
    push" / "foot asked to go through the ground during the step" defect: Stage
    3's per-frame IK has root-position freedom (unlike Stage 4's Stage B, which
    only touches joint angles with the root frozen from Stage A), so a
    root-level sunk pose CAN be corrected here, not just a local limb dip.

    ONE ROW PER COLLIDING BODY, deepest contact only — MuJoCo's convex-hull-vs-
    plane narrow phase returns MULTIPLE simultaneous contact points per body
    (measured: 3 per body here), unlike the sparser point contacts typical of
    self_collision_rows' link-vs-link case. Emitting a row per raw contact
    silently multiplies a body's effective weight by however many contact
    points it happens to generate (confirmed: weight=20 diverged to metres of
    tracking error within ~200 frames with the raw per-contact version; the
    same weight is stable once deduplicated to one row per body)."""
    sqw = float(np.sqrt(weight))
    deepest: dict[int, tuple] = {}   # robot body id -> (penetration, contact, b1, b2)

    for c_idx in range(data.ncon):
        ct = data.contact[c_idx]
        if ct.geom1 != floor_gid and ct.geom2 != floor_gid:
            continue
        b1 = int(model.geom_bodyid[ct.geom1])
        b2 = int(model.geom_bodyid[ct.geom2])

        penetration = margin - float(ct.dist)
        if penetration <= 0:
            continue

        robot_bid = b2 if b1 == model.geom_bodyid[floor_gid] else b1
        prev = deepest.get(robot_bid)
        if prev is None or penetration > prev[0]:
            deepest[robot_bid] = (penetration, ct, b1, b2)

    rows: list[np.ndarray] = []
    rhs_vals: list[float] = []

    for penetration, ct, b1, b2 in deepest.values():
        normal = ct.frame[:3].copy()
        if float(np.dot(normal, data.xpos[b1] - data.xpos[b2])) < 0:
            normal = -normal

        jacp1 = np.zeros((3, nv), dtype=np.float64)
        jacp2 = np.zeros((3, nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp1, None, ct.pos, b1)
        mujoco.mj_jac(model, data, jacp2, None, ct.pos, b2)

        jac_sep = normal @ (jacp1 - jacp2)

        if np.linalg.norm(jac_sep) < 1e-9:
            continue

        target = min(penetration, 0.05) * gain

        rows.append(sqw * jac_sep)
        rhs_vals.append(sqw * target)

    return rows, rhs_vals


def solve_frame_position_ik(
    model,
    data,
    role_to_body_id,
    targets,
    q_init,
    *,
    ori_role_to_body_id=None,
    ori_targets=None,
    ori_scale=1.0,
    align_constraints=None,
    skip_ori_body_ids=None,
    pos_site_constraints=None,
    skip_pos_roles=None,
    ori_weight_scale=None,
    pos_weight_scale=None,
    hold_pos_roles=None,
    hierarchical=True,
    knee_bias=None,
    iters=30,
    damping=1e-3,
    step_scale=0.7,
    max_step_norm=0.20,
    root_reg=1e-3,
    posture_reg=1e-3,
    coll_weight=0.0,
    coll_margin=0.02,
    coll_gain=5.0,
    coll_hops=2,
    floor_weight=0.0,
    floor_gid=None,
    floor_margin=0.0,
    floor_gain=5.0,
    q_ref=None,
    floor_hard=False,
    swing_clear_sites=None,
    diag_out=None,
):
    """`floor_hard` (hierarchical-v1 H2): route floor_collision_rows into the
    level-1 (hard) tier instead of level-2 (soft) -- only meaningful when
    `hierarchical=True` (rows1 is otherwise concatenated with rows2 into a
    single weighted solve, so the routing is a no-op). Deliberately does NOT
    touch hand contacts (see --hard-tier's help text / wiki/experiments/
    retired-approaches.md) -- hands stay in rows2 exactly as before, whatever
    `hierarchical`/`floor_hard` are set to.

    `diag_out` (optional mutable dict): if provided, filled in-place after
    the iteration loop with `floor_pen_cm` (deepest floor penetration
    remaining, any geom) and `hold_slip_cm` (max XY distance between an
    active hold_pos_roles target -- already the frozen anchor blend by the
    time this function is called -- and its ACHIEVED position). Adapts
    plan.md's original OSQP-slack-and-log H2 sketch to this solver's actual
    architecture (damped least-squares nullspace projection, not OSQP -- it
    never reports infeasible, so the analogous signal is "how far did the
    hard tier's own tasks land from where they were asked to be" rather
    than a slack variable)."""
    """`q_ref`: regularization target for the posture_reg term, decoupled from
    `q_init` (the IK's warm-start / iteration starting point). Defaults to
    `q_init` (original behaviour: "pull toward where you started"). Used by
    refine_arm_floor_transitions to warm-start EVERY joint from Pass 1's own
    solved pose each frame (a good starting guess) while regularizing
    differently per joint: the target arm's chain toward the previously
    REFINED frame (temporal continuity across the transition), everything
    else toward Pass 1's OWN value for that exact frame (so locked joints
    keep following their already-fine original trajectory instead of being
    frozen — freezing them for the whole window created a NEW discontinuity
    at the window's exit boundary when the freeze let go)."""
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)
    q_ref = (q_init if q_ref is None else q_ref).copy()

    nv = model.nv

    hold = hold_pos_roles or set()

    for _ in range(iters):
        rows1 = []; rhs1 = []   # level 1: contact tasks (high priority)
        rows2 = []; rhs2 = []   # level 2: body tracking + regularisers

        for role, bid in role_to_body_id.items():
            # Suppressed/faded while the matching hand is in contact: the palm
            # position pin places the hand instead of the wrist-body target.
            # pos_weight_scale cross-fades this over the contact ramp (0..1).
            if skip_pos_roles and role in skip_pos_roles:
                continue
            wscale = pos_weight_scale.get(role, 1.0) if pos_weight_scale else 1.0
            weight = TARGET_WEIGHTS.get(role, 1.0) * wscale
            if weight <= 0.0:
                continue

            current = data.xpos[bid].copy()
            err = targets[role] - current

            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)

            if role in hold:   # planted-foot position hold -> high priority
                rows1.append(np.sqrt(weight) * jacp)
                rhs1.append(np.sqrt(weight) * err)
            else:
                rows2.append(np.sqrt(weight) * jacp)
                rhs2.append(np.sqrt(weight) * err)

        if ori_role_to_body_id is not None and ori_targets is not None and ori_scale > 0.0:
            for role, bid in ori_role_to_body_id.items():
                # Suppressed/faded while this body is in contact: the contact
                # axis-alignment constraint replaces the human orientation.
                # ori_weight_scale cross-fades this over the contact ramp (0..1).
                if skip_ori_body_ids is not None and bid in skip_ori_body_ids:
                    continue
                wscale = ori_weight_scale.get(bid, 1.0) if ori_weight_scale else 1.0

                R_current = body_xmat(data, bid)
                R_target = ori_targets[role]
                R_err = R_target @ R_current.T
                err_rot = rotmat_to_rotvec(R_err)

                weight = ori_scale * ORI_WEIGHTS.get(role, 0.0) * wscale
                if weight <= 0.0:
                    continue

                jacp = np.zeros((3, nv), dtype=np.float64)
                jacr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacBody(model, data, jacp, jacr, bid)

                rows2.append(np.sqrt(weight) * jacr)
                rhs2.append(np.sqrt(weight) * err_rot)

        # Contact position pin: drive a site (palm contact patch) to a world
        # target — the fist support location during hand contact. Level 2: the
        # palm pin is best-effort by design (Alex arm shorter than human, the
        # target is often out of reach) — promoting an infeasible task to hard
        # priority starves the body tracking in its nullspace.
        if pos_site_constraints:
            for sid, target, weight in pos_site_constraints:
                cur = data.site_xpos[sid].copy()
                err = target - cur
                jacp = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacSite(model, data, jacp, None, sid)
                rows2.append(np.sqrt(weight) * jacp)
                rhs2.append(np.sqrt(weight) * err)

        # Contact axis-alignment: drive a body-local axis to a world direction
        # (foot up -> +Z; palm normal -> -Z). Cross product gives the world-frame
        # rotation vector that brings the current axis onto the target, leaving
        # spin about that axis free. `hard` routes feet to level 1 (achievable,
        # must not be compromised) and hands to level 2 (best-effort by design).
        if align_constraints:
            for bid, axis_local, world_dir, weight, hard in align_constraints:
                R_current = body_xmat(data, bid)
                a_world = R_current @ axis_local
                # Well-conditioned axis-alignment error = theta * unit_axis (cost
                # theta^2, gradient always drives theta->0). The bare cross product
                # a x d has magnitude sin(theta), whose cost sin^2 has a spurious
                # stable minimum at 180deg, so a stiff flat term can flip a foot
                # through the singularity. This form removes that.
                c = np.cross(a_world, world_dir)
                s = float(np.linalg.norm(c))
                dot = float(np.clip(a_world @ world_dir, -1.0, 1.0))
                theta = float(np.arctan2(s, dot))
                if s > 1e-6:
                    err_rot = (theta / s) * c
                elif dot >= 0.0:
                    err_rot = np.zeros(3)
                else:                                   # antipodal: any perpendicular axis
                    perp = np.cross(a_world, np.array([1.0, 0.0, 0.0]))
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.cross(a_world, np.array([0.0, 1.0, 0.0]))
                    err_rot = theta * perp / (np.linalg.norm(perp) + 1e-12)

                jacp = np.zeros((3, nv), dtype=np.float64)
                jacr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacBody(model, data, jacp, jacr, bid)

                if hard:
                    rows1.append(np.sqrt(weight) * jacr)
                    rhs1.append(np.sqrt(weight) * err_rot)
                else:
                    rows2.append(np.sqrt(weight) * jacr)
                    rhs2.append(np.sqrt(weight) * err_rot)

        # Pull actuated joints toward q_ref (start of this frame's solve).
        # Root DOFs (0-5) are left at zero — the position/orientation tasks steer them.
        # Actuated joints: dq[6:35] corresponds to qpos[7:36].
        #
        # posture_reg may be a scalar (uniform, the original behaviour) OR a
        # per-DOF array of length nv — used by the floor-transition arm
        # refinement pass (see refine_arm_floor_transitions) to LOCK unrelated
        # joints near q_ref (huge weight) while leaving one arm's chain at the
        # normal light weight, so a local re-solve can't disturb the rest of
        # the body while still being free to move the root (root DOFs 0-5
        # are unaffected either way, per the comment above).
        desired_dq = np.zeros(nv)
        desired_dq[6:] = q_ref[7:] - data.qpos[7:]
        preg = np.full(nv, posture_reg) if np.isscalar(posture_reg) else np.asarray(posture_reg)
        rows2.append(np.diag(np.sqrt(preg)))
        rhs2.append(np.sqrt(preg) * desired_dq)

        # One-sided knee-flexion bias: KNEE_Y straight (q=0) sits exactly at the
        # joint's lower limit AND the leg Jacobian singularity. Weakly push any
        # knee straighter than min_flex back to min_flex; silent once bent, so it
        # cannot over-constrain tracking — it only breaks the straight-leg lock.
        if knee_bias is not None:
            entries, min_flex, kb_weight = knee_bias
            if kb_weight > 0.0:
                for qadr, dofadr in entries:
                    cur = float(data.qpos[qadr])
                    if cur < min_flex:
                        row = np.zeros((1, nv))
                        row[0, dofadr] = np.sqrt(kb_weight)
                        rows2.append(row)
                        rhs2.append(np.array([np.sqrt(kb_weight) * (min_flex - cur)]))

        # Self-collision repulsion: push apart any non-adjacent penetrating body pairs.
        # floor_gid passed here REGARDLESS of floor_weight (the avoidance
        # toggle) — the injected floor geom must always be excluded from this
        # scan once present, or its contacts get treated as self-colliding
        # robot links (see self_collision_rows docstring).
        if coll_weight > 0.0:
            coll_rows, coll_rhs = self_collision_rows(
                model, data, nv, coll_weight, coll_margin, coll_gain,
                exclude_hops=coll_hops, floor_gid=floor_gid,
            )
            rows2.extend(coll_rows)
            rhs2.extend([np.array([v]) for v in coll_rhs])

        # Floor avoidance: unlike Stage 4's Stage B (root frozen, joints only),
        # this per-frame solve has root-position freedom, so it can correct a
        # root-level sunk pose, not just a local limb dip. See
        # floor_collision_rows docstring.
        if floor_weight > 0.0 and floor_gid is not None:
            floor_rows, floor_rhs = floor_collision_rows(
                model, data, nv, floor_weight, floor_gid, floor_margin, floor_gain,
            )
            if floor_hard:
                rows1.extend(floor_rows)
                rhs1.extend([np.array([v]) for v in floor_rhs])
            else:
                rows2.extend(floor_rows)
                rhs2.extend([np.array([v]) for v in floor_rhs])

        # Swing-foot toe-clearance (--swing-clear, soft half). One-sided Z rows:
        # any listed sole-corner site currently below its clearance target is
        # pulled UP to it, weighted per (site, clear_z, weight). Soft (level 2) and
        # re-evaluated every IK iteration against the CURRENT site height, so the
        # correction is exactly the residual dip and the solver stays near its
        # warm-start (continuous) -- unlike the hard pitch cap, this cannot force a
        # large leg reconfiguration / branch flip. Only swing (non-planted) feet
        # are ever listed (see main loop), so planted feet are untouched.
        if swing_clear_sites:
            for _sid, _clear_z, _w in swing_clear_sites:
                cur_z = float(data.site_xpos[_sid][2])
                if cur_z >= _clear_z or _w <= 0.0:
                    continue
                jacp = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacSite(model, data, jacp, None, _sid)
                sqw = float(np.sqrt(_w))
                rows2.append(sqw * jacp[2:3, :])
                rhs2.append(np.array([sqw * (_clear_z - cur_z)]))

        I_nv = np.eye(nv)

        def _dsolve(H, g):
            try:
                return np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(H, g, rcond=None)[0]

        if hierarchical and rows1:
            # Task-priority (nullspace) solve: satisfy the level-1 FOOT contact
            # tasks (foot-flat/yaw, planted-foot position hold) first, then track
            # the level-2 targets (body + best-effort hand contacts) ONLY in the
            # nullspace of level 1 — the pelvis/chain cannot drag a planted foot.
            A1 = np.vstack(rows1); b1 = np.concatenate(rhs1)
            A2 = np.vstack(rows2); b2 = np.concatenate(rhs2)
            H1 = A1.T @ A1 + damping * I_nv
            d1 = _dsolve(H1, A1.T @ b1)
            N1 = I_nv - _dsolve(H1, A1.T @ A1)      # damped nullspace projector
            M = A2 @ N1
            z = _dsolve(M.T @ M + damping * I_nv, M.T @ (b2 - A2 @ d1))
            dq = d1 + N1 @ z
        else:
            # Single-level (order-independent; identical to the pre-hierarchy solve).
            A = np.vstack(rows1 + rows2)
            b = np.concatenate(rhs1 + rhs2)
            dq = _dsolve(A.T @ A + damping * I_nv, A.T @ b)

        n = np.linalg.norm(dq)
        if n > max_step_norm:
            dq = dq * (max_step_norm / n)

        dq *= step_scale

        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)

        clamp_hinge_joint_limits(model, data.qpos)

        # Do not force root orientation here.
        # For get-up motions, forcing an upright root causes the limbs to absorb
        # the whole lying-to-standing rotation and produces bad branch choices.
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])

        mujoco.mj_forward(model, data)

    if diag_out is not None:
        max_floor_pen = 0.0
        if floor_gid is not None:
            for cc in range(data.ncon):
                ct = data.contact[cc]
                if ct.geom1 != floor_gid and ct.geom2 != floor_gid:
                    continue
                max_floor_pen = max(max_floor_pen, -float(ct.dist))
        diag_out["floor_pen_cm"] = max_floor_pen * 100.0
        max_hold_slip = 0.0
        for role in hold:
            bid = role_to_body_id.get(role)
            if bid is None or role not in targets:
                continue
            max_hold_slip = max(max_hold_slip, float(np.linalg.norm((data.xpos[bid] - targets[role])[:2])))
        diag_out["hold_slip_cm"] = max_hold_slip * 100.0

    return data.qpos.copy()


def _detect_arm_floor_onset_windows(model, data, qpos, floor_gid, preroll, ramp):
    """For each hand effector, find frames where its wrist/gripper chain first
    touches the floor (true penetration, dist<0) after being clear, and return
    windows [onset-preroll, onset+ramp] (clipped to the clip) with the onset
    frame index — the swing/reach transition that Stage 3's flat floor
    repulsion currently corrects in ONE frame with no ease-in (see
    collisionFixPlan.md wrist-flick diagnosis).

    Only ONSET (clear->penetrating) transitions matter — a body that starts
    already penetrating and stays that way isn't a sudden reconfiguration."""
    T = qpos.shape[0]
    windows = {}   # eff -> list of (lo, onset, hi)
    for eff, body_names in ARM_CHAIN_FLOOR_BODIES.items():
        body_ids = {body_id(model, n) for n in body_names}
        was_pen = False
        runs = []
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            pen = False
            for c in range(data.ncon):
                ct = data.contact[c]
                if ct.geom1 != floor_gid and ct.geom2 != floor_gid:
                    continue
                other = ct.geom2 if ct.geom1 == floor_gid else ct.geom1
                if int(model.geom_bodyid[other]) in body_ids and ct.dist < 0:
                    pen = True
                    break
            if pen and not was_pen:
                lo = max(0, t - preroll)
                hi = min(T - 1, t + ramp)
                runs.append((lo, t, hi))
            was_pen = pen
        if runs:
            windows[eff] = runs
    return windows


def refine_arm_floor_transitions(
    model, data, qpos_pass1, windows, frame_cache, role_to_body_id,
    ori_role_to_body_id, floor_weight, floor_margin, floor_gain,
    coll_kwargs, arm_posture_reg=0.02, lock_weight=1.0e4, iters=30,
):
    """Second pass (collisionFixPlan.md / notes.md two-pass idea, scoped): for
    each detected floor-onset window, LOCALLY re-solve just the affected arm's
    chain over a short window around the transition, instead of letting Stage
    3's single whole-body per-frame solve absorb a sudden floor-repulsion
    demand as one large joint reconfiguration (measured: RIGHT_WRIST_X slammed
    68deg in one frame into its hard joint limit — see collisionFixPlan.md).

    Mechanism: process the window frame-by-frame, warm-starting AND
    regularizing (posture_reg) each frame against the PREVIOUSLY REFINED frame
    (not its own Pass-1 raw q_ref) — this is what forces temporal continuity
    THROUGH the transition instead of jumping to it. A per-DOF posture_reg
    vector locks every joint except the target arm's 7-joint chain at a huge
    weight (root DOFs are always free — the posture_reg term never touches
    them, see solve_frame_position_ik), so the local re-solve can't disturb
    the rest of the already-good Pass-1 body, and floor_weight is
    cosine-ramped 0->1 across the preroll leading up to the onset frame
    (mirrors the existing contact_ramp/contact_preroll pattern used for
    hand/foot contact, which floor_collision_rows otherwise has none of).

    `frame_cache[t]` must hold the EXACT (targets, ori_targets,
    align_constraints, pos_site_constraints, skip_pos_roles, skip_ori_body_ids,
    ori_weight_scale, pos_weight_scale, hold_pos_roles) tuple Pass 1 used for
    frame t — reused verbatim rather than recomputed, since target
    construction has cross-frame state (the foot-hold anchor) that only Pass
    1's sequential loop computes correctly."""
    qpos_out = qpos_pass1.copy()
    nv = model.nv

    for eff, runs in windows.items():
        chain_dofadr = [joint_info(model, n)[1] for n in ARM_CHAIN_JOINTS[eff]]
        for lo, onset, hi in runs:
            preg = np.full(nv, lock_weight)
            preg[0:6] = 0.0                      # root: posture_reg never applies here anyway
            for d in chain_dofadr:
                preg[d] = arm_posture_reg

            ramp_len = onset - lo
            q_prev = qpos_out[max(lo - 1, 0)].copy()
            for t in range(lo, hi + 1):
                if ramp_len > 0 and t <= onset:
                    w = 0.5 - 0.5 * np.cos(np.pi * (t - lo) / ramp_len)
                else:
                    w = 1.0
                (targets, ori_targets, align_constraints, pos_site_constraints,
                 skip_pos_roles, skip_ori_body_ids, ori_weight_scale,
                 pos_weight_scale, hold_pos_roles) = frame_cache[t]

                # Warm-start EVERY joint from Pass 1's OWN solved pose at this
                # exact frame (a good starting guess — avoids drift). Regularize
                # (q_ref) differently per joint: the target arm's chain toward
                # q_prev (the PREVIOUSLY REFINED frame — this is what forces
                # temporal continuity across the transition); every other joint
                # toward its OWN Pass-1 value at this frame (so a "locked" joint
                # keeps following whatever it was already smoothly doing,
                # instead of being frozen at a stale value for the whole window
                # — freezing was the bug in an earlier version: it created a
                # NEW discontinuity when the freeze let go at the window exit).
                q_init_t = qpos_pass1[t].copy()
                q_ref_t = qpos_pass1[t].copy()
                q_ref_t[7:] = np.where(
                    np.isin(np.arange(nv - 6), np.asarray(chain_dofadr) - 6),
                    q_prev[7:], q_ref_t[7:])

                # NOTE: coll_kwargs already carries floor_gid (see main()'s
                # coll_kwargs docstring — always passed, independent of floor
                # avoidance, so self_collision_rows correctly excludes the
                # floor geom) — do not pass it again here.
                q_t = solve_frame_position_ik(
                    model, data, role_to_body_id, targets, q_init_t,
                    ori_role_to_body_id=ori_role_to_body_id, ori_targets=ori_targets,
                    align_constraints=align_constraints,
                    skip_ori_body_ids=skip_ori_body_ids,
                    pos_site_constraints=pos_site_constraints,
                    skip_pos_roles=skip_pos_roles,
                    ori_weight_scale=ori_weight_scale, pos_weight_scale=pos_weight_scale,
                    hold_pos_roles=hold_pos_roles,
                    iters=iters, posture_reg=preg, q_ref=q_ref_t,
                    floor_weight=w * floor_weight,
                    floor_margin=floor_margin, floor_gain=floor_gain,
                    **coll_kwargs,
                )
                qpos_out[t] = q_t
                q_prev = q_t
            print(f"  [floor-refine] {eff}: window [{lo},{hi}] onset={onset} "
                  f"({hi - lo + 1} frames re-solved)")
    return qpos_out


def _leg_floor_pen_flags(model, data, qpos, alex_floor_z, sole_corner_sids, pen_tol):
    """Per-frame boolean: does this foot's lowest sole corner sit more than
    `pen_tol` below `alex_floor_z`? Purely geometric (mesh-accurate site
    height vs the same target-space floor estimate the rest of Stage 3 uses)
    -- deliberately independent of the contact-flag 'planted' label, since the
    failure mode this feeds (refine_leg_floor_transitions) can be labelled
    EITHER way: a swinging foot (unplanted) or a genuinely 'planted' foot
    whose registration/phase-gating still leaves it penetrating (e.g. the
    knee-140deg-saturated deep-tuck case, see wiki/results/tradeoffs-limits.md)."""
    T = qpos.shape[0]
    flags = np.zeros(T, dtype=bool)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        z = min(float(data.site_xpos[s][2]) for s in sole_corner_sids)
        flags[t] = (z - alex_floor_z) < -pen_tol
    return flags


def refine_leg_floor_transitions(
    model, data, qpos_pass1, frame_cache, role_to_body_id, ori_role_to_body_id,
    alex_floor_z, ankle_clearance, coll_kwargs,
    pen_tol=0.015, ramp=20, preroll=20,
    leg_posture_reg=0.02, lock_weight=1.0e4,
    root_pos_relief=0.3, foot_flat_weight=3.0, iters=30,
):
    """Third floor-transition refine pass (mirrors refine_arm_floor_transitions's
    architecture, targets a DIFFERENT failure mode). A prior attempt at get-up
    swing-foot floor penetration (`--swing-clear`, an orientation-pitch-cap
    only) drove the numbers down but CONTORTED the leg on TUCKED phases --
    forcing a foot flat while it sits folded under the body, without letting
    the leg/root reconfigure, demands a locally infeasible pose (rejected on
    visual review, see wiki/experiments/retired-approaches.md). The root cause
    there wasn't the orientation cap per se, it was that nothing else was
    given room to move.

    This pass instead SYNTHESIZES A TEMPORARY PLANT for the ramped window: the
    ankle position target is blended toward floor+clearance and a foot-flat
    align_constraint (the SAME mechanism a genuine contact uses) is turned on
    -- i.e. treat the foot as if it just committed to the ground, exactly like
    an already-proven real plant, rather than only editing its orientation.
    Critically, pelvis/torso POSITION TRACKING is ALSO relaxed
    (pos_weight_scale) during the window so the ROOT (translation, and
    indirectly pitch via the whole-body solve) is actually free to rise/shift
    to make room for the leg -- Stage 3 already solves root position freely
    (posture_reg never touches DOFs 0-5, same convention as
    refine_arm_floor_transitions), the missing piece was that pelvis/torso's
    OWN heavy tracking weight was fighting that freedom every single frame.

    Detection is geometric (see _leg_floor_pen_flags), not contact-flag-based,
    so it also catches a nominally-'planted' foot left penetrating by
    phase-aware gating (the luigi_standSupine_08 knee-140 case).

    Same temporal-continuity trick as the arm pass: warm-start every touched
    frame from Pass 1's OWN value (good initial guess), regularize the leg
    chain toward the PREVIOUSLY REFINED frame (not a frozen value -- avoids
    creating a new discontinuity at the window's exit), lock everything else
    at `lock_weight` so a local re-solve can't disturb an already-good body.
    Uses `ramp_envelope` (imported from contact_labels, the SAME cosine
    onset/release ramp every contact term in this codebase uses) so a
    multi-frame sustained dig gets a full-strength plateau, not just a
    fixed-width bump around a single onset instant -- unlike the arm pass's
    abrupt bumps, a tuck-phase dig can last many tens of frames."""
    qpos_out = qpos_pass1.copy()
    nv = model.nv

    for eff, sole_sids in SOLE_CORNER_SITES.items():
        sids = [site_id(model, s) for s in sole_sids]
        pen_flags = _leg_floor_pen_flags(model, data, qpos_pass1, alex_floor_z, sids, pen_tol)
        if not pen_flags.any():
            continue
        alpha = ramp_envelope(pen_flags, ramp, preroll)

        # Contiguous nonzero segments of alpha are independent "windows" for
        # q_prev-reset purposes (a later, separate dig event on the same foot
        # must not warm-continuity from a much earlier one).
        T = qpos_pass1.shape[0]
        touched = np.where(alpha > 1e-9)[0]
        if touched.size == 0:
            continue
        segments = []
        seg_start = touched[0]
        prev_t = touched[0]
        for t in touched[1:]:
            if t != prev_t + 1:
                segments.append((seg_start, prev_t))
                seg_start = t
            prev_t = t
        segments.append((seg_start, prev_t))

        ank_role = FOOT_POS_ROLE[eff]
        cfg = CONTACT_EFFECTORS[eff]
        foot_bid = body_id(model, cfg["body"])
        ori_bid = ori_role_to_body_id[cfg["ori_role"]]
        chain_dofadr = [joint_info(model, n)[1] for n in LEG_CHAIN_JOINTS[eff]]

        preg = np.full(nv, lock_weight)
        preg[0:6] = 0.0                     # root: always free, matches arm-refine
        for d in chain_dofadr:
            preg[d] = leg_posture_reg

        n_frames_touched = 0
        for lo, hi in segments:
            q_prev = qpos_out[max(lo - 1, 0)].copy()
            for t in range(lo, hi + 1):
                w = float(alpha[t])
                if w <= 0.0:
                    continue
                (targets, ori_targets, align_constraints, pos_site_constraints,
                 skip_pos_roles, skip_ori_body_ids, ori_weight_scale,
                 pos_weight_scale, hold_pos_roles) = frame_cache[t]

                # Synthetic plant: blend the ankle Z target toward floor+clearance
                # (never touches XY -- this is a vertical rest correction, not a
                # foot relocation) and add a foot-flat align constraint, ramped
                # by w. Copy dicts/lists -- must not mutate frame_cache (shared
                # across effectors/other refine passes).
                targets_t = dict(targets)
                rest_z = alex_floor_z + ankle_clearance
                tgt = targets[ank_role].copy()
                tgt[2] = (1.0 - w) * tgt[2] + w * rest_z
                targets_t[ank_role] = tgt

                align_constraints_t = list(align_constraints) + [
                    (foot_bid, cfg["axis_local"], cfg["world_dir"], foot_flat_weight * w, True)
                ]
                ori_weight_scale_t = dict(ori_weight_scale)
                ori_weight_scale_t[ori_bid] = min(ori_weight_scale_t.get(ori_bid, 1.0), 1.0 - w)

                # Relax pelvis/torso tracking so the root is actually free to
                # rise/shift to make room for the leg (see docstring) -- scaled
                # from 1.0 (untouched) down to root_pos_relief at full w.
                pos_weight_scale_t = dict(pos_weight_scale)
                for role in ("pelvis", "torso"):
                    base = pos_weight_scale_t.get(role, 1.0)
                    pos_weight_scale_t[role] = base * (1.0 - w * (1.0 - root_pos_relief))

                q_init_t = qpos_pass1[t].copy()
                q_ref_t = qpos_pass1[t].copy()
                q_ref_t[7:] = np.where(
                    np.isin(np.arange(nv - 6), np.asarray(chain_dofadr) - 6),
                    q_prev[7:], q_ref_t[7:])

                q_t = solve_frame_position_ik(
                    model, data, role_to_body_id, targets_t, q_init_t,
                    ori_role_to_body_id=ori_role_to_body_id, ori_targets=ori_targets,
                    align_constraints=align_constraints_t,
                    skip_ori_body_ids=skip_ori_body_ids,
                    pos_site_constraints=pos_site_constraints,
                    skip_pos_roles=skip_pos_roles,
                    ori_weight_scale=ori_weight_scale_t, pos_weight_scale=pos_weight_scale_t,
                    hold_pos_roles=hold_pos_roles,
                    iters=iters, posture_reg=preg, q_ref=q_ref_t,
                    **coll_kwargs,
                )
                qpos_out[t] = q_t
                q_prev = q_t
                n_frames_touched += 1
        print(f"  [leg-floor-refine] {eff}: {len(segments)} window(s), "
              f"{n_frames_touched} frames re-solved (pen_tol={pen_tol*100:.1f}cm)")
    return qpos_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_DEFAULT, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--ik-iters", type=int, default=40)
    ap.add_argument("--ori-scale", type=float, default=1.0)
    ap.add_argument("--coll-weight", type=float, default=20.0,
                    help="Self-collision repulsion weight (0=disabled, default: 20.0 — "
                         "see self_collision_rows docstring for sweep results)")
    ap.add_argument("--coll-margin", type=float, default=0.02,
                    help="Repulsion activates when bodies within this margin (m) of contact (default: 0.02)")
    ap.add_argument("--coll-gain", type=float, default=5.0,
                    help="Correction speed: target separation per metre of penetration (default: 5.0)")
    ap.add_argument("--coll-hops", type=int, default=2,
                    help="Kinematic-tree hop threshold for exclusion beyond MuJoCo's built-in 1-hop filter "
                         "(default: 2, which also excludes grandparent-grandchild pairs like HEAD↔TORSO)")
    ap.add_argument("--floor-weight", type=float, default=0.0,
                    help="Robot-vs-floor avoidance weight (0=disabled, EXPERIMENTAL — default OFF, "
                         "opt-in only). Mesh-accurate (injected floor plane geom, reuses the "
                         "mj_forward contact narrow-phase) and root-aware (this per-frame solve has "
                         "root-position freedom, unlike Stage 4's Stage B) — but on "
                         "luigi_standProne_03, weight=20 (after fixing two real bugs: the "
                         "self_collision_rows floor leak, and one-row-per-raw-contact tripling the "
                         "effective weight) improves the reported LEFT_FOOT violation but creates a "
                         "WORSE new one (RIGHT_GRIPPER 35.8cm) elsewhere — a genuine multi-effector "
                         "priority tension, not a tuned/validated fix yet. See collision.md. See "
                         "floor_collision_rows docstring.")
    ap.add_argument("--floor-margin", type=float, default=0.0,
                    help="Repulsion activates within this margin (m) of the floor (default: 0.0 — "
                         "unlike --coll-margin's 2cm buffer, a genuinely planted foot/hand sits AT "
                         "the floor by design; a nonzero margin would fight the contact terms)")
    ap.add_argument("--floor-gain", type=float, default=5.0,
                    help="Correction speed: target separation per metre of floor penetration (default: 5.0)")
    ap.add_argument("--floor-phase-aware", dest="floor_phase_aware", action="store_true",
                    default=False,
                    help="Scale --floor-weight per-frame by floor_phase_weight() (pelvis-target-Z "
                         "smoothstep between the clip's low phase and its planted-foot/standing "
                         "phase). Default off (identity, byte-for-byte same as before this flag "
                         "existed). For clips with a genuine lying/standing phase split (e.g. "
                         "get-ups): forcing full floor collision through the lying phase misreads "
                         "the legitimately-low pelvis as a violation (see floor_phase_weight "
                         "docstring) — this ramps the term down there instead of applying it "
                         "uniformly. No-op (all-1s) for single-phase clips.")
    ap.add_argument("--floor-refine", dest="floor_refine", action="store_true", default=True,
                    help="Second pass (only runs when --floor-weight > 0): when a hand's wrist/"
                         "gripper chain first touches the floor (an onset the whole-body per-frame "
                         "solve otherwise absorbs as ONE large joint reconfiguration — measured: "
                         "RIGHT_WRIST_X slammed 68deg into its hard limit in a single frame), locally "
                         "re-solve just that arm over a short window around the transition, warm-"
                         "started/regularized against the PREVIOUSLY REFINED frame for temporal "
                         "continuity, with floor_weight cosine-ramped in (matching the existing "
                         "contact_ramp/contact_preroll pattern, which the flat floor term otherwise "
                         "lacks). Default on (inert unless --floor-weight > 0). See "
                         "refine_arm_floor_transitions / collisionFixPlan.md.")
    ap.add_argument("--no-floor-refine", dest="floor_refine", action="store_false")
    ap.add_argument("--floor-refine-preroll", type=int, default=8,
                    help="Frames before a floor-onset event over which floor_weight ramps 0->1 "
                         "(default: 8, matching --contact-ramp's scale)")
    ap.add_argument("--floor-refine-ramp", type=int, default=8,
                    help="Frames the refined window extends PAST the onset frame (default: 8)")
    ap.add_argument("--floor-refine-posture-reg", type=float, default=0.02,
                    help="posture_reg for the target arm's own 7 joints during refinement — higher "
                         "than the global default (1e-3) to bias toward the previous refined frame "
                         "for smoothness, without fighting the position/floor targets (default: 0.02)")
    ap.add_argument("--floor-refine-lock-weight", type=float, default=1.0e4,
                    help="posture_reg for every OTHER joint during refinement — locks them near the "
                         "previous frame so the local re-solve can't disturb the rest of the already-"
                         "good Pass-1 body (root DOFs are always free regardless). Default: 1e4.")
    ap.add_argument("--leg-floor-refine", dest="leg_floor_refine", action="store_true", default=False,
                    help="Third floor-transition refine pass (see refine_leg_floor_transitions): a "
                         "tucked/deep-crouch leg whose rigid foot plate or knee-range limit drives it "
                         "through the floor even with a faithful human copy (the knee-140deg embodiment "
                         "gap, wiki/results/tradeoffs-limits.md) gets a local re-solve that synthesizes "
                         "a temporary PLANT (ankle blended to floor+clearance, foot-flat align turned "
                         "on) AND relaxes pelvis/torso tracking so the root can rise/shift to make room. "
                         "Supersedes the rejected --swing-clear (orientation-cap-only, contorted tucked "
                         "legs). Default off, independent of --floor-weight/--floor-refine.")
    ap.add_argument("--leg-floor-refine-pen-tol", type=float, default=0.015,
                    help="With --leg-floor-refine: trigger when a foot's lowest sole corner sits more "
                         "than this (m) below the target-space floor estimate (default: 0.015 = 1.5cm)")
    ap.add_argument("--leg-floor-refine-preroll", type=int, default=20,
                    help="With --leg-floor-refine: frames the ramp extends BEFORE a penetration onset "
                         "(default: 20 — wider than the arm pass's 8, since a tuck-phase dig develops "
                         "more gradually than a sudden reach)")
    ap.add_argument("--leg-floor-refine-ramp", type=int, default=20,
                    help="With --leg-floor-refine: cosine ramp width (frames) at each penetration run's "
                         "onset/release edge (default: 20)")
    ap.add_argument("--leg-floor-refine-posture-reg", type=float, default=0.02,
                    help="With --leg-floor-refine: posture_reg for the target leg's own 6 joints during "
                         "refinement (default: 0.02, matches the arm pass)")
    ap.add_argument("--leg-floor-refine-lock-weight", type=float, default=1.0e4,
                    help="With --leg-floor-refine: posture_reg for every OTHER joint (default: 1e4, "
                         "matches the arm pass; root DOFs always free regardless)")
    ap.add_argument("--leg-floor-refine-root-relief", type=float, default=0.3,
                    help="With --leg-floor-refine: at full window strength, scale pelvis/torso position "
                         "tracking weight to this fraction of normal (default: 0.3) so the root is free "
                         "to rise/shift and relieve a knee-saturated leg instead of the foot being "
                         "forced flat while pelvis/torso stay rigidly pinned to the human copy.")
    ap.add_argument("--leg-floor-refine-flat-weight", type=float, default=3.0,
                    help="With --leg-floor-refine: foot-flat align_constraint weight at full window "
                         "strength (default: 3.0, matches --foot-flat-weight's default)")
    ap.add_argument("--foot-contact-height", type=float, default=0.07,
                    help="Foot marker height (m) above clip floor below which a foot is in contact (default: 0.07)")
    ap.add_argument("--hand-contact-height", type=float, default=0.08,
                    help="Hand marker height (m) above clip floor below which a hand is in contact (default: 0.08)")
    ap.add_argument("--contact-speed", type=float, default=0.4,
                    help="Marker speed (m/s) below which an effector can be in contact (default: 0.4)")
    ap.add_argument("--contact-on-height-frac", type=float, default=0.7,
                    help="Onset hysteresis: contact turns ON only below height*frac (stays on under "
                         "the base threshold). Delays touchdown labelling until genuinely settled. "
                         "1.0 disables (default: 0.7)")
    ap.add_argument("--contact-on-speed-frac", type=float, default=0.5,
                    help="Onset hysteresis: contact turns ON only below speed*frac (stays on under "
                         "the base threshold). 1.0 disables (default: 0.5)")
    ap.add_argument("--contact-onset-max-delay", type=float, default=0.15,
                    help="Cap (s) on the hysteresis onset delay: a plant that never passes the strict "
                         "gate is trimmed by at most this, never dropped (default: 0.15)")
    ap.add_argument("--foot-flat-tilt", type=float, default=40.0,
                    help="FALLBACK absolute cap: max tilt (deg) of the human foot frame local-Z from "
                         "vertical, used only when a foot has too few candidate frames to self-calibrate "
                         "its baseline (default: 40)")
    ap.add_argument("--foot-flat-margin", type=float, default=6.0,
                    help="Primary flatness gate: a foot counts as a flat plantar support when its frame "
                         "tilt is within this many deg ABOVE the foot's own self-calibrated flat baseline "
                         "(p15 of its candidate-contact tilt). Corrects the ~18deg toe-ankle frame offset "
                         "+ L/R skew that makes the raw --foot-flat-tilt gate nearly inert. Set at the "
                         "corpus break (clean plants <3deg above baseline, phantoms >=6deg). (default: 6)")
    ap.add_argument("--foot-yaw-weight", type=float, default=1.5,
                    help="Weight for the foot heading (yaw) align during contact: drives the foot "
                         "forward axis to the HUMAN foot heading so the planted foot follows the "
                         "human's small heading change instead of free-drifting (inner/outer slip). "
                         "0 disables (default: 1.5)")
    ap.add_argument("--swing-clear", dest="swing_clear", action="store_true", default=False,
                    help="Swing-foot toe-clearance: on airborne (non-planted) foot frames, cap the "
                         "toe-down pitch of the foot ORIENTATION target so the IK spends Alex's unused "
                         "ankle dorsiflexion headroom to lift the toe instead of copying the human's "
                         "plantarflexed low-step pose (which drives the rigid ~20cm foot plate's toe "
                         "corner through the floor -- measured 10-18cm on the get-up clip class). "
                         "Ramped by the contact envelope (full cap airborne, fades to 0 as the foot "
                         "plants). Deliberately deviates from the human mid-step (feasible+smooth > "
                         "faithful-but-underground). Default off.")
    ap.add_argument("--swing-max-pitch", type=float, default=5.0,
                    help="With --swing-clear: max allowed toe-down pitch (deg) of a swing foot's "
                         "forward axis below horizontal. 5 (aggressive) drives swing-foot floor "
                         "penetration ~10->1.5cm on the get-up class. It is SAFE from the per-frame-IK "
                         "branch flip (which Stage-4 cannot smooth) BECAUSE of the paired "
                         "--swing-continuity-reg, which stops the redundant leg from jumping branches. "
                         "Without that reg, caps below ~9 flip; see planLog swing-clear. (default: 5)")
    ap.add_argument("--swing-clear-height", type=float, default=1.0,
                    help="With --swing-clear: proximity gate ZERO height (m) -- the de-pitch fades to 0 "
                         "when a swing foot's achieved ankle sits this far above the floor. Default 1.0 "
                         "= effectively OFF (un-gated): at the shipped 10deg cap the gate is unnecessary "
                         "and only trims the benefit. It exists for AGGRESSIVE-cap experiments, where "
                         "gating out high feet avoids some (not all) branch flips (experimental).")
    ap.add_argument("--swing-clear-band", type=float, default=0.04,
                    help="With --swing-clear: width (m) of the gate ramp below --swing-clear-height "
                         "(full de-pitch below height-band, zero above height). Only relevant when the "
                         "gate is active (swing-clear-height set low, experimental) (default: 0.04).")
    ap.add_argument("--swing-clear-weight", type=float, default=0.0,
                    help="With --swing-clear: least-squares weight of the OPTIONAL soft one-sided "
                         "toe-clearance rows (a swing foot's sole corner below floor+margin pulled UP). "
                         "Default 0 (OFF): in testing the soft term fought the plant machinery and did "
                         "not improve on the gated pitch cap alone. Kept for experiments.")
    ap.add_argument("--swing-coll-boost", type=float, default=3.0,
                    help="With --swing-clear: multiply Stage-3 self-collision repulsion weight by "
                         "(1 + this * de-pitch strength) on de-pitch frames, so the paired continuity "
                         "reg (which holds hip/knee near the previous pose) cannot cancel self-collision "
                         "avoidance -- e.g. the thigh jamming the torso on a kneeling get-up. 0 = off "
                         "(default: 3.0).")
    ap.add_argument("--swing-continuity-reg", type=float, default=0.9,
                    help="With --swing-clear: temporal-continuity posture reg on the hip/knee DOFs "
                         "(NOT the ankle -- it must stay free to flatten), applied only on de-pitch "
                         "frames and RAMPED by the de-pitch strength. Pulls the redundant leg toward "
                         "the PREVIOUS frame's pose so an aggressive cap can't make the per-frame IK "
                         "jump to a different solution BRANCH (a large single-frame flip GlobalOPT "
                         "cannot smooth -- the root problem). 0.9 is the shipped value (cap 5 -> ~1.5cm "
                         "swing pen, 0 spikes). 0 = off (aggressive cap then flips). (default: 0.9)")
    ap.add_argument("--swing-clear-margin", type=float, default=0.005,
                    help="With --swing-clear: clearance height (m) a swing foot's sole corner is held "
                         "above the floor by the soft term. Kept near 0 (like floor_collision_rows' "
                         "margin=0): a larger margin lifts a PLANTING foot off the floor and fights "
                         "foot-hold/foot-flat (tug-of-war -> branch flip). Corrects real penetration "
                         "only (default: 0.005 = 5mm).")
    ap.add_argument("--contact-min-run", type=int, default=3,
                    help="Debounce: remove contact ON/OFF runs shorter than this many SOLVED "
                         "frames (kills marginal-threshold flicker). 1 disables (default: 3)")
    ap.add_argument("--contact-ramp", type=int, default=4,
                    help="Cross-fade the contact constraints (flat/align, palm-pin, foot-yaw) over "
                         "this many solved frames at make/break instead of snapping. 0 = binary "
                         "(default: 4)")
    ap.add_argument("--contact-preroll", type=int, default=2,
                    help="Look-ahead: begin easing contact constraints in this many solved frames "
                         "BEFORE touchdown so the foot/hand is prepared (default: 2)")
    ap.add_argument("--foot-hold", dest="foot_hold", action="store_true", default=True,
                    help="During foot contact, freeze the foot POSITION target at the frame the foot "
                         "first becomes planted and cross-fade the moving human target onto it, so a "
                         "flat foot stops sliding while it is down (kills standup plant-slip). Default on.")
    ap.add_argument("--no-foot-hold", dest="foot_hold", action="store_false",
                    help="Disable the planted-foot position hold (feet track the moving human target).")
    ap.add_argument("--foot-hold-latch", type=float, default=0.5,
                    help="Contact cross-fade weight [0,1] at/above which the foot-hold anchor latches "
                         "(below it the anchor keeps tracking the descending foot). Default 0.5.")
    ap.add_argument("--foot-hold-weight", type=float, default=10.0,
                    help="While held, multiply the foot (ankle) position weight by this so the planted "
                         "foot resists being dragged by the heavier pelvis/chain targets. Cross-faded "
                         "by the contact weight. 1.0 = freeze target only (no boost). Default 10.0 "
                         "(3.0 let the shovel body motion drag the plant 38-72cm; 10 -> 23-38cm and "
                         "also cuts standup foot drag with no tracking/smoothness cost).")
    ap.add_argument("--hierarchical", dest="hierarchical", action="store_true", default=False,
                    help="Task-priority IK: FOOT contact tasks (flat/yaw/hold) at high priority, body "
                         "+ best-effort hand contacts tracked in their nullspace. Default OFF and NOT "
                         "used by the unified pipeline: hold-weight 10 + GlobalOPT Stage B reaches "
                         "lower plant slip (shovel 1.5cm vs 4.7cm) with one config for all actions, "
                         "and hierarchy still regresses on pivoting get-up contacts (standup_natural_01 "
                         "tracking +13%, jumps +35% even with hands demoted to soft).")
    ap.add_argument("--no-hierarchical", dest="hierarchical", action="store_false")
    ap.add_argument("--hard-tier", dest="hard_tier", action="store_true", default=False,
                    help="hierarchical-v1 H2 (plan.md): forces --hierarchical on -- SAFE, verified "
                         "on standup_natural_01 (the exact clip the ORIGINAL --hierarchical "
                         "regression named, wiki/experiments/retired-approaches.md) to behave the "
                         "same order-of-magnitude as the non-hierarchical baseline (mean_err "
                         "0.08-0.11 either way) when hands stay soft. Does NOT promote floor to "
                         "hard -- see --floor-hard, which is a SEPARATE flag, default off, and "
                         "CONFIRMED BROKEN (not just untested): combining floor-collision rows "
                         "into the SAME level-1 tier as foot-hold position-equality rows caused a "
                         "44-METRE divergence on standup_natural_01 (mean_err 0.05m -> 44.4m by "
                         "frame 657) -- a conflicting-equality-row blowup, isolated via direct A/B "
                         "against this flag alone (which does NOT blow up). Do not combine "
                         "--hard-tier with --floor-hard without a redesign (e.g. a proper nested "
                         "3-tier priority: hold > floor > tracking, not floor competing WITHIN the "
                         "same undifferentiated level-1 system as hold). See planLog.md H2.")
    ap.add_argument("--floor-hard", dest="floor_hard_flag", action="store_true", default=False,
                    help="SEPARATE from --hard-tier, default off, CONFIRMED BROKEN -- do not use "
                         "without a redesign. See --hard-tier's help text and planLog.md H2 for "
                         "the 44-metre divergence this caused on standup_natural_01. Kept in the "
                         "codebase (not deleted) per this project's retired-approaches convention "
                         "-- don't resurrect without new evidence.")
    ap.add_argument("--foot-flat-weight", type=float, default=3.0,
                    help="Foot-flat (up-axis->+Z) align weight during contact. NOTE: raising this to "
                         "out-prioritise the hold backfires - the cross-product orientation error has a "
                         "spurious 180-deg equilibrium and high stiffness flips a near-limit foot. "
                         "Proper flatness priority needs a hierarchical/nullspace solve. Default 3.0.")
    ap.add_argument("--shank-clamp", dest="shank_clamp", action="store_true", default=True,
                    help="During foot contact, project the KNEE position target into the shank-tilt "
                         "region reachable with a flat foot (ankle pitch/roll joint ranges, read from "
                         "the model). Retargeting-side feasibility: removes the near-limit-ankle fight "
                         "between foot-flat and human leg tracking. Default on.")
    ap.add_argument("--no-shank-clamp", dest="shank_clamp", action="store_false")
    ap.add_argument("--shank-margin-deg", type=float, default=5.0,
                    help="Keep the clamped shank tilt this many degrees INSIDE the ankle joint range "
                         "(default: 5)")
    ap.add_argument("--coplanar-feet-mode", choices=["min", "mean", "off"], default="mean",
                    help="When BOTH feet are contact-engaged, snap their ankle-height targets to a "
                         "common Z (foot-flat makes equal ankle Z ⇒ equal sole Z, so the feet come out "
                         "coplanar). Fixes inconsistent 'both planted but several-cm-apart' targets that "
                         "a rigid grounding shift can't reconcile (one foot floats in playback). "
                         "mean = meet in the middle (default — distributes the correction, lowest "
                         "self-collision cost); min = snap the higher foot DOWN to the lower/grounded "
                         "one (more source-faithful but a more extended pose → more self-collision); "
                         "off = legacy (no enforcement).")
    ap.add_argument("--knee-bias-weight", type=float, default=0.5,
                    help="Weight of the one-sided knee-flexion bias: weakly push a knee straighter than "
                         "--knee-min-flex-deg back toward it (straight knee = joint lower limit + leg "
                         "Jacobian singularity; humanoids stand slightly bent). Inactive once the knee "
                         "is bent, so it cannot over-constrain. 0 disables (default: 0.5)")
    ap.add_argument("--knee-min-flex-deg", type=float, default=12.0,
                    help="Knee flexion (deg) below which the knee-bend bias engages (default: 12)")
    ap.add_argument("--log-every", type=int, default=1,
                    help="Print a per-frame line every N solved frames (1=all). "
                         "Final frame + summary always printed. Use >1 to cut log volume.")
    ap.add_argument("--no-contact-first", action="store_true",
                    help="Disable contact-gated foot-flat / fist-down overrides (baseline behaviour)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    (roles, role_to_idx, src_positions, fps, orientation_roles, ori_to_idx, orientation_mats,
     persisted_contacts, persisted_eff_names) = load_canonical(args.canonical)

    missing = [r for r in ROLE_TO_ALEX_BODY if r not in role_to_idx]
    if missing:
        raise RuntimeError(f"Canonical missing required roles: {missing}")

    model, data, floor_gid, floor_mocap_id = _load_model_with_floor(args.model)

    role_to_body_id = {
        role: body_id(model, body_name)
        for role, body_name in ROLE_TO_ALEX_BODY.items()
    }

    missing_ori = [r for r in ORI_TO_ALEX_BODY if r not in ori_to_idx]
    if missing_ori:
        raise RuntimeError(f"Canonical missing required orientation roles: {missing_ori}")

    ori_role_to_body_id = {
        role: body_id(model, body_name)
        for role, body_name in ORI_TO_ALEX_BODY.items()
    }

    # Contact-first: ground contact per effector from the human data, and resolve
    # the Alex bodies whose axes we align during contact.
    #
    # Prefer PERSISTED labels from Stage 2.5 (scripts/ground_canonical_human.py,
    # phasic-v2 M1/T1.3) -- one detector, run once, upstream of morphology
    # scaling, instead of every consumer re-deriving its own. Stage 2.5 also
    # rigidly shifts the canonical positions so the floor is z=0 BY
    # CONSTRUCTION; floor_z is therefore 0.0 here, not re-estimated (the old
    # 1st-percentile-of-feet-markers estimate is deleted, per plan.md T1.3 --
    # it was exactly the kind of per-stage floor re-estimation this redesign
    # replaces with a single upstream invariant).
    #
    # Falls back to on-the-fly detection (the pre-phasic-v2 behaviour, floor_z
    # re-estimated) only when given a plain _with_orient.npz that never went
    # through Stage 2.5 -- keeps this script runnable standalone / on old data.
    contact_first = not args.no_contact_first
    if persisted_contacts is not None:
        contacts = persisted_contacts
        floor_z = 0.0
        print(f"  [contacts] using PERSISTED labels from {args.canonical.name} "
              f"({', '.join(persisted_eff_names)}); floor_z=0.0 (invariant)")
    else:
        print(f"  [contacts] WARNING: {args.canonical.name} has no persisted contact_flags "
              f"(not run through Stage 2.5 / scripts/ground_canonical_human.py) -- "
              f"falling back to on-the-fly detection with a re-estimated floor_z")
        contacts, floor_z = detect_contacts_from_human(
            src_positions, role_to_idx, fps,
            orientation_mats=orientation_mats, ori_to_idx=ori_to_idx,
            foot_height=args.foot_contact_height,
            hand_height=args.hand_contact_height,
            speed_thresh=args.contact_speed,
            foot_flat_tilt=args.foot_flat_tilt,
            foot_flat_margin=args.foot_flat_margin,
            on_height_frac=args.contact_on_height_frac,
            on_speed_frac=args.contact_on_speed_frac,
            onset_max_delay=args.contact_onset_max_delay,
        )
    effector_body_id = {
        eff: body_id(model, cfg["body"]) for eff, cfg in CONTACT_EFFECTORS.items()
    }
    effector_site_id = {
        eff: site_id(model, cfg["site"]) for eff, cfg in CONTACT_POS.items()
    }
    # Sole-corner site ids per foot for the swing-foot toe-clearance term.
    sole_corner_sids = {
        eff: [site_id(model, s) for s in names]
        for eff, names in SOLE_CORNER_SITES.items()
    }
    # Hip+knee DOF addresses (NOT ankle -- the ankle must stay free to dorsiflex and
    # flatten the foot). The swing-clear temporal-continuity reg boosts ONLY these, so
    # it stops the redundant leg from branch-flipping without suppressing the de-pitch
    # itself or touching the arms/spine (a global boost caused a wrist flip in testing).
    _leg_cont_joints = ["LEFT_HIP_X", "LEFT_HIP_Z", "LEFT_HIP_Y", "LEFT_KNEE_Y",
                        "RIGHT_HIP_X", "RIGHT_HIP_Z", "RIGHT_HIP_Y", "RIGHT_KNEE_Y"]
    leg_cont_dofs = [joint_info(model, nm)[1] for nm in _leg_cont_joints]

    # Shank-tilt clamp limits from the model's ankle joint ranges. With the foot
    # flat, shank forward-lean = -ANKLE_Y and leftward-lean = +ANKLE_X (chain:
    # R_foot = R_shin * Ry(ankle_y) * Rx(ankle_x); shank axis = shin +Z).
    shank_limits = {}
    if args.shank_clamp:
        sm = np.radians(args.shank_margin_deg)
        for eff, side in (("left_foot", "LEFT"), ("right_foot", "RIGHT")):
            _, _, (lo_y, hi_y) = joint_info(model, f"{side}_ANKLE_Y")
            _, _, (lo_x, hi_x) = joint_info(model, f"{side}_ANKLE_X")
            shank_limits[eff] = (
                (-hi_y + sm, -lo_y - sm),   # forward-lean range
                (lo_x + sm, hi_x - sm),     # leftward-lean range
            )

    # One-sided knee-flexion bias (see solve_frame_position_ik).
    knee_bias = None
    if args.knee_bias_weight > 0.0:
        entries = []
        for side in ("LEFT", "RIGHT"):
            qadr, dofadr, _ = joint_info(model, f"{side}_KNEE_Y")
            entries.append((qadr, dofadr))
        knee_bias = (entries, np.radians(args.knee_min_flex_deg), args.knee_bias_weight)
    ori_role_body_id = {
        role: body_id(model, ORI_TO_ALEX_BODY[role]) for role in ORI_TO_ALEX_BODY
    }

    frame_ids = list(range(0, src_positions.shape[0], args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    # Contact conditioning at the SOLVED frame rate: debounce marginal-threshold
    # flicker, then build a cross-fade envelope (cosine ramp + look-ahead preroll)
    # so contact constraints ease in/out instead of snapping full-weight at onset
    # (measured: pose jump is ~2.8x larger at raw contact transitions; feet were
    # yanked flat from a mean 47deg tilt).
    fidx = np.asarray(frame_ids)
    contacts_solved = {eff: debounce_flags(contacts[eff][fidx], args.contact_min_run)
                       for eff in CONTACT_EFFECTORS}
    contact_env = {eff: ramp_envelope(contacts_solved[eff], args.contact_ramp, args.contact_preroll)
                   for eff in CONTACT_EFFECTORS}

    first_src_pos = src_positions[frame_ids[0]]
    first_src_ori = orientation_mats[frame_ids[0]]

    alex_pelvis_to_head = measure_alex_pelvis_to_head(model, data, role_to_body_id)
    root_scale = estimate_source_scale(first_src_pos, role_to_idx, alex_pelvis_to_head)

    print("Canonical:", args.canonical)
    print("Model:", args.model)
    print("Frames:", len(frame_ids), "stride:", args.stride)
    print("Source fps:", fps)
    print(f"Alex pelvis-to-head (model): {alex_pelvis_to_head:.4f} m")
    print("Estimated root scale:", root_scale)
    print("Targets:")
    for role, body_name in ROLE_TO_ALEX_BODY.items():
        print(f"  {role:14s} -> {body_name}")

    print("Orientation targets:")
    for role, body_name in ORI_TO_ALEX_BODY.items():
        print(f"  {role:14s} -> {body_name}  weight={ORI_WEIGHTS.get(role, 0.0) * args.ori_scale:.3f}")

    qpos_list = []
    err_list = []
    target_pos_list = []
    human_target_pos_list = []
    achieved_pos_list = []
    target_ori_list = []
    achieved_ori_list = []
    ori_err_deg_list = []
    n_self_coll_list = []
    n_floor_pen_list = []
    contact_flags_list = []
    contact_align_err_deg_list = []

    q = np.zeros(model.nq)
    q[3] = 1.0  # root quaternion w

    coll_kwargs = dict(
        coll_weight=args.coll_weight,
        coll_margin=args.coll_margin,
        coll_gain=args.coll_gain,
        coll_hops=args.coll_hops,
        # Always passed (not gated behind --floor-weight): self_collision_rows
        # must exclude the injected floor geom regardless of whether floor
        # AVOIDANCE is enabled, or it treats floor contacts as self-colliding
        # robot links and corrupts the solve. See self_collision_rows docstring.
        floor_gid=floor_gid,
    )

    print(f"Self-collision: weight={args.coll_weight}  margin={args.coll_margin} m  "
          f"gain={args.coll_gain}  exclude_hops={args.coll_hops}")
    print()
    print("Solving initial Alex rest-alignment pose...")
    initial_targets = make_initial_alignment_targets(first_src_pos, role_to_idx, root_scale)
    q = solve_frame_position_ik(
        model,
        data,
        role_to_body_id,
        initial_targets,
        q,
        knee_bias=knee_bias,
        iters=max(args.ik_iters * 3, 80),
        **coll_kwargs,
    )
    mujoco.mj_forward(model, data)

    target_rest_positions = {
        role: data.xpos[bid].copy()
        for role, bid in role_to_body_id.items()
    }

    target_rest_orientations = {
        role: body_xmat(data, bid).copy()
        for role, bid in ori_role_to_body_id.items()
    }

    # Per-role scales from comparing source vs Alex achieved-rest proportions.
    role_scales = compute_per_role_scales(first_src_pos, role_to_idx, target_rest_positions)

    # Palm-site rest positions + per-hand morphology scale for the fist position
    # pin (mirrors compute_per_role_scales, but for the palm contact site).
    palm_rest_pos = {eff: data.site_xpos[sid].copy() for eff, sid in effector_site_id.items()}
    src_pelvis0_pos = first_src_pos[role_to_idx["pelvis"]]
    alex_pelvis_rest = target_rest_positions["pelvis"]
    palm_pos_scale = {}
    for eff, cfg in CONTACT_POS.items():
        mk = cfg["marker"]
        if mk not in role_to_idx:
            palm_pos_scale[eff] = root_scale
            continue
        sd = float(np.linalg.norm(first_src_pos[role_to_idx[mk]] - src_pelvis0_pos))
        ad = float(np.linalg.norm(palm_rest_pos[eff] - alex_pelvis_rest))
        palm_pos_scale[eff] = float(np.clip(ad / max(sd, 1e-6), 0.4, 2.5))

    # Alex-frame floor height (target-space derivation — collisionFixPlan.md Fix
    # A). The old pelvis-delta-transform estimate (human floor_z -> Alex frame
    # via root_scale) measured ~8cm too high on luigi_standProne_03 (-0.0355 vs
    # the actual plant height -0.115), which made the floor-repulsion term fight
    # every prone-phase contact frame. Target space is self-consistent: derive
    # the floor from where the SAME ankle position TARGETS the IK is already
    # being asked to track put the foot, instead of transforming a separate
    # human-frame reference.
    #
    # Computed UNCONDITIONALLY (not gated behind --floor-weight): the palm-pin
    # floor clamp below (Fix B) needs it independent of whether the Stage-3
    # floor-REPULSION term is enabled, so the two fixes can be validated in
    # isolation (see collisionFixPlan.md's validation ladder). Only the
    # repulsion term's own inputs (mocap position, floor_kwargs) are gated on
    # --floor-weight.
    # Ankle-to-sole clearance: a fixed geometric property of the robot (how far
    # the ankle joint sits above the sole plane with a flat foot), NOT something
    # that should vary per clip. Measure it from a NEUTRAL default pose in a
    # throwaway MjData — NOT from `data`'s achieved rest pose, which for a
    # prone-start clip like luigi_standProne_03 has the ankle in a contorted,
    # non-flat configuration (measured: 13.9cm there vs the correct ~7.0cm at
    # neutral — a nearly 2x error that would have put the floor 7cm too low).
    neutral_data = mujoco.MjData(model)
    neutral_data.qpos[:] = 0.0
    neutral_data.qpos[3] = 1.0   # root quaternion w
    mujoco.mj_forward(model, neutral_data)
    clearances = []
    for eff, sole_names in SOLE_CORNER_SITES.items():
        ankle_bid = role_to_body_id[FOOT_POS_ROLE[eff]]
        sole_z = [neutral_data.site_xpos[site_id(model, s)][2] for s in sole_names]
        clearances.append(float(neutral_data.xpos[ankle_bid][2] - min(sole_z)))
    ankle_clearance = float(np.mean(clearances))

    # Per-contact-WINDOW target-space floor correction (phasic-v2 M2/T2.1,
    # generalizing collisionFixPlan.md's Fix A/B from a single clip-wide
    # estimate + hands-only to every contacting effector, windowed).
    #
    # WHY WINDOWED, not clip-wide (the mechanism this replaces): Stage 2.5
    # (scripts/ground_canonical_human.py) already grounds the canonical human
    # data so floor=0 BY CONSTRUCTION — but morphology scaling (rest-relative
    # deltas landing on Alex's own achieved-rest pose `a_r`) doesn't
    # automatically preserve that invariant, since `a_r` isn't itself
    # floor-referenced. The old fix pooled ONE alex_floor_z estimate across
    # the WHOLE clip (median over every contact window's onset) — exactly the
    # single-clip-wide-floor pattern this whole redesign replaces elsewhere
    # (see wiki/concepts/grounding.md "Get-up floor residual is
    # BETWEEN-PHASE"): a lying-phase window and a standing-phase window can
    # need different corrections, and pooling them just like the old
    # Stage-3/4 hard-floor-collision bug (see wiki/concepts/globalopt.md)
    # under/over-corrects one phase to accommodate the other. Scoping the
    # SAME onset-median technique to just ONE window at a time removes that
    # coupling for free — no new estimator, just narrower scope.
    #
    # Reuses `contacts_solved` (computed above, line ~1377) — must be
    # bit-identical to what the per-frame loop below treats as "planted".
    # Onset-only within each window (not whole-window), same rationale as the
    # original: the actual per-frame loop FREEZES the tracked target near
    # onset (foot-hold anchor) and holds it, so the robot never tracks a
    # plant's later within-window drift; pooling only the first ONSET_WINDOW
    # frames of a window matches what actually gets tracked.
    #
    # floor_target_z[eff] holds, per window, the median onset-frame target Z
    # of eff's OWN tracked site (ankle for feet, palm site for hands) — same
    # site, same space, no cross-type conversion. NOT floor height: for feet
    # this sits `ankle_clearance` ABOVE the true floor (the ankle joint sits
    # above the sole) by construction, since it's measured directly from the
    # ankle target itself. Clamping a window's later frames against this
    # window's OWN onset value (one-sided max, never push down) stops the
    # "target keeps drifting down through a plant" pattern the original
    # Fix A/B comments already documented, generalized to every effector.
    #
    # (A clearance-subtracted CROSS-TYPE conversion — ankle onset minus
    # ankle_clearance to get a floor-HEIGHT value — is a different quantity,
    # only needed below for the legacy floor-repulsion mocap plane, which
    # places a physical floor surface, not an effector-site target.)
    _window_effectors = list(FOOT_POS_ROLE.keys()) + list(CONTACT_POS.keys())

    def _effector_raw_target_z(eff, src_i):
        """Target Z for eff's tracked site BEFORE floor correction — ankle
        position target for feet, palm-site target for hands. Same formula
        the main per-frame loop below uses, so the onset estimate matches
        what actually gets tracked."""
        if eff in FOOT_POS_ROLE:
            t = make_targets_for_frame(src_positions[src_i], role_to_idx,
                                       first_src_pos, target_rest_positions,
                                       root_scale, role_scales)
            return float(t[FOOT_POS_ROLE[eff]][2])
        cpos = CONTACT_POS[eff]
        mk = cpos["marker"]
        if mk not in role_to_idx:
            return None
        src_pelvis0 = first_src_pos[role_to_idx["pelvis"]]
        src_pelvis = src_positions[src_i][role_to_idx["pelvis"]]
        root_delta = root_scale * (src_pelvis - src_pelvis0)
        rel0 = first_src_pos[role_to_idx[mk]] - src_pelvis0
        rel = src_positions[src_i][role_to_idx[mk]] - src_pelvis
        return float((palm_rest_pos[eff] + root_delta + palm_pos_scale[eff] * (rel - rel0))[2])

    # PER-WINDOW for feet (FOOT_POS_ROLE): a get-up genuinely puts the ankle at
    # different heights-above-floor in different postural phases (lying vs
    # standing), so each window needs its OWN onset reference -- this is the
    # exact between-phase problem the whole redesign targets. Once committed,
    # foot-hold freezes the ankle anyway, so a per-window reference only
    # affects the brief pre-latch settling -- bounded risk even if a window's
    # onset sample is a little noisy.
    ONSET_WINDOW = 10   # frames; run-start proxy for the foot-hold-latch frame
    floor_target_z = {eff: np.full(len(frame_ids), np.nan) for eff in _window_effectors}
    for eff in FOOT_POS_ROLE:
        flag = contacts_solved[eff]
        k = 0
        while k < len(flag):
            if not flag[k]:
                k += 1
                continue
            j = k
            while j < len(flag) and flag[j]:
                j += 1
            onset_zs = [_effector_raw_target_z(eff, frame_ids[kk])
                       for kk in range(k, min(k + ONSET_WINDOW, j))]
            onset_zs = [z for z in onset_zs if z is not None]
            if onset_zs:
                floor_target_z[eff][k:j] = float(np.median(onset_zs))
            k = j

    # Floor HEIGHT (sole-level, not ankle-space): pool ALL feet windows'
    # values and convert back via -ankle_clearance, matching the original
    # Fix A exactly. Feet are the calibration source (ankle_clearance is a
    # known, fixed robot constant tying ankle-space to true floor height);
    # hands have no equivalent constant relating palm-site targets to floor,
    # so they borrow this foot-derived estimate below rather than deriving
    # their own -- see planLog.md M2 for why a hand-own-data estimate (either
    # per-window OR pooled-from-hand-onset) regressed corpus metrics on
    # standup_01 (plPen% 0%->35.6%, coll% 9.0%->20.4%) even though the
    # per-window VALUES individually looked reasonable: without a
    # foot-hold-style freeze, the palm target keeps tracking the moving human
    # target for its whole window, so an under-calibrated hand-own reference
    # (no fixed clearance constant to anchor it) let it sink further than the
    # foot-calibrated one does. Also used for the legacy Stage-3
    # floor-REPULSION QP term's mocap floor-plane placement (`--floor-weight`,
    # default OFF post-M2 — plan.md T2.2), which needs one physical plane and
    # structurally can't be per-window anyway.
    _foot_floor_vals = np.concatenate([floor_target_z[e][~np.isnan(floor_target_z[e])]
                                       for e in FOOT_POS_ROLE if e in floor_target_z])
    alex_floor_z = float(np.median(_foot_floor_vals)) - ankle_clearance if _foot_floor_vals.size else None

    # Hands (CONTACT_POS): clip-wide, foot-derived floor HEIGHT (clearance 0
    # -- the palm/gripper site IS the support surface, same space as
    # alex_floor_z already computed above) -- this is exactly the original
    # Fix B, generalized only in sharing the same clamp code path as feet
    # below, not in its aggregation scope.
    if alex_floor_z is not None:
        for eff in CONTACT_POS:
            flag = contacts_solved[eff]
            floor_target_z[eff][flag] = alex_floor_z

    n_windowed = {e: int(np.sum(~np.isnan(v))) for e, v in floor_target_z.items()}
    print(f"Floor correction (per-window feet / foot-derived clip-wide hands, phasic-v2 M2): "
          f"ankle_clearance={ankle_clearance:.4f} alex_floor_z={alex_floor_z}  "
          f"corrected-frame counts={n_windowed}")

    floor_kwargs = {}
    floor_phase_w = np.ones(len(frame_ids))
    if args.floor_weight > 0.0:
        if alex_floor_z is None:
            print("  [warn] --floor-weight set but no foot-contact windows in this clip — "
                  "floor repulsion plane falling back to z=0.0")
            alex_floor_z = 0.0
        data.mocap_pos[floor_mocap_id] = [0.0, 0.0, alex_floor_z]
        floor_kwargs = dict(
            floor_margin=args.floor_margin, floor_gain=args.floor_gain,
        )
        print(f"Floor repulsion (Stage-3 QP term): ON weight={args.floor_weight} "
              f"margin={args.floor_margin}")
        if args.floor_phase_aware:
            planted_any = np.zeros(len(frame_ids), dtype=bool)
            for eff in FOOT_POS_ROLE:
                planted_any |= contacts_solved[eff]
            pelvis_z_target = np.array([
                make_targets_for_frame(src_positions[frame_ids[t]], role_to_idx, first_src_pos,
                                       target_rest_positions, root_scale, role_scales)["pelvis"][2]
                for t in range(len(frame_ids))
            ])
            floor_phase_w = floor_phase_weight(pelvis_z_target, planted_any)
            print(f"Floor phase weight: min={floor_phase_w.min():.2f} max={floor_phase_w.max():.2f} "
                  f"active(>=0.5)={(floor_phase_w >= 0.5).mean() * 100:.1f}% of frames")

    rest_errs = {
        role: float(np.linalg.norm(initial_targets[role] - data.xpos[bid]))
        for role, bid in role_to_body_id.items()
    }
    print("Initial rest-alignment errors:")
    for role, err in rest_errs.items():
        print(f"  {role:12s} {err:.4f}")
    print(f"  mean={np.mean(list(rest_errs.values())):.4f} max={np.max(list(rest_errs.values())):.4f}")
    print("Per-role scales:")
    for role, s in role_scales.items():
        print(f"  {role:14s} {s:.4f}")

    if contact_first:
        print(f"\nContact-first: ENABLED  (floor_z={floor_z:.3f} m, "
              f"foot_h<{args.foot_contact_height} hand_h<{args.hand_contact_height} "
              f"speed<{args.contact_speed})")
        for eff in CONTACT_EFFECTORS:
            pct = contacts[eff][frame_ids].mean() * 100.0
            print(f"  {eff:11s} in contact {pct:5.1f}% of solved frames")
    else:
        print("\nContact-first: DISABLED (--no-contact-first)")

    # Per-effector frozen anchor for the planted-foot position hold. Set when a
    # foot commits to the ground (w_env >= foot_hold_latch), cleared on release.
    foot_hold_anchor = {}

    # hierarchical-v1 H2 (--hard-tier) per-frame diagnostics: how far the hard
    # tier's own tasks (foot hold, floor) landed from where they were asked to
    # be, adapted from plan.md's OSQP-slack-and-log sketch to this solver's
    # actual damped-least-squares architecture (see solve_frame_position_ik's
    # `diag_out` docstring). Empty/unused unless args.hard_tier.
    hard_tier_frame_diag = {}
    hard_tier_floor_pen_cm = []
    hard_tier_hold_slip_cm = []

    # Frames where the knee target needed the shank-tilt feasibility projection.
    shank_clamp_count = {eff: 0 for eff in FOOT_KNEE_ROLE}

    # Gate canary (phasic-v2 M2/T2.1): count/depth of any contacting effector's
    # FINAL target Z landing below its own floor_target_z reference. Should
    # always be 0 given the max() clamps in the loop below.
    floor_violation_count = [0]
    floor_violation_max_depth = [0.0]

    # Per-frame constraint cache for the floor-transition arm refinement pass
    # (refine_arm_floor_transitions, run after this loop) — each entry is
    # reused VERBATIM rather than recomputed, since target construction has
    # cross-frame state (foot_hold_anchor) only this sequential loop tracks
    # correctly.
    frame_cache = []

    for ti, src_i in enumerate(frame_ids):
        targets = make_targets_for_frame(
            src_positions[src_i],
            role_to_idx,
            first_src_pos,
            target_rest_positions,
            root_scale,
            role_scales,
        )
        # Snapshot BEFORE the contact machinery edits targets (foot-hold freeze,
        # shank-tilt clamp): the pure morphology-scaled human. Saved separately so
        # the renderer can overlay "what the human did" vs "what IK aimed at".
        human_targets = {r: t.copy() for r, t in targets.items()}

        ori_targets = make_orientation_targets_for_frame(
            orientation_mats[src_i],
            ori_to_idx,
            first_src_ori,
            target_rest_orientations,
        )

        # Swing-foot toe-clearance (--swing-clear). A swing (airborne) foot tracks
        # the human's world-delta orientation at full weight; humans plantarflex
        # ~20-25 deg toe-down on a low get-up step, and on Alex's rigid ~20cm foot
        # plate that pitch drives the toe corner up to ~18cm through the floor while
        # the ankle itself sits above it (measured across the get-up clip class; the
        # ankle is NOT saturated -- 30-78 deg of unused dorsiflexion headroom). Cap
        # the toe-down pitch of the foot orientation TARGET on airborne frames so the
        # IK spends that headroom to lift the toe rather than copy the human. Ramped
        # by the SAME contact envelope (lift = swing-ness = 1 - w_env): full cap
        # airborne, fades to 0 as the foot plants (where the flat-align term below
        # takes over). Feet only. Runs before frame_cache/solve so the arm-refine
        # pass and the saved target orientation both see the capped target.
        swing_clear_sites = []
        swing_depitch_active = False   # any foot de-pitched this frame (for continuity reg)
        swing_depitch_lift = 0.0       # max de-pitch strength this frame (ramps continuity reg)
        if args.swing_clear and contact_first:
            for _eff, _cfg in CONTACT_EFFECTORS.items():
                if _eff not in FOOT_POS_ROLE:
                    continue
                _swing = 1.0 - float(contact_env[_eff][ti])
                if _swing <= 0.0:
                    continue
                # Proximity gate: only de-pitch a swing foot that is actually LOW
                # (near the floor, i.e. digging or about to). A foot held HIGH with a
                # steep toe-down pitch (e.g. ankle 15cm up, heel 27cm -- not digging)
                # must NOT be flattened: doing so demands a large leg reconfiguration
                # (drop the heel 20cm+) that flips the per-frame IK to a different
                # branch (measured: a surviving velocity spike). Gate on where the foot
                # ACTUALLY IS -- the PREVIOUS frame's achieved ankle height (data holds
                # it here, pre-solve) -- NOT the ankle TARGET: the worst digs are a
                # tracking SAG (target ankle high, achieved ankle 7-11cm lower, see
                # collisionFixPlan.md), which a target-height gate would wrongly read as
                # "high, skip". `_h` ~ the sole height above the floor (achieved ankle
                # minus the ankle-above-sole clearance minus the floor). prox ramps
                # 1 (foot on/below floor) -> 0 (foot >= swing_clear_height above it).
                if alex_floor_z is not None:
                    _ank_bid = role_to_body_id[FOOT_POS_ROLE[_eff]]
                    _h = (float(data.xpos[_ank_bid][2]) - alex_floor_z) - ankle_clearance
                    # Two-threshold ramp: FULL cap for the whole dig band (foot low,
                    # incl. the descent into a plant), hard 0 above it (foot genuinely
                    # high, e.g. frame 243). A single-threshold linear ramp left the
                    # descent frames only partially capped, so the dig developed before
                    # the cap reached full strength. `swing_clear_height` = the zero
                    # height; full strength kicks in `swing_clear_band` below it.
                    _h_zero = args.swing_clear_height
                    _h_full = max(_h_zero - args.swing_clear_band, 0.0)
                    _prox = float(np.clip((_h_zero - _h) / max(_h_zero - _h_full, 1e-6), 0.0, 1.0))
                else:
                    _prox = 1.0
                _lift = _swing * _prox
                if _lift <= 0.0:
                    continue
                swing_depitch_active = True
                swing_depitch_lift = max(swing_depitch_lift, _lift)
                # (1) de-pitch: cap the toe-down pitch of the foot orientation target,
                # ramped by swing-ness AND floor proximity.
                _orole = _cfg["ori_role"]
                ori_targets[_orole] = cap_foot_pitch(
                    ori_targets[_orole], np.radians(args.swing_max_pitch), _lift)
                # (2) soft clearance (optional, default off -- weight 0): one-sided
                # non-penetration rows on this foot's sole corners. Same proximity/
                # swing ramp. Kept in the code but off by default; the gated pitch cap
                # alone is the effective, stable mechanism (the soft term fought the
                # plant machinery in testing -- see planLog).
                if args.swing_clear_weight > 0.0 and alex_floor_z is not None:
                    _cz = alex_floor_z + args.swing_clear_margin
                    _w = args.swing_clear_weight * _lift
                    for _sid in sole_corner_sids[_eff]:
                        swing_clear_sites.append((_sid, _cz, _w))

        # Contact-first: build the foot-flat / fist-down axis-alignment
        # constraints active this frame, and suppress the matching human
        # world-delta orientation terms.
        align_constraints = []
        skip_ori_body_ids = set()
        pos_site_constraints = []
        skip_pos_roles = set()
        ori_weight_scale = {}
        pos_weight_scale = {}
        hold_pos_roles = set()   # ankle roles under an active planted-foot hold
        frame_contacts = {}
        palm_targets = {}
        if contact_first:
            src_pelvis0 = first_src_pos[role_to_idx["pelvis"]]
            src_pelvis = src_positions[src_i][role_to_idx["pelvis"]]
            root_delta = root_scale * (src_pelvis - src_pelvis0)
            for eff, cfg in CONTACT_EFFECTORS.items():
                w_env = float(contact_env[eff][ti])   # cross-fade weight in [0,1]
                # Report as "in contact" once the effector is at least half engaged
                # (used for diagnostics/renderer); constraints themselves are scaled
                # continuously by w_env so they ease in/out with no snap.
                frame_contacts[eff] = w_env >= 0.5
                if w_env <= 0.0:
                    foot_hold_anchor.pop(eff, None)   # released: drop the frozen anchor
                    continue
                bid = effector_body_id[eff]

                # Foot floor clamp (phasic-v2 M2/T2.1, generalized Fix B — see
                # floor_target_z computation above): correct the ankle position
                # target UP to this window's floor-target Z if it maps below.
                # Applied BEFORE foot-hold captures its anchor below, so a
                # held plant's frozen anchor is already floor-correct instead
                # of freezing a still-penetrating pose.
                if eff in FOOT_POS_ROLE:
                    pr0 = FOOT_POS_ROLE[eff]
                    fz0 = floor_target_z[eff][ti]
                    if not np.isnan(fz0):
                        targets[pr0][2] = max(targets[pr0][2], fz0)

                # Planted-foot position hold: freeze the foot position target (the
                # ankle role) at the pose where the foot commits to the ground, and
                # cross-fade the moving human target onto it, so a flat foot stops
                # sliding. While the foot is still descending (w_env < latch) the
                # anchor keeps tracking the target; once committed it latches/holds.
                if args.foot_hold and eff in FOOT_POS_ROLE:
                    pr = FOOT_POS_ROLE[eff]
                    if eff not in foot_hold_anchor or w_env < args.foot_hold_latch:
                        foot_hold_anchor[eff] = targets[pr].copy()
                    targets[pr] = (1.0 - w_env) * targets[pr] + w_env * foot_hold_anchor[eff]
                    # Boost the held foot's position weight so the frozen anchor is
                    # not dragged off by the heavier pelvis/chain targets.
                    boost = 1.0 + (args.foot_hold_weight - 1.0) * w_env
                    pos_weight_scale[pr] = max(pos_weight_scale.get(pr, 1.0), boost)
                    hold_pos_roles.add(pr)   # promote to level-1 in the hierarchical solve

                # Shank-tilt clamp: with the foot flat, only shank tilts inside
                # the ankle joint range are reachable — project the knee target
                # into that region (about the held ankle target, along the human
                # foot heading) and cross-fade the projection in with the contact.
                # Human targets past the range are *kinematically infeasible*
                # flat-footed; retargeting yields there by design (RL won't fix
                # an impossible kinematic demand, so the target must).
                if args.shank_clamp and eff in FOOT_KNEE_ROLE:
                    fwd_h = ori_targets[cfg["ori_role"]] @ np.array([1.0, 0.0, 0.0])
                    n_h = float(np.linalg.norm(fwd_h[:2]))
                    if n_h > 0.1:   # heading undefined if the human foot points near-vertical
                        kr = FOOT_KNEE_ROLE[eff]
                        pitch_rng, roll_rng = shank_limits[eff]
                        knee_c, was_clamped = clamp_shank_tilt(
                            targets[FOOT_POS_ROLE[eff]], targets[kr],
                            fwd_h[:2] / n_h, pitch_rng, roll_rng,
                        )
                        if was_clamped:
                            targets[kr] = (1.0 - w_env) * targets[kr] + w_env * knee_c
                            shank_clamp_count[eff] += 1

                # Foot-flat weight is kept above the position hold so flatness stays
                # (near-)dominant: the hold must not tilt the planted foot.
                flat_w = args.foot_flat_weight if eff in FOOT_POS_ROLE else CONTACT_ALIGN_WEIGHT[eff]
                align_constraints.append(
                    (bid, cfg["axis_local"], cfg["world_dir"], flat_w * w_env,
                     eff in FOOT_POS_ROLE)   # feet hard (level 1), hands soft
                )
                # Cross-fade the human world-delta orientation out as the contact
                # constraint fades in (weight *= 1-w_env), so the foot eases from
                # its human pose onto flat instead of the target snapping off.
                obid = ori_role_body_id[cfg["ori_role"]]
                ori_weight_scale[obid] = min(ori_weight_scale.get(obid, 1.0), 1.0 - w_env)

                # Foot heading (yaw) align: flat pins pitch/roll but leaves yaw a
                # free DOF, so a planted foot free-drifts in-plane (inner/outer
                # rotation slip). Drive the foot forward axis (+X) to the HUMAN
                # foot heading (ground-projected), so it follows the human's small
                # heading change instead of the leg chain spinning it freely.
                # Hands keep yaw free (fist support face doesn't need it).
                if "foot" in eff and args.foot_yaw_weight > 0.0:
                    fwd_tgt = ori_targets[cfg["ori_role"]] @ np.array([1.0, 0.0, 0.0])
                    fwd_xy = np.array([fwd_tgt[0], fwd_tgt[1], 0.0])
                    n_xy = np.linalg.norm(fwd_xy)
                    if n_xy > 0.1:   # skip if the human foot points near-vertical
                        align_constraints.append(
                            (bid, np.array([1.0, 0.0, 0.0]), fwd_xy / n_xy,
                             args.foot_yaw_weight * w_env, True)
                        )

                # Hand contact also pins the palm site to the human hand location
                # (and suppresses the now-redundant wrist-body position target).
                if eff in CONTACT_POS:
                    cpos = CONTACT_POS[eff]
                    mk = cpos["marker"]
                    if mk in role_to_idx:
                        rel0 = first_src_pos[role_to_idx[mk]] - src_pelvis0
                        rel = src_positions[src_i][role_to_idx[mk]] - src_pelvis
                        tgt = palm_rest_pos[eff] + root_delta + palm_pos_scale[eff] * (rel - rel0)
                        # Floor clamp (phasic-v2 M2/T2.1, generalized Fix B):
                        # the scaled human hand-contact target can map below
                        # Alex's floor (measured: 6.5-7.2cm on
                        # luigi_standProne_03's push phase) — the palm pin
                        # then faithfully tracks the fist underground. Palm
                        # site IS the support surface, so clearance 0 is
                        # correct (no margin). PER-WINDOW now (floor_target_z
                        # computed above), not a clip-wide scalar. Independent
                        # of --floor-weight.
                        fz = floor_target_z[eff][ti]
                        if not np.isnan(fz):
                            tgt[2] = max(tgt[2], fz)
                        pos_site_constraints.append(
                            (effector_site_id[eff], tgt, CONTACT_POS_WEIGHT * w_env)
                        )
                        # Cross-fade the wrist-body position target out as the palm
                        # pin fades in (weight *= 1-w_env).
                        spr = cpos["skip_pos_role"]
                        pos_weight_scale[spr] = min(pos_weight_scale.get(spr, 1.0), 1.0 - w_env)
                        palm_targets[eff] = tgt

            # Coplanar planted feet. The retargeted foot-height targets can sit
            # several cm apart in Z while BOTH feet are contact-labelled (the source
            # ankles differ in height relative to the pelvis, or the per-leg
            # morphology scale differs). That is an inconsistent input — "both
            # planted" yet not coplanar — which the downstream rigid grounding shift
            # cannot reconcile: it plants only the lower foot and the higher one
            # floats (standup_02: ankle targets 5.78 cm apart → a foot off the ground
            # in RDX). Fix it at the target: when both feet are engaged, snap the
            # higher ankle target DOWN to the lower (grounded) one, cross-faded by
            # the weaker engagement so it eases in with contact. Foot-flat makes
            # equal ankle Z ⇒ equal sole Z, so the IK then produces coplanar feet
            # directly — no post-hoc, reach-limited patch needed.
            if args.coplanar_feet_mode != "off" \
                    and "left_foot" in contact_env and "right_foot" in contact_env:
                lpr = FOOT_POS_ROLE["left_foot"]; rpr = FOOT_POS_ROLE["right_foot"]
                wcp = min(float(contact_env["left_foot"][ti]),
                          float(contact_env["right_foot"][ti]))
                if wcp > 0.0:
                    zl = targets[lpr][2]; zr = targets[rpr][2]
                    z_common = min(zl, zr) if args.coplanar_feet_mode == "min" \
                        else 0.5 * (zl + zr)
                    targets[lpr][2] = (1.0 - wcp) * zl + wcp * z_common
                    targets[rpr][2] = (1.0 - wcp) * zr + wcp * z_common
                    # Re-clamp EACH foot to its OWN floor_target_z after the
                    # snap (phasic-v2 M2/T2.1): coplanar averaging can pull a
                    # foot below its own per-window floor reference if the two
                    # feet sit in different postural-phase windows (e.g. one
                    # leg mid-transition while the other is settled) — the
                    # exact invariant this milestone's gate requires ("no
                    # contact target ever below floor") must hold AFTER every
                    # target-construction step, not just the first clamp.
                    for eff2, pr2 in (("left_foot", lpr), ("right_foot", rpr)):
                        fz2 = floor_target_z[eff2][ti]
                        if not np.isnan(fz2):
                            targets[pr2][2] = max(targets[pr2][2], fz2)

        # Gate check (phasic-v2 M2/T2.1, "no contact target ever below floor"):
        # verify every contacting effector's FINAL target Z (after foot-hold,
        # shank-clamp, coplanar-snap — everything that can touch it) is at or
        # above its own floor reference. Should always hold given the max()
        # clamps above; this is a canary against a future edit reordering
        # those clamps, not a live correction path.
        for eff2, fz_arr in floor_target_z.items():
            fz2 = fz_arr[ti]
            if np.isnan(fz2):
                continue
            pr2 = FOOT_POS_ROLE.get(eff2)
            z_final = targets[pr2][2] if pr2 is not None else palm_targets.get(eff2, [None, None, None])[2]
            if z_final is not None and z_final < fz2 - 1e-6:
                floor_violation_count[0] += 1
                floor_violation_max_depth[0] = max(floor_violation_max_depth[0], float(fz2 - z_final))

        frame_cache.append((targets, ori_targets, align_constraints, pos_site_constraints,
                           skip_pos_roles, skip_ori_body_ids, ori_weight_scale,
                           pos_weight_scale, hold_pos_roles))

        # Self-collision guard for the swing-clear de-pitch: on de-pitch frames the
        # continuity reg holds the hip/knee near the previous pose, which can roughly
        # cancel the (soft, weight-20) self-collision repulsion on those DOFs -- e.g.
        # the thigh jamming against the torso on a kneeling get-up (measured: RIGHT_
        # THIGH<->TORSO, the dominant regression). Scale self-collision weight UP with
        # the same de-pitch strength so repulsion always dominates the hold. No-op off
        # de-pitch frames (swing_depitch_lift = 0).
        frame_coll_kwargs = coll_kwargs
        if swing_depitch_lift > 0.0 and args.swing_coll_boost > 0.0:
            frame_coll_kwargs = dict(coll_kwargs)
            frame_coll_kwargs["coll_weight"] = coll_kwargs["coll_weight"] * (
                1.0 + args.swing_coll_boost * swing_depitch_lift)

        q = solve_frame_position_ik(
            model,
            data,
            role_to_body_id,
            targets,
            q,
            ori_role_to_body_id=ori_role_to_body_id,
            ori_targets=ori_targets,
            ori_scale=args.ori_scale,
            align_constraints=align_constraints,
            skip_ori_body_ids=skip_ori_body_ids,
            ori_weight_scale=ori_weight_scale,
            pos_weight_scale=pos_weight_scale,
            hold_pos_roles=hold_pos_roles,
            hierarchical=(args.hierarchical or args.hard_tier),
            knee_bias=knee_bias,
            pos_site_constraints=pos_site_constraints,
            skip_pos_roles=skip_pos_roles,
            iters=args.ik_iters,
            **frame_coll_kwargs,
            **floor_kwargs,
            floor_weight=(args.floor_weight * float(floor_phase_w[ti]) if floor_kwargs else 0.0),
            floor_hard=args.floor_hard_flag,
            swing_clear_sites=swing_clear_sites,
            posture_reg=_swing_posture_reg(model.nv, leg_cont_dofs,
                                           args.swing_continuity_reg * swing_depitch_lift),
            diag_out=hard_tier_frame_diag,
        )
        if args.hard_tier or args.floor_hard_flag:
            hard_tier_floor_pen_cm.append(hard_tier_frame_diag.get("floor_pen_cm", 0.0))
            hard_tier_hold_slip_cm.append(hard_tier_frame_diag.get("hold_slip_cm", 0.0))

        mujoco.mj_forward(model, data)

        # Count remaining self-collisions post-solve (using same filter as the
        # constraint). Floor contacts are excluded here too — the injected
        # floor body's id is never 0, so the old cb1>0/cb2>0 filter alone
        # doesn't catch it (same leak class as self_collision_rows/
        # floor_collision_rows; see _load_model_with_floor).
        n_self_coll = 0
        n_floor_pen = 0
        for c_idx in range(data.ncon):
            ct = data.contact[c_idx]
            is_floor = ct.geom1 == floor_gid or ct.geom2 == floor_gid
            if is_floor:
                if ct.dist < 0:
                    n_floor_pen += 1
                continue
            cb1 = int(model.geom_bodyid[ct.geom1])
            cb2 = int(model.geom_bodyid[ct.geom2])
            if cb1 > 0 and cb2 > 0 and ct.dist < 0:
                if not _within_k_hops(model, cb1, cb2, args.coll_hops):
                    n_self_coll += 1

        errs = {}
        for role, bid in role_to_body_id.items():
            e = np.linalg.norm(targets[role] - data.xpos[bid])
            errs[role] = float(e)

        qpos_list.append(q.copy())
        err_list.append(errs)
        n_self_coll_list.append(n_self_coll)
        n_floor_pen_list.append(n_floor_pen)

        role_order = list(ROLE_TO_ALEX_BODY.keys())
        target_pos_list.append(np.asarray([targets[r] for r in role_order], dtype=np.float64))
        human_target_pos_list.append(np.asarray([human_targets[r] for r in role_order], dtype=np.float64))
        achieved_pos_list.append(np.asarray([data.xpos[role_to_body_id[r]].copy() for r in role_order], dtype=np.float64))

        ori_order = list(ORI_TO_ALEX_BODY.keys())
        target_ori_list.append(np.asarray([ori_targets[r] for r in ori_order], dtype=np.float64))
        achieved_ori_list.append(np.asarray([body_xmat(data, ori_role_to_body_id[r]).copy() for r in ori_order], dtype=np.float64))

        ori_errs = []
        for r in ori_order:
            R_current = body_xmat(data, ori_role_to_body_id[r])
            R_target = ori_targets[r]
            R_err = R_target @ R_current.T
            rv = rotmat_to_rotvec(R_err)
            ori_errs.append(np.linalg.norm(rv) * 180.0 / np.pi)
        ori_err_deg_list.append(np.asarray(ori_errs, dtype=np.float64))

        # Contact diagnostics: per-effector flag + achieved alignment error
        # (angle between the driven axis and its world target) in degrees.
        eff_order = list(CONTACT_EFFECTORS.keys())
        flags = np.array([frame_contacts.get(e, False) for e in eff_order], dtype=bool)
        contact_flags_list.append(flags)
        align_errs = []
        for eff in eff_order:
            cfg = CONTACT_EFFECTORS[eff]
            R_cur = body_xmat(data, effector_body_id[eff])
            a_world = R_cur @ cfg["axis_local"]
            cosang = float(np.clip(np.dot(a_world, cfg["world_dir"]), -1.0, 1.0))
            align_errs.append(np.degrees(np.arccos(cosang)))
        contact_align_err_deg_list.append(np.asarray(align_errs, dtype=np.float64))

        # Palm-pin position error (does the fist reach the human contact point?)
        palm_pos_err = {}
        for eff, tgt in palm_targets.items():
            palm_pos_err[eff] = float(np.linalg.norm(data.site_xpos[effector_site_id[eff]] - tgt))

        mean_err = np.mean(list(errs.values()))
        max_err = np.max(list(errs.values()))
        coll_str = f"  coll={n_self_coll}" if n_self_coll > 0 else ""
        # Only meaningful once the floor plane is actually positioned
        # (--floor-weight > 0) — otherwise it sits at its unused default and
        # this count is noise.
        if args.floor_weight > 0.0:
            coll_str += f"  floor_pen={n_floor_pen}" if n_floor_pen > 0 else ""
        active = [e for e in eff_order if frame_contacts.get(e, False)]
        con_str = ""
        if active:
            angs = " ".join(f"{e}:{align_errs[eff_order.index(e)]:.0f}deg"
                            + (f"/{palm_pos_err[e]*100:.0f}cm" if e in palm_pos_err else "")
                            for e in active)
            con_str = f"  contact[{angs}]"
        if ti % args.log_every == 0 or ti == len(frame_ids) - 1:
            print(f"frame {ti:04d} source={src_i:04d} mean_err={mean_err:.4f} max_err={max_err:.4f}{coll_str}{con_str}")

    # Floor-transition arm refinement (Pass 2) — see refine_arm_floor_transitions
    # docstring / collisionFixPlan.md. Only meaningful once floor avoidance is
    # active; inert (and skipped) otherwise.
    if args.floor_weight > 0.0 and args.floor_refine:
        qpos_pass1 = np.asarray(qpos_list)
        windows = _detect_arm_floor_onset_windows(
            model, data, qpos_pass1, floor_gid,
            args.floor_refine_preroll, args.floor_refine_ramp)
        if windows:
            qpos_refined = refine_arm_floor_transitions(
                model, data, qpos_pass1, windows, frame_cache,
                role_to_body_id, ori_role_to_body_id,
                args.floor_weight, args.floor_margin, args.floor_gain,
                coll_kwargs,
                arm_posture_reg=args.floor_refine_posture_reg,
                lock_weight=args.floor_refine_lock_weight,
                iters=args.ik_iters,
            )
            qpos_list = list(qpos_refined)
        else:
            print("  [floor-refine] no floor-onset transitions detected — nothing to refine")

    # Floor-transition LEG refinement (Pass 3) — see refine_leg_floor_transitions
    # docstring. Independent of --floor-weight/--floor-refine (geometric
    # detection, not gated on the repulsion mechanism being on); runs after the
    # arm pass so it sees any arm-refine correction already applied.
    if args.leg_floor_refine:
        qpos_pass2 = np.asarray(qpos_list)
        qpos_refined2 = refine_leg_floor_transitions(
            model, data, qpos_pass2, frame_cache,
            role_to_body_id, ori_role_to_body_id,
            alex_floor_z if alex_floor_z is not None else 0.0,
            ankle_clearance,
            coll_kwargs,
            pen_tol=args.leg_floor_refine_pen_tol,
            ramp=args.leg_floor_refine_ramp,
            preroll=args.leg_floor_refine_preroll,
            leg_posture_reg=args.leg_floor_refine_posture_reg,
            lock_weight=args.leg_floor_refine_lock_weight,
            root_pos_relief=args.leg_floor_refine_root_relief,
            foot_flat_weight=args.leg_floor_refine_flat_weight,
            iters=args.ik_iters,
        )
        qpos_list = list(qpos_refined2)

    metadata = {
        "format": "alex_contactfirst_v1",
        "canonical": str(args.canonical),
        "model": str(args.model),
        "role_to_alex_body": ROLE_TO_ALEX_BODY,
        "target_weights": TARGET_WEIGHTS,
        "orientation_cost": args.ori_scale,
        "ori_to_alex_body": ORI_TO_ALEX_BODY,
        "ori_weights": ORI_WEIGHTS,
        "root_scale": root_scale,
        "role_scales": role_scales,
        "alex_pelvis_to_head": alex_pelvis_to_head,
        "frame_ids": frame_ids,
        "contact_first": contact_first,
        "contact_effectors": list(CONTACT_EFFECTORS.keys()),
        "contact_align_weight": CONTACT_ALIGN_WEIGHT,
        "contact_pos_weight": CONTACT_POS_WEIGHT,
        "contact_pos_sites": {e: c["site"] for e, c in CONTACT_POS.items()},
        "contact_floor_z": floor_z,
        "contact_params": {
            "foot_height": args.foot_contact_height,
            "hand_height": args.hand_contact_height,
            "speed": args.contact_speed,
            "on_height_frac": args.contact_on_height_frac,
            "on_speed_frac": args.contact_on_speed_frac,
            "onset_max_delay": args.contact_onset_max_delay,
        },
        "shank_clamp": args.shank_clamp,
        "shank_margin_deg": args.shank_margin_deg,
        "shank_clamp_frames": shank_clamp_count,
        "knee_bias_weight": args.knee_bias_weight,
        "knee_min_flex_deg": args.knee_min_flex_deg,
        "hard_tier": args.hard_tier,
        "floor_hard": args.floor_hard_flag,
        "hard_tier_floor_pen_max_cm": float(max(hard_tier_floor_pen_cm) if hard_tier_floor_pen_cm else 0.0),
        "hard_tier_hold_slip_max_cm": float(max(hard_tier_hold_slip_cm) if hard_tier_hold_slip_cm else 0.0),
        "notes": [
            "Contact-first QP IK on Alex V2.",
            "Foot contact -> foot up-axis forced to world +Z (flat); hand contact ->"
            " gripper +X palm-normal forced to world -Z (closed-fist palm/finger support).",
            "Contact detected from human marker height above clip floor + low speed.",
            "Human world-delta orientation suppressed on an effector while it is in contact.",
            "Intermediate segments (upperarm/forearm/shin) untracked in orientation.",
        ],
    }

    n_coll_arr = np.asarray(n_self_coll_list, dtype=np.int32)
    print(f"\nSelf-collision summary: {n_coll_arr.sum()} total penetrating frames "
          f"({(n_coll_arr > 0).mean() * 100:.1f}% of frames)")
    if args.floor_weight > 0.0:
        n_floor_arr = np.asarray(n_floor_pen_list, dtype=np.int32)
        print(f"Floor-penetration summary: {n_floor_arr.sum()} total penetrating contacts "
              f"({(n_floor_arr > 0).mean() * 100:.1f}% of frames)")
    if args.shank_clamp:
        cc = " ".join(f"{e}:{n}" for e, n in shank_clamp_count.items())
        print(f"Shank-tilt clamp engaged (frames): {cc}")
    if floor_violation_count[0] > 0:
        print(f"  [WARNING] floor-invariant gate (phasic-v2 M2): {floor_violation_count[0]} "
              f"contacting-effector target(s) landed below their floor reference "
              f"(max depth {floor_violation_max_depth[0]*100:.2f}cm) -- should be 0, investigate.")
    else:
        print("Floor-invariant gate (phasic-v2 M2): PASS -- 0 contacting-effector targets "
              "below their floor reference.")

    if args.hard_tier or args.floor_hard_flag:
        fp = np.asarray(hard_tier_floor_pen_cm) if hard_tier_floor_pen_cm else np.zeros(1)
        hs = np.asarray(hard_tier_hold_slip_cm) if hard_tier_hold_slip_cm else np.zeros(1)
        n_floor_viol = int((fp > 0.1).sum())
        n_hold_viol = int((hs > 0.5).sum())
        print(f"Hard-tier (H2) diagnostics: floor_pen max={fp.max():.2f}cm "
              f"({n_floor_viol} frames > 0.1cm tol)  |  hold_slip max={hs.max():.2f}cm "
              f"({n_hold_viol} frames > 0.5cm tol)")

    np.savez(
        args.out,
        qpos=np.asarray(qpos_list),
        fps=np.float64(fps),
        self_collision_counts=n_coll_arr,
        target_positions=np.asarray(target_pos_list),
        human_target_positions=np.asarray(human_target_pos_list),
        achieved_positions=np.asarray(achieved_pos_list),
        orientation_role_names=np.asarray(list(ORI_TO_ALEX_BODY.keys()), dtype=object),
        target_orientations=np.asarray(target_ori_list),
        achieved_orientations=np.asarray(achieved_ori_list),
        orientation_errors_deg=np.asarray(ori_err_deg_list),
        source_frame_ids=np.asarray(frame_ids, dtype=np.int64),
        role_names=np.asarray(list(ROLE_TO_ALEX_BODY.keys()), dtype=object),
        alex_body_names=np.asarray(list(ROLE_TO_ALEX_BODY.values()), dtype=object),
        contact_effector_names=np.asarray(list(CONTACT_EFFECTORS.keys()), dtype=object),
        contact_flags=np.asarray(contact_flags_list, dtype=bool),
        contact_align_errors_deg=np.asarray(contact_align_err_deg_list, dtype=np.float64),
        errors_json=np.asarray(json.dumps(err_list, indent=2), dtype=object),
        metadata_json=np.asarray(json.dumps(metadata, indent=2), dtype=object),
    )

    print()
    print("Wrote:", args.out)
    print("qpos:", np.asarray(qpos_list).shape)


if __name__ == "__main__":
    main()

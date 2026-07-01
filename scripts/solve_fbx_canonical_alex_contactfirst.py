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


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites_v2.xml"

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
# Each end effector that can take ground support. `markers` are canonical human
# roles used to detect contact; `body` is the Alex body whose axis we align;
# `axis_local` is that body's local axis to drive; `world_dir` is where it must
# point during contact; `ori_role` is the world-delta orientation term to
# suppress while in contact (the contact constraint replaces it).
CONTACT_EFFECTORS = {
    "left_foot": dict(
        markers=["left_ankle", "left_toe"],
        body="LEFT_FOOT",
        axis_local=np.array([0.0, 0.0, 1.0]),   # foot up-axis
        world_dir=np.array([0.0, 0.0, 1.0]),    # -> world +Z (flat)
        ori_role="left_foot",
        # Only treat as a flat support when the *human* foot is itself near-flat:
        # the canonical foot frame's local-Z (sole normal) within tilt of world +Z.
        flat_ori_role="left_foot",
    ),
    "right_foot": dict(
        markers=["right_ankle", "right_toe"],
        body="RIGHT_FOOT",
        axis_local=np.array([0.0, 0.0, 1.0]),
        world_dir=np.array([0.0, 0.0, 1.0]),
        ori_role="right_foot",
        flat_ori_role="right_foot",
    ),
    "left_hand": dict(
        markers=["left_wrist", "left_hand_middle"],
        body="LEFT_GRIPPER_Z_LINK",
        axis_local=np.array([1.0, 0.0, 0.0]),   # gripper +X = palm/finger-front normal
        world_dir=np.array([0.0, 0.0, -1.0]),   # -> world -Z (press down)
        ori_role="left_hand",
    ),
    "right_hand": dict(
        markers=["right_wrist", "right_hand_middle"],
        body="RIGHT_GRIPPER_Z_LINK",
        axis_local=np.array([1.0, 0.0, 0.0]),
        world_dir=np.array([0.0, 0.0, -1.0]),
        ori_role="right_hand",
    ),
}

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


def detect_contacts_from_human(positions, role_to_idx, fps, *,
                               orientation_mats=None, ori_to_idx=None,
                               foot_height=0.07, hand_height=0.08,
                               speed_thresh=0.4, foot_flat_tilt=40.0, floor_pct=1.0):
    """Per-frame ground-contact flags for each effector, from human mocap.

    An effector is "in contact" at frame t when the lowest of its markers is
    within `*_height` metres of the clip floor AND moving slower than
    `speed_thresh` (m/s). The floor is the low percentile of the feet markers'
    height across the whole clip.

    For effectors with a `flat_ori_role`, contact additionally requires the
    *human* segment to be near-flat: its canonical frame local-Z (sole normal)
    within `foot_flat_tilt` degrees of world +Z. This distinguishes a flat
    plantar support from a foot that is merely near the floor while folded
    (toes/side down during a get-up), where forcing the robot foot flat would
    just fight position tracking.

    Returns: (dict effector -> bool array (N,), floor_z).
    """
    N = positions.shape[0]
    dt = 1.0 / float(fps)

    def marker_z(role):
        return positions[:, role_to_idx[role], 2]

    def marker_speed(role):
        p = positions[:, role_to_idx[role], :]
        v = np.zeros(N)
        v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
        v[0] = v[1] if N > 1 else 0.0
        return v

    # Floor estimate from the feet markers (lowest they reach).
    foot_roles = [r for eff in ("left_foot", "right_foot")
                  for r in CONTACT_EFFECTORS[eff]["markers"] if r in role_to_idx]
    foot_min_z = np.min([marker_z(r) for r in foot_roles], axis=0)
    floor_z = float(np.percentile(foot_min_z, floor_pct))

    contacts = {}
    for eff, cfg in CONTACT_EFFECTORS.items():
        markers = [r for r in cfg["markers"] if r in role_to_idx]
        if not markers:
            contacts[eff] = np.zeros(N, dtype=bool)
            continue
        h = np.min([marker_z(r) for r in markers], axis=0) - floor_z
        spd = np.min([marker_speed(r) for r in markers], axis=0)
        hthr = foot_height if "foot" in eff else hand_height
        flag = (h < hthr) & (spd < speed_thresh)

        flat_role = cfg.get("flat_ori_role")
        if flat_role is not None and orientation_mats is not None and ori_to_idx is not None \
                and flat_role in ori_to_idx:
            up = orientation_mats[:, ori_to_idx[flat_role], :, 2]  # local-Z (sole normal)
            tilt = np.degrees(np.arccos(np.clip(np.abs(up @ np.array([0.0, 0.0, 1.0])), -1, 1)))
            flag = flag & (tilt < foot_flat_tilt)

        contacts[eff] = flag

    return contacts, floor_z


def debounce_flags(flag, min_run):
    """Remove ON/OFF runs shorter than min_run (fill gaps, drop specks).

    Kills marginal-threshold flicker without touching genuine long contacts."""
    if min_run <= 1:
        return flag.copy()
    out = flag.copy().astype(bool)
    n = len(out)
    # drop short ON specks, then fill short OFF gaps
    for target in (True, False):
        i = 0
        while i < n:
            j = i
            while j < n and out[j] == out[i]:
                j += 1
            if out[i] == target and (j - i) < min_run and not (i == 0 and j == n):
                out[i:j] = not target
            i = j
    return out


def ramp_envelope(flag, ramp, preroll):
    """Per-frame contact weight in [0,1] from a boolean contact timeline.

    `preroll` extends each contact earlier (anticipation: begin easing the foot/
    hand toward the support face before touchdown). `ramp` applies a cosine rise
    into each leading edge and fall out of each trailing edge, so the contact
    constraints cross-fade in/out instead of snapping at full weight."""
    n = len(flag)
    base = flag.copy().astype(bool)
    if preroll > 0:
        pr = base.copy()
        idx = np.where(base)[0]
        for i in idx:
            pr[max(0, i - preroll):i] = True
        base = pr
    env = base.astype(np.float64)
    if ramp > 0:
        def cosramp(k):   # k=1..ramp -> rises toward 1
            return 0.5 * (1.0 - np.cos(np.pi * k / (ramp + 1)))
        for i in range(n):
            if not base[i]:
                continue
            if i == 0 or not base[i - 1]:            # leading edge -> ramp preceding frames
                for k in range(1, ramp + 1):
                    p = i - k
                    if p >= 0 and not base[p]:
                        env[p] = max(env[p], cosramp(ramp - k + 1))
            if i == n - 1 or not base[i + 1]:        # trailing edge -> ramp following frames
                for k in range(1, ramp + 1):
                    p = i + k
                    if p < n and not base[p]:
                        env[p] = max(env[p], cosramp(ramp - k + 1))
    return env


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

    return roles, role_to_idx, positions, fps, orientation_roles, ori_to_idx, orientation_mats


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
    """
    rows: list[np.ndarray] = []
    rhs_vals: list[float] = []

    sqw = float(np.sqrt(weight))

    for c_idx in range(data.ncon):
        ct = data.contact[c_idx]
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
):
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)

    nv = model.nv
    q_ref = q_init.copy()

    for _ in range(iters):
        rows = []
        rhs = []

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

            rows.append(np.sqrt(weight) * jacp)
            rhs.append(np.sqrt(weight) * err)

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

                rows.append(np.sqrt(weight) * jacr)
                rhs.append(np.sqrt(weight) * err_rot)

        # Contact position pin: drive a site (palm contact patch) to a world
        # target — the fist support location during hand contact.
        if pos_site_constraints:
            for sid, target, weight in pos_site_constraints:
                cur = data.site_xpos[sid].copy()
                err = target - cur
                jacp = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacSite(model, data, jacp, None, sid)
                rows.append(np.sqrt(weight) * jacp)
                rhs.append(np.sqrt(weight) * err)

        # Contact axis-alignment: drive a body-local axis to a world direction
        # (foot up -> +Z; palm normal -> -Z). Cross product gives the world-frame
        # rotation vector that brings the current axis onto the target, leaving
        # spin about that axis free.
        if align_constraints:
            for bid, axis_local, world_dir, weight in align_constraints:
                R_current = body_xmat(data, bid)
                a_world = R_current @ axis_local
                err_rot = np.cross(a_world, world_dir)

                jacp = np.zeros((3, nv), dtype=np.float64)
                jacr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacBody(model, data, jacp, jacr, bid)

                rows.append(np.sqrt(weight) * jacr)
                rhs.append(np.sqrt(weight) * err_rot)

        # Pull actuated joints toward q_ref (start of this frame's solve).
        # Root DOFs (0-5) are left at zero — the position/orientation tasks steer them.
        # Actuated joints: dq[6:35] corresponds to qpos[7:36].
        desired_dq = np.zeros(nv)
        desired_dq[6:] = q_ref[7:] - data.qpos[7:]
        rows.append(np.sqrt(posture_reg) * np.eye(nv))
        rhs.append(np.sqrt(posture_reg) * desired_dq)

        # Self-collision repulsion: push apart any non-adjacent penetrating body pairs.
        if coll_weight > 0.0:
            coll_rows, coll_rhs = self_collision_rows(
                model, data, nv, coll_weight, coll_margin, coll_gain,
                exclude_hops=coll_hops,
            )
            rows.extend(coll_rows)
            rhs.extend([np.array([v]) for v in coll_rhs])

        A = np.vstack(rows)
        b = np.concatenate(rhs)

        H = A.T @ A + damping * np.eye(nv)
        g = A.T @ b

        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(H, g, rcond=None)[0]

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

    return data.qpos.copy()


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
    ap.add_argument("--foot-contact-height", type=float, default=0.07,
                    help="Foot marker height (m) above clip floor below which a foot is in contact (default: 0.07)")
    ap.add_argument("--hand-contact-height", type=float, default=0.08,
                    help="Hand marker height (m) above clip floor below which a hand is in contact (default: 0.08)")
    ap.add_argument("--contact-speed", type=float, default=0.4,
                    help="Marker speed (m/s) below which an effector can be in contact (default: 0.4)")
    ap.add_argument("--foot-flat-tilt", type=float, default=40.0,
                    help="Max tilt (deg) of the human foot sole-normal from vertical for a foot to count "
                         "as a flat plantar support (default: 40)")
    ap.add_argument("--foot-yaw-weight", type=float, default=1.5,
                    help="Weight for the foot heading (yaw) align during contact: drives the foot "
                         "forward axis to the HUMAN foot heading so the planted foot follows the "
                         "human's small heading change instead of free-drifting (inner/outer slip). "
                         "0 disables (default: 1.5)")
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
    ap.add_argument("--log-every", type=int, default=1,
                    help="Print a per-frame line every N solved frames (1=all). "
                         "Final frame + summary always printed. Use >1 to cut log volume.")
    ap.add_argument("--no-contact-first", action="store_true",
                    help="Disable contact-gated foot-flat / fist-down overrides (baseline behaviour)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    roles, role_to_idx, src_positions, fps, orientation_roles, ori_to_idx, orientation_mats = load_canonical(args.canonical)

    missing = [r for r in ROLE_TO_ALEX_BODY if r not in role_to_idx]
    if missing:
        raise RuntimeError(f"Canonical missing required roles: {missing}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

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

    # Contact-first: detect ground contact per effector from the human data,
    # and resolve the Alex bodies whose axes we align during contact.
    contact_first = not args.no_contact_first
    contacts, floor_z = detect_contacts_from_human(
        src_positions, role_to_idx, fps,
        orientation_mats=orientation_mats, ori_to_idx=ori_to_idx,
        foot_height=args.foot_contact_height,
        hand_height=args.hand_contact_height,
        speed_thresh=args.contact_speed,
        foot_flat_tilt=args.foot_flat_tilt,
    )
    effector_body_id = {
        eff: body_id(model, cfg["body"]) for eff, cfg in CONTACT_EFFECTORS.items()
    }
    effector_site_id = {
        eff: site_id(model, cfg["site"]) for eff, cfg in CONTACT_POS.items()
    }
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
    achieved_pos_list = []
    target_ori_list = []
    achieved_ori_list = []
    ori_err_deg_list = []
    n_self_coll_list = []
    contact_flags_list = []
    contact_align_err_deg_list = []

    q = np.zeros(model.nq)
    q[3] = 1.0  # root quaternion w

    coll_kwargs = dict(
        coll_weight=args.coll_weight,
        coll_margin=args.coll_margin,
        coll_gain=args.coll_gain,
        coll_hops=args.coll_hops,
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

    for ti, src_i in enumerate(frame_ids):
        targets = make_targets_for_frame(
            src_positions[src_i],
            role_to_idx,
            first_src_pos,
            target_rest_positions,
            root_scale,
            role_scales,
        )

        ori_targets = make_orientation_targets_for_frame(
            orientation_mats[src_i],
            ori_to_idx,
            first_src_ori,
            target_rest_orientations,
        )

        # Contact-first: build the foot-flat / fist-down axis-alignment
        # constraints active this frame, and suppress the matching human
        # world-delta orientation terms.
        align_constraints = []
        skip_ori_body_ids = set()
        pos_site_constraints = []
        skip_pos_roles = set()
        ori_weight_scale = {}
        pos_weight_scale = {}
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
                    continue
                bid = effector_body_id[eff]
                align_constraints.append(
                    (bid, cfg["axis_local"], cfg["world_dir"], CONTACT_ALIGN_WEIGHT[eff] * w_env)
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
                             args.foot_yaw_weight * w_env)
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
                        pos_site_constraints.append(
                            (effector_site_id[eff], tgt, CONTACT_POS_WEIGHT * w_env)
                        )
                        # Cross-fade the wrist-body position target out as the palm
                        # pin fades in (weight *= 1-w_env).
                        spr = cpos["skip_pos_role"]
                        pos_weight_scale[spr] = min(pos_weight_scale.get(spr, 1.0), 1.0 - w_env)
                        palm_targets[eff] = tgt

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
            pos_site_constraints=pos_site_constraints,
            skip_pos_roles=skip_pos_roles,
            iters=args.ik_iters,
            **coll_kwargs,
        )

        mujoco.mj_forward(model, data)

        # Count remaining self-collisions post-solve (using same filter as the constraint)
        n_self_coll = 0
        for c_idx in range(data.ncon):
            ct = data.contact[c_idx]
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

        role_order = list(ROLE_TO_ALEX_BODY.keys())
        target_pos_list.append(np.asarray([targets[r] for r in role_order], dtype=np.float64))
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
        active = [e for e in eff_order if frame_contacts.get(e, False)]
        con_str = ""
        if active:
            angs = " ".join(f"{e}:{align_errs[eff_order.index(e)]:.0f}deg"
                            + (f"/{palm_pos_err[e]*100:.0f}cm" if e in palm_pos_err else "")
                            for e in active)
            con_str = f"  contact[{angs}]"
        if ti % args.log_every == 0 or ti == len(frame_ids) - 1:
            print(f"frame {ti:04d} source={src_i:04d} mean_err={mean_err:.4f} max_err={max_err:.4f}{coll_str}{con_str}")

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
        },
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

    np.savez(
        args.out,
        qpos=np.asarray(qpos_list),
        fps=np.float64(fps),
        self_collision_counts=n_coll_arr,
        target_positions=np.asarray(target_pos_list),
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

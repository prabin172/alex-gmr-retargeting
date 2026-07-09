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

# Foot effector -> the position-target role that actually places it (the ankle
# landmark; there is no separate "foot" position role). Used by the planted-foot
# position hold to freeze the right target.
FOOT_POS_ROLE = {"left_foot": "left_ankle", "right_foot": "right_ankle"}

# Foot effector -> the knee position role above it (shank-tilt clamp).
FOOT_KNEE_ROLE = {"left_foot": "left_knee", "right_foot": "right_knee"}

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


def detect_contacts_from_human(positions, role_to_idx, fps, *,
                               orientation_mats=None, ori_to_idx=None,
                               foot_height=0.07, hand_height=0.08,
                               speed_thresh=0.4, foot_flat_tilt=40.0, floor_pct=1.0,
                               foot_flat_margin=6.0, foot_flat_min_base_frames=20,
                               on_height_frac=1.0, on_speed_frac=1.0,
                               onset_max_delay=0.15):
    """Per-frame ground-contact flags for each effector, from human mocap.

    An effector is "in contact" at frame t when the lowest of its markers is
    within `*_height` metres of the clip floor AND moving slower than
    `speed_thresh` (m/s). The floor is the low percentile of the feet markers'
    height across the whole clip.

    Onset hysteresis (`on_height_frac`/`on_speed_frac` < 1): the START of each
    contact interval is delayed until the effector passes STRICTER thresholds
    (height*frac, speed*frac) — the loose thresholds fire while it is still
    descending into a pose. The delay is capped at `onset_max_delay` seconds so
    a crouched plant that hovers under the loose gate without ever passing the
    strict one is trimmed, never dropped (uncapped hysteresis deleted whole
    genuine plant intervals on get-up clips). Release unchanged. 1.0/1.0 = off.

    For effectors with a `flat_ori_role`, contact additionally requires the
    *human* segment to be near-flat. This distinguishes a flat plantar support
    from a foot that is merely near the floor while folded (toes/side down during
    a get-up), where forcing the robot foot flat would just fight tracking.

    IMPORTANT — the flatness gate is RELATIVE to a per-foot self-calibrated
    baseline, not absolute. The canonical foot frame's local-Z is NOT the true
    sole normal: `x = toe−ankle` is the foot's bone axis, declined ~18° below
    horizontal (ankle sits above the grounded toe), so a perfectly FLAT foot reads
    ~18° tilt (+ a ~4° L/R skew) — pure frame geometry, not motion (see
    wiki/concepts/orientation-frames.md FOOTGUN). Gating raw `tilt < 40°` is
    therefore nearly inert. Instead we estimate each foot's own flat baseline as
    the p15 of its tilt over height+speed candidate-contact frames (robust to the
    tilted phantoms in the tail), and require `tilt − baseline < foot_flat_margin`.
    Falls back to the absolute `tilt < foot_flat_tilt` cap if a foot has fewer
    than `foot_flat_min_base_frames` candidate frames (baseline unreliable).

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
            up = orientation_mats[:, ori_to_idx[flat_role], :, 2]  # frame local-Z (NOT true sole normal)
            tilt = np.degrees(np.arccos(np.clip(np.abs(up @ np.array([0.0, 0.0, 1.0])), -1, 1)))
            # Per-foot self-calibrated flat baseline from height+speed candidate frames
            # (p15 = the foot's flattest ≈ its anatomical ~18° declination; robust to
            # tilted phantoms which live in the upper tail). Gate on tilt-above-baseline.
            cand = tilt[flag]
            if cand.size >= foot_flat_min_base_frames:
                base = float(np.percentile(cand, 15))
                flag = flag & ((tilt - base) < foot_flat_margin)
            else:
                flag = flag & (tilt < foot_flat_tilt)   # too few candidates → absolute cap

        if on_height_frac < 1.0 or on_speed_frac < 1.0:
            strict = flag & (h < hthr * on_height_frac) & (spd < speed_thresh * on_speed_frac)
            cap = max(0, int(round(onset_max_delay * fps)))
            out = np.zeros(N, dtype=bool)
            t = 0
            while t < N:
                if not flag[t]:
                    t += 1
                    continue
                a = t
                while t < N and flag[t]:
                    t += 1
                b = t                                   # loose interval [a, b)
                s = np.where(strict[a:b])[0]
                onset = a + (min(int(s[0]), cap) if len(s) else cap)
                out[min(onset, b - 1):b] = True         # trim the start, never drop
            flag = out

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
):
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)

    nv = model.nv
    q_ref = q_init.copy()

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
        desired_dq = np.zeros(nv)
        desired_dq[6:] = q_ref[7:] - data.qpos[7:]
        rows2.append(np.sqrt(posture_reg) * np.eye(nv))
        rhs2.append(np.sqrt(posture_reg) * desired_dq)

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
            rows2.extend(floor_rows)
            rhs2.extend([np.array([v]) for v in floor_rhs])

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

    roles, role_to_idx, src_positions, fps, orientation_roles, ori_to_idx, orientation_mats = load_canonical(args.canonical)

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

    # Median ankle-target z at CONTACT ONSET (first few frames of each planted
    # run), pooled over both feet, in TARGET space (position-only — no IK,
    # cheap). Reuse `contacts_solved` (already computed above, line ~1296) —
    # must be bit-identical to what the per-frame loop below treats as
    # "planted".
    #
    # Onset-only, not whole-run: the raw morphology-scaled human target keeps
    # DRIFTING DOWN through a plant (measured: left ankle target ranges -0.038
    # at touchdown to -0.067 by run-end, as the human's own foot keeps
    # flattening/settling in the source mocap) — but the actual per-frame loop
    # FREEZES the tracked target near onset (the foot-hold anchor, `w_env >=
    # foot_hold_latch`, ~half the contact-ramp into the run) and holds it, so
    # the robot never actually tracks that later drift. Pooling the whole run
    # measured 2.8cm too low (-0.0663) vs the real solved/frozen anchor
    # (-0.0384); pooling just the onset window matches it almost exactly
    # (onset-frame check: -0.039).
    ONSET_WINDOW = 10   # frames; run-start proxy for the foot-hold-latch frame
    planted_ankle_z = []
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
            for kk in range(k, min(k + ONSET_WINDOW, j)):
                t = make_targets_for_frame(src_positions[frame_ids[kk]], role_to_idx,
                                           first_src_pos, target_rest_positions,
                                           root_scale, role_scales)
                planted_ankle_z.append(float(t[FOOT_POS_ROLE[eff]][2]))
            k = j

    if planted_ankle_z:
        alex_floor_z = float(np.median(planted_ankle_z)) - ankle_clearance
    else:
        # No foot-contact frames in this clip (none in the corpus currently) —
        # fall back to the human-frame transform.
        human_pelvis0_z = float(first_src_pos[role_to_idx["pelvis"]][2])
        alex_floor_z = float(target_rest_positions["pelvis"][2]) \
            + root_scale * (floor_z - human_pelvis0_z)
        print("  [warn] no planted-foot frames — floor_z falling back to pelvis-delta transform")

    print(f"Floor estimate (target-space): ankle_clearance={ankle_clearance:.4f} "
          f"alex_floor_z={alex_floor_z:.4f} (human floor_z={floor_z:.4f}) "
          f"n_planted={len(planted_ankle_z)} median_ankle_target_z="
          f"{float(np.median(planted_ankle_z)) if planted_ankle_z else float('nan'):.4f}")

    floor_kwargs = {}
    if args.floor_weight > 0.0:
        data.mocap_pos[floor_mocap_id] = [0.0, 0.0, alex_floor_z]
        floor_kwargs = dict(
            floor_weight=args.floor_weight,
            floor_margin=args.floor_margin, floor_gain=args.floor_gain,
        )
        print(f"Floor repulsion (Stage-3 QP term): ON weight={args.floor_weight} "
              f"margin={args.floor_margin}")

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

    # Frames where the knee target needed the shank-tilt feasibility projection.
    shank_clamp_count = {eff: 0 for eff in FOOT_KNEE_ROLE}

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
                        # Floor clamp (Fix B, collisionFixPlan.md): the scaled
                        # human hand-contact target can map below Alex's floor
                        # (measured: 6.5-7.2cm on luigi_standProne_03's push
                        # phase) — the palm pin then faithfully tracks the fist
                        # underground. Palm site IS the support surface, so
                        # clearance 0 is correct (no margin). Independent of
                        # --floor-weight (see alex_floor_z computation above).
                        if alex_floor_z is not None:
                            tgt[2] = max(tgt[2], alex_floor_z)
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
            hierarchical=args.hierarchical,
            knee_bias=knee_bias,
            pos_site_constraints=pos_site_constraints,
            skip_pos_roles=skip_pos_roles,
            iters=args.ik_iters,
            **coll_kwargs,
            **floor_kwargs,
        )

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

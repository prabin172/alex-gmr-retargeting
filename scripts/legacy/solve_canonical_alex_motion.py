#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MVNX_SOLVER_PATH = Path(__file__).resolve().parent / "solve_mvnx_alex_motion.py"

spec = importlib.util.spec_from_file_location("mvnx_solver_module", MVNX_SOLVER_PATH)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

CANONICAL_BODY_NAMES = list(S.CANONICAL_BODY_NAMES)
PALM_ROLES = ["left_palm", "right_palm"]
SOLE_ROLES = ["left_sole", "right_sole"]
SOLE_CORNER_NAMES = [
    "toe_body_left",
    "toe_body_right",
    "heel_body_left",
    "heel_body_right",
]
SOLE_CORNER_LOCAL_OFFSETS = {
    # Offsets from alex_*_sole_contact_site at [0.05, 0, -0.07] to the four
    # bottom corners of the foot collision box. These are expressed in the
    # semantic sole frame: +X toe-forward, +Y body-left, +Z sole normal.
    "toe_body_left": np.array([0.11, 0.05, 0.0], dtype=float),
    "toe_body_right": np.array([0.11, -0.05, 0.0], dtype=float),
    "heel_body_left": np.array([-0.11, 0.05, 0.0], dtype=float),
    "heel_body_right": np.array([-0.11, -0.05, 0.0], dtype=float),
}
SOLE_CORNER_ROLES = [
    f"{side}_sole_corner_{corner}"
    for side in ("left", "right")
    for corner in SOLE_CORNER_NAMES
]
SOLE_CORNER_TO_SOLE_ROLE = {
    f"{side}_sole_corner_{corner}": f"{side}_sole"
    for side in ("left", "right")
    for corner in SOLE_CORNER_NAMES
}
CANONICAL_AUXILIARY_ROLES = PALM_ROLES + SOLE_ROLES
CANONICAL_SOURCE_ROLES = CANONICAL_BODY_NAMES + CANONICAL_AUXILIARY_ROLES

# Keep legacy left_hand/right_hand in the canonical skeleton, but make Alex's
# physical end-effector tasks use explicit marker-centre palms.
IK_ROLES = [
    "pelvis",
    "head",
    "left_knee",
    "left_sole",
    "right_knee",
    "right_sole",
    "left_shoulder",
    "left_elbow",
    "left_palm",
    "right_shoulder",
    "right_elbow",
    "right_palm",
]

LEGACY_ORIENTATION_ROLE_ALIASES = {
    "left_hand": "left_palm",
    "right_hand": "right_palm",
    "left_foot": "left_sole",
    "right_foot": "right_sole",
}

CONTACT_MASK_ROLE_ALIASES = {
    "left_sole": "left_foot",
    "right_sole": "right_foot",
}





def merge_robot_config(robot_cfg_path):
    """Load robot config; if partial site config, merge over alex.json and map roles to site names."""
    robot_cfg_path = Path(robot_cfg_path)
    cfg = json.loads(robot_cfg_path.read_text())

    base_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex.json"
    if robot_cfg_path.resolve() == base_path.resolve():
        return cfg

    base = json.loads(base_path.read_text())

    merged = dict(base)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            tmp = dict(merged[key])
            tmp.update(value)
            merged[key] = tmp
        else:
            merged[key] = value

    # alex_retarget_sites.json stores a list of site definitions.
    # The MVNX solver expects retarget_site_names = {role: site_name}.
    sites = cfg.get("sites", [])
    site_names = {s.get("name") for s in sites if isinstance(s, dict)}

    role_to_site = {}
    candidates = {
        "pelvis": "alex_pelvis_site",
        "head": "alex_head_site",
        "left_foot": "alex_left_sole_site",
        "right_foot": "alex_right_sole_site",
        "left_sole": "alex_left_sole_contact_site",
        "right_sole": "alex_right_sole_contact_site",
        "left_sole_corner_toe_body_left": "alex_left_sole_corner_toe_body_left_site",
        "left_sole_corner_toe_body_right": "alex_left_sole_corner_toe_body_right_site",
        "left_sole_corner_heel_body_left": "alex_left_sole_corner_heel_body_left_site",
        "left_sole_corner_heel_body_right": "alex_left_sole_corner_heel_body_right_site",
        "right_sole_corner_toe_body_left": "alex_right_sole_corner_toe_body_left_site",
        "right_sole_corner_toe_body_right": "alex_right_sole_corner_toe_body_right_site",
        "right_sole_corner_heel_body_left": "alex_right_sole_corner_heel_body_left_site",
        "right_sole_corner_heel_body_right": "alex_right_sole_corner_heel_body_right_site",
        "left_palm": "alex_left_palm_contact_site",
        "right_palm": "alex_right_palm_contact_site",
        # Keep these aliases available to older tooling; canonical IK below
        # uses the explicit palm roles.
        "left_hand": "alex_left_palm_contact_site",
        "right_hand": "alex_right_palm_contact_site",
    }

    for role, site_name in candidates.items():
        if site_name in site_names:
            role_to_site[role] = site_name

    if role_to_site:
        merged["retarget_site_names"] = role_to_site

    # Site configs should use the site-enabled XML unless they explicitly provide a model path.
    if not any(k in cfg for k in [
        "model_path", "mujoco_model_path", "xml_path", "model_xml_path", "robot_model_path", "robot_xml"
    ]):
        merged["model_path"] = "assets/alex/alex_floating_base_with_sites.xml"

    return merged


def resolve_model_path(robot_cfg, robot_cfg_path):
    for key in [
        "model_path",
        "mujoco_model_path",
        "xml_path",
        "model_xml_path",
        "robot_model_path",
        "robot_xml",
    ]:
        if key in robot_cfg:
            return Path(robot_cfg[key])

    # Site configs created for Alex may only store site/body mappings.
    # Use the site-enabled MuJoCo XML for site configs; otherwise use the floating-base URDF.
    name = robot_cfg_path.name.lower()
    if "site" in name:
        return Path("assets/alex/alex_floating_base_with_sites.xml")
    return Path("assets/alex/alex_floating_base.urdf")


def load_canonical_frames(npz_path, start_frame=0, stride=1, max_frames=None, recenter=True):
    d = np.load(npz_path, allow_pickle=True)
    positions = np.asarray(d["positions"], dtype=float)
    roles = [str(x) for x in d["roles"].tolist()]
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])

    role_to_idx = {r: i for i, r in enumerate(roles)}
    missing = [r for r in CANONICAL_SOURCE_ROLES if r not in role_to_idx]
    if missing:
        raise RuntimeError(f"Canonical NPZ missing roles: {missing}")

    frame_ids = np.arange(positions.shape[0])
    frame_ids = frame_ids[start_frame::stride]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    source_frames = []
    for src_i in frame_ids:
        frame = {}
        for role in CANONICAL_SOURCE_ROLES:
            p = positions[int(src_i), role_to_idx[role]]
            frame[role] = {
                "pos": [float(x) for x in p],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        source_frames.append(frame)

    if recenter and source_frames:
        source_frames = S.recenter_clip_xy(source_frames)

    meta = {
        "npz_path": str(npz_path),
        "source_fps": fps,
        "stride": stride,
        "output_fps": fps / float(stride),
        "source_frame_start_index": int(start_frame),
        "source_frame_ids": [int(x) for x in frame_ids.tolist()],
        "source_roles": roles,
    }
    return source_frames, meta


def base_skeleton_frame(frame):
    """Copy only the legacy skeleton roles for existing morphology utilities."""
    return {
        role: {
            "pos": list(frame[role]["pos"]),
            "quat_wxyz": list(frame[role]["quat_wxyz"]),
        }
        for role in CANONICAL_BODY_NAMES
    }


def add_palm_morphology_scales(source_rest, target_rest, scales):
    """Add palm/sole-specific reach scales without changing legacy scales."""
    out = dict(scales)
    for side in ("left", "right"):
        shoulder = f"{side}_shoulder"
        elbow = f"{side}_elbow"
        palm = f"{side}_palm"
        source_reach = (
            np.linalg.norm(
                np.asarray(source_rest[elbow]["pos"], dtype=float)
                - np.asarray(source_rest[shoulder]["pos"], dtype=float)
            )
            + np.linalg.norm(
                np.asarray(source_rest[palm]["pos"], dtype=float)
                - np.asarray(source_rest[elbow]["pos"], dtype=float)
            )
        )
        target_reach = (
            np.linalg.norm(
                np.asarray(target_rest[elbow]["pos"], dtype=float)
                - np.asarray(target_rest[shoulder]["pos"], dtype=float)
            )
            + np.linalg.norm(
                np.asarray(target_rest[palm]["pos"], dtype=float)
                - np.asarray(target_rest[elbow]["pos"], dtype=float)
            )
        )
        scale = 1.0 if source_reach < 1e-8 else target_reach / source_reach
        out[palm] = float(np.clip(scale, 0.70, 1.30))
    for side in ("left", "right"):
        hip = f"{side}_hip"
        knee = f"{side}_knee"
        sole = f"{side}_sole"
        source_reach = (
            np.linalg.norm(
                np.asarray(source_rest[knee]["pos"], dtype=float)
                - np.asarray(source_rest[hip]["pos"], dtype=float)
            )
            + np.linalg.norm(
                np.asarray(source_rest[sole]["pos"], dtype=float)
                - np.asarray(source_rest[knee]["pos"], dtype=float)
            )
        )
        target_reach = (
            np.linalg.norm(
                np.asarray(target_rest[knee]["pos"], dtype=float)
                - np.asarray(target_rest[hip]["pos"], dtype=float)
            )
            + np.linalg.norm(
                np.asarray(target_rest[sole]["pos"], dtype=float)
                - np.asarray(target_rest[knee]["pos"], dtype=float)
            )
        )
        scale = 1.0 if source_reach < 1e-8 else target_reach / source_reach
        out[sole] = float(np.clip(scale, 0.70, 1.30))
    return out


def add_semantic_aux_targets(
    base_target_frame,
    source_frame,
    source_rest,
    target_rest,
    target_generation,
    motion_scale,
    morphology_scales,
):
    """Append physically matched palm/sole targets to a legacy-skeleton target frame."""
    out = {
        role: {
            "pos": list(value["pos"]),
            "quat_wxyz": list(value["quat_wxyz"]),
        }
        for role, value in base_target_frame.items()
    }
    target_pelvis = np.asarray(out["pelvis"]["pos"], dtype=float)
    source_pelvis = np.asarray(source_frame["pelvis"]["pos"], dtype=float)
    source_rest_pelvis = np.asarray(source_rest["pelvis"]["pos"], dtype=float)
    target_rest_pelvis = np.asarray(target_rest["pelvis"]["pos"], dtype=float)

    for role in CANONICAL_AUXILIARY_ROLES:
        source_pos = np.asarray(source_frame[role]["pos"], dtype=float)
        source_rest_pos = np.asarray(source_rest[role]["pos"], dtype=float)
        target_rest_pos = np.asarray(target_rest[role]["pos"], dtype=float)

        if target_generation == "morphology-delta":
            source_local_delta = (
                (source_pos - source_pelvis)
                - (source_rest_pos - source_rest_pelvis)
            )
            target_rest_local = target_rest_pos - target_rest_pelvis
            scale = float(morphology_scales[role])
            target_pos = target_pelvis + target_rest_local + scale * source_local_delta
        else:
            # For rest-delta and tree-scale modes, maintain the explicit
            # endpoint's source motion delta. The active experiments use
            # morphology-delta above.
            target_pos = target_rest_pos + float(motion_scale) * (source_pos - source_rest_pos)

        out[role] = {
            "pos": [float(value) for value in target_pos],
            "quat_wxyz": list(target_rest[role]["quat_wxyz"]),
        }
    return out



def load_contact_masks_for_solver(contact_mask_npz, source_frame_ids, roles):
    if contact_mask_npz is None:
        return None

    contact_mask_npz = Path(contact_mask_npz)
    if not contact_mask_npz.is_absolute():
        contact_mask_npz = REPO_ROOT / contact_mask_npz

    d = np.load(contact_mask_npz, allow_pickle=True)
    out = {}

    source_frame_ids = np.asarray(source_frame_ids, dtype=int)

    for role in roles:
        key = f"{role}_stable_contact"
        mask_role = role
        if key not in d.files:
            fallback_role = CONTACT_MASK_ROLE_ALIASES.get(role)
            fallback_key = f"{fallback_role}_stable_contact" if fallback_role else None
            if fallback_key and fallback_key in d.files:
                key = fallback_key
                mask_role = fallback_role
            else:
                raise RuntimeError(f"Contact mask missing key: {key} in {contact_mask_npz}")

        full = np.asarray(d[key], dtype=bool)
        safe_ids = np.clip(source_frame_ids, 0, len(full) - 1)
        out[role] = full[safe_ids]
        if mask_role != role:
            print(f"  contact mask alias: {role}_stable_contact <- {mask_role}_stable_contact")

    return out


def compute_semantic_sole_planted_masks(
    canonical_npz_path,
    source_frame_ids,
    roles,
    *,
    ground_threshold_m,
    speed_threshold_mps,
    tilt_threshold_deg,
):
    """Compute support-planted sole masks from semantic sole role kinematics.

    This is intentionally stricter than the legacy stable foot-contact masks:
    a sole is considered planted only when it is near the floor, moving slowly,
    and its semantic sole normal is already reasonably close to vertical.
    Large-tilt ground contact during get-up is therefore treated as incidental
    contact, not flat support contact.
    """
    canonical_npz_path = Path(canonical_npz_path)
    if not canonical_npz_path.is_absolute():
        canonical_npz_path = REPO_ROOT / canonical_npz_path

    data = np.load(canonical_npz_path, allow_pickle=True)
    if "positions" not in data.files or "roles" not in data.files:
        raise RuntimeError(f"{canonical_npz_path} must contain positions and roles")
    if "role_quats_wxyz" not in data.files:
        raise RuntimeError(
            f"{canonical_npz_path} must contain role_quats_wxyz for sole-planted masks"
        )

    canonical_roles = [str(role) for role in data["roles"].tolist()]
    role_to_index = {role: index for index, role in enumerate(canonical_roles)}
    missing = [role for role in roles if role not in role_to_index]
    if missing:
        raise RuntimeError(f"Canonical NPZ missing semantic sole roles: {missing}")

    positions = np.asarray(data["positions"], dtype=float)
    quats = np.asarray(data["role_quats_wxyz"], dtype=float)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    source_frame_ids = np.asarray(source_frame_ids, dtype=int)
    safe_ids = np.clip(source_frame_ids, 0, positions.shape[0] - 1)

    floor_z = min(float(np.nanmin(positions[:, role_to_index[role], 2])) for role in roles)
    out = {}
    diagnostics = {}
    for role in roles:
        idx = role_to_index[role]
        pos = positions[:, idx, :]
        # First-order velocity estimate in the full-rate canonical sequence.
        vel = np.zeros_like(pos)
        if len(pos) > 1:
            vel[1:] = (pos[1:] - pos[:-1]) * fps
            vel[0] = vel[1]
        speed = np.linalg.norm(vel, axis=1)

        normal_z = np.zeros(len(pos), dtype=float)
        for frame_id in range(len(pos)):
            R = quat_wxyz_to_rotmat(quats[frame_id, idx])
            normal_z[frame_id] = float(np.clip(R[:, 2].dot(np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        tilt_deg = np.degrees(np.arccos(normal_z))

        near_ground = pos[:, 2] <= (floor_z + float(ground_threshold_m))
        slow = speed <= float(speed_threshold_mps)
        upright = tilt_deg <= float(tilt_threshold_deg)
        full_mask = near_ground & slow & upright
        out[role] = full_mask[safe_ids]
        diagnostics[role] = {
            "active_frames": int(np.sum(out[role])),
            "num_frames": int(len(out[role])),
            "floor_z_m": float(floor_z),
            "sampled_min_z_m": float(np.min(pos[safe_ids, 2])),
            "sampled_max_z_m": float(np.max(pos[safe_ids, 2])),
            "sampled_mean_speed_mps": float(np.mean(speed[safe_ids])),
            "sampled_max_speed_mps": float(np.max(speed[safe_ids])),
            "sampled_mean_tilt_deg": float(np.mean(tilt_deg[safe_ids])),
            "sampled_min_tilt_deg": float(np.min(tilt_deg[safe_ids])),
            "sampled_max_tilt_deg": float(np.max(tilt_deg[safe_ids])),
            "near_ground_frames": int(np.sum(near_ground[safe_ids])),
            "slow_frames": int(np.sum(slow[safe_ids])),
            "upright_frames": int(np.sum(upright[safe_ids])),
        }

    return out, diagnostics


def contact_aware_target_stabilize(frame, frame_idx, contact_masks, contact_state, roles, mode="xy", alpha=0.75):
    """Softly anchor contact-role target positions in target space. Does not modify source motion."""
    if contact_masks is None:
        return frame

    new_frame = {}
    for role, value in frame.items():
        new_frame[role] = dict(value)
        new_frame[role]["pos"] = list(value["pos"])

    for role in roles:
        if role not in new_frame or role not in contact_masks:
            continue

        active = bool(contact_masks[role][frame_idx])
        pos = np.asarray(new_frame[role]["pos"], dtype=float)

        if active:
            if (role not in contact_state) or (not contact_state[role].get("active", False)):
                contact_state[role] = {
                    "active": True,
                    "anchor": pos.copy(),
                    "start_frame": frame_idx,
                }

            anchor = np.asarray(contact_state[role]["anchor"], dtype=float)

            if mode == "xy":
                pos[:2] = (1.0 - alpha) * pos[:2] + alpha * anchor[:2]
            elif mode == "xyz":
                pos[:] = (1.0 - alpha) * pos[:] + alpha * anchor[:]
            else:
                raise RuntimeError(f"Unknown contact anchor mode: {mode}")

            new_frame[role]["pos"] = [float(x) for x in pos]
        else:
            if role in contact_state:
                contact_state[role]["active"] = False

    return new_frame


def apply_planted_sole_position_targets(
    frame,
    frame_idx,
    sole_planted_masks,
    roles,
    mode="none",
    ground_z=0.0,
    alpha=1.0,
    sole_orientation_quats=None,
):
    """Override semantic sole target positions during planted support frames.

    ``z_ground`` is intentionally weaker than contact anchoring and flat-yaw
    orientation control: it keeps the retargeted x/y target untouched and only
    blends the sole-site target z toward the ground plane when the semantic
    sole-planted support mask is active.

    ``corners_plane_ground`` preserves the retargeted/human sole orientation:
    it builds the four bottom-corner targets from the sole frame, then applies
    a single vertical shift so the lowest target corner touches the ground. If
    the human sole is flat, all corners land on the ground; if it is tilted,
    the high edge remains above the ground instead of being forced flat.
    """
    if sole_planted_masks is None or mode == "none":
        return frame

    if mode not in {"z_ground", "corners_plane_ground"}:
        raise RuntimeError(f"Unknown planted sole position mode: {mode}")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    new_frame = {}
    for role, value in frame.items():
        new_frame[role] = dict(value)
        if "pos" in value:
            new_frame[role]["pos"] = list(value["pos"])

    if mode == "corners_plane_ground":
        for sole_role in roles:
            if sole_role not in new_frame:
                continue
            side = sole_role.split("_", 1)[0]
            center_pos = np.asarray(new_frame[sole_role]["pos"], dtype=float)
            if sole_orientation_quats is not None and sole_role in sole_orientation_quats:
                center_quat = np.asarray(sole_orientation_quats[sole_role], dtype=float)
            else:
                center_quat = np.asarray(new_frame[sole_role]["quat_wxyz"], dtype=float)
            R_sole = quat_wxyz_to_rotmat(center_quat)

            corner_positions = {}
            for corner_name, offset in SOLE_CORNER_LOCAL_OFFSETS.items():
                corner_role = f"{side}_sole_corner_{corner_name}"
                corner_positions[corner_role] = center_pos + R_sole @ offset

            active = (
                sole_role in sole_planted_masks
                and bool(sole_planted_masks[sole_role][frame_idx])
            )
            if active and corner_positions:
                min_corner_z = min(float(pos[2]) for pos in corner_positions.values())
                z_shift = alpha * (float(ground_z) - min_corner_z)
                center_pos = center_pos.copy()
                center_pos[2] += z_shift
                new_frame[sole_role]["pos"] = [float(x) for x in center_pos]
                for corner_role in corner_positions:
                    corner_positions[corner_role] = corner_positions[corner_role].copy()
                    corner_positions[corner_role][2] += z_shift

            for corner_role, corner_pos in corner_positions.items():
                new_frame[corner_role] = {
                    "pos": [float(x) for x in corner_pos],
                    "quat_wxyz": [float(x) for x in center_quat],
                    "quat": [float(x) for x in center_quat],
                }

        return new_frame

    for role in roles:
        if role not in new_frame or role not in sole_planted_masks:
            continue
        if not bool(sole_planted_masks[role][frame_idx]):
            continue

        pos = np.asarray(new_frame[role]["pos"], dtype=float)
        pos[2] = (1.0 - alpha) * pos[2] + alpha * float(ground_z)
        new_frame[role]["pos"] = [float(x) for x in pos]

    return new_frame


def flat_yaw_rotation_from_quat(quat_wxyz):
    """Preserve heading from a target frame while forcing +Z/normal vertical."""
    R = quat_wxyz_to_rotmat(quat_wxyz)
    x_axis = np.asarray(R[:, 0], dtype=float)
    x_flat = np.array([x_axis[0], x_axis[1], 0.0], dtype=float)
    norm = float(np.linalg.norm(x_flat))
    if norm < 1e-8:
        x_flat = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        x_flat = x_flat / norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = np.cross(z_axis, x_flat)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    return np.column_stack([x_axis, y_axis, z_axis])


def quat_slerp_wxyz(q0, q1, alpha):
    """Spherical linear interpolation for scalar-first quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / max(float(np.linalg.norm(q)), 1e-12)

    theta0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta0 = float(np.sin(theta0))
    theta = theta0 * alpha
    sin_theta = float(np.sin(theta))

    s0 = np.cos(theta) - dot * sin_theta / max(sin_theta0, 1e-12)
    s1 = sin_theta / max(sin_theta0, 1e-12)
    q = s0 * q0 + s1 * q1
    return q / max(float(np.linalg.norm(q)), 1e-12)


def apply_planted_sole_orientation_targets(
    frame,
    frame_idx,
    sole_planted_masks,
    roles,
    mode="none",
    ramp_state=None,
    ramp_frames=1,
):
    """Override sole orientation targets during semantic sole-planted frames.

    This is intentionally stricter than the legacy foot-contact mask. It only
    applies when the semantic sole-planted mask says the sole is near the
    floor, slow, and already close enough to flat support. When a planted phase
    begins later in the clip, the target is ramped from the unmodified human
    target toward the flat-yaw target to avoid an instant 70-90 degree jump
    under strict output-frame joint caps.
    """
    if sole_planted_masks is None or mode == "none":
        return frame

    if mode != "flat_yaw":
        raise RuntimeError(f"Unknown planted sole orientation mode: {mode}")

    new_frame = {}
    for role, value in frame.items():
        new_frame[role] = dict(value)
        if "pos" in value:
            new_frame[role]["pos"] = list(value["pos"])

    for role in roles:
        if role not in new_frame or role not in sole_planted_masks:
            continue
        active = bool(sole_planted_masks[role][frame_idx])
        if not active:
            if ramp_state is not None and role in ramp_state:
                ramp_state[role]["active"] = False
            continue

        if ramp_state is not None:
            if (role not in ramp_state) or (not ramp_state[role].get("active", False)):
                ramp_state[role] = {"active": True, "start_frame": int(frame_idx)}
            start_frame = int(ramp_state[role]["start_frame"])
        else:
            start_frame = int(frame_idx)

        ramp_frames = max(1, int(ramp_frames))
        alpha = min(1.0, float(frame_idx - start_frame + 1) / float(ramp_frames))
        q_original = np.asarray(new_frame[role]["quat_wxyz"], dtype=float)
        R_target = flat_yaw_rotation_from_quat(new_frame[role]["quat_wxyz"])
        q_flat = rotmat_to_quat_wxyz_single(R_target)
        q_target = quat_slerp_wxyz(q_original, q_flat, alpha)
        new_frame[role]["quat_wxyz"] = [float(x) for x in q_target]
        new_frame[role]["quat"] = [float(x) for x in q_target]

    return new_frame


def set_task_position_cost(task, cost):
    if hasattr(task, "set_position_cost"):
        task.set_position_cost(cost)
        return

    task_cost = np.asarray(getattr(task, "cost", np.zeros(6)), dtype=float)
    if task_cost.shape[0] < 6:
        raise AttributeError("Task does not expose set_position_cost() or a 6D cost vector")
    task_cost[:3] = np.asarray(cost, dtype=float)
    task.cost = task_cost


def apply_sole_corner_position_costs_to_tasks(
    tasks,
    sole_planted_masks,
    frame_idx,
    *,
    mode,
    corner_z_cost,
):
    """Activate z-only corner constraints only during semantic planted support."""
    if mode != "corners_plane_ground":
        return

    for corner_role in SOLE_CORNER_ROLES:
        if corner_role not in tasks:
            continue
        sole_role = SOLE_CORNER_TO_SOLE_ROLE[corner_role]
        active = (
            sole_planted_masks is not None
            and sole_role in sole_planted_masks
            and bool(sole_planted_masks[sole_role][frame_idx])
        )
        cost = [0.0, 0.0, float(corner_z_cost) if active else 0.0]
        set_task_position_cost(tasks[corner_role], cost)



def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
    ], dtype=float)


def rotmat_to_quat_wxyz_single(R):
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(max(tr + 1.0, 1e-12)) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=float)
    return q / max(np.linalg.norm(q), 1e-12)


def orientation_error_rad(target_quat_wxyz, solved_rotation):
    target_rotation = quat_wxyz_to_rotmat(target_quat_wxyz)
    relative = target_rotation.T @ np.asarray(solved_rotation, dtype=float)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(cosine))


def get_frame_world_rotation_from_task_target(configuration, task):
    """Return current world rotation of the task frame/site/body at the rest configuration."""
    # Mink FrameTask stores the task frame under frame_name/frame_type.  Do not
    # silently fall back to identity here: doing so makes relative orientation
    # transfer independent of Alex's actual task-frame convention.
    frame_name = getattr(task, "frame_name", None)
    frame_type = getattr(task, "frame_type", None)
    if frame_name is None or frame_type is None:
        raise AttributeError(
            "Mink FrameTask must expose frame_name and frame_type to capture "
            "the Alex rest orientation."
        )

    transform = configuration.get_transform_frame_to_world(frame_name, frame_type)
    return np.asarray(transform.rotation().as_matrix(), dtype=float)


def load_human_role_quats_for_solver(canonical_npz_path, source_frame_ids, roles):
    d = np.load(canonical_npz_path, allow_pickle=True)
    if "role_quats_wxyz" not in d.files:
        raise RuntimeError(
            f"{canonical_npz_path} does not contain role_quats_wxyz. "
            "Run scripts/build_canonical_segment_orientations.py first."
        )

    canonical_roles = [str(x) for x in d["roles"].tolist()]
    role_to_idx = {r: i for i, r in enumerate(canonical_roles)}
    full_quats = np.asarray(d["role_quats_wxyz"], dtype=float)

    source_frame_ids = np.asarray(source_frame_ids, dtype=int)
    safe_ids = np.clip(source_frame_ids, 0, full_quats.shape[0] - 1)

    out = {}
    for role in roles:
        if role not in role_to_idx:
            raise RuntimeError(f"Role {role} missing from canonical role_quats_wxyz roles: {canonical_roles}")
        out[role] = full_quats[safe_ids, role_to_idx[role], :]

    return out


def apply_human_relative_orientations_to_frame(
    frame,
    frame_idx,
    human_quats,
    human_rest_R,
    robot_rest_R,
    roles,
    transfer_mode="world_delta",
):
    """
    True segment orientation transfer:
        R_delta_human = R_human_t @ R_human_rest.T
        R_robot_target = R_delta_human @ R_robot_rest
    """
    new_frame = {}
    for role, value in frame.items():
        new_frame[role] = dict(value)
        if "pos" in value:
            new_frame[role]["pos"] = list(value["pos"])

    for role in roles:
        if role not in new_frame or role not in human_quats:
            continue

        R_h0 = human_rest_R[role]
        R_ht = quat_wxyz_to_rotmat(human_quats[role][frame_idx])
        R_r0 = robot_rest_R[role]

        if transfer_mode == "world_delta":
            # Apply the human segment's world-space change to the robot rest frame.
            R_delta = R_ht @ R_h0.T
            R_target = R_delta @ R_r0
        elif transfer_mode == "local_delta":
            # Apply the human segment's local rest-to-current change in the robot's rest frame.
            R_delta = R_h0.T @ R_ht
            R_target = R_r0 @ R_delta
        elif transfer_mode == "source_absolute":
            # Directly copy the semantic source frame in world coordinates.
            # This is mainly for roles whose source/robot axes are explicitly constructed
            # to have the same semantic meaning, e.g. sole +X toe-forward, +Y left, +Z normal.
            R_target = R_ht
        else:
            raise ValueError(f"Unknown human orientation transfer mode: {transfer_mode}")

        q_target = rotmat_to_quat_wxyz_single(R_target)

        new_frame[role]["quat_wxyz"] = [float(x) for x in q_target]
        new_frame[role]["quat"] = [float(x) for x in q_target]

    return new_frame


def apply_human_orientation_costs_to_tasks(tasks, roles, args):
    role_to_cost = {
        "pelvis": args.human_pelvis_ori_cost,
        "head": args.human_head_ori_cost,
        "left_foot": args.human_foot_ori_cost,
        "right_foot": args.human_foot_ori_cost,
        "left_sole": (
            args.planted_sole_ori_cost
            if args.planted_sole_ori_cost > 0.0
            else args.human_foot_ori_cost
        ),
        "right_sole": (
            args.planted_sole_ori_cost
            if args.planted_sole_ori_cost > 0.0
            else args.human_foot_ori_cost
        ),
        "left_knee": args.human_knee_ori_cost,
        "right_knee": args.human_knee_ori_cost,
        "left_palm": args.human_hand_ori_cost,
        "right_palm": args.human_hand_ori_cost,
        "left_hand": args.human_hand_ori_cost,
        "right_hand": args.human_hand_ori_cost,
        "left_elbow": args.human_elbow_ori_cost,
        "right_elbow": args.human_elbow_ori_cost,
        "left_shoulder": args.human_shoulder_ori_cost,
        "right_shoulder": args.human_shoulder_ori_cost,
    }

    applied = {}
    for role in roles:
        if role not in tasks:
            print(f"WARNING: orientation role {role} has no IK task; skipping.")
            continue

        cost = float(role_to_cost.get(role, 0.0))
        task = tasks[role]

        if not hasattr(task, "set_orientation_cost"):
            raise AttributeError(
                f"Task for role {role} does not expose set_orientation_cost(); "
                "do not assign orientation_cost directly."
            )

        # FrameTask stores the actual orientation weights in task.cost[3:].
        # Assigning task.orientation_cost only changes an attribute and does
        # not update the QP objective in Mink.
        task.set_orientation_cost(cost)
        effective_cost = np.asarray(task.cost[3:], dtype=float)
        if not np.allclose(effective_cost, cost):
            raise RuntimeError(
                f"Failed to apply orientation cost for {role}: "
                f"expected {cost}, got {effective_cost.tolist()}"
            )

        applied[role] = cost

    print()
    print("Applied HUMAN segment orientation costs:")
    for role in roles:
        if role in applied:
            print(f"  {role:14s}: {applied[role]:.4f} | task.cost={tasks[role].cost}")

    return applied


def joint_qpos_width(model, joint_id):
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def output_step_cap_for_joint_name(name, args):
    """Return the configured output-frame cap for an Alex joint, or None."""
    upper = name.upper()
    if "SPINE" in upper or "NECK" in upper:
        return args.max_spine_neck_step_rad
    if "HIP" in upper or "KNEE" in upper or "ANKLE" in upper:
        return args.max_leg_step_rad
    if any(token in upper for token in ("SHOULDER", "ELBOW", "WRIST", "GRIPPER", "FINGER", "THUMB", "INDEX", "MIDDLE", "RING", "PINKY")):
        return args.max_arm_step_rad
    return args.max_other_step_rad


def build_output_joint_step_caps(model, joint_names, args):
    """Build an nq-sized cap vector; inf means no output-frame cap."""
    caps = np.full(model.nq, np.inf, dtype=float)
    for name in joint_names:
        cap = output_step_cap_for_joint_name(name, args)
        if cap is None or cap <= 0.0:
            continue
        joint_id = model.joint(name).id
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos_width = joint_qpos_width(model, joint_id)
        caps[qpos_adr:qpos_adr + qpos_width] = float(cap)
    return caps if np.any(np.isfinite(caps)) else None


def build_output_joint_velocity_limit(model, joint_names, output_joint_step_caps, max_iter):
    """
    Convert output-frame step caps into conservative per-QP velocity limits.

    solve_frame integrates ``max_iter`` steps of size model.opt.timestep. A
    cap of d radians per 30 Hz output frame therefore maps to d / (N * dt)
    rad/s inside each QP. This makes intermediate candidates available for the
    temporal selector instead of letting the first QP step leap past the cap.
    """
    if output_joint_step_caps is None:
        return None, {}
    if max_iter <= 0:
        raise ValueError("max_iter must be positive when output-frame caps are enabled")

    velocities = {}
    for name in joint_names:
        joint_id = model.joint(name).id
        if joint_qpos_width(model, joint_id) != 1:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        cap = float(output_joint_step_caps[qpos_adr])
        if np.isfinite(cap):
            velocities[name] = cap / (float(max_iter) * float(model.opt.timestep))

    if not velocities:
        return None, {}
    return S.mink.VelocityLimit(model, velocities=velocities), velocities


def joint_limit_margins(model, qpos, joint_names):
    """Return lower, upper, and nearest-limit margins for the requested joints."""
    lower = np.full(len(joint_names), np.inf, dtype=float)
    upper = np.full(len(joint_names), np.inf, dtype=float)

    for index, name in enumerate(joint_names):
        joint_id = model.joint(name).id
        if not bool(model.jnt_limited[joint_id]):
            continue
        if joint_qpos_width(model, joint_id) != 1:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        value = float(qpos[qpos_adr])
        qmin, qmax = np.asarray(model.jnt_range[joint_id], dtype=float)
        lower[index] = value - qmin
        upper[index] = qmax - value

    return lower, upper, np.minimum(lower, upper)


def main():
    parser = argparse.ArgumentParser(description="Retarget canonical human role positions to Alex IK.")
    parser.add_argument("canonical_npz", type=Path)
    parser.add_argument("--robot-config", type=Path, default=Path("general_motion_retargeting/robot_configs/alex.json"))
    parser.add_argument("--output-stem", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/debug"))

    parser.add_argument("--solver", choices=["auto", "proxqp", "daqp"], default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-ik-iter", type=int, default=50)

    parser.add_argument(
        "--candidate-selection",
        choices=["position_only", "position_tiebreak", "combined", "balanced"],
        default="position_only",
        help="How to select among inner IK iterates. position_only preserves the baseline.",
    )
    parser.add_argument(
        "--candidate-position-tolerance",
        type=float,
        default=0.0,
        help="Relative position-score tolerance for position_tiebreak, e.g. 0.05 for 5%%.",
    )
    parser.add_argument("--candidate-orientation-weight", type=float, default=1.0)
    parser.add_argument("--candidate-temporal-weight", type=float, default=1.0)
    parser.add_argument("--candidate-root-weight", type=float, default=0.0)
    parser.add_argument("--candidate-root-step-scale-m", type=float, default=0.03)
    parser.add_argument("--candidate-max-root-step-m", type=float, default=None)
    parser.add_argument("--candidate-limit-weight", type=float, default=0.0)
    parser.add_argument("--candidate-limit-margin-rad", type=float, default=0.05)
    parser.add_argument("--max-leg-step-rad", type=float, default=None)
    parser.add_argument("--max-spine-neck-step-rad", type=float, default=None)
    parser.add_argument("--max-arm-step-rad", type=float, default=None)
    parser.add_argument("--max-other-step-rad", type=float, default=None)
    parser.add_argument(
        "--joint-limit-warning-margin-rad",
        type=float,
        default=0.05,
        help="Report output frames with a joint this close to a hard position limit.",
    )

    parser.add_argument("--posture-cost", type=float, default=0.0)
    parser.add_argument("--posture-neutral-blend", type=float, default=0.02)
    parser.add_argument("--posture-mode", choices=["scalar", "selective"], default="scalar")
    parser.add_argument("--posture-wrist-cost", type=float, default=1.0)
    parser.add_argument("--posture-gripper-cost", type=float, default=0.7)
    parser.add_argument("--posture-ankle-cost", type=float, default=0.10)
    parser.add_argument("--posture-spine-neck-cost", type=float, default=0.05)
    parser.add_argument("--posture-arm-cost", type=float, default=0.10)
    parser.add_argument("--posture-leg-cost", type=float, default=0.0)
    parser.add_argument("--posture-base-cost", type=float, default=0.0)

    parser.add_argument("--foot-cost-mult", type=float, default=1.0)
    parser.add_argument("--head-cost-mult", type=float, default=1.0)
    parser.add_argument("--hand-cost-mult", type=float, default=1.0)
    parser.add_argument("--pelvis-cost-mult", type=float, default=1.0)

    # True human-segment relative orientation tracking.
    # Requires canonical NPZ field: role_quats_wxyz [T, R, 4].
    parser.add_argument("--human-orientation-roles", type=str, default="")
    parser.add_argument(
        "--human-orientation-transfer-mode",
        type=str,
        default="world_delta",
        choices=["world_delta", "local_delta", "source_absolute"],
        help="world_delta: R_h(t)R_h(0)^T R_r(0); local_delta: R_r(0)R_h(0)^T R_h(t); source_absolute: R_h(t) directly",
    )
    parser.add_argument("--human-pelvis-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-head-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-foot-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-knee-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-hand-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-elbow-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-shoulder-ori-cost", type=float, default=0.0)

    parser.add_argument("--target-rest-mode", choices=["aligned-source-rest", "raw-alex-default"], default="aligned-source-rest")
    parser.add_argument("--target-generation", choices=["rest-delta", "tree-scale", "morphology-delta"], default="morphology-delta")
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--no-recenter", action="store_true")

    parser.add_argument("--contact-mask-npz", type=Path, default=None)
    parser.add_argument("--contact-roles", type=str, default="")
    parser.add_argument("--contact-anchor-mode", choices=["xy", "xyz"], default="xy")
    parser.add_argument("--contact-anchor-alpha", type=float, default=0.75)
    parser.add_argument(
        "--planted-sole-position-mode",
        choices=["none", "z_ground", "corners_plane_ground"],
        default="none",
        help=(
            "When semantic sole-planted support masks are active, optionally "
            "ground the center sole target z, or ground the human-oriented "
            "multi-corner sole plane without forcing it flat."
        ),
    )
    parser.add_argument(
        "--planted-sole-ground-z",
        type=float,
        default=0.0,
        help="Ground-plane z used by --planted-sole-position-mode z_ground.",
    )
    parser.add_argument(
        "--planted-sole-position-alpha",
        type=float,
        default=1.0,
        help="Blend amount for planted sole position targets: 1.0 applies the full vertical grounding shift.",
    )
    parser.add_argument(
        "--planted-sole-corner-z-cost",
        type=float,
        default=20.0,
        help=(
            "Per-corner z-only position cost for corners_plane_ground. "
            "Four corners at 20 roughly match one center z task at 80."
        ),
    )
    parser.add_argument(
        "--planted-sole-orientation-mode",
        choices=["none", "flat_yaw"],
        default="none",
        help="When sole contact masks are active, optionally force sole +Z vertical while preserving yaw.",
    )
    parser.add_argument(
        "--planted-sole-ori-cost",
        type=float,
        default=0.0,
        help="Orientation cost for left_sole/right_sole when planted sole flatness is enabled.",
    )
    parser.add_argument(
        "--sole-planted-mask-mode",
        choices=["heuristic", "none"],
        default="heuristic",
        help=(
            "Mask source for planted-flat sole orientation. heuristic requires near-ground, "
            "low-speed, and already-near-flat semantic sole frames; none disables flat targets."
        ),
    )
    parser.add_argument(
        "--sole-planted-ground-threshold-m",
        type=float,
        default=0.08,
        help="Semantic sole is near ground when z <= global sole floor + this threshold.",
    )
    parser.add_argument(
        "--sole-planted-speed-threshold-mps",
        type=float,
        default=0.20,
        help="Semantic sole speed threshold for planted support detection.",
    )
    parser.add_argument(
        "--sole-planted-tilt-threshold-deg",
        type=float,
        default=45.0,
        help="Maximum semantic sole-normal tilt from world-up for planted support detection.",
    )
    parser.add_argument(
        "--planted-sole-ramp-frames",
        type=int,
        default=8,
        help="Output frames over which flat-yaw sole targets ramp in at the start of a planted phase.",
    )

    args = parser.parse_args()

    planted_sole_position_enabled = args.planted_sole_position_mode != "none"
    planted_sole_orientation_enabled = args.planted_sole_orientation_mode != "none"
    planted_sole_corner_enabled = args.planted_sole_position_mode == "corners_plane_ground"
    ik_roles = list(IK_ROLES)
    if planted_sole_corner_enabled:
        ik_roles.extend(SOLE_CORNER_ROLES)
        for role in SOLE_CORNER_ROLES:
            # Corner tasks are activated per frame with z-only costs. Keep the
            # construction-time cost zero so non-planted frames have no corner
            # influence unless explicitly enabled below.
            S.COSTS[role] = ([0.0, 0.0, 0.0], 0.0)

    if (
        planted_sole_orientation_enabled
        and args.planted_sole_ori_cost > 0.0
        and args.candidate_selection == "position_only"
    ):
        print(
            "WARNING: planted-flat sole orientation is enabled, but "
            "--candidate-selection=position_only can select iteration 0 when "
            "position is already good and orientation is wrong. Upgrading to "
            "--candidate-selection=combined for this run."
        )
        args.candidate_selection = "combined"

    S.apply_cost_multipliers(args)

    canonical_npz = args.canonical_npz.resolve()
    robot_cfg_path = args.robot_config
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = REPO_ROOT / robot_cfg_path
    robot_cfg = merge_robot_config(robot_cfg_path)

    model_path = resolve_model_path(robot_cfg, robot_cfg_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Canonical NPZ:", canonical_npz)
    print("Robot config:", robot_cfg_path)
    print("Model:", model_path)
    print("Output stem:", args.output_stem)

    print()
    print("Loading canonical source frames...")
    source_frames, source_meta = load_canonical_frames(
        canonical_npz,
        start_frame=args.start_frame,
        stride=args.stride,
        max_frames=args.max_frames,
        recenter=not args.no_recenter,
    )

    if not source_frames:
        raise RuntimeError("No canonical source frames loaded.")

    output_fps = float(source_meta["output_fps"])

    print("Loaded source frames:", len(source_frames))
    print("Source FPS:", source_meta["source_fps"])
    print("Stride:", args.stride)
    print("Output FPS:", output_fps)

    contact_roles = [r.strip() for r in args.contact_roles.split(",") if r.strip()]
    contact_masks = None
    contact_state = {}
    if args.contact_mask_npz is not None and contact_roles:
        contact_masks = load_contact_masks_for_solver(
            contact_mask_npz=args.contact_mask_npz,
            source_frame_ids=source_meta["source_frame_ids"],
            roles=contact_roles,
        )
        print()
        print("Contact-aware target stabilization:")
        print("  mask npz:", args.contact_mask_npz)
        print("  roles:", contact_roles)
        print("  mode:", args.contact_anchor_mode)
        print("  alpha:", args.contact_anchor_alpha)
        for role in contact_roles:
            print(f"  {role:12s} active frames: {int(contact_masks[role].sum())}/{len(contact_masks[role])}")

    sole_planted_masks = None
    sole_planted_diagnostics = {}
    if planted_sole_position_enabled or planted_sole_orientation_enabled:
        if args.sole_planted_mask_mode == "heuristic":
            sole_planted_masks, sole_planted_diagnostics = compute_semantic_sole_planted_masks(
                canonical_npz_path=canonical_npz,
                source_frame_ids=source_meta["source_frame_ids"],
                roles=SOLE_ROLES,
                ground_threshold_m=args.sole_planted_ground_threshold_m,
                speed_threshold_mps=args.sole_planted_speed_threshold_mps,
                tilt_threshold_deg=args.sole_planted_tilt_threshold_deg,
            )
            print()
            print("Semantic sole-planted support masks:")
            print("  mode: heuristic")
            print("  ground threshold m:", args.sole_planted_ground_threshold_m)
            print("  speed threshold m/s:", args.sole_planted_speed_threshold_mps)
            print("  tilt threshold deg:", args.sole_planted_tilt_threshold_deg)
            for role in SOLE_ROLES:
                count = int(sole_planted_masks[role].sum())
                print(f"  {role:12s} semantic planted frames: {count}/{len(sole_planted_masks[role])}")
        else:
            print()
            print("Semantic sole-planted support masks disabled; planted sole position/orientation targets will not apply.")

    source_rest = source_frames[0]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    solver = S.choose_solver(args.solver)
    print("Using solver:", solver)

    qpos0 = np.asarray(model.qpos0, dtype=float)
    if qpos0.shape[0] != 36:
        raise RuntimeError(f"Expected Alex nq=36, got {qpos0.shape[0]}")

    first_pelvis = np.asarray(source_frames[0]["pelvis"]["pos"], dtype=float)
    qpos0[0:3] = first_pelvis
    qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

    limits = [S.mink.ConfigurationLimit(model)]
    tasks = S.make_tasks(model, robot_cfg, roles=ik_roles)
    output_joint_step_caps = build_output_joint_step_caps(
        model,
        robot_cfg["actuated_joint_order"],
        args,
    )
    output_joint_velocity_limit, output_joint_velocity_limits = build_output_joint_velocity_limit(
        model,
        robot_cfg["actuated_joint_order"],
        output_joint_step_caps,
        args.max_ik_iter,
    )
    motion_limits = list(limits)
    if output_joint_velocity_limit is not None:
        motion_limits.append(output_joint_velocity_limit)
    if output_joint_step_caps is not None:
        capped = []
        for name in robot_cfg["actuated_joint_order"]:
            joint_id = model.joint(name).id
            qpos_adr = int(model.jnt_qposadr[joint_id])
            cap = output_joint_step_caps[qpos_adr]
            if np.isfinite(cap):
                capped.append(f"{name}={cap:.3f}")
        print("Output-frame joint-step caps (rad):", ", ".join(capped))
        print(
            "Derived inner-QP velocity limits (rad/s):",
            ", ".join(f"{name}={value:.3f}" for name, value in output_joint_velocity_limits.items()),
        )
    raw_human_orientation_roles = [
        role.strip() for role in args.human_orientation_roles.split(",") if role.strip()
    ]
    human_orientation_roles = [
        LEGACY_ORIENTATION_ROLE_ALIASES.get(role, role)
        for role in raw_human_orientation_roles
    ]
    human_orientation_roles = list(dict.fromkeys(human_orientation_roles))
    if args.planted_sole_orientation_mode != "none" and args.planted_sole_ori_cost > 0.0:
        human_orientation_roles = list(dict.fromkeys(human_orientation_roles + SOLE_ROLES))
    if raw_human_orientation_roles != human_orientation_roles:
        alias_pairs = [
            f"{old}->{LEGACY_ORIENTATION_ROLE_ALIASES.get(old, old)}"
            for old in raw_human_orientation_roles
            if LEGACY_ORIENTATION_ROLE_ALIASES.get(old, old) != old
        ]
        if alias_pairs:
            print("Canonical orientation role aliases:", ", ".join(alias_pairs))
    human_orientation_costs = {}

    human_quats_by_role = None
    human_rest_R = {}
    robot_rest_R = {}
    orientation_metric_roles = []

    # Always transfer semantic palm/sole targets when this canonical NPZ provides
    # them, even for no-orientation controls. This keeps stored target frames
    # meaningful for end-effector orientation error reporting while task
    # orientation costs may remain zero.
    canonical_npz_for_orient = Path(args.canonical_npz)
    if not canonical_npz_for_orient.is_absolute():
        canonical_npz_for_orient = REPO_ROOT / canonical_npz_for_orient
    orientation_npz = np.load(canonical_npz_for_orient, allow_pickle=True)
    available_orientation_roles = (
        {str(role) for role in orientation_npz["roles"].tolist()}
        if "role_quats_wxyz" in orientation_npz.files
        else set()
    )
    requested_transfer_roles = list(
        dict.fromkeys(human_orientation_roles + CANONICAL_AUXILIARY_ROLES)
    )
    orientation_transfer_roles = [
        role for role in requested_transfer_roles if role in available_orientation_roles
    ]
    if human_orientation_roles and set(human_orientation_roles) - set(orientation_transfer_roles):
        missing = sorted(set(human_orientation_roles) - set(orientation_transfer_roles))
        raise RuntimeError(
            f"Canonical NPZ does not provide orientation frames for requested roles: {missing}"
        )

    if orientation_transfer_roles:
        # Use the same canonical input file that generated source_frames.
        human_quats_by_role = load_human_role_quats_for_solver(
            canonical_npz_path=canonical_npz_for_orient,
            source_frame_ids=source_meta["source_frame_ids"],
            roles=orientation_transfer_roles,
        )

        # Human rest rotations from the first selected source frame.
        for role in orientation_transfer_roles:
            human_rest_R[role] = quat_wxyz_to_rotmat(human_quats_by_role[role][0])
        orientation_metric_roles = [
            role for role in CANONICAL_AUXILIARY_ROLES if role in orientation_transfer_roles
        ]

        # Keep orientation costs at zero for rest alignment.  source_frames
        # intentionally carry identity quaternions; applying human orientation
        # costs before rest alignment would incorrectly target world identity.

    posture_task = S.make_posture_task(model, args)

    target_rest = S.robot_rest_frame_from_mujoco(
        model=model,
        qpos=qpos0.copy(),
        role_to_robot=robot_cfg,
        roles=CANONICAL_SOURCE_ROLES,
    )

    rest_score = None
    rest_errors = None
    qpos_rest = None

    if args.target_rest_mode == "aligned-source-rest":
        print()
        print("Aligning Alex target rest pose to first canonical source frame...")
        rest_configuration = S.mink.Configuration(model, q=qpos0.copy())
        rest_source_target = source_rest
        frame0_position_grounded_soles = []
        if planted_sole_position_enabled:
            rest_source_target = {
                role: {
                    "pos": list(value["pos"]),
                    "quat_wxyz": list(value["quat_wxyz"]),
                    "quat": list(value.get("quat", value["quat_wxyz"])),
                }
                for role, value in source_rest.items()
            }
            if (
                args.planted_sole_position_mode == "corners_plane_ground"
                and human_quats_by_role is not None
            ):
                for sole_role in SOLE_ROLES:
                    if sole_role in human_quats_by_role:
                        q = np.asarray(human_quats_by_role[sole_role][0], dtype=float)
                        rest_source_target[sole_role]["quat_wxyz"] = [float(x) for x in q]
                        rest_source_target[sole_role]["quat"] = [float(x) for x in q]

            frame0_position_grounded_soles = [
                role
                for role in SOLE_ROLES
                if (
                    sole_planted_masks is not None
                    and role in sole_planted_masks
                    and bool(sole_planted_masks[role][0])
                )
            ]
            if frame0_position_grounded_soles:
                print(
                    "Adding planted sole position grounding to rest alignment "
                    "for frame-0 semantic planted support roles:",
                    ", ".join(frame0_position_grounded_soles),
                )
            else:
                print(
                    "No semantic sole-planted support at frame 0 for z-ground position; "
                    "rest position target remains ungrounded."
                )

            rest_source_target = apply_planted_sole_position_targets(
                frame=rest_source_target,
                frame_idx=0,
                sole_planted_masks=sole_planted_masks,
                roles=SOLE_ROLES,
                mode=args.planted_sole_position_mode,
                ground_z=args.planted_sole_ground_z,
                alpha=args.planted_sole_position_alpha,
                sole_orientation_quats=(
                    {
                        role: human_quats_by_role[role][0]
                        for role in SOLE_ROLES
                        if human_quats_by_role is not None and role in human_quats_by_role
                    }
                    if args.planted_sole_position_mode == "corners_plane_ground"
                    else None
                ),
            )

        rest_target_by_role = S.set_task_targets(tasks, rest_source_target, robot_cfg)
        apply_sole_corner_position_costs_to_tasks(
            tasks,
            sole_planted_masks,
            0,
            mode=args.planted_sole_position_mode,
            corner_z_cost=args.planted_sole_corner_z_cost * args.foot_cost_mult,
        )

        S.set_posture_target(posture_task, rest_configuration, qpos0.copy())

        qpos_rest, rest_score, rest_errors, rest_solved_positions = S.solve_frame(
            model=model,
            configuration=rest_configuration,
            tasks=tasks,
            target_by_role=rest_target_by_role,
            solver=solver,
            limits=limits,
            max_iter=args.max_ik_iter,
            posture_task=posture_task,
        )

        print(f"Rest-alignment score: {rest_score:.6f}")
        for role, err in rest_errors.items():
            print(f"  rest {role:10s} error: {err:.6f} m")

        target_rest = S.robot_rest_frame_from_mujoco(
            model=model,
            qpos=qpos_rest.copy(),
            role_to_robot=robot_cfg,
            roles=CANONICAL_SOURCE_ROLES,
        )
        configuration = S.mink.Configuration(model, q=qpos_rest.copy())
        q_prev = qpos_rest.copy()

        frame0_planted_soles = []
        if args.planted_sole_orientation_mode != "none" and sole_planted_masks is not None:
            frame0_planted_soles = [
                role
                for role in SOLE_ROLES
                if role in sole_planted_masks and bool(sole_planted_masks[role][0])
            ]

        if frame0_planted_soles:
            if args.planted_sole_ori_cost <= 0.0:
                print()
                print(
                    "Frame 0 has semantic sole-planted roles, but "
                    "--planted-sole-ori-cost is 0; leaving rest alignment position-only."
                )
            else:
                print()
                print(
                    "Adding contact-aware flat-yaw sole orientation to rest alignment "
                    "for frame-0 planted support roles:",
                    ", ".join(frame0_planted_soles),
                )

                rest_flat_frame = {}
                for role, value in source_rest.items():
                    rest_flat_frame[role] = dict(value)
                    rest_flat_frame[role]["pos"] = list(value["pos"])
                    rest_flat_frame[role]["quat_wxyz"] = list(value["quat_wxyz"])
                    rest_flat_frame[role]["quat"] = list(value.get("quat", value["quat_wxyz"]))

                for role in frame0_planted_soles:
                    R_current = get_frame_world_rotation_from_task_target(configuration, tasks[role])
                    q_current = rotmat_to_quat_wxyz_single(R_current)
                    R_flat = flat_yaw_rotation_from_quat(q_current)
                    q_flat = rotmat_to_quat_wxyz_single(R_flat)
                    rest_flat_frame[role]["quat_wxyz"] = [float(x) for x in q_flat]
                    rest_flat_frame[role]["quat"] = [float(x) for x in q_flat]

                    task = tasks[role]
                    if not hasattr(task, "set_orientation_cost"):
                        raise AttributeError(
                            f"Task for role {role} does not expose set_orientation_cost(); "
                            "cannot apply contact-aware rest alignment."
                        )
                    task.set_orientation_cost(float(args.planted_sole_ori_cost))

                rest_target_by_role = S.set_task_targets(tasks, rest_flat_frame, robot_cfg)
                if posture_task is not None:
                    S.set_posture_target(posture_task, configuration, qpos0.copy())

                qpos_rest, rest_score, rest_errors, rest_solved_positions, rest_selection_info = S.solve_frame(
                    model=model,
                    configuration=configuration,
                    tasks=tasks,
                    target_by_role=rest_target_by_role,
                    solver=solver,
                    limits=limits,
                    max_iter=args.max_ik_iter,
                    posture_task=posture_task,
                    candidate_selection="combined",
                    candidate_orientation_weight=max(1.0, float(args.candidate_orientation_weight)),
                    candidate_temporal_weight=0.0,
                    candidate_root_weight=0.0,
                    candidate_root_step_scale_m=args.candidate_root_step_scale_m,
                    candidate_max_root_step_m=None,
                    candidate_limit_weight=0.0,
                    candidate_limit_margin_rad=args.candidate_limit_margin_rad,
                    return_diagnostics=True,
                )

                print(f"Contact-aware rest-alignment score: {rest_score:.6f}")
                print(f"  selected IK iteration: {rest_selection_info['selected_iteration']}")
                for role, err in rest_errors.items():
                    print(f"  rest {role:10s} error: {err:.6f} m")

                target_rest = S.robot_rest_frame_from_mujoco(
                    model=model,
                    qpos=qpos_rest.copy(),
                    role_to_robot=robot_cfg,
                    roles=CANONICAL_SOURCE_ROLES,
                )
                configuration = S.mink.Configuration(model, q=qpos_rest.copy())
                q_prev = qpos_rest.copy()
        elif args.planted_sole_orientation_mode != "none":
            print()
            print(
                "No semantic sole-planted support at frame 0; "
                "rest alignment remains position-only."
            )
    else:
        configuration = S.mink.Configuration(model, q=qpos0.copy())
        q_prev = qpos0.copy()

    q_neutral = q_prev.copy()

    if orientation_transfer_roles:
        print()
        print("Capturing Alex rest task orientations after final rest alignment...")
        for role in orientation_transfer_roles:
            if role not in tasks:
                if role in orientation_metric_roles:
                    raise RuntimeError(f"Palm orientation metric role {role} has no IK task")
                print(f"  {role:14s}: missing task")
                continue
            robot_rest_R[role] = get_frame_world_rotation_from_task_target(configuration, tasks[role])
            print(f"  {role:14s}: captured")

        missing_rest_frames = sorted(set(orientation_transfer_roles) - set(robot_rest_R))
        if missing_rest_frames:
            raise RuntimeError(
                "Cannot enable human orientation tracking without Alex rest task frames: "
                f"{missing_rest_frames}"
            )

        human_orientation_costs = apply_human_orientation_costs_to_tasks(
            tasks,
            human_orientation_roles,
            args,
        )

    morphology_scales = None
    if args.target_generation == "morphology-delta":
        morphology = S.compute_morphology_scales(
            source_rest=base_skeleton_frame(source_rest),
            target_rest=base_skeleton_frame(target_rest),
            preserve_root_translation=True,
            clamp_min=0.70,
            clamp_max=1.30,
        )
        morphology_scales = add_palm_morphology_scales(
            source_rest,
            target_rest,
            morphology.role_scales,
        )
        print()
        print("Morphology-delta role scales:")
        for role, scale in morphology_scales.items():
            print(f"  {role:16s}: {scale:.4f}")

    qpos_traj = []
    rows = []
    target_positions = []
    solved_ik_positions = []
    target_orientations_wxyz = []
    solved_ik_orientations_wxyz = []
    joint_limit_lower_margins = []
    joint_limit_upper_margins = []
    joint_limit_nearest_margins_by_frame = []
    selection_diagnostics = []
    sole_planted_ramp_state = {}

    for frame_idx, source_frame in enumerate(source_frames):
        if args.target_generation == "morphology-delta":
            base_scaled_frame = S.make_morphology_delta_target_frame(
                source_frame=base_skeleton_frame(source_frame),
                source_rest=base_skeleton_frame(source_rest),
                target_rest=base_skeleton_frame(target_rest),
                scales=morphology_scales,
            )
        elif args.target_generation == "rest-delta":
            base_scaled_frame = S.make_rest_delta_target_frame(
                source_frame=base_skeleton_frame(source_frame),
                source_rest=base_skeleton_frame(source_rest),
                target_rest=base_skeleton_frame(target_rest),
                motion_scale=args.motion_scale,
            )
        else:
            base_scaled_frame = S.scale_frame_by_rest_pose(
                frame=base_skeleton_frame(source_frame),
                source_rest_frame=base_skeleton_frame(source_rest),
                target_rest_frame=base_skeleton_frame(target_rest),
            )

        scaled_frame = add_semantic_aux_targets(
            base_target_frame=base_scaled_frame,
            source_frame=source_frame,
            source_rest=source_rest,
            target_rest=target_rest,
            target_generation=args.target_generation,
            motion_scale=args.motion_scale,
            morphology_scales=morphology_scales,
        )

        if contact_masks is not None:
            scaled_frame = contact_aware_target_stabilize(
                frame=scaled_frame,
                frame_idx=frame_idx,
                contact_masks=contact_masks,
                contact_state=contact_state,
                roles=contact_roles,
                mode=args.contact_anchor_mode,
                alpha=args.contact_anchor_alpha,
            )

        if human_quats_by_role is not None:
            scaled_frame = apply_human_relative_orientations_to_frame(
                frame=scaled_frame,
                frame_idx=frame_idx,
                human_quats=human_quats_by_role,
                human_rest_R=human_rest_R,
                robot_rest_R=robot_rest_R,
                roles=orientation_transfer_roles,
                transfer_mode=args.human_orientation_transfer_mode,
            )

        scaled_frame = apply_planted_sole_position_targets(
            frame=scaled_frame,
            frame_idx=frame_idx,
            sole_planted_masks=sole_planted_masks,
            roles=SOLE_ROLES,
            mode=args.planted_sole_position_mode,
            ground_z=args.planted_sole_ground_z,
            alpha=args.planted_sole_position_alpha,
            sole_orientation_quats=(
                {
                    role: human_quats_by_role[role][frame_idx]
                    for role in SOLE_ROLES
                    if human_quats_by_role is not None and role in human_quats_by_role
                }
                if args.planted_sole_position_mode == "corners_plane_ground"
                else None
            ),
        )

        scaled_frame = apply_planted_sole_orientation_targets(
            frame=scaled_frame,
            frame_idx=frame_idx,
            sole_planted_masks=sole_planted_masks,
            roles=SOLE_ROLES,
            mode=args.planted_sole_orientation_mode,
            ramp_state=sole_planted_ramp_state,
            ramp_frames=args.planted_sole_ramp_frames,
        )

        target_by_role = S.set_task_targets(tasks, scaled_frame, robot_cfg)
        apply_sole_corner_position_costs_to_tasks(
            tasks,
            sole_planted_masks,
            frame_idx,
            mode=args.planted_sole_position_mode,
            corner_z_cost=args.planted_sole_corner_z_cost * args.foot_cost_mult,
        )

        if posture_task is not None:
            q_ref = S.blended_posture_target(
                q_prev=q_prev,
                q_neutral=q_neutral,
                neutral_blend=args.posture_neutral_blend,
            )
            S.set_posture_target(posture_task, configuration, q_ref)

        qpos, score, errors, solved_positions, selection_info = S.solve_frame(
            model=model,
            configuration=configuration,
            tasks=tasks,
            target_by_role=target_by_role,
            solver=solver,
            limits=motion_limits,
            max_iter=args.max_ik_iter,
            posture_task=posture_task,
            candidate_selection=args.candidate_selection,
            candidate_position_tolerance=args.candidate_position_tolerance,
            candidate_orientation_weight=args.candidate_orientation_weight,
            candidate_temporal_weight=args.candidate_temporal_weight,
            candidate_root_weight=args.candidate_root_weight,
            candidate_root_step_scale_m=args.candidate_root_step_scale_m,
            candidate_max_root_step_m=args.candidate_max_root_step_m,
            candidate_limit_weight=args.candidate_limit_weight,
            candidate_limit_margin_rad=args.candidate_limit_margin_rad,
            output_joint_step_caps=output_joint_step_caps,
            return_diagnostics=True,
        )

        q_prev = qpos.copy()
        qpos_traj.append(qpos)

        target_positions.append([
            np.asarray(target_by_role[role]["target_pos"], dtype=float)
            for role in ik_roles
        ])
        solved_ik_positions.append([
            np.asarray(solved_positions[role], dtype=float)
            for role in ik_roles
        ])
        target_orientations_wxyz.append([
            np.asarray(scaled_frame[role]["quat_wxyz"], dtype=float)
            for role in ik_roles
        ])
        solved_ik_orientations_wxyz.append([
            rotmat_to_quat_wxyz_single(
                get_frame_world_rotation_from_task_target(configuration, tasks[role])
            )
            for role in ik_roles
        ])
        lower_margin, upper_margin, nearest_margin = joint_limit_margins(
            model,
            qpos,
            robot_cfg["actuated_joint_order"],
        )
        joint_limit_lower_margins.append(lower_margin)
        joint_limit_upper_margins.append(upper_margin)
        joint_limit_nearest_margins_by_frame.append(nearest_margin)
        selection_diagnostics.append(selection_info)

        nearest_index = int(np.argmin(nearest_margin))
        nearest_joint = robot_cfg["actuated_joint_order"][nearest_index]
        nearest_value = float(nearest_margin[nearest_index])
        nearest_side = "lower" if lower_margin[nearest_index] <= upper_margin[nearest_index] else "upper"

        palm_err = [errors[role] for role in PALM_ROLES if role in errors]
        sole_err = [errors[role] for role in SOLE_ROLES if role in errors]
        orientation_errors = {}
        for role in orientation_metric_roles:
            orientation_errors[role] = orientation_error_rad(
                scaled_frame[role]["quat_wxyz"],
                get_frame_world_rotation_from_task_target(configuration, tasks[role]),
            )
        palm_orientation_errors = {
            role: orientation_errors[role]
            for role in PALM_ROLES
            if role in orientation_errors
        }
        sole_orientation_errors = {
            role: orientation_errors[role]
            for role in SOLE_ROLES
            if role in orientation_errors
        }

        row = {
            "frame": frame_idx,
            "left_sole_semantic_planted": (
                bool(sole_planted_masks["left_sole"][frame_idx])
                if sole_planted_masks is not None and "left_sole" in sole_planted_masks
                else False
            ),
            "right_sole_semantic_planted": (
                bool(sole_planted_masks["right_sole"][frame_idx])
                if sole_planted_masks is not None and "right_sole" in sole_planted_masks
                else False
            ),
            "left_sole_z_grounded": (
                args.planted_sole_position_mode == "z_ground"
                and sole_planted_masks is not None
                and "left_sole" in sole_planted_masks
                and bool(sole_planted_masks["left_sole"][frame_idx])
            ),
            "right_sole_z_grounded": (
                args.planted_sole_position_mode == "z_ground"
                and sole_planted_masks is not None
                and "right_sole" in sole_planted_masks
                and bool(sole_planted_masks["right_sole"][frame_idx])
            ),
            "left_sole_planted_flat": (
                args.planted_sole_orientation_mode == "flat_yaw"
                and sole_planted_masks is not None
                and "left_sole" in sole_planted_masks
                and bool(sole_planted_masks["left_sole"][frame_idx])
            ),
            "right_sole_planted_flat": (
                args.planted_sole_orientation_mode == "flat_yaw"
                and sole_planted_masks is not None
                and "right_sole" in sole_planted_masks
                and bool(sole_planted_masks["right_sole"][frame_idx])
            ),
            "left_sole_support_mask": (
                bool(sole_planted_masks["left_sole"][frame_idx])
                if sole_planted_masks is not None and "left_sole" in sole_planted_masks
                else False
            ),
            "right_sole_support_mask": (
                bool(sole_planted_masks["right_sole"][frame_idx])
                if sole_planted_masks is not None and "right_sole" in sole_planted_masks
                else False
            ),
            "position_score": float(score),
            "pelvis_error_m": float(errors.get("pelvis", np.nan)),
            "left_sole_error_m": float(errors.get("left_sole", np.nan)),
            "right_sole_error_m": float(errors.get("right_sole", np.nan)),
            "mean_sole_error_m": float(np.nanmean(sole_err)) if sole_err else np.nan,
            "left_sole_orientation_error_rad": float(sole_orientation_errors.get("left_sole", np.nan)),
            "right_sole_orientation_error_rad": float(sole_orientation_errors.get("right_sole", np.nan)),
            "mean_sole_orientation_error_rad": (
                float(np.nanmean(list(sole_orientation_errors.values())))
                if sole_orientation_errors
                else np.nan
            ),
            # Backward-compatible aliases: the active lower end-effector task
            # is now a sole contact role, not the legacy foot segment role.
            "left_foot_error_m": float(errors.get("left_sole", np.nan)),
            "right_foot_error_m": float(errors.get("right_sole", np.nan)),
            "mean_foot_error_m": float(np.nanmean(sole_err)) if sole_err else np.nan,
            "left_palm_error_m": float(errors.get("left_palm", np.nan)),
            "right_palm_error_m": float(errors.get("right_palm", np.nan)),
            "mean_palm_error_m": float(np.nanmean(palm_err)) if palm_err else np.nan,
            "left_palm_orientation_error_rad": float(palm_orientation_errors.get("left_palm", np.nan)),
            "right_palm_orientation_error_rad": float(palm_orientation_errors.get("right_palm", np.nan)),
            "mean_palm_orientation_error_rad": (
                float(np.nanmean(list(palm_orientation_errors.values())))
                if palm_orientation_errors
                else np.nan
            ),
            # Backward-compatible aliases: the active end-effector task is
            # now a palm, not the legacy hand role.
            "left_hand_error_m": float(errors.get("left_palm", np.nan)),
            "right_hand_error_m": float(errors.get("right_palm", np.nan)),
            "mean_hand_error_m": float(np.nanmean(palm_err)) if palm_err else np.nan,
            "root_x": float(qpos[0]),
            "root_y": float(qpos[1]),
            "root_z": float(qpos[2]),
            "nearest_joint_limit": nearest_joint,
            "nearest_joint_limit_side": nearest_side,
            "nearest_joint_limit_margin_rad": nearest_value,
            "num_joints_within_limit_margin": int(np.sum(nearest_margin <= args.joint_limit_warning_margin_rad)),
            "selection_position_score": float(selection_info["selected_position_score"]),
            "selection_orientation_score": float(selection_info["selected_orientation_score"]),
            "selection_temporal_displacement": float(selection_info["selected_temporal_displacement"]),
            "selection_max_actuated_step_rad": float(selection_info["selected_max_actuated_step_rad"]),
            "selection_candidate_count": int(selection_info["candidate_count"]),
            "selection_eligible_candidate_count": int(selection_info["eligible_candidate_count"]),
            "selection_rejected_candidate_count": int(selection_info["rejected_candidate_count"]),
            "selection_iteration": int(selection_info["selected_iteration"]),
        }
        rows.append(row)

        if frame_idx % 10 == 0 or frame_idx == len(source_frames) - 1:
            print(
                f"frame {frame_idx:03d}: "
                f"score={score:.4f}, "
                f"pelvis={row['pelvis_error_m']:.3f}, "
                f"feet={row['mean_foot_error_m']:.3f}, "
                f"palms={row['mean_palm_error_m']:.3f}, "
                f"limit={nearest_joint}:{nearest_value:.3f}"
            )

    qpos_traj = np.asarray(qpos_traj, dtype=float)
    source_positions = S.frame_positions_array(source_frames, CANONICAL_SOURCE_ROLES)
    target_positions = np.asarray(target_positions, dtype=float)
    solved_ik_positions = np.asarray(solved_ik_positions, dtype=float)
    target_orientations_wxyz = np.asarray(target_orientations_wxyz, dtype=float)
    solved_ik_orientations_wxyz = np.asarray(solved_ik_orientations_wxyz, dtype=float)
    joint_limit_lower_margins = np.asarray(joint_limit_lower_margins, dtype=float)
    joint_limit_upper_margins = np.asarray(joint_limit_upper_margins, dtype=float)
    joint_limit_nearest_margins_by_frame = np.asarray(joint_limit_nearest_margins_by_frame, dtype=float)

    joint_delta = np.diff(qpos_traj[:, 7:], axis=0)
    root_delta = np.diff(qpos_traj[:, 0:3], axis=0)
    if sole_planted_masks is None:
        sole_planted_mask_array = np.zeros((len(source_frames), len(SOLE_ROLES)), dtype=bool)
    else:
        sole_planted_mask_array = np.stack(
            [
                np.asarray(sole_planted_masks.get(role, np.zeros(len(source_frames), dtype=bool)), dtype=bool)
                for role in SOLE_ROLES
            ],
            axis=1,
        )

    npz_path = out_dir / f"{args.output_stem}.npz"
    csv_path = out_dir / f"{args.output_stem}_errors.csv"
    json_path = out_dir / f"{args.output_stem}_summary.json"
    plot_path = out_dir / f"{args.output_stem}_errors.png"

    summary = {
        "note": "Canonical human skeleton roles plus explicit semantic palm/sole contact roles to Alex IK.",
        "canonical_npz": str(canonical_npz),
        "robot_config": str(robot_cfg_path),
        "model_path": str(model_path),
        "solver": solver,
        "start_frame": args.start_frame,
        "stride": args.stride,
        "max_frames_requested": args.max_frames,
        "num_frames": len(source_frames),
        "source_fps": source_meta["source_fps"],
        "output_fps": output_fps,
        "max_ik_iter": args.max_ik_iter,
        "posture_cost": args.posture_cost,
        "posture_neutral_blend": args.posture_neutral_blend,
        "posture_mode": args.posture_mode,
        "target_rest_mode": args.target_rest_mode,
        "target_generation": args.target_generation,
        "motion_scale": args.motion_scale,
        "contact_mask_npz": None if args.contact_mask_npz is None else str(args.contact_mask_npz),
        "contact_roles": contact_roles,
        "contact_anchor_mode": args.contact_anchor_mode,
        "contact_anchor_alpha": args.contact_anchor_alpha,
        "planted_sole_position_mode": args.planted_sole_position_mode,
        "planted_sole_ground_z": args.planted_sole_ground_z,
        "planted_sole_position_alpha": args.planted_sole_position_alpha,
        "planted_sole_corner_z_cost": args.planted_sole_corner_z_cost,
        "planted_sole_orientation_mode": args.planted_sole_orientation_mode,
        "planted_sole_ori_cost": args.planted_sole_ori_cost,
        "sole_planted_mask_mode": args.sole_planted_mask_mode,
        "sole_planted_ground_threshold_m": args.sole_planted_ground_threshold_m,
        "sole_planted_speed_threshold_mps": args.sole_planted_speed_threshold_mps,
        "sole_planted_tilt_threshold_deg": args.sole_planted_tilt_threshold_deg,
        "planted_sole_ramp_frames": args.planted_sole_ramp_frames,
        "sole_planted_diagnostics": sole_planted_diagnostics,
        "left_sole_semantic_planted_frames": int(sole_planted_mask_array[:, 0].sum()),
        "right_sole_semantic_planted_frames": int(sole_planted_mask_array[:, 1].sum()),
        "left_sole_z_grounded_frames": (
            int(sole_planted_mask_array[:, 0].sum())
            if args.planted_sole_position_mode == "z_ground"
            else 0
        ),
        "right_sole_z_grounded_frames": (
            int(sole_planted_mask_array[:, 1].sum())
            if args.planted_sole_position_mode == "z_ground"
            else 0
        ),
        "left_sole_planted_flat_frames": (
            int(sole_planted_mask_array[:, 0].sum())
            if args.planted_sole_orientation_mode == "flat_yaw"
            else 0
        ),
        "right_sole_planted_flat_frames": (
            int(sole_planted_mask_array[:, 1].sum())
            if args.planted_sole_orientation_mode == "flat_yaw"
            else 0
        ),
        "foot_cost_mult": args.foot_cost_mult,
        "head_cost_mult": args.head_cost_mult,
        "hand_cost_mult": args.hand_cost_mult,
        "pelvis_cost_mult": args.pelvis_cost_mult,
        "human_orientation_roles_requested": raw_human_orientation_roles,
        "human_orientation_roles": human_orientation_roles,
        "orientation_metric_roles": orientation_metric_roles,
        "human_orientation_transfer_mode": args.human_orientation_transfer_mode,
        "human_orientation_costs": human_orientation_costs,
        "orientation_tracking_active": bool(human_orientation_roles),
        "candidate_selection": args.candidate_selection,
        "candidate_position_tolerance": args.candidate_position_tolerance,
        "candidate_orientation_weight": args.candidate_orientation_weight,
        "candidate_temporal_weight": args.candidate_temporal_weight,
        "candidate_root_weight": args.candidate_root_weight,
        "candidate_root_step_scale_m": args.candidate_root_step_scale_m,
        "candidate_max_root_step_m": args.candidate_max_root_step_m,
        "candidate_limit_weight": args.candidate_limit_weight,
        "candidate_limit_margin_rad": args.candidate_limit_margin_rad,
        "max_leg_step_rad": args.max_leg_step_rad,
        "max_spine_neck_step_rad": args.max_spine_neck_step_rad,
        "max_arm_step_rad": args.max_arm_step_rad,
        "max_other_step_rad": args.max_other_step_rad,
        "output_step_velocity_limit_enabled": output_joint_velocity_limit is not None,
        "output_joint_velocity_limits_rad_s": output_joint_velocity_limits,
        "joint_limit_warning_margin_rad": args.joint_limit_warning_margin_rad,
        "num_frames_near_joint_limit": int(np.sum(np.min(joint_limit_nearest_margins_by_frame, axis=1) <= args.joint_limit_warning_margin_rad)),
        "min_joint_limit_margin_rad": float(np.min(joint_limit_nearest_margins_by_frame)),
        "morphology_scales": morphology_scales,
        "rest_alignment_score": None if rest_score is None else float(rest_score),
        "rest_alignment_errors_m": rest_errors,
        "qpos_shape": list(qpos_traj.shape),
        "qpos_layout": robot_cfg["floating_base"]["qpos_layout"],
        "source_roles": list(CANONICAL_SOURCE_ROLES),
        "ik_roles": list(ik_roles),
        "sole_corner_roles": list(SOLE_CORNER_ROLES if planted_sole_corner_enabled else []),
        "mean_position_score": float(np.mean([r["position_score"] for r in rows])),
        "max_position_score": float(np.max([r["position_score"] for r in rows])),
        "mean_hand_error_m": float(np.nanmean([r["mean_hand_error_m"] for r in rows])),
        "max_hand_error_m": float(np.nanmax([r["mean_hand_error_m"] for r in rows])),
        "mean_palm_error_m": float(np.nanmean([r["mean_palm_error_m"] for r in rows])),
        "max_palm_error_m": float(np.nanmax([r["mean_palm_error_m"] for r in rows])),
        "mean_palm_orientation_error_rad": float(np.nanmean([r["mean_palm_orientation_error_rad"] for r in rows])),
        "max_palm_orientation_error_rad": float(np.nanmax([r["mean_palm_orientation_error_rad"] for r in rows])),
        "mean_palm_orientation_error_deg": float(np.degrees(np.nanmean([r["mean_palm_orientation_error_rad"] for r in rows]))),
        "max_palm_orientation_error_deg": float(np.degrees(np.nanmax([r["mean_palm_orientation_error_rad"] for r in rows]))),
        "mean_foot_error_m": float(np.nanmean([r["mean_foot_error_m"] for r in rows])),
        "max_foot_error_m": float(np.nanmax([r["mean_foot_error_m"] for r in rows])),
        "mean_sole_error_m": float(np.nanmean([r["mean_sole_error_m"] for r in rows])),
        "max_sole_error_m": float(np.nanmax([r["mean_sole_error_m"] for r in rows])),
        "mean_sole_orientation_error_rad": float(np.nanmean([r["mean_sole_orientation_error_rad"] for r in rows])),
        "max_sole_orientation_error_rad": float(np.nanmax([r["mean_sole_orientation_error_rad"] for r in rows])),
        "mean_sole_orientation_error_deg": float(np.degrees(np.nanmean([r["mean_sole_orientation_error_rad"] for r in rows]))),
        "max_sole_orientation_error_deg": float(np.degrees(np.nanmax([r["mean_sole_orientation_error_rad"] for r in rows]))),
        "mean_pelvis_error_m": float(np.nanmean([r["pelvis_error_m"] for r in rows])),
        "max_pelvis_error_m": float(np.nanmax([r["pelvis_error_m"] for r in rows])),
        "max_abs_joint_step_rad": float(np.max(np.abs(joint_delta))) if len(joint_delta) else 0.0,
        "max_root_step_m": float(np.max(np.linalg.norm(root_delta, axis=1))) if len(root_delta) else 0.0,
        "source_meta": source_meta,
    }

    np.savez(
        npz_path,
        qpos=qpos_traj,
        fps=np.array([output_fps], dtype=float),
        joint_names=np.asarray(robot_cfg["actuated_joint_order"], dtype=object),
        source_positions=source_positions,
        source_roles=np.asarray(CANONICAL_SOURCE_ROLES, dtype=object),
        target_positions=target_positions,
        solved_ik_positions=solved_ik_positions,
        target_orientations_wxyz=target_orientations_wxyz,
        solved_ik_orientations_wxyz=solved_ik_orientations_wxyz,
        ik_roles=np.asarray(ik_roles, dtype=object),
        source_frame_ids=np.asarray(source_meta["source_frame_ids"], dtype=int),
        human_orientation_roles=np.asarray(human_orientation_roles, dtype=object),
        orientation_metric_roles=np.asarray(orientation_metric_roles, dtype=object),
        human_orientation_transfer_mode=np.asarray(args.human_orientation_transfer_mode, dtype=object),
        planted_sole_position_mode=np.asarray(args.planted_sole_position_mode, dtype=object),
        planted_sole_ground_z=np.asarray(args.planted_sole_ground_z, dtype=float),
        planted_sole_position_alpha=np.asarray(args.planted_sole_position_alpha, dtype=float),
        planted_sole_corner_z_cost=np.asarray(args.planted_sole_corner_z_cost, dtype=float),
        planted_sole_orientation_mode=np.asarray(args.planted_sole_orientation_mode, dtype=object),
        sole_planted_roles=np.asarray(SOLE_ROLES, dtype=object),
        sole_corner_roles=np.asarray(SOLE_CORNER_ROLES if planted_sole_corner_enabled else [], dtype=object),
        sole_planted_masks=sole_planted_mask_array,
        sole_planted_mask_mode=np.asarray(args.sole_planted_mask_mode, dtype=object),
        joint_limit_joint_names=np.asarray(robot_cfg["actuated_joint_order"], dtype=object),
        joint_limit_lower_margin_rad=joint_limit_lower_margins,
        joint_limit_upper_margin_rad=joint_limit_upper_margins,
        joint_limit_margin_rad=joint_limit_nearest_margins_by_frame,
        output_joint_step_caps_rad=(
            np.full(len(robot_cfg["actuated_joint_order"]), np.inf, dtype=float)
            if output_joint_step_caps is None
            else np.asarray(output_joint_step_caps[7:], dtype=float)
        ),
        output_joint_velocity_limits_rad_s=np.asarray(
            [output_joint_velocity_limits.get(name, np.inf) for name in robot_cfg["actuated_joint_order"]],
            dtype=float,
        ),
        robot_config_path=np.asarray(str(robot_cfg_path), dtype=object),
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(summary, indent=2))

    frames = np.asarray([r["frame"] for r in rows])
    plt.figure(figsize=(10, 6))
    plt.plot(frames, [r["pelvis_error_m"] for r in rows], label="pelvis")
    plt.plot(frames, [r["mean_foot_error_m"] for r in rows], label="feet mean")
    plt.plot(frames, [r["mean_palm_error_m"] for r in rows], label="palms mean")
    plt.xlabel("frame")
    plt.ylabel("position error (m)")
    plt.title(args.output_stem)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print()
    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        elif key != "source_meta":
            print(f"  {key}: {value}")

    print()
    print("Wrote:")
    print(" ", npz_path)
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", plot_path)


if __name__ == "__main__":
    main()

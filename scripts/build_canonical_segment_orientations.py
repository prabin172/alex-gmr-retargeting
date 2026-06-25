#!/usr/bin/env python3
"""Build semantic segment and end-effector frames for a canonical human NPZ.

The canonical position roles remain backward compatible (``left_hand`` and
``right_hand``), but their orientation fields become true marker-derived palm
frames.  This supersedes the earlier first-pass convention that copied each
forearm frame into the hand role.

Frame convention for the semantic end-effectors:

* feet: +X toe-forward, +Y body-left across the sole, +Z sole normal;
* palms: +X wrist-to-hand/finger-forward, +Y body-left across the palm,
  +Z palm normal.

These are reconstructed frames, not raw FBX-bone rotations.  They should be
transferred as rest-relative orientation deltas, never as absolute robot-link
orientations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", type=Path, help="Canonical NPZ with marker data.")
    parser.add_argument("output_npz", type=Path, help="Output NPZ with role frames/quaternions.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults beside output_npz.",
    )
    return parser.parse_args()


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def frame_from_yz(y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """Return a right-handed frame whose columns are world X/Y/Z axes."""
    y_axis = normalize(y_axis)
    z_axis = z_axis - np.sum(z_axis * y_axis, axis=-1, keepdims=True) * y_axis
    z_axis = normalize(z_axis)
    x_axis = normalize(np.cross(y_axis, z_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def frame_from_xy(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """Return a right-handed frame whose columns are world X/Y/Z axes."""
    x_axis = normalize(x_axis)
    y_axis = y_axis - np.sum(y_axis * x_axis, axis=-1, keepdims=True) * x_axis
    y_axis = normalize(y_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def rotmat_to_quat_wxyz(rotations: np.ndarray) -> np.ndarray:
    """Convert a ``[T, 3, 3]`` rotation stack to normalized WXYZ quaternions."""
    rotations = np.asarray(rotations, dtype=float)
    quats = np.empty((rotations.shape[0], 4), dtype=float)

    for index, rotation in enumerate(rotations):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12)) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif rotation[1, 1] > rotation[2, 2]:
            scale = np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12)) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12)) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
        quats[index] = quat / np.linalg.norm(quat)

    # q and -q encode the same rotation. Keep the stored trajectory continuous.
    for index in range(1, len(quats)):
        if np.dot(quats[index - 1], quats[index]) < 0.0:
            quats[index] *= -1.0
    return quats


def require_markers(marker_positions: np.ndarray, marker_names: list[str], names: list[str]) -> list[np.ndarray]:
    name_to_index = {name: index for index, name in enumerate(marker_names)}
    missing = [name for name in names if name not in name_to_index]
    if missing:
        raise RuntimeError(f"Canonical source is missing required semantic markers: {missing}")
    return [marker_positions[:, name_to_index[name], :] for name in names]


def sign_to_body_left(axis: np.ndarray, body_left: np.ndarray) -> np.ndarray:
    sign = np.sum(axis * body_left, axis=-1, keepdims=True)
    return np.where(sign < 0.0, -axis, axis)


def semantic_foot_frame(
    marker_positions: np.ndarray,
    marker_names: list[str],
    side: str,
    body_left: np.ndarray,
) -> np.ndarray:
    prefix = "L" if side == "left" else "R"
    heel, toe, mt1, mt5 = require_markers(
        marker_positions,
        marker_names,
        [f"{prefix}HEL", f"{prefix}TOE", f"{prefix}MT1", f"{prefix}MT5"],
    )
    toe_centre = (toe + mt1 + mt5) / 3.0
    forward = toe_centre - heel
    # The marker labels are mirrored.  Sign against the pelvis lateral axis so
    # both feet use +Y = body-left rather than flipping the right foot by pi.
    width = sign_to_body_left(mt5 - mt1, body_left)
    return frame_from_xy(forward, width)


def semantic_palm_frame(
    marker_positions: np.ndarray,
    marker_names: list[str],
    side: str,
    body_left: np.ndarray,
) -> np.ndarray:
    prefix = "L" if side == "left" else "R"
    iwr, owr, ihand, ohand = require_markers(
        marker_positions,
        marker_names,
        [f"{prefix}IWR", f"{prefix}OWR", f"{prefix}IHAND", f"{prefix}OHAND"],
    )
    wrist_centre = 0.5 * (iwr + owr)
    hand_centre = 0.5 * (ihand + ohand)
    forward = hand_centre - wrist_centre
    width = sign_to_body_left(ihand - ohand, body_left)
    return frame_from_xy(forward, width)


def semantic_palm_position(
    marker_positions: np.ndarray,
    marker_names: list[str],
    side: str,
) -> np.ndarray:
    """Return the marker-derived centre of the human palm/contact patch."""
    prefix = "L" if side == "left" else "R"
    ihand, ohand = require_markers(
        marker_positions,
        marker_names,
        [f"{prefix}IHAND", f"{prefix}OHAND"],
    )
    return 0.5 * (ihand + ohand)


def semantic_sole_position(
    marker_positions: np.ndarray,
    marker_names: list[str],
    side: str,
    body_left: np.ndarray,
) -> np.ndarray:
    """Return a marker-derived sole/contact centre for the human foot.

    The mocap markers are not physical contact points: heel markers are often
    high on the heel, while toe/metatarsal markers sit closer to the floor.
    We therefore use the four available foot markers as a footprint and project
    its centre along the semantic sole normal to the lowest marker plane.  This
    gives an explicit contact-like role without redefining the legacy foot role.
    """
    prefix = "L" if side == "left" else "R"
    heel, toe, mt1, mt5 = require_markers(
        marker_positions,
        marker_names,
        [f"{prefix}HEL", f"{prefix}TOE", f"{prefix}MT1", f"{prefix}MT5"],
    )
    markers = np.stack([heel, toe, mt1, mt5], axis=1)
    footprint_center = np.mean(markers, axis=1)
    foot_frame = semantic_foot_frame(marker_positions, marker_names, side, body_left)
    normal = foot_frame[:, :, 2]
    normal_heights = np.einsum(
        "td,tmd->tm",
        normal,
        markers - footprint_center[:, None, :],
    )
    bottom_offset = np.min(normal_heights, axis=1)
    return footprint_center + bottom_offset[:, None] * normal


def main() -> None:
    args = parse_args()
    source = np.load(args.input_npz, allow_pickle=True)
    data = {key: source[key] for key in source.files}

    if "positions" not in data or "roles" not in data:
        raise RuntimeError("Canonical input must contain positions and roles fields.")
    if "marker_positions" not in data or "marker_names" not in data:
        raise RuntimeError(
            "Semantic feet and palms require marker_positions and marker_names in the canonical NPZ."
        )

    input_positions = np.asarray(data["positions"], dtype=float)
    input_roles = [str(role) for role in data["roles"].tolist()]
    # Rebuilding from an already semantic NPZ is idempotent: preserve all
    # legacy roles in their original order, then refresh semantic aux roles.
    roles = [
        role
        for role in input_roles
        if role not in {"left_palm", "right_palm", "left_sole", "right_sole"}
    ]
    input_role_index = {role: index for index, role in enumerate(input_roles)}
    positions = np.asarray(
        [input_positions[:, input_role_index[role], :] for role in roles],
        dtype=float,
    ).transpose(1, 0, 2)
    role_index = {role: index for index, role in enumerate(roles)}
    required_roles = {
        "pelvis", "torso", "head", "left_hip", "left_knee", "left_foot",
        "right_hip", "right_knee", "right_foot", "left_shoulder", "left_elbow",
        "left_hand", "right_shoulder", "right_elbow", "right_hand",
    }
    missing_roles = sorted(required_roles - set(role_index))
    if missing_roles:
        raise RuntimeError(f"Canonical source is missing required roles: {missing_roles}")

    marker_positions = np.asarray(data["marker_positions"], dtype=float)
    marker_names = [str(name) for name in data["marker_names"].tolist()]
    role_positions = {role: positions[:, index, :] for role, index in role_index.items()}
    body_left = role_positions["left_hip"] - role_positions["right_hip"]

    left_palm_position = semantic_palm_position(marker_positions, marker_names, "left")
    right_palm_position = semantic_palm_position(marker_positions, marker_names, "right")
    left_sole_position = semantic_sole_position(marker_positions, marker_names, "left", body_left)
    right_sole_position = semantic_sole_position(marker_positions, marker_names, "right", body_left)
    positions = np.concatenate(
        [
            positions,
            left_palm_position[:, None, :],
            right_palm_position[:, None, :],
            left_sole_position[:, None, :],
            right_sole_position[:, None, :],
        ],
        axis=1,
    )
    roles.extend(["left_palm", "right_palm", "left_sole", "right_sole"])
    role_index = {role: index for index, role in enumerate(roles)}
    role_positions = {role: positions[:, index, :] for role, index in role_index.items()}
    body_left = role_positions["left_hip"] - role_positions["right_hip"]

    pelvis_frame = frame_from_yz(body_left, role_positions["torso"] - role_positions["pelvis"])
    torso_frame = frame_from_yz(
        role_positions["left_shoulder"] - role_positions["right_shoulder"],
        role_positions["head"] - role_positions["torso"],
    )
    left_foot_frame = semantic_foot_frame(marker_positions, marker_names, "left", body_left)
    right_foot_frame = semantic_foot_frame(marker_positions, marker_names, "right", body_left)
    left_palm_frame = semantic_palm_frame(marker_positions, marker_names, "left", body_left)
    right_palm_frame = semantic_palm_frame(marker_positions, marker_names, "right", body_left)
    role_frames = {
        "pelvis": pelvis_frame,
        "torso": torso_frame,
        "head": torso_frame,
        "left_hip": frame_from_yz(body_left, role_positions["left_hip"] - role_positions["left_knee"]),
        "left_knee": frame_from_yz(body_left, role_positions["left_knee"] - role_positions["left_foot"]),
        "left_foot": left_foot_frame,
        "left_sole": left_foot_frame,
        "right_hip": frame_from_yz(body_left, role_positions["right_hip"] - role_positions["right_knee"]),
        "right_knee": frame_from_yz(body_left, role_positions["right_knee"] - role_positions["right_foot"]),
        "right_foot": right_foot_frame,
        "right_sole": right_foot_frame,
        "left_shoulder": frame_from_yz(
            role_positions["torso"] - role_positions["pelvis"],
            role_positions["left_elbow"] - role_positions["left_shoulder"],
        ),
        "left_elbow": frame_from_yz(
            role_positions["torso"] - role_positions["pelvis"],
            role_positions["left_hand"] - role_positions["left_elbow"],
        ),
        "left_hand": left_palm_frame,
        "left_palm": left_palm_frame,
        "right_shoulder": frame_from_yz(
            role_positions["torso"] - role_positions["pelvis"],
            role_positions["right_elbow"] - role_positions["right_shoulder"],
        ),
        "right_elbow": frame_from_yz(
            role_positions["torso"] - role_positions["pelvis"],
            role_positions["right_hand"] - role_positions["right_elbow"],
        ),
        "right_hand": right_palm_frame,
        "right_palm": right_palm_frame,
    }

    frame_stack = np.broadcast_to(np.eye(3), (len(positions), len(roles), 3, 3)).copy()
    quat_stack = np.zeros((len(positions), len(roles), 4), dtype=float)
    quat_stack[..., 0] = 1.0
    for role, frame in role_frames.items():
        role_id = role_index[role]
        frame_stack[:, role_id] = frame
        quat_stack[:, role_id] = rotmat_to_quat_wxyz(frame)

    data["positions"] = positions
    data["roles"] = np.asarray(roles, dtype=object)
    data["role_frames"] = frame_stack
    data["role_quats_wxyz"] = quat_stack
    data["role_orientation_version"] = np.array(4, dtype=np.int64)
    data["role_orientation_convention"] = np.array(
        "Reconstructed semantic frames (world columns X/Y/Z, quaternion WXYZ). "
        "Feet: +X toe-forward, +Y body-left across sole, +Z sole normal. "
        "Palms: +X wrist-to-hand/finger-forward, +Y body-left across palm, +Z palm normal. "
        "left_palm/right_palm positions are 0.5*(IHAND+OHAND); legacy left_hand/right_hand positions are unchanged. "
        "left_sole/right_sole positions are marker-footprint centres projected to the lowest marker plane along the semantic sole normal; "
        "legacy left_foot/right_foot positions are unchanged. "
        "Other limbs use keypoint-derived segment frames. Transfer only rest-relative deltas.",
        dtype=object,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **data)

    summary_path = args.summary or args.output_npz.with_name(
        f"{args.output_npz.stem}_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_npz": str(args.input_npz),
        "output_npz": str(args.output_npz),
        "num_frames": int(len(positions)),
        "orientation_version": 4,
        "semantic_orientation_roles": [
            "left_foot",
            "right_foot",
            "left_sole",
            "right_sole",
            "left_hand",
            "right_hand",
            "left_palm",
            "right_palm",
        ],
        "semantic_position_roles": ["left_palm", "right_palm", "left_sole", "right_sole"],
        "hand_orientation_source": "LIWR/LOWR/LIHAND/LOHAND and RIWR/ROWR/RIHAND/ROHAND marker frames",
        "foot_orientation_source": "LHEL/LTOE/LMT1/LMT5 and RHEL/RTOE/RMT1/RMT5 marker frames",
        "sole_position_source": "LHEL/LTOE/LMT1/LMT5 and RHEL/RTOE/RMT1/RMT5 footprint centres projected to the lowest marker plane along the semantic sole normal",
        "note": "left_hand/right_hand and left_foot/right_foot preserve legacy positions; left_palm/right_palm and left_sole/right_sole are explicit contact-like end-effector roles.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Wrote:", args.output_npz)
    print("Wrote:", summary_path)
    print("Semantic end-effectors: feet + palms + soles")


if __name__ == "__main__":
    main()

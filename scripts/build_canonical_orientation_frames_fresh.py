#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ORIENTATION_ROLES = [
    "pelvis",
    "torso",
    "head",
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
]


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def fallback_axis(preferred, avoid):
    candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    avoid = normalize(avoid)
    best = None
    best_score = 1e9
    for c in candidates:
        score = abs(float(np.dot(c, avoid)))
        if score < best_score:
            best_score = score
            best = c
    return best if preferred is None else preferred


def frame_from_yz(y_hint, z_hint):
    """
    Build frame with columns [x forward, y left, z up-ish].
    Uses y and z hints, then orthonormalizes.
    """
    z = normalize(z_hint)
    if z is None:
        z = np.array([0.0, 0.0, 1.0])

    y = np.asarray(y_hint, dtype=np.float64)
    y = y - np.dot(y, z) * z
    y = normalize(y)
    if y is None:
        y = fallback_axis(None, z)
        y = y - np.dot(y, z) * z
        y = normalize(y)

    # canonical right-handed convention: x cross y = z, so x = y cross z
    x = normalize(np.cross(y, z))
    y = normalize(np.cross(z, x))
    return np.column_stack([x, y, z])


def frame_from_xy(x_hint, y_hint):
    """
    Build frame with columns [x forward/along segment, y side, z normal].
    """
    x = normalize(x_hint)
    if x is None:
        x = np.array([1.0, 0.0, 0.0])

    y = np.asarray(y_hint, dtype=np.float64)
    y = y - np.dot(y, x) * x
    y = normalize(y)
    if y is None:
        y = fallback_axis(None, x)
        y = y - np.dot(y, x) * x
        y = normalize(y)

    z = normalize(np.cross(x, y))
    y = normalize(np.cross(z, x))
    return np.column_stack([x, y, z])


def yaw_matrix(deg: float) -> np.ndarray:
    """3x3 rotation matrix around Z by deg degrees."""
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def detect_facing_yaw_deg(positions: np.ndarray, role_to_idx: dict, n_frames: int = 10) -> float:
    """
    Auto-detect the yaw correction (degrees) needed to make the actor face +X.

    Canonical convention: +X forward, +Y left, +Z up.
    The actor's left direction is left_hip - right_hip.
    We find the yaw angle that rotates this XY-projected left vector to point in +Y.

    Result is snapped to the nearest 90° to avoid applying tiny floating-point
    corrections to clips that are already correctly oriented.
    """
    li = role_to_idx["left_hip"]
    ri = role_to_idx["right_hip"]
    n = min(n_frames, positions.shape[0])

    left_vecs = positions[:n, li, :] - positions[:n, ri, :]  # (n, 3)
    left_mean = left_vecs.mean(axis=0)

    # Project to XY plane and normalize.
    lx, ly = float(left_mean[0]), float(left_mean[1])
    mag = np.hypot(lx, ly)
    if mag < 1e-6:
        return 0.0

    # Angle to rotate left_xy → +Y: theta = atan2(lx, ly).
    # (rotating (lx,ly) by theta gives (0, sqrt(lx²+ly²)) ≈ +Y direction)
    raw_deg = float(np.degrees(np.arctan2(lx, ly)))

    # Snap to nearest 90° — MoCap clips are always axis-aligned in practice.
    snapped = round(raw_deg / 90.0) * 90.0
    return float(snapped)


def apply_yaw_to_positions(positions: np.ndarray, role_to_idx: dict, deg: float) -> np.ndarray:
    """Rotate all positions around the first-frame pelvis by deg degrees around Z."""
    R = yaw_matrix(deg)
    origin = positions[0, role_to_idx["pelvis"]].copy()
    return origin + np.einsum("ij,tkj->tki", R, positions - origin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True, type=Path)
    ap.add_argument("--out-npz", required=True, type=Path)
    args = ap.parse_args()

    z = np.load(args.in_npz, allow_pickle=True)
    roles = [str(x) for x in z["roles"]]
    role_to_idx = {r: i for i, r in enumerate(roles)}
    positions = np.asarray(z["positions"], dtype=np.float64)

    required = [
        "pelvis", "torso", "neck", "head",
        "left_hip", "right_hip",
        "left_shoulder", "right_shoulder",
        "left_ankle", "left_toe",
        "right_ankle", "right_toe",
        "left_wrist", "left_hand_middle", "left_hand_thumb",
        "right_wrist", "right_hand_middle", "right_hand_thumb",
    ]
    missing = [r for r in required if r not in role_to_idx]
    if missing:
        raise RuntimeError(f"Missing roles needed for orientation frames: {missing}")

    # Auto-detect and correct the actor's facing direction.
    # Canonical: actor must face +X. Snap-correct any clip that faces a different axis.
    yaw_correction_deg = detect_facing_yaw_deg(positions, role_to_idx)
    if yaw_correction_deg != 0.0:
        positions = apply_yaw_to_positions(positions, role_to_idx, yaw_correction_deg)
        print(f"Auto-corrected facing direction: applied yaw {yaw_correction_deg:+.0f}°")
    else:
        print("Facing direction: already correct (+X forward), no yaw correction needed.")

    T = positions.shape[0]
    mats = np.zeros((T, len(ORIENTATION_ROLES), 3, 3), dtype=np.float64)

    for t in range(T):
        p = {r: positions[t, role_to_idx[r]] for r in roles}

        pelvis_y = p["left_hip"] - p["right_hip"]
        pelvis_z = p["torso"] - p["pelvis"]
        mats[t, ORIENTATION_ROLES.index("pelvis")] = frame_from_yz(pelvis_y, pelvis_z)

        torso_y = p["left_shoulder"] - p["right_shoulder"]
        torso_z = p["neck"] - p["torso"]
        mats[t, ORIENTATION_ROLES.index("torso")] = frame_from_yz(torso_y, torso_z)

        head_y = torso_y
        head_z = p["head"] - p["neck"]
        mats[t, ORIENTATION_ROLES.index("head")] = frame_from_yz(head_y, head_z)

        # Feet: x from ankle to toe, y from pelvis left-right axis.
        mats[t, ORIENTATION_ROLES.index("left_foot")] = frame_from_xy(
            p["left_toe"] - p["left_ankle"],
            pelvis_y,
        )
        mats[t, ORIENTATION_ROLES.index("right_foot")] = frame_from_xy(
            p["right_toe"] - p["right_ankle"],
            pelvis_y,
        )

        # Hands: x from wrist to middle finger, y from wrist to thumb.
        mats[t, ORIENTATION_ROLES.index("left_hand")] = frame_from_xy(
            p["left_hand_middle"] - p["left_wrist"],
            p["left_hand_thumb"] - p["left_wrist"],
        )
        mats[t, ORIENTATION_ROLES.index("right_hand")] = frame_from_xy(
            p["right_hand_middle"] - p["right_wrist"],
            p["right_hand_thumb"] - p["right_wrist"],
        )

    out = {k: z[k] for k in z.files}
    out["positions"] = positions  # store the yaw-corrected positions
    out["orientation_role_names"] = np.asarray(ORIENTATION_ROLES, dtype=object)
    out["orientation_mats"] = mats
    out["orientation_valid"] = np.ones((len(ORIENTATION_ROLES),), dtype=bool)
    out["facing_yaw_correction_deg"] = np.float64(yaw_correction_deg)

    meta = {}
    if "metadata_json" in out:
        try:
            meta = json.loads(str(out["metadata_json"]))
        except Exception:
            meta = {}
    meta["orientation_frame_version"] = "fresh_segment_frames_v2"
    meta["facing_yaw_correction_deg"] = yaw_correction_deg
    meta["orientation_roles"] = ORIENTATION_ROLES
    meta["orientation_notes"] = [
        "Facing direction auto-corrected: actor is rotated so they face +X (canonical forward).",
        "Yaw is snapped to nearest 90° — MoCap clips are always axis-aligned.",
        "Pelvis/torso/head frames use left-right and vertical segment hints.",
        "Foot frames use ankle-to-toe as x-axis and pelvis left-right as side reference.",
        "Hand frames use wrist-to-middle-finger as x-axis and wrist-to-thumb as side reference.",
        "These are semantic segment frames, not raw FBX bone rotations.",
    ]
    out["metadata_json"] = np.asarray(json.dumps(meta, indent=2), dtype=object)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **out)

    print("input:", args.in_npz)
    print("output:", args.out_npz)
    print("positions:", positions.shape)
    print("orientation_mats:", mats.shape)
    print("orientation roles:", ORIENTATION_ROLES)


if __name__ == "__main__":
    main()

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
    out["orientation_role_names"] = np.asarray(ORIENTATION_ROLES, dtype=object)
    out["orientation_mats"] = mats
    out["orientation_valid"] = np.ones((len(ORIENTATION_ROLES),), dtype=bool)

    meta = {}
    if "metadata_json" in out:
        try:
            meta = json.loads(str(out["metadata_json"]))
        except Exception:
            meta = {}
    meta["orientation_frame_version"] = "fresh_segment_frames_v1"
    meta["orientation_roles"] = ORIENTATION_ROLES
    meta["orientation_notes"] = [
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

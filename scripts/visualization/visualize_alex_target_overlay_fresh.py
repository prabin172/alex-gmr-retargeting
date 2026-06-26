#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def equal_axes_3d(ax, xyz, pad_frac=0.20):
    xyz = np.asarray(xyz, dtype=float)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 0.30) * (1.0 + pad_frac)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


CHAIN = [
    ("pelvis", "torso"),
    ("torso", "head"),

    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),

    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]


def plot_one(npz_path: Path, out_dir: Path, idx: int):
    z = np.load(npz_path, allow_pickle=True)
    roles = [str(x) for x in z["role_names"]]
    role_to_idx = {r: i for i, r in enumerate(roles)}

    target = np.asarray(z["target_positions"][idx], dtype=float)
    achieved = np.asarray(z["achieved_positions"][idx], dtype=float)
    source = int(z["source_frame_ids"][idx])

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=45, marker="x", label="target")
    ax.scatter(achieved[:, 0], achieved[:, 1], achieved[:, 2], s=25, marker="o", label="Alex achieved")

    for a, b in CHAIN:
        if a not in role_to_idx or b not in role_to_idx:
            continue
        ia, ib = role_to_idx[a], role_to_idx[b]

        for pts, style in [(target, "--"), (achieved, "-")]:
            p0, p1 = pts[ia], pts[ib]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                linestyle=style,
                linewidth=2,
            )

    for r, p in zip(roles, achieved):
        ax.text(p[0], p[1], p[2], r, fontsize=7)

    all_pts = np.vstack([target, achieved])
    equal_axes_3d(ax, all_pts)

    ax.set_title(f"Alex achieved vs target, frame {idx}, source {source}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_frame_{idx:04d}_src_{source:04d}.png"
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 5, 10, 15, -1])
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    T = z["qpos"].shape[0]

    for f in args.frames:
        idx = T - 1 if f < 0 else f
        if 0 <= idx < T:
            plot_one(args.npz, args.out_dir, idx)


if __name__ == "__main__":
    main()

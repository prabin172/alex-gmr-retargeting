#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def equal_axes_3d(ax, xyz, pad_frac=0.15):
    xyz = np.asarray(xyz, dtype=float)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 0.25) * (1.0 + pad_frac)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_frame(npz_path: Path, out_dir: Path, frame_idx: int):
    z = np.load(npz_path, allow_pickle=True)

    roles = [str(x) for x in z["roles"]]
    pos = z["positions"][frame_idx]
    seg_pairs = z["segment_role_pairs"]

    role_to_idx = {r: i for i, r in enumerate(roles)}

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=28, label="canonical points")

    for pair in seg_pairs:
        a, b = str(pair[0]), str(pair[1])
        ia, ib = role_to_idx[a], role_to_idx[b]
        p0, p1 = pos[ia], pos[ib]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=2)

    for r, p in zip(roles, pos):
        ax.text(p[0], p[1], p[2], r, fontsize=7)

    ax.set_title(f"Fresh FBX canonical human frame {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    equal_axes_3d(ax, pos)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"canonical_frame_{frame_idx:04d}.png"
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 200, 500, 900, -1])
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    T = z["positions"].shape[0]

    for f in args.frames:
        idx = T - 1 if f < 0 else f
        if idx < 0 or idx >= T:
            print("skip invalid frame:", f)
            continue
        plot_frame(args.npz, args.out_dir, idx)


if __name__ == "__main__":
    main()

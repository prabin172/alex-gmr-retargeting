#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EDGES = [
    ("pelvis", "torso"),
    ("torso", "head"),

    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_palm"),

    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_palm"),

    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_toe"),
    ("left_heel", "left_toe"),
    ("left_foot", "left_heel"),
    ("left_foot", "left_toe"),

    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_toe"),
    ("right_heel", "right_toe"),
    ("right_foot", "right_heel"),
    ("right_foot", "right_toe"),
]


def to_str_list(arr):
    return [str(x) for x in arr.tolist()]


def draw_skeleton(ax, pts, roles, label, color, linestyle="-", alpha=1.0):
    role_to_i = {r: i for i, r in enumerate(roles)}

    for a, b in EDGES:
        if a not in role_to_i or b not in role_to_i:
            continue

        pa = pts[role_to_i[a]]
        pb = pts[role_to_i[b]]

        if np.isfinite(pa).all() and np.isfinite(pb).all():
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]],
                color=color,
                linestyle=linestyle,
                linewidth=2.5,
                alpha=alpha,
            )

    valid = np.isfinite(pts).all(axis=1)
    if valid.any():
        ax.scatter(
            pts[valid, 0],
            pts[valid, 1],
            pts[valid, 2],
            color=color,
            s=18,
            alpha=alpha,
            label=label,
        )


def set_axes_equal(ax, point_sets, z_floor_zero=True):
    pts = np.concatenate([p.reshape(-1, 3) for p in point_sets], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if pts.size == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 2)
        return

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)
    radius = max(float((mx - mn).max()) * 0.62, 0.5)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)

    z_min = 0.0 if z_floor_zero else center[2] - radius
    ax.set_zlim(z_min, center[2] + radius)


def render_frame(raw_pts, target_pts, solved_pts, roles, frame_idx, title, elev, azim):
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    draw_skeleton(
        ax,
        raw_pts,
        roles,
        label="raw human target",
        color="tab:blue",
        linestyle=":",
        alpha=0.45,
    )
    draw_skeleton(
        ax,
        target_pts,
        roles,
        label="morphology-scaled target",
        color="tab:orange",
        linestyle="-",
        alpha=0.9,
    )
    draw_skeleton(
        ax,
        solved_pts,
        roles,
        label="solved robot landmarks",
        color="tab:green",
        linestyle="--",
        alpha=0.9,
    )

    set_axes_equal(ax, [raw_pts, target_pts, solved_pts], z_floor_zero=True)

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{title} | frame {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--elev", type=float, default=18.0)
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)

    roles = to_str_list(d["ik_roles"])
    raw = np.asarray(d["raw_human_target_positions"], dtype=float)
    target = np.asarray(d["target_positions"], dtype=float)
    solved = np.asarray(d["solved_ik_positions"], dtype=float)

    n = solved.shape[0]
    if args.frames is None:
        frames = list(range(0, n, max(1, args.stride)))
    else:
        frames = [f for f in args.frames if 0 <= f < n]

    if not frames:
        raise RuntimeError("No valid frames selected.")

    images = []
    for k, f in enumerate(frames):
        img = render_frame(
            raw_pts=raw[f],
            target_pts=target[f],
            solved_pts=solved[f],
            roles=roles,
            frame_idx=f,
            title=args.npz.name,
            elev=args.elev,
            azim=args.azim,
        )
        images.append(img)

        if k % 25 == 0:
            print(f"rendered {k + 1}/{len(frames)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.suffix.lower() == ".mp4":
        imageio.mimsave(args.out, images, fps=args.fps, codec="libx264", quality=8)
    else:
        imageio.mimsave(args.out, images, fps=args.fps)

    print("wrote", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_MODEL = Path("assets/alex/alex_floating_base_with_sites.xml")


IMPORTANT_BODY_TERMS = [
    "PELVIS", "TORSO", "NECK", "HEAD",
    "HIP", "THIGH", "SHIN", "ANKLE", "FOOT",
    "SHOULDER", "ELBOW", "WRIST", "GRIPPER",
]


def mj_name(model, objtype, idx):
    name = mujoco.mj_id2name(model, objtype, idx)
    return "" if name is None else name


def important_body_ids(model):
    ids = []
    for b in range(1, model.nbody):
        name = mj_name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if any(term in name.upper() for term in IMPORTANT_BODY_TERMS):
            ids.append(b)
    return ids


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


def plot_frame(model, qpos, out_path, title):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    body_ids = important_body_ids(model)

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Draw parent-child body tree for important bodies.
    important = set(body_ids)
    for b in body_ids:
        parent = int(model.body_parentid[b])
        if parent <= 0:
            continue

        # Draw to parent if parent is important; otherwise still draw because it shows chain.
        p0 = data.xpos[parent]
        p1 = data.xpos[b]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            linewidth=2,
        )

    pts = np.asarray([data.xpos[b].copy() for b in body_ids])
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=30)

    for b in body_ids:
        name = mj_name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        p = data.xpos[b]
        ax.text(p[0], p[1], p[2], name, fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    equal_axes_3d(ax, pts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("wrote:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 5, 10, 15, -1])
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=float)
    source_frame_ids = np.asarray(z["source_frame_ids"], dtype=int)

    model = mujoco.MjModel.from_xml_path(str(args.model))

    T = qpos.shape[0]
    for f in args.frames:
        idx = T - 1 if f < 0 else f
        if idx < 0 or idx >= T:
            print("skip invalid frame:", f)
            continue

        src = int(source_frame_ids[idx])
        out_path = args.out_dir / f"alex_bodypos_qp_frame_{idx:04d}_src_{src:04d}.png"
        title = f"Alex body-position QP frame {idx}, source {src}"
        plot_frame(model, qpos[idx], out_path, title)


if __name__ == "__main__":
    main()

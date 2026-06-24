#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

SKELETON_EDGES = [
    ("PELVIS_LINK", "TORSO_LINK"),
    ("TORSO_LINK", "HEAD_LINK"),

    ("PELVIS_LINK", "LEFT_HIP_X_LINK"),
    ("LEFT_HIP_X_LINK", "LEFT_SHIN"),
    ("LEFT_SHIN", "LEFT_FOOT"),

    ("PELVIS_LINK", "RIGHT_HIP_X_LINK"),
    ("RIGHT_HIP_X_LINK", "RIGHT_SHIN"),
    ("RIGHT_SHIN", "RIGHT_FOOT"),

    ("TORSO_LINK", "LEFT_SHOULDER_Y_LINK"),
    ("LEFT_SHOULDER_Y_LINK", "LEFT_ELBOW_Y_LINK"),
    ("LEFT_ELBOW_Y_LINK", "LEFT_WRIST_X_LINK"),
    ("LEFT_WRIST_X_LINK", "LEFT_GRIPPER_Z_LINK"),

    ("TORSO_LINK", "RIGHT_SHOULDER_Y_LINK"),
    ("RIGHT_SHOULDER_Y_LINK", "RIGHT_ELBOW_Y_LINK"),
    ("RIGHT_ELBOW_Y_LINK", "RIGHT_WRIST_X_LINK"),
    ("RIGHT_WRIST_X_LINK", "RIGHT_GRIPPER_Z_LINK"),
]


def body_pos(model, data, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Missing body {name}")
    return np.asarray(data.xpos[bid], dtype=float)


def site_pos(model, data, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Missing site {name}")
    return np.asarray(data.site_xpos[sid], dtype=float)


def site_rot(model, data, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Missing site {name}")
    return np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3)


def set_equal_2d(ax, xs, ys, pad=0.20):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    r = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2 + pad
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal", adjustable="box")


def draw_projected_axis(ax, p, axis_vec, dims, color, label=None, scale=0.12):
    a = p[list(dims)]
    b = (p + scale * axis_vec)[list(dims)]
    ax.arrow(
        a[0],
        a[1],
        b[0] - a[0],
        b[1] - a[1],
        head_width=0.015,
        length_includes_head=True,
        color=color,
        linewidth=1.5,
    )
    if label:
        ax.text(b[0], b[1], label, fontsize=7, color=color)


def main() -> None:
    cfg_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex_with_sites.json"
    cfg = json.loads(cfg_path.read_text())

    model_path = REPO_ROOT / cfg["model_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    primary_sites = list(cfg["retarget_site_names"].values())
    optional_sites = list(cfg.get("optional_retarget_site_names", {}).values())
    site_names = primary_sites + optional_sites

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    views = [
        (axes[0], "Front view Y-Z", (1, 2), ("Y left", "Z up")),
        (axes[1], "Side view X-Z", (0, 2), ("X forward", "Z up")),
        (axes[2], "Top view X-Y", (0, 1), ("X forward", "Y left")),
    ]

    all_points = []

    for ax, title, dims, labels in views:
        for a, b in SKELETON_EDGES:
            pa = body_pos(model, data, a)
            pb = body_pos(model, data, b)
            ax.plot(
                [pa[dims[0]], pb[dims[0]]],
                [pa[dims[1]], pb[dims[1]]],
                linewidth=2,
                alpha=0.75,
            )
            all_points.extend([pa, pb])

        for site in site_names:
            p = site_pos(model, data, site)
            R = site_rot(model, data, site)

            # Columns are local site axes expressed in world coordinates.
            x_axis = R[:, 0]
            y_axis = R[:, 1]
            z_axis = R[:, 2]

            ax.scatter(p[dims[0]], p[dims[1]], s=45, marker="x", color="black")
            short = site.replace("alex_", "").replace("_site", "")
            ax.text(p[dims[0]], p[dims[1]], short, fontsize=7, color="black")

            draw_projected_axis(ax, p, x_axis, dims, color="red", label="x")
            draw_projected_axis(ax, p, y_axis, dims, color="green", label="y")
            draw_projected_axis(ax, p, z_axis, dims, color="blue", label="z")

            all_points.append(p)
            all_points.append(p + 0.12 * x_axis)
            all_points.append(p + 0.12 * y_axis)
            all_points.append(p + 0.12 * z_axis)

        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True)

    pts = np.asarray(all_points)
    set_equal_2d(axes[0], pts[:, 1], pts[:, 2])
    set_equal_2d(axes[1], pts[:, 0], pts[:, 2])
    set_equal_2d(axes[2], pts[:, 0], pts[:, 1])

    fig.suptitle("Alex retargeting sites with local axes: red=x, green=y, blue=z")
    fig.tight_layout()

    out = REPO_ROOT / "outputs/debug/alex_retarget_sites_axes_neutral.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print("Wrote", out)

    print()
    print("Site world poses in neutral pose:")
    print("Axes are local site axes expressed in world coordinates.")
    for site in site_names:
        p = site_pos(model, data, site)
        R = site_rot(model, data, site)
        print()
        print(f"{site}")
        print(f"  pos xyz: [{p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f}]")
        print(f"  local x in world: [{R[0,0]: .3f}, {R[1,0]: .3f}, {R[2,0]: .3f}]")
        print(f"  local y in world: [{R[0,1]: .3f}, {R[1,1]: .3f}, {R[2,1]: .3f}]")
        print(f"  local z in world: [{R[0,2]: .3f}, {R[1,2]: .3f}, {R[2,2]: .3f}]")


if __name__ == "__main__":
    main()

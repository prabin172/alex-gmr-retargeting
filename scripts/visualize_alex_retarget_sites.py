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


def body_pos(model, data, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.asarray(data.xpos[bid], dtype=float)


def site_pos(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return np.asarray(data.site_xpos[sid], dtype=float)


def set_equal_2d(ax, xs, ys, pad=0.15):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    r = max(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 2 + pad
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_aspect("equal", adjustable="box")


def main() -> None:
    cfg_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex_with_sites.json"
    cfg = json.loads(cfg_path.read_text())
    model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / cfg["model_path"]))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    site_names = list(cfg["retarget_site_names"].values()) + list(cfg.get("optional_retarget_site_names", {}).values())

    all_points = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, title, dims, labels in [
        (axes[0], "Front view Y-Z", (1, 2), ("Y", "Z")),
        (axes[1], "Side view X-Z", (0, 2), ("X", "Z")),
    ]:
        for a, b in SKELETON_EDGES:
            pa = body_pos(model, data, a)
            pb = body_pos(model, data, b)
            ax.plot([pa[dims[0]], pb[dims[0]]], [pa[dims[1]], pb[dims[1]]], linewidth=2)
            all_points.extend([pa, pb])

        for site in site_names:
            p = site_pos(model, data, site)
            ax.scatter(p[dims[0]], p[dims[1]], s=80, marker="x")
            ax.text(p[dims[0]], p[dims[1]], site.replace("alex_", "").replace("_site", ""), fontsize=8)
            all_points.append(p)

        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True)

    pts = np.asarray(all_points)
    set_equal_2d(axes[0], pts[:, 1], pts[:, 2])
    set_equal_2d(axes[1], pts[:, 0], pts[:, 2])

    out = REPO_ROOT / "outputs/debug/alex_retarget_sites_neutral.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print("Wrote", out)

    print()
    print("Site world positions in neutral pose:")
    for site in site_names:
        p = site_pos(model, data, site)
        print(f"  {site:28s} xyz=[{p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f}]")


if __name__ == "__main__":
    main()

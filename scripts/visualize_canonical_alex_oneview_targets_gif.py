#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MVNX_SOLVER_PATH = Path(__file__).resolve().parent / "solve_mvnx_alex_motion.py"

spec = importlib.util.spec_from_file_location("mvnx_solver_module", MVNX_SOLVER_PATH)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

CANONICAL_BODY_NAMES = list(S.CANONICAL_BODY_NAMES)

BODY_EDGES = [
    ("pelvis", "torso"), ("torso", "head"),
    ("pelvis", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
    ("pelvis", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("torso", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("torso", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
]

TARGET_EDGES = [
    ("pelvis", "head"),
    ("left_knee", "left_foot"),
    ("right_knee", "right_foot"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
]


def merge_robot_config(robot_cfg_path):
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

    sites = cfg.get("sites", [])
    site_names = {s.get("name") for s in sites if isinstance(s, dict)}

    role_to_site = {}
    candidates = {
        "pelvis": "alex_pelvis_site",
        "head": "alex_head_site",
        "left_foot": "alex_left_sole_site",
        "right_foot": "alex_right_sole_site",
        "left_hand": "alex_left_palm_site",
        "right_hand": "alex_right_palm_site",
    }

    for role, site_name in candidates.items():
        if site_name in site_names:
            role_to_site[role] = site_name

    if role_to_site:
        merged["retarget_site_names"] = role_to_site

    if "model_path" not in merged or "site" in robot_cfg_path.name.lower():
        merged["model_path"] = "assets/alex/alex_floating_base_with_sites.xml"

    return merged


def resolve_model_path(robot_cfg):
    model_path = Path(robot_cfg["model_path"])
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    return model_path


def compute_robot_positions(model, qpos_traj, robot_cfg, roles):
    data = mujoco.MjData(model)
    out = np.zeros((len(qpos_traj), len(roles), 3), dtype=float)

    for ti, q in enumerate(qpos_traj):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for ri, role in enumerate(roles):
            frame_name, frame_type = S.role_frame_name_and_type(robot_cfg, role)
            out[ti, ri] = S.frame_pos(model, data, frame_name, frame_type)

    return out


def role_index(roles):
    return {r: i for i, r in enumerate(roles)}


def draw_skeleton_3d(ax, positions, roles, edges, title, annotate=False):
    idx = role_index(roles)

    for a, b in edges:
        if a not in idx or b not in idx:
            continue
        pa = positions[idx[a]]
        pb = positions[idx[b]]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            linewidth=2,
        )

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=18)

    if annotate:
        for r, i in idx.items():
            p = positions[i]
            ax.text(p[0], p[1], p[2], r, fontsize=6)

    ax.set_title(title)
    ax.set_xlabel("X forward")
    ax.set_ylabel("Y left")
    ax.set_zlabel("Z up")
    ax.view_init(elev=18, azim=-70)


def draw_targets_3d(ax, target_positions, ik_roles):
    idx = role_index(ik_roles)

    for a, b in TARGET_EDGES:
        if a not in idx or b not in idx:
            continue
        pa = target_positions[idx[a]]
        pb = target_positions[idx[b]]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            linestyle="--",
            linewidth=1,
        )

    ax.scatter(
        target_positions[:, 0],
        target_positions[:, 1],
        target_positions[:, 2],
        s=45,
        marker="x",
        label="IK targets",
    )
    ax.legend(loc="upper right", fontsize=8)


def set_equal_3d_limits(ax, center, half):
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def main():
    parser = argparse.ArgumentParser(description="Visualize canonical human, Alex robot, and IK targets.")
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif-fps", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()

    npz_path = args.npz_path
    if not npz_path.is_absolute():
        npz_path = REPO_ROOT / npz_path

    robot_cfg_path = args.robot_config
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = REPO_ROOT / robot_cfg_path

    out_path = args.output
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    data_npz = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(data_npz["qpos"], dtype=float)
    source_positions = np.asarray(data_npz["source_positions"], dtype=float)
    source_roles = [str(x) for x in data_npz["source_roles"].tolist()]
    target_positions = np.asarray(data_npz["target_positions"], dtype=float)
    ik_roles = [str(x) for x in data_npz["ik_roles"].tolist()]

    robot_cfg = merge_robot_config(robot_cfg_path)
    model_path = resolve_model_path(robot_cfg)
    model = mujoco.MjModel.from_xml_path(str(model_path))

    robot_roles = list(CANONICAL_BODY_NAMES)
    robot_positions = compute_robot_positions(
        model=model,
        qpos_traj=qpos,
        robot_cfg=robot_cfg,
        roles=robot_roles,
    )

    frame_ids = np.arange(len(qpos))[:: max(1, args.frame_skip)]
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    all_positions = np.concatenate(
        [
            source_positions[frame_ids].reshape(-1, 3),
            robot_positions[frame_ids].reshape(-1, 3),
            target_positions[frame_ids].reshape(-1, 3),
        ],
        axis=0,
    )

    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.55 * max(span, 1.0)

    fig = plt.figure(figsize=(12, 6))
    ax_human = fig.add_subplot(1, 2, 1, projection="3d")
    ax_robot = fig.add_subplot(1, 2, 2, projection="3d")

    def update(anim_i):
        t = int(frame_ids[anim_i])
        ax_human.clear()
        ax_robot.clear()

        draw_skeleton_3d(
            ax_human,
            source_positions[t],
            source_roles,
            BODY_EDGES,
            title=f"Canonical human | frame {t}",
            annotate=args.annotate,
        )

        draw_skeleton_3d(
            ax_robot,
            robot_positions[t],
            robot_roles,
            BODY_EDGES,
            title=f"Alex IK + targets | frame {t}",
            annotate=args.annotate,
        )
        draw_targets_3d(ax_robot, target_positions[t], ik_roles)

        set_equal_3d_limits(ax_human, center, half)
        set_equal_3d_limits(ax_robot, center, half)

        fig.suptitle(npz_path.stem, fontsize=12)
        fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / args.gif_fps)
    ani.save(out_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)

    print("Loaded:", npz_path)
    print("qpos shape:", qpos.shape)
    print("source_positions shape:", source_positions.shape)
    print("robot_positions shape:", robot_positions.shape)
    print("target_positions shape:", target_positions.shape)
    print("frames in GIF:", len(frame_ids))
    print("view: ax.view_init(elev=18, azim=-70)")
    print("Wrote GIF:", out_path)


if __name__ == "__main__":
    main()

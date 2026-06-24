from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import mujoco
import numpy as np

from general_motion_retargeting.source_adapters.canonical_human import CANONICAL_BODY_NAMES

repo_root = Path(__file__).resolve().parents[1]

SKELETON_EDGES = [
    ("pelvis", "torso"),
    ("torso", "head"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
]

IK_TARGET_EDGES = [
    ("pelvis", "head"),
    ("pelvis", "left_foot"),
    ("pelvis", "right_foot"),
    ("head", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),
    ("head", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
]

LABEL_ROLES = ["pelvis", "head", "left_foot", "right_foot", "left_hand", "right_hand"]


def body_pos(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=float)


def compute_robot_positions(model, qpos_traj, role_to_robot, roles):
    data = mujoco.MjData(model)
    out = np.zeros((len(qpos_traj), len(roles), 3), dtype=float)

    for t, qpos in enumerate(qpos_traj):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        for i, role in enumerate(roles):
            out[t, i] = body_pos(model, data, role_to_robot[role])

    return out


def role_index(roles):
    return {str(role): i for i, role in enumerate(roles)}


def make_equal_limits(arrays, dim0, dim1, pad_frac=0.12):
    vals0 = []
    vals1 = []

    for arr in arrays:
        if arr is None or arr.size == 0:
            continue
        vals0.append(arr[..., dim0].reshape(-1))
        vals1.append(arr[..., dim1].reshape(-1))

    vals0 = np.concatenate(vals0)
    vals1 = np.concatenate(vals1)

    lo0, hi0 = float(np.min(vals0)), float(np.max(vals0))
    lo1, hi1 = float(np.min(vals1)), float(np.max(vals1))

    span0 = hi0 - lo0
    span1 = hi1 - lo1
    span = max(span0, span1, 1e-3)
    span = span * (1.0 + pad_frac)

    c0 = 0.5 * (lo0 + hi0)
    c1 = 0.5 * (lo1 + hi1)

    return (c0 - 0.5 * span, c0 + 0.5 * span), (c1 - 0.5 * span, c1 + 0.5 * span)


def follow_limits(positions, roles, dim0, horizontal_half_width, z_half_height):
    idx = role_index(roles)
    pelvis = positions[idx["pelvis"]]
    center0 = float(pelvis[dim0])
    center_z = float(pelvis[2])

    return (
        (center0 - horizontal_half_width, center0 + horizontal_half_width),
        (center_z - z_half_height, center_z + z_half_height),
    )


def draw_skeleton(ax, positions, roles, dim0, dim1, marker_size=18, annotate=True):
    idx = role_index(roles)

    for a, b in SKELETON_EDGES:
        if a not in idx or b not in idx:
            continue
        pa = positions[idx[a]]
        pb = positions[idx[b]]
        ax.plot([pa[dim0], pb[dim0]], [pa[dim1], pb[dim1]], linewidth=2)

    ax.scatter(positions[:, dim0], positions[:, dim1], s=marker_size)

    if annotate:
        for role in LABEL_ROLES:
            if role not in idx:
                continue
            p = positions[idx[role]]
            ax.text(p[dim0], p[dim1], role, fontsize=7)


def draw_targets(ax, target_positions, ik_roles, dim0, dim1):
    idx = role_index(ik_roles)

    for a, b in IK_TARGET_EDGES:
        if a not in idx or b not in idx:
            continue
        pa = target_positions[idx[a]]
        pb = target_positions[idx[b]]
        ax.plot([pa[dim0], pb[dim0]], [pa[dim1], pb[dim1]], linestyle="--", linewidth=1)

    ax.scatter(target_positions[:, dim0], target_positions[:, dim1], s=36, marker="x", label="IK targets")


def setup_axis(ax, title, xlabel, ylabel, xlim, ylim):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")


def main():
    parser = argparse.ArgumentParser(description="Visualize MVNX-to-Alex IK as a 2x2 GIF.")
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--gif-fps", type=int, default=15)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera", choices=["follow", "fixed"], default="follow")
    parser.add_argument("--front-half-width", type=float, default=0.85)
    parser.add_argument("--side-half-width", type=float, default=0.85)
    parser.add_argument("--z-half-height", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    npz_path = args.npz_path
    if not npz_path.is_absolute():
        npz_path = repo_root / npz_path

    data_npz = np.load(npz_path, allow_pickle=True)

    qpos = np.asarray(data_npz["qpos"], dtype=float)
    source_positions = np.asarray(data_npz["source_positions"], dtype=float)
    source_roles = [str(x) for x in data_npz["source_roles"].tolist()]
    target_positions = np.asarray(data_npz["target_positions"], dtype=float)
    ik_roles = [str(x) for x in data_npz["ik_roles"].tolist()]

    robot_cfg = json.loads((repo_root / "general_motion_retargeting/robot_configs/alex.json").read_text())
    model_path = repo_root / robot_cfg["model_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))

    robot_roles = list(CANONICAL_BODY_NAMES)
    robot_positions = compute_robot_positions(
        model=model,
        qpos_traj=qpos,
        role_to_robot=robot_cfg["retarget_body_names"],
        roles=robot_roles,
    )

    frame_ids = np.arange(len(qpos))[:: args.frame_skip]
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    if args.output is None:
        suffix = "2x2_follow" if args.camera == "follow" else "2x2_fixed"
        out_path = npz_path.with_name(npz_path.stem + f"_{suffix}.gif")
    else:
        out_path = args.output
        if not out_path.is_absolute():
            out_path = repo_root / out_path

    if args.camera == "fixed":
        front_xlim, front_ylim = make_equal_limits(
            [source_positions[frame_ids], robot_positions[frame_ids], target_positions[frame_ids]],
            dim0=1,
            dim1=2,
        )
        side_xlim, side_ylim = make_equal_limits(
            [source_positions[frame_ids], robot_positions[frame_ids], target_positions[frame_ids]],
            dim0=0,
            dim1=2,
        )
    else:
        front_xlim = front_ylim = side_xlim = side_ylim = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    def update(anim_i):
        t = int(frame_ids[anim_i])

        for ax in axes.reshape(-1):
            ax.clear()

        if args.camera == "follow":
            src_front_xlim, src_front_ylim = follow_limits(
                source_positions[t], source_roles, dim0=1,
                horizontal_half_width=args.front_half_width,
                z_half_height=args.z_half_height,
            )
            src_side_xlim, src_side_ylim = follow_limits(
                source_positions[t], source_roles, dim0=0,
                horizontal_half_width=args.side_half_width,
                z_half_height=args.z_half_height,
            )
            rob_front_xlim, rob_front_ylim = follow_limits(
                robot_positions[t], robot_roles, dim0=1,
                horizontal_half_width=args.front_half_width,
                z_half_height=args.z_half_height,
            )
            rob_side_xlim, rob_side_ylim = follow_limits(
                robot_positions[t], robot_roles, dim0=0,
                horizontal_half_width=args.side_half_width,
                z_half_height=args.z_half_height,
            )
        else:
            src_front_xlim, src_front_ylim = front_xlim, front_ylim
            src_side_xlim, src_side_ylim = side_xlim, side_ylim
            rob_front_xlim, rob_front_ylim = front_xlim, front_ylim
            rob_side_xlim, rob_side_ylim = side_xlim, side_ylim

        draw_skeleton(axes[0, 0], source_positions[t], source_roles, dim0=1, dim1=2)
        setup_axis(
            axes[0, 0],
            title=f"MVNX human front | frame {t}",
            xlabel="Y left",
            ylabel="Z up",
            xlim=src_front_xlim,
            ylim=src_front_ylim,
        )

        draw_skeleton(axes[0, 1], source_positions[t], source_roles, dim0=0, dim1=2)
        setup_axis(
            axes[0, 1],
            title=f"MVNX human side | frame {t}",
            xlabel="X forward",
            ylabel="Z up",
            xlim=src_side_xlim,
            ylim=src_side_ylim,
        )

        draw_skeleton(axes[1, 0], robot_positions[t], robot_roles, dim0=1, dim1=2)
        draw_targets(axes[1, 0], target_positions[t], ik_roles, dim0=1, dim1=2)
        setup_axis(
            axes[1, 0],
            title="Alex IK front + targets",
            xlabel="Y left",
            ylabel="Z up",
            xlim=rob_front_xlim,
            ylim=rob_front_ylim,
        )
        axes[1, 0].legend(loc="upper right", fontsize=8)

        draw_skeleton(axes[1, 1], robot_positions[t], robot_roles, dim0=0, dim1=2)
        draw_targets(axes[1, 1], target_positions[t], ik_roles, dim0=0, dim1=2)
        setup_axis(
            axes[1, 1],
            title="Alex IK side + targets",
            xlabel="X forward",
            ylabel="Z up",
            xlim=rob_side_xlim,
            ylim=rob_side_ylim,
        )
        axes[1, 1].legend(loc="upper right", fontsize=8)

        fig.suptitle(
            f"MVNX-to-Alex IK diagnostic | {npz_path.name} | camera={args.camera}",
            fontsize=12,
        )
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / args.gif_fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)

    print("Loaded:", npz_path)
    print("qpos shape:", qpos.shape)
    print("source_positions shape:", source_positions.shape)
    print("robot_positions shape:", robot_positions.shape)
    print("target_positions shape:", target_positions.shape)
    print("frames in source trajectory:", len(qpos))
    print("frames in GIF:", len(frame_ids))
    print("camera:", args.camera)
    print()
    print("Wrote GIF:")
    print(" ", out_path)


if __name__ == "__main__":
    main()

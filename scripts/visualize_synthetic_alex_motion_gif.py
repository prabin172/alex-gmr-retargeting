from pathlib import Path
import json

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from general_motion_retargeting.source_adapters.canonical_human import (
    CANONICAL_BODY_NAMES,
    make_neutral_standing_frame,
    validate_canonical_human_frame,
)
from general_motion_retargeting.retargeting.rest_pose_scaling import (
    scale_frame_by_rest_pose,
)

repo_root = Path(__file__).resolve().parents[1]

robot_cfg_path = repo_root / "general_motion_retargeting/robot_configs/alex.json"
robot_cfg = json.loads(robot_cfg_path.read_text())

model_path = repo_root / robot_cfg["model_path"]
qpos_path = repo_root / "outputs/debug/synthetic_alex_motion_ik_qpos.npz"
out_dir = repo_root / "outputs/debug"
out_dir.mkdir(parents=True, exist_ok=True)

gif_path = out_dir / "synthetic_alex_motion_side_by_side.gif"

if not model_path.exists():
    raise FileNotFoundError(f"Missing model: {model_path}")

if not qpos_path.exists():
    raise FileNotFoundError(
        f"Missing qpos trajectory: {qpos_path}\n"
        "Run python scripts/solve_synthetic_alex_motion.py first."
    )

def body_pos(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=float)

def robot_rest_frame_from_mujoco(model, qpos, role_to_robot):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    frame = {}
    for role in CANONICAL_BODY_NAMES:
        body_name = role_to_robot[role]
        frame[role] = {
            "pos": [float(x) for x in body_pos(model, data, body_name)],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

    validate_canonical_human_frame(frame)
    return frame

def make_source_motion(num_frames):
    frames = []

    for i in range(num_frames):
        phase = i / max(num_frames - 1, 1)
        lift = np.sin(np.pi * phase)

        frame = make_neutral_standing_frame()

        hand_y = 0.52 + 0.06 * lift
        elbow_y = 0.40 + 0.04 * lift
        hand_z = 1.00 + 0.22 * lift
        elbow_z = 1.12 + 0.16 * lift
        hand_x = 0.10 + 0.02 * lift
        elbow_x = 0.06 + 0.02 * lift

        frame["left_hand"]["pos"] = [hand_x, hand_y, hand_z]
        frame["left_elbow"]["pos"] = [elbow_x, elbow_y, elbow_z]
        frame["right_hand"]["pos"] = [hand_x, -hand_y, hand_z]
        frame["right_elbow"]["pos"] = [elbow_x, -elbow_y, elbow_z]

        validate_canonical_human_frame(frame)
        frames.append(frame)

    return frames

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

qpos_traj = np.load(qpos_path, allow_pickle=True)["qpos"]
num_frames = qpos_traj.shape[0]

qpos0 = np.asarray(model.qpos0, dtype=float)
qpos0[0:3] = [0.0, 0.0, 1.0]
qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

source_rest = make_neutral_standing_frame()
target_rest = robot_rest_frame_from_mujoco(model, qpos0, robot_cfg["retarget_body_names"])
source_frames = make_source_motion(num_frames)

ik_roles = ["pelvis", "head", "left_foot", "right_foot", "left_hand", "right_hand"]

context_roles = [
    "pelvis",
    "torso",
    "head",
    "left_hip",
    "left_knee",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
]

edges = [
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

def source_points_for_frame(frame_idx):
    frame = source_frames[frame_idx]
    return {
        role: np.asarray(frame[role]["pos"], dtype=float)
        for role in context_roles
    }

def solved_points_for_frame(frame_idx):
    data.qpos[:] = qpos_traj[frame_idx]
    mujoco.mj_forward(model, data)

    points = {}
    for role in context_roles:
        body_name = robot_cfg["retarget_body_names"][role]
        points[role] = body_pos(model, data, body_name)
    return points

def target_points_for_frame(frame_idx):
    scaled_frame = scale_frame_by_rest_pose(
        frame=source_frames[frame_idx],
        source_rest_frame=source_rest,
        target_rest_frame=target_rest,
    )
    return {
        role: np.asarray(scaled_frame[role]["pos"], dtype=float)
        for role in ik_roles
    }

all_source = []
all_alex = []

for i in range(num_frames):
    source = source_points_for_frame(i)
    solved = solved_points_for_frame(i)
    target = target_points_for_frame(i)

    all_source.extend(list(source.values()))
    all_alex.extend(list(solved.values()))
    all_alex.extend(list(target.values()))

all_source = np.asarray(all_source)
all_alex = np.asarray(all_alex)

pad = 0.12

source_y_lim = (np.min(all_source[:, 1]) - pad, np.max(all_source[:, 1]) + pad)
source_z_lim = (max(0.0, np.min(all_source[:, 2]) - pad), np.max(all_source[:, 2]) + pad)

alex_x_lim = (np.min(all_alex[:, 0]) - pad, np.max(all_alex[:, 0]) + pad)
alex_y_lim = (np.min(all_alex[:, 1]) - pad, np.max(all_alex[:, 1]) + pad)
alex_z_lim = (max(0.0, np.min(all_alex[:, 2]) - pad), np.max(all_alex[:, 2]) + pad)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

def draw_source_panel(ax, source, dim0, dim1, xlabel, ylabel, title):
    ax.clear()

    for a, b in edges:
        pa = source[a]
        pb = source[b]
        ax.plot([pa[dim0], pb[dim0]], [pa[dim1], pb[dim1]], linewidth=2)

    xs = [source[r][dim0] for r in context_roles]
    ys = [source[r][dim1] for r in context_roles]
    ax.scatter(xs, ys, marker="o", s=25, label="source human")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="upper right")

def draw_alex_panel(ax, solved, target, dim0, dim1, xlabel, ylabel, title):
    ax.clear()

    for a, b in edges:
        pa = solved[a]
        pb = solved[b]
        ax.plot([pa[dim0], pb[dim0]], [pa[dim1], pb[dim1]], linewidth=2)

    solved_x = [solved[r][dim0] for r in context_roles]
    solved_y = [solved[r][dim1] for r in context_roles]
    ax.scatter(solved_x, solved_y, marker="o", s=25, label="solved Alex")

    target_x = [target[r][dim0] for r in ik_roles]
    target_y = [target[r][dim1] for r in ik_roles]
    ax.scatter(target_x, target_y, marker="x", s=70, label="IK targets")

    for role in ik_roles:
        s = solved[role]
        t = target[role]
        ax.plot([s[dim0], t[dim0]], [s[dim1], t[dim1]], linestyle="--", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="upper right")

def update(frame_idx):
    source = source_points_for_frame(frame_idx)
    solved = solved_points_for_frame(frame_idx)
    target = target_points_for_frame(frame_idx)

    draw_source_panel(
        axes[0],
        source,
        dim0=1,
        dim1=2,
        xlabel="Y left",
        ylabel="Z up",
        title=f"Input human, frame {frame_idx:03d}",
    )

    draw_alex_panel(
        axes[1],
        solved,
        target,
        dim0=1,
        dim1=2,
        xlabel="Y left",
        ylabel="Z up",
        title=f"Alex front, frame {frame_idx:03d}",
    )

    draw_alex_panel(
        axes[2],
        solved,
        target,
        dim0=0,
        dim1=2,
        xlabel="X forward",
        ylabel="Z up",
        title=f"Alex side, frame {frame_idx:03d}",
    )

    axes[0].set_xlim(source_y_lim)
    axes[0].set_ylim(source_z_lim)

    axes[1].set_xlim(alex_y_lim)
    axes[1].set_ylim(alex_z_lim)

    axes[2].set_xlim(alex_x_lim)
    axes[2].set_ylim(alex_z_lim)

    fig.suptitle("Synthetic motion retargeting diagnostic: source input → Alex IK")
    fig.tight_layout()

animation = FuncAnimation(fig, update, frames=num_frames, interval=100)
animation.save(gif_path, writer=PillowWriter(fps=10))
plt.close(fig)

print("Loaded qpos:")
print(" ", qpos_path)
print("qpos shape:", qpos_traj.shape)
print()
print("Wrote patched 3-panel GIF:")
print(" ", gif_path)
print()
print("Done.")

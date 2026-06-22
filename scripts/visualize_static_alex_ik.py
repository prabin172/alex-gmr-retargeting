from pathlib import Path
import json

import mujoco
import numpy as np
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]

robot_cfg_path = repo_root / "general_motion_retargeting/robot_configs/alex.json"
robot_cfg = json.loads(robot_cfg_path.read_text())

model_path = repo_root / robot_cfg["model_path"]
summary_path = repo_root / "outputs/debug/static_alex_ik_summary.json"
out_dir = repo_root / "outputs/debug"
out_dir.mkdir(parents=True, exist_ok=True)

if not model_path.exists():
    raise FileNotFoundError(f"Missing model: {model_path}")

if not summary_path.exists():
    raise FileNotFoundError(
        f"Missing IK summary: {summary_path}\n"
        "Run python scripts/solve_static_alex_ik.py first."
    )

summary = json.loads(summary_path.read_text())
qpos = np.asarray(summary["qpos"], dtype=float)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
data.qpos[:] = qpos
mujoco.mj_forward(model, data)

role_to_robot = robot_cfg["retarget_body_names"]
target_by_role = summary["target_by_role"]

roles = [
    "pelvis",
    "head",
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
]

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

def body_pos(role):
    body_name = role_to_robot[role]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=float)

solved_pts = {role: body_pos(role) for role in context_roles}
target_pts = {
    role: np.asarray(target_by_role[role]["target_pos"], dtype=float)
    for role in roles
}

def draw_view(path, title, dim0, dim1, label0, label1):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw solved Alex semantic skeleton.
    for a, b in edges:
        pa = solved_pts[a]
        pb = solved_pts[b]
        ax.plot(
            [pa[dim0], pb[dim0]],
            [pa[dim1], pb[dim1]],
            linewidth=2,
            label="solved Alex skeleton" if a == "pelvis" and b == "torso" else None,
        )

    solved_x = [solved_pts[r][dim0] for r in context_roles]
    solved_y = [solved_pts[r][dim1] for r in context_roles]
    ax.scatter(solved_x, solved_y, marker="o", s=35, label="solved Alex body points")

    # Draw target IK points.
    target_x = [target_pts[r][dim0] for r in roles]
    target_y = [target_pts[r][dim1] for r in roles]
    ax.scatter(target_x, target_y, marker="x", s=90, label="IK targets")

    # Connect each target to solved point for error visualization.
    for role in roles:
        t = target_pts[role]
        s = solved_pts[role]
        ax.plot(
            [t[dim0], s[dim0]],
            [t[dim1], s[dim1]],
            linestyle="--",
            linewidth=1,
        )
        ax.text(t[dim0], t[dim1], f"target {role}", fontsize=8)
        ax.text(s[dim0], s[dim1], f"solved {role}", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

front_path = out_dir / "static_alex_ik_front_yz.png"
side_path = out_dir / "static_alex_ik_side_xz.png"

draw_view(front_path, "Static Alex IK: front view", 1, 2, "Y left", "Z up")
draw_view(side_path, "Static Alex IK: side view", 0, 2, "X forward", "Z up")

print("Loaded:")
print(" ", summary_path)
print()
print("Wrote:")
print(" ", front_path)
print(" ", side_path)
print()
print("Target errors from summary:")
for row in summary["target_errors"]:
    print(f"  {row['role']:12s} error_m={row['position_error_m']:.4f}")

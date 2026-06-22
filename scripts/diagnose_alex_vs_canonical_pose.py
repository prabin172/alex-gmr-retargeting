from pathlib import Path
import json
import csv

import mujoco
import numpy as np
import matplotlib.pyplot as plt

from general_motion_retargeting.source_adapters.canonical_human import (
    make_neutral_standing_frame,
    validate_canonical_human_frame,
)

repo_root = Path(__file__).resolve().parents[1]

robot_cfg_path = repo_root / "general_motion_retargeting/robot_configs/alex.json"
robot_cfg = json.loads(robot_cfg_path.read_text())

model_path = repo_root / robot_cfg["model_path"]
out_dir = repo_root / "outputs/debug"
out_dir.mkdir(parents=True, exist_ok=True)

print("Robot config:", robot_cfg_path)
print("Model:", model_path)
print("Model exists:", model_path.exists())

if not model_path.exists():
    raise FileNotFoundError(
        "Missing floating-base Alex model. Run:\n"
        "  python scripts/prepare_alex_mujoco_assets.py\n"
        "  python scripts/prepare_alex_floating_base_model.py"
    )

human = make_neutral_standing_frame()
validate_canonical_human_frame(human)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

qpos = np.array(model.qpos0, dtype=float)
qpos[0:3] = [0.0, 0.0, 1.0]
qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[:] = qpos
mujoco.mj_forward(model, data)

role_to_robot = robot_cfg["retarget_body_names"]

role_to_human = {
    "pelvis": "pelvis",
    "torso": "torso",
    "head": "head",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_foot": "left_foot",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_foot": "right_foot",
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_hand",
    "left_hand": "left_hand",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_hand",
    "right_hand": "right_hand",
}

roles_for_plot = [
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

human_pts = {}
alex_pts = {}
rows = []

for role in roles_for_plot:
    human_name = role_to_human[role]
    robot_body = role_to_robot[role]

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
    if body_id < 0:
        raise RuntimeError(f"Robot body not found: {robot_body}")

    h = np.array(human[human_name]["pos"], dtype=float)
    a = np.array(data.xpos[body_id], dtype=float)

    human_pts[role] = h
    alex_pts[role] = a

    diff = a - h
    rows.append({
        "role": role,
        "human_name": human_name,
        "robot_body": robot_body,
        "human_x": float(h[0]),
        "human_y": float(h[1]),
        "human_z": float(h[2]),
        "alex_x": float(a[0]),
        "alex_y": float(a[1]),
        "alex_z": float(a[2]),
        "diff_x": float(diff[0]),
        "diff_y": float(diff[1]),
        "diff_z": float(diff[2]),
        "distance": float(np.linalg.norm(diff)),
    })

print()
print("Canonical human vs Alex neutral pose")
print("This is only a geometry sanity check, not retargeting.")
print()
for row in rows:
    print(
        f"{row['role']:15s} "
        f"H=[{row['human_x']: .3f},{row['human_y']: .3f},{row['human_z']: .3f}] "
        f"A=[{row['alex_x']: .3f},{row['alex_y']: .3f},{row['alex_z']: .3f}] "
        f"|diff|={row['distance']:.3f}"
    )

csv_path = out_dir / "alex_vs_canonical_neutral_positions.csv"
json_path = out_dir / "alex_vs_canonical_neutral_positions.json"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(json.dumps(rows, indent=2))

def draw_2d(ax, view_name, dim0, dim1, label0, label1):
    for pts, marker, name in [
        (human_pts, "o", "canonical human"),
        (alex_pts, "^", "Alex neutral"),
    ]:
        xs = [pts[r][dim0] for r in roles_for_plot]
        ys = [pts[r][dim1] for r in roles_for_plot]
        ax.scatter(xs, ys, marker=marker, label=name)

        for a, b in edges:
            ax.plot(
                [pts[a][dim0], pts[b][dim0]],
                [pts[a][dim1], pts[b][dim1]],
            )

    for role in ["pelvis", "head", "left_foot", "right_foot", "left_hand", "right_hand"]:
        p = human_pts[role]
        ax.text(p[dim0], p[dim1], role, fontsize=8)

    ax.set_title(view_name)
    ax.set_xlabel(label0)
    ax.set_ylabel(label1)
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

front_path = out_dir / "alex_vs_canonical_front_yz.png"
side_path = out_dir / "alex_vs_canonical_side_xz.png"

fig, ax = plt.subplots(figsize=(7, 7))
draw_2d(ax, "Front view: Y-left vs Z-up", 1, 2, "Y left", "Z up")
fig.tight_layout()
fig.savefig(front_path, dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7))
draw_2d(ax, "Side view: X-forward vs Z-up", 0, 2, "X forward", "Z up")
fig.tight_layout()
fig.savefig(side_path, dpi=200)
plt.close(fig)

print()
print("Wrote:")
print(" ", csv_path)
print(" ", json_path)
print(" ", front_path)
print(" ", side_path)
print()
print("Interpretation:")
print("  This should show whether the body mapping and axes are sane.")
print("  It will not look perfectly aligned because the canonical human pose is synthetic.")

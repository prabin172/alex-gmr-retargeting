from pathlib import Path
import json
import csv
from collections import OrderedDict

import mujoco
import numpy as np

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
qpos[0:3] = human["pelvis"]["pos"]
qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[:] = qpos
mujoco.mj_forward(model, data)

role_to_robot = robot_cfg["retarget_body_names"]

roles = [
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

segments = OrderedDict([
    ("pelvis_to_torso", ("pelvis", "torso")),
    ("torso_to_head", ("torso", "head")),

    ("hip_width", ("left_hip", "right_hip")),
    ("shoulder_width", ("left_shoulder", "right_shoulder")),
    ("foot_width", ("left_foot", "right_foot")),
    ("hand_width", ("left_hand", "right_hand")),

    ("left_thigh", ("left_hip", "left_knee")),
    ("left_shin", ("left_knee", "left_foot")),
    ("right_thigh", ("right_hip", "right_knee")),
    ("right_shin", ("right_knee", "right_foot")),

    ("left_upper_arm", ("left_shoulder", "left_elbow")),
    ("left_forearm_hand", ("left_elbow", "left_hand")),
    ("right_upper_arm", ("right_shoulder", "right_elbow")),
    ("right_forearm_hand", ("right_elbow", "right_hand")),
])

def get_robot_pos(role: str) -> np.ndarray:
    body_name = role_to_robot[role]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Robot body not found for role={role}: {body_name}")
    return np.array(data.xpos[body_id], dtype=float)

def get_human_pos(role: str) -> np.ndarray:
    return np.array(human[role]["pos"], dtype=float)

human_pts = {role: get_human_pos(role) for role in roles}
alex_pts = {role: get_robot_pos(role) for role in roles}

human_pelvis = human_pts["pelvis"]
alex_pelvis = alex_pts["pelvis"]

landmark_rows = []
for role in roles:
    h_rel = human_pts[role] - human_pelvis
    a_rel = alex_pts[role] - alex_pelvis
    diff = a_rel - h_rel

    landmark_rows.append({
        "role": role,
        "robot_body": role_to_robot[role],
        "human_rel_x": float(h_rel[0]),
        "human_rel_y": float(h_rel[1]),
        "human_rel_z": float(h_rel[2]),
        "alex_rel_x": float(a_rel[0]),
        "alex_rel_y": float(a_rel[1]),
        "alex_rel_z": float(a_rel[2]),
        "diff_rel_x": float(diff[0]),
        "diff_rel_y": float(diff[1]),
        "diff_rel_z": float(diff[2]),
        "diff_norm": float(np.linalg.norm(diff)),
    })

segment_rows = []
for name, (a, b) in segments.items():
    h_len = float(np.linalg.norm(human_pts[b] - human_pts[a]))
    r_len = float(np.linalg.norm(alex_pts[b] - alex_pts[a]))
    ratio = r_len / h_len if h_len > 1e-9 else None

    segment_rows.append({
        "segment": name,
        "from": a,
        "to": b,
        "human_length": h_len,
        "alex_length": r_len,
        "alex_over_human": ratio,
    })

left_leg_h = (
    np.linalg.norm(human_pts["left_knee"] - human_pts["left_hip"])
    + np.linalg.norm(human_pts["left_foot"] - human_pts["left_knee"])
)
right_leg_h = (
    np.linalg.norm(human_pts["right_knee"] - human_pts["right_hip"])
    + np.linalg.norm(human_pts["right_foot"] - human_pts["right_knee"])
)
left_leg_a = (
    np.linalg.norm(alex_pts["left_knee"] - alex_pts["left_hip"])
    + np.linalg.norm(alex_pts["left_foot"] - alex_pts["left_knee"])
)
right_leg_a = (
    np.linalg.norm(alex_pts["right_knee"] - alex_pts["right_hip"])
    + np.linalg.norm(alex_pts["right_foot"] - alex_pts["right_knee"])
)

left_arm_h = (
    np.linalg.norm(human_pts["left_elbow"] - human_pts["left_shoulder"])
    + np.linalg.norm(human_pts["left_hand"] - human_pts["left_elbow"])
)
right_arm_h = (
    np.linalg.norm(human_pts["right_elbow"] - human_pts["right_shoulder"])
    + np.linalg.norm(human_pts["right_hand"] - human_pts["right_elbow"])
)
left_arm_a = (
    np.linalg.norm(alex_pts["left_elbow"] - alex_pts["left_shoulder"])
    + np.linalg.norm(alex_pts["left_hand"] - alex_pts["left_elbow"])
)
right_arm_a = (
    np.linalg.norm(alex_pts["right_elbow"] - alex_pts["right_shoulder"])
    + np.linalg.norm(alex_pts["right_hand"] - alex_pts["right_elbow"])
)

summary = {
    "coordinate_convention": {
        "x": "forward",
        "y": "left",
        "z": "up",
        "units": "meters",
        "quat": "wxyz",
    },
    "note": "This is Step 2 rest-pose alignment diagnostic. It uses Alex MuJoCo body origins, not yet calibrated task sites or offsets.",
    "human_pelvis_world": human_pelvis.tolist(),
    "alex_pelvis_world": alex_pelvis.tolist(),
    "aggregate_lengths": {
        "human_avg_leg_length": float((left_leg_h + right_leg_h) / 2.0),
        "alex_avg_leg_length": float((left_leg_a + right_leg_a) / 2.0),
        "leg_scale_alex_over_human": float(((left_leg_a + right_leg_a) / 2.0) / ((left_leg_h + right_leg_h) / 2.0)),
        "human_avg_arm_length": float((left_arm_h + right_arm_h) / 2.0),
        "alex_avg_arm_length": float((left_arm_a + right_arm_a) / 2.0),
        "arm_scale_alex_over_human": float(((left_arm_a + right_arm_a) / 2.0) / ((left_arm_h + right_arm_h) / 2.0)),
    },
    "landmarks": landmark_rows,
    "segments": segment_rows,
}

landmark_csv = out_dir / "alex_rest_alignment_landmarks.csv"
segment_csv = out_dir / "alex_rest_alignment_segments.csv"
json_path = out_dir / "alex_rest_alignment_summary.json"

with landmark_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(landmark_rows[0].keys()))
    writer.writeheader()
    writer.writerows(landmark_rows)

with segment_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(segment_rows[0].keys()))
    writer.writeheader()
    writer.writerows(segment_rows)

json_path.write_text(json.dumps(summary, indent=2))

print()
print("Pelvis-centered landmark comparison:")
print("role                 robot_body                 human_rel              alex_rel               diff_norm")
for row in landmark_rows:
    print(
        f"{row['role']:20s} {row['robot_body']:26s} "
        f"[{row['human_rel_x']: .3f},{row['human_rel_y']: .3f},{row['human_rel_z']: .3f}] "
        f"[{row['alex_rel_x']: .3f},{row['alex_rel_y']: .3f},{row['alex_rel_z']: .3f}] "
        f"{row['diff_norm']:.3f}"
    )

print()
print("Segment length comparison:")
print("segment              from             to               human_m   alex_m    alex/human")
for row in segment_rows:
    ratio = row["alex_over_human"]
    ratio_text = f"{ratio: .3f}" if ratio is not None else "None"
    print(
        f"{row['segment']:20s} {row['from']:16s} {row['to']:16s} "
        f"{row['human_length']: .3f}   {row['alex_length']: .3f}   {ratio_text}"
    )

print()
print("Aggregate scale hints:")
for k, v in summary["aggregate_lengths"].items():
    print(f"  {k}: {v:.4f}")

print()
print("Wrote:")
print(" ", landmark_csv)
print(" ", segment_csv)
print(" ", json_path)
print()
print("Next interpretation:")
print("  Large landmark diffs suggest body-origin offsets/sites are needed.")
print("  Segment ratios are early hints for Step 3 non-uniform local scaling.")

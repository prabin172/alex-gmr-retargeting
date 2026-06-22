from pathlib import Path
import csv
import json

import mujoco
import numpy as np

from general_motion_retargeting.source_adapters.canonical_human import (
    CANONICAL_BODY_NAMES,
    make_neutral_standing_frame,
    validate_canonical_human_frame,
)
from general_motion_retargeting.retargeting.rest_pose_scaling import (
    CANONICAL_TREE_SEGMENTS,
    compute_group_scale_summary,
    compute_segment_scales,
    scale_frame_by_rest_pose,
    segment_length,
    segment_scales_to_jsonable,
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

source_rest = make_neutral_standing_frame()
validate_canonical_human_frame(source_rest)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

qpos = np.array(model.qpos0, dtype=float)
qpos[0:3] = source_rest["pelvis"]["pos"]
qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[:] = qpos
mujoco.mj_forward(model, data)

target_rest = {}

for role in CANONICAL_BODY_NAMES:
    body_name = robot_cfg["retarget_body_names"][role]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Missing robot body for role={role}: {body_name}")

    target_rest[role] = {
        "pos": [float(x) for x in data.xpos[body_id]],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }

validate_canonical_human_frame(target_rest)

scales = compute_segment_scales(source_rest, target_rest)
group_summary = compute_group_scale_summary(scales)

scaled_source = scale_frame_by_rest_pose(
    frame=source_rest,
    source_rest_frame=source_rest,
    target_rest_frame=target_rest,
)

rows = []
print()
print("Step 3 non-uniform local scaling validation")
print("This uses synthetic source_rest only as a debug source. Later each dataset/subject supplies its own rest pose.")
print()
print("segment                    source_len  target_len  scale    scaled_len  abs_error")

for parent, child in CANONICAL_TREE_SEGMENTS:
    key = f"{parent}->{child}"
    s = scales[key]
    scaled_len = segment_length(scaled_source, parent, child)
    err = abs(scaled_len - s.target_length)

    rows.append({
        "segment": key,
        "parent": parent,
        "child": child,
        "source_length": s.source_length,
        "target_length": s.target_length,
        "scale": s.scale,
        "scaled_length": scaled_len,
        "abs_error": err,
    })

    print(
        f"{key:26s} "
        f"{s.source_length:10.4f}  {s.target_length:10.4f}  {s.scale:7.4f}  "
        f"{scaled_len:10.4f}  {err:9.6f}"
    )

print()
print("Group scale summary:")
for group, info in group_summary.items():
    print(
        f"  {group:12s} source_total={info['source_total_length']:.4f} "
        f"target_total={info['target_total_length']:.4f} scale={info['scale']:.4f}"
    )

max_err = max(row["abs_error"] for row in rows)
print()
print("Max scaled segment-length error:", f"{max_err:.8f}")

if max_err > 1e-6:
    raise RuntimeError("Scaled segment lengths do not match target rest lengths closely enough.")

csv_path = out_dir / "rest_pose_segment_scales.csv"
json_path = out_dir / "rest_pose_scaling_profile.json"
scaled_frame_path = out_dir / "scaled_synthetic_human_to_alex_rest_lengths.json"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(json.dumps({
    "note": "Step 3 local scaling profile computed from source rest pose to Alex target rest pose. Source is currently synthetic debug human.",
    "source_rest": "synthetic_canonical_neutral",
    "target_rest": "alex_mujoco_neutral_body_origins",
    "segment_scales": segment_scales_to_jsonable(scales),
    "group_summary": group_summary,
}, indent=2))

scaled_frame_path.write_text(json.dumps({
    "format": "canonical_human_frame_v1",
    "note": "Synthetic neutral frame after local segment scaling to Alex rest segment lengths.",
    "frame": scaled_source,
}, indent=2))

print()
print("Wrote:")
print(" ", csv_path)
print(" ", json_path)
print(" ", scaled_frame_path)
print()
print("Validation passed.")

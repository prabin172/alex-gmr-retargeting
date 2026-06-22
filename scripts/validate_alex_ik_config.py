from pathlib import Path
import json
import math
import mujoco

repo_root = Path(__file__).resolve().parents[1]

ik_path = repo_root / "general_motion_retargeting/ik_configs/smplx_to_alex.json"
robot_cfg_path = repo_root / "general_motion_retargeting/robot_configs/alex.json"

ik_cfg = json.loads(ik_path.read_text())
robot_cfg = json.loads(robot_cfg_path.read_text())

model_path = repo_root / robot_cfg["model_path"]

print("IK config:", ik_path)
print("Robot config:", robot_cfg_path)
print("Model:", model_path)
print("Model exists:", model_path.exists())

if not model_path.exists():
    raise FileNotFoundError(
        "Missing generated floating-base model. Run both asset generation scripts first."
    )

model = mujoco.MjModel.from_xml_path(str(model_path))

body_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(model.nbody)
}

expected_human_names = set(ik_cfg["human_body_names_expected"])

def validate_entry(table_name, robot_body, entry):
    if robot_body not in body_names:
        raise ValueError(f"{table_name}: robot body does not exist in MuJoCo model: {robot_body}")

    if not isinstance(entry, list) or len(entry) != 5:
        raise ValueError(f"{table_name}/{robot_body}: entry must be [human_name, pos_w, rot_w, xyz_offset, quat_wxyz]")

    human_name, pos_w, rot_w, xyz_offset, quat = entry

    if human_name not in expected_human_names:
        raise ValueError(f"{table_name}/{robot_body}: unknown human body name: {human_name}")

    if not isinstance(pos_w, (int, float)) or pos_w < 0:
        raise ValueError(f"{table_name}/{robot_body}: invalid position weight: {pos_w}")

    if not isinstance(rot_w, (int, float)) or rot_w < 0:
        raise ValueError(f"{table_name}/{robot_body}: invalid rotation weight: {rot_w}")

    if not isinstance(xyz_offset, list) or len(xyz_offset) != 3:
        raise ValueError(f"{table_name}/{robot_body}: xyz offset must be length 3")

    if not all(isinstance(x, (int, float)) for x in xyz_offset):
        raise ValueError(f"{table_name}/{robot_body}: xyz offset must contain numbers")

    if not isinstance(quat, list) or len(quat) != 4:
        raise ValueError(f"{table_name}/{robot_body}: quaternion must be length 4 wxyz")

    if not all(isinstance(x, (int, float)) for x in quat):
        raise ValueError(f"{table_name}/{robot_body}: quaternion must contain numbers")

    norm = math.sqrt(sum(float(x) * float(x) for x in quat))
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"{table_name}/{robot_body}: quaternion not normalized: norm={norm}")

for table_name in ["ik_match_table1", "ik_match_table2"]:
    table = ik_cfg.get(table_name)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"Missing or empty {table_name}")

    print()
    print(table_name)
    for robot_body, entry in table.items():
        validate_entry(table_name, robot_body, entry)
        human_name, pos_w, rot_w, xyz_offset, quat = entry
        print(f"  {robot_body:25s} <- {human_name:15s} pos_w={pos_w:6.1f} rot_w={rot_w:5.1f}")

print()
print("Validation passed.")
print("MuJoCo bodies:", model.nbody)
print("IK table1 tasks:", len(ik_cfg["ik_match_table1"]))
print("IK table2 tasks:", len(ik_cfg["ik_match_table2"]))
print("Calibration status:", ik_cfg["calibration_status"])

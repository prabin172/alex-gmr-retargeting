from pathlib import Path
import json
import mujoco

repo_root = Path(__file__).resolve().parents[1]
config_path = repo_root / "general_motion_retargeting/robot_configs/alex.json"

cfg = json.loads(config_path.read_text())
model_path = repo_root / cfg["model_path"]

print("Config:", config_path)
print("Model:", model_path)
print("Model exists:", model_path.exists())

if not model_path.exists():
    raise FileNotFoundError(
        "Missing generated floating-base model. Run: python scripts/prepare_alex_mujoco_assets.py && python scripts/prepare_alex_floating_base_model.py"
    )

model = mujoco.MjModel.from_xml_path(str(model_path))

print()
print("Loaded MuJoCo model")
print("nbody:", model.nbody)
print("njnt:", model.njnt)
print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)

expected = cfg["expected_counts"]
floating = cfg["floating_base"]

assert model.nbody == expected["nbody"], (model.nbody, expected["nbody"])
assert model.njnt == expected["njnt"], (model.njnt, expected["njnt"])
assert model.nq == floating["expected_nq"], (model.nq, floating["expected_nq"])
assert model.nv == floating["expected_nv"], (model.nv, floating["expected_nv"])

joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    for i in range(model.njnt)
]

root_joint = joint_names[0]
assert root_joint == floating["root_joint_name"], (root_joint, floating["root_joint_name"])

actuated = joint_names[1:]
expected_joints = cfg["actuated_joint_order"]

if actuated != expected_joints:
    print()
    print("Joint order mismatch")
    for i, (a, b) in enumerate(zip(actuated, expected_joints)):
        marker = "OK" if a == b else "DIFF"
        print(f"{i:2d}: actual={a:25s} expected={b:25s} {marker}")
    raise SystemExit(1)

body_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    for i in range(model.nbody)
}

missing_bodies = []
for role, body in cfg["retarget_body_names"].items():
    if body not in body_names:
        missing_bodies.append((role, body))

if missing_bodies:
    print()
    print("Missing retarget bodies:")
    for role, body in missing_bodies:
        print(f"  {role}: {body}")
    raise SystemExit(1)

print()
print("Validation passed.")
print("Floating base:", root_joint)
print("Actuated joints:", len(actuated))
print("Retarget bodies:", len(cfg["retarget_body_names"]))

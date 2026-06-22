from pathlib import Path
import csv
import json

import mujoco
import mink
import numpy as np

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
out_dir = repo_root / "outputs/debug"
out_dir.mkdir(parents=True, exist_ok=True)

def choose_solver():
    import qpsolvers
    solvers = qpsolvers.available_solvers
    if callable(solvers):
        solvers = solvers()
    solvers = list(solvers)
    print("Available QP solvers:", solvers)
    for name in ["quadprog", "proxqp", "daqp", "osqp", "clarabel", "scs"]:
        if name in solvers:
            return name
    raise RuntimeError(f"No supported QP solver found. Available: {solvers}")

def body_pos(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=float)

def task_error_norm(tasks, configuration):
    return float(np.linalg.norm(np.concatenate([
        task.compute_error(configuration) for task in tasks
    ])))

def position_error_score(model, data, target_by_role):
    terms = []
    for role, info in target_by_role.items():
        target = np.asarray(info["target_pos"], dtype=float)
        solved = body_pos(model, data, info["robot_body"])
        weight = float(info["position_cost"])
        terms.append(np.sqrt(max(weight, 0.0)) * np.linalg.norm(solved - target))
    return float(np.linalg.norm(np.asarray(terms, dtype=float)))

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

def make_source_pose(hand_y_abs, hand_z, elbow_y_abs, elbow_z):
    frame = make_neutral_standing_frame()

    frame["left_hand"]["pos"] = [0.12, hand_y_abs, hand_z]
    frame["left_elbow"]["pos"] = [0.08, elbow_y_abs, elbow_z]

    frame["right_hand"]["pos"] = [0.12, -hand_y_abs, hand_z]
    frame["right_elbow"]["pos"] = [0.08, -elbow_y_abs, elbow_z]

    validate_canonical_human_frame(frame)
    return frame

def solve_case(model, solver, qpos0, source_rest, target_rest, case):
    source_pose = make_source_pose(
        hand_y_abs=case["hand_y_abs"],
        hand_z=case["hand_z"],
        elbow_y_abs=case["elbow_y_abs"],
        elbow_z=case["elbow_z"],
    )

    scaled_pose = scale_frame_by_rest_pose(
        frame=source_pose,
        source_rest_frame=source_rest,
        target_rest_frame=target_rest,
    )

    ik_roles = ["pelvis", "head", "left_foot", "right_foot", "left_hand", "right_hand"]

    costs = {
        "pelvis": (100.0, 0.0),
        "head": (20.0, 0.0),
        "left_foot": (80.0, 0.0),
        "right_foot": (80.0, 0.0),
        "left_hand": (case["hand_weight"], 0.0),
        "right_hand": (case["hand_weight"], 0.0),
    }

    configuration = mink.Configuration(model, q=qpos0.copy())
    limits = [mink.ConfigurationLimit(model)]

    tasks = []
    target_by_role = {}

    for role in ik_roles:
        body_name = robot_cfg["retarget_body_names"][role]
        position_cost, orientation_cost = costs[role]

        task = mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )

        target_pos = np.asarray(scaled_pose[role]["pos"], dtype=float)
        target_quat = np.asarray(scaled_pose[role]["quat_wxyz"], dtype=float)

        task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(target_quat),
                target_pos,
            )
        )

        tasks.append(task)
        target_by_role[role] = {
            "robot_body": body_name,
            "target_pos": target_pos.tolist(),
            "position_cost": position_cost,
        }

    initial_position_score = position_error_score(model, configuration.data, target_by_role)

    dt = model.opt.timestep
    damping = 1e-4
    max_iter = 300

    best_position_score = initial_position_score
    best_qpos = np.asarray(configuration.data.qpos, dtype=float).copy()

    for _ in range(max_iter):
        vel = mink.solve_ik(
            configuration=configuration,
            tasks=tasks,
            dt=dt,
            solver=solver,
            damping=damping,
            safety_break=False,
            limits=limits,
        )
        configuration.integrate_inplace(vel, dt)

        pos_score = position_error_score(model, configuration.data, target_by_role)
        if pos_score < best_position_score:
            best_position_score = pos_score
            best_qpos = np.asarray(configuration.data.qpos, dtype=float).copy()

    configuration.update(best_qpos)
    mujoco.mj_forward(model, configuration.data)

    errors = {}
    for role, info in target_by_role.items():
        target = np.asarray(info["target_pos"], dtype=float)
        solved = body_pos(model, configuration.data, info["robot_body"])
        errors[role] = float(np.linalg.norm(solved - target))

    return {
        "case": case["name"],
        "hand_y_abs": case["hand_y_abs"],
        "hand_z": case["hand_z"],
        "hand_weight": case["hand_weight"],
        "initial_position_score": initial_position_score,
        "best_position_score": best_position_score,
        "pelvis_error_m": errors["pelvis"],
        "head_error_m": errors["head"],
        "left_foot_error_m": errors["left_foot"],
        "right_foot_error_m": errors["right_foot"],
        "left_hand_error_m": errors["left_hand"],
        "right_hand_error_m": errors["right_hand"],
        "mean_hand_error_m": 0.5 * (errors["left_hand"] + errors["right_hand"]),
        "mean_foot_error_m": 0.5 * (errors["left_foot"] + errors["right_foot"]),
    }

print("Robot config:", robot_cfg_path)
print("Model:", model_path)
print("Model exists:", model_path.exists())

if not model_path.exists():
    raise FileNotFoundError(
        "Missing floating-base Alex model. Run:\n"
        "  python scripts/prepare_alex_mujoco_assets.py\n"
        "  python scripts/prepare_alex_floating_base_model.py"
    )

solver = choose_solver()
print("Using solver:", solver)

model = mujoco.MjModel.from_xml_path(str(model_path))

qpos0 = np.asarray(model.qpos0, dtype=float)
qpos0[0:3] = [0.0, 0.0, 1.0]
qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

source_rest = make_neutral_standing_frame()
target_rest = robot_rest_frame_from_mujoco(model, qpos0, robot_cfg["retarget_body_names"])

cases = [
    {
        "name": "baseline_current",
        "hand_y_abs": 0.58,
        "hand_z": 1.18,
        "elbow_y_abs": 0.42,
        "elbow_z": 1.28,
        "hand_weight": 40.0,
    },
    {
        "name": "baseline_high_hand_weight",
        "hand_y_abs": 0.58,
        "hand_z": 1.18,
        "elbow_y_abs": 0.42,
        "elbow_z": 1.28,
        "hand_weight": 120.0,
    },
    {
        "name": "easier_closer_hands",
        "hand_y_abs": 0.48,
        "hand_z": 1.08,
        "elbow_y_abs": 0.36,
        "elbow_z": 1.18,
        "hand_weight": 40.0,
    },
    {
        "name": "easier_closer_hands_high_weight",
        "hand_y_abs": 0.48,
        "hand_z": 1.08,
        "elbow_y_abs": 0.36,
        "elbow_z": 1.18,
        "hand_weight": 120.0,
    },
    {
        "name": "near_neutral_hands",
        "hand_y_abs": 0.50,
        "hand_z": 1.00,
        "elbow_y_abs": 0.40,
        "elbow_z": 1.12,
        "hand_weight": 40.0,
    },
]

rows = []
for case in cases:
    row = solve_case(model, solver, qpos0, source_rest, target_rest, case)
    rows.append(row)

print()
print("Hand target sensitivity sweep")
print("case                              hand_w  y_abs   z      pos_score   pelvis  head   feet_mean  hand_mean")
for row in rows:
    print(
        f"{row['case']:33s} "
        f"{row['hand_weight']:6.1f} "
        f"{row['hand_y_abs']:6.2f} "
        f"{row['hand_z']:6.2f} "
        f"{row['best_position_score']:10.4f} "
        f"{row['pelvis_error_m']:7.3f} "
        f"{row['head_error_m']:6.3f} "
        f"{row['mean_foot_error_m']:10.3f} "
        f"{row['mean_hand_error_m']:10.3f}"
    )

csv_path = out_dir / "static_alex_ik_hand_sensitivity.csv"
json_path = out_dir / "static_alex_ik_hand_sensitivity.json"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(json.dumps({
    "note": "Sensitivity sweep for static Alex IK hand targets. Synthetic poses only.",
    "solver": solver,
    "rows": rows,
}, indent=2))

print()
print("Wrote:")
print(" ", csv_path)
print(" ", json_path)

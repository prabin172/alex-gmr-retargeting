from pathlib import Path
import csv
import json

import mujoco
import mink
import numpy as np

from general_motion_retargeting.source_adapters.canonical_human import (
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

print("Robot config:", robot_cfg_path)
print("Model:", model_path)
print("Model exists:", model_path.exists())

if not model_path.exists():
    raise FileNotFoundError(
        "Missing floating-base Alex model. Run:\n"
        "  python scripts/prepare_alex_mujoco_assets.py\n"
        "  python scripts/prepare_alex_floating_base_model.py"
    )

def available_qp_solvers():
    try:
        import qpsolvers
        solvers = qpsolvers.available_solvers
        if callable(solvers):
            solvers = solvers()
        return list(solvers)
    except Exception:
        return []

def choose_solver():
    solvers = available_qp_solvers()
    print("Available QP solvers:", solvers)
    for name in ["quadprog", "proxqp", "osqp", "clarabel", "scs"]:
        if name in solvers:
            return name
    raise RuntimeError(
        "No supported QP solver found. Install one, for example: pip install qpsolvers[quadprog]"
    )

def robot_rest_frame_from_mujoco(model, qpos, role_to_robot):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    frame = {}
    for role, body_name in role_to_robot.items():
        if role not in [
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
        ]:
            continue

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise RuntimeError(f"Missing body {body_name} for role {role}")

        frame[role] = {
            "pos": [float(x) for x in data.xpos[body_id]],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

    validate_canonical_human_frame(frame)
    return frame

def make_source_pose():
    """
    Synthetic source pose for the first IK smoke test.

    It starts from neutral standing, then slightly raises both hands.
    This is not real data; it is only a controlled target to verify IK plumbing.
    """
    frame = make_neutral_standing_frame()

    frame["left_hand"]["pos"] = [0.12, 0.58, 1.18]
    frame["left_elbow"]["pos"] = [0.08, 0.42, 1.28]

    frame["right_hand"]["pos"] = [0.12, -0.58, 1.18]
    frame["right_elbow"]["pos"] = [0.08, -0.42, 1.28]

    validate_canonical_human_frame(frame)
    return frame

def task_error_norm(tasks, configuration):
    if not tasks:
        return 0.0
    return float(np.linalg.norm(np.concatenate([
        task.compute_error(configuration) for task in tasks
    ])))

def body_pos(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    return np.array(data.xpos[body_id], dtype=float)

def position_error_score(model, data, target_by_role):
    """Weighted position-only diagnostic score in meters * sqrt(weight)."""
    terms = []
    for role, info in target_by_role.items():
        body_name = info["robot_body"]
        target = np.array(info["target_pos"], dtype=float)
        solved = body_pos(model, data, body_name)
        weight = float(info["position_cost"])
        terms.append(np.sqrt(max(weight, 0.0)) * np.linalg.norm(solved - target))
    if not terms:
        return 0.0
    return float(np.linalg.norm(np.asarray(terms, dtype=float)))

solver = choose_solver()
print("Using solver:", solver)

model = mujoco.MjModel.from_xml_path(str(model_path))

qpos0 = np.array(model.qpos0, dtype=float)
if qpos0.shape[0] != 36:
    raise RuntimeError(f"Expected nq=36 for Alex floating base, got nq={qpos0.shape[0]}")

# Put root/pelvis near our source-human pelvis height.
qpos0[0:3] = [0.0, 0.0, 1.0]
qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

source_rest = make_neutral_standing_frame()
target_rest = robot_rest_frame_from_mujoco(model, qpos0, robot_cfg["retarget_body_names"])

source_pose = make_source_pose()
scaled_pose = scale_frame_by_rest_pose(
    frame=source_pose,
    source_rest_frame=source_rest,
    target_rest_frame=target_rest,
)

# First static IK test: use a small robust subset of tasks.
# Avoid torso for now because TORSO_LINK is not a good chest landmark.
ik_roles = [
    "pelvis",
    "head",
    "left_foot",
    "right_foot",
    "left_hand",
    "right_hand",
]

costs = {
    # First smoke test: position-only IK.
    # Rotation constraints are added after position IK behaves.
    "pelvis": (100.0, 0.0),
    "head": (20.0, 0.0),
    "left_foot": (80.0, 0.0),
    "right_foot": (80.0, 0.0),
    "left_hand": (40.0, 0.0),
    "right_hand": (40.0, 0.0),
}

configuration = mink.Configuration(model, q=qpos0)
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

    target_pos = np.array(scaled_pose[role]["pos"], dtype=float)
    target_quat = np.array(scaled_pose[role]["quat_wxyz"], dtype=float)

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
        "orientation_cost": orientation_cost,
    }

initial_error = task_error_norm(tasks, configuration)
initial_position_score = position_error_score(model, configuration.data, target_by_role)

print()
print("Initial Mink task error norm:", f"{initial_error:.6f}")
print("Initial weighted position score:", f"{initial_position_score:.6f}")

dt = model.opt.timestep
step_dt = dt
damping = 1e-4
max_iter = 300
patience = 40
min_best_improvement = 1e-6

best_task_error = initial_error
best_position_score = initial_position_score
best_qpos = np.array(configuration.data.qpos, dtype=float)
num_iter = 0
steps_since_best = 0

for i in range(max_iter):
    vel = mink.solve_ik(
        configuration=configuration,
        tasks=tasks,
        dt=dt,
        solver=solver,
        damping=damping,
        safety_break=False,
        limits=limits,
    )

    configuration.integrate_inplace(vel, step_dt)

    task_err = task_error_norm(tasks, configuration)
    pos_score = position_error_score(model, configuration.data, target_by_role)
    num_iter = i + 1

    if pos_score < best_position_score - min_best_improvement:
        best_position_score = pos_score
        best_task_error = task_err
        best_qpos = np.array(configuration.data.qpos, dtype=float)
        steps_since_best = 0
    else:
        steps_since_best += 1

    if i % 10 == 0 or i == max_iter - 1:
        print(
            f"iter {i + 1:03d}: "
            f"task_error={task_err:.6f}, "
            f"position_score={pos_score:.6f}, "
            f"best_position_score={best_position_score:.6f}"
        )

    if steps_since_best >= patience:
        print(f"Stopping after {patience} iterations without position-score improvement.")
        break

configuration.update(best_qpos)
final_error = task_error_norm(tasks, configuration)
final_position_score = position_error_score(model, configuration.data, target_by_role)

print()
print("Final Mink task error norm:", f"{final_error:.6f}")
print("Best Mink task error norm:", f"{best_task_error:.6f}")
print("Final weighted position score:", f"{final_position_score:.6f}")
print("Best weighted position score:", f"{best_position_score:.6f}")
print("Iterations:", num_iter)

solved_qpos = np.array(configuration.data.qpos, dtype=float)
mujoco.mj_forward(model, configuration.data)

rows = []
print()
print("Solved body target errors:")
print("role          robot_body                 target_xyz               solved_xyz               error_m")

for role, info in target_by_role.items():
    body_name = info["robot_body"]
    target = np.array(info["target_pos"], dtype=float)
    solved = body_pos(model, configuration.data, body_name)
    err = float(np.linalg.norm(solved - target))

    rows.append({
        "role": role,
        "robot_body": body_name,
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_z": float(target[2]),
        "solved_x": float(solved[0]),
        "solved_y": float(solved[1]),
        "solved_z": float(solved[2]),
        "position_error_m": err,
    })

    print(
        f"{role:13s} {body_name:26s} "
        f"[{target[0]: .3f},{target[1]: .3f},{target[2]: .3f}] "
        f"[{solved[0]: .3f},{solved[1]: .3f},{solved[2]: .3f}] "
        f"{err:.4f}"
    )

qpos_path = out_dir / "static_alex_ik_qpos.npy"
json_path = out_dir / "static_alex_ik_summary.json"
csv_path = out_dir / "static_alex_ik_target_errors.csv"

np.save(qpos_path, solved_qpos)

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(json.dumps({
    "note": "First Step 4 static IK smoke test. Synthetic source pose only; not real motion.",
    "solver": solver,
    "dt": dt,
    "damping": damping,
    "max_iter": max_iter,
    "iterations": num_iter,
    "initial_error_norm": initial_error,
    "final_error_norm": final_error,
    "initial_position_score": initial_position_score,
    "final_position_score": final_position_score,
    "ik_roles": ik_roles,
    "target_by_role": target_by_role,
    "qpos_layout": robot_cfg["floating_base"]["qpos_layout"],
    "qpos": solved_qpos.tolist(),
    "target_errors": rows,
}, indent=2))

print()
print("Solved qpos summary:")
print("  shape:", solved_qpos.shape)
print("  root xyz:", solved_qpos[0:3])
print("  root quat wxyz:", solved_qpos[3:7])
print("  first 10 joint values:", solved_qpos[7:17])

print()
print("Wrote:")
print(" ", qpos_path)
print(" ", csv_path)
print(" ", json_path)

print()
print("Static IK smoke test completed.")

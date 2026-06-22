from pathlib import Path
import csv
import json

import mujoco
import mink
import numpy as np
import matplotlib.pyplot as plt

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
    """
    Tiny synthetic sequence:
      neutral -> raise both hands -> lower both hands.

    This is only for IK continuity/plumbing validation.
    """
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

def make_tasks(model, ik_roles, costs):
    tasks = {}
    for role in ik_roles:
        body_name = robot_cfg["retarget_body_names"][role]
        position_cost, orientation_cost = costs[role]

        tasks[role] = mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
    return tasks

def set_task_targets(tasks, scaled_frame):
    target_by_role = {}

    for role, task in tasks.items():
        body_name = robot_cfg["retarget_body_names"][role]
        target_pos = np.asarray(scaled_frame[role]["pos"], dtype=float)
        target_quat = np.asarray(scaled_frame[role]["quat_wxyz"], dtype=float)

        task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(target_quat),
                target_pos,
            )
        )

        target_by_role[role] = {
            "robot_body": body_name,
            "target_pos": target_pos.tolist(),
        }

    return target_by_role

def position_error_score(model, data, target_by_role, costs):
    terms = []
    for role, info in target_by_role.items():
        target = np.asarray(info["target_pos"], dtype=float)
        solved = body_pos(model, data, info["robot_body"])
        weight = float(costs[role][0])
        terms.append(np.sqrt(max(weight, 0.0)) * np.linalg.norm(solved - target))
    return float(np.linalg.norm(np.asarray(terms, dtype=float)))

def solve_frame(model, configuration, tasks, target_by_role, costs, solver, limits):
    dt = model.opt.timestep
    damping = 1e-4
    max_iter = 80

    best_score = position_error_score(model, configuration.data, target_by_role, costs)
    best_qpos = np.asarray(configuration.data.qpos, dtype=float).copy()

    for _ in range(max_iter):
        vel = mink.solve_ik(
            configuration=configuration,
            tasks=list(tasks.values()),
            dt=dt,
            solver=solver,
            damping=damping,
            safety_break=False,
            limits=limits,
        )

        configuration.integrate_inplace(vel, dt)
        score = position_error_score(model, configuration.data, target_by_role, costs)

        if score < best_score:
            best_score = score
            best_qpos = np.asarray(configuration.data.qpos, dtype=float).copy()

    configuration.update(best_qpos)
    mujoco.mj_forward(model, configuration.data)

    errors = {}
    for role, info in target_by_role.items():
        target = np.asarray(info["target_pos"], dtype=float)
        solved = body_pos(model, configuration.data, info["robot_body"])
        errors[role] = float(np.linalg.norm(solved - target))

    return best_qpos.copy(), best_score, errors

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
if qpos0.shape[0] != 36:
    raise RuntimeError(f"Expected Alex nq=36, got {qpos0.shape[0]}")

qpos0[0:3] = [0.0, 0.0, 1.0]
qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

source_rest = make_neutral_standing_frame()
target_rest = robot_rest_frame_from_mujoco(model, qpos0, robot_cfg["retarget_body_names"])

num_frames = 60
source_frames = make_source_motion(num_frames)

ik_roles = ["pelvis", "head", "left_foot", "right_foot", "left_hand", "right_hand"]

costs = {
    "pelvis": (100.0, 0.0),
    "head": (20.0, 0.0),
    "left_foot": (80.0, 0.0),
    "right_foot": (80.0, 0.0),
    "left_hand": (40.0, 0.0),
    "right_hand": (40.0, 0.0),
}

configuration = mink.Configuration(model, q=qpos0.copy())
limits = [mink.ConfigurationLimit(model)]
tasks = make_tasks(model, ik_roles, costs)

qpos_traj = []
rows = []

for frame_idx, source_frame in enumerate(source_frames):
    scaled_frame = scale_frame_by_rest_pose(
        frame=source_frame,
        source_rest_frame=source_rest,
        target_rest_frame=target_rest,
    )

    target_by_role = set_task_targets(tasks, scaled_frame)
    qpos, score, errors = solve_frame(
        model=model,
        configuration=configuration,
        tasks=tasks,
        target_by_role=target_by_role,
        costs=costs,
        solver=solver,
        limits=limits,
    )

    qpos_traj.append(qpos)

    row = {
        "frame": frame_idx,
        "position_score": score,
        "pelvis_error_m": errors["pelvis"],
        "head_error_m": errors["head"],
        "left_foot_error_m": errors["left_foot"],
        "right_foot_error_m": errors["right_foot"],
        "left_hand_error_m": errors["left_hand"],
        "right_hand_error_m": errors["right_hand"],
        "mean_foot_error_m": 0.5 * (errors["left_foot"] + errors["right_foot"]),
        "mean_hand_error_m": 0.5 * (errors["left_hand"] + errors["right_hand"]),
        "root_x": float(qpos[0]),
        "root_y": float(qpos[1]),
        "root_z": float(qpos[2]),
    }
    rows.append(row)

    if frame_idx % 10 == 0 or frame_idx == num_frames - 1:
        print(
            f"frame {frame_idx:03d}: "
            f"score={score:.4f}, "
            f"pelvis={row['pelvis_error_m']:.3f}, "
            f"feet={row['mean_foot_error_m']:.3f}, "
            f"hands={row['mean_hand_error_m']:.3f}"
        )

qpos_traj = np.asarray(qpos_traj, dtype=float)

joint_delta = np.diff(qpos_traj[:, 7:], axis=0)
root_delta = np.diff(qpos_traj[:, 0:3], axis=0)

summary = {
    "note": "Tiny synthetic motion IK test: neutral -> raise hands -> lower hands. Position-only IK.",
    "solver": solver,
    "num_frames": num_frames,
    "qpos_shape": list(qpos_traj.shape),
    "qpos_layout": robot_cfg["floating_base"]["qpos_layout"],
    "mean_position_score": float(np.mean([r["position_score"] for r in rows])),
    "max_position_score": float(np.max([r["position_score"] for r in rows])),
    "mean_hand_error_m": float(np.mean([r["mean_hand_error_m"] for r in rows])),
    "max_hand_error_m": float(np.max([r["mean_hand_error_m"] for r in rows])),
    "mean_foot_error_m": float(np.mean([r["mean_foot_error_m"] for r in rows])),
    "max_foot_error_m": float(np.max([r["mean_foot_error_m"] for r in rows])),
    "mean_pelvis_error_m": float(np.mean([r["pelvis_error_m"] for r in rows])),
    "max_pelvis_error_m": float(np.max([r["pelvis_error_m"] for r in rows])),
    "max_abs_joint_step_rad": float(np.max(np.abs(joint_delta))) if len(joint_delta) else 0.0,
    "max_root_step_m": float(np.max(np.linalg.norm(root_delta, axis=1))) if len(root_delta) else 0.0,
}

npz_path = out_dir / "synthetic_alex_motion_ik_qpos.npz"
csv_path = out_dir / "synthetic_alex_motion_ik_errors.csv"
json_path = out_dir / "synthetic_alex_motion_ik_summary.json"
plot_path = out_dir / "synthetic_alex_motion_ik_errors.png"

np.savez(
    npz_path,
    qpos=qpos_traj,
    fps=np.array([30.0], dtype=float),
    joint_names=np.asarray(robot_cfg["actuated_joint_order"], dtype=object),
)

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

json_path.write_text(json.dumps(summary, indent=2))

frames = np.asarray([r["frame"] for r in rows])
plt.figure(figsize=(10, 6))
plt.plot(frames, [r["pelvis_error_m"] for r in rows], label="pelvis")
plt.plot(frames, [r["head_error_m"] for r in rows], label="head")
plt.plot(frames, [r["mean_foot_error_m"] for r in rows], label="feet mean")
plt.plot(frames, [r["mean_hand_error_m"] for r in rows], label="hands mean")
plt.xlabel("frame")
plt.ylabel("position error (m)")
plt.title("Synthetic Alex motion IK target errors")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=200)
plt.close()

print()
print("Summary:")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

print()
print("Wrote:")
print(" ", npz_path)
print(" ", csv_path)
print(" ", json_path)
print(" ", plot_path)

print()
print("Synthetic motion IK test completed.")

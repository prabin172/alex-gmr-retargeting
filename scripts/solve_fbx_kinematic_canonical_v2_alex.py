#!/usr/bin/env python3
"""Pure kinematic Alex IK from FBX canonical v2.

This is intentionally separate from the contact-heavy canonical solver.
It does not use contact masks, sole grounding, support/friction constraints,
physics feasibility, or the balanced candidate selector.  It simply loads the
source-only canonical v2 skeleton and solves warm-started per-frame Alex IK.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mink
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_IK_ROLES = [
    "pelvis",
    "torso",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_palm",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_palm",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_heel",
    "left_toe",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_heel",
    "right_toe",
    "right_foot",
]


KINEMATIC_TREE_EDGES = [
    ("pelvis", "torso"),
    ("torso", "neck"),
    ("neck", "head"),
    ("head", "head_top"),
    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_palm"),
    ("left_palm", "left_hand_tip"),
    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_palm"),
    ("right_palm", "right_hand_tip"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_toe"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_toe"),
]


# Position costs are intentionally simple.  They are not contact weights.
DEFAULT_POSITION_COSTS = {
    "pelvis": 100.0,
    "torso": 45.0,
    "neck": 15.0,
    "head": 35.0,
    "head_top": 10.0,
    "left_shoulder": 10.0,
    "right_shoulder": 10.0,
    "left_elbow": 25.0,
    "right_elbow": 25.0,
    "left_wrist": 20.0,
    "right_wrist": 20.0,
    "left_palm": 60.0,
    "right_palm": 60.0,
    "left_hip": 10.0,
    "right_hip": 10.0,
    "left_knee": 25.0,
    "right_knee": 25.0,
    "left_ankle": 30.0,
    "right_ankle": 30.0,
    "left_heel": 20.0,
    "right_heel": 20.0,
    "left_toe": 20.0,
    "right_toe": 20.0,
    "left_foot": 70.0,
    "right_foot": 70.0,
}


DEFAULT_ORIENTATION_COSTS = {
    "pelvis": 0.5,
    "head": 0.25,
    "left_palm": 0.5,
    "right_palm": 0.5,
    "left_foot": 2.0,
    "right_foot": 2.0,
}


def resolve_repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(resolve_repo_path(path).read_text())


def merge_robot_config(robot_cfg_path: Path) -> Dict[str, Any]:
    """Merge alex_retarget_sites.json-style partial site config over alex.json."""
    robot_cfg_path = resolve_repo_path(robot_cfg_path)
    cfg = json.loads(robot_cfg_path.read_text())
    base_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex.json"
    base = json.loads(base_path.read_text())

    merged = dict(base)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            tmp = dict(merged[key])
            tmp.update(value)
            merged[key] = tmp
        else:
            merged[key] = value

    sites = cfg.get("sites", [])
    site_names = {s.get("name") for s in sites if isinstance(s, dict)}
    role_to_site = dict(merged.get("retarget_site_names", {}))
    candidates = {
        "pelvis": "alex_pelvis_site",
        "head": "alex_head_site",
        "left_palm": "alex_left_palm_contact_site",
        "right_palm": "alex_right_palm_contact_site",
        "left_foot": "alex_left_sole_contact_site",
        "right_foot": "alex_right_sole_contact_site",
        # Heel/toe are approximate point targets because the current Alex XML
        # exposes sole-corner sites, not center heel/toe sites.
        "left_heel": "alex_left_sole_corner_heel_body_left_site",
        "left_toe": "alex_left_sole_corner_toe_body_left_site",
        "right_heel": "alex_right_sole_corner_heel_body_left_site",
        "right_toe": "alex_right_sole_corner_toe_body_left_site",
    }
    for role, site_name in candidates.items():
        if site_name in site_names:
            role_to_site[role] = site_name
    if role_to_site:
        merged["retarget_site_names"] = role_to_site

    body_names = dict(merged.get("retarget_body_names", {}))
    body_names.setdefault("neck", "NECK_Z_LINK")
    body_names.setdefault("left_ankle", "LEFT_FOOT")
    body_names.setdefault("right_ankle", "RIGHT_FOOT")
    body_names.setdefault("left_wrist", "LEFT_WRIST_X_LINK")
    body_names.setdefault("right_wrist", "RIGHT_WRIST_X_LINK")
    merged["retarget_body_names"] = body_names

    if "model_path" not in merged or "site" in robot_cfg_path.name.lower():
        merged["model_path"] = "assets/alex/alex_floating_base_with_sites.xml"
    return merged


def resolve_model_path(robot_cfg: Mapping[str, Any], override: Optional[Path] = None) -> Path:
    if override is not None:
        return resolve_repo_path(override)
    return resolve_repo_path(Path(robot_cfg["model_path"]))


def choose_solver(preferred: str) -> str:
    import qpsolvers

    solvers = qpsolvers.available_solvers
    if callable(solvers):
        solvers = solvers()
    solvers = list(solvers)
    print("Available QP solvers:", solvers)
    if preferred != "auto":
        if preferred not in solvers:
            raise RuntimeError(f"Requested solver {preferred!r} not available. Available: {solvers}")
        return preferred
    for name in ["daqp", "proxqp", "quadprog", "osqp", "clarabel", "scs"]:
        if name in solvers:
            return name
    raise RuntimeError(f"No supported QP solver found. Available: {solvers}")


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=float,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(max(tr + 1.0, 1e-12)) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def frame_name_and_type(robot_cfg: Mapping[str, Any], role: str) -> Optional[Tuple[str, str]]:
    site_names = robot_cfg.get("retarget_site_names", {})
    body_names = robot_cfg.get("retarget_body_names", {})
    if role in site_names:
        return str(site_names[role]), "site"
    if role in body_names:
        return str(body_names[role]), "body"
    return None


def frame_exists(model: mujoco.MjModel, frame_name: str, frame_type: str) -> bool:
    obj = mujoco.mjtObj.mjOBJ_SITE if frame_type == "site" else mujoco.mjtObj.mjOBJ_BODY
    return mujoco.mj_name2id(model, obj, frame_name) >= 0


def frame_pos(model: mujoco.MjModel, data: mujoco.MjData, frame_name: str, frame_type: str) -> np.ndarray:
    if frame_type == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
        return np.asarray(data.site_xpos[sid], dtype=float).copy()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    return np.asarray(data.xpos[bid], dtype=float).copy()


def frame_rot(model: mujoco.MjModel, data: mujoco.MjData, frame_name: str, frame_type: str) -> np.ndarray:
    if frame_type == "site":
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
        return np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3).copy()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
    return np.asarray(data.xmat[bid], dtype=float).reshape(3, 3).copy()


def load_canonical_v2(path: Path, start_frame: int, stride: int, max_frames: Optional[int]) -> Dict[str, Any]:
    path = resolve_repo_path(path)
    d = np.load(path, allow_pickle=True)
    roles = [str(x) for x in d["roles"].tolist()]
    positions = np.asarray(d["positions"], dtype=float)
    orientations = np.asarray(d["orientations"], dtype=float)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    frame_ids = np.arange(positions.shape[0], dtype=int)[int(start_frame) :: int(stride)]
    if max_frames is not None:
        frame_ids = frame_ids[: int(max_frames)]
    metadata = json.loads(str(d["metadata_json"].item())) if "metadata_json" in d.files else {}
    return {
        "path": str(path),
        "roles": roles,
        "positions": positions[frame_ids],
        "orientations": orientations[frame_ids],
        "fps": fps,
        "output_fps": fps / float(stride),
        "source_frame_ids": frame_ids,
        "metadata": metadata,
    }


def make_tasks(model: mujoco.MjModel, robot_cfg: Mapping[str, Any], roles: Sequence[str], orientation_costs: Mapping[str, float]) -> Dict[str, mink.FrameTask]:
    tasks: Dict[str, mink.FrameTask] = {}
    print("\nIK task frames:")
    for role in roles:
        info = frame_name_and_type(robot_cfg, role)
        if info is None:
            print(f"  {role:16s}: skipped, no robot frame mapping")
            continue
        frame_name, frame_type = info
        if not frame_exists(model, frame_name, frame_type):
            print(f"  {role:16s}: skipped, missing {frame_type} {frame_name}")
            continue
        pos_cost = float(DEFAULT_POSITION_COSTS.get(role, 10.0))
        ori_cost = float(orientation_costs.get(role, 0.0))
        task = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=pos_cost,
            orientation_cost=ori_cost,
        )
        if ori_cost > 0.0 and hasattr(task, "set_orientation_cost"):
            task.set_orientation_cost(ori_cost)
        tasks[role] = task
        print(f"  {role:16s} -> {frame_type:4s} {frame_name:38s} pos={pos_cost:7.2f} ori={ori_cost:6.2f}")
    return tasks


def set_posture_target(posture_task: Optional[mink.PostureTask], model: mujoco.MjModel, qpos_target: np.ndarray) -> None:
    if posture_task is None:
        return
    if hasattr(posture_task, "set_target"):
        posture_task.set_target(qpos_target)
    elif hasattr(posture_task, "set_target_from_configuration"):
        posture_task.set_target_from_configuration(mink.Configuration(model, q=qpos_target.copy()))
    else:
        raise RuntimeError("Could not set Mink PostureTask target")


def make_output_joint_caps(model: mujoco.MjModel, cap: Optional[float]) -> Optional[np.ndarray]:
    if cap is None or cap <= 0.0:
        return None
    caps = np.full(model.nq, np.inf, dtype=float)
    caps[7:] = float(cap)
    return caps


def clamp_actuated_step(qpos: np.ndarray, q_prev: np.ndarray, max_joint_step_rad: Optional[float]) -> np.ndarray:
    if max_joint_step_rad is None or max_joint_step_rad <= 0.0:
        return qpos
    out = qpos.copy()
    step = np.clip(out[7:] - q_prev[7:], -float(max_joint_step_rad), float(max_joint_step_rad))
    out[7:] = q_prev[7:] + step
    return out


def robot_rest_positions(model: mujoco.MjModel, qpos: np.ndarray, tasks: Mapping[str, mink.FrameTask]) -> Dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return {role: frame_pos(model, data, task.frame_name, task.frame_type) for role, task in tasks.items()}


def robot_rest_rotations(model: mujoco.MjModel, qpos: np.ndarray, tasks: Mapping[str, mink.FrameTask]) -> Dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return {role: frame_rot(model, data, task.frame_name, task.frame_type) for role, task in tasks.items()}


def compute_morphology_segments(
    source_rest_pos: Mapping[str, np.ndarray],
    robot_rest_pos: Mapping[str, np.ndarray],
    roles: Sequence[str],
) -> Tuple[List[Tuple[str, str]], Dict[str, Dict[str, float]]]:
    role_set = set(roles)
    segments = [(parent, child) for parent, child in KINEMATIC_TREE_EDGES if parent in role_set and child in role_set]
    info: Dict[str, Dict[str, float]] = {}
    for parent, child in segments:
        human_len = float(np.linalg.norm(source_rest_pos[child] - source_rest_pos[parent]))
        robot_len = float(np.linalg.norm(robot_rest_pos[child] - robot_rest_pos[parent]))
        scale = 1.0 if human_len < 1e-8 else robot_len / human_len
        info[f"{parent}->{child}"] = {
            "human_rest_length_m": human_len,
            "robot_rest_length_m": robot_len,
            "scale": scale,
        }
    return segments, info


def print_morphology_summary(segment_info: Mapping[str, Mapping[str, float]]) -> None:
    print("\nMorphology-aware segment scales:")
    for name, item in segment_info.items():
        print(
            f"  {name:28s} "
            f"human={item['human_rest_length_m']:.4f} m  "
            f"robot={item['robot_rest_length_m']:.4f} m  "
            f"scale={item['scale']:.4f}"
        )


def make_morphology_target_positions(
    src_pos: np.ndarray,
    role_to_idx: Mapping[str, int],
    source_rest_pos: Mapping[str, np.ndarray],
    robot_rest_pos: Mapping[str, np.ndarray],
    roles: Sequence[str],
    segments: Sequence[Tuple[str, str]],
    segment_info: Mapping[str, Mapping[str, float]],
    motion_scale: float,
) -> Dict[str, np.ndarray]:
    target_pos: Dict[str, np.ndarray] = {}
    if "pelvis" in roles:
        pelvis_idx = role_to_idx["pelvis"]
        target_pos["pelvis"] = robot_rest_pos["pelvis"] + float(motion_scale) * (
            src_pos[pelvis_idx] - source_rest_pos["pelvis"]
        )

    pending = list(segments)
    while pending:
        progressed = False
        next_pending = []
        for parent, child in pending:
            if parent not in target_pos:
                next_pending.append((parent, child))
                continue
            idx_parent = role_to_idx[parent]
            idx_child = role_to_idx[child]
            source_vec = src_pos[idx_child] - src_pos[idx_parent]
            scale = float(segment_info[f"{parent}->{child}"]["scale"])
            target_pos[child] = target_pos[parent] + float(motion_scale) * scale * source_vec
            progressed = True
        if not progressed:
            break
        pending = next_pending

    # Canonical foot center is a semantic midpoint. Keep it tied to scaled
    # heel/toe targets instead of a raw FBX LeftFoot ankle point.
    for side in ("left", "right"):
        foot = f"{side}_foot"
        heel = f"{side}_heel"
        toe = f"{side}_toe"
        if foot in roles and heel in target_pos and toe in target_pos:
            target_pos[foot] = 0.5 * (target_pos[heel] + target_pos[toe])

    # Fallback for any role not connected in the tree or unavailable as a child:
    # rest-delta with that role's own Alex rest point. This keeps optional roles
    # usable without silently dropping them.
    for role in roles:
        if role in target_pos:
            continue
        idx = role_to_idx[role]
        target_pos[role] = robot_rest_pos[role] + float(motion_scale) * (src_pos[idx] - source_rest_pos[role])

    return target_pos


def make_target_frame(
    src_pos: np.ndarray,
    src_ori: np.ndarray,
    role_to_idx: Mapping[str, int],
    source_rest_pos: Mapping[str, np.ndarray],
    source_rest_ori: Mapping[str, np.ndarray],
    robot_rest_pos: Mapping[str, np.ndarray],
    robot_rest_ori: Mapping[str, np.ndarray],
    roles: Sequence[str],
    morphology_segments: Sequence[Tuple[str, str]],
    morphology_segment_info: Mapping[str, Mapping[str, float]],
    orientation_roles: Sequence[str],
    orientation_transfer: str,
    motion_scale: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    target_pos = make_morphology_target_positions(
        src_pos=src_pos,
        role_to_idx=role_to_idx,
        source_rest_pos=source_rest_pos,
        robot_rest_pos=robot_rest_pos,
        roles=roles,
        segments=morphology_segments,
        segment_info=morphology_segment_info,
        motion_scale=motion_scale,
    )
    target_rot: Dict[str, np.ndarray] = {}
    orientation_set = set(orientation_roles)

    for role in roles:
        idx = role_to_idx[role]
        if role in orientation_set:
            R_src = np.asarray(src_ori[idx], dtype=float)
            if orientation_transfer == "source_absolute":
                R_target = R_src
            elif orientation_transfer == "world_delta":
                R_target = R_src @ source_rest_ori[role].T @ robot_rest_ori[role]
            else:
                raise RuntimeError(f"Unknown orientation_transfer: {orientation_transfer}")
            target_rot[role] = R_target

    return target_pos, target_rot


def set_task_targets(
    tasks: Mapping[str, mink.FrameTask],
    target_pos: Mapping[str, np.ndarray],
    target_rot: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    target_by_role: Dict[str, Dict[str, Any]] = {}
    for role, task in tasks.items():
        pos = np.asarray(target_pos[role], dtype=float)
        R = target_rot.get(role, np.eye(3))
        quat = rotmat_to_quat_wxyz(R)
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos))
        target_by_role[role] = {
            "frame_name": task.frame_name,
            "frame_type": task.frame_type,
            "target_pos": pos.copy(),
            "target_quat_wxyz": quat.copy(),
        }
    return target_by_role


def compute_errors(model: mujoco.MjModel, data: mujoco.MjData, target_by_role: Mapping[str, Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:
    pos_errors: Dict[str, float] = {}
    ori_errors: Dict[str, float] = {}
    solved: Dict[str, np.ndarray] = {}
    for role, info in target_by_role.items():
        p = frame_pos(model, data, info["frame_name"], info["frame_type"])
        solved[role] = p
        pos_errors[role] = float(np.linalg.norm(p - info["target_pos"]))
        R_solved = frame_rot(model, data, info["frame_name"], info["frame_type"])
        R_target = quat_wxyz_to_rotmat(info["target_quat_wxyz"])
        cos_angle = np.clip((np.trace(R_target.T @ R_solved) - 1.0) * 0.5, -1.0, 1.0)
        ori_errors[role] = float(np.arccos(cos_angle))
    return pos_errors, solved, ori_errors


def solve_one_frame(
    model: mujoco.MjModel,
    configuration: mink.Configuration,
    tasks: Mapping[str, mink.FrameTask],
    limits: Sequence[Any],
    solver: str,
    max_iter: int,
    posture_task: Optional[mink.PostureTask],
) -> None:
    dt = model.opt.timestep
    active_tasks = list(tasks.values())
    if posture_task is not None:
        active_tasks.append(posture_task)
    for _ in range(int(max_iter)):
        vel = mink.solve_ik(configuration, active_tasks, dt, solver=solver, damping=1e-4, limits=limits)
        configuration.integrate_inplace(vel, dt)


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, default=Path("general_motion_retargeting/robot_configs/alex_retarget_sites.json"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--max-ik-iter", type=int, default=30)
    parser.add_argument("--solver", choices=["auto", "daqp", "proxqp", "quadprog", "osqp", "clarabel", "scs"], default="auto")
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--orientation-transfer", choices=["source_absolute", "world_delta"], default="source_absolute")
    parser.add_argument("--orientation-roles", default="left_palm,right_palm,left_foot,right_foot,pelvis,head")
    parser.add_argument("--ik-roles", default=",".join(DEFAULT_IK_ROLES))
    parser.add_argument("--prev-reg", "--joint-step-penalty", dest="prev_reg", type=float, default=2.0)
    parser.add_argument("--posture-neutral-blend", type=float, default=0.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.20)
    parser.add_argument("--no-rest-align", action="store_true", help="Skip static first-frame rest alignment.")
    args = parser.parse_args()

    canonical = load_canonical_v2(args.canonical_v2, args.start_frame, args.stride, args.max_frames)
    roles = canonical["roles"]
    role_to_idx = {role: i for i, role in enumerate(roles)}
    requested_roles = [role for role in parse_csv_list(args.ik_roles) if role in role_to_idx]
    orientation_roles = [role for role in parse_csv_list(args.orientation_roles) if role in requested_roles]

    robot_cfg = merge_robot_config(args.robot_config)
    model_path = resolve_model_path(robot_cfg, args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    solver = choose_solver(args.solver)
    print("Using solver:", solver)
    print("Canonical v2:", canonical["path"])
    print("Model:", model_path)

    orientation_costs = {role: DEFAULT_ORIENTATION_COSTS.get(role, 0.0) for role in orientation_roles}
    tasks = make_tasks(model, robot_cfg, requested_roles, orientation_costs)
    ik_roles = list(tasks.keys())
    if not ik_roles:
        raise RuntimeError("No IK tasks could be created from canonical roles and robot config.")

    qpos0 = np.asarray(model.qpos0, dtype=float).copy()
    first_pelvis = canonical["positions"][0, role_to_idx["pelvis"]]
    qpos0[0:3] = first_pelvis
    qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

    limits = [mink.ConfigurationLimit(model)]
    posture_task = None
    if args.prev_reg > 0.0:
        posture_task = mink.PostureTask(model, cost=float(args.prev_reg))

    source_rest_pos = {role: canonical["positions"][0, role_to_idx[role]].copy() for role in ik_roles}
    source_rest_ori = {role: canonical["orientations"][0, role_to_idx[role]].copy() for role in ik_roles}

    configuration = mink.Configuration(model, q=qpos0.copy())
    if not args.no_rest_align:
        print("\nRest-aligning Alex to canonical v2 first frame (position only).")
        zero_ori_costs = {role: 0.0 for role in ik_roles}
        rest_tasks = make_tasks(model, robot_cfg, ik_roles, zero_ori_costs)
        rest_pos = {role: source_rest_pos[role] for role in rest_tasks.keys()}
        rest_rot = {}
        set_task_targets(rest_tasks, rest_pos, rest_rot)
        set_posture_target(posture_task, model, qpos0.copy())
        rest_configuration = mink.Configuration(model, q=qpos0.copy())
        solve_one_frame(model, rest_configuration, rest_tasks, limits, solver, max(args.max_ik_iter, 120), posture_task)
        configuration = mink.Configuration(model, q=np.asarray(rest_configuration.data.qpos, dtype=float).copy())
        print("Rest alignment complete.")

    q_rest = np.asarray(configuration.data.qpos, dtype=float).copy()
    robot_rest_pos = robot_rest_positions(model, q_rest, tasks)
    robot_rest_ori = robot_rest_rotations(model, q_rest, tasks)
    morphology_segments, morphology_segment_info = compute_morphology_segments(
        source_rest_pos=source_rest_pos,
        robot_rest_pos=robot_rest_pos,
        roles=ik_roles,
    )
    print_morphology_summary(morphology_segment_info)
    q_neutral = qpos0.copy()
    q_prev = q_rest.copy()

    qpos_traj: List[np.ndarray] = []
    raw_human_target_positions: List[List[np.ndarray]] = []
    target_positions: List[List[np.ndarray]] = []
    solved_positions: List[List[np.ndarray]] = []
    target_orientations: List[List[np.ndarray]] = []
    rows: List[Dict[str, Any]] = []

    for out_i in range(canonical["positions"].shape[0]):
        src_pos = canonical["positions"][out_i]
        src_ori = canonical["orientations"][out_i]
        target_pos, target_rot = make_target_frame(
            src_pos=src_pos,
            src_ori=src_ori,
            role_to_idx=role_to_idx,
            source_rest_pos=source_rest_pos,
            source_rest_ori=source_rest_ori,
            robot_rest_pos=robot_rest_pos,
            robot_rest_ori=robot_rest_ori,
            roles=ik_roles,
            morphology_segments=morphology_segments,
            morphology_segment_info=morphology_segment_info,
            orientation_roles=orientation_roles,
            orientation_transfer=args.orientation_transfer,
            motion_scale=args.motion_scale,
        )
        target_by_role = set_task_targets(tasks, target_pos, target_rot)

        if posture_task is not None:
            q_posture = q_prev.copy()
            if args.posture_neutral_blend > 0.0:
                blend = float(np.clip(args.posture_neutral_blend, 0.0, 1.0))
                q_posture[7:] = (1.0 - blend) * q_posture[7:] + blend * q_neutral[7:]
            set_posture_target(posture_task, model, q_posture)

        solve_one_frame(model, configuration, tasks, limits, solver, args.max_ik_iter, posture_task)
        q_candidate = np.asarray(configuration.data.qpos, dtype=float).copy()
        q_candidate = clamp_actuated_step(q_candidate, q_prev, args.max_joint_step_rad)
        configuration.update(q_candidate)
        mujoco.mj_forward(model, configuration.data)

        pos_errors, solved_by_role, ori_errors = compute_errors(model, configuration.data, target_by_role)
        q_prev = q_candidate.copy()
        qpos_traj.append(q_candidate)

        raw_human_target_positions.append([src_pos[role_to_idx[role]].copy() for role in ik_roles])
        target_positions.append([target_by_role[role]["target_pos"] for role in ik_roles])
        target_orientations.append([target_by_role[role]["target_quat_wxyz"] for role in ik_roles])
        solved_positions.append([solved_by_role[role] for role in ik_roles])

        row = {
            "frame": out_i,
            "source_frame_id": int(canonical["source_frame_ids"][out_i]),
            "mean_position_error_m": float(np.mean([pos_errors[r] for r in ik_roles])),
            "max_position_error_m": float(np.max([pos_errors[r] for r in ik_roles])),
            "mean_orientation_error_deg": float(np.degrees(np.mean([ori_errors[r] for r in orientation_roles if r in ori_errors]))) if orientation_roles else 0.0,
            "max_orientation_error_deg": float(np.degrees(np.max([ori_errors[r] for r in orientation_roles if r in ori_errors]))) if orientation_roles else 0.0,
            "root_x": float(q_candidate[0]),
            "root_y": float(q_candidate[1]),
            "root_z": float(q_candidate[2]),
        }
        for role in ik_roles:
            row[f"{role}_error_m"] = pos_errors[role]
        rows.append(row)

        if out_i % 10 == 0 or out_i == canonical["positions"].shape[0] - 1:
            print(
                f"frame {out_i:04d}: mean_err={row['mean_position_error_m']:.4f} "
                f"max_err={row['max_position_error_m']:.4f} "
                f"ori_mean_deg={row['mean_orientation_error_deg']:.1f}"
            )

    qpos_arr = np.asarray(qpos_traj, dtype=float)
    raw_human_target_positions_arr = np.asarray(raw_human_target_positions, dtype=float)
    target_positions_arr = np.asarray(target_positions, dtype=float)
    solved_positions_arr = np.asarray(solved_positions, dtype=float)
    target_orientations_arr = np.asarray(target_orientations, dtype=float)

    out_path = resolve_repo_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(qpos_arr) > 1:
        joint_delta = np.abs(np.diff(qpos_arr[:, 7:], axis=0))
        root_delta = np.linalg.norm(np.diff(qpos_arr[:, 0:3], axis=0), axis=1)
        max_joint_step = float(joint_delta.max())
        max_root_step = float(root_delta.max())
    else:
        max_joint_step = 0.0
        max_root_step = 0.0

    np.savez(
        out_path,
        qpos=qpos_arr,
        ik_roles=np.asarray(ik_roles, dtype=object),
        source_roles=np.asarray(roles, dtype=object),
        source_frame_ids=np.asarray(canonical["source_frame_ids"], dtype=int),
        source_positions=canonical["positions"],
        source_orientations=canonical["orientations"],
        raw_human_target_positions=raw_human_target_positions_arr,
        morphology_scaled_target_positions=target_positions_arr,
        target_positions=target_positions_arr,
        target_orientations_wxyz=target_orientations_arr,
        solved_ik_positions=solved_positions_arr,
        qpos_rest=q_rest,
        robot_model_path=np.asarray(str(model_path), dtype=object),
        morphology_segments=np.asarray(morphology_segments, dtype=object),
        morphology_segment_info_json=np.asarray(json.dumps(morphology_segment_info, indent=2), dtype=object),
    )

    csv_path = out_path.with_name(out_path.stem + "_errors.csv")
    with csv_path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "canonical_v2": canonical["path"],
        "output": str(out_path),
        "robot_config": str(resolve_repo_path(args.robot_config)),
        "model_path": str(model_path),
        "num_frames": int(len(qpos_arr)),
        "source_fps": float(canonical["fps"]),
        "output_fps": float(canonical["output_fps"]),
        "start_frame": int(args.start_frame),
        "stride": int(args.stride),
        "max_frames": args.max_frames,
        "solver": solver,
        "max_ik_iter": int(args.max_ik_iter),
        "prev_reg": float(args.prev_reg),
        "posture_neutral_blend": float(args.posture_neutral_blend),
        "max_joint_step_rad": float(args.max_joint_step_rad) if args.max_joint_step_rad else None,
        "max_abs_joint_step_rad": max_joint_step,
        "max_root_step_m": max_root_step,
        "ik_roles": ik_roles,
        "orientation_roles": orientation_roles,
        "orientation_transfer": args.orientation_transfer,
        "target_position_generation": "morphology_aware_segment_scaled",
        "morphology_segments": morphology_segment_info,
        "mean_position_error_m": float(np.mean([r["mean_position_error_m"] for r in rows])) if rows else None,
        "max_position_error_m": float(np.max([r["max_position_error_m"] for r in rows])) if rows else None,
        "mean_orientation_error_deg": float(np.mean([r["mean_orientation_error_deg"] for r in rows])) if rows else None,
        "max_orientation_error_deg": float(np.max([r["max_orientation_error_deg"] for r in rows])) if rows else None,
        "notes": [
            "Pure kinematic canonical-v2 Alex IK.",
            "No contact masks, sole grounding, support/friction constraints, physics, or balanced candidate selector.",
            "Heel/toe robot tasks use available sole-corner sites as approximate point frames.",
        ],
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    # Quick diagnostic plot.
    fig_path = out_path.with_name(out_path.stem + "_errors.png")
    if rows:
        plt.figure(figsize=(9, 4))
        plt.plot([r["mean_position_error_m"] for r in rows], label="mean pos err")
        plt.plot([r["max_position_error_m"] for r in rows], label="max pos err")
        plt.xlabel("output frame")
        plt.ylabel("m")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=140)
        plt.close()

    print("\nWrote:")
    print(" ", out_path)
    print(" ", csv_path)
    print(" ", summary_path)
    print(" ", fig_path)
    print("\nSummary:")
    for key in ["num_frames", "mean_position_error_m", "max_position_error_m", "max_abs_joint_step_rad", "max_root_step_m"]:
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()

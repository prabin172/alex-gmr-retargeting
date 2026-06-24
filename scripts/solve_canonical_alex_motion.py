#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MVNX_SOLVER_PATH = Path(__file__).resolve().parent / "solve_mvnx_alex_motion.py"

spec = importlib.util.spec_from_file_location("mvnx_solver_module", MVNX_SOLVER_PATH)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

CANONICAL_BODY_NAMES = list(S.CANONICAL_BODY_NAMES)
IK_ROLES = list(S.IK_ROLES)





def merge_robot_config(robot_cfg_path):
    """Load robot config; if partial site config, merge over alex.json and map roles to site names."""
    robot_cfg_path = Path(robot_cfg_path)
    cfg = json.loads(robot_cfg_path.read_text())

    base_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex.json"
    if robot_cfg_path.resolve() == base_path.resolve():
        return cfg

    base = json.loads(base_path.read_text())

    merged = dict(base)
    for key, value in cfg.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            tmp = dict(merged[key])
            tmp.update(value)
            merged[key] = tmp
        else:
            merged[key] = value

    # alex_retarget_sites.json stores a list of site definitions.
    # The MVNX solver expects retarget_site_names = {role: site_name}.
    sites = cfg.get("sites", [])
    site_names = {s.get("name") for s in sites if isinstance(s, dict)}

    role_to_site = {}
    candidates = {
        "pelvis": "alex_pelvis_site",
        "head": "alex_head_site",
        "left_foot": "alex_left_sole_site",
        "right_foot": "alex_right_sole_site",
        "left_hand": "alex_left_palm_site",
        "right_hand": "alex_right_palm_site",
    }

    for role, site_name in candidates.items():
        if site_name in site_names:
            role_to_site[role] = site_name

    if role_to_site:
        merged["retarget_site_names"] = role_to_site

    # Site configs should use the site-enabled XML unless they explicitly provide a model path.
    if not any(k in cfg for k in [
        "model_path", "mujoco_model_path", "xml_path", "model_xml_path", "robot_model_path", "robot_xml"
    ]):
        merged["model_path"] = "assets/alex/alex_floating_base_with_sites.xml"

    return merged


def resolve_model_path(robot_cfg, robot_cfg_path):
    for key in [
        "model_path",
        "mujoco_model_path",
        "xml_path",
        "model_xml_path",
        "robot_model_path",
        "robot_xml",
    ]:
        if key in robot_cfg:
            return Path(robot_cfg[key])

    # Site configs created for Alex may only store site/body mappings.
    # Use the site-enabled MuJoCo XML for site configs; otherwise use the floating-base URDF.
    name = robot_cfg_path.name.lower()
    if "site" in name:
        return Path("assets/alex/alex_floating_base_with_sites.xml")
    return Path("assets/alex/alex_floating_base.urdf")


def load_canonical_frames(npz_path, start_frame=0, stride=1, max_frames=None, recenter=True):
    d = np.load(npz_path, allow_pickle=True)
    positions = np.asarray(d["positions"], dtype=float)
    roles = [str(x) for x in d["roles"].tolist()]
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])

    role_to_idx = {r: i for i, r in enumerate(roles)}
    missing = [r for r in CANONICAL_BODY_NAMES if r not in role_to_idx]
    if missing:
        raise RuntimeError(f"Canonical NPZ missing roles: {missing}")

    frame_ids = np.arange(positions.shape[0])
    frame_ids = frame_ids[start_frame::stride]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]

    source_frames = []
    for src_i in frame_ids:
        frame = {}
        for role in CANONICAL_BODY_NAMES:
            p = positions[int(src_i), role_to_idx[role]]
            frame[role] = {
                "pos": [float(x) for x in p],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
            }
        source_frames.append(frame)

    if recenter and source_frames:
        source_frames = S.recenter_clip_xy(source_frames)

    meta = {
        "npz_path": str(npz_path),
        "source_fps": fps,
        "stride": stride,
        "output_fps": fps / float(stride),
        "source_frame_start_index": int(start_frame),
        "source_frame_ids": [int(x) for x in frame_ids.tolist()],
        "source_roles": roles,
    }
    return source_frames, meta



def load_contact_masks_for_solver(contact_mask_npz, source_frame_ids, roles):
    if contact_mask_npz is None:
        return None

    contact_mask_npz = Path(contact_mask_npz)
    if not contact_mask_npz.is_absolute():
        contact_mask_npz = REPO_ROOT / contact_mask_npz

    d = np.load(contact_mask_npz, allow_pickle=True)
    out = {}

    source_frame_ids = np.asarray(source_frame_ids, dtype=int)

    for role in roles:
        key = f"{role}_stable_contact"
        if key not in d.files:
            raise RuntimeError(f"Contact mask missing key: {key} in {contact_mask_npz}")

        full = np.asarray(d[key], dtype=bool)
        safe_ids = np.clip(source_frame_ids, 0, len(full) - 1)
        out[role] = full[safe_ids]

    return out


def contact_aware_target_stabilize(frame, frame_idx, contact_masks, contact_state, roles, mode="xy", alpha=0.75):
    """Softly anchor contact-role target positions in target space. Does not modify source motion."""
    if contact_masks is None:
        return frame

    new_frame = {}
    for role, value in frame.items():
        new_frame[role] = dict(value)
        new_frame[role]["pos"] = list(value["pos"])

    for role in roles:
        if role not in new_frame or role not in contact_masks:
            continue

        active = bool(contact_masks[role][frame_idx])
        pos = np.asarray(new_frame[role]["pos"], dtype=float)

        if active:
            if (role not in contact_state) or (not contact_state[role].get("active", False)):
                contact_state[role] = {
                    "active": True,
                    "anchor": pos.copy(),
                    "start_frame": frame_idx,
                }

            anchor = np.asarray(contact_state[role]["anchor"], dtype=float)

            if mode == "xy":
                pos[:2] = (1.0 - alpha) * pos[:2] + alpha * anchor[:2]
            elif mode == "xyz":
                pos[:] = (1.0 - alpha) * pos[:] + alpha * anchor[:]
            else:
                raise RuntimeError(f"Unknown contact anchor mode: {mode}")

            new_frame[role]["pos"] = [float(x) for x in pos]
        else:
            if role in contact_state:
                contact_state[role]["active"] = False

    return new_frame



def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,         2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,         1 - 2*x*x - 2*y*y],
    ], dtype=float)


def rotmat_to_quat_wxyz_single(R):
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(max(tr + 1.0, 1e-12)) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
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
    return q / max(np.linalg.norm(q), 1e-12)


def get_frame_world_rotation_from_task_target(configuration, task):
    """Return current world rotation of the task frame/site/body at the rest configuration."""
    # Mink FrameTask stores the task frame under frame_name/frame_type.  Do not
    # silently fall back to identity here: doing so makes relative orientation
    # transfer independent of Alex's actual task-frame convention.
    frame_name = getattr(task, "frame_name", None)
    frame_type = getattr(task, "frame_type", None)
    if frame_name is None or frame_type is None:
        raise AttributeError(
            "Mink FrameTask must expose frame_name and frame_type to capture "
            "the Alex rest orientation."
        )

    transform = configuration.get_transform_frame_to_world(frame_name, frame_type)
    return np.asarray(transform.rotation().as_matrix(), dtype=float)


def load_human_role_quats_for_solver(canonical_npz_path, source_frame_ids, roles):
    d = np.load(canonical_npz_path, allow_pickle=True)
    if "role_quats_wxyz" not in d.files:
        raise RuntimeError(f"{canonical_npz_path} does not contain role_quats_wxyz. Run temp_add_human_segment_orientations.py first.")

    canonical_roles = [str(x) for x in d["roles"].tolist()]
    role_to_idx = {r: i for i, r in enumerate(canonical_roles)}
    full_quats = np.asarray(d["role_quats_wxyz"], dtype=float)

    source_frame_ids = np.asarray(source_frame_ids, dtype=int)
    safe_ids = np.clip(source_frame_ids, 0, full_quats.shape[0] - 1)

    out = {}
    for role in roles:
        if role not in role_to_idx:
            raise RuntimeError(f"Role {role} missing from canonical role_quats_wxyz roles: {canonical_roles}")
        out[role] = full_quats[safe_ids, role_to_idx[role], :]

    return out


def apply_human_relative_orientations_to_frame(
    frame,
    frame_idx,
    human_quats,
    human_rest_R,
    robot_rest_R,
    roles,
    transfer_mode="world_delta",
):
    """
    True segment orientation transfer:
        R_delta_human = R_human_t @ R_human_rest.T
        R_robot_target = R_delta_human @ R_robot_rest
    """
    new_frame = {}
    for role, value in frame.items():
        new_frame[role] = dict(value)
        if "pos" in value:
            new_frame[role]["pos"] = list(value["pos"])

    for role in roles:
        if role not in new_frame or role not in human_quats:
            continue

        R_h0 = human_rest_R[role]
        R_ht = quat_wxyz_to_rotmat(human_quats[role][frame_idx])
        R_r0 = robot_rest_R[role]

        if transfer_mode == "world_delta":
            # Apply the human segment's world-space change to the robot rest frame.
            R_delta = R_ht @ R_h0.T
            R_target = R_delta @ R_r0
        elif transfer_mode == "local_delta":
            # Apply the human segment's local rest-to-current change in the robot's rest frame.
            R_delta = R_h0.T @ R_ht
            R_target = R_r0 @ R_delta
        else:
            raise ValueError(f"Unknown human orientation transfer mode: {transfer_mode}")

        q_target = rotmat_to_quat_wxyz_single(R_target)

        new_frame[role]["quat_wxyz"] = [float(x) for x in q_target]
        new_frame[role]["quat"] = [float(x) for x in q_target]

    return new_frame


def apply_human_orientation_costs_to_tasks(tasks, roles, args):
    role_to_cost = {
        "pelvis": args.human_pelvis_ori_cost,
        "head": args.human_head_ori_cost,
        "left_foot": args.human_foot_ori_cost,
        "right_foot": args.human_foot_ori_cost,
        "left_knee": args.human_knee_ori_cost,
        "right_knee": args.human_knee_ori_cost,
        "left_hand": args.human_hand_ori_cost,
        "right_hand": args.human_hand_ori_cost,
        "left_elbow": args.human_elbow_ori_cost,
        "right_elbow": args.human_elbow_ori_cost,
        "left_shoulder": args.human_shoulder_ori_cost,
        "right_shoulder": args.human_shoulder_ori_cost,
    }

    applied = {}
    for role in roles:
        if role not in tasks:
            print(f"WARNING: orientation role {role} has no IK task; skipping.")
            continue

        cost = float(role_to_cost.get(role, 0.0))
        task = tasks[role]

        if not hasattr(task, "set_orientation_cost"):
            raise AttributeError(
                f"Task for role {role} does not expose set_orientation_cost(); "
                "do not assign orientation_cost directly."
            )

        # FrameTask stores the actual orientation weights in task.cost[3:].
        # Assigning task.orientation_cost only changes an attribute and does
        # not update the QP objective in Mink.
        task.set_orientation_cost(cost)
        effective_cost = np.asarray(task.cost[3:], dtype=float)
        if not np.allclose(effective_cost, cost):
            raise RuntimeError(
                f"Failed to apply orientation cost for {role}: "
                f"expected {cost}, got {effective_cost.tolist()}"
            )

        applied[role] = cost

    print()
    print("Applied HUMAN segment orientation costs:")
    for role in roles:
        if role in applied:
            print(f"  {role:14s}: {applied[role]:.4f} | task.cost={tasks[role].cost}")

    return applied


def joint_qpos_width(model, joint_id):
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def output_step_cap_for_joint_name(name, args):
    """Return the configured output-frame cap for an Alex joint, or None."""
    upper = name.upper()
    if "SPINE" in upper or "NECK" in upper:
        return args.max_spine_neck_step_rad
    if "HIP" in upper or "KNEE" in upper or "ANKLE" in upper:
        return args.max_leg_step_rad
    if any(token in upper for token in ("SHOULDER", "ELBOW", "WRIST", "GRIPPER", "FINGER", "THUMB", "INDEX", "MIDDLE", "RING", "PINKY")):
        return args.max_arm_step_rad
    return args.max_other_step_rad


def build_output_joint_step_caps(model, joint_names, args):
    """Build an nq-sized cap vector; inf means no output-frame cap."""
    caps = np.full(model.nq, np.inf, dtype=float)
    for name in joint_names:
        cap = output_step_cap_for_joint_name(name, args)
        if cap is None or cap <= 0.0:
            continue
        joint_id = model.joint(name).id
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos_width = joint_qpos_width(model, joint_id)
        caps[qpos_adr:qpos_adr + qpos_width] = float(cap)
    return caps if np.any(np.isfinite(caps)) else None


def build_output_joint_velocity_limit(model, joint_names, output_joint_step_caps, max_iter):
    """
    Convert output-frame step caps into conservative per-QP velocity limits.

    solve_frame integrates ``max_iter`` steps of size model.opt.timestep. A
    cap of d radians per 30 Hz output frame therefore maps to d / (N * dt)
    rad/s inside each QP. This makes intermediate candidates available for the
    temporal selector instead of letting the first QP step leap past the cap.
    """
    if output_joint_step_caps is None:
        return None, {}
    if max_iter <= 0:
        raise ValueError("max_iter must be positive when output-frame caps are enabled")

    velocities = {}
    for name in joint_names:
        joint_id = model.joint(name).id
        if joint_qpos_width(model, joint_id) != 1:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        cap = float(output_joint_step_caps[qpos_adr])
        if np.isfinite(cap):
            velocities[name] = cap / (float(max_iter) * float(model.opt.timestep))

    if not velocities:
        return None, {}
    return S.mink.VelocityLimit(model, velocities=velocities), velocities


def joint_limit_margins(model, qpos, joint_names):
    """Return lower, upper, and nearest-limit margins for the requested joints."""
    lower = np.full(len(joint_names), np.inf, dtype=float)
    upper = np.full(len(joint_names), np.inf, dtype=float)

    for index, name in enumerate(joint_names):
        joint_id = model.joint(name).id
        if not bool(model.jnt_limited[joint_id]):
            continue
        if joint_qpos_width(model, joint_id) != 1:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        value = float(qpos[qpos_adr])
        qmin, qmax = np.asarray(model.jnt_range[joint_id], dtype=float)
        lower[index] = value - qmin
        upper[index] = qmax - value

    return lower, upper, np.minimum(lower, upper)


def main():
    parser = argparse.ArgumentParser(description="Retarget canonical human role positions to Alex IK.")
    parser.add_argument("canonical_npz", type=Path)
    parser.add_argument("--robot-config", type=Path, default=Path("general_motion_retargeting/robot_configs/alex.json"))
    parser.add_argument("--output-stem", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/debug"))

    parser.add_argument("--solver", choices=["auto", "proxqp", "daqp"], default="auto")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-ik-iter", type=int, default=50)

    parser.add_argument(
        "--candidate-selection",
        choices=["position_only", "position_tiebreak", "combined"],
        default="position_only",
        help="How to select among inner IK iterates. position_only preserves the baseline.",
    )
    parser.add_argument(
        "--candidate-position-tolerance",
        type=float,
        default=0.0,
        help="Relative position-score tolerance for position_tiebreak, e.g. 0.05 for 5%%.",
    )
    parser.add_argument("--candidate-orientation-weight", type=float, default=1.0)
    parser.add_argument("--candidate-temporal-weight", type=float, default=1.0)
    parser.add_argument("--max-leg-step-rad", type=float, default=None)
    parser.add_argument("--max-spine-neck-step-rad", type=float, default=None)
    parser.add_argument("--max-arm-step-rad", type=float, default=None)
    parser.add_argument("--max-other-step-rad", type=float, default=None)
    parser.add_argument(
        "--joint-limit-warning-margin-rad",
        type=float,
        default=0.05,
        help="Report output frames with a joint this close to a hard position limit.",
    )

    parser.add_argument("--posture-cost", type=float, default=0.0)
    parser.add_argument("--posture-neutral-blend", type=float, default=0.02)
    parser.add_argument("--posture-mode", choices=["scalar", "selective"], default="scalar")
    parser.add_argument("--posture-wrist-cost", type=float, default=1.0)
    parser.add_argument("--posture-gripper-cost", type=float, default=0.7)
    parser.add_argument("--posture-ankle-cost", type=float, default=0.10)
    parser.add_argument("--posture-spine-neck-cost", type=float, default=0.05)
    parser.add_argument("--posture-arm-cost", type=float, default=0.10)
    parser.add_argument("--posture-leg-cost", type=float, default=0.0)
    parser.add_argument("--posture-base-cost", type=float, default=0.0)

    parser.add_argument("--foot-cost-mult", type=float, default=1.0)
    parser.add_argument("--head-cost-mult", type=float, default=1.0)
    parser.add_argument("--hand-cost-mult", type=float, default=1.0)
    parser.add_argument("--pelvis-cost-mult", type=float, default=1.0)

    # True human-segment relative orientation tracking.
    # Requires canonical NPZ field: role_quats_wxyz [T, R, 4].
    parser.add_argument("--human-orientation-roles", type=str, default="")
    parser.add_argument(
        "--human-orientation-transfer-mode",
        type=str,
        default="world_delta",
        choices=["world_delta", "local_delta"],
        help="world_delta: R_h(t)R_h(0)^T R_r(0); local_delta: R_r(0)R_h(0)^T R_h(t)",
    )
    parser.add_argument("--human-pelvis-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-head-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-foot-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-knee-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-hand-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-elbow-ori-cost", type=float, default=0.0)
    parser.add_argument("--human-shoulder-ori-cost", type=float, default=0.0)

    parser.add_argument("--target-rest-mode", choices=["aligned-source-rest", "raw-alex-default"], default="aligned-source-rest")
    parser.add_argument("--target-generation", choices=["rest-delta", "tree-scale", "morphology-delta"], default="morphology-delta")
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--no-recenter", action="store_true")

    parser.add_argument("--contact-mask-npz", type=Path, default=None)
    parser.add_argument("--contact-roles", type=str, default="")
    parser.add_argument("--contact-anchor-mode", choices=["xy", "xyz"], default="xy")
    parser.add_argument("--contact-anchor-alpha", type=float, default=0.75)

    args = parser.parse_args()

    S.apply_cost_multipliers(args)

    canonical_npz = args.canonical_npz.resolve()
    robot_cfg_path = args.robot_config
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = REPO_ROOT / robot_cfg_path
    robot_cfg = merge_robot_config(robot_cfg_path)

    model_path = resolve_model_path(robot_cfg, robot_cfg_path)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Canonical NPZ:", canonical_npz)
    print("Robot config:", robot_cfg_path)
    print("Model:", model_path)
    print("Output stem:", args.output_stem)

    print()
    print("Loading canonical source frames...")
    source_frames, source_meta = load_canonical_frames(
        canonical_npz,
        start_frame=args.start_frame,
        stride=args.stride,
        max_frames=args.max_frames,
        recenter=not args.no_recenter,
    )

    if not source_frames:
        raise RuntimeError("No canonical source frames loaded.")

    output_fps = float(source_meta["output_fps"])

    print("Loaded source frames:", len(source_frames))
    print("Source FPS:", source_meta["source_fps"])
    print("Stride:", args.stride)
    print("Output FPS:", output_fps)

    contact_roles = [r.strip() for r in args.contact_roles.split(",") if r.strip()]
    contact_masks = None
    contact_state = {}
    if args.contact_mask_npz is not None and contact_roles:
        contact_masks = load_contact_masks_for_solver(
            contact_mask_npz=args.contact_mask_npz,
            source_frame_ids=source_meta["source_frame_ids"],
            roles=contact_roles,
        )
        print()
        print("Contact-aware target stabilization:")
        print("  mask npz:", args.contact_mask_npz)
        print("  roles:", contact_roles)
        print("  mode:", args.contact_anchor_mode)
        print("  alpha:", args.contact_anchor_alpha)
        for role in contact_roles:
            print(f"  {role:12s} active frames: {int(contact_masks[role].sum())}/{len(contact_masks[role])}")

    source_rest = source_frames[0]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    solver = S.choose_solver(args.solver)
    print("Using solver:", solver)

    qpos0 = np.asarray(model.qpos0, dtype=float)
    if qpos0.shape[0] != 36:
        raise RuntimeError(f"Expected Alex nq=36, got {qpos0.shape[0]}")

    first_pelvis = np.asarray(source_frames[0]["pelvis"]["pos"], dtype=float)
    qpos0[0:3] = first_pelvis
    qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

    limits = [S.mink.ConfigurationLimit(model)]
    tasks = S.make_tasks(model, robot_cfg)
    output_joint_step_caps = build_output_joint_step_caps(
        model,
        robot_cfg["actuated_joint_order"],
        args,
    )
    output_joint_velocity_limit, output_joint_velocity_limits = build_output_joint_velocity_limit(
        model,
        robot_cfg["actuated_joint_order"],
        output_joint_step_caps,
        args.max_ik_iter,
    )
    motion_limits = list(limits)
    if output_joint_velocity_limit is not None:
        motion_limits.append(output_joint_velocity_limit)
    if output_joint_step_caps is not None:
        capped = []
        for name in robot_cfg["actuated_joint_order"]:
            joint_id = model.joint(name).id
            qpos_adr = int(model.jnt_qposadr[joint_id])
            cap = output_joint_step_caps[qpos_adr]
            if np.isfinite(cap):
                capped.append(f"{name}={cap:.3f}")
        print("Output-frame joint-step caps (rad):", ", ".join(capped))
        print(
            "Derived inner-QP velocity limits (rad/s):",
            ", ".join(f"{name}={value:.3f}" for name, value in output_joint_velocity_limits.items()),
        )
    human_orientation_roles = [r.strip() for r in args.human_orientation_roles.split(",") if r.strip()]
    human_orientation_costs = {}

    human_quats_by_role = None
    human_rest_R = {}
    robot_rest_R = {}

    if human_orientation_roles:
        # Use the same canonical input file that generated source_frames.
        canonical_npz_for_orient = Path(args.canonical_npz)
        if not canonical_npz_for_orient.is_absolute():
            canonical_npz_for_orient = REPO_ROOT / canonical_npz_for_orient

        human_quats_by_role = load_human_role_quats_for_solver(
            canonical_npz_path=canonical_npz_for_orient,
            source_frame_ids=source_meta["source_frame_ids"],
            roles=human_orientation_roles,
        )

        # Human rest rotations from the first selected source frame.
        for role in human_orientation_roles:
            human_rest_R[role] = quat_wxyz_to_rotmat(human_quats_by_role[role][0])

        # Keep orientation costs at zero for rest alignment.  source_frames
        # intentionally carry identity quaternions; applying human orientation
        # costs before rest alignment would incorrectly target world identity.

    posture_task = S.make_posture_task(model, args)

    target_rest = S.robot_rest_frame_from_mujoco(
        model=model,
        qpos=qpos0.copy(),
        role_to_robot=robot_cfg["retarget_body_names"],
    )

    rest_score = None
    rest_errors = None
    qpos_rest = None

    if args.target_rest_mode == "aligned-source-rest":
        print()
        print("Aligning Alex target rest pose to first canonical source frame...")
        rest_configuration = S.mink.Configuration(model, q=qpos0.copy())
        rest_target_by_role = S.set_task_targets(tasks, source_rest, robot_cfg)

        S.set_posture_target(posture_task, rest_configuration, qpos0.copy())

        qpos_rest, rest_score, rest_errors, rest_solved_positions = S.solve_frame(
            model=model,
            configuration=rest_configuration,
            tasks=tasks,
            target_by_role=rest_target_by_role,
            solver=solver,
            limits=limits,
            max_iter=args.max_ik_iter,
            posture_task=posture_task,
        )

        print(f"Rest-alignment score: {rest_score:.6f}")
        for role, err in rest_errors.items():
            print(f"  rest {role:10s} error: {err:.6f} m")

        target_rest = S.robot_rest_frame_from_mujoco(
            model=model,
            qpos=qpos_rest.copy(),
            role_to_robot=robot_cfg["retarget_body_names"],
        )
        configuration = S.mink.Configuration(model, q=qpos_rest.copy())
        q_prev = qpos_rest.copy()
    else:
        configuration = S.mink.Configuration(model, q=qpos0.copy())
        q_prev = qpos0.copy()

    q_neutral = q_prev.copy()

    if human_orientation_roles:
        print()
        print("Capturing Alex rest task orientations after position-only rest alignment...")
        for role in human_orientation_roles:
            if role not in tasks:
                print(f"  {role:14s}: missing task")
                continue
            robot_rest_R[role] = get_frame_world_rotation_from_task_target(configuration, tasks[role])
            print(f"  {role:14s}: captured")

        missing_rest_frames = sorted(set(human_orientation_roles) - set(robot_rest_R))
        if missing_rest_frames:
            raise RuntimeError(
                "Cannot enable human orientation tracking without Alex rest task frames: "
                f"{missing_rest_frames}"
            )

        human_orientation_costs = apply_human_orientation_costs_to_tasks(
            tasks,
            human_orientation_roles,
            args,
        )

    morphology_scales = None
    if args.target_generation == "morphology-delta":
        morphology = S.compute_morphology_scales(
            source_rest=source_rest,
            target_rest=target_rest,
            preserve_root_translation=True,
            clamp_min=0.70,
            clamp_max=1.30,
        )
        morphology_scales = morphology.role_scales
        print()
        print("Morphology-delta role scales:")
        for role, scale in morphology.role_scales.items():
            print(f"  {role:16s}: {scale:.4f}")

    qpos_traj = []
    rows = []
    target_positions = []
    solved_ik_positions = []
    target_orientations_wxyz = []
    solved_ik_orientations_wxyz = []
    joint_limit_lower_margins = []
    joint_limit_upper_margins = []
    joint_limit_nearest_margins_by_frame = []
    selection_diagnostics = []

    for frame_idx, source_frame in enumerate(source_frames):
        if args.target_generation == "morphology-delta":
            scaled_frame = S.make_morphology_delta_target_frame(
                source_frame=source_frame,
                source_rest=source_rest,
                target_rest=target_rest,
                scales=morphology_scales,
            )
        elif args.target_generation == "rest-delta":
            scaled_frame = S.make_rest_delta_target_frame(
                source_frame=source_frame,
                source_rest=source_rest,
                target_rest=target_rest,
                motion_scale=args.motion_scale,
            )
        else:
            scaled_frame = S.scale_frame_by_rest_pose(
                frame=source_frame,
                source_rest_frame=source_rest,
                target_rest_frame=target_rest,
            )

        if contact_masks is not None:
            scaled_frame = contact_aware_target_stabilize(
                frame=scaled_frame,
                frame_idx=frame_idx,
                contact_masks=contact_masks,
                contact_state=contact_state,
                roles=contact_roles,
                mode=args.contact_anchor_mode,
                alpha=args.contact_anchor_alpha,
            )

        if human_quats_by_role is not None:
            scaled_frame = apply_human_relative_orientations_to_frame(
                frame=scaled_frame,
                frame_idx=frame_idx,
                human_quats=human_quats_by_role,
                human_rest_R=human_rest_R,
                robot_rest_R=robot_rest_R,
                roles=human_orientation_roles,
                transfer_mode=args.human_orientation_transfer_mode,
            )

        target_by_role = S.set_task_targets(tasks, scaled_frame, robot_cfg)

        if posture_task is not None:
            q_ref = S.blended_posture_target(
                q_prev=q_prev,
                q_neutral=q_neutral,
                neutral_blend=args.posture_neutral_blend,
            )
            S.set_posture_target(posture_task, configuration, q_ref)

        qpos, score, errors, solved_positions, selection_info = S.solve_frame(
            model=model,
            configuration=configuration,
            tasks=tasks,
            target_by_role=target_by_role,
            solver=solver,
            limits=motion_limits,
            max_iter=args.max_ik_iter,
            posture_task=posture_task,
            candidate_selection=args.candidate_selection,
            candidate_position_tolerance=args.candidate_position_tolerance,
            candidate_orientation_weight=args.candidate_orientation_weight,
            candidate_temporal_weight=args.candidate_temporal_weight,
            output_joint_step_caps=output_joint_step_caps,
            return_diagnostics=True,
        )

        q_prev = qpos.copy()
        qpos_traj.append(qpos)

        target_positions.append([
            np.asarray(target_by_role[role]["target_pos"], dtype=float)
            for role in IK_ROLES
        ])
        solved_ik_positions.append([
            np.asarray(solved_positions[role], dtype=float)
            for role in IK_ROLES
        ])
        target_orientations_wxyz.append([
            np.asarray(scaled_frame[role]["quat_wxyz"], dtype=float)
            for role in IK_ROLES
        ])
        solved_ik_orientations_wxyz.append([
            rotmat_to_quat_wxyz_single(
                get_frame_world_rotation_from_task_target(configuration, tasks[role])
            )
            for role in IK_ROLES
        ])
        lower_margin, upper_margin, nearest_margin = joint_limit_margins(
            model,
            qpos,
            robot_cfg["actuated_joint_order"],
        )
        joint_limit_lower_margins.append(lower_margin)
        joint_limit_upper_margins.append(upper_margin)
        joint_limit_nearest_margins_by_frame.append(nearest_margin)
        selection_diagnostics.append(selection_info)

        nearest_index = int(np.argmin(nearest_margin))
        nearest_joint = robot_cfg["actuated_joint_order"][nearest_index]
        nearest_value = float(nearest_margin[nearest_index])
        nearest_side = "lower" if lower_margin[nearest_index] <= upper_margin[nearest_index] else "upper"

        hand_err = []
        for role in ("left_hand", "right_hand"):
            if role in errors:
                hand_err.append(errors[role])

        foot_err = []
        for role in ("left_foot", "right_foot"):
            if role in errors:
                foot_err.append(errors[role])

        row = {
            "frame": frame_idx,
            "position_score": float(score),
            "pelvis_error_m": float(errors.get("pelvis", np.nan)),
            "left_foot_error_m": float(errors.get("left_foot", np.nan)),
            "right_foot_error_m": float(errors.get("right_foot", np.nan)),
            "mean_foot_error_m": float(np.nanmean(foot_err)) if foot_err else np.nan,
            "left_hand_error_m": float(errors.get("left_hand", np.nan)),
            "right_hand_error_m": float(errors.get("right_hand", np.nan)),
            "mean_hand_error_m": float(np.nanmean(hand_err)) if hand_err else np.nan,
            "root_x": float(qpos[0]),
            "root_y": float(qpos[1]),
            "root_z": float(qpos[2]),
            "nearest_joint_limit": nearest_joint,
            "nearest_joint_limit_side": nearest_side,
            "nearest_joint_limit_margin_rad": nearest_value,
            "num_joints_within_limit_margin": int(np.sum(nearest_margin <= args.joint_limit_warning_margin_rad)),
            "selection_position_score": float(selection_info["selected_position_score"]),
            "selection_orientation_score": float(selection_info["selected_orientation_score"]),
            "selection_temporal_displacement": float(selection_info["selected_temporal_displacement"]),
            "selection_max_actuated_step_rad": float(selection_info["selected_max_actuated_step_rad"]),
            "selection_candidate_count": int(selection_info["candidate_count"]),
            "selection_eligible_candidate_count": int(selection_info["eligible_candidate_count"]),
            "selection_rejected_candidate_count": int(selection_info["rejected_candidate_count"]),
            "selection_iteration": int(selection_info["selected_iteration"]),
        }
        rows.append(row)

        if frame_idx % 10 == 0 or frame_idx == len(source_frames) - 1:
            print(
                f"frame {frame_idx:03d}: "
                f"score={score:.4f}, "
                f"pelvis={row['pelvis_error_m']:.3f}, "
                f"feet={row['mean_foot_error_m']:.3f}, "
                f"hands={row['mean_hand_error_m']:.3f}, "
                f"limit={nearest_joint}:{nearest_value:.3f}"
            )

    qpos_traj = np.asarray(qpos_traj, dtype=float)
    source_positions = S.frame_positions_array(source_frames, CANONICAL_BODY_NAMES)
    target_positions = np.asarray(target_positions, dtype=float)
    solved_ik_positions = np.asarray(solved_ik_positions, dtype=float)
    target_orientations_wxyz = np.asarray(target_orientations_wxyz, dtype=float)
    solved_ik_orientations_wxyz = np.asarray(solved_ik_orientations_wxyz, dtype=float)
    joint_limit_lower_margins = np.asarray(joint_limit_lower_margins, dtype=float)
    joint_limit_upper_margins = np.asarray(joint_limit_upper_margins, dtype=float)
    joint_limit_nearest_margins_by_frame = np.asarray(joint_limit_nearest_margins_by_frame, dtype=float)

    joint_delta = np.diff(qpos_traj[:, 7:], axis=0)
    root_delta = np.diff(qpos_traj[:, 0:3], axis=0)

    npz_path = out_dir / f"{args.output_stem}.npz"
    csv_path = out_dir / f"{args.output_stem}_errors.csv"
    json_path = out_dir / f"{args.output_stem}_summary.json"
    plot_path = out_dir / f"{args.output_stem}_errors.png"

    summary = {
        "note": "Canonical human role positions to Alex IK.",
        "canonical_npz": str(canonical_npz),
        "robot_config": str(robot_cfg_path),
        "model_path": str(model_path),
        "solver": solver,
        "start_frame": args.start_frame,
        "stride": args.stride,
        "max_frames_requested": args.max_frames,
        "num_frames": len(source_frames),
        "source_fps": source_meta["source_fps"],
        "output_fps": output_fps,
        "max_ik_iter": args.max_ik_iter,
        "posture_cost": args.posture_cost,
        "posture_neutral_blend": args.posture_neutral_blend,
        "posture_mode": args.posture_mode,
        "target_rest_mode": args.target_rest_mode,
        "target_generation": args.target_generation,
        "motion_scale": args.motion_scale,
        "contact_mask_npz": None if args.contact_mask_npz is None else str(args.contact_mask_npz),
        "contact_roles": contact_roles,
        "contact_anchor_mode": args.contact_anchor_mode,
        "contact_anchor_alpha": args.contact_anchor_alpha,
        "foot_cost_mult": args.foot_cost_mult,
        "head_cost_mult": args.head_cost_mult,
        "hand_cost_mult": args.hand_cost_mult,
        "pelvis_cost_mult": args.pelvis_cost_mult,
        "human_orientation_roles": human_orientation_roles,
        "human_orientation_transfer_mode": args.human_orientation_transfer_mode,
        "human_orientation_costs": human_orientation_costs,
        "orientation_tracking_active": bool(human_orientation_roles),
        "candidate_selection": args.candidate_selection,
        "candidate_position_tolerance": args.candidate_position_tolerance,
        "candidate_orientation_weight": args.candidate_orientation_weight,
        "candidate_temporal_weight": args.candidate_temporal_weight,
        "max_leg_step_rad": args.max_leg_step_rad,
        "max_spine_neck_step_rad": args.max_spine_neck_step_rad,
        "max_arm_step_rad": args.max_arm_step_rad,
        "max_other_step_rad": args.max_other_step_rad,
        "output_step_velocity_limit_enabled": output_joint_velocity_limit is not None,
        "output_joint_velocity_limits_rad_s": output_joint_velocity_limits,
        "joint_limit_warning_margin_rad": args.joint_limit_warning_margin_rad,
        "num_frames_near_joint_limit": int(np.sum(np.min(joint_limit_nearest_margins_by_frame, axis=1) <= args.joint_limit_warning_margin_rad)),
        "min_joint_limit_margin_rad": float(np.min(joint_limit_nearest_margins_by_frame)),
        "morphology_scales": morphology_scales,
        "rest_alignment_score": None if rest_score is None else float(rest_score),
        "rest_alignment_errors_m": rest_errors,
        "qpos_shape": list(qpos_traj.shape),
        "qpos_layout": robot_cfg["floating_base"]["qpos_layout"],
        "source_roles": list(CANONICAL_BODY_NAMES),
        "ik_roles": list(IK_ROLES),
        "mean_position_score": float(np.mean([r["position_score"] for r in rows])),
        "max_position_score": float(np.max([r["position_score"] for r in rows])),
        "mean_hand_error_m": float(np.nanmean([r["mean_hand_error_m"] for r in rows])),
        "max_hand_error_m": float(np.nanmax([r["mean_hand_error_m"] for r in rows])),
        "mean_foot_error_m": float(np.nanmean([r["mean_foot_error_m"] for r in rows])),
        "max_foot_error_m": float(np.nanmax([r["mean_foot_error_m"] for r in rows])),
        "mean_pelvis_error_m": float(np.nanmean([r["pelvis_error_m"] for r in rows])),
        "max_pelvis_error_m": float(np.nanmax([r["pelvis_error_m"] for r in rows])),
        "max_abs_joint_step_rad": float(np.max(np.abs(joint_delta))) if len(joint_delta) else 0.0,
        "max_root_step_m": float(np.max(np.linalg.norm(root_delta, axis=1))) if len(root_delta) else 0.0,
        "source_meta": source_meta,
    }

    np.savez(
        npz_path,
        qpos=qpos_traj,
        fps=np.array([output_fps], dtype=float),
        joint_names=np.asarray(robot_cfg["actuated_joint_order"], dtype=object),
        source_positions=source_positions,
        source_roles=np.asarray(CANONICAL_BODY_NAMES, dtype=object),
        target_positions=target_positions,
        solved_ik_positions=solved_ik_positions,
        target_orientations_wxyz=target_orientations_wxyz,
        solved_ik_orientations_wxyz=solved_ik_orientations_wxyz,
        ik_roles=np.asarray(IK_ROLES, dtype=object),
        source_frame_ids=np.asarray(source_meta["source_frame_ids"], dtype=int),
        human_orientation_roles=np.asarray(human_orientation_roles, dtype=object),
        human_orientation_transfer_mode=np.asarray(args.human_orientation_transfer_mode, dtype=object),
        joint_limit_joint_names=np.asarray(robot_cfg["actuated_joint_order"], dtype=object),
        joint_limit_lower_margin_rad=joint_limit_lower_margins,
        joint_limit_upper_margin_rad=joint_limit_upper_margins,
        joint_limit_margin_rad=joint_limit_nearest_margins_by_frame,
        output_joint_step_caps_rad=(
            np.full(len(robot_cfg["actuated_joint_order"]), np.inf, dtype=float)
            if output_joint_step_caps is None
            else np.asarray(output_joint_step_caps[7:], dtype=float)
        ),
        output_joint_velocity_limits_rad_s=np.asarray(
            [output_joint_velocity_limits.get(name, np.inf) for name in robot_cfg["actuated_joint_order"]],
            dtype=float,
        ),
        robot_config_path=np.asarray(str(robot_cfg_path), dtype=object),
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(summary, indent=2))

    frames = np.asarray([r["frame"] for r in rows])
    plt.figure(figsize=(10, 6))
    plt.plot(frames, [r["pelvis_error_m"] for r in rows], label="pelvis")
    plt.plot(frames, [r["mean_foot_error_m"] for r in rows], label="feet mean")
    plt.plot(frames, [r["mean_hand_error_m"] for r in rows], label="hands mean")
    plt.xlabel("frame")
    plt.ylabel("position error (m)")
    plt.title(args.output_stem)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print()
    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        elif key != "source_meta":
            print(f"  {key}: {value}")

    print()
    print("Wrote:")
    print(" ", npz_path)
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", plot_path)


if __name__ == "__main__":
    main()

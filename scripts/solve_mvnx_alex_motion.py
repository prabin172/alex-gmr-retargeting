from pathlib import Path
import argparse
import csv
import json

import mujoco
import mink
import numpy as np
import matplotlib.pyplot as plt

from general_motion_retargeting.source_adapters.canonical_human import (
    CANONICAL_BODY_NAMES,
    validate_canonical_human_frame,
)
from general_motion_retargeting.source_adapters.mvnx import (
    read_mvnx_canonical_frames,
)
from general_motion_retargeting.retargeting.morphology_delta import (
    compute_morphology_scales,
    make_morphology_delta_target_frame,
)
from general_motion_retargeting.retargeting.rest_pose_scaling import (
    scale_frame_by_rest_pose,
)

repo_root = Path(__file__).resolve().parents[1]

IK_ROLES = [
    "pelvis",
    "head",

    # Legs: knees are included to prevent foot-only IK branch flips.
    "left_knee",
    "left_foot",
    "right_knee",
    "right_foot",

    # Arms: elbows are included to prevent hand-only IK branch flips.
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
]

COSTS = {
    "pelvis": (100.0, 0.0),
    "head": (20.0, 0.0),

    # Knees prevent leg branch flips. Keep lower than feet so feet remain primary.
    "left_knee": (25.0, 0.0),
    "right_knee": (25.0, 0.0),

    # Feet should stay accurate because they anchor the body.
    "left_foot": (80.0, 0.0),
    "right_foot": (80.0, 0.0),

    # Shoulder and elbow constraints prevent elbow-backward / branch-flip IK.
    # Keep shoulder lower than hand/foot because robot shoulder body origins may
    # not exactly match MVNX anatomical shoulder landmarks.
    "left_shoulder": (10.0, 0.0),
    "left_elbow": (30.0, 0.0),
    "left_hand": (40.0, 0.0),

    "right_shoulder": (10.0, 0.0),
    "right_elbow": (30.0, 0.0),
    "right_hand": (40.0, 0.0),
}

def choose_solver(preferred="auto"):
    import qpsolvers
    solvers = qpsolvers.available_solvers
    if callable(solvers):
        solvers = solvers()
    solvers = list(solvers)
    print("Available QP solvers:", solvers)

    if preferred and preferred != "auto":
        if preferred not in solvers:
            raise RuntimeError(f"Requested solver {preferred!r} not available. Available: {solvers}")
        return preferred

    for name in ["quadprog", "proxqp", "daqp", "osqp", "clarabel", "scs"]:
        if name in solvers:
            return name
    raise RuntimeError(f"No supported QP solver found. Available: {solvers}")


def apply_cost_multipliers(args):
    multipliers = {}
    for role in list(COSTS.keys()):
        mult = 1.0
        if role in ("left_foot", "right_foot"):
            mult *= args.foot_cost_mult
        if role == "head":
            mult *= args.head_cost_mult
        if role in ("left_hand", "right_hand"):
            mult *= args.hand_cost_mult
        if role == "pelvis":
            mult *= args.pelvis_cost_mult

        pos_cost, ori_cost = COSTS[role]
        COSTS[role] = (float(pos_cost) * float(mult), float(ori_cost))
        multipliers[role] = mult

    print()
    print("Applied IK cost multipliers:")
    print(f"  foot_cost_mult:   {args.foot_cost_mult}")
    print(f"  head_cost_mult:   {args.head_cost_mult}")
    print(f"  hand_cost_mult:   {args.hand_cost_mult}")
    print(f"  pelvis_cost_mult: {args.pelvis_cost_mult}")
    return multipliers


def body_pos(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=float).copy()


def site_pos(model, data, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Missing site: {site_name}")
    return np.asarray(data.site_xpos[site_id], dtype=float).copy()


def role_frame_name_and_type(robot_cfg, role):
    site_names = robot_cfg.get("retarget_site_names", {})
    if role in site_names:
        return site_names[role], "site"
    return robot_cfg["retarget_body_names"][role], "body"


def frame_pos(model, data, frame_name, frame_type):
    if frame_type == "site":
        return site_pos(model, data, frame_name)
    if frame_type == "body":
        return body_pos(model, data, frame_name)
    raise ValueError(f"Unsupported frame_type: {frame_type}")


def role_pos(model, data, robot_cfg, role):
    frame_name, frame_type = role_frame_name_and_type(robot_cfg, role)
    return frame_pos(model, data, frame_name, frame_type)

def robot_rest_frame_from_mujoco(model, qpos, role_to_robot):
    """
    Build a canonical robot rest frame from MuJoCo.

    Backward compatible:
      - role_to_robot can be the old retarget_body_names dict
      - or the full robot_cfg with retarget_body_names and retarget_site_names
    """
    if "retarget_body_names" in role_to_robot:
        robot_cfg = role_to_robot
    else:
        robot_cfg = {
            "retarget_body_names": role_to_robot,
            "retarget_site_names": {},
        }

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    frame = {}
    for role in CANONICAL_BODY_NAMES:
        if role in robot_cfg.get("retarget_site_names", {}) or role in robot_cfg["retarget_body_names"]:
            p = role_pos(model, data, robot_cfg, role)
        else:
            p = body_pos(model, data, robot_cfg["retarget_body_names"][role])

        frame[role] = {
            "pos": [float(x) for x in p],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

    validate_canonical_human_frame(frame)
    return frame


def copy_frame(frame):
    return {
        role: {
            "pos": list(frame[role]["pos"]),
            "quat_wxyz": list(frame[role]["quat_wxyz"]),
        }
        for role in CANONICAL_BODY_NAMES
    }

def recenter_clip_xy(frames):
    if not frames:
        return frames

    initial_pelvis = np.asarray(frames[0]["pelvis"]["pos"], dtype=float)
    xy_offset = np.array([initial_pelvis[0], initial_pelvis[1], 0.0], dtype=float)

    out = []
    for frame in frames:
        new_frame = copy_frame(frame)
        for role in CANONICAL_BODY_NAMES:
            p = np.asarray(new_frame[role]["pos"], dtype=float)
            p = p - xy_offset
            new_frame[role]["pos"] = [float(x) for x in p]
        validate_canonical_human_frame(new_frame)
        out.append(new_frame)

    return out


def make_rest_delta_target_frame(source_frame, source_rest, target_rest, motion_scale=1.0):
    """
    Generate IK targets by applying source motion deltas to the aligned Alex rest pose.

    This guarantees:
      source_frame == source_rest  ->  target_frame == target_rest

    That property is exactly what we want for the first real MVNX-to-Alex test.
    """
    frame = {}

    for role in CANONICAL_BODY_NAMES:
        source_pos = np.asarray(source_frame[role]["pos"], dtype=float)
        source_rest_pos = np.asarray(source_rest[role]["pos"], dtype=float)
        target_rest_pos = np.asarray(target_rest[role]["pos"], dtype=float)

        delta = source_pos - source_rest_pos
        target_pos = target_rest_pos + float(motion_scale) * delta

        frame[role] = {
            "pos": [float(x) for x in target_pos],
            "quat_wxyz": list(target_rest[role]["quat_wxyz"]),
        }

    validate_canonical_human_frame(frame)
    return frame


def frame_positions_array(frames, roles):
    arr = np.zeros((len(frames), len(roles), 3), dtype=float)
    for t, frame in enumerate(frames):
        for i, role in enumerate(roles):
            arr[t, i] = np.asarray(frame[role]["pos"], dtype=float)
    return arr


def joint_dof_count(model, jid):
    jtype = int(model.jnt_type[jid])
    if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if jtype == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1


def posture_cost_for_joint_name(name, args):
    u = name.upper()

    if "WRIST" in u:
        return args.posture_wrist_cost
    if "GRIPPER" in u or "FINGER" in u or "THUMB" in u or "INDEX" in u or "MIDDLE" in u or "RING" in u or "PINKY" in u:
        return args.posture_gripper_cost
    if "ANKLE" in u:
        return args.posture_ankle_cost
    if "SPINE" in u or "NECK" in u:
        return args.posture_spine_neck_cost
    if "SHOULDER" in u or "ELBOW" in u:
        return args.posture_arm_cost
    if "HIP" in u or "KNEE" in u:
        return args.posture_leg_cost

    return args.posture_base_cost


def make_selective_posture_cost(model, args):
    full = np.zeros(model.nv, dtype=float)

    group_counts = {
        "wrist": 0,
        "gripper": 0,
        "ankle": 0,
        "spine_neck": 0,
        "arm": 0,
        "leg": 0,
        "base_other": 0,
    }

    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        dof_adr = int(model.jnt_dofadr[jid])
        ndof = joint_dof_count(model, jid)

        # Do not regularize the floating base through posture.
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue

        c = float(args.posture_cost) * float(posture_cost_for_joint_name(name, args))
        full[dof_adr:dof_adr + ndof] = c

        u = name.upper()
        if "WRIST" in u:
            group_counts["wrist"] += ndof
        elif "GRIPPER" in u or "FINGER" in u or "THUMB" in u or "INDEX" in u or "MIDDLE" in u or "RING" in u or "PINKY" in u:
            group_counts["gripper"] += ndof
        elif "ANKLE" in u:
            group_counts["ankle"] += ndof
        elif "SPINE" in u or "NECK" in u:
            group_counts["spine_neck"] += ndof
        elif "SHOULDER" in u or "ELBOW" in u:
            group_counts["arm"] += ndof
        elif "HIP" in u or "KNEE" in u:
            group_counts["leg"] += ndof
        else:
            group_counts["base_other"] += ndof

    return full, group_counts


def make_posture_task(model, args):
    if args.posture_cost <= 0.0:
        return None
    if not hasattr(mink, "PostureTask"):
        raise RuntimeError("mink.PostureTask is not available in this environment.")

    if args.posture_mode == "scalar":
        return mink.PostureTask(model, cost=args.posture_cost)

    full_cost, group_counts = make_selective_posture_cost(model, args)

    print()
    print("Selective posture cost profile:")
    print(f"  global multiplier: {args.posture_cost}")
    print(f"  wrist:            {args.posture_wrist_cost}")
    print(f"  gripper/fingers:  {args.posture_gripper_cost}")
    print(f"  ankle:            {args.posture_ankle_cost}")
    print(f"  spine/neck:       {args.posture_spine_neck_cost}")
    print(f"  shoulder/elbow:   {args.posture_arm_cost}")
    print(f"  hip/knee:         {args.posture_leg_cost}")
    print(f"  other:            {args.posture_base_cost}")
    print(f"  dof groups:       {group_counts}")

    # Try full nv cost vector first. Some Mink versions expect model.nv.
    try:
        return mink.PostureTask(model, cost=full_cost)
    except Exception as e_full:
        # Floating-base variants sometimes expect actuated-only cost vector.
        try:
            return mink.PostureTask(model, cost=full_cost[6:])
        except Exception as e_act:
            raise RuntimeError(
                "Could not create selective PostureTask with full or actuated cost vector. "
                f"full error={e_full}; actuated error={e_act}"
            )



def set_posture_target(posture_task, configuration, qpos_target):
    if posture_task is None:
        return

    # Preferred Mink API.
    if hasattr(posture_task, "set_target"):
        posture_task.set_target(qpos_target)
        return

    # Common fallback API.
    if hasattr(posture_task, "set_target_from_configuration"):
        tmp = mink.Configuration(configuration.model, q=qpos_target.copy())
        posture_task.set_target_from_configuration(tmp)
        return

    raise RuntimeError("Could not find a supported PostureTask target setter.")


def blended_posture_target(q_prev, q_neutral, neutral_blend):
    blend = float(np.clip(neutral_blend, 0.0, 1.0))
    q_target = np.asarray(q_prev, dtype=float).copy()

    # Do not blend floating-base position/orientation aggressively.
    # For this model qpos layout is:
    #   [base_x, base_y, base_z, base_qw, base_qx, base_qy, base_qz, actuated joints...]
    if q_target.shape[0] > 7:
        q_target[7:] = (1.0 - blend) * q_target[7:] + blend * q_neutral[7:]

    return q_target


def make_tasks(model, robot_cfg):
    tasks = {}

    print()
    print("IK task frames:")
    for role in IK_ROLES:
        frame_name, frame_type = role_frame_name_and_type(robot_cfg, role)
        position_cost, orientation_cost = COSTS[role]

        tasks[role] = mink.FrameTask(
            frame_name=frame_name,
            frame_type=frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
        )

        print(
            f"  {role:16s} -> {frame_type:4s} {frame_name:28s} "
            f"pos_cost={position_cost:.3f} ori_cost={orientation_cost:.3f}"
        )

    return tasks


def set_task_targets(tasks, scaled_frame, robot_cfg):
    target_by_role = {}

    for role, task in tasks.items():
        frame_name, frame_type = role_frame_name_and_type(robot_cfg, role)
        target_pos = np.asarray(scaled_frame[role]["pos"], dtype=float)
        target_quat = np.asarray(scaled_frame[role]["quat_wxyz"], dtype=float)

        task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(target_quat),
                target_pos,
            )
        )

        target_by_role[role] = {
            "robot_frame": frame_name,
            "frame_type": frame_type,
            "target_pos": target_pos.tolist(),
        }

    return target_by_role


def position_error_score(model, data, target_by_role):
    terms = []
    for role, info in target_by_role.items():
        target = np.asarray(info["target_pos"], dtype=float)
        solved = frame_pos(model, data, info["robot_frame"], info["frame_type"])
        weight = float(COSTS[role][0])
        terms.append(np.sqrt(max(weight, 0.0)) * np.linalg.norm(solved - target))
    return float(np.linalg.norm(np.asarray(terms, dtype=float)))


def solve_frame(model, configuration, tasks, target_by_role, solver, limits, max_iter, posture_task=None):
    dt = model.opt.timestep
    damping = 1e-4

    best_score = position_error_score(model, configuration.data, target_by_role)
    best_qpos = np.asarray(configuration.data.qpos, dtype=float).copy()

    for _ in range(max_iter):
        active_tasks = list(tasks.values())
        if posture_task is not None:
            active_tasks.append(posture_task)

        vel = mink.solve_ik(
            configuration,
            active_tasks,
            dt,
            solver=solver,
            damping=damping,
            limits=limits,
        )

        configuration.integrate_inplace(vel, dt)
        score = position_error_score(model, configuration.data, target_by_role)

        if score < best_score:
            best_score = score
            best_qpos = np.asarray(configuration.data.qpos, dtype=float).copy()

    configuration.update(best_qpos)
    mujoco.mj_forward(model, configuration.data)

    errors = {}
    solved_positions = {}

    for role, info in target_by_role.items():
        target = np.asarray(info["target_pos"], dtype=float)
        solved = frame_pos(model, configuration.data, info["robot_frame"], info["frame_type"])
        errors[role] = float(np.linalg.norm(solved - target))
        solved_positions[role] = solved

    return best_qpos.copy(), best_score, errors, solved_positions

def main():
    parser = argparse.ArgumentParser(description="Retarget a short MVNX clip to Alex using position-only IK.")
    parser.add_argument("mvnx_path", type=Path)
    parser.add_argument("--robot-config", type=Path, default=Path("general_motion_retargeting/robot_configs/alex.json"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--max-ik-iter", type=int, default=50)
    parser.add_argument("--solver", choices=["auto", "proxqp", "daqp"], default="auto")
    parser.add_argument("--posture-cost", type=float, default=0.0,
                        help="Soft posture/smoothness cost. 0 disables posture regularization.")
    parser.add_argument("--posture-neutral-blend", type=float, default=0.05,
                        help="Blend target posture toward neutral qpos0 each frame. 0 means previous q only.")
    parser.add_argument("--posture-mode", choices=["scalar", "selective"], default="scalar",
                        help="scalar uses one posture cost for all joints; selective uses joint-group costs.")
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
    parser.add_argument("--source-rest", choices=["first-normal", "mvnx-tpose"], default="first-normal")
    parser.add_argument("--target-rest-mode", choices=["aligned-source-rest", "raw-alex-default"], default="aligned-source-rest")
    parser.add_argument("--target-generation", choices=["rest-delta", "tree-scale", "morphology-delta"], default="rest-delta")
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--no-heading-canonicalization", action="store_true")
    parser.add_argument("--no-recenter", action="store_true")
    args = parser.parse_args()
    cost_multipliers = apply_cost_multipliers(args)

    robot_cfg_path = args.robot_config
    if not robot_cfg_path.is_absolute():
        robot_cfg_path = repo_root / robot_cfg_path
    robot_cfg = json.loads(robot_cfg_path.read_text())
    model_path = repo_root / robot_cfg["model_path"]

    out_dir = repo_root / "outputs/debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("MVNX:", args.mvnx_path)
    print("Robot config:", robot_cfg_path)
    print("Model:", model_path)
    print("Model exists:", model_path.exists())

    if not model_path.exists():
        raise FileNotFoundError(
            "Missing floating-base Alex model. Run:\n"
            "  python scripts/prepare_alex_mujoco_assets.py\n"
            "  python scripts/prepare_alex_floating_base_model.py"
        )

    print()
    print("Loading MVNX normal frames...")
    source_frames, source_meta = read_mvnx_canonical_frames(
        mvnx_path=args.mvnx_path,
        frame_type="normal",
        start_frame=args.start_frame,
        stride=args.stride,
        max_frames=args.max_frames,
        canonicalize_axes=True,
        canonicalize_heading=not args.no_heading_canonicalization,
    )

    if not source_frames:
        raise RuntimeError("No MVNX normal frames loaded.")

    if not args.no_recenter:
        source_frames = recenter_clip_xy(source_frames)

    print("Loaded source frames:", len(source_frames))
    print("MVNX frame rate:", source_meta.get("frame_rate"))
    print("Stride:", args.stride)

    output_fps = None
    if source_meta.get("frame_rate"):
        output_fps = source_meta["frame_rate"] / args.stride
    else:
        output_fps = 30.0

    print("Output FPS:", output_fps)

    print()
    print("Choosing source rest frame...")
    if args.source_rest == "first-normal":
        source_rest = source_frames[0]
        print("Using first loaded normal MVNX frame as source rest pose.")
        print("This is preferred for Alex neutral-pose matching because Alex target_rest is also neutral/default.")
    else:
        print("Loading MVNX tpose rest frame...")
        rest_frames, rest_meta = read_mvnx_canonical_frames(
            mvnx_path=args.mvnx_path,
            frame_type="tpose",
            start_frame=0,
            stride=1,
            max_frames=1,
            canonicalize_axes=True,
        )

        if rest_frames:
            source_rest = rest_frames[0]
            print("Using MVNX tpose as source rest pose.")
            print("WARNING: this only makes sense if target_rest is also a robot T-pose.")
        else:
            source_rest = source_frames[0]
            print("WARNING: no MVNX tpose frame found. Using first normal frame as source rest pose.")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    solver = choose_solver(args.solver)
    print("Using solver:", solver)

    qpos0 = np.asarray(model.qpos0, dtype=float)
    if qpos0.shape[0] != 36:
        raise RuntimeError(f"Expected Alex nq=36, got {qpos0.shape[0]}")

    first_pelvis = np.asarray(source_frames[0]["pelvis"]["pos"], dtype=float)
    qpos0[0:3] = first_pelvis
    qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

    limits = [mink.ConfigurationLimit(model)]
    tasks = make_tasks(model, robot_cfg)
    posture_task = make_posture_task(model, args)
    if posture_task is not None:
        print()
        print(f"Using PostureTask regularization: cost={args.posture_cost}, neutral_blend={args.posture_neutral_blend}")

    if args.target_rest_mode == "aligned-source-rest":
        print()
        print("Aligning Alex target rest pose to source rest frame...")
        print("This solves a static Alex IK pose whose pelvis/head/feet/hands match the MVNX source rest.")
        print("Then that solved Alex pose becomes target_rest for local scaling.")

        rest_configuration = mink.Configuration(model, q=qpos0.copy())
        rest_target_by_role = set_task_targets(tasks, source_rest, robot_cfg)

        set_posture_target(posture_task, rest_configuration, qpos0.copy())

        qpos_rest, rest_score, rest_errors, rest_solved_positions = solve_frame(
            model=model,
            configuration=rest_configuration,
            tasks=tasks,
            target_by_role=rest_target_by_role,
            solver=solver,
            limits=limits,
            max_iter=max(args.max_ik_iter, 200),
            posture_task=posture_task,
        )

        print("Rest-alignment score:", f"{rest_score:.6f}")
        for role in IK_ROLES:
            print(f"  rest {role:10s} error: {rest_errors[role]:.6f} m")

        target_rest = robot_rest_frame_from_mujoco(model, qpos_rest, robot_cfg)
        configuration = mink.Configuration(model, q=qpos_rest.copy())

    else:
        print()
        print("Using raw Alex default qpos0 as target_rest.")
        print("WARNING: this can create low hand targets if Alex default hand pose differs from source rest.")
        qpos_rest = qpos0.copy()
        rest_score = None
        rest_errors = {}
        target_rest = robot_rest_frame_from_mujoco(model, qpos0, robot_cfg)
        configuration = mink.Configuration(model, q=qpos0.copy())

    qpos_traj = []
    rows = []
    target_positions = []
    solved_ik_positions = []

    morphology_scales = None
    if args.target_generation == "morphology-delta":
        morphology = compute_morphology_scales(
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

    for frame_idx, source_frame in enumerate(source_frames):
        if args.target_generation == "morphology-delta":
            if morphology_scales is None:
                raise RuntimeError("morphology_scales was not computed.")
            scaled_frame = make_morphology_delta_target_frame(
                source_frame=source_frame,
                source_rest=source_rest,
                target_rest=target_rest,
                scales=morphology_scales,
            )
        elif args.target_generation == "rest-delta":
            scaled_frame = make_rest_delta_target_frame(
                source_frame=source_frame,
                source_rest=source_rest,
                target_rest=target_rest,
                motion_scale=args.motion_scale,
            )
        else:
            scaled_frame = scale_frame_by_rest_pose(
                frame=source_frame,
                source_rest_frame=source_rest,
                target_rest_frame=target_rest,
            )

        target_by_role = set_task_targets(tasks, scaled_frame, robot_cfg)

        q_posture_target = blended_posture_target(
            np.asarray(configuration.data.qpos, dtype=float),
            qpos0,
            args.posture_neutral_blend,
        )
        set_posture_target(posture_task, configuration, q_posture_target)

        qpos, score, errors, solved_positions = solve_frame(
            model=model,
            configuration=configuration,
            tasks=tasks,
            target_by_role=target_by_role,
            solver=solver,
            limits=limits,
            max_iter=args.max_ik_iter,
            posture_task=posture_task,
        )

        qpos_traj.append(qpos)

        target_positions.append([
            np.asarray(target_by_role[role]["target_pos"], dtype=float)
            for role in IK_ROLES
        ])
        solved_ik_positions.append([
            np.asarray(solved_positions[role], dtype=float)
            for role in IK_ROLES
        ])

        row = {
            "frame": frame_idx,
            "source_mvnx_normal_frame": source_meta["matched_frame_indices"][frame_idx],
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

        if frame_idx % 10 == 0 or frame_idx == len(source_frames) - 1:
            print(
                f"frame {frame_idx:03d}: "
                f"score={score:.4f}, "
                f"pelvis={row['pelvis_error_m']:.3f}, "
                f"feet={row['mean_foot_error_m']:.3f}, "
                f"hands={row['mean_hand_error_m']:.3f}"
            )

    qpos_traj = np.asarray(qpos_traj, dtype=float)
    source_positions = frame_positions_array(source_frames, CANONICAL_BODY_NAMES)
    target_positions = np.asarray(target_positions, dtype=float)
    solved_ik_positions = np.asarray(solved_ik_positions, dtype=float)

    joint_delta = np.diff(qpos_traj[:, 7:], axis=0)
    root_delta = np.diff(qpos_traj[:, 0:3], axis=0)

    stem = args.mvnx_path.stem
    prefix = f"{stem}_alex_ik_start{args.start_frame}_stride{args.stride}_n{len(source_frames)}"

    npz_path = out_dir / f"{prefix}.npz"
    csv_path = out_dir / f"{prefix}_errors.csv"
    json_path = out_dir / f"{prefix}_summary.json"
    plot_path = out_dir / f"{prefix}_errors.png"

    summary = {
        "note": "Short MVNX-to-Alex IK test. Position-only IK.",
        "mvnx_path": str(args.mvnx_path),
        "solver": solver,
        "start_frame": args.start_frame,
        "stride": args.stride,
        "max_frames_requested": args.max_frames,
        "num_frames": len(source_frames),
        "output_fps": output_fps,
        "max_ik_iter": args.max_ik_iter,
        "posture_cost": args.posture_cost,
        "posture_neutral_blend": args.posture_neutral_blend,
        "posture_mode": args.posture_mode,
        "posture_wrist_cost": args.posture_wrist_cost,
        "posture_gripper_cost": args.posture_gripper_cost,
        "posture_ankle_cost": args.posture_ankle_cost,
        "posture_spine_neck_cost": args.posture_spine_neck_cost,
        "posture_arm_cost": args.posture_arm_cost,
        "posture_leg_cost": args.posture_leg_cost,
        "posture_base_cost": args.posture_base_cost,
        "recenter_xy": not args.no_recenter,
        "source_rest": args.source_rest,
        "target_rest_mode": args.target_rest_mode,
        "target_generation": args.target_generation,
        "motion_scale": args.motion_scale,
        "foot_cost_mult": args.foot_cost_mult,
        "head_cost_mult": args.head_cost_mult,
        "hand_cost_mult": args.hand_cost_mult,
        "pelvis_cost_mult": args.pelvis_cost_mult,
        "morphology_scales": morphology_scales,
        "heading_canonicalization": not args.no_heading_canonicalization,
        "rest_alignment_score": None if rest_score is None else float(rest_score),
        "rest_alignment_errors_m": rest_errors,
        "qpos_shape": list(qpos_traj.shape),
        "qpos_layout": robot_cfg["floating_base"]["qpos_layout"],
        "source_roles": list(CANONICAL_BODY_NAMES),
        "ik_roles": list(IK_ROLES),
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
        "source_meta": {
            "frame_rate": source_meta.get("frame_rate"),
            "matched_frame_indices_first": source_meta["matched_frame_indices"][:5],
            "matched_frame_indices_last": source_meta["matched_frame_indices"][-5:],
            "coordinate_assumption": source_meta.get("coordinate_assumption"),
        },
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
        ik_roles=np.asarray(IK_ROLES, dtype=object),
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
    plt.title("MVNX-to-Alex IK target errors")
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
        elif key != "source_meta":
            print(f"  {key}: {value}")

    print()
    print("Wrote:")
    print(" ", npz_path)
    print(" ", csv_path)
    print(" ", json_path)
    print(" ", plot_path)

    print()
    print("MVNX-to-Alex short clip IK completed.")

if __name__ == "__main__":
    main()

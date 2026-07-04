#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


MAIN_KEY = "toolbox_msgs.msg.dds.KinematicsToolboxOutputStatus"
INNER_KEY = "toolbox_msgs::msg::dds_::KinematicsToolboxOutputStatus_"

ISAAC_JOINT_NAMES_FULLBODY = [
    "LEFT_HIP_X",
    "LEFT_HIP_Z",
    "LEFT_HIP_Y",
    "LEFT_KNEE_Y",
    "LEFT_ANKLE_Y",
    "LEFT_ANKLE_X",
    "RIGHT_HIP_X",
    "RIGHT_HIP_Z",
    "RIGHT_HIP_Y",
    "RIGHT_KNEE_Y",
    "RIGHT_ANKLE_Y",
    "RIGHT_ANKLE_X",
    "SPINE_Z",
    "LEFT_SHOULDER_Y",
    "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Z",
    "LEFT_ELBOW_Y",
    "LEFT_WRIST_Z",
    "LEFT_WRIST_X",
    "LEFT_GRIPPER_Z",
    "NECK_Z",
    "NECK_Y",
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
    "RIGHT_GRIPPER_Z",
]


def vec3(xyz):
    return {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
    }


def quat_xyzw_from_wxyz(q):
    # MuJoCo free joint qpos is [w, x, y, z] after root xyz.
    return {
        "x": float(q[1]),
        "y": float(q[2]),
        "z": float(q[3]),
        "w": float(q[0]),
    }


def load_mujoco_joint_order(model_path: Path) -> list[str]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    names = []
    for j in range(model.njnt):
        qadr = int(model.jnt_qposadr[j])
        if qadr >= 7:
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    return names


def finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    v = np.zeros_like(x, dtype=np.float32)
    if len(x) <= 1:
        return v
    v[1:-1] = 0.5 * (x[2:] - x[:-2]) * fps
    v[0] = (x[1] - x[0]) * fps
    v[-1] = (x[-1] - x[-2]) * fps
    return v


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float32)


def quat_to_angvel_wxyz(q: np.ndarray, fps: float) -> np.ndarray:
    # Simple adjacent-frame estimate in local-ish convention.
    # Good enough for JSON replay/mimic bootstrap; Isaac can recompute body velocities later.
    w = np.zeros((len(q), 3), dtype=np.float32)
    if len(q) <= 1:
        return w

    qn = q / np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
    for i in range(len(qn) - 1):
        dq = quat_mul_wxyz(qn[i + 1], quat_conj_wxyz(qn[i]))
        if dq[0] < 0:
            dq = -dq
        angle = 2.0 * np.arctan2(np.linalg.norm(dq[1:]), max(float(dq[0]), 1e-8))
        axis = dq[1:] / max(np.linalg.norm(dq[1:]), 1e-8)
        w[i] = axis * angle * fps
    w[-1] = w[-2]
    return w


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    q0 = q0 / np.linalg.norm(q0).clip(min=1e-8)
    q1 = q1 / np.linalg.norm(q1).clip(min=1e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return (out / np.linalg.norm(out)).astype(np.float32)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return (s0 * q0 + s1 * q1).astype(np.float32)


def resample_contacts_bool(flags: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if flags.ndim == 1:
        flags = flags.reshape(-1, 1)
    src_len = flags.shape[0]
    if src_len == 0:
        return flags.copy()
    src_times = np.arange(src_len, dtype=np.float64) / src_fps
    dst_len = int(round((src_len - 1) * dst_fps / src_fps)) + 1
    dst_times = np.arange(dst_len, dtype=np.float64) / dst_fps
    indices = np.searchsorted(src_times, dst_times, side="right") - 1
    np.clip(indices, 0, src_len - 1, out=indices)
    return flags[indices]


def resample_qpos(qpos: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if src_fps == dst_fps:
        return qpos.copy()
    src_len = qpos.shape[0]
    if src_len == 0:
        return qpos.copy()
    src_times = np.arange(src_len, dtype=np.float64) / src_fps
    dst_len = int(round((src_len - 1) * dst_fps / src_fps)) + 1
    dst_times = np.arange(dst_len, dtype=np.float64) / dst_fps

    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]
    joints = qpos[:, 7:36]

    dst_root_pos = np.empty((dst_len, 3), dtype=np.float32)
    dst_root_quat = np.empty((dst_len, 4), dtype=np.float32)
    dst_joints = np.empty((dst_len, 29), dtype=np.float32)

    for d in range(3):
        dst_root_pos[:, d] = np.interp(dst_times, src_times, root_pos[:, d])
    for j in range(29):
        dst_joints[:, j] = np.interp(dst_times, src_times, joints[:, j])

    src_indices = np.searchsorted(src_times, dst_times, side="right")
    src_indices = np.clip(src_indices, 1, src_len - 1)
    prev_indices = src_indices - 1
    alpha = (dst_times - src_times[prev_indices]) / np.maximum(src_times[src_indices] - src_times[prev_indices], 1e-8)

    for i in range(dst_len):
        dst_root_quat[i] = quat_slerp_wxyz(root_quat[prev_indices[i]], root_quat[src_indices[i]], float(alpha[i]))

    dst_qpos = np.empty((dst_len, 36), dtype=np.float32)
    dst_qpos[:, 0:3] = dst_root_pos
    dst_qpos[:, 3:7] = dst_root_quat
    dst_qpos[:, 7:36] = dst_joints
    return dst_qpos


def infer_foot_contacts_from_sites(
    qpos: np.ndarray,
    model_path: Path,
    fps: float,
    height_threshold_m: float,
    speed_threshold_mps: float,
) -> np.ndarray:
    """Infer [left,right] foot contacts from Alex sole corner height + sole speed."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    left_center = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_left_sole_contact_site")
    right_center = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_right_sole_contact_site")

    left_corners = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_left_sole_corner_toe_body_left_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_left_sole_corner_toe_body_right_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_left_sole_corner_heel_body_left_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_left_sole_corner_heel_body_right_site"),
    ]
    right_corners = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_right_sole_corner_toe_body_left_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_right_sole_corner_toe_body_right_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_right_sole_corner_heel_body_left_site"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "alex_right_sole_corner_heel_body_right_site"),
    ]

    needed = [left_center, right_center] + left_corners + right_corners
    if any(i < 0 for i in needed):
        raise RuntimeError("Missing one or more Alex sole/contact sites in model.")

    T = qpos.shape[0]
    left_center_xyz = np.zeros((T, 3), dtype=np.float32)
    right_center_xyz = np.zeros((T, 3), dtype=np.float32)
    left_corner_z = np.zeros((T, 4), dtype=np.float32)
    right_corner_z = np.zeros((T, 4), dtype=np.float32)

    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        left_center_xyz[t] = data.site_xpos[left_center]
        right_center_xyz[t] = data.site_xpos[right_center]
        left_corner_z[t] = [data.site_xpos[sid, 2] for sid in left_corners]
        right_corner_z[t] = [data.site_xpos[sid, 2] for sid in right_corners]

    all_corner_z = np.concatenate([left_corner_z.reshape(-1), right_corner_z.reshape(-1)])
    floor_z = float(np.percentile(all_corner_z, 1.0))

    left_height = left_corner_z.min(axis=1) - floor_z
    right_height = right_corner_z.min(axis=1) - floor_z

    left_vel = finite_diff(left_center_xyz, fps)
    right_vel = finite_diff(right_center_xyz, fps)
    left_speed = np.linalg.norm(left_vel, axis=1)
    right_speed = np.linalg.norm(right_vel, axis=1)

    contacts = np.zeros((T, 2), dtype=bool)
    contacts[:, 0] = (left_height <= height_threshold_m) & (left_speed <= speed_threshold_mps)
    contacts[:, 1] = (right_height <= height_threshold_m) & (right_speed <= speed_threshold_mps)

    # Fill tiny one-frame holes to avoid flickery contact labels.
    for foot in range(2):
        c = contacts[:, foot]
        for i in range(1, T - 1):
            if (not c[i]) and c[i - 1] and c[i + 1]:
                c[i] = True
        contacts[:, foot] = c

    print("contact inference:")
    print("  floor_z:", floor_z)
    print("  left contact frames:", int(contacts[:, 0].sum()), "/", T)
    print("  right contact frames:", int(contacts[:, 1].sum()), "/", T)
    print("  height threshold m:", height_threshold_m)
    print("  speed threshold m/s:", speed_threshold_mps)

    return contacts

def load_contact_effector_flags(z: np.lib.npyio.NpzFile) -> dict[str, np.ndarray] | None:
    if "contact_effector_names" not in z or "contact_flags" not in z:
        return None

    names = np.asarray(z["contact_effector_names"]).reshape(-1).astype(str)
    flags = np.asarray(z["contact_flags"])
    if flags.ndim == 1:
        flags = flags.reshape(-1, 1)

    mapping: dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        key = name.strip().lower().replace(" ", "_")
        if key in {"left_foot", "right_foot", "left_hand", "right_hand"}:
            mapping[key] = flags[:, idx].astype(bool)
    if not mapping:
        return None
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_npz", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("assets/alex/alex_floating_base_with_sites.xml"))
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--com-offset", type=float, default=0.5)
    ap.add_argument("--left-contact", type=int, default=1)
    ap.add_argument("--right-contact", type=int, default=1)
    ap.add_argument("--left-hand-contact", type=int, default=0)
    ap.add_argument("--right-hand-contact", type=int, default=0)
    ap.add_argument(
        "--contact-mode",
        choices=["constant", "infer-sites"],
        default="constant",
        help="constant uses --left-contact/--right-contact; infer-sites infers per-frame foot contact from MuJoCo sole site height and speed.",
    )
    ap.add_argument("--contact-height-threshold-m", type=float, default=0.04)
    ap.add_argument("--contact-speed-threshold-mps", type=float, default=0.20)
    ap.add_argument("--solution-quality", type=float, default=0.0,
                    help="Constant solution_quality to write into each frame.")
    ap.add_argument("--solution-quality-random", action="store_true",
                    help="If set, generate a random solution_quality per frame instead of a constant value.")
    args = ap.parse_args()

    z = np.load(args.input_npz, allow_pickle=True)
    if "qpos" not in z:
        raise KeyError(f"{args.input_npz} does not contain qpos. Keys: {z.files}")

    qpos = np.asarray(z["qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T, 36), got {qpos.shape}")

    # Real-time source fps. The grounded NPZ stores `fps` = the CAPTURE rate
    # (e.g. 120), but the solved qpos is strided (see source_frame_ids), so the
    # true real-time rate is capture_fps / stride (e.g. 120/4 = 30 Hz). Using the
    # raw capture fps mislabels the motion and plays it back stride-times too fast
    # (and inflates all velocities by the same factor).
    src_fps = None
    if "output_fps" in z:
        src_fps = float(np.asarray(z["output_fps"]).reshape(-1)[0])
    elif "fps" in z:
        capture_fps = float(np.asarray(z["fps"]).reshape(-1)[0])
        stride = 1
        if "source_frame_ids" in z:
            sfi = np.asarray(z["source_frame_ids"]).reshape(-1)
            if sfi.size > 1:
                stride = max(int(np.round(np.median(np.diff(sfi)))), 1)
        src_fps = capture_fps / stride
        if stride != 1:
            print(f"Real-time src fps = capture {capture_fps:.2f} / stride {stride} = {src_fps:.2f} Hz")
    else:
        src_fps = 30.0

    dst_fps = float(args.fps) if args.fps is not None else src_fps
    resample = abs(dst_fps - src_fps) > 1e-6

    if resample:
        print(f"Resampling from {src_fps:.2f} Hz to {dst_fps:.2f} Hz")
        qpos = resample_qpos(qpos, src_fps, dst_fps)

    mj_names = load_mujoco_joint_order(args.model)
    if len(mj_names) != 29:
        raise ValueError(f"Expected 29 MuJoCo non-root joints, got {len(mj_names)}: {mj_names}")

    mj_name_to_idx = {name: i for i, name in enumerate(mj_names)}
    reorder = [mj_name_to_idx[name] for name in ISAAC_JOINT_NAMES_FULLBODY]

    root_pos = qpos[:, 0:3]
    root_quat_wxyz = qpos[:, 3:7]
    mj_joint_pos = qpos[:, 7:36]
    isaac_joint_pos = mj_joint_pos[:, reorder]

    joint_vel = finite_diff(isaac_joint_pos, dst_fps)
    root_lin_vel = finite_diff(root_pos, dst_fps)
    root_ang_vel = quat_to_angvel_wxyz(root_quat_wxyz, dst_fps)

    contact_effector_map = load_contact_effector_flags(z)
    if contact_effector_map is not None:
        print("Loaded contact effector flags from NPZ:", sorted(contact_effector_map.keys()))

    if args.contact_mode == "infer-sites":
        foot_contacts = infer_foot_contacts_from_sites(
            qpos=qpos,
            model_path=args.model,
            fps=dst_fps,
            height_threshold_m=args.contact_height_threshold_m,
            speed_threshold_mps=args.contact_speed_threshold_mps,
        )
        left_foot = foot_contacts[:, 0]
        right_foot = foot_contacts[:, 1]
    else:
        left_foot = np.full((qpos.shape[0],), bool(args.left_contact), dtype=bool)
        right_foot = np.full((qpos.shape[0],), bool(args.right_contact), dtype=bool)

    if contact_effector_map is not None:
        left_foot = contact_effector_map.get("left_foot", left_foot)
        right_foot = contact_effector_map.get("right_foot", right_foot)
        if resample:
            left_foot = resample_contacts_bool(left_foot, src_fps, dst_fps).reshape(-1)
            right_foot = resample_contacts_bool(right_foot, src_fps, dst_fps).reshape(-1)

    left_hand = np.full((qpos.shape[0],), bool(args.left_hand_contact), dtype=bool)
    right_hand = np.full((qpos.shape[0],), bool(args.right_hand_contact), dtype=bool)
    if contact_effector_map is not None:
        left_hand = contact_effector_map.get("left_hand", left_hand)
        right_hand = contact_effector_map.get("right_hand", right_hand)
        if resample:
            left_hand = resample_contacts_bool(left_hand, src_fps, dst_fps).reshape(-1)
            right_hand = resample_contacts_bool(right_hand, src_fps, dst_fps).reshape(-1)

    if args.solution_quality_random:
        rng = np.random.default_rng(0)
        solution_quality = rng.random(qpos.shape[0]).astype(np.float32)
    else:
        solution_quality = np.full((qpos.shape[0],), float(args.solution_quality), dtype=np.float32)

    timestamps_ms = np.arange(qpos.shape[0], dtype=np.float64) * (1000.0 / dst_fps)

    messages = []
    for i in range(qpos.shape[0]):
        status = {
            "sequence_id": int(i),
            "current_toolbox_state": 3,
            "joint_name_hash": -1087810655,
            "desired_joint_angles": [float(x) for x in isaac_joint_pos[i]],
            "desired_root_position": vec3(root_pos[i]),
            "desired_root_orientation": quat_xyzw_from_wxyz(root_quat_wxyz[i]),
            "desired_joint_velocities": [float(x) for x in joint_vel[i]],
            "desired_root_linear_velocity": vec3(root_lin_vel[i]),
            "desired_root_angular_velocity": vec3(root_ang_vel[i]),
            "support_region": [],
            "desired_joint_velocities_publishing_period": [float(x) for x in joint_vel[i]],
            "desired_root_linear_velocity_publishing_period": vec3(root_lin_vel[i]),
            "desired_root_angular_velocity_publishing_period": vec3(root_ang_vel[i]),
            "desired_torso_position": vec3(root_pos[i]),
            "desired_torso_orientation": quat_xyzw_from_wxyz(root_quat_wxyz[i]),
            "com_offset": float(args.com_offset),
            "left_foot_in_contact": bool(left_foot[i]),
            "right_foot_in_contact": bool(right_foot[i]),
            "left_hand_in_contact": bool(left_hand[i]),
            "right_hand_in_contact": bool(right_hand[i]),
            "solution_quality": float(solution_quality[i]),
        }
        messages.append(json.dumps({INNER_KEY: status}, separators=(",", ":")))

    out = {
        MAIN_KEY: {
            "timestamps": [float(x) for x in timestamps_ms],
            "messages": messages,
        }
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print("wrote:", args.out)
    print("frames:", qpos.shape[0])
    print("fps:", dst_fps)
    print("MuJoCo order:", mj_names)
    print("Isaac order:", ISAAC_JOINT_NAMES_FULLBODY)
    print("reorder:", reorder)


if __name__ == "__main__":
    main()

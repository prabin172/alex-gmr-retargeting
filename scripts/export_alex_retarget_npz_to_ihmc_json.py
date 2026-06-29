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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_npz", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("assets/alex/alex_floating_base_with_sites.xml"))
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--com-offset", type=float, default=0.5)
    ap.add_argument("--left-contact", type=int, default=1)
    ap.add_argument("--right-contact", type=int, default=1)
    ap.add_argument(
        "--contact-mode",
        choices=["constant", "infer-sites"],
        default="constant",
        help="constant uses --left-contact/--right-contact; infer-sites infers per-frame foot contact from MuJoCo sole site height and speed.",
    )
    ap.add_argument("--contact-height-threshold-m", type=float, default=0.04)
    ap.add_argument("--contact-speed-threshold-mps", type=float, default=0.20)
    args = ap.parse_args()

    z = np.load(args.input_npz, allow_pickle=True)
    if "qpos" not in z:
        raise KeyError(f"{args.input_npz} does not contain qpos. Keys: {z.files}")

    qpos = np.asarray(z["qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"Expected qpos shape (T, 36), got {qpos.shape}")

    fps = float(args.fps)
    if args.fps is None:
        if "output_fps" in z:
            fps = float(np.asarray(z["output_fps"]).reshape(-1)[0])
        elif "fps" in z:
            fps = float(np.asarray(z["fps"]).reshape(-1)[0])
        else:
            fps = 30.0

    mj_names = load_mujoco_joint_order(args.model)
    if len(mj_names) != 29:
        raise ValueError(f"Expected 29 MuJoCo non-root joints, got {len(mj_names)}: {mj_names}")

    mj_name_to_idx = {name: i for i, name in enumerate(mj_names)}
    reorder = [mj_name_to_idx[name] for name in ISAAC_JOINT_NAMES_FULLBODY]

    root_pos = qpos[:, 0:3]
    root_quat_wxyz = qpos[:, 3:7]
    mj_joint_pos = qpos[:, 7:36]
    isaac_joint_pos = mj_joint_pos[:, reorder]

    joint_vel = finite_diff(isaac_joint_pos, fps)
    root_lin_vel = finite_diff(root_pos, fps)
    root_ang_vel = quat_to_angvel_wxyz(root_quat_wxyz, fps)

    if args.contact_mode == "infer-sites":
        foot_contacts = infer_foot_contacts_from_sites(
            qpos=qpos,
            model_path=args.model,
            fps=fps,
            height_threshold_m=args.contact_height_threshold_m,
            speed_threshold_mps=args.contact_speed_threshold_mps,
        )
    else:
        foot_contacts = np.zeros((qpos.shape[0], 2), dtype=bool)
        foot_contacts[:, 0] = bool(args.left_contact)
        foot_contacts[:, 1] = bool(args.right_contact)

    timestamps_ms = np.arange(qpos.shape[0], dtype=np.float64) * (1000.0 / fps)

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
            "left_foot_in_contact": bool(foot_contacts[i, 0]),
            "right_foot_in_contact": bool(foot_contacts[i, 1]),
            "solution_quality": 0.0,
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
    print("fps:", fps)
    print("MuJoCo order:", mj_names)
    print("Isaac order:", ISAAC_JOINT_NAMES_FULLBODY)
    print("reorder:", reorder)


if __name__ == "__main__":
    main()

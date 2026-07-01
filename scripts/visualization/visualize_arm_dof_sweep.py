#!/usr/bin/env python3
"""Visualize each left-arm DOF of Alex by sweeping it through its range one at a time.

Renders Alex in T-pose (all joints zero), then for each of the 7 left-arm joints
sweeps it from min to max and back, with an annotation overlay showing the joint
name and current angle. All other joints stay at zero.

Usage:
    MUJOCO_GL=egl conda run -n gmr python scripts/visualization/visualize_arm_dof_sweep.py \\
        --out-mp4 outputs/renders/arm_dof_sweep.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Right-arm joints in kinematic order (shoulder → gripper)
# ---------------------------------------------------------------------------
ARM_JOINTS = [
    ("RIGHT_SHOULDER_Y", "Shoulder Pitch (Y)"),
    ("RIGHT_SHOULDER_X", "Shoulder Roll (X)"),
    ("RIGHT_SHOULDER_Z", "Shoulder Yaw (Z)"),
    ("RIGHT_ELBOW_Y",    "Elbow Pitch (Y)"),
    ("RIGHT_WRIST_Z",    "Wrist Yaw (Z)"),
    ("RIGHT_WRIST_X",    "Wrist Roll (X)"),
    ("RIGHT_GRIPPER_Z",  "Gripper Yaw (Z)"),
]

# Highlight colour per joint (cycling through a palette)
JOINT_COLORS = [
    (255, 100, 100),   # red
    (255, 180,  60),   # orange
    (240, 240,  60),   # yellow
    ( 80, 220,  80),   # green
    ( 80, 200, 255),   # cyan
    (160, 100, 255),   # purple
    (255, 100, 200),   # pink
]

# Base pose: root pitched forward + legs compensated (from test_deadlift_pose.py)
TORSO_PITCH = 0.70   # rad forward lean (~40°)

FPS         = 30
HOLD_FRAMES = 20   # frames held at min/max/zero
SWEEP_STEPS = 60   # frames for one-way sweep (min→max)


def _base_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = np.zeros(model.nq)
    # Pitch root forward
    hw = TORSO_PITCH / 2
    q[2] = 2.0                  # start high; will be grounded after fwd kinematics
    q[3] = np.cos(hw)           # qw
    q[4] = 0.0                  # qx
    q[5] = np.sin(hw)           # qy — forward pitch
    q[6] = 0.0                  # qz

    def _set(name, val):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[model.jnt_qposadr[jid]] = val

    # Legs: counter-rotate to keep femurs roughly vertical
    _set("LEFT_HIP_Y",   -0.50); _set("RIGHT_HIP_Y",   -0.50)
    _set("LEFT_KNEE_Y",   0.30); _set("RIGHT_KNEE_Y",   0.30)
    _set("LEFT_ANKLE_Y", -0.20); _set("RIGHT_ANKLE_Y", -0.20)

    # Arms: shoulder pitch to cancel torso lean → arms hang down
    _set("LEFT_SHOULDER_Y",  -TORSO_PITCH * 0.9)
    _set("RIGHT_SHOULDER_Y", -TORSO_PITCH * 0.9)

    return q


def _ground_qpos(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    """Shift root Z so the lowest geom sits just above Z=0."""
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    min_z = np.min(data.geom_xpos[:, 2])
    q = q.copy()
    q[2] -= (min_z - 0.01)
    return q


def _add_overlay(img: np.ndarray, joint_label: str, angle_deg: float,
                 joint_idx: int, color: tuple) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    W, H = pil.size

    try:
        font_big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font_big   = ImageFont.load_default()
        font_small = font_big

    # Semi-transparent banner at top
    banner_h = 80
    overlay = Image.new("RGBA", (W, banner_h), (0, 0, 0, 160))
    pil_rgba = pil.convert("RGBA")
    pil_rgba.paste(overlay, (0, 0), overlay)
    pil = pil_rgba.convert("RGB")
    draw = ImageDraw.Draw(pil)

    # DOF index badge  e.g.  "DOF 1/7"
    badge = f"DOF {joint_idx + 1} / {len(ARM_JOINTS)}"
    draw.text((11, 11), badge, font=font_small, fill=(0, 0, 0))
    draw.text((10, 10), badge, font=font_small, fill=(180, 180, 180))

    # Joint name in highlight colour
    draw.text((11, 37), joint_label, font=font_big, fill=(0, 0, 0))
    draw.text((10, 36), joint_label, font=font_big, fill=color)

    # Angle readout at top-right
    angle_str = f"{angle_deg:+.1f}°"
    bbox = draw.textbbox((0, 0), angle_str, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.text((W - tw - 11, 37), angle_str, font=font_big, fill=(0, 0, 0))
    draw.text((W - tw - 12, 36), angle_str, font=font_big, fill=(220, 220, 220))

    # Sweep progress bar along the bottom
    bar_y = H - 6
    bar_h = 5
    lo, hi = -1.0, 1.0   # normalised range passed separately; use angle_deg for now
    draw.rectangle([0, bar_y, W, bar_y + bar_h], fill=(40, 40, 40))

    return np.array(pil)


def _add_progress_bar(img: np.ndarray, frac: float, color: tuple) -> np.ndarray:
    """Draw a thin coloured progress bar at the bottom of the image."""
    pil  = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    W, H = pil.size
    bar_h = 5
    draw.rectangle([0, H - bar_h, int(W * frac), H], fill=color)
    return np.array(pil)


def _render_frame(model, data, renderer, cam, qpos: np.ndarray,
                  joint_label: str, angle_deg: float,
                  joint_idx: int, color: tuple, sweep_frac: float) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    img = _add_overlay(img, joint_label, angle_deg, joint_idx, color)
    img = _add_progress_bar(img, sweep_frac, color)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",   type=Path,
                    default=Path("assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml"))
    ap.add_argument("--out-mp4", type=Path,
                    default=Path("outputs/renders/arm_dof_sweep.mp4"))
    ap.add_argument("--width",   type=int, default=640)
    ap.add_argument("--height",  type=int, default=480)
    args = ap.parse_args()

    model    = mujoco.MjModel.from_xml_path(str(args.model))
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    # Resolve joint name → qpos index
    joint_info = []
    for jname, label in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint '{jname}' not found in model")
        qadr = model.jnt_qposadr[jid]
        lo   = float(model.jnt_range[jid, 0])
        hi   = float(model.jnt_range[jid, 1])
        joint_info.append((jname, label, qadr, lo, hi))

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    tpose = _ground_qpos(model, _base_qpos(model))

    # Fixed camera: front-left view to see the right arm clearly
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    tmp = mujoco.MjData(model); tmp.qpos[:] = tpose; mujoco.mj_forward(model, tmp)
    cam.lookat[:] = tmp.qpos[:3] + np.array([0.05, 0.0, 0.25])
    cam.distance  = 2.4
    cam.azimuth   = 60.0
    cam.elevation = -10.0

    total_frames_est = len(ARM_JOINTS) * (HOLD_FRAMES + SWEEP_STEPS + HOLD_FRAMES + SWEEP_STEPS + HOLD_FRAMES)
    print(f"Estimated frames: {total_frames_est}  (~{total_frames_est / FPS:.1f} s)")
    print(f"Output: {args.out_mp4}")

    with imageio.get_writer(str(args.out_mp4), fps=FPS, codec="libx264", quality=8) as writer:
        for j_idx, (jname, label, qadr, lo, hi) in enumerate(joint_info):
            color = JOINT_COLORS[j_idx % len(JOINT_COLORS)]
            print(f"  [{j_idx + 1}/{len(ARM_JOINTS)}] {label}  range=[{np.rad2deg(lo):.1f}°, {np.rad2deg(hi):.1f}°]")

            total_seg = HOLD_FRAMES + SWEEP_STEPS + HOLD_FRAMES + SWEEP_STEPS + HOLD_FRAMES
            frame_n   = 0

            def emit(angle_rad: float, frac: float) -> None:
                nonlocal frame_n
                q = tpose.copy()
                q[qadr] = angle_rad
                img = _render_frame(model, data, renderer, cam, q,
                                    label, np.rad2deg(angle_rad),
                                    j_idx, color, frac)
                writer.append_data(img)
                frame_n += 1

            # Hold at zero
            for _ in range(HOLD_FRAMES):
                emit(0.0, frame_n / total_seg)

            # Sweep 0 → max (or min if max is small)
            # Pick the direction with larger absolute range
            if abs(hi) >= abs(lo):
                first_target, second_target = hi, lo
            else:
                first_target, second_target = lo, hi

            for k in range(SWEEP_STEPS):
                emit(first_target * (k + 1) / SWEEP_STEPS, frame_n / total_seg)

            # Hold at extreme
            for _ in range(HOLD_FRAMES):
                emit(first_target, frame_n / total_seg)

            # Sweep back through zero to other extreme
            full_swing = SWEEP_STEPS * 2
            for k in range(full_swing):
                t = (k + 1) / full_swing
                angle = first_target + t * (second_target - first_target)
                emit(angle, frame_n / total_seg)

            # Hold at other extreme
            for _ in range(HOLD_FRAMES):
                emit(second_target, frame_n / total_seg)

            # Return to zero
            for k in range(SWEEP_STEPS):
                t = (k + 1) / SWEEP_STEPS
                angle = second_target * (1.0 - t)
                emit(angle, frame_n / total_seg)

    renderer.close()
    print("Done:", args.out_mp4)


if __name__ == "__main__":
    main()

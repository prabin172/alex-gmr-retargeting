#!/usr/bin/env python3
"""Animate the right arm from the deadlift base pose to a target joint configuration.

Interpolates RIGHT_SHOULDER_Z, RIGHT_ELBOW_Y, RIGHT_WRIST_Z to target angles
over 3 seconds, holding briefly at start and end.

Usage:
    MUJOCO_GL=egl conda run -n gmr python scripts/visualization/visualize_arm_pose_transition.py \\
        --out-mp4 outputs/renders/arm_pose_transition.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS          = 30
HOLD_FRAMES  = 30   # 1 s hold at start and end
TRANS_FRAMES = 90   # 3 s transition

# Base pose parameters (same as arm_dof_sweep)
TORSO_PITCH = 0.70

# Target angles (absolute, in degrees → converted below)
TARGET_DEG = {
    "RIGHT_SHOULDER_Z": +20.0,
    "RIGHT_ELBOW_Y":    -30.0,
    "RIGHT_WRIST_Z":    +90.0,
    "RIGHT_WRIST_X":    -80.0,
}


def _base_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = np.zeros(model.nq)
    hw = TORSO_PITCH / 2
    q[2] = 2.0
    q[3] = np.cos(hw); q[5] = np.sin(hw)

    def _set(name, val):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[model.jnt_qposadr[jid]] = val

    _set("LEFT_HIP_Y",   -0.50); _set("RIGHT_HIP_Y",   -0.50)
    _set("LEFT_KNEE_Y",   0.30); _set("RIGHT_KNEE_Y",   0.30)
    _set("LEFT_ANKLE_Y", -0.20); _set("RIGHT_ANKLE_Y", -0.20)
    _set("LEFT_SHOULDER_Y",  -TORSO_PITCH * 0.9)
    _set("RIGHT_SHOULDER_Y", -TORSO_PITCH * 0.9)

    # Ground via geoms
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    q[2] -= (np.min(data.geom_xpos[:, 2]) - 0.01)
    return q


def _add_overlay(img: np.ndarray, t: float, joint_angles: dict[str, float]) -> np.ndarray:
    pil  = Image.fromarray(img)
    W, H = pil.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        fsm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = fsm = ImageFont.load_default()

    banner = Image.new("RGBA", (W, 120), (0, 0, 0, 160))
    rgba   = pil.convert("RGBA")
    rgba.paste(banner, (0, 0), banner)
    pil  = rgba.convert("RGB")
    draw = ImageDraw.Draw(pil)

    draw.text((10, 6),  "Right arm pose transition", font=font, fill=(220, 220, 100))
    draw.text((10, 32), f"Shoulder Yaw:  {joint_angles['RIGHT_SHOULDER_Z']:+.1f}°", font=fsm, fill=(80, 200, 255))
    draw.text((10, 52), f"Elbow Pitch:   {joint_angles['RIGHT_ELBOW_Y']:+.1f}°",    font=fsm, fill=(80, 200, 255))
    draw.text((10, 72), f"Wrist Yaw:     {joint_angles['RIGHT_WRIST_Z']:+.1f}°",    font=fsm, fill=(80, 200, 255))
    draw.text((10, 92), f"Wrist Roll:    {joint_angles['RIGHT_WRIST_X']:+.1f}°",    font=fsm, fill=(80, 200, 255))

    # Progress bar
    frac = np.clip(t, 0.0, 1.0)
    draw.rectangle([0, H - 5, W, H], fill=(40, 40, 40))
    draw.rectangle([0, H - 5, int(W * frac), H], fill=(80, 200, 255))

    return np.array(pil)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",   type=Path,
                    default=Path("assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml"))
    ap.add_argument("--out-mp4", type=Path,
                    default=Path("outputs/renders/arm_pose_transition.mp4"))
    args = ap.parse_args()

    model    = mujoco.MjModel.from_xml_path(str(args.model))
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    base = _base_qpos(model)

    # Resolve target joint qpos addresses and convert deg → rad
    targets = {}
    for jname, deg in TARGET_DEG.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        targets[jname] = (model.jnt_qposadr[jid], np.deg2rad(deg))
        print(f"  {jname}: 0° → {deg:+.1f}°")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    tmp = mujoco.MjData(model); tmp.qpos[:] = base; mujoco.mj_forward(model, tmp)
    cam.lookat[:] = tmp.qpos[:3] + np.array([0.05, 0.0, 0.25])
    cam.distance  = 2.4
    cam.azimuth   = 150.0
    cam.elevation = -10.0

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    total = HOLD_FRAMES + TRANS_FRAMES + HOLD_FRAMES
    print(f"Frames: {total}  ({total / FPS:.1f} s)  →  {args.out_mp4}")

    def render(q: np.ndarray, t: float) -> None:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        angles = {jn: np.rad2deg(q[adr]) for jn, (adr, _) in targets.items()}
        writer.append_data(_add_overlay(img, t, angles))

    with imageio.get_writer(str(args.out_mp4), fps=FPS, codec="libx264", quality=8) as writer:
        # Hold at base
        for _ in range(HOLD_FRAMES):
            render(base.copy(), 0.0)

        # Smooth interpolation (cosine ease-in-out)
        for k in range(TRANS_FRAMES):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (k + 1) / TRANS_FRAMES))
            q = base.copy()
            for jname, (adr, target_rad) in targets.items():
                q[adr] = alpha * target_rad
            render(q, alpha)

        # Hold at target
        q_end = base.copy()
        for jname, (adr, target_rad) in targets.items():
            q_end[adr] = target_rad
        for _ in range(HOLD_FRAMES):
            render(q_end, 1.0)

    renderer.close()
    print("Done:", args.out_mp4)


if __name__ == "__main__":
    main()

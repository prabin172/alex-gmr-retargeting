#!/usr/bin/env python3
"""Visualize each left-leg DOF of Alex by sweeping it through its range one at a time.

Alex stands upright (all joints zero, feet grounded). For each of the 6 left-leg
joints the joint is swept from min to max and back with an annotation overlay.

Usage:
    MUJOCO_GL=egl conda run -n gmr python scripts/visualization/visualize_leg_dof_sweep.py \\
        --out-mp4 outputs/renders/leg_dof_sweep.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Left-leg joints in kinematic order (hip → ankle)
# ---------------------------------------------------------------------------
LEG_JOINTS = [
    ("LEFT_HIP_X", "Hip Abduction/Adduction (X)"),
    ("LEFT_HIP_Z", "Hip Internal/External Rot (Z)"),
    ("LEFT_HIP_Y", "Hip Flexion/Extension (Y)"),
    ("LEFT_KNEE_Y",   "Knee Flexion (Y)"),
    ("LEFT_ANKLE_Y",  "Ankle Pitch (Y)"),
    ("LEFT_ANKLE_X",  "Ankle Roll (X)"),
]

JOINT_COLORS = [
    (255, 100, 100),
    (255, 180,  60),
    (240, 240,  60),
    ( 80, 220,  80),
    ( 80, 200, 255),
    (160, 100, 255),
]

FPS         = 30
HOLD_FRAMES = 20
SWEEP_STEPS = 60


def _base_qpos(model: mujoco.MjModel) -> np.ndarray:
    """Standing straight — root upright, all joints zero, feet auto-grounded."""
    q = np.zeros(model.nq)
    q[2] = 2.0      # start high
    q[3] = 1.0      # qw upright
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    min_z = np.min(data.geom_xpos[:, 2])
    q[2] -= (min_z - 0.01)
    return q


def _add_overlay(img: np.ndarray, label: str, angle_deg: float,
                 joint_idx: int, color: tuple) -> np.ndarray:
    pil = Image.fromarray(img)
    W, H = pil.size

    try:
        font_big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    except OSError:
        font_big = font_small = ImageFont.load_default()

    banner = Image.new("RGBA", (W, 76), (0, 0, 0, 160))
    pil_rgba = pil.convert("RGBA")
    pil_rgba.paste(banner, (0, 0), banner)
    pil = pil_rgba.convert("RGB")
    draw = ImageDraw.Draw(pil)

    badge = f"DOF {joint_idx + 1} / {len(LEG_JOINTS)}"
    draw.text((11, 10), badge, font=font_small, fill=(0, 0, 0))
    draw.text((10, 10), badge, font=font_small, fill=(180, 180, 180))

    draw.text((11, 34), label, font=font_big, fill=(0, 0, 0))
    draw.text((10, 33), label, font=font_big, fill=color)

    angle_str = f"{angle_deg:+.1f}°"
    bbox = draw.textbbox((0, 0), angle_str, font=font_big)
    tw = bbox[2] - bbox[0]
    draw.text((W - tw - 11, 34), angle_str, font=font_big, fill=(0, 0, 0))
    draw.text((W - tw - 12, 33), angle_str, font=font_big, fill=(220, 220, 220))

    return np.array(pil)


def _add_progress_bar(img: np.ndarray, frac: float, color: tuple) -> np.ndarray:
    pil  = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    W, H = pil.size
    bar_h = 5
    draw.rectangle([0, H - bar_h, W, H], fill=(40, 40, 40))
    draw.rectangle([0, H - bar_h, int(W * frac), H], fill=color)
    return np.array(pil)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",   type=Path,
                    default=Path("assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml"))
    ap.add_argument("--out-mp4", type=Path,
                    default=Path("outputs/renders/leg_dof_sweep.mp4"))
    ap.add_argument("--width",   type=int, default=640)
    ap.add_argument("--height",  type=int, default=480)
    args = ap.parse_args()

    model    = mujoco.MjModel.from_xml_path(str(args.model))
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    joint_info = []
    for jname, label in LEG_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint '{jname}' not found in model")
        qadr = model.jnt_qposadr[jid]
        lo   = float(model.jnt_range[jid, 0])
        hi   = float(model.jnt_range[jid, 1])
        joint_info.append((jname, label, qadr, lo, hi))
        print(f"  {jname:20s}  range=[{np.rad2deg(lo):.1f}°, {np.rad2deg(hi):.1f}°]")

    base = _base_qpos(model)

    # Camera: slight front-left to see the left leg clearly
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = base[:3] + np.array([0.0, 0.0, 0.0])  # pelvis level
    cam.distance  = 2.8
    cam.azimuth   = 150.0   # front-right so left leg is visible
    cam.elevation = -8.0

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    total_frames_est = len(LEG_JOINTS) * (HOLD_FRAMES + SWEEP_STEPS + HOLD_FRAMES + SWEEP_STEPS * 2 + HOLD_FRAMES + SWEEP_STEPS)
    print(f"\nEstimated frames: {total_frames_est}  (~{total_frames_est / FPS:.1f} s)")
    print(f"Output: {args.out_mp4}")

    with imageio.get_writer(str(args.out_mp4), fps=FPS, codec="libx264", quality=8) as writer:
        for j_idx, (jname, label, qadr, lo, hi) in enumerate(joint_info):
            color     = JOINT_COLORS[j_idx % len(JOINT_COLORS)]
            total_seg = HOLD_FRAMES + SWEEP_STEPS + HOLD_FRAMES + SWEEP_STEPS * 2 + HOLD_FRAMES + SWEEP_STEPS
            frame_n   = 0
            print(f"  [{j_idx + 1}/{len(LEG_JOINTS)}] {label}")

            def emit(angle_rad: float, frac: float) -> None:
                nonlocal frame_n
                q = base.copy()
                q[qadr] = angle_rad
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=cam)
                img = renderer.render()
                img = _add_overlay(img, label, np.rad2deg(angle_rad), j_idx, color)
                img = _add_progress_bar(img, frac, color)
                writer.append_data(img)
                frame_n += 1

            # Wider range first: pick direction with larger absolute limit
            if abs(hi) >= abs(lo):
                first, second = hi, lo
            else:
                first, second = lo, hi

            for _ in range(HOLD_FRAMES):                        # hold at zero
                emit(0.0, frame_n / total_seg)
            for k in range(SWEEP_STEPS):                        # 0 → first extreme
                emit(first * (k + 1) / SWEEP_STEPS, frame_n / total_seg)
            for _ in range(HOLD_FRAMES):                        # hold at first extreme
                emit(first, frame_n / total_seg)
            for k in range(SWEEP_STEPS * 2):                    # first → second extreme
                t = (k + 1) / (SWEEP_STEPS * 2)
                emit(first + t * (second - first), frame_n / total_seg)
            for _ in range(HOLD_FRAMES):                        # hold at second extreme
                emit(second, frame_n / total_seg)
            for k in range(SWEEP_STEPS):                        # return to zero
                emit(second * (1.0 - (k + 1) / SWEEP_STEPS), frame_n / total_seg)

    renderer.close()
    print("Done:", args.out_mp4)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a retargeted Alex qpos NPZ to MP4, with side-by-side canonical human stick figure.

Left panel : Alex robot via MuJoCo EGL renderer.
Right panel: Canonical human skeleton (from target_positions in the NPZ), drawn as a
             stick figure using the same camera azimuth/elevation as the robot panel.

Frame-step is auto-computed from the source FPS stored in the NPZ so the output video
plays at approximately real-time speed (source 60 fps → output 30 fps = frame-step 2).
Override with --frame-step to force a specific decimation.

Usage:
    MUJOCO_GL=egl conda run -n gmr python scripts/visualization/render_alex_qp_direct_mp4_fresh.py \\
        --npz outputs/grounded/standup_02_grounded.npz \\
        --model assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml \\
        --out-mp4 outputs/renders/standup_02_grounded.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Skeleton connectivity (role names from the IK solver)
# Each entry: (role_a, role_b, RGB_color)
# Left side = blue family, right side = orange family, spine = white.
# ---------------------------------------------------------------------------
SKELETON_EDGES = [
    # Spine
    ("pelvis",         "torso",           (230, 230, 230)),
    ("torso",          "head",            (230, 230, 230)),
    # Left leg
    ("pelvis",         "left_hip",        (100, 160, 255)),
    ("left_hip",       "left_knee",       (100, 160, 255)),
    ("left_knee",      "left_ankle",      (100, 160, 255)),
    # Right leg
    ("pelvis",         "right_hip",       (255, 160, 80)),
    ("right_hip",      "right_knee",      (255, 160, 80)),
    ("right_knee",     "right_ankle",     (255, 160, 80)),
    # Left arm
    ("torso",          "left_shoulder",   (80, 200, 255)),
    ("left_shoulder",  "left_elbow",      (80, 200, 255)),
    ("left_elbow",     "left_wrist",      (80, 200, 255)),
    # Right arm
    ("torso",          "right_shoulder",  (255, 200, 80)),
    ("right_shoulder", "right_elbow",     (255, 200, 80)),
    ("right_elbow",    "right_wrist",     (255, 200, 80)),
]

# Joint dot colour per side (index matches role list order)
LEFT_JOINT_COLOR  = (120, 180, 255)
RIGHT_JOINT_COLOR = (255, 180, 100)
SPINE_JOINT_COLOR = (255, 255, 200)

LEFT_ROLES  = {"left_hip", "left_knee", "left_ankle", "left_shoulder", "left_elbow", "left_wrist"}
RIGHT_ROLES = {"right_hip", "right_knee", "right_ankle", "right_shoulder", "right_elbow", "right_wrist"}


def _get_source_fps(z: np.lib.npyio.NpzFile) -> float | None:
    """Read source FPS from the NPZ (stored as a top-level key or in metadata_json)."""
    if "fps" in z.files:
        return float(z["fps"])
    if "metadata_json" in z.files:
        try:
            meta = json.loads(str(z["metadata_json"]))
            if "fps" in meta:
                return float(meta["fps"])
        except Exception:
            pass
    return None


def _camera_basis(azimuth_deg: float, elevation_mujoco_deg: float):
    """
    Orthographic projection basis (right, up) for a given MuJoCo free-camera pose.

    MuJoCo convention: azimuth is CCW from the -Y axis (looking down), and
    elevation is negative for "above" horizontal.
    """
    az  = np.deg2rad(azimuth_deg)
    el  = np.deg2rad(-elevation_mujoco_deg)  # flip: MuJoCo -20 = 20° above

    # Camera right (always horizontal)
    right = np.array([np.cos(az), np.sin(az), 0.0])

    # Camera forward direction (from camera toward the lookat point)
    view_fwd = np.array([-np.sin(az) * np.cos(el),
                          np.cos(az) * np.cos(el),
                         -np.sin(el)])

    # Camera up = right × view_forward (right-handed, image Y points up)
    up = np.cross(right, view_fwd)
    norm = np.linalg.norm(up)
    if norm < 1e-9:
        up = np.array([0.0, 0.0, 1.0])
    else:
        up /= norm

    return right, up


def _project(
    positions: np.ndarray,    # (K, 3)
    center: np.ndarray,       # (3,)
    right: np.ndarray,        # (3,)
    up: np.ndarray,           # (3,)
    scale: float,             # pixels per metre
    W: int,
    H: int,
) -> list[tuple[int, int]]:
    """Orthographic project 3D positions onto a WxH panel."""
    pts: list[tuple[int, int]] = []
    for p in positions:
        d = p - center
        px = int(W // 2 + np.dot(d, right) * scale)
        py = int(H // 2 - np.dot(d, up) * scale)
        pts.append((px, py))
    return pts


def _draw_human_panel(
    target_positions: np.ndarray,  # (K, 3) for this frame
    role_to_idx: dict[str, int],
    azimuth_deg: float,
    elevation_deg: float,
    W: int,
    H: int,
    scale: float,
) -> np.ndarray:
    """Return an HxWx3 uint8 image of the human skeleton stick figure."""
    right, up = _camera_basis(azimuth_deg, elevation_deg)

    # Centre the figure on the pelvis (if available) or the mean joint position.
    if "pelvis" in role_to_idx:
        center = target_positions[role_to_idx["pelvis"]].copy()
    else:
        center = target_positions.mean(axis=0)

    pts = _project(target_positions, center, right, up, scale, W, H)

    img = Image.new("RGB", (W, H), color=(18, 18, 28))
    draw = ImageDraw.Draw(img)

    # Ground plane guide line (Z = 0 in world = pelvis Z level approx)
    ground_screen_y = H // 2
    draw.line([(0, ground_screen_y), (W, ground_screen_y)],
              fill=(60, 80, 60), width=1)

    # Bones
    for role_a, role_b, color in SKELETON_EDGES:
        if role_a not in role_to_idx or role_b not in role_to_idx:
            continue
        pa = pts[role_to_idx[role_a]]
        pb = pts[role_to_idx[role_b]]
        # Clip to panel bounds before drawing
        draw.line([pa, pb], fill=color, width=3)

    # Joints
    for role, idx in role_to_idx.items():
        px, py = pts[idx]
        r = 5
        if role in LEFT_ROLES:
            color = LEFT_JOINT_COLOR
        elif role in RIGHT_ROLES:
            color = RIGHT_JOINT_COLOR
        else:
            color = SPINE_JOINT_COLOR
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    return np.array(img)


def _add_label(img: np.ndarray, text: str, pos=(8, 6)) -> np.ndarray:
    """Overlay a small text label onto a numpy image array (in-place via PIL)."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    # Shadow
    draw.text((pos[0] + 1, pos[1] + 1), text, fill=(0, 0, 0))
    draw.text(pos, text, fill=(220, 220, 220))
    return np.array(pil)


def set_camera(model, data, cam, frame_idx: int, total_frames: int) -> None:
    """Slightly orbiting camera that follows the robot centroid."""
    pts = np.asarray([data.xpos[b].copy() for b in range(1, model.nbody)])
    center = pts.mean(axis=0)
    extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))

    progress = frame_idx / max(total_frames - 1, 1)
    cam.lookat[:] = center
    cam.distance  = max(1.2, extent * (2.4 - 0.5 * progress))
    cam.azimuth   = 135.0 + 25.0 * progress
    cam.elevation  = -20.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz",       required=True, type=Path)
    ap.add_argument("--model",     required=True, type=Path)
    ap.add_argument("--out-mp4",   required=True, type=Path)
    ap.add_argument("--width",     type=int,   default=640)
    ap.add_argument("--height",    type=int,   default=480)
    ap.add_argument("--fps",       type=float, default=30.0,
                    help="Output video FPS (default: 30). Source FPS is read from the "
                         "NPZ to auto-compute frame-step for real-time playback.")
    ap.add_argument("--frame-step", type=int,  default=None,
                    help="Explicit frame decimation. Default: auto (source_fps / output_fps).")
    ap.add_argument("--no-human",  action="store_true",
                    help="Disable the side-by-side human stick figure (single panel output).")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=np.float64)

    # ---------------------------------------------------------------------------
    # Auto frame-step for real-time playback
    # ---------------------------------------------------------------------------
    source_fps = _get_source_fps(z)
    if args.frame_step is not None:
        frame_step = args.frame_step
    elif source_fps is not None and source_fps > args.fps * 1.1:
        frame_step = max(1, round(source_fps / args.fps))
        print(f"Auto frame-step: {frame_step}  "
              f"(source {source_fps:.1f} fps → output {args.fps:.1f} fps = real-time)")
    else:
        frame_step = 1
        if source_fps is not None:
            print(f"Source FPS ({source_fps:.1f}) ≤ output FPS ({args.fps:.1f}), frame-step=1")

    # ---------------------------------------------------------------------------
    # Human stick figure data (optional — needs target_positions + role_names)
    # ---------------------------------------------------------------------------
    show_human = (not args.no_human
                  and "target_positions" in z.files
                  and "role_names" in z.files)

    if show_human:
        target_positions = np.asarray(z["target_positions"], dtype=np.float64)  # (N, K, 3)
        role_names = [str(r) for r in z["role_names"]]
        role_to_idx = {r: i for i, r in enumerate(role_names)}
        # Scale: pixels per metre. Alex is ~1.5 m tall; fill ~60% of panel height.
        human_scale = args.height * 0.6 / 1.5
        print(f"Human overlay: {len(role_names)} roles  scale={human_scale:.0f} px/m")
    else:
        if not args.no_human:
            print("No target_positions in NPZ — skipping human overlay.")

    # ---------------------------------------------------------------------------
    # MuJoCo setup
    # ---------------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data  = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    frame_ids  = list(range(0, qpos.shape[0], frame_step))
    total_out  = len(frame_ids)
    out_width  = args.width * 2 if show_human else args.width

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {total_out} frames  ({qpos.shape[0]} source → step={frame_step})")
    print(f"Output: {args.out_mp4}  ({out_width}×{args.height} @ {args.fps:.0f} fps)")

    with imageio.get_writer(str(args.out_mp4),
                            fps=args.fps,
                            codec="libx264",
                            quality=8) as writer:
        for out_i, src_i in enumerate(frame_ids):
            # --- Robot panel ---
            data.qpos[:] = qpos[src_i]
            mujoco.mj_forward(model, data)
            set_camera(model, data, cam, out_i, total_out)
            renderer.update_scene(data, camera=cam)
            robot_img = renderer.render()                       # (H, W, 3)
            robot_img = _add_label(robot_img, "Alex (IK)")

            if show_human:
                # --- Human stick figure panel ---
                human_img = _draw_human_panel(
                    target_positions[src_i],
                    role_to_idx,
                    azimuth_deg=float(cam.azimuth),
                    elevation_deg=float(cam.elevation),
                    W=args.width,
                    H=args.height,
                    scale=human_scale,
                )
                human_img = _add_label(human_img, "Canonical human (IK targets)")
                frame = np.concatenate([robot_img, human_img], axis=1)
            else:
                frame = robot_img

            writer.append_data(frame)

            if out_i % 100 == 0 or out_i == total_out - 1:
                print(f"  {out_i + 1}/{total_out}")

    renderer.close()
    print("Wrote:", args.out_mp4)


if __name__ == "__main__":
    main()

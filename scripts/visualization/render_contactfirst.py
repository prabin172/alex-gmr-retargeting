#!/usr/bin/env python3
"""Render a contact-first IK NPZ to MP4 with a per-frame contact indicator.

Left panel : Alex V2 robot (MuJoCo EGL).
Right panel: canonical human stick figure (IK targets), same camera.
Bottom strip: contact status for each effector (foot/hand) — green = in contact
(constraint active this frame), grey = free — read from `contact_flags` /
`contact_effector_names` written by solve_fbx_canonical_alex_contactfirst.py.

Usage:
    MUJOCO_GL=egl python scripts/visualization/render_contactfirst.py \
        --npz outputs/contactfirst/standup_02_contactfirst.npz \
        --out-mp4 outputs/renders/contactfirst/standup_02_contactfirst.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites_v2.xml"

SKELETON_EDGES = [
    ("pelvis", "torso", (230, 230, 230)),
    ("torso", "head", (230, 230, 230)),
    ("pelvis", "left_hip", (100, 160, 255)),
    ("left_hip", "left_knee", (100, 160, 255)),
    ("left_knee", "left_ankle", (100, 160, 255)),
    ("pelvis", "right_hip", (255, 160, 80)),
    ("right_hip", "right_knee", (255, 160, 80)),
    ("right_knee", "right_ankle", (255, 160, 80)),
    ("torso", "left_shoulder", (80, 200, 255)),
    ("left_shoulder", "left_elbow", (80, 200, 255)),
    ("left_elbow", "left_wrist", (80, 200, 255)),
    ("torso", "right_shoulder", (255, 200, 80)),
    ("right_shoulder", "right_elbow", (255, 200, 80)),
    ("right_elbow", "right_wrist", (255, 200, 80)),
]
LEFT_ROLES = {"left_hip", "left_knee", "left_ankle", "left_shoulder", "left_elbow", "left_wrist"}
RIGHT_ROLES = {"right_hip", "right_knee", "right_ankle", "right_shoulder", "right_elbow", "right_wrist"}


def _get_source_fps(z):
    if "fps" in z.files:
        return float(z["fps"])
    return None


def _camera_basis(azimuth_deg, elevation_mujoco_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(-elevation_mujoco_deg)
    right = np.array([np.cos(az), np.sin(az), 0.0])
    view_fwd = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])
    up = np.cross(right, view_fwd)
    n = np.linalg.norm(up)
    up = up / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    return right, up


def _project(positions, center, right, up, scale, W, H):
    pts = []
    for p in positions:
        d = p - center
        pts.append((int(W // 2 + np.dot(d, right) * scale), int(H // 2 - np.dot(d, up) * scale)))
    return pts


def _draw_human_panel(target_positions, role_to_idx, az, el, W, H, scale):
    right, up = _camera_basis(az, el)
    center = target_positions[role_to_idx["pelvis"]].copy() if "pelvis" in role_to_idx \
        else target_positions.mean(axis=0)
    pts = _project(target_positions, center, right, up, scale, W, H)
    img = Image.new("RGB", (W, H), color=(18, 18, 28))
    draw = ImageDraw.Draw(img)
    draw.line([(0, H // 2), (W, H // 2)], fill=(60, 80, 60), width=1)
    for ra, rb, color in SKELETON_EDGES:
        if ra in role_to_idx and rb in role_to_idx:
            draw.line([pts[role_to_idx[ra]], pts[role_to_idx[rb]]], fill=color, width=3)
    for role, idx in role_to_idx.items():
        px, py = pts[idx]
        c = (120, 180, 255) if role in LEFT_ROLES else (255, 180, 100) if role in RIGHT_ROLES else (255, 255, 200)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=c)
    return np.array(img)


def _add_label(img, text, pos=(8, 6)):
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.text((pos[0] + 1, pos[1] + 1), text, fill=(0, 0, 0))
    d.text(pos, text, fill=(220, 220, 220))
    return np.array(pil)


def _contact_strip(effector_names, flags, align_errs, W, height=46):
    """Bottom status strip: one chip per effector, green if in contact."""
    img = Image.new("RGB", (W, height), color=(12, 12, 16))
    draw = ImageDraw.Draw(img)
    n = len(effector_names)
    cw = W // max(n, 1)
    for i, name in enumerate(effector_names):
        on = bool(flags[i])
        x0 = i * cw
        bg = (30, 150, 60) if on else (45, 45, 52)
        draw.rectangle([x0 + 4, 6, x0 + cw - 4, height - 6], fill=bg, outline=(90, 90, 100))
        txt = name.replace("_", " ")
        if on and align_errs is not None:
            txt += f"  {align_errs[i]:.0f}deg"
        draw.text((x0 + 12, 10), txt, fill=(235, 235, 235))
        draw.text((x0 + 12, 24), "CONTACT" if on else "free",
                  fill=(220, 255, 220) if on else (150, 150, 160))
    return np.array(img)


def set_camera(model, data, cam, frame_idx, total_frames, *,
               fixed=False, az=135.0, el=-20.0, dist=None, lookat=None):
    """Position the camera. Default: slow orbit + zoom (progress-based).
    fixed=True: constant azimuth/elevation/distance AND a constant lookat
    (`lookat`, the clip-global center) so the WORLD is static and only Alex
    moves in frame — no orbit, no zoom, no pan."""
    if fixed:
        cam.lookat[:] = lookat if lookat is not None else data.xpos[1:model.nbody].mean(axis=0)
        pts = np.asarray([data.xpos[b].copy() for b in range(1, model.nbody)])
        extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        cam.distance = dist if (dist and dist > 0) else max(1.2, extent * 2.2)
        cam.azimuth = az
        cam.elevation = el
    else:
        pts = np.asarray([data.xpos[b].copy() for b in range(1, model.nbody)])
        cam.lookat[:] = pts.mean(axis=0)
        extent = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        progress = frame_idx / max(total_frames - 1, 1)
        cam.distance = max(1.2, extent * (2.4 - 0.5 * progress))
        cam.azimuth = 135.0 + 25.0 * progress
        cam.elevation = -20.0


def add_ground_plane(scene, z, center_xy, half=2.5, rgba=(0.45, 0.5, 0.55, 0.35)):
    """Append a semi-transparent ground plane geom to the scene at height z."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_PLANE,
        np.array([half, half, 0.1]),
        np.array([center_xy[0], center_xy[1], z], dtype=np.float64),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_DEFAULT, type=Path)
    ap.add_argument("--out-mp4", required=True, type=Path)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frame-step", type=int, default=None)
    ap.add_argument("--no-human", action="store_true")
    ap.add_argument("--fixed-cam", action="store_true",
                    help="Freeze camera orbit/zoom (constant azimuth/elevation/distance) "
                         "so contact make/break flicker is easy to see; lookat still recenters.")
    ap.add_argument("--cam-azimuth", type=float, default=135.0)
    ap.add_argument("--cam-elevation", type=float, default=-20.0)
    ap.add_argument("--cam-distance", type=float, default=0.0,
                    help="Fixed-cam distance (0 = auto from clip extent).")
    ap.add_argument("--ground", action="store_true",
                    help="Draw a semi-transparent ground plane below Alex (at the clip's lowest point).")
    ap.add_argument("--ground-z", type=float, default=None,
                    help="Pin the ground plane to this world Z instead of the clip's lowest body origin "
                         "(use 0.0 for z-grounded NPZs).")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=np.float64)

    # Contact overlay data (written by the contact-first solver)
    eff_names = [str(x) for x in z["contact_effector_names"]] if "contact_effector_names" in z.files else []
    contact_flags = np.asarray(z["contact_flags"]) if "contact_flags" in z.files else None
    align_errs = np.asarray(z["contact_align_errors_deg"]) if "contact_align_errors_deg" in z.files else None

    source_fps = _get_source_fps(z)
    if args.frame_step is not None:
        frame_step = args.frame_step
    elif source_fps is not None and source_fps > args.fps * 1.1:
        frame_step = max(1, round(source_fps / args.fps))
    else:
        frame_step = 1

    show_human = (not args.no_human and "target_positions" in z.files and "role_names" in z.files)
    if show_human:
        target_positions = np.asarray(z["target_positions"], dtype=np.float64)
        role_names = [str(r) for r in z["role_names"]]
        role_to_idx = {r: i for i, r in enumerate(role_names)}
        human_scale = args.height * 0.6 / 1.5

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    frame_ids = list(range(0, qpos.shape[0], frame_step))
    total_out = len(frame_ids)
    out_width = args.width * 2 if show_human else args.width

    # Fixed camera / ground: precompute the clip-GLOBAL bounding box (over all
    # frames) so lookat, distance and ground height stay constant and the world
    # is static (only Alex moves). Distance from the global extent keeps the
    # whole trajectory in frame.
    fixed_dist = args.cam_distance
    fixed_lookat = None
    ground_z = None
    ground_xy = (0.0, 0.0)
    if args.fixed_cam or args.ground:
        gmin = np.full(3, np.inf)
        gmax = np.full(3, -np.inf)
        for src_i in frame_ids[:: max(1, len(frame_ids) // 40)]:
            data.qpos[:] = qpos[src_i]
            mujoco.mj_forward(model, data)
            pts = np.asarray([data.xpos[b] for b in range(1, model.nbody)])
            gmin = np.minimum(gmin, pts.min(axis=0))
            gmax = np.maximum(gmax, pts.max(axis=0))
        gcenter = (gmin + gmax) / 2.0
        gextent = float(np.max(gmax - gmin))
        fixed_lookat = gcenter
        if fixed_dist <= 0.0:
            fixed_dist = max(1.7, gextent * 1.7)   # pulled back so the full robot stays in frame
        ground_z = args.ground_z if args.ground_z is not None else float(gmin[2])
        ground_xy = (float(gcenter[0]), float(gcenter[1]))

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {total_out} frames -> {args.out_mp4}  ({out_width}x{args.height} @ {args.fps:.0f}fps)"
          + (f"  [fixed cam az={args.cam_azimuth} el={args.cam_elevation} d={fixed_dist:.2f}]" if args.fixed_cam else ""))
    if len(eff_names):
        print(f"Contact effectors: {eff_names}")

    with imageio.get_writer(str(args.out_mp4), fps=args.fps, codec="libx264", quality=8) as writer:
        for out_i, src_i in enumerate(frame_ids):
            data.qpos[:] = qpos[src_i]
            mujoco.mj_forward(model, data)
            set_camera(model, data, cam, out_i, total_out,
                       fixed=args.fixed_cam, az=args.cam_azimuth,
                       el=args.cam_elevation, dist=fixed_dist, lookat=fixed_lookat)
            renderer.update_scene(data, camera=cam)
            if args.ground and ground_z is not None:
                add_ground_plane(renderer.scene, ground_z, ground_xy)
            robot_img = _add_label(renderer.render(), "Alex V2 (contact-first IK)")

            if show_human:
                human_img = _draw_human_panel(target_positions[src_i], role_to_idx,
                                              float(cam.azimuth), float(cam.elevation),
                                              args.width, args.height, human_scale)
                human_img = _add_label(human_img, "Canonical human (IK targets)")
                frame = np.concatenate([robot_img, human_img], axis=1)
            else:
                frame = robot_img

            if contact_flags is not None and len(eff_names):
                strip = _contact_strip(
                    eff_names, contact_flags[src_i],
                    align_errs[src_i] if align_errs is not None else None,
                    out_width,
                )
                frame = np.concatenate([frame, strip], axis=0)

            writer.append_data(frame)
            if out_i % 100 == 0 or out_i == total_out - 1:
                print(f"  {out_i + 1}/{total_out}")

    renderer.close()
    print("Wrote:", args.out_mp4)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render a qpos npz (this project's G1 solves, e.g. ours_g1_corpus/*.npz) to
video using the vetted-collision+floor model (real G1 mesh visuals kept
untouched, ground plane visible, matches the SAME model every eval in this
sprint uses -- g1_model_setup.py). Overlays per-frame penetration depth +
deepest-offending body name (mesh-accurate, same _geom_lowest_z machinery as
sprint_s3_full_corpus.py's whole_clip_metrics), red flash when penetrating.

Usage:
    conda run -n gmr python scripts/g1/render_penetration_annotated.py \\
        --qpos outputs/gmr_baseline/sprint/ours_g1_corpus/walk1_subject1_ours.npz \\
        --out outputs/gmr_baseline/sprint/renders/walk1_subject1_ours_annotated.mp4 \\
        --start 0 --frames 1800
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _geom_lowest_z  # noqa: E402

ROBOT_BASE = "pelvis"


def _font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", type=Path, help="npz with a 'qpos' array")
    ap.add_argument("--pkl", type=Path, help="GMR pkl (alternative to --qpos)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=None, help="default: npz's own 'fps' or 30")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=None, help="default: rest of clip")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=110.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--distance", type=float, default=2.6)
    ap.add_argument("--pen-threshold-cm", type=float, default=0.5,
                    help="Flash red when penetration exceeds this (cm).")
    args = ap.parse_args()
    assert (args.qpos is None) != (args.pkl is None), "pass exactly one of --qpos / --pkl"

    if args.pkl is not None:
        from load_gmr_pkl import load_gmr_pkl
        qpos, fps = load_gmr_pkl(args.pkl)
        fps = args.fps if args.fps is not None else fps
    else:
        z = np.load(args.qpos, allow_pickle=True)
        qpos = z["qpos"]
        fps = args.fps if args.fps is not None else float(z["fps"]) if "fps" in z else 30.0

    start = args.start
    end = qpos.shape[0] if args.frames is None else min(qpos.shape[0], start + args.frames)

    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    body_of_geom = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                   for g in geom_ids}

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    robot_base_id = model.body(ROBOT_BASE).id
    cam = mujoco.MjvCamera()
    cam.distance = args.distance
    cam.elevation = args.elevation
    cam.azimuth = args.azimuth

    font_big = _font(28)
    font_small = _font(20)

    frames = []
    n_pen = 0
    max_pen = 0.0
    for t in range(start, end):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)

        best_z, best_g = 1e9, None
        for g in geom_ids:
            zz = _geom_lowest_z(g, model, data, mesh_cache)
            if zz < best_z:
                best_z, best_g = zz, g
        pen_cm = max(0.0, -best_z) * 100.0
        body_name = body_of_geom[best_g] if pen_cm > 0 else None
        if pen_cm > args.pen_threshold_cm:
            n_pen += 1
        max_pen = max(max_pen, pen_cm)

        cam.lookat = data.xpos[robot_base_id]
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render().copy())
        draw = ImageDraw.Draw(img, "RGBA")

        penetrating = pen_cm > args.pen_threshold_cm
        if penetrating:
            draw.rectangle([0, 0, args.width, args.height], outline=(255, 0, 0, 255), width=10)
        label = f"frame {t}   t={t/fps:5.2f}s"
        draw.text((14, 12), label, font=font_small, fill=(255, 255, 255, 255))
        pen_label = f"PENETRATION: {pen_cm:5.2f} cm" if penetrating else f"penetration: {pen_cm:5.2f} cm"
        pen_color = (255, 40, 40, 255) if penetrating else (120, 255, 120, 255)
        draw.text((14, 40), pen_label, font=font_big, fill=pen_color)
        if body_name:
            draw.text((14, 74), f"body: {body_name}", font=font_small, fill=pen_color)

        frames.append(np.asarray(img))
        if (t - start) % 300 == 0:
            print(f"  frame {t}/{end} pen={pen_cm:.2f}cm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    imageio.mimwrite(args.out, frames, fps=fps)
    print(f"Wrote {args.out} ({len(frames)} frames @ {fps}fps)  "
          f"penetrating(> {args.pen_threshold_cm}cm) frames: {n_pen}/{len(frames)} "
          f"({100*n_pen/len(frames):.1f}%)  max_pen={max_pen:.2f}cm")


if __name__ == "__main__":
    main()

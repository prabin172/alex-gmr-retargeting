#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import imageio.v2 as imageio


DEFAULT_MODEL = Path("assets/alex/temp_alex_floating_base_visual_mesh.xml")


def set_camera_from_bodies(model, data, cam):
    pts = []
    for b in range(1, model.nbody):
        pts.append(data.xpos[b].copy())
    pts = np.asarray(pts)

    center = pts.mean(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extent = np.max(maxs - mins)

    cam.lookat[:] = center
    cam.distance = max(1.4, float(extent) * 2.2)
    cam.azimuth = 135
    cam.elevation = -20


def render_one(model, qpos, out_path: Path, width: int, height: int):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    set_camera_from_bodies(model, data, cam)

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    renderer.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, img)
    print("wrote:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 5, 10, 15, -1])
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--gif", type=Path, default=None)
    ap.add_argument("--mp4", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=float)
    source_frame_ids = np.asarray(z["source_frame_ids"], dtype=int)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    T = qpos.shape[0]

    written = []
    frames_to_render = list(range(T)) if args.all_frames else args.frames

    for f in frames_to_render:
        idx = T - 1 if f < 0 else f
        if idx < 0 or idx >= T:
            print("skip invalid frame:", f)
            continue

        src = int(source_frame_ids[idx])
        out_path = args.out_dir / f"alex_visual_frame_{idx:04d}_src_{src:04d}.png"
        render_one(model, qpos[idx], out_path, args.width, args.height)
        written.append(out_path)

    if args.gif is not None and written:
        imgs = [imageio.imread(p) for p in written]
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, imgs, duration=1.0 / args.fps)
        print("wrote:", args.gif)

    if args.mp4 is not None and written:
        args.mp4.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.mp4, fps=args.fps, codec="libx264", quality=8) as writer:
            for p in written:
                writer.append_data(imageio.imread(p))
        print("wrote:", args.mp4)


if __name__ == "__main__":
    main()

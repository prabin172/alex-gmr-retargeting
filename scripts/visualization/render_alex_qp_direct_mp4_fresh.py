#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def set_camera_from_bodies(model, data, cam, frame_idx, total_frames):
    pts = np.asarray([data.xpos[b].copy() for b in range(1, model.nbody)])
    center = pts.mean(axis=0)
    extent = np.max(pts.max(axis=0) - pts.min(axis=0))

    # Slight moving camera, like the one you liked.
    progress = frame_idx / max(total_frames - 1, 1)
    cam.lookat[:] = center
    cam.distance = max(1.2, float(extent) * (2.4 - 0.5 * progress))
    cam.azimuth = 135 + 25 * progress
    cam.elevation = -20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out-mp4", required=True, type=Path)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--frame-step", type=int, default=1)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=float)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    frame_ids = list(range(0, qpos.shape[0], args.frame_step))

    args.out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(args.out_mp4, fps=args.fps, codec="libx264", quality=8) as writer:
        for out_i, src_i in enumerate(frame_ids):
            data.qpos[:] = qpos[src_i]
            mujoco.mj_forward(model, data)

            set_camera_from_bodies(model, data, cam, out_i, len(frame_ids))
            renderer.update_scene(data, camera=cam)
            img = renderer.render()
            writer.append_data(img)

            if out_i % 100 == 0 or out_i == len(frame_ids) - 1:
                print(f"rendered {out_i + 1}/{len(frame_ids)} frames")

    renderer.close()
    print("wrote:", args.out_mp4)


if __name__ == "__main__":
    main()

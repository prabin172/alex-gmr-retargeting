#!/usr/bin/env python3
"""Render a GMR pkl (raw or polished) to video, offscreen (EGL, camera tracking pelvis)
-- same rendering path as gmr_headless_retarget.py's --video_path, factored out so it
can run on an already-saved pkl without re-retargeting.

Usage:
    conda run -n gmr python scripts/g1/render_gmr_pkl.py \\
        --pkl outputs/gmr_baseline/pkl/walk1_subject1.pkl \\
        --out outputs/gmr_baseline/videos/walk1_subject1_v2.mp4
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_gmr_pkl import load_gmr_pkl  # noqa: E402

G1_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
ROBOT_BASE = "pelvis"
CAM_DISTANCE = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=G1_MODEL_DEFAULT)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=90.0)
    args = ap.parse_args()

    qpos, fps = load_gmr_pkl(args.pkl)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    robot_base_id = model.body(ROBOT_BASE).id
    cam = mujoco.MjvCamera()
    cam.distance = CAM_DISTANCE
    cam.elevation = -10
    cam.azimuth = args.azimuth

    frames = []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        cam.lookat = data.xpos[robot_base_id]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    import imageio
    imageio.mimwrite(args.out, frames, fps=fps)
    print(f"Wrote {args.out} ({len(frames)} frames @ {fps}fps)")


if __name__ == "__main__":
    main()

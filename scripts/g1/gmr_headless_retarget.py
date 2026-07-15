"""Headless GMR retarget: BVH -> G1 pkl (+ optional offscreen-rendered video).

Same retargeting core as GMR's own scripts/bvh_to_robot.py (GeneralMotionRetargeting
class, load_bvh_file loader) but bypasses RobotMotionViewer's launch_passive(), which
opens a GLFW window and hard-requires a display (this machine has none: DISPLAY="").
Video, if requested, uses mujoco.Renderer with MUJOCO_GL=egl (offscreen, confirmed
working) instead.

Output pkl format matches bvh_to_robot.py's --save_path exactly:
    {fps, root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,29), local_body_pos: None,
     link_body_list: None}

Usage:
    conda run -n gmr python scripts/g1/gmr_headless_retarget.py \\
        --bvh_file data/raw/lafan1/walk1_subject1.bvh --robot unitree_g1 \\
        --save_path outputs/gmr_baseline/pkl/walk1_subject1.pkl \\
        --video_path outputs/gmr_baseline/videos/walk1_subject1.mp4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pathlib
import pickle

import mujoco
import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_file", required=True, type=str)
    ap.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--motion_fps", type=int, default=30)
    ap.add_argument("--save_path", required=True, type=str)
    ap.add_argument("--video_path", type=str, default=None)
    ap.add_argument("--video_width", type=int, default=640)
    ap.add_argument("--video_height", type=int, default=480)
    args = ap.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if args.video_path:
        vdir = os.path.dirname(args.video_path)
        if vdir:
            os.makedirs(vdir, exist_ok=True)

    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)

    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    qpos_list = []
    frames_rgb = [] if args.video_path else None

    model = None
    data = None
    renderer = None
    cam = None
    robot_base_id = None
    if args.video_path:
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        robot_base_id = model.body(ROBOT_BASE_DICT[args.robot]).id
        cam = mujoco.MjvCamera()
        cam.distance = VIEWER_CAM_DISTANCE_DICT[args.robot]
        cam.elevation = -10
        cam.azimuth = 90

    for smplx_data in tqdm(lafan1_data_frames, desc="Retargeting"):
        qpos = retargeter.retarget(smplx_data)
        qpos_list.append(qpos)
        if args.video_path:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            cam.lookat = data.xpos[robot_base_id]
            renderer.update_scene(data, camera=cam)
            frames_rgb.append(renderer.render().copy())

    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
    dof_pos = np.array([q[7:] for q in qpos_list])

    motion_data = {
        "fps": args.motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved {len(qpos_list)} frames to {args.save_path}")

    if args.video_path:
        import imageio

        imageio.mimwrite(args.video_path, frames_rgb, fps=args.motion_fps)
        print(f"Saved video to {args.video_path}")


if __name__ == "__main__":
    main()

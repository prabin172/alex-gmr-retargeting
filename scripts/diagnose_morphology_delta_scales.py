#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
REPO_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import mink
import numpy as np

from solve_mvnx_alex_motion import (
    repo_root,
    choose_solver,
    make_tasks,
    set_task_targets,
    solve_frame,
    robot_rest_frame_from_mujoco,
    recenter_clip_xy,
)
from general_motion_retargeting.source_adapters.mvnx import read_mvnx_canonical_frames
from general_motion_retargeting.retargeting.morphology_delta import compute_morphology_scales


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mvnx_path", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-ik-iter", type=int, default=200)
    args = parser.parse_args()

    source_frames, meta = read_mvnx_canonical_frames(
        mvnx_path=args.mvnx_path,
        frame_type="normal",
        start_frame=args.start_frame,
        stride=args.stride,
        max_frames=1,
        canonicalize_axes=True,
        canonicalize_heading=True,
    )
    source_frames = recenter_clip_xy(source_frames)
    source_rest = source_frames[0]

    robot_cfg = json.loads(Path("general_motion_retargeting/robot_configs/alex.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(repo_root / robot_cfg["model_path"]))

    qpos0 = np.asarray(model.qpos0, dtype=float)
    qpos0[0:3] = np.asarray(source_rest["pelvis"]["pos"], dtype=float)
    qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

    solver = choose_solver()
    tasks = make_tasks(model, robot_cfg)
    limits = [mink.ConfigurationLimit(model)]

    rest_configuration = mink.Configuration(model, q=qpos0.copy())
    rest_target_by_role = set_task_targets(tasks, source_rest, robot_cfg)

    qpos_rest, rest_score, rest_errors, _ = solve_frame(
        model=model,
        configuration=rest_configuration,
        tasks=tasks,
        target_by_role=rest_target_by_role,
        solver=solver,
        limits=limits,
        max_iter=args.max_ik_iter,
    )

    target_rest = robot_rest_frame_from_mujoco(
        model,
        qpos_rest,
        robot_cfg["retarget_body_names"],
    )

    scales = compute_morphology_scales(
        source_rest=source_rest,
        target_rest=target_rest,
        preserve_root_translation=True,
        clamp_min=0.70,
        clamp_max=1.30,
    )

    print()
    print("=== Morphology-aware delta scales ===")
    print(f"MVNX: {args.mvnx_path}")
    print(f"start_frame: {args.start_frame}")
    print(f"stride: {args.stride}")
    print(f"rest_alignment_score: {rest_score:.6f}")
    print()

    print("Role scales:")
    for role, scale in scales.role_scales.items():
        print(f"  {role:16s}: {scale:.4f}")

    print()
    print("Measurements:")
    for key in sorted(scales.measurements):
        val = scales.measurements[key]
        if isinstance(val, bool):
            print(f"  {key:32s}: {val}")
        else:
            print(f"  {key:32s}: {val:.6f}")

    out_path = Path("outputs/debug/morphology_delta_scales.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "mvnx_path": str(args.mvnx_path),
        "start_frame": args.start_frame,
        "stride": args.stride,
        "rest_alignment_score": rest_score,
        "rest_alignment_errors_m": rest_errors,
        "role_scales": scales.role_scales,
        "measurements": scales.measurements,
    }, indent=2))

    print()
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S5-B0.3: quantify the "flicker" Prabin flagged visually.

Two independent signals from a qpos (T,36) sequence:
  1. joint jerk: third finite difference of the 29 actuated joint angles (rad/s^3).
  2. body linear-accel jerk: FK'd world position of left/right ankle + wrist bodies,
     third finite difference (m/s^3). Catches root/whole-chain snap that pure joint-jerk
     can miss if it's spread across several joints.

Usage (as a library, called by sprint_s5_smoothness_report.py or ad hoc):
    conda run -n gmr python scripts/g1/motion_smoothness.py \\
        --qpos outputs/gmr_baseline/sprint/ours_g1_corpus/walk1_subject1_ours.npz --fps 30
    conda run -n gmr python scripts/g1/motion_smoothness.py \\
        --pkl outputs/gmr_baseline/sprint/pkl/walk1_subject1_gmrfix.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402

FK_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_yaw_link", "right_wrist_yaw_link"]


def third_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Third finite difference along axis 0, same convention as a jerk estimate."""
    d1 = np.diff(x, axis=0) / dt
    d2 = np.diff(d1, axis=0) / dt
    d3 = np.diff(d2, axis=0) / dt
    return d3


def compute_smoothness(model, data, qpos: np.ndarray, fps: float) -> dict:
    dt = 1.0 / fps
    T = qpos.shape[0]

    # 1. joint jerk (29 actuated joints, qpos[7:36])
    joints = qpos[:, 7:36]
    jj = third_diff(joints, dt)  # (T-3, 29)
    joint_jerk_mag = np.linalg.norm(jj, axis=1)

    # 2. body-position jerk via FK
    body_ids = [model.body(n).id for n in FK_BODIES]
    pos = np.zeros((T, len(FK_BODIES), 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for i, bid in enumerate(body_ids):
            pos[t, i] = data.xpos[bid]
    body_jerk = np.stack([third_diff(pos[:, i], dt) for i in range(len(FK_BODIES))], axis=1)
    body_jerk_mag = np.linalg.norm(body_jerk, axis=2)  # (T-3, nbodies)

    return {
        "joint_jerk_mean": float(joint_jerk_mag.mean()),
        "joint_jerk_p95": float(np.percentile(joint_jerk_mag, 95)),
        "joint_jerk_max": float(joint_jerk_mag.max()),
        "body_jerk_mean": float(body_jerk_mag.mean()),
        "body_jerk_p95": float(np.percentile(body_jerk_mag, 95)),
        "body_jerk_max": float(body_jerk_mag.max()),
    }


def load_qpos(qpos_path, pkl_path, fps_arg):
    if pkl_path is not None:
        from load_gmr_pkl import load_gmr_pkl
        qpos, fps = load_gmr_pkl(pkl_path)
        return qpos, (fps_arg if fps_arg is not None else fps)
    z = np.load(qpos_path, allow_pickle=True)
    fps = fps_arg if fps_arg is not None else (float(z["fps"]) if "fps" in z else 30.0)
    return z["qpos"], fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qpos", type=Path)
    ap.add_argument("--pkl", type=Path)
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()
    assert (args.qpos is None) != (args.pkl is None), "pass exactly one of --qpos / --pkl"

    qpos, fps = load_qpos(args.qpos, args.pkl, args.fps)
    model, data, _, _ = load_g1_model_with_vetted_collision_and_floor()
    r = compute_smoothness(model, data, qpos, fps)
    src = args.qpos or args.pkl
    print(f"{src.name}  (fps={fps}, T={qpos.shape[0]})")
    print(f"  joint_jerk  mean={r['joint_jerk_mean']:8.1f}  p95={r['joint_jerk_p95']:8.1f}  max={r['joint_jerk_max']:8.1f}  rad/s^3")
    print(f"  body_jerk   mean={r['body_jerk_mean']:8.2f}  p95={r['body_jerk_p95']:8.2f}  max={r['body_jerk_max']:8.2f}  m/s^3")


if __name__ == "__main__":
    main()

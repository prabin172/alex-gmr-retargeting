#!/usr/bin/env python3
"""S5-A2: the new metrics this sprint needs, on top of S3's whole_clip_metrics/
held_metrics (imported, not mutated -- S3's script stays the frozen reference).

  joint_ok_pct: per frame with >=1 held foot, success = every currently-held foot
    has |support_z| < 3cm AND whole-body penetration < 5mm. The un-gameable
    headline metric (S4-T5's definition) -- a constant Z shift cannot satisfy both
    at once, unlike floorPen or held-frac3 alone (see planLogGMR.md S3's z-shift
    oracle kill).
  skate_cm: per held (contiguous-run) segment, max XY drift of that foot's body
    position from its value at segment onset. 0 = perfectly locked, no sliding.
  fidelity: mean position error (cm) + mean geodesic orientation error (deg)
    between a run's OWN achieved body FK and its OWN saved human-targets npz
    (--save_human_targets), over the NON-foot tracked bodies (everything in
    ik_match_table2 except the two ankle links). Compares tracking quality, not
    absolute human accuracy -- meant to catch contact-override regressions on
    bodies the override never touches.
  jerk: from motion_smoothness.py.

Usage (as a library):
    from sprint_s5_metrics import joint_ok_pct, skate_cm, fidelity_metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from post_process_ground_contactfirst import _robot_lowest_z  # noqa: E402
from stage_b_g1 import support_z  # noqa: E402
from motion_smoothness import compute_smoothness  # noqa: E402

# G1 body -> human bone, table2 (ik_configs/bvh_lafan1_to_g1.json), non-foot only.
FIDELITY_BODY_TO_BONE = {
    "pelvis": "Hips",
    "left_hip_yaw_link": "LeftUpLeg", "left_knee_link": "LeftLeg",
    "right_hip_yaw_link": "RightUpLeg", "right_knee_link": "RightLeg",
    "torso_link": "Spine2",
    "left_shoulder_yaw_link": "LeftArm", "left_elbow_link": "LeftForeArm",
    "left_wrist_yaw_link": "LeftHand",
    "right_shoulder_yaw_link": "RightArm", "right_elbow_link": "RightForeArm",
    "right_wrist_yaw_link": "RightHand",
}
FOOT_BODY = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}


def _held_segments(held_bool):
    """Contiguous True runs -> list of (start, end_exclusive)."""
    segs = []
    n = len(held_bool)
    i = 0
    while i < n:
        if held_bool[i]:
            j = i
            while j < n and held_bool[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def joint_ok_pct(model, data, mesh_cache, geom_ids, role_bid, held, qpos):
    """held: {"left_foot": bool array, "right_foot": bool array} (same keys as
    FOOT_POS_ROLE elsewhere in this project)."""
    T = qpos.shape[0]
    any_held = held["left_foot"] | held["right_foot"]
    frame_idx = np.where(any_held)[0]
    if frame_idx.size == 0:
        return float("nan"), 0
    n_ok = 0
    for t in frame_idx:
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        pen = max(0.0, -_robot_lowest_z(model, data, mesh_cache, geom_ids))
        ok = pen < 0.005
        if ok:
            for eff, role in (("left_foot", "left_ankle"), ("right_foot", "right_ankle")):
                if held[eff][t]:
                    sz = support_z(model, data, mesh_cache, role_bid[role])
                    if abs(sz) >= 0.03:
                        ok = False
                        break
        if ok:
            n_ok += 1
    return 100.0 * n_ok / frame_idx.size, int(frame_idx.size)


def skate_cm(model, data, held, qpos):
    """Per foot: mean and max, over all held segments, of max XY drift from the
    segment's own onset position (cm)."""
    out = {}
    for foot, body_name in FOOT_BODY.items():
        bid = model.body(body_name).id
        key = "left_foot" if foot == "left" else "right_foot"
        segs = _held_segments(held[key])
        drifts = []
        for (s, e) in segs:
            xy0 = None
            seg_max = 0.0
            for t in range(s, e):
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                xy = data.xpos[bid][:2].copy()
                if xy0 is None:
                    xy0 = xy
                seg_max = max(seg_max, float(np.linalg.norm(xy - xy0)))
            if xy0 is not None:
                drifts.append(seg_max)
        if drifts:
            out[foot] = dict(mean_cm=float(np.mean(drifts) * 100), max_cm=float(np.max(drifts) * 100),
                             n_segments=len(drifts))
        else:
            out[foot] = dict(mean_cm=float("nan"), max_cm=float("nan"), n_segments=0)
    return out


def fidelity_metrics(model, data, qpos, human_targets_npz_path):
    """mean position error (cm) + mean geodesic orientation error (deg) between
    achieved FK and the run's own saved human-targets, over FIDELITY_BODY_TO_BONE.
    One FK pass per frame (not per body) -- 12x fewer mj_forward calls."""
    z = np.load(human_targets_npz_path, allow_pickle=True)
    T = qpos.shape[0]
    items = [(model.body(bn).id, np.asarray(z[f"pos__{bone}"][:T]), np.asarray(z[f"rot__{bone}"][:T]))
             for bn, bone in FIDELITY_BODY_TO_BONE.items() if f"pos__{bone}" in z.files]
    pos_errs = []
    ori_errs = []
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for bid, tgt_pos_arr, tgt_quat_arr in items:
            achieved_pos = data.xpos[bid]
            achieved_quat = data.xquat[bid]  # MuJoCo: wxyz
            tgt_pos = tgt_pos_arr[t]
            tgt_quat = tgt_quat_arr[t]  # wxyz, GMR's own convention (mink.SO3 direct input)
            pos_errs.append(float(np.linalg.norm(achieved_pos - tgt_pos)))
            dot = float(np.clip(abs(np.dot(achieved_quat, tgt_quat)), -1.0, 1.0))
            ori_errs.append(2.0 * np.degrees(np.arccos(dot)))
    return dict(pos_err_cm=float(np.mean(pos_errs) * 100) if pos_errs else float("nan"),
               ori_err_deg=float(np.mean(ori_errs)) if ori_errs else float("nan"))


def jerk_metrics(model, data, qpos, fps):
    return compute_smoothness(model, data, qpos, fps)

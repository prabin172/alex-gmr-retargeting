#!/usr/bin/env python3
"""S9-T1: joint-limit-aware null-space repulsion, on top of T0's posture-
continuity bias (leg_floor_clamp.clamp_limb's `limit_margin`/`limit_weight`).

Targets the residual T0 finding: posture-continuity alone cleanly fixes
`sprint1_subject4`'s left_hip_yaw branch-flip but NOT right_ankle_pitch's
hard-limit bang-bang (t~6296-6306, flips between 0.5236 rad -- the joint's
exact upper limit -- and ~-0.5 to -0.8 rad, on a flat raw target).

Three variants, same shipped settings otherwise (max_dq=0.15,
avoid_self_collision=True, rate_limit=0.15, T6 localground on top):
  off        -- shipped `perframelimb_smrc_rl_localground` (byte-identical
                no-op check)
  posture    -- T0's posture_continuity=True, posture_weight=1.0 alone
  posture+lr -- posture_continuity=True PLUS limit_margin/limit_weight

Gate (per GMR-S9-plan.md T1): on sprint1_subject4's diagnosed window,
right_ankle_pitch stops alternating between its hard limit and a free
value; vMax improves further vs posture-only; worst_float's +9% cost from
posture-only should shrink, not grow. Checked on the S8-T0b 5 dev clips
(walk1_subject1, walk3_subject1, run2_subject1, ground1_subject1,
fallAndGetUp1_subject1) + sprint1_subject4 itself, so a fix for the
outlier clip isn't validated in isolation from the clips S8's gates were
built on.

Usage:
    conda run -n gmr python scripts/g1/sprint_s9_t1_limitrepel_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from smooth_heldaware import save_pkl  # noqa: E402
from polish_median_limbwise import _limbwise_pass, RAMP_FRAMES  # noqa: E402
from gmr_contact_retarget import FEET  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from sprint_s8_t6_localground import _envelope  # noqa: E402
from sprint_s8_t3_corpus import eval_one, CANON_DIR, HUMAN_TARGETS_DIR, PKL_S5_DIR  # noqa: E402

CLIPS = ["walk1_subject1", "walk3_subject1", "run2_subject1",
         "ground1_subject1", "fallAndGetUp1_subject1", "sprint1_subject4"]
DEV_DIR = REPO_ROOT / "outputs/gmr_baseline/dev"
WINDOW = range(6294, 6312)  # sprint1_subject4 only

VARIANTS = {
    "off":        dict(posture_continuity=False, posture_weight=1.0, limit_margin=0.0, limit_weight=0.0),
    "posture":    dict(posture_continuity=True,  posture_weight=1.0, limit_margin=0.0, limit_weight=0.0),
    "posture+lr": dict(posture_continuity=True,  posture_weight=1.0, limit_margin=0.2, limit_weight=3.0),
}


def build(model, data, mesh_cache, held_mask, clip, kwargs):
    src_sm = PKL_S5_DIR / f"{clip}_perframelimb_sm.pkl"
    qpos_sm, fps = load_gmr_pkl(src_sm)
    out = _limbwise_pass(model, data, mesh_cache, qpos_sm.copy(), held_mask, FEET,
                          RAMP_FRAMES, max_dq=0.15, avoid_self_collision=True,
                          rate_limit=0.15, **kwargs)
    return out, fps


def ground(model, data, mesh_cache, geom_ids, qpos, fps):
    lowest = np.zeros(qpos.shape[0])
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)
    required = np.maximum(0.0, -lowest)
    envelope = _envelope(required, fps)
    out = qpos.copy()
    out[:, 2] += envelope
    return out


def main():
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)

    all_qpos = {}  # (clip, tag) -> qpos
    print("=== building + evaluating ===")
    for clip in CLIPS:
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        held_mask, _ = compute_held_mask(canon)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}.npz"
        for tag, kwargs in VARIANTS.items():
            qpos, fps = build(model, data, mesh_cache, held_mask, clip, kwargs)
            qpos = ground(model, data, mesh_cache, geom_ids, qpos, fps)
            all_qpos[(clip, tag)] = qpos
            pkl_path = DEV_DIR / f"{clip}_s9t1_{tag.replace('+', '_')}.pkl"
            save_pkl(pkl_path, qpos, fps)
            row = eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held_mask,
                            human_targets_path, vmax_ctx, clip, tag, pkl_path)
            print(f"{clip:<25} [{tag:<11}] joint_ok={row['joint_ok_pct']:6.1f}% "
                  f"floorPen={row['floorPen_cm']:5.2f}cm coll={row['coll_pct']:5.2f}% "
                  f"worst_float={row['worst_float_cm']:6.2f}cm vMax={row['vMax_rad_s']:6.1f} "
                  f"n_spikes={row['n_spikes']:.0f}")

    print("\n=== sprint1_subject4 window t=6294..6311, off vs posture vs posture+lr ===")
    print(f"{'t':>6} | {'off_hipyaw':>10} {'off_anklep':>10} | "
          f"{'post_hipyaw':>11} {'post_anklep':>11} | "
          f"{'lr_hipyaw':>10} {'lr_anklep':>10}")
    q_off = all_qpos[("sprint1_subject4", "off")]
    q_post = all_qpos[("sprint1_subject4", "posture")]
    q_lr = all_qpos[("sprint1_subject4", "posture+lr")]
    for t in WINDOW:
        print(f"{t:>6} | {q_off[t,7+2]:>10.3f} {q_off[t,7+10]:>10.3f} | "
              f"{q_post[t,7+2]:>11.3f} {q_post[t,7+10]:>11.3f} | "
              f"{q_lr[t,7+2]:>10.3f} {q_lr[t,7+10]:>10.3f}")


if __name__ == "__main__":
    main()

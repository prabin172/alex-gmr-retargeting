#!/usr/bin/env python3
"""Dev probe (not S9, not shipped): quick test of leg_floor_clamp.py's new
`q_prev_chain`/`posture_weight` null-space posture-continuity bias (see that
module's clamp_limb docstring) against the branch-flip chatter found on
`sprint1_subject4` t~6296-6306 (right_ankle_pitch flipping between its own
hard joint limit and ~-0.5 to -0.8 rad, left_hip_yaw flipping between ~+0.9-
1.3 rad and ~-0.05 to -0.4 rad, frame to frame, while GMR's raw target is
flat there -- a solver branch-flip, not a real correction).

Runs `_limbwise_pass` on sprint1_subject4 twice from the SAME shipped input
(`sprint1_subject4_perframelimb_sm.pkl`, matching S8-T8's exact settings:
max_dq=0.15, avoid_self_collision=True, rate_limit=0.15), with
posture_continuity off (matches the shipped `smrc_rl` variant exactly) vs on,
then T6 local-grounding on both, then evals both + prints the raw joint
trajectory at the diagnosed window so the fix can be checked directly, not
just via the aggregate metrics.

Usage:
    conda run -n gmr python scripts/g1/posture_reg_probe.py
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
from sprint_s8_t3_corpus import eval_one, CANON_DIR, HUMAN_TARGETS_DIR  # noqa: E402

CLIP = "sprint1_subject4"
SRC_SM = REPO_ROOT / f"outputs/gmr_baseline/sprint/pkl_s5/{CLIP}_perframelimb_sm.pkl"
DEV_DIR = REPO_ROOT / "outputs/gmr_baseline/dev"
WINDOW = range(6294, 6312)


def build(model, data, mesh_cache, held_mask, posture_continuity):
    qpos_sm, fps = load_gmr_pkl(SRC_SM)
    out = _limbwise_pass(model, data, mesh_cache, qpos_sm.copy(), held_mask, FEET,
                          RAMP_FRAMES, max_dq=0.15, avoid_self_collision=True,
                          rate_limit=0.15, posture_continuity=posture_continuity,
                          posture_weight=1.0)
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
    canon = CANON_DIR / f"{CLIP}_lafan1c_grounded.npz"
    held_mask, _ = compute_held_mask(canon)

    variants = {}
    for tag, pc in [("off", False), ("on", True)]:
        qpos, fps = build(model, data, mesh_cache, held_mask, pc)
        qpos = ground(model, data, mesh_cache, geom_ids, qpos, fps)
        pkl_path = DEV_DIR / f"{CLIP}_probe_postcont_{tag}.pkl"
        save_pkl(pkl_path, qpos, fps)
        variants[tag] = (qpos, fps, pkl_path)
        print(f"built posture_continuity={pc} -> {pkl_path}")

    print("\n=== window t=6294..6311, off vs on (left_hip_yaw, right_ankle_pitch) ===")
    print(f"{'t':>6} {'raw_hipyaw':>11} {'raw_anklep':>11} | "
          f"{'off_hipyaw':>11} {'off_anklep':>11} | {'on_hipyaw':>10} {'on_anklep':>10}")
    qpos_raw, _ = load_gmr_pkl(REPO_ROOT / "outputs/gmr_baseline/sprint/pkl" / f"{CLIP}.pkl")
    qpos_off = variants["off"][0]
    qpos_on = variants["on"][0]
    for t in WINDOW:
        print(f"{t:>6} {qpos_raw[t,7+2]:>11.3f} {qpos_raw[t,7+10]:>11.3f} | "
              f"{qpos_off[t,7+2]:>11.3f} {qpos_off[t,7+10]:>11.3f} | "
              f"{qpos_on[t,7+2]:>10.3f} {qpos_on[t,7+10]:>10.3f}")

    print("\n=== aggregate metrics, off vs on ===")
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)
    human_targets_path = HUMAN_TARGETS_DIR / f"{CLIP}.npz"
    for tag in ("off", "on"):
        _, _, pkl_path = variants[tag]
        row = eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held_mask,
                        human_targets_path, vmax_ctx, CLIP, f"postcont_{tag}", pkl_path)
        print(f"[{tag}] joint_ok={row['joint_ok_pct']:.1f}% floorPen={row['floorPen_cm']:.2f}cm "
              f"coll={row['coll_pct']:.2f}% worst_float={row['worst_float_cm']:.2f}cm "
              f"vMax={row['vMax_rad_s']:.1f} n_spikes={row['n_spikes']} "
              f"fidelity_pos={row['fidelity_pos_err_cm']:.2f}cm")


if __name__ == "__main__":
    main()

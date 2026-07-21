#!/usr/bin/env python3
"""S9-T0-gate: raw-velocity-gated posture-continuity (`polish_median_limbwise.
_limbwise_pass`'s new `posture_gate_lo`/`posture_gate_hi`), answering the
open item S9-T0/T1 (planLogGMR.md) left: T0's blanket `posture_weight=1.0`
fixes `sprint1_subject4`'s branch-flip but regresses 4/5 of the S8-T0b dev
clips (worst_float up to +70% on walk3_subject1) because it pulls every
frame toward the previous frame's posture, including frames where the human
is genuinely moving and "the previous frame's posture" is simply wrong.

Mechanism: gate `posture_weight` per chain per frame by that chain's own
frame-to-frame delta in the TRUE, untouched GMR raw pkl (not this pass's own
smoothed/pre-clamped input, which already carries the branch-flip artifact
and would gate the wrong way) -- full weight when raw is near-static (delta
<= lo), zero when genuinely moving (delta >= hi), ramp between. Confirmed via
scratch analysis before picking thresholds: `sprint1_subject4`'s diagnosed
flat window (t=6294-6302) has raw chain deltas 0.002-0.017 rad/frame, while
ordinary walking clips run 0.03-0.16 rad/frame (p50) -- lo=0.02/hi=0.05 sits
below normal locomotion, only engaging on genuinely static/idle frames.

Five variants, same shipped settings otherwise (max_dq=0.15,
avoid_self_collision=True, rate_limit=0.15, T6 localground on top):
  off      -- shipped `perframelimb_smrc_rl_localground` (byte-identical
              no-op check)
  posture  -- T0's blanket posture_continuity=True, posture_weight=1.0
              (for reference -- the thing being fixed)
  gate1    -- attempt 1: posture_gate_lo=0.02, posture_gate_hi=0.05. Result:
              dev-clip regression collapsed to near-zero (4/5 clips exactly
              0.00 delta, ground1 +0.16cm/-0.61pp vs blanket's +8cm/-1.8pp),
              but only ~25% of the target clip's vMax win retained (46.1 vs
              blanket's 40.8 vs off's 47.9) -- the worst vMax frame itself
              (t=6306, raw delta 0.054-0.058) sits almost exactly AT the
              hi=0.05 cutoff, getting ~0 weight.
  gate2    -- attempt 2 (2-attempt cap, LAST try): posture_gate_lo=0.02,
              posture_gate_hi=0.065 -- a small, targeted widen specifically
              to give t=6306's own measured delta (0.054-0.058) partial
              weight (~20-45%), chosen to stay well below run2_subject1's
              p50 (0.161) and fallAndGetUp1's p50 (0.084) raw-delta so those
              two stay ~unaffected; walk1/walk3/ground1 (p50 0.030-0.058)
              get somewhat more exposure than gate1 -- the actual risk being
              tested here, per Prabin's call: if gate2 doesn't clearly beat
              gate1, drop back to gate1 as the final answer.

Gate (per GMR-S9-plan.md's 2-attempt cap): gated variant must keep T0's
target-clip win (sprint1_subject4 vMax improvement, hip_yaw flip fixed)
while NOT regressing the S8-T0b 5 dev clips' worst_float/joint_ok vs the
shipped `off` baseline (T0's own actual failure mode).

Usage:
    conda run -n gmr python scripts/g1/sprint_s9_t0gate_probe.py
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
DEV_CLIPS = CLIPS[:-1]  # S8-T0b's 5 -- the clips T0's regression showed up on
DEV_DIR = REPO_ROOT / "outputs/gmr_baseline/dev"
RAW_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"  # pristine GMR raw, pre-Phase-A
WINDOW = range(6294, 6312)  # sprint1_subject4 only

VARIANTS = {
    "off":     dict(posture_continuity=False, posture_weight=1.0,
                     posture_gate_lo=None, posture_gate_hi=None),
    "posture": dict(posture_continuity=True, posture_weight=1.0,
                     posture_gate_lo=None, posture_gate_hi=None),
    "gate1":   dict(posture_continuity=True, posture_weight=1.0,
                     posture_gate_lo=0.02, posture_gate_hi=0.05),
    "gate2":   dict(posture_continuity=True, posture_weight=1.0,
                     posture_gate_lo=0.02, posture_gate_hi=0.065),
}


def build(model, data, mesh_cache, held_mask, clip, kwargs):
    src_sm = PKL_S5_DIR / f"{clip}_perframelimb_sm.pkl"
    qpos_sm, fps = load_gmr_pkl(src_sm)
    kwargs = dict(kwargs)
    if kwargs.get("posture_gate_lo") is not None:
        raw_qpos, _ = load_gmr_pkl(RAW_PKL_DIR / f"{clip}.pkl")
        assert raw_qpos.shape[0] == qpos_sm.shape[0], \
            f"{clip}: raw T={raw_qpos.shape[0]} != perframelimb_sm T={qpos_sm.shape[0]}"
        kwargs["raw_gate_qpos"] = raw_qpos
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

    rows = {}  # (clip, tag) -> row dict
    all_qpos = {}
    print("=== building + evaluating ===")
    for clip in CLIPS:
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        held_mask, _ = compute_held_mask(canon)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}.npz"
        for tag, kwargs in VARIANTS.items():
            qpos, fps = build(model, data, mesh_cache, held_mask, clip, kwargs)
            qpos = ground(model, data, mesh_cache, geom_ids, qpos, fps)
            all_qpos[(clip, tag)] = qpos
            pkl_path = DEV_DIR / f"{clip}_s9t0gate_{tag}.pkl"
            save_pkl(pkl_path, qpos, fps)
            row = eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held_mask,
                            human_targets_path, vmax_ctx, clip, tag, pkl_path)
            rows[(clip, tag)] = row
            print(f"{clip:<25} [{tag:<8}] joint_ok={row['joint_ok_pct']:6.1f}% "
                  f"floorPen={row['floorPen_cm']:5.2f}cm coll={row['coll_pct']:5.2f}% "
                  f"worst_float={row['worst_float_cm']:6.2f}cm vMax={row['vMax_rad_s']:6.1f} "
                  f"n_spikes={row['n_spikes']:.0f}")

    for gtag in ("gate1", "gate2"):
        print(f"\n=== gate check: {gtag} vs off, S8-T0b 5 dev clips (must not regress) ===")
        for clip in DEV_CLIPS:
            r_off = rows[(clip, "off")]
            r_gate = rows[(clip, gtag)]
            d_float = r_gate["worst_float_cm"] - r_off["worst_float_cm"]
            d_joint = r_gate["joint_ok_pct"] - r_off["joint_ok_pct"]
            print(f"{clip:<25} worst_float off={r_off['worst_float_cm']:6.2f} "
                  f"{gtag}={r_gate['worst_float_cm']:6.2f} (d={d_float:+.2f}) | "
                  f"joint_ok off={r_off['joint_ok_pct']:5.1f} {gtag}={r_gate['joint_ok_pct']:5.1f} "
                  f"(d={d_joint:+.2f})")

    print("\n=== sprint1_subject4 target-clip win check: off vs posture vs gate1 vs gate2 ===")
    for tag in ("off", "posture", "gate1", "gate2"):
        r = rows[("sprint1_subject4", tag)]
        print(f"[{tag:<8}] vMax={r['vMax_rad_s']:6.1f} worst_float={r['worst_float_cm']:6.2f}cm "
              f"joint_ok={r['joint_ok_pct']:5.1f}%")

    print("\n=== sprint1_subject4 window t=6294..6311, off vs posture vs gate1 vs gate2 ===")
    print(f"{'t':>6} | {'off_hy':>8} {'off_ap':>8} | {'post_hy':>8} {'post_ap':>8} | "
          f"{'g1_hy':>8} {'g1_ap':>8} | {'g2_hy':>8} {'g2_ap':>8}")
    q_off = all_qpos[("sprint1_subject4", "off")]
    q_post = all_qpos[("sprint1_subject4", "posture")]
    q_g1 = all_qpos[("sprint1_subject4", "gate1")]
    q_g2 = all_qpos[("sprint1_subject4", "gate2")]
    for t in WINDOW:
        print(f"{t:>6} | {q_off[t,7+2]:>8.3f} {q_off[t,7+10]:>8.3f} | "
              f"{q_post[t,7+2]:>8.3f} {q_post[t,7+10]:>8.3f} | "
              f"{q_g1[t,7+2]:>8.3f} {q_g1[t,7+10]:>8.3f} | "
              f"{q_g2[t,7+2]:>8.3f} {q_g2[t,7+10]:>8.3f}")


if __name__ == "__main__":
    main()

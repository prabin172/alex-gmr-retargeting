#!/usr/bin/env python3
"""S8-T0b: spike attribution for the ~5 worst clips by n_spikes for
gmr_contact_fc and perframelimb. For each spike frame, classifies the cause:
  (a) clamp activation toggling (clamp was OFF at t-1, ON at t, or vice-versa)
  (b) phase-2 self-collision correction overpowering phase-1 (frame has high ncon)
  (c) perframe root-lift discontinuity (large diff in the lift curve)
  (d) held-release ramp interaction (held[eff] transitions at spike frame)

Method: for each spike frame (max |dq|*fps > 60 rad/s), we instrument:
  - which joint(s) carry the spike
  - whether the clamp is active (compare qpos to the raw pkl at that frame)
  - whether the frame has high self-collision contact count
  - whether held state changes at that frame
  - (perframelimb only) whether the root-lift diff is large at that frame

Output: one summary table per variant.

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t0b_spike_attribution.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY  # noqa: E402

BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
SMOOTH_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_smoothness.csv"

SPIKE_RAD_PER_S = 60.0  # matches eval_ihmc_json.py

# Top 5 worst clips per variant (from smoothness CSV, by n_spikes desc)
WORST_CLIPS = {
    "gmr_contact_fc": [
        "obstacles6_subject5",    # 155
        "fallAndGetUp1_subject4", # 69
        "obstacles5_subject2",    # 24
        "walk3_subject3",         # 24
        "fallAndGetUp2_subject2", # 20
    ],
    "perframelimb": [
        "obstacles4_subject3",    # 7
        "walk2_subject3",         # 6
        "obstacles5_subject3",    # 5
        "aiming1_subject4",       # 4
        "pushAndFall1_subject4",  # 3
    ],
}

SUFFIX = {
    "gmr_contact_fc": "_gmrcontact_fc",
    "perframelimb": "_perframelimb",
}


def get_ncon_at_frame(model, data, qpos_t, floor_gid):
    """Return total non-floor self-collision contact count at frame t."""
    data.qpos[:] = qpos_t
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    count = 0
    for cc in range(data.ncon):
        ct = data.contact[cc]
        if ct.geom1 == floor_gid or ct.geom2 == floor_gid:
            continue
        b1 = int(model.geom_bodyid[ct.geom1])
        b2 = int(model.geom_bodyid[ct.geom2])
        if b1 == 0 or b2 == 0:
            continue
        count += 1
    return count


def attribute_spike(t, qpos_variant, qpos_raw, fps, held, model, data,
                    floor_gid, variant, lift_curve=None, joint_names=None):
    """Classify a spike at frame t. Returns dict of cause flags."""
    T = qpos_variant.shape[0]
    dq_variant = np.abs(np.diff(qpos_variant[:, 7:], axis=0)) * fps  # (T-1, nj)
    spike_joints = np.where(dq_variant[t - 1] > SPIKE_RAD_PER_S)[0] if t > 0 else []
    max_vel = float(dq_variant[t - 1].max()) if t > 0 else 0.0

    # Which joints spike?
    spiking = [(int(j), float(dq_variant[t - 1, j]),
                joint_names[j] if joint_names and j < len(joint_names) else f"j{j}")
               for j in spike_joints]
    spiking.sort(key=lambda x: -x[1])

    # (a) Clamp toggling: did the clamp correction change sign between t-1 and t?
    # Proxy: |delta_variant - delta_raw| is large at the spike joint(s)
    delta_variant_t = (qpos_variant[t] - qpos_variant[t - 1])[7:] if t > 0 else None
    delta_raw_t = (qpos_raw[t] - qpos_raw[t - 1])[7:] if t > 0 else None
    clamp_correction = None
    if delta_variant_t is not None and delta_raw_t is not None:
        clamp_correction = delta_variant_t - delta_raw_t  # variant's extra delta vs raw
    cause_a = False
    if clamp_correction is not None and len(spike_joints) > 0:
        # Check if clamp contribution at t is large in opposite direction to t-1
        if t >= 2:
            cc_prev = (qpos_variant[t - 1] - qpos_variant[t - 2])[7:] - \
                      (qpos_raw[t - 1] - qpos_raw[t - 2])[7:]
        else:
            cc_prev = np.zeros_like(clamp_correction)
        for j in spike_joints:
            # Toggle: correction was near zero or opposite sign at t-1
            if abs(cc_prev[j]) < 0.01 * fps and abs(clamp_correction[j]) > 0.01 * fps:
                cause_a = True
            # Or: correction flipped sign
            if cc_prev[j] * clamp_correction[j] < -1e-6:
                cause_a = True

    # (b) High ncon at spike frame (phase-2 self-collision overpowering)
    ncon_t = get_ncon_at_frame(model, data, qpos_variant[t], floor_gid)
    ncon_prev = get_ncon_at_frame(model, data, qpos_variant[t - 1], floor_gid) if t > 0 else 0
    # Also check the RAW ncon for comparison
    ncon_raw_t = get_ncon_at_frame(model, data, qpos_raw[t], floor_gid)
    cause_b = ncon_t > 5  # more than 5 self-collision contacts is "high"

    # (c) Root-lift discontinuity (perframelimb only)
    lift_diff = None
    cause_c = False
    if lift_curve is not None and t > 0:
        lift_diff = float(abs(lift_curve[t] - lift_curve[t - 1]))
        cause_c = lift_diff > 0.02  # > 2cm/frame in the lift curve

    # (d) Held-release at spike frame
    cause_d = False
    held_transition_effs = []
    for eff, hmask in held.items():
        if t > 0 and t < len(hmask):
            prev_held = bool(hmask[t - 1])
            curr_held = bool(hmask[t])
            if prev_held != curr_held:
                cause_d = True
                held_transition_effs.append(f"{eff}:{'on' if curr_held else 'off'}")

    return {
        "t": t,
        "max_vel": max_vel,
        "spike_joints": spiking[:2],  # top 2
        "ncon": ncon_t,
        "ncon_prev": ncon_prev,
        "ncon_raw": ncon_raw_t,
        "lift_diff_cm": (lift_diff * 100 if lift_diff is not None else None),
        "held_transitions": held_transition_effs,
        "cause_a": cause_a,
        "cause_b": cause_b,
        "cause_c": cause_c,
        "cause_d": cause_d,
    }


def compute_lift_curve(qpos_raw, qpos_variant):
    """Reconstruct the effective per-frame root lift as qpos_variant[:,2] - qpos_raw[:,2]."""
    return qpos_variant[:, 2] - qpos_raw[:, 2]


def main():
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    # Get joint names from model
    joint_names = []
    for j in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        joint_names.append(jname if jname else f"j{j}")
    # qpos joint names = all joints with qposadr >= 7 (skip root free joint's 7 scalars)
    # The qpos[:, 7:] corresponds to actuated joints (29 DOFs)
    # joint names for qpos[:,7:] = joints starting from index 1 (0 = free-joint)
    act_joint_names = []
    for j in range(1, model.njnt):  # skip free joint at index 0
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        act_joint_names.append(jname if jname else f"j{j}")

    all_results = {}

    for variant in ["gmr_contact_fc", "perframelimb"]:
        suffix = SUFFIX[variant]
        clips = WORST_CLIPS[variant]
        print(f"\n{'='*60}")
        print(f"VARIANT: {variant}")
        print(f"{'='*60}")

        variant_results = []
        for clip in clips:
            grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
            pkl_path = PKL_S5_DIR / f"{clip}{suffix}.pkl"
            raw_pkl_path = GMR_PKL_DIR / f"{clip}.pkl"

            if not pkl_path.exists():
                print(f"  SKIP {clip}: {pkl_path} missing")
                continue
            if not raw_pkl_path.exists():
                print(f"  SKIP {clip}: raw pkl missing")
                continue
            if not grounded.exists():
                print(f"  SKIP {clip}: grounded canonical missing")
                continue

            qpos_v, fps = load_gmr_pkl(pkl_path)
            qpos_r, _ = load_gmr_pkl(raw_pkl_path)
            held, _ = compute_held_mask(grounded)

            # Find spike frames
            dq_v = np.abs(np.diff(qpos_v[:, 7:], axis=0)) * fps
            spike_mask = dq_v.max(axis=1) > SPIKE_RAD_PER_S
            spike_frames = np.where(spike_mask)[0] + 1  # t=1..T-1 (frame index where spike appears)
            n_spikes = len(spike_frames)
            print(f"\n  {clip}: {n_spikes} spikes (frames: {spike_frames[:10].tolist()})")

            lift_curve = None
            if variant == "perframelimb":
                # Align lengths (sometimes variant has same T as raw)
                minT = min(qpos_v.shape[0], qpos_r.shape[0])
                lift_curve = compute_lift_curve(qpos_r[:minT], qpos_v[:minT])

            clip_attrs = []
            for t in spike_frames:
                if t >= qpos_v.shape[0]:
                    continue
                attr = attribute_spike(
                    t, qpos_v, qpos_r, fps, held, model, data, floor_gid,
                    variant, lift_curve=lift_curve, joint_names=act_joint_names
                )
                clip_attrs.append(attr)
                sj_str = ", ".join(f"{n}({v:.0f})" for _, v, n in attr["spike_joints"])
                lc_str = f" lift_diff={attr['lift_diff_cm']:.1f}cm" if attr["lift_diff_cm"] is not None else ""
                hd_str = f" held_trans={attr['held_transitions']}" if attr["held_transitions"] else ""
                causes = "".join([
                    "A" if attr["cause_a"] else "-",
                    "B" if attr["cause_b"] else "-",
                    "C" if attr["cause_c"] else "-",
                    "D" if attr["cause_d"] else "-",
                ])
                print(f"    t={t:5d}: vMax={attr['max_vel']:6.1f} ncon={attr['ncon']:3d} "
                      f"ncon_raw={attr['ncon_raw']:3d} causes={causes} "
                      f"joints=[{sj_str}]{lc_str}{hd_str}")

            variant_results.append({"clip": clip, "n_spikes": n_spikes, "attrs": clip_attrs})

        all_results[variant] = variant_results

    # Summary table: cause counts across all 5 clips per variant
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    for variant, results in all_results.items():
        total_spikes = sum(r["n_spikes"] for r in results)
        total_a = sum(1 for r in results for a in r["attrs"] if a["cause_a"])
        total_b = sum(1 for r in results for a in r["attrs"] if a["cause_b"])
        total_c = sum(1 for r in results for a in r["attrs"] if a["cause_c"])
        total_d = sum(1 for r in results for a in r["attrs"] if a["cause_d"])
        # Multi-cause: a frame can have multiple causes
        total_analyzed = sum(len(r["attrs"]) for r in results)
        # Worst magnitude
        all_attrs = [a for r in results for a in r["attrs"]]
        worst_mag = max((a["max_vel"] for a in all_attrs), default=0.0)
        worst_mag_a = max((a["max_vel"] for a in all_attrs if a["cause_a"]), default=0.0)
        worst_mag_b = max((a["max_vel"] for a in all_attrs if a["cause_b"]), default=0.0)
        worst_mag_c = max((a["max_vel"] for a in all_attrs if a["cause_c"]), default=0.0)
        worst_mag_d = max((a["max_vel"] for a in all_attrs if a["cause_d"]), default=0.0)

        print(f"\nVariant: {variant} | total_spikes_in_worst5={total_spikes} | analyzed={total_analyzed}")
        print(f"  Cause | count | %  | worst_mag rad/s")
        if total_analyzed > 0:
            for label, cnt, wm in [
                ("A: clamp toggle",  total_a, worst_mag_a),
                ("B: phase2 coll",   total_b, worst_mag_b),
                ("C: lift disc",     total_c, worst_mag_c),
                ("D: held ramp",     total_d, worst_mag_d),
            ]:
                pct = 100.0 * cnt / total_analyzed
                print(f"  {label:<18} {cnt:5d}  {pct:5.1f}%  {wm:8.1f}")

        # Per-clip breakdown
        print(f"  Per-clip:")
        for r in results:
            clip_a = sum(1 for a in r["attrs"] if a["cause_a"])
            clip_b = sum(1 for a in r["attrs"] if a["cause_b"])
            clip_c = sum(1 for a in r["attrs"] if a["cause_c"])
            clip_d = sum(1 for a in r["attrs"] if a["cause_d"])
            print(f"    {r['clip']:<35} n={r['n_spikes']:4d} A={clip_a} B={clip_b} C={clip_c} D={clip_d}")


if __name__ == "__main__":
    main()

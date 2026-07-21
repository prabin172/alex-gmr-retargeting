#!/usr/bin/env python3
"""S8-T2 gate eval: held-aware smoothing (smooth_heldaware.py) on the 10-clip
gate set (5 dev + 5 perframelimb worst-spike clips), both arms.

Builds:
  pkl_s5/{clip}_perframelimb_sm.pkl    (ours + smoother)
  pkl_s5/{clip}_heightfix_sm.pkl       (GMR-full + smoother, fairness arm)
  pkl_s5/{clip}_perframelimb_smrc.pkl  (S8-T2c: smooth -> re-clamp, limbs only)

Evaluates all axes needed for the T2 gate:
  - Smoothness: joint_jerk, body_jerk, vMax, n_spikes (vs gmr_raw gate)
  - Contact/contact-preservation: joint_ok, floorPen, coll_pct, worst_float,
    skate_mean (vs unsmoothed perframelimb gate)
  - Fidelity: ori_err_deg (vs gmr_raw + 3 gate)

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t2_gate.py --build
    conda run -n gmr python scripts/g1/sprint_s8_t2_gate.py --eval
    conda run -n gmr python scripts/g1/sprint_s8_t2_gate.py --build --eval
    conda run -n gmr python scripts/g1/sprint_s8_t2_gate.py --build-smrc --eval
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY, FOOT_POS_ROLE, whole_clip_metrics  # noqa: E402
from sprint_s5_metrics import joint_ok_pct, skate_cm, fidelity_metrics, jerk_metrics  # noqa: E402
from sprint_s6_range_summary import compute_held_mask, clip_worst_float_pen  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from smooth_heldaware import smooth_heldaware, save_pkl  # noqa: E402
from gmr_contact_retarget import compute_held_masks, FEET  # noqa: E402
from polish_median_limbwise import _limbwise_pass, RAMP_FRAMES  # noqa: E402
from sprint_s8_t2d_repair import repair_clip  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from eval_ihmc_json import evaluate as eval_ihmc_evaluate  # noqa: E402

CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
HUMAN_TARGETS_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/human_targets"

GATE_CLIPS = [
    # 5 dev clips
    "walk1_subject1", "walk3_subject1", "run2_subject1",
    "ground1_subject1", "fallAndGetUp1_subject1",
    # 5 perframelimb worst-spike clips (T0b table)
    "obstacles4_subject3", "walk2_subject3", "obstacles5_subject3",
    "aiming1_subject4", "pushAndFall1_subject4",
]

# class labels from s1t4_reclass.csv, baked here for the 10 gate clips
CLIP_CLASS = {
    "walk1_subject1": "loco",
    "walk3_subject1": "loco",
    "run2_subject1": "loco",
    "ground1_subject1": "floor",
    "fallAndGetUp1_subject1": "floor",
    "obstacles4_subject3": "loco",
    "walk2_subject3": "loco",
    "obstacles5_subject3": "loco",
    "aiming1_subject4": "loco",
    "pushAndFall1_subject4": "loco",
}


def do_build(lambda_track=1.0, lambda_smooth=20.0, force=False):
    print(f"[T2 build] lambda_track={lambda_track}  lambda_smooth={lambda_smooth}  force={force}")
    for clip in GATE_CLIPS:
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not canon.exists():
            print(f"  SKIP {clip} (no canonical)")
            continue

        held, T = compute_held_masks(canon, FEET)

        # --- perframelimb_sm ---
        src_pfl = PKL_S5_DIR / f"{clip}_perframelimb.pkl"
        dst_pfl = PKL_S5_DIR / f"{clip}_perframelimb_sm.pkl"
        if dst_pfl.exists() and not force:
            print(f"  SKIP (done) {clip} perframelimb_sm")
        elif src_pfl.exists():
            print(f"  smoothing {clip} perframelimb -> perframelimb_sm ...")
            qpos, fps = load_gmr_pkl(src_pfl)
            out = smooth_heldaware(qpos, held, lambda_track=lambda_track,
                                   lambda_smooth=lambda_smooth, fps=fps)
            save_pkl(dst_pfl, out, fps)
            print(f"    saved {dst_pfl.name}")
        else:
            print(f"  SKIP {clip} perframelimb (source pkl missing)")

        # --- heightfix_sm ---
        src_hf = GMR_PKL_DIR / f"{clip}_gmrfix.pkl"
        dst_hf = PKL_S5_DIR / f"{clip}_heightfix_sm.pkl"
        if dst_hf.exists() and not force:
            print(f"  SKIP (done) {clip} heightfix_sm")
        elif src_hf.exists():
            print(f"  smoothing {clip} heightfix -> heightfix_sm ...")
            qpos, fps = load_gmr_pkl(src_hf)
            out = smooth_heldaware(qpos, held, lambda_track=lambda_track,
                                   lambda_smooth=lambda_smooth, fps=fps)
            save_pkl(dst_hf, out, fps)
            print(f"    saved {dst_hf.name}")
        else:
            print(f"  SKIP {clip} gmrfix (source pkl missing)")


def do_build_smrc(force=False, max_dq=None):
    """S8-T2c: re-clamp the perframelimb_sm (attempt-2, spike-unlock) pkls --
    limbs only (phase-1 floor/held + phase-2 self-collision via
    polish_median_limbwise._limbwise_pass), NO root re-lift (center=none
    equivalent -- the smoothed root is trusted as-is). Each frame's DLS starts
    from that frame's own (already-smooth) input, same per-frame-independent
    mechanism perframelimb itself uses -- not a sequential warm start, but the
    input is temporally clean so corrections should be small.

    max_dq (T2c attempt 2, diagnosed 2026-07-18): attempt 1 (max_dq=None)
    produced vMax up to 94.2 rad/s -- direct instrumentation on
    walk2_subject3 t=6318 showed left_hip_yaw (dof 2, limits +/-1.57rad)
    bouncing 1.57 -> 1.57 -> -1.57 -> 1.57 across consecutive frames: the
    uncapped DLS repeatedly snapping to the opposite joint limit, the SAME
    near-singular full-extension divergence class already documented in
    polish_median_limbwise.py's `--center perframe` docstring (S7-T3) --
    perframelimb's own build sits leg DOFs near their limits far more often
    than gmr_raw (4224 near-limit (frame,dof) pairs in the smoothed input
    here), so re-clamping without the SAME trust region that mode already
    uses elsewhere in this codebase (0.15 rad/frame) hits the identical bug.
    Fix: pass max_dq through to clamp_limb, reusing the established value."""
    print(f"[T2c build] re-clamp perframelimb_sm -> perframelimb_smrc "
          f"(limbs only, avoid_self_collision=True, max_dq={max_dq}, force={force})")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    for clip in GATE_CLIPS:
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not canon.exists():
            print(f"  SKIP {clip} (no canonical)")
            continue
        src = PKL_S5_DIR / f"{clip}_perframelimb_sm.pkl"
        dst = PKL_S5_DIR / f"{clip}_perframelimb_smrc.pkl"
        if dst.exists() and not force:
            print(f"  SKIP (done) {clip} perframelimb_smrc")
            continue
        if not src.exists():
            print(f"  SKIP {clip} (source perframelimb_sm missing)")
            continue
        held, T = compute_held_masks(canon, FEET)
        qpos, fps = load_gmr_pkl(src)
        assert T == qpos.shape[0], f"canonical T={T} != qpos T={qpos.shape[0]}"
        print(f"  re-clamping {clip} perframelimb_sm -> perframelimb_smrc ...")
        qpos = _limbwise_pass(model, data, mesh_cache, qpos, held, FEET,
                               RAMP_FRAMES, max_dq=max_dq,
                               avoid_self_collision=True, rate_limit=None)
        save_pkl(dst, qpos, fps)
        print(f"    saved {dst.name}")


def do_build_repair(force=False, window_only=True):
    """S8-T2d: local spike repair on unsmoothed perframelimb (no global
    smoothing) -- see sprint_s8_t2d_repair.py for the mechanism."""
    print(f"[T2d build] local spike repair perframelimb -> perframelimb_repair "
          f"(window_only={window_only}, force={force})")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    for clip in GATE_CLIPS:
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not canon.exists():
            print(f"  SKIP {clip} (no canonical)")
            continue
        src = PKL_S5_DIR / f"{clip}_perframelimb.pkl"
        dst = PKL_S5_DIR / f"{clip}_perframelimb_repair.pkl"
        if dst.exists() and not force:
            print(f"  SKIP (done) {clip} perframelimb_repair")
            continue
        if not src.exists():
            print(f"  SKIP {clip} (source perframelimb missing)")
            continue
        held, T = compute_held_masks(canon, FEET)
        qpos, fps = load_gmr_pkl(src)
        assert T == qpos.shape[0], f"canonical T={T} != qpos T={qpos.shape[0]}"
        print(f"  repairing {clip} ...")
        qpos = repair_clip(qpos, held, fps, model, data, mesh_cache, window_only=window_only)
        save_pkl(dst, qpos, fps)
        print(f"    saved {dst.name}")


def do_eval():
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)

    # Variants: (name, dir, suffix)
    VARIANTS = [
        ("gmr_raw", GMR_PKL_DIR, ""),
        ("gmr_heightfix", GMR_PKL_DIR, "_gmrfix"),
        ("perframelimb", PKL_S5_DIR, "_perframelimb"),
        ("perframelimb_sm", PKL_S5_DIR, "_perframelimb_sm"),
        ("heightfix_sm", PKL_S5_DIR, "_heightfix_sm"),
        ("perframelimb_smrc", PKL_S5_DIR, "_perframelimb_smrc"),
        ("perframelimb_repair", PKL_S5_DIR, "_perframelimb_repair"),
    ]

    rows = []
    for clip in GATE_CLIPS:
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not canon.exists():
            print(f"  SKIP {clip} (no canonical)")
            continue
        held, T = compute_held_mask(canon)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}_human_targets.npz"
        cls = CLIP_CLASS.get(clip, "?")

        for vname, vdir, vsuffix in VARIANTS:
            pkl_path = vdir / f"{clip}{vsuffix}.pkl"
            if not pkl_path.exists():
                print(f"  MISSING {clip} {vname}")
                continue

            qpos, fps = load_gmr_pkl(pkl_path)
            # Jerk
            jm = jerk_metrics(model, data, qpos, fps)
            # Skate
            sk = skate_cm(model, data, held, qpos)
            # Fidelity
            if human_targets_path.exists():
                fm = fidelity_metrics(model, data, qpos, human_targets_path)
            else:
                fm = {"pos_err_cm": float("nan"), "ori_err_deg": float("nan")}
            # vMax / n_spikes
            (vmodel, vdata, vmesh_cache, vgeom_ids, vjoint_names, vq_lo, vq_hi) = vmax_ctx
            vr = eval_ihmc_evaluate(pkl_path.stem, qpos, fps, {}, vmodel, vdata, vmesh_cache,
                                     vgeom_ids, {}, vq_lo, vq_hi, vjoint_names)
            # Contact metrics
            wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
            jok, n_held = joint_ok_pct(model, data, mesh_cache, geom_ids, role_bid, held, qpos)
            res = clip_worst_float_pen(model, data, mesh_cache, role_bid, held, qpos)
            wf = res[0] if res is not None else float("nan")
            wp = res[1] if res is not None else float("nan")

            row = {
                "clip": clip, "class": cls, "variant": vname,
                "joint_jerk_mean": jm["joint_jerk_mean"],
                "body_jerk_mean": jm["body_jerk_mean"],
                "vMax_rad_s": vr["vel_max_rad_s"],
                "n_spikes": vr["n_spikes"],
                "skate_left_mean_cm": sk["left"]["mean_cm"],
                "skate_right_mean_cm": sk["right"]["mean_cm"],
                "fidelity_ori_err_deg": fm["ori_err_deg"],
                "joint_ok_pct": jok,
                "floorPen_cm": wm["floorPen_cm"],
                "coll_pct": wm["coll_pct"],
                "worst_float_cm": wf,
            }
            rows.append(row)
            print(f"  {clip} {vname}: jerk={jm['joint_jerk_mean']:.0f} "
                  f"body_jerk={jm['body_jerk_mean']:.0f} "
                  f"vMax={vr['vel_max_rad_s']:.1f} spk={vr['n_spikes']} "
                  f"skL={sk['left']['mean_cm']:.3f} jok={jok:.2f}% fp={wm['floorPen_cm']:.3f}cm")

    # Print class-mean table
    _print_class_means(rows)
    return rows


def _class_mean(rows, variant, cls, key, nanfn=np.nanmean):
    vals = [r[key] for r in rows if r["variant"] == variant and r["class"] == cls]
    return nanfn(vals) if vals else float("nan")


def _print_class_means(rows):
    VARIANTS = ["gmr_raw", "gmr_heightfix", "perframelimb", "perframelimb_sm",
                "heightfix_sm", "perframelimb_smrc", "perframelimb_repair"]
    AXES = ["joint_jerk_mean", "body_jerk_mean", "vMax_rad_s", "n_spikes",
            "skate_left_mean_cm", "skate_right_mean_cm", "fidelity_ori_err_deg",
            "joint_ok_pct", "floorPen_cm", "coll_pct", "worst_float_cm"]
    for cls in ["floor", "loco"]:
        print(f"\n=== class={cls} ===")
        header = f"{'metric':<25}" + "".join(f"{v:>18}" for v in VARIANTS)
        print(header)
        for ax in AXES:
            vals = [_class_mean(rows, v, cls, ax) for v in VARIANTS]
            line = f"{ax:<25}" + "".join(f"{x:>18.3f}" if not np.isnan(x) else f"{'nan':>18}" for x in vals)
            print(line)

    results = {}
    if any(r["variant"] == "perframelimb_sm" for r in rows):
        results["perframelimb_sm"] = _gate_check(rows, "perframelimb_sm")
    if any(r["variant"] == "perframelimb_smrc" for r in rows):
        results["perframelimb_smrc"] = _gate_check(rows, "perframelimb_smrc")
    if any(r["variant"] == "perframelimb_repair" for r in rows):
        # R2.2 (approved 2026-07-18): joint_jerk demoted to report-only, 1.75x raw ceiling
        results["perframelimb_repair"] = _gate_check(rows, "perframelimb_repair",
                                                       jerk_gate=False, jerk_ceiling_mult=1.75)
    return results


def _gate_check(all_rows, variant, jerk_gate=True, jerk_ceiling_mult=1.3):
    """T2 gate check for `variant` vs gmr_raw (smoothness axes) and vs
    perframelimb (contact-preservation axes). class-mean over all clips
    passed in (combined, not per-class).

    jerk_gate=False (R2.2, approved 2026-07-18): joint_jerk becomes
    report-only against a jerk_ceiling_mult*raw sanity ceiling instead of a
    gating check -- use this ONLY when Prabin has approved the demotion for
    the phase being run (T2d). T2c's own gate stays jerk_gate=True (the
    ORIGINAL unmodified T2 gate) per the plan."""
    def m(key, var=variant):
        return np.nanmean([r[key] for r in all_rows if r["variant"] == var])

    raw_jerk, raw_bjerk = m("joint_jerk_mean", "gmr_raw"), m("body_jerk_mean", "gmr_raw")
    raw_vmax, raw_spk = m("vMax_rad_s", "gmr_raw"), m("n_spikes", "gmr_raw")
    raw_skl, raw_skr = m("skate_left_mean_cm", "gmr_raw"), m("skate_right_mean_cm", "gmr_raw")
    raw_ori = m("fidelity_ori_err_deg", "gmr_raw")

    v_jerk, v_bjerk = m("joint_jerk_mean"), m("body_jerk_mean")
    v_vmax, v_spk = m("vMax_rad_s"), m("n_spikes")
    v_skl, v_skr = m("skate_left_mean_cm"), m("skate_right_mean_cm")
    v_ori, v_jok = m("fidelity_ori_err_deg"), m("joint_ok_pct")
    v_fp, v_coll, v_wf = m("floorPen_cm"), m("coll_pct"), m("worst_float_cm")

    pfl_jok, pfl_fp = m("joint_ok_pct", "perframelimb"), m("floorPen_cm", "perframelimb")
    pfl_coll, pfl_wf = m("coll_pct", "perframelimb"), m("worst_float_cm", "perframelimb")

    jerk_label = (f"joint_jerk ≤ {jerk_ceiling_mult}× raw" +
                  ("" if jerk_gate else " (report-only, R2.2)"))
    checks = [
        (jerk_label, v_jerk, "≤", jerk_ceiling_mult * raw_jerk, jerk_gate),
        ("n_spikes ≤ 0.5/clip", v_spk, "≤", 0.5, True),
        ("vMax ≤ 1.2× raw", v_vmax, "≤", 1.2 * raw_vmax, True),
        ("body_jerk ≤ 1.3× raw", v_bjerk, "≤", 1.3 * raw_bjerk, True),
        ("joint_ok within 1.0pt of pfl", v_jok, "≥", pfl_jok - 1.0, True),
        ("floorPen within 0.5cm of pfl", v_fp, "≤", pfl_fp + 0.5, True),
        ("coll_pct within 0.05 of pfl", v_coll, "≤", pfl_coll + 0.05, True),
        ("worst_float within 1.0cm of pfl", v_wf, "≤", pfl_wf + 1.0, True),
        ("skate_mean(L) ≤ 2× raw", v_skl, "≤", 2.0 * raw_skl, True),
        ("skate_mean(R) ≤ 2× raw", v_skr, "≤", 2.0 * raw_skr, True),
        ("ori_fidelity ≤ raw+3deg", v_ori, "≤", raw_ori + 3.0, True),
    ]

    print(f"\n=== T2 GATE CHECK ({variant}, class-mean over all clips combined) ===")
    all_pass = True
    for label, val, op, threshold, gating in checks:
        if np.isnan(val):
            print(f"  N/A   {label}: no data for {variant}")
            continue
        passed = (val <= threshold) if op == "≤" else (val >= threshold)
        if gating and not passed:
            all_pass = False
        tag = "" if gating else " [report-only]"
        status = "PASS" if passed else "FAIL"
        print(f"  {status}{tag}  {label}: got {val:.3f} (threshold {threshold:.3f})")

    print(f"\n  GATE ({variant}): {'ALL PASS' if all_pass else 'FAILED'}")
    return all_pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--build-smrc", action="store_true",
                     help="S8-T2c: re-clamp perframelimb_sm -> perframelimb_smrc (limbs only)")
    ap.add_argument("--smrc-max-dq", type=float, default=None,
                     help="S8-T2c attempt 2: trust region (rad/frame) for the re-clamp DLS")
    ap.add_argument("--build-repair", action="store_true",
                     help="S8-T2d: local spike repair perframelimb -> perframelimb_repair")
    ap.add_argument("--repair-whole-clip", action="store_true",
                     help="S8-T2d attempt 1 (kept for the record): re-clamp the whole "
                          "clip instead of window-only (attempt 2, default)")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--lambda-track", type=float, default=1.0)
    ap.add_argument("--lambda-smooth", type=float, default=20.0)
    ap.add_argument("--force", action="store_true", help="rebuild pkls even if they exist")
    args = ap.parse_args()
    if not args.build and not args.build_smrc and not args.build_repair and not args.eval:
        ap.error("pass --build and/or --build-smrc and/or --build-repair and/or --eval")
    if args.build:
        do_build(lambda_track=args.lambda_track, lambda_smooth=args.lambda_smooth, force=args.force)
    if args.build_smrc:
        do_build_smrc(force=args.force, max_dq=args.smrc_max_dq)
    if args.build_repair:
        do_build_repair(force=args.force, window_only=not args.repair_whole_clip)
    if args.eval:
        do_eval()


if __name__ == "__main__":
    main()

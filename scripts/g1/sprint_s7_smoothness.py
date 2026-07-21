#!/usr/bin/env python3
"""S7-T1: smoothness/skate/fidelity eval for S6's shipped variants -- the eval
hole S6's own tables never filled (s6_full_corpus.csv has no jerk column at all).
Reuses S5/S6 library code unchanged: motion_smoothness.compute_smoothness,
sprint_s5_metrics.{skate_cm, fidelity_metrics}, sprint_s6_range_summary's
held-mask/path conventions. No new mechanism, eval only.

  --dev     5 S6 dev clips x {gmr_raw, gmr_polished, gmr_contact, gmr_contact_fc,
            medianlimb} (+ stacked on walk1/ground1 where it exists). Prints a
            table, does not write a CSV (planLogGMR.md ## S7-T1a is the record).
  --build   no-op placeholder (nothing to build -- all inputs already exist from
            S1/S5/S6); kept for symmetry with sprint_s6_corpus.py's --build/--eval
            pattern. Present so this script's usage matches its siblings.
  --eval    all 77 clips x {gmr_raw, gmr_polished, gmr_contact, gmr_contact_fc,
            medianlimb} -> s7_smoothness.csv. Resumable: skips rows whose (clip,
            variant) already appear in an existing CSV.

Usage:
    conda run -n gmr python scripts/g1/sprint_s7_smoothness.py --dev
    conda run -n gmr python scripts/g1/sprint_s7_smoothness.py --eval
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY, FOOT_POS_ROLE  # noqa: E402
from sprint_s5_metrics import skate_cm, fidelity_metrics, jerk_metrics  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from eval_ihmc_json import evaluate as eval_ihmc_evaluate  # noqa: E402

GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
HUMAN_TARGETS_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/human_targets"
BVH_DIR = REPO_ROOT / "data/raw/lafan1"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_smoothness.csv"

DEV_CLIPS = ["walk1_subject1", "walk3_subject1", "run2_subject1",
             "ground1_subject1", "fallAndGetUp1_subject1"]

VARIANT_SPECS = [
    ("gmr_raw", GMR_PKL_DIR, ""),
    ("gmr_polished", GMR_PKL_DIR, "_polished"),
    ("gmr_contact", PKL_S5_DIR, "_gmrcontact"),
    ("gmr_contact_fc", PKL_S5_DIR, "_gmrcontact_fc"),
    ("medianlimb", PKL_S5_DIR, "_medianlimb"),
]
STACKED_CLIPS = {"walk1_subject1", "ground1_subject1"}


def variant_path(vname, subdir, suffix, clip):
    return subdir / f"{clip}{suffix}.pkl"


def eval_one(model, data, mesh_cache, role_bid, clip, vname, pkl_path, held,
             human_targets_path, vmax_ctx):
    qpos, fps = load_gmr_pkl(pkl_path)
    jm = jerk_metrics(model, data, qpos, fps)
    sk = skate_cm(model, data, held, qpos)
    if human_targets_path.exists():
        fm = fidelity_metrics(model, data, qpos, human_targets_path)
    else:
        fm = dict(pos_err_cm=float("nan"), ori_err_deg=float("nan"))
    (vmodel, vdata, vmesh_cache, vgeom_ids, vjoint_names, vq_lo, vq_hi) = vmax_ctx
    vr = eval_ihmc_evaluate(pkl_path.stem, qpos, fps, {}, vmodel, vdata, vmesh_cache,
                             vgeom_ids, {}, vq_lo, vq_hi, vjoint_names)
    return dict(
        clip=clip, variant=vname,
        joint_jerk_mean=jm["joint_jerk_mean"], joint_jerk_p95=jm["joint_jerk_p95"],
        body_jerk_mean=jm["body_jerk_mean"], body_jerk_p95=jm["body_jerk_p95"],
        skate_left_mean_cm=sk["left"]["mean_cm"], skate_left_max_cm=sk["left"]["max_cm"],
        skate_right_mean_cm=sk["right"]["mean_cm"], skate_right_max_cm=sk["right"]["max_cm"],
        fidelity_pos_err_cm=fm["pos_err_cm"], fidelity_ori_err_deg=fm["ori_err_deg"],
        vMax_rad_s=vr["vel_max_rad_s"], vP95_rad_s=vr["vel_p95_rad_s"],
        n_spikes=vr["n_spikes"],
    )


def do_dev():
    model, data, _, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)

    rows = []
    for clip in DEV_CLIPS:
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        held, _ = compute_held_mask(grounded)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}.npz"
        specs = list(VARIANT_SPECS)
        if clip in STACKED_CLIPS:
            specs = specs + [("stacked", PKL_S5_DIR, "_stacked")]
        for vname, subdir, suffix in specs:
            p = variant_path(vname, subdir, suffix, clip)
            if not p.exists():
                print(f"SKIP {clip} {vname}: {p} missing")
                continue
            row = eval_one(model, data, mesh_cache, role_bid, clip, vname, p, held,
                            human_targets_path, vmax_ctx)
            rows.append(row)
            print(f"{clip:<26}{vname:<16}"
                  f"jjerk={row['joint_jerk_mean']:8.1f} bjerk={row['body_jerk_mean']:7.2f} "
                  f"skateL={row['skate_left_mean_cm']:6.2f} skateR={row['skate_right_mean_cm']:6.2f} "
                  f"fidPos={row['fidelity_pos_err_cm']:6.2f}cm fidOri={row['fidelity_ori_err_deg']:6.2f}deg "
                  f"vMax={row['vMax_rad_s']:6.1f} vP95={row['vP95_rad_s']:6.1f} spikes={row['n_spikes']:5}")

    # %delta vs gmr_raw, per clip, for joint_jerk_mean and body_jerk_mean
    raw = {r["clip"]: r for r in rows if r["variant"] == "gmr_raw"}
    print("\n-- %% delta vs gmr_raw (jerk) --")
    for r in rows:
        if r["variant"] == "gmr_raw" or r["clip"] not in raw:
            continue
        rj, rb = raw[r["clip"]]["joint_jerk_mean"], raw[r["clip"]]["body_jerk_mean"]
        djj = 100.0 * (r["joint_jerk_mean"] - rj) / rj if rj else float("nan")
        dbj = 100.0 * (r["body_jerk_mean"] - rb) / rb if rb else float("nan")
        print(f"{r['clip']:<26}{r['variant']:<16} joint_jerk {djj:+7.1f}%  body_jerk {dbj:+7.1f}%")
    return rows


def do_eval():
    model, data, _, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)

    done = set()
    rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for r in csv.DictReader(f):
                done.add((r["clip"], r["variant"]))
                rows.append({k: (float(v) if k not in ("clip", "variant") else v)
                             for k, v in r.items()})

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not grounded.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP {clip} (no grounded canonical)")
            continue
        held, _ = compute_held_mask(grounded)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}.npz"
        n_new = 0
        for vname, subdir, suffix in VARIANT_SPECS:
            if (clip, vname) in done:
                continue
            p = variant_path(vname, subdir, suffix, clip)
            if not p.exists():
                continue
            row = eval_one(model, data, mesh_cache, role_bid, clip, vname, p, held,
                            human_targets_path, vmax_ctx)
            rows.append(row)
            n_new += 1
        print(f"[{i+1}/{len(bvhs)}] {clip}: {n_new} new rows")
        if n_new:
            cols = list(rows[0].keys())
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT_CSV, "w") as f:
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")

    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.dev:
        do_dev()
    if args.eval:
        do_eval()
    if not args.dev and not args.eval:
        ap.error("pass --dev and/or --eval")


if __name__ == "__main__":
    main()

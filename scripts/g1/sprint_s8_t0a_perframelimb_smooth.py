#!/usr/bin/env python3
"""S8-T0a: add perframelimb smoothness rows to s7_smoothness.csv.
Pure eval — no new pkls generated. Resumable: skips (clip, 'perframelimb')
pairs already present. Follows the same pattern as sprint_s7_fcsm_corpus.py's
smoothness-append block.

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t0a_perframelimb_smooth.py
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
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY  # noqa: E402
from sprint_s5_metrics import skate_cm, fidelity_metrics, jerk_metrics  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from eval_ihmc_json import evaluate as eval_ihmc_evaluate  # noqa: E402

BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
HUMAN_TARGETS_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/human_targets"
SMOOTH_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_smoothness.csv"

VARIANT = "perframelimb"
SUFFIX = "_perframelimb"


def main():
    model, data, _, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)

    # Load existing CSV, find already-done pairs
    existing = []
    done = set()
    if SMOOTH_CSV.exists():
        with open(SMOOTH_CSV) as f:
            for r in csv.DictReader(f):
                existing.append(r)
                done.add((r["clip"], r["variant"]))

    scols = list(existing[0].keys()) if existing else [
        "clip", "variant",
        "joint_jerk_mean", "joint_jerk_p95",
        "body_jerk_mean", "body_jerk_p95",
        "skate_left_mean_cm", "skate_left_max_cm",
        "skate_right_mean_cm", "skate_right_max_cm",
        "fidelity_pos_err_cm", "fidelity_ori_err_deg",
        "vMax_rad_s", "vP95_rad_s", "n_spikes",
    ]

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    new_rows = []
    for i, clip in enumerate(bvhs):
        if (clip, VARIANT) in done:
            print(f"[{i+1}/{len(bvhs)}] SKIP (done) {clip}")
            continue
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        pkl_path = PKL_S5_DIR / f"{clip}{SUFFIX}.pkl"
        if not grounded.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP {clip} (no grounded canonical)")
            continue
        if not pkl_path.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP {clip} (no perframelimb pkl)")
            continue

        held, _ = compute_held_mask(grounded)
        qpos, fps = load_gmr_pkl(pkl_path)
        jm = jerk_metrics(model, data, qpos, fps)
        sk = skate_cm(model, data, held, qpos)
        ht = HUMAN_TARGETS_DIR / f"{clip}.npz"
        if ht.exists():
            fm = fidelity_metrics(model, data, qpos, ht)
        else:
            fm = dict(pos_err_cm=float("nan"), ori_err_deg=float("nan"))
        (vmodel, vdata, vmesh_cache, vgeom_ids, vjoint_names, vq_lo, vq_hi) = vmax_ctx
        vr = eval_ihmc_evaluate(pkl_path.stem, qpos, fps, {}, vmodel, vdata, vmesh_cache,
                                 vgeom_ids, {}, vq_lo, vq_hi, vjoint_names)

        row = dict(
            clip=clip, variant=VARIANT,
            joint_jerk_mean=jm["joint_jerk_mean"], joint_jerk_p95=jm["joint_jerk_p95"],
            body_jerk_mean=jm["body_jerk_mean"], body_jerk_p95=jm["body_jerk_p95"],
            skate_left_mean_cm=sk["left"]["mean_cm"], skate_left_max_cm=sk["left"]["max_cm"],
            skate_right_mean_cm=sk["right"]["mean_cm"], skate_right_max_cm=sk["right"]["max_cm"],
            fidelity_pos_err_cm=fm["pos_err_cm"], fidelity_ori_err_deg=fm["ori_err_deg"],
            vMax_rad_s=vr["vel_max_rad_s"], vP95_rad_s=vr["vel_p95_rad_s"],
            n_spikes=vr["n_spikes"],
        )
        new_rows.append(row)
        print(f"[{i+1}/{len(bvhs)}] {clip}: jjerk={row['joint_jerk_mean']:.1f} "
              f"vMax={row['vMax_rad_s']:.1f} spikes={row['n_spikes']}")

        # Incremental write after each clip
        with open(SMOOTH_CSV, "w") as f:
            f.write(",".join(scols) + "\n")
            for r in existing:
                f.write(",".join(str(r[c]) for c in scols) + "\n")
            for r in new_rows:
                f.write(",".join(str(r[c]) for c in scols) + "\n")

    print(f"\nAdded {len(new_rows)} perframelimb rows to {SMOOTH_CSV}")


if __name__ == "__main__":
    main()

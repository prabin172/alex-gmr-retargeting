#!/usr/bin/env python3
"""S8-T7: relax tracking / raise smoothing regularization in the smrc build,
then re-run T6 local grounding on top. Tests Prabin's hypothesis (2026-07-18,
post-T6): lose a bit more on tracking fidelity in exchange for gains on the
two axes T6 couldn't touch (n_spikes, vMax) -- while floorPen/coll_pct/
worst_float/joint_ok stay protected, because they're enforced downstream
(smrc's re-clamp step, then T6's envelope) rather than by the smoothing
weights themselves.

`smrc` = smooth_heldaware.py's stage_a (tridiagonal smoother, weights
lambda_track vs lambda_smooth) -> polish_median_limbwise._limbwise_pass
(re-clamp, restores floor/collision safety the smoothing pass may have
perturbed). Current corpus default (sprint_s8_t3_corpus.py do_build):
lambda_track=1.0, lambda_smooth=20.0 (smooth_heldaware.py's own defaults).
This script sweeps both down/up respectively, producing tagged variants:

  pkl_s5/{clip}_perframelimb_sm_{tag}.pkl      (smoothed, pre-reclamp)
  pkl_s5/{clip}_perframelimb_smrc_{tag}.pkl    (re-clamped)
  pkl_s5/{clip}_smrc_{tag}_localground.pkl     (T6 envelope on top)

variant name in the eval CSV: perframelimb_smrc_{tag}_localground

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t7_smooth_sweep.py \\
        --tag relaxA --lambda-track 0.5 --lambda-smooth 40 --build --ground --eval
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
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from smooth_heldaware import smooth_heldaware, save_pkl  # noqa: E402
from gmr_contact_retarget import compute_held_masks, FEET  # noqa: E402
from polish_median_limbwise import _limbwise_pass, RAMP_FRAMES  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from sprint_s8_t6_localground import _envelope  # noqa: E402
from sprint_s8_t3_corpus import (  # noqa: E402
    eval_one, load_class_map, BVH_DIR, CANON_DIR, PKL_S5_DIR, HUMAN_TARGETS_DIR, OUT_CSV,
)


def do_build(tag, lambda_track, lambda_smooth, force=False):
    print(f"[T7 build:{tag}] lambda_track={lambda_track} lambda_smooth={lambda_smooth} force={force}")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        src_pfl = PKL_S5_DIR / f"{clip}_perframelimb.pkl"
        dst_sm = PKL_S5_DIR / f"{clip}_perframelimb_sm_{tag}.pkl"
        dst_smrc = PKL_S5_DIR / f"{clip}_perframelimb_smrc_{tag}.pkl"
        if not canon.exists() or not src_pfl.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (missing canonical or perframelimb pkl)")
            continue
        held, T = compute_held_masks(canon, FEET)

        if dst_sm.exists() and not force:
            qpos_sm, fps = load_gmr_pkl(dst_sm)
        else:
            qpos, fps = load_gmr_pkl(src_pfl)
            qpos_sm = smooth_heldaware(qpos, held, lambda_track=lambda_track,
                                        lambda_smooth=lambda_smooth, fps=fps)
            save_pkl(dst_sm, qpos_sm, fps)

        if dst_smrc.exists() and not force:
            print(f"[{i + 1}/{len(bvhs)}] SKIP (done) {clip}")
            continue
        held_mask, _ = compute_held_mask(canon)
        out = _limbwise_pass(model, data, mesh_cache, qpos_sm, held_mask, FEET,
                              RAMP_FRAMES, max_dq=0.15, avoid_self_collision=True,
                              rate_limit=None)
        save_pkl(dst_smrc, out, fps)
        print(f"[{i + 1}/{len(bvhs)}] {clip}: smrc_{tag} saved")


def do_ground(tag, force=False):
    print(f"[T7 ground:{tag}]")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        src = PKL_S5_DIR / f"{clip}_perframelimb_smrc_{tag}.pkl"
        dst = PKL_S5_DIR / f"{clip}_smrc_{tag}_localground.pkl"
        if not src.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (no smrc_{tag} pkl)")
            continue
        if dst.exists() and not force:
            print(f"[{i + 1}/{len(bvhs)}] SKIP (done) {clip}")
            continue
        qpos, fps = load_gmr_pkl(src)
        lowest = np.zeros(qpos.shape[0])
        for t in range(qpos.shape[0]):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)
        required = np.maximum(0.0, -lowest)
        envelope = _envelope(required, fps)
        out = qpos.copy()
        out[:, 2] += envelope
        save_pkl(dst, out, fps)
        print(f"[{i + 1}/{len(bvhs)}] {clip}: peak envelope={envelope.max() * 100:.2f}cm")


def do_eval(tag):
    variant = f"perframelimb_smrc_{tag}_localground"
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)
    class_map = load_class_map()

    done = set()
    rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for r in csv.DictReader(f):
                done.add((r["clip"], r["variant"]))
                rows.append({k: (float(v) if k not in ("clip", "variant", "class") else v)
                             for k, v in r.items()})

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    n_new = 0
    for i, clip in enumerate(bvhs):
        if (clip, variant) in done:
            continue
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        p = PKL_S5_DIR / f"{clip}_smrc_{tag}_localground.pkl"
        if not canon.exists() or not p.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (missing canonical or pkl)")
            continue
        held, _ = compute_held_mask(canon)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}.npz"
        row = eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held,
                        human_targets_path, vmax_ctx, clip, variant, p)
        row["class"] = class_map.get(clip, "?")
        rows.append(row)
        n_new += 1
        print(f"[{i + 1}/{len(bvhs)}] {clip}: row added")

    if n_new:
        cols = list(rows[0].keys())
        with open(OUT_CSV, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\n{n_new} new rows. Total {len(rows)} rows in {OUT_CSV}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lambda-track", type=float, default=1.0)
    ap.add_argument("--lambda-smooth", type=float, default=20.0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ground", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.build or args.ground or args.eval):
        ap.error("pass --build and/or --ground and/or --eval")
    if args.build:
        do_build(args.tag, args.lambda_track, args.lambda_smooth, force=args.force)
    if args.ground:
        do_ground(args.tag, force=args.force)
    if args.eval:
        do_eval(args.tag)


if __name__ == "__main__":
    main()

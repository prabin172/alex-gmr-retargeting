#!/usr/bin/env python3
"""S8-T8: rate-limited re-clamp + T6 local grounding on top.

T7 (planLogGMR.md ## S8-T7) root-caused vMax/n_spikes to the smrc re-clamp
step's own per-frame DLS correction, not the smoothing weights -- relaxing
tracking just gives the re-clamp more work, raising jerk instead of
lowering it. The matching lever is `_limbwise_pass`'s `rate_limit` param
(`CorrectionRateLimiter`, S8-T1b): caps the re-clamp's applied per-frame
correction directly instead of letting it swing freely within its max_dq
trust region each iteration.

T1b already tested this exact mechanism once (pre-T6, pre-held-aware-
smoothing, applied to the ORIGINAL perframelimb clamp, not the smrc
re-clamp) and found it converts spikes into drift (float/range/skate all
regressed). Prabin's call (2026-07-18): try it again in the current
pipeline -- T6's local grounding runs downstream of this clamp now, which
didn't exist during T1b's test, so it's an open question whether the drift
lands somewhere T6 can absorb it.

Pipeline: perframelimb_sm.pkl (standard lambda_track=1.0/lambda_smooth=20,
already built corpus-wide) -> _limbwise_pass with rate_limit=R (instead of
T3's unlimited re-clamp) -> perframelimb_smrc_rl.pkl -> T6 envelope on top
-> perframelimb_smrc_rl_localground.pkl.

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t8_rl_localground.py \\
        --rate 0.15 --build --ground --eval
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
from smooth_heldaware import save_pkl  # noqa: E402
from polish_median_limbwise import _limbwise_pass, RAMP_FRAMES  # noqa: E402
from gmr_contact_retarget import FEET  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from sprint_s8_t6_localground import _envelope  # noqa: E402
from sprint_s8_t3_corpus import (  # noqa: E402
    eval_one, load_class_map, BVH_DIR, CANON_DIR, PKL_S5_DIR, HUMAN_TARGETS_DIR, OUT_CSV,
)


def do_build(rate, tag, force=False):
    print(f"[T8 build:{tag}] rate_limit={rate} rad/frame")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        src_sm = PKL_S5_DIR / f"{clip}_perframelimb_sm.pkl"
        dst_smrc = PKL_S5_DIR / f"{clip}_perframelimb_smrc_{tag}.pkl"
        if not canon.exists() or not src_sm.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (missing canonical or perframelimb_sm pkl)")
            continue
        if dst_smrc.exists() and not force:
            print(f"[{i + 1}/{len(bvhs)}] SKIP (done) {clip}")
            continue
        held_mask, _ = compute_held_mask(canon)
        qpos_sm, fps = load_gmr_pkl(src_sm)
        out = _limbwise_pass(model, data, mesh_cache, qpos_sm, held_mask, FEET,
                              RAMP_FRAMES, max_dq=0.15, avoid_self_collision=True,
                              rate_limit=rate)
        save_pkl(dst_smrc, out, fps)
        print(f"[{i + 1}/{len(bvhs)}] {clip}: smrc_{tag} saved")


def do_ground(tag, force=False):
    print(f"[T8 ground:{tag}]")
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
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--tag", default="rl")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ground", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.build or args.ground or args.eval):
        ap.error("pass --build and/or --ground and/or --eval")
    if args.build:
        do_build(args.rate, args.tag, force=args.force)
    if args.ground:
        do_ground(args.tag, force=args.force)
    if args.eval:
        do_eval(args.tag)


if __name__ == "__main__":
    main()

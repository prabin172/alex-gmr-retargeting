#!/usr/bin/env python3
"""S10: full 77-clip corpus build + eval for the global-Stage-1 redesign
(2026-07-20 session): GMR raw -> ONE whole-trajectory floor+self-collision
QP (`stage_b`, self-collision scoring only -- see below) -> existing
held-aware smoothing (Stage 2, unchanged) -> existing local grounding
(unchanged). Replaces the entire per-frame clamp -> smooth -> re-clamp ->
rate-limit -> posture-gate chain (S6-S9) with two stages instead of five.

Locked-in config, from single-clip probes (`sprint_s10_globalto_probe.py`,
same session):
  - Anchors: human-timed (`compute_held_masks`, canonical human's own
    contact labels -- WHEN) with sub-segmented stillness runs (robot's own
    IK-point speed -- avoids locking a whole human-labeled interval to one
    XY through real repositioning motion) and a RIGID per-robot Z constant
    (`compute_z_support`, computed once at qpos0 -- WHERE), not a per-run
    data-dependent median. Matches the already-shipped
    `gmr_contact_retarget.ContactAwareGMR` mechanism's own design.
  - `count_floor=False` (self-collision QP rows only, floor rows OFF):
    found on `ground1_subject1` that scoring Stage 1's keep-best against raw
    floor penetration is self-defeating -- `stage_b`'s decision variable is
    actuated joints ONLY (root frozen), so it structurally cannot lower the
    pelvis to fix a crawl-class floor violation; every candidate looked
    catastrophic pre-grounding (17-23cm), so keep-best always preferred
    "do nothing" even over genuinely-improving self-collision fixes. Floor
    is `localground`'s job regardless (confirmed: floorPen_cm == 0.00 in
    EVERY test regardless of Stage 1's own outcome) -- removing it from
    Stage 1's scoring unblocked self-collision convergence that was
    previously impossible (ground1_subject1: coll_pct 2.57%->0.19%,
    joint_ok_pct 57.3%->69.5%, both previously stuck at the pass-0 no-op).
  - `lambda_coll=0.5` (leg_floor_clamp's own shipped default), `n_outer=4`,
    `trust=0.15`.

Not corpus-scale validated yet -- 3 clips tested individually
(sprint1_subject4: moderate difficulty, Stage 1 converges;
walk1_subject1: easy, Stage 1 correctly no-ops safely; ground1_subject1:
severe crawl, Stage 1 now converges after the count_floor fix). This build
is the first corpus-scale test.

Known, measured trade-off (not yet resolved, same on all 3 clips): universal
win on joint_jerk_mean (3-11x lower) and vMax_rad_s (2.5-5.5x lower) vs BOTH
gmr_heightfix and shipped `perframelimb_smrc_pg_localground`, zero spikes,
zero floor penetration everywhere -- but joint_ok_pct/coll_pct are worse than
shipped on every clip tested (shipped's per-frame Newton solve, 10 iters x
every frame, is simply more thorough than a handful of global SCA outer
steps). vs gmr_heightfix specifically, average contact quality (avg_float/
avg_range, NOT the worst-case single-frame stat) is a clear win (ground1:
12.94cm avg range -> 2.64cm), though a worst-case outlier frame can still
be as bad as heightfix's own worst case -- not yet root-caused which
frame/mechanism drives that outlier.

Usage:
    conda run -n gmr python scripts/g1/sprint_s10_globalto_full_corpus.py --build
    conda run -n gmr python scripts/g1/sprint_s10_globalto_full_corpus.py --eval
    conda run -n gmr python scripts/g1/sprint_s10_globalto_full_corpus.py --table
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from smooth_heldaware import save_pkl, smooth_heldaware  # noqa: E402
from gmr_contact_retarget import compute_held_masks, FEET  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY  # noqa: E402
from sprint_s8_t3_corpus import eval_one, load_class_map, HUMAN_TARGETS_DIR, OUT_CSV  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from sprint_s10_globalto_probe import run_stage_b_global, ground  # noqa: E402

BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S10_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s10"

VARIANT = "globalto_coll_localground"
LAMBDA_COLL = 0.5
N_OUTER = 4
TRUST = 0.15


def do_build(force=False, clip_match=None):
    print(f"[S10 corpus build] variant={VARIANT} lambda_coll={LAMBDA_COLL} "
          f"n_outer={N_OUTER} trust={TRUST}")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    if clip_match:
        bvhs = [c for c in bvhs if clip_match in c]
    PKL_S10_DIR.mkdir(parents=True, exist_ok=True)

    for i, clip in enumerate(bvhs):
        i1 = i + 1
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        raw_pkl = GMR_PKL_DIR / f"{clip}.pkl"
        out_pkl = PKL_S10_DIR / f"{clip}_{VARIANT}.pkl"
        if not canon.exists() or not raw_pkl.exists():
            print(f"[{i1}/{len(bvhs)}] SKIP {clip} (missing canonical or raw pkl)")
            continue
        if out_pkl.exists() and not force:
            print(f"[{i1}/{len(bvhs)}] SKIP (done) {clip}")
            continue

        t0 = time.time()
        qpos_in, fps = load_gmr_pkl(raw_pkl)
        try:
            qpos_b = run_stage_b_global(
                model, data, mesh_cache, floor_gid, qpos_in, fps,
                LAMBDA_COLL, N_OUTER, TRUST,
                count_floor=False,  # see module docstring -- Stage 1 cannot fix
                                     # floor pen (root frozen), scoring against it
                                     # blocks self-collision fixes it CAN make;
                                     # localground fixes floor regardless.
                human_contact=True, canon=canon)
            held, T = compute_held_masks(canon, FEET)
            assert T == qpos_b.shape[0], f"canonical T={T} != qpos T={qpos_b.shape[0]}"
            qpos_sm = smooth_heldaware(qpos_b, held, fps=fps)
            qpos_g = ground(model, data, mesh_cache, geom_ids, qpos_sm, fps)
            save_pkl(out_pkl, qpos_g, fps)
            print(f"[{i1}/{len(bvhs)}] {clip}: OK ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"[{i1}/{len(bvhs)}] {clip}: FAILED ({time.time()-t0:.0f}s) -- {e!r}")


def do_eval():
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
        if (clip, VARIANT) in done:
            continue
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        p = PKL_S10_DIR / f"{clip}_{VARIANT}.pkl"
        if not canon.exists() or not p.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (missing canonical or pkl)")
            continue
        held, _ = compute_held_mask(canon)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}.npz"
        row = eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held,
                        human_targets_path, vmax_ctx, clip, VARIANT, p)
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


def print_table():
    VARIANTS = ["gmr_heightfix", "perframelimb_smrc_pg_localground", VARIANT]
    AXES = ["joint_ok_pct", "floorPen_cm", "coll_pct", "worst_float_cm",
            "range_cm", "fidelity_pos_err_cm", "joint_jerk_mean",
            "skate_left_mean_cm", "vMax_rad_s", "n_spikes"]
    rows = []
    with open(OUT_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    class_map = load_class_map()
    for cls in ["floor", "loco"]:
        print(f"\n=== class={cls} ===")
        header = f"{'metric':<22}" + "".join(f"{v:>32}" for v in VARIANTS)
        print(header)
        for ax in AXES:
            vals = []
            for v in VARIANTS:
                xs = [float(r[ax]) for r in rows if r["variant"] == v
                      and class_map.get(r["clip"], "?") == cls and r.get(ax) not in (None, "", "nan")]
                vals.append(np.mean(xs) if xs else float("nan"))
            line = f"{ax:<22}" + "".join(f"{x:>32.3f}" if not np.isnan(x) else f"{'nan':>32}" for x in vals)
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--clip-match", default=None, help="only build clips whose name contains this substring")
    args = ap.parse_args()
    if not (args.build or args.eval or args.table):
        ap.error("pass --build and/or --eval and/or --table")
    if args.build:
        do_build(force=args.force, clip_match=args.clip_match)
    if args.eval:
        do_eval()
    if args.table:
        print_table()


if __name__ == "__main__":
    main()

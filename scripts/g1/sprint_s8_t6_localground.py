#!/usr/bin/env python3
"""S8-T6: local (windowed) grounding on top of the S8 winner (perframelimb_smrc).

T5 (planLogGMR.md ## S8-T5) proved a GLOBAL constant per-clip z-shift zeroes
floorPen and beats heightfix on worst_float, at zero cost to coll/vMax/
spikes/jerk/skate/fidelity_ori -- but collapses joint_ok_pct (97.9->32.7%
floor, 98.6->50.8% loco), because the shift is sized to the clip's single
worst (usually transient) frame and then overshoots the +/-3cm stance-frame
band on every OTHER (already-clean) held frame.

T6 fixes this by making the shift LOCAL: a per-frame envelope that is zero
almost everywhere and only rises near frames that actually penetrate, so
stance frames far from a penetration event are untouched. Same windowed-
repair philosophy as T2d (sprint_s8_t2d_repair.py), applied to root z
instead of leg-DOF angles.

Envelope construction (guarantees zero residual penetration by
construction, not by tuning):
  1. required[t] = max(0, -lowest_z[t])  -- the exact per-frame shift that
     alone would clear frame t.
  2. widened = maximum_filter1d(required, window=2*RAMP_HALF+1) -- widens
     each spike into a plateau. A maximum filter can only INCREASE values,
     so widened[t] >= required[t] everywhere, trivially (the window always
     includes t itself).
  3. smoothed = gaussian_filter1d(widened, sigma=RAMP_SIGMA) -- rounds the
     plateau's sharp corners so the envelope (and thus root z velocity) is
     C-infinity smooth. Gaussian smoothing CAN pull values below the local
     max, so this step alone does not guarantee no residual penetration.
  4. envelope = max(smoothed, required) pointwise -- restores the guarantee
     from step 1 at the (rare, isolated) points where step 3 undershot.
     After this, envelope[t] >= required[t] for every t by construction, so
     lowest_z[t] + envelope[t] >= 0 always: floorPen_cm = 0.00 is an
     algebraic guarantee, not an empirical result, for every clip.
qpos[:, 2] += envelope (root z only, per frame -- NOT a rigid whole-clip
shift like T5, so it is not just additive to velocity-based metrics the way
T5's shift was; the whole point of this script is to verify empirically
whether the smooth ramps keep vMax/n_spikes/jerk acceptable and whether
joint_ok survives, since neither is guaranteed analytically the way
floorPen is.

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t6_localground.py --build
    conda run -n gmr python scripts/g1/sprint_s8_t6_localground.py --eval
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import maximum_filter1d, gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from smooth_heldaware import save_pkl  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from sprint_s8_t3_corpus import (  # noqa: E402
    eval_one, load_class_map, BVH_DIR, CANON_DIR, PKL_S5_DIR, HUMAN_TARGETS_DIR, OUT_CSV,
)

VARIANT = "perframelimb_smrc_localground"
SRC_SUFFIX = "_perframelimb_smrc"
DST_SUFFIX = "_smrc_localground"

RAMP_HALF_SEC = 0.15   # maximum-filter half-width: widen each spike into a plateau
RAMP_SIGMA_SEC = 0.07  # gaussian sigma: round the plateau corners
REQUIRED_EPS_M = 0.001  # 1mm -- below this, treat a frame as "not penetrating" for reporting


def _envelope(required: np.ndarray, fps: float) -> np.ndarray:
    half = max(1, round(RAMP_HALF_SEC * fps))
    sigma = max(0.5, RAMP_SIGMA_SEC * fps)
    widened = maximum_filter1d(required, size=2 * half + 1, mode="nearest")
    smoothed = gaussian_filter1d(widened, sigma=sigma, mode="nearest")
    return np.maximum(smoothed, required)


def do_build(force=False):
    print(f"[T6 build] {VARIANT}, force={force}")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        src = PKL_S5_DIR / f"{clip}{SRC_SUFFIX}.pkl"
        dst = PKL_S5_DIR / f"{clip}{DST_SUFFIX}.pkl"
        if not src.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (no {SRC_SUFFIX} pkl)")
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
        n_touched = int((required > REQUIRED_EPS_M).sum())
        peak_cm = float(envelope.max()) * 100
        print(f"[{i + 1}/{len(bvhs)}] {clip}: {n_touched}/{qpos.shape[0]} penetrating frames, "
              f"peak envelope={peak_cm:.2f}cm")


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
        p = PKL_S5_DIR / f"{clip}{DST_SUFFIX}.pkl"
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
    _print_table(rows)


def _print_table(rows):
    variants = ["gmr_heightfix", "perframelimb_smrc", "perframelimb_smrc_ground", VARIANT]
    axes = ["joint_ok_pct", "floorPen_cm", "pen_pct", "coll_pct", "coll_peak_cm",
            "worst_float_cm", "worst_pen_cm", "range_cm", "fidelity_pos_err_cm",
            "fidelity_ori_err_deg", "joint_jerk_mean", "body_jerk_mean",
            "skate_left_mean_cm", "skate_right_mean_cm", "vMax_rad_s", "n_spikes"]
    for cls in ["floor", "loco"]:
        print(f"\n=== class={cls} ===")
        header = f"{'metric':<25}" + "".join(f"{v:>26}" for v in variants)
        print(header)
        for ax in axes:
            vals = []
            for v in variants:
                xs = [r[ax] for r in rows if r["variant"] == v and r["class"] == cls
                      and ax in r and not (isinstance(r[ax], float) and np.isnan(r[ax]))]
                vals.append(np.mean(xs) if xs else float("nan"))
            line = f"{ax:<25}" + "".join(f"{x:>26.3f}" if not np.isnan(x) else f"{'nan':>26}" for x in vals)
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.build and not args.eval:
        ap.error("pass --build and/or --eval")
    if args.build:
        do_build(force=args.force)
    if args.eval:
        do_eval()


if __name__ == "__main__":
    main()

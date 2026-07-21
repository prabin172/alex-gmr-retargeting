#!/usr/bin/env python3
"""S9-T0-gate, full 77-clip corpus: raw-velocity-gated posture-continuity
(attempt 1, `posture_gate_lo=0.02, posture_gate_hi=0.05` -- the one that
cleared the dev-clip check and beat attempt 2, see
`sprint_s9_t0gate_probe.py`), on top of the exact shipped pipeline
(`perframelimb_smrc_rl_localground`, rate_limit=0.15, T6 localground).

New variant: `perframelimb_smrc_rlpg_localground` ("pg" = posture-gated).
Same as the shipped `..._rl_localground` build in
`sprint_s8_t8_rl_localground.py` except `_limbwise_pass` also gets
`posture_continuity=True, posture_weight=1.0, posture_gate_lo=0.02,
posture_gate_hi=0.05, raw_gate_qpos=<that clip's pristine gmr_raw pkl>`.
Appends rows to the SAME `s8_t3_full_corpus.csv` the shipped baseline and
`gmr_heightfix` rows already live in (reused, not rebuilt) so the 3-way
comparison is apples to apples.

Usage:
    conda run -n gmr python scripts/g1/sprint_s9_t0gate_full_corpus.py --build --ground --eval
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
    eval_one, load_class_map, BVH_DIR, CANON_DIR, GMR_PKL_DIR, PKL_S5_DIR,
    HUMAN_TARGETS_DIR, OUT_CSV,
)

TAG = "pg"
VARIANT = f"perframelimb_smrc_{TAG}_localground"
POSTURE_GATE_LO = 0.02
POSTURE_GATE_HI = 0.05
RATE = 0.15


def do_build(force=False):
    print(f"[T0gate-corpus build] rate_limit={RATE}, posture_gate=[{POSTURE_GATE_LO},{POSTURE_GATE_HI}]")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        src_sm = PKL_S5_DIR / f"{clip}_perframelimb_sm.pkl"
        raw_pkl = GMR_PKL_DIR / f"{clip}.pkl"
        dst_smrc = PKL_S5_DIR / f"{clip}_perframelimb_smrc_{TAG}.pkl"
        if not canon.exists() or not src_sm.exists() or not raw_pkl.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (missing canonical/perframelimb_sm/raw pkl)")
            continue
        if dst_smrc.exists() and not force:
            print(f"[{i + 1}/{len(bvhs)}] SKIP (done) {clip}")
            continue
        held_mask, _ = compute_held_mask(canon)
        qpos_sm, fps = load_gmr_pkl(src_sm)
        raw_qpos, _ = load_gmr_pkl(raw_pkl)
        if raw_qpos.shape[0] != qpos_sm.shape[0]:
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (raw T={raw_qpos.shape[0]} != "
                  f"perframelimb_sm T={qpos_sm.shape[0]})")
            continue
        out = _limbwise_pass(model, data, mesh_cache, qpos_sm, held_mask, FEET,
                              RAMP_FRAMES, max_dq=0.15, avoid_self_collision=True,
                              rate_limit=RATE, posture_continuity=True, posture_weight=1.0,
                              posture_gate_lo=POSTURE_GATE_LO, posture_gate_hi=POSTURE_GATE_HI,
                              raw_gate_qpos=raw_qpos)
        save_pkl(dst_smrc, out, fps)
        print(f"[{i + 1}/{len(bvhs)}] {clip}: smrc_{TAG} saved")


def do_ground(force=False):
    print(f"[T0gate-corpus ground:{TAG}]")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        src = PKL_S5_DIR / f"{clip}_perframelimb_smrc_{TAG}.pkl"
        dst = PKL_S5_DIR / f"{clip}_smrc_{TAG}_localground.pkl"
        if not src.exists():
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (no smrc_{TAG} pkl)")
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
        p = PKL_S5_DIR / f"{clip}_smrc_{TAG}_localground.pkl"
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


def print_3way_table():
    """gmr_heightfix vs shipped perframelimb_smrc_rl_localground vs this
    variant, class-split, dense table -- same axes S8's own gate used."""
    VARIANTS = ["gmr_heightfix", "perframelimb_smrc_rl_localground", VARIANT]
    AXES = ["joint_ok_pct", "floorPen_cm", "pen_pct", "coll_pct", "coll_peak_cm",
            "worst_float_cm", "worst_pen_cm", "range_cm", "fidelity_pos_err_cm",
            "fidelity_ori_err_deg", "joint_jerk_mean", "body_jerk_mean",
            "skate_left_mean_cm", "skate_right_mean_cm", "vMax_rad_s", "n_spikes"]
    rows = []
    with open(OUT_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    class_map = load_class_map()
    for cls in ["floor", "loco"]:
        print(f"\n=== class={cls} ===")
        header = f"{'metric':<25}" + "".join(f"{v:>32}" for v in VARIANTS)
        print(header)
        for ax in AXES:
            vals = []
            for v in VARIANTS:
                xs = [float(r[ax]) for r in rows if r["variant"] == v
                      and class_map.get(r["clip"], "?") == cls and r.get(ax) not in (None, "", "nan")]
                vals.append(np.mean(xs) if xs else float("nan"))
            line = f"{ax:<25}" + "".join(f"{x:>32.3f}" if not np.isnan(x) else f"{'nan':>32}" for x in vals)
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ground", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not (args.build or args.ground or args.eval or args.table):
        ap.error("pass --build and/or --ground and/or --eval and/or --table")
    if args.build:
        do_build(force=args.force)
    if args.ground:
        do_ground(force=args.force)
    if args.eval:
        do_eval()
    if args.table:
        print_3way_table()


if __name__ == "__main__":
    main()

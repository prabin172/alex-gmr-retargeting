#!/usr/bin/env python3
"""S8 LOCK: full-corpus metrics dump for the locked variant
(`perframelimb_smrc_rl_localground`, T6 local grounding + T8 rate-limited
re-clamp -- 5/6 never-tradeable axes, T4 visual veto passed, `## S8-T9`).

Prabin (2026-07-18): lock this variant as the working baseline for now.
Pulls its 77 rows straight out of the existing corpus CSV
(s8_t3_full_corpus.csv, already computed: floorPen, float, coll, fidelity/
tracking error, foot slip via skate_left/right, vMax, spikes, jerk,
joint_ok, range) and ADDS hand slip -- never tracked corpus-wide before
(the existing `skate_cm` in sprint_s5_metrics.py is hardcoded to feet
only). Hand slip reuses the exact same held-segment/XY-drift recipe,
generalized to any effector body, over `compute_held_masks(canon, HANDS)`
-- canonical human contact labels already include left_hand/right_hand
(confirmed non-trivial contact counts, e.g. walk3_subject1: 388/457
frames), so hand-hold segments exist wherever a clip's motion touches a
hand down (falls, push-offs, crawling), NaN elsewhere (no held segments in
that clip).

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_lock_final.py
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
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from gmr_contact_retarget import compute_held_masks, HANDS, EFF_BODY  # noqa: E402
from sprint_s5_metrics import _held_segments  # noqa: E402
from sprint_s8_t3_corpus import BVH_DIR, CANON_DIR, PKL_S5_DIR  # noqa: E402

VARIANT = "perframelimb_smrc_rl_localground"
SRC_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s8_t3_full_corpus.csv"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s8_LOCKED_perframelimb_smrc_rl_localground.csv"

DISPLAY_COLS = [
    "clip", "class",
    "floorPen_cm", "pen_pct", "worst_float_cm", "worst_pen_cm", "range_cm",
    "coll_pct", "coll_peak_cm",
    "fidelity_pos_err_cm", "fidelity_ori_err_deg",
    "skate_left_mean_cm", "skate_right_mean_cm",
    "hand_slip_left_mean_cm", "hand_slip_right_mean_cm",
    "vMax_rad_s", "n_spikes", "joint_jerk_mean", "body_jerk_mean",
    "joint_ok_pct",
]


def slip_cm(model, data, held_bool, qpos, body_name):
    bid = model.body(body_name).id
    segs = _held_segments(held_bool)
    drifts = []
    for (s, e) in segs:
        xy0 = None
        seg_max = 0.0
        for t in range(s, e):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            xy = data.xpos[bid][:2].copy()
            if xy0 is None:
                xy0 = xy
            seg_max = max(seg_max, float(np.linalg.norm(xy - xy0)))
        if xy0 is not None:
            drifts.append(seg_max)
    if drifts:
        return float(np.mean(drifts) * 100), len(drifts)
    return float("nan"), 0


def main():
    assert SRC_CSV.exists(), f"missing {SRC_CSV} -- run sprint_s8_t3_corpus.py --eval first"
    base_rows = {}
    with open(SRC_CSV) as f:
        for r in csv.DictReader(f):
            if r["variant"] == VARIANT:
                base_rows[r["clip"]] = r

    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    out_rows = []
    for i, clip in enumerate(bvhs):
        if clip not in base_rows:
            print(f"[{i + 1}/{len(bvhs)}] SKIP {clip} (no {VARIANT} row in {SRC_CSV.name})")
            continue
        row = dict(base_rows[clip])

        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        pkl_path = PKL_S5_DIR / f"{clip}_smrc_rl_localground.pkl"
        if canon.exists() and pkl_path.exists():
            qpos, fps = load_gmr_pkl(pkl_path)
            held, _ = compute_held_masks(canon, HANDS)
            l_cm, l_n = slip_cm(model, data, held["left_hand"], qpos, EFF_BODY["left_hand"])
            r_cm, r_n = slip_cm(model, data, held["right_hand"], qpos, EFF_BODY["right_hand"])
            row["hand_slip_left_mean_cm"] = l_cm
            row["hand_slip_right_mean_cm"] = r_cm
            row["hand_slip_left_n_segments"] = l_n
            row["hand_slip_right_n_segments"] = r_n
        else:
            row["hand_slip_left_mean_cm"] = row["hand_slip_right_mean_cm"] = float("nan")
            row["hand_slip_left_n_segments"] = row["hand_slip_right_n_segments"] = 0
            print(f"[{i + 1}/{len(bvhs)}] {clip}: missing canonical/pkl for hand slip, left NaN")

        out_rows.append(row)
        print(f"[{i + 1}/{len(bvhs)}] {clip}: done")

    cols = list(out_rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in out_rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nWrote {len(out_rows)} rows -> {OUT_CSV}")

    _print_averages(out_rows)


def _print_averages(rows):
    axes = [c for c in DISPLAY_COLS if c not in ("clip", "class")]
    for label, subset in [("ALL 77", rows),
                           ("floor class", [r for r in rows if r["class"] == "floor"]),
                           ("loco class", [r for r in rows if r["class"] == "loco"])]:
        print(f"\n=== {label} (n={len(subset)}) ===")
        for ax in axes:
            xs = [float(r[ax]) for r in subset if r.get(ax, "") not in ("", "nan") and not (
                isinstance(r[ax], float) and np.isnan(r[ax]))]
            xs = [x for x in xs if not np.isnan(x)]
            if xs:
                print(f"  {ax:<28} mean={np.mean(xs):9.3f}  (n={len(xs)})")
            else:
                print(f"  {ax:<28} mean=  n/a  (n=0)")


if __name__ == "__main__":
    main()

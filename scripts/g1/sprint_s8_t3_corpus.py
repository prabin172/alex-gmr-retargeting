#!/usr/bin/env python3
"""S8-T3: 77-clip corpus build + full 13-axis eval for the S8 winner
(`perframelimb_smrc`, T2c attempt-2: smooth->re-clamp, max_dq=0.15) and the
fairness/baseline columns per REVISION R1.2/R1.3/T3 and the S8-T2-DECISION
(planLogGMR.md) to ship it.

Builds (resumable, skip-if-exists), for every clip with a grounded canonical
+ gmr_raw pkl + an existing perframelimb pkl:
  pkl_s5/{clip}_perframelimb_sm.pkl    (held-aware smoother, spike-unlock)
  pkl_s5/{clip}_perframelimb_smrc.pkl  (T2c: re-clamp the _sm pkl, max_dq=0.15)
  pkl_s5/{clip}_heightfix_sm.pkl       (fairness arm: same smoother on GMR-full)

Evaluates ALL 13 axes for 5 primary columns (gmr_raw | gmr_heightfix |
gmr_heightfix_sm | perframelimb | perframelimb_smrc): joint_ok, floorPen,
pen%, coll%, coll_peak, worst_float, worst_pen, range, fidelity_pos,
fidelity_ori, joint_jerk(+body_jerk), skate, vMax, n_spikes. Class labels
from s1t4_reclass.csv (floor_class 1/0 -> floor/loco), matching S8's
existing convention.

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t3_corpus.py --build
    conda run -n gmr python scripts/g1/sprint_s8_t3_corpus.py --eval
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
from sprint_s3_full_corpus import ROLE_TO_G1_BODY, FOOT_POS_ROLE, whole_clip_metrics  # noqa: E402
from sprint_s5_metrics import joint_ok_pct, skate_cm, fidelity_metrics, jerk_metrics  # noqa: E402
from sprint_s6_range_summary import compute_held_mask, clip_worst_float_pen  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from smooth_heldaware import smooth_heldaware, save_pkl  # noqa: E402
from gmr_contact_retarget import compute_held_masks, FEET  # noqa: E402
from polish_median_limbwise import _limbwise_pass, RAMP_FRAMES  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from eval_ihmc_json import evaluate as eval_ihmc_evaluate  # noqa: E402

BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
HUMAN_TARGETS_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/human_targets"
RECLASS_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s1t4_reclass.csv"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s8_t3_full_corpus.csv"

SMRC_MAX_DQ = 0.15  # T2c attempt-2's established trust region


def load_class_map():
    m = {}
    with open(RECLASS_CSV) as f:
        for r in csv.DictReader(f):
            m[r["clip"]] = "floor" if r["floor_class"] == "1" else "loco"
    return m


def do_build(force=False):
    print(f"[T3 build] perframelimb_sm / perframelimb_smrc / heightfix_sm, force={force}")
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    for i, clip in enumerate(bvhs):
        i1 = i + 1
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not canon.exists():
            print(f"[{i1}/{len(bvhs)}] SKIP {clip} (no grounded canonical)")
            continue
        held, T = compute_held_masks(canon, FEET)

        src_pfl = PKL_S5_DIR / f"{clip}_perframelimb.pkl"
        dst_sm = PKL_S5_DIR / f"{clip}_perframelimb_sm.pkl"
        dst_smrc = PKL_S5_DIR / f"{clip}_perframelimb_smrc.pkl"
        src_hf = GMR_PKL_DIR / f"{clip}_gmrfix.pkl"
        dst_hfsm = PKL_S5_DIR / f"{clip}_heightfix_sm.pkl"

        n_new = 0
        if src_pfl.exists():
            if not dst_sm.exists() or force:
                qpos, fps = load_gmr_pkl(src_pfl)
                out = smooth_heldaware(qpos, held, fps=fps)
                save_pkl(dst_sm, out, fps)
                n_new += 1
            if (not dst_smrc.exists() or force) and dst_sm.exists():
                qpos, fps = load_gmr_pkl(dst_sm)
                assert T == qpos.shape[0], f"canonical T={T} != qpos T={qpos.shape[0]}"
                out = _limbwise_pass(model, data, mesh_cache, qpos, held, FEET,
                                      RAMP_FRAMES, max_dq=SMRC_MAX_DQ,
                                      avoid_self_collision=True, rate_limit=None)
                save_pkl(dst_smrc, out, fps)
                n_new += 1
        else:
            print(f"[{i1}/{len(bvhs)}] {clip}: SKIP perframelimb-derived (source missing)")

        if src_hf.exists() and (not dst_hfsm.exists() or force):
            qpos, fps = load_gmr_pkl(src_hf)
            out = smooth_heldaware(qpos, held, fps=fps)
            save_pkl(dst_hfsm, out, fps)
            n_new += 1

        print(f"[{i1}/{len(bvhs)}] {clip}: {n_new} new pkl(s)")


def eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held, human_targets_path,
             vmax_ctx, clip, vname, pkl_path):
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
    wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
    jok, n_held = joint_ok_pct(model, data, mesh_cache, geom_ids, role_bid, held, qpos)
    res = clip_worst_float_pen(model, data, mesh_cache, role_bid, held, qpos)
    wf, wp = res if res is not None else (float("nan"), float("nan"))
    return dict(
        clip=clip, variant=vname,
        joint_ok_pct=jok, n_held_frames=n_held,
        floorPen_cm=wm["floorPen_cm"], pen_pct=wm["pen_pct"],
        coll_pct=wm["coll_pct"], coll_peak_cm=wm["coll_peak_cm"],
        worst_float_cm=wf, worst_pen_cm=wp,
        range_cm=(wf - wp) if not (np.isnan(wf) or np.isnan(wp)) else float("nan"),
        fidelity_pos_err_cm=fm["pos_err_cm"], fidelity_ori_err_deg=fm["ori_err_deg"],
        joint_jerk_mean=jm["joint_jerk_mean"], body_jerk_mean=jm["body_jerk_mean"],
        skate_left_mean_cm=sk["left"]["mean_cm"], skate_right_mean_cm=sk["right"]["mean_cm"],
        vMax_rad_s=vr["vel_max_rad_s"], n_spikes=vr["n_spikes"],
    )


VARIANT_SPECS = [
    ("gmr_raw", GMR_PKL_DIR, ""),
    ("gmr_heightfix", GMR_PKL_DIR, "_gmrfix"),
    ("heightfix_sm", PKL_S5_DIR, "_heightfix_sm"),
    ("perframelimb", PKL_S5_DIR, "_perframelimb"),
    ("perframelimb_smrc", PKL_S5_DIR, "_perframelimb_smrc"),
]


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
    for i, clip in enumerate(bvhs):
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not canon.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP {clip} (no grounded canonical)")
            continue
        held, _ = compute_held_mask(canon)
        human_targets_path = HUMAN_TARGETS_DIR / f"{clip}.npz"
        cls = class_map.get(clip, "?")
        n_new = 0
        for vname, subdir, suffix in VARIANT_SPECS:
            if (clip, vname) in done:
                continue
            p = subdir / f"{clip}{suffix}.pkl"
            if not p.exists():
                print(f"  MISSING {clip} {vname}")
                continue
            row = eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held,
                            human_targets_path, vmax_ctx, clip, vname, p)
            row["class"] = cls
            rows.append(row)
            n_new += 1
        print(f"[{i+1}/{len(bvhs)}] {clip}: {n_new} new rows")
        if n_new:
            cols = list(rows[0].keys())
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT_CSV, "w") as f:
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")
    _print_class_table(rows)


def _print_class_table(rows):
    VARIANTS = [v[0] for v in VARIANT_SPECS]
    AXES = ["joint_ok_pct", "floorPen_cm", "pen_pct", "coll_pct", "coll_peak_cm",
            "worst_float_cm", "worst_pen_cm", "range_cm", "fidelity_pos_err_cm",
            "fidelity_ori_err_deg", "joint_jerk_mean", "body_jerk_mean",
            "skate_left_mean_cm", "skate_right_mean_cm", "vMax_rad_s", "n_spikes"]
    for cls in ["floor", "loco"]:
        print(f"\n=== class={cls} ===")
        header = f"{'metric':<25}" + "".join(f"{v:>18}" for v in VARIANTS)
        print(header)
        for ax in AXES:
            vals = []
            for v in VARIANTS:
                xs = [r[ax] for r in rows if r["variant"] == v and r["class"] == cls
                      and ax in r and not (isinstance(r[ax], float) and np.isnan(r[ax]))]
                vals.append(np.mean(xs) if xs else float("nan"))
            line = f"{ax:<25}" + "".join(f"{x:>18.3f}" if not np.isnan(x) else f"{'nan':>18}" for x in vals)
            print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
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

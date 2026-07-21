#!/usr/bin/env python3
"""S8-T1b gate eval: both metric families for the rate-limited variants on the
15-clip gate set, written to s8_t1b_eval.csv. Also prints a before/after gate
comparison against:
  - joint metrics before: s7b_full_corpus.csv (gmr_contact_fc / perframelimb)
  - smoothness before:    s7_smoothness.csv   (same variants + gmr_raw)

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t1b_eval.py
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
from sprint_s3_full_corpus import ROLE_TO_G1_BODY, FOOT_POS_ROLE, whole_clip_metrics, held_metrics  # noqa: E402
from sprint_s5_metrics import joint_ok_pct, skate_cm, fidelity_metrics, jerk_metrics  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from eval_ihmc_json import evaluate as eval_ihmc_evaluate  # noqa: E402

CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
HUMAN_TARGETS_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/human_targets"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s8_t1b_eval.csv"
S7B_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7b_full_corpus.csv"
SMOOTH_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_smoothness.csv"
RECLASS_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s1t4_reclass.csv"

DEV_CLIPS = ["walk1_subject1", "walk3_subject1", "run2_subject1",
             "ground1_subject1", "fallAndGetUp1_subject1"]
FC_WORST = ["obstacles6_subject5", "fallAndGetUp1_subject4", "obstacles5_subject2",
            "walk3_subject3", "fallAndGetUp2_subject2"]
PFL_WORST = ["obstacles4_subject3", "walk2_subject3", "obstacles5_subject3",
             "aiming1_subject4", "pushAndFall1_subject4"]
ALL_CLIPS = DEV_CLIPS + FC_WORST + PFL_WORST

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", type=str, default="rl")
    args = ap.parse_args()
    sfx = args.suffix
    VARIANTS = [(f"gmr_contact_fc_{sfx}", f"_gmrcontact_fc_{sfx}"),
                (f"perframelimb_{sfx}", f"_perframelimb_{sfx}")]
    BEFORE_OF = {f"gmr_contact_fc_{sfx}": "gmr_contact_fc",
                 f"perframelimb_{sfx}": "perframelimb"}
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)

    done = set()
    rows = []
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for r in csv.DictReader(f):
                done.add((r["clip"], r["variant"]))
                rows.append(r)

    for i, clip in enumerate(ALL_CLIPS):
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        held, _ = compute_held_mask(grounded)
        for vname, suffix in VARIANTS:
            if (clip, vname) in done:
                continue
            p = PKL_S5_DIR / f"{clip}{suffix}.pkl"
            if not p.exists():
                print(f"[{i+1}/{len(ALL_CLIPS)}] SKIP {clip} {vname} (pkl missing)")
                continue
            qpos, fps = load_gmr_pkl(p)
            wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
            jok, n_held = joint_ok_pct(model, data, mesh_cache, geom_ids, role_bid, held, qpos)
            sk = skate_cm(model, data, held, qpos)
            jm = jerk_metrics(model, data, qpos, fps)
            ht = HUMAN_TARGETS_DIR / f"{clip}.npz"
            fm = fidelity_metrics(model, data, qpos, ht) if ht.exists() \
                else dict(pos_err_cm=float("nan"), ori_err_deg=float("nan"))
            (vmodel, vdata, vmesh_cache, vgeom_ids, vjoint_names, vq_lo, vq_hi) = vmax_ctx
            vr = eval_ihmc_evaluate(p.stem, qpos, fps, {}, vmodel, vdata, vmesh_cache,
                                     vgeom_ids, {}, vq_lo, vq_hi, vjoint_names)
            row = dict(
                clip=clip, variant=vname,
                joint_ok_pct=jok, floorPen_cm=wm["floorPen_cm"], pen_pct=wm["pen_pct"],
                coll_pct=wm["coll_pct"], coll_peak_cm=wm["coll_peak_cm"],
                joint_jerk_mean=jm["joint_jerk_mean"],
                skate_left_mean_cm=sk["left"]["mean_cm"],
                skate_right_mean_cm=sk["right"]["mean_cm"],
                fidelity_pos_err_cm=fm["pos_err_cm"], fidelity_ori_err_deg=fm["ori_err_deg"],
                vMax_rad_s=vr["vel_max_rad_s"], vP95_rad_s=vr["vel_p95_rad_s"],
                n_spikes=vr["n_spikes"],
            )
            rows.append(row)
            print(f"[{i+1}/{len(ALL_CLIPS)}] {clip} {vname}: jok={jok:.2f} "
                  f"fp={wm['floorPen_cm']:.2f} coll={wm['coll_pct']:.3f} "
                  f"vMax={vr['vel_max_rad_s']:.1f} spikes={vr['n_spikes']}", flush=True)
            cols = list(rows[0].keys())
            with open(OUT_CSV, "w") as f:
                f.write(",".join(cols) + "\n")
                for r in rows:
                    f.write(",".join(str(r[c]) for c in cols) + "\n")

    # ---- before/after comparison ----
    floor_class = {}
    with open(RECLASS_CSV) as f:
        for r in csv.DictReader(f):
            floor_class[r["clip"]] = int(r["floor_class"])

    before_joint = {}
    with open(S7B_CSV) as f:
        for r in csv.DictReader(f):
            before_joint[(r["clip"], r["variant"])] = r
    before_smooth = {}
    with open(SMOOTH_CSV) as f:
        for r in csv.DictReader(f):
            before_smooth[(r["clip"], r["variant"])] = r

    after = {(r["clip"], r["variant"]): r for r in rows}

    print("\n==== BEFORE -> AFTER per clip ====")
    hdr = (f"{'clip':<26}{'variant':<20}{'jok b->a':>16}{'fp b->a':>14}"
           f"{'coll b->a':>14}{'vMax b->a':>14}{'spk b->a':>10}")
    print(hdr)
    for vname, _ in VARIANTS:
        bv = BEFORE_OF[vname]
        for clip in ALL_CLIPS:
            a = after.get((clip, vname))
            bj = before_joint.get((clip, bv))
            bs = before_smooth.get((clip, bv))
            if not a or not bj or not bs:
                continue
            print(f"{clip:<26}{vname:<20}"
                  f"{float(bj['joint_ok_pct']):7.2f}->{float(a['joint_ok_pct']):6.2f}"
                  f"{float(bj['floorPen_cm']):7.2f}->{float(a['floorPen_cm']):5.2f}"
                  f"{float(bj['coll_pct']):7.3f}->{float(a['coll_pct']):5.3f}"
                  f"{float(bs['vMax_rad_s']):7.1f}->{float(a['vMax_rad_s']):5.1f}"
                  f"{int(float(bs['n_spikes'])):5d}->{int(float(a['n_spikes'])):3d}")

    print("\n==== class means over the 15-clip gate set ====")
    for vname, _ in VARIANTS:
        bv = BEFORE_OF[vname]
        for cval, cname in [(1, "floor"), (0, "loco")]:
            clips = [c for c in ALL_CLIPS if floor_class.get(c, -1) == cval]
            aa = [after[(c, vname)] for c in clips if (c, vname) in after]
            bj = [before_joint[(c, bv)] for c in clips if (c, bv) in before_joint]
            bs = [before_smooth[(c, bv)] for c in clips if (c, bv) in before_smooth]
            braw = [before_smooth[(c, "gmr_raw")] for c in clips
                    if (c, "gmr_raw") in before_smooth]
            if not aa:
                continue
            def m(rows_, k):
                return float(np.mean([float(r[k]) for r in rows_]))
            print(f"{vname:<20}{cname:<7}n={len(aa)}  "
                  f"jok {m(bj,'joint_ok_pct'):6.2f}->{m(aa,'joint_ok_pct'):6.2f}  "
                  f"fp {m(bj,'floorPen_cm'):5.2f}->{m(aa,'floorPen_cm'):5.2f}  "
                  f"coll {m(bj,'coll_pct'):6.3f}->{m(aa,'coll_pct'):6.3f}  "
                  f"vMax {m(bs,'vMax_rad_s'):5.1f}->{m(aa,'vMax_rad_s'):5.1f} "
                  f"(raw {m(braw,'vMax_rad_s'):5.1f}, 1.2x={1.2*m(braw,'vMax_rad_s'):5.1f})  "
                  f"spk {m(bs,'n_spikes'):5.2f}->{m(aa,'n_spikes'):5.2f}  "
                  f"jerk {m(bs,'joint_jerk_mean'):7.0f}->{m(aa,'joint_jerk_mean'):7.0f}")


if __name__ == "__main__":
    main()

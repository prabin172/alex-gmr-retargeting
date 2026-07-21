#!/usr/bin/env python3
"""S7-T2: full 77-clip corpus build + eval for `gmr_contact_fc_sm` (Stage-A
smoothing + re-clamp on top of `gmr_contact_fc` -- see smooth_then_clamp.py and
planLogGMR.md S7-T2 for the gate that authorized this). Resumable (skip-if-
exists), matching S6/S7's own pattern. Eval covers BOTH batteries in one pass:
the joint-metric/range table (s7_fcsm_full_corpus.csv / s7_fcsm_range.csv) and
smoothness (appended to s7_smoothness.csv alongside T1's other 5 variants).

Usage:
    conda run -n gmr python scripts/g1/sprint_s7_fcsm_corpus.py --build
    conda run -n gmr python scripts/g1/sprint_s7_fcsm_corpus.py --eval
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from contact_labels import debounce_flags  # noqa: E402
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import load_canonical  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY, FOOT_POS_ROLE, whole_clip_metrics, held_metrics  # noqa: E402
from sprint_s5_metrics import joint_ok_pct, skate_cm, fidelity_metrics, jerk_metrics  # noqa: E402
from sprint_s6_range_summary import clip_worst_float_pen, compute_held_mask  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from eval_ihmc_json import evaluate as eval_ihmc_evaluate  # noqa: E402

BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
HUMAN_TARGETS_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/human_targets"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_fcsm_full_corpus.csv"
RANGE_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_fcsm_range.csv"
SMOOTH_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_smoothness.csv"
PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def do_build():
    log = REPO_ROOT / "outputs/gmr_baseline/sprint/s7_fcsm_build.log"
    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    total = len(bvhs)
    for i, clip in enumerate(bvhs):
        i1 = i + 1
        in_pkl = PKL_S5_DIR / f"{clip}_gmrcontact_fc.pkl"
        if not in_pkl.exists():
            print(f"[{i1}/{total}] SKIP {clip} (no gmr_contact_fc)")
            continue
        out_pkl = PKL_S5_DIR / f"{clip}_gmrcontact_fc_sm.pkl"
        if out_pkl.exists():
            print(f"[{i1}/{total}] SKIP (done) {clip} gmr_contact_fc_sm")
            continue
        t0 = time.time()
        ok = run([PY, str(SCRIPTS / "smooth_then_clamp.py"),
                  "--in", str(in_pkl), "--out", str(out_pkl)], log)
        print(f"[{i1}/{total}] {clip} gmr_contact_fc_sm {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")


def do_eval():
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    rows, range_rows, smooth_rows = [], [], []
    for i, clip in enumerate(bvhs):
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        p = PKL_S5_DIR / f"{clip}_gmrcontact_fc_sm.pkl"
        if not grounded.exists() or not p.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP {clip} (missing input)")
            continue
        (roles, role_to_idx, src_positions, fps, ori_roles, ori_to_idx, ori_mats,
         contacts, eff_names) = load_canonical(grounded)
        T = src_positions.shape[0]
        contacts_solved = {eff: debounce_flags(contacts[eff], 2) for eff in eff_names}
        held = {}
        for eff, role in FOOT_POS_ROLE.items():
            src_pt = src_positions[:, role_to_idx[role]]
            v = np.zeros(T)
            v[1:] = np.linalg.norm(np.diff(src_pt, axis=0), axis=1) * fps
            v[0] = v[1] if T > 1 else 0.0
            held[eff] = contacts_solved[eff] & (v < 0.05)

        qpos, cfps = load_gmr_pkl(p)
        wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
        hm = held_metrics(model, data, mesh_cache, role_bid, held, qpos)
        jok, n_held = joint_ok_pct(model, data, mesh_cache, geom_ids, role_bid, held, qpos)
        row = dict(clip=clip, variant="gmr_contact_fc_sm", joint_ok_pct=jok, n_held_frames=n_held, **wm)
        for eff in FOOT_POS_ROLE:
            row[f"held_{eff}_median_cm"] = hm[eff]["median_cm"]
            row[f"held_{eff}_frac3_pct"] = hm[eff]["frac3_pct"]
        rows.append(row)

        res = clip_worst_float_pen(model, data, mesh_cache, role_bid, held, qpos)
        if res is not None:
            wf, wp = res
            range_rows.append(dict(clip=clip, variant="gmr_contact_fc_sm", worst_float_cm=wf,
                                    worst_pen_cm=wp, range_cm=wf - wp))

        sk = skate_cm(model, data, held, qpos)
        ht = HUMAN_TARGETS_DIR / f"{clip}.npz"
        fm = fidelity_metrics(model, data, qpos, ht) if ht.exists() else dict(pos_err_cm=float("nan"), ori_err_deg=float("nan"))
        jm = jerk_metrics(model, data, qpos, cfps)
        (vmodel, vdata, vmesh_cache, vgeom_ids, vjoint_names, vq_lo, vq_hi) = vmax_ctx
        vr = eval_ihmc_evaluate(p.stem, qpos, cfps, {}, vmodel, vdata, vmesh_cache, vgeom_ids, {}, vq_lo, vq_hi, vjoint_names)
        smooth_rows.append(dict(
            clip=clip, variant="gmr_contact_fc_sm",
            joint_jerk_mean=jm["joint_jerk_mean"], joint_jerk_p95=jm["joint_jerk_p95"],
            body_jerk_mean=jm["body_jerk_mean"], body_jerk_p95=jm["body_jerk_p95"],
            skate_left_mean_cm=sk["left"]["mean_cm"], skate_left_max_cm=sk["left"]["max_cm"],
            skate_right_mean_cm=sk["right"]["mean_cm"], skate_right_max_cm=sk["right"]["max_cm"],
            fidelity_pos_err_cm=fm["pos_err_cm"], fidelity_ori_err_deg=fm["ori_err_deg"],
            vMax_rad_s=vr["vel_max_rad_s"], vP95_rad_s=vr["vel_p95_rad_s"], n_spikes=vr["n_spikes"],
        ))
        print(f"[{i+1}/{len(bvhs)}] {clip}: evaluated")

    cols = list(rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")

    rcols = list(range_rows[0].keys())
    with open(RANGE_CSV, "w") as f:
        f.write(",".join(rcols) + "\n")
        for r in range_rows:
            f.write(",".join(str(r[c]) for c in rcols) + "\n")
    print(f"Wrote {RANGE_CSV} ({len(range_rows)} rows)")

    # Append to s7_smoothness.csv (skip if gmr_contact_fc_sm rows already present).
    existing = []
    done = set()
    if SMOOTH_CSV.exists():
        with open(SMOOTH_CSV) as f:
            for r in csv.DictReader(f):
                existing.append(r)
                done.add((r["clip"], r["variant"]))
    scols = list(existing[0].keys()) if existing else list(smooth_rows[0].keys())
    new_smooth = [r for r in smooth_rows if (r["clip"], r["variant"]) not in done]
    with open(SMOOTH_CSV, "w") as f:
        f.write(",".join(scols) + "\n")
        for r in existing:
            f.write(",".join(str(r[c]) for c in scols) + "\n")
        for r in new_smooth:
            f.write(",".join(str(r[c]) for c in scols) + "\n")
    print(f"Appended {len(new_smooth)} gmr_contact_fc_sm rows to {SMOOTH_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    if args.build:
        do_build()
    if args.eval:
        do_eval()
    if not args.build and not args.eval:
        ap.error("pass --build and/or --eval")


if __name__ == "__main__":
    main()

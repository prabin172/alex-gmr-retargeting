#!/usr/bin/env python3
"""S3 (Prabin's request, follow-up to the S2 4-clip validation + knee_bias probe):
full 77-clip corpus, GMR (raw/gmrfix/polished, reused from S1) vs OURS (raw/StageA/
ctground, built here), one combined metrics CSV.

Resumable (skips any step whose output file already exists), matching this sprint's
established ground rule. Two phases:

  --build   Build OURS per-clip artifacts (canonical_human -> Stage3 solve -> StageA
            polish -> contact-aware ground) for every LAFAN1 clip. Slow (~1-2 min/clip
            for the Stage3 solve; the rest is seconds). Safe to interrupt and resume.
  --eval    Compute whole-clip (floorPen/pen%/self-collision) + held-frame support_z
            metrics for GMR's 3 variants (reusing existing sprint pkls, S1) and OURS's
            3 variants (this script's --build output), write one CSV.

No knee_bias -- this uses the plain, already-validated S2-T9 default config (the
knee_bias probe on 4 clips found a genuinely mixed result, not yet a ship decision;
keeping the corpus run on the shipped baseline keeps this comparable to the existing
4-clip numbers already in the wiki).

Usage:
    conda run -n gmr python scripts/g1/sprint_s3_full_corpus.py --build
    conda run -n gmr python scripts/g1/sprint_s3_full_corpus.py --eval --out outputs/gmr_baseline/sprint/s3_full_corpus.csv
"""
from __future__ import annotations

import argparse
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
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import load_canonical  # noqa: E402
from solve_global_trajectory_opt_contactfirst import _collision_stats  # noqa: E402
from solve_lafan1_canonical_g1_contactfirst import ROLE_TO_G1_BODY, FOOT_POS_ROLE  # noqa: E402
from stage_b_g1 import support_z  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402

BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human"
OURS_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/ours_g1_corpus"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def build_one(clip, log):
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    OURS_DIR.mkdir(parents=True, exist_ok=True)
    bvh = BVH_DIR / f"{clip}.bvh"
    canon = CANON_DIR / f"{clip}_lafan1c.npz"
    grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
    raw = OURS_DIR / f"{clip}_ours.npz"
    stagea = OURS_DIR / f"{clip}_ours_stageA.npz"
    ctground = OURS_DIR / f"{clip}_ours_ctground.npz"

    if not canon.exists():
        ok = run([PY, str(SCRIPTS / "lafan1_to_canonical_human.py"),
                  "--bvh", str(bvh), "--out", str(canon)], log)
        if not ok:
            return False, "canon"
    if not grounded.exists():
        ok = run([PY, str(REPO_ROOT / "scripts" / "ground_canonical_human.py"),
                  "--in-npz", str(canon), "--out-npz", str(grounded),
                  "--plant-min-run", "2"], log)
        if not ok:
            return False, "grounded"
    if not raw.exists():
        ok = run([PY, str(SCRIPTS / "solve_lafan1_canonical_g1_contactfirst.py"),
                  "--canonical", str(grounded), "--out", str(raw)], log)
        if not ok:
            return False, "raw solve"
    if not stagea.exists():
        ok = run([PY, str(SCRIPTS / "polish_ours_g1.py"),
                  "--in", str(raw), "--out", str(stagea)], log)
        if not ok:
            return False, "stageA"
    if not ctground.exists():
        ok = run([PY, str(SCRIPTS / "ground_ours_contact_aware.py"),
                  "--in", str(stagea), "--canonical", str(grounded),
                  "--out", str(ctground)], log)
        if not ok:
            return False, "ctground"
    return True, "ok"


def do_build():
    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    log = REPO_ROOT / "outputs/gmr_baseline/sprint/s3_build.log"
    print(f"{len(bvhs)} clips. Log: {log}")
    fails = []
    for i, clip in enumerate(bvhs):
        ctground = OURS_DIR / f"{clip}_ours_ctground.npz"
        if ctground.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP (done) {clip}")
            continue
        t0 = time.time()
        ok, stage = build_one(clip, log)
        dt = time.time() - t0
        status = "OK" if ok else f"FAIL@{stage}"
        print(f"[{i+1}/{len(bvhs)}] {status} {clip} ({dt:.0f}s)")
        if not ok:
            fails.append((clip, stage))
    print(f"\nDONE. {len(fails)} failures.")
    for clip, stage in fails:
        print(f"  FAIL {clip} at {stage}")


# ---------------- eval ----------------

def whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos):
    lowest = np.zeros(qpos.shape[0])
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)
    pen = np.maximum(0, -lowest)
    cs = _collision_stats(model, data, qpos, floor_gid=floor_gid, count_floor=False)
    return dict(floorPen_cm=pen.max() * 100, pen_pct=100 * (pen > 0.005).mean(),
               coll_pct=cs["pct"], coll_peak_cm=cs["max_pen_cm"])


def held_metrics(model, data, mesh_cache, role_bid, held, qpos):
    out = {}
    for eff, role in FOOT_POS_ROLE.items():
        idx = np.where(held[eff])[0]
        if idx.size == 0:
            out[eff] = dict(n=0, median_cm=float("nan"), frac3_pct=float("nan"))
            continue
        bid = role_bid[role]
        szs = np.zeros(idx.size)
        for k, t in enumerate(idx):
            if t >= qpos.shape[0]:
                szs[k] = np.nan
                continue
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            szs[k] = support_z(model, data, mesh_cache, bid)
        szs = szs[~np.isnan(szs)]
        out[eff] = dict(n=int(idx.size), median_cm=float(np.median(szs) * 100) if szs.size else float("nan"),
                        frac3_pct=float(np.mean(np.abs(szs) < 0.03) * 100) if szs.size else float("nan"))
    return out


def do_eval(out_csv):
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    rows = []
    for i, clip in enumerate(bvhs):
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not grounded.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP (no OURS canonical yet) {clip}")
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

        variants = []
        gmr_raw = GMR_PKL_DIR / f"{clip}.pkl"
        gmr_fix = GMR_PKL_DIR / f"{clip}_gmrfix.pkl"
        gmr_pol = GMR_PKL_DIR / f"{clip}_polished.pkl"
        if gmr_raw.exists():
            variants.append(("gmr_raw", load_gmr_pkl(gmr_raw)[0]))
        if gmr_fix.exists():
            variants.append(("gmr_heightfix", load_gmr_pkl(gmr_fix)[0]))
        if gmr_pol.exists():
            variants.append(("gmr_polished", load_gmr_pkl(gmr_pol)[0]))

        for suffix, name in [("_ours.npz", "ours_raw"), ("_ours_stageA.npz", "ours_stageA"),
                             ("_ours_ctground.npz", "ours_ctground")]:
            p = OURS_DIR / f"{clip}{suffix}"
            if p.exists():
                variants.append((name, np.load(p)["qpos"]))

        for vname, qpos in variants:
            wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
            hm = held_metrics(model, data, mesh_cache, role_bid, held, qpos)
            row = dict(clip=clip, variant=vname, T=qpos.shape[0], **wm)
            for eff in FOOT_POS_ROLE:
                row[f"held_{eff}_n"] = hm[eff]["n"]
                row[f"held_{eff}_median_cm"] = hm[eff]["median_cm"]
                row[f"held_{eff}_frac3_pct"] = hm[eff]["frac3_pct"]
            rows.append(row)
        print(f"[{i+1}/{len(bvhs)}] {clip}: {len(variants)} variants evaluated")

    cols = list(rows[0].keys()) if rows else []
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {out_csv} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/gmr_baseline/sprint/s3_full_corpus.csv")
    args = ap.parse_args()
    if args.build:
        do_build()
    if args.eval:
        do_eval(args.out)
    if not args.build and not args.eval:
        ap.error("pass --build and/or --eval")


if __name__ == "__main__":
    main()

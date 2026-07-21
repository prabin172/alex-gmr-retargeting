#!/usr/bin/env python3
"""S6-A5: full 77-clip corpus build + eval for the S6-A floor-clamp mechanism
(`gmr_contact_retarget.py --floor-clamp`). Resumable (skip-if-exists), matching
S5/S3's own pattern -- do not build `gmr_contact` itself here, that's S5's job
and already done (`pkl_s5/{clip}_gmrcontact.pkl` exists for all 77 clips).

  --build   Per clip with an existing grounded canonical (from S5): build
            `{clip}_gmrcontact_fc.pkl` via gmr_contact_retarget.py --floor-clamp
            (feet-only, ramp=5, matching S5's gmr_contact scope for a fair
            comparison).
  --eval    whole_clip_metrics + held_metrics + joint_ok_pct for gmr_raw,
            gmr_heightfix, gmr_polished, gmr_contact (S5), gmr_contact_fc (S6-A)
            -> s6_full_corpus.csv. Also writes the held-range table (worst
            float/worst pen/range) via sprint_s6_range_summary.py's functions.

Usage:
    conda run -n gmr python scripts/g1/sprint_s6_corpus.py --build
    conda run -n gmr python scripts/g1/sprint_s6_corpus.py --eval
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
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import load_canonical  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY, FOOT_POS_ROLE, whole_clip_metrics, held_metrics  # noqa: E402
from sprint_s5_metrics import joint_ok_pct  # noqa: E402
from sprint_s6_range_summary import clip_worst_float_pen  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402

BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s6_full_corpus.csv"
PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def do_build():
    log = REPO_ROOT / "outputs/gmr_baseline/sprint/s6_build.log"
    PKL_S5_DIR.mkdir(parents=True, exist_ok=True)
    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    total = len(bvhs)
    for i, clip in enumerate(bvhs):
        i1 = i + 1
        grounded_npz = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not grounded_npz.exists():
            print(f"[{i1}/{total}] SKIP {clip} (no grounded canonical)")
            continue

        fc_pkl = PKL_S5_DIR / f"{clip}_gmrcontact_fc.pkl"
        if fc_pkl.exists():
            print(f"[{i1}/{total}] SKIP (done) {clip} gmr_contact_fc")
            continue
        t0 = time.time()
        ok = run([PY, str(SCRIPTS / "gmr_contact_retarget.py"),
                  "--bvh_file", str(BVH_DIR / f"{clip}.bvh"),
                  "--canonical", str(grounded_npz),
                  "--save_path", str(fc_pkl),
                  "--ramp-frames", "5", "--floor-clamp",
                  "--avoid-self-collision"], log)  # S7-T7
        print(f"[{i1}/{total}] {clip} gmr_contact_fc {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")


def do_eval():
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    bvhs = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    rows = []
    range_rows = []
    for i, clip in enumerate(bvhs):
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not grounded.exists():
            print(f"[{i+1}/{len(bvhs)}] SKIP {clip} (no grounded canonical)")
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

        variants = [
            ("gmr_raw", GMR_PKL_DIR / f"{clip}.pkl"),
            ("gmr_heightfix", GMR_PKL_DIR / f"{clip}_gmrfix.pkl"),
            ("gmr_polished", GMR_PKL_DIR / f"{clip}_polished.pkl"),
            ("gmr_contact", PKL_S5_DIR / f"{clip}_gmrcontact.pkl"),
            ("gmr_contact_fc", PKL_S5_DIR / f"{clip}_gmrcontact_fc.pkl"),
        ]
        for vname, p in variants:
            if not p.exists():
                continue
            qpos, cfps = load_gmr_pkl(p)
            wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
            hm = held_metrics(model, data, mesh_cache, role_bid, held, qpos)
            jok, n_held = joint_ok_pct(model, data, mesh_cache, geom_ids, role_bid, held, qpos)
            row = dict(clip=clip, variant=vname, joint_ok_pct=jok, n_held_frames=n_held, **wm)
            for eff in FOOT_POS_ROLE:
                row[f"held_{eff}_median_cm"] = hm[eff]["median_cm"]
                row[f"held_{eff}_frac3_pct"] = hm[eff]["frac3_pct"]
            rows.append(row)

            res = clip_worst_float_pen(model, data, mesh_cache, role_bid, held, qpos)
            if res is not None:
                wf, wp = res
                range_rows.append(dict(clip=clip, variant=vname, worst_float_cm=wf,
                                        worst_pen_cm=wp, range_cm=wf - wp))
        print(f"[{i+1}/{len(bvhs)}] {clip}: {sum(1 for _, p in variants if p.exists())} variants evaluated")

    cols = list(rows[0].keys()) if rows else []
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")

    range_csv = REPO_ROOT / "outputs/gmr_baseline/sprint/s6_range.csv"
    rcols = list(range_rows[0].keys()) if range_rows else []
    with open(range_csv, "w") as f:
        f.write(",".join(rcols) + "\n")
        for r in range_rows:
            f.write(",".join(str(r[c]) for c in rcols) + "\n")
    print(f"Wrote {range_csv} ({len(range_rows)} rows)")


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

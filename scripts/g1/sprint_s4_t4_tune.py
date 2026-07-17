#!/usr/bin/env python3
"""S4-T4 tuning: --swing-max-pitch / --swing-continuity-reg grid, currently
still Alex's own unvalidated-for-G1 defaults (5deg / 0.9). Resumable (skips
any (clip, mp, cr) combo whose npz already exists), same pattern as
sprint_s3_full_corpus.py / sprint_s4_t2_eval.py.

Usage:
    conda run -n gmr python scripts/g1/sprint_s4_t4_tune.py --build
    conda run -n gmr python scripts/g1/sprint_s4_t4_tune.py --eval
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
from solve_lafan1_canonical_g1_contactfirst import ROLE_TO_G1_BODY, FOOT_POS_ROLE  # noqa: E402
from sprint_s3_full_corpus import whole_clip_metrics, held_metrics, OURS_DIR, CANON_DIR  # noqa: E402

TUNE_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/s4_dev/tune"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s4_t4_tune.csv"
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable

CLIPS = ["walk1_subject1", "run2_subject1", "jumps1_subject1"]
MAX_PITCH = [3.0, 5.0, 8.0, 12.0]
CONT_REG = [0.5, 0.9, 1.8]


def do_build():
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    log = REPO_ROOT / "outputs/gmr_baseline/sprint/s4_dev/tune_build.log"
    combos = [(mp, cr) for mp in MAX_PITCH for cr in CONT_REG]
    total = len(CLIPS) * len(combos)
    i = 0
    for clip in CLIPS:
        canon = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        for mp, cr in combos:
            i += 1
            out = TUNE_DIR / f"{clip}_mp{mp:g}_cr{cr:g}.npz"
            if out.exists():
                print(f"[{i}/{total}] SKIP (done) {clip} mp={mp} cr={cr}")
                continue
            t0 = time.time()
            cmd = [PY, str(SCRIPTS / "solve_lafan1_canonical_g1_contactfirst.py"),
                   "--canonical", str(canon), "--out", str(out),
                   "--swing-clear", "--swing-max-pitch", str(mp),
                   "--swing-continuity-reg", str(cr)]
            with open(log, "a") as f:
                f.write(f"\n$ {' '.join(cmd)}\n")
                f.flush()
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            status = "OK" if r.returncode == 0 else "FAIL"
            print(f"[{i}/{total}] {status} {clip} mp={mp} cr={cr} ({dt:.0f}s)")


def do_eval():
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    combos = [(mp, cr) for mp in MAX_PITCH for cr in CONT_REG]
    rows = []
    for clip in CLIPS:
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
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

        variants = [("s3_raw", OURS_DIR / f"{clip}_ours.npz")]
        for mp, cr in combos:
            variants.append((f"mp{mp:g}_cr{cr:g}", TUNE_DIR / f"{clip}_mp{mp:g}_cr{cr:g}.npz"))

        for vname, p in variants:
            if not p.exists():
                continue
            qpos = np.load(p)["qpos"]
            wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
            hm = held_metrics(model, data, mesh_cache, role_bid, held, qpos)
            row = dict(clip=clip, variant=vname, **wm)
            for eff in FOOT_POS_ROLE:
                row[f"held_{eff}_frac3_pct"] = hm[eff]["frac3_pct"]
            rows.append(row)
        print(f"{clip}: {len(variants)} variants evaluated")

    cols = list(rows[0].keys()) if rows else []
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


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

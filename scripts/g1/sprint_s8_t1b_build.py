#!/usr/bin/env python3
"""S8-T1b: build the rate-limited clamp variants on the gate eval set
(5 dev clips + T0b's 10 worst-spike clips), both variants:
  {clip}_gmrcontact_fc_rl.pkl   -- gmr_contact_retarget.py --floor-clamp
                                   --avoid-self-collision --clamp-rate-limit R
  {clip}_perframelimb_rl.pkl    -- polish_median_limbwise.py --center perframe
                                   --avoid-self-collision --clamp-rate-limit R
New suffixes only -- never overwrites existing pkls (baseline integrity).
Resumable (skip-if-exists), same pattern as sprint_s6_corpus.py.

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t1b_build.py [--rate 0.15]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BVH_DIR = REPO_ROOT / "data/raw/lafan1"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
PY = sys.executable
SCRIPTS = Path(__file__).resolve().parent

DEV_CLIPS = ["walk1_subject1", "walk3_subject1", "run2_subject1",
             "ground1_subject1", "fallAndGetUp1_subject1"]
FC_WORST = ["obstacles6_subject5", "fallAndGetUp1_subject4", "obstacles5_subject2",
            "walk3_subject3", "fallAndGetUp2_subject2"]
PFL_WORST = ["obstacles4_subject3", "walk2_subject3", "obstacles5_subject3",
             "aiming1_subject4", "pushAndFall1_subject4"]
ALL_CLIPS = DEV_CLIPS + FC_WORST + PFL_WORST


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--suffix", type=str, default="rl",
                    help="pkl suffix tag: 'rl' = attempt 1 (rate limit on the "
                         "total correction incl. phase 2), 'rl2' = attempt 2 "
                         "(phase-1 limited, collision post-pass un-limited -- "
                         "the drivers' behavior changed between attempts, the "
                         "tag records which code state built the pkl).")
    args = ap.parse_args()

    log = REPO_ROOT / "outputs/gmr_baseline/sprint/s8_t1b_build.log"
    total = len(ALL_CLIPS)
    for i, clip in enumerate(ALL_CLIPS):
        i1 = i + 1
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        raw_pkl = GMR_PKL_DIR / f"{clip}.pkl"
        if not grounded.exists() or not raw_pkl.exists():
            print(f"[{i1}/{total}] SKIP {clip} (missing inputs)", flush=True)
            continue

        # perframelimb (cheap, post-hoc)
        out = PKL_S5_DIR / f"{clip}_perframelimb_{args.suffix}.pkl"
        if out.exists():
            print(f"[{i1}/{total}] SKIP (done) {clip} perframelimb_{args.suffix}", flush=True)
        else:
            t0 = time.time()
            ok = run([PY, str(SCRIPTS / "polish_median_limbwise.py"),
                      "--in", str(raw_pkl), "--canonical", str(grounded),
                      "--out", str(out), "--center", "perframe",
                      "--avoid-self-collision",
                      "--clamp-rate-limit", str(args.rate)], log)
            print(f"[{i1}/{total}] {clip} perframelimb_{args.suffix} {'OK' if ok else 'FAIL'} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # gmrcontact_fc (full GMR retarget + inline clamp)
        out = PKL_S5_DIR / f"{clip}_gmrcontact_fc_{args.suffix}.pkl"
        if out.exists():
            print(f"[{i1}/{total}] SKIP (done) {clip} gmrcontact_fc_{args.suffix}", flush=True)
        else:
            t0 = time.time()
            ok = run([PY, str(SCRIPTS / "gmr_contact_retarget.py"),
                      "--bvh_file", str(BVH_DIR / f"{clip}.bvh"),
                      "--canonical", str(grounded),
                      "--save_path", str(out),
                      "--ramp-frames", "5", "--floor-clamp",
                      "--avoid-self-collision",
                      "--clamp-rate-limit", str(args.rate)], log)
            print(f"[{i1}/{total}] {clip} gmrcontact_fc_{args.suffix} {'OK' if ok else 'FAIL'} "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

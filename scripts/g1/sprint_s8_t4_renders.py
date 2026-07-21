#!/usr/bin/env python3
"""S8-T4: render the S8 winner (`perframelimb_smrc`) vs `gmr_heightfix` on the
R0 3 clips + the worst remaining floor-class clip by floorPen from the T3
corpus table (per GMR-S8-plan.md R1.5/Phase T4). Saves to repo-root
s8_renders/, filenames `{clip}__{variant}.mp4` (matches the existing R0
naming). Requires `outputs/gmr_baseline/sprint/s8_t3_full_corpus.csv` to
exist (T3 --eval).

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t4_renders.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable

GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
T3_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s8_t3_full_corpus.csv"
OUT_DIR = REPO_ROOT / "s8_renders"

R0_CLIPS = ["walk3_subject1", "fallAndGetUp1_subject1", "ground1_subject1"]
VARIANTS = [
    ("gmr_heightfix", GMR_PKL_DIR, "_gmrfix"),
    ("perframelimb_smrc", PKL_S5_DIR, "_perframelimb_smrc"),
]


def worst_floor_clip():
    assert T3_CSV.exists(), f"T3 CSV missing: {T3_CSV} -- run sprint_s8_t3_corpus.py --eval first"
    worst = None
    with open(T3_CSV) as f:
        for r in csv.DictReader(f):
            if r["variant"] != "perframelimb_smrc" or r["class"] != "floor":
                continue
            if r["clip"] in R0_CLIPS:
                continue  # don't re-pick an R0 clip
            fp = float(r["floorPen_cm"])
            if worst is None or fp > worst[1]:
                worst = (r["clip"], fp)
    return worst


def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode == 0


def main():
    worst = worst_floor_clip()
    clips = list(R0_CLIPS)
    if worst is not None:
        print(f"Worst remaining floor clip by floorPen (perframelimb_smrc): "
              f"{worst[0]} ({worst[1]:.2f}cm)")
        clips.append(worst[0])
    else:
        print("WARNING: no worst-floor-clip candidate found (T3 CSV empty/missing floor rows)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "t4_render.log"
    for clip in clips:
        for vname, subdir, suffix in VARIANTS:
            pkl_path = subdir / f"{clip}{suffix}.pkl"
            if not pkl_path.exists():
                print(f"  SKIP {clip} {vname} (missing {pkl_path})")
                continue
            out_mp4 = OUT_DIR / f"{clip}__{vname}.mp4"
            if out_mp4.exists():
                print(f"  SKIP (done) {out_mp4.name}")
                continue
            print(f"  rendering {out_mp4.name} ...")
            ok = run([PY, str(SCRIPTS / "render_penetration_annotated.py"),
                      "--pkl", str(pkl_path), "--out", str(out_mp4)], log)
            print(f"    {'OK' if ok else 'FAIL'}")

    print(f"\nDone. Renders in {OUT_DIR}")


if __name__ == "__main__":
    main()

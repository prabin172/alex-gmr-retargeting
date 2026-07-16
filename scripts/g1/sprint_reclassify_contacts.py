#!/usr/bin/env python3
"""Reclassify the 77-clip corpus by REAL multi-surface human contact instead of
hip-Z-p5 alone (S1-T4's own honest caveat: hip-only misses brief-but-real
non-hip floor contact -- a hand/knee touching down while the pelvis stays up).

Reuses `human_contacts_lafan1.py`'s `detect()`/`LANDMARKS` UNCHANGED (same
thresholds W2-T3 calibrated and validated on the 5-clip corpus: feet 0.05m,
hands/knees/elbows 0.08m, pelvis/torso 0.15m) -- no new detection logic, just
applied to all 77 clips and given an explicit classification rule.

Classification rule: floor-class if ANY non-foot landmark (hand/knee/elbow/
pelvis/torso -- feet excluded, walking alone lights up feet on every clip)
has a contiguous in-zone run >= 1 second (round(fps) frames). W2-T3's own
kill-test language distinguished "sustained" contact (25-88% zone, the 3 known
floor clips) from "noise-level" brief dips (dance1_subject1's hand blips,
0.2-0.6%, described there as "a brief low gesture, not sustained contact") --
a run-length bar operationalizes that same distinction (a genuine ground-use
phase persists; a reach/gesture/punch doesn't), rather than a bare zone-%
threshold that a busy fight/dance clip could cross via many short, scattered
dips without ever resting on anything.

Usage:
    conda run -n gmr python scripts/g1/sprint_reclassify_contacts.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from human_contacts_lafan1 import LANDMARKS, detect  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BVH_DIR = REPO_ROOT / "data" / "raw" / "lafan1"
OUT_DIR = REPO_ROOT / "outputs" / "gmr_baseline" / "human_contacts"
OUT_CSV = REPO_ROOT / "outputs" / "gmr_baseline" / "sprint" / "s1t4_reclass.csv"

NON_FOOT_LANDMARKS = [lm for lm in LANDMARKS if lm not in ("left_foot", "right_foot", "head")]
SUSTAIN_SECONDS = 1.0


def max_run_length(zone: np.ndarray) -> int:
    if not zone.any():
        return 0
    # contiguous True-run lengths via boundary diff
    padded = np.concatenate(([False], zone, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int((ends - starts).max())


def main():
    clips = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    thresholds = {lm: thr for lm, (_b, thr) in LANDMARKS.items() if thr is not None}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = ["clip", "fps", "sustain_thr_frames", "floor_class"] + \
             [f"{lm}_pct" for lm in NON_FOOT_LANDMARKS] + \
             [f"{lm}_maxrun_s" for lm in NON_FOOT_LANDMARKS]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ci, clip in enumerate(clips, 1):
            bvh_path = BVH_DIR / f"{clip}.bvh"
            npz_path = OUT_DIR / f"{clip}.npz"
            if npz_path.exists():
                d = np.load(npz_path, allow_pickle=True)
                zones = {lm: d[f"zone_{lm}"] for lm in NON_FOOT_LANDMARKS}
                fps = 30.0  # all LAFAN1 clips are 30fps (confirmed S1-T1/T3)
            else:
                zones, heights = detect(bvh_path, thresholds)
                fps = 30.0
                save_dict = {f"zone_{lm}": z for lm, z in zones.items()}
                save_dict.update({f"height_{lm}": h for lm, h in heights.items()})
                save_dict["thresholds_keys"] = np.array(list(thresholds.keys()))
                save_dict["thresholds_vals"] = np.array(list(thresholds.values()))
                np.savez_compressed(npz_path, **save_dict)

            sustain_frames = round(SUSTAIN_SECONDS * fps)
            max_runs = {lm: max_run_length(zones[lm]) for lm in NON_FOOT_LANDMARKS}
            floor_class = any(r >= sustain_frames for r in max_runs.values())

            row = {
                "clip": clip, "fps": fps, "sustain_thr_frames": sustain_frames,
                "floor_class": int(floor_class),
            }
            for lm in NON_FOOT_LANDMARKS:
                row[f"{lm}_pct"] = round(float(zones[lm].mean()) * 100, 2)
                row[f"{lm}_maxrun_s"] = round(max_runs[lm] / fps, 2)
            writer.writerow(row)
            print(f"[{ci}/{len(clips)}] {clip}: floor_class={floor_class}  "
                  f"maxrun_s={max(max_runs.values()) / fps:.2f}")

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()

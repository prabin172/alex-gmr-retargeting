#!/usr/bin/env python3
"""W2-T3: human-side multi-surface contact-zone labels from LAFAN1 BVH.

Detects contact on the HUMAN source (via GMR's own `load_bvh_file` loader --
per-frame {bone_name: [pos, quat]} dicts), not on the retargeted robot output.
This is the E4-lesson pivot (GMR-baseline.md SS7.2 item 2): a corpse-pose robot
with floating feet carries no usable contact signal; the human source is
uncorrupted and says directly which body parts bear weight.

Contact-zone rule: HEIGHT GATE ONLY, deliberately no speed gate (the other
E4 lesson -- planLogGMR.md "E4": a naive speed gate returns zero contacts
everywhere for rolling/complex contact; stillness sub-segmentation belongs to
`_compute_anchors`, downstream, not this detector). LAFAN1's floor is z=0
(established week 1, T2 clip-selection table: hipZ min as low as 0.028m on
the deepest fall clip -- consistent with a z=0 floor, no separate floor-height
estimation needed the way contact_labels.py's human-side detector does for
Alex's FBX sources, which lack that guarantee).

Landmarks (BVH bone names, LAFAN1 skeleton): feet (LeftFoot/RightFoot +
LeftToe/RightToe, min of the pair), hands (LeftHand/RightHand), knees
(LeftLeg/RightLeg -- LAFAN1's "Leg" bone sits at the knee, between UpLeg/thigh
and Foot/shank), elbows (LeftForeArm/RightForeArm), pelvis (Hips), torso
(Spine1 -- middle of the 3-segment spine chain). Head (Head) is tracked for
sanity/diagnostic plots only, never anchored.

Usage:
    # distribution report (calibration pass, no thresholds applied):
    conda run -n gmr python scripts/g1/human_contacts_lafan1.py --report-only \\
        data/raw/lafan1/walk1_subject1.bvh data/raw/lafan1/fallAndGetUp2_subject2.bvh ...

    # full detection + NPZ:
    conda run -n gmr python scripts/g1/human_contacts_lafan1.py \\
        data/raw/lafan1/fallAndGetUp2_subject2.bvh \\
        --out outputs/gmr_baseline/human_contacts/fallAndGetUp2_subject2.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from general_motion_retargeting.utils.lafan1 import load_bvh_file

# landmark -> (list of BVH bone names to min-combine, default height threshold in m)
LANDMARKS = {
    "left_foot":   (["LeftFoot", "LeftToe"], 0.05),
    "right_foot":  (["RightFoot", "RightToe"], 0.05),
    "left_hand":   (["LeftHand"], 0.08),
    "right_hand":  (["RightHand"], 0.08),
    "left_knee":   (["LeftLeg"], 0.08),
    "right_knee":  (["RightLeg"], 0.08),
    "left_elbow":  (["LeftForeArm"], 0.08),
    "right_elbow": (["RightForeArm"], 0.08),
    "pelvis":      (["Hips"], 0.15),
    "torso":       (["Spine1"], 0.15),
    "head":        (["Head"], None),  # diagnostic only, never anchored -- no threshold
}


def load_landmark_heights(bvh_path: Path) -> dict[str, np.ndarray]:
    frames, _ = load_bvh_file(str(bvh_path), format="lafan1")
    T = len(frames)
    heights = {}
    for lm, (bones, _thr) in LANDMARKS.items():
        z = np.stack([np.array([f[b][0][2] for f in frames]) for b in bones], axis=0)
        heights[lm] = z.min(axis=0)  # (T,)
    return heights


def report(bvh_paths):
    print(f"{'clip':<28}{'landmark':<12}{'min':>7}{'p1':>7}{'p5':>7}{'p25':>7}{'median':>8}{'p75':>7}")
    for path in bvh_paths:
        heights = load_landmark_heights(path)
        for lm in LANDMARKS:
            z = heights[lm]
            print(f"{path.stem:<28}{lm:<12}{z.min():>7.3f}{np.percentile(z,1):>7.3f}"
                  f"{np.percentile(z,5):>7.3f}{np.percentile(z,25):>7.3f}"
                  f"{np.median(z):>8.3f}{np.percentile(z,75):>7.3f}")


def detect(bvh_path: Path, thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    heights = load_landmark_heights(bvh_path)
    zones = {}
    for lm, (_bones, default_thr) in LANDMARKS.items():
        thr = thresholds.get(lm, default_thr)
        if thr is None:  # head: diagnostic only
            continue
        zones[lm] = heights[lm] < thr
    return zones, heights


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bvh_files", nargs="+", type=Path)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output NPZ dir (one file per input clip) when not --report-only.")
    args = ap.parse_args()

    if args.report_only:
        report(args.bvh_files)
        return

    thresholds = {lm: thr for lm, (_b, thr) in LANDMARKS.items() if thr is not None}
    for path in args.bvh_files:
        zones, heights = detect(path, thresholds)
        print(f"\n{path.stem}:")
        for lm, z in zones.items():
            print(f"  {lm:<12} zone {z.mean()*100:5.1f}%  (thr={thresholds[lm]:.2f}m)")
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            out_path = args.out / f"{path.stem}.npz"
            save_dict = {f"zone_{lm}": z for lm, z in zones.items()}
            save_dict.update({f"height_{lm}": h for lm, h in heights.items()})
            save_dict["thresholds_keys"] = np.array(list(thresholds.keys()))
            save_dict["thresholds_vals"] = np.array(list(thresholds.values()))
            np.savez_compressed(out_path, **save_dict)
            print(f"  -> {out_path}")


if __name__ == "__main__":
    main()

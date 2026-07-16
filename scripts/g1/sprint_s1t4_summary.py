#!/usr/bin/env python3
"""S1-T4: aggregate scripts/g1/sprint_eval_batch.py's CSV by motion class
(locomotion vs floor-contact, hipZ p5 < 0.3 -- T2's exact screening convention,
GMR-baseline-plan.md SPRINT S1-T4) and print a markdown summary for
planLogGMR.md. Stdlib only (no pandas in the `gmr` conda env).

Usage:
    conda run -n gmr python scripts/g1/sprint_s1t4_summary.py \\
        outputs/gmr_baseline/sprint/s1t3_eval.csv
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

FLOOR_THRESHOLD = 0.3
METRICS = ["floorPen_max_cm", "pen_pct", "floatMax_cm", "float_pct",
           "collPct_vetted", "collPeak_vetted_cm", "vMax", "n_spikes",
           "faith_mean_cm", "faith_max_cm"]


def load_reclass(reclass_path: Path) -> dict[str, bool]:
    with open(reclass_path, newline="") as f:
        return {r["clip"]: bool(int(r["floor_class"])) for r in csv.DictReader(f)}


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "outputs/gmr_baseline/sprint/s1t3_eval.csv")
    reclass_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "outputs/gmr_baseline/sprint/s1t4_reclass.csv")
    rows = list(csv.DictReader(open(csv_path, newline="")))

    reclass = load_reclass(reclass_path) if reclass_path.exists() else None
    if reclass is not None:
        for r in rows:
            r["cls"] = "floor" if reclass.get(r["clip"], False) else "locomotion"
        cls_label = f"multi-surface contact (sustained >=1s, non-foot) [{reclass_path.name}]"
    else:
        for r in rows:
            r["cls"] = "floor" if float(r["hipZ_p5"]) < FLOOR_THRESHOLD else "locomotion"
        cls_label = f"hipZ p5 < {FLOOR_THRESHOLD} (fallback -- {reclass_path.name} not found)"

    clips = {r["clip"] for r in rows}
    floor_clips = {r["clip"] for r in rows if r["cls"] == "floor"}
    loco_clips = clips - floor_clips
    print(f"# S1-T4 summary ({csv_path})\n")
    print(f"Classification: {cls_label}\n")
    print(f"{len(clips)} clips total: {len(floor_clips)} floor-class, "
          f"{len(loco_clips)} locomotion-class.\n")

    for cls, cls_clips in [("locomotion", loco_clips), ("floor", floor_clips)]:
        print(f"## {cls} class (n={len(cls_clips)} clips)\n")
        print(f"| variant | n | {' | '.join(METRICS)} |")
        print(f"|---|---|{'---|' * len(METRICS)}")
        for variant in ["raw", "gmrfix", "polished"]:
            vs = [r for r in rows if r["cls"] == cls and r["variant"] == variant]
            if not vs:
                continue
            means = []
            for m in METRICS:
                vals = [float(r[m]) for r in vs if r[m] not in ("", "nan")]
                means.append(f"{sum(vals) / len(vals):.2f}" if vals else "n/a")
            print(f"| {variant} | {len(vs)} | {' | '.join(means)} |")
        print()

    per_clip_variants = defaultdict(set)
    for r in rows:
        per_clip_variants[r["clip"]].add(r["variant"])
    incomplete = {c: v for c, v in per_clip_variants.items() if len(v) < 3}
    if incomplete:
        print(f"## Incomplete clips ({len(incomplete)}, missing a variant)\n")
        for clip, have in sorted(incomplete.items()):
            print(f"- {clip}: has {sorted(have)}")
    else:
        print(f"All {len(clips)} clips have all 3 variants.")


if __name__ == "__main__":
    main()

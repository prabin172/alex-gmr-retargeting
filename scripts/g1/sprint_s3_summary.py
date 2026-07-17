#!/usr/bin/env python3
"""S3 summary: aggregate sprint_s3_full_corpus.py's combined GMR+OURS CSV by
motion class (locomotion vs floor-contact, s1t4_reclass.csv convention) and
print a markdown table for GMR-baseline-plan.md / wiki.

Usage:
    conda run -n gmr python scripts/g1/sprint_s3_summary.py \\
        outputs/gmr_baseline/sprint/s3_full_corpus.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

VARIANTS = ["gmr_raw", "gmr_heightfix", "gmr_polished",
            "ours_raw", "ours_stageA", "ours_ctground"]
METRICS = ["floorPen_cm", "pen_pct", "coll_pct", "coll_peak_cm",
           "held_left_foot_median_cm", "held_left_foot_frac3_pct",
           "held_right_foot_median_cm", "held_right_foot_frac3_pct"]


def load_reclass(reclass_path: Path) -> dict[str, bool]:
    with open(reclass_path, newline="") as f:
        return {r["clip"]: bool(int(r["floor_class"])) for r in csv.DictReader(f)}


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "outputs/gmr_baseline/sprint/s3_full_corpus.csv")
    reclass_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "outputs/gmr_baseline/sprint/s1t4_reclass.csv")
    rows = list(csv.DictReader(open(csv_path, newline="")))

    reclass = load_reclass(reclass_path) if reclass_path.exists() else None
    if reclass is not None:
        for r in rows:
            r["cls"] = "floor" if reclass.get(r["clip"], False) else "locomotion"
        cls_label = f"multi-surface contact reclass [{reclass_path.name}]"
    else:
        cls_label = "NO RECLASS FOUND -- all clips lumped as 'locomotion'"
        for r in rows:
            r["cls"] = "locomotion"

    clips = {r["clip"] for r in rows}
    floor_clips = {r["clip"] for r in rows if r["cls"] == "floor"}
    loco_clips = clips - floor_clips
    print(f"# S3 full-corpus summary ({csv_path})\n")
    print(f"Classification: {cls_label}\n")
    print(f"{len(clips)} clips total: {len(floor_clips)} floor-class, "
          f"{len(loco_clips)} locomotion-class.\n")

    for cls, cls_clips in [("locomotion", loco_clips), ("floor", floor_clips)]:
        print(f"## {cls} class (n={len(cls_clips)} clips)\n")
        print(f"| variant | n | {' | '.join(METRICS)} |")
        print(f"|---|---|{'---|' * len(METRICS)}")
        for variant in VARIANTS:
            vs = [r for r in rows if r["cls"] == cls and r["variant"] == variant]
            if not vs:
                continue
            means = []
            for m in METRICS:
                vals = [float(r[m]) for r in vs if r[m] not in ("", "nan")]
                means.append(f"{sum(vals) / len(vals):.2f}" if vals else "n/a")
            print(f"| {variant} | {len(vs)} | {' | '.join(means)} |")
        print()

    per_clip_variants = {}
    for r in rows:
        per_clip_variants.setdefault(r["clip"], set()).add(r["variant"])
    incomplete = {c: v for c, v in per_clip_variants.items() if len(v) < len(VARIANTS)}
    if incomplete:
        print(f"## Incomplete clips ({len(incomplete)}, missing a variant)\n")
        for clip, have in sorted(incomplete.items()):
            print(f"- {clip}: has {sorted(have)}")
    else:
        print(f"All {len(clips)} clips have all {len(VARIANTS)} variants.")


if __name__ == "__main__":
    main()

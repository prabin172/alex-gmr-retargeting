#!/usr/bin/env python3
"""S5-A6 summary: class-split table from s5_full_corpus.csv, same class-split
convention as S1-T4/S3 (s1t4_reclass.csv, multi-surface human-contact detection,
NOT hip-height alone). Prints locomotion-class and floor-class means per variant.

Usage:
    conda run -n gmr python scripts/g1/sprint_s5_summary.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s5_full_corpus.csv"
RECLASS_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s1t4_reclass.csv"

METRICS = ["pen_pct", "floorPen_cm", "coll_pct", "joint_ok_pct",
           "held_left_foot_frac3_pct", "held_right_foot_frac3_pct",
           "held_left_foot_median_cm", "held_right_foot_median_cm"]


def main():
    floor_class = {}
    with open(RECLASS_CSV) as f:
        for row in csv.DictReader(f):
            floor_class[row["clip"]] = int(row["floor_class"])

    rows = defaultdict(list)  # (class, variant) -> list of row dicts
    with open(CORPUS_CSV) as f:
        for row in csv.DictReader(f):
            clip = row["clip"]
            if clip not in floor_class:
                continue
            cls = "floor" if floor_class[clip] else "locomotion"
            rows[(cls, row["variant"])].append(row)

    variants = ["gmr_raw", "gmr_heightfix", "gmr_polished", "gmr_contact"]
    print(f"{'class':<12}{'variant':<16}" + "".join(f"{m:>16}" for m in METRICS))
    for cls in ["locomotion", "floor"]:
        n_clips = len({r["clip"] for v in variants for r in rows[(cls, v)]})
        for v in variants:
            rs = rows[(cls, v)]
            if not rs:
                continue
            means = {m: sum(float(r[m]) for r in rs) / len(rs) for m in METRICS}
            print(f"{cls+f'({n_clips})':<12}{v:<16}" +
                  "".join(f"{means[m]:16.2f}" for m in METRICS))


if __name__ == "__main__":
    main()

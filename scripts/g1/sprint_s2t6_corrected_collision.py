#!/usr/bin/env python3
"""S2-T6 (N1-a): corrected vetted self-collision re-measurement.

Bug being corrected: prior ad-hoc measurements called
`_collision_stats(model, data, qpos, floor_gid=None, ...)` on the COMBINED
model from `g1_model_setup.py`, which contains an INJECTED floor mocap body.
`_collision_stats`'s own docstring requires `floor_gid` to be passed whenever
the model has an injected floor, REGARDLESS of `count_floor` -- otherwise
floor contacts leak into the self-collision count (the floor body is a mocap
child of worldbody, id != 0, so the `b1==0 or b2==0` exclusion misses it).

This script re-measures every OURS "ramped" variant (raw + polished, the
current non-superseded build per clip) and the corresponding GMR-polished
comparison row, with floor_gid passed correctly both times.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from solve_global_trajectory_opt_contactfirst import _collision_stats  # noqa: E402

SPRINT_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/ours_g1"
WEEK1_DIR = REPO_ROOT / "outputs/gmr_baseline/pkl"

CLIPS = ["walk1_subject1", "fallAndGetUp1_subject1", "fallAndGetUp2_subject2", "ground1_subject1"]


def measure(model, data, floor_gid, qpos):
    wrong = _collision_stats(model, data, qpos, floor_gid=None, count_floor=False)
    right = _collision_stats(model, data, qpos, floor_gid=floor_gid, count_floor=False)
    return wrong, right


def main():
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()

    rows = []
    for clip in CLIPS:
        # OURS raw (ramped)
        raw_path = SPRINT_DIR / f"{clip}_ours_ramped.npz"
        pol_path = SPRINT_DIR / f"{clip}_ours_ramped_polished.npz"
        gmr_pol_path = WEEK1_DIR / f"{clip}_polished_constant.pkl"

        for label, path, is_npz in [
            (f"{clip}__OURS_raw", raw_path, True),
            (f"{clip}__OURS_polished", pol_path, True),
            (f"{clip}__GMR_polished", gmr_pol_path, False),
        ]:
            if not path.exists():
                print(f"  [skip] {label}: {path} not found")
                continue
            if is_npz:
                qpos = np.load(path)["qpos"]
            else:
                qpos, _ = load_gmr_pkl(path)
            wrong, right = measure(model, data, floor_gid, qpos)
            rows.append((label, wrong, right))
            print(f"{label:<38} WRONG(floor_gid=None): {wrong['pct']:5.1f}% "
                  f"peak={wrong['max_pen_cm']:6.2f}cm   "
                  f"CORRECTED(floor_gid passed): {right['pct']:5.1f}% "
                  f"peak={right['max_pen_cm']:6.2f}cm")

    return rows


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S8-T1b spot-check: the two known phase-2 blowup frames, before vs after the
rate limiter. Measurement only (decides whether T1a happens):
  - fallAndGetUp2_subject2 t=212: S7-T6 measured left_ankle_roll_link at
    -18.6cm (one-frame spike, ncon=30) in perframelimb.
  - fallAndGetUp1_subject1 t=2251: S7-T7's known residual -- held-right-foot
    support_z +31.42cm in gmr_contact_fc (phase 2 disrupting the held lock).

Prints the watched body's lowest-z (and right-foot support_z for the t=2251
case) over a +-4-frame window for the before/after pkl pairs.

Usage:
    conda run -n gmr python scripts/g1/sprint_s8_t1b_spotcheck.py
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
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from leg_floor_clamp import _lowest_point  # noqa: E402
from stage_b_g1 import support_z  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402

PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--suffix", type=str, default="rl")
_args, _ = _ap.parse_known_args()
_SFX = _args.suffix

CASES = [
    # (clip, frame, watch_body, before_suffix, after_suffix, use_support_z)
    ("fallAndGetUp2_subject2", 212, "left_ankle_roll_link",
     "_perframelimb", f"_perframelimb_{_SFX}", False),
    ("fallAndGetUp1_subject1", 2251, "right_ankle_roll_link",
     "_gmrcontact_fc", f"_gmrcontact_fc_{_SFX}", True),
]


def ncon_nonfloor(model, data, floor_gid):
    n = 0
    for cc in range(data.ncon):
        ct = data.contact[cc]
        if ct.geom1 == floor_gid or ct.geom2 == floor_gid:
            continue
        b1 = int(model.geom_bodyid[ct.geom1])
        b2 = int(model.geom_bodyid[ct.geom2])
        if b1 == 0 or b2 == 0:
            continue
        n += 1
    return n


def main():
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)

    for clip, frame, body_name, bsuf, asuf, use_sz in CASES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print(f"\n=== {clip} t={frame} watch={body_name} ===")
        for label, suffix in [("BEFORE", bsuf), ("AFTER", asuf)]:
            p = PKL_S5_DIR / f"{clip}{suffix}.pkl"
            if not p.exists():
                print(f"  {label} ({suffix}): pkl missing")
                continue
            qpos, fps = load_gmr_pkl(p)
            print(f"  {label} ({suffix}):")
            for t in range(max(0, frame - 4), min(qpos.shape[0], frame + 5)):
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                mujoco.mj_collision(model, data)
                _, z = _lowest_point(model, data, mesh_cache, bid)
                nc = ncon_nonfloor(model, data, floor_gid)
                sz = support_z(model, data, mesh_cache, bid) if use_sz else None
                mark = " <-- flagged frame" if t == frame else ""
                sz_str = f" support_z={sz*100:+7.2f}cm" if sz is not None else ""
                print(f"    t={t:5d}: lowest_z={z*100:+7.2f}cm{sz_str} ncon={nc:3d}{mark}")


if __name__ == "__main__":
    main()

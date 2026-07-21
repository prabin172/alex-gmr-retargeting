#!/usr/bin/env python3
"""S6: promoted from session scratchpad (was scratchpad/range_summary.py) so it
survives past one conversation. Computes, per class (s1t4_reclass.csv) and per
variant, the held-foot support_z RANGE (worst float minus worst penetration,
per clip, averaged) -- the number Prabin asked to track directly: does a
mechanism collapse the spread between worst-case float and worst-case
penetration, not just move it around (a rigid Z-shift provably cannot change
this number, see GMR-S6-plan.md "Why S6 exists").

Usage (corpus-wide, from s5_full_corpus-style variant pkl layout):
    conda run -n gmr python scripts/g1/sprint_s6_range_summary.py

Or import `held_range_table(model, data, mesh_cache, role_bid, variants, clips)`
for a custom clip/variant set (e.g. S6-A4's 5 dev clips + gmr_contact_fc).
"""
from __future__ import annotations

import sys
import pathlib

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sprint_s3_full_corpus import FOOT_POS_ROLE  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import load_canonical  # noqa: E402
from contact_labels import debounce_flags  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from stage_b_g1 import support_z  # noqa: E402


def compute_held_mask(grounded_npz_path):
    (roles, role_to_idx, src_positions, fps, ori_roles, ori_to_idx, ori_mats,
     contacts, eff_names) = load_canonical(grounded_npz_path)
    T = src_positions.shape[0]
    contacts_solved = {eff: debounce_flags(contacts[eff], 2) for eff in eff_names}
    held = {}
    for eff, role in FOOT_POS_ROLE.items():
        src_pt = src_positions[:, role_to_idx[role]]
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(src_pt, axis=0), axis=1) * fps
        v[0] = v[1] if T > 1 else 0.0
        held[eff] = contacts_solved[eff] & (v < 0.05)
    return held, T


def clip_worst_float_pen(model, data, mesh_cache, role_bid, held, qpos):
    """(worst_float_cm, worst_pen_cm) across all held foot-frames in one clip.
    worst_pen is negative (more negative = deeper penetration)."""
    all_sz = []
    for eff, role in FOOT_POS_ROLE.items():
        idx = np.where(held[eff])[0]
        for t in idx:
            if t >= qpos.shape[0]:
                continue
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            all_sz.append(support_z(model, data, mesh_cache, role_bid[role]))
    if not all_sz:
        return None
    all_sz = np.array(all_sz) * 100
    return float(all_sz.max()), float(all_sz.min())


def held_range_table(model, data, mesh_cache, role_bid, variants: dict, clips: dict):
    """variants: {name: {clip: pkl_path}}. clips: {clip: grounded_npz_path}.
    Returns {(variant, clip): (worst_float, worst_pen, range)}."""
    held_cache = {clip: compute_held_mask(p)[0] for clip, p in clips.items()}
    out = {}
    for vname, per_clip in variants.items():
        for clip, pkl_path in per_clip.items():
            qpos, fps = load_gmr_pkl(pkl_path)
            res = clip_worst_float_pen(model, data, mesh_cache, role_bid, held_cache[clip], qpos)
            if res is None:
                continue
            wf, wp = res
            out[(vname, clip)] = (wf, wp, wf - wp)
    return out


if __name__ == "__main__":
    import csv
    from g1_model_setup import load_g1_model_with_vetted_collision_and_floor
    from post_process_ground_contactfirst import _build_mesh_cache
    from sprint_s3_full_corpus import ROLE_TO_G1_BODY

    ROOT = pathlib.Path(__file__).resolve().parents[2]
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}

    floor_class = {}
    with open(ROOT / "outputs/gmr_baseline/sprint/s1t4_reclass.csv") as f:
        for row in csv.DictReader(f):
            floor_class[row["clip"]] = int(row["floor_class"])

    clips = {}
    for clip in floor_class:
        p = ROOT / f"outputs/gmr_baseline/sprint/canonical_human_s5/{clip}_lafan1c_grounded.npz"
        if p.exists():
            clips[clip] = p

    variants = {}
    for vname, subdir, suffix in [
        ("gmr_raw", "pkl", ""), ("gmr_heightfix", "pkl", "_gmrfix"),
        ("gmr_contact", "pkl_s5", "_gmrcontact"),
    ]:
        variants[vname] = {}
        for clip in clips:
            p = ROOT / f"outputs/gmr_baseline/sprint/{subdir}/{clip}{suffix}.pkl"
            if p.exists():
                variants[vname][clip] = p

    table = held_range_table(model, data, mesh_cache, role_bid, variants, clips)

    print(f"{'class':<12}{'variant':<16}{'mean worst float':>18}{'mean worst pen':>18}{'mean range':>14}")
    for cls in ["locomotion", "floor"]:
        for vname in variants:
            rows = [table[(vname, c)] for c in clips
                    if floor_class[c] == (1 if cls == "floor" else 0) and (vname, c) in table]
            if not rows:
                continue
            wf = np.mean([r[0] for r in rows])
            wp = np.mean([r[1] for r in rows])
            rg = np.mean([r[2] for r in rows])
            print(f"{cls:<12}{vname:<16}{wf:18.2f}{wp:18.2f}{rg:14.2f}")

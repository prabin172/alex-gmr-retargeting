#!/usr/bin/env python3
"""S2-T8: full variant comparison on the gmrscale outputs (post S2-T7 scale fix).

Promoted from the session-local validation script that produced the tables in
planLogGMR.md's `## S2-T6 -- CORRECT baseline framing` and `## S2-T7` entries,
with the OURS paths updated from `_ours_floorfix` to `_ours_gmrscale` variants.
Variants (the correctly-framed comparison, per the standing rules in
GMR-baseline-plan.md):
  - GMR+heightfix: their own published method (the fair baseline column)
  - GMR+ourpolish: our Z-fix on their raw output
  - OURS raw / +StageA / +StageA+ctground: the gmrscale build's three stages

Metrics: whole-clip floorPen max / pen% / vetted self-collision% (floor_gid
ALWAYS passed -- the combined model has an injected floor, see planLogGMR.md
N1-a), plus held-frame support_z (median + frac<3cm), the discriminating
metric. Held-frame definition identical to all S2-T5/T6/T7 audits: debounced
human contact zone AND source marker speed < 0.05 m/s.

Usage:
    conda run -n gmr python scripts/g1/eval_g1_gmrscale_variants.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "g1"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mujoco
import numpy as np
from contact_labels import debounce_flags
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z
from solve_fbx_canonical_alex_contactfirst import load_canonical
from solve_global_trajectory_opt_contactfirst import _collision_stats
from solve_lafan1_canonical_g1_contactfirst import ROLE_TO_G1_BODY, FOOT_POS_ROLE
from stage_b_g1 import support_z
from load_gmr_pkl import load_gmr_pkl

CANON = {
    "walk1_subject1": "walk1_subject1_v3_grounded.npz",
    "fallAndGetUp1_subject1": "fallAndGetUp1_subject1_grounded.npz",
    "fallAndGetUp2_subject2": "fallAndGetUp2_subject2_grounded.npz",
    "ground1_subject1": "ground1_subject1_grounded.npz",
}

model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
mesh_cache = _build_mesh_cache(model)
role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
           for role, name in ROLE_TO_G1_BODY.items()}
geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
           and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]


def whole_clip_metrics(qpos):
    lowest = np.zeros(qpos.shape[0])
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)
    pen = np.maximum(0, -lowest)
    cs = _collision_stats(model, data, qpos, floor_gid=floor_gid, count_floor=False)
    return dict(floorPen=pen.max() * 100, pen_pct=100 * (pen > 0.005).mean(),
               coll_pct=cs["pct"], coll_peak=cs["max_pen_cm"])


def _npz(rel):
    return lambda clip: np.load(
        REPO_ROOT / f"outputs/gmr_baseline/sprint/ours_g1/{clip}{rel}.npz")["qpos"]


VARIANTS = [
    ("GMR+heightfix (their described method)",
     lambda clip: load_gmr_pkl(REPO_ROOT / f"outputs/gmr_baseline/pkl_w2/{clip}_gmrfix.pkl")[0]),
    ("GMR+ourpolish (our Z-fix on their raw)",
     lambda clip: load_gmr_pkl(REPO_ROOT / f"outputs/gmr_baseline/pkl/{clip}_polished_constant.pkl")[0]),
    ("OURS gmrscale raw", _npz("_ours_gmrscale")),
    ("OURS gmrscale +StageA", _npz("_ours_gmrscale_stageA")),
    ("OURS gmrscale +StageA+ctground", _npz("_ours_gmrscale_ctground")),
]

for clip, canon_name in CANON.items():
    canon = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human" / canon_name
    (roles, role_to_idx, src_positions, fps, orientation_roles, ori_to_idx, orientation_mats,
     persisted_contacts, persisted_eff_names) = load_canonical(canon)
    T = src_positions.shape[0]
    contacts_solved = {eff: debounce_flags(persisted_contacts[eff], 2)
                       for eff in persisted_eff_names}
    held = {}
    for eff, role in FOOT_POS_ROLE.items():
        src_ankle = src_positions[:, role_to_idx[role]]
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(src_ankle, axis=0), axis=1) * fps
        v[0] = v[1] if T > 1 else 0.0
        held[eff] = contacts_solved[eff] & (v < 0.05)

    print(f"\n{'='*78}\n{clip}\n{'='*78}")
    qpos_by_variant = {}
    for label, loader in VARIANTS:
        try:
            qpos_by_variant[label] = loader(clip)
        except FileNotFoundError as e:
            print(f"  [missing] {label}: {e}")

    print("Whole-clip metrics:")
    for label, qpos in qpos_by_variant.items():
        m = whole_clip_metrics(qpos)
        print(f"  {label:<40} floorPen={m['floorPen']:6.2f}cm pen%={m['pen_pct']:5.1f}% "
              f"coll%={m['coll_pct']:5.1f}% peak={m['coll_peak']:.2f}cm")

    print("Held-frame contact quality (support_z vs floor, the discriminating metric):")
    for eff, role in FOOT_POS_ROLE.items():
        bid = role_bid[role]
        held_idx = np.where(held[eff])[0]
        if held_idx.size == 0:
            print(f"  {eff}: no held frames")
            continue
        print(f"  {eff} ({held_idx.size} held frames):")
        for label, qpos in qpos_by_variant.items():
            szs = []
            for t in held_idx:
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                szs.append(support_z(model, data, mesh_cache, bid))
            szs = np.array(szs)
            print(f"    {label:<40} median={np.median(szs)*100:+7.2f}cm  "
                  f"frac<3cm={np.mean(np.abs(szs)<0.03)*100:5.1f}%")

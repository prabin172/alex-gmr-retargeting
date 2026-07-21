#!/usr/bin/env python3
"""S7-T6: torso/waist residual probe -- exploratory, 2-attempt cap, NOT a
mechanism commitment (GMR-S7-plan.md Phase T6). perframelimb (S7-T3) is the
best floor-class mechanism shipped so far, but still leaves real torso_link
penetration on the hardest floor-class clips (found via a per-body worst-z
diagnostic, not assumed: fallAndGetUp2_subject2 torso_link -3.6cm@frame400,
fallAndGetUp1_subject1 torso_link -3.5cm@frame3256 -- pelvis penetration also
present but out of scope, see leg_floor_clamp.py's "waist" EFF_BODY comment,
pelvis is the free-joint root body here, not reachable via the waist chain).

Mechanism: ONE extra `leg_floor_clamp.clamp_limb` pass per frame, eff="waist"
(new chain: waist_yaw/roll/pitch), watch_body="torso_link" (the chain's only
reachable effector), floor_margin=0.0, clearance-only mode (target_xy=None,
matches CLAMP_TARGETS' own convention -- no held-target semantics for torso).
Applied AFTER the existing perframelimb output (does not touch the shipped
perframelimb pkl or its generator -- reads pkl_s5/*_perframelimb.pkl, writes
nowhere, this is eval-only for the probe decision).

Usage:
    conda run -n gmr python scripts/g1/sprint_s7_t6_waist_probe.py
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
from solve_fbx_canonical_alex_contactfirst import load_canonical  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY, FOOT_POS_ROLE, whole_clip_metrics, held_metrics  # noqa: E402
from sprint_s5_metrics import joint_ok_pct  # noqa: E402
from sprint_s6_range_summary import clip_worst_float_pen  # noqa: E402
from contact_labels import debounce_flags  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from leg_floor_clamp import build_chain_dofs, clamp_limb  # noqa: E402

PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
CANON_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/canonical_human_s5"

# Picked from a per-body worst-z sweep over ALL 34 floor-class clips' perframelimb
# output (not eyeballed): the 2 clips with the deepest measured torso_link penetration.
PROBE_CLIPS = ["fallAndGetUp2_subject2", "fallAndGetUp1_subject1"]


def apply_waist_clamp(model, data, mesh_cache, qpos, avoid_self_collision=False):
    chain = build_chain_dofs(model, "waist")
    out = qpos.copy()
    n_corrected = 0
    for t in range(out.shape[0]):
        data.qpos[:] = out[t]
        mujoco.mj_forward(model, data)
        applied = clamp_limb(model, data, mesh_cache, "waist", chain,
                              floor_margin=0.0, target_xy=None, max_iters=10,
                              avoid_self_collision=avoid_self_collision, coll_weight=0.5)
        if applied:
            out[t] = data.qpos.copy()
            n_corrected += 1
    return out, n_corrected


def eval_clip(model, data, mesh_cache, geom_ids, floor_gid, role_bid, clip, qpos):
    grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
    (roles, role_to_idx, src_positions, fps, ori_roles, ori_to_idx, ori_mats,
     contacts, eff_names) = load_canonical(grounded)
    T = src_positions.shape[0]
    contacts_solved = {eff: debounce_flags(contacts[eff], 2) for eff in eff_names}
    held = {}
    for eff, role in FOOT_POS_ROLE.items():
        src_pt = src_positions[:, role_to_idx[role]]
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(src_pt, axis=0), axis=1) * fps
        v[0] = v[1] if T > 1 else 0.0
        held[eff] = contacts_solved[eff] & (v < 0.05)

    wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
    jok, n_held = joint_ok_pct(model, data, mesh_cache, geom_ids, role_bid, held, qpos)
    wf, wp = clip_worst_float_pen(model, data, mesh_cache, role_bid, held, qpos) or (float("nan"), float("nan"))
    return dict(joint_ok_pct=jok, range_cm=wf - wp, **wm)


def main():
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    for clip in PROBE_CLIPS:
        qpos, fps = load_gmr_pkl(PKL_S5_DIR / f"{clip}_perframelimb.pkl")
        before = eval_clip(model, data, mesh_cache, geom_ids, floor_gid, role_bid, clip, qpos)
        for attempt, avoid_sc in [(1, False), (2, True)]:
            qpos_waist, n_corrected = apply_waist_clamp(model, data, mesh_cache, qpos,
                                                          avoid_self_collision=avoid_sc)
            after = eval_clip(model, data, mesh_cache, geom_ids, floor_gid, role_bid, clip, qpos_waist)
            print(f"\n=== {clip} attempt {attempt} (avoid_self_collision={avoid_sc}, "
                  f"waist-clamped {n_corrected}/{qpos.shape[0]} frames) ===")
            for k in ["joint_ok_pct", "floorPen_cm", "pen_pct", "coll_pct", "coll_peak_cm", "range_cm"]:
                print(f"  {k:14} {before[k]:8.3f} -> {after[k]:8.3f}")


if __name__ == "__main__":
    main()

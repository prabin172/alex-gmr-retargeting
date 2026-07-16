#!/usr/bin/env python3
"""S2-T6 (non-contact-frame residual fix, 2026-07-16): contact-aware grounding
for OURS -- corrects the whole-clip floorPen residual on frames with NO
detected contact, WITHOUT disturbing the already-validated-good held/contact
frames (median support_z within ~1cm of the floor, S2-T6's held-frame audit).

Reuses `_solve_lift_qp` (imported UNCHANGED from `post_process_ground_
contactfirst.py` -- the SAME smooth per-frame lift QP already used by that
script's "hybrid" grounding mode) with a NEW cap rule:
  - held frame (any effector's contact-zone stillness anchor active): cap=0.0
    -- the lift QP is FORBIDDEN from moving this frame at all, since Stage 3's
    own pull-to-floor mechanism already placed it correctly.
  - non-held frame: cap=+inf -- free to lift as much as needed to remove
    penetration.
The QP's own smoothness term (banded second-difference penalty on the lift
curve) keeps the transition between "don't touch" and "fully correct" regions
smooth by construction -- this is NOT the same failure mode as pull-to-floor's
earlier per-frame independent blend (which had no global smoothness objective
tying frames together), it's a single QP solve over the WHOLE clip's lift
trajectory at once.

Usage:
    conda run -n gmr python scripts/g1/ground_ours_contact_aware.py \\
        --in outputs/gmr_baseline/sprint/ours_g1/walk1_subject1_ours_floorfix_stageAonly.npz \\
        --canonical outputs/gmr_baseline/sprint/canonical_human/walk1_subject1_v3_grounded.npz \\
        --out outputs/gmr_baseline/sprint/ours_g1/walk1_subject1_ours_contactground.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # NOT repo root -- see planLogGMR.md T1

from contact_labels import debounce_flags  # noqa: E402
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import (  # noqa: E402
    _build_mesh_cache, _robot_lowest_z, _solve_lift_qp)
from solve_fbx_canonical_alex_contactfirst import load_canonical  # noqa: E402
from solve_lafan1_canonical_g1_contactfirst import FOOT_POS_ROLE  # noqa: E402


def compute_held_mask(canonical_path: Path, T: int, plant_speed: float = 0.05,
                      contact_min_run: int = 2) -> np.ndarray:
    """Same held-frame definition used throughout S2-T5/S2-T6's audits: a foot
    effector's human-side contact zone (debounced) AND its source marker speed
    below plant_speed. held_mask[t] = True if ANY foot is held at frame t."""
    (roles, role_to_idx, src_positions, fps, orientation_roles, ori_to_idx, orientation_mats,
     persisted_contacts, persisted_eff_names) = load_canonical(canonical_path)
    assert src_positions.shape[0] == T, (
        f"canonical clip length {src_positions.shape[0]} != qpos length {T}")

    held_mask = np.zeros(T, dtype=bool)
    for eff, role in FOOT_POS_ROLE.items():
        if eff not in persisted_eff_names:
            continue
        zone = debounce_flags(persisted_contacts[eff], contact_min_run)
        src_pt = src_positions[:, role_to_idx[role]]
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(src_pt, axis=0), axis=1) * fps
        v[0] = v[1] if T > 1 else 0.0
        held_mask |= zone & (v < plant_speed)
    return held_mask


def contact_aware_ground(qpos: np.ndarray, model, held_mask: np.ndarray,
                         smooth: float = 1e4) -> np.ndarray:
    T = qpos.shape[0]
    data = mujoco.MjData(model)
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom)
               if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    lowest = np.zeros(T)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)

    need = np.maximum(0.0, -lowest)
    cap = np.where(held_mask, 0.0, np.inf)
    lift = _solve_lift_qp(need, cap, smooth)

    qpos_out = qpos.copy()
    qpos_out[:, 2] += lift
    return qpos_out, lift, lowest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--canonical", required=True, type=Path,
                    help="The grounded canonical-human NPZ used to solve this clip (S2-T1/T2 "
                         "output) -- needed to re-derive the held-frame mask.")
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--smooth", type=float, default=1e4,
                    help="Lift-curve smoothness weight (same convention/default as hybrid "
                         "grounding's --lift-smooth in post_process_ground_contactfirst.py).")
    ap.add_argument("--plant-speed", type=float, default=0.05)
    args = ap.parse_args()

    d = np.load(args.in_path)
    qpos, fps = d["qpos"], float(d["fps"])

    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    held_mask = compute_held_mask(args.canonical, qpos.shape[0], args.plant_speed)
    print(f"  held frames: {held_mask.sum()}/{qpos.shape[0]} ({held_mask.mean()*100:.1f}%)")

    qpos_out, lift, lowest = contact_aware_ground(qpos, model, held_mask, args.smooth)
    print(f"  lift stats: max={lift.max()*100:.2f}cm at non-held frames, "
          f"lift at held frames max={np.abs(lift[held_mask]).max()*100 if held_mask.any() else 0:.4f}cm "
          f"(should be ~0)")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_path, qpos=qpos_out, fps=np.float64(fps))
    print(f"Wrote {args.out_path} (contact-aware-ground, {qpos_out.shape[0]} frames)")


if __name__ == "__main__":
    main()

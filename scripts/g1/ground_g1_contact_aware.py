#!/usr/bin/env python3
"""W2-T7: contact-aware grounding on G1, using W2-T3's human-side foot contact
labels -- the thin adapter the plan anticipated ("If the script hard-requires
Alex-specific bits (sole sites, contact NPZ keys), write the thin adapter in
scripts/g1/, do NOT fork the QP/mesh code").

`post_process_ground_contactfirst.py`'s `hybrid`/`constant-contact` modes need
per-foot planted-sole-height samples via `_foot_plant_frames`, which hard-codes
Alex's NAMED `SOLE_CORNER_SITES` (a site-based Z lookup) -- G1 has no named
sole sites (E4's discovery: only unnamed sphere geoms), so that function always
falls back to the same global-lowest constant week 1 already tested, giving
zero improvement. This script re-derives the SAME `plant_data` shape
`_foot_plant_frames` produces (per foot: min_z (T,), labelled (T,), still (T,))
using G1's own sole-corner sphere geoms + our W2-T3 human-side foot contact
labels, then hands off to `_planted_foot_sole_samples`/`_solve_lift_qp`
(imported UNCHANGED) for everything downstream -- the QP/mesh code itself is
never forked, only the Alex-specific site lookup is replaced.

Usage:
    conda run -n gmr python scripts/g1/ground_g1_contact_aware.py \\
        --in outputs/gmr_baseline/pkl/fallAndGetUp2_subject2_stageA.pkl \\
        --out outputs/gmr_baseline/pkl_w2/fallAndGetUp2_subject2_ground_hybrid.pkl \\
        --human-contacts outputs/gmr_baseline/human_contacts/fallAndGetUp2_subject2.npz \\
        --mode hybrid
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
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from polish_gmr_pkl import save_gmr_pkl  # noqa: E402
from post_process_ground_contactfirst import (  # noqa: E402
    _build_mesh_cache, _planted_foot_sole_samples, _robot_lowest_z, _solve_lift_qp)
from stage_b_g1 import G1_CONTACT_GEOM, _foot_sole_geom_ids  # noqa: E402

G1_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
HUMAN_CONTACTS_DEFAULT_DIR = REPO_ROOT / "outputs" / "gmr_baseline" / "human_contacts"


def _g1_plant_data(qpos, model, data, human_zones, fps, still_speed=0.05):
    """Same shape as _foot_plant_frames' return: {col: {min_z, labelled, still}},
    but min_z from G1's own sole-corner SPHERE geoms (not sites) and labelled
    from W2-T3's human-side foot zone (not the robot's own detector)."""
    eff_names = ["left_foot", "right_foot"]
    sole_geoms = {eff: _foot_sole_geom_ids(model, G1_CONTACT_GEOM[eff]["body"])
                 for eff in eff_names}
    T = qpos.shape[0]
    min_z = {eff: np.zeros(T) for eff in eff_names}
    body_pos = {eff: np.zeros((T, 3)) for eff in eff_names}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for eff in eff_names:
            pts = np.array([data.geom_xpos[g] for g in sole_geoms[eff]])
            min_z[eff][t] = pts[:, 2].min()
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, G1_CONTACT_GEOM[eff]["body"])
            body_pos[eff][t] = data.xpos[bid]

    out = {}
    for i, eff in enumerate(eff_names):
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(body_pos[eff], axis=0), axis=1) * fps
        v[0] = v[1] if T > 1 else 0.0
        labelled = human_zones[eff].astype(bool)
        out[i] = {"min_z": min_z[eff], "labelled": labelled,
                  "still": labelled & (v < still_speed)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--human-contacts", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=G1_MODEL_DEFAULT)
    ap.add_argument("--mode", choices=["constant-contact", "hybrid"], default="constant-contact")
    ap.add_argument("--contact-percentile", type=float, default=50.0)
    ap.add_argument("--still-speed", type=float, default=0.05)
    ap.add_argument("--lift-smooth", type=float, default=1e4)
    ap.add_argument("--lift-float-tol", type=float, default=0.005)
    args = ap.parse_args()

    if args.human_contacts is None:
        stem = args.in_path.stem
        for suf in ("_stageA", "_polished_constant", "_polished_perframe"):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        args.human_contacts = HUMAN_CONTACTS_DEFAULT_DIR / f"{stem}.npz"

    qpos, fps = load_gmr_pkl(args.in_path)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    zdata = np.load(args.human_contacts)
    human_zones = {"left_foot": zdata["zone_left_foot"], "right_foot": zdata["zone_right_foot"]}
    for eff, z in human_zones.items():
        assert z.shape[0] == qpos.shape[0], f"{eff}: human zone length != robot pkl frames"

    T = qpos.shape[0]
    lowest = np.empty(T)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)

    plant_data = _g1_plant_data(qpos, model, data, human_zones, fps, args.still_speed)
    samples = _planted_foot_sole_samples(plant_data)
    if samples.size:
        floor = float(np.percentile(samples, args.contact_percentile))
        floor_src = f"planted-foot-sole p{args.contact_percentile:g} (n={samples.size})"
    else:
        print("  [warn] no planted-foot samples -- falling back to global-lowest constant")
        floor = float(np.percentile(lowest, 1.0))
        floor_src = "global-lowest (fallback)"
    print(f"  floor = {floor:.4f}m  ({floor_src})")
    shift = np.full(T, -floor)

    if args.mode == "hybrid":
        need = np.maximum(0.0, -(lowest + shift))
        cap = np.full(T, np.inf)
        for d in plant_data.values():
            foot_after = d["min_z"] + shift
            foot_cap = np.maximum(0.0, -foot_after) + args.lift_float_tol
            cap = np.where(d["still"], np.minimum(cap, foot_cap), cap)
        lift = _solve_lift_qp(need, cap, args.lift_smooth)
        shift = shift + lift

    qpos_grounded = qpos.copy()
    qpos_grounded[:, 2] += shift

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gmr_pkl(args.out_path, qpos_grounded, fps)
    print(f"Wrote {args.out_path} ({args.mode}, {T} frames)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E4: Stage B contact-anchoring QP, ported to G1 (feet only, MVP scope).

Reuses `stage_b`, `_compute_anchors`, `_load_model_with_floor`, `_get_joint_limits`
from `solve_global_trajectory_opt_contactfirst.py` COMPLETELY UNCHANGED -- all of
them already take model/data/resolved/role_to_body as arguments, no Alex globals
read inside their bodies (verified by inspection, see planLogGMR.md T-E4). The ONLY
thing that's genuinely Alex-specific is `_resolve_contact_geom`, which reads a
module-level `CONTACT_GEOM` dict of ALEX body names -- forked here as
`_resolve_g1_feet` (~20 lines: same shape of resolved dict, G1 body names, sole_sites
always [] since G1 has no NAMED sole-corner sites -- see note below). This is "port
the naming glue", not "fork the QP/mesh code".

Design (mirrors the already-validated "polish Luigi" recipe,
scripts/ihmc_json_to_stage4_npz.py, wiki/log.md 2026-07-14): the ALREADY-POLISHED
(Stage A + grounded) G1 motion is both the warm start AND its own tracking target
(self-tracking -- "stay close to where you already are") for a minimal role set
(pelvis + both feet), while contact-anchoring terms (higher weight, from
`_build_contact`) do the actual cleanup: pinning each foot's position + flat
orientation during DETECTED ground contact, reducing plant slip/wobble that Stage A's
pure smoothing can't touch (it has no concept of "planted").

Contact detection (GMR gives no contact flags): height+speed gate on each foot's own
FK trajectory, mirroring `contact_labels.py`'s human-side gate but applied to the
ROBOT body instead of a human marker -- see `detect_g1_foot_contacts`. Height
reference = the MIN Z of each foot's 4 small sphere geoms already present on
`left_ankle_roll_link`/`right_ankle_roll_link` in the G1 mocap XML (unnamed, but
positioned exactly like sole corners -- toe/heel x left/right -- discovered by
inspection; NOT the same as Alex's NAMED sole-corner SITES, so `_build_contact`'s
on-floor/flat/coplanar shared-Z refinement (which needs `info["sole_sites"]`, a SITE
list) does not engage this pass -- known, deliberate, documented limitation).

Self-collision is OFF this pass (`lambda_coll=0`) -- G1's mocap-model collision
pairs were flagged as noisy this week (eval_motion.py's own caveat: 18.2%
self-collision on a clean walk clip), so mixing that into a hard-constraint QP risked
contaminating the one new mechanism (contact anchoring) this pass is meant to
isolate and measure. Floor-collision QP rows also OFF (`count_floor=False`) for the
same isolation reason -- grounding (T8) already handles clip-level floor placement;
this pass tests contact anchoring alone.

Usage:
    conda run -n gmr python scripts/g1/stage_b_g1.py \\
        --in outputs/gmr_baseline/pkl/fallAndGetUp2_subject2_polished_constant.pkl \\
        --out outputs/gmr_baseline/pkl/fallAndGetUp2_subject2_stageB.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # NOT repo root -- see planLogGMR.md T1
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from polish_gmr_pkl import save_gmr_pkl  # noqa: E402
from solve_global_trajectory_opt_contactfirst import (  # noqa: E402
    N_ACT, _compute_anchors, _get_joint_limits, _load_model_with_floor, stage_b)

G1_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")

# G1 analog of Alex's CONTACT_GEOM (module global in the imported script) -- feet
# only. Body-frame position pin (kind="foot" => data.xpos[bid], no site needed).
G1_CONTACT_GEOM = {
    "left_foot":  dict(body="left_ankle_roll_link",
                       axis_local=np.array([0.0, 0.0, 1.0]), world_dir=np.array([0.0, 0.0, 1.0])),
    "right_foot": dict(body="right_ankle_roll_link",
                       axis_local=np.array([0.0, 0.0, 1.0]), world_dir=np.array([0.0, 0.0, 1.0])),
}
# Track roles == the two feet + pelvis (root anchor). Self-tracking targets (FK of
# the polished qpos itself) -- MVP scope, not the full Alex canonical-human role set.
G1_TRACK_ROLES = ["pelvis", "left_foot", "right_foot"]
G1_ROLE_BODY = {"pelvis": "pelvis", "left_foot": "left_ankle_roll_link",
                "right_foot": "right_ankle_roll_link"}


def _resolve_g1_feet(model, eff_names):
    """Fork of `_resolve_contact_geom`, G1 body names, sole_sites always []
    (no named sole-corner sites on G1 -- see module docstring)."""
    resolved = {}
    for eff in eff_names:
        g = G1_CONTACT_GEOM[eff]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, g["body"])
        assert bid >= 0, f"G1 body {g['body']!r} not found"
        resolved[eff] = dict(body_id=bid, site_id=-1, kind="foot",
                             axis_local=g["axis_local"], world_dir=g["world_dir"],
                             sole_sites=[])
    return resolved


def _foot_sole_geom_ids(model, ankle_body_name):
    """The 4 small sphere geoms on left/right_ankle_roll_link (unnamed in the
    XML, positioned at toe/heel x left/right corners -- discovered by inspection,
    same PURPOSE as Alex's named sole-corner sites, just geoms not sites)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ankle_body_name)
    return [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) == bid
            and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)]


def detect_g1_foot_contacts(qpos, model, data, fps, height_thresh=0.05):
    """Per-foot per-frame contact flag from the ROBOT's own FK (no external
    contact source -- GMR gives none). A foot is "in the contact zone" when its
    sole (min of its 4 corner-sphere Z's) sits within `height_thresh` of the
    floor. HEIGHT ONLY, deliberately -- no speed gate here. Debug sweep found
    the sole-point speed during near-ground frames on walk1_subject1 was, if
    anything, HIGHER than the unconditional median (a heel-to-toe rolling
    contact moves the 4-corner CENTROID even while the true contact patch is
    quasi-stationary -- a geometric artifact of collapsing 4 corners to one
    point, not a real "never stops" motion). The already-imported, unmodified
    `_compute_anchors` does its OWN speed-based stillness sub-segmentation
    inside each contact interval (via its `plant_speed` param, matching Alex's
    exact convention) -- redundant/conflicting gating here would just double
    up on a metric that's noisier at this layer. This function's only job is
    the coarse "is the foot anywhere near the floor" zone."""
    eff_names = ["left_foot", "right_foot"]
    sole_geoms = {eff: _foot_sole_geom_ids(model, G1_CONTACT_GEOM[eff]["body"])
                 for eff in eff_names}
    for eff, ids in sole_geoms.items():
        assert len(ids) == 4, f"{eff}: expected 4 sole-corner sphere geoms, found {len(ids)}"

    T = qpos.shape[0]
    sole_z = {eff: np.zeros(T) for eff in eff_names}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for eff in eff_names:
            pts = np.array([data.geom_xpos[g] for g in sole_geoms[eff]])
            sole_z[eff][t] = pts[:, 2].min()

    flags = np.zeros((T, len(eff_names)), dtype=bool)
    for i, eff in enumerate(eff_names):
        flags[:, i] = sole_z[eff] < height_thresh
    return eff_names, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=Path,
                    help="Already Stage-A+grounded ('polished') pkl.")
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=G1_MODEL_DEFAULT)
    ap.add_argument("--height-thresh", type=float, default=0.05,
                    help="Coarse 'contact zone' gate for detect_g1_foot_contacts. "
                         "Calibrated empirically (planLogGMR.md E4): height-only at 0.05 "
                         "on walk1_subject1 gives ~48% frames near-ground, a plausible "
                         "stance-fraction ballpark. Real plant/moving split happens in "
                         "_compute_anchors via --plant-speed, unchanged.")
    ap.add_argument("--plant-speed", type=float, default=0.05,
                    help="Alex CLI default -- passed to _compute_anchors' OWN stillness "
                         "sub-segmentation (body-origin speed), unrelated to the coarse "
                         "height gate above.")
    ap.add_argument("--plant-min-run", type=int, default=2,
                    help="8 @120Hz (~66ms) linearly scaled -> 2 @30Hz, LAFAN1's rate "
                         "(same scaling convention as ihmc_json_to_stage4_npz.py's fps note).")
    ap.add_argument("--foot-weight", type=float, default=40.0, help="Alex CLI default.")
    ap.add_argument("--move-ratio", type=float, default=0.15, help="Alex CLI default.")
    ap.add_argument("--foot-flat-weight", type=float, default=3.0, help="Alex CLI default.")
    ap.add_argument("--contact-downweight", type=float, default=0.1, help="Alex CLI default.")
    ap.add_argument("--lambda-track", type=float, default=1.0)
    ap.add_argument("--lambda-smooth", type=float, default=20.0, help="30Hz-scaled, per T7.")
    ap.add_argument("--n-outer", type=int, default=6, help="Pipeline default (retargetingPipeline.sh).")
    ap.add_argument("--trust", type=float, default=0.15, help="Alex CLI default.")
    args = ap.parse_args()

    qpos, fps = load_gmr_pkl(args.in_path)
    model, data, floor_gid, _ = _load_model_with_floor(args.model)

    eff_names, flags = detect_g1_foot_contacts(qpos, model, data, fps, args.height_thresh)
    for i, eff in enumerate(eff_names):
        pct = flags[:, i].mean() * 100
        print(f"  {eff}: {int(flags[:, i].sum())}/{qpos.shape[0]} frames in contact ({pct:.1f}%)")

    resolved = _resolve_g1_feet(model, eff_names)
    tgt, wgt, planted = _compute_anchors(
        model, data, qpos, eff_names, flags, resolved, fps,
        args.plant_speed, args.foot_weight, 0.0, args.move_ratio, args.plant_min_run)
    for eff in resolved:
        n = int((~np.isnan(tgt[eff][:, 0])).sum())
        npl = int(planted[eff].sum())
        print(f"  {eff}: contact {n}/{qpos.shape[0]} ({n/qpos.shape[0]*100:.1f}%), "
              f"planted {npl} ({npl/max(n,1)*100:.0f}% of contact)")

    role_to_body = {r: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                    for r, b in G1_ROLE_BODY.items()}
    T = qpos.shape[0]
    target_positions = np.zeros((T, len(G1_TRACK_ROLES), 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for ri, role in enumerate(G1_TRACK_ROLES):
            target_positions[t, ri] = data.xpos[role_to_body[role]]
    target_weights = {r: 1.0 for r in G1_TRACK_ROLES}

    downweight_roles = [set() for _ in range(T)]
    for eff in resolved:
        col = flags[:, eff_names.index(eff)]
        for t in np.where(col)[0]:
            downweight_roles[t].add(eff)  # role name == effector name here

    q_lo, q_hi = _get_joint_limits(model)
    assert qpos.shape[1] - 7 == N_ACT, f"expected {N_ACT} actuated joints, got {qpos.shape[1]-7}"

    qpos_out = stage_b(
        qpos, target_positions, G1_TRACK_ROLES, role_to_body, target_weights,
        tgt, wgt, planted, resolved, downweight_roles,
        model, data, q_lo, q_hi,
        lambda_track=args.lambda_track, lambda_smooth=args.lambda_smooth, lambda_coll=0.0,
        foot_flat_w=args.foot_flat_weight, fist_w=0.0,
        downweight_factor=args.contact_downweight, n_outer=args.n_outer, trust=args.trust,
        collision_penalty=1000.0, floor_z=None, floor_w=0.0,
        floor_gid=floor_gid, count_floor=False)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gmr_pkl(args.out_path, qpos_out, fps)
    print(f"Wrote {args.out_path} ({qpos_out.shape[0]} frames)")


if __name__ == "__main__":
    main()

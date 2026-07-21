#!/usr/bin/env python3
"""S7-T2: smoothing pass on top of `gmr_contact_fc` -- S7-T1a/T1b found
`gmr_contact_fc`'s discrete per-frame floor clamp (no ramp) carries a real jerk
cost (loco dev clips +40-78% body_jerk vs gmr_raw on 2/3 clips, tripping the
plan's >50% activation threshold; floor class +76.0% at full corpus scale, plus
velocity spikes on 22/34 floor clips gmr_raw never has at all).

Mechanism: Stage-A tridiagonal smoothing (the SAME function `polish_gmr_pkl.py`
uses for `gmr_polished` -- imported unchanged, `polish_gmr_pkl.py` itself is
NOT modified, per baseline-integrity rule) applied to `gmr_contact_fc` output,
THEN one full-clip re-clamp pass (same `leg_floor_clamp.clamp_limb` over
`CLAMP_TARGETS`, same call pattern as `gmr_contact_retarget.py --floor-clamp`'s
own inline block) because smoothing WILL reintroduce small penetrations at the
frames the clamp had corrected. Order matters: smooth-then-clamp, never
clamp-then-smooth-and-stop (a trailing smooth pass would just re-break the
clamp's exact floor contact with no correction after it).

Usage:
    conda run -n gmr python scripts/g1/smooth_then_clamp.py \\
        --in outputs/gmr_baseline/sprint/pkl_s5/walk1_subject1_gmrcontact_fc.pkl \\
        --out outputs/gmr_baseline/sprint/pkl_s5/walk1_subject1_gmrcontact_fc_sm.pkl
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from leg_floor_clamp import build_chain_dofs, clamp_limb, CLAMP_TARGETS  # noqa: E402
from gmr_contact_retarget import FEET, HANDS  # noqa: E402
from solve_global_trajectory_opt_contactfirst import stage_a, N_ACT  # noqa: E402

G1_MODEL_DEFAULT = pathlib.Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


def _joint_limits(model_path, n_act):
    """Verbatim copy of polish_gmr_pkl.py's --stage-a joint-limit extraction --
    kept in sync deliberately (same GMR mocap XML, same free-joint-skip logic),
    not imported from there to avoid coupling this driver to polish_gmr_pkl.py's
    CLI/argument surface (that module is the gmr_polished baseline generator,
    left untouched per baseline-integrity rule)."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    q_lo = np.full(n_act, -1e6)
    q_hi = np.full(n_act, 1e6)
    act = 0
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            continue
        if bool(model.jnt_limited[j]):
            q_lo[act] = model.jnt_range[j, 0]
            q_hi[act] = model.jnt_range[j, 1]
        act += 1
    return q_lo, q_hi


def smooth_then_clamp(qpos, lambda_track, lambda_smooth):
    n_act = qpos.shape[1] - 7
    assert n_act == N_ACT, f"stage_a hardcodes N_ACT={N_ACT}, got {n_act}"
    q_lo, q_hi = _joint_limits(G1_MODEL_DEFAULT, n_act)
    qpos = stage_a(qpos, lambda_track, lambda_smooth, q_lo, q_hi, smooth_root=True)

    vmodel, vdata, _, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(vmodel)
    clamp_chains = {eff: build_chain_dofs(vmodel, eff) for eff in FEET + HANDS}

    T = qpos.shape[0]
    for t in range(T):
        vdata.qpos[:] = qpos[t]
        for eff, watch_body in CLAMP_TARGETS:
            # S7-T7: always self-collision-aware here (not opt-in) -- this is a
            # freshly introduced S7 variant, not a byte-identical-with-prior
            # shipped baseline, and Stage-A smoothing can itself perturb
            # self-collision, so the re-clamp pass should always check.
            clamp_limb(vmodel, vdata, mesh_cache, eff, clamp_chains[eff],
                       floor_margin=0.0, watch_body=watch_body,
                       avoid_self_collision=True)
        qpos[t] = vdata.qpos.copy()
    return qpos


def save_gmr_pkl(path, qpos, fps):
    root_pos = qpos[:, 0:3]
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    dof_pos = qpos[:, 7:]
    motion_data = {"fps": fps, "root_pos": root_pos, "root_rot": root_rot_xyzw,
                   "dof_pos": dof_pos, "local_body_pos": None, "link_body_list": None}
    with open(path, "wb") as f:
        pickle.dump(motion_data, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=pathlib.Path)
    ap.add_argument("--out", dest="out_path", required=True, type=pathlib.Path)
    ap.add_argument("--lambda-track", type=float, default=1.0)
    ap.add_argument("--lambda-smooth", type=float, default=20.0)
    args = ap.parse_args()

    qpos, fps = load_gmr_pkl(args.in_path)
    qpos = smooth_then_clamp(qpos, args.lambda_track, args.lambda_smooth)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gmr_pkl(args.out_path, qpos, fps)
    print(f"Wrote {args.out_path} ({qpos.shape[0]} frames)")


if __name__ == "__main__":
    main()

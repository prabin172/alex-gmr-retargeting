#!/usr/bin/env python3
"""IHMC KinematicsToolboxOutputStatus JSON -> Stage-4-ready NPZ.

Purpose (2026-07-14): use an externally-authored robot motion — the mentor's
manual Blender retargets in data/blender-retargeted/ — as BOTH the warm start
and the tracking reference for our Stage-4 optimizer ("polish Luigi's motion":
Stage A removes manual-keyframing velocity roughness, Stage B enforces
contact/floor/self-collision while deviating minimally from HIS motion, not the
raw human's). This is the "right basin" idea from the 2026-07-13 Slack thread:
his edit supplies the feasible strategy our tracking cost can't discover;
Stage 4 supplies the precision manual keyframing can't.

Mapping into the Stage-4 input contract (what
solve_global_trajectory_opt_contactfirst.py reads from --ik-npz):
  qpos              — his motion, converted to MuJoCo convention
                      (joint reorder inverted, root quat xyzw->wxyz)
  target_positions  — FK of his OWN qpos at the 16 canonical role bodies
                      (self-referential: tracking cost pulls toward him)
  contact_flags     — his hand-authored foot flags (feet "locked in place" in
                      Blender); hands as present in the JSON (typically absent
                      -> False)
  metadata_json     — Stage-3's TARGET_WEIGHTS so role weighting matches the
                      pipeline's (pelvis 4.0, ankles 1.5, ...)
  fps/source_frame_ids — from the JSON timestamps (typically 50 Hz)

Rate note for the Stage-4 call itself (NOT this script): lambda_smooth scales
as fps^2 (pipeline's 320 = 20 @30Hz x16 @120Hz) -> ~56 at 50 Hz; plant_min_run
8 @120Hz (~66ms) -> 3 at 50 Hz.

Usage:
    conda run -n gmr python scripts/ihmc_json_to_stage4_npz.py \\
        --json data/blender-retargeted/standSupine.json \\
        --out outputs/cont_dev/blender_standSupine_stage4in.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ihmc_json import _load  # noqa: E402  (JSON -> qpos/fps/contacts)
from export_alex_retarget_npz_to_ihmc_json import load_mujoco_joint_order  # noqa: E402
from post_process_ground_contactfirst import MODEL_DEFAULT  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import (  # noqa: E402
    ROLE_TO_ALEX_BODY, TARGET_WEIGHTS)

EFFECTORS = ["left_foot", "right_foot", "left_hand", "right_hand"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mj_joint_names = load_mujoco_joint_order(args.model)

    qpos, fps, contacts = _load(args.json, mj_joint_names)
    T = qpos.shape[0]

    role_names = list(ROLE_TO_ALEX_BODY.keys())
    body_names = [ROLE_TO_ALEX_BODY[r] for r in role_names]
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_names]
    missing = [b for b, i in zip(body_names, bids) if i < 0]
    if missing:
        sys.exit(f"bodies not in model: {missing}")

    target_positions = np.zeros((T, len(role_names), 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for ri, bid in enumerate(bids):
            target_positions[t, ri] = data.xpos[bid]

    contact_flags = np.zeros((T, len(EFFECTORS)), dtype=bool)
    for ci, eff in enumerate(EFFECTORS):
        if eff in contacts:
            contact_flags[:, ci] = contacts[eff]

    meta = {"target_weights": {r: float(TARGET_WEIGHTS.get(r, 1.0)) for r in role_names}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        qpos=qpos,
        target_positions=target_positions,
        role_names=np.asarray(role_names, dtype=object),
        alex_body_names=np.asarray(body_names, dtype=object),
        contact_effector_names=np.asarray(EFFECTORS, dtype=object),
        contact_flags=contact_flags,
        metadata_json=json.dumps(meta),
        fps=np.float64(fps),
        source_frame_ids=np.arange(T),
    )
    nfoot = int(contact_flags[:, :2].sum())
    print(f"[stage4in] {args.json.name}  T={T} fps={fps:g}  foot-contact frames={nfoot}  -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-frame robot-vs-floor penetration report for a solved NPZ (Stage 3/4/4.5).

Injects the same in-memory floor plane the solvers use (mesh-accurate, via
mj_forward's narrow phase — catches a tilted toe that named sites miss) and
reports the worst penetrating frames + which body.

IMPORTANT — fixed floor reference for before/after comparisons: by default the
floor height is re-estimated from THIS npz's planted feet, which moves when the
fix under test lifts the plants (a run that lifts its feet 7 cm re-registers the
floor 7 cm higher and inflates every other body's "penetration" by 7 cm — this
artifact produced a phantom 35.8 cm regression during the 2026-07-09 Stage-3
experiment, see collision.md). When comparing two runs, measure BOTH against the
baseline's floor via --floor-z.

Usage:
    conda run -n gmr python scripts/diagnose_floor_penetration.py \
        --npz outputs/contactfirst/luigi_standProne_03_contactfirst.npz
    # fixed-reference comparison:
    ... --npz <candidate.npz> --floor-z -0.1152
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_global_trajectory_opt_contactfirst import (  # noqa: E402
    _load_model_with_floor, _estimate_floor_z, _resolve_contact_geom,
    _compute_anchors, MODEL_DEFAULT,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_DEFAULT, type=Path)
    ap.add_argument("--floor-z", type=float, default=None,
                    help="Fixed floor height (Alex frame). Default: re-estimate from "
                         "this npz's planted feet — do NOT use the default when "
                         "comparing runs (see module docstring).")
    ap.add_argument("--qpos-key", default="qpos",
                    help="Which qpos array to evaluate (qpos, qpos_stage_a, ...).")
    ap.add_argument("--top", type=int, default=15, help="How many worst frames to list.")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z[args.qpos_key], dtype=np.float64)
    model, data, floor_gid, floor_mocap_id = _load_model_with_floor(args.model)

    floor_z = args.floor_z
    if floor_z is None:
        fps = float(z["fps"]) if "fps" in z.files else 120.0
        eff_names = [str(x) for x in z["contact_effector_names"]]
        flags = np.asarray(z["contact_flags"], dtype=bool)
        meta = json.loads(z["metadata_json"].item()) if "metadata_json" in z.files else {}
        resolved = _resolve_contact_geom(model, eff_names, meta.get("contact_pos_sites", {}))
        _, _, planted = _compute_anchors(model, data, qpos, eff_names, flags, resolved,
                                         fps, 0.05, 160, 32, 0.15, 8)
        floor_z = _estimate_floor_z(model, data, qpos, planted, resolved)
        print(f"floor_z (estimated from this npz's planted feet): {floor_z:+.4f}")
    else:
        print(f"floor_z (fixed, caller-supplied): {floor_z:+.4f}")
    data.mocap_pos[floor_mocap_id] = [0.0, 0.0, floor_z]

    T = qpos.shape[0]
    worst = []          # (frame, pen_m, body)
    per_body_max = {}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        mx, mxbody = 0.0, None
        for c in range(data.ncon):
            ct = data.contact[c]
            if ct.geom1 != floor_gid and ct.geom2 != floor_gid:
                continue
            if ct.dist < 0 and abs(ct.dist) > mx:
                mx = abs(float(ct.dist))
                other = ct.geom2 if ct.geom1 == floor_gid else ct.geom1
                mxbody = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                           int(model.geom_bodyid[other]))
        worst.append((t, mx, mxbody))
        if mxbody is not None:
            per_body_max[mxbody] = max(per_body_max.get(mxbody, 0.0), mx)

    arr = np.array([w[1] for w in worst])
    print(f"\noverall: max={arr.max()*100:.1f}cm  mean-of-frame-max={arr.mean()*100:.2f}cm  "
          f"frames penetrating >1cm: {100*(arr > 0.01).mean():.1f}%")
    print(f"\nworst {args.top} frames:")
    for t, pen, body in sorted(worst, key=lambda w: -w[1])[:args.top]:
        print(f"  t={t:4d}  pen={pen*100:5.1f}cm  {body}")
    print("\nper-body max penetration:")
    for body, pen in sorted(per_body_max.items(), key=lambda kv: -kv[1]):
        print(f"  {body:24s} {pen*100:5.1f}cm")


if __name__ == "__main__":
    main()

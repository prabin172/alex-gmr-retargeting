#!/usr/bin/env python3
"""Scratch dev tool for the continuation-v1 plan (plan.md) — NOT part of the
shipped pipeline. Loads a Stage-4 output NPZ and prints floor-penetration,
self-collision, velocity-spike, slip, and tracking stats using the same
machinery `solve_global_trajectory_opt_contactfirst.py` uses internally, so
gate numbers are directly comparable to what Stage B itself reports.

Usage:
    conda run -n gmr python scripts/dev_cont_probe.py --npz outputs/cont_dev/foo.npz \\
        [--floor-z -0.0123] [--floor-phase-aware]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solve_global_trajectory_opt_contactfirst as m


def floor_pen_by_frame(model, data, qpos, floor_gid, floor_active_frames=None):
    """(T,) raw robot-vs-floor penetration depth (m, 0 = none/not touching).
    Unlike `_collision_stats` (COLL_MARGIN-relative, mixed with self-collision),
    this isolates floor-only, un-marginned penetration per frame — the quantity
    the continuation homotopy schedule (plan.md S3.1/S3.2) needs."""
    T = qpos.shape[0]
    pen = np.zeros(T)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        if floor_active_frames is not None and not floor_active_frames[t]:
            continue
        mx = 0.0
        for c in range(data.ncon):
            ct = data.contact[c]
            if not (ct.geom1 == floor_gid or ct.geom2 == floor_gid):
                continue
            if ct.dist < 0:
                mx = max(mx, -float(ct.dist))
        pen[t] = mx
    return pen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=m.MODEL_DEFAULT)
    ap.add_argument("--floor-z", type=float, default=None,
                    help="Override floor_z; default = re-estimate from the NPZ's own "
                         "contact/plant data (same as the solver would).")
    ap.add_argument("--floor-phase-aware", action="store_true")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=np.float64)
    T = qpos.shape[0]
    fps = float(z["fps"]) if "fps" in z.files else 120.0
    role_names = [str(r) for r in z["role_names"]]
    target_positions = np.asarray(z["target_positions"], dtype=np.float64)
    eff_names = [str(x) for x in z["contact_effector_names"]] if "contact_effector_names" in z.files else []
    flags = np.asarray(z["contact_flags"], dtype=bool) if "contact_flags" in z.files else np.zeros((T, 0), bool)
    meta = json.loads(z["metadata_json"].item()) if "metadata_json" in z.files else {}
    contact_sites = meta.get("contact_pos_sites", {})

    model, data, floor_gid, floor_mocap_id = m._load_model_with_floor(args.model)
    role_to_body = {}
    for ri, role in enumerate(role_names):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(z["alex_body_names"][ri]))
        if bid >= 0:
            role_to_body[role] = bid

    resolved = m._resolve_contact_geom(model, eff_names, contact_sites)
    tgt, wgt, planted = m._compute_anchors(
        model, data, qpos, eff_names, flags, resolved, fps,
        plant_speed=0.05, foot_w=160.0, hand_w=32.0, move_ratio=0.15, plant_min_run=8)

    floor_z = args.floor_z
    if floor_z is None:
        floor_z = m._estimate_floor_z(model, data, qpos, planted, resolved)
    print(f"floor_z used: {floor_z}")
    data.mocap_pos[floor_mocap_id] = [0.0, 0.0, floor_z if floor_z is not None else 0.0]

    floor_active_frames = None
    if args.floor_phase_aware:
        planted_any = np.zeros(T, dtype=bool)
        for eff, info in resolved.items():
            if info["kind"] == "foot":
                planted_any |= planted[eff]
        w = m.floor_phase_weight(qpos[:, 2], planted_any)
        floor_active_frames = w >= 0.5

    pen = floor_pen_by_frame(model, data, qpos, floor_gid, floor_active_frames)
    d = m._delta_stats(qpos)
    cs_self = m._collision_stats(model, data, qpos, floor_gid=floor_gid, count_floor=False)
    tr = m._tracking_stats(qpos, target_positions, role_to_body, role_names, model, data)
    ss = m._contact_slip_stats(model, data, qpos, tgt, wgt, planted, resolved)
    ferr = m._foot_floor_err_cm(model, data, qpos, planted, resolved, floor_z)

    print(f"{args.npz.name}")
    print(f"  floor pen (raw, isolated): max={pen.max()*100:.2f}cm  "
          f"frames>0.5cm={(pen>0.005).sum()}/{T} ({(pen>0.005).mean()*100:.1f}%)")
    print(f"  self-collision: pct={cs_self['pct']:.1f}%  peak={cs_self['max_pen_cm']:.2f}cm")
    print(f"  spikes(>0.5rad): {d['n_spikes_05']}  max_dq={d['max']:.3f}")
    print(f"  plant_slip_max={ss['plant_slip_max_cm']:.2f}cm  flat_mean={ss['flat_mean_deg']:.2f}deg  "
          f"foot_floor_err={ferr:.2f}cm")
    print(f"  tracking mean={tr['mean']:.4f}m max={tr['max']:.4f}m")


if __name__ == "__main__":
    main()

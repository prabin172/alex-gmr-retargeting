#!/usr/bin/env python3
"""W2-T2: independent cross-check of E4's walk1_subject1 slip-reduction claim.

E4 (planLogGMR.md "E4") reported a 25% plant-slip reduction (1.2->0.9cm) on
walk1_subject1 using `_contact_slip_stats` -- a function INSIDE the imported
`solve_global_trajectory_opt_contactfirst.py`, measuring drift against `tgt`,
a target computed by the SAME layer's `_compute_anchors` from the SAME
detected contact flags. Not circular in the strict sense (the target is fixed
once from the warm-start data, not re-derived from stage_b's own output), but
it shares every intermediate object with the mechanism being scored, so an
outside check on a genuinely separate code path is worth the half day
(GMR-baseline.md SS7.2 item 4).

This script:
  1. Re-derives contact-ZONE flags with `detect_g1_foot_contacts` (imported --
     deterministic, same as the E4 run; this is the one piece explicitly
     reused per the plan, since re-deriving contact zones from scratch would
     just reinvent the same height-gate logic).
  2. Does NOT import `_contact_point` / `_compute_anchors` / `_contact_slip_stats`
     -- computes each foot's BODY-ORIGIN xyz (data.xpos, not stage_b's own
     sole-corner contact point) via plain mujoco FK, a genuinely different
     measurement of "where is the foot".
  3. Segments zone frames into stillness sub-runs using body-origin XY speed
     < 0.05 m/s (Alex's plant_speed convention), debounced at a 2-frame
     minimum run length (matches stage_b_g1.py's own --plant-min-run default,
     applied identically to both warm/best so it can't bias the comparison).
  4. Per run: XY drift = max distance from the run's own first-frame position
     (not median -- avoids reusing _compute_anchors' exact convention).
  5. Reports per-foot mean/max drift on BOTH the polished (pre-StageB, "warm")
     and StageB ("best") motions, plus the direction of the effect.

Usage:
    conda run -n gmr python scripts/g1/check_slip_independent.py \\
        --warm outputs/gmr_baseline/pkl/walk1_subject1_polished_constant.pkl \\
        --best outputs/gmr_baseline/pkl/walk1_subject1_stageB.pkl
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
from stage_b_g1 import G1_CONTACT_GEOM, detect_g1_foot_contacts  # noqa: E402

G1_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


def _body_xy_trace(qpos, model, data, body_id):
    T = qpos.shape[0]
    xy = np.zeros((T, 2))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        xy[t] = data.xpos[body_id][:2]
    return xy


def _stillness_runs(xy, zone_flags, fps, plant_speed=0.05, min_run=2):
    """Within zone_flags==True frames, sub-segment by XY speed < plant_speed,
    debounced at min_run frames. Returns list of (start, end) inclusive index
    pairs, mirroring _compute_anchors' interval-then-substillness structure
    but independently coded (no shared helper)."""
    T = xy.shape[0]
    speed = np.zeros(T)
    speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    speed[0] = speed[1] if T > 1 else 0.0

    runs = []
    t = 0
    while t < T:
        if not zone_flags[t]:
            t += 1
            continue
        s = t
        while t < T and zone_flags[t]:
            t += 1
        e = t  # [s, e) is one contact-zone interval
        k = s
        while k < e:
            if speed[k] < plant_speed:
                j = k
                while j < e and speed[j] < plant_speed:
                    j += 1
                if j - k >= min_run:
                    runs.append((k, j))
                k = j
            else:
                k += 1
    return runs


def _drift_stats(xy, runs):
    drifts = []
    for s, e in runs:
        seg = xy[s:e]
        d = np.linalg.norm(seg - seg[0], axis=1).max()
        drifts.append(float(d))
    return drifts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", required=True, type=Path, help="Pre-StageB (polished) pkl.")
    ap.add_argument("--best", required=True, type=Path, help="Post-StageB pkl.")
    ap.add_argument("--model", type=Path, default=G1_MODEL_DEFAULT)
    ap.add_argument("--height-thresh", type=float, default=0.05)
    ap.add_argument("--plant-speed", type=float, default=0.05)
    ap.add_argument("--min-run", type=int, default=2)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    qpos_warm, fps = load_gmr_pkl(args.warm)
    qpos_best, fps2 = load_gmr_pkl(args.best)
    assert qpos_warm.shape == qpos_best.shape, "warm/best frame count or DOF mismatch"
    assert fps == fps2

    # Contact zones re-derived on the WARM motion only (best is a small local
    # perturbation of warm by construction -- stage_b never changes WHEN a foot
    # is near the ground, only where/how it sits there; using the same zone
    # windows for both keeps the comparison apples-to-apples).
    eff_names, flags = detect_g1_foot_contacts(qpos_warm, model, data, fps, args.height_thresh)

    for i, eff in enumerate(eff_names):
        body_name = G1_CONTACT_GEOM[eff]["body"]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        zone = flags[:, i]

        xy_warm = _body_xy_trace(qpos_warm, model, data, bid)
        xy_best = _body_xy_trace(qpos_best, model, data, bid)

        runs = _stillness_runs(xy_warm, zone, fps, args.plant_speed, args.min_run)
        drift_warm = _drift_stats(xy_warm, runs)
        drift_best = _drift_stats(xy_best, runs)  # SAME runs (from warm), applied to best's trace

        print(f"\n{eff} ({body_name}): {len(runs)} planted runs "
              f"(zone {int(zone.sum())}/{qpos_warm.shape[0]} frames = {zone.mean()*100:.1f}%)")
        if not runs:
            print("  no planted runs found -- nothing to compare")
            continue
        dw = np.array(drift_warm) * 100
        db = np.array(drift_best) * 100
        print(f"  drift (cm), independent body-origin XY, per-run max vs run-start:")
        print(f"    warm: mean={dw.mean():.2f} max={dw.max():.2f}")
        print(f"    best: mean={db.mean():.2f} max={db.max():.2f}")
        delta_mean = (dw.mean() - db.mean()) / dw.mean() * 100 if dw.mean() > 0 else float("nan")
        delta_max = (dw.max() - db.max()) / dw.max() * 100 if dw.max() > 0 else float("nan")
        direction = "REDUCED" if db.mean() < dw.mean() else "INCREASED or unchanged"
        print(f"  mean drift {direction}: {delta_mean:+.1f}% (max: {delta_max:+.1f}%)")


if __name__ == "__main__":
    main()

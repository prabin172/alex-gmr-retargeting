#!/usr/bin/env python3
"""Per-frame foot-slip vs inter-limb-penetration diagnostic (Stage 4).

Answers the tuning question before we touch FOOT_WEIGHT: on the worst clip, do
the high-slip frames COINCIDE with high self-penetration (legs crossing → a
kinematic floor we shouldn't chase) or NOT (pure soft-weight deficit → raising
FOOT_WEIGHT will help)?

Reuses the Stage-4 solver's own helpers so the frozen anchor (`_compute_anchors`)
and the penetration definition (COLL_MARGIN / COLL_HOPS filter) are IDENTICAL to
what the QP scores. The frozen anchor `tgt`/`planted` depend only on
plant-speed / plant-min-run / move-ratio — NOT on FOOT_WEIGHT — so this baseline
is weight-independent: it measures how far the SOLVED foot sits from the target
the QP was pinning it to.

Window-widened correlation (per the mimic-repo review): LAMBDA_SMOOTH couples
adjacent frames, so a penetration spike at t can smear slip into t±W. We test
overlap in a window, not exact-frame alignment, before concluding "no correlation".

Usage:
  python scripts/diagnose_foot_slip.py \
      --ik-npz outputs/contactfirst/standup_side_04_contactfirst.npz \
      --opt-npz outputs/global_opt_contactfirst/standup_side_04_global_opt.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_global_trajectory_opt_contactfirst import (  # noqa: E402
    COLL_HOPS, COLL_MARGIN, MODEL_DEFAULT,
    _compute_anchors, _contact_point, _resolve_contact_geom, _within_k_hops,
)


def _max_penetration_cm(model, data, qpos_t):
    """Max inter-limb penetration (cm) at one frame — same filter as the QP's
    collision rows and _collision_stats (skip world + within-COLL_HOPS pairs)."""
    data.qpos[:] = qpos_t
    mujoco.mj_forward(model, data)
    mx = 0.0
    for c in range(data.ncon):
        ct = data.contact[c]
        b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
        if b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS):
            continue
        if ct.dist < 0:
            mx = max(mx, abs(float(ct.dist)))
    return mx * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ik-npz", required=True, type=Path,
                    help="Stage-3 output (Stage-4 INPUT) — used to rebuild anchors.")
    ap.add_argument("--opt-npz", required=True, type=Path,
                    help="Stage-4 output — the SOLVED qpos to score.")
    ap.add_argument("--model", default=MODEL_DEFAULT, type=Path)
    ap.add_argument("--qpos-key", default="qpos",
                    help="Which solved trajectory to score (qpos=best, "
                         "qpos_stage_b, qpos_stage_a).")
    # anchor knobs — MUST match the pipeline (FOOT_WEIGHT is irrelevant to tgt/planted)
    ap.add_argument("--plant-speed", type=float, default=0.05)
    ap.add_argument("--foot-weight", type=float, default=160.0)
    ap.add_argument("--hand-weight", type=float, default=32.0)
    ap.add_argument("--move-ratio", type=float, default=0.15)
    ap.add_argument("--plant-min-run", type=int, default=8)
    ap.add_argument("--window", type=int, default=6,
                    help="Half-width (frames) for the widened slip↔penetration "
                         "correlation (LAMBDA_SMOOTH phase-shift tolerance).")
    ap.add_argument("--slip-hi-cm", type=float, default=3.0,
                    help="Threshold defining a 'high slip' frame for the overlap test.")
    ap.add_argument("--pen-hi-cm", type=float, default=0.5,
                    help="Threshold defining a 'high penetration' frame.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional per-frame CSV dump (frame,foot_slip_cm,pen_cm,"
                         "pen_win_cm,planted_foot).")
    args = ap.parse_args()

    z = np.load(args.ik_npz, allow_pickle=True)
    qpos_ik = np.asarray(z["qpos"], dtype=np.float64)
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    T = qpos_ik.shape[0]
    eff_names = [str(x) for x in z["contact_effector_names"]] if "contact_effector_names" in z.files else []
    flags = np.asarray(z["contact_flags"], dtype=bool) if "contact_flags" in z.files else np.zeros((T, 0), bool)
    meta = json.loads(z["metadata_json"].item()) if "metadata_json" in z.files else {}
    contact_sites = meta.get("contact_pos_sites", {})

    zo = np.load(args.opt_npz, allow_pickle=True)
    if args.qpos_key not in zo.files:
        sys.exit(f"key {args.qpos_key!r} not in {args.opt_npz} (have: {list(zo.files)})")
    qpos_opt = np.asarray(zo[args.qpos_key], dtype=np.float64)
    assert qpos_opt.shape[0] == T, f"frame mismatch ik={T} opt={qpos_opt.shape[0]}"

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    resolved = _resolve_contact_geom(model, eff_names, contact_sites)
    tgt, wgt, planted = _compute_anchors(
        model, data, qpos_ik, eff_names, flags, resolved, fps,
        args.plant_speed, args.foot_weight, args.hand_weight, args.move_ratio,
        args.plant_min_run)

    feet = [e for e, i in resolved.items() if i["kind"] == "foot"]
    print(f"Clip T={T} fps={fps:.0f}  feet={feet}")
    for e in feet:
        print(f"  {e:11s} planted frames: {int(planted[e].sum())}")

    # per-frame: max planted-foot XY slip from frozen anchor, and which foot
    slip = np.zeros(T)
    slip_foot = [""] * T
    for t in range(T):
        data.qpos[:] = qpos_opt[t]
        mujoco.mj_forward(model, data)
        for e in feet:
            if not planted[e][t] or np.isnan(tgt[e][t, 0]):
                continue
            d = float(np.linalg.norm((_contact_point(data, resolved[e]) - tgt[e][t])[:2]))
            if d > slip[t]:
                slip[t] = d; slip_foot[t] = e
    slip *= 100.0  # cm

    pen = np.array([_max_penetration_cm(model, data, qpos_opt[t]) for t in range(T)])

    # windowed penetration: max over [t-W, t+W] (phase-shift-tolerant)
    W = args.window
    pen_win = np.array([pen[max(0, t - W):min(T, t + W + 1)].max() for t in range(T)])

    planted_any = np.array([bool(slip_foot[t]) for t in range(T)])
    npl = int(planted_any.sum())
    print(f"\nPlanted-foot frames: {npl}/{T}")
    if npl == 0:
        sys.exit("no planted-foot frames — nothing to correlate")

    sp_ = slip[planted_any]
    print(f"Foot slip (planted frames):  max={sp_.max():.2f}cm  "
          f"p95={np.percentile(sp_, 95):.2f}cm  mean={sp_.mean():.2f}cm")
    print(f"Penetration (all frames):    max={pen.max():.2f}cm  "
          f"p95={np.percentile(pen, 95):.2f}cm  mean={pen.mean():.2f}cm")

    # correlation over planted frames (exact-frame and windowed)
    def _corr(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    print(f"\nPearson r (planted frames):")
    print(f"  slip vs pen  (exact-frame): {_corr(sp_, pen[planted_any]):+.3f}")
    print(f"  slip vs pen  (±{W} window):  {_corr(sp_, pen_win[planted_any]):+.3f}")

    # overlap test: of the high-slip frames, what fraction sit near a pen spike?
    hi = planted_any & (slip >= args.slip_hi_cm)
    n_hi = int(hi.sum())
    print(f"\nHigh-slip frames (≥{args.slip_hi_cm}cm, planted): {n_hi}")
    if n_hi:
        near_pen = int((pen_win[hi] >= args.pen_hi_cm).sum())
        print(f"  ...of which within ±{W} frames of pen ≥{args.pen_hi_cm}cm: "
              f"{near_pen}/{n_hi} ({near_pen/n_hi*100:.0f}%)")
        print(f"  their penetration (windowed): max={pen_win[hi].max():.2f}cm "
              f"mean={pen_win[hi].mean():.2f}cm")

    # worst frames table
    order = np.argsort(slip)[::-1][:12]
    print(f"\nWorst {len(order)} slip frames:")
    print(f"  {'frame':>6} {'slip_cm':>8} {'pen_cm':>7} {'pen_win':>8} {'foot':>12}")
    for t in sorted(order, key=lambda x: -slip[x]):
        print(f"  {t:6d} {slip[t]:8.2f} {pen[t]:7.2f} {pen_win[t]:8.2f} "
              f"{slip_foot[t] or '-':>12}")

    # verdict
    print("\n" + "=" * 60)
    if n_hi == 0:
        print("VERDICT: no high-slip frames at this threshold — slip already low.")
    else:
        frac = (pen_win[hi] >= args.pen_hi_cm).mean()
        if frac >= 0.5:
            print("VERDICT: COLLISION-BOUND — most high-slip frames coincide with "
                  "self-penetration.\n  Much of the slip is a kinematic floor "
                  "(legs crossing). Raising FOOT_WEIGHT will fight the collision "
                  "slack; expect limited headroom. Document as a contact-geometry limit.")
        else:
            print("VERDICT: WEIGHT-BOUND — high-slip frames are mostly collision-FREE.\n"
                  "  The soft pin (FOOT_WEIGHT) is just too weak here. Raise it "
                  "(2000→8000), re-run this clip, stop when slip plateaus or pen>2cm.")
    print("=" * 60)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w") as f:
            f.write("frame,foot_slip_cm,pen_cm,pen_win_cm,planted_foot\n")
            for t in range(T):
                f.write(f"{t},{slip[t]:.4f},{pen[t]:.4f},{pen_win[t]:.4f},"
                        f"{slip_foot[t]}\n")
        print(f"\nWrote per-frame CSV: {args.csv}")


if __name__ == "__main__":
    main()

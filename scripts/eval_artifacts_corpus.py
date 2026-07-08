#!/usr/bin/env python3
"""Corpus-wide artifact evaluation for the contact-first pipeline.

One honest table across all clips of the residual artifacts a downstream RL
mimic tracker inherits:

  * joint limits   — worst actuated-joint range usage, frames at a bound (±1°),
                     hard violations (angle outside the model/URDF range).
  * contact slip   — max drift of a PLANTED contact off its frozen anchor
                     (feet horizontal, hands 3D), + how long it slides
                     (seconds with slip > --slip-thresh-cm).
  * self-collision — % of frames with any inter-limb penetration, peak depth.

Reuses the Stage-4 solver's own helpers (`_compute_anchors`, `_contact_point`,
the COLL_HOPS/COLL_MARGIN penetration filter) so every number matches what the
QP actually scored — no re-derived definitions.

Metrics are read off the Stage-4 output (`global_opt_contactfirst/`). Grounding
(Stage 4.5) is a rigid Z shift, so joint angles, self-collision geometry and
HORIZONTAL slip are identical in the grounded export — evaluating pre-grounding
is equivalent and lets us pair each clip with its Stage-3 anchors.

Usage:
  python scripts/eval_artifacts_corpus.py                    # all clips
  python scripts/eval_artifacts_corpus.py --csv outputs/artifact_table.csv
  python scripts/eval_artifacts_corpus.py --match standup
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solve_global_trajectory_opt_contactfirst import (  # noqa: E402
    COLL_HOPS, MODEL_DEFAULT,
    _compute_anchors, _contact_point, _resolve_contact_geom, _within_k_hops,
)

GO_DIR = Path("outputs/global_opt_contactfirst")
CF_DIR = Path("outputs/contactfirst")
GR_DIR = Path("outputs/grounded_contactfirst")


def _joint_limit_stats(model, qpos, tol_rad):
    """Per actuated joint: range used, frames at a bound, hard violations.
    Also names the single most-saturated joint (highest at-limit frame frac)."""
    names, qadrs, los, his, lim = [], [], [], [], []
    for j in range(model.njnt):
        qadr = int(model.jnt_qposadr[j])
        if qadr < 7:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        qadrs.append(qadr)
        limited = bool(model.jnt_limited[j])
        lim.append(limited)
        los.append(float(model.jnt_range[j][0]) if limited else -math.inf)
        his.append(float(model.jnt_range[j][1]) if limited else math.inf)
    qadrs = np.array(qadrs); los = np.array(los); his = np.array(his)
    lim = np.array(lim)
    vals = qpos[:, qadrs]                                  # (T, nJ)
    rng = np.where(lim, his - los, np.inf)
    used = np.where(lim & (rng > 0),
                    (vals.max(0) - vals.min(0)) / np.where(rng > 0, rng, 1), 0.0)
    at = lim & ((vals <= los + tol_rad) | (vals >= his - tol_rad))  # (T,nJ)
    viol = lim & ((vals < los) | (vals > his))
    at_frac = at.mean(axis=0)                              # per-joint fraction of frames at a bound
    w = int(np.argmax(at_frac))
    return {
        "max_range_pct": float(used.max() * 100),
        "atlim_frame_pct": float((at.any(axis=1)).mean() * 100),  # frames touching ANY bound
        "worst_joint": names[w] if at_frac[w] > 0 else "-",
        "worst_joint_atlim_pct": float(at_frac[w] * 100),
        "worst_joint_range_pct": float(used[w] * 100),
        "n_viol_joints": int(viol.any(axis=0).sum()),
        "worst_viol_deg": float(np.degrees(np.maximum(
            (los[None, :] - vals) * viol, (vals - his[None, :]) * viol).max())) if viol.any() else 0.0,
    }


def _pen_series(model, data, qpos):
    """Per-frame max inter-limb penetration (m), solver's COLL_HOPS filter."""
    out = np.zeros(qpos.shape[0])
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        mx = 0.0
        for c in range(data.ncon):
            ct = data.contact[c]
            b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
            if b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS):
                continue
            if ct.dist < 0:
                mx = max(mx, abs(float(ct.dist)))
        out[t] = mx
    return out


def _ground_series(model, data, qpos_grounded, planted, resolved, tol_m):
    """Foot vs the floor (z=0) on the GROUNDED qpos, split by contact state:
      * PLANTED penetration — a foot LABELED in contact whose lowest sole corner
        goes below 0. A support foot through the floor = a hard physics violation
        (the RL-critical one).
      * ANY penetration — lowest sole corner below 0 over ALL frames incl. swing.
        A lifted/tucked foot clipping the floor plane during a deep crouch is
        infeasible too, but a softer class than a bad plant.
      * planted-foot float — lowest sole corner hovering ABOVE 0 while planted
        (a hovering plant = a phantom target for the tracker)."""
    feet = [e for e, i in resolved.items() if i["kind"] == "foot" and i["sole_sites"]]
    T = qpos_grounded.shape[0]
    min_z_any = np.full(T, np.inf)          # lowest sole corner across both feet, per frame
    plant_low = []                          # signed lowest-corner height on planted-foot frames
    for t in range(T):
        data.qpos[:] = qpos_grounded[t]
        mujoco.mj_forward(model, data)
        for e in feet:
            zc = min(float(data.site_xpos[s][2]) for s in resolved[e]["sole_sites"])
            min_z_any[t] = min(min_z_any[t], zc)
            if planted[e][t]:
                plant_low.append(zc)
    plant_low = np.array(plant_low) if plant_low else np.zeros(0)
    plant_pen = np.clip(-plant_low, 0, None)          # depth below floor while planted
    return {
        "grnd_pen_plant_cm": float(plant_pen.max() * 100) if plant_low.size else 0.0,
        "grnd_pen_plant_pct": float((plant_low < -tol_m).mean() * 100) if plant_low.size else 0.0,
        "grnd_pen_any_cm": float(max(0.0, -min_z_any.min()) * 100),
        "grnd_pen_any_pct": float((min_z_any < -tol_m).mean() * 100),
        "foot_float_max_cm": float(np.clip(plant_low, 0, None).max() * 100) if plant_low.size else 0.0,
        "foot_float_mean_cm": float(np.clip(plant_low, 0, None).mean() * 100) if plant_low.size else 0.0,
    }


def _slip_series(model, data, qpos, tgt, planted, resolved):
    """Per-frame max PLANTED-contact drift off frozen anchor (m), split foot/hand.
    Feet horizontal, hands 3D — matches _contact_slip_stats. Returns dict of
    per-kind slip[T] + planted-mask[T]."""
    T = qpos.shape[0]
    out = {k: {"slip": np.zeros(T), "planted": np.zeros(T, bool)} for k in ("foot", "hand")}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for eff, info in resolved.items():
            if not planted[eff][t] or np.isnan(tgt[eff][t, 0]):
                continue
            kind = info["kind"]
            d = _contact_point(data, info) - tgt[eff][t]
            disp = float(np.linalg.norm(d[:2] if kind == "foot" else d))
            out[kind]["slip"][t] = max(out[kind]["slip"][t], disp)
            out[kind]["planted"][t] = True
    return out


def eval_clip(go_path, cf_dir, gr_dir, model, data, args):
    clip = go_path.name.replace("_global_opt.npz", "")
    cf_path = cf_dir / f"{clip}_contactfirst.npz"
    gr_path = gr_dir / f"{clip}_grounded.npz"
    if not cf_path.exists():
        return None, f"no anchor NPZ {cf_path.name}"

    zo = np.load(go_path, allow_pickle=True)
    qpos = np.asarray(zo["qpos"], dtype=np.float64)       # best (=stage_b when on)
    fps = float(zo["fps"]) if "fps" in zo.files else 30.0
    T = qpos.shape[0]

    zc = np.load(cf_path, allow_pickle=True)
    qpos_ik = np.asarray(zc["qpos"], dtype=np.float64)
    eff_names = [str(x) for x in zc["contact_effector_names"]] if "contact_effector_names" in zc.files else []
    flags = np.asarray(zc["contact_flags"], dtype=bool) if "contact_flags" in zc.files else np.zeros((T, 0), bool)
    meta = json.loads(zc["metadata_json"].item()) if "metadata_json" in zc.files else {}
    resolved = _resolve_contact_geom(model, eff_names, meta.get("contact_pos_sites", {}))

    tgt, wgt, planted = _compute_anchors(
        model, data, qpos_ik, eff_names, flags, resolved, fps,
        args.plant_speed, args.foot_weight, args.hand_weight,
        args.move_ratio, args.plant_min_run)

    jl = _joint_limit_stats(model, qpos, math.radians(args.limit_tol_deg))
    pen = _pen_series(model, data, qpos)
    ss = _slip_series(model, data, qpos, tgt, planted, resolved)

    if gr_path.exists():
        qpos_gr = np.asarray(np.load(gr_path, allow_pickle=True)["qpos"], dtype=np.float64)
        grnd = _ground_series(model, data, qpos_gr, planted, resolved, args.floor_tol_cm / 100.0)
    else:
        grnd = {"grnd_pen_plant_cm": float("nan"), "grnd_pen_plant_pct": float("nan"),
                "grnd_pen_any_cm": float("nan"), "grnd_pen_any_pct": float("nan"),
                "foot_float_max_cm": float("nan"), "foot_float_mean_cm": float("nan")}

    thr = args.slip_thresh_cm / 100.0

    def _slip_cols(kind):
        s = ss[kind]["slip"]; m = ss[kind]["planted"]
        sp = s[m]; n = int(m.sum())
        return {
            f"slip_{kind}_max_cm": float(sp.max() * 100) if n else 0.0,
            f"slip_{kind}_p95_cm": float(np.percentile(sp, 95) * 100) if n else 0.0,
            f"slip_{kind}_time_s": float((s > thr).sum()) / fps,   # frames sliding >thresh
            f"contact_{kind}_pct": n / T * 100,
        }

    row = {"clip": clip, "frames": T, "dur_s": T / fps,
           "jl_viol": jl["n_viol_joints"], "jl_viol_deg": jl["worst_viol_deg"],
           "jl_max_range_pct": jl["max_range_pct"], "jl_atlim_pct": jl["atlim_frame_pct"],
           "jl_worst_joint": jl["worst_joint"],
           "jl_worst_joint_atlim_pct": jl["worst_joint_atlim_pct"],
           "jl_worst_joint_range_pct": jl["worst_joint_range_pct"]}
    row.update(_slip_cols("foot"))
    row.update(_slip_cols("hand"))
    row.update(grnd)
    row.update({"coll_pct": float((pen > 0).mean() * 100),
                "peak_pen_cm": float(pen.max() * 100)})
    return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go-dir", type=Path, default=GO_DIR)
    ap.add_argument("--cf-dir", type=Path, default=CF_DIR)
    ap.add_argument("--gr-dir", type=Path, default=GR_DIR,
                    help="grounded NPZ dir (floor penetration / foot-float metrics).")
    ap.add_argument("--floor-tol-cm", type=float, default=0.5,
                    help="sole below −this counts as floor penetration.")
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--match", default="", help="substring filter on clip name")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--slip-thresh-cm", type=float, default=1.0,
                    help="slip above this (cm) counts toward slip-time.")
    ap.add_argument("--limit-tol-deg", type=float, default=1.0,
                    help="within this of a bound = 'at limit'.")
    # anchor knobs — MUST match the pipeline
    ap.add_argument("--plant-speed", type=float, default=0.05)
    ap.add_argument("--foot-weight", type=float, default=160.0)
    ap.add_argument("--hand-weight", type=float, default=32.0)
    ap.add_argument("--move-ratio", type=float, default=0.15)
    ap.add_argument("--plant-min-run", type=int, default=8)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    files = sorted(p for p in args.go_dir.glob("*_global_opt.npz") if args.match in p.name)
    if not files:
        sys.exit(f"no *_global_opt.npz in {args.go_dir} matching {args.match!r}")

    rows = []
    for p in files:
        r, err = eval_clip(p, args.cf_dir, args.gr_dir, model, data, args)
        if err:
            print(f"  [skip] {p.name}: {err}", file=sys.stderr); continue
        rows.append(r)
        print(f"  done {r['clip']:28s} plantPen={r['grnd_pen_plant_cm']:.1f} "
              f"anyPen={r['grnd_pen_any_cm']:.1f} float={r['foot_float_mean_cm']:.1f}",
              file=sys.stderr)

    # table
    hdr = (f"{'clip':<26}{'JLvi':>5}{'worst_joint(@lim%)':>24}"
           f"{'ftSlip':>7}{'hdSlip':>7}"
           f"{'plPen':>6}{'plPen%':>7}{'anyPen':>7}{'anyPen%':>8}{'flAvg':>6}"
           f"{'coll%':>7}{'selfPen':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))

    def _f(v, w, p=1):
        return f"{'--':>{w}}" if (isinstance(v, float) and math.isnan(v)) else f"{v:>{w}.{p}f}"
    for r in rows:
        wj = f"{r['jl_worst_joint']}({r['jl_worst_joint_atlim_pct']:.0f})"
        print(f"{r['clip']:<26}{r['jl_viol']:>5}{wj:>24}"
              f"{r['slip_foot_max_cm']:>7.1f}{r['slip_hand_max_cm']:>7.1f}"
              f"{_f(r['grnd_pen_plant_cm'],6)}{_f(r['grnd_pen_plant_pct'],7)}"
              f"{_f(r['grnd_pen_any_cm'],7)}{_f(r['grnd_pen_any_pct'],8)}"
              f"{_f(r['foot_float_mean_cm'],6)}"
              f"{r['coll_pct']:>7.1f}{r['peak_pen_cm']:>8.1f}")
    print("-" * len(hdr))

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)
    for stat, fn in [("median", np.nanmedian), ("max", np.nanmax)]:
        print(f"{'CORPUS '+stat:<26}{int(fn(col('jl_viol'))):>5}{'':>24}"
              f"{fn(col('slip_foot_max_cm')):>7.1f}{fn(col('slip_hand_max_cm')):>7.1f}"
              f"{fn(col('grnd_pen_plant_cm')):>6.1f}{fn(col('grnd_pen_plant_pct')):>7.1f}"
              f"{fn(col('grnd_pen_any_cm')):>7.1f}{fn(col('grnd_pen_any_pct')):>8.1f}"
              f"{fn(col('foot_float_mean_cm')):>6.1f}"
              f"{fn(col('coll_pct')):>7.1f}{fn(col('peak_pen_cm')):>8.1f}")
    print(f"\n{len(rows)} clips.  ftSlip/hdSlip = max foot(horiz)/hand(3D) planted slip cm.  "
          f"plPen = deepest PLANTED-foot sole below floor (hard physics violation); "
          f"anyPen = deepest sole below floor incl. swing/tucked feet (softer). "
          f"%>{ args.floor_tol_cm:.1f}cm under.  flAvg = planted-foot hover above floor.  "
          f"selfPen = peak inter-limb penetration.  JLvi = hard joint violations.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        keys = list(rows[0].keys())
        with open(args.csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                                 for k in keys) + "\n")
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()

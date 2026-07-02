#!/usr/bin/env python3
"""Aggregate contact-slip + inter-frame-jump metrics over GlobalOPT NPZs.

For every outputs/global_opt_contactfirst/*_global_opt.npz:
  * Inter-frame jump  — per frame, max over actuated joints of |Δq| (rad).
      Reported for per-frame IK vs Stage-A (the shipped, smoothed trajectory):
      mean, p95, max, and spike count (>0.5 rad).
  * Contact slip (position) — while an effector is in contact, the per-frame
      displacement of its contact point (foot body / palm site), cm/frame.
      Split foot vs hand: mean and max.
  * Foot heading drift — during contact, |achieved − target| foot heading (yaw),
      mean; the free-yaw "inner/outer rotation slip" we constrain.

Usage:
    python scripts/analyze_contact_slip_jumps.py
"""
from __future__ import annotations
import glob
import os
import numpy as np
import mujoco

MODEL = "assets/alex/alex_floating_base_with_sites.xml"
GO_DIR = "outputs/global_opt_contactfirst"

GEOM = {
    "left_foot":  ("body", "LEFT_FOOT"),
    "right_foot": ("body", "RIGHT_FOOT"),
    "left_hand":  ("site", "alex_left_palm_contact_site"),
    "right_hand": ("site", "alex_right_palm_contact_site"),
}


def heading_deg(R):
    fwd = R @ np.array([1.0, 0.0, 0.0])
    return np.degrees(np.arctan2(fwd[1], fwd[0]))


def jump_stats(qpos):
    dq = np.abs(np.diff(qpos[:, 7:], axis=0))
    mpf = dq.max(axis=1)
    return dict(mean=float(mpf.mean()), p95=float(np.percentile(mpf, 95)),
               max=float(mpf.max()), spikes=int((mpf > 0.5).sum()))


def contact_point(model, data, kind, name):
    if kind == "site":
        return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    data = mujoco.MjData(model)
    files = sorted(glob.glob(os.path.join(GO_DIR, "*_global_opt.npz")))
    if not files:
        print("No NPZs in", GO_DIR); return

    hdr = (f"{'clip':26s} {'jump_pf':>8} {'jump_A':>8} {'spk_pf':>6} {'spk_A':>6} "
           f"{'foot_slip':>10} {'hand_slip':>10} {'foot_yaw_err':>12}")
    print(hdr); print("-" * len(hdr))
    agg = {k: [] for k in ("jump_A", "spk_A", "foot_slip_mean", "hand_slip_mean",
                           "foot_slip_max", "hand_slip_max", "foot_yaw")}

    for f in files:
        z = np.load(f, allow_pickle=True)
        qA = np.asarray(z["qpos"], float)
        qpf = np.asarray(z["qpos_per_frame"], float) if "qpos_per_frame" in z.files else qA
        flags = z["contact_flags"].astype(bool)
        eff = [str(x) for x in z["contact_effector_names"]]
        T = qA.shape[0]

        jpf, jA = jump_stats(qpf), jump_stats(qA)

        # FK Stage-A to get contact-point tracks
        pts = {e: np.full((T, 3), np.nan) for e in eff if e in GEOM}
        for t in range(T):
            data.qpos[:] = qA[t]; mujoco.mj_forward(model, data)
            for e in pts:
                pts[e][t] = contact_point(model, data, *GEOM[e])

        foot_sl, hand_sl = [], []
        for ei, e in enumerate(eff):
            if e not in GEOM:
                continue
            col = flags[:, ei]
            both = col[1:] & col[:-1]                       # in contact at t-1 AND t
            disp = np.linalg.norm(np.diff(pts[e], axis=0), axis=1)[both] * 100  # cm/frame
            if disp.size:
                (foot_sl if "foot" in e else hand_sl).append(disp)
        foot_sl = np.concatenate(foot_sl) if foot_sl else np.array([0.0])
        hand_sl = np.concatenate(hand_sl) if hand_sl else np.array([0.0])

        # foot heading drift (contact) from stored orientations
        yaw_errs = []
        if "orientation_role_names" in z.files:
            oro = [str(x) for x in z["orientation_role_names"]]
            Rt, Ra = z["target_orientations"], z["achieved_orientations"]
            for foot in ("left_foot", "right_foot"):
                if foot in oro and foot in eff:
                    oi, ci = oro.index(foot), eff.index(foot)
                    con = flags[:, ci]
                    th_t = np.unwrap(np.radians([heading_deg(Rt[t, oi]) for t in range(T)]))
                    th_a = np.unwrap(np.radians([heading_deg(Ra[t, oi]) for t in range(T)]))
                    d = np.degrees(np.abs(th_a - th_t))[con]
                    if d.size:
                        yaw_errs.append(d.mean())
        yaw = float(np.mean(yaw_errs)) if yaw_errs else 0.0

        clip = os.path.basename(f).replace("_global_opt.npz", "")
        print(f"{clip:26s} {jpf['mean']:8.3f} {jA['mean']:8.3f} {jpf['spikes']:6d} {jA['spikes']:6d} "
              f"{foot_sl.mean():6.2f}/{foot_sl.max():4.1f} {hand_sl.mean():6.2f}/{hand_sl.max():4.1f} "
              f"{yaw:10.1f}deg")
        agg["jump_A"].append(jA["mean"]); agg["spk_A"].append(jA["spikes"])
        agg["foot_slip_mean"].append(foot_sl.mean()); agg["foot_slip_max"].append(foot_sl.max())
        agg["hand_slip_mean"].append(hand_sl.mean()); agg["hand_slip_max"].append(hand_sl.max())
        agg["foot_yaw"].append(yaw)

    print("-" * len(hdr))
    print(f"AVERAGE over {len(files)} clips:")
    print(f"  inter-frame jump (Stage A): mean={np.mean(agg['jump_A']):.3f} rad   "
          f"spikes(>0.5): mean={np.mean(agg['spk_A']):.1f}")
    print(f"  contact slip  foot: mean={np.mean(agg['foot_slip_mean']):.2f} cm/frame  "
          f"(max {np.mean(agg['foot_slip_max']):.1f})   "
          f"hand: mean={np.mean(agg['hand_slip_mean']):.2f} cm/frame "
          f"(max {np.mean(agg['hand_slip_max']):.1f})")
    print(f"  foot heading drift during contact: mean={np.mean(agg['foot_yaw']):.1f} deg")


if __name__ == "__main__":
    main()

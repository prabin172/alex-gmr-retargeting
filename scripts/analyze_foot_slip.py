#!/usr/bin/env python3
"""Analyse foot yaw (heading) drift during contact in a contact-first NPZ.

Hypothesis: while a foot is in contact the solver suppresses its world-delta
orientation and applies only the flat align (up-axis -> world +Z). Flat pins
pitch/roll but NOT yaw, so the foot heading is a free DOF that drifts as the rest
of the leg chain moves = the inner/outer rotation "slip". This compares the
STORED human target heading (what the foot *should* do) against the achieved
heading (what the solve did) over the clip, shading contact intervals.

Usage:
    python scripts/analyze_foot_slip.py \
        --npz outputs/contactfirst/shovel_fronthard_02_contactfirst.npz \
        --out outputs/analysis/shovel_fronthard_02_footslip.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def heading_deg(R):
    """Yaw of the foot forward axis (local +X) projected on the ground, degrees."""
    fwd = R @ np.array([1.0, 0.0, 0.0])
    return np.degrees(np.arctan2(fwd[1], fwd[0]))


def tilt_deg(R):
    """Angle of the foot up axis (local +Z) from world +Z, degrees."""
    up = R @ np.array([0.0, 0.0, 1.0])
    return np.degrees(np.arccos(np.clip(up[2], -1, 1)))


def intervals(mask):
    out, t, n = [], 0, len(mask)
    while t < n:
        if mask[t]:
            s = t
            while t < n and mask[t]:
                t += 1
            out.append((s, t - 1))
        else:
            t += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    ori_roles = [str(x) for x in z["orientation_role_names"]]
    eff = [str(x) for x in z["contact_effector_names"]]
    Rt = z["target_orientations"]      # (T,7,3,3)
    Ra = z["achieved_orientations"]    # (T,7,3,3)
    flags = z["contact_flags"].astype(bool)
    T = Rt.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for row, foot in enumerate(["left_foot", "right_foot"]):
        oi = ori_roles.index(foot)
        ei = eff.index(foot)
        th_t = np.unwrap(np.radians([heading_deg(Rt[t, oi]) for t in range(T)]))
        th_a = np.unwrap(np.radians([heading_deg(Ra[t, oi]) for t in range(T)]))
        th_t = np.degrees(th_t); th_a = np.degrees(th_a)
        tilt_a = np.array([tilt_deg(Ra[t, oi]) for t in range(T)])
        con = flags[:, ei]

        ax = axes[row]
        for (s, e) in intervals(con):
            ax.axvspan(s, e, color=(0.6, 0.9, 0.6), alpha=0.25, lw=0)
        ax.plot(th_t, label="target heading (human)", color="tab:blue", lw=1.5)
        ax.plot(th_a, label="achieved heading (solved)", color="tab:red", lw=1.5)
        ax2 = ax.twinx()
        ax2.plot(tilt_a, label="achieved flat-tilt", color="tab:gray", lw=0.8, alpha=0.7)
        ax2.set_ylabel("flat tilt (deg)", color="tab:gray")
        ax.set_ylabel("heading yaw (deg)")
        ax.set_title(f"{foot}  (green = in contact)")
        ax.legend(loc="upper left", fontsize=8)

        # stats over contact frames
        cerr = th_a[con] - th_t[con]
        print(f"\n{foot}: contact {con.sum()}/{T} ({con.sum()/T*100:.0f}%)")
        print(f"  achieved-vs-target heading (contact):  mean={np.mean(cerr):+.1f}  "
              f"std={np.std(cerr):.1f}  range={np.ptp(cerr):.1f} deg")
        # yaw wander of ACHIEVED within each contact interval vs target wander
        aw, tw = [], []
        for (s, e) in intervals(con):
            if e - s < 3:
                continue
            aw.append(np.ptp(th_a[s:e + 1]))
            tw.append(np.ptp(th_t[s:e + 1]))
        if aw:
            print(f"  per-interval heading wander: achieved={np.max(aw):.1f} deg  "
                  f"target={np.max(tw):.1f} deg  (achieved should ~ target if not drifting)")
        print(f"  achieved flat-tilt (contact): mean={tilt_a[con].mean():.1f}  max={tilt_a[con].max():.1f} deg")

    axes[1].set_xlabel("frame")
    fig.tight_layout()
    out = args.out or args.npz.with_suffix(".footslip.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nWrote plot: {out}")


if __name__ == "__main__":
    main()

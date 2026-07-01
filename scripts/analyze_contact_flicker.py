#!/usr/bin/env python3
"""Diagnose contact make/break flicker on the standup clips.

Contact is a per-frame binary AND of raw signals:
  foot: height<0.07  AND  speed<0.4  AND  sole_tilt<40deg
  hand: height<0.08  AND  speed<0.4
With no hysteresis/debounce, whenever a signal hovers near its threshold the flag
toggles frame-to-frame → the flat/align + palm-pin constraints snap on/off → pose
flicker. This reconstructs the raw signals from the canonical human NPZ (path read
from each GlobalOPT NPZ's metadata) and quantifies:
  * contact transitions + short on/off runs (flicker)
  * how many frames sit in a near-threshold margin per signal (chatter source)
  * the flat-tilt the solver is suddenly forced to correct at each foot make/break

Usage:
    python scripts/analyze_contact_flicker.py            # all standup clips
    python scripts/analyze_contact_flicker.py --all      # every clip
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import numpy as np

GO_DIR = "outputs/global_opt_contactfirst"
MARKERS = {
    "left_foot": (["left_ankle", "left_toe"], 0.07, "left_foot"),
    "right_foot": (["right_ankle", "right_toe"], 0.07, "right_foot"),
    "left_hand": (["left_wrist", "left_hand_middle"], 0.08, None),
    "right_hand": (["right_wrist", "right_hand_middle"], 0.08, None),
}
SPEED_THR = 0.4
FLAT_THR = 40.0
FLOOR_PCT = 1.0


def runs(mask):
    out, t, n = [], 0, len(mask)
    while t < n:
        s = t
        while t < n and mask[t] == mask[s]:
            t += 1
        out.append((mask[s], s, t - 1))
    return out


def signals(canon_path):
    z = np.load(canon_path, allow_pickle=True)
    roles = [str(x) for x in z["roles"]]
    ridx = {r: i for i, r in enumerate(roles)}
    P = np.asarray(z["positions"], float)          # (T,R,3)
    fps = float(z["fps"])
    T = P.shape[0]
    dt = 1.0 / fps
    ori_names = [str(x) for x in z["orientation_role_names"]]
    oidx = {r: i for i, r in enumerate(ori_names)}
    OM = np.asarray(z["orientation_mats"], float)   # (T,O,3,3)

    def mz(r): return P[:, ridx[r], 2]
    def spd(r):
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(P[:, ridx[r], :], axis=0), axis=1) / dt
        v[0] = v[1] if T > 1 else 0.0
        return v

    foot_roles = [r for e in ("left_foot", "right_foot") for r in MARKERS[e][0] if r in ridx]
    floor = float(np.percentile(np.min([mz(r) for r in foot_roles], axis=0), FLOOR_PCT))

    sig = {}
    for eff, (mks, hthr, flat) in MARKERS.items():
        mks = [r for r in mks if r in ridx]
        if not mks:
            continue
        h = np.min([mz(r) for r in mks], axis=0) - floor
        s = np.min([spd(r) for r in mks], axis=0)
        tilt = None
        if flat and flat in oidx:
            up = OM[:, oidx[flat], :, 2]
            tilt = np.degrees(np.arccos(np.clip(np.abs(up @ np.array([0, 0, 1.0])), -1, 1)))
        sig[eff] = dict(h=h, hthr=hthr, spd=s, tilt=tilt)
    return sig, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-run", type=int, default=4, help="runs shorter than this = flicker")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(GO_DIR, "*_global_opt.npz")))
    if not args.all:
        files = [f for f in files if "standup" in os.path.basename(f)]

    tot = {"trans": [], "flick": [], "marg_h": [], "marg_s": [], "marg_t": []}
    for f in files:
        z = np.load(f, allow_pickle=True)
        flags = z["contact_flags"].astype(bool)
        eff = [str(x) for x in z["contact_effector_names"]]
        meta = json.loads(z["metadata_json"].item())
        canon = meta["canonical"]
        clip = os.path.basename(f).replace("_global_opt.npz", "")
        try:
            sig, T = signals(canon)
        except Exception as e:
            print(f"{clip}: [skip] {e}"); continue

        print(f"\n=== {clip} ===")
        for ei, e in enumerate(eff):
            col = flags[:, ei]
            rr = runs(col)
            trans = len(rr) - 1
            flick = sum(1 for v, s, en in rr if (en - s + 1) < args.min_run)
            s = sig.get(e, {})
            # near-threshold margins (chatter-prone frames)
            mh = ms = mt = 0
            if s:
                mh = int((np.abs(s["h"] - s["hthr"]) < 0.02).sum())
                ms = int((np.abs(s["spd"] - SPEED_THR) < 0.1).sum())
                if s["tilt"] is not None:
                    mt = int((np.abs(s["tilt"] - FLAT_THR) < 10).sum())
            note = f"trans={trans:3d} flicker_runs={flick:3d}  near-thr[h={mh} spd={ms} tilt={mt}]"
            print(f"  {e:11s} contact={col.sum():3d}/{T}  {note}")
            tot["trans"].append(trans); tot["flick"].append(flick)
            tot["marg_h"].append(mh); tot["marg_s"].append(ms); tot["marg_t"].append(mt)

    print(f"\nAVERAGE over {len(files)} clips x effectors:")
    print(f"  transitions/eff={np.mean(tot['trans']):.1f}  flicker_runs/eff={np.mean(tot['flick']):.1f}")
    print(f"  near-threshold frames/eff: height={np.mean(tot['marg_h']):.1f}  "
          f"speed={np.mean(tot['marg_s']):.1f}  tilt={np.mean(tot['marg_t']):.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Joint-angle range audit for solved Alex qpos NPZs vs the model (URDF) limits.

Sweeps every `*.npz` in a folder, reads `qpos (T,36)` (free root 0-6, then the 29
actuated hinges 7-35), and for each actuated joint reports the achieved
min / mean / max angle across ALL frames of ALL npzs, next to the model's joint
limits. Flags any joint whose achieved range touches or exceeds a limit and how
much of the range it uses.

Usage:
    python scripts/jointLimitCheck.py outputs/grounded_contactfirst
    python scripts/jointLimitCheck.py <npz_dir> [--model <xml>] [--pattern '*.npz'] [--radians]

The npz folder is the positional argument. Angles printed in degrees by default
(--radians to print radians). Limits come from the MuJoCo model (default the
canonical fullmesh Alex), which mirrors the URDF joint ranges.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz_dir", type=Path, help="folder of solved *.npz (reads qpos)")
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--pattern", default="*.npz", help="glob within npz_dir (default *.npz)")
    ap.add_argument("--radians", action="store_true", help="print radians (default degrees)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))

    # Actuated hinge joints = every joint whose qpos address is >= 7 (0-6 = free root).
    joints = []
    for j in range(model.njnt):
        qadr = int(model.jnt_qposadr[j])
        if qadr < 7:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        limited = bool(model.jnt_limited[j])
        lo, hi = (float(model.jnt_range[j][0]), float(model.jnt_range[j][1])) if limited \
            else (-math.inf, math.inf)
        joints.append(dict(name=name, qadr=qadr, limited=limited, lo=lo, hi=hi))

    files = sorted(glob.glob(str(args.npz_dir / args.pattern)))
    if not files:
        raise SystemExit(f"[jointLimitCheck] no npz matching {args.pattern} in {args.npz_dir}")

    qadrs = np.array([j["qadr"] for j in joints])
    los = np.array([j["lo"] for j in joints])
    his = np.array([j["hi"] for j in joints])
    limited = np.array([j["limited"] for j in joints])
    tol_rad = math.radians(1.0)   # "at limit" = within 1 deg of a bound
    nJ = len(joints)
    gmin = np.full(nJ, np.inf)
    gmax = np.full(nJ, -np.inf)
    gsum = np.zeros(nJ)
    gcount = 0
    n_at_limit = np.zeros(nJ)     # frames within tol of EITHER bound (finite limits only)
    # which file drives each extreme (handy for debugging outliers)
    fmin = [""] * nJ
    fmax = [""] * nJ

    for f in files:
        z = np.load(f, allow_pickle=True)
        if "qpos" not in z:
            print(f"  [skip] {Path(f).name}: no qpos")
            continue
        q = np.asarray(z["qpos"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] < 36:
            print(f"  [skip] {Path(f).name}: qpos shape {q.shape}")
            continue
        vals = q[:, qadrs]                      # (T, nJ)
        vmin = vals.min(axis=0)
        vmax = vals.max(axis=0)
        gsum += vals.sum(axis=0)
        gcount += vals.shape[0]
        at = limited & ((vals <= los + tol_rad) | (vals >= his - tol_rad))  # (T,nJ)
        n_at_limit += at.sum(axis=0)
        for k in range(nJ):
            if vmin[k] < gmin[k]:
                gmin[k] = vmin[k]; fmin[k] = Path(f).name
            if vmax[k] > gmax[k]:
                gmax[k] = vmax[k]; fmax[k] = Path(f).name

    gmean = gsum / max(gcount, 1)
    conv = (lambda r: r) if args.radians else np.degrees
    unit = "rad" if args.radians else "deg"

    print(f"\nJoint-angle audit  |  {len(files)} npz, {gcount} frames total  |  model={args.model.name}")
    print(f"units = {unit}   (limits from model = URDF ranges)\n")
    hdr = (f"{'joint':<18}{'lim_lo':>9}{'lim_hi':>9} | "
           f"{'min':>9}{'mean':>9}{'max':>9} | {'%range':>7}{'%@lim':>7}  flag")
    print(hdr)
    print("-" * len(hdr))

    any_viol = False
    for k, j in enumerate(joints):
        lo, hi = j["lo"], j["hi"]
        amin, amean, amax = gmin[k], gmean[k], gmax[k]
        # flag: touching (within ~1 deg) or exceeding a finite limit. All compares
        # in radians (model native); tol is 1 deg expressed in radians.
        tol_rad = math.radians(1.0)
        flag = ""
        pct = float("nan")
        if j["limited"]:
            rng = hi - lo
            pct = 100.0 * (amax - amin) / rng if rng > 0 else float("nan")
            over_lo = amin < lo
            over_hi = amax > hi
            near_lo = amin <= lo + tol_rad
            near_hi = amax >= hi - tol_rad
            if over_lo or over_hi:
                flag = "VIOLATE" + (" lo" if over_lo else "") + (" hi" if over_hi else "")
                any_viol = True
            elif near_lo or near_hi:
                flag = "at-limit" + (" lo" if near_lo else "") + (" hi" if near_hi else "")
        lo_s = f"{conv(lo):>9.1f}" if j["limited"] else f"{'-inf':>9}"
        hi_s = f"{conv(hi):>9.1f}" if j["limited"] else f"{'+inf':>9}"
        pct_s = f"{pct:>7.0f}" if j["limited"] and not math.isnan(pct) else f"{'--':>7}"
        atlim_pct = 100.0 * n_at_limit[k] / max(gcount, 1)
        atlim_s = f"{atlim_pct:>7.0f}" if j["limited"] else f"{'--':>7}"
        print(f"{j['name']:<18}{lo_s}{hi_s} | "
              f"{conv(amin):>9.1f}{conv(amean):>9.1f}{conv(amax):>9.1f} | "
              f"{pct_s}{atlim_s}  {flag}")

    print("-" * len(hdr))
    print("VIOLATIONS PRESENT" if any_viol else "no limit violations")
    # show which file drove any violated extreme
    if any_viol:
        print("\nextreme-driving file per violated joint:")
        for k, j in enumerate(joints):
            if not j["limited"]:
                continue
            if gmin[k] < j["lo"] or gmax[k] > j["hi"]:
                print(f"  {j['name']:<18} min<-{fmin[k]}   max<-{fmax[k]}")


if __name__ == "__main__":
    main()

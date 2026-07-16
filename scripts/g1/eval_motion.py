#!/usr/bin/env python3
"""Model-agnostic reference-free physical-plausibility eval — G1 port of
`scripts/eval_ihmc_json.py`. Reuses `evaluate()` unchanged (it already only takes
model/data/mesh_cache/geom_ids/sole_sids/q_lo/q_hi/mj_joint_names as arguments —
nothing Alex-specific baked in); only the per-format loader and default model path
differ. Stance/slip metrics (need contact flags) and self-collision (G1 mocap XML's
collision pairs not vetted this week) are reported but flagged n/a-ish for G1 sources
— see the printed caveat.

Two input formats:
  --ihmc-json   Alex Stage-6 export / externally-produced IHMC JSON (identical to
                eval_ihmc_json.py's own `_load`, imported unchanged).
  --gmr-pkl     GMR's own pkl format (root_pos, root_rot xyzw, dof_pos) via
                scripts/g1/load_gmr_pkl.py — no contact flags, so stance/slip skipped.

Usage:
    conda run -n gmr python scripts/g1/eval_motion.py --gmr-pkl \\
        outputs/gmr_baseline/pkl/walk1_subject1.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Insert scripts/, NOT repo root -- repo root contains an empty, gitignored
# `general_motion_retargeting/` leftover dir that SHADOWS the real pip-installed GMR
# package if it ever lands on sys.path (see planLogGMR.md, T1 "package shadowing").
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_ihmc_json import (  # noqa: E402
    MODEL_DEFAULT as ALEX_MODEL_DEFAULT, SPIKE_RAD_PER_S, _load as _load_ihmc_json,
    evaluate)
from export_alex_retarget_npz_to_ihmc_json import load_mujoco_joint_order  # noqa: E402
from post_process_ground_contactfirst import (  # noqa: E402
    SOLE_CORNER_SITES, _build_mesh_cache, _robot_lowest_z)

G1_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


def build_eval_context(model_path: Path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    mj_joint_names = load_mujoco_joint_order(model_path)
    n_act = len(mj_joint_names)
    q_lo = np.full(n_act, -1e6)
    q_hi = np.full(n_act, 1e6)
    act = 0
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:  # free joint (root)
            continue
        if bool(model.jnt_limited[j]):
            q_lo[act] = model.jnt_range[j, 0]
            q_hi[act] = model.jnt_range[j, 1]
        act += 1
    return model, data, mesh_cache, geom_ids, mj_joint_names, q_lo, q_hi


def _alex_sole_sids(model):
    sole_sids = {}
    for f, names in SOLE_CORNER_SITES.items():
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]
        ids = [s for s in ids if s >= 0]
        if len(ids) == 4:
            sole_sids[f] = ids
    return sole_sids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ihmc-json", action="store_true")
    src.add_argument("--gmr-pkl", action="store_true")
    ap.add_argument("--model", type=Path, default=None,
                    help="Override model XML (default: Alex for --ihmc-json, G1 for --gmr-pkl).")
    ap.add_argument("--limit-tol-deg", type=float, default=1.0)
    args = ap.parse_args()

    model_path = args.model or (ALEX_MODEL_DEFAULT if args.ihmc_json else G1_MODEL_DEFAULT)
    model, data, mesh_cache, geom_ids, mj_joint_names, q_lo, q_hi = build_eval_context(model_path)
    sole_sids = _alex_sole_sids(model) if args.ihmc_json else {}
    has_contacts = args.ihmc_json  # G1/GMR sources carry no contact flags this week

    rows = []
    for path in args.inputs:
        if args.ihmc_json:
            qpos, fps, contacts = _load_ihmc_json(path, mj_joint_names)
        else:
            from load_gmr_pkl import load_gmr_pkl  # noqa: E402  (deferred import, T5)
            qpos, fps = load_gmr_pkl(path)
            contacts = {}
        print(f"[load] {path}  T={qpos.shape[0]}  fps={fps:g}")
        rows.append(evaluate(path.stem, qpos, fps, contacts, model, data, mesh_cache,
                             geom_ids, sole_sids, q_lo, q_hi, mj_joint_names,
                             args.limit_tol_deg))

    hdr = ["name", "T", "fps", "floorPen", "pen%", "coll%", "collPk", "JLvi", "worst_joint",
           "vMax", "vP95", "spikes", "rootV", "plPen", "plFloat", "plSlip"]
    print()
    print(f"{hdr[0]:<26}{hdr[1]:>6}{hdr[2]:>5}{hdr[3]:>9}{hdr[4]:>7}{hdr[5]:>7}{hdr[6]:>7}"
          f"{hdr[7]:>5}{hdr[8]:>18}{hdr[9]:>7}{hdr[10]:>7}{hdr[11]:>7}{hdr[12]:>7}"
          f"{hdr[13]:>7}{hdr[14]:>8}{hdr[15]:>7}")
    for r in rows:
        print(f"{r['name']:<26}{r['T']:>6}{r['fps']:>5.0f}{r['floor_pen_max_cm']:>8.1f}c"
              f"{r['floor_pen_pct']:>6.1f}%{r['coll_pct']:>6.1f}%{r['coll_peak_cm']:>6.1f}c"
              f"{r['jl_viol']:>5}{r['worst_joint']:>18}{r['vel_max_rad_s']:>7.1f}"
              f"{r['vel_p95_rad_s']:>7.1f}{r['n_spikes']:>7}{r['root_v_max']:>7.2f}"
              f"{r['plant_pen_max_cm']:>6.1f}c{r['plant_float_med_cm']:>7.1f}c"
              f"{r['plant_slip_max_cm']:>6.1f}c")
    print("\nfloorPen = mesh-exact whole-body max below z=0 (cm); pen% = frames >0.5cm."
          "\ncoll% / collPk = self-collision incidence / peak depth (fullmesh)."
          + ("" if has_contacts else
             "  [G1: model's self-collision pairs not vetted this week -- treat as informational, not a hard number.]")
          + "\nJLvi = hard joint-limit violations (frame*joint); worst_joint = % frames pinned within "
          f"{args.limit_tol_deg:g} deg of a limit."
          f"\nvMax/vP95 (rad/s), spikes = frames > {SPIKE_RAD_PER_S:g} rad/s (0.5 rad/frame @120Hz)."
          "\nrootV = max root linear speed (m/s)."
          "\nplPen/plFloat/plSlip = planted-sole pen / median float / max in-run XY drift (cm), from "
          "the source's OWN foot contact flags."
          + ("" if has_contacts else "  [n/a for GMR pkl sources -- no contact flags; always 0.]"))


if __name__ == "__main__":
    main()

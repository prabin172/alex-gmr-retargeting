#!/usr/bin/env python3
"""Reference-free physical-plausibility eval for IHMC KinematicsToolboxOutputStatus
JSONs (the Stage-6 export format, and the format of externally-produced motions
like the manual Blender retargets in data/blender-retargeted/).

Loads each JSON back into MuJoCo qpos convention ([x y z qw qx qy qz | 29 joints],
inverting export_alex_retarget_npz_to_ihmc_json.py's joint reorder + quat swap)
and measures ONLY metrics that need no human reference — an externally edited /
retimed motion has no frame correspondence to our canonical human data, so
tracking-fidelity metrics would be meaningless for it:

  * floor penetration  — mesh-exact whole-body lowest-Z vs z=0 (same
    _robot_lowest_z machinery as Stage 4.5 grounding)
  * self-collision     — incidence % + peak depth (fullmesh, same
    _collision_stats as Stage 4)
  * joint limits       — hard violations + worst near-limit-pinned joint
  * velocity           — rad/s (rate-aware: files may be 50 Hz or 120 Hz);
    spike threshold 60 rad/s = the codebase's 0.5 rad/frame @120 Hz convention
  * stance (from the JSON's own foot_in_contact flags) — planted-foot sole
    penetration/float vs z=0, and XY drift within each contact run (slip proxy)

Usage:
    conda run -n gmr python scripts/eval_ihmc_json.py \\
        data/blender-retargeted/standSupine.json \\
        outputs/ihmcJsons50hz/luigi_standSupine_08.json \\
        [--save-npz outputs/cont_dev]   # dump qpos NPZs for the renderer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_alex_retarget_npz_to_ihmc_json import (  # noqa: E402
    ISAAC_JOINT_NAMES_FULLBODY, load_mujoco_joint_order)
from post_process_ground_contactfirst import (  # noqa: E402
    MODEL_DEFAULT, SOLE_CORNER_SITES, _build_mesh_cache, _robot_lowest_z)
from solve_global_trajectory_opt_contactfirst import _collision_stats  # noqa: E402

MAIN_KEY = "toolbox_msgs.msg.dds.KinematicsToolboxOutputStatus"
INNER_KEY = "toolbox_msgs::msg::dds_::KinematicsToolboxOutputStatus_"
SPIKE_RAD_PER_S = 60.0   # = 0.5 rad/frame at 120 Hz, the codebase's spike convention


def _load(path: Path, mj_joint_names):
    """JSON -> (qpos (T,36) MuJoCo order/convention, fps, contacts dict)."""
    d = json.loads(path.read_text())
    inner = d[MAIN_KEY]
    ts = np.asarray(inner["timestamps"], dtype=np.float64)
    fps = 1000.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 120.0

    # Invert the exporter's reorder: isaac[i] = mj[reorder[i]]  =>  mj[reorder[i]] = isaac[i]
    mj_idx = {n: i for i, n in enumerate(mj_joint_names)}
    reorder = [mj_idx[n] for n in ISAAC_JOINT_NAMES_FULLBODY]

    T = len(inner["messages"])
    qpos = np.zeros((T, 36))
    contacts = {"left_foot": np.zeros(T, bool), "right_foot": np.zeros(T, bool)}
    for t, msg in enumerate(inner["messages"]):
        rec = json.loads(msg)[INNER_KEY]
        p = rec["desired_root_position"]
        q = rec["desired_root_orientation"]          # xyzw in JSON
        qpos[t, 0:3] = [p["x"], p["y"], p["z"]]
        quat = np.array([q["w"], q["x"], q["y"], q["z"]])   # -> wxyz
        qpos[t, 3:7] = quat / max(np.linalg.norm(quat), 1e-12)
        ja = np.asarray(rec["desired_joint_angles"], dtype=np.float64)
        for i, mj_i in enumerate(reorder):
            qpos[t, 7 + mj_i] = ja[i]
        contacts["left_foot"][t] = bool(rec.get("left_foot_in_contact", False))
        contacts["right_foot"][t] = bool(rec.get("right_foot_in_contact", False))
    return qpos, fps, contacts


def _contact_runs(flags):
    runs, t, n = [], 0, len(flags)
    while t < n:
        if flags[t]:
            s = t
            while t < n and flags[t]:
                t += 1
            runs.append((s, t))
        else:
            t += 1
    return runs


def evaluate(name, qpos, fps, contacts, model, data, mesh_cache, geom_ids,
             sole_sids, q_lo, q_hi, mj_joint_names, limit_tol_deg=1.0):
    T = qpos.shape[0]

    # --- floor penetration (mesh-exact, vs z=0) + sole-corner Zs cached per frame
    lowest = np.zeros(T)
    sole_z = {f: np.zeros(T) for f in sole_sids}
    sole_xy = {f: np.zeros((T, 2)) for f in sole_sids}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)
        for f, sids in sole_sids.items():
            pts = np.array([data.site_xpos[s] for s in sids])
            sole_z[f][t] = pts[:, 2].min()
            sole_xy[f][t] = pts[:, :2].mean(axis=0)
    pen = np.maximum(0.0, -lowest)

    # --- self-collision (plain model has no floor geom; worldbody excluded by bodyid==0)
    cs = _collision_stats(model, data, qpos, floor_gid=None, count_floor=False)

    # --- joint limits
    qj = qpos[:, 7:]
    tol = 1e-3
    hard_viol = int(((qj < q_lo - tol) | (qj > q_hi + tol)).sum())
    tol_r = np.radians(limit_tol_deg)
    near = ((qj <= q_lo + tol_r) | (qj >= q_hi - tol_r))
    pinned_pct = near.mean(axis=0) * 100
    worst_j = int(np.argmax(pinned_pct))

    # --- velocity (rate-aware)
    vel = np.abs(np.diff(qj, axis=0)) * fps            # rad/s
    vmax_pf = vel.max(axis=1) if T > 1 else np.zeros(1)
    n_spikes = int((vmax_pf > SPIKE_RAD_PER_S).sum())
    root_v = (np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * fps) if T > 1 else np.zeros(1)

    # --- stance metrics from the JSON's own contact flags
    plant_pen, plant_float, slip = [], [], []
    for f in sole_sids:
        fl = contacts.get(f, np.zeros(T, bool))
        z = sole_z[f][fl]
        if z.size:
            plant_pen.extend((-z[z < 0]).tolist())
            plant_float.extend(z[z >= 0].tolist())
        for s, e in _contact_runs(fl):
            xy = sole_xy[f][s:e]
            slip.append(float(np.linalg.norm(xy - xy[0], axis=1).max()))
    contact_frames = int(sum(contacts[f].sum() for f in sole_sids))

    return {
        "name": name, "T": T, "fps": fps,
        "floor_pen_max_cm": pen.max() * 100,
        "floor_pen_pct": (pen > 0.005).mean() * 100,
        "lowest_z_med_cm": float(np.median(lowest)) * 100,
        "coll_pct": cs["pct"], "coll_peak_cm": cs["max_pen_cm"],
        "jl_viol": hard_viol,
        "worst_joint": f"{mj_joint_names[worst_j]}({pinned_pct[worst_j]:.0f}%)",
        "vel_max_rad_s": float(vmax_pf.max()), "vel_p95_rad_s": float(np.percentile(vmax_pf, 95)),
        "n_spikes": n_spikes, "root_v_max": float(root_v.max()),
        "contact_frames": contact_frames,
        "plant_pen_max_cm": (max(plant_pen) * 100 if plant_pen else 0.0),
        "plant_float_med_cm": (float(np.median(plant_float)) * 100 if plant_float else 0.0),
        "plant_slip_max_cm": (max(slip) * 100 if slip else 0.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsons", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--limit-tol-deg", type=float, default=1.0)
    ap.add_argument("--save-npz", type=Path, default=None,
                    help="Also write <stem>_ihmc.npz (qpos/fps/source_frame_ids) per input "
                         "into this dir — renderable via render_contactfirst.py.")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    mj_joint_names = load_mujoco_joint_order(args.model)
    sole_sids = {}
    for f, names in SOLE_CORNER_SITES.items():
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]
        ids = [s for s in ids if s >= 0]
        if len(ids) == 4:
            sole_sids[f] = ids
    q_lo = np.full(29, -1e6); q_hi = np.full(29, 1e6)
    act = 0
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            continue
        if bool(model.jnt_limited[j]):
            q_lo[act] = model.jnt_range[j, 0]; q_hi[act] = model.jnt_range[j, 1]
        act += 1

    rows = []
    for path in args.jsons:
        qpos, fps, contacts = _load(path, mj_joint_names)
        print(f"[load] {path}  T={qpos.shape[0]}  fps={fps:g}")
        rows.append(evaluate(path.stem, qpos, fps, contacts, model, data, mesh_cache,
                             geom_ids, sole_sids, q_lo, q_hi, mj_joint_names,
                             args.limit_tol_deg))
        if args.save_npz is not None:
            args.save_npz.mkdir(parents=True, exist_ok=True)
            out = args.save_npz / f"{path.stem}_ihmc.npz"
            np.savez_compressed(out, qpos=qpos, fps=np.float64(fps),
                                source_frame_ids=np.arange(qpos.shape[0]))
            print(f"  -> {out}")

    hdr = ["name", "T", "fps", "floorPen", "pen%", "coll%", "collPk", "JLvi", "worst_joint",
           "vMax", "vP95", "spikes", "rootV", "plPen", "plFloat", "plSlip"]
    print()
    print(f"{hdr[0]:<22}{hdr[1]:>6}{hdr[2]:>5}{hdr[3]:>9}{hdr[4]:>7}{hdr[5]:>7}{hdr[6]:>7}"
          f"{hdr[7]:>5}{hdr[8]:>18}{hdr[9]:>7}{hdr[10]:>7}{hdr[11]:>7}{hdr[12]:>7}"
          f"{hdr[13]:>7}{hdr[14]:>8}{hdr[15]:>7}")
    for r in rows:
        print(f"{r['name']:<22}{r['T']:>6}{r['fps']:>5.0f}{r['floor_pen_max_cm']:>8.1f}c"
              f"{r['floor_pen_pct']:>6.1f}%{r['coll_pct']:>6.1f}%{r['coll_peak_cm']:>6.1f}c"
              f"{r['jl_viol']:>5}{r['worst_joint']:>18}{r['vel_max_rad_s']:>7.1f}"
              f"{r['vel_p95_rad_s']:>7.1f}{r['n_spikes']:>7}{r['root_v_max']:>7.2f}"
              f"{r['plant_pen_max_cm']:>6.1f}c{r['plant_float_med_cm']:>7.1f}c"
              f"{r['plant_slip_max_cm']:>6.1f}c")
    print("\nfloorPen = mesh-exact whole-body max below z=0 (cm); pen% = frames >0.5cm."
          "\ncoll% / collPk = self-collision incidence / peak depth (fullmesh)."
          "\nJLvi = hard joint-limit violations (frame*joint); worst_joint = % frames pinned within "
          f"{args.limit_tol_deg:g} deg of a limit."
          f"\nvMax/vP95 (rad/s), spikes = frames > {SPIKE_RAD_PER_S:g} rad/s (0.5 rad/frame @120Hz)."
          "\nrootV = max root linear speed (m/s)."
          "\nplPen/plFloat/plSlip = planted-sole pen / median float / max in-run XY drift (cm), from "
          "the JSON's OWN foot contact flags.")


if __name__ == "__main__":
    main()

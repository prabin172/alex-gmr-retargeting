"""
Compute per-clip and aggregate metrics for all global_opt NPZ outputs.
Prints a table matching the whitePaper.md Section 6.4 format and
an extended breakdown per clip.

Usage:
    python scripts/compute_globalopt_metrics.py
"""

import json
import os
import sys

import mujoco
import numpy as np

GLOBAL_OPT_DIR = "outputs/global_opt"
MODEL_PATH = "assets/alex/alex_floating_base_with_sites.xml"
COLL_HOPS = 4
COLL_MARGIN = 0.0  # dist < 0 means penetrating


# ---------------------------------------------------------------------------
# Helpers (mirrors solve_global_trajectory_opt.py)
# ---------------------------------------------------------------------------

def _within_k_hops(model, b1, b2, k):
    visited = {b1}
    frontier = {b1}
    for _ in range(k):
        nxt = set()
        for b in frontier:
            p = model.body_parentid[b]
            if p not in visited:
                nxt.add(p)
            for c in range(model.nbody):
                if model.body_parentid[c] == b and c not in visited:
                    nxt.add(c)
        visited |= nxt
        frontier = nxt
        if b2 in visited:
            return True
    return False


def _delta_stats(qpos):
    dq = np.abs(np.diff(qpos[:, 7:], axis=0))
    max_per_frame = dq.max(axis=1)
    return {
        "max":         float(max_per_frame.max()),
        "p95":         float(np.percentile(max_per_frame, 95)),
        "n_spikes_05": int((max_per_frame > 0.5).sum()),
    }


def _collision_stats(model, data, qpos):
    pen_per_frame = []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        max_pen = 0.0
        for c in range(data.ncon):
            ct = data.contact[c]
            b1 = int(model.geom_bodyid[ct.geom1])
            b2 = int(model.geom_bodyid[ct.geom2])
            if b1 == 0 or b2 == 0:
                continue
            if _within_k_hops(model, b1, b2, COLL_HOPS):
                continue
            if ct.dist < 0:
                max_pen = max(max_pen, abs(float(ct.dist)))
        pen_per_frame.append(max_pen)
    arr = np.array(pen_per_frame)
    n_coll = int((arr > 0).sum())
    return {
        "pct":         n_coll / len(arr) * 100,
        "max_pen_cm":  float(arr.max()) * 100,
    }


def _tracking_stats(qpos, target_positions, role_names, role_to_body, model, data):
    errs = []
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for ri, role in enumerate(role_names):
            if role not in role_to_body:
                continue
            bid = role_to_body[role]
            errs.append(float(np.linalg.norm(target_positions[t, ri] - data.xpos[bid])))
    arr = np.array(errs)
    return {"mean": float(arr.mean())}


def _row(label, d, c, tr):
    return (f"  {label:22s}  "
            f"spikes={d['n_spikes_05']:4d}  "
            f"max_dq={d['max']:.3f}  "
            f"p95_dq={d['p95']:.3f}  "
            f"coll={c['pct']:5.1f}%  "
            f"peak_pen={c['max_pen_cm']:.1f}cm  "
            f"track_mean={tr['mean']:.4f}m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    npz_files = sorted(
        f for f in os.listdir(GLOBAL_OPT_DIR) if f.endswith(".npz")
    )
    if not npz_files:
        print(f"No NPZ files found in {GLOBAL_OPT_DIR}", file=sys.stderr)
        sys.exit(1)

    # accumulators for aggregate stats
    agg = {k: {"spikes": [], "max_dq": [], "p95_dq": [], "coll_pct": [],
               "peak_pen": [], "track_mean": []}
           for k in ("per_frame", "stage_a", "stage_b")}

    print(f"\nProcessing {len(npz_files)} clips from {GLOBAL_OPT_DIR}/\n")

    for fname in npz_files:
        path = os.path.join(GLOBAL_OPT_DIR, fname)
        d = np.load(path, allow_pickle=True)

        meta        = json.loads(d["metadata_json"].item())
        role_names  = list(d["role_names"])
        alex_bodies = list(d["alex_body_names"])
        role_to_body = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
                        for role, bname in zip(role_names, alex_bodies)
                        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname) >= 0}

        target_pos  = d["target_positions"]   # (T, 15, 3)
        qpos_pf     = d["qpos_per_frame"]
        qpos_a      = d["qpos_stage_a"]
        qpos_b      = d["qpos_stage_b"]

        clip = fname.replace("_global_opt.npz", "")
        print(f"{'─'*80}")
        print(f"  Clip: {clip}  ({qpos_pf.shape[0]} frames @ {float(d['fps']):.0f} fps)")
        print(f"{'─'*80}")

        for key, qpos, label in [
            ("per_frame", qpos_pf, "per-frame IK"),
            ("stage_a",   qpos_a,  "Stage A (smooth)"),
            ("stage_b",   qpos_b,  "Stage B (QP+SCA)"),
        ]:
            ds = _delta_stats(qpos)
            cs = _collision_stats(model, data, qpos)
            ts = _tracking_stats(qpos, target_pos, role_names, role_to_body, model, data)
            print(_row(label, ds, cs, ts))

            agg[key]["spikes"].append(ds["n_spikes_05"])
            agg[key]["max_dq"].append(ds["max"])
            agg[key]["p95_dq"].append(ds["p95"])
            agg[key]["coll_pct"].append(cs["pct"])
            agg[key]["peak_pen"].append(cs["max_pen_cm"])
            agg[key]["track_mean"].append(ts["mean"])

        print()

    # -----------------------------------------------------------------------
    # Aggregate table (mean across clips) — goes into whitePaper Section 6.4
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  AGGREGATE (mean across all clips)")
    print(f"{'='*80}")
    print(f"  {'method':22s}  {'spikes':>6}  {'max_dq':>7}  {'p95_dq':>7}  "
          f"{'coll%':>6}  {'peak_pen':>9}  {'track_mean':>11}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*9}  {'-'*11}")

    labels = {
        "per_frame": "per-frame IK",
        "stage_a":   "Stage A (smooth)",
        "stage_b":   "Stage B (QP+SCA)",
    }
    for key, label in labels.items():
        a = agg[key]
        print(f"  {label:22s}  "
              f"{np.mean(a['spikes']):6.1f}  "
              f"{np.mean(a['max_dq']):7.3f}  "
              f"{np.mean(a['p95_dq']):7.3f}  "
              f"{np.mean(a['coll_pct']):5.1f}%  "
              f"{np.mean(a['peak_pen']):7.1f}cm  "
              f"{np.mean(a['track_mean']):10.4f}m")

    print(f"\n  Clips: {len(npz_files)}")

    # -----------------------------------------------------------------------
    # Reduction ratios (for the paper)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  REDUCTION vs per-frame IK")
    print(f"{'='*80}")
    pf_spikes = np.mean(agg["per_frame"]["spikes"])
    pf_coll   = np.mean(agg["per_frame"]["coll_pct"])
    for key, label in [("stage_a", "Stage A"), ("stage_b", "Stage B")]:
        sp  = np.mean(agg[key]["spikes"])
        col = np.mean(agg[key]["coll_pct"])
        spike_red = (pf_spikes - sp) / max(pf_spikes, 1e-9) * 100
        coll_red  = (pf_coll  - col) / max(pf_coll,  1e-9) * 100
        print(f"  {label}: spike reduction={spike_red:.1f}%  coll reduction={coll_red:.1f}%")

    print()


if __name__ == "__main__":
    main()

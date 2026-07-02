#!/usr/bin/env python3
"""Naive Z-grounding post-step for the contact-first pipeline.

Takes a smoothed GlobalOPT NPZ (`solve_global_trajectory_opt_contactfirst.py`),
computes the robot's true lowest collision point per frame (mesh-vertex aware —
the v2 model uses convex-hull meshes on the fists/limbs), and shifts the free
root Z so the clip rests on the ground plane z=0.

Modes:
  constant  (default) — ONE shift for the whole clip: floor = a low percentile
            of per-frame lowest-Z; shift all frames by -floor. Preserves every
            bit of vertical motion (crouch/stand), never adds jitter. The clip's
            lowest moments touch the floor; other frames float slightly above.
  perframe  — shift each frame so its lowest point sits exactly at 0 (full
            plant, both up and down). Optional --smooth-shift to de-jitter.

Only qpos[:,2] (root Z) is modified. All other keys are copied through; the
original ungrounded qpos is kept as `qpos_ungrounded`.

Usage:
    python scripts/post_process_ground_contactfirst.py \\
        --npz outputs/global_opt_contactfirst/standup_01_global_opt.npz \\
        --out outputs/grounded_contactfirst/standup_01_grounded.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"


def _build_mesh_cache(model: mujoco.MjModel):
    """geom_id -> (V,3) local mesh vertices, for MESH geoms only."""
    cache = {}
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        if mid < 0:
            continue
        adr = int(model.mesh_vertadr[mid])
        num = int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[adr:adr + num].reshape(-1, 3).astype(np.float64)
        cache[g] = verts
    return cache


def _geom_lowest_z(g, model, data, mesh_cache):
    gtype = int(model.geom_type[g])
    pos = data.geom_xpos[g]
    mat = data.geom_xmat[g].reshape(3, 3)
    sz = model.geom_size[g]

    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        verts = mesh_cache.get(g)
        if verts is None:
            return float(pos[2])
        # world = pos + R @ v ; we only need the Z row of R.
        world_z = pos[2] + verts @ mat[2, :]
        return float(world_z.min())

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(pos[2] - sz[0])

    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half_len = float(sz[0]), float(sz[1])
        axis_z = float(mat[2, 2])
        return min(pos[2] + axis_z * half_len, pos[2] - axis_z * half_len) - radius

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        hx, hy, hz = float(sz[0]), float(sz[1]), float(sz[2])
        return float(pos[2]) - abs(mat[2, 0]) * hx - abs(mat[2, 1]) * hy - abs(mat[2, 2]) * hz

    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, half_len = float(sz[0]), float(sz[1])
        axis_z = float(mat[2, 2])
        sin_tilt = float(np.sqrt(max(0.0, 1.0 - axis_z ** 2)))
        return min(pos[2] + axis_z * half_len, pos[2] - axis_z * half_len) - radius * sin_tilt

    return float(pos[2])  # plane/hfield — conservative


def _robot_lowest_z(model, data, mesh_cache, geom_ids):
    return min(_geom_lowest_z(g, model, data, mesh_cache) for g in geom_ids)


def _smooth1d(x, w):
    """Tridiagonal (implicit) smoothing: (I + w L) y = x. Endpoints natural."""
    n = len(x)
    if n < 3 or w <= 0:
        return x.copy()
    a = np.full(n, -w)
    b = np.full(n, 1.0 + 2.0 * w)
    c = np.full(n, -w)
    b[0] = 1.0 + w
    b[-1] = 1.0 + w
    a[0] = 0.0
    c[-1] = 0.0
    # Thomas algorithm
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = x[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (x[i] - a[i] * dp[i - 1]) / m
    y = np.zeros(n)
    y[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        y[i] = dp[i] - cp[i] * y[i + 1]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    ap.add_argument("--mode", choices=["constant", "perframe"], default="constant")
    ap.add_argument("--percentile", type=float, default=1.0,
                    help="constant mode: floor = this percentile of per-frame lowest-Z (robust to outliers).")
    ap.add_argument("--smooth-shift", type=float, default=0.0,
                    help="perframe mode: tridiagonal smoothing weight on the shift series (0 = off).")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    z = np.load(args.npz, allow_pickle=True)
    data_dict = {k: z[k] for k in z.files}
    qpos = data_dict["qpos"].astype(np.float64).copy()
    N = qpos.shape[0]

    lowest = np.empty(N)
    for t in range(N):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)

    if args.mode == "constant":
        floor = float(np.percentile(lowest, args.percentile))
        shift = np.full(N, -floor)
    else:
        shift = -lowest
        if args.smooth_shift > 0:
            shift = _smooth1d(shift, args.smooth_shift)

    qpos_grounded = qpos.copy()
    qpos_grounded[:, 2] += shift

    grounded_lowest = lowest + shift
    data_dict["qpos_ungrounded"] = qpos
    data_dict["qpos"] = qpos_grounded
    data_dict["ground_shift"] = shift
    data_dict["ground_mode"] = args.mode
    data_dict["ground_lowest_before"] = lowest
    data_dict["ground_lowest_after"] = grounded_lowest

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data_dict)

    print(f"[ground] {args.npz.name}  N={N}  mode={args.mode}")
    print(f"  lowest-Z before: min={lowest.min():+.4f} med={np.median(lowest):+.4f} max={lowest.max():+.4f}")
    if args.mode == "constant":
        print(f"  floor(p{args.percentile})={floor:+.4f}  constant shift={shift[0]:+.4f} m")
    else:
        print(f"  perframe shift: min={shift.min():+.4f} max={shift.max():+.4f} smooth={args.smooth_shift}")
    print(f"  lowest-Z after : min={grounded_lowest.min():+.4f} med={np.median(grounded_lowest):+.4f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Naive Z-grounding post-step for the contact-first pipeline.

Takes a smoothed GlobalOPT NPZ (`solve_global_trajectory_opt_contactfirst.py`),
computes the robot's true lowest collision point per frame (mesh-vertex aware —
the v2 model uses convex-hull meshes on the fists/limbs), and shifts the free
root Z so the clip rests on the ground plane z=0.

Modes:
  hybrid    — constant-contact base shift + a smooth NON-NEGATIVE per-frame lift
            solved as a small banded QP. The lift raises frames whose whole-body
            lowest point is below the floor (the between-phase sink: a get-up's
            lying/crouch phase registers metres of torso/foot "penetration" when
            the single shift is keyed to the standing stance) while a per-frame
            cap keeps every STILL-PLANTED foot pinned to the floor (it may only
            lift as far as that foot's own penetration + --lift-float-tol, so
            plants never float). Smoothness term prevents bobbing/spikes; lift is
            0 wherever the base shift already clears the floor.
  constant-contact — ONE shift for the whole clip, but the floor is
            registered to the PLANTED FEET (sole-corner sites on frames where a
            foot is contact-labelled), not the global lowest geom. Fixes the two
            failure modes of the other modes at once: a single shift adds ZERO
            vertical wander (no bobbing), and keying off the feet keeps them on the
            floor even when hands/knees are the global-lowest point earlier in a
            get-up (plain `constant` grounds on those and floats the feet metres
            up). Falls back to `constant` if no foot-contact frames / sole sites.
  constant  — ONE shift for the whole clip: floor = a low percentile of per-frame
            lowest-Z (ANY geom); shift all frames by -floor. Preserves every bit of
            vertical motion, never adds jitter, but grounds on whatever is lowest —
            wrong reference during get-ups (see above).
  perframe  — shift each frame so its lowest point sits exactly at 0 (full
            plant, both up and down). Optional --smooth-shift to de-jitter. Plants
            the feet but the per-frame shift wanders (~7-9 cm on get-ups) = bobbing
            in a fixed world frame.

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

# Sole corner sites per foot effector — the contact reference for constant-contact
# grounding (mirrors the Stage-4 on-floor rows).
SOLE_CORNER_SITES = {
    "left_foot": [f"alex_left_sole_corner_{a}_body_{b}_site"
                  for a in ("toe", "heel") for b in ("left", "right")],
    "right_foot": [f"alex_right_sole_corner_{a}_body_{b}_site"
                   for a in ("toe", "heel") for b in ("left", "right")],
}


def _foot_plant_frames(model, data, qpos, contact_flags, eff_names,
                       fps=120.0, still_speed=0.05):
    """Per-foot per-frame sole data used by the floor registration AND the hybrid
    lift cap: min sole-corner Z (T,) and the STILL-plant mask (T,) per foot.

    A foot is a still plant when it is contact-labelled AND its body is moving
    slower than `still_speed` (m/s). This MUST match the Stage-4 solve's plant
    definition (`_compute_anchors` splits contact intervals into stationary
    sub-segments at the same speed and only *those* get the on-floor rows). The
    grounding previously keyed on raw `contact_flags`, which also include the
    MOVING approach/transition frames (e.g. a foot descending through a get-up, or
    a supine-phase touch during standSupine): those sit several cm off the true
    stance ground, so registering on them drags the shift and floats the actual
    stance.

    Returns {} if no foot contacts / sole sites resolve."""
    foot_sites = {}
    for eff, names in SOLE_CORNER_SITES.items():
        if eff not in eff_names:
            continue
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]
        ids = [i for i in ids if i >= 0]
        if ids:
            foot_sites[eff_names.index(eff)] = ids
    if not foot_sites or contact_flags is None:
        return {}

    # Per-foot body speed (m/s) from the sole sites' shared body position.
    T = qpos.shape[0]
    body_pos = {c: np.zeros((T, 3)) for c in foot_sites}
    min_z = {c: np.zeros(T) for c in foot_sites}
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for c, ids in foot_sites.items():
            body_pos[c][t] = data.xpos[int(model.site_bodyid[ids[0]])]
            min_z[c][t] = min(float(data.site_xpos[s][2]) for s in ids)
    out = {}
    for c, p in body_pos.items():
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
        v[0] = v[1] if T > 1 else 0.0
        out[c] = {"min_z": min_z[c],
                  "labelled": np.asarray(contact_flags[:, c], dtype=bool),
                  "still": np.asarray(contact_flags[:, c], dtype=bool) & (v < still_speed)}
    return out


def _planted_foot_sole_samples(plant_data):
    """Flat array of the heights the planted feet actually rest at, from
    `_foot_plant_frames` output. Per foot: falls back to all contact-labelled
    frames if it has no still frames (never drop a foot's samples entirely)."""
    samples = []
    for d in plant_data.values():
        use = d["still"] if d["still"].any() else d["labelled"]
        samples.extend(d["min_z"][use].tolist())
    return np.asarray(samples)


def _solve_lift_qp(need, cap, smooth):
    """min ||x - need||^2 + smooth*||D2 x||^2  s.t.  0 <= x <= cap.

    Banded QP over the per-frame lift (metres). `need` = whole-body penetration
    depth after the base shift; `cap` = how far the frame may be lifted before a
    still-planted foot floats (inf when no still plant)."""
    import osqp
    import scipy.sparse as sp

    n = len(need)
    if n < 3:
        return np.clip(need, 0.0, cap)
    d2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    P = sp.csc_matrix(2.0 * (sp.eye(n, format="csc") + smooth * (d2.T @ d2)))
    q = -2.0 * need
    A = sp.eye(n, format="csc")
    prob = osqp.OSQP()
    prob.setup(P, q, A, np.zeros(n), cap, verbose=False,
               eps_abs=1e-6, eps_rel=1e-6, max_iter=50000, polish=True)
    res = prob.solve()
    if res.info.status not in ("solved", "solved inaccurate", "solved_inaccurate"):
        raise RuntimeError(f"hybrid lift QP: OSQP status {res.info.status!r}")
    return np.clip(res.x, 0.0, cap)   # polish tolerance can leave 1e-8 residues


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
    ap.add_argument("--mode", choices=["hybrid", "constant-contact", "constant", "perframe"],
                    default="constant-contact")
    ap.add_argument("--percentile", type=float, default=1.0,
                    help="constant mode: floor = this percentile of per-frame lowest-Z. Low (1) so "
                         "the clip's lowest moment touches the floor.")
    ap.add_argument("--contact-percentile", type=float, default=50.0,
                    help="constant-contact mode: floor = this percentile of the PLANTED-FOOT sole "
                         "heights. Default 50 (median) keys on the stable stance, not the brief "
                         "touchdown transient (a heel-strike corner dips several cm — a low "
                         "percentile there would leave the whole standing phase floating).")
    ap.add_argument("--smooth-shift", type=float, default=0.0,
                    help="perframe mode: tridiagonal smoothing weight on the shift series (0 = off).")
    ap.add_argument("--lift-smooth", type=float, default=1e4,
                    help="hybrid mode: smoothness weight on the lift's second difference "
                         "(per-frame, 120 Hz). Higher = flatter lift, more residual penetration "
                         "near sharp need transitions.")
    ap.add_argument("--lift-float-tol", type=float, default=0.005,
                    help="hybrid mode: a still-planted foot may be lifted at most its own "
                         "penetration + this (m) — bounds plant float introduced by the lift.")
    ap.add_argument("--still-speed", type=float, default=0.05,
                    help="constant-contact: a contact-labelled foot only anchors the floor registration "
                         "on frames where its body moves slower than this (m/s) — the STILL-plant "
                         "definition, matching the Stage-4 solve. Excludes moving approach/transition "
                         "contacts (get-up descents, supine touches) that otherwise drag the shift and "
                         "float the true stance. (default: 0.05, = the solver's plant-speed)")
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

    floor_src = "lowest-geom"
    lift = None
    if args.mode in ("hybrid", "constant-contact"):
        eff_names = [str(x) for x in data_dict["contact_effector_names"]] \
            if "contact_effector_names" in data_dict else []
        cflags = np.asarray(data_dict["contact_flags"], dtype=bool) \
            if "contact_flags" in data_dict else None
        gfps = float(data_dict["fps"]) if "fps" in data_dict else 120.0
        plant_data = _foot_plant_frames(model, data, qpos, cflags, eff_names,
                                        fps=gfps, still_speed=args.still_speed)
        samples = _planted_foot_sole_samples(plant_data)
        if samples.size:
            floor = float(np.percentile(samples, args.contact_percentile))
            floor_src = f"planted-foot-sole p{args.contact_percentile:g} (n={samples.size})"
        else:
            print(f"  [warn] {args.mode}: no planted-foot samples — "
                  "falling back to global-lowest constant")
            floor = float(np.percentile(lowest, args.percentile))
        shift = np.full(N, -floor)

        if args.mode == "hybrid":
            need = np.maximum(0.0, -(lowest + shift))
            cap = np.full(N, np.inf)
            for d in plant_data.values():
                foot_after = d["min_z"] + shift
                foot_cap = np.maximum(0.0, -foot_after) + args.lift_float_tol
                cap = np.where(d["still"], np.minimum(cap, foot_cap), cap)
            lift = _solve_lift_qp(need, cap, args.lift_smooth)
            shift = shift + lift
    elif args.mode == "constant":
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
    if lift is not None:
        data_dict["ground_lift"] = lift

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data_dict)

    print(f"[ground] {args.npz.name}  N={N}  mode={args.mode}")
    print(f"  lowest-Z before: min={lowest.min():+.4f} med={np.median(lowest):+.4f} max={lowest.max():+.4f}")
    if args.mode == "hybrid":
        gfps_v = float(data_dict["fps"]) if "fps" in data_dict else 120.0
        peak_v = float(np.max(np.abs(np.diff(lift)))) * gfps_v if N > 1 else 0.0
        print(f"  floor({floor_src})={floor:+.4f}  base shift={-floor:+.4f} m")
        print(f"  lift: max={lift.max():+.4f} m  frames>0: {(lift > 1e-4).mean()*100:.1f}%  "
              f"peak dz/dt={peak_v:.3f} m/s  (smooth={args.lift_smooth:g} float_tol={args.lift_float_tol:g})")
    elif args.mode in ("constant", "constant-contact"):
        print(f"  floor({floor_src})={floor:+.4f}  constant shift={shift[0]:+.4f} m")
    else:
        print(f"  perframe shift: min={shift.min():+.4f} max={shift.max():+.4f} smooth={args.smooth_shift}")
    print(f"  lowest-Z after : min={grounded_lowest.min():+.4f} med={np.median(grounded_lowest):+.4f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

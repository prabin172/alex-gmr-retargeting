#!/usr/bin/env python3
"""S2-T4/S2-T6: Stage A (+ optional grounding) on our OWN G1 Stage-3 output
(polish(OURS), the bottom-right cell of the 2x2 in GMR-baseline.md SS7.4).

**Grounding is OFF by default now (Prabin, 2026-07-16) -- found actively
harmful for OURS specifically.** After the S2-T6 floor-referenced-rest-anchor
fix, Stage 3's own pull-to-floor mechanism already lands held (contact) frames
within ~1cm of the true floor in the RAW output (validated across all 4
tested clips: median support_z -1.1 to +1.5cm, 53-81% of held frames within
3cm -- genuinely BEATS GMR-polished's own 4.6-11.7cm on this metric, the one
GMR structurally cannot achieve at all). `ground_qpos`'s `constant` mode
computes ONE percentile-based Z-shift from the WHOLE clip's blind minimum
(some unrelated non-contact frame, e.g. a normal swing-phase foot-clearance
dip) and applies it UNIFORMLY to every frame -- it has no concept of "this
frame is already correctly floor-referenced, don't move it." Confirmed
directly: applying it after Stage 3's pull-to-floor DRAGS the already-good
held frames 12-19cm away from the floor (verified before/after on
`walk1_subject1`). Stage A alone (pure temporal smoothing, no absolute Z-shift)
does NOT have this problem -- held-frame quality survives it unchanged.
**Trade-off, stated honestly**: skipping grounding leaves whole-clip aggregate
floorPen/self-collision WORSE than GMR-polished (15-24cm vs 1-5cm, 15-21% vs
0-5%) on the tested clips -- real residual retargeting/morphology error on
NON-contact frames that grounding used to paper over at the cost of breaking
contact quality. `--ground` re-enables the old (harmful) behavior for A/B
comparison only; do not ship it as the default.

**S2-T9 fix (2026-07-17): Stage A itself was found to REGRESS floorPen/
self-collision on 3/4 gmrscale clips** (worst: fallAndGetUp2_subject2 39.7cm
raw -> 52.3cm post-StageA, coll 14.3%->22.7%). Root-caused as the SAME
floor-blind-smoothing-overshoot mechanism `_detect_floor_sensitive_frames`'s
docstring already documents for the Alex pipeline (measured there on
luigi_standProne_03: a sharp Stage-3-fixed violation re-inflated 2.4cm->13.9cm)
-- confirmed here directly: `fallAndGetUp2_subject2`'s raw signal around its
worst frame is jagged/narrow (0.5cm right next to 39.7cm, frame-to-frame), and
Stage A's floor-blind tridiagonal smoother blends across that transition,
producing a NEW peak (52.3cm) deeper than either raw neighbour -- classic
smoothing-overshoot on a sharp signal, not a new bug. Fixed by porting the
mainline pipeline's `lambda_track_frames` local-boost mechanism (locally
raise the tracking weight at floor- and self-collision-sensitive frames so
Stage A trusts the raw signal there instead of blending it toward smoother
but wronger neighbours), extended here to ALSO cover self-collision (the
mainline version only protects against floor violations; OURS's own
self-collision regression needed the same treatment, per Prabin's ask).
Default ON (`--no-sensitivity-boost` to disable for A/B).

Usage:
    conda run -n gmr python scripts/g1/polish_ours_g1.py \\
        --in outputs/gmr_baseline/sprint/ours_g1/walk1_subject1_ours.npz \\
        --out outputs/gmr_baseline/sprint/ours_g1/walk1_subject1_ours_polished.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # NOT repo root -- see planLogGMR.md T1
from polish_gmr_pkl import ground_qpos  # noqa: E402
from solve_global_trajectory_opt_contactfirst import (  # noqa: E402
    COLL_HOPS, N_ACT, _within_k_hops, stage_a)
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402

G1_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


def _sustained_ramp_weight(viol, min_run=8, pad=5):
    """(T,) bool -> (T,) float [0,1], sustained (>=min_run) violation runs only,
    cosine-ramped to 0 over `pad` frames at each boundary. Same algorithm as
    the mainline pipeline's `_detect_floor_sensitive_frames` (see its
    docstring for why: not proximity, not a hard mask, sustained-only)."""
    T = len(viol)
    weight = np.zeros(T, dtype=np.float64)
    runs = []
    k = 0
    while k < T:
        if not viol[k]:
            k += 1
            continue
        j = k
        while j < T and viol[j]:
            j += 1
        if (j - k) >= min_run:
            runs.append((k, j))
        k = j
    if not runs:
        return weight
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, pad + 2))[1:-1]
    for k, j in runs:
        run_weight = np.zeros(T, dtype=np.float64)
        run_weight[k:j] = 1.0
        lo = max(0, k - pad)
        run_weight[lo:k] = ramp[-(k - lo):] if k > lo else ramp[:0]
        hi = min(T, j + pad)
        run_weight[j:hi] = ramp[::-1][:hi - j]
        weight = np.maximum(weight, run_weight)
    return weight


def _sensitivity_weight(qpos, min_floor_pen=0.015, min_coll_pen=0.005,
                        min_run=8, pad=5):
    """Per-frame [0,1] weight: sustained floor penetration OR sustained
    self-collision, on the SAME vetted G1 model + mesh-exact metrics used
    throughout this project's floorPen/coll% reporting (not MuJoCo's native
    narrowphase against the floor plane, which would disagree with the
    mesh-exact `_robot_lowest_z` this codebase treats as ground truth)."""
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    T = qpos.shape[0]
    floor_pen = np.zeros(T)
    coll_pen = np.zeros(T)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        floor_pen[t] = max(0.0, -_robot_lowest_z(model, data, mesh_cache, geom_ids))
        mx = 0.0
        for c in range(data.ncon):
            ct = data.contact[c]
            if ct.geom1 == floor_gid or ct.geom2 == floor_gid:
                continue
            b1 = int(model.geom_bodyid[ct.geom1]); b2 = int(model.geom_bodyid[ct.geom2])
            if b1 == 0 or b2 == 0 or _within_k_hops(model, b1, b2, COLL_HOPS):
                continue
            if ct.dist < 0:
                mx = max(mx, abs(float(ct.dist)))
        coll_pen[t] = mx
    floor_w = _sustained_ramp_weight(floor_pen > min_floor_pen, min_run, pad)
    coll_w = _sustained_ramp_weight(coll_pen > min_coll_pen, min_run, pad)
    return np.maximum(floor_w, coll_w), floor_w, coll_w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--model", type=Path, default=G1_MODEL_DEFAULT)
    ap.add_argument("--lambda-track", type=float, default=1.0)
    ap.add_argument("--lambda-smooth", type=float, default=20.0, help="30Hz-scaled, per W2-T7.")
    ap.add_argument("--ground-percentile", type=float, default=1.0)
    ap.add_argument("--ground", action="store_true",
                    help="Re-enable the whole-clip constant-percentile grounding shift. OFF by "
                         "default -- confirmed to drag already-correct contact-held frames away "
                         "from the floor (S2-T6). Only for A/B comparison, not the shipped path.")
    ap.add_argument("--no-sensitivity-boost", action="store_true",
                    help="Disable the S2-T9 floor/self-collision-sensitive local tracking-weight "
                         "boost (see module docstring). Default ON -- only for A/B comparison.")
    args = ap.parse_args()

    d = np.load(args.in_path)
    qpos, fps = d["qpos"], float(d["fps"])
    n_act = qpos.shape[1] - 7
    assert n_act == N_ACT, f"expected {N_ACT} actuated joints, got {n_act}"

    model = mujoco.MjModel.from_xml_path(str(args.model))
    q_lo = np.full(n_act, -1e6)
    q_hi = np.full(n_act, 1e6)
    act = 0
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == 0:
            continue
        if bool(model.jnt_limited[j]):
            q_lo[act] = model.jnt_range[j, 0]
            q_hi[act] = model.jnt_range[j, 1]
        act += 1

    lambda_track_frames = None
    if not args.no_sensitivity_boost:
        sens_w, floor_w, coll_w = _sensitivity_weight(qpos)
        if sens_w.any():
            boost = max(args.lambda_track, args.lambda_smooth * 2.0)
            per_frame = args.lambda_track + sens_w * (boost - args.lambda_track)
            lambda_track_frames = np.tile(per_frame[:, None], (1, N_ACT))
            print(f"  sensitivity boost: floor-sensitive {int((floor_w > 0.99).sum())}/{qpos.shape[0]} "
                  f"frames, self-coll-sensitive {int((coll_w > 0.99).sum())}/{qpos.shape[0]} frames "
                  f"(boosted lambda_track={boost:.0f})")
        else:
            print("  sensitivity boost: no sustained floor/self-collision violation found, no-op")

    qpos_out = stage_a(qpos, args.lambda_track, args.lambda_smooth, q_lo, q_hi, smooth_root=True,
                       lambda_track_frames=lambda_track_frames)
    tag = "stageA"
    if args.ground:
        qpos_out = ground_qpos(qpos_out, args.model, "constant", args.ground_percentile, 0.0)
        tag = "stageA+ground"

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_path, qpos=qpos_out, fps=np.float64(fps))
    print(f"Wrote {args.out_path} ({tag}, {qpos_out.shape[0]} frames)")


if __name__ == "__main__":
    main()

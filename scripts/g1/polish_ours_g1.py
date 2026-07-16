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
from solve_global_trajectory_opt_contactfirst import N_ACT, stage_a  # noqa: E402

G1_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")


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

    qpos_out = stage_a(qpos, args.lambda_track, args.lambda_smooth, q_lo, q_hi, smooth_root=True)
    tag = "stageA"
    if args.ground:
        qpos_out = ground_qpos(qpos_out, args.model, "constant", args.ground_percentile, 0.0)
        tag = "stageA+ground"

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_path, qpos=qpos_out, fps=np.float64(fps))
    print(f"Wrote {args.out_path} ({tag}, {qpos_out.shape[0]} frames)")


if __name__ == "__main__":
    main()

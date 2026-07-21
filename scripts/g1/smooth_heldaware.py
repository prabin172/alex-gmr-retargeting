#!/usr/bin/env python3
"""S8-T2: held-aware temporal smoothing for perframelimb (and gmr_heightfix as
fairness arm). Attempt 1 mechanism:

- During frames where a foot is held (debounced contact + speed < 0.05 m/s),
  lock that leg chain's DOFs AND the root to their input (clamped) values.
- Smooth the free DOFs (arms, waist, root on non-held stretches).
- Lock is ramped in/out over RAMP_FRAMES=5 frames (cosine) so the lock boundary
  itself does not introduce a new velocity discontinuity.
- Applied to BOTH arms: `perframelimb_sm` (ours) and `gmr_heightfix_sm` (fairness).

Implementation: stage_a's lambda_track_frames API (T, N_ACT 2D array) with
  lambda_lock at held frames (ramp-weighted) to pin those DOFs while the
  banded tridiagonal solve still handles everything globally. Root uses the
  per-frame MAX of the 2D weight matrix (existing stage_a logic).

Do NOT reuse smooth_then_clamp.py's blind design. Do NOT modify polish_gmr_pkl.py.

Usage:
    # perframelimb arm
    conda run -n gmr python scripts/g1/smooth_heldaware.py \\
        --in outputs/gmr_baseline/sprint/pkl_s5/walk1_subject1_perframelimb.pkl \\
        --canonical outputs/gmr_baseline/sprint/canonical_human_s5/walk1_subject1_lafan1c_grounded.npz \\
        --out outputs/gmr_baseline/sprint/pkl_s5/walk1_subject1_perframelimb_sm.pkl

    # heightfix arm (fairness)
    conda run -n gmr python scripts/g1/smooth_heldaware.py \\
        --in outputs/gmr_baseline/sprint/pkl/walk1_subject1_gmrfix.pkl \\
        --canonical outputs/gmr_baseline/sprint/canonical_human_s5/walk1_subject1_lafan1c_grounded.npz \\
        --out outputs/gmr_baseline/sprint/pkl_s5/walk1_subject1_heightfix_sm.pkl
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from gmr_contact_retarget import compute_held_masks, FEET  # noqa: E402
from solve_global_trajectory_opt_contactfirst import stage_a, N_ACT  # noqa: E402

# Left-foot leg chain DOF indices in qpos[7:] (qpos indices 7-12, dof indices 0-5)
LEFT_CHAIN_DOFS = [0, 1, 2, 3, 4, 5]   # left_hip_{pitch,roll,yaw}, left_knee, left_ankle_{pitch,roll}
RIGHT_CHAIN_DOFS = [6, 7, 8, 9, 10, 11]  # right_hip_{pitch,roll,yaw}, right_knee, right_ankle_{pitch,roll}

RAMP_FRAMES = 5
LAMBDA_TRACK_DEFAULT = 1.0
LAMBDA_SMOOTH_DEFAULT = 20.0
LAMBDA_LOCK = 1e8  # effectively pins held DOFs to input
SPIKE_UNLOCK_THRESH = 40.0  # rad/s; below the T2 gate's 1.2x-raw vMax ceiling


def _cosramp(age, ramp):
    if ramp <= 0:
        return 1.0
    age = max(0, min(age, ramp))
    return 0.5 * (1.0 - np.cos(np.pi * age / ramp))


def build_lock_weights(held, T, ramp_frames, lambda_track, lambda_lock,
                        qpos=None, fps=None, spike_thresh=SPIKE_UNLOCK_THRESH):
    """Build per-frame per-DOF lambda_track_frames (T, N_ACT) array.

    Held-leg DOFs (and root, via stage_a's max() logic) ramp from lambda_track
    to lambda_lock over ramp_frames when a foot enters held, ramp back to
    lambda_track when it exits. Free DOFs (waist, arms) stay at lambda_track.

    T2 attempt-2 fix (diagnosed 2026-07-18): the ramp-out tail after a hold
    ends can itself span a release-transition frame where perframelimb's
    per-frame IK has a real one-frame branch-flip artifact (e.g. hip_yaw
    jumping ~2.7rad in a single frame, surrounded on both sides by ordinary
    values). Attempt-1's lock (near-full weight during the ramp) faithfully
    preserves that artifact instead of letting it get smoothed -- confirmed
    via obstacles4_subject3 frame 2116/2117 (bit-identical raw vs smoothed,
    lock weight ~9e7 there despite held=False). Fix: if the RAW input already
    has a velocity spike on a given joint at a given frame transition, drop
    that joint's lock back to lambda_track for both endpoints of the
    transition so the tridiagonal solve can interpolate through it, while
    every other locked frame/joint is untouched.

    Returns: (T, N_ACT) float array.
    """
    ltrack = np.full((T, N_ACT), lambda_track, dtype=np.float64)

    for eff, dof_idx in [("left_foot", LEFT_CHAIN_DOFS), ("right_foot", RIGHT_CHAIN_DOFS)]:
        held_arr = held[eff]
        held_prev = False
        ramp_age = 0
        for t in range(T):
            is_held = bool(held_arr[t])
            if is_held and not held_prev:
                ramp_age = 0
            held_prev = is_held
            ramp_age = min(ramp_frames, ramp_age + 1) if is_held else max(0, ramp_age - 1)
            frac = _cosramp(ramp_age, ramp_frames)
            w = lambda_track + (lambda_lock - lambda_track) * frac
            for d in dof_idx:
                ltrack[t, d] = max(ltrack[t, d], w)

    if qpos is not None and fps is not None:
        qj = qpos[:, 7:]
        vel = np.abs(np.diff(qj, axis=0)) * fps  # (T-1, N_ACT)
        for eff, dof_idx in [("left_foot", LEFT_CHAIN_DOFS), ("right_foot", RIGHT_CHAIN_DOFS)]:
            for d in dof_idx:
                for t in np.where(vel[:, d] > spike_thresh)[0]:
                    ltrack[t, d] = lambda_track
                    ltrack[t + 1, d] = lambda_track

    return ltrack


def smooth_heldaware(qpos, held, lambda_track=LAMBDA_TRACK_DEFAULT,
                     lambda_smooth=LAMBDA_SMOOTH_DEFAULT, ramp_frames=RAMP_FRAMES,
                     lambda_lock_override=None, fps=None,
                     spike_thresh=SPIKE_UNLOCK_THRESH):
    """Apply held-aware tridiagonal smoothing.

    - Free DOFs (arms, waist, root on non-held stretches): smoothed normally.
    - Held leg-chain DOFs + root: locked to input via very high tracking weight.
    - Exception: a joint/frame where the raw input already has a velocity
      spike (> spike_thresh) is unlocked for that transition even if it falls
      inside a held/ramp window, so smoothing can remove it (see
      build_lock_weights docstring). Pass fps=None to disable (attempt-1
      behavior).

    Args:
        qpos: (T, 36) full qpos array (root 0:7, joints 7:36).
        held: dict with 'left_foot' and 'right_foot' bool arrays of length T.
        lambda_track: base tracking weight for free DOFs.
        lambda_smooth: smoothing strength (same for all DOFs).
        ramp_frames: cosine ramp duration at hold boundaries.
        fps: clip frame rate, needed for the spike-unlock exception.

    Returns:
        smoothed qpos (T, 36), copy (input not mutated).
    """
    T = qpos.shape[0]
    assert qpos.shape[1] == 36, f"expected 36 qpos, got {qpos.shape[1]}"
    assert N_ACT == 29, f"N_ACT mismatch: {N_ACT}"

    # Joint limits from the vetted model (not GMR's xml -- our model has 36 DOF qpos)
    from g1_model_setup import load_g1_model_with_vetted_collision_and_floor
    model, _, _, _ = load_g1_model_with_vetted_collision_and_floor()
    q_lo = model.jnt_range[1:, 0].copy()   # joints 1..njnt-1 in qpos order
    q_hi = model.jnt_range[1:, 1].copy()

    lock = lambda_lock_override if lambda_lock_override is not None else LAMBDA_LOCK
    ltrack_frames = build_lock_weights(held, T, ramp_frames, lambda_track, lambda_lock=lock,
                                        qpos=qpos, fps=fps, spike_thresh=spike_thresh)

    out = stage_a(qpos, lambda_track=lambda_track, lambda_smooth=lambda_smooth,
                  q_lo=q_lo, q_hi=q_hi, smooth_root=True,
                  lambda_track_frames=ltrack_frames)
    return out


def save_pkl(path, qpos, fps):
    root_pos = qpos[:, 0:3]
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]   # wxyz -> xyzw for pkl storage
    dof_pos = qpos[:, 7:]
    data = {"fps": fps, "root_pos": root_pos, "root_rot": root_rot_xyzw,
            "dof_pos": dof_pos, "local_body_pos": None, "link_body_list": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=pathlib.Path)
    ap.add_argument("--canonical", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--lambda-track", type=float, default=LAMBDA_TRACK_DEFAULT)
    ap.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH_DEFAULT)
    ap.add_argument("--lambda-lock", type=float, default=LAMBDA_LOCK)
    ap.add_argument("--ramp-frames", type=int, default=RAMP_FRAMES)
    args = ap.parse_args()

    qpos, fps = load_gmr_pkl(args.in_path)
    held, T = compute_held_masks(args.canonical, FEET)
    assert T == qpos.shape[0], f"canonical T={T} != qpos T={qpos.shape[0]}"

    # Report held-frame fraction
    lf = held["left_foot"].sum()
    rf = held["right_foot"].sum()
    either = (held["left_foot"] | held["right_foot"]).sum()
    print(f"[smooth_heldaware] T={T}  held L={lf} R={rf} either={either} "
          f"({100*either/T:.1f}%)")
    print(f"[smooth_heldaware] lambda_track={args.lambda_track}  "
          f"lambda_smooth={args.lambda_smooth}  "
          f"lambda_lock={args.lambda_lock}  ramp={args.ramp_frames}")

    out = smooth_heldaware(qpos, held,
                           lambda_track=args.lambda_track,
                           lambda_smooth=args.lambda_smooth,
                           ramp_frames=args.ramp_frames,
                           lambda_lock_override=args.lambda_lock,
                           fps=fps)
    save_pkl(args.out, out, fps)
    print(f"Saved {out.shape[0]} frames -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""S8-T2d: local spike repair on perframelimb, NO global smoothing.

Fallback mechanism per GMR-S8-plan.md REVISION R2.4, tried after T2c (smooth
-> re-clamp) hit its 2-attempt cap close but not clean (spikes/vMax fail,
concentrated on perframelimb's own worst clips).

Detect leg-DOF frame transitions where the RAW `perframelimb` input already
has a velocity spike (>SPIKE_REPAIR_THRESH rad/s, deliberately BELOW the 60
rad/s metric spike threshold so the repair set is a strict superset of what
gets counted -- not metric-targeted). Merge overlapping +/-3-frame windows
per joint, PCHIP-interpolate through each window using untouched context
points just outside it, then re-run the SAME limb clamp that built
perframelimb in the first place (`polish_median_limbwise._limbwise_pass`,
feet only, max_dq=0.15, avoid_self_collision=True) over the whole clip.

Why re-running that pass over the WHOLE clip is safe here (unlike T2c):
perframelimb's own build already used this exact call with max_dq=0.15
(`--center perframe` always sets `limbwise_max_dq=0.15`, see
polish_median_limbwise.py). Every frame we don't touch is therefore ALREADY
a fixed point of this operation -- re-running it is a near no-op except at
the repaired windows and their immediate downstream (proximal-first chain
order). T2c's re-clamp diverged because ITS input (a globally tridiagonal-
smoothed trajectory) was not a fixed point anywhere, so a full re-clamp made
large corrections everywhere and some hit joint limits (walk2_subject3
t=6318: left_hip_yaw bounced 1.57->1.57->-1.57->1.57, pi rad/frame = 94.2
rad/s @ 30fps -- the exact "solution-branch flip" class documented in
leg_floor_clamp.py's CorrectionRateLimiter docstring).

Re-detects after each repair pass; up to MAX_ITERS total; reports what
remains uninterpolatable (only possible at a clip boundary, where there
isn't enough untouched context on one side).
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from smooth_heldaware import LEFT_CHAIN_DOFS, RIGHT_CHAIN_DOFS  # noqa: E402
from polish_median_limbwise import _limbwise_pass, RAMP_FRAMES  # noqa: E402
from gmr_contact_retarget import FEET, HANDS, EFF_BODY  # noqa: E402
from leg_floor_clamp import build_chain_dofs, clamp_limb, CLAMP_TARGETS  # noqa: E402

SPIKE_REPAIR_THRESH = 40.0  # rad/s -- below the 60 rad/s metric spike threshold
WINDOW_HALF = 3
MAX_ITERS = 2
LEG_DOFS = LEFT_CHAIN_DOFS + RIGHT_CHAIN_DOFS


def detect_windows(qpos, fps, spike_thresh=SPIKE_REPAIR_THRESH, window_half=WINDOW_HALF):
    """Return {dof: [(lo, hi), ...]} merged, sorted, non-overlapping
    +/-window_half windows around every raw velocity-spike transition."""
    qj = qpos[:, 7:]
    vel = np.abs(np.diff(qj, axis=0)) * fps  # (T-1, N_ACT)
    T = qpos.shape[0]
    windows = {}
    for d in LEG_DOFS:
        spikes = np.where(vel[:, d] > spike_thresh)[0]
        if spikes.size == 0:
            continue
        raw = [(max(0, t - window_half), min(T - 1, t + 1 + window_half)) for t in spikes]
        raw.sort()
        merged = [raw[0]]
        for lo, hi in raw[1:]:
            if lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        windows[d] = merged
    return windows


def pchip_repair(qpos, windows):
    """Replace each (dof, window) span with a PCHIP interpolant built from
    untouched context points just outside the window (2 points each side
    where available). Returns a NEW array; input not mutated. Windows that
    can't get >=2 context points (clip boundary) are left untouched and
    reported by the caller via a second detect_windows pass."""
    out = qpos.copy()
    T = qpos.shape[0]
    for d, wins in windows.items():
        for lo, hi in wins:
            ctx = [t for t in (lo - 2, lo - 1, hi + 1, hi + 2) if 0 <= t < T and (t < lo or t > hi)]
            if len(ctx) < 2:
                continue
            xs = np.array(sorted(ctx))
            ys = out[xs, 7 + d]
            pchip = PchipInterpolator(xs, ys)
            t_range = np.arange(lo, hi + 1)
            out[t_range, 7 + d] = pchip(t_range)
    return out


def _limbwise_pass_frames(model, data, mesh_cache, qpos, held, effectors, frame_indices,
                           max_dq=0.15, avoid_self_collision=True):
    """T2d attempt 2: same per-frame clamp logic as
    `polish_median_limbwise._limbwise_pass`, but applied ONLY to the given
    `frame_indices`, each processed independently (no cross-frame ramp
    state). Every frame NOT in `frame_indices` is bit-identical to the input
    -- unlike attempt 1 (which re-ran the pass over the WHOLE clip and,
    despite perframelimb's own build already being that operation's fixed
    point almost everywhere, still perturbed several already-chaotic/near-
    singular frames onto a DIFFERENT unstable branch: walk2_subject3 spikes
    went 6 (original perframelimb) -> 18 (attempt 1), worse, not better).
    Restricting the touched set to exactly the repair windows means a
    frame outside every window literally cannot change, so it cannot
    acquire a new spike; the only frames that can still show a residual
    spike are at a window's own boundary, caught by the caller's re-detect."""
    chains = {eff: build_chain_dofs(model, eff) for eff in FEET + HANDS}
    out = qpos.copy()
    for t in sorted(frame_indices):
        data.qpos[:] = out[t]
        mujoco.mj_forward(model, data)
        for eff, watch_body in CLAMP_TARGETS:
            if watch_body == EFF_BODY.get(eff) and eff in effectors:
                continue
            clamp_limb(model, data, mesh_cache, eff, chains[eff],
                       floor_margin=0.0, watch_body=watch_body, max_dq=max_dq,
                       avoid_self_collision=avoid_self_collision)
        for eff in effectors:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EFF_BODY[eff])
            if bool(held[eff][t]):
                # No held-transition ramp here (short isolated window, not a
                # full-clip hold-onset/-release boundary) -- lock to the
                # current (post phase-1/2, pre-effector-pass) XY so a held
                # foot doesn't drift within the repair window.
                target_xy = data.xpos[bid][:2].copy()
                clamp_limb(model, data, mesh_cache, eff, chains[eff],
                           floor_margin=0.0, target_xy=target_xy, max_dq=max_dq,
                           avoid_self_collision=avoid_self_collision)
            else:
                clamp_limb(model, data, mesh_cache, eff, chains[eff],
                           floor_margin=0.0, watch_body=EFF_BODY[eff], max_dq=max_dq,
                           avoid_self_collision=avoid_self_collision)
        out[t] = data.qpos.copy()
    return out


def repair_clip(qpos, held, fps, model, data, mesh_cache, max_iters=MAX_ITERS, verbose=True,
                 window_only=True):
    """T2d repair loop: detect -> PCHIP -> re-clamp -> re-detect, up to
    max_iters. window_only=True (attempt 2, default): re-clamp restricted to
    the repair windows only. window_only=False (attempt 1, kept for the
    record): re-clamp the whole clip every iteration."""
    cur = qpos.copy()
    for it in range(1, max_iters + 1):
        windows = detect_windows(cur, fps)
        n_windows = sum(len(w) for w in windows.values())
        if n_windows == 0:
            if verbose:
                print(f"    iter {it}: no spikes >{SPIKE_REPAIR_THRESH} rad/s detected, stopping")
            break
        if verbose:
            print(f"    iter {it}: {n_windows} window(s) across {len(windows)} dof(s)")
        cur = pchip_repair(cur, windows)
        if window_only:
            touched = set()
            for wins in windows.values():
                for lo, hi in wins:
                    touched.update(range(lo, hi + 1))
            cur = _limbwise_pass_frames(model, data, mesh_cache, cur, held, FEET,
                                        sorted(touched), max_dq=0.15,
                                        avoid_self_collision=True)
        else:
            cur = _limbwise_pass(model, data, mesh_cache, cur, held, FEET,
                                  RAMP_FRAMES, max_dq=0.15,
                                  avoid_self_collision=True, rate_limit=None)
    else:
        windows = detect_windows(cur, fps)
        n_windows = sum(len(w) for w in windows.values())
        if n_windows > 0 and verbose:
            print(f"    WARNING: {n_windows} window(s) remain after {max_iters} iterations "
                  f"(likely clip-boundary windows with insufficient context)")
    return cur

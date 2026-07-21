#!/usr/bin/env python3
"""S6-B1: post-hoc median-centering + limb-wise floor/contact polish. Prabin's
idea (2026-07-17): center GMR's held-frame error distribution (median shift, so
float and penetration are balanced and every subsequent correction is small),
then run a cheap per-limb pass -- held effectors maintain contact, non-held
limbs maintain floor clearance.

Retargeter-agnostic: works on any qpos pkl (gmr_raw, gmr_heightfix, or Phase-A
gmr_contact_fc output). Reuses leg_floor_clamp.py's clamp_limb (S6-A2/A4) for the
limb-wise pass -- does NOT reimplement the DLS chain solve.

Why the old week-2 "contact-aware grounding" negative doesn't apply here: that
mechanism shifted the whole clip by a constant Z and stopped -- it converted
penetration into held-foot float with nothing to close the float back down.
Here, the limb-wise pass is precisely that missing closer.

Two centering modes:
  --center median   Prabin's original: single per-clip Z shift = -median(held-
                     frame support_z). Cannot fix floor-class TRUNK penetration
                     (limbs can't lift the pelvis) -- see --center perframe.
  --center perframe  Fable's amendment (B1b): per-frame offset = max(median
                     shift, lift needed so mesh-exact whole-body lowest point
                     >= 0), Savitzky-Golay-smoothed (or moving-average fallback)
                     to control jerk from the frame-varying lift.

Usage:
    conda run -n gmr python scripts/g1/polish_median_limbwise.py \\
        --in outputs/gmr_baseline/sprint/pkl/walk1_subject1.pkl \\
        --canonical outputs/gmr_baseline/sprint/canonical_human_s5/walk1_subject1_lafan1c_grounded.npz \\
        --out outputs/gmr_baseline/sprint/pkl_s5/walk1_subject1_medianlimb.pkl \\
        --center median
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402
from solve_lafan1_canonical_g1_contactfirst import ROLE_TO_G1_BODY, FOOT_POS_ROLE  # noqa: E402
from stage_b_g1 import support_z  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from gmr_contact_retarget import compute_held_masks, EFF_BODY, FEET, HANDS  # noqa: E402
from leg_floor_clamp import (build_chain_dofs, clamp_limb, CLAMP_TARGETS,  # noqa: E402
                             CorrectionRateLimiter, joint_ranges)

RAMP_FRAMES = 5


def _cosramp(age, ramp):
    if ramp <= 0:
        return 1.0
    age = max(0, min(age, ramp))
    return 0.5 * (1.0 - np.cos(np.pi * age / ramp))


def _median_shift(model, data, mesh_cache, role_bid, held, qpos):
    """Scalar per-clip Z shift = -median(held-frame support_z), over both feet
    combined. Returns 0.0 if no held frames (nothing to center)."""
    vals = []
    for eff, role in FOOT_POS_ROLE.items():
        idx = np.where(held[eff])[0]
        for t in idx:
            if t >= qpos.shape[0]:
                continue
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            vals.append(support_z(model, data, mesh_cache, role_bid[role]))
    return -float(np.median(vals)) if vals else 0.0


def _perframe_shift(model, data, mesh_cache, geom_ids, role_bid, held, qpos, window=15):
    """B1b: per-frame offset = max(median shift, lift needed so whole-body
    mesh-exact lowest point >= 0), smoothed with a moving-average window to
    control jerk (Savitzky-Golay would be a drop-in upgrade; moving-average is
    the simple, robust default -- revisit only if jerk gate fails)."""
    median = _median_shift(model, data, mesh_cache, role_bid, held, qpos)
    T = qpos.shape[0]
    raw = np.full(T, median)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest = _robot_lowest_z(model, data, mesh_cache, geom_ids)
        lift_needed = max(0.0, -lowest)
        raw[t] = max(median, lift_needed)
    if window > 1 and T > window:
        kernel = np.ones(window) / window
        pad = window // 2
        padded = np.pad(raw, (pad, pad), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[:T]
    else:
        smoothed = raw
    return smoothed


def _limbwise_pass(model, data, mesh_cache, qpos, held, effectors, ramp_frames,
                    max_dq=None, avoid_self_collision=False, rate_limit=None,
                    posture_continuity=False, posture_weight=1.0,
                    limit_margin=0.0, limit_weight=0.0,
                    posture_gate_lo=None, posture_gate_hi=None,
                    raw_gate_qpos=None):
    """Per frame: held effector -> locked-target DLS via clamp_limb(target_xy=...),
    ramped in/out (RAMP_FRAMES heritage, S5); everything else (swing effectors,
    knee/hip_yaw/elbow -- CLAMP_TARGETS, S6-A4) -> clearance-clamp only. Mutates
    qpos in place, frame by frame.

    floor_margin is always 0.0 here (NOT a per-effector z_support offset): both
    clamp_limb branches target `_lowest_point`'s Z directly (the actual sole/
    mesh-bottom height), which belongs at world Z=0 for a foot resting on the
    floor -- z_support (body-origin-to-sole offset) is a different quantity
    entirely and doesn't belong in floor_margin at all. An earlier version
    passed z_support as floor_margin here, producing a systematic +4cm float
    bias on every held frame (0% joint_ok despite 0% whole-body pen) -- caught
    via a support_z distribution check on walk1_subject1 (median +4.0/+4.3cm on
    both feet, suspiciously exact and uniform, not scatter -- a real bug, not
    noise).

    `max_dq` (S7-T3, opt-in): forwarded to every `clamp_limb` call here. Only
    `--center perframe` passes a value (0.15 rad, see leg_floor_clamp.py's
    docstring for the full root-cause writeup of why perframe specifically
    needs it); `--center median` and all other callers pass None (uncapped,
    byte-identical to pre-S7 behavior) -- the trust region is NOT a safe
    default, it regresses Phase A's already-shipped corpus numbers when
    applied globally (tested and reverted, see leg_floor_clamp.py).

    `avoid_self_collision` (S7-T7, opt-in): forwarded to every `clamp_limb`
    call here -- same self-collision repulsion term Phase A uses (see
    leg_floor_clamp.py's clamp_limb docstring), reusing the SAME shared
    mechanism rather than a second implementation.

    `rate_limit` (S8-T1b, opt-in, rad/frame): temporal trust region on the
    total per-frame applied correction -- see leg_floor_clamp.py's
    CorrectionRateLimiter docstring. Applied AFTER all of a frame's clamps,
    against that frame's pre-clamp (centered) pose; result clipped back into
    joint range. None (default) = off, byte-identical to pre-S8 behavior.

    `posture_continuity` / `posture_weight` (dev probe, opt-in): forwards
    `q_prev_chain`/`posture_weight` to every phase-1 `clamp_limb` call --
    see leg_floor_clamp.py's clamp_limb docstring. `q_prev_chain` is this
    chain's slice of the PREVIOUS frame's own post-clamp qpos (None at t=0,
    same convention as CorrectionRateLimiter's first-frame handling).
    Default False = byte-identical to pre-existing behavior (q_prev_chain
    never passed).

    `limit_margin` / `limit_weight` (S9-T1, opt-in): forwarded verbatim to
    every phase-1 `clamp_limb` call -- see leg_floor_clamp.py's clamp_limb
    docstring. Default `limit_weight=0.0` = byte-identical no-op.

    `posture_gate_lo` / `posture_gate_hi` (S9-T0-gate, opt-in, rad) /
    `raw_gate_qpos`: the S9-T0/T1 dev-clip run (planLogGMR.md) found blanket
    `posture_weight=1.0` -- applied every frame regardless of what the human
    motion is actually doing -- regresses 4/5 of the S8-T0b dev clips
    (worst_float up to +70% on walk3_subject1) even though it fixes the ONE
    outlier clip (`sprint1_subject4`) it was diagnosed on. The branch-flip
    failure mode this bias targets can only happen when the chain's TRUE,
    UNTOUCHED GMR raw target is itself near-static frame to frame -- a
    genuinely moving target has an unambiguous primary-task solution, no
    leftover null-space freedom to flip. Gate `posture_weight` per chain per
    frame by that chain's own raw joint-space frame-to-frame delta in
    `raw_gate_qpos` (caller-supplied, full qpos array shape-matched to
    `qpos`, MUST be the pristine pre-Phase-A-clamp/pre-smoothing GMR output
    -- NOT this function's own `qpos` argument, which by the time this
    pass runs already carries any upstream branch-flip artifact and would
    gate backwards, treating the artifact itself as "genuine motion" and
    disabling the fix exactly where it's needed; confirmed by direct
    measurement, see `sprint_s9_t0gate_probe.py`'s module docstring) --
    full weight when the raw target is near-static (delta <=
    `posture_gate_lo`), zero when it's genuinely moving (delta >=
    `posture_gate_hi`), linear ramp between. `posture_gate_lo`/`hi` both
    `None` (default) = byte-identical to blanket `posture_weight` (pre-gate
    behavior, unchanged); gating on requires `raw_gate_qpos`."""
    chains = {eff: build_chain_dofs(model, eff) for eff in FEET + HANDS}
    if posture_gate_lo is not None or posture_gate_hi is not None:
        assert posture_gate_lo is not None and posture_gate_hi is not None, \
            "posture_gate_lo/hi must both be set or both be None"
        assert raw_gate_qpos is not None, \
            "posture_gate_lo/hi requires raw_gate_qpos (the pristine GMR raw signal)"
        assert raw_gate_qpos.shape[0] == qpos.shape[0], \
            f"raw_gate_qpos T={raw_gate_qpos.shape[0]} != qpos T={qpos.shape[0]}"

    def _gated_weight(eff, t):
        if posture_gate_lo is None or posture_gate_hi is None:
            return posture_weight
        if t == 0:
            return posture_weight
        qadr = chains[eff][0]
        delta = float(np.max(np.abs(raw_gate_qpos[t, qadr] - raw_gate_qpos[t - 1, qadr])))
        if delta <= posture_gate_lo:
            return posture_weight
        if delta >= posture_gate_hi:
            return 0.0
        frac = (posture_gate_hi - delta) / (posture_gate_hi - posture_gate_lo)
        return posture_weight * frac

    onset_xy = {eff: None for eff in effectors}
    held_prev = {eff: False for eff in effectors}
    ramp_age = {eff: 0 for eff in effectors}
    limiter = CorrectionRateLimiter(rate_limit) if rate_limit is not None else None
    if limiter is not None:
        rl_lo, rl_hi = joint_ranges(model)
    prev_full_qpos = None

    T = qpos.shape[0]
    for t in range(T):
        data.qpos[:] = qpos[t]
        q_ref_joints = qpos[t, 7:].copy() if limiter is not None else None
        mujoco.mj_forward(model, data)

        # Onset/ramp bookkeeping first (read-only wrt qpos -- uses the frame's
        # natural, pre-correction pose, same as S5's onset-lock convention).
        frac = {}
        for eff in effectors:
            is_held = bool(held[eff][t])
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EFF_BODY[eff])
            if is_held and not held_prev[eff]:
                onset_xy[eff] = data.xpos[bid][:2].copy()
            held_prev[eff] = is_held
            ramp_age[eff] = min(ramp_frames, ramp_age[eff] + 1) if is_held \
                else max(0, ramp_age[eff] - 1)
            frac[eff] = _cosramp(ramp_age[eff], ramp_frames)

        # Proximal-to-distal order matters (S6-A4 finding, see CLAMP_TARGETS
        # comment in leg_floor_clamp.py): correct hip_yaw/knee/elbow FIRST,
        # since those DOFs also move the downstream ankle/wrist -- doing it the
        # other way round lets a proximal correction silently re-violate an
        # already-corrected distal effector.
        # S8-T1b attempt 2: when rate-limiting, run phase 1 (floor/held) WITHOUT
        # inline self-collision -- the limiter would otherwise cap the phase-2
        # collision corrections away (attempt 1 measured coll_pct regressing to
        # gmr_raw's level). Self-collision runs as an UN-limited post-pass after
        # the limiter instead (T0b: phase 2 is 1%/0% of spikes).
        inline_avoid_coll = avoid_self_collision and limiter is None

        def _qprev(eff):
            return prev_full_qpos[chains[eff][0]] \
                if (posture_continuity and prev_full_qpos is not None) else None

        for eff, watch_body in CLAMP_TARGETS:
            if watch_body == EFF_BODY.get(eff) and eff in effectors:
                continue  # the effector's own (most distal) body is handled below
            clamp_limb(model, data, mesh_cache, eff, chains[eff],
                       floor_margin=0.0, watch_body=watch_body, max_dq=max_dq,
                       avoid_self_collision=inline_avoid_coll,
                       q_prev_chain=_qprev(eff), posture_weight=_gated_weight(eff, t),
                       limit_margin=limit_margin, limit_weight=limit_weight)

        # Effector-level (most distal) pass, run last: held -> locked target,
        # swing -> clearance only.
        for eff in effectors:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EFF_BODY[eff])
            if frac[eff] <= 0.0:
                clamp_limb(model, data, mesh_cache, eff, chains[eff],
                           floor_margin=0.0, watch_body=EFF_BODY[eff], max_dq=max_dq,
                           avoid_self_collision=inline_avoid_coll,
                           q_prev_chain=_qprev(eff), posture_weight=_gated_weight(eff, t),
                           limit_margin=limit_margin, limit_weight=limit_weight)
                continue
            # Blend target XY from the pre-ramp current position toward the
            # onset-locked position as frac climbs 0->1 -- smooth engagement
            # instead of an instantaneous jump (mirrors S5's cost-ramp intent,
            # but as a position blend since clamp_limb is a direct DLS solve,
            # not a weighted QP cost).
            cur_xy = data.xpos[bid][:2]
            target_xy = onset_xy[eff] * frac[eff] + cur_xy * (1.0 - frac[eff]) \
                if onset_xy[eff] is not None else cur_xy
            clamp_limb(model, data, mesh_cache, eff, chains[eff],
                       floor_margin=0.0, target_xy=target_xy, max_dq=max_dq,
                       avoid_self_collision=inline_avoid_coll,
                       q_prev_chain=_qprev(eff), posture_weight=_gated_weight(eff, t),
                       limit_margin=limit_margin, limit_weight=limit_weight)

        if limiter is not None:
            limited = limiter.apply(q_ref_joints, data.qpos[7:])
            data.qpos[7:] = np.clip(limited, rl_lo, rl_hi)
            mujoco.mj_forward(model, data)
            if avoid_self_collision:
                # Un-limited self-collision post-pass, one collision-only call
                # per chain (S8-T1b attempt 2 -- see leg_floor_clamp.clamp_limb's
                # collision_only docstring).
                for eff in FEET + HANDS:
                    clamp_limb(model, data, mesh_cache, eff, chains[eff],
                               max_dq=max_dq, avoid_self_collision=True,
                               collision_only=True)

        qpos[t] = data.qpos.copy()
        prev_full_qpos = qpos[t]
    return qpos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=str)
    ap.add_argument("--canonical", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--center", choices=["median", "perframe", "none"], default="median",
                     help="'none' (S8-T2c): skip centering entirely -- for re-clamping "
                          "an input whose root is already correct (e.g. a held-aware "
                          "smoothed pkl), so the root lift is not applied twice.")
    ap.add_argument("--effectors", choices=["feet", "feet+hands"], default="feet")
    ap.add_argument("--ramp-frames", type=int, default=RAMP_FRAMES)
    ap.add_argument("--smooth-window", type=int, default=15,
                     help="perframe centering's moving-average window (frames).")
    ap.add_argument("--avoid-self-collision", action="store_true", default=False,
                     help="S7-T7: self-collision repulsion term in every clamp_limb "
                          "call (see leg_floor_clamp.py's clamp_limb docstring).")
    ap.add_argument("--clamp-rate-limit", type=float, default=None,
                     help="S8-T1b: temporal trust region (rad/frame) on the total "
                          "per-frame applied correction (see leg_floor_clamp.py's "
                          "CorrectionRateLimiter). None (default) = off.")
    args = ap.parse_args()

    effectors = FEET if args.effectors == "feet" else FEET + HANDS

    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}

    qpos, fps = load_gmr_pkl(args.in_path)
    held, T = compute_held_masks(args.canonical, effectors)
    assert T == qpos.shape[0], f"canonical T={T} != qpos T={qpos.shape[0]}"

    # Step 1: centering.
    limbwise_max_dq = None
    if args.center == "none":
        print("[polish_median_limbwise] center=none -- skipping root-lift, "
              "re-clamping input qpos as-is")
    elif args.center == "median":
        shift = _median_shift(model, data, mesh_cache, role_bid, held, qpos)
        print(f"[polish_median_limbwise] median shift = {shift*100:.2f}cm (constant)")
        qpos[:, 2] += shift
    else:
        shift = _perframe_shift(model, data, mesh_cache, geom_ids, role_bid, held, qpos,
                                 window=args.smooth_window)
        print(f"[polish_median_limbwise] perframe shift: mean={shift.mean()*100:.2f}cm "
              f"max={shift.max()*100:.2f}cm")
        qpos[:, 2] += shift
        # S7-T3: perframe (unlike median) can land the leg chain in a near-
        # singular full-extension basin (knee pinned at its lower joint
        # limit) where clamp_limb's uncapped DLS diverges within a single
        # frame's iteration loop -- see leg_floor_clamp.py's clamp_limb
        # docstring for the full root-cause writeup. Opt-in trust region,
        # perframe only (median never triggers this, and capping by default
        # regresses Phase A's legitimate large corrections -- tested).
        limbwise_max_dq = 0.15

    # Step 2: limb-wise pass. No z_support here -- clamp_limb targets the
    # actual lowest mesh point directly (see _limbwise_pass docstring).
    qpos = _limbwise_pass(model, data, mesh_cache, qpos, held, effectors,
                           args.ramp_frames, max_dq=limbwise_max_dq,
                           avoid_self_collision=args.avoid_self_collision,
                           rate_limit=args.clamp_rate_limit)

    root_pos = qpos[:, :3]
    root_rot_wxyz = qpos[:, 3:7]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos[:, 7:]
    motion_data = {
        "fps": fps, "root_pos": root_pos, "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos, "local_body_pos": None, "link_body_list": None,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved {qpos.shape[0]} frames to {out_path}")


if __name__ == "__main__":
    main()

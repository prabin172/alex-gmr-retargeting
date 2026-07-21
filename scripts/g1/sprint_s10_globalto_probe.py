#!/usr/bin/env python3
"""S10 Phase 0: single-clip probe for replacing the per-frame re-clamp +
CorrectionRateLimiter + posture-gate tail of the pipeline (S7-S9's `_smrc*`/
`_rl*`/`_pg*` chain) with ONE whole-trajectory global solve.

Motivation (Prabin, 2026-07-20): `leg_floor_clamp.clamp_limb` is a per-frame
INDEPENDENT DLS solve with zero temporal coupling -- confirmed root cause of
`sprint1_subject4`'s right_ankle_pitch flipping ~t=6296-6306 between its own
hard limit (0.5236 rad) and -0.5..-0.8 rad frame to frame (S9-T0/T1, see
`leg_floor_clamp.py`'s `q_prev_chain`/`limit_weight` docstrings). S9's fix was
a null-space BIAS bolted onto the same per-frame solve (posture-continuity,
then a raw-velocity gate on top) -- both patches, not a structural fix, and
the gated version cost +6.6-13.1%/+13-21% jerk/skate corpus-wide (floor
class) for a narrowed (not closed) vMax win.

This script does NOT add a new stage. It reuses `stage_b` (already imported
into this project's G1 pipeline via `stage_b_g1.py` for E4/S5-A4/W2-T5
contact-anchoring) with two of its EXISTING, already-wired, previously-unused
params turned on: `count_floor=True` (whole-body floor-collision QP rows,
built from real MuJoCo contacts against the injected floor plane -- same
`_load_model_with_floor`/vetted-collision model every other G1 tool in this
project uses) and `lambda_coll>0` (self-collision QP rows, the SAME
`_build_collision` math `leg_floor_clamp`'s own phase-2 already reuses). Both
were hardcoded OFF (0.0/False) at every G1 call site to date (verified via
grep) -- this project has built the machinery for a global floor+collision
solve but never turned it on for G1.

Why this can't branch-flip the way `clamp_limb` does: `stage_b` solves the
WHOLE trajectory's actuated-joint correction jointly per SCA outer iteration,
with `lambda_smooth` (temporal smoothness) in the SAME objective as
floor/collision/tracking -- a frame that flips branches costs the objective
directly. `clamp_limb` is a per-frame independent minimum-norm DLS with no
such coupling, so it structurally can (and does) pick different null-space
solutions on near-identical consecutive frames.

Known risk (already documented on the Alex side, `SESSION_HANDOFF.md`
continuation-v1 gate): `count_floor=True` doesn't always converge -- 2/3
gate clips there never beat their own warm start even at n_outer=20. Nobody
has run floor/collision-on `stage_b` on G1 at all before this script.

Scope: takes the clip's EXISTING `_perframelimb_sm.pkl` (perframelimb's
first floor-clamp pass, then held-aware stage_a smoothing -- both UNCHANGED,
upstream of what this replaces) and runs ONE `stage_b` call in place of
[re-clamp, CorrectionRateLimiter, posture-gate], then the existing local
grounding envelope (`sprint_s8_t6_localground._envelope`, unchanged, cheap
final pass) -- same shape as the shipped chain's tail, fewer stages.

Usage:
    conda run -n gmr python scripts/g1/sprint_s10_globalto_probe.py \\
        --clip sprint1_subject4 --lambda-coll 0.5 --n-outer 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _robot_lowest_z  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from smooth_heldaware import save_pkl, smooth_heldaware  # noqa: E402
from sprint_s6_range_summary import compute_held_mask  # noqa: E402
from sprint_s8_t3_corpus import eval_one, load_class_map, CANON_DIR, HUMAN_TARGETS_DIR  # noqa: E402
from sprint_s3_full_corpus import ROLE_TO_G1_BODY  # noqa: E402
from sprint_s8_t6_localground import _envelope  # noqa: E402
from eval_motion import build_eval_context, G1_MODEL_DEFAULT  # noqa: E402
from stage_b_g1 import (  # noqa: E402
    detect_g1_foot_contacts, _resolve_g1_feet, G1_TRACK_ROLES, G1_ROLE_BODY,
    G1_CONTACT_GEOM, _pull_to_floor)
from solve_global_trajectory_opt_contactfirst import (  # noqa: E402
    _compute_anchors, _contact_intervals, _get_joint_limits, _run_continuation,
    stage_a, stage_b, N_ACT)
from gmr_contact_retarget import compute_held_masks, compute_z_support, FEET  # noqa: E402

PKL_S5_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s5"
GMR_PKL_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl"
OUT_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/pkl_s10"

# The known S9-T0/T1 flip window on sprint1_subject4 (right_ankle_pitch vs its
# own hard limit) -- qpos[7:] indices per smooth_heldaware.py's RIGHT_CHAIN_DOFS.
FLIP_WINDOW = (6290, 6312)
RIGHT_HIP_YAW_QIDX = 7 + 8     # RIGHT_CHAIN_DOFS[2]
RIGHT_ANKLE_PITCH_QIDX = 7 + 10  # RIGHT_CHAIN_DOFS[4]


RAMP_FRAMES_ANCHOR = 5


def _cosramp(age, ramp):
    if ramp <= 0:
        return 1.0
    age = max(0, min(age, ramp))
    return 0.5 * (1.0 - np.cos(np.pi * age / ramp))


def _build_human_timed_anchors(model, data, mesh_cache, qpos, canon, eff_names,
                                resolved, foot_w, fps, plant_speed=0.05,
                                plant_min_run=2, move_ratio=0.15,
                                ramp_frames=RAMP_FRAMES_ANCHOR):
    """Reconciles the human-vs-robot contact-label mismatch flagged 2026-07-20:
    the eval harness's `worst_float_cm` uses the CANONICAL HUMAN's own contact
    labels (`compute_held_mask`/`compute_held_masks`, human-side height+speed
    gate); `detect_g1_foot_contacts` (robot-FK based, this project's older
    feet-only E4-era detector) is a DIFFERENT signal that need not agree --
    morphological scaling (G1 is ~64% human size) can shift exactly WHEN the
    robot's own foot reaches the ground relative to the human's, so anchoring
    off robot-FK zones can miss frames the eval (and the semantic motion
    itself) considers held, and vice versa.

    Split per Prabin's framing (2026-07-20): human decides WHEN (timing is a
    property of the motion's intent, not of this robot's specific leg length),
    robot decides WHERE (the physical placement has to respect G1's own
    geometry). This is exactly the shipped `gmr_contact_retarget.ContactAwareGMR`
    mechanism's own design (S5-A1, already validated -- held mask from
    `compute_held_masks` from canonical human, ramped 5-frame cosine on/off,
    Z from `compute_z_support` -- a RIGID per-robot constant computed once at
    model.qpos0, not a per-run data-dependent median like `_pull_to_floor`,
    so it can't be biased by whatever the raw pose happened to be during that
    run). This function is the stage_b/global equivalent of that same
    mechanism, previously only wired into the inline GMR-solve path.

    XY comes from the robot's OWN trajectory. First attempt locked the WHOLE
    human-labeled interval to one XY median -- WRONG: a human "contact" window
    can span real repositioning motion (that's the exact point Prabin raised:
    scaling can shift robot-vs-human timing), so locking the whole span fought
    the actual motion and keep-best rejected every outer step on 2/3 test
    clips (walk1_subject1, ground1_subject1 -- zero net effect even at
    n_outer=10, confirmed identical `Stage B best` to n_outer=1). Fixed by
    sub-segmenting EACH human interval by the robot's OWN point speed
    (`plant_speed`, same idea `_compute_anchors` already uses) -- genuinely
    still sub-runs get a hard per-run median XY anchor; still-moving frames
    inside the same human-labeled window follow the per-frame IK point at low
    weight (`move_ratio`), so the target never fights real motion. Z always
    uses the rigid constant regardless of sub-segment (compute_z_support does
    not depend on stillness)."""
    held, T = compute_held_masks(canon, eff_names)
    z_support = compute_z_support(model, data, mesh_cache, eff_names)
    print(f"  z_support (rigid per-robot constant): "
          + ", ".join(f"{eff}={z_support[eff]*100:.2f}cm" for eff in eff_names))

    tgt = {eff: np.full((T, 3), np.nan) for eff in resolved}
    wgt = {eff: np.zeros(T) for eff in resolved}
    planted = {eff: np.zeros(T, bool) for eff in resolved}
    w_move = foot_w * move_ratio

    for eff, info in resolved.items():
        bid = info["body_id"]
        col = held[eff]
        pct = col.mean() * 100
        print(f"  {eff}: {int(col.sum())}/{T} frames human-held ({pct:.1f}%)")
        pts = np.zeros((T, 3))
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            pts[t] = data.xpos[bid]
        speed = np.zeros(T)
        speed[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) * fps
        speed[0] = speed[1] if T > 1 else 0.0

        for (k, j) in _contact_intervals(col):  # inclusive [k, j]
            still = speed[k:j + 1] < plant_speed
            n_planted = 0
            m = k
            while m <= j:
                if still[m - k]:
                    n = m
                    while n <= j and still[n - k]:
                        n += 1
                    if n - m >= plant_min_run:
                        med_xy = np.median(pts[m:n, :2], axis=0)
                        tgt[eff][m:n, :2] = med_xy
                        wgt[eff][m:n] = foot_w
                        planted[eff][m:n] = True
                        n_planted += n - m
                    else:
                        tgt[eff][m:n, :2] = pts[m:n, :2]
                        wgt[eff][m:n] = w_move
                    m = n
                else:
                    tgt[eff][m, :2] = pts[m, :2]
                    wgt[eff][m] = w_move
                    m += 1
            tgt[eff][k:j + 1, 2] = z_support[eff]  # Z always the rigid constant

            run_len = j - k + 1
            for idx, t in enumerate(range(k, j + 1)):
                age_in = idx
                age_out = run_len - 1 - idx
                ramp = min(_cosramp(age_in, ramp_frames), _cosramp(age_out, ramp_frames))
                wgt[eff][t] *= ramp
    return tgt, wgt, planted


def run_stage_b_global(model, data, mesh_cache, floor_gid, qpos, fps,
                        lambda_coll, n_outer, trust, count_floor=True,
                        human_contact=False, canon=None, continuation=0,
                        plant_speed=0.05):
    eff_names, flags = detect_g1_foot_contacts(qpos, model, data, fps, height_thresh=0.05)
    for i, eff in enumerate(eff_names):
        pct = flags[:, i].mean() * 100
        print(f"  {eff}: {int(flags[:, i].sum())}/{qpos.shape[0]} frames in contact zone ({pct:.1f}%)")

    resolved = _resolve_g1_feet(model, eff_names)

    if human_contact:
        assert canon is not None, "human_contact=True requires canon path"
        tgt, wgt, planted = _build_human_timed_anchors(
            model, data, mesh_cache, qpos, canon, eff_names, resolved, foot_w=40.0,
            fps=fps, plant_speed=plant_speed, plant_min_run=2, move_ratio=0.15)
    else:
        tgt, wgt, planted = _compute_anchors(
            model, data, qpos, eff_names, flags, resolved, fps,
            plant_speed=0.05, foot_w=40.0, hand_w=0.0,
            move_ratio=0.15, plant_min_run=2)
        # Anchor's kind="foot" contact point is the ANKLE BODY ORIGIN (_contact_point),
        # not the sole -- honoring the raw median origin-Z as the plant target bakes in
        # whatever float the raw GMR pose already had at that origin (measured on
        # sprint1_subject4: origin-to-sole offset ~4.2cm median, anchor origin-Z ~6.5cm
        # -> ~2.2cm median / ~4.5cm worst-case float baked into the "planted" target
        # itself, invisible to XY/Z coupling -- both were always locked together, this
        # is a different bug: locked to the WRONG Z). _pull_to_floor (already built for
        # the W2-T5 multi-surface path, unused on this feet-only path until now)
        # replaces each plant run's anchor Z with the origin height that puts the SAME
        # run's own mesh-exact sole point at world Z=0 -- X,Y untouched.
        tgt = _pull_to_floor(tgt, planted, resolved, qpos, model, data, mesh_cache)
    for eff in resolved:
        n = int((~np.isnan(tgt[eff][:, 0])).sum())
        npl = int(planted[eff].sum())
        print(f"  {eff}: contact {n}/{qpos.shape[0]} ({n/qpos.shape[0]*100:.1f}%), "
              f"planted {npl} ({npl/max(n,1)*100:.0f}% of contact)")

    role_to_body = {r: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                    for r, b in G1_ROLE_BODY.items()}
    T = qpos.shape[0]
    target_positions = np.zeros((T, len(G1_TRACK_ROLES), 3))
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        for ri, role in enumerate(G1_TRACK_ROLES):
            target_positions[t, ri] = data.xpos[role_to_body[role]]
    target_weights = {r: 1.0 for r in G1_TRACK_ROLES}

    downweight_roles = [set() for _ in range(T)]
    for eff in resolved:
        anchored = ~np.isnan(tgt[eff][:, 0])  # actually-anchored frames, not the
        for t in np.where(anchored)[0]:       # robot-FK zone (mismatched under human_contact)
            downweight_roles[t].add(eff)

    q_lo, q_hi = _get_joint_limits(model)
    assert qpos.shape[1] - 7 == N_ACT, f"expected {N_ACT} actuated joints, got {qpos.shape[1]-7}"

    t0 = time.time()
    qpos_out = stage_b(
        qpos, target_positions, G1_TRACK_ROLES, role_to_body, target_weights,
        tgt, wgt, planted, resolved, downweight_roles,
        model, data, q_lo, q_hi,
        lambda_track=1.0, lambda_smooth=20.0, lambda_coll=lambda_coll,
        foot_flat_w=3.0, fist_w=0.0,
        downweight_factor=0.1, n_outer=n_outer, trust=trust,
        collision_penalty=1000.0, floor_z=None, floor_w=0.0,
        floor_gid=floor_gid, count_floor=count_floor)
    elapsed = time.time() - t0
    print(f"  stage_b wall time: {elapsed:.1f}s ({T} frames, {elapsed/T*1000:.1f}ms/frame)")

    if continuation > 0:
        # Homotopy schedule for clips whose starting violation is too severe to
        # close in ONE trust-region-bounded SCA step (measured: ground1_subject1,
        # 89.4% baseline self+floor collision incidence, every single-shot outer
        # iteration up to n_outer=10 made pen/coll WORSE, keep-best always
        # reverted to no-op). `_run_continuation` (already built, Alex-side,
        # continuation-v1) shrinks the ALLOWED floor penetration per-frame from
        # this pass's own value toward 0 over K passes, hardens the floor row's
        # slack penalty each pass, and relaxes tracking only on violating
        # limb/frame windows -- gives the linearization a much easier problem to
        # solve each step instead of one huge jump. Cross-pass keep-best can
        # never return worse than the plain (pass-0) result above.
        cont_args = argparse.Namespace(
            continuation=continuation, lambda_track=1.0, lambda_smooth=20.0,
            lambda_coll=lambda_coll, foot_flat_weight=3.0, fist_weight=0.0,
            contact_downweight=0.1, n_outer=n_outer, trust=trust,
            collision_penalty=1000.0, floor_weight=0.0,
            cont_track_min=0.05, cont_window_pad=12,
            cont_floor_penalty_max=1e5, cont_stall_frac=0.10)
        t1 = time.time()
        qpos_out, pen_per_pass = _run_continuation(
            qpos_out, cont_args, model, data, q_lo, q_hi,
            target_positions, G1_TRACK_ROLES, role_to_body, target_weights,
            tgt, wgt, planted, resolved, downweight_roles,
            floor_z=None, floor_gid=floor_gid, floor_active_frames=None)
        print(f"  continuation wall time: {time.time()-t1:.1f}s, "
              f"floor_pen per pass (cm): {pen_per_pass}")
    return qpos_out


def ground(model, data, mesh_cache, geom_ids, qpos, fps):
    lowest = np.zeros(qpos.shape[0])
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        lowest[t] = _robot_lowest_z(model, data, mesh_cache, geom_ids)
    required = np.maximum(0.0, -lowest)
    envelope = _envelope(required, fps)
    out = qpos.copy()
    out[:, 2] += envelope
    print(f"  grounding: peak envelope={envelope.max()*100:.2f}cm")
    return out


def print_flip_window(label, qpos):
    lo, hi = FLIP_WINDOW
    lo = max(0, lo); hi = min(qpos.shape[0], hi)
    print(f"  [{label}] right_hip_yaw / right_ankle_pitch, t={lo}..{hi}:")
    for t in range(lo, hi):
        print(f"    t={t:5d}  hip_yaw={qpos[t, RIGHT_HIP_YAW_QIDX]:+.4f}  "
              f"ankle_pitch={qpos[t, RIGHT_ANKLE_PITCH_QIDX]:+.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="sprint1_subject4")
    ap.add_argument("--lambda-coll", type=float, default=0.5, help="matches leg_floor_clamp's own shipped default")
    ap.add_argument("--n-outer", type=int, default=6, help="pipeline default")
    ap.add_argument("--trust", type=float, default=0.15)
    ap.add_argument("--no-floor-collision", action="store_true",
                     help="isolation probe: self-collision only, floor rows off")
    ap.add_argument("--from-raw", action="store_true",
                     help="Stage 1 test: input = GMR raw directly (this clip's own "
                          "gmr_raw pkl), global clamp REPLACES the per-frame "
                          "perframelimb pass entirely, instead of running downstream "
                          "of it.")
    ap.add_argument("--smooth-after", action="store_true",
                     help="apply the existing (unchanged) held-aware stage_a smoothing "
                          "pass after the global clamp -- Stage 2, kept as-is.")
    ap.add_argument("--cleanup-n-outer", type=int, default=0,
                     help="Stage 3: a SECOND global clamp pass (same stage_b call, "
                          "fresh anchors) on the smoothed output, cleaning up whatever "
                          "smoothing reintroduced -- global analog of the old per-frame "
                          "re-clamp. 0 = skip (default).")
    ap.add_argument("--human-contact", action="store_true",
                     help="anchor timing from the canonical HUMAN's own contact labels "
                          "(compute_held_masks) instead of detect_g1_foot_contacts' "
                          "robot-FK zone -- reconciles the eval-vs-anchor mismatch found "
                          "2026-07-20. Z target from compute_z_support (rigid per-robot "
                          "constant), not a per-run median.")
    ap.add_argument("--continuation", type=int, default=0,
                     help="homotopy passes on Stage 1 (ported from continuation-v1, "
                          "solve_global_trajectory_opt_contactfirst._run_continuation) -- "
                          "for clips whose starting violation is too severe for one "
                          "trust-region-bounded SCA step (ground1_subject1: 89.4% "
                          "baseline, every single-shot outer iteration up to n_outer=10 "
                          "made things worse). 0 = off (default).")
    ap.add_argument("--plant-speed", type=float, default=0.05,
                     help="stillness threshold (m/s) for the human-timed anchor's "
                          "within-interval sub-segmentation -- default matches "
                          "_compute_anchors' own convention.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.from_raw:
        src = GMR_PKL_DIR / f"{args.clip}.pkl"
        before_label = "BEFORE (gmr_raw)"
        out_tag = "globalto_fromraw"
    else:
        src = PKL_S5_DIR / f"{args.clip}_perframelimb_sm.pkl"
        before_label = "BEFORE (perframelimb_sm, pre re-clamp)"
        out_tag = "globalto_localground"
    assert src.exists(), f"missing {src}"
    out_path = args.out or (OUT_DIR / f"{args.clip}_{out_tag}.pkl")

    print(f"[S10 Phase 0] clip={args.clip} from_raw={args.from_raw} "
          f"lambda_coll={args.lambda_coll} n_outer={args.n_outer} trust={args.trust} "
          f"smooth_after={args.smooth_after}")
    print(f"  input: {src}")

    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    canon = CANON_DIR / f"{args.clip}_lafan1c_grounded.npz"
    qpos_in, fps = load_gmr_pkl(src)
    print(f"  T={qpos_in.shape[0]} fps={fps}")
    print_flip_window(before_label, qpos_in)

    qpos_b = run_stage_b_global(model, data, mesh_cache, floor_gid, qpos_in, fps,
                                 args.lambda_coll, args.n_outer, args.trust,
                                 count_floor=not args.no_floor_collision,
                                 human_contact=args.human_contact, canon=canon,
                                 continuation=args.continuation, plant_speed=args.plant_speed)
    print_flip_window("AFTER global clamp (Stage 1, pre-smooth)", qpos_b)

    if args.smooth_after:
        held, T = compute_held_masks(canon, FEET)
        assert T == qpos_b.shape[0], f"canonical T={T} != qpos T={qpos_b.shape[0]}"
        qpos_b = smooth_heldaware(qpos_b, held, fps=fps)
        print_flip_window("AFTER Stage 2 (held-aware smoothing, unchanged)", qpos_b)

    if args.cleanup_n_outer > 0:
        qpos_b = run_stage_b_global(model, data, mesh_cache, floor_gid, qpos_b, fps,
                                     args.lambda_coll, args.cleanup_n_outer, args.trust,
                                     count_floor=not args.no_floor_collision,
                                     human_contact=args.human_contact, canon=canon)
        print_flip_window("AFTER Stage 3 (global cleanup pass)", qpos_b)

    qpos_g = ground(model, data, mesh_cache, geom_ids, qpos_b, fps)
    print_flip_window("AFTER localground (final)", qpos_g)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(out_path, qpos_g, fps)
    print(f"  wrote {out_path}")

    # Eval, same axes/harness as the S8/S9 corpus CSV, for direct comparison.
    canon = CANON_DIR / f"{args.clip}_lafan1c_grounded.npz"
    held, _ = compute_held_mask(canon)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for role, name in ROLE_TO_G1_BODY.items()}
    vmax_ctx = build_eval_context(G1_MODEL_DEFAULT)
    human_targets_path = HUMAN_TARGETS_DIR / f"{args.clip}.npz"
    row = eval_one(model, data, mesh_cache, geom_ids, floor_gid, role_bid, held,
                    human_targets_path, vmax_ctx, args.clip, "s10_globalto_localground", out_path)
    print("\n[S10 Phase 0 result]")
    for k, v in row.items():
        print(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()

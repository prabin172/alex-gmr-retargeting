#!/usr/bin/env python3
"""S2-T3: contact-first per-frame IK on G1 -- the "OURS" row of the 2x2
(GMR-baseline.md SS7.4), and the mechanism W2-T5's checkpoint concluded this
motion class actually needs (root+contacts solved jointly, not anchored
post-hoc on a fixed trajectory).

Reuses UNCHANGED from `solve_fbx_canonical_alex_contactfirst.py` (all take
role-name-keyed dicts or generic model/data -- role names are shared
vocabulary between our canonical-human schema and Alex's, so these work
correctly for G1 as long as the SAME role names are used, which
`lafan1_to_canonical_human.py` (S2-T1) already guarantees):
  load_canonical, measure_alex_pelvis_to_head, estimate_source_scale,
  make_initial_alignment_targets, make_targets_for_frame,
  make_orientation_targets_for_frame, clamp_hinge_joint_limits,
  solve_frame_position_ik. `compute_per_role_scales` is NO LONGER used --
  replaced by `gmr_grouped_role_scales` (this file), matching GMR's own
  per-body-GROUP scale factors instead of independent per-role ratios
  (S2-T6, see that function's docstring for the diagnosed bug this fixes).

G1-specific (this file): ROLE_TO_G1_BODY (15 roles, exact same KEY SET as
ROLE_TO_ALEX_BODY -- required, since the reused functions above iterate that
Alex dict's KEYS directly for role selection), ORI_TO_G1_BODY (7 roles,
matches ORI_TO_ALEX_BODY), the combined vetted-collision+floor model
(g1_model_setup.py, W2-T6 extended).

**FOOTGUN, confirmed by direct measurement**: `g1_mocap_29dof.xml`'s body
named "head_link" sits at pelvis height (world [0,0,0] at the neutral pose --
its local offset from torso is the exact geometric negation of torso's own
offset from waist_roll, confirmed numerically) -- it's a cosmetic/logo-adjacent
body, NOT the anatomical head (GMR's own ik_config never maps anything to it
either). The correct "head" analog is **`head_mocap`** (a plain fixed body,
despite the name -- no `mocap="true"` attribute, ordinary FK; "mocap" here
means physical motion-capture MARKER placement, not MuJoCo's mocap body type),
which measures a sane pelvis-to-head distance of 0.444m at rest. Using
`head_link` silently zeroes `root_scale` (pelvis-to-head divides through
`estimate_source_scale`) and cascades into a degenerate, all-collapsed solve.

**v1 scope, documented simplifications (time-boxed per the plan, 2-day
tier)**: NO fist/palm position pin (G1 has no fist/gripper -- CONTACT_POS
skipped entirely), NO foot-flat orientation alignment term during contact
(position-only contact-first: detected-stance feet get `hold_pos_roles`
treatment -- frozen high-priority position target -- but no axis-alignment
row), NO shank-clamp / swing-clear / arm-floor-transition / leg-floor-transition
refinement passes (Alex-specific polish for bugs found on Alex's skeleton,
not core to the contact-first claim). This is the CORE mechanism: per-frame
damped-least-squares IK solving ALL joints + root jointly against
morphology-scaled targets, with detected-contact effectors held in place at
high priority and real (vetted) self-collision + floor-collision avoidance
rows -- genuinely different from GMR's uniform per-frame differential IK,
which has no contact-aware anchoring at all.

Usage:
    conda run -n gmr python scripts/g1/solve_lafan1_canonical_g1_contactfirst.py \\
        --canonical outputs/gmr_baseline/sprint/canonical_human/walk1_subject1_grounded.npz \\
        --out outputs/gmr_baseline/sprint/ours_g1/walk1_subject1_ours.npz
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

from contact_labels import debounce_flags, ramp_envelope  # noqa: E402
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _geom_lowest_z  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import (  # noqa: E402
    body_xmat, cap_foot_pitch, clamp_hinge_joint_limits,
    estimate_source_scale, load_canonical, make_initial_alignment_targets,
    make_orientation_targets_for_frame, make_targets_for_frame,
    measure_alex_pelvis_to_head, solve_frame_position_ik, _swing_posture_reg)
from stage_b_g1 import support_z  # noqa: E402

# GMR's OWN per-body-GROUP scale factors (planLogGMR.md S2-T6, Prabin's request
# to match their method so contact enforcement is the only remaining
# difference). Read directly from their published, LAFAN1-specific ik_config
# (general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json's
# "human_scale_table" -- the EXACT LAFAN1-to-G1 pipeline being compared
# against, not a generic config). Confirmed: every lower-body joint (Hips,
# Spine2, UpLeg, Leg, FootMod) shares s_b=0.9; every upper-body joint (Arm,
# ForeArm, Hand) shares s_b=0.75; anything unlisted (Head, Neck) is implicit
# 1.0. This REPLACES `compute_per_role_scales` (imported from Alex's solver,
# now unused here) -- that function computes an INDEPENDENT ratio per
# individual role (G1's own achieved-rest distance from pelvis / this human's
# distance from pelvis), with NO constraint that hip/knee/ankle scales agree
# with each other. Root-caused (S2-T6): on `left_hip_yaw_link` specifically,
# this produced scales of 2.45 (hip) / 0.97 (knee) / 0.79 (ankle) -- three
# unrelated numbers applied to three points on the SAME rigid leg, yielding a
# TARGET thigh length of 36cm against G1's actual 19cm thigh (nearly 2x) --
# a kinematically impossible chain no solver could satisfy, with tracking
# error compounding down the leg (hip -2.8cm, knee -6.0cm, ankle -8.5cm).
# GMR's grouped scaling cannot produce this: every joint in one limb shares
# the SAME factor, so the human's own hip-knee-ankle proportions are
# uniformly rescaled, never internally distorted, regardless of which
# specific G1 body is chosen to track each role.
#
# Deliberately NOT adopting GMR's `(h/h_ref)` height-normalization term on
# top of this -- that serves the same conceptual purpose as our OWN
# `root_scale` (already computed per-clip from G1's vs this human's own
# pelvis-to-head distance, a more adaptive measurement than GMR's fixed
# h_ref=1.8m/LAFAN1-hardcoded-1.75m constant) and stacking both would
# double-apply a human-size correction. Also deliberately NOT adopting GMR's
# formula's lack of a delta-from-rest anchor -- this project's morphology-
# scaling convention (CLAUDE.md: "motion deltas from rest only, never
# absolute root/pelvis position") is a separate, already-settled design
# choice serving a different purpose (never teleporting away from G1's own
# natural rest pose) and stays unchanged; only the SOURCE of the per-role
# scale multiplier changes here.
GMR_GROUPED_SCALES = {
    "torso": 0.9, "left_hip": 0.9, "right_hip": 0.9,
    "left_knee": 0.9, "right_knee": 0.9, "left_ankle": 0.9, "right_ankle": 0.9,
    "left_shoulder": 0.75, "right_shoulder": 0.75,
    "left_elbow": 0.75, "right_elbow": 0.75,
    "left_wrist": 0.75, "right_wrist": 0.75,
    "head": 1.0,  # not in GMR's human_scale_table -- implicit default
}


def gmr_grouped_role_scales(role_to_body_id):
    """Per-role scale dict matching GMR's own published grouped constants,
    same shape/keys `compute_per_role_scales` would have returned (pelvis
    hardcoded to 1.0, matching that function's own pelvis special-case --
    pelvis tracks via root_scale, not a role scale, at all times).

    **S2-T10 fix attempt, TRIED AND REJECTED (2026-07-17)**: also multiplying
    by `root_scale` (matching GMR's own `(h/h_ref)` hitting both the root AND
    relative terms of their formula, vs our current code only applying
    `root_scale` to the root term) was tested end-to-end on all 4 clips.
    Correctly shrank the diagnosed over-reach (fallAndGetUp2_subject2 worst
    hip-ankle target distance 181.6%->117.1% of G1's max leg reach, frame-legs
    exceeding 100% reach 2650->852) but made EVERY other metric substantially
    WORSE: self-collision (ground1_subject1 11.6%->34.5%, walk1 2.1%->16.8%,
    fallAndGetUp1 6.4%->19.4%), floorPen (fallAndGetUp1 25.5->50.9cm, nearly
    doubled), even held-frame frac<3cm (walk1 78.6%->62.2%). Root cause of the
    regression: `root_scale` (~0.64-0.65, G1 is only ~64% this human's size)
    applied on TOP of the 0.9/0.75 group constant shrinks the whole clip's
    limb excursion far more aggressively than the rare extreme-reach frames
    need -- limbs stay closer to the body/pelvis THROUGHOUT the entire
    clip (not just at the few over-reach frames), which increases
    self-collision everywhere and doesn't help floor contact (compressed
    swing-foot lift can scuff the floor instead of clearing it). REVERTED --
    the rare-frame over-reach problem (a genuine, small, honest residual:
    G1 being ~64% human-sized means SOME extreme poses are unreachable by
    ANY uniform linear scale) is a smaller cost than what a blanket
    additional shrink causes across the whole clip. See planLogGMR.md S2-T10
    for the full numbers; not fixed further this pass -- would need a
    LOCAL/adaptive correction (e.g. only softly pulling back the rare
    over-reach frames specifically, not a global multiplier) to do better,
    which is a different, larger mechanism than this one-line change."""
    scales = {"pelvis": 1.0}
    for role in role_to_body_id:
        if role == "pelvis":
            continue
        scales[role] = GMR_GROUPED_SCALES.get(role, 1.0)
    return scales

# Matches ROLE_TO_ALEX_BODY's exact KEY SET -- required, since
# make_initial_alignment_targets/compute_per_role_scales iterate that dict's
# keys directly (module-level import), not a passed-in argument. Values are
# G1 body names (verified present in the combined model, g1_model_setup.py).
ROLE_TO_G1_BODY = {
    "pelvis": "pelvis",
    "torso": "torso_link",
    "head": "head_mocap",  # NOT head_link -- see g1_model_setup.py note below

    "left_hip": "left_hip_yaw_link",     # GMR's own ik_config uses this as its "thigh" analog
    "right_hip": "right_hip_yaw_link",

    "left_shoulder": "left_shoulder_yaw_link",   # matches GMR ik_config's LeftArm correspondence
    "right_shoulder": "right_shoulder_yaw_link",

    "left_knee": "left_knee_link",
    "left_ankle": "left_ankle_roll_link",
    "right_knee": "right_knee_link",
    "right_ankle": "right_ankle_roll_link",

    "left_elbow": "left_elbow_link",
    "left_wrist": "left_wrist_yaw_link",
    "right_elbow": "right_elbow_link",
    "right_wrist": "right_wrist_yaw_link",
}

# Matches ORI_TO_ALEX_BODY's exact key set.
ORI_TO_G1_BODY = {
    "pelvis": "pelvis",
    "torso": "torso_link",
    "head": "head_mocap",  # NOT head_link -- see g1_model_setup.py note below
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_hand": "left_rubber_hand",
    "right_hand": "right_rubber_hand",
}

FOOT_POS_ROLE = {"left_foot": "left_ankle", "right_foot": "right_ankle"}
# Hands added (planLogGMR.md S2-T5): found necessary after the worst whole-clip
# penetration frame turned out to be a HAND (left_wrist_pitch_link, -91cm), not
# a foot -- pull-to-floor only covered FOOT_POS_ROLE, leaving hand-contact
# frames (a fall/lying clip's hands ARE a real support surface, W2-T3 already
# found 41.2%/34.5% hand contact zone on this exact clip) with zero
# floor-referencing at all.
HAND_POS_ROLE = {"left_hand": "left_wrist", "right_hand": "right_wrist"}
CONTACT_POS_ROLE = {**FOOT_POS_ROLE, **HAND_POS_ROLE}

# S4-T3: leg-floor-transition refine pass, ported from
# `refine_leg_floor_transitions` (solve_fbx_canonical_alex_contactfirst.py).
# G1 has no equivalent of Alex's SOLE_CORNER_SITES (hand-authored sites on
# `alex_floating_base_with_sites.xml`) or foot-flat align_constraint (this
# file's v1 scope deliberately skips orientation-alignment during contact --
# see module docstring) -- detection here is mesh-accurate penetration of the
# leg's OWN geoms (reusing `_geom_lowest_z`, already used by the eval/grounding
# scripts) instead of named sites, and the synthetic-plant Z target reuses this
# file's OWN pull-to-floor formula (`support_z` offset) instead of Alex's fixed
# `alex_floor_z + ankle_clearance` (G1's floor is always z=0 by this project's
# grounding convention -- no clip-wide floor-height estimate needed).
LEG_CHAIN_JOINTS = {
    "left_foot": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "right_foot": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                   "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
}
LEG_BODY_NAMES = {
    "left_foot": ["left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
                  "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link"],
    "right_foot": ["right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
                   "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link"],
}
# Hip+knee only (NOT ankle -- must stay free to dorsiflex/flatten), both legs
# combined -- matches Alex's `leg_cont_dofs` exactly (solve_fbx_canonical_alex_
# contactfirst.py, "a global boost caused a wrist flip in testing" -- scoped to
# just these 8 DOFs). Used by --swing-clear's temporal-continuity reg (S4-T4).
LEG_CONT_JOINTS = [n for names in LEG_CHAIN_JOINTS.values() for n in names[:4]]


def _joint_dofadr(model, jname):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    assert jid >= 0, f"G1 joint {jname!r} not found"
    return int(model.jnt_dofadr[jid])


def _leg_floor_pen_flags_g1(model, data, qpos, mesh_cache, leg_geom_ids, pen_tol):
    """Per-frame boolean: does this leg's own deepest mesh point sit more than
    `pen_tol` below the floor (z=0, this project's fixed grounding convention)?
    Purely geometric, independent of the contact-flag 'planted'/hold_pos_roles
    label -- catches a genuinely swinging leg (walk3_subject1's failure mode,
    S4-T1/T3) exactly as well as a nominally-planted foot left penetrating."""
    T = qpos.shape[0]
    flags = np.zeros(T, dtype=bool)
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        z = min(_geom_lowest_z(g, model, data, mesh_cache) for g in leg_geom_ids)
        flags[t] = z < -pen_tol
    return flags


def refine_leg_floor_transitions_g1(
    model, data, qpos_pass1, frame_cache, role_to_body_id, ori_role_to_body_id,
    mesh_cache, leg_geom_ids_by_eff, coll_kwargs,
    floor_weight, floor_margin, floor_gain,
    pen_tol=0.015, ramp=20, preroll=20,
    leg_posture_reg=0.02, lock_weight=1.0e4, root_pos_relief=0.3, iters=30,
):
    """G1 adaptation of `refine_leg_floor_transitions` (S4-T3, see planLogGMR.md):
    root cause of the `--floor-weight` blowup (373cm on walk3_subject1 at
    weight=20 -- S4-T3) is the SAME class as Alex's original wrist-flick
    (SESSION_HANDOFF "Session 2026-07-09/10", Fix C bug #3): an UNRAMPED floor
    correction applied in one frame to a limb that just crossed the floor
    threshold overwhelms that frame's step budget (`max_step_norm=0.20`), and
    because qpos warm-starts frame-to-frame the damage compounds instead of
    self-correcting. This pass never runs `--floor-weight` in the main
    per-frame loop (stays default 0, unstable per S4-T3) -- instead it
    LOCALLY re-solves just the affected leg's 6-joint chain over a short,
    cosine-ramped window around each detected floor-onset/sustained-dig
    segment, exactly mirroring `refine_arm_floor_transitions`'s architecture:
    warm-start every touched frame from Pass 1's OWN solved pose (good initial
    guess), regularize the leg chain toward the PREVIOUSLY REFINED frame (not
    a frozen value -- avoids creating a new discontinuity at the window's
    exit), lock everything else at `lock_weight` so the local re-solve can't
    disturb an already-good body, root DOFs always free (posture_reg never
    touches them).

    Synthetic plant: blend the ankle Z target toward this file's OWN
    pull-to-floor estimate (body origin height above its OWN mesh-exact
    support point, at the CURRENT warm-start orientation -- same formula as
    the main loop's pull-to-floor block) instead of a fixed clip-wide floor
    estimate, and relax pelvis/torso position tracking (`pos_weight_scale`)
    during the window so the root is actually free to rise/shift to make room
    for the leg, not fighting it every frame.

    `frame_cache[i]` must hold the (targets, ori_targets, hold_pos_roles)
    tuple Pass 1 used for sequential index i (this file's simpler per-frame
    call has no align_constraints/pos_site_constraints/skip_pos_roles/
    skip_ori_body_ids/ori_weight_scale/pos_weight_scale -- v1 scope, see
    module docstring -- so the cache is lighter than Alex's version)."""
    qpos_out = qpos_pass1.copy()
    nv = model.nv

    for eff, geom_ids in leg_geom_ids_by_eff.items():
        pen_flags = _leg_floor_pen_flags_g1(model, data, qpos_pass1, mesh_cache, geom_ids, pen_tol)
        if not pen_flags.any():
            continue
        alpha = ramp_envelope(pen_flags, ramp, preroll)

        T = qpos_pass1.shape[0]
        touched = np.where(alpha > 1e-9)[0]
        if touched.size == 0:
            continue
        segments = []
        seg_start = touched[0]
        prev_t = touched[0]
        for t in touched[1:]:
            if t != prev_t + 1:
                segments.append((seg_start, prev_t))
                seg_start = t
            prev_t = t
        segments.append((seg_start, prev_t))

        ank_role = FOOT_POS_ROLE[eff]
        ank_bid = role_to_body_id[ank_role]
        chain_dofadr = [_joint_dofadr(model, n) for n in LEG_CHAIN_JOINTS[eff]]

        preg = np.full(nv, lock_weight)
        preg[0:6] = 0.0                     # root: posture_reg never applies here anyway
        for d in chain_dofadr:
            preg[d] = leg_posture_reg

        n_frames_touched = 0
        for lo, hi in segments:
            q_prev = qpos_out[max(lo - 1, 0)].copy()
            for t in range(lo, hi + 1):
                w = float(alpha[t])
                if w <= 0.0:
                    continue
                targets, ori_targets, hold_pos_roles = frame_cache[t]

                # Synthetic plant: blend the ankle Z target toward THIS FRAME's
                # own pull-to-floor estimate, evaluated at the CURRENT warm-start
                # (qpos_out[t] not yet overwritten -- still Pass 1's pose, or a
                # prior refine pass's if this is the second effector).
                data.qpos[:] = qpos_out[t]
                mujoco.mj_forward(model, data)
                origin_z_now = float(data.xpos[ank_bid][2])
                support_z_now = support_z(model, data, mesh_cache, ank_bid)
                rest_z = origin_z_now - support_z_now

                targets_t = dict(targets)
                tgt = targets[ank_role].copy()
                tgt[2] = (1.0 - w) * tgt[2] + w * rest_z
                targets_t[ank_role] = tgt

                pos_weight_scale_t = {}
                for role in ("pelvis", "torso"):
                    pos_weight_scale_t[role] = 1.0 - w * (1.0 - root_pos_relief)

                q_init_t = qpos_pass1[t].copy()
                q_ref_t = qpos_pass1[t].copy()
                q_ref_t[7:] = np.where(
                    np.isin(np.arange(nv - 6), np.asarray(chain_dofadr) - 6),
                    q_prev[7:], q_ref_t[7:])

                q_t = solve_frame_position_ik(
                    model, data, role_to_body_id, targets_t, q_init_t,
                    ori_role_to_body_id=ori_role_to_body_id, ori_targets=ori_targets,
                    hold_pos_roles=hold_pos_roles, pos_weight_scale=pos_weight_scale_t,
                    iters=iters, posture_reg=preg, q_ref=q_ref_t,
                    floor_weight=w * floor_weight,
                    floor_margin=floor_margin, floor_gain=floor_gain,
                    **coll_kwargs,
                )

                # Divergence guard (S4-T3, found by direct measurement on
                # fallAndGetUp1_subject1/fallAndGetUp2_subject2): this per-frame
                # DLS solve occasionally lands in a bad local optimum WITHIN one
                # frame's iters budget -- root plunges meters below floor, then
                # the VERY NEXT frame recovers cleanly on its own (isolated,
                # self-correcting, not a sustained lying-phase conflict; measured
                # example: pelvis_z -0.01 -> -2.6 -> +0.03 across 6 frames). If the
                # refined frame is WORSE than Pass 1's own (already-decent) result
                # at this exact frame by more than 3cm, reject it -- keep Pass 1's
                # value and don't let the bad frame poison q_prev for the rest of
                # the window.
                data.qpos[:] = q_t
                mujoco.mj_forward(model, data)
                depth_new = -min(_geom_lowest_z(g, model, data, mesh_cache) for g in geom_ids)
                data.qpos[:] = qpos_pass1[t]
                mujoco.mj_forward(model, data)
                depth_pass1 = -min(_geom_lowest_z(g, model, data, mesh_cache) for g in geom_ids)
                if depth_new > depth_pass1 + 0.03:
                    q_t = qpos_pass1[t].copy()

                qpos_out[t] = q_t
                q_prev = q_t
                n_frames_touched += 1
        print(f"  [leg-floor-refine] {eff}: {len(segments)} window(s), "
              f"{n_frames_touched} frames re-solved (pen_tol={pen_tol*100:.1f}cm)")
    return qpos_out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--ik-iters", type=int, default=30)
    ap.add_argument("--coll-weight", type=float, default=20.0)
    ap.add_argument("--coll-margin", type=float, default=0.02)
    ap.add_argument("--coll-gain", type=float, default=5.0)
    ap.add_argument("--coll-hops", type=int, default=2)
    ap.add_argument("--floor-weight", type=float, default=0.0,
                    help="Floor-repulsion during solving. 0 = off this v1 (Stage 2.5 already "
                         "grounds the canonical human to floor=0; grounding QP (post_process_"
                         "ground_contactfirst.py) handles clip-level floor placement after, "
                         "same division of labor as the G1 GMR pipeline).")
    ap.add_argument("--foot-weight", type=float, default=40.0,
                    help="hold_pos_roles priority weight for a planted foot's ankle target.")
    ap.add_argument("--plant-speed", type=float, default=0.05)
    ap.add_argument("--contact-min-run", type=int, default=2, help="30fps-scaled, W2-T2 convention.")
    ap.add_argument("--contact-ramp", type=int, default=2)
    ap.add_argument("--contact-preroll", type=int, default=1)
    ap.add_argument("--pull-to-floor", dest="pull_to_floor", action="store_true", default=True,
                    help="For held (planted) roles, override the target Z so the body's own "
                         "mesh-exact support point lands at the floor (z=0), instead of just "
                         "freezing whatever Z the morphology-scaled target computed. Found "
                         "necessary by direct measurement: without this, held frames' support "
                         "points sat a median 12-13cm above the floor even after grounding -- "
                         "the whole-body floorPen metric improved from a lucky global shift, not "
                         "genuine per-frame floor contact. See planLogGMR.md S2-T4/S2-T5.")
    ap.add_argument("--no-pull-to-floor", dest="pull_to_floor", action="store_false")
    ap.add_argument("--pull-to-floor-alpha", type=float, default=0.15,
                    help="EMA smoothing factor for the pull-to-floor offset across a zone "
                         "interval (1.0 = no smoothing, raw per-frame snapshot -- found too "
                         "noisy for a moving limb, planLogGMR.md S2-T5 fix attempt 4). Reset "
                         "to a fresh raw estimate at every zone onset, never blended across a "
                         "contact-interval boundary.")
    ap.add_argument("--knee-bias-weight", type=float, default=0.0,
                    help="One-sided knee-flexion bias (ported from solve_fbx_canonical_alex_"
                         "contactfirst.py, never wired into this v1 script -- diagnosed missing "
                         "post-S2-T9, see planLogGMR.md 'swing leak' probe). G1's knee joint "
                         "range is [-0.087, 2.88] -- straight sits at the LOWER limit (unlike "
                         "Alex, where straight is q=0), both a leg-Jacobian singularity and a "
                         "per-iteration clamp (clamp_hinge_joint_limits runs every iteration, "
                         "not just at the end) that a warm start already at that limit cannot "
                         "climb back out of within one frame's small step budget. Weakly pushes "
                         "any knee straighter than --knee-min-flex-deg back toward it; silent "
                         "once bent, so it cannot over-constrain tracking. Default 0 (OFF, "
                         "matches this file's existing opt-in convention) -- Alex's own shipped "
                         "default is 0.5, pass --knee-bias-weight 0.5 to enable the equivalent "
                         "here pending its own validation on G1.")
    ap.add_argument("--knee-min-flex-deg", type=float, default=12.0,
                    help="Flexion (deg) PAST the knee's own lower limit below which the bias "
                         "engages (default: 12, matches Alex's shipped default; inert while "
                         "--knee-bias-weight is 0).")
    ap.add_argument("--knee-bias-skip-held", action="store_true",
                    help="S4-T2: exclude a leg's knee_bias row on any frame where that leg's "
                         "foot is currently held (in hold_pos_roles). Root-caused regression "
                         "(fallAndGetUp1_subject1, S2-T12: floorPen 25.5->36.4cm): bias forced a "
                         "flexion angle on an already-planted, foot-pinned leg at frame 347-348, "
                         "over-determining the leg IK; the perturbation then cascaded through the "
                         "per-frame warm start (chaotically sensitive, S2-T11) into an unrelated, "
                         "much larger spike 3500 frames later. knee_bias's own rationale "
                         "(S2-T11) was always about SWING-leg reach failure, never about held/"
                         "planted legs, where straight is often the physically correct pose. "
                         "Default off for backward compat with the already-recorded S2-T12 "
                         "numbers; recommended on whenever --knee-bias-weight > 0.")
    ap.add_argument("--floor-leg-refine", action="store_true",
                    help="S4-T3: two-pass local floor-avoidance refine (ported from Alex's "
                         "refine_leg_floor_transitions), run AFTER the main per-frame loop. "
                         "Root cause this replaces: --floor-weight applied globally/unramped in "
                         "the main loop diverges on G1 (walk3_subject1: 373cm floorPen at "
                         "weight=20 -- planLogGMR.md S4-T3), the same warm-start-lock-in failure "
                         "as Alex's original wrist-flick bug before its own two-pass fix. This "
                         "flag detects floor-onset/sustained-penetration windows per leg "
                         "(geometric, mesh-accurate, independent of --floor-weight/hold_pos_roles) "
                         "and locally re-solves just that leg's 6-joint chain over a short "
                         "cosine-ramped window, everything else locked. Independent of "
                         "--floor-weight (which stays default-0/unstable, not recommended) -- "
                         "this pass supplies its own --floor-refine-* weight/margin/gain.")
    ap.add_argument("--floor-refine-weight", type=float, default=20.0,
                    help="floor_collision_rows weight used INSIDE the leg-floor refine windows "
                         "only (matches Alex's own luigi_standProne_03 shipped value). Inert "
                         "unless --floor-leg-refine.")
    ap.add_argument("--floor-refine-margin", type=float, default=0.0)
    ap.add_argument("--floor-refine-gain", type=float, default=5.0)
    ap.add_argument("--floor-refine-pen-tol", type=float, default=0.015,
                    help="Meters of mesh-accurate sub-floor depth (per leg) that counts as a "
                         "'floor onset' worth refining. Matches Alex's refine_leg_floor_transitions "
                         "default.")
    ap.add_argument("--floor-refine-ramp", type=int, default=20,
                    help="Frames of cosine ramp-out after a sustained-penetration window ends "
                         "(ramp_envelope convention, matches Alex's default).")
    ap.add_argument("--floor-refine-preroll", type=int, default=20,
                    help="Frames of cosine ramp-in anticipating a floor-onset event (matches "
                         "Alex's default).")
    ap.add_argument("--floor-refine-posture-reg", type=float, default=0.02,
                    help="Regularization weight on the refined leg's OWN chain toward the "
                         "temporal-continuity target (light -- matches Alex's default).")
    ap.add_argument("--floor-refine-lock-weight", type=float, default=1.0e4,
                    help="Regularization weight LOCKING every joint outside the refined leg's "
                         "chain to its Pass-1 value (heavy -- matches Alex's default).")
    ap.add_argument("--floor-refine-root-relief", type=float, default=0.3,
                    help="Pelvis/torso position-tracking weight multiplier at full ramp (1.0 = "
                         "untouched, matches Alex's default 0.3 -- gives the root room to rise/"
                         "shift for the leg instead of fighting it every frame).")
    ap.add_argument("--swing-clear", action="store_true",
                    help="S4-T4 (Prabin's redirect, 2026-07-17): port of Alex's --swing-clear "
                         "core mechanism -- diagnosed cause is narrower than --floor-leg-refine "
                         "assumed. Direct measurement across 8 locomotion clips (~8300 penetrating "
                         "sampled frames): the deepest-penetrating body is ALWAYS an ankle "
                         "(left/right_ankle_roll_link, ~50/50 split) -- never a knee, hip, or "
                         "torso. On pure locomotion this is a classic swing-foot-toe-through-floor "
                         "problem, not the chronic whole-leg/root conflict --floor-leg-refine was "
                         "built for. Mechanism (Alex's own history: tried the soft one-sided "
                         "position-lift term FIRST -- `--floor-weight`/`floor_collision_rows` -- "
                         "found it fought the plant machinery; the thing that actually worked was "
                         "capping the SWING foot's ORIENTATION target's toe-down pitch, spending "
                         "unused ankle dorsiflexion headroom to lift the toe, paired with a "
                         "temporal-continuity posture_reg boost on hip/knee ONLY (not ankle) to "
                         "stop the redundant leg from branch-flipping). `cap_foot_pitch`/"
                         "`_swing_posture_reg` imported UNCHANGED from Alex's solver -- both "
                         "generic, no Alex-specific state. No proximity gate ported (Alex's own "
                         "shipped config leaves it at the effectively-off default -- swing-ness "
                         "via the existing `zone_env` contact envelope alone was sufficient there); "
                         "no soft position-lift term either (Alex found it added nothing over the "
                         "pitch cap alone, off by default there too). Default off.")
    ap.add_argument("--swing-max-pitch", type=float, default=8.0,
                    help="With --swing-clear: max allowed toe-down pitch (deg) of a swing foot's "
                         "forward axis below horizontal. G1-tuned (S4-T4 grid, "
                         "walk1_subject1/run2_subject1/jumps1_subject1, 4x3 mp x cr grid): mp had a "
                         "MUCH weaker effect than --swing-continuity-reg in the tested range "
                         "(3/5/8/12) -- 8 was the best mean floorPen/coll%% tradeoff paired with "
                         "cr=0.2, not dominant on its own. Was 5.0 (Alex's shipped value, "
                         "unvalidated for G1) before this tuning pass.")
    ap.add_argument("--swing-continuity-reg", type=float, default=0.2,
                    help="With --swing-clear: posture_reg boost on LEG_CONT_JOINTS (hip+knee, both "
                         "legs, NOT ankle) on de-pitch frames, ramped by de-pitch strength -- stops "
                         "the redundant leg from jumping IK solution branches when the pitch cap "
                         "engages. G1-tuned (S4-T4): this is the DOMINANT lever, not mp -- Alex's "
                         "shipped 0.9 cost +1.6cm mean floorPen vs baseline (3-clip grid) purely "
                         "from over-regularizing; 0.2 (mp=8) gets -0.8cm mean floorPen (BETTER than "
                         "baseline) while keeping ~all of the held-frac3 gain. 0 confirmed unsafe on "
                         "G1 too (matches Alex's own warning): walk1_subject1's held gain collapsed "
                         "back to near-baseline (78.8/81.5 vs baseline 78.6/81.5, cr=0.2 gets "
                         "92.2/95.4) -- the leg branch-flips instead of tracking cleanly. Was 0.9 "
                         "(Alex's shipped value, unvalidated for G1) before this tuning pass.")
    args = ap.parse_args()

    (roles, role_to_idx, src_positions, fps, orientation_roles, ori_to_idx, orientation_mats,
     persisted_contacts, persisted_eff_names) = load_canonical(args.canonical)

    missing = [r for r in ROLE_TO_G1_BODY if r not in role_to_idx]
    if missing:
        raise RuntimeError(f"Canonical missing required roles: {missing}")
    missing_ori = [r for r in ORI_TO_G1_BODY if r not in ori_to_idx]
    if missing_ori:
        raise RuntimeError(f"Canonical missing required orientation roles: {missing_ori}")

    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)

    role_to_body_id = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                       for role, name in ROLE_TO_G1_BODY.items()}
    ori_role_to_body_id = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                          for role, name in ORI_TO_G1_BODY.items()}
    for role, bid in {**role_to_body_id, **ori_role_to_body_id}.items():
        assert bid >= 0, f"G1 body for role {role!r} not found"

    leg_geom_ids_by_eff = None
    if args.floor_leg_refine:
        body_name_to_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b): b
                           for b in range(model.nbody)}
        leg_geom_ids_by_eff = {}
        for eff, names in LEG_BODY_NAMES.items():
            bids = {body_name_to_id[n] for n in names}
            leg_geom_ids_by_eff[eff] = [
                g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bids
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)
            ]

    leg_cont_dofs = ([_joint_dofadr(model, n) for n in LEG_CONT_JOINTS]
                     if args.swing_clear else None)

    # One-sided knee-flexion bias (see solve_frame_position_ik's docstring in
    # solve_fbx_canonical_alex_contactfirst.py). min_flex is relative to THIS
    # joint's own lower limit, not an absolute radian value -- G1's knee range
    # is [-0.087, 2.88] (straight = lower limit, not q=0 like Alex's KNEE_Y).
    knee_bias = None
    knee_bias_by_side = None  # {"left_ankle": (qadr, dofadr), "right_ankle": (qadr, dofadr)}
    if args.knee_bias_weight > 0.0:
        entries = []
        side_entry = {}
        for jname, foot_role in (("left_knee_joint", "left_ankle"),
                                 ("right_knee_joint", "right_ankle")):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            assert jid >= 0, f"G1 knee joint {jname!r} not found"
            qadr = int(model.jnt_qposadr[jid])
            dofadr = int(model.jnt_dofadr[jid])
            lo = float(model.jnt_range[jid][0])
            entries.append((qadr, dofadr, lo))
            side_entry[foot_role] = (qadr, dofadr)
        # single scalar min_flex shared by both knees, per Alex's convention --
        # only valid if both knees share the same lower limit (true for G1,
        # a symmetric model); assert rather than silently assume.
        los = {lo for _, _, lo in entries}
        assert len(los) == 1, f"asymmetric knee lower limits: {entries}"
        min_flex = los.pop() + np.radians(args.knee_min_flex_deg)
        knee_bias = ([(qadr, dofadr) for qadr, dofadr, _ in entries], min_flex, args.knee_bias_weight)
        if args.knee_bias_skip_held:
            knee_bias_by_side = (side_entry, min_flex, args.knee_bias_weight)

    if persisted_contacts is None:
        raise RuntimeError(f"{args.canonical} has no persisted contact_flags -- "
                           "run scripts/ground_canonical_human.py first (S2-T2).")
    contacts = persisted_contacts
    eff_names = persisted_eff_names
    print(f"  [contacts] persisted labels: {eff_names}")

    frame_ids = list(range(0, src_positions.shape[0], args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]
    fidx = np.asarray(frame_ids)

    contacts_solved = {eff: debounce_flags(contacts[eff][fidx], args.contact_min_run)
                      for eff in eff_names}

    # Ramp cross-fade envelope for the pull-to-floor Z-blend (planLogGMR.md
    # S2-T5, fix attempt 5): `ramp_envelope` (imported unchanged from
    # contact_labels.py -- already used elsewhere in this codebase for contact
    # transitions) turns the boolean zone into a [0,1] weight per frame, cosine
    # ramped in/out over `--contact-ramp` frames (+ `--contact-preroll` frames
    # of anticipation before the raw zone even starts). Precomputed once here
    # (needs the FULL timeline per effector), indexed per-frame in the loop.
    zone_env = {eff: ramp_envelope(contacts_solved[eff], args.contact_ramp, args.contact_preroll)
               for eff in eff_names}

    # Per-frame "planted" mask (still-plant, same convention throughout this
    # project): contact-point body speed < plant_speed AND in-contact-envelope.
    # Covers feet AND hands (CONTACT_POS_ROLE) -- see S2-T5 note above.
    contact_pt_speed = {}
    for eff, role in CONTACT_POS_ROLE.items():
        if eff not in eff_names:
            continue  # e.g. a canonical NPZ without hand contact labels
        # placeholder, filled after the solve loop warms up -- speed needs the
        # SOLVED trajectory, so plant detection here uses the SOURCE (human)
        # marker speed instead (a reasonable proxy: a still human contact point
        # implies a still target, independent of the not-yet-solved robot pose).
        src_pt = src_positions[fidx, role_to_idx[role]]
        v = np.zeros(len(frame_ids))
        v[1:] = np.linalg.norm(np.diff(src_pt, axis=0), axis=1) * fps
        v[0] = v[1] if len(v) > 1 else 0.0
        contact_pt_speed[eff] = v

    first_src_pos = src_positions[frame_ids[0]]
    alex_pelvis_to_head = measure_alex_pelvis_to_head(model, data, role_to_body_id)
    root_scale = estimate_source_scale(first_src_pos, role_to_idx, alex_pelvis_to_head)

    print(f"Canonical: {args.canonical}")
    print(f"Frames: {len(frame_ids)}  stride: {args.stride}  fps: {fps}")
    print(f"G1 pelvis-to-head: {alex_pelvis_to_head:.4f} m  root_scale: {root_scale:.4f}")

    coll_kwargs = dict(coll_weight=args.coll_weight, coll_margin=args.coll_margin,
                       coll_gain=args.coll_gain, coll_hops=args.coll_hops,
                       floor_gid=floor_gid)

    q = np.zeros(model.nq)
    q[3] = 1.0
    print("Solving initial G1 rest-alignment pose...")
    initial_targets = make_initial_alignment_targets(first_src_pos, role_to_idx, root_scale)

    # FLOOR-REFERENCED REST ANCHOR (planLogGMR.md S2-T6/N1-b root cause + Prabin's follow-up
    # question, 2026-07-16): `make_initial_alignment_targets` (imported UNCHANGED from Alex's
    # solver) sets pelvis's one-time rest target to literal world-origin [0,0,0] -- not floor-
    # referenced, not G1's natural standing height, just the coordinate origin. Confirmed by
    # direct measurement this drags G1's whole leg chain to ~-0.5 to -0.7m below true floor
    # (G1's own kinematic definition has the leg extending that far below the pelvis-as-origin
    # by construction), and EVERY subsequent frame's target inherits this same offset via
    # `make_targets_for_frame`'s additive delta formula:
    #   pelvis_target(t) = target_rest_positions[pelvis] + root_scale*(human_pelvis(t)-human_pelvis(0))
    # Since target_rest_positions[pelvis] comes from solving TOWARD initial_targets[pelvis], and
    # Stage 2.5 already grounds the HUMAN data to floor=0, setting
    #   initial_targets[pelvis]_z = root_scale * human_pelvis_z(0)   (already floor-referenced)
    # makes the formula above telescope to EXACTLY:
    #   pelvis_target_z(t) = root_scale * human_pelvis_z(t)
    # for the WHOLE clip -- i.e. G1's pelvis height becomes directly proportional to the human's
    # OWN real, already-grounded floor-referenced height at every frame, for free, with no
    # per-frame correction needed anywhere else. Applying the SAME uniform Z-shift to every OTHER
    # role's initial target (not just pelvis) keeps the initial pose internally consistent (every
    # role was computed as an OFFSET from pelvis's own target -- shifting only pelvis would
    # ask the solver to satisfy contradictory absolute positions for the same rest pose).
    # Does NOT modify the shared Alex function -- pure glue-code post-processing of its returned
    # dict, this file's own established convention for G1-specific adjustments.
    pelvis_floor_z0 = root_scale * float(first_src_pos[role_to_idx["pelvis"]][2])
    z_shift = np.array([0.0, 0.0, pelvis_floor_z0])
    initial_targets = {role: t + z_shift for role, t in initial_targets.items()}
    print(f"  Floor-referenced rest anchor: pelvis initial target Z = {pelvis_floor_z0:.4f} m "
          f"(root_scale={root_scale:.4f} x human pelvis z(0)="
          f"{first_src_pos[role_to_idx['pelvis']][2]:.4f} m)")
    # BUG FOUND + FIXED (planLogGMR.md S2-T3): `root_reg` (solve_frame_position_ik's
    # kwarg) is DEAD -- never referenced in the function body; posture_reg's
    # desired_dq explicitly zeroes DOFs 0-5 (the free root), so root ORIENTATION
    # gets zero regularization from posture_reg by design ("position/orientation
    # tasks steer it" per that function's own comment). With NO orientation
    # targets at all, this initial call's root is free to rotate arbitrarily --
    # confirmed by direct measurement: it drifted to a ~74-degree-rotated
    # quaternion, collapsing the whole upper body into a folded heap (rendered
    # and visually confirmed). Fix: give it identity orientation targets for
    # pelvis/torso/head (the same 3 roles used at rest -- "stay upright") so the
    # ori-task rows anchor root rotation exactly like the per-frame loop's real
    # orientation targets do later.
    # Extended to ALL 7 ORI_TO_G1_BODY roles (not just pelvis/torso/head) after
    # a second, related bug: with only 3 roles anchored, "left_foot"/"right_foot"
    # had a POSITION target on the ankle body but NO orientation target -- 3 free
    # rotational DOFs the solver could spend on anything. Confirmed by direct
    # measurement: left_ankle_pitch_joint landed EXACTLY at its 0.5236 rad
    # (30 deg) dorsiflexion limit after this very first solve, before any
    # per-frame tracking -- the same "unconstrained redundant DOF drifts to an
    # extreme" failure mode as the root-rotation bug above, just on the ankle
    # instead of the root. Identity is a defensible "at-rest" reference for all
    # 7 roles here (same logic as pelvis/torso/head): this call's only job is to
    # produce a plausible achieved-rest baseline for morphology scaling, not to
    # match the human's specific first-frame pose.
    identity_ori_targets = {r: np.eye(3) for r in ORI_TO_G1_BODY}
    q = solve_frame_position_ik(model, data, role_to_body_id, initial_targets, q,
                                ori_role_to_body_id=ori_role_to_body_id,
                                ori_targets=identity_ori_targets, ori_scale=1.0,
                                knee_bias=knee_bias,
                                iters=max(args.ik_iters * 3, 80), **coll_kwargs)
    mujoco.mj_forward(model, data)

    target_rest_positions = {role: data.xpos[bid].copy() for role, bid in role_to_body_id.items()}
    role_scales = gmr_grouped_role_scales(role_to_body_id)
    print("  role_scales (GMR's own grouped constants): "
          + ", ".join(f"{r}={s}" for r, s in sorted(role_scales.items())))
    target_rest_orientations = {role: body_xmat(data, bid).copy()
                               for role, bid in ori_role_to_body_id.items()}
    first_src_ori = orientation_mats[frame_ids[0]]

    qpos_list = []
    frame_cache = [] if args.floor_leg_refine else None  # S4-T3: (targets, ori_targets, hold_pos_roles) per i
    offset_ema = {eff: None for eff in CONTACT_POS_ROLE}
    for i, t in enumerate(frame_ids):
        targets = make_targets_for_frame(src_positions[t], role_to_idx, first_src_pos,
                                         target_rest_positions, root_scale, role_scales)
        ori_targets = make_orientation_targets_for_frame(
            orientation_mats[t], ori_to_idx, first_src_ori, target_rest_orientations)

        hold_pos_roles = set()
        for eff, role in CONTACT_POS_ROLE.items():
            if eff in contacts_solved and contacts_solved[eff][i]:
                if contact_pt_speed[eff][i] < args.plant_speed:
                    hold_pos_roles.add(role)

        # Pull-to-floor (found necessary by direct measurement, planLogGMR.md
        # S2-T4/S2-T5): `data` still holds the PREVIOUS frame's solved pose here
        # (this frame's solve hasn't run yet) -- the warm-start orientation a
        # still plant is assumed to keep (weaker assumption for a moving-through-
        # zone frame, but still a better Z estimate than the raw morphology
        # delta). For each role, measure how far its body ORIGIN currently sits
        # above its OWN mesh-exact support point at that warm-start orientation,
        # then blend the target's Z toward exactly that offset -- so if the
        # solve lands close to the warm start, the support point ends up at
        # z=0, not wherever the morphology-scaled delta happened to put it.
        # X,Y are left as the morphology-scaled target (unchanged) -- only the
        # floor-relative Z is corrected.
        #
        # EXTENDED to the full contact ZONE, not just the stricter still/held
        # subset (`hold_pos_roles`) -- found necessary by direct measurement
        # (planLogGMR.md S2-T5): with pull-to-floor applied ONLY to held
        # frames (~18% of the clip on fallAndGetUp2_subject2), the other ~82%
        # stayed just as un-grounded as before (whole-clip median lowest-point
        # -62cm), so the single whole-clip grounding shift afterward sized
        # itself to fix THAT majority and overshot, dragging the now-correct
        # held frames 50-90cm away from the floor as collateral damage.
        #
        # SMOOTHED (fix attempt 4): a raw per-frame offset snapshot was fine
        # for STILL frames but too noisy for a MOVING limb passing through the
        # zone -- EMA-smoothed per effector, reset at zone onset.
        #
        # RAMP CROSS-FADED (fix attempt 5, this pass): fix attempt 4 didn't
        # touch the DOMINANT spike source -- 70% of remaining spikes landed
        # within 1 frame of a zone onset/offset boundary, because the Z-target
        # was a HARD SWITCH between "pure morphology delta" and "pull-to-floor
        # offset" the instant a zone began/ended. Fix: blend by `zone_env`
        # (the SAME `ramp_envelope` cosine cross-fade already used elsewhere in
        # this codebase for contact transitions, precomputed above) instead of
        # a boolean gate -- env=0 outside the zone (pure morphology delta),
        # env=1 in the zone's core (pure pull-to-floor), smoothly ramped
        # in/out over `--contact-ramp` frames at the edges (+ `--contact-
        # preroll` frames of anticipation, so the EMA has time to initialize
        # before the ramp reaches full weight).
        if args.pull_to_floor:
            for eff, role in CONTACT_POS_ROLE.items():
                if eff not in zone_env:
                    continue
                env = float(zone_env[eff][i])
                if env <= 0.0:
                    offset_ema[eff] = None  # fully outside -- next onset starts fresh
                    continue
                bid = role_to_body_id[role]
                origin_z_now = float(data.xpos[bid][2])
                support_z_now = support_z(model, data, mesh_cache, bid)
                raw_offset = origin_z_now - support_z_now
                if offset_ema[eff] is None:  # entering the ramp -- start fresh, no history
                    offset_ema[eff] = raw_offset
                else:
                    offset_ema[eff] = (args.pull_to_floor_alpha * raw_offset
                                       + (1.0 - args.pull_to_floor_alpha) * offset_ema[eff])
                targets[role] = targets[role].copy()
                targets[role][2] = env * offset_ema[eff] + (1.0 - env) * targets[role][2]

        frame_knee_bias = knee_bias
        if knee_bias_by_side is not None:
            side_entry, min_flex, kb_weight = knee_bias_by_side
            active = [e for role, e in side_entry.items() if role not in hold_pos_roles]
            frame_knee_bias = (active, min_flex, kb_weight) if active else None

        # Swing-foot toe-clearance (S4-T4, --swing-clear -- see that flag's
        # help text for the diagnosis this replaces --floor-leg-refine with,
        # for locomotion clips specifically). Cap the swing foot's orientation
        # TARGET toe-down pitch (spends unused ankle dorsiflexion headroom
        # instead of copying the human's plantarflexed step into the floor),
        # ramped by swing-ness (1 - zone_env, the SAME envelope already used
        # for pull-to-floor above -- 0 planted, 1 fully airborne). No
        # proximity gate (see flag help). posture_reg gets a matching
        # temporal-continuity boost on hip/knee (not ankle) to stop the
        # redundant leg from branch-flipping when the cap engages.
        frame_posture_reg = 1e-3
        if args.swing_clear:
            swing_depitch_lift = 0.0
            for eff in FOOT_POS_ROLE:
                lift = 1.0 - float(zone_env[eff][i])
                if lift <= 0.0:
                    continue
                ori_targets[eff] = cap_foot_pitch(
                    ori_targets[eff], np.radians(args.swing_max_pitch), lift)
                swing_depitch_lift = max(swing_depitch_lift, lift)
            if swing_depitch_lift > 0.0:
                frame_posture_reg = _swing_posture_reg(
                    model.nv, leg_cont_dofs, args.swing_continuity_reg * swing_depitch_lift)

        q = solve_frame_position_ik(
            model, data, role_to_body_id, targets, q,
            ori_role_to_body_id=ori_role_to_body_id, ori_targets=ori_targets, ori_scale=1.0,
            hold_pos_roles=hold_pos_roles, knee_bias=frame_knee_bias,
            iters=args.ik_iters, floor_weight=args.floor_weight, posture_reg=frame_posture_reg,
            **coll_kwargs)
        clamp_hinge_joint_limits(model, q)  # mutates q in-place, no return value
        mujoco.mj_forward(model, data)
        qpos_list.append(q.copy())
        if frame_cache is not None:
            frame_cache.append((targets, ori_targets, set(hold_pos_roles)))

        if i % 200 == 0 or i == len(frame_ids) - 1:
            print(f"  frame {i+1}/{len(frame_ids)} (src {t})")

    qpos_arr = np.stack(qpos_list, axis=0)

    if args.floor_leg_refine:
        print("Leg-floor-transition refine pass (S4-T3)...")
        qpos_arr = refine_leg_floor_transitions_g1(
            model, data, qpos_arr, frame_cache, role_to_body_id, ori_role_to_body_id,
            mesh_cache, leg_geom_ids_by_eff, coll_kwargs,
            floor_weight=args.floor_refine_weight, floor_margin=args.floor_refine_margin,
            floor_gain=args.floor_refine_gain, pen_tol=args.floor_refine_pen_tol,
            ramp=args.floor_refine_ramp, preroll=args.floor_refine_preroll,
            leg_posture_reg=args.floor_refine_posture_reg,
            lock_weight=args.floor_refine_lock_weight,
            root_pos_relief=args.floor_refine_root_relief, iters=args.ik_iters,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, qpos=qpos_arr, fps=np.float64(fps),
        source_frame_ids=fidx,
        role_names=np.asarray(list(ROLE_TO_G1_BODY.keys()), dtype=object),
        g1_body_names=np.asarray(list(ROLE_TO_G1_BODY.values()), dtype=object),
    )
    print(f"Wrote {args.out} ({qpos_arr.shape[0]} frames)")


if __name__ == "__main__":
    main()

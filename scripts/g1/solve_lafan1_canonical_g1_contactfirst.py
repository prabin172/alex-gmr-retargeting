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
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import (  # noqa: E402
    body_xmat, clamp_hinge_joint_limits,
    estimate_source_scale, load_canonical, make_initial_alignment_targets,
    make_orientation_targets_for_frame, make_targets_for_frame,
    measure_alex_pelvis_to_head, solve_frame_position_ik)
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
    pelvis tracks via root_scale, not a role scale, at all times)."""
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

        q = solve_frame_position_ik(
            model, data, role_to_body_id, targets, q,
            ori_role_to_body_id=ori_role_to_body_id, ori_targets=ori_targets, ori_scale=1.0,
            hold_pos_roles=hold_pos_roles,
            iters=args.ik_iters, floor_weight=args.floor_weight,
            **coll_kwargs)
        clamp_hinge_joint_limits(model, q)  # mutates q in-place, no return value
        mujoco.mj_forward(model, data)
        qpos_list.append(q.copy())

        if i % 200 == 0 or i == len(frame_ids) - 1:
            print(f"  frame {i+1}/{len(frame_ids)} (src {t})")

    qpos_arr = np.stack(qpos_list, axis=0)
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

#!/usr/bin/env python3
"""S5-A1: GMR's own retargeter (mink QP, orientation-first, hard joint limits) +
a minimal per-frame contact layer for held (planted) feet. GMR itself has ZERO
per-frame contact handling (its only floor mechanism is one constant per-clip Z
offset, `set_ground_offset` -- same mechanism the S3 z-shift oracle already showed
is gameable). This wraps `GeneralMotionRetargeting` (never edits it -- read-only
reference at ~/projects/GMR) and overrides ONLY the held foot's table-2 FrameTask
target+cost, ramped in/out, everything else left exactly as GMR computes it.

Held mask: debounced human contact flag (this project's own canonical schema,
`contact_flags`) AND human foot speed <0.05 m/s -- IDENTICAL recipe to
`sprint_s3_full_corpus.py::do_eval`'s held mask, so eval numbers are comparable.

Contact override (when held, ramped by `ramp_envelope`, GMR-S5-plan.md A1):
  position target: XY locked at the robot's OWN FK position at hold onset (kills
    skate), Z = z_sole[foot] (this robot's own ankle-origin-above-sole constant,
    computed once from model.qpos0 -- makes the sole land exactly on the floor).
  orientation target: flat foot (roll=pitch=0), yaw = yaw component of GMR's OWN
    current target for that foot (so it doesn't fight the natural facing direction).
  costs: ramp position 50->200, orientation 10->50 (GMR's foot table2 weights are
    50/10 -- this raises them, not replaces them, while the ramp is mid-transition).

Sanity check: `--no-contact` runs the wrapper with the override fully disabled; its
output must byte-match a plain `gmr_headless_retarget.py` run of the same BVH (run
`--verify-sanity` to check this automatically against a reference pkl).

Usage:
    conda run -n gmr python scripts/g1/gmr_contact_retarget.py \\
        --bvh data/raw/lafan1/walk1_subject1.bvh \\
        --canonical outputs/gmr_baseline/sprint/canonical_human_s5/walk1_subject1_lafan1c_grounded.npz \\
        --save_path outputs/gmr_baseline/sprint/pkl_s5/walk1_subject1_gmrcontact.pkl
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pathlib
import pickle
import sys

import mink
import mujoco
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from contact_labels import debounce_flags  # noqa: E402
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from stage_b_g1 import support_z  # noqa: E402
from leg_floor_clamp import (build_chain_dofs, clamp_limb, CLAMP_TARGETS,  # noqa: E402
                             CorrectionRateLimiter, joint_ranges)

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

# Effector keys match canonical's own `contact_effector_names` exactly
# (left_foot/right_foot/left_hand/right_hand). S5-A1 shipped feet only; S5-A5 adds
# hands (opt-in via --effectors, see main()) for the hard fall/get-up class.
EFF_BODY = {
    "left_foot": "left_ankle_roll_link", "right_foot": "right_ankle_roll_link",
    "left_hand": "left_wrist_yaw_link", "right_hand": "right_wrist_yaw_link",
}
EFF_HUMAN_KEY = {
    "left_foot": "LeftFootMod", "right_foot": "RightFootMod",
    "left_hand": "LeftHand", "right_hand": "RightHand",
}
EFF_CANON_ROLE = {  # canonical role used for the held-speed check
    "left_foot": "left_ankle", "right_foot": "right_ankle",
    "left_hand": "left_wrist", "right_hand": "right_wrist",
}
FEET = ("left_foot", "right_foot")
HANDS = ("left_hand", "right_hand")

RAMP_FRAMES = 5
POS_COST_HELD = 100.0
ORI_COST_HELD = 20.0


def compute_z_support(model, data, mesh_cache, effectors):
    """Height of each effector body origin above its own limb's lowest mesh
    point, at `model.qpos0` (all joints 0 -- flat-footed/neutral-armed stance).
    A rigid per-robot geometric constant, invariant to translating the whole
    robot -- only valid when the limb is "flat" (roll=pitch=0 in the same sense
    as a foot sole), which qpos0 guarantees for feet; for hands this is a
    reasonable v1 approximation (S5-A5), not separately validated."""
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    out = {}
    for eff in effectors:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EFF_BODY[eff])
        out[eff] = float(data.xpos[bid][2] - support_z(model, data, mesh_cache, bid))
    return out


def compute_held_masks(canonical_path: pathlib.Path, effectors):
    """Debounced human contact flag AND human limb speed <0.05 m/s -- identical
    recipe to sprint_s3_full_corpus.py::do_eval's held mask."""
    z = np.load(canonical_path, allow_pickle=True)
    roles = list(z["roles"])
    role_to_idx = {r: i for i, r in enumerate(roles)}
    positions = z["positions"]
    fps = float(z["fps"])
    eff_names = list(z["contact_effector_names"])
    contact_flags = z["contact_flags"]
    T = positions.shape[0]

    held = {}
    for eff in effectors:
        col = eff_names.index(eff)
        contacts_debounced = debounce_flags(contact_flags[:, col].astype(bool), 2)
        src_pt = positions[:, role_to_idx[EFF_CANON_ROLE[eff]]]
        v = np.zeros(T)
        v[1:] = np.linalg.norm(np.diff(src_pt, axis=0), axis=1) * fps
        v[0] = v[1] if T > 1 else 0.0
        held[eff] = contacts_debounced & (v < 0.05)
    return held, T


class ContactAwareGMR(GMR):
    """Subclass, never edits ~/projects/GMR. Overrides ONLY the held foot's
    table-2 FrameTask when `set_frame_contact(held_left, held_right)` says so;
    everything else (table1, all other bodies, GMR's own scaling/offset/ground
    logic) is untouched -- calls `super().update_targets` first, every time."""

    def __init__(self, *args, effectors=FEET, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact_enabled = False
        self._solved_once = False  # see update_targets docstring
        self._ramp_frame_count = RAMP_FRAMES  # overridable, see main()
        self.effectors = tuple(effectors)
        self._z_support = {eff: 0.0 for eff in self.effectors}
        self._held_prev = {eff: False for eff in self.effectors}
        self._onset_xy = {eff: None for eff in self.effectors}
        self._ramp_age = {eff: 0 for eff in self.effectors}  # frames since onset (or
                                    # since release started, counting down), 0..ramp_frames
        self.swing_floor_margin = None  # None = off; see --swing-floor-margin (S5-A3,
                                          # FEET only -- see update_targets)

    def set_z_support(self, z_support: dict):
        self._z_support = dict(z_support)

    @staticmethod
    def _cosramp(age, ramp):
        """0..1 with ZERO derivative at both ends (matches contact_labels.py's
        ramp_envelope cosine shape) -- age in [0, ramp]. Found in A2 (S5-A2) that a
        plain linear step here left visible jerk at onset/release (body_jerk +58 to
        +75% vs gmr_raw, failing the gate's <=20% bar); this is the fix."""
        if ramp <= 0:
            return 1.0
        age = max(0, min(age, ramp))
        return 0.5 * (1.0 - np.cos(np.pi * age / ramp))

    def update_targets(self, human_data, offset_to_ground=False, held=None):
        # GMR's own update_targets() has no return statement (implicit None) --
        # its real output is the side effect of setting every task's target via
        # self.human_body_to_task{1,2}, which is all this override needs.
        super().update_targets(human_data, offset_to_ground)

        if not self.contact_enabled or held is None:
            return

        if not self._solved_once:
            # Found in A2 eval (walk1_subject1, S5-A2): a clip that's ALREADY held
            # at frame 0 would lock onset XY from self.configuration's pre-solve
            # default/rest pose (no retarget() has run yet) -- garbage, ~2m from
            # the human's real start position. Skip the override entirely until
            # the driver marks one full solve done (`retargeter._solved_once =
            # True`, set in main()'s loop after the first frame's solve); frame 0
            # runs pure natural GMR tracking regardless of held state, and the
            # first REAL onset lock happens no earlier than frame 1, using a
            # legitimate FK position.
            return

        for eff in self.effectors:
            is_held = bool(held[eff])
            body_name = EFF_BODY[eff]
            human_key = EFF_HUMAN_KEY[eff]
            task2 = self.human_body_to_task2[human_key]
            bid = self.robot_body_names[body_name]

            if is_held and not self._held_prev[eff]:
                # Onset: lock XY at the robot's OWN FK position (prev frame's
                # already-solved pose -- this frame's solve hasn't run yet).
                self._onset_xy[eff] = self.configuration.data.xpos[bid][:2].copy()
            self._held_prev[eff] = is_held

            # Cosine-ramped "age" INSIDE the held region, counted from onset (not
            # ramp_envelope's pre-onset anticipation -- there's no legitimate onset_xy
            # to anticipate toward before the onset frame itself has run).
            ramp = self._ramp_frame_count
            if is_held:
                self._ramp_age[eff] = min(ramp, self._ramp_age[eff] + 1)
            else:
                self._ramp_age[eff] = max(0, self._ramp_age[eff] - 1)
            frac = self._cosramp(self._ramp_age[eff], ramp)
            if frac <= 0.0:
                # S5-A3: swing-floor clamp (FEET only -- diagnosed on run2_subject1,
                # never validated for hands). 100% of run2's joint_ok_pct failures
                # were whole-body pen from the CURRENTLY NOT-held ankle (confirmed by
                # direct check: 10 right + 7 left ankle hits, zero other bodies). GMR
                # itself has no per-frame floor avoidance at all -- this is a soft,
                # target-space-only clamp on the swing foot's own Z (leaves XY/
                # orientation/cost untouched, so it can't fight swing clearance the
                # way a hard task would).
                if eff in FEET and self.swing_floor_margin is not None:
                    gmr_target = task2.transform_target_to_world
                    cur_pos = gmr_target.translation().copy()
                    floor_z = self._z_support[eff] + self.swing_floor_margin
                    if cur_pos[2] < floor_z:
                        cur_pos[2] = floor_z
                        task2.set_target(mink.SE3.from_rotation_and_translation(
                            gmr_target.rotation(), cur_pos))
                continue  # fully released: leave GMR's own natural cost alone

            # Flat orientation, yaw taken from GMR's OWN current target (already set
            # by super().update_targets above) so it doesn't fight facing direction.
            # "Flat" for a hand (S5-A5) is an unvalidated v1 approximation -- a
            # planted palm's real orientation constraint is more like "normal points
            # into the floor," which isn't generally the same as zero roll/pitch;
            # not corrected this pass, logged as a known limitation.
            gmr_target = task2.transform_target_to_world
            yaw = float(gmr_target.rotation().compute_yaw_radians())
            flat_rot = mink.SO3.from_z_radians(yaw)
            xy = self._onset_xy[eff] if self._onset_xy[eff] is not None \
                else self.configuration.data.xpos[bid][:2]
            locked_pos = np.array([xy[0], xy[1], self._z_support[eff]])
            task2.set_target(mink.SE3.from_rotation_and_translation(flat_rot, locked_pos))

            # GMR's own table2 weight is the base (feet 50/10, hands 10/5 --
            # bvh_lafan1_to_g1.json) -- ramp toward the SAME elevated ceiling
            # regardless of effector (not separately tuned for hands this pass).
            base_pos_cost, base_ori_cost = (50.0, 10.0) if eff in FEET else (10.0, 5.0)
            task2.set_position_cost(base_pos_cost + frac * (POS_COST_HELD - base_pos_cost))
            task2.set_orientation_cost(base_ori_cost + frac * (ORI_COST_HELD - base_ori_cost))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh_file", required=True, type=str)
    ap.add_argument("--canonical", required=True, type=pathlib.Path,
                    help="This project's canonical-human npz for the SAME clip -- source "
                         "of the held-foot mask (contact_flags + ankle speed).")
    ap.add_argument("--format", choices=["lafan1", "nokov"], default="lafan1")
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--motion_fps", type=int, default=30)
    ap.add_argument("--save_path", required=True, type=str)
    ap.add_argument("--video_path", type=str, default=None)
    ap.add_argument("--video_width", type=int, default=640)
    ap.add_argument("--video_height", type=int, default=480)
    ap.add_argument("--save_human_targets", type=str, default=None)
    ap.add_argument("--no-contact", dest="contact", action="store_false", default=True,
                    help="Disable the contact override entirely -- sanity-check mode, "
                         "output must byte-match plain gmr_headless_retarget.py.")
    ap.add_argument("--ramp-frames", type=int, default=RAMP_FRAMES,
                    help="Cosine ramp length (frames) for the held-foot cost/target "
                         "cross-fade, counted from the onset/release frame. Default 5 "
                         "failed the A2 jerk gate (body_jerk +58 to +75%% vs gmr_raw on "
                         "the 3 loco dev clips) -- try longer (10, 15).")
    ap.add_argument("--swing-floor-margin", type=float, default=None,
                    help="S5-A3: soft target-space Z clamp on the NOT-held foot's own "
                         "table-2 target (GMR's own target Z, raised to z_support+margin "
                         "if below it -- XY/orientation/cost untouched, FEET only). "
                         "Diagnosed on run2_subject1: 100%% of joint_ok_pct failures were "
                         "whole-body pen from the swing (not held) foot. Try 0.01-0.02 "
                         "(1-2cm). None (default) = off.")
    ap.add_argument("--effectors", choices=["feet", "feet+hands"], default="feet",
                    help="S5-A5: which limbs get the held-contact override. 'feet' "
                         "(default) = A1's original locomotion scope. 'feet+hands' = "
                         "hard fall/get-up class extension -- unvalidated, expect "
                         "partial success (reach limits are real on this ~0.64-scale "
                         "robot, see planLogGMR.md S2).")
    ap.add_argument("--floor-clamp", action="store_true", default=False,
                    help="S6-A3: exact per-frame floor clamp (leg_floor_clamp.py) "
                         "applied after each frame's solve, on OUR vetted mesh "
                         "geometry -- not a QP inequality (see planLogGMR.md S6-A1 "
                         "for why that path was dropped: rate-limited QP inequality "
                         "inside GMR's few-iteration solve loop never converges "
                         "close to zero). Same effector set as --effectors. "
                         "Composes with --contact/--no-contact.")
    ap.add_argument("--avoid-self-collision", action="store_true", default=False,
                    help="S7-T7: --floor-clamp corrects one limb chain's floor "
                         "clearance with zero awareness of other bodies, which can "
                         "drive a corrected elbow/knee INTO the torso/head on cramped "
                         "floor poses (confirmed: ground1_subject1 self-collision "
                         "coll_pct 2.57%%->13.12%% after --floor-clamp alone). Adds a "
                         "self-collision repulsion term to clamp_limb's DLS solve "
                         "(see leg_floor_clamp.py's clamp_limb docstring). No effect "
                         "unless --floor-clamp is also set.")
    ap.add_argument("--coll-weight", type=float, default=0.5,
                    help="S7-T7 tuning knob, forwarded to clamp_limb's coll_weight. "
                         "0.5 chosen after a dev-clip sweep (1.0/2.0 both caused a "
                         "cascading floor-pen regression on the hardest clips via "
                         "Phase A's warm-starting, see leg_floor_clamp.py).")
    ap.add_argument("--clamp-rate-limit", type=float, default=None,
                    help="S8-T1b: temporal trust region (rad/frame) on the total "
                         "per-frame clamp correction relative to GMR's own solve "
                         "(see leg_floor_clamp.py's CorrectionRateLimiter). The "
                         "rate-limited pose is what gets fed back into GMR's "
                         "warm start. No effect unless --floor-clamp is set. "
                         "None (default) = off, byte-identical to pre-S8.")
    args = ap.parse_args()

    effectors = FEET if args.effectors == "feet" else FEET + HANDS

    for p in (args.save_path, args.video_path, args.save_human_targets):
        if p:
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)

    lafan1_data_frames, actual_human_height = load_bvh_file(args.bvh_file, format=args.format)

    held, T = compute_held_masks(args.canonical, effectors)
    assert T == len(lafan1_data_frames), \
        f"canonical T={T} != BVH frame count={len(lafan1_data_frames)} ({args.canonical})"

    retargeter = ContactAwareGMR(
        src_human=f"bvh_{args.format}", tgt_robot=args.robot,
        actual_human_height=actual_human_height, effectors=effectors,
    )

    # z_support computed on OUR vetted-collision model (mesh-exact, matches every
    # eval this project uses) -- NOT GMR's own model, which has different collision
    # geoms.
    vetted_model, vetted_data, _, _ = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(vetted_model)
    z_support = compute_z_support(vetted_model, vetted_data, mesh_cache, effectors)
    retargeter.set_z_support(z_support)
    print(f"[gmr_contact_retarget] z_support = {z_support}")

    retargeter.contact_enabled = args.contact
    retargeter._ramp_frame_count = args.ramp_frames
    retargeter.swing_floor_margin = args.swing_floor_margin

    # S6-A3/A4: chains built once (not per-frame) on the SAME vetted model/
    # mesh_cache used for z_support and every eval in this project. Floor-clamp
    # ALWAYS builds all 4 limb chains regardless of --effectors -- S6-A4 found
    # the worst whole-body penetration is often NOT a foot/hand at all (elbow
    # during fast arm-swing, hip_yaw during floor-contact clips), so clearance
    # protection must cover the whole lower/upper body, independent of which
    # effectors are held-contact targets.
    clamp_chains = {eff: build_chain_dofs(vetted_model, eff) for eff in FEET + HANDS} \
        if args.floor_clamp else None
    rate_limiter = None
    rl_lo = rl_hi = None
    if args.floor_clamp and args.clamp_rate_limit is not None:
        rate_limiter = CorrectionRateLimiter(args.clamp_rate_limit)
        rl_lo, rl_hi = joint_ranges(vetted_model)
        print(f"[gmr_contact_retarget] S8-T1b clamp rate limit: "
              f"{args.clamp_rate_limit} rad/frame")
    # CLAMP_TARGETS imported from leg_floor_clamp.py -- proximal-to-distal order
    # matters (see that module's comment), do not redefine a local copy here.

    if args.contact:
        held_summary = "  ".join(f"{eff}={held[eff].sum()}/{T}" for eff in effectors)
        print(f"[gmr_contact_retarget] held frames: {held_summary}  "
              f"ramp_frames={args.ramp_frames}")
    else:
        print("[gmr_contact_retarget] --no-contact: sanity-check mode, override disabled")

    qpos_list = []
    frames_rgb = [] if args.video_path else None
    model = data = renderer = cam = robot_base_id = None
    if args.video_path:
        model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
        robot_base_id = model.body(ROBOT_BASE_DICT[args.robot]).id
        cam = mujoco.MjvCamera()
        cam.distance = VIEWER_CAM_DISTANCE_DICT[args.robot]
        cam.elevation = -10
        cam.azimuth = 90

    human_target_pos = None
    human_target_quat = None

    for t, smplx_data in enumerate(tqdm(lafan1_data_frames, desc="Retargeting")):
        frame_held = {eff: bool(held[eff][t]) for eff in effectors} if args.contact else None
        retargeter.update_targets(smplx_data, held=frame_held)

        # update_targets(held=...) was already called above (our subclass-only
        # override point); retarget() itself doesn't take `held`, so run its SOLVE
        # portion directly (table1 then table2, verbatim GMR logic) without calling
        # update_targets a second time.
        qpos = _solve_after_targets(retargeter)

        if args.floor_clamp:
            # Correction runs in OUR vetted-model space (exact mesh/cylinder
            # geometry, matches every eval this project uses -- not GMR's own
            # collision geoms, see leg_floor_clamp.py docstring / planLogGMR.md
            # S6-A1 for why). Same qpos layout (free root + 29 actuated joints),
            # direct assignment is valid.
            qpos_pre_clamp = qpos.copy() if rate_limiter is not None else None
            vetted_data.qpos[:] = qpos
            # S8-T1b attempt 2: when rate-limiting, phase 1 runs WITHOUT inline
            # self-collision (the limiter would cap the collision corrections
            # away -- attempt 1 measured coll_pct regressing to gmr_raw's
            # level); self-collision runs as an UN-limited post-pass below.
            inline_avoid_coll = args.avoid_self_collision and rate_limiter is None
            for eff, watch_body in CLAMP_TARGETS:
                clamp_limb(vetted_model, vetted_data, mesh_cache, eff, clamp_chains[eff],
                           floor_margin=0.0, watch_body=watch_body,
                           avoid_self_collision=inline_avoid_coll,
                           coll_weight=args.coll_weight)
            qpos = vetted_data.qpos.copy()
            if rate_limiter is not None:
                # S8-T1b: bound the clamp's per-frame correction delta vs the
                # previous frame's APPLIED correction (temporal trust region).
                # Reference = GMR's own solve this frame (qpos_pre_clamp); the
                # rate-limited pose is also what feeds back into GMR's warm
                # start below, stabilizing the solve/clamp feedback loop.
                limited = rate_limiter.apply(qpos_pre_clamp[7:], qpos[7:])
                qpos[7:] = np.clip(limited, rl_lo, rl_hi)
                if args.avoid_self_collision:
                    vetted_data.qpos[:] = qpos
                    for eff in FEET + HANDS:
                        clamp_limb(vetted_model, vetted_data, mesh_cache, eff,
                                   clamp_chains[eff], avoid_self_collision=True,
                                   coll_weight=args.coll_weight, collision_only=True)
                    qpos = vetted_data.qpos.copy()
            # Feed the correction back into GMR's own configuration so next
            # frame's warm-start (and S5's onset-XY lock, which reads
            # retargeter.configuration.data.xpos) sees the corrected pose, not
            # the pre-clamp one -- avoids compounding drift between what we
            # solve from and what we actually output.
            retargeter.configuration.data.qpos[:] = qpos
            mujoco.mj_forward(retargeter.model, retargeter.configuration.data)

        retargeter._solved_once = True
        qpos_list.append(qpos)

        if args.save_human_targets:
            shd = retargeter.scaled_human_data
            if human_target_pos is None:
                human_target_pos = {b: [] for b in shd}
                human_target_quat = {b: [] for b in shd}
            for b, (pos, rot) in shd.items():
                human_target_pos[b].append(np.asarray(pos, dtype=np.float64).copy())
                human_target_quat[b].append(np.asarray(rot, dtype=np.float64).copy())

        if args.video_path:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            cam.lookat = data.xpos[robot_base_id]
            renderer.update_scene(data, camera=cam)
            frames_rgb.append(renderer.render().copy())

    root_pos = np.array([q[:3] for q in qpos_list])
    root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])
    dof_pos = np.array([q[7:] for q in qpos_list])

    motion_data = {
        "fps": args.motion_fps, "root_pos": root_pos, "root_rot": root_rot,
        "dof_pos": dof_pos, "local_body_pos": None, "link_body_list": None,
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved {len(qpos_list)} frames to {args.save_path}")

    if args.save_human_targets:
        save_dict = {}
        for b in human_target_pos:
            save_dict[f"pos__{b}"] = np.stack(human_target_pos[b], axis=0)
            save_dict[f"rot__{b}"] = np.stack(human_target_quat[b], axis=0)
        save_dict["body_names"] = np.array(list(human_target_pos.keys()))
        np.savez_compressed(args.save_human_targets, **save_dict)
        print(f"Saved human targets to {args.save_human_targets}")

    if args.video_path:
        import imageio
        imageio.mimwrite(args.video_path, frames_rgb, fps=args.motion_fps)
        print(f"Saved video to {args.video_path}")


def _solve_after_targets(retargeter):
    """Copy of GeneralMotionRetargeting.retarget()'s SOLVE portion (table1 then
    table2, same convergence loop), assuming update_targets() was already called
    separately (our subclass needs the `held` kwarg threaded through
    update_targets, which retarget() itself doesn't accept). ONE deliberate
    deviation from GMR's own code (S6-A1, planLogGMR.md): GMR calls
    `mink.solve_ik(configuration, tasks, dt, solver, damping, self.ik_limits)` --
    6 positional args. The installed mink's solve_ik signature has `safety_break`
    at position 6 and `limits` at position 7, so GMR's own call silently binds
    `ik_limits` to `safety_break` and NEVER passes it as `limits` -- confirmed via
    direct QP diff (build_ik with limits=None vs limits=r.ik_limits: nonzero dq).
    This is a bug in GMR's reference code, not in our copy of it; fixing it here
    (passing `limits=` as a keyword) only affects OUR ContactAwareGMR class, never
    GMR's own retarget() or the gmr_raw/gmr_heightfix baseline-generation path
    (gmr_headless_retarget.py calls GMR's retarget() directly, untouched -- see
    "Baseline integrity" in GMR-S6-plan.md). Without this fix, any limit WE append
    to retargeter.ik_limits (S6-A's --floor-clamp does not rely on ik_limits at
    all, but a future QP-based mechanism would) is silently dropped too."""
    r = retargeter
    if r.use_ik_match_table1:
        curr_error = r.error1()
        dt = r.configuration.model.opt.timestep
        vel1 = mink.solve_ik(r.configuration, r.tasks1, dt, r.solver, r.damping, limits=r.ik_limits)
        r.configuration.integrate_inplace(vel1, dt)
        next_error = r.error1()
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < r.max_iter:
            curr_error = next_error
            dt = r.configuration.model.opt.timestep
            vel1 = mink.solve_ik(r.configuration, r.tasks1, dt, r.solver, r.damping, limits=r.ik_limits)
            r.configuration.integrate_inplace(vel1, dt)
            next_error = r.error1()
            num_iter += 1

    if r.use_ik_match_table2:
        curr_error = r.error2()
        dt = r.configuration.model.opt.timestep
        vel2 = mink.solve_ik(r.configuration, r.tasks2, dt, r.solver, r.damping, limits=r.ik_limits)
        r.configuration.integrate_inplace(vel2, dt)
        next_error = r.error2()
        num_iter = 0
        while curr_error - next_error > 0.001 and num_iter < r.max_iter:
            curr_error = next_error
            dt = r.configuration.model.opt.timestep
            vel2 = mink.solve_ik(r.configuration, r.tasks2, dt, r.solver, r.damping, limits=r.ik_limits)
            r.configuration.integrate_inplace(vel2, dt)
            next_error = r.error2()
            num_iter += 1

    return r.configuration.data.qpos.copy()


if __name__ == "__main__":
    main()

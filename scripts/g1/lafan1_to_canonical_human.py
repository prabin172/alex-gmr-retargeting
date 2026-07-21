#!/usr/bin/env python3
"""S2-T1: LAFAN1 BVH -> our canonical-human NPZ schema (the Stage-3 input format).

Schema verified against a REAL Stage-3 input NPZ (`outputs/canonical_human/fbx_fresh/
shovel_fronthard_02_with_orient.npz`), not guessed: `roles (R,)`, `positions (T,R,3)`,
`frames (T,)`, `fps ()`, `orientation_role_names (7,)`, `orientation_mats (T,7,3,3)`,
`orientation_valid (7,)`, `facing_yaw_correction_deg ()`, `metadata_json ()`.
`segment_names`/`segment_role_pairs`/`segment_lengths_*` are NOT produced -- confirmed
by grep that Stage 3's morphology scaling is computed inline from `positions`/`roles`
alone (wiki/concepts/morphology-scaling.md), never reads segment fields.

GMR's own `load_bvh_file` per-frame `{bone: [pos, quat]}` dicts ARE our canonical-human
shape already (GMR-baseline.md SS6 Q3's unlock) -- this script is the role-name mapping
+ orientation-frame construction on top.

Role mapping (LAFAN1 bone -> canonical role), 20 of Alex's 24 roles populated. Omitted
(confirmed NOT load-bearing -- absent from ROLE_TO_ALEX_BODY/ORI_TO_ALEX_BODY/
CONTACT_EFFECTORS/CONTACT_POS in solve_fbx_canonical_alex_contactfirst.py, zero grep
hits): left/right_toe_end, left/right_hand_thumb.

    pelvis <- Hips                  left_hip <- LeftUpLeg
    torso  <- Spine2 (matches GMR's own ik_config choice)
    neck   <- Neck                  left_knee <- LeftLeg
    head   <- Head                  left_ankle <- LeftFoot
                                     left_toe <- LeftToe
    left_shoulder <- LeftArm (matches GMR ik_config; LAFAN1's "LeftShoulder" is the
                              clavicle base, GMR itself uses "LeftArm" = true shoulder)
    left_elbow <- LeftForeArm       left_wrist <- LeftHand
    left_hand_middle <- LeftHand (SAME bone as left_wrist -- KNOWN SIMPLIFICATION,
                              LAFAN1 has no separate hand-centroid bone; this makes
                              CONTACT_POS's palm-vs-wrist delta always ~0 for our G1
                              adapter. Logged, not silently absorbed.)
    (right_* mirrors left_*)

Orientation frames (7 roles: pelvis/torso/head/left_foot/right_foot/left_hand/
right_hand): reuses `frame_from_yz`/`frame_from_xy`/`detect_facing_yaw_deg`/
`apply_yaw_to_positions` UNCHANGED from `build_canonical_orientation_frames_fresh.py`
(pure geometry functions, no Alex globals) for pelvis/torso/head/feet exactly as Stage 2
does.

Hands (S5-B1 fix, planLogGMR.md ## S5-B0.1/B1): ORIGINALLY used `frame_from_xy(wrist -
elbow, pelvis_y)` -- primary axis (forearm direction) was real motion, but the secondary
axis (pelvis_y, a near-rigid reference) fixed the frame's ROLL about that axis to a
geometric artifact, structurally unable to represent forearm twist/pronation-supination.
Diagnostic (`## S5-B0.1`) confirmed this against GMR's own hand target: residual after
removing the best single clip-constant offset was 43.5/49.9 deg mean (91.9/76.0 p90) --
not a fixed-frame-convention mismatch, a genuinely missing DOF. FIX: use the RAW BVH bone
orientation directly (`load_bvh_file`'s own `f["LeftHand"][1]` / `f["RightHand"][1]`,
already world-frame wxyz FK'd quaternions -- the exact same signal GMR's own IK targets
`LeftHand`/`RightHand` with, per `bvh_lafan1_to_g1.json`). LAFAN1 is a single clean mocap
rig (not a vendor-varying FBX bind-pose rig), so CLAUDE.md's "semantic frames, not raw
FBX rotations" rule (written for the Alex/FBX pipeline's vendor bind-pose problem) does
not apply here -- GMR's own SOTA pipeline already validates raw BVH bone rotation as
reliable for this exact task. Feet/pelvis/torso/head UNCHANGED (still landmark-derived
semantic frames -- this fix is hands-only, where the missing-twist problem lives).

Usage:
    conda run -n gmr python scripts/g1/lafan1_to_canonical_human.py \\
        --bvh data/raw/lafan1/walk1_subject1.bvh \\
        --out outputs/gmr_baseline/sprint/canonical_human/walk1_subject1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # NOT repo root -- see planLogGMR.md T1
from scipy.spatial.transform import Rotation as _Rot  # noqa: E402

from build_canonical_orientation_frames_fresh import (  # noqa: E402
    apply_yaw_to_positions, detect_facing_yaw_deg, frame_from_xy, frame_from_yz, yaw_matrix)
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

ORIENTATION_ROLES = ["pelvis", "torso", "head", "left_foot", "right_foot",
                     "left_hand", "right_hand"]

# canonical role -> LAFAN1 bone name
ROLE_TO_LAFAN1_BONE = {
    "pelvis": "Hips", "torso": "Spine2", "neck": "Neck", "head": "Head",
    "left_hip": "LeftUpLeg", "left_knee": "LeftLeg", "left_ankle": "LeftFoot",
    "left_toe": "LeftToe", "left_shoulder": "LeftArm", "left_elbow": "LeftForeArm",
    "left_wrist": "LeftHand", "left_hand_middle": "LeftHand",
    "right_hip": "RightUpLeg", "right_knee": "RightLeg", "right_ankle": "RightFoot",
    "right_toe": "RightToe", "right_shoulder": "RightArm", "right_elbow": "RightForeArm",
    "right_wrist": "RightHand", "right_hand_middle": "RightHand",
}


def build_canonical(bvh_path: Path, motion_fps: float = 30.0) -> dict:
    frames, _human_height = load_bvh_file(str(bvh_path), format="lafan1")
    T = len(frames)
    roles = list(ROLE_TO_LAFAN1_BONE.keys())
    role_to_idx = {r: i for i, r in enumerate(roles)}
    positions = np.zeros((T, len(roles), 3), dtype=np.float64)
    hand_quat_wxyz = {"left_hand": np.zeros((T, 4)), "right_hand": np.zeros((T, 4))}
    for t, f in enumerate(frames):
        for r, bone in ROLE_TO_LAFAN1_BONE.items():
            positions[t, role_to_idx[r]] = f[bone][0]
        hand_quat_wxyz["left_hand"][t] = f["LeftHand"][1]
        hand_quat_wxyz["right_hand"][t] = f["RightHand"][1]

    # S5-B1: raw BVH hand rotation (world-frame FK quat, wxyz -- see load_bvh_file) as
    # the source of hand orientation (see the module docstring for why). Convert to
    # matrices BEFORE the yaw correction below, then rotate them the same way positions
    # are rotated (world-frame Z rotation -> left-multiply the frame matrix).
    hand_mats_raw = {r: _Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()  # wxyz -> xyzw
                     for r, q in hand_quat_wxyz.items()}

    yaw_deg = detect_facing_yaw_deg(positions, role_to_idx)
    if yaw_deg != 0.0:
        positions = apply_yaw_to_positions(positions, role_to_idx, yaw_deg)
        yawR = yaw_matrix(yaw_deg)
        for r in hand_mats_raw:
            hand_mats_raw[r] = np.einsum("ij,tjk->tik", yawR, hand_mats_raw[r])

    mats = np.zeros((T, len(ORIENTATION_ROLES), 3, 3), dtype=np.float64)
    for t in range(T):
        p = {r: positions[t, role_to_idx[r]] for r in roles}
        pelvis_y = p["left_hip"] - p["right_hip"]
        pelvis_z = p["torso"] - p["pelvis"]
        mats[t, ORIENTATION_ROLES.index("pelvis")] = frame_from_yz(pelvis_y, pelvis_z)

        torso_y = p["left_shoulder"] - p["right_shoulder"]
        torso_z = p["head"] - p["torso"]  # LAFAN1 has no separate "neck-to-torso" hint bone gap
        mats[t, ORIENTATION_ROLES.index("torso")] = frame_from_yz(torso_y, torso_z)

        head_z = p["head"] - p["neck"]
        mats[t, ORIENTATION_ROLES.index("head")] = frame_from_yz(torso_y, head_z)

        mats[t, ORIENTATION_ROLES.index("left_foot")] = frame_from_xy(
            p["left_toe"] - p["left_ankle"], pelvis_y)
        mats[t, ORIENTATION_ROLES.index("right_foot")] = frame_from_xy(
            p["right_toe"] - p["right_ankle"], pelvis_y)

        # Hands: raw BVH bone rotation (S5-B1 fix, see module docstring + planLogGMR.md
        # ## S5-B0.1/B1) -- NOT frame_from_xy anymore. The earlier forearm-direction /
        # pelvis_y construction (S2-T3/T4 fix) solved a degenerate-primary-axis bug but
        # still couldn't represent forearm twist (pelvis_y as secondary axis pins roll
        # to a geometric artifact, not real wrist rotation). hand_mats_raw was computed
        # + yaw-corrected before this loop; just place it here.
        mats[t, ORIENTATION_ROLES.index("left_hand")] = hand_mats_raw["left_hand"][t]
        mats[t, ORIENTATION_ROLES.index("right_hand")] = hand_mats_raw["right_hand"][t]

    meta = {
        "format": "canonical_human_lafan1_v1",
        "bvh_path": str(bvh_path),
        "source": "lafan1_bvh_via_gmr_loader",
        "fps": motion_fps,
        "orientation_frame_version": "lafan1_adapter_v1",
        "facing_yaw_correction_deg": yaw_deg,
        "known_simplifications": [
            "left/right_hand_middle == left/right_wrist (same LAFAN1 bone, no hand-centroid marker)",
            "left/right_toe_end, left/right_hand_thumb omitted (not load-bearing in Stage 3)",
            "hand orientation = raw BVH bone rotation (S5-B1), NOT a landmark-derived "
            "semantic frame like every other role -- deliberate, see module docstring",
        ],
    }

    return {
        "roles": np.asarray(roles, dtype=object),
        "positions": positions,
        "frames": np.arange(T, dtype=np.int64),
        "fps": np.float64(motion_fps),
        "orientation_role_names": np.asarray(ORIENTATION_ROLES, dtype=object),
        "orientation_mats": mats,
        "orientation_valid": np.ones((len(ORIENTATION_ROLES),), dtype=bool),
        "facing_yaw_correction_deg": np.float64(yaw_deg),
        "metadata_json": json.dumps(meta),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bvh", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--motion-fps", type=float, default=30.0)
    args = ap.parse_args()

    out = build_canonical(args.bvh, args.motion_fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Wrote {args.out} (T={out['positions'].shape[0]}, "
          f"{len(out['roles'])} roles, yaw_correction={out['facing_yaw_correction_deg']:.0f} deg)")


if __name__ == "__main__":
    main()

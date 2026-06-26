#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


ROLE_TO_BONE = {
    # trunk
    "pelvis": "Hips",
    "torso": "Spine3",
    "neck": "Neck1",
    "head": "Head",

    # left leg
    "left_hip": "LeftUpLeg",
    "left_knee": "LeftLeg",
    "left_ankle": "LeftFoot",
    "left_toe": "LeftToeBase",
    "left_toe_end": "LeftToeBaseEnd",

    # right leg
    "right_hip": "RightUpLeg",
    "right_knee": "RightLeg",
    "right_ankle": "RightFoot",
    "right_toe": "RightToeBase",
    "right_toe_end": "RightToeBaseEnd",

    # left arm
    "left_shoulder": "LeftArm",
    "left_elbow": "LeftForeArm",
    "left_wrist": "LeftHand",
    "left_hand_middle": "LeftHandMiddle1",
    "left_hand_thumb": "LeftHandThumb1",

    # right arm
    "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm",
    "right_wrist": "RightHand",
    "right_hand_middle": "RightHandMiddle1",
    "right_hand_thumb": "RightHandThumb1",
}


SEGMENTS = [
    ("pelvis_to_torso", "pelvis", "torso"),
    ("torso_to_neck", "torso", "neck"),
    ("neck_to_head", "neck", "head"),

    ("left_thigh", "left_hip", "left_knee"),
    ("left_shin", "left_knee", "left_ankle"),
    ("left_foot", "left_ankle", "left_toe"),
    ("left_toe", "left_toe", "left_toe_end"),

    ("right_thigh", "right_hip", "right_knee"),
    ("right_shin", "right_knee", "right_ankle"),
    ("right_foot", "right_ankle", "right_toe"),
    ("right_toe", "right_toe", "right_toe_end"),

    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("left_hand_middle", "left_wrist", "left_hand_middle"),
    ("left_hand_thumb", "left_wrist", "left_hand_thumb"),

    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
    ("right_hand_middle", "right_wrist", "right_hand_middle"),
    ("right_hand_thumb", "right_wrist", "right_hand_thumb"),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def strip_namespace(name: str) -> str:
    return name.split(":")[-1]


def find_armature():
    arms = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("No armature found in imported FBX.")
    if len(arms) > 1:
        print("WARNING: multiple armatures found, using first:", [a.name for a in arms])
    return arms[0]


def blender_to_canonical_xyz(v):
    """
    Convert Blender/FBX coordinates to internal canonical convention.

    Observed FBX convention:
      +X = subject right
      +Y = subject forward
      +Z = up

    Canonical convention:
      +X = forward
      +Y = left
      +Z = up
    """
    return np.array([v.y, -v.x, v.z], dtype=np.float64)


def world_bone_head(armature, pose_bone):
    # pose_bone.head is in armature/object space in evaluated pose.
    v = armature.matrix_world @ pose_bone.head
    return blender_to_canonical_xyz(v)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    clear_scene()
    bpy.ops.import_scene.fbx(filepath=str(args.fbx))

    scene = bpy.context.scene
    arm = find_armature()

    # Map stripped FBX names to pose bones.
    pose_by_short = {}
    for pb in arm.pose.bones:
        short = strip_namespace(pb.name)
        pose_by_short[short] = pb

    missing = []
    role_bones_full = {}
    for role, short_bone in ROLE_TO_BONE.items():
        if short_bone not in pose_by_short:
            missing.append((role, short_bone))
        else:
            role_bones_full[role] = pose_by_short[short_bone].name

    if missing:
        print("Missing role bone mappings:")
        for role, bone in missing:
            print(f"  {role}: {bone}")
        raise RuntimeError("Required FBX bones are missing.")

    frame_start = args.start_frame if args.start_frame is not None else int(scene.frame_start)
    frame_end = args.end_frame if args.end_frame is not None else int(scene.frame_end)
    frames = list(range(frame_start, frame_end + 1, args.stride))

    roles = list(ROLE_TO_BONE.keys())
    role_to_idx = {r: i for i, r in enumerate(roles)}

    positions = np.zeros((len(frames), len(roles), 3), dtype=np.float64)

    print("FBX:", args.fbx)
    print("Armature:", arm.name)
    print("Scene frames:", scene.frame_start, "to", scene.frame_end)
    print("Extracting frames:", frame_start, "to", frame_end, "stride", args.stride)
    print("Roles:", len(roles))

    depsgraph = bpy.context.evaluated_depsgraph_get()

    for ti, frame in enumerate(frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph.update()

        for role, short_bone in ROLE_TO_BONE.items():
            pb = arm.pose.bones[role_bones_full[role]]
            positions[ti, role_to_idx[role]] = world_bone_head(arm, pb)

        if ti % 100 == 0 or ti == len(frames) - 1:
            print(f"  frame {frame} -> {ti + 1}/{len(frames)}")

    segment_names = []
    segment_role_pairs = []
    segment_indices = []

    for seg_name, parent_role, child_role in SEGMENTS:
        segment_names.append(seg_name)
        segment_role_pairs.append((parent_role, child_role))
        segment_indices.append((role_to_idx[parent_role], role_to_idx[child_role]))

    segment_indices = np.asarray(segment_indices, dtype=np.int64)
    vecs = positions[:, segment_indices[:, 1], :] - positions[:, segment_indices[:, 0], :]
    dynamic_lengths = np.linalg.norm(vecs, axis=-1)
    median_lengths = np.median(dynamic_lengths, axis=0)

    fps = float(scene.render.fps) / float(scene.render.fps_base)

    metadata = {
        "format": "canonical_human_fbx_positions_v2",
        "fbx_path": str(args.fbx),
        "armature": arm.name,
        "source": "blender_fbx",
        "fps": fps,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "stride": args.stride,
        "frames": frames,
        "role_to_bone": ROLE_TO_BONE,
        "role_to_full_bone": role_bones_full,
        "notes": [
            "Fresh canonical extraction from FBX.",
            "Positions are bone heads converted from Blender/FBX world coordinates to canonical +X forward, +Y left, +Z up.",
            "Endpoint helper roles are stored for orientation/segment-frame construction, but do not need to be used as position IK targets.",
            "No Alex-specific sites or fake semantic markers are used.",
        ],
    }

    np.savez(
        args.out,
        roles=np.asarray(roles, dtype=object),
        positions=positions,
        frames=np.asarray(frames, dtype=np.int64),
        fps=np.asarray(fps, dtype=np.float64),
        segment_names=np.asarray(segment_names, dtype=object),
        segment_role_pairs=np.asarray(segment_role_pairs, dtype=object),
        segment_indices=segment_indices,
        segment_lengths_dynamic=dynamic_lengths,
        segment_lengths_median=median_lengths,
        metadata_json=np.asarray(json.dumps(metadata, indent=2), dtype=object),
    )

    print()
    print("Wrote:", args.out)
    print("positions:", positions.shape)
    print("segments:", len(segment_names))
    print("fps:", fps)
    print()
    print("Median segment lengths:")
    for name, L in zip(segment_names, median_lengths):
        print(f"  {name:20s} {L:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a source-only kinematic canonical v2 skeleton from FBX-like data.

This script intentionally does *not* do contact masking, sole grounding, support
constraints, or robot IK.  Its job is just to make a richer human/source
canonical skeleton that can be inspected visually before any Alex retargeting.

Supported inputs:
  - .fbx: imported through the vendored PoseLib importer.
  - .pkl/.pickle: list of per-frame body dictionaries from fbx_offline_to_robot.py.
  - .npz: existing canonical-like arrays with roles/positions and optionally
    marker_positions/marker_names.

The output NPZ is deliberately plain:
  roles, positions [T, R, 3], orientations [T, R, 3, 3], orientation_valid [R],
  edges [E, 2], fps, metadata_json, and role_status_json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))


V2_ROLES: List[str] = [
    "pelvis",
    "torso",
    "neck",
    "head",
    "head_top",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_palm",
    "left_hand_tip",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_palm",
    "right_hand_tip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_heel",
    "left_toe",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_heel",
    "right_toe",
    "right_foot",
]

EDGE_NAMES: List[Tuple[str, str]] = [
    ("pelvis", "torso"),
    ("torso", "neck"),
    ("neck", "head"),
    ("head", "head_top"),
    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_palm"),
    ("left_palm", "left_hand_tip"),
    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_palm"),
    ("right_palm", "right_hand_tip"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("left_foot", "left_heel"),
    ("left_foot", "left_toe"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("right_foot", "right_heel"),
    ("right_foot", "right_toe"),
]


# Common FBX/Mixamo/OptiTrack/Xsens/canonical-ish aliases.  Matching is
# normalized, so "LeftToeBase", "Left_Toe_Base", and "mixamorig:LeftToeBase"
# compare cleanly.
ROLE_ALIASES: Dict[str, Sequence[str]] = {
    "pelvis": ("pelvis", "hips", "hip", "root"),
    "torso": ("spine3", "spine2", "spine1", "spine", "chest", "torso", "upperchest"),
    "neck": ("neck", "neck1"),
    "head": ("head",),
    "head_top": ("headtop", "headend", "headtail", "head_tail", "head_end", "head_end_site"),
    "left_shoulder": ("leftshoulder", "leftarm", "lshoulder", "larm", "left_shoulder"),
    "left_elbow": ("leftforearm", "leftelbow", "lforearm", "lelbow", "left_elbow"),
    "left_wrist": ("lefthand", "leftwrist", "lhand", "lwrist", "left_wrist"),
    "left_palm": ("leftpalm", "left_palm"),
    "left_hand_tip": ("lefthandtip", "lefthandtail", "left_hand_tail", "leftmiddleend", "lefthandend", "left_hand_tip"),
    "right_shoulder": ("rightshoulder", "rightarm", "rshoulder", "rarm", "right_shoulder"),
    "right_elbow": ("rightforearm", "rightelbow", "rforearm", "relbow", "right_elbow"),
    "right_wrist": ("righthand", "rightwrist", "rhand", "rwrist", "right_wrist"),
    "right_palm": ("rightpalm", "right_palm"),
    "right_hand_tip": ("righthandtip", "righthandtail", "right_hand_tail", "rightmiddleend", "righthandend", "right_hand_tip"),
    "left_hip": ("leftupleg", "lefthip", "lhip", "left_hip"),
    "left_knee": ("leftleg", "leftknee", "lknee", "left_knee"),
    "left_ankle": ("leftfoot", "leftankle", "lankle", "left_ankle"),
    "left_heel": ("leftheel", "lheel", "lhel", "left_heel"),
    "left_toe": ("lefttoebase", "lefttoe", "ltoe", "lmt1", "lmt5", "left_toe"),
    "left_foot": ("leftfoot", "leftsole", "left_foot", "left_sole"),
    "right_hip": ("rightupleg", "righthip", "rhip", "right_hip"),
    "right_knee": ("rightleg", "rightknee", "rknee", "right_knee"),
    "right_ankle": ("rightfoot", "rightankle", "rankle", "right_ankle"),
    "right_heel": ("rightheel", "rheel", "rhel", "right_heel"),
    "right_toe": ("righttoebase", "righttoe", "rtoe", "rmt1", "rmt5", "right_toe"),
    "right_foot": ("rightfoot", "rightsole", "right_foot", "right_sole"),
}


MARKER_ALIASES: Dict[str, Sequence[str]] = {
    "left_inner_wrist": ("LIWR", "L_IWR", "left_inner_wrist"),
    "left_outer_wrist": ("LOWR", "L_OWR", "left_outer_wrist"),
    "left_inner_hand": ("LIHAND", "L_IHAND", "left_inner_hand"),
    "left_outer_hand": ("LOHAND", "L_OHAND", "left_outer_hand"),
    "right_inner_wrist": ("RIWR", "R_IWR", "right_inner_wrist"),
    "right_outer_wrist": ("ROWR", "R_OWR", "right_outer_wrist"),
    "right_inner_hand": ("RIHAND", "R_IHAND", "right_inner_hand"),
    "right_outer_hand": ("ROHAND", "R_OHAND", "right_outer_hand"),
    "left_heel": ("LHEL", "L_HEEL", "left_heel"),
    "left_toe": ("LTOE", "L_TOE", "left_toe"),
    "left_mt1": ("LMT1", "L_MT1", "left_mt1"),
    "left_mt5": ("LMT5", "L_MT5", "left_mt5"),
    "right_heel": ("RHEL", "R_HEEL", "right_heel"),
    "right_toe": ("RTOE", "R_TOE", "right_toe"),
    "right_mt1": ("RMT1", "R_MT1", "right_mt1"),
    "right_mt5": ("RMT5", "R_MT5", "right_mt5"),
}


def normalize_name(name: str) -> str:
    name = name.split(":")[-1]
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, np.maximum(n, eps))


def replace_small_vectors(v: np.ndarray, fallback: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    """Replace near-zero per-frame vectors with a fixed fallback direction."""
    out = np.asarray(v, dtype=float).copy()
    norms = np.linalg.norm(out, axis=-1)
    small = ~np.isfinite(norms) | (norms < eps)
    if np.any(small):
        out[small] = np.asarray(fallback, dtype=float)
    return out


def finite_mask(x: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(x), axis=-1)


def make_frame_from_xy(x_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    x = normalize_vec(x_axis)
    z = normalize_vec(np.cross(x, y_hint))
    y = normalize_vec(np.cross(z, x))
    return np.stack([x, y, z], axis=-1)


def make_frame_from_yz(y_axis: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    y = normalize_vec(y_axis)
    x = normalize_vec(np.cross(y, z_hint))
    z = normalize_vec(np.cross(x, y))
    return np.stack([x, y, z], axis=-1)


def make_frame_from_xz(x_axis: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    x = normalize_vec(x_axis)
    y = normalize_vec(np.cross(z_hint, x))
    z = normalize_vec(np.cross(x, y))
    return np.stack([x, y, z], axis=-1)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return quat_wxyz_to_matrix(q[..., [3, 0, 1, 2]])


def index_by_normalized_name(names: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, name in enumerate(names):
        out.setdefault(normalize_name(str(name)), i)
    return out


def find_name(names: Sequence[str], aliases: Iterable[str]) -> Optional[int]:
    lookup = index_by_normalized_name(names)
    for alias in aliases:
        idx = lookup.get(normalize_name(alias))
        if idx is not None:
            return idx
    return None


def load_npz_points(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    meta: Dict[str, Any] = {"input_kind": "npz", "npz_keys": list(data.files)}

    points: Dict[str, np.ndarray] = {}
    frames: Dict[str, np.ndarray] = {}

    if "roles" in data.files and "positions" in data.files:
        roles = [str(x) for x in data["roles"].tolist()]
        positions = np.asarray(data["positions"], dtype=float)
        for i, role in enumerate(roles):
            points[role] = positions[:, i, :]
        meta["source_roles"] = roles

    if "role_frames" in data.files and "roles" in data.files:
        roles = [str(x) for x in data["roles"].tolist()]
        role_frames = np.asarray(data["role_frames"], dtype=float)
        for i, role in enumerate(roles):
            frames[role] = role_frames[:, i, :, :]

    if "orientations" in data.files and "roles" in data.files:
        roles = [str(x) for x in data["roles"].tolist()]
        orientations = np.asarray(data["orientations"], dtype=float)
        if orientations.ndim == 4 and orientations.shape[-2:] == (3, 3):
            for i, role in enumerate(roles):
                frames.setdefault(role, orientations[:, i, :, :])

    if "role_quats_wxyz" in data.files and "roles" in data.files:
        roles = [str(x) for x in data["roles"].tolist()]
        quats = np.asarray(data["role_quats_wxyz"], dtype=float)
        for i, role in enumerate(roles):
            frames.setdefault(role, quat_wxyz_to_matrix(quats[:, i, :]))

    if "marker_positions" in data.files and "marker_names" in data.files:
        marker_names = [str(x) for x in data["marker_names"].tolist()]
        marker_positions = np.asarray(data["marker_positions"], dtype=float)
        for i, name in enumerate(marker_names):
            points[f"marker:{name}"] = marker_positions[:, i, :]
        meta["marker_names"] = marker_names

    fps = None
    for key in ("fps", "source_fps", "mocap_fps"):
        if key in data.files:
            fps = float(np.asarray(data[key]).reshape(-1)[0])
            break
    meta["fps"] = fps
    return points, frames, meta


def load_pickle_frames(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    with path.open("rb") as f:
        frames_in = pickle.load(f)
    if not isinstance(frames_in, Sequence) or len(frames_in) == 0:
        raise ValueError(f"Expected a non-empty list/sequence of frame dictionaries in {path}")

    names = sorted({str(k) for frame in frames_in for k in frame.keys()})
    points: Dict[str, List[np.ndarray]] = {name: [] for name in names}
    rotations: Dict[str, List[np.ndarray]] = {name: [] for name in names}

    for frame in frames_in:
        for name in names:
            if name not in frame:
                points[name].append(np.full(3, np.nan))
                rotations[name].append(np.full(4, np.nan))
                continue
            pos, quat = frame[name]
            points[name].append(np.asarray(pos, dtype=float))
            rotations[name].append(np.asarray(quat, dtype=float))

    point_arrays = {name: np.asarray(vals, dtype=float) for name, vals in points.items()}
    frame_arrays = {
        name: quat_wxyz_to_matrix(np.asarray(vals, dtype=float))
        for name, vals in rotations.items()
        if np.all(np.isfinite(vals))
    }
    return point_arrays, frame_arrays, {"input_kind": "pickle_frames", "source_names": names, "fps": None}


def blender_module_available() -> bool:
    try:
        import bpy  # noqa: F401

        return True
    except Exception:
        return False


def action_frame_range_blender(scene: Any, bpy_module: Any) -> Tuple[int, int]:
    frames: List[float] = []
    for action in bpy_module.data.actions:
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                frames.append(float(kp.co.x))
    if not frames:
        return int(scene.frame_start), int(scene.frame_end)
    return int(np.floor(min(frames))), int(np.ceil(max(frames)))


def load_fbx_blender(path: Path, fps: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """Load raw FBX through Blender's importer.

    This is the preferred raw-FBX path for the Alex repo.  Normal Python does
    not reliably understand FBX without Autodesk/PoseLib plumbing; Blender does.
    Run this script as:

      blender --background --python scripts/build_fbx_kinematic_canonical_v2.py -- motion.fbx --output out.npz
    """
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.fbx(filepath=str(path))

    scene = bpy.context.scene
    armatures = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    if not armatures:
        raise RuntimeError("No armature found after Blender FBX import.")

    arm = armatures[0]

    def clean_name(name: str) -> str:
        return str(name).split(":")[-1]

    frame_start, frame_end = action_frame_range_blender(scene, bpy)
    frames = list(range(frame_start, frame_end + 1))
    if not frames:
        raise RuntimeError(f"No frames found in FBX: start={frame_start}, end={frame_end}")

    bone_names = [clean_name(pb.name) for pb in arm.pose.bones]
    raw_roles = bone_names + [f"{name}_tail" for name in bone_names]
    raw = np.full((len(frames), len(raw_roles), 3), np.nan, dtype=float)

    for ti, frame in enumerate(frames):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        world = arm.matrix_world
        for bi, pb in enumerate(arm.pose.bones):
            head = world @ pb.head
            tail = world @ pb.tail
            raw[ti, bi, :] = [float(head.x), float(head.y), float(head.z)]
            raw[ti, bi + len(bone_names), :] = [float(tail.x), float(tail.y), float(tail.z)]

    # Match the earlier Alex FBX export convention: first-frame pelvis origin,
    # X forward, Y left, Z Blender/world up.  This is not contact logic; it is
    # just a stable source-skeleton coordinate convention.
    role_lookup = index_by_normalized_name(raw_roles)
    hips_i = role_lookup.get(normalize_name("Hips"))
    left_hip_i = role_lookup.get(normalize_name("LeftUpLeg"))
    right_hip_i = role_lookup.get(normalize_name("RightUpLeg"))
    left_shoulder_i = role_lookup.get(normalize_name("LeftArm"))
    right_shoulder_i = role_lookup.get(normalize_name("RightArm"))

    transform_kind = "raw"
    origin = np.zeros(3, dtype=float)
    forward = np.array([1.0, 0.0, 0.0], dtype=float)
    left = np.array([0.0, 1.0, 0.0], dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    if None in (hips_i, left_hip_i, right_hip_i, left_shoulder_i, right_shoulder_i):
        canonical = raw
        canonical_note = "Could not estimate FBX heading; kept Blender world coordinates."
    else:
        origin = raw[0, int(hips_i)].copy()
        centered = raw - origin[None, None, :]
        left_vec = (
            raw[0, int(left_hip_i)]
            - raw[0, int(right_hip_i)]
            + raw[0, int(left_shoulder_i)]
            - raw[0, int(right_shoulder_i)]
        )
        left_vec[2] = 0.0
        if np.linalg.norm(left_vec) < 1e-8:
            canonical = centered
            transform_kind = "centered"
            canonical_note = "Could not estimate FBX left direction; only centered at first-frame pelvis."
        else:
            left = left_vec / np.linalg.norm(left_vec)
            forward = np.cross(left, up)
            forward = forward / np.linalg.norm(forward)
            left = np.cross(up, forward)
            left = left / np.linalg.norm(left)
            canonical = np.empty_like(centered)
            canonical[..., 0] = centered @ forward
            canonical[..., 1] = centered @ left
            canonical[..., 2] = centered @ up
            transform_kind = "heading"
            canonical_note = "Centered at first-frame Hips; X forward, Y left, Z up from first-frame hips/shoulders."

    points = {name: canonical[:, i, :] for i, name in enumerate(raw_roles)}

    markers = [obj for obj in scene.objects if obj.type == "EMPTY"]
    if markers:
        # Marker coordinates must use the exact same source canonical frame as
        # bones; otherwise palms/heels derived from markers are in a different
        # coordinate system from the skeleton.
        for obj in markers:
            vals = np.full((len(frames), 3), np.nan, dtype=float)
            for ti, frame in enumerate(frames):
                scene.frame_set(frame)
                bpy.context.view_layer.update()
                p = obj.matrix_world.translation
                raw_p = np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
                if transform_kind == "heading":
                    centered_p = raw_p - origin
                    vals[ti, 0] = centered_p @ forward
                    vals[ti, 1] = centered_p @ left
                    vals[ti, 2] = centered_p @ up
                elif transform_kind == "centered":
                    vals[ti] = raw_p - origin
                else:
                    vals[ti] = raw_p
            points[f"marker:{clean_name(obj.name)}"] = vals

    scene_fps = float(scene.render.fps) if scene.render.fps else float(fps)
    return points, {}, {
        "input_kind": "fbx_blender",
        "source_names": raw_roles,
        "armature": arm.name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "fps": scene_fps,
        "coordinate_note": canonical_note,
        "loader_note": "Loaded directly by Blender bpy importer.",
    }


def load_fbx_poselib(path: Path, root_joint: str, fps: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    try:
        from poselib.skeleton.skeleton3d import SkeletonMotion
    except Exception as exc:  # pragma: no cover - depends on local deps
        raise RuntimeError(
            "Could not import PoseLib. Try running from the repo root, or use the vendored "
            "third_party/poselib path."
        ) from exc

    motion = SkeletonMotion.from_fbx(fbx_file_path=str(path), root_joint=root_joint, fps=fps)
    names = [str(x) for x in motion.skeleton_tree.node_names]
    positions = motion.global_translation.detach().cpu().numpy()
    quats_xyzw = motion.global_rotation.detach().cpu().numpy()

    # PoseLib FBX import typically returns cm and y-up.  Match the existing
    # PoseLib exporter convention used in this repo: rotate to z-up and scale
    # cm -> m.  Keep this explicit in metadata so it can be audited.
    rot_yup_to_zup_wxyz = np.array([0.70711, 0.0, 0.0, 0.70711])
    rot_yup_to_zup = quat_wxyz_to_matrix(rot_yup_to_zup_wxyz)

    positions = np.einsum("ij,tkj->tki", rot_yup_to_zup, positions) / 100.0
    rotations = np.einsum("ij,tkjl->tkil", rot_yup_to_zup, quat_xyzw_to_matrix(quats_xyzw))

    point_arrays = {name: positions[:, i, :] for i, name in enumerate(names)}
    frame_arrays = {name: rotations[:, i, :, :] for i, name in enumerate(names)}
    return point_arrays, frame_arrays, {
        "input_kind": "fbx_poselib",
        "source_names": names,
        "root_joint": root_joint,
        "fps": float(fps),
        "coordinate_note": "PoseLib FBX converted from y-up centimeters to z-up meters.",
    }


def get_marker(points: Mapping[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    names = list(points.keys())
    aliases = [f"marker:{x}" for x in MARKER_ALIASES[key]] + list(MARKER_ALIASES[key])
    idx = find_name(names, aliases)
    if idx is None:
        return None
    return points[names[idx]]


def direct_role_point(points: Mapping[str, np.ndarray], role: str) -> Optional[Tuple[str, np.ndarray]]:
    names = list(points.keys())
    idx = find_name(names, [role, *ROLE_ALIASES.get(role, ())])
    if idx is None:
        return None
    name = names[idx]
    return name, np.asarray(points[name], dtype=float)


def direct_role_frame(frames: Mapping[str, np.ndarray], role: str) -> Optional[Tuple[str, np.ndarray]]:
    names = list(frames.keys())
    idx = find_name(names, [role, *ROLE_ALIASES.get(role, ())])
    if idx is None:
        return None
    name = names[idx]
    return name, np.asarray(frames[name], dtype=float)


def derive_points(points: Mapping[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    out: Dict[str, np.ndarray] = {}
    status: Dict[str, Dict[str, Any]] = {}

    for role in V2_ROLES:
        direct = direct_role_point(points, role)
        if direct is not None:
            source_name, value = direct
            out[role] = value
            status[role] = {"kind": "direct", "source": source_name}

    def put(role: str, value: np.ndarray, kind: str, source: str) -> None:
        if role not in out and value is not None:
            out[role] = value
            status[role] = {"kind": kind, "source": source}

    for side in ("left", "right"):
        prefix = "left" if side == "left" else "right"
        iw = get_marker(points, f"{prefix}_inner_wrist")
        ow = get_marker(points, f"{prefix}_outer_wrist")
        ih = get_marker(points, f"{prefix}_inner_hand")
        oh = get_marker(points, f"{prefix}_outer_hand")
        heel = get_marker(points, f"{prefix}_heel")
        toe = get_marker(points, f"{prefix}_toe")
        mt1 = get_marker(points, f"{prefix}_mt1")
        mt5 = get_marker(points, f"{prefix}_mt5")

        if iw is not None and ow is not None:
            put(f"{prefix}_wrist", 0.5 * (iw + ow), "derived", "mean(inner_wrist,outer_wrist)")
        if ih is not None and oh is not None:
            put(f"{prefix}_palm", 0.5 * (ih + oh), "derived", "mean(inner_hand,outer_hand)")
            put(f"{prefix}_hand_tip", 0.5 * (ih + oh), "derived", "mean(inner_hand,outer_hand)")
        if heel is not None:
            put(f"{prefix}_heel", heel, "derived", "heel_marker")
        toe_candidates = [x for x in (toe, mt1, mt5) if x is not None]
        if toe_candidates:
            put(f"{prefix}_toe", np.mean(np.stack(toe_candidates, axis=0), axis=0), "derived", "mean(toe/metatarsal markers)")
        sole_candidates = [x for x in (heel, toe, mt1, mt5) if x is not None]
        if sole_candidates:
            put(f"{prefix}_foot", np.mean(np.stack(sole_candidates, axis=0), axis=0), "derived", "mean(heel/toe/metatarsal markers)")
        if f"{prefix}_ankle" not in out and f"{prefix}_foot" in out:
            put(f"{prefix}_ankle", out[f"{prefix}_foot"], "approximate", "fallback_to_foot_center")

    if "neck" not in out and "torso" in out and "head" in out:
        put("neck", 0.65 * out["head"] + 0.35 * out["torso"], "approximate", "interpolate(torso,head)")
    if "head_top" not in out and "head" in out:
        if "neck" in out:
            direction = normalize_vec(out["head"] - out["neck"])
        else:
            direction = np.tile(np.array([0.0, 0.0, 1.0]), (out["head"].shape[0], 1))
        put("head_top", out["head"] + 0.15 * direction, "approximate", "head + 0.15m head_up")
    if "torso" not in out and "pelvis" in out and "head" in out:
        put("torso", 0.55 * out["head"] + 0.45 * out["pelvis"], "approximate", "interpolate(pelvis,head)")

    for side in ("left", "right"):
        if f"{side}_palm" not in out and f"{side}_wrist" in out:
            put(f"{side}_palm", out[f"{side}_wrist"], "approximate", "fallback_to_wrist")
        if f"{side}_hand_tip" not in out and f"{side}_palm" in out:
            if f"{side}_wrist" in out:
                direction = normalize_vec(
                    replace_small_vectors(out[f"{side}_palm"] - out[f"{side}_wrist"], [1.0, 0.0, 0.0])
                )
            else:
                direction = np.tile(np.array([1.0, 0.0, 0.0]), (out[f"{side}_palm"].shape[0], 1))
            put(f"{side}_hand_tip", out[f"{side}_palm"] + 0.08 * direction, "approximate", "palm + 0.08m wrist_to_palm")

    return out, status


def body_left_hint(out: Mapping[str, np.ndarray], nframes: int) -> np.ndarray:
    if "left_hip" in out and "right_hip" in out:
        return normalize_vec(out["left_hip"] - out["right_hip"])
    if "left_shoulder" in out and "right_shoulder" in out:
        return normalize_vec(out["left_shoulder"] - out["right_shoulder"])
    return np.tile(np.array([0.0, 1.0, 0.0]), (nframes, 1))


def derive_orientations(
    points_out: Mapping[str, np.ndarray],
    source_frames: Mapping[str, np.ndarray],
    role_status: Mapping[str, Dict[str, Any]],
    nframes: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    frames: Dict[str, np.ndarray] = {}
    status: Dict[str, Dict[str, Any]] = {}
    identity = np.tile(np.eye(3)[None, :, :], (nframes, 1, 1))
    body_left = body_left_hint(points_out, nframes)
    world_up = np.tile(np.array([0.0, 0.0, 1.0]), (nframes, 1))

    for role in V2_ROLES:
        direct = direct_role_frame(source_frames, role)
        if direct is not None:
            source_name, value = direct
            frames[role] = value
            status[role] = {"kind": "direct", "source": source_name}

    def put(role: str, value: np.ndarray, kind: str, source: str) -> None:
        if role in points_out and role not in frames:
            frames[role] = value
            status[role] = {"kind": kind, "source": source}

    if {"left_hip", "right_hip", "torso"}.issubset(points_out):
        y = points_out["left_hip"] - points_out["right_hip"]
        z = points_out["torso"] - points_out["pelvis"] if "pelvis" in points_out else world_up
        put("pelvis", make_frame_from_yz(y, z), "derived", "hips_width + pelvis_to_torso")

    if {"torso", "neck"}.issubset(points_out):
        put("torso", make_frame_from_yz(body_left, points_out["neck"] - points_out["torso"]), "derived", "body_left + torso_to_neck")
    if {"neck", "head"}.issubset(points_out):
        put("head", make_frame_from_yz(body_left, points_out["head"] - points_out["neck"]), "derived", "body_left + neck_to_head")

    for side in ("left", "right"):
        sign = 1.0 if side == "left" else -1.0
        wrist = points_out.get(f"{side}_wrist")
        palm = points_out.get(f"{side}_palm")
        tip = points_out.get(f"{side}_hand_tip")
        if palm is not None:
            if wrist is not None:
                x = replace_small_vectors(palm - wrist, [1.0, 0.0, 0.0])
            elif tip is not None:
                x = replace_small_vectors(tip - palm, [1.0, 0.0, 0.0])
            else:
                x = np.tile(np.array([1.0, 0.0, 0.0]), (nframes, 1))
            y_hint = sign * body_left
            frame = make_frame_from_xy(x, y_hint)
            put(f"{side}_palm", frame, "derived", "wrist_to_palm + body_left")
            put(f"{side}_wrist", frame, "approximate", "same_as_palm_frame")
            put(f"{side}_hand_tip", frame, "approximate", "same_as_palm_frame")

        heel = points_out.get(f"{side}_heel")
        toe = points_out.get(f"{side}_toe")
        foot = points_out.get(f"{side}_foot")
        if foot is not None:
            if heel is not None and toe is not None:
                x = replace_small_vectors(toe - heel, [1.0, 0.0, 0.0])
            elif toe is not None:
                x = replace_small_vectors(toe - foot, [1.0, 0.0, 0.0])
            else:
                x = np.tile(np.array([1.0, 0.0, 0.0]), (nframes, 1))
            frame = make_frame_from_xy(x, sign * body_left)
            put(f"{side}_foot", frame, "derived", "heel_to_toe + body_left")
            put(f"{side}_ankle", frame, "approximate", "same_as_foot_frame")
            put(f"{side}_heel", frame, "approximate", "same_as_foot_frame")
            put(f"{side}_toe", frame, "approximate", "same_as_foot_frame")

    for role in V2_ROLES:
        if role in points_out and role not in frames:
            frames[role] = identity.copy()
            status[role] = {"kind": "identity_fallback", "source": "missing_frame"}

    return frames, status


def build_canonical(points: Mapping[str, np.ndarray], source_frames: Mapping[str, np.ndarray], meta: Dict[str, Any]) -> Dict[str, Any]:
    point_roles, point_status = derive_points(points)
    if not point_roles:
        raise ValueError("No recognizable source points found. Inspect source joint/marker names first.")

    nframes = next(iter(point_roles.values())).shape[0]
    positions = np.full((nframes, len(V2_ROLES), 3), np.nan, dtype=float)
    position_valid = np.zeros(len(V2_ROLES), dtype=bool)

    for i, role in enumerate(V2_ROLES):
        if role in point_roles:
            positions[:, i, :] = point_roles[role]
            position_valid[i] = np.any(finite_mask(point_roles[role]))
        else:
            point_status[role] = {"kind": "missing", "source": None}

    frame_roles, frame_status = derive_orientations(point_roles, source_frames, point_status, nframes)
    orientations = np.full((nframes, len(V2_ROLES), 3, 3), np.nan, dtype=float)
    orientation_valid = np.zeros(len(V2_ROLES), dtype=bool)
    for i, role in enumerate(V2_ROLES):
        if role in frame_roles:
            orientations[:, i, :, :] = frame_roles[role]
            orientation_valid[i] = np.all(np.isfinite(frame_roles[role]))
        else:
            frame_status[role] = {"kind": "missing", "source": None}

    edges = np.asarray(
        [(V2_ROLES.index(a), V2_ROLES.index(b)) for a, b in EDGE_NAMES if a in V2_ROLES and b in V2_ROLES],
        dtype=int,
    )

    role_status = {
        role: {
            "position": point_status.get(role, {"kind": "missing", "source": None}),
            "orientation": frame_status.get(role, {"kind": "missing", "source": None}),
        }
        for role in V2_ROLES
    }
    metadata = {
        **meta,
        "format": "fbx_kinematic_canonical_v2",
        "notes": [
            "Source-only kinematic canonical representation.",
            "No contact mask, grounding, support constraint, physics, or robot IK applied.",
            "Role orientations are best-effort frames; inspect visually before using as robot targets.",
        ],
    }
    return {
        "roles": np.asarray(V2_ROLES, dtype=object),
        "positions": positions,
        "position_valid": position_valid,
        "orientations": orientations,
        "orientation_valid": orientation_valid,
        "edges": edges,
        "edge_names": np.asarray(EDGE_NAMES, dtype=object),
        "fps": np.asarray(float(meta.get("fps") or 120.0), dtype=float),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2), dtype=object),
        "role_status_json": np.asarray(json.dumps(role_status, indent=2), dtype=object),
    }


def print_summary(bundle: Mapping[str, Any]) -> None:
    roles = [str(x) for x in bundle["roles"].tolist()]
    pos_valid = np.asarray(bundle["position_valid"], dtype=bool)
    ori_valid = np.asarray(bundle["orientation_valid"], dtype=bool)
    status = json.loads(str(bundle["role_status_json"].item()))
    print("\nCanonical v2 summary")
    print(f"  frames: {bundle['positions'].shape[0]}")
    print(f"  roles:  {len(roles)}")
    print(f"  fps:    {float(bundle['fps']):.3f}")
    print(f"  valid positions:    {int(pos_valid.sum())}/{len(roles)}")
    print(f"  valid orientations: {int(ori_valid.sum())}/{len(roles)}")
    print("\nRole status:")
    for role in roles:
        p = status[role]["position"]
        o = status[role]["orientation"]
        print(f"  {role:16s} pos={p['kind']:18s} src={str(p['source']):28s} | ori={o['kind']:18s} src={str(o['source'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Input .fbx, .pkl/.pickle, or .npz file.")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output canonical v2 .npz path.")
    parser.add_argument("--fps", type=int, default=120, help="FPS to use for FBX import if source does not provide one.")
    parser.add_argument("--root-joint", default="Hips", help="Root joint name for PoseLib FBX import.")
    # Blender passes its own arguments before "--".  Support both normal Python:
    #   python script.py input.fbx --output out.npz
    # and Blender:
    #   blender --background --python script.py -- input.fbx --output out.npz
    parse_argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    args = parser.parse_args(parse_argv)

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    suffix = input_path.suffix.lower()
    if suffix == ".fbx":
        if blender_module_available():
            points, frames, meta = load_fbx_blender(input_path, fps=args.fps)
        else:
            try:
                points, frames, meta = load_fbx_poselib(input_path, root_joint=args.root_joint, fps=args.fps)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Raw FBX loading from normal Python failed. For this Alex repo, use Blender for raw FBX:\n\n"
                    "  blender --background --python scripts/build_fbx_kinematic_canonical_v2.py -- "
                    f"{input_path} --output {output_path} --fps {args.fps}\n\n"
                    "Why: Blender provides the FBX importer. Normal Python needs a separate PoseLib/FBX-SDK setup."
                ) from exc
    elif suffix in {".pkl", ".pickle"}:
        points, frames, meta = load_pickle_frames(input_path)
        meta["fps"] = meta.get("fps") or float(args.fps)
    elif suffix == ".npz":
        points, frames, meta = load_npz_points(input_path)
        meta["fps"] = meta.get("fps") or float(args.fps)
    else:
        raise ValueError(f"Unsupported input type {suffix!r}; expected .fbx, .pkl/.pickle, or .npz")

    meta["source_path"] = str(input_path)
    bundle = build_canonical(points, frames, meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **bundle)
    print_summary(bundle)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()

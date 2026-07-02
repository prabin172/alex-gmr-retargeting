from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from general_motion_retargeting.source_adapters.canonical_human import (
    CANONICAL_BODY_NAMES,
    CanonicalHumanFrame,
    validate_canonical_human_frame,
)


MVNX_TO_CANONICAL_SEGMENT = {
    "pelvis": "Pelvis",
    "torso": "T8",
    "head": "Head",

    "left_hip": "LeftUpperLeg",
    "left_knee": "LeftLowerLeg",
    "left_foot": "LeftFoot",

    "right_hip": "RightUpperLeg",
    "right_knee": "RightLowerLeg",
    "right_foot": "RightFoot",

    "left_shoulder": "LeftUpperArm",
    "left_elbow": "LeftForeArm",
    "left_hand": "LeftHand",

    "right_shoulder": "RightUpperArm",
    "right_elbow": "RightForeArm",
    "right_hand": "RightHand",
}


# Xsens MVNX global frame observed in VTech NMP preview:
#   left/right separation is mostly MVNX X, with left at negative X.
# Canonical frame used by this repo:
#   +X forward, +Y left, +Z up.
#
# This maps:
#   canonical X = MVNX Y
#   canonical Y = -MVNX X
#   canonical Z = MVNX Z
MVNX_ZUP_TO_CANONICAL_R = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


def strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def floats_from_text(text: Optional[str]) -> List[float]:
    if text is None:
        return []
    out = []
    for x in text.replace(",", " ").split():
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    trace = float(np.trace(r))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    return normalize_quat_wxyz(np.array([w, x, y, z], dtype=float))


def mvnx_position_to_canonical(pos: np.ndarray) -> np.ndarray:
    return MVNX_ZUP_TO_CANONICAL_R @ np.asarray(pos, dtype=float)


def mvnx_quat_to_canonical(quat_wxyz: np.ndarray) -> np.ndarray:
    old_r = quat_wxyz_to_matrix(quat_wxyz)
    new_r = MVNX_ZUP_TO_CANONICAL_R @ old_r
    return matrix_to_quat_wxyz(new_r)


def read_mvnx_header(mvnx_path: Path) -> dict:
    """
    Read MVNX subject metadata and segment labels without loading the full file.

    MVNX files can be huge. We use start events so we can read subject attrs
    before the enclosing subject element closes, then stop once frame data begins.
    """
    mvnx_path = Path(mvnx_path)

    subject_attrs = {}
    segments = []
    joints = []

    for event, elem in ET.iterparse(mvnx_path, events=("start",)):
        tag = strip_ns(elem.tag)

        if tag == "subject":
            subject_attrs = dict(elem.attrib)

        elif tag == "segment":
            segments.append({
                "id": elem.attrib.get("id"),
                "label": elem.attrib.get("label"),
                "name": elem.attrib.get("name"),
            })

        elif tag == "joint":
            joints.append(dict(elem.attrib))

        elif tag == "frame":
            break

    segment_labels = [
        (s.get("label") or s.get("name") or s.get("id"))
        for s in segments
    ]

    frame_rate = None
    if subject_attrs.get("frameRate"):
        try:
            frame_rate = int(float(subject_attrs["frameRate"]))
        except ValueError:
            frame_rate = None

    return {
        "path": str(mvnx_path),
        "subject_attrs": subject_attrs,
        "segment_labels": segment_labels,
        "segments": segments,
        "joints": joints,
        "num_segments": len(segment_labels),
        "frame_rate": frame_rate,
    }


def iter_mvnx_raw_frames(
    mvnx_path: Path,
    frame_type: str = "normal",
    start_frame: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
) -> Iterator[dict]:
    """
    Stream MVNX frames.

    start_frame and stride are counted over frames matching frame_type, not over
    all XML frame elements.
    """
    mvnx_path = Path(mvnx_path)

    matched_idx = -1
    yielded = 0

    for event, elem in ET.iterparse(mvnx_path, events=("end",)):
        tag = strip_ns(elem.tag)

        if tag != "frame":
            # Do not clear child elements here. In iterparse, <position> and
            # <orientation> end before their parent <frame>. Clearing them here
            # would erase the text before the frame is processed.
            continue

        current_type = elem.attrib.get("type", "normal")
        if current_type != frame_type:
            elem.clear()
            continue

        matched_idx += 1

        if matched_idx < start_frame:
            elem.clear()
            continue

        if (matched_idx - start_frame) % stride != 0:
            elem.clear()
            continue

        child_values = {}
        for child in list(elem):
            child_values[strip_ns(child.tag)] = floats_from_text(child.text)

        yield {
            "matched_frame_index": matched_idx,
            "attrs": dict(elem.attrib),
            "values": child_values,
        }

        yielded += 1
        elem.clear()

        if max_frames is not None and yielded >= max_frames:
            break


def raw_frame_to_segment_arrays(
    raw_frame: dict,
    segment_labels: List[str],
    canonicalize_axes: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    values = raw_frame["values"]
    n_segments = len(segment_labels)

    if "position" not in values:
        raise ValueError("MVNX frame is missing <position> data.")
    if "orientation" not in values:
        raise ValueError("MVNX frame is missing <orientation> data.")

    pos_values = values["position"]
    quat_values = values["orientation"]

    expected_pos = 3 * n_segments
    expected_quat = 4 * n_segments

    if len(pos_values) != expected_pos:
        raise ValueError(f"Expected {expected_pos} position floats, got {len(pos_values)}.")
    if len(quat_values) != expected_quat:
        raise ValueError(f"Expected {expected_quat} orientation floats, got {len(quat_values)}.")

    pos_arr = np.asarray(pos_values, dtype=float).reshape(n_segments, 3)
    quat_arr = np.asarray(quat_values, dtype=float).reshape(n_segments, 4)

    positions = {}
    orientations = {}

    for i, label in enumerate(segment_labels):
        pos = pos_arr[i]
        quat = quat_arr[i]

        if canonicalize_axes:
            pos = mvnx_position_to_canonical(pos)
            quat = mvnx_quat_to_canonical(quat)
        else:
            quat = normalize_quat_wxyz(quat)

        positions[label] = pos
        orientations[label] = quat

    return positions, orientations


def mvnx_raw_frame_to_canonical(
    raw_frame: dict,
    segment_labels: List[str],
    mapping: Dict[str, str] = MVNX_TO_CANONICAL_SEGMENT,
    canonicalize_axes: bool = True,
) -> CanonicalHumanFrame:
    positions, orientations = raw_frame_to_segment_arrays(
        raw_frame=raw_frame,
        segment_labels=segment_labels,
        canonicalize_axes=canonicalize_axes,
    )

    frame: CanonicalHumanFrame = {}

    for canonical_name in CANONICAL_BODY_NAMES:
        mvnx_segment = mapping[canonical_name]

        if mvnx_segment not in positions:
            available = ", ".join(segment_labels)
            raise KeyError(
                f"MVNX segment {mvnx_segment!r} for canonical {canonical_name!r} not found. "
                f"Available segments: {available}"
            )

        pos = positions[mvnx_segment]
        quat = orientations[mvnx_segment]

        frame[canonical_name] = {
            "pos": [float(x) for x in pos],
            "quat_wxyz": [float(x) for x in quat],
        }

    validate_canonical_human_frame(frame)
    return frame


def _frame_pos(frame: CanonicalHumanFrame, role: str) -> np.ndarray:
    return np.asarray(frame[role]["pos"], dtype=float)


def estimate_heading_rotation_from_frame(frame: CanonicalHumanFrame) -> np.ndarray:
    """
    Estimate a horizontal heading-canonicalization rotation.

    The input frame is already in a Z-up canonical-like coordinate system.
    This function estimates the subject's initial left direction from hips and
    shoulders, then defines forward = left x up.

    Returned R maps old coordinates into heading-canonical coordinates:
      new_x = dot(old, initial_forward)
      new_y = dot(old, initial_left)
      new_z = dot(old, up)
    """
    up = np.array([0.0, 0.0, 1.0], dtype=float)

    left_vecs = []

    for left_role, right_role in [
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
    ]:
        left_vec = _frame_pos(frame, left_role) - _frame_pos(frame, right_role)
        left_vec[2] = 0.0
        if np.linalg.norm(left_vec) > 1e-8:
            left_vecs.append(left_vec)

    if not left_vecs:
        raise ValueError("Could not estimate heading: hip/shoulder left-right vectors are degenerate.")

    left = np.mean(np.stack(left_vecs, axis=0), axis=0)
    left[2] = 0.0
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-8:
        raise ValueError("Could not estimate heading: averaged left vector is degenerate.")
    left = left / left_norm

    forward = np.cross(left, up)
    forward[2] = 0.0
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Could not estimate heading: forward vector is degenerate.")
    forward = forward / forward_norm

    # Re-orthogonalize left so the basis is clean.
    left = np.cross(up, forward)
    left = left / np.linalg.norm(left)

    return np.stack([forward, left, up], axis=0)


def apply_heading_rotation_to_frame(
    frame: CanonicalHumanFrame,
    rotation_old_to_new: np.ndarray,
    origin: np.ndarray,
) -> CanonicalHumanFrame:
    """
    Rotate positions around origin and rotate orientations into the new frame.
    """
    out: CanonicalHumanFrame = {}

    for role in CANONICAL_BODY_NAMES:
        pos_old = np.asarray(frame[role]["pos"], dtype=float)
        quat_old = np.asarray(frame[role]["quat_wxyz"], dtype=float)

        pos_new = rotation_old_to_new @ (pos_old - origin) + origin

        rot_old = quat_wxyz_to_matrix(quat_old)
        rot_new = rotation_old_to_new @ rot_old
        quat_new = matrix_to_quat_wxyz(rot_new)

        out[role] = {
            "pos": [float(x) for x in pos_new],
            "quat_wxyz": [float(x) for x in quat_new],
        }

    validate_canonical_human_frame(out)
    return out


def canonicalize_initial_heading(
    frames: List[CanonicalHumanFrame],
    reference_frame: Optional[CanonicalHumanFrame] = None,
) -> Tuple[List[CanonicalHumanFrame], dict]:
    """
    Rotate a sequence so the reference frame faces canonical +X.

    This removes dependence on the Xsens subject's initial yaw/global heading.
    """
    if not frames:
        return frames, {
            "enabled": True,
            "applied": False,
            "reason": "no frames",
        }

    ref = reference_frame if reference_frame is not None else frames[0]
    origin = _frame_pos(ref, "pelvis")
    rotation = estimate_heading_rotation_from_frame(ref)

    out = [
        apply_heading_rotation_to_frame(frame, rotation, origin)
        for frame in frames
    ]

    return out, {
        "enabled": True,
        "applied": True,
        "origin_pelvis_xyz": [float(x) for x in origin],
        "rotation_old_to_new": rotation.tolist(),
        "definition": "new_x=initial_forward, new_y=initial_left, new_z=up",
    }


def read_mvnx_canonical_frames(
    mvnx_path: Path,
    frame_type: str = "normal",
    start_frame: int = 0,
    stride: int = 1,
    max_frames: Optional[int] = None,
    canonicalize_axes: bool = True,
    canonicalize_heading: bool = False,
    heading_reference_frame: Optional[CanonicalHumanFrame] = None,
) -> Tuple[List[CanonicalHumanFrame], dict]:
    header = read_mvnx_header(mvnx_path)
    segment_labels = header["segment_labels"]

    frames = []
    raw_indices = []

    for raw_frame in iter_mvnx_raw_frames(
        mvnx_path=mvnx_path,
        frame_type=frame_type,
        start_frame=start_frame,
        stride=stride,
        max_frames=max_frames,
    ):
        frames.append(
            mvnx_raw_frame_to_canonical(
                raw_frame=raw_frame,
                segment_labels=segment_labels,
                canonicalize_axes=canonicalize_axes,
            )
        )
        raw_indices.append(raw_frame["matched_frame_index"])

    heading_metadata = {
        "enabled": bool(canonicalize_heading),
        "applied": False,
    }
    if canonicalize_heading:
        frames, heading_metadata = canonicalize_initial_heading(
            frames=frames,
            reference_frame=heading_reference_frame,
        )

    metadata = {
        **header,
        "frame_type": frame_type,
        "start_frame": start_frame,
        "stride": stride,
        "max_frames": max_frames,
        "loaded_frames": len(frames),
        "matched_frame_indices": raw_indices,
        "canonicalize_axes": canonicalize_axes,
        "canonicalize_heading": bool(canonicalize_heading),
        "heading_canonicalization": heading_metadata,
        "coordinate_assumption": {
            "source_mvnx": "observed VTech MVNX: +X roughly subject-right, +Z up",
            "canonical": "+X forward, +Y left, +Z up",
            "position_mapping": "canonical_xyz = [mvnx_y, -mvnx_x, mvnx_z]",
            "units": "meters",
            "quat_order": "wxyz",
        },
    }

    return frames, metadata

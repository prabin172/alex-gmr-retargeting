from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from general_motion_retargeting.source_adapters.canonical_human import (
    CanonicalHumanFrame,
    validate_canonical_human_frame,
)


Segment = Tuple[str, str]


CANONICAL_PARENT_TREE: Dict[str, str | None] = {
    "pelvis": None,
    "torso": "pelvis",
    "head": "torso",

    "left_hip": "pelvis",
    "left_knee": "left_hip",
    "left_foot": "left_knee",

    "right_hip": "pelvis",
    "right_knee": "right_hip",
    "right_foot": "right_knee",

    "left_shoulder": "torso",
    "left_elbow": "left_shoulder",
    "left_hand": "left_elbow",

    "right_shoulder": "torso",
    "right_elbow": "right_shoulder",
    "right_hand": "right_elbow",
}


CANONICAL_TREE_SEGMENTS: List[Segment] = [
    (parent, child)
    for child, parent in CANONICAL_PARENT_TREE.items()
    if parent is not None
]


SEGMENT_GROUPS: Dict[str, List[Segment]] = {
    "torso_head": [
        ("pelvis", "torso"),
        ("torso", "head"),
    ],
    "left_leg": [
        ("pelvis", "left_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_foot"),
    ],
    "right_leg": [
        ("pelvis", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_foot"),
    ],
    "left_arm": [
        ("torso", "left_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_hand"),
    ],
    "right_arm": [
        ("torso", "right_shoulder"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_hand"),
    ],
}


@dataclass(frozen=True)
class SegmentScale:
    parent: str
    child: str
    source_length: float
    target_length: float
    scale: float


def _pos(frame: CanonicalHumanFrame, name: str) -> np.ndarray:
    return np.asarray(frame[name]["pos"], dtype=float)


def segment_length(frame: CanonicalHumanFrame, parent: str, child: str) -> float:
    return float(np.linalg.norm(_pos(frame, child) - _pos(frame, parent)))


def compute_segment_scales(
    source_rest_frame: CanonicalHumanFrame,
    target_rest_frame: CanonicalHumanFrame,
    segments: Iterable[Segment] = CANONICAL_TREE_SEGMENTS,
    eps: float = 1e-8,
) -> Dict[str, SegmentScale]:
    validate_canonical_human_frame(source_rest_frame)
    validate_canonical_human_frame(target_rest_frame)

    scales: Dict[str, SegmentScale] = {}

    for parent, child in segments:
        src_len = segment_length(source_rest_frame, parent, child)
        tgt_len = segment_length(target_rest_frame, parent, child)

        if src_len < eps:
            scale = 1.0
        else:
            scale = tgt_len / src_len

        key = f"{parent}->{child}"
        scales[key] = SegmentScale(
            parent=parent,
            child=child,
            source_length=src_len,
            target_length=tgt_len,
            scale=scale,
        )

    return scales


def compute_group_scale_summary(
    segment_scales: Dict[str, SegmentScale],
    segment_groups: Dict[str, List[Segment]] = SEGMENT_GROUPS,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    for group_name, segments in segment_groups.items():
        source_total = 0.0
        target_total = 0.0

        for parent, child in segments:
            key = f"{parent}->{child}"
            s = segment_scales[key]
            source_total += s.source_length
            target_total += s.target_length

        out[group_name] = {
            "source_total_length": source_total,
            "target_total_length": target_total,
            "scale": target_total / source_total if source_total > 1e-8 else 1.0,
        }

    return out


def scale_frame_by_rest_pose(
    frame: CanonicalHumanFrame,
    source_rest_frame: CanonicalHumanFrame,
    target_rest_frame: CanonicalHumanFrame,
    parent_tree: Dict[str, str | None] = CANONICAL_PARENT_TREE,
) -> CanonicalHumanFrame:
    """
    Locally scale a human frame using source-vs-target rest-pose segment ratios.

    This keeps the input pelvis world position and scales every child vector relative
    to its parent according to the corresponding rest-pose segment length ratio.

    This is the Step 3 operation in simplified landmark form:
      source motion frame
        + source subject rest pose
        + Alex rest pose
        -> locally scaled source motion frame

    Orientations are copied unchanged here. Rotation retargeting is Step 4/5.
    """

    validate_canonical_human_frame(frame)
    validate_canonical_human_frame(source_rest_frame)
    validate_canonical_human_frame(target_rest_frame)

    segment_scales = compute_segment_scales(source_rest_frame, target_rest_frame)

    scaled = deepcopy(frame)

    root = "pelvis"
    scaled[root]["pos"] = [float(x) for x in _pos(frame, root)]

    # The dict insertion order follows root-to-leaf order because parent_tree is defined that way.
    for child, parent in parent_tree.items():
        if parent is None:
            continue

        key = f"{parent}->{child}"
        scale = segment_scales[key].scale

        parent_scaled_pos = np.asarray(scaled[parent]["pos"], dtype=float)
        original_parent_pos = _pos(frame, parent)
        original_child_pos = _pos(frame, child)

        local_vec = original_child_pos - original_parent_pos
        scaled_child_pos = parent_scaled_pos + scale * local_vec

        scaled[child]["pos"] = [float(x) for x in scaled_child_pos]
        scaled[child]["quat_wxyz"] = list(frame[child]["quat_wxyz"])

    validate_canonical_human_frame(scaled)
    return scaled


def segment_scales_to_jsonable(segment_scales: Dict[str, SegmentScale]) -> List[dict]:
    return [
        {
            "segment": key,
            "parent": s.parent,
            "child": s.child,
            "source_length": s.source_length,
            "target_length": s.target_length,
            "scale": s.scale,
        }
        for key, s in segment_scales.items()
    ]

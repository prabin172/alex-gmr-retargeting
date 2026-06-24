from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np

from general_motion_retargeting.source_adapters.canonical_human import (
    CANONICAL_BODY_NAMES,
    CanonicalHumanFrame,
    validate_canonical_human_frame,
)


@dataclass(frozen=True)
class MorphologyScales:
    role_scales: Dict[str, float]
    measurements: Dict[str, float]


def _pos(frame: CanonicalHumanFrame, role: str) -> np.ndarray:
    return np.asarray(frame[role]["pos"], dtype=float)


def _dist(frame: CanonicalHumanFrame, a: str, b: str) -> float:
    return float(np.linalg.norm(_pos(frame, a) - _pos(frame, b)))


def _safe_ratio(target: float, source: float, default: float = 1.0) -> float:
    if not np.isfinite(source) or not np.isfinite(target) or source < 1e-6:
        return float(default)
    return float(target / source)


def measure_canonical_morphology(frame: CanonicalHumanFrame) -> Dict[str, float]:
    """
    Measure coarse morphology from canonical landmarks.

    These are intentionally role-level measures, not exact anatomical segment
    lengths. They are used to scale motion deltas conservatively.
    """
    validate_canonical_human_frame(frame)

    left_arm = _dist(frame, "left_shoulder", "left_elbow") + _dist(frame, "left_elbow", "left_hand")
    right_arm = _dist(frame, "right_shoulder", "right_elbow") + _dist(frame, "right_elbow", "right_hand")

    left_leg = _dist(frame, "left_hip", "left_knee") + _dist(frame, "left_knee", "left_foot")
    right_leg = _dist(frame, "right_hip", "right_knee") + _dist(frame, "right_knee", "right_foot")

    shoulder_width = _dist(frame, "left_shoulder", "right_shoulder")
    hip_width = _dist(frame, "left_hip", "right_hip")
    upper_body = _dist(frame, "pelvis", "head")
    pelvis_to_torso = _dist(frame, "pelvis", "torso")

    return {
        "left_arm_reach": left_arm,
        "right_arm_reach": right_arm,
        "mean_arm_reach": 0.5 * (left_arm + right_arm),
        "left_leg_reach": left_leg,
        "right_leg_reach": right_leg,
        "mean_leg_reach": 0.5 * (left_leg + right_leg),
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "upper_body_height": upper_body,
        "pelvis_to_torso": pelvis_to_torso,
    }


def compute_morphology_scales(
    source_rest: CanonicalHumanFrame,
    target_rest: CanonicalHumanFrame,
    preserve_root_translation: bool = True,
    clamp_min: float = 0.70,
    clamp_max: float = 1.30,
) -> MorphologyScales:
    """
    Compute role-specific delta scales from source and target rest morphology.

    This is a safer alternative to recursive tree scaling:
    - root/pelvis trajectory is preserved by default
    - arm landmarks use arm reach ratio
    - leg landmarks use leg reach ratio
    - head/torso use upper-body ratio
    - shoulder/hip lateral landmarks use width ratios
    """
    source_m = measure_canonical_morphology(source_rest)
    target_m = measure_canonical_morphology(target_rest)

    arm_scale = _safe_ratio(target_m["mean_arm_reach"], source_m["mean_arm_reach"])
    left_arm_scale = _safe_ratio(target_m["left_arm_reach"], source_m["left_arm_reach"], arm_scale)
    right_arm_scale = _safe_ratio(target_m["right_arm_reach"], source_m["right_arm_reach"], arm_scale)

    leg_scale = _safe_ratio(target_m["mean_leg_reach"], source_m["mean_leg_reach"])
    left_leg_scale = _safe_ratio(target_m["left_leg_reach"], source_m["left_leg_reach"], leg_scale)
    right_leg_scale = _safe_ratio(target_m["right_leg_reach"], source_m["right_leg_reach"], leg_scale)

    upper_body_scale = _safe_ratio(target_m["upper_body_height"], source_m["upper_body_height"])
    shoulder_width_scale = _safe_ratio(target_m["shoulder_width"], source_m["shoulder_width"])
    hip_width_scale = _safe_ratio(target_m["hip_width"], source_m["hip_width"])

    def clamp(x: float) -> float:
        return float(np.clip(x, clamp_min, clamp_max))

    arm_scale = clamp(arm_scale)
    left_arm_scale = clamp(left_arm_scale)
    right_arm_scale = clamp(right_arm_scale)
    leg_scale = clamp(leg_scale)
    left_leg_scale = clamp(left_leg_scale)
    right_leg_scale = clamp(right_leg_scale)
    upper_body_scale = clamp(upper_body_scale)
    shoulder_width_scale = clamp(shoulder_width_scale)
    hip_width_scale = clamp(hip_width_scale)

    root_scale = 1.0 if preserve_root_translation else leg_scale

    role_scales = {
        "pelvis": root_scale,
        "torso": upper_body_scale,
        "head": upper_body_scale,

        "left_hip": hip_width_scale,
        "right_hip": hip_width_scale,

        "left_knee": left_leg_scale,
        "left_foot": left_leg_scale,
        "right_knee": right_leg_scale,
        "right_foot": right_leg_scale,

        "left_shoulder": shoulder_width_scale,
        "right_shoulder": shoulder_width_scale,

        "left_elbow": left_arm_scale,
        "left_hand": left_arm_scale,
        "right_elbow": right_arm_scale,
        "right_hand": right_arm_scale,
    }

    measurements = {}
    for k, v in source_m.items():
        measurements[f"source_{k}"] = float(v)
    for k, v in target_m.items():
        measurements[f"target_{k}"] = float(v)

    measurements.update({
        "arm_scale": arm_scale,
        "left_arm_scale": left_arm_scale,
        "right_arm_scale": right_arm_scale,
        "leg_scale": leg_scale,
        "left_leg_scale": left_leg_scale,
        "right_leg_scale": right_leg_scale,
        "upper_body_scale": upper_body_scale,
        "shoulder_width_scale": shoulder_width_scale,
        "hip_width_scale": hip_width_scale,
        "root_scale": root_scale,
        "clamp_min": float(clamp_min),
        "clamp_max": float(clamp_max),
        "preserve_root_translation": bool(preserve_root_translation),
    })

    return MorphologyScales(role_scales=role_scales, measurements=measurements)


def make_morphology_delta_target_frame(
    source_frame: CanonicalHumanFrame,
    source_rest: CanonicalHumanFrame,
    target_rest: CanonicalHumanFrame,
    scales: Mapping[str, float],
    roles: Iterable[str] = CANONICAL_BODY_NAMES,
) -> CanonicalHumanFrame:
    """
    Create target frame by applying morphology-aware scaled *local* source deltas.

    Important:
      - Root/pelvis displacement is preserved once.
      - Non-root body parts are expressed relative to the pelvis.
      - Scaling is applied only to local motion around the pelvis, not to the
        full global walking displacement.

    This avoids tearing the body apart during walking.
    """
    validate_canonical_human_frame(source_frame)
    validate_canonical_human_frame(source_rest)
    validate_canonical_human_frame(target_rest)

    out: CanonicalHumanFrame = {}

    source_pelvis = _pos(source_frame, "pelvis")
    source_rest_pelvis = _pos(source_rest, "pelvis")
    target_rest_pelvis = _pos(target_rest, "pelvis")

    root_scale = float(scales.get("pelvis", 1.0))
    source_root_delta = source_pelvis - source_rest_pelvis
    target_pelvis = target_rest_pelvis + root_scale * source_root_delta

    for role in roles:
        if role == "pelvis":
            target_pos = target_pelvis
        else:
            s = float(scales.get(role, 1.0))

            source_local = _pos(source_frame, role) - source_pelvis
            source_rest_local = _pos(source_rest, role) - source_rest_pelvis
            source_local_delta = source_local - source_rest_local

            target_rest_local = _pos(target_rest, role) - target_rest_pelvis
            target_pos = target_pelvis + target_rest_local + s * source_local_delta

        # Orientation is copied from target rest for now.
        # Later this should be replaced with source-relative orientation deltas.
        out[role] = {
            "pos": [float(x) for x in target_pos],
            "quat_wxyz": [float(x) for x in target_rest[role]["quat_wxyz"]],
        }

    validate_canonical_human_frame(out)
    return out

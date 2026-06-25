from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, TypedDict


class CanonicalBodyPose(TypedDict):
    pos: List[float]
    quat_wxyz: List[float]


CanonicalHumanFrame = Dict[str, CanonicalBodyPose]


IDENTITY_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]


CANONICAL_BODY_NAMES = [
    "pelvis",
    "torso",
    "head",
    "left_hip",
    "left_knee",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_shoulder",
    "right_elbow",
    "right_hand",
]

# These are semantic end-effectors rather than skeleton-joint roles.  Keep
# them separate from CANONICAL_BODY_NAMES so legacy skeleton adapters and
# visualizers remain valid, while canonical NPZs can carry physically matched
# palm/sole targets for contact-aware retargeting.
CANONICAL_AUXILIARY_BODY_NAMES = [
    "left_palm",
    "right_palm",
    "left_sole",
    "right_sole",
]

CANONICAL_NPZ_ROLE_NAMES = CANONICAL_BODY_NAMES + CANONICAL_AUXILIARY_BODY_NAMES


def _body(x: float, y: float, z: float) -> CanonicalBodyPose:
    return {
        "pos": [float(x), float(y), float(z)],
        "quat_wxyz": IDENTITY_QUAT_WXYZ.copy(),
    }


def make_neutral_standing_frame() -> CanonicalHumanFrame:
    """
    Return one synthetic neutral standing human frame.

    Coordinate convention for this internal canonical frame:
      +X: forward
      +Y: left
      +Z: up

    This is not a final anatomical model. It is a deterministic debugging pose
    used to validate source adapters, IK configs, and retargeting plumbing.
    """

    frame: CanonicalHumanFrame = {
        "pelvis": _body(0.00, 0.00, 1.00),
        "torso": _body(0.00, 0.00, 1.35),
        "head": _body(0.00, 0.00, 1.65),

        "left_hip": _body(0.00, 0.10, 0.95),
        "left_knee": _body(0.00, 0.10, 0.55),
        "left_foot": _body(0.08, 0.10, 0.05),

        "right_hip": _body(0.00, -0.10, 0.95),
        "right_knee": _body(0.00, -0.10, 0.55),
        "right_foot": _body(0.08, -0.10, 0.05),

        "left_shoulder": _body(0.00, 0.25, 1.45),
        "left_elbow": _body(0.05, 0.45, 1.20),
        "left_hand": _body(0.10, 0.60, 1.00),

        "right_shoulder": _body(0.00, -0.25, 1.45),
        "right_elbow": _body(0.05, -0.45, 1.20),
        "right_hand": _body(0.10, -0.60, 1.00),
    }

    return frame


def make_t_pose_frame() -> CanonicalHumanFrame:
    """
    Return one synthetic T-pose style frame.

    Useful later for checking left/right, shoulder, elbow, wrist, and hand mapping.
    """

    frame = make_neutral_standing_frame()

    frame["left_elbow"]["pos"] = [0.00, 0.55, 1.42]
    frame["left_hand"]["pos"] = [0.00, 0.85, 1.42]

    frame["right_elbow"]["pos"] = [0.00, -0.55, 1.42]
    frame["right_hand"]["pos"] = [0.00, -0.85, 1.42]

    return frame


def copy_frame(frame: CanonicalHumanFrame) -> CanonicalHumanFrame:
    return deepcopy(frame)


def validate_canonical_human_frame(
    frame: CanonicalHumanFrame,
    *,
    allow_auxiliary_roles: bool = False,
) -> None:
    missing = sorted(set(CANONICAL_BODY_NAMES) - set(frame.keys()))
    allowed = set(CANONICAL_BODY_NAMES)
    if allow_auxiliary_roles:
        allowed.update(CANONICAL_AUXILIARY_BODY_NAMES)
    extra = sorted(set(frame.keys()) - allowed)

    if missing:
        raise ValueError(f"Missing canonical body names: {missing}")

    if extra:
        raise ValueError(f"Unexpected canonical body names: {extra}")

    roles_to_validate = list(CANONICAL_BODY_NAMES)
    if allow_auxiliary_roles:
        roles_to_validate.extend(
            role for role in CANONICAL_AUXILIARY_BODY_NAMES if role in frame
        )

    for body_name in roles_to_validate:
        pose = frame[body_name]

        if "pos" not in pose:
            raise ValueError(f"{body_name}: missing pos")

        if "quat_wxyz" not in pose:
            raise ValueError(f"{body_name}: missing quat_wxyz")

        pos = pose["pos"]
        quat = pose["quat_wxyz"]

        if not isinstance(pos, list) or len(pos) != 3:
            raise ValueError(f"{body_name}: pos must be a length-3 list")

        if not all(isinstance(v, (int, float)) for v in pos):
            raise ValueError(f"{body_name}: pos must contain numbers")

        if not isinstance(quat, list) or len(quat) != 4:
            raise ValueError(f"{body_name}: quat_wxyz must be a length-4 list")

        if not all(isinstance(v, (int, float)) for v in quat):
            raise ValueError(f"{body_name}: quat_wxyz must contain numbers")

        norm = sum(float(v) * float(v) for v in quat) ** 0.5
        if abs(norm - 1.0) > 1e-5:
            raise ValueError(f"{body_name}: quat_wxyz must be normalized, got norm={norm}")


def frame_to_jsonable(frame: CanonicalHumanFrame) -> dict:
    validate_canonical_human_frame(frame)
    return {
        "format": "canonical_human_frame_v1",
        "coordinate_convention": {
            "x": "forward",
            "y": "left",
            "z": "up",
            "quat": "wxyz",
            "units": "meters",
        },
        "body_names": CANONICAL_BODY_NAMES,
        "frame": frame,
    }

#!/usr/bin/env python3
"""Validate Alex palm task sites against hand geometry and wrist perturbations."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

PALM_SPECS = {
    "left": {
        "site": "alex_left_palm_contact_site",
        "body": "LEFT_GRIPPER_Z_LINK",
        "palm_mesh": "palm_hull",
        "joints": ["LEFT_WRIST_Z", "LEFT_WRIST_X", "LEFT_GRIPPER_Z"],
        "normal_offset_m": 0.04,
    },
    "right": {
        "site": "alex_right_palm_contact_site",
        "body": "RIGHT_GRIPPER_Z_LINK",
        "palm_mesh": "palm_hull_mir",
        "joints": ["RIGHT_WRIST_Z", "RIGHT_WRIST_X", "RIGHT_GRIPPER_Z"],
        "normal_offset_m": 0.04,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver-model",
        type=Path,
        default=REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml",
    )
    parser.add_argument(
        "--visual-model",
        type=Path,
        default=REPO_ROOT / "assets/alex/temp_alex_floating_base_visual_mesh.xml",
        help="Optional visual model. It is checked when present.",
    )
    parser.add_argument("--delta-rad", type=float, default=0.20)
    return parser.parse_args()


def object_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"Missing {object_type} named {name!r}")
    return object_id


def forward(model: mujoco.MjModel, qpos: np.ndarray | None = None) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 if qpos is None else qpos
    mujoco.mj_forward(model, data)
    return data


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    return (
        np.asarray(data.site_xpos[site_id], dtype=float).copy(),
        np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy(),
    )


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def geom_id_for_mesh(model: mujoco.MjModel, body_id: int, mesh_name: str) -> int:
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id < 0:
            continue
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) == mesh_name:
            return geom_id
    raise RuntimeError(f"Could not find mesh geom {mesh_name!r} on body id={body_id}")


def finger_direction(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, palm_position: np.ndarray) -> np.ndarray:
    finger_ids = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        mesh_id = model.geom_dataid[geom_id]
        if mesh_id >= 0 and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) == "idx-F1-hull":
            finger_ids.append(geom_id)
    if len(finger_ids) != 4:
        raise RuntimeError(f"Expected four first-finger hulls, found {len(finger_ids)}")
    finger_centre = np.mean(data.geom_xpos[finger_ids], axis=0)
    direction = finger_centre - palm_position
    return direction / np.linalg.norm(direction)


def check_model_sites(model: mujoco.MjModel, label: str) -> None:
    print(f"\n{label} model")
    for side, spec in PALM_SPECS.items():
        site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, spec["site"])
        body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
        if model.site_bodyid[site_id] != body_id:
            actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[site_id])
            raise RuntimeError(
                f"{spec['site']} is attached to {actual}, expected {spec['body']}"
            )
        print(f"  {side:5s} {spec['site']} -> {spec['body']}")


def validate_solver_model(model: mujoco.MjModel, delta_rad: float) -> None:
    data = forward(model)
    print("\nSolver-model semantic frame checks")
    for side, spec in PALM_SPECS.items():
        body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
        palm_geom_id = geom_id_for_mesh(model, body_id, spec["palm_mesh"])
        site_position, site_rotation = site_pose(model, data, spec["site"])
        palm_centre = np.asarray(data.geom_xpos[palm_geom_id], dtype=float)
        finger_forward = finger_direction(model, data, body_id, palm_centre)
        normal = site_rotation[:, 2]
        offset = site_position - palm_centre
        normal_offset = float(np.dot(offset, normal))
        tangential_offset = float(np.linalg.norm(offset - normal_offset * normal))
        body_left_alignment = float(np.dot(site_rotation[:, 1], np.array([0.0, 1.0, 0.0])))
        forward_alignment = float(np.dot(site_rotation[:, 0], finger_forward))
        handedness = float(np.dot(np.cross(site_rotation[:, 0], site_rotation[:, 1]), normal))

        print(f"\n  {side} palm")
        print("    contact position:", np.array2string(site_position, precision=5))
        print("    +X finger-forward alignment:", f"{forward_alignment:+.4f}")
        print("    +Y body-left alignment at qpos0:", f"{body_left_alignment:+.4f}")
        print("    +Z palmar-contact normal:", np.array2string(normal, precision=5))
        print("    +Z offset from palm hull centre:", f"{normal_offset:.5f} m")
        print("    tangential hull-centre offset:", f"{tangential_offset:.6f} m")

        if forward_alignment < 0.95:
            raise RuntimeError(f"{side} site +X is not aligned with fingers")
        if body_left_alignment < 0.90:
            raise RuntimeError(f"{side} site +Y is not body-left in the nominal pose")
        if handedness < 0.999:
            raise RuntimeError(f"{side} palm site axes are not right-handed")
        if abs(normal_offset - spec["normal_offset_m"]) > 2e-4 or tangential_offset > 2e-4:
            raise RuntimeError(f"{side} site is not on the configured +Z palm-contact face")

        baseline_position, baseline_rotation = site_pose(model, data, spec["site"])
        for joint_name in spec["joints"]:
            joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_address = model.jnt_qposadr[joint_id]
            lower, upper = model.jnt_range[joint_id]
            perturbed_qpos = np.array(model.qpos0, copy=True)
            target = perturbed_qpos[qpos_address] + delta_rad
            perturbed_qpos[qpos_address] = np.clip(target, lower + 1e-4, upper - 1e-4)
            actual_delta = perturbed_qpos[qpos_address] - model.qpos0[qpos_address]
            perturbed_data = forward(model, perturbed_qpos)
            position, rotation = site_pose(model, perturbed_data, spec["site"])
            position_change = float(np.linalg.norm(position - baseline_position))
            orientation_change = rotation_angle(baseline_rotation.T @ rotation)
            print(
                f"    {joint_name:16s} dq={actual_delta:+.3f} rad "
                f"| dp={position_change:.5f} m | dR={orientation_change:.5f} rad"
            )
            if position_change < 1e-5 or orientation_change < 1e-4:
                raise RuntimeError(f"{joint_name} does not move/orient {spec['site']} as expected")


def compare_visual_model(solver_model: mujoco.MjModel, visual_model: mujoco.MjModel) -> None:
    if solver_model.nq != visual_model.nq:
        raise RuntimeError("Solver and visual models have different qpos dimensions")
    solver_data = forward(solver_model)
    visual_data = forward(visual_model, solver_data.qpos)
    print("\nSolver/visual task-frame agreement")
    for side, spec in PALM_SPECS.items():
        solver_position, solver_rotation = site_pose(solver_model, solver_data, spec["site"])
        visual_position, visual_rotation = site_pose(visual_model, visual_data, spec["site"])
        position_error = float(np.linalg.norm(solver_position - visual_position))
        orientation_error = rotation_angle(solver_rotation.T @ visual_rotation)
        print(
            f"  {side:5s} dp={position_error:.3e} m "
            f"dR={orientation_error:.3e} rad"
        )
        # The independently compiled visual mesh differs at float-roundoff level.
        if position_error > 1e-7 or orientation_error > 1e-7:
            raise RuntimeError(f"Solver and visual frame disagree for {spec['site']}")


def main() -> None:
    args = parse_args()
    solver_model = mujoco.MjModel.from_xml_path(str(args.solver_model))
    check_model_sites(solver_model, "Solver")
    validate_solver_model(solver_model, args.delta_rad)

    if args.visual_model.exists():
        visual_model = mujoco.MjModel.from_xml_path(str(args.visual_model))
        check_model_sites(visual_model, "Visual")
        compare_visual_model(solver_model, visual_model)
    else:
        print("\nVisual model not found; skipped visual consistency check:", args.visual_model)
    print("\nPASS: Alex palm contact task frames are geometry- and joint-chain-consistent.")


if __name__ == "__main__":
    main()

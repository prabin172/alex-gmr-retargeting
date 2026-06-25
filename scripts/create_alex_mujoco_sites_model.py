#!/usr/bin/env python
from __future__ import annotations

import copy
import json
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]

# Robot-side task frames.  These do not depend on the human motion source.
#
# The palm contact frames were measured from the compiled Ability Hand palm
# collision hulls.  In the nominal Alex pose their axes are:
#   +X: wrist-to-fingers / hand-forward
#   +Y: body-left across the palm
#   +Z: palm contact normal (= +X x +Y)
# Their origins are 40 mm along +Z from the hull centre, on the palmar contact
# face.  They intentionally live below *_GRIPPER_Z_LINK so they follow all
# three wrist/hand joints, including GRIPPER_Z.
SITE_SPECS = [
    # Pelvis/head reference sites.
    {
        "name": "alex_pelvis_site",
        "parent_body": "PELVIS_LINK",
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.035,
        "rgba": [1.0, 0.2, 0.2, 1.0],
    },
    {
        "name": "alex_head_site",
        "parent_body": "HEAD_LINK",
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.035,
        "rgba": [1.0, 0.2, 0.2, 1.0],
    },

    # Legacy foot task sites. These sit above the true bottom contact surface
    # and remain available for older left_foot/right_foot experiments.
    {
        "name": "alex_left_sole_site",
        "parent_body": "LEFT_FOOT",
        "pos": [0.08, 0.0, -0.035],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.03,
        "rgba": [0.1, 0.8, 0.1, 1.0],
    },
    {
        "name": "alex_right_sole_site",
        "parent_body": "RIGHT_FOOT",
        "pos": [0.08, 0.0, -0.035],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.03,
        "rgba": [0.1, 0.8, 0.1, 1.0],
    },

    # Explicit sole support/contact frames from the Alex foot collision/visual
    # geometry. The foot collision box is centred at [0.05, 0, -0.06] with
    # half-height 0.01, and the visual mesh has nearly the same bottom, so the
    # bottom contact plane is z ~= -0.07 in the LEFT_FOOT/RIGHT_FOOT body frame.
    # Axes:
    #   +X: toe/foot-forward
    #   +Y: body-left across sole width
    #   +Z: sole normal/up
    {
        "name": "alex_left_sole_contact_site",
        "parent_body": "LEFT_FOOT",
        "pos": [0.05, 0.0, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": [0.10, 0.045, 0.002],
        "type": "box",
        "rgba": [0.1, 0.9, 0.1, 0.65],
        "semantic_axes": {
            "x": "toe_forward",
            "y": "body_left_sole_width",
            "z": "sole_contact_normal_up",
        },
    },
    {
        "name": "alex_right_sole_contact_site",
        "parent_body": "RIGHT_FOOT",
        "pos": [0.05, 0.0, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": [0.10, 0.045, 0.002],
        "type": "box",
        "rgba": [0.1, 0.9, 0.1, 0.65],
        "semantic_axes": {
            "x": "toe_forward",
            "y": "body_left_sole_width",
            "z": "sole_contact_normal_up",
        },
    },
    # Four bottom-corner support points for z-only planted-sole constraints.
    # They match the foot collision box corners:
    #   center [0.05, 0, -0.06], half-size [0.11, 0.05, 0.01]
    # so bottom plane z = -0.07, toe x = 0.16, heel x = -0.06.
    # Names use body-left/body-right rather than medial/lateral because the
    # same local +Y convention is used for both LEFT_FOOT and RIGHT_FOOT.
    {
        "name": "alex_left_sole_corner_toe_body_left_site",
        "parent_body": "LEFT_FOOT",
        "pos": [0.16, 0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },
    {
        "name": "alex_left_sole_corner_toe_body_right_site",
        "parent_body": "LEFT_FOOT",
        "pos": [0.16, -0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },
    {
        "name": "alex_left_sole_corner_heel_body_left_site",
        "parent_body": "LEFT_FOOT",
        "pos": [-0.06, 0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },
    {
        "name": "alex_left_sole_corner_heel_body_right_site",
        "parent_body": "LEFT_FOOT",
        "pos": [-0.06, -0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },
    {
        "name": "alex_right_sole_corner_toe_body_left_site",
        "parent_body": "RIGHT_FOOT",
        "pos": [0.16, 0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },
    {
        "name": "alex_right_sole_corner_toe_body_right_site",
        "parent_body": "RIGHT_FOOT",
        "pos": [0.16, -0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },
    {
        "name": "alex_right_sole_corner_heel_body_left_site",
        "parent_body": "RIGHT_FOOT",
        "pos": [-0.06, 0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },
    {
        "name": "alex_right_sole_corner_heel_body_right_site",
        "parent_body": "RIGHT_FOOT",
        "pos": [-0.06, -0.05, -0.07],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.012,
        "rgba": [0.2, 1.0, 0.2, 0.9],
    },

    # Palm support/contact frames from the actual Ability Hand palm geometry.
    {
        "name": "alex_left_palm_contact_site",
        "parent_body": "LEFT_GRIPPER_Z_LINK",
        "pos": [0.039306155, -0.011246511, -0.073823097],
        "quat": [0.697532226, 0.046261709, 0.713627319, 0.045218326],
        "size": [0.045, 0.018, 0.002],
        "type": "box",
        "rgba": [0.1, 0.3, 1.0, 0.75],
        "semantic_axes": {
            "x": "finger_forward",
            "y": "body_left_palm_width",
            "z": "palmar_contact_normal",
        },
        "contact_normal_offset_m": 0.04,
    },
    {
        "name": "alex_right_palm_contact_site",
        "parent_body": "RIGHT_GRIPPER_Z_LINK",
        "pos": [0.039305502, 0.011246527, -0.073851064],
        "quat": [0.697282707, -0.046276407, 0.713871266, -0.045201061],
        "size": [0.045, 0.018, 0.002],
        "type": "box",
        "rgba": [0.1, 0.3, 1.0, 0.75],
        "semantic_axes": {
            "x": "finger_forward",
            "y": "body_left_palm_width",
            "z": "palmar_contact_normal",
        },
        "contact_normal_offset_m": 0.04,
    },

    # Optional distal hand-tip sites for reaching/grasping diagnostics.
    {
        "name": "alex_left_hand_tip_site",
        "parent_body": "LEFT_GRIPPER_Z_LINK",
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.02,
        "rgba": [0.8, 0.3, 1.0, 1.0],
    },
    {
        "name": "alex_right_hand_tip_site",
        "parent_body": "RIGHT_GRIPPER_Z_LINK",
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.02,
        "rgba": [0.8, 0.3, 1.0, 1.0],
    },
]


DEPRECATED_SITE_NAMES = {
    "alex_left_palm_site",
    "alex_right_palm_site",
}


def find_body(elem: ET.Element, name: str) -> ET.Element | None:
    if elem.tag == "body" and elem.attrib.get("name") == name:
        return elem
    for child in elem:
        found = find_body(child, name)
        if found is not None:
            return found
    return None


def fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _as_values(value):
    if isinstance(value, (list, tuple)):
        return value
    return [value]


def upsert_sites_in_xml(xml_path: Path, site_specs) -> None:
    """Replace task sites in one MuJoCo XML, keeping geometry untouched."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")

    known_names = {spec["name"] for spec in site_specs} | DEPRECATED_SITE_NAMES
    for body in worldbody.iter("body"):
        for child in list(body):
            if child.tag == "site" and child.attrib.get("name") in known_names:
                body.remove(child)

    for spec in site_specs:
        body = find_body(worldbody, spec["parent_body"])
        if body is None:
            raise RuntimeError(
                f"Could not find parent body {spec['parent_body']!r} in {xml_path}"
            )

        ET.SubElement(
            body,
            "site",
            {
                "name": spec["name"],
                "pos": fmt(spec["pos"]),
                "quat": fmt(spec["quat"]),
                "size": fmt(_as_values(spec["size"])),
                "rgba": fmt(spec["rgba"]),
                "type": spec.get("type", "sphere"),
            },
        )

    try:
        ET.indent(tree, space="  ")
    except AttributeError:  # pragma: no cover - for older Python only.
        pass
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Alex task sites in solver and optional visual MuJoCo models."
    )
    parser.add_argument(
        "--visual-model",
        type=Path,
        default=REPO_ROOT / "assets/alex/temp_alex_floating_base_visual_mesh.xml",
        help="Visual-mesh XML to keep task sites in, if it exists.",
    )
    parser.add_argument(
        "--skip-visual-model",
        action="store_true",
        help="Only write the collision/solver model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_cfg_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex.json"
    robot_cfg = json.loads(robot_cfg_path.read_text())

    input_model_path = REPO_ROOT / robot_cfg["model_path"]
    out_xml_path = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"
    tmp_xml_path = REPO_ROOT / "assets/alex/alex_floating_base_compiled_tmp.xml"
    out_cfg_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex_with_sites.json"
    site_cfg_path = REPO_ROOT / "general_motion_retargeting/robot_configs/alex_retarget_sites.json"

    print("Input model:", input_model_path)
    print("Output model:", out_xml_path)

    model = mujoco.MjModel.from_xml_path(str(input_model_path))
    mujoco.mj_saveLastXML(str(tmp_xml_path), model)

    upsert_sites_in_xml(tmp_xml_path, SITE_SPECS)
    tmp_xml_path.replace(out_xml_path)

    # Verify model loads and all sites exist.
    model2 = mujoco.MjModel.from_xml_path(str(out_xml_path))
    missing = []
    for spec in SITE_SPECS:
        sid = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_SITE, spec["name"])
        if sid < 0:
            missing.append(spec["name"])

    if missing:
        raise RuntimeError(f"Missing sites after reload: {missing}")

    cfg2 = copy.deepcopy(robot_cfg)
    cfg2["model_path"] = str(out_xml_path.relative_to(REPO_ROOT))
    cfg2["retarget_site_names"] = {
        "pelvis": "alex_pelvis_site",
        "head": "alex_head_site",
        "left_foot": "alex_left_sole_site",
        "right_foot": "alex_right_sole_site",
        "left_sole": "alex_left_sole_contact_site",
        "right_sole": "alex_right_sole_contact_site",
        "left_sole_corner_toe_body_left": "alex_left_sole_corner_toe_body_left_site",
        "left_sole_corner_toe_body_right": "alex_left_sole_corner_toe_body_right_site",
        "left_sole_corner_heel_body_left": "alex_left_sole_corner_heel_body_left_site",
        "left_sole_corner_heel_body_right": "alex_left_sole_corner_heel_body_right_site",
        "right_sole_corner_toe_body_left": "alex_right_sole_corner_toe_body_left_site",
        "right_sole_corner_toe_body_right": "alex_right_sole_corner_toe_body_right_site",
        "right_sole_corner_heel_body_left": "alex_right_sole_corner_heel_body_left_site",
        "right_sole_corner_heel_body_right": "alex_right_sole_corner_heel_body_right_site",
        "left_palm": "alex_left_palm_contact_site",
        "right_palm": "alex_right_palm_contact_site",
        # Legacy aliases preserve older tools that still name the endpoint
        # left_hand/right_hand. New canonical IK uses left_palm/right_palm.
        "left_hand": "alex_left_palm_contact_site",
        "right_hand": "alex_right_palm_contact_site",
    }
    cfg2["optional_retarget_site_names"] = {
        "left_hand_tip": "alex_left_hand_tip_site",
        "right_hand_tip": "alex_right_hand_tip_site",
    }

    out_cfg_path.write_text(json.dumps(cfg2, indent=2) + "\n")
    site_cfg_path.write_text(json.dumps({"sites": SITE_SPECS}, indent=2) + "\n")

    if not args.skip_visual_model:
        visual_model_path = args.visual_model
        if not visual_model_path.is_absolute():
            visual_model_path = REPO_ROOT / visual_model_path
        if visual_model_path.exists():
            upsert_sites_in_xml(visual_model_path, SITE_SPECS)
            visual_model = mujoco.MjModel.from_xml_path(str(visual_model_path))
            missing_visual = [
                spec["name"]
                for spec in SITE_SPECS
                if mujoco.mj_name2id(
                    visual_model, mujoco.mjtObj.mjOBJ_SITE, spec["name"]
                )
                < 0
            ]
            if missing_visual:
                raise RuntimeError(
                    f"Missing sites from visual model after sync: {missing_visual}"
                )
            print("Synced visual task sites:", visual_model_path)
        else:
            print("Visual model not found; skipped visual-site sync:", visual_model_path)

    print()
    print("Created:")
    print(" ", out_xml_path)
    print(" ", out_cfg_path)
    print(" ", site_cfg_path)
    print()
    print("Verified sites:")
    for spec in SITE_SPECS:
        sid = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_SITE, spec["name"])
        print(f"  {spec['name']:28s} id={sid:3d} parent={spec['parent_body']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


REPO_ROOT = Path(__file__).resolve().parents[1]

# Rough first-pass semantic sites.
# These are robot-side definitions. They do not depend on MVNX/FBX/BVH subject.
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

    # Sole/contact sites. Initial offsets are rough and should be inspected.
    # These are intended to represent the support/contact frame, not just foot body origin.
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

    # Palm/support sites. First pass uses wrist-link frames rather than gripper tip.
    # Later we can tune local offsets once we inspect the hand geometry.
    {
        "name": "alex_left_palm_site",
        "parent_body": "LEFT_WRIST_X_LINK",
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.03,
        "rgba": [0.1, 0.3, 1.0, 1.0],
    },
    {
        "name": "alex_right_palm_site",
        "parent_body": "RIGHT_WRIST_X_LINK",
        "pos": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "size": 0.03,
        "rgba": [0.1, 0.3, 1.0, 1.0],
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


def main() -> None:
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

    tree = ET.parse(tmp_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("No <worldbody> found in compiled XML.")

    for spec in SITE_SPECS:
        body = find_body(worldbody, spec["parent_body"])
        if body is None:
            raise RuntimeError(f"Could not find parent body {spec['parent_body']!r}")

        # Remove old site with same name if rerunning.
        for child in list(body):
            if child.tag == "site" and child.attrib.get("name") == spec["name"]:
                body.remove(child)

        ET.SubElement(
            body,
            "site",
            {
                "name": spec["name"],
                "pos": fmt(spec["pos"]),
                "quat": fmt(spec["quat"]),
                "size": fmt([spec["size"]]),
                "rgba": fmt(spec["rgba"]),
                "type": "sphere",
            },
        )

    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass

    tree.write(out_xml_path, encoding="utf-8", xml_declaration=True)

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
        "left_hand": "alex_left_palm_site",
        "right_hand": "alex_right_palm_site",
    }
    cfg2["optional_retarget_site_names"] = {
        "left_hand_tip": "alex_left_hand_tip_site",
        "right_hand_tip": "alex_right_hand_tip_site",
    }

    out_cfg_path.write_text(json.dumps(cfg2, indent=2) + "\n")
    site_cfg_path.write_text(json.dumps({"sites": SITE_SPECS}, indent=2) + "\n")

    tmp_xml_path.unlink(missing_ok=True)

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

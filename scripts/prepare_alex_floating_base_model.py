from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]

IN_URDF = REPO_ROOT / "assets/alex/alex_mujoco_ready.urdf"
OUT_URDF = REPO_ROOT / "assets/alex/alex_floating_base.urdf"


def find_root_link(root: ET.Element) -> str:
    links = {x.attrib["name"] for x in root.findall("link")}
    children = set()

    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            children.add(child.attrib["link"])

    roots = sorted(links - children)

    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one root link, found: {roots}")

    return roots[0]


def add_floating_root_joint(root: ET.Element, root_link: str) -> None:
    if root.find("link[@name='WORLD']") is None:
        ET.SubElement(root, "link", {"name": "WORLD"})

    old_joint = root.find("joint[@name='ROOT_FLOATING_BASE']")
    if old_joint is not None:
        root.remove(old_joint)

    joint = ET.Element(
        "joint",
        {
            "name": "ROOT_FLOATING_BASE",
            "type": "floating",
        },
    )

    ET.SubElement(joint, "parent", {"link": "WORLD"})
    ET.SubElement(joint, "child", {"link": root_link})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    # Put the floating joint near the top of the URDF, after links are declared.
    root.append(joint)


def main() -> None:
    if not IN_URDF.exists():
        raise FileNotFoundError(
            f"Missing {IN_URDF}. Run scripts/prepare_alex_mujoco_assets.py first."
        )

    tree = ET.parse(IN_URDF)
    root = tree.getroot()

    root_link = find_root_link(root)
    print("Input URDF:", IN_URDF)
    print("Detected root link:", root_link)

    add_floating_root_joint(root, root_link)

    tree.write(OUT_URDF, encoding="utf-8", xml_declaration=True)

    print("Wrote floating-base URDF:", OUT_URDF)
    print()

    print("Testing MuJoCo load...")
    try:
        model = mujoco.MjModel.from_xml_path(str(OUT_URDF))
    except Exception as e:
        print()
        print("FAILED: MuJoCo could not load floating-base URDF")
        print(type(e).__name__)
        print(e)
        raise SystemExit(1)

    print()
    print("SUCCESS: MuJoCo loaded floating-base Alex")
    print("nbody:", model.nbody)
    print("njnt:", model.njnt)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)

    print()
    print("Bodies:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"{i:3d}: {name}")

    print()
    print("Joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = int(model.jnt_type[i])
        qadr = int(model.jnt_qposadr[i])
        dadr = int(model.jnt_dofadr[i])
        print(f"{i:3d}: {name:30s} type={jtype} qpos_adr={qadr:2d} dof_adr={dadr:2d}")

    print()
    if model.nq == 36 and model.nv == 35:
        print("GOOD: floating-base layout looks correct for 7 base qpos + 29 joints.")
    else:
        print("CHECK: expected nq=36 and nv=35 for 7 base qpos + 29 joints.")


if __name__ == "__main__":
    main()

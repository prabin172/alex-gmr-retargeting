from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]

SRC_URDF = Path.home() / "IsaacData/ihmc-alex-sdk/alex-models/alex_V2_description/urdf/heh.urdf"

OUT_DIR = REPO_ROOT / "assets" / "alex"
OUT_SOURCE_DIR = OUT_DIR / "source"
OUT_MESH_DIR = OUT_DIR / "meshes"

OUT_ORIGINAL_URDF = OUT_SOURCE_DIR / "heh_original.urdf"
OUT_READY_URDF = OUT_DIR / "alex_mujoco_ready.urdf"

PACKAGE_ROOTS = {
    "alex_V1_description": Path.home() / "IsaacData/ihmc-alex-sdk/alex-models/alex_V1_description",
    "alex_V2_description": Path.home() / "IsaacData/ihmc-alex-sdk/alex-models/alex_V2_description",
    "abilityHand": Path.home() / "IsaacData/ihmc-alex-sdk/alex-ros2/ihmc_hands_ros2/meshes/abilityHand",
}


def resolve_mesh_ref(ref: str) -> tuple[Path, Path]:
    """
    Returns:
      src_path: actual source mesh path on disk
      rel_out_path: desired relative mesh path inside assets/alex/
    """
    if ref.startswith("package://"):
        rest = ref[len("package://"):]
        package, _, rel = rest.partition("/")
        if package not in PACKAGE_ROOTS:
            raise FileNotFoundError(f"Unknown package in mesh ref: {ref}")

        src_path = PACKAGE_ROOTS[package] / rel

        # For abilityHand, rel is just the filename.
        # For Alex descriptions, rel starts with meshes/...
        if package == "abilityHand":
            rel_out_path = Path("meshes") / "abilityHand" / rel
        else:
            rel_no_meshes = Path(rel)
            if rel_no_meshes.parts and rel_no_meshes.parts[0] == "meshes":
                rel_no_meshes = Path(*rel_no_meshes.parts[1:])
            rel_out_path = Path("meshes") / package / rel_no_meshes

        return src_path, rel_out_path

    p = Path(ref)

    if p.is_absolute():
        src_path = p
    else:
        # Generated hand paths look like:
        # ../IsaacData/ihmc-alex-sdk/alex-ros2/...
        if ref.startswith("../IsaacData/"):
            src_path = Path.home() / ref.replace("../IsaacData/", "IsaacData/")
        else:
            src_path = (SRC_URDF.parent / p).resolve()

    # Put all non-package refs under meshes/misc while preserving basename.
    # Most of these are ability hand visual meshes.
    if "abilityHand" in str(src_path):
        rel_out_path = Path("meshes") / "abilityHand" / src_path.name
    else:
        rel_out_path = Path("meshes") / "misc" / src_path.name

    return src_path, rel_out_path


def main() -> None:
    if not SRC_URDF.exists():
        raise FileNotFoundError(SRC_URDF)

    OUT_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MESH_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SRC_URDF, OUT_ORIGINAL_URDF)

    tree = ET.parse(SRC_URDF)
    root = tree.getroot()

    copied = {}
    rewritten = 0

    for mesh in root.findall(".//mesh"):
        ref = mesh.attrib.get("filename")
        if not ref:
            continue

        src_path, rel_out_path = resolve_mesh_ref(ref)
        src_path = src_path.expanduser().resolve()

        if not src_path.exists():
            raise FileNotFoundError(f"Missing mesh:\n  ref: {ref}\n  tried: {src_path}")

        dst_path = OUT_DIR / rel_out_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            # If same output name already exists, require same source file path.
            prev = copied.get(rel_out_path)
            if prev is not None and prev != src_path:
                raise RuntimeError(
                    "Mesh output collision:\n"
                    f"  output: {rel_out_path}\n"
                    f"  previous: {prev}\n"
                    f"  new: {src_path}"
                )
        else:
            shutil.copy2(src_path, dst_path)

        copied[rel_out_path] = src_path

        # MuJoCo should resolve this relative to alex_mujoco_ready.urdf location.
        mesh.set("filename", rel_out_path.as_posix())
        rewritten += 1

    tree.write(OUT_READY_URDF, encoding="utf-8", xml_declaration=True)

    print("Source URDF:")
    print(" ", SRC_URDF)
    print()
    print("Copied original URDF:")
    print(" ", OUT_ORIGINAL_URDF)
    print()
    print("Wrote MuJoCo-ready URDF:")
    print(" ", OUT_READY_URDF)
    print()
    print("Unique meshes copied:", len(copied))
    print("Mesh references rewritten:", rewritten)
    print()
    print("Asset folder:")
    print(" ", OUT_DIR)


if __name__ == "__main__":
    main()

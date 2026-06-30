"""Build the Alex V2 collision MJCF from alexFullConvex.urdf.

Surgically swaps collision geometry in the hand-authored
`alex_floating_base_with_sites.xml` (kinematics, inertials, sites preserved)
to mirror the mentor's `alexFullConvex.urdf` collision scheme:

  - Arms (shoulder Y/X/Z, elbow, wrist Z/X), head, and the single closed-fist
    per hand  -> convex hull meshes (`*_convex.stl`)
  - Legs, pelvis, torso  -> primitives, left exactly as-is (URDF keeps them)

Per-link convex STLs are copied into assets/alex/meshes/alex_V2_description/
preserving the package-relative subpath, and referenced directly (MuJoCo reads
STL natively).

Writes a NEW file (alex_floating_base_with_sites_v2.xml); the original is left
untouched.
"""
from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALEX = REPO / "assets" / "alex"
URDF = ALEX / "source" / "alex_V2_description" / "urdf" / "alexFullConvex.urdf"
BASE_MJCF = ALEX / "alex_floating_base_with_sites.xml"
OUT_MJCF = ALEX / "alex_floating_base_with_sites_v2.xml"

PKG_SRC = ALEX / "source" / "alex_V2_description"      # vendored upstream package
MESH_DST_ROOT = ALEX / "meshes" / "alex_V2_description"  # where MuJoCo resolves them


def rpy_to_quat(rpy: str) -> tuple[float, float, float, float]:
    """URDF rpy (fixed-axis X,Y,Z) -> MuJoCo quat (w, x, y, z)."""
    r, p, y = (float(v) for v in rpy.split())
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    # R = Rz(y) Ry(p) Rx(r)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y_ = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y_, z)


def fmt(v: float) -> str:
    return f"{v:.9g}"


def resolve_pkg(filename: str) -> tuple[Path, str, str]:
    """package://alex_V2_description/meshes/REST ->
    (source path, dest rel-to-mjcf path, mesh asset name)."""
    assert filename.startswith("package://alex_V2_description/meshes/"), filename
    rest = filename[len("package://alex_V2_description/meshes/"):]
    src = PKG_SRC / "meshes" / rest
    dst_rel = f"meshes/alex_V2_description/{rest}"
    name = Path(rest).stem  # e.g. LeftFist_convex
    return src, dst_rel, name


def main() -> None:
    urdf = ET.parse(URDF).getroot()

    # link name -> (mesh asset name, dst rel path, pos str, quat str, src path)
    mesh_swaps: dict[str, tuple[str, str, str, str, Path]] = {}
    for link in urdf.findall("link"):
        name = link.get("name")
        col = link.find("collision")
        if col is None:
            continue
        mesh = col.find("geometry/mesh")
        if mesh is None:
            continue  # primitive collision (legs/pelvis/torso) -> leave MJCF as-is
        src, dst_rel, asset = resolve_pkg(mesh.get("filename"))
        origin = col.find("origin")
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        pos = " ".join(fmt(float(v)) for v in xyz.split())
        quat = " ".join(fmt(v) for v in rpy_to_quat(rpy))
        mesh_swaps[name] = (asset, dst_rel, pos, quat, src)

    # --- copy meshes ---
    copied = []
    for asset, dst_rel, _, _, src in mesh_swaps.values():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = ALEX / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst_rel)

    # --- rewrite MJCF ---
    tree = ET.parse(BASE_MJCF)
    root = tree.getroot()

    # asset block: drop abilityHand hulls, add the V2 convex meshes
    asset = root.find("asset")
    for m in list(asset.findall("mesh")):
        asset.remove(m)
    for asset_name, dst_rel, _, _, _ in sorted(
        {(a, d, p, q, s) for a, d, p, q, s in mesh_swaps.values()}
    ):
        el = ET.SubElement(asset, "mesh")
        el.set("name", asset_name)
        el.set("file", dst_rel)

    # body geoms: swap collisions
    swapped = []
    for body in root.iter("body"):
        name = body.get("name")
        if name not in mesh_swaps:
            continue
        asset_name, _, pos, quat, _ = mesh_swaps[name]
        # remove existing collision primitives + old abilityHand mesh geoms,
        # keep everything else (sites, child bodies, inertial, joint)
        for g in list(body.findall("geom")):
            gname = g.get("name") or ""
            if gname.endswith("_collision") or g.get("type") == "mesh":
                body.remove(g)
        geom = ET.Element("geom")
        geom.set("name", f"{name.lower()}_convex_collision")
        geom.set("type", "mesh")
        geom.set("mesh", asset_name)
        if pos != "0 0 0":
            geom.set("pos", pos)
        if quat != "1 0 0 0":
            geom.set("quat", quat)
        # insert geom right after inertial/joint, before child bodies/sites
        body.insert(_geom_insert_idx(body), geom)
        swapped.append(name)

    ET.indent(tree, space="  ")
    tree.write(OUT_MJCF, encoding="utf-8", xml_declaration=True)

    print(f"meshes copied: {len(copied)}")
    print(f"bodies swapped to convex mesh collision: {len(swapped)}")
    for n in swapped:
        print(f"  {n} -> {mesh_swaps[n][0]}")
    print(f"\nwrote {OUT_MJCF.relative_to(REPO)}")


def _geom_insert_idx(body: ET.Element) -> int:
    """Index after the last inertial/joint child, before bodies/sites."""
    idx = 0
    for i, child in enumerate(body):
        if child.tag in ("inertial", "joint"):
            idx = i + 1
    return idx


if __name__ == "__main__":
    main()

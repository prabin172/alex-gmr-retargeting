#!/usr/bin/env python3
"""S2-T3: G1 model with vetted collision + injected floor, in-memory only (never
writes to the GMR clone).

Root cause of week-1/W2-T6's "self-collision noise" (18.2% on a clean walk):
`g1_mocap_29dof.xml` carries a DUPLICATE mesh geom per body -- one visual-only
(contype=0/conaffinity=0) and one full-mesh "collision" copy (contype=1/
conaffinity=1, confirmed by direct inspection: pelvis/hip/knee/torso/elbow all
show this pattern). That full-mesh duplicate is what W2-T6 found noisy, NOT an
absence of collision geometry.

This loader: (1) disables every contype=1 MESH geom (sets contype=conaffinity=0
-- visual-only, matches the OTHER copy each body already has), (2) grafts the
11-body/15-geom cylinder primitives from GMR's own `g1_custom_collision_29dof.urdf`
(W2-T6's vetted model -- local pos/quat/size read directly off that already-
compiled MuJoCo model, so no manual URDF rpy/origin math), (3) injects a floor
mocap-body plane exactly like `_load_model_with_floor` in
`solve_fbx_canonical_alex_contactfirst.py`/`solve_global_trajectory_opt_contactfirst.py`
(same technique, duplicated per this codebase's convention for independent CLI
scripts). Sphere geoms (the sole-corner contact markers on ankle_roll_link,
already used by `stage_b_g1.py`) are left untouched.

Gives Stage 3 on G1 real, non-noisy self-collision rows using the SAME 39-body
mocap model (head/hands/sole-markers all present) `self_collision_rows` and the
role maps need -- the standalone vetted-collision URDF alone is missing
head_link/rubber_hand bodies, unusable as the solving model by itself.
"""
from __future__ import annotations

from pathlib import Path

import mujoco

MOCAP_MODEL_DEFAULT = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
VETTED_COLLISION_URDF = Path(
    "/home/ptimilsina/projects/alex-gmr-retargeting/outputs/gmr_baseline/g1_collision/"
    "g1_collision_vetted.urdf")

FLOOR_BODY_NAME = "g1_floor_mocap"
FLOOR_GEOM_NAME = "g1_floor_geom"


def _extract_cylinder_specs(vetted_path: Path):
    """(body_name, pos, quat, size) for every collision cylinder in the vetted
    model, read directly off its compiled geoms (no manual URDF rpy parsing)."""
    m = mujoco.MjModel.from_xml_path(str(vetted_path))
    specs = []
    for g in range(m.ngeom):
        if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue
        bid = int(m.geom_bodyid[g])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        specs.append((bname, m.geom_pos[g].tolist(), m.geom_quat[g].tolist(),
                      m.geom_size[g].tolist()))
    return specs


def load_g1_model_with_vetted_collision_and_floor(
    mocap_path: Path = MOCAP_MODEL_DEFAULT,
    vetted_collision_path: Path = VETTED_COLLISION_URDF,
):
    """Returns (model, data, floor_gid, floor_mocap_id) -- same return shape as
    `_load_model_with_floor` elsewhere in this codebase."""
    cyl_specs = _extract_cylinder_specs(vetted_collision_path)

    spec = mujoco.MjSpec.from_file(str(mocap_path))

    n_disabled = 0
    for body in spec.bodies:
        for g in body.geoms:
            if (int(g.type) == int(mujoco.mjtGeom.mjGEOM_MESH)
                    and int(g.contype) == 1 and int(g.conaffinity) == 1):
                g.contype = 0
                g.conaffinity = 0
                n_disabled += 1

    n_added = 0
    for bname, pos, quat, size in cyl_specs:
        body = spec.body(bname)
        newg = body.add_geom(name=f"{bname}_collcyl{n_added}",
                             type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                             pos=pos, quat=quat, size=size)
        newg.contype = 1
        newg.conaffinity = 1
        n_added += 1

    floor_body = spec.worldbody.add_body(name=FLOOR_BODY_NAME, mocap=True)
    floor_body.add_geom(name=FLOOR_GEOM_NAME, type=mujoco.mjtGeom.mjGEOM_PLANE,
                        size=[0, 0, 0.01], pos=[0, 0, 0])

    model = spec.compile()
    data = mujoco.MjData(model)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    floor_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FLOOR_BODY_NAME)
    floor_mocap_id = int(model.body_mocapid[floor_bid])

    print(f"[g1_model_setup] disabled {n_disabled} noisy full-mesh collision geoms, "
          f"grafted {n_added} vetted cylinders, injected floor plane.")
    return model, data, floor_gid, floor_mocap_id


if __name__ == "__main__":
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    print(f"nbody={model.nbody} ngeom={model.ngeom} floor_gid={floor_gid}")

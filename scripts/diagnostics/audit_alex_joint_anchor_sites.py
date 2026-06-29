#!/usr/bin/env python3
import mujoco
import numpy as np
from pathlib import Path

MODEL = Path("assets/alex/alex_floating_base_with_sites.xml")
model = mujoco.MjModel.from_xml_path(str(MODEL))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("MODEL:", MODEL)
print("nq:", model.nq, "nv:", model.nv)
print()

print("=== JOINTS: name, type, body, qposadr, anchor world ===")
for j in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
    body_id = int(model.jnt_bodyid[j])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    jtype = int(model.jnt_type[j])
    adr = int(model.jnt_qposadr[j])
    anchor = data.xanchor[j]
    print(f"{j:3d} {name:35s} type={jtype} body={body_name:35s} qposadr={adr:3d} anchor={anchor}")

print()
print("=== BODIES likely relevant ===")
for b in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    u = name.upper()
    if any(k in u for k in ["HIP", "KNEE", "ANKLE", "FOOT", "SHIN", "SHOULDER", "ELBOW", "WRIST", "PALM", "TORSO", "NECK", "HEAD"]):
        print(f"{b:3d} {name:45s} xpos={data.xpos[b]}")

print()
print("=== SITES likely relevant ===")
for s in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or ""
    u = name.upper()
    if any(k in u for k in ["HIP", "KNEE", "ANKLE", "SOLE", "TOE", "HEEL", "FOOT", "PALM", "PELVIS", "HEAD", "SHOULDER", "ELBOW", "WRIST"]):
        print(f"{s:3d} {name:55s} xpos={data.site_xpos[s]}")

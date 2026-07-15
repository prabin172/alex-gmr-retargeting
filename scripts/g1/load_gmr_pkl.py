#!/usr/bin/env python3
"""GMR pkl -> (qpos (T,36) MuJoCo free-root convention, fps).

GMR's own save format (`bvh_to_robot.py`, `gmr_headless_retarget.py`):
    {fps, root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,29), local_body_pos, link_body_list}

Converts root_rot xyzw -> wxyz (this repo's convention everywhere, per CLAUDE.md) at this
one boundary, nowhere else. qpos layout: [x,y,z, qw,qx,qy,qz, 29 joints] -- same shape as
Alex's free-root qpos (7+29), per GMR-baseline.md SS6.

Usage:
    conda run -n gmr python scripts/g1/load_gmr_pkl.py outputs/gmr_baseline/pkl/walk1_subject1.pkl
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


def load_gmr_pkl(path: Path) -> tuple[np.ndarray, float]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)      # (T,3)
    root_rot_xyzw = np.asarray(d["root_rot"], dtype=np.float64)  # (T,4) xyzw
    dof_pos = np.asarray(d["dof_pos"], dtype=np.float64)         # (T,29)
    T = root_pos.shape[0]

    norms = np.linalg.norm(root_rot_xyzw, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-6), \
        f"{path}: root_rot quats not unit-norm (min={norms.min()}, max={norms.max()})"

    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    qpos = np.zeros((T, 7 + dof_pos.shape[1]), dtype=np.float64)
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_rot_wxyz
    qpos[:, 7:] = dof_pos
    return qpos, float(d["fps"])


def _fk_sanity_check(qpos, model_path, frame=0):
    """FK a single frame; feet should be near z~=0 on a standing/walking first frame --
    a wrong quat order shows up immediately as an absurdly pitched/rolled robot."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[frame]
    mujoco.mj_forward(model, data)
    lowest = min(data.geom_xpos[g][2] for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) != 0)
    return lowest


if __name__ == "__main__":
    p = Path(sys.argv[1])
    qpos, fps = load_gmr_pkl(p)
    print(f"{p}: qpos {qpos.shape}, fps={fps}")
    print(f"  root_pos[0]={qpos[0,:3]}  root_quat[0](wxyz)={qpos[0,3:7]}")
    g1_xml = Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
    lowest = _fk_sanity_check(qpos, g1_xml)
    print(f"  FK frame-0 lowest geom-origin z = {lowest:.3f} m (sanity: should be roughly "
          f"pelvis-height-ish, not wildly negative/positive if quat order is right)")

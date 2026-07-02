#!/usr/bin/env python3
"""Post-process a solved IK NPZ: apply root-Z grounding and add contact labels.

Stage 4 of the retargeting pipeline (between IK solve and render).

Usage:
    python scripts/post_process_grounding_contacts.py \\
        --in-npz  outputs/standup_02_ik.npz \\
        --out-npz outputs/standup_02_grounded.npz

The output NPZ keeps all keys from the input and adds:
  qpos_raw           (N, 36) — original ungrounded qpos from the IK solver
  qpos               (N, 36) — grounded qpos (root-Z shifted so lowest geom Z >= 0)
  contact_labels     (N, K)  — boolean: body k is in contact with ground at frame t
  contact_body_names (K,)    — body names for contact_labels columns

Root-Z grounding: after each frame's IK, find the minimum Z of ALL collision geom
surfaces (accounting for geom shape and orientation). If any geom penetrates the
ground (Z < 0), translate qpos[2] (root Z) upward by that amount.

Contact detection: after grounding, flag a body as "in contact" if the lowest
surface Z of any of its collision geoms is within --contact-threshold of Z=0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"

# Bodies tracked for contact labels (ordered — defines column indices).
# Covers: standing (feet), get-up (thighs, shins, pelvis, torso, head, hands).
CONTACT_BODIES = [
    "LEFT_FOOT",
    "RIGHT_FOOT",
    "LEFT_SHIN",
    "RIGHT_SHIN",
    "LEFT_THIGH",
    "RIGHT_THIGH",
    "PELVIS_LINK",
    "TORSO_LINK",
    "HEAD_LINK",
    "LEFT_GRIPPER_Z_LINK",
    "RIGHT_GRIPPER_Z_LINK",
]

CONTACT_THRESHOLD_DEFAULT = 0.02  # metres


def _geom_lowest_z(g: int, model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """World-space lowest Z coordinate of collision geom g (accounts for shape + orientation)."""
    gtype = int(model.geom_type[g])
    pos = data.geom_xpos[g]
    mat = data.geom_xmat[g].reshape(3, 3)  # columns = local axes in world frame
    sz = model.geom_size[g]

    SPHERE   = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    CAPSULE  = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    BOX      = int(mujoco.mjtGeom.mjGEOM_BOX)
    CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)

    if gtype == SPHERE:
        return float(pos[2] - sz[0])

    if gtype == CAPSULE:
        # Axis is local Z; two hemispherical end-caps at ±half_len along axis.
        radius, half_len = float(sz[0]), float(sz[1])
        axis_z = float(mat[2, 2])  # Z component of the capsule's local Z in world
        e1_z = float(pos[2]) + axis_z * half_len
        e2_z = float(pos[2]) - axis_z * half_len
        return min(e1_z, e2_z) - radius

    if gtype == BOX:
        # Minimum corner Z = pos_z - sum of |row2(R)| * half_extents.
        hx, hy, hz = float(sz[0]), float(sz[1]), float(sz[2])
        return float(pos[2]) - abs(mat[2, 0]) * hx - abs(mat[2, 1]) * hy - abs(mat[2, 2]) * hz

    if gtype == CYLINDER:
        # Two circular caps at ±half_len along local Z.  Rim of lower cap extends
        # further down by radius * |sin(tilt)| when the cylinder is tilted.
        radius, half_len = float(sz[0]), float(sz[1])
        axis_z = float(mat[2, 2])
        e1_z = float(pos[2]) + axis_z * half_len
        e2_z = float(pos[2]) - axis_z * half_len
        sin_tilt = float(np.sqrt(max(0.0, 1.0 - axis_z ** 2)))
        return min(e1_z, e2_z) - radius * sin_tilt

    # Plane, mesh, hfield — return geom centre (conservative: won't over-ground).
    return float(pos[2])


def _robot_lowest_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Minimum Z over all robot collision geoms (excludes static world geoms)."""
    min_z = float("inf")
    for g in range(model.ngeom):
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        if int(model.geom_bodyid[g]) == 0:  # worldbody (floor etc.)
            continue
        lz = _geom_lowest_z(g, model, data)
        if lz < min_z:
            min_z = lz
    return min_z


def _body_contact_flags(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids: list[int],
    threshold: float,
) -> np.ndarray:
    """Boolean (K,) array: body k is in contact if its lowest geom Z <= threshold."""
    flags = np.zeros(len(body_ids), dtype=bool)
    for k, bid in enumerate(body_ids):
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != bid:
                continue
            if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
                continue
            if _geom_lowest_z(g, model, data) <= threshold:
                flags[k] = True
                break
    return flags


def process(
    in_npz: Path,
    model_path: Path,
    out_npz: Path,
    contact_threshold: float,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    contact_body_names: list[str] = []
    contact_body_ids: list[int] = []
    for bname in CONTACT_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid == -1:
            print(f"WARNING: body '{bname}' not found in model — skipping.")
        else:
            contact_body_names.append(bname)
            contact_body_ids.append(bid)

    z_in = np.load(in_npz, allow_pickle=True)
    qpos_original = np.asarray(z_in["qpos"], dtype=np.float64)
    N = qpos_original.shape[0]

    qpos_grounded = qpos_original.copy()
    contact_labels = np.zeros((N, len(contact_body_ids)), dtype=bool)
    root_z_shifts = np.zeros(N, dtype=np.float64)

    print(f"Input:  {in_npz}  ({N} frames)")
    print(f"Model:  {model_path}")
    print(f"Contact bodies ({len(contact_body_names)}): {contact_body_names}")
    print(f"Contact threshold: {contact_threshold:.3f} m")
    print()

    for t in range(N):
        data.qpos[:] = qpos_grounded[t]
        mujoco.mj_forward(model, data)

        lowest_z = _robot_lowest_z(model, data)
        if lowest_z < 0.0:
            shift = -lowest_z
            qpos_grounded[t, 2] += shift
            root_z_shifts[t] = shift
            data.qpos[2] = qpos_grounded[t, 2]
            mujoco.mj_forward(model, data)

        contact_labels[t] = _body_contact_flags(
            model, data, contact_body_ids, contact_threshold
        )

        if t % 50 == 0 or t == N - 1:
            shift_str = f"+{root_z_shifts[t]:.4f}m" if root_z_shifts[t] > 0 else "none"
            flags_str = "".join(str(int(b)) for b in contact_labels[t])
            print(f"  frame {t + 1:4d}/{N}  root_z_shift={shift_str:10s}  contacts=[{flags_str}]")

    # Build output NPZ: preserve all original keys, overwrite qpos, add new keys.
    out: dict[str, object] = {k: z_in[k] for k in z_in.files}
    out["qpos_raw"] = qpos_original
    out["qpos"] = qpos_grounded
    out["root_z_shifts"] = root_z_shifts
    out["contact_labels"] = contact_labels
    out["contact_body_names"] = np.asarray(contact_body_names, dtype=object)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_npz), **out)

    print()
    print(f"Output: {out_npz}")
    print(f"  qpos_raw       {qpos_original.shape}  (original IK output)")
    print(f"  qpos           {qpos_grounded.shape}  (grounded)")
    print(f"  root_z_shifts  {root_z_shifts.shape}  min={root_z_shifts.min():.4f} max={root_z_shifts.max():.4f} m")
    print(f"  contact_labels {contact_labels.shape}")
    print()
    print("Contact frequency per body:")
    for k, bname in enumerate(contact_body_names):
        pct = contact_labels[:, k].mean() * 100.0
        bar = "#" * int(pct / 5)
        print(f"  {bname:30s} {pct:5.1f}%  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-npz", required=True, type=Path, help="IK solver output NPZ")
    ap.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        type=Path,
        help=f"Alex MuJoCo model XML (default: {MODEL_DEFAULT})",
    )
    ap.add_argument("--out-npz", required=True, type=Path, help="Grounded + contact-labelled output NPZ")
    ap.add_argument(
        "--contact-threshold",
        type=float,
        default=CONTACT_THRESHOLD_DEFAULT,
        help=f"Z threshold (m) for ground contact detection (default: {CONTACT_THRESHOLD_DEFAULT})",
    )
    args = ap.parse_args()
    process(args.in_npz, args.model, args.out_npz, args.contact_threshold)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Inspect a FBX kinematic canonical v2 NPZ.

This is source-only validation: it checks the canonical human skeleton before
any robot retargeting, contact logic, or physics enters the picture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_FRAME_ROLES = ("pelvis", "head", "left_palm", "right_palm", "left_foot", "right_foot")


def load_json_scalar(data: np.lib.npyio.NpzFile, key: str):
    if key not in data.files:
        return None
    return json.loads(str(data[key].item()))


def segment_lengths(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    a = positions[:, edges[:, 0], :]
    b = positions[:, edges[:, 1], :]
    return np.linalg.norm(b - a, axis=-1)


def frame_orthogonality(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eye = np.eye(3)
    gram = np.einsum("...ji,...jk->...ik", R, R)
    det = np.linalg.det(R)
    err = np.linalg.norm(gram - eye, axis=(-2, -1))
    return err, det


def print_role_status(roles: Sequence[str], status: dict | None) -> None:
    if not status:
        return
    print("\nRole status:")
    for role in roles:
        item = status.get(role, {})
        p = item.get("position", {})
        o = item.get("orientation", {})
        print(
            f"  {role:16s} "
            f"pos={p.get('kind', 'unknown'):18s} src={str(p.get('source')):30s} | "
            f"ori={o.get('kind', 'unknown'):18s} src={str(o.get('source'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_npz", type=Path)
    parser.add_argument(
        "--frame-roles",
        default=",".join(DEFAULT_FRAME_ROLES),
        help="Comma-separated roles whose orientation matrices should be checked in detail.",
    )
    args = parser.parse_args()

    path = args.canonical_npz.expanduser().resolve()
    data = np.load(path, allow_pickle=True)

    roles = [str(x) for x in data["roles"].tolist()]
    positions = np.asarray(data["positions"], dtype=float)
    orientations = np.asarray(data["orientations"], dtype=float) if "orientations" in data.files else None
    edges = np.asarray(data["edges"], dtype=int)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else float("nan")
    position_valid = np.asarray(data["position_valid"], dtype=bool) if "position_valid" in data.files else np.all(np.isfinite(positions), axis=(0, 2))
    orientation_valid = (
        np.asarray(data["orientation_valid"], dtype=bool)
        if "orientation_valid" in data.files
        else np.zeros(len(roles), dtype=bool)
    )
    metadata = load_json_scalar(data, "metadata_json")
    status = load_json_scalar(data, "role_status_json")

    print(f"File: {path}")
    print(f"Format: {(metadata or {}).get('format', '<unknown>')}")
    print(f"FPS: {fps:.3f}")
    print(f"Positions shape: {positions.shape}")
    if orientations is not None:
        print(f"Orientations shape: {orientations.shape}")
    print(f"Edges shape: {edges.shape}")
    print(f"Valid position roles: {int(position_valid.sum())}/{len(roles)}")
    print(f"Valid orientation roles: {int(orientation_valid.sum())}/{len(roles)}")

    total_nan_pos = int(np.isnan(positions).sum())
    total_inf_pos = int(np.isinf(positions).sum())
    print(f"Position NaNs/Infs: {total_nan_pos}/{total_inf_pos}")
    if orientations is not None:
        total_nan_ori = int(np.isnan(orientations).sum())
        total_inf_ori = int(np.isinf(orientations).sum())
        print(f"Orientation NaNs/Infs: {total_nan_ori}/{total_inf_ori}")

    missing_positions = [role for role, ok in zip(roles, position_valid) if not ok]
    missing_orientations = [role for role, ok in zip(roles, orientation_valid) if not ok]
    print(f"Missing/invalid position roles: {missing_positions}")
    print(f"Missing/invalid orientation roles: {missing_orientations}")

    if edges.size:
        lengths = segment_lengths(positions, edges)
        print("\nSegment length summary over valid edges:")
        for edge, vals in zip(edges, lengths.T):
            a, b = roles[int(edge[0])], roles[int(edge[1])]
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                print(f"  {a:16s}->{b:16s} no finite samples")
                continue
            print(
                f"  {a:16s}->{b:16s} "
                f"mean={finite.mean():.4f} m  min={finite.min():.4f}  max={finite.max():.4f}  std={finite.std():.4f}"
            )

    if orientations is not None:
        print("\nOrientation frame checks:")
        selected = [x.strip() for x in args.frame_roles.split(",") if x.strip()]
        for role in selected:
            if role not in roles:
                print(f"  {role:16s} missing role")
                continue
            ri = roles.index(role)
            R = orientations[:, ri, :, :]
            finite = np.all(np.isfinite(R), axis=(1, 2))
            if not np.any(finite):
                print(f"  {role:16s} no finite orientation samples")
                continue
            err, det = frame_orthogonality(R[finite])
            print(
                f"  {role:16s} "
                f"orth_err mean/max={err.mean():.3e}/{err.max():.3e}  "
                f"det mean/min/max={det.mean():+.4f}/{det.min():+.4f}/{det.max():+.4f}"
            )

    if metadata:
        print("\nMetadata:")
        print(json.dumps(metadata, indent=2))
    print_role_status(roles, status)


if __name__ == "__main__":
    main()

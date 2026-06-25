#!/usr/bin/env python3
"""Compare explicit-palm Alex retargeting runs from their NPZ/summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=NPZ",
        help="Repeat for each run to compare.",
    )
    return parser.parse_args()


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=NPZ, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def orientation_error_deg(target_wxyz: np.ndarray, solved_wxyz: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(target_wxyz * solved_wxyz, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def run_metrics(path: Path) -> dict:
    run = np.load(path, allow_pickle=True)
    summary_path = path.with_name(f"{path.stem}_summary.json")
    summary = json.loads(summary_path.read_text())

    ik_roles = [str(role) for role in run["ik_roles"].tolist()]
    palm_indices = [ik_roles.index("left_palm"), ik_roles.index("right_palm")]
    target_positions = np.asarray(run["target_positions"], dtype=float)[:, palm_indices]
    solved_positions = np.asarray(run["solved_ik_positions"], dtype=float)[:, palm_indices]
    position_errors = np.linalg.norm(solved_positions - target_positions, axis=-1)

    target_quats = np.asarray(run["target_orientations_wxyz"], dtype=float)[:, palm_indices]
    solved_quats = np.asarray(run["solved_ik_orientations_wxyz"], dtype=float)[:, palm_indices]
    orientation_errors = orientation_error_deg(target_quats, solved_quats)

    return {
        "mean_palm_position_m": float(position_errors.mean()),
        "max_palm_position_m": float(position_errors.max()),
        "mean_left_palm_position_m": float(position_errors[:, 0].mean()),
        "mean_right_palm_position_m": float(position_errors[:, 1].mean()),
        "mean_palm_orientation_deg": float(orientation_errors.mean()),
        "max_palm_orientation_deg": float(orientation_errors.max()),
        "mean_left_palm_orientation_deg": float(orientation_errors[:, 0].mean()),
        "mean_right_palm_orientation_deg": float(orientation_errors[:, 1].mean()),
        "mean_position_score": float(summary["mean_position_score"]),
        "joint_limit_frames": int(summary["num_frames_near_joint_limit"]),
        "max_joint_step_rad": float(summary["max_abs_joint_step_rad"]),
        "orientation_costs": summary.get("human_orientation_costs", {}),
    }


def main() -> None:
    args = parse_args()
    print(
        "run | palm-pos mean/max m | palm-ori mean/max deg | "
        "left/right pos m | left/right ori deg | limit frames | max joint step rad"
    )
    for spec in args.run:
        label, path = parse_run(spec)
        metrics = run_metrics(path)
        print(
            f"{label} | "
            f"{metrics['mean_palm_position_m']:.4f}/{metrics['max_palm_position_m']:.4f} | "
            f"{metrics['mean_palm_orientation_deg']:.2f}/{metrics['max_palm_orientation_deg']:.2f} | "
            f"{metrics['mean_left_palm_position_m']:.4f}/{metrics['mean_right_palm_position_m']:.4f} | "
            f"{metrics['mean_left_palm_orientation_deg']:.2f}/{metrics['mean_right_palm_orientation_deg']:.2f} | "
            f"{metrics['joint_limit_frames']} | {metrics['max_joint_step_rad']:.4f}"
        )
        print(f"  orientation costs: {metrics['orientation_costs']}")


if __name__ == "__main__":
    main()

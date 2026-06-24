#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def as_str_list(x):
    return [str(v) for v in np.asarray(x).tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--robot-config", type=Path, default=Path("general_motion_retargeting/robot_configs/alex.json"))
    parser.add_argument("--near-limit-deg", type=float, default=1.0)
    args = parser.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)
    qpos = d["qpos"]

    robot_cfg = json.loads(args.robot_config.read_text())
    model = mujoco.MjModel.from_xml_path(robot_cfg["model_path"])

    out_base = args.npz_path.with_suffix("")
    summary_path = out_base.with_name(out_base.name + "_eval_summary.json")
    role_error_path = out_base.with_name(out_base.name + "_eval_role_errors.csv")
    limit_path = out_base.with_name(out_base.name + "_eval_joint_limits.csv")

    metrics = {
        "npz_path": str(args.npz_path),
        "qpos_shape": list(qpos.shape),
    }

    # Try to include original solver summary if present.
    solver_summary_path = out_base.with_name(out_base.name + "_summary.json")
    if solver_summary_path.exists():
        solver_summary = json.loads(solver_summary_path.read_text())
        for key in [
            "target_generation",
            "ik_roles",
            "mean_position_score",
            "max_position_score",
            "mean_hand_error_m",
            "max_hand_error_m",
            "mean_foot_error_m",
            "max_foot_error_m",
            "mean_pelvis_error_m",
            "max_pelvis_error_m",
            "max_abs_joint_step_rad",
            "max_root_step_m",
            "heading_canonicalization",
            "morphology_scales",
        ]:
            if key in solver_summary:
                metrics[key] = solver_summary[key]

    # Tracking role errors from stored target and solved positions.
    role_rows = []
    if "target_positions" in d and "solved_ik_positions" in d and "ik_roles" in d:
        target = d["target_positions"]
        solved = d["solved_ik_positions"]
        ik_roles = as_str_list(d["ik_roles"])

        err = np.linalg.norm(solved - target, axis=-1)

        for i, role in enumerate(ik_roles):
            vals = err[:, i]
            role_rows.append({
                "role": role,
                "mean_error_m": float(vals.mean()),
                "max_error_m": float(vals.max()),
                "p95_error_m": float(np.percentile(vals, 95)),
            })

        metrics["mean_all_role_error_m"] = float(err.mean())
        metrics["max_all_role_error_m"] = float(err.max())
        metrics["p95_all_role_error_m"] = float(np.percentile(err, 95))

    with role_error_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["role", "mean_error_m", "max_error_m", "p95_error_m"])
        writer.writeheader()
        writer.writerows(role_rows)

    # Smoothness.
    if qpos.shape[0] > 1:
        root_step = np.linalg.norm(np.diff(qpos[:, 0:3], axis=0), axis=1)
        joint_step = np.abs(np.diff(qpos[:, 7:], axis=0))

        flat = int(np.argmax(joint_step))
        f_idx, j_idx = np.unravel_index(flat, joint_step.shape)
        joint_names = robot_cfg["actuated_joint_order"]

        metrics.update({
            "mean_root_step_m": float(root_step.mean()),
            "max_root_step_m_eval": float(root_step.max()),
            "p95_root_step_m": float(np.percentile(root_step, 95)),
            "mean_abs_joint_step_rad": float(joint_step.mean()),
            "max_abs_joint_step_rad_eval": float(joint_step.max()),
            "p95_abs_joint_step_rad": float(np.percentile(joint_step, 95)),
            "largest_joint_step_frame": int(f_idx),
            "largest_joint_step_joint": joint_names[int(j_idx)],
            "largest_joint_step_rad": float(joint_step[f_idx, j_idx]),
            "largest_joint_step_deg": float(np.degrees(joint_step[f_idx, j_idx])),
        })

    # Joint limits and saturation.
    near_tol = np.deg2rad(args.near_limit_deg)
    hard_tol = 1e-4
    limit_rows = []
    meaningful_violations = 0

    for name in robot_cfg["actuated_joint_order"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        vals = qpos[:, qadr]

        below_amt = max(0.0, float(lo - vals.min()))
        above_amt = max(0.0, float(vals.max() - hi))
        max_violation = max(below_amt, above_amt)

        if max_violation > hard_tol:
            meaningful_violations += 1

        near = ((vals - lo) < near_tol) | ((hi - vals) < near_tol)
        near_count = int(near.sum())
        near_pct = 100.0 * near_count / len(vals)

        limit_rows.append({
            "joint": name,
            "qmin_rad": float(vals.min()),
            "qmax_rad": float(vals.max()),
            "lower_rad": float(lo),
            "upper_rad": float(hi),
            "max_violation_rad": float(max_violation),
            "near_limit_count": near_count,
            "near_limit_pct": float(near_pct),
        })

    with limit_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "joint",
                "qmin_rad",
                "qmax_rad",
                "lower_rad",
                "upper_rad",
                "max_violation_rad",
                "near_limit_count",
                "near_limit_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(limit_rows)

    metrics["meaningful_joint_limit_violation_count"] = int(meaningful_violations)
    metrics["top_near_limit_joints"] = sorted(
        [
            {
                "joint": r["joint"],
                "near_limit_pct": r["near_limit_pct"],
                "near_limit_count": r["near_limit_count"],
            }
            for r in limit_rows
        ],
        key=lambda x: x["near_limit_pct"],
        reverse=True,
    )[:10]

    # Rough foot sliding proxy from robot_positions if available.
    if "robot_positions" in d and "source_roles" in d:
        roles = as_str_list(d["source_roles"])
        robot_pos = d["robot_positions"]
        fps = float(metrics.get("output_fps", 30.0))

        foot_metrics = {}
        for foot in ["left_foot", "right_foot"]:
            if foot in roles:
                idx = roles.index(foot)
                p = robot_pos[:, idx, :]
                xy_step = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
                xy_speed = xy_step * fps

                z = p[:-1, 2]
                threshold = float(np.percentile(z, 20) + 0.02)
                contact_like = z <= threshold

                if contact_like.any():
                    foot_metrics[f"{foot}_contact_like_mean_xy_speed_mps"] = float(xy_speed[contact_like].mean())
                    foot_metrics[f"{foot}_contact_like_max_xy_speed_mps"] = float(xy_speed[contact_like].max())
                    foot_metrics[f"{foot}_contact_like_frame_count"] = int(contact_like.sum())

        metrics.update(foot_metrics)

    summary_path.write_text(json.dumps(metrics, indent=2))

    print()
    print("=== Evaluation summary ===")
    for key in [
        "target_generation",
        "mean_position_score",
        "max_position_score",
        "mean_all_role_error_m",
        "max_all_role_error_m",
        "p95_all_role_error_m",
        "meaningful_joint_limit_violation_count",
        "max_abs_joint_step_rad_eval",
        "largest_joint_step_joint",
        "largest_joint_step_deg",
    ]:
        if key in metrics:
            print(f"{key}: {metrics[key]}")

    print()
    print("Top near-limit joints:")
    for r in metrics["top_near_limit_joints"]:
        print(f"  {r['joint']:24s} {r['near_limit_pct']:6.1f}%")

    print()
    print("Wrote:")
    print(" ", summary_path)
    print(" ", role_error_path)
    print(" ", limit_path)


if __name__ == "__main__":
    main()

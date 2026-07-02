#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_DEFAULT = REPO_ROOT / "assets/alex/alex_floating_base_with_sites.xml"

# Position-only mapping: canonical human role -> real Alex body.
# No added palm/sole/contact sites.
ROLE_TO_ALEX_BODY = {
    "pelvis": "PELVIS_LINK",
    "torso": "TORSO_LINK",
    "head": "HEAD_LINK",

    "left_hip": "LEFT_THIGH",
    "right_hip": "RIGHT_THIGH",

    "left_shoulder": "LEFT_SHOULDER_Z_LINK",
    "right_shoulder": "RIGHT_SHOULDER_Z_LINK",

    "left_knee": "LEFT_SHIN",
    "left_ankle": "LEFT_ANKLE_Y_LINK",
    "right_knee": "RIGHT_SHIN",
    "right_ankle": "RIGHT_ANKLE_Y_LINK",

    "left_elbow": "LEFT_ELBOW_Y_LINK",
    "left_wrist": "LEFT_WRIST_X_LINK",
    "right_elbow": "RIGHT_ELBOW_Y_LINK",
    "right_wrist": "RIGHT_WRIST_X_LINK",
}

TARGET_WEIGHTS = {
    "pelvis": 4.0,
    "torso": 2.0,
    "head": 2.0,

    "left_hip": 0.8,
    "right_hip": 0.8,

    "left_shoulder": 0.8,
    "right_shoulder": 0.8,

    "left_knee": 1.0,
    "left_ankle": 1.5,
    "right_knee": 1.0,
    "right_ankle": 1.5,

    "left_elbow": 1.0,
    "left_wrist": 1.5,
    "right_elbow": 1.0,
    "right_wrist": 1.5,
}


# Orientation-delta tracking.
# These are real Alex bodies, not added sites.
ORI_TO_ALEX_BODY = {
    "pelvis": "PELVIS_LINK",
    "torso": "TORSO_LINK",
    "head": "HEAD_LINK",
    "left_foot": "LEFT_FOOT",
    "right_foot": "RIGHT_FOOT",
    "left_hand": "LEFT_GRIPPER_Z_LINK",
    "right_hand": "RIGHT_GRIPPER_Z_LINK",
}

ORI_WEIGHTS = {
    "pelvis": 0.5,
    "torso": 0.25,
    "head": 0.20,
    "left_foot": 0.35,
    "right_foot": 0.35,
    "left_hand": 0.20,
    "right_hand": 0.20,
}


def mj_name(model, objtype, idx):
    out = mujoco.mj_id2name(model, objtype, idx)
    return "" if out is None else out


def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Missing body in Alex model: {name}")
    return bid


def load_canonical(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    roles = [str(x) for x in z["roles"]]
    role_to_idx = {r: i for i, r in enumerate(roles)}
    positions = np.asarray(z["positions"], dtype=np.float64)
    fps = float(z["fps"])

    if "orientation_role_names" not in z.files or "orientation_mats" not in z.files:
        raise RuntimeError(
            f"{npz_path} must contain orientation_role_names and orientation_mats. "
            "Run scripts/build_canonical_orientation_frames_fresh.py first."
        )

    orientation_roles = [str(x) for x in z["orientation_role_names"]]
    ori_to_idx = {r: i for i, r in enumerate(orientation_roles)}
    orientation_mats = np.asarray(z["orientation_mats"], dtype=np.float64)

    return roles, role_to_idx, positions, fps, orientation_roles, ori_to_idx, orientation_mats


def estimate_source_scale(first_pos, role_to_idx):
    """
    Simple initial scaling from human to Alex-ish size.
    For the first pass, use human pelvis-to-head height.
    """
    pelvis = first_pos[role_to_idx["pelvis"]]
    head = first_pos[role_to_idx["head"]]
    human_height = np.linalg.norm(head - pelvis)
    # Alex default pelvis to head is around 0.61 m from earlier inspection.
    alex_pelvis_to_head = 0.615
    return alex_pelvis_to_head / max(human_height, 1e-6)


def make_initial_alignment_targets(first_src_pos, role_to_idx, scale):
    """
    Initial Alex rest-alignment target.

    The first human frame defines the source rest/reference pose for this clip.
    Alex is first solved into a corresponding pose. Later frames are expressed
    as deltas from this solved Alex pose.
    """
    src_pelvis0 = first_src_pos[role_to_idx["pelvis"]]

    targets = {}
    for role in ROLE_TO_ALEX_BODY:
        if role == "pelvis":
            targets[role] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            rel0 = first_src_pos[role_to_idx[role]] - src_pelvis0
            targets[role] = scale * rel0

    return targets


def make_targets_for_frame(src_pos, role_to_idx, first_src_pos, target_rest_positions, scale):
    """
    Rest-aligned morphology-delta target.

    For each role:
      target(t) = achieved_alex_rest(role)
                + scaled pelvis displacement from source rest
                + scaled role motion relative to pelvis from source rest

    This is much closer to the previous method than direct absolute scaling.
    """
    src_pelvis0 = first_src_pos[role_to_idx["pelvis"]]
    src_pelvis = src_pos[role_to_idx["pelvis"]]
    root_delta = scale * (src_pelvis - src_pelvis0)

    targets = {}
    for role in ROLE_TO_ALEX_BODY:
        if role == "pelvis":
            targets[role] = target_rest_positions[role] + root_delta
        else:
            rel0 = first_src_pos[role_to_idx[role]] - src_pelvis0
            rel = src_pos[role_to_idx[role]] - src_pelvis
            rel_delta = scale * (rel - rel0)
            targets[role] = target_rest_positions[role] + root_delta + rel_delta

    return targets


def clamp_hinge_joint_limits(model, qpos):
    """
    MuJoCo does not automatically clamp qpos during our manual least-squares IK.
    Clamp scalar hinge/slide joints that have limits. Skip free root.
    """
    for j in range(model.njnt):
        if not bool(model.jnt_limited[j]):
            continue
        jtype = int(model.jnt_type[j])
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            continue

        qadr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        qpos[qadr] = np.clip(qpos[qadr], lo, hi)


def rotmat_to_rotvec(R):
    """
    Convert rotation matrix to axis-angle vector.
    Returns world-frame small rotation vector.
    """
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-8:
        return 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ], dtype=np.float64)

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2.0 * np.sin(theta))

    return theta * axis


def body_xmat(data, body_id):
    return np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)


def make_orientation_targets_for_frame(src_ori_mats, ori_to_idx, first_src_ori_mats, target_rest_orientations):
    """
    World orientation delta transfer, matching the older working solver:
      R_delta_world = R_source_current @ R_source_rest.T
      R_target = R_delta_world @ R_alex_rest

    Do NOT copy absolute human orientation into Alex.
    Transfer only the world-frame change from the source rest frame.
    """
    targets = {}
    for role in ORI_TO_ALEX_BODY:
        R0 = first_src_ori_mats[ori_to_idx[role]]
        Rt = src_ori_mats[ori_to_idx[role]]
        delta_world = Rt @ R0.T
        targets[role] = delta_world @ target_rest_orientations[role]
    return targets


def solve_frame_position_ik(
    model,
    data,
    role_to_body_id,
    targets,
    q_init,
    *,
    ori_role_to_body_id=None,
    ori_targets=None,
    ori_scale=1.0,
    iters=30,
    damping=1e-3,
    step_scale=0.7,
    max_step_norm=0.20,
    root_reg=1e-3,
    posture_reg=1e-3,
):
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)

    nv = model.nv
    q_ref = q_init.copy()

    for _ in range(iters):
        rows = []
        rhs = []

        for role, bid in role_to_body_id.items():
            current = data.xpos[bid].copy()
            err = targets[role] - current
            weight = TARGET_WEIGHTS.get(role, 1.0)

            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)

            rows.append(np.sqrt(weight) * jacp)
            rhs.append(np.sqrt(weight) * err)

        if ori_role_to_body_id is not None and ori_targets is not None and ori_scale > 0.0:
            for role, bid in ori_role_to_body_id.items():
                R_current = body_xmat(data, bid)
                R_target = ori_targets[role]
                R_err = R_target @ R_current.T
                err_rot = rotmat_to_rotvec(R_err)

                weight = ori_scale * ORI_WEIGHTS.get(role, 0.0)
                if weight <= 0.0:
                    continue

                jacp = np.zeros((3, nv), dtype=np.float64)
                jacr = np.zeros((3, nv), dtype=np.float64)
                mujoco.mj_jacBody(model, data, jacp, jacr, bid)

                rows.append(np.sqrt(weight) * jacr)
                rhs.append(np.sqrt(weight) * err_rot)

        # regularize joint velocities
        rows.append(np.sqrt(posture_reg) * np.eye(nv))
        rhs.append(np.zeros(nv))

        A = np.vstack(rows)
        b = np.concatenate(rhs)

        H = A.T @ A + damping * np.eye(nv)
        g = A.T @ b

        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(H, g, rcond=None)[0]

        n = np.linalg.norm(dq)
        if n > max_step_norm:
            dq = dq * (max_step_norm / n)

        dq *= step_scale

        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)

        clamp_hinge_joint_limits(model, data.qpos)

        # Do not force root orientation here.
        # For get-up motions, forcing an upright root causes the limbs to absorb
        # the whole lying-to-standing rotation and produces bad branch choices.
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])

        mujoco.mj_forward(model, data)

    return data.qpos.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True, type=Path)
    ap.add_argument("--model", default=MODEL_DEFAULT, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--ik-iters", type=int, default=40)
    ap.add_argument("--ori-scale", type=float, default=1.0)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    roles, role_to_idx, src_positions, fps, orientation_roles, ori_to_idx, orientation_mats = load_canonical(args.canonical)

    missing = [r for r in ROLE_TO_ALEX_BODY if r not in role_to_idx]
    if missing:
        raise RuntimeError(f"Canonical missing required roles: {missing}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    role_to_body_id = {
        role: body_id(model, body_name)
        for role, body_name in ROLE_TO_ALEX_BODY.items()
    }

    missing_ori = [r for r in ORI_TO_ALEX_BODY if r not in ori_to_idx]
    if missing_ori:
        raise RuntimeError(f"Canonical missing required orientation roles: {missing_ori}")

    ori_role_to_body_id = {
        role: body_id(model, body_name)
        for role, body_name in ORI_TO_ALEX_BODY.items()
    }

    frame_ids = list(range(0, src_positions.shape[0], args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]

    first_src_pos = src_positions[frame_ids[0]]
    first_src_ori = orientation_mats[frame_ids[0]]
    scale = estimate_source_scale(first_src_pos, role_to_idx)

    print("Canonical:", args.canonical)
    print("Model:", args.model)
    print("Frames:", len(frame_ids), "stride:", args.stride)
    print("Source fps:", fps)
    print("Estimated human->Alex scale:", scale)
    print("Targets:")
    for role, body_name in ROLE_TO_ALEX_BODY.items():
        print(f"  {role:14s} -> {body_name}")

    print("Orientation targets:")
    for role, body_name in ORI_TO_ALEX_BODY.items():
        print(f"  {role:14s} -> {body_name}  weight={ORI_WEIGHTS.get(role, 0.0) * args.ori_scale:.3f}")

    qpos_list = []
    err_list = []
    target_pos_list = []
    achieved_pos_list = []
    target_ori_list = []
    achieved_ori_list = []
    ori_err_deg_list = []

    q = np.zeros(model.nq)
    q[3] = 1.0  # root quaternion w

    print()
    print("Solving initial Alex rest-alignment pose...")
    initial_targets = make_initial_alignment_targets(first_src_pos, role_to_idx, scale)
    q = solve_frame_position_ik(
        model,
        data,
        role_to_body_id,
        initial_targets,
        q,
        iters=max(args.ik_iters * 3, 80),
    )
    mujoco.mj_forward(model, data)

    target_rest_positions = {
        role: data.xpos[bid].copy()
        for role, bid in role_to_body_id.items()
    }

    target_rest_orientations = {
        role: body_xmat(data, bid).copy()
        for role, bid in ori_role_to_body_id.items()
    }

    rest_errs = {
        role: float(np.linalg.norm(initial_targets[role] - data.xpos[bid]))
        for role, bid in role_to_body_id.items()
    }
    print("Initial rest-alignment errors:")
    for role, err in rest_errs.items():
        print(f"  {role:12s} {err:.4f}")
    print(f"  mean={np.mean(list(rest_errs.values())):.4f} max={np.max(list(rest_errs.values())):.4f}")

    for ti, src_i in enumerate(frame_ids):
        targets = make_targets_for_frame(
            src_positions[src_i],
            role_to_idx,
            first_src_pos,
            target_rest_positions,
            scale,
        )

        ori_targets = make_orientation_targets_for_frame(
            orientation_mats[src_i],
            ori_to_idx,
            first_src_ori,
            target_rest_orientations,
        )

        q = solve_frame_position_ik(
            model,
            data,
            role_to_body_id,
            targets,
            q,
            ori_role_to_body_id=ori_role_to_body_id,
            ori_targets=ori_targets,
            ori_scale=args.ori_scale,
            iters=args.ik_iters,
        )

        mujoco.mj_forward(model, data)

        errs = {}
        for role, bid in role_to_body_id.items():
            e = np.linalg.norm(targets[role] - data.xpos[bid])
            errs[role] = float(e)

        qpos_list.append(q.copy())
        err_list.append(errs)

        role_order = list(ROLE_TO_ALEX_BODY.keys())
        target_pos_list.append(np.asarray([targets[r] for r in role_order], dtype=np.float64))
        achieved_pos_list.append(np.asarray([data.xpos[role_to_body_id[r]].copy() for r in role_order], dtype=np.float64))

        ori_order = list(ORI_TO_ALEX_BODY.keys())
        target_ori_list.append(np.asarray([ori_targets[r] for r in ori_order], dtype=np.float64))
        achieved_ori_list.append(np.asarray([body_xmat(data, ori_role_to_body_id[r]).copy() for r in ori_order], dtype=np.float64))

        ori_errs = []
        for r in ori_order:
            R_current = body_xmat(data, ori_role_to_body_id[r])
            R_target = ori_targets[r]
            R_err = R_target @ R_current.T
            rv = rotmat_to_rotvec(R_err)
            ori_errs.append(np.linalg.norm(rv) * 180.0 / np.pi)
        ori_err_deg_list.append(np.asarray(ori_errs, dtype=np.float64))

        mean_err = np.mean(list(errs.values()))
        max_err = np.max(list(errs.values()))
        print(f"frame {ti:04d} source={src_i:04d} mean_err={mean_err:.4f} max_err={max_err:.4f}")

    metadata = {
        "format": "alex_posori_qp_fresh_worlddelta_v1",
        "canonical": str(args.canonical),
        "model": str(args.model),
        "role_to_alex_body": ROLE_TO_ALEX_BODY,
        "target_weights": TARGET_WEIGHTS,
        "orientation_cost": args.ori_scale,
        "ori_to_alex_body": ORI_TO_ALEX_BODY,
        "ori_weights": ORI_WEIGHTS,
        "scale": scale,
        "frame_ids": frame_ids,
        "notes": [
            "Fresh position-only Alex body IK with initial rest-alignment and rest-pose delta targets.",
            "No palm/sole/contact sites used.",
            "No orientation cost used.",
            "Targets are canonical human role positions mapped to real Alex body origins.",
        ],
    }

    np.savez(
        args.out,
        qpos=np.asarray(qpos_list),
        target_positions=np.asarray(target_pos_list),
        achieved_positions=np.asarray(achieved_pos_list),
        orientation_role_names=np.asarray(list(ORI_TO_ALEX_BODY.keys()), dtype=object),
        target_orientations=np.asarray(target_ori_list),
        achieved_orientations=np.asarray(achieved_ori_list),
        orientation_errors_deg=np.asarray(ori_err_deg_list),
        source_frame_ids=np.asarray(frame_ids, dtype=np.int64),
        role_names=np.asarray(list(ROLE_TO_ALEX_BODY.keys()), dtype=object),
        alex_body_names=np.asarray(list(ROLE_TO_ALEX_BODY.values()), dtype=object),
        errors_json=np.asarray(json.dumps(err_list, indent=2), dtype=object),
        metadata_json=np.asarray(json.dumps(metadata, indent=2), dtype=object),
    )

    print()
    print("Wrote:", args.out)
    print("qpos:", np.asarray(qpos_list).shape)


if __name__ == "__main__":
    main()

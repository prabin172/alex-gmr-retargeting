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


def measure_alex_pelvis_to_head(model, data, role_to_body_id):
    """Query Alex's pelvis-to-head distance at the default (all-joints-zero) pose."""
    q_rest = np.zeros(model.nq)
    q_rest[3] = 1.0
    data.qpos[:] = q_rest
    mujoco.mj_forward(model, data)
    pelvis_pos = data.xpos[role_to_body_id["pelvis"]].copy()
    head_pos = data.xpos[role_to_body_id["head"]].copy()
    return float(np.linalg.norm(head_pos - pelvis_pos))


def estimate_source_scale(first_pos, role_to_idx, alex_pelvis_to_head):
    """Global scale for root translation: ratio of Alex to human pelvis-to-head height."""
    pelvis = first_pos[role_to_idx["pelvis"]]
    head = first_pos[role_to_idx["head"]]
    human_height = np.linalg.norm(head - pelvis)
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


def compute_per_role_scales(first_src_pos, role_to_idx, target_rest_positions, clamp=(0.4, 2.5)):
    """
    Per-role scale factors from comparing source vs Alex achieved-rest proportions.

    For each non-root role, we compare the pelvis-relative distance in the human
    rest frame to the pelvis-relative distance Alex achieved after initial IK.
    This captures limb-proportion differences (e.g. shorter arms on Alex) that a
    single global scale misses.

    The global scale is not needed here; the root translation uses its own scale.
    """
    src_pelvis = first_src_pos[role_to_idx["pelvis"]]
    alex_pelvis = target_rest_positions["pelvis"]

    scales = {}
    for role in ROLE_TO_ALEX_BODY:
        if role == "pelvis":
            scales[role] = 1.0
            continue
        src_dist = float(np.linalg.norm(first_src_pos[role_to_idx[role]] - src_pelvis))
        alex_dist = float(np.linalg.norm(target_rest_positions[role] - alex_pelvis))
        if src_dist < 1e-6:
            scales[role] = 1.0
        else:
            scales[role] = float(np.clip(alex_dist / src_dist, clamp[0], clamp[1]))

    return scales


def make_targets_for_frame(src_pos, role_to_idx, first_src_pos, target_rest_positions, root_scale, role_scales):
    """
    Rest-aligned morphology-delta target with per-role scales.

    For each role:
      target(t) = achieved_alex_rest(role)
                + root_scale * pelvis displacement from source rest
                + role_scales[role] * role motion relative to pelvis from source rest
    """
    src_pelvis0 = first_src_pos[role_to_idx["pelvis"]]
    src_pelvis = src_pos[role_to_idx["pelvis"]]
    root_delta = root_scale * (src_pelvis - src_pelvis0)

    targets = {}
    for role in ROLE_TO_ALEX_BODY:
        if role == "pelvis":
            targets[role] = target_rest_positions[role] + root_delta
        else:
            rel0 = first_src_pos[role_to_idx[role]] - src_pelvis0
            rel = src_pos[role_to_idx[role]] - src_pelvis
            s = role_scales.get(role, root_scale)
            rel_delta = s * (rel - rel0)
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


def _within_k_hops(model: mujoco.MjModel, b1: int, b2: int, k: int) -> bool:
    """True if b1 is an ancestor of b2 (or vice versa) within k steps in the kinematic tree."""
    cur = b2
    for _ in range(k):
        cur = int(model.body_parentid[cur])
        if cur == b1:
            return True
        if cur == 0:
            break
    cur = b1
    for _ in range(k):
        cur = int(model.body_parentid[cur])
        if cur == b2:
            return True
        if cur == 0:
            break
    return False


def self_collision_rows(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nv: int,
    weight: float,
    margin: float,
    gain: float,
    exclude_hops: int = 2,
) -> tuple[list, list]:
    """
    Build QP rows that push apart self-colliding body pairs.

    MuJoCo excludes direct parent-child contacts (1 hop) automatically.
    We also exclude bodies within `exclude_hops` of each other in the kinematic
    tree — this catches structural near-misses like HEAD↔TORSO (2 hops, always
    overlapping due to large geom radii, not a real cross-body collision).

    For each remaining contact with dist < margin, adds one row:
        sqrt(w) * n·(J1 - J2) @ dq = sqrt(w) * min(penetration, 0.05) * gain
    where n is the contact normal pushing b1 away from b2.
    """
    rows: list[np.ndarray] = []
    rhs_vals: list[float] = []

    sqw = float(np.sqrt(weight))

    for c_idx in range(data.ncon):
        ct = data.contact[c_idx]
        b1 = int(model.geom_bodyid[ct.geom1])
        b2 = int(model.geom_bodyid[ct.geom2])

        # Skip floor/static contacts
        if b1 == 0 or b2 == 0:
            continue

        # Skip bodies that are kinematically close (structural near-misses)
        if _within_k_hops(model, b1, b2, exclude_hops):
            continue

        penetration = margin - float(ct.dist)   # positive when too close or penetrating
        if penetration <= 0:
            continue

        # Contact normal — sign-correct so it pushes b1 away from b2
        normal = ct.frame[:3].copy()
        if float(np.dot(normal, data.xpos[b1] - data.xpos[b2])) < 0:
            normal = -normal

        # Jacobian at the contact point for each body
        jacp1 = np.zeros((3, nv), dtype=np.float64)
        jacp2 = np.zeros((3, nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp1, None, ct.pos, b1)
        mujoco.mj_jac(model, data, jacp2, None, ct.pos, b2)

        jac_sep = normal @ (jacp1 - jacp2)       # (nv,) separation Jacobian

        if np.linalg.norm(jac_sep) < 1e-9:
            continue

        target = min(penetration, 0.05) * gain   # cap to 5 cm/iter

        rows.append(sqw * jac_sep)
        rhs_vals.append(sqw * target)

    return rows, rhs_vals


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
    coll_weight=0.0,
    coll_margin=0.02,
    coll_gain=5.0,
    coll_hops=2,
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

        # Pull actuated joints toward q_ref (start of this frame's solve).
        # Root DOFs (0-5) are left at zero — the position/orientation tasks steer them.
        # Actuated joints: dq[6:35] corresponds to qpos[7:36].
        desired_dq = np.zeros(nv)
        desired_dq[6:] = q_ref[7:] - data.qpos[7:]
        rows.append(np.sqrt(posture_reg) * np.eye(nv))
        rhs.append(np.sqrt(posture_reg) * desired_dq)

        # Self-collision repulsion: push apart any non-adjacent penetrating body pairs.
        if coll_weight > 0.0:
            coll_rows, coll_rhs = self_collision_rows(
                model, data, nv, coll_weight, coll_margin, coll_gain,
                exclude_hops=coll_hops,
            )
            rows.extend(coll_rows)
            rhs.extend([np.array([v]) for v in coll_rhs])

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
    ap.add_argument("--coll-weight", type=float, default=5.0,
                    help="Self-collision repulsion weight (0=disabled, default: 5.0)")
    ap.add_argument("--coll-margin", type=float, default=0.02,
                    help="Repulsion activates when bodies within this margin (m) of contact (default: 0.02)")
    ap.add_argument("--coll-gain", type=float, default=5.0,
                    help="Correction speed: target separation per metre of penetration (default: 5.0)")
    ap.add_argument("--coll-hops", type=int, default=2,
                    help="Kinematic-tree hop threshold for exclusion beyond MuJoCo's built-in 1-hop filter "
                         "(default: 2, which also excludes grandparent-grandchild pairs like HEAD↔TORSO)")
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

    alex_pelvis_to_head = measure_alex_pelvis_to_head(model, data, role_to_body_id)
    root_scale = estimate_source_scale(first_src_pos, role_to_idx, alex_pelvis_to_head)

    print("Canonical:", args.canonical)
    print("Model:", args.model)
    print("Frames:", len(frame_ids), "stride:", args.stride)
    print("Source fps:", fps)
    print(f"Alex pelvis-to-head (model): {alex_pelvis_to_head:.4f} m")
    print("Estimated root scale:", root_scale)
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
    n_self_coll_list = []

    q = np.zeros(model.nq)
    q[3] = 1.0  # root quaternion w

    coll_kwargs = dict(
        coll_weight=args.coll_weight,
        coll_margin=args.coll_margin,
        coll_gain=args.coll_gain,
        coll_hops=args.coll_hops,
    )

    print(f"Self-collision: weight={args.coll_weight}  margin={args.coll_margin} m  "
          f"gain={args.coll_gain}  exclude_hops={args.coll_hops}")
    print()
    print("Solving initial Alex rest-alignment pose...")
    initial_targets = make_initial_alignment_targets(first_src_pos, role_to_idx, root_scale)
    q = solve_frame_position_ik(
        model,
        data,
        role_to_body_id,
        initial_targets,
        q,
        iters=max(args.ik_iters * 3, 80),
        **coll_kwargs,
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

    # Per-role scales from comparing source vs Alex achieved-rest proportions.
    role_scales = compute_per_role_scales(first_src_pos, role_to_idx, target_rest_positions)

    rest_errs = {
        role: float(np.linalg.norm(initial_targets[role] - data.xpos[bid]))
        for role, bid in role_to_body_id.items()
    }
    print("Initial rest-alignment errors:")
    for role, err in rest_errs.items():
        print(f"  {role:12s} {err:.4f}")
    print(f"  mean={np.mean(list(rest_errs.values())):.4f} max={np.max(list(rest_errs.values())):.4f}")
    print("Per-role scales:")
    for role, s in role_scales.items():
        print(f"  {role:14s} {s:.4f}")

    for ti, src_i in enumerate(frame_ids):
        targets = make_targets_for_frame(
            src_positions[src_i],
            role_to_idx,
            first_src_pos,
            target_rest_positions,
            root_scale,
            role_scales,
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
            **coll_kwargs,
        )

        mujoco.mj_forward(model, data)

        # Count remaining self-collisions post-solve (using same filter as the constraint)
        n_self_coll = 0
        for c_idx in range(data.ncon):
            ct = data.contact[c_idx]
            cb1 = int(model.geom_bodyid[ct.geom1])
            cb2 = int(model.geom_bodyid[ct.geom2])
            if cb1 > 0 and cb2 > 0 and ct.dist < 0:
                if not _within_k_hops(model, cb1, cb2, args.coll_hops):
                    n_self_coll += 1

        errs = {}
        for role, bid in role_to_body_id.items():
            e = np.linalg.norm(targets[role] - data.xpos[bid])
            errs[role] = float(e)

        qpos_list.append(q.copy())
        err_list.append(errs)
        n_self_coll_list.append(n_self_coll)

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
        coll_str = f"  coll={n_self_coll}" if n_self_coll > 0 else ""
        print(f"frame {ti:04d} source={src_i:04d} mean_err={mean_err:.4f} max_err={max_err:.4f}{coll_str}")

    metadata = {
        "format": "alex_posori_qp_fresh_worlddelta_v2",
        "canonical": str(args.canonical),
        "model": str(args.model),
        "role_to_alex_body": ROLE_TO_ALEX_BODY,
        "target_weights": TARGET_WEIGHTS,
        "orientation_cost": args.ori_scale,
        "ori_to_alex_body": ORI_TO_ALEX_BODY,
        "ori_weights": ORI_WEIGHTS,
        "root_scale": root_scale,
        "role_scales": role_scales,
        "alex_pelvis_to_head": alex_pelvis_to_head,
        "frame_ids": frame_ids,
        "notes": [
            "Position + orientation QP IK with world-delta orientation transfer.",
            "Per-role morphology scales computed from source vs Alex achieved-rest proportions.",
            "Root translation uses global pelvis-to-head scale queried from the MuJoCo model.",
            "Posture regularization pulls actuated joints toward start-of-frame reference.",
        ],
    }

    n_coll_arr = np.asarray(n_self_coll_list, dtype=np.int32)
    print(f"\nSelf-collision summary: {n_coll_arr.sum()} total penetrating frames "
          f"({(n_coll_arr > 0).mean() * 100:.1f}% of frames)")

    np.savez(
        args.out,
        qpos=np.asarray(qpos_list),
        fps=np.float64(fps),
        self_collision_counts=n_coll_arr,
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

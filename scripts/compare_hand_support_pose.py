#!/usr/bin/env python3
"""Compare IK-solved right arm angles during hand-support frames against a target pose.

1. Plots the 4 right arm joint angles over the full motion, with hand-contact
   frames shaded and target angles shown as dashed lines.
2. Finds the frame during contact closest to the target in joint space.
3. Renders that frame side-by-side with the manually-tuned target pose.

Usage:
    conda run -n gmr python scripts/compare_hand_support_pose.py
    MUJOCO_GL=egl conda run -n gmr python scripts/compare_hand_support_pose.py --render
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NPZ  = Path("outputs/grounded_highori/standup_02_grounded_highori.npz")
MODEL_XML = Path("assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml")
OUT_PLOT  = Path("outputs/renders/hand_support_angles.png")
OUT_VIDEO = Path("outputs/renders/hand_support_comparison.mp4")

# Target pose (degrees)
TARGET_DEG = {
    "RIGHT_SHOULDER_Z": +20.0,
    "RIGHT_ELBOW_Y":    -30.0,
    "RIGHT_WRIST_Z":    +90.0,
    "RIGHT_WRIST_X":    -80.0,
}

# qpos indices for right arm joints (from model)
RIGHT_ARM_QIDX = {
    "RIGHT_SHOULDER_Y": 29,
    "RIGHT_SHOULDER_X": 30,
    "RIGHT_SHOULDER_Z": 31,
    "RIGHT_ELBOW_Y":    32,
    "RIGHT_WRIST_Z":    33,
    "RIGHT_WRIST_X":    34,
    "RIGHT_GRIPPER_Z":  35,
}
CONTACT_COL_RIGHT_HAND = 10   # RIGHT_GRIPPER_Z_LINK


def _base_qpos_deadlift(model):
    """Same deadlift base pose used in visualize_arm_pose_transition.py."""
    import mujoco
    TORSO_PITCH = 0.70
    q = np.zeros(model.nq)
    hw = TORSO_PITCH / 2
    q[2] = 2.0; q[3] = np.cos(hw); q[5] = np.sin(hw)

    def _set(name, val):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[model.jnt_qposadr[jid]] = val

    _set("LEFT_HIP_Y", -0.50);  _set("RIGHT_HIP_Y", -0.50)
    _set("LEFT_KNEE_Y", 0.30);  _set("RIGHT_KNEE_Y", 0.30)
    _set("LEFT_ANKLE_Y", -0.20); _set("RIGHT_ANKLE_Y", -0.20)
    _set("LEFT_SHOULDER_Y", -TORSO_PITCH * 0.9)
    _set("RIGHT_SHOULDER_Y", -TORSO_PITCH * 0.9)

    # Apply target arm angles
    for jname, deg in TARGET_DEG.items():
        _set(jname, np.deg2rad(deg))

    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    q[2] -= (np.min(data.geom_xpos[:, 2]) - 0.01)
    return q


def plot_angles(qpos: np.ndarray, contact: np.ndarray, fps: float) -> None:
    joints = list(TARGET_DEG.keys())
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    t = np.arange(len(qpos)) / fps

    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Right arm joint angles — standup_02\n(shaded = right hand contact)",
                 fontsize=13, fontweight="bold")

    contact_frames = np.where(contact[:, CONTACT_COL_RIGHT_HAND])[0]

    for ax, jname, color in zip(axes, joints, colors):
        qidx   = RIGHT_ARM_QIDX[jname]
        angles = np.rad2deg(qpos[:, qidx])
        target = TARGET_DEG[jname]

        ax.plot(t, angles, color=color, lw=1.5, label=jname.replace("RIGHT_", ""))

        # Shade hand-contact regions
        in_contact = contact[:, CONTACT_COL_RIGHT_HAND].astype(bool)
        for start, end in _contiguous_regions(in_contact):
            ax.axvspan(t[start], t[min(end, len(t)-1)], alpha=0.15, color="yellow")

        ax.axhline(target, color=color, lw=1.2, ls="--", alpha=0.8,
                   label=f"target = {target:+.0f}°")

        ax.set_ylabel("deg", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {OUT_PLOT}")


def _contiguous_regions(mask: np.ndarray):
    """Yield (start, end) index pairs for contiguous True runs."""
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if mask[0]:  starts = np.r_[0, starts]
    if mask[-1]: ends   = np.r_[ends, len(mask)]
    return zip(starts, ends)


def _palm_normal(R: np.ndarray) -> np.ndarray:
    """Extract palm-facing normal from a 3x3 rotation matrix (world frame +Z column)."""
    return R[:, 2]   # local Z axis in world coords


def _rot_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angular distance between two rotation matrices in degrees."""
    R_rel  = R1.T @ R2
    trace  = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(trace))


def find_best_match(qpos: np.ndarray, contact: np.ndarray,
                    target_ori: np.ndarray, achieved_ori: np.ndarray) -> int:
    """Frame during right-hand contact closest to manual target angles in joint space."""
    import mujoco

    # Index of right_hand in orientation_role_names
    ORI_IDX_RIGHT_HAND = 6

    contact_mask = contact[:, CONTACT_COL_RIGHT_HAND].astype(bool)
    target_rad   = np.array([np.deg2rad(TARGET_DEG[j]) for j in TARGET_DEG])
    qidxs        = [RIGHT_ARM_QIDX[j] for j in TARGET_DEG]

    best_frame, best_dist = -1, np.inf
    for i in np.where(contact_mask)[0]:
        dist = np.linalg.norm(qpos[i, qidxs] - target_rad)
        if dist < best_dist:
            best_dist, best_frame = dist, i

    frame = best_frame

    # --- IK target orientation (from human mocap retargeting) ---
    R_ik_target   = target_ori[frame, ORI_IDX_RIGHT_HAND]    # what IK was asked for
    R_ik_achieved = achieved_ori[frame, ORI_IDX_RIGHT_HAND]  # what IK actually got

    # --- Manual target orientation: run FK on manual pose ---
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data  = mujoco.MjData(model)
    data.qpos[:] = _base_qpos_deadlift(model)
    mujoco.mj_forward(model, data)
    # Find right gripper body to get its orientation
    rg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "RIGHT_GRIPPER_Z_LINK")
    R_manual = data.xmat[rg_id].reshape(3, 3)

    pn_ik_tgt  = _palm_normal(R_ik_target)
    pn_ik_ach  = _palm_normal(R_ik_achieved)
    pn_manual  = _palm_normal(R_manual)

    ori_err_ik   = _rot_angle_deg(R_ik_target, R_ik_achieved)
    ori_err_man  = _rot_angle_deg(R_ik_target, R_manual)

    print(f"\nBest-match frame: {frame}  (joint-space dist {np.rad2deg(best_dist):.1f}° RMS)\n")
    print(f"{'Joint':<24} {'IK target':>12} {'IK solved':>12} {'Manual target':>14} {'IK err':>8} {'Man err':>8}")
    print("-" * 82)
    for jname in TARGET_DEG:
        solved = np.rad2deg(qpos[frame, RIGHT_ARM_QIDX[jname]])
        tgt    = TARGET_DEG[jname]
        print(f"  {jname:<22} {'N/A (pos)':>12} {solved:>+11.1f}° {tgt:>+13.1f}°"
              f" {solved-tgt:>+7.1f}°  {'':>7}")

    print()
    print(f"  {'Palm normal (IK tgt)':<22} {pn_ik_tgt[0]:+.3f} {pn_ik_tgt[1]:+.3f} {pn_ik_tgt[2]:+.3f}")
    print(f"  {'Palm normal (IK ach)':<22} {pn_ik_ach[0]:+.3f} {pn_ik_ach[1]:+.3f} {pn_ik_ach[2]:+.3f}  "
          f"(ori err vs target: {ori_err_ik:.1f}°)")
    print(f"  {'Palm normal (manual)':<22} {pn_manual[0]:+.3f} {pn_manual[1]:+.3f} {pn_manual[2]:+.3f}  "
          f"(ori err vs target: {ori_err_man:.1f}°)")

    return frame


def render_comparison(best_frame: int, qpos: np.ndarray) -> None:
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    model    = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    def make_cam():
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance  = 2.4
        cam.azimuth   = 150.0
        cam.elevation = -10.0
        return cam

    def render_qpos(q: np.ndarray, cam, label: str) -> np.ndarray:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3] + np.array([0.05, 0.0, 0.25])
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        pil  = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 10), label, font=font, fill=(220, 220, 100))
        return np.array(pil)

    q_data   = qpos[best_frame].copy()
    q_target = _base_qpos_deadlift(model)

    left_img  = render_qpos(q_data,   make_cam(), f"IK solved  (frame {best_frame})")
    right_img = render_qpos(q_target, make_cam(), "Manual target pose")

    frame = np.concatenate([left_img, right_img], axis=1)
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(OUT_VIDEO), fps=30, codec="libx264", quality=8) as w:
        for _ in range(90):   # 3 s hold so any player can open it
            w.append_data(frame)

    renderer.close()
    print(f"Saved comparison: {OUT_VIDEO}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="Also render side-by-side video")
    args = ap.parse_args()

    z           = np.load(NPZ, allow_pickle=True)
    qpos        = np.asarray(z["qpos"],               dtype=np.float64)
    contact     = np.asarray(z["contact_labels"],     dtype=np.float32)
    target_ori  = np.asarray(z["target_orientations"],   dtype=np.float64)
    achieved_ori= np.asarray(z["achieved_orientations"], dtype=np.float64)
    fps         = float(z["fps"]) if "fps" in z.files else 30.0

    print(f"Loaded {NPZ.name}: {len(qpos)} frames @ {fps:.0f} fps")
    print(f"Right hand contact frames: {contact[:, CONTACT_COL_RIGHT_HAND].sum():.0f}")

    plot_angles(qpos, contact, fps)
    best_frame = find_best_match(qpos, contact, target_ori, achieved_ori)

    if args.render:
        render_comparison(best_frame, qpos)


if __name__ == "__main__":
    main()

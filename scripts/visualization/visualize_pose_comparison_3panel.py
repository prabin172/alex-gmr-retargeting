#!/usr/bin/env python3
"""Three-panel pose comparison at the best hand-support frame.

Panel 1 — Robot in manually-tuned target pose (arm_pose_transition values)
Panel 2 — Robot in IK-solved pose at best-match contact frame
Panel 3 — Human stick figure (target_positions) at that same frame

Usage:
    MUJOCO_GL=egl conda run -n gmr python scripts/visualization/visualize_pose_comparison_3panel.py
"""
from __future__ import annotations
from pathlib import Path

import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
NPZ       = Path("outputs/grounded_highori/standup_02_grounded_highori.npz")
MODEL_XML = Path("assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml")
OUT_IMG   = Path("outputs/renders/pose_comparison_3panel.png")
OUT_VID   = Path("outputs/renders/pose_comparison_3panel.mp4")

PANEL_W, PANEL_H = 640, 480
TORSO_PITCH = 0.70

# Manual target joint angles (degrees)
TARGET_DEG = {
    "RIGHT_SHOULDER_Z": +20.0,
    "RIGHT_ELBOW_Y":    -30.0,
    "RIGHT_WRIST_Z":    +90.0,
    "RIGHT_WRIST_X":    -80.0,
}

RIGHT_ARM_QIDX = {
    "RIGHT_SHOULDER_Z": 31,
    "RIGHT_ELBOW_Y":    32,
    "RIGHT_WRIST_Z":    33,
    "RIGHT_WRIST_X":    34,
}
CONTACT_COL_RIGHT_HAND = 10

# Skeleton edges — (role_a, role_b, color, highlight)
# highlight=True keeps the color; highlight=False → light gray
_GRAY = (170, 170, 170)
SKELETON_EDGES = [
    ("pelvis",         "torso",          _GRAY,           False),
    ("torso",          "head",           _GRAY,           False),
    ("pelvis",         "left_hip",       _GRAY,           False),
    ("left_hip",       "left_knee",      _GRAY,           False),
    ("left_knee",      "left_ankle",     _GRAY,           False),
    ("pelvis",         "right_hip",      _GRAY,           False),
    ("right_hip",      "right_knee",     _GRAY,           False),
    ("right_knee",     "right_ankle",    _GRAY,           False),
    ("torso",          "left_shoulder",  _GRAY,           False),
    ("left_shoulder",  "left_elbow",     _GRAY,           False),
    ("left_elbow",     "left_wrist",     _GRAY,           False),
    ("torso",          "right_shoulder", (255, 200,  80), True),
    ("right_shoulder", "right_elbow",    (255, 160,  50), True),
    ("right_elbow",    "right_wrist",    (255, 120,  30), True),
]
RIGHT_ARM_ROLES = {"right_shoulder", "right_elbow", "right_wrist"}

FINGER_LENGTH = 0.18   # metres — length of hand-orientation indicator line


# ---------------------------------------------------------------------------
# Base pose helpers
# ---------------------------------------------------------------------------

def _set_joint(model, q, name, val):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    q[model.jnt_qposadr[jid]] = val


def _ground(model, q):
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    q[2] -= (np.min(d.geom_xpos[:, 2]) - 0.01)
    return q


def _manual_target_qpos(model):
    """Deadlift base + manual arm angles."""
    q = np.zeros(model.nq)
    hw = TORSO_PITCH / 2
    q[2] = 2.0; q[3] = np.cos(hw); q[5] = np.sin(hw)
    for name, val in [
        ("LEFT_HIP_Y",  -0.50), ("RIGHT_HIP_Y",  -0.50),
        ("LEFT_KNEE_Y",  0.30), ("RIGHT_KNEE_Y",  0.30),
        ("LEFT_ANKLE_Y",-0.20), ("RIGHT_ANKLE_Y",-0.20),
        ("LEFT_SHOULDER_Y",  -TORSO_PITCH * 0.9),
        ("RIGHT_SHOULDER_Y", -TORSO_PITCH * 0.9),
    ]:
        _set_joint(model, q, name, val)
    for jname, deg in TARGET_DEG.items():
        _set_joint(model, q, jname, np.deg2rad(deg))
    return _ground(model, q)


def _best_contact_frame(qpos, contact):
    mask       = contact[:, CONTACT_COL_RIGHT_HAND].astype(bool)
    target_rad = np.array([np.deg2rad(TARGET_DEG[j]) for j in TARGET_DEG])
    qidxs      = [RIGHT_ARM_QIDX[j] for j in TARGET_DEG]
    best, dist = -1, np.inf
    for i in np.where(mask)[0]:
        d = np.linalg.norm(qpos[i, qidxs] - target_rad)
        if d < dist:
            dist, best = d, i
    return best


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _make_cam(lookat, azimuth=150.0, elevation=-10.0, distance=2.6):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance  = distance
    cam.azimuth   = azimuth
    cam.elevation = elevation
    return cam


def _render_robot(model, data, renderer, q, cam, title, subtitle=""):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    return _overlay(img, title, subtitle)


def _overlay(img, title, subtitle=""):
    pil  = Image.fromarray(img)
    W, H = pil.size
    try:
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        fb = fs = ImageFont.load_default()
    bh = 52 if subtitle else 34
    banner = Image.new("RGBA", (W, bh), (0, 0, 0, 170))
    rgba   = pil.convert("RGBA"); rgba.paste(banner, (0, 0), banner)
    pil    = rgba.convert("RGB"); draw = ImageDraw.Draw(pil)
    draw.text((10, 8),  title,    font=fb, fill=(220, 220, 100))
    if subtitle:
        draw.text((10, 32), subtitle, font=fs, fill=(180, 180, 180))
    return np.array(pil)


# ---------------------------------------------------------------------------
# Human stick figure panel
# ---------------------------------------------------------------------------

def _camera_basis(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(-el_deg)
    right    = np.array([np.cos(az), np.sin(az), 0.0])
    view_fwd = np.array([-np.sin(az)*np.cos(el), np.cos(az)*np.cos(el), -np.sin(el)])
    up = np.cross(right, view_fwd)
    up /= max(np.linalg.norm(up), 1e-9)
    return right, up


def _project(positions, center, right, up, scale, W, H):
    pts = []
    for p in positions:
        d  = p - center
        px = int(W // 2 + np.dot(d, right) * scale)
        py = int(H // 2 - np.dot(d, up)    * scale)
        pts.append((px, py))
    return pts


def _seg_box(draw, pa, pb, color, width_px, outline=(20, 20, 20)):
    """Draw a filled 2D rectangle between two projected points."""
    ax, ay = pa; bx, by = pb
    dx, dy = bx - ax, by - ay
    length = max(np.sqrt(dx*dx + dy*dy), 1.0)
    px, py = -dy / length, dx / length   # perpendicular unit vector
    h = width_px / 2
    corners = [
        (ax + px*h, ay + py*h),
        (bx + px*h, by + py*h),
        (bx - px*h, by - py*h),
        (ax - px*h, ay - py*h),
    ]
    draw.polygon(corners, fill=color)
    draw.polygon(corners, outline=outline)


def _draw_human_panel(positions, role_names, frame_idx,
                      hand_R=None,
                      az=330.0, el=-10.0):
    role_to_idx = {r: i for i, r in enumerate(role_names)}
    right_v, up_v = _camera_basis(az, el)
    center = positions[role_to_idx["pelvis"]].copy() if "pelvis" in role_to_idx else positions.mean(0)
    scale  = PANEL_H * 0.55 / 1.7

    pts = _project(positions, center, right_v, up_v, scale, PANEL_W, PANEL_H)

    img  = Image.new("RGB", (PANEL_W, PANEL_H), (18, 18, 28))
    draw = ImageDraw.Draw(img)
    draw.line([(0, PANEL_H//2), (PANEL_W, PANEL_H//2)], fill=(40, 60, 40), width=1)

    # --- Non-arm skeleton: thin gray lines ---
    for ra, rb, color, highlight in SKELETON_EDGES:
        if highlight:
            continue   # drawn separately below
        if ra not in role_to_idx or rb not in role_to_idx:
            continue
        draw.line([pts[role_to_idx[ra]], pts[role_to_idx[rb]]], fill=_GRAY, width=1)

    # --- Left arm: thin gray boxes ---
    for ra, rb in [("torso","left_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist")]:
        if ra in role_to_idx and rb in role_to_idx:
            _seg_box(draw, pts[role_to_idx[ra]], pts[role_to_idx[rb]], (100,100,100), 6)

    # --- Right arm: bright orange boxes, thicker ---
    arm_segments = [
        ("torso",           "right_shoulder", (255, 210, 100), 10),
        ("right_shoulder",  "right_elbow",    (255, 170,  60), 13),
        ("right_elbow",     "right_wrist",    (255, 130,  30), 11),
    ]
    for ra, rb, color, w in arm_segments:
        if ra in role_to_idx and rb in role_to_idx:
            _seg_box(draw, pts[role_to_idx[ra]], pts[role_to_idx[rb]], color, w)

    # --- Joint dots: right arm bright, others small gray ---
    for role, idx in role_to_idx.items():
        px, py = pts[idx]
        if role in RIGHT_ARM_ROLES:
            r, col = 6, (255, 220, 120)
        else:
            r, col = 2, (120, 120, 120)
        draw.ellipse([px-r, py-r, px+r, py+r], fill=col)

    # --- Hand rectangle from right_wrist using orientation ---
    if hand_R is not None and "right_wrist" in role_to_idx:
        wrist_3d    = positions[role_to_idx["right_wrist"]]
        finger_dir  = hand_R[:, 0]   # X — finger length direction
        palm_across = hand_R[:, 1]   # Y — across palm (thumb to pinky)
        hand_len    = 0.16           # wrist to fingertip (m)
        hand_w      = 0.09           # palm width (m)

        corners_3d = [
            wrist_3d + palm_across * hand_w / 2,
            wrist_3d + finger_dir  * hand_len + palm_across * hand_w / 2,
            wrist_3d + finger_dir  * hand_len - palm_across * hand_w / 2,
            wrist_3d - palm_across * hand_w / 2,
        ]
        corners_2d = _project(corners_3d, center, right_v, up_v, scale, PANEL_W, PANEL_H)
        draw.polygon(corners_2d, fill=(255, 100, 20))
        draw.polygon(corners_2d, outline=(255, 220, 120))

        # Palm normal indicator (Z axis) as a short line from hand centre
        hand_centre_3d = wrist_3d + finger_dir * hand_len / 2
        palm_tip_3d    = hand_centre_3d + hand_R[:, 2] * 0.07
        hc_pt  = _project([hand_centre_3d], center, right_v, up_v, scale, PANEL_W, PANEL_H)[0]
        pn_pt  = _project([palm_tip_3d],    center, right_v, up_v, scale, PANEL_W, PANEL_H)[0]
        draw.line([hc_pt, pn_pt], fill=(80, 180, 255), width=2)
        px2, py2 = pn_pt
        draw.ellipse([px2-3, py2-3, px2+3, py2+3], fill=(80, 180, 255))

        try:
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except OSError:
            fnt = ImageFont.load_default()
        draw.text((px2+5, py2-6), "palm normal", font=fnt, fill=(80, 180, 255))

    img = np.array(img)
    return _overlay(img, "Human (mocap target)", f"frame {frame_idx}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    z            = np.load(NPZ, allow_pickle=True)
    qpos         = np.asarray(z["qpos"],                  dtype=np.float64)
    contact      = np.asarray(z["contact_labels"],        dtype=np.float32)
    tgt_pos      = np.asarray(z["target_positions"],      dtype=np.float64)
    tgt_ori      = np.asarray(z["target_orientations"],   dtype=np.float64)
    ori_names    = [str(r) for r in z["orientation_role_names"]]
    role_names   = [str(r) for r in z["role_names"]]
    rh_ori_idx   = ori_names.index("right_hand")

    best = _best_contact_frame(qpos, contact)
    print(f"Best-match frame: {best}")

    model    = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL_H, width=PANEL_W)

    # Camera: same viewpoint for both robot panels
    q_manual = _manual_target_qpos(model)
    data.qpos[:] = q_manual; mujoco.mj_forward(model, data)
    lookat = data.qpos[:3] + np.array([0.05, 0.0, 0.25])
    cam = _make_cam(lookat, azimuth=150.0, elevation=-10.0, distance=2.6)

    # Panel 1: manual target pose
    sub1 = (f"Shoulder Yaw {TARGET_DEG['RIGHT_SHOULDER_Z']:+.0f}°  "
            f"Elbow {TARGET_DEG['RIGHT_ELBOW_Y']:+.0f}°  "
            f"Wrist Yaw {TARGET_DEG['RIGHT_WRIST_Z']:+.0f}°  "
            f"Wrist Roll {TARGET_DEG['RIGHT_WRIST_X']:+.0f}°")
    p1 = _render_robot(model, data, renderer, q_manual, cam,
                       "Manual target pose", sub1)

    # Panel 2: IK-solved pose at best-match frame (re-centre xy for camera)
    q_ik = qpos[best].copy()
    q_ik[0] = 0.0; q_ik[1] = 0.0   # centre in scene
    data.qpos[:] = q_ik; mujoco.mj_forward(model, data)
    lookat_ik = data.qpos[:3] + np.array([0.05, 0.0, 0.25])
    cam_ik = _make_cam(lookat_ik, azimuth=150.0, elevation=-10.0, distance=2.6)

    solved_str = "  ".join(
        f"{j.replace('RIGHT_','')}: {np.rad2deg(q_ik[idx]):+.0f}°"
        for j, idx in RIGHT_ARM_QIDX.items()
    )
    p2 = _render_robot(model, data, renderer, q_ik, cam_ik,
                       f"IK solved  (frame {best})", solved_str)

    renderer.close()

    # Panel 3: human stick figure with hand orientation axes
    hand_R = tgt_ori[best, rh_ori_idx]   # 3×3 rotation matrix
    p3 = _draw_human_panel(tgt_pos[best], role_names, best, hand_R=hand_R)

    # Concatenate side by side
    combined = np.concatenate([p1, p2, p3], axis=1)

    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(OUT_IMG)
    print(f"Saved: {OUT_IMG}")

    # Also save as a short video (3 s) so it's easy to view
    with imageio.get_writer(str(OUT_VID), fps=30, codec="libx264", quality=8) as w:
        for _ in range(90):
            w.append_data(combined)
    print(f"Saved: {OUT_VID}")


if __name__ == "__main__":
    main()

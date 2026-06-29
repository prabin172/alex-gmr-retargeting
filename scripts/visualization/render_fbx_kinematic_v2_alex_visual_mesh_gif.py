#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image


CANONICAL_V2_EDGES: List[Tuple[str, str]] = [
    ("pelvis", "torso"),
    ("torso", "neck"),
    ("neck", "head"),
    ("head", "head_top"),

    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_palm"),
    ("left_palm", "left_hand_tip"),

    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_palm"),
    ("right_palm", "right_hand_tip"),

    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_toe"),
    ("left_heel", "left_toe"),
    ("left_foot", "left_heel"),
    ("left_foot", "left_toe"),

    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_toe"),
    ("right_heel", "right_toe"),
    ("right_foot", "right_heel"),
    ("right_foot", "right_toe"),
]


HUMAN_AXIS_ROLES = [
    "left_palm",
    "right_palm",
    "left_foot",
    "right_foot",
    "pelvis",
    "head",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("solution_npz", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--model",
        type=Path,
        default=Path("assets/alex/temp_alex_floating_base_visual_mesh.xml"),
    )
    p.add_argument(
        "--fallback-model",
        type=Path,
        default=Path("assets/alex/alex_floating_base_with_sites.xml"),
    )

    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=int, default=12)

    # Keep defaults within MuJoCo's usual offscreen framebuffer.
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)

    p.add_argument("--azim", type=float, default=110.0)
    p.add_argument("--elev", type=float, default=-18.0)
    p.add_argument("--distance", type=float, default=3.0)
    p.add_argument("--lookat", type=float, nargs=3, default=None)
    p.add_argument("--fixed-camera", action="store_true", default=False)
    p.add_argument("--z-floor-zero", action="store_true", default=False)
    p.add_argument("--force-ground-plane", action="store_true", default=False)
    p.add_argument("--ground-size", type=float, default=3.0)
    p.add_argument("--ground-grid-spacing", type=float, default=0.25)
    p.add_argument("--ground-line-radius", type=float, default=0.0025)
    p.add_argument("--ground-alpha", type=float, default=0.55)

    # Mesh/crude geom controls.
    p.add_argument("--mesh-only", action="store_true", default=False)
    p.add_argument("--show-crude-geoms", action="store_true", default=False)

    # Human side panel.
    p.add_argument("--side-by-side-human", action="store_true", default=False)
    p.add_argument("--human-panel-width", type=int, default=None)
    p.add_argument("--human-view-elev", type=float, default=18.0)
    p.add_argument("--human-view-azim", type=float, default=-65.0)
    p.add_argument("--draw-human-axes", action="store_true", default=False)
    p.add_argument("--human-axis-scale", type=float, default=0.08)

    return p.parse_args()


def keep_geom_visible(geom_type: int) -> bool:
    """Keep visual meshes and floor/ground plane. Hide primitive robot crude geoms."""
    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_MESH):
        return True
    if int(geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return True
    return False


def hide_non_mesh_robot_geoms(renderer: mujoco.Renderer) -> None:
    """Hide primitive/collision-looking robot geoms in the already-built render scene.

    This affects visualization only. It does not change the model, qpos, IK, or physics.
    """
    scene = renderer.scene
    for i in range(scene.ngeom):
        g = scene.geoms[i]
        if not keep_geom_visible(g.type):
            g.rgba[3] = 0.0


def add_visual_ground_plane(
    renderer: mujoco.Renderer,
    size: float = 3.0,
    spacing: float = 0.25,
    line_radius: float = 0.0025,
    alpha: float = 0.55,
) -> None:
    """Add a render-only z=0 ground grid.

    This intentionally uses thin capsule lines instead of a solid plane, so it
    does not block robot parts that pass below z=0. It does not change the
    MuJoCo model, IK, contact, physics, or qpos.
    """
    scene = renderer.scene

    coords = np.arange(-float(size), float(size) + 1e-9, float(spacing))
    rgba_major = np.array([0.25, 0.25, 0.25, float(alpha)], dtype=float)
    rgba_minor = np.array([0.55, 0.55, 0.55, float(alpha) * 0.7], dtype=float)

    def add_line(p0, p1, rgba, radius):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.asarray(p0, dtype=float),
            np.asarray(p1, dtype=float),
        )
        geom.rgba[:] = rgba
        scene.ngeom += 1

    for v in coords:
        is_major = abs(v) < 1e-9 or abs((v / 1.0) - round(v / 1.0)) < 1e-6
        rgba = rgba_major if is_major else rgba_minor
        radius = line_radius * (1.6 if is_major else 1.0)

        add_line([v, -size, 0.0], [v, size, 0.0], rgba, radius)
        add_line([-size, v, 0.0], [size, v, 0.0], rgba, radius)

def as_str_list(arr: np.ndarray) -> List[str]:
    return [str(x) for x in arr.tolist()]


def source_frame_for_output(solution: Dict[str, np.ndarray], output_i: int, qpos_len: int) -> int:
    source_positions = np.asarray(solution["source_positions"])
    if source_positions.shape[0] == qpos_len:
        return output_i
    if "source_frame_ids" in solution:
        ids = np.asarray(solution["source_frame_ids"]).astype(int)
        if output_i < len(ids):
            sid = int(ids[output_i])
            if 0 <= sid < source_positions.shape[0]:
                return sid
    return min(output_i, source_positions.shape[0] - 1)


def compute_human_bounds(
    source_positions: np.ndarray,
    source_roles: List[str],
    selected_source_frames: Iterable[int],
) -> Tuple[np.ndarray, float]:
    pts = []
    for f in selected_source_frames:
        if 0 <= f < source_positions.shape[0]:
            frame_pts = source_positions[f]
            valid = np.isfinite(frame_pts).all(axis=1)
            pts.append(frame_pts[valid])
    if not pts:
        return np.zeros(3), 1.0

    all_pts = np.concatenate(pts, axis=0)
    center = 0.5 * (np.nanmin(all_pts, axis=0) + np.nanmax(all_pts, axis=0))
    span = np.nanmax(np.nanmax(all_pts, axis=0) - np.nanmin(all_pts, axis=0))
    radius = max(float(span) * 0.62, 0.5)
    return center, radius


def draw_axis(ax, origin: np.ndarray, R: np.ndarray, scale: float) -> None:
    if not np.isfinite(origin).all() or not np.isfinite(R).all():
        return
    colors = ["r", "g", "b"]
    for j, c in enumerate(colors):
        v = R[:, j] * scale
        ax.plot(
            [origin[0], origin[0] + v[0]],
            [origin[1], origin[1] + v[1]],
            [origin[2], origin[2] + v[2]],
            color=c,
            linewidth=2,
        )


def render_human_panel(
    solution: Dict[str, np.ndarray],
    output_i: int,
    qpos_len: int,
    width: int,
    height: int,
    elev: float,
    azim: float,
    bounds_center: np.ndarray,
    bounds_radius: float,
    draw_axes: bool,
    axis_scale: float,
) -> np.ndarray:
    source_roles = as_str_list(np.asarray(solution["source_roles"]))
    role_to_i = {r: i for i, r in enumerate(source_roles)}

    source_positions = np.asarray(solution["source_positions"], dtype=float)
    source_frame = source_frame_for_output(solution, output_i, qpos_len)
    pts = source_positions[source_frame]

    source_orientations = None
    if "source_orientations" in solution:
        source_orientations = np.asarray(solution["source_orientations"], dtype=float)

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # Edges
    for a, b in CANONICAL_V2_EDGES:
        if a not in role_to_i or b not in role_to_i:
            continue
        pa = pts[role_to_i[a]]
        pb = pts[role_to_i[b]]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]],
                color="0.15",
                linewidth=3,
            )

    valid = np.isfinite(pts).all(axis=1)
    ax.scatter(pts[valid, 0], pts[valid, 1], pts[valid, 2], s=20)

    if draw_axes and source_orientations is not None:
        ori_frame = source_frame_for_output(solution, output_i, qpos_len)
        if 0 <= ori_frame < source_orientations.shape[0]:
            R_all = source_orientations[ori_frame]
            for r in HUMAN_AXIS_ROLES:
                if r in role_to_i:
                    idx = role_to_i[r]
                    if idx < R_all.shape[0]:
                        draw_axis(ax, pts[idx], R_all[idx], axis_scale)

    c = bounds_center
    rad = bounds_radius
    ax.set_xlim(c[0] - rad, c[0] + rad)
    ax.set_ylim(c[1] - rad, c[1] + rad)
    z_min = 0.0 if getattr(render_human_panel, "_z_floor_zero", False) else c[2] - rad
    ax.set_zlim(z_min, c[2] + rad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Human canonical v2 | source frame {source_frame}")
    ax.grid(True)

    fig.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGB"))
    return img


def write_video_or_gif(path: Path, images: List[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        imageio.mimsave(path, images, fps=fps, codec="libx264", quality=8)
    else:
        imageio.mimsave(path, images, fps=fps)


def main() -> None:
    args = parse_args()

    if not args.solution_npz.exists():
        raise FileNotFoundError(args.solution_npz)

    model_path = args.model if args.model.exists() else args.fallback_model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Neither visual model nor fallback model exists: {args.model}, {args.fallback_model}"
        )

    d_npz = np.load(args.solution_npz, allow_pickle=True)
    solution = {k: d_npz[k] for k in d_npz.files}
    qpos = np.asarray(solution["qpos"], dtype=float)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width {qpos.shape[1]} does not match model.nq {model.nq}")

    start = max(0, args.start)
    end = qpos.shape[0] if args.end is None else min(args.end, qpos.shape[0])
    frames = list(range(start, end, max(1, args.stride)))
    if not frames:
        raise RuntimeError("No frames selected.")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = args.azim
    cam.elevation = args.elev
    cam.distance = args.distance

    if args.lookat is not None:
        cam.lookat[:] = np.asarray(args.lookat, dtype=float)
    else:
        cam.lookat[:] = np.array([0.0, 0.0, 0.6], dtype=float)

    human_panel_width = args.human_panel_width or args.width
    render_human_panel._z_floor_zero = bool(args.z_floor_zero)

    human_center = np.zeros(3)
    human_radius = 1.0
    if args.side_by_side_human:
        source_positions = np.asarray(solution["source_positions"], dtype=float)
        source_roles = as_str_list(np.asarray(solution["source_roles"]))
        selected_source_frames = [
            source_frame_for_output(solution, f, len(qpos)) for f in frames
        ]
        human_center, human_radius = compute_human_bounds(
            source_positions,
            source_roles,
            selected_source_frames,
        )

    images: List[np.ndarray] = []

    for k, f in enumerate(frames):
        data.qpos[:] = qpos[f]
        mujoco.mj_forward(model, data)

        # Optionally follow floating base lightly. For comparison videos,
        # fixed camera is usually better because the frame does not drift.
        if not args.fixed_camera:
            base = np.asarray(data.qpos[:3], dtype=float)
            cam.lookat[:] = 0.85 * cam.lookat + 0.15 * np.array(
                [base[0], base[1], max(0.0, base[2]) + 0.4]
            )

        renderer.update_scene(data, camera=cam)

        if args.mesh_only and not args.show_crude_geoms:
            hide_non_mesh_robot_geoms(renderer)

        if args.force_ground_plane:
            add_visual_ground_plane(
                renderer,
                size=args.ground_size,
                spacing=args.ground_grid_spacing,
                line_radius=args.ground_line_radius,
                alpha=args.ground_alpha,
            )

        robot_img = renderer.render()

        if args.side_by_side_human:
            human_img = render_human_panel(
                solution=solution,
                output_i=f,
                qpos_len=len(qpos),
                width=human_panel_width,
                height=args.height,
                elev=args.human_view_elev,
                azim=args.human_view_azim,
                bounds_center=human_center,
                bounds_radius=human_radius,
                draw_axes=args.draw_human_axes,
                axis_scale=args.human_axis_scale,
            )

            # Match heights if Matplotlib gives slight pixel variation.
            if human_img.shape[0] != robot_img.shape[0]:
                human_img = np.asarray(
                    Image.fromarray(human_img).resize(
                        (human_img.shape[1], robot_img.shape[0])
                    )
                )

            frame_img = np.concatenate([robot_img, human_img], axis=1)
        else:
            frame_img = robot_img

        images.append(frame_img)

        if k % 25 == 0:
            print(f"rendered {k + 1}/{len(frames)} frames")

    write_video_or_gif(args.out, images, fps=args.fps)

    print("Model:", model_path)
    print("qpos:", qpos.shape)
    print("frames rendered:", len(frames))
    print("mesh_only:", bool(args.mesh_only and not args.show_crude_geoms))
    print("side_by_side_human:", args.side_by_side_human)
    print("Wrote:", args.out)


if __name__ == "__main__":
    main()

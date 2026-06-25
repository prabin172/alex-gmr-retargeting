#!/usr/bin/env python3
"""Visualize a source-only FBX kinematic canonical v2 skeleton.

This viewer intentionally draws only the canonical human/source skeleton.  It is
meant to answer: are ankle/toe/heel/palm/head roles and local axes sane before
we retarget anything to Alex?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_AXIS_ROLES = ("left_palm", "right_palm", "left_foot", "right_foot", "pelvis", "head")


def equalize_axes(ax, pts: np.ndarray, margin: float = 0.15) -> None:
    finite = pts[np.all(np.isfinite(pts), axis=-1)]
    if finite.size == 0:
        return
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 0.25) * (1.0 + margin)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def load_json_scalar(data: np.lib.npyio.NpzFile, key: str):
    if key not in data.files:
        return None
    return json.loads(str(data[key].item()))


def draw_frame_axes(ax, origin: np.ndarray, R: np.ndarray, scale: float) -> None:
    colors = ("red", "green", "blue")
    for i, color in enumerate(colors):
        v = R[:, i] * scale
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            v[0],
            v[1],
            v[2],
            color=color,
            linewidth=1.4,
            arrow_length_ratio=0.18,
        )


def draw_frame(
    positions: np.ndarray,
    orientations: np.ndarray | None,
    roles: Sequence[str],
    edges: np.ndarray,
    frame_idx: int,
    axis_roles: Sequence[str],
    draw_axes: bool,
    title: str,
    elev: float,
    azim: float,
    axis_scale: float,
    label_roles: bool,
    figsize: Tuple[float, float],
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    pts = positions[frame_idx]

    for a, b in edges:
        pa = pts[int(a)]
        pb = pts[int(b)]
        if not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
            continue
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="0.25", linewidth=2.2)

    finite = np.all(np.isfinite(pts), axis=-1)
    ax.scatter(pts[finite, 0], pts[finite, 1], pts[finite, 2], s=22, c="tab:blue", depthshade=True)

    if label_roles:
        for role, p in zip(roles, pts):
            if np.all(np.isfinite(p)):
                ax.text(p[0], p[1], p[2], role, fontsize=7)

    if draw_axes and orientations is not None:
        for role in axis_roles:
            if role not in roles:
                continue
            ri = roles.index(role)
            p = pts[ri]
            R = orientations[frame_idx, ri]
            if np.all(np.isfinite(p)) and np.all(np.isfinite(R)):
                draw_frame_axes(ax, p, R, axis_scale)

    equalize_axes(ax, pts)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def render_image(fig) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical_npz", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output .gif, .mp4, or static image path.")
    parser.add_argument("--frame", type=int, default=None, help="Static frame index. If omitted, render an animation.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None, help="Output animation FPS. Defaults to canonical fps / stride.")
    parser.add_argument("--draw-axes", action="store_true")
    parser.add_argument("--axis-roles", default=",".join(DEFAULT_AXIS_ROLES))
    parser.add_argument("--axis-scale", type=float, default=0.08)
    parser.add_argument("--label-roles", action="store_true")
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=-70.0)
    parser.add_argument("--fig-width", type=float, default=8.0)
    parser.add_argument("--fig-height", type=float, default=7.0)
    args = parser.parse_args()

    path = args.canonical_npz.expanduser().resolve()
    out = args.out.expanduser().resolve()
    data = np.load(path, allow_pickle=True)
    roles = [str(x) for x in data["roles"].tolist()]
    positions = np.asarray(data["positions"], dtype=float)
    orientations = np.asarray(data["orientations"], dtype=float) if "orientations" in data.files else None
    edges = np.asarray(data["edges"], dtype=int)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 30.0
    metadata = load_json_scalar(data, "metadata_json") or {}
    axis_roles = [x.strip() for x in args.axis_roles.split(",") if x.strip()]

    out.parent.mkdir(parents=True, exist_ok=True)
    nframes = positions.shape[0]
    end = nframes if args.end is None else min(args.end, nframes)

    if args.frame is not None:
        frame_idx = int(np.clip(args.frame, 0, nframes - 1))
        title = f"{metadata.get('format', 'kinematic canonical v2')} | frame {frame_idx}"
        fig = draw_frame(
            positions,
            orientations,
            roles,
            edges,
            frame_idx,
            axis_roles,
            args.draw_axes,
            title,
            args.elev,
            args.azim,
            args.axis_scale,
            args.label_roles,
            (args.fig_width, args.fig_height),
        )
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Wrote: {out}")
        return

    frame_ids = list(range(max(args.start, 0), end, max(args.stride, 1)))
    if not frame_ids:
        raise ValueError("No frames selected for rendering.")
    output_fps = float(args.fps if args.fps is not None else max(fps / max(args.stride, 1), 1.0))

    ext = out.suffix.lower()
    if ext == ".gif":
        import imageio.v2 as imageio

        images = []
        for k, frame_idx in enumerate(frame_ids):
            title = f"{metadata.get('format', 'kinematic canonical v2')} | frame {frame_idx}"
            fig = draw_frame(
                positions,
                orientations,
                roles,
                edges,
                frame_idx,
                axis_roles,
                args.draw_axes,
                title,
                args.elev,
                args.azim,
                args.axis_scale,
                args.label_roles,
                (args.fig_width, args.fig_height),
            )
            images.append(render_image(fig))
            plt.close(fig)
            if k % 25 == 0:
                print(f"rendered {k + 1}/{len(frame_ids)} frames")
        imageio.mimsave(out, images, fps=output_fps)
        print(f"Wrote: {out}")
    elif ext == ".mp4":
        import imageio.v2 as imageio

        with imageio.get_writer(out, fps=output_fps, codec="libx264", quality=8) as writer:
            for k, frame_idx in enumerate(frame_ids):
                title = f"{metadata.get('format', 'kinematic canonical v2')} | frame {frame_idx}"
                fig = draw_frame(
                    positions,
                    orientations,
                    roles,
                    edges,
                    frame_idx,
                    axis_roles,
                    args.draw_axes,
                    title,
                    args.elev,
                    args.azim,
                    args.axis_scale,
                    args.label_roles,
                    (args.fig_width, args.fig_height),
                )
                writer.append_data(render_image(fig))
                plt.close(fig)
                if k % 25 == 0:
                    print(f"rendered {k + 1}/{len(frame_ids)} frames")
        print(f"Wrote: {out}")
    else:
        frame_idx = frame_ids[0]
        title = f"{metadata.get('format', 'kinematic canonical v2')} | frame {frame_idx}"
        fig = draw_frame(
            positions,
            orientations,
            roles,
            edges,
            frame_idx,
            axis_roles,
            args.draw_axes,
            title,
            args.elev,
            args.azim,
            args.axis_scale,
            args.label_roles,
            (args.fig_width, args.fig_height),
        )
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Wrote static image for frame {frame_idx}: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render two pkls of the SAME clip side by side (e.g. gmr_heightfix vs our
latest variant), same camera both panels, white floor (g1_model_setup.py's
`white_floor=True` -- the base G1 XML's own `floor` geom and this loader's
injected mocap floor plane are coincident and z-fight under the default
checker+edge-mark material; recoloring both to flat white removes it).
Each panel keeps the penetration annotation from render_penetration_annotated.py
(mesh-accurate _geom_lowest_z, red border flash + depth/body label).

Usage:
    conda run -n gmr python scripts/g1/render_sidebyside.py \\
        --pkl-left outputs/gmr_baseline/sprint/pkl/walk3_subject1_gmrfix.pkl \\
        --label-left "GMR-full (heightfix)" \\
        --pkl-right outputs/gmr_baseline/sprint/pkl_s5/walk3_subject1_smrc_rl_localground.pkl \\
        --label-right "Ours (rl + localground)" \\
        --out s8_renders_t8/walk3_subject1__sidebyside.mp4
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache, _geom_lowest_z  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402

ROBOT_BASE = "pelvis"


def _font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


class Panel:
    def __init__(self, model, geom_ids, body_of_geom, width, height, cam):
        self.model = model
        self.data = mujoco.MjData(model)
        self.mesh_cache = _build_mesh_cache(model)
        self.geom_ids = geom_ids
        self.body_of_geom = body_of_geom
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = cam
        self.robot_base_id = model.body(ROBOT_BASE).id

    def render(self, qpos_t, font_big, font_small, pen_threshold_cm, label, width, height):
        self.data.qpos[:] = qpos_t
        mujoco.mj_forward(self.model, self.data)

        best_z, best_g = 1e9, None
        for g in self.geom_ids:
            zz = _geom_lowest_z(g, self.model, self.data, self.mesh_cache)
            if zz < best_z:
                best_z, best_g = zz, g
        pen_cm = max(0.0, -best_z) * 100.0
        body_name = self.body_of_geom[best_g] if pen_cm > 0 else None
        penetrating = pen_cm > pen_threshold_cm

        self.cam.lookat = self.data.xpos[self.robot_base_id]
        self.renderer.update_scene(self.data, camera=self.cam)
        img = Image.fromarray(self.renderer.render().copy())
        draw = ImageDraw.Draw(img, "RGBA")

        if penetrating:
            draw.rectangle([0, 0, width, height], outline=(255, 0, 0, 255), width=10)
        draw.text((14, 12), label, font=font_big, fill=(255, 255, 255, 255))
        pen_label = f"PENETRATION: {pen_cm:5.2f} cm" if penetrating else f"penetration: {pen_cm:5.2f} cm"
        pen_color = (255, 40, 40, 255) if penetrating else (40, 160, 40, 255)
        draw.text((14, 46), pen_label, font=font_small, fill=pen_color)
        if body_name:
            draw.text((14, 70), f"body: {body_name}", font=font_small, fill=pen_color)
        return np.asarray(img), pen_cm, penetrating


def make_panel(width, height, azimuth, elevation, distance):
    model, data, floor_gid, _ = load_g1_model_with_vetted_collision_and_floor(white_floor=True)
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
                and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]
    body_of_geom = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                    for g in geom_ids}
    cam = mujoco.MjvCamera()
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    return Panel(model, geom_ids, body_of_geom, width, height, cam)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl-left", type=Path, required=True)
    ap.add_argument("--pkl-right", type=Path, required=True)
    ap.add_argument("--label-left", default="left")
    ap.add_argument("--label-right", default="right")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=720, help="per-panel width")
    ap.add_argument("--height", type=int, default=720, help="per-panel height")
    ap.add_argument("--azimuth", type=float, default=110.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--distance", type=float, default=2.6)
    ap.add_argument("--pen-threshold-cm", type=float, default=0.5)
    args = ap.parse_args()

    qpos_l, fps_l = load_gmr_pkl(args.pkl_left)
    qpos_r, fps_r = load_gmr_pkl(args.pkl_right)
    fps = args.fps if args.fps is not None else fps_l

    start = args.start
    n = min(qpos_l.shape[0], qpos_r.shape[0])
    end = n if args.frames is None else min(n, start + args.frames)

    left = make_panel(args.width, args.height, args.azimuth, args.elevation, args.distance)
    right = make_panel(args.width, args.height, args.azimuth, args.elevation, args.distance)

    font_big = _font(24)
    font_small = _font(18)
    gap = 6
    out_w = args.width * 2 + gap
    out_h = args.height

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    n_pen_l = n_pen_r = 0
    nfr = 0
    with imageio.get_writer(args.out, fps=fps) as writer:
        for t in range(start, end):
            img_l, pen_l, flag_l = left.render(qpos_l[t], font_big, font_small,
                                                args.pen_threshold_cm, args.label_left,
                                                args.width, args.height)
            img_r, pen_r, flag_r = right.render(qpos_r[t], font_big, font_small,
                                                 args.pen_threshold_cm, args.label_right,
                                                 args.width, args.height)
            n_pen_l += int(flag_l)
            n_pen_r += int(flag_r)

            combo = np.full((out_h, out_w, 3), 30, dtype=np.uint8)
            combo[:, :args.width] = img_l
            combo[:, args.width + gap:] = img_r
            writer.append_data(combo)
            nfr += 1
            if (t - start) % 300 == 0:
                print(f"  frame {t}/{end}  {args.label_left}={pen_l:.2f}cm  {args.label_right}={pen_r:.2f}cm")

    print(f"Wrote {args.out} ({nfr} frames @ {fps}fps)  "
          f"{args.label_left} penetrating: {n_pen_l}/{nfr} ({100*n_pen_l/nfr:.1f}%)  "
          f"{args.label_right} penetrating: {n_pen_r}/{nfr} ({100*n_pen_r/nfr:.1f}%)")


if __name__ == "__main__":
    main()

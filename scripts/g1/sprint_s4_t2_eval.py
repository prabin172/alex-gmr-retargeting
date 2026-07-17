#!/usr/bin/env python3
"""S4-T2: eval the s4_dev probe npz's (knee_bias on/off, knee-bias-skip-held, floor-weight
sweep) against the S3 baseline (ours_raw, no knee_bias). Reuses sprint_s3_full_corpus.py's
whole_clip_metrics/held_metrics verbatim -- does not mutate that script, it's the frozen S3
reference. Read-only eval, writes one CSV.

Usage: conda run -n gmr python scripts/g1/sprint_s4_t2_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from contact_labels import debounce_flags  # noqa: E402
from g1_model_setup import load_g1_model_with_vetted_collision_and_floor  # noqa: E402
from post_process_ground_contactfirst import _build_mesh_cache  # noqa: E402
from solve_fbx_canonical_alex_contactfirst import load_canonical  # noqa: E402
from solve_lafan1_canonical_g1_contactfirst import ROLE_TO_G1_BODY, FOOT_POS_ROLE  # noqa: E402
from sprint_s3_full_corpus import whole_clip_metrics, held_metrics, CANON_DIR, OURS_DIR  # noqa: E402

S4_DEV_DIR = REPO_ROOT / "outputs/gmr_baseline/sprint/s4_dev"
OUT_CSV = REPO_ROOT / "outputs/gmr_baseline/sprint/s4_t2_dev_eval.csv"

CLIPS = ["walk1_subject1", "fallAndGetUp1_subject1", "fallAndGetUp2_subject2",
         "ground1_subject1", "walk3_subject1"]

VARIANT_SUFFIXES = [
    ("s3_raw", OURS_DIR, "_ours.npz"),
    ("s4_kb", S4_DEV_DIR, "_ours_kb.npz"),
    ("s4_kbsh", S4_DEV_DIR, "_ours_kbsh.npz"),
    ("s4_fw1", S4_DEV_DIR, "_ours_fw1.npz"),
    ("s4_floorw", S4_DEV_DIR, "_ours_floorw.npz"),
    ("s4_fw2", S4_DEV_DIR, "_ours_fw2.npz"),
    ("s4_fw3", S4_DEV_DIR, "_ours_fw3.npz"),
    ("s4_fw5", S4_DEV_DIR, "_ours_fw5.npz"),
]


def main():
    model, data, floor_gid, floor_mocap_id = load_g1_model_with_vetted_collision_and_floor()
    mesh_cache = _build_mesh_cache(model)
    role_bid = {role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
               for role, name in ROLE_TO_G1_BODY.items()}
    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) != 0
               and not (int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0)]

    rows = []
    for clip in CLIPS:
        grounded = CANON_DIR / f"{clip}_lafan1c_grounded.npz"
        if not grounded.exists():
            print(f"SKIP {clip}: no canonical")
            continue
        (roles, role_to_idx, src_positions, fps, ori_roles, ori_to_idx, ori_mats,
         contacts, eff_names) = load_canonical(grounded)
        T = src_positions.shape[0]
        contacts_solved = {eff: debounce_flags(contacts[eff], 2) for eff in eff_names}
        held = {}
        for eff, role in FOOT_POS_ROLE.items():
            src_pt = src_positions[:, role_to_idx[role]]
            v = np.zeros(T)
            v[1:] = np.linalg.norm(np.diff(src_pt, axis=0), axis=1) * fps
            v[0] = v[1] if T > 1 else 0.0
            held[eff] = contacts_solved[eff] & (v < 0.05)

        for vname, vdir, suffix in VARIANT_SUFFIXES:
            p = vdir / f"{clip}{suffix}"
            if not p.exists():
                continue
            qpos = np.load(p)["qpos"]
            wm = whole_clip_metrics(model, data, mesh_cache, geom_ids, floor_gid, qpos)
            hm = held_metrics(model, data, mesh_cache, role_bid, held, qpos)
            row = dict(clip=clip, variant=vname, T=qpos.shape[0], **wm)
            for eff in FOOT_POS_ROLE:
                row[f"held_{eff}_n"] = hm[eff]["n"]
                row[f"held_{eff}_median_cm"] = hm[eff]["median_cm"]
                row[f"held_{eff}_frac3_pct"] = hm[eff]["frac3_pct"]
            rows.append(row)
            print(f"{clip:28s} {vname:10s} floorPen={wm['floorPen_cm']:7.2f}cm "
                  f"pen%={wm['pen_pct']:5.1f} coll%={wm['coll_pct']:5.2f}")

    cols = list(rows[0].keys()) if rows else []
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

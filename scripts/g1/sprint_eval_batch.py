#!/usr/bin/env python3
"""S1-T3: eval + faithfulness, 77 clips x 3 variants (raw / gmrfix / polished).

One CSV: `evaluate()`'s reference-free kinematic metrics (imported UNCHANGED from
eval_ihmc_json.py, same code eval_motion.py uses) + self-collision via the W2-T6
vetted collision model (separate pass, same qpos, appended columns) + a
faithfulness guard (robot-body FK position vs GMR's own scaled-human target, per
the bvh_lafan1_to_g1.json ik_match_table2 correspondence -- see planLogGMR.md
S1-T3 for why table2 not table1: table1's position_cost is 0 for pelvis and low
elsewhere, table2 carries the real position-tracking weight (10-100), so it's the
correspondence GMR itself actually optimizes position against) + hipZ p5 (from
GMR's own load_bvh_file, T2's exact convention) for T4's locomotion/floor split.

Resumable: skips a (clip, variant) row already present in the output CSV.

Usage:
    conda run -n gmr python scripts/g1/sprint_eval_batch.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ihmc_json import MODEL_DEFAULT as ALEX_MODEL_DEFAULT, evaluate  # noqa: E402
from eval_motion import G1_MODEL_DEFAULT, build_eval_context  # noqa: E402
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from solve_global_trajectory_opt_contactfirst import _collision_stats  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

BVH_DIR = REPO_ROOT / "data" / "raw" / "lafan1"
PKL_DIR = REPO_ROOT / "outputs" / "gmr_baseline" / "sprint" / "pkl"
HT_DIR = REPO_ROOT / "outputs" / "gmr_baseline" / "sprint" / "human_targets"
OUT_CSV = REPO_ROOT / "outputs" / "gmr_baseline" / "sprint" / "s1t3_eval.csv"
VETTED_URDF = REPO_ROOT / "outputs" / "gmr_baseline" / "g1_collision" / "g1_collision_vetted.urdf"

VARIANTS = ["raw", "gmrfix", "polished"]
VARIANT_SUFFIX = {"raw": "", "gmrfix": "_gmrfix", "polished": "_polished"}

# Position-weighted correspondence, ik_match_table2 of bvh_lafan1_to_g1.json (14
# pairs, pos_offset==[0,0,0] and ground_height==0.0 for all -- confirmed by direct
# read, so target position == the saved human_targets npz value with no further
# transform needed).
ROBOT_TO_HUMAN_BONE = {
    "pelvis": "Hips",
    "left_hip_yaw_link": "LeftUpLeg", "right_hip_yaw_link": "RightUpLeg",
    "left_knee_link": "LeftLeg", "right_knee_link": "RightLeg",
    "left_ankle_roll_link": "LeftFootMod", "right_ankle_roll_link": "RightFootMod",
    "torso_link": "Spine2",
    "left_shoulder_yaw_link": "LeftArm", "right_shoulder_yaw_link": "RightArm",
    "left_elbow_link": "LeftForeArm", "right_elbow_link": "RightForeArm",
    "left_wrist_yaw_link": "LeftHand", "right_wrist_yaw_link": "RightHand",
}

CSV_FIELDS = [
    "clip", "variant", "T", "fps", "hipZ_p5",
    "floorPen_max_cm", "pen_pct", "floatMax_cm", "float_pct",
    "collPct_vetted", "collPeak_vetted_cm",
    "jl_viol", "worst_joint", "vMax", "vP95", "n_spikes", "rootV_max",
    "faith_mean_cm", "faith_max_cm", "faith_n_pairs",
]


def hip_z_p5(bvh_path: Path) -> float:
    frames, _ = load_bvh_file(str(bvh_path), format="lafan1")
    z = np.array([f["Hips"][0][2] for f in frames])
    return float(np.percentile(z, 5))


def faithfulness(qpos: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData,
                  ht_npz_path: Path) -> tuple[float, float, int]:
    """Mean/max position error (cm) of FK'd robot bodies vs GMR's own scaled
    human targets, over the table2-mapped pairs actually present in both."""
    if not ht_npz_path.exists():
        return float("nan"), float("nan"), 0
    ht = np.load(ht_npz_path, allow_pickle=True)
    T = qpos.shape[0]
    errs = []
    for robot_body, human_bone in ROBOT_TO_HUMAN_BONE.items():
        key = f"pos__{human_bone}"
        if key not in ht:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
        if bid < 0:
            continue
        target = ht[key]
        Tm = min(T, target.shape[0])
        for t in range(Tm):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            errs.append(float(np.linalg.norm(data.xpos[bid] - target[t])))
    if not errs:
        return float("nan"), float("nan"), 0
    arr = np.array(errs) * 100  # m -> cm
    return float(arr.mean()), float(arr.max()), len(ROBOT_TO_HUMAN_BONE)


def load_existing_rows(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["clip"], row["variant"]))
    return done


def main():
    clips = sorted(p.stem for p in BVH_DIR.glob("*.bvh"))
    print(f"{len(clips)} clips found")

    # Main (unvetted) eval context, same model/mesh cache eval_motion.py uses by
    # default -- floorPen/pen%/float/joint-limit/velocity metrics.
    model, data, mesh_cache, geom_ids, mj_joint_names, q_lo, q_hi = build_eval_context(
        G1_MODEL_DEFAULT)

    # Separate vetted-collision context (W2-T6/S2-T3) for real self-collision numbers --
    # NOT the default eval_motion.py model, whose collision pairs are unvetted noise.
    vmodel = mujoco.MjModel.from_xml_path(str(VETTED_URDF))
    vdata = mujoco.MjData(vmodel)

    done = load_existing_rows(OUT_CSV)
    write_header = not OUT_CSV.exists()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fail_log = OUT_CSV.with_suffix(".fail")
    if not done:
        fail_log.write_text("")

    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for ci, clip in enumerate(clips, 1):
            bvh_path = BVH_DIR / f"{clip}.bvh"
            p5 = None
            for variant in VARIANTS:
                if (clip, variant) in done:
                    continue
                pkl_path = PKL_DIR / f"{clip}{VARIANT_SUFFIX[variant]}.pkl"
                if not pkl_path.exists():
                    print(f"[{ci}/{len(clips)}] MISSING {pkl_path.name}, skip")
                    continue
                try:
                    if p5 is None:
                        p5 = hip_z_p5(bvh_path)
                    qpos, fps = load_gmr_pkl(pkl_path)
                    row = evaluate(clip, qpos, fps, {}, model, data, mesh_cache,
                                   geom_ids, {}, q_lo, q_hi, mj_joint_names)
                    cs = _collision_stats(vmodel, vdata, qpos, floor_gid=None,
                                          count_floor=False)
                    ht_path = HT_DIR / f"{clip}.npz"
                    faith_mean, faith_max, n_pairs = faithfulness(qpos, model, data, ht_path)
                    out_row = {
                        "clip": clip, "variant": variant, "T": row["T"], "fps": row["fps"],
                        "hipZ_p5": p5,
                        "floorPen_max_cm": row["floor_pen_max_cm"],
                        "pen_pct": row["floor_pen_pct"],
                        "floatMax_cm": row["float_max_cm"], "float_pct": row["float_pct"],
                        "collPct_vetted": cs["pct"], "collPeak_vetted_cm": cs["max_pen_cm"],
                        "jl_viol": row["jl_viol"], "worst_joint": row["worst_joint"],
                        "vMax": row["vel_max_rad_s"], "vP95": row["vel_p95_rad_s"],
                        "n_spikes": row["n_spikes"], "rootV_max": row["root_v_max"],
                        "faith_mean_cm": faith_mean, "faith_max_cm": faith_max,
                        "faith_n_pairs": n_pairs,
                    }
                    writer.writerow(out_row)
                    f.flush()
                    print(f"[{ci}/{len(clips)}] OK {clip} {variant}  "
                          f"floorPen={row['floor_pen_max_cm']:.1f}cm  "
                          f"faith_mean={faith_mean:.1f}cm")
                except Exception as e:  # noqa: BLE001
                    with open(fail_log, "a") as ff:
                        ff.write(f"{clip}:{variant}:{e}\n")
                    print(f"[{ci}/{len(clips)}] FAIL {clip} {variant}: {e}")

    print("DONE")


if __name__ == "__main__":
    main()

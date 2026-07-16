#!/usr/bin/env python3
"""Polish a GMR pkl: Stage A smoothing + grounding, model-agnostic (G1 or any GMR robot).

Round-trips through the exact same pkl dict shape GMR's own tooling expects
(`batch_gmr_pkl_to_csv.py` -> BeyondMimic, `vis_robot_motion.py`) so a polished pkl is a
drop-in replacement for a raw one -- no downstream format changes.

    load pkl (load_gmr_pkl: xyzw -> wxyz) -> qpos (T,36)
        -> [--stage-a: tridiagonal smoothing, T7]
        -> [--ground: Z-shift, T8 -- shells out to post_process_ground_contactfirst.py
            UNMODIFIED via a minimal temp NPZ (just `qpos`) -- constant/perframe modes
            don't touch contact_flags/effector_names at all, so no adapter logic needed
            beyond the NPZ boundary itself; "do not fork the QP/mesh code" from the plan]
        -> qpos (T,36) -> save pkl (wxyz -> xyzw, mirrors bvh_to_robot.py:166)

With no flags this is the identity transform -- the round-trip gate for T6.

Stage A (--stage-a) imports `stage_a` from `solve_global_trajectory_opt_contactfirst.py`
UNCHANGED -- it's pure qpos-level tridiagonal smoothing (`_banded_smoother`/`_smooth_channel`,
no model/robot references beyond the module's `N_ACT=29` constant, which numerically matches
G1's 29 actuated joints). `q_lo`/`q_hi` come from the target model's own joint ranges (via
`eval_motion.build_eval_context`) -- NOT Alex's, this was the one thing that had to be
robot-specific and it already was a function argument, not a hardcoded default.

Usage:
    conda run -n gmr python scripts/g1/polish_gmr_pkl.py \\
        --in outputs/gmr_baseline/pkl/walk1_subject1.pkl \\
        --out outputs/gmr_baseline/pkl/walk1_subject1_stageA.pkl \\
        --stage-a
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling load_gmr_pkl
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # NOT repo root -- see planLogGMR.md T1
from load_gmr_pkl import load_gmr_pkl  # noqa: E402
from solve_global_trajectory_opt_contactfirst import stage_a  # noqa: E402

GROUND_SCRIPT = REPO_ROOT / "scripts" / "post_process_ground_contactfirst.py"


def ground_qpos(qpos: np.ndarray, model_path: Path, mode: str, percentile: float,
                smooth_shift: float) -> np.ndarray:
    """Z-ground qpos by shelling out to post_process_ground_contactfirst.py,
    UNCHANGED, via a minimal temp NPZ. constant/perframe modes only ever read the
    `qpos` key (contact_flags/contact_effector_names/fps are optional, guarded by
    `if "key" in data_dict` in that script) -- verified by reading its source."""
    with tempfile.TemporaryDirectory() as td:
        in_npz = Path(td) / "in.npz"
        out_npz = Path(td) / "out.npz"
        np.savez_compressed(in_npz, qpos=qpos)
        cmd = [sys.executable, str(GROUND_SCRIPT),
               "--npz", str(in_npz), "--out", str(out_npz),
               "--model", str(model_path), "--mode", mode]
        if mode == "constant":
            cmd += ["--percentile", str(percentile)]
        elif mode == "perframe":
            cmd += ["--smooth-shift", str(smooth_shift)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return np.load(out_npz)["qpos"]


def gmr_heightfix(qpos: np.ndarray, model_path: Path) -> tuple[np.ndarray, float]:
    """Replicate GMR's own paper-described (but shipped-code-disabled) height fix --
    GMR/scripts/bvh_to_robot_dataset.py:127-138, HEIGHT_ADJUST=True/PERFRAME_ADJUST=False
    path (their clip-global, non-per-frame default). Their method FKs every frame with a
    torch KinematicsModel and takes `torch.min(body_pos[..., 2])` -- BODY-ORIGIN xpos,
    NOT mesh vertices -- then subtracts that single scalar from root z (+ground_offset=0.0)
    for the whole clip. Faithfully reproduced here with plain mujoco FK (equivalent to
    their torch FK, mesh-blindness included -- that mesh-blindness is exactly what W2-T1's
    floating metric is meant to expose). Does NOT touch our own grounding/Stage-A machinery.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    T = qpos.shape[0]
    global_min = np.inf
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        body_z = data.xpos[1:, 2]  # exclude body 0 (world)
        global_min = min(global_min, float(body_z.min()))
    out = qpos.copy()
    out[:, 2] = out[:, 2] - global_min  # ground_offset = 0.0, per their code
    return out, global_min


def save_gmr_pkl(path: Path, qpos: np.ndarray, fps: float):
    """Inverse of load_gmr_pkl: qpos (T,36) wxyz -> pkl dict, root_rot wxyz -> xyzw."""
    root_pos = qpos[:, 0:3]
    root_rot_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]  # wxyz -> xyzw
    dof_pos = qpos[:, 7:]
    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(path, "wb") as f:
        pickle.dump(motion_data, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--model", type=Path,
                    default=Path("/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml"),
                    help="Robot model, for joint limits (q_lo/q_hi) used by --stage-a clipping.")
    ap.add_argument("--stage-a", action="store_true", help="Apply Stage-A tridiagonal smoothing.")
    ap.add_argument("--lambda-track", type=float, default=1.0,
                    help="Alex CLI default (solve_global_trajectory_opt_contactfirst.py).")
    ap.add_argument("--lambda-smooth", type=float, default=20.0,
                    help="fps^2-scaled: pipeline's 320 @120Hz -> 20 @30Hz (LAFAN1's rate).")
    ap.add_argument("--ground", action="store_true",
                    help="Z-ground via post_process_ground_contactfirst.py (constant/perframe "
                         "only -- no contact flags available for hybrid/constant-contact).")
    ap.add_argument("--ground-mode", choices=["constant", "perframe"], default="constant")
    ap.add_argument("--ground-percentile", type=float, default=1.0)
    ap.add_argument("--ground-smooth-shift", type=float, default=0.0)
    ap.add_argument("--heightfix", action="store_true",
                    help="Replicate GMR's own paper-described height fix (W2-T1) -- a "
                         "fair-baseline addendum, NOT part of our polish chain. Mutually "
                         "exclusive with --stage-a/--ground in one invocation.")
    args = ap.parse_args()

    if args.heightfix and (args.stage_a or args.ground):
        ap.error("--heightfix replicates GMR's OWN fix as a separate baseline column -- "
                 "run it alone, not combined with --stage-a/--ground in one invocation.")

    qpos, fps = load_gmr_pkl(args.in_path)

    if args.stage_a:
        import mujoco

        from solve_global_trajectory_opt_contactfirst import N_ACT
        n_act = qpos.shape[1] - 7
        assert n_act == N_ACT, (
            f"stage_a hardcodes N_ACT={N_ACT} (Alex's actuated joint count) internally -- "
            f"this pkl has {n_act} actuated joints, would silently misindex. "
            f"Only safe when n_act == {N_ACT}.")
        model = mujoco.MjModel.from_xml_path(str(args.model))
        q_lo = np.full(n_act, -1e6)
        q_hi = np.full(n_act, 1e6)
        act = 0
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 0:
                continue
            if bool(model.jnt_limited[j]):
                q_lo[act] = model.jnt_range[j, 0]
                q_hi[act] = model.jnt_range[j, 1]
            act += 1
        qpos = stage_a(qpos, args.lambda_track, args.lambda_smooth, q_lo, q_hi,
                       smooth_root=True)

    if args.ground:
        qpos = ground_qpos(qpos, args.model, args.ground_mode, args.ground_percentile,
                           args.ground_smooth_shift)

    if args.heightfix:
        qpos, global_min = gmr_heightfix(qpos, args.model)
        print(f"heightfix: clip-global min body-origin z = {global_min:.4f} m, subtracted")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gmr_pkl(args.out_path, qpos, fps)
    tag = "+".join(t for t, on in
                   [("stageA", args.stage_a), ("ground", args.ground),
                    ("gmrfix", args.heightfix)] if on) or "identity"
    print(f"Wrote {args.out_path} ({tag}, {qpos.shape[0]} frames)")


if __name__ == "__main__":
    main()

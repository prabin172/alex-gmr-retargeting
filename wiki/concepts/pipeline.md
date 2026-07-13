# Pipeline

Human MoCap (FBX) → canonical skeleton → contact-first MuJoCo QP IK → smoothed, grounded `qpos (T,36)` for IHMC Alex. Kinematic only — downstream physics-RL supplies torques. Full math: `METHOD.md`.

> **phasic-v2 branch note**: this page describes the `main`-branch stage list. The `phasic-v2`
> branch adds Stage 2.5 (canonical grounding, before Stage 3) and two opt-in stages, 4.6
> (physics-plausibility) and 4.7 (limb-cleanup), between grounding (4.5) and render (5). See
> [[phasic-architecture]] for the full phase map and [[physics-plausibility]]/[[limb-cleanup]] for
> the new stages themselves.

## Stages

| # | Stage | Script | Notes |
|---|-------|--------|-------|
| 1 | FBX → canonical positions | `scripts/build_fbx_canonical_human.py` | Blender background mode, per-FBX by hand. Positions only; raw FBX rotations discarded |
| 2 | Positions → orientation frames | `scripts/build_canonical_orientation_frames_fresh.py` | Semantic frames + facing-yaw auto-snap. See [[orientation-frames]] |
| 3 | Contact-first IK | `scripts/solve_fbx_canonical_alex_contactfirst.py` | Per-frame damped Gauss–Newton. See [[contact-first-ik]] |
| 4 | GlobalOPT | `scripts/solve_global_trajectory_opt_contactfirst.py` | Stage A tridiag smoothing + Stage B contact QP. See [[globalopt]] |
| 4.5 | Z-grounding | `scripts/post_process_ground_contactfirst.py` | Mesh-exact lowest-point planting. See [[grounding]] |
| 5 | Render | `scripts/visualization/render_contactfirst.py` | MP4 + contact strip; `--no-human` for robot-only, `--fixed-cam` |
| 6 | IHMC JSON export | `scripts/export_alex_retarget_npz_to_ihmc_json.py` | grounded NPZ → IHMC replay JSON; native 120 Hz (NO `--fps`). See [[ihmc-export]] |

Stages 1–2 run per new FBX by hand; stages 3–6 are the batch **`./retargetingPipeline.sh`** (loops the CLIPS list; identical flags for every clip by design — the per-clip flag fields exist but stay empty).

Stage 3 skips clips whose NPZ already exists; stages 4/4.5/5/6 recompute everything (deterministic). Log any batch run to `outputs/logs/pipeline_native120_<ts>.log`.

## Stage 1–2 commands (per new FBX)
```bash
blender --background --python scripts/build_fbx_canonical_human.py -- \
  --fbx data/raw/inhouse/<action>/<clip>.fbx \
  --out outputs/canonical_human/fbx_fresh/<clip>.npz
python scripts/build_canonical_orientation_frames_fresh.py \
  --in-npz  outputs/canonical_human/fbx_fresh/<clip>.npz \
  --out-npz outputs/canonical_human/fbx_fresh/<clip>_with_orient.npz
```

## Batch env knobs (all optional; shipped defaults shown)
`LAMBDA_SMOOTH=320`, `N_OUTER=6`, `FOOT_WEIGHT=160`, `HAND_WEIGHT=32`, `PLANT_MIN_RUN=8`, `COPLANAR_FEET_MODE=mean` (Stage 3, [[contact-first-ik]]), `FLOOR_WEIGHT=200`/`FLOOR_MODE=estimate` (Stage 4 on-floor rows, [[globalopt]]), `GROUND_MODE=constant-contact`/`GROUND_SMOOTH=80` (Stage 4.5, [[grounding]]), `RENDER_MESH=visual|collision|<path>`, `RENDER_DIR`, `GO_DIR`, `GR_DIR`, `IHMC_DIR=outputs/ihmcJsons-native120hz`, `RENDER_EXTRA="--fixed-cam --no-human"`.

## Solve rate = native 120 Hz (STRIDE=1, 2026-07-05)
The whole solve (Stages 3/4/4.5) runs at `120/STRIDE`. **Switched STRIDE 4→1 ⇒ native 120 Hz.** Why: the downstream IHMC RL tracker consumes at **50 Hz** with ZOH (no interp) — see [[ihmc-export]]. The old 30 Hz solve was sub-Nyquist for that gate; self-upsampling 30→120 at export never restored the lost content. STRIDE=1 solves at capture rate, nothing self-upsamples (export stays native, their `json_to_npz --output_fps 50` does the only downsample). Render auto-runs real-time at `fps/stride`=120.

**Rate-dependent knobs rescaled for dt/4 (validated on standup_01 / shovel_fronthard_02 / kneelingFall_02):**
| knob | 30 Hz | 120 Hz | rule |
|---|---|---|---|
| `LAMBDA_SMOOTH` (Stage A+B) | 20 | **320** | first-diff (velocity) penalty ∝ fps² → ×16 |
| `GROUND_SMOOTH` (Stage 4.5) | 5 | **80** | same first-diff smoother → ×16 |
| `--contact-min-run / -ramp / -preroll` | 3/4/2 | **12/16/8** | measured in FRAMES → ×4 |
| `PLANT_MIN_RUN` (stillness debounce) | 2 | **8** | measured in FRAMES → ×4 |
| `N_OUTER` (Stage B SCA) | 3 | **6** | more chances to find a clean SCA iterate (NOT correctness — see [[globalopt]] parity fix) |
| `FOOT_WEIGHT / HAND_WEIGHT` (plant pins) | 40/8 | **160/32** | dt-invariant; ×4 to **rebalance** vs the ×16 smoothing |

Derivation: divide the continuous objective by dt ⇒ position terms (track w=1, contact pins, collision ρ=1000) are **dt-invariant**; only the derivative (smoothness) term carries 1/dt² ⇒ ×16. So collision/track/trust/posture_reg **unchanged**; smoothing (and frame-count debounce) rescale; the pin ×4 is a *relative* rebalance, not a correctness need. Speeds (m/s) + onset-delay (s) already auto-scale via `×fps`.

Validation result: Stage A reproduces 30 Hz smoothing exactly (spikes→0, track preserved); collision-free clips (shovel) clean at 120 Hz. Also fixed a latent OSQP-status bug + the SCA parity bug (see [[globalopt]]).

**Latest batch = coplanar re-run (2026-07-06)**, superseding the finalized native-120 numbers above. With `COPLANAR_FEET_MODE=mean` + `FLOOR_WEIGHT=200` + `GROUND_MODE=constant-contact`: ok=18 fail=0, spikes 0. Peak self-penetration **≤1.86 cm** (kneelingFall_02; up from the pre-coplanar ≤0.88 cm — the coplanar/on-floor rows extend get-up legs for ~1 cm more shallow grazing, the deliberate trade for planted feet); shovels+squat 0.00. **Feet planted:** the 7 standup/get-up clips finish both feet within **≈0.6 cm** of the floor, coplanar pair split ≤0.6 cm — the RDX floating-foot symptom is fixed. Caveat: the 13–16 cm Stage-B `floor_err` on natural/side/slideHandsBack is a max-over-planted-frames artefact (early high-plant frames), not final stance. Two **fall** clips penetrate the floor plane at the end (kneelingFall_02 L −11 cm, kneelingFall_03 R −15.8 cm): a late free foot below the single constant-contact shift — separate regime, wants per-frame/hybrid grounding. See METHOD §9.

## Data flow
Inputs: `outputs/canonical_human/fbx_fresh/*_with_orient.npz`, 120 fps, **stride 1 ⇒ 120 fps** solve/render.
Outputs per clip: `outputs/contactfirst/<clip>_contactfirst.npz` → `outputs/global_opt_contactfirst/<clip>_global_opt.npz` → `outputs/grounded_contactfirst/<clip>_grounded.npz` → `outputs/renders/contactfirst/...mp4` → `outputs/ihmcJsons-native120hz/<clip>.json`. See [[outputs-layout]]. IHMC export detail: [[ihmc-export]].

## Environment
Python ≥3.10 conda env `gmr`; `mujoco`, `numpy`, `scipy`, `osqp`, `imageio`; `MUJOCO_GL=egl`; Blender 4.x for stage 1. `pip install -e .`

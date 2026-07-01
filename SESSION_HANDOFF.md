# Session Handoff — Alex V2 contact-first + GlobalOPT

Branch: `feature/alex-v2-contact-first-ik`. Prabin commits himself.

## Already committed (baseline for this branch)
- `c59c93a` Contact-first IK on **Alex V2** (`assets/alex/alex_floating_base_with_sites_v2.xml`,
  convex hulls on arms/head/fist). Foot-flat + fist/palm support. Solver:
  `scripts/solve_fbx_canonical_alex_contactfirst.py`.
- `439c80b` Contact-aware **GlobalOPT** (`scripts/solve_global_trajectory_opt_contactfirst.py`)
  + full-clip batch (`run_globalopt_all.sh`).
- Design/decisions in `CONTACT_FIRST_SUMMARY.md`, `GLOBALOPT_CONTACTFIRST_PLAN.md`.

## Pipeline (contact-first)
per clip: **contact-first solve** (Stage 3) → **GlobalOPT Stage-A smoothing** → **render**.
Run all 12 clips: `run_globalopt_all.sh` (env knobs: `LAMBDA_SMOOTH`, `RENDER_EXTRA`,
`RENDER_DIR`). Inputs = `outputs/canonical_human/fbx_fresh/*_with_orient.npz`.
Source 120 fps, `--stride 4` ⇒ real-time render fps = 30.

## Uncommitted this session (working, ready to commit)
1. **Foot-yaw fix** (solver): during foot contact, flat pins pitch/roll but yaw was
   a free DOF → foot free-drifted 40–67°. Added yaw-align to the HUMAN foot heading
   (`--foot-yaw-weight 1.5`). Now shovels/clean-standups 0.3–1.9°.
2. **Contact make/break blend** (solver): raw contact snapped constraints on/off
   (pose jump 2.8× at transitions; foot yanked flat from ~47°). Added:
   debounce `--contact-min-run 3`, cosine cross-fade `--contact-ramp 4`, look-ahead
   `--contact-preroll 2`, and **continuous** cross-fade of the human ori/pos
   suppression (`ori_weight_scale`/`pos_weight_scale` in `solve_frame_position_ik`,
   not a binary skip). Flat-snap ~47°→26°.
3. **Root smoothing** (GlobalOPT Stage A): the 7-DOF root was passing through
   UNSMOOTHED → whole-body flick (per-frame root jumps ~3cm/10°). Now Stage A
   smooths root pos (tridiagonal) + quaternion (hemisphere-aligned + renormalize).
   Default on; `--no-root-smooth`, `--root-smooth W`. Cut root-rot pops 2–4×;
   also collapsed shovel plant-slip 13→0.9cm. Root smoothing was the big visual win.
4. **Renderer** (`render_contactfirst.py`): `--fixed-cam` (static WORLD — constant
   lookat/az/el/dist from clip-global bbox, only Alex moves), `--ground`
   (semi-transparent plane at clip's lowest point), `--cam-azimuth/elevation/distance`.
5. **Analysis tools**: `scripts/analyze_foot_slip.py` (foot yaw drift),
   `analyze_contact_slip_jumps.py` (slip + inter-frame jumps),
   `analyze_contact_flicker.py` (contact transitions + near-threshold dwell).
6. **`--log-every`** throttle on both solvers.

## Key decisions (don't re-litigate)
- **Stage B (contact-pin QP) OFF by default** (`--n-outer 0`). Contact intervals are
  NOT stationary plants (feet/hands reposition ~30cm while labelled in-contact), so
  pinning is ill-posed / infeasible. Stage A smoothing is the shipped win. Reviving
  Stage B needs upstream detection that isolates true stationary plants.
- **flat gate stays at 40°** (`--foot-flat-tilt`). Tried 25: no snap benefit (the
  snap is the robot foot genuinely tilted at plant, not the gate) and it drops ~56%
  of foot-contact frames on some clips.
- **λ_smooth=30** now (was 10). λ=20 also fine; λ=30 safe once root smoothing is on.

## Current outputs
12 videos in `outputs/renders/contactfirst/blend/` (λ=30, fixed-cam, ground,
root-smoothed, blend). `outputs/` is git-ignored (videos/NPZs local, shared via Slack).

## State of the flick
- Joint jumps: spikes 17→**0**, jmax 1.9→0.25 (Stage A).
- Root pops: cut 2–4× by root smoothing.
- Residual: get-up clips still show ~26–46° foot flat-snap where the human foot is
  genuinely tilted until touchdown (partly faithful). Any remaining pop is a
  per-frame IK config flip (null-space / weak intermediate tracking).

## NEXT (priority)
1. If flick persists after root smoothing → solver-side: posture reg on knees/wrists
   + down-weight intermediate-segment position (kills null-space flips). Position-
   side twin of the orientation relaxation already done.
2. Contact detection that isolates true stationary plants → unlocks Stage B.
3. Commit the above (Prabin).

# Session Handoff — Alex V2 contact-first + GlobalOPT

Branch: `feature/alex-v2-contact-first-ik`. Prabin commits himself.
Last session: 2026-07-01. Everything below "Uncommitted" is working & verified, ready to commit.

## Committed baseline
- `c59c93a` Contact-first IK on **Alex V2** (`assets/alex/alex_floating_base_with_sites_v2.xml`,
  convex hulls). Solver: `scripts/solve_fbx_canonical_alex_contactfirst.py`.
- `439c80b` Contact-aware **GlobalOPT** (`scripts/solve_global_trajectory_opt_contactfirst.py`)
  + batch (`run_globalopt_all.sh`).
- Design docs: `CONTACT_FIRST_SUMMARY.md`, `GLOBALOPT_CONTACTFIRST_PLAN.md`.

## Pipeline + UNIFIED config (one retargeter for all actions — Prabin's rule)
Per clip: contact-first solve → GlobalOPT **Stage A** (contact-blind smoothing,
tridiag joints + root pos/quat) → **Stage B** (contact-aware QP: feet pinned hard
to per-interval median anchor, hands soft, SCA, trust region) → Z-grounding → render.
Batch: `run_globalopt_all.sh`, env knobs `LAMBDA_SMOOTH=20`, `N_OUTER=3`,
`RENDER_EXTRA`, `RENDER_DIR`. **Identical flags for every clip** — the CLIPS 3rd/4th
per-clip flag fields exist but are empty by design (experiments only).
Inputs `outputs/canonical_human/fbx_fresh/*_with_orient.npz`; 120fps, stride 4 ⇒ 30fps render.

## Uncommitted (this session's solver/renderer work)
Solver `solve_fbx_canonical_alex_contactfirst.py`:
1. **Foot-yaw align** during contact (`--foot-yaw-weight 1.5`): flat pins pitch/roll,
   yaw was free → 40–67° drift; now 0.3–1.9° on clean clips.
2. **Contact make/break blend**: debounce `--contact-min-run 3`, cosine `--contact-ramp 4`,
   look-ahead `--contact-preroll 2`, continuous cross-fade of human ori/pos suppression.
3. **Foot-hold** (`--foot-hold`, on): freeze ankle target at touchdown, cross-fade onto it,
   **weight ×10** (`--foot-hold-weight`; 3 let shovel body motion drag the plant 38–72cm).
4. **Well-conditioned foot-flat error**: `θ·unit_axis` (cost θ²) replaces bare cross
   product (cost sin²θ had a spurious stable minimum at 180° → foot flips).
5. **Shank-tilt clamp** (`--shank-clamp`, on): TARGET-side feasibility — knee target
   projected into the flat-foot-reachable tilt cone from the model's ankle ranges
   (ANKLE_X ±25°, fwd-lean ∈ [−30°,60°]; verified: fwd-lean=−ANKLE_Y, left-lean=+ANKLE_X),
   5° margin, faded by contact weight. Kills the unwinnable flat-vs-tracking fight.
6. **One-sided knee-flexion bias** (`--knee-bias-weight 0.5`, min-flex 12°): KNEE_Y=0
   is joint lower limit AND leg Jacobian singularity; silent once bent → no over-constraint.
7. **Contact-onset hysteresis** (`--contact-on-height-frac 0.7 --contact-on-speed-frac 0.5
   --contact-onset-max-delay 0.15`): onset waits for stricter thresholds, **capped 150ms**
   (uncapped deleted whole crouch plants: slideHandsBack L foot 92→59% coverage; capped 87.7%).
   Release unchanged. Fixes "robot plants before the human does" (shovel setup).
8. **`human_target_positions`** saved (pure morphology-scaled human, pre contact edits)
   alongside edited `target_positions`.
9. `--hierarchical` (OFF, experimental): foot tasks hard, body+hands nullspace. Retired —
   see key decisions. Hands-demoted-to-soft code kept.
10. `--log-every` throttle.

GlobalOPT: **Stage-A root smoothing** (pos tridiag + quat hemisphere-aligned; default on,
`--no-root-smooth`) — was the big visual win (root pops cut 2–4×, spikes → 0).
Renderer `render_contactfirst.py`: `--fixed-cam` (static world), `--ground`, cam knobs;
right panel = human skeleton (color) + solver targets overlaid **green** where they diverge
(old NPZs without the field render as before).
Analysis: `analyze_foot_slip.py`, `analyze_contact_slip_jumps.py`, `analyze_contact_flicker.py`.

## Key decisions (don't re-litigate)
- **ONE config for all actions** (Prabin). Solver defaults + GlobalOPT λ=20, n-outer 3.
- **Stage B ON everywhere** (supersedes the old all-off rule). Old blocker "plants not
  stationary / feet reposition 30cm in contact" was an ARTIFACT of loose contact
  detection; with hysteresis + hold10 the solve is 0.1–0.3cm/plant → Stage B well-posed.
  Stage A alone re-adds ~8cm plant drift (contact-blind; its `--foot-weight` is
  Stage-B-only). Stage B pins it: shovels 1.5cm, get-ups ≤2.8cm, 0 spikes.
- **`--hierarchical` retired**: regresses on pivoting get-up contacts (natural_01 tracking
  +13%, jumps +35%) even with hands soft; hold10+StageB beats it on shovels (1.5 vs 4.7cm).
  Root failure was promoting the reach-limited (by-design best-effort) palm pin to hard.
  Don't revive without a per-interval stationarity gate.
- **Foot-drag diagnosis**: shovel "feet moving in contact" was NOT target slip (held anchor
  path 4–9cm) — the ACHIEVED foot was dragged 28–72cm by heavier pelvis/torso tasks.
  Weight-based fixes cap out; Stage B is the principled pin.
- **flat gate stays 40°** (`--foot-flat-tilt`): 25° drops ~56% of contact frames, no benefit.
- **λ_smooth=20 + Stage B** (was 30 Stage-A-only). Also damps the shovel thrust less.

## Measured (final unified batch, 12/12 OK)
- Shovels: plant_slip **1.0–1.5cm**, flat 0.1–0.2°, coll 0, spikes 0.
- Standups: plant_slip 2.7–3.7cm (side_05 outlier 7.9), spikes 0; crouch-phase flat
  angles 9.7–12.7° are faithful human tilt.
- Since blend era: foot-flat err in contact 12.7°→7.7° mean (shovels ~6°→0.1°), knee
  straight-lock dwell 26.5%→**0%**, standing knees ~10° bent, jump spikes 0.

## Outputs (git-ignored, shared via Slack)
Latest videos: `outputs/renders/contactfirst/unified/` (fixed-cam, ground, overlay).
Earlier: `.../foothold_fix/`, `.../onset_hysteresis/`, `.../shankclamp_kneebias/`, `.../blend/`.
Stage NPZs by era: current dirs = unified; preserved `outputs/*_foothold_fix/`,
`*_onset_hyst/`, `*_shankclamp/`, `*_pre_shankclamp/` (blend).

## Watch items (from unified batch review)
- `standup_side_04`: Stage B self-penetration peak 3.9→6.6cm — if visible, tune Stage-B
  trust/collision weights (clip-agnostic), not per-clip flags.
- `standup_side_05`: plant_slip 7.9cm outlier.
- Residual get-up flat-snap ~26–46° at touchdown is partly faithful (human foot tilted).

## NEXT
1. Prabin reviews `unified/` videos (esp. side_04/05) → commit everything.
2. If flick persists: down-weight intermediate-segment position (wrists still open;
   knees done via flexion bias).
3. Then: Mimic-ready export / contact labels pipeline (Stage 4 in CLAUDE.md).

## Design stance (retargeting philosophy)
Physical feasibility > verbatim copying. Contacts + end effectors exact; body interior
approximate. Kinematic infeasibility is fixed in the TARGETS (e.g. shank clamp), not by
weight fights. Downstream physics-RL absorbs dynamics errors, not kinematic impossibilities.

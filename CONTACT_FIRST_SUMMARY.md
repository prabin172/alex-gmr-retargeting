# Contact-First IK — short summary (vs. worlddelta baseline)

Branch `feature/alex-v2-contact-first-ik`. Solver:
`scripts/solve_fbx_canonical_alex_contactfirst.py` (format `alex_contactfirst_v1`).

## What we optimize
Per-frame robot config `q` (36-DOF: 7 free root + 29 joints) that matches the
human targets. Solved frame-by-frame, warm-started from previous frame.

## Loss function (per frame)
Damped Gauss-Newton, stacked weighted least squares → normal equations → damped
solve → `mj_integratePos`, iterated (`--ik-iters`), with per-iter step cap.
Stacked residual rows:
- **Position**: `w·(p_body(q) − p_target)` per role. Target =
  `rest_pos + root_scale·root_delta + role_scale·(rel − rel0)` (morphology delta).
- **Orientation**: `w·log(R_body(q)·R_target⁻¹)`, world-delta transfer
  `R_target = (R_src·R_src_rest⁻¹)·R_alex_rest`.
- **Posture reg**: `λ·(q − q_rest)` (keeps redundant DOFs sane).
- **Self-collision**: soft repulsion rows, weight 20, `_within_k_hops` excluded.
- **[contact-first, new]** contact rows (below), active only on contact frames.

## How contact-first differs from previous
Previous (worlddelta): track **every** role's position + world-delta orientation,
uniform priority. Now: **end-effector contact is #1 priority**; intermediate
segments demoted.
- During contact, per effector: **suppress** the wrist/ankle-body position &
  world-delta orientation targets, **replace** with contact constraints.
- Intermediate segments (upperarm/forearm/shin) left **orientation-free**.
- Knees/wrists left loose within joint limits (rely on posture reg).
- Orientation kept mainly for **distal segments + trunk/torso**.
- Model swapped to **Alex V2** (`..._with_sites_v2.xml`), convex-hull collision
  on arms/head/fist.

## How contact surfaces chosen
- **Feet**: sole (foot local +Z). Contact = human marker height-over-floor +
  low speed, **gated on human foot actually flat** (canonical sole normal within
  40° of vertical) → avoids forcing flat when human isn't.
- **Hands (closed fist, no flat palm)**: support face = gripper **+X**
  (palm/finger-front), NOT knuckles. Reuses `alex_{l,r}_palm_contact_site`.

## Contact constraints / losses (active on contact frames)
- **Foot-flat**: foot local +Z → world +Z, weight 3.0. Axis-align via
  `err_rot = cross(a_world, world_dir)` as an orientation row (locks tilt, frees spin).
- **Fist-down**: gripper +X → world −Z, soft weight 0.8 (best-effort).
- **Palm position pin**: pin `palm_contact_site` to human hand location (`mj_jacSite`),
  weight 3.0; **suppresses the wrist-body position target** so they don't fight.

## Why we get flicks / slips
- **No cross-frame temporal term** — only warm-start + posture reg. Free DOFs pick
  slightly different configs frame-to-frame → **flick**.
- Relaxed intermediate orientation + loose knees/wrists → more null-space to wander.
- Contact is **soft** (penalty, not hard constraint) + reach-limited on dynamic
  push (Alex arm shorter than human) → contact point drifts → **slip**.

## How we might solve
1. **Global-OPT smoothing** (main lever): trajectory-level min velocity/accel/jerk
   with **contacts pinned as hard constraints**. Kills flick + stops slip.
2. **Per-frame forward temporal term**: `‖qₜ − qₜ₋₁‖` velocity penalty (causal, cheap).
3. **Position-side reweighting**: down-weight intermediate-segment position, add
   posture reg on knees/wrists (source of flick).
4. Do NOT over-stiffen per-frame IK to hide jitter — keep contacts free, smooth after.

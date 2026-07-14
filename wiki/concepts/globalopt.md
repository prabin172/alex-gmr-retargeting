# GlobalOPT (Stage 4)

`scripts/solve_global_trajectory_opt_contactfirst.py`. Per-frame IK leaves velocity spikes (branch flips = 1–2 rad single-frame jumps) and root pops that per-frame methods can't fix by construction (frame t−3 can't know a topology change comes at t). Offline = whole trajectory available ⇒ optimize all T frames jointly. Math: METHOD.md §6.

## Stage A — closed-form tridiagonal smoothing
Per-channel Tikhonov: `min λ_track·Σ(y−x)² + λ_smooth·Σ(y_t−y_{t−1})²` — tridiagonal normal equations, `scipy.linalg.solve_banded`, O(T)/channel. All 29 joints + **the floating base** (root pos via same solve; root quat via hemisphere-align → per-component smooth → renormalize; `--no-root-smooth` to disable). Unified config: `λ_smooth=20`, `λ_track=1` (script default is 10 — the batch overrides). Stage A can only redistribute a spike, not change a mean.

## Stage B — sparse contact-aware QP over all frames
Single QP over actuated increments `δQ ∈ ℝ^{T·29}` (root left from Stage A), OSQP (`max_iter=20000`, accepts `"solved"` AND `"solved inaccurate"`), re-linearized `n_outer=6` times at 120 Hz (SCA; script default is `--n-outer 0` = Stage A only — pipeline passes 6).

- **Objective**: block-tridiag smoothness Hessian + per-frame tracking (contacting effector's own role down-weighted ×0.1) + contact terms.
- **Contact anchor = per-interval median**: contact intervals split into stationary sub-segments (contact-point speed < 0.05 m/s), each anchored to its median position at foot weight **160**, hand weight **32** (soft! ×4 the 40/8 CLI defaults — pipeline passes `--foot-weight 160 --hand-weight 32`); non-stationary contact frames follow per-frame IK at ×0.15. Foot-flat (w 3.0) + fist-down (w 0.8) rows on planted frames.
- **On-floor / coplanar rows** (2026-07-06, `--floor-weight`, pipeline `FLOOR_WEIGHT=200`): on planted frames, drive each foot's **4 sole-corner** site Zs to a shared `floor_z` (row `J_z·δq = floor_z − corner_z`). One row type gives on-floor + flat + inter-foot **coplanarity** at once. `floor_z` = median of the two feet's warm-start ground heights (`--floor-mode estimate`) so both share the correction — Stage B holds the root fixed, so leg-only reach saturates ~3 cm if one foot must travel the full gap. The position pin drops to **X,Y only** on these frames so it doesn't fight the height row; the plant-slip metric is likewise **horizontal-only for feet** (vertical foot motion is the deliberate correction, not slip). This is only a *cleanup* — the real coplanarity fix is upstream (Stage-3 coplanar targets, [[contact-first-ik]]); with those, these rows close the residual to ~0.5 cm. Cost: peak self-penetration 0.5→~1.4 cm.
- **Stillness debounce `plant_min_run=8` frames** (×4 knob, ≈2 @30Hz): a sub-segment must be still for ≥8 frames to count as a plant; shorter dips → moving (low weight, follow IK). Kills phantom 1-frame plants from velocity zero-crossings on a lifting-off hand. standup_side_05 right_hand slip 14.7→6.8 cm (25 single-frame blips removed). See [[metrics]].
- **Hard constraints ONLY**: joint-limit box, trust region ±0.15 rad, self-collision inequalities (λ_coll=5).
- **Soft-slack self-collision (always-on)**: one slack s≥0 per collision row + quadratic penalty ρ=1000. Exists because fullmesh legs made the hard inequalities primal-infeasible (424 rows vs ~80–194) → hard QP silently no-op'd (|δQ|max=0). Soft version always feasible, degrades gracefully; the old hard path + `--soft-collision` toggle are gone. See [[fullmesh-vs-primitive]].

## Keep-best-iterate + slip-aware selection (2026-07-05, the SCA convergence fix)
The SCA loop is **not** monotone in penetration: collision rows are linearized only at each outer's *start*, so an outer that begins collision-free carries 0 collision rows and takes an unconstrained tracking+smoothing step straight back into ~6 cm penetration → per-outer penetration **oscillates** (clean→bad→clean…) on get-ups. The original loop returned the **last** iterate unconditionally ⇒ result depended on `n_outer` **parity** (odd=lucky-resolving, even=bad-victory-lap). The "30 Hz fine, 120 Hz regressed" story was pure parity: 30 Hz `n_outer=3` (odd), 120 Hz `n_outer=6` (even) — NOT a rate effect.
- **Fix**: track the best iterate across outers, return it (`best_qpos`), seeded with the Stage-A warm start so Stage B never ships worse than its input.
- **Score is slip-aware, lexicographic**: `(max(0, pen_max − PEN_TOL), slip_max + foot_floor_err, pen+slip, coll%)`. First term = a hard-fail gate: penetration beyond `PEN_TOL` (1 cm; **2 cm when floor rows are on** — pressing feet onto the floor costs ~1.5 cm of extra self-penetration) is never traded for contact quality. Below the gate, minimize total contact error = horizontal slip + vertical foot-off-floor. `foot_floor_err` (added 2026-07-06) is essential once floor rows exist: without it every floor-improving iterate scores *worse* (nudges pen up, no credit for the foot reaching the floor) and keep-best ships the feet-apart warm start. Pure-penetration argmin would silently ship a clean-but-slid iterate — the pins-×4 change makes that trade real, so slip entered the score.
- **Result** (standup_side_04): old last-iterate shipped 6.59 cm pen / 42.9% coll; keep-best ships 0.49 cm / 4.5% / slip 6.3 cm. Whole batch: peak pen ≤0.88 cm on all 18.

> **Contacts are soft, not equalities**: every contact term is `add_soft` (weight 160 foot / 32 hand), never a hard equality. Residual slip is a high-weight equilibrium, not zero by construction. (The old "hard equality" docstring is already corrected.)

## Hard mesh floor collision (2026-07-09/10, `--floor-collision`, opt-in per-clip)
Separate from the soft on-floor rows above (which only ever touch **planted feet**). A floor
plane geom is injected in-memory (`_load_model_with_floor`, `mujoco.MjSpec` mocap body — never
touches the asset XML), and `_build_collision` treats floor-vs-robot contacts exactly like
self-collision pairs (same soft-slack QP rows, `count_floor=True`), catching **any** fullmesh
geometry — swing feet, hands, a tilted toe — not just planted feet. `floor_gid` must always be
recognized (its body id is never 0) even when `count_floor=False`, else floor contacts silently
leak into self-collision counting. First shipped on `luigi_standProne_03` only (paired with a
Stage-3 floor-repulsion term + 2-pass arm refinement for onset transitions, see
[[contact-first-ik]]); root-cause history in `collision.md`/`collisionFixPlan.md` (repo root).

**Phase-aware gating (2026-07-10, `--floor-phase-aware`)**: a single clip-wide `floor_z` is
calibrated to the standing/planted-foot stance and misreads a lying/supine phase's legitimately-
low pelvis as violation (see [[grounding]]'s "Get-up floor residual is BETWEEN-PHASE"). Same
between-phase root-Z non-invariance, just hitting the Stage 3/4 collision term instead of Stage
4.5's registration percentile. `floor_phase_weight()` (duplicated in both solver scripts):
smoothstep of pelvis/root height between the clip's low reference and its planted-foot/standing
height, thresholded at 0.5 to gate `count_floor`/collision rows on/off per frame instead of
clip-wide. Enabled it on `luigi_standSupine_08`: fixed the false pelvis violation AND caught a
real bug the untouched baseline had — `RIGHT_FOOT` genuinely clipped 4.4cm through the floor
during the stand-up transition, invisible to the planted-foot-only eval check. Small cost at the
phase boundary (~5 frames, self-pen to 2.1cm, no spikes) plus a real slip/flat-error increase
(1.6→4.3cm, 0→5.2deg) — accepted trade. No-op for single-phase clips. See `wiki/log.md`
2026-07-10 and `SESSION_HANDOFF.md` for the full validation.

## FOOT_WEIGHT ceiling — residual slip is a smoothness floor, not a weight deficit (2026-07-08)
The 6.3 cm standup_side_04 slip was **stale** — it predates the Stage-3 coplanar targets + on-floor rows. On the CURRENT config the same clip's plant-slip is **1.9 cm**. Ran `scripts/diagnose_foot_slip.py` (per-frame foot XY deviation from the frozen anchor vs max inter-limb penetration, ±6-frame windowed correlation per the mimic-repo review):
- slip↔penetration corr **−0.08** (exact and windowed) — worst-slip frames are collision-**free** (0/9 frames ≥1.5 cm slip sit within ±6 fr of any pen ≥0.5 cm). NOT collision-bound; the earlier leg-crossing prior was wrong *for the slip* (the 1.61 cm peak pen lands in different frames).
- FOOT_WEIGHT sweep (all else fixed, all "solved"): **160**→1.9 cm slip / 1.61 cm pen / 8.8% coll; **1000**→1.7 / 2.16 / 9.7%; **4000**→1.7 / 2.46 / 12.8%. Slip floors at 1.7 cm; pen + coll% climb monotonically. 25× the weight buys 0.2 cm slip for +0.85 cm penetration.
- **Why it floors**: worst-slip frames are contiguous ramps at plant **edges** (e.g. fr 495→505 ramping 1.4→1.9 cm), not mid-plant sliding — the `λ_smooth` transition-blend easing the foot in/out while the frozen-median anchor sits a couple cm away by construction. Can't pin without fighting smoothness.
- **Decision**: FOOT_WEIGHT stays **160**. No Stage 4.6, no bump. (`standupTuned` branch = leg-only footlock post-step; "didn't help" for the same reason — built to kill a slip that no longer exists, collision-blind on top.) Open: is 1.7 cm below the mimic tracker's reward std? If so, done. See [[metrics]].

## Rate dependence + OSQP status bug (2026-07-05, native 120 Hz switch)
- **Latent bug fixed**: OSQP ≥1.x reports the inaccurate status as `"solved inaccurate"` (space); the accept-check only listed `"solved_inaccurate"` (underscore) → any inaccurate solve was silently discarded (`|dQ|=0`, Stage B no-op). Dormant at 30 Hz (solves reached full `"solved"`); the 4× larger 120 Hz QP triggered it. Fix: accept both strings + `max_iter 8000→20000` so the bigger problems reach full accuracy.
- **Only `λ_smooth` / `GROUND_SMOOTH` scale with rate** (×16 = fps², first-difference penalty in both `_banded_smoother` and `_build_smoothness_hessian`). Collision penalty ρ, trust region, λ_track are position/per-frame terms ⇒ **dt-invariant** (confirmed empirically: standup slip/coll insensitive to λ AND ρ sweeps). Contact pins (foot/hand) are also dt-invariant *for correctness*, but were bumped ×4 (40/8→160/32) to **rebalance** against the ×16 smoothing (see keep-best section above / [[metrics]]).
- **`n_outer=6` at 120 Hz, but the original reasoning was WRONG.** The old note said the 4× larger QP "needs more SCA passes" because get-up coll regressed to ~33% at n=3. That regression was actually the **parity bug** (last-iterate return), now fixed by keep-best-iterate. With keep-best the shipped penetration is parity-immune; `n_outer=6` still gives the loop more chances to *find* a clean iterate but is no longer load-bearing for correctness. Kept at 6.

## Continuation-v1 (2026-07-14, `--continuation N`, EXPERIMENTAL, default off) — gate did NOT clear ship bar
Attempt to close the residual floor penetration Stage B's single pass leaves on hard get-up
clips, by running extra passes: floor rows' allowed penetration shrinks PER-FRAME from that
frame's own pass-0 value toward 0 (`eps_k(t) = pen0(t)·(1−k/K)`), the floor-row slack penalty
hardens 1000→1e5, and tracking relaxes only in violating frame-windows × limbs (never
trunk/root — Stage B's decision vars are actuated joints only). Cross-pass keep-best
(lexicographic: spikes, self-collision, floor pen, slip+floor-err, tracking) seeded with pass 0,
so it can never ship worse than the plain solve.

**Gated on 3 clips, cleared on only 1** (`luigi_standSupine_08`: floor pen 3.56→2.68cm, self-pen
headroom 0.20→0.01cm, 0 spikes — genuine, safe improvement). **The other 2
(`standup_natural_01`, `standup_side_05`) showed zero effect for a reason unrelated to
continuation**: their base Stage-B solve, under `--floor-collision on` (never validated on them
before — see this file's `FLOOR_COLLISION` comment in `retargetingPipeline.sh`), never converges
below its own warm start even at `n_outer=20` (control experiment) — the pre-existing SCA
oscillation this page's own keep-best section already documents, just never exercised on these
two clips until continuation's testing forced floor-collision on for them. A homotopy schedule
can't rescue a base solve that oscillates instead of converging. **Not wired into pipeline
defaults.** Full trail: `wiki/experiments/continuation-v1-gate.md`, `planLog.md` (repo root).

## Why Stage B is on now (history)
Originally shelved: loose contact labels → non-stationary plants → inconsistent median anchors → infeasible/regressing. Onset hysteresis + foot-hold ×10 in [[contact-first-ik]] made plants near-stationary (0.1–0.3 cm), so median anchoring + all-soft + trust region is well-posed. Stage A alone re-adds ~8 cm plant drift.

## Mesh → QP-row hand-off (the mesh never enters the QP)
The dense fullmesh legs are consumed **outside** the solver: `_build_collision` runs `mj_forward` (MuJoCo's mesh narrow-phase), which collapses each colliding pair to a contact point = 3 scalars (`ct.dist`, `ct.frame[:3]` normal, `ct.pos`). One QP row per contact: `pen = margin−dist` is the RHS bound, `j_sep = n̂·(J1−J2)` (via `mj_jac` at `ct.pos`) is the coefficients. QP collision block is `(#active contacts) × T·29` — mesh resolution affects only `mj_forward` cost, **not** QP size/conditioning. Re-queried each SCA outer. Assembly into OSQP `P/q/A/l/u` + DLS→Stage4 data hand-off (qpos_ik = smoothing target + anchor source + linearisation origin) now written in full in METHOD.md §6 intro + §6.2. See [[fullmesh-vs-primitive]].

## Output
`qpos` (= Stage B if run, else Stage A) + `qpos_per_frame`, `qpos_stage_a`, `qpos_stage_b`, contact arrays passed through for the renderer.

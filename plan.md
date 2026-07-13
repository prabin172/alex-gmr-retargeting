# Plan — Hierarchical cascade v1 (CLOSED 2026-07-11)

> **CLOSED, per Prabin's instruction, after H2.** Do not resume execution from this file without
> new direction from Prabin. H1 (root-Z fallback probe) and H2 (Stage-3 hard-tier) both done and
> reported: H1 safe but zero benefit (`planLog.md` H1, `wiki/concepts/limb-cleanup.md`); H2 safe in
> its narrower form but ALSO zero measurable end-to-end benefit and its floor-hard half is
> confirmed broken (`planLog.md` H2, `wiki/experiments/hierarchical-v1-H2-gate.md`). H3/H4 never
> started (H3 was explicitly conditional on H2 showing improvement, which it didn't). Pipeline is
> back to plain phasic-v2 defaults (`S3_HARD_TIER`/`--root-z`/`--floor-hard` all default off,
> verified byte-identical/no-op). See `SESSION_HANDOFF.md` for current state.

> **To the executing model — read this preamble fully before touching anything.**
>
> You are implementing a pre-approved redesign. The design decisions here are settled (made with
> Prabin, 2026-07-11, recorded in `hierarchical.md` §"Claude's comment") — do not re-litigate
> them, do not "improve" the architecture, do not extend scope. Faithful execution, milestone by
> milestone.
>
> **Session start, in order:** read `CLAUDE.md` (repo root), `wiki/index.md`, then ONLY the wiki
> pages a task references. `wiki/concepts/phasic-architecture.md` is the current-architecture
> entry point. Read `planLog.md`'s M5 section BEFORE touching `refine_limbs_contactfirst.py` —
> several fix attempts were tried and REJECTED with measured numbers; do not re-try them.
>
> **Working rules:**
> 1. Execute milestones strictly in order (H0 → H4). Within a milestone, tasks in order.
> 2. Every milestone ends with its Gate. If the gate FAILS: stop, write down what failed and the
>    numbers, do NOT proceed or silently tune around it — report to Prabin.
> 3. Shipped numbers come from the real pipeline (`retargetingPipeline.sh`), never scratch calls.
> 4. Prabin commits himself. Do not commit or push unless he instructs. Leave work staged-ready.
> 5. Keep existing code paths flag-gated rather than deleted (fallbacks are part of the design).
> 6. After each milestone: update the touched wiki pages, append one line to `wiki/log.md`
>    (`## [YYYY-MM-DD] <op> | <what>`). Update `SESSION_HANDOFF.md` only when Prabin instructs.
> 7. Log the build/debug trail (every bug, every rejected fix, with numbers) in `planLog.md`
>    under new `H*` headings — same discipline as the phasic-v2 sections.
>
> **Footguns (violations corrupt data silently — check before every edit):**
> - Coord frame +X fwd / +Y left / +Z up. Quaternions **wxyz** everywhere.
> - Free root qpos layout `[x, y, z, qw, qx, qy, qz, 29 joints]` — 0–6 root, 7–35 actuated.
> - `assets/alex/alex_floating_base_with_sites.xml` is hand-maintained — READ ONLY. Never run
>   `create_alex_mujoco_sites_model.py`, `build_alex_v2_collision_model.py`, or any `prepare_*`.
> - Morphology scaling applies ONLY to motion deltas from rest — never absolute root/pelvis position.
> - `diagnose_floor_penetration.py`: always pass a fixed `--floor-z` when comparing runs.
> - Env: conda `gmr`, `MUJOCO_GL=egl`.

## Background: phasic-v2 outcome (executed 2026-07-10 — compressed from this file's prior content)

Phasic-v2 (M0–M6, all done, branch `phasic-v2`, uncommitted) replaced per-clip floor hacks with an
upstream floor=0 invariant + decoupled phases. Shipped always-on: `scripts/contact_labels.py`,
`scripts/ground_canonical_human.py` (Stage 2.5), Stage-3 per-window target-space floor correction,
`S3_FLOOR_WEIGHT=10`. Shipped opt-in (default off, byte-identical no-op verified):
`scripts/physics_plausibility_pass.py` (Stage 4.6, `PHYSICS_PASS=on`) and
`scripts/refine_limbs_contactfirst.py` (Stage 4.7, `LIMB_REFINE=on`).

**Honest verdict: architecture win, not a decisive metric win.** Corpus medians improved
(planted-foot pen 2.38→1.60 cm, self-pen peak 0.77→0.39 cm, coll% 8.3→3.8) but 5/20 clips
regressed on planted pen (`standupFromKneeling_01` 3.6→7.7 cm worst) and swing-foot `anyPen`
slightly worsened. The two Luigi clips — the original motivation — stayed flat vs main's per-clip
hacks (each now carries 1 minimal flag, down from 5–9). M5's per-limb root-frozen solver: 8/20
clean, 5/20 near-miss, **7/20 whole-body-lying clips structurally out of reach** (root frozen ⇒
can't lift the body; keep-best correctly protects them). Full trail: `planLog.md` M0–M6; gates:
`wiki/experiments/phasic-v2-*.md`; baseline-vs-current table:
`wiki/experiments/phasic-v2-M2-T2.2-gate.md`.

## New direction (settled 2026-07-11 — full rationale in `hierarchical.md` §"Claude's comment")

Origin: Prabin's HQP-NLP cascade proposal (`hierarchical.md`), reviewed and reduced. Settled
rulings — do not re-litigate:

1. **No CoM tier.** Quasi-static CoM-in-polygon is the wrong instrument for dynamic get-ups
   (M4's ~40 cm finding). `--enable-com` stays off; build nothing new on CoM.
2. **Zero-slip active contacts are a design GOAL** (Prabin: robots cannot and must not slip).
   Supersedes "a cm of slip is learnable" for active-contact frames. Contacts become hard
   constraints, not weighted rows.
3. **No null-space HQP, no new stack.** With CoM dropped there is one objective tier left, so the
   hierarchy degenerates to: **hard constraints (contact pinning, floor non-penetration, joint
   limits) + soft tracking objective = a single QP per frame.** Stay in MuJoCo/OSQP; position-
   level; reuse `contact_labels.py`, `_load_model_with_floor`, the narrow-phase row machinery,
   `contact_ramp` (temporal ramps remain MANDATORY on any new constraint — the wrist-flick
   lesson).
4. **Torque limits deferred** until IHMC provides actuator specs (model XML has no `<actuator>`
   section). M4's vel/accel pass is the stand-in.

## Milestones & tasks

### H0 — Baseline freeze
- T0.1 Confirm branch `phasic-v2` (or its successor if Prabin committed/renamed). The frozen
  baseline for every H-gate is `wiki/experiments/phasic-v2-M2-T2.2-gate.csv` — this IS the
  current shipped-default pipeline state (verified unchanged after that gate). Do not regenerate
  it; if the tree has drifted (check `git log`/`git status` against `SESSION_HANDOFF.md`), stop
  and report before proceeding.

### H1 — Root-z probe in the per-limb cleanup (cheap probe — run FIRST)
Targets the 7 whole-body-lying clips M5 could not improve: `luigi_standSupine_08`, `standup_02`,
`standup_natural_01/02`, `standup_side_04/05`, `standup_slideHandsBack_03`.
- T1.1 Read `planLog.md` M5 in full (rejected fixes: fixed-reference regularization,
  reset-on-first-failure — do not re-try; PATIENCE=2 and the XY/Z-split ridge are load-bearing).
- T1.2 In `scripts/refine_limbs_contactfirst.py`: add an optional root-z decision variable,
  active only in Gauss-Seidel round ≥ 2, flag `--root-z` (default off). Root x/y/orientation stay
  frozen. The root-z column enters: floor rows for ALL body geoms (this makes CORE-class
  penetration fixable — update `_limb_body_ids()` gating accordingly), effector tracking rows,
  smoothness (banded, same λ convention), and a posture ridge toward the input root-z.
- T1.3 Keep-best lexicographic score unchanged. `--root-z` off must remain byte-identical no-op
  (verify with two consecutive off-runs, the M5 stale-comparison lesson).
- **Gate:** on the 7 lying clips with `--root-z on`: `floor_pen(limb)` AND `floor_pen(core)`
  materially reduced (report exact numbers per clip); self-pen ≤ baseline
  (`phasic-v2-M2-T2.2-gate.csv` `peak_pen_cm`); 0 velocity spikes; keep-best still protects
  (never worse than input). The other 13 clips: unchanged with the flag off. **Report to Prabin
  at this gate regardless of pass/fail — the data decides whether root-z ships inside
  `LIMB_REFINE` by default.** This is a probe; do not proceed to H2 without reporting.

### H2 — Hard-constraint QP in Stage 3 (the hierarchy, reduced to one QP)
- T2.1 In `scripts/solve_fbx_canonical_alex_contactfirst.py`, new mode `--hard-tier on` (pipeline
  env `S3_HARD_TIER`, default off until H4 decides): per frame, per Gauss-Newton iteration,
  convert from weighted rows to OSQP **hard constraints** `l ≤ Ax ≤ u`:
  (a) active-contact support-point pinning to its anchor — zero slip, position-level, α-ramped
      via the existing `contact_ramp` machinery at onset/release (ramp = the hard bound widens
      smoothly, never a step);
  (b) floor non-penetration rows for all geoms (`_load_model_with_floor` + existing narrow-phase
      row assembly);
  (c) joint limits.
  Tracking (effector rows + posture reg + smoothness/warm-start) stays in the objective,
  unchanged.
- T2.2 Infeasibility escape hatch: if OSQP reports primal infeasible on a frame, re-solve with
  slack on the contact rows only (penalty 1e6), and LOG every frame where slack > 1e-4 m —
  per-clip slack report is a gate deliverable. Never silently relax floor or joint limits.
- T2.3 Sliding-contact check BEFORE tuning anything: inspect `contact_labels.py` output on
  `standup_slideHandsBack_03` and both Luigi clips. If a label holds one continuous "plant"
  across a phase where the human support point translates > 2 cm, the labeler must segment it
  into re-plants (hysteresis re-trigger), else the hard pin will fight the motion. Report what
  you find with numbers before changing the labeler.
- **Gate (corpus, `S3_HARD_TIER=on`, real pipeline):** planted-frame support-point slip p95
  ≤ 0.1 cm in Stage-3 output; Stage-3 floor pen ≤ 0.1 cm all geoms; end-to-end
  `eval_artifacts_corpus.py` vs baseline — `jl_viol` stays 0, self-pen ≤ baseline, tracking
  delta ≤ +1 cm RMS per clip, 0 spikes; slack-active frames listed per clip (expect ~0 on most).
  Also attempt: both Luigi clips with their last remaining flag removed — report whether the hard
  tier makes them flag-free.

### H3 — Hard anchors + floor in Stage 4 (CONDITIONAL)
- T3.1 First measure: does Stage 4 (GlobalOPT) reintroduce slip/penetration on H2's output beyond
  the H2 gate numbers? If planted slip p95 stays ≤ 0.1 cm and pen ≤ 0.5 cm end-to-end, SKIP this
  milestone (record the numbers, move on).
- T3.2 If needed: in `scripts/solve_global_trajectory_opt_contactfirst.py`, convert planted-anchor
  rows and floor rows from soft to hard constraints (same OSQP bound pattern), same slack-and-log
  escape hatch as T2.2. Whole-trajectory hard constraints can be infeasible where smoothness and
  an anchor conflict — the slack log tells us where; report, don't tune blind.
- **Gate:** end-to-end corpus planted slip p95 ≤ 0.1 cm, floor pen ≤ 0.5 cm, 0 spikes, tracking
  within +1 cm RMS of baseline.

### H4 — Corpus gate, config decision, ablations, wiki
- T4.1 Full 20-clip batch, `eval_artifacts_corpus.py` + `diagnose_floor_penetration.py
  --floor-z 0` vs the frozen baseline. RENDER=1 visual spot-check on: both Luigi clips,
  `standup_slideHandsBack_03` (deliberate slide-deviation — see H2/T2.3), `kneelingFall_02/03`,
  any clip whose numbers moved.
- T4.2 Defaults decision (data-driven, report to Prabin with the table): does `S3_HARD_TIER`
  flip default-on? Does root-z ship inside `LIMB_REFINE`? Do `PHYSICS_PASS`/`LIMB_REFINE` flip
  on? Present numbers; Prabin decides.
- T4.3 Ablation table: baseline vs `S3_HARD_TIER` vs `S3_HARD_TIER`+`LIMB_REFINE(root-z)` vs all-on.
  Save `wiki/experiments/hierarchical-v1-gate.md` (+ CSVs).
- T4.4 Wiki: update `wiki/concepts/phasic-architecture.md` (hard-tier ownership map),
  `contact-first-ik.md`, `limb-cleanup.md`, `globalopt.md` if H3 ran; log line in `wiki/log.md`.

## Risks / fallbacks
- **Hard contact + joint limits infeasible on some frames** (reach saturation): the T2.2 slack
  log localizes it. If systematic on a clip class, report — the fallback is widening the ramp
  window, NOT weakening the pin.
- **Sliding-contact clips deviate visibly from the human** — this is intended (ruling 2), but
  needs Prabin's visual sign-off at H4; flag those clips explicitly.
- **Whole-body lying frames**: Stage 3's root is free (full qpos solve), so hard floor rows are
  satisfiable by lifting the root — but watch tracking delta on lying phases; if the body
  "hovers" unnaturally, report with renders before tuning.
- **H1 root-z oscillates in Gauss-Seidel**: the M5 divergence lesson applies (PATIENCE=2
  protects); if root-z itself oscillates between rounds, damp it (ridge up), report numbers.

## Verification (every milestone)
Real pipeline only for shipped numbers; compare against the frozen
`phasic-v2-M2-T2.2-gate.csv`; `diagnose_floor_penetration.py --floor-z 0` for any floor claim;
RENDER=1 spot-check any clip whose numbers moved; two consecutive runs for any "no-op" claim.
Done = H4 table + defaults decision delivered to Prabin.

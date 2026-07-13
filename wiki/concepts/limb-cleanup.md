# Per-Limb Cleanup Solver (Stage 4.7, phasic-v2 P4 — plan.md's core deliverable)

`scripts/refine_limbs_contactfirst.py`. Optional post-physics-plausibility stage: fixes floor
penetration, self-collision, and swing clearance for an ISOLATED swinging limb (a foot/hand dipping
through the floor during otherwise-normal motion). Flag-gated (`LIMB_REFINE=on/off` in
`retargetingPipeline.sh`, default **off** — confirmed byte-identical when off, two fresh consecutive
off-runs produce identical output). Runs after physics-plausibility if both are enabled (consumes
whichever stage's output ran last).

## Root frozen — the whole design

`qpos[:, 0:7]` (root position + quaternion) is copied unchanged to every output frame, every round.
Every remaining DOF is a plain hinge joint (`qpos_adr = dof_adr + 1` always, since the free root
joint is the only non-hinge joint) — NO quaternion retraction is needed anywhere in this script,
simpler than `physics_plausibility_pass.py`'s tangent-space machinery.

4 limb chains, each solved as its own whole-clip banded QP: `LEFT_LEG`/`RIGHT_LEG` (6 DOF: hip×3,
knee, ankle×2), `LEFT_ARM`/`RIGHT_ARM` (7 DOF: shoulder×3, elbow, wrist×2, gripper). Gauss-Seidel
over the 4 limbs (legs first), `PATIENCE=2`-based keep-best-iterate across rounds (see FOOTGUN
below).

## Mechanism

Per-limb QP objective: posture ridge (toward the limb's own current joint angles — never freeze,
the `refine_arm_floor_transitions` lesson) + banded smoothness (λ=320, same pattern as Stage
4/physics-plausibility) + Cartesian effector-tracking ridge, split XY (boosted on planted frames,
`--plant-hold-boost`) vs Z (never boosted — a floor-fix must never be fought, same pattern as Stage
4's on-floor rows dropping to X,Y only).

Inequality rows, all restricted to nonzero-Jacobian columns for THIS limb (excludes frozen-body-only
contacts automatically, no explicit filtering needed):
- Floor non-penetration for every geom on the limb (`_load_model_with_floor`, same in-memory
  injection pattern as Stage 3/4).
- Self-collision vs the rest of the body, k=2-hop-adjacency skip (matches Stage 3/4).
- Swing clearance (new, T5.3): when this limb's OWN contact envelope alpha (reconstructed from the
  persisted bool `contact_flags` via `contact_labels.py`'s `ramp_envelope`) is near 0 AND its
  support point is within an activation band, a one-sided row keeps it at or above
  `--swing-clearance` (default 2cm), weighted by `(1-alpha)` — fades OUT as contact fades IN, never
  fights touchdown.

## FOOTGUN — Gauss-Seidel needs patience-based reset, not unconditional (2026-07-10)

A naive "carry `qpos_cur` forward regardless of round outcome" loop DIVERGES on clips with a severe,
widespread violation (`standup_02`: `pen+self` grew monotonically 11.26→26.72cm over 10 rounds, no
recovery). The fix is NOT simply "reset to best_qpos on every failure" — that was tried and
regressed a previously-clean case (`shovel_fronthard_02`: a round-1 result 0.03cm short of
improving got its near-miss thrown away every round instead of given one more round to close).
**Final design**: `PATIENCE=2` — a failing round is allowed to keep accumulating for up to 2
consecutive failures before the loop resets to the last known-good state. Verified all three
properties hold together: no regression on cleanly-converging clips, genuinely safe (no divergence)
on clips that can't improve, and near-misses get the extra round they need. See `planLog.md` M5 for
the full investigation (5 real bugs found and fixed, 2 rejected alternatives with measured
trade-offs each).

## Scope: works for isolated swing violations, safely no-ops on whole-body lying phases

Verified on the full 20-clip corpus (`wiki/experiments/phasic-v2-M5-gate.md`):
- **8/20 clean pass** — isolated swing-limb violations, the case this was built for.
- **5/20 near-miss** — real improvement, just short of the strict 0.5cm gate.
- **7/20 severe, safely protected** — an entire limb genuinely lying against/through the floor for
  an extended phase (confirmed via direct body-contact-count inspection: torso, pelvis, both
  thighs, both feet, elbows, shoulders ALL in floor contact simultaneously — not an isolated dip).
  This is a genuine structural limit of a ROOT-FROZEN, per-limb-only solver — **exactly the risk
  plan.md's own "Risks/fallbacks" section anticipated** ("root frozen ⇒ reach saturation...
  fallback = allow a root-z DOF"), not built here given time budget. `keep-best-iterate` correctly
  protects all 7 (never ships worse than input) but cannot improve them.

**On 4 of the 20 clips, keep-best accepts a self-collision INCREASE over the pre-M5 baseline** in
exchange for floor-pen/slip improvement elsewhere — correct per the lexicographic score's stated
priority (floor-pen > slip > tracking > selfpen), but a real cost not visible from the pass's own
printed metrics alone (only surfaced by cross-checking against the M4 baseline's `peak_pen_cm`).

## Root-Z probe (hierarchical-v1 H1, 2026-07-11) — tried, safe, but NO benefit

`--root-z` (default off): plan.md's own named M5 fallback ("allow a root-z DOF in P4 round 2"),
targeting exactly the 7 whole-body-lying clips above. Adds a 5th 1-DOF pseudo-limb (qpos index
2 only — root x/y/orientation stay hard-frozen) using the same `_solve_limb_qp` machinery,
active from Gauss-Seidel round 1 onward (`--root-z-start-round`), trust region 3cm/round
(`--root-z-trust-region`). Floor rows for this DOF are NOT limb-restricted, so it's the one
mechanism that could touch CORE-classified (torso/pelvis) penetration.

**Result: engages every round (verified via nonzero `root_z_floor_rows` in the per-round log) but
never wins keep-best, on any of the 7 target clips** — `Root-Z delta: max=0.00cm` every time,
`floor_pen(core)` bit-for-bit identical warm→final on all 7. The moment root-z-touched rounds
enter the mix, self-collision jumps sharply enough to lose lexicographically even in rounds where
`floor_pen(core)` itself improved transiently. Mechanism is safe (0 spikes, root x/y/quat exactly
frozen, keep-best never regresses) but delivers zero value as tuned. Root cause not isolated —
candidates: Gauss-Seidel solves limbs before root-z each round so effects entangle; root-z could
instead run first, or from round 0, or with a looser self-collision penalty for its own rows — none
tried. Full per-clip numbers and per-round trace: `planLog.md` H1. **Not shipped as part of
`LIMB_REFINE`'s default behavior; stays in the codebase, opt-in, off by default.**

## Why NOT a replacement for the Luigi per-clip floor flags

`luigi_standSupine_08` is one of the 7 severe cases M5 cannot improve — its existing
`--floor-collision on --floor-phase-aware on` mechanism (M2/M3, see [[globalopt]]) already handles
what M5 structurally cannot. Removing it because M5 exists would be a regression for this clip, not
a supersession. M5 ships as an ADDITIONAL, optional layer.

## Output NPZ

Same schema as the input, `qpos` replaced with the cleaned-up trajectory, plus
`qpos_pre_limb_refine` and `limb_refine_meta_json` (limits used, floor_pen_limb/core, plant_slip,
selfpen, spikes, root-frozen check).

Related: [[physics-plausibility]] (Stage 4.6, upstream), [[globalopt]] (Stage 4, on-floor rows
pattern reused here), [[contact-first-ik]] (Stage 3, `refine_arm_floor_transitions`'s
never-freeze lesson reused here).

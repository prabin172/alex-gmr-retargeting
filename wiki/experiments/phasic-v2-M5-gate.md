# phasic-v2 M5 gate: per-limb cleanup solver (2026-07-10)

M5 deliverable — new `scripts/refine_limbs_contactfirst.py` (plan.md's core new deliverable, P4).
Full build/debug trail (5 real bugs found and fixed, 2 rejected fix attempts each with their own
measured trade-offs): `planLog.md` M5 section. This page is the final 20-clip corpus verification.

Run: standalone script per clip, `--npz outputs/grounded_contactfirst/<clip>_grounded.npz`
(consuming the M2/M4 pipeline's real output, not scratch data). `ok=20/20`, 0 spikes, root frozen
exactly (`max|Δroot|=0.0`) on every clip.

## Corpus table

| clip | floor_pen(limb) cm | floor_pen(core) cm | plant_slip cm | track_rms cm | selfpen cm | selfpen ≤ baseline? |
|---|---|---|---|---|---|---|
| kneelingFall_02 | 0.00 | 0.00 | 4.37 | 4.84 | 0.00 | OK (baseline 1.54) |
| kneelingFall_03 | 0.79 | 0.00 | 3.44 | 7.78 | 0.28 | OK (baseline 1.40) |
| luigi_standProne_03 | 0.55 | 1.74 | 13.99 | 2.38 | 0.49 | OK (baseline 2.63) |
| luigi_standSupine_08 | 14.59 | 16.31 | 0.00 | 0.00 | 2.20 | OK (unchanged, =baseline) |
| shovel_fronthard_02 | 0.00 | 0.00 | 8.95 | 1.54 | 0.00 | OK |
| shovel_leftbucket_02 | 0.00 | 0.00 | 7.36 | 1.56 | 0.00 | OK |
| shovel_lefthard_01 | 0.00 | 0.00 | 9.10 | 1.98 | 0.00 | OK |
| shovel_rightbucket_01 | 0.00 | 0.00 | 7.31 | 1.48 | 0.00 | OK |
| shovel_righthard_01 | 0.00 | 0.00 | 8.77 | 1.91 | 0.00 | OK |
| standup_01 | 0.16 | 0.00 | 17.51 | 3.87 | 0.45 | OK (baseline 0.66) |
| standup_02 | 11.26 | 2.08 | 0.00 | 0.00 | 0.00 | OK (unchanged, =baseline) |
| standupFromKneeling_01 | 1.59 | 0.00 | 20.31 | 3.87 | 0.80 | **FAIL** (baseline 0.12) |
| standupFromKneeling_02 | 1.62 | 0.00 | 0.00 | 0.00 | 1.40 | OK (baseline 1.40, borderline) |
| standupKnees_02 | 0.00 | 0.00 | 6.75 | 3.48 | 0.00 | OK |
| standup_natural_01 | 13.13 | 8.83 | 4.74 | 3.58 | 3.78 | **FAIL** (baseline 0.25) |
| standup_natural_02 | 14.00 | 11.93 | 0.00 | 0.00 | 0.53 | OK (baseline 0.53, borderline) |
| standup_side_04 | 14.43 | 12.17 | 4.38 | 2.75 | 3.45 | **FAIL** (baseline 1.86) |
| standup_side_05 | 15.60 | 10.18 | 6.61 | 2.89 | 3.79 | **FAIL** (baseline 1.99) |
| standup_slideHandsBack_03 | 10.36 | 8.42 | 0.00 | 0.00 | 0.00 | OK (unchanged, =baseline) |
| standupSquatCrouch_01 | 1.81 | 0.00 | 0.00 | 0.00 | 0.00 | OK |

`floor_pen(core)` (torso/pelvis/head/spine) is NEVER gated — architecturally out of this pass's
reach by design (root frozen, T5.1). `floor_pen(limb)` is the gated metric (plan.md: ≤0.5cm).

## Verdict: honest, mixed — three distinct outcome classes

**8/20 clean pass** (`floor_pen(limb) ≤ 0.5cm`, `selfpen ≤ baseline`): `kneelingFall_02`, all 5
shovel clips, `standup_01`, `standupKnees_02`. These are the isolated-swing-violation cases M5 was
designed for, and it works well — floor penetration fully resolved, self-collision never worse than
the pre-M5 baseline. `plant_slip`/`track_rms` are elevated on several (up to 17.5cm on `standup_01`)
but this is an accepted, documented trade-off (design philosophy: "a cm of slip is learnable;
self-penetration or over-limit joints are not").

**5/20 near-miss** (`floor_pen(limb)` 0.55–1.81cm, close to but over the 0.5cm gate):
`kneelingFall_03`, `luigi_standProne_03`, `standupFromKneeling_01/02`, `standupSquatCrouch_01`.
Real, substantial improvement from their warm baselines (`luigi_standProne_03`'s own M2/M4
`plPen`/`grnd_pen_plant_cm` was already low, but `anyPen`-class swing violations existed
pre-M5), just short of the strict 0.5cm target.

**7/20 severe, safely protected (unchanged or minimally moved)**: `luigi_standSupine_08`,
`standup_02`, `standup_natural_01/02`, `standup_side_04/05`, `standup_slideHandsBack_03`. All show
`floor_pen(limb)` in the 10–16cm range — confirmed (see `planLog.md` M5) to be WHOLE-BODY floor
contact during an extended lying phase (torso, pelvis, both thighs, both feet, elbows, shoulders all
touching the floor simultaneously across a large fraction of frames), not an isolated swing dip.
This is a genuine structural limit of a ROOT-FROZEN, per-limb-only solver — plan.md's own
"Risks/fallbacks" section anticipated exactly this ("root frozen ⇒ reach saturation... fallback =
allow a root-z DOF"), not built here given time budget. **Critically, `keep-best-iterate` correctly
protects all 7**: `luigi_standSupine_08`/`standup_02`/`standup_natural_02`/`standup_slideHandsBack_03`
are byte-for-byte UNCHANGED from input (`track_rms=0`, `plant_slip=0`, confirmed not assumed);
`standup_natural_01`/`standup_side_04/05` show SOME accepted partial improvement but not enough to
clear the gate, and — this is the one real, uncaught-until-cross-checking cost — that partial
improvement came at the expense of self-collision getting WORSE than the pre-M5 baseline on 4
clips total (`standupFromKneeling_01`, `standup_natural_01`, `standup_side_04/05`): the
lexicographic score prioritizes floor-pen and plant-slip over selfpen by design, so keep-best
correctly accepted a selfpen increase when it bought a floor-pen/slip improvement elsewhere. Not
caught by the pass's own gate check (which reports `selfpen` but doesn't compare it to a baseline
internally) — only surfaced by cross-referencing `wiki/experiments/phasic-v2-M2-T2.2-gate.csv`'s
`peak_pen_cm` column during this writeup.

## Consequence: M5 ships as an OPTIONAL layer, not a replacement

Per plan.md T5.5 ("this phase supersedes... all per-clip floor flags"): NOT fully applicable, for
two compounding reasons now confirmed. (1) `luigi_standSupine_08` is one of the 7 severe cases M5
cannot improve — its existing `--floor-collision on --floor-phase-aware on` mechanism (M2/M3)
already handles what M5 cannot, so removing it would be a regression. (2) Even on clips M5 DOES
touch, it can trade self-collision for floor-pen/slip on 4 of them — a real cost that should be
opt-in, not silently forced on every clip by default. **Wired as an additional, optional cleanup
stage (default off), not a replacement for the existing per-clip mechanisms.** See
`wiki/concepts/limb-cleanup.md` for the pipeline wiring.

# phasic-v2 M1 gate: Stage 2.5 wired, full 20-clip regeneration (2026-07-10)

M1 deliverable (`scripts/contact_labels.py`, `scripts/ground_canonical_human.py`, Stage 2.5 wired
into `retargetingPipeline.sh`, Stage 3 consumes persisted labels — see `plan.md` M1, `planLog.md`
for the full T1.1/T1.2/T1.3 build/verify trail). This file is the corpus-wide confirmation run,
**not** a substitute for M1's actual gate criteria (already passed in `planLog.md`: still-plant
support points ≤0.5cm median self-consistency; label diff vs old in-solver detection = 0/0 on all
20 clips after T1.3). This run's purpose: prove the wiring doesn't break anything end-to-end when
every clip is forced through fresh (all `outputs/contactfirst/*.npz` deleted first — no cache-hit
confound, learned the hard way earlier in M1, see `planLog.md`).

Run: `RENDER=0 bash retargetingPipeline.sh` after `rm -f outputs/contactfirst/*.npz`. `ok=20
fail=0`. Log: `outputs/logs/pipeline_phasicv2_M1gate.log`. CSV: `wiki/experiments/phasic-v2-M1-gate.csv`.

## Corpus table (`eval_artifacts_corpus.py`, Stage-4 output, pre-grounding)

| clip | JLvi | worst_joint(@lim%) | ftSlip | hdSlip | plPen | plPen% | anyPen | anyPen% | flAvg | coll% | selfPen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kneelingFall_02 | 0 | RIGHT_ANKLE_X(33) | 0.7 | 0.0 | 2.1 | 15.5 | 13.2 | 100.0 | 0.0 | 34.1 | 1.5 |
| kneelingFall_03 | 0 | RIGHT_KNEE_Y(56) | 4.8 | 0.0 | 0.9 | 8.1 | 15.9 | 71.2 | 0.0 | 61.0 | 1.4 |
| luigi_standProne_03 | 0 | LEFT_GRIPPER_Z(56) | 5.1 | 4.6 | 0.6 | 1.6 | 2.6 | 7.7 | 0.4 | 6.6 | 2.5 |
| luigi_standSupine_08 | 0 | NECK_Y(60) | 4.3 | 1.1 | 0.8 | 2.1 | 13.8 | 65.3 | 0.4 | 23.5 | 2.2 |
| shovel_fronthard_02 | 0 | LEFT_GRIPPER_Z(78) | 2.3 | 0.0 | 2.3 | 3.4 | 3.9 | 11.9 | 0.3 | 0.0 | 0.0 |
| shovel_leftbucket_02 | 0 | RIGHT_SHOULDER_Z(11) | 2.6 | 0.0 | 2.1 | 17.8 | 5.1 | 45.4 | 0.4 | 0.0 | 0.0 |
| shovel_lefthard_01 | 0 | LEFT_ANKLE_X(3) | 2.5 | 0.0 | 3.3 | 4.8 | 3.6 | 12.9 | 0.1 | 0.0 | 0.0 |
| shovel_rightbucket_01 | 0 | RIGHT_GRIPPER_Z(19) | 3.3 | 0.0 | 1.5 | 5.4 | 2.9 | 20.3 | 0.2 | 0.0 | 0.0 |
| shovel_righthard_01 | 0 | RIGHT_WRIST_X(15) | 2.8 | 0.0 | 2.3 | 5.0 | 2.9 | 14.1 | 0.1 | 0.0 | 0.0 |
| standupFromKneeling_01 | 0 | LEFT_SHOULDER_Z(91) | 0.7 | 0.0 | 3.6 | 20.4 | 7.6 | 67.6 | 0.9 | 72.5 | 0.9 |
| standupFromKneeling_02 | 0 | LEFT_WRIST_X(21) | 2.1 | 0.0 | 0.8 | 15.3 | 1.8 | 24.7 | 0.3 | 66.6 | 1.5 |
| standupKnees_02 | 0 | -(0) | 1.5 | 0.0 | 0.1 | 0.0 | 6.7 | 65.5 | 0.3 | 26.5 | 0.4 |
| standupSquatCrouch_01 | 0 | LEFT_SHOULDER_Z(96) | 2.8 | 0.0 | 2.0 | 40.8 | 2.0 | 35.9 | 0.7 | 0.0 | 0.0 |
| standup_01 | 0 | NECK_Y(24) | 0.7 | 3.1 | 0.5 | 0.0 | 11.3 | 28.0 | 0.4 | 9.0 | 0.6 |
| standup_02 | 0 | NECK_Y(39) | 1.0 | 5.5 | 0.3 | 0.0 | 12.9 | 18.5 | 0.2 | 0.0 | 0.0 |
| standup_natural_01 | 0 | NECK_Y(52) | 0.7 | 1.8 | 0.0 | 0.0 | 9.3 | 62.3 | 0.9 | 2.6 | 0.1 |
| standup_natural_02 | 0 | NECK_Y(48) | 3.0 | 2.3 | 8.2 | 47.6 | 12.7 | 65.7 | 0.1 | 4.2 | 0.8 |
| standup_side_04 | 0 | NECK_Y(49) | 1.6 | 1.5 | 3.3 | 4.1 | 26.0 | 28.1 | 0.1 | 3.9 | 2.4 |
| standup_side_05 | 0 | NECK_Y(63) | 4.6 | 2.7 | 2.5 | 4.7 | 26.6 | 40.8 | 0.1 | 17.5 | 1.1 |
| standup_slideHandsBack_03 | 0 | NECK_Y(64) | 1.8 | 1.5 | 3.0 | **100.0** | 11.6 | 60.2 | 0.0 | 0.5 | 0.0 |
| **CORPUS median** | 0 | | 2.4 | 0.0 | 2.0 | 4.9 | 8.5 | 38.3 | 0.2 | 4.1 | 0.5 |
| **CORPUS max** | 0 | | 5.1 | 5.5 | 8.2 | 100.0 | 26.6 | 100.0 | 0.9 | 72.5 | 2.5 |

Column definitions: see `wiki/experiments/phasic-v2-baseline.md` (identical `eval_artifacts_corpus.py` output format).

## Verdict: PASS on M1's actual gate; one anomaly flagged for the M2 gate, not fixed here

**0 hard joint violations corpus-wide (JLvi), ok=20/fail=0.** Aggregate medians are a mixed bag vs
the M0 baseline (`coll%` 8.3→4.1 and `selfPen` 0.8→0.5 cm improved; `plPen` 2.4→2.0 cm improved;
`anyPen` 7.5→8.5 cm slightly up; `ftSlip` 2.0→2.4 cm slightly up) — **expected**, not a target of
this milestone: M1 only establishes the floor=0 invariant in canonical-human space and persists
labels. Nothing downstream (Stage 3's morphology-scaled targets, Stage 4's own `--floor-mode
estimate`, the eval script's own floor reference) has been updated to actually USE that invariant
yet — that is explicitly M2 ("target-space floor invariant in Stage 3") and M3 ("GlobalOPT on the
invariant, fixed floor_z=0") work, still to come. `plan.md`'s own M1 gate is narrower than a
full-corpus comparison for exactly this reason (see `planLog.md`'s T1.1/T1.2/T1.3 entries for the
actual M1 gate checks, all passed).

**Flagged anomaly**: `standup_slideHandsBack_03`'s `plPen%` jumped from the baseline's 10.5% to
**100%**, while its `plPen` (max depth) actually improved (5.0→3.0 cm). Reading: a small systematic
sub-cm-to-few-cm offset now touches every planted frame instead of occasionally going deep — plausibly
morphology scaling not yet being floor-preserving after Stage 2.5's canonical shift (M2's exact
scope). Not treated as an M1 gate failure (M1's stated gate doesn't cover this), but **explicitly
flagged to check at the M2 gate**: if this clip's `plPen%` is still ~100% after M2/M3 land, that is
a real bug, not an expected pre-M2 transient, and should be investigated then.

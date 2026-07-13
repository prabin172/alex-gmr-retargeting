# phasic-v2 M0 frozen baseline (2026-07-10)

Baseline for the phasic-v2 redesign (see `plan.md`, repo root). Branch `phasic-v2`, forked from
`main`@`0d79f53` ("Tried phase aware floor collision, didn't work, keeping code") — code is
byte-identical to `main` at this point; only this table + `plan.md`/`planLog.md` are new.

Run: `RENDER=0 bash retargetingPipeline.sh` (all 20 clips, Stage 3 cache-hit from the prior
main-branch session's outputs — same HEAD, so still a valid baseline; Stage 4/4.5/6 recomputed).
`ok=20 fail=0`. Log: `outputs/logs/pipeline_phasicv2_M0baseline.log`. Raw CSV:
`wiki/experiments/phasic-v2-baseline.csv` (`eval_artifacts_corpus.py --csv ...`).

**Every later milestone's gate compares against this table.** Do not regenerate/overwrite it —
if the redesign needs a new baseline, save a new dated file instead.

## Corpus table (`eval_artifacts_corpus.py`, Stage-4 output, pre-grounding)

| clip | JLvi | worst_joint(@lim%) | ftSlip | hdSlip | plPen | plPen% | anyPen | anyPen% | flAvg | coll% | selfPen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kneelingFall_02 | 0 | RIGHT_ANKLE_X(33) | 0.7 | 0.0 | 2.1 | 15.5 | 13.2 | 100.0 | 0.0 | 34.1 | 1.5 |
| kneelingFall_03 | 0 | RIGHT_KNEE_Y(56) | 4.8 | 0.0 | 0.9 | 8.1 | 15.9 | 71.2 | 0.0 | 61.0 | 1.4 |
| luigi_standProne_03 | 0 | LEFT_GRIPPER_Z(56) | 5.1 | 4.6 | 0.6 | 1.6 | 2.6 | 7.7 | 0.4 | 6.6 | 2.5 |
| luigi_standSupine_08 | 0 | NECK_Y(60) | 4.3 | 1.1 | 0.8 | 2.1 | 13.8 | 65.3 | 0.4 | 23.5 | 2.2 |
| shovel_fronthard_02 | 0 | LEFT_GRIPPER_Z(77) | 2.6 | 0.0 | 3.3 | 8.9 | 4.5 | 14.8 | 0.1 | 0.0 | 0.0 |
| shovel_leftbucket_02 | 0 | RIGHT_WRIST_X(4) | 2.4 | 0.0 | 3.1 | 7.5 | 3.1 | 16.1 | 0.2 | 0.0 | 0.0 |
| shovel_lefthard_01 | 0 | LEFT_WRIST_Z(2) | 2.3 | 0.0 | 3.3 | 5.6 | 3.5 | 13.9 | 0.1 | 0.0 | 0.0 |
| shovel_rightbucket_01 | 0 | RIGHT_GRIPPER_Z(19) | 3.5 | 0.0 | 2.3 | 7.3 | 2.3 | 14.9 | 0.2 | 0.0 | 0.0 |
| shovel_righthard_01 | 0 | RIGHT_WRIST_X(14) | 1.9 | 0.0 | 2.7 | 4.7 | 2.7 | 11.7 | 0.1 | 0.0 | 0.0 |
| standupFromKneeling_01 | 0 | LEFT_SHOULDER_Z(91) | 0.7 | 0.0 | 3.6 | 20.4 | 7.6 | 67.6 | 0.9 | 72.5 | 0.9 |
| standupFromKneeling_02 | 0 | LEFT_WRIST_X(21) | 2.1 | 0.0 | 0.8 | 15.3 | 1.8 | 24.7 | 0.3 | 66.6 | 1.5 |
| standupKnees_02 | 0 | -(0) | 1.5 | 0.0 | 0.1 | 0.0 | 6.7 | 65.5 | 0.3 | 26.5 | 0.4 |
| standupSquatCrouch_01 | 0 | LEFT_SHOULDER_Z(96) | 2.7 | 0.0 | 2.2 | 41.0 | 2.2 | 36.3 | 0.6 | 0.0 | 0.0 |
| standup_01 | 0 | NECK_Y(23) | 0.7 | 2.2 | 0.5 | 0.5 | 10.3 | 29.7 | 0.2 | 6.3 | 0.8 |
| standup_02 | 0 | NECK_Y(38) | 1.0 | 6.1 | 0.5 | 0.0 | 7.5 | 31.8 | 0.2 | 11.0 | 1.4 |
| standup_natural_01 | 0 | RIGHT_WRIST_Z(48) | 1.7 | 0.6 | 2.5 | 8.2 | 12.6 | 52.4 | 0.1 | 8.7 | 0.2 |
| standup_natural_02 | 0 | NECK_Y(48) | 1.8 | 1.2 | 6.6 | 18.5 | 12.5 | 56.1 | 0.1 | 4.1 | 0.8 |
| standup_side_04 | 0 | NECK_Y(50) | 1.9 | 1.7 | 4.1 | 23.8 | 16.8 | 46.9 | 0.0 | 8.8 | 1.6 |
| standup_side_05 | 0 | NECK_Y(63) | 3.7 | 4.1 | 2.9 | 17.9 | 28.5 | 53.0 | 0.2 | 17.7 | 1.1 |
| standup_slideHandsBack_03 | 0 | NECK_Y(63) | 1.8 | 1.5 | 5.0 | 10.5 | 9.4 | 55.0 | 0.1 | 7.9 | 0.6 |
| **CORPUS median** | 0 | | 2.0 | 0.0 | 2.4 | 8.1 | 7.5 | 41.6 | 0.2 | 8.3 | 0.8 |
| **CORPUS max** | 0 | | 5.1 | 6.1 | 6.6 | 41.0 | 28.5 | 100.0 | 0.9 | 72.5 | 2.5 |

Columns (cm unless noted): `ftSlip`/`hdSlip` = max planted foot(horiz)/hand(3D) slip off frozen
anchor. `plPen` = deepest PLANTED-foot sole below floor (hard violation); `plPen%` = % of planted
frames past 0.5cm tol. `anyPen`/`anyPen%` = same but incl. swing/tucked feet (softer — includes the
known non-contact-foot float during deep-crouch phases, see `wiki/concepts/grounding.md`).
`flAvg` = planted-foot hover above floor. `coll%` = frames with self-collision. `selfPen` = peak
inter-limb penetration. `JLvi` = hard joint-limit violations (all 0 — clean corpus-wide).

**Known baseline characteristics carried into this run** (context, not new findings — see
`wiki/concepts/globalopt.md`/`grounding.md`): the two Luigi clips run with per-clip floor flags
(`luigi_standProne_03`: `--floor-refine` + `--floor-collision on`; `luigi_standSupine_08`: same +
`--floor-phase-aware on` both stages) — this is exactly the per-clip config the redesign eliminates
(M2/M3/M5). `anyPen` is high corpus-wide (median 41.6% of frames) — this is the swing/tucked-foot
float the redesign's P4 swing-clearance + floor rows target. Fall clips (`kneelingFall_02/03`) carry
the largest `anyPen` (100%/71.2%) — the known late free-foot-below-floor regime.

## Pre-grounding floor reference per clip (Alex frame, `z_min` before the Stage-4.5 shift)

For fixed-`--floor-z` comparisons with `diagnose_floor_penetration.py` against the Stage-4 output
in later milestones (per plan.md's "always pass a fixed --floor-z" rule) — this is the
`constant-contact` planted-foot-sole p50 that Stage 4.5 shifted to 0 in this baseline run.

| clip | floor_z (pre-shift) |
|---|---|
| standup_01 | -0.0883 |
| standup_02 | -0.1042 |
| standup_natural_01 | -0.0850 |
| standup_natural_02 | -0.0600 |
| standup_side_04 | -0.1012 |
| standup_side_05 | -0.1023 |
| standup_slideHandsBack_03 | -0.0852 |
| shovel_fronthard_02 | -0.8199 |
| shovel_leftbucket_02 | -0.8294 |
| shovel_lefthard_01 | -0.8189 |
| shovel_rightbucket_01 | -0.8244 |
| shovel_righthard_01 | -0.8304 |
| standupFromKneeling_01 | -0.4775 |
| standupFromKneeling_02 | -0.4606 |
| standupKnees_02 | -0.3868 |
| standupSquatCrouch_01 | -0.2682 |
| kneelingFall_02 | -0.8829 |
| kneelingFall_03 | -0.8806 |
| luigi_standProne_03 | -0.1071 |
| luigi_standSupine_08 | -0.0631 |

(shovel/kneeling clips sit far below 0 because those FBX rigs' root frame differs — expected, not
a bug; only the shift magnitude matters, not the absolute value.)

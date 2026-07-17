# GMR-baseline Sprint S1 (2026-07-16, branch `gmr-baseline`)

Follow-up to [[gmr-baseline-week2]]. Part of the Humanoids-2026 sprint (`GMR-baseline-plan.md`'s
SPRINT section): the full 2×2 comparison (GMR-as-published vs OURS-on-G1, each with/without
polish, `GMR-baseline.md` §7.4). This page covers **S1** (top row: full-corpus kinematic sweep),
DONE. S2 (bottom row: OURS on G1) is in progress, parked at its own checkpoint — see
`GMR-baseline-results.md`'s "Sprint S2" section and `planLogGMR.md` `## S2-T1..T5`; not detailed
here. Full task-by-task trail: `planLogGMR.md` `## S1-Tn` headings.

## S1-T1/T2/T3 — batch retarget, polish variants, eval, all 77 LAFAN1 clips

Extended week-1/2's pipeline from 5 hand-picked clips to the full corpus: `gmr_headless_retarget.py`
(+`--save_human_targets`, GMR's own scaled-human FK targets per frame) → `polish_gmr_pkl.py`
(`--heightfix`, `--stage-a --ground`) → `eval_motion.py`'s `evaluate()`, all reused unchanged, just
batched (new: `sprint_batch_retarget.sh`, `sprint_polish_batch.sh`, `sprint_eval_batch.py`).
**77/77 clips, 0 failures, all 3 batches.** One bug caught and fixed mid-run: the first polish-batch
script globbed its clip list from the pkl output directory, which also held stale week-1/2 leftover
files with colliding names — fixed by sourcing the clip list from the BVH directory instead (S1-T1's
own ground truth), no real clip corrupted.

**Two new metrics added at corpus scale**:
- **Self-collision via the vetted model** (W2-T6's `g1_collision_vetted.urdf`), not the unvetted
  default `eval_motion.py` still uses — a separate `_collision_stats` pass on the same qpos.
- **Faithfulness guard**: FK'd robot-body position vs GMR's own scaled-human target
  (`human_targets/<clip>.npz`), using `bvh_lafan1_to_g1.json`'s `ik_match_table2` — confirmed by
  reading `motion_retarget.py` that `table2` (not `table1`) carries the real position-tracking
  weight (10-100 across all 14 pairs; `table1`'s `position_cost` is 0 for pelvis and low
  elsewhere, i.e. mostly orientation). `pos_offset==[0,0,0]` and `ground_height==0.0` for every
  entry, so the saved target position needs no further transform. Result: polish doesn't wander —
  deltas are small, concentrated on floor-class clips where whole-clip Z-shift genuinely trades
  fidelity for floor placement (expected), several clips even improve slightly.

**Regression gate: passes exactly.** All 5 week-1/2 clips reproduce their original numbers to the
decimal on every metric — this sprint's new batch tooling introduced no drift from the original
manually-run pipeline.

## S1-T4 — class split, corrected mid-sprint

**Initial split** (hip-Z p5 < 0.3, week-1 T2's own convention, applied at scale): 20 floor-class /
57 locomotion-class. Immediately flagged as suspect: 29 of the 57 "locomotion" clips showed raw
floorPen > 5cm, several worse than the hand-picked floor clips (`pushAndStumble1_subject3` 31.5cm,
`walk2_subject3` 21.1cm) — a hand or knee touching the ground briefly while the hip stays up, the
exact blind spot [[gmr-baseline-week2]]'s W2-T3 already diagnosed ("hip height alone is an
incomplete floor-contact signal").

**Fix**: reclassified using W2-T3's own multi-surface human-contact detector
(`human_contacts_lafan1.py`, unchanged thresholds) run over all 77 clips, with an explicit rule —
floor-class if ANY non-foot landmark (hand/knee/elbow/pelvis/torso) has a CONTIGUOUS in-zone run
≥1 second, not just a nonzero percentage (a bare %-threshold would let a busy fight/dance clip
cross via many short scattered dips without ever resting on anything). Sanity-checked against the
5 known clips: `dance1_subject1`'s previously-noted "noise-level" hand blip (0.43s) correctly
stays below the 1s bar; the 3 hand-picked floor clips all clear it by 3-10×.

**Result: 34 floor-class / 43 locomotion-class** — 14 more floor clips than the hip-only split
found. This is the split shipped in the paper table.

| class | variant | floorPen | pen% | float% | coll%(vetted) | vMax | faith_mean |
|---|---|---|---|---|---|---|---|
| locomotion (43) | raw | 4.77cm | 1.91% | 93.5% | 3.85% | 32.82 | 10.19cm |
| locomotion (43) | polished | 2.59cm | 0.49% | 97.9% | 3.75% | 5.45 | 10.90cm |
| floor (34) | raw | 17.00cm | 27.35% | 69.0% | 6.32% | 34.04 | 10.23cm |
| floor (34) | polished | 4.14cm | 0.70% | 98.5% | 5.79% | 5.80 | 11.74cm |

Full raw/gmrfix/polished table: `GMR-baseline-results.md`'s "Sprint S1" section. Order-of-magnitude
gap between classes holds on every metric, raw or fixed; the corrected split removes the earlier
version's locomotion-class contamination without needing any new detection mechanism — same
detector W2-T3 already validated, just run at scale.

## Not resolved

**Table-I→BVH mapping**: checked the paper website (jaraujo98.github.io/retargeting_matters, via
WebFetch — confirmed no filename/subject mapping present, only generic category names) and GMR's
repo configs/README (nothing there either). Unmapped — no published-number annotation possible on
any clip in this table yet; author contact is the remaining option, Prabin's call, not actioned.
Doesn't block the kinematic table itself, which is fully ours to compute either way.

## New code (all under `scripts/g1/`, branch `gmr-baseline`)

`sprint_batch_retarget.sh` (S1-T1, pre-existing from before this page), `sprint_polish_batch.sh`,
`sprint_eval_batch.py`, `sprint_s1t4_summary.py`, `sprint_reclassify_contacts.py` (new). No core
solver/eval code changed — same discipline as week 1/2, reuse via import/subprocess, never fork.

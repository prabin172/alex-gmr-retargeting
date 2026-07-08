# Measured Metrics

Sources: `outputs/logs/pipeline_native120_*.log` (finalized native-120 batch), `compute_globalopt_metrics.py`, SESSION_HANDOFF. Report the FULL distribution — the shovel numbers are the best case, not the headline.

## Corpus artifact table (2026-07-08) — the honest full-distribution deliverable
Tool: `scripts/eval_artifacts_corpus.py` (reuses solver `_compute_anchors`/penetration filter, **COLL_HOPS=2** = stricter than the legacy `compute_globalopt_metrics.py`'s 4, so coll% here reads HIGHER — it's the honest strict count). Evaluated on `global_opt_contactfirst/` = grounded-equivalent (grounding is a rigid Z shift → joint angles, self-collision, horizontal slip all identical). CSV: `outputs/artifact_table.csv` (regenerate any time). Slip split **foot (horizontal) vs hand (3D)**; slip-time = seconds sliding >1 cm.
- **Joint limits: ZERO hard violations, all 20 clips.** IK never requests an angle the robot lacks.
- **`NECK_Y` is the systematic saturated joint** across the standup family (48–63% of frames pinned at its pitch bound) — head-orientation target over-drives neck pitch into its stop. The one *actionable* limit finding (clamp head target or widen neck-pitch range). Kneeling clips saturate knees/shoulders (expected for the pose); `LEFT_SHOULDER_Z` 91–96% on standupFromKneeling_01 / standupSquatCrouch_01.
- **Foot slip: median max 2.1 cm, p95 1.0 cm.** Clean standups ≤1.9 cm foot (coplanar+floor changes hold the family here). Worst foot slip is on the **falls** — kneelingFall_03 **8.3 cm**, kneelingFall_02 4.8 cm (feet scrabble during collapse; these clips already need hybrid grounding).
- **Hand slip: mostly 0 (no hand contact) or ≤~2 cm.** The standup_02 6.1 cm "outlier" is a HAND (its foot is 1.0 cm) — a push-off, benign.
- **Self-collision: median 8.3%, peak penetration ≤1.9 cm everywhere.** High coll% clips are shallow **grazing** during kneeling (standupFromKneeling_01 72.5%, kneelingFall_03 61%), not deep interpenetration — depth stays sub-2cm. Shovels a clean 0%.
- **Residual foot slip is a smoothness floor, not tunable** — see FOOT_WEIGHT ceiling in [[globalopt]] (sweep 160/1000/4000 → 1.9/1.7/1.7 cm while pen+coll climb). 1.7–1.9 cm is below the mimic tracker's reward std (std=0.3 m; 0.3% reward delta) → retargeting side done.

### Ground contact (grounded output, floor z=0) — the ACTUAL RL blocker, not slip
Added to the same tool (reads `grounded_contactfirst/`). Two distinct problems, both DOWNWARD (penetration, not float):
- **Planted-foot floor penetration: median 2.5 cm, up to 6.5 cm (standup_natural_02), on nearly every clip** (incl. clean shovels 2–3 cm). A *support* foot below the floor = hard physics infeasibility. **Root cause is fixable registration**: constant-contact keys the single Z shift to the planted-sole **median** (`post_process_ground_contactfirst.py --contact-percentile 50`), so ~half the planted frames sit below floor by construction. Lowering the percentile (p10–p25) trades this against float — the one clean knob. See [[grounding]].
- **Swing/tucked-foot clipping: up to 28 cm (standup_side_05), median deepest 8.6 cm, ~42% of frames.** A *lifted* foot's sole plate clips below the floor plane during deep-crouch/get-up phases (the −28 cm frame has BOTH feet non-contact, root z=0.21). A single rigid shift can't fix it; per-frame wanders/bobs. Matches the known kneelingFall −11/−15.8 cm. Needs geometry-aware / hybrid grounding — the long-standing fall/get-up grounding gap.
- **Foot float while planted: tiny — median 0.2 cm, max 2.2 cm.** The feared "2 cm hovering plant → phantom target" is basically absent; grounding errs downward. So the phantom-target risk is a below-floor foot, not a floating one.
- **Reframing**: RL-readiness is gated by ground PENETRATION (planted 2.5 cm one-knob-fixable; swing 8–28 cm = the crouch-grounding problem), NOT by the 1.9 cm XY slip. Shovels/standups clean the planted side with a percentile tweak; falls need real hybrid grounding.

## Finalized native-120 batch (2026-07-05: keep-best-iterate + pins ×4), Stage B, all 18 clips
- **Spikes 0 and peak self-penetration ≤ 0.88 cm on EVERY clip** (keep-best `τ=1cm` gate holds — no deep penetration anywhere).
- Shovels ×6: slip 2.0–3.3 cm, foot-flat ~0.1°, 0% collision. Squat: slip 4.0, 0%. Kneeling-falls: slip 7.1 / 10.7 cm, 0%.
- Standups/get-ups: slip 3.0–9.8 cm, shallow (<1 cm) grazing 4–28%.
- **pins ×4 trade**: standup_side_04 slip 10.4→6.3 cm for 0→0.5 cm (sub-tol) peak penetration; slip-aware keep-best chose the 0.49 cm/6.3 cm iterate over the last-outer 6.59 cm/42.9% one.

## Slip outliers are usually a metric phantom (plant-min-run fix, 2026-07-05)
- `standup_side_05` reported **14.7 cm** — per-effector dig: **entirely right_hand**, **entirely 25 single-frame "plants"** (velocity zero-crossings while the hand lifts off in late standup). IK-vs-median ≤0.4 cm (the plant IS locally stationary); Stage A smooths the hand along its real moving path away from the 1-frame anchor = phantom slip. Filtering planted runs ≥8 fr → **6.8 cm** (real). Feet/left_hand were already fine (≤6.8).
- Fix: `plant_min_run=8` stillness debounce in `_compute_anchors` (a still run <8 fr → moving, not a plant). Standalone standup_side_05 slip **14.7→6.8 cm**, coll/spikes/track unchanged. Full batch re-run pending review (watch `kneelingFall_03` 10.7 cm — may also be phantom).
- **Method lesson**: audit a slip outlier per-effector + IK-vs-median BEFORE blaming the solver. My first two takes (frame-rate; then "structural repositioning floor") were both wrong.

## Historical (pre-finalize, retained for comparison)

## Headline (clean)
- **Velocity spikes: per-frame IK 14–31 per clip → 0 after smoothing, every clip.**
- Straight-knee lock: 26.5% → 0% (shank clamp).
- Contact foot-flat error: 12.7° → 7.7° mean (shovels ~0.1°).
- `standup_side_04` self-penetration: 32.6% frames / 5.2 cm peak → 0 (fullmesh+soft).

## Plant slip (honest distribution)
- Shovels: 1.0–1.5 cm, flat 0.1–0.2°, collisions 0, spikes 0.
- Standups (primitive-era batch): 2.7–3.7 cm, `standup_side_05` outlier 7.9 cm.
- Get-up/kneel on fullmesh Stage B: 2.6, 2.9, 3.7, 3.8, 4.2, **8.0, 8.6, 9.3 cm**.
- **Stage B sometimes INCREASES slip vs Stage A on the hardest clips** (e.g. 0.2 cm warm → 2.3 cm Stage A → 9.3 cm Stage B). The "Stage B reduces drift" narrative holds on average, not per-clip.

## Residual self-collision (fullmesh, honest)
Several get-up/kneel clips retain **11.0 / 19.7 / 20.2 / 20.6 / 32.5%** collision frames, peak penetration ~2.3 cm. "Penetration eliminated" is true on many clips, NOT all — see [[tradeoffs-limits]].

## Faithful-vs-error notes
Crouch-phase foot-flat angles 9.7–12.7° are genuine human tilt, not solver error. Residual get-up "flat-snap" near touchdown is partly faithful, partly the 40° gate's hard boundary.

Metric definitions/tooling: velocity spikes, per-frame max/p95 Δq, collision-frame %, peak penetration, track error, plant slip, foot-flat error — all from `compute_globalopt_metrics.py` (point it at a global_opt dir + the model).

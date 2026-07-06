# Measured Metrics

Sources: `outputs/logs/pipeline_native120_*.log` (finalized native-120 batch), `compute_globalopt_metrics.py`, SESSION_HANDOFF. Report the FULL distribution — the shovel numbers are the best case, not the headline.

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

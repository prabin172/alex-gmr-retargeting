# hierarchical-v1 H2 gate: Stage-3 hard-tier contact/floor constraints (2026-07-11)

H2 deliverable — `--hard-tier` in `scripts/solve_fbx_canonical_alex_contactfirst.py` (forces the
dormant `--hierarchical` task-priority solve, feet-only hard, hands untouched — see
`planLog.md` H2 for the full build story, including the 44-metre `--floor-hard` divergence found
and decoupled during this milestone). This page is the real-pipeline 20-clip corpus verification.

Run: `S3_HARD_TIER=on RENDER=0 bash retargetingPipeline.sh` with fresh output dirs
(`outputs/{contactfirst,global_opt_contactfirst,grounded_contactfirst}_h2`, avoiding Stage 3's
skip-if-exists cache). `ok=20/20 fail=0`. Compared against the frozen baseline
(`wiki/experiments/phasic-v2-M2-T2.2-gate.csv`, the same table every phasic-v2/H-gate compares
against). Raw CSV: `wiki/experiments/hierarchical-v1-H2-gate.csv`.

## Corpus deltas vs frozen baseline (hard-tier minus baseline)

| metric | median Δ | mean Δ | max Δ (worst) | min Δ (best) |
|---|---|---|---|---|
| foot slip (`ftSlip`, final shipped output) | **0.000cm** | **0.000cm** | 0.000cm | 0.000cm |
| self-collision % frames (`coll%`) | **0.000** | **0.000** | 0.000 | 0.000 |
| self-penetration peak (`selfPen`) | **0.000cm** | **0.000cm** | 0.000cm | 0.000cm |
| planted-foot floor pen (`plPen`) | +0.10cm | +0.60cm | +8.72cm (`standupSquatCrouch_01`) | −3.84cm (`standupFromKneeling_01`) |

**`ftSlip`, `coll%`, and `selfPen` are bit-identical to the baseline on every single one of the 20
clips** — not just similar, exactly equal to 2 decimal places, clip by clip. Only `plPen` (floor
penetration) moves at all, and the net effect is slightly negative (mean +0.60cm worse), with one
severe regression (`standupSquatCrouch_01` +8.72cm, `standup_natural_02` +8.09cm,
`luigi_standSupine_08` +3.99cm) partially offset by three real improvements
(`standupFromKneeling_01` −3.85cm, `standup_side_04` −2.87cm, `standup_slideHandsBack_03` −3.05cm).

## Why ftSlip/coll%/selfPen don't move at all

Stage 4's GlobalOPT Stage-B contact QP (foot-weight 160, hand-weight 32, always-on soft
self-collision) re-solves the WHOLE trajectory downstream of Stage 3 and already drives planted-foot
slip and self-collision to their final values regardless of what mechanism Stage 3 used to hold
contacts — a heavily-weighted soft pin or a hard task-priority tier land in the same place once
Stage 4 is done. This is exactly what `--hierarchical`'s ORIGINAL (pre-this-session) help text
already said before this milestone even started: "hold-weight 10 + GlobalOPT Stage B reaches lower
plant slip... with one config for all actions" — H2's corpus run re-confirms this with fresh
numbers on the current phasic-v2 architecture rather than overturning it.

## Stage-3-level diagnostic (before Stage 4 smooths it away): hard tier does NOT cleanly self-converge

Per-clip `floor_pen_cm`/`hold_slip_cm` (from `solve_frame_position_ik`'s new `diag_out`, measuring
the hard tier's OWN achieved-vs-target gap, Stage-3 output only): several clips show large
`hold_slip` even though the task is nominally "hard" (level 1, uncontested at that tier) —
`standup_natural_02` 41.10cm, `standup_side_04` 22.48cm, `luigi_standSupine_08` 22.74cm,
`standup_02` 13.02cm. This means even the narrower, verified-non-blowup form of the hard tier
doesn't actually achieve near-zero slip WITHIN Stage 3 itself on every clip — likely double-support
frames where both feet's hold + flat + yaw rows compete inside the SAME undifferentiated level-1
system (never blows up like mixing in floor did, but doesn't cleanly converge either). Full numbers:
`planLog.md` H2 (grepped from `outputs/logs/pipeline_h2_hardtier_corpus.log`).

## Verdict: H2 provides no measurable end-to-end benefit, at real (if now-contained) risk

- **Zero benefit** on the metrics `--hard-tier` was meant to improve (slip, self-collision) — Stage
  4 already delivers the "robots cannot slip" goal at the shipped-output level today, with or
  without Stage-3 hard-tier.
- **Slightly negative** on floor penetration on average, with one clip regressing badly
  (`standupSquatCrouch_01` +8.72cm).
- **Does not itself cleanly achieve zero-slip** even where it's "safe" (large Stage-3-level
  `hold_slip` on several clips) — the narrower form avoids catastrophic divergence but doesn't
  deliver the intended guarantee either.
- **`--floor-hard` (the other half of H2's original design) is confirmed broken** (44m divergence,
  `planLog.md` H2) and would need a structural redesign (nested 3-tier priority) to attempt safely,
  not attempted here.

**Recommendation: do not ship `--hard-tier`/`--floor-hard` as pipeline defaults.** This reconfirms,
with fresh 2026-07-11 data on the current architecture, the same conclusion the original
(pre-phasic-v2) `--hierarchical` retirement reached: the shipped single-level all-soft-weights path
plus Stage 4's contact QP is the better-performing design for this solver. `H3` (Stage-4 hard
anchors) was explicitly conditional on H2 showing improvement per `plan.md` — given this result, not
started; flagged to Prabin for a decision on whether hierarchical-v1 continues in a different form
or closes here.

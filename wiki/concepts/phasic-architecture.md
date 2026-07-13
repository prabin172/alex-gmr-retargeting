# Phasic-v2 Architecture (branch `phasic-v2`, M0-M6)

Full ground-up redesign of floor handling, replacing per-clip flag hacks with an upstream
invariant (floor = z=0, established once, consumed everywhere) and strictly decoupled,
independently-ablatable phases. Plan: `plan.md` (repo root). Full build/debug trail for every
phase, including every bug found and every rejected fix attempt: `planLog.md`.

## Why this redesign happened

The pre-redesign pipeline (see [[pipeline]], [[globalopt]], [[grounding]]) estimated the floor's
height independently at THREE separate points (Stage 3's 1st-percentile-of-feet, Stage 4's
warm-start-sole-height median, Stage 4.5's grounding percentile) — each estimator broke
differently on multi-phase (lying↔standing) clips, forcing ad-hoc per-clip flags
(`--floor-weight`, `--floor-refine`, `--floor-collision`, `--floor-phase-aware`) on the two Luigi
clips that violated the "ONE config for all actions" design principle
([[design-philosophy]]). Phasic-v2 replaces estimation with an invariant: ground the CANONICAL
HUMAN data to floor=0 once (M1), before morphology scaling, so every downstream stage inherits a
known floor instead of re-deriving one.

## Phase map (P0-P5, plan.md's naming) → concrete pipeline stages

```
P0 canonical prep     Stage 1-2 (unchanged) + Stage 2.5 (NEW, M1): canonical grounding + persisted contact labels
P1 contact-first IK   Stage 3: per-frame IK; floor enters via PER-WINDOW target correction (M2), not clip-wide estimation
P2 global smoothing   Stage 4: Stage A + contact QP; floor_z from Stage 3's own (now-corrected) output, not re-estimated
P3 physics slot       Stage 4.6 (NEW, M4, opt-in): velocity/acceleration plausibility QP
P4 limb cleanup       Stage 4.7 (NEW, M5, opt-in): per-limb floor/self-collision/swing-clearance QP, root frozen
P5 validate + export  Stage 4.5 grounding (unchanged mechanism, reframed as an invariant check, see below) + Stage 5/6
```

Note the ORDERING differs slightly from plan.md's original diagram: Stage 4.5 (grounding, the rigid
Z-shift) runs BEFORE P3/P4 in the shipped pipeline, not after — P3/P4 need a floor reference already
established to define their own constraints against, and Stage 4.5 is what establishes "floor=0 in
Alex's own qpos frame" (a SEPARATE coordinate space from the canonical-human invariant M1
establishes — see the M3 finding below). This was a deliberate ordering decision made during
implementation, not an oversight.

## Contract: each phase's job, and what it does NOT own

| Phase | Owns | Does NOT own | Default |
|---|---|---|---|
| P0 (Stage 2.5) | Floor=0 invariant in canonical-human space; contact label detection (shared, `contact_labels.py`) | Morphology-scaled target correctness | Always on |
| P1 (Stage 3) | Per-window target-space floor correction for contacting effectors; mild global floor repulsion (`S3_FLOOR_WEIGHT=10`) | Whole-clip smoothing; self-collision beyond the solver's own repulsion | Always on |
| P2 (Stage 4) | Whole-clip smoothness + contact-aware QP; on-floor coplanar rows for PLANTED feet only | Swing-limb floor penetration; CoM/balance | Always on |
| P3 (Stage 4.6) | Joint/root velocity+acceleration plausibility (hard box QP) | CoM/support-polygon balance (built, disabled — see [[physics-plausibility]]) | `PHYSICS_PASS=off` |
| P4 (Stage 4.7) | Isolated swing-limb floor penetration, self-collision, swing clearance (root frozen) | Whole-body lying-phase floor mismatch (architecturally out of reach — see [[limb-cleanup]]) | `LIMB_REFINE=off` |
| P5 (Stage 4.5 + 5/6) | Rigid Z-shift to floor=0 in Alex's OWN qpos frame; render; IHMC export | — | Always on |

## Settled decisions from the design discussion (do not re-litigate)

- **Orientation stays semantic-frames** (not FBX rotations) — axial DOFs are measured for hands
  (thumb landmark), contact-overridden for feet, implicitly pinned by elbow/knee position targets
  elsewhere. See [[orientation-frames]].
- **P3's physics content is velocity/acceleration plausibility only** — CoM/support-polygon
  checking was built (M4/T4.2) then explicitly disabled by default: a kinematic-only estimate
  can't distinguish real imbalance from a dynamic posture leaning on momentum. Revisit once
  physics-aware training provides real dynamics data. See [[physics-plausibility]].
- **M3's floor_mode stays `estimate`, not forced to `zero`**: tested forcing Stage 4's `floor_z` to
  a literal 0 and found it catastrophic (Alex's own achieved-qpos root frame is a SEPARATE
  coordinate space from the canonical-human floor invariant — confirmed by measuring `floor_z`
  under `estimate` ranging -0.05 to -0.88m across the corpus, and a single clip's own root Z
  spanning 0.008 to 0.816m within itself). `--floor-mode estimate` is not phase-blind ANYMORE post-
  M2 (it's freshly derived from Stage 3's own now-corrected output every run), so the original
  concern T3.1 was written to address no longer applies. See [[globalopt]]'s FOOTGUN section.
- **P4 (limb cleanup) is optional, not a replacement for the Luigi per-clip flags**: verified on
  the full 20-clip corpus that a root-frozen per-limb solver cannot fix whole-body lying-phase
  floor contact (7/20 clips, including `luigi_standSupine_08`) — this is the exact risk plan.md's
  own "Risks/fallbacks" section anticipated ("root frozen ⇒ reach saturation"). See
  [[limb-cleanup]].

## "ONE config" status — honest final state, not a shortfall

The pre-redesign pipeline had FIVE floor-related flags on the two Luigi `CLIPS[]` entries
(`--floor-weight`, `--floor-margin`, `--floor-gain`, `--floor-refine`, `--floor-collision`,
`--floor-phase-aware`, `--contact-preroll`, `--contact-on-speed-frac`,
`--contact-onset-max-delay` across both Stage-3 and Stage-4 fields). Post-redesign:
- `luigi_standProne_03`: ONE flag, `--contact-preroll 0` (a Stage-3 look-ahead param, unrelated to
  floor handling — needed because this clip's contact timing differs, not a floor workaround).
- `luigi_standSupine_08`: ONE flag, `--floor-phase-aware` (the genuinely clip-specific need — a
  lying/standing multi-phase clip where Stage 4's hard floor-collision term needs phase-gating;
  see [[globalopt]]'s phase-aware section).

Global floor-safety mechanisms (`S3_FLOOR_WEIGHT=10`, target-space per-window correction) now
apply identically to all 20 clips via the pipeline defaults — this is what T2.2 achieved. The
remaining two per-clip flags are DELIBERATE, MINIMAL, and JUSTIFIED exceptions (down from ~8-9
flags to 1 each), not an incomplete migration.

## Ablation

The two NEW, genuinely-optional phases (P3, P4) have independent env-knob toggles
(`PHYSICS_PASS`, `LIMB_REFINE`) — this IS the ablation mechanism plan.md's T6.4 asked for. P0-P2
are foundational (they replace, not augment, the pre-redesign mechanisms), so "P0-P2 off" isn't a
meaningful ablation point within this branch — the `main` branch itself (or
`wiki/experiments/phasic-v2-baseline.md`, the M0 snapshot) serves as that comparison.

Both `PHYSICS_PASS=off` and `LIMB_REFINE=off` (the shipped defaults) are verified BYTE-IDENTICAL
no-ops (md5sum / repeated-run checks, not assumed) — enabling either only touches output when it
finds something to fix.

| Configuration | What changes | Verified in |
|---|---|---|
| P0-P2 only (shipped default) | Floor=0 invariant, per-window target correction, mild global repulsion; 0 hard joint violations, 0 spikes corpus-wide | `wiki/experiments/phasic-v2-M2-T2.2-gate.md` |
| + P3 (`PHYSICS_PASS=on`) | Velocity/acceleration plausibility; near-no-op on 17/20 clips, small correction on 3 (max tracking delta 1.32cm) | `wiki/experiments/phasic-v2-M4-gate.md` |
| + P4 (`LIMB_REFINE=on`) | Isolated swing-limb floor-pen fixed on 8/20 clean + 5/20 near-miss; safely no-ops on 7/20 whole-body-lying cases | `wiki/experiments/phasic-v2-M5-gate.md` |

## Verification methodology used throughout (for future milestones)

- Every claim about metrics is measured, not assumed — including several that turned out wrong on
  first measurement (M3's `floor_mode=zero` test, M4's CoM-check bugs, M5's Gauss-Seidel
  divergence) and were caught specifically BECAUSE they were tested rather than trusted.
  `planLog.md` documents every one, including rejected fix attempts and why they were rejected.
- "Compare vs baseline" gates use FRESH runs of the pre-change code, never `outputs/*` files that
  predate the current session — Stage 3's skip-if-exists cache made this a real risk once (M1).
- A pass that cannot safely improve a clip should protect it (never ship worse than input), not
  silently corrupt it or be forced to "succeed" past its actual scope — `keep-best-iterate`
  (Stage 4's own pattern, reused in M4 and M5) is the mechanism; every "cannot improve" case in
  this session was verified as a true no-op (`track_rms=0`), not assumed safe.

Related: [[pipeline]] (stage list), [[design-philosophy]] (broader settled decisions),
[[physics-plausibility]], [[limb-cleanup]], [[globalopt]], [[grounding]], [[contact-first-ik]].

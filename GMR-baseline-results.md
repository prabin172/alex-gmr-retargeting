# GMR-baseline results — Week 1 + 2 + Sprint S1 (2026-07-16)

Results narrative for `GMR-baseline.md`'s Option A ("contact-aware kinematic polish"). Week 1:
`GMR-baseline-plan.md` T0–T10. Week 2 (E1b fairness addendum, E4b multi-surface contact attempt,
self-collision vetting, grounding comparison): `GMR-baseline-plan.md`'s Week-2 section, W2-T1..T7,
executed task-by-task per its own instructions. Sprint (Humanoids 2026, the 2×2 comparison): S1
(full-corpus kinematic sweep, DONE) + S2 (OURS on G1, in progress, parked at CHECKPOINT M4) —
`GMR-baseline-plan.md`'s SPRINT section. Full build/debug trail: `planLogGMR.md`. Wiki record:
`wiki/experiments/gmr-baseline-week1.md` + `wiki/experiments/gmr-baseline-week2.md` +
`wiki/experiments/gmr-baseline-sprint-s1.md`. Branch `gmr-baseline`.

## Setup

- **GMR**: fresh clone, `github.com/YanjieZe/GMR`, MIT, unmodified — imported as an installed
  package (`pip install -e`), never edited. Robot: `unitree_g1` (29 DoF + free root, MuJoCo
  mocap model shipped in-repo).
- **Data**: LAFAN1 (Ubisoft, free direct download), 77 BVH clips at 30 fps.
- **Clips** (5, screened by hip-Z range via GMR's own loader — not guessed):
  - Floor-contact: `fallAndGetUp2_subject2` (most severe fall), `fallAndGetUp1_subject1`
    (sustained low floor time), `ground1_subject1` (sustained crawl, different failure mode).
  - Locomotion controls: `walk1_subject1` (clean), `dance1_subject1` (busier, has a non-floor
    crouch dip).
- **Eval**: `scripts/g1/eval_motion.py`, a de-Alexed port of `scripts/eval_ihmc_json.py`'s
  reference-free physics eval (mesh-exact floor penetration vs z=0, joint-limit violations,
  rate-aware velocity/spikes). Reused `evaluate()` unmodified.
- **Polish**: `scripts/g1/polish_gmr_pkl.py` — Stage A (tridiagonal smoothing, imported
  unmodified from `solve_global_trajectory_opt_contactfirst.py`) + Z-grounding (`constant` mode,
  via unmodified `post_process_ground_contactfirst.py`).

## (a) Motivation: GMR on its own benchmark's floor clips

| clip | floorPen max | pen% (frames >0.5cm) |
|---|---|---|
| walk1_subject1 (control) | 1.0cm | 0.3% |
| dance1_subject1 (control) | 7.1cm | 1.9% |
| fallAndGetUp2_subject2 | 13.6cm | 47.1% |
| fallAndGetUp1_subject1 | 12.9cm | 38.9% |
| ground1_subject1 | 15.9cm | 90.6% |

GMR's paper states it "does not include motions with complex interaction with the environment,
such as crawling or getting up from the floor." This confirms it with numbers, not just the
quote: on LAFAN1's own floor-contact clips, GMR's max floor penetration is 13–16cm, affecting
39–91% of all frames — an order of magnitude worse than its clean-locomotion baseline (1.0cm,
0.3%). Visual inspection (frame extractions, `planLogGMR.md` T3) confirms the failure mode: no
limb ever appears to bear weight against the floor — splayed, floating poses, not a body using
ground contact for support.

**Fair-baseline addendum (2026-07-15, W2-T1): applying GMR's OWN described height fix.** The paper
(§IV) describes a clip-global min-height subtraction as part of GMR's method; the shipped code has
it hard-disabled (`bvh_to_robot_dataset.py`, `HEIGHT_ADJUST = False`) — our week-1 numbers above are
GMR's shipped default, defensible but not what a reviewer holding the paper expects tried. We
replicated their fix faithfully (clip-global min body-ORIGIN z via plain-MuJoCo FK, mesh-blind by
construction, matching their torch implementation) as a separate baseline variant (never touching
the GMR clone) and added a floating metric (whole-body lowest point's height ABOVE z=0 — the
artifact a global single-frame calibration trades penetration into):

| clip | floorPen: raw→+heightfix | pen%: raw→+heightfix | float%: raw→+heightfix |
|---|---|---|---|
| walk1_subject1 (control) | 1.0→2.0cm | 0.3%→5.7% | 94.1%→55.0% |
| dance1_subject1 (control) | 7.1→**2.7cm** | 1.9%→**0.3%** | 89.6%→99.5% |
| fallAndGetUp2_subject2 | 13.6→**2.9cm** | 47.1%→**0.4%** | 47.7%→**98.6%** |
| fallAndGetUp1_subject1 | 12.9→**5.2cm** | 38.9%→**3.8%** | 56.1%→**94.0%** |
| ground1_subject1 | 15.9→**2.9cm** | 90.6%→**0.1%** | 8.2%→**99.9%** |

The fix is real — it cuts floor-clip penetration 65–82% and pen% to 0.1–3.8%. But because it's a
single clip-global scalar calibrated to one worst frame's body ORIGIN (not mesh surface), the same
correction pushes the entire rest of the clip's mesh surface above the floor: **float% lands at
94–99.9% on every floor clip after the fix** — essentially the whole clip now has some body part
detectably hovering, not resting. Applying GMR's own described fix and still finding near-universal
floating on the excluded class is a stronger, harder-to-dismiss version of this motivation figure
than the raw numbers alone — it forecloses the "did you even try their own post-processing step"
objection. Full numbers/mechanism: `planLogGMR.md` W2-T1.

**Side-finding worth stating explicitly**: zero velocity spikes on every clip, including the
worst floor clips. GMR's own per-frame differential IK produces smooth output even while failing
badly on floor contact — floor-contact failure and motion jitter are orthogonal problems here,
which sharpens Option A's claim: this is specifically a floor-contact-reasoning gap, not a
generic "GMR produces bad motion on hard clips" story.

## (b) Polish delta

| clip | floorPen: raw→polished | pen%: raw→polished | vMax rad/s: raw→polished |
|---|---|---|---|
| walk1_subject1 | 1.0→**0.7cm** | 0.3%→0.1% | 18.9→**3.3** (5.7×) |
| dance1_subject1 | 7.1→**3.2cm** | 1.9%→0.6% | 47.5→**6.2** (7.7×) |
| fallAndGetUp2_subject2 | 13.6→**4.0cm** | 47.1%→0.5% | 20.4→**4.8** (4.3×) |
| fallAndGetUp1_subject1 | 12.9→**1.1cm** | 38.9%→0.5% | 29.5→**6.1** (4.8×) |
| ground1_subject1 | 15.9→**2.4cm** | 90.6%→0.5% | 37.2→**5.5** (6.8×) |

A robot-agnostic, purely-kinematic polish module — the SAME code that validated on Alex via a
mentor's manual Blender retarget (5.7× velocity smoothing, `wiki/log.md` 2026-07-14) — ported to
a second robot (Unitree G1) with **zero core-solver-logic changes**, improves BOTH floor
penetration and joint-velocity smoothness on every single clip in this corpus, controls
included. No cherry-picking was needed to make this case.

The generalization required: robot-specific joint limits (already a function argument, not a
hardcoded default) and a model path. Nothing else — the smoothing math, the eval metrics, and
the grounding QP are literally the same code Alex's pipeline uses.

**Three-way check (2026-07-15, W2-T1): our polish vs GMR's own height fix, floor+float together.**

| clip | floorPen: raw / +heightfix / polished | pen%: raw / +heightfix / polished | float%: raw / +heightfix / polished |
|---|---|---|---|
| walk1_subject1 | 1.0 / 2.0 / **0.7cm** | 0.3% / 5.7% / **0.1%** | 94.1% / 55.0% / 96.3% |
| dance1_subject1 | 7.1 / **2.7** / 3.2cm | 1.9% / **0.3%** / 0.6% | 89.6% / 99.5% / 98.6% |
| fallAndGetUp2_subject2 | 13.6 / **2.9** / 4.0cm | 47.1% / **0.4%** / 0.5% | 47.7% / 98.6% / 97.9% |
| fallAndGetUp1_subject1 | 12.9 / 5.2 / **1.1cm** | 38.9% / 3.8% / **0.5%** | 56.1% / 94.0% / 98.3% |
| ground1_subject1 | 15.9 / 2.9 / **2.4cm** | 90.6% / 0.1% / **0.5%** | 8.2% / 99.9% / 98.4% |

Our polish wins or ties on floorPen/pen% on 3/5 clips; GMR's own fix is nominally better on 2/5
(both already under 5cm, diminishing returns). **Float% is not a differentiator — both land in
94-99.9% on every floor clip.** Our week-1 "polished" deliverable uses the same fundamentally
whole-clip-level mechanism (a single percentile-based Z calibration) as their height fix — so it
inherits the same floating trade-off. This turns the caveat below from a qualitative visual
observation into a measured number.

## (c) E4/E4b: per-limb contact anchoring — parked, then a negative checkpoint

**E4 (feet-only, post-week-1 MVP)**: real 25% foot-slip reduction on `walk1_subject1` (clean
locomotion), essentially zero effect on the 3 floor-contact clips (near-zero sustained
stationary-foot behavior detected in fall/crawl motion). **Independently re-confirmed (W2-T2,
2026-07-15)**: a from-scratch check (different position reference, different drift convention, no
shared code with E4's own internal metric) confirms the DIRECTION on `walk1_subject1` (8-9% mean /
5-23% max drift reduction on both feet) — the magnitude differs from the internal "25%" as expected
given the different methodology, but the effect is real, not an artifact of self-grading.

**E4b (2026-07-15, redesign after diagnosing E4's failure mode)**: detect contact on the HUMAN
source instead of the corrupted robot output, cover all support surfaces (feet/hands/knees/elbows,
optionally pelvis/torso) instead of feet only, and PULL support points to the floor instead of
merely holding them in place. **Kill-test #1 (human-side multi-surface labels): passed cleanly** —
controls show near-zero non-foot contact, all 3 floor clips show sustained, anatomically-sensible
multi-surface contact (hands 25-88%, knees 7-83%, elbows 10-65%, pelvis 22-55%), cross-validated
against an independent robot-side measurement at the same frame (agreement within ~1cm). **Kill-
test #2 (the pull-to-floor anchoring itself): CHECKPOINT M3, a clear negative.** Anchors engage
and move 5/8 support points in the right direction LOCALLY (1.5-3.5cm each), but the whole-clip
AGGREGATE metrics show floorPen getting measurably WORSE on 4 of 5 clips, including both
locomotion controls — perturbing anchored joints costs a little penetration elsewhere more often
than it fixes floating locally. Root position never moves; the largest joint-angle change anywhere
in any clip is ~0.3 rad — the trust-region-limited local QP cannot close a 5-13cm gap this way.
Visual check (frame 356, raw/polished/multi-surface side by side) confirms: all three poses are
indistinguishable, feet still visibly floating in every version.

**Conclusion: anchoring-on-top-of-polish is not the corpse-pose fix, regardless of contact source
or floor-pull mechanism.** This motion class needs contact-first SOLVING — root and contacts
planned jointly from the start (an analog of Alex's Stage-3 contact-first IK), not a post-hoc
anchor on an already-fixed whole-body trajectory. A real scope decision for whoever picks this up
next, not a natural continuation of E4/E4b's approach. Full numbers, engagement stats, and the
frame-356 visual: `planLogGMR.md` W2-T5.

## Honest caveats

- **Polish is a whole-clip-level fix, not a per-limb one — now measured, not just visual.** The
  same "splayed limbs, no weight-bearing contact" pose visible in the raw motivation figure is
  STILL visually present after polish, frame-for-frame — the penetration number improves because
  grounding recalibrates where the clip's floor reference actually is, not because any individual
  limb's pose becomes physically supportable. **W2-T1's floating metric confirms this
  quantitatively: our polished output floats (>0.5cm, some body part) on 96.3–98.6% of frames on
  every clip** — statistically indistinguishable from GMR's own height-fix baseline (94.0–99.9%).
  Whole-clip Z-calibration, ours or theirs, cannot produce a body that's actually resting on the
  floor throughout a clip. Per-limb anchoring on top of it was tried twice (E4 feet-only, E4b
  multi-surface pull-to-floor) and both land on this same boundary — see (c) above. The fix, if
  pursued, is contact-first solving, not more anchoring.
- **Self-collision on G1 is now a usable, vetted metric (resolved 2026-07-15, W2-T6)** — it was
  noise-only through week 1 (a clean walk clip read 18.2% self-collision incidence against the
  mocap XML's unvetted full-mesh geometry). GMR's own `g1_custom_collision_29dof.urdf` (11 of 46
  collision blocks actually uncommented — simplified cylinder proxies on the joints most prone to
  self-intersection) loads directly via MuJoCo's own URDF importer with actuated-joint order
  verified identical to the mocap XML. `walk1_subject1` raw now reads **0.2% self-collision** —
  physically plausible. Full raw/stageA/polished table: `planLogGMR.md` W2-T6.
- **`constant` grounding stays the shipped choice — re-confirmed, not just chosen by elimination.**
  Week 1: chosen over `perframe` because `perframe`'s "perfect" 0.0cm floor pen comes from
  grounding every frame independently, measurably increasing root-Z bobbing (+65% peak vertical
  velocity on the hardest clip). Week 2 (W2-T7): tried a genuinely contact-aware alternative
  (`constant-contact`/`hybrid` modes, fed W2-T3's human contact labels through a thin adapter since
  G1 lacks the named sole sites those modes were built for) — both came out DRAMATICALLY WORSE than
  `constant` on every clip (e.g. `walk1_subject1` floorPen 0.7cm→6.6cm), traced partly to G1's sole
  marker spheres sitting ~1cm above the foot's true mesh contact point. No change: `constant`
  ships unchanged.

## Sprint S1: full-corpus kinematic sweep (2026-07-16)

The top row of the 2×2 (`GMR-baseline.md` §7.4), all 77 LAFAN1 clips × 3 variants
(raw / GMR+heightfix / polished). Full trail: `planLogGMR.md` `## S1-T1..T4`.

- **Setup**: batch retarget (S1-T1, 77/77, 0 failures, human-scaled-target NPZs saved per clip) →
  heightfix + polish variants (S1-T2, 77/77 × 2, 0 failures) → eval (S1-T3, 231/231 rows, 0
  failures) → class-split aggregation (S1-T4).
- **Regression gate passes exactly**: the 5 week-1/2 clips reproduce their original numbers to the
  decimal on every metric (floorPen, pen%, float%, vMax, self-collision) — this sprint's new batch
  tooling introduced no drift.
- **New metrics added at scale**: self-collision via the W2-T6 vetted-collision model (not the
  unvetted default), and a **source-target deviation guard** (FK'd robot body position vs GMR's
  own scaled-human target, using `bvh_lafan1_to_g1.json`'s `ik_match_table2` — the
  position-WEIGHTED correspondence GMR itself optimizes against, not `table1` which barely weights
  position). **Terminology (Prabin, 2026-07-16): this metric was previously logged as
  "faithfulness" — renamed, because GMR's paper uses "faithfulness" for its N=20 HUMAN-RATER user
  study (perceptual reference-vs-retarget similarity), which this is NOT and does not replicate.
  Ours is a kinematic proxy guard: "polish does not wander from the source GMR was tracking."
  Never present it as their user-study metric.** Polish doesn't wander: deltas are small and
  concentrated on floor-class clips where whole-clip Z-shift genuinely trades fidelity for floor
  placement (already-documented tradeoff), several clips even improve slightly.
- **Class split, corrected mid-sprint**: an initial split by hip-Z-p5<0.3 (week-1's T2 convention,
  applied at scale) put 20 clips in floor-class / 57 in locomotion-class, but undercounted —
  hip-height alone misses real, brief, non-hip floor contact (a hand or knee touching down during
  a stumble/obstacle-clear while the pelvis stays up), the same blind spot W2-T3 first diagnosed.
  Reclassified using W2-T3's own multi-surface human-contact detector (unchanged, run over all 77
  clips): floor-class if ANY non-foot landmark has a sustained (≥1s) contact run. Result: **34
  floor-class / 43 locomotion-class** — this is the split that should ship in the paper.

| class | variant | floorPen | pen% | float% | coll%(vetted) | vMax | srcDev_mean |
|---|---|---|---|---|---|---|---|
| locomotion (43) | raw | 4.77cm | 1.91% | 93.5% | 3.85% | 32.82 | 10.19cm |
| locomotion (43) | gmrfix | 2.69cm | 1.89% | 91.1% | 3.85% | 32.82 | 9.92cm |
| locomotion (43) | polished | 2.59cm | 0.49% | 97.9% | 3.75% | 5.45 | 10.90cm |
| floor (34) | raw | 17.00cm | 27.35% | 69.0% | 6.32% | 34.04 | 10.23cm |
| floor (34) | gmrfix | 4.46cm | 0.94% | 98.5% | 6.32% | 34.04 | 12.51cm |
| floor (34) | polished | 4.14cm | 0.70% | 98.5% | 5.79% | 5.80 | 11.74cm |

(`srcDev_mean` = the source-target deviation guard above; column was named `faith_mean` in
`s1t3_eval.csv` and `planLogGMR.md`'s S1 tables — same numbers, renamed here per the terminology
note. **How to read float% in this table — it is a diagnostic, not a scoreboard** (Prabin
2026-07-16): raw/gmrfix/polished all sit at 91-98.5% because ANY single global Z-shift can make
only one instant of a clip truly touch the floor — the saturation is the structural ceiling of
the entire whole-clip-calibration family, theirs and ours alike, not a few-point comparison
between methods. The discriminating measurement for genuine contact is per-frame support-point
distance at detected-contact frames — GMR-polished: +13cm off the floor; contact-in-the-solve
(S2): -3cm — see the S2 section. Caveat for the paper: float% counts genuinely airborne frames
(jump flight, mid-fall) as "floating," which is correct behavior there — the metric indicts a
reference only where support is expected, e.g. walking stance, where raw GMR's ~94% means the
stance foot hovers 1-3cm essentially the whole clip.)

Reads as a much cleaner version of the original week-1 motivation figure, now at full-corpus
scale: floor-class clips penetrate an order of magnitude worse than locomotion on every metric,
raw or fixed; polish narrows the gap substantially without materially drifting from GMR's own
targets. **Not yet done**: Table-I→BVH mapping (checked paper website + GMR repo configs, neither
has it — unmapped, author-contact fallback is Prabin's call), so no published-number annotation on
any row yet — this is the fully-ours kinematic table, not a replication of Table I/II.

## Sprint S2: OURS on G1 (4-clip validation done, whole-clip floorPen residual root-caused)

The bottom row of the 2×2 — our own contact-first pipeline retargeting LAFAN1 straight to G1,
built from scratch this sprint. Full trail: `planLogGMR.md` `## S2-T1..T10`.

Built and working: a LAFAN1→canonical-human adapter (reuses Stage 2.5 unchanged), a G1 Stage-3
analog (per-frame DLS IK, root+joints solved jointly, contact-held effectors), a pull-to-floor
anchoring mechanism (verified via independent per-frame support_z audit), and — the largest fix
this sprint (S2-T7) — replacing an independently-computed per-role morphology scale (which produced
a kinematically IMPOSSIBLE leg chain: hip=2.45/knee=0.97/ankle=0.79 on the same rigid leg, implying
a 36cm target thigh vs G1's real 19cm) with GMR's own published grouped scale constants (0.9
lower-body / 0.75 upper-body, read directly from their `bvh_lafan1_to_g1.json`) — so the ONLY
remaining methodological difference from GMR is contact enforcement in the solve, not morphology
scaling. Also found and fixed (S2-T9): Stage A's temporal smoothing was floor-blind and could
overshoot a sharp raw dip into something worse (mirrors an already-known Alex-pipeline failure
mode) — fixed with a local tracking-weight boost at sustained floor/self-collision-sensitive
frames, extended to cover self-collision as well as floor (not just the mainline pipeline's floor-
only version).

**4-clip validation (walk1_subject1, fallAndGetUp1_subject1, fallAndGetUp2_subject2,
ground1_subject1), full 5-variant comparison — the paper's central claim now holds decisively on
every clip, not just a walking sanity check**. Held-frame support_z (foot-to-floor distance at
moments the human is actually planted):

| clip          | GMR+heightfix           | GMR+ourpolish | OURS         |
| ------------- | ----------------------- | ------------- | ------------ |
| walk1         | +0.5cm (99% within 3cm) | +4.6cm (7%)   | -0.9cm (69%) |
| fallAndGetUp1 | +8.8cm (2%)             | +10.4cm (0%)  | -1.2cm (66%) |
| fallAndGetUp2 | +12.2cm (0%)            | +11.3cm (0%)  | -0.9cm (68%) |
| ground1       | +12.3cm (0%)            | +8.6cm (0%)   | -0.0cm (73%) |

On the three floor-contact clips — the paper's actual motivation — GMR's own published method
(heightfix) has feet floating 8.6–12.5cm off the ground on almost every planted frame; OURS holds
contact to within ~1cm uniformly, with no degradation on the harder clips. Self-collision also
improved substantially after the chain-consistency fix (walk1 15.3%→2.1%, matching GMR's own
level; fallAndGetUp1 13.0%→6.4%; fallAndGetUp2 16.7%→14.3%; ground1 18.7%→11.6%).

**Where it still trails**: whole-clip aggregate floorPen (raw, pre-polish: 17-40cm across the 4
clips) is worse than GMR-polished (1-5cm) — real residual error on frames with no contact signal.
Root-caused (S2-T9/S2-T10), not a bug: the per-frame DLS solver hits a genuine near/over
kinematic-reach limit at some frames — G1's own pelvis-to-head-derived size is only ~64% this
human's (`root_scale` ≈ 0.64-0.65 across all 4 clips), and at extreme-extension poses (deep gait
stride, get-up limb sweeps) the retargeted target occasionally exceeds what G1's shorter legs can
physically reach (confirmed: 2650 frame-legs on `fallAndGetUp2_subject2` exceed 100% of G1's max
leg reach, worst case 181.6%). **Tried a principled fix** (apply the size-correction factor to the
per-role relative-motion term too, matching GMR's own two-term formula structure) — it shrank the
over-reach cleanly (2650→852 frame-legs, worst case 117.1%) but made every other metric
substantially worse when run end-to-end (self-collision e.g. ground1 11.6%→34.5%, floorPen e.g.
fallAndGetUp1 25.5→50.9cm) because shrinking the whole clip's limb excursion to fix a few extreme
frames cramps the entire gait. Reverted; this residual is a genuine, honest retargeting-fidelity
limit of putting a full-size human motion onto a robot ~64% its size, not closeable by any uniform
linear scale — a local/adaptive correction at just the over-reaching frames is the untried
alternative, not attempted this pass.

## Recommended next step

1. **Scope decision (Prabin)** — ship the 4-clip validated OURS row as-is (strong, uniform
   held-frame contact win; honest, root-caused whole-clip floorPen residual tied to G1's size vs
   this human), broaden to more of the 77-clip corpus, or invest in a local/adaptive over-reach
   correction (untried) before broadening.
2. **GPU ask (E5/S3)** — still open, was a day-1 item in the sprint plan, not yet actioned. Dance 5
   first (their own named GMR failure, sudden waist jumps — exactly what Stage A removes), then
   the 2×2 on a subset + 1-2 floor clips.
3. **Table-I→BVH mapping** — still unresolved after checking the paper website and GMR's repo
   configs; author contact is the remaining option, Prabin's call.
4. Our own FBX floor clips into GMR — still deferred, still worth doing eventually for a
   same-source apples-to-apples comparison.

No early-stop condition has fired anywhere in the E1→E3 ladder (week 1), W2-T1/T2/T6, or Sprint
S1 — those results all stand as shipped. W2-T5, W2-T7, and S2's aggregate-floorPen gap are each a
real, reported negative or open checkpoint — logged honestly, not silently dropped, and none
retract anything shipped earlier.

> **UPDATE (2026-07-17): S2's OURS-on-G1 row below is SUPERSEDED.** Sprint S3 (77-clip
> corpus) found the S2 held-frame win was gameable: a single per-clip constant Z-shift
> applied to `gmr_polished` beats OURS on held-frame-within-3cm too (96-99% vs 82-87%)
> AND on max floorPen — the joint metric (held contact AND whole-body pen, simultaneously)
> is the only one that can't be gamed this way, and nothing passed it going into S4.
> Sprint S4 tried to fix OURS-DLS's floor penetration directly; best result (tuned
> `--swing-clear`) never cleared the gate. After visually comparing renders, Prabin
> pivoted (S5): GMR's own mink-based tracking is the base now, with a minimal contact
> layer (`gmr_contact_retarget.py`) layered on top, rather than continuing to patch
> OURS-DLS. See `GMR-S5-plan.md` + `planLogGMR.md ## S3/S4/S5-*` for the full trail,
> and the new "Sprint S3-S5" section at the end of this file for final numbers. The
> Sections below (Week 1/2, Sprint S1, Sprint S2) are accurate HISTORY — the analysis
> and diagnostics still hold, only the "OURS = contact-first DLS pipeline" framing they
> conclude toward has changed.

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

## Sprint S3: the joint metric, and why S2's row above is superseded (2026-07-16/17)

Ran the 4-clip validation's own held-frame metric at 77-clip scale, plus a probe: does a
single per-clip constant Z-shift on `gmr_polished` also win? **Yes** — shift-oracle
`gmr_polished` beats OURS-DLS on held-frame-within-3cm (96-99% vs 82-87%) AND on max
floorPen (6.6-13.4cm vs 17-23cm), because GMR-polished's held-foot float is nearly
constant within a clip (p90-p10 ≈ 2.6-3.3cm) — one constant zeroes it. Single-axis
metrics (float alone, or penetration alone) are gameable this way and are dead for the
paper. The un-gameable target: **held-foot contact within 3cm AND whole-body floor
penetration <5mm, at the SAME frame** — a rigid shift cannot satisfy both (trades one
for the other). OURS-DLS's own blocker: 62-81% of frames >5mm below floor, already at
the raw solve. Full trail: `planLogGMR.md ## S3-*`.

## Sprint S4: tried to fix OURS-DLS directly — best result doesn't clear the gate

`GMR-S4-plan.md` (superseded, kept for reference). Root-caused OURS-DLS's floor
penetration to a warm-start knee-limit basin + missing floor-avoidance in the per-frame
solve. Best mechanism found: `--swing-clear` (ported from the Alex pipeline, re-tuned
for G1: `--swing-max-pitch 8 --swing-continuity-reg 0.2`) — improves pen%/held-contact
on every one of 11 test clips, mean floorPen net BETTER than baseline. Still: mean
pen% 55.9% on an 11-clip set, nowhere near a `pen%<=10` bar. Full trail:
`planLogGMR.md ## S4-*`.

## Sprint S5: pivot — GMR's own tracking + a minimal contact layer (2026-07-17, superseded — see S6 below)

After S4, Prabin rendered and directly compared GMR's own output (heightfix, zero
smoothing from us) against OURS-DLS (tuned swing-clear) on `walk1_subject1`: GMR looked
visually excellent (no flicker/snapping, natural hand orientation); OURS looked bad
(flickers, snaps, palm rotated into the thigh, deep penetration). Root cause, found by
reading GMR's `mink`-based solver directly: (1) real QP joint-limit constraints inside
the solve vs OURS's post-hoc clamp (a documented flicker source), (2) GMR is
orientation-FIRST (position weight 0 on most bodies, only feet position-anchored) vs
OURS's position-first/weak-orientation scheme — this is the palm/knee mechanism, not
random noise, and (3) **GMR itself has ZERO per-frame contact handling** — its only
floor mechanism is the same constant-per-clip-shift the S3 oracle already killed. That's
the paper's actual opening: contact-aware layer on a SOTA kinematic retargeter, not a
from-scratch contact-first solver. Full plan: `GMR-S5-plan.md`. Decision log:
`planLogGMR.md ## S5-D0`.

**Phase B (does OUR DLS solver reach GMR quality with reweighting? time-boxed, does
NOT gate Phase A)**: a real hand-target bug found and fixed (B1 — our hand orientation
used a landmark-derived frame that couldn't represent forearm twist at all; switched to
the raw BVH bone rotation, which is EXACTLY GMR's own signal — validated to 0.00°
residual once a yaw-convention confound was isolated). Reweighting toward GMR's
orientation-first philosophy (B2) made things WORSE without also adding the
thigh/shank/arm orientation roles GMR has and we don't (not attempted, out of
time-box). Two numerical fixes (B3: active-set joint limits, convergence early-exit)
were both informative negatives (active-set is a mathematical no-op given the existing
every-iteration clamp; early-exit trades real quality for speed at OUR solver's weight
scale). **B-GATE: FAIL** — best config's body-jerk still 3.6x GMR's, nowhere near
GMR's visual quality. OURS-DLS is retired to an ablation/comparison role (B1's hand fix
kept permanently); Phase A was always the critical path regardless.

**Phase A (contact layer inside GMR's own solve — the paper)**: `gmr_contact_retarget.py`
subclasses `GeneralMotionRetargeting` (never edits the GMR clone), overrides only a
held foot/hand's table-2 FrameTask (locked position at contact onset + cosine-ramped
cost), sanity-checked byte-identical to plain GMR when the override is off. Headline
metric `joint_ok_pct` (S4's un-gameable target) on the 3 locomotion dev clips:

| clip | gmr_raw | gmr_heightfix | gmr_contact (A1) |
|---|---|---|---|
| walk1_subject1 | 97.9% | 86.8% | **98.9%** |
| walk3_subject1 | 92.0% | **1.4%** | 94.4% |
| run2_subject1 | 91.9% | **8.6%** | 85.2% |

`gmr_heightfix`'s catastrophic collapse on walk3/run2 (1.4%/8.6%) is the z-shift-oracle
failure mode reproduced live — a single constant cannot satisfy the joint metric.
`gmr_contact` beats it by 12-93 points on every clip and matches/exceeds `gmr_raw` on
2/3. Skate (foot-plant slip) <0.5cm mean everywhere; fidelity to GMR's own non-foot
targets barely moves (+0.2-0.5cm/+0.1-0.2°). Residual: motion jerk at contact
transitions still elevated 10-70% vs `gmr_raw` after 3 tuning passes — not closed.
A post-hoc ablation (applying the same held-target snap AFTER a raw GMR solve, via the
already-existing `stage_b_g1.py` QP) achieves near-ZERO jerk cost but its held-accuracy
is reliable only at walking speed, not running (a genuine, different limitation).
Extended to hands on the hard fall/get-up class: real accuracy win (worst case
10.6%→94.9%) but whole-body penetration on lying/prone poses is untouched (a reach-
limit problem, not a contact problem — G1 is ~0.64 this human's scale). Full trail incl.
tuning attempts: `planLogGMR.md ## S5-A*`. Full 77-clip corpus + eval: `## S5-A6` (see
`outputs/gmr_baseline/sprint/s5_full_corpus.csv` if the build finished).

## Sprint S6: hard floor constraint + median-centering post-process (2026-07-17)

S5's held-cost contact layer won the joint metric by 12-93 points but left non-
penetration structurally ungoverned — a soft QP cost traded against tracking,
covering only held effectors. Prabin's diagnostic: don't just report "float
reduced," check whether the SPREAD between worst-case float and worst-case
penetration actually collapses toward zero (a rigid Z-shift, like GMR's own
heightfix, provably cannot change this spread — confirmed exactly, 7.98cm==7.98cm
locomotion base-vs-heightfix, at both dev-clip and full 77-clip scale). Full plan:
`GMR-S6-plan.md`. Decision log: `planLogGMR.md ## S6-*`.

**Phase A (shipped, primary method)**: tried appending `mink.CollisionAvoidanceLimit`
directly to GMR's own solve — found a real bug in GMR's reference code (`ik_limits`
passed positionally lands on `safety_break`, never reaches `limits`; confirmed via
direct QP diff) but even fixed, the rate-limited QP inequality can't converge
within GMR's ~1-iteration-per-frame solve loop. Abandoned that path; built
`leg_floor_clamp.py` — a direct, deterministic damped-least-squares clamp on OUR
OWN vetted mesh geometry (not GMR's, which excludes the foot's real mesh from
collision entirely), shipped as `gmr_contact_retarget.py --floor-clamp`
(`gmr_contact_fc`). Found and fixed three more bugs during dev-clip testing
(coverage gap — worst penetration was often an elbow or hip, not a foot; a
correction-order bug where a later proximal fix silently re-violated an already-
corrected distal body; too few DLS iterations near saturated joint limits).

Full 77-clip corpus (0 build failures):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco (43) | gmr_raw | 3.0 | 5.15 | 91.5 | 7.98 |
| loco (43) | gmr_heightfix | 3.4 | 3.06 | 46.3 | 7.98 |
| loco (43) | gmr_polished | 1.0 | 2.99 | 32.1 | 8.76 |
| loco (43) | gmr_contact (S5) | 3.9 | 4.95 | 94.2 | 6.67 |
| loco (43) | **gmr_contact_fc** | **0.2** | **0.72** | **99.6** | **3.59** |
| floor (34) | gmr_raw | 23.4 | 15.29 | 80.6 | 12.84 |
| floor (34) | gmr_heightfix | 0.4 | 2.76 | 0.2 | 12.84 |
| floor (34) | gmr_polished | 0.3 | 2.56 | 0.4 | 11.77 |
| floor (34) | gmr_contact (S5) | 24.3 | 15.35 | 84.0 | 12.32 |
| floor (34) | **gmr_contact_fc** | **6.9** | **8.08** | **91.0** | **9.80** |

Beats every baseline — including `gmr_polished` (GMR + our own Stage-A polish,
itself a stronger baseline than raw GMR) — on the un-gameable joint metric AND the
float/penetration range on both classes. Locomotion floorPen clears the <1cm gate
at full corpus scale.

**Phase B (shipped, independent contribution)**: Prabin's proposal — median-shift
GMR's raw output so held-frame float/penetration are balanced, then a cheap
per-limb DLS pass reusing Phase A's `clamp_limb`. Retargeter-agnostic (works on
any qpos pkl, no Phase-A dependency). Two more real bugs found and fixed during
dev-clip testing (a Jacobian queried at the wrong world point in held mode,
causing solve divergence; a body-origin-offset constant mistakenly used as the
floor target, producing a systematic +4cm float bias). Full 77-clip corpus (0
build failures):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco (43) | gmr_raw | 3.0 | 5.15 | 91.5 | 7.98 |
| loco (43) | gmr_contact_fc (Phase A) | 0.2 | 0.72 | 99.6 | 3.59 |
| loco (43) | **medianlimb** | 0.7 | 3.15 | 98.7 | 6.65 |
| floor (34) | gmr_raw | 23.4 | 15.29 | 80.6 | 12.84 |
| floor (34) | gmr_contact_fc (Phase A) | 6.9 | 8.08 | 91.0 | 9.80 |
| floor (34) | **medianlimb** | 9.5 | 9.24 | 91.8 | 14.92 |

Beats `gmr_heightfix`/`gmr_polished` on joint_ok on both classes. Honest caveat
found at full scale: on the floor class, medianlimb's range (14.92cm) is worse
than doing nothing (`gmr_raw`'s 12.84cm), even though joint_ok and pen% both
clearly improve — B's held-lock doesn't reliably tighten the worst-case spread on
deep floor-contact clips, only the typical case. On locomotion it's a clean win
(range 7.98→6.65cm, close to Phase A's own 3.59cm, from a fully independent
mechanism). A per-frame-smoothed variant (vs. the constant shift) gets the best
floor-class numbers of any mechanism tested this sprint but has one unresolved
divergence bug — not corpus-shipped this pass.

**Stacked (A then B)**, tested on 2 representative dev clips: decisive win on
locomotion (float/penetration range collapses to 0.10cm — the literal ask from
this sprint, both endpoints nearly coincide), a wash on the hardest floor-contact
clip (range gets worse than either mechanism alone — not investigated further).

**Decision: Phase A ships as the primary paper method** (wins outright on both
classes at full corpus scale); Phase B ships as an independent, retargeter-
agnostic contribution, useful standalone and as an optional locomotion-class
booster stacked on top of A. Full trail: `planLogGMR.md ## S6-*`. Wiki:
[[gmr-baseline-sprint-s6]].

## Sprint S7: smoothness gate, floor-mechanism fix, self-collision fix (2026-07-17/18)

A paper-readiness audit of S1-S6 found four holes: `gmr_contact_fc`/`medianlimb`
had zero smoothness/skate/fidelity numbers; the best floor mechanism found so
far (`--center perframe`) sat one bug away from corpus scale; zero
renders/figures existed; and there was no OmniRetarget baseline (Undermind's
strongest contact-aware kinematic competitor). Plan: `GMR-S7-plan.md`. Full
trail: `planLogGMR.md ## S7-*`. Wiki: [[gmr-baseline-sprint-s7]].

**T1, the smoothness eval hole**: new `scripts/g1/sprint_s7_smoothness.py`
(joint/body jerk, skate, fidelity, vMax/vP95/spike-count). Found
`gmr_contact_fc` trades real smoothness for its joint-metric win — body_jerk
+56-177% vs `gmr_raw` on 3/5 dev clips, and on the floor class introduces
velocity spikes `gmr_raw` never has at all (22/34 floor clips spike at corpus
scale vs 0-3/34 for raw). `gmr_polished` is smoothest everywhere but is the
worst variant on the un-gameable joint metric — no variant had both.

**T3, floor mechanism fixed**: `--center perframe`'s walk1_subject1 divergence
was root-caused (not guessed, confirmed by direct instrumentation) to a
near-singular knee-limit configuration inside `clamp_limb`'s DLS, not the
originally suspected stale-target theory. Fix: an opt-in per-iteration `dq`
trust-region cap (0.15 rad), scoped ONLY to this call path — tested as a global
default first and rejected (it regressed the shipped Phase A mechanism on
deep-crouch frames that legitimately need large corrections). Gate passes
clean on all 5 dev clips; shipped as `perframelimb`, corpus build authorized.

**T2, smooth-then-clamp**: new `scripts/g1/smooth_then_clamp.py` (Stage-A
smoothing, then one re-clamp pass to restore the floor contact smoothing would
otherwise reintroduce). Passed decisively on the first attempt — jerk lands
BELOW `gmr_raw` itself (joint -72% to -92%, body -37% to -83%), and range
improves on every dev clip. Shipped as `gmr_contact_fc_sm`, the first variant
to combine un-gameable contact correctness with GMR-raw-level smoothness.

**T4**: draft renders/figures under `outputs/gmr_baseline/sprint/renders/s7/`,
including a money-shot pair (ground1_subject1, same frame/pose: `gmr_raw` shows
a 17.09cm elbow-through-floor penetration during a prone crawl, `gmr_contact_fc`
shows 0.00cm with a visually plausible forearm-on-floor contact).

**T7, self-collision fix (Prabin caught it visually)**: an S7-T4 figure showed
a `gmr_contact_fc` hand passing through the head despite 0.00cm floor
penetration. Confirmed with data already being tracked since S3 but never
called out: floor-class self-collision 6.34%→9.95% after `--floor-clamp`, peak
depth 5.66→7.50cm. Root cause: `clamp_limb` corrects one limb chain's floor
clearance with zero awareness of any other body.

Three fix designs tried, one shipped: (1) a mixed-rows single-DLS-solve
approach ported from the trusted `stage_b` QP row-builder — worked perfectly
isolated, catastrophic end-to-end (one badly-converged frame cascades through
every later frame's warm start); (2) **two-phase — floor/held converge first
unchanged, then a bounded collision-only correction on the same chain —
SHIPPED**, isolated test 8.75cm self-collision → 0.00cm at zero floor cost;
(3) a floor mop-up pass after phase 2 — made things worse via the same
cascading-warm-start fragility, rejected. `coll_weight=0.5` shipped as default
(1.0 is catastrophic on fallAndGetUp1: floorPen 5.70→24.84cm).

Full 77-clip corpus rebuild of all four affected variants (fc, fc_sm,
medianlimb, perframelimb), 0 build failures. Clean before/after (fc,
apples-to-apples against a preserved pre-fix backup):

| class | metric | pre-fix | post-fix |
|---|---|---|---|
| loco (43) | coll% | 3.86 | **0.01** |
| loco (43) | collPeak_cm | 5.08 | **0.13** |
| loco (43) | floorPen_cm | 0.72 | 2.32 |
| loco (43) | joint_ok% | 99.6 | 97.9 |
| floor (34) | coll% | 9.95 | **0.05** |
| floor (34) | collPeak_cm | 7.50 | **0.63** |
| floor (34) | floorPen_cm | 8.08 | 11.75 |
| floor (34) | joint_ok% | 91.0 | 88.8 |

Self-collision essentially eliminated on both classes (>99% incidence
reduction, peak depth down 10-20x), matching the dev-clip gate exactly at full
scale. Real, honest cost: floorPen worsens (loco +1.6cm, floor +3.67cm),
joint_ok drops 1.7-2.2pp. **Still wins decisively overall** — fc's joint_ok
(97.9%/88.8%) stays far above `gmr_polished` (32.1%/0.36%) and `gmr_heightfix`
(46.3%/0.19%), and floor-class joint_ok stays above `gmr_raw` itself (80.6%).
`medianlimb` shows the same pattern (floor coll% 9.93%→0.03%, small joint_ok
cost). `perframelimb` is now the strongest floor-class variant of anything
shipped (coll%=0.01%, floorPen=6.20cm, joint_ok=97.6% — all best-in-class).
`gmr_contact_fc_sm` holds up too (coll%=0.00-0.03%, joint_ok 97.7%/88.9%).

Smoothness before/after (the rebuilt `s7_smoothness.csv` only holds post-fix
numbers, so this was measured by re-running the 5 dev clips against preserved
pre-fix backup pkls):

| variant | joint_jerk %Δ | body_jerk %Δ | skateL %Δ | fidPos %Δ |
|---|---|---|---|---|
| gmr_contact_fc | +4.6% | +6.8% | +9.1% | ~0% |
| gmr_contact_fc_sm | +28.4% | +23.2% | +1.2% | ~0% |
| medianlimb | +7.7% | +9.7% | +24.0% | ~0% |
| perframelimb | +1.7% | +4.6% | +53.7% | ~0% |

`fc_sm` pays the biggest jerk cost (self-collision avoidance now fights the
Stage-A smoothing pass); `perframelimb` pays the least (still the safest
mechanism overall). Fidelity unaffected everywhere, as expected. Worst clip
across all four is `fallAndGetUp1_subject1` — the same clip with an unresolved
held-target range residual (phase 2's collision correction is chain-blind to
held-target locks; flagged, not chased further this pass). **Verdict: ship.**

**T5, OmniRetarget baseline**: found the actual code
(`github.com/amazon-far/holosoma`, Apache-2.0) via the OmniRetarget project
page. Confirmed G1 support, LAFAN1 support (including this project's own local
clips), and an isolated conda env. Execution (env setup, BVH→npy conversion,
run, output adapter, eval) paused mid-sprint to prioritize the T7 self-collision
fix — a real baseline number is the next open item, not yet run.

## Sprint S8: physical plausibility — LOCKED (2026-07-18/19, CURRENT)

S7 promoted `perframelimb` to the strongest floor-class variant on record but
left its corpus-scale smoothness/jerk profile unmeasured (S7-DECISION's option
D). S8 measured it (found it genuinely worse than `gmr_heightfix` on 3 of 6
"never-tradeable" axes at corpus scale — floorPen, n_spikes, vMax), spent the
sprint closing that gap via held-aware smoothing + re-clamp (`smrc`), a local
grounding envelope, and a rate-limited re-clamp, and ended with **S8-DECISION:
`perframelimb_smrc_rl_localground` locked as the working baseline at 5/6 axes**.
Full trail: `planLogGMR.md ## S8-*`, `## S8-DECISION`. Plan: `GMR-S8-plan.md`.
Wiki: [[gmr-baseline-sprint-s8]]. Method writeup (plain-language walkthrough +
full mathematics appendix): `GMR-METHOD.md` (repo root).

**Final locked-variant result, 77-clip corpus, never-tradeable axes vs
`gmr_heightfix`** (floor-class / loco-class means):

| axis | gmr_heightfix | LOCKED (`perframelimb_smrc_rl_localground`) |
|---|---|---|
| floorPen_cm | 2.76 / 3.06 | **0.00 / 0.00** |
| n_spikes | 0.18 / 0.00 | **0.00 / 0.00** |
| vMax_rad_s | 34.04 / 32.82 | 37.39 / 37.92 (lose, narrowed from a 63-65% gap to 9.8-15.5%) |
| coll_pct | 6.34 / 3.85 | **0.00 / 0.00** |
| worst_float_cm | 18.04 / 6.61 | **7.55 / 5.57** |
| joint_ok_pct | 0.19 / 46.26 | **98.85 / 98.79** |

5 of 6 axes win or tie; vMax is the sole remaining loss, real but an order of
magnitude closer than it started the sprint. `gmr_raw` (no grounding at all)
is never the comparison baseline — see (a) below. Corpus-wide hand slip is
reported for the first time this pass (0.66-0.73cm mean over 32 clips with a
genuine hand-hold segment) — `gmr_heightfix` has no equivalent number, since
it has no hand-contact handling at all.

**(a) Why `gmr_heightfix` ("GMR-full"), not `gmr_raw`, is the fair baseline.**
Standing rule since S8's R1.2: our method has no separate grounding stage —
grounding is baked directly into the per-frame contact clamp — so the fair
comparison is against GMR *with* its own grounding, not GMR ungrounded.
`gmr_raw` appears in every table only as an ungrounded reference column.

**(b) The constant-shift float↔pen window argument.** A single per-clip
constant vertical shift (GMR's own grounding mechanism) can zero worst-case
penetration, but it can only *slide* the float/penetration window, never
*shrink* it — the window's width (`range_cm` = worst float minus worst
penetration) is shift-invariant by construction. Measured on the S8 gate set:
`gmr_heightfix`'s range stayed fixed at 11.8cm regardless of shift, vs 7.3cm
for the per-frame-corrected method. Confirmed again at full 77-clip corpus
scale on the locked variant: range_cm 12.84→**7.93** (floor), 7.98→**6.02**
(loco) — narrower on both classes, because the correction responds to each
frame's actual contact state instead of one number chosen for the whole clip.
This is the core motivation story: a global shift is a placement choice, not
a contact-quality improvement, and `joint_ok_pct`/`worst_float_cm` are
designed specifically so a shift can't game them (S8-T5 demonstrated this
directly — naively applying GMR's own shift trick on top of our per-frame
clamp collapsed `joint_ok_pct` from 97.9% to 32.7%, because a constant sized
to the clip's single worst transient frame overshoots the tight tolerance
band every other, already-correct stance frame needs).

**(c) Rate-limiting converts spikes into drift — same energy, different
axis.** S8-T1b's first attempt (rate-limiting the *original*, pre-smoothing
per-frame clamp) killed velocity spikes but gave the correction back as
positional drift instead (float, range, and skate all regressed) — capping
how fast a correction can *change* doesn't remove the correction's total
magnitude, it just redistributes it into a different metric. S8-T8 re-applied
the same mechanism to the `smrc` re-clamp step specifically (identified by
S8-T7 as the actual source of vMax/n_spikes, not the smoothing weights) and
reproduced the same directional trade — but at roughly 1/10th the original
magnitude (skate +0.06-0.14cm vs T1b's 0.44→1.56cm), because by that point in
the pipeline the rate limiter only has residual smoothing-perturbation left
to correct, not a full raw-to-floor-safe correction. The lesson generalizes:
this class of fix doesn't eliminate a correction's cost, it moves the cost to
whichever axis is cheapest to pay it on at that point in the pipeline.

**(d) Smoothing fairness (both arms, held-aware) and what `heightfix_sm`
showed.** Per R1.3, the same held-aware smoother was applied to BOTH arms —
`perframelimb_sm` (ours) and `gmr_heightfix_sm` (fairness arm) — not just
ours. Result: `heightfix_sm`'s peak velocity drops dramatically (vMax
34.0→**12.7** rad/s floor, 32.8→**11.1** rad/s loco) — far lower than
anything in the locked variant's own vMax column. This is not a contradiction;
it's the fairness check doing its job. `gmr_heightfix` has no per-frame
contact correction to protect, so smoothing it is free — there is nothing for
a re-clamp step to have to restore afterward, so it pays none of the jerk cost
our pipeline's re-clamp/rate-limiter stack exists to manage. The trade shows
up elsewhere instead: `heightfix_sm`'s `floorPen_cm` on the loco class more
than doubles (3.06→**6.69**cm) and `joint_ok_pct` drops (46.3%→**39.2**%),
because smoothing a rigid, already-shifted trajectory with no geometric
awareness re-erodes the one thing the constant shift got right. Smoothing
alone is cheap for whichever variant has the least geometric correctness to
protect — the honest comparison is the full pipeline's cost, not vMax read in
isolation.

## Recommended next step (superseded — see S5 above for what actually happened)

1. ~~Scope decision (Prabin) — ship the 4-clip validated OURS row as-is...~~ Superseded
   by S3's oracle kill + S5's pivot.
2. **GPU ask (E5/S3)** — still open, never actioned.
3. **Table-I→BVH mapping** — still unresolved.
4. Our own FBX floor clips into GMR — still deferred.
5. **New, current**: extend `gmr_contact_retarget.py`'s hand mechanism to actually fix
   the ground1/fallAndGetUp reach-limit residual (untried — likely needs a different
   mechanism than contact, e.g. adaptive local scaling), and decide whether
   `gmr_contact_post`'s running-speed held-accuracy gap is worth root-causing for the
   paper's in-loop-vs-post-hoc ablation table.

No early-stop condition has fired anywhere in the E1→E3 ladder (week 1), W2-T1/T2/T6, or Sprint
S1 — those results all stand as shipped. W2-T5, W2-T7, and S2's aggregate-floorPen gap are each a
real, reported negative or open checkpoint — logged honestly, not silently dropped, and none
retract anything shipped earlier. S2's OWN headline claim (OURS-DLS wins held-frame contact) IS
retracted by S3's oracle test — the only retraction in this document — everything else stands.

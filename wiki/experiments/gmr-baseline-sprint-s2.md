# Sprint S2: OURS on G1 (contact-first solver, native port)

Bottom row of the 2×2 (`GMR-baseline.md` §7.4): our contact-first retargeter ported to G1 from
scratch (not an extension of E4/E4b's post-hoc anchoring, which W2-T5 ruled out for this motion
class). Full build/debug trail: `planLogGMR.md` `## S2-T1..T12`. Results narrative:
`GMR-baseline-results.md`'s "Sprint S2" section.

**Status (2026-07-18, Prabin's call): committing to a paper on this line — contact-first solving
(OURS) stays a core mechanism, not reduced to polish-only.** `scripts/g1/sprint_s3_full_corpus.py`
broadened OURS from the 4 validated clips to the full 77-clip LAFAN1 corpus (matching GMR's side,
already complete via Sprint S1). **Build + eval both DONE same day** — see "Full-corpus result
(S3)" below. `--knee-bias-weight` stayed OFF (plain S2-T9 config) pending the fallAndGetUp1
investigation. See `planLogGMR.md` `## S2-T12`'s tail and `wiki/log.md`'s 2026-07-18 entries.

## What's built

- `scripts/g1/lafan1_to_canonical_human.py` — LAFAN1 → canonical-human adapter (reuses Stage 2.5
  grounding unchanged).
- `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` — G1 Stage-3 analog: per-frame DLS IK,
  root+joints solved jointly, contact-held effectors anchored via pull-to-floor. Reuses
  `solve_frame_position_ik`/`make_targets_for_frame`/etc. UNCHANGED from Alex's solver.
- `scripts/g1/g1_model_setup.py` — vetted-collision + floor model (15 vetted cylinders from
  GMR's own URDF, 24 noisy duplicate mesh geoms disabled, floor mocap plane injected).
- `scripts/g1/polish_ours_g1.py` — Stage A (smoothing, with a floor/self-collision sensitivity
  boost, S2-T9) + optional contact-aware grounding.
- `scripts/g1/ground_ours_contact_aware.py` — lift QP with a hard cap (0 at held/contact frames,
  ∞ elsewhere) so grounding can't disturb already-correct contact frames.
- `scripts/g1/eval_g1_gmrscale_variants.py` — the 5-variant comparison (GMR+heightfix,
  GMR+ourpolish, OURS raw/+StageA/+StageA+ctground) across all 4 test clips.
- `scripts/g1/sprint_s3_full_corpus.py` (S2-T12) — same pipeline, broadened to all 77 LAFAN1
  clips, resumable, writes one combined GMR+OURS metrics CSV.
- `--knee-bias-weight`/`--knee-min-flex-deg` on `solve_lafan1_canonical_g1_contactfirst.py`
  (S2-T12, opt-in, default off) — one-sided knee-flexion DLS regularizer ported from the Alex
  pipeline, fixes the warm-start-basin mechanism S2-T11 found (mixed 4-clip result, not shipped
  as default yet).

## Bugs found and fixed (chronological, all in `planLogGMR.md`)

1. **Mocap-XML body-naming footgun**: `head_link` sits at pelvis height (cosmetic body, not
   anatomical head); `head_mocap` is correct. Silently zeroed `root_scale` if wrong.
2. **Degenerate hand-orientation fallback**: `hand_middle - wrist` axis was always zero (LAFAN1
   maps both to the same bone); fixed to `wrist - elbow` (forearm direction).
3. **`floor_gid=None` self-collision leak** (N1-a): combined model has an injected floor;
   omitting `floor_gid` in `_collision_stats` leaks floor contacts into self-collision counts.
4. **Pelvis pinned to world origin** (S2-T6): `make_initial_alignment_targets`'s rest target was
   literal `[0,0,0]`, not floor-referenced — dragged G1's whole leg chain ~0.5-0.7m below true
   floor. Fixed with a floor-referenced rest anchor (`pelvis_floor_z0 = root_scale *
   human_pelvis_z(0)`, uniform Z-shift on all initial targets).
5. **Root/ankle orientation runaway**: zero orientation regularization on unconstrained roles
   let the root drift 74° and the ankle pin to its joint limit during the one-time rest solve.
   Fixed with identity orientation targets on all 7 `ORI_TO_G1_BODY` roles.
6. **Grounding destroys contact quality**: GMR-style blind whole-clip Z-shift grounding, applied
   to OURS's already-correct contact frames, dragged them 12-19cm away from the floor. Fixed:
   grounding OFF by default in `polish_ours_g1.py`; contact-aware grounding (hard cap) is the
   safe alternative.
7. **Kinematically-inconsistent leg-chain scaling** (S2-T7, the big one): independent per-role
   scale computation produced hip=2.45/knee=0.97/ankle=0.79 on the same rigid leg — a target
   thigh length nearly 2x G1's real one. Root cause: `left_hip_yaw_link` (matching GMR's own
   ik_config) sits 28cm from pelvis (near-knee), not a hip-joint analog — harmless for GMR's IK,
   poisonous for an independent-ratio formula. **Fixed by replacing per-role scales with GMR's
   own published grouped constants** (0.9 lower-body, 0.75 upper-body, from
   `bvh_lafan1_to_g1.json`) — the only remaining OURS-vs-GMR difference is now contact
   enforcement in the solve, not morphology scaling.
8. **Stage A floor-blind smoothing overshoot** (S2-T9): Stage A's tridiagonal smoother blended a
   sharp, narrow raw dip toward smoother-but-wronger neighbours, producing a NEW worse peak (same
   mechanism already documented for the Alex pipeline, `_detect_floor_sensitive_frames`). Fixed
   with a local tracking-weight boost at sustained floor- AND self-collision-sensitive frames
   (extended beyond the mainline's floor-only version, per Prabin's ask) — confirmed to still
   roughly halve peak joint velocity despite wide sensitivity coverage (Stage A's actual job
   survives).

## Validated result (4 clips: walk1_subject1, fallAndGetUp1_subject1, fallAndGetUp2_subject2,
ground1_subject1)

**Held-frame support_z (the discriminating metric — foot-to-floor distance when the human is
actually planted)**: GMR+heightfix and GMR+ourpolish both fail badly on the 3 floor-contact
clips (feet floating 8.6-12.5cm up, 0-5% within 3cm). OURS holds contact within ~1cm median,
66-73% within 3cm, uniformly across all 4 clips including the hardest ones. This is the paper's
central claim, now decisively validated beyond just a walking sanity check.

**Self-collision** improved substantially after the S2-T7 chain fix: walk1 15.3%→2.1% (matches
GMR's own level), fallAndGetUp1 13.0%→6.4%, fallAndGetUp2 16.7%→14.3%, ground1 18.7%→11.6%.

**Whole-clip aggregate floorPen** still trails GMR-polished (17-40cm raw vs 1-5cm) — see below.

## The remaining residual: G1's kinematic reach limit, PLUS a solver warm-start-basin issue (S2-T9/S2-T10/S2-T11/S2-T12)

Diagnostic ladder (all confirmed by direct measurement, not inference): ruled out iteration count
(30-1000 iters converge identically), self-collision competition (coll_weight 20→0 identical).
Root-caused (first pass, S2-T9/T10): `root_scale` (G1-to-human size ratio via pelvis-to-head)
measures ~0.64-0.65 across all 4 clips — G1 is genuinely only ~64% this human's size. At extreme-
extension poses (deep gait stride, get-up limb sweeps), the retargeted target occasionally exceeds
G1's max physical leg reach (confirmed: 2650 frame-legs on fallAndGetUp2_subject2 exceed 100% of
max reach, worst case 181.6%).

**Tried and rejected (S2-T10)**: applying the size-correction factor to the per-role
relative-motion term too (matching GMR's own two-term `(h/h_ref)` formula structure) shrinks the
over-reach cleanly (2650→852 frame-legs) but regresses every other metric badly when run
end-to-end (self-collision e.g. ground1 11.6%→34.5%, floorPen e.g. fallAndGetUp1 25.5→50.9cm) —
shrinking the WHOLE clip's limb excursion to fix a handful of extreme frames cramps the entire
gait. Reverted; confirmed output files match the pre-attempt validated state exactly.

**Correction (S2-T11, 2026-07-18)**: the earlier claim above ("stuck at its best achievable pose,
not fighting anything") was WRONG, caught by Prabin's pushback that a too-short leg overshooting
THROUGH the floor (not falling short of it) doesn't fit a pure reach-limit story. Direct test: the
knee IS pinned at its hard lower limit (straight) at the worst frames, but manually pre-bending the
knee in the warm start before re-solving the SAME target resolved 2 of 4 tested frames almost
completely (one clip: 9.76cm pen → 0.00cm) — this is a **solver local-minimum / warm-start-basin
problem**, not purely a hard physical wall. Each frame warm-starts from the previous frame's
solved pose; once a knee lands near its limit, the per-iteration clamp keeps re-clipping it there
and the small per-iteration step budget never lets the solver climb back out to the bent-knee
alternative that demonstrably fits better. Also directly refuted a competing "hold-tier leak"
hypothesis (does holding one foot drag the other through the floor via shared root DOFs?) — empty
`hold_pos_roles` at every tested worst frame, nothing to leak from.

**Fix built (S2-T12)**: ported the Alex pipeline's existing `knee_bias` DLS regularizer (never
wired into the G1 script before) as opt-in `--knee-bias-weight`/`--knee-min-flex-deg`. 4-clip
result is genuinely mixed — 2 clips clearly better (incl. held-frame contact, the paper's central
metric, improving further), 1 clearly worse on whole-clip floorPen (fallAndGetUp1, +11cm,
unexplained), 1 inert. **Not shipped as default** (`--knee-bias-weight` stays 0.0) pending either
root-causing fallAndGetUp1's regression or a scope decision to ship anyway given the central metric
improves. Full numbers: `planLogGMR.md` `## S2-T11/T12`.

**Genuinely-remaining conclusion**: part of the residual is an honest retargeting-fidelity limit
(some poses really are geometrically unreachable at G1's size, e.g. the harder 598/601 frames that
DIDN'T fully resolve even with a bent warm start) — not closeable by any uniform linear scale, per
S2-T10. But part of it (confirmed at 2/4 tested frames) is solver warm-start chaos with a concrete,
partially-working fix already in hand. The two are entangled and not yet fully separated.

## Full-corpus result (S3, 2026-07-18)

> **INVALIDATED as a contribution claim by the z-shift oracle kill-test (same day, next section).**
> The numbers below are correct but the "decisive win" framing is not: a per-clip constant
> downward shift of GMR-polished beats OURS on this same metric. Kept for the record.

77-clip build (`sprint_s3_full_corpus.py --build`, resumable, 0 failures) then eval
(`--eval`, combined GMR 3-variant + OURS 3-variant CSV, 462 rows) both completed same day as the
S2-T12 write-up above. Class split via the S1-T4 reclass convention (`s1t4_reclass.csv`,
multi-surface contact, not hip-height alone): 34 floor-class, 43 locomotion-class. Summary tool:
`scripts/g1/sprint_s3_summary.py` (S1-T4-style class-split markdown table).

**Held-frame support_z (the central metric) — confirms the 4-clip finding at full corpus scale,
same direction, larger gap than expected:**

| class (n) | variant | held-foot median (cm) | held-foot within-3cm (%) |
|---|---|---|---|
| floor (34) | gmr_heightfix | 14.26 / 14.34 (L/R) | 0.15 / 0.49 |
| floor (34) | gmr_polished | 11.72 / 11.83 | 0.27 / 0.62 |
| floor (34) | ours_ctground | **-0.05 / -0.03** | **82.3 / 84.2** |
| locomotion (43) | gmr_polished | 4.38 / 4.37 | 31.4 / 31.0 |
| locomotion (43) | ours_ctground | **-0.10 / 0.37** | **87.0 / 86.8** |

On the floor-contact clips specifically (the paper's actual target class) GMR-polished's held
foot floats ~11.7cm median and is within 3cm of the floor on well under 1% of held frames — i.e.
at corpus scale GMR essentially never satisfies contact fidelity once you check it directly,
not just gross floor pen. OURS holds ~0cm median, 82-84% within 3cm, on both classes.

**Whole-clip aggregate floorPen still trails GMR-polished at corpus scale**, same known mechanism
as the 4-clip run (S2-T9/T10/T11, reach limit + warm-start-basin, `knee_bias` OFF here):
floor-class 23.0cm (OURS ctground) vs 2.56cm (GMR-polished); locomotion-class 17.0cm vs 2.99cm.
Not a new finding, just now corpus-confirmed rather than 4-clip.

Raw CSV: `outputs/gmr_baseline/sprint/s3_full_corpus.csv`.

## Z-shift oracle kill-test (2026-07-18): the S3 held-frame win is NOT a contribution as framed

Prabin's objection, immediately on seeing the S3 table: "if I just move the robot down, float
will decrease and penetration will increase" — i.e. the held-frame support_z metric might be
gameable by a rigid per-clip vertical shift. Tested directly (scratchpad `zshift_oracle.py`,
output `outputs/gmr_baseline/sprint/s3_zshift_oracle.csv`): for each clip, shift GMR-polished
down by (a) its pooled held-float median, (b) the frac3-optimal constant (fine grid sweep).
Rigid z-shift is exact-analytic for these metrics (support_z and lowest-z shift by dz;
self-collision invariant), so one forward-pass sweep per clip suffices.

**Result: the trivial shift beats OURS-ctground on BOTH axes of the S3 table** (means over clips):

| class | variant | frac3 L/R (%) | floorPen max (cm) | pen% (>5mm frames) |
|---|---|---|---|---|
| loco (43) | GMR-pol + oracle shift | 98.9 / 99.1 | 6.6 | 100 |
| loco (43) | ours_ctground | 87.0 / 86.8 | 17.0 | 69.4 |
| floor (34) | GMR-pol + oracle shift | 96.0 / 95.3 | 13.4 | 100 |
| floor (34) | ours_ctground | 82.3 / 84.2 | 23.0 | 65.7 |

Clips with both feet frac3 ≥ 80%: shifted-GMR 76/77, OURS 50/77. Mechanism: GMR-polished's
held-frame float is nearly CONSTANT within a clip (p90−p10 spread ≈ 2.6cm loco / 3.3cm floor),
so one constant shift zeroes it. The S3 table only looked decisive because GMR never bothers to
shift.

**What survives**: nothing about the current metric pair, but the physical desideratum is still
real — the shifted baseline buries the support foot 3.6cm (loco) / 10.8cm (floor) below the floor
on essentially every frame (pen% = 100), which is visibly wrong; the honest target is
**simultaneous** held-frame contact AND no penetration, per frame, which no rigid shift can give
when float varies across contact events. But OURS doesn't deliver that either right now: its own
pen% is 62-81% starting at the RAW solve (not introduced by grounding), from the known reach-limit
+ warm-start-basin mechanism. Under a joint metric (contact within 3cm AND pen < 5mm), every
variant on the board currently fails. **OURS has no defensible corpus-scale claim until its
penetration mechanism is fixed** — which was already the top open item (knee_bias root-cause,
local over-reach correction). The kill-test also defines the bar any future claim must clear:
beat the per-clip-oracle-shifted baseline, not vanilla GMR.

## Open (superseded — see below)

- fallAndGetUp1's knee_bias regression — not root-caused, needed before a ship decision.
- Held-classifier false positive (a few isolated frames classified "held" despite large error) —
  flagged, not fixed, small.
- Local/adaptive over-reach correction (for the genuinely-unreachable-pose remainder) — untried.
- Whether to re-run the S3 corpus eval with `knee_bias` ON once the fallAndGetUp1 regression is
  root-caused, to see if the aggregate floorPen gap narrows at corpus scale too.

## SUPERSEDED (2026-07-17): the "Open" items above were never actioned under this framing

Sprint S4 tried to fix OURS-DLS's penetration directly (the natural next step from this
page's own "Open" list) — root-caused the warm-start-basin mechanism further, shipped a
tuned `--swing-clear` mechanism, still didn't clear the joint-metric gate. After
visually comparing GMR's own output against OURS-DLS, Prabin pivoted (S5): GMR's own
tracking is the base now, OURS-DLS retired to an ablation role. See
[gmr-baseline-sprint-s4-s5](gmr-baseline-sprint-s4-s5.md).

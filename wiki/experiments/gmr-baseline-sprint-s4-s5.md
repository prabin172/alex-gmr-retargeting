# Sprint S4-S5: fixing OURS-DLS (failed), then pivoting to GMR + a contact layer

Continuation of [gmr-baseline-sprint-s2](gmr-baseline-sprint-s2.md) after the z-shift
oracle kill-test invalidated S3's held-frame win as a contribution claim. Full trail:
`planLogGMR.md ## S4-*` / `## S5-*`. Plans: `GMR-S4-plan.md` (superseded),
`GMR-S5-plan.md` (current). Results narrative: `GMR-baseline-results.md`'s Sprint S4/S5
sections.

## Sprint S4: root-caused OURS-DLS's penetration, best fix didn't clear the gate

Diagnosed the warm-start knee-limit basin further and found the per-frame solve had no
real floor-avoidance mechanism outside contact-held effectors. Tried several fixes:

- `refine_leg_floor_transitions_g1` (ported from Alex's onset-ramped leg-chain refine):
  needed a divergence guard after an early version hit +200-370cm floorPen on
  fallAndGetUp clips; even guarded, improves the mean but costs the tail — not shipped.
- **`--swing-clear`** (capped swing-foot toe-pitch + posture-reg continuity boost,
  ported from Alex, retuned for G1: `--swing-max-pitch 8 --swing-continuity-reg 0.2`
  vs Alex's 5/0.9): the sprint's cleanest result — improves pen%/held-contact on every
  one of 11 test clips, mean floorPen net BETTER than baseline. Still: mean pen% 55.9%
  on the 11-clip set, nowhere near a `pen%<=10` gate.

S4-T5/T6 (joint metric + oracle baseline into the eval harness, full corpus rerun) were
planned but never run — the pivot below happened first.

## The pivot trigger: visual comparison of GMR's own output vs OURS-DLS

Prabin rendered and compared three annotated videos on `walk1_subject1` (raw OURS,
tuned swing-clear OURS, GMR heightfix) using `scripts/g1/render_penetration_annotated.py`.
Verdict: GMR's own output (zero smoothing from us) looked visually excellent — no
flicker, no snapping, natural hand orientation (15.9% frames pen>0.5cm, max 1.66cm).
OURS looked bad even tuned — flickers, snaps, palm rotated into the thigh, deep
penetration (56.6% pen, max 8.76cm).

**Root cause, found by reading GMR's `mink`-based solver directly**
(`~/projects/GMR/general_motion_retargeting/motion_retarget.py`):

1. **Solver architecture**: GMR uses `mink` (velocity-space QP, `mink.solve_ik` +
   `configuration.integrate_inplace`, joint limits as HARD `mink.ConfigurationLimit`
   constraints INSIDE the solve). OURS-DLS clamps joint limits AFTER integrating each
   iteration (`clamp_hinge_joint_limits`) — a documented flicker/branch-flip source.
2. **Weighting philosophy**: GMR is orientation-FIRST — its table1 pass has POSITION
   weight 0 on almost every body (only feet get position weight 50), driving the whole
   pose via per-segment ORIENTATION (weight 10-100). OURS-DLS is position-first (15
   position-tracked landmarks, orientation only on 7 roles, max weight 0.70). A
   position-only limb target leaves hand roll (forearm-axis twist) and knee bend-plane
   completely unconstrained — this is the palm-toward-body and knee-weirdness
   mechanism, not noise.
3. **Contact**: GMR has ZERO per-frame contact handling. Its only floor mechanism is
   one constant Z offset for the whole clip (`set_ground_offset`) — literally the same
   mechanism as the S3 z-shift oracle. This is the paper's real opening: a contact-aware
   layer on top of a SOTA kinematic retargeter that has none, not a from-scratch
   contact-first solver competing with GMR's own tracking quality.

Decision (`planLogGMR.md ## S5-D0`): Phase B (bounded, ~2 days, does NOT gate Phase A) —
test whether reweighting/fixing OURS-DLS can reach GMR quality. Phase A (the paper) —
build the contact layer inside GMR's own solve, regardless of B's outcome.

## Phase B: can OURS-DLS reach GMR quality? No — B-GATE failed

- **B1 (shipped, real fix)**: our hand orientation target used a landmark-derived frame
  (forearm direction + pelvis-lateral secondary axis) that structurally could not
  represent forearm twist at all — not a weighting problem, a missing DOF in the
  target itself. Fixed by switching to the raw BVH bone rotation (`load_bvh_file`'s own
  per-frame quat) — validated to **exactly 0.00° residual** against GMR's own hand
  target once a yaw-facing-convention confound was isolated (GMR doesn't
  yaw-normalize clips; our pipeline does, for every role, not just hands — this
  explained most of the apparent mismatch in the first diagnostic pass).
- **B2 (reweighting toward GMR's orientation-first philosophy)**: made things WORSE.
  Cutting knee/elbow position weight without ALSO adding the thigh/shank/upper-arm/
  forearm orientation roles GMR has (B2b, not attempted — biggest remaining B item,
  explicitly skippable under time pressure) removes a constraint without replacing it;
  redundancy went up, not down (floorPen, self-collision, and knee flexion all got
  worse in the aggressive attempt; a softer attempt was still a net negative).
- **B3 (two solver mechanics)**: both informative negatives. Active-set joint-limit
  handling (zero a limited DOF's step before integrating, instead of clamping after)
  is a mathematical NO-OP given the existing every-iteration clamp — confirmed by
  direct instrumentation (8468 triggers in 300 frames, yet byte-identical output).
  Convergence early-exit (GMR's own `curr_error - next_error > 0.001` criterion) is a
  real 3.5x speedup but a real quality cost — OUR solver's weight scale (O(0.2-4)) is
  much smaller than GMR's (O(5-100)), so the same absolute tolerance triggers far
  earlier here, before self-collision/floor terms converge.
- **B-GATE: FAIL.** Best config's body-jerk still ~3.6x GMR's, nowhere near GMR's
  visual quality. OURS-DLS retired to an ablation role; B1's hand fix kept permanently.

## Phase A: contact layer inside GMR's own solve (the paper)

`scripts/g1/gmr_contact_retarget.py`: subclasses `GeneralMotionRetargeting` (never
edits the GMR clone), overrides ONLY a held effector's table-2 `mink.FrameTask` —
locked XY at contact onset (from the robot's own FK, not a synthesized value), Z at
the effector's own sole/support-mesh offset, flat orientation (yaw taken from GMR's
own current target), cost ramped in/out via a cosine cross-fade. Sanity-checked
byte-identical to plain GMR when the override is off (root_pos/root_rot/dof_pos max
diff 0.0 across 7840 frames).

**Bug found + fixed**: a clip already held at frame 0 would lock onset XY from mink's
pre-solve default configuration (garbage, ~2m off) — fixed by deferring the first
legitimate lock until after one full solve has run.

**Headline metric** (`joint_ok_pct`, S4's un-gameable target: every currently-held
effector within 3cm of its own floor AND whole-body pen <5mm, same frame), 3 locomotion
dev clips:

| clip | gmr_raw | gmr_heightfix | gmr_contact |
|---|---|---|---|
| walk1_subject1 | 97.9% | 86.8% | **98.9%** |
| walk3_subject1 | 92.0% | **1.4%** | 94.4% |
| run2_subject1 | 91.9% | **8.6%** | 85.2% |

`gmr_heightfix`'s collapse on walk3/run2 is the z-shift-oracle failure reproduced
live — a single constant cannot satisfy the joint metric on faster gaits where held
float isn't constant within the clip. `gmr_contact` beats it by 12-93 points
everywhere and matches/exceeds `gmr_raw` on 2/3 clips. Skate (foot-plant slip) <0.5cm
mean; fidelity to GMR's own non-foot targets barely moves.

**Residual**: contact-transition jerk is still elevated 10-70% vs `gmr_raw` after 3
tuning passes (ramp length, cosine vs linear shape, cost ceiling) — not closed. A
post-hoc ablation (same held-target snap applied AFTER a raw GMR solve, reusing the
existing `stage_b_g1.py` whole-trajectory QP) achieves near-zero jerk cost (its
`lambda_smooth` regularizer absorbs the transition) but its held-accuracy is reliable
only at walking speed — collapses on faster gaits even with an identical contact mask
to the in-loop version. Report both mechanisms; neither is a clean universal win.

Extended to hands on the hard fall/get-up class (`--effectors feet+hands`): real
accuracy win (worst case 10.6%→94.9% frac3), but whole-body pen on lying/prone poses
is essentially untouched (41.0%→40.5%, 87.8%→87.6%) — a genuine reach-limit problem
(G1 is ~0.64 this human's scale), not something a contact mechanism can fix.

## Open

- Contact-transition jerk residual (A2) — 3 tuning attempts, not closed. Candidate
  untried idea: soften the XY lock (a spring/margin instead of a hard pin) or blend
  toward GMR's own moving target rather than an independent locked one.
- `gmr_contact_post`'s running-speed held-accuracy collapse — not root-caused (likely
  `_pull_to_floor`'s per-run median-offset mechanism or `_compute_anchors`'s
  speed/min-run defaults undersuited to fast/short contact intervals).
- Hard-class whole-body reach-limit residual (A5) — same open problem as S2's, now
  confirmed orthogonal to contact accuracy specifically.
- ground1's right-foot regression when hands are also held (A5) — not root-caused,
  likely task-priority competition between simultaneously-held foot+hand overrides.
- Full 77-clip corpus (A6): build launched, may still be running — check
  `outputs/gmr_baseline/sprint/s5_full_corpus.csv` / `s5_build_nohup.log`.

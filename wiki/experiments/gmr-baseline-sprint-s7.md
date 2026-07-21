# Sprint S7: smoothness gate, floor-mechanism fix, self-collision fix

Continuation of [gmr-baseline-sprint-s6](gmr-baseline-sprint-s6.md). Paper-readiness
audit found four holes in S1-S6: no smoothness/skate/fidelity numbers for
`gmr_contact_fc`/`medianlimb`, `--center perframe` (the best floor mechanism)
one bug away from corpus scale, no renders, no OmniRetarget baseline. Full trail:
`planLogGMR.md ## S7-*`. Plan: `GMR-S7-plan.md`.

## T1: the smoothness eval hole

New `scripts/g1/sprint_s7_smoothness.py` — joint/body jerk, skate, fidelity,
vMax/vP95/spike-count (GMR's own mocap XML context). Found `gmr_contact_fc`
trades real smoothness for its joint-metric win: body_jerk +56-177% vs `gmr_raw`
on 3/5 dev clips, and on the floor class introduces velocity spikes `gmr_raw`
never has at all (corpus scale: 22/34 floor clips spike vs 0-3/34 for raw).
`gmr_polished` is smoothest everywhere but is the worst variant on the
un-gameable joint metric (S6). No variant had both — this tension is what T2
targets.

## T3: perframe floor mechanism fixed, now corpus-ready

`--center perframe` (S6 Phase B's best floor result, dev joint_ok 97.4%) had an
unresolved walk1_subject1 divergence. Root-caused via direct instrumentation
(not guessed): the right leg's stance phase drives `right_knee_joint` to its
exact lower limit, a near-singular configuration for `clamp_limb`'s
undamped-enough DLS — a ~1cm residual produces a `dq` large enough to snap the
ankle to world Z=0.80m in one frame, then re-diverge frame-to-frame since each
per-frame solve restarts from that frame's own raw pose. `--center median`
doesn't hit it (its constant shift avoids the exact singular basin).

Fix: opt-in per-iteration `dq` trust-region cap (0.15 rad), wired ONLY into
`polish_median_limbwise.py --center perframe`'s call path — tested as a global
default first and REJECTED (regressed Phase A's `gmr_contact_fc` on ground1,
which legitimately needs large single-iteration corrections on deep-crouch
frames that aren't singularities). Verified byte-identical elsewhere after the
revert. Gate passes on all 5 dev clips, no divergence, slightly beats the
pre-bugfix ballpark (joint_ok 97.8-100%, range ≤3.16cm on ground1/fallAndGetUp1).
Shipped as `perframelimb`, corpus build authorized.

## T2: smooth-then-clamp — decisive, first attempt

New `scripts/g1/smooth_then_clamp.py`: Stage-A tridiagonal smoothing (the same
function `gmr_polished` uses, untouched) applied to `gmr_contact_fc`, then one
full-clip re-clamp pass to restore the floor contact smoothing would otherwise
reintroduce. Result on 5 dev clips: jerk lands BELOW `gmr_raw` itself
(joint -72% to -92%, body -37% to -83%), joint_ok/pen% unchanged or improved on
4/5 clips, range improves on every clip (best case fallAndGetUp1 6.96→2.74cm).
Velocity spikes eliminated on all loco clips, reduced on floor clips.

**`gmr_contact_fc_sm` dominates `gmr_contact_fc` on smoothness AND range
simultaneously** — the combination T1 found no prior variant achieved. Shipped
as a paper-primary-variant candidate.

## T4: renders and figures

`outputs/gmr_baseline/sprint/renders/s7/` — full-clip renders, hstacked
comparisons, and penetration-annotated videos (existing tools, no new
mechanism). Money-shot pair: ground1_subject1 t=1407 (17.09cm elbow-through-floor
on `gmr_raw` during a prone crawl vs 0.00cm/plausible contact on
`gmr_contact_fc`, same frame/pose). Draft figures, composition deferred to
Prabin.

## T7: self-collision-aware clamp_limb

Prabin caught it visually in a T4 figure: a `gmr_contact_fc` still shows a hand
passing through the head despite 0.00cm floor penetration. Confirmed with data
(the project's own vetted collision cylinders, tracked since S3 but never
called out as an S7 finding): floor-class self-collision 6.34%→9.95% after
`--floor-clamp`, peak depth 5.66→7.50cm. Root cause: `clamp_limb` corrects one
limb chain's floor clearance with zero awareness of any other body — on cramped
floor poses it can drive a corrected elbow/knee straight into the torso/head
while satisfying its own floor target perfectly.

**Three fix designs tried, one shipped** (Prabin's framing: relax absolute
tracking, use relative inter-body terms):
1. Mixed rows (floor+collision in one weighted DLS solve every iteration),
   ported from the trusted `stage_b` QP row-builder — REJECTED. Perfect in an
   isolated single-frame test, catastrophic end-to-end (ground1 floorPen
   4.74→38.76cm) because Phase A's small 10-iteration-per-frame loop has none
   of `stage_b`'s whole-trajectory convergence guarantees, so one
   badly-converged frame cascades through every later frame's warm start.
2. **Two-phase (floor/held converge first unchanged, THEN bounded
   collision-only correction on the same chain) — SHIPPED.** Isolated test:
   8.75cm self-collision → 0.00cm, zero floor cost. Full clip: floorPen
   unchanged, coll% 13.12%→0.00%, joint_ok 95.3%→94.9%.
3. Phase-3 floor mop-up (re-run phase-1 once more after phase 2) — TRIED AND
   REJECTED, made ground1 worse via the same cascading-warm-start fragility.

`coll_weight` tuning: 1.0 works on 4/5 dev clips but is catastrophic on
fallAndGetUp1 (floorPen 5.70→24.84cm, joint_ok 97.1%→66.7%). **0.5 shipped as
default** — coll% → ~0% everywhere, small joint_ok cost (-0 to -1.6pp),
moderate floorPen cost on floor-class clips, a real honest trade.

**Known residual, not resolved**: at coll_weight=0.5, fallAndGetUp1's range
metric spikes 6.96→39.15cm, traced to one held-right-foot frame where phase 2's
collision correction (chain-blind to held-target locks) disrupts that lock
badly. Flagged for follow-up, not chased further (diminishing returns after 3
mechanism designs + 3 weight values tested).

Shipped into `gmr_contact_retarget.py --floor-clamp` (Phase A),
`polish_median_limbwise.py` (Phase B, both center modes), and
`smooth_then_clamp.py` (unconditional). Default OFF everywhere else, verified
byte-identical when off.

### Full 77-clip corpus verdict

All four `clamp_limb`-dependent variants (fc, fc_sm, medianlimb, perframelimb)
rebuilt with `--avoid-self-collision --coll-weight 0.5`, 0 build failures. Clean
before/after (fc, apples-to-apples against the pre-fix backup):

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
reduction, peak depth down 10-20x), matching the dev-clip gate exactly at
scale. Real cost: floorPen worsens (loco +1.6cm, floor +3.67cm), joint_ok drops
1.7-2.2pp. **Still wins decisively overall** — fc's joint_ok (97.9%/88.8%)
stays far above `gmr_polished` (32.1%/0.36%) and `gmr_heightfix`
(46.3%/0.19%), and floor-class joint_ok stays above `gmr_raw` itself (80.6%).

Other variants, same pattern: `medianlimb` floor coll% 9.93%→0.03% (small
joint_ok cost, loco unaffected); `perframelimb` is now the STRONGEST
floor-class variant of anything shipped (coll%=0.01%, floorPen=6.20cm best of
any variant, joint_ok=97.6% also best — the T3 fix holds up and even improves
relatively); `gmr_contact_fc_sm` coll%=0.00-0.03%, joint_ok 97.7%/88.9%.

**Smoothness before/after** (the corpus `s7_smoothness.csv` only holds
post-fix numbers — rebuilt fresh, no pre-fix rows kept — so the actual delta
was measured by re-running the 5 T1 dev clips against the preserved pre-fix
backup pkls, `pkl_s5_prefix_backup/`):

| variant | joint_jerk %Δ | body_jerk %Δ | skateL %Δ | fidPos %Δ |
|---|---|---|---|---|
| gmr_contact_fc | +4.6% | +6.8% | +9.1% | ~0% |
| gmr_contact_fc_sm | +28.4% | +23.2% | +1.2% | ~0% |
| medianlimb | +7.7% | +9.7% | +24.0% | ~0% |
| perframelimb | +1.7% | +4.6% | +53.7% | ~0% |

`fc_sm` takes the biggest jerk hit — expected, self-collision avoidance now
fights the Stage-A smoothing pass on frames it didn't touch before.
`perframelimb` pays the least jerk (still the safest mechanism overall) but its
skate %Δ is largest in relative terms, off a very small pre-fix base
(0.06-0.12cm) so the absolute cost is small. Worst clip across all four is
`fallAndGetUp1_subject1` — same clip flagged for the held-target range
residual above; the get-up motion triggers the most self-collision correction.
Fidelity unaffected everywhere, as expected (self-collision avoidance doesn't
touch target tracking). Not a blocker — still a fraction of the S7-T1
smoothness gap between contact-correct variants and `gmr_polished`/`gmr_raw`.

**Verdict: ship.** Old pre-fix pkls preserved at `pkl_s5_prefix_backup/` for
reference/reproducibility.

### T3b backfill — perframelimb's corpus build, and a bigger finding than expected

The plan wanted a standalone `S7-T3b` log entry once perframelimb's corpus
build landed; it built successfully (0 failures, all 77 clips) but got folded
silently into T7's rebuild instead — backfilled per the plan's own backfill
rule. The final (self-collision-fixed) numbers turned out bigger than "best
floor mechanism": at full corpus scale, POST the T7 fix, perframelimb beats
`gmr_contact_fc` on every un-gameable metric on BOTH classes, not just floor —
including locomotion range (4.12cm vs fc's 8.38cm, roughly half), where it was
never specifically targeted:

| class | variant | joint_ok% | floorPen_cm | range_cm |
|---|---|---|---|---|
| loco (43) | gmr_contact_fc | 97.93 | 2.32 | 8.38 |
| loco (43) | **perframelimb** | **98.95** | **2.24** | **4.12** |
| floor (34) | gmr_contact_fc | 88.82 | 11.75 | 18.14 |
| floor (34) | **perframelimb** | **97.60** | **6.20** | **7.39** |

Known gap: perframelimb has NO corpus-scale smoothness/jerk number
(`s7_smoothness.csv`'s variant set predates T3's fix). The only data point is
T7's 5-dev-clip pre/post-self-collision-fix DELTA (+1.7%/+4.6% jerk, smallest
of the four touched variants) — a relative number, not an absolute comparison
against fc's own jerk.

### S7-DECISION — perframelimb vs gmr_contact_fc as primary, presented not decided

Per the plan's own instruction, options laid out for Prabin rather than
auto-promoting the better numbers:
- **A. Promote perframelimb to primary.** Best numbers on record. Needs: a
  corpus smoothness pass (the gap above), probably a `_sm`-style smoothing
  companion if that pass trips T1's >50%-jerk threshold, and reframing the
  paper's "in-solve contact layer" narrative (perframelimb is fully post-hoc,
  unlike fc's in-mink-solve override).
- **B. Keep fc/fc_sm primary, report perframelimb as the strongest
  ablation/ceiling.** Lower risk (fc is validated on every axis already);
  costs the paper its best number if a reviewer compares floor-class directly.
- **C. Per-class hybrid** (fc/fc_sm loco, perframelimb floor) — splits the
  difference, adds a "why two mechanisms" question.
- **D. Close the smoothness gap first, then decide.** Lowest-regret if there's
  no deadline forcing the call now.

Full writeup: `planLogGMR.md ## S7-DECISION`.

## T6: torso/waist residual probe — negative, but surfaced a broader open item

Exploratory, 2-attempt cap per the plan. Added a `"waist"` chain to
`leg_floor_clamp.py` (waist_yaw/roll/pitch → `torso_link`, probe-only, not in
the shipped `CLAMP_TARGETS` list) — confirmed via `body_parentid` first that
`pelvis` attaches directly to `world` here (the free-joint root body), so a
waist correction can only ever reach `torso_link` and above; pelvis
penetration stays out of scope by design ("root stays frozen").

Clip selection wasn't eyeballed: swept every floor-class clip's per-BODY
worst-z (not just the whole-clip worst) and picked the 2 with the deepest
measured `torso_link` penetration (`fallAndGetUp2_subject2` -3.6cm,
`fallAndGetUp1_subject1` -3.5cm). Same sweep surfaced something unexpected:
`left_ankle_roll_link` has a far deeper residual on both clips (-18.6cm/
-12.4cm) despite being inside `CLAMP_TARGETS`' scope already.

**Attempt 1** (waist clamp, clearance-only): floorPen_cm/range_cm unchanged on
both clips (torso was never the deepest violator, so correcting it can't move
a metric it wasn't driving) — `pen_pct` improved on one clip, worsened on the
other; self-collision also worsened on both (the waist correction has zero
collision awareness of its own). **Attempt 2** (+ `avoid_self_collision=True`,
T7's fix reused): closed the induced collision regression exactly, but the
net result stayed a wash — floorPen_cm/range_cm still unchanged, joint_ok flat
to slightly worse, pen_pct still mixed. **Not shipped.**

**More important byproduct**: the -18.6cm/-12.4cm ankle residuals are each a
single isolated frame (fallAndGetUp2_subject2 t=212 of 4918; neighbors read
-5.9/-2.9/+1.9/+0.8/+0.4cm) coinciding with an unusually high active-contact
count at that exact frame (`ncon=30` vs 0-20 on neighbors) — the same class of
bug already flagged in T7's self-collision fix ("phase 2 is floor/held-blind,
can overpower phase 1's convergence within the bounded iteration budget on a
hard frame"), just showing up on a different chain/frame and a deeper
magnitude than the one instance T7 specifically flagged. Confirms the residual
is broader than previously scoped — same root cause, still open, not chased
further (same diminishing-returns reasoning T7 gave).

## T5: OmniRetarget baseline — feasibility confirmed, execution skipped (Prabin's call)

Undermind's strongest contact-aware kinematic competitor had no baseline
number. Found the actual code (`github.com/amazon-far/holosoma`,
Apache-2.0) via the OmniRetarget project page — Undermind's report didn't link
it directly. Confirmed: G1 support (`models/g1/`), LAFAN1 support
(`--data_format lafan`, a documented BVH→npy conversion path, an example
command using `dance2_subject1` — this project's own local clips), isolated
conda env (no dependency clash). README explicitly flags floor-contact
tolerance issues on LAFAN1 — the authors hit the same problem this project's
whole floor-class effort targets. Execution paused mid-sprint to prioritize
T7's self-collision fix, then explicitly skipped per Prabin's direct
instruction (2026-07-18) rather than resumed — feasibility-confirmed /
documented-exclusion is the final T5 deliverable this sprint, per the plan's
own "outcome A or outcome B, both acceptable" framing.

## Open items
- fallAndGetUp1's held-target range residual under `coll_weight=0.5`, and its
  broader form surfaced by T6 (phase 2's collision correction is chain-blind
  to held-target locks / floor convergence on any hard, high-contact-count
  frame, not just the one instance originally flagged) — not chased further
  this pass, two separate mechanism-design passes already spent on it.
- S7-DECISION (perframelimb vs fc as primary method) — presented, Prabin's
  call, not yet made.
- perframelimb's corpus-scale smoothness/jerk number — the prerequisite for
  S7-DECISION option A or D.
- T5 (OmniRetarget) — explicitly skipped this sprint, not resumed.
- T6 (conditional torso probe) — not activated this sprint.
- Neither self-collision phase is aware of held-target locks on the same
  chain — the general form of the fallAndGetUp1 residual, not just that one
  clip.

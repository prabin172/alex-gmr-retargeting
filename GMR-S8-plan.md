# GMR-S8 Plan — Physical Plausibility: Kill the Spikes, Fix the Skate, One Honest Winner

Written by Fable 2026-07-18 for Sonnet to execute. If cold, read in order:
`GMR-S7-plan.md` (standing rules + T3/T7 history), `planLogGMR.md` from
`## S7-T7` to end (the self-collision fix, the post-fix corpus rebuild, T3b
backfill, S7-DECISION, T6), `wiki/experiments/gmr-baseline-sprint-s7.md`.

---

## REVISION R7 (2026-07-18/19, post-S8-DECISION) — SPRINT CLOSED, read this first

**S8-DECISION is written** (`planLogGMR.md ## S8-DECISION`): Prabin locked
`perframelimb_smrc_rl_localground` (T6 grounding + T8 rate-limited
re-clamp) as the working baseline, accepting the 5/6 never-tradeable
scorecard rather than chasing vMax (the sole remaining loss, narrowed from
a 63-65% gap to 9.8-15.5%) further before locking. `scripts/g1/
sprint_s8_lock_final.py` produced the canonical results file
(`outputs/gmr_baseline/sprint/s8_LOCKED_perframelimb_smrc_rl_localground.csv`,
adds corpus-wide hand slip, tracked for the first time). `GMR-METHOD.md`
(new, repo root) is the method writeup — plain-language walkthrough +
full-mathematics appendix (§12) for every `[ours]` stage. Docs & discussions
pass (below) is done: `wiki/experiments/gmr-baseline-sprint-s8.md` (new),
`wiki/index.md`, `wiki/log.md`, `GMR-baseline-results.md` S8 section.

**S8 is closed.** Next gate is S9 (mimic-training pilot) — Prabin's to
start, not scoped here. Nothing below this point should be executed
without new direction; R0-R6 are kept for history only.

## REVISION R6 (Sonnet, 2026-07-18, post-T9) — READ BEFORE R5; current state

State: **T0-T9 are DONE and logged** (`planLogGMR.md` through `## S8-T9`).
T4's visual veto check (outstanding since the start of S8, flagged in
every prior revision) is finally done, on the T8 variant, alongside new
render tooling Prabin asked for: `scripts/g1/render_sidebyside.py` (twin-
panel GMR-full vs ours in one video) and `g1_model_setup.py`'s
`white_floor=True` opt-in (fixes a real bug found by inspection: the base
XML's own static floor geom and this loader's injected mocap floor sit
exactly coincident and z-fight under the default checker+edge-mark
material — that z-fighting was the "black and white madness," not a mesh
issue). 5 clips rendered/inspected (R0's 3 + the historically-worst floor
clip + this variant's own worst-vMax clip), 20 sampled frames: **PASSES
clean, no teleport/contortion/snapping on any sampled frame**, poses track
GMR-full's own timing closely, floor contact stays at 0.00cm throughout
(full detail `## S8-T9`). This removes the one non-metric blocker that
applied regardless of the scorecard. **Still not 6/6** on the
never-tradeable axes (vMax remains a narrowed-but-real loss) — S8-DECISION
still not written, presenting not deciding.

## REVISION R5 (Sonnet, 2026-07-18, post-T8) — superseded by R6 above, kept for history

State: **T0-T8 are DONE and logged** (`planLogGMR.md` through `## S8-T8`).
Two of Prabin's post-T6 ideas were tried: T7 (relax tracking / raise
smoothing regularization) was NEGATIVE and closed out (root cause:
vMax/n_spikes come from `smrc`'s re-clamp step, not the smoothing weights
— relaxing tracking just gives the re-clamp more corrective work, raising
jerk instead of lowering it, confirmed by a monotonic 2-point sweep). T8
(rate-limit the re-clamp step directly, `CorrectionRateLimiter`
`rate_limit=0.15`, same mechanism T1b tried once on the ORIGINAL clamp
pre-T6/pre-smoothing and found converts spikes to drift) — **this time it
worked**: `perframelimb_smrc_rl_localground` clears **5 of 6
never-tradeable axes** vs `gmr_heightfix` (floorPen, coll_pct, worst_float,
joint_ok all win; **n_spikes flips from loss to win/tie**, 0.00 vs
heightfix's 0.18 floor / 0.00-0.00 tie loco). vMax is the sole remaining
loss, shrunk from a 63-65% gap (T6) to 10-16% (T8). `joint_jerk_mean`
dropped too (-11.3%/-7.5%), not the cost T7 produced — confirms rate-
limiting the re-clamp attacks the real mechanism, unlike relaxing
smoothing weights upstream of it. Small real costs vs T6: worst_float
+~1cm, joint_ok -0.1 to -0.45pp (still crushes heightfix), skate +0.1-0.15cm
(T1b's "drift" effect reproduced but at ~1/10th its original magnitude,
because this rate limiter only has residual smoothing-perturbation left to
correct, not a full raw-to-floor-safe correction). **Closest any variant
has come: 5/6.** S8-DECISION still NOT written — presenting, not deciding.
T4 visual veto check still outstanding on every variant, now most
importantly on this one. `--rate 0.15` untuned/unswept (T1b's own first
choice, worked first try) — a different rate might trade the vMax gap
against skate/float further, untested.

## REVISION R4 (Sonnet, 2026-07-18, post-T6) — superseded by R5 above, kept for history

State: **T0-T6 are DONE and logged** (`planLogGMR.md ## S8-T0/T1/T2/T2c/T2d/
T2-DECISION/T3/T5/T6`). Prabin signed off on the local-grounding avenue R3
flagged; it's built and evaluated for real
(`scripts/g1/sprint_s8_t6_localground.py`, full detail `## S8-T6`).
**Result: `perframelimb_smrc_localground` clears 4 of 6 never-tradeable
axes vs `gmr_heightfix`** (floorPen win, coll_pct win, worst_float win,
joint_ok win — and joint_ok actually improves past plain smrc, 97.93→99.30
floor, 98.65→98.89 loco, not just "survives"). The remaining 2 losses
(n_spikes, vMax) are bit-identical to plain `smrc` — grounding never
touches them by construction, confirmed empirically — they're the
pre-existing smoothing/dynamics-quality gap, not a grounding problem.
**Closest any variant has come to the full bar; still not 6/6.**
S8-DECISION has NOT been written — presenting the table, not deciding.
T4 visual veto check exists only for `smrc` (not `localground`) and is
still outstanding. See `## S8-T6`'s scorecard table for the full picture.

## REVISION R3 (Sonnet, 2026-07-18, post-T3) — superseded by R4 above, kept for history

State: **T0-T4 are DONE and logged** (`planLogGMR.md ## S8-T0/T1/T2/T2c/T2d/
T2-DECISION/T3/T5`). T3's 77-clip corpus table does NOT match R2.5's
predicted shape: `perframelimb_smrc` loses 3 of R2.2's 6 never-tradeable
axes (floorPen, n_spikes, vMax) at corpus scale, not just on hard clips —
see `## S8-T3`. Prabin's follow-up hypothesis (naive per-clip grounding,
GMR's own height-shift trick, applied on top of `smrc`) was built and
re-evaluated for real (`## S8-T5`, `scripts/g1/sprint_s8_t5_grounding.py`):
it zeroes floorPen and keeps worst_float ahead of heightfix with zero cost
to coll/vMax/spikes/jerk/skate/fidelity, but collapses `joint_ok_pct`
(97.9%→32.7% floor) because a single clip-wide constant shift, calibrated
to the clip's worst *transient* frame, overshoots the tight ±3cm band
`joint_ok` requires on *stance* frames. **No variant on record clears all
six never-tradeable axes simultaneously.** S8-DECISION has NOT been
written. T4's visual teleport-veto check on the existing renders
(`s8_renders/`) is still outstanding regardless of which variant is chosen.
Next identified (not yet authorized) avenue: a LOCAL version of grounding —
shift only around each clip's handful of offending transient frames,
looping back toward T2c/T2d's windowed-repair philosophy instead of a
global constant. Awaiting Prabin's direction before further mechanism work.

## REVISION R2 (Fable, 2026-07-18, post-T2 cap) — superseded by R3 above, kept for history

State: **T0, T1, T2 are DONE and logged** (`planLogGMR.md ## S8-T0/T1/T2`).
T2 hit its 2-attempt cap without clearing the gate. Do not redo any of them.
R2 authorizes the next mechanism round and fixes the paper framing.

### R2.1 What T2 actually established (context, no action)

The two attempts decompose into two INDEPENDENT effects:
- **Contact erosion is attempt-independent:** floorPen 7.78/7.78 cm, coll
  2.15/2.14 %, joint_ok 85.05/84.90 % (attempt1/attempt2, 10-clip gate-set
  combined means vs perframelimb's 4.27 / 0.002 / 98.59). It is caused by
  smoothing the NON-locked frames: perframelimb corrects EVERY frame (swing
  clearance and collision push-out, not just stance), the lock only protects
  held frames, so the tridiagonal blend drags swing-leg corrections back
  toward the uncorrected input. joint_ok requires whole-body pen < 5mm, so
  eroded swing clearance kills it even though it is evaluated on held frames.
- **The lock/unlock toggle moves only the temporal axes:** spikes 2.00→0.00,
  vMax 59.46→36.61 rad/s — plus worst_float 3.11→5.05 cm as its one cost.

Conclusion: contact-blind temporal smoothing structurally cannot
Pareto-improve a per-frame contact-corrected motion. That is a finding (a
controlled ablation for the paper, R2.5), not a tuning miss. No further
tuning of `smooth_heldaware.py` alone.

### R2.2 Training-relevance ruling — APPROVED by Prabin 2026-07-18, with one condition

**Status: the joint_jerk demotion below is approved.** Prabin's stated
principle: the preferred trade is losing some tracking fidelity (motion not
exactly human-like) while contact and physical plausibility stay right — the
robot must still stand up. Elevated jerk is acceptable **on condition that
the T4 renders look good: no teleport-like motion, nothing that reads as
physically absurd / super-high-torque.** The T4 watch is therefore a hard
VETO axis for the winner, not just a logging step (see R2.6).

RetargetMatters (`RetargetMatters.pdf`, the GMR paper) names the
training-critical artifacts explicitly: *"foot penetration,
self-intersection, and abrupt velocity spikes are all critical artifacts
that should be avoided during retargeting"* — and its three case-study
policy failures map 1:1 (PHC ground penetration, ProtoMotions
self-intersection, GMR "Dance 5" sudden waist-value jumps). Mean joint jerk
is NOT on that list and appears in none of their evaluation metrics (success
rate, tracking error, user-study faithfulness); a PD-tracked policy
low-passes reference jitter it is not rewarded to reproduce. Therefore:

- **Never tradeable:** floorPen/pen%, coll%, n_spikes, vMax, worst_float,
  joint_ok (the un-gameable composite). Both T2 attempts violated critical
  axes → both correctly rejected; shipping either gifts the baseline a
  rebuttal.
- **Tradeable (approved):** `joint_jerk` demoted from gated (≤1.3×raw) to
  report-only with a 1.75×raw sanity ceiling. `body_jerk` stays gated at
  1.3×raw (perframelimb already passes: 263.0 vs 265.7 on the gate set).
- Related knob if a future attempt needs headroom: the fidelity axes
  (fidelity_pos / ori ≤ raw+3°) are the sanctioned release valve — losing
  tracking accuracy to preserve physical plausibility matches the approved
  principle and the method's architecture (the clamp already overrides
  tracking where they conflict). Relax there before ever touching a
  critical-artifact axis, and only with a logged rationale.

This is a gate revision by Prabin, not gate-weakening by the executor; the
no-weakening rule still binds the executor for everything else.

### R2.3 T2c — smooth → re-clamp (new mechanism, authorized, 2-attempt round)

Hypothesis: T0b's cause-A spikes are DLS instability under LARGE corrections
from raw GMR input. Re-clamping the SMOOTHED trajectory needs only cm-level
corrections from a temporally clean start, so the instability may not
re-arise — and the re-clamp restores swing clearance and collision push-out
by construction.

- Input: the on-disk `pkl_s5/*_perframelimb_sm.pkl` (these are the T2
  attempt-2 build — spike-unlock lockweights — 10 gate clips).
- Re-clamp LIMBS ONLY: the phase-1 floor/held + phase-2 self-collision
  per-frame clamp from the perframelimb build path, every frame, warm-started
  from the previous frame's applied correction. Do NOT re-run the per-frame
  root lift — the smoothed root already keeps stance pinned (locked frames)
  and free stretches smooth; re-lifting would double-correct. If the build
  path has no limbs-only entry point for an arbitrary input pkl, add a flag —
  do not change the perframelimb build defaults.
- Output suffix `_smrc`. Fairness arm stays `heightfix_sm` (smoothing only):
  the clamp IS the method under test, not a shared post-process — applying it
  to heightfix would turn the baseline into ours. State this in the docs.
- Gate: the ORIGINAL 10-axis T2 gate, unchanged, same 10 clips (extend
  `sprint_s8_t2_gate.py` with the new suffix).
- Attempt 2 (only if attempt 1 fails ≤2 axes with a DIAGNOSED cause — probe
  actual frames, lockweight-probe style, don't guess): ONE targeted knob —
  λ_smooth 20→10 if contact axes fail marginally, or T2d's local repair pass
  applied to residual spike frames if spikes reappear sparsely.

### R2.4 T2d — local spike repair, no global smoothing (fallback; PRE-APPROVED per R2.2)

If T2c hits its cap: proceed directly to T2d without waiting for further
sign-off (R2.2 is approved). Drop global smoothing entirely.

- Input: unsmoothed `perframelimb` pkls (contact axes then pass by
  construction).
- Detect: leg-DOF transitions with |Δq|·fps > 40 rad/s — deliberately BELOW
  the 60 rad/s metric threshold so the repair set is a superset of the
  counted spikes, not metric-targeted.
- Repair per spike: merge overlapping ±3-frame windows per joint; replace the
  joint's in-window values with PCHIP interpolation between the window
  endpoints; re-run the limb clamp on in-window frames only (warm-started) to
  restore any contact the interpolation broke; re-detect (max 2 iterations,
  then report what remains).
- Gate: R2.2-revised (joint_jerk report-only ≤1.75×raw; every other axis
  unchanged). Expected: contact axes ≈ perframelimb exactly, spikes → 0,
  vMax passes, joint_jerk stays ~1.4–1.7×raw and is reported as the
  consciously accepted cost.
- Docs note: the 10-clip gate set is deliberately worst-spike-biased (5 clips
  are perframelimb's corpus-worst); corpus-level perframelimb spikes are
  0.91/clip floor, 0.12/clip loco. Report both, never swap one for the other.

### R2.5 Paper-framing directives (carry into T3 table, S8-DECISION, docs)

- Present T2 attempts 1/2 as a CONTROLLED ABLATION, not a failed gate: the
  lock/unlock toggle isolates the temporal axes, the smoothing pass isolates
  the contact axes → "contact-blind smoothing cannot Pareto-improve
  per-frame contact-corrected motion" is a supporting result FOR whichever
  mechanism wins (it motivates re-clamping / targeted repair).
- Headline scorecard = RetargetMatters' own critical-artifact list. GMR-full
  (heightfix) carries 2 of the 3 critical classes: self-intersection
  inherited from raw (coll 6.34% floor / 3.85% loco — a constant root shift
  cannot change joint configs) and the float↔pen window (worst_float 16.7cm,
  range 11.8cm unchanged); its spikes equal raw (0.18/clip floor). Ours,
  post-T2c/T2d: zero critical classes. That scorecard is the paper's spine;
  joint_ok sits on top as the un-gameable composite; jerk/skate/fidelity are
  secondary reported axes.
- The consciously made trade gets ONE discussion paragraph: elevated mean
  jerk (≤1.75×raw, zero spikes, vMax ≤1.2×raw) exchanged for eliminating all
  three critical artifact classes — grounded in the baseline paper's own
  artifact→policy-failure evidence, and testable: S9 (BeyondMimic pilot, own
  plan doc after S8-DECISION) remains the arbiter and the paper gate
  (kinematic win necessary, mimic-training win sufficient).
- R1.2 (heightfix = primary baseline) and R1.3 (fairness arm reported, not
  gated) unchanged.

### R2.6 Execution order

T2c (gate → cap) → [if capped: T2d, pre-approved] → T3 (corpus build for the
winner suffix `_smrc` or `_repair`, replacing `perframelimb_sm` in the R1/T3
column layout; the sm attempts stay as ablation rows from existing gate-set
data, no corpus build for them) → T4 renders → S8-DECISION → docs.

T4 addition (Prabin's condition on the jerk demotion): the render watch is a
HARD VETO for the winner. For each rendered clip, explicitly answer: any
teleport-like frame? any motion that reads as physically absurd /
super-high-torque? If yes → the variant does NOT ship regardless of its
metric table; log which clip/frame and STOP at the T4 boundary for Prabin.

Supervision protocol unchanged: stop at every phase boundary, report numbers
before continuing. ONE log entry per phase (`## S8-T2c`, and `## S8-T2d` if
reached).

---

## REVISION R1 (Fable, 2026-07-18, post-T1) — READ THIS FIRST, IT SUPERSEDES T2–T4 BELOW

State: **T0 and T1 are DONE and logged** (`planLogGMR.md ## S8-T0`, `## S8-T1`).
Do not redo them. What changed and what Sonnet executes now:

### R1.1 T1 outcome (context, no action)
The rate limiter (T1b, suffixes `rl`/`rl2`, 2-attempt cap spent) *converts
spikes into drift*: `perframelimb_rl` reaches raw parity on vMax/n_spikes but
gives back float (2.9→6.1cm), range (7.3→10.3cm) and skate (0.44→1.56cm).
`rl`/`rl2` are hereby **demoted to ablation rows** — keep their pkls and CSV
rows, build nothing more on them. The candidate going into T2 is
**`perframelimb` (unlimited)** — it wins every contact axis AND skate/float;
its only losses are temporal (n_spikes 2.08/clip floor, vMax 1.6× raw,
joint_jerk 1.7× raw, body_jerk 1.4–1.6× raw).

### R1.2 BASELINE CORRECTION (Prabin, standing rule — violations = redo)
The primary baseline in EVERY table from now on is **`gmr_heightfix`**,
labeled "GMR-full" (GMR's complete method = raw retarget + per-clip
constant-height grounding). `gmr_raw` appears only as an ungrounded reference
column. Rationale (put this in the final docs' discussion): our method has no
separate grounding stage — grounding is baked into the contact clamp — so the
fair comparison is against GMR *with* its grounding. GMR-full kills
penetration (pen 0.45% floor) by floating everything: worst_float 16.7cm,
joint_ok 0.30% floor / 33.0% loco (gate-set means). A constant shift can only
slide the float↔pen window (range_cm unchanged at 11.8); ours shrinks it
(7.3). That tradeoff IS the motivation story.
Note: heightfix's smoothness equals raw's by construction (constant root-z
shift, joint velocities untouched) — but compute its actual rows anyway
(fidelity_pos DOES change with the shift; don't assert, measure).

### R1.3 Fairness rule for smoothing (Prabin)
Any smoothing stage must be applied to BOTH arms and be contact/held-aware:
`gmr_heightfix_sm` (GMR-full + same smoother) and `perframelimb_sm` (ours +
same smoother). Expected: heightfix gains ≈ nothing (already smooth) — that is
the point, it shows our gains don't come from smoothing. If it does change,
report honestly.

### R1.4 Root-bounce finding (feeds T2's gate)
Prabin's concern, confirmed by measurement: the per-frame root lift makes the
body move more than raw at low frequency even though the lift curve is
15-frame-MA smoothed (T0b cause C = 0 spikes): body_jerk_mean 299 vs raw 215
(1.4× floor), 374 vs 230 (1.6× loco). Part of that motion is the fix working
(raw's root height is wrong wherever limbs penetrate), part is artifact. T2's
smoother must include the root DOFs on non-held stretches, and T2's gate now
includes body_jerk. If the gate fails on body_jerk specifically, the attempt-2
knob is a stronger/wider low-pass on the lift curve itself (currently 15-frame
MA in `polish_median_limbwise.py`).

### R1.5 Render deliverables to REPO ROOT (Prabin will push, view, delete)
Renders go in a repo-root folder `s8_renders/` (NOT outputs/ — outputs/ is
git-ignored; repo root is intentional, Prabin pushes it to view remotely and
deletes later). Two batches:
- **R0, immediately, before T2 work starts:** `gmr_heightfix` vs `perframelimb`
  (existing pkls) on 3 clips: `walk3_subject1`, `fallAndGetUp1_subject1`,
  `ground1_subject1`, via `scripts/g1/render_penetration_annotated.py`.
  Filenames `{clip}__{variant}.mp4`. Keep sizes reasonable (≤ ~25MB each; drop
  resolution/fps if needed).
- **T4, at the end:** the T2 winner (`perframelimb_sm`) same 3 clips + the
  worst remaining floor clip by floorPen from the T3 corpus table.

### R1.6 Execution order for Sonnet
R0 renders → T2 (held-aware smoothing, both arms) → T3 (corpus + one honest
table) → T4 (final renders, watch them) → S8-DECISION → docs & discussions
(per the rewritten sections below). Supervision protocol unchanged: stop at
each phase boundary, report numbers, Fable/Prabin review before you continue.
S9 (BeyondMimic mimic-training pilot on the RTX 5080, 3 clips ×
{GMR-full+sm, ours+sm}) is approved in principle but NOT Sonnet's to start —
it gets its own plan doc after S8-DECISION. The paper gate remains: ours must
win in mimic training for this to become a paper.

---

## Why S8 exists — the honest post-fix audit (Fable, 2026-07-18)

S7-T7's self-collision fix forced an honest corpus rebuild, and the rebuilt
numbers changed the situation. Prabin's read: the earlier "positive" results
were achieved by per-frame geometric lifting of penetrating limbs with no
regard for anything else. That read is correct in a precise, fixable way.
What survives and what doesn't:

**What survives (post-fix, 77-clip corpus, class means):**

| variant | class | joint_ok% | floorPen cm | pen% | coll% | coll_peak cm |
|---|---|---|---|---|---|---|
| gmr_raw | floor | 80.64 | 15.29 | 23.38 | 6.337 | 5.66 |
| gmr_heightfix | floor | 0.19 | 2.76 | 0.38 | 6.337 | 5.66 |
| gmr_polished | floor | 0.36 | 2.56 | 0.33 | 5.835 | 5.04 |
| gmr_contact_fc | floor | 88.82 | 11.75 | 7.63 | 0.048 | 0.63 |
| **perframelimb** | floor | **97.60** | **6.20** | **1.30** | **0.013** | **0.27** |
| gmr_raw | loco | 91.52 | 5.15 | 3.03 | 3.853 | 5.05 |
| gmr_contact_fc | loco | 97.93 | 2.32 | 0.36 | 0.009 | 0.13 |
| **perframelimb** | loco | **98.95** | **2.24** | **0.29** | **0.002** | **0.10** |

The un-gameable joint metric win is real and honest, and self-collision is
now >99% BELOW every baseline (the baselines inherit gmr_raw's 3.8–6.3%
self-collision; ours is 0.001–0.05%). Nobody is screwed on the contact story.

**What does NOT survive — motion plausibility (s7_smoothness.csv, class means):**

| variant | class | joint_jerk | skate_mean cm | skate_max cm | vMax rad/s | n_spikes/clip |
|---|---|---|---|---|---|---|
| gmr_raw | floor | 5.0e3 | 0.44 | 4.1 | 34.0 | 0.18 |
| gmr_contact_fc | floor | 7.9e3 | 1.30 | 16.9 | 78.7 | 7.88 |
| gmr_contact_fc_sm | floor | 1.4e3 | 3.89 | 22.1 | 61.5 | 3.56 |
| medianlimb | floor | 8.0e3 | 0.87 | 23.5 | 84.7 | 10.1 |
| perframelimb | — | **NO DATA** | — | — | — | — |

Three distinct failure modes, all now measured or witnessed:

1. **Single-frame velocity spikes.** fc/medianlimb hit vMax 79–85 rad/s with
   4–10 spikes per clip (raw: ~0.2). Root cause is known: the clamp is a
   per-frame independent DLS correction with no temporal coupling — it toggles
   on/off frame to frame, and self-collision phase 2 is floor/held-blind and
   can overpower phase 1 on hard frames (S7-T6 measured an isolated one-frame
   -18.6cm ankle at a frame with ncon=30; S7-T7 flagged the same class at
   fallAndGetUp1 t=2251). A 80 rad/s frame is physically impossible motion;
   a reviewer or a downstream policy will reject it.
2. **Smoothing destroys held-contact stationarity.** fc_sm fixes jerk
   decisively (1.4e3, below raw's 5.0e3) but its blind Stage-A pass drags
   held feet: skate_mean 3.9cm vs raw's 0.44cm, ori fidelity 13.9° vs 7.0°.
   We smooth the very DOFs whose whole point is to stay put.
3. **perframelimb — the best contact variant on record — has zero smoothness
   numbers.** Its mechanism (per-frame root lift + per-frame limb clamp) is
   the most per-frame-independent of all variants; assume it spikes at least
   as badly as fc until measured.

**S8's single goal:** produce ONE variant that keeps the joint_ok/coll win
AND is at-or-near `gmr_raw` on jerk, skate, vMax, and n_spikes. That variant
supersedes S7-DECISION (which is hereby paused — do not pick A/B/C/D from an
incomplete picture; S8's outcome IS the decision input).

NOT in S8 scope: OmniRetarget execution (T5 stays skipped per Prabin),
BeyondMimic/policy eval, venue decisions, any new contact mechanism. S8 is
about making the existing winners physically plausible, nothing else.

---

## Standing rules (unchanged from S6/S7 — violations corrupt the paper)

- **Baseline integrity:** `gmr_raw`/`gmr_heightfix`/`gmr_polished` generators
  and pkls are never touched. New variants are new pkls + new CSV rows.
- **2-attempt tuning cap** per gate per mechanism. Hit the cap → log numbers
  honestly, stop, move on. No silent third attempts.
- **Never gameable-metric-optimize:** joint_ok_pct and the smoothness gates
  must move together; a change that trades one for the other is a finding to
  log, not a knob to hide.
- All Python via `conda run -n gmr python ...`. Corpus builds are resumable
  (skip-if-done rows) and background-safe, same pattern as
  `sprint_s6_corpus.py`.
- Existing pkls under `outputs/gmr_baseline/sprint/pkl_s5/` are the current
  post-fix state — do NOT overwrite them; new variants get new suffixes.

## Documentation policy for S8 (Prabin's explicit instruction)

Document at phase boundaries, not per-step. Concretely:
- ONE `planLogGMR.md ## S8-Tn` entry when a phase completes or hits a
  decision/cap — dense tables, minimal prose. Nothing mid-phase.
- `wiki/experiments/gmr-baseline-sprint-s8.md` + `wiki/index.md` +
  `wiki/log.md`: written ONCE, at sprint end (or at S8-DECISION, whichever
  comes first). Do not touch the wiki mid-sprint.
- `GMR-baseline-results.md`: only after S8-DECISION, one section.
- No re-summarizing prior entries; link to them. If a result is already in a
  CSV, the log entry cites the CSV and shows only the class-mean table.

---

## Phase T0 — measure before fixing  [P0, do first, no mechanisms]

### S8-T0a: perframelimb smoothness at corpus scale
- Extend the `sprint_s7_smoothness.py` run to add `perframelimb` rows to
  `s7_smoothness.csv` (77 clips; pkls exist at `pkl_s5/*_perframelimb.pkl`).
  This is FK + finite differences — minutes, not hours. It is also
  S7-DECISION option D, so it retires that open item for free.
- Log the class-mean row into the T0 log entry alongside the table above.

### S8-T0b: spike attribution
- Take the ~5 worst clips by n_spikes for fc and perframelimb (from the CSV,
  not eyeballed). For each spike frame (the frames `n_spikes` counts), classify
  the cause by direct instrumentation — the categories we already know exist:
  (a) clamp activation toggling on/off between adjacent frames,
  (b) self-collision phase 2 overpowering phase 1 (check: does the spike frame
      have unusually high `ncon`? does re-running with
      `avoid_self_collision=False` remove it?),
  (c) perframe root-lift discontinuity (perframelimb only — diff the lift
      curve),
  (d) held-release ramp interaction (the S7-T3 bug class — check `held`
      transitions at the spike frame).
- Deliverable: one table — cause × count × worst-magnitude. This decides how
  much of T1 is needed and in what order. If one cause is >80% of spikes,
  say so and prioritize only that in T1.
- Budget: this is diagnosis, not fixing. Half a day max. Log `## S8-T0`.

## Phase T1 — temporal coherence in the clamp  [P0, the core fix]

Goal: n_spikes → raw level (≤0.5/clip class mean) and vMax ≤ 1.2× raw,
WITHOUT giving back the joint_ok/floorPen/coll numbers (each stays within
1.0 point / 0.5cm / 0.01% of the post-fix values). Attack in T0b-priority
order; the mechanisms below are the candidate set, not a mandate to build all:

### T1a: phase-2 acceptance check (fixes cause b — the known -18.6cm class)
- In `leg_floor_clamp.clamp_limb`: after the phase-2 self-collision step,
  re-evaluate the phase-1 objective (floor clearance + held-target error).
  If phase 2 worsened it beyond a small epsilon, either (i) scale the phase-2
  step down (backtracking line search on the collision step, 2–3 halvings),
  or (ii) re-run a bounded phase-1 pass after phase 2 (one interleave, not a
  loop). Pick ONE of (i)/(ii) as attempt 1; the other is attempt 2.
- This is the third mechanism-design pass on phase-1/phase-2 interaction —
  S7 already spent two and stopped. The difference now: T0b will have
  measured exactly which frames fail and why, so this is targeted, not
  exploratory. If both attempts fail the gate, log and STOP; do not design a
  third mechanism — escalate to Prabin instead.
- Verify the fix on the two known bad frames first (fallAndGetUp2_subject2
  t=212, fallAndGetUp1_subject1 t=2251), then the 5 dev clips.

### T1b: activation continuity (fixes cause a)
- The clamp currently applies a full correction the first frame a violation
  appears and nothing the frame before. Candidates (pick by T0b evidence):
  - warm-start each frame's DLS from the previous frame's applied correction
    delta (not just the raw pose), plus a global per-iteration `max_dq` cap —
    NOTE: S7-T3 already showed a global `max_dq` default regresses Phase A's
    legitimate large corrections, so gate this per-variant, don't flip the
    default;
  - blend corrections in/out over ±k frames (ramp the applied delta near
    activation boundaries, the same trick the held-release ramp already uses).
- Same gate, same 2-attempt cap.

### T1c: perframe lift smoothing (cause c, perframelimb only)
- If T0b shows the root-lift curve itself spikes: low-pass the per-frame lift
  BEFORE applying it (the S6-B2 fix already smooths it once — measure whether
  the residual is at held-release discontinuities and widen/relocate the
  filter accordingly). Cheap, self-contained, same gate.

Each T1 sub-fix is validated on the 5 dev clips (walk1_subject1,
walk3_subject1, run2_subject1, ground1_subject1, fallAndGetUp1_subject1)
plus the T0b worst-spike clips before any corpus build. ONE log entry
`## S8-T1` when the phase is done, covering all sub-fixes attempted.

## Phase T2 — held-aware smoothing, BOTH arms  [P0, the core remaining fix — per R1, this replaces the original T2]

Input candidate: `perframelimb` (unlimited). Target: kill its temporal losses
(n_spikes 2.08/clip floor, vMax 1.6× raw, joint_jerk 1.7× raw, body_jerk
1.4–1.6× raw) without touching its contact/skate/float wins.

Mechanism (new script, e.g. `scripts/g1/smooth_heldaware.py` — do NOT reuse
`smooth_then_clamp.py`'s blind design, and do not modify `polish_gmr_pkl.py`):
- Attempt 1: exclude held segments from smoothing — during frames where a
  foot is held, lock that leg chain's DOFs AND the root to their input
  (clamped) values; smooth only the free DOFs. Include the root in the
  smoothed set on non-held stretches (R1.4 — this is what absorbs the
  root-lift bounce). Ramp the lock in/out over the same ramp window the clamp
  uses (5 frames), or the lock boundary itself becomes a new discontinuity.
- Attempt 2 (if 1 fails gate): iterate smooth→re-clamp with decaying smoothing
  strength (2–3 rounds) so each re-clamp correction is too small to
  re-introduce jerk; if the specific failure is body_jerk, the alternative
  attempt-2 knob is widening/strengthening the lift-curve low-pass in
  `polish_median_limbwise.py` (R1.4).
- Apply the SAME smoother to `gmr_heightfix` → `gmr_heightfix_sm` (R1.3). For
  the heightfix arm there are no clamped values; lock held-leg DOFs + root to
  their existing (floating) values — the smoother must not "fix" heightfix's
  float, only preserve it while smoothing free DOFs. Its numbers are reported,
  not gated (it is a fairness arm, not a candidate).

Gate for `perframelimb_sm`, on the 5 dev clips + the 5 perframelimb
worst-spike clips (obstacles4_subject3, walk2_subject3, obstacles5_subject3,
aiming1_subject4, pushAndFall1_subject4), class means, all vs `gmr_raw`
(= heightfix smoothness):
- n_spikes ≤ 0.5/clip AND vMax ≤ 1.2× raw AND joint_jerk ≤ 1.3× raw AND
  body_jerk ≤ 1.3× raw;
- joint_ok within 1.0 point of unsmoothed perframelimb, floorPen within
  0.5cm, coll_pct within 0.05 points, worst_float within 1.0cm, skate_mean
  ≤ 2× raw, ori fidelity ≤ raw + 3°.
2-attempt cap. Both attempts fail → log honestly, STOP, escalate to Prabin
(do not weaken the gate, do not design a third mechanism).
ONE log entry `## S8-T2` at the phase boundary.

## Phase T3 — corpus rebuild + the single honest table  [P0 once T2 gates]

- Build at 77-clip corpus scale: `perframelimb_sm` and `gmr_heightfix_sm`
  (new pkl suffixes, resumable script per the `sprint_s6_corpus.py` pattern;
  post-hoc smoothing = fast). Do not rebuild anything else.
- Evaluate ALL 13 axes for every variant in the table — BOTH CSV families
  PLUS float/range (via `sprint_s6_range_summary.py` machinery): joint_ok,
  floorPen, pen%, coll%, coll_peak, worst_float, worst_pen, range,
  fidelity_pos, fidelity_ori, joint_jerk (+body_jerk), skate, vMax, n_spikes.
  A variant missing any axis does not exist for decision purposes.
- Fill known holes in existing CSVs while at it: `gmr_heightfix` has no
  smoothness rows (compute them — don't assert they equal raw's) and no
  fidelity rows.
- Table columns, in this order: `gmr_raw` (ref) | `gmr_heightfix` (GMR-full,
  PRIMARY baseline) | `gmr_heightfix_sm` | `perframelimb` | `perframelimb_sm`.
  Ablation rows below the fold: `gmr_contact_fc`, `perframelimb_rl`,
  `gmr_contact_fc_sm`, `medianlimb` (existing CSV data only, no new builds).
- Deliverable: ONE class-split table, all variants, all axes, in
  `planLogGMR.md ## S8-T3`.

## Phase T4 — visual verification  [P0, cheap, do not skip] — DONE (`## S8-T9`)

- S6's lesson: two real bugs were only ever caught by watching renders.
- Render `perframelimb_sm` on the R0 3 clips + the worst remaining
  floor-class clip by floorPen from the T3 table, with
  `render_penetration_annotated.py`, side-by-side vs `gmr_heightfix` where
  the tooling allows. Save to repo-root `s8_renders/` (R1.5 naming).
- WATCH them. Log one line per clip: clean / artifact seen (what, which
  frame). An artifact the metrics missed is a finding, not a failure — log it.

Done on the T8 variant (`perframelimb_smrc_rl_localground`), not just
`perframelimb_sm`, per Prabin's ask for real side-by-side renders + a
white floor. New `scripts/g1/render_sidebyside.py` (twin-panel, both
GMR-full and ours in one video) + `g1_model_setup.py`'s opt-in
`white_floor=True`. 5 clips rendered (R0's 3 + the historically-worst
floor clip + this variant's own worst-vMax clip), 20 sampled frames
visually inspected. **Result: PASSES, no disqualifying artifact** — see
`## S8-T9` for the full writeup. Renders in `s8_renders_t8/`.

## S8-DECISION — WRITTEN (`planLogGMR.md ## S8-DECISION`, 2026-07-18/19): locked at 5/6, vMax accepted as an open cost

T3's table didn't clear the bar cleanly (`## S8-T3`), T5's global-shift
grounding mechanism made it worse on one axis (`## S8-T5`, joint_ok
collapses), T6's local-shift grounding got 4/6 (`## S8-T6`), T7's
smoothing-weight relaxation was negative (`## S8-T7`), and T8's rate-
limited re-clamp on top of T6 got furthest: 5/6, with the sixth (vMax)
narrowed from a 63-65% gap to 9.8-15.5% rather than closed (`## S8-T8`).
T4's visual veto check passed clean on this exact variant (`## S8-T9`).
**Prabin's call: lock it at 5/6 rather than continue chasing vMax** —
`scripts/g1/sprint_s8_lock_final.py` produced the canonical results file,
`GMR-METHOD.md` (new) is the method writeup. `planLogGMR.md ## S8-DECISION`
has the full table, the honest pareto (what vMax still costs), confirms
this supersedes S7-DECISION's options A–D (S8 = A then D, in that order —
perframelimb promoted, then its smoothness/jerk gap measured and worked
down, not closed to parity), and confirms S9 (mimic pilot) is the next
gate — this kinematic lock is necessary but NOT sufficient for the paper.

## Docs & discussions (after S8-DECISION, one pass — DONE, 2026-07-18/19)

- `wiki/experiments/gmr-baseline-sprint-s8.md`: the sprint story — T0
  measurements, T1 spike↔drift tradeoff finding, T2 mechanism, T3 table, T4
  render verdicts.
- Propagate the R1.2 baseline correction EVERYWHERE it matters: the new wiki
  page, `wiki/results/metrics.md` (if it presents comparisons),
  `GMR-baseline-results.md` — GMR-full/heightfix as primary baseline, raw as
  reference. Add a short discussion subsection in `GMR-baseline-results.md`
  covering: (a) why heightfix is the fair baseline (grounding baked into our
  clamp), (b) the constant-shift float↔pen window argument (range 11.8cm
  unchanged vs ours 7.3cm), (c) the T1 finding that rate-limiting converts
  spikes into drift — same energy, different axis, (d) smoothing fairness
  (both arms, held-aware) and what heightfix_sm showed.
- `wiki/index.md` updated, one `wiki/log.md` line.
- `GMR-baseline-plan.md`/`planLogGMR.md`: nothing beyond the phase entries
  already specified.

## Failure handling

If T1 hits its caps without reaching the spike gate: S8 still ships T0's
measurements, the phase-2 acceptance findings, and an honest T3 table of
whatever improved — the paper's story becomes "contact correctness with a
quantified, bounded plausibility cost," which is defensible; "unmeasured
spikes" is not. Do NOT respond to a failed gate by weakening the gate or by
optimizing the metric definition. Escalate to Prabin with the numbers.

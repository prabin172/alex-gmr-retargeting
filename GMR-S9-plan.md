# GMR-S9 Plan — Corpus-Visible Motion Artifacts: Flicker, Floating, Standing-Still Wobble

Written 2026-07-19, based on this session's visual review + live diagnostic/
prototyping work (all done in-session, described below as T0). If cold, read
in order: `GMR-METHOD.md` (the locked S8 pipeline, especially §§5-8/12 — the
per-frame clamp, held-aware smoothing, re-clamp, rate-limit, local grounding
chain this whole plan is about), `GMR-S8-plan.md`'s S8-DECISION (why
`perframelimb_smrc_rl_localground` is the current baseline and what it still
loses on — vMax).

---

## Why S9 exists

S8 locked `perframelimb_smrc_rl_localground` on aggregate, corpus-scale
metrics (5/6 never-tradeable axes beat `gmr_heightfix`, vMax left open as a
narrowed-but-real cost). Prabin then watched the actual renders. The
aggregate numbers didn't fully capture what's visible: joint flicks a bit
worse than GMR on outlier clips, `sprint1_subject4` (sprint) specifically
looks wrong — reduced leg lift, a floating appearance — plus a foot flicker
before the motion even starts (idle stance), and a hip in/out rotation
wobble when the robot is simply standing still.

This is not just visual impression — `sprint1_subject4` is a genuine,
measurable corpus outlier (`s8_t3_full_corpus.csv`):

| metric | `sprint1_subject4` (locked) | `gmr_heightfix` | corpus-average gap |
|---|---|---|---|
| worst_float_cm | 20.62 | 8.98 | ~5.6-7.5 vs ~6.6-18.0 (method doc §11) |
| vMax_rad_s | 47.9 | 31.1 (54% gap) | 9.8-15.5% gap |

**S9's goal**: close the gap between what the aggregate table says and what
the renders show, on the specific mechanisms diagnosed below — without
re-opening S8's already-won axes (floorPen, coll_pct, joint_ok).

---

## T0 — done this session (diagnostic + first mechanism; keep, do not redo)

**Root cause of the worst vMax event** (`sprint1_subject4` t=6306, 47.9
rad/s, matches the CSV exactly): traced raw GMR → `perframelimb` → final
variant across t=6294-6311. GMR's raw retarget is flat there
(`left_hip_yaw` 0.04-0.06 rad, `right_ankle_pitch` 0.07-0.10 rad — no real
motion happening). `perframelimb`'s per-frame clamp already shows the full
oscillation on the SAME flat input: `left_hip_yaw` alternates between
~+0.9-1.3 rad and ~-0.05 to -0.4 rad frame to frame; `right_ankle_pitch`
alternates between exactly its hard upper joint limit (0.5236 rad = 30°)
and ~-0.5 to -0.8 rad. This is a DLS solution branch-flip: `clamp_limb`'s
phase-1 solve is a per-frame INDEPENDENT minimum-norm solve over a
redundant 6-DOF chain (1-3 row task) with no memory of the previous
frame's own posture — two frames with a near-identical target can land on
two different null-space solutions. Confirmed this is the same DOF
(`hip_yaw`) flagged separately for the standing-still in/out wobble — same
mechanism, different clip.

**Ruled out local time-warping ("adding frames") for THIS defect class**:
`clamp_limb` computes each frame's correction as a pure function of that
frame's own input pose — zero dependence on neighboring frames or frame
spacing. Since the raw target here is already flat, there's no real motion
to spread over more time; adding frames wouldn't change what any single
frame's independent solve computes. Time-warping might still be the right
lever for a genuinely large, real, single-frame-necessary correction
elsewhere in the corpus — none has been found yet. Not built; not
prioritized below unless T1-T3 leave a residual that looks like this
class.

**Built + tested: null-space posture-continuity regularization.**
`leg_floor_clamp.clamp_limb` gained opt-in `q_prev_chain`/`posture_weight`
params — biases `dq` toward the previous frame's OWN chain posture via
null-space projection (`dq = dq_bias + J^+(e - J@dq_bias)`), so the primary
floor/held task is still solved exactly and only the leftover null-space
freedom gets pinned toward temporal continuity. Default `None` = byte-
identical no-op (verified: reproduces the shipped
`s8_t3_full_corpus.csv` row for `sprint1_subject4` exactly with the flag
off). `polish_median_limbwise._limbwise_pass` gained matching opt-in
`posture_continuity`/`posture_weight`, threading the previous frame's own
post-clamp qpos through every phase-1 call. Dev probe:
`scripts/g1/posture_reg_probe.py`.

Result on `sprint1_subject4` (posture_weight=1.0):

| | off (shipped) | on |
|---|---|---|
| left_hip_yaw @ 6294-6311 | flips between two branches | **stable, no flip** |
| right_ankle_pitch @ same | flips between hard limit and ~-0.5 to -0.8 | **still flips** |
| vMax (whole clip) | 47.9 rad/s | **40.8 rad/s (-15%)** |
| worst_float | 20.62cm | 22.59cm (**+9%, worse**) |
| joint_ok% | 71.7% | 73.1% (flat) |

**Partial win**: real vMax reduction, hip-yaw flip cleanly fixed. Ankle
flip persists (near a hard limit, the set of DOFs that's "free" vs
"task-required" can itself flip depending on which side of the limit you
started from — a kinematic branch point, not a null-space preference, so a
pure previous-frame bias can't out-vote it). Float regressed slightly —
pinning hip_yaw pushes more correction burden onto Z. Real trade, not
free.

**Confirmed via code read** (`smooth_heldaware.py`): the held-aware
smoother's tracking target `x_t` is whatever qpos came INTO that pipeline
stage (`perframelimb`'s own clamped output), never the original human/raw-
GMR target. Only GMR's own first-stage `mink` solve ever looks at human
landmarks. Practical implication for T3/T4 below: a smoothing pass added
AFTER the final clamp would NOT drag the trajectory back toward the human
target and undo contact correctness — that channel doesn't exist
downstream of GMR's own solve. It would still be geometry-blind (no
floor/collision term at all today), which is the real gap T4 targets.

**UPDATE (2026-07-19, same day, T0-gate + T1 both run to their cap — see
`planLogGMR.md` `## S9-T1` and `## S9-T0-gate` for full numbers):**

- **T1 (joint-limit-aware null-space repulsion, below): RAN, FAILED its
  gate at the 2-attempt cap.** Pushing the ankle-limit repulsion harder
  made the target clip's vMax worse, not better (a kinematic branch point
  at the joint limit, not a null-space preference a bias term can out-vote
  regardless of weight). Code stays in (`limit_margin`/`limit_weight`,
  default 0.0 no-op), not enabled by default.
- **T0's own open item (blanket `posture_weight=1.0` regressing 4/5 dev
  clips) RESOLVED via gating, not weight-sweeping**: `posture_gate_lo`/
  `posture_gate_hi`/`raw_gate_qpos` (gate `posture_weight` per chain per
  frame by that chain's TRUE raw-signal frame-to-frame delta — full
  weight only when the human motion is genuinely near-static). 2-attempt
  cap spent (`lo=0.02/hi=0.05` passes the dev-clip check; widening to
  `hi=0.065` to chase more of the target win FAILS, opens a new
  regression elsewhere) — `lo=0.02/hi=0.05` is the validated choice.
- **Full 77-clip corpus verdict on the gated mechanism: MIXED, NOT
  shipped as default.** vMax improvement generalizes past the dev set
  (-8.1% floor class, -3.5% loco class, corpus-wide) — but jerk
  (`joint_jerk_mean`/`body_jerk_mean`, +6.6%/+13.1% floor class) and foot
  slip (`skate_left/right`, +13-21% floor class) get WORSE on average, an
  axis the narrow 6-clip dev gate never tracked and so never caught. Skate
  is the more concerning of the two: `gmr_heightfix` already beats this
  pipeline's own clamp mechanism on skate (this pipeline's phase-1/phase-2
  DLS clamp has no zero-slip guarantee, unlike Alex's Stage-B contact QP),
  and this mechanism widens that gap further — on the exact axis most
  central to a contact-aware method's own story. Open item, not yet
  root-caused: which clips/frames drive the jerk/skate cost, and whether
  it's a fixable side-effect (narrower gate, per-DOF instead of
  whole-chain-max delta) or an inherent trade of any null-space
  posture-continuity term. Not re-attempted past the 2-attempt cap;
  escalated to Prabin.

---

## Standing rules (reused from S8, still binding)

- Baseline integrity: `gmr_raw`/`gmr_heightfix`/`gmr_polished` generators
  and pkls untouched. New mechanisms are new opt-in params + new pkl
  suffixes, default off = byte-identical to shipped.
- 2-attempt tuning cap per gate per mechanism. Hit the cap → log honestly,
  stop, escalate to Prabin. No silent third attempts.
- Never gameable-metric-optimize: a change trading one never-tradeable
  axis for another is a finding to log, not a knob to hide.
- **New for S9**: every mechanism's before/after must be shown on the
  SPECIFIC diagnosed frame window (not just the aggregate table) before
  it's declared working — S8 locked on aggregates alone, and this sprint
  exists precisely because the aggregate table hid what the renders
  showed.
- All Python via `conda run -n gmr python ...`. Don't overwrite existing
  `pkl_s5/` files — new suffixes only.

---

## Phase T1 — joint-limit-aware null-space term (closes T0's residual)

**STATUS: RUN, FAILED at the 2-attempt cap (2026-07-19) — see
`planLogGMR.md` `## S9-T1`. Do not re-attempt without a new mechanism
idea, not just a weight retune** (regressions grew monotonically with
`limit_weight`, and the target metric moved the wrong direction — a
kinematic branch point at the joint limit, not something a null-space
bias can out-vote). Left below for the original problem statement.

Goal: fix `right_ankle_pitch`'s (and any other joint's) hard-limit
bang-bang that posture-continuity alone didn't reach.

Mechanism: a second null-space bias — repel from `lo`/`hi` proportional to
proximity (not just attract toward the previous frame's posture),
combined with T0's posture-continuity term (order/weighting is the design
question — try posture-continuity first with the limit-repulsion breaking
ties only near a limit, since T0 showed pure previous-frame bias alone
isn't enough there).

Gate: on `sprint1_subject4`'s diagnosed window (t=6294-6311) + the S8-T0b
5 dev clips — `right_ankle_pitch` no longer alternates between its hard
limit and a free value frame to frame; vMax improves further without
`worst_float`'s current +9% cost growing (ideally shrinking it, since less
of the correction burden should need to route through hip_yaw/Z once the
ankle stops flip-flopping).

## Phase T2 — ramped/soft floor-margin clearance

`clamp_limb`'s clearance-only branch is a binary gate: no correction above
`floor_margin`, snap-exactly-to-margin below it. Replace with a soft
activation zone — correction magnitude scales smoothly with proximity to
the floor instead of an on/off step. Swing-phase (non-contact) limbs get a
small standing clearance buffer instead of being pulled exactly to margin;
only frames the canonical-human contact labels actually mark as a plant
snap tight.

Targets: the pre-motion idle-frame foot flicker (repeated threshold-
crossing chatter at/near `floor_margin`) and the general "joint flicks a
bit more than GMR" complaint.

Gate: on an idle-stance repro window (find one via the same held-mask
tooling used for T0/T1) + the 5 dev clips — chatter at the floor_margin
boundary disappears without regressing floorPen or joint_ok.

## Phase T3 — raw-velocity-gated smoothing weight ("smooth the error, not the motion")

Goal: stop the held-aware smoother from flattening genuine fast motion
(working hypothesis for `sprint1_subject4`'s "leg doesn't lift enough"
complaint) while still killing clamp-introduced chatter.

Mechanism: make `λ_smooth` (or `λ_track`) per-frame/per-joint data-driven
instead of the current constant — reduce smoothing weight where the RAW
GMR signal itself has high velocity (real motion, protect amplitude);
increase it where raw velocity is near-zero but the corrected signal
diverges heavily from raw (high-confidence artifact — exactly the
flat-raw/jumpy-corrected signature already measured at `sprint1_subject4`
t=6294-6311 in T0). Current `build_lock_weights` only varies weight by
held/not-held status; this adds a second, independent signal.

Gate: `sprint1_subject4`'s sprint-phase leg-lift amplitude should not
shrink vs `perframelimb` (pre-smooth) at genuine fast-motion frames, while
jerk at the diagnosed chatter window still drops at least as much as plain
`λ_smooth=20` achieves today.

## Phase T4 — joint constrained-QP smoothing (Stage-B port) — scope only, do not build without go-ahead

The architecture-level fix for "whichever stage runs last wins, nothing
global runs after the final geometric correction": port the Alex
pipeline's Stage B (`solve_global_trajectory_opt_contactfirst.py` — floor-
collision rows + self-collision rows + tracking + smoothness, ALL in one
QP over the whole trajectory, not sequential passes) to G1's leg/arm
chains, replacing the current `clamp → smooth → re-clamp → rate-limit →
localground` chain of sequential patches with one joint optimization.

This is a genuinely bigger piece of work, not a quick prototype — some
groundwork exists (the parked E4 MVP, `scripts/g1/stage_b_g1.py`, ported
Stage B's contact-anchoring machinery to G1 already) but E4 was foot-slip-
only with no smoothness term, so the row-builders need real extension, not
just reuse. Decide whether to start this AFTER T1-T3 land: if the smaller
patches close vMax/float/flicker close enough, T4 may not be worth its
cost; if a real residual remains (especially vMax), this is the principled
fix rather than a fourth round of sequential patching.

---

## Corpus-scale validation (every phase)

Each of T1-T3, once it clears its own dev-clip gate, needs the same
visual-first check that started this sprint — do not trust the aggregate
table alone. Re-render `sprint1_subject4`, an idle-stance clip, and a
standing-still clip; watch them before calling a phase done. This is the
lesson S9 exists to apply: S8 shipped on aggregates and the renders still
had real, findable defects.

## Failure handling

If a phase hits its 2-attempt cap without clearing its gate: log the
numbers honestly, keep whatever mechanism is safe/no-op-verified in the
code (opt-in, off by default), and escalate to Prabin rather than
weakening the gate or silently trying a third mechanism.

## Docs

Same policy as S8 — one `planLogGMR.md` entry per phase boundary, dense
tables not prose. Wiki (`wiki/experiments/`, `wiki/index.md`, `wiki/log.md`)
updated once, at S9's end or decision point, not mid-sprint.

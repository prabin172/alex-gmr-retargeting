# Plan Log — Continuation/Homotopy Floor-Penetration Solve

Execution trail for `plan.md`. Every claim in the gate traces to a command here.
Env for every python call: `source /home/ptimilsina/miniforge3/etc/profile.d/conda.sh && conda activate gmr`.

## Pre-T1 note: FLOOR_COLLISION is off by default for 2 of the 3 gate clips

Checked `retargetingPipeline.sh`'s `CLIPS[]` entries and global defaults:
- Global default `FLOOR_COLLISION="${FLOOR_COLLISION:-off}"`.
- `standup_natural_01` and `standup_side_05` both have empty `go_extra` → run with
  floor-collision OFF in the actual pipeline today. Stage 4's hard floor QP rows never
  activate for them; their only floor mechanism today is Stage 4.5 grounding.
- `luigi_standSupine_08` has `go_extra="--floor-collision on --floor-phase-aware on"`.

Continuation (§3.2/§3.5 of plan.md) extends the floor QP rows inside `_build_collision` —
it has nothing to act on if `--floor-collision off`. **Deviation from plan.md's literal
"mirror the CLIPS[] entry" instruction**: for all three gate clips (T1 baseline AND every
continuation dev run), I explicitly pass `--floor-collision on` (and, for luigi,
`--floor-phase-aware on` matching its own entry). This is necessary for the mechanism under
test to run at all, not a scope change to the shipped pipeline defaults. Recorded here per
plan.md's ground rule 7 (stop and log deviations, don't improvise silently).

Common flags used for every dev run below (mirrors pipeline globals):
```
--lambda-smooth 320 --n-outer 6 --foot-weight 160 --hand-weight 32 --plant-min-run 8 \
--floor-weight 200 --floor-mode estimate --floor-collision on \
--sens-min-pen 0.015 --sens-foot-min-pen 0.015 --collision-penalty 1000
```
Plus per-clip: `luigi_standSupine_08` additionally gets `--floor-phase-aware on`.

## T1 — Baseline capture (2026-07-14)

Ran Stage 4 standalone on the three gate clips from their existing Stage-3
(`outputs/contactfirst/*_contactfirst.npz`) outputs, saved to `outputs/cont_dev/<clip>_base.npz`.
Then `scripts/dev_cont_probe.py` (new, isolates floor-only raw penetration per frame — the
metric the continuation homotopy needs, distinct from `_collision_stats`' self+floor-mixed
`max_pen_cm`).

| clip | Stage-4-reported pen (mixed self+floor, cm) | isolated floor pen max (cm) | frames>0.5cm pen | self-pen peak (cm) | spikes | plant_slip max (cm) | flat_mean (deg) | foot_floor_err (cm) | tracking mean (m) |
|---|---|---|---|---|---|---|---|---|---|
| standup_natural_01 | 13.8 | **13.48** | 658/658 (100%) | 2.22 | 0 | 0.67 | 17.17 | 7.49 | 0.0780 |
| standup_side_05 | 24.4 | **24.37** | 1120/1323 (84.7%) | 1.29 | 1 | 1.29 | 3.52 | 13.72 | 0.0930 |
| luigi_standSupine_08 (phase-aware) | 3.6 | **4.62** | 365/1163 (31.4%) | 2.20 | 0 | 0.92 | 7.56 | 10.03 | 0.0850 |

Notes:
- `standup_natural_01`: 100% of frames register >0.5cm floor pen at Stage 4 with hard
  floor-collision forced on — this clip has never run with `--floor-collision on` in the
  shipped pipeline (see deviation note above), so this is the first time its mesh has been
  checked against a hard floor constraint at all. Matches the solver's own `coll=100.0%`
  Stage-B-best line.
- `standup_side_05` already carries 1 velocity spike in Stage A even before continuation —
  pre-existing under forced floor-collision-on, not something continuation introduces. Watch
  this in T6 (spikes must not exceed baseline, and baseline here is already non-zero).
- All three confirm large `foot_floor_err` (7-14cm) alongside floor pen — consistent with
  wiki's between-phase diagnosis (coplanarity/floor-height conflict, not just depth).
- `standup_natural_01`/`standup_side_05` floor_z re-estimated by the probe from the NPZ's own
  planted-foot data (`_estimate_floor_z`), matching what the solver itself used (same
  function) — not an independent guess.

## T2/T3 — Probe + allowance + per-row penalty + tracking relaxation (2026-07-14)

Implemented in `scripts/solve_global_trajectory_opt_contactfirst.py`:
- `_floor_pen_by_frame(model, data, qpos, floor_gid, floor_active_frames=None)` — new module
  function (next to `_collision_stats`), isolates floor-only RAW penetration (`-ct.dist`, no
  `COLL_MARGIN` offset) per frame, plus per-frame set of penetrating body ids. `dev_cont_probe.py`
  was written before this landed in the main module (had its own copy for T1) — left as-is,
  duplication acceptable per this codebase's own convention (`_load_model_with_floor`/
  `floor_phase_weight` are already duplicated across the two solver scripts).
- `_build_collision(...)` gained `floor_pen_allow=0.0` (subtracts from FLOOR rows' demanded
  correction only, before the `pen <= 0` skip test) and now returns a 4th value `is_floor_row`
  (bool array, parallel to rows). Only call site (`stage_b`) updated.
- `_build_tracking(...)` gained `extra_downweight=None` (per-frame dict `{role: factor}`,
  multiplies that role's tracking weight at that frame on top of the existing
  `downweight_roles`/`downweight_factor`).
- `stage_b(...)` gained `floor_pen_allow=0.0`, `floor_slack_penalty=None`, `extra_downweight=None`,
  threads all three through; per-row slack penalty now `rho = where(is_floor_row,
  floor_slack_penalty or collision_penalty, collision_penalty)` (self-collision rows always at
  `collision_penalty`, never hardened by continuation).

**No-op check (plan.md T2/T3 acceptance):** re-ran `standup_natural_01` with the new code, all
new params at their defaults (`floor_pen_allow=0.0` implicit, no CLI flag yet added):
`cmp outputs/cont_dev/standup_natural_01_base.npz outputs/cont_dev/standup_natural_01_noop_check.npz`
→ **BYTE-IDENTICAL**. Confirmed by inspection too: `floor_pen_allow=0.0` fails the `> 0.0` guard
(pen unchanged), `floor_slack_penalty=None` makes `rho_floor == collision_penalty` so `rho` is
uniform (identical to the old `np.full(m, 2*collision_penalty)`), `extra_downweight=None` skips
the multiply in `_build_tracking`.

## T4 — Continuation loop: first implementation used the WRONG homotopy schedule (bug, found + fixed)

First-pass implementation used a single GLOBAL scalar `eps_k = P0 * (1 - k/K)` (`P0` = max
penetration over the whole clip) applied uniformly to every floor row via `_build_collision`'s
new `floor_pen_allow` scalar. This contradicts plan.md §3.2's actual design ("Measure its
residual **per-frame** penetration `p0(t)`... allowed penetration `ε_k(t) = p0(t)·(1-k/K)`") —
a written design I didn't carry through into the code correctly on the first pass.

**Symptom, caught by actually running it** (`--continuation 4` on `standup_side_05`, the worst
gate clip, P0=24.44cm): pass 1's `eps_1 = 18.33cm` (75% of the clip's WORST frame) is applied to
EVERY floor row clip-wide — so any frame with penetration below 18.33cm (nearly the whole clip
except the worst few frames) has its floor row's demanded correction reduced to ≤0, which
`_build_collision`'s existing `if pen <= 0: continue` then SKIPS entirely. Net effect: pass 1
disabled almost all floor rows instead of asking for gradual improvement everywhere. Confirmed
in the log — Stage B's own internal SCA outers wandered (pen climbed 24→39cm across outers with
the floor rows gone) and pass 1's own keep-best correctly fell back to the pass-0 iterate
unchanged, so the cross-pass safety net worked (never shipped worse) but the mechanism did zero
useful work: `Continuation: pass 1/4 ... floor_pen_max=24.44cm ... stalled at pass 1 (improved
0.00cm, 0.0%)`.

**Fix**: `_build_collision`'s `floor_pen_allow` now accepts EITHER a scalar (0.0 default, exact
no-op, unchanged) OR a `(T,)` per-frame array — `allow_t = floor_pen_allow if scalar else
floor_pen_allow[t]`, subtracted from that frame's own floor row(s) only. `_run_continuation` now
computes `pen0_by_frame` (the full `(T,)` array from `_floor_pen_by_frame` on `qpos0`, not just
its max) ONCE, and each pass's schedule is `eps_k = pen0_by_frame * (1 - k/K)` — an elementwise
array, so a frame that started at 2cm shrinks its own allowance toward 0 on the same K-step
schedule as the frame that started at 24cm, instead of the worst frame's schedule silently
disabling every other frame's constraint. Stall detection and cross-pass scoring are unaffected
(still driven by the whole-clip `pen.max()`, which is what the gate cares about).

Lesson for future continuation-style plans: when a written plan specifies a per-element
quantity (`ε_k(t)`), verify the implementation actually indexes per-element before running —
collapsing to a summary statistic (max) changes the mechanism's behavior, not just its
numbers, and the failure mode (silently disabling constraints instead of tightening them) can
look like "converged, no more to gain" rather than "wrong code" unless checked against the
written design.

## T6 — Gate: bigger finding — 2 of 3 gate clips have a PRE-EXISTING Stage-B oscillation problem, unrelated to continuation

After the per-frame schedule fix (above), re-ran `--continuation 4` on `standup_side_05` —
**still zero improvement, still stalls at pass 1 with the SAME 24.44cm.** Traced why by reading
the full per-outer trace: `Stage B best: pen=24.44cm` for BOTH pass 0 and pass 1 — this value is
literally the Stage-A/pre-Stage-B "warm" score. None of pass 1's 6 outers ever beat it (all
landed 26–39cm).

**Control experiment**: ran plain Stage B (no continuation) on `standup_side_05` with
`--n-outer 20` (over 3x the pipeline default). Every single one of 20 outers scored WORSE than
the warm value on the internal `_iter_score` (range 24.6–33cm, oscillating, never trending down).
`Stage B best: pen=24.44cm` — identical to the warm value, meaning Stage B's own keep-best
mechanism never found a single improving iterate in 20 tries.

Re-checked `standup_natural_01` the same way (plain Stage B, `--n-outer 6`, from the T1 run's own
full log): `warm: pen=13.77cm`, all 6 outers score 13.97–16.93cm (worse), `Stage B best:
pen=13.77cm` = the warm value again. Same pattern.

**Conclusion**: this is NOT a continuation problem. It's a pre-existing weakness in Stage B's SCA
(sequential-convex-approximation) outer loop, on THESE TWO CLIPS, UNDER `--floor-collision on`
— exactly the "SCA outer loop oscillates" behavior this codebase's own comments already document
(stage_b's keep-best docstring, ~l.935), and matches `retargetingPipeline.sh`'s own
`FLOOR_COLLISION` comment: *"validated on 1 clip only so far — opt-in pending corpus
validation"*. `standup_natural_01` and `standup_side_05` have NEVER run with `--floor-collision
on` in the shipped pipeline before this session (their `CLIPS[]` `go_extra` is empty) — turning
it on for them (a deviation I made and flagged at the top of this log, necessary for continuation
to have floor rows to act on at all) exposes a solver behavior nobody had ever exercised on these
two clips: with hard floor rows added, self-collision rows (thousands, ~90–100% active every
outer) and the floor rows fight each other and the SCA never settles on an improving step, on
either clip, regardless of outer-iteration budget. **Continuation cannot rescue a base solve that
oscillates rather than converges** — a homotopy schedule only helps a solver that is making
correct-direction progress but can't close the full gap in one linearization; it does nothing if
every individual pass is itself non-convergent.

**The one clip where `--floor-collision on` IS already shipped and Stage B genuinely
converges** (`luigi_standSupine_08`, `--floor-phase-aware on`): continuation shows a REAL,
measured improvement. Full trace: `/tmp/luigi_cont4.log` (background run, `outputs/cont_dev/
luigi_standSupine_08_cont4.npz`).

| pass | floor_pen_max (cm) | selfpen_over (cm) | spikes | kept as cross-pass best? |
|---|---|---|---|---|
| 0 (plain Stage B) | 3.56 | 0.20 | 0 | seed |
| 1 | 1.87 | 1.24 | 0 | NO — selfpen_over regressed (0.20→1.24), loses lexicographically despite better floor pen |
| 2 | 2.68 | 0.01 | 0 | **YES** — selfpen_over beats pass 0 (0.01<0.20), wins lexicographically even though its own floor_pen_max (2.68) is worse than pass 1's (rejected) 1.87 |

**Shipped result: floor_pen_max 3.56→2.68cm (−25%), selfpen_over 0.20→0.01cm (−95%), spikes
stayed 0.** A genuine, safe, measured improvement on the one clip where the base mechanism is
healthy — exactly the outcome the plan's safety design (lexicographic keep-best, spikes/self-pen
gated ahead of floor pen) was built to produce: it did NOT ship pass 1's better-looking floor
number because that iterate cost self-collision headroom; it shipped the pass that improved
kinematic quality overall.

**Known imprecision, not a correctness bug**: the stall check compares `Pk` (each pass's OWN raw
floor pen) against `P_prev`, regardless of whether that pass's iterate was actually kept as
cross-pass best. Pass 2 was flagged as "stalled" because its raw pen (2.68) was worse than pass
1's raw pen (1.87) — even though pass 1 was REJECTED and pass 2 was the one actually shipped. A
stall metric tracking the kept-best trend instead of the raw per-pass trend might have continued
further (unknown whether pass 3/4 would improve more or also regress) — not fixed, given the
scope decision below.

### Gate verdict vs plan.md's ship bar

Plan's bar: every bold metric passes on ≥2 of 3 clips. Actual: 1 of 3 clips shows benefit
(`luigi_standSupine_08`); the other 2 don't engage the mechanism at all because their base solve
(under forced `--floor-collision on`) doesn't converge, independent of continuation. **Does not
clear the ship bar — NOT wired into `retargetingPipeline.sh` defaults.**

This is still a useful, informative result (per plan.md §T6's own "partial result" guidance):
continuation-v1 is validated as sound and safe (per-frame schedule correct, cross-pass keep-best
correctly protects self-collision/spikes, provably never regresses a healthy base solve) but its
scope is narrower than hoped — it is a polish mechanism for clips whose base Stage-B solve
converges, not a fix for the SCA-oscillation problem the other two gate clips exposed. That
oscillation problem is a separate, bigger, pre-existing issue (likely needs its own investigation
— e.g. per-clip trust-region tuning, a different linearization order, or simply confirming
`--floor-collision` isn't safe corpus-wide yet, which the pipeline's own comment already
flagged) and was NOT in scope for this plan.

### T5 — no-op certification (trivially satisfied, no new full-pipeline run needed)

`retargetingPipeline.sh`'s Stage-4 invocation (checked again) does not pass `--continuation`
anywhere — every clip in the shipped pipeline runs with the CLI default (`0`), which is verified
byte-identical in T2–T4 above. Since continuation did not clear the ship bar, §3.7's pipeline
wiring was correspondingly NOT added (no `CONTINUATION` env knob) — nothing in
`retargetingPipeline.sh` changed this session, so a full-corpus re-run to prove no-op would be
re-proving something no code touches. Skipped on that basis rather than run redundantly.

---

# Feasibility-First-v1

Execution trail for the current `plan.md` (feasibility-first-v1, replaces continuation-v1 above).
Env for every python call: `source /home/ptimilsina/miniforge3/etc/profile.d/conda.sh && conda activate gmr`.

## T1 — Stage-3 baseline (2026-07-15)

Ran Stage 3 standalone on the two gate clips from their existing Stage-2.5 outputs
(`outputs/canonical_human/fbx_fresh/*_canonical_grounded.npz`), mirroring the pipeline's exact
Stage-3 flags (`STRIDE=1 IK_ITERS=40`, `--contact-min-run 12 --contact-ramp 16 --contact-preroll 8`,
`--coplanar-feet-mode mean`, `--floor-weight 10 --floor-margin 0 --floor-gain 5`, no hard-tier/
swing-clear/leg-floor-refine flags, empty per-clip `solve_extra` for both — confirmed from
`retargetingPipeline.sh`'s `CLIPS[]` entries). Saved to `outputs/ff_dev/<clip>_cf_base.npz`.

Depths measured with `_ff_measure_frame` (the new §3.2 module function) directly against these
NPZs (floor mocap positioned at z=0.0, the Stage-2.5 invariant -- NOT re-estimated, matching how
Stage 3 itself treats `floor_z` when persisted contacts are present):

| clip | floor_pen max (cm) | floor_pen p50/p95 (cm) | frames >2cm | self_pen max (cm) | frames >1.5cm |
|---|---|---|---|---|---|
| standup_natural_01 | 22.24 | 20.17 / 21.80 | 100.0% | 1.19 | 0.0% |
| standup_side_05 | 25.03 | 13.97 / 24.31 | 100.0% | 3.68 | 9.0% |

Both clips show DEEP, NEAR-UNIVERSAL floor penetration at Stage 3 already (not a shallow
floor-vs-mesh artifact -- `_detect_floor_sensitive_frames`'s docstring documents a sub-cm
universal artifact this is NOT: p50 alone is 14-20cm). Consistent with the between-phase
diagnosis (wiki/concepts/grounding.md): during the clip's lying/crouch phase, large chunks of
the body (torso/thighs) sit genuinely below the robot's floor=0 reference by construction -- this
is exactly the target failure mode feasibility-first is meant to address. Stage 3's own printed
summary ("Floor-penetration summary: ... 100.0%/74.8% of frames") undercounts by only checking
ct.dist<0 (any contact) not depth -- matches directionally, confirms via a second measurement
path. `standup_natural_01` also shows a floor-invariant-gate WARNING (14 contacting-effector
targets landed below their floor reference, max depth 0.62cm) -- a small, separate, pre-existing
target-construction residual (not this session's concern, tracked by the existing gate check).

**Accept**: baseline depths logged for both gate clips (table above) -- T1 done.

## T3 — STOPPED: the retry wrapper's weight-boost mechanism triggers a pre-existing solver instability

Implemented §3.1 (phase boundaries), §3.2 (`_ff_measure_frame`/`_ff_limb_roles_for_body`), §3.3
(the retry wrapper), §3.5 (CLI + NPZ keys) in `solve_fbx_canonical_alex_contactfirst.py`. No-op
certified first: `--feasibility-first` OFF is BYTE-IDENTICAL to T1's baseline
(`cmp outputs/ff_dev/standup_natural_01_cf_base.npz outputs/ff_dev/standup_natural_01_noop_check.npz`
-- IDENTICAL). T2 passes.

**T3's own acceptance ("retries happened, floor_pen_cm.max() improves vs T1 baseline") FAILED --
but not from a bug in the wrapper's bookkeeping. Root cause isolated by direct A/B, independent
of any of my new code:**

`--feasibility-first` on `standup_natural_01` (first 30 frames) diverges catastrophically:
`mean_err` climbs monotonically frame-to-frame (0.10 -> 2.42 by frame 29), hand-contact
"distance" blows past 270cm, `floor_pen max=306.55cm` (3+ METRES). Isolated which of the retry
wrapper's THREE mechanisms (relax tracking / boost floor+coll weight / restart noise) causes it:

1. Disabled noise entirely (`--ff-restart-from 100`) -- STILL diverges identically (floor_pen
   max=305.55cm). Not the noise.
2. Reduced to `--ff-max-attempts 1` (a single retry, boost=2x, no noise) -- STILL diverges
   (floor_pen max=309.82cm after just ONE mild retry). Not an accumulation-of-attempts effect.
3. **Removed the retry wrapper entirely** -- ran PLAIN Stage 3 (no `--feasibility-first` at all)
   with `--floor-weight 20` (exactly 2x the shipped default of 10, matching what one retry's
   `boost=2.0` would produce) on the same clip/frames: **diverges identically** -- `mean_err`
   climbs every single frame (0.40 -> 4.28 over 15 frames), hand-contact distance grows
   unbounded (53cm -> 473cm). Tried `--floor-weight 15` (a MILD 1.5x) -- **same runaway pattern**
   (mean_err 0.39 -> 3.44 over 15 frames).

**Conclusion: this is a PRE-EXISTING instability in Stage 3's soft floor-repulsion term
(`--floor-weight`), triggered by ANY increase above the shipped default of 10 on
`standup_natural_01`, with NOTHING from feasibility-first-v1 involved.** The shipped default
(10) sits at or near a stability edge for this clip; the retry wrapper's core mechanism (§3.3's
`boost = 2.0 ** a`, meant to push a violating frame toward feasibility) instead walks straight
into this edge on attempt 1 and the warm-started per-frame chain compounds it every subsequent
frame (once one frame's kept qpos is bad, every later frame inherits a bad warm start and gets
worse). This is NOT the previously-documented `--floor-hard`/hard-tier 44-metre blowup
(`wiki/experiments/retired-approaches.md`) -- that was a hierarchical/task-priority conflict;
this is the plain single-level DLS solve with an ordinary soft floor-repulsion weight, diverging
on its own with no hierarchical/hard-tier flags anywhere in these runs.

**Per plan.md ground rule 6 (two failed attempts -> stop, log, don't redesign): STOPPING here.**
The core §3.3 mechanism ("boost floor/collision weight to force feasibility") is unsound as
designed for THIS solver -- boosting is not a safe lever at all on at least this clip, at any
tested multiplier from 1.5x to 64x. Flagged to Prabin for a decision before continuing (see
chat): whether to (a) drop the weight-boost half of the retry entirely and rely only on
tracking-relaxation + restart-noise (untested whether that alone is enough without the boost),
(b) first characterize/fix the underlying floor_weight instability as its OWN separate
investigation (a bigger, more fundamental finding than this plan anticipated -- the shipped
default of 10 being this close to an edge on a get-up clip is itself concerning for the shipped
pipeline, independent of feasibility-first), or (c) abandon this plan's per-frame weight-boost
approach and reconsider the mechanism.

**Code state**: the retry wrapper (§3.1/§3.2/§3.3/§3.5) is fully implemented and no-op-certified
when off (`--feasibility-first` default False). Left in place, NOT further exercised pending
direction -- do not run it with attempts > 0 on more clips until the floor_weight instability
question above is resolved, since every observed run so far diverges.

## T3 continued — redesign attempt 1 (drop boost) still diverges; root cause is deeper than the boost

Per Prabin's direction: dropped the `boost = 2.0 ** a` weight-boost entirely (floor_weight/
coll_weight now IDENTICAL to attempt 0 on every retry -- only tracking relaxation + phase-boundary
noise vary). No-op re-verified (BYTE-IDENTICAL with the flag off). Re-tested on
standup_natural_01, first 30 frames:

- With noise enabled (default `--ff-restart-from 3`): STILL diverges (floor_pen max=238.67cm,
  mean_err climbing).
- Noise disabled (`--ff-restart-from 100`, i.e. PURE tracking-relaxation retries, zero boost,
  zero noise): STILL diverges (floor_pen max=302.14cm, mean_err 0.33->2.36 over 30 frames).

**So the boost was never the sole cause -- pure tracking relaxation alone is enough to diverge.**
Diagnosed why: the keep-best score `(hard, pf+ps, terr)` is lexicographic, and two different
qpos solutions almost NEVER tie exactly on the `pf+ps` (pen) term -- so in practice the score
comparison is decided by `pf+ps` alone, and `terr` (tracking error) is essentially never reached
as a tie-break. A retry that shaves a hair off floor/self penetration wins over attempt 0 even if
its tracking error is enormous (a nonsensical pose), because the tuple comparison stops at the
2nd element long before reaching the 3rd. That bad pose then becomes NEXT FRAME's warm start,
and the same thing happens again -- compounding into the observed runaway.

**Fix attempt 2: a tracking-error sanity cap.** Added `_ff_terr_cap = max(terr0 * 3.0, terr0 +
0.5)` (terr0 = attempt 0's own tracking error for this frame) and disqualified any retry whose
tracking error exceeds it (skipped from best-q consideration entirely, not just deprioritized).
No-op re-verified. **STILL diverges** (floor_pen max=246.69cm over the same 30 frames) -- though
notably the error now PLATEAUS around frame 15-29 (~2.02) rather than continuing to grow
unboundedly, a partial improvement but not a fix.

**Why the cap didn't work, reasoned through (not yet re-tested)**: the cap is RELATIVE to THIS
frame's own attempt-0 tracking error (`terr0`). But attempt 0 itself warm-starts from the
PREVIOUS frame's kept q -- if that previous frame's kept q was already corrupted (from an earlier
retry that passed a now-too-permissive relative cap), attempt 0's OWN terr0 is already elevated,
and `terr0 * 3.0` is proportionally even MORE permissive, not less. The cap is relative to a
reference that can itself be already compromised, so it doesn't prevent the compounding chain --
it only slows it.

**Proposed fix 3 (not yet implemented)**: switch the disqualification criterion from task-space
tracking error (`terr`, which is a downstream, amplifiable proxy) to JOINT-SPACE displacement
from the INCOMING warm start (`ff_q_warm_in`, i.e. `np.abs(q_try[7:] - ff_q_warm_in[7:]).max()`).
This is scale-invariant and, critically, bounds each frame's step relative to what it STARTED
from rather than a same-frame reference that can already be corrupted -- if every frame's kept q
is guaranteed to move at most X radians from ITS OWN incoming warm start, the worst case across T
frames is bounded linear drift, not compounding/exponential blowup. Threshold candidate: 0.5
rad/frame, reusing this codebase's OWN existing convention for a velocity spike
(`SPIKE_RAD_PER_S = 60` in `solve_global_trajectory_opt_contactfirst.py` = 0.5 rad/frame @120Hz)
rather than inventing a new number.

**STOPPING here per plan.md ground rule 6 (two failed attempts -> stop, log, ask) rather than
implementing fix 3 silently** -- this is the second fix attempt to fail, and while fix 3 has a
principled argument behind it (relative-to-corrupted-reference vs relative-to-incoming-warm-start
is a real, reasoned distinction, not a guess), it has NOT been tested yet. Flagged to Prabin.

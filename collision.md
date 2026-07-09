# Ground-penetration investigation (2026-07-09)

Status: **experiments documented, nothing shipped by default.** Both mechanisms below are
implemented, tested, and gated `off` by default in the shipped pipeline. Written up for
mentor/second-opinion review before deciding how to proceed.

## The report

Mentor reviewed an IsaacLab mimic-training run on `luigi_standProne_03` and said:

> "the ground penetrations are the biggest problem here. you can see the hands at the beginning
> are not even able to position correctly for the push because they are forced into the ground.
> Also the left foot can't make it to the front of the robot because it's asked to get through
> the ground."

A reference target the robot physically cannot reach (limb below the floor) produces a hard,
un-trainable tracking error for the whole reach/step — worse than the previously-documented "float"
issue (a few cm of registration bias mimic can at least try to imitate).

## Root cause

Traced every floor-interaction mechanism in the pipeline. **None of them stop a hand or a swing
limb from passing through the floor:**

- **Stage 3** (`solve_fbx_canonical_alex_contactfirst.py`, per-frame IK): no floor constraint at
  all — collision-blind with respect to the floor by original design (self-collision between robot
  links IS handled here, via `self_collision_rows`, but nothing checks against the ground).
- **Stage 4** (`solve_global_trajectory_opt_contactfirst.py`, `_build_contact`): the only floor
  handling is a **soft equality pin** driving a foot's sole-corner Z to a shared `floor_z`, gated
  `info["kind"]=="foot" and planted[eff][t]`. Hands get no floor term ever; swing (non-planted)
  feet get none either.
- **Stage 4.5** (`post_process_ground_contactfirst.py`): a single rigid Z shift for the whole clip,
  registered to the median of planted-foot heights. 1 DOF — cannot fix a local per-frame violation
  on a swinging limb.
- The model XML has no floor geom at all — floor height only ever existed as a computed scalar,
  never as MuJoCo-collidable geometry.

This matches and extends an already-documented, already-quantified defect
(`wiki/results/metrics.md`: swing/tucked-foot clipping up to 28cm, planted-foot penetration median
2.5cm up to 6.5cm) — previously scoped to feet only. The mentor's report shows it also hits hands,
which had never been measured (the eval tooling is feet-only).

## Two mechanisms tried

### 1. Stage 4: mesh-accurate floor collision — VALIDATED, works, limited by scope

Injects a floor PLANE geom into the model at solve time, in-memory only (via `mujoco.MjSpec`, a
**mocap body** — not a normal static/welded body, since a static body's world position is baked in
at compile time and does not respond to a post-compile position write; a mocap body's does, via
`data.mocap_pos`, without adding any DOFs). Never touches the hand-maintained asset XML.

Reuses `_build_collision`'s existing self-collision machinery (mesh-accurate contact detection via
`mj_forward`, soft-slack QP augmentation) against this floor plane instead of another robot link —
same code path, same proven convergence behavior (`n_outer=6` SCA re-linearization, keep-best-iterate
scoring).

**Bug found and fixed along the way:** the floor body's id is never 0 (its own mocap child of
worldbody, not worldbody itself), so the pre-existing `b1==0 or b2==0` self-collision exclusion
silently didn't catch it — floor contacts leaked into "self-collision" counting regardless of the
on/off toggle. Fixed by always recognizing the floor geom for exclusion (`floor_gid`, always known
once the model has a floor), separately gating whether it's actually counted/enforced
(`count_floor`, the real toggle). Verified `--floor-collision off` now reproduces the original
solver's numbers exactly (pen=1.01cm, coll=4.7%, byte-identical to a pre-edit run).

**Result on `luigi_standProne_03`:** turning it on correctly and precisely surfaces both reported
defects for the first time, with frame numbers:
- **LEFT_FOOT** penetrates **10–11.5cm at frames 320–328** (root_z≈0.42, the stand-up transition —
  the stepping-forward swing foot: "asked to get through the ground").
- **LEFT_WRIST** penetrates **7.6cm at frames 259–261** (root_z≈0.3, the push phase: "forced into
  the ground").

Both already present in Stage 3's raw per-frame IK output, before any smoothing.

**Limitation found:** Stage B's SCA/keep-best loop (tuned for self-collision penetration typically
≤2cm) doesn't fully resolve violations this large. It measurably improves (16.5cm→13.3cm best
overall penetration) but plateaus — oscillates between "fix floor, reopen self-collision" / "fix
self-collision, reopen floor" each outer, because collision rows are only re-linearized at each
outer's start (documented pre-existing behavior, see `wiki/concepts/globalopt.md`). Tried more
outers (12 vs 6): no further improvement, same plateau. Tried a bigger trust region (0.3 vs 0.15):
pen inches to 11.85cm but plant slip explodes to 33cm — a bad trade, not a real win.

**Why it's structurally limited**: Stage B only optimizes actuated joint increments `δQ`; the root
pose is frozen from Stage A. A large violation concentrated at the ROOT level (not just a local limb
dip) cannot be corrected here no matter how many outers — there's no root DOF to move.

**Status:** real, measured, safe improvement. `--floor-collision {on,off}` / pipeline
`FLOOR_COLLISION`, **default off** pending corpus-wide validation (only tested on this one clip so
far).

### 2. Stage 3: root-aware floor avoidance upstream — UNRESOLVED, real tension found

Since Stage 3's per-frame IK solves the root pose too (unlike Stage 4's Stage B), it can in
principle correct a root-level sunk pose, not just a local limb dip. Added `floor_collision_rows`
(mirrors the existing `self_collision_rows` self-collision term exactly — same weighted
least-squares row-building, level-2/soft in the task-priority stack) against the same injected
floor plane, using an Alex-frame floor-height estimate transformed from the human-frame `floor_z`
already computed for contact detection (`alex_floor_z = alex_rest_pelvis_z + root_scale *
(human_floor_z - human_pelvis0_z)` — same delta-scaling convention already used for
`root_delta`/`morphology-scaling.md`, no second solve pass needed).

**Two real bugs found and fixed:**
1. Same leak as Stage 4: the *pre-existing* `self_collision_rows` term also only excluded
   `b1==0 or b2==0` — once a floor body exists in the model AT ALL (regardless of whether floor
   avoidance is enabled), its contacts leaked into the self-collision repulsion term, treating the
   floor as a self-colliding robot link and repelling the whole robot away from it. This alone
   caused catastrophic divergence (mean tracking error to 12+ metres by frame 400) even with
   `--floor-weight 0`. Fixed by passing `floor_gid` into `self_collision_rows` unconditionally (not
   gated behind the avoidance toggle).
2. MuJoCo's convex-hull-vs-plane collision returns **multiple simultaneous contact points per
   colliding body** (measured: 3, vs the sparser point contacts typical of link-vs-link
   self-collision). Emitting one QP row per raw contact point silently **tripled** each body's
   effective correction weight — this alone was enough to destabilize the solve at
   `--coll-weight`'s validated-for-self-collision default (20). Fixed by deduplicating to one row
   per colliding body (deepest contact only) — this stabilized it (no more runaway divergence).

**After both fixes, still not resolved:** with the stabilized mechanism (`--floor-weight 20`,
deduplicated), the reported LEFT_FOOT violation measurably improves, but a **new, worse** violation
appears: **RIGHT_GRIPPER now penetrates 35.8cm at frames 296–310** — three times deeper than the
11.5cm LEFT_FOOT violation it's nominally fixing. This is not numerical instability (errors stay
bounded, ~0.05–0.1m tracking error, sane order of magnitude) — it's the soft per-frame IK **trading
one violation for a worse one elsewhere**: pushing the foot up drags the coupled kinematic chain
(posture regularization, joint coupling across the whole-body least-squares stack) into a
configuration that sinks the other hand further. A single flat (level-2, soft) weighted pass isn't
enough to resolve a violation this large without some real prioritization — candidates not yet
tried: hierarchical/level-1 treatment (like the existing planted-foot hold), a weight ramp across
iterations, or per-effector sequencing.

**Status:** genuinely unresolved. `--floor-weight` **default 0 (off)**, opt-in only, documented as
experimental in its own `--help` text.

## Step-0 addendum (2026-07-09, later): target-space diagnostic settles the design

Measured target vs achieved z at the offending frames of the baseline Stage-3 NPZ
(`target_positions` / `human_target_positions` / `achieved_positions`). Result — the "clamp the
targets" hypothesis was **wrong for the foot, right for the hand**, and experiment 2's failure is
now fully explained:

- **Foot window (317–328): targets are fine** (2.3–8.9 cm above the planted reference); the
  achieved ankle sags 7–11 cm BELOW its target, toe-down pitch digs the toe to −12 cm. Contact
  flag is OFF the whole window (turns on at 340) — pure free-swing tracking failure with no floor
  term to oppose it. Target clamping cannot fix this; a floor-repulsion term can, and tracking
  AGREES with it here (wants the ankle 10 cm higher) — no priority conflict.
- **Hand window (257–263): the palm-site PIN target is below the floor** (palm site tracks to
  −6.5…−7.2 cm below the mesh floor while contact-flagged; the wrist role target is suppressed by
  design during contact). Direct target-space fix: clamp the pin target z to the floor.
- **Experiment 2's floor was placed ~8 cm too high**: pelvis-transform estimate −0.0355 vs the
  actual plant height −0.115. The repulsion term fought every prone-phase contact frame — that,
  not soft-IK priority tension, was the dominant failure. Target-space derivation
  `median(planted ankle targets) − ankle_clearance = −0.0384 − 0.070 = −0.108` lands within 7 mm
  of the measured mesh floor. The "RIGHT_GRIPPER 35.8 cm" number was also inflated by a moved
  floor reference (that run re-registered its floor 7 cm higher); `scripts/
  diagnose_floor_penetration.py` now exists with a `--floor-z` fixed-reference flag to prevent a
  repeat.

**Implementation plan for the fix: `collisionFixPlan.md` (repo root).** The open question below is
retained for history but is now settled: fix floor placement (target-space), clamp palm-pin
targets, re-enable the Stage-3 repulsion at weight 20, Stage-4 floor collision as residual cleanup.

## Open question for the next round

Is the right move (a) ship Stage-4-only as real, bounded, safe progress and treat full resolution as
a follow-up; (b) invest in a smarter Stage-3 prioritization scheme (hierarchical floor term, weight
ramp) to stop the whack-a-mole between effectors; or (c) something else (e.g. a proper two-pass
Stage-3 warm-start, or accepting the Stage-4 structural ceiling and looking at a different mechanism
entirely for root-level corrections). Bringing this + the numbers above to get a second opinion
before continuing.

## Code state (uncommitted, all gated off by default)

- `scripts/solve_global_trajectory_opt_contactfirst.py`: `_load_model_with_floor`, floor-aware
  `_collision_stats`/`_build_collision` (via `floor_gid`+`count_floor`), `--floor-collision {on,off}`
  CLI (default `on` in the script's own default, but pipeline forces `FLOOR_COLLISION=off`).
- `scripts/solve_fbx_canonical_alex_contactfirst.py`: `_load_model_with_floor` (duplicated, same
  technique — independent CLI scripts, not sharing imports), `floor_collision_rows`, the
  `self_collision_rows` floor-exclusion fix (`floor_gid` param, always passed), `--floor-weight`
  (default 0.0, off), `--floor-margin`, `--floor-gain`.
- `retargetingPipeline.sh`: `FLOOR_COLLISION` env knob (default `off`) wired into the Stage-4 call.
  Stage-3's new flags are NOT yet wired into the pipeline script (only reachable by invoking the
  solver directly) — deliberately, since they're not validated.

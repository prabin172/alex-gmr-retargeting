# Floor-penetration fix — implementation plan

For the implementing session: read `collision.md` first (investigation history + two prior
experiments). This plan supersedes its "open question" section — Step 0 (target-space diagnostic,
2026-07-09, results below) settled the design. Everything here is on branch `develop`-era working
tree; the floor-injection machinery is ALREADY IMPLEMENTED and uncommitted — do not re-derive it,
and do not revert the bug fixes listed under "Already in the working tree".

## Confirmed diagnosis (Step 0, measured on `luigi_standProne_03` baseline Stage-3 NPZ)

Three distinct mechanisms produce the mimic-blocking floor penetration:

1. **Palm-pin targets below the floor (hand push phase, frames ~250–268).** During hand contact
   the wrist-body target is suppressed by design and the palm SITE is pinned to the scaled human
   contact point — which maps **6.5–7.2 cm below Alex's floor**. The pin tracks faithfully →
   fist underground. This is a target-space error; fix = clamp the pin target.
2. **Free-swing foot sag + toe-down dig (stand-up step, frames ~314–331).** Contact flag is OFF the
   whole window (turns on at 340) — contact machinery is NOT involved. The ankle target is FINE
   (2.3–8.9 cm above the planted reference) but the achieved ankle sags 7–11 cm BELOW its target,
   and the foot is pitched hard toe-down: toe reaches **−12 cm** below floor while the heel starts
   above. Nothing in the per-frame IK opposes this because nothing knows the floor exists. Target
   clamping does NOT fix this (target already fine). Fix = the Stage-3 floor-repulsion term — which
   is expected to work now because **tracking and floor AGREE here** (tracking wants the ankle
   10 cm higher than where it ended); the 2026-07-09 whack-a-mole failure was caused by floor
   MISPLACEMENT (see 3), not by soft-IK being structurally unable.
3. **The Stage-3 floor plane was placed ~8 cm too high.** The pelvis-delta-transform estimate gave
   `alex_floor_z = −0.0355` while the solve actually plants at **−0.115** — so the repulsion term
   fought EVERY contact frame of the prone phase, distorting the whole solve (and the reported
   "35.8 cm RIGHT_GRIPPER regression" was partly a moved-floor-reference metric artifact on top).
   Fix = derive the floor in TARGET space: `median(planted-frame ankle target z) − ankle_clearance
   = −0.0384 − 0.070 = −0.108`, vs measured mesh floor −0.115 → within 7 mm. Validated.

Baseline reference numbers (from `scripts/diagnose_floor_penetration.py`, floor_z=−0.1152):
max pen **11.5 cm** (LEFT_FOOT t=321), LEFT_WRIST_X_LINK 7.6 cm, LEFT_GRIPPER 6.9 cm,
RIGHT_GRIPPER 4.0 cm, mean-of-frame-max 1.88 cm, 44.1% of frames penetrate >1 cm.

## Already in the working tree (uncommitted — keep, do not revert)

- Both solvers have `_load_model_with_floor()` (MjSpec-injected floor plane as a **mocap** body —
  mocap, not static, because a static body's position is baked at compile; `data.mocap_pos` moves a
  mocap body every `mj_forward`, zero added DOFs). Never writes the hand-maintained asset XML.
- **Bug fix (both solvers):** the floor body's id ≠ 0, so the old `b1==0 or b2==0` exclusions
  missed it — floor contacts leaked into self-collision terms/stats. `floor_gid` is now recognized
  and excluded **unconditionally** (independent of any floor-feature toggle). This must stay.
- **Bug fix (Stage 3 `floor_collision_rows`):** MuJoCo's hull-vs-plane collision returns ~3 contact
  points per body; one QP row per raw contact silently tripled the effective weight and diverged.
  Deduplicated to one row per body (deepest contact). This must stay.
- Stage 4: `--floor-collision {on,off}` + pipeline `FLOOR_COLLISION` (currently default `off`).
- Stage 3: `floor_collision_rows` + `--floor-weight` (default 0.0 = off), `--floor-margin` (0.0),
  `--floor-gain` (5.0). Margin 0 is deliberate — planted limbs sit AT the floor by design; a margin
  would fight the contact terms.
- `scripts/diagnose_floor_penetration.py` — validation tool. **Always pass `--floor-z -0.1152`
  when comparing runs on this clip** (fixed reference; the default re-estimates from the run's own
  planted feet and moves when a fix lifts the plants — that artifact already burned us once).

## Fix A — correct the Stage-3 floor placement (target-space derivation)

File: `scripts/solve_fbx_canonical_alex_contactfirst.py`, in `main()`. Replace the body of the
`floor_kwargs` block (search `floor_kwargs = {}`; it currently computes `alex_floor_z` from the
pelvis-delta transform — the diagnosed 8 cm-off estimate) with:

1. **Ankle clearance** (do this while `data` still holds the just-solved rest pose, or reset to the
   rest `q` and `mj_forward` first): for each foot, `clearance = data.xpos[ankle_body].z −
   min(sole corner site z)`. Corner site names are in the Stage-4 solver's `SOLE_CORNER_SITES`
   dict — copy the 8 names (they are model sites, identical for both scripts). Use the mean of the
   two feet's clearances (~0.070 m).
2. **Planted ankle target heights**: loop `for src_i in frame_ids`, call `make_targets_for_frame(
   src_positions[src_i], role_to_idx, first_src_pos, target_rest_positions, root_scale,
   role_scales)` (positions only — cheap, no IK), and for each foot with
   `contacts[eff][src_i] == True` collect `targets[FOOT_POS_ROLE[eff]][2]`. `FOOT_POS_ROLE` maps
   left_foot→left_ankle etc. (already defined in the file). Pool both feet.
3. `alex_floor_z = median(pooled ankle target z) − clearance`. Fall back to the old
   pelvis-transform estimate ONLY if the pool is empty (clip with no foot contact — none in the
   corpus). Print both estimates so drift is visible per clip.
4. Keep the rest of the block (mocap positioning, `floor_kwargs`) unchanged.

Expected on this clip: `alex_floor_z ≈ −0.108` (was −0.0355).

## Fix B — clamp the palm-pin target to the floor

Same file, in the frame loop where the palm pin is built (search
`tgt = palm_rest_pos[eff] + root_delta + palm_pos_scale[eff] * (rel - rel0)` — currently line
~1548). Immediately after computing `tgt`:

```python
if args.floor_weight > 0.0:          # floor known ⇒ never pin the palm below it
    tgt[2] = max(tgt[2], alex_floor_z)
```

(Palm site is ON the fist support surface — clearance 0 is correct.) Note this must use the Fix-A
`alex_floor_z`, so hoist it out of the `floor_kwargs` gating enough to be visible in the loop, or
compute it unconditionally and only gate the repulsion kwargs. Prefer: compute `alex_floor_z`
unconditionally, gate only the mocap/`floor_kwargs` on `--floor-weight > 0`, and gate the palm
clamp on `alex_floor_z is not None`.

Optional cheap safety (same pattern, same place it's done for coplanar feet): clamp ankle role
targets to `≥ alex_floor_z + clearance`. On this clip it's a no-op (Step 0 showed ankle targets
never dip below); include it only if it stays a strict `max()` one-liner.

## Fix C — enable the Stage-3 floor repulsion (now that the floor is placed right)

No new code — run with `--floor-weight 20` (the `--coll-weight`-validated scale; dedup fix already
in). Sequence:

1. Re-run Stage 3 on the clip with Fix A+B and `--floor-weight 0` first (isolate the clamp):
   ```
   conda run -n gmr python scripts/solve_fbx_canonical_alex_contactfirst.py \
     --canonical outputs/canonical_human/fbx_fresh/luigi_standProne_03_with_orient.npz \
     --out <scratch>/s3_AB.npz --stride 1 --max-frames 99999 --ik-iters 40 \
     --contact-min-run 12 --contact-ramp 16 --contact-preroll 0 \
     --contact-on-speed-frac 0.25 --contact-onset-max-delay 0.35 \
     --coplanar-feet-mode mean --log-every 200 --floor-weight 0
   ```
   (Those `--contact-*` values are this clip's per-clip flags from the `CLIPS[]` entry — keep them.)
   Expect: palm windows (~250–268 and the right-hand equivalent) drop from ~7 cm to ≈0; LEFT_FOOT
   swing window unchanged (~11–12 cm) — B doesn't touch it.
2. Same command with `--floor-weight 20` → expect the LEFT_FOOT window to collapse (tracking and
   floor agree; the term only opposes true penetration now). Gate on the acceptance criteria below.
3. If 20 shows any residual fight (oscillation, tracking-error growth), try 5 and 10 before
   touching anything else — do NOT go above 20 (documented over-constraint regression in
   `self_collision_rows`' weight sweep).

Acceptance for the clip (all via `diagnose_floor_penetration.py --floor-z -0.1152`, plus the
solver's own printed stats):
- max floor pen **< 2 cm** (from 11.5), no new body in the worst-list that wasn't there before;
- solver `mean_err` at the logged frames within **+0.01 m** of baseline (0.0097/0.0243/0.0572/
  0.0565 at frames 0/200/400/800);
- self-collision summary not worse than baseline's 5.1% by more than a few points.

## Fix D — Stage 4 floor collision as residual cleanup

Flip `FLOOR_COLLISION=on` for the test runs (pipeline env; solver flag `--floor-collision on`).
With Stage 3 delivering ≤2 cm violations, Stage B operates in the same regime as self-collision
(≤2 cm) where its SCA/keep-best machinery is proven — the 2026-07-09 plateau (16.5→13.3 cm) was a
16 cm hole it structurally can't close (root frozen), which no longer exists. Expect final pen
≲1 cm. If Stage B's keep-best starts discarding floor-improving iterates, that's a score-tuning
issue — check the printed per-outer `pen/slip/floor_err` lines before touching code.

## Validation ladder (in order, stop at any failure)

1. **No-op check**: Fix A+B code in, `--floor-weight 0`, `FLOOR_COLLISION=off`, on a clip WITHOUT
   hand contact below floor (e.g. `shovel_fronthard_02`): output must match current baseline
   (palm clamp inert when targets are above floor; floor estimate is print-only).
2. Clip-level A+B, then +C, then +D per above.
3. **Full corpus**: `RENDER=0 bash retargetingPipeline.sh` (all 20 clips) with the chosen config,
   then `python scripts/eval_artifacts_corpus.py`. Watch: plant slip, coll%, joint-limit columns
   vs the current `outputs/artifact_table.csv` — no regression beyond ~1 cm slip / few % coll
   (the documented floor-row trade scale). Known offenders that should improve: standup_side_05
   (−28 cm swing foot), kneelingFall_02/03 (−11/−15.8 cm).
4. **Extend `eval_artifacts_corpus.py`**: the ground-contact section is feet-only — add hand
   (palm site or gripper body) penetration columns so this defect class stays visible. This is
   why a mentor's manual IsaacLab review had to find it.
5. Regenerate the two luigi clips end-to-end (grounded NPZ → 50 Hz JSON → render), visual check of
   the push phase and the forward step, hand `outputs/ihmcJsons50hz/luigi_standProne_03.json` to
   Prabin for the next mimic training run.

## Docs after validation

Update `collision.md` (outcome), `wiki/concepts/contact-first-ik.md` (floor placement + palm clamp
+ repulsion term), `wiki/concepts/globalopt.md` (floor collision), `wiki/results/metrics.md`
(before/after + hand columns), `wiki/results/tradeoffs-limits.md` (retire the swing-clip line),
append `wiki/log.md`. Defaults: flip pipeline `FLOOR_COLLISION=on` and wire Stage-3
`--floor-weight` into the pipeline (new env knob, e.g. `S3_FLOOR_WEIGHT`, default 20) only after
the corpus passes.

## Pitfalls (read before touching anything)

- **NEVER run the model-prep scripts** (`create_alex_mujoco_sites_model.py`,
  `build_alex_v2_collision_model.py`, `prepare_*`) — they overwrite the hand-maintained
  `assets/alex/alex_floating_base_with_sites.xml`. The floor geom is injected in-memory only.
- **Fixed floor reference in every comparison** (`--floor-z -0.1152` on this clip) — a run that
  lifts its plants re-registers the floor higher and fabricates regressions elsewhere.
- The `floor_gid` exclusions in `self_collision_rows` / `_build_collision` / `_collision_stats`
  must stay **unconditional** — gating them on the feature toggle re-introduces the divergence.
- One-row-per-body dedup in `floor_collision_rows` must stay.
- `luigi_standProne_03` has per-clip solver flags in its `CLIPS[]` entry — use them in standalone
  Stage-3 runs or the comparison is invalid.
- Env: `conda run -n gmr`. Coord frame +X fwd/+Y left/+Z up; quats wxyz; qpos = [x y z qw qx qy qz,
  29 joints].
- Temp outputs go to the session scratchpad, not `outputs/` (don't clobber the shipped NPZs until
  the validation ladder passes).

# Z-Grounding (Stage 4.5)

`scripts/post_process_ground_contactfirst.py`. Purely-vertical rigid shift of the free base plants the motion on z=0; joints and horizontal motion untouched. Math: METHOD.md §7.

- Per frame, compute the robot's TRUE lowest world point over all collision geoms. Mesh geoms (convex hulls): transform every hull vertex, take min z — exact. Primitives: closed-form lowest-extent formulas (sphere/capsule/box/cylinder support functions), NOT bounding boxes (bounding boxes over-correct tilted shapes → floating robot).
- Excluded: floor/worldbody geoms (bodyid 0) and non-colliding geoms.
- **`constant-contact`** (batch + script default, 2026-07-06): a single Δ for the clip, but the floor is registered to the **planted feet** — the sole-corner sites (`alex_{l,r}_sole_corner_*`) on frames where that foot is a **STILL plant** (`contact_flags`-labelled AND body speed < `--still-speed` 0.05 m/s, 2026-07-08), `floor = median` of those heights (`--contact-percentile 50`). **The stillness filter is essential and MUST match the Stage-4 plant definition** (`_compute_anchors` splits contact intervals into stationary sub-segments at the same speed and only *those* get the on-floor rows). Registering on raw `contact_flags` instead keys on the MOVING approach/transition frames too (a foot descending through a get-up, a supine-phase touch) — those sit several cm off the true stance ground and drag the shift, floating the actual stance. Fixed luigi_standSupine_08's standing stance +2.02→+0.04 cm; noop (≤0.2 cm) on clips without moving-contact pollution. This closed the get-up float that the [[contact-first-ik]] baseline-relative flat labels alone couldn't: the labels stop *tilted* phantoms from being planted, but *flat* moving contacts (a flat foot descending) still slip into raw `contact_flags` — the stillness filter is what excludes them from registration. One shift ⇒ **zero vertical wander** (no bobbing); foot reference ⇒ feet stay on the floor. Median (not a low percentile) so it keys the stable **stance**, not the brief touchdown transient (a heel-strike corner dips several cm; a low percentile there floats the whole standing phase). Falls back to `constant` if no foot-contact frames / sole sites. standup_02: feet within 0.6 cm of z=0 at clip end, shift wander 0 cm.
- **`perframe`**: Δ(t) = −z_min(t) each frame, de-jittered by implicit tridiagonal smoother (`--smooth-shift`). Plants whatever is lowest every frame — but on a get-up the lowest point migrates hands→knees→feet, so Δ(t) **wanders 7–9 cm** = the robot bobbing up/down in a fixed world frame (RDX). Superseded by `constant-contact` for the batch.
- **`constant`**: single Δ = −percentile of per-frame z_min over ANY geom; zero wander but grounds on whatever is globally lowest — during a get-up that is the early hands/knees, leaving the final feet floating (+9.8 cm on standup_02). Use `constant-contact` instead.

A rigid vertical shift is **1 DOF** — it cannot co-plant two feet the *solve* left non-coplanar; that is fixed upstream (Stage-3 `--coplanar-feet-mode` coplanar targets + Stage-4 on-floor rows; see [[contact-first-ik]], [[globalopt]]). Once the feet are coplanar, one `constant-contact` shift plants both with no bobbing.

### Measured floor-contact residual (2026-07-08, `eval_artifacts_corpus.py`)
Registering to the **median** has a quantified cost: ~half of planted-foot frames land BELOW z=0. Corpus: **planted-foot penetration median 2.5 cm, up to 6.5 cm** (present on nearly every clip incl. shovels). Foot **float is tiny** (median 0.2 cm) — median-registration errs downward, not up. **Actionable knob**: lower `--contact-percentile` (p10–p25) to plant the lower stance frames at 0, trading planted-penetration for a little float; sweep it against the float column. Separately, **swing/tucked feet clip up to 28 cm below floor** during deep-crouch phases (non-contact feet — a single rigid shift can't help), the get-up/fall grounding gap still needing geometry-aware/hybrid grounding. See [[metrics]].

Saves `qpos_ungrounded`, `ground_shift`, `ground_lowest_before/after`.

### Get-up floor residual is BETWEEN-PHASE, not within-plant drift (2026-07-08, corrected)
Per-plant-window analysis (diagnostic in `scratchpad`, not committed) overturned an earlier "within-foot Z-drift" read: on every get-up checked (luigi_standSupine_08, standup_natural_02, standupFromKneeling_01) each foot is **frozen to 0.2–0.6 cm WITHIN a plant window**; the multi-cm "spread" is entirely **BETWEEN windows at different postural phases** — the foot plants several cm lower in the low/lying/kneeling phase (root-z≈0) than in the terminal standing stance (root-z≈0.5). A single rigid shift can only zero one phase. So a Z-anchor freeze in the solver does nothing here (tried it, moved 3.51→2.89 cm, reverted) — it's real geometry, not a soft-anchor gap.
- **Consequence for the percentile knob**: the earlier "lower `--contact-percentile` to cut planted penetration" is BACKWARDS for get-ups. Their functional stance is the HIGHER foot-height phase, so a low percentile registers the lying phase and leaves the STANDING stance (the one the policy balances on) floating. Get-ups want a HIGHER percentile. luigi_standSupine_08: p50 standing floats +2.56 cm → p70 standing plants +0.12 cm (lying phase then penetrates ~3.4 cm, acceptable — non-weight-bearing). p70 also keeps luigi_standProne_03 planted (−0.15 cm). standProne_03's ground was already fine at p50; its residual is 4.2 cm XY slip (separate axis).
- **Principled general fix (unbuilt)**: register constant-contact on the **highest-root-z (terminal/standing) plant window** rather than a percentile of pooled samples — robust across get-ups with no per-clip number. See [[metrics]]. Still unbuilt as of 2026-07-10; `luigi_standSupine_08` still ships with the default p50 (not the p70 this page recommends for get-ups).
- **Related, built (2026-07-10)**: the hard floor-COLLISION term (Stage 3/4, see [[globalopt]]) hit the same between-phase problem — a single clip-wide `floor_z` misread the lying phase's legitimately-low pelvis/hip as violation. Fixed with `floor_phase_weight()` (both solver scripts): a smoothstep of pelvis/root height between the clip's low reference and its planted-foot/standing height, gating hard floor-collision on/off per frame instead of clip-wide. This is a DIFFERENT mechanism from the grounding-percentile fix above — it fixes the Stage 3/4 collision term's phase-blindness, not Stage 4.5's registration percentile, which remains the unbuilt item. `luigi_standSupine_08` now runs with `--floor-phase-aware` in both stages; see `wiki/log.md` 2026-07-10 and `SESSION_HANDOFF.md`.

### `hybrid` mode (2026-07-14, branch `p0-grounding`, uncommitted) — partial fix, does NOT close the gap
New mode in `post_process_ground_contactfirst.py`: `constant-contact` base shift + a per-frame
NON-NEGATIVE lift solved as a banded OSQP QP (`_solve_lift_qp`, min `||x-need||² + smooth·||D²x||²`
s.t. `0 <= x <= cap`). `need` = whole-body penetration depth after the base shift; `cap` per frame
= a still-planted foot's own penetration + `--lift-float-tol` (5mm default) — never let a plant
float. This is the mechanism `grounding.md` had flagged as the "principled unbuilt fix" for the
between-phase sink and the swing/tucked-foot clipping.

Tested on the two known trouble clips (this branch's fresh Stage-4 outputs, `gmr` conda env):
whole-body lowest-Z after grounding, constant-contact → hybrid:
- `luigi_standSupine_08`: −16.3cm → **−4.7cm**.
- `standup_side_05` (28cm swing-foot case): −25.9cm → **−21.2cm**, barely moved.

Real improvement, but neither fully closes to ~0. Isolated the cause on both (not a smoothness-
tuning issue — swept `--lift-smooth` 1e4→10 on luigi, residual only moved −4.69→−4.58cm): the
**still-plant cap** is what binds. At the worst frame, a *different* body part than the one
penetrating deepest is a still-planted foot with only a few mm of its own penetration, so its tiny
cap limits the frame's *whole* lift, even though some other part of the body (lying torso/hip on
luigi, a free swing foot on standup_side_05) needs far more. luigi frame 375: cap≈5.2cm (from a
still foot) vs whole-body need≈9.8cm → 4.6cm residual. standup_side_05 frame 595: cap=5mm (still
foot already flush with floor) vs need≈26cm → 21cm residual.

**Conclusion**: this is the same "between-phase" conflict as the old single-shift design (two body
parts disagreeing on floor height), just moved from whole-clip to per-frame granularity — a single
scalar lift per frame still can't satisfy two body parts that disagree *within the same frame*. The
cap is doing its job (never floats a plant) but is structurally incompatible with also fully
lifting a different, deeply-penetrating part in that same frame. Closing these two cases needs
something finer-grained than a scalar lift — e.g. per-limb correction, which is the same structural
limit M5's `refine_limbs_contactfirst.py` (phasic-v2 branch) already hit on whole-body-lying clips.
`hybrid` is safe to ship as a strict improvement over `constant-contact` (never worse, substantially
better on average) but should not be presented as solving the between-phase/swing-clip gap.

> The Mimic-ready `contact_labels (T,11)` export (11 bodies, 2 cm threshold) lives in `scripts/legacy/post_process_grounding_contacts.py` — built for the RETIRED pipeline, not yet wired into the contact-first path. See [[open-questions]].

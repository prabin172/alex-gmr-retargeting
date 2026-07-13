# Physics Plausibility Pass (Stage 4.6, phasic-v2 P3)

`scripts/physics_plausibility_pass.py`. Optional post-GlobalOPT stage: clips joint and root
velocity/acceleration into conservative bounds via a least-perturbation QP. Flag-gated
(`PHYSICS_PASS=on/off` in `retargetingPipeline.sh`, default **off** — confirmed byte-identical
output when off, see below). This page's "Mechanism"/"Limits"/"Verified behavior" sections cover
Increment 1 (plan.md's M4/T4.1), which is what ships when `PHYSICS_PASS=on`. Increment 2 (CoM
support-polygon check, T4.2) is built but disabled by default — see its own section below.

## Why a separate phase

Design philosophy (`wiki/concepts/design-philosophy.md`): "Physics-RL absorbs dynamics errors, not
kinematic impossibilities." This pass sits at the boundary — velocity/acceleration plausibility is
softer than a hard kinematic constraint (self-collision, joint limits) but harder than pure
tracking error. Kept as its own phase, after GlobalOPT (Stage 4) and grounding (Stage 4.5), so it's
independently ablatable and doesn't couple to the contact/collision machinery those stages already
own.

## Mechanism

Decision variable `δQ ∈ R^{T·35}` — MuJoCo's own tangent-space convention (6 free-root DOF + 29
actuated), NOT the 29-actuated-only slice Stage 4's own QP uses (`N_ACT`/`DV_ACT_SLICE`), since root
linear/angular vel/accel needs its own DOFs. Uses `mj_differentiatePos`/`mj_integratePos` (MuJoCo
built-ins) for velocity extraction and retraction — correctly handles the free-joint quaternion via
MuJoCo's own exponential-map machinery, no hand-rolled rotation math needed.

To first order (valid for small corrections — documented as an explicit limitation, not hidden):
velocity/acceleration are linear in `δQ`. Objective: `minimize ||δQ||²` (least perturbation) subject
to hard box inequality rows on velocity and acceleration — same banded/tridiagonal 3-point-stencil
row structure as Stage 4's `_build_smoothness_hessian`, solved via OSQP the same way `stage_b` does.

**FOOTGUN — root-angular linearization residual (2026-07-10)**: retracting `δQ` through
`mj_integratePos`'s quaternion exponential map is EXACT for every Euclidean DOF (root linear, all 29
actuated joints) but introduces a small nonlinearity residual (~1% of the bound, measured on
`luigi_standProne_03`) for root ORIENTATION specifically — the one genuinely non-Euclidean channel.
Fixed via standard bound-tightening: `_vel_bounds_internal`/`_acc_bounds_internal` shrink ONLY the
root-angular bounds by 10% (`_ANGULAR_MARGIN`) during QP construction; the REPORTED/verified/saved
limits stay at the true nominal values, giving the post-hoc check headroom to absorb the residual.

## Limits (NOT model-derived)

`assets/alex/alex_floating_base_with_sites.xml` has no `<actuator>` section — joints carry only
`range` (position) and `actuatorfrcrange` (torque), no velocity spec. Calibrated from OBSERVED peaks
on 4 representative clips post-M2/M3 (via `mj_differentiatePos`): joint_vel 3.8-12.6 rad/s, joint_acc
31-125 rad/s², root_lin_vel 0.4-0.8 m/s, root_lin_acc 1.3-2.4 m/s², root_ang_vel 0.65-2.0 rad/s,
root_ang_acc 3.7-5.7 rad/s². Defaults set at ~2-3x headroom: `JOINT_VEL_LIMIT=25`,
`JOINT_ACC_LIMIT=400`, `ROOT_LIN_VEL_LIMIT=3.0`, `ROOT_LIN_ACC_LIMIT=10.0`, `ROOT_ANG_VEL_LIMIT=6.0`,
`ROOT_ANG_ACC_LIMIT=20.0` — a **plausibility** check (catch genuine insanity a residual spike missed),
not a hardware-accurate torque limit.

## Verified behavior (2026-07-10, all 20 clips, via the real pipeline)

Gate: velocity/acceleration within true limits (post-hoc re-verified independently of the QP's own
row accounting), effector tracking delta ≤1cm RMS, 0 velocity spikes. **PASS on all 20 clips.** 17/20
are a TRUE no-op (0 rows ever engage — GlobalOPT's own smoothing already keeps output within these
conservative bounds). 3 clips (`standup_side_05`, both Luigi clips) show real, small, localized
engagement (acceleration rows only, max tracking delta 1.32cm) — consistent with the intended role:
a rarely-triggered safety net, not a routine reshaping step. Full results:
`wiki/experiments/phasic-v2-M4-gate.md`.

`PHYSICS_PASS=off` (default) confirmed byte-identical to before the stage existed (md5sum check on
the grounded NPZ + IHMC JSON export).

## Increment 2 — CoM support-polygon check (built, DISABLED BY DEFAULT)

`--enable-com` (default **off**, 2026-07-10 decision). Would nudge the whole-body CoM (XY) back
inside the support polygon during DOUBLE-SUPPORT still-plant frames (both feet simultaneously
contact-labelled + sole-centroid speed < `--still-speed`), via `mj_jacSubtreeCom` + a one-sided
soft-slack QP row per violated polygon half-space (`scipy.spatial.ConvexHull`'s half-space form
gives both the inside/outside test and the outward normal directly).

**Why disabled**: a purely kinematic CoM estimate has no mass/inertia/contact-force data behind it,
so it cannot distinguish a genuine static-balance problem from a dynamic posture legitimately
leaning on momentum. Measured directly: `luigi_standSupine_08` (a lying-to-standing get-up) has
double-support frames where the CoM sits **~40cm** outside the support polygon — a real, deliberate
part of that motion (leaning back, momentum-driven), not a retargeting defect. No small nudge should
try to close a 40cm gap. Correctly judging real-vs-momentum needs actual dynamics data, which
belongs in a later physics-aware training loop (design philosophy: "Physics-RL absorbs dynamics
errors, not kinematic impossibilities"), not this kinematic pass.

**4 real bugs found and fixed while building it** (code is correct and tested; the decision to
disable is a scoping call, not a bug-avoidance one):
1. **Single-support false positives**: an early version checked single-foot stances too (treating
   one foot's own sole as "the polygon"). On `standup_01`, 105/122 "violations" were single-support
   frames mid dynamic weight-transfer (violation depth up to 19cm) — normal locomotion, not
   imbalance. Fixed: restricted to double-support only.
2. **Independent-QP vel/accel violation**: an early version solved the CoM correction as an
   INDEPENDENT least-perturbation QP, separate from Increment 1's vel/accel QP. On
   `luigi_standSupine_08` this reintroduced a severe violation: root linear acceleration -577.7 m/s²
   vs a ±10.0 bound (58x overshoot) — an isolated large correction doesn't respect derivative
   constraints a separate problem already enforced. Fixed: refactored the vel/accel row-building
   into a shared `_vel_acc_rows()` helper, solved as ONE combined QP (CoM soft-slack rows + vel/accel
   hard rows together) — structurally impossible to violate what Increment 1 established.
3. **Solver-tolerance false positive**: after fix #2, a much smaller residual appeared (0.001-0.004
   physical units over the true bound) — acceleration rows carry a `1/dt²` coefficient (14400 at
   120Hz), amplifying OSQP's own `eps_abs=1e-5` solve tolerance when mapped to physical units. Fixed:
   `ACC_CHECK_TOL=1e-2` (documented, justified post-hoc tolerance — the true nominal limits used
   everywhere else are unchanged).
4. **Large-violation over-correction**: even after fixes #1-3, `luigi_standSupine_08`'s 9
   double-support frames (the genuine ~40cm case) still produced a correction that shifted tracked
   bodies by 8.7cm RMS / 43.6cm max — nowhere near the 1cm gate. Swept the objective weight three
   orders of magnitude; barely changed (confirming it wasn't a tuning issue — the QP genuinely
   needed ~40cm of motion to satisfy the constraint). Fixed: added `--com-max-correction` (default
   8cm) — violations beyond the cap are FLAGGED (counted, reported) but NOT corrected. This is what
   led directly to the disable-by-default decision: a real, structural sign that kinematic-only CoM
   checking can't safely handle the corpus's genuinely momentum-driven postures.

Full build/debug trail: `planLog.md` M4/T4.2.

## Output NPZ

Same schema as the grounded NPZ, `qpos` replaced with the corrected trajectory, plus
`qpos_pre_plausibility` (the input, for comparison) and `plausibility_meta_json` (limits used, rows
active, gate pass/fail; includes a `com_check` sub-object when `--enable-com` is passed).

Related: [[globalopt]] (Stage 4, upstream), [[grounding]] (Stage 4.5, upstream), [[pipeline]] (stage
list).

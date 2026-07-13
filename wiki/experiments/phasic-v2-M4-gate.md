# phasic-v2 M4 gate: physics plausibility pass (2026-07-10)

M4/T4.1 deliverable — new `scripts/physics_plausibility_pass.py`, wired as an opt-in Stage 4.6 in
`retargetingPipeline.sh` (`PHYSICS_PASS=on`, default off). Full build/debug trail: `planLog.md` M4
section. This page covers Increment 1 (joint + root velocity/acceleration box constraints), which
is what SHIPS by default when `PHYSICS_PASS=on`.

**Increment 2 (CoM support-polygon check, plan.md T4.2) was built, tested, and then explicitly
DISABLED BY DEFAULT** (`--enable-com`, defaults off) — a kinematic-only CoM estimate can't
distinguish a genuine static-balance problem from a dynamic posture legitimately leaning on momentum
(measured ~40cm "violations" on a get-up transition that were not real problems). Revisit once
physics-aware training provides actual dynamics data. Full story, including 4 real bugs found and
fixed while building it, in `planLog.md` M4/T4.2 and `wiki/concepts/physics-plausibility.md`.

## What it does

Clips joint and root velocity/acceleration into conservative bounds via a least-perturbation QP:
`minimize ||δQ||² subject to vel/accel box constraints`, decision variable δQ ∈ R^{T·35} (MuJoCo's
own tangent-space convention — 6 free-root DOF + 29 actuated), using `mj_differentiatePos`/
`mj_integratePos` for velocity extraction and retraction (correctly handles the free-joint
quaternion, no hand-rolled rotation math). Reuses the same banded/tridiagonal row-assembly pattern
and OSQP solve convention as Stage 4's own `stage_b`.

**Limits are NOT model-derived** — `assets/alex/alex_floating_base_with_sites.xml` has no
`<actuator>` section at all (confirmed by reading it, read-only). Calibrated from OBSERVED peaks on
4 representative clips (see planLog.md): joint_vel 3.8-12.6 rad/s, joint_acc 31-125 rad/s²,
root_lin_vel 0.4-0.8 m/s, root_lin_acc 1.3-2.4 m/s², root_ang_vel 0.65-2.0 rad/s, root_ang_acc
3.7-5.7 rad/s². Defaults set at ~2-3x headroom: `JOINT_VEL_LIMIT=25`, `JOINT_ACC_LIMIT=400`,
`ROOT_LIN_VEL_LIMIT=3.0`, `ROOT_LIN_ACC_LIMIT=10.0`, `ROOT_ANG_VEL_LIMIT=6.0`,
`ROOT_ANG_ACC_LIMIT=20.0`. This is a plausibility check (catch a residual spike Stage 4 missed), not
a hardware-accurate torque limit.

## Bug found and fixed during verification

`luigi_standProne_03`'s post-hoc gate initially FAILED (8 frame/dof pairs overshot the root angular
acceleration bound by up to ~1%) despite the QP itself reporting success. Root cause: the decision
variable is a first-order tangent-space model — retracting it through `mj_integratePos`'s quaternion
exponential map is EXACT for every Euclidean DOF (root linear, all 29 actuated joints) but
introduces a small nonlinearity residual for root ORIENTATION specifically. Fixed with standard
bound-tightening: internal QP construction shrinks only the root-angular bounds by 10%
(`_ANGULAR_MARGIN=0.90`), while the reported/verified/saved limits stay at the true nominal values —
so the post-hoc check (against the true limit) has headroom to absorb the residual.

## Verification

1. **Near-no-op on already-clean input**: `standup_01` (M2/M3 output) → max|δQ|=0, 0 rows active,
   0cm tracking delta, 0 spikes.
2. **Genuinely engages on real violations**: deliberately tightened limits (well below observed
   peaks) on `standup_01` → 764 velocity + 232 acceleration rows engaged, independently re-verified
   within the tightened bounds — proves the mechanism corrects, not just trivially passes.
3. **Full 20-clip run, standalone script**: `ok=20`, all pass (vel/acc within true limits, RMS
   ≤0.098cm, 0 spikes).
4. **Full 20-clip run, through the real pipeline** (`PHYSICS_PASS=on RENDER=0 bash
   retargetingPipeline.sh`): `ok=20 fail=0`. Confirms identical results end-to-end, not just via
   standalone calls. Log: `outputs/logs/pipeline_phasicv2_M4_test.log`.
5. **`PHYSICS_PASS=off` (default) is a true byte-identical no-op**: md5sum of the grounded NPZ and
   IHMC JSON export before/after an unset-`PHYSICS_PASS` run on `standup_01` — identical.

## Corpus results (`PHYSICS_PASS=on`, all 20 clips, via real pipeline)

| clip | max\|δQ\| | vel rows active | acc rows active | vel OK | acc OK | tracking RMS (cm) | tracking max (cm) | spikes |
|---|---|---|---|---|---|---|---|---|
| standup_01 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standup_02 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standup_natural_01 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standup_natural_02 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standup_side_04 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standup_side_05 | 0.0093 | 0 | 51 | True | True | 0.098 | 0.929 | 0 |
| standup_slideHandsBack_03 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| shovel_fronthard_02 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| shovel_leftbucket_02 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| shovel_lefthard_01 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| shovel_rightbucket_01 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| shovel_righthard_01 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standupFromKneeling_01 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standupFromKneeling_02 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standupKnees_02 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| standupSquatCrouch_01 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| kneelingFall_02 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| kneelingFall_03 | 0.0000 | 0 | 0 | True | True | 0.000 | 0.000 | 0 |
| luigi_standProne_03 | 0.0148 | 0 | 61 | True | True | 0.049 | 1.323 | 0 |
| luigi_standSupine_08 | 0.0144 | 0 | 196 | True | True | 0.053 | 0.916 | 0 |

## Verdict

**Gate: PASS on all 20 clips.** 17/20 clips are a true no-op (0 rows ever engage — GlobalOPT's own
smoothness term already keeps them well within these conservative bounds). 3 clips
(`standup_side_05`, both Luigi clips) show the pass genuinely engaging on ACCELERATION rows only
(never velocity) — small, localized corrections (max tracking delta 1.32cm on one clip, still well
under any concerning threshold) that don't introduce spikes or break tracking. This is consistent
with the pass's intended role: a rarely-triggered safety net, not a routine reshaping step.

**M4 status: DONE.** Increment 1 ships (default off at the pipeline level via `PHYSICS_PASS`,
verified byte-identical no-op when off, verified genuinely engages+corrects when on). Increment 2
built and documented but disabled by default — see `wiki/concepts/physics-plausibility.md`.

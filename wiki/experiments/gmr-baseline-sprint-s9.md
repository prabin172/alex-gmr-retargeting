# Sprint S9: Visual Review Finds What the Aggregate Table Hid

Continuation of [gmr-baseline-sprint-s8](gmr-baseline-sprint-s8.md), which locked
`perframelimb_smrc_rl_localground` on aggregate, corpus-scale metrics (5/6
never-tradeable axes beat `gmr_heightfix`, vMax left open as a narrowed-but-real
cost). Prabin then watched the actual renders: joint flicks a bit worse than GMR
on outlier clips, `sprint1_subject4` (sprint) specifically looks wrong — reduced
leg lift, a floating appearance, a foot flicker before the motion even starts,
and a hip in/out rotation wobble when the robot is simply standing still.
Full trail: `planLogGMR.md ## S9-*`. Plan: `GMR-S9-plan.md`.

**Status: mixed, no clean ship this sprint.** The mechanism this sprint built
(gated posture-continuity) closes most of S8's open vMax gap and the fix
*generalizes* to the full 77-clip corpus — but corpus scale also surfaced a new
cost (jerk, foot slip) the small dev-clip gate never had eyes on. Not shipped as
the new default; escalated as an open item.

## T0: root-caused the worst vMax event to a solver branch-flip

Traced `sprint1_subject4`'s worst vMax event (t=6306, 47.9 rad/s) back through
raw GMR → `perframelimb`'s clamp → final. GMR's raw retarget is flat there
(no real human motion happening) but `clamp_limb`'s phase-1 DLS — a per-frame
INDEPENDENT solve over a redundant 6-DOF chain — finds two different
null-space solutions for the same near-static target and flips between them
frame to frame (`left_hip_yaw`, `right_ankle_pitch`). Ruled out local
time-warping ("give it more frames") for this defect class: the raw target
has no real motion to spread over more time, so extra frames wouldn't change
what an independent per-frame solve computes.

Fix: null-space posture-continuity bias (`leg_floor_clamp.clamp_limb`'s
`q_prev_chain`/`posture_weight`, default `None`/no-op) — bias the leftover
null-space freedom toward the previous frame's own posture, so the primary
floor/held task still solves exactly. On the target clip alone: hip_yaw's
flip cleanly fixed, vMax -15%, but `right_ankle_pitch`'s hard-limit bang-bang
persisted and `worst_float` regressed +9%.

## T1: joint-limit repulsion — FAILED at the 2-attempt cap

Added a second null-space term (`limit_margin`/`limit_weight`) repelling each
chain DOF from its own hard limit, targeting the ankle-limit residual T0
couldn't reach. Attempt 1 barely moved the target case but clawed back most
of T0's own dev-clip regression as a side effect. Attempt 2 (pushed harder)
made the target clip's vMax *worse*, not better, while regressing further
elsewhere. Root cause: near a hard limit, which chain DOFs are "free" vs
"task-required" can itself flip depending on which side of the limit a frame
started from — a genuine kinematic branch point a null-space bias term
structurally cannot out-vote, whatever its weight. Code stays in, default
off, not re-attempted.

## New finding: T0's fix was only validated on the clip it was built for

Running T0's blanket `posture_weight=1.0` against the S8-T0b 5 dev clips
(never checked before committing to the fix) showed real, uncosted
regressions on 4/5 — `worst_float` up to +8.0cm (ground1_subject1) — because
blanket continuity pulls every frame toward the previous frame's posture,
including frames where the human is genuinely moving and "the previous
frame's posture" is simply wrong.

## T0-gate: gate the bias by the human motion's OWN velocity

Mechanism: gate `posture_weight` per chain per frame by that chain's
frame-to-frame delta in the TRUE, untouched GMR raw signal (not this pass's
own smoothed/pre-clamped input — that already carries the branch-flip
artifact and would gate backwards, confirmed by direct measurement before
shipping). Full weight when raw is near-static, zero when genuinely moving,
ramped between. Thresholds picked from data, not guessed: the diagnosed flat
window runs 0.002-0.017 rad/frame; ordinary locomotion runs 0.03-0.16
rad/frame — `lo=0.02/hi=0.05` sits below normal gait everywhere in the dev
set.

**Attempt 1 (`lo=0.02/hi=0.05`)**: dev-clip regression collapses to near-zero
(4/5 clips exactly byte-identical to shipped, ground1 down to +0.16cm from
+8.00cm) while keeping ~25% of T0's target-clip vMax win. `worst_float`'s
regression is NOT fixed (same magnitude as blanket).

**Attempt 2 (`hi=0.065`, widen to catch the worst-vMax frame itself)**:
FAILS — zero additional vMax gain, but opens a NEW +2.99cm regression on
`fallAndGetUp1_subject1` and doubles ground1's residual. 2-attempt cap spent;
dropped back to attempt 1 per Prabin's own pre-stated fallback rule.

## Full 77-clip corpus: the win generalizes, but so does a new cost

New variant `perframelimb_smrc_pg_localground` (gate1, on top of the
unmodified shipped pipeline — local grounding runs downstream, untouched).
3-way vs `gmr_heightfix` / shipped baseline, both classes:

| axis | class | shipped | pg | verdict |
|---|---|---|---|---|
| vMax_rad_s | floor | 37.39 | 34.35 | **-8.1%, generalizes past the dev set** |
| vMax_rad_s | loco | 37.92 | 36.57 | **-3.5%, generalizes** |
| joint_jerk_mean | floor | 2940.3 | 3134.8 | +6.6%, new — dev gate never tracked jerk |
| body_jerk_mean | floor | 171.0 | 193.3 | +13.1%, new |
| skate_left/right | floor | 0.63/0.51cm | 0.72/0.62cm | +13-21%, new |
| floorPen/coll/joint_ok/fidelity/n_spikes | both | — | — | wash, all safety axes untouched |

The vMax reduction is real and holds at corpus scale — genuine evidence
against the "overfit to 6 dev clips" worry. But jerk and foot slip (`skate`)
get corpus-wide worse, an axis the narrow dev gate had no visibility into.
Skate is the more concerning of the two: `gmr_heightfix` already beats this
pipeline's own clamp mechanism on skate (this project's phase-1/phase-2 DLS
clamp has no zero-slip guarantee, unlike Alex's Stage-B contact QP), and `pg`
widens that pre-existing gap further — on the exact axis ("does contact-aware
correction stop foot slip") most central to this method's own claim.

**Verdict: mixed, not shipped as the new default.** Code
(`posture_gate_lo`/`hi`/`raw_gate_qpos` in `leg_floor_clamp.py`/
`polish_median_limbwise.py`) stays in, opt-in, default `None` = byte-identical
no-op. Open item, not yet root-caused: which clips/frames drive the
jerk/skate cost, and whether it's fixable (narrower gate, per-DOF instead of
whole-chain-max delta) or an inherent trade of any null-space
posture-continuity mechanism. Not re-attempted past the 2-attempt cap;
escalated to Prabin.

## Not yet started this sprint

T2 (ramped/soft floor-margin clearance, targets idle-frame foot flicker), T3
(raw-velocity-gated smoothing weight, targets "leg doesn't lift enough"), T4
(joint constrained-QP Stage-B port, scope-only) — see `GMR-S9-plan.md`.

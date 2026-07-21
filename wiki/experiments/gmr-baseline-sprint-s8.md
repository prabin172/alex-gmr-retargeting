# Sprint S8: Physical Plausibility — Kill the Spikes, Fix the Skate, One Honest Winner

Continuation of [gmr-baseline-sprint-s7](gmr-baseline-sprint-s7.md), which promoted
`perframelimb` (S7-T3's fixed `--center perframe`) to the strongest floor-class
variant but left its corpus-scale smoothness/jerk profile unmeasured (S7-DECISION
option D). S8 measured it, found it genuinely worse than `gmr_heightfix` on 3 of 6
"never-tradeable" axes at corpus scale, and spent the sprint closing that gap.
**LOCKED at sprint end**: `perframelimb_smrc_rl_localground`, 5/6 axes.
Full trail: `planLogGMR.md ## S8-*`, `## S8-DECISION`. Plan: `GMR-S8-plan.md`.
Method writeup (plain-language + full math appendix): `GMR-METHOD.md` (repo root).

## T0-T1: held-aware smoothing attempted, gate fails twice

`perframelimb` corrects every frame independently — genuinely correct per-frame,
but consecutive frames can disagree, producing joint-velocity spikes a
contact-blind temporal smoother can't safely remove (it would erode the very
per-frame correction that makes the method work). Two smoothing attempts (a
plain held-aware lock, then a spike-aware unlock exception) both hit their
2-attempt cap without clearing the spike gate — the underlying finding: a
lock/unlock boundary trades one artifact for another (over-eager lock preserves
the spike it's supposed to fix; over-eager unlock erodes the contact correction
elsewhere), and neither ordering wins outright. `perframelimb_sm` pkls shipped
anyway (Prabin's re-framing, `## S8-T2-DECISION`) as the input to a *second*
independent fix: re-clamp after smoothing (`smrc`) — see below.

## T2c/T2-DECISION: smooth → re-clamp (`smrc`), shipped as the new base

Re-running the exact per-frame floor/self-collision clamp (`leg_floor_clamp.
clamp_limb`, unchanged math) on top of the held-aware-smoothed output restores
whatever geometry the smoothing pass perturbed. `perframelimb_smrc` becomes the
new base variant carried into the 77-clip corpus build.

## T3: 77-clip corpus contradicts the plan's predicted shape

`perframelimb_smrc` loses 3 of 6 never-tradeable axes vs `gmr_heightfix` at
corpus scale — not a handful of outlier clips (floor-class *median* floorPen
alone, 4.36cm, beats heightfix's *mean*): floorPen 6.05 vs 2.76cm, n_spikes
1.24 vs 0.18, vMax 55.4 vs 34.0 rad/s. Wins the other 3 big (coll_pct,
worst_float, joint_ok). No variant on record clears all six at once.

## T5: naive global grounding (GMR's own trick) — collapses joint_ok

Prabin's follow-up hypothesis: apply GMR's own per-clip constant-height shift
on top of `smrc`. Zeroes floorPen, keeps worst_float ahead of heightfix, zero
cost to coll/vMax/spikes/jerk/skate/fidelity — but **collapses joint_ok_pct**
(97.9%→32.7% floor) because a clip-wide constant, sized to the single worst
*transient* frame, overshoots the tight ±3cm band `joint_ok` demands on
*stance* frames. Confirms why a global shift structurally can't be the fix.

## T6: local (windowed) grounding — 4/6, sign-off given

Per-frame envelope (`max_filter1d` widen a spike into a plateau → `gaussian_
filter1d` round the corners → pointwise `max` restore) added to root z only,
near-zero except around actual penetration events. Guarantees floorPen=0
*algebraically* (not empirically — see `GMR-METHOD.md` §12.4 for the proof).
77-clip result: clears 4/6 axes (floorPen, coll_pct, worst_float, joint_ok all
win — joint_ok even *improves* past plain `smrc`, since local grounding can
only help held frames that overlapped a penetration window, never hurt clean
ones). n_spikes/vMax/jerk bit-identical to `smrc` — grounding never touches
them by construction, confirming they're a pre-existing smoothing/dynamics
gap, not a grounding cost.

## T7: smoothing-weight sweep — negative, root cause found

Tried relaxing tracking / raising smoothing regularization to trade fidelity
for gains on n_spikes/vMax. Monotonically negative (fidelity, jerk, worst_float
all get worse; vMax/n_spikes never improve, tick up slightly at the most
aggressive setting). Root cause: vMax/n_spikes come from `smrc`'s **re-clamp
step's own per-frame DLS correction**, not the smoothing weights — a
less-tracked, more-smoothed input just gives the re-clamp more work to do,
raising jerk instead of lowering it. Closes out the smoothing-weight lever;
identifies the re-clamp step itself as the real target.

## T8: rate-limited re-clamp — real gain, 5/6

`CorrectionRateLimiter` (S8-T1b's mechanism, first tried on the *original*
pre-smoothing clamp and found to convert spikes into drift) applied instead to
`smrc`'s **re-clamp step** — the actual source T7 identified. `n_spikes` flips
from a clear loss to a win/tie (0.00 vs heightfix's 0.18 floor; exact 0.00-0.00
tie loco). `vMax_rad_s` shrinks from a 63-65% gap to 9.8-15.5% — still the sole
loss, not closed. Unlike T7, `joint_jerk_mean` goes *down* (-11.3%/-7.5%),
confirming the re-clamp (not the smoothing weights) was the true mechanism.
Small real costs (worst_float +~1cm, joint_ok -0.1 to -0.45pp, skate
+0.06-0.14cm) reproduce T1b's original "spikes become drift" finding
directionally but at roughly 1/10th the magnitude — this rate limiter only has
residual smoothing-perturbation left to correct, not a full raw-to-floor-safe
correction. `perframelimb_smrc_rl_localground` — 5 of 6 never-tradeable axes,
closest any variant reached this sprint.

## T9: T4 visual veto (finally performed) + side-by-side white-floor renders

T4's visual teleport/contortion veto had been outstanding since S8 began.
Root-caused the "mujoco black and white madness" render artifact along the
way: G1's own XML ships a static checker-textured floor geom, coincident with
this project's own injected mocap floor plane — the two z-fight; not a mesh or
lighting bug. Fix: `g1_model_setup.py`'s opt-in `white_floor=True` (both
geoms recolored flat white, default off, byte-identical elsewhere). New
`scripts/g1/render_sidebyside.py` (twin-panel GMR-full vs ours, one video,
penetration overlay both sides). 5 clips × 4 frames = 20 sampled frames,
inspected directly: **passes clean** — no teleporting, no contortion, no
snapping; poses track GMR-full's own timing closely across walking, sprinting,
crouching, falling, crawling, and prone poses; floor contact 0.00cm throughout
(one frame even shows GMR-full itself penetrating 0.28cm where ours stayed
clean).

## S8-DECISION: locked at 5/6, vMax accepted as an open cost

Prabin's call: lock `perframelimb_smrc_rl_localground` as the working
baseline rather than continue chasing the 6th axis. `scripts/g1/
sprint_s8_lock_final.py` produced the canonical results file
(`outputs/gmr_baseline/sprint/s8_LOCKED_perframelimb_smrc_rl_localground.csv`,
adds corpus-wide hand slip — tracked for the first time this pass, 0.66-0.73cm
mean over 32 clips with a genuine hand-hold segment). Full table, the honest
pareto (vMax's real, open 9.8-15.5% cost), and the S7-DECISION supersession
are in `planLogGMR.md ## S8-DECISION`.

**Final locked-variant averages, 77-clip corpus** (vs `gmr_heightfix`,
floor-class / loco-class):

| axis | gmr_heightfix | LOCKED |
|---|---|---|
| floorPen_cm | 2.76 / 3.06 | **0.00 / 0.00** |
| n_spikes | 0.18 / 0.00 | **0.00 / 0.00** |
| vMax_rad_s | 34.04 / 32.82 | 37.39 / 37.92 (lose, narrowed) |
| coll_pct | 6.34 / 3.85 | **0.00 / 0.00** |
| worst_float_cm | 18.04 / 6.61 | **7.55 / 5.57** |
| joint_ok_pct | 0.19 / 46.26 | **98.85 / 98.79** |

**Supersedes S7-DECISION**: S8 is option A (promote `perframelimb`) followed
by option D (close the smoothness gap first) in the order D suggested —
`gmr_contact_fc`/`gmr_contact_fc_sm` retired from primary-method
consideration. The locked variant descends from the `perframelimb` lineage
(S6 Phase B → S7-T3 fix → S8's smoothing/re-clamp/grounding/rate-limit
stack), not from S5/S6's in-solve contact-layer mechanism.

**Next gate: S9 (mimic-training pilot)**, Prabin's to start. This kinematic
lock is necessary but not sufficient for the paper (Prabin: "if our method
wins in mimic training, then only this can become a paper") — nothing here
has been checked through a physics simulator or a learned imitation policy.

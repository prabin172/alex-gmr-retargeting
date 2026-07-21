# Wiki Index

LLM-maintained knowledge base. Read this first, then open ONLY the pages the task needs.
Math ground truth = `METHOD.md` (Alex/FBX pipeline, repo root) / `GMR-METHOD.md` (Unitree
G1/LAFAN1/GMR pipeline, repo root — branch `gmr-baseline`). Current state = `SESSION_HANDOFF.md`.
Wiki = condensed operational knowledge on top of both.

## Concepts
- [pipeline](concepts/pipeline.md) — 6 stages, scripts, batch entrypoint, env knobs
- [design-philosophy](concepts/design-philosophy.md) — feasibility > verbatim; one config for all actions; settled decisions (don't re-litigate)
- [contact-first-ik](concepts/contact-first-ik.md) — Stage 3: detection, hysteresis, blending, θ·axis flat, shank clamp, foot-hold, fist pin
- [globalopt](concepts/globalopt.md) — Stage 4: tridiagonal Stage A + contact-aware QP Stage B, soft-slack self-collision
- [morphology-scaling](concepts/morphology-scaling.md) — rest-relative delta scaling, s_root vs per-role s_r
- [orientation-frames](concepts/orientation-frames.md) — semantic frames from landmarks, world-delta transfer, facing-yaw snap
- [grounding](concepts/grounding.md) — Stage 4.5 mesh-exact z-min planting
- [alex-model](concepts/alex-model.md) — 36-DOF model, ankle asymmetry, canonical XML, FOOTGUNS
- [ihmc-export](concepts/ihmc-export.md) — KinematicsToolboxOutputStatus JSON export, 120 Hz resample
- [related-work](concepts/related-work.md) — Undermind lit landscape: unoccupied niche, 4 gaps, closest competitors

## Data
- [clips](data/clips.md) — 18-clip inventory, motion families, raw FBX layout
- [outputs-layout](data/outputs-layout.md) — outputs/ dirs, era dirs, NPZ schemas

## Experiments
- [era-ablations](experiments/era-ablations.md) — pre_shankclamp / shankclamp / onset_hyst / foothold_fix era dirs + collision weight sweep
- [fullmesh-vs-primitive](experiments/fullmesh-vs-primitive.md) — fullmesh adoption, hard-QP infeasibility, penetration-vs-slip trade
- [retired-approaches](experiments/retired-approaches.md) — hierarchical solve, hard equality, upright root, legacy worlddelta family
- [continuation-v1-gate](experiments/continuation-v1-gate.md) — Stage-4 homotopy passes, gated 1/3 clips (branch `p0-grounding`), exposed a separate pre-existing SCA-oscillation issue under `--floor-collision on`
- [gmr-baseline-week1](experiments/gmr-baseline-week1.md) — Option A kill-test on Unitree G1 (branch `gmr-baseline`): GMR out-of-box vs polished (Stage A + grounding), zero core-logic changes, clears cleanly on all 5 clips
- [gmr-baseline-week2](experiments/gmr-baseline-week2.md) — fair-baseline addendum (GMR's own height fix), E4b multi-surface pull-to-floor anchoring (CHECKPOINT, negative), self-collision vetting (passed), contact-aware grounding (negative, `constant` mode stays shipped)
- [gmr-baseline-sprint-s1](experiments/gmr-baseline-sprint-s1.md) — full 77-clip kinematic sweep (2×2's top row), DONE; corrected class split (34 floor/43 locomotion) via multi-surface contact detection, not hip-height alone
- [gmr-baseline-sprint-s2](experiments/gmr-baseline-sprint-s2.md) — OURS contact-first solver ported to G1. 77-clip corpus (S3) held-frame win **INVALIDATED by the z-shift oracle kill-test** (a per-clip constant shift of GMR-polished beats OURS too). SUPERSEDED by S4/S5 below — see that page for what actually shipped.
- [gmr-baseline-sprint-s4-s5](experiments/gmr-baseline-sprint-s4-s5.md) — S4: tried to fix OURS-DLS's floor penetration directly, best mechanism (`--swing-clear`) never cleared the joint-metric gate. S5: pivoted — GMR's own `mink` tracking is the base, `gmr_contact_retarget.py` layers a minimal held-effector contact override on top. Beats `gmr_heightfix` by 12-93 points on the joint metric, but the held-cost-only design left non-penetration ungoverned — see S6.
- [gmr-baseline-sprint-s6](experiments/gmr-baseline-sprint-s6.md) — Phase A: exact per-frame floor clamp (`leg_floor_clamp.py`, DLS on OUR mesh geometry, not GMR's) shipped as `gmr_contact_retarget.py --floor-clamp`. Beats every baseline (`gmr_raw`/`gmr_heightfix`/`gmr_polished`/S5's `gmr_contact`) on the un-gameable joint metric AND the float/penetration range on both classes, at full 77-clip corpus scale (loco floorPen 0.72cm, clears the <1cm gate). Phase B: Prabin's median-centering + limb-wise polish (`polish_median_limbwise.py`), independent retargeter-agnostic mechanism, real and working. Stacking A+B is a clean win on locomotion (range->0.10cm), a wash on the hardest floor-contact clips.
- [gmr-baseline-sprint-s7](experiments/gmr-baseline-sprint-s7.md) — Paper-readiness pass: smoothness/skate/fidelity gate found `gmr_contact_fc` trades real smoothness for its joint-metric win; `smooth_then_clamp.py` (`gmr_contact_fc_sm`) fixes it decisively (jerk below `gmr_raw`, range improves everywhere). `--center perframe` divergence root-caused + fixed (`perframelimb`, now the strongest floor-class variant shipped). Self-collision-aware `clamp_limb` fix (two-phase floor-then-collision DLS, `coll_weight=0.5`) — >99% self-collision reduction at full 77-clip corpus scale on all four affected variants, honest floorPen/joint_ok cost, method still wins decisively. OmniRetarget baseline feasibility confirmed (holosoma repo, G1+LAFAN1 support), execution pending.
- [gmr-baseline-sprint-s8](experiments/gmr-baseline-sprint-s8.md) — Closed the corpus-scale smoothness/jerk gap S7-DECISION flagged: held-aware smoothing + re-clamp (`smrc`), local (windowed) grounding envelope (algebraic floorPen=0 guarantee, `GMR-METHOD.md` §12.4), rate-limited re-clamp (root-causes vMax/n_spikes to the re-clamp step itself, not smoothing weights). **S8-DECISION: `perframelimb_smrc_rl_localground` locked** as the working baseline — 5/6 never-tradeable axes beat `gmr_heightfix` (floorPen, n_spikes, coll_pct, worst_float, joint_ok all win; vMax narrowed from a 63-65% gap to 9.8-15.5%, the one open cost), T4 visual veto passed clean. Supersedes S7-DECISION (perframelimb lineage promoted over `gmr_contact_fc`). Method writeup: `GMR-METHOD.md` (repo root, plain-language + full math appendix).
- [gmr-baseline-sprint-s9](experiments/gmr-baseline-sprint-s9.md) — CURRENT. Prabin's visual review of S8's own renders found real defects the aggregate table hid (joint flicker, `sprint1_subject4` floating/reduced leg-lift, idle foot flicker, standing-still hip wobble) — root-caused the worst vMax event to a DLS null-space branch-flip on a near-static raw target. Fix (posture-continuity null-space bias) works on its target clip but regresses 4/5 dev clips when blanket; gating the bias by the human motion's OWN raw velocity fixes the dev-clip regression and the vMax win generalizes to the full 77-clip corpus — but corpus scale also surfaces a NEW cost the dev gate never tracked (jerk +6.6-13.1%, foot slip +13-21% floor class). **Mixed result, not shipped as default** — open item (which clips drive the jerk/skate cost) escalated to Prabin. A separate mechanism (joint-limit-aware null-space repulsion, T1) failed its own 2-attempt cap outright.

## Results
- [metrics](results/metrics.md) — measured numbers per clip family, honest full distribution (incl. slip outliers)
- [tradeoffs-limits](results/tradeoffs-limits.md) — kinematics-only, residual collision/slip, overclaim risks

## Questions
- [open-questions](questions/open-questions.md) — watch items, unverified behaviors
- [publication](questions/publication.md) — Humanoids 2026 Angle A + 2027 "Any-Contact" full paper (paperIdea3.md)

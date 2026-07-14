# Wiki Index

LLM-maintained knowledge base. Read this first, then open ONLY the pages the task needs.
Math ground truth = `METHOD.md` (repo root). Current state = `SESSION_HANDOFF.md`. Wiki = condensed operational knowledge on top of both.

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

## Results
- [metrics](results/metrics.md) — measured numbers per clip family, honest full distribution (incl. slip outliers)
- [tradeoffs-limits](results/tradeoffs-limits.md) — kinematics-only, residual collision/slip, overclaim risks

## Questions
- [open-questions](questions/open-questions.md) — watch items, unverified behaviors
- [publication](questions/publication.md) — Humanoids 2026 Angle A + 2027 "Any-Contact" full paper (paperIdea3.md)

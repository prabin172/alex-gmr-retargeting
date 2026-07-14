# Scoping the "hard clips" problem — decomposition (2026-07-15)

Replaces the earlier raw note (a reward-augmented-QP sketch and the worry "then I'm doing real
mimic, not retargeting — I'm having a really hard problem scoping this"). The scoping struggle
was real because the note described three different problems as one. Decomposed, with the
evidence this week's experiments produced:

## 1. "Contact z constrained, slip x,y constrained, track as much as possible" — ALREADY BUILT

Stage 4's contact QP does exactly this: on-floor rows on Z, XY pins on planted frames, tracking
down-weighted for contacting effectors. Zero-slip was confirmed already-met during
hierarchical-v1 H2 (hard-tier A/B: slip bit-identical — Stage 4's soft QP delivers it without
new machinery). Nothing to build here.

## 2. "Null space of the contact-consistent solution space" — TESTED, DEAD

This was hierarchical-v1's premise (HQP-NLP proposal, reduced in design review, narrow version
gated): zero measurable end-to-end benefit. Continuation-v1 (2026-07-14) sharpened the lesson:
constraint machinery of ANY sophistication only helps when the solver already sits in a basin
containing a feasible solution. On the whole-body get-up clips it doesn't — every
objective/constraint enrichment tried on the wrong basin (floor-hard, hard-tier, continuation,
root-z probe) returned zero. The missing ingredient is not a better constraint set.

## 3. "Rewards: torso up, joints off extremes, stand up without slipping, tracking demoted" —
## THIS IS MIMIC WITHOUT PHYSICS. DON'T.

The original note caught it itself: "then I'm doing real mimic, not retargeting." Sharpened: a
reward-shaped kinematic QP inherits all of mimic's problems (reward design per phase, per
motion) while keeping none of its advantage (the simulator enforcing feasibility for free). The
downstream RL tracker already IS this machine, WITH physics. A kinematic imitation of it is the
weakest point on the spectrum. Settled design philosophy agrees: retargeting = kinematic
reference; downstream RL handles physics.

## The actual scope (three-way split, each piece evidenced)

- **Feasible clips** (most of the corpus): retargeting proper, pipeline as-is, tracking primary.
- **Infeasible clips** (whole-body get-ups): the STRATEGY must come from outside the QP —
  Luigi-style manual edit, recapture with robot limits in mind, or eventually physics-based
  synthesis. Not from reward terms in a kinematic solver. Evidence: Luigi's manual standSupine
  edit supplied the strategy our tracking cost can never discover (it deviates structurally from
  the human), and nothing less did.
- **Polish** (validated 2026-07-14): Stage A/4 runs on ANY strategy source, ours or external —
  "polish Luigi" took his 25.7 rad/s keyframed motion to 4.5 rad/s, floor pen 3.0→2.8cm, self-
  collision 0, at 3.3→4.4cm slip cost (`scripts/ihmc_json_to_stage4_npz.py` +
  `scripts/eval_ihmc_json.py`, wiki/log.md 2026-07-14). This piece makes the split composable
  instead of either/or.

One-sentence scope statement: **faithful when feasible, polished when given a feasible strategy,
honest about which clips need one.**

## Salvageable small items from the original note (regularizers, not rewards)

- *Joints away from extremes*: a mild limit-distance regularization is a standard IK term —
  cheap in Stage 3/4 if NECK_Y's 48–63% limit-pinning ever matters downstream. Not built.
- *Torso up*: that's the torso orientation weight (already the identified lever for the
  front-tilt finding). A single global weight = still retargeting; per-phase scheduling of it =
  reward engineering, out of scope per §3.

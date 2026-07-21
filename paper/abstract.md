# Abstract

Floor contact — crawling, falling, prone/supine lying, getting up — is
routinely excluded from human-to-humanoid motion retargeting. Across the
retargeting literature, from classical whole-body geometric IK to the most
recent contact-aware, canonical-rig, and reinforcement-learning-backbone
methods, contact modeling stops at feet and hands, and every floor
transition demonstrated on real hardware to date is an emergent behavior of
a learned policy, not the output of a retargeter. General Motion Retargeting
(GMR), the current standard tool for this step, makes the exclusion
explicit in its own evaluation, and applying it out-of-box to exactly the
excluded class produces 13–16 cm of floor penetration on up to 91% of
frames — an order of magnitude worse than its own clean-locomotion clips
(1 cm, 0.3%). Floor contact is common in fall recovery, manipulation from
the ground, and full-body interaction — all targets for humanoid deployment
— so this is not a corner case, and GMR's own admission is the sharpest
single piece of evidence for a gap that runs through the field.

We present **ContactFirst**, a contact-aware kinematic correction layer that
treats floor and self-collision contact as a first-class, per-frame
geometric target rather than leaving it to a downstream controller or
policy. ContactFirst sits on top of an existing per-frame retargeter's
output without modifying that retargeter's own solver — we build it on GMR
in this paper as a widely-used, representative backbone, not because the
gap is GMR-specific. We evaluate it with a reference-free
physics-plausibility suite built for this purpose (no ground-truth robot
motion required). On the full 77-clip LAFAN1 corpus (34 floor-contact / 43
locomotion clips), ContactFirst eliminates floor penetration and
self-collision almost entirely relative to GMR's own published height-fix
mitigation (floor class: penetration 2.76 cm → 0.00 cm, self-collision
6.3% → 0.002%) and raises an un-gameable joint contact-correctness metric
from 0.2% to 98.8% of frames — a metric we designed specifically because a
naive constant-offset baseline can satisfy simpler, single-axis versions of
the same check while leaving the robot floating well above the ground at
held frames.

We report two instantiations of ContactFirst. **ContactFirst-MS**, a
multi-stage per-frame pipeline, achieves the strongest contact precision but
carries a 9.8–15.5% peak-joint-velocity cost relative to the height-fix
baseline, traced to an architectural property of independent per-frame
contact correction: adjacent, nearly-identical frames can select different,
equally-valid null-space solutions, producing joint-velocity artifacts that
aggregate contact metrics do not surface. **ContactFirst-Global**, a
whole-trajectory reformulation, removes this artifact by construction and
reduces peak joint velocity and jerk several-fold below both baselines, at a
currently lower level of contact precision (evaluated on 36 of 77 corpus
clips at time of writing). We report both variants and their trade-off
rather than declaring one solved.

The kinematic result is necessary but not sufficient for downstream use;
imitation-policy training on the retargeted corpus is in progress and its
results, which will make the precision/smoothness trade-off decidable
against real downstream performance, are planned as part of this paper.

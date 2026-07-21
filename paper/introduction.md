# Introduction

## Motivation

Training humanoid whole-body control policies via imitation learning
requires large volumes of robot-executable reference motion. Since motion
capture is collected on humans, every such pipeline needs a retargeting step
that maps human kinematics onto a specific robot's morphology while
respecting its joint limits, self-collision geometry, and physical
plausibility.

Across this literature, floor contact — crawling, falling, prone/supine
lying, getting up — is routinely out of scope. A broader survey of
human-to-humanoid retargeting methods (per-frame kinematic retargeters,
whole-sequence optimization retargeters, contact-aware sequence retargeters,
kinodynamic refiners, and learning-based retargeters — Related Work) finds
the same exclusion pattern repeated across three decades of the field:
contact modeling stops at feet and hands, target-side feasibility editing
addresses foot placement and root trajectories but not knee, pelvis, or
forearm support, classical whole-sequence optimizers assume the robot stays
continuously balanced and in known ground contact throughout, and every
hardware demonstration of a real floor transition to date treats it as an
*emergent behavior of a learned policy* rather than the output of a
retargeter. This is not a narrow gap: humanoid robots that fall, that
manipulate objects from the ground, or that recover from being knocked over
need exactly this class of motion as training data, and if the retargeting
step that feeds their training pipeline cannot handle it, either that data
is silently dropped or a policy trains on badly-retargeted data without
anyone knowing.

General Motion Retargeting (GMR) — a widely used per-frame
inverse-kinematics retargeter (QP-based, built on `mink`) that tracks human
landmark targets on the robot subject to hard joint limits — makes this
exclusion explicit rather than implicit: its own published evaluation states
that it does "not include motions with complex interaction with the
environment, such as crawling or getting up from the floor"
\cite{araujo2025gmr}. This is a deliberate, documented scoping choice by a
current, actively maintained tool, not an oversight in an old paper, and it
is the sharpest single piece of evidence for the field-wide gap described
above: if the field's own current standard makes this exclusion explicit,
the gap is real and current, not a historical artifact of older methods.

We share this observation with a larger, forward-looking research plan for
this line of work (`paperIdea3.md`, in this repository) that scopes a full
contact-template, hardware-validated system as a 2027 target. The present
paper is deliberately narrower: it validates the first, load-bearing claim
that plan depends on — that this exclusion is a real, measurable,
field-wide gap, and that a targeted contact-aware correction layer, built on
top of an existing per-frame retargeter's output without modifying that
retargeter's own solver, closes most of it kinematically. We build this
layer on GMR in this paper because it is a widely used, representative,
and currently-maintained backbone whose own documented scope makes the gap
concretely measurable — not because the gap is specific to GMR. We return
to the larger contact-template scope in the Conclusion.

## Related Work

**Per-frame kinematic and teleoperation retargeters.** \cite{darvish2019geometric}
established geometric per-frame IK retargeting validated across multiple
humanoid platforms and remains a widely-referenced baseline pattern in this
literature. Such systems enforce joint limits and simple center-of-mass or
zero-moment-point constraints per frame, but have no explicit multi-link
floor-contact modeling and no whole-clip temporal optimization — smoothness
is whatever the per-frame solve and a downstream controller happen to
produce.

**Offline whole-sequence optimization retargeters.** \cite{ayusawa2017morphing}
solve human-model identification, human-to-robot morphing, and robot motion
planning as one simultaneous trajectory optimization, enforcing joint limits
and a simplified linear-inverted-pendulum center-of-mass/zero-moment-point
balance constraint. This is the same style of quasi-static balance check an
earlier phase of this project built and deliberately left disabled by
default (Discussion, Limitations), because a kinematic center-of-mass check
cannot distinguish genuine imbalance from momentum-carried dynamic posture —
a limitation of this entire class of check, not particular to our own
attempt at it. \cite{otani2017adaptive}, building on
\cite{difava2016multicontact}, extend this line to explicit multi-contact
retargeting (hands, feet, and environment contacts jointly) and remain a
widely-cited foundation for later contact-aware pipelines. All three are, like
ours, target-side geometric or optimization-based methods rather than
RL- or controller-side ones; none evaluate on floor-contact postural
transitions (kneeling, prone/supine, crawling, getting up) as a first-class
case, and each assumes the robot stays continuously balanced and in known
contact with the ground — an assumption every clip in our floor-contact
class violates by construction.

**Modern contact-aware sequence retargeters.** \cite{jeong2025rigunification}
unifies diverse motion sources onto a canonical rig, then refines
trajectories to enforce foot-contact constraints and stability, validated on
12 simulated and 3 real humanoids — the most recent, and methodologically
closest, prior work to ours. It is explicitly scoped to foot-fixed,
upper-body-expressive motion: its own feasibility constraint requires both
feet to remain in continuous ground contact throughout, which structurally
excludes exactly the motion class this paper targets, where the robot must
lift, reposition, and re-plant feet, or bear weight through knees, pelvis, or
a prone torso. \cite{yang2025omniretarget} (OmniRetarget) preserves
human-environment-object contact relationships through an interaction mesh
and Laplacian deformation, and is highly influential as a data-generation
backbone for later reinforcement-learning pipelines; its contact modeling,
like ours, is target-side and geometric, but its demonstrated scope is
loco-manipulation and terrain interaction, not floor-contact postural
transitions. \cite{jeong2025core} (CoRe) is philosophically closest to our
own position — fix the geometry before anything downstream has to
compensate for it — applying contact-aware optimization refinement before an
RL policy, but it too is scoped to loco-manipulation contact (foot sliding,
floating), not knee, pelvis, or prone floor support.

**Retargeting as a reinforcement-learning data-generation front end.** A
large and fast-growing body of recent work — TeleGate
\cite{li2026telegate}, the Humanoid Manipulation Interface
\cite{nai2026humi}, HumanPlus \cite{fu2024humanplus}, and the tracking-policy
ecosystem those and GMR feed, including BeyondMimic
\cite{liao2025beyondmimic} — demonstrates that floor-contact transitions
(standing up, fall recovery, kneeling) are achievable on real hardware
today. In every one of these systems the transition is a *learned policy
behavior*, executed from a reference trajectory the retargeter did not need
to get right at the contact level; the retargeter's job is only to produce a
plausible-enough reference for the policy to correct. Our contribution is
upstream of and complementary to this line of work: we ask whether the
retargeted reference itself can be made contact-correct by construction,
kinematically, before any RL policy sees it, matching this literature's own
standard practice of validating reference-motion quality as a step distinct
from policy training (Scope) — and directly motivating the policy-training
evaluation planned for this paper (Conclusion).

**A kinematic-only correction is a deliberate scope choice, not an
oversight.** An alternative to our approach is to add dynamics directly to
the correction layer: DynaRetarget \cite{dhedin2026dynaretarget} uses
sampling-based trajectory optimization to convert imperfect kinematic
retargets into dynamically feasible motions for loco-manipulation. We do not
take that route here, so that the correction layer stays solver-agnostic and
inexpensive enough to run offline over a full 77-clip corpus without
kinodynamic optimization or a physics simulator in the loop (Methods,
Overview). Whether the resulting kinematic, geometric contact-correctness
guarantee is also sufficient for dynamic trackability is exactly the open
question the planned policy-training evaluation (Conclusion) answers — this
paper's contribution stops at the kinematic guarantee, deliberately.

**Where this leaves the gap.** Across all four groups above, floor-contact
postural transitions are either out of scope by explicit assumption
(continuous ground contact, feet fixed), addressed only for feet, hands, or
manipulated objects rather than knees, pelvis, or a prone torso, or resolved
at the policy level rather than the retargeter level. We are not aware of
prior work that treats crawling, kneeling, prone/supine lying, or getting up
as a first-class kinematic retargeting target with an explicit,
non-gameable contact-correctness guarantee (Section 4). That is the gap this
paper closes.

## The gap

We first quantify the gap. Running GMR out-of-box on LAFAN1 clips it did not
validate against — clips involving crawling, falling, and getting up — we
measure 13–16 cm of floor penetration on up to 91% of frames, an
order-of-magnitude worse than GMR's own clean-locomotion clips (1 cm, 0.3% of
frames). This is confirmed at full-corpus scale (77 LAFAN1 clips, split
34 floor-contact / 43 locomotion by a multi-surface human contact detector,
not just a hip-height heuristic — hip-height alone undercounts floor contact
by missing hand/knee-supported poses during brief stumbles).

We also show that GMR's own published mitigation — a single clip-wide
constant height offset — cannot fix this: applying it removes most of the
penetration but leaves the robot floating 6–18 cm above the floor on
average during exactly the frames that should show tight ground contact,
because a single rigid shift cannot simultaneously satisfy a clip's standing
phase and its lying phase. We built a specific metric to make this failure
mode visible and non-gameable (Section 4), since floor penetration alone and
floating alone can each be minimized independently by a shift in the wrong
direction.

## Contributions

1. **ContactFirst, a contact-aware correction layer**, in two instantiations,
   that sits entirely on top of an existing per-frame retargeter's output
   (GMR in this paper) with zero changes to that retargeter's own solver:
   **ContactFirst-MS**, a multi-stage per-frame pipeline (floor and
   self-collision correction, held-aware temporal smoothing, rate-limited
   correction), and **ContactFirst-Global**, a global whole-trajectory
   pipeline (a single sparse QP jointly enforcing floor/collision
   correctness, contact anchoring, and temporal smoothness), both
   terminating in a local grounding envelope with an algebraic
   zero-penetration guarantee.
2. **A reference-free physics-plausibility evaluation suite** — no motion-
   capture ground truth on the robot is available, so every metric we report
   (floor penetration, self-collision, contact correctness, tracking
   fidelity, smoothness) is computed from the retargeted motion and the
   robot's own geometry alone.
3. **A non-gameable joint contact-correctness metric** (Section 4): a naive
   constant Z-shift can independently minimize floor penetration or
   eliminate floating, but not both at the same held frame simultaneously —
   our metric requires both, closing a gaming loophole present in simpler,
   single-axis versions of the same check.
4. **Full 77-clip LAFAN1 corpus validation**, floor/locomotion class split,
   against GMR's own published height-fix baseline.
5. **An honest architectural finding and open trade-off**: per-frame
   independent contact correction can select different, equally-valid
   null-space solutions on adjacent frames, producing a visible "branch-flip"
   artifact that aggregate corpus metrics do not surface — caught only by
   watching renders. We trace this to the correction's lack of temporal
   coupling, characterize a partial per-frame mitigation, and report results
   from an alternative whole-trajectory optimization formulation that removes
   the artifact structurally but currently trails the per-frame approach on
   contact precision. We present this as an open problem, not a solved one.

## Scope

This is a module/polish paper, not a new solver. We do not replace GMR's own
tracking IK; we validate that a targeted contact-correctness layer on top of
it closes a real, documented gap in its applicability, and we report the
result honestly including where it does not yet fully close. Kinematic
plausibility — physically executable, contact-correct, smooth motion — is
necessary but not sufficient for downstream use, consistent with the
standard practice in this literature of validating retargeting quality
before the separate, more expensive step of training a tracking policy on
it. Imitation-policy training on the retargeted corpus, which closes that
gap, is in progress at the time of writing and its results are planned as
part of this paper (Conclusion).

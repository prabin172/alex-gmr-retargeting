# Introduction (draft) — Any-Contact Retargeting for Whole-Body Floor-Contact Motions

*Draft v0.1, 2026-07-03, for the Humanoids 2027 full paper (`paperIdea3.md`). Structure:
intro → background → literature → gap → contributions. Citation keys are placeholders for the
eventual BibTeX; `[TBD-x]` marks numbers/claims that require experiments not yet run. Written under
the overclaim discipline of `wiki/results/tradeoffs-limits.md` — every quantitative claim in the
final version must trace to a table.*

---

## I. INTRODUCTION

A humanoid robot that falls must get back up. Between the upright behaviors that dominate current
humanoid research — walking, loco-manipulation, expressive whole-body motion — and the floor lies a
class of motions the field has quietly stepped around: kneeling down and standing back up, crawling,
transitioning through prone and supine postures, recovering from a fall, and working from a kneel or
a squat. These *whole-body floor-contact motions* are not exotic: they are the entry and exit of
nearly every failure mode, and they are daily work for humanoids intended for industrial and
domestic settings. They are also precisely the motions where the dominant paradigm for acquiring
humanoid skills — imitating retargeted human motion capture \cite{peng2018deepmimic, liao2025beyondmimic} —
breaks down, because the retargeting step itself does not support them.

**Background.** Human-to-humanoid retargeting maps captured human motion onto a robot with
different limb proportions, joint limits, and link geometry. Classical retargeters solve per-frame
inverse kinematics with joint-limit and balance constraints \cite{naksuk2005wholebody,
montecillo2010humanoidnorm, darvish2019geometric, oh2019realtime}, a formulation inherited from
real-time teleoperation, where future frames are unavailable. More recent work exploits the offline
setting: whole-sequence kinematic optimizers enforce contact consistency and temporal smoothness
over the entire clip \cite{ayusawa2017morphing, gomes2019wholebody, jeong2025rigunification,
yang2025omniretarget, araujo2025gmr}, kinodynamic refiners impose dynamic feasibility through
torque and contact-force constraints \cite{wang2026spark, zhang2026kdmr, dhedin2026dynaretarget},
and learning-based retargeters compress the mapping into latent or generative models
\cite{choi2020lwl2, choi2021s3le, chen2025ikmr, zhang2026adamorph}. In parallel, retargeting has
become the data-generation backbone of RL-based whole-body control: large pipelines retarget human
datasets and train tracking policies that absorb residual kinematic error
\cite{he2024h2o, he2024omnih2o, fu2024humanplus, cheng2024exbody, ze2025twist, liao2025beyondmimic}.
Across all of these lines, one empirical point is now well established: reference-motion quality
directly limits downstream policy success — retargeting artifacts such as foot sliding,
interpenetration, and velocity spikes measurably degrade tracking policies \cite{araujo2025gmr}.

**The problem.** For floor-contact motions, every existing pipeline delegates the hard part.
Consider a human standing up from a kneel: support transfers from two shins and a foot, through a
half-kneel, to two feet, while the trunk sweeps through configurations that a morphology-constrained
robot — with, for our platform, an ankle allowing 60° dorsiflexion but only 30° plantarflexion, no
ankle yaw, and a rigid foot — often cannot reproduce verbatim. General-purpose retargeters exclude
this class outright: the authors of GMR state that they "do not include motions with complex
interaction with the environment, such as crawling or getting up from the floor"
\cite{araujo2025gmr}. Contact-aware sequence retargeters model feet, hands, and manipulated objects,
but not knees, shins, pelvis, or forearms as support contacts \cite{yang2025omniretarget,
jeong2025rigunification, jeong2025core}. And where floor transitions *do* appear on real hardware —
standing up and fall recovery in teleoperation frameworks \cite{li2026telegate, nai2026humi,
fu2024humanplus}, kneeling manipulation \cite{nai2026humi}, acrobatic ground contact
\cite{huang2026omnitrack} — the transitions are *emergent behaviors of RL policies*, not the output
of a retargeter: the reference motion is rough, and a learned controller is trusted to invent the
contact geometry. To our knowledge, no published method retargets long floor-contact sequences
kinematically, with explicit multi-link contact geometry, such that the result can be consumed by a
standard whole-body control stack — a conclusion supported by a systematic survey of the 2005–2026
retargeting literature we conducted across ~170 works.

**The gap, precisely.** Four ingredients are missing from the literature, individually and — more
importantly — in combination. (1) *Contacts beyond feet and hands:* no retargeter treats knees,
shins, pelvis, or elbows as first-class contacts to be detected in the human data and imposed on
the robot. (2) *Robot-feasible floor-stance modeling:* no method precomputes which kneeling,
all-fours, or prone support configurations a given robot can actually attain under its joint limits
and collision geometry, nor maps human floor postures onto such robot-specific stance templates.
(3) *Target-side feasibility editing for floor postures:* existing target editing addresses foot
placement, root trajectories, and velocity profiles \cite{jeong2025rigunification, jeong2025core,
tu2025samretarget}; reconciling an infeasible *limb* configuration during ground support is left to
solver-side weight fights or to RL. (4) *Hardware-ready kinematic output for this motion class:*
every real-robot floor transition demonstrated to date relies on RL or a bespoke controller to
"make it work"; none plays a retargeted floor-contact clip through a standard stabilization stack.

**Our approach and contributions.** We present *Any-Contact Retargeting*, a purely kinematic,
contact-first retargeting pipeline for whole-body floor-contact motions, instantiated and evaluated
on the IHMC Alex humanoid — a 36-DOF industrial biped — and demonstrated on get-up, kneeling,
falling, and ground-work sequences captured from human operators. Our design principle is that
kinematic infeasibility must be fixed in the *targets*, before solving, rather than fought with
soft weights or deferred to a learned controller: where the human's support configuration is
unreachable, we edit the reference into the nearest robot-feasible support configuration and let
the solver track something attainable. Concretely, we contribute:

1. **A generalized contact-first formulation** in which every support link — soles, fists, knees,
   shins, pelvis — is detected from the human capture (height, velocity, and orientation gates with
   onset hysteresis) and imposed on the robot through smoothly cross-faded support-face alignment
   and position-anchor tasks, overriding the captured limb orientation wherever it conflicts with
   physical support.
2. **A robot floor-stance template library and mapping**: an offline enumeration of the robot's
   feasible floor-support stances (double-kneel, half-kneel, all-fours, prone-on-elbows, seated,
   squat) under its true joint limits and collision geometry, together with a per-interval
   classifier that assigns detected human support phases to templates and a family of geometric
   target edits — generalizing our shank-tilt clamp, which projects knee targets into the
   flat-foot-reachable cone of a range-limited ankle — that reshape the human reference into each
   template's feasible manifold.
3. **A contact-anchored whole-clip optimization with quasi-static stability**: a closed-form
   temporal smoother followed by a sparse convex QP over all frames that pins each stationary
   support sub-interval to a median anchor, maintains soft-slack self-collision separation on the
   robot's full convex-hull collision body, and constrains the ground-projected center of mass to a
   conservatively shrunk support region of the active template during static support phases — linear
   rows that preserve convexity, deliberately avoiding torque-level kinodynamic optimization.
4. **A hardware-ready evaluation on the motion class the field excludes.** We retarget [TBD-N]
   floor-contact clips under a single untuned configuration and evaluate three ways: (i) kinematic
   quality (penetration, plant slip reported as a full distribution, limb-flatness, joint-limit
   margin, velocity spikes) against faithfulness-first and contact-aware baselines
   \cite{araujo2025gmr, yang2025omniretarget}; (ii) downstream policy success, training imitation
   policies per clip and method under the protocol of \cite{araujo2025gmr, liao2025beyondmimic}
   [TBD-policy-results]; and (iii) direct playback of retargeted clips through the robot's standard
   whole-body kinematics stack on hardware [TBD-hardware-results] — evidence, to our knowledge
   unavailable for any prior retargeter on this motion class, that the output is consumable without
   motion-specific learning.

The method is kinematic by design: we do not claim dynamic feasibility, and we quantify the residual
trade-offs — plant slip of [TBD] cm at the [TBD] percentile and residual self-collision on the
hardest clips — rather than eliminate them. Our results indicate [TBD — one-sentence headline after
experiments: e.g., "that contact-first references substantially improve policy success on
floor-contact motions relative to faithfulness-first retargeting, and that a purely kinematic
pipeline with quasi-static stability editing is sufficient for hardware playback of kneel and
stand-up sequences"].

---

*Notes for revision (not part of the paper):*
- *The "~170 works survey" sentence footnotes the Undermind report; decide whether to cite it as a
  technical report or fold into Related Work prose.*
- *GMR exclusion quote verified against `paperIdea2.md` §1 / the GMR paper; re-verify exact wording
  before submission.*
- *Keep the ankle-range aside in ¶3 — it grounds the abstract problem in one concrete number pair
  early, which reviewers of this venue respond to.*
- *Contribution 4's "to our knowledge unavailable" must survive a fresh lit pass at submission time
  (12 months is long in this field — re-run the search ~May 2027).*
- *If BeyondMimic-on-Alex falls to the G1 fallback (see `paperIdea3.md` §5), contribution 4(ii)
  wording changes from "the robot" to "two platforms."*

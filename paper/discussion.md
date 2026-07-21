# Discussion

## Why contact-first correction beats a global shift, structurally

The result in Results §2–4 is not just an empirical win, it follows from a
structural argument. A single per-clip constant offset has exactly one
degree of freedom to satisfy every phase of a clip at once. Any clip that
transitions between a standing phase and a lying phase (nearly every
floor-contact clip in this corpus) has, by construction, no single value of
that one degree of freedom that is correct for both phases. Our method
instead computes a per-frame correction driven by that frame's own detected
contact state, so it has as many degrees of freedom as the clip has frames —
it is not "better tuned," it is solving a different, correctly-posed
problem. This is why we designed `joint_ok%` to require simultaneous
held-frame accuracy and whole-body clearance at the same frame: it is
specifically the property a single global shift cannot have, and confirming
our method has it (§4) is the load-bearing validation of the paper's central
claim, not a secondary metric.

## The smoothness/precision trade-off is not incidental

The peak-velocity cost in Results §3 is worth dwelling on rather than
minimizing, because we believe it is a general property of any per-frame
independent contact-correction scheme, not an implementation defect specific
to ours. A per-frame solve over a redundant kinematic chain has, by
definition, a null space — directions it can move in without changing the
frame's own task error. Nothing in a per-frame objective distinguishes
between two null-space solutions that are both locally optimal for that
frame alone, so which one the solver lands on can be an artifact of warm-start
sensitivity rather than a meaningful choice. Two nearly-identical human
target frames can therefore produce two very different robot poses. This
does not show up in floor-penetration or contact-correctness metrics at
all — both frames may be perfectly contact-correct — it shows up only as an
implausible joint-velocity spike between them, visible in rendered video but
not in the aggregate corpus table. This is a methodological point beyond our
own results: aggregate
kinematic-plausibility metrics, including several we report ourselves, are
necessary but not sufficient evidence of retargeting quality, and spot-check
rendering should be treated as part of the evaluation protocol, not an
optional sanity check.

Our two responses to this — a gated null-space posture-continuity bias
within the multi-stage pipeline (Results §3) and the global pipeline
(Results §5) — sit at different points on the same trade-off, and we think
that trade-off
is informative rather than merely unresolved. The per-frame method, even
patched, keeps the advantage of a very large total solve budget: a full
iterative local solve at every one of thousands of frames. The
whole-trajectory method removes the branch-flip by construction, because its
smoothness term lives in the same objective as its contact and collision
terms — a flipping solution costs the objective directly — but it currently
gets only a handful of whole-clip outer iterations, bounded by a trust
region for numerical stability, and so does not yet match the per-frame
method's contact precision. We do not think either point is simply "better";
we think a system that needs both properties (temporal consistency and
contact precision) at once is a genuine open problem, and we report both
sides of it rather than picking the one that makes a cleaner headline number.

## A second, related failure mode: scoring a fix by a metric it cannot move

The global pipeline's convergence behavior on severe floor-contact
clips (Results §5, Methods §3) has a specific, generalizable lesson. That solver's
decision variable is the robot's actuated joints only; it does not, and by
its formulation cannot, move the robot's root/pelvis position. On a clip
whose floor violation is severe enough that no joint-only articulation
around a fixed pelvis can resolve it — a genuine reachability limit, not a
convergence failure — every candidate correction the solver considers will
score badly on floor penetration, including candidates that make real
progress on a different violation (self-collision) the solver *can* fix. A
convergence procedure that scores the *whole* problem, including the part it
structurally cannot affect, will silently refuse to accept improvements to
the part it can. This is a caution for any staged pipeline: each stage's own
internal acceptance criterion should be scoped to what that stage's decision
variable can actually change, or a downstream stage that already owns the
rest of the problem (here, a separate grounding pass that guarantees zero
floor penetration by construction regardless of this stage's outcome) can be
silently starving an upstream stage of improvements it should have kept.

## Limitations

- **Kinematic-only validation, so far.** Every metric reported in Results is
  a kinematic-plausibility check computed without a physics simulator or a
  downstream tracking policy in the loop — necessary evidence that a motion
  is not obviously broken, not sufficient evidence that a policy trained on
  it will succeed. This is not a corner we cut alone: it is a limitation of
  the entire kinematic/geometric retargeting literature we compare against
  (Related Work) — even \cite{ayusawa2017morphing}'s zero-moment-point balance check,
  the closest any prior kinematic retargeter comes to modeling dynamics,
  assumes quasi-static balance rather than real contact forces. An earlier
  phase of this project built and deliberately disabled-by-default a
  kinematic center-of-mass/support-polygon check for the same underlying
  reason: it cannot tell a genuine loss of balance apart from momentum-
  carried dynamic posture, so a kinematic-only signal here would either miss
  real failures or flag correct, dynamic motion as broken. The
  policy-training evaluation that closes this gap (§Future Work) is planned
  as part of this paper, not deferred to separate future work; it is not yet
  complete at time of writing.
- **A tracking-fidelity trade-off, deliberately accepted.** Our method's
  FK-tracking error against GMR's own scaled human targets is slightly worse
  than GMR-plus-heightfix's, because we prioritize contact correctness over
  verbatim reproduction of the (sometimes physically wrong) input signal.
  This is a design choice we believe is correct for this use case, not a
  free win.
- **A chain of independently-motivated fixes, not one unified optimization.**
  The shipped pipeline (Methods §1–6) is staged: each stage was added to
  fix a specific, measured problem the previous stages left behind. This
  gives good explainability — every stage's necessity is backed by a
  before/after measurement — but it is not the single clean formulation a
  from-scratch design might produce; the global pipeline (Methods §3,
  Results §5) is a step toward one.
- **One robot, one human motion source.** All results are on the Unitree G1
  humanoid retargeting LAFAN1 motion capture. We have not yet validated
  generalization to a second robot morphology or a second motion-capture
  source in this paper.
- **The multi-stage pipeline's smoothness mitigation (Results §3) has its
  own cost**, a secondary regression in jerk and contact-slip on part of the
  floor class, visible only at full-corpus scale and not on a small
  development set. We report full-corpus numbers throughout this paper for
  this reason: no comparative claim is made without the complete-corpus
  measurement behind it.

## Relation to a larger planned system

The contributions here are the load-bearing first step of a larger research
direction sketched in `paperIdea3.md` (this repository): a full
contact-template system generalizing the single foot/hand contact set used
here to knees, pelvis, and forearms, mapped onto a library of
robot-feasible floor-support stances, validated by downstream policy
training and real-hardware playback. We deliberately did not attempt that
larger scope in this paper. The central empirical claim that larger system
depends on — that GMR's floor-contact exclusion is a real, measurable,
correctable gap, and that a targeted contact-aware layer closes most of it
kinematically without touching GMR's own solver — is exactly what this paper
validates, at full-corpus scale, before committing to the larger system's
cost.

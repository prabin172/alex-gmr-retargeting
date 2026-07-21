# Methods

## Overview

Our pipeline consumes a human motion-capture clip and GMR's own per-frame
retarget of it, and produces a contact-corrected, smoothed, floor-safe
trajectory for the Unitree G1 humanoid. Every stage after GMR's own solve is
ours; GMR itself (its tracking cost, its joint-limit handling, its solver) is
used completely unmodified. We present two instantiations of the correction
layer that sits on top of GMR's output: a **multi-stage** pipeline built from
a sequence of per-frame corrections (§2), and a **global** pipeline that
replaces that sequence with a single whole-trajectory optimization (§3). Both
share the same canonical-human contact labeling (§1), the same evaluation
protocol (§4), and the same final grounding step. §5 compares them.

Throughout this paper, "contact-aware" and "contact-first" refer to
*geometric* contact handling: per-frame detection of which effector is in
contact (§1), and enforcement of floor-clearance, held-position, and
self-collision constraints derived from that detection. Neither pipeline
models contact forces, friction, or joint torque — the dynamics of contact
are outside this paper's scope by design (Introduction, Related Work), the
same scope every kinematic retargeter we compare against shares.
\cite{ayusawa2017morphing}'s simplified zero-moment-point check is the closest any of them
come to a dynamics proxy, and even it assumes quasi-static balance rather
than modeling contact forces directly. The planned policy-training
evaluation (Conclusion) is the step that tests whether a geometrically
contact-correct reference is also dynamically trackable; this paper's
contribution stops at the kinematic guarantee.

## 1. Canonical human and contact detection

Raw BVH motion capture (LAFAN1, 30 fps) is mapped to a robot-agnostic
canonical skeleton: per-frame landmark positions and orientations derived
from bone geometry, with hand orientation taken directly from GMR's own
raw-bone-rotation signal to preserve wrist twist. Contact is a per-frame
detection gate, not a correction: an effector is flagged "in contact" only
when it is simultaneously below a height threshold and moving slower than
0.4 m/s (7 cm for feet, 8 cm for hands) — height alone is insufficient, since
a fast-swinging limb can pass close to the floor without being planted.
Crossing the threshold marks a frame as a candidate for anchoring by a later
stage; it does not move anything by itself. This detector covers feet and
hands and is shared verbatim by every downstream stage that needs to know
which frames are in contact.

For corpus-level evaluation, clips are additionally classified into a
floor-contact class and a locomotion class using a broader six-region human
contact labeler (feet, hands, knees, elbows, pelvis, torso; 5/8/15 cm
respectively), since a feet-only or hip-height signal undercounts floor
contact on clips where support is briefly taken through a hand or knee.

## 2. Multi-stage pipeline

**Floor and self-collision clamp.** For each frame, a two-phase sequential
damped-least-squares correction runs per limb chain (hip→knee→ankle,
shoulder→elbow→wrist), proximal-to-distal. Phase 1 resolves floor violations
and pins held effectors to their contact target; Phase 2, run strictly
afterward rather than jointly with Phase 1, resolves self-collision using the
same contact-normal and relative-Jacobian formulation as the collision term
in the global pipeline (§3).

**Held-aware smoothing.** A closed-form per-joint tridiagonal smoother
removes joint-velocity spikes. Held-effector degrees of freedom are locked to
their input value at high weight during contact, ramped in and out over five
frames, so smoothing does not erode a contact correction the clamp stage
already made. The floor/self-collision clamp is re-applied once more after
smoothing, since geometry-blind smoothing can reintroduce a violation at a
non-held frame.

**Rate-limited correction.** A temporal trust region on the applied
correction bounds how fast it can change from one frame to the next,
independent of the pose itself. This mitigates branch-flip artifacts
(discrete jumps between distinct, equally-valid per-frame solutions on a
redundant kinematic chain — Discussion) without addressing their cause.

**Grounding.** See §4 below (shared with the global pipeline).

## 3. Global pipeline

The global pipeline replaces the clamp / smoothing / rate-limit sequence in
§2 with one whole-trajectory optimization. GMR's raw per-frame output is
consumed directly as input.

**Floor and self-collision correction.** A sparse QP over the actuated joint
trajectory of the entire clip, solved by sequential convex approximation: at
each outer iteration, floor and self-collision constraints are linearized
into soft, slack-penalized rows (never primal-infeasible), a trust region
bounds the step, and a smoothness term penalizes the change in the SCA
correction between adjacent frames, in the same objective as the floor,
collision, and contact-tracking terms. Held effectors are anchored to a
per-region target: contact timing comes from the canonical human's own
labels (§1), a stillness sub-check on the robot's own trajectory splits each
labeled contact interval into stationary sub-runs (anchored to a single
position, ramped in/out over five frames) and repositioning sub-frames
(lightly regularized toward their own per-frame position), and the anchor's
target height is a fixed per-effector geometric constant (the robot's own
origin-to-support-point offset at a neutral pose), not a value computed from
the possibly-unreliable input trajectory. Floor penetration is intentionally
excluded from this stage's own convergence scoring, since its solve variable
is the robot's joints only — a pelvis-level violation on a floor-contact
pose is outside what this stage can correct, and grounding (§4) resolves it
regardless of this stage's outcome; scoring against it would only reject
otherwise-improving self-collision corrections. A cross-iteration keep-best
selection guarantees the result is never worse than GMR's own raw output on
this stage's own objective.

**Smoothing and grounding.** The corrected trajectory passes through the
same held-aware smoothing stage used in §2, then grounding (§4). No
per-frame re-clamp or rate limiter is applied — the whole-trajectory
formulation's own smoothness term is the temporal-coupling mechanism, in
place of the multi-stage pipeline's rate limiter.

Because its smoothness term shares an objective with its floor, collision,
and tracking terms, a solution that alternates between distinct null-space
branches on adjacent frames costs the objective directly; the global
pipeline cannot produce a branch-flip by construction, independent of any
per-frame tuning.

## 4. Grounding and evaluation

**Grounding.** A final, deterministic pass computes the per-frame vertical
shift required to clear the floor, applies a causal max-filter and Gaussian
smoothing to that requirement, and takes the pointwise maximum against the
raw requirement. This guarantees the shifted trajectory has zero floor
penetration by construction. Unlike GMR's own height-fix — one rigid
per-clip shift — this is computed per frame, so it does not force a single
Z-offset to serve both a clip's standing phase and its lying phase.

**Metrics.** No ground-truth robot motion exists, so every metric is computed
from the output trajectory and the robot's own vetted collision geometry:
floor penetration and self-collision incidence (mesh-exact lowest point vs.
z = 0, k-hop-filtered for anatomically adjacent pairs); worst float (height
of a held effector's support point above the floor); tracking fidelity
(FK'd body position/orientation vs. GMR's own scaled human target); jerk,
peak joint velocity, and velocity-spike count; and skate (horizontal drift
of a held contact point from its own contact-onset position).

**Joint contact-correctness.** A held frame passes only if its support point
is simultaneously within a tight band of the floor and the whole-body mesh
clears the floor everywhere at that same frame. Both conditions are
necessary: a metric that checks either alone is satisfiable by a per-clip
constant Z-shift with no notion of per-frame contact at all, since a rigid
shift can be tuned to land in the right place at a clip's labeled held
frames while leaving the rest of the clip penetrating or floating
arbitrarily. Requiring both simultaneously is not satisfiable by any single
rigid shift on a multi-phase clip (Results, oracle comparison), which is
why we report this joint metric as the primary measure of contact quality
rather than penetration or float in isolation.

## 5. Comparing the two pipelines

The multi-stage pipeline (§2) benefits from a large total per-frame solve
budget — a full iterative local correction at every frame — at the cost of
no explicit temporal coupling between frames, which manifests as branch-flip
artifacts and their associated joint-velocity cost. The global pipeline (§3)
removes that artifact by construction but currently allocates a smaller
total solve budget (a bounded number of whole-clip outer iterations) to
contact precision. Results reports both.

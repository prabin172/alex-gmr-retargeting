# Paper Idea: Global Trajectory Optimization for Offline Humanoid Motion Retargeting

**Target**: IEEE-RAS International Conference on Humanoid Robots (Humanoids 2026)
**Key Dates**:
- July 24, 2026 — Paper submission deadline (25 days from June 29, 2026)
- July 27, 2026 — Supplementary video deadline (optional)
- October 6, 2026 — Notification of acceptance/rejection
- October 26, 2026 — Camera-ready deadline
- December 6–9, 2026 — Conference (paper presentations Dec 8–9)

**Status**: Idea / early pipeline — retargeting pipeline working end-to-end on IHMC Alex

---

## The Core Insight (one paragraph)

The widely-used teleoperation-oriented retargeting methods (PHC, ProtoMotions, GMR) use per-frame
IK. The implicit justification is real-time teleoperation: you only have present and past frames, so
global optimization over the whole trajectory is impossible. But when building a *training dataset*
for an imitation learning policy (Mimic, BeyondMimic), the entire motion sequence is available
upfront, and the teleoperation constraint no longer applies. Removing it enables global trajectory
optimization, which produces smoother, more physically coherent reference trajectories.

> **Novelty caveat (added after literature pass, June 2026):** This offline / whole-trajectory
> framing is NOT new. Several 2024–2026 papers already formulate retargeting as whole-sequence
> optimization for offline imitation-learning reference generation, and at least one explicitly
> critiques frame-by-frame retargeting. See the **Related Work & Novelty Risk** section below.
> The "first to exploit offline context" claim is **not defensible** and has been removed. What
> remains potentially novel is narrower: a lightweight convex kinematic-only refiner usable as a
> retargeter-agnostic post-process, and the floor-contact / get-up motion class.

---

## Problem Statement

### What is wrong with per-frame IK for offline dataset generation?

Per-frame QP/IK solves each frame independently, initialized from the previous frame's solution.
The only temporal coupling is a soft posture regularization pulling each frame toward the previous
one — backward-looking and weak. This produces three known artifacts (identified in the GMR paper):

1. **Ground penetration** — floating or sinking root
2. **Self-intersection** — limbs passing through each other
3. **Velocity spikes (joint flicks)** — single frames where a joint jumps 1-2+ radians because
   the IK converges to a different local minimum (topology flip: elbow configuration flipping,
   shoulder config switching, etc.)

The GMR paper shows empirically that all three reduce policy success rates and tracking performance.
They fix (1) and (2) with better per-frame methods, but (3) persists in their Dance 5 result
(waist roll jumps) and causes a visible drop in success rate.

### Why per-frame methods cannot fix (3) by construction

To distribute a large joint change over several frames, the solver at frame t-3 would need to
"know" that a topology change is coming at frame t. In per-frame IK this is impossible — each
frame only sees its own target and the previous frame's q. The correct formulation is a joint
optimization over all T frames:

```
minimize  Σ_t [ λ_track · ||FK(q_t) - target_t||²
              + λ_coll  · collision_cost(q_t)
              + λ_smooth · ||q_t - q_{t-1}||² ]
subject to joint limits at all t
```

The smoothness term `||q_t - q_{t-1}||²` couples all frames bidirectionally. The gradient flows
backward: if frame 100 has a big jump, the optimizer nudges frames 95-99 to pre-lean, distributing
the change. Per-frame IK has no equivalent mechanism.

### The key observation that makes this tractable

For teleoperation you cannot do this — future frames are unknown. But for offline dataset
generation you have the full trajectory. This observation motivates the global formulation, but
it is **not** by itself novel: the whole-trajectory framing for offline retargeting already exists
in the literature (SPARK, KDMR, IKMR, STMR — see Related Work). The contribution, if any, must
come from the *specific* method (a cheap convex kinematic-only refiner usable as a wrapper) and the
motion class (whole-body floor contact), not from the framing.

---

## Proposed Method: Two-Stage Pipeline

### Stage 1 — Per-frame QP IK (warm start)
Exactly the current pipeline: damped least-squares QP IK in MuJoCo velocity space, with
world-delta orientation transfer, per-role morphology scaling, and soft self-collision repulsion
(our w=20 constraint). This gives a good warm-start trajectory q^(0) that already satisfies
joint limits and is roughly correct.

### Stage 2 — Global Trajectory Refinement
Starting from q^(0), run a global optimizer over all T frames jointly.

**Tractable formulation (convex core):**
If the collision term is dropped and only tracking + smoothness are retained, the per-joint
problem decouples into T independent 1D signal smoothing problems. This is equivalent to a
Tikhonov-regularized signal — closed-form solution via tridiagonal system, solvable in O(T)
per joint. For 29 actuated joints × 1500 frames, this takes milliseconds.

**Adding collision back (iterative):**
Use the same contact-Jacobian rows from Stage 1. At each outer iteration:
  1. Linearize collision constraint at current trajectory
  2. Solve the global QP (now still convex, just larger)
  3. Re-run mj_forward on all frames to update contact geometry
  4. Repeat until convergence (typically 3-5 outer iterations in practice)

This is a Sequential Convex Approximation (SCA) / Linearize-and-Solve loop — standard in
trajectory optimization and well-understood.

**What this fixes:**
- Velocity spikes: smoothness term distributes large changes across neighboring frames
- Residual collisions from Stage 1: collision rows are now globally consistent
- Does NOT change the fundamental tracking objective — just imposes temporal coherence

### Why wrapping is the right architecture

The Stage 2 refiner takes *any* per-frame retarget as input. This means we can run it on PHC
outputs, ProtoMotions outputs, and GMR outputs — and show improvement on all of them. The
contribution is not tied to one specific retargeter, which makes the paper's claims general.

---

## Proposed Evaluation

Following GMR exactly so results are directly comparable.

### Datasets
- **LAFAN1 subset** (same 21 sequences used by GMR, to allow direct comparison)
- **IHMC in-house MoCap** (shoveling, standup, get-up — motions GMR explicitly excluded,
  which is a differentiation point)

### Robot
- **IHMC Alex** (primary — our contribution, different from GMR's Unitree G1)
- Optionally **Unitree G1** to allow apples-to-apples comparison on GMR's own setup

### Baselines
- PHC (per-frame, gradient descent)
- ProtoMotions (per-frame, Mink differential IK)
- GMR (per-frame, convergence-based Mink) — the current state of the art
- **Ours** (GMR warm start + global trajectory refinement) — proposed method
- Optionally: Unitree closed-source dataset as upper-bound reference (as GMR did)

### Policy training
- BeyondMimic (same as GMR — developed independently of retargeting, fair evaluator)
- Needs to be adapted/ported for IHMC Alex (see Infrastructure section)
- Single-trajectory policies per clip
- Evaluation: sim (100 rollouts, no DR), sim-dr (4096 rollouts, domain randomization),
  sim2sim (MuJoCo/ROS, 100 rollouts, realistic noise + latency)

### Metrics (same as GMR Table I and II)
- **Success rate** (primary): policy reaches end of episode without falling
- **Eg-mpbpe**: global body position error (mm)
- **Empbpe**: root-relative body position error (mm)
- **Empjpe**: joint angle error (×10⁻³ rad)
- **New metric we add**: frame-to-frame joint velocity (rad/frame), p95 and max — directly
  quantifies the smoothness improvement, which the existing papers do not report

### Ablation study (critical for reviewers)
- Stage 1 only (per-frame, no refinement) — shows per-frame baseline
- Stage 2 smoothness only (no collision term in global stage) — isolates smoothness contribution
- Stage 2 collision only (no smoothness term) — isolates collision contribution
- Stage 1 + Stage 2 full (proposed) — combined
- This directly answers: "is the improvement from smoothness, collision, or the combination?"

### Get-up / floor-contact motions (differentiation from GMR)
GMR explicitly excludes motions with whole-body floor contact. We include them — standup,
shoveling, get-up from prone. This is a real contribution because:
- These motions are clinically and industrially relevant for IHMC Alex
- They have the heaviest self-collision (lying-down phase) — where per-frame methods fail worst
- Global optimization should show the largest improvement on exactly these clips

---

## Key Claims (what the paper will argue)

> **Removed claims** (preempted by prior work — see Related Work & Novelty Risk):
> - ~~"We are the first to address the offline/online gap explicitly."~~ IKMR (2025) and the
>   whole-trajectory retargeting line (SPARK, KDMR, STMR) already do. Not defensible.
> - ~~"Global optimization is novel for retargeting."~~ Already the premise of SPARK/KDMR/STMR.

Surviving claims (each must be backed by an experiment, not framing):

1. **A cheap convex kinematic-only refiner is enough**: Most of the policy benefit of full
   kinodynamic trajectory optimization (SPARK, KDMR) can be recovered by a lightweight
   tridiagonal/SCA smoother that needs no dynamics or torque model and runs in milliseconds per
   clip. *Only defensible if the head-to-head experiment supports it.*

2. **Retargeter-agnostic post-process**: The Stage 2 refiner improves the output of any per-frame
   retargeter (PHC, ProtoMotions, GMR) without being a competing retargeter. SPARK is an integrated
   pipeline; a true drop-in wrapper is a different (engineering) contribution.

3. **The improvement is largest for dynamic/complex motions**: Get-up, floor contact, and
   topology-change-heavy sequences — where per-frame methods struggle most.

4. **The whole-body floor-contact gap (strongest remaining wedge)**: We demonstrate good retargeting
   on motions GMR explicitly excludes (crawling, get-up from prone), which the kinodynamic
   retargeting papers also do not center.

---

## What is Already Done (June 2026)

- [x] End-to-end pipeline: FBX/MVNX → canonical skeleton → per-frame QP IK → grounded qpos
- [x] Alex model integration (36-DOF, free root, sites)
- [x] World-delta orientation transfer (semantic frames, auto-facing correction)
- [x] Per-role morphology scaling (non-uniform, computed from rest-pose alignment IK)
- [x] Soft self-collision repulsion in IK QP (w=20, kinematic adjacency filter)
- [x] Shape-aware post-hoc grounding (box, capsule, cylinder exact formulas)
- [x] Contact label generation (11 bodies)
- [x] Pipeline automation script (retargetingPipeline.sh)
- [x] Render pipeline with side-by-side canonical human overlay
- [x] Weight sweep analysis (w=20 optimal for collision)
- [ ] Velocity spike analysis done, in-solver cap has collision interaction problem — unresolved

---

## What Needs to Be Built (roughly in order)

### Technical (retargeting)
1. **Global trajectory refinement (Stage 2)** — the core algorithmic contribution
   - Convex QP over all frames, tridiagonal system per joint for closed-form base solution
   - SCA outer loop for collision constraints
   - Estimated: 3-4 weeks of focused work

2. **Velocity smoothness metric** — frame-to-frame joint velocity statistics to quantify
   the improvement. Already partly done (joint_deltas in NPZ). Need formal reporting.

3. **Baseline retargeters on Alex** — PHC and/or ProtoMotions must be run on Alex to produce
   comparison reference trajectories. GMR is open-source. This requires adapting their
   pipelines to Alex's URDF/XML. Estimated: 2-3 weeks.

4. **LAFAN1 subset processing** — download LAFAN1, run through our pipeline for the same
   21 clips GMR used. Allows direct table comparison.

### Infrastructure (policy training)
5. **BeyondMimic port to Alex** — BeyondMimic is written for Unitree G1 (IsaacSim + ROS).
   Porting to Alex is the single biggest unknown: requires adapting the observation/action
   space, PD controller tuning, sim-to-sim setup with Alex's MuJoCo model.
   **This is the highest-risk item.** If BeyondMimic cannot be adapted to Alex in time,
   fallback is to use Unitree G1 and show the method is robot-agnostic.
   Estimated: 4-8 weeks depending on how much of BeyondMimic is robot-specific.

6. **Policy training compute** — 21 clips × 4 methods × 3 evaluation conditions ≈ 252 training
   runs. Need IsaacSim access and enough GPU hours. Should discuss with mentor early.

7. **sim2sim eval setup** — MuJoCo + ROS node mimicking the BeyondMimic paper's sim2sim
   evaluation for Alex. May already partially exist at IHMC.

### Writing
8. **User study** — optional but strengthens perceptual faithfulness claim. 20 users, 45
   questions (matching GMR's study). Can be done on Amazon Mechanical Turk.

---

## Timeline Options

### REALITY CHECK: 25 days to July 24, 2026

This is extremely tight for the full paper vision (global optimization + policy training).
Three realistic paths are described below, in order of feasibility.

---

### Option A — Workshop paper at Humanoids 2026 (RECOMMENDED if no prior paper)
**Deadline**: August 3, 2026 (workshop proposals — check if a retargeting workshop exists)
Workshop papers are 4–6 pages, can be more preliminary, no full policy training required.
Scope: present the pipeline, self-collision results, get-up motion capability.
This gets work in front of the community and sets up a full paper for 2027.

---

### Option B — Main paper submission (July 24) without policy training
**Feasible if**: mentor approves scope, you write very fast, and you accept a weaker evaluation.

What you have RIGHT NOW that could go into a paper:
- Working end-to-end pipeline for FBX/MVNX → Alex qpos
- Self-collision constraint with quantitative analysis (71% → 24% on standup)
- Weight sweep study (w=20 sweet spot finding)
- Get-up + floor-contact motions (not in GMR)
- Side-by-side render comparisons

What would be missing vs a strong conference paper:
- Global trajectory optimization (the main algorithmic contribution — not built yet)
- Policy training evaluation (the gold standard evaluation — not set up yet)
- Baseline comparison against PHC / ProtoMotions (not done yet)
- LAFAN1 experiments for direct GMR comparison (not done yet)

**Risk**: Without policy training, the paper is weaker than GMR on evaluation. Reviewers will ask why
you don't compare success rates. You'd need a compelling argument for why retargeting quality metrics
alone are sufficient — harder to defend.

**Possible angle for B**: Frame as a "system paper" focused on IHMC Alex specifically — a real
industrial humanoid with different challenges (larger geoms, shoveling/standup motions). This
reduces the comparison burden but narrows the contribution claim.

25-day sprint for Option B:
```
Jun 30 – Jul 5    Write paper outline, get mentor sign-off on scope
Jul 5  – Jul 12   Implement velocity post-processing smoother
                   Run baseline comparison on 3-4 clips (vs no-collision baseline)
                   Produce all figures and render videos
Jul 12 – Jul 19   Full paper draft
Jul 19 – Jul 24   Revision, mentor review, submit
```

---

### Option C — Skip Humanoids 2026, target a 2027 venue (ICRA, RSS, or Humanoids 2027)
**Deadline**: ICRA 2027 ~September 2026, RSS 2027 ~January 2027, Humanoids 2027 ~July 2027

This is the right choice if the global optimization + policy training evaluation is the real
contribution. Rushing a weaker paper now could mean missing the opportunity to publish the
stronger version later (some conferences disallow submissions that overlap with prior workshop papers).

12-month plan for Option C:
```
Jul-Aug 2026      Post-processing velocity smoother; LAFAN1 pipeline setup
Sep-Oct 2026      Global trajectory refinement implementation + initial experiments
Nov-Dec 2026      BeyondMimic port to Alex; start policy training experiments
Jan-Feb 2027      Full policy training sweep (252 runs); quantitative evaluation
Mar-Apr 2027      Baseline comparisons (PHC, GMR); ablation studies
May-Jun 2027      Paper writing
Jul 2027          Humanoids 2027 submission
```

**Critical path for Option C**: BeyondMimic port to Alex. Start scoping this immediately with mentor.

---

### RECOMMENDATION

Talk to your mentor this week with this framing:
- "I have a working pipeline, self-collision results, and a novel insight about offline vs online."
- "For July 24, I could submit a system/method paper without policy training — is that strong enough for Humanoids?"
- "If not, I should skip this cycle and target ICRA 2027 or Humanoids 2027 with the full evaluation."

Mentor's assessment of whether a retargeting methods paper without policy training is publishable
at Humanoids 2026 is the single most important input you need right now.

---

## Related Work & Novelty Risk (literature pass, June 2026)

The closest prior work is **NOT** GMR — it is the recent line of whole-trajectory / kinodynamic
retargeting papers that already occupy most of this idea's conceptual space. These must be cited as
named baselines and the framing must be positioned against them, not around them.

| Paper | Venue/ID | What it does | Overlap with our idea |
|-------|----------|--------------|-----------------------|
| **SPARK** — Skeleton-Parameter Aligned Retargeting with Kinodynamic Trajectory Optimization | arXiv 2603.11480 (Mar 2026) | Two-stage: IK reference → **whole-trajectory** kinodynamic optimization; self-collision avoidance constraints; acceleration regularization for smoothness; offline reference for RL; G1/H1/T1/PM01/Kuavo | **Highest overlap.** Almost exactly our Stage1+Stage2 architecture, plus dynamics/torques. Pre-dates our deadline. |
| **KDMR** — Kinodynamic Motion Retargeting via Multi-Contact Whole-Body Trajectory Optimization | arXiv 2603.09956 (Mar 2026) | Retargeting as whole-body trajectory optimization with contact complementarity + GRF; "dynamically viable reference trajectories that accelerate policy convergence" | Preempts Claim "global opt → better policies." Humanoid, offline. |
| **IKMR** — Implicit Kinodynamic Motion Retargeting | arXiv 2509.15443 (Sep 2025) | Neural whole-**sequence** retargeting, offline pretraining; explicitly states frame-by-frame "lacks scalability"; lower accel/jerk; Unitree G1 | **Kills the "first to make the per-frame-vs-whole-sequence distinction" claim.** |
| **STMR** — Spatio-Temporal Motion Retargeting | arXiv 2404.11557 (Apr 2024) | Finite-horizon OCP over the **whole trajectory** as IL preprocessing; explicit temporal retargeting | Same "global trajectory opt as IL preprocessing" framing (quadruped). |
| **STaR** — Spatial-Temporal Aware Motion Retargeting | arXiv 2504.06504 (Apr 2025) | Predicts entire sequence at once; temporal consistency + penetration constraints | Whole-sequence smoothness (character/mesh domain). |

**Implications:**
- The conceptual "core insight" of the original draft (offline ⇒ global opt, and nobody has done it)
  is **occupied**. Reposition against SPARK/KDMR/IKMR as baselines.
- The honest, defensible angles are narrower: (a) a *cheap convex kinematic-only* wrapper vs SPARK's
  full kinodynamic pipeline — "do you actually need dynamics/torques?"; (b) retargeter-agnostic
  post-processing; (c) floor-contact / get-up motions.
- Defending (a) requires comparing against SPARK **on policy outcomes** → needs the policy-training
  infrastructure (the highest-risk, not-yet-built item). This strengthens the case for **Option C**
  (2027 venue with real evaluation) over a rushed July 2026 submission.
- **Action:** Bring this table to the mentor meeting. The novelty conversation, not just the
  timeline, is the key decision.

---

## Differentiation from GMR (a secondary baseline, not the closest)

| Aspect | GMR | Ours |
|--------|-----|------|
| Optimization scope | Per-frame (sequential) | Global trajectory (all frames jointly) |
| Temporal smoothness | Implicit only (initialization) | Explicit term in objective |
| Velocity spikes | Still present (Dance 5 failure) | Eliminated by global smoothing |
| Self-collision in IK | None (post-hoc detection only) | Soft repulsion in QP (w=20) |
| Floor-contact motions | Explicitly excluded | Primary motion class |
| Offline/online distinction | Not made by GMR | Used, but NOT novel (see SPARK/KDMR/IKMR) |
| Robot | Unitree G1 | IHMC Alex (+ optionally G1) |
| Source format | BVH + SMPL | FBX + MVNX (broader industrial support) |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| BeyondMimic not portable to Alex | Medium | Fallback: use G1 or use a simpler policy trainer |
| Policy improvement marginal (smoothing doesn't help enough) | Low-medium | The GMR paper's own data shows velocity spikes cause failures — our ablation will isolate this |
| Global optimization doesn't scale to 1500 frames | Low | Tridiagonal solve is O(T), SCA outer loop is 3-5 iters — estimated <30s per clip |
| Baselines (PHC/GMR) hard to adapt to Alex | Medium | Even one baseline (PHC) is enough for a compelling comparison |
| Too similar to GMR | Low | The offline/online insight + global opt + floor-contact motions are all novel |
| Conference reject for scope | Low | Humanoids is the right venue — directly about humanoid retargeting |

---

## Open Questions to Discuss with Mentor

1. **BeyondMimic on Alex**: Does IHMC have existing infrastructure for policy training on Alex
   in simulation? BeyondMimic is the obvious evaluation backbone but porting it is the biggest
   unknown.

2. **Unitree G1 as co-evaluation robot**: Would running experiments on G1 (to be directly
   comparable to GMR) plus Alex (to show generalization) be feasible? Requires adapting our
   pipeline to G1 as well.

3. **LAFAN1 licensing**: LAFAN1 is publicly available for research. Confirm this is acceptable
   for the publication context.

4. **Compute allocation**: ~252 policy training runs in IsaacSim. How much GPU time is available?

5. **Co-authorship**: Who from the lab would be appropriate co-authors (mentor, others who
   contribute infrastructure)?

6. **Claim scope**: ~~Is "first to exploit offline context for global trajectory optimization in
   humanoid retargeting" a claim we can defend?~~ **ANSWERED (June 2026 lit pass): No.** SPARK,
   KDMR, IKMR, and STMR already do whole-trajectory offline retargeting. The remaining defensible
   contributions are narrower (cheap kinematic-only wrapper; floor-contact motions) — discuss with
   mentor whether they are sufficient for a Humanoids-level paper, or whether to target a 2027 venue
   with full policy-training evaluation against these baselines.

---

## Notes from the "Retargeting Matters" (GMR) Paper

Key quotes that motivate our work:
- "artifacts introduced during retargeting... are often left in the reference trajectories for
  the RL policy to correct" — we eliminate the temporal artifacts by construction
- "foot penetration, self-intersection, and abrupt velocity spikes are all critical artifacts
  that should be avoided" — we address all three (grounding, collision constraints, global smoothing)
- "we do not include motions with complex interaction with the environment, such as crawling or
  getting up from the floor" — this is our primary motion class, explicitly not addressed by GMR

GMR is a useful per-frame baseline and metric template, but it is **not** the strongest related
work for *our* contribution — the whole-trajectory retargeting papers (SPARK, KDMR, IKMR) are
closer and must be the primary comparison. Beating GMR on its own metrics while adding the get-up
motions it couldn't handle is still worth showing, but it does not by itself establish novelty over
the kinodynamic-retargeting line.

---

*Last updated: June 2026. Next review: after Stage 2 (global refinement) prototype is working.*



New Idea:

What matters in retargeting?
-> All the papers we've read evaluate a retargeting method based on downstream mimic policy performance.
-> But what do we really want in retargeting?
-> Faithfulness by GMR was a really good metric, and generally I also think the same, retargeting means the new motion looks pretty similar to the original motion.
-> But is that what we want in humanoids?
-> Yeah sure, we can get similar motion, and then use those motion as reference frames and again hand tune rewards for robot specific stability, end effector artifacts and other physics that we generally ignore during retargeting?
-> But a good retargeter would take care of that, would it? or a good retargeter always a combination of motion copying to another domain + phyiscs based filtering with some physics engine. Why not copy motion in a way that physics is respected?
-> Will downstream performance increase further? Or are the gains not worth it the time spent on retargeting?

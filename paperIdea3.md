# Paper Idea 3 — "Any-Contact" Retargeting for Humanoids 2027

**Written:** 2026-07-03. **Supersedes:** `paper_idea.md` (June 2026, global-opt framing — dead) and
`paperIdea2.md` (July 2026, 20-day Humanoids-2026 sprint plan — that assessment stands for 2026;
this doc is the 2027 full paper). **Venue:** IEEE-RAS Humanoids 2027 (deadline ~July 2027, ≈12 months).
**Grounded in:** the Undermind deep-research report (2026-07-03, 170 papers, repo root PDF).

---

## 1. What the literature pass settled (Undermind synthesis)

The report's high-level conclusion, verbatim in spirit:

> No single work combines: (i) contact-first, target-side geometric editing for complex floor
> postures, (ii) full-clip temporal smoothing, (iii) explicit adaptation to real robot morphology,
> (iv) hardware-ready trajectories without relying mainly on RL or heavy kinodynamic optimization.
> "This comparative picture supports the view that your proposed direction targets a **genuine gap**."

Four things the report flags as **notably absent** across all 170 papers:

1. **Systematic modeling of floor-contact postures for real humanoids** — no one enumerates or
   precomputes feasible kneeling / all-fours / prone stances under strict joint limits and link
   geometry; no mapping pipeline from human floor postures to robot-specific contact templates.
2. **Full-clip contact sequencing for floor transitions** — no kinematic pipeline that extracts human
   contact sequences (feet, knees, pelvis, hands), edits them into morphologically feasible robot
   contact sequences, and optimizes the whole clip with contact + CoM feasibility + joint-margin safety.
3. **Target-side editing beyond feet** — editing for **knees, pelvis, forearms** and their mutual
   support configurations is essentially ignored (target editing today = foot placement, root
   trajectory, velocity smoothing).
4. **Hardware-focused sequence-level kinematic retargeter for floor motions** — every real-hardware
   floor transition in the literature relies on RL or a strong whole-body controller to "make it
   work." *"There is no evidence of a purely kinematic, sequence-aware, target-side retargeter
   producing long floor-contact clips that can be played with only a standard stabilization stack."*

The report also prescribes the shape of the system that fills the niche (its "Practical
Implications"): offline full-clip optimization; contact geometry as PRIMARY (derive contact
sequence from human data → map to a finite set of robot-feasible contact templates → edit targets
into those templates before solving); target-side geometric editing instead of solver-side soft
weights or RL delegation; explicit tailoring to one real humanoid's morphology; **quasi-static
stability by geometry, not kinodynamics**.

### Who is closest, and why they don't block us

| Work | What it has | Why it doesn't occupy the niche |
|------|-------------|--------------------------------|
| OmniRetarget [7] | Interaction mesh, strongly contact-first at kinematic level, sequence deformation | Loco-manipulation + terrain; **no explicit kneel/prone floor transitions**; feet/hands+object contacts |
| GMR [28] | Clip-level target-side kinematic optimizer, artifact evaluation, BeyondMimic eval | **Explicitly excludes** crawling/get-up-from-floor; upright locomotion + acrobatics |
| Jeong rig unification [21] / CoRe [14] | Explicit target-side trajectory editing, contact-aware refinement | Foot-contact only; no secondary-contact modeling; upright |
| SPARK [94] / KDMR [50] / DynaRetarget [23] | Whole-trajectory optimization, dynamically feasible references | Kinodynamic-heavy (torque/GRF/complementarity); loco-manipulation focus; not floor-postural, not templates |
| IKMR [27] / AdaMorph [24] / diffusion [39] | Sequence-aware neural retargeting | Learning-based, not kinematic-centric; no explicit floor-contact geometry |
| TeleGate [10] / HuMI [16] / HumanPlus [20] / OmniTrack [29] | REAL floor transitions (stand-up, fall recovery, kneeling) on hardware | Transitions are **RL-policy-emergent**, not retargeted; no contact-first kinematic layer |
| Classic multi-contact [5,13,33,36] | Explicit multi-contact optimization, morphing | 2015–2019, manipulation-oriented, no floor-postural focus, optimization-heavy, no modern eval |
| PressMimic [158] (adjacent) | Pressure-guided capture for floor contact ground truth | Perception/control-side, not a retargeter — potential citation + possible contact-label validation source |
| Infant retargeting [3,22] | Non-upright postures (lying, rolling) | Small robots, developmental analysis, not hardware-ready retargeting |

**Reviewer-proofing note:** the "global optimization is novel" claim from `paper_idea.md` stays dead
(SPARK/KDMR/IKMR/STMR). The 2027 claim is NOT "offline global opt" — it is the **contact-first
floor-posture system** with the four missing pieces above, evaluated the gold-standard way.

---

## 2. The idea (one paragraph)

**Any-Contact Retargeting: a purely kinematic, contact-first retargeter that treats EVERY support
link — feet, knees, shins, pelvis, fists — as a first-class contact, maps detected human floor
contacts onto a precomputed library of robot-feasible contact templates (kneel, half-kneel,
all-fours, prone, sit), edits the kinematic targets into those templates (generalizing our
shank-tilt clamp into a per-chain family of target-side feasibility edits), and solves the whole
clip with contact-anchored smoothing plus a quasi-static CoM-in-support-region constraint —
producing floor-contact reference trajectories for a real industrial humanoid (IHMC Alex) that
(a) train markedly better imitation policies than faithfulness-first retargets and (b) play back
on hardware through the standard IHMC whole-body stack with no motion-specific RL.**

One sentence: *general retargeters do feet; we do floors.*

### Why "impenetrable"

Every standard reviewer attack has a pre-built answer:

- **"Not novel — X does whole-sequence retargeting."** X does upright/loco-manipulation with
  feet(+hands). Nobody does knees/pelvis/elbows as first-class retargeting contacts, nobody has
  contact templates, nobody hardware-plays floor clips kinematically. Undermind table, 170 papers.
- **"Why not just let RL fix it?"** That is the field's current answer (TeleGate, HuMI, HumanPlus) —
  and GMR's own result shows reference quality drives policy success. Our ablation directly measures
  policy-success delta from contact-first references vs faithfulness-first ones ON floor motions.
- **"Where's the gold-standard evaluation?"** Policy training (BeyondMimic-style) success rates +
  the GMR metric suite, on the motion class GMR excluded. Plus hardware playback — stronger evidence
  than any of the kinematic-only competitors present.
- **"Is it just your 2026 system paper + more clips?"** No: generalized contact set (knees/pelvis),
  template library + template mapping, and quasi-static stability constraint are new machinery, each
  ablatable. The 2026-era pipeline is the warm start, not the contribution.
- **"Kinematic-only can't be dynamically feasible."** Stated up front; the quasi-static margin is
  the deliberate middle ground (report's own recommendation), and the policy-training + hardware
  experiments empirically measure whether kinematic + quasi-static is *enough* — that measurement is
  itself a contribution ("do you need kinodynamics for floor motions?" — the honest heir of the old
  Angle B, now answerable against SPARK/KDMR as cited positions).

---

## 3. Technical contributions (each new, each ablatable)

**C1 — Generalized contact set.** Extend contact detection + contact-first override from
{soles, fists} to {knees/shins, pelvis/seat, elbows/forearms}. Human-side detection generalizes our
height+speed+orientation gates per landmark group; robot-side each contact gets a support face and
a position anchor (the fist-pin pattern) on the corresponding Alex link. Directly fills absent-item 3.
*Have already:* the full per-effector machinery for 2 effector types; kneeling/fall clips where knee
contact currently goes unmodeled.

**C2 — Robot contact-template library + template mapping.** Precompute (offline, once per robot)
a small library of feasible floor-support stances for Alex under true joint limits and collision
geometry: double-kneel, half-kneel L/R, all-fours, prone-on-elbows, seated, squat. Each template =
a set of active contact faces + feasible pose manifold (joint boxes + CoM support region). At
retarget time, classify each human floor-contact interval to a template and edit the targets into
the template's feasible manifold — the shank clamp generalized from "one ankle cone" to "a family of
per-chain geometric clamps." Fills absent-items 1 and 2. This is the paper's centerpiece and the
part hardest to reproduce quickly.

**C3 — Quasi-static stability in the full-clip QP.** Add to Stage B a soft CoM constraint: ground-
projected CoM inside (a conservatively shrunk) support region of the active template's contact
faces, active only during labeled static support phases (not ballistic fall segments). Linear in
δQ per frame (CoM Jacobian), so the QP stays sparse-convex — no torques, no GRF, no complementarity.
Exactly the report's "geometry and conservative quasi-static stability constraints" recipe.

**C4 — Hardware-ready output, demonstrated.** The IHMC `KinematicsToolboxOutputStatus` export
(already built, 120 Hz) played through IHMC's standard whole-body/stabilization stack on real Alex
(or, fallback, the IHMC sim stack) for a subset of clips: kneel-down, stand-from-kneel, sit-to-stand.
This claims absent-item 4 — the single claim NO paper in the survey can make.

Supporting (kept from current pipeline, presented as system, not novelty): semantic frames +
world-delta transfer, rest-relative morphology scaling, θ·unit-axis alignment, onset hysteresis +
make/break blending, tridiagonal Stage A, median-anchored soft Stage B with soft-slack collision,
mesh-exact grounding.

---

## 4. Evaluation plan (gold standard, GMR-comparable)

**Motion set.** The 18 in-house floor-contact clips (shovels, get-ups, kneeling falls) + new
captures to round out template coverage (crawl, prone-to-stand, sit-to-floor-to-stand). Stretch:
the LAFAN1 get-up subset for external comparability. Mentor question: can any in-house clips be
released as a small **floor-contact retargeting benchmark**? A released benchmark + eval code makes
the paper a reference point, not just a method (big impenetrability multiplier).

**Baselines (all on the same clips, same robot).**
- GMR (open source) — expected to degrade/exclude on floor contact; its failure mode IS the motivation figure.
- OmniRetarget (open source) — the strongest contact-aware kinematic baseline.
- Our own "faithfulness-first" ablation (contact machinery off) — the controlled in-solver baseline.
- SPARK/KDMR: cite and position; compare only if code exists (do not put on critical path).

**Metrics.**
1. Kinematic (ours + GMR's): penetration %, peak penetration, foot/limb-flat error, plant slip
   distribution (FULL distribution — the 8–9 cm tail reported, per `paperIdea2.md` discipline),
   joint-limit margin, velocity spikes / Δq p95.
2. NEW contact-sequence metrics (natural with C1/C2): contact-set F1 vs human labels, template
   classification accuracy, quasi-static margin violation %.
3. **Policy success** (primary): BeyondMimic-style single-clip tracking policies per {clip × method},
   success rate + tracking errors, sim / sim-DR / sim2sim — the GMR protocol on the motion class GMR excluded.
4. **Hardware**: N clips played through the IHMC stack — completion, tracking error, operator-rated
   playability. Even 3 clips on real Alex beats every competitor's evidence for this class.

**Ablations.** C1 off (feet+fists only — the 2026 system), C2 off (detection without templates),
C3 off (no CoM term), Stage B off. Era dirs already provide the historical ones (shank clamp, hysteresis).

---

## 5. 12-month plan (deadline ~2026-07 → 2027-07)

Critical path = policy training infrastructure (BeyondMimic-on-Alex), flagged highest-risk since
`paper_idea.md`. Start it FIRST, in parallel with method work — not after.

| When | Method track | Eval track |
|------|--------------|-----------|
| Jul–Aug 2026 | C1: generalized contact detection + knee/pelvis/elbow anchors; verify on kneeling-fall clips | **Scope BeyondMimic-on-Alex with mentor; secure GPU + hardware time; run GMR/OmniRetarget out-of-box on 2–3 clips** (motivation figures) |
| Sep–Oct 2026 | C2: template library for Alex + template classification/mapping | BeyondMimic port milestone: one policy trained on one existing clip (go/no-go gate; fallback below) |
| Nov–Dec 2026 | C3: quasi-static CoM term in Stage B; full pipeline on all clips + new captures | Policy sweep infra; kinematic metrics harness over all methods |
| Jan–Feb 2027 | Freeze method; ablation runs | Full policy sweep: clips × {ours, GMR, OmniRetarget, faithfulness-first} × ablations |
| Mar–Apr 2027 | — | **Hardware playback** on real Alex (IHMC stack); sim2sim numbers |
| May–Jun 2027 | Paper writing, figures, video | Mentor/lab review passes |
| Jul 2027 | Submit | — |

**Fallbacks.** BeyondMimic un-portable by Oct gate → (a) any available IHMC in-house RL tracking
stack, or (b) port pipeline to Unitree G1 and run BeyondMimic as-is (method is robot-parametric via
templates — G1 second robot also strengthens generality). Hardware slot unavailable → IHMC sim stack
playback still claims "standard stabilization stack, no RL" (weaker but defensible). Policy gains
marginal → the paper survives on hardware + kinematic + contact-sequence results; report the null
honestly (it would itself answer "does reference quality matter for floor motions?").

---

## 6. What carries over from the two superseded docs

- **From `paper_idea.md`:** the SPARK/KDMR/IKMR/STMR novelty table and the June-2026 concession
  (global-opt framing dead) — now reinforced by Undermind; the BeyondMimic-critical-path warning;
  the metric suite. Its "New Idea" closing question (faithfulness vs physics-respecting copying) is
  answered by this paper's design: contact/support correctness IS the faithfulness that matters for
  floor motions, and the policy experiment measures it.
- **From `paperIdea2.md`:** the overclaim discipline (report full slip distribution, residual
  collision %, no comparative language without the baseline run) — adopt verbatim; the honest-
  abstract style; Angle C (shank clamp = target-side editing) which now scales up into C2; the era-
  dir ablation inventory. If the 2026 short paper was submitted, this is its full-paper successor
  with disjoint claims (templates, generalized contacts, stability, policy+hardware eval).
- **Dropped for good:** "first to exploit offline context"; global-opt-as-novelty; user study
  (optional at best); 252-run sweep as a requirement (scope to clips that exist).

## 7. Open items for the mentor meeting

1. BeyondMimic-on-Alex vs G1-fallback — which do we resource? (Gate: one trained policy by Oct 2026.)
2. Hardware time on Alex for playback experiments (Mar–Apr 2027 window).
3. Can a subset of the in-house floor-contact clips be released as a public benchmark?
4. New captures needed for template coverage (crawl, prone-to-stand) — capture session scheduling.
5. Was the Humanoids 2026 short paper submitted? (Determines what this paper may not reuse.)

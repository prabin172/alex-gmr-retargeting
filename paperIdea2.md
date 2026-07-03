# Paper Idea 2 — Humanoids 2026 Viability Assessment & 20-Day Plan

**Written:** 2026-07-02. **Deadline:** ~2026-07-20 (≈18–20 days). **Venue:** IEEE-RAS Humanoids 2026.
**Supersedes / builds on:** `paper_idea.md` (the June-2026 "global trajectory optimization" draft).

This document is an honest go/no-go assessment plus a concrete work plan. Bottom line up front:
**there is a viable *short/system* paper here, but only if we drop the global-optimization novelty
claim (occupied by prior art) and reframe around what the repo actually and uniquely does well:
contact-first retargeting of *whole-body floor-contact motions* (get-ups, kneeling falls, shovel
plants) onto a real industrial humanoid (IHMC Alex), a motion class the dominant retargeter (GMR)
explicitly excludes.** A full-conference-strength paper is *not* achievable in 20 days without a
downstream policy or baseline result, both of which are not built. Plan below scopes to the strongest
defensible submission.

---

## 1. What the repo actually is (technical contribution, grounded in code)

A kinematic human-MoCap → humanoid retargeting pipeline, specialised for **contact-heavy motion**.
Verified against `METHOD.md`, the four active solver scripts, and `outputs/`:

- **Input class:** FBX in-house IHMC MoCap. 18 clips currently processed end-to-end:
  5 shovels, ~9 standup/get-up variants (side, natural, from-kneeling, squat-crouch, slide-hands-back),
  2 kneeling falls, kneeling. All of these involve hands and/or knees on the floor.
- **Method (5 stages, all kinematic):**
  1. Canonical-human intermediate (Blender, positions only).
  2. Semantic orientation frames rebuilt from landmark geometry + auto facing-yaw + **world-delta**
     orientation transfer (copy only the change-since-rest, never absolute orientation).
  3. **Contact-first per-frame QP IK** (MuJoCo, damped Gauss–Newton). Distinctive pieces:
     rest-relative per-role morphology scaling; contact detection from *human* data with onset
     hysteresis + make/break cosine cross-fade; well-conditioned `θ·unit-axis` foot-flat error
     (avoids the `sin²θ` 180° trap); **target-side shank-tilt clamp** that edits the *knee target*
     into the flat-foot-reachable cone so Alex's stiff/asymmetric ankle stops fighting the leg track;
     foot-hold freeze-at-touchdown (×10 weight); foot-yaw lock; fist support-site position pin.
  4. **Global trajectory opt:** closed-form tridiagonal Stage-A smoothing (incl. floating base) +
     contact-aware sparse QP Stage-B with median-anchored plants and **soft-slack self-collision**
     that stays feasible on the dense convex-hull ("fullmesh") legs.
  5. Mesh-exact vertical grounding (plants whichever geom — seat/knee/foot — is lowest).
- **Output:** `qpos (T,36)` NPZ + Mimic-oriented contact info + **IHMC `KinematicsToolboxOutputStatus`
  JSON export** (`export_alex_retarget_npz_to_ihmc_json.py`) — i.e. the motions are in the message
  format the real IHMC Alex whole-body/kinematics toolbox consumes.

**One-line contribution:** *a single, un-tuned, contact-first kinematic retargeter that produces
self-collision-aware, contact-faithful, controller-ready reference motions for whole-body
floor-contact behaviours on a real 36-DOF humanoid — the motion class general-purpose retargeters
sidestep.*

### Novelty vs prior art (what I know)
- **GMR (General Motion Retargeting)** — per-frame Mink IK, faithfulness metric + BeyondMimic eval,
  Unitree G1. **Explicitly excludes** "crawling / getting up from the floor." This is the paper we
  are directly complementary to (and our best framing anchor).
- **PHC / PerpetualHumanoidControl, ProtoMotions** — per-frame IK / physics-based imitation; general
  locomotion, not floor-contact-centric.
- **H2O / OmniH2O, HumanPlus, ExBody, TWIST** — retargeting + RL for teleop/shadowing; largely
  standing/whole-body but not centred on prone-to-stand floor contact, and coupled to their own RL.
- **Whole-trajectory / kinodynamic retargeting: SPARK, KDMR, IKMR, STMR, STaR** (2024–2026). These
  **occupy the "offline ⇒ global optimisation" framing** that `paper_idea.md` originally pitched as
  novel. SPARK is almost exactly our Stage-1→Stage-2 architecture *plus dynamics*. **Global-opt
  novelty is dead** (this was already conceded in `paper_idea.md`'s June lit-pass note; we keep that
  conclusion).

**Honest surviving wedge:** not the optimiser, but (a) the **contact-first target-editing machinery
for a stiff-ankle real robot** (shank clamp, θ·axis flat, fist support pin) and (b) the **floor-contact
motion class on a real industrial humanoid with controller-ready export**. That is a *systems +
focused-method* contribution, not a new optimisation theory.

---

## 2. What experimental evidence actually exists (verified in `outputs/` + logs)

Present:
- **18 clips** fully through stages 3→5 (`contactfirst/`, `global_opt_contactfirst/`,
  `grounded_contactfirst/`), 18 IHMC JSON exports (`ihmcJsons/`, `ihmcJsons-120hz/`), and renders
  (`renders/contactfirst/fullURDF/`, `unified/`, plus era-comparison dirs).
- **Quantitative metrics per clip** (from `fullurdf_pass_20260702_135608.log`, `compute_globalopt_metrics.py`):
  velocity spikes, per-frame max/p95 Δq, collision-frame %, peak penetration, track error,
  plant slip, foot-flat error. Clean headline result: **per-frame IK 14–31 velocity spikes → 0
  after smoothing on every clip**; collision frames reduced by Stage B; straight-knee lock 26.5%→0%;
  contact foot-flat error 12.7°→7.7° mean.
- A foot-slip diagnostic plot (`outputs/analysis/shovel_fronthard_02_footslip.png`).
- Ablation-adjacent artifacts: era directories (`pre_shankclamp`, `shankclamp`, `onset_hyst`,
  `foothold_fix`, `ab_primitive` vs `ab_fullmesh`) — raw material for before/after comparisons.

Absent (this is the crux):
- **No downstream policy / RL training or success-rate evaluation.** The field's gold-standard metric.
- **No baseline retargeter run** (GMR/PHC/ProtoMotions not executed on Alex; SPARK/KDMR not compared).
- **No hardware run.** IHMC JSON export exists, but there is no evidence of on-robot playback.
- **No public benchmark** (LAFAN1 not processed) → no direct external comparability.
- **No user study** (GMR has one).
- **Single robot** (Alex), **single MoCap source** (in-house FBX), **18 clips**.

---

## 3. Overclaiming risks (specific, verified)

1. **"Self-collision-free" is not universal on the shipped fullmesh model.** METHOD §9 and the README
   imply penetration is eliminated. The fullmesh log shows Stage-B **residual collision frames of
   11.0%, 19.7%, 20.2%, 20.6%, 32.5%** on several get-up/kneel clips (peak penetration up to ~2.3 cm).
   True claim: penetration is *reduced and driven to 0 on many clips*, not eliminated everywhere.
2. **Plant-slip numbers are cherry-picked to shovels.** Docs headline "1.0–1.5 cm." The log shows
   get-up/kneel Stage-B slip of **2.6, 2.9, 3.7, 3.8, 4.2, 8.0, 8.6, 9.3 cm**, and — importantly —
   **Stage B sometimes *increases* slip vs Stage A** (e.g. 0.2 cm warm → 2.3 cm Stage A → 9.3 cm
   Stage B). The "Stage B reduces plant drift" narrative does not hold on the hardest clips. Report
   the full distribution, not the shovel best-case.
3. **Kinematic only — no dynamic feasibility.** Correctly disclosed in METHOD §9; must stay explicit
   in the paper (no torque/GRF/ZMP feasibility, no guarantee the trajectory is trackable).
4. **No baseline ⇒ no evidence the method is *better*, only that it *runs*.** Any comparative
   language ("smoother than", "outperforms") is unsupported until a baseline is run.
5. **Contact detection is heuristic** (height/speed thresholds on human markers), not validated
   against ground-truth contact.
6. **"One config for all actions" is a design stance, not a validated generalisation claim** — it is
   18 in-house clips of 3 broad motion families, not a diverse benchmark.

---

## 4. Publication angles, ranked

### Angle A (RECOMMENDED) — System/method paper: contact-first retargeting for whole-body floor-contact motions on a real humanoid
- **Framing:** "General retargeters exclude floor-contact motion (GMR says so explicitly). We present
  a contact-first kinematic retargeter that handles get-ups, kneeling falls, and shovel plants on the
  IHMC Alex, producing self-collision-aware, contact-faithful, controller-ready references — under one
  un-tuned config." Position *complementary* to GMR, not competing on its metrics.
- **Humanoids fit:** High. Humanoids values real-platform system contributions and hard, embodied
  motion classes. Floor contact + a real industrial biped is squarely in scope.
- **Evidence strength:** Medium-high. 18 clips in the target class, full kinematic metrics, spike
  elimination, controller export, renders, before/after era ablations already on disk.
- **20-day work:** Moderate. Mostly (i) an honest metrics table over all 18 clips, (ii) 1–2 targeted
  ablations from existing era dirs (shank clamp on/off; Stage-B on/off; per-frame vs smoothed spikes),
  (iii) ideally one *lightweight* baseline (see plan) showing a naïve/off-the-shelf retargeter fails
  or self-penetrates on floor-contact clips. No policy training required.
- **Honest caliber:** **Full short paper / borderline full paper.** Becomes a solid full paper *iff*
  we add even one baseline comparison on the floor-contact clips. Without any baseline it is a
  defensible 6-page system paper but reviewers will push on "vs what?".
- **Verdict:** Best risk-adjusted fit. Defensible, in-scope, achievable.

### Angle B — "Do you actually need dynamics?" cheap kinematic refiner vs kinodynamic retargeting (SPARK/KDMR)
- **Framing:** carried over from `paper_idea.md`: a convex kinematic-only refiner recovers most of the
  benefit of full kinodynamic trajectory optimisation.
- **Fit:** High conceptually.
- **Evidence:** **Weak/absent.** The claim is only defensible via *policy outcomes* against SPARK/KDMR,
  which requires BeyondMimic-on-Alex + baseline reimplementations — none built, not buildable in 20 days.
- **Verdict:** **Not feasible for this deadline.** Defer to a 2027 venue (ICRA/RSS/Humanoids 2027) with
  real policy evaluation. Keep as the long-game paper.

### Angle C — Focused method note: target-side feasibility editing for kinematically-mismatched ankles
- **Framing:** the shank-tilt clamp + `θ·unit-axis` foot-flat as a general recipe for "fix the target,
  don't fight it with weights" when the robot's joint range can't reproduce a human contact.
- **Fit:** Medium (narrow but concrete and genuinely novel-feeling).
- **Evidence:** Ablation-able from existing `pre_shankclamp` vs `shankclamp` dirs; straight-lock 26.5%→0%,
  foot-flat 12.7°→7.7° already measured.
- **Verdict:** **Workshop / short-paper caliber on its own.** Best *folded into Angle A as its core
  technical novelty* rather than submitted standalone.

**Ranking: A > C(as part of A) > B(defer).**

---

## 5. Recommendation + honest abstract draft

**Submit Angle A**, with Angle C's shank-clamp/flat-error machinery as the technical core, as a
**Humanoids 2026 short/system paper**. Explicitly complementary to GMR; no policy or hardware claims.

> **Abstract (honest draft).** General-purpose human-to-humanoid motion retargeters deliberately
> exclude whole-body floor-contact motion — crawling, getting up from the ground, kneeling — because
> ground contact and severe joint-range mismatch break per-frame, faithfulness-first pipelines. We
> present a contact-first kinematic retargeter that targets exactly this class on the IHMC Alex, a
> 36-DOF industrial humanoid. Contact is detected from the human capture and used to *override* the
> captured limb orientation with the physical support surface under a smooth make/break cross-fade;
> a well-conditioned axis-angle foot-flat term and a target-side "shank-tilt clamp" reconcile flat
> ground support with Alex's stiff, asymmetric ankle by editing the kinematic target rather than
> fighting it with weights. A closed-form temporal smoother followed by a contact-aware sparse QP
> with soft-slack self-collision removes the per-frame velocity spikes (14–31 → 0 across all clips)
> and reduces self-penetration on the dense collision body, and a mesh-exact grounding step plants
> whichever contact is lowest each frame. A single un-tuned configuration handles 18 clips spanning
> shovel plants, sit/kneel-to-stand, and falls; we report contact fidelity, foot-flatness,
> self-penetration, and smoothness, and export controller-ready trajectories in the robot's native
> whole-body-control message format. The method is kinematic — dynamic feasibility is left to
> downstream control/RL — and we characterise the residual plant slip (1–9 cm, largest on dynamic
> get-ups) as the explicit trade for zero-to-low self-penetration.

Note the abstract states the slip range and the kinematic limitation up front — that pre-empts the
two biggest overclaim risks.

---

## 6. Concrete 20-day work plan (Angle A)

Assumes ~18 days of writing/experiment runway. No policy training. Goal: 6-page (or full-length if a
baseline lands) honest system paper + supplementary video.

### Week 0 (Days 1–2) — decision + scaffolding
- **Mentor sign-off on Angle A scope** (system paper, no policy eval, complementary-to-GMR framing).
  This is the single most important gate — same advice as `paper_idea.md`, unchanged.
- Set up the paper repo (IEEE Humanoids template), section skeleton, figure list.
- Freeze the claim set to the honest ones in §3–§5 here.

### Week 1 (Days 3–8) — lock the numbers, honestly
- **Full metrics table over all 18 clips** via `compute_globalopt_metrics.py` (point it at
  `global_opt_contactfirst/` + fullmesh model): spikes (per-frame→Stage A→Stage B), Δq p95/max,
  collision %, peak penetration, plant slip, foot-flat, track error. **Report the full distribution
  incl. the 8–9 cm slip outliers** — do not headline shovels only.
- **Ablation 1 (spikes):** per-frame IK vs Stage-A smoothing — already in the log; formalise as a
  figure (velocity-spike count + a Δq-over-time trace on one get-up).
- **Ablation 2 (shank clamp):** `contactfirst_pre_shankclamp/` vs `contactfirst_shankclamp/` —
  straight-knee lock %, foot-flat error, over-limit ankle frames. This is Angle C's core evidence.
- **Ablation 3 (Stage-B / soft-collision):** `ab_primitive` vs `ab_fullmesh`, hard-vs-soft — show
  the soft-slack keeps the QP feasible (report the "hard QP no-ops, |δQ|=0" failure explicitly).
- Regenerate 3–4 render clips (get-up, kneeling fall, shovel) for the video; include the human
  side-by-side and the contact strip.

### Week 2 (Days 9–14) — the differentiator experiment (baseline)
- **Run ONE baseline retargeter on the floor-contact clips.** Cheapest defensible option:
  a naïve per-frame position-only IK (no contact override, no shank clamp) *in our own solver*
  (flags already exist) — call it "faithfulness-first per-frame IK." Show it (i) self-penetrates,
  (ii) leaves feet non-flat / floating, (iii) picks over-limit ankles, on the exact get-up clips.
  This is the "vs what?" answer without needing to port GMR.
  - Stretch (only if Week 1 finishes early): run **GMR open-source** on the same skeleton for one or
    two clips — expected to fail/degrade on floor contact, which is itself the result. Do **not** put
    this on the critical path; the in-solver ablation baseline is sufficient.
- Build the comparison figure/table: ours vs faithfulness-first on penetration, foot-flat, ankle-limit
  violations, spikes.

### Week 3 (Days 15–18) — write + submit
- Full draft: Intro (GMR excludes floor contact ⇒ gap), Related Work (GMR + honest SPARK/KDMR/IKMR
  positioning — we are kinematic + floor-contact-focused, they are dynamic + general), Method
  (stages 3–4 are the meat; lead with contact-first + shank clamp + θ·axis + soft-collision),
  Experiments (§Week1–2 tables/figures), Limitations (kinematic-only, slip range, residual collision
  on hard clips, single robot, no policy eval — state all of §3).
- Supplementary video (Day 16–17).
- Mentor review pass (Day 17), revise, **submit Day 18** with buffer.

### Cut-if-behind priority
1. Keep: metrics table + spike ablation + shank-clamp ablation + video. (This alone = short paper.)
2. Add if time: in-solver faithfulness-first baseline comparison. (Upgrades toward full paper.)
3. Drop first: GMR/G1 external baseline, LAFAN1, any policy language.

---

## 7. What to keep vs drop from `paper_idea.md`

- **Drop:** the global-optimisation-as-novelty framing (occupied by SPARK/KDMR/IKMR/STMR — their own
  lit pass already conceded this); the "first to exploit offline context" claim; the policy-training
  evaluation plan (BeyondMimic-on-Alex) for *this* deadline; the 252-run sweep; the user study.
- **Keep:** the honest Related-Work table (SPARK/KDMR/IKMR/STMR/STaR) — reuse verbatim; the
  floor-contact-motion wedge (their "strongest remaining wedge" — we make it *the* wedge); the GMR
  quotes ("we do not include ... getting up from the floor"); the Option-C long game (Angle B here) as
  the 2027 follow-up with real policy eval.
- **Change:** move from "global opt improves policies" (unprovable in 20 days) to "contact-first
  kinematics makes a hard motion class feasible on a real robot" (provable now with what's on disk).

---

## 8. Go / no-go

- **Go** for a Humanoids 2026 **short/system paper on Angle A**, contingent on mentor approving a
  no-policy-eval scope. Everything needed for the minimum viable version already exists in `outputs/`;
  the 20 days buy an honest metrics table, 2–3 ablations, one in-solver baseline, and the write-up.
- **No-go** for the original global-opt / policy-comparison paper on this deadline — defer to 2027
  (Angle B) with BeyondMimic-on-Alex.
- **Overclaim discipline is the main risk to acceptance**, not the amount of work: the slip range and
  residual-collision numbers must be reported in full, and all comparative language must be backed by
  the baseline experiment.

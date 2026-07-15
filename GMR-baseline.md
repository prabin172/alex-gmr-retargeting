# GMR-baseline — de-Alexing the paper (2026-07-15)

**Why this doc:** Prabin's worry, verbatim in spirit: *"this IHMC-specific work is very one-robot
centric and probably not generalizable."* Seed: `notes.md`'s three-way scope split (feasible /
infeasible-needs-external-strategy / polish). Question: with GMR as the baseline (and possibly the
platform), what paper — part or full — can we actually write?

**Relation to existing plans:** does NOT kill `paperIdea3.md` (Any-Contact 2027). Options A/B below
can be its eval backbone, a standalone precursor, or both. Old paper tracks:
`git show 7aeaed6^:wiki/questions/publication.md`. Lit landscape: Undermind PDF (repo root,
170 papers, 2026-07-03).

---

## 1. GMR in one paragraph, and why it's the right baseline

GMR (General Motion Retargeting, `araujo2025gmr` in `paper_intro.md`'s cite keys) is the open-source
multi-robot kinematic retargeter (Unitree G1 and others, MuJoCo models shipped) whose paper
established the field's key empirical point: **reference-motion quality directly limits downstream
policy success** (evaluated with BeyondMimic). And it **explicitly excludes our motion class**:
*"we do not include motions with complex interaction with the environment, such as crawling or
getting up from the floor."* So GMR is simultaneously (a) the standard baseline reviewers expect,
(b) the motivation figure (its failure on floor clips), and (c) a multi-robot vehicle that makes any
module we port to it generalizable by construction.

**Repo status:** the vendored `general_motion_retargeting/` package here is source-stripped (only
`__pycache__` remains). Baseline runs need a fresh clone of upstream GMR.

**The strategic unlock:** paperIdea3's #1 risk was BeyondMimic-on-Alex (policy-eval infrastructure,
go/no-go gate Oct 2026, G1 listed as the *fallback*). Flip it: make **G1 the primary eval platform**.
BeyondMimic runs on G1 out of the box, GMR retargets to G1 natively, and our modules prove
generality by running on a second robot. Alex stays as the "industrial biped + IHMC stack" hardware
story, not the load-bearing eval platform.

---

## 2. What we own that generalizes (assets, with evidence)

| Asset | What it is | Robot-specific parts | Evidence |
|---|---|---|---|
| **Polish module** (Stage A smoothing + Stage B contact QP + 4.5 grounding) | Post-processor for ANY motion source — ours, manual edits, other retargeters | MuJoCo model, body-role map, contact flags | Luigi's manual Blender retarget: spikes 25.7→4.5 rad/s (5.7×), self-collision→0, floor pen 3.0→2.8 cm, slip cost 3.3→4.4 cm (`scripts/ihmc_json_to_stage4_npz.py`, wiki/log.md 2026-07-14) |
| **Reference-free eval suite** (`scripts/eval_ihmc_json.py`) | Physics-plausibility metrics with no ground truth needed: mesh-exact floor pen, self-collision, joint-limit margin, rate-aware velocity spikes, stance/slip from contact flags | JSON joint order + Alex model paths (thin layer) | Built + validated on Luigi's JSONs, 2026-07-14 |
| **Contact-first formulation** (stages 2.5–4) | Canonical-human grounding + persisted contact labels; contact-anchored whole-clip convex QP | Role maps, support faces, Alex XML | phasic-v2 corpus: planted pen median 2.38→1.60 cm, selfpen 0.77→0.39 cm |
| **The feasibility taxonomy** (notes.md) | Feasible / infeasible-needs-external-strategy / polish — each piece evidenced | None — it's a finding | hierarchical-v1 (zero benefit), continuation-v1 (1/3 clips), feasibility-first-v1 (diverges); Luigi's manual edit supplying what no QP could |
| **Negative-results corpus** | "Constraint machinery of any sophistication only helps when the solver already sits in a feasible basin" | None | planLog.md, wiki/experiments/ gates |

One-sentence scope statement (from notes.md, now the paper's honest thesis candidate): **faithful
when feasible, polished when given a feasible strategy, honest about which clips need one.**

---

## 3. Three paper shapes

### Option A — "Contact-aware kinematic polish" (module paper) ★ recommended lead

**Claim:** a robot-agnostic, purely-kinematic, convex post-processing module (temporal smoothing +
contact-anchored QP + grounding) measurably cleans retargeting artifacts from ANY source — GMR
output, manual keyframe edits, our own IK — and the cleanup transfers to downstream policy success.

**GMR's role:** both baseline AND input. Polish GMR's own G1 outputs on its home turf (locomotion
clips) → show artifact reduction even where GMR is strong; then on floor-contact clips → show the
gap. GMR's paper already argues artifacts hurt policies; we supply the fix and measure the delta
with BeyondMimic-on-G1 (native, no port).

**Work needed:** port polish stage to G1 (G1 MuJoCo XML ships with GMR; need body-role map + floor
geom + contact detection for sources without contact flags — our human-side height/velocity gates
adapt); generalize `eval_ihmc_json.py` past Alex joint order (thin layer). No new solver machinery.

**Size/venue:** RA-L (rolling, 6–8 pp) or ICRA 2027 (~Sep 2026 deadline). "Part of a paper" answer:
this IS paperIdea3's eval-infrastructure track, publishable standalone.

**Risks:** polish deltas on GMR's clean locomotion output may be small (then floor clips carry the
paper — that's fine, it's our class); slip-vs-smoothness tradeoff must be reported as a full
distribution (overclaim discipline).

### Option B — "Feasibility taxonomy + reference-free benchmark" (analysis paper)

**Claim:** whole-body floor-contact clips split measurably into feasible / infeasible-without-
external-strategy / polishable; we release the reference-free metric suite + protocol and
characterize where GMR and contact-first solvers fail and why (the basin argument, evidenced by the
negative-results corpus).

**GMR's role:** the characterized baseline — quantify its excluded class rather than take the
exclusion on faith.

**Work needed:** mostly harness + writing; needs releasable clips (open mentor question,
paperIdea3 §7.3).

**Size/venue:** workshop → short paper. Composes with A (shared metric suite); weak standalone
(negative results + taxonomy without a fix reads thin) — best as A's Section 5 or a workshop teaser.

### Option C — Any-Contact 2027, re-platformed (full paper)

`paperIdea3.md` as written, with one amendment: **G1 primary eval platform** (BeyondMimic native,
GMR baseline native), Alex as second robot + IHMC-hardware story. C2 templates become genuinely
robot-parametric (built per-robot from XML limits + collision geometry) — which was always the
claim; now it's demonstrated, not asserted. Heaviest; unchanged 12-month scale.

### Recommendation

**A now, B folded into A, C stays the 2027 vehicle with the G1 amendment.** A is the only piece
validated end-to-end this month (the Luigi polish result), is generalizable by construction, reuses
GMR's own eval protocol on GMR's own robot, and every hour spent on it (G1 port, eval
generalization, BeyondMimic runs) is directly reusable by C. Nothing is thrown away.

---

## 4. De-risk experiments (ordered, each cheap, each a kill-test)

- **E1 — GMR out-of-box on floor clips.** Fresh upstream clone; run GMR→G1 on 2–3 of our
  floor-contact motions (check ingest path: GMR takes SMPL-X/BVH sources — may need our FBX via
  Blender→BVH, stages 1–2 analog). Expected: visible failure (their own exclusion). Deliverable:
  the motivation figure. ~days.
- **E2 — Eval suite de-Alexed.** Refactor `eval_ihmc_json.py` into `eval_motion.py(model_xml, qpos,
  contact_flags?)`; run on GMR's G1 outputs. Kill-test: if metrics aren't meaningful cross-robot,
  A's protocol is weaker than hoped. ~days.
- **E3 — Polish on G1.** Port Stage A + 4.5 (no Stage B yet — smoothing+grounding alone was the
  recommended Luigi deliverable) to G1; polish one GMR locomotion clip + one floor clip; E2 metrics
  before/after. Kill-test for A's core claim. ~1 week.
- **E4 — Stage B contact QP on G1.** Needs role map + support faces for G1 feet (+hands if fists
  exist). Only after E3 shows headroom.
- **E5 — BeyondMimic delta.** Train raw-vs-polished tracking policies on G1 for 1–2 clips. The
  killer figure if positive; honest null is still reportable (it answers GMR's own open question
  for the floor class). Needs GPU allocation. ~weeks, start scoping at E3 time.

Stop-loss: if E3 shows no measurable polish delta on either clip type, Option A dies cheaply and
C's G1 amendment still stands on E1+E2 alone.

---

## 5. Mentor questions (delta vs paperIdea3 §7)

1. G1-as-primary-eval-platform: any objection? (It was already the sanctioned fallback.)
2. RA-L vs ICRA 2027 for Option A — timeline preference?
3. Clip release for the benchmark piece (unchanged, still open).
4. GPU time for E5 BeyondMimic runs.
5. Does the lab care that the *hardware* story stays Alex/IHMC while the *eval* story moves to G1?

---

## 6. Feasibility assessment (2026-07-15): code/data availability + effort estimates

Verified against the local clone at `../GMR` (github.com/YanjieZe/GMR, MIT license, remote-tracked,
up to date through Jan 2026). Paper: "Retargeting Matters: General Motion Retargeting for Humanoid
Motion Tracking" (Araujo, Ze, Xu, Wu, Liu — arXiv:2510.02252; GMR consistently beats PHC/
ProtoMotions/Unitree-dataset on LAFAN1 via BeyondMimic policy training).

### Availability — everything needed is public

- **Code:** fully open. Per-frame differential IK on mink+MuJoCo, `qpsolvers[proxqp]`. Robot models
  ship in-repo: `assets/unitree_g1/g1_mocap_29dof.xml` **plus a custom collision URDF**
  (`g1_custom_collision_29dof.urdf`) — collision geometry problem partially pre-solved.
- **Data:** LAFAN1 BVH = free direct download (Ubisoft repo, `lafan1.zip`); AMASS SMPL-X = free
  registration; OMOMO = public Google Drive. The paper's eval set is "a diverse subset of LAFAN1" —
  exact clip list must be extracted from paper/website (risk: may need author contact).
- **Policy-eval pipe pre-built:** GMR ships `scripts/batch_gmr_pkl_to_csv.py` explicitly "for
  beyondmimic"; BeyondMimic is open and G1-native. Replication cost = GPU hours, not engineering.
- **Conventions match ours:** GMR robot motion = (base trans, base quat **wxyz**, joint positions),
  MuJoCo-native. G1 29-DoF + free root = same qpos shape as Alex (7+29). No convention war.
- **Head start already in the clone (untracked, ours):** `scripts/build_fbx_kinematic_canonical_v2.py`
  (FBX/pkl/npz → canonical skeleton NPZ), plus `scripts/diagnostics/` and `scripts/visualization/` —
  this repo grew out of that clone; the bridge-building started once before.

### Q1 — Run GMR / replicate their results: EASY to MODERATE

Run out-of-box on G1: `pip install -e .` + LAFAN1 download → **~2–3 days** including smoke tests
(E1). Replicate the *paper* (policy training vs their baselines): the conversion pipe exists;
**~2–4 weeks dominated by BeyondMimic setup + GPU time**, not code. We don't need full replication
for Option A — raw-vs-polished on a few clips suffices (E5).

### Q2 — Our smoothing/GlobalOPT polish on GMR outputs: MODERATE (~1–2 weeks to first full run)

| Piece | Effort | Why |
|---|---|---|
| pkl→Stage-4 NPZ bridge | ~1 day | Mirror of `ihmc_json_to_stage4_npz.py`, simpler (no joint-order hash, same wxyz) |
| Stage A smoothing | ~free | Operates on qpos, model-agnostic |
| Stage 4.5 grounding | days | Needs G1 floor setup; mesh-exact machinery ports |
| Stage B contact QP | ~1 week | Needs ROLE_TO_G1_BODY map, foot support faces, **contact flags GMR doesn't output** — detect from G1 FK or source human (our height/velocity gates port), collision model from the shipped custom-collision URDF |

Cheapest credible demo (E3, Stage A + grounding only — the validated Luigi-polish recipe) is
**days, not weeks**.

### Q3 — Our contact-first Stage 3 + GlobalOPT on all their data: MODERATE-HEAVY (~3–5 weeks)

The blocker is stages 1–2 (currently manual Blender per-FBX — doesn't scale to LAFAN1/AMASS). The
unlock: **GMR's own `data_loader.py` already normalizes AMASS/LAFAN1/FBX into per-frame
(body_name → global pos + quat wxyz) dicts — exactly our canonical-human shape.** Adapter = role
mapping + our stage-2.5 contact labeling/grounding on top of their loader. Then Stage 3 needs G1
role map + support faces (fist analog: G1 hands/rubber tips), Stage 4 per Q2. Batch is fine —
our solver is offline and LAFAN1's eval subset is tens of clips.

Honest caveats: (a) per-robot tuning tax is real — ankle limits, foot geometry, support faces all
needed Alex-side iteration and will need it again; (b) on upright LAFAN1 locomotion/dance GMR is
well-tuned and we may only tie — our edge is floor-contact clips; verify which LAFAN1 clips
actually contain floor contact before promising that comparison; (c) their exact eval-subset clip
list is the one thing not confirmed downloadable.

### Bottom line

Nothing is blocked on availability — code MIT, data public, conventions compatible, collision model
shipped, BeyondMimic pipe pre-built, and our own FBX→canonical bridge already half-exists in the
clone. Effort ladder: **polish demo (days) → full polish w/ contact QP (1–2 wks) → full
contact-first pipeline on their data (3–5 wks) → paper-grade policy eval (add GPU weeks)**. This
matches the E1–E5 ladder in §4 — E3 remains the cheapest kill-test.

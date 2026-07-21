# GMR-S6 Plan — Hard Floor Constraint + Median-Centered Limb-wise Polish

Written by Fable 2026-07-17 for Sonnet to execute. Supersedes nothing — S5 stands; this
extends it. Read `GMR-S5-plan.md` §context first if cold. All conventions (held-mask,
class split, metrics, resumable-batch pattern) carry over from S5 unchanged.

## Why S6 exists — diagnosis of S5's shortfall (Fable analysis, 2026-07-17)

S5's contact layer (`gmr_contact_retarget.py`) wins the un-gameable joint metric by
12–93 points but fails Prabin's core expectation: *"even if tracking is compromised a
little, the contact physics we set will be respected."* Root causes, ranked, each
backed by the 77-clip range table (`scratchpad/range_summary.py` output, logged in
planLogGMR.md):

1. **Soft cost, not constraint.** The held-effector override is a weighted task
   (POS_COST_HELD=100) inside the same QP as tracking. The solver *trades* contact
   error against tracking error, so worst-case tails survive: loco worst-float only
   4.52→3.11 cm, worst-pen essentially unchanged (−3.45→−3.56 cm). Physics that can
   be traded away is not physics.
2. **Coverage: held effectors only.** Whole-body penetration is byte-for-byte
   untouched (floor class mean worst pen 15.29→15.35 cm, max 31.02→31.05 cm). Knees,
   toes, torso, swing feet — the majority of penetration mass — have no term at all.
3. **Nothing forbids the floor.** When human data + morphology mismatch makes
   "everything above floor AND feet on floor" conflict with strong pelvis tracking,
   the QP resolves by penetrating, because penetration costs zero.
4. Contact-transition jerk residual (known, secondary — do not chase in S6 unless a
   gate below forces it).

Consequence: `gmr_heightfix` still "beats" us on the whole-body pen column (the one
number a rigid shift is constructed to minimize), even though its held-foot range is
provably identical to raw (7.98=7.98, 12.84=12.84 — a rigid shift cannot change
spread). To win BOTH columns we must make non-penetration structural.

**S6-A1 (DONE, by Fable, before handoff — see `## S6-A1` in planLogGMR.md for full
detail) already ran the "append `mink.CollisionAvoidanceLimit`" experiment so Sonnet
doesn't have to re-derive it from scratch. Two real findings changed Phase A's
mechanism. Read the planLogGMR.md entry before starting; summary:**

1. **Real bug, fixed, worth keeping regardless of mechanism.** GMR's own `retarget()`
   calls `mink.solve_ik(configuration, tasks, dt, solver, damping, self.ik_limits)` —
   6 positional args. Installed mink's signature has `safety_break` at slot 6,
   `limits` at slot 7. So `self.ik_limits` silently binds to `safety_break`, never
   reaches `limits` — any limit GMR (or we) append is silently dropped; mink falls
   back to a fresh default `ConfigurationLimit` (accidentally equivalent to GMR's
   own first list entry, so joint limits still work by luck — but `VelocityLimit` and
   anything we append do not). Our own `_solve_after_targets` in
   `gmr_contact_retarget.py` is a verbatim copy of this same buggy call — fix it
   there (never touch `~/projects/GMR`): pass `limits=r.ik_limits` as a **keyword**,
   not positionally.
2. **Even fixed, `CollisionAvoidanceLimit` doesn't reach near-zero penetration in
   GMR's usage pattern — this is why Phase A's mechanism changes below.** GMR's
   table2 solve loop exits after ~1 solve+integrate call per frame (task error stops
   improving fast), which starves the constraint's rate-limited (CBF-style) bound of
   the iterations it needs to converge. Measured on walk1_subject1 frame 135 (worst
   violation, left foot): 0 extra iterations = −1.274cm (byte-identical to no
   constraint at all); +5 forced extra iterations = −0.416cm; +20 = −0.344cm; +50 =
   plateaus ~−0.342cm (never reaches exact zero — residual equilibrium against
   competing tracking pull). Forcing 50 extra QP solves/frame is also a real
   per-frame cost multiplier. Additionally, GMR's own G1 XML explicitly excludes the
   foot's real mesh from collision (`contype="0" conaffinity="0"` on the STL geom,
   `g1_mocap_29dof.xml:83`) — the only collidable foot geometry is 4 tiny 5mm
   marker-dot spheres (likely leftover mocap-marker artifacts, not a designed contact
   proxy) that don't match OUR vetted eval mesh at all. Two independent, compounding
   reasons the pure rate-limited-QP-inequality approach won't hit the <1cm gate.

**Revised Phase A mechanism: skip rate-limited QP inequalities for the floor term.
Use a direct, exact, deterministic per-frame clamp instead** — compute each watched
body's lowest point via OUR OWN mesh cache (exact, matches every eval this project
uses, sidesteps GMR's crude/excluded foot collision geometry entirely), and if it's
below floor, apply a small follow-up damped-least-squares correction on that limb's
joint chain to lift it to zero. This is structurally the SAME correction Phase B
needs (S6-B1's "swing foot: clearance clamp" and "held foot: locked target") — build
ONE shared leg/arm-DLS clamp module and apply it in two modes: **A = inline, per
frame, inside GMR's own solve loop** (this phase); **B = post-hoc, corpus-level, on
top of median-centered raw/heightfix output** (Prabin's original idea, unchanged
below). Do not duplicate the DLS-correction implementation between A and B.

## Phase A — exact per-frame floor clamp inside GMR's solve loop

Goal: whole-body penetration → ~0 **by construction** (exact, not asymptotic), held-
foot contact kept by the existing S5 cost layer, tracking allowed to degrade as
needed (explicitly accepted by Prabin — record fidelity, do not gate on it).

### S6-A2: `scripts/g1/leg_floor_clamp.py` — shared correction module (build first, used by both A and B)
- `compute_lowest_point(model, data, mesh_cache, body_id)`: reuse
  `post_process_ground_contactfirst.py`'s `_geom_lowest_z`/mesh-cache pattern,
  restricted to geoms under one body (or a whole kinematic chain) — exact mesh
  point, not GMR's collision proxy.
- `clamp_limb(model, data, mesh_cache, chain_dofs, watched_bodies, floor_margin=0.0)`:
  for each watched body (feet: `left_ankle_roll_link`/`right_ankle_roll_link`; extend
  to hands for `--effectors feet+hands` per S5 convention), if lowest mesh point <
  floor_margin, do a small DLS correction restricted to `chain_dofs` (leg: hip×3,
  knee, ankle×2; arm: shoulder×3, elbow, wrist×2) that lifts the body by exactly the
  violation amount along +Z — target-space-only (position, not orientation), so it
  can't fight a held-effector's own XY/yaw target. Held effectors already at their
  locked Z from S5's `_z_support` should rarely trigger this (verify empirically,
  don't assume).
- No QP, no rate limiting, no iteration tuning — single closed-form DLS solve per
  violating body per frame. Cheap (small chain, few DOF), exact (our mesh geometry),
  deterministic (no asymptote).
- Unit-test standalone on a few frames of walk1 before wiring into either A or B.

### S6-A3: wire into `gmr_contact_retarget.py` as opt-in flag
- Fix the positional-arg bug in `_solve_after_targets` first (`limits=r.ik_limits`
  keyword) — real fix, keep regardless of what else ships.
- Add `--floor-clamp` flag. When set, call `clamp_limb(...)` for all four limbs (or
  feet-only per `--effectors`) once per frame, right after `_solve_after_targets`
  returns, before appending to `qpos_list`. Do NOT change defaults; S5 behavior must
  remain reproducible without the flag.
- Sanity: `--no-contact --floor-clamp` = "GMR + floor physics, no contact targets" —
  build this variant on dev clips too; cleanest ablation, isolates the clamp from the
  S5 cost layer.

### S6-A4: dev-clip eval + gate
- Dev clips: walk1_subject1, walk3_subject1, run2_subject1 (loco); ground1_subject1,
  fallAndGetUp1_subject1 (floor).
- Variants: `gmr_raw`, `gmr_heightfix`, `gmr_polished`, `gmr_contact` (S5),
  `gmr_contact --floor-clamp` (new, call it `gmr_contact_fc`), `--no-contact
  --floor-clamp` ablation. Always include `gmr_polished` — see "Baseline integrity"
  below, it's a real comparison target, not optional.
- Metrics: everything from S5 (`sprint_s5_metrics.py`: joint_ok, skate, fidelity,
  jerk) PLUS the range analysis (worst float / worst pen / range per clip) — promote
  `scratchpad/range_summary.py` into `scripts/g1/sprint_s6_range_summary.py` first
  (it currently lives only in the session scratchpad and will be lost).
- **Gate (loco dev clips):** mesh-exact whole-body worst pen < 1cm (should be much
  tighter than that now — it's an exact clamp, not asymptotic; if it's not near-exact
  zero, something is wrong with the clamp, not a tuning question); held-foot range
  meaningfully collapsed (target: < 4cm vs current 6.67cm mean). Fidelity and jerk:
  RECORD, do not gate — Prabin explicitly accepts tracking degradation. Watch
  specifically for jerk at the clamp's on/off transition (no ramp on this mechanism
  by design — cosine-ramp it like S5's contact layer if jerk is bad, 1 tuning
  attempt).
- **Floor clips:** expect large tracking distortion (crawl frames force lifted/bent
  poses). Record honestly. If the solver fights itself into garbage (oscillation,
  non-convergence), report that as a finding, don't tune endlessly — 2 tuning
  attempts max, then log and move on.
- Log `## S6-A4` with the full table.

### S6-A5: 77-clip corpus (only if A4 gate passes on loco)
- Extend `sprint_s5_corpus.py` (or new `sprint_s6_corpus.py`) to build
  `gmr_contact_fc` for all 77 clips — resumable, skip-if-exists, `run_in_background`.
- Eval → append variant rows to a new `s6_full_corpus.csv`; class-split summary
  (extend `sprint_s5_summary.py` pattern) + the range table.
- Log `## S6-A5`.

## Phase B — Prabin's experiment: median height fix + limb-wise IK polish

Prabin's proposal, Fable-endorsed with one amendment. Idea: take GMR's raw output,
apply a *median* height fix (center the held-frame error distribution so float and
penetration are balanced and every subsequent correction is small), then run cheap
per-limb IK passes: held effectors → maintain contact, swing/non-contact ones →
maintain clearance.

Why this is sound (and why the old negative doesn't apply): week-2's "contact-aware
grounding" negative failed because *shifting alone* converts pen into float at held
effectors with nothing to close it. Here the limb-wise pass exists precisely to close
that float. Median-centering first means corrections are minimal and two-sided —
well-conditioned, low jerk risk, and root/torso tracking is untouched by construction.
It's also retargeter-agnostic: a post-process applicable to raw GMR, heightfix, or
Phase-A output — paper-valuable on its own.

**Fable's amendment (build as variant B1b, not instead):** a rigid median shift
cannot fix torso/pelvis penetration on floor-class clips (limbs can't lift the
trunk). B1b = replace the constant shift with a *per-frame smoothed* root-z offset:
per frame, offset = max(median-shift, lift needed so mesh-exact lowest robot point
≥ 0), then smooth the offset curve (moving average or Savitzky-Golay, window ~15
frames @30fps) to kill jerk, then the same limb-wise pass. B1b is the only version
with a chance on the floor class.

### S6-B1: `scripts/g1/polish_median_limbwise.py`
- Reuse `leg_floor_clamp.py` (built in S6-A2 — same DLS correction, same
  `compute_lowest_point`/`clamp_limb` functions) for the per-limb pass below. Do NOT
  reimplement the DLS chain solve here — this script is a corpus-level driver around
  the shared module, not a second mechanism.
- Input: any qpos pkl + grounded canonical (for held masks, same recipe as
  everywhere: debounced contact_flags AND speed < 0.05).
- Step 1 — centering: per-clip median shift over held-frame `support_z` (flag
  `--center median`); variant `--center perframe` = B1b smoothed root-z as above
  (mesh-exact lowest z from OUR vetted model, reuse mesh_cache).
- Step 2 — limb-wise IK, per frame, root+waist frozen, via `clamp_limb`:
  - Held foot: target = (onset-locked XY, `_z_support` Z, flat yaw) — same target
    recipe as S5's contact layer (this is a small extension to `clamp_limb`'s
    "clearance only" mode from S6-A2 — held effectors need a full locked-position
    target, not just a Z-floor; add a `target_xy`/`target_yaw` param, keep the
    clearance-only path as the default when no target is given).
  - Swing foot: `clamp_limb`'s clearance-clamp-only mode, unchanged from S6-A2.
  - Hands (`--effectors feet+hands`): same held logic via arm chains — floor class
    only benefit.
  - Cosine ramp corrections in/out of held segments (RAMP_FRAMES=5 heritage) to
    control transition jerk.
- Keep it per-frame and local — NO trajectory-wide QP here; the whole point is
  cheap, well-conditioned, small corrections.

### S6-B2: dev-clip eval
- Apply to `gmr_raw` (primary) on the 5 dev clips; both `--center` variants.
- Same metric battery + range table. Compare against: gmr_heightfix, S5 gmr_contact,
  and S6-A's gmr_contact_fc.
- **Gate:** beats gmr_heightfix AND gmr_polished on joint_ok AND range on loco dev
  clips; jerk not worse than S5 gmr_contact.
- Log `## S6-B2` with the comparison table and an explicit verdict on Prabin's
  question ("is median-centering + limb-wise IK a bad idea?" — answer with numbers).

### S6-B3: 77-clip corpus (only if B2 gate passes)
- Winning `--center` variant only. Resumable build + eval + class-split + range.
- Log `## S6-B3`.

### S6-B4: decision memo
- One planLogGMR.md entry `## S6-DECISION`: which is the paper's method —
  integrated (A), post-hoc (B), or A-then-B stacked (if both pass, try the stack on
  the 5 dev clips before deciding; cheap). State the honest trade: tracking-fidelity
  cost of each on both classes.

## T-DOC (pre-authorized for this sprint only)
- Update `wiki/experiments/` — new page `gmr-baseline-sprint-s6.md` (diagnosis above,
  A/B results, decision); mark s4-s5 page's "open items" accordingly.
- Update `wiki/index.md` (one line), append `wiki/log.md` one-liners per phase.
- Update `GMR-baseline-results.md` with an S6 section.
- No other docs. No commits (standing rule: never git add/commit/push unprompted).

## Baseline integrity (Prabin, 2026-07-17 — read before touching anything GMR-related)

**The comparison points are `gmr_raw`/`gmr_heightfix` (GMR's own code, exactly as
shipped — matches their paper's numbers) and `gmr_polished` (GMR + our Stage-A
tridiagonal-smoothing + grounding polish, `polish_gmr_pkl.py` — a global-trajectory-
optimization baseline with some respect to contacts). Both are fixed comparison
targets. We are trying to beat both with OURS. Never "fix" anything in how these two
are generated, including bugs — GMR-as-shipped means GMR-as-shipped, bugs included,
so the comparison stays honest against what their paper actually reports.**

Concretely: `gmr_headless_retarget.py` (source of `gmr_raw`/`gmr_heightfix`) calls
GMR's own unmodified `retargeter.retarget()` — confirmed untouched, leave it that
way. The positional-arg `limits=`/`safety_break` bug found in S6-A1 lives ONLY in
`_solve_after_targets` inside `gmr_contact_retarget.py` (OUR `ContactAwareGMR`
wrapper, used for OUR method's contact layer, not the baseline generator) — fixing
it there is fixing OUR code, not GMR's, and is correct/required. Do not fix the same
bug (or any other) in `gmr_headless_retarget.py` or `polish_gmr_pkl.py`.

**Status check (Fable, 2026-07-17, from existing `s5_full_corpus.csv` — answers
"how are we doing on the stack against GMR+polish"):** on the un-gameable joint
metric (`joint_ok_pct`), `gmr_polished` is not a strong baseline — it's WORSE than
doing nothing (`gmr_raw`) on both classes, because Stage-A's smoothing/grounding
amplifies the same float-to-hide-penetration problem as `gmr_heightfix`:

| class | gmr_raw | gmr_heightfix | gmr_polished | gmr_contact (OURS, S5) |
|---|---|---|---|---|
| locomotion (43) | 91.52% | 46.26% | 32.15% | **94.18%** |
| floor (34) | 80.64% | 0.19% | 0.36% | **84.01%** |

OURS already clearly beats `gmr_polished` on both classes, by a wide margin, before
any S6 work. S6's job is to close the remaining gap the joint metric doesn't
capture (worst-case penetration/float range, per Prabin's range-collapse ask) — not
to catch up to `gmr_polished`, which we're already ahead of. Keep `gmr_polished` as
an explicit column in every S6 table (A4, A5, B2, B3, decision memo) so this stays
visible, not just gmr_raw/gmr_heightfix.

## Standing rules for Sonnet (unchanged from S5)
- Held-mask convention, class split (`s1t4_reclass.csv`), eval always on OUR vetted
  collision model, honest negatives logged, 2-attempt tuning cap per mechanism,
  background builds with skip-if-exists resumability.
- planLogGMR.md is the only doc written unprompted outside pre-authorized T-DOC.
- If a phase-A/B gate fails: log it, stop that phase, do NOT silently continue to
  corpus scale.

# GMR-baseline week 2 (2026-07-15, branch `gmr-baseline`)

Follow-up to [[gmr-baseline-week1]] after a paper cross-read (`RetargetMatters.pdf`) and Prabin's
directive to also demonstrate parity/wins on GMR's OWN included motion class, not just the
excluded one. Full task-by-task trail: `planLogGMR.md` (repo root, `## W2-Tn` headings). Plan:
`GMR-baseline-plan.md` Week-2 section. Strategy doc: `GMR-baseline.md` §7.

## W2-T1 — fair-baseline addendum: GMR's own height fix + a floating metric

Added a floating metric (`float_max_cm`/`float_pct` — mirror of the existing penetration metric,
whole-body lowest point's height ABOVE z=0) to the shared `evaluate()` — zero new computation,
additive columns only, regression-gated. Replicated GMR's OWN paper-described (but shipped-code
`HEIGHT_ADJUST=False`-disabled) height fix as a separate baseline column (`gmr_heightfix()` in
`polish_gmr_pkl.py`, never touching the GMR clone).

**Result: the fix is real (65-82% floor-pen reduction on the 3 floor clips) but trades directly
into near-universal floating (94-99.9% of frames on every floor clip)** — a single clip-global,
body-origin-calibrated shift can't reconcile "worst frame's penetration" with "every other frame's
correct height" at once. Applying GMR's own described fix and still finding ~99% floating
STRENGTHENS the motivation figure (forecloses "did you even try their own fix"). Also found: our
OWN week-1 "polished" deliverable shares this exact float% signature (96-99% on every floor clip) —
whole-clip Z-calibration, ours or theirs, cannot produce a body that's actually resting on the
floor throughout a clip. Numbers: `planLogGMR.md` W2-T1.

## W2-T2 — closed the one unverified E4 claim

E4's `walk1_subject1` 25%-slip-reduction was stage_b's own internal metric. Independent check
(`scripts/g1/check_slip_independent.py`, no shared code with the metric being checked — different
position reference, different drift convention): **direction confirms** (8-9% mean / 5-23% max
drift reduction, both feet), magnitude differs from the internal 25% as expected given the
different methodology. Safe to cite now, with the caveat that "25%" specifically stays an internal
number.

## W2-T3/T4/T5 — E4b: multi-surface contact anchoring attempt, CHECKPOINT M3 (negative)

The E4b redesign (diagnosed after E4's feet-only park): detect contact on the HUMAN source
(uncorrupted) instead of the robot's own corrupted output, cover ALL support surfaces (feet,
hands, knees, elbows, +pelvis/torso opt-in) not just feet, and PULL support points to the floor
instead of merely holding them in place.

- **W2-T3 (kill-test #1, human-side multi-surface labels): PASSED cleanly.** Height-gate-only
  detector (`scripts/g1/human_contacts_lafan1.py`, no speed gate — the E4 lesson) on LAFAN1's own
  BVH bones. Controls near-zero non-foot contact (walk1 exactly 0%, dance1's hand blips are
  noise-level 0.2-0.6%); all 3 floor clips show sustained, anatomically-sensible multi-surface
  contact (hands 25-88%, knees 7-83%, elbows 10-65%, pelvis 22-55%, torso correctly 0% on the
  hands-and-knees crawl clip vs 14-19% on the two lying-flat clips). Cross-validated against
  independent robot-side `support_z` measurements at the same frame (pelvis heights agree within
  ~1cm across two completely different measurement paths).
- **W2-T4 (role map + support_z): passed.** `ROLE_TO_G1_BODY` (10 roles) + mesh-exact
  per-body-lowest-point helper, gated on standing/lying poses.
- **W2-T5 (pull-to-floor Stage B): CHECKPOINT M3, negative result — reported, not pushed
  further.** Anchors engage (confirmed nonzero zone/planted/`|dQ|`), move 5/8 roles in the right
  direction LOCALLY (1.5-3.5cm each, confirmed via direct whole-body-lowest-z measurement at the
  known frame-356 corpse pose), but **the whole-clip aggregate metrics show a net REGRESSION on
  floorPen on 4/5 clips, including both controls** — pulling anchored joints costs a little
  penetration elsewhere more often than it fixes floating locally. Root position never moves;
  max joint-angle change anywhere in any clip is only ~0.3 rad — the trust-region-limited local QP
  fundamentally cannot close a 5-13cm gap this way. Visual check (frame 356, raw/polished/
  multisurface side by side) confirms: all three poses are indistinguishable, feet still visibly
  floating. **Conclusion: anchoring-on-top-of-polish is not the corpse-pose fix, regardless of
  which contact source or floor-pull mechanism drives it — this motion class needs contact-first
  SOLVING (root + contacts planned jointly from the start, à la Alex's Stage 3), not a post-hoc
  anchor on an already-fixed whole-body trajectory.** A real scope decision, not a natural
  continuation of this week's approach — flagged rather than unilaterally expanded (`--anchor-trunk`
  not tried given this result). Full numbers + the frame-356 visual: `planLogGMR.md` W2-T5.

## W2-T6 — self-collision vetting: PASSED

GMR ships `g1_custom_collision_29dof.urdf` with only 11 of 46 `<collision>` blocks actually
uncommented (simplified cylinder proxies on hip-yaw/knee/torso/shoulder/elbow joints — the ones
most prone to self-intersection). Loads directly via MuJoCo's own URDF importer (no graft needed —
its compiler already separates visual meshes, contype=0, from the 11 real collision cylinders,
contype=1); actuated joint order verified identical to the mocap XML, so existing qpos arrays feed
it directly. **`walk1_subject1` raw: 0.2% self-collision (down from 18.2% mesh-noise)** — clears
the <1% gate. Full raw/stageA/polished table now available and physically sensible (controls near
zero, floor clips 2.5-5.8%, plausible for genuine lying/crawling self-contact). File:
`outputs/gmr_baseline/g1_collision/g1_collision_vetted.urdf` (copied+patched from the GMR clone,
never edited in place).

## W2-T7 — contact-aware grounding: negative, `constant` mode stays shipped

Built the thin adapter the plan anticipated (`scripts/g1/ground_g1_contact_aware.py` — G1 has no
named sole sites, so `post_process_ground_contactfirst.py`'s own `hybrid`/`constant-contact` modes
would silently no-op; re-derived the same `plant_data` shape from G1's sole-marker geoms + W2-T3's
human zones, handed off to the unmodified percentile/lift-QP code). **Result: both new modes are
dramatically WORSE than the already-shipped `constant` mode on every clip** (e.g. `walk1_subject1`
floorPen 0.7cm→6.6cm constant-contact / 3.3cm hybrid) — opposite of the plan's expectation.
Partially diagnosed: G1's sole-corner marker spheres sit ~0.6-1.3cm above the foot's true mesh
floor contact (confirmed directly), likely compounded by the human zone's loose 5cm threshold
not tightly indicating genuine full-weight stance the way Alex's site-based convention does.
**Decision: no change — `constant` (week 1's choice) ships as-is.**

## Net effect on the paper narrative

- Axis 2 (floor-contact class) motivation is now STRONGER (fair-baseline addendum) but the
  "per-limb contact fix" story did not advance this week — E4/E4b both landed on the same
  boundary (whole-clip fixes work, per-limb anchoring on top of them doesn't), sharpening rather
  than closing the gap the paper needs to address. Next real lever: contact-first solving on G1,
  a Week-3+-scope decision.
- Self-collision is now a usable, citable axis on G1 (was noise-only through week 1).
- No wasted week-1 work: `constant` grounding and Stage-A polish are unaffected and remain
  shipped exactly as validated.

## New code (all under `scripts/g1/`, branch `gmr-baseline`)

`check_slip_independent.py`, `human_contacts_lafan1.py`, `ground_g1_contact_aware.py` (new);
`eval_motion.py`, `polish_gmr_pkl.py`, `stage_b_g1.py` (extended, backward compatible — original
E4 CLI paths untouched). `scripts/eval_ihmc_json.py` gained the float metric (additive, shared
with Alex).

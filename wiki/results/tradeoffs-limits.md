# Trade-offs, Limits, Overclaim Risks

## Session conclusion (2026-07-13, Prabin): embodiment gap, not formulation — read before another retargeting-tuning pass
A full session was spent chasing the get-up-class foot-floor penetration purely through retargeting-
formulation changes (Stage-3 orientation caps, continuity regularization, self-collision boosts,
Stage-3 synthetic replanting, Stage-4 tracking-weight scoping) — see `wiki/experiments/retired-
approaches.md` (`--swing-clear`, `--leg-floor-refine`) and [[globalopt]] (`--sens-foot-min-pen`) for
the full mechanism-by-mechanism trail. Two clean, independent mechanisms were separated:
1. **Stage-4-introduced** (smoothing amplifies a borderline Stage-3 dig) — genuinely fixable by
   formulation, and fixed (`--sens-foot-min-pen`, [[globalopt]]).
2. **Stage-3-originating, joint-range-limited** (the knee-140° embodiment gap below) — NOT fixable
   by formulation. Every attempt (orientation cap, synthetic-plant + relaxed root tracking) either
   did nothing (correctly recognized Stage 3 wasn't the bottleneck) or produced a genuinely
   contorted pose (forcing a foot flat while the leg physically cannot fold as tight as the human
   demands a locally infeasible configuration — no amount of weight-tuning fixes a body that is
   the wrong shape).

**Conclusion, confirmed by every mechanism tried, not just asserted**: where Alex's morphology/
joint range genuinely diverges from the human demonstrator's, faithfully copying the human is the
wrong objective — there is no tuning that reconciles a physically impossible target. The
actionable shift: treat human motion as a **reference/inspiration** for the retargeted motion where
kinematics allow, not a literal target to match everywhere. Formulation-only fixes (weights, caps,
ramps, regularizers) remain the right tool for genuine PIPELINE artifacts (Stage-4 smoothing
blind spots, registration mismatches) — but stop reaching for them once a clip's problem traces to
morphology/joint-range, and reach instead for a different objective (rest-on-floor, feasible-pose)
rather than a closer-fidelity one.

## Fundamental limits
- **Kinematics only.** No torque/GRF/ZMP feasibility; no guarantee the trajectory is dynamically trackable. Job: hand RL a trajectory with no kinematic impossibilities.
- **Contacts are high-weight SOFT, not exact.** Residual slip is a weight equilibrium (see [[metrics]] for the honest 1–9 cm range).
- **Contact detection is heuristic** (height/speed thresholds on human markers), never validated against ground-truth contact.
- **Joint-range embodiment gap on deep-tuck get-ups (knee 140° cap).** Alex's `RIGHT/LEFT_KNEE_Y` max flexion is **140°**; a human supine/prone get-up folds the leg tighter than that (heel toward glute). When the human leg is folded past 140°, Alex's knee saturates at the limit and the foot cannot tuck as far — the rigid ~20 cm foot plate's heel/toe corner then juts to **~14 cm below the floor** during the deepest crouch of the transition. Verified on `luigi_standSupine_08` (its ~14 cm `anyPen`, right foot @fr679, pelvis 32 cm mid-rise): the ankle *target* is correct (+8 cm above floor), but the achieved ankle sags ~10 cm below it with the knee pinned at 140° the whole window — **not a registration error, not a swing dig, not a tracking bug**. Compounded by `--floor-phase-aware` legitimately gating floor collision OFF during this low-pelvis phase (active only ~34% of frames). This `anyPen` has been **~constant (13.8→14.1 cm) across every pipeline version** (M0 baseline → M2 → swing-clear) — nothing has ever reduced it, because no retargeting trick adds joint range. Same class as the 7 whole-body-lying clips [[limb-cleanup]]'s root-frozen solver can't reach; the only real fix is a coordinated whole-foot re-placement (rest the foot ON the floor, let leg/pelvis absorb the deep-tuck deficit), a bigger-IK project. **For RL: this foot is not load-bearing during the transient, so a dipping heel here is likely tolerable — accept, don't chase with target/phase tweaks.** Do NOT re-investigate as a bug.

## Deliberate trades
- Fullmesh + soft-collision: ~zero penetration bought with ~1 cm extra slip on hard plants (see [[fullmesh-vs-primitive]]). Penetration poisons physics-RL; slip is learnable.
- "One config for all actions" is a design stance, not a validated generalization claim — 18 in-house clips, 3 motion families.

## Overclaim risks (for any writing — paper, slides, README)
1. "Self-collision-free" is FALSE universally: residual 11–32.5% collision frames on several get-up/kneel clips, peak ~2.3 cm. Say "reduced, driven to 0 on many clips".
2. Plant-slip "1.0–1.5 cm" is shovels-only cherry-pick; get-ups reach 8–9.3 cm, and Stage B sometimes increases slip vs Stage A on the hardest clips.
3. No baseline retargeter has been run ⇒ no evidence the method is *better*, only that it *runs*. No comparative language without a baseline experiment.
4. No hardware playback, no policy training, no public benchmark (LAFAN1), no user study — see [[publication]].

## Known cosmetic/debt items
- Stage B docstring still claims "hard equality" feet — stale, fix on next touch of the file.
- Already-shared `unified/` videos show the OLD V1-mesh render body.

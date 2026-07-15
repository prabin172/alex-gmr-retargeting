# Related-Work Landscape (Undermind report, 2026-07-03)

Source: `Undermind - Hardware-ready kinematic contact-first humanoid motion retargeting...pdf` (repo root, 38 pp, 170 refs). Free-tier report; full version online (link on every page footer).

## Verdict
Our niche — hardware-ready, kinematic-centric, contact-first retargeting of floor-contact postural motions with full-clip smoothing and target-side geometric feasibility editing — is **not occupied by any single work**. Field splits into: (1) per-frame kinematic retargeters (upright, feet-only), (2) offline whole-sequence retargeters (mature but upright/loco-manipulation), (3) teleop/learning pipelines (floor transitions exist but are RL-policy-emergent), (4) IK target-feasibility editors (closest philosophy, feet-only, upright).

## Four things NOBODY does
1. Precomputed robot-feasible floor-stance libraries (kneel/all-fours/prone templates under joint limits) + human→template mapping.
2. Full-clip contact SEQUENCING for floor transitions (extract → edit to feasible robot contact sequence → optimize with CoM feasibility).
3. Target-side editing for knees/pelvis/forearms (today: foot placement + root only).
4. Purely kinematic sequence-aware retargeter whose floor clips play on hardware through a standard stabilization stack (all hardware floor transitions today are RL-made).

## Closest competitors (cite as named baselines)
- **OmniRetarget** — interaction mesh, contact-first kinematic, loco-manipulation; no kneel/prone transitions. Open source ⇒ baseline.
- **GMR** — clip-level target-side kinematic optimizer + BeyondMimic eval; EXPLICITLY excludes floor motions. Open source ⇒ baseline + motivation quote. **Empirically confirmed 2026-07-14** (not just quoted): fresh upstream clone, run on its own LAFAN1 benchmark's floor-contact clips (unmodified) — 12.9-15.9cm max floor penetration affecting 39-91% of frames, vs 1.0-7.1cm/0.3-1.9% on locomotion controls. Zero velocity spikes throughout (its per-frame IK is smooth even while failing on floor contact — the floor-contact gap is orthogonal to jitter, not correlated with it). Full numbers: [[gmr-baseline-week1]].
- **Jeong rig-unification / CoRe** — explicit target-side editing + contact-aware refinement, foot-only, upright. Closest philosophy cluster (Choi/Jeong/Kim group).
- **SPARK / KDMR / DynaRetarget** — whole-trajectory kinodynamic (torque/GRF); killed the old "offline global-opt is novel" claim; we position as the kinematic+quasi-static alternative.
- **IKMR / AdaMorph / diffusion retargeters** — learning-based sequence retargeting, not kinematic-centric, no floor geometry.
- **TeleGate / HuMI / HumanPlus / OmniTrack** — real stand-up/fall-recovery/kneeling on hardware but via RL policies, not retargeting.
- **PressMimic** (adjacent) — pressure-guided capture for floor contact; possible contact-label validation source.
- **Infant retargeting (Fiala, López)** — non-upright postures but small robots, developmental focus.

Most-cited foundations in this space: GMR (0.62 reference rate), OmniRetarget (0.38), Darvish whole-body geometric retargeting (0.34), Penco robust real-time (0.28), Ayusawa & Yoshida morphing (0.19).

Full positioning table + reviewer-attack defenses: `paperIdea3.md` §1–2. See [[publication]].

# alex-gmr-retargeting

Human MoCap (FBX) → canonical skeleton → contact-first MuJoCo QP IK → IHMC Alex biped (36-DOF).
Math ground truth: `METHOD.md`. Knowledge base: `wiki/`. Current state: `SESSION_HANDOFF.md`.

## Session start (do this before exploring code)
1. Read `SESSION_HANDOFF.md` — current state, decisions, next steps.
2. Read `wiki/index.md` — then open ONLY the wiki pages the task needs. Do NOT re-explore the codebase for anything the wiki already covers.

## Session upkeep
- Update `SESSION_HANDOFF.md` only when Prabin instructs. Keep it a pointer, not an archive — trim
  finished threads back to one line once their detail lives in `wiki/log.md`/`planLog*.md`.
- The wiki is yours to maintain: after meaningful work (new result, decision, experiment, diagnosis), update the touched wiki pages, keep `wiki/index.md` lean and current, append one line to `wiki/log.md` (`## [YYYY-MM-DD] <op> | <what>`). Don't duplicate METHOD.md math in the wiki — summarize and point.
- `wiki/log.md` is append-only and can get long — never read it in full. Newest entries are at the
  bottom (confirmed by inspection, not by the header's own wording): use `tail -N` for "what's
  recent," or `grep "^## \["` for a specific date/topic, and quote only the matched lines back.

## Conventions (critical — violations corrupt data silently)
- Coord frame: +X forward, +Y left, +Z up. Quaternions: **wxyz** everywhere.
- Free root qpos: `[x, y, z, qw, qx, qy, qz, 29 joints]` — 0–6 root, 7–35 actuated.
- Morphology scaling: motion *deltas* from rest only, never absolute root/pelvis position.
- Orientation: semantic frames from landmark positions (not raw FBX rotations), world-delta transfer.

## Footguns
- `assets/alex/alex_floating_base_with_sites.xml` is hand-maintained; the model-prep scripts (`create_alex_mujoco_sites_model.py`, `build_alex_v2_collision_model.py`, `prepare_*`) are historical and would OVERWRITE it — never run blindly.
- Settled decisions in `wiki/concepts/design-philosophy.md` — don't re-litigate.

## Pipeline (details: wiki/concepts/pipeline.md)
Stages 1–2 per-FBX by hand (Blender); stages 3–5 = `./retargetingPipeline.sh`.
`assets/`, `data/`, `outputs/`, `SESSION_HANDOFF.md` are git-ignored, local only.

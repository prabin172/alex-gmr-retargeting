# Fullmesh vs Primitive Collision (FullURDF adoption, 2026-07-02)

Real V2 URDF arrived with LEG collision as convex-hull STL meshes instead of the old hand-tuned primitives. Kinematics UNCHANGED (joints/axes/limits/link frames identical) ⇒ Stage 3 IK output identical, NOT re-solved; only Stage 4+ re-ran on fullmesh.

## The failure it exposed
Fullmesh legs made Stage B's **hard** collision inequalities primal-infeasible (row explosion: 424 vs ~80–194; genuinely-close legs in get-ups/kneels) → hard QP **silently no-op'd** (|δQ|max = 0). Fix: soft-slack reformulation (slack per collision row + quadratic penalty ρ=1000) — always feasible, degrades gracefully. Default hard path verified byte-identical to the shipped primitive result before the gate was removed. Now always-on (see [[globalopt]]).

## Measured trade-off (chosen deliberately)
- Self-penetration: `standup_side_04` collision frames 32.6% → **0%**, peak 5.2 cm → 0.
- Cost: get-up/kneel plant slip 2.8–3.4 cm → ~4.2 cm (~1 cm more).
- Rationale: penetration is physically impossible and poisons physics-RL; RL absorbs a cm of slip.
- BUT not universal: several get-up/kneel clips retain residual collision frames 11–32.5% (peak ~2.3 cm) — see [[tradeoffs-limits]] before claiming "collision-free".

## Aftermath (repo cleanup, committed 82ae714)
Fullmesh promoted to the ONLY model (`alex_floating_base_with_sites.xml`); primitive XML deleted; `_fullurdf` output dir suffixes promoted to plain names; `--soft-collision` gate removed. Ablation dirs `ab_primitive` / `ab_fullmesh` referenced in paper notes predate the rename.

Reference run log: `outputs/logs/fullurdf_pass_20260702_135608.log` (17/18 clips; `standupFromKneeling_02` completed separately).

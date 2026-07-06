# Trade-offs, Limits, Overclaim Risks

## Fundamental limits
- **Kinematics only.** No torque/GRF/ZMP feasibility; no guarantee the trajectory is dynamically trackable. Job: hand RL a trajectory with no kinematic impossibilities.
- **Contacts are high-weight SOFT, not exact.** Residual slip is a weight equilibrium (see [[metrics]] for the honest 1–9 cm range).
- **Contact detection is heuristic** (height/speed thresholds on human markers), never validated against ground-truth contact.

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

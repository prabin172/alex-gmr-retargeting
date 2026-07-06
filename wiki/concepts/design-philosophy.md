# Design Philosophy & Settled Decisions

## Philosophy (the through-line)
1. **Physical feasibility > verbatim copying.** Where the human is ground-supported, the robot reproduces the support surface even at the cost of departing from captured limb orientation.
2. **Contacts + end-effectors exact; body interior approximate.** Upper-arm/forearm/shin orientations left free.
3. **Kinematic infeasibility is fixed in the TARGETS, not by weight fights** — e.g. the shank clamp edits the knee target instead of fighting "foot flat" vs an infeasible knee (see [[contact-first-ik]]).
4. **Physics-RL absorbs dynamics errors, not kinematic impossibilities.** A cm of slip is learnable; self-penetration or over-limit joints are not.

## Settled decisions — do NOT re-litigate
- **ONE config for all actions** (Prabin's rule). Solver defaults + GlobalOPT λ=20, n_outer=3. Per-clip tuning forbidden (experiments only).
- **Stage B ON everywhere.** Old "plants not stationary" blocker was a contact-detection artifact; with hysteresis + foot-hold ×10 the plants are 0.1–0.3 cm and Stage B is well-posed. Stage A alone re-adds ~8 cm plant drift.
- **Everything is SOFT weights — no hierarchical optimization anywhere.** `--hierarchical` retired (see [[retired-approaches]]). Stage B foot pin is `add_soft` at weight 40; hard constraints only joint-limit box, trust region, collision inequality rows.
- **Soft self-collision always-on** in Stage B (`--collision-penalty 1000`); the `--soft-collision` gate was removed.
- **Root orientation NOT forced upright** — get-ups need the root to rotate; forcing upright makes limbs absorb the lying-to-standing rotation and pick bad IK branches.
- **Foot-flat gate stays 40°** (`--foot-flat-tilt`).
- **Foot-drag diagnosis:** achieved foot dragged by heavier pelvis/torso tasks, not target slip. Weight fixes cap out; Stage B is the principled pin.
- **Fullmesh is the real body**; primitives were a stopgap (see [[fullmesh-vs-primitive]]). Trade: 0% penetration for ~1 cm more slip — chosen deliberately.

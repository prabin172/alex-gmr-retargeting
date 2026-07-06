# Alex Model

IHMC Alex V2 humanoid, 36-DOF: 7-DOF free root + 29 actuated. All model files git-ignored (local only; obtain from teammate).

## Canonical files (hand-maintained — see FOOTGUNS)
- `assets/alex/alex_floating_base_with_sites.xml` — THE solver/GlobalOPT/grounding model. Convex-hull collision on ALL links incl. legs (fullmesh), named palm/sole sites.
- `assets/alex/alex_visual_mesh_fist_hands.xml` — render body: full visual mesh, closed-fist hands, native V2.
- `assets/alex/meshes/alex_V2_description/` — collision+visual STLs; `assets/alex/source/alex_V2_description/` — the real V2 URDF they came from (received 2026-07; kinematics identical to old model, only ZED mount removed + torso collision origin −0.018 z).

## Ankle (why the shank clamp exists)
Dorsiflexion 60° (human ~20°), plantarflexion 30° (human ~50°), roll ±25°, **no ankle yaw** (comes from hip), rigid foot. Asymmetric and stiffer than human → a trivially flat human plant can be kinematically impossible flat-footed on Alex → the shank clamp in [[contact-first-ik]] edits the knee target into the feasible cone.

## Canonical roles (15 + 4 contact sites)
`pelvis, torso, head, {left,right}×{hip, knee, foot, shoulder, elbow, hand}` + `{left,right}×{palm, sole}` sites. Solver maps some roles to Alex bodies under different names (`left_knee → LEFT_SHIN`, `left_ankle → LEFT_ANKLE_Y_LINK`, `left_wrist → LEFT_WRIST_X_LINK`). Canonical NPZ carries finer landmarks (`left_ankle, left_toe, left_hand_middle, left_hand_thumb, neck`, …).

## FOOTGUNS
- **Model-prep scripts are historical**: `create_alex_mujoco_sites_model.py` writes `alex_floating_base_with_sites.xml`; `build_alex_v2_collision_model.py` targets the deleted primitive model. Running them blindly OVERWRITES the hand-maintained fullmesh model. Same for the `prepare_*` scripts.
- Old render XML history: pre-cleanup `alex_visual_mesh_fist_hands.xml` rendered mostly V1 legs/arms — the already-shared `unified/` videos show the old body. Current file is true V2.

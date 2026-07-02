# alex-gmr-retargeting

Human MoCap (FBX) → canonical skeleton → MuJoCo QP IK → IHMC Alex biped.
Contact-first retargeting on the V2 (convex-hull) Alex model. Full method + math: `METHOD.md`.

## Layout
- `general_motion_retargeting/` — library (morphology scaling, canonical source adapter, robot configs)
- `scripts/` — pipeline scripts by stage (not importable)
- `scripts/legacy/` — retired code (old worlddelta solver family, MVNX path, old renders); kept for reference, not run
- `scripts/visualization/` — renderers
- `assets/`, `data/`, `outputs/` — git-ignored, local only (see README for expected structure)

## Conventions (critical)
- **Coord frame**: +X forward, +Y left, +Z up
- **Quaternions**: wxyz order everywhere (not xyzw)
- **Free root qpos**: [x, y, z, qw, qx, qy, qz, 29 joints] — indices 0–6 root, 7–35 actuated
- **Morphology scaling**: apply only to motion *deltas* from rest pose, never to absolute root/pelvis position
- **Orientation frames**: semantic (built from landmark positions), not raw FBX rotations — world-delta transfer

## Pipeline
Stages 1–2 are run per-FBX by hand (Blender); stages 3–5 are the batch `retargetingPipeline.sh`.
1. **FBX → positions** — `scripts/build_fbx_canonical_human.py` (Blender: `blender --background --python ... -- --fbx <f> --out <npz>`)
2. **Positions → orientation frames** — `scripts/build_canonical_orientation_frames_fresh.py` (auto-detects facing yaw)
3. **Contact-first IK** — `scripts/solve_fbx_canonical_alex_contactfirst.py` → `qpos (N,36)` NPZ
4. **GlobalOPT** — `scripts/solve_global_trajectory_opt_contactfirst.py` (Stage A smoothing + Stage B contact QP with always-on soft self-collision)
5. **Ground** — `scripts/post_process_ground_contactfirst.py`
6. **Render** — `scripts/visualization/render_contactfirst.py`

Batch (stages 3–5 for every clip in the CLIPS list): `./retargetingPipeline.sh`. Env knobs:
`LAMBDA_SMOOTH=20`, `N_OUTER=3`, `GROUND_MODE`, `RENDER_MESH` (visual|collision|path), `RENDER_DIR`, `GO_DIR`, `GR_DIR`.

## Model (single canonical)
`assets/alex/alex_floating_base_with_sites.xml` — 36-DOF (7-DOF free root + 29 actuated), convex-hull
collision on ALL links incl. legs. This is the solver default. Render body: `alex_visual_mesh_fist_hands.xml`
(full visual mesh, closed-fist hands). Both are git-ignored (local only).
- Self-collision in Stage B is **always-on soft slack** (`--collision-penalty 1000`); no hard-equality path.
- Ankle ranges: dorsiflexion 60°, plantarflexion 30°, roll ±25°, no ankle yaw (from hip), rigid foot —
  this asymmetry is why the shank-tilt clamp exists (see METHOD.md §5).
- **Model-prep scripts are historical**: `create_alex_mujoco_sites_model.py` writes
  `alex_floating_base_with_sites.xml` and `build_alex_v2_collision_model.py` targets the deleted primitive —
  do NOT run them blindly, they would overwrite the hand-maintained fullmesh model.

## Canonical roles (15 + 4 contact sites)
`pelvis, torso, head, left_hip, left_knee, left_foot, right_hip, right_knee, right_foot, left_shoulder,
left_elbow, left_hand, right_shoulder, right_elbow, right_hand` + `left_palm, right_palm, left_sole, right_sole`

## Working docs (git-ignored, local)
`SESSION_HANDOFF.md` (current state / decisions — keep updated), `paper_idea.md` (submission draft).

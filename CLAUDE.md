# alex-gmr-retargeting

Human MoCap (FBX/MVNX) → canonical skeleton → MuJoCo QP IK → IHMC Alex biped.

## Layout
- `general_motion_retargeting/` — library (morphology scaling, source adapters)
- `scripts/` — pipeline scripts by stage (not importable)
- `assets/alex/`, `data/`, `outputs/` — git-ignored, local only
- `experiments/` — scratch; not production

## Conventions (critical)
- **Coord frame**: +X forward, +Y left, +Z up
- **Quaternions**: wxyz order everywhere (not xyzw)
- **Free root qpos**: [x, y, z, qw, qx, qy, qz, 29 joints] — indices 0–6 root, 7–35 actuated
- **Morphology scaling**: apply only to motion *deltas* from rest pose, never to absolute root/pelvis position
- **Orientation frames**: semantic (built from landmark positions), not raw FBX rotations — use world-delta transfer

## Pipeline (4 stages)
1. **FBX → positions** — `scripts/build_fbx_canonical_human.py` (Blender)
2. **Positions → orientation frames** — `scripts/build_canonical_orientation_frames_fresh.py` (auto-detects facing yaw)
3. **IK solve** — `scripts/solve_fbx_canonical_alex_posori_qp_fresh_worlddelta.py` → `qpos (N,36)` NPZ
4. **Ground + contact labels** — `scripts/post_process_grounding_contacts.py` → Mimic-ready NPZ (`qpos_grounded`, `contact_labels (N,11)`)
5. **Render** — `scripts/visualization/render_alex_qp_direct_mp4_fresh.py`

## Active solver (Stage 3)
`scripts/solve_fbx_canonical_alex_posori_qp_fresh_worlddelta.py`
Uses rest-pose delta targets + world-delta orientation transfer. This is the canonical solver.

## Contact-first on Alex V2 (active — branch `feature/alex-v2-contact-first-ik`)
Contact-first IK + trajectory smoothing. Model `assets/alex/alex_floating_base_with_sites_v2.xml` (convex hulls).
- Solve: `scripts/solve_fbx_canonical_alex_contactfirst.py` (foot-flat + yaw, fist/palm support, make/break blend)
- Smooth: `scripts/solve_global_trajectory_opt_contactfirst.py` (Stage-A: joints + root; Stage-B contact-pin QP is off by default)
- Render: `scripts/visualization/render_contactfirst.py` (`--fixed-cam`, `--ground`)
- Run all clips: `run_globalopt_all.sh`. Details + decisions in `SESSION_HANDOFF.md`.

## Canonical roles (15 + 4 contact sites)
`pelvis, torso, head, left_hip, left_knee, left_foot, right_hip, right_knee, right_foot, left_shoulder, left_elbow, left_hand, right_shoulder, right_elbow, right_hand` + `left_palm, right_palm, left_sole, right_sole`

## Alex model
36-DOF: 7-DOF free root + 29 actuated joints. Model: `assets/alex/alex_floating_base_with_sites.xml`

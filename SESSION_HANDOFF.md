# Session Handoff — 2026-06-30 (contact-first redesign)

## Branch: `feature/alex-v2-contact-first-ik`

Big shift this session: moved from "track every joint's position + world-delta
orientation" to a **contact-first** IK on the new **Alex V2** model. Nothing is
committed yet — Prabin commits himself.

---

## Open issue to resolve next: JITTER / FLICKER

The contact-first videos have **more jitter and flicker** than the worlddelta
baseline. This is expected — there is still **no explicit cross-frame temporal
smoothing**. The solver only has per-frame warm-start + posture regularization;
relaxing intermediate-segment orientation and leaving knees/wrists loose lets
them flick within joint limits.

**Plan: absorb it with the global-OPT smoothing pass** (trajectory-level
smoothing with contacts pinned). Also still want the per-frame forward temporal
term. Do NOT over-stiffen the per-frame IK to hide jitter — keep contacts free
to find the pose, smooth afterward. See NEXT steps below.

---

## What was done this session

### 1. Alex V2 collision model
- `scripts/build_alex_v2_collision_model.py` builds
  `assets/alex/alex_floating_base_with_sites_v2.xml` from the mentor's
  `assets/alex/source/alex_V2_description/urdf/alexFullConvex.urdf`.
- **Convex hull collision** on arms (shoulder Y/X/Z, elbow, wrist Z/X), head,
  and the single **closed fist** per hand; **primitives** on legs/pelvis/torso
  (mirrors the URDF — "FullConvex" only meshes arms/head/fist).
- Original `alex_floating_base_with_sites.xml` left untouched.
- Convex STLs copied into `assets/alex/meshes/alex_V2_description/`.
- NOTE: `assets/` + `outputs/` are git-ignored → model/meshes/videos are
  local-only, shared via Slack. The V2 package `hardware/*.xml` are IHMC
  motor-calibration files, irrelevant to us.

### 2. Closed-fist support face decision
- Fist is a rounded convex blob, no flat palm. Support surface =
  **gripper +X (palm/finger-front)**, NOT knuckles.
- Reuses the existing `alex_{left,right}_palm_contact_site` (Prabin-authored in
  `create_alex_mujoco_sites_model.py`; it is NOT in the URDF).

### 3. Contact-first solver
`scripts/solve_fbx_canonical_alex_contactfirst.py` (forked from worlddelta,
format tag `alex_contactfirst_v1`, defaults to the V2 model).
- **Contact detection from human data**: marker height above clip floor + low
  speed, per foot/hand.
- **Foot-flat** (foot +Z → world +Z), **gated on the human foot actually being
  flat** (canonical foot frame local-Z within 40° of vertical). Without the gate
  it over-triggered and fought tracking; with it, planted feet hit 0–5°.
- **Hand fist support** = best-effort fist-down (palm +X → world −Z, weight 0.8)
  + **palm-site position pin** to the human hand contact location (weight 3.0),
  which **suppresses the wrist-body position target** during contact (else they
  fight). Static support (lying) lands 0°/0cm; dynamic get-up push is
  reach-limited (Alex arm shorter than human) → best-effort, expected.
- Intermediate segments (upperarm/forearm/shin) stay orientation-free; world-
  delta orientation is suppressed per-effector during its contact.
- Inherited unchanged from worlddelta: position tracking + per-role morphology
  scales, posture reg, self-collision repulsion (weight 20), per-iteration step
  cap (max_step_norm 0.20, step_scale 0.7), joint-limit clamp.
- Output NPZ adds: `contact_flags`, `contact_effector_names`,
  `contact_align_errors_deg`. Per-frame log shows `align°/pos-cm` per effector.

### 4. Renderer
`scripts/visualization/render_contactfirst.py` — V2 robot + canonical human
stick figure + **per-effector contact status strip** (green=CONTACT + angle,
grey=free). Outputs in `outputs/renders/contactfirst/`.

Solved NPZs: `outputs/contactfirst/{standup_02,shovel_fronthard_02}_contactfirst.npz`
Videos:      `outputs/renders/contactfirst/*.mp4`

---

## Next steps (priority order)

### NEXT-1: Global-OPT smoothing pass (fixes the jitter)
Trajectory-level smoothing (minimize joint velocity/accel/jerk) **with contacts
pinned** so feet/hands don't slide off their contact points. Started thinking:
`scripts/compute_globalopt_metrics.py` (untracked). Apply to the contact-first
NPZ output; keep contact frames as hard constraints, smooth free DOFs +
transitions.

### NEXT-2: Per-frame forward temporal term
Add an explicit `‖qₜ − qₜ₋₁‖` velocity penalty in the per-frame QP (causal,
cheap) to reduce jitter before the global pass.

### NEXT-3: Position-side reweighting
Down-weight intermediate-segment **position** targets; add a posture regularizer
on knees/wrists (they're loose now → flick). This is the position-side twin of
the orientation relaxation already done.

### NEXT-4: Tuning / validation
- Sweep contact thresholds (`--foot-contact-height`, `--hand-contact-height`,
  `--contact-speed`, `--foot-flat-tilt`) and the contact weights.
- The dynamic-push hand reach limit is morphology — accept it / let global-OPT
  blend the transition.

---

## Key file locations
```
V2 model build:  scripts/build_alex_v2_collision_model.py
V2 model:        assets/alex/alex_floating_base_with_sites_v2.xml
Contact-first:   scripts/solve_fbx_canonical_alex_contactfirst.py
Renderer:        scripts/visualization/render_contactfirst.py
Baseline solver: scripts/solve_fbx_canonical_alex_posori_qp_fresh_worlddelta.py
Test canonical:  outputs/canonical_human/fbx_fresh/standup_02_canonical_human_fresh_with_orient.npz
                 outputs/canonical_human/fbx_fresh/PrabinRef_Shovel_FrontHard_02_with_orient.npz
```

## Contact effector indices (contact_flags columns)
```
0: left_foot   1: right_foot   2: left_hand   3: right_hand
```

## Alex Right Arm qpos Indices
```
RIGHT_SHOULDER_Y: 29   RIGHT_SHOULDER_X: 30   RIGHT_SHOULDER_Z: 31
RIGHT_ELBOW_Y:    32   RIGHT_WRIST_Z:    33   RIGHT_WRIST_X:    34
RIGHT_GRIPPER_Z:  35
```

## Contact label indices (11 bodies — post_process_grounding_contacts.py)
```
0: LEFT_FOOT       1: RIGHT_FOOT      2: LEFT_SHIN       3: RIGHT_SHIN
4: LEFT_THIGH      5: RIGHT_THIGH     6: PELVIS_LINK     7: TORSO_LINK
8: HEAD_LINK       9: LEFT_GRIPPER_Z_LINK   10: RIGHT_GRIPPER_Z_LINK
```

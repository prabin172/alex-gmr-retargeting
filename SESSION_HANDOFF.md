# Session Handoff — Alex V2 contact-first + GlobalOPT

Branch: `feature/alex-v2-contact-first-ik`. Prabin commits himself.
Last session: 2026-07-02. **All solver/renderer work is committed (`dfa1e95`) and pushed.**
**Mentor approved the unified videos — reference motions cleared for training.** Videos shared
externally for feedback on alternative motion recordings (esp. standup-from-lying variants).

## ACTIVE branch `FullURDF` (2026-07-02, uncommitted) — fullmesh collision adoption
Prabin received the real V2 robot URDF (`assets/alex/source/alex_V2_description-new/`, copied over
`alex_V2_description/`) with LEG collision as convex-hull STL meshes instead of the old hand-tuned
primitives. Kinematics UNCHANGED (joints/axes/limits/link frames identical; only a fixed ZED camera
mount frame removed, torso collision origin nudged −0.018 z). So IK output is identical — Stage 3 is
NOT re-solved. Decision: **fullmesh is the real body; primitives were a stopgap.** On repo cleanup,
promote fullmesh to the only model, drop the primitive path, make `--soft-collision` always-on.

New/changed (all uncommitted on `FullURDF`; `assets/` is git-ignored so only script edits show in `git status`):
- **Collision model**: `assets/alex/alex_floating_base_with_sites_v2_fullmesh.xml` — v2 XML with the 8
  leg primitive geoms swapped to `type="mesh"` (convex STLs from `meshes/alex_V2_description/legs/`,
  zero offset). Arms/head already used meshes; sites/joints/inertials untouched. Primitive XML kept as fallback.
- **Render model**: `assets/alex/alex_visual_mesh_fist_hands_v2.xml` — true-V2 visual body (legs+arms
  now V2, not V1) with closed-fist hands. NOTE the OLD `alex_visual_mesh_fist_hands.xml` renders mostly
  **V1** legs/arms — the already-shared `unified/` videos show the old body.
- **Stage-B fix** (`solve_global_trajectory_opt_contactfirst.py`): fullmesh legs make the hard
  collision inequalities `primal infeasible` (row explosion: 424 vs ~80–194 rows; genuinely-close
  legs in get-ups/kneels) → Stage B silently no-op'd (|dQ|max=0). Fix: `--soft-collision` +
  `--collision-penalty 1000` (default OFF; slack var per collision row + quadratic penalty → always
  feasible, degrades gracefully). Default path byte-identical to shipped (verified).
- **Batch knobs** (`run_globalopt_all.sh`): `STAGEB_MODEL`, `STAGEB_EXTRA`, `GO_DIR`, `GR_DIR`,
  `RENDER_MESH=visualv2`. All default to prior behavior.

Fullmesh-adopted pass (17/18 clips OK; `standupFromKneeling_02` was missing Stage-1/2 input — being
generated): Stage 4 re-solved on fullmesh + soft-collision → `outputs/global_opt_contactfirst_fullurdf/`
+ `outputs/grounded_contactfirst_fullurdf/`, V2 renders → `outputs/renders/contactfirst/fullURDF/`.
Primitive NPZs in the non-suffixed dirs untouched. Command that ran it (log
`outputs/logs/fullurdf_pass_20260702_135608.log`):
```
STAGEB_MODEL=assets/alex/alex_floating_base_with_sites_v2_fullmesh.xml \
STAGEB_EXTRA="--soft-collision --collision-penalty 1000" \
GO_DIR=outputs/global_opt_contactfirst_fullurdf GR_DIR=outputs/grounded_contactfirst_fullurdf \
RENDER_MESH=visualv2 RENDER_DIR=outputs/renders/contactfirst/fullURDF \
RENDER_EXTRA="--fixed-cam --no-human" ./run_globalopt_all.sh
```
**Trade-off (measured):** fullmesh+soft eliminates self-penetration (standup_side_04 coll 32.6%→0%,
peak 5.2→0cm) at the cost of ~1cm more foot slip (get-up/kneel plant_slip 2.8–3.4cm → 4.2cm). Chosen
deliberately: penetration is physically impossible (bad for physics-RL); RL absorbs a cm of slip.

## Committed
- `c59c93a` Contact-first IK on **Alex V2** (`assets/alex/alex_floating_base_with_sites_v2.xml`,
  convex hulls). Solver: `scripts/solve_fbx_canonical_alex_contactfirst.py`.
- `439c80b` Contact-aware **GlobalOPT** (`scripts/solve_global_trajectory_opt_contactfirst.py`)
  + batch (`run_globalopt_all.sh`).
- `74d3501` Z-grounding post-step, fist-hand visual render, foot-yaw/blend/root-smooth.
- `dfa1e95` **Unified config**: shank clamp, knee bias, foot-hold ×10, onset hysteresis,
  Stage-B re-enabled, foot-yaw align, make/break blend, θ·axis foot-flat error,
  `human_target_positions` saved, `--hierarchical` (retired, off), analysis scripts.
- Design docs: `CONTACT_FIRST_SUMMARY.md`, `GLOBALOPT_CONTACTFIRST_PLAN.md`.

## Pipeline + UNIFIED config (one retargeter for all actions — Prabin's rule)
Per clip: contact-first solve → GlobalOPT **Stage A** (contact-blind smoothing,
tridiag joints + root pos/quat) → **Stage B** (contact-aware QP: feet pinned to
per-interval median anchor at weight 40, hands soft, SCA, trust region) → Z-grounding → render.
Batch: `run_globalopt_all.sh`, env knobs `LAMBDA_SMOOTH=20`, `N_OUTER=3`,
`RENDER_EXTRA`, `RENDER_DIR`. **Identical flags for every clip** — the CLIPS 3rd/4th
per-clip flag fields exist but are empty by design (experiments only).
Inputs `outputs/canonical_human/fbx_fresh/*_with_orient.npz`; 120fps, stride 4 ⇒ 30fps render.
Stages 1–2 (FBX → canonical → orient) are NOT in the batch script — run
`blender --background --python scripts/build_fbx_canonical_human.py -- --fbx ... --out ...`
then `scripts/build_canonical_orientation_frames_fresh.py` per new FBX
(see `retargetingPipeline.sh` steps 1–2; its stages 3–5 are the OLD solver — don't use).

## New clips (this session — Prabin runs the pass himself)
6 new FBX in `data/raw/inhouse/`: `standFromKnees/PrabinRef_STandupFromKneeling_01/02`,
`standFromKnees/PrabinRef_StandupKnees_02`, `crouchStand/PrabinRef_StandupSquatCrouch_01`,
`KneelingFall/PrabinRef_KneelingFall_02/03`.
- `run_globalopt_all.sh` CLIPS extended to 18 entries (names: `standupFromKneeling_01/02`,
  `standupKnees_02`, `standupSquatCrouch_01`, `kneelingFall_02/03`), empty flag fields.
- Run: stages 1–2 loop for the 6 FBX, then
  `RENDER_DIR=outputs/renders/contactfirst/unifiedRobot RENDER_EXTRA="--fixed-cam --no-human" ./run_globalopt_all.sh`
  → robot-only renders (+ contact strip) into `outputs/renders/contactfirst/unifiedRobot/`.
  Stage 3 skips the 12 old clips (NPZs exist); Stages 4/4.5 recompute all 18 (deterministic).
- Robot-only rendering needed NO code change — `render_contactfirst.py --no-human`
  already drops the human panel and keeps the contact strip at single-panel width.

## Key decisions (don't re-litigate)
- **ONE config for all actions** (Prabin). Solver defaults + GlobalOPT λ=20, n-outer 3.
- **Stage B ON everywhere** (supersedes the old all-off rule). Old blocker "plants not
  stationary" was an artifact of loose contact detection; with hysteresis + hold10 the
  solve is 0.1–0.3cm/plant → Stage B well-posed. Stage A alone re-adds ~8cm plant drift.
- **Everything is SOFT weights — no hierarchical optimization anywhere.** `--hierarchical`
  retired (regressed pivoting get-ups; root failure = promoting the reach-limited palm pin
  to hard). Stage B foot pin is `add_soft` at `--foot-weight 40` — the file DOCSTRING
  saying "hard equality" is STALE (no `add_hard` exists; fix when touching the file).
  Hard in Stage B: only joint-limit/trust-region bounds + collision inequality rows.
  Residual slip 1.0–1.5cm is a high-weight equilibrium, not zero by construction.
- **Foot-drag diagnosis**: achieved foot dragged by heavier pelvis/torso tasks, not target
  slip. Weight fixes cap out; Stage B is the principled pin.
- **flat gate stays 40°** (`--foot-flat-tilt`); λ_smooth=20 + Stage B (was 30 Stage-A-only).
- **Shank clamp** edits the KNEE target (not ankle angles): pitch clip [−25°,+55°] fwd-lean,
  roll ±20° (ankle ranges minus 5° margin), cross-faded by contact weight; plantarflexion
  side (−25° floor) is the tight side. Skipped when knee not above ankle (deep kneel) —
  relevant for the new kneeling clips.
- Alex ankle vs human: dorsiflexion 60° (human ~20°), plantarflexion 30° (human ~50°),
  roll ±25°, no ankle yaw (comes from hip), rigid foot.

## Measured (12-clip unified batch)
- Shovels: plant_slip 1.0–1.5cm, flat 0.1–0.2°, coll 0, spikes 0.
- Standups: plant_slip 2.7–3.7cm (side_05 outlier 7.9), spikes 0; crouch-phase flat
  angles 9.7–12.7° are faithful human tilt.
- Foot-flat err in contact 12.7°→7.7° mean (shovels ~0.1°), knee straight-lock 26.5%→0%.

## Outputs (git-ignored, shared via Slack + external channels)
- **Current (FullURDF): `outputs/renders/contactfirst/fullURDF/`** — V2 body + fullmesh+soft motion, 18 clips.
- Approved two-panel primitive batch (old V1-mesh render): `outputs/renders/contactfirst/unified/`.
- Earlier eras preserved: `.../foothold_fix/`, `.../onset_hysteresis/`, `.../shankclamp_kneebias/`, `.../blend/`.

## Watch items
- `kneelingFall_02/03` are FALLS — first clips with contact onset on a descending body;
  hysteresis (capped 150ms) was tuned on plants from crouches. Check knee/hand touchdowns.
- Kneeling clips likely trigger the shank-clamp deep-kneel skip (`vz < 0.2L`) — fine by
  design, but verify flat behavior while kneeling.
- `standup_side_04`: primitive Stage B self-penetration peak 6.6cm — **resolved on FullURDF**
  (fullmesh+soft drives it to 0%). `standup_side_05`: slip 7.9cm (primitive; recheck on fullmesh).
- Stage B docstring still says feet are "hard equality" — code is all soft (`add_soft`); fix on next touch.

## NEXT
1. Inspect the 18 `outputs/renders/contactfirst/fullURDF/` videos (V2 body, fullmesh+soft motion),
   esp. get-up/kneel plant behavior and the higher 4.2cm slip. `standupFromKneeling_02` Stages 1–5
   were run separately to complete the set to 18.
2. If satisfied: commit `FullURDF` (script edits; `assets/` git-ignored so models/meshes stay local).
3. Repo cleanup: promote fullmesh to the only model, delete the primitive XML + `_fullurdf` dir suffixes,
   make `--soft-collision` always-on (drop the gate).
4. Training launches on the reference motions → torque requirements from RL.
5. Then: Mimic-ready export / contact labels pipeline (Stage 4 in CLAUDE.md).

## Design stance (retargeting philosophy)
Physical feasibility > verbatim copying. Contacts + end effectors exact; body interior
approximate. Kinematic infeasibility is fixed in the TARGETS (e.g. shank clamp), not by
weight fights. Downstream physics-RL absorbs dynamics errors, not kinematic impossibilities.

## Session context (2026-07-02, non-pipeline)
- Global `~/.claude/CLAUDE.md`: added **Model delegation** (Fable 5 plans; Opus 4.8
  subagents implement; confirm plans first; subagent for >3-file tasks).
- git: dfa1e95 pushed; only `prompt.md` + `.obsidian/` untracked (suggest .gitignore).

# alex-gmr-retargeting

Retarget human motion-capture (FBX) onto the IHMC **Alex** humanoid, contact-first, in MuJoCo.
A human clip is converted to a canonical skeleton, solved onto Alex with per-frame QP IK that keeps
feet/hands planted on their real contacts, then globally smoothed and grounded into a physics-RL-ready
`qpos` trajectory. Kinematic retargeting only — no dynamics; downstream RL supplies torques.

For the full method and all the mathematics, see **[METHOD.md](METHOD.md)**.

## What you get after cloning

The code and library are here. **The robot assets, input data, and all outputs are NOT in git** — they
are large and/or internal (see `.gitignore`). A fresh clone has the pipeline but no model and no data,
so it will not run until you populate the local-only directories below.

```
general_motion_retargeting/   library: morphology scaling, canonical source adapter, robot configs (in git)
scripts/                      pipeline stages 1–5 (in git)
scripts/legacy/               retired code — old worlddelta solver family, MVNX path, old renders (in git, not run)
scripts/visualization/        renderers (in git)
METHOD.md                     full method + math (in git)
CLAUDE.md                     repo conventions / agent instructions (in git)

assets/alex/                  ROBOT MODEL — git-ignored, local only
data/                         INPUT FBX + canonical NPZs — git-ignored, local only
outputs/                      ALL generated NPZs, renders, logs — git-ignored, local only
SESSION_HANDOFF.md            current working state & decisions — git-ignored, kept on disk
paper_idea.md                 submission draft — git-ignored, kept on disk
```

### Git-ignored layout you must provide locally

`assets/alex/` (only `README.md`/`.gitkeep` are tracked) — obtain from a teammate; it must contain:
```
assets/alex/alex_floating_base_with_sites.xml     canonical model: 36-DOF, convex-hull collision (SOLVER default)
assets/alex/alex_visual_mesh_fist_hands.xml       render body: full visual mesh, closed-fist hands
assets/alex/meshes/alex_V2_description/            collision + visual meshes (legs/, cycloidal_arm/, head, torso...)
assets/alex/source/alex_V2_description/            the V2 URDF description these are built from
```
`data/raw/inhouse/<action>/*.fbx` — the source mocap clips.
`outputs/` — created by the pipeline; nothing to provide.

> These assets are **hand-maintained**, not regenerated from the `prepare_*`/`create_*`/`build_*`
> model-prep scripts (those are historical and would overwrite the current model — don't run them blindly).

## Environment

Python ≥3.10 in a conda env (referred to as `gmr`), plus **Blender** (4.x) for stage 1 only.
Core deps: `mujoco`, `numpy`, `scipy`, `osqp`, `imageio` (+ ffmpeg). MuJoCo rendering uses EGL
(`MUJOCO_GL=egl`). Install the package in editable mode: `pip install -e .`

## Running the pipeline

Stages 1–2 are per-FBX (Blender + orientation frames); stages 3–5 are the batch.

**Stages 1–2** — FBX → canonical positions → semantic orientation frames (once per new clip):
```bash
blender --background --python scripts/build_fbx_canonical_human.py -- \
  --fbx data/raw/inhouse/<action>/<clip>.fbx \
  --out outputs/canonical_human/fbx_fresh/<clip>.npz
python scripts/build_canonical_orientation_frames_fresh.py \
  --in-npz  outputs/canonical_human/fbx_fresh/<clip>.npz \
  --out-npz outputs/canonical_human/fbx_fresh/<clip>_with_orient.npz
```

**Stages 3–5** — contact-first IK → GlobalOPT → grounding → render, for every clip in the CLIPS list:
```bash
./retargetingPipeline.sh
```
The Stage-4 solver defaults to the single canonical model with always-on soft self-collision — no
model/flag knobs needed. Useful env overrides (all optional):
`LAMBDA_SMOOTH=20`, `N_OUTER=3`, `GROUND_MODE=perframe`, `RENDER_MESH=visual|collision|<path>`,
`RENDER_DIR=...`, `GO_DIR=...`, `GR_DIR=...`, `RENDER_EXTRA="--fixed-cam --no-human"`.

Outputs per clip: `outputs/contactfirst/<clip>_contactfirst.npz`,
`outputs/global_opt_contactfirst/<clip>_global_opt.npz`,
`outputs/grounded_contactfirst/<clip>_grounded.npz`,
`outputs/renders/contactfirst/<clip>_globalopt.mp4`.

## Conventions

+X forward / +Y left / +Z up; quaternions **wxyz**; free-root `qpos = [x,y,z,qw,qx,qy,qz, 29 joints]`.
Morphology scaling applies to motion *deltas* from rest pose, never to the absolute root. Details and
the 15 canonical roles + 4 contact sites are in `CLAUDE.md`; the math is in `METHOD.md`.

## Status / limitations

Kinematics only (no dynamics). Self-penetration is eliminated by the soft self-collision Stage B at the
cost of ~4.2cm foot slip on hard get-up/kneel plants (deliberate — RL absorbs a little slip, not body
interpenetration). The Mimic-ready contact-labels export is not yet wired on the contact-first path
(the old `scripts/legacy/post_process_grounding_contacts.py` produced (N,11) labels for the retired pipeline).

## Branches

`main` — current contact-first + fullmesh line. `initialBaseline` — the earlier posori/worlddelta baseline
(tag `baseline-posori-worlddelta-v1`) kept for reference. `feature/fbx-kinematic-canonical-v2` — a parallel
solver with different segment assumptions, kept for exploration.

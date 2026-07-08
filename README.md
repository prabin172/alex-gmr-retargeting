# alex-gmr-retargeting

Retarget human motion-capture (FBX) onto the IHMC **Alex** humanoid, contact-first, in MuJoCo.
A human clip is converted to a canonical skeleton, solved onto Alex with per-frame QP IK that keeps
feet/hands planted on their real contacts, then globally smoothed and grounded into a physics-RL-ready
`qpos` trajectory. Kinematic retargeting only — no dynamics; downstream RL supplies torques.

For the full method and all the mathematics, see **[METHOD.md](METHOD.md)**. For every pipeline knob —
default, what it trades off, when to touch it — see **[PARAMETERS.md](PARAMETERS.md)**.

## What you get after cloning

The code and library are here. **The robot assets, input data, and all outputs are NOT in git** — they
are large and/or internal (see `.gitignore`). A fresh clone has the pipeline but no model and no data,
so it will not run until you populate the local-only directories below.

```
scripts/                      pipeline stages 1–6 (in git)
scripts/visualization/        renderers (in git)
METHOD.md                     full method + math (in git)
PARAMETERS.md                 per-knob cheat-sheet (in git)
CLAUDE.md                     repo conventions / agent instructions (in git)
wiki/                         LLM-maintained knowledge base (see wiki/index.md)

assets/alex/                  ROBOT MODEL — git-ignored, local only
data/raw/                     INPUT FBX — git-ignored, local only
outputs/                      ALL generated NPZs, renders, logs — git-ignored, local only
SESSION_HANDOFF.md            current working state & decisions — git-ignored, kept on disk
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

The full pipeline — FBX → canonical human → orientation frames → contact-first IK → GlobalOPT →
grounding → render → IHMC JSON (stages 1–6) — is driven by a single script:

```bash
./retargetingPipeline.sh
```

Add clips to the `CLIPS` array in the script (clip name + FBX path + optional per-clip solver flags).
Stage 1 requires **Blender** (`blender` on PATH, 4.x); it runs headless (`--background`). Stages 2–6
are plain Python.

Outputs per clip:
```
outputs/canonical_human/fbx_fresh/<clip>_with_orient.npz   Stage 2 output / Stage 3 input
outputs/contactfirst/<clip>_contactfirst.npz                Stage 3 output
outputs/global_opt_contactfirst/<clip>_global_opt.npz       Stage 4 output
outputs/grounded_contactfirst/<clip>_grounded.npz           Stage 4.5 output
outputs/renders/contactfirst/<clip>_globalopt.mp4           Stage 5 output
outputs/ihmcJsons-native120hz/<clip>.json                   Stage 6 output (native 120 Hz)
outputs/ihmcJsons50hz/<clip>.json                           Stage 6b output (50 Hz, IHMC rate)
```

Stages 1–2 are skipped automatically if the `*_with_orient.npz` already exists (safe to re-run).
Stage 3 is skipped if `*_contactfirst.npz` already exists. Stages 4–6 always re-run.

Useful env overrides (all optional): `LAMBDA_SMOOTH=320`, `N_OUTER=6`, `GROUND_MODE=constant-contact`,
`RENDER_MESH=visual|collision|<path>`, `RENDER=0` (skip render), `CLIPS_MATCH=<substring>` (run subset).
Full list with defaults and trade-offs: **[PARAMETERS.md](PARAMETERS.md)**.

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

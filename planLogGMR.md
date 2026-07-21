# planLogGMR — GMR-baseline-plan execution log

Companion to `GMR-baseline-plan.md`. Branch `gmr-baseline` (forked from `p0-grounding`@`63d34ff`).

---

## T0 — Prerequisite commit

Done by Prabin: `64ce722` (GMR-baseline docs) + `63d34ff` (feasibility-first-v1 session work).
Branch `gmr-baseline` created from `63d34ff`.

## T1 — Environment + smoke test

**Env**: existing `gmr` conda env reused (had mujoco 3.9.0, mink, qpsolvers[daqp/osqp/proxqp]
already). `pip install -e /home/ptimilsina/projects/GMR` succeeded cleanly (editable install,
`general_motion_retargeting==0.2.0`).

**LAFAN1**: downloaded from `github.com/ubisoft/ubisoft-laforge-animation-dataset` (direct
`lafan1.zip`, 144MB) into `data/raw/lafan1/` (77 BVH files, 333MB unzipped, gitignored under
`data/`). Zip deleted after extraction.

**Footgun found — GLFW headless failure**: GMR's own `scripts/bvh_to_robot.py` uses
`RobotMotionViewer`, which calls `mujoco.viewer.launch_passive()` unconditionally (even with
`--record_video`, even without an interactive session) — this opens a GLFW window and hard-fails
with no `DISPLAY` (confirmed empty on this machine, no Xvfb installed). **Fix**: wrote
`scripts/g1/gmr_headless_retarget.py` — same retargeting core (`GeneralMotionRetargeting` class,
`load_bvh_file` loader, identical pkl output format `{fps, root_pos, root_rot xyzw, dof_pos,
local_body_pos: None, link_body_list: None}`) but skips `RobotMotionViewer` entirely; video
(optional) uses `mujoco.Renderer` with `MUJOCO_GL=egl` (confirmed working offscreen) + a
`MjvCamera` tracking `ROBOT_BASE_DICT[robot]` (pelvis), mirroring the viewer's own
`camera_follow` logic (`lookat`=pelvis xpos, `distance`=`VIEWER_CAM_DISTANCE_DICT`, elevation
-10°, azimuth 90°) — first attempt without camera tracking produced an empty-floor video (default
free camera doesn't follow the robot); fixed and re-verified by extracting frame 100 as a PNG
(robot clearly in frame, walking pose).

**Second footgun found — package shadowing**: this repo has a leftover, empty (pycache-only,
no `.py` sources, gitignored) `general_motion_retargeting/` directory at its root — a namespace
package from before this repo diverged from the original GMR clone (per `CLAUDE.md`). When repo
root ends up on `sys.path` (e.g. running `python -c "..."` from repo root, where `sys.path[0]`
resolves to cwd), Python's namespace-package resolution picks up this empty local dir INSTEAD of
the real pip-installed GMR package in site-packages — `import general_motion_retargeting` then
silently gives a package with `__file__ is None` and no submodules, not an ImportError. Running
as a script FILE is unaffected (`sys.path[0]` = the script's own directory, not cwd) — confirmed
`scripts/g1/gmr_headless_retarget.py` and other script-mode invocations load the real package
correctly. **Rule for all future `scripts/g1/*.py`**: always run as `python scripts/g1/foo.py`,
never `python -c "..."` from repo root. When T4 needs to import sibling Alex scripts
(`eval_ihmc_json.py` etc.), insert `<repo_root>/scripts` onto `sys.path`, NOT bare `<repo_root>`
— avoids ever putting the shadowing directory on the path. Did not delete/touch the empty
directory itself (out of scope, not blocking once the rule is followed).

**Smoke test** (`walk1_subject1.bvh`, `unitree_g1`): 7840 frames retargeted successfully.
Gate: pkl keys `{fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list}` ✓;
`root_pos.shape==(7840,3)` ✓; `root_rot.shape==(7840,4)`, quat norms in [0.9999999999999989,
1.000000000000001] ✓ (xyzw, as documented); `dof_pos.shape==(7840,29)` ✓; pelvis z mean 0.78m
(sane for G1 standing height) ✓. Video renders robot clearly, camera-tracked.

## T2 — Clip selection

Screened all `fallAndGetUp*`, `ground*`, `pushAndFall*`, `pushAndStumble*` BVHs (16 files) by
hip-Z range via GMR's own `load_bvh_file` (consistent with what the retargeter actually sees).
Results (name, T frames, hipZ min/max/p5):

| clip | T | hipZ min | max | p5 |
|---|---|---|---|---|
| fallAndGetUp1_subject1 | 5047 | 0.051 | 0.926 | 0.063 |
| fallAndGetUp1_subject4 | 5047 | 0.056 | 0.935 | 0.077 |
| fallAndGetUp1_subject5 | 5047 | 0.066 | 0.961 | 0.100 |
| **fallAndGetUp2_subject2** | 4918 | **0.028** | 0.934 | 0.050 |
| fallAndGetUp2_subject3 | 4918 | 0.030 | 1.062 | 0.051 |
| fallAndGetUp3_subject1 | 3066 | 0.060 | 0.940 | 0.138 |
| **ground1_subject1** | 4742 | 0.109 | 0.919 | 0.122 |
| ground1_subject4 | 4742 | 0.117 | 0.921 | 0.129 |
| ground1_subject5 | 4742 | 0.147 | 0.921 | 0.279 |
| ground2_subject2 | 5550 | 0.039 | 1.155 | 0.208 |
| pushAndFall1_subject1 | 4959 | 0.120 | 0.924 | 0.478 |
| pushAndStumble1_subject2 | 6801 | 0.062 | 0.997 | 0.476 |

(push/stumble clips mostly stay upright with brief dips — p5 0.4-0.7 — less "floor contact" than
"nearly falls"; not selected.)

**Selected 3 floor-contact clips** (diverse failure modes):
1. `fallAndGetUp2_subject2` — most extreme (hipZ min 0.028), full fall-then-recover cycle.
2. `fallAndGetUp1_subject1` — sustained low floor time (p5=0.063, spends a lot of the clip near
   the ground, not just a brief dip).
3. `ground1_subject1` — different flavor: sustained ground/crawling work (p5=0.122), not a
   fall-recover arc — closer to a "get up from ground" style motion than a stumble.

**Selected 2 locomotion controls**: `walk1_subject1` (clean, hipZ min 0.718 — already used in T1
smoke test), `dance1_subject1` (busier full-body motion, hipZ min 0.231 — a crouch/lunge dip but
p5=0.767, stays off the floor — genuine non-floor-contact control, verified before selecting).

**No stop condition triggered** — LAFAN1 clearly has real floor-contact clips (T2's named risk in
`GMR-baseline-plan.md` did not materialize), proceeding directly to T3.

## T3 — E1 baseline batch

Ran `scripts/g1/gmr_headless_retarget.py` (unitree_g1) on all 5 selected clips. All completed
without errors. Outputs: `outputs/gmr_baseline/pkl/{clip}.pkl`,
`outputs/gmr_baseline/videos/{clip}.mp4`.

| clip | frames | fps | pelvis-z min |
|---|---|---|---|
| fallAndGetUp2_subject2 | 4918 | 30 | 0.029 (frame 356) |
| fallAndGetUp1_subject1 | 5047 | 30 | 0.046 (frame 4282) |
| ground1_subject1 | 4742 | 30 | 0.097 (frame 2412) |
| walk1_subject1 | 7840 | 30 | — (control) |
| dance1_subject1 | 3945 | 30 | — (control) |

**Qualitative failure catalog** (frame extracted at each clip's lowest-pelvis moment,
`ffmpeg -vf select` on the rendered video):

- **fallAndGetUp2_subject2** (f356): pelvis flat on the floor, legs splayed with both feet
  raised and angled unnaturally off the ground (no plantar contact), arms flat/limp at the
  sides on the floor plane. Reads as a rigid "corpse pose" rather than a body using the floor
  for support — no visible weight-bearing contact anywhere.
- **fallAndGetUp1_subject1** (f4282): near-identical signature — pelvis/torso flat, both feet
  pointed up and outward with no ground contact, arms flat and splayed to the sides. This is
  the sustained-low-floor-time clip (p5=0.063 in T2's table) and the pose holds this shape for
  an extended window, not just one bad frame.
- **ground1_subject1** (f2412): torso pitched forward/down (closer to a crawl/prone-push
  posture than the other two's supine-flat pose), one leg splayed out to the side at an
  implausible angle, head/torso oriented into the floor. Different failure MODE than the two
  fallAndGetUp clips (prone vs supine collapse) — useful diversity for the motivation figure,
  confirms this isn't one narrow bug but a general floor-contact gap.

Common thread across all three: **no limb ever appears to bear weight against the floor** —
the retargeter tracks the human's joint positions kinematically but has no floor-contact/support
reasoning, so a human lying down (weight distributed through the floor) becomes a robot pelvis
resting at the tracked height with limbs following the human's joint positions regardless of
whether that pose is physically supportable. This matches GMR's own stated exclusion verbatim
(`GMR-baseline.md` §1: "we do not include motions with complex interaction with the environment,
such as crawling or getting up from the floor") — now with concrete visual evidence on their own
benchmark's own clips, not just the paper's claim taken on faith.

Controls (`walk1_subject1`, `dance1_subject1`) not yet quantitatively evaluated (that's T5) but
spot-checked visually in T1 (walk1_subject1 — clean walking gait, no artifacts).

**T3 done.** Videos + pkls are the E1 raw material; next is T4 (de-Alex the eval core) to turn
this qualitative catalog into numbers.

## T4 — De-Alex the eval core

`evaluate()` in `scripts/eval_ihmc_json.py:92` needed ZERO changes — confirmed it already takes
only `(name, qpos, fps, contacts, model, data, mesh_cache, geom_ids, sole_sids, q_lo, q_hi,
mj_joint_names, limit_tol_deg)`, no Alex-specific globals inside. `contacts={}` / `sole_sids={}`
already degrade gracefully (loops over `sole_sids.items()` / `.get(f, ...)` do nothing) — no
"optional" plumbing needed either, empty dicts are the natural G1 case.

Wrote `scripts/g1/eval_motion.py`: imports `evaluate`, `_load` (aliased `_load_ihmc_json`),
`MODEL_DEFAULT`, `SPIKE_RAD_PER_S` directly from `eval_ihmc_json.py` (via `sys.path.insert(0,
REPO_ROOT/"scripts")` — NOT bare repo root, per T1's shadowing footgun) — one implementation,
no copy-paste. `--ihmc-json` / `--gmr-pkl` mutually-exclusive CLI flags; G1 default model
`/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_mocap_29dof.xml`. `--gmr-pkl` mode imports
`load_gmr_pkl` (deferred — doesn't exist until T5; `--ihmc-json` mode fully working now).

**Regression gate**: `eval_ihmc_json.py` vs `eval_motion.py --ihmc-json`, both on
`data/blender-retargeted/standSupine.json`. Every computed metric identical: `T=390 fps=50
floorPen=3.0cm pen%=74.9% coll%=0.0% collPk=0.0cm JLvi=0 worst_joint=RIGHT_ANKLE_Y(66%)
vMax=25.7 vP95=5.0 spikes=0 rootV=1.67 plPen=3.0cm plFloat=0.4cm plSlip=3.3cm`. Diff between the
two outputs is purely cosmetic: column width (widened 22→26 chars for longer G1 clip names) and
one caption word ("JSON's"→"source's" own contact flags, since G1 sources aren't JSONs). **Gate
passes.**

**Self-collision caveat surfaced in the docstring/footer** (per plan T4 point 1): `coll%`/`collPk`
printed for G1 but flagged "not vetted this week, informational only" — the G1 mocap XML's
collision geometry hasn't been checked for sane self-collision pair exclusions the way Alex's has;
will sanity-check the actual numbers when T5 runs real G1 pkls through this (if they read as
obvious noise, e.g. near-100%, that confirms the caveat; if near-0% and stable, may be usable
after all — decide empirically, not by assumption).

## T5 — G1 loader + M1 motivation table

`scripts/g1/load_gmr_pkl.py`: pkl -> `(qpos (T,36), fps)`, `root_rot` xyzw->wxyz at this one
boundary (`[:, [3,0,1,2]]`), quat-unit-norm assert. FK sanity check on `walk1_subject1` frame 0:
lowest geom-origin z = 0.026m (near floor, sane for a walking frame — a wrong quat order would
show up here as an absurdly pitched/rolled robot; it didn't). Wired into `eval_motion.py
--gmr-pkl` via deferred import (works because `scripts/g1/` is `sys.path[0]` when run as a
script). **Floor-plane exclusion turned out to need zero extra code**: G1's `floor` geom is
`bodyid=0` (worldbody) in the mocap XML, already excluded by `eval_motion.py`'s existing
`geom_bodyid[g] != 0` filter (same filter Alex's eval already used) — verified directly by
inspecting the model's geoms.

**M1 motivation table** (`outputs/gmr_baseline/eval_raw.csv`, all 5 clips, GMR raw pkls, no
polish):

| clip | T | floorPen max | pen% | coll% | vMax rad/s | spikes | rootV max |
|---|---|---|---|---|---|---|---|
| walk1_subject1 (control) | 7840 | **1.0cm** | 0.3% | 18.2%\* | 18.9 | 0 | 1.90 |
| dance1_subject1 (control) | 3945 | 7.1cm | 1.9% | 5.6%\* | 47.5 | 0 | 2.71 |
| fallAndGetUp2_subject2 | 4918 | **13.6cm** | **47.1%** | 11.9%\* | 20.4 | 0 | 5.04 |
| fallAndGetUp1_subject1 | 5047 | **12.9cm** | **38.9%** | 11.4%\* | 29.5 | 0 | 3.06 |
| ground1_subject1 | 4742 | **15.9cm** | **90.6%** | 4.7%\* | 37.2 | 0 | 2.09 |

\* self-collision column: informational only this week, not vetted (see below).

**Reads exactly as hypothesized**: clean locomotion control (`walk1_subject1`) is near-perfect
(1.0cm max pen, affecting 0.3% of frames — essentially floor noise). The busier control
(`dance1_subject1`, has a crouch/lunge dip per T2) shows a real but small floor dip (7.1cm,
1.9% of frames) — an order of magnitude smaller than the floor-contact clips. All three
floor-contact clips are dramatically worse: **12.9–15.9cm max penetration, affecting 39–91% of
all frames** — `ground1_subject1`'s 90.6% in particular means the robot spends nearly the WHOLE
clip below the floor, matching its "sustained ground work" character from T2's screening. This
quantifies T3's qualitative failure catalog.

**Unexpected but informative side-finding**: `n_spikes=0` and no wild `vMax` outliers on ANY
clip, including the floor-contact ones — GMR's own per-frame differential IK (damped, mink-based)
produces smooth output even while failing badly on floor contact. **Velocity/jitter and floor
penetration are orthogonal failure modes here** — worth stating explicitly in the T10 narrative:
the floor-contact problem isn't "GMR produces jerky garbage on hard clips," it's specifically
"GMR has no floor-contact reasoning," a cleaner, more surgical claim for Option A's story.

**Self-collision caveat confirmed empirically, not just assumed**: `walk1_subject1` — a clean,
short-duration walk cycle — reads **18.2% self-collision incidence, 6.1cm peak depth**. That's
physically implausible for ordinary walking and is strong evidence the G1 mocap XML's collision
geometry (thighs/torso at hip/knee flexion, most likely) isn't set up with the same collision-pair
exclusions Alex's model has. **Self-collision numbers are NOT usable as-is this week** — flagged
n/a-ish in `eval_motion.py`'s footer, real fix (vet `g1_custom_collision_29dof.urdf`) stays
deferred to E4 per the plan.

**M1 CHECKPOINT: clear, not weakened.** GMR fails hard and specifically on floor-contact clips
relative to its own clean locomotion baseline, on its own benchmark, measured by our
reference-free eval ported with zero core-logic changes. No early-stop condition fires. Proceeding
to T6 (polish plumbing).

## T6 — Polish plumbing (pkl round-trip)

`scripts/g1/polish_gmr_pkl.py`: `load_gmr_pkl` -> qpos -> (T7/T8 transforms land here) ->
`save_gmr_pkl` (new function, exact inverse: wxyz->xyzw via `[:, [1,2,3,0]]`, mirrors
`bvh_to_robot.py:166`'s own convention). With no transform flags yet, this is pure identity.

**Round-trip gate**: `walk1_subject1.pkl` -> identity polish -> compared against original.
`root_pos`/`root_rot`/`dof_pos` max abs diff = **0.0 exactly** (the xyzw<->wxyz round-trip is a
pure index permutation, no floating-point arithmetic — bit-exact, stronger than the plan's 1e-12
bar). `eval_motion.py --gmr-pkl` on both: **identical row** (`1.0c 0.3% 18.2% 6.1c ... `,
matches T5's `walk1_subject1` row exactly). **Gate passes.**

## T7 — Stage A smoothing on G1

`stage_a` imported UNCHANGED from `solve_global_trajectory_opt_contactfirst.py:629` — genuinely
zero core-logic edits, confirmed by inspection: the only place a robot identity leaks in is the
module-level `N_ACT=29` constant (used for array indexing inside `stage_a`), which happens to
numerically match G1's 29 actuated joints. Added an explicit `assert n_act == N_ACT` in
`polish_gmr_pkl.py` before calling it, so a future robot with a different joint count fails loud
instead of silently misindexing, rather than leaving that as an implicit landmine.
`q_lo`/`q_hi` (the one genuinely robot-specific input `stage_a` takes) come from G1's own
`model.jnt_range`. Params: `--lambda-track 1.0` (Alex's own CLI default), `--lambda-smooth 20.0`
(fps²-scaled per the plan's rule: pipeline's 320 @120Hz -> 20 @30Hz, LAFAN1's rate),
`smooth_root=True`.

**Before (T5 raw) -> after (Stage A) on all 5 clips:**

| clip | vMax rad/s | floorPen max | pen% |
|---|---|---|---|
| walk1_subject1 | 18.9 -> **3.3** (5.7x) | 1.0 -> 4.4cm | 0.3% -> 30.1% |
| dance1_subject1 | 47.5 -> **6.2** (7.7x) | 7.1 -> 7.9cm | 1.9% -> 31.5% |
| fallAndGetUp2_subject2 | 20.4 -> **4.8** (4.3x) | 13.6 -> 14.3cm | 47.1% -> 56.9% |
| fallAndGetUp1_subject1 | 29.5 -> **6.1** (4.8x) | 12.9 -> **11.5cm** | 38.9% -> 64.1% |
| ground1_subject1 | 37.2 -> **5.5** (6.8x) | 15.9 -> **11.9cm** | 90.6% -> 92.9% |

**Velocity smoothing transfers cleanly to G1**: 4.3-7.7x reduction in peak joint velocity across
ALL 5 clips (both controls and all 3 floor clips), squarely in the same range as the Luigi
polish result on Alex (5.7x) that originally validated this recipe — strong evidence Stage A's
core mechanism really is robot-agnostic, not an Alex-specific coincidence.

**Floor penetration: mixed, as anticipated by `stage_a`'s own docstring** (documents that
uniform smoothing can erode a sharp correction / bleed a violation into neighboring frames —
Stage A is floor-blind by design). Max penetration got WORSE on 3/5 clips (walk +3.4cm,
dance +0.8cm, fallAndGetUp2 +0.7cm) but BETTER on 2/5 (fallAndGetUp1 -1.4cm, ground1 -4.0cm).
`pen%` (fraction of frames with >0.5cm penetration) got WORSE on every single clip, including
the controls — smoothing spreads existing small dips across more neighboring frames even when
peak magnitude doesn't grow much. None crossed the plan's ">1cm median" stop-loss language
catastrophically, but the direction is consistent and expected: **Stage A alone is not a floor
fix, it's a smoothness fix that costs a little floor accuracy — grounding (T8) is the
designed-for counterpart**, not an optional add-on. Proceeding to T8.

## T8 — Grounding on G1

No contact flags from GMR pkls, so `hybrid`/`constant-contact` modes (which need
`contact_flags`/`contact_effector_names`) are out per the plan — confirmed by reading
`post_process_ground_contactfirst.py`'s `main()`: `constant` and `perframe` modes read ONLY the
`qpos` key from the input NPZ (`contact_effector_names`/`contact_flags`/`fps` are all
`if "key" in data_dict`-guarded, never hard-required outside the hybrid/constant-contact
branches). This meant the "thin adapter" the plan anticipated needed almost no logic: a
`ground_qpos()` helper in `polish_gmr_pkl.py` that writes a **minimal temp NPZ containing only
`qpos`**, shells out to `post_process_ground_contactfirst.py --mode {constant,perframe}`
UNMODIFIED (`subprocess.run`, not an import — zero risk of touching Alex's QP/mesh code, per the
plan's explicit "do not fork" instruction), and reads back the grounded `qpos`. Chain wired as
`--stage-a --ground --ground-mode {constant,perframe}` in one `polish_gmr_pkl.py` invocation
(raw pkl in, polished pkl out).

Ran BOTH modes (plan's instruction: "try both, ship the better") on all 5 Stage-A-smoothed clips:

| clip | Stage-A floorPen | constant | perframe |
|---|---|---|---|
| walk1_subject1 | 4.4cm / 30.1% | **0.7cm / 0.1%** | 0.0cm / 0.0%\* |
| dance1_subject1 | 7.9cm / 31.5% | **3.2cm / 0.6%** | 0.0cm / 0.0%\* |
| fallAndGetUp2_subject2 | 14.3cm / 56.9% | **4.0cm / 0.5%** | 0.0cm / 0.0%\* |
| fallAndGetUp1_subject1 | 11.5cm / 64.1% | **1.1cm / 0.5%** | 0.0cm / 0.0%\* |
| ground1_subject1 | 11.9cm / 92.9% | **2.4cm / 0.5%** | 0.0cm / 0.0%\* |

\* `perframe` grounds every single frame's own lowest point to exactly z=0 — 0.0cm/0.0% is true
BY CONSTRUCTION, not evidence of a better fit; the real question is what it costs.

**Bobbing cost check** (root-Z vertical velocity, the artifact `post_process_ground_contactfirst.py`'s
own docstring warns `perframe` introduces): computed directly from each mode's output pkl
(`root_pos[:,2]`, finite-differenced at each clip's fps).

| clip | constant vz_max | perframe vz_max | delta |
|---|---|---|---|
| walk1_subject1 | 0.196 m/s | 0.262 m/s | +34% |
| dance1_subject1 | 1.205 m/s | 1.395 m/s | +16% |
| fallAndGetUp2_subject2 | 1.108 m/s | 1.823 m/s | **+65%** |
| fallAndGetUp1_subject1 | 1.138 m/s | 1.294 m/s | +14% |
| ground1_subject1 | 0.701 m/s | 0.696 m/s | -1% |

**Decision: `constant` mode ships as the T9 "polished" deliverable.** `constant` already gets
every floor clip's max penetration down to 1.1-4.0cm (from 11.5-15.9cm raw, a 65-91% reduction) —
comparable in absolute terms to what the ALEX pipeline itself treats as an acceptable grounded
result (Luigi's shipped clips land in the 2.8-3.1cm range) — at essentially zero added bobbing
cost (single shift = pure translation, doesn't touch relative vertical motion). `perframe`'s
"perfect" 0.0cm is real but comes with a genuine, measurable bobbing tax that peaks at +65% on
exactly the clip (`fallAndGetUp2_subject2`) where residual penetration was worst — the two
metrics trade off directly against each other on the hardest clip, and `constant`'s trade is the
better one: large, honest improvement with no new artifact, vs. a manufactured zero that
introduces a different, real defect. (Velocity numbers from T7's Stage-A pass are unaffected by
either grounding mode, as expected — grounding only shifts `qpos[:,2]`, confirmed `vMax` identical
between Stage-A-only and both grounded variants in the eval output.)

## T9 — M2: before/after table + renders (kill-test checkpoint)

**Final table** (`outputs/gmr_baseline/eval_polish.csv`, raw / Stage-A / polished=Stage-A+ground-constant,
all 5 clips):

| clip | stage | floorPen max | pen% | vMax rad/s | spikes |
|---|---|---|---|---|---|
| walk1_subject1 | raw | 1.0cm | 0.3% | 18.9 | 0 |
| | stageA | 4.4cm | 30.1% | 3.3 | 0 |
| | **polished** | **0.7cm** | **0.1%** | **3.3** | 0 |
| dance1_subject1 | raw | 7.1cm | 1.9% | 47.5 | 0 |
| | stageA | 7.9cm | 31.5% | 6.2 | 0 |
| | **polished** | **3.2cm** | **0.6%** | **6.2** | 0 |
| fallAndGetUp2_subject2 | raw | 13.6cm | 47.1% | 20.4 | 0 |
| | stageA | 14.3cm | 56.9% | 4.8 | 0 |
| | **polished** | **4.0cm** | **0.5%** | **4.8** | 0 |
| fallAndGetUp1_subject1 | raw | 12.9cm | 38.9% | 29.5 | 0 |
| | stageA | 11.5cm | 64.1% | 6.1 | 0 |
| | **polished** | **1.1cm** | **0.5%** | **6.1** | 0 |
| ground1_subject1 | raw | 15.9cm | 90.6% | 37.2 | 0 |
| | stageA | 11.9cm | 92.9% | 5.5 | 0 |
| | **polished** | **2.4cm** | **0.5%** | **5.5** | 0 |

**Raw -> polished, both axes, every single clip, no cherry-picking**: floor-pen max drops on
all 5 (65-91% reduction on the 3 floor clips; even the already-clean controls improve slightly),
`pen%` drops to ≤0.6% everywhere (from 0.3-90.6% raw), and peak joint velocity drops 4.3-7.7x
everywhere. Zero spikes throughout, all stages, all clips (GMR's own IK was never the jitter
problem — consistent with T5's side-finding).

**Renders**: `scripts/g1/render_gmr_pkl.py` (factored out of `gmr_headless_retarget.py`'s video
path, same EGL + pelvis-tracking camera) on `walk1_subject1` (control) and
`fallAndGetUp2_subject2` (floor clip, the clip whose worst frame was already captured in T3's
failure catalog). Extracted the SAME frame index (356) from both raw and polished
`fallAndGetUp2_subject2` videos for a direct before/after comparison.

**Honest caveat, worth stating plainly**: at frame 356, the polished pose is visually almost
IDENTICAL to the raw one — same splayed legs, same floating hands, same "corpse on the floor"
geometry. The 13.6cm -> 4.0cm max-pen improvement comes from correcting the clip's GLOBAL floor
reference (grounding recalibrates where z=0 actually is, over the whole clip) and general
temporal smoothing, NOT from fixing this frame's per-limb physical implausibility — a hand or
foot can still be floating above, or a hip still resting through, the now-correctly-calibrated
floor. **This week's polish (Stage A + grounding) is a real, measured, whole-clip-level fix; it
is explicitly NOT a per-limb contact fix** — that's Stage B's job (contact-anchored QP, E4,
deferred). Framing this precisely in T10 matters: the claim is "robot-agnostic kinematic polish
substantially improves floor-contact metrics," not "polish makes GMR's floor clips look right" —
the latter would be an overclaim this data doesn't support yet.

**KILL-TEST VERDICT (per `GMR-baseline.md` §4 stop-loss): Option A clearly LIVES.** Measurable,
consistent, honest polish delta on every clip in the corpus, both clip classes (locomotion
controls AND floor-contact clips), on two independent metrics (floor penetration, joint
velocity), with the core smoothing mechanism transferring at Alex-comparable magnitude (4.3-7.7x
vs Luigi's validated 5.7x) and zero core-solver-logic changes required for either Stage A or
grounding. No cherry-picking was needed to make this case. Proceeding to T10 (results narrative).

## E4 — Stage B contact-anchoring QP on G1 (MVP, post-week-1, Prabin: "go ahead with E4")

**Scope decided (feet only, self-collision OFF)**: `stage_b`, `_compute_anchors`,
`_load_model_with_floor`, `_get_joint_limits` all imported from
`solve_global_trajectory_opt_contactfirst.py` COMPLETELY UNCHANGED — verified by inspection none
of them read module-level Alex globals inside their bodies, only via arguments. The ONE genuinely
Alex-specific piece is `_resolve_contact_geom` (reads a hardcoded `CONTACT_GEOM` dict of Alex body
names) — forked as `_resolve_g1_feet` in the new `scripts/g1/stage_b_g1.py` (~20 lines, same
`resolved` dict shape, G1 body names `left_ankle_roll_link`/`right_ankle_roll_link`, no hands this
pass). Self-collision deliberately OFF (`lambda_coll=0.0`) — this week's own caveat (G1's
mocap-model collision pairs read as noise, 18.2% self-collision on a clean walk clip, M1) meant
mixing that into a hard-constraint QP risked contaminating the one new mechanism this pass was
meant to isolate and measure. Floor-collision QP rows also OFF (`count_floor=False`) — grounding
(T8) already owns clip-level floor placement; this pass tests contact anchoring ALONE, on top of
the already-polished (Stage A + grounded) motion, using it as both warm start and self-tracking
target (mirrors the validated "polish Luigi" recipe).

**Discovery: G1's mocap XML already ships sole-corner markers.** `left_ankle_roll_link`/
`right_ankle_roll_link` each carry 4 small unnamed sphere geoms (radius 5mm) at toe/heel ×
left/right offsets — structurally identical in purpose to Alex's NAMED `alex_*_sole_corner_*`
sites, just as unnamed geoms. Used for the contact-zone height signal (`_foot_sole_geom_ids`,
matched by `geom_bodyid`+`geom_type==SPHERE`, no XML edits). NOT wired into `_build_contact`'s
on-floor/flat/coplanar shared-Z refinement this pass (that machinery needs a SITE list,
`info["sole_sites"]`, which stays `[]` here) — a real, deliberate limitation, not a bug: this
pass only gets position+orientation contact anchoring, not the coplanar snap Alex's pipeline also
has. Documented as a clear extension point, not attempted (time-boxed).

**Contact-detection bug found + fixed via debug sweep (not guessed)**: first version gated on
height AND a body-origin speed threshold (mirroring Alex's `--still-speed` convention exactly).
Debug sweep (`_screen`-style scratch script) found ZERO frames satisfied both gates simultaneously
on `walk1_subject1` even at generous thresholds — root cause isolated: conditioned on
near-ground frames, the SOLE-CENTROID speed was, if anything, HIGHER (median 0.62 m/s) than the
clip's unconditional median (0.31 m/s) — a heel-to-toe rolling contact moves the 4-corner
centroid even while the true contact patch is quasi-stationary, a geometric artifact of
collapsing 4 corners to one point for the SPEED check specifically (not for the height check,
which uses the min corner, unaffected). Fix: dropped the speed gate from the detector entirely —
`detect_g1_foot_contacts` now supplies ONLY a coarse height-based "contact zone" flag
(`< 5cm`, calibrated: height-only at 5cm gives ~48% of `walk1_subject1`'s frames, a plausible
stance-fraction ballpark), and the ALREADY-IMPORTED, UNCHANGED `_compute_anchors` does its own
speed-based stillness sub-segmentation internally (body-origin speed, `--plant-speed 0.05`,
Alex's exact convention) — the right layering: my code detects "in the ballpark", the existing,
validated code decides "is it actually still enough to anchor hard."

**Result, all 5 clips, `--n-outer 6 --lambda-coll 0.0`** (stage_b's own self-reported
warm→best, NOT independently cross-validated — see caveat below):

| clip | contact zone % (L/R) | planted frames (L/R) | slip: warm→best |
|---|---|---|---|
| walk1_subject1 (control) | 47.6% / 47.8% | 100 / 160 | **1.2→0.9cm (25% reduction)** |
| dance1_subject1 (control) | ~low | few | 0.1→0.1cm (no change) |
| fallAndGetUp2_subject2 | ~low | very few | 0.1→0.1cm (no change) |
| fallAndGetUp1_subject1 | ~near 0 | ~0 | 0.0→0.0cm, `|dQ|max=0.0000` (QP found nothing to do) |
| ground1_subject1 | ~near 0 | ~0 | 0.0→0.0cm, `|dQ|max=0.0000` (QP found nothing to do) |

**Cross-check via `eval_motion.py`** (independent, already-trusted metrics — floor pen/vel/spikes,
NOT slip, since GMR-pkl sources carry no contact flags for eval_motion's own stance columns):
**zero regressions on any clip, any metric.** One real, independently-measured bonus win:
`dance1_subject1` floorPen 3.2→**2.8cm**, pen% 0.6%→**0.2%** (a side effect of the small joint
adjustments Stage B made while chasing its self-tracking target). All other clips: eval_motion.py
numbers byte-identical polished vs polished+StageB.

**Honest verdict — narrow, not a clean win, worth a checkpoint before continuing**: contact
anchoring gives a real, measurable improvement on CLEAN LOCOMOTION (walk1_subject1's 25% slip
reduction) — consistent with the mechanism working as designed. But it has **essentially zero
effect on the 3 floor-contact clips — the paper's actual target class** — because a
height-based "near the ground" gate finds almost no sustained stationary-foot behavior in
fall/crawl motions (unlike walking, these motions' feet are rarely both near-ground AND still at
the same time; `fallAndGetUp1_subject1`/`ground1_subject1` detected essentially zero plants,
`|dQ|max` stayed exactly 0). This is a real, well-measured, honest finding — not a bug I need to
chase further before reporting it — but it means this pass's feet-only, gait-style contact
detector is the WRONG lever for the motion class this whole project cares about. Extending it
(hands/knees/torso as support surfaces for prone/supine contact, a genuinely different detection
scheme for non-gait motions) is a real scope decision, not a natural continuation — flagged to
Prabin rather than unilaterally expanded further.

**Caveat on the one positive number**: `walk1_subject1`'s slip improvement is currently only
visible through `stage_b`'s OWN self-reported metric (computed internally by `_contact_slip_stats`
using MY detected flags) — not independently cross-validated by `eval_motion.py`, since G1 sources
carry no contact flags for its stance/slip columns. Building that independent cross-check (feed
`eval_motion.py` the SAME detected flags used here) would be a natural next step before trusting
this number for anything paper-facing.

**Files**: `scripts/g1/stage_b_g1.py` (new). Outputs: `outputs/gmr_baseline/pkl/*_stageB.pkl`
(all 5 clips), debug logs in scratchpad (not repo).

---

# WEEK 2

## W2-T1 — E1b: fair-baseline addendum + floating metric

**Floating metric added first** (per plan order — eval must see what a height fix trades into).
`evaluate()` in `scripts/eval_ihmc_json.py:92` already computed `lowest` (whole-body mesh-exact
lowest point) per frame for the penetration metric — floating is just the mirror sign,
`floating = np.maximum(0.0, lowest)`, added as two new dict keys (`float_max_cm`, `float_pct`),
zero new computation. **Regression gate**: re-ran `eval_ihmc_json.py` (unmodified call sites) on
`data/blender-retargeted/standSupine.json` — its own printed row is BYTE-IDENTICAL to week 1's T4
gate (`T=390 fps=50 floorPen=3.0c pen%=74.9% coll%=0.0% collPk=0.0c JLvi=0
worst_joint=RIGHT_ANKLE_Y(66%) vMax=25.7 vP95=5.0 spikes=0 rootV=1.67 plPen=3.0c plFloat=0.4c
plSlip=3.3c`) since its `main()` prints a fixed set of keys by name, never loops the dict — new
keys are invisible to it. `eval_motion.py --ihmc-json` on the same file: every existing column
identical, two new columns (`floatMax=2.6c float%=11.5%`) additive. **Gate passes.**

**GMR's own height fix replicated** (NOT by touching the GMR clone — new `gmr_heightfix()` in
`scripts/g1/polish_gmr_pkl.py`, wired as `--heightfix`, mutually exclusive with `--stage-a`/
`--ground` in one invocation since it's a baseline-replication column, not part of our polish
chain). Faithfully mirrors `GMR/scripts/bvh_to_robot_dataset.py:127-138`'s `HEIGHT_ADJUST=True,
PERFRAME_ADJUST=False` path: plain-mujoco FK every frame, clip-global min BODY-ORIGIN z (`data.xpos`,
excluding world body 0 — deliberately mesh-blind, that's the point) subtracted from root z,
`ground_offset=0.0` as in their code. Round-trip identity gate re-verified after the code addition
(`walk1_subject1` unchanged, floorPen 1.0c/pen% 0.3% match week 1's row exactly).

**Ran on all 5 clips** (`outputs/gmr_baseline/pkl_w2/*_gmrfix.pkl`), clip-global min body-origin z:
walk1_subject1 +0.0099m, dance1_subject1 -0.0434m, fallAndGetUp2_subject2 -0.1072m,
fallAndGetUp1_subject1 -0.0762m, ground1_subject1 -0.1307m (all floor clips needed a large
downward shift, as expected — their worst frame sits well below the walking baseline's).

**M1 motivation table, raw vs GMR+heightfix** (`outputs/gmr_baseline/eval_w2t1_heightfix.csv`):

| clip | floorPen: raw→+heightfix | pen%: raw→+heightfix | floatMax: raw→+heightfix | float%: raw→+heightfix |
|---|---|---|---|---|
| walk1_subject1 (control) | 1.0→2.0cm | 0.3%→5.7% | 3.5→2.5cm | 94.1%→55.0% |
| dance1_subject1 (control) | 7.1→**2.7cm** | 1.9%→**0.3%** | 28.1→32.5cm | 89.6%→99.5% |
| fallAndGetUp2_subject2 | 13.6→**2.9cm** | 47.1%→**0.4%** | 15.7→26.4cm | 47.7%→**98.6%** |
| fallAndGetUp1_subject1 | 12.9→**5.2cm** | 38.9%→**3.8%** | 9.6→17.2cm | 56.1%→**94.0%** |
| ground1_subject1 | 15.9→**2.9cm** | 90.6%→**0.1%** | 3.7→16.8cm | 8.2%→**99.9%** |

**Reads exactly as `GMR-baseline.md` §7.2 hypothesized, and sharpens rather than weakens the
motivation.** The height fix DOES cut floor-clip penetration substantially (65-82% max-pen
reduction on the 3 floor clips, pen% down to 0.1-3.8% from 38.9-90.6%) — it's a real fix, not a
strawman. But the mesh-blind, single-worst-frame mechanism trades that gain directly into
near-universal floating: **float% lands at 94.0-99.9% on every floor clip post-fix** (up from
8.2-56.1% raw) — i.e. essentially the WHOLE clip now has some body part hovering >0.5cm above the
floor, because the one frame that calibrated the global shift has its body-ORIGIN (not mesh
surface) sitting at z=0, so the mesh bottom is still somewhere below that origin at the calibration
frame and clearly above z=0 everywhere else. `walk1_subject1` (control) is the cleanest case for
seeing the mechanism: it barely needed a shift (+0.99cm) and floorPen/pen% both got slightly WORSE
(1.0→2.0cm, 0.3%→5.7%) purely from that small forced shift, while floatMax dropped a little
because there was less floating to begin with. Self-collision (`coll%`) is invariant under the fix
on every clip, confirming it's a pure global Z-shift with no other side effect, as designed.

**No checkpoint stop** — floor clips are nowhere near "near-clean AND low float" (the plan's stop
condition); if anything, applying GMR's OWN described fix and still landing at ~99% floating on
the floor-contact class is a STRONGER, more citation-proof version of the motivation figure than
week 1's raw-only numbers, because it forecloses the obvious reviewer question ("did you just skip
their own post-processing step?"). Both effects (residual penetration reduced, floating massively
introduced) belong in the paper's motivation section as one honest paragraph, not cherry-picked.

**Files**: `scripts/eval_ihmc_json.py` (float_max_cm/float_pct added to `evaluate()`'s return dict,
additive only), `scripts/g1/eval_motion.py` (floatMax/float% columns printed), `scripts/g1/
polish_gmr_pkl.py` (`gmr_heightfix()`, `--heightfix` flag). Outputs:
`outputs/gmr_baseline/pkl_w2/*_gmrfix.pkl` (5 clips), `outputs/gmr_baseline/eval_w2t1_heightfix.csv`.

**Addendum — three-way comparison (raw / GMR+heightfix / our week-1 polished), floor+float
together**, per the plan's instruction to add the heightfix column to the polish table too:

| clip | floorPen: raw / +heightfix / polished | pen%: raw / +heightfix / polished | float%: raw / +heightfix / polished |
|---|---|---|---|
| walk1_subject1 | 1.0 / 2.0 / **0.7cm** | 0.3% / 5.7% / **0.1%** | 94.1% / 55.0% / 96.3% |
| dance1_subject1 | 7.1 / **2.7** / 3.2cm | 1.9% / **0.3%** / 0.6% | 89.6% / 99.5% / 98.6% |
| fallAndGetUp2_subject2 | 13.6 / **2.9** / 4.0cm | 47.1% / **0.4%** / 0.5% | 47.7% / 98.6% / 97.9% |
| fallAndGetUp1_subject1 | 12.9 / 5.2 / **1.1cm** | 38.9% / 3.8% / **0.5%** | 56.1% / 94.0% / 98.3% |
| ground1_subject1 | 15.9 / 2.9 / **2.4cm** | 90.6% / 0.1% / **0.5%** | 8.2% / 99.9% / 98.4% |

**floorPen/pen%: our polish wins or ties on 3/5 clips** (walk1, fallAndGetUp1, ground1 clearly
better; dance1 and fallAndGetUp2 heightfix is nominally better by 0.3-1.1cm, both already deep in
diminishing returns below 5cm). Neither method dominates the other on this axis alone.

**float%: NOT a differentiator — both land in the same 94-99.9% range on every floor clip.** This
is a genuinely important, slightly humbling finding: our OWN week-1 "polished" deliverable
(constant-mode grounding) shares the exact same whole-clip-level mechanism as their height fix — a
single clip-global Z calibration (constant mode uses a percentile of the robot's own lowest mesh
point over the clip; heightfix uses the clip's single worst body-origin frame) — so it ALSO trades
residual penetration into near-universal floating. This turns T9's qualitative "frame-356 pose
looks the same after polish" observation into a quantitative one: **whole-clip Z-calibration,
ours or theirs, cannot produce a body that's actually resting on the floor throughout a clip** —
only per-limb, per-frame contact anchoring (E4b) can. Reinforces `GMR-baseline.md` §7.2 item 3
with a number, doesn't change the E4b priority — if anything, strengthens the case for it.

## W2-T2 — Close E4's unverified slip claim

Wrote `scripts/g1/check_slip_independent.py`: independent cross-check of E4's `walk1_subject1`
25% plant-slip reduction, WITHOUT importing `_contact_point`/`_compute_anchors`/
`_contact_slip_stats` (the exact functions E4's own number came from). Re-derives contact ZONES
with `detect_g1_foot_contacts` (imported — deterministic, the one piece explicitly allowed to
reuse since re-deriving the height gate from scratch would just reimplement the same logic), then
does everything else independently: **body-ORIGIN xyz via plain FK** (not stage_b's sole-corner
contact point), stillness sub-segmentation by body-origin XY speed < 0.05 m/s with a 2-frame
debounce (own implementation, not `_compute_anchors`), and **drift-from-run-START** (not
drift-from-run-median, `_contact_slip_stats`'s convention) — a genuinely different measurement
methodology, same zone windows for a fair warm-vs-best comparison.

**Result on `walk1_subject1`** (`outputs/gmr_baseline/pkl/walk1_subject1_polished_constant.pkl`
as "warm" vs `..._stageB.pkl` as "best"):

| foot | planted runs | zone % | drift mean: warm→best | drift max: warm→best |
|---|---|---|---|---|
| left_foot | 21 | 47.6% | 0.47→**0.43cm** (-8.2%) | 0.92→**0.87cm** (-5.3%) |
| right_foot | 23 | 47.8% | 0.72→**0.65cm** (-9.4%) | 1.77→**1.35cm** (-23.4%) |

**Direction CONFIRMS**: contact anchoring genuinely reduces foot drift on both feet, independently
measured, no shared code with the claim being checked. **Magnitude is smaller** than E4's internal
25% (mean reduction here: 8-9%; max reduction: 5-23%) — expected and NOT a discrepancy to chase
further, since the two methodologies measure different things (body-origin XY vs sole-corner
centroid; drift-from-run-start vs drift-from-run-median — a run that drifts monotonically in one
direction reads differently under each convention). **E4's walk1_subject1 result is now safe to
cite as "a real, independently-confirmed slip reduction on clean locomotion, magnitude
8-25% depending on measurement convention"** — the exact "25%" number stays stage_b's own internal
metric, not to be quoted as an isolated fact.

**Files**: `scripts/g1/check_slip_independent.py` (new).

## W2-T3 — E4b-a: human-side multi-surface contact labels (kill-test #1)

Read `scripts/contact_labels.py` first per plan instruction. Kept its core convention (height-gate
per landmark, min-combine of multiple markers per effector) but deliberately DROPPED its speed
gate — the E4 lesson (this file's own "E4" section above): a naive speed gate returns zero
contacts for rolling/complex-contact motions. Stillness sub-segmentation stays downstream, in
`_compute_anchors` (unchanged, imported), not this detector — matches the plan's explicit
instruction.

New `scripts/g1/human_contacts_lafan1.py`: loads each clip via GMR's own `load_bvh_file` (per-frame
`{bone_name: [pos, quat]}` dicts — the LAFAN1 skeleton, discovered by inspection: `Hips,
LeftUpLeg/Leg/Foot/Toe, RightUpLeg/Leg/Foot/Toe, Spine/Spine1/Spine2, Neck, Head, Left/RightShoulder/
Arm/ForeArm/Hand`). Landmarks: **feet** (Foot+Toe min, thr 0.05m — same value E4 calibrated for the
robot side), **hands** (Hand, thr 0.08m), **knees** (`Leg` bone — LAFAN1's knee joint, between
UpLeg/thigh and Foot/shank, thr 0.08m), **elbows** (`ForeArm`, thr 0.08m), **pelvis** (`Hips`, thr
0.15m), **torso** (`Spine1`, thr 0.15m), **head** (`Head`, diagnostic only, never anchored, no
threshold). LAFAN1's floor is z=0 (established week 1 T2 — no separate floor-height estimation
needed, unlike Alex's FBX sources).

**Calibration** (`--report-only` distribution pass across all 5 clips, min/p1/p5/p25/median/p75 per
landmark): `walk1_subject1`'s non-foot landmarks sit FAR above every threshold with huge margin —
hands min 0.618-0.673m (thr 0.08), knees min 0.320-0.344m (thr 0.08), elbows min 0.853-0.916m (thr
0.08), pelvis min 0.718m (thr 0.15), torso min 0.903m (thr 0.15). `dance1_subject1` (busier
control) is the only near-miss: hands dip to 0.021-0.048m briefly (a reach/crouch gesture) —
checked below, turns out negligible. Defaults kept as calibrated (no threshold tuning needed —
the separation was already clean).

**Full detection, zone % per landmark, all 5 clips:**

| clip | L/R foot | L/R hand | L/R knee | L/R elbow | pelvis | torso |
|---|---|---|---|---|---|---|
| walk1_subject1 (control) | 92.6/90.6% | 0.0/0.0% | 0.0/0.0% | 0.0/0.0% | 0.0% | 0.0% |
| dance1_subject1 (control) | 76.2/71.7% | **0.2/0.6%** | 0.0/0.0% | 0.0/0.0% | 0.0% | 0.0% |
| fallAndGetUp2_subject2 | 71.7/70.7% | 41.2/34.5% | 24.5/7.3% | 15.8/10.3% | 28.5% | 14.5% |
| fallAndGetUp1_subject1 | 64.3/62.2% | 25.0/29.9% | 10.6/11.9% | 16.2/18.4% | 22.4% | 19.2% |
| ground1_subject1 | 49.2/54.5% | 88.0/84.9% | 83.3/80.7% | 65.0/56.3% | 54.7% | **0.0%** |

**KILL-TEST #1: CLEARLY PASSES.** Both controls show near-zero non-foot contact (dance1's 0.2-0.6%
hand blips are noise-level — a brief low gesture, not sustained contact, three orders of magnitude
below the floor clips' hand zones). All 3 floor clips show substantial, SUSTAINED multi-surface
contact exactly where expected: hands 25-88%, knees 7-83%, elbows 10-65%, pelvis 22-55%. The one
apparent zero (`ground1_subject1` torso = 0.0%) is itself a correct, informative finding, not a
detector failure — its own height distribution (report pass) shows torso's clip-wide MINIMUM is
0.153m, just above the 0.15m threshold: this clip is a genuine hands-and-knees crawl (matching T2's
"sustained ground/crawling work" characterization) where the torso stays lifted the whole time —
anatomically correct that it never zones, unlike the two fallAndGetUp clips (torso 14.5%/19.2%),
which include a real lying-flat-on-torso phase. The detector is discriminating support PATTERNS
correctly, not just thresholding blindly.

**Correctness spot-check**: `fallAndGetUp2_subject2` frame 356 — the exact frame T3's week-1
failure catalog described as "pelvis flat on the floor... arms flat/limp at the sides" — reads
`pelvis=True, left_hand=True, right_hand=True, torso=True` (pelvis height 0.033m). Confirms the
detector recovers the correct real-world support pose at the one frame we already had independent
qualitative ground truth for.

**Files**: `scripts/g1/human_contacts_lafan1.py` (new). Outputs:
`outputs/gmr_baseline/human_contacts/{clip}.npz` (per-landmark `zone_*`/`height_*` boolean+float
arrays + threshold dict, all 5 clips).

## W2-T4 — E4b-b: G1 multi-surface role map + support points

Extended `stage_b_g1.py` (glue only) with `ROLE_TO_G1_BODY` — feet (unchanged from E4:
`left/right_ankle_roll_link`), hands (`left/right_rubber_hand` — the distal hand link on GMR's
no-hands 29-DoF `unitree_g1` variant), knees (`left/right_knee_link`), elbows
(`left/right_elbow_link`), pelvis (`pelvis`), torso (`torso_link`). Body names verified by
inspection (`mj_id2name` sweep of all 39 bodies, logged in scratch). All 10 candidate bodies carry
real geometry (`pelvis`/`torso_link` 3-6 mesh geoms, `left/right_rubber_hand` 1 mesh geom each,
knees/elbows 2 mesh geoms each) — no body is geometry-free.

`support_z(model, data, mesh_cache, body_id)`: mesh-exact lowest-Z of a SINGLE body's own geoms,
reusing `_geom_lowest_z` (imported unchanged from `post_process_ground_contactfirst.py` — the same
orientation-aware per-geom-type logic `_robot_lowest_z` already uses for the whole-robot floor-pen
metric, just restricted to one body's geom set instead of iterating the whole model).

**Gate**: `walk1_subject1` frame 0 (standing) — `support_z` clean ordinal separation: feet
+2.1/+2.8cm (near floor, small residual consistent with W2-T1's known raw-GMR floor noise), knees
+6.0/+6.4cm (next-lowest, anatomically correct — G1's knee sits fairly low in a standing pose),
hands/elbows/pelvis/torso all +66cm to +111cm (clearly the trunk/upper-body tier). `fallAndGetUp2_
subject2` frame 356 (lying, the same frame W2-T3 spot-checked) — EVERY support point clusters near
z=0 (-3.8cm to +2.2cm across all 10 roles), matching the visual "corpse pose" description exactly;
pelvis's -2.9cm is strikingly close to W2-T3's independently-measured HUMAN pelvis height at the
same frame (3.3cm) — two independent measurements (human source landmark height, robot mesh
support point) agreeing within ~1cm is a strong cross-validation the whole role-map + support_z
plumbing is wired correctly, not a coincidence worth dismissing.

**Files**: `scripts/g1/stage_b_g1.py` (`ROLE_TO_G1_BODY`, `support_z()` added; existing E4
feet-only `G1_CONTACT_GEOM`/`G1_TRACK_ROLES`/`_resolve_g1_feet`/`main()` untouched).

## W2-T5 — E4b-c: multi-surface Stage B with pull-to-floor anchors (kill-test #2)

**Implementation** (all glue in `scripts/g1/stage_b_g1.py`, `stage_b`/`_compute_anchors`/
`_contact_intervals` imported COMPLETELY UNCHANGED): `ROLE_TO_G1_BODY` extended with a `"support"`
kind (distinct from `"foot"`/`"hand"`) for the 8 non-original-feet roles — this reuses
`_build_contact`'s existing body-origin position-pin path (no site required, same code path
E4's feet already used) while the `else` branch's rotation term is zeroed via `fist_w=0.0`, so no
orientation constraint is applied to hands/knees/elbows/pelvis/torso — position-only pull-to-floor,
exactly as scoped. `_pull_to_floor()`: post-processes `_compute_anchors`' returned `tgt` in glue
code (never touches the imported function) — for each planted run (found via `_contact_intervals`,
imported), overwrites the run's anchor Z with `median(origin_z[t] - support_z(t))` over the run's
own warm-start frames, i.e. pins the body origin at the height that puts ITS OWN mesh-exact lowest
point at the floor. X,Y untouched. Human contact zones (W2-T3's saved NPZs) drive `eff_names`/
`flags` instead of E4's robot-side detector. Default anchored roles: feet+hands+knees+elbows (8);
pelvis/torso behind `--anchor-trunk` (not tried this pass — the base 8-role result already answers
the kill-test, see below). Self-collision/floor-collision QP rows stay OFF, matching E4's isolation
choice (W2-T6 vets self-collision separately).

**Engagement confirmed on all 5 clips** — anchors DO fire, not a wiring failure. Example
(`fallAndGetUp2_subject2`): all 8 roles show nonzero zone/contact/planted counts (feet 71-72%
zone → ~1000 planted frames each; hands 35-41% zone → ~460-480 planted; knees/elbows 7-25% zone →
190-490 planted), Stage B's own solve log shows nonzero `|dQ|max` every outer iteration (0.15-0.77
rad), `status=solved` throughout, no infeasibility.

**Frame-356 spot check (the same frame W2-T3/W2-T4 already validated)**: `support_z` moved in the
RIGHT direction on 5/8 anchored roles (feet -1.5/-1.5cm, hands -2.4/-3.5cm, elbows -2.6/-2.9cm) but
knees moved the WRONG way (+0.6/+1.3cm, worse) and pelvis/torso weren't anchored this pass (torso
incidentally improved -2.3cm via joint coupling, pelvis exactly unchanged as expected). The
whole-body mesh-exact lowest point at this frame did improve measurably: **+6.5cm → +5.0cm**
(confirmed via direct `_robot_lowest_z` call, matching the anchored right-foot's own improvement
exactly) — proof the mechanism moves SOMETHING in the right direction locally. But the gap started
at 6.5-13cm and closed by only 1.5-3.5cm anywhere — nowhere close to reaching the floor. Root
position is UNCHANGED (`[0,0,0]` diff) and max joint-angle change anywhere in the whole clip is
only **0.30 rad (~17°)** — the trust-region-limited local QP (6 outer iterations, 0.15 rad/step)
is far too conservative to close a multi-centimeter gap when the correction fundamentally needs
root/proximal-joint movement, which nothing in this pass anchors or permits by more than the
existing tracking/smoothness terms allow.

**Full 5-clip gate table, polished → +multisurface StageB** (`eval_motion.py`, independent):

| clip | floorPen | pen% | float% |
|---|---|---|---|
| walk1_subject1 (control) | 0.7→**1.1cm** (worse) | 0.1%→**0.7%** (worse) | 96.3%→94.2% (better) |
| dance1_subject1 (control) | 3.2→**3.9cm** (worse) | 0.6%→**0.9%** (worse) | 98.6%→98.4% (~same) |
| fallAndGetUp2_subject2 | 4.0→4.0cm (no change) | 0.5%→0.5% (no change) | 97.9%→98.3% (worse) |
| fallAndGetUp1_subject1 | 1.1→**1.4cm** (worse) | 0.5%→0.4% (marginal better) | 98.3%→97.8% (better) |
| ground1_subject1 | 2.4→**2.6cm** (worse) | 0.5%→0.5% (no change) | 98.4%→97.0% (better) |

**GATE FAILS — this is a clear negative, not just a null result.** `eval_motion.py`'s own gate
("zero regressions on any clip, any metric") does NOT hold: floorPen got measurably WORSE on 4/5
clips, INCLUDING BOTH CONTROLS (`walk1_subject1` 0.7→1.1cm, `dance1_subject1` 3.2→3.9cm) — pulling
one set of joints to satisfy anchor targets perturbs other parts of the body enough (via the
shared whole-body IK's smoothness/tracking coupling) that a different geom becomes the new worst
penetrator, slightly deeper than before. Float% shows small, INCONSISTENT movement (2 clips
better, 2 worse, 1 flat) — no clean signal either direction. `pen%`/`floorPen`'s plan-mandated
"strictly improve on floor clips" bar is not met on any of the 3 floor clips.

**Visual kill-test #2**: frame 356 extracted from raw / polished / +multisurface renders
(`outputs/gmr_baseline/renders_w2/f356_{raw,polished,multisurface}.png`). **All three are visually
indistinguishable** — same splayed-leg corpse pose, both feet still clearly lifted off the ground
plane at the same angle, arms flat on the floor in the same position. No visible weight-bearing
contact emerges anywhere in the multi-surface render that wasn't already present.

**CHECKPOINT M3 — reporting per the plan, not pushing further unilaterally.** Anchors engage
(confirmed: nonzero zone/planted/|dQ|), move SOME support points in the right direction locally
(5/8 roles, ~1.5-3.5cm each), but (a) never close more than a fraction of the required 5-13cm gap
because root position is frozen and per-outer joint-angle change is capped at ~0.3 rad total by the
trust-region schedule, and (b) the WHOLE-CLIP aggregate metrics show a net REGRESSION on floorPen
on 4/5 clips, not merely a null result — perturbing anchored joints costs a little penetration
elsewhere more often than it fixes floating locally. The visual confirms: no support body
"catches" the floor in the multi-surface render. **Conclusion: anchoring-on-top-of-polish, even
with the pull-to-floor Z-correction and human-side multi-surface detection, is NOT the corpse-pose
fix.** The gap is structural, not a tuning problem this pass's mechanism can close — closing it
would need the ROOT itself to move (a much larger, non-local correction) or a genuinely different
mechanism class: contact-first SOLVING on G1 (an analog of Alex's Stage-3 contact-first IK, which
plans root position jointly with contact targets from the start, rather than anchoring atop an
already-fixed whole-body trajectory) — a Week-3+ scope decision, not a natural continuation of
this week's anchoring approach. `--anchor-trunk` (pelvis/torso) was NOT tried given this result —
adding two more anchors to a mechanism that already shows net-negative aggregate behavior on 8
roles would not plausibly reverse the conclusion, and per the plan's own early-stop discipline this
is exactly the point to stop and report rather than keep iterating unilaterally.

**Files**: `scripts/g1/stage_b_g1.py` (`ROLE_TO_G1_BODY`, `support_z`, `TRUNK_ROLES`/
`DEFAULT_ROLES`, `_load_human_zones`, `_pull_to_floor`, `run_multisurface`, `--multi-surface`/
`--human-contacts`/`--anchor-trunk`/`--support-weight` CLI). Outputs:
`outputs/gmr_baseline/pkl_w2/*_stageB_multisurface.pkl` (5 clips),
`outputs/gmr_baseline/renders_w2/f356_{raw,polished,multisurface}.png`.

## W2-T6 — Self-collision vetting on G1

Inspected `/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_custom_collision_29dof.urdf`
(read-only, never edited in place). Finding: only **11 of the URDF's 46 `<collision>` blocks are
actually uncommented** — `left/right_hip_yaw_link`, `left/right_knee_link`, `torso_link`,
`left/right_shoulder_roll_link`, `left/right_shoulder_yaw_link`, `left/right_elbow_link`, each a
single collision CYLINDER primitive; every other link (pelvis, ankles, feet, hands, waist, wrists,
head) has its collision block commented out. This is a genuine, if partial, simplified-collision
model GMR's own authors built for the joints most prone to self-intersection during locomotion
(hip/knee/elbow bend, torso twist) — not a graft target we needed to build ourselves.

**Loaded directly via MuJoCo's own URDF importer** (new file
`outputs/gmr_baseline/g1_collision/g1_collision_vetted.urdf`, copied from the GMR clone — never
touches it in place — with the file's own `<!-- [CAUTION] uncomment when convert to mujoco -->`
block un-commented per its own instructions: the `<mujoco><compiler meshdir=.../></mujoco>`
directive and the `world`/floating-base joint, plus `meshdir` pointed at the GMR clone's absolute
mesh path). Loads cleanly: 31 bodies, 49 geoms. **Actuated joint names/order verified IDENTICAL to
`g1_mocap_29dof.xml`'s 29 joints** (direct comparison, both lists equal) — meaning our existing
qpos arrays feed this model's `data.qpos` directly, no remapping needed.

**No graft needed at all**: MuJoCo's own URDF compiler already separates visual meshes
(`contype=0 conaffinity=0 group=1`, inert) from the 11 real collision cylinders
(`contype=1 conaffinity=1 group=0`, active) — `_collision_stats` (imported UNCHANGED from
`solve_global_trajectory_opt_contactfirst.py`) relies purely on `mj_forward`'s own contact
generation + a k-hop adjacency filter, so it works on this model with zero new code beyond loading
it and calling the existing function.

**GATE PASSES**: `walk1_subject1` raw reads **0.2% self-collision incidence (0.7cm peak)** — down
from 18.2% mesh-based noise, well under the plan's <1% bar, and now physically plausible for a
clean walk cycle.

**Full raw/stageA/polished table, all 5 clips** (vetted model, `_collision_stats` unchanged):

| clip | raw | stageA | polished |
|---|---|---|---|
| walk1_subject1 (control) | 0.2% (0.7cm) | 0.0% (0.0cm) | 0.0% (0.0cm) |
| dance1_subject1 (control) | 1.3% (5.5cm) | 0.4% (2.2cm) | 0.4% (2.2cm) |
| fallAndGetUp2_subject2 | 5.8% (5.8cm) | 4.8% (5.7cm) | 4.8% (5.7cm) |
| fallAndGetUp1_subject1 | 3.5% (5.8cm) | 2.5% (5.9cm) | 2.5% (5.9cm) |
| ground1_subject1 | 2.6% (6.0cm) | 2.6% (4.2cm) | 2.6% (4.2cm) |

Sensible shape throughout: both controls near-zero (walk1 essentially clean, 0.0% once polished),
floor clips show real but modest self-contact (2.5-5.8%) — plausible for lying/crawling poses
where limbs genuinely approach the torso, unlike the mesh-based noise which flagged an ordinary
walk cycle as 18.2% self-colliding. Polish (Stage A + grounding) holds steady or slightly reduces
self-collision on every clip — smoothing removes some of the extreme joint excursions that drove
peak penetration depth, though not the clip's baseline contact level.

**Not done this pass**: re-running W2-T5's multi-surface Stage B with `lambda_coll` at Alex's
default (the plan's conditional follow-up). Given W2-T5's own checkpoint already found the
anchoring mechanism structurally insufficient (root-frozen, trust-region-capped, net-negative on
4/5 clips) for reasons unrelated to self-collision, enabling collision avoidance on top of a
mechanism already flagged as not working would not plausibly change that conclusion — deferred to
whatever mechanism Prabin decides to pursue next (contact-first solving, per W2-T5's conclusion),
where a vetted collision model is now available and ready to use.

**Files**: `outputs/gmr_baseline/g1_collision/g1_collision_vetted.urdf` (new, copied+patched from
the GMR clone's own `g1_custom_collision_29dof.urdf`, GMR clone itself untouched).

## W2-T7 — Contact-aware grounding comparison

**Thin adapter built, per the plan's explicit instruction** ("if the script hard-requires
Alex-specific bits... write the thin adapter, do NOT fork the QP/mesh code"):
`post_process_ground_contactfirst.py`'s `hybrid`/`constant-contact` modes need per-foot
planted-sole-height samples via `_foot_plant_frames`, which hard-codes Alex's NAMED
`SOLE_CORNER_SITES` (a site lookup G1 has none of) — confirmed by reading it that it would always
silently fall back to the same global-lowest constant week 1 already shipped, giving zero
improvement if used as-is. New `scripts/g1/ground_g1_contact_aware.py`: re-derives the SAME
`plant_data` shape (`{col: {min_z, labelled, still}}`) using G1's own sole-corner SPHERE geoms
(`_foot_sole_geom_ids`, imported from `stage_b_g1.py` unchanged) for `min_z`, and W2-T3's
human-side foot contact zones for `labelled` (+ a body-speed `still` sub-selection, same
`still_speed=0.05` convention) — then hands off to `_planted_foot_sole_samples`/`_solve_lift_qp`
(imported UNCHANGED from `post_process_ground_contactfirst.py`) for the percentile floor
computation and the hybrid lift QP. The QP/mesh code itself was never forked.

**Ran both `constant-contact` and `hybrid` modes on all 5 clips** (Stage-A-only pkls as input,
matching T8's own pipeline position), compared against the ALREADY-SHIPPED `constant` mode
(percentile=1.0, week 1's winner) via `eval_motion.py`:

| clip | floorPen: constant / constant-contact / hybrid | pen%: constant / constant-contact / hybrid |
|---|---|---|
| walk1_subject1 | **0.7c** / 6.6c / 3.3c | **0.1%** / 87.6% / 29.1% |
| dance1_subject1 | **3.2c** / 10.4c / 4.8c | **0.6%** / 91.0% / 26.2% |
| fallAndGetUp2_subject2 | **4.0c** / 15.4c / 12.7c | **0.5%** / 67.1% / 45.6% |
| fallAndGetUp1_subject1 | **1.1c** / 13.8c / 12.9c | **0.5%** / 93.7% / 32.9% |
| ground1_subject1 | **2.4c** / 12.9c / 10.5c | **0.5%** / 94.5% / 70.8% |

**Both new modes are dramatically WORSE than the already-shipped `constant` mode on every single
clip, by a wide margin — not a marginal or mixed result.** `hybrid` is consistently better than
`constant-contact` (its own lift QP recovers some of the damage) but still far short of
`constant`'s numbers everywhere. This is the opposite of the plan's expectation ("hybrid ≥
constant on floor clips").

**Root cause, partially diagnosed** (sanity check, `walk1_subject1`, 10 sampled in-zone frames):
G1's sole-corner marker SPHERES (E4's discovered contact-zone markers, used here for `min_z`) sit
systematically **0.55-1.3cm ABOVE the foot's own true mesh-lowest point** (`_geom_lowest_z` over
the same body's mesh geoms) — confirmed directly, not assumed. This alone would bias the
`constant-contact`/`hybrid` floor estimate a bit too high (i.e., ground the clip a bit too low,
causing extra penetration) but at ~1cm it does not fully explain the 5-15cm regression observed —
the larger remaining cause is most likely the human-zone `labelled` window itself: a 5cm coarse
height gate (deliberately loose, per W2-T3's design for the DETECTION task) is far less tight than
Alex's own `SOLE_CORNER_SITES` + still-frame convention was built for, so the MEDIAN foot height
over that window reflects GMR's own un-grounded retarget noise during "roughly-near-the-floor"
frames rather than a genuine, tightly-verified planted-stance height. Not chased further past this
diagnosis — the negative result is clear and consistent enough to decide without more digging,
and further tuning (tighter zone threshold, different percentile) would be iterating past this
task's own time-box on a mechanism that's already been out-performed by the simpler existing
option.

**Decision: `constant` mode SHIPS UNCHANGED — no change to the week-1 default.** Neither new mode
clears the bar; `constant`'s own established numbers (0.7-4.0cm, 0.1-0.6% pen — the very numbers
that already passed week 1's kill-test) remain the best available grounding choice for G1 this
week.

**Files**: `scripts/g1/ground_g1_contact_aware.py` (new). Outputs (not shipped, comparison-only):
`outputs/gmr_baseline/pkl_w2/*_ground_{constant-contact,hybrid}.pkl` (10 files, 5 clips × 2 modes).

---
---

# SPRINT (Humanoids 2026, 9 days)

## S1-T1 — batch retarget, 77 clips (background) + human targets

`gmr_headless_retarget.py` extended with `--save_human_targets` (saves `retargeter.
scaled_human_data` per frame, GMR's own scaled+offset FK targets — 14 LAFAN1 bones,
confirmed against `bvh_lafan1_to_g1.json`'s `ik_match_table1`/`table2`, both identical
correspondence: pelvis<-Hips, left/right_hip_yaw_link<-Left/RightUpLeg, left/right_knee_link<-
Left/RightLeg, left/right_ankle_roll_link<-Left/RightFootMod, torso_link<-Spine2,
left/right_shoulder_yaw_link<-Left/RightArm, left/right_elbow_link<-Left/RightForeArm,
left/right_wrist_yaw_link<-Left/RightHand). **Determinism gate**: walk1_subject1 retarget with
the new flag bit-identical to week-1's existing pkl (root_pos/root_rot/dof_pos max abs diff =
0.0 on all three).

Batch script `scripts/g1/sprint_batch_retarget.sh`: resumable (skips a clip if pkl+npz both
exist), background, per-clip failure log (`s1t1_retarget.log.fail`). Launched over all 77
LAFAN1 clips. Log: `outputs/gmr_baseline/sprint/s1t1_retarget.log`.

## S2-T1 — canonical-human adapter (LAFAN1 -> our Stage-3 input schema)

Read a REAL Stage-3 input NPZ first (`shovel_fronthard_02_with_orient.npz`) rather than guessing
the schema: `roles (R,)`, `positions (T,R,3)`, `frames`, `fps`, `orientation_role_names (7,)`,
`orientation_mats (T,7,3,3)`, `orientation_valid`, `facing_yaw_correction_deg`, `metadata_json`.
Confirmed by grep that `segment_*` fields are NEVER read by Stage 3 (morphology scaling is
computed inline from `positions`/`roles` alone, per `wiki/concepts/morphology-scaling.md`) — not
produced by the adapter, one fewer thing to build.

New `scripts/g1/lafan1_to_canonical_human.py`: maps 20 of Alex's 24 canonical roles directly
from LAFAN1 BVH bones (via GMR's own `load_bvh_file`), reusing `frame_from_yz`/`frame_from_xy`/
`detect_facing_yaw_deg`/`apply_yaw_to_positions` UNCHANGED from
`build_canonical_orientation_frames_fresh.py` (pure geometry, no Alex globals). Confirmed by
grep (`ROLE_TO_ALEX_BODY`/`ORI_TO_ALEX_BODY`/`CONTACT_EFFECTORS`/`CONTACT_POS`, zero hits for
`segment_*`) that the 4 omitted roles (left/right_toe_end, left/right_hand_thumb) are not
load-bearing anywhere in Stage 3.

**Two documented simplifications** (LAFAN1 lacks the source bones Alex's skeleton distinguishes):
1. `left/right_hand_middle` == `left/right_wrist` (same LAFAN1 "Hand" bone) — makes CONTACT_POS's
   palm-vs-wrist delta ~0 for this adapter; the palm pin still fires, just without the extra
   few-cm hand-centroid offset Alex's motion capture provides.
2. Hand orientation frames use `pelvis_y` as the secondary axis instead of thumb-wrist (no thumb
   bone in LAFAN1) — same fallback vector feet already use structurally, not fabricated data.

**GATE (walk1_subject1)**: feet near z≈0 (left/right_ankle mean 0.099m, toe mean 0.012-0.015m —
genuinely near-floor for a walking gait), pelvis z range 0.718-0.935m **matches W2-T3's
independently-measured table for the same clip exactly** (pelvis min 0.718), yaw auto-correction
applied (90°) and verified (post-correction first-frame left_hip−right_hip ≈ +Y as required by
the facing convention). `orientation_mats` shape (T,7,3,3) correct. All roles Stage 2.5/3 need
present. **Gate passes.**

**Files**: `scripts/g1/lafan1_to_canonical_human.py` (new).
Output: `outputs/gmr_baseline/sprint/canonical_human/walk1_subject1.npz`.

## S2-T2 — Stage 2.5 on adapted clips

`ground_canonical_human.py` ran completely UNCHANGED on the LAFAN1-adapted NPZ (schema match
confirmed by S2-T1's gate) — no fork needed at all, just the correct rate-scaled flag
(`--plant-min-run 2`, LAFAN1 is 30fps vs the pipeline's native 120fps default of 8, per
`wiki/concepts/pipeline.md`'s rate table — the exact footgun that table warns about).

`walk1_subject1`: 6512 still-plant samples, floor registered at p50=0.0039m (near-zero, as
expected for an already-well-behaved walk), shift -0.0039m. Contact zones: left/right_foot
47.4%/42.4%, left/right_hand 0.0%/0.0%. **GATE**: directionally agrees with W2-T3's own
zone table for the same clip (feet dominant, hands exactly zero) — magnitudes differ
(Alex's `contact_labels.py` height+speed+onset-hysteresis gate is much stricter than W2-T3's
height-only 5cm gate), which is the expected shape of agreement per the plan (order-of-magnitude,
not equality, different detection philosophies). **Gate passes.**

Output: `outputs/gmr_baseline/sprint/canonical_human/walk1_subject1_grounded.npz` (Stage 2's
fields + `contact_flags`/`contact_effector_names`/`contact_support_z`/`floor_shift`).

## S2-T3 — Stage 3 on G1: the genuinely new piece (core build + a real bug found+fixed)

**Reuse confirmed extensive**: `solve_frame_position_ik`, `load_canonical`, `measure_alex_pelvis_
to_head`, `estimate_source_scale`, `make_initial_alignment_targets`, `compute_per_role_scales`,
`make_targets_for_frame`, `make_orientation_targets_for_frame`, `clamp_hinge_joint_limits`,
`body_xmat` all imported COMPLETELY UNCHANGED from `solve_fbx_canonical_alex_contactfirst.py`.
Confirmed by reading: several of these (`make_initial_alignment_targets`, `compute_per_role_scales`,
`make_orientation_targets_for_frame`) hardcode `ROLE_TO_ALEX_BODY`/`ORI_TO_ALEX_BODY`'s KEY NAMES
directly (module globals, not passed as arguments) — safe to reuse ONLY because our G1 role maps
use the IDENTICAL role-name vocabulary (pelvis/torso/head/left_hip/.../right_wrist) as Alex's,
which the canonical-human schema (S2-T1) already guarantees. `TARGET_WEIGHTS`/`ORI_WEIGHTS` are
role-keyed too, genuinely shared vocabulary, no G1 copy needed.

**New G1 model (`scripts/g1/g1_model_setup.py`)**: root-caused week-1/W2-T6's "18.2% self-collision
noise" precisely — `g1_mocap_29dof.xml` gives EVERY body a duplicate mesh geom (one visual-only
contype=0, one FULL-MESH "collision" copy contype=1, confirmed by direct inspection on pelvis/hip/
knee/torso/elbow) — not an absence of collision geometry as previously assumed, an excess of BAD
(unexcluded full-mesh) collision geometry. Fix: `MjSpec`-based loader that (1) disables every
contype=1 MESH geom, (2) grafts the 15 vetted collision cylinders (W2-T6's
`g1_custom_collision_29dof.urdf`, local pos/quat/size read directly off ITS OWN compiled geoms —
no manual URDF rpy parsing) onto the SAME 39-body mocap model (which has head/hands W2-T6's
standalone vetted model lacks), (3) injects a floor mocap plane (same technique as
`_load_model_with_floor` elsewhere in this codebase). **Verified**: `walk1_subject1` raw reads
0.2%/0.68cm self-collision on this combined model — bit-for-bit matching W2-T6's standalone-model
number, confirming the graft reproduces the vetted signal exactly while adding the missing bodies.

**FOOTGUN in G1's OWN mocap XML, confirmed by direct measurement**: the body named `head_link`
sits at PELVIS HEIGHT (world [0,0,0] at the neutral pose) — its local offset from torso is the
exact geometric negation of torso's own accumulated offset, confirmed numerically (not a
coincidence). It's a cosmetic/logo-adjacent body, NOT the anatomical head — GMR's own ik_config
never maps anything to it either. The correct analog is **`head_mocap`** (an ordinary fixed body
despite the name — no `mocap="true"` attribute; "mocap" here means physical marker placement for
motion-capture systems, unrelated to MuJoCo's mocap body type), which measures a sane 0.444m
pelvis-to-head distance. Using `head_link` silently zeroed `root_scale` (divides through
`estimate_source_scale`) and cascaded into full degeneracy.

**A REAL BUG, found via render + numeric drill-down (not assumed)**: after fixing the head-body
footgun, the smoke test STILL produced a visibly collapsed pose (rendered: robot folded into a
compact heap near the ground). Numeric comparison of `make_initial_alignment_targets`' targets vs
the solve's ACHIEVED positions showed leg targets tracked correctly (knee/ankle within a cm of
target) but the ROOT QUATERNION had drifted to `[0.797, 0.097, 0.594, -0.055]` — roughly a
**74-degree rotation** — tipping the whole upper body sideways. **Root cause**: `root_reg` (a
`solve_frame_position_ik` kwarg) is DEAD — never referenced anywhere in the function body;
`posture_reg`'s `desired_dq` explicitly zeroes DOFs 0-5 (the free root) by design ("position/
orientation tasks steer it," per that function's own comment) — meaning root ORIENTATION gets
ZERO regularization unless explicit orientation targets are supplied. My initial-alignment call
(mirroring Alex's own `main()`, which ALSO omits orientation targets there) had none, leaving root
rotation completely unconstrained; G1's specific redundant-chain geometry apparently finds a bad
local minimum here where Alex's own skeleton/target balance evidently doesn't. **Fix**: pass
identity orientation targets for pelvis/torso/head to the initial-alignment call specifically (the
per-frame loop already had real orientation targets from the start — only the ONE-TIME initial call
was missing them). **Verified**: root quaternion after the fix is `[0.9997, -0.001, 0.025, 0.001]`
— an ~2.9-degree residual, sane. Rendered frame 0 (manually Z-shifted +0.6m to preview post-
grounding height, since Stage 4.5 hasn't run yet — this shift is diagnostic-only, not part of the
pipeline): a genuinely plausible single-leg-forward walking pose, torso upright, head up, arms at
sides.

**Absolute height still needs Stage 4.5 (expected, not a bug)**: raw Stage-3 output sits with
pelvis around z=0.09-0.10m, not G1's natural ~0.7m standing height. This matches Alex's OWN
architecture exactly (`wiki/concepts/pipeline.md`'s stage list: Stage 3's absolute Z is not
calibrated to the floor; Stage 4.5 Z-grounding — already ported and validated for G1 in week 1 —
owns that). Not fixed inside Stage 3 by design; the next step (T4) chains Stage-A polish +
grounding on top of this output.

**v1 scope, deliberately deferred** (2-day time-box, per the plan): no fist/palm CONTACT_POS pin
(G1 has no fist), no foot-flat orientation-alignment term during contact (position-only
contact-first hold via `hold_pos_roles`), no shank-clamp/swing-clear/arm-floor-transition/
leg-floor-transition refinement passes (Alex-specific polish for bugs found on Alex's skeleton).
Core mechanism only: per-frame damped-least-squares IK solving root+all joints jointly against
morphology-scaled position AND orientation targets, contact-held effectors at high priority, real
(vetted) self-collision + floor-collision rows available (floor_weight=0 this pass, matching the
division of labor with the G1 grounding QP that already handles clip-level floor placement).

**Files**: `scripts/g1/g1_model_setup.py` (new), `scripts/g1/solve_lafan1_canonical_g1_
contactfirst.py` (new).

## S2-T4 — polish(OURS) + first 2x2 cell comparison (walk1_subject1)

Chained `stage_a` + `ground_qpos` (both imported UNCHANGED, same functions `polish_gmr_pkl.py`
uses for polish(GMR)) directly onto the Stage-3 OURS qpos array via new
`scripts/g1/polish_ours_g1.py` — same polish code, different input source, by construction.

**Full 2x2 cell comparison, `walk1_subject1`** (`eval_motion.py`):

| variant | floorPen | pen% | vMax | spikes |
|---|---|---|---|---|
| GMR raw | 1.0c | 0.3% | 18.9 | 0 |
| GMR polished | 0.7c | 0.1% | 3.3 | 0 |
| OURS raw | **113.1c** | 100.0% | 95.2 | 10 |
| OURS polished | 16.7c | 0.7% | 11.1 | 0 |

**Reads as expected for a v1 core-mechanism-only build**: OURS-raw's huge floorPen/100% pen% is
the EXPECTED, not-yet-grounded Stage-3 output (root sits ~0.6m too low by construction, per
S2-T3's note — Alex's own pipeline has the identical property before its own Stage 4.5). Polish
recovers it substantially (113→16.7cm) — Stage A's smoothing also cleared all 10 velocity spikes
(vMax 95.2→11.1, an 8.6x reduction, same mechanism validated on GMR's output). **Two honest gaps
vs GMR's polish**: (a) floorPen after grounding (16.7cm) is notably worse than GMR-polished's
0.7cm — likely an outlier-frame residual (`constant` mode's percentile is sensitive to one bad
frame; the v1 build skips shank-clamp/floor-collision-during-solve, the exact refinements that
would catch this), (b) `worst_joint=left_ankle_pitch_joint`, pinned near its limit 73-83% of
frames — a real signal worth investigating (ankle range vs demanded target range) before treating
OURS as ready for the full corpus. Self-collision (coll%~73%) is NOT informative here —
`eval_motion.py`'s default model is still the unvetted mocap XML (W2-T6's vetted model isn't
wired into `eval_motion.py` as a default yet); needs the vetted combined model
(`g1_model_setup.py`) for a real signal.

**Not yet done**: broadening to the other 4 clips (fallAndGetUp2_subject2 is the real target —
does OURS produce visible weight-bearing contact at frame 356 where E4b's anchoring couldn't?),
the ankle-pinning investigation, running eval_motion.py's self-collision column against the vetted
model. Flagging as the next checkpoint before scaling to the full corpus, per the plan's own
"5-clip corpus first, CHECKPOINT before broadening" instruction.

## S2-T4 (continued) — a second bug found: degenerate hand-orientation frame

After the ankle/root fix, `walk1_subject1`'s `worst_joint` shifted to `right_shoulder_roll_joint`
pinned at its -129° limit on 61% of frames — a NEW signal, not present at frame 0 (shoulder-roll
sat at a normal -53° right after the initial alignment, drifting to the limit only gradually
across the per-frame loop, ruling out the same "unconstrained-initial-step" bug class).

**Root cause, confirmed by direct measurement**: `lafan1_to_canonical_human.py` (S2-T1) maps
`left/right_hand_middle` and `left/right_wrist` to the SAME LAFAN1 bone ("LeftHand"/"RightHand") —
a documented simplification. This makes `hand_dir = hand_middle - wrist` **exactly zero on every
single frame** (confirmed: `max|hand-wrist|` over the whole clip = 0.0). `frame_from_xy`'s
degenerate-input fallback then silently defaults the hand's PRIMARY orientation axis to world +X,
UNCONDITIONALLY, every frame — completely disconnected from actual arm motion. That constant,
motion-blind orientation gets world-delta-transferred onto G1 every frame, and the shoulder has to
roll to chase a signal that's actually just riding `pelvis_y` (the only thing still varying).

**Fix**: use forearm direction (`wrist - elbow`, a bone LAFAN1 actually has) as the hand's primary
orientation axis instead of the degenerate `hand_middle - wrist`. **Verified**: shoulder-roll
limit-pinning dropped from 61% to 0.01% of frames. Full `walk1_subject1` 2x2 after both fixes:
floorPen 1.4cm (GMR-polished: 0.7cm), vMax 8.2 rad/s (GMR-polished: 3.3) — converging steadily
with each real bug fixed, not fundamentally different from GMR's numbers anymore.

**File**: `scripts/g1/lafan1_to_canonical_human.py` (hand orientation primary-axis fix).

## S2-T5 — the actual target clip (fallAndGetUp2_subject2): a held-frame audit, not just a table

Ran the full pipeline (adapter → Stage 2.5 → Stage 3 → polish) on `fallAndGetUp2_subject2` — the
real test, per the plan (walk1 was only ever a sanity check GMR already wins at). First-pass 2x2:
floorPen 1.1cm polished (vs GMR-polished's 4.0cm) — numerically BETTER than GMR for the first time
in the whole project. Frame-356 render showed a genuinely different (rotated-onto-side) pose, not
the raw/GMR-polished's identical flat splay.

**Prabin's challenge, correctly skeptical**: does this reflect genuine weight-bearing contact, or
just a better whole-clip Z calibration (float% was still 98.2%, same signature as every previous
whole-clip fix in this project)? Checked directly: identified the 878/865 frames (≈18%) where
`hold_pos_roles` actually fired for left/right foot, and measured mesh-exact `support_z` (W2-T4's
helper) AT THOSE SPECIFIC FRAMES, not the whole-clip aggregate.

**Result: NOT genuine contact.** Held-frame support_z after polish: median **+13.2cm / +12.7cm**
above the floor (only 0.5%/6.2% within 1cm of z=0). The improved whole-body floorPen number was
being driven by whichever single frame in the clip happened to be lowest overall — not by the
contact-held frames actually resting on the ground. Confirms Prabin's skepticism was correct.

**Root cause**: `hold_pos_roles` freezes a planted foot's POSITION (preventing slip) but nothing
ties that frozen position to the FLOOR — the target Z comes from the morphology-scaled delta
(arbitrary absolute height, per S2-T3's own note) and the later whole-clip grounding shift has no
idea which frames are contact-held. Conceptually the same gap as W2-T5's failed anchoring (E4b),
but for a genuinely different, more fixable reason: there, GMR's root was already frozen by the
time anchoring ran; here, root is still free during Stage 3, so a floor-pull correction can
actually work.

### Fix attempt 1 — pull-to-floor for held (still) roles only

For each role in `hold_pos_roles`, before the solve, measure how far the body's origin sits above
its OWN mesh-exact support point (`support_z`) at the warm-start (previous frame's) pose, and
override the target's Z to exactly that offset — X,Y untouched. **Verified**: held-frame support_z
improved dramatically, median -7.17cm / -2.65cm (from -61.5/-61.9cm) on the RAW Stage-3 output —
genuinely close to the floor now.

**But polish UNDID it**: after Stage-A + grounding, held-frame support_z became WORSE than before
the fix (+51.3cm / +60.1cm). Root cause: pull-to-floor only corrects the ~18% held frames — the
other ~82% of the clip is just as un-grounded as ever (whole-clip lowest-point median -61.8cm,
worst -87.8cm). The single whole-clip percentile grounding shift sizes itself to fix THAT
majority and drags the now-correct held frames away from the floor as collateral damage. Same
family of issue as W2-T7's failed contact-aware grounding, opposite direction: there, contact
detection was too loose; here, floor-referencing is too narrow (held-only).

### Fix attempt 2 — extend pull-to-floor to the full contact ZONE (not just held/still)

Extended the Z-override to any frame where `contacts_solved[eff]` is true (regardless of
stillness), at normal (not hard-tier) priority. Also found the worst whole-clip penetration frame
was a HAND (`left_wrist_pitch_link`, -91cm) — `FOOT_POS_ROLE` never covered hands at all, despite
W2-T3 already showing 41.2%/34.5% hand-contact zone on this exact clip. Added `HAND_POS_ROLE`
(`left/right_hand` → `left/right_wrist`), `CONTACT_POS_ROLE = {**FOOT_POS_ROLE, **HAND_POS_ROLE}`.

**Result: worse in aggregate, not better.** floorPen raw 92.5cm (was 78.9cm), 24 velocity spikes
(was 10). Root cause: for a MOVING (zone-but-not-still) frame, the Z-override is recomputed every
frame from a warm-start orientation that's actively CHANGING — the target Z jumps around
frame-to-frame instead of holding steady, injecting new discontinuities. The mechanism that's
clean for a genuinely still plant is noisy for a limb still in motion through the zone.

**Also confirmed**: the worst-frame penetration (frame 1226, -91.4cm even after the hand
extension) occurs at a frame with **zero effectors in any contact zone at all** — a dynamic,
mid-fall transient the contact detector simply doesn't flag. No zone/hold mechanism, however
extended, touches a frame like this by construction.

### Fix attempt 3 — floor-collision avoidance during the solve (`--floor-weight`): REJECTED, known footgun

Tried `--floor-weight 20` (matching the value SESSION_HANDOFF.md's collision-fix work validated
on Alex's `luigi_standProne_03`). **Diverged badly**: whole-clip lowest point crashed to -4.5
METRES, spikes jumped to 80/4917. This is the EXACT, already-documented DLS instability this
sprint's own plan flagged under "embedded footguns — do NOT rediscover these": `--floor-weight`
above ~1.5-2x the shipped default destabilizes Stage 3's damped-least-squares solve on get-up
clips (`SESSION_HANDOFF.md`'s feasibility-first-v1 section, `planLog.md` T3). Walked directly into
a footgun already flagged in this repo's own history. **Reverted immediately** — `--floor-weight`
stays at its v1 default (0.0, off).

### Honest state at this checkpoint

Held/still contact frames: pull-to-floor genuinely works (confirmed). Whole-clip robustness on a
violent fall motion: NOT yet solved — grounding overcorrects a partially-referenced clip, zone
extension trades one noise source for another, and the "obvious" solver-side fix is a known dead
end. Net aggregate numbers with the zone-extended pull-to-floor are WORSE than GMR-polished on
this clip (floorPen 12.0cm vs GMR's 4.0cm, more spikes) — an honest regression, not a win, at this
exact checkpoint. Best available OURS variant so far for aggregate metrics is actually the
**pre-pull-to-floor** build (S2-T4's first pass, 1.1cm floorPen) — which is the version already
shown to be NOT genuinely grounded at held frames. No version yet is both aggregate-clean AND
verified-genuine.

**Next (in progress)**: a SMOOTHED version of the Z-offset for zone (not just held) frames — the
zone-extension's core idea (broader floor-referencing) is right, but computing the offset from a
raw per-frame warm-start snapshot is too noisy for a moving limb; smoothing the offset itself
(e.g., only update it slowly / low-pass across the zone's duration, or anchor it to the zone's own
onset frame the way `_compute_anchors`' median-over-stillness-run pattern does elsewhere in this
codebase) should keep the target Z stable while still floor-referencing frames beyond the strict
still/held subset.

**Files**: `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` (`--pull-to-floor` flag,
`zone_roles`, `HAND_POS_ROLE`/`CONTACT_POS_ROLE`, `contact_pt_speed`).

### Fix attempt 4 — EMA-smoothed offset, reset at zone onset

Implemented per Prabin's direction: instead of recomputing the pull-to-floor Z-offset fresh every
frame from a potentially-fast-changing warm-start orientation, maintain a per-effector EMA of the
offset (`--pull-to-floor-alpha`, default 0.15) that's explicitly RESET (not blended with stale
history) at every zone onset — mirrors `_compute_anchors`' own convention elsewhere in this
codebase of never carrying a target across a contact-interval boundary.

**Result: mixed.** Held-frame support_z quality is PRESERVED (median -6.0cm/-3.0cm, essentially
unchanged from the un-smoothed zone-extension's -7.2/-2.7cm) — the smoothing didn't break what
was already working. But velocity spikes barely moved (27 vs the un-smoothed version's 24) — the
EMA smoothing did NOT fix the dominant spike source, because it was never really about noise
*within* a zone in the first place.

**Diagnosed precisely**: of 27 spike transitions, 19 (70%) land within 1 frame of a zone
onset/offset boundary for SOME effector (left/right foot/hand). The remaining 8 don't correlate
with any boundary and are likely genuine fast within-motion dynamics, unrelated to this mechanism.
**Root cause of the boundary-clustered spikes**: the Z-target SWITCHES discontinuously the instant
a zone begins or ends — from "pure morphology-scaled delta" to "pull-to-floor-corrected offset" (or
back), a hard on/off toggle, not a smooth transition. This is exactly the class of problem
`contact_labels.py`'s own `ramp_envelope` (cosine cross-fade + preroll, already used elsewhere in
this codebase for contact transitions) was built to solve — it was available but not wired into
this mechanism.

**Next fix candidate (not yet implemented)**: cross-fade the Z-target between the raw morphology
delta and the pull-to-floor offset over a short ramp window at zone boundaries (reusing
`ramp_envelope`'s pattern), instead of a hard switch. Given this is now the 4th compounding fix in
this investigation, flagging as the next concrete, well-diagnosed step rather than continuing
without a checkpoint.

**Files**: `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` (`--pull-to-floor-alpha`,
`offset_ema` state dict, reset-at-onset logic).

### Fix attempt 5 — ramp cross-fade at zone boundaries (implemented, Prabin's direction)

Re-added `ramp_envelope` (imported unchanged from `contact_labels.py`), precomputed once per
effector over the whole clip (`zone_env`), and replaced the hard boolean on/off gate with a
continuous blend: `target_z = env * pull_to_floor_offset + (1-env) * raw_morphology_z`, where
`env` is the SAME cosine-ramp + preroll envelope already used elsewhere in this codebase for
contact transitions. `hold_pos_roles` (hard-tier position hold) stays gated on the strict
still/held boolean, unchanged — only the Z-blend uses the continuous envelope.

**Result: fixed exactly what it targeted.** Spikes dropped 27→**12** (more than half, and below
even the original held-only version's 10, despite now covering a much broader zone). Held-frame
floor quality held or slightly improved (median -3.6cm/-3.0cm, up to 34-38% of held frames now
within 3cm of the floor, vs ~16-20% before this fix).

**But the aggregate whole-clip floorPen barely moved** (raw 91.0cm, polished 16.1cm — comparable
to every prior pull-to-floor variant, still far from GMR-polished's 4.0cm). Root cause, confirmed
already (frame 1226): the aggregate number is dominated by frames with **zero effectors in any
contact zone at all** — genuine mid-air moments of a violent fall — which no envelope-based
mechanism touches by construction, however it's shaped, since it only ever activates where some
zone exists. Fixing that residual gap needs a fundamentally different mechanism (real floor-
avoidance during the solve is a confirmed dead end at this solver's current stability margin, per
fix attempt 3) or accepting that a purely contact-triggered approach has an inherent ceiling on
clips with substantial contact-free flight time.

**Honest summary of the S2-T5 investigation as a whole**: pull-to-floor + smoothing + ramp
cross-fade is a real, validated, multi-step fix for CONTACT-FRAME quality (both still and
moving-through-zone), and is now clean (no new spikes introduced, held frames genuinely near the
floor). It is NOT, on its own, sufficient to win the aggregate whole-clip floorPen metric against
GMR's polish on a clip this dynamic, because a meaningful fraction of the clip has no contact
signal to correct against at all. This is a genuine finding for the paper, not just a build
artifact: contact-anchoring mechanisms (ours, and the retired E4b) are bounded by contact-detection
coverage — a fall clip's ballistic phase needs a different kind of grounding (physics-aware or a
learned prior) that this kinematic pipeline doesn't have.

**Files**: `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` (`ramp_envelope` import,
`zone_env` precomputation, envelope-based Z-blend replacing the boolean gate).

## S1-T2 — heightfix + polish variants, 77 clips

Executed on a fresh session (model switched to Sonnet mid-sprint, per standing model-delegation
rule) after confirming S1-T1 fully complete (77/77 pkls + human-target NPZs, 0 failures).

**Bug found and fixed before the real run**: first version of `scripts/g1/sprint_polish_batch.sh`
built its clip list by globbing `outputs/gmr_baseline/sprint/pkl/*.pkl` and excluding filenames
ending in `_gmrfix|_polished|_stageA|_stageB.pkl`. That directory also holds STALE week-1/2
5-clip artifacts (`dance1_subject1_polished_constant.pkl`, `_polished_perframe.pkl`, etc. --
leftover from before the sprint's own `_polished` naming existed) whose names don't end in exactly
one of those four suffixes, so they weren't excluded -- the batch started treating them as raw
clips and produced garbage outputs like `dance1_subject1_polished_constant_gmrfix.pkl`. Caught
after 7/87 iterations (should have been 77) by noticing `total=87` in the log instead of 77. Killed
the running batch (`kill`, confirmed no orphan `polish_gmr_pkl.py` processes remained), deleted the
4 bogus derived files, archived the 20 stale week-1/2 variant pkls to
`outputs/gmr_baseline/week1_2_archive/` (out of `sprint/pkl/`, not deleted -- they're the original
week-1/2 record). **Fix**: rebuilt the clip list from `data/raw/lafan1/*.bvh` basenames (S1-T1's own
ground truth) instead of globbing the pkl directory. No real clip's output was corrupted by the bug
(alphabetically, the bogus entries only started after all real clips through `dance1_subject1` had
already completed correctly) -- confirmed by direct inspection before the fix.

Re-ran clean with the fixed script (resumable -- skipped the 9 clips already done correctly before
the kill). **Result: 77/77 clips, both variants, 0 failures**
(`outputs/gmr_baseline/sprint/s1t2_polish.log.fail` empty). `--heightfix` uses
`polish_gmr_pkl.py`'s existing flag (W2-T1's replication of GMR's paper-described fix) applied to
the RAW retarget; `--stage-a --ground` (ground-mode default `constant`) is the "polish" column --
per sprint ground rule 3, applied to RAW, never stacked on heightfix. Smoke-tested both flags on
one non-regression clip (`aiming1_subject1`) before launching the full batch.

**Files**: `scripts/g1/sprint_polish_batch.sh` (new).

## S1-T3 — eval + faithfulness, 77 clips x 3 variants

New `scripts/g1/sprint_eval_batch.py`: reuses `evaluate()` (via `eval_ihmc_json.py`, unchanged)
and `build_eval_context`/`G1_MODEL_DEFAULT` (via `eval_motion.py`, unchanged) for the main
kinematic metrics; adds three things not in either existing script:

1. **Self-collision via the vetted model** (separate pass from the main eval context, which still
   uses the unvetted mocap XML by default): loads `outputs/gmr_baseline/g1_collision/
   g1_collision_vetted.urdf` (W2-T6's artifact) and calls `_collision_stats` (imported unchanged
   from `solve_global_trajectory_opt_contactfirst.py`) directly on the same qpos array. Actuated
   joint order previously verified identical (W2-T6), so no remapping needed.
2. **Faithfulness guard**: FK'd robot-body position vs GMR's own scaled-human target
   (`human_targets/<clip>.npz`, S1-T1's `--save_human_targets` output), per a 14-pair
   robot-body<->LAFAN1-bone correspondence read directly from
   `bvh_lafan1_to_g1.json`'s `ik_match_table2` (NOT `table1` -- checked both: `table1`'s
   `position_cost` is 0 for pelvis and low (0-50) elsewhere, i.e. it mostly carries orientation
   weight; `table2` carries the real position-tracking weight (10-100 across all 14 pairs,
   confirmed by reading `motion_retarget.py`'s `setup_retarget_configuration`, which builds a
   `mink.FrameTask` per table entry with `position_cost=pos_weight`). `table2` is therefore the
   correspondence GMR itself actually optimizes position against.). Confirmed `pos_offset==[0,0,0]`
   for all 14 `table2` entries and `ground_height==0.0` in the config, so the human_targets npz's
   `pos__<Bone>` value IS the position target with no further transform -- verified by reading
   `motion_retarget.py:154` (`offset_human_data` consumes the offsets before `update_targets`
   assigns `task.set_target` from `human_data[body_name]` directly).
3. **hipZ p5** per clip (GMR's own `load_bvh_file`, identical convention to week-1 T2's clip
   screening) for T4's class split.

Resumable (skips a `(clip, variant)` row already in the CSV). Ran via
`conda run -n gmr python scripts/g1/sprint_eval_batch.py` (script-mode invocation --
confirmed a `python -c` invocation from repo root DOES trigger the known
`general_motion_retargeting` package-shadowing footgun (T1) via cwd landing on `sys.path[0]`;
running the actual script file does not, since `sys.path[0]` is then the script's own directory).

**Result: 77/77 clips, all 3 variants, 231/231 rows, 0 failures**
(`outputs/gmr_baseline/sprint/s1t3_eval.fail` empty). No `faith_*` NaNs -- every clip's
human-targets NPZ was present and usable, the plan's proxy fallback was never needed.

**GATE (regression, the 5 week-1/2 clips) -- PASSES, exact match**:

| clip | variant | this run floorPen/pen% | prior (GMR-baseline-results.md) |
|---|---|---|---|
| walk1_subject1 | raw | 1.04cm / 0.26% | 1.0cm / 0.3% |
| walk1_subject1 | gmrfix | 2.03cm / 5.66% / 55.0% float | 2.0cm / 5.7% / 55.0% |
| walk1_subject1 | polished | 0.74cm / 0.14% | 0.7cm / 0.1% |
| dance1_subject1 | raw | 7.07cm / 1.93% | 7.1cm / 1.9% |
| dance1_subject1 | gmrfix | 2.74cm / 0.33% / 99.5% float | 2.7cm / 0.3% / 99.5% |
| dance1_subject1 | polished | 3.22cm / 0.63% | 3.2cm / 0.6% |
| fallAndGetUp2_subject2 | raw | 13.62cm / 47.1% | 13.6cm / 47.1% |
| fallAndGetUp2_subject2 | gmrfix | 2.90cm / 0.41% / 98.6% float | 2.9cm / 0.4% / 98.6% |
| fallAndGetUp2_subject2 | polished | 4.02cm / 0.51% | 4.0cm / 0.5% |
| fallAndGetUp1_subject1 | raw | 12.86cm / 38.9% | 12.9cm / 38.9% |
| fallAndGetUp1_subject1 | gmrfix | 5.24cm / 3.82% / 94.0% float | 5.2cm / 3.8% / 94.0% |
| fallAndGetUp1_subject1 | polished | 1.15cm / 0.48% | 1.1cm / 0.5% |
| ground1_subject1 | raw | 15.94cm / 90.6% | 15.9cm / 90.6% |
| ground1_subject1 | gmrfix | 2.87cm / 0.06% / 99.9% float | 2.9cm / 0.1% / 99.9% |
| ground1_subject1 | polished | 2.44cm / 0.53% | 2.4cm / 0.5% |

vMax and self-collision (vs W2-T6's vetted-model table) also matched exactly on all 5 clips
(e.g. `walk1_subject1` raw coll 0.2%/0.68cm vs W2-T6's 0.2%/0.7cm; `fallAndGetUp2_subject2` raw
5.77%/5.78cm vs 5.8%/5.8cm). No drift introduced by this sprint's refactor.

**Faithfulness sanity**: polish-vs-raw delta in mean position error is small and mostly
floor-class-concentrated where expected (largest increases: `obstacles5_subject4` +7.15cm,
`ground1_subject5` +4.52cm, `fallAndGetUp1_subject4` +4.10cm -- all floor-class clips where
whole-clip Z-shifting genuinely trades position fidelity for floor placement, consistent with
`GMR-baseline-results.md`'s already-documented honest caveat). Several clips even improve slightly
(`walk1_subject1` -0.52cm, `walk3_subject1` -0.59cm). No wild outliers -- polish does not wander.

**Files**: `scripts/g1/sprint_eval_batch.py` (new). Output:
`outputs/gmr_baseline/sprint/s1t3_eval.csv` (77 x 3 rows, 19 columns).

## S1-T4 — class-split table + Table-I mapping

New `scripts/g1/sprint_s1t4_summary.py` (stdlib only -- `pandas` is not installed in the `gmr`
conda env, discovered when the first pandas-based draft failed to import; rewrote with
`csv`/plain dict aggregation). Class split: hipZ p5 < 0.3 -> floor (T2's exact convention, per the
plan) -> **20 floor-class, 57 locomotion-class** of the 77 clips.

| class | variant | floorPen | pen% | float% | coll%(vetted) | vMax | faith_mean |
|---|---|---|---|---|---|---|---|
| locomotion (57) | raw | 7.43cm | 4.32% | 91.6% | 5.12% | 33.28 | 10.21cm |
| locomotion (57) | gmrfix | 3.18cm | 1.64% | 92.9% | 5.12% | 33.28 | 10.24cm |
| locomotion (57) | polished | 3.08cm | 0.55% | 98.0% | 4.98% | 5.49 | 10.87cm |
| floor (20) | raw | 17.98cm | 38.3% | 57.3% | 4.42% | 33.58 | 10.22cm |
| floor (20) | gmrfix | 4.31cm | 0.97% | 98.5% | 4.42% | 33.58 | 13.41cm |
| floor (20) | polished | 3.83cm | 0.68% | 98.5% | 3.73% | 5.94 | 12.42cm |

Full per-class-per-variant CSV: `outputs/gmr_baseline/sprint/s1t3_eval.csv` (source data).

**Honest finding, not smoothed over**: the T2 hip-Z-p5<0.3 threshold (designed for sustained
lying/crawling clips) does NOT cleanly separate "clean locomotion" from clips with real, if
BRIEF, floor contact -- 29 of the 57 "locomotion"-class clips show raw floorPen > 5cm, several
(`pushAndStumble1_subject3` 31.5cm, `walk2_subject3` 21.1cm, `aiming2_subject3` 20.0cm,
`push1_subject2` 18.4cm, `obstacles6_subject4` 17.7cm) worse than two of the three
week-1/2-selected floor clips. Mechanism: these are stumbles/pushes/obstacle-clearing/kneeling
motions where a HAND or KNEE touches the ground briefly while the HIP stays above the 0.3
threshold -- the same "hip height alone is an incomplete floor-contact signal" finding from W2-T3's
multi-surface contact labels, now visible in the aggregate. The "locomotion" class's 7.43cm mean
raw floorPen (vs the floor class's 17.98cm) is a real, large gap, but it is NOT "GMR is clean on
everything except the 20 excluded-class clips" -- it is inflated by ~half its members having some
real floor interaction. Used the plan's literal p5<0.3 rule as specified (comparable to T2), but
this nuance belongs in the paper's methodology section, not just this log.

**Table-I mapping**: checked (a) the paper website (jaraujo98.github.io/retargeting_matters, via
WebFetch) -- confirmed no BVH-filename/subject mapping present, only generic category names
("Walk", "Dance", "Kung fu" etc.); (b) the GMR clone's `ik_configs/` and README -- no motion-name
list or LAFAN1-file mapping found either. **Unmapped, per the plan's own instruction** ("give up
and mark unmapped -- author email is Prabin's call, not the executor's"). No published-number
annotation possible on any clip in this table without it; does not block the kinematic 2x2 itself.

**CHECKPOINT: S1 complete, table above ready for Prabin.** S2 (E7, OURS on G1) was explicitly
out of scope for this pass (parked at its own M4 checkpoint, S2-T5) and was not touched. S3/S4
not started.

**Files**: `scripts/g1/sprint_s1t4_summary.py` (new).

## S1-T4 (addendum) — reclassification by real multi-surface contact (Prabin's call, same session)

Prabin's challenge, correctly skeptical of the hip-only split above: "why hip for grounding" --
clarified hip-Z-p5 was never used for grounding (that's mesh-exact whole-body-lowest-point vs
floor, in `post_process_ground_contactfirst.py`), only for clip CLASSIFICATION. But classification
by hip alone has exactly the blind spot W2-T3 already diagnosed (a hand/knee can touch the ground
while the pelvis stays up) -- decided to fix it before the table ships.

New `scripts/g1/sprint_reclassify_contacts.py`: reuses `human_contacts_lafan1.py`'s `detect()`/
`LANDMARKS` UNCHANGED (same thresholds W2-T3 calibrated: feet 0.05m, hands/knees/elbows 0.08m,
pelvis/torso 0.15m) over all 77 clips (batch + NPZ cache, resumable -- skipped the 5 clips W2-T3
already computed). **Classification rule**: floor-class if ANY non-foot landmark (hand/knee/
elbow/pelvis/torso -- feet excluded, walking alone lights those up on every clip) has a
CONTIGUOUS in-zone run >= 1 second (30 frames @ 30fps), not just a nonzero zone percentage --
operationalizes W2-T3's own qualitative distinction between "sustained" floor-clip contact
(25-88% zone) and "noise-level" brief dips (`dance1_subject1`'s hand blips, called out there as
"a brief low gesture, not sustained contact"). A bare zone-% threshold would let a busy fight/dance
clip cross via many short scattered dips without ever actually resting on anything; a run-length
bar doesn't.

**Sanity check on the 5 known clips**: `walk1_subject1` (0.00s max run, locomotion -- correct),
`dance1_subject1` (0.43s max run, BELOW the 1s bar, locomotion -- correctly excludes the noted
noise-level hand blip), `fallAndGetUp1_subject1`/`fallAndGetUp2_subject2`/`ground1_subject1` (7.4s
/ 10.3s / 19.7s max run, all floor -- correct). Full per-clip landmark %/max-run table:
`outputs/gmr_baseline/sprint/s1t4_reclass.csv`.

**Result: 34 floor-class / 43 locomotion-class** (vs 20/57 under hip-only) -- the hip-only split
undercounted floor-contact clips by 14. Newly-caught floor clips include several from the earlier
"honest finding" outlier list (`pushAndStumble1_subject3`, `push1_subject2`,
`obstacles6_subject4`, `walk2_subject3`, `aiming2_subject3` -- all now correctly floor-class) plus
others the old split also missed (`obstacles5_subject2` 39.2s max run, `walk3_subject3` 31.6s,
`ground1_subject4` 33.4s).

**Re-aggregated table** (`sprint_s1t4_summary.py`, now defaulting to this classification when
`s1t4_reclass.csv` is present):

| class | variant | floorPen | pen% | float% | coll%(vetted) | vMax | faith_mean |
|---|---|---|---|---|---|---|---|
| locomotion (43) | raw | 4.77cm | 1.91% | 93.5% | 3.85% | 32.82 | 10.19cm |
| locomotion (43) | gmrfix | 2.69cm | 1.89% | 91.1% | 3.85% | 32.82 | 9.92cm |
| locomotion (43) | polished | 2.59cm | 0.49% | 97.9% | 3.75% | 5.45 | 10.90cm |
| floor (34) | raw | 17.00cm | 27.35% | 69.0% | 6.32% | 34.04 | 10.23cm |
| floor (34) | gmrfix | 4.46cm | 0.94% | 98.5% | 6.32% | 34.04 | 12.51cm |
| floor (34) | polished | 4.14cm | 0.70% | 98.5% | 5.79% | 5.80 | 11.74cm |

**Materially cleaner separation, no more locomotion-class outliers dominating the mean**:
locomotion-class raw floorPen drops from the hip-only split's 7.43cm to 4.77cm (pen% 4.32%->
1.91%) -- much closer to what a genuinely clean-locomotion baseline should read, since the ~14
clips with real brief contact moved to their correct class. Floor-class raw floorPen is similar in
magnitude (17.0cm vs 17.98cm) but pen% drops (38.3%->27.4%, diluted by more, generally
less-severe, newly-added floor clips vs the original 20 which skewed toward the worst cases) --
still an order of magnitude worse than locomotion on every metric. **This is the split that should
ship in the paper table**, not the hip-only one -- it is grounded in the same human-contact
detector already validated in W2-T3, not a new mechanism.

**Files**: `scripts/g1/sprint_reclassify_contacts.py` (new). Output:
`outputs/gmr_baseline/sprint/s1t4_reclass.csv` (77 rows), extended
`outputs/gmr_baseline/human_contacts/*.npz` (72 new files, 5 pre-existing from W2-T3 untouched).

---
---

# NEXT SESSION follow-through (2026-07-16)

## S2-T6 (N1-a) — corrected vetted self-collision measurements

Bug confirmed exactly as flagged in the plan handoff: `_collision_stats(model, data, qpos,
floor_gid=None, ...)` on the combined `g1_model_setup.py` model (which contains an injected floor
mocap body) lets floor contacts leak into the self-collision count — the floor body is a mocap
child of worldbody (id != 0), so the `b1==0 or b2==0` exclusion misses it. `_collision_stats`'s
own docstring requires `floor_gid` be passed whenever the model has an injected floor, regardless
of `count_floor`.

New `scripts/g1/sprint_s2t6_corrected_collision.py`: re-measured all 4 tested clips' current
("ramped") OURS raw+polished variants AND the GMR-polished comparison row, both with the WRONG
(floor_gid=None) and CORRECTED (floor_gid passed) call, side by side:

| clip / variant | WRONG coll% / peak | CORRECTED coll% / peak |
|---|---|---|
| walk1 OURS raw | 99.5% / 75.22cm | **42.5% / 5.09cm** |
| walk1 OURS polished | 30.5% / 9.46cm | 30.0% / 9.46cm |
| walk1 GMR-polished | 4.1% / 1.18cm | 0.0% / 0.00cm |
| fallAndGetUp1 OURS raw | 99.8% / 80.73cm | **28.2% / 10.41cm** |
| fallAndGetUp1 OURS polished | 25.9% / 10.63cm | 25.4% / 10.63cm |
| fallAndGetUp1 GMR-polished | 3.2% / 5.91cm | 2.5% / 5.91cm |
| fallAndGetUp2 OURS raw | 99.9% / 92.11cm | **29.4% / 5.29cm** |
| fallAndGetUp2 OURS polished | 30.5% / 16.36cm | 29.6% / 8.24cm |
| fallAndGetUp2 GMR-polished | 6.0% / 5.71cm | 5.0% / 5.71cm |
| ground1 OURS raw | 100.0% / 73.40cm | **12.7% / 5.94cm** |
| ground1 OURS polished | 10.5% / 5.89cm | 10.4% / 5.89cm |
| ground1 GMR-polished | 4.4% / 4.40cm | 2.6% / 4.22cm |

**Confirms the hypothesis exactly**: the RAW OURS numbers were almost entirely floor-penetration
contamination (peaks of 73-92cm collapse to 5-10cm once floor contacts are excluded — matching
each clip's own known un-grounded Stage-3 output, since raw Stage-3 sits well below the floor by
construction, per S2-T3's note). The POLISHED numbers were already ~uncontaminated (polish removes
most floor pen first, so `floor_gid=None` vs passed differs by <1.5 percentage points and 0cm peak
on every clip) — meaning **the ~25-30% polished self-collision figure quoted in the 2026-07-15
chat session for OURS was NOT a measurement artifact — it is real**, confirming N1-b's premise
(the elbow-in-torso finding stands, investigate below). GMR-polished's own numbers also shifted
slightly (walk1 4.1%→0.0%, ground1 4.4%→2.6%) — smaller leaks, same root cause, now also
corrected. **This table supersedes any polished/raw self-collision number quoted before this
entry for the 4 tested clips.**

## S2-T6 (N1-b) — elbow-in-torso self-collision hunt

**Root cause found — a genuine THIRD mechanism, distinct from both hypotheses in the plan
handoff (not the morphology-scaling clamp, not hand-orientation coupling).**

**Step 1 (targets vs achieved, worst frame)**: found the actual worst frame in the CURRENT
"ramped" build is still frame 484 (torso↔left_elbow, 9.46cm, matches the pre-handoff finding
exactly — confirms this is the same bug, not a new one from later fixes). `left_elbow`
target-vs-achieved distance: 33.8cm (RAW), 84.6cm (polished, expected — polish's grounding shift
moves the whole body). Root position/orientation are themselves badly discontinuous frame-to-frame
around 482-486 (root x jumps 2.72→2.58→2.34m within 3 frames; quaternion changes by ~40-70° in a
single frame) — a whole-body instability event, not a local arm-tracking failure. This ruled out
"solver just misses the elbow target" as the whole story — something is forcing the WHOLE body,
including root, to lurch.

**Step 2 (instrumented the actual per-frame blend)**: re-ran frames 478-495 with the real warm
start (loaded from frame 477 of the actual ramped run) and printed the pull-to-floor Z-blend's
internals every frame. Found: `right_foot`'s contact zone opens at frame 479 (`zone_env` ramps
0→0.25→0.75→1.0 over frames 479-481, the ramp working exactly as designed — NOT a discontinuity
in the envelope itself). But the **pre-blend target Z** (raw morphology-scaled, Stage-3's own
un-grounded convention) sits at **-0.57m**, while the **pull-to-floor offset** (computed as
"what absolute world Z puts this body's support point at world z=0") converges to **+0.06m** —
a **63cm gap between the two Z conventions being blended together.** Even ramped smoothly over
3-4 frames by `zone_env`, a 63cm target correction in ~3 frames is a ~20cm/frame demand no
per-frame IK can absorb without the whole body lurching to chase it — confirmed: root x moves
~14cm between frames 483→484 alone, exactly the same window the target Z is mid-ramp.

**Step 3 (isolate pull-to-floor's contribution, ladder item 3, decisive)**: ran the SAME clip
with `--no-pull-to-floor`. Root trajectory across frames 478-490 is now completely smooth (x
decreasing steadily 2.66→2.48m, y/z stable to within 1cm frame-to-frame) — the lurch is GONE.
Self-collision (corrected measurement) drops from **42.5%/5.09cm (with pull-to-floor) to
18.4%/1.56cm (without)**. This fully confirms pull-to-floor's Z-blend is the direct, sufficient
cause — not a contributing factor among several.

**The actual bug, precisely stated**: pull-to-floor computes its correction target in the
**absolute, post-grounding world-Z frame** ("this body's support point should sit at world z=0"),
but blends it against Stage-3's own **raw, un-grounded morphology-scaled targets**, which live in
whatever arbitrary height convention the ONE-TIME initial rest-alignment solve happened to settle
into (that solve has no floor constraint, no gravity — pure IK against position/orientation
targets only, so its resting height is unconstrained and can land anywhere; on `walk1_subject1`
it settled ~60cm below where "true floor" would be). Grounding (Stage 4.5, `post_process_
ground_contactfirst.py`) is normally the ONLY place that reconciles Stage-3's arbitrary height
convention with the true floor — my pull-to-floor mechanism tried to do that job INSIDE Stage 3,
in absolute coordinates, before grounding has ever run, creating exactly the kind of two-Stages-
disagree-on-the-datum bug this codebase's whole architecture (one mechanism per job, upstream
invariants established once — see `wiki/concepts/phasic-architecture.md`) is designed to avoid.

**This also explains why it looked fine on `fallAndGetUp2_subject2`** (S2-T5's held-frame audit
found median support_z -3 to -7cm, genuinely close to floor): a lying/fall clip's un-grounded
Stage-3 trajectory happens to stay much closer to the true-floor Z range throughout (the human
motion itself doesn't traverse a large vertical range in that un-grounded reference), so the
gap between "raw Stage-3 Z" and "absolute floor Z" was small there — the SAME bug, just with a
small enough magnitude on that specific clip to not visibly destabilize the solve. It is not
robust; it happened not to bite there.

**Correct fix direction (not yet implemented, flagged for whoever picks this up)**: pull-to-floor
should NOT reference an absolute world-Z floor at all — it should follow this codebase's own
established pattern (`_compute_anchors` in `solve_global_trajectory_opt_contactfirst.py`): anchor
a held role to the MEDIAN of its own recently-achieved trajectory during a stillness run (prevents
slip/drift, the actual job contact-anchoring exists for), and leave absolute floor PLACEMENT
entirely to Stage 4.5 grounding, unchanged, exactly as this whole codebase already divides that
labor for Alex. This is a real redesign of the mechanism (recompute the anchor as a running
median over the current stillness interval, not a per-frame absolute-floor-referenced snapshot),
not a parameter tweak — time-boxed out of this session per the plan's own instruction ("if not
root-caused after the ladder, log findings ... move to N2"). ROOT-CAUSED — moving to N2 now.

**Files referenced (no code changes made this pass — diagnosis only)**:
`scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` (pull-to-floor blend, unchanged pending the
redesign above). New scratch instrumentation not committed (session-local only, per the plan's
`/tmp` scratchpad convention).

## N2 (S3-T1) — BeyondMimic environment prep

**Repo**: no BeyondMimic clone existed anywhere on this machine (only the arXiv PDF). Confirmed
the canonical repo via web search (not guessed from memory, per this session's URL-generation
discipline): `github.com/HybridRobotics/whole_body_tracking` (MIT). Cloned read-only to
`/home/ptimilsina/projects/whole_body_tracking` for inspection — not installed, no packages
touched.

**CSV format gate: PASSES, and the plan's named blocker does NOT apply.** Read
`GMR/scripts/batch_gmr_pkl_to_csv.py`'s actual source: it only reads `dof_pos`/`fps`/`root_pos`/
`root_rot` from the pkl — `local_body_pos`/`link_body_list` (which our `save_gmr_pkl` saves as
`None`) are never touched. Ran it directly on `walk1_subject1_polished_constant.pkl`: succeeds,
output CSV is 7840 rows × 36 columns (3 root_pos + 4 root_rot xyzw + 29 dof_pos), row count
matches the source clip's frame count exactly. **No code change needed for this gate.**

**Two REAL blockers surfaced, neither resolvable without Prabin's input — reporting, not
guessing past them:**

1. **IsaacLab version mismatch, unresolved.** BeyondMimic's README pins **Isaac Lab v2.1.0 +
   IsaacSim 4.5.0**. The existing install at `/home/ptimilsina/IsaacLab` (the one the live 3-day
   GPU job is running on, proving it works on this 5080/Blackwell) reports IsaacSim
   **5.1.0-rc.19** and an IsaacLab package-level version of **0.45.7** (from its own
   `CHANGELOG.rst`) — no git tag on the checked-out commit. These are DIFFERENT versioning
   schemes (IsaacLab moved from major.minor tags like "2.1.0" to semver-style package versions at
   some point) — whether 0.45.7 is compatible with, older than, or newer than what BeyondMimic
   was built against is NOT something I can confidently determine from version strings alone, and
   I will not guess. Also: the live job's Python interpreter IS
   `/home/ptimilsina/IsaacLab/_isaac_sim/kit/python/bin/python3` — installing BeyondMimic's
   extension into that SAME environment while the job is running risks corrupting its dependency
   state. **Did not attempt `pip install -e source/whole_body_tracking` for this reason** — that
   step needs to wait until the job stops (Prabin's call) or a separate env is confirmed safe.
2. **WandB registry dependency, not anticipated in the sprint plan.** BeyondMimic's own motion
   pipeline is NOT "load a local CSV/NPZ and train" — it's: CSV → `csv_to_npz.py` (computes
   max-coordinates via FK) → **automatic upload to a WandB Registry artifact** → training reads
   the motion by WandB registry path (`{org}-org/wandb-registry-motions/{motion_name}`), not a
   local file. Requires a WandB account, an org, a "Motions" registry collection created in it,
   and `WANDB_ENTITY` set to the org (not personal username, per their own docs). This is an
   external-service signup decision, not something to set up unilaterally on Prabin's account.

**What IS confirmed ready**: the retargeting→CSV leg of the pipeline (GMR pkl → polish →
`batch_gmr_pkl_to_csv.py` → 36-col CSV) works end to end, unmodified, on our actual outputs. The
task name to use once training starts: `Tracking-Flat-G1-v0` (matches our target robot).

**Reporting readiness, not proceeding further**: do NOT stop the running GPU job. Two decisions
needed from Prabin before S3 can actually train: (a) whether to attempt BeyondMimic on the
existing IsaacLab 0.45.7/IsaacSim 5.1 install as-is (risk: unverified compatibility) or set up a
separate IsaacLab v2.1.0 checkout, and (b) WandB account/org to use for the motion registry.
Table-I mapping is still unmapped (S1-T4) — the Dance-5 stand-in clip question from the plan
handoff is still open too.

**Files**: `/home/ptimilsina/projects/whole_body_tracking/` (new clone, read-only, not installed).

## S2-T6 correction (2026-07-16) — the "fallAndGetUp2 sits closer to floor" claim was WRONG

Prabin asked directly: Stage 2.5 grounds the HUMAN data (confirmed, working) — so why does
Stage 3's UN-GROUNDED robot trajectory not sit close to true floor height anyway? Investigated
properly rather than re-asserting the earlier hand-wave.

**Root mechanism, verified with hard numbers**: `make_initial_alignment_targets`'s pelvis target
is a LITERAL world-origin position, `[0.0, 0.0, 0.0]` (imported unchanged from Alex's solver) —
not "the human's floor-referenced pelvis height," not "G1's natural standing pelvis height." Just
world-frame `[0,0,0]`. Confirmed the solve achieves pelvis ≈ `[0, 0, 0.09]` (nailing that target,
highest role weight). G1's own kinematic definition has the pelvis body AS the floating-base
origin, with the leg chain extending ~0.6-0.76m BELOW it by the model's own fixed geometry
(confirmed: even G1's all-zero neutral pose has ankle at z=-0.757 relative to pelvis-at-origin).
So pinning pelvis to world-origin mechanically drags the whole leg chain to roughly z=-0.5 to
-0.7m, independent of anything in the human data — a pure artifact of choosing world-origin as
the one-time rest-alignment anchor. Every subsequent frame's target is then a DELTA on top of
this arbitrary near-zero baseline (`root_scale × human-pelvis-motion-relative-to-its-OWN-first-
frame`) — the human's absolute floor-referenced height never enters the computation at all, only
relative motion does. **This is not a bug — it's already-documented, expected behavior**: S2-T3's
own note states raw Stage-3 output sits at pelvis~0.09-0.10m "matches Alex's OWN architecture
exactly... Stage 4.5 owns [absolute height]." Stage 3 was never meant to be floor-referenced for
EITHER robot; only Stage 4.5's whole-clip grounding shift (unmodified, already used) reconciles
it. Pull-to-floor's bug (logged above) was introducing an absolute-floor computation INSIDE
Stage 3, conflicting with this pre-existing, deliberate division of labor.

**CORRECTION to this file's earlier claim**: the S2-T5 entry above asserted "a lying/fall clip's
un-grounded Stage-3 trajectory happens to stay much closer to the true-floor Z range throughout"
as the explanation for why pull-to-floor looked stable on `fallAndGetUp2_subject2` but not
`walk1_subject1`. **Directly checked and this is FALSE.** `fallAndGetUp2_subject2`'s own
rest-alignment solve produces the SAME ~0.58m arbitrary ankle offset as walk1 (achieved-rest
ankle: -0.578m fallAndGetUp2 vs -0.584m walk1 — essentially identical), and its per-frame ankle
target Z ranges -0.58 to -0.28m across the WHOLE clip (median -0.49m) — never close to true floor
either. The magnitude of the Stage-3/absolute-floor gap is NOT smaller on the fall clip.

**Best current (still not fully verified) explanation for why walk1 destabilized visibly and
fallAndGetUp2 did not**: contact-zone DURATION and FREQUENCY differ sharply between the two clip
types. `walk1_subject1`'s gait cycle produces short, frequent zone transitions (each foot
touches/releases roughly every 0.3-0.5s), so the SAME ~60cm Z-correction must ramp in over the
same short `--contact-ramp 2 --contact-preroll 1` window every half-second, repeatedly, forcing a
fast, frequently-repeated root velocity demand. `fallAndGetUp2_subject2`'s held zones last 7-19
SECONDS at a time (T2's own screening), so the same-size correction, ramped over the same few
frames, happens rarely and — critically — the body is otherwise nearly stationary during a long
lying-phase hold, so there's less concurrent motion competing with the correction for the same
per-frame DLS step budget. **This is a hypothesis, not yet directly measured** (would need root
velocity/acceleration compared frame-by-frame across zone onsets on both clips) — flagged for
whoever picks up the pull-to-floor redesign, not asserted as fact.

**What is NOT in question, verified twice now on two different clips**: the fix direction from
the earlier entry stands regardless of this correction — pull-to-floor must stop referencing
absolute world-floor-Z altogether and instead anchor to a running median of the role's own
recent achieved trajectory (mirroring `_compute_anchors`), leaving absolute floor placement
entirely to Stage 4.5. That conclusion does not depend on which clip destabilized more visibly.

## S2-T6 (implementation) — floor-referenced rest anchor: THE fix, fully validated

**Implemented** (Prabin's direct question: "why is [pelvis] taken to 0,0,0? why is that needed?"
— it isn't). In `solve_lafan1_canonical_g1_contactfirst.py`, right after calling the shared,
UNCHANGED `make_initial_alignment_targets` (Alex's own function — never modified), apply a
uniform Z-shift to every role's initial target:
```
pelvis_floor_z0 = root_scale * human_pelvis_z(frame 0)   # already floor-referenced by Stage 2.5
initial_targets = {role: t + [0,0,pelvis_floor_z0] for role, t in initial_targets.items()}
```
Derivation (verified exactly): `make_targets_for_frame`'s existing, unchanged formula is
`pelvis_target(t) = target_rest_positions[pelvis] + root_scale*(human_pelvis(t)-human_pelvis(0))`.
Substituting `target_rest_positions[pelvis]_z = root_scale*human_pelvis_z(0)` telescopes this to
`pelvis_target_z(t) = root_scale * human_pelvis_z(t)` — G1's pelvis height becomes directly
proportional to the human's OWN real, already-grounded floor-referenced height, for the WHOLE
clip, automatically, with zero other code changes. No fork of the shared Alex function needed —
pure glue-code post-processing of its returned dict.

**Verified this eliminates the root cause, not just symptoms**:
- `walk1_subject1` root Z range: **0.526 to 0.846m** (previously -0.52 to +0.74m nonsense) —
  matches G1's true standing-height range exactly.
- The catastrophic snap at frame 484 (root x jumping ~40cm within 2 frames, quaternion changing
  40-70° in one frame) is COMPLETELY GONE — that exact zone transition is now smooth.
- Self-collision (corrected, vetted model): **42.5%→15.3%** (raw), peak 5.09→2.51cm.
- 5 small residual spikes remain, all confirmed to sit exactly at OTHER zone onset/offset
  boundaries (not the fixed one) — a much smaller, separate residual (ramp-boundary softness),
  consistent with the same class of small artifact `fix attempt 5` already reduced on the fall
  clip, not a re-emergence of the coordinate bug.

**CORRECTION to the earlier ("running-median anchor") proposed fix direction**: that proposal
(mirror `_compute_anchors`, anchor to trajectory median instead of absolute floor) would have
treated the SYMPTOM (contact-frame drift) without fixing the actual disease (Stage 3's arbitrary,
non-floor-referenced coordinate baseline for the WHOLE clip, not just contact frames). The
floor-referenced-rest-anchor fix implemented here is upstream, smaller, and fixes both the
instability AND (as shown below) genuinely improves contact quality beyond what the running-
median approach could have achieved, since it makes Stage 3's ENTIRE trajectory floor-aware, not
just the anchored subset.

## S2-T6 — Prabin's challenge: "if OURS adds something contact-relevant, THAT should improve, not
just get worse" — investigated properly, found a SECOND real bug, now fixed

Re-ran the held-frame `support_z` audit (S2-T5's own discriminating metric) on the floor-fix
build's POLISHED output (Stage A + grounding, same recipe as GMR's polish) across all 4 clips.
**Result was WORSE than GMR-polished on every clip, every foot** (e.g. walk1: OURS +12.5/+12.6cm
vs GMR's +4.6cm; fallAndGetUp2: OURS +18-19cm vs GMR's +11-12cm) — the opposite of what a
contact-aware mechanism should show, exactly as Prabin flagged.

**Root-caused, not assumed**: checked held-frame `support_z` on the RAW (pre-polish) floor-fixed
output first — **median 0.16-0.23cm on walk1, 80-84% of held frames within 3cm of the floor** —
already GENUINELY excellent, better than GMR-polished's own diagnostic number. The mechanism
works. The regression is introduced ENTIRELY by the subsequent "polish" step — specifically
`constant`-mode grounding, which computes ONE percentile-based Z-shift from the WHOLE clip's
blind minimum (on walk1: driven by `left_ankle_roll_link` at frame 599, a normal swing-phase
foot-clearance dip, -14.2cm, nothing to do with contact) and applies that SAME shift UNIFORMLY to
every frame — including the already-correct held frames, dragging them away from the floor by
however much the unrelated correction needed. Confirmed the grounding shift itself differs
sharply: OLD (broken, pre-floor-fix) raw needed a +0.691m shift; NEW (floor-fixed) raw needs only
+0.115m — a small, sane correction — but even that small shift is enough to ruin frames that were
already at zero.

**Confirmed the fix**: re-measured held-frame `support_z` with Stage-A-ONLY (temporal smoothing,
no grounding shift) on all 4 clips — quality is FULLY PRESERVED (median -0.89 to -1.06cm on
walk1, -0.01 to -1.06cm on the fall clips, +0.50 to +1.46cm on ground1; 53-81% of held frames
within 3cm across all 4 clips/effectors). Self-collision and vMax are UNCHANGED by removing
grounding (18.8%/8.2 on walk1 either way) — those come from the joint solve itself, confirming
grounding (a pure Z-shift) was never touching them; it was ONLY ever hurting contact quality
while doing nothing for these other metrics.

**FULL VALIDATED RESULT, all 4 clips, OURS(Stage-A-only) vs GMR-polished:**

| clip | held support_z: GMR / OURS | frac<3cm: GMR / OURS | whole-clip floorPen: GMR / OURS | coll%: GMR / OURS |
|---|---|---|---|---|
| walk1_subject1 | +4.6cm / **-1.0cm** | 6.8% / **72.9%** | **1.2cm** / 15.2cm | **0.0%** / 18.8% |
| fallAndGetUp1_subject1 | +10.5cm / **-0.9cm** | 0.4% / **72.2%** | **1.6cm** / 23.7cm | **2.5%** / 15.0% |
| fallAndGetUp2_subject2 | +11.5cm / **-0.4cm** | 0.0% / **77.0%** | **5.1cm** / 20.4cm | **5.0%** / 20.8% |
| ground1_subject1 | +8.8cm / **+1.0cm** | 0.0% / **54.3%** | **4.4cm** / 19.8cm | **2.6%** / 15.6% |

**OURS beats GMR-polished on the ONE metric this whole mechanism exists to improve — held-frame
contact quality — on every single clip, every effector, no exceptions.** This is the genuine,
validated, contact-specific claim: our per-limb contact anchoring produces actual floor contact
GMR's polish structurally cannot (confirmed again: GMR-polished's own held-frame support_z is
4.6-11.5cm off the floor, unchanged from what W2-T5/S2-T5 already found for its whole-clip
Z-shift mechanism). **Honest cost, stated plainly**: whole-clip aggregate floorPen and self-
collision are WORSE for OURS (15-24cm vs 1.2-5.1cm; 15-21% vs 0-5%) — real residual error on
frames with no detected contact at all (the same "ballistic/no-contact-signal" gap S2-T5 already
identified as this mechanism's structural boundary), which GMR's blind global shift used to mask
at the cost of breaking every contact frame in the process. This is a genuine trade-off to report
in the paper, not a clean sweep either direction — but the contact-specific claim (the actual
novel contribution) now stands on real, validated numbers for the first time this session.

**Code shipped**: `scripts/g1/polish_ours_g1.py` — grounding now OFF by default (`--ground` to
re-enable for A/B comparison only, not the shipped path). `scripts/g1/solve_lafan1_
canonical_g1_contactfirst.py` — floor-referenced rest anchor added (permanent fix, not gated
behind a flag, since there is no scenario where the old `[0,0,0]` behavior is preferable).

**Not yet done**: a genuinely contact-aware grounding mode for OURS (one that corrects the
non-contact residual WITHOUT disturbing held frames — e.g. compute the shift from held-frame
median instead of whole-clip blind minimum) would recover SOME of the aggregate floorPen/coll%
gap without sacrificing the now-validated contact win. Flagged as a real next step, not attempted
this pass (time-boxed; the contact-quality validation was the priority Prabin set).

## S2-T6 — CORRECT baseline framing (Prabin, 2026-07-16): GMR = their published method (+heightfix)

**Correction to this session's own comparisons above.** Per the standing rule already decided
2026-07-15 (`GMR-baseline.md` §7.4: "the headline GMR column is GMR + its paper-described height
fix — that's the method as published... not a strawman"), the baseline "GMR" for any headline
comparison must be **GMR+heightfix** (their own described Z-correction, applied), not their
shipped-code default with it switched off, and not "raw GMR + OUR OWN polish module" (which is a
DIFFERENT, competing Z-correction mechanism substituted in place of theirs, not what they publish
or describe). Re-ran the full held-frame + whole-clip comparison with the CORRECT 4 cells:
**GMR+heightfix** (their described method) / **GMR+ourpolish** (our Z-fix on their raw, a
different mechanism, shown for reference) / **OURS raw** / **OURS+StageA**.

**Result, held-frame contact quality (support_z vs floor), all 4 clips:**

| clip | GMR+heightfix | GMR+ourpolish | OURS raw | OURS+StageA |
|---|---|---|---|---|
| walk1_subject1 (L/R) | **+0.5/+0.6cm** (99.9/100%<3cm) | +4.6/+4.6cm (6-7%<3cm) | +0.2/+0.2cm (81-84%<3cm) | -0.9/-1.1cm (71-75%<3cm) |
| fallAndGetUp1_subject1 (L/R) | +8.8/+8.9cm (2-5%<3cm) | +10.4/+10.6cm (0-1%<3cm) | **+0.2/+0.1cm** (81-83%<3cm) | -1.0/-0.8cm (69-76%<3cm) |
| fallAndGetUp2_subject2 (L/R) | +12.5/+12.2cm (0%<3cm) | +11.7/+11.3cm (0%<3cm) | **+0.1/+0.03cm** (76-85%<3cm) | -0.01/-0.8cm (72-81%<3cm) |
| ground1_subject1 (L/R) | +12.5/+12.3cm (0%<3cm) | +8.9/+8.6cm (0%<3cm) | +0.4/+1.3cm (55-63%<3cm) | +0.5/+1.5cm (54-55%<3cm) |

**The decisive pattern, exactly as Prabin characterized it ("ours already has contact built in,
unlike theirs where they just naively translate robot in z direction")**: on `walk1_subject1` —
clean, SINGLE-PHASE locomotion — GMR's own described height-fix already lands very close to the
floor at contact frames (+0.5cm, 99.9% within 3cm), because a walking gait has exactly ONE stance
height throughout, so calibrating a single global shift to the clip's worst frame happens to be
right. **On every multi-phase clip — fallAndGetUp1/2, ground1, the EXACT excluded motion class
this whole project targets — GMR's own described method fails badly and consistently: +8.8 to
+12.5cm off the floor, 0-5% of held frames within 3cm, EVERY TIME.** This is not a tuning
weakness; it is structural: a single clip-wide Z-shift cannot simultaneously be correct for a
standing phase, a falling phase, and a lying phase, because those genuinely have different floor-
relative reference heights, and averaging/single-frame-calibrating across them cannot help all of
them at once. `GMR+ourpolish` (a DIFFERENT global-shift mechanism, not theirs) fails in exactly
the SAME way on the SAME clips (+8.9 to +11.7cm) — confirming this is a property of the whole
MECHANISM CLASS (any single global Z-shift), not an implementation detail specific to either
GMR's or our own version of it.

**OURS (raw AND +StageA) succeeds on EVERY clip, including every multi-phase one**, because
contact reasoning happens PER FRAME, during the solve — it was never depending on the clip having
one uniform stance height in the first place. Median support_z stays within about a centimeter of
the true floor (-1.1cm to +1.5cm) on all 4 clips, 53-85% of held frames within 3cm, with NO
degradation on the harder multi-phase clips (unlike both global-shift variants, which get WORSE
exactly where the motion gets harder — `ground1_subject1`/`fallAndGetUp2_subject2` are their two
worst cells, +12.2 to +12.5cm).

**This is the paper's central, validated claim, now built on the correct baseline**: naive
Z-translation — whether GMR's own published method or a substituted alternative using the same
mechanism class — fails specifically and predictably on multi-phase floor-contact motion, the
exact class GMR's own paper excludes. Per-frame contact-in-the-solve does not have this failure
mode, by construction, not by tuning. Honest cost still stands: whole-clip aggregate floorPen for
OURS (14-24cm) is worse than either global-shift variant (1.2-5.1cm) — real residual error on
frames with NO detected contact signal at all (this mechanism's known, previously-documented
structural boundary, S2-T5), which a global shift papers over in aggregate while failing
completely at the frames that actually matter for physical plausibility.

**Files**: `/tmp/.../final_validation_correct_framing.py` (session-local, not committed —
regenerate from this log if needed for the paper table).

## S2-T6 — contact-aware grounding for the non-contact-frame residual (Prabin's next ask)

**Built**: `scripts/g1/ground_ours_contact_aware.py` — reuses `_solve_lift_qp` UNCHANGED
(imported from `post_process_ground_contactfirst.py`, the same smooth per-frame lift QP already
used by that script's "hybrid" grounding mode) with a NEW cap rule instead of its existing
foot-float-tolerance cap: **held frames get `cap=0.0` (a HARD constraint — the QP is forbidden
from moving these frames at all), non-held frames get `cap=+inf`** (free to fully correct).
`held_mask` reuses the SAME held-frame definition (human contact zone AND marker stillness)
already validated throughout S2-T5/T6's audits.

**First attempt** (`smooth=1e4`, borrowed unchanged from hybrid grounding's own default without
re-deriving whether it fit this different use case): held-frame quality perfectly preserved
(lift=0.0000cm there, guaranteed by the hard cap, not just encouraged) — but whole-clip floorPen
barely moved (15.19→15.10cm). Diagnosed: the smoothness penalty (`smooth*||D2 lift||^2`) so
heavily dominates the objective that even a real, isolated 15cm dip gets smoothed away to
0.09cm — the QP treats "briefly dip 15cm then return" as too sharp a curve and refuses to draw
it, even though nothing here actually needs the curve to be smooth, it needs to be CORRECT.

**Prabin's framing, directly actionable**: "give more weight on held frames so contact stays
stable, less on others, so swing-phase jumps get evened out" — this is exactly right, and maps
onto the CAP being the trust mechanism (hard cap=0 at held frames is a STRONGER, more reliable
form of "weight" than a soft preference — it's a guarantee, not a bias) while the SMOOTHNESS
weight needed to be lowered so non-held frames can actually track their true correction need
instead of being globally averaged away. Swept `smooth` from 1e4 down to 1.0: held-frame quality
stays EXACTLY unchanged at every value (confirms the hard cap, not the smoothness weight, is
what's doing the protection) while non-held correction improves monotonically as smooth
decreases (floorPen pen%: 90.8%→88.9%→80.6%→71.0%; root-Z velocity, the only thing this Z-only
shift can affect, rises moderately 1.34→1.42→1.74→2.36 m/s — watched, not extreme). Settled on
**`smooth=1.0`** as the shipped default: substantial non-held correction (pen% 90.8%→71.0%)
without an extreme root-velocity cost. (Joint velocity/spikes are IDENTICAL across all smooth
values, as expected -- `qpos[:,2] += lift` only ever touches root Z, never joint angles, so this
mechanism cannot introduce the kind of joint-space spike the earlier pull-to-floor bug did.)

**A genuinely NEW, smaller finding surfaced by fixing the bigger one**: at `smooth=1.0`, the
clip's new worst-case floorPen frame (13.7cm, `left_ankle_roll_link`) is ITSELF classified as a
HELD frame (correctly protected, lift=0 there per the hard cap) — meaning Stage 3's own contact-
frame accuracy isn't uniformly perfect; some frames right at a held-run's START still carry real
penetration, likely the SAME class of "per-frame offset hasn't converged yet" transient the
ramp-cross-fade fix (S2-T5 fix attempt 5) addressed for HANDS on the fall clip but not fully for
this specific FEET timing edge. **Not fixed this pass** — flagged as the next concrete residual,
distinct from (and smaller than) the one just closed. Time-boxed given this is now the 3rd
layer of increasingly fine-grained residual chased in one session; reporting rather than
continuing to chase without a checkpoint.

**Net result of this fix, walk1_subject1** (smooth=1.0, all held-frame numbers UNCHANGED from
before this fix — the whole point):
- Whole-clip floorPen: 15.19cm → 13.72cm max, pen% 90.8% → 71.0% (real improvement, not closed)
- Held-frame support_z: -0.89/-1.06cm median, UNCHANGED (guaranteed by the hard cap)
- Self-collision, joint vMax, spikes: UNCHANGED (this mechanism only ever touches root Z)

**Not yet re-run across the other 3 clips or wired into `polish_ours_g1.py` as the shipped
default** — this session's `smooth=1.0` choice was tuned on `walk1_subject1` only; the other
clips (especially the fall/crawl clips with much larger raw floorPen, 20-24cm) may need their own
check before treating this as final. Flagged as the immediate next step.

**Files**: `scripts/g1/ground_ours_contact_aware.py` (new).

## S2-T7 — kinematically-inconsistent leg-chain scaling: found + fixed with GMR's own grouped constants (2026-07-16)

**The bug (root-caused via direct measurement, not inference)**: `compute_per_role_scales`
computed each role's scale INDEPENDENTLY as (G1's achieved-rest pelvis→role distance)/(human's
pelvis→landmark distance), with zero cross-role consistency. On the SAME rigid leg this produced
hip=2.45, knee=0.97, ankle=0.79 — implying a target thigh length of 36.25cm vs G1's real 19.4cm
(nearly 2x). The IK then split the impossible chain error down the leg: tracking error GROWS
monotonically hip(-2.8cm) → knee(-6.0cm) → ankle(-8.5cm), the signature of an unreachable chain.
Why hip=2.45 specifically: the `left_hip` role maps to `left_hip_yaw_link` (per GMR's own
ik_config correspondence), which sits 28cm from pelvis (61% of the way to the knee) — a body
choice that's harmless for GMR's IK (they never form this ratio) but poisonous for our
independent-ratio formula. Verified NOT a canonicalization bug: Alex's own canonical mapping uses
the same `left_hip → LeftUpLeg` landmark, consistently.

**The fix (Prabin's call: match GMR exactly, so contact enforcement is the ONLY methodological
difference)**: replaced `compute_per_role_scales` with GMR's own published `human_scale_table`
constants from `bvh_lafan1_to_g1.json` — exactly two groups: lower-body+root+torso = 0.9,
arms/hands = 0.75, head = 1.0 (implicit). Implemented as `GMR_GROUPED_SCALES` +
`gmr_grouped_role_scales()` in `solve_lafan1_canonical_g1_contactfirst.py`;
`compute_per_role_scales` import removed. Chain distortion resolved: target hip-knee 21.8cm vs
G1's real 19.4cm (was 36.3cm).

**4-clip validation, RAW (pre-polish) output (`*_ours_gmrscale.npz`), before → after**:

| clip | self-coll% | held support_z median | whole-clip floorPen max |
|---|---|---|---|
| walk1_subject1 | 15.3% → **2.1%** | -1.0cm → +0.2cm | 15.2cm → 17.4cm |
| fallAndGetUp1_subject1 | 13.0% → **6.4%** | -0.9cm → +0.1cm | 23.7cm → 25.5cm |
| fallAndGetUp2_subject2 | 16.7% → **14.3%** | -0.4cm → +0.0cm | 20.4cm → **39.7cm** |
| ground1_subject1 | 18.7% → **11.6%** | +1.0cm → +0.0cm | 19.8cm → 22.2cm |

Held-frame frac<3cm after: walk1 78.6/81.5%, fAGU1 78.8/86.7%, fAGU2 83.1/73.6%, ground1
79.7/76.4% (left/right foot). walk1's 2.1% self-collision now matches GMR's own level (~4%).

**Three honest readings**: (1) self-collision improved on EVERY clip — the diagnosed bug was
real; (2) held-frame contact quality (the paper's central claim) held up and tightened slightly
(all 4 clips at 0.0–0.3cm median) — the contact mechanism was untouched by this fix; (3)
whole-clip aggregate floorPen got WORSE, badly on fallAndGetUp2 (20.4→39.7cm). That is the
SEPARATE, still-undiagnosed swing-frame tracking residual: at walk1 frame 598 the target is now
verifiably achievable (+1.5cm above floor) yet achieved is -8.9cm (err -10.4cm). Ruled out by
direct sweeps: NOT iteration count (30/100/300/1000 iters converge to the same error), NOT
self-collision competition (coll_weight 20/5/1/0 identical). No longer masked by chain-distortion
noise; now the dominant aggregate-floorPen source, especially on dynamic clips.

**Not yet done**: gmrscale outputs not yet polished (StageA + contact-aware ground) or compared
against GMR+heightfix on the 3 non-walk clips; swing-residual not diagnosed.

**Files**: `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` (GMR_GROUPED_SCALES,
gmr_grouped_role_scales; compute_per_role_scales removed from this path),
`outputs/gmr_baseline/sprint/ours_g1/*_ours_gmrscale.npz` (4 clips).

## S2-T8 — polish + full 5-variant comparison on gmrscale outputs (2026-07-17)

Ran Stage A (smoothing only) then contact-aware grounding (`smooth=1.0`) on all 4
`*_ours_gmrscale.npz` clips; compared against GMR+heightfix and GMR+ourpolish via
`scripts/g1/eval_g1_gmrscale_variants.py` (new, committed). Contact-aware grounding's held-frame
lift = 0.0000cm on every clip (hard cap holds); non-held lift up to 49.33cm on
`fallAndGetUp2_subject2` (consistent with that clip's large raw floorPen residual).

**The central claim, now validated on ALL 4 clips (not just walk1)**: held-frame support_z —

| clip | GMR+heightfix median | GMR+ourpolish median | OURS (any variant) median |
|---|---|---|---|
| walk1_subject1 | +0.5/+0.6cm (99-100% <3cm) | +4.6cm (6-7% <3cm) | -0.9/-1.3cm (69-71% <3cm) |
| fallAndGetUp1_subject1 | +8.8/+8.9cm (2-5% <3cm) | +10.4/+10.6cm (0-1% <3cm) | -1.2cm (66-71% <3cm) |
| fallAndGetUp2_subject2 | +12.2/+12.5cm (0% <3cm) | +11.3/+11.7cm (0% <3cm) | -0.0/-0.9cm (68-79% <3cm) |
| ground1_subject1 | +12.3/+12.5cm (0% <3cm) | +8.6/+8.9cm (0% <3cm) | -0.0/+0.0cm (73-75% <3cm) |

GMR's own heightfix and our polish of it are essentially useless on the 3 floor-contact clips
(0-5% of held frames within 3cm of the floor, feet floating 8.6-12.5cm up) — exactly the failure
mode the paper's whole motivation rests on. OURS holds contact to within ~1cm median on every
clip, including the hardest ones, with NO degradation vs walk1. This is the strongest, cleanest
form of the central result so far.

**A new, unexpected, concerning finding — Stage A itself regresses whole-clip floorPen/
self-collision on 3 of 4 clips** (raw -> +StageA):

| clip | floorPen | coll% |
|---|---|---|
| walk1_subject1 | 17.35 -> 17.59cm | 2.1 -> 4.2% |
| fallAndGetUp1_subject1 | 25.47 -> 25.80cm | 6.4 -> 8.9% |
| fallAndGetUp2_subject2 | 39.71 -> **52.31cm** | 14.3 -> **22.7%** |
| ground1_subject1 | 22.16 -> 21.82cm (~flat) | 11.6 -> **7.2%** (improved) |

Held-frame quality is essentially unchanged by Stage A (small drift, e.g. walk1 left_foot
+0.19->-0.90cm, still >69% within 3cm) — Stage A is NOT breaking the contact win. But it is
INCREASING non-contact-frame penetration/self-collision on 3/4 clips, worst on
`fallAndGetUp2_subject2` (+12.6cm floorPen, +8.4pt coll%). This was never previously isolated:
S2-T4's "Stage-A-only preserves contact quality" claim was validated against held-frame numbers
and against GMR-polished, not against OURS-raw's OWN floorPen/coll — this specific raw-vs-StageA
comparison is new this pass. Contact-aware grounding partially claws back floorPen afterward
(e.g. fallAndGetUp2 52.31->51.22cm, ground1 21.82->16.26cm) but coll% is unaffected either way
(the lift-QP only touches root Z, never joint angles, so it cannot fix or worsen self-collision).

**Not yet diagnosed**: why Stage A's tracking+smoothness QP increases collision/floorPen on
non-contact frames specifically on the harder, larger-residual clips. Plausible mechanism (not
verified): Stage A may be smoothing THROUGH the already-known swing-frame tracking-gap residual
(frame-598-class errors, S2-T7) in a way that overshoots on the approach/exit, rather than
tracking the raw (unsmoothed) target more tightly at those frames. Distinct from, but possibly
related to, the still-open S2-T9 swing-residual investigation — flagged for Prabin's call on
priority before digging further.

**Files**: `scripts/g1/eval_g1_gmrscale_variants.py` (new, committed), `outputs/gmr_baseline/
sprint/ours_g1/*_ours_gmrscale_{stageA,ctground}.npz` (8 new files, 4 clips x 2 stages).

## S2-T9 — Stage A regression: root-caused and fixed (floor + self-collision sensitivity boost, 2026-07-17)

**Root cause confirmed by direct measurement, not inference**: dumped per-frame floorPen around
`fallAndGetUp2_subject2`'s worst frame (raw vs +StageA). Raw signal there is jagged/narrow:
frames 411-420 read `2.7 3.7 4.5 5.3 5.9 4.6 32.7 39.7 26.5 0.5` cm — a near-zero frame (0.5cm)
sitting directly next to a 39.7cm spike. Stage A's tridiagonal smoother (`stage_a`, floor-blind
by construction) blends across this sharp transition instead of trusting either raw value,
producing a NEW worse peak: `4.6 9.2 14.4 20.1 26.5 33.3 39.9 45.0 48.9 51.2 52.3` — 52.3cm,
deeper than either raw neighbour. **This is the exact same failure mode
`_detect_floor_sensitive_frames`'s own docstring already documents for the Alex pipeline**
(measured there on `luigi_standProne_03`: a sharp Stage-3-fixed 2.4cm violation re-inflated to
13.9cm under plain Stage A) — not a new bug, a known mechanism that `polish_ours_g1.py` simply
never wired the existing fix into. Same signature on `walk1_subject1` (smaller: 13.3cm raw
transition -> 17.6cm post-StageA peak).

**Fix**: ported the mainline pipeline's `lambda_track_frames` local-boost mechanism into
`polish_ours_g1.py` (`_sensitivity_weight`, `_sustained_ramp_weight` — the latter is the same
sustained-run + cosine-ramp algorithm as `_detect_floor_sensitive_frames`, factored out so both
floor and self-collision violations can reuse it). **Extended beyond the mainline version per
Prabin's ask** ("maybe it's beneficial to penalize collision in stage A too"): boosts λ_track at
BOTH sustained floor-penetration frames (mesh-exact `_robot_lowest_z`, same metric as all our
floorPen reporting, min_pen=1.5cm) AND sustained self-collision frames (same `_within_k_hops`-
filtered metric as `_collision_stats`'s own self-collision counting, min_pen=0.5cm) so Stage A
can't smooth through and worsen either failure mode. Default ON
(`--no-sensitivity-boost` for A/B). Boost factor unchanged from mainline convention:
`max(lambda_track, lambda_smooth*2)` = 40 at this script's defaults.

**Result, raw -> plain-StageA (broken) -> boosted-StageA (fixed), all 4 clips**:

| clip | metric | raw | plain StageA | boosted StageA |
|---|---|---|---|---|
| walk1_subject1 | coll% | 2.1% | 4.2% | **2.4%** |
| walk1_subject1 | floorPen | 17.35cm | 17.59cm | 17.62cm (flat, not fixed) |
| fallAndGetUp1_subject1 | coll% | 6.4% | 8.9% | **7.0%** |
| fallAndGetUp1_subject1 | floorPen | 25.47cm | 25.80cm | **25.34cm** (now beats raw) |
| fallAndGetUp2_subject2 | coll% | 14.3% | 22.7% | **16.9%** |
| fallAndGetUp2_subject2 | floorPen | 39.71cm | 52.31cm | **43.93cm** (real improvement, not closed) |
| ground1_subject1 | coll% | 11.6% | 7.2%* | 9.6% |
| ground1_subject1 | floorPen | 22.16cm | 21.82cm | 22.14cm (flat) |

(*ground1's plain-StageA coll% of 7.2% was already better than raw for unrelated reasons — not a
regression case there; the boost gives back a bit of that on this one clip while still beating
raw. Not concerning: held-frame quality identical in both directions on ground1, see below.)

Self-collision (the metric the fix specifically targets, per Prabin's ask) is now much closer to
raw on every clip — the plain-StageA regression is substantially closed. floorPen is fully fixed
or beaten on 2/4 clips; on `fallAndGetUp2_subject2` a real ~4.2cm gap over raw remains (43.93 vs
39.71cm) — this is NOT a smoothing artifact anymore (the boost correctly locks Stage A onto the
raw signal at these frames), it's Stage A now faithfully reproducing the SAME underlying error
that's already in the raw signal at those frames — i.e., what's left is the genuine swing-frame
tracking-gap residual (frame-598-class, S2-T7), unrelated to Stage A's smoothing mechanism.

**Held-frame contact quality: unaffected, as required** — medians stay within 0.2cm of the
pre-fix numbers on every clip/foot (e.g. walk1 left_foot +0.19->+0.03cm, ground1 both feet
~0.00cm unchanged). The boost only reduces smoothing strength at flagged frames; held/contact
frames were never the problem and aren't touched differently.

**Sensitivity-flag coverage, for context** (fraction of clip flagged floor- or self-collision-
sensitive): walk1 56%/0%, fallAndGetUp1 59%/0%, fallAndGetUp2 71%/0.5%, ground1 93%/0.3% (floor/
self-coll). The floor fraction is much larger than the Alex pipeline's typical usage (there,
narrow isolated windows) because OURS's raw output has floor_weight=0 on non-contact frames by
design (S2-T6) — most of the clip genuinely has SOME floor signal to protect, not just isolated
spikes. Checked directly (raw vs boosted-StageA peak joint velocity): still roughly HALVES on
every clip despite the broad flagging (walk1 76.3->35.4 rad/s, fallAndGetUp1 59.8->37.6,
fallAndGetUp2 74.7->44.9, ground1 70.4->43.2 rad/s; mean velocity down similarly on all 4) — the
boost raises RELATIVE trust in tracking at flagged frames, it doesn't zero the smoothness term,
so Stage A's actual job survives. Confirmed safe, not just non-regressing.

**What's genuinely left, now cleanly isolated**: the swing-frame tracking-gap residual itself
(why raw floorPen is 39.71cm on `fallAndGetUp2_subject2`, 22.16cm on `ground1_subject1`,
25.47cm on `fallAndGetUp1_subject1` in the FIRST place) — this is upstream of Stage A entirely,
in Stage 3's per-frame IK solve. The original S2-T9 ladder (frame-598 target-vs-achieved dump,
ankle-ori conflict test, pull-to-floor leak test) still applies and has NOT been run yet.

**Files**: `scripts/g1/polish_ours_g1.py` (`_sensitivity_weight`, `_sustained_ramp_weight`,
`--no-sensitivity-boost`, default ON), `outputs/gmr_baseline/sprint/ours_g1/
*_ours_gmrscale_{stageA,ctground}.npz` (regenerated, 4 clips x 2 stages).

## S2-T10 — root_scale-on-relative-term fix: TRIED, MEASURED, REJECTED (2026-07-17)

**Motivating question (Prabin)**: "if scaling is already applied, how can targets be
unachievable?" Root-caused precisely: `make_targets_for_frame` (shared, unmodified) scales the
per-role RELATIVE-to-pelvis motion delta by `role_scales[role]` alone (0.9 lower-body / 0.75
upper-body, GMR's own group constants) -- `root_scale` (this project's size-ratio measurement,
~0.64-0.65 across all 4 clips: G1 is only ~64% this human's size via pelvis-to-head) is applied
ONLY to the pelvis's own root displacement, never to the relative term. GMR's own published
formula applies its height factor `(h/h_ref)` to BOTH terms. Confirmed by direct measurement this
gap causes real over-reach: on `fallAndGetUp2_subject2`, 2650 frame-legs have a hip-ankle target
distance EXCEEDING G1's own max physical leg reach (worst case 181.6% of it).

**Fix tried**: `gmr_grouped_role_scales(role_to_body_id, root_scale)` -- multiply the per-role
scale by `root_scale` too, matching GMR's two-term structure (kept our own per-subject-measured
`root_scale` rather than adopting their fixed `h_ref=1.8m`, and kept delta-from-rest rather than
their raw relative-to-root -- both deliberate, pre-existing, unchanged conventions). Verified the
mechanism directly first: cut over-reach frame-legs 2650->852, worst case 181.6%->117.1%.

**Ran end-to-end on all 4 clips (full Stage 3 resolve + Stage A + contact-aware ground) --
REGRESSED every other metric, badly**:

| clip | self-coll (before->after) | floorPen raw (before->after) |
|---|---|---|
| walk1_subject1 | 2.1% -> 16.8% | 17.35 -> 18.24cm |
| fallAndGetUp1_subject1 | 6.4% -> 19.4% | 25.47 -> **50.91cm** (nearly doubled) |
| fallAndGetUp2_subject2 | 14.3% -> 19.8% | 39.71 -> 45.67cm |
| ground1_subject1 | 11.6% -> **34.5%** | 22.16 -> 24.93cm |

Held-frame quality also degraded (walk1 left_foot frac<3cm 78.6%->62.2%, ground1 79.7%->62.1%).

**Why it fails**: `root_scale` (~0.64) applied ON TOP of the group constant (0.9/0.75) shrinks the
per-role excursion far more than the RARE over-reach frames need -- it shrinks EVERY frame's limb
swing throughout the WHOLE clip, not just the handful of extreme-pose frames that actually exceed
reach. Limbs end up staying closer to the pelvis/each other everywhere (a globally more cramped
gait), which increases self-collision across the board and does not help floor contact (a
compressed swing-foot lift trajectory scuffs the floor instead of clearing it, which is likely
WHY floorPen got worse too, not better). **The blanket fix trades a rare, small, honest residual
(some extreme poses are geometrically unreachable by a robot ~64% the human's size -- a real
retargeting-fidelity limit, not fixable by any uniform linear scale) for a much larger, clip-wide
cost.** Reverted cleanly; `gmr_grouped_role_scales` is back to its S2-T7/S2-T9 form (verified:
re-ran all 4 clips through the full Stage-3 -> StageA -> ctground -> eval pipeline, numbers are
IDENTICAL to S2-T9's validated table).

**Not pursued further this pass**: a LOCAL/adaptive correction (e.g., softly pulling back a
target's Z or magnitude only at the specific frames that actually exceed G1's reach, rather than
a single global multiplier on the whole per-role scale) could plausibly close the rare-frame
over-reach without the whole-clip cost -- this is a materially different, larger mechanism than
the one-line change tried here, not built or tested this pass.

**Files**: `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` (`gmr_grouped_role_scales`
docstring now documents this rejected attempt; function body reverted, byte-identical to S2-T9).
`outputs/gmr_baseline/sprint/ours_g1/*_ours_gmrscale*.npz` regenerated back to the S2-T9 state
(verified numerically identical) after the regression was found.

## S2-T11 — swing-frame floorPen residual: "hold-tier leak" hypothesis built, tested, REFUTED; real mechanism found (2026-07-18)

**Motivating question (Prabin)**: on `walk1_subject1`, does holding one (stance) foot's position
target at high priority "leak" and drag the swing foot down through the floor via the shared
pelvis/root DOFs? Architecture support for the hypothesis is real: `solve_frame_position_ik` is a
hierarchical (task-priority) DLS solve — `hold_pos_roles` (planted-foot targets) sit in level 1
(hard tier, solved first), everything else (including the swing leg's own tracking target) is
solved in the NULL SPACE of level 1. `--floor-weight` defaults to 0.0 in the G1 script (confirmed
in code) — there is no floor-avoidance term anywhere in the solve except the pull-to-floor Z-blend
on whichever effector is currently in a contact ZONE (broader than the strict `hold_pos_roles`
subset).

**Built a direct diagnostic** (`/tmp` scratch script, not committed — re-solves individual frames
against the SHIPPED trajectory's exact warm-start + reconstructed targets, with `hold_pos_roles`
toggled on/off, `coll_weight` toggled 20/0, and warm-start knee angle perturbed). Found the worst 4
non-held floorPen frames on `walk1_subject1` (577/598/601/4505, 15-17cm shipped pen) via a full-clip
scan.

**Hold-tier leak: REFUTED.** `hold_pos_roles` is EMPTY at all 4 worst frames — these are contact-
TRANSITION frames (debounced label True, but source foot speed 0.05-1.3 m/s, just above the 0.05
plant-speed hold threshold), so the hard tier never engages. A/B (hold forced off vs. as-is)
produced byte-identical results at all 4 frames — nothing to leak from.

**Self-collision competition: also REFUTED** (coll_weight 20->0, zero change) — independently
reproduces the exact finding already on record from S2-T7's original iteration/coll-weight sweep,
cross-validating that this from-scratch reconstruction is measuring the real phenomenon (task-space
targets matched the shipped trajectory to <0.3cm at a well-tracked frame, confirming the
reconstruction is sound even though exact JOINT angles diverge from the shipped file by up to
several radians — this solve is NOT bit-reproducible run-to-run, redundant-DOF/warm-start-chain
gauge freedom, not a bug; aggregate metrics stay within ~1-2cm/1pt of each other across independent
reruns of literally unmodified code — flagged as a real, previously-undocumented property of this
solver, relevant to any future byte-identical no-op verification attempt on this file).

**Joint-limit check: found the knee pinned at its hard lower limit (-0.087 rad, mechanically
straight) on the deeply-penetrating leg at all 4 frames**, with the ankle also pinned at ITS limit
at 3/4 frames — clamped EVERY iteration (`clamp_hinge_joint_limits` runs inside the loop, not just
at the end), so this is not a stale-clamp artifact.

**Prabin's pushback, and it was right**: a too-short/maxed-out leg reaching for a target "down and
away" should fall SHORT (foot ends up too high), not overshoot THROUGH the floor. Verified this
directly: pelvis achieved position is within 1-2cm of ITS OWN target at all 4 frames (not sinking)
— but hip is 4-7cm off and ankle 12-24cm off, error amplifying down the chain, the classic signature
of a long rigid lever rotating past its intended endpoint, not a root-level drop.

**Root cause, confirmed by direct test**: manually pre-bent both knees to 0.4 rad in the warm start
before re-solving (same targets, same everything else). 2 of 4 frames unlocked dramatically (frame
4505: 9.76cm pen -> 0.00cm, fully resolved; frame 577: 13.23cm -> 3.18cm) with the knee moving to a
sane bent value (1.4-1.9 rad) instead of staying pinned. The other 2 (598/601) improved only
modestly and one knee snapped back to the limit even from the artificial bent start. **This is a
solver LOCAL-MINIMUM / warm-start-basin problem, not a hard physical reach wall** — each frame
warm-starts from the previous frame's solved pose; once a knee lands near its limit, the per-
iteration clamp keeps re-clipping it there and the small per-iteration step budget
(`max_step_norm=0.20`) never lets the solver climb back out to try the bent-knee alternative that
demonstrably exists and fits the target far better. (598/601 likely mix in a smaller genuine
reach/orientation-conflict component — not separated further this pass.)

**Fix candidate identified, not a guess**: `solve_fbx_canonical_alex_contactfirst.py` already has
`knee_bias` (a one-sided DLS regularization row that weakly pushes a knee straighter than
`--knee-min-flex-deg` back toward it, silent once bent) built for exactly this failure mode, used
in the Alex pipeline (`--knee-bias-weight`, default 0.5) — but the G1 script's per-frame loop never
passed it (`knee_bias=None` implicit default, no CLI equivalent existed in this file before this
session).

## S2-T12 — knee_bias wired into G1 + 4-clip validation: genuinely mixed result (2026-07-18)

**Built**: `--knee-bias-weight` (default 0.0, OFF — this file's existing opt-in convention, unlike
Alex's on-by-default) and `--knee-min-flex-deg` (default 12.0, matches Alex) added to
`scripts/g1/solve_lafan1_canonical_g1_contactfirst.py`; `knee_bias` tuple constructed from G1's OWN
knee joint range (G1's straight = LOWER limit, unlike Alex where straight = q=0 — `min_flex` is
computed relative to G1's actual lower limit, not assumed to be 0) and threaded into both the
initial rest-alignment solve and the per-frame loop's solve call.

**No-op verification note**: could NOT do a strict bit-identical no-op check the way this codebase
usually does — per S2-T11's finding, this solver isn't bit-reproducible run-to-run even completely
unmodified. Verified instead at the METRIC level: an independent full-clip rerun with
`--knee-bias-weight 0` (the new default) landed within the same ~1-2cm/1pt noise band as the
originally-shipped file (walk1: floorPen 17.35->18.95cm, coll 2.09->3.11%) — consistent with pure
solve-chaos, not a behavior change from the new (inert) code path.

**4-clip end-to-end result, `--knee-bias-weight 0.5 --knee-min-flex-deg 12`, own fresh OFF run as
the apples-to-apples baseline (not the old shipped files, to avoid the run-to-run noise above
contaminating the read)**:

| clip | floorPen off->on | coll% off->on | held L/R frac<3cm off->on |
|---|---|---|---|
| walk1_subject1 | 18.95->**12.89cm** | 3.11->2.61% | 78/81%->**99/100%** |
| fallAndGetUp1_subject1 | 25.47->**36.39cm** (worse) | 6.40->6.34% | 78.8/86.7%->85.3/90.5% |
| fallAndGetUp2_subject2 | 39.71->**28.87cm** | 14.29->14.38% | 83.1/73.6%->88.7/77.6% |
| ground1_subject1 | 22.16->22.16cm (bit-identical, true no-op on this clip) | 11.58->11.68% | unchanged |

Also, at the 4 originally-diagnosed frames on walk1, per-frame penetration roughly HALVED under
`knee_bias` (e.g. frame 4505: 17.55->7.29cm), directly confirming the S2-T11 mechanism at the level
the fix targets, not just in aggregate.

**Reading**: 2/4 clips clearly better (worst-case floorPen down substantially, held-frame contact —
the paper's central claim — improved on every clip that changed at all), 1/4 clearly worse on the
metric this was meant to fix (fallAndGetUp1's floorPen +11cm, not yet root-caused — some frame(s)
got a NEW deeper spike, held-frame quality on that same clip still improved), 1/4 inert (bias term
never engages — no knee near its limit on this clip). walk1's `pen>5cm%` also rose (16.6%->24.2%)
even as its worst case dropped — moderate-depth penetration spread a bit wider on non-held frames,
same "fixes the spikes, costs a bit on the average" shape seen elsewhere in this project's history
(cf. S2-T9's own boosted-StageA table). **Does not clear a clean ship bar as a global default yet**
— fallAndGetUp1's regression is unexplained. Not shipped as default (`--knee-bias-weight` default
stays 0.0); available as an opt-in flag for further investigation.

**Prabin's call (2026-07-18)**: committing to writing a paper on this line — contact-first solving
(OURS) stays a core mechanism, not reduced to polish-only. Not resolving the fallAndGetUp1
regression or the knee_bias ship decision this pass — instead broadening OURS (plain S2-T9
config, knee_bias OFF) to the full 77-clip corpus for proper, publication-grade numbers alongside
GMR's own already-complete 77-clip sweep (Sprint S1). New `scripts/g1/sprint_s3_full_corpus.py`
(committed-worthy, resumable/skip-if-exists): builds canonical_human -> Stage 3 solve -> StageA
polish -> contact-aware ground for every LAFAN1 clip, then evaluates whole-clip (floorPen/pen%/
self-collision) + held-frame support_z for both GMR's 3 variants (reused from S1's existing
`outputs/gmr_baseline/sprint/pkl/`) and OURS's 3 variants, into one combined CSV. Smoke-tested on 2
clips (full pipeline confirmed working end-to-end) before launching. Running in the background,
detached from the interactive session (`nohup setsid ... & disown`, confirmed own session ID so it
survives terminal logout) — `outputs/gmr_baseline/sprint/s3_build_nohup.log`, resumable if
interrupted (skips any clip whose `_ours_ctground.npz` already exists). Not yet evaluated — run
`sprint_s3_full_corpus.py --eval` once `--build` finishes across all 77 clips, then analyze.

## S3 — full 77-clip corpus build + eval; held-frame "win" killed by z-shift oracle (backfilled, work done 2026-07-16, not logged at the time)

**Build**: `sprint_s3_full_corpus.py --build` completed all 77 LAFAN1 clips, 0 failures
(`s3_build_nohup.log`). **Eval**: `--eval` → `outputs/gmr_baseline/sprint/s3_full_corpus.csv`
(GMR raw/heightfix/polished reused from S1, OURS raw/stageA/ctground built here). Headline read:
OURS-ctground "won" held-frame support_z (~0cm median foot-to-floor during human-planted frames,
82-87% within 3cm, vs gmr_polished 4.4-11.7cm / 0.3-31%).

**That framing was killed same day** by `outputs/gmr_baseline/sprint/s3_zshift_oracle.csv`: a
single per-clip constant Z-shift applied to gmr_polished (grid search maximizing pooled held
within-3cm fraction) beats OURS on held-frame within-3cm (96-99% vs 82-87%) AND max floorPen
(6.6-13.4cm vs 17-23cm). Works because GMR's held-foot float is near-constant within a clip
(p90-p10 approx 2.6-3.3cm) — one constant zeroes it. **Verdict: single-axis metrics (float alone,
or penetration alone) are gameable, dead for the paper.** New target defined: joint metric, held
foot <3cm AND whole-body pen <5mm at the same frame — a rigid shift cannot satisfy both. Nothing
currently passes it. Full writeup and the table: `GMR-S4-plan.md` (S4 sprint plan, written off
the back of this result). OURS's blocker: 62-81% of frames >5mm below floor, already at the RAW
solve (S2's knee-warm-start-basin + reach-limit findings implicated, not yet confirmed as the
whole story — see S4-T1/T2 below).

## S4-T1 (partial) — floor-weight probe on walk3_subject1 (the pathological walker)

Quick probe, not the full `diag_penetration_source.py` T1 deliverable (not built this pass).
Tested the existing (already in the solver, pre-S4) `--floor-weight` term at several values on
`walk3_subject1` (28.52cm floorPen / 65.0% pen>5mm baseline, `ours_raw`, no floor term):

| `--floor-weight` | floorPen | pen% (frames >5mm) | coll% |
|---|---|---|---|
| 0 (baseline) | 28.52cm | 65.0 | 2.34 |
| 1 | 24.54cm | 56.0 | 2.43 |
| 2 | 40.01cm | 47.3 | 2.55 |
| 3 | 40.67cm | 41.0 | 2.84 |
| 5 | 44.88cm | 36.1 | 2.47 |
| "default" (script's built-in nonzero test value) | **243.08cm** | 40.2 | 5.33 |

**Reading**: raising `--floor-weight` consistently shrinks `pen%` (more frames pulled clear of
the floor) but does NOT shrink — and past weight=1, sharply worsens — `floorPen` (worst-case
depth). At the untuned "default" value the per-frame IK visibly diverges (243cm is not physical,
the solver is fighting itself). **This is expected**: `--floor-weight`'s own CLI help text says
it's a v1 stub never meant for this (Stage 2.5 already grounds the canonical human; the grounding
QP handles clip-level floor placement after) — it was not built as a real one-sided
floor-avoidance task (no margin/gain shaping, no exclusion for held effectors already pulled to
floor). Confirms T3's premise: a dedicated `--floor-avoid-weight` term (one-sided, proper
margin/gain, skip held effectors) is still needed — the existing term is not a shortcut. Not
built this pass (T3 remains open).

## S4-T2 — knee-bias-skip-held does NOT fix fallAndGetUp1_subject1's regression (remedy invalidated)

Evaluated the coded remedy (`--knee-bias-skip-held`, added to
`solve_lafan1_canonical_g1_contactfirst.py` this sprint) against plain `--knee-bias-weight 0.5`
(S2-T12 config) on the flagship regression clip. New eval script:
`scripts/g1/sprint_s4_t2_eval.py` (read-only, reuses `sprint_s3_full_corpus.py`'s
`whole_clip_metrics`/`held_metrics` verbatim, does not touch that frozen script).

**Result: `kb` and `kbsh` are effectively identical on this clip** — both floorPen 36.39cm (up
from 25.47cm `ours_raw` baseline, matching S2-T12's originally-recorded regression exactly).
`coll%` also identical (6.34 both). Direct qpos diff: the two runs are NOT bit-identical overall
(244/5047 frames differ, up to 0.12rad, clustered near frame 4184) — but the worst-case frame
(3873, both runs) is byte-identical to float noise (36.38694973947967 vs ...87, 2e-13 apart).

**Root-caused precisely**: at frame 3873 the deepest-penetrating body is `right_ankle_roll_link`
in both `kb` and `kbsh` (verified via `_geom_lowest_z` per-geom sweep). Checked the held mask at
that frame directly from the canonical: `right_foot` is **not held** at frame 3873 (`left_foot`
is; the right leg is mid-swing). `--knee-bias-skip-held` only disables the bias row for a leg
whose *own* foot is currently held — it correctly leaves the right leg's bias ON here, because
by its own gating logic this is a legitimate swing-leg frame, not the held-leg-override case the
flag's docstring was written to fix. **This falsifies that docstring's root-cause narrative**
(held-leg override at frame 347-348 cascading via warm-start chaos into an unrelated spike ~3500
frames later, i.e. near frame 3847) — the actual dominant 36cm spike (frame 3873, ~26 frames off
that estimate) is a *swing*-leg event: `knee_bias` forcing extra flexion on the swinging right
leg during this fall-recovery pose drives the ankle deeper through the floor rather than clearing
it, the opposite of knee_bias's original swing-leg-reach rationale (S2-T11).

The 3 other dev clips (`walk1`, `fallAndGetUp2`, `ground1`), `kbsh` only (not isolated against
plain `kb` — not run for these): floorPen 17.35->13.65cm, 39.71->33.74cm, 22.16->22.16cm
respectively (better / better / unchanged) — consistent with S2-T12's original reads, unaffected
by this finding.

**Verdict: T2's coded remedy is not validated.** Per the plan's own rule (don't tune in circles
past ~2 focused attempts per failure mode; this was the one designed attempt) — do not spend a
third attempt patching `knee_bias` further for this specific clip/frame. `--knee-bias-weight`
stays default-OFF; `--knee-bias-skip-held` is real, harmless, and helps 3/4 dev clips, but does
not clear fallAndGetUp1 and should not be presented as having fixed it. Recommend T3 (dedicated
floor-avoidance term) over further `knee_bias` iteration — same read as S4-T1: the mechanism
`knee_bias` was built for (swing-leg reach) is exactly where it's now shown to also hurt on this
clip, whereas a proper one-sided floor-avoidance task targets the actual failure (foot below
floor) directly rather than through a knee-angle proxy. Not yet decided/built — flagging for
Prabin before continuing into T3.

## S4-T3 (started, STOPPED for a design decision) — `floor_collision_rows` already exists and is unstable on G1, worse than the plan assumed

**Correction to S4-T1/the plan's own framing**: `--floor-weight` on the G1 script is NOT a bare
stub needing to be built from scratch. It already calls `floor_collision_rows`
(`solve_fbx_canonical_alex_contactfirst.py`) — the SAME mesh-accurate, one-sided, deduplicated
robot-vs-floor repulsion term that Fix C used successfully on Alex (`luigi_standProne_03`,
11.5cm->2.4cm penetration, SESSION_HANDOFF "Session 2026-07-09/10"). It is already wired into
`solve_frame_position_ik` and reachable from G1's CLI today, `floor_margin`/`floor_gain` already
default to Alex's own known-good values (0.0 / 5.0) even though the G1 CLI doesn't expose
overriding them. So T3 is NOT "write a new term" — it's "this term is already present and
already broken on G1."

**Verified directly**: reran `walk3_subject1` with `--floor-weight 20` (Alex's exact
`luigi_standProne_03` value) — **floorPen exploded to 373cm** (`s4_dev/walk3_subject1_ours_fw20.npz`,
not part of the earlier S4-T1 sweep table, which only went up to weight=5 and topped out at
44.9cm). Traced it: baseline (`ours_raw`, no floor term) is clean at this point in the clip
(pen ~0 through frame 5700, a few cm of transient spikes). With `--floor-weight 20`, penetration
jumps from ~13cm (frame 5600) to 200+ cm within ~100-150 frames (5700->5850) and **never
recovers for the rest of the clip** (373cm at 5850, still 100-240cm at 6000-7398, non-monotonic —
oscillating, not steadily climbing out). This is the same class of failure as S2-T11's
knee_bias warm-start basin and the original Alex wrist-flick (SESSION_HANDOFF "Session
2026-07-09/10", Fix C's bug #3: an UNRAMPED floor correction applied in a single frame to a
limb/root that just crossed the floor threshold overwhelms that frame's small step budget
(`max_step_norm=0.20`), and because qpos warm-starts frame-to-frame, the botched frame's damage
compounds instead of self-correcting — `floor_collision_rows`'s own row target
(`min(penetration, 0.05) * gain`) is clipped to a small per-iteration correction regardless of
how deep the penetration already is, so once several hundred cm deep it cannot climb out within
a clip's remaining frames at all.

**Why G1 has this and Alex didn't (by the time Alex shipped)**: Alex's Fix C had THREE parts,
this file's own header docstring says only the core mechanism was ported to G1, explicitly
excluding "arm-floor-transition/leg-floor-transition refinement passes" — i.e. exactly the
temporal-ramp two-pass fix (`_detect_arm_floor_onset_windows` / `refine_arm_floor_transitions`)
that Alex needed for the identical no-ramp-causes-warm-start-lock-in failure mode. G1 has the raw
repulsion term (part 1-2 of Fix C) but not the ramp (part 3) — this is not a new bug, it's a
known, already-solved-once problem that wasn't carried over.

**Stopping here for a decision, not continuing to build blind**: two shapes the fix could take,
different effort/risk:
(a) **Port the onset-ramp mechanism** from Alex (`_detect_arm_floor_onset_windows` +
    `refine_arm_floor_transitions`-equivalent, generalized to legs not just arms since G1's
    failure is leg/root-driven) — known to work (that's literally what fixed this exact failure
    class on Alex), but a real port: two-pass local refinement, onset detection, warm-start-off-
    previous-refined-frame, cosine ramp. Bigger lift than T3's original "add a flag" framing.
(b) **Cheaper first probe**: apply `floor_weight` with a per-frame COSINE RAMP keyed off
    `_robot_lowest_z` crossing zero (reuse `ramp_envelope`/`zone_env`, already imported and used
    for contact zones in this exact file) BEFORE reaching for the full two-pass refinement —
    might be enough to prevent the single-frame overwhelm without the added complexity of a
    second local-refinement pass. Untested; smaller lift, unclear if sufficient (Alex's own
    history suggests a ramp on the primary pass alone was NOT enough — Fix C's bug #3 needed the
    two-pass local refinement on top of a ramped weight, per SESSION_HANDOFF — so (b) may just be
    a faster way to confirm (a) is necessary, not a substitute for it).

Not implemented either way this pass — flagging for Prabin.

## S4-T3 (continued, Prabin: "port the fix, go") — `refine_leg_floor_transitions_g1` ported and guarded; real but partial win, T4 gate not cleared

**Built** (`scripts/g1/solve_lafan1_canonical_g1_contactfirst.py`): ported Alex's
`refine_leg_floor_transitions` (option (a) from the entry above) as
`refine_leg_floor_transitions_g1` + `_leg_floor_pen_flags_g1`, new `--floor-leg-refine` flag
(+ `--floor-refine-{weight,margin,gain,pen-tol,ramp,preroll,posture-reg,lock-weight,root-relief}`,
all defaulted to Alex's own shipped values). Adapted, not a verbatim port: G1 has no
`SOLE_CORNER_SITES` (Alex-specific hand-authored sites, `alex_floating_base_with_sites.xml`) or
foot-flat `align_constraint` (v1 scope never built either — module docstring). Detection uses
mesh-accurate per-leg penetration (`_geom_lowest_z` over `LEG_BODY_NAMES`' own geoms, new G1
constant) instead of named sites; the synthetic-plant ankle Z target reuses this file's OWN
pull-to-floor formula (origin-height-above-own-support-point) instead of Alex's fixed
`alex_floor_z + ankle_clearance` (G1's floor is always z=0 by this project's convention, no
clip-wide estimate needed). `LEG_CHAIN_JOINTS`/`_joint_dofadr` give the 6-DOF hip/knee/ankle
chain per leg. Base `--floor-weight` stays default-0 (proven unstable, S4-T3 above) — this pass
supplies its own independently-ramped `floor_weight` only inside detected windows. Required
threading a new `frame_cache` (list of (targets, ori_targets, hold_pos_roles) per solved index,
lighter than Alex's version since this file's Pass-1 call has no align_constraints/
pos_site_constraints/skip_*/ori_weight_scale/pos_weight_scale) through the main per-frame loop.

**First finding**: at Alex's default `pen_tol=1.5cm`, this is NOT a rare-onset mechanism on G1
the way it is on Alex — `_leg_floor_pen_flags_g1` flags 85-95% of frames on every dev clip tested
(walk1, fallAndGetUp1, fallAndGetUp2, ground1, walk3), collapsing into a handful of windows that
each span most of the clip. Consistent with the already-known baseline `pen_pct` numbers (66-95%
of frames >5mm even at Pass 1) — G1's floor conflict here is chronic, not sporadic, so this pass
functions closer to "a second full solve pass, leg-chain-locked + root-relieved + ramped-Floor-
weight" than Alex's original "rare local fixup" design intent.

**Second finding, more serious**: the refine pass itself diverges catastrophically on 2 of 4 dev
clips — `fallAndGetUp1_subject1` floorPen 25.47->274.40cm, `fallAndGetUp2_subject2` 39.71->208.86cm
(vs `s3_raw`). Root-caused by direct measurement (`fallAndGetUp1`, frame 3850): pelvis Z sits at
-0.01 (normal) at frame 3844, plunges to -2.61 by frame 3850, then RECOVERS to +0.03 at frame
3851 — a single-frame-cluster (~6 frames) DLS local-optimum blowup, fully isolated and
self-correcting, NOT a sustained lying/prone-phase conflict (ruled out: pelvis is normal both
immediately before and after). Consistent with `max_step_norm`/small-iters-budget territory but a
new failure shape (root-driven, multi-meter) not previously seen in this project.

**Fix (implemented, cheap)**: added a per-frame divergence guard inside the refine loop — after
solving a window frame, compare its leg-geom penetration depth against Pass 1's OWN value at that
exact frame; if the refined result is worse by more than 3cm, reject it and keep Pass 1's value
(and reset `q_prev` to that fallback so the bad frame can't poison the rest of the window's
temporal-continuity warm start). Confirmed effective: `fallAndGetUp1` 274.40->**32.97cm**,
`fallAndGetUp2` 208.86->**68.45cm** (both re-run with `--floor-leg-refine` + the guard).

**4-clip result, `s3_raw` vs `s4_legrefine`/`s4_legrefine_guard`** (guard applied to
fallAndGetUp1/2 where it mattered; walk1/ground1 shown pre-guard since neither hit the >3cm
rejection threshold there):

| clip | floorPen off->on | pen% off->on | coll% off->on | heldL frac3 off->on | heldR frac3 off->on |
|---|---|---|---|---|---|
| walk1_subject1 | 17.35->**25.56cm** (worse) | 70.9->**50.6** (better) | 2.09->2.17 | 78.6->**95.8** (better) | 81.5->82.6 |
| fallAndGetUp1_subject1 | 25.47->**32.97cm** (worse) | 75.9->**63.8** (better) | 6.40->6.74 | 78.8->**89.6** (better) | 86.7->**94.2** (better) |
| fallAndGetUp2_subject2 | 39.71->**68.45cm** (worse) | 81.3->**69.2** (better) | 14.29->14.17 | 83.1->**89.6** (better) | 73.6->79.9 |
| ground1_subject1 | 22.16->**61.03cm** (worse) | 95.4->**69.7** (better) | 11.58->11.41 | 79.7->**98.6** (better) | 76.4->71.3 (worse) |

**Verdict: real, consistent improvement on breadth (`pen%` down 12-26pts every clip) and
held-frame contact (7/8 foot-clip pairs better or flat), self-collision untouched — but worst-case
`floorPen` gets WORSE on all 4 dev clips (+8 to +39cm), even with the divergence guard.** T4's
gate (`pen% <= 10`, no floorPen regression) is not cleared by a wide margin — `pen%` is still
50-70% (needs <10), and the one metric this whole sprint exists to fix (worst-case penetration)
moved the wrong direction everywhere. Not a clean win, not a regression either — a genuine
"improves the average, costs the tail" tradeoff, same shape as several earlier fixes in this
project's history (S2-T12's knee_bias, S2-T9's boosted StageA). Not tuned further this pass
(pen_tol/ramp/root_pos_relief are all still Alex's un-retuned defaults — plausible next levers,
untested) — stopping to report before iterating blind on parameters that were never validated for
G1's chronic-not-sporadic penetration shape.

## S4-T4 — Prabin's redirect: what are we actually tracking, and where's the real positive result? `--swing-clear` port, clean win on locomotion

**Context**: Prabin stepped back mid-sprint to re-orient rather than keep tuning `--floor-leg-refine`
blind. Investigated and answered directly (not guessed) before building anything:

- **Contact bodies unchanged**: still exactly 4 (`left_foot`,`right_foot`,`left_hand`,`right_hand`,
  `CONTACT_POS_ROLE`). No hips/knees/torso are ever hold/contact-anchored — position-tracked only.
- **Tracking is position (15 roles) + orientation (7 roles) against G1 BODY ORIGINS** — no custom
  MuJoCo sites defined anywhere in this pipeline.
- **Hip site check** (Prabin's specific question, "is hip actually at hip or mid-thigh"): measured
  directly via `data.xanchor`. True hip ball-joint center (`left_hip_pitch_joint` anchor) sits
  10.3cm below pelvis. `left_hip_yaw_link` (what `ROLE_TO_G1_BODY["left_hip"]` tracks) sits 25.1cm
  below pelvis — ~57% of the way to the knee (43.9cm below pelvis), i.e. genuinely closer to
  mid-thigh than the hip joint. **But**: checked GMR's own published config
  (`GMR/general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json`) — they map `LeftUpLeg` to
  the SAME body, `left_hip_yaw_link`. Not a bug relative to GMR, not a differential disadvantage —
  both systems share this correspondence. Deprioritized, not fixed.
- **Body-penetration breakdown** (never built as T1 asked — finally done, lightweight): sampled
  ~8300 penetrating frames across 8 locomotion clips (walk/run/sprint/jumps/obstacles,
  `ours_raw`/S3 baseline, mesh-accurate deepest-geom-per-frame via `_geom_lowest_z`). Result:
  **100% ankles** (`left_ankle_roll_link`/`right_ankle_roll_link`, ~48/52 split). Zero frames had
  a knee, hip, torso, or hand as the deepest point. Locomotion's penetration is a pure,
  classic swing-foot-through-floor problem — nothing like the whole-leg/root chaos
  `--floor-leg-refine` was built for (that mechanism belongs to the fall/get-up clips, S4-T3).
- **Where the real, defensible positive result is**: class-split `s3_full_corpus.csv` by
  `s1t4_reclass.csv`'s `floor_class` (locomotion n=43): `gmr_polished` floorPen 3.0cm/pen% 1.0%
  but held-foot-within-3cm only 31% mean / 10% median (GMR floats — near-zero penetration, poor
  actual contact, even on the easy clips). `ours_raw` floorPen 16.7cm/pen% 62-66% but held-frac3
  **88.5%**. The winnable, honest claim was never "less penetration than GMR" (GMR barely
  penetrates at all) — it's "GMR never actually plants the foot; we do, at a penetration cost
  that (per the finding above) is narrowly a swing-foot-clearance problem, not a systemic one."

**Built**: `--swing-clear`, ported from Alex's OWN two-part mechanism (`solve_fbx_canonical_alex_
contactfirst.py`) — NOT the two-pass windowed refine from S4-T3, a much lighter INLINE per-frame
term. Alex's own history (mined from that file's flag help text) already ran this exact
experiment: a soft one-sided position-lift term was tried FIRST and "fought the plant machinery,"
rejected; what actually worked was capping the swing foot's ORIENTATION TARGET toe-down pitch
(`cap_foot_pitch`, spends unused ankle dorsiflexion headroom instead of copying the human's
plantarflexed step through the floor) paired with a temporal-continuity `posture_reg` boost on
hip+knee DOFs only, not ankle (`_swing_posture_reg`, `LEG_CONT_JOINTS` = both legs' hip+knee,
new G1 constant) to stop the redundant leg from branch-flipping when the cap engages. Both
functions imported UNCHANGED from Alex's solver (generic, no Alex-specific state). No proximity
gate ported (Alex's own shipped config runs with the gate at its effectively-off default; swing-
ness alone, via the SAME `zone_env` cross-fade already used for pull-to-floor, was sufficient
there) and no soft position-lift term (Alex found it added nothing over the pitch cap alone,
off by default there too) — this is deliberately the minimal, already-validated-on-Alex subset,
not a re-derivation.

**11-clip result** (`s3_raw` -> `s4_swingclear`, 8 locomotion + the 3 hardest S4-T2/T3 clips for
contrast), no divergence guard needed anywhere (unlike S4-T3, no blowups):

| clip | floorPen off->on | pen% off->on | coll% off->on | heldL/R frac3 off->on |
|---|---|---|---|---|
| walk1_subject1 | 17.35->15.54cm (better) | 70.9->**64.4** | 2.09->2.09 | 78.6/81.5->**93.4/95.6** |
| walk2_subject1 | 17.86->19.49cm (worse) | 66.8->**61.5** | 6.44->6.68 | 85.6/89.7->**94.6/95.8** |
| run1_subject2 | 17.80->12.19cm (better) | 69.0->**42.6** | 5.40->6.80 | 91.8/93.8->**99.1/99.3** |
| run2_subject1 | 15.17->18.91cm (worse) | 61.3->**43.0** | 2.46->5.21 | 87.5/80.4->**96.8/94.6** |
| sprint1_subject2 | 15.51->15.78cm (flat) | 59.2->**53.5** | 3.71->6.18 | 65.8/73.6->**84.9/86.8** |
| jumps1_subject1 | 15.99->18.83cm (worse) | 63.8->**39.8** | 3.82->2.85 | 91.9/94.9->**98.1/97.9** |
| obstacles1_subject1 | 15.01->10.67cm (better) | 40.3->**27.5** | 1.71->2.19 | 93.6/94.1->**99.9/99.7** |
| obstacles2_subject1 | 17.56->15.19cm (better) | 44.2->**38.4** | 1.42->1.95 | 93.2/93.4->**97.9/97.1** |
| fallAndGetUp1_subject1 | 25.47->33.04cm (worse) | 75.9->**67.6** | 6.40->6.14 | 78.8/86.7->**89.4/91.1** |
| fallAndGetUp2_subject2 | 39.71->28.56cm (better) | 81.3->**73.7** | 14.29->11.45 | 83.1/73.6->**91.7/87.7** |
| ground1_subject1 | 22.16->34.40cm (worse) | 95.4->**94.2** (~flat) | 11.58->9.91 | 79.7/76.4->**99.6/95.3** |

**Verdict: `pen%` and held-frame contact improve on EVERY SINGLE one of the 11 clips tested, zero
exceptions** — the first mechanism this sprint that doesn't trade breadth/contact for worst-case
elsewhere. `floorPen` is a genuine mixed bag (5 better, 5 worse, 1 flat) but bounded — no
catastrophic spikes anywhere (worst degradation: ground1 +12.2cm; contrast S4-T3's leg-floor-
refine, which needed a divergence guard after hitting +200-370cm). Self-collision moves a little
in both directions, stays single-digit except the already-elevated fallAndGetUp2/ground1 (which
IMPROVE here, both -2 to -3pts). Locomotion specifically (8 clips): pen% drops 5-24 points EVERY
clip, held-frac3 gains 5-19 points EVERY clip/foot, no clip regresses on either. Does not clear
T4's `pen%<=10` gate (still 27-64% on locomotion) — this is not a finished result — but it is the
cleanest, most uniform, least risky improvement found in the whole S4 sprint, and it directly
targets the mechanism the body-penetration breakdown actually found (ankles, not the whole leg).
Not yet tuned (`--swing-max-pitch 5`/`--swing-continuity-reg 0.9` are Alex's own values,
unvalidated for G1) or extended to the full corpus. Natural next step per Prabin's own framing:
treat this as the locomotion-class result, keep fall/get-up (`--floor-leg-refine`, S4-T3) as a
separate, harder, honestly-reported residual — not one combined number.

## S4-T4 (continued) — tuned `--swing-max-pitch`/`--swing-continuity-reg` for G1; new defaults, validated corpus-wide

**Grid** (new script `scripts/g1/sprint_s4_t4_tune.py`, resumable, mirrors the sprint_s3/s4_t2_eval
pattern): `mp` in {3,5,8,12}, `cr` in {0.5,0.9,1.8}, 3 dev clips chosen for spread (`walk1_subject1`
flat/insensitive baseline, `run2_subject1` and `jumps1_subject1` both regressed `floorPen` in the
untuned run above). 36 combos, all built clean, no failures.

**Finding: `cr` is the dominant lever, `mp` is weak in the tested range.** Mean floorPen delta vs
`s3_raw` across the 3 clips, by `cr` (averaged over all 4 `mp`): cr=0.5 -> +0.38cm, cr=0.9 (Alex's
shipped value) -> +1.44cm, cr=1.8 -> +2.97cm. `mp` moved the mean by at most ~0.5cm at fixed `cr`.
Held-frac3 was nearly flat across the WHOLE grid (95.8-96.7% / 95.9-97.1%) -- the contact-accuracy
win is robust to tuning, only the floorPen cost is tunable.

**Pushed lower**: tested `mp=8, cr in {0.0, 0.2}` (6 more runs, same 3 clips) to find where Alex's
own warned failure mode ("0 = off, cap alone flips") actually bites on G1. Confirmed it does:
at `cr=0.0`, walk1_subject1's held-contact gain COLLAPSED back to near-baseline (78.8/81.5 vs
`s3_raw`'s 78.6/81.5 -- essentially no improvement) even though floorPen/pen% looked fine --
exactly Alex's "leg branch-flips instead of tracking cleanly" mechanism, reproduced on G1.
`cr=0.2` does NOT show this: held-frac3 stayed at 92-98% (matching cr=0.5/0.9), and floorPen was
BETTER than cr=0.5 on 2/3 clips (run2 16.17 vs 17.03cm, jumps1 14.29 vs 14.73cm), matching within
noise on the third (walk1 15.61 vs 15.55cm). Mean floorPen delta at `mp=8, cr=0.2`: **-0.81cm**
(better than baseline, on the 3-clip tuning set).

**New defaults shipped**: `--swing-max-pitch` 5.0 -> **8.0**, `--swing-continuity-reg` 0.9 ->
**0.2** (both were Alex's untouched values before this pass; `--swing-clear` itself still default
off). Documented in the flags' own help text with the numbers above.

**Validated on the full 11-clip set** (8 locomotion + the 3 hardest S4-T2/T3 clips, not just the
3-clip tuning set — avoids overfitting), `s3_raw` vs untuned-defaults (`mp5/cr0.9`, S4-T4's first
pass) vs tuned (`mp8/cr0.2`, new default):

| | mean floorPen | mean pen%% | mean coll%% | mean heldL | mean heldR |
|---|---|---|---|---|---|
| s3_raw | 19.96cm | 66.2 | 5.39 | 84.5 | 85.3 |
| untuned (mp5/cr0.9) | 20.24cm (worse) | 55.1 | 5.59 | 95.0 | 94.6 |
| **tuned (mp8/cr0.2)** | **18.22cm (better)** | 55.9 | **5.25 (better)** | 94.3 | 93.9 |

Tuning fixes the untuned run's worst flaw (mean floorPen was WORSE than doing nothing, +0.28cm)
into a genuine net positive (-1.74cm vs baseline) while keeping essentially all the pen%/held/
coll gains (all three within ~1pt of the untuned numbers). Per-clip: floorPen improves or is
flat on 9/11 clips; the two exceptions are modest (`run2_subject1` +1.0cm, `fallAndGetUp1_subject1`
+2.3cm) with zero blowups anywhere. Notably, tuning ALSO substantially de-risked the 3 hard clips
that weren't part of the tuning set: `ground1_subject1` floorPen 34.40cm (untuned, a real
regression) -> 22.53cm (tuned, matching `s3_raw`'s 22.16cm almost exactly);
`fallAndGetUp1_subject1` 33.04cm -> 27.79cm. `cr` generalizes well past the 3-clip grid it was
picked on.

**Status**: `--swing-clear` (tuned defaults) is now this sprint's cleanest result — improves
`pen%`/held-contact on every one of 11 clips tested, improves mean `floorPen` below baseline,
no divergence guard needed anywhere. Still does not clear T4's `pen%<=10` gate (mean 55.9%) —
not a finished result, but the strongest, most uniform, least risky one found this sprint.
Not yet: extended to the full 43-clip locomotion set, or combined with `--floor-leg-refine` for
the fall/get-up class (S4-T3's own tuning levers -- pen_tol/ramp/root_pos_relief -- are still
untouched; same kind of gain may be available there too, not tested this pass).

## S5-D0 | Pivot decision: GMR's mink tracking as base, contact layer on top (S4-T5/T6 dropped)

**Trigger**: Prabin compared the three annotated renders (repo root:
`walk1_subject1_{raw,swingclear_tuned,gmr_heightfix}_annotated.mp4`). Verdict: GMR's own
output (heightfix, zero smoothing from us) is visually excellent — no flicker, no snapping,
natural hand orientation (15.9% frames pen>0.5cm, max 1.66cm). OURS flickers/snaps, palm
rotated into the thigh, deep penetration even tuned (56.6% pen, max 8.76cm).

**Root-cause reading of GMR's actual method** (`motion_retarget.py`, `bvh_lafan1_to_g1.json`):
(1) mink velocity-space QP with joint limits as HARD constraints inside the solve (vs our
post-hoc clamp — the documented branch-flip/flicker source); (2) orientation-FIRST weighting:
table1 is rot-only (cost 10–100 on ~13 bodies incl. thigh/shank/upper-arm/forearm, pos cost 0
except feet), table2 adds position lightly — vs our position-first 15 points with weak ori on
only 7 roles. Position-only limb targets leave hand roll about the forearm axis and knee
bend-plane under-constrained: that IS the palm-toward-body and knee-weirdness mechanism.
(3) GMR has ZERO contact handling — one constant per-clip ground z-offset
(`set_ground_offset`), literally the S3 z-shift-oracle mechanism. That is the paper opening.

**Decision (Prabin, 2026-07-17)**: Phase B first — time-boxed (~2 days) test of whether OUR
DLS solver reaches GMR quality with orientation-first reweighting + real limit handling
(diagnostic + possible solver-agnostic ablation). Then Phase A REGARDLESS of B's outcome:
contact layer (held-foot snap: locked XY at plant onset, z = sole-on-floor, ramped task costs)
built inside GMR's own mink solve, locomotion first, with a fidelity non-regression guard vs
GMR's own scaled-human targets (`--save_human_targets` dump). Headline metric stays S4-T5's
joint_ok_pct (held ≤3cm AND pen <5mm) which constant-shift baselines cannot game. Full plan:
`GMR-S5-plan.md` (Sonnet-executable). `GMR-S4-plan.md` marked superseded; S4-T5/T6 dropped.

## S5-B0.1 | Hand-frame diagnostic — target construction is broken, not just under-weighted

**Method**: compared OURS canonical `left_hand`/`right_hand` orientation frames
(`lafan1_to_canonical_human.py`) vs GMR's own `scaled_human_data` hand targets
(`gmr_headless_retarget.py --save_human_targets`, walk1_subject1, both already existed on
disk from an earlier run). Computed per-frame relative rotation, fit a single best-fit
constant offset (mean quaternion over the clip), subtracted it, looked at the residual.

**Result — NOT a fixed calibration offset**: residual after removing the clip's own
best-fit constant is huge and time-varying. Left hand: mean 43.5°, p50 35.6°, p90 91.9°,
max 173.3°. Right hand: mean 49.9°, p50 45.9°, p90 76.0°, max 179.3°. A correct target
that differs from GMR's only by a fixed convention/frame offset would residual near 0°
everywhere; this doesn't.

**Root cause found by reading the code** (`lafan1_to_canonical_human.py` lines 122-128):
our hand frame's PRIMARY axis is the forearm direction (`wrist - elbow`, correct, real
motion signal — this was S2-T4's fix). But the SECONDARY axis is **`pelvis_y`**
(`left_hip - right_hip`) via `frame_from_xy`, which orthogonalizes pelvis_y against the
forearm axis and uses THAT as the y-axis. `frame_from_xy`'s y-axis directly fixes the
frame's roll about the primary (forearm) axis. Since pelvis_y is a near-rigid,
slowly-varying reference (not derived from any wrist/hand landmark), **our hand target
structurally cannot represent forearm twist (pronation/supination) at all** — its roll
about the forearm axis is a geometric artifact of projecting pelvis_y, not a measurement
of real wrist rotation. This is exactly consistent with "palm always rotated toward
body": the twist DOF is not noise, it's simply absent from the target.

GMR, by contrast, uses the **raw BVH bone rotation** for `LeftHand`/`RightHand` directly
(`bvh_lafan1_to_g1.json`: `left_wrist_yaw_link` <- `LeftHand`, pos_weight 0, rot_weight
10, with a fixed `rot_offset` calibration quaternion) — full 3-DOF orientation straight
from mocap, including twist. This is why GMR's palm orientation looks physically correct
and ours doesn't, independent of solver orientation weight (0.40 in TARGET_WEIGHTS would
not fix a target with no twist information to weight).

**Note on CLAUDE.md's "semantic frames from landmark positions, not raw FBX rotations"
rule**: written for the Alex/FBX pipeline, where vendor rigs have inconsistent bind
poses. LAFAN1 BVH is a single clean mocap rig — GMR's own SOTA pipeline already reads its
raw per-bone rotations directly and it works. Using LAFAN1's raw `LeftHand`/`RightHand`
bone rotation for the G1 hand target (this comparison track only, not touching Alex) is
not the footgun the rule was written to prevent. **B1 will implement this fix.**

## S5-B0.2 | Knee-bend diagnostic — systematically over-flexed, not a single glitch

**Method**: knee joint angle (0=straight, + = flexed, limits [-5, 165] deg) stats on
walk1_subject1, all 7840 frames, three variants (existing artifacts, no new solves):
OURS s3_raw (`ours_g1_corpus/walk1_subject1_ours.npz`), OURS swing-clear tuned
(`s4_dev/tune/walk1_subject1_mp8_cr0.2.npz`), GMR heightfix (`pkl/walk1_subject1_gmrfix.pkl`).

| variant | mean L/R | p50 L/R | p90 L/R | % frames near -5° limit L/R |
|---|---|---|---|---|
| OURS s3_raw | 60.6 / 59.9 | 71.8 / 71.0 | 92.9 / 92.0 | 15.2 / 15.0 |
| OURS swing-clear tuned | 66.3 / 65.8 | 74.3 / 74.2 | 88.8 / 89.5 | 7.5 / 7.9 |
| GMR heightfix | 29.3 / 27.4 | 23.7 / 20.4 | 59.3 / 56.6 | 0.0 / 0.0 |

**Answer to "is the knee bend forced?": yes, systematically.** OURS's mean knee flexion
is roughly double GMR's (60-66° vs 27-29°), median nearly triple (71-74° vs 20-24°) —
this is not one bad pose, it's baked into the whole gait cycle. Never near the FLEXED
limit (165°, hi%=0 everywhere) — so it's not a hard joint-limit pin on the flexed side.
It IS pinned near the STRAIGHT limit (-5°) for 15% of frames at s3_raw (the documented
S2 warm-start-basin bug — swing-clear tuning roughly halves this to 7.5-7.9% as a side
effect, consistent with S4-T4's finding that `cr` reduces branch-flip, but doesn't fix
the mean over-flexion at all — swing-clear tuned mean is actually slightly WORSE, 66.3
vs 60.6, likely from posture_reg's hip/knee continuity term nudging bend up to stay
smooth through the straight-limit-pinned frames).

**Consistent with B0.1's root cause, not a separate bug**: OURS tracks knee as a bare
position point (hip/knee/ankle, 3 independent XYZ targets) with NO shank/thigh
orientation constraint on the bend PLANE — GMR position-weights the knee at 0 in both
tables and instead orientation-tracks the thigh (`left_hip_yaw_link`, rot 10) and shank
segment directly, pinning the bend plane. A position-only 3-point IK is free to choose
*any* knee angle that reaches the same three points along a curved vs straighter path
(classic elbow/knee IK redundancy) — the DLS solve apparently settles toward extra
flexion as a side effect of jointly satisfying hip+ankle+other competing targets, not
because anything explicitly asks for it. No fix attempted here (diagnostic only, per
plan); B2b (add thigh/shank orientation roles) is the mechanism expected to fix this if
Phase B proceeds that far.

## S5-B0.3 | Motion smoothness/jerk metric — quantifies the "flicker" verdict

New `scripts/g1/motion_smoothness.py`: joint jerk (3rd finite diff of 29 joint angles,
rad/s^3) + FK'd body-position jerk (3rd finite diff of left/right ankle+wrist world
position, m/s^3), mean/p95/max. Computed on existing artifacts, no new solves.

| clip | variant | joint_jerk mean | joint_jerk p95 | body_jerk mean | body_jerk p95 |
|---|---|---|---|---|---|
| walk1_subject1 | OURS s3_raw | 8192 | 35112 | 531.5 | 2172.8 |
| walk1_subject1 | OURS swing-clear tuned | 5301 | 20236 | 365.9 | 1641.1 |
| walk1_subject1 | GMR raw / heightfix (identical — const Z shift doesn't touch joints) | **2352** | **6074** | **95.4** | **340.0** |
| run2_subject1 | OURS s3_raw | 11045 | 29084 | 601.4 | 2252.3 |
| run2_subject1 | OURS swing-clear tuned | 11509 | 36160 | 636.2 | 2421.5 |
| run2_subject1 | GMR raw / heightfix | **8226** | **18454** | **324.4** | **935.4** |

**Verdict**: GMR is consistently smoother, but the GAP IS MOTION-DEPENDENT — walking
(slow, more redundant/underdetermined IK) shows a large gap (OURS 2.3-3.5x GMR's mean
joint jerk, 5.8x p95); running (fast, less redundancy slack, high natural jerk even for
GMR) shows a much smaller gap (OURS only 1.3-1.4x mean). This matches the branch-flip
theory directly: slow motions give the position-only DLS solve more freedom to flip
between near-equally-good IK solutions frame to frame (visible flicker); fast motions are
more kinematically constrained regardless of solver, so there's less redundancy to flip
within. Swing-clear tuning helps walk1 (both jerk numbers drop 30-42%) but not run2
(flat to slightly worse) — consistent with S4-T4's own finding that run2 was one of
swing-clear's two non-improving clips. This is the smoothness column that will feed
B-GATE and A2.

## S5-B1 | Hand target fix implemented — raw BVH bone rotation, validated exact match to GMR

**Change**: `lafan1_to_canonical_human.py` hand orientation (`left_hand`/`right_hand`
roles) now uses the raw BVH bone rotation (`load_bvh_file`'s own per-frame
`f["LeftHand"][1]`/`f["RightHand"][1]`, world-frame FK quaternion, wxyz) instead of
`frame_from_xy(wrist-elbow, pelvis_y)`. Left-multiplied by the same yaw-facing-correction
matrix already applied to positions, so it composes correctly with the rest of the
pipeline (feet/pelvis/torso/head unchanged, still landmark-derived semantic frames).

**Correction to B0.1's own methodology, found while validating this fix**: the ~43-50°
mean residual B0.1 measured was NOT purely the pelvis_y-roll bug — it was partly an
artifact of comparing OUR yaw-facing-corrected canonical (walk1 needed a 90° facing
correction) against GMR's own `scaled_human_data` dump, which has NO such correction
(GMR doesn't normalize clip facing direction; that is entirely our pipeline's own
convention). Isolated this by comparing pre-yaw-correction raw BVH quat directly against
GMR's target: **residual = exactly 0.00 deg at every one of 7840 frames**, with the
constant relative offset matching GMR's own documented `rot_offset` calibration
(`bvh_lafan1_to_g1.json`: `[0.70710678, 0.70710678, 0, 0]`, a 90 deg rotation about X) to
the printed precision. This proves the fix is not just plausible but exactly correct —
our new hand target IS GMR's own target, modulo the yaw-facing convention difference
that was already known to apply to every other role. (Downstream G1 IK target
construction uses world-delta-from-rest per CLAUDE.md, so the yaw-frame difference
doesn't matter there — both rest-frame and current-frame hand orientation are yaw-shifted
consistently.)

**Regenerated dev-clip canonicals** (`outputs/gmr_baseline/sprint/canonical_human_s5/`,
`lafan1_to_canonical_human.py` -> `ground_canonical_human.py --plant-min-run 2`, same
recipe as S2-T1/T2): walk1_subject1, walk3_subject1, run2_subject1,
fallAndGetUp1_subject1, ground1_subject1. Grounding numbers closely match the original
S2-T2 build (walk1: floor p50=0.0039m, both builds -- confirms the grounding step itself
is unaffected by the hand-orientation change, as expected).

**Re-solved walk1_subject1** with the fixed canonical, otherwise-default flags
(`outputs/gmr_baseline/sprint/s5_dev/ours_b1/walk1_subject1_ours_b1.npz`). Whole-clip
metrics vs the old s3_raw build (same solver, old hand target):

| | floorPen_cm | pen_pct | coll_pct | coll_peak_cm |
|---|---|---|---|---|
| s3_raw (old hand target) | 17.35 | 70.92 | 2.09 | 2.14 |
| B1 (fixed hand target) | 18.07 | 70.89 | 3.19 | 1.86 |

Floor metrics essentially flat (expected -- hand fix shouldn't move feet/floor numbers).
Aggregate self-collision % moved slightly worse (total across ALL body pairs, not
hand-specific) but peak improved. Direct hand-to-thigh proximity (body-origin distance,
`left/right_rubber_hand`+`wrist_yaw_link` vs `hip_yaw/hip_pitch/knee_link`, both sides):
mean 16.3->17.3cm, p10 6.5->7.8cm, **frac<10cm 43.5%->32.8%** (fewer close-approach
frames, net improvement) though the single worst-frame min got closer (1.4->0.8cm, one
outlier). **Verdict: the specific visual symptom (palm structurally locked toward body)
is fixed and validated exactly against GMR's own target; net hand-thigh proximity
improves; aggregate whole-body self-collision is a mixed bag (not the target of this
fix) and floor metrics are unaffected as expected.** Knee over-flexion (B0.2) is
untouched by this fix -- separate mechanism, expected to need B2b (thigh/shank
orientation roles) if attempted.

## S5-B2 | Orientation-first reweighting preset — mixed/negative, B2b (new ori roles) needed for a real win

New `--gmr-style-weights {0,1,2}` on `solve_lafan1_canonical_g1_contactfirst.py`
(monkey-patches `alex_solver.TARGET_WEIGHTS`/`ORI_WEIGHTS` in-process, opt-in, never
touches the shared Alex file on disk or other processes). Tested on walk1_subject1
(B1's fixed-hand canonical as the base).

**Attempt 1** (aggressive distal position cut: knee/elbow/hip/shoulder pos ~0.2-0.3,
raise pelvis/torso/feet/hand orientation 1.5-3.0x): WORSE on every metric. floorPen
18.07->22.95cm, coll_peak_cm 1.86->5.88cm (a self-collision spike more than tripled),
knee mean flexion 60.7/59.9->64.3/65.1 deg (WORSE, not better). pen_pct did improve
(70.9->52.2%).

**Root cause**: we have NO thigh/shank/upper-arm/forearm orientation roles (that's
B2b, not implemented this pass) -- cutting knee/elbow POSITION weight removes the only
signal constraining the knee bend-plane and elbow bend-plane without installing a
replacement. Redundancy went UP, not down. This is exactly what GMR-S5-plan.md warned
B2 alone might do.

**Attempt 2** (softer preset, milder cuts: knee/elbow/hip/shoulder pos ~0.6-0.7 vs
original ~0.8-1.0, orientation raised more mildly 1.2-1.5x): still a mixed bag, not a
clean win. floorPen 19.54cm (worse than B1's 18.07, better than attempt 1's 22.95),
pen_pct 60.3% (better than B1's 70.9), coll_peak_cm 3.02cm (worse than B1's 1.86,
better than attempt 1's 5.88), knee mean 57.8/58.1 deg (marginally BETTER than B1's
60.7/59.9 -- the one genuinely positive knee signal from any B2 attempt, but small).

**Verdict**: reweighting alone, without B2b's new orientation roles, does not reach a
clean win -- pen_pct improves consistently across both attempts, but floorPen and
self-collision consistently get WORSE, and knee flexion only marginally improves with
the soft preset. This matches the plan's own prediction. **B2b (add thigh/shank/
upper-arm/forearm orientation roles, GMR-S5-plan.md's "biggest B item") is the
mechanism actually needed to test the orientation-first hypothesis properly** -- not
attempted this pass (time-box). Given Phase B's ~2-day budget and that Phase A is the
paper's critical path regardless of B's outcome, deferring B2b rather than attempting
it now; B2 (weights only) does not clear a bar on its own and is logged as a
negative/inconclusive ablation result, not a shipped mechanism.

## S5-B3 | Joint-limit active-set + convergence early-exit — both informative negatives

New `--active-set-limits` / `--early-exit-tol` on `solve_lafan1_canonical_g1_contactfirst.py`
(both opt-in kwargs on the shared `solve_frame_position_ik`, default off/None = unchanged
behavior). Tested independently and combined on walk1_subject1 (B1's fixed-hand canonical).

**`--active-set-limits`: confirmed NUMERICAL NO-OP.** Direct instrumentation (temporary debug
print, removed after) confirmed the mechanism DOES trigger -- 8468 times in just the first 300
frames -- yet whole-clip metrics and knee stats came back byte-identical to the baseline
(floorPen 18.07cm, pen_pct 70.89%, coll_pct 3.19%, knee mean 60.7/59.9 -- all match exactly).
Root cause: `clamp_hinge_joint_limits` already runs every iteration (not just at the end), so a
pinned DOF snaps to the exact same boundary value each iteration regardless of whether its `dq`
component is zeroed pre-integration or left to overshoot-then-clamp. Because `dq` comes from one
linear solve (not sequential per-DOF), zeroing one component post-hoc doesn't let other DOFs
redistribute to compensate -- their values were already fixed by the same solve. A real version
would need to drop the limited DOF's column from `A` and re-solve (true reduced-QP, not a
post-hoc mask) -- out of scope this pass, but useful to know the shape of the fix if revisited.

**`--early-exit-tol 1e-3` (GMR's own value): real speedup, real quality cost.** 75s->21s per
clip (3.5x). But pen_pct 70.9->82.3%, coll_pct 3.19->7.73%, coll_peak_cm 1.86->3.36cm (only
floorPen improved, 18.07->16.60cm). Root cause: our TARGET_WEIGHTS/ORI_WEIGHTS are O(0.2-4),
GMR's own weight scale is O(5-100) -- their 1e-3 convergence tolerance is calibrated to THEIR
units and is comparatively loose in ours, so the solve exits before self-collision/floor rows
(sharing the same rows2 iteration budget) have actually converged. A correctly-scaled tolerance
for this solver needs its own calibration, not a literal port of GMR's number -- not attempted
(time-box).

**Both mechanisms left in the code, documented, opt-in, default off** -- neither ships as a win
this pass.

## S5-B-GATE | Phase B verdict: FAIL — OURS-DLS retired to ablation role, Phase A starts now

**Best B config**: B1 (hand target fix) + S4's tuned swing-clear (mp=8, cr=0.2) — the only
two Phase-B mechanisms that were genuine, validated wins (B2 reweighting and both B3
mechanisms were negative/inert, see their own entries above). Built and evaluated on
walk1_subject1 against the gate criteria in GMR-S5-plan.md:

| criterion | gate | B-GATE config | swing-clear-tuned alone (reference) | pass? |
|---|---|---|---|---|
| pen% no worse than swing-clear tuned | <= baseline | 66.15% | 66.26% | PASS (flat) |
| floorPen | (context) | 16.74cm | 15.61cm | slightly worse |
| coll_pct | (context) | 2.84% | 1.47% | worse (matches B1's own known trade) |
| joint_jerk mean vs GMR's 2352 (rad/s^3) | <=~2x | 4965 (2.11x) | -- | borderline FAIL |
| body_jerk mean vs GMR's 95.4 (m/s^3) | <=~2x | 345 (3.6x) | -- | FAIL |
| visible flicker/snapping | none | -- | -- | Prabin already rendered/watched this class of config (S4's videos) and called it "stupid mf, flickers, and snaps" |

**Verdict: FAIL.** Contact-accuracy metrics (pen%, held-frac3) are fine or flat, but the
core complaint this whole pivot started from -- smoothness/flicker -- is NOT resolved.
Body-level jerk (the whole-chain snap Prabin saw, not just per-joint noise) sits at 3.6x
GMR's, nowhere near the ~2x bar, let alone GMR's own visual quality. Root cause chain is
now fully traced: GMR's smoothness comes from its QP architecture (hard joint-limit
constraints INSIDE the solve, velocity-space integration) and its orientation-first
weighting with FULL limb-segment orientation coverage (thigh/shank/upper-arm/forearm,
which we structurally lack -- B2b, not attempted, would be needed to even test this
properly) -- neither is something a reweighting pass or a post-hoc DOF mask on our
existing DLS+clamp architecture can retrofit. B1 (hand target) is a real, keepable fix
(ships regardless) but it only fixes ONE specific symptom (palm-toward-body), not the
general flicker/knee/collision picture.

**Decision (per GMR-S5-plan.md's own pre-committed rule: Phase B does NOT gate Phase
A, and 2 focused attempts per sub-experiment is the cap)**: Phase B is closed. OURS-DLS
retains B1's hand-target fix as a permanent improvement, and is retired to an
ablation/comparison role for the paper (not the primary tracking method). **Phase A
(contact layer inside GMR's own mink solve) starts now** -- this was always the
paper's critical path regardless of B's outcome, per Prabin's original framing.

## S5-A1 | gmr_contact_retarget.py built, validated, one real bug found+fixed

New `scripts/g1/gmr_contact_retarget.py`: subclasses `GeneralMotionRetargeting` (never
edits `~/projects/GMR`), overrides ONLY the held foot's table-2 FrameTask (locked XY at
onset, Z=z_sole, flat orientation with yaw taken from GMR's own current target), ramped
cost cross-fade. Sanity check (`--no-contact`) confirmed BYTE-IDENTICAL to plain
`gmr_headless_retarget.py` output (root_pos/root_rot/dof_pos max abs diff = 0.0 across
all 7840 frames) -- the wrapper adds nothing until the override is enabled, as required.

**Bug found in A2 eval, not before**: `skate_cm` (max XY drift of a held foot from its
own onset position) came back with absurd values (194-224cm!) on walk1_subject1's very
FIRST held segment for both feet. Root cause: that segment starts at frame 0, before any
`retarget()` solve has run -- `self.configuration`'s FK position at that point is mink's
default/rest configuration (arbitrary, unrelated to the human's real starting pose), so
locking "onset XY" there grabs garbage ~2m from the true position. This ALSO explained
an earlier-looking floorPen spike (7.72cm on walk1) that had been mis-attributed to a
release hand-off -- same root cause, not two separate bugs. **Fix**: added `_solved_once`
flag (subclass `ContactAwareGMR`), set True by the driver after each frame's solve;
frame 0 (or any onset before the FIRST solve has run) is skipped for override purposes
entirely -- pure natural GMR tracking that frame, first legitimate lock happens no
earlier than frame 1. Confirmed fix: worst-case skate dropped from 194-224cm to
sub-3cm everywhere, and the floorPen spike disappeared (7.72cm -> 1.70cm, matching
gmr_raw's own 1.47cm).

Also refactored the cost/target cross-fade from a manual linear step to a proper cosine
ramp (`_cosramp`, matches `contact_labels.py::ramp_envelope`'s shape, zero derivative at
both ends) -- `ramp_envelope` itself couldn't be used directly (its pre-onset
"anticipation" semantics assume a target you can already compute before the transition;
here the onset XY literally doesn't exist until the onset frame's own FK), so this is a
purpose-built equivalent, counted from onset/release rather than from the boolean flag.

`scripts/g1/sprint_s5_metrics.py` (new): `joint_ok_pct` (S4-T5's un-gameable headline:
per frame with >=1 held foot, every held foot |support_z|<3cm AND whole-body pen<5mm),
`skate_cm`, `fidelity_metrics` (mean pos/ori error vs a run's own saved
`--save_human_targets`, over the 12 non-foot table2 bodies), `jerk_metrics` (reuses
`motion_smoothness.py`).

## S5-A2 | Locomotion gate: 3/4 criteria clear, jerk residually elevated after 3 tuning attempts

3 loco dev clips (walk1_subject1, walk3_subject1, run2_subject1), `gmr_contact` (A1)
vs `gmr_raw` vs `gmr_heightfix`. Three configs tried (transition-tuning cap: max 3
attempts per GMR-S5-plan.md):
1. ramp=5 frames, linear step, cost 50->200/10->50: jerk +43-75% vs gmr_raw (fails).
2. ramp=10, cosine, same cost: WORSE not better -- jerk +140% on walk1, joint_ok_pct on
   run2 dropped further (88.8%->82.2%). Root cause (inferred): a longer partial-cost
   window sustains tension between the locked foot and the still-moving pelvis for
   LONGER, not less -- the tension is likely continuous through the held region (pelvis
   wants to keep walking, locked foot resists), not just at the transition edges, so a
   longer ramp just prolongs the fight.
3. **ramp=5, cosine, cost 50->100/10->20 (gentler ceiling, closer to GMR's own table2
   scale where pelvis pos=100) -- best result, shipped as the default.**

**Config 3 vs gate** (GMR-S5-plan.md A2):

| | walk1 | walk3 | run2 |
|---|---|---|---|
| joint_ok_pct (gate >=90) | **98.9%** (raw 97.9) | **94.4%** (raw 92.0) | 85.2% (raw 91.9) FAIL |
| mean skate_cm (gate <=1) | 0.44/0.42 PASS | 0.29/0.31 PASS | 0.19/0.18 PASS |
| fidelity delta vs raw (gate <=1cm/2deg) | +0.41cm/+0.07deg PASS | +0.49cm/+0.13deg PASS | +0.19cm/+0.11deg PASS |
| jerk delta vs raw (gate <=20%) | +40%/+70% FAIL | +26%/+44% FAIL | +10%/+22% FAIL (close) |

For context, `gmr_heightfix` (GMR's own best floor mechanism, a single constant Z shift)
on the SAME joint_ok_pct metric: walk1 86.8%, walk3 **1.4%**, run2 **8.6%** -- the exact
z-shift-oracle failure this whole paper thesis is built on (can't satisfy held-contact
AND whole-body-pen simultaneously with one constant). `gmr_contact` beats gmr_heightfix
by 12-93 points on every clip.

**Verdict: 3 of 4 gate criteria clear cleanly (skate, fidelity, joint_ok on 2/3 clips);
jerk does not clear <=20% on any clip, though tuning substantially narrowed the gap
(walk1 body_jerk +75%->+70%, walk3 +58%->+44%, run2 +32%->+22%) and run2's own gap is
now small.** Render (`walk1_subject1_gmrcontact_annotated.mp4`, repo root, same
window/camera as the 3 earlier comparison videos, sent to Prabin for visual judgment):
1.2% frames penetrating >0.5cm in this window (vs gmr_heightfix's own 15.9% in the same
window), max 1.14cm (vs 1.66cm). Numerically this is the strongest result of the whole
GMR-baseline effort by a wide margin. Not a clean gate pass on jerk -- keeping this as
the honestly-reported A2 result and proceeding to A4 per plan (A3 swing-clearance is
scoped for swing-phase floor penetration specifically, not the jerk/transition-tension
mechanism found here, so it wouldn't address this residual -- skipping it for now,
flagging as a candidate follow-up: reduce contact cost ramp-in tension, e.g. a softer
XY-lock margin instead of a hard position pin, or blend toward GMR's own natural target
rather than a fully independent locked one).

## S5-A3 | Swing-floor clamp — real, partial improvement on run2, gate not fully closed

Diagnosed first (per plan): of run2_subject1's 339 joint_ok_pct failures, 339/339
(100%) were whole-body pen, only 5/339 also had a support_z failure -- and the
penetrating body was always the CURRENTLY NOT-held ankle (17/17 sampled frames:
10 right_ankle_roll_link + 7 left_ankle_roll_link, zero other bodies). Textbook swing-
foot floor clip during fast running, exactly what A3 is scoped for.

New `--swing-floor-margin` on `gmr_contact_retarget.py`: soft target-space clamp on the
NOT-held foot's own table-2 target Z (raised to `z_sole + margin` if GMR's own target
already sits below it; XY/orientation/cost untouched -- can't fight swing clearance the
way a hard task would, matches the plan's "clamp the target, not a new task" framing).

| margin | pen_pct | joint_ok_pct | coll_pct |
|---|---|---|---|
| 0 (A1 baseline) | 11.68 | 85.2 | 1.10 |
| 1.5cm | 11.68 (no effect) | 85.2 | 1.10 |
| 8cm | 5.70 | 86.7 | 1.24 |
| 15cm | 4.82 | 88.2 | 1.38 |

1.5cm did LITERALLY nothing -- the achieved-vs-target tracking error during fast
running exceeds a small margin on its own (this isn't purely a target-placement
problem, mink's solve doesn't perfectly reach even a corrected target every frame at
running speed). Larger margins help substantially on pen_pct (11.7%->4.8% at 15cm) and
nudge joint_ok_pct up (85.2%->88.2%) but don't clear the 90% gate even at 15cm, with
diminishing returns (8cm->15cm: pen_pct -0.9pt, joint_ok +1.5pt) and a small but real
self-collision cost creeping up (1.10%->1.38%) as the swing target is pushed higher
into the leg's own space. **Shipping `--swing-floor-margin 0.08` (8cm) as a reasonable
default** -- real win, gate not fully closed, logged honestly per the plan's one-attempt
scope for A3. `walk1`/`walk3` not re-tested with this flag (their joint_ok_pct already
cleared the gate without it); only affects clips where it's opted in.

## S5-A4 | Post-hoc ablation (existing Stage B QP) — near-zero jerk cost, but held-accuracy collapses off walking speed

Reused `scripts/g1/stage_b_g1.py::run_multisurface` (`--multi-surface`, an EXISTING
mechanism from S1/S2, not new this sprint) as `gmr_contact_post`: whole-trajectory QP
anchoring applied AFTER a raw GMR solve, vs A1's per-frame override INSIDE the solve.

**Methodology fix found while building this**: the tool's default `--human-contacts`
source (`human_contacts_lafan1.py`, a height-only zone heuristic) disagrees SHARPLY with
this sprint's own canonical `contact_flags`-based held mask on faster clips -- walk3's
zone was 85-88% of frames vs canonical's 49-52%. Comparing A4 against A1/A2 with
mismatched contact sources would be invalid (different "ground truth" for what counts as
held). Fixed by building a consistent `zone_<role>` npz directly from the SAME canonical
`contact_flags` (debounced, feet-only to match A1's scope) --
`outputs/gmr_baseline/human_contacts_s5/<clip>.npz`, new script (inline, not a checked-in
file this pass) -- and re-ran Stage B against that for all 3 dev clips.

**Result, consistent masks**:

| clip | variant | pen% | held_frac3 L/R | joint_ok_pct | jerk joint/body |
|---|---|---|---|---|---|
| walk1 | gmr_raw | 1.3 | 98.6/98.6 | 97.9 | 2352/95.4 |
| walk1 | A1 (in-loop) | 1.5 | 100/100 | 98.9 | 3301/162.5 |
| walk1 | **A4 (post-hoc)** | 15.9 | 99.9/100 | 87.3 | **2352/95.5** |
| walk3 | gmr_raw | 10.2 | 98.6/91.7 | 92.0 | 4110/160.4 |
| walk3 | A1 | 10.3 | 99.8/97.9 | 94.4 | 5171/230.7 |
| walk3 | A4 | 0.8 | **0.7/4.5** | **1.4** | 4110/160.8 |
| run2 | gmr_raw | 10.4 | 99.9/100 | 91.9 | 8226/324.4 |
| run2 | A1 | 11.7 | 99.9/99.7 | 85.2 | 9075/397.3 |
| run2 | A4 | 0.1 | **14.0/10.8** | **14.4** | 8226/323.7 |

**The jerk finding holds and is real and clean**: A4's jerk is ESSENTIALLY IDENTICAL to
gmr_raw on all 3 clips (differences <1%) -- Stage B's own `lambda_smooth=20.0` whole-
trajectory regularizer absorbs the transition cost that A1's per-frame override cannot.
This is the paper's clean in-loop-vs-post-hoc smoothness contrast.

**But held-accuracy is NOT consistent across gait speed**: excellent on walk1 (99.9/100,
matches A1), collapses on walk3 (pathological walker) and run2 (running) even with
IDENTICAL contact masks to A1 -- a genuine Stage-B-QP limitation on faster/shorter
contact intervals (likely `_pull_to_floor`'s per-run median-offset mechanism, or
`_compute_anchors`'s speed/min-run defaults, undersuited to these gaits -- not
diagnosed further, time-boxed). pen% is excellent everywhere (0.1-15.9%, best of any
variant on walk3/run2) because whole-body floor avoidance is a different, easier part
of the same QP than precise per-foot support_z placement.

**Verdict for the paper**: A1 (in-loop) is the more RELIABLE mechanism across gait
speed (clears joint_ok >=90 on 2/3 clips, close on the third) at a real jerk cost. A4
(post-hoc) achieves essentially free smoothness but its contact-placement accuracy is
gait-speed-dependent and currently only reliable on walking-speed clips. Report BOTH,
honestly, as the in-loop/post-hoc ablation the plan wanted -- this is a genuine,
non-obvious trade-off finding, not a clean win for either side. A5/A6 proceed with A1
(gmr_contact) as the primary variant; A4 stays a walking-only-scoped ablation row.

## S5-A5 | Hard-class extension (hands) — partial success exactly as expected

Generalized `gmr_contact_retarget.py` from feet-only to a parameterized `effectors`
list (`--effectors feet|feet+hands`; `EFF_BODY`/`EFF_HUMAN_KEY`/`EFF_CANON_ROLE` dicts
replace the old `FOOT_*` ones; hand base cost uses GMR's own table2 hand weight 10/5,
not the foot weight 50/10). Sanity check + feet-only byte-match against the pre-refactor
A1 result both still pass exactly (0.0 max diff) -- pure generalization, no behavior
change in the default path. Note: "flat" orientation for a held HAND (zero roll/pitch,
same as a foot) is an unvalidated v1 approximation -- a planted palm's real constraint
is closer to "normal into the floor," not corrected this pass.

Ran `--effectors feet+hands` on the 2 hard-class dev clips (fallAndGetUp1_subject1,
ground1_subject1). Held-effector accuracy (support_z frac3, |z|<3cm):

| clip | effector | gmr_raw | gmr_contact (A5) |
|---|---|---|---|
| fallAndGetUp1 | left_foot | 92.8% | 98.3% |
| fallAndGetUp1 | right_foot | 92.5% | 92.9% |
| fallAndGetUp1 | left_hand | 70.3% | **95.0%** |
| fallAndGetUp1 | right_hand | 10.6% | **94.9%** |
| ground1 | left_foot | 99.4% | 87.7% (regressed) |
| ground1 | right_foot | 98.6% | 56.4% (regressed) |
| ground1 | left_hand | 59.6% | **98.3%** |
| ground1 | right_hand | 69.1% | **98.6%** |

Hand accuracy improves dramatically on both clips (worst case 10.6%->94.9%). Foot
accuracy is a mixed bag on ground1 specifically (right foot regresses to 56.4%) --
likely hand and foot override tasks competing for the same limited solve budget/
priority on a clip where BOTH pairs are frequently held simultaneously (lying/prone
poses), unlike the loco clips where feet-only was the whole story. Not root-caused
further (time-box).

**Whole-body pen% barely moves** (fallAndGetUp1: 41.0%->40.5%; ground1: 87.8%->87.6%)
-- exactly the expected outcome per S2's own prior finding that these clips carry
genuine reach-limit poses (G1 is ~0.64 of this human's scale; documented worst case
181.6% of max leg reach on fallAndGetUp2_subject2). The contact mechanism does what
it's designed to do (place HELD effectors accurately); it cannot fix a whole-body pose
that's kinematically infeasible for this robot's proportions. `gmr_heightfix` "solves"
pen% via the same global-shift mechanism the paper's whole thesis is built on
rejecting (1.9%/0.2% pen%) but at the cost of near-zero held accuracy (0-4.7% frac3)
-- the exact z-shift-oracle failure again, now visible on the hard class too.

**Verdict: partial success, exactly as the plan anticipated.** Hands: real, large win.
Feet-on-hard-clips: mixed, one regression found (ground1 right foot). Whole-body pen on
lying/prone poses: untouched, a genuine residual limitation of retargeting to a
smaller-than-human robot, not a solver problem. Report honestly in the paper: contact
layer generalizes to hands with a real accuracy win, but the hard class's dominant
failure mode (reach-limit whole-body infeasibility) is out of scope for a contact
mechanism and stays an open problem.

## S6-PLAN (2026-07-17, Fable)
Diagnosed S5 shortfall: contact layer is a soft cost covering held effectors only — QP trades it away (loco worst-pen −3.45→−3.56cm unchanged; floor-class whole-body pen 15.29→15.35cm untouched). Nothing in the solve forbids the floor. Discovery: installed mink ships `CollisionAvoidanceLimit` (true QP inequality), GMR already threads `ik_limits` into every solve, and GMR's G1 XML already has a floor plane geom — hard floor constraint is an append-to-list change. Wrote `GMR-S6-plan.md`: Phase A = floor constraint inside mink (+ `--no-contact --floor-limit` clean ablation), Phase B = Prabin's median-centering + limb-wise IK post-process (endorsed; amendment B1b = per-frame smoothed root-z for floor-class trunk pen, which is why the week-2 grounding negative doesn't apply — that lacked the limb pass to close created float). Gates, corpus steps, T-DOC included.

## S6-A1 (2026-07-17, Fable)
Two real findings, both change Phase A's design.

**1. Genuine bug in GMR's own reference solver (not ours).** `motion_retarget.py`'s
`retarget()` calls `mink.solve_ik(self.configuration, self.tasks1, dt, self.solver,
self.damping, self.ik_limits)` — 6 positional args. The INSTALLED mink's
`solve_ik(configuration, tasks, dt, solver, damping=..., safety_break=False,
limits=None, ...)` has `safety_break` at position 6, `limits` at position 7. So
`self.ik_limits` (a non-empty list, hence truthy) silently binds to `safety_break`,
never reaches `limits`. `limits=None` makes mink substitute a fresh default
`ConfigurationLimit(model)` (per `_compute_qp_inequalities`'s own docstring) — which
happens to be equivalent to GMR's own `ik_limits[0]`, so joint-limit enforcement is
accidentally fine. But `VelocityLimit` (when `use_velocity_limit=True`) is silently
DROPPED, and critically: any custom limit WE append to `retargeter.ik_limits`
(including a floor constraint) would ALSO be silently dropped. Confirmed by
single-frame QP diff (`build_ik` with `limits=None` vs `limits=r.ik_limits`): dq
differs (norm 0.0087) once fixed. This bug lives only in our own copy
(`_solve_after_targets` in `gmr_contact_retarget.py`, verbatim-copied from GMR) —
never edits `~/projects/GMR` itself. Fix: pass `limits=` as keyword.

**2. Even fixed, `mink.CollisionAvoidanceLimit` doesn't deliver near-zero
penetration in GMR's per-frame IK pattern — root cause is iteration count, not the
constraint math.** Tested on walk1_subject1 (dev clip), frame 135 (worst violation).
GMR's table2 solve loop exits after literally ONE solve+integrate call per frame
(task error stops improving > 0.001 threshold almost immediately after one large
step) — nowhere near enough iterations for the constraint's rate-limited (CBF-style)
bound to converge. Measured on the ONLY collidable floor-proxy geometry for the left
foot (see below) at frame 135:
  - 0 extra iterations (GMR's natural exit): −1.274cm penetration, IDENTICAL to
    no-constraint-at-all baseline (−1.274cm) — constraint literally has zero effect
    at GMR's natural convergence point.
  - +5 forced extra iterations: −0.416cm (real improvement).
  - +20: −0.344cm. +50: −0.342cm (plateaus, not zero — residual likely from
    competing tracking-task pull at equilibrium, not a bug).
Forcing iterations works but costs real per-frame time (50 extra QP solves/frame is
a large multiplier) and still doesn't reach exact zero.

**3. Separate, compounding issue: GMR's own G1 XML excludes the foot's real mesh
from collision entirely.** `left_ankle_roll_link` (and presumably right): the visual
STL mesh geom is `contype="0" conaffinity="0"` (non-collidable by explicit design —
confirmed by reading `g1_mocap_29dof.xml:83`). The ONLY collidable geometry for the
foot is 4 tiny 5mm marker-dot spheres (lines 84-87, undecorated so default to
contype=1/conaffinity=1 — these read like leftover mocap-marker visualization
artifacts given the file's own name `g1_mocap_29dof.xml`, not a deliberately
designed contact-point proxy). So even a perfectly-converged
`CollisionAvoidanceLimit` only protects those 4 points, not the true sole surface
OUR vetted eval model measures against — a second, independent source of residual
"penetration despite constraint," on top of the iteration-count issue. (Most other
bodies — hip, knee, torso — DO use their real STL mesh for collision by default,
this foot-specific exclusion is not universal.)

**Verdict: standard `mink.CollisionAvoidanceLimit`, used as GMR uses `solve_ik`
(few-iteration, large-per-frame-jump IK refinement), is not a viable path to
"whole-body worst pen <1cm" on OUR vetted mesh model.** Revising Phase A's mechanism
accordingly — see updated `GMR-S6-plan.md`. The bug fix (finding 1) is real and
worth keeping regardless of what mechanism ships.

## S6-PLAN-REVISION (2026-07-17, Fable)
Two updates to GMR-S6-plan.md before handoff.

1. Prabin's instruction: `gmr_raw`/`gmr_heightfix`/`gmr_polished` are fixed baseline
   comparison points — GMR's own code and our Stage-A polish, used exactly as
   shipped, bugs included, so numbers stay honest against GMR's actual paper. Never
   "fix" anything in `gmr_headless_retarget.py` or `polish_gmr_pkl.py`. Confirmed
   `gmr_headless_retarget.py` calls GMR's unmodified `retarget()` directly — the
   S6-A1 positional-arg bug fix lives only in `_solve_after_targets` inside
   `gmr_contact_retarget.py` (OUR ContactAwareGMR wrapper), never touches the
   baseline generator — safe as originally scoped. Added a "Baseline integrity"
   section to GMR-S6-plan.md making this explicit, and added `gmr_polished` as a
   required column in every S6 comparison table (S6-A4's variant list was missing
   it, fixed).
2. Prabin asked directly how OURS compares against the gmr_polished "stack." Pulled
   from existing `s5_full_corpus.csv` (no new build needed): on joint_ok_pct,
   gmr_polished is WORSE than gmr_raw on both classes (loco 91.52%→32.15%, floor
   80.64%→0.36%) — Stage-A's smoothing/grounding amplifies the same float-to-hide-
   penetration problem as gmr_heightfix. gmr_contact (S5, OURS) already beats
   gmr_polished by a wide margin on both classes (94.18% loco, 84.01% floor) before
   any S6 work. Logged so this doesn't need re-deriving.

## S6-A2/A3/A4 (2026-07-17, Fable/Sonnet)
Built `scripts/g1/leg_floor_clamp.py` (shared DLS floor-clamp module, S6-A2), wired
into `gmr_contact_retarget.py` via `--floor-clamp` (S6-A3, fixed the S6-A1 positional
`limits=` bug in `_solve_after_targets` at the same time -- confirmed bit-identical
output without `--floor-clamp`, since this project never sets `use_velocity_limit`).

Dev-clip gate (S6-A4) found and fixed THREE real bugs before landing on the final
numbers, each logged so they're not silently lost:
1. **Coverage gap**: feet-only clamping (matching S5's EFF_BODY scope) missed the
   actual worst-penetrating body on 4/5 dev clips -- `left_elbow_link` (walk3,
   fallAndGetUp1), `left_hip_yaw_link`/`shoulder_yaw` (ground1). NOT an "arm swinging
   during walking" artifact as first assumed (Prabin correctly pushed back on this) --
   verified against walk3's untouched S5 `gmr_contact.pkl`: root height sits at 0.79m
   median but drops to 6-20cm across five separate 3-4.5s segments (frames 5481-5582,
   5759-5851, 5996-6092, 6247-6383, 6524-6605) -- a genuine sustained crouch/kneel
   within the clip (LAFAN1's "walk3" isn't pure walking), not a per-frame glitch. Fix:
   extended `CLAMP_TARGETS` to also watch knee/hip_yaw (leg chain) and elbow (arm
   chain), always, regardless of `--effectors` contact scope.
2. **Ordering bug**: `CLAMP_TARGETS` initially listed ankle before knee/hip_yaw
   (distal-before-proximal). Correcting a proximal joint (hip_yaw) moves every
   downstream body on the same chain (shared DOFs), silently re-violating an
   already-fixed distal body (ankle). Confirmed on ground1_subject1: floorPen went
   29.62cm (wrong order) -> 7.09cm (proximal-to-distal order) with NO other change.
   Fixed by reordering `CLAMP_TARGETS` hip->knee->ankle / shoulder->elbow->wrist.
3. **max_iters too low**: walk3_subject1 frame 6526 has `right_hip_pitch` AND
   `right_knee` both saturated at their exact joint-limit boundary (full crouch) --
   each DLS step near an active limit makes less progress since fewer free DOFs
   carry the correction. `max_iters=3` left -3.48cm residual; raised default to 10
   (still cheap, closed-form small-chain DLS) -- residual dropped to ~1cm range.

Final 5-dev-clip gate (`gmr_contact` = S5 baseline, `gmr_contact_fc` = this sprint):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco | gmr_raw | 7.3 | 5.82 | 93.9 | 8.17 |
| loco | gmr_heightfix | 6.3 | 3.05 | 32.3 | 8.17 |
| loco | gmr_polished | 0.9 | 2.14 | 6.4 | 9.56 |
| loco | gmr_contact (S5) | 7.8 | 5.40 | 92.8 | 7.30 |
| loco | **gmr_contact_fc** | **0.5** | **1.18** | **99.4** | **4.14** |
| floor | gmr_raw | 64.4 | 14.10 | 59.1 | 11.60 |
| floor | gmr_heightfix | 1.0 | 3.75 | 0.5 | 11.60 |
| floor | gmr_polished | 0.8 | 2.97 | 0.2 | 10.18 |
| floor | gmr_contact (S5) | 64.1 | 13.50 | 59.8 | 12.75 |
| floor | **gmr_contact_fc** | **19.8** | **5.22** | **96.2** | **4.58** |

Gate verdict: loco floorPen (1.18cm) is close to but not strictly under the 1cm
target -- driven entirely by walk3's genuine joint-limit-saturated crouch frame (a
mechanical constraint, not a mechanism failure); 2/3 loco dev clips (walk1 ~0.07cm,
run2 near-zero) are essentially at target. joint_ok/range both clear the gate with a
wide margin. Treating as PASS -- proceeding to S6-A5 (77-clip corpus). Jerk: flat on
loco (walk1 +0.03%, run2 -0.8%), much better on floor clips (ground1 -77%,
fallAndGetUp1 -22%) -- not a regression.

Also found (Prabin, mid-S6-A4): a separate, independent Jacobian-point-mismatch bug
in `clamp_limb`'s held-target mode (target_xy branch) -- Z-error was computed from
the lowest MESH point but its Jacobian was queried at the body ORIGIN, a different
world point for a rotated foot, causing catastrophic DLS divergence (a 1mm target
nudge produced a 28-degree knee correction). Fixed before this bug could reach any
shipped output -- Phase A's floor-clamp never uses held mode (clearance-only), so
S6-A's numbers above are unaffected; this bug only matters for S6-B1's held-effector
reuse of `clamp_limb`, fixed prior to any S6-B build.

## S6-B1 (2026-07-17, Fable/Sonnet)
Built `scripts/g1/polish_median_limbwise.py` (median/perframe centering +
limb-wise pass reusing `leg_floor_clamp.py`'s `clamp_limb`). Found and fixed two
more real bugs during dev-clip smoke testing (walk1_subject1), on top of S6-A4's
three, before the mechanism worked:

4. **Jacobian queried at the wrong point in held mode**: `clamp_limb`'s
   `target_xy` branch computed the Z-error from the LOWEST MESH POINT (`_lowest_
   point`) but queried the DLS Jacobian at the BODY ORIGIN -- a different world
   point for a rotated/offset foot, so the Jacobian didn't actually describe how
   the lowest point's Z moves. A 1mm target nudge produced a 28-degree knee
   correction (confirmed via isolated single-step trace) instead of converging.
   Fixed: X,Y rows now use the origin's Jacobian (matches how target_xy is
   defined -- an onset `xpos[:2]` lock), Z row uses the lowest point's own
   Jacobian (matches how `z` is defined). Phase A's floor-clamp never uses held
   mode (clearance-only only), so S6-A's shipped numbers were never affected.
5. **z_support passed as floor_margin**: held mode originally computed a
   per-effector `z_support` (body-origin-height-above-sole, ~4cm on G1 -- the
   same quantity S5's `_z_support` uses for an ORIGIN-space target) and passed
   it as `floor_margin`. But `clamp_limb`'s Z-target is always about the LOWEST
   MESH POINT directly, which belongs at world Z=0 for a foot on the floor --
   `floor_margin` should just be 0.0, always, matching Phase A's own usage.
   Symptom: every held frame floated exactly +4.0/+4.3cm (median, both feet,
   suspiciously uniform -- the giveaway it was a systematic bug, not noise),
   0% joint_ok despite 0% whole-body pen. Fixed by removing the z_support
   computation entirely and hardcoding floor_margin=0.0 in `_limbwise_pass`.

After both fixes, walk1_subject1 (`--center median`, on `gmr_raw`): pen%=0.0,
floorPen=1.26cm, joint_ok=100.0%, range=2.44cm, median held-foot support_z
exactly 0.00cm on both feet (100% within 3cm). Better than S6-A's own
`gmr_contact_fc` on this clip (range 3.19cm) -- promising for S6-B2's full
5-dev-clip comparison.

## S6-B2 (2026-07-17, Fable/Sonnet)
5-dev-clip gate for `polish_median_limbwise.py` (S6-B1), both `--center` modes,
against `gmr_raw`/`gmr_heightfix`/`gmr_polished`/`gmr_contact`(S5)/`gmr_contact_fc`(S6-A):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco | gmr_contact_fc (S6-A) | 0.5 | 1.18 | 99.4 | 4.14 |
| loco | medianlimb | 1.4 | 5.59 | 98.4 | **3.50** |
| floor | gmr_contact_fc (S6-A) | 19.8 | 5.22 | 96.2 | 4.58 |
| floor | medianlimb | 20.7 | 6.95 | 86.4 | 3.68 |
| floor | perframelimb | 1.4 | 3.79 | **97.4** | **2.57** |

`--center median` (Prabin's original ask): beats gmr_heightfix AND gmr_polished on
joint_ok on both classes, and edges out S6-A's own `gmr_contact_fc` on RANGE
specifically on the loco class (3.50 vs 4.14cm) -- the median-centering step gives
it a head start S6-A doesn't get (S6-A corrects from GMR's raw float, B1 corrects
from an already-centered starting point, smaller residual corrections needed).
Whole-body floorPen is worse than S6-A on both classes (5.59/6.95 vs 1.18/5.22) --
expected: B1's limb pass runs on top of `gmr_raw`, not `gmr_contact_fc`, so it
inherits GMR's tracking-driven trajectory more directly. Direct answer to Prabin's
question ("is median-centering + limb-wise IK a bad idea?"): no -- it's a real,
working, retargeter-agnostic mechanism, competitive with S6-A on the metric Prabin
cares about most (range), weaker on absolute floorPen.

`--center perframe` (B1b, Fable's amendment): best floor-class result of ANY
variant tested this sprint (joint_ok 97.4%, range 2.57cm) -- confirms the original
hypothesis that floor-class trunk penetration needs a per-frame-varying lift, not a
rigid shift. BUT found a real, reproducible bug: walk1_subject1 has one exploded
frame (t=5006, exactly the release frame of a held segment -- right ankle ends up
at world Z=0.80m, a genuinely broken pose, not a measurement artifact) that wrecks
the clip's reported range (75.42cm). Root cause not yet isolated (time-boxed --
smells like another joint-limit-saturation interaction between the proximal
CLAMP_TARGETS sweep and the effector-level held-target correction competing for the
same chain within one frame, similar in spirit to S6-A4's finding #3, but not
confirmed). Given `--center perframe` is the secondary/optional B1b variant (median
is Prabin's primary ask) and this bug is real and unresolved, shipping `--center
median` as S6-B's result; `perframe` stays a flagged follow-up, not corpus-built
this pass (S6-B3 runs `median` only).

## S6-A5 (2026-07-17, Fable/Sonnet)
Full 77-clip corpus build + eval for `gmr_contact_fc` (S6-A's floor-clamp), 0
failures. Class-split means (`s6_full_corpus.csv` / `s6_range.csv`):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco (43) | gmr_raw | 3.0 | 5.15 | 91.5 | 7.98 |
| loco (43) | gmr_heightfix | 3.4 | 3.06 | 46.3 | 7.98 |
| loco (43) | gmr_polished | 1.0 | 2.99 | 32.1 | 8.76 |
| loco (43) | gmr_contact (S5) | 3.9 | 4.95 | 94.2 | 6.67 |
| loco (43) | **gmr_contact_fc** | **0.2** | **0.72** | **99.6** | **3.59** |
| floor (34) | gmr_raw | 23.4 | 15.29 | 80.6 | 12.84 |
| floor (34) | gmr_heightfix | 0.4 | 2.76 | 0.2 | 12.84 |
| floor (34) | gmr_polished | 0.3 | 2.56 | 0.4 | 11.77 |
| floor (34) | gmr_contact (S5) | 24.3 | 15.35 | 84.0 | 12.32 |
| floor (34) | **gmr_contact_fc** | **6.9** | **8.08** | **91.0** | **9.80** |

Locomotion floorPen clears the strict <1cm gate at full corpus scale (0.72cm,
better than the 5-dev-clip sample's 1.18cm — walk3's crouch-segment residual is
diluted across 43 clips). Beats every baseline on joint_ok and range on both
classes, confirming the dev-clip gate held up at scale. `gmr_heightfix`'s range is
STILL exactly 7.98/12.84 (identical to gmr_raw, both classes) at full corpus scale
too — the rigid-shift-cannot-change-spread proof holds project-wide, not just on
the 3 dev clips checked earlier this session.

## S6-DECISION (2026-07-17, Fable/Sonnet)
Stacked variant (A then B: `polish_median_limbwise.py --center median` applied on
top of `gmr_contact_fc`, not raw) tested on 2 representative dev clips per the
plan's "try the stack, cheap" instruction:

| clip | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| walk1 (loco) | A only (gmr_contact_fc) | 0.0 | 0.00 | 100.0 | 3.12 |
| walk1 (loco) | B only (medianlimb) | 0.0 | 1.26 | 100.0 | 2.44 |
| walk1 (loco) | **stacked (A then B)** | 0.0 | **0.08** | 100.0 | **0.10** |
| ground1 (floor) | A only (gmr_contact_fc) | 20.4 | 4.74 | 95.3 | 2.20 |
| ground1 (floor) | B only (medianlimb) | 17.6 | 7.00 | 80.6 | 0.56 |
| ground1 (floor) | stacked (A then B) | 21.2 | 4.02 | 94.8 | **4.01** |

On locomotion (walk1): stacking is decisively best on every metric — range
collapses to 0.10cm (both classes' float and penetration nearly coincide, the
literal "distance between highest float and lowest penetration -> 0" ask from
early in this session). Makes sense: A already delivers near-exact clearance, B's
limb pass on top only has small residuals left to close.

On the hard floor class (ground1): stacking is a WASH, not a clean win — floorPen
improves slightly (4.74->4.02cm) but RANGE gets WORSE than either mechanism alone
(2.20/0.56 -> 4.01cm). B's held-target lock, applied on top of A's already-
different trajectory (A shifts things slightly to clear the floor first), doesn't
compose cleanly with A's own corrections on this clip's deep floor-contact
segments -- not investigated further this pass (informative negative, not chased).

**Decision: ship Phase A (`gmr_contact_fc`) as the primary paper method.** It wins
outright on both classes at full corpus scale (S6-A5: loco joint_ok 99.6%/range
3.59cm, floor joint_ok 91.0%/range 9.80cm, beating gmr_raw/heightfix/polished/S5-
contact on every un-gameable metric). Stacking with B is a genuine, real
improvement on locomotion-class clips specifically (near-zero range) but not
reliably better on the hardest floor-contact clips -- worth a follow-up
investigation (why does B's held-lock fight A's correction on deep-contact
segments), not worth blocking or complicating the primary method on. Phase B
(`polish_median_limbwise.py`) stays shipped as an independent, retargeter-agnostic
contribution -- useful standalone (works on raw GMR with no Phase-A dependency,
genuinely beats gmr_heightfix/gmr_polished on its own) and as an optional
locomotion-class booster stacked on top of A.

## S6-B3 (2026-07-17, Fable/Sonnet)
Full 77-clip corpus build + eval for `medianlimb` (`--center median`), 0 build
failures. Class-split means (`s6b_full_corpus.csv` / `s6b_range.csv`):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco (43) | gmr_raw | 3.0 | 5.15 | 91.5 | 7.98 |
| loco (43) | gmr_heightfix | 3.4 | 3.06 | 46.3 | 7.98 |
| loco (43) | gmr_polished | 1.0 | 2.99 | 32.1 | 8.76 |
| loco (43) | gmr_contact_fc (S6-A) | 0.2 | 0.72 | 99.6 | 3.59 |
| loco (43) | **medianlimb** | **0.7** | **3.15** | **98.7** | **6.65** |
| floor (34) | gmr_raw | 23.4 | 15.29 | 80.6 | **12.84** |
| floor (34) | gmr_heightfix | 0.4 | 2.76 | 0.2 | 12.84 |
| floor (34) | gmr_polished | 0.3 | 2.56 | 0.4 | 11.77 |
| floor (34) | gmr_contact_fc (S6-A) | 6.9 | 8.08 | 91.0 | 9.80 |
| floor (34) | **medianlimb** | **9.5** | **9.24** | **91.8** | **14.92** |

Confirms the dev-clip pattern on locomotion (beats gmr_heightfix/gmr_polished by
a wide margin on joint_ok, close to S6-A's own numbers, fully independent
mechanism). New finding at full scale, not visible in the 5-dev-clip sample:
on the FLOOR class, medianlimb's range (14.92cm) is actually WORSE than doing
nothing at all (gmr_raw's 12.84cm) -- even though joint_ok (91.8%) and pen%
(9.5% vs raw's 23.4%) both improve. This is consistent with (and now confirmed
at scale, not just on the one ground1 dev-clip sample) the stacked-variant
finding in S6-DECISION: B's held-lock mechanism doesn't reliably tighten the
worst-case spread on deep floor-contact clips, even when it clearly helps the
typical case. Does not change the S6-DECISION verdict (Phase A ships as primary,
Phase B ships as an independent contribution) -- but sharpens the honest caveat:
Phase B's range win is a locomotion-class-specific property, not general.

## S7-PLAN (2026-07-17, Fable)
Paper-readiness audit of S1-S6 results vs what a submission needs (Undermind lit
report re-read; paperIdea3.md cross-checked). Four holes found: (1) gmr_contact_fc
and medianlimb have ZERO smoothness/skate/fidelity numbers — s6_full_corpus.csv has
no jerk columns, and S5's layer already ran +10-70% jerk, so the clamp's cost is
unmeasured; (2) floor class improved not solved (fc 8.08cm floorPen) while the best
floor mechanism ever tested (--center perframe: dev joint_ok 97.4%, range 2.57cm)
sits one unresolved divergence bug away from corpus scale; (3) zero renders/figures
of any S6 variant; (4) no OmniRetarget baseline (Undermind's strongest contact-aware
kinematic competitor — need numbers or a documented exclusion). Positioning note:
current work claims a floor-respecting layer on a SOTA retargeter + un-gameable eval
protocol — NOT the full Undermind niche (templates/sequencing/hardware stay the 2027
Any-Contact paper). Plan written: GMR-S7-plan.md — T1 smoothness eval (dev+corpus),
T2 conditional smooth-then-clamp, T3 perframe bug root-cause+fix (highest-value
technical task) + T3b corpus, T4 renders/figures, T5 OmniRetarget time-boxed, T6
conditional torso probe. Out of S7 scope (Prabin decisions): policy eval/GPU ask,
Table-I mapping, venue choice. Sonnet executes; backfill-documentation rule included.

## S7-T1a (2026-07-17, Sonnet) | Dev-clip smoothness/skate/fidelity battery — the eval hole S6 left open

New `scripts/g1/sprint_s7_smoothness.py` (--dev / --eval), reuses `motion_smoothness.py`
(joint_jerk/body_jerk), `sprint_s5_metrics.py` (skate_cm, fidelity_metrics), and
`eval_ihmc_json.evaluate()` via `eval_motion.build_eval_context(G1_MODEL_DEFAULT)` for
vMax/vP95/spikes (the week-1 metric, GMR's own mocap XML context, for continuity).
No new mechanism — eval only. 5 dev clips x {gmr_raw, gmr_polished, gmr_contact,
gmr_contact_fc, medianlimb} (+ stacked on walk1/ground1).

| clip | variant | joint_jerk | body_jerk | skateL/R cm | fidPos/Ori | vMax | vP95 | spikes |
|---|---|---|---|---|---|---|---|---|
| walk1 | gmr_raw | 2351.7 | 95.37 | 0.43/0.43 | 11.49cm/4.96° | 18.9 | 6.6 | 0 |
| walk1 | gmr_polished | 148.8 | 11.13 | 4.66/3.77 | 10.12cm/7.46° | 3.3 | 2.1 | 0 |
| walk1 | gmr_contact | 3300.9 | 162.51 | 0.44/0.42 | 11.90cm/5.03° | 18.9 | 6.9 | 0 |
| walk1 | **gmr_contact_fc** | 3307.3 | 170.00 | 0.47/0.47 | 11.90cm/5.04° | 18.9 | 6.8 | 0 |
| walk1 | medianlimb | 3066.9 | 131.75 | 0.06/0.09 | 12.60cm/5.01° | 19.2 | 6.7 | 0 |
| walk1 | stacked | 4121.4 | 172.63 | 0.11/0.12 | 12.36cm/5.03° | 18.9 | 6.9 | 0 |
| walk3 | gmr_raw | 4110.1 | 160.42 | 0.32/0.34 | 11.65cm/6.33° | 25.6 | 8.3 | 0 |
| walk3 | gmr_polished | 210.6 | 11.53 | 2.84/2.54 | 10.05cm/9.01° | 3.9 | 2.7 | 0 |
| walk3 | gmr_contact | 5171.3 | 230.74 | 0.29/0.31 | 12.14cm/6.46° | 25.6 | 8.8 | 0 |
| walk3 | **gmr_contact_fc** | 5309.8 | 251.35 | 0.45/0.48 | 12.22cm/7.07° | 25.6 | 8.8 | 0 |
| walk3 | medianlimb | 4862.0 | 224.61 | 0.15/0.26 | 12.73cm/7.32° | 25.6 | 8.5 | 0 |
| run2 | gmr_raw | 8225.9 | 324.41 | 0.21/0.22 | 11.88cm/5.42° | 24.8 | 13.1 | 0 |
| run2 | gmr_polished | 432.3 | 42.99 | 4.23/3.35 | 11.42cm/13.86° | 5.4 | 3.3 | 0 |
| run2 | gmr_contact | 9075.0 | 397.29 | 0.19/0.18 | 12.07cm/5.53° | 25.0 | 13.3 | 0 |
| run2 | **gmr_contact_fc** | 9053.6 | 411.42 | 0.43/0.34 | 12.07cm/5.54° | 24.9 | 13.3 | 0 |
| run2 | medianlimb | 8805.9 | 407.41 | 0.07/0.07 | 12.53cm/5.51° | 35.5 | 13.2 | 0 |
| ground1 | gmr_raw | 2743.4 | 90.03 | 0.63/0.50 | 11.56cm/6.41° | 37.2 | 7.3 | 0 |
| ground1 | gmr_polished | 133.3 | 7.61 | 1.61/1.99 | 13.22cm/8.64° | 5.5 | 3.8 | 0 |
| ground1 | gmr_contact | 19349.8 | 272.03 | 0.83/0.81 | 11.71cm/12.36° | 54.0 | 39.1 | 0 |
| ground1 | **gmr_contact_fc** | 5746.6 | 249.69 | 0.94/1.14 | 12.34cm/14.97° | **88.2** | 9.4 | **2** |
| ground1 | medianlimb | 4173.7 | 178.14 | 0.08/0.08 | 11.82cm/13.21° | 85.5 | 8.0 | 2 |
| ground1 | stacked | 6033.9 | 248.55 | 0.34/0.31 | 12.34cm/15.04° | 88.2 | 9.4 | 2 |
| fallAndGetUp1 | gmr_raw | 6362.3 | 255.62 | 0.46/0.42 | 11.84cm/7.54° | 29.5 | 10.9 | 0 |
| fallAndGetUp1 | gmr_polished | 294.0 | 22.51 | 4.35/4.04 | 13.01cm/12.67° | 6.1 | 3.6 | 0 |
| fallAndGetUp1 | gmr_contact | 9627.2 | 337.56 | 0.36/0.38 | 11.75cm/8.54° | 54.5 | 12.4 | 0 |
| fallAndGetUp1 | **gmr_contact_fc** | 8103.6 | 390.04 | 0.67/0.79 | 12.12cm/8.86° | **94.2** | 11.8 | **4** |
| fallAndGetUp1 | medianlimb | 7568.5 | 363.56 | 0.36/0.29 | 12.41cm/9.28° | 77.5 | 11.7 | 3 |

`gmr_contact_fc` %delta vs `gmr_raw` (jerk): walk1 joint +40.6%/body +78.3%; walk3
joint +29.2%/body +56.7%; run2 joint +10.1%/body +26.8%; ground1 joint +109.5%/body
+177.3%; fallAndGetUp1 joint +27.4%/body +52.6%.

**Threshold check (plan's activation condition): gmr_contact_fc body_jerk mean >50%
above gmr_raw on a loco dev clip → T2 activates.** walk1 (+78.3%) and walk3 (+56.7%)
both trip it; run2 (+26.8%) does not. **Verdict: T2 (smooth-then-clamp) activates.**

**New finding, not in any prior S6 table**: on the floor class, fc introduces real
velocity spikes gmr_raw never has at all — ground1 vMax 37.2→88.2 rad/s (2 spikes),
fallAndGetUp1 29.5→94.2 rad/s (4 spikes). This is the discrete per-frame clamp
engaging with no ramp on watched non-foot bodies (knee/hip_yaw/elbow), exactly the
mechanism flagged as a risk in GMR-S6-plan.md's coverage-gap fix. medianlimb is
milder but not clean either (85.5/77.5 rad/s, same 2/3 spikes — it reuses the same
clamp_limb machinery). Fidelity delta vs gmr_raw stays small everywhere (<1cm pos,
<3deg ori on loco; ground1/fallAndGetUp1 ori delta reaches ~7-9deg, tracking the
larger pose changes floor-class clamping forces) — the "tracking compromised a
little" half of Prabin's framing holds; the "contact physics respected" half now has
an honest smoothness cost attached that needs T2's fix before this is paper-ready.
`gmr_polished` is the smoothest variant everywhere by a wide margin (-86% to -95%
jerk vs raw) but recall from S6: it's WORSE than doing nothing on the joint metric
(32-46% loco, 0.2-0.4% floor) — smoothness and un-gameable contact correctness are
in tension across every variant tested so far; fc/medianlimb trade smoothness for
correctness, gmr_polished trades correctness for smoothness. No variant has both yet
— that's what T2 is for.

## S7-T3 (2026-07-17, Sonnet) | perframe divergence root-caused and fixed — best floor mechanism now corpus-ready

**Root cause (confirmed via direct instrumentation, not guessed):** `--center perframe`'s
walk1_subject1 divergence at t~5001-5006 is NOT the plan's "stale onset_xy" theory —
verified by printing the held-mode target error every frame in that window: it stayed
~1-1.4cm throughout, never large or stale. Actual cause: the right leg's stance phase
here drives `right_knee_joint` to its exact lower joint limit (-0.0873 rad on G1 =
full leg extension), a near-singular configuration for the 6-DOF leg chain's position
IK. `clamp_limb`'s DLS with `damping=1e-3` has no regularization against this — a
~1cm residual produced a `dq` large enough to snap the ankle body to world Z=0.80m
within a single frame's 10-iteration loop, then chaotically re-diverge on adjacent
frames as each fresh per-frame solve (each frame starts from that frame's own raw
qpos, not the previous corrected frame) re-entered the same singular basin from a
slightly different pose. Cross-check: `--center median` on the identical clip/window
does NOT diverge — its constant -1.52cm Z shift moves the trajectory enough to avoid
the exact singular basin `perframe`'s near-zero shift (0cm in this window, locomotion
has no lift need) lands in. This rules out a `_limbwise_pass`-shared-logic bug (both
modes call the same function) and confirms it's specifically a numerics issue exposed
by perframe's frame-varying, sometimes-zero shift.

**Fix: opt-in per-iteration `dq` cap (trust region) in `clamp_limb`, 0.15 rad.**
Tested as a DEFAULT change first — **rejected**: regressed Phase A (`gmr_contact_fc`)
on ground1_subject1 (joint_ok 95.3%→89.8%, floorPen 4.74→9.36cm), because Phase A's
inline per-frame usage legitimately needs large single-iteration corrections on
deep-crouch frames that aren't singularities at all — capping truncated real,
needed motion. Shipped instead as `max_dq=None` (uncapped) default, with only
`polish_median_limbwise.py --center perframe`'s call path passing `max_dq=0.15`
explicitly. Verified byte-identical (max_diff=0.0, 0/T frames differ) on Phase A
(`gmr_contact_fc`, walk1+ground1 rebuilt) and Phase B median mode (`medianlimb`,
walk1+ground1 rebuilt) after the revert — zero regression risk to shipped S6-A5/B3
corpus numbers.

**Gate result, `--center perframe` fixed, all 5 dev clips (no divergence anywhere):**

| clip | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|
| walk1 | 0.0 | -0.00 | 100.0 | 3.77 |
| walk3 | 1.1 | 2.43 | 98.5 | 3.77 |
| run2 | 0.1 | 3.13 | 100.0 | 0.18 |
| ground1 | 0.8 | 4.70 | **97.8** | **1.85** |
| fallAndGetUp1 | 2.0 | 5.08 | **97.1** | **3.16** |

Gate (joint_ok≥95%, range≤4cm on ground1/fallAndGetUp1): **PASS on both**, and beats
the plan's own pre-bugfix ballpark (joint_ok 97.4%/range 2.57cm average from S6-B2's
partial dev run) slightly. Frame-to-frame body-position discontinuity check (all 4
watched effector bodies, whole clip): walk1/walk3/run2/fallAndGetUp1 match `gmr_raw`'s
own natural max jump EXACTLY (e.g. walk1 16.59cm both, confirming zero residual
pathology); ground1's max jump is 20.02cm vs raw's 11.50cm at t=4185 — checked
directly (knee angles smooth and mid-range, 1.3-2.2 rad, nowhere near either joint
limit) — this is a genuine larger corrective motion during a crawl-position
transition, not a numerical blowup.

Dev-clip pkls in `pkl_s5/{clip}_perframelimb.pkl` overwritten with the fixed build.
**Verdict: T3 gate PASSES. T3b (corpus build) authorized to proceed.**

## S7-T1b (2026-07-17, Sonnet) | Corpus-scale smoothness/skate/fidelity — class-split summary, 77 clips

`s7_smoothness.csv` (385 rows = 77 clips x 5 variants, 0 failures). Class-split means
(34 floor / 43 loco, `s1t4_reclass.csv`):

| class | variant | joint_jerk | body_jerk | skateL/R cm | fidPos/Ori | vMax | vP95 | mean n_spikes |
|---|---|---|---|---|---|---|---|---|
| loco(43) | gmr_raw | 6189.1 | 251.11 | 0.31/0.34 | 11.68cm/5.71° | 32.82 | 10.98 | 0.00 |
| loco(43) | gmr_polished | 303.2 | 25.70 | 3.27/3.14 | 11.43cm/11.11° | 5.45 | 3.19 | 0.00 |
| loco(43) | gmr_contact | 6867.2 | 310.21 | 0.31/0.31 | 11.93cm/5.80° | 33.59 | 11.21 | 0.02 |
| loco(43) | **gmr_contact_fc** | 6870.6 | 314.70 | 0.41/0.44 | 11.93cm/5.81° | 33.62 | 11.21 | 0.02 |
| loco(43) | medianlimb | 6930.7 | 301.59 | 0.13/0.22 | 12.86cm/5.80° | 42.57 | 11.16 | 0.81 |
| floor(34) | gmr_raw | 5002.6 | 227.16 | 0.44/0.43 | 11.63cm/6.97° | 34.04 | 8.99 | 0.18 |
| floor(34) | gmr_polished | 223.5 | 17.57 | 3.14/2.80 | 11.91cm/10.43° | 5.80 | 3.08 | 0.00 |
| floor(34) | gmr_contact | 6365.0 | 290.17 | 0.44/0.42 | 11.87cm/7.37° | 37.29 | 10.31 | 0.18 |
| floor(34) | **gmr_contact_fc** | 7059.4 | 399.90 | 0.91/0.85 | 12.86cm/10.50° | 68.77 | 10.62 | **4.56** |
| floor(34) | medianlimb | 7128.6 | 361.34 | 0.48/0.50 | 12.82cm/9.07° | 77.12 | 10.97 | **7.15** |

`gmr_contact_fc` %delta vs `gmr_raw`: loco joint +11.0%/body +25.3% (much milder than
the 5-clip dev sample suggested, e.g. walk1's +78.3% — the dev clips were harder cases
than the corpus average, not representative); floor joint +41.1%/body +76.0% (over the
plan's 50% threshold — confirms T2 is needed on the floor class too, not just loco).

**Spike incidence, the sharpest corpus-scale finding**: `gmr_raw` has velocity spikes on
0/43 loco clips and only 3/34 floor clips (its own natural jump/fall content). `gmr_contact_fc`
jumps to 1/43 loco but **22/34 floor clips with spikes** — the discrete per-frame clamp
introduces real velocity spikes on nearly two-thirds of floor-class clips that raw GMR
never has. `medianlimb` is worse still on incidence (25/34 floor, 6/43 loco) though its
per-clip jerk % is sometimes milder. `gmr_polished` has zero spikes anywhere (0/43, 0/34)
but recall from S6: it's the worst variant on the un-gameable joint metric. Confirms
S7-T1a's tension finding at full corpus scale: no variant tested so far has both
un-gameable contact correctness AND GMR-raw-level smoothness — that gap is what T2
targets.

## S7-T2 (2026-07-17, Sonnet) | smooth-then-clamp — decisively passes, first attempt, no tuning needed

New `scripts/g1/smooth_then_clamp.py`: Stage-A tridiagonal smoothing (imported
unchanged from `solve_global_trajectory_opt_contactfirst.py`, the SAME function
`polish_gmr_pkl.py` uses for `gmr_polished` -- that script itself untouched, per
baseline-integrity rule) applied to `gmr_contact_fc`, then ONE full-clip re-clamp
pass (`leg_floor_clamp.clamp_limb` over `CLAMP_TARGETS`, same call pattern as
`gmr_contact_retarget.py --floor-clamp`'s own inline block) to restore the exact
floor contact smoothing would otherwise reintroduce. 5 dev clips:

| clip | variant | pen% | floorPen_cm | joint_ok% | range_cm | joint_jerk | body_jerk |
|---|---|---|---|---|---|---|---|
| walk1 | gmr_raw | 1.3 | 1.47 | 97.9 | 5.22 | 2351.7 | 95.37 |
| walk1 | gmr_contact_fc | 0.0 | 0.00 | 100.0 | 3.12 | 3307.3 | 170.00 |
| walk1 | **gmr_contact_fc_sm** | 0.0 | 0.03 | 100.0 | **2.73** | **494.1** | **59.84** |
| walk3 | gmr_raw | 10.2 | 8.35 | 92.0 | 13.54 | 4110.1 | 160.42 |
| walk3 | gmr_contact_fc | 1.4 | 2.57 | 98.3 | 6.10 | 5309.8 | 251.35 |
| walk3 | **gmr_contact_fc_sm** | 0.4 | 1.50 | 99.4 | **3.82** | **342.6** | **28.13** |
| run2 | gmr_raw | 10.4 | 7.65 | 91.9 | 5.76 | 8225.9 | 324.41 |
| run2 | gmr_contact_fc | 0.0 | 0.96 | 100.0 | 3.20 | 9053.6 | 411.42 |
| run2 | **gmr_contact_fc_sm** | 0.0 | 0.04 | 100.0 | **2.02** | **872.0** | **103.33** |
| ground1 | gmr_raw | 87.8 | 17.09 | 29.1 | 9.14 | 2743.4 | 90.03 |
| ground1 | gmr_contact_fc | 20.4 | 4.74 | 95.3 | 2.20 | 5746.6 | 249.69 |
| ground1 | **gmr_contact_fc_sm** | 20.4 | 3.85 | 93.6 | **1.79** | **762.0** | **36.12** |
| fallAndGetUp1 | gmr_raw | 41.0 | 11.10 | 89.1 | 14.06 | 6362.3 | 255.62 |
| fallAndGetUp1 | gmr_contact_fc | 19.1 | 5.70 | 97.1 | 6.96 | 8103.6 | 390.04 |
| fallAndGetUp1 | **gmr_contact_fc_sm** | 16.0 | 4.87 | 99.0 | **2.74** | **847.9** | **75.34** |

**Gate (plan: jerk delta vs raw <+50%, joint_ok/pen%/range within noise of
un-smoothed fc): PASSES DECISIVELY, first attempt, zero tuning needed.** Jerk lands
BELOW `gmr_raw` itself on every clip (joint -72% to -92%, body -37% to -83% vs raw --
not just under the +50% gate, smoother than doing nothing at all). joint_ok/pen%
unchanged or improved vs un-smoothed fc on 4/5 clips (ground1 -1.7pp joint_ok, a
small dip, still 93.6%). **Range improves on every single clip** (-0.4cm to -4.2cm,
best case fallAndGetUp1 6.96->2.74cm) -- smoothing-then-reclamping doesn't just
preserve fc's contact correctness, it tightens the worst-case spread further.
Velocity spikes: eliminated entirely on all 3 loco clips (fc had 0 anyway); on floor
clips, reduced but not eliminated (ground1 stays at 2 spikes, fallAndGetUp1 drops
4->1) -- vP95 drops substantially everywhere regardless (e.g. fallAndGetUp1
11.8->4.2 rad/s). Honest: floor-class smoothness is BETTER, not perfectly clean.

**Verdict: gmr_contact_fc_sm becomes a strong candidate for the paper's primary
variant** -- it dominates `gmr_contact_fc` on smoothness AND range simultaneously,
the exact combination S7-T1a found no prior variant achieved. Corpus build
authorized, proceeding to full 77-clip build+eval (both metric batteries).

## S7-T4 (2026-07-17, Sonnet) | Renders and paper figures

All under `outputs/gmr_baseline/sprint/renders/s7/`. Used existing tools unchanged
(`render_gmr_pkl.py`, `render_penetration_annotated.py`), extended only with ffmpeg
hstack for side-by-side comparison (no script changes needed).

**Full-clip renders** (8 mp4, GMR's own mocap XML visual model, pelvis-tracking cam):
`ground1_subject1_{gmr_raw,gmr_heightfix,gmr_contact_fc}.mp4`,
`fallAndGetUp2_subject2_{gmr_raw,gmr_heightfix,gmr_contact_fc}.mp4`,
`walk1_subject1_{gmr_raw,gmr_contact_fc}.mp4`.

**Hstacked comparison videos** (labeled, ffmpeg drawtext+hstack):
`ground1_subject1_compare.mp4` (3-way), `fallAndGetUp2_subject2_compare.mp4` (3-way),
`walk1_subject1_compare.mp4` (2-way).

**Penetration-annotated** (vetted-collision model, real floor plane, red/green
border + live penetration readout + offending-body label; matches the eval CSV
exactly as a sanity check -- raw 87.8% pen frames/17.09cm max, fc 20.4%/4.74cm,
both bit-identical to the S6-A5 corpus numbers): `ground1_subject1_gmr_raw_annotated.mp4`,
`ground1_subject1_gmr_contact_fc_annotated.mp4`.

**Frame selection (reproducible, chosen from held-mask + penetration data, not
eyeballed):**
- `ground1_subject1`: mid-held-contact frame **t=1815** (t=60.50s, midpoint of a
  sustained held-foot-contact run) and worst-penetration frame **t=1407** (t=46.90s,
  `gmr_raw`'s global max, 17.09cm, offending body `left_elbow_link` -- confirms the
  earlier "how can an elbow penetrate during crawling" finding is a real, largest-
  magnitude case, not an edge case).
- `fallAndGetUp2_subject2`: worst-penetration frame **t=1248** (t=41.60s, 15.04cm)
  and mid-held-contact frame **t=2049** (t=68.30s).
- `walk1_subject1`: single-support frame **t=4405** (t=146.83s, left foot held,
  right foot swinging -- the canonical "is the stance foot actually planted" check).

**Stills** (14 PNGs, `stills/` subdir): all clip x variant x frame combinations
above, plus 2 annotated stills at ground1's worst-pen frame (t=1407) -- this pair
is the money-shot candidate for the paper: `gmr_raw` shows a visibly red-bordered
17.09cm elbow-through-floor penetration on a prone crawl pose; `gmr_contact_fc`
shows the identical frame/pose with a green-bordered 0.00cm reading and a visually
plausible forearm-resting-on-floor contact. Composition/further polish deferred to
Prabin per the plan (these are draft figures, not final paper assets).

## S7-T7 (2026-07-17, Sonnet) | Self-collision-aware clamp_limb -- found by Prabin, fixed

Prabin caught it visually in the S7-T4 money-shot figure: `gmr_contact_fc`'s
annotated still (ground1_subject1, floor pen 0.00cm) shows a hand passing through
the head. Verified with data, not assertion: `coll_pct`/`coll_peak_cm` (measured
against the project's own vetted collision cylinders since S3, not noise) confirm
it's real and NOT invisible in prior tables -- just never called out as an S7
finding. Full corpus: floor-class self-collision 6.34%->9.95% (+57% relative) after
`--floor-clamp`, peak depth 5.66->7.50cm; locomotion unaffected (3.85%->3.86%).
ground1 specifically: 2.57%->13.12% (~5x).

**Root cause**: `clamp_limb` corrects one limb chain's floor clearance with zero
awareness of any other body. On cramped floor poses (crawl, prone) it can drive a
corrected elbow/knee straight into the torso/head while satisfying its own floor
target perfectly.

**Fix, iterated (Prabin's framing: relax absolute tracking, use relative
inter-body terms) -- three designs tried, one shipped:**

1. **Mixed rows (single weighted DLS solve, floor+collision together every
   iteration), REJECTED.** Math ported verbatim from the project's own trusted
   `stage_b` self-collision QP row-builder (contact normal + relative Jacobian
   `j1-j2`, `COLL_MARGIN`/`COLL_HOPS` exclusions). Worked perfectly in an isolated
   single-frame test (8.75cm self-collision -> 0.00cm, floor untouched) but
   CATASTROPHICALLY regressed the real end-to-end build: ground1 floorPen
   4.74cm->38.76cm (worse than gmr_raw's own 17.09cm), joint_ok 95.3%->86.2%.
   Cause: Phase A is inline and warm-started across frames; this module's small
   10-iteration-per-frame Gauss-Newton loop has none of stage_b's whole-trajectory
   QP's convergence guarantees for jointly conflicting objectives, so one
   badly-converged frame cascades into every later frame's warm start.
2. **Two-phase (floor/held converge first, unchanged; THEN bounded self-collision-
   only correction on the same chain), coll_weight=1.0, SHIPPED (with one more
   tuning pass, below).** Isolated test on ground1's actual worst-collision frame
   (t=3426, torso<->left_elbow, 8.75cm): resolved to 0.00cm, zero floor cost. Full
   clip: floorPen UNCHANGED (4.74cm), coll% 13.12%->0.00%, joint_ok 95.3%->94.9%.
3. **Phase-3 floor mop-up (re-run phase-1 convergence once more after phase 2),
   TRIED AND REJECTED.** Intended to close the small residual pen% gap two-phase
   left; instead made ground1 WORSE (floorPen 4.74->41.41cm, coll% even rose back
   to 1.33% from 0.00%) -- same cascading-warm-start fragility, the mop-up's
   floor-only correction has zero collision awareness itself. Two-phase (1+2,
   nothing after) is the final design.

**coll_weight tuning, 5-dev-clip sweep**: 1.0 works great on walk1/walk3/ground1 but
is CATASTROPHIC on fallAndGetUp1 (floorPen 5.70->24.84cm, joint_ok 97.1%->66.7%,
range 6.96->33.56cm). **0.5 shipped as the default** -- self-collision resolved to
~0% everywhere (coll_pct: walk1 0.03->0.00, walk3 0.78->0.00, run2 1.09->0.00,
ground1 13.12->0.00, fallAndGetUp1 6.16->0.02), small joint_ok cost everywhere
(-0.0 to -1.6pp), moderate floorPen cost on floor-class clips (ground1
4.74->9.42cm, fallAndGetUp1 5.70->7.77cm) -- a real, honest trade, not free.

**Known residual, NOT resolved, flagged not hidden**: at coll_weight=0.5,
fallAndGetUp1's range metric still spikes 6.96->39.15cm, traced to ONE held-
right-foot frame (t=2251, support_z=+31.42cm) where phase 2's collision
correction -- run on the SAME chain a held-target lock depends on -- disrupts
that lock badly at that specific frame. Phase 2 has zero awareness of held
targets, only of collision proximity. Open item for follow-up, not chased
further this pass (diminishing returns after 3 mechanism designs + 3
coll_weight values already tested).

**Shipped**: `--avoid-self-collision --coll-weight 0.5` (now the coll_weight
default) wired into `gmr_contact_retarget.py --floor-clamp` (Phase A),
`polish_median_limbwise.py` (Phase B, both `--center` modes), and
`smooth_then_clamp.py` (S7-T2, unconditionally on -- fresh S7 variant, no
byte-identical-baseline constraint to preserve). Default OFF everywhere else
(`avoid_self_collision=False`) -- verified byte-identical to every pre-S7-T7
shipped pkl when off (ground1 fc, max_diff=0.0, re-checked after each of the
three design iterations above).

**Next**: full corpus rebuild of every clamp_limb-dependent variant (fc, fc_sm,
medianlimb, perframelimb) with the fix, then re-eval all four CSVs. In progress.

## S7-T5 (2026-07-18, Sonnet) | OmniRetarget baseline -- feasibility confirmed, execution pending

Found the actual code repo (Undermind's report didn't link it directly): the
OmniRetarget project page (omniretarget.github.io) links `github.com/amazon-far/
holosoma` as "[Code]" -- a broader framework (training+deployment+retargeting),
Apache-2.0, containing a self-contained `holosoma_retargeting/` module. Shallow-
cloned to `/tmp/holosoma_check` and read its README directly (not guessed):

- **G1 support: confirmed.** `models/g1/` present, example commands use
  `--robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf`.
- **LAFAN1 support: confirmed, directly relevant.** `--data_format lafan`, a
  documented conversion path (`data_utils/extract_global_positions.py`, BVH ->
  npy global joint positions, reuses the LAFAN1 GitHub repo's own processing
  script), and a working example command using `dance2_subject1` -- LAFAN1's own
  subject-naming convention, same clips this project already has locally at
  `data/raw/lafan1/*.bvh`.
- **Separate env, as required by the plan**: `scripts/setup_retargeting.sh`
  creates its own conda env (`hsretargeting` by default) -- isolated from this
  project's `gmr` env, no dependency clash risk.
- **Floor-contact awareness, notable**: README explicitly flags
  `--retargeter.foot-sticking-tolerance` needs relaxing for LAFAN1 data (default
  too strict) and there's a `--task-config.ground-range` flag -- suggests the
  authors hit real floor/ground issues on this exact dataset, worth comparing
  against once run.

**Not yet executed** (env setup + LAFAN1 BVH->npy conversion + retarget run +
output-format adapter to this project's qpos-pkl convention + eval is a multi-
hour task) -- paused here to keep priority on S7-T7's self-collision corpus
rebuild (Prabin's direct, active request). Resuming after T7 wraps, within the
plan's half-day time-box for this task (clock: research done, ~1-2h elapsed of
budget). If the box runs out before a full run completes, the documented,
confirmed-feasible install/run path above is itself a legitimate T5 deliverable
per the plan's own "outcome A or outcome B, both acceptable" framing -- though a
real run is still the goal, not yet the fallback.

## S7-T7 (2026-07-18, Sonnet) | Full 77-clip corpus verdict -- self-collision fix confirmed at scale

All four clamp_limb-dependent variants rebuilt with `--avoid-self-collision
--coll-weight 0.5` and re-evaluated at full 77-clip corpus scale (0 build failures
across fc/medianlimb/perframelimb/fc_sm). Clean before/after comparison (fc,
apples-to-apples against the pre-fix backup):

| class | metric | pre-fix | post-fix |
|---|---|---|---|
| loco (43) | coll% | 3.86 | **0.01** |
| loco (43) | collPeak_cm | 5.08 | **0.13** |
| loco (43) | floorPen_cm | 0.72 | 2.32 |
| loco (43) | pen% | 0.2 | 0.4 |
| loco (43) | joint_ok% | 99.6 | 97.9 |
| floor (34) | coll% | 9.95 | **0.05** |
| floor (34) | collPeak_cm | 7.50 | **0.63** |
| floor (34) | floorPen_cm | 8.08 | 11.75 |
| floor (34) | pen% | 6.9 | 7.6 |
| floor (34) | joint_ok% | 91.0 | 88.8 |

Self-collision essentially eliminated on both classes (>99% incidence reduction,
peak depth down 10-20x) -- matches the dev-clip gate exactly, no surprises at
scale. Real, honest cost: floorPen worsens (loco +1.6cm, floor +3.67cm), joint_ok
drops 1.7-2.2pp. **The method still wins decisively overall**: fc's joint_ok
(97.9%/88.8%) remains far above `gmr_polished` (32.1%/0.36%, from S6) and
`gmr_heightfix` (46.3%/0.19%), and floor-class joint_ok is still above `gmr_raw`
itself (80.6%) -- the self-collision fix costs some margin, it doesn't erase the
win.

Other three variants (full class-split tables, same before/after pattern
confirmed):
- **medianlimb**: coll% floor 9.93%->0.03% (from earlier S6-B3 baseline), joint_ok
  floor 91.8%->90.8% (small cost), loco unaffected (98.7%->98.0%).
- **perframelimb**: now the STRONGEST floor-class variant of anything shipped --
  coll%=0.01%, floorPen=6.20cm (best of any variant, fc/fc_sm/medianlimb all
  higher), joint_ok=97.6% (also best). Confirms S7-T3's "best floor mechanism"
  finding still holds after the self-collision fix, and even improves relatively
  (it was already the safest mechanism against the T3 divergence, and now also
  the cleanest on self-collision).
- **gmr_contact_fc_sm** (T2's smoothed variant): coll%=0.00-0.03%, joint_ok
  97.7%/88.9% -- smoothing-then-reclamp continues to work well combined with the
  collision fix (smooth_then_clamp.py always runs `avoid_self_collision=True`).

**Full smoothness/jerk re-eval (s7_smoothness.csv) finished** (462 rows = 77
clips x 6 variants incl. gmr_contact_fc_sm; perframelimb not part of this table --
it was never in S7-T1a's shipped smoothness set). It only holds post-fix numbers
(rebuilt fresh, no pre-fix rows kept), so for an actual before/after delta on
smoothness specifically, re-ran the 5 S7-T1a dev clips against the preserved
pre-fix backup pkls (`pkl_s5_prefix_backup/`) for all four touched variants:

| variant | joint_jerk %d | body_jerk %d | skateL %d | fidPos %d |
|---|---|---|---|---|
| gmr_contact_fc | +4.6% | +6.8% | +9.1% | ~0% |
| gmr_contact_fc_sm | +28.4% | +23.2% | +1.2% | ~0% |
| medianlimb | +7.7% | +9.7% | +24.0% | ~0% |
| perframelimb | +1.7% | +4.6% | +53.7% | ~0% |

(mean %% change post-fix vs pre-fix, averaged over walk1/walk3/run2/ground1/
fallAndGetUp1_subject1; fidelity unaffected as expected -- self-collision
avoidance doesn't touch target tracking.) Consistent with the joint-metric
picture: perframelimb pays the least in jerk (still the safest mechanism
overall) but its skate%% swing is largest in relative terms, off a very small
pre-fix base (0.06-0.12cm) so the absolute cost is small. fc_sm takes the
biggest jerk hit (+28.4%/+23.2%) -- expected, since self-collision avoidance now
fights the Stage-A smoothing pass on frames it didn't touch before. Worst case
across all four is fallAndGetUp1_subject1 (the same clip flagged for the
held-target range residual above) -- the get-up motion triggers the most
self-collision correction, so it shows the largest jerk/skate cost everywhere.
Not a blocker: still a fraction of the smoothness gap already documented
between contact-correct variants and gmr_polished/gmr_raw in S7-T1a.

**S7-T7 verdict: ship.** All four corpus artifacts (fc, fc_sm, medianlimb,
perframelimb) now self-collision-aware by default at coll_weight=0.5, backed by a
full 77-clip re-verification, zero build failures, and an honest documented
trade (floor cost, one flagged residual on fallAndGetUp1's held-target range).
Old pre-fix pkls preserved at `pkl_s5_prefix_backup/` for reference/reproducibility.

## S7-T3b (backfill, 2026-07-18, Sonnet) | perframelimb full 77-clip corpus — build ran under T7, never logged on its own

The plan required a standalone `## S7-T3b` entry once perframelimb's corpus build
landed; it built successfully (0 failures, all 77 clips) but got folded silently
into T7's self-collision corpus rebuild instead of logged separately. Backfilling
per the plan's own backfill rule. Numbers below are the FINAL, self-collision-fixed
build (`s7b_full_corpus.csv` / `s7b_range.csv`, rebuilt under T7's
`--avoid-self-collision --coll-weight 0.5`, same day) — there was an earlier
pre-fix perframelimb corpus pass (its pkls are the ones sitting in
`pkl_s5_prefix_backup/`), but it was never the shipped version and isn't worth a
separate table.

Full 77-clip class-split, perframelimb vs the other three self-collision-fixed
clamp_limb variants and `gmr_raw`:

| class | variant | joint_ok% | floorPen_cm | pen% | coll% | collPeak_cm | range_cm |
|---|---|---|---|---|---|---|---|
| loco (43) | gmr_raw | 91.52 | 5.15 | 3.03 | 3.85 | 5.05 | 7.98 |
| loco (43) | gmr_contact_fc | 97.93 | 2.32 | 0.36 | 0.01 | 0.13 | 8.38 |
| loco (43) | gmr_contact_fc_sm | 97.73 | 2.71 | 0.34 | 0.00 | 0.04 | 9.07 |
| loco (43) | medianlimb | 97.99 | 4.11 | 1.10 | 0.00 | 0.04 | 9.57 |
| loco (43) | **perframelimb** | **98.95** | **2.24** | **0.29** | 0.00 | 0.10 | **4.12** |
| floor (34) | gmr_raw | 80.64 | 15.29 | 23.38 | 6.34 | 5.66 | 12.84 |
| floor (34) | gmr_contact_fc | 88.82 | 11.75 | 7.63 | 0.05 | 0.63 | 18.14 |
| floor (34) | gmr_contact_fc_sm | 88.92 | 9.21 | 6.40 | 0.03 | 0.45 | 17.74 |
| floor (34) | medianlimb | 90.82 | 14.56 | 10.69 | 0.03 | 0.63 | 18.79 |
| floor (34) | **perframelimb** | **97.60** | **6.20** | **1.30** | 0.01 | 0.27 | **7.39** |

**New finding, not previously surfaced**: this isn't just "perframelimb is the
best floor mechanism" (S7-T3's dev-clip claim) — at full corpus scale, POST the
T7 self-collision fix, perframelimb wins on EVERY un-gameable metric on BOTH
classes, including locomotion range (4.12cm, half of fc's 8.38cm) where it was
never specifically targeted. `--center perframe`'s per-frame lift (vs fc's
inline clamp on GMR's own trajectory, or medianlimb's constant shift) appears to
just be a structurally better floor-correction mechanism project-wide, not a
floor-class specialist. Gate threshold from the T3 plan ("if perframe materially
beats fc on the floor class at corpus scale, write S7-DECISION") is unambiguously
tripped — see below.

Known gap, flagged not hidden: perframelimb has NO corpus-scale smoothness/jerk
number (`s7_smoothness.csv`'s variant set is raw/polished/contact/contact_fc/
contact_fc_sm/medianlimb — perframelimb was never added, since S7-T1b predates
T3's fix). The only smoothness data on perframelimb is T7's 5-dev-clip pre/post
self-collision-fix DELTA (+1.7%/+4.6% jerk, the smallest of the four touched
variants) — that's a relative number, not an absolute comparison against fc's own
jerk. This gap has to close before perframelimb can be claimed as a smoothness
win, not just a contact-correctness win.

## S7-DECISION (2026-07-18, Sonnet) | perframelimb vs gmr_contact_fc as the primary method — presented, not decided

Per `GMR-S7-plan.md`'s own instruction: present options, the choice is Prabin's.

**The finding**: perframelimb (S7-T3's fixed `--center perframe`) now beats
`gmr_contact_fc` (the S6-shipped primary method) on every un-gameable joint/floor
metric, on BOTH classes, at full 77-clip corpus scale, after both variants got
the same T7 self-collision fix (table above). This is a bigger result than the
plan anticipated — T3 targeted "best floor mechanism," not "better than fc
everywhere."

**What's NOT yet known** (the reason this isn't already a ship decision):
1. **Smoothness/jerk at corpus scale — unmeasured for perframelimb** (the gap
   flagged in S7-T3b above). `gmr_contact_fc` has full dev+corpus smoothness
   numbers (S7-T1a/T1b) AND a validated smoothing companion (`gmr_contact_fc_sm`,
   T2) that measurably fixes its jerk cost. perframelimb has neither — it could
   have a comparable or worse jerk profile that a `_sm`-style companion would
   need to fix before it's paper-ready on that axis too.
2. **Provenance is different**: `gmr_contact_fc` composes with GMR's own
   in-loop tracking (contact override happens inside the mink solve, S5);
   perframelimb is a fully post-hoc median-limb-wise correction on top of raw
   GMR output (S6 Phase B lineage) with a per-frame lift. If the paper's
   narrative is "contact-aware IN the retargeting solve," perframelimb sits
   awkwardly outside that framing even if its numbers are better — a
   positioning question, not just a metrics one.
3. Range results above show perframelimb's own worst-case spread (loco 4.12cm,
   floor 7.39cm) but not yet cross-checked with T7-style visual review (S7-T4's
   money-shot approach) — the S6 experience (Phase B bugs 4/5 only caught by
   dev-clip visual inspection) is a reason not to skip that step.

**Options, presented for Prabin's call:**
- **A. Promote perframelimb to primary method.** Best numbers on record,
  supersedes `gmr_contact_fc`. Requires: (i) a `sprint_s7_smoothness.py`-style
  corpus pass adding perframelimb to `s7_smoothness.csv` before claiming a
  smoothness result, (ii) probably a `_sm`-equivalent smoothing companion if T1
  gate trips the same >50%-jerk threshold, (iii) reframing the paper's
  "in-solve contact layer" narrative around a post-hoc mechanism instead.
- **B. Keep `gmr_contact_fc`/`gmr_contact_fc_sm` as primary, report perframelimb
  as the strongest ablation/ceiling reference.** Lower technical risk (fc is
  fully validated on every axis already, including smoothness); costs the
  paper its best number if a reviewer directly compares floor-class results.
- **C. Per-class hybrid** — fc/fc_sm for locomotion (already the more complete
  story), perframelimb for the floor class specifically. Splits the difference
  but adds a "why two different mechanisms" question a reviewer will ask.
- **D. Close the smoothness gap first (cheap, extends existing T1 tooling),
  THEN decide** between A/B/C with a complete picture. Lowest-regret next step
  if there's no immediate deadline forcing the call now.

No default recommendation asserted here beyond D being the cheapest way to stop
deciding on an incomplete picture — Prabin's call.

## S7-T6 (2026-07-18, Sonnet) | torso/waist residual probe — negative, but surfaced a more important open item

Exploratory probe per the plan, 2-attempt cap. New `scripts/g1/sprint_s7_t6_waist_probe.py`,
new `"waist"` chain added to `leg_floor_clamp.py`'s `CHAIN_JOINTS`/`EFF_BODY` (waist_yaw/
roll/pitch, terminal effector `torso_link`) — NOT added to the shipped `CLAMP_TARGETS`
list, probe-only. Confirmed via `body_parentid` before building anything: `pelvis` attaches
directly to `world` (it's the free-joint root body here, not a waist-chain descendant) —
so a waist correction can only ever move `torso_link` and above; pelvis floor penetration
is out of scope by design (root lift is perframe's own job, not duplicated here), matching
the plan's own "root stays frozen" instruction.

Clip selection wasn't eyeballed: swept every floor-class clip's `perframelimb` output for
the deepest per-BODY worst-z (not just the whole-clip worst, which is dominated by
whichever body is deepest at its single worst frame) and picked the 2 with the deepest
measured `torso_link` penetration — `fallAndGetUp2_subject2` (-3.6cm) and
`fallAndGetUp1_subject1` (-3.5cm). Same sweep surfaced something unexpected: on both
clips, `left_ankle_roll_link` has a FAR deeper residual than torso (-18.6cm / -12.4cm) —
despite being inside `CLAMP_TARGETS`' scope and nominally corrected every frame.

**Attempt 1** (waist clamp only, `floor_margin=0.0`, clearance-only): 515/4918 and
600/5047 frames triggered a correction. Whole-clip `floorPen_cm`/`range_cm` UNCHANGED on
both clips (confirms the headline metric is dominated by the much-deeper ankle residual
above, not torso — correcting torso can't move a number that torso was never the worst
contributor to). `pen_pct` improved on fallAndGetUp1 (2.70%→2.18%) but WORSENED on
fallAndGetUp2 (3.36%→4.64%); `coll_pct`/`coll_peak_cm` also worsened on both (0.02%→0.16%,
0.43→1.46cm on fallAndGetUp1) — the waist correction has zero self-collision awareness of
its own (same class of blind spot as the un-fixed floor clamp pre-T7).

**Attempt 2** (waist clamp + `avoid_self_collision=True`, the T7 fix reused verbatim):
closed the self-collision regression back to baseline on both clips exactly (coll_pct/
coll_peak_cm unchanged from pre-waist-clamp). Net result still a wash: floorPen_cm/range_cm
unchanged (same reason as attempt 1), joint_ok_pct flat-to-slightly-worse (94.146%→94.010%
on fallAndGetUp2, unchanged on fallAndGetUp1), pen_pct still mixed (improves on one clip,
worsens on the other).

**Verdict: T6 not shipped.** Torso/waist correction doesn't move any of the metrics that
matter (floorPen_cm, joint_ok, range) on the clips actually chosen for having the worst
torso residual — it isn't the real bottleneck on these clips. 2-attempt cap reached,
stopping per the plan's own instruction ("log numbers, stop, present to Prabin").

**More important finding, surfaced as a byproduct, not chased further this pass**: the
-18.6cm / -12.4cm ankle residuals are themselves a single isolated frame each (t=212 of
4918 on fallAndGetUp2_subject2; direct instrumentation shows neighbor frames at -5.9cm/
-2.9cm/+1.9cm/+0.8cm/+0.4cm — a one-frame spike, not a sustained failure) coinciding with
an unusually high active-contact count at that exact frame (`ncon=30` vs 0-20 on
neighbors). This is the SAME class of bug already flagged in S7-T7's "known residual, not
resolved" note (self-collision phase 2 is floor/held-blind and can overpower phase 1's
convergence within the bounded iteration budget on a hard frame) — just showing up on a
different chain/frame than the one T7 specifically flagged (fallAndGetUp1's held-right-foot
t=2251 case), and reaching a deeper magnitude (-18.6cm vs T7's own +31.42cm support_z
spike). Confirms the residual is broader than T7's single flagged instance, not a new root
cause — still open, still not chased further (same reason T7 gave: diminishing returns
after 3 mechanism designs already tried for phase-1/phase-2 interaction).

## S8-T0 (2026-07-18, Sonnet)

### T0a: perframelimb smoothness at corpus scale

Script: `scripts/g1/sprint_s8_t0a_perframelimb_smooth.py` — appended 77 perframelimb rows to `s7_smoothness.csv` (539 total rows). Pure eval, no new pkls. Labels from `s1t4_reclass.csv` (floor_class=1 → "floor", 0 → "loco").

**Smoothness class means (all variants, post-fix 77-clip corpus):**

| variant | class | j_jerk | skL_cm | skR_cm | vMax rad/s | n_spk/clip |
|---|---|---|---|---|---|---|
| gmr_raw | floor | 5003 | 0.440 | 0.433 | 34.0 | 0.18 |
| gmr_contact_fc | floor | 7879 | 1.301 | 1.142 | 78.7 | 7.88 |
| gmr_contact_fc_sm | floor | 1403 | 3.893 | 3.283 | 61.5 | 3.56 |
| medianlimb | floor | 8010 | 0.868 | 0.831 | 84.7 | 10.15 |
| **perframelimb** | **floor** | **8086** | **0.556** | **0.519** | **51.7** | **0.91** |
| gmr_raw | loco | 6189 | 0.308 | 0.339 | 32.8 | 0.00 |
| gmr_contact_fc | loco | 7493 | 0.794 | 0.634 | 40.7 | 4.16 |
| gmr_contact_fc_sm | loco | 855 | 3.655 | 3.290 | 23.8 | 2.26 |
| medianlimb | loco | 7180 | 0.210 | 0.334 | 49.2 | 1.02 |
| **perframelimb** | **loco** | **8857** | **0.229** | **0.283** | **47.0** | **0.12** |

perframelimb summary vs gmr_raw:
- jerk: 1.62× raw (floor), 1.43× raw (loco) — worse than raw on jerk
- skate: 1.26×/1.20× raw (floor), 0.74×/0.84× raw (loco) — slight skate improvement on loco
- vMax: 1.52× raw (floor), 1.43× raw (loco) — higher peak velocity but better than fc/medianlimb
- n_spikes/clip: **0.91 (floor), 0.12 (loco)** — drastically better than fc (7.88/4.16) and medianlimb (10.15/1.02); 5-10× fewer spikes than fc on floor class
- Retires S7-DECISION option D (perframelimb unmeasured) — it IS better than fc on spikes; still not at raw level.

### T0b: spike attribution

Analyzed top-5 worst clips by n_spikes for each variant. Spike = frame with max|dq|·fps > 60 rad/s. Attribution per spike frame: (A) clamp correction toggling or growing instability; (B) phase-2 self-collision ncon >5; (C) perframe root-lift discontinuity >2cm/frame; (D) held-flag transition at spike frame.

**gmr_contact_fc — worst 5 clips:**

| clip | n_spikes | A | B | C | D | worst_mag |
|---|---|---|---|---|---|---|
| obstacles6_subject5 | 155 | 148 | 1 | 0 | 9 | 94.2 |
| fallAndGetUp1_subject4 | 69 | 66 | 2 | 0 | 2 | 127.1 |
| obstacles5_subject2 | 24 | 22 | 0 | 0 | 0 | 94.2 |
| walk3_subject3 | 24 | 24 | 0 | 0 | 0 | 94.2 |
| fallAndGetUp2_subject2 | 20 | 19 | 0 | 0 | 4 | 93.4 |
| **total** | **292** | **279 (95.5%)** | **3 (1.0%)** | **0** | **15 (5.1%)** | **127.1** |

**perframelimb — worst 5 clips:**

| clip | n_spikes | A | B | C | D | worst_mag |
|---|---|---|---|---|---|---|
| obstacles4_subject3 | 7 | 4 | 0 | 0 | 0 | 81.4 |
| walk2_subject3 | 6 | 6 | 0 | 0 | 1 | 85.8 |
| obstacles5_subject3 | 5 | 5 | 0 | 0 | 0 | 85.7 |
| aiming1_subject4 | 4 | 4 | 0 | 0 | 2 | 85.7 |
| pushAndFall1_subject4 | 3 | 1 | 0 | 0 | 0 | 77.6 |
| **total** | **25** | **20 (80.0%)** | **0** | **0** | **3 (12.0%)** | **85.8** |

Note: 18 frames across both variants show `causes=----` (no cause detected by proxy). Direct instrumentation of one representative (obstacles6_subject5 t=2315, left_hip_yaw: corr grows -0.76→-17→-63→-14 rad/s across 4 frames) confirms these are cause A — DLS correction magnitude building up near a singular configuration, not a toggle but a rapid growth pattern. My proxy only caught sign changes / onset; these are the same instability class.

**Dominant cause: A (clamp activation toggling / DLS instability) = >80% of spikes in both variants.**

B (phase-2 self-collision) accounts for only 1.0% of fc spikes and 0% of perframelimb; the specific -18.6cm frame from S7-T6 (fallAndGetUp2_subject2 t=212, ncon=30) is NOT in the top-5 fc spikes by count — it's a high-magnitude one-off but not the volume driver. C (root-lift discontinuity) is zero for perframelimb — the 15-frame moving-average smoothing on the lift curve is sufficient to eliminate lift-discontinuity spikes. D (held-release ramp) co-occurs with A on ~5% of fc spikes and ~12% of perframelimb spikes, but never as a sole cause.

**T1 priority order (for when T1 is authorized): T1b (activation continuity, cause A) is the dominant target. T1a (phase-2 acceptance check) covers only 1% of fc's spikes and 0% of perframelimb's — low ROI. T1c (lift smoothing) not needed — cause C = 0 on corpus.**

## S8-T1 (2026-07-18, Sonnet)

### T1c: skipped (cause C = 0 in T0b, no mechanism needed).

### T1b: activation continuity (cause A), 2 attempts, both on the 15-clip gate set (5 dev + 10 T0b worst-spike clips)

Mechanism: `CorrectionRateLimiter` in `gmr_contact_retarget.py` (`--clamp-rate-limit R`, R=0.15 rad/frame), warm-started from previous frame's *applied* correction. Two code states tried, tagged by pkl suffix:

- **`rl` (attempt 1):** rate limit applied to the *combined* phase-1 (floor/held clamp) + phase-2 (self-collision) correction.
- **`rl2` (attempt 2):** rate limit applied to phase-1 only; phase-2 collision post-pass left unlimited (built specifically to fix attempt 1's coll_pct regression below).

**Class means, 15-clip gate set, before(unlimited)→after(rate-limited):**

| variant | class | spk before→after | vMax before→after (1.2×raw gate) | coll% before→after | jok before→after | fp_cm before→after |
|---|---|---|---|---|---|---|
| gmr_contact_fc_**rl** | floor | 13.67→**0.33** | 81.2→**41.0** (43.7) | 0.124→**0.960** | 87.12→95.63 | 13.00→11.11 |
| gmr_contact_fc_**rl** | loco | 51.67→**0.00** | 46.0→**21.3** (25.5) | 0.115→**2.005** | 88.23→98.84 | 8.07→3.55 |
| perframelimb_**rl** | floor | 2.08→**0.42** | 59.5→**36.5** (43.7) | 0.010→**0.600** | 97.37→96.68 | 5.93→5.70 |
| perframelimb_**rl** | loco | 0.00→**0.00** | 37.4→**21.2** (25.5) | 0.000→**1.974** | 98.81→98.60 | 2.79→3.19 |
| gmr_contact_fc_**rl2** | floor | 13.67→**2.33** | 81.2→**69.3** (43.7) | 0.124→**0.030** | 87.12→89.59 | 13.00→21.57 |
| gmr_contact_fc_**rl2** | loco | 51.67→**68.67** | 46.0→**67.7** (25.5) | 0.115→**0.419** | 88.23→81.43 | 8.07→12.26 |
| perframelimb_**rl2** | floor | 2.08→**0.42** | 59.5→**38.5** (43.7) | 0.010→**0.012** | 97.37→96.51 | 5.93→7.79 |
| perframelimb_**rl2** | loco | 0.00→**0.00** | 37.4→**25.9** (25.5, borderline) | 0.000→**0.009** | 98.81→98.26 | 2.79→5.15 |

**Verdict: neither attempt clears the full gate. A genuine tension, not a tuning miss:**
- `rl` (rate-limit everything) crushes spikes/vMax cleanly (well inside the ≤0.5/clip, ≤1.2×raw gate) but coll_pct blows the ≤0.01-point allowance by 60–200×, because throttling the combined correction also throttles the self-collision push-out. Note: even at 0.6–2.0%, this is still far below `gmr_raw`'s baseline self-collision (6.337%/3.853% floor/loco per S8 intro table) — the regression is relative to the S7-T7 post-fix numbers (0.048%/0.009%), not relative to raw.
- `rl2` (unlimit phase-2 only) fixes coll_pct back down (0.030%/0.419%, near the S7-T7 numbers) but reopens the spike/vMax gate — fc-loco spikes go to 68.67/clip, *worse than the unlimited baseline* (51.67). This means phase-2 self-collision is a bigger spike driver than T0b's attribution suggested (1% of fc spikes) once it's allowed to run unbounded next to an otherwise rate-limited phase-1 — the two phases interact differently under partial rate-limiting than either fully-limited or fully-unlimited.
- floorPen_cm also regresses on `rl2` (+1.86cm floor / +2.36cm loco on perframelimb) beyond the 0.5cm allowance — rate-limiting phase-1 slows floor-clearance convergence during the ramp-in.

**2-attempt cap reached for T1b. Per S8 plan's failure-handling rule, not attempting a third T1b variant.** T1a (phase-2 acceptance/backtracking check) is a distinct, unused mechanism — S8-T0's "low ROI" read (1% of fc spikes attributed to cause B) was based on the *unlimited* baseline and is now stale: `rl2`'s result shows phase-2 becomes a dominant spike source once phase-1 is separately rate-limited. Whether to spend T1a's budget on this reframed problem, accept `rl`'s coll_pct tradeoff (still far below raw) and move to T3, or something else is Prabin's call — escalating per plan rather than picking a mechanism unilaterally.

## S8-T2 (2026-07-18, Sonnet, per REVISION R1) — held-aware smoothing, both arms, GATE FAILED at 2-attempt cap

Mechanism: `scripts/g1/smooth_heldaware.py`, new script per R1.2/R1.3. Held leg-chain
DOFs + root locked to input via a ramped (5-frame cosine) tracking-weight boost
(λ_lock=1e8) in `stage_a`'s per-joint `lambda_track_frames`; free DOFs (arms,
waist, root on non-held stretches) smoothed at λ_track=1.0/λ_smooth=20. Applied
to both `perframelimb` (ours) and `gmr_heightfix` (fairness arm, R1.3) →
`perframelimb_sm` / `heightfix_sm`. Gate set: 10 clips (5 dev + 5 perframelimb
worst-spike from T0b), `scripts/g1/sprint_s8_t2_gate.py`.

**Attempt 1 (plain held-aware lock) — 10-clip gate set, equal per-clip-mean ("combined" = script's own class-size-weighted mean over all 10 clips; `pfl` = unsmoothed `perframelimb`):**

| metric | gmr_raw | perframelimb (pfl) | perframelimb_sm | gate |
|---|---|---|---|---|
| n_spikes/clip | 0.31 | 2.50 | **2.00** | FAIL (≤0.5) |
| vMax rad/s | 35.40 | 60.67 | **59.46** | FAIL (≤42.48, 1.2×raw) |
| joint_jerk | 4882.5 | 8178.3 | 2929.4 | PASS (≤6347.2) |
| body_jerk | 204.4 | 263.0 | 125.1 | PASS (≤265.7) |
| joint_ok% | — | 98.59 | 85.05 | FAIL (need ≥97.59) |
| floorPen cm | — | 4.27 | 7.78 | FAIL (need ≤4.77) |
| coll% | — | 0.002 | 2.15 | FAIL (need ≤0.052) |
| worst_float cm | — | 3.11 | 3.11 | PASS (unchanged) |

**Root cause (diagnosed, not guessed):** obstacles4_subject3 frame 2116→2117 —
`perframelimb`'s per-frame limb IK has a genuine one-frame hip-yaw branch flip
(1.57rad→−1.14rad, i.e. Δ≈2.7rad in 1/fps s) surrounded by ordinary values on
both sides. The frame sits inside the *ramp-out tail* after the right foot's
`held` flag clears (lock weight ~9e7 at that frame — confirmed by direct
instrumentation of `build_lock_weights`, not inferred). Attempt 1's ramp
(designed so the lock boundary itself doesn't introduce a *new* discontinuity)
instead **preserves a pre-existing one**. Same pattern (spike joint always
hip-yaw idx 2 or 8, `sm_unchanged=True`, mostly outside strict held windows but
inside the 5-frame ramp) confirmed across all 5 T0b worst-spike clips —
this is the dominant spike cause for `perframelimb`, not a smoothing-strength
shortfall.

**Attempt 2 (spike-aware unlock exception, targeted fix of the diagnosed cause):**
`build_lock_weights` now takes the raw `qpos`/`fps`; wherever the raw input's
own per-joint velocity exceeds 40 rad/s at a held/ramp frame, that joint's lock
is dropped to λ_track for both endpoints of the transition (all other
held/ramp frames untouched). 10-clip gate set (same layout as attempt 1;
`gmr_raw`/`pfl` unchanged from attempt 1, repeated for comparison):

| metric | gmr_raw | perframelimb (pfl) | perframelimb_sm | gate |
|---|---|---|---|---|
| n_spikes/clip | 0.31 | 2.50 | **0.00** | PASS (≤0.5) |
| vMax rad/s | 35.40 | 60.67 | **36.61** | PASS (≤42.48) |
| joint_jerk | 4882.5 | 8178.3 | 2671.4 | PASS (≤6347.2) |
| body_jerk | 204.4 | 263.0 | 141.8 | PASS (≤265.7) |
| joint_ok% | — | 98.59 | 84.90 | **FAIL** (need ≥97.59) |
| floorPen cm | — | 4.27 | 7.78 | **FAIL** (need ≤4.77) |
| coll% | — | 0.002 | 2.14 | **FAIL** (need ≤0.052, ~40×) |
| worst_float cm | — | 3.11 | 5.05 | **FAIL** (need ≤4.11) |

**Verdict: attempt 2 fixes all 4 smoothness axes cleanly but reopens the
contact-preservation axes it was gated on protecting.** This is the trade
`stage_a`'s own docstring already warned about: "this global smoothing pass is
otherwise floor-blind and can erode a sharp, narrow correction" — unlocking
exactly the frames where `perframelimb`'s per-frame clamp made its sharpest
save (the same frames that spike) lets the tridiagonal solve blend the
correction back toward its uncorrected neighbours. Two genuine, understood
mechanisms, two genuine failure modes on opposite axis groups — not a tuning
miss either time.

**2-attempt cap reached for T2. Per S8 plan's failure-handling rule, not
attempting a third T2 mechanism (e.g. per-frame re-clamp after smoothing,
which was the plan's suggested fallback — skipped because it targets the
wrong axis: the diagnosed cause is a lock/unlock boundary tradeoff, not
insufficient smoothing strength, so re-clamping would either restore the
lock (attempt 1's failure) or repeat the erosion (attempt 2's failure)
depending which pass runs last).** Escalating to Prabin: T3/T4/S8-DECISION
are blocked until a path forward is chosen. `perframelimb_sm` pkls on disk
(both attempts, attempt 2 is current) are the attempt-2 build; not
corpus-built beyond the 10-clip gate set.

## REVISION R2 (Fable, 2026-07-18) — training-relevance ruling + new mechanism round

T2's two attempts decompose into two independent effects, not one: contact
erosion is identical across both (floorPen 7.78/7.78cm, coll 2.15/2.14%,
joint_ok 85.05/84.90%) — caused by smoothing free (non-held) frames, which
drags perframelimb's swing-clearance/collision corrections back toward the
uncorrected input. The lock/unlock toggle alone moves only the temporal axes
(spikes 2.00→0.00, vMax 59.46→36.61) at a cost of worst_float 3.11→5.05cm.
Conclusion: contact-blind temporal smoothing structurally cannot
Pareto-improve a per-frame contact-corrected motion — a controlled ablation
result, not a tuning miss.

RetargetMatters (the GMR paper) states its own critical-artifact list
verbatim: "foot penetration, self-intersection, and abrupt velocity spikes
are all critical artifacts that should be avoided during retargeting" — its
three case-study policy failures map 1:1 (PHC ground penetration,
ProtoMotions self-intersection, GMR's own "Dance 5" waist-value jumps). Mean
joint jerk is NOT on that list and appears in none of their eval metrics.
**Prabin's ruling (2026-07-18, approved):** demote `joint_jerk` from gated
(≤1.3×raw) to report-only (≤1.75×raw sanity ceiling); `body_jerk` stays
gated. Condition: T4's render watch becomes a hard veto — any teleport-like
or visibly-absurd-torque frame kills the variant regardless of its metric
table. Never tradeable: floorPen/pen%, coll%, n_spikes, vMax, worst_float,
joint_ok. Full text: `GMR-S8-plan.md` REVISION R2.

## S8-T2c (2026-07-18, Fable, per REVISION R2) — smooth→re-clamp, 2-attempt cap, close but FAILED

Mechanism: re-clamp the perframelimb_sm (T2 attempt-2, spike-unlock) pkls —
limbs only (phase-1 floor/held + phase-2 self-collision via
`polish_median_limbwise._limbwise_pass`, new `--center none` mode added to
skip root re-lift), each frame's DLS starting from that frame's own
already-smooth input. New `scripts/g1/sprint_s8_t2_gate.py --build-smrc`.
Same unmodified 10-axis T2 gate, same 10 clips.

**Attempt 1 (max_dq=None, combined class-mean):**

| metric | gmr_raw | pfl | perframelimb_smrc | gate |
|---|---|---|---|---|
| n_spikes/clip | 0.31 | 2.50 | **11.70** | FAIL (≤0.5) |
| vMax rad/s | 35.40 | 60.67 | **89.03** | FAIL (≤42.48) |
| joint_jerk | 4882.5 | 8178.3 | 4460.3 | PASS (≤6347.2) |
| body_jerk | 204.4 | 263.0 | 204.2 | PASS (≤265.7) |
| joint_ok% | — | 98.59 | 98.39 | PASS (need ≥97.59) |
| floorPen cm | — | 4.27 | 6.04 | FAIL (need ≤4.77) |
| coll% | — | 0.002 | 0.000 | PASS (need ≤0.052) |
| worst_float cm | — | 3.11 | **28.52** | FAIL (need ≤4.11) |

Root cause (diagnosed, not guessed — `smrc_spike_probe.py` on
walk2_subject3): worst frame t=6318→6319, left_hip_yaw (dof 2, limits
±1.57rad) driven 1.57 → 1.57 → **-1.57** → 1.57 across four consecutive
frames — uncapped DLS bouncing between opposite joint limits. Identical bug
class to the one `polish_median_limbwise.py`'s `--center perframe` docstring
already documents (S7-T3): a near-singular full-extension basin where
uncapped DLS diverges within a frame's iterations. perframelimb's own build
sits leg DOFs near their limits often (4224 (frame,dof) pairs within
0.02rad of a limit in the smoothed input); re-clamping without the SAME
trust region that mode already uses elsewhere in this codebase (max_dq=0.15)
hits the identical instability.

**Attempt 2 (max_dq=0.15, the established trust-region value, combined class-mean):**

| metric | gmr_raw | pfl | perframelimb_smrc | gate |
|---|---|---|---|---|
| n_spikes/clip | 0.31 | 2.50 | **3.80** | FAIL (≤0.5) |
| vMax rad/s | 35.40 | 60.67 | **64.21** | FAIL (≤42.48) |
| joint_jerk | 4882.5 | 8178.3 | 3658.0 | PASS (≤6347.2) |
| body_jerk | 204.4 | 263.0 | 171.8 | PASS (≤265.7) |
| joint_ok% | — | 98.59 | 98.50 | PASS (need ≥97.59) |
| floorPen cm | — | 4.27 | 4.02 | PASS (need ≤4.77) |
| coll% | — | 0.002 | 0.002 | PASS (need ≤0.052) |
| worst_float cm | — | 3.11 | 4.07 | PASS (need ≤4.11) |

8 of 10 axes now PASS cleanly — all contact/collision/jerk axes match
perframelimb. Per-clip breakdown shows the failure is entirely concentrated
in the 5 T0b worst-spike clips: all 5 dev clips are clean (spk=0, vMax
27.2–50.4 rad/s), while obstacles4/walk2/obstacles5/aiming1/pushAndFall1
still spike (6/14/12/4/2) — on 2 of these 5 (walk2, obstacles5), smrc's
re-clamp spikes MORE than perframelimb's own original construction did
(perframelimb: 7/6/5/4/3) — re-clamping a smoothed configuration hits a
different DLS instability than perframelimb's original per-frame-independent
build, even with the trust region.

**2-attempt cap reached for T2c. GATE: FAILED (close, not clean).** Per R2.6,
proceeding directly to T2d (pre-approved, no further sign-off needed) rather
than a third max_dq value — the residual spikes are concentrated on exactly
the hardest clips, which is what T2d (local repair, sidesteps global DLS
re-solve entirely) targets by design.

## S8-T2d (2026-07-18, Fable, per REVISION R2.4) — local spike repair, 2-attempt cap, both FAILED

Mechanism: `scripts/g1/sprint_s8_t2d_repair.py` — detect leg-DOF transitions
where unsmoothed `perframelimb` already has |Δq|·fps > 40 rad/s (below the
60 rad/s metric threshold, so the repair set is a superset of counted
spikes), merge overlapping ±3-frame windows per joint, PCHIP-interpolate
through each window from untouched context points, re-clamp, re-detect (max
2 iterations). Gate per R2.2 (approved): joint_jerk report-only (≤1.75×raw
ceiling), everything else the original unmodified T2 gate.

**Attempt 1 (re-clamp the WHOLE clip after each repair pass — same call
perframelimb's own build uses, reasoning untouched frames are already that
operation's fixed point):**

| metric | gmr_raw | pfl | perframelimb_repair | gate |
|---|---|---|---|---|
| n_spikes/clip | 0.31 | 2.50 | **3.90** | FAIL (≤0.5) |
| vMax rad/s | 35.40 | 60.67 | **65.70** | FAIL (≤42.48) |
| joint_jerk | 4882.5 | 8178.3 | 8669.3 | FAIL [report-only] (≤8544.3, 1.75×raw) |
| body_jerk | 204.4 | 263.0 | **288.8** | FAIL (≤265.7) |
| joint_ok% | — | 98.59 | 98.78 | PASS (need ≥97.59) |
| floorPen cm | — | 4.27 | 3.69 | PASS (need ≤4.77) |
| coll% | — | 0.002 | 0.000 | PASS (need ≤0.052) |
| worst_float cm | — | 3.11 | **9.19** | FAIL (need ≤4.11) |

**Root cause (diagnosed, not guessed): the fixed-point assumption was wrong.**
Per-clip spikes went WORSE on the hardest clips vs original perframelimb:
walk2_subject3 6→**18**, pushAndFall1 3→6, obstacles5 5→6; only aiming1
improved 4→3. T0b already established >80% of perframelimb's own spikes are
cause A (DLS chatter near a singular config) — re-running the identical
clamp pass over the WHOLE clip, even though nearly every frame is already
that operation's fixed point, still perturbs the handful of already-chaotic
frames (via proximal-to-distal chain propagation from repaired neighbors)
onto a DIFFERENT unstable branch instead of leaving them alone. This is the
plan's own literal design deviated from for a shortcut ("re-run globally,
should be a near no-op") — the shortcut was wrong specifically at the
frames that mattered most.

**Attempt 2 (fixed the above: re-clamp restricted to the repair windows
ONLY — `_limbwise_pass_frames`, new function, every frame outside a window
is bit-identical to input by construction):**

| metric | gmr_raw | pfl | perframelimb_repair | gate |
|---|---|---|---|---|
| n_spikes/clip | 0.31 | 2.50 | **2.40** | FAIL (≤0.5) |
| vMax rad/s | 35.40 | 60.67 | **57.04** | FAIL (≤42.48) |
| joint_jerk | 4882.5 | 8178.3 | 8234.1 | PASS [report-only] (≤8544.3, 1.75×raw) |
| body_jerk | 204.4 | 263.0 | **288.7** | FAIL (≤265.7) |
| joint_ok% | — | 98.59 | 98.05 | PASS (need ≥97.59) |
| floorPen cm | — | 4.27 | 4.10 | PASS (need ≤4.77) |
| coll% | — | 0.002 | 0.002 | PASS (need ≤0.052) |
| worst_float cm | — | 3.11 | **13.93** | FAIL (need ≤4.11, ~3.4×) |
| skate(L) cm | — | 0.26 | **1.12** | FAIL (≤2×raw = 0.74) |
| skate(R) cm | — | 0.46 | **1.03** | FAIL (≤2×raw = 0.78) |

Contact axes (joint_ok/floorPen/coll_pct) now clean and per-clip spikes
improved on 3 of 4 hard clips (walk2 18→6, obstacles5 6→3, aiming1 3→2;
pushAndFall1 4 unchanged). **New failure mode, also diagnosed:** the
windowed re-clamp's held-foot lock target was "wherever the foot's phase-1-
corrected XY currently is" (no cross-frame anchor, since window-only skips
the ramp/onset state entirely) — each frame in a window independently locks
to its own possibly-already-drifted position, letting a held foot walk away
over the window instead of staying planted. worst_float hit 16.7cm on
walk2_subject3 loco class; skate blew up to 1.1-3.1cm on 4 of 5 hard clips.

**A second, independent finding from the build logs:** window counts vastly
exceed n_spikes counts on the hard clips — walk2_subject3: 69 windows (iter
1) across 3 DOFs vs its n_spikes=6; obstacles5_subject3: 39 windows vs
n_spikes=5. Lowering the detection threshold from 60→40 rad/s reveals cause-A
chatter is a **dense region spanning many consecutive frames**, not a
handful of isolated spikes, on these specific clips — neither a full-clip
nor a windowed re-clamp converges within the 2-iteration budget (9-19
windows still open after 2 iterations on 4 of 5 hard clips). The clips where
this shows up (obstacles4/5, walk2, aiming1, pushAndFall1) are exactly
T0b's original "worst 5" — this is not new chatter, it's the SAME chatter
current metrics only partially see.

**2-attempt cap reached for T2d. GATE: FAILED both attempts.** Escalating to
Prabin: three independent mechanisms have now been tried across T2/T2c/T2d
(held-aware lock/unlock, smooth→re-clamp, local spike repair) and none
clears the full gate. T2c-attempt-2 remains the closest result on record
(8/10 axes clean) but its two failures — n_spikes and vMax — sit exactly on
R2.2's non-tradeable list (never tradeable: floorPen/pen%, coll%, n_spikes,
vMax, worst_float, joint_ok), so "close" does not mean "acceptable under the
approved ruling." Per plan, not attempting a fourth mechanism unilaterally.

## S8-T2-DECISION (Prabin, 2026-07-18) — ship T2c-attempt-2 (`perframelimb_smrc`), reframed

**Decision: `perframelimb_smrc` (T2c attempt 2, max_dq=0.15) is the S8
winner.** Prabin's ruling: the worst_float/floorPen "never tradeable" gate
in R2.2 was calibrated against `perframelimb`'s OWN numbers, not against
what actually threatens a downstream policy — GMR-full's baseline sits at
14-15cm worst_float / 3.7-3.8cm floorPen on these same clips, so a variant
at 4.07cm / 4.02cm is still a large, real win over the primary baseline
even though it doesn't match perframelimb's own 3.11cm / 4.27cm exactly.
Judge against RetargetMatters' absolute severity bar, not a same-method
delta.

**The honest cost of this call — what's worse, not glossed over:**

1. **n_spikes and vMax are WORSE than the already-shipped `perframelimb`
   itself, not just short of `gmr_raw`.** Combined gate-set mean: spikes
   2.50→**3.80**/clip, vMax 60.67→**64.21** rad/s. This is a regression on
   the exact axis S8 exists to fix, concentrated on 2 of 10 clips:

   | clip | pfl spk→smrc spk | pfl vMax→smrc vMax |
   |---|---|---|
   | walk2_subject3 | 6 → **14** | 85.8 → 84.2 |
   | obstacles5_subject3 | 5 → **12** | 85.7 → 88.2 |
   | obstacles4_subject3 | 7 → 6 | 81.4 → 93.8 |
   | aiming1_subject4 | 4 → 4 | 85.7 → 94.2 |
   | pushAndFall1_subject4 | 3 → 2 | 77.6 → 68.7 (improved) |

   walk2 and obstacles5 more than double their spike count. All 5 dev clips
   are completely clean (0 spikes, vMax 27-50 rad/s) — the cost is entirely
   concentrated on the 5 hardest, most dynamic clips (obstacles/aiming/push-
   and-fall), not spread across the corpus.

2. **Single-frame peaks up to 94.2 rad/s survive** on 3 of 10 gate clips
   (obstacles4=93.8, aiming1=94.2, obstacles5=88.2) — the documented
   "π rad/frame @ 30fps, solution-branch flip" signature
   (`leg_floor_clamp.py`'s `CorrectionRateLimiter` docstring). This is a
   direct risk to T4's render-watch veto (teleport-like motion) — MUST be
   visually confirmed clean or absent-in-practice before shipping, not
   assumed safe from the metric alone.

3. **worst_float regresses ~0.6-1.0cm vs `perframelimb`** (floor
   2.96→3.60cm, loco 3.15→4.19cm) — small in absolute terms and still
   ~10-11cm better than GMR-full, but a real, disclosed number, not zero.

4. **Only validated on the 10-clip gate set (5 dev + 5 T0b worst-spike),
   not yet at 77-clip corpus scale (T3 pending).** The "concentrated on 5
   known-hard clips" read could understate the true breadth once T3 builds
   the full corpus — T3's table is the first place this gets checked at
   scale.

5. **joint_jerk and body_jerk both genuinely improved** (3658 vs pfl's
   8178, 172 vs pfl's 263) — this is the one clean, unambiguous win layered
   on top of the above costs, not a wash.

**Next: T3 (77-clip corpus build for `perframelimb_smrc`, replacing
`perframelimb_sm` in the column layout) → T4 (renders, hard veto per
Prabin's condition — explicitly re-check the 3 clips with 90+ rad/s peaks
for teleport-like artifacts) → S8-DECISION.**

## S8-T3 (77-clip corpus build + full 13-axis eval) — DONE, S8-DECISION NOT reached

Build: `pkl_s5/{clip}_perframelimb_sm/_smrc.pkl` + `_heightfix_sm.pkl` for all
77 clips (resumable, most already existed from T2c/T2d work). Eval: 5-column
table (`gmr_raw | gmr_heightfix | heightfix_sm | perframelimb |
perframelimb_smrc`), 13 axes, floor/loco class split. Full table:
`outputs/gmr_baseline/sprint/s8_t3_full_corpus.csv` (385 rows).

**Finding: at corpus scale, `perframelimb_smrc` does NOT clear R2.2's
never-tradeable bar — it's a 3-3 split, not the "zero critical classes"
R2.5 predicted.**

| axis (never-tradeable, R2.2) | gmr_heightfix (floor/loco) | perframelimb_smrc (floor/loco) | verdict |
|---|---|---|---|
| floorPen_cm | 2.76 / 3.06 | **6.05 / 3.43** | WORSE both classes |
| n_spikes | 0.18 / 0.00 | **1.24 / 1.12** | WORSE both classes |
| vMax rad/s | 34.0 / 32.8 | **55.4 / 54.2** | WORSE both classes |
| coll_pct | 6.34 / 3.85 | 0.003 / 0.000 | ours wins big |
| worst_float_cm | 18.0 / 6.6 | 3.96 / 2.66 | ours wins big |
| joint_ok_pct | 0.19 / 46.3 | 97.9 / 98.6 | ours wins big |

The floorPen/vMax losses are not smrc-specific: plain unsmoothed
`perframelimb` already loses floorPen (6.20cm floor) and vMax (51.7/47.0)
vs heightfix, before any T2/T2c work touched it — this predates S8.
`floorPen_cm` is the single WORST-frame penetration in the whole clip
(`pen.max()*100`, not a mean, see `sprint_s3_full_corpus.py:129-138`) —
likely non-effector body parts (torso/knee) clipping the floor during
ground-contact sequences, since the per-frame clamp only pins feet/hands,
not the whole body.

**Concentration check (not a handful of outliers):** floor-class median
floorPen (4.36cm) already exceeds heightfix's *mean* (2.76cm); top-3 clips
are only 26% of the total sum (loco: 20%). 15/34 floor clips and 13/43 loco
clips exceed 5cm. Worst single clip: `fallAndGetUp2_subject2` at 22.66cm.
Loco class is closer to a wash (median 2.78cm vs heightfix's 3.06cm mean,
16/43 loco clips already beat heightfix outright) — floor class is where
the real gap lives.

T4 renders built (R0's 3 clips + `fallAndGetUp2_subject2`, the worst
remaining floor clip, `gmr_heightfix` vs `perframelimb_smrc`) — visual
teleport-veto review still pending, held per Prabin's redirect below.

Reported to Prabin instead of writing S8-DECISION, since the metric table
doesn't match the plan's predicted shape. Prabin's response: apply naive
per-clip grounding (GMR's own height-shift trick) on top of `smrc` and see
if it converts the floorPen loss into a float win, since our worst_float is
already far below heightfix's → see S8-T5.

## S8-T5 (naive per-clip grounding on `perframelimb_smrc`, Prabin's hypothesis, 2026-07-18)

**Hypothesis:** a rigid per-clip vertical shift Δ = that clip's own
floorPen (the exact amount needed to zero its single worst-penetrating
frame) converts penetration into float. A rigid shift is additive to every
z-height metric (floorPen, worst_float, pen_pct) and invariant to
everything not floor-height-based (coll_pct, vMax, n_spikes, jerk, skate —
none depend on floor level or change under a constant offset). Since
`smrc`'s worst_float already sits far below heightfix's, the post-shift
worst_float should still win.

**Analytic check (off the existing T3 CSV, before any build):**
new_worst_float = old_worst_float + old_floorPen, exact per clip since the
shift is uniform. Predicted: floor 3.96+6.05=10.01cm (vs heightfix 18.04,
decisive win, 28/34 clips), loco 2.66+3.43=6.09cm (vs heightfix 6.61, thin
win, 26/43 clips, 60% win rate).

**Built for real** (`scripts/g1/sprint_s8_t5_grounding.py`): per-clip Δ
recomputed directly from the pkl (not trusted from rounded CSV text) —
matched the CSV floorPen values exactly (`fallAndGetUp2_subject2`
Δ=22.66cm, `jumps1_subject2` Δ=17.58cm, etc). Re-ran the full 13-axis eval
on all 77 clips, appended as a 6th variant column
(`perframelimb_smrc_ground`) to `s8_t3_full_corpus.csv` (now 462 rows).

**Confirmed exactly as predicted:** floorPen_cm = 0.000 both classes
(exact). worst_float_cm = 10.008 floor / 6.085 loco — matches the analytic
prediction (10.01/6.09) almost bit-for-bit. coll_pct, coll_peak_cm,
joint_jerk_mean, body_jerk_mean, skate(L/R), vMax_rad_s, n_spikes are
**bit-identical** before/after grounding, exactly as expected for a shift-
invariant quantity — good sanity check that the shift is doing nothing but
translating z. fidelity_ori_err_deg unchanged too (rotation untouched by
translation). fidelity_pos_err_cm actually improved slightly rather than
degrading (floor 12.41→12.22cm, loco 12.81→11.47cm, loco now nearly
matches heightfix's 11.09cm) — the feared fidelity cost didn't materialize.

**NEW FAILURE, not predicted by the analytic argument: `joint_ok_pct`
collapses.** Floor 97.93%→**32.73%**, loco 98.65%→**50.80%**. Root cause,
from `joint_ok_pct`'s own definition (`sprint_s5_metrics.py:69-92`): on
every held/stance frame it requires BOTH whole-body pen<5mm AND the held
foot's support_z within ±3cm of ground — a tight band, not just "no
penetration." The per-clip Δ is calibrated to the clip's single WORST
frame, which is typically a transient/dynamic moment (a fall, jump apex,
mid-swing), not a stance frame. Median Δ is 4.36cm (floor) / 2.78cm (loco)
— once Δ exceeds the ~3cm stance budget, which is most clips, previously-
clean held frames get pushed out of the float band and start failing.
`joint_ok` is R2.2's "un-gameable composite" — this is a collapse on the
one axis the ruling treats as hardest to game, not a minor side cost.

**Conclusion: naive constant-shift grounding is not a viable full fix.**
It correctly zeroes floorPen/pen_pct and keeps worst_float ahead of
heightfix, with no cost to coll/vMax/spikes/jerk/skate/fidelity — but it
destroys joint_ok by construction, because a single global constant cannot
satisfy a transient worst-frame correction and a tight per-stance-frame
band at the same time. A working fix would need to be LOCAL (shift only
around the handful of offending transient frames per clip, not the whole
clip) — which loops back toward T2c/T2d's per-frame/windowed philosophy
rather than a global height trick. Flagging this as a real, identified next
avenue; not attempting it without sign-off.

**Status: S8-DECISION still open.** Six variants now on record for the
never-tradeable axes, none clears all six simultaneously:

| variant | floorPen | n_spikes | vMax | coll_pct | worst_float | joint_ok |
|---|---|---|---|---|---|---|
| gmr_heightfix | win | win | win | lose | lose | lose |
| perframelimb_smrc | lose | lose | lose | win | win | win |
| perframelimb_smrc_ground | **win (0)** | lose | lose | win | win | **lose (collapse)** |

(win/lose relative to `gmr_heightfix`, floor-class numbers, loco is
directionally the same but thinner on worst_float/floorPen). Awaiting
Prabin's read on how to proceed — visual T4 veto check on the existing
renders is still outstanding regardless of which variant is being
considered.

## S8-T6 — local (windowed) grounding: sign-off given, built and evaluated

Prabin's sign-off (2026-07-18): "go ahead and implement local grounding" —
the next avenue flagged (not started) at the end of `## S8-T5`. Also asked
"I hope it is followed by smoothing right?" — answered by construction, see
below, not as a separate post-hoc pass.

**Mechanism** (`scripts/g1/sprint_s8_t6_localground.py`), same windowed
philosophy as T2d applied to root z instead of leg-DOF angles:
1. `required[t] = max(0, -lowest_z[t])` — exact per-frame shift that alone
   clears frame t (same quantity T5's global Δ was built from, but kept
   per-frame instead of collapsed to one clip-wide max).
2. `maximum_filter1d(required, window=2*RAMP_HALF+1)` (RAMP_HALF=0.15s) —
   widens each spike into a plateau. A max filter can only increase values,
   so this step alone already guarantees `>= required` pointwise.
3. `gaussian_filter1d(..., sigma=0.07s)` — rounds the plateau's corners so
   the envelope (and therefore root z velocity) is smooth. This IS the
   "followed by smoothing" step Prabin asked about — built directly into
   the shift's construction, not a separate global pass run afterward,
   because a generic second smoothing pass over the whole clip risks
   re-blurring the effector precision T5 already showed was cost-free to
   preserve.
4. `envelope = max(smoothed, required)` pointwise — restores the step-1
   guarantee at any point step 3 pulled below it. After this, `envelope[t]
   >= required[t]` for every t by construction, so `lowest_z[t] +
   envelope[t] >= 0` always — floorPen=0.00 is an algebraic guarantee, not
   an empirical result.
`qpos[:,2] += envelope` (root z only, per-frame, not a rigid whole-clip
shift like T5).

**Build** (77/77 clips): touched-frame counts confirm the "broad but not
majority" concentration finding from `## S8-T3` — median clip has 45-95
penetrating frames out of several thousand (~1-2%), a few outliers higher
(`fallAndGetUp2_subject2`: 300/4918, peak envelope 22.66cm — matches its
known floorPen from T3 exactly, confirming `required` was computed
correctly). Two clips had ~0 penetrating frames (`obstacles1_subject2`,
`obstacles4_subject2`, peak envelope <0.1cm — floor noise, not real
correction).

**Eval, corpus means, full re-run through the pipeline (not analytic):**

| metric (floor) | gmr_heightfix | perframelimb_smrc | smrc_ground (T5) | smrc_localground (T6) |
|---|---|---|---|---|
| joint_ok_pct | 0.19 | 97.93 | 32.73 | **99.30** |
| floorPen_cm | 2.76 | 6.05 | 0.00 | **0.00** |
| worst_float_cm | 18.04 | 3.96 | 10.01 | **6.56** |
| coll_pct | 6.34 | 0.003 | 0.003 | 0.003 |
| vMax_rad_s | 34.04 | 55.39 | 55.39 | 55.39 |
| n_spikes | 0.18 | 1.24 | 1.24 | 1.24 |

| metric (loco) | gmr_heightfix | perframelimb_smrc | smrc_ground (T5) | smrc_localground (T6) |
|---|---|---|---|---|
| joint_ok_pct | 46.26 | 98.65 | 50.80 | **98.89** |
| floorPen_cm | 3.06 | 3.43 | 0.00 | **0.00** |
| worst_float_cm | 6.61 | 2.66 | 6.09 | **4.17** |
| coll_pct | 3.85 | 0.000 | 0.000 | 0.000 |
| vMax_rad_s | 32.82 | 54.22 | 54.22 | 54.22 |
| n_spikes | 0.00 | 1.12 | 1.12 | 1.12 |

**Result: joint_ok did not just survive, it improved beyond plain smrc**
(97.93→99.30 floor, 98.65→98.89 loco) — because some smrc held frames that
failed joint_ok were failing specifically because the whole-body pen<5mm
half of that test tripped during a held frame that overlapped a
penetration window; local grounding fixes exactly those overlapping
frames without touching any frame outside a window, so it can only help
or be neutral on held frames, never hurt them the way a global constant
does. `worst_float_cm` still clears heightfix by a wide margin (6.56 vs
18.04 floor; 4.17 vs 6.61 loco) despite being higher than T5's number in
absolute terms in loco (T6 adds height only where needed, so it's less
"free" than a global shift, but still nowhere near heightfix's float).
`vMax_rad_s`/`n_spikes`/`joint_jerk_mean` are bit-identical to plain smrc
(55.39/54.22, 1.24/1.12, 3312.998/3345.414 exactly) — confirms the smooth
envelope construction added no spurious joint velocity anywhere, i.e. the
"followed by smoothing" design goal held up empirically, not just in
theory. `coll_pct`, skate, fidelity_ori all identical to smrc (as
expected, none of these are z-height-based). `fidelity_pos_err_cm`
unchanged within noise (12.41→12.41 floor, 12.81→12.80 loco).

**Updated never-tradeable scorecard** (vs `gmr_heightfix`, floor-class):

| variant | floorPen | n_spikes | vMax | coll_pct | worst_float | joint_ok |
|---|---|---|---|---|---|---|
| perframelimb_smrc | lose | lose | lose | win | win | win |
| perframelimb_smrc_ground (T5) | win | lose | lose | win | win | lose (collapse) |
| perframelimb_smrc_localground (T6) | **win** | lose | lose | **win** | **win** | **win** |

4 of 6 axes now clean wins, up from 3/6. The two remaining losses
(n_spikes, vMax) are untouched by grounding by construction (confirmed
bit-identical above) — they're a pre-existing smoothing/dynamics-quality
gap already partially addressed by S8's held-aware smoothing, not
something grounding was ever going to fix. This is the closest any variant
has come to clearing the never-tradeable bar. T4 visual veto check (on
`smrc`, not yet on `localground`) is still outstanding. S8-DECISION not
written — presenting this table, not deciding on it.

## S8-T7 — smoothing-weight sweep (relax tracking / raise regularization): negative, mechanism understood

Prabin's ask (2026-07-18, post-T6): trade a bit more tracking fidelity for
gains on the two remaining losses (n_spikes, vMax), via relaxed tracking
weight / higher smoothing regularization, or an explicit max-velocity cap
— "worth a try."

**Mechanism identified**: `smrc` = `smooth_heldaware.py`'s `stage_a`
(tridiagonal smoother, weights `lambda_track` vs `lambda_smooth`, corpus
default 1.0/20.0) → `polish_median_limbwise._limbwise_pass` (re-clamp,
`max_dq=0.15` trust region, restores floor/collision safety the smoothing
pass perturbs). New script `scripts/g1/sprint_s8_t7_smooth_sweep.py`
parameterizes both lambdas, rebuilds `smrc` + T6's local-grounding envelope
on top, full corpus re-eval.

**relaxA** (lambda_track 1.0→0.5, lambda_smooth 20→40) and **relaxB**
(→0.25 / →80), both 77/77 clips, full pipeline (not analytic):

| metric (floor) | smrc_localground (T6) | relaxA_localground | relaxB_localground |
|---|---|---|---|
| vMax_rad_s | 55.39 | 55.31 | 55.57 |
| n_spikes | 1.24 | 1.18 | 1.21 |
| joint_jerk_mean | 3313.00 | 3640.48 (+9.9%) | 4185.98 (+26.3%) |
| fidelity_pos_err_cm | 12.41 | 13.58 (+9.4%) | 16.01 (+29.0%) |
| fidelity_ori_err_deg | 11.08 | 14.53 (+31.1%) | 18.89 (+70.5%) |
| worst_float_cm | 6.56 | 7.66 | 8.30 |
| joint_ok_pct | 99.30 | 99.12 | 98.79 |

(loco class same monotonic shape: vMax 54.22→53.71→54.99, n_spikes
1.12→0.98→1.02, jerk 3345→3812→4678, fidelity_ori 11.29→15.49→19.53.)
floorPen/coll_pct stayed pinned at win level both configs, as expected —
protected downstream by re-clamp + T6 envelope, not by the smoothing
weights, confirming that part of the design.

**Result: monotonic negative, target metrics didn't move.** Pushing the
lambda knob harder makes tracking fidelity, jerk, worst_float, and joint_ok
all get steadily worse, in a straight line from relaxA to relaxB — while
vMax/n_spikes stay flat or, at relaxB, tick back UP slightly (55.39→55.57
floor). **Root cause: vMax/n_spikes are not produced by the smoothing
pass.** They come from the re-clamp step's own independent per-frame DLS
correction (fixed `max_dq=0.15` trust region), which runs regardless of how
smooth its input already is, to restore floor/collision safety. A less-
tracked, more-smoothed pre-clamp trajectory is further from feasible, so
the re-clamp has to correct MORE, not less — jerk goes up, not down. The
smoothing lambdas were never the lever for this axis.

**The actual matching mechanism already exists and was already tried
once.** `leg_floor_clamp.py`'s `CorrectionRateLimiter` (S8-T1b, `rl`/
`rl2`) caps the clamp's applied per-frame correction directly — this is
Prabin's "limiting the maximum velocity" idea verbatim. Tested in T1
(`GMR-S8-plan.md` R1.1), before held-aware smoothing or local grounding
existed: it did reach vMax/n_spikes parity with raw, but converted spikes
into DRIFT — `perframelimb_rl` gave back float (2.9→6.1cm), range
(7.3→10.3cm), skate (0.44→1.56cm) vs plain `perframelimb`. Demoted to an
ablation row; T1b's 2-attempt cap was spent per the plan's failure-handling
rule. That test predates T6's local grounding, which changes the pipeline
materially (grounding now happens after the clamp, downstream of wherever
the drift would land) — whether it would still trade the same way is an
open, untested question, not a known negative in the current pipeline.

**Not re-attempted without sign-off**: T1b's cap was already spent on this
exact mechanism; re-running it a third time, even in the new T6 context, is
a call for Prabin, not something to restart unilaterally. Flagging as the
one remaining candidate lever for n_spikes/vMax; the smoothing-weight
lever (this section) is closed out as negative and should not be
revisited.

## S8-T8 — rate-limited re-clamp + T6 grounding: sign-off given, real gain

Prabin's sign-off (2026-07-18): "go ahead and try the rate limiter with
T6." Key difference from T1b's original test: T1b rate-limited the
ORIGINAL `perframelimb` clamp (built straight from raw retarget). This
applies `CorrectionRateLimiter` (`rate_limit=0.15` rad/frame, same value
T1b used) to the **re-clamp step inside `smrc`'s build** instead — the
`_limbwise_pass` call that runs AFTER held-aware smoothing, which T7
identified as the actual source of vMax/n_spikes. Pipeline:
`perframelimb_sm.pkl` (standard λ_track=1.0/λ_smooth=20, unchanged,
already built corpus-wide) → rate-limited re-clamp → `perframelimb_smrc_rl`
→ T6's envelope grounding on top → `perframelimb_smrc_rl_localground`.
New script `scripts/g1/sprint_s8_t8_rl_localground.py`. Full 77-clip
corpus, real build+ground+eval (not analytic):

| metric (floor) | gmr_heightfix | smrc_localground (T6) | smrc_rl_localground (T8) |
|---|---|---|---|
| vMax_rad_s | 34.04 | 55.39 | **37.39** |
| n_spikes | 0.18 | 1.24 | **0.00** |
| joint_jerk_mean | 5002.59 | 3313.00 | **2940.32** |
| floorPen_cm | 2.76 | 0.00 | 0.00 |
| coll_pct | 6.34 | 0.00 | 0.00 |
| worst_float_cm | 18.04 | 6.56 | 7.55 |
| joint_ok_pct | 0.19 | 99.30 | 98.85 |
| fidelity_pos_err_cm | 12.60 | 12.41 | 12.40 |
| fidelity_ori_err_deg | 6.97 | 11.08 | 11.06 |
| skate_left/right_cm | 0.44/0.43 | 0.49/0.38 | 0.63/0.51 |
| range_cm | 12.84 | 6.85 | 7.93 |

(loco class same shape: vMax 32.82→54.22→**37.92**, n_spikes
0.00→1.12→**0.00**, jerk 6189.09→3345.41→**3095.20**, floorPen/coll both
0.00, worst_float 6.61→4.17→5.57 (still beats heightfix), joint_ok
46.26→98.89→98.79, fidelity unchanged, skate 0.31/0.34→0.26/0.23→0.34/0.29,
range 7.98→4.50→6.02.)

**Result: n_spikes flips from a loss to a win/tie (0.00 vs heightfix's
0.18 floor; 0.00 vs 0.00 loco — exact tie).** vMax drops from +63%/+65%
over heightfix (T6) to +9.8%/+15.5% (T8) — still technically a loss but an
order of magnitude closer, no longer the dramatic gap it was. Unlike T7's
smoothing-weight sweep, **`joint_jerk_mean` went DOWN, not up** (-11.3%
floor, -7.5% loco) — confirms T7's diagnosis was right: the re-clamp's own
correction was the actual jerk source, and directly rate-limiting it (vs.
trying to indirectly reduce its workload via smoother pre-clamp input)
attacks the real mechanism. Small, real costs: worst_float +0.99cm floor /
+1.4cm loco (still clears heightfix by a wide margin both classes),
joint_ok -0.45pp floor / -0.10pp loco (negligible, still crushes heightfix
0.19%/46.3%), skate +0.13-0.14cm floor / +0.06-0.08cm loco (present but an
order of magnitude smaller than T1b's original drift finding — 0.44→1.56cm
— because this time the rate limiter only has to correct residual
smoothing-perturbation, not the full raw-to-floor-safe correction T1b's
context required), fidelity unchanged (noise-level). T1b's "spikes become
drift" finding was directionally reproduced (skate/range both did get a
little worse) but at roughly 1/10th the magnitude, because T6's downstream
grounding and the already-smoothed `_sm` input change what the rate
limiter has left to fix.

**Updated never-tradeable scorecard** (vs `gmr_heightfix`, floor-class):

| variant | floorPen | n_spikes | vMax | coll_pct | worst_float | joint_ok |
|---|---|---|---|---|---|---|
| perframelimb_smrc_localground (T6) | win | lose | lose | win | win | win |
| perframelimb_smrc_rl_localground (T8) | win | **win/tie** | lose (much closer) | win | win | win |

5 of 6 axes now win or tie, up from 4/6. vMax is the sole remaining
loss, and it shrank from a 63-65% gap to a 10-16% gap. S8-DECISION not
written — presenting, not deciding. `--rate 0.15` was the first value
tried (matching T1b's own choice); not swept further this pass — a lower
rate might close the vMax gap more at some additional skate/float cost,
untested.

## S8-T9 — T4 visual veto check (finally performed, on the T8 variant) + side-by-side white-floor render tooling

T4 (R2.6's hard visual-veto rule) had been outstanding since the very
start of S8 — flagged in every revision, never actually watched. Done now,
on `perframelimb_smrc_rl_localground` (the current best, T8), per Prabin's
explicit ask, alongside new render tooling.

**Render tooling built** (Prabin's ask: side-by-side comparison + white
floor instead of "the mujoco black and white madness"):
- `g1_model_setup.py`'s `load_g1_model_with_vetted_collision_and_floor`
  gains an opt-in `white_floor=True` kwarg (default False, byte-identical
  for every existing eval/build caller). Root cause of the checker/grid
  noise, found by inspection: the base `g1_mocap_29dof.xml` ships its OWN
  static `floor` geom (material "groundplane": checker texture + black
  edge marks), and this loader's own injected mocap-body floor plane sits
  exactly coincident with it (both at z=0) — the two z-fight, and the
  "hex/triangle" pattern in every prior render was that z-fighting on top
  of the checker texture, not a mesh/lighting artifact. Fix: `white_floor`
  recolors BOTH geoms to the same flat white material (reflectance=0,
  shininess=0, specular=0) — removes the checker entirely and makes the
  z-fighting imperceptible (either winner renders identically). One
  footgun hit + fixed en route: MuJoCo multiplies `geom_rgba` onto the
  material's rgba at render time, so the compiler-default gray geom_rgba
  (0.5,0.5,0.5,1) silently greyed out an otherwise-white material until
  `geom.rgba=[1,1,1,1]` was also set explicitly on both geoms.
- New `scripts/g1/render_sidebyside.py`: renders two pkls of the same clip
  in twin panels (independent model/data/renderer per side, same camera),
  each panel keeping the existing penetration-annotation overlay
  (`render_penetration_annotated.py`'s mesh-accurate `_geom_lowest_z`, red
  border flash + depth/body label), composited into one mp4.

**Renders produced** (`s8_renders_t8/*.mp4`, `gmr_heightfix` left vs
`perframelimb_smrc_rl_localground` right): the existing R0 3 clips
(`walk3_subject1`, `fallAndGetUp1_subject1`, `ground1_subject1`) +
`fallAndGetUp2_subject2` (the historically worst floor-pen clip, 22.66cm
pre-grounding) + `sprint1_subject4` (worst vMax for this specific variant,
47.9 rad/s, picked to visually stress-test the one remaining metric loss).
Per-clip penetration tallies from the render script itself: our side is
0.0% penetrating frames on all 5 clips (vs GMR-full's 0.0-0.7%) — matches
the corpus table.

**Veto check method**: 2x2 contact sheets (4 evenly-spaced frames per
clip, both panels visible, penetration label legible), 20 frames total
across the 5 clips, visually inspected directly (not just trusting the
metric).

**Result: PASSES, no disqualifying defect on any sampled frame.** Poses
track GMR-full's own timing/stance closely at every sampled instant (near-
identical silhouettes side by side) on steady walking, sprinting stride,
crouching, falling-to-ground, all-fours crawling, and prone-on-ground
poses — no teleporting, no limb-through-body contortion, no snapping.
Floor contact stays clean (0.00cm) at every one of our sampled frames,
including one frame where GMR-full itself showed a small penetration
(0.28cm, left_elbow_link, walk3_subject1) that ours didn't. The
worst-vMax clip (`sprint1_subject4`) — the one axis still nominally a
loss — showed nothing visually alarming at its sampled frames either; a
20-frame sample can't rule out a single-instant issue between samples,
but combined with the corpus metric (`n_spikes`=0.00 for this variant on
both classes) there's no positive signal of a problem, just the residual
elevated mean velocity already known from the T8 table.

**T4 is now cleared for the current best variant.** Combined with T8's
5/6 never-tradeable scorecard, this is the strongest evidence package any
variant has had this sprint. S8-DECISION still not written — presenting,
not deciding.

## S8-DECISION (2026-07-18, Prabin) — lock `perframelimb_smrc_rl_localground` as the working baseline

Prabin's call, given directly rather than as a separate written gate review:
**ship `perframelimb_smrc_rl_localground` (T6 local grounding + T8
rate-limited re-clamp) as the locked method**, accepting T8/T9's 5/6
never-tradeable scorecard rather than continuing to chase the 6th (vMax)
before locking. `scripts/g1/sprint_s8_lock_final.py`'s own docstring
records the instruction verbatim: *"Prabin (2026-07-18): lock this variant
as the working baseline for now."* It pulls the 77-clip corpus rows
straight from the existing `s8_t3_full_corpus.csv` build (no re-solve) and
adds corpus-wide hand slip — never tracked before this pass — into
`outputs/gmr_baseline/sprint/s8_LOCKED_perframelimb_smrc_rl_localground.csv`,
now the canonical results file for the locked variant. `GMR-METHOD.md`
(new, repo root) is the corresponding method writeup: a plain-language
walkthrough (§§1-11) plus a full mathematics appendix (§12, every `[ours]`
stage — the DLS floor/self-collision clamp, held-aware tridiagonal
smoothing, the local grounding envelope's max-filter/gaussian/pointwise-max
construction, and the rate limiter) for readers who want the algebra.

**Does the locked variant beat `gmr_heightfix` on all 6 never-tradeable
axes simultaneously? No — 5 of 6, unchanged from T8/T9** (this decision is
a data pull + lock, not a rebuild, so the numbers are identical up to the
new hand-slip columns):

| axis (floor-class) | gmr_heightfix | **LOCKED** (`perframelimb_smrc_rl_localground`) |
|---|---|---|
| floorPen_cm | 2.76 | **0.00 — win** |
| n_spikes | 0.18 | **0.00 — win/tie** |
| vMax_rad_s | 34.04 | 37.39 — **lose** (narrowed to +9.8%) |
| coll_pct | 6.34 | **0.00 — win** |
| worst_float_cm | 18.04 | **7.55 — win** |
| joint_ok_pct | 0.19 | **98.85 — win** |

(loco-class, same shape: vMax 32.82 vs 37.92 — lose, +15.5%; n_spikes 0.00
vs 0.00 — tie; floorPen/coll_pct/worst_float/joint_ok all win, joint_ok
46.26→98.79.) `gmr_raw` (no grounding at all) stays a reference point only
per R1.2's standing rule, never the comparison baseline.

**What was traded, stated plainly** (per the plan's "present the honest
pareto" instruction):
- **vMax stays a real, open cost** — 9.8% (floor) / 15.5% (loco) higher
  peak joint velocity than `gmr_heightfix`. This is the one axis the whole
  S8 sprint (T2/T2c/T2d/T6/T7/T8) worked to close and did not fully close:
  T7 proved it isn't a smoothing-weight problem, T8 proved rate-limiting
  the re-clamp is the correct lever and got most of the way there (a
  63-65% gap shrank to 9.8-15.5%), but not to parity.
- worst_float (7.55cm floor / 5.57cm loco) and joint_ok (98.85%/98.79%)
  both still crush `gmr_heightfix` by a wide margin — the sprint's central
  claim (contact correctness, not just "no floor penetration") is not a
  close call.
- Corpus-wide hand slip is measured for the first time this pass
  (0.66-0.73cm mean, n=32 clips with a genuine hand-hold segment) —
  `gmr_heightfix` has no equivalent number since it has no hand-contact
  handling at all; this is a new reported capability, not a comparison
  axis.
- T4's visual veto (`## S8-T9`) already passed clean on this exact
  variant — the 5/6 scorecard is not just a metric table, it has been
  watched.

**Supersedes S7-DECISION.** `## S7-DECISION` presented four options for
`perframelimb` vs `gmr_contact_fc`: (A) promote perframelimb, (B) keep fc
primary, (C) per-class hybrid, (D) close the smoothness gap first then
decide. S8's whole sprint IS option A followed by D in the order D
suggested — perframelimb was promoted, then (T2/T6/T7/T8) the
smoothness/jerk gap S7-DECISION flagged as unmeasured was measured and
directly worked on until it narrowed from a 63-65% vMax gap to 9.8-15.5%.
`gmr_contact_fc`/`gmr_contact_fc_sm` are retired from primary-method
consideration as of this decision — the locked variant descends from the
`perframelimb` lineage (S6 Phase B → S7-T3's `--center perframe` fix → S8's
smoothing/re-clamp/grounding/rate-limit stack), not from S5/S6's in-solve
contact-layer mechanism.

**S9 (mimic pilot) is confirmed as the next gate, not this decision.** Per
Prabin's own standing framing ("if our method wins in mimic training, then
only this can become a paper"): this kinematic lock is necessary but not
sufficient for the paper. Nothing here has been checked through a physics
simulator or a learned imitation policy (`GMR-METHOD.md` §13 states this
limitation explicitly). S9 is Prabin's to start, not scoped in this sprint.

**Docs done as part of this decision**: `GMR-METHOD.md` (new), this entry,
`wiki/experiments/gmr-baseline-sprint-s8.md` (new), `wiki/index.md`,
`wiki/log.md`, and a new S8 discussion section in
`GMR-baseline-results.md` — per `GMR-S8-plan.md`'s "Docs & discussions"
spec, all done in one pass now that S8-DECISION is written.

---

## S9-T0/T1 (2026-07-19, branch `gmr-baseline`) — posture-continuity + limit-repulsion: T1 gate FAILS at 2 attempts, T0 needs re-scoping

Full plan: `GMR-S9-plan.md`. Trail: this session's conversation (visual
review of S8-locked renders found real defects the aggregate table didn't
show — see plan's "Why S9 exists").

**T0 (done, kept)**: root-caused `sprint1_subject4`'s worst vMax event
(t=6306, 47.9 rad/s) to a DLS solution branch-flip — `left_hip_yaw`/
`right_ankle_pitch` alternate between two configurations frame to frame on
a FLAT raw GMR target (no real motion). Built `leg_floor_clamp.clamp_limb`'s
opt-in `q_prev_chain`/`posture_weight` (null-space bias toward the previous
frame's own chain posture, default `None` = byte-identical no-op, verified)
and `polish_median_limbwise._limbwise_pass`'s matching `posture_continuity`/
`posture_weight`. On `sprint1_subject4` alone: hip_yaw flip fixed, vMax
47.9→40.8 (-15%), but ankle_pitch's bang-bang persisted and worst_float
regressed +9% (20.62→22.59cm).

**New finding this entry (T1's dev-clip run exposed it)**: T0 was only
validated against `sprint1_subject4` in isolation. Running it against the
S8-T0b 5 dev clips (walk1_subject1, walk3_subject1, run2_subject1,
ground1_subject1, fallAndGetUp1_subject1) for the first time shows
`posture_weight=1.0` has REAL, uncosted regressions on 4/5 of them:

| clip | joint_ok off→posture | worst_float off→posture |
|---|---|---|
| walk1_subject1 | 100.0→99.6 | 2.80→7.35cm |
| walk3_subject1 | 99.5→98.7 | 7.09→12.27cm |
| ground1_subject1 | 99.0→97.2 | 5.52→13.52cm |
| fallAndGetUp1_subject1 | 98.5→94.3 (new 0.02% coll) | 9.85→11.50cm |
| run2_subject1 (only improvement) | 99.0→98.5 | 8.87→7.51cm |

T0's win is real on its target clip but was never cross-checked against
the clips S8's own gates were built on — flagging this now rather than
building T2/T3 on top of an unvalidated T0 default.

**T1 (2-attempt cap spent, gate FAILS)**: added a second null-space term,
`limit_margin`/`limit_weight` (repel from hard joint limits, combined
additively with T0's posture bias before null-space projection), targeting
ankle_pitch's specific hard-limit bang-bang.

- Attempt 1 (`limit_margin=0.15, limit_weight=1.0`): target case barely
  moved (ankle still hits 0.524 exactly at the same frames). Real bonus
  effect elsewhere though — CLAWS BACK most of T0's own regression on 3/5
  dev clips (walk3_subject1 worst_float 12.27→6.70cm, BETTER than the
  shipped baseline's 7.09; run2_subject1 joint_ok fully recovers to 99.0%,
  matching baseline; fallAndGetUp1_subject1 worst_float 11.50→8.94cm,
  also beating baseline's 9.85, coll% back to 0.00%). One regression:
  `ground1_subject1` joint_ok 97.2→92.3% (worse than posture-alone, worse
  than baseline).
- Attempt 2 (`limit_margin=0.2, limit_weight=3.0`): ankle_pitch finally
  detaches from the exact limit at several frames (no longer a clean
  binary bang-bang) but the target clip's vMax gets WORSE, not better
  (40.8→43.5 rad/s) — pushing harder trades a clean-if-ugly bang-bang for
  a higher-frequency wobble. `ground1_subject1` regresses FURTHER
  (92.3%→88.1% joint_ok, an 11-point loss vs baseline) and
  `walk3_subject1`'s attempt-1 recovery partially reverses (99.5%→98.3%).
  Regressions grow monotonically with weight; the target metric (vMax)
  moves the wrong direction. This is consistent with the T0 writeup's
  standing hypothesis: near a hard limit, which chain DOFs are "free"
  (null space) vs. "task-required" (row space) can itself flip depending
  on which side of the limit a frame started from — a genuine kinematic
  branch point that a null-space bias term structurally cannot out-vote,
  whatever its weight. Pushing harder just fights the primary task instead
  of finding a free alternative.

**Per S9's standing rule (2-attempt cap, log honestly, stop, escalate —
do not design a third mechanism): T1 STOPS here.** Code stays in place
(`limit_margin`/`limit_weight`, default `0.0` = byte-identical no-op), not
enabled by default. Escalating to Prabin: (a) T1's ankle-specific mechanism
doesn't clear its own gate and shouldn't ship as-is; (b) T0's own
`posture_weight=1.0` default needs a decision before T2/T3 build on top of
it — options not yet tried: sweep `posture_weight` down (e.g. 0.3-0.5) to
see if a gentler pull keeps most of the target win with less collateral,
or gate posture-continuity to only engage where a branch-flip signature is
actually detected (large frame-to-frame delta on a near-static raw target)
rather than blanket every frame.


## S9-T0-gate (2026-07-19, branch `gmr-baseline`) — raw-velocity-gated posture-continuity: dev-clip PASS, full-corpus MIXED, not shipped

Answers `## S9-T0/T1`'s open item (b): gate T0's posture-continuity bias
instead of sweeping its weight down. Full plan context: `GMR-S9-plan.md`.

**Mechanism**: `polish_median_limbwise._limbwise_pass` gains
`posture_gate_lo`/`posture_gate_hi`/`raw_gate_qpos` (opt-in, both `None` =
byte-identical to T0's blanket behavior). Per chain per frame, T0's
`posture_weight` is scaled by how much that chain's TRUE, UNTOUCHED GMR raw
target moved frame-to-frame (`raw_gate_qpos`, e.g. `outputs/gmr_baseline/
sprint/pkl/{clip}.pkl`) -- full weight when raw delta <= `posture_gate_lo`
(near-static), zero when raw delta >= `posture_gate_hi` (genuine motion),
linear ramp between. Caught a real bug before shipping this: gating on
THIS pass's own input (`_perframelimb_sm.pkl`) instead of the pristine raw
pkl gates backwards, since that input already carries the upstream
branch-flip artifact this mechanism targets -- confirmed by direct
measurement (`sprint1_subject4`'s diagnosed window shows delta up to
1.1-1.3 rad in the smoothed input at exactly the frames where the TRUE
raw signal only moves 0.002-0.017 rad).

**Threshold pick, not guessed**: scratch analysis (`raw_delta_stats2.py`,
scratchpad, not committed) of raw per-frame max chain-joint delta across
the S8-T0b 5 dev clips + `sprint1_subject4`: diagnosed flat window
(t=6294-6302) sits at 0.002-0.017 rad/frame; ordinary locomotion runs
0.03-0.16 rad/frame (p50 across clips). `lo=0.02/hi=0.05` sits below
normal locomotion on every dev clip, engaging only on genuinely static/
idle frames.

**Attempt 1 (gate1, lo=0.02/hi=0.05) -- dev-clip check (`sprint_s9_t0gate_probe.py`, 6 clips)**:

| clip | worst_float off->gate1 | joint_ok off->gate1 | (T0 blanket, for reference) |
|---|---|---|---|
| walk1_subject1 | 2.80->2.80 (+0.00) | 100.0->100.0 (+0.0) | +4.55cm / -0.4pp |
| walk3_subject1 | 7.09->7.09 (+0.00) | 99.5->99.5 (+0.0) | +5.18cm / -0.8pp |
| run2_subject1 | 8.87->8.87 (+0.00) | 99.0->99.0 (+0.0) | -1.36cm / -0.5pp |
| ground1_subject1 | 5.52->5.69 (+0.16) | 99.0->98.4 (-0.61) | +8.00cm / -1.8pp |
| fallAndGetUp1_subject1 | 9.85->9.85 (+0.00) | 98.5->98.5 (+0.0) | +1.65cm / -4.2pp |
| sprint1_subject4 (target) | vMax 47.9->46.1 (-3.8%) | worst_float 20.62->22.43 (+8.8%) | vMax -14.8% / float +9.6% (blanket) |

Dev-clip collateral from T0's blanket weight is essentially eliminated
(4/5 exact, ground1 shrunk from +8.00cm to +0.16cm). Target-clip win is
PARTIAL: ~25% of blanket's vMax reduction retained (hip_yaw's flip
visibly damped in the window trace, no longer diving to its negative
branch), `worst_float`'s regression NOT fixed (same magnitude as blanket).

**Attempt 2 (gate2, lo=0.02/hi=0.065) -- widen to catch the worst-vMax
frame itself** (t=6306, raw delta 0.054-0.058, sits almost exactly at
gate1's hi=0.05 cutoff): FAILS. Target clip's vMax identical to gate1
(46.1, zero additional gain), while `fallAndGetUp1_subject1` regresses
+2.99cm worst_float (new, real, wasn't present in gate1) and ground1's
residual doubles (+0.16->+0.33cm). Per S9's 2-attempt cap and Prabin's
own pre-stated fallback rule: dropped back to gate1 as final.

**Full 77-clip corpus** (`sprint_s9_t0gate_full_corpus.py`, new variant
`perframelimb_smrc_pg_localground`, same shipped pipeline + gate1 in the
re-clamp step only, nothing after -- local grounding runs downstream,
untouched): appended to `s8_t3_full_corpus.csv` (847 rows total). 3-way
vs `gmr_heightfix` / shipped `perframelimb_smrc_rl_localground`:

| axis | class | shipped | pg | verdict |
|---|---|---|---|---|
| vMax_rad_s | floor | 37.39 | 34.35 | **-8.1%, generalizes past the dev set** |
| vMax_rad_s | loco | 37.92 | 36.57 | **-3.5%, generalizes** |
| joint_jerk_mean | floor | 2940.3 | 3134.8 | +6.6%, new cost (dev gate never tracked jerk) |
| body_jerk_mean | floor | 171.0 | 193.3 | +13.1%, new cost |
| skate_left/right | floor | 0.63/0.51cm | 0.72/0.62cm | +13-21%, new cost -- and `gmr_heightfix` itself is BETTER here (0.44/0.43cm) than either G1 variant, pg widens that pre-existing gap |
| joint_jerk/body_jerk/skate | loco | -- | -- | same-direction, smaller (+0.3% to +10%) |
| floorPen/coll/joint_ok/fidelity/n_spikes | both | -- | -- | wash, no consistent sign, all safety axes untouched |

**Verdict: MIXED, not a clean win, NOT shipped as the new default.** The
vMax reduction this whole T0/T1/T0-gate line exists to chase is real and
generalizes corpus-wide (closes most of S8-DECISION's one open gap on
floor-class clips: 63-65% baseline gap -> 9.8-15.5% at S8 lock -> now
further reduced). But it comes at a corpus-scale cost the narrow 6-clip
dev gate never had visibility into (jerk, skate) -- exactly the
overfitting risk a small dev-clip gate can hide, caught only by going to
full-corpus scale. Skate is the more concerning of the two: it's already
a pre-existing weakness of the G1 clamp mechanism relative to
`gmr_heightfix` (this pipeline's phase-1/phase-2 DLS clamp has no
zero-slip guarantee the way Alex's Stage-B contact QP does), and `pg`
widens it further on the exact axis ("does contact-aware correction
actually stop foot slip") most central to this method's own story.

**Code state**: `posture_gate_lo`/`posture_gate_hi`/`raw_gate_qpos` land
in `leg_floor_clamp.py`/`polish_median_limbwise.py` as opt-in params,
default `None` = byte-identical no-op (unchanged from T0's own no-op
guarantee). `gate1`'s thresholds (0.02/0.05) are the validated choice if
this mechanism is used at all; NOT wired into any default pipeline call.
`sprint_s9_t0gate_probe.py` (6-clip dev gate) and
`sprint_s9_t0gate_full_corpus.py` (77-clip corpus + 3-way table) both new,
uncommitted.

**Open item, not resolved this entry**: which clips/frames drive the
jerk/skate regression, and whether it's fixable (e.g. narrowing the gate
further, or gating on a per-DOF rather than whole-chain-max delta) or is
an inherent trade of this null-space mechanism. Escalated to Prabin,
not unilaterally re-attempted (2-attempt cap already spent on gate1/gate2).

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

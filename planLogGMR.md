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

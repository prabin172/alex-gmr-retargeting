# GMR-baseline-plan — Week 1 execution plan (E1+E2+E3)

**Status: DRAFT — awaiting Prabin's approval + T0 (his commit) before any execution.**

**What this is:** the implementation plan for `GMR-baseline.md` §4's first three de-risk
experiments (E1 GMR-on-floor-clips, E2 eval de-Alexing, E3 Stage-A+grounding polish on G1),
scoped to one week, ending in the **first results narrative** for Option A ("contact-aware
kinematic polish"). Written for a cheaper model to execute task-by-task; Prabin gates at the
marked checkpoints. Log every task's numbers/decisions to `planLogGMR.md` (create it, same
convention as `planLog.md`).

**Not in scope this week:** Stage B contact QP on G1 (E4), BeyondMimic policy runs (E5),
ingesting OUR FBX clips into GMR (see "Deferred" at bottom), any change to the Alex pipeline.

---

## Ground rules for the executing model

1. Read this file + `GMR-baseline.md` first. Do NOT re-explore either codebase for what's
   already stated here.
2. **Never modify the GMR clone** at `/home/ptimilsina/projects/GMR`. It must stay pristine
   upstream — the paper claim is "GMR out-of-box". It is imported as an installed package.
   New code lives in THIS repo under `scripts/g1/`.
3. **Quaternion footgun:** GMR's saved pkl stores `root_rot` as **xyzw** (converted on save,
   `scripts/bvh_to_robot.py:166` in the clone). Everything in OUR repo is **wxyz**. Convert at
   the pkl boundary, once, in the loader (T5) — nowhere else.
4. LAFAN1 is 30 fps. Stage-A `lambda_smooth` scales as fps² (docstring of
   `scripts/ihmc_json_to_stage4_npz.py`: pipeline's 320 @120 Hz ≡ 20 @30 Hz). At 30 fps use
   **20** as the starting value.
5. Alex conventions still apply to shared code: free-root qpos `[x,y,z,qw,qx,qy,qz,joints]`
   (G1 = 7+29 = same shape as Alex, per `GMR-baseline.md` §6).
6. Every refactor of existing Alex scripts must pass a **byte/number-identical regression
   gate** before proceeding (stated per task). This repo's discipline: verify no-op, then build.
7. Outputs go to `outputs/gmr_baseline/` (gitignored, local). Wiki updates only at T10.
8. Environment: conda env `gmr` (exists — usage strings in this repo already say
   `conda run -n gmr python ...`). Install GMR into it: `pip install -e /home/ptimilsina/projects/GMR`.

## Repository strategy (decided)

- **This repo** (`alex-gmr-retargeting`) is the working repo. New branch `gmr-baseline`,
  created AFTER T0. All new scripts: `scripts/g1/`. Plan log: `planLogGMR.md` (repo root).
- **GMR clone** (`/home/ptimilsina/projects/GMR`, MIT, tracked to upstream, current through
  Jan 2026) is a read-only dependency: models (`assets/unitree_g1/g1_mocap_29dof.xml` — has a
  floor plane geom already, line ~313), entry scripts (`scripts/bvh_to_robot.py`), package
  (`general_motion_retargeting`). Its untracked scripts (`build_fbx_kinematic_canonical_v2.py`
  etc.) are ours from an earlier bridge attempt — leave them, don't build on them this week.
- **Data:** LAFAN1 BVH → `data/raw/lafan1/` in this repo (gitignored under `data/`).

## Why LAFAN1's own floor clips for E1 (decided)

`GMR-baseline.md` §4 E1 suggested our FBX clips via a Blender→BVH bridge. Cheaper: LAFAN1
itself contains floor-contact sequences (fall/get-up/crawl-type names — verify at T2), which
GMR loads **natively** (`--format lafan1`) — zero ingest work, and the motivation figure
becomes "GMR on its own benchmark's floor clips", which is stronger, not weaker. Our FBX
ingest is deferred to week 2+.

---

## Week overview

| Day | Tasks | Milestone |
|---|---|---|
| 1 | T0 (Prabin) + T1 setup/smoke + T2 clip selection | GMR runs locally on one walk clip |
| 2 | T3 baseline batch + videos | E1 raw material in hand |
| 3 | T4 eval refactor (regression-gated) | eval core is model-agnostic |
| 4 | T5 G1 loader + baseline metrics | **M1: motivation table (E1+E2 done)** |
| 5 | T6+T7 polish script: Stage A on G1 | first polish deltas |
| 6 | T8 grounding + T9 full before/after table + renders | **M2: E3 kill-test result** |
| 7 | T10 narrative writeup + wiki | **First results narrative** |

Checkpoints for Prabin: after T2 (clip list), M1 (table sanity), M2 (kill-test verdict), T10.

---

## Tasks

### T0 — Prerequisite commit (PRABIN, manual — blocks everything)

T4/T7 build on `scripts/eval_ihmc_json.py`, `scripts/ihmc_json_to_stage4_npz.py`, and the
current `scripts/solve_global_trajectory_opt_contactfirst.py` — all **uncommitted on
`p0-grounding`**. Prabin commits the p0-grounding session work (he commits himself, standing
rule), then executor creates branch `gmr-baseline` from that commit. Do not start T4+ from a
dirty tree.

### T1 — Environment + smoke test (~half day)

1. `conda run -n gmr pip install -e /home/ptimilsina/projects/GMR` (pulls mink, mujoco,
   `qpsolvers[proxqp]` per its `setup.py`). If the env fights, make a fresh `gmr-baseline`
   conda env instead — record which in `planLogGMR.md`.
2. Download LAFAN1: Ubisoft repo (github.com/ubisoft/ubisoft-laforge-animation-dataset),
   `lafan1.zip` direct download → unzip BVHs to `data/raw/lafan1/`.
3. Smoke test, headless, on one locomotion clip (any `walk*_subject*.bvh`):
   ```
   conda run -n gmr python /home/ptimilsina/projects/GMR/scripts/bvh_to_robot.py \
       --bvh_file data/raw/lafan1/walk1_subject1.bvh --robot unitree_g1 \
       --save_path outputs/gmr_baseline/pkl/walk1_subject1.pkl --record_video \
       --video_path outputs/gmr_baseline/videos/walk1_subject1.mp4
   ```
   (Check the script's actual flags first; if the viewer demands a display, look for a
   headless/offscreen option or set `MUJOCO_GL=egl`.)
   **GATE:** pkl exists and `pickle.load` gives keys
   `{fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list}` with
   `root_pos.shape==(T,3)`, `dof_pos.shape==(T,29)`.

### T2 — Clip selection (~1 h) → CHECKPOINT

List all BVH basenames; grep for floor-contact candidates (`fall`, `getup`, `ground`,
`crawl`, `push`, case-insensitive). Pick **3 floor-contact + 2 locomotion controls**
(one walk, one dance/fight for a busier control). Eyeball each candidate BVH in the GMR
viewer (or skim joint Z channels) to confirm the human actually goes to the floor.
Record the 5 chosen names + one-line reasons in `planLogGMR.md`. **If LAFAN1 has no true
get-up/crawl clips** (risk noted in `GMR-baseline.md` §6 caveat c): stop, report to Prabin —
fallback is the deferred FBX bridge, a scope change he must approve.

### T3 — E1 baseline batch (~half day)

Run the T1 command for all 5 clips (small bash loop or 5 invocations; no new script needed).
Deliverables: 5 pkls in `outputs/gmr_baseline/pkl/`, 5 videos in
`outputs/gmr_baseline/videos/`. Watch each floor-clip video and write a qualitative failure
catalog in `planLogGMR.md` (what breaks: penetration? limb folding? root pops? tracking
loss?) — this text seeds the motivation-figure caption.

### T4 — E2a: de-Alex the eval core (~1 day, regression-gated)

`evaluate()` in `scripts/eval_ihmc_json.py:92` is ALREADY model-generic — it takes
`(name, qpos, fps, contacts, model, data, mesh_cache, geom_ids, sole_sids, q_lo, q_hi,
mj_joint_names)`. Alex-specifics are only: `_load()` (IHMC JSON parsing), `MODEL_DEFAULT`,
and `SOLE_CORNER_SITES` (imported from `post_process_ground_contactfirst`).

1. New file `scripts/g1/eval_motion.py`: imports `evaluate`, `_build_mesh_cache` etc. FROM
   `eval_ihmc_json.py` (add repo root to `sys.path`; do NOT copy-paste the functions — one
   implementation). CLI: `--model <xml>`, `--fps`, input format flag (`--ihmc-json` or
   `--gmr-pkl`, pkl loader lands in T5). Make `contacts`/`sole_sids` optional: when absent,
   skip stance/slip metrics and print `stance: n/a (no contact flags)` — G1 has no sole
   sites and GMR outputs no contact flags; floor pen + joint limits + velocity spikes carry
   week 1 (self-collision also skipped on G1 this week — the mocap XML's collision pairs are
   not vetted; noted as E4-adjacent work).
   If small refactors inside `eval_ihmc_json.py` are needed to expose pieces (e.g. hoist a
   helper), keep them minimal.
2. **GATE (regression):** run the ORIGINAL `eval_ihmc_json.py` and the new wrapper in
   `--ihmc-json` mode on `data/blender-retargeted/stdSupine.json` — every printed metric
   identical. Record both outputs in `planLogGMR.md`.

### T5 — E2b: G1 loader + M1 motivation table (~1 day) → CHECKPOINT M1

1. `scripts/g1/load_gmr_pkl.py` (module + tiny CLI): pkl → `(qpos (T,36), fps)`.
   `qpos = [root_pos, root_rot xyzw→**wxyz**, dof_pos]`. Sanity assert:
   `abs(norm(quat)-1) < 1e-6` per frame, and FK a single frame in
   `g1_mocap_29dof.xml` — feet near z≈0 on the walk clip's first frame (catches a wrong
   quat order immediately: wrong order = robot pitched/rolled absurdly).
2. Wire into `eval_motion.py --gmr-pkl`. Model: the G1 XML has a `floor` plane geom —
   `evaluate()`'s self/floor logic assumes a floor-free model (`eval_ihmc_json.py:110`
   comment), so EXCLUDE the plane geom from `geom_ids`/mesh-pen the same way worldbody is
   excluded, and compute floor pen vs z=0 as on Alex. Check `mj_joint_names` handling works
   on G1 names.
3. Run on all 5 pkls → `outputs/gmr_baseline/eval_raw.csv` + a markdown table in
   `planLogGMR.md`: per clip — max/median mesh floor pen, joint-limit violations, velocity
   spikes, worst joint velocity.
   **M1 = this table + T3's videos.** Expected shape: locomotion controls near-clean,
   floor clips visibly bad. If floor clips come out CLEAN, that's a finding, not a bug —
   stop and report (it would weaken Option A's motivation; Prabin decides).

### T6 — Polish plumbing: pkl in, pkl out (~half day)

`scripts/g1/polish_gmr_pkl.py` skeleton: load pkl (T5 loader) → qpos → [transform] → write
pkl back in the SAME dict format (wxyz→xyzw on save, mirroring `bvh_to_robot.py:166`) so
GMR's own tooling (`batch_gmr_pkl_to_csv.py` → BeyondMimic, `vis_robot_motion.py`) consumes
polished output unchanged — that's the E5 on-ramp.
**GATE:** identity run (no transform) → load(save(pkl)) round-trips: arrays equal to 1e-12,
and the eval (T5) numbers identical to the raw pkl's.

### T7 — E3a: Stage A smoothing on G1 (~half day + eval)

1. In `polish_gmr_pkl.py`, import `stage_a` from
   `scripts/solve_global_trajectory_opt_contactfirst.py` (`:629`; it's pure qpos-level:
   tridiagonal smoothing of joints + root pos + hemisphere-aligned quat smoothing — no
   model, no Alex assumptions). `q_lo/q_hi` from the G1 model's joint ranges
   (`model.jnt_range` for the 29 actuated joints). `lambda_smooth=20` (30 fps, rule §4 above),
   `lambda_track` at the Alex default (read it from the script's argparse defaults),
   `smooth_root=True`, no `lambda_track_frames` (that machinery is Stage-3-floor-specific).
2. Run on all 5 clips → `*_stageA.pkl`; eval each; before/after columns into the T5 table.
   Expected: velocity spikes / worst-joint-velocity drop (the Luigi result was 5.7× on vMax),
   floor pen roughly unchanged. If smoothing makes floor pen notably WORSE (>1 cm median),
   note it — grounding (T8) is the counterpart, but record the interaction.

### T8 — E3b: grounding on G1 (~1 day)

Reuse `scripts/post_process_ground_contactfirst.py` machinery (mesh cache + `_robot_lowest_z`
already generic). Without contact flags, plant-aware `hybrid` mode is out; use the flag-free
path: run with `--model <g1 xml>` `--mode constant` (and try `perframe` for comparison —
record both, ship the better per eval; expectation from Alex experience: `constant` for
locomotion, and floor clips may genuinely need per-frame — note whichever wins). If the
script hard-requires Alex-specific bits (sole sites, contact NPZ keys), write the thin
adapter in `scripts/g1/` (feed it a minimal NPZ with qpos+fps), do NOT fork the QP/mesh code.
The G1 XML's floor plane must again be excluded from the robot's own lowest-z geom set.
Chain: raw → Stage A → ground → `*_polished.pkl`.
**GATE:** on every clip, polished floor pen ≤ Stage-A floor pen, and no new velocity spikes
introduced by grounding (it's a Z-shift/lift QP — spikes would mean the lift smoothing broke;
`--lift-smooth` default is the knob).

### T9 — M2: before/after table + renders (~half day) → CHECKPOINT M2 (kill-test)

1. Final table (raw / stageA / polished × 5 clips × metrics) → `outputs/gmr_baseline/eval_polish.csv`
   + markdown in `planLogGMR.md`.
2. Side-by-side videos for 1 locomotion + 1 floor clip (GMR's `vis_robot_motion.py` on raw
   vs polished pkl, or the T1 record path re-driven with the polished pkl).
3. **Kill-test read-out (from `GMR-baseline.md` §4 stop-loss):** measurable polish delta on
   either clip class → Option A lives. No delta on both → Option A dies cheaply; E1+E2 (M1)
   still stand for Option C's G1 amendment. Either way: report honestly, full distributions,
   no cherry-picking (overclaim discipline, `GMR-baseline.md` §3 risks).

### T10 — First results narrative (~half day)

1. `GMR-baseline-results.md` (repo root): (a) setup one-pager — GMR version/commit, clip
   list, metric definitions; (b) M1 motivation table + failure catalog; (c) M2 polish table;
   (d) the narrative paragraph: *"GMR, out of the box, on LAFAN1's own floor-contact clips,
   produces [X]; our reference-free eval quantifies it at [Y]; a robot-agnostic Stage-A +
   grounding polish — ported to G1 in [N] lines, no new solver machinery — improves [Z]"*
   (or the honest negative). (e) recommended next step: E4 (Stage B on G1) vs E5 (BeyondMimic
   scoping) vs stop.
2. Wiki: new `wiki/experiments/gmr-baseline-week1.md` (tables + verdict), one line in
   `wiki/log.md`, link from `wiki/index.md`. Update `wiki/concepts/related-work.md` GMR entry
   with what running it actually showed.
3. Do NOT update `SESSION_HANDOFF.md` (Prabin instructs when).

---

## Risks / early-stop conditions

- **T2: LAFAN1 lacks real floor clips** → stop at checkpoint; FBX-bridge fallback needs
  Prabin's approval (it's the deferred item below, ~days more).
- **T5: GMR is fine on floor clips** → motivation weakens; stop and report before T6.
- **T7/T8: no polish delta anywhere** → Option A stop-loss fires at M2; M1 still deliverable.
- **GMR paper's exact eval-subset clip list** unconfirmed downloadable (`GMR-baseline.md`
  §6 caveat) — irrelevant this week (we pick our own 5), flag again at E5 time.
- Headless rendering on this machine (`MUJOCO_GL=egl` usually suffices; else videos are
  Day-2 nice-to-have, metrics are the deliverable).

## Deferred (explicitly out of week 1)

- **Our FBX floor clips → GMR** (E1's original form): adapter from our canonical-human NPZ
  to GMR's per-frame `{body_name: (pos, quat)}` dict + an ik_config for our skeleton — the
  right base is GMR's `data_loader.py` + our stage-1 outputs. ~days. Week 2 candidate.
- **Stage B contact QP on G1** (E4): needs ROLE_TO_G1_BODY map, foot support faces, contact
  detection (our height/velocity gates port). Only if M2 shows headroom.
- **Self-collision eval on G1**: vet `g1_custom_collision_29dof.urdf` → MJCF path.
- **BeyondMimic** (E5): start scoping conversation at M2 if positive (GPU ask = mentor Q4).

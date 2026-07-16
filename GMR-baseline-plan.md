# GMR-baseline-plan — execution plan

**Status 2026-07-15: Week 1 (T0–T10) DONE, Week 2 (W2-T1..T7) DONE.** Results:
`GMR-baseline-results.md`. Full trail: `planLogGMR.md` (`## W2-Tn` headings). Wiki:
`wiki/experiments/gmr-baseline-week2.md`. Week-1 committed (`0f3f157`); Week-2 work is uncommitted
(Prabin commits himself, per standing rule). **Week-2 verdict in one line**: fairness addendum
(W2-T1) and slip-claim close-out (W2-T2) both landed clean; self-collision vetting (W2-T6) passed;
E4b multi-surface anchoring (W2-T3/T4/T5) and contact-aware grounding (W2-T7) both hit real,
reported negative checkpoints — logged honestly, nothing pushed further unilaterally past them.
**Current plan: the SPRINT section below** (Humanoids 2026, 9 days, Prabin 2026-07-15): the 2×2
comparison — GMR-as-published vs OURS-on-G1, each with/without polish (`GMR-baseline.md` §7.4) —
S1 full-corpus kinematic sweep, S2 our-pipeline-on-G1 port (which IS the contact-first solving
W2-T5 called for), S3 BeyondMimic policy runs (GPU ask day 1), S4 paper assembly. S1/S2 start
immediately, no external dependency.

**S1 status (2026-07-16): DONE (S1-T1..T4 + reclassification addendum), CHECKPOINT.** 77/77 clips,
all 3 variants (raw/gmrfix/polished), 0 failures across both batches; regression gate against the
5 week-1/2 clips passes exactly. Class split: **34 floor-class / 43 locomotion-class**, by
sustained (>=1s) multi-surface human contact (`human_contacts_lafan1.py`'s detector, reused
unchanged, run over all 77 clips) — supersedes the initial hip-Z-p5<0.3 split (20/57), which
undercounted floor-contact clips by 14 (brief hand/knee contact while the hip stays up). This is
the split to use in the paper table. Table-I clip-name mapping: checked, not found (paper website
+ GMR repo configs both lack it) — unmapped, author-email fallback is Prabin's call. Full tables +
numbers: `planLogGMR.md` `## S1-T4` and `## S1-T4 (addendum)`. **S2 was explicitly not touched
this pass** (still parked at its own M4 checkpoint, S2-T5). S3/S4 not started.

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

---
---

# WEEK 2 — E1b fairness addendum + E4b multi-surface contact anchoring

**Written 2026-07-15 after the E4 park + a full cross-read of `RetargetMatters.pdf`. Rationale for
every task: `GMR-baseline.md` §7 (read §7.2/§7.3 before starting — do not re-derive them).**

## Ground rules (week-1 rules still apply, plus)

1. Week-1 ground rules 1–8 all still hold (GMR clone read-only, xyzw↔wxyz at the pkl boundary
   only, `sys.path` gets `<repo_root>/scripts` never bare repo root, regression-gate every
   refactor, outputs to `outputs/gmr_baseline/`, log to `planLogGMR.md` under `## W2-Tn` headings).
2. Read `planLogGMR.md`'s "E4" section BEFORE touching `scripts/g1/stage_b_g1.py` — it documents
   the contact-detection bug (speed-gate/centroid artifact), the fix layering (coarse zone flags
   from us, stillness sub-segmentation inside `_compute_anchors`), and which imports are verified
   Alex-global-free. Do not re-learn those lessons.
3. The E4 discipline stands: import Alex core functions UNCHANGED; fork only thin G1 glue inside
   `scripts/g1/`. If a change seems to require editing an imported core function, stop and flag —
   that's a scope decision for Prabin.
4. Every claim that could appear in a paper needs an independent cross-check (the E4 lesson:
   stage_b's self-reported slip number is still unverified — that's W2-T2).
5. Time-box each task to its estimate ×2; past that, log where it stands and move on (or
   checkpoint if it blocks downstream tasks).

## Task order and why

T1 (fairness) and T2 (slip close-out) are integrity items — do them first, they're cheap and they
protect everything already shipped. T3 is the kill-test gate for the whole E4b idea — run it before
building any anchoring machinery. T4–T5 are the build. T6–T7 are quality unlocks. T8 wraps.

### W2-T1 — E1b: fair-baseline addendum + floating metric (~1 day)

Why: `GMR-baseline.md` §7.2 item 1. Our E1 numbers are GMR's shipped default; the paper describes
a min-height fix their code has hard-disabled (`HEIGHT_ADJUST = False`,
`/home/ptimilsina/projects/GMR/scripts/bvh_to_robot_dataset.py:127-142`). Reviewers will ask.

1. **Floating metric first** (eval must see what the height fix trades into). In
   `scripts/g1/eval_motion.py` (or a helper it imports — keep `evaluate()` shared with Alex intact
   and regression-gated): per frame, compute the robot's lowest mesh point z (machinery already
   exists for penetration — same quantity, opposite sign). New columns: `floatMax` = max over
   frames of lowest-point z, `float%` = fraction of frames with lowest point > 0.5cm above floor.
   **GATE (regression):** re-run the week-1 T4 gate (`--ihmc-json` on
   `data/blender-retargeted/standSupine.json`) — every EXISTING metric byte-identical; new columns
   additive only. Sanity: `walk1_subject1` raw should read float% ≈ 0 (a walking robot almost
   always has one foot at the floor).
2. **Replicate their height fix** as a post-process flag (new small function in
   `scripts/g1/polish_gmr_pkl.py`, e.g. `--gmr-heightfix`): FK all frames in the G1 mocap XML
   (plain mujoco, no torch — their torch `KinematicsModel` is just batched FK), take
   `min over all frames, all bodies of body-ORIGIN z` (xpos, NOT mesh — replicate their method
   faithfully, mesh-blindness included; that's the point), subtract it from `root_pos[:,2]`
   clip-globally, `ground_offset = 0.0` as in their code. Do NOT also run our grounding in the
   same invocation.
3. Produce `*_gmrfix.pkl` for all 5 clips; eval; add a "GMR+heightfix" column to BOTH tables
   (motivation and polish) in `planLogGMR.md` under `## W2-T1`.
4. **Expected** (from `GMR-baseline.md` §7.2): floor clips improve partially but keep multi-cm
   mesh penetration (body origins sit above the mesh bottom) AND gain floating elsewhere
   (single worst-frame shift lifts the whole clip). If instead heightfix makes floor clips
   near-clean (<2cm max pen, low float), the motivation table weakens materially —
   **CHECKPOINT: stop and report to Prabin before any further task.** Either way update
   `GMR-baseline-results.md`'s motivation section with the new column + one honest paragraph.

### W2-T2 — Close E4's unverified slip claim (~half day)

Why: §7.2 item 4. The walk1_subject1 "25% slip reduction" is stage_b's own `_contact_slip_stats`
on its own flags — circular enough to need an outside check before it's paper-facing.

1. Write `scripts/g1/check_slip_independent.py`: load `walk1_subject1` polished and
   polished+StageB pkls, load the SAME contact-zone flags used in the E4 run (re-derive with
   `detect_g1_foot_contacts` — deterministic), FK both motions with plain mujoco (do NOT import
   any stage_b internals — the point is an independent code path), segment planted runs with the
   same stillness convention (body speed < 0.05 m/s within zone frames), and compute per-run XY
   drift of each foot body origin. Report per-foot mean/max drift, both motions.
2. **GATE:** direction must confirm (StageB drift < polished drift on walk1). Magnitude may differ
   from 25% (different segmentation details are fine — log both numbers side by side). If
   direction does NOT confirm, the E4 positive result is dead — log it, update
   `GMR-baseline-results.md`'s E4 mention, and tell Prabin at the next checkpoint (not a stop —
   W2-T3+ doesn't depend on this).

### W2-T3 — E4b-a: human-side multi-surface contact labels (~1 day) → CHECKPOINT (kill-test #1)

Why: §7.2 item 2. This is the pivot: detect contact on the HUMAN source (uncorrupted, knows which
body parts bear weight), not on the robot output. It is also the cheapest possible test of whether
E4b can work at all — run it BEFORE building any robot-side machinery.

1. Read `scripts/contact_labels.py` (Alex stage 2.5) first — reuse its conventions (hysteresis,
   min-run length, speed windows) wherever they translate; log which you kept/dropped.
2. New `scripts/g1/human_contacts_lafan1.py`: GMR's `load_bvh_file` → per-frame
   `{body_name: (pos, quat wxyz)}` dicts. Enumerate the LAFAN1 skeleton's body names from an
   actual loaded frame (log the list). Landmarks to label: feet/toes, hands, knees, elbows/
   forearms, hips/pelvis, spine/torso, head (head so prone clips can be sanity-read; it need not
   become an anchor).
3. **Contact-zone rule: height gate ONLY** (the E4 lesson — stillness sub-segmentation belongs to
   `_compute_anchors`, not the detector). Per-landmark height thresholds, calibrated from the
   data, not guessed: plot/log each landmark's z distribution on `walk1_subject1` (contact
   negative control for hands/knees, positive for feet) and on the 3 floor clips. Starting points:
   feet 5cm (E4's calibrated value), hands/knees/elbows ~8cm, pelvis/torso ~15cm (they're thick) —
   ADJUST from the distributions and log final values + a one-line justification each. LAFAN1's
   floor is z≈0 (verified week 1: hipZ min 0.028 on the deepest fall clip).
4. Output: per-clip NPZ (`outputs/gmr_baseline/human_contacts/{clip}.npz`) with per-landmark
   boolean (T,) arrays + the threshold dict, plus a summary table (clip × landmark → zone %)
   in `planLogGMR.md` under `## W2-T3`.
5. **KILL-TEST #1 (checkpoint, report to Prabin either way):**
   - Floor clips MUST show sustained (multi-second) non-foot contact zones — pelvis/torso/hands
     during lying phases. Sanity: `walk1_subject1` feet ~40–60% each alternating, everything else
     ≈0%; `fallAndGetUp2_subject2` should show pelvis/torso zones exactly in its lying window
     (pelvis-z minimum around frame 356, week-1 catalog).
   - If floor clips show NO usable non-foot zones, E4b's premise is wrong — STOP, report. (Do not
     soften thresholds until it passes; calibrate from distributions once, honestly.)

### W2-T4 — E4b-b: G1 multi-surface role map + support points (~1 day)

Why: anchors need to know WHICH robot body carries each human contact and WHERE that body's
support surface is.

1. Enumerate the G1 mocap XML's body list (log it). Extend `stage_b_g1.py`'s `_resolve_g1_feet`
   into `ROLE_TO_G1_BODY` covering: feet → `left/right_ankle_roll_link` (exists, E4), hands →
   the wrist/hand bodies (discover the names; GMR's `unitree_g1` is the no-hands 29-DoF model —
   the distal wrist link is the hand-side support body), knees → knee links, pelvis → pelvis,
   torso → the torso link. Log the chosen mapping with body names verbatim.
2. Support-point helper `support_z(model, data, body_id)` → lowest z of that body's geoms at the
   current frame's pose. Use the same mesh-vertex machinery the eval already uses
   (`_build_mesh_cache` path) — orientation-aware, mesh-exact where meshes exist, analytic for
   sphere/capsule/box primitives. This is glue, lives in `scripts/g1/`.
3. **GATE:** on `walk1_subject1` frame 0 (standing), `support_z` of each foot ≈ 0 (±1cm), of
   knees/hands/pelvis clearly > 10cm. On `fallAndGetUp2_subject2` frame 356 (lying), pelvis/torso
   `support_z` near 0.

### W2-T5 — E4b-c: multi-surface Stage B with pull-to-floor anchors (~1–2 days) → CHECKPOINT M3 (kill-test #2)

Why: §7.2 items 2+3. Two changes vs E4: contact zones come from the HUMAN labels (T3, transferred
via T4's role map — same 30fps, no resampling), and floor-contact anchors PULL the support body to
the floor instead of holding it wherever the corrupted motion put it.

1. In `stage_b_g1.py` (glue only, imported core stays unchanged): for each robot body with an
   active human-contact zone, run the existing `_compute_anchors` stillness sub-segmentation as in
   E4. For each resulting anchor, adjust its z: `anchor_z = body_z - support_z(frame)` so the
   body's support point sits AT z=0 during the anchored run. XY stays from the motion (anchoring
   semantics unchanged). If `_compute_anchors`' output shape makes per-anchor z-adjustment awkward
   from glue, flag before hacking the core.
2. **Scope**: feet + hands + knees anchored by default. Pelvis/torso behind `--anchor-trunk`
   (default OFF) — pinning trunk-adjacent bodies starved tracking in the retired hierarchical-v1
   experiment (`wiki/experiments/retired-approaches.md`); run once with and once without, compare,
   let the numbers decide. Self-collision stays OFF (`lambda_coll=0.0`) until W2-T6 lands.
3. Run the 5-clip corpus on top of polished (Stage A + ground constant), self-tracking-target
   setup exactly as E4.
4. **GATES:**
   - `eval_motion.py` (independent): zero regressions on any clip, any metric (floorPen, pen%,
     vMax, spikes, float from T1).
   - Floor clips: floorPen/pen% strictly improve vs polished (this is where E4 flat-lined — the
     whole point of E4b).
   - Log per-body planted-frame counts + `|dQ|max` per clip (E4's zero-work symptom must NOT
     recur on floor clips; if it does with T3's labels active, the transfer wiring is broken —
     debug before concluding anything).
   - **KILL-TEST #2 (visual):** extract `fallAndGetUp2_subject2` frame 356 from raw / polished /
     +StageB-multisurface renders (`render_gmr_pkl.py`). Question: does at least one labeled
     support body now visibly bear on the floor (feet down, or hands/pelvis in contact) instead of
     the corpse pose? Include the three frames side by side in the log.
5. **CHECKPOINT M3, report to Prabin either way:** if anchors engage (nonzero planted frames,
   nonzero |dQ|) but the pose does NOT visibly improve, then anchoring-on-top-of-polish is not the
   corpse-pose fix and the honest conclusion is that this class needs contact-first SOLVING (our
   Stage-3 analog on G1, a Week-3+ scope decision) — report, don't push further unilaterally.

### W2-T6 — Self-collision vetting on G1 (~1 day, parallelizable after T1)

Why: §7.2 item 5 — blocks one of the paper's own three artifact axes AND `lambda_coll` in Stage B.

1. Inspect `/home/ptimilsina/projects/GMR/assets/unitree_g1/g1_custom_collision_29dof.urdf` (GMR
   ships it precisely because the visual model's collision geometry is unusable). Either convert
   to MJCF or graft its collision primitives / exclusion pairs onto a copy of the mocap XML —
   **new file under `scripts/g1/` or `outputs/`, never edit the GMR clone**.
2. **GATE:** `walk1_subject1` raw re-evaluated with the vetted collision model reads < 1%
   self-collision incidence (vs 18.2% noise today). If the URDF route fails, fallback: hand-build
   an exclusion-pair list on the mocap XML (adjacent-link pairs + torso/thigh at hip flexion —
   Alex's model shows the pattern) and gate the same way.
3. If the gate passes: re-run the M1/M2 tables' coll% column with the vetted model (append, don't
   overwrite, in the log), and re-run W2-T5's best config with `lambda_coll` at Alex's default —
   compare, keep the better per eval.

### W2-T7 — Contact-aware grounding (~half day, after T3)

Why: T3's labels unlock the grounding modes that were locked out in week 1 (`hybrid` /
`constant-contact` need contact flags). Constant-mode won by default, not by comparison.

1. Feed T3's labels (mapped to robot bodies via T4) into `post_process_ground_contactfirst.py`'s
   flag-driven modes through the existing minimal-NPZ adapter in `polish_gmr_pkl.py` (extend the
   NPZ with the contact keys the script expects — read its `main()` for exact key names).
2. Compare vs `constant` on all 5 clips: floorPen, float (T1's metric), and the T8 bobbing check
   (root-Z vz_max). **Ship whichever wins; log the table.** Expectation: hybrid ≥ constant on
   floor clips (phase-aware reference), equal on locomotion.

### W2-T8 — Results refresh + wiki (~half day) → CHECKPOINT

1. Update `GMR-baseline-results.md`: heightfix baseline column + honest paragraph (T1), corrected/
   confirmed slip claim (T2), E4b section (T3–T5 verdicts, the three-frame visual), self-collision
   status (T6), grounding-mode decision (T7). Same overclaim discipline as week 1.
2. Wiki: update `wiki/experiments/gmr-baseline-week1.md` (or add `-week2.md`), one line in
   `wiki/log.md`, keep `wiki/index.md` current.
3. Do NOT update `SESSION_HANDOFF.md` (Prabin instructs when). Do NOT commit (Prabin commits).

## Week-2 risks / early stops

- **T1 flips the motivation** (heightfix near-cleans floor clips) → stop, checkpoint. Everything
  downstream still runs, but the paper framing changes and Prabin re-scopes.
- **T3 kill-test fails** (no non-foot human contact zones on floor clips) → E4b dead as designed;
  stop, checkpoint. T1/T2/T6 results still stand alone.
- **T5 anchors engage but pose doesn't improve** → anchoring is the wrong mechanism class;
  checkpoint M3 decides between contact-first solving on G1 (heavy, Week 3+) and stopping at
  whole-clip polish + honest boundary for the paper.
- **T6 URDF conversion fights back** → fallback exclusion list; if that also fails the <1% gate,
  self-collision stays un-claimable on G1 — report, don't fudge.

---

# SPRINT — Humanoids 2026, 9 days (2026-07-15, supersedes the old "Week 3 outline")

Target: the full **2×2 comparison** (`GMR-baseline.md` §7.4 — GMR+heightfix / polish(GMR) / OURS /
polish(OURS), both motion classes, kinematic always + policy where GPUs land) and a paper draft.
Timescale calibration: week-1's "one week" plan executed in ~1 hour of agent time — these "days"
are generous; the real serialization points are GPU availability (E5) and Prabin's checkpoints.
Everything not GPU-blocked runs immediately and in parallel where independent.

## Sprint ground rules (on top of Week-1/2 rules, which all still hold)

1. Log under `## S<n>-T<m>` headings in `planLogGMR.md`. GMR clone stays read-only. Outputs under
   `outputs/gmr_baseline/sprint/`.
2. **Every batch script must be resumable**: skip outputs that already exist (check file presence
   before computing), so a crashed/interrupted 77-clip run re-launches and continues. Run long
   batches in the background; end each batch with a failure report (clip → stage → error one-liner)
   — a clip that fails does NOT block the batch, it gets logged and skipped.
3. **The 2×2's polish column is Stage A + `constant` grounding applied to the RAW retarget** —
   NOT stacked on heightfix (our constant grounding and their heightfix are both global Z
   calibrations; stacking double-shifts). The heightfix column exists only as the polish-OFF
   baseline cell. Never produce a `gmrfix+polish` artifact.
4. Time-boxes are stated per task. Hitting one = log where it stands, apply the task's named
   fallback, move on. Never silently extend.

## S1 — E6 kinematic sweep, all of LAFAN1 (start immediately)

### S1-T1 — batch retarget, 77 clips (~hours of compute, background)
Loop `scripts/g1/gmr_headless_retarget.py` (no `--video_path` — videos only for paper-selected
clips later) over every BVH in `data/raw/lafan1/`. Resumable per ground rule 2. Before the batch,
add a `--save-human-targets <npz>` option to `gmr_headless_retarget.py` (our-repo file): persist
`retargeter.scaled_human_data` per frame — INSPECT its actual structure at runtime first (expected:
dict body_name → (pos, quat)); save positions at minimum. **GATE**: one clip (walk1_subject1)
retargets bit-identically to its existing week-1 pkl (the retargeter is deterministic — if this
fails, something in the env changed; stop and report) and its targets NPZ has T matching the pkl.

### S1-T2 — heightfix + polish variants, 77 clips
Per clip, produce: `<clip>_gmrfix.pkl` (heightfix on raw) and `<clip>_polished.pkl` (Stage A +
ground-constant on raw — ground rule 3). Both via existing `polish_gmr_pkl.py` flags. Resumable.

### S1-T3 — eval + faithfulness, 77 clips × 3 variants
1. `eval_motion.py` metrics (incl. float) for raw / gmrfix / polished → one CSV.
2. Self-collision via the vetted model (W2-T6, `outputs/gmr_baseline/g1_collision/
   g1_collision_vetted.urdf`) — separate pass (joint order verified identical, qpos feeds
   directly), columns appended to the same CSV.
3. **Faithfulness**: find GMR's LAFAN1→G1 ik_config in the clone
   (`general_motion_retargeting/ik_configs/`, the bvh/lafan1→g1 file) — it defines the
   human-body→robot-body correspondence GMR itself optimized. Per mapped pair: mean/max position
   error of the ROBOT body (FK) vs the scaled-human target (S1-T1's NPZ), for raw and for
   polished. Faithfulness guard = polished error minus raw error; report the full distribution.
   **Fallback** (if the scaled-human-target hook proves unreliable): faithfulness proxy =
   polished-vs-raw robot body-position deviation (drift from THEIR solution) — clearly labeled a
   proxy in the log, still a valid "polish doesn't wander" guard.
4. **GATE (regression)**: the 5 week-1/2 clips' raw + polished rows must reproduce the existing
   numbers exactly (same code paths — any drift means a bug introduced this sprint; stop).

### S1-T4 — table + Table-I mapping (parallel, ~1h)
Aggregate by class (locomotion vs floor-contact, split by the T2 hipZ screening convention:
p5 < 0.3 → floor class). Markdown summary in the log. Table-I mapping best-effort: paper website
(jaraujo98.github.io/retargeting_matters) → GMR/BeyondMimic repo configs → give up and mark
unmapped (author email is Prabin's call, not the executor's). Unmapped motions only lose the
published-number annotation. **CHECKPOINT: S1 table to Prabin.**

## S2 — E7: OURS on G1, through and through (start immediately, the big build)

The bottom row of the 2×2, and simultaneously the contact-first solving W2-T5 called for. Before
starting read: `wiki/concepts/pipeline.md`, `wiki/concepts/contact-first-ik.md`,
`wiki/concepts/morphology-scaling.md`, `wiki/concepts/alex-model.md` (the ankle treatment to
analogize), and `SESSION_HANDOFF.md`'s p0-grounding section (Stage-3 DLS instability — a shipped
finding, not folklore).

**Embedded footguns — do NOT rediscover these (each cost a past session):**
- Stage-3 floor repulsion `--floor-weight` at just 1.5-2× the default 10 DIVERGES on get-up clips
  (pre-existing DLS instability, planLog.md feasibility-first T3). Never crank floor weight to fix
  penetration; that lever is at its edge already.
- Floor-as-hard-tier is a CONFIRMED dead end (44m divergence, hierarchical-v1 H2). Soft only.
- Per-frame retry/perturbation wrappers around the DLS solve diverge (feasibility-first, all
  tested configs). Don't add retry loops; warm-start from the previous frame as Stage 3 already
  does, and if a clip diverges, log it as a per-clip failure and move on — a diverging clip is a
  RESULT (goes in the honest-failures table), not a bug to fix mid-sprint.

### S2-T1 — canonical-human adapter (~half day, time-box 1 day)
GMR's `load_bvh_file` per-frame `{bone: [pos, quat]}` → our canonical-human NPZ. First READ one
real Stage-3 input NPZ (under `data/`, schema in `wiki/concepts/pipeline.md`) and list its exact
keys/role names — build to that schema, don't guess it. Map LAFAN1 bones (enumerated in W2-T3:
Hips, Spine/1/2, Neck, Head, L/R UpLeg/Leg/Foot/Toe, L/R Shoulder/Arm/ForeArm/Hand) → canonical
roles. Known gap: no `*_hand_middle` analog in LAFAN1 — synthesize by extrapolating the
ForeArm→Hand direction a few cm, OR omit (contact_labels filters missing markers gracefully);
try omit first, log which. **GATE**: adapted walk1_subject1 — feet near z≈0, marker heights
plausible (compare against W2-T3's height table), all roles Stage 2.5/3 hard-require present.

### S2-T2 — Stage 2.5 on adapted clips (~half day)
Run `ground_canonical_human.py` + `contact_labels.py` detection on the adapted NPZs (5-clip
corpus first). **GATE**: contact labels directionally agree with W2-T3's zone table (different
thresholds/markers, so order-of-magnitude agreement per landmark class, not equality — feet ~50%
on walk, near-zero hands on controls, sustained multi-surface on floor clips).

### S2-T3 — Stage 3 on G1 (the genuinely new piece; time-box 2 days before fallback)
G1 analog of `solve_fbx_canonical_alex_contactfirst.py` — fork GLUE (role→body map, support
faces, joint indexing, model paths), import solver math where it's already argument-driven.
Reuse: W2-T4's `ROLE_TO_G1_BODY` + `support_z`, sole-corner sphere markers as the foot support
face, `rubber_hand` as a point-contact hand (no fist face — G1 has no fist; document the
simplification). Morphology scaling per `wiki/concepts/morphology-scaling.md` (motion DELTAS from
rest, never absolute positions — conventions section of CLAUDE.md) with G1's rest pose from the
mocap XML. **Robot-specific adaptations, deliberate and documented for the paper** ("in the
retargeting itself we changed X"): dump G1's `jnt_range` for ankle/waist/wrist, compare against
the motion's demanded ranges on the 5-clip corpus, apply the Alex-ankle-style treatment where a
joint pins (read `wiki/concepts/alex-model.md` first). Order of attack: `walk1_subject1` solving
cleanly end-to-end BEFORE any floor clip; then the 3 floor clips.
**GATE (per clip)**: tracking error sane (compare walk1 against GMR's own output visually +
numerically), no divergence, eval metrics computed.
**FALLBACK A** (time-box hit or Stage-3 math fights G1): "ours" = our contact-first TARGET
corrections (grounded canonical human + our contact-corrected targets) fed through GMR's own mink
IK via a custom ik_config — an honest hybrid row ("our contact-first front-end + their solver"),
clearly labeled, still a novel method row. **FALLBACK B** (A also stalls): OURS row ships on the
5-clip corpus only, not all 77 — smaller table, same claims structure.

### S2-T4 — polish(OURS) + eval (~fast, reuses everything)
`polish_gmr_pkl.py` chain on S2-T3 output (save OURS output in the same pkl dict format so every
existing tool works on it), eval both cells, render frame 356. **CHECKPOINT M4 (the sprint's
biggest): does OURS produce visible weight-bearing contact at frame 356 where anchoring couldn't?
Full 4-row × 5-clip table (GMR+heightfix / polish(GMR) / OURS / polish(OURS)) to Prabin.** Clears
→ S2-T5 broadens to the full corpus (resumable batch). Fails → the paper's OURS story is the
5-clip honest version; do not broaden.

## S3 — E5: BeyondMimic policy runs (GPU-gated — unblock day 1)

**Run budget (hardware: Prabin's RTX 5080, 16GB, single GPU).** One run = one trained policy for
one (clip × retarget-variant) pair — every comparison cell costs a full training run. Full wishlist
= 10-16 policies; expected ~2-6h each at ~4096 IsaacLab envs (4090-class norm — NOT verified for
their exact config, so **the first Dance 5 run is the calibration run: measure it, rescale the
whole queue from that number before promising anything**). Serial total ≈ 2-3 days of continuous
training → queue must run 24/7 from the moment S3-T1's gate passes. Evals (sim 100 / sim-dr 4096
rollouts, parallel envs) are minutes, not hours — training dominates. **Priority queue if hours
run short** (minimum viable ≈ 8 runs ≈ ~1.5 days): (1) Dance 5 × 2 variants, (2) one floor clip ×
all 4 cells of the 2×2 (needs S2's OURS output — sequence accordingly), (3) remaining their-turf
motions × 2. Skip sim2sim entirely (their ROS setup, serial rollouts — poor hours-to-claim ratio;
sim + sim-dr match most of Table I).
**5080-specific checks in S3-T1, before anything else**: (a) Blackwell needs a recent IsaacSim
(4.5+) + current drivers — the most likely setup fight; (b) 16GB VRAM: 4096 envs should fit
(typical 6-12GB), on OOM drop to 2048 and expect ~1.5-2× wall-clock per run — rescale the queue.

### S3-T1 — day 1, in parallel with everything: GPU ask (Prabin/mentor) + environment prep
Clone BeyondMimic, attempt setup as far as hardware allows. **Named foreseeable blocker — check
FIRST**: `batch_gmr_pkl_to_csv.py` may require `local_body_pos`/`link_body_list`, which our pkls
save as `None` (so does GMR's own single-clip script; only their dataset script fills them). Read
the converter's source; if required, compute via identity-root FK per frame (mirror
`bvh_to_robot_dataset.py:117-125`, plain mujoco) and fill on save — small patch to
`polish_gmr_pkl.py`'s `save_gmr_pkl`, verify walk1 round-trips. **GATE**: one pkl → CSV conversion
succeeds and the CSV's shape/columns match what a GMR-produced pkl yields.

### S3-T2 — Dance 5 (first GPU hours)
Their own named GMR failure (sudden waist jumps, Fig. 3c; success 92.75 sim / 51 sim2sim).
Requires its Table-I mapping (S1-T4) — if unmapped, ask Prabin before burning GPU on a guess.
Train raw-vs-polished, their protocol (sim 100 rollouts, sim-dr 4096 — budget accordingly; skip
sim2sim unless time allows, it needs their ROS setup). Compare against the published row.

### S3-T3 — 2×2 policy subset (as GPU allows)
2-3 their-turf motions + 1-2 floor clips across the 2×2's four cells. Floor-clip flag: their
success definition (anchor-body height/orientation deviation) may need adaptation for lying
motions — surface the issue in the log + paper, never silently redefine. Every result lands next
to the published Table I/II number where mapped.

## S4 — paper assembly (last 2-3 days, overlaps S3's training wall-clock)

Humanoids 2026 format. Draft the skeleton EARLY (when S1's table lands, not after S3): intro/
related work can cite week-1/2 numbers already logged. Content: the 2×2 tables (S1+S2), policy
deltas (S3 — whatever lands by deadline; the paper states plainly which cells are kinematic-only),
motivation figures (raw vs heightfix vs polish, frame-356 series, float% story), the
negative-results boundary (E4/E4b: anchoring cannot fix per-limb pose; OURS is the mechanism that
can — or honestly didn't, per M4), robot-specific adaptations as method contributions. Overclaim
discipline: full distributions, no cherry-picking, every number traceable to a `planLogGMR.md`
entry.

**Checkpoints**: S1-T4 (full-corpus table), S2-T4/M4 (the 4-row table + frame-356 visual — the
sprint's pivotal result), S3-T2 (before any paper claim built on policy numbers). Early-stop:
none of S1/S2/S3 blocks the others — a stalled track narrows the paper, never kills it.

---

# NEXT SESSION (2026-07-17, Sonnet executor) — S2-T8 then S2-T9

Context: N1 is DONE and root-caused far deeper than the original ladder — the elbow/self-collision
hunt led through 5 real bugs, ending at the kinematically-inconsistent per-role leg scaling, now
FIXED with GMR's own grouped constants (full trail: planLogGMR.md `## S2-T6` entries + `## S2-T7`).
Read `## S2-T7` FIRST — it defines the current state and the open residual. N2 is DONE up to two
blockers that are Prabin's decisions (IsaacLab version mismatch, WandB setup) — do NOT act on N2.
The running IsaacLab GPU job is Prabin's — do NOT stop it.

Standing rules that bit us this session (violating these silently corrupts numbers):
- ALWAYS pass `floor_gid` to `_collision_stats` on the combined model from `g1_model_setup.py`
  (it has an injected floor; `floor_gid=None` leaks floor contacts into self-collision).
- The GMR baseline column is GMR+heightfix (`outputs/gmr_baseline/pkl_w2/{clip}_gmrfix.pkl`),
  their own published method — NOT raw shipped output.
- float% is a diagnostic, never a scoreboard. The discriminating metric is held-frame
  support_z. srcDev, never "faithfulness."
- Prabin commits git himself. Never touch `/home/ptimilsina/projects/GMR`.

## S2-T8 — FIRST: polish + full comparison on the gmrscale outputs (mechanical)

The 4 RAW `*_ours_gmrscale.npz` clips (validated in `## S2-T7`) have not been polished. Per clip
(walk1_subject1, fallAndGetUp1_subject1, fallAndGetUp2_subject2, ground1_subject1):
1. `conda run -n gmr python scripts/g1/polish_ours_g1.py --in outputs/gmr_baseline/sprint/ours_g1/{clip}_ours_gmrscale.npz --out .../{clip}_ours_gmrscale_stageA.npz` (NO `--ground` — grounding-off is the shipped default, see script docstring).
2. `conda run -n gmr python scripts/g1/ground_ours_contact_aware.py --in .../{clip}_ours_gmrscale_stageA.npz --canonical outputs/gmr_baseline/sprint/canonical_human/{canon}.npz --out .../{clip}_ours_gmrscale_ctground.npz --smooth 1.0` (canon names: walk1 uses `walk1_subject1_v3_grounded.npz`, others `{clip}_grounded.npz`). smooth=1.0 was tuned on walk1 ONLY — for the other 3, check the printed lift stats + root-Z velocity sanity; held-frame lift must be ~0 by the hard cap.
3. Run `scripts/g1/eval_g1_gmrscale_variants.py` (committed this session, paths pre-set for the
   gmrscale variants): whole-clip floorPen/pen%/coll% + held-frame support_z for
   GMR+heightfix / GMR+ourpolish / OURS-gmrscale raw / +StageA / +StageA+ctground, all 4 clips.
4. Verify held-frame support_z survives BOTH polish steps essentially unchanged from the raw
   numbers in `## S2-T7`'s table (StageA is pure smoothing; ctground has the hard cap). Any
   held-frame regression >0.5cm median = stop and report, don't tune around it.
Log under `## S2-T8` in planLogGMR.md; update GMR-baseline-results.md's S2 table with the
before/after and the correct 4-column framing.

## S2-T9 — SECOND: the swing-frame tracking residual (diagnostic ladder, time-boxed)

The open bug from `## S2-T7`: walk1 frame 598, left foot — target verifiably +1.5cm ABOVE floor,
achieved -8.9cm (err -10.4cm). Already ruled out by direct sweeps (do NOT re-run): iteration
count (30/100/300/1000 identical), self-collision competition (coll_weight 20/5/1/0 identical).
This residual dominates aggregate floorPen now (fallAndGetUp2 raw hit 39.7cm). Ladder, in order,
stop when root-caused:
1. At frame 598, dump target-vs-achieved as FULL 3D error vectors for ALL 15 position roles (not
   just the leg). Splits the world: error localized to swing foot vs distributed (root/pelvis
   tracking error dragging everything); Z-only vs 3D (horizontally-unreachable target where Z
   eats the compromise).
2. Ankle-orientation conflict test (top suspect): S2-T6 added IDENTITY orientation targets for
   all 7 ori roles including ankles — during swing the human foot pitches, a flat-foot identity
   ori target may fight the position target. Re-solve walk1 with ankle ori weights 0 (or the
   ori-weight-scale hook if present in `solve_lafan1_canonical_g1_contactfirst.py`), compare
   frame-598 error + whole-clip floorPen + held support_z.
3. Pull-to-floor leak test: `--no-pull-to-floor` run, compare frame-598 error (checks the EMA/
   zone-envelope leaking outside contact zones).
4. Find fallAndGetUp2's 39.7cm worst frame (body + frame id), dump target vs achieved there —
   same class as frame 598 or a different mechanism? (It regressed 20.4→39.7cm with the scale
   fix, so the scale change interacts with it somehow — characterize, don't assume.)
Time-box: if not root-caused after the ladder, log all findings under `## S2-T9` and stop.
Footgun: don't crank weights to paper over it — find the mechanism (Alex's own sweep shows
coll_weight 50/100 REGRESS; same philosophy applies to every weight here).

## Parked (do not action without Prabin)

- Held-classifier false positive (walk1 frames 3205-3208: classified held, foot 14cm below floor
  mid-swing) + held-run-START transient penetration (13.7cm) — smaller residuals, logged.
- N2/S3 blockers: BeyondMimic wants IsaacLab v2.1.0/IsaacSim 4.5.0, installed is 5.1.0-rc.19
  (compat unverified); WandB account/registry needed. Prabin decides.
- S1 Table-I→BVH mapping UNMAPPED — author-contact fallback is Prabin's call.

## Standing terminology/framing rules for all future tables + paper text (Prabin, 2026-07-16)

1. **Never call the kinematic guard "faithfulness."** GMR's paper reserves "faithfulness" for its
   N=20 human-rater user study — ours is a kinematic proxy (FK robot body position vs GMR's own
   scaled-human targets, ik_match_table2 correspondence). Call it **"source-target deviation"**
   (`srcDev`). Existing CSVs/log entries keep their `faith*` column names (historical record);
   every NEW table and all paper text uses srcDev.
2. **float% is a diagnostic, never a scoreboard.** All variants of global Z-calibration (their
   heightfix, our grounding) saturate at 91-99% by construction — a single shift can only make one
   instant touch. Never present few-point float% differences as method comparisons. The genuine-
   contact discriminator is per-frame support-point distance at detected-contact frames
   (GMR-polished +13cm vs contact-in-the-solve -3cm). Also always carry the caveat: float% counts
   genuinely airborne frames (jumps, mid-fall flight) — it indicts a reference only where support
   is expected.

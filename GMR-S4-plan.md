# GMR Sprint S4 plan — fix OURS floor penetration, then re-earn the corpus claim

> **SUPERSEDED 2026-07-17 by `GMR-S5-plan.md`.** T1–T4 were executed (see
> `planLogGMR.md ## S4-*`; best result: tuned `--swing-clear`, mean pen% 55.9 — gate
> pen%≤10 not cleared). T5/T6 will never run under this plan: after visual comparison
> of the annotated renders, Prabin pivoted to GMR's mink-based tracking as the base
> with a contact layer on top. The joint metric (T5) and oracle-shift baseline carry
> over into S5 unchanged. Kept for reference only — do not execute.

Execution plan for a lower model. Self-contained: read this fully before touching anything.
Branch: `gmr-baseline`. All Python via `conda run -n gmr python ...` from repo root.

## Why this sprint exists (context you must not lose)

Sprint S3 ran the full 77-clip LAFAN1 corpus: GMR (raw / heightfix / polished) vs OURS
(raw / stageA / ctground) on Unitree G1. OURS "won" the held-frame support_z metric
(~0cm median foot-to-floor during human-planted frames, 82–87% within 3cm, vs GMR-polished
4.4–11.7cm / 0.3–31%). **That framing was killed the same day by a z-shift oracle test**
(`outputs/gmr_baseline/sprint/s3_zshift_oracle.csv`): shifting GMR-polished down by one
per-clip constant beats OURS on held-frame within-3cm (96–99% vs 82–87%) AND max floorPen
(6.6–13.4cm vs 17–23cm). Works because GMR's held-foot float is near-constant within a clip
(p90−p10 ≈ 2.6–3.3cm) — one constant zeroes it. So single-axis metrics (float alone, or
penetration alone) are gameable and are dead for the paper.

The un-gameable target is **joint and per-frame: held-foot contact within 3cm AND whole-body
floor penetration <5mm**. A rigid shift cannot satisfy both (it trades one for the other).
Nothing currently passes it:

| variant | held frac3 (floor class) | pen% (frames >5mm) |
|---|---|---|
| gmr_polished | ~0.5% | 0.3 |
| gmr_polished + oracle shift | ~96% | 100 |
| ours_ctground | ~83% | 66 |

OURS's blocker is its own penetration: **62–81% of frames >5mm below floor, already at the RAW
solve** (not introduced by polish/grounding; stageA 74–81%, ctground 66–69%). Known mechanisms
from S2 (see `planLogGMR.md ## S2-T9..T12` and
`wiki/experiments/gmr-baseline-sprint-s2.md`):
1. **Warm-start basin**: knee gets pinned straight at its joint limit; per-iteration clamp keeps
   it there; a bent-knee warm start demonstrably fixes some worst frames (9.76cm pen → 0.00cm).
   A fix exists: `--knee-bias-weight` / `--knee-min-flex-deg` on the solver (opt-in, default
   OFF), but its 4-clip trial regressed fallAndGetUp1_subject1 floorPen by +11cm — unexplained.
2. **Genuine reach limit**: G1 is ~0.64 of this human's size; some poses exceed max leg reach
   (worst 181.6% of reach on fallAndGetUp2_subject2). Not all penetration is fixable.
3. **Suspicion of a third, dumber cause**: walk3_subject1 has 32cm max penetration and 75% pen%
   — on a *walking* clip, nowhere near reach limits. The per-frame Stage-3 solve may simply have
   NO floor-avoidance term for anything except the contact-held effectors. Never verified.

Sprint goal: kill OURS's penetration (T1–T4), then re-run eval with the joint metric and the
oracle-shift baseline included (T5–T6). Only that combination is presentable.

## Ground rules (violations corrupt data or trust)

- Coord frame +X fwd / +Y left / +Z up. Quaternions **wxyz**. G1 free-root qpos:
  `[x, y, z, qw, qx, qy, qz, 29 joints]` (36 total).
- G1 model/floor ONLY via `scripts/g1/g1_model_setup.py::load_g1_model_with_vetted_collision_and_floor`
  (vetted collision cylinders + injected floor plane). Never hand-build a G1 MJCF.
- NEVER overwrite existing outputs. S3 artifacts (`outputs/gmr_baseline/sprint/ours_g1_corpus/`,
  `s3_full_corpus.csv`) are the frozen "before" state. All S4 rebuilds go to NEW paths
  (`ours_g1_corpus_s4/`, `s4_*.csv`).
- No git add/commit/push. Do not touch `SESSION_HANDOFF.md`. Do not write to `wiki/` — Prabin
  logs when he says so. The ONLY log you write is `planLogGMR.md`: append `## S4-T<n> | <title>`
  entries as you finish tasks (same style as the existing S2 entries — what you did, exact
  numbers, verdict).
- Never edit `assets/alex/alex_floating_base_with_sites.xml` or run the historical model-prep
  scripts (`create_alex_mujoco_sites_model.py`, `build_alex_v2_collision_model.py`,
  `prepare_*`). (Alex-side footgun; you shouldn't be near Alex files at all this sprint.)
- Long solves: run resumable / per-clip, log to a file, check the log — don't sit blocked.

## Key files

- `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py` — Stage-3 per-frame DLS IK (root+joints
  joint solve, contact-held effectors pulled to floor). Has the opt-in knee_bias flags.
- `scripts/g1/polish_ours_g1.py` — Stage A smoothing (floor/self-collision sensitivity boost).
- `scripts/g1/ground_ours_contact_aware.py` — lift QP, hard cap 0 at held/contact frames.
- `scripts/g1/sprint_s3_full_corpus.py` — corpus build (resumable) + eval → combined CSV. The
  eval defines every metric you'll reuse: `whole_clip_metrics` (floorPen = max over frames of
  deepest sub-floor point over vetted geoms, cm; pen_pct = % frames pen>5mm; self-collision via
  `_collision_stats` with `floor_gid` excluded) and `held_metrics` (held mask = debounced
  human contact flags AND human foot speed <0.05 m/s; support_z from `stage_b_g1.support_z`).
- `scripts/g1/sprint_s3_summary.py` — class-split summary (floor n=34 / locomotion n=43 via
  `outputs/gmr_baseline/sprint/s1t4_reclass.csv`).
- Inputs that already exist for all 77 clips: canonical humans
  `outputs/gmr_baseline/sprint/canonical_human/<clip>_lafan1c_grounded.npz`, GMR pkls
  `outputs/gmr_baseline/sprint/pkl/<clip>{,_gmrfix,_polished}.pkl`.
- `planLogGMR.md ## S2-T12` — exact knee_bias weight/threshold values used in the 4-clip trial.
  Read it before T2; reuse those values, don't invent new ones.

The 4 validated dev clips (all gates run on these): `walk1_subject1`, `fallAndGetUp1_subject1`,
`fallAndGetUp2_subject2`, `ground1_subject1`. Add `walk3_subject1` as a 5th for T1 only (the
pathological walker).

---

## T1 — Diagnose WHAT is penetrating (do this first; it steers T3)

New script `scripts/g1/diag_penetration_source.py`. For the 5 dev clips × variants
{ours_raw, ours_stageA, ours_ctground} (existing S3 npz's, `qpos` key, in
`outputs/gmr_baseline/sprint/ours_g1_corpus/`):

Per frame with penetration >5mm: find the deepest sub-floor point and which body/geom owns it
(reuse the mesh-min-z machinery: `post_process_ground_contactfirst._build_mesh_cache` /
`_robot_lowest_z` — mirror how `sprint_s3_full_corpus.py` sets up model, geom_ids, mesh_cache).
Record: frame, deepest body name, depth, and whether that frame had a held contact on either foot
(recompute the held mask exactly as `sprint_s3_full_corpus.py::do_eval` does).

Output: CSV per clip+variant in `outputs/gmr_baseline/sprint/s4_diag/` plus a printed summary
table: per clip+variant, % of penetrating frames attributable to each body (top 5 bodies), split
by "penetrating body IS a held foot" vs "free body".

Interpretation guide (write your verdict in planLog):
- Mostly **free swing foot / knee at moderate depth on walk clips** → per-frame floor avoidance
  is simply missing → T3 is the main fix.
- Mostly **held feet themselves overshooting through floor** → pull-to-floor anchor overshoot /
  warm-start basin → knee_bias (T2) is the main fix.
- Mostly **pelvis/torso at prone frames on floor-class clips** → reach-limit territory → expect
  a residual; T3 helps but won't zero it.

## T2 — Root-cause fallAndGetUp1's knee_bias regression

Setup: re-solve `fallAndGetUp1_subject1` twice from the existing grounded canonical (bias OFF —
should byte-match or metric-match the existing `_ours.npz`; bias ON with the S2-T12 values).
Write bias-ON output to `outputs/gmr_baseline/sprint/s4_kneebias/` (do NOT overwrite S3 files).

Then per-frame floorPen curves for both. Find frames where ON is worse by >2cm. At the 5 worst:
dump knee joint angles (both legs), which body penetrates (reuse T1 tooling), held flags, and
the same frame's values in the OFF run. Test these hypotheses explicitly:
(a) bias forces knee flexion during prone/get-up poses where a straight leg was correct, dropping
    shank/knee into the floor;
(b) bias fights the pull-to-floor anchor on a held foot (check: does it happen only at held
    frames?);
(c) warm-start divergence — one early bad frame, everything downstream inherits it (check: is
    the damage contiguous from a single onset frame?).

Deliverable: a one-paragraph root cause in planLog + a proposed remedy. Likely remedy shapes
(pick based on evidence, implement the smallest one): gate the bias to frames where the knee is
AT its limit AND its foot has sub-floor error; or ramp bias weight down when the human pose is
prone (pelvis below some height); or per-frame bias only on the leg whose target is unreachable.
Implement it behind a flag; verify fallAndGetUp1 floorPen returns to ≤ OFF level while the other
3 dev clips keep their bias-ON gains (numbers in `planLogGMR.md ## S2-T12`).

## T3 — Floor avoidance in the per-frame solve (scope depends on T1)

If T1 shows free-body penetration: add a one-sided floor-avoidance task to
`solve_lafan1_canonical_g1_contactfirst.py`, opt-in flag `--floor-avoid-weight` (default 0.0 =
current behavior, exactly like knee_bias was added). Shape: for each monitored body (both feet
always; knees/pelvis if T1 implicates them), if its support_z (or lowest point) < 0, add a
z-only task pushing it back to 0 — one-sided: NO force when above floor, so it cannot fight
swing clearance. It must not conflict with the pull-to-floor anchor on held effectors (held
effectors already have a floor target; skip them or make the avoidance target identical).
Mind the existing per-iteration step clamp — same mechanism that traps the knee can trap this;
if the term stalls, check whether the clamp is eating it before raising the weight.

Sanity gate before proceeding: walk1 + walk3 with `--floor-avoid-weight` ON — pen% on walk clips
should collapse (walking has no reach excuse), held-frame medians must stay ~0cm, self-collision
% must not blow up (compare against the S3 CSV rows).

## T4 — 4-clip gate (the go/no-go for the corpus)

Full pipeline (solve → stageA → ctground) on the 4 dev clips with the S4 config (knee_bias ON
post-T2-remedy + floor-avoid ON if built), outputs under `outputs/gmr_baseline/sprint/s4_dev/`.
Evaluate with the same metrics as S3 plus the joint metric (T5's definition — fine to prototype
it here first).

**GATE (all four clips):**
- pen% ≤ 10 (from 66–81);
- held-foot median within ±1cm and frac3 ≥ current S3 values (no contact regression);
- self-collision % not worse than S3 by more than 2 points;
- joint metric (below) beats BOTH gmr_polished and the oracle-shifted baseline on ≥3 of 4 clips.

If the gate fails: STOP. Write the numbers and your diagnosis in planLog and end the sprint
there. Do not run the corpus, do not tune in circles past ~2 focused attempts per failure mode.

## T5 — Joint metric + oracle baseline in the eval harness

Extend the eval (new script `scripts/g1/sprint_s4_eval.py`, modeled on
`sprint_s3_full_corpus.py::do_eval` — don't mutate the S3 script, it's the frozen reference):

1. **Joint metric**: per frame, success = (every currently-held foot has |support_z| < 3cm) AND
   (whole-body penetration < 5mm). Report `joint_ok_pct` over frames that have ≥1 held foot, per
   clip. This is the paper's headline number.
2. **Oracle-shift baseline** as a first-class variant `gmr_shift_oracle`: from gmr_polished,
   per clip compute per-frame lowest-z and per-held-frame support_z once, then pick the constant
   dz maximizing pooled held within-3cm fraction (grid over [min−3cm, max+3cm], 400 steps) and
   report all metrics at that dz. Shifts are exact-analytic: support_z' = support_z − dz,
   lowest' = lowest − dz, self-collision unchanged. (Reference numbers to reproduce: means
   frac3 ≈ 96–99%, floorPen ≈ 6.6/13.4cm loco/floor, pen% = 100 — see
   `outputs/gmr_baseline/sprint/s3_zshift_oracle.csv`.)
3. Variants in the S4 CSV: gmr_raw, gmr_heightfix, gmr_polished, gmr_shift_oracle, ours_raw,
   ours_stageA, ours_ctground (S4 config). Columns = S3 columns + `joint_ok_pct` + `dz_cm`
   (oracle only).
4. Extend `sprint_s3_summary.py` conventions into a `sprint_s4_summary.py` (same class split via
   `s1t4_reclass.csv`) including the joint metric column.

## T6 — Corpus rerun (ONLY if T4 gate passed)

Rebuild OURS for all 77 clips with the S4 config into
`outputs/gmr_baseline/sprint/ours_g1_corpus_s4/` (copy `sprint_s3_full_corpus.py --build`'s
resumable pattern; canonical humans already exist, only solve→stageA→ctground rerun; ~1–2 min
per clip solve, run it detached with a log, resume on interruption). Then `sprint_s4_eval.py`
over all 77 → `outputs/gmr_baseline/sprint/s4_full_corpus.csv` → summary.

Success statement to check at the end (this is what makes the results presentable):
**OURS (S4) beats gmr_polished AND gmr_shift_oracle on joint_ok_pct on both classes, with pen%
in single digits and held-frame contact preserved.** Report the honest residual too (clips where
reach limit keeps joint_ok low — expect some floor-class clips to stay hard; list the worst 5
with their over-reach character).

Final deliverable: planLog `## S4-T6` with the class-split table (all 7 variants × floorPen /
pen% / frac3 / joint_ok_pct), the success-statement verdict clip-counted, and the worst-5
residual list. No wiki writes.

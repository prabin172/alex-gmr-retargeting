# GMR Sprint S5 plan — pivot: GMR-quality tracking base + contact layer

Execution plan for Sonnet. Self-contained: read this fully before touching anything.
Branch: `gmr-baseline`. All Python via `conda run -n gmr python ...` from repo root.
Written 2026-07-17 by Fable after Prabin's visual verdict and pivot decision (see
`planLogGMR.md ## S5-D0`). Supersedes `GMR-S4-plan.md` (S4-T5/T6 will never run).

## Why this sprint exists (context you must not lose)

S4 ended with `--swing-clear` (tuned `mp=8/cr=0.2`) as the best OURS mechanism: improved
pen%/held-contact on all 11 test clips, mean floorPen below baseline. It still left mean
pen% at 55.9 — nowhere near the ≤10 gate. Then Prabin watched the three annotated renders
(repo root: `walk1_subject1_{raw,swingclear_tuned,gmr_heightfix}_annotated.mp4`) and gave
the deciding verdict:

- **GMR (heightfix, their own method, zero smoothing from us)**: visually excellent. No
  flicker, no snapping, natural hand orientation. 15.9% frames pen >0.5cm, max 1.66cm.
- **OURS**: "flickers, and snaps, and all penetration" — palm rotated toward body,
  penetrating the thigh; heavy knee bend of unclear origin; max 8.76cm pen even tuned.

Root architectural differences (verified by reading
`~/projects/GMR/general_motion_retargeting/motion_retarget.py` and
`ik_configs/bvh_lafan1_to_g1.json` directly):

1. **Solver**: GMR = `mink` velocity-space QP (`mink.solve_ik` + `integrate_inplace`,
   dt-sized steps, up to 10 iters/table, early exit when error improvement <0.001), with
   joint limits as HARD QP constraints (`mink.ConfigurationLimit`, `mink.VelocityLimit`).
   OURS = custom position-space DLS with per-iteration post-hoc clamping
   (`clamp_hinge_joint_limits`) — the documented warm-start/branch-flip flicker source.
2. **Weighting philosophy**: GMR is orientation-first. Table 1 is essentially
   rotation-only (rot cost 10–100 on ~13 bodies incl. thigh/shank/upper-arm/forearm;
   pos cost 0 everywhere except feet 50). Table 2 adds position lightly (pelvis 100,
   most others 5–10). OURS is position-first: 15 position points, orientation on only
   7 roles with max weight 0.70. Position-only limb targets leave hand roll about the
   forearm axis and the knee bend-plane UNDER-CONSTRAINED — that is the palm-toward-body
   and knee-weirdness mechanism, and much of the branch-flip redundancy.
3. **Contact**: GMR has NONE. Its only floor handling is one constant Z offset for the
   whole clip (`set_ground_offset`/`apply_ground_offset`), same mechanism as the S3
   z-shift oracle. Nothing per-frame. **This is the paper's opening.**

Paper thesis (unchanged in spirit from S4, new base): *contact-aware retargeting layered
on a SOTA kinematic retargeter*. Headline metric = the un-gameable joint per-frame metric
from S4-T5: every held foot within 3cm of floor AND whole-body penetration <5mm
(`joint_ok_pct`). Constant-shift baselines (incl. the oracle) cannot pass it; nothing
currently does.

Prabin's decided strategy (do not re-litigate):
- **Phase B first, strictly time-boxed (~2 working days)**: test whether OUR DLS solver
  reaches GMR-grade visual quality via orientation-first reweighting + real joint-limit
  handling. Outcome is diagnostic + a possible paper ablation ("is it the weights or the
  QP architecture?"). **Phase B's outcome does NOT gate Phase A.**
- **Phase A regardless**: build the contact layer inside GMR's own mink solve. This is
  the paper's critical path.
- Simple motions first: make walking perfect (foot snaps to ground, zero skate, no
  regression of GMR's tracking quality), then extend toward hard clips honestly.

## Ground rules (violations corrupt data or trust)

- Coord frame +X fwd / +Y left / +Z up. Quaternions **wxyz** everywhere in THIS repo.
  Known boundaries: GMR pkl `root_rot` is **xyzw** (`load_gmr_pkl.py` converts — reuse
  it, never hand-convert); mink's `SE3`/target quaternion order must be VERIFIED
  empirically before you write targets (identity + known-rotation probe), do not assume.
- G1 free-root qpos: `[x, y, z, qw, qx, qy, qz, 29 joints]` (36 total).
- `~/projects/GMR` is a READ-ONLY reference. Never edit it. Integrate by importing and
  subclassing `GeneralMotionRetargeting` in our repo. If subclassing is impractical
  (private attrs, rigid init), copy `motion_retarget.py` into our repo as
  `scripts/g1/gmr_vendored_retarget.py` with a provenance header + minimal diffs, and
  log why.
- IK for Phase A runs on GMR's own XML (`ROBOT_XML_DICT["unitree_g1"]` =
  `g1_mocap_29dof.xml`). ALL eval/render stays on OUR vetted model via
  `scripts/g1/g1_model_setup.py::load_g1_model_with_vetted_collision_and_floor` (same 29
  joints/order — already validated by every GMR-pkl eval to date). Keep that split.
- `scripts/solve_fbx_canonical_alex_contactfirst.py` is SHARED with the Alex pipeline.
  Any Phase-B change there must be opt-in (flag/param defaulting to current behavior).
- NEVER overwrite existing outputs. S3/S4 artifacts are frozen. New solves go to new
  paths: `outputs/gmr_baseline/sprint/pkl_s5/`, `canonical_human_s5/` (if regenerating
  canonicals), `s5_*.csv`.
- No git add/commit/push, ever. Do not touch `SESSION_HANDOFF.md`.
- Logging: append `## S5-<task> | <title>` entries to `planLogGMR.md` as you finish
  tasks (what you did, exact numbers, verdict — same style as S4 entries). Wiki/docs:
  ONLY in the final task (T-DOC), which Prabin pre-authorized for this sprint.
- Long solves: resumable per-clip loops, log to file, don't sit blocked.
- Never edit `assets/alex/alex_floating_base_with_sites.xml` or run the historical
  model-prep scripts (`create_alex_mujoco_sites_model.py`,
  `build_alex_v2_collision_model.py`, `prepare_*`).

## Key files & facts (all verified 2026-07-17)

- `scripts/g1/gmr_headless_retarget.py` — runs GMR's real retargeter headless
  (BVH → pkl). Already has `--save_human_targets` (dumps `retargeter.scaled_human_data`
  per frame: body_name → (pos, quat) — GMR's own scaled-human FK targets). This is your
  fidelity ground truth AND your hand-frame diagnostic source.
- `scripts/g1/load_gmr_pkl.py::load_gmr_pkl` — pkl → (qpos (T,36) wxyz, fps).
- GMR internals: `~/projects/GMR/general_motion_retargeting/motion_retarget.py`
  (class `GeneralMotionRetargeting`: `ik_match_table1/2`, `mink.FrameTask` per body,
  `retarget()` per frame, `scale_human_data`/`offset_human_data`/ground-offset helpers),
  `params.py` (`ROBOT_XML_DICT`, `IK_CONFIG_DICT`), `ik_configs/bvh_lafan1_to_g1.json`
  (the correspondence + cost tables — copy its body mapping when you need limb-segment
  roles). `mink` is installed in env `gmr`; API surface includes `FrameTask`,
  `PostureTask`, `DampingTask` (verify setter names — `set_target`,
  `set_position_cost`, `set_orientation_cost` — before relying on per-frame mutation).
- Canonical humans (all 77 clips):
  `outputs/gmr_baseline/sprint/canonical_human/<clip>_lafan1c_grounded.npz` with
  `positions (T,20,3)`, `roles (20)`, `orientation_mats (T,7,3,3)`,
  `orientation_role_names (7: pelvis, torso, head, feet, hands)`,
  `contact_flags (T,4)`, `contact_effector_names (4: feet+hands)`,
  `contact_support_z (T,4)`, `fps`. T matches the BVH frame count (walk1: 7840) —
  ASSERT this per clip anyway before using contacts against a GMR run.
- Raw BVH (all clips): `data/raw/lafan1/<clip>.bvh`.
- Eval machinery: `scripts/g1/sprint_s3_full_corpus.py::whole_clip_metrics/held_metrics`
  (floorPen, pen%, coll%, held support_z frac3; held mask = debounced human contact AND
  human foot speed <0.05 m/s), `sprint_s3_summary.py` (class split via
  `outputs/gmr_baseline/sprint/s1t4_reclass.csv`), S4-T5's `joint_ok_pct` definition
  (never implemented — you build it here).
- Contact tooling: `scripts/contact_labels.py::debounce_flags`, `ramp_envelope`.
  Sole-height: `scripts/g1/stage_b_g1.py::support_z` + mesh machinery
  `post_process_ground_contactfirst._build_mesh_cache/_geom_lowest_z`.
- Renderer: `scripts/g1/render_penetration_annotated.py` (`--qpos` or `--pkl`; the three
  repo-root videos used `--start 3000 --frames 1800 --width 960 --height 720` — reuse the
  exact same window/camera for any comparison render).
- OURS solver (Phase B target): `scripts/g1/solve_lafan1_canonical_g1_contactfirst.py`
  (driver, `ROLE_TO_G1_BODY` 15 pos roles, `ORI_TO_G1_BODY` 7 ori roles) +
  `scripts/solve_fbx_canonical_alex_contactfirst.py` (`solve_frame_position_ik`,
  `TARGET_WEIGHTS`, `ORI_WEIGHTS`, `clamp_hinge_joint_limits`).
- Canonical converter (Phase B1 target): `scripts/g1/lafan1_to_canonical_human.py`.

Dev clips: locomotion = `walk1_subject1` (visual reference), `walk3_subject1`
(pathological walker, 32cm pen in S3), `run2_subject1`. Hard class (A6 only) =
`fallAndGetUp1_subject1`, `ground1_subject1`.

---

# Phase B — can OURS reach GMR quality? (time-box: ~2 days, then move on NO MATTER WHAT)

Purpose: distinguish "our weights were wrong" from "the QP architecture is the point".
Either answer is a paper sentence. Max 2 focused attempts per sub-task; log and move on.

## B0 — Diagnostics first (no solver edits; ~half day)

**B0.1 Hand-frame diagnostic** (explains palm-toward-body or exonerates the target):
Run `gmr_headless_retarget.py` on walk1 with `--save_human_targets`. Load the dump's
`LeftHand`/`RightHand` quats (verify quat order empirically first: check norms, then
sanity-probe a frame where the human clearly faces +X). Build our canonical hand
orientation frames for the same clip (`orientation_mats`, roles `left_hand`/`right_hand`).
Per frame compute the geodesic angle between our hand frame and GMR's hand frame
(after aligning any fixed convention offset — estimate the median relative rotation over
the clip first, subtract it, then look at the residual). Deliverable in planLog:
- If the *residual* is small (<10° typical) → our hand TARGET is fine; the solver's weak
  ori weight (0.40) is the culprit → B2 will fix or nothing will.
- If there's a large systematic offset about the forearm axis (~90°) → our hand target
  itself is wrong (S2-T4's forearm-fallback fix was insufficient) → fix target
  construction in `lafan1_to_canonical_human.py` (B1), regenerate dev-clip canonicals to
  `canonical_human_s5/`.

**B0.2 Knee diagnostic** (answers Prabin's "is the knee bend forced?"):
knee joint angle stats (mean/p50/p90, both legs) on walk1: OURS s3_raw npz vs OURS
swing-clear-tuned npz vs GMR heightfix pkl (via `load_gmr_pkl`). Also count frames at the
knee joint limit. One table in planLog. No fix here — just the number Prabin asked for.

**B0.3 Smoothness metric** (the "flicker" number, reused all sprint):
New `scripts/g1/motion_smoothness.py`: given qpos (T,36) + fps → mean and p95 of joint
jerk (third difference of the 29 joints, rad/s³) and of foot/hand body linear
acceleration (FK on our vetted model). Compute for: OURS s3_raw, OURS swing-clear tuned,
GMR raw, GMR heightfix (walk1 + run2). Log table. Expect GMR ≪ OURS — this quantifies
the verdict and becomes the smoothness column in A7.

## B1 — Hand target fix (ONLY if B0.1 implicates the target)

Fix the hand semantic frame in `lafan1_to_canonical_human.py` (use GMR's own Hand-bone
frame convention as reference truth from the B0.1 dump). Regenerate dev-clip canonicals
into `outputs/gmr_baseline/sprint/canonical_human_s5/`. Re-solve walk1 with existing
defaults; confirm palm-toward-body gone in a render; log before/after hand-thigh
collision % (arm-related pairs from `_collision_stats`).

## B2 — Orientation-first reweighting (opt-in preset)

Add `--gmr-style-weights` to `solve_lafan1_canonical_g1_contactfirst.py`: a preset that
overrides the weight dicts passed into the solve — orientation up (pelvis/torso ~2–4,
feet ~2, hands ~1.5–2), distal position down (knee/elbow/wrist/hand ~0.1–0.3; keep
pelvis pos 4.0 and ankle pos ~1.0). Numbers above are starting points, not gospel — one
round of adjustment allowed.

Known coverage gap: we have NO thigh/shank/upper-arm/forearm ori roles (GMR tracks all
four; that's what pins knee bend-plane and hand roll). **B2b (only if B2 helps but
plateaus):** add those 8 roles end-to-end — semantic frames in the canonical converter
(long axis from landmark pairs hip→knee, knee→ankle, shoulder→elbow, elbow→wrist;
secondary axis from the adjacent segment, mirroring existing semantic-frame construction),
new entries in `ORI_TO_G1_BODY` copying GMR's mapping (`bvh_lafan1_to_g1.json`:
LeftUpLeg→`left_hip_yaw_link`, LeftLeg→`left_knee_link`, arms analogous), regenerated
dev canonicals in `canonical_human_s5/`. This is the biggest B item — skip if the
time-box is tight and note it as untested.

## B3 — Joint-limit handling + convergence exit (opt-in)

Two bounded changes in `solve_frame_position_ik` (both opt-in, Alex default unchanged):
1. Active-set limit handling: when a hinge is AT its limit and the DLS step pushes
   further in, zero that DOF's step component (and optionally its Jacobian column) for
   the remaining iterations of the frame, instead of clamp-after-step. This targets the
   documented knee warm-start basin.
2. Early exit: stop iterating when task-error improvement <1e-3 between iterations
   (GMR's own criterion) instead of always burning the full budget.

## B-GATE (walk1 + run2; write ## S5-B verdict in planLog, then go to Phase A)

Render walk1 (same window as repo-root videos) with best B config. Pass =
- no visible flicker/snapping (your honest judgment, stated plainly),
- arm/hand self-collision ≈ 0,
- jerk (B0.3) within ~2× GMR's,
- pen% / held-frac3 no worse than swing-clear tuned (`mp8/cr0.2`).

Pass or fail, Phase A starts next. If pass: OURS becomes a real second solver for the
paper (contact layer shown solver-agnostic). If fail: one honest paragraph on which
mechanism resisted, and OURS-DLS is retired to an ablation row.

---

# Phase A — contact layer inside GMR's mink solve (the paper)

## A1 — `scripts/g1/gmr_contact_retarget.py` (feet only, v1)

New driver, modeled on `gmr_headless_retarget.py` (reuse its BVH loading, pkl saving,
`--save_human_targets`, optional video). Wraps/subclasses `GeneralMotionRetargeting`.

Per clip:
1. Load canonical npz; ASSERT its T == number of BVH frames. Held mask per foot =
   `debounce_flags(contact_flags[:, foot], 2)` AND human foot speed <0.05 m/s (speed
   from canonical `positions` of `left_ankle`/`right_ankle` × fps — identical recipe to
   `sprint_s3_full_corpus.py::do_eval`; yes, this is also the eval's trigger — fine,
   because the eval MEASURES robot-side geometry, the trigger only says when to try).
2. Constants, computed once on OUR vetted model: `z_sole[foot]` = height of
   `*_ankle_roll_link` body origin above the foot's lowest mesh point with the foot
   flat at qpos0 (use `_build_mesh_cache`/`_geom_lowest_z` or `stage_b_g1.support_z`).
3. Per frame, BEFORE `retargeter.retarget(frame)`: if a foot is held, override that
   foot's table-2 FrameTask (the one targeting `left/right_ankle_roll_link`):
   - position target: XY = robot's own ankle XY at hold onset (from FK of the previous
     frame's qpos at onset — locks the foot, kills skate), Z = `z_sole[foot]`;
   - orientation target: flat foot — roll=pitch=0, yaw = yaw-component of GMR's own
     current target for that foot;
   - costs: ramp position cost 50→~200 and orientation cost 10→~50 with
     `ramp_envelope` over ~5 frames at hold entry/exit (mutate via
     `set_position_cost`/`set_orientation_cost`; verify the API once at startup).
   When not held: restore GMR's original target/costs exactly (byte-identical behavior
   to raw GMR must be recoverable — see A2 fidelity guard).
4. Ground offset: keep GMR's existing per-clip constant ground offset exactly as the
   heightfix pipeline does (check how `_gmrfix` pkls were produced — planLog S1 — and
   replicate), so swing-phase tracking is already height-sane; the snap then makes
   stance exact rather than fighting a global bias.
5. Save pkl (GMR format) to `outputs/gmr_baseline/sprint/pkl_s5/<clip>_gmrcontact.pkl`
   + the human-targets npz for fidelity eval.

Sanity check FIRST: run with the contact override disabled and diff qpos vs a fresh
plain `gmr_headless_retarget.py` run of the same BVH — must match to float tolerance.
That proves the wrapper adds nothing until asked.

## A2 — Metrics + locomotion gate

New `scripts/g1/sprint_s5_metrics.py` (import, don't mutate, the S3 eval):
- `joint_ok_pct` (S4-T5 definition): per frame with ≥1 held foot, success = every held
  foot |support_z| <3cm AND whole-body pen <5mm. THE headline number.
- `skate_cm`: per held segment, max XY drift of the ankle body within the segment;
  report per-clip mean and max.
- `fidelity`: vs the run's own saved human-targets npz — mean position error (cm) and
  mean geodesic orientation error (deg) over the NON-foot tracked bodies, all frames.
  Compare gmr_contact vs gmr_raw: budget = ≤1cm and ≤2° mean increase.
- `jerk`: from `motion_smoothness.py` (B0.3).
- Plus the S3 columns (floorPen, pen%, coll%, held frac3) unchanged.

**A2 GATE (3 loco dev clips):** `joint_ok_pct` ≥90; mean skate ≤1cm; fidelity within
budget; jerk not worse than gmr_raw by >20%; render walk1 (same window/camera as the
repo-root videos) — visually indistinguishable from gmr_heightfix except feet planted.
If the solve oscillates at hold boundaries: lengthen ramp (5→10→15 frames) before
touching costs; max 3 attempts, then log and reassess with Prabin.

## A3 — Swing clearance (small; only if joint_ok needs it)

GMR's residual swing pen is small (walk1 max 1.66cm). If it's what's keeping
`joint_ok_pct` <90: clamp the swing-foot POSITION TARGET z to ≥ `z_sole + margin` in
target space pre-IK (no new QP machinery). One attempt.

## A4 — Post-hoc ablation ("contact post-process")

Same held targets applied AFTER a raw GMR solve instead of inside it: extend
`scripts/g1/polish_gmr_pkl.py` (or a new script reusing `stage_b_g1.py`'s QP anchoring)
to snap held feet to floor + zero skate on the raw pkl → variant `gmr_contact_post`,
`outputs/gmr_baseline/sprint/pkl_s5/<clip>_gmrcontactpost.pkl`. This is the paper's
in-loop vs post-hoc ablation AND keeps our S1–S4 machinery as a contribution. Same A2
metrics.

## A5 — Hard-class extension (ONLY after A2 gate passes; time-boxed 1 day)

Canonical `contact_flags` covers hands too. Extend the held mechanism to hands
(`left/right_wrist_yaw_link` per GMR's table; z_sole analog from wrist body) on
`fallAndGetUp1_subject1` + `ground1_subject1`. Expect partial success (reach limits are
real: G1 is ~0.64 human scale, S2 documented 181% over-reach poses). Report honestly —
locomotion solved + hard-class honestly characterized is a viable paper; don't sink time.

## A6 — Corpus + eval + summary

1. Resumable batch (loco class first, then floor class):
   `gmr_contact` + `gmr_contact_post` for all 77 clips → `pkl_s5/`. (~1–2 min/clip;
   detached with a log.)
2. `scripts/g1/sprint_s5_eval.py` → `outputs/gmr_baseline/sprint/s5_full_corpus.csv`.
   Variants: gmr_raw, gmr_heightfix, gmr_polished, gmr_shift_oracle (port S4-T5's
   analytic-shift recipe — reference numbers in `s3_zshift_oracle.csv`), ours_swingclear
   (S4 best, existing npz where built — rebuild missing loco clips only if cheap),
   gmr_contact, gmr_contact_post, plus ours_B if the B-gate passed.
   Columns: S3 columns + joint_ok_pct + skate + jerk + fidelity + dz_cm (oracle only).
3. `sprint_s5_summary.py`: class split via `s1t4_reclass.csv`.

**Success statement to check:** *gmr_contact beats every baseline including
gmr_shift_oracle on joint_ok_pct on BOTH classes, with skate ≈0, fidelity within budget
of gmr_raw, and GMR-level smoothness.* Report the worst-5 residual clips with their
failure character (reach limit vs contact ambiguity vs other).

## A7 — Renders

Annotated render of walk1 `gmr_contact`, exact same window/camera as the three repo-root
videos, to repo root as `walk1_subject1_gmrcontact_annotated.mp4`. Tell Prabin it's
ready to push; do NOT git-commit it yourself.

---

# T-DOC — docs/wiki refresh (pre-authorized by Prabin for this sprint; do it LAST)

- `GMR-baseline-results.md`: mark S3-era claims that the z-shift oracle killed; add the
  S4 outcome (swing-clear tuned = best OURS, gate not cleared) and the S5 pivot + final
  S5 numbers.
- `GMR-baseline-plan.md` / `GMR-baseline.md` / `notes.md`: fix stale statements (e.g.
  anything implying OURS-DLS is the paper's primary method, or that GMR has no
  reproduction/eval harness).
- `wiki/experiments/`: new page `gmr-baseline-sprint-s4-s5.md` (S4 summary + S5 results),
  keep `wiki/index.md` lean + current, one-liners in `wiki/log.md`
  (`## [YYYY-MM-DD] <op> | <what>`).
- Do NOT touch `SESSION_HANDOFF.md`. Do NOT duplicate METHOD.md math.
- `planLogGMR.md` should already have your `## S5-*` entries from along the way.

# Order of work

B0 → (B1 if implicated) → B2 (→ B2b if promising) → B3 → B-GATE → A1 → A2-gate →
(A3 if needed) → A4 → A5 → A6 → A7 → T-DOC.

Time-boxes: Phase B ≤2 days hard stop. A1–A2 ≈2–3 days. A4 ≈1 day. A5 ≤1 day.
A6 ≈half day compute + eval. T-DOC ≈half day. When a time-box expires, write the honest
verdict in planLog and move to the next task — do not tune in circles.

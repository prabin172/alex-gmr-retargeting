# Plan: Feasibility-First-v1 — plan feasibility globally per frame, then optimize similarity (Stage 3, branch `p0-grounding`)

**Status: ACTIVE. Written 2026-07-15 for a lower-capability model to execute exactly.**
Author of record: Prabin (decisions), plan drafted by Claude. Prabin commits himself — the
executor NEVER runs `git add/commit/push`.

Replaces the continuation-v1 plan (GATED 2026-07-14, did not clear ship bar — preserved in
`planLog.md` T-sections and `wiki/experiments/continuation-v1-gate.md`; its code stays in
`solve_global_trajectory_opt_contactfirst.py`, opt-in, and is REUSED by this plan's T6).

---

## 0. Problem statement and hypothesis (read, don't skip)

Established across three sessions (continuation-v1 gate, hierarchical-v1, the Luigi manual-edit
comparison — see `wiki/log.md` 2026-07-14 entries):

- On whole-body get-up clips (`standup_natural_01`, `standup_side_05`), the human's motion
  strategy is kinematically infeasible for Alex. Every LOCAL trajectory-level mechanism tried
  from the human warm start returned zero (floor-hard: diverged; hard-tier: no benefit;
  continuation: base solve won't converge; multi-seed: rejected — 36k-dim trajectory space).
- A kinematically feasible solution EXISTS (mentor's manual Blender edit proves it for supine),
  it just lives in a different strategy basin our tracking objective + local solver can't reach.
- KEY DECOMPOSITION this plan tests: global search fails in trajectory space but works in POSE
  space. Each frame's IK is ~35-dim, where adaptive relaxation + restarts genuinely cross
  basins. So: make each FRAME feasible first (floor/self-collision as a gate, tracking relaxed
  as much as needed, restarts at contact-phase boundaries), THEN let Stage 4 (+ the
  already-built continuation) pull the feasible trajectory toward the human. This inverts the
  current architecture (similarity first, feasibility patched after).

**Hypothesis to gate**: per-frame feasibility-first solving yields a warm start from which
Stage 4 converges (the "healthy basin" case continuation-v1 validated), cutting the two gate
clips' floor penetration from 13.5/24.4cm to ≤ half, without spikes or visual contortion.

**Explicit non-goals** (do not build):
- No contact-schedule invention/search — the human's contact schedule (Stage 2.5 persisted
  labels) is kept as-is. No keyframe+interpolation architecture (v1 solves every frame, as the
  solver already does; only the retry wrapper is new).
- No CoM/balance/dynamics anything (settled ruling). No new solver, no OSQP in Stage 3, no
  changes to Stage 3's existing DLS machinery internals.
- No mid-phase random restarts (continuity: within a phase, only adaptive relaxation — the
  phase-level restart-plus-blend fallback is v2, documented in §5, NOT built in v1).
- No pipeline default changes. `--feasibility-first` is opt-in, default off, byte-identical.

---

## 1. Ground rules for the executor

Identical to continuation-v1's (see `planLog.md` header + the rules below):
1. `source /home/ptimilsina/miniforge3/etc/profile.d/conda.sh && conda activate gmr` for every
   python call (base env has no mujoco).
2. Conventions: +X fwd/+Y left/+Z up; quats wxyz; qpos `[x y z qw qx qy qz | 29 actuated]`.
3. NEVER run `create_alex_mujoco_sites_model.py` / `build_alex_v2_collision_model.py` /
   `prepare_*` (they overwrite the hand-maintained XML). NEVER `git add/commit/push`.
4. Default-off discipline: `--feasibility-first` off must be BYTE-IDENTICAL (T2 acceptance).
5. Append your trail to `planLog.md` (repo root) under a `# Feasibility-First-v1` heading —
   same numbers-or-it-didn't-happen standard as the existing sections.
6. Two failed attempts on a task's acceptance ⇒ STOP and log, don't redesign.
7. Scratch outputs under `outputs/ff_dev/` (gitignored). Never overwrite `outputs/contactfirst/`.

---

## 2. Code anchors (verified 2026-07-15; line numbers approximate)

File: `scripts/solve_fbx_canonical_alex_contactfirst.py` (2669 lines) — Stage 3.

| Anchor | What it is |
|---|---|
| `solve_frame_position_ik(...)` (~l.761) | per-frame DLS IK. Already has the hooks this plan needs: `pos_weight_scale`/`ori_weight_scale` (per-role weight dicts), `q_ref` (posture-reg target decoupled from `q_init`), `floor_weight/floor_gid/floor_margin/floor_gain`, `coll_weight/...`, `posture_reg` |
| main per-frame loop, `q = solve_frame_position_ik(...)` (~l.2404) | warm-chained via `q`; `floor_kwargs`/`coll_kwargs` built once before the loop |
| post-solve diagnostics (~l.2437) | already loops `data.contact` per frame counting `n_floor_pen`/`n_self_coll` — extend to track DEPTHS (see §3.2) |
| `persisted_contacts` (~l.355, l.1698) | Stage 2.5 contact labels dict `{eff: (T,) bool}` — source of phase boundaries |
| `frame_contacts` (in-loop, ~l.2489) | per-frame contact state actually used |
| `_load_model_with_floor` (~l.47) | injected floor plane, `floor_gid` |
| `refine_arm_floor_transitions` (~l.1124) | the existing local-window re-solve pattern — READ ITS q_ref DOCSTRING before implementing; v1 does not build windows but copies its q_ref/continuity reasoning |
| CLI: `args = ap.parse_args()` region | add the new flags next to `--floor-weight` etc. |

Stage 4 / eval / render / grounding: same anchors as continuation-v1 (`planLog.md`). Probe:
`scripts/dev_cont_probe.py`. Continuation: `--continuation N` on the Stage-4 script (built,
validated safe).

Rate note: gate clips are 120 Hz through the normal pipeline — use the pipeline's own Stage-3/4
flags (copy from `retargetingPipeline.sh` exactly as continuation-v1's planLog header did,
including forcing `--floor-collision on` for Stage 4 on the gate clips, same logged deviation).

---

## 3. Design (what to build — all in Stage 3 unless said otherwise)

### 3.1 Phase boundaries

From `persisted_contacts` (the (T,4) label matrix): a phase boundary is any frame t where the
4-bool contact-state vector differs from t−1. Also treat frame 0 as a boundary. Compute once
before the main loop → `phase_boundary` (T,) bool. Log the count per clip (a get-up should have
~5–15).

### 3.2 Per-frame feasibility measurement

Extend the EXISTING post-solve diagnostics loop (~l.2437) to also record depths:
`frame_floor_pen_m` = max over floor contacts of `-ct.dist` (0 if none); `frame_self_pen_m` =
max over counted self-collision contacts of `-ct.dist`. No new mj_forward — the loop already
runs. Feasibility gate: `frame_floor_pen_m <= FF_FLOOR_TOL (0.010)` AND `frame_self_pen_m <=
FF_SELF_TOL (0.015)`. (Below Stage 4's keep-best gates, so a feasible Stage-3 frame stays
acceptable downstream.)

### 3.3 The retry wrapper (`--feasibility-first`, default off)

Wrap the existing `q = solve_frame_position_ik(...)` call site. Pseudocode — follow exactly:

```
q_attempt = solve_frame_position_ik(<exactly the current call>)     # attempt 0 = today's solve
pen_f, pen_s = measure(q_attempt)                                    # §3.2
if not feasibility_first or feasible(pen_f, pen_s):
    q = q_attempt                                                    # byte-identical path when off
else:
    candidates = [(score(q_attempt), q_attempt)]
    viol_roles = roles_of_violating_limbs(contacts at this frame)    # §3.4
    for a in 1..FF_MAX_ATTEMPTS(6):
        relax  = 0.5 ** a                                            # tracking relaxation
        boost  = 2.0 ** a                                            # constraint boost
        pws    = {r: relax for r in viol_roles}                      # merge over existing pos_weight_scale
        ows    = {r: relax for r in viol_roles}                      # ori likewise
        q0     = q_prev_frame                                        # warm start unchanged...
        if phase_boundary[t] and a >= FF_RESTART_FROM(3):
            q0 = q_prev_frame + noise(sigma=0.05*(a-2), seed=(clip,t,a), actuated dofs only)
        q_try = solve_frame_position_ik(..., q_init=q0, q_ref=q_prev_frame,
                    pos_weight_scale=merge(pws), ori_weight_scale=merge(ows),
                    floor_weight=floor_weight_frame * boost,
                    coll_weight=coll_weight * boost, iters=args.ik_iters)
        candidates.append((score(q_try), q_try))
        if feasible(q_try): break
    q = min(candidates)[1]                                           # keep-best
record ff_attempts[t], ff_floor_pen_cm[t], ff_self_pen_cm[t]
```

`score(q)` lexicographic: `(max(0, pen_f - FF_FLOOR_TOL) + max(0, pen_s - FF_SELF_TOL),
pen_f + pen_s, sum of role tracking errors)` — infeasibility first, then depth, then fidelity.
Seeded noise (`np.random.default_rng(hash((clip_name, t, a)) % 2**32)`) so runs are reproducible.
`q_ref=q_prev_frame` on retries: regularize toward the previous frame (temporal continuity),
NOT toward the failed attempt — this is the `refine_arm_floor_transitions` lesson (its docstring,
§2 anchor). Note attempt 0 must remain the EXACT existing call — same args, same order — so the
off path and the feasible-first-try path are both byte-identical to today.

### 3.4 Violating-limb → roles map

From the frame's violating contact bodies (the diagnostics loop knows which geoms/bodies
violated): body name prefixed LEFT_/RIGHT_ + HIP/THIGH/SHIN/ANKLE/FOOT → that side's
`{side}_hip, {side}_knee, {side}_ankle` roles; SHOULDER/ELBOW/WRIST/GRIPPER → `{side}_shoulder,
{side}_elbow, {side}_wrist`. Trunk/pelvis/torso/head → relax NOTHING for it (but root IS a DOF
in Stage 3's IK, unlike Stage B — so a trunk violation may still resolve via root motion +
constraint boost; do not special-case beyond not relaxing trunk tracking). Mirror of
continuation-v1's `_limb_roles_for_body` but with Stage-3 role names (`ROLE_TO_ALEX_BODY` keys).

### 3.5 CLI + NPZ additions

```
--feasibility-first            action=store_true, default off
--ff-floor-tol   0.010        --ff-self-tol  0.015
--ff-max-attempts 6           --ff-restart-from 3
```
NPZ gains `ff_attempts` (T,), `ff_floor_pen_cm` (T,), `ff_self_pen_cm` (T,) when the flag is on.
End-of-solve print: frames needing retries, frames still infeasible after all attempts (the
per-frame infeasibility certificate — this number is a result even if the gate fails).

### 3.6 Downstream (nothing new): Stage 4 + continuation + grounding + eval

Stage 4 exactly as continuation-v1 ran it (flags in `planLog.md` header, `--floor-collision on`
forced on gate clips) — run BOTH plain and `--continuation 8` variants: the hypothesis says the
feasible warm start turns continuation's 2 dead clips into its healthy-basin case. Then Stage 4.5
`constant-contact`, then `dev_cont_probe.py` + `eval_artifacts_corpus.py`-style numbers + render.

---

## 4. Task list (execute in order)

### T1 — Baselines (mostly exist)
Continuation-v1's T1 numbers for `standup_natural_01`/`standup_side_05` are in `planLog.md`
(13.48/24.37cm isolated floor pen at Stage 4). Add the STAGE-3-level baseline: run Stage 3 as
the pipeline does (copy its stage-3 invocation + each clip's `solve_extra`) into
`outputs/ff_dev/<clip>_cf_base.npz`, and record per-frame floor/self pen depths (a 20-line
scratch script reusing §3.2's logic against the existing outputs is fine). Log table.
**Accept:** Stage-3 baseline depths logged for both gate clips.

### T2 — §3.1 + §3.2 + no-op certification
Phase boundaries + depth measurement + the flag parsing (wrapper not yet active).
**Accept:** with the new code, a full Stage-3 run WITHOUT `--feasibility-first` on
`standup_natural_01` is byte-identical (`cmp`) to T1's `_cf_base.npz`. Boundary counts logged.

### T3 — §3.3–§3.5 wrapper
**Accept:** `--feasibility-first` on `standup_natural_01` runs to completion; NPZ has the three
`ff_*` keys; the printed summary shows retries happened on the frames T1 flagged; byte-identity
still holds with the flag off. Sanity: `ff_floor_pen_cm.max()` strictly below T1's Stage-3
baseline max (any improvement passes T3; the real bar is T5).

### T4 — Full-stack run
Both gate clips: Stage 3 `--feasibility-first` → Stage 4 (plain AND `--continuation 8`) →
Stage 4.5 → probe + render (`outputs/ff_dev/renders/`). Also two no-regression clips through the
same stack WITHOUT the flag (`shovel_leftbucket_02`, `luigi_standProne_03` with its own
per-clip flags): byte-identical Stage-3 outputs expected (flag off = untouched path).
**Accept:** all runs complete, numbers logged per stage.

### T5 — Gate
| metric | bar |
|---|---|
| post-4.5 floor pen (cm), gate clips | **≤ 50% of baseline (13.5→≤6.7, 24.4→≤12.2); ≤2cm = full win** |
| velocity spikes | **0 (side_05's pre-existing 1 must not grow)** |
| self-pen peak | **≤ baseline + 0.5cm** |
| plant slip | ≤ baseline + 1.5cm (report; soft bar — feasibility-first may legitimately move plants) |
| frames still infeasible after retries | report (the certificate) |
| visual render | **no contorted poses / limb snaps at phase boundaries — screenshot verdicts** |
| Stage-3 wall-clock | report; flag if >3× baseline |

Ship bar: bold rows pass on BOTH gate clips + no-regression clips byte-identical. Partial (e.g.
24.4→15cm then plateau): do NOT wire into pipeline; write the per-frame infeasibility
certificate analysis (which frames, which bodies, what depth remained) — that's the measured
"this capture has no near-human feasible version" evidence per clip, publishable on its own.

### T6 — Wrap-up (ship or no-ship)
`wiki/experiments/feasibility-first-v1-gate.md` (+csv), one `wiki/log.md` line, a subsection in
`wiki/concepts/contact-first-ik.md` (mechanism + verdict + flags), plan.md header flipped to
GATED/SHIPPED. If shipped: `FEASIBILITY_FIRST` env knob in `retargetingPipeline.sh` (default
off) + wiki note. Everything uncommitted.

---

## 5. Known risks (pre-answered)

- **Mid-phase relaxation may not cross a basin.** By design (continuity). If T5 shows a clip
  stuck because a WHOLE phase needs a different strategy (constant per-frame residual across a
  phase, retries maxing out uniformly), that's the documented v2: phase-level restart of the
  phase's ENTRY pose + blend-in over a ramp window (the `refine_arm_floor_transitions` pattern).
  Log it, don't build it.
- **Restart noise at boundaries can create a visible snap.** `q_ref=q_prev_frame` regularization
  bounds it; the render check (T5) is the arbiter. If snaps appear, REDUCE sigma / raise
  FF_RESTART_FROM before anything fancier; log each change.
- **Boosted floor/coll weights can degrade tracking on feasible frames** — they can't: the
  wrapper only engages on frames that FAILED the gate; feasible frames take attempt 0
  (byte-identical path).
- **Stage 4 may partially undo Stage-3 feasibility** (it re-optimizes toward the human). That's
  expected and correct — keep-best + floor rows + continuation bound how far it slides back;
  the T5 numbers are measured AFTER Stage 4.5, not at Stage 3.
- **Luigi may be right that no near-human feasible version exists for some captures.** Then the
  retry loop exhausts and the certificate says so, per frame, with depths — a clean negative,
  more useful than a diverging solve. Report it as a result, not a failure of the plan.
- **Runtime**: retries only on violating frames (T1 says ~50–100% of frames on gate clips at
  Stage 3 — could be ~7× ik cost worst case on those clips). Acceptable for a gate; it's a
  ship-decision input.

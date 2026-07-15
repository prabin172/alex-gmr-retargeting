# GMR-baseline week 1 (2026-07-15, branch `gmr-baseline`)

**Verdict: Option A ("contact-aware kinematic polish", `GMR-baseline.md` §3) LIVES — kill-test
clears cleanly, no cherry-picking needed.** Full task-by-task trail: `planLogGMR.md` (repo root).
Plan: `GMR-baseline-plan.md` (repo root, T0–T10). Results writeup for outside readers:
`GMR-baseline-results.md` (repo root).

## What this is

E1+E2+E3 from `GMR-baseline.md` §4's de-risk ladder, in one week: run upstream GMR (fresh clone,
`github.com/YanjieZe/GMR`, unmodified) on its own LAFAN1 benchmark's floor-contact clips, port
our reference-free eval (`eval_ihmc_json.py`) and our Stage-A-smoothing + grounding polish to
Unitree G1 with **zero core-solver-logic changes**, and measure whether the polish transfers.

## Corpus

5 LAFAN1 clips, screened by hip-Z range via GMR's own `load_bvh_file` (not guessed): 3
floor-contact (`fallAndGetUp2_subject2`, `fallAndGetUp1_subject1`, `ground1_subject1` — diverse
failure modes: severe fall, sustained-low fall, sustained crawl) + 2 locomotion controls
(`walk1_subject1` clean, `dance1_subject1` busier with a non-floor crouch dip). Selection detail
+ full hip-Z table: `planLogGMR.md` T2.

## M1 — motivation (GMR raw, out-of-box, on its own benchmark)

| clip | floorPen max | pen% (frames >0.5cm) |
|---|---|---|
| walk1_subject1 (control) | 1.0cm | 0.3% |
| dance1_subject1 (control) | 7.1cm | 1.9% |
| fallAndGetUp2_subject2 | 13.6cm | 47.1% |
| fallAndGetUp1_subject1 | 12.9cm | 38.9% |
| ground1_subject1 | 15.9cm | 90.6% |

Confirms GMR's own stated exclusion ("we do not include motions with complex interaction with
the environment, such as crawling or getting up from the floor") with concrete numbers AND
visual evidence (frame extractions in `planLogGMR.md` T3 show splayed limbs, no weight-bearing
contact, on all 3 floor clips) — not just the paper's claim taken on faith. **Side-finding**:
`n_spikes=0` on every clip including the worst floor clips — GMR's own per-frame differential IK
is smooth even while failing on floor contact. Floor-contact and jitter are orthogonal failure
modes here; the polish story is specifically about the former.

## M2 — polish delta (Stage A smoothing + Z-grounding, ported to G1)

| clip | floorPen: raw→polished | pen%: raw→polished | vMax rad/s: raw→polished |
|---|---|---|---|
| walk1_subject1 | 1.0→**0.7cm** | 0.3%→0.1% | 18.9→**3.3** (5.7x) |
| dance1_subject1 | 7.1→**3.2cm** | 1.9%→0.6% | 47.5→**6.2** (7.7x) |
| fallAndGetUp2_subject2 | 13.6→**4.0cm** | 47.1%→0.5% | 20.4→**4.8** (4.3x) |
| fallAndGetUp1_subject1 | 12.9→**1.1cm** | 38.9%→0.5% | 29.5→**6.1** (4.8x) |
| ground1_subject1 | 15.9→**2.4cm** | 90.6%→0.5% | 37.2→**5.5** (6.8x) |

Improves on BOTH axes, on EVERY clip, controls included. Velocity smoothing factor (4.3–7.7x)
lands in the same range as the Alex-side validation that seeded this whole plan (Luigi's manual
retarget polish, 5.7x, `wiki/log.md` 2026-07-14). Full before/Stage-A/polished 3-stage table:
`planLogGMR.md` T9.

## What "zero core-logic changes" actually meant

- `evaluate()` (`eval_ihmc_json.py`) — imported unmodified; already took model/data/geom_ids as
  arguments, no Alex globals inside. `contacts={}`/`sole_sids={}` (G1 has neither) degrade
  gracefully with no code changes.
- `stage_a()` (`solve_global_trajectory_opt_contactfirst.py`) — imported unmodified; pure
  qpos-level tridiagonal smoothing. Only genuinely robot-specific input (`q_lo`/`q_hi`) was
  already a function argument, not hardcoded.
- `post_process_ground_contactfirst.py`'s `constant`/`perframe` grounding modes — invoked via
  `subprocess` against an unmodified copy, through a minimal temp NPZ (`qpos` only — those two
  modes never touch `contact_flags`/`contact_effector_names`).
- G1's floor-plane geom needed no special exclusion — it's `bodyid=0` (worldbody), already
  caught by the existing `geom_bodyid[g] != 0` filter both eval and grounding already used.

The generalization work was genuinely "port + wire," not "redesign" — supports Option A's
robot-agnostic claim structurally, not just empirically.

## Caveats (don't overclaim)

- **Self-collision numbers on G1 are noise, not signal, this week.** `walk1_subject1` — clean
  short walk — reads 18.2% self-collision incidence, physically implausible. G1's mocap XML
  collision pairs aren't vetted (unlike Alex's). Flagged in `eval_motion.py`'s own footer.
  Deferred to E4 (needs `g1_custom_collision_29dof.urdf` review).
- **Polish fixes clip-level floor calibration + smoothness, NOT per-limb physical plausibility.**
  Frame-by-frame visual comparison (`planLogGMR.md` T9): the same splayed-limb pose that appears
  in the raw motivation figure is STILL visually present after polish — the penetration number
  improves because grounding recalibrates the clip's global floor reference, not because the
  pose itself becomes physically supportable. That's Stage B's job (contact-anchored QP), E4,
  deferred — this week deliberately didn't attempt it (no contact flags from GMR, no G1
  role/support-face map yet).
- `constant` grounding mode was chosen over `perframe` specifically because `perframe`'s "perfect"
  0.0cm floor pen comes from grounding every frame independently, which measurably increases
  root-Z bobbing (+65% peak vertical velocity on the worst floor clip) — `constant`'s large,
  honest improvement with no new artifact was judged the better tradeoff. Full numbers:
  `planLogGMR.md` T8.

## Footguns hit (useful for anyone extending this)

1. **GMR's own `RobotMotionViewer` hard-requires a GLFW display** even with only
   `--record_video` set (`launch_passive()` is unconditional) — fails on any headless machine.
   Worked around with `scripts/g1/gmr_headless_retarget.py`, same retargeting core, offscreen
   `mujoco.Renderer` + `MUJOCO_GL=egl` instead (confirmed working) for video.
2. **This repo has an empty, gitignored `general_motion_retargeting/` leftover directory** at
   its root (pre-dates the repo's divergence from the original GMR clone). If repo root ever
   lands on `sys.path` (e.g. `python -c "..."` run from repo root, where `sys.path[0]` resolves
   to cwd), it silently SHADOWS the real pip-installed GMR package (`__file__` becomes `None`,
   no ImportError). Rule: always run `scripts/g1/*.py` as script files, and when inserting
   sys.path entries for sibling-script imports, insert `<repo_root>/scripts`, never bare
   `<repo_root>`. Full diagnosis: `planLogGMR.md` T1.

## New code (all under `scripts/g1/`, branch `gmr-baseline`)

`gmr_headless_retarget.py` (BVH→G1 pkl, headless), `eval_motion.py` (de-Alexed eval core,
`--ihmc-json`/`--gmr-pkl`), `load_gmr_pkl.py` (pkl↔qpos, xyzw↔wxyz boundary), `polish_gmr_pkl.py`
(Stage A + grounding chain), `render_gmr_pkl.py` (offscreen video from any pkl).

## Next

Per `GMR-baseline-plan.md`'s deferred list: E4 (Stage B contact QP on G1 — needs role/support-face
map + real contact detection) if headroom is wanted beyond this week's clip-level fix; E5
(BeyondMimic tracking-policy delta) is the eventual killer figure but needs GPU scoping first.
Our own FBX floor clips into GMR (this week deliberately used LAFAN1's own clips instead — see
`GMR-baseline-plan.md`'s "Why LAFAN1" section) stays deferred too.

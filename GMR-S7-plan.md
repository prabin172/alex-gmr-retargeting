# GMR-S7 Plan — Paper-Readiness: Close the Eval Holes, Fix perframe, Figures, OmniRetarget

Written by Fable 2026-07-17 for Sonnet to execute. S6 stands and shipped its decision
(Phase A `gmr_contact_fc` = primary method; Phase B `medianlimb` = independent
contribution). S7 is NOT a new mechanism sprint — it is the gap between "beats
baselines" and "submittable paper." If cold, read in order: `GMR-S6-plan.md` (esp.
"Baseline integrity"), `wiki/experiments/gmr-baseline-sprint-s6.md`,
`planLogGMR.md ## S6-DECISION`.

## Why S7 exists — paper-readiness audit (Fable, 2026-07-17)

Audited results docs + Undermind lit report against what a reviewer needs. Four real
holes, in priority order:

1. **`gmr_contact_fc` and `medianlimb` have NO smoothness/skate/fidelity numbers at
   all.** `s6_full_corpus.csv` columns are joint_ok/floorPen/pen/coll/held only. S5's
   contact layer already carried +10-70% jerk vs `gmr_raw`; the S6 per-frame clamp is
   a discrete correction with no ramp — it plausibly adds more. GMR's headline virtue
   is smooth output; a reviewer asks about jerk on page one. Also unmeasured: the
   fidelity (srcDev) cost of the clamp — Prabin's framing is "tracking compromised a
   little, contact physics respected," and the "a little" needs a number.
2. **The floor class is improved, not solved** (fc: 8.08cm floorPen, 6.9 pen%,
   range 9.80cm). The best floor-class mechanism ever tested in this project is
   S6-B1b's `--center perframe` (dev: joint_ok 97.4%, range 2.57cm — better than
   everything else on every floor metric) and it is ONE unresolved bug away from
   corpus-buildable. Fixing that bug is the highest-value technical task left.
3. **No figures.** Zero renders of any S6 variant. The paper's money shot is
   raw-vs-heightfix-vs-ours side-by-side on a crawl/get-up clip (splayed floating
   pose vs limbs actually bearing on the floor).
4. **No OmniRetarget baseline.** Undermind names it the strongest contact-aware
   kinematic competitor (open source). The paper needs either its numbers on our
   clips or a documented, cited exclusion reason — "we didn't try" is not a defense.

NOT in S7 scope (Prabin-level decisions, listed so they aren't forgotten):
BeyondMimic-style policy eval (needs the still-open GPU ask), Table-I→BVH mapping
(author contact is Prabin's call), venue/scope decision (short paper on G1 now vs
2027 Any-Contact full paper — see `paperIdea3.md`).

---

## Phase T1 — smoothness/skate/fidelity eval of the S6 variants  [P0, do first]

### S7-T1a: dev-clip battery
- Clips: the 5 S6 dev clips (walk1_subject1, walk3_subject1, run2_subject1,
  ground1_subject1, fallAndGetUp1_subject1).
- Variants: `gmr_raw`, `gmr_polished`, `gmr_contact` (S5), `gmr_contact_fc`,
  `medianlimb`, plus `stacked` on the 2 clips that have it (walk1, ground1). Pkls
  already exist under `outputs/gmr_baseline/sprint/pkl_s5/` (and `pkl/`/`gmrfix/`/
  `polished/` for the older variants — path conventions are in
  `scripts/g1/sprint_s6_corpus.py`, reuse them).
- Metrics, all existing code, no new mechanisms:
  - `scripts/g1/motion_smoothness.py` — joint_jerk mean/p95, body_jerk mean/p95.
  - `scripts/g1/sprint_s5_metrics.py` — `skate_cm` (held-foot XY drift),
    `fidelity_metrics` (mean pos/ori error vs GMR's own scaled human targets over
    the 12 non-foot table2 bodies).
  - `scripts/g1/eval_motion.py` — vMax, velocity spikes (the week-1 metric, for
    continuity with older tables).
- Fidelity targets note: human targets are variant-independent (they are GMR's
  scaled-human tracking targets per clip, not per variant). Check
  `outputs/gmr_baseline/sprint/human_targets/` first — S1 saved them per clip. If a
  dev clip is missing, regenerate via `gmr_headless_retarget.py --save_human_targets`
  (this does NOT touch the baseline pkls; it only re-derives targets).
- Deliverable: one table in `planLogGMR.md ## S7-T1a` — rows variant×clip, columns
  joint_jerk/body_jerk (each also as %delta vs `gmr_raw`), skate_cm, fidelity
  pos/ori delta vs raw, vMax. RECORD everything; the only decision threshold:
  **if `gmr_contact_fc` body_jerk mean is >50% above `gmr_raw` on any loco dev clip,
  Phase T2 activates.** Otherwise skip T2 entirely and say so in the log.

### S7-T1b: corpus scale
- New `scripts/g1/sprint_s7_smoothness.py`, same resumable pattern as
  `sprint_s6_corpus.py` (skip-if-done rows, per-clip logging, background-safe):
  all 77 clips × {gmr_raw, gmr_polished, gmr_contact, gmr_contact_fc, medianlimb} →
  `outputs/gmr_baseline/sprint/s7_smoothness.csv` (joint_jerk mean/p95, body_jerk
  mean/p95, skate_cm, fidelity pos/ori, vMax).
- This is FK + finite differences — much cheaper than a retarget build; it should
  run in minutes-to-an-hour, not the many-hours S6 builds took. Do not background
  it unless it actually proves slow.
- Class-split summary (34 floor / 43 loco via `s1t4_reclass.csv`, same as always) →
  log `## S7-T1b` with the table. This is the paper's smoothness/fidelity column,
  done.

## Phase T2 — smoothing pass on top of fc  [CONDITIONAL: only if T1a trips the >50% threshold]

- Mechanism: Stage-A tridiagonal smoothing (the exact code `polish_gmr_pkl.py` uses —
  Stage A ONLY, no grounding step, grounding would re-break the clamp's exact floor
  contact) applied to `gmr_contact_fc` output, then ONE full-clip re-clamp pass
  (iterate frames, apply `leg_floor_clamp.clamp_limb` over `CLAMP_TARGETS`, same
  call pattern as the `--floor-clamp` block in `gmr_contact_retarget.py main()`)
  because smoothing WILL reintroduce small penetrations. Order matters:
  smooth-then-clamp, never clamp-then-smooth-and-stop.
- Build as a small standalone driver (e.g. `scripts/g1/smooth_then_clamp.py`), input
  any pkl → output pkl. Do NOT modify `polish_gmr_pkl.py` (it generates the
  `gmr_polished` baseline — Baseline integrity rule applies).
- Gate on the 5 dev clips: jerk delta vs raw back under +50%, AND joint_ok/pen%/range
  within noise of un-smoothed fc (the clamp re-pass should guarantee this — verify,
  don't assume). Standard 2-attempt tuning cap, then log honestly and stop.
- If gate passes: corpus-build the smoothed variant (`gmr_contact_fc_sm`), add to
  both eval CSVs (joint metric + smoothness), log `## S7-T2`. This becomes the
  paper's primary variant ONLY if it also holds the S6 corpus numbers; otherwise fc
  stays primary with jerk reported honestly.

## Phase T3 — root-cause and fix `--center perframe` divergence  [P0, independent of T1/T2]

The prize: dev-clip floor-class joint_ok 97.4% / range 2.57cm — the best floor
numbers of any mechanism tested in this entire project, blocked by one bug.

- **Known facts (from S6-B2, all verified):** `polish_median_limbwise.py --center
  perframe` on walk1_subject1 explodes at frame t=5006: right ankle jumps to world
  Z=0.80m. The frame is a held-segment RELEASE (held True→False at t=5006-5007),
  ramp fully engaged just before, chain qpos inspection shows a broken pose. The
  `--center median` path on the same clip is clean — the bug is in the perframe
  lift path or its interaction with the held-release ramp, not in `clamp_limb`'s
  core (which A5/B3 corpus runs exercised thousands of times cleanly).
- **Debug plan (in order, stop when found):**
  1. Instrument `_perframe_shift` and `_limbwise_pass` for t∈[4995, 5015]: per-frame
     lift value, per-effector `held`, `ramp_age`, `frac`, `onset_xy`, and the `e`
     vector + `|dq|` per `clamp_limb` iteration. The smoothed lift curve itself may
     spike at the release (moving-average window straddling a discontinuity).
  2. Check `onset_xy` staleness: on release, `frac` decays over ramp_frames while
     `onset_xy` still points at the (possibly far-away) onset position — a large XY
     error into the DLS with small damping is exactly the 28°-knee failure shape
     seen in bug (4). Candidate fix: on release, re-anchor the blend target to the
     CURRENT xy at release frame, not the onset xy.
  3. If neither: add a trust region to `clamp_limb` (cap per-iteration `|dq|`, e.g.
     0.15 rad, and cap total per-frame correction) — this is a robustness fix that
     is defensible regardless of root cause, but ONLY ship it in addition to an
     identified root cause, not as a blind band-aid (Prabin will ask why).
- Gate: perframe runs the 5 dev clips with zero divergence (assert max |ankle world
  Z| sane, e.g. < 1.0m for non-jump clips is NOT a valid check — jumps exist; assert
  instead no single-frame body-position discontinuity > 10cm frame-to-frame on any
  watched body), and floor-class dev numbers reproduce the S6-B2 ballpark
  (joint_ok ≥ 95%, range ≤ 4cm on ground1/fallAndGetUp1).
- Log `## S7-T3` with root cause, fix, and gate table.

### S7-T3b: perframe corpus  [only if T3 gate passes]
- Extend `sprint_s6b_corpus.py` (or new s7 sibling) to build+eval `perframe` for all
  77 clips, resumable, background. Eval BOTH batteries: joint metric + range
  (`s7b_full_corpus.csv`/`s7b_range.csv`) and smoothness (append to
  `s7_smoothness.csv`).
- Also rebuild the stacked variant (A-then-perframe) on the 5 dev clips — S6's stack
  used median and was a wash on floor; perframe may compose differently. Dev clips
  only, no corpus stack unless it clearly wins.
- Log `## S7-T3b`. If perframe materially beats fc on the floor class at corpus
  scale, write `## S7-DECISION` laying out the revised method story options
  (fc everywhere / fc+perframe per class / stacked) with the tables — but the final
  method choice is Prabin's; present, don't decide.

## Phase T4 — renders and paper figures  [P0, can run parallel to anything]

- Tools exist: `scripts/g1/render_gmr_pkl.py`, `render_penetration_annotated.py`.
  Read them first — reuse their camera/format conventions; extend only if side-by-side
  tiling is missing (ffmpeg hstack is fine).
- Clips × variants:
  - `ground1_subject1` (sustained crawl — the motivation centerpiece):
    gmr_raw, gmr_heightfix, gmr_contact_fc (+ best T3 variant if it landed).
  - `fallAndGetUp2_subject2` (most severe fall/get-up): same variants.
  - `walk1_subject1` (locomotion control, shows we don't wreck the easy case):
    gmr_raw, gmr_contact_fc.
- Deliverables into `outputs/gmr_baseline/sprint/renders/s7/`:
  1. Full-clip mp4 per clip×variant + one hstacked comparison mp4 per clip.
  2. Stills at hand-picked support moments (crawl mid-stance, get-up push-off,
     walking single-support) — pick frames using the held-mask + penetration data,
     not eyeballing; note chosen frame indices in the log so figures are
     reproducible.
  3. Penetration-annotated versions (the annotated renderer) for at least ground1.
- Log `## S7-T4` with file paths + chosen frames. These are draft paper figures —
  composition polish comes later with Prabin.

## Phase T5 — OmniRetarget baseline or documented exclusion  [P1]

- Time-box: ~half a day of effort. Outcome A (it runs) and outcome B (documented
  exclusion) are BOTH acceptable deliverables; an undocumented "couldn't get it
  working" is not.
- Steps:
  1. Locate the official repo (Undermind cites it as open source; search GitHub for
     "OmniRetarget"). Read README: supported robots (G1?), input format (BVH/AMASS/
     LAFAN1?), license.
  2. Install in a SEPARATE venv/conda env — do not touch the project env (GMR is
     pip-installed editable there; dependency clash risk is real).
  3. Try the 5 dev clips out-of-box. No tuning beyond their own documented flags —
     this is an as-shipped baseline, same integrity rule as GMR (bugs included,
     never fixed).
  4. If retargets land: convert output to our qpos-pkl convention (write the thin
     adapter, document the joint-order mapping carefully — wxyz quaternions, qpos
     layout per project conventions), eval with the standard battery, add rows to
     the dev tables.
  5. If blocked (no G1 support, no BVH path, needs data we don't have, crashes on
     floor clips): write the exclusion memo with exact evidence — README quotes,
     error output, what its paper says it supports. That memo IS the deliverable.
- Log `## S7-T5` either way.

## Phase T6 — torso/waist residual probe  [P1, CONDITIONAL: only if T3/T3b does NOT close the floor class]

- "Close" = floor-class corpus floorPen ≤ ~3cm and range ≤ ~5cm from some S7 variant.
  If perframe gets there, skip T6 entirely.
- If needed: ONE exploratory mechanism, 2-attempt cap — extend the clamp's corrective
  DOF set with the waist joints (waist_yaw/roll/pitch) for torso-penetrating frames
  only, watched body = torso/pelvis links, same `clamp_limb` machinery (the chain
  dict is already parameterized). Root stays frozen (root lift is perframe's job,
  don't duplicate).
- This is a probe, not a commitment: log numbers, stop, present to Prabin.

---

## T-DOC (pre-authorized for this sprint only)
- `planLogGMR.md` `## S7-*` entries per phase, as specified above — numbers + honest
  verdicts, same style as S6.
- New wiki page `wiki/experiments/gmr-baseline-sprint-s7.md` when phases land
  (audit summary, T1 smoothness tables, T3 root cause + fix, T4 figure inventory,
  T5 baseline outcome); one-line update to `wiki/index.md` (S6 line loses CURRENT,
  S7 gets it); `wiki/log.md` one-liners per phase.
- `GMR-baseline-results.md`: add an S7 section when T1b lands (smoothness columns
  complete the S6 story) and update again if T3b changes the floor-class picture.
- **Backfill rule (Prabin, 2026-07-17): while working, if Sonnet finds anything
  shipped-but-undocumented from earlier sprints (a script with no wiki/log mention,
  a result CSV nothing references, an eval convention that exists only in code),
  document it in the appropriate existing page/log — briefly, where it belongs, no
  new pages beyond the S7 page above.**
- No other docs. No commits (standing rule: never git add/commit/push unprompted).

## Standing rules (unchanged from S5/S6 — read GMR-S6-plan.md "Baseline integrity" in full)
- `gmr_raw`/`gmr_heightfix`/`gmr_polished` are fixed comparison targets, generated by
  as-shipped code, bugs included. Never fix or "improve" their generators
  (`gmr_headless_retarget.py`, `polish_gmr_pkl.py`). Same rule extends to
  OmniRetarget in T5.
- Held-mask convention (debounced contact_flags AND speed <0.05), class split from
  `s1t4_reclass.csv` (34 floor / 43 loco), eval always on OUR vetted collision model
  (`g1_model_setup.py::load_g1_model_with_vetted_collision_and_floor`).
- 2-attempt tuning cap per mechanism; honest negatives logged, never silently
  dropped; gate fails stop the phase, no silent continuation to corpus scale.
- Corpus builds: resumable, skip-if-exists, background with per-clip logging.
- Order of execution: T1a → (T2 if tripped) and T3 → T3b can interleave with T1b;
  T4 anytime after its input variants exist; T5 independent; T6 last and only if
  triggered. If wall-clock is tight, T1a + T3 + T4 are the paper-critical trio.

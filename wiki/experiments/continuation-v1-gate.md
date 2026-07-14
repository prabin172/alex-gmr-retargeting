# continuation-v1 gate (2026-07-14, branch `p0-grounding`)

**Verdict: does NOT clear the ship bar. Not wired into `retargetingPipeline.sh` defaults.
Code stays in `solve_global_trajectory_opt_contactfirst.py`, opt-in via `--continuation N`
(default 0, verified byte-identical no-op).** Full build/debug trail: `planLog.md` (repo root,
sections "T2/T3", "T4", "T6"). Plan: `plan.md` (repo root, still marked ACTIVE — this gate is
its first result, not its closure; see "Open follow-up" below).

## Mechanism (recap)

Extra Stage-B passes after the normal solve, each warm-started from the previous pass's best
iterate: floor rows' allowed penetration shrinks PER-FRAME from that frame's own pass-0
penetration toward 0 (`eps_k(t) = pen0(t)·(1−k/K)`), the floor-row slack penalty hardens
1000→1e5 geometrically, and tracking is relaxed only in violating frame-windows × limbs (never
trunk/root — Stage B's decision variables are actuated joints only). Cross-pass keep-best is
lexicographic (velocity spikes, then self-collision beyond a 2cm gate, then floor penetration,
then slip+floor-err, then tracking) and seeded with the plain pass-0 result, so it can never
ship worse than today's solve.

## Gate clips and result

| clip | continuation engaged? | floor_pen_max: pass 0 → shipped | selfpen_over | spikes |
|---|---|---|---|---|
| `standup_natural_01` | **No — see below** | 13.77 → 13.77 (unchanged) | n/a | 0 |
| `standup_side_05` | **No — see below** | 24.44 → 24.44 (unchanged) | n/a | 1 (pre-existing, not introduced) |
| `luigi_standSupine_08` | **Yes** | 3.56 → **2.68cm (−25%)** | 0.20 → **0.01cm (−95%)** | 0 → 0 |

Only 1 of 3 clips shows any effect — below the plan's ≥2/3 ship bar.

## Why 2 of 3 clips show zero effect — NOT a continuation bug

`standup_natural_01` and `standup_side_05` have never run with `--floor-collision on` in the
shipped pipeline (their `CLIPS[]` entries have empty `go_extra`; only `luigi_standSupine_08` has
`--floor-collision on` in production, per `retargetingPipeline.sh`'s own comment: *"validated on
1 clip only so far — opt-in pending corpus validation"*). Continuation needs floor rows to act
on, so testing it required forcing `--floor-collision on` for all three gate clips (a deliberate,
logged deviation — see `planLog.md`'s pre-T1 note).

Diagnosed with a control experiment: plain Stage B (no continuation), `--n-outer 20` (over 3× the
pipeline default) on `standup_side_05` — **every one of 20 outers scored worse than the warm
start**, oscillating 24.6–33cm, never trending down. Same pattern on `standup_natural_01` at the
default `n_outer=6`. This is the SCA-oscillation weakness `stage_b`'s own keep-best docstring
already documents, just never exercised on these two clips before (floor rows were always off for
them). **Continuation cannot rescue a base solve that oscillates instead of converging** — the
homotopy schedule only helps a solver making correct-direction progress that can't close the
whole gap in one linearization; it does nothing if every individual pass is non-convergent to
begin with.

## Where it DID work: `luigi_standSupine_08`

The one clip already shipped with `--floor-collision on` — its Stage B genuinely converges.
Continuation ran 2 passes before stalling:

| pass | floor_pen_max | selfpen_over | kept as best? |
|---|---|---|---|
| 0 | 3.56cm | 0.20cm | seed |
| 1 | 1.87cm | 1.24cm | **No** — self-collision regressed, loses lexicographically despite better floor number |
| 2 | 2.68cm | 0.01cm | **Yes** — beats pass 0 on self-collision, even though its own floor number is worse than pass 1's (rejected) result |

Ships pass 2: floor pen down 25%, self-collision headroom essentially eliminated, 0 spikes
throughout. This is the safety design working exactly as intended — it refused the
better-looking-on-one-axis pass 1 because it cost self-collision quality, and shipped the pass
that improved overall kinematic quality.

**Known imprecision** (not a correctness bug): the stall detector compares each pass's own raw
floor pen against the previous pass's raw floor pen, not the kept-best trend — so it flagged
"stalled" at pass 2 by comparing against pass 1's (rejected) better raw number. A kept-best-aware
stall metric might have continued further; not fixed, given the scope call below.

## Scope decision

Not extended further this session. The real blocker on 2/3 gate clips is the pre-existing
Stage-B SCA-oscillation-under-floor-collision problem — a separate, likely bigger investigation
(per-clip trust-region tuning, a different linearization order, or simply confirming
`--floor-collision` isn't safe corpus-wide, matching the pipeline's own pending-validation
comment). Continuation-v1 as designed is a polish mechanism for an already-converging base solve,
not a fix for a non-converging one, and conflating the two would have meant chasing the wrong
problem.

## Open follow-up (unbuilt)

- Investigate the `standup_natural_01`/`standup_side_05` Stage-B oscillation under
  `--floor-collision on` directly — separate from continuation.
- If that's fixed, re-run this exact gate — continuation's mechanism doesn't need to change, it
  just needs a converging base solve to act on.
- Consider a kept-best-aware stall metric (compare `Pk` against the LAST KEPT pass's pen, not
  the last pass's raw pen) — minor, would only extend how long continuation keeps trying.

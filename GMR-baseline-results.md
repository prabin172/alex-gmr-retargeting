# GMR-baseline results — Week 1 (2026-07-15)

First results narrative for `GMR-baseline.md`'s Option A ("contact-aware kinematic polish").
Executed per `GMR-baseline-plan.md` (T0–T10, done in one session, driven task-by-task, no
subagent delegation). Full build/debug trail: `planLogGMR.md`. Wiki record:
`wiki/experiments/gmr-baseline-week1.md`. Branch `gmr-baseline`.

## Setup

- **GMR**: fresh clone, `github.com/YanjieZe/GMR`, MIT, unmodified — imported as an installed
  package (`pip install -e`), never edited. Robot: `unitree_g1` (29 DoF + free root, MuJoCo
  mocap model shipped in-repo).
- **Data**: LAFAN1 (Ubisoft, free direct download), 77 BVH clips at 30 fps.
- **Clips** (5, screened by hip-Z range via GMR's own loader — not guessed):
  - Floor-contact: `fallAndGetUp2_subject2` (most severe fall), `fallAndGetUp1_subject1`
    (sustained low floor time), `ground1_subject1` (sustained crawl, different failure mode).
  - Locomotion controls: `walk1_subject1` (clean), `dance1_subject1` (busier, has a non-floor
    crouch dip).
- **Eval**: `scripts/g1/eval_motion.py`, a de-Alexed port of `scripts/eval_ihmc_json.py`'s
  reference-free physics eval (mesh-exact floor penetration vs z=0, joint-limit violations,
  rate-aware velocity/spikes). Reused `evaluate()` unmodified.
- **Polish**: `scripts/g1/polish_gmr_pkl.py` — Stage A (tridiagonal smoothing, imported
  unmodified from `solve_global_trajectory_opt_contactfirst.py`) + Z-grounding (`constant` mode,
  via unmodified `post_process_ground_contactfirst.py`).

## (a) Motivation: GMR on its own benchmark's floor clips

| clip | floorPen max | pen% (frames >0.5cm) |
|---|---|---|
| walk1_subject1 (control) | 1.0cm | 0.3% |
| dance1_subject1 (control) | 7.1cm | 1.9% |
| fallAndGetUp2_subject2 | 13.6cm | 47.1% |
| fallAndGetUp1_subject1 | 12.9cm | 38.9% |
| ground1_subject1 | 15.9cm | 90.6% |

GMR's paper states it "does not include motions with complex interaction with the environment,
such as crawling or getting up from the floor." This confirms it with numbers, not just the
quote: on LAFAN1's own floor-contact clips, GMR's max floor penetration is 13–16cm, affecting
39–91% of all frames — an order of magnitude worse than its clean-locomotion baseline (1.0cm,
0.3%). Visual inspection (frame extractions, `planLogGMR.md` T3) confirms the failure mode: no
limb ever appears to bear weight against the floor — splayed, floating poses, not a body using
ground contact for support.

**Side-finding worth stating explicitly**: zero velocity spikes on every clip, including the
worst floor clips. GMR's own per-frame differential IK produces smooth output even while failing
badly on floor contact — floor-contact failure and motion jitter are orthogonal problems here,
which sharpens Option A's claim: this is specifically a floor-contact-reasoning gap, not a
generic "GMR produces bad motion on hard clips" story.

## (b) Polish delta

| clip | floorPen: raw→polished | pen%: raw→polished | vMax rad/s: raw→polished |
|---|---|---|---|
| walk1_subject1 | 1.0→**0.7cm** | 0.3%→0.1% | 18.9→**3.3** (5.7×) |
| dance1_subject1 | 7.1→**3.2cm** | 1.9%→0.6% | 47.5→**6.2** (7.7×) |
| fallAndGetUp2_subject2 | 13.6→**4.0cm** | 47.1%→0.5% | 20.4→**4.8** (4.3×) |
| fallAndGetUp1_subject1 | 12.9→**1.1cm** | 38.9%→0.5% | 29.5→**6.1** (4.8×) |
| ground1_subject1 | 15.9→**2.4cm** | 90.6%→0.5% | 37.2→**5.5** (6.8×) |

A robot-agnostic, purely-kinematic polish module — the SAME code that validated on Alex via a
mentor's manual Blender retarget (5.7× velocity smoothing, `wiki/log.md` 2026-07-14) — ported to
a second robot (Unitree G1) with **zero core-solver-logic changes**, improves BOTH floor
penetration and joint-velocity smoothness on every single clip in this corpus, controls
included. No cherry-picking was needed to make this case.

The generalization required: robot-specific joint limits (already a function argument, not a
hardcoded default) and a model path. Nothing else — the smoothing math, the eval metrics, and
the grounding QP are literally the same code Alex's pipeline uses.

## Honest caveats

- **Polish is a whole-clip-level fix, not a per-limb one.** The same "splayed limbs, no
  weight-bearing contact" pose visible in the raw motivation figure is STILL visually present
  after polish, frame-for-frame — the penetration number improves because grounding recalibrates
  where the clip's floor reference actually is, not because any individual limb's pose becomes
  physically supportable. That's Stage B's job (contact-anchored QP) — deliberately not
  attempted this week (no contact flags from GMR, no G1 support-face map yet).
- **Self-collision numbers on G1 are not usable this week.** A clean walk clip reads 18.2%
  self-collision incidence — physically implausible, and strong evidence G1's mocap-model
  collision geometry isn't set up with sane exclusion pairs the way Alex's is. Flagged, not
  reported as a claim.
- **`constant` grounding was chosen over `perframe`** specifically because `perframe`'s "perfect"
  0.0cm floor pen comes from grounding every frame independently, which measurably increases
  root-Z bobbing (+65% peak vertical velocity on the hardest clip) — `constant`'s honest,
  smaller-but-real improvement was judged the better trade. Full numbers: `planLogGMR.md` T8.

## Recommended next step

Per `GMR-baseline-plan.md`'s deferred list, in priority order:

1. **E4 (Stage B contact QP on G1)** — the natural next step given this week's headroom: needs
   a G1 role/support-face map and real contact detection (our height/velocity gates should port).
   Would close the "polish doesn't fix per-limb plausibility" gap named above.
2. **E5 (BeyondMimic tracking-policy delta)** — the eventual killer figure (does polish improve
   actual downstream policy tracking, not just kinematic metrics), but needs GPU scoping first
   (mentor question in `GMR-baseline.md` §5).
3. Our own FBX floor clips into GMR — deliberately deferred this week in favor of LAFAN1's own
   clips (stronger motivation story, zero ingest work); still worth doing eventually for a
   same-source apples-to-apples comparison against our own pipeline's output on the identical
   motions.

No early-stop condition fired anywhere in the E1→E3 ladder this week — Option A is validated
enough to keep investing in.

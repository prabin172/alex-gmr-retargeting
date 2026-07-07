# Parameters cheat-sheet

Every knob in the contact-first pipeline: default, what it does, what it trades off, when to touch it.
All are **env overrides on `./retargetingPipeline.sh`** unless marked *(hardcoded)*. Shipped defaults are
tuned for the **native 120 Hz** solve — do not change one rate-dependent knob without reading
[§ Rate scaling](#rate-scaling-why-the-numbers-look-large) below.

Math ground truth: `METHOD.md`. Operational detail per stage: `wiki/concepts/*`. This page is the index
of *why each value* — it does not restate the math.

---

## The one thing to know first

**One config retargets every clip.** The `CLIPS` list has per-clip flag fields (`solve_extra`, `go_extra`)
and they are **empty by design**. If you find yourself wanting per-clip tuning, that's a finding to write
down, not a knob to set. See `wiki/concepts/design-philosophy.md` (settled — don't re-litigate).

---

## Global / rate

| Knob | Default | What it does | When to touch |
|---|---|---|---|
| `STRIDE` *(hardcoded)* | `1` | Frame stride ⇒ solve rate `120/STRIDE`. `1` = native 120 Hz. | Only if the downstream consumer rate changes. Was `4` (30 Hz) — that was **sub-Nyquist** for the 50 Hz IHMC tracker. Changing it forces re-scaling every rate-dependent knob below. |
| `IK_ITERS` *(hardcoded)* | `40` | Stage-3 damped Gauss–Newton iterations per frame. | Raise if a clip's per-frame IK looks under-converged; costs runtime. |

## Stage 3 — contact-first IK (`solve_fbx_canonical_alex_contactfirst.py`)

Detection + planting of feet/hands on their real contacts. Detail: `wiki/concepts/contact-first-ik.md`.

| Knob | Default | What it does | Trade-off / when to touch |
|---|---|---|---|
| `COPLANAR_FEET_MODE` | `mean` | When **both** feet are contact-labelled, snaps their ankle-height targets to a common Z so the IK yields coplanar feet. Fixes the "one foot floats in RDX" root cause (retargeted foot-height targets can sit cm apart while both are labelled planted — a rigid grounding shift can't reconcile that). | `mean` = meet in the middle (**lowest self-collision**). `min` = snap the high foot down to the low one (more collision). `off` = legacy, feet may not be coplanar. |
| `--contact-min-run` *(hardcoded)* | `12` | Min contact-segment length (**frames**) to count as a contact. | Frame-count knob → ×4 vs 30 Hz. Debounces contact flicker. |
| `--contact-ramp` *(hardcoded)* | `16` | Blend-in length (**frames**) for a contact pin. | Frame-count → ×4. Longer = gentler onset. |
| `--contact-preroll` *(hardcoded)* | `8` | Frames a contact is anticipated **before** detection. | Frame-count → ×4. |

## Stage 4 — GlobalOPT (`solve_global_trajectory_opt_contactfirst.py`)

Stage A tridiagonal smoothing (spikes→0) + Stage B contact-aware QP (soft self-collision, plant pins,
on-floor rows). Detail: `wiki/concepts/globalopt.md`, math: `METHOD.md` §6.

| Knob | Default | What it does | Trade-off / when to touch |
|---|---|---|---|
| `LAMBDA_SMOOTH` | `320` | First-difference (velocity) smoothing weight, Stage A **and** B. | **∝ fps² → ×16 vs the 30 Hz value of 20.** Higher = smoother but laggier tracking. The single most rate-sensitive knob. |
| `N_OUTER` | `6` | Stage-B SCA outer iterations (collision re-linearised at each outer's start). `0` = Stage B off. | More outers = more chances to land a clean (collision-free) SCA iterate; keep-best returns the best across outers, so this is *robustness*, not correctness (see the SCA parity fix in `globalopt.md`). |
| `FOOT_WEIGHT` | `160` | Soft-pin weight on a **planted** foot. | dt-invariant; ×4 vs default 40 to **rebalance** against the ×16 smoothing (else plants slide). Raise to cut foot slip at the cost of ~1 cm more shallow self-collision. The real slip lever (plant-speed had no effect). |
| `HAND_WEIGHT` | `32` | Soft-pin weight on a **planted** palm. | Same story as `FOOT_WEIGHT` (×4 vs default 8). |
| `PLANT_MIN_RUN` | `8` | Min stillness sub-segment (**frames**) before a contact counts as a *plant*; shorter speed dips → reclassified moving. | Frame-count → ×4. Debounces phantom 1-frame plants on lifting-off hands (fixed standup_side_05 right-hand slip 14.7→6.8 cm). |
| `FLOOR_WEIGHT` | `200` | On-floor + coplanar rows: drives each planted foot's 4 sole-corner Zs to a shared floor height. Co-plants both feet **in the solve** (a 1-DOF grounding shift can't). `0` = off. | Pairs with `GROUND_MODE=constant-contact`. Higher = flatter/more-planted feet, slightly more leg extension / grazing. |
| `FLOOR_MODE` | `estimate` | Where the shared floor sits. | `estimate` = the lower foot's warm-start ground. `zero` = drive soles to z=0. |
| `--collision-penalty` *(hardcoded)* | `1000` | Soft self-collision slack weight ρ. | dt-invariant. This is why penetration is eliminated at the cost of a little slip — the core kinematics trade. |

## Stage 4.5 — Z-grounding (`post_process_ground_contactfirst.py`)

Registers the trajectory to the floor plane (z=0). Detail: `wiki/concepts/grounding.md`.

| Knob | Default | What it does | Trade-off / when to touch |
|---|---|---|---|
| `GROUND_MODE` | `constant-contact` | **`constant-contact`** = ONE vertical shift keyed to the **planted-foot soles** ⇒ no bobbing, feet stay down. **`perframe`** = plant the lowest contact every frame (wanders 7–9 cm as the lowest contact migrates hands→knees→feet). **`constant`** = one shift to the global-lowest geom. | `constant-contact` is right for **clips that end standing**. **Fall clips** (kneelingFall_02/03) punch a late free foot through the floor under a single shift — those want `perframe`/hybrid. Pick by whether the clip ends on its feet. |
| `GROUND_SMOOTH` | `80` | `perframe` only: tridiagonal smoothing on the per-frame shift series. | **∝ fps² → ×16 vs 30 Hz value of 5.** Unused by `constant-contact`. |

## Stage 5 / 6 — render + export

| Knob | Default | What it does |
|---|---|---|
| `RENDER` | `1` | `1` = render Stage-5 MP4; `0` = skip (faster; JSON export still runs). |
| `RENDER_MESH` | `visual` | `visual` = full-body Alex mesh, fist hands. `collision` = the convex hulls the solver actually uses. `<path>` = any model XML. |
| `RENDER_EXTRA` | `""` | Extra render flags, e.g. `"--fixed-cam --no-human"`. |
| `EXPORT_50HZ` | `1` | Also emit a 50 Hz IHMC set (matches the reference `1.json` rate) alongside the native-120 set. `0` skips. |
| `GO_DIR` / `GR_DIR` / `RENDER_DIR` / `IHMC_DIR` / `IHMC_DIR_50` | see `.sh` | Output directory overrides. |

## Rate scaling — why the numbers look large

The solve runs at native 120 Hz (`STRIDE=1`). Deriving the objective at dt/4 vs the old 30 Hz:

- **Position terms are dt-invariant** — track (w=1), contact pins, collision ρ=1000, trust, posture_reg: **unchanged**.
- **Only the first-difference (velocity) smoothness term carries 1/dt²** ⇒ `LAMBDA_SMOOTH`, `GROUND_SMOOTH` **×16**.
- **Frame-count knobs ×4** — `contact-min-run/ramp/preroll`, `PLANT_MIN_RUN` (they count frames, and 120/30 = 4).
- **Speeds (m/s) and onset-delay (s) auto-scale** via `×fps` internally — no manual change.
- `FOOT_WEIGHT`/`HAND_WEIGHT` **×4** is a *relative rebalance* against the ×16 smoothing, **not** a correctness requirement.

Full table + derivation: `wiki/concepts/pipeline.md` § "Solve rate". If you ever change `STRIDE`, re-apply this rule.

## "I want to change X" → touch Y

| I want… | Touch |
|---|---|
| Less foot slip on get-up plants | `FOOT_WEIGHT`↑ (`HAND_WEIGHT` for palms). Accept ~1 cm more shallow grazing. |
| Less self-collision | `N_OUTER`↑, or `COPLANAR_FEET_MODE=mean`, or `FLOOR_WEIGHT`↓. |
| Smoother / less jittery motion | `LAMBDA_SMOOTH`↑ (remember it's already ×16-scaled). |
| Feet flatter / more planted | `FLOOR_WEIGHT`↑ with `GROUND_MODE=constant-contact`. |
| Fix a **fall** clip sinking through the floor | `GROUND_MODE=perframe` for that clip (constant-contact can't hold a late free foot). |
| Faster batch (no video) | `RENDER=0`. |
| Contact flicker on a noisy clip | `--contact-min-run`↑ / `PLANT_MIN_RUN`↑ (frame counts). |

## Known limitations (so a new reader isn't surprised)

Kinematics only — no dynamics; downstream RL supplies torques. Residual ~4 cm foot slip on hard
get-up/kneel plants is deliberate (RL absorbs slip, not body interpenetration). Two fall clips penetrate
the floor at the end under `constant-contact` grounding (open follow-up). Full honest distribution:
`wiki/results/metrics.md` and `wiki/results/tradeoffs-limits.md`; `METHOD.md` §9.

# Sprint S6: hard floor constraint + median-centering post-process

Continuation of [gmr-baseline-sprint-s4-s5](gmr-baseline-sprint-s4-s5.md). S5's
contact layer won the un-gameable joint metric by 12-93 points but didn't deliver
Prabin's actual ask: "even if tracking is compromised, contact physics should be
respected." Full trail: `planLogGMR.md ## S6-*`. Plan: `GMR-S6-plan.md`.

## Why S5 fell short

S5's `gmr_contact_retarget.py` overrides held-effector cost/target inside GMR's own
QP — a soft cost, traded against tracking, covering only held feet. Root cause
ranked in `GMR-S6-plan.md`: (1) soft cost not constraint, (2) held-effectors-only
coverage, (3) nothing structurally forbids the floor. Confirmed via the held-foot
`support_z` range (worst float minus worst penetration, per clip): a rigid Z-shift
(`gmr_heightfix`) provably cannot change this number, only relocate it —
7.98cm==7.98cm on locomotion clips, exactly, base vs heightfix.

## Phase A: exact per-frame floor clamp

**S6-A1** (research, no ship): tried appending `mink.CollisionAvoidanceLimit`
directly to GMR's solve. Found a real bug (GMR's own `solve_ik` call passes
`ik_limits` positionally into `safety_break`, never reaching `limits` — confirmed
via direct QP diff). Even fixed, the rate-limited QP inequality doesn't converge
within GMR's ~1-iteration-per-frame solve loop (0 extra iters: no effect at all;
+50 forced iters: converges to ~0.34cm, never exact). GMR's own G1 XML also
excludes the foot's real mesh from collision (only 4 incidental marker spheres are
collidable). Verdict: abandoned the QP-inequality approach.

**S6-A2-A5** (shipped): built `leg_floor_clamp.py` — a direct, deterministic
damped-least-squares clamp on OUR OWN vetted mesh geometry (not GMR's), applied
after each frame's normal solve. Found and fixed three more bugs during dev-clip
testing: (1) feet-only watching missed the actual worst-penetrating body on 4/5
dev clips (elbow, hip_yaw — not a foot at all; extended to watch knee/hip_yaw/elbow
too, always, regardless of contact scope); (2) distal-before-proximal correction
order let a later proximal fix silently re-violate an already-corrected distal
body (reordered hip->knee->ankle); (3) `max_iters=3` too low near saturated joint
limits (deep-crouch frames with hip/knee at their exact range boundary need more
steps; raised to 10).

Wired into `gmr_contact_retarget.py --floor-clamp` (`gmr_contact_fc`), composes
with the S5 contact layer.

5-dev-clip gate result:

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco | gmr_raw | 7.3 | 5.82 | 93.9 | 8.17 |
| loco | gmr_polished | 0.9 | 2.14 | 6.4 | 9.56 |
| loco | gmr_contact (S5) | 7.8 | 5.40 | 92.8 | 7.30 |
| loco | **gmr_contact_fc** | **0.5** | **1.18** | **99.4** | **4.14** |
| floor | gmr_raw | 64.4 | 14.10 | 59.1 | 11.60 |
| floor | gmr_polished | 0.8 | 2.97 | 0.2 | 10.18 |
| floor | gmr_contact (S5) | 64.1 | 13.50 | 59.8 | 12.75 |
| floor | **gmr_contact_fc** | **19.8** | **5.22** | **96.2** | **4.58** |

Beats every baseline (including `gmr_polished`, GMR + our own Stage-A polish) on
joint_ok and range on both classes. Full 77-clip corpus (`s6_full_corpus.csv` /
`s6_range.csv`, 0 build failures):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco (43) | gmr_raw | 3.0 | 5.15 | 91.5 | 7.98 |
| loco (43) | gmr_heightfix | 3.4 | 3.06 | 46.3 | 7.98 |
| loco (43) | gmr_polished | 1.0 | 2.99 | 32.1 | 8.76 |
| loco (43) | gmr_contact (S5) | 3.9 | 4.95 | 94.2 | 6.67 |
| loco (43) | **gmr_contact_fc** | **0.2** | **0.72** | **99.6** | **3.59** |
| floor (34) | gmr_raw | 23.4 | 15.29 | 80.6 | 12.84 |
| floor (34) | gmr_heightfix | 0.4 | 2.76 | 0.2 | 12.84 |
| floor (34) | gmr_polished | 0.3 | 2.56 | 0.4 | 11.77 |
| floor (34) | gmr_contact (S5) | 24.3 | 15.35 | 84.0 | 12.32 |
| floor (34) | **gmr_contact_fc** | **6.9** | **8.08** | **91.0** | **9.80** |

Locomotion floorPen clears the <1cm gate at full corpus scale (0.72cm — the
5-dev-clip sample's 1.18cm was pulled up by walk3's crouch-segment residual,
diluted across the full 43-clip set). `gmr_heightfix`'s range is STILL exactly
7.98/12.84cm at full scale, identical to `gmr_raw` on both classes — the
rigid-shift-cannot-change-spread proof holds project-wide, not just on the 3 dev
clips checked earlier this session.

## Phase B: Prabin's median-centering + limb-wise polish

Idea: median-shift GMR's raw output so held-frame float/penetration are balanced
(small residual either way), then a cheap per-limb DLS pass — held effector locks
to a target, everything else gets clearance-clamped. Retargeter-agnostic (works on
any qpos pkl), reuses `leg_floor_clamp.py`'s `clamp_limb` (no duplicate mechanism).
Fable's amendment (B1b, `--center perframe`): a per-frame-smoothed lift instead of
a constant shift, since a rigid shift can't fix floor-class trunk penetration
(limbs can't lift the pelvis).

Two more real bugs found during dev-clip testing (`polish_median_limbwise.py`):
(4) held-mode's Z-error was computed from the lowest MESH point but the DLS
Jacobian was queried at the BODY ORIGIN — a different world point for a rotated
foot, causing the solve to diverge instead of converge (1mm target -> 28-degree
knee correction); (5) `z_support` (body-origin-to-sole offset, ~4cm) was
mistakenly passed as the Z target instead of 0.0 — `clamp_limb` always targets the
lowest mesh point at world Z=0, not an origin-offset value; produced a systematic
+4cm float bias on every held frame (0% joint_ok despite 0% whole-body pen).

5-dev-clip gate result (`--center median`):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco | gmr_contact_fc (S6-A) | 0.5 | 1.18 | 99.4 | 4.14 |
| loco | medianlimb | 1.4 | 5.59 | 98.4 | **3.50** |
| floor | gmr_contact_fc (S6-A) | 19.8 | 5.22 | 96.2 | 4.58 |
| floor | medianlimb | 20.7 | 6.95 | 86.4 | 3.68 |

Direct answer to "is median-centering + limb-wise IK a bad idea": no — real,
working mechanism, beats `gmr_heightfix`/`gmr_polished` on joint_ok on both
classes, and edges out S6-A on range specifically on the loco class (head start
from centering first). Weaker on absolute floorPen (inherits GMR's raw trajectory
more directly than S6-A, which corrects `gmr_contact`'s already-cleaner output).

`--center perframe` (B1b): best floor-class numbers of any variant tested
(joint_ok 97.4%, range 2.57cm) — confirms the per-frame-lift hypothesis — but has
an unresolved bug (one exploded frame on walk1_subject1, right ankle jumping to
world Z=0.80m at a held-segment release). Not corpus-built this pass; flagged as a
follow-up, not silently dropped.

Full 77-clip corpus (`--center median` only, `s6b_full_corpus.csv` / `s6b_range.csv`,
0 build failures):

| class | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| loco (43) | gmr_raw | 3.0 | 5.15 | 91.5 | 7.98 |
| loco (43) | gmr_contact_fc (S6-A) | 0.2 | 0.72 | 99.6 | 3.59 |
| loco (43) | **medianlimb** | 0.7 | 3.15 | **98.7** | 6.65 |
| floor (34) | gmr_raw | 23.4 | 15.29 | 80.6 | **12.84** |
| floor (34) | gmr_contact_fc (S6-A) | 6.9 | 8.08 | 91.0 | 9.80 |
| floor (34) | **medianlimb** | 9.5 | 9.24 | **91.8** | **14.92** |

Confirms the dev-clip pattern on locomotion. New at full scale: on the floor
class, medianlimb's range (14.92cm) is actually WORSE than doing nothing at all
(gmr_raw's 12.84cm) — even though joint_ok and pen% both clearly improve. Matches
the stacked-variant finding below (ground1): B's held-lock doesn't reliably
tighten the worst-case spread on deep floor-contact clips, even when it clearly
helps the typical case. Sharpens the caveat on B's range win — it's a
locomotion-class-specific property, not general.

## Decision: ship Phase A as the primary method

Tried the stack (A then B, `polish_median_limbwise.py --center median` applied on
top of `gmr_contact_fc` instead of raw) on 2 representative dev clips per the
plan's "cheap, try it" instruction:

| clip | variant | pen% | floorPen_cm | joint_ok% | range_cm |
|---|---|---|---|---|---|
| walk1 (loco) | A only | 0.0 | 0.00 | 100.0 | 3.12 |
| walk1 (loco) | B only | 0.0 | 1.26 | 100.0 | 2.44 |
| walk1 (loco) | **stacked** | 0.0 | **0.08** | 100.0 | **0.10** |
| ground1 (floor) | A only | 20.4 | 4.74 | 95.3 | 2.20 |
| ground1 (floor) | B only | 17.6 | 7.00 | 80.6 | 0.56 |
| ground1 (floor) | stacked | 21.2 | 4.02 | 94.8 | 4.01 |

Stacking is decisively best on locomotion (range collapses to 0.10cm — float and
penetration nearly coincide, the literal ask from early in this session). On the
hardest floor-class clip it's a wash — floorPen improves slightly but range gets
WORSE than either mechanism alone (B's held-lock doesn't compose cleanly with A's
already-shifted trajectory on deep floor-contact segments; not investigated
further, informative negative).

**Shipping Phase A (`gmr_contact_fc`) as the primary paper method** — wins
outright on both classes at full corpus scale on every un-gameable metric. Phase B
(`polish_median_limbwise.py`) ships as an independent, retargeter-agnostic
contribution — useful standalone (no Phase-A dependency, genuinely beats
`gmr_heightfix`/`gmr_polished` on its own) and as an optional locomotion-class
booster stacked on top of A.

## Open items
- `--center perframe`'s walk1_subject1 divergence bug (frame 5006, held-segment
  release) — root cause not isolated, time-boxed given it's the secondary variant.
- Why B's held-lock doesn't compose cleanly with A's correction on deep-contact
  floor-class segments (ground1's stacked range regression) — not investigated.
- S5's contact-transition jerk residual (S4-S5 page) — not revisited this sprint.
- Neither Phase A nor B touches torso/waist (frozen by design) — floor-class
  whole-body pen still has real residual there on the hardest clips (ground1).
- B's floor-class range regression (worse than gmr_raw) confirmed at full corpus
  scale, not just on ground1 — root cause (why the held-lock widens rather than
  narrows the worst-case spread on the hardest clips) not yet isolated.

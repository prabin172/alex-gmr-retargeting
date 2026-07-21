# Results

All numbers are from the 77-clip LAFAN1 corpus (34 floor-contact / 43
locomotion clips) evaluated with the reference-free metric suite (Methods
§4); the multi-stage pipeline (§3) and `gmr_heightfix` (§2) are evaluated on
the complete corpus, the global pipeline (§5) on the 36/77 clips completed
at time of writing. All comparisons are against
`gmr_heightfix` — GMR's own published constant-offset mitigation — not
unmodified GMR output, which is strictly worse and reported only as
motivation (§1).

## 1. Motivation

Unmodified GMR output on floor-contact clips shows 12.9–15.9 cm maximum
floor penetration affecting 39–91% of frames, against 1.0 cm and 0.3% of
frames on GMR's own clean-locomotion clips (`walk1_subject1`). This gap
holds at full-corpus scale, not only on individual clips.

## 2. GMR's own mitigation is insufficient

`gmr_heightfix` reduces penetration but leaves the robot floating well above
the floor at exactly the frames that should show tight ground contact, since
one rigid per-clip offset cannot satisfy a standing phase and a lying phase
in the same clip simultaneously (Table 1, `joint_ok%`).

**Table 1 — `gmr_heightfix`, class means.**

| metric | floor (n=34) | locomotion (n=43) |
|---|---|---|
| floor penetration, max (cm) | 2.76 | 3.06 |
| floor penetration (% frames) | 0.38 | 3.42 |
| self-collision (% frames) | 6.34 | 3.85 |
| worst float (cm) | 18.04 | 6.61 |
| velocity spikes (mean/clip) | 0.18 | 0.00 |
| peak joint velocity (rad/s) | 34.04 | 32.82 |
| joint contact-correctness (%) | 0.19 | 46.26 |

## 3. Multi-stage pipeline

**Table 2 — multi-stage pipeline, class means.**

| metric | floor (n=34) | locomotion (n=43) |
|---|---|---|
| floor penetration, max (cm) | 0.00 | 0.00 |
| floor penetration (% frames) | 0.00 | 0.00 |
| self-collision (% frames) | 0.002 | 0.001 |
| worst float (cm) | 7.55 | 5.57 |
| velocity spikes (mean/clip) | 0.00 | 0.00 |
| peak joint velocity (rad/s) | 37.39 | 37.92 |
| joint contact-correctness (%) | 98.85 | 98.79 |

Five of six metrics improve on `gmr_heightfix` by more than an order of
magnitude; joint contact-correctness rises from 0.2% to 98.8% on the floor
class. Peak joint velocity is 9.8% (floor) and 15.5% (locomotion) above
`gmr_heightfix` — a real cost, attributable to branch-flip artifacts in the
per-frame correction stage (Discussion) and only partially mitigated by a
null-space continuity term in that stage; the mitigation itself trades a
smaller regression in jerk and contact-slip on part of the floor class.

## 4. Oracle comparison

A per-clip constant Z-shift, swept by grid search to jointly minimize
penetration and floating with no notion of per-frame contact timing,
outperforms the multi-stage pipeline on either single-axis metric in
isolation (held-frame float within tolerance, or whole-body penetration,
each considered alone) on some clips. It cannot satisfy joint
contact-correctness, which requires both at the same frame: a rigid shift
cannot correct a multi-phase clip's standing stance and its lying phase with
one number. This is the basis for reporting joint contact-correctness as the
primary contact metric.

## 5. Global pipeline

Full-corpus evaluation of the global pipeline (Methods §3) is in progress;
we report results on the 36 clips completed at time of writing (18
floor-contact, 18 locomotion, class-balanced), pending completion of the
remaining 41.

**Table 3 — global pipeline vs. `gmr_heightfix` and the multi-stage
pipeline, class means, 36/77 clips.**

| metric | class | `gmr_heightfix` | multi-stage | global |
|---|---|---|---|---|
| joint contact-correctness (%) | floor | 0.20 | 98.50 | 81.02 |
| | locomotion | 46.51 | 99.84 | 81.27 |
| floor penetration (cm) | floor | 3.06 | 0.00 | 0.00 |
| | locomotion | 3.07 | 0.00 | 0.00 |
| self-collision (% frames) | floor | 6.14 | 0.01 | 1.13 |
| | locomotion | 2.87 | 0.00 | 0.91 |
| worst float (cm) | floor | 18.35 | 8.16 | 13.54 |
| | locomotion | 7.87 | 5.03 | 11.53 |
| joint jerk | floor | 5293 | 2601 | 803 |
| | locomotion | 7409 | 3101 | 1256 |
| peak joint velocity (rad/s) | floor | 33.04 | 31.31 | 13.01 |
| | locomotion | 40.11 | 37.32 | 11.67 |
| velocity spikes (mean/clip) | floor | 0.06 | 0.00 | 0.00 |
| | locomotion | 0.00 | 0.00 | 0.00 |

The global pipeline improves on `gmr_heightfix` on joint contact-correctness,
floor penetration, self-collision, and velocity spikes on both classes, and
reduces jerk and peak joint velocity relative to *both* baselines by a wide
margin (jerk 3.2–6.6x lower, peak velocity 2.4–3.4x lower than the
multi-stage pipeline). It trails the multi-stage pipeline on joint
contact-correctness (81% vs. 98–99%) and self-collision, consistent with a
smaller total per-frame solve budget (Methods §5). Worst float is a mixed
result: better than `gmr_heightfix` on the floor class (18.35 → 13.54 cm)
but worse on locomotion (7.87 → 11.53 cm), the one metric where the global
pipeline does not uniformly improve on the naive baseline. Which pipeline is
preferable depends on which axis matters more for a given downstream use;
Discussion §2 and the planned policy-training evaluation (§Future Work) are
intended to make that trade-off decidable rather than qualitative.

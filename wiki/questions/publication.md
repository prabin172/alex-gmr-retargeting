# Publication Plan

Two tracks. Lit landscape: [[related-work]] (Undermind report, 2026-07-03).

## Track 1 — Humanoids 2026 short/system paper (deadline ~2026-07-20/24)
Detail was in `paperIdea2.md` (deleted 2026-07-03; recover via `git show HEAD:paperIdea2.md` — has
the 20-day week plan + honest abstract draft). Angle A: contact-first retargeting of floor-contact motions on Alex,
complementary to GMR; shank clamp + θ·axis as technical core; no policy/hardware claims.
Evidence on disk: 18 clips, metrics log, spike 14–31→0, era ablations ([[era-ablations]]), IHMC export.
Gate: mentor sign-off on no-policy-eval scope. Status: decision pending.

## Track 2 — Humanoids 2027 full paper: **"Any-Contact Retargeting"** (deadline ~2027-07)
Detail: **`paperIdea3.md`** (2026-07-03, supersedes paper_idea.md + paperIdea2.md as the 2027 plan).
One line: *general retargeters do feet; we do floors.* Fills the 4 gaps no paper covers ([[related-work]]):
- **C1** generalized contact set — knees/shins/pelvis/elbows as first-class contacts (extends [[contact-first-ik]]).
- **C2** robot floor-stance TEMPLATE library (kneel/half-kneel/all-fours/prone/seated under Alex limits) + human→template mapping; shank clamp generalized to per-chain target edits. Centerpiece.
- **C3** quasi-static CoM-in-support-region soft constraint in Stage B (linear rows, stays convex — no kinodynamics).
- **C4** hardware playback of floor clips through the standard IHMC stack via [[ihmc-export]] — the claim nobody in 170 papers can make.
Eval: GMR metric suite + policy training (BeyondMimic-on-Alex, G1 fallback) + hardware, vs GMR/OmniRetarget/faithfulness-first baselines. Full slip/collision distributions per overclaim discipline ([[tradeoffs-limits]]).
Critical path: BeyondMimic-on-Alex — go/no-go gate Oct 2026 (one policy on one clip).
Mentor items: policy infra choice, hardware time (Mar–Apr 2027), benchmark release of in-house clips, new captures (crawl, prone-to-stand), whether the 2026 short paper was submitted.

## Dead claims (never revive)
"First to exploit offline context" / global-opt-as-novelty — occupied by SPARK/KDMR/IKMR/STMR, confirmed twice (June 2026 lit pass + Undermind).

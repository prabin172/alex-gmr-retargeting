# Z-Grounding (Stage 4.5)

`scripts/post_process_ground_contactfirst.py`. Purely-vertical rigid shift of the free base plants the motion on z=0; joints and horizontal motion untouched. Math: METHOD.md §7.

- Per frame, compute the robot's TRUE lowest world point over all collision geoms. Mesh geoms (convex hulls): transform every hull vertex, take min z — exact. Primitives: closed-form lowest-extent formulas (sphere/capsule/box/cylinder support functions), NOT bounding boxes (bounding boxes over-correct tilted shapes → floating robot).
- Excluded: floor/worldbody geoms (bodyid 0) and non-colliding geoms.
- **`constant-contact`** (batch + script default, 2026-07-06): a single Δ for the clip, but the floor is registered to the **planted feet** — the sole-corner sites (`alex_{l,r}_sole_corner_*`) on frames where that foot is `contact_flags`-labelled, `floor = median` of those heights (`--contact-percentile 50`). One shift ⇒ **zero vertical wander** (no bobbing); foot reference ⇒ feet stay on the floor. Median (not a low percentile) so it keys the stable **stance**, not the brief touchdown transient (a heel-strike corner dips several cm; a low percentile there floats the whole standing phase). Falls back to `constant` if no foot-contact frames / sole sites. standup_02: feet within 0.6 cm of z=0 at clip end, shift wander 0 cm.
- **`perframe`**: Δ(t) = −z_min(t) each frame, de-jittered by implicit tridiagonal smoother (`--smooth-shift`). Plants whatever is lowest every frame — but on a get-up the lowest point migrates hands→knees→feet, so Δ(t) **wanders 7–9 cm** = the robot bobbing up/down in a fixed world frame (RDX). Superseded by `constant-contact` for the batch.
- **`constant`**: single Δ = −percentile of per-frame z_min over ANY geom; zero wander but grounds on whatever is globally lowest — during a get-up that is the early hands/knees, leaving the final feet floating (+9.8 cm on standup_02). Use `constant-contact` instead.

A rigid vertical shift is **1 DOF** — it cannot co-plant two feet the *solve* left non-coplanar; that is fixed upstream (Stage-3 `--coplanar-feet-mode` coplanar targets + Stage-4 on-floor rows; see [[contact-first-ik]], [[globalopt]]). Once the feet are coplanar, one `constant-contact` shift plants both with no bobbing.

Saves `qpos_ungrounded`, `ground_shift`, `ground_lowest_before/after`.

> The Mimic-ready `contact_labels (T,11)` export (11 bodies, 2 cm threshold) lives in `scripts/legacy/post_process_grounding_contacts.py` — built for the RETIRED pipeline, not yet wired into the contact-first path. See [[open-questions]].

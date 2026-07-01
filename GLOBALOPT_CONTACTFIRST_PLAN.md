# GlobalOPT for Contact-First — implementation plan

Goal: kill flick + slip in the contact-first output via a **contact-aware**
two-stage global trajectory optimizer. Plan only — not yet implemented.

## Base to fork
`globalOPT` branch commit `d1bdf9a`: `scripts/solve_global_trajectory_opt.py`
(547 lines). Already does Stage A tridiag smoothing + Stage B OSQP QP
(re-linearized tracking Jacobians + SCA self-collision). Cut shoveling 21→0
spikes. Recover with:
`git show d1bdf9a:scripts/solve_global_trajectory_opt.py`

New file: `scripts/solve_global_trajectory_opt_contactfirst.py`.
Output dir: `outputs/global_opt_contactfirst/`.

## Why the base is insufficient
Position-tracking-only and **contact-blind**. Smoothing joint angles moves the
end-effector → feet/hands drift off contact points = **slip**. Ignores
foot-flat / fist-down / palm-pin.

## Locked design decisions
1. **Anchor = median IK position** — per contact interval, anchor = median of the
   per-frame IK contact-point world positions over that interval. Robust to
   jitter, minimal tracking disruption.
2. **Feet hard, hands soft** — foot contacts = hard equality (no-slip + flat);
   hand contacts = high-weight soft cost (reach-limited dynamic pushes must not
   make the QP infeasible).
3. **Full A+B contact-aware in one pass** — Stage A smoothing + full
   contact-aware Stage B (anchoring + flat + reweight) together.

## Inputs (from contact-first NPZ)
`qpos`, `contact_flags (T,4)`, `contact_effector_names`, `target_positions`,
`target_orientations`, metadata (`contact_pos_sites`, foot up-axis, gripper +X
axis, `contact_floor_z`, `target_weights`). Model = **V2**
(`assets/alex/alex_floating_base_with_sites_v2.xml`).

## Keep from base
Stage A per-joint tridiagonal smoothing; Stage B smoothness Hessian (velocity);
SCA self-collision inequalities; joint-limit box; OSQP + SCA outer loop.

## Add — contact-aware Stage B
1. **Contact anchoring (anti-slip).** Split each effector's `contact_flags` into
   contiguous intervals. Per interval: anchor = median per-frame IK contact-point
   position (site for hands `alex_{l,r}_palm_contact_site`; foot body/sole for
   feet). For every frame in the interval add a task-space row
   `J_pt_act·δqₜ = anchor − p_pt(q_cur_t)`:
   - **feet: hard equality** (l = u).
   - **hands: soft** high-weight cost row (append to tracking Hessian/grad, not A).
2. **Foot-flat / fist-down (during contact).** Rotational-Jacobian rows:
   - foot up-axis → world +Z: **hard equality**, align-error linearized
     (`err_rot = cross(a_world, +Z)` via `jacr`).
   - gripper +X → world −Z: **soft** (weight ≈ 0.8, matches per-frame).
3. **Tracking reweighting.** Load real `target_weights` from metadata. For a
   contacting effector, **down-weight its position tracking** during its contact
   (anchor governs the point) — mirrors per-frame `skip_pos_roles` suppress.
   Non-contact roles keep normal tracking.

## Linearization details
- Contact point Jacobian: `mj_jacSite` (hands) / `mj_jac` (foot body), take
  actuated cols `[6:]` (nv=35 → 29 act), same convention as base
  (`DV_ACT_SLICE`).
- Re-linearize anchors + flat rows each SCA outer iter (like tracking/collision).
- Root DOF (`qpos[0:7]`) still untouched by Stage A; Stage B optimizes actuated
  δq only (root left as-is, consistent with base).

## Outputs
Save-through all input keys + `qpos`(best), `qpos_per_frame`, `qpos_stage_a`,
`qpos_stage_b`, and carry `contact_flags` / `contact_effector_names` /
`contact_align_errors_deg` so `render_contactfirst.py` still overlays.

## Validation
Extend `compute_globalopt_metrics.py` → point at V2 model, add two columns:
- **contact-slip**: max contact-point world displacement within each interval
  (target ≈ 0 after anchoring).
- **foot-flat angle** during contact (should stay ≤ per-frame achieved).
Keep spikes / max_dq / p95_dq / coll% / peak_pen / track_mean.
Render before/after with `render_contactfirst.py` on standup + shovel clips.

## RESULTS — implemented 2026-07-01 (`solve_global_trajectory_opt_contactfirst.py`)

**Stage A (closed-form smoothing) is the win. Stage B (contact-pin QP) does not pay off yet — off by default (`--n-outer 0`).**

Per-frame IK → Stage A (λ_smooth=10, λ_track=1.0):
- standup_02:  spikes 17→0, coll 16.3%→7.3% (→2.4% at λ_track=0.5), track 0.049→0.051m.
- shovel_02:   spikes 17→0, coll 0.1%→0.0%,  track 0.046→0.054m.
Videos: `outputs/renders/contactfirst/{standup_02,shovel_fronthard_02}_globalopt.mp4`.

**Why Stage B was shelved (key finding):** the contact labels are **not stationary
plants** — a foot/hand repositions up to ~28 cm while staying labelled in-contact
(right_foot: 1 interval, 101 frames, 28.7 cm; hands: 18–24 cm). So:
1. Median-per-interval anchor is ill-posed → hard equalities are inconsistent →
   OSQP **primal infeasible**.
2. Restricting anchors to genuinely-stationary sub-segments leaves very few planted
   frames (5–37% of contact), so little true slip to pin.
3. All-soft contact + tracking pull-back toward collision-heavy per-frame targets
   destabilises the SCA collision loop → collisions **regress** (2.4%→24–34%);
   trust region then goes infeasible against the hard collision inequalities.
The "slip" the eye sees is mostly **genuine repositioning**, not smoothing drift.

**So real slip-removal is UPSTREAM**: tighten contact detection to isolate true
stationary plants (zero-velocity intervals), then Stage B has something well-posed
to pin. Implemented pieces are ready for that: stationary-sub-segment anchoring,
per-frame soft weights, trust region — all behind flags.

## Open tuning (when Stage B is revisited)
Prereq: contact detection that yields stationary plants. Then: soft-vs-hard on
planted-only sub-segments; λ_smooth / λ_track / λ_coll / foot/hand weights;
contact make/break edge blending; optional jerk (2nd-difference) term. Also decide
whether collision should be soft cost (avoids infeasibility with trust region)
rather than hard inequality.

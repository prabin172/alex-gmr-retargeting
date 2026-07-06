# Era Ablations (on-disk before/after material)

Each solver milestone kept its full output triplet (contactfirst / global_opt / grounded era dirs — see [[outputs-layout]]). These are ready-made ablations for the paper (see [[publication]]).

## Eras, in order
1. **pre_shankclamp → shankclamp**: the shank-tilt clamp + knee bias. Measured: straight-knee lock 26.5% → 0%; contact foot-flat error 12.7° → 7.7° mean. Angle-C core evidence.
2. **onset_hyst**: contact-onset hysteresis (capped 150 ms) — killed descending-body false onsets.
3. **foothold_fix**: foot-hold weight ×10 — plant drag 38–72 cm → 23–38 cm at solve time (rest removed by Stage B); made plants near-stationary (0.1–0.3 cm), which is what made Stage B well-posed.
4. **blend** (render era): make/break cosine cross-fade — raw contact switching measured ~2.8× larger pose jumps at transitions.
5. **fullmesh + soft-collision**: see [[fullmesh-vs-primitive]].

## Collision weight sweep (Stage 3 repulsion)
152-frame get-up, sweep of repulsion weight: **w=20 optimal** — collision frames 71.7% → 23.7% at +2.7% tracking cost; above ~20 the QP over-constrains and the solver oscillates stuck.

## Spike ablation (per-frame vs smoothed)
Per-frame IK: 14–31 velocity spikes per clip → **0 after Stage A smoothing on every clip** (in `fullurdf_pass_20260702_135608.log`). Cleanest headline result.

## Retired experiments
`--hierarchical`, hard-equality feet, upright-root constraint — see [[retired-approaches]].

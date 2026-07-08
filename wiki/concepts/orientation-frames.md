# Semantic Orientation Frames & World-Delta Transfer (Stages 1–2)

Raw FBX bone rotations are bind-pose/vendor-specific → **discarded in Stage 1**; only bone-head positions kept (converted to +x fwd/+y left/+z up). All orientation is rebuilt geometrically in Stage 2 (`build_canonical_orientation_frames_fresh.py`). Math: METHOD.md §3.

## Frame construction (7 oriented roles)
`pelvis, torso, head, left/right_foot, left/right_hand`. Two landmark-difference directions → Gram–Schmidt orthonormal frame (`frame_from_yz`, `frame_from_xy`):

| Role | primary | secondary |
|------|---------|-----------|
| pelvis | z = torso−pelvis | y = left_hip−right_hip |
| torso | z = neck−torso | y = shoulder lateral |
| head | z = head−neck | y = shoulder lateral |
| feet | x = toe−ankle | y = pelvis lateral |
| hands | x = middle_finger−wrist | y = thumb−wrist |

Foot local +z = sole normal; foot local +x = toe heading — both used directly by the contact machinery in [[contact-first-ik]].

> **FOOTGUN — foot local +z is NOT the true sole normal (2026-07-08, settled).** `x = toe−ankle` is the foot's *bone* axis: the ankle joint sits elevated above the ground while the toe is on it, so `toe−ankle` is declined ~18° below horizontal even when the sole is flat (measured: toe 4.6 cm below ankle over 14 cm → `arctan(4.6/14)=18.2°`, matching the frame's measured 18.7° tilt-from-vertical almost exactly). So `z = x×y` reads **~18° from vertical for a perfectly flat foot**, plus a ~4° systematic L/R skew (per-foot toe declination + toe-out against the shared pelvis-lateral axis). This is pure frame geometry, NOT motion tilt — constant across all 20 clips AND both subjects (Prabin+Luigi). **Impact is confined to the Stage-3 contact-detection tilt gate** (`tilt < foot_flat_tilt`); the world-DELTA orientation transfer below is unaffected (a constant offset cancels in the delta), and foot-flat *enforcement* uses the robot model's own sole normal, not this frame. **Fix = per-foot self-calibrated baseline subtraction** (gate on `tilt − p15(that foot's planted-frame tilt)`), NOT projecting `x` onto horizontal — projection zeroes the pitch signal entirely (verified: flat/+10°/−10° all read 0° projected), because the vertical component of `toe−ankle` is `baseline + real_pitch` summed, not pure bias. Markers are ankle/toe/toe_end, **no heel**, so a two-ground-point sole plane isn't available; self-calibrated baseline is the only pitch-recovering fix. After correction, tilt-above-baseline cleanly separates flat plants (258/286 windows <3°, p90=2.9°) from genuinely non-flat contacts (phantoms 6–16°, e.g. luigi_standSupine_08's supine foot at 7.6°). See [[grounding]], [[metrics]].

## Facing-yaw auto-detect
From first 10 frames' mean hip-width vector, compute correction yaw, **snap to nearest 90°** (avoids micro-corrections on aligned clips). If ≠0, rotate all positions about first-frame pelvis so every clip faces +x.

## World-delta orientation target (used in Stage 3)
Never copy absolute human orientation (rest conventions differ). Transfer only the world-frame change since rest, applied on Alex's achieved rest orientation:
`R_r*(t) = (R_r(t)·R_r(t₀)ᵀ) · R_r^alex-rest`.

NPZ adds: `orientation_mats (T,7,3,3)`, `orientation_role_names`, `facing_yaw_correction_deg`.

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

## Facing-yaw auto-detect
From first 10 frames' mean hip-width vector, compute correction yaw, **snap to nearest 90°** (avoids micro-corrections on aligned clips). If ≠0, rotate all positions about first-frame pelvis so every clip faces +x.

## World-delta orientation target (used in Stage 3)
Never copy absolute human orientation (rest conventions differ). Transfer only the world-frame change since rest, applied on Alex's achieved rest orientation:
`R_r*(t) = (R_r(t)·R_r(t₀)ᵀ) · R_r^alex-rest`.

NPZ adds: `orientation_mats (T,7,3,3)`, `orientation_role_names`, `facing_yaw_correction_deg`.

# Morphology Scaling (rest-relative delta scaling)

Targets are **scaled deltas from rest**, never absolute positions — a limb-length mismatch must never teleport the robot. Math: METHOD.md §4.

- Human rest = first frame t₀. Alex rest = configuration reached by an extended initial IK solve (max(3·ik_iters, 80)) onto the scaled first frame; `a_r` = Alex's achieved rest position of role r.
- **Global root scale** `s_root` = pelvis-to-head height ratio (Alex model zero pose / human t₀). Applies to pelvis displacement only.
- **Per-role scales** `s_r` = ratio of pelvis-relative rest distances, clamped to **[0.4, 2.5]**.
- Target: `p_r*(t) = a_r + s_root·Δpelvis(t) + s_r·Δ(pelvis-relative limb offset)(t)`.

Invariants:
1. Scaling applies ONLY to motion deltas from rest — never absolute root/pelvis position (would tear the body apart while walking).
2. Global displacement rides `s_root`; local limb gestures ride `s_r` (captures e.g. Alex's shorter arms).

Same machinery pins the **fist support point**: contacting hand's palm site gets its own rest position + per-hand scale in [0.4, 2.5] (see [[contact-first-ik]] fist pin).

> Library note: `general_motion_retargeting/retargeting/morphology_delta.py` + `rest_pose_scaling.py` implement the same idea with role-group ratios and a [0.70, 1.30] clamp — but the ACTIVE Stage-3 solver uses its own inlined version with the wider [0.4, 2.5] per-role clamp. Trust the solver for shipped numbers.

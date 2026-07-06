# Contact-First IK (Stage 3)

`scripts/solve_fbx_canonical_alex_contactfirst.py`. Per-frame damped Gauss–Newton (LM) least squares in MuJoCo velocity space, warm-started from previous frame. Contact detected from the HUMAN data; on contacting effectors the captured orientation is overridden by the physical support surface. Math: METHOD.md §5.

## Solver core
Stack weighted task rows `√w·J / √w·e`, solve damped normal equations, trust-region cap, step scale, manifold retract, clamp joints. Defaults: damping 1e-3, δ_max 0.20, step 0.70, 40 iters/frame.

Standing tasks: position (15 roles; pelvis 4.0, torso/head 2.0, ankle/wrist 1.5, knee/elbow 1.0, hip/shoulder 0.8), orientation (7 roles; pelvis 0.50, torso 0.25, head 0.20, **feet 0.70, hands 0.40**), posture reg (μ=1e-3, actuated only), self-collision repulsion (w=20, margin 2 cm, gain 5, skip ≤2 kinematic hops — w=20 found optimal by sweep, see [[era-ablations]]).

## Contact machinery (the distinctive part)
- **Detection** (from human markers): lowest marker within height threshold of clip floor (feet 0.07 m, hands 0.08 m; floor = 1st percentile of feet z) AND speed < 0.4 m/s. Feet additionally require human sole normal within **40°** of vertical (flat gate) — distinguishes a plant from a folded foot near floor.
- **Onset hysteresis**: contact START delayed until stricter gates (0.7·height, 0.5·speed) pass, capped at **0.15 s** (trim, never drop). Release unchanged. Kills "still descending" false onsets.
- **Make/break blending**: binary flags → continuous α∈[0,1]: debounce (min run 3 frames), preroll 2 frames, cosine cross-fade over 4 frames. All contact terms scale by α; competing human pos/ori terms on the same effector scale by (1−α).
- **Foot-flat**: align foot local +z to world +z using the **θ·unit-axis** error (cost θ², gradient always → 0) — NOT the cross-product form (cost sin²θ, spurious stable minimum at 180° flips feet upside-down). Weight 3.0. Spin about the axis stays free.
- **Fist-down**: align gripper local +x to −z, weight 0.8 (best-effort; the position pin does the real work).
- **Shank-tilt clamp** (target-side feasibility fix): projects the KNEE position target into the flat-foot-reachable tilt cone about the ankle — pitch clip ≈[−25°,+55°] (plantarflexion side tight), roll ±20° (ankle ranges minus 5° margin). Cross-faded by α. **Skipped when knee not meaningfully above ankle** (v·ẑ < 0.2L — deep kneel). Exists because Alex's ankle is stiff/asymmetric vs human (see [[alex-model]]).
- **Foot-hold**: when α ≥ 0.5, freeze the ankle position target at that pose (anchor) and boost its weight ×10 — resists being dragged by heavier trunk tasks (measured: weight 3 → 38–72 cm drag; weight 10 → 23–38 cm, rest removed by Stage B).
- **Foot-yaw align**: drives foot +x to human foot heading (ground-projected), weight 1.5·α — kills in-plane spin slip. Hands keep yaw free.
- **Fist position pin**: palm contact site (`alex_{l,r}_palm_contact_site`) pinned to the morphology-scaled human hand contact location, weight 3.0·α; wrist-body position target cross-faded out.
- **Coplanar-feet targets** (2026-07-06, `--coplanar-feet-mode {mean,min,off}`, default **mean**): when BOTH feet are contact-engaged, snap their **ankle-height (Z) targets** to a common value, cross-faded by `min(α_L, α_R)`. Foot-flat makes equal ankle Z ⇒ equal sole Z, so the IK produces **coplanar feet directly**. Fixes an inconsistent input: the morphology-scaled targets can put the two ankles several cm apart in Z (source ankles differ rel. to pelvis, or per-leg scale differs) while both are contact-labelled — "both planted" yet not coplanar, which a downstream 1-DOF grounding shift can't reconcile (one foot floats in RDX; standup_02 was 5.78 cm apart → achieved gap 0.95 cm after this). `mean` = meet in the middle (distributes the correction, lowest self-collision); `min` = snap the higher foot down to the lower/grounded one (more source-faithful, more extended pose → more self-collision). Only X,Y is left to the human target; the rest is [[globalopt]] on-floor rows + [[grounding]] `constant-contact`.

## Output NPZ (`alex_contactfirst_v1`)
`qpos (T,36)`, target/achieved pos+ori, `contact_flags (T,4)`, `contact_effector_names`, `contact_align_errors_deg`, `human_target_positions` (pre-contact-edit), `self_collision_counts`, `metadata_json`.

Related: [[morphology-scaling]] builds the targets; [[globalopt]] consumes the output.

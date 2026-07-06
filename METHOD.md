# Contact-First Human-to-Humanoid Motion Retargeting — Complete Method

*Single self-contained technical reference for the `alex-gmr-retargeting` pipeline.
Ground truth is the code on branch `main`; where an older note or the white
paper disagreed, this document follows the code and flags the correction.*

*The pipeline solves natively at **120 Hz** (the capture rate) and the active
collision model is the single canonical fullmesh XML
`assets/alex/alex_floating_base_with_sites.xml` with **always-on** soft
self-collision — there is no longer a `_v2` / `_v2_fullmesh` split or a
`--soft-collision` flag. §6.4 gives the rate-scaling recipe; §6.2 the Stage-B
QP as it actually runs.*

---

## 1. Overview and design philosophy

We retarget human motion capture (FBX / MVNX) onto the IHMC **Alex V2** humanoid — a
36-DOF floating-base biped — producing physically coherent joint trajectories usable
as reference motions for downstream imitation / physics-RL (e.g. BeyondMimic).

The core problem is **morphology mismatch**: a human and Alex have different limb
proportions, different DOFs, different joint ranges, and different coordinate
conventions, so raw human poses cannot be copied directly. We bridge the gap through a
geometry-neutral *canonical human* intermediate and a per-frame MuJoCo IK solve that
gives **priority to ground contact**, then regularise the whole trajectory in time and
plant it on the floor.

### Design stance (the through-line of every stage)

1. **Physical feasibility > verbatim copying.** Where the human is supported by the
   ground (feet flat, hands/fists pressing down), the robot must reproduce that
   *support surface* even at the cost of departing from the captured limb orientation.
2. **Contacts and end-effectors exact; body interior approximate.** Distal segments and
   the trunk are tracked; upper-arm / forearm / shin orientations are left free so the
   limb can bend however the contact demands.
3. **Kinematic infeasibility is fixed in the TARGETS, not by weight fights.** When a
   human pose demands an ankle beyond Alex's joint range, no weighting can reconcile
   "foot flat" with the human knee target — so we *edit the target itself* to be
   feasible (the shank-tilt clamp), rather than fighting it with soft weights.
4. **Downstream physics-RL absorbs dynamics, not kinematic impossibilities.** A
   centimetre of foot slip or a slightly-off velocity profile is something a learned
   policy can correct; self-penetration or an over-limit joint is not.

### Pipeline stages

| # | Stage | Input → Output | Nature | Script |
|---|-------|----------------|--------|--------|
| 1 | Canonical human | FBX/MVNX → landmark positions `(T,N,3)` | geometric re-expression | `build_fbx_canonical_human.py` (Blender) |
| 2 | Orientation frames | positions → per-part `SO(3)` frames + facing yaw | geometric construction | `build_canonical_orientation_frames_fresh.py` |
| 3 | Contact-first IK | canonical human → robot poses `q(t)` `(T,36)` | per-frame damped Gauss–Newton | `solve_fbx_canonical_alex_contactfirst.py` |
| 4 | Global trajectory opt | `q(t)` → smoothed `q(t)` | global temporal regularisation | `solve_global_trajectory_opt_contactfirst.py` |
| 4.5 | Z-grounding | `q(t)` → grounded `q(t)` | vertical rigid shift to floor | `post_process_ground_contactfirst.py` |
| 5 | Render | grounded `q(t)` → MP4 + contact strip | visualisation | `visualization/render_contactfirst.py` |
| 6 | IHMC JSON export | grounded `q(t)` → IHMC replay JSON | format conversion (native 120 Hz, no resample) | `export_alex_retarget_npz_to_ihmc_json.py` |

Intermediate representations are NumPy `.npz` at every stage boundary, so the pipeline
is inspectable and resumable. The whole per-clip run (stages 3–6) is driven by
`retargetingPipeline.sh` with **one identical config for every action** (Prabin's rule: a
single retargeter, no per-clip tuning). Stages 1–2 run per new FBX by hand in Blender.

```
FBX/MVNX ─▶ [1] canonical positions ─▶ [2] semantic orientation frames + facing yaw
          ─▶ [3] contact-first per-frame IK  ─▶ [4] Stage-A smoothing (+ Stage-B QP)
          ─▶ [4.5] Z-grounding ─▶ [5] render / export
```

---

## 2. Coordinate and convention primer

- **World frame:** `+x` forward, `+y` left, `+z` up. Unit up vector `ẑ = (0,0,1)`.
- **Quaternions:** stored **wxyz** everywhere (not xyzw).
- **Robot configuration** `q = [p, ρ, θ]`:
  - root position `p ∈ ℝ³` (`qpos[0:3]`),
  - root quaternion `ρ ∈ S³` (`qpos[3:7]`, wxyz),
  - `n_θ = 29` actuated joints `θ ∈ ℝ²⁹` (`qpos[7:36]`).
  - `dim q = 36` (`nq`).
- **Velocity / tangent space** has `n_v = 35` (`nv`): a 6-vector root twist (`dv[0:6]`)
  followed by the 29 joint rates (`dv[6:35]`). Configuration updates use the manifold
  retraction `q ← q ⊞ δ` via MuJoCo `mj_integratePos`, which applies the exponential map
  on the root's `SE(3)` component and adds on the hinge joints.
- **`Log : SO(3) → ℝ³`** returns the axis-angle (rotation) vector. For `R` with angle
  `ϑ = arccos((tr R − 1)/2)`:

  $$\mathrm{Log}(R)=\frac{\vartheta}{2\sin\vartheta}\,(R_{32}-R_{23},\ R_{13}-R_{31},\ R_{21}-R_{12}).$$

  (In code, `rotmat_to_rotvec`; the small-angle branch uses the linear term.)
- **Canonical roles.** 15 body roles plus 4 contact sites:

  `pelvis, torso, head, {left,right}×{hip, knee, foot, shoulder, elbow, hand}`
  \+ contact sites `{left,right}×{palm, sole}`.

  Note the IK solver maps some roles to Alex bodies under slightly different names
  (`left_knee → LEFT_SHIN`, `left_ankle → LEFT_ANKLE_Y_LINK`, `left_wrist →
  LEFT_WRIST_X_LINK`); the canonical NPZ carries the finer landmarks
  (`left_ankle, left_toe, left_hand_middle, left_hand_thumb, neck`, …) needed to build
  frames and detect contact.
- **Alex model.** 36-DOF: 7-DOF free root + 29 actuated joints. Active model
  `assets/alex/alex_floating_base_with_sites_v2.xml` (convex-hull collision on
  arms/head/fist), with the FullURDF fullmesh variant adding convex-hull STL legs
  (§8).

Notation: `x_r(t) ∈ ℝ³` is the canonical position of role `r` at frame `t`;
`R_r(t) ∈ SO(3)` its semantic orientation frame (Stage 2). Frame `t₀` (the first solved
frame) is the clip's **reference / rest** frame.

---

## 3. Stage 1–2: canonical human and semantic orientation frames

### 3.1 Stage 1 — canonical positions

FBX files are imported in Blender background mode; at each animation frame the script
extracts **bone-head positions** for a fixed mapping from vendor bone names to canonical
roles and converts from Blender/FBX axes to the canonical `+x`-fwd/`+y`-left/`+z`-up
frame. Only *positions* are retained — raw FBX bone rotations are discarded because they
are bind-pose- and vendor-convention-specific. All orientation used downstream is rebuilt
geometrically in Stage 2. Output: `positions (T, N_roles, 3)`, `roles`, `fps`.

(Xsens MVNX has a dedicated Python adapter writing the same schema.)

### 3.2 Stage 2 — semantic orientation frames

For 7 oriented roles — `pelvis, torso, head, left_foot, right_foot, left_hand,
right_hand` — we build an orthonormal frame `R_r(t) ∈ SO(3)` from the **geometry of
neighbouring landmarks**, not from raw rotations. Two landmark-difference directions seed
a primary and a secondary axis; the third is their cross product, then Gram–Schmidt:

$$\hat u=\frac{d_1}{\lVert d_1\rVert},\qquad
\hat w=\frac{\hat u\times d_2}{\lVert \hat u\times d_2\rVert},\qquad
\hat v=\hat w\times\hat u,\qquad R_r=[\,\hat u\ \hat v\ \hat w\,].$$

Two primitives implement this (`frame_from_yz`, `frame_from_xy`). The specific
constructions (columns `[x forward, y left, z up/normal]`):

| Role | primary hint | secondary hint |
|------|--------------|----------------|
| pelvis | `z = torso − pelvis` | `y = left_hip − right_hip` |
| torso  | `z = neck − torso`   | `y = left_shoulder − right_shoulder` |
| head   | `z = head − neck`    | `y = shoulder lateral (as torso)` |
| left/right foot | `x = toe − ankle` | `y = pelvis lateral (left_hip − right_hip)` |
| left/right hand | `x = middle_finger − wrist` | `y = thumb − wrist` |

So the **foot frame's local `+z` is the sole normal** and the **foot's local `+x` is the
toe-forward heading** — both used directly by the contact machinery in Stage 3.

### 3.3 Facing-yaw auto-detect and snap

Lab-frame clips may face any axis, but the canonical convention needs the actor facing
`+x`. From the first 10 frames we take the mean hip-width vector `left_hip − right_hip`
(the actor's left = `+y` when facing `+x`), project to the ground plane `(l_x, l_y)`, and
compute the correction yaw:

$$\psi_{\text{raw}}=\operatorname{atan2}(l_x,\ l_y),\qquad
\psi=90^\circ\cdot\operatorname{round}(\psi_{\text{raw}}/90^\circ).$$

Snapping to the nearest 90° prevents applying tiny floating-point corrections to
already-aligned clips. If `ψ ≠ 0`, **all positions are rotated about the first-frame
pelvis by `ψ`** and stored back, so every downstream stage sees a consistently
`+x`-facing actor. Output adds `orientation_mats (T,7,3,3)`, `orientation_role_names`,
`facing_yaw_correction_deg`.

### 3.4 World-delta orientation target (used in Stage 3)

We never copy the human's *absolute* orientation onto Alex (their rest orientations
differ by joint-axis conventions). Instead we transfer only the **world-frame change**
of each part's frame from its own rest frame, applied on top of Alex's achieved rest
orientation:

$$R_r^\star(t)=\underbrace{\big(R_r(t)\,R_r(t_0)^{\!\top}\big)}_{\text{world delta since rest}}\;R_r^{\text{alex-rest}}.$$

---

## 4. Morphology scaling (rest-relative delta scaling)

Targets are built as **scaled deltas from a rest pose**, so a limb-length mismatch never
teleports the robot. Two rest poses are involved: the human rest = first frame `t₀`;
Alex's rest = the configuration Alex reaches from an initial extended IK solve
(`iters = max(3·ik_iters, 80)`) onto the scaled first frame. Let `a_r` be Alex's achieved
rest position of role `r`.

**Global root scale** (translation only) is the pelvis-to-head height ratio:

$$s_{\text{root}}=\frac{\lVert a_{\text{head}}-a_{\text{pelvis}}\rVert_{\text{model, zero pose}}}
{\lVert x_{\text{head}}(t_0)-x_{\text{pelvis}}(t_0)\rVert}.$$

**Per-role scales** compare pelvis-relative rest proportions (Alex vs human), clamped:

$$s_r=\operatorname{clip}\!\left(\frac{\lVert a_r-a_{\text{pelvis}}\rVert}
{\lVert x_r(t_0)-x_{\text{pelvis}}(t_0)\rVert},\ 0.4,\ 2.5\right)
\quad(\text{code: }\texttt{compute\_per\_role\_scales}).$$

**Position target** for role `r` at frame `t`:

$$\boxed{\,p_r^\star(t)=a_r
+\underbrace{s_{\text{root}}\big(x_{\text{pelvis}}(t)-x_{\text{pelvis}}(t_0)\big)}_{\text{root displacement (unscaled per-role)}}
+\underbrace{s_r\big[(x_r(t)-x_{\text{pelvis}}(t))-(x_r(t_0)-x_{\text{pelvis}}(t_0))\big]}_{\text{pelvis-relative limb motion, per-role scaled}}\,}$$

The pelvis itself omits the third term (`p_pelvis = a_pelvis + s_root·Δpelvis`).

The two invariants that matter:
- **Scaling is applied only to motion *deltas* from rest**, never to absolute
  root/pelvis position — that would tear the body apart during walking.
- **Global walking displacement** rides the single `s_root`; **local limb gestures**
  ride the per-role `s_r` (which capture e.g. Alex's shorter arms).

The same machinery pins the **fist support point**: for a contacting hand the palm
contact site gets its own rest position `a_{\text{palm}}` and per-hand scale (§5.6).

*(Note: `general_motion_retargeting/retargeting/morphology_delta.py` and
`rest_pose_scaling.py` are the library implementations of the same idea — role-group
reach ratios with a `[0.70, 1.30]` clamp and `preserve_root_translation=True`. The
active Stage-3 solver uses its own inlined version with the wider `[0.4, 2.5]` clamp and
per-role rest-distance ratios; trust the solver for the shipped numbers.)*

---

## 5. Stage 3 — contact-first inverse kinematics

Each frame is solved independently by damped Gauss–Newton (Levenberg–Marquardt) least
squares in MuJoCo velocity space, warm-started from the previous frame. Contact is
detected from the **human** data; on contacting effectors the captured orientation is
overridden by the physical support surface.

### 5.1 Per-frame least-squares core

At iterate `q_k` we stack weighted task rows into `A δ = b` with `δ ∈ ℝ^{n_v}`; each task
contributes `√w · J` (rows of `A`) and `√w · e` (rows of `b`). Solve the damped normal
equations, cap the step to a trust region, scale, retract, clamp joints:

$$\delta=\big(A^{\!\top}A+\lambda I\big)^{-1}A^{\!\top}b,\quad
\delta\leftarrow\min\!\Big(1,\tfrac{\delta_{\max}}{\lVert\delta\rVert}\Big)\delta,\quad
\delta\leftarrow s_{\text{step}}\,\delta,\quad q_{k+1}=q_k\boxplus\delta.$$

Defaults: damping `λ = 1e-3`, `δ_max = 0.20`, `s_step = 0.70`, `ik_iters = 40` per frame.
After each step, hinge joints are clamped to their limits and the root quaternion is
renormalised. The root orientation is **deliberately not forced upright** — for get-up
motions an upright-root constraint makes the limbs absorb the whole lying-to-standing
rotation and pick bad IK branches.

*(The solver also has a two-level "hierarchical" nullspace mode — level-1 foot tasks,
level-2 body/hand tasks. It is **retired / OFF** in the unified pipeline: promoting the
reach-limited palm pin to a hard task starved body tracking and regressed pivoting
get-ups. The shipped path is the single-level soft-weighted solve; all priorities are
weights, no hard hierarchy.)*

**Standing task rows** (always active unless suppressed by contact):

- **Position tracking** (15 roles): `e = p_r^⋆(t) − p_r(q_k)`, `J = J_r^p` (body positional
  Jacobian, `mj_jacBody`). Weights `w_r`: pelvis 4.0; torso, head 2.0; ankle, wrist 1.5;
  knee, elbow 1.0; hip, shoulder 0.8.
- **Orientation tracking** (7 roles): `R_err = R_r^⋆(t) R_r(q_k)^⊤`, `e = Log(R_err)`,
  `J = J_r^ω` (rotational Jacobian). Weights: pelvis 0.50, torso 0.25, head 0.20,
  **left/right foot 0.70, left/right hand 0.40** (these last four are higher than the
  old white-paper values 0.35/0.20 — trust the code). Scaled by CLI `--ori-scale` (1.0).
- **Posture regularisation:** `√μ · I` on the actuated block with residual
  `√μ (θ_ref − θ_k)` where `θ_ref` is the start-of-frame warm-start; root DOFs left free.
  Biases the null space toward the previous pose (`μ = posture_reg = 1e-3`).
- **Self-collision repulsion** (§5.7).

### 5.2 Contact detection from human data

An effector `e` (each foot, each hand) is in contact at frame `t` iff its lowest marker
is within a height threshold of the clip floor **and** moving slower than a speed
threshold:

$$c_e(t)=\Big[\min_{m\in e}\big(z_m(t)-z_{\text{floor}}\big)<h_e\Big]\ \wedge\
\Big[\min_{m\in e}\lVert\dot x_m(t)\rVert<v_{\text{thresh}}\Big],$$

with `z_floor` = the 1st percentile of the feet markers' height over the clip. Defaults:
foot height `h = 0.07 m`, hand height `h = 0.08 m`, speed `v_thresh = 0.4 m/s`.

**Foot flatness gate.** For feet, contact additionally requires the *human* foot to be
near-flat: its canonical sole normal (foot-frame local `+z`) within `foot_flat_tilt =
40°` of world `+z`. This distinguishes a plantar plant from a foot merely near the floor
while folded (toes/side down during a get-up), where forcing the robot foot flat would
just fight position tracking.

**Contact-onset hysteresis.** The loose thresholds fire while the effector is still
*descending* into a pose. The **start** of each contact interval is delayed until the
effector passes stricter gates `h·on_height_frac` (0.7) and `v·on_speed_frac` (0.5);
release is unchanged. To avoid deleting genuine plants that hover under the loose gate
without ever passing the strict one, the delay is **capped** at `onset_max_delay = 0.15 s`
(the start is *trimmed*, never dropped). `frac = 1.0` disables hysteresis.

### 5.3 Make/break blending (continuous contact weight)

Raw `c_e(t)` would snap constraints on/off and jerk the pose (measured: ~2.8× larger pose
jump at raw transitions). We convert `c_e` to a continuous weight `α_e(t) ∈ [0,1]`:

1. **Debounce** — remove ON/OFF runs shorter than `contact_min_run = 3` solved frames
   (kills marginal-threshold flicker), filling short gaps and dropping short specks.
2. **Preroll** — extend each contact `contact_preroll = 2` frames earlier
   (anticipation: begin easing the effector toward the support face before touchdown).
3. **Cosine cross-fade** — over `contact_ramp = 4` frames at each leading/trailing edge,
   `α` rises/falls as `½(1 − cos(π k/(ramp+1)))`.

Every contact term below is scaled by `α_e`; the competing human position/orientation
terms on the same effector are simultaneously scaled by `(1 − α_e)`
(`ori_weight_scale` / `pos_weight_scale` cross-fade). Support engages and releases
smoothly rather than as a binary switch.

### 5.4 Foot-flat: the θ·unit-axis error (why it beats sin²θ)

During foot contact we align a body-fixed axis `a` (foot local `+z`) to a world direction
`d` (`+ẑ`). For hands we align gripper local `+x` (the closed-fist palm/finger-front
normal) to `−ẑ` (pressing down). Only the *axis* is locked — spin about it stays free, so
heading/yaw is left for position tracking (feet also get an explicit yaw term, §5.5).

Given `a_w = R a` (current world axis), define

$$c=a_w\times d,\quad s=\lVert c\rVert=\sin\theta,\quad
\theta=\operatorname{atan2}(s,\ a_w\!\cdot d),\qquad
e_{\text{align}}=\frac{\theta}{s}\,c=\theta\,\hat c.$$

So the residual is `θ · unit_axis` (magnitude = the angle, direction = the rotation axis),
with Jacobian `J^ω`. **Why this form.** The bare cross product `a_w × d` has magnitude
`sin θ`, so its least-squares cost is `sin²θ`, which has a **spurious stable minimum at
θ = 180°**: a stiff flat term can flip a near-limit foot right through the singularity and
lock it upside-down. The `θ·unit_axis` form has cost `θ²`, whose gradient always drives
`θ → 0`. The antipodal case (`s ≈ 0`, `dot < 0`) is handled by picking any perpendicular
axis. Foot-flat weight `= 3.0`; fist-down weight `= 0.8` (best-effort — the arm is often
reach-limited on dynamic pushes, and the fist *position* pin is what actually establishes
the support).

### 5.5 Shank-tilt clamp (target-side feasibility fix)

With the foot flat, the ankle chain is `R_foot = R_shin · R_y(ankle_y) · R_x(ankle_x)`,
so the **shank direction** (shin `+z`) can only tilt within the ankle joint ranges:
forward-lean `pitch = −ankle_y`, leftward-lean `roll = +ankle_x`. Alex's ankle is
*asymmetric and stiffer than a human's* (dorsiflexion 60°, plantarflexion 30°, roll ±25°,
no ankle yaw, rigid foot; human ≈ 20°/50°). Copying the human knee verbatim during a
plant therefore demands near/over-limit ankle angles, and **no weighting can reconcile
"foot flat" with an infeasible knee target**. So we edit the target.

`clamp_shank_tilt` projects the **knee position target** into the flat-foot-reachable
tilt cone about the (held) ankle target, along the human foot heading `f` (ground-projected
`+x` of the foot frame), lateral `lat = ẑ × f`:

$$v=\text{knee}^\star-\text{ankle}^\star,\quad L=\lVert v\rVert,\quad
\text{pitch}=\operatorname{atan2}(v\!\cdot f,\ v\!\cdot\hat z),\quad
\text{roll}=\operatorname{atan2}(v\!\cdot lat,\ v\!\cdot\hat z),$$
$$\widehat{\text{pitch}}=\operatorname{clip}(\text{pitch},\ \text{pitch\_rng}),\quad
\widehat{\text{roll}}=\operatorname{clip}(\text{roll},\ \text{roll\_rng}),\quad
u=\frac{f\tan\widehat{\text{pitch}}+lat\tan\widehat{\text{roll}}+\hat z}{\lVert\cdot\rVert},\quad
\text{knee}^\star\!\leftarrow\text{ankle}^\star+L\,u.$$

Ranges are read from the model's ankle limits with a 5° inside margin:
`pitch_rng = (−ankle_y_hi + 5°, −ankle_y_lo − 5°)`, `roll_rng = (ankle_x_lo + 5°,
ankle_x_hi − 5°)` (effectively ≈ `[−25°, +55°]` pitch, `±20°` roll). The clamped knee is
cross-faded in by the contact weight `α_e` (`knee ← (1−α)knee + α·knee_clamped`). It is
**skipped when the knee is not meaningfully above the ankle** (`v·ẑ < 0.2 L`, i.e. deep
kneel / data glitch), where the flat-foot decomposition is undefined — relevant for
kneeling clips.

### 5.6 Foot-hold, foot-yaw, and the fist position pin

- **Planted-foot position hold.** A flat foot that keeps tracking the *moving* human
  ankle target slides ("plant slip"). When a foot commits (`α_e ≥ foot_hold_latch = 0.5`)
  we **freeze its ankle position target** at that pose (`foot_hold_anchor`) and cross-fade
  the moving target onto the frozen anchor: `target ← (1−α)target + α·anchor`. The held
  foot's position weight is boosted by `1 + (foot_hold_weight − 1)·α` with
  `foot_hold_weight = 10.0` so the frozen anchor resists being dragged by the heavier
  pelvis/torso tasks (a foot dragged by heavier trunk targets was the real slip source; at
  weight 3 the shovel body dragged the plant 38–72 cm, at 10 → 23–38 cm and the rest is
  removed by Stage 4). While held, the ankle role is promoted to level-1 in the
  (optional) hierarchical solve.
- **Foot-yaw align.** Flat pins pitch/roll but leaves yaw free, so a planted foot can spin
  in-plane (inner/outer slip). An extra align row drives the foot's forward axis `+x` to
  the **human** foot heading (ground-projected), weight `foot_yaw_weight = 1.5·α`, skipped
  when the human foot points near-vertical. Hands keep yaw free (the fist support face
  doesn't need it).
- **Fist position pin.** For a contacting hand, the wrist-body position target alone
  leaves the fist a few cm off the floor. So we pin the **palm contact site**
  (`alex_{l,r}_palm_contact_site`, via `mj_jacSite`) to the human hand contact location
  built with the same morphology-delta machinery (own palm rest position + per-hand scale
  in `[0.4, 2.5]`), weight `3.0·α`. The now-redundant wrist-body position target is
  cross-faded out (`pos_weight_scale = 1 − α`). This is the substantive "fist support"
  term; the fist-down align (§5.4) only orients it.

### 5.7 Self-collision repulsion (in-solver, soft)

For each MuJoCo contact between two robot bodies that are *not* within `k = 2` kinematic
hops (MuJoCo already excludes 1-hop parent-child; 2 hops also drops structural near-misses
like HEAD↔TORSO that always overlap on large geom radii), with penetration
`pen = margin − dist > 0`, add one separation row:

$$\sqrt{w}\;\hat n^{\!\top}\big(J_1^{p}-J_2^{p}\big)\,\delta=\sqrt{w}\;\min(\text{pen},\,0.05)\cdot\text{gain},$$

where `n̂` is the contact normal signed to push `b1` off `b2`, and `J_{1,2}^p` are the
contact-point Jacobians (`mj_jac`). Defaults: `w = 20`, `margin = 0.02 m`, `gain = 5.0`.
A weight sweep on a 152-frame get-up found **w = 20 optimal** (collision frames
71.7% → 23.7% with +2.7% tracking); above ≈20 the QP over-constrains and the solver
oscillates in a stuck configuration.

### 5.8 Stage-3 output

`qpos (T,36)`, plus the target/achieved positions and orientations, `contact_flags
(T,4)`, `contact_effector_names`, `contact_align_errors_deg`, `human_target_positions`
(pure morphology-scaled human, before the contact edits), `self_collision_counts`, and a
`metadata_json` carrying weights, scales, floor `z`, contact params, and the palm site
names. Format tag `alex_contactfirst_v1`.

---

## 6. Stage 4 — global trajectory optimisation

Per-frame IK has no cross-frame coupling beyond the warm-start and posture reg, so it
leaves velocity spikes (an elbow/shoulder branch flip = a 1–2 rad single-frame jump) and
root pops. These cannot be fixed within a per-frame framework — the frame at `t−3` would
have to "know" a topology change is coming at `t`. Because we build datasets **offline**,
the whole trajectory is available, so we optimise over all `T` frames jointly. This
offline/online distinction is the methodological lever.

### 6.1 Stage A — closed-form tridiagonal smoothing

For a scalar channel `x_{1:T}` we solve, per channel independently,

$$\min_{y}\ \lambda_{\text{track}}\sum_t (y_t-x_t)^2+\lambda_{\text{smooth}}\sum_t (y_t-y_{t-1})^2,$$

a **first-difference (velocity) penalty**. Its normal equations are the tridiagonal
system

$$\big(\lambda_{\text{track}} I+\lambda_{\text{smooth}} D^{\!\top}\!D\big)\,y=\lambda_{\text{track}}\,x,$$

where `DᵀD` is the path-graph Laplacian: interior diagonal `λ_track + 2λ_smooth`,
endpoints `λ_track + λ_smooth`, off-diagonals `−λ_smooth`. Solved directly with the banded
solver `scipy.linalg.solve_banded` in `O(T)` per channel (`_banded_smoother` /
`_smooth_channel`). All 29 actuated channels are smoothed then clipped to joint limits.
Stage A cannot change a channel's mean, only redistribute a spike over neighbours.
`λ_track = 1.0`; `λ_smooth = 320` at the shipped native 120 Hz rate (the script default is
`λ_smooth = 10`; the pipeline passes `--lambda-smooth 320`). The first-difference penalty
is a *velocity* term, so it scales with `fps²` — `λ_smooth = 20` at 30 Hz became `320 =
20·16` at 120 Hz. §6.4 gives the full rate-scaling table and the derivation.

**The floating base is smoothed too** (else the whole body flicks — per-frame IK root has
~3 cm / 10° pops): root position `qpos[0:3]` uses the same tridiagonal solve; the root
quaternion `qpos[3:7]` is smoothed by (i) hemisphere-aligning consecutive quaternions
(`ρ_t ← −ρ_t` if `ρ_t·ρ_{t−1} < 0`), (ii) smoothing the four components independently,
(iii) renormalising. `--root-smooth` can set a gentler root weight; `--no-root-smooth`
disables it.

### 6.2 Stage B — sparse contact-aware QP over all frames

Stage B refines the Stage-A warm start with a single sparse QP over the actuated
increments of every frame, `δQ ∈ ℝ^{T·29}` (root left as-is from Stage A):

$$\min_{\delta Q}\ \tfrac12\,\delta Q^{\!\top}P\,\delta Q+q^{\!\top}\delta Q
\quad\text{s.t.}\quad l\le A\,\delta Q\le u.$$

**Objective.** `P = 2(H_task + H_smooth)`, `q = g_track + g_contact`.

- `H_smooth`: the block-tridiagonal first-difference Hessian over all `T·29` variables
  (the Stage-A operator assembled sparsely; `_build_smoothness_hessian`).
- `H_task` (per-frame block-diagonal): tracking + contact, both re-linearised each outer
  iteration at the current trajectory:
  - *Tracking*: `Σ_r w_r ‖J_r δq_t − e_r‖²` with `e_r = target_r − FK_r(q_cur)`, `J_r` the
    actuated columns of the body position Jacobian. A contacting effector's **own** role
    is down-weighted (`× contact_downweight = 0.1`) while in contact — the contact anchor
    governs that point (mirrors the per-frame `skip_pos_roles`).
  - *Contact* (`_build_contact`, all **soft** — see box below): position pin of each
    effector to its anchor at per-frame weight; foot-flat (`Jr · err_rot`, weight
    `foot_flat_w = 3.0`) on planted foot frames; fist-down (weight `fist_w = 0.8`) while a
    hand is in contact.

**Contact anchor = per-interval median.** Labelled contact intervals are *not* stationary
plants (a foot/hand can reposition ~30 cm while staying labelled in-contact). So within
each contiguous interval we split into **stationary sub-segments** (per-frame IK
contact-point speed `< plant_speed = 0.05 m/s`) and anchor each sub-segment to its own
**median** contact-point position (high weight `foot_weight = 160`, `hand_weight = 32`,
`planted = True`). Non-stationary contact frames follow the per-frame IK point at a low
weight (`× move_ratio = 0.15`) — just enough to stop smoothing from *adding* drift without
fighting genuine repositioning.

A stillness sub-segment must run **at least `plant_min_run = 8` frames** (a frame-count knob,
≈2 at 30 Hz) to count as a plant; shorter dips are demoted to the low-weight moving path. This
debounces momentary velocity **zero-crossings** — e.g. a hand reversing as it lifts off during a
get-up dips below `plant_speed` for a single frame, which would otherwise become a 1-frame plant
anchored to that instant while the smoothed trajectory carries the hand away, inflating the
plant-slip metric with a phantom (measured on `standup_side_05`: the right hand's 14.7 cm "slip"
was 25 single-frame plants; with the debounce the real-plant slip is 6.8 cm). It mirrors the
Stage-3 `contact_min_run` contact debounce, applied to the stillness split.

> **Pin weights ×4 for the 120 Hz solve.** The CLI defaults are `foot_weight = 40`,
> `hand_weight = 8` (the 30 Hz values); the pipeline passes `--foot-weight 160 --hand-weight
> 32`. These are *position* terms, so unlike `λ_smooth` they are dt-invariant and did **not**
> need rescaling for correctness — but because `λ_smooth` went ×16 at 120 Hz while the pins
> stayed, the pins were left 16× weaker *relative to smoothing*, and plants slid. Restoring
> ×4 rebalances them: on `standup_side_04` this cut plant slip 10.4 → 6.3 cm for < 1 cm of
> (shallow, sub-tol) added self-collision. `plant_speed` is **not** a useful lever here — the
> dominant slip is a slow steady drift that stays under any sane speed threshold, so lowering
> it just reclassifies frames without moving the worst-frame slip; pin *weight* is the lever.

> **Contacts are soft, not equalities.** Every contact term is `add_soft` (`_build_contact`),
> so a planted foot is a high-weight soft cost (weight 160), not an equality. The only hard
> constraints in Stage B are the joint-limit box, the trust region, and the self-collision
> inequality rows. Residual plant slip is therefore a high-weight equilibrium, not zero by
> construction (shovels ≈1.0–1.5 cm; get-ups higher — see §9).

**Constraints.**
- *Joint-limit box*: `q_lo − q_warm ≤ δQ ≤ q_hi − q_warm` (identity rows).
- *Trust region* (SCA stabiliser): also intersect the box with `δ_prev ± trust`
  (`trust = 0.15 rad`) so the collision re-linearisation cannot oscillate.
- *Self-collision inequalities*: per penetrating non-adjacent contact at frame `t`,
  `√λ_coll · j_sep · δq_t ≥ √λ_coll · min(pen, 0.05)` (`λ_coll = 5`), the actuated slice of
  the same separation Jacobian as Stage 3, re-linearised each outer iter.

**Sequential Convex Approximation (SCA).** Collision (and contact/tracking) rows are only
linear at the current point, so an outer loop re-linearises → assembles → solves with
OSQP (`eps_abs = eps_rel = 1e-4`, `max_iter = 20000`, `polish=True`, warm-started across
iters) → applies `δQ` with joint clamps → repeats `n_outer` times (`--n-outer 6` at 120 Hz;
the *script* default is `--n-outer 0` = Stage A only). The OSQP accept-check treats both
`"solved"` and `"solved inaccurate"` (note the **space** — OSQP ≥1.x; a stale check for the
underscored `"solved_inaccurate"` silently discarded every inaccurate step, no-op'ing Stage B
on the larger 120 Hz QP). `max_iter` was raised 8000 → 20000 so the 4×-larger 120 Hz problems
reach full `"solved"`.

**Keep-best-iterate with a slip-aware score (the SCA convergence fix).** The SCA loop does
**not** monotonically decrease penetration. Collision rows are linearised only at the *start*
of each outer, so an outer that happens to begin collision-free carries **zero** collision
rows and takes an unconstrained tracking+smoothing step that walks straight back into
penetration. On the get-up clips the per-outer penetration therefore *oscillates*
(clean → ~6 cm → clean → …). Returning the **last** iterate unconditionally — as the original
loop did — makes the shipped result depend on `n_outer` **parity**: an odd count happened to
stop on a collision-resolving step, an even count on a bad victory-lap step. (The apparent
"30 Hz was fine, 120 Hz regressed" was this: 30 Hz used `n_outer = 3` (odd, lucky), 120 Hz used
`n_outer = 6` (even, unlucky) — not a rate effect at all.)

The fix: **keep the best iterate across outers and return it**, seeded with the Stage-A warm
start so Stage B is never worse than its own input. "Best" is a slip-aware lexicographic score
computed on the full trajectory after each accepted outer:

$$\text{score}(q)=\Big(\underbrace{\max(0,\ \text{pen}_{\max}-\tau)}_{\text{hard: real penetration}},\ \ \underbrace{\text{pen}_{\max}+\text{slip}_{\max}}_{\text{tie-break: total drift}},\ \ \text{coll\%}\Big),\qquad \tau=1\ \text{cm},$$

compared lexicographically (smaller wins). The first term makes any penetration beyond
`τ = 1 cm` a *hard* failure that is never traded away for slip; among iterates that are all
under `τ`, the second term picks the one with the least combined penetration + plant slip.
Without the slip term a pure-penetration argmin would silently ship a collision-clean iterate
that had pushed a foot off its plant (the pins-×4 change makes that trade real). This makes
Stage B parity-immune and monotone-non-worsening in the score, and the two changes compose:
`stage_b` returns `best_qpos`, not the last `qpos_cur`.

**Soft-slack self-collision (always on).** The dense convex-hull legs of the fullmesh body
make **hard** collision inequalities primal-infeasible (row count explodes, e.g. 424 vs
~80–194, with genuinely-close legs in get-ups/kneels), so a hard QP silently no-ops
(`|δQ|max = 0`). Stage B therefore reformulates each collision row with a non-negative slack
`s ≥ 0` and a quadratic penalty, so the QP is **always feasible** and degrades gracefully —
this is the only code path now (the old hard variant and its `--soft-collision` toggle are
gone). Augment the decision vector with one slack per collision row:

$$P=\begin{bmatrix}2(H_{\text{task}}+H_{\text{smooth}}) & 0\\[2pt] 0 & 2\rho I_m\end{bmatrix},\quad
\text{collision rows: } \sqrt{\lambda_{\text{coll}}}\,j_{\text{sep}}\!\cdot\delta q + s_i \ \ge\ \sqrt{\lambda_{\text{coll}}}\,\min(\text{pen},0.05),\quad
0\le s\le 10^6,$$

with penalty `ρ = collision_penalty = 1000`. Genuinely-close legs relax through the
(penalised) slack instead of driving OSQP infeasible. With no collision rows in an
iteration it falls back to the plain joint-limit QP. The pipeline runs this unconditionally
(`--collision-penalty 1000`).

### 6.3 Why Stage B is worth it now (history)

Stage B was originally shelved (`n_outer 0`) because the loosely-labelled contacts were
not stationary plants: median anchors were inconsistent → hard equalities infeasible, and
pulling back toward collision-heavy per-frame targets regressed collisions. The onset
hysteresis + foot-hold(×10) of Stage 3 now yield near-stationary plants (0.1–0.3 cm/plant),
so with the median/stationary-sub-segment anchoring + all-soft weights + trust region,
Stage B is well-posed and **enabled everywhere** in the unified pipeline. Stage A alone
re-adds ≈8 cm plant drift. The last robustness gap — the SCA loop shipping a
parity-dependent, sometimes-colliding last iterate — was closed by the keep-best-iterate
selection above (§6.2).

Output saves `qpos` (= the Stage-B best iterate if run, else Stage A), plus
`qpos_per_frame`, `qpos_stage_a`, `qpos_stage_b` (`qpos_stage_b` is the *returned* best
iterate, not the last outer), and carries the contact arrays through for the renderer.

### 6.4 Native 120 Hz solve and rate scaling

The pipeline solves at the **native 120 Hz capture rate** (`STRIDE = 1`). The motivation is
downstream: the IHMC RL tracker consumes reference motion at **50 Hz with zero-order hold
(no interpolation)**, so the earlier 30 Hz solve was sub-Nyquist for that gate and
self-upsampling 30 → 120 at export never restored the aliased content. Solving at 120 Hz and
letting the consumer's `json_to_npz --output_fps 50` do the *only* downsample removes the
aliasing. Export (§ stage 6) is run with **no `--fps`** so it stays native 120 Hz.

Only the **first-difference (velocity) penalties scale with rate**, as `fps²`. Everything
else in the objective is a position or per-frame term and is **dt-invariant**. Concretely,
going 30 → 120 Hz (dt/4):

| knob | 30 Hz | 120 Hz | scaling rule |
|------|-------|--------|--------------|
| `λ_smooth` (Stage A + Stage-B `H_smooth`) | 20 | **320** | velocity penalty ∝ `fps²` → ×16 |
| `GROUND_SMOOTH` (Stage 4.5 shift smoother) | 5 | **80** | same first-diff smoother → ×16 |
| `contact_min_run / ramp / preroll` | 3 / 4 / 2 | **12 / 16 / 8** | measured in *frames* → ×4 |
| `plant_min_run` (stillness debounce, §6.2) | 2 | **8** | measured in *frames* → ×4 |
| `n_outer` (Stage-B SCA passes) | 3 | **6** | 4× larger QP needs more re-linearisation passes |
| `foot_weight / hand_weight` (plant pins) | 40 / 8 | **160 / 32** | dt-invariant, but ×4 to rebalance vs the ×16 smoothing (§6.2) |

Derivation: dividing the continuous objective by `dt`, position terms (track `w=1`, contact
pins, collision `ρ`, trust, posture_reg) carry no `dt`, while the squared first difference
carries `1/dt²`. So `λ_smooth` and `GROUND_SMOOTH` take the ×16; frame-count debounce knobs
take ×4 (they count frames); speeds (m/s) and onset delays (s) already auto-scale through
`×fps` in code and need no change. The pin ×4 is a *relative* rebalance, not a correctness
requirement (§6.2). This was validated empirically: Stage A reproduces the 30 Hz smoothing,
and standup slip / collision are insensitive to `λ` and `ρ` sweeps at fixed rate.

---

## 7. Stage 4.5 — Z-grounding

The solved motion is planted on `z = 0` by a purely-vertical rigid shift of the free
base; joint angles and horizontal motion are untouched. For each frame we compute the
robot's true lowest world-space point over all collision geoms:

$$z_{\min}(t)=\min_{g}\ z^{\text{low}}_g(t).$$

Mesh geoms (convex-hull fists/limbs) are handled exactly by transforming every hull vertex
and taking the min Z: `z_g^low = min_i (p_g + R_g v_i)·ẑ` (only the Z row of `R_g` is
needed). Primitives use closed-form lowest-extent formulas rather than bounding boxes
(bounding boxes over-correct tilted shapes → floating robots):

- Sphere: `center_z − r`.
- Capsule (axis = local z): `min(center_z ± axis_z·half_len) − r`.
- Box: `center_z − |R_{20}|h_x − |R_{21}|h_y − |R_{22}|h_z` (support function).
- Cylinder: `min(center_z ± axis_z·half_len) − r·√(1 − axis_z²)`.

Only floor/worldbody geoms (`bodyid = 0`) and non-colliding geoms are excluded. Two modes:

- **`perframe`** (shipped default in `run_globalopt_all.sh`): `Δ(t) = −z_min(t)`, planting
  every frame; the shift series is lightly de-jittered by an implicit tridiagonal smoother
  `(I + wL)y = x` (`--smooth-shift 5`). Correctly handles sit-to-stand — the seat is
  planted while sitting, the feet once standing, since `z_min` tracks whichever is lowest.
- **`constant`** (script default): a single `Δ = −percentile_p(z_min(t))` for the whole
  clip, preserving all vertical motion but leaving the body above the floor except at its
  lowest moments.

`qpos[:,2] ← qpos[:,2] + Δ`. The pre-grounding trajectory is kept as `qpos_ungrounded`;
`ground_shift`, `ground_lowest_before/after` are saved.

*(A separate `post_process_grounding_contacts.py` produces the Mimic-ready
`contact_labels (T,11)` over 11 bodies — feet, shins, thighs, pelvis, torso, head, both
grippers — using the same exact per-shape `z_min` within a 2 cm threshold. That is the
BeyondMimic export format.)*

### 7.1 Stage 6 — IHMC JSON export

`export_alex_retarget_npz_to_ihmc_json.py` converts a grounded NPZ into the IHMC
`KinematicsToolboxOutputStatus` replay JSON (root pose + 29 joints per frame, MuJoCo joint
order remapped to the Isaac full-body order). The one subtlety is the **real-time frame
rate**: the NPZ stores `fps` = the capture rate (120), but the solved qpos is strided, so the
true rate is `capture_fps / stride`, derived from the median stride of `source_frame_ids`. At
the shipped `STRIDE = 1` that is `120 / 1 = 120` Hz, so with **no `--fps`** the export writes
native 120 Hz with no resample (`--fps X` would resample to `X`). The downstream
`json_to_npz --output_fps 50` performs the only downsample (§6.4). The pipeline writes
`outputs/ihmcJsons-native120hz/<clip>.json`.

---

## 8. Model note: single mesh-collision body and the shank clamp

The active solver / GlobalOPT / grounding / export model is the single canonical, hand-
maintained XML **`assets/alex/alex_floating_base_with_sites.xml`**: full convex-hull STL
mesh collision on legs *and* arms/head/fist, plus named palm/sole sites for end-effector
targeting. (Historically there were a primitive `_v2` model and a separate `_v2_fullmesh`
variant on the `FullURDF` branch; those were consolidated into this one fullmesh model, and
soft-slack self-collision (§6.2) was made unconditional. The model-prep scripts
`create_alex_mujoco_sites_model.py` / `build_alex_v2_collision_model.py` / `prepare_*` are
historical and would **overwrite** the hand-maintained XML — never run them blindly.)
Kinematics match the real V2 URDF exactly (joints, axes, limits, link frames), so Stage 3
IK is model-independent to within collision geometry.

The **shank clamp exists because Alex's ankle range is much tighter and asymmetric versus
a human's** (dorsiflexion 60° vs ~20°, plantarflexion 30° vs ~50°, roll ±25°, no ankle
yaw, rigid foot). A human plant that is trivially flat-footed can be kinematically
impossible for Alex to reproduce flat-footed; §5.5 makes the knee target feasible so the
"foot flat" and "track the leg" objectives stop fighting.

---

## 9. Known trade-offs and limits

- **Kinematics only, no dynamics.** The output is a joint trajectory; it is not
  dynamically consistent (no torque/contact-force feasibility). Downstream physics-RL is
  expected to absorb the dynamics gap. The pipeline's job is to hand RL a trajectory with
  *no kinematic impossibilities* (no self-penetration, no over-limit joints, exact
  contacts), which RL cannot fix on its own.
- **Keep-best-iterate caps penetration; pins ×4 trade shallow grazing for less slip.**
  With the slip-aware keep-best (§6.2, `τ = 1 cm`), the shipped Stage-B **peak penetration
  is ≤ 0.88 cm on all 18 clips** — no real self-penetration anywhere. The pins-×4 rebalance
  reduces plant slip on get-ups (e.g. `standup_side_04` 10.4 → 6.3 cm) at the cost of some
  *shallow, sub-1 cm* grazing frames (`coll%` rises to 4–38% on get-up clips, but every one
  of those contacts is < 1 cm deep). This is the deliberate trade: a sub-centimetre graze is
  within the robot's collision margin and learnable; deep penetration is physically
  impossible and poisons physics-RL, so the `τ` gate never trades it away.
- **Residual get-up "flat-snap".** On some get-ups the foot still snaps toward flat near
  touchdown; this is partly faithful (the human foot does flatten) and partly the hard
  boundary of the flatness gate. Crouch-phase foot-flat angles of ~9–13° are genuine
  human tilt, not error.
- **Contacts are high-weight soft, not exact.** Plant slip is ≈1.0–1.5 cm on shovels,
  higher on dynamic get-ups; the median-anchor Stage B reduces but does not zero it.
- **Measured (native-120 batch, Stage B).** Velocity spikes 0 and peak self-penetration
  ≤ 0.88 cm on **every** clip. Shovels: plant slip 2.0–3.3 cm, foot-flat ~0.1°, 0% collision.
  Squat / kneeling-fall: 0% collision, slip 4.0 / 7.1 / (kneelingFall_03) cm. Standups &
  get-ups: slip 3.0–9.8 cm with shallow (< 1 cm, within-margin) grazing 4–28% on the harder
  get-ups (standupKnees_02, standupFromKneeling_02).
- **Plant-slip outliers are usually a metric phantom, not real slip.** `standup_side_05`
  reported 14.7 cm; a per-effector dig showed it was **entirely** the right hand and
  **entirely** 25 single-frame "plants" (velocity zero-crossings while the hand lifts off) —
  the IK contact point never left its anchor (IK-vs-median ≤ 0.4 cm), but Stage A smoothed the
  hand along its real moving path away from the 1-frame anchor. The `plant_min_run = 8` stillness
  debounce (§6.2) removes these; standup_side_05 drops to 6.8 cm (real-plant slip). Lesson: audit
  a slip outlier per-effector before attributing it to the solver.

---

## 10. Summary of distinctive choices

1. **Geometry-rebuilt semantic orientation frames** (from landmark positions, not raw
   FBX rotations) with **world-delta transfer** (copy only the change since rest).
2. **Rest-relative morphology scaling** that scales motion deltas per role and never
   displaces the root absolutely.
3. **Contact-first IK** that overrides captured orientation with the physical support
   surface under a smooth make/break cross-fade, using the well-conditioned `θ·unit_axis`
   flat error, a **target-side shank-tilt clamp** for ankle-range feasibility, foot-hold
   freeze-at-touchdown, foot-yaw lock, fist position pin, and onset hysteresis.
4. **Whole-body (including floating-base) global smoothing** — closed-form tridiagonal
   Stage A plus a **contact-aware sparse QP Stage B** with median-anchored plants and
   **soft-slack self-collision** that stays feasible on the dense fullmesh body.
5. **Mesh-exact vertical grounding** that plants whichever geom is lowest (seat, knee,
   or foot) each frame.

Together these produce contact-faithful, temporally smooth, floor-planted, self-collision-
free motion on Alex V2, suitable as reference motion for imitation / physics-RL.

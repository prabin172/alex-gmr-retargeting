# Contact-First Human-to-Humanoid Motion Retargeting — Method

*White-paper-style method section for the Alex V2 contact-first pipeline.*

## 1. Overview

We retarget human motion-capture onto the IHMC **Alex V2** humanoid (a
36-DOF floating-base biped). The pipeline maps a captured human performance to a
geometry-neutral *canonical human*, transfers it to Alex through a per-frame
inverse-kinematics (IK) solve that gives **priority to ground contact**,
temporally regularises the whole trajectory, and finally grounds the result on
the floor. The design goal is *contact-faithful* motion: where the human is
supported by the ground (feet flat, hands/fists pressing down), the robot
reproduces that support surface even when it means departing from the raw
captured limb orientation.

The stages are:

| # | Stage | Input → Output | Nature |
|---|-------|----------------|--------|
| 1 | Canonical human | capture → landmark trajectories | geometric re-expression |
| 2 | Orientation frames | positions → per-part $SO(3)$ frames + facing | geometric construction |
| 3 | Contact-first IK | canonical human → robot poses $q(t)$ | per-frame damped Gauss–Newton |
| 4 | Trajectory smoothing | $q(t)$ → smoothed $q(t)$ | global temporal regularisation |
| 4.5 | Z-grounding | $q(t)$ → grounded $q(t)$ | vertical projection to floor |
| 5 | Render | $q(t)$ → video / motion | visualisation |

## 2. Notation and conventions

- World frame: $+x$ forward, $+y$ left, $+z$ up. $\hat{z}=(0,0,1)$.
- Quaternions are stored $\mathrm{wxyz}$.
- Robot configuration $q = [\,p,\ \rho,\ \theta\,]$ with root position
  $p\in\mathbb{R}^3$, root quaternion $\rho\in S^3$, and $n_\theta=29$ actuated
  joints $\theta\in\mathbb{R}^{29}$; $\dim q = 36$.
- Velocity/tangent space has $n_v = 35$ coordinates: a $6$-vector root twist
  followed by the $29$ joint rates. Configuration updates use the manifold
  retraction $q \leftarrow q \boxplus \delta$ (MuJoCo `mj_integratePos`), which
  applies the exponential map on the root's $SE(3)$ component.
- $\mathrm{Log}:SO(3)\to\mathbb{R}^3$ is the matrix logarithm returning an
  axis–angle (rotation) vector; for $R$ with angle $\vartheta$,
  $\mathrm{Log}(R)=\dfrac{\vartheta}{2\sin\vartheta}\big(R_{32}-R_{23},\,R_{13}-R_{31},\,R_{21}-R_{12}\big)$.
- Canonical roles $r\in\mathcal{R}$: the 15 body landmarks
  (`pelvis, torso, head, {left,right}×{hip,knee,foot,shoulder,elbow,hand}`) plus
  4 contact sites (`{left,right}×{palm,sole}`).
- $x_r(t)\in\mathbb{R}^3$: canonical position of role $r$ at frame $t$;
  $R_r(t)\in SO(3)$: its semantic orientation frame (Stage 2). Frame $t_0$ is the
  clip's reference/rest frame.

## 3. Stage 1 — Canonical human

The captured skeleton (FBX/MVNX) is resampled onto a fixed set of canonical
roles, producing landmark trajectories $\{x_r(t)\}$ independent of the capture
rig. Only landmark *positions* are retained at this stage; all orientation
information used downstream is rebuilt geometrically in Stage 2 (raw capture
joint rotations are discarded, which removes rig-specific axis conventions).

## 4. Stage 2 — Semantic orientation frames

For each oriented role we construct an orthonormal frame $R_r(t)\in SO(3)$ from
the *geometry of neighbouring landmarks* rather than the capture's raw
rotations. Two landmark-difference directions define the primary and a secondary
axis; the third is their cross product, followed by Gram–Schmidt
orthonormalisation:
$$
\hat{u}=\frac{d_1}{\lVert d_1\rVert},\quad
\hat{w}=\frac{\hat{u}\times d_2}{\lVert \hat{u}\times d_2\rVert},\quad
\hat{v}=\hat{w}\times\hat{u},\qquad R_r=[\hat{u}\ \hat{v}\ \hat{w}].
$$
A global **facing yaw** $\psi$ is detected automatically (from the ground-projected
lateral shoulder/hip axis) and a yaw correction is applied so the performer faces
$+x$. This makes the whole clip expressible in the world convention of §2.

## 5. Stage 3 — Contact-first inverse kinematics

Each frame is solved independently by damped Gauss–Newton (Levenberg–Marquardt)
least squares over robot poses, warm-started from the previous frame.

### 5.1 Morphology-aware targets

Targets are built as **scaled deltas from a rest pose**, never as absolute human
coordinates, so limb-length mismatch does not teleport the robot. Let $a_r$ be
Alex's achieved rest position for role $r$ (from an initial rest solve), $t_0$ the
human rest frame, and define per-role scales comparing rest proportions:
$$
s_r=\mathrm{clip}\!\left(\frac{\lVert a_r-a_{\text{pelvis}}\rVert}
{\lVert x_r(t_0)-x_{\text{pelvis}}(t_0)\rVert},\ 0.4,\ 2.5\right),
$$
with a separate global root scale $s_{\text{root}}$. The **position target** for
role $r$ is
$$
p_r^\*(t)=a_r+\underbrace{s_{\text{root}}\big(x_{\text{pelvis}}(t)-x_{\text{pelvis}}(t_0)\big)}_{\text{root displacement}}
+\underbrace{s_r\big[(x_r(t)-x_{\text{pelvis}}(t))-(x_r(t_0)-x_{\text{pelvis}}(t_0))\big]}_{\text{pelvis-relative limb motion}},
$$
(the pelvis itself omits the third term). The **orientation target** uses
*world-delta transfer* — only the change of each part's frame relative to its rest
frame is copied onto Alex's rest orientation:
$$
R_r^\*(t)=\big(R_r(t)\,R_r(t_0)^{\!\top}\big)\,R_r^{\text{alex-rest}} .
$$

### 5.2 Per-frame least-squares system

At iterate $q_k$ we stack weighted residual rows into $A\,\delta = b$ with
$\delta\in\mathbb{R}^{n_v}$. Each task contributes rows $\sqrt{w}\,J$ (Jacobian)
and $\sqrt{w}\,e$ (residual):

- **Position tracking** (role $r$): $e = p_r^\*(t)-p_r(q_k)$, $J=J_r^{p}$ (body
  positional Jacobian).
- **Orientation tracking**: $R_{\text{err}}=R_r^\*(t)\,R_r(q_k)^{\!\top}$,
  $e=\mathrm{Log}(R_{\text{err}})$, $J=J_r^{\omega}$ (rotational Jacobian).
- **Contact axis alignment** (replaces the orientation term while in contact):
  align a body-fixed axis $a$ to a world direction $d$ via the small-angle error
  $e = (R\,a)\times d$, $J=J^{\omega}$. Feet: $a=$ foot up-axis, $d=+\hat z$
  (foot flat). Hands: $a=$ gripper $+x$ (palm normal of the closed fist),
  $d=-\hat z$ (fist pressing down). Only the *axis* is locked, so heading/yaw
  remains free for position tracking to determine.
- **Contact point pinning**: the palm/sole site is pulled to the contact anchor
  with weight $w_{\text{pos}}$, so the contact does not drift off the wrist body.
- **Posture regularisation**: $\sqrt{\mu}\,I$ on the actuated block with residual
  $\sqrt{\mu}\,(\theta_{\text{ref}}-\theta_k)$ (root DOFs left free), biasing the
  null space toward the warm-start pose.
- **Self-collision repulsion**: for each non-adjacent penetrating body pair
  (excluded within $k$ kinematic hops) with contact normal $\hat n$, a separation
  row $\hat n^{\!\top}(J_1^{p}-J_2^{p})$ pushes the pair apart with a
  margin/gain schedule.

The step solves the regularised normal equations
$$
\big(A^{\!\top}A+\lambda I\big)\,\delta = A^{\!\top}b,
$$
is clipped to a trust region $\lVert\delta\rVert\le\delta_{\max}$, scaled, and
retracted $q_{k+1}=q_k\boxplus\delta$; hinge joints are then clamped to their
limits. The root orientation is deliberately **not** constrained upright — for
get-up motions forcing an upright root makes the limbs absorb the discrepancy.

### 5.3 Contact detection and make/break blending

Contact is decided from the **human** data. An effector $e$ is in contact at
frame $t$ iff its lowest marker is within a height threshold $h_e$ of the clip
floor *and* moving slower than $v_{\text{thresh}}$:
$$
c_e(t)=\big[\ \min_{m\in e} z_m(t)-z_{\text{floor}} < h_e\ \big]\ \wedge\ \big[\ \lVert \dot x_e(t)\rVert < v_{\text{thresh}}\ \big],
$$
where $z_{\text{floor}}$ is a low percentile of the feet markers' height. A foot's
flat constraint is additionally gated to activate only when the foot tilt is below
$40^\circ$ (so a folded foot merely near the floor is not forced flat).

Raw $c_e$ would snap constraints on/off and jerk the pose. We therefore convert
$c_e$ to a continuous weight $\alpha_e(t)\in[0,1]$ by (i) debouncing short runs,
(ii) a look-ahead preroll, and (iii) a cosine cross-fade over $R$ frames. Contact
terms are scaled by $\alpha_e$ while the competing human position/orientation
terms are simultaneously scaled by $(1-\alpha_e)$ (the `pos_weight_scale` /
`ori_weight_scale` cross-fade), so support engages and releases smoothly rather
than as a binary switch.

## 6. Stage 4 — Global trajectory smoothing

Per-frame IK leaves velocity spikes and root pops. **Stage A** applies a
closed-form, per-channel temporal smoother. For a scalar channel $x_{1:T}$ it
solves
$$
\min_{y}\ \lambda_{\text{track}}\sum_{t}(y_t-x_t)^2+\lambda_{\text{smooth}}\sum_{t}(y_t-y_{t-1})^2,
$$
whose normal equations are the tridiagonal system
$$
\big(\lambda_{\text{track}} I+\lambda_{\text{smooth}} L\big)\,y=\lambda_{\text{track}}\,x,
$$
with $L$ the second-difference (path-graph Laplacian) operator: interior
diagonal $\lambda_{\text{track}}+2\lambda_{\text{smooth}}$, off-diagonals
$-\lambda_{\text{smooth}}$, endpoints $\lambda_{\text{track}}+\lambda_{\text{smooth}}$.
It is solved directly (banded/Thomas). All 29 actuated channels are smoothed and
clipped to joint limits.

Crucially the **floating base** is smoothed too, or the whole body flicks
(per-frame IK root has ~3 cm / 10° pops): root position uses the same tridiagonal
solve; the root quaternion is smoothed by (i) hemisphere-aligning consecutive
quaternions ($\rho_t\!\leftarrow\!-\rho_t$ if $\rho_t\!\cdot\!\rho_{t-1}<0$),
(ii) smoothing the four components independently, and (iii) renormalising.

An optional **Stage B** (a sparse global QP over all frames' joint increments,
with a second-difference smoothness Hessian, joint-limit and self-collision
constraints, and fixed per-interval contact anchors) is implemented but **off by
default**: labelled contact intervals in these clips are not true stationary
plants (feet/hands reposition tens of cm while "in contact"), so hard pinning is
ill-posed. Stage A smoothing is the shipped result.

## 7. Stage 4.5 — Z-grounding

The solved motion is planted on the floor $z=0$ as a post-step. For each frame we
compute the robot's true lowest world-space point over all collision geoms,
$$
z_{\min}(t)=\min_{g}\ \min_{\text{surface of }g}\ \big(\,\text{world }z\,\big).
$$
Mesh geoms (the convex-hull fists/limbs) are handled by transforming every hull
vertex, $z_g^{\text{low}}=\min_i\,(p_g+R_g v_i)\cdot\hat z$; primitives (box,
capsule, cylinder, sphere) use closed-form lowest-extent expressions. Only the
root height is modified, $p_z(t)\leftarrow p_z(t)+\Delta(t)$, in one of two modes:

- **Per-frame** (default): $\Delta(t)=-z_{\min}(t)$, planting every frame; the
  shift series is lightly smoothed with the Stage-A tridiagonal operator to
  suppress sub-cm jitter. Correctly handles sit-to-stand (the seat is planted
  while sitting, the feet once standing, since $z_{\min}$ tracks whichever is
  lowest).
- **Constant**: a single $\Delta=-\,\mathrm{percentile}_p\big(z_{\min}(t)\big)$
  for the whole clip, preserving all vertical motion but leaving the body above
  the floor except at its lowest moments.

Grounding is purely a vertical rigid shift of the free base, so joint angles and
horizontal motion are untouched.

## 8. Stage 5 — Rendering

The grounded trajectory is rendered in MuJoCo (EGL) alongside the canonical human
target stick figure and a per-frame contact indicator. Rendering can use either
the **collision** model (convex hulls — what the solver actually reasons about) or
the **visual** Alex mesh; in the visual model the hands are drawn as the closed
**fist** hull (the actual support surface) rather than the open manipulation hand,
which would visually pierce the floor during hand support. The ground plane is
pinned to $z=0$ to match the grounded motion.

## 9. Summary

The method's distinctive choices are: (i) geometry-rebuilt semantic orientation
frames with world-delta transfer, (ii) rest-relative morphology scaling that
never displaces the root absolutely, (iii) a contact-first IK that overrides
captured orientation with the physical support surface under a smooth make/break
cross-fade, (iv) whole-body (including floating-base) temporal smoothing, and
(v) a mesh-exact vertical grounding step. Together they produce contact-faithful,
temporally smooth, floor-planted motion on Alex V2.

# GMR-METHOD — How This Actually Works

Plain-language method description for the G1/GMR baseline work (branch
`gmr-baseline`). Covers: what the motion data is, how it gets converted into
a robot motion, which parts are GMR's (untouched) and which parts are ours,
and what to actually expect from the result. Written to be readable without
the codebase open.

This is a different pipeline from `METHOD.md` (that one is the older
Alex/FBX pipeline, a 36-DOF IHMC biped). This document is the Unitree
G1/LAFAN1/GMR pipeline this whole sprint has been about.

---

## 1. The goal, in one sentence

Take human motion-capture clips, retarget them onto the Unitree G1 humanoid
robot, and make the robot's feet/hands actually behave physically at the
ground — without inventing a whole new retargeter, by adding a small,
inspectable stack of fixes on top of an existing open-source one (GMR).

## 2. The data

- **Source**: LAFAN1, a public motion-capture dataset (BVH format — skeleton
  + per-frame joint rotations), the same dataset GMR's own paper evaluates
  on.
- **Scope**: all 77 LAFAN1 clips used in GMR's evaluation set. No cherry-
  picking — every clip that has a valid mapping goes into every corpus run.
- **Two classes, for reporting only** (not two different pipelines): we
  split the 77 clips into **floor-class** (34 clips — falls, get-ups,
  crawling, anything that touches hands/knees to the ground) and
  **locomotion-class** (43 clips — walking, running, jumping, fighting,
  etc. — normal upright gaits). The split itself is detected automatically
  from the human motion (see step 3), not hand-labeled.

## 3. Step 1 — turning a BVH clip into a "canonical human" (ours)

Before any retargeting happens, each BVH clip is converted into our own
intermediate format: a per-frame skeleton in a robot-agnostic shape (which
joint is where, in world coordinates, plus body-part orientation), and —
critically — a per-frame **contact label** for four body parts: left/right
foot, left/right hand.

- **What GMR gives us here**: only the raw BVH reader (`load_bvh_file`) —
  a utility that reads the file into per-frame `{bone_name: position,
  rotation}` dictionaries. We don't touch that reader.
- **What's ours**: everything after that.
  - Mapping LAFAN1's bone names onto our own role names (pelvis, torso,
    knees, ankles, wrists, etc.).
  - Building an orientation frame for each body part from raw landmark
    positions (not from the mocap rig's own bone rotations, except for
    hands — see below) — this is deliberately geometry-first so it isn't
    tied to any one mocap rig's bind pose.
  - Hands are the one exception: we use the BVH's own raw wrist bone
    rotation directly, because a landmark-only frame can't represent wrist
    twist (there's no second hand bone in LAFAN1 to build a twist axis
    from). This matches what GMR's own retargeter targets for hands, so
    it's not introducing a mismatch, just filling a gap.
  - **Contact detection**: for every frame, we check whether the human's
    foot, hand, knee, elbow, pelvis, or torso is low enough to the ground
    to count as "in contact" — a simple height threshold on the human's
    own body, not on any robot output. This is what gives us the
    floor-class/locomotion-class split, and it's also the signal every
    later "held" / "planted" decision in the pipeline is built from.

Output: one NPZ per clip containing positions, orientations, and contact
flags. This is the shared input to everything downstream.

## 4. Step 2 — the base retarget (GMR, unmodified)

This is the one stage where we deliberately do nothing. `GeneralMotionRetargeting`
is GMR's own retargeter: a per-frame inverse-kinematics solve (a QP, via the
`mink` library) that tracks human landmark targets on the G1 robot, with
hard joint limits and orientation-first weighting. We call it exactly as
GMR ships it — read-only, no edits — and it produces the "raw" robot motion
(`gmr_raw`) for every clip.

GMR also ships its own fix for one obvious problem: a raw retarget commonly
has the robot's feet floating above or sinking below the floor, because the
IK has no idea where the floor is. GMR's fix is a single constant vertical
shift per clip (move the whole clip up or down by one fixed amount, chosen
so the worst frame just touches the floor). We use this exactly as GMR
built it too, and treat it as the baseline we compare against — we call it
**`gmr_heightfix`** ("GMR-full") everywhere in results. It is not part of
our pipeline; it's the thing our pipeline is measured against.

We also reused one more GMR asset as-is: their own hand-vetted collision
geometry for the G1 (a simplified capsule/cylinder collision model, built
by GMR to replace the noisy full-body collision mesh). We didn't design
this geometry; we did have to notice that G1's shipped model has a
duplicate full-mesh collision copy of every body that was producing fake
self-collision noise, and swap in GMR's vetted geometry in its place. That
swap is the only thing we changed about the robot model itself.

## 5. Step 3 — a per-frame contact-and-floor clamp (ours)

This is the first stage that's genuinely new. GMR's raw retarget has zero
per-frame floor or self-collision awareness — its only floor mechanism is
the one constant shift described above, and that shift can't fix a foot
that clips through the floor at one specific moment while another frame in
the same clip is floating in the air.

So, frame by frame, on top of GMR's raw output, we run our own small
inverse-kinematics correction:

- For every limb (each foot, each hand, plus knee/hip/elbow as secondary
  watch points), check whether any part of it has gone through the floor
  or into another part of the robot's own body.
- If a foot or hand is meant to be planted (per the human contact label
  from step 1), pull it back to a locked position so it doesn't skate.
- If a limb is swinging free, just push it back above the floor / away
  from self-collision, with the smallest joint change that fixes it (a
  damped least-squares correction, not a full re-solve).

This produces the variant we call **`perframelimb`** — GMR's own retarget,
corrected frame by frame so it never floats through the floor or through
itself. This step is retarget-agnostic; it doesn't know or care that its
input came from GMR specifically.

## 6. Step 4 — smoothing, then re-fixing what smoothing broke (ours)

Fixing every frame independently, one at a time, has a cost: consecutive
frames can each be individually correct but disagree with each other,
producing a small jump/spike in joint velocity from frame to frame.

- We run a temporal smoother over the whole clip (a closed-form banded
  solve, minimizing frame-to-frame jerk) — but any foot or hand that's
  supposed to be planted is locked hard to its input value while smoothing
  happens elsewhere, so a "held" contact never gets smoothed away.
  (This smoother itself isn't new — it's the same smoothing machinery from
  our earlier Alex-robot pipeline, reused here rather than rewritten.)
- Smoothing can, in turn, nudge a limb back through the floor or into
  self-collision (it doesn't know about geometry, only about joint
  trajectories). So step 3's per-frame clamp runs a second time, ONLY on
  top of the smoothed result, to restore floor/collision safety.

This two-part result (smooth, then re-clamp) is what we call **`smrc`**
("smooth + re-clamp"): `perframelimb_smrc`.

## 7. Step 5 — local grounding (ours)

Even after step 6, a handful of individual frames per clip can still dip
slightly through the floor (typically during a fast, transient moment —
a fall, a jump apex — not during a stance). Rather than shifting the whole
clip up by a constant amount (which is GMR's own trick, and which we found
overcorrects: it fixes the transient but then floats every OTHER,
already-fine frame too high), we compute a **per-frame** vertical lift that
is zero almost everywhere and only rises exactly where and when a frame
actually penetrates the floor, tapering smoothly in and out around each
event. This guarantees zero floor penetration by construction — it's an
algebraic property of how the lift is built, not something we had to tune
to achieve — while leaving clean stance frames completely untouched.

## 8. Step 6 — capping the correction speed (ours)

One thing step 4's re-clamp (step 6) still does: it corrects each frame
independently within a bounded step size, and that correction can itself
change abruptly from one frame to the next, adding a small amount of
avoidable joint-velocity spiking. The final piece caps how fast that
correction is allowed to change frame-to-frame (a simple rate limiter on
the applied correction, not on the tracking itself), so a frame that needs
a big fix doesn't apply all of it in one instant.

## 9. The final, locked pipeline

```
BVH clip
  -> canonical human + contact labels                [ours]
  -> GMR raw retarget                                  [GMR, unmodified]
  -> per-frame floor/self-collision clamp              [ours]  = perframelimb
  -> held-aware smoothing                              [ours, reused from Alex pipeline]
  -> re-clamp (restore floor/collision after smoothing) [ours]  = perframelimb_smrc
  -> rate-limited re-clamp (cap correction speed)       [ours]
  -> local grounding envelope (zero floor penetration)  [ours]  = perframelimb_smrc_rl_localground
```

This last name — **`perframelimb_smrc_rl_localground`** — is the current
locked variant. "Locked" means: this is the one we report against GMR-full
going forward, until/unless a specific new idea is tried and shown to beat
it on the full 77-clip corpus.

The separate comparison baseline, **`gmr_heightfix`** ("GMR-full"), is
GMR's own retarget plus GMR's own single-shift grounding trick — nothing of
ours is in it. It exists purely so we have something fair to measure
against: same robot, same IK, same dataset, the one difference being
whether floor/contact correctness is handled per-frame (ours) or by one
global height shift (theirs).

## 10. How we measure it (ours)

All of the following are our own evaluation code, run identically over
every variant on all 77 clips:

- **Floor penetration** — the single worst frame's depth below the floor,
  across the whole robot mesh, for the whole clip.
- **Float** — how far a foot/hand sits above the floor during frames where
  the human intended it to be planted.
- **Range** — float minus penetration, i.e. how wide the float/penetration
  window is. This one is deliberately shift-invariant: a global vertical
  shift can move the whole window up or down but can't shrink it, so it
  measures genuine contact quality rather than a placement choice.
- **Self-collision %** — how often, and how deeply, the robot's own mesh
  passes through itself.
- **Tracking error** — position (cm) and orientation (deg) difference
  between the robot's achieved pose and GMR's own human-derived target, at
  the same body points GMR itself targets. This is the "did we copy the
  human faithfully" number.
- **Foot slip / hand slip** — for any frame where a foot or hand is
  supposed to be planted, how far it drifts in the horizontal plane while
  it's meant to be still.
- **Joint velocity spikes / peak velocity (vMax) / jerk** — smoothness of
  the output motion; a spike is a frame-to-frame joint velocity jump big
  enough to be physically implausible for a real actuator.
- **joint_ok%** — the one composite, "hardest to game" number: the
  fraction of contact frames where the robot is BOTH not penetrating the
  floor (whole body) AND within a tight (±3cm) band of correct float/
  penetration at the planted point, simultaneously. A method can cheat any
  one of the metrics above with a vertical shift; it can't cheat this one,
  because a shift that fixes penetration will push float out of the tight
  band, and vice versa.

## 11. What to actually expect (results snapshot, 77-clip corpus)

Averages for the locked variant (`perframelimb_smrc_rl_localground`), full
CSV per clip in `outputs/gmr_baseline/sprint/s8_LOCKED_perframelimb_smrc_rl_localground.csv`:

| Metric | Locked variant (ours) | GMR-full (`gmr_heightfix`) |
|---|---|---|
| Floor penetration | 0.00 cm | ~2.8-3.1 cm |
| Self-collision % | ~0.001% | ~4-6% |
| Worst float (held frames) | ~5.6-7.5 cm | ~6.6-18.0 cm |
| joint_ok% (the hard one) | ~98.8% | 0.2% (floor) / 46% (loco) |
| Peak joint velocity (vMax) | ~37.7 rad/s | ~33-34 rad/s |
| Velocity spikes/clip | 0.00 | ~0.0-0.2 |
| Tracking error (position) | ~12.6 cm | ~11.1-12.6 cm |
| Foot slip | ~0.39-0.47 cm | ~0.31-0.44 cm |
| Hand slip (only clips with a hand-hold) | ~0.66-0.73 cm | — (GMR-full has no hand-contact handling) |

Read plainly: our method wins decisively on the metric that can't be
gamed by a vertical shift (`joint_ok%`, self-collision, and float
simultaneously), essentially ties on foot slip and tracking error, and is
still slightly behind on raw peak joint velocity — closer than it used to
be, but not fully closed. That gap is a real, open, and currently
unaddressed cost of correcting contacts per-frame instead of once per clip.

## 12. Appendix — the mathematics behind each `[ours]` step

This section derives the corrections from §§5–8 in full. Skip it if the
plain-language description above already answered your question — nothing
here changes the conclusions in §§9–11, it just shows the algebra. Notation
follows this project's convention throughout: `qpos ∈ ℝ^{36}` is
`[x, y, z, qw, qx, qy, qz, 29 joints]` (root free-joint pose + actuated
joints), quaternions **wxyz**. `δ`/`dq` denotes a joint-space correction
restricted to one limb chain's DOFs unless stated otherwise.

### 12.1 Step 3 — the per-frame contact-and-floor clamp (`leg_floor_clamp.clamp_limb`)

For one limb's kinematic chain (hip→ankle, 6 hinge DOFs; or shoulder→wrist,
7 hinge DOFs) correction proceeds in two **sequential** phases, never mixed
into one solve — mixing was tried and found to destabilize convergence
(see below).

**Phase 1 — floor / held-position convergence.** Let `p` be the chain's
watched body's lowest mesh point (exact for a mesh geom; closed-form for
primitive geoms — sphere/capsule/cylinder/box) and `z = p_z`.

*Clearance-only* (a swinging, non-held limb): if `z ≥ floor_margin`, no
correction. Otherwise, with `J = ∂p_z/∂q` restricted to this chain's DOFs
(the z-row of the positional Jacobian, `mj_jac`) and scalar error
`e = floor_margin − z`,

$$\delta = J^{\top}\big(JJ^{\top}+\lambda^2 I\big)^{-1} e,\qquad \lambda=10^{-3}.$$

*Held mode* (a locked foot/hand, a target `(x,y)` given): the error is
3-row — `(x, y)` at the body's **origin** `cur` (a different world point
than `p` for a rotated/offset foot, so its Jacobian is queried separately)
and `z` at the lowest point `p`:

$$e=\begin{bmatrix}\text{target}_{xy}-\text{cur}_{xy}\\ \text{floor\_margin}-z\end{bmatrix},
\quad J=\begin{bmatrix}J^{p}_{xy}(\text{cur})\\ J^{p}_{z}(p)\end{bmatrix},
\quad \delta=J^{\top}(JJ^{\top}+\lambda^2I)^{-1}e.$$

Either branch: `qpos[chain] += δ` (optionally clipped elementwise to
`±max_dq`), clip to joint range, re-linearize and repeat — bounded
Gauss–Newton, `max_iters = 10` by default. (Near a saturated joint limit,
each step's remaining free DOFs carry more of the correction, so deep
corrections genuinely need more steps; confirmed down to sub-mm residuals
on hip/knee-at-limit frames.)

**Phase 2 — self-collision (opt-in).** Runs *after* phase 1 has converged
(or exhausted its iterations), reusing the SAME chain DOFs, over every
currently-active non-floor, non-anatomically-adjacent contact pair (read
straight from MuJoCo's own contact list after `mj_forward`). For contact
`c` with penetration `pen_c = \text{margin} - \text{dist}_c > 0`, normal
`n_c`, and the **relative** Jacobian of the two contacting bodies
restricted to this chain,

$$J_c = n_c\cdot\big(J_{b_1}-J_{b_2}\big)\big|_{\text{chain dofs}},\qquad
e_c=\min(pen_c,\,0.05\text{ m}),$$

stacked over all active pairs into one damped least-squares solve per
iteration:

$$\delta = w_{\text{coll}}\,J_c^{\top}\big(J_cJ_c^{\top}+\lambda^2 I\big)^{-1} e_c,
\qquad w_{\text{coll}}=0.5\ \text{(shipped)}.$$

**Why sequential, not mixed.** Stacking floor/held and collision rows into
one weighted solve was tried first: on a full clip, one badly-linearized
frame's collision term destabilized the floor term's convergence and
cascaded through every later frame's warm start — floor penetration got
**worse than doing nothing** (4.74→38.76cm on one dev clip). Phase 2
strictly after phase 1 has already converged never shows this failure —
confirmed the same way (a "phase-3 floor mop-up" re-run after phase 2 was
also tried and also made things worse, for the identical cascading-warm-
start reason, and is not shipped).

**Ordering across the whole body.** Corrections are applied
proximal-to-distal (hip→knee→ankle, shoulder→elbow): a chain's Jacobian at
an upstream point has exactly zero columns for downstream-only DOFs, so an
upstream correction can never be silently undone by a later, distal-only
correction on the same chain.

### 12.2 Step 4 — held-aware smoothing (reusing the Alex pipeline's tridiagonal solver)

Each of the 29 actuated channels, plus root position and (separately) root
quaternion, is smoothed **independently** by the same closed-form
tridiagonal solve as the Alex pipeline (`METHOD.md` §6.1):

$$\min_{y}\ \sum_t \lambda_{\text{track}}(t,j)\,(y_t-x_t)^2 \;+\;
\lambda_{\text{smooth}}\sum_t (y_t-y_{t-1})^2,$$

whose normal equations are the tridiagonal system
`(diag(λ_track) + λ_smooth·DᵀD) y = λ_track ⊙ x` (`DᵀD` the path-graph
Laplacian), solved in `O(T)` via a banded solver. Shipped defaults:
`λ_track = 1.0`, `λ_smooth = 20.0` (30 fps native — no rate-rescale applied
here, unlike the Alex pipeline's 120 Hz table, §6.4 there).

**What makes it "held-aware"**: `λ_track(t, j)` is not a constant — it is a
*per-frame, per-joint* weight matrix. On a held leg's own 6 chain DOFs
(and, via a per-frame max over columns, the root), it ramps from
`λ_track` toward `λ_lock = 10^8` (effectively an equality pin) over a
5-frame cosine window at hold onset/release:

$$\text{frac}(a)=\tfrac12\big(1-\cos(\pi a/5)\big),\quad a\in[0,5],\qquad
w = \lambda_{\text{track}} + (\lambda_{\text{lock}}-\lambda_{\text{track}})\,\text{frac}(a).$$

Free DOFs (arms, waist, and the root during non-held stretches) stay at
`λ_track`, fully governed by the smoothing term. One exception: if the RAW
input already has a per-joint velocity spike (`|Δq|·fps > 40` rad/s) at a
transition inside a held/ramp window, both endpoint frames' weight is
dropped back to `λ_track` for that joint only, so the solve is free to
interpolate through a genuine artifact instead of faithfully preserving it
— a plain lock would otherwise reproduce the spike exactly, since
`λ_lock` is high enough to pin the channel to its own noisy input.

### 12.3 Re-clamp (the "rc" in `smrc`)

Exactly §12.1's clamp (phase 1 + phase 2, `w_coll = 0.5`, `max_dq = 0.15`),
applied a second time to the *smoothed* output. Necessary because §12.2's
solve has zero geometry awareness — it can only see joint angles, not
meshes — and a tridiagonal blend toward neighbouring frames can reintroduce
the exact floor/self-collision violations phase 1 of the original clamp
had already zeroed.

### 12.4 Step 5 — local grounding envelope

Let `z_min(t)` be the whole-robot-mesh lowest point at frame `t` (the min
over every watched body's own lowest point, same primitive as §12.1's
lowest-point computation), and `required(t) = max(0, −z_min(t))` — the
exact vertical shift that alone would clear frame `t`. Two filters compose
into an envelope that is provably `≥ required` everywhere:

$$\text{widened} = \text{max\_filter}_{1d}\!\big(\text{required},\ \text{width}=2h{+}1\big),
\quad h=\lfloor 0.15\cdot\text{fps}\rfloor,$$
$$\text{smoothed} = \text{gaussian\_filter}_{1d}(\text{widened},\ \sigma=0.07\cdot\text{fps}),$$
$$\text{envelope}(t)=\max\big(\text{smoothed}(t),\ \text{required}(t)\big).$$

A max filter can only increase values (its window always includes `t`
itself), so `widened ≥ required` pointwise by construction. Gaussian
smoothing *can* dip back below that — hence the final pointwise max against
`required`, which restores the guarantee everywhere the Gaussian step
undershot. The result: `envelope(t) ≥ required(t)` for every `t`, so

$$z_{\min}(t) + \text{envelope}(t) \;\ge\; 0 \quad \forall t$$

is an **algebraic identity** of the construction — floor penetration = 0.00
follows from the shape of the formula, not from tuning any threshold.
Applied as `qpos_z(t) += envelope(t)`, root height only, per frame — NOT a
rigid whole-clip shift. (Contrast GMR's own single-constant shift, §4: a
constant sized to clear one clip's single worst, usually transient, frame
was tried directly on top of the clamp+smooth pipeline and found to
overshoot the tight tolerance band around ordinary, already-correct stance
frames — this per-frame envelope is what fixes that.)

### 12.5 Step 6 — rate-limited re-clamp

A rate limiter wraps §12.3's re-clamp call. Let
`c_des(t) = q_corrected(t) − q_ref(t)` be that frame's *desired* total
correction (joint space, against the frame's pre-clamp pose), and `c_prev`
the correction actually **applied** on the previous frame:

$$c_{\text{app}}(t) = c_{\text{prev}} + \operatorname{clip}\big(c_{\text{des}}(t)-c_{\text{prev}},\,-r,\,+r\big),
\qquad r = 0.15\text{ rad/frame},$$
$$q_{\text{limited}}(t) = q_{\text{ref}}(t) + c_{\text{app}}(t),\ \text{clipped to joint range};
\qquad c_{\text{prev}} \leftarrow c_{\text{app}}(t).$$

(First frame: no previous frame, so `c_app = c_des` — the full correction
applies, at zero velocity cost.) This bounds how fast the *applied
correction itself* can change frame-to-frame (`≤ r·fps` rad/s) — a
different quantity from §12.1's `max_dq`, which caps a single Gauss–Newton
*iterate* within one frame's solve. The rate limiter caps the *total*
correction's drift across frames, which is what actually produces velocity
spikes at solver-branch flips (a correction jumping between two very
different DLS solutions on consecutive, nearly-identical frames).

When rate-limiting is active, phase-2 self-collision does **not** run
inline inside phase 1's loop — it runs as a separate, un-limited pass
after the limiter has been applied, one call per chain. Rate-limiting both
phases together was tried first and suppressed the self-collision
correction back toward its pre-fix contamination level; phase 2 accounts
for a small fraction of the total spikes this step targets, so leaving it
unlimited does not reintroduce the problem the rate limiter exists to
solve.

## 13. Limitations, stated plainly

- **vMax is still ~10-16% higher than GMR-full's own baseline.** This is
  the one metric where GMR-full genuinely wins, and it hasn't been closed.
- **Tracking fidelity is a deliberate tradeoff, not a bug.** Prioritizing
  contact correctness over verbatim copying means the robot sometimes
  departs slightly from GMR's own human-derived target to keep a foot
  planted or avoid self-collision. The tracking-error numbers reflect that
  choice; they are not evidence of a broken retarget.
- **This has only been checked kinematically.** Nothing here has been
  tested through a physics simulator or a learned imitation policy yet —
  that's a separate, later gate (mimic-training pilot), and a kinematic win
  here does not guarantee a win there.
- **The method is a chain of independently-motivated fixes, not one
  unified algorithm.** Each stage (per-frame clamp, smoothing, re-clamp,
  local grounding, rate limiting) was added because a specific, measured
  problem remained after the previous stage — this is a strength for
  explainability (every stage's job is inspectable and its cost/benefit is
  logged) but it does mean the "method" is a pipeline, not a single clean
  optimization.

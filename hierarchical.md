# Project Plan: Robust Humanoid Motion Retargeting via HQP-NLP Cascade

## 1. Executive Summary

This project outlines a rigorous pipeline for retargeting complex, multi-contact human motions (e.g., getting up from the ground) to a humanoid robot. Traditional weighted optimization (NLP) or RL approaches often struggle with the embodiment gap, requiring endless tuning to balance contact stability against posture tracking.

This pipeline solves this by decoupling strict physics from global smoothness using a three-phase approach:

1. **Phase 0:** Contact Management (Extracting logical constraints).
    
2. **Phase 1:** Hierarchical Quadratic Programming (HQP) to guarantee physical feasibility frame-by-frame.
    
3. **Phase 2:** Global Non-Linear Programming (NLP) to smooth the trajectory and enforce actuation limits.
    

Also, note the current results that when we try to impose COM, we found there is 40cm error (what?). So tracking human might not be the best way??
## 2. Phase 0: Data Preprocessing & Contact Management

Before optimization begins, the raw human mocap data must be translated into the robot's kinematic space and segmented into logical contact phases.

- **Kinematic Scaling:** Map human joint positions to the robot using segment-based scaling (e.g., matching vector directions rather than absolute lengths) to generate the baseline reference motion $q_{ref}$.
    
- **Pseudo-Contact Labeling:** Calculate the velocity and height of key end-effectors (Left/Right Foot, Left/Right Palm, Left/Right Knee) relative to the ground.
    
- **Hysteresis Filtering (Crucial):** To prevent solver jitter, pass pseudo-labels through a hysteresis filter. An end-effector must meet contact thresholds for $N$ consecutive frames to be labeled "Active," and fail thresholds for $M$ consecutive frames to be labeled "Inactive."
    

## 3. Phase 1: Frame-by-Frame HQP (The Strict Hierarchy)

For each frame $t$, solve a cascade of QPs to generate physically viable joint velocities $\dot{q}_{hqp}$. The hierarchy guarantees that lower-priority tasks can never violate higher-priority physics constraints.

### Task 1: Environmental Contacts (Highest Priority)

- **Goal:** Keep active contacts planted; prevent inactive bodies from penetrating the floor.
    
- **Math:** Define $J_1$ as the Jacobian for all currently active contact sites.
    
    - Minimize: $\|\dot{q}_1\|^2$
        
    - Subject to (Equality): $J_1 \dot{q}_1 = 0$ (Zero velocity at contacts).
        
    - Subject to (Inequality): $z_{bodies} > 0$ (Non-penetration).
        
- **Output:** Joint velocities $\dot{q}_1$ and the first null-space projector $N_1 = I - J_1^+ J_1$.
    

### Task 2: Center of Mass (CoM) & Balance

- **Goal:** Move the CoM to maintain balance over the active support polygon.
    
- **Math:** Define $J_2$ as the CoM Jacobian. Project it into the null space of Task 1: $\tilde{J}_2 = J_2 N_1$.
    
    - Minimize: $\| \tilde{J}_2 \dot{q}_2 - (\dot{x}_{com\_desired} - J_2 \dot{q}_1) \|^2$
        
    - Subject to: $\dot{q}_2 = N_1 \dot{q}_2$
        
- **Output:** Balance velocities $\dot{q}_2$ and the second null-space projector $N_2 = N_1(I - \tilde{J}_2^+ \tilde{J}_2)$.
    

### Task 3: Postural Tracking (Lowest Priority)

- **Goal:** Track the scaled human reference motion $q_{ref}$ using remaining degrees of freedom.
    
- **Math:** Define $J_3$ as the postural Jacobian. Project it into the null space of Task 2: $\tilde{J}_3 = J_3 N_2$.
    
    - Minimize: $\| \tilde{J}_3 \dot{q}_3 - (\dot{q}_{ref} - J_3(\dot{q}_1 + \dot{q}_2)) \|^2_{W}$ (where $W$ is a diagonal weighting matrix prioritizing spine/torso over free limbs).
        
    - Subject to: $\dot{q}_3 = N_2 \dot{q}_3$
        
- **Final Frame Output:** $\dot{q}_{hqp} = \dot{q}_1 + \dot{q}_2 + \dot{q}_3$.
    

## 4. The Escape Hatches (Handling HQP Failures)

Strict HQP is brittle. As the robot transitions between contacts or approaches kinematic limits, the math will attempt to divide by zero or crash if constraints conflict. The following "escape hatches" must be implemented in Phase 1:

### A. Slack Variables (For Infeasible Contacts)

- **Problem:** Task 1 requests a mathematically impossible pose (e.g., foot contact conflicts with joint limits).
    
- **Solution:** Convert hard equality constraints into soft constraints with massive penalty weights. Instead of $J_1 \dot{q}_1 = 0$, formulate as $J_1 \dot{q}_1 = \epsilon$. Minimize $\epsilon$ with a weight of $10^6$. The solver will violate the contact by sub-millimeters rather than crashing.
    

### B. Damped Least Squares (For Singularities)

- **Problem:** A joint locks out (e.g., straight knee), causing the Jacobian to lose rank. The pseudo-inverse $J^+$ explodes toward infinity, causing violent velocity spikes.
    
- **Solution:** Replace standard pseudo-inverses $J^+ = J^T(JJ^T)^{-1}$ with Damped Pseudo-inverses $J^+_{damped} = J^T(JJ^T + \lambda^2 I)^{-1}$. The damping factor $\lambda$ guarantees the denominator never hits zero, sacrificing microscopic accuracy to maintain numerical stability.
    

### C. Continuous Activation Parameters (For Contact Transitions)

- **Problem:** When a new contact is added, $N_1$ instantly shrinks. This discontinuous jump in the available null space causes a severe jerk in the calculated velocities.
    
- **Solution:** Introduce an activation scalar $\alpha \in [0, 1]$ that ramps up over $K$ frames when a contact is detected. Blend the tasks smoothly: $J_{active} = \alpha J_{contact}$.
    

### D. Task Starvation

- **Problem:** Tasks 1 and 2 consume all degrees of freedom, leaving $N_2 = 0$. The robot ignores the human posture reference entirely.
    
- **Solution:** _Do nothing in Phase 1._ This is the intended behavior of a strict hierarchy (physics > aesthetics). Phase 2 will resolve the aesthetic visual jarring.
    

## 5. Phase 2: Global Trajectory Optimization (NLP)

Phase 1 produces a sequence of states ($q_{hqp}$, $\dot{q}_{hqp}$) that are strictly physics-compliant but temporally jerky (due to the greedy, local nature of QPs and dynamic contact switching). Phase 2 smooths this into a deployable trajectory.

- **Warm Start:** Initialize the NLP solver using the entire trajectory generated by Phase 1. (This prevents the NLP from getting stuck in local minima).
    
- **Cost Function:**
    
    1. **Smoothness:** Minimize $\sum \|\ddot{q}\|^2$ (or joint jerk) over the whole trajectory.
        
    2. **Tracking:** Minimize $\sum \|q_{nlp} - q_{hqp}\|^2$. (Track the physics-corrected HQP data, _not_ the raw human mocap).
        
- **Constraints:**
    
    1. **Actuation Limits:** Enforce strict motor torque ($\tau_{max}$) and velocity limits.
        
    2. **Kinematic Limits:** Enforce joint position limits ($q_{min} < q < q_{max}$).
        
    3. **Relaxed Contacts:** Model the hard contacts from Phase 1 as spring-damper constraints, allowing the solver to permit microscopic foot slip if it drastically reduces overall joint jerk.
        

## 6. Recommended Tech Stack

- **Rigid Body Dynamics / Kinematics:** Pinocchio (C++/Python) - Extremely fast for calculating Jacobians ($J$) and Mass matrices.
    
- **HQP Solver:** eiquadprog or OSQP (configured for hierarchical cascading).
    
- **NLP Solver:** CasADi + IPOPT, or Crocoddyl (Differential Dynamic Programming, excellent for whole-body trajectory optimization).

---

## Claude's comment (2026-07-11 review with Prabin — rulings marked SETTLED)

Reviewed against the phasic-v2 outcome (`planLog.md`, `wiki/experiments/phasic-v2-*`). Verdict:
the hierarchy idea is worth building, but as hard constraints inside the existing MuJoCo/OSQP
stack — not a null-space HQP on a new Pinocchio/CasADi stack. Roughly half of this doc already
exists in the repo. Execution plan: `plan.md` (repo root, "Hierarchical cascade v1").

### Already built — do not rebuild
- **Phase 0** ≈ `scripts/contact_labels.py` (phasic-v2 M1): height/speed gates + onset
  hysteresis, shared by Stage 3. Verify the N/M debounce params; port only if missing.
- **Kinematic scaling** = settled morphology scaling (motion deltas only — CLAUDE.md conventions).
- **Phase 2** ≈ Stage 4 GlobalOPT + M4 `physics_plausibility_pass.py` (vel/accel bounds).

### Rulings (SETTLED — do not re-litigate)
1. **CoM tier (Task 2): DROPPED.** Prabin: neglect CoM violations. M4 measured ~40 cm
   CoM-outside-polygon "violations" on get-up transitions that were legitimate momentum-based
   motion — quasi-static CoM-in-polygon is the wrong instrument for dynamic clips. The margin
   note above ("40cm error (what?)") is exactly this finding, not an error to fix.
2. **Zero-slip contacts: KEPT, upgraded to a design GOAL.** Prabin: "humans might have slipped,
   but robots cannot, and we don't want them to — that's the point." This supersedes the earlier
   "a cm of slip is learnable" stance in `wiki/concepts/design-philosophy.md` for active-contact
   frames: contacts get a hard no-slip constraint, not a weighted penalty. Consequence to accept:
   on clips where the human genuinely slides a contact (e.g. `standup_slideHandsBack_03`), the
   retargeted motion will deliberately deviate from the human — labels must segment sliding
   contact into re-plants, and those clips need visual review.
3. **Strict null-space HQP: NO.** The escape hatches (slack@1e6 ≈ high-weight soft row; damped
   pseudo-inverse ≈ regularization; activation ramp ≈ existing `contact_ramp`) collapse a strict
   hierarchy back into a weighted QP plus extra machinery. And with the CoM tier dropped there is
   only ONE objective tier left (tracking), so the "hierarchy" degenerates to: hard constraints
   (contacts, floor, joint limits) + soft tracking objective = **a single QP per frame**. No
   cascade, no null-space projectors, no pseudo-inverses — escape hatch B unnecessary by
   construction. Hatch C (activation ramp) already exists and stays mandatory (the wrist-flick
   lesson).
4. **Position-level, not velocity-level.** Integrating q̇ drifts anchor positions over 800–4000
   frame clips even at exactly zero contact velocity. Keep Stage 3's position-level IK with
   pinned anchors. (As written, Task 1's minimizer is trivially q̇₁=0, and z>0 is a position
   constraint inside a velocity QP — sketch-level issues that confirm re-derivation was needed
   anyway.)
5. **Stack: stay MuJoCo/OSQP.** Pinocchio/CasADi/Crocoddyl means MJCF→URDF conversion and
   re-deriving all the mesh-accurate floor/self-collision machinery built in MuJoCo. Rejected.
6. **Torque limits: BLOCKED on data.** `alex_floating_base_with_sites.xml` has no `<actuator>`
   section (confirmed at M4, read-only). Until IHMC provides actuator specs, M4's observed-peak
   vel/accel bounds are the stand-in; the Phase-2-as-NLP upgrade is deferred.
7. **Cheap probe first.** The one real gap phasic-v2 left (7/20 whole-body-lying clips beyond a
   root-frozen solver) already has a named, untested fallback: allow a root-z DOF in P4 round 2
   (plan.md's own Risks section). Days not weeks; run it before the Stage-3 restructure.
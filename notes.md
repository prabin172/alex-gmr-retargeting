# Two-pass contact-aware IK plan

## Goal
Make the solver keep the body near the full-body target while also enforcing contact discipline and avoiding floor penetration, without letting the wrist absorb the correction and flick.

## Core idea
Do a two-pass solve per frame:

1. Whole-body pass
   - Solve the full body from the current target pose.
   - This gives a near-target posture and keeps the body coherent.
   - Use a strong but not overly aggressive floor/contact term here.
   - The goal is to get the root/pelvis/legs/torso near the intended pose.

2. Contact refinement pass
   - Starting from the result of pass 1, refine only the limbs that are actually involved in contact or near contact.
   - For each relevant limb, solve from the shoulder/hip chain outward.
   - Keep the wrist and hand as soft targets, not the main correction handle.
   - Enforce palm/foot contact discipline and floor-safe placement.

This keeps the wrist from becoming the slack variable that causes the flicks.

## Why this could help
The current issue looks like a priority problem:
- the floor correction is being resolved partly through the wrist/hand chain,
- which creates fast, visible joint spikes.

With a two-pass strategy:
- the first pass solves the whole body near the target,
- the second pass uses the limb chain to satisfy contact constraints and safe placement,
- so the arm/hand can adapt more smoothly through shoulder/elbow rather than snapping through wrist.

## Proposed implementation shape
### Pass 1: whole-body solve
Reuse the existing frame-level IK solve as the first pass.

- Keep current body targets and orientation targets.
- Keep the floor repulsion term active.
- Keep contact targets soft-to-medium weight.
- Do not let the wrist be the primary actuator for floor correction.

### Pass 2: per-limb refinement
After pass 1, run a refinement loop for each contact-relevant limb:

- left_arm: shoulder -> elbow -> wrist -> palm site
- right_arm: shoulder -> elbow -> wrist -> palm site
- left_leg: hip -> knee -> ankle -> foot site
- right_leg: hip -> knee -> ankle -> foot site

For each limb:
- solve only the joints in that chain,
- use the current whole-body pose as the initialization,
- apply a target for the distal site or body,
- use a small regularization term to keep motion smooth,
- clamp the palm/foot site to a floor-safe height if needed.

The limb refinement should be lower priority than the whole-body solve, so it only fixes local contact issues instead of destroying the global posture.

## Contact discipline rules
For the refinement pass:
- if a hand is in contact, prioritize palm placement and palm normal over wrist motion,
- if a foot is in contact, prioritize foot-flat/foot-site placement over ankle drift,
- if a palm or foot would go below the floor, project it to a small safe clearance instead of letting the wrist/ankle snap through.

## Practical implementation plan
1. Create a branch such as `contact-aware-two-pass-ik`.
2. Keep the current solver as the baseline.
3. Add a second refinement pass in the per-frame IK loop.
4. Start with only the arms first, because the flick issue is clearly wrist-driven.
5. Compare:
   - floor penetration,
   - wrist velocity spike,
   - palm/contact target error,
   - tracking error.
6. If the arm pass works, extend the same pattern to the feet.

## Suggested first test
Use the current problematic clip and compare three variants:
- baseline current solver,
- whole-body only with stronger floor term,
- whole-body + arm refinement pass.

The first target metric should be: floor penetration stays near zero while wrist max velocity drops significantly.

## Notes for later
If the refinement pass still causes drift, we can make it even more conservative:
- only run it when contact is active or floor penetration is detected,
- use smaller step sizes,
- use a strong previous-frame regularizer so the refinement remains smooth.

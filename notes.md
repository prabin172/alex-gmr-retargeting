I setup a training in isaaclab for mimic and stopped it because of a bug. But, we were able to visualize it.

also, there is some issue in my setup that it doesn't allow me to visualize the training results in isaaclab isaacsim. but my mentor visualized it (the previous run), and says: "ok, not too bad. I think that the ground penetrations are the biggest problem here. you can see the hands at the beginning are not even able to position correctly for the push because they are forced into the ground. Also the left foot can't make it to the front of the robot because it's asked to get through the ground" And, I replied, "Okay then. Before I start another training, I'll try to tune that in retargeting."

Show less

Good call, and it lines up with the asymmetry we already established — this is actually a worse case than the "float" problem, not a milder one.

A few cm of registration float is something mimic can at least try to imitate (tracking error, maybe a balance risk on load-bearing frames). What your mentor describes — hands and a foot being pushed into/through the ground during a reach and a step — is different: the reference is asking for a target that's **geometrically impossible** for the real robot to occupy. The rigid floor won't let the limb go there at all, so those frames won't produce "slightly wrong tracking," they'll produce a hard block: the limb stops at the floor surface while the reference wants it somewhere below/through, constant large error for the whole reach/swing, and the policy has no way to close that gap no matter how well it's trained. Training mimic against that reference would just be reinforcing a target it structurally can't reach.

This also isn't the same axis as the percentile/registration-bias discussion from before. That was about the static z-offset calibration (where to put "zero" for planted contact). This sounds like a dynamic conflict during the transition itself — the floor constraint and the reach/step objective competing for the same frames, with the floor constraint losing (or the collision slack letting it slip through). Worth checking specifically in those windows:

- whether `FLOOR_WEIGHT` is actually active/binding during the reach/swing frames, or if it's getting relaxed by the collision slack right when it matters most
- whether the reach/step objective's cost is high enough at those specific frames to be dragging the solve through the floor rather than around/over it — i.e. is this a genuine weight-tradeoff loss, or a missing constraint (foot/hand collision geometry not checked against the floor during transit, only at labeled "contact" frames)

Fixing that before training is the right sequencing — no point burning training compute on a reference that's asking for something the physics will refuse outright.
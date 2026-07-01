# Retargeting pipeline — process flow

Human motion → canonical human → scaling → per-frame IK → smoothing → grounding → render.

Conventions: +X forward / +Y left / +Z up · quaternions wxyz.

```mermaid
flowchart TD
    A["Human motion capture<br/><i>marker/skeleton animation of a real person</i>"]

    A --> B["Canonical human<br/><i>map the capture onto a standard set of body<br/>landmarks — geometry-neutral, source-independent</i>"]

    B --> C["Orientation frames<br/><i>build a semantic frame at each body part from the<br/>landmark geometry; detect the facing direction</i>"]

    C --> D["Morphology scaling<br/><i>rescale the motion (deltas from the rest pose) to the<br/>robot's proportions — absolute root position untouched</i>"]

    D --> E["Per-frame contact-first IK<br/><i>solve the robot pose each frame to match the human;<br/>detect ground contact from the human motion and, at<br/>contacting limbs, enforce the support surface<br/>(feet flat, hands/fists pressing down) over the raw pose</i>"]

    E --> F["Trajectory smoothing<br/><i>smooth the whole solved trajectory over time —<br/>joints and the floating base — to remove jitter and pops</i>"]

    F --> G["Z-grounding<br/><i>drop/lift the whole body so its lowest contact point<br/>rests on the floor (z = 0) every frame</i>"]

    G --> H["Render / output<br/><i>grounded robot trajectory → video and motion file</i>"]
```

## The idea, step by step
1. **Human motion capture** — the raw performance of a real person.
2. **Canonical human** — re-express that motion on a standard skeleton of body
   landmarks, so everything downstream is independent of the capture source.
3. **Orientation frames** — derive an orientation for each body part from the
   landmark *positions* (not the raw capture rotations), and find which way the
   person faces.
4. **Morphology scaling** — adapt the human-sized motion to the robot's limb
   proportions, scaling only the movement away from a rest pose so the robot
   isn't teleported.
5. **Per-frame contact-first IK** — for each frame, solve the robot's joints to
   follow the human. Where the human is touching the ground, that contact takes
   priority: the foot is held flat, a supporting hand/fist is pressed down, and
   the limb is free to bend however the contact needs.
6. **Trajectory smoothing** — treat the whole clip at once and smooth it in time
   (both the joints and the free base) to kill per-frame jitter and pops.
7. **Z-grounding** — shift the body vertically so its lowest point sits on the
   floor, so the robot is planted rather than floating or sinking.
8. **Render / output** — produce the final video and the grounded motion.

# Conclusions

General Motion Retargeting explicitly excludes floor-contact motion from its
own evaluation. We show this exclusion reflects a real, severe, and
correctable gap: unmodified GMR output on floor-contact clips reaches
12.9–15.9 cm of floor penetration on 39–91% of frames, against 1 cm and 0.3%
on GMR's own clean-locomotion clips; GMR's own published mitigation (a
single per-clip constant offset) reduces penetration but cannot satisfy a
multi-phase clip's standing and lying stances with one number, leaving the
robot floating well above the floor at exactly the frames that should show
contact; and a contact-aware correction layer on top of GMR's unmodified
solver closes most of this gap at full 77-clip corpus scale — floor
penetration eliminated by construction, self-collision cut by roughly three
orders of magnitude on the floor class, and our joint contact-correctness
metric raised from 0.2% to 98.8% of frames. That metric was itself a
required contribution: an earlier, single-axis version of the same check is
satisfiable by a rigid per-clip shift with no notion of contact timing at
all, which would silently overstate the result.

We report two instantiations of the correction layer and are transparent
about the trade-off between them. The multi-stage, per-frame pipeline
achieves the strongest contact precision (Results §3) but carries a real
peak-joint-velocity cost traced to branch-flip artifacts inherent to
independent per-frame correction on a redundant kinematic chain. The global,
whole-trajectory pipeline removes that artifact by construction and reduces
jerk and peak velocity by a wide margin relative to both baselines, at a
currently lower level of contact precision (Results §5, evaluated so far on
36 of 77 corpus clips). We present this as an open, general trade-off
between temporal consistency and per-frame solving power rather than
resolving it in favor of whichever number looks cleaner.

## Status and next steps

Kinematic plausibility is a necessary but not sufficient condition for a
retargeted motion to be useful — the retargeting literature this paper
builds on has established that reference-motion quality measurably affects
downstream imitation-policy success. Two pieces of work close that gap for
this paper and are in progress at time of writing, not deferred to separate
future publications:

1. **Policy-training validation.** We are training imitation-learning
   tracking policies (BeyondMimic) on the retargeted floor-contact corpus
   and will report success rate and tracking error against policies trained
   on `gmr_heightfix` references and, separately, per pipeline (multi-stage
   vs. global), following the evaluation protocol GMR itself uses for
   locomotion. This is the experiment that makes the smoothness/precision
   trade-off in Results §5 decidable rather than qualitative, and results
   will be added to this paper once training completes.
2. **Completing the global pipeline's corpus evaluation.** The remaining 41
   of 77 clips, and continued work narrowing the contact-precision gap to
   the multi-stage pipeline — the most promising direction identified so
   far is a staged/homotopy solve that gives the whole-trajectory optimizer
   an easier problem at each step, rather than asking it to close a severe
   violation in one bounded step.

## Further future work

- **Generalization.** A second robot morphology and a second motion-capture
  source, to separate what in these results is fundamental to the
  floor-contact retargeting problem from what is specific to the Unitree G1
  or to LAFAN1.
- **The larger contact-template system.** This paper is the validated first
  step of the direction sketched in `paperIdea3.md`: generalizing from
  foot/hand contact to knees, pelvis, and forearms as first-class contacts,
  mapping detected human floor postures onto a library of robot-feasible
  support-stance templates, and validating with real-hardware playback —
  evidence, to our knowledge, no published floor-contact retargeter
  currently provides. Having established that the underlying gap is real
  and kinematically closable, that system is the natural next target once
  the policy-training results above are in hand.

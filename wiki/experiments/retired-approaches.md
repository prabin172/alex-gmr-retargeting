# Retired Approaches (why they died — don't resurrect without new evidence)

- **Hierarchical two-level solve** (`--hierarchical`, level-1 foot tasks / level-2 body+hand): REGRESSED pivoting get-ups. Root failure: promoting the reach-limited palm pin to hard starved body tracking. Flag retired, OFF. Shipped path = single-level all-soft weights.
- **Hard-equality foot pins in Stage B**: never actually shipped (docstring lies — see [[globalopt]]); the hard COLLISION path did ship and died on fullmesh infeasibility (see [[fullmesh-vs-primitive]]).
- **Upright-root constraint**: makes limbs absorb the whole lying-to-standing rotation on get-ups, picks bad IK branches. Root orientation left free.
- **Per-frame velocity cap in-solver**: had a collision interaction problem, unresolved — superseded by Stage A global smoothing.
- **Stage B all-off rule** (old): artifact of loose contact detection; superseded once hysteresis + foot-hold ×10 made plants stationary.
- **Legacy worlddelta solver family** (`scripts/legacy/`: `solve_fbx_canonical_alex_posori_qp_fresh*.py`, bodypos variants, MVNX path, old renders): the pre-contact-first pipeline, baseline tag `baseline-posori-worlddelta-v1`, branch `initialBaseline`. Kept for reference, not run. Its contact-labels exporter `post_process_grounding_contacts.py` is the template for the future Mimic export (see [[open-questions]]).
- **Branch `feature/fbx-kinematic-canonical-v2`**: parallel solver with different segment assumptions — kept for exploration, not retired-dead but not active.

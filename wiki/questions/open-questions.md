# Open Questions & Watch Items

## Unverified behavior (check when inspecting renders/metrics)
- **Kneeling falls (`kneelingFall_02/03`)**: first clips with contact onset on a DESCENDING body; hysteresis (150 ms cap) was tuned on plants from crouches. Verify knee/hand touchdowns.
- Kneeling clips likely trigger the shank-clamp deep-kneel skip (`v·ẑ < 0.2L`) — fine by design, but verify flat behavior while kneeling.
- `standup_side_05`: 7.9 cm slip on primitive era — recheck on fullmesh.
- Inspect all 18 `outputs/renders/contactfirst/fullURDF/` videos, esp. get-up/kneel plant behavior and the higher ~4.2 cm slip (SESSION_HANDOFF NEXT item 1).

## Not yet built
- **Mimic-ready contact-labels export on the contact-first path** — legacy `post_process_grounding_contacts.py` produced `(T,11)` labels for the retired pipeline; needs porting.
- Hardware playback of the IHMC JSONs (export exists, no on-robot evidence).
- Baseline retargeter on Alex (in-solver "faithfulness-first" ablation is the cheap option — flags exist).
- BeyondMimic-on-Alex policy training (the 2027 long game; torque requirements expected back from RL training on the reference motions).

## Method questions (from paper notes)
- Does Stage B's slip increase on the hardest clips (0.2→9.3 cm case) indicate median-anchor failure on repositioning contacts? Worth a look before the paper metrics table.
- What do we really want from retargeting: faithfulness vs physics-respecting copying? Is a "good retargeter" = motion copy + physics filtering built in, and does that improve downstream policy performance enough to justify the effort? (paper_idea.md "New Idea" section — unresolved.)

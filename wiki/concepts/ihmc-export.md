# IHMC JSON Export

`scripts/export_alex_retarget_npz_to_ihmc_json.py` — converts grounded `qpos (T,36)` NPZ to IHMC **`KinematicsToolboxOutputStatus`** DDS-message JSON (outer key `toolbox_msgs.msg.dds.KinematicsToolboxOutputStatus`, unchanged; inner key `us::ihmc::robotDataLogger::KinematicsToolboxOutputStatus` as of 2026-07-22, see below), i.e. the format the real IHMC Alex whole-body/kinematics toolbox consumes. Joint order = ISAAC_JOINT_NAMES_FULLBODY (hips→knees→ankles→spine→arms→neck).

### Key rename (2026-07-22): snake_case → camelCase, and a DDS message-type change

The mentor's Java side regenerated the `KinematicsToolboxOutputStatus` DDS bindings, pushed to
IsaacLab `origin/feature/zest` (commit `3dcfec2b`). Verified via `git diff feature/zest
origin/feature/zest -- scripts/ihmc/rsl_rl/json_to_npz.py` (not guessed — an earlier attempt to
infer this from a copied `json_to_npz.py` file failed because that copy was one commit behind and
still read the old keys). Two things changed, not just casing:

- **Inner DDS key**: `toolbox_msgs::msg::dds_::KinematicsToolboxOutputStatus_` →
  `us::ihmc::robotDataLogger::KinematicsToolboxOutputStatus` — a different message namespace
  entirely, not a rename of the same type.
- **9 fields, verified** (present in the actual diff, i.e. `json_to_npz.py` reads them):
  `desired_root_position`→`desiredRootPosition`, `desired_root_orientation`→`desiredRootOrientation`,
  `desired_joint_angles`→`desiredJointAngles`, `desired_joint_velocities`→`desiredJointVelocities`,
  `desired_root_linear_velocity`→`desiredRootLinearVelocity`,
  `desired_root_angular_velocity`→`desiredRootAngularVelocity`, `com_offset`→`comOffset`,
  `left_foot_in_contact`→`leftFootInContact`, `right_foot_in_contact`→`rightFootInContact`.
- **11 more fields, camelCased but NOT verified** — `json_to_npz.py` doesn't read them, so there's
  no diff to confirm against: `sequence_id`, `current_toolbox_state`, `joint_name_hash`,
  `support_region`, `desired_joint_velocities_publishing_period`,
  `desired_root_linear_velocity_publishing_period`,
  `desired_root_angular_velocity_publishing_period`, `desired_torso_position`,
  `desired_torso_orientation`, `left_hand_in_contact`/`right_hand_in_contact`,
  `solution_quality`. Applied the same mechanical transform on the assumption the whole message
  schema shifted uniformly (consistent with a regenerated DDS binding); flag to the mentor if
  anything else downstream reads these under a different convention.

Our exporter (`export_alex_retarget_npz_to_ihmc_json.py`) updated to match, smoke-tested by
round-tripping a real export and diffing every key against the verified 9 — exact match, zero
leftover snake_case keys. Any JSON exported before this change (e.g. the original, non-heightfix
`outputs/ihmcJsons50hz/shovel_fronthard_02.json`) is now stale against the updated
`json_to_npz.py` and will not parse there.

- Outputs on disk (current, 2026-07-06): `outputs/ihmcJsons-native120hz/` (18 clips, native-120 solve) and `outputs/ihmcJsons50hz/` (18 clips, same source NPZs exported `--fps 50` to match the IHMC reference `1.json` rate). Superseded/stale: `outputs/ihmcJsons/` (30 Hz era) and `outputs/ihmcJsons-120hz/` (upsampled-from-30, pre native-120 solve).
- **Pipeline emits both** (`retargetingPipeline.sh` Stage 6 + 6b): native-120 to `IHMC_DIR` (no `--fps`) and 50 Hz to `IHMC_DIR_50` (`--fps 50`). Knobs: `EXPORT_50HZ=1` (default; `0` skips), `IHMC_DIR_50=outputs/ihmcJsons50hz`. The 50 Hz set is a genuine 120→50 downsample of the native-120 grounded NPZ — NOT upsampled-from-30 (`1.npz` is itself 50 Hz, so json→npz keeps the rate).
- **120 Hz resample** (commit `665b2da` + uncommitted extension, 2026-07): linear interp for root pos + joints, **slerp (wxyz, hemisphere-corrected)** for root quat, zero-order hold for contact flags (`resample_qpos`, `quat_slerp_wxyz`, `resample_contacts_bool`). Pipeline solves at 30 fps (stride 4 of 120 fps capture); export upsamples back.
- Root angular velocity computed from quat finite differences (`quat_to_angvel_wxyz`).
- **fps-correctness fix (2026-07-03)**: the grounded NPZ stores `fps` = the *capture* rate (120), but the solved qpos is strided → real-time is `fps/stride` = 30 Hz. The exporter now derives `src_fps` from `source_frame_ids` stride (not raw `fps`). Before the fix, `--fps 120` was a no-op relabel → JSON played **4× too fast** with 4× inflated velocities (e.g. kneelingFall_02: 123 frames @ 8.33 ms = 1.02 s instead of the true 4.07 s). Correct 120 Hz export now resamples 30→120 (123→489 frames, 4.07 s), restoring the original capture timing. Regenerate with `--fps 120`. The old `outputs/ihmcJsons/` + prior `-120hz/` were WRONG (4× fast) — superseded.

## Downstream consumer + the 50 Hz requirement (rsl_rl, 2026-07-05)

The JSON is ingested by IHMC's RL motion-tracking harness (repo-root `rsl_rl/`, an RSL-RL / BeyondMimic-style tracker for Alex: `train_debug.py`, `eval_debug.py`, `cluster_train.py`, `json_to_npz.py`, `replay_npz.py`, `export_cfg.py`; reference motions in `motions/` + `motions_orig/`, all `fps=50`).

**Consumption rate is 50 Hz, hard:**
- `json_to_npz.py`: `--output_fps` **default 50**; `sim_cfg.dt = 1/output_fps = 0.02`. Reads the JSON `timestamps` (ms), builds a 50 Hz grid `t_out = arange(t_start, t_end, 1/output_fps)`, and **ZOH-resamples** frames onto it (nearest previous sample, *no interpolation*).
- `replay_npz.py`: `sim_cfg.dt = 0.02` hardcoded = 50 Hz.
- All their reference npz store `fps=50`. Our `standup_01_grounded.npz` already round-tripped through their `json_to_npz` → came back `fps=50, 120 frames`.

**Implication — our 30 Hz solve is sub-Nyquist for this consumer.** ZOH adds no information; it only picks samples. So the JSON must carry ≥50 Hz of *genuine* content. We solve at 30 Hz (`STRIDE=4`), so our JSON's real content is 30 Hz regardless of whether we upsample it to 120 first. Their ZOH then stair-steps 30 Hz content onto a 50 Hz grid — fast transients (touchdowns, hand plants, impacts) aliased by the stride-4 decimation are gone before the policy sees them. **Fix = solve natively at ≥50 Hz** (stride 1 → 120 Hz clean, or stride 2 → 60 Hz), export at native rate, let their `json_to_npz --output_fps 50` do the *only* downsample. Self-upsampling 30→120 is pointless — it launders 30 Hz content as 120. See [[pipeline]] solve-rate note. Caveat: `LAMBDA_SMOOTH` / Stage A-B velocity terms are dt-dependent, tuned at 30 Hz → need retuning at the new rate.

No evidence yet of on-robot playback — export exists, hardware run does not (see [[publication]] §missing evidence).

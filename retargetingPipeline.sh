#!/usr/bin/env bash
# retargetingPipeline.sh
# Full contact-first pipeline for every fbx clip:
#   Stage 2.5  canonical-human grounding + persisted contact labels (phasic-v2
#              M1: floor = z=0 by construction upstream, one detector shared by
#              every downstream stage instead of each re-estimating its own;
#              always recomputed, never skip-if-exists -- see planLog.md M1 for
#              why: Stage 3's skip-if-exists caused a stale-cache confound
#              during this milestone's own gate testing)
#   Stage 3    contact-first IK  (skipped if the NPZ already exists)
#   Stage 4    GlobalOPT Stage-A smoothing (spikes->0, collisions down)
#   Stage 4.5  Z-grounding (plant lowest contact point on the floor z=0)
#   Stage 4.6  physics plausibility pass (phasic-v2 M4, opt-in via PHYSICS_PASS=on;
#              default off -- byte-identical to before this stage existed)
#   Stage 4.7  per-limb cleanup solver (phasic-v2 M5, opt-in via LIMB_REFINE=on;
#              default off -- see wiki/experiments/phasic-v2-M5-gate.md for scope:
#              fixes isolated swing-limb floor/self-collision violations, safely
#              no-ops on whole-body-lying-phase clips it structurally cannot fix)
#   Stage 5    render, REAL TIME (fps = source_fps / stride)
#
# Outputs:
#   outputs/contactfirst/<clip>_contactfirst.npz
#   outputs/global_opt_contactfirst/<clip>_global_opt.npz
#   outputs/grounded_contactfirst/<clip>_grounded.npz
#   outputs/physics_plausibility/<clip>_physics.npz          (PHYSICS_PASS=on only)
#   outputs/limb_refine/<clip>_limbrefine.npz                (LIMB_REFINE=on only)
#   outputs/renders/contactfirst/<clip>_globalopt.mp4
#
# The Stage-4 solver defaults to the single canonical fullmesh collision model
# (assets/alex/alex_floating_base_with_sites.xml) with always-on soft self-
# collision — no model/flag knobs needed for the shipped setup.
#
# Env knobs (all optional):
#   CF_DIR         (default outputs/contactfirst)             Stage-3 output dir.
#   GO_DIR         (default outputs/global_opt_contactfirst)  Stage-4 output dir.
#   GR_DIR         (default outputs/grounded_contactfirst)    Stage-4.5 output dir.
#   RENDER_MESH / RENDER_DIR / RENDER_EXTRA  render-stage controls (see below).
set -uo pipefail

STRIDE=1
IK_ITERS=40
# --- Native 120 Hz solve (2026-07-05) ---
# Downstream IHMC RL tracker consumes at 50 Hz with ZOH (no interp); solving at
# 30 Hz (old STRIDE=4) was sub-Nyquist for that gate. STRIDE=1 solves at the
# native 120 fps capture rate; nothing self-upsamples anymore (export stays
# native, their json_to_npz --output_fps 50 does the only downsample).
# Rate-dependent knobs rescaled for dt/4 (see wiki/concepts/pipeline.md):
#   LAMBDA_SMOOTH / GROUND_SMOOTH first-difference penalties  -> x16 (∝ fps²)
#   contact min-run / ramp / preroll  (measured in FRAMES)     -> x4
# All time/physical/dimensionless knobs (speeds, onset-delay-s, contact weights,
# trust, posture_reg) are rate-invariant and unchanged.
LAMBDA_SMOOTH="${LAMBDA_SMOOTH:-320}"    # GlobalOPT Stage-A smoothing weight (20 @ 30 Hz × 16)
N_OUTER="${N_OUTER:-6}"                  # Stage-B contact-QP outer iterations (0=off).
# Bumped 3->6 for the native 120 Hz solve: the 4x larger QP needs more SCA
# re-linearisation passes for self-collision to converge (get-up clips regressed
# to ~33% coll at n=3, back to ~14% ≈ 30 Hz baseline at n=6). See wiki/log.md 2026-07-05.
# Contact pin weights x4 (defaults 40/8) for the 120 Hz solve: LAMBDA_SMOOTH went
# x16 but the position-space pins did not, so plants slid. x4 cut standup_side_04
# plant-slip 10.4->6.3cm for <1cm added collision. plant-speed had no effect (slow
# steady drift stays sub-threshold); weight is the lever. See wiki/log.md.
FOOT_WEIGHT="${FOOT_WEIGHT:-160}"        # soft pin on a PLANTED foot (40 default x4)
HAND_WEIGHT="${HAND_WEIGHT:-32}"         # soft pin on a PLANTED palm  (8 default x4)
# Min length (frames) of a stillness sub-segment before it counts as a plant;
# shorter speed dips are reclassified moving. Debounces phantom 1-frame plants on
# lifting-off hands (standup_side_05 right_hand 14.7->6.8cm). Frame-count knob -> x4
# for 120 Hz (2 @ 30 Hz).
PLANT_MIN_RUN="${PLANT_MIN_RUN:-8}"
CONTACT_PREROLL="${CONTACT_PREROLL:-8}"    # frames of look-ahead before touchdown (0 = no anticipation)
# Stage-3 coplanar-feet targets: when both feet are contact-labelled, snap their
# ankle-height targets to a common Z so the IK produces coplanar feet. Fixes the
# root cause of the "one foot floats in RDX" bug — the retargeted foot-height
# targets can sit several cm apart (source ankles / per-leg scale differ) while
# both are labelled planted, which a rigid grounding shift can't reconcile.
# mean = distribute (lowest self-collision) | min = snap to lower foot | off = legacy.
COPLANAR_FEET_MODE="${COPLANAR_FEET_MODE:-mean}"
# Stage-3 fullmesh robot-vs-floor repulsion. This is the proper upstream fix
# path for swing feet / hands below the floor: Stage 3 can move the root, unlike
# Stage 4's joint-only collision cleanup. MILD DEFAULT ON (phasic-v2 M2/T2.2,
# 2026-07-10): plan.md's own Risks section anticipated needing this fallback
# ("re-enable mild Stage-3 repulsion (weight ~5) default-on WITH ramp") after
# T2.1's target-only correction alone proved insufficient for
# luigi_standProne_03 (tested: zero Stage-3 floor flags -> Stage-4 output
# regresses to 14.29cm pen + 3 spikes, vs. the historical 2.4cm/0 spikes -- see
# planLog.md M2). The "ramp" is `--floor-refine` (already default ON, see
# below) -- proven-safe, unchanged mechanism, just no longer gated behind a
# per-clip flag. Weight tuned 5->10 (5 left 3-4 residual spikes on
# luigi_standProne_03; 10 reaches 0). NOT fully "ONE config yet": both Luigi
# clips still carry ONE small non-floor per-clip flag (contact-preroll /
# floor-phase-aware respectively) -- see their CLIPS[] entries and
# planLog.md M2 T2.2-continued for what remains open.
S3_FLOOR_WEIGHT="${S3_FLOOR_WEIGHT:-10}"
S3_FLOOR_MARGIN="${S3_FLOOR_MARGIN:-0}"
S3_FLOOR_GAIN="${S3_FLOOR_GAIN:-5}"
# Hierarchical-v1 H2 (2026-07-11): promotes foot-hold/foot-flat/foot-yaw to
# Stage 3's level-1 (hard) task-priority tier -- a "robots cannot slip" hard
# constraint instead of a heavily-weighted soft pin. Deliberately does NOT
# touch hand contacts (a prior --hierarchical experiment that ALSO hardened
# the hand palm pin regressed pivoting get-ups, wiki/experiments/
# retired-approaches.md) NOR floor non-penetration (CONFIRMED BROKEN if
# combined with this tier -- a 44-metre divergence on standup_natural_01,
# see solve_fbx_canonical_alex_contactfirst.py's --floor-hard help text and
# planLog.md H2 -- floor stays soft/rows2 regardless of this flag). Default
# OFF, verified byte-identical no-op via two consecutive off-runs, and
# verified non-blowup on standup_natural_01 (the exact clip the original
# --hierarchical regression named) when on.
S3_HARD_TIER="${S3_HARD_TIER:-off}"          # on|off
S3_HARD_TIER_FLAG=""
[ "$S3_HARD_TIER" = "on" ] && S3_HARD_TIER_FLAG="--hard-tier"
# Swing-foot toe-clearance: cap toe-down pitch of an airborne foot's orientation
# target so the rigid foot plate's toe doesn't dig through the floor on get-up
# steps (measured 10-18cm swing-foot penetration on the get-up clip class). Opt-in
# until corpus-validated. See solve_fbx_canonical_alex_contactfirst.py --swing-clear.
SWING_CLEAR="${SWING_CLEAR:-off}"            # on|off
SWING_CLEAR_FLAG=""
SWING_MAX_PITCH="${SWING_MAX_PITCH:-5}"      # max toe-down pitch (deg) for a swing foot (aggressive; safe w/ continuity reg)
SWING_CLEAR_HEIGHT="${SWING_CLEAR_HEIGHT:-1.0}"  # proximity gate zero height (m); 1.0 = gate off (experimental knob)
SWING_CLEAR_BAND="${SWING_CLEAR_BAND:-0.04}"     # gate ramp band below the zero height (m)
SWING_CLEAR_WEIGHT="${SWING_CLEAR_WEIGHT:-0}"    # optional soft clearance row weight (0=off, cap only)
SWING_CLEAR_MARGIN="${SWING_CLEAR_MARGIN:-0.005}" # soft-clearance height above floor (m)
SWING_CONTINUITY_REG="${SWING_CONTINUITY_REG:-0.9}" # hip/knee continuity reg on de-pitch frames (prevents IK branch flip)
SWING_COLL_BOOST="${SWING_COLL_BOOST:-3.0}"     # self-collision weight boost on de-pitch frames (guards vs thigh-torso jam)
[ "$SWING_CLEAR" = "on" ] && SWING_CLEAR_FLAG="--swing-clear --swing-max-pitch $SWING_MAX_PITCH --swing-clear-height $SWING_CLEAR_HEIGHT --swing-clear-band $SWING_CLEAR_BAND --swing-clear-weight $SWING_CLEAR_WEIGHT --swing-clear-margin $SWING_CLEAR_MARGIN --swing-continuity-reg $SWING_CONTINUITY_REG --swing-coll-boost $SWING_COLL_BOOST"
# Leg-floor-refine: local re-solve synthesizing a temporary PLANT (ankle blended
# to floor+clearance, foot-flat align) for a tucked/deep-crouch leg that would
# otherwise dig through the floor (knee-140 embodiment gap), WITH pelvis/torso
# tracking relaxed so root can rise/shift to make room. Supersedes --swing-clear
# (rejected: contorted tucked legs). Opt-in, independent of --floor-weight/S3_HARD_TIER.
LEG_FLOOR_REFINE="${LEG_FLOOR_REFINE:-off}"   # on|off
LEG_FLOOR_REFINE_FLAG=""
[ "$LEG_FLOOR_REFINE" = "on" ] && LEG_FLOOR_REFINE_FLAG="--leg-floor-refine"
# On-floor + coplanar rows: drive each PLANTED foot's 4 sole-corner Zs to a shared
# floor height (the lower foot's warm-start ground). Co-plants both feet in the
# SOLVE, which a rigid 1-DOF grounding shift cannot. 0 = off. Pairs with
# GROUND_MODE=constant (constant does the absolute z=0 registration; the feet are
# already coplanar so one shift plants both).
FLOOR_WEIGHT="${FLOOR_WEIGHT:-200}"
FLOOR_MODE="${FLOOR_MODE:-estimate}"      # estimate (lower foot's ground) | zero (soles->z=0)
# Hard mesh-accurate robot-vs-floor collision: injects a floor plane geom
# in-memory (never touches the hand-maintained asset XML) and reuses the
# self-collision soft-slack QP machinery against it. Unlike FLOOR_WEIGHT (soft
# pin, planted feet only), this stops ANY fullmesh geometry — swing feet,
# hands, a tilted toe mid-get-up — from passing through the floor. on|off.
FLOOR_COLLISION="${FLOOR_COLLISION:-off}"   # validated on 1 clip only so far — opt-in pending corpus validation, see collision.md
# Foot-scoped Stage-A floor-sensitivity threshold (default = SENS_MIN_PEN, i.e.
# no-op/inert): lowers protection threshold ONLY for leg/foot floor contacts
# (and only boosts leg/ankle joint columns) so Stage A's tridiagonal smoothing
# doesn't drag an unprotected, borderline swing-foot dig into a much deeper one
# -- root cause of the get-up-class swing-foot floor penetration (see
# wiki/results/tradeoffs-limits.md and planLog.md). Opt-in pending corpus test.
SENS_MIN_PEN="${SENS_MIN_PEN:-0.015}"
SENS_FOOT_MIN_PEN="${SENS_FOOT_MIN_PEN:-0.015}"
RENDER_EXTRA="${RENDER_EXTRA:-}"          # extra render flags, e.g. "--fixed-cam"
RENDER="${RENDER:-1}"                      # 1 = render Stage 5 mp4 | 0 = skip (faster; JSON export still runs)
# --- Z-grounding (Stage 4.5) ---
GROUND_MODE="${GROUND_MODE:-constant-contact}"   # constant-contact (single shift keyed to planted feet — no bobbing, feet stay down) | perframe (plant every frame, wanders) | constant (single shift, global lowest)
GROUND_SMOOTH="${GROUND_SMOOTH:-80}"      # perframe: tridiagonal smoothing on the shift series (5 @ 30 Hz × 16)
GROUND_PERCENTILE="${GROUND_PERCENTILE:-1}"              # constant mode: 0 = lift all frames above floor; 1 = robust near-min
GROUND_CONTACT_PERCENTILE="${GROUND_CONTACT_PERCENTILE:-50}"  # constant-contact: planted-foot sole percentile
GROUND_STILL_SPEED="${GROUND_STILL_SPEED:-0.05}"         # constant-contact: still-plant speed threshold (m/s)
# --- Render mesh (Stage 5) ---
# visual    = full-body Alex visual mesh (legs+arms+torso), hands as closed fists
# collision = fullmesh collision convex hulls (what the solver actually uses)
# <path>    = any explicit model xml
RENDER_MESH="${RENDER_MESH:-visual}"
case "$RENDER_MESH" in
  visual)    RMODEL="assets/alex/alex_visual_mesh_fist_hands.xml" ;;
  collision) RMODEL="assets/alex/alex_floating_base_with_sites.xml" ;;
  *)         RMODEL="$RENDER_MESH" ;;
esac
IN=outputs/canonical_human/fbx_fresh
CF="${CF_DIR:-outputs/contactfirst}"
mkdir -p "$IN"
GO="${GO_DIR:-outputs/global_opt_contactfirst}"
GR="${GR_DIR:-outputs/grounded_contactfirst}"
# Stage 4.6 — physics plausibility pass (phasic-v2 M4, scripts/physics_plausibility_pass.py).
# Flag-gated, default OFF: clips joint + root velocity/acceleration into
# conservative bounds via a least-perturbation QP (Increment 1 only — CoM
# support-polygon check, T4.2, unbuilt). Separate phase from GlobalOPT by
# design (independently ablatable); consumes the grounded NPZ, output feeds
# render/export in place of it when enabled. On already-clean Stage 4/4.5
# output this is a verified near-no-op (0 spikes, RMS<=0.1cm on all 20 clips,
# see planLog.md M4) — it only engages when a real violation exists.
PHYSICS_PASS="${PHYSICS_PASS:-off}"        # on|off
PH="${PHYSICS_DIR:-outputs/physics_plausibility}"
# Stage 4.7 — per-limb cleanup solver (phasic-v2 M5, scripts/refine_limbs_contactfirst.py).
# Flag-gated, default OFF: root-frozen, per-limb whole-clip QP fixing floor
# penetration / self-collision / swing clearance for a SWINGING limb (an
# isolated foot/hand dipping through the floor). NOT a replacement for the
# Luigi clips' per-clip floor flags -- verified on the full 20-clip corpus
# that it structurally cannot help a clip with an extended WHOLE-BODY lying
# phase (root frozen, no DOF to lift the whole body clear of a floor it's
# legitimately lying along) -- keep-best-iterate safely protects those clips
# (never worse than input) but doesn't improve them either. 8/20 clean pass,
# 5/20 near-miss, 7/20 safely no-op — see wiki/experiments/phasic-v2-M5-gate.md.
# Runs AFTER physics-plausibility if both are enabled (consumes $final from
# whichever stage ran last).
LIMB_REFINE="${LIMB_REFINE:-off}"          # on|off
LR="${LIMB_REFINE_DIR:-outputs/limb_refine}"
RD="${RENDER_DIR:-outputs/renders/contactfirst}"
# Stage 6 — IHMC JSON export. Fresh dir: the old outputs/ihmcJsons{,-120hz} were
# upsampled-from-30 and are superseded by this native-120 solve. At STRIDE=1 the
# grounded NPZ is native 120 Hz, so we export with NO --fps (identity); their
# json_to_npz --output_fps 50 does the only downsample. We ALSO emit a 50 Hz set
# (--fps 50) matching the IHMC reference 1.json rate; set EXPORT_50HZ=0 to skip it.
# See wiki/concepts/ihmc-export.md.
IH="${IHMC_DIR:-outputs/ihmcJsons-native120hz}"
EXPORT_50HZ="${EXPORT_50HZ:-1}"           # also emit a 50 Hz set (IHMC 1.json rate)
IH50="${IHMC_DIR_50:-outputs/ihmcJsons50hz}"
mkdir -p "$CF" "$GO" "$GR" "$RD" "$IH"
[ "$EXPORT_50HZ" = "1" ] && mkdir -p "$IH50"
[ "$PHYSICS_PASS" = "on" ] && mkdir -p "$PH"
[ "$LIMB_REFINE" = "on" ] && mkdir -p "$LR"
echo "Stage-3 floor: weight=$S3_FLOOR_WEIGHT margin=$S3_FLOOR_MARGIN gain=$S3_FLOOR_GAIN   |  Stage-3 hard-tier: $S3_HARD_TIER   |  Stage-4 floor collision: $FLOOR_COLLISION"
echo "Physics plausibility pass (Stage 4.6): $PHYSICS_PASS   |  Limb cleanup (Stage 4.7): $LIMB_REFINE"
echo "Render mesh: $RENDER_MESH -> $RMODEL   |  ground: $GROUND_MODE (smooth=$GROUND_SMOOTH pct=$GROUND_PERCENTILE contact_pct=$GROUND_CONTACT_PERCENTILE)"

# clip name  ->  source FBX path (stages 1-2 produce the canonical NPZ + with_orient NPZ).
# Optional 3rd field: extra contact-first SOLVER flags. 4th: extra GLOBALOPT flags.
# 5th (phasic-v2 M1): extra Stage-2.5 GROUNDING/detection flags -- e.g. a clip's
# contact-detection onset-hysteresis tuning now lives here (Stage 3 consumes
# PERSISTED labels, it no longer re-detects, so a detection-flag override has to
# reach Stage 2.5, not Stage 3 -- see planLog.md M1/T1.3).
# ALL EMPTY by design — one retargeter config for all actions; the fields exist for experiments only.
CLIPS=(
  "standup_01|data/raw/inhouse/get_up_from_ground/fbx/standup_01.fbx|||"
  "standup_02|data/raw/inhouse/get_up_from_ground/fbx/standup_02.fbx|||"
  "standup_natural_01|data/raw/inhouse/get_up_from_ground/fbx/standup_natural_01.fbx|||"
  "standup_natural_02|data/raw/inhouse/get_up_from_ground/fbx/standup_natural_02.fbx|||"
  "standup_side_04|data/raw/inhouse/get_up_from_ground/fbx/standup_side_04.fbx|||"
  "standup_side_05|data/raw/inhouse/get_up_from_ground/fbx/standup_side_05.fbx|||"
  "standup_slideHandsBack_03|data/raw/inhouse/get_up_from_ground/fbx/standup_slideHandsBack_03.fbx|||"
  "shovel_fronthard_02|data/raw/inhouse/shoveling/PrabinRef_Shovel_FrontHard_02.fbx|||"
  "shovel_leftbucket_02|data/raw/inhouse/shoveling/PrabinRef_Shovel_LeftBucket_02.fbx|||"
  "shovel_lefthard_01|data/raw/inhouse/shoveling/PrabinRef_Shovel_LeftHard_01.fbx|||"
  "shovel_rightbucket_01|data/raw/inhouse/shoveling/PrabinRef_Shovel_RightBucket_01.fbx|||"
  "shovel_righthard_01|data/raw/inhouse/shoveling/PrabinRef_Shovel_RightHard_01.fbx|||"
  "standupFromKneeling_01|data/raw/inhouse/standFromKnees/PrabinRef_STandupFromKneeling_01.fbx|||"
  "standupFromKneeling_02|data/raw/inhouse/standFromKnees/PrabinRef_STandupFromKneeling_02.fbx|||"
  "standupKnees_02|data/raw/inhouse/standFromKnees/PrabinRef_StandupKnees_02.fbx|||"
  "standupSquatCrouch_01|data/raw/inhouse/crouchStand/PrabinRef_StandupSquatCrouch_01.fbx|||"
  "kneelingFall_02|data/raw/inhouse/KneelingFall/PrabinRef_KneelingFall_02.fbx|||"
  "kneelingFall_03|data/raw/inhouse/KneelingFall/PrabinRef_KneelingFall_03.fbx|||"
  "luigi_standProne_03|data/raw/inhouse/LuigiStand/LuigiRef_StandProne_03.fbx|--contact-preroll 0|--floor-collision on|--contact-on-speed-frac 0.25 --contact-onset-max-delay 0.35"
  "luigi_standSupine_08|data/raw/inhouse/LuigiStand/LuigiRef_StandSupine_08.fbx|--floor-phase-aware|--floor-collision on --floor-phase-aware on|"
)

# Optional substring filter on the clip NAME: run only clips whose name contains
# CLIPS_MATCH (empty = all). E.g. CLIPS_MATCH=luigi bash retargetingPipeline.sh.
CLIPS_MATCH="${CLIPS_MATCH:-}"

ok=0; fail=0
for entry in "${CLIPS[@]}"; do
  IFS='|' read -r name fbx solve_extra go_extra ground_extra <<< "$entry"
  if [[ -n "$CLIPS_MATCH" && "$name" != *"$CLIPS_MATCH"* ]]; then continue; fi
  can="$IN/${name}_canonical_human.npz"
  src="$IN/${name}_with_orient.npz"
  cg="$IN/${name}_canonical_grounded.npz"
  cf="$CF/${name}_contactfirst.npz"
  go="$GO/${name}_global_opt.npz"
  gr="$GR/${name}_grounded.npz"
  mp4="$RD/${name}_globalopt.mp4"
  echo "======================================================================"
  echo ">>> $name   ($fbx)"

  # Stage 1 — FBX → canonical human NPZ (Blender Python)
  # Stage 2 — canonical NPZ → with_orient NPZ (pure Python)
  # Both skipped if the with_orient output already exists.
  if [[ -f "$src" ]]; then
    echo "  [have] $src (stages 1-2 skipped)"
  else
    if [[ ! -f "$fbx" ]]; then echo "  [SKIP] missing FBX $fbx"; fail=$((fail+1)); continue; fi
    echo "  [stage1] FBX -> canonical NPZ (Blender) ..."
    blender --background --python scripts/build_fbx_canonical_human.py -- \
      --fbx "$fbx" --out "$can" \
      || { echo "  [FAIL] stage1 fbx->canonical"; fail=$((fail+1)); continue; }
    echo "  [stage2] canonical NPZ -> with_orient ..."
    python scripts/build_canonical_orientation_frames_fresh.py \
      --in-npz "$can" --out-npz "$src" \
      || { echo "  [FAIL] stage2 canonical->with_orient"; fail=$((fail+1)); continue; }
  fi

  # Stage 2.5 — canonical grounding + persisted contact labels (phasic-v2 M1).
  # ALWAYS recomputed (no skip-if-exists): cheap (pure numpy, no MuJoCo), and
  # Stage 3's skip-if-exists already caused a stale-cache confound during this
  # milestone's own gate testing (see planLog.md M1/T1.1) -- not repeating that
  # risk here for a step this fast.
  echo "  [ground] canonical grounding ${ground_extra:+(extra: $ground_extra) }..."
  python scripts/ground_canonical_human.py \
    --in-npz "$src" --out-npz "$cg" \
    $ground_extra \
    || { echo "  [FAIL] stage2.5 canonical-grounding"; fail=$((fail+1)); continue; }

  # Stage 3 — contact-first IK (only if missing)
  if [[ -f "$cf" ]]; then
    echo "  [have] $cf"
  else
    echo "  [solve] contact-first ${solve_extra:+(extra: $solve_extra) }..."
    python scripts/solve_fbx_canonical_alex_contactfirst.py \
      --canonical "$cg" --out "$cf" \
      --stride "$STRIDE" --max-frames 99999 --ik-iters "$IK_ITERS" \
      --contact-min-run 12 --contact-ramp 16 --contact-preroll "$CONTACT_PREROLL" \
      --coplanar-feet-mode "$COPLANAR_FEET_MODE" \
      --floor-weight "$S3_FLOOR_WEIGHT" --floor-margin "$S3_FLOOR_MARGIN" --floor-gain "$S3_FLOOR_GAIN" \
      $S3_HARD_TIER_FLAG $SWING_CLEAR_FLAG $LEG_FLOOR_REFINE_FLAG \
      --log-every 200 $solve_extra \
      || { echo "  [FAIL] contact-first"; fail=$((fail+1)); continue; }
  fi

  # Stage 4 — GlobalOPT Stage-A smoothing + Stage-B contact QP (soft self-
  # collision always on; solver defaults to the canonical fullmesh model)
  echo "  [smooth] GlobalOPT ${go_extra:+(clip-extra: $go_extra) }..."
  python scripts/solve_global_trajectory_opt_contactfirst.py \
    --ik-npz "$cf" --out "$go" --lambda-smooth "$LAMBDA_SMOOTH" --n-outer "$N_OUTER" \
    --foot-weight "$FOOT_WEIGHT" --hand-weight "$HAND_WEIGHT" \
    --plant-min-run "$PLANT_MIN_RUN" \
    --floor-weight "$FLOOR_WEIGHT" --floor-mode "$FLOOR_MODE" \
    --floor-collision "$FLOOR_COLLISION" \
    --sens-min-pen "$SENS_MIN_PEN" --sens-foot-min-pen "$SENS_FOOT_MIN_PEN" \
    --collision-penalty 1000 $go_extra \
    || { echo "  [FAIL] globalopt"; fail=$((fail+1)); continue; }

  # Stage 4.5 — Z-grounding (plant lowest contact point on the floor)
  echo "  [ground] $GROUND_MODE ..."
  python scripts/post_process_ground_contactfirst.py \
    --npz "$go" --out "$gr" --mode "$GROUND_MODE" --smooth-shift "$GROUND_SMOOTH" \
    --percentile "$GROUND_PERCENTILE" --contact-percentile "$GROUND_CONTACT_PERCENTILE" \
    --still-speed "$GROUND_STILL_SPEED" \
    || { echo "  [FAIL] ground"; fail=$((fail+1)); continue; }

  # Stage 4.6 — physics plausibility pass (phasic-v2 M4, opt-in). $final is
  # what every downstream stage (render/export) consumes — $gr unmodified
  # when off, so PHYSICS_PASS=off (default) is byte-for-byte identical to
  # before this stage existed.
  final="$gr"
  if [ "$PHYSICS_PASS" = "on" ]; then
    ph="$PH/${name}_physics.npz"
    echo "  [physics] plausibility pass ..."
    python scripts/physics_plausibility_pass.py \
      --npz "$gr" --out "$ph" \
      || { echo "  [FAIL] physics-plausibility"; fail=$((fail+1)); continue; }
    final="$ph"
  fi

  # Stage 4.7 — per-limb cleanup solver (phasic-v2 M5, opt-in). Consumes
  # $final from whichever stage ran last (physics-plausibility if that's
  # also on, else grounding directly) — $final unmodified when off, so
  # LIMB_REFINE=off (default) is byte-for-byte identical to before this
  # stage existed.
  if [ "$LIMB_REFINE" = "on" ]; then
    lr="$LR/${name}_limbrefine.npz"
    echo "  [limb-refine] per-limb cleanup ..."
    python scripts/refine_limbs_contactfirst.py \
      --npz "$final" --out "$lr" \
      || { echo "  [FAIL] limb-refine"; fail=$((fail+1)); continue; }
    final="$lr"
  fi

  # Stage 5 — render REAL TIME:  fps = source_fps / stride ; no extra frame skip
  RT=$(python - "$final" <<'PY'
import numpy as np, sys
z=np.load(sys.argv[1],allow_pickle=True)
fps=float(z["fps"]); sfi=z["source_frame_ids"]
stride=int(np.round(np.median(np.diff(sfi)))) if len(sfi)>1 else 1
print(f"{fps/max(stride,1):.4f}")
PY
)
  if [ "$RENDER" = "1" ]; then
    echo "  [render] real-time fps=$RT  (mesh=$RENDER_MESH)"
    MUJOCO_GL=egl python scripts/visualization/render_contactfirst.py \
      --npz "$final" --model "$RMODEL" --out-mp4 "$mp4" \
      --width 640 --height 480 --fps "$RT" --frame-step 1 \
      --ground --ground-z 0 $RENDER_EXTRA \
      || { echo "  [FAIL] render"; fail=$((fail+1)); continue; }
  else
    echo "  [render] skipped (RENDER=0)"
  fi

  # Stage 6 — IHMC JSON export from the final NPZ (native 120 Hz, no --fps)
  js="$IH/${name}.json"
  echo "  [export] IHMC json -> $js ..."
  python scripts/export_alex_retarget_npz_to_ihmc_json.py \
    "$final" --out "$js" \
    || { echo "  [FAIL] ihmc-export"; fail=$((fail+1)); continue; }

  # Stage 6b — 50 Hz set (--fps 50), matching the IHMC reference 1.json rate
  js50_note=""
  if [ "$EXPORT_50HZ" = "1" ]; then
    js50="$IH50/${name}.json"
    echo "  [export] IHMC json 50Hz -> $js50 ..."
    python scripts/export_alex_retarget_npz_to_ihmc_json.py \
      "$final" --out "$js50" --fps 50 \
      || { echo "  [FAIL] ihmc-export-50hz"; fail=$((fail+1)); continue; }
    js50_note=" + $js50"
  fi

  echo "  [OK] $mp4 + $js$js50_note"; ok=$((ok+1))
done

echo "======================================================================"
echo "DONE  ok=$ok  fail=$fail  (of ${#CLIPS[@]} clips)"

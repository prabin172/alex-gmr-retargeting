#!/usr/bin/env bash
# retargetingPipeline.sh
# Full contact-first pipeline for every fbx clip:
#   Stage 3    contact-first IK  (skipped if the NPZ already exists)
#   Stage 4    GlobalOPT Stage-A smoothing (spikes->0, collisions down)
#   Stage 4.5  Z-grounding (plant lowest contact point on the floor z=0)
#   Stage 5    render, REAL TIME (fps = source_fps / stride)
#
# Outputs:
#   outputs/contactfirst/<clip>_contactfirst.npz
#   outputs/global_opt_contactfirst/<clip>_global_opt.npz
#   outputs/grounded_contactfirst/<clip>_grounded.npz
#   outputs/renders/contactfirst/<clip>_globalopt.mp4
#
# The Stage-4 solver defaults to the single canonical fullmesh collision model
# (assets/alex/alex_floating_base_with_sites.xml) with always-on soft self-
# collision — no model/flag knobs needed for the shipped setup.
#
# Env knobs (all optional):
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
# Stage-3 coplanar-feet targets: when both feet are contact-labelled, snap their
# ankle-height targets to a common Z so the IK produces coplanar feet. Fixes the
# root cause of the "one foot floats in RDX" bug — the retargeted foot-height
# targets can sit several cm apart (source ankles / per-leg scale differ) while
# both are labelled planted, which a rigid grounding shift can't reconcile.
# mean = distribute (lowest self-collision) | min = snap to lower foot | off = legacy.
COPLANAR_FEET_MODE="${COPLANAR_FEET_MODE:-mean}"
# On-floor + coplanar rows: drive each PLANTED foot's 4 sole-corner Zs to a shared
# floor height (the lower foot's warm-start ground). Co-plants both feet in the
# SOLVE, which a rigid 1-DOF grounding shift cannot. 0 = off. Pairs with
# GROUND_MODE=constant (constant does the absolute z=0 registration; the feet are
# already coplanar so one shift plants both).
FLOOR_WEIGHT="${FLOOR_WEIGHT:-200}"
FLOOR_MODE="${FLOOR_MODE:-estimate}"      # estimate (lower foot's ground) | zero (soles->z=0)
RENDER_EXTRA="${RENDER_EXTRA:-}"          # extra render flags, e.g. "--fixed-cam"
RENDER="${RENDER:-1}"                      # 1 = render Stage 5 mp4 | 0 = skip (faster; JSON export still runs)
# --- Z-grounding (Stage 4.5) ---
GROUND_MODE="${GROUND_MODE:-constant-contact}"   # constant-contact (single shift keyed to planted feet — no bobbing, feet stay down) | perframe (plant every frame, wanders) | constant (single shift, global lowest)
GROUND_SMOOTH="${GROUND_SMOOTH:-80}"      # perframe: tridiagonal smoothing on the shift series (5 @ 30 Hz × 16)
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
CF=outputs/contactfirst
GO="${GO_DIR:-outputs/global_opt_contactfirst}"
GR="${GR_DIR:-outputs/grounded_contactfirst}"
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
echo "Render mesh: $RENDER_MESH -> $RMODEL   |  ground: $GROUND_MODE (smooth=$GROUND_SMOOTH)"

# clip name  ->  canonical *_with_orient.npz input (one per clip; variants deduped)
# Optional 3rd field: extra contact-first SOLVER flags. 4th: extra GLOBALOPT
# flags. BOTH EMPTY by design — one retargeter config for all actions; the
# fields exist for experiments only.
CLIPS=(
  "standup_01|standup_01_with_orient.npz||"
  "standup_02|standup_02_canonical_human_fresh_with_orient.npz||"
  "standup_natural_01|standup_natural_01_with_orient.npz||"
  "standup_natural_02|standup_natural_02_with_orient.npz||"
  "standup_side_04|standup_side_04_with_orient.npz||"
  "standup_side_05|standup_side_05_with_orient.npz||"
  "standup_slideHandsBack_03|standup_slideHandsBack_03_with_orient.npz||"
  "shovel_fronthard_02|PrabinRef_Shovel_FrontHard_02_with_orient.npz||"
  "shovel_leftbucket_02|PrabinRef_Shovel_LeftBucket_02_with_orient.npz||"
  "shovel_lefthard_01|PrabinRef_Shovel_LeftHard_01_with_orient.npz||"
  "shovel_rightbucket_01|PrabinRef_Shovel_RightBucket_01_with_orient.npz||"
  "shovel_righthard_01|PrabinRef_Shovel_RightHard_01_with_orient.npz||"
  "standupFromKneeling_01|PrabinRef_STandupFromKneeling_01_with_orient.npz||"
  "standupFromKneeling_02|PrabinRef_STandupFromKneeling_02_with_orient.npz||"
  "standupKnees_02|PrabinRef_StandupKnees_02_with_orient.npz||"
  "standupSquatCrouch_01|PrabinRef_StandupSquatCrouch_01_with_orient.npz||"
  "kneelingFall_02|PrabinRef_KneelingFall_02_with_orient.npz||"
  "kneelingFall_03|PrabinRef_KneelingFall_03_with_orient.npz||"
)

ok=0; fail=0
for entry in "${CLIPS[@]}"; do
  IFS='|' read -r name infile solve_extra go_extra <<< "$entry"
  src="$IN/$infile"
  cf="$CF/${name}_contactfirst.npz"
  go="$GO/${name}_global_opt.npz"
  gr="$GR/${name}_grounded.npz"
  mp4="$RD/${name}_globalopt.mp4"
  echo "======================================================================"
  echo ">>> $name   ($infile)"
  if [[ ! -f "$src" ]]; then echo "  [SKIP] missing input $src"; fail=$((fail+1)); continue; fi

  # Stage 3 — contact-first IK (only if missing)
  if [[ -f "$cf" ]]; then
    echo "  [have] $cf"
  else
    echo "  [solve] contact-first ${solve_extra:+(extra: $solve_extra) }..."
    python scripts/solve_fbx_canonical_alex_contactfirst.py \
      --canonical "$src" --out "$cf" \
      --stride "$STRIDE" --max-frames 99999 --ik-iters "$IK_ITERS" \
      --contact-min-run 12 --contact-ramp 16 --contact-preroll 8 \
      --coplanar-feet-mode "$COPLANAR_FEET_MODE" \
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
    --collision-penalty 1000 $go_extra \
    || { echo "  [FAIL] globalopt"; fail=$((fail+1)); continue; }

  # Stage 4.5 — Z-grounding (plant lowest contact point on the floor)
  echo "  [ground] $GROUND_MODE ..."
  python scripts/post_process_ground_contactfirst.py \
    --npz "$go" --out "$gr" --mode "$GROUND_MODE" --smooth-shift "$GROUND_SMOOTH" \
    || { echo "  [FAIL] ground"; fail=$((fail+1)); continue; }

  # Stage 5 — render REAL TIME:  fps = source_fps / stride ; no extra frame skip
  RT=$(python - "$gr" <<'PY'
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
      --npz "$gr" --model "$RMODEL" --out-mp4 "$mp4" \
      --width 640 --height 480 --fps "$RT" --frame-step 1 \
      --ground --ground-z 0 $RENDER_EXTRA \
      || { echo "  [FAIL] render"; fail=$((fail+1)); continue; }
  else
    echo "  [render] skipped (RENDER=0)"
  fi

  # Stage 6 — IHMC JSON export from the grounded NPZ (native 120 Hz, no --fps)
  js="$IH/${name}.json"
  echo "  [export] IHMC json -> $js ..."
  python scripts/export_alex_retarget_npz_to_ihmc_json.py \
    "$gr" --out "$js" \
    || { echo "  [FAIL] ihmc-export"; fail=$((fail+1)); continue; }

  # Stage 6b — 50 Hz set (--fps 50), matching the IHMC reference 1.json rate
  js50_note=""
  if [ "$EXPORT_50HZ" = "1" ]; then
    js50="$IH50/${name}.json"
    echo "  [export] IHMC json 50Hz -> $js50 ..."
    python scripts/export_alex_retarget_npz_to_ihmc_json.py \
      "$gr" --out "$js50" --fps 50 \
      || { echo "  [FAIL] ihmc-export-50hz"; fail=$((fail+1)); continue; }
    js50_note=" + $js50"
  fi

  echo "  [OK] $mp4 + $js$js50_note"; ok=$((ok+1))
done

echo "======================================================================"
echo "DONE  ok=$ok  fail=$fail  (of ${#CLIPS[@]} clips)"

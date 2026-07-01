#!/usr/bin/env bash
# run_globalopt_all.sh
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
set -uo pipefail

STRIDE=4
IK_ITERS=40
LAMBDA_SMOOTH="${LAMBDA_SMOOTH:-10}"     # GlobalOPT Stage-A smoothing weight
RENDER_EXTRA="${RENDER_EXTRA:-}"          # extra render flags, e.g. "--fixed-cam"
# --- Z-grounding (Stage 4.5) ---
GROUND_MODE="${GROUND_MODE:-perframe}"    # perframe (plant every frame) | constant (single per-clip shift)
GROUND_SMOOTH="${GROUND_SMOOTH:-5}"       # perframe: tridiagonal smoothing on the shift series
# --- Render mesh (Stage 5) ---
# visual    = Alex visual mesh, hands drawn as closed fists (good visualisation)
# collision = v2 collision convex hulls (what the solver actually uses)
# <path>    = any explicit model xml
RENDER_MESH="${RENDER_MESH:-visual}"
case "$RENDER_MESH" in
  visual)    RMODEL="assets/alex/alex_visual_mesh_fist_hands.xml" ;;
  collision) RMODEL="assets/alex/alex_floating_base_with_sites_v2.xml" ;;
  *)         RMODEL="$RENDER_MESH" ;;
esac
IN=outputs/canonical_human/fbx_fresh
CF=outputs/contactfirst
GO=outputs/global_opt_contactfirst
GR=outputs/grounded_contactfirst
RD="${RENDER_DIR:-outputs/renders/contactfirst}"
mkdir -p "$CF" "$GO" "$GR" "$RD"
echo "Render mesh: $RENDER_MESH -> $RMODEL   |  ground: $GROUND_MODE (smooth=$GROUND_SMOOTH)"

# clip name  ->  canonical *_with_orient.npz input (one per clip; variants deduped)
CLIPS=(
  "standup_01|standup_01_with_orient.npz"
  "standup_02|standup_02_canonical_human_fresh_with_orient.npz"
  "standup_natural_01|standup_natural_01_with_orient.npz"
  "standup_natural_02|standup_natural_02_with_orient.npz"
  "standup_side_04|standup_side_04_with_orient.npz"
  "standup_side_05|standup_side_05_with_orient.npz"
  "standup_slideHandsBack_03|standup_slideHandsBack_03_with_orient.npz"
  "shovel_fronthard_02|PrabinRef_Shovel_FrontHard_02_with_orient.npz"
  "shovel_leftbucket_02|PrabinRef_Shovel_LeftBucket_02_with_orient.npz"
  "shovel_lefthard_01|PrabinRef_Shovel_LeftHard_01_with_orient.npz"
  "shovel_rightbucket_01|PrabinRef_Shovel_RightBucket_01_with_orient.npz"
  "shovel_righthard_01|PrabinRef_Shovel_RightHard_01_with_orient.npz"
)

ok=0; fail=0
for entry in "${CLIPS[@]}"; do
  name="${entry%%|*}"; infile="${entry##*|}"
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
    echo "  [solve] contact-first ..."
    python scripts/solve_fbx_canonical_alex_contactfirst.py \
      --canonical "$src" --out "$cf" \
      --stride "$STRIDE" --max-frames 99999 --ik-iters "$IK_ITERS" \
      --log-every 200 || { echo "  [FAIL] contact-first"; fail=$((fail+1)); continue; }
  fi

  # Stage 4 — GlobalOPT Stage-A smoothing
  echo "  [smooth] GlobalOPT ..."
  python scripts/solve_global_trajectory_opt_contactfirst.py \
    --ik-npz "$cf" --out "$go" --lambda-smooth "$LAMBDA_SMOOTH" \
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
  echo "  [render] real-time fps=$RT  (mesh=$RENDER_MESH)"
  MUJOCO_GL=egl python scripts/visualization/render_contactfirst.py \
    --npz "$gr" --model "$RMODEL" --out-mp4 "$mp4" \
    --width 640 --height 480 --fps "$RT" --frame-step 1 \
    --ground --ground-z 0 $RENDER_EXTRA \
    || { echo "  [FAIL] render"; fail=$((fail+1)); continue; }

  echo "  [OK] $mp4"; ok=$((ok+1))
done

echo "======================================================================"
echo "DONE  ok=$ok  fail=$fail  (of ${#CLIPS[@]} clips)"

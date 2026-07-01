#!/usr/bin/env bash
# run_highori_weights.sh
#
# Re-runs stages 3-5 only (canonical orientation NPZs already exist) with the
# doubled distal orientation weights:
#   left_foot / right_foot:  0.35 → 0.70
#   left_hand / right_hand:  0.20 → 0.40
#
# Outputs go to separate directories so old results are preserved for comparison:
#   outputs/ik_highori/       ← IK NPZs (with orientation_errors_deg)
#   outputs/grounded_highori/ ← grounded NPZs
#   outputs/renders_highori/  ← MP4s

set -euo pipefail

CONDA_ENV="gmr"
MODEL="assets/alex/alex_floating_base_with_sites.xml"
VISUAL_MODEL="assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml"

IK_ITERS=80
ORI_SCALE=1.0
STRIDE=1
MAX_FRAMES=99999

RENDER_WIDTH=640
RENDER_HEIGHT=480
RENDER_FPS=30

OUT_CANONICAL="outputs/canonical_human/fbx_fresh"
OUT_IK="outputs/ik_highori"
OUT_GROUNDED="outputs/grounded_highori"
OUT_RENDERS="outputs/renders_highori"

mkdir -p "$OUT_IK" "$OUT_GROUNDED" "$OUT_RENDERS"

if [ "$#" -gt 0 ]; then
    FBX_FILES=("$@")
else
    mapfile -t FBX_FILES < <(find data/raw/inhouse -name "*.fbx" | sort)
fi

echo "===================================================================="
echo "High-ori-weight run — ${#FBX_FILES[@]} clip(s)"
echo "  left_foot/right_foot: 0.35 → 0.70"
echo "  left_hand/right_hand: 0.20 → 0.40"
echo "===================================================================="
echo ""

for FBX in "${FBX_FILES[@]}"; do
    STEM=$(basename "$FBX" .fbx)

    ORIENT_NPZ="$OUT_CANONICAL/${STEM}_with_orient.npz"
    IK_NPZ="$OUT_IK/${STEM}_ik_highori.npz"
    GROUNDED_NPZ="$OUT_GROUNDED/${STEM}_grounded_highori.npz"
    RENDER_MP4="$OUT_RENDERS/${STEM}_highori.mp4"

    if [ ! -f "$ORIENT_NPZ" ]; then
        echo "SKIP $STEM — orientation NPZ not found: $ORIENT_NPZ"
        continue
    fi

    echo "--------------------------------------------------------------------"
    echo "Clip: $STEM"
    echo "--------------------------------------------------------------------"

    echo "[3/5] IK solve (highori weights)"
    conda run -n "$CONDA_ENV" python scripts/solve_fbx_canonical_alex_posori_qp_fresh_worlddelta.py \
        --canonical "$ORIENT_NPZ" \
        --model "$MODEL" \
        --out "$IK_NPZ" \
        --stride "$STRIDE" \
        --max-frames "$MAX_FRAMES" \
        --ik-iters "$IK_ITERS" \
        --ori-scale "$ORI_SCALE"
    echo "      -> $IK_NPZ"

    echo "[4/5] Grounding + contact labels"
    conda run -n "$CONDA_ENV" python scripts/post_process_grounding_contacts.py \
        --in-npz "$IK_NPZ" \
        --model "$MODEL" \
        --out-npz "$GROUNDED_NPZ"
    echo "      -> $GROUNDED_NPZ"

    echo "[5/5] Render"
    MUJOCO_GL=egl conda run -n "$CONDA_ENV" python scripts/visualization/render_alex_qp_direct_mp4_fresh.py \
        --npz "$GROUNDED_NPZ" \
        --model "$VISUAL_MODEL" \
        --out-mp4 "$RENDER_MP4" \
        --width "$RENDER_WIDTH" \
        --height "$RENDER_HEIGHT" \
        --fps "$RENDER_FPS"
    echo "      -> $RENDER_MP4"

    echo ""
done

echo "===================================================================="
echo "Done. Compare orientation errors with:"
echo "  conda run -n gmr python scripts/compare_ori_weights.py"
echo "===================================================================="

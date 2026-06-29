#!/usr/bin/env bash
# retargetingPipeline.sh — End-to-end retargeting pipeline for all FBX clips
#
# Stages:
#   1. FBX → canonical positions  (Blender)
#   2. Canonical positions → positions + orientation frames  (Python)
#   3. IK solve: canonical → Alex qpos  (Python / MuJoCo)
#   4. Grounding + contact labels  (Python / MuJoCo)
#   5. Render: grounded qpos → MP4  (Python / MuJoCo)
#
# Usage (full pipeline, all clips):
#   bash retargetingPipeline.sh
#
# Usage (specific clips only):
#   bash retargetingPipeline.sh data/raw/inhouse/shoveling/PrabinRef_Shovel_LeftBucket_02.fbx

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BLENDER="${BLENDER:-blender}"          # override: BLENDER=/path/to/blender bash ...
CONDA_ENV="gmr"
MODEL="assets/alex/alex_floating_base_with_sites.xml"
VISUAL_MODEL="assets/alex/temp_alex_floating_base_visual_mesh_only_nosites.xml"

# IK solve parameters (tune per-project if needed)
IK_ITERS=80
ORI_SCALE=1.0
STRIDE=1
MAX_FRAMES=99999

# Render parameters (frame-step is auto-detected from source FPS for real-time playback)
RENDER_WIDTH=640    # each panel; total output is 1280x480 (side-by-side robot + human)
RENDER_HEIGHT=480
RENDER_FPS=30

# ---------------------------------------------------------------------------
# Output directories (all flat — no subdirectory per clip)
# ---------------------------------------------------------------------------
OUT_CANONICAL="outputs/canonical_human/fbx_fresh"
OUT_IK="outputs/ik"
OUT_GROUNDED="outputs/grounded"
OUT_RENDERS="outputs/renders"

mkdir -p "$OUT_CANONICAL" "$OUT_IK" "$OUT_GROUNDED" "$OUT_RENDERS"

# ---------------------------------------------------------------------------
# File discovery
# If arguments are given, use them as the FBX list; otherwise find all FBX
# files under data/raw/inhouse/ (merging both subdirectories into one flat
# output namespace).
# ---------------------------------------------------------------------------
if [ "$#" -gt 0 ]; then
    FBX_FILES=("$@")
else
    mapfile -t FBX_FILES < <(find data/raw/inhouse -name "*.fbx" | sort)
fi

echo "===================================================================="
echo "Retargeting pipeline — ${#FBX_FILES[@]} clip(s)"
echo "===================================================================="
echo ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for FBX in "${FBX_FILES[@]}"; do
    STEM=$(basename "$FBX" .fbx)

    # Intermediate and final output paths for this clip
    POS_NPZ="$OUT_CANONICAL/${STEM}.npz"
    ORIENT_NPZ="$OUT_CANONICAL/${STEM}_with_orient.npz"
    IK_NPZ="$OUT_IK/${STEM}_ik.npz"
    GROUNDED_NPZ="$OUT_GROUNDED/${STEM}_grounded.npz"
    RENDER_MP4="$OUT_RENDERS/${STEM}_grounded.mp4"

    echo "--------------------------------------------------------------------"
    echo "Clip: $STEM  ($FBX)"
    echo "--------------------------------------------------------------------"

    # -----------------------------------------------------------------------
    # Step 1 — FBX → canonical positions (Blender)
    #
    # Imports the FBX in Blender, walks each frame, extracts bone-head
    # positions for all canonical roles, and converts from Blender/FBX
    # coordinates (+X right, +Y fwd, +Z up) to canonical (+X fwd, +Y left,
    # +Z up). The actual animation length is detected from the armature
    # actions, so clips longer than Blender's default 250-frame scene range
    # are handled correctly.
    # -----------------------------------------------------------------------
    echo "[1/5] FBX → canonical positions"
    "$BLENDER" --background --python scripts/build_fbx_canonical_human.py -- \
        --fbx "$FBX" \
        --out "$POS_NPZ"
    echo "      -> $POS_NPZ"

    # -----------------------------------------------------------------------
    # Step 2 — Canonical positions → positions + orientation frames (Python)
    #
    # Builds per-frame semantic orientation matrices for 7 roles (pelvis,
    # torso, head, both feet, both hands) from landmark positions rather than
    # raw FBX bone rotations. Also auto-detects and corrects the actor's
    # facing direction: if the actor faces any axis other than +X, a yaw
    # correction (snapped to 90°) is applied to all positions before building
    # the orientation frames. The corrected positions are stored back so all
    # downstream stages see a consistently +X-facing actor.
    # -----------------------------------------------------------------------
    echo "[2/5] Positions → orientation frames  (facing auto-corrected)"
    conda run -n "$CONDA_ENV" python scripts/build_canonical_orientation_frames_fresh.py \
        --in-npz "$POS_NPZ" \
        --out-npz "$ORIENT_NPZ"
    echo "      -> $ORIENT_NPZ"

    # -----------------------------------------------------------------------
    # Step 3 — IK solve: canonical → Alex qpos (Python / MuJoCo)
    #
    # Damped-least-squares QP IK in MuJoCo velocity space. Position targets
    # for proximal joints + world-delta orientation targets for distal joints
    # (feet, hands, head). Per-role morphology scales are computed after an
    # initial alignment IK so limb proportions match Alex rather than the
    # human source. Posture regularization pulls the robot toward its rest
    # pose between frames. Output: qpos (N, 36) in Alex's free-root format
    # [x, y, z, qw, qx, qy, qz, 29_joints].
    # -----------------------------------------------------------------------
    echo "[3/5] IK solve"
    conda run -n "$CONDA_ENV" python scripts/solve_fbx_canonical_alex_posori_qp_fresh_worlddelta.py \
        --canonical "$ORIENT_NPZ" \
        --model "$MODEL" \
        --out "$IK_NPZ" \
        --stride "$STRIDE" \
        --max-frames "$MAX_FRAMES" \
        --ik-iters "$IK_ITERS" \
        --ori-scale "$ORI_SCALE"
    echo "      -> $IK_NPZ"

    # -----------------------------------------------------------------------
    # Step 4 — Grounding + contact labels (Python / MuJoCo)
    #
    # Post-hoc root-Z correction: for each frame, finds the minimum Z of all
    # robot collision geom surfaces (using exact formulas per geom type: box
    # corners, capsule endpoints ± radius, cylinder rim) and shifts qpos[2]
    # upward so the lowest surface is at Z = 0.
    #
    # Contact labels: after grounding, flags 11 bodies as "in contact" if
    # any of their collision geoms is within 2 cm of the ground plane.
    # Bodies: LEFT_FOOT, RIGHT_FOOT, LEFT_SHIN, RIGHT_SHIN, LEFT_THIGH,
    #         RIGHT_THIGH, PELVIS_LINK, TORSO_LINK, HEAD_LINK,
    #         LEFT_GRIPPER_Z_LINK, RIGHT_GRIPPER_Z_LINK.
    #
    # Output NPZ adds: qpos_raw (original), qpos (grounded),
    # root_z_shifts (N,), contact_labels (N, 11), contact_body_names.
    # -----------------------------------------------------------------------
    echo "[4/5] Grounding + contact labels"
    conda run -n "$CONDA_ENV" python scripts/post_process_grounding_contacts.py \
        --in-npz "$IK_NPZ" \
        --model "$MODEL" \
        --out-npz "$GROUNDED_NPZ"
    echo "      -> $GROUNDED_NPZ"

    # -----------------------------------------------------------------------
    # Step 5 — Render: grounded qpos → MP4 (Python / MuJoCo)
    #
    # Renders each frame of the grounded qpos using MuJoCo's EGL renderer
    # and the visual-mesh-only Alex model (no collision geoms or sites shown).
    # The camera slowly orbits and zooms out, tracking the robot's centroid.
    # -----------------------------------------------------------------------
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
    echo "Done: $STEM"
    echo ""
done

echo "===================================================================="
echo "Pipeline complete — ${#FBX_FILES[@]} clip(s) processed."
echo "Grounded NPZ files: $OUT_GROUNDED/"
echo "Renders:            $OUT_RENDERS/"
echo "===================================================================="

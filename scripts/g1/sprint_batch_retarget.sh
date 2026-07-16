#!/usr/bin/env bash
# S1-T1: batch retarget all LAFAN1 clips -> G1 pkl + human targets NPZ.
# Resumable: skips a clip if its pkl already exists (sprint ground rule 2).
set -uo pipefail

REPO="/home/ptimilsina/projects/alex-gmr-retargeting"
BVH_DIR="$REPO/data/raw/lafan1"
PKL_DIR="$REPO/outputs/gmr_baseline/sprint/pkl"
HT_DIR="$REPO/outputs/gmr_baseline/sprint/human_targets"
LOG="$REPO/outputs/gmr_baseline/sprint/s1t1_retarget.log"

mkdir -p "$PKL_DIR" "$HT_DIR"
: > "$LOG.fail"

n=0
total=$(ls "$BVH_DIR"/*.bvh | wc -l)
for bvh in "$BVH_DIR"/*.bvh; do
  n=$((n+1))
  clip=$(basename "$bvh" .bvh)
  pkl="$PKL_DIR/${clip}.pkl"
  ht="$HT_DIR/${clip}.npz"
  if [ -f "$pkl" ] && [ -f "$ht" ]; then
    echo "[$n/$total] SKIP (exists) $clip" >> "$LOG"
    continue
  fi
  echo "[$n/$total] retargeting $clip ..." >> "$LOG"
  if conda run -n gmr python "$REPO/scripts/g1/gmr_headless_retarget.py" \
      --bvh_file "$bvh" --robot unitree_g1 \
      --save_path "$pkl" \
      --save_human_targets "$ht" >> "$LOG" 2>&1; then
    echo "[$n/$total] OK $clip" >> "$LOG"
  else
    echo "[$n/$total] FAIL $clip" >> "$LOG"
    echo "$clip" >> "$LOG.fail"
  fi
done
echo "DONE. failures: $(wc -l < "$LOG.fail")" >> "$LOG"

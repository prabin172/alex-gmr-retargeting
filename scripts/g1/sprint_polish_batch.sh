#!/usr/bin/env bash
# S1-T2: per clip, produce <clip>_gmrfix.pkl (heightfix on raw) and
# <clip>_polished.pkl (Stage A + ground-constant on raw -- sprint ground rule 3:
# polish is applied to RAW, never stacked on heightfix). Resumable: skips a
# variant if its pkl already exists.
set -uo pipefail

REPO="/home/ptimilsina/projects/alex-gmr-retargeting"
BVH_DIR="$REPO/data/raw/lafan1"
PKL_DIR="$REPO/outputs/gmr_baseline/sprint/pkl"
LOG="$REPO/outputs/gmr_baseline/sprint/s1t2_polish.log"

: > "$LOG.fail"

n=0
# Clip list from the BVH source directory (S1-T1's ground truth), NOT a glob over
# PKL_DIR -- that dir also holds stray non-sprint variant pkls (week-1/2 leftovers)
# whose names collide with our own suffix convention; globbing them as "raw clips"
# corrupted a prior run (see planLogGMR.md S1-T2).
mapfile -t clips < <(ls "$BVH_DIR"/*.bvh | xargs -n1 basename | sed 's/\.bvh$//' | sort)
total=${#clips[@]}
for clip in "${clips[@]}"; do
  n=$((n+1))
  raw="$PKL_DIR/${clip}.pkl"
  gmrfix="$PKL_DIR/${clip}_gmrfix.pkl"
  polished="$PKL_DIR/${clip}_polished.pkl"

  if [ -f "$gmrfix" ]; then
    echo "[$n/$total] SKIP gmrfix (exists) $clip" >> "$LOG"
  else
    echo "[$n/$total] gmrfix $clip ..." >> "$LOG"
    if conda run -n gmr python "$REPO/scripts/g1/polish_gmr_pkl.py" \
        --in "$raw" --out "$gmrfix" --heightfix >> "$LOG" 2>&1; then
      echo "[$n/$total] OK gmrfix $clip" >> "$LOG"
    else
      echo "[$n/$total] FAIL gmrfix $clip" >> "$LOG"
      echo "gmrfix:$clip" >> "$LOG.fail"
    fi
  fi

  if [ -f "$polished" ]; then
    echo "[$n/$total] SKIP polished (exists) $clip" >> "$LOG"
  else
    echo "[$n/$total] polished $clip ..." >> "$LOG"
    if conda run -n gmr python "$REPO/scripts/g1/polish_gmr_pkl.py" \
        --in "$raw" --out "$polished" --stage-a --ground >> "$LOG" 2>&1; then
      echo "[$n/$total] OK polished $clip" >> "$LOG"
    else
      echo "[$n/$total] FAIL polished $clip" >> "$LOG"
      echo "polished:$clip" >> "$LOG.fail"
    fi
  fi
done
echo "DONE. failures: $(wc -l < "$LOG.fail")" >> "$LOG"

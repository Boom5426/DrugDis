#!/usr/bin/env bash
# M3 (factorized heads, response-only supervision) under LCLO and LSO, five seeds,
# on the frozen benchmark dataset. Every other argument is the trainer default,
# as recorded in the results.json of the reported M3 runs.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT="$RUNS/Phase2C_M3"
LOG="$OUT/logs"; mkdir -p "$LOG"
cd "$REPO/drugdis/models"
for SP in LCLO LSO; do
  for S in 3407 3408 3409 3410 3411; do
    n="M3_${SP}_s${S}"
    echo "=== start $n $(date -Is) ===" >> "$LOG/m3.log"
    $PY train_2a2.py --model M3 --seed $S --split_type $SP --output_dir "$OUT" \
        --substrate_config "$CFG" --manifest_dir "$MAN" > "$LOG/${n}.log" 2>&1
    echo "=== end   $n $(date -Is) rc=$? ===" >> "$LOG/m3.log"
  done
done
echo "M3_DONE $(date -Is)" >> "$LOG/m3.log"

#!/usr/bin/env bash
# M0 against M4 under held-out compounds (LSO) on the frozen benchmark dataset.
# Five paired seeds, checkpoint = argmin validation MSE_raw (best_valmse.pth).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT="$RUNS/Phase2B_LSO/"
LOG="$OUT/logs"; mkdir -p "$LOG"
cd "$REPO/drugdis/models"
run () {
  local name="$1"; shift
  echo "=== start $name $(date -Is) ===" >> "$LOG/bridge.log"
  $PY train_2a2.py "$@" --output_dir "$OUT" --substrate_config "$CFG" --manifest_dir "$MAN" \
      > "$LOG/${name}.log" 2>&1
  echo "=== end   $name $(date -Is) rc=$? ===" >> "$LOG/bridge.log"
}
for S in 3407 3408 3409 3410 3411; do
  run M0_s${S} --model M0 --seed $S --split_type LSO
  run M4_s${S} --model M4 --seed $S --split_type LSO
done
echo "LSO_DONE $(date -Is)" >> "$LOG/bridge.log"

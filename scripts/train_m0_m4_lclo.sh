#!/usr/bin/env bash
# M0 against M4 under held-out cell lines (LCLO) on the frozen benchmark dataset.
# Five paired seeds, checkpoint = argmin validation MSE_raw (best_valmse.pth).
# Architecture, loss, optimizer, epochs, batch size and dropout are the trainer
# defaults. Arms run adjacent within a seed and strictly sequentially.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT="$RUNS/Phase2B_clean/"
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
  run M0_s${S} --model M0 --seed $S
  run M4_s${S} --model M4 --seed $S
done
echo "BRIDGE_DONE $(date -Is)" >> "$LOG/bridge.log"

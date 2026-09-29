#!/usr/bin/env bash
# One parallel worker over a disjoint slice of the deterministic benchmark job list.
#   bash scripts/run_worker.sh <n_workers> <worker_index>
# The slice is materialised to a file first and the trainer's stdin is closed;
# otherwise the trainer inherits the job-list pipe as stdin and consumes the
# remaining lines. Concurrency was verified neutral (29 of 29 epochs bit-identical
# against a sequential run).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
OUT="$RUNS/PaperRerun/"
LOG="$OUT/logs"; mkdir -p "$LOG"
N="$1"; W="$2"
SLICE="$LOG/.slice_${N}_${W}"
$PY "$REPO/scripts/gen_joblist.py" "$N" "$W" 2>/dev/null > "$SLICE"
cd "$REPO/drugdis/models"
echo "=== w$W has $(wc -l < "$SLICE") jobs $(date -Is) ===" >> "$LOG/benchmark.log"
while IFS='|' read -r tag sp sd extra; do
  [ -z "$tag" ] && continue
  [ -f "${OUT}ECFP4__baseline/${sp}/${tag}__M0_seed${sd}/results.json" ] && continue
  n="${tag}__${sp}_s${sd}"
  echo "=== w$W start $n $(date -Is) ===" >> "$LOG/benchmark.log"
  # shellcheck disable=SC2086
  $PY train_2a2.py --model M0 --seed "$sd" --split_type "$sp" --rep_tag "$tag" \
      --output_dir "$OUT" --substrate_config "$CFG" --manifest_dir "$MAN" $extra \
      > "$LOG/${n}.log" 2>&1 < /dev/null
  echo "=== w$W end   $n $(date -Is) rc=$? ===" >> "$LOG/benchmark.log"
done < "$SLICE"
echo "=== WORKER${W}_DONE $(date -Is) ===" >> "$LOG/benchmark.log"

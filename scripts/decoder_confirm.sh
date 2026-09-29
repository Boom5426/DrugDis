#!/usr/bin/env bash
# Three-seed confirmation of the transcriptome-side ordering under the two
# decoders that can express interaction (seeds 3408 and 3409 added to 3407).
# Ridge cannot express interaction and is excluded. The L2 penalty of the
# bilinear decoder is re-selected on each seed's own validation split.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
LOG="$RUNS/DecoderBench/logs"; mkdir -p "$LOG"
cd "$REPO/drugdis/decoder"
until grep -q SWEEP_DONE "$LOG/sweep.log" 2>/dev/null; do sleep 30; done
for S in 3408 3409; do
  for G in raw scGPT CellPLM BulkFormer; do
    n="ECFP4__${G}__tower__s${S}"
    echo "=== start $n $(date -Is) ===" >> "$LOG/confirm.log"
    $PY decoder_bench.py --drug_rep ECFP4 --gene_rep "$G" --decoder tower --seed "$S" \
        > "$LOG/${n}.log" 2>&1
    echo "=== end   $n $(date -Is) rc=$? ===" >> "$LOG/confirm.log"
    for WD in 1e-5 1e-3 1e-1 1e1; do
      n="ECFP4__${G}__bilinear__s${S}__wd${WD}"
      echo "=== start $n $(date -Is) ===" >> "$LOG/confirm.log"
      $PY decoder_bench.py --drug_rep ECFP4 --gene_rep "$G" --decoder bilinear --seed "$S" \
          --weight_decay "$WD" --tag_extra "wd${WD}" > "$LOG/${n}.log" 2>&1
      echo "=== end   $n $(date -Is) rc=$? ===" >> "$LOG/confirm.log"
    done
  done
done
echo "CONFIRM_DONE $(date -Is)" >> "$LOG/confirm.log"

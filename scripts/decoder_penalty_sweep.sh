#!/usr/bin/env bash
# The frozen weight_decay of 1e-5 is not a ridge penalty for a 15,961-dimensional
# linear model fitted from 690 training cell lines, so for the two
# normalisation-free decoders the penalty is selected on VALIDATION from a fixed
# grid that contains the frozen value; selection uses validation MSE only, never
# test. The dual tower keeps its frozen hyperparameters, which favours the two
# comparison decoders and makes agreeing rankings a conservative finding.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
LOG="$RUNS/DecoderBench/logs"; mkdir -p "$LOG"
cd "$REPO/drugdis/decoder"
until grep -q MATRIX_DONE "$LOG/matrix.log" 2>/dev/null; do sleep 30; done
for WD in 1e-3 1e-1 1e1; do
  for K in ridge bilinear; do
    for D in ECFP4 MolCLR KPGT UniMol; do
      n="${D}__raw__${K}__wd${WD}"
      echo "=== start $n $(date -Is) ===" >> "$LOG/sweep.log"
      $PY decoder_bench.py --drug_rep "$D" --gene_rep raw --decoder "$K" \
          --weight_decay "$WD" --tag_extra "wd${WD}" > "$LOG/${n}.log" 2>&1
      echo "=== end   $n $(date -Is) rc=$? ===" >> "$LOG/sweep.log"
    done
    for G in scGPT CellPLM BulkFormer; do
      n="ECFP4__${G}__${K}__wd${WD}"
      echo "=== start $n $(date -Is) ===" >> "$LOG/sweep.log"
      $PY decoder_bench.py --drug_rep ECFP4 --gene_rep "$G" --decoder "$K" \
          --weight_decay "$WD" --tag_extra "wd${WD}" > "$LOG/${n}.log" 2>&1
      echo "=== end   $n $(date -Is) rc=$? ===" >> "$LOG/sweep.log"
    done
  done
done
echo "SWEEP_DONE $(date -Is)" >> "$LOG/sweep.log"

#!/usr/bin/env bash
# Decoder-robustness matrix, seed 3407. Anchor design: 7 unique representation
# pairs x 3 decoder classes = 21 fits; (ECFP4, measured expression) is the anchor
# and appears on both sides. Inputs: aligned embeddings from
# drugdis/decoder/align_gene_reps.py (or drugdis/data/align_all_gene_reps.py).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
LOG="$RUNS/DecoderBench/logs"; mkdir -p "$LOG"
cd "$REPO/drugdis/decoder"
run () {
  local d="$1" g="$2" k="$3"
  local n="${d}__${g}__${k}"
  echo "=== start $n $(date -Is) ===" >> "$LOG/matrix.log"
  $PY decoder_bench.py --drug_rep "$d" --gene_rep "$g" --decoder "$k" > "$LOG/${n}.log" 2>&1
  echo "=== end   $n $(date -Is) rc=$? ===" >> "$LOG/matrix.log"
}
for K in ridge bilinear tower; do
  for D in ECFP4 MolCLR KPGT UniMol; do run "$D" raw "$K"; done
  for G in scGPT CellPLM BulkFormer; do run ECFP4 "$G" "$K"; done
done
echo "MATRIX_DONE $(date -Is)" >> "$LOG/matrix.log"

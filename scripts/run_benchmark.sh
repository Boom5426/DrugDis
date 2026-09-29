#!/usr/bin/env bash
# Representation benchmark on the frozen benchmark dataset with the M0 dual tower,
# frozen manifests and the frozen checkpoint rule (argmin validation MSE_raw).
# Anchor design: 12 compound representations against measured expression and 11
# transcriptome representations against ECFP4, with (ECFP4, expression) shared.
# LCLO and LSO, seeds 3407 to 3409. Sequential; see run_worker.sh for parallel
# workers over the same job list. Transcriptome representations must first be
# aligned to the frozen samples with drugdis/data/align_all_gene_reps.py.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MOL="$DRUGDIS_DATA/Molecule_Embeddings"
ALN="$WORK/aligned_ccle"
OUT="$RUNS/PaperRerun/"
LOG="$OUT/logs"; mkdir -p "$LOG"
cd "$REPO/drugdis/models"
run () {
  local tag="$1"; shift
  local sp="$1"; shift
  local sd="$1"; shift
  local n="${tag}__${sp}_s${sd}"
  [ -f "${OUT}ECFP4__baseline/${sp}/${tag}__M0_seed${sd}/results.json" ] && return 0
  echo "=== start $n $(date -Is) ===" >> "$LOG/benchmark.log"
  $PY train_2a2.py --model M0 --seed "$sd" --split_type "$sp" --rep_tag "$tag" \
      --output_dir "$OUT" --substrate_config "$CFG" --manifest_dir "$MAN" "$@" \
      > "$LOG/${n}.log" 2>&1
  echo "=== end   $n $(date -Is) rc=$? ===" >> "$LOG/benchmark.log"
}
for SD in 3407 3408 3409; do
  for SP in LCLO LSO; do
    # transcriptome side, compound representation fixed at ECFP4
    run "gene-raw" "$SP" "$SD"
    for G in scGPT CellPLM BulkFormer Geneformer scFoundation SCimilarity STATE Tahoe_x1 tGPT UCE Cell2Sentence; do
      run "gene-${G}" "$SP" "$SD" --gene_rep_path "$ALN/${G}_ccle_aligned.parquet"
    done
    # compound side, transcriptome fixed at measured expression
    for D in ChemBERTa2:384 Chemprop:300 InfoAlign:300 KPGT:2304 MolCLR:512 MolT5:768 \
             Mole_BERT:300 Ouroboros:2048 UniMol:512; do
      nm="${D%%:*}"; dm="${D##*:}"
      run "drug-${nm}" "$SP" "$SD" --drug_rep_path "$MOL/${nm}_emb${dm}.pickle"
    done
    run "drug-GeminiMol" "$SP" "$SD" --drug_rep_path "$MOL/GeminiMol_emb2048.pickle" --allow_partial_drug_rep
    run "drug-UniMolV2"  "$SP" "$SD" --drug_rep_path "$MOL/UniMolV2_emb1024.pickle" --allow_partial_drug_rep
  done
done
echo "BENCHMARK_DONE $(date -Is)" >> "$LOG/benchmark.log"

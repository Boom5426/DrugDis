#!/usr/bin/env bash
# Rebuild every canonical table reported in the manuscript from the processed data
# (DRUGDIS_DATA) and the training runs (DRUGDIS_RUNS), in dependency order, then
# compare each rebuilt table with results/tables/TABLES.json by SHA-256.
#
# Expected run layout under DRUGDIS_RUNS (written by the launchers in scripts/):
#   Phase2B_clean/ECFP4__baseline/LCLO/{M0,M4}_seed3407..3411   train_m0_m4_lclo.sh
#   Phase2B_LSO/ECFP4__baseline/LSO/{M0,M4}_seed3407..3411      train_m0_m4_lso.sh
#   Phase2C_M3/ECFP4__baseline/{LCLO,LSO}/M3_seed3407..3411     train_m3.sh
#   PaperRerun/ECFP4__baseline/{LCLO,LSO}/<rep>__M0_seed*       run_benchmark.sh
#   DecoderBench/<drug>__<gene>__<decoder>__seed*               decoder_*.sh
# Everything is written under DRUGDIS_WORK: tables/, staging/, predictions/ and
# the JSON records. Nothing in the repository is modified.
#
#   bash scripts/reproduce_tables.sh            # all steps
#   STEPS="direct T03" bash scripts/reproduce_tables.sh
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
D="$REPO/drugdis"
T="$WORK/tables"; S="$WORK/staging"; P="$WORK/predictions"
mkdir -p "$T" "$S"
STEPS="${STEPS:-verify records direct T03 valmse T18 T19-T22 T23 T24 T25 T26 compare}"
want () { [[ " $STEPS " == *" $1 "* ]]; }
GDSC="$DRUGDIS_DATA/gdsc_sensitivity_data.parquet"
ECFP4="$DRUGDIS_DATA/Molecule_Embeddings/ECFP4_emb2048.pickle"
SEEDS="3407 3408 3409 3410 3411"
fresh () { rm -rf "$1"; mkdir -p "$(dirname "$1")"; }

if want verify; then   # frozen dataset and manifests against their recorded checksums
  $PY "$D/splits/verify_frozen.py"
fi
if want records; then  # intermediate records read by build_tables.py
  $PY "$D/organoid/pdo_preflight.py"
  $PY "$D/organoid/pdo_stress_test.py"
  $PY "$D/decoder/analyse_matrix.py"
  $PY "$D/decoder/analyse_confirm.py"
fi
if want direct; then   # tables assembled directly from runs and records
  rm -f "$T/TABLES.json"
  $PY "$D/tables/build_tables.py"          # T01 T02 T07 T08 T09
  $PY "$D/tables/build_tables_bench.py"    # T04 T05 T06
  $PY "$D/tables/separability.py"          # T12
  $PY "$D/tables/build_T15.py"
  $PY "$D/tables/build_T16.py"
  $PY "$D/tables/build_T17.py"
fi
if want T03; then      # cross-assay reproducibility, multiplicity-preserving bootstrap
  fresh "$S/T03"
  $PY "$D/tables/rebuild_measurement_bootstrap_v2.py" --substrate-config "$CFG" \
      --gdsc-raw "$GDSC" --output-dir "$S/T03"
fi
if want valmse; then   # test predictions at best_valmse.pth, then T10 and T13
  for SP in LCLO LSO; do
    if [ "$SP" = LCLO ]; then ROOT="$RUNS/Phase2B_clean/ECFP4__baseline/LCLO";
    else ROOT="$RUNS/Phase2B_LSO/ECFP4__baseline/LSO"; fi
    for ARM in M0 M4; do for SD in $SEEDS; do
      F="$P/$SP/${ARM}_seed$SD/test_predictions_frozen_valmse.csv.gz"
      [ -f "$F" ] && continue
      mkdir -p "$(dirname "$F")"
      $PY "$D/tables/export_frozen_valmse_predictions_v1.py" --run-dir "$ROOT/${ARM}_seed$SD" \
          --substrate-config "$CFG" --manifest-dir "$MAN" --output-file "$F"
    done; done
  done
  fresh "$S/valmse"
  $PY "$D/tables/rebuild_valmse_identity_tables_v1.py" --prediction-root "$P" \
      --substrate-config "$CFG" --manifest-dir "$MAN" --gdsc-raw "$GDSC" \
      --clio-dir "$DRUGDIS_DATA" --t07 "$T/T07_constructive_m0_m4.csv" --output-dir "$S/valmse"
fi
if want T18; then      # amplitude-sensitive interaction R2 per run
  fresh "$S/T18"
  $PY "$D/tables/build_interaction_r2_table.py" --t07 "$T/T07_constructive_m0_m4.csv" \
      --t08 "$T/T08_pdo_stress.csv" --output-file "$S/T18/T18_interaction_r2.csv" \
      --audit-file "$S/T18/interaction_r2_audit.json"
fi
if want T19-T22; then  # frozen validation protocol, then packages A and B
  fresh "$S/nm"          # the freezer creates the directory and refuses an existing one
  $PY "$D/tables/freeze_nm_validation_protocol.py" --prediction-root "$P" \
      --substrate-config "$CFG" --manifest-dir "$MAN" --gdsc-raw "$GDSC" \
      --response-table "$DRUGDIS_DATA/drug_response.parquet" \
      --output "$S/nm/PROTOCOL.json" --stamp 2026-09-11T19:31:00+08:00
  $PY "$D/tables/run_nm_validation_packages_ab.py" --protocol "$S/nm/PROTOCOL.json" \
      --output-dir "$S/nm/results"
fi
if want T23; then      # matched M0/M3/M4 comparison
  fresh "$S/T23"
  $PY "$D/tables/audit_current_substrate_m3.py" --m3-root "$RUNS/Phase2C_M3" \
      --lclo-root "$RUNS/Phase2B_clean" --lso-root "$RUNS/Phase2B_LSO" \
      --substrate-config "$CFG" --manifest-dir "$MAN" --output-dir "$S/T23"
fi
if want T24; then      # zero-shot organoid evaluation of the M3 LCLO checkpoints
  fresh "$S/T24"          # the evaluator creates the directory and refuses an existing one
  $PY "$D/organoid/evaluate_current_m3_pdo.py" --processed-root "$DRUGDIS_DATA" \
      --drug-features "$ECFP4" --checkpoint-root "$RUNS/Phase2C_M3" \
      --output "$S/T24/results.json" --table "$S/T24/T24_current_m3_pdo.csv"
fi
if want T25; then      # organoid panel counts, cross-checked against T08
  fresh "$S/T25"
  $PY "$D/tables/build_T25_pdo_panel_counts.py" --processed-root "$DRUGDIS_DATA" \
      --reference-audit "$S/T24/results.json" --t08 "$T/T08_pdo_stress.csv" \
      --registry "$REPO/results/tables/TABLES.json" --out-dir "$S/T25"
fi
if want T26; then      # decomposition within each response resource
  fresh "$S/T26"
  $PY "$D/tables/build_T26_resource_decomposition.py" --config "$CFG" \
      --droma "$DRUGDIS_DATA/droma.sqlite" --registry "$REPO/results/tables/TABLES.json" \
      --t01 "$T/T01_substrate.csv" --t02 "$T/T02_decomposition.csv" --out-dir "$S/T26"
fi
if want compare; then  # collect, run the consistency checks, compare with the canonical set
  $PY "$D/tables/compare_with_canonical.py"
  $PY "$D/tables/consistency_pass.py"
fi

# Project structure

| path | contents |
| :--- | :--- |
| `drugdis/evaluate.py` | public API: `decompose` and `component_profile` |
| `drugdis/paths.py` | resolves `DRUGDIS_DATA`, `DRUGDIS_RUNS`, `DRUGDIS_WORK` and the benchmark-dataset config |
| `drugdis/data/` | processed inputs from the DROMA database, ECFP4 fingerprints, the CCLE transcriptomic input matrix, alignment of pretrained transcriptomic embeddings |
| `drugdis/splits/` | prespecified split manifests (`make_manifests.py`) and their checksum verification (`verify_frozen.py`) |
| `drugdis/decomposition/` | the orthogonal response decomposition (`decompositions.py`, alternating projections to 1e-10) and the numerical checks reported in Methods |
| `drugdis/models/` | M0, M3 and M4 (`models.py`), the data loader (`data_fast.py`), the trainer (`train_2a2.py`) and the earlier pipeline's `utils.py` and `model.py`, which the trainer imports |
| `drugdis/decoder/` | decoder-dependence analysis (ridge, bilinear and dual-tower decoders) |
| `drugdis/organoid/` | input-compatibility check and zero-shot evaluation on patient-derived organoid cohorts |
| `drugdis/tables/` | builders of the canonical result tables, and `compare_with_canonical.py` |
| `configs/` | the benchmark-dataset definition (`substrate_config.frozen.json`); data paths are relative to `DRUGDIS_DATA` |
| `manifests/` | prespecified LCLO and LSO split manifests, seeds 3407 to 3411, and their checksum record |
| `results/tables/` | the 24 canonical tables, their registry and an [index](results/tables/README.md) |
| `scripts/` | launchers: training, the representation and decoder benchmarks, manifest rebuild, full table rebuild |
| `examples/` | a synthetic demo of the API |
| `tests/` | unit tests: decomposition, splits, path handling, evaluation API |
| `manuscript/` | manuscript and Supplementary Information (PDF) |
| `assets/` | Fig. 1 of the manuscript, as a static excerpt for the README |
| `FILE_MAP.tsv` | for every file taken from the analysis host: its source path, both SHA-256 values and the reason for any edit |

## Scripts run as files

The analysis scripts import their neighbours by module name (`import
make_manifests`, `from decompositions import ...`) and are run as files, e.g.
`python drugdis/tables/build_T17.py`. Only `drugdis.evaluate` is meant to be
imported as a package.

## Names in the code

The code predates the manuscript's final vocabulary. Field and file names were
kept so that every table, run directory and checksum record stays traceable.

| in the code | in the manuscript |
| :--- | :--- |
| `shared` (`sharedPCC`, `E_shared`, `A_shared`, `var_shared`) | additive component (correlation, error, amplitude, variance) |
| `int`, `interaction`, `residual` (`intPCC`, `E_interaction`, `A_int`) | interaction component |
| `raw` (`rawPCC`, `MSE_raw`) | total response (correlation, prediction error) |
| `substrate`, frozen substrate | benchmark dataset; the CCLE matrix is the transcriptomic input matrix |
| LCLO, LSO | leave-cell-line-out, leave-small-molecule-out |
| PDO | patient-derived organoid cohort |
| `Phase2B_clean`, `Phase2B_LSO`, `Phase2C_M3` | run directories of M0/M4 (LCLO, LSO) and M3 |
| `PaperRerun`, `DecoderBench` | representation benchmark runs, decoder-dependence runs |
| `CLIOSingleHead`, `CLIOFactorized` | the M0 dual tower; the two-head M3/M4 predictor |
| `iePCC`, `dsPCC` | the earlier pipeline's metric; used here only as the learning-rate scheduler's signal |
| `Sensitivity`, `Target_AAC` | the response value as provided by DROMA |
| RISE | the framework's earlier name, kept in schema and file names |

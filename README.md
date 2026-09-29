# DrugDis

Code for **DrugDis: Disentangling general and context-specific effects in
drug-response prediction**.

Bo Li, Chengliang Liu, Yuzhong Peng, Bob Zhang, Qing Wang, Pinxian Zeng, Mengran Li,
Shenghui Huang, Chuxia Deng and Yang Zhang. The manuscript and its Supplementary
Information are in [`paper/`](paper/).

DrugDis applies an exact orthogonal decomposition, y = Hy + My, to measured and
predicted drug responses on the same observed drug–sample pairs. Hy is the
additive component (drug and sample marginals, the best additive fit to the
observed pairs) and My is the drug–sample interaction component. Models are then
evaluated on each component separately: direction (correlation), amplitude and
prediction error, with cross-assay reproducibility as an empirical reference.
Across 3.14 million drug–sample pairs (986 cancer cell lines, 54,180 compounds,
eleven response resources), additive effects accounted for 75.8% of response
variance, and 59–83% within each of the five largest response resources. Under
held-out cell lines, a model with an aggregate correlation of 0.86 recovered
interactions at a correlation of only 0.32.

This repository holds the code for data construction and preprocessing, data
splitting, the decomposition, M0/M3/M4 training and evaluation, and every
analysis table the manuscript reports. The figure-drawing code is not included;
every number a figure shows is in [`results/tables/`](results/tables/).

## Layout

| path | contents |
|---|---|
| `drugdis/data/` | processed inputs from the DROMA database, ECFP4 fingerprints, the CCLE transcriptomic input matrix, alignment of pretrained transcriptomic embeddings |
| `drugdis/splits/` | prespecified split manifests (`make_manifests.py`) and their checksum verification |
| `drugdis/decomposition/` | the orthogonal response decomposition (`decompositions.py`) and the numerical checks reported in Methods |
| `drugdis/models/` | M0, M3 and M4 (`models.py`), the data loader (`data_fast.py`) and the trainer (`train_2a2.py`) |
| `drugdis/decoder/` | decoder-dependence analysis (ridge, bilinear and dual-tower decoders) |
| `drugdis/organoid/` | input-compatibility check and zero-shot evaluation on patient-derived organoid cohorts |
| `drugdis/tables/` | builders of the canonical result tables T01 to T26 |
| `configs/` | the frozen benchmark-dataset definition (`substrate_config.frozen.json`) |
| `manifests/` | the prespecified LCLO and LSO split manifests, seeds 3407 to 3411, and their checksum record |
| `results/tables/` | the canonical tables with their SHA-256 registry (`TABLES.json`) |
| `scripts/` | launchers for training, the benchmarks and the full table rebuild |
| `paper/` | manuscript and Supplementary Information (PDF) |
| `tests/` | unit tests of the decomposition, the splits and the path handling |
| `FILE_MAP.tsv` | the analysis-host source of every published file, both SHA-256 values and the reason for any edit |

## Installation

Python 3.11. The reported analyses ran with the versions pinned in
`requirements.txt` (NumPy 1.26.4, pandas 1.5.3, SciPy 1.15.1, scikit-learn 1.5.2,
PyArrow 18.0.0, PyTorch 2.4.1 built against CUDA 11.8, RDKit 2023.09.4).

```bash
git clone https://github.com/Boom5426/DrugDis
cd DrugDis
conda create -n drugdis python=3.11 && conda activate drugdis
pip install -r requirements.txt
pytest                       # seconds; no data needed
```

Training uses one CUDA GPU; one M0 run on the benchmark dataset takes about seven
minutes on an RTX 4090. The table rebuild from existing runs needs a GPU only for
re-deriving test predictions from checkpoints.

## Data

All inputs are public. The processed inputs are in the Hugging Face dataset
[Boom5426/DrugDis](https://huggingface.co/datasets/Boom5426/DrugDis):

| file | contents |
|---|---|
| `droma.sqlite` | the DROMA database of harmonized preclinical drug-response and omics data |
| `master_table.parquet` | one row per DROMA sample: project, model type, tumour type |
| `drug_response.parquet` | one response per (compound name, sample), with canonical SMILES and resource |
| `gdsc_sensitivity_data.parquet` | GDSC1 and GDSC2 responses kept apart, for the cross-assay reproducibility reference |
| `omics_mrna_raw/` | expression per cohort (samples × genes) |
| `omics_baseline/` | the 15,961-gene list shared by CCLE and GDSC |
| `omics_baseline_frozen/` | the CCLE-derived transcriptomic input matrix of the benchmark dataset and its provenance record |
| `Molecule_Embeddings/` | ECFP4 and eleven pretrained molecular representations |
| `Gene_Embeddings/` | eleven pretrained transcriptomic embeddings |
| `annotations/` | compound name to SMILES tables used to build `drug_response.parquet` |

The DROMA collection is archived at
[doi:10.5281/zenodo.18503188](https://doi.org/10.5281/zenodo.18503188), with its
input-data provenance record at
[doi:10.5281/zenodo.17498421](https://doi.org/10.5281/zenodo.17498421). The
upstream resources are NCI60, PRISM, CTRP1, CTRP2, GDSC1, GDSC2, CCLE, GRAY, gCSI,
FIMM and UHNBreast for cell lines, and UMPDO1–3, HKUPDO and LICOB for organoids.

Every script finds its inputs and outputs through three environment variables
(`drugdis/paths.py`):

```bash
export DRUGDIS_DATA=/path/to/DrugDis-data    # the Hugging Face dataset; required
export DRUGDIS_RUNS=/path/to/runs            # training outputs; default ./runs
export DRUGDIS_WORK=/path/to/work            # analysis outputs, rebuilt tables; default ./work
```

```python
from huggingface_hub import snapshot_download
snapshot_download("Boom5426/DrugDis", repo_type="dataset", local_dir="/path/to/DrugDis-data")
```

## Reproducing the analyses

### 1. Processed inputs (optional; the dataset already contains them)

```bash
python drugdis/data/build_processed_tables.py --out "$DRUGDIS_DATA"
python drugdis/data/build_gdsc_sensitivity.py --out "$DRUGDIS_DATA/gdsc_sensitivity_data.parquet"
python drugdis/data/encode_ecfp4.py --out "$DRUGDIS_DATA/Molecule_Embeddings/ECFP4_emb2048.pickle"
python drugdis/data/build_ccle_substrate.py       # omics_baseline_frozen/
python drugdis/data/align_all_gene_reps.py        # transcriptomic embeddings -> $DRUGDIS_WORK/aligned_ccle
```

`build_processed_tables.py` applies the overlap rule stated in Methods: a compound
name measured on the same sample by several response resources keeps the
measurement of the first resource in a fixed order.

### 2. Benchmark dataset and split manifests

`configs/substrate_config.frozen.json` defines the benchmark dataset: a pair is
eligible if its sample is a cell line with a CCLE expression profile, is not in
the excluded Tavor project, and its compound has an ECFP4 fingerprint; repeated
measurements of a pair are averaged.

```bash
python drugdis/splits/verify_frozen.py     # inputs and manifests against their recorded checksums
bash scripts/make_manifests.sh             # rebuild the manifests into $DRUGDIS_WORK/manifests
```

Rebuilding reproduces the ten shipped manifests byte for byte.

### 3. The decomposition

```python
import sys; sys.path.insert(0, "drugdis/decomposition")
from decompositions import AdditiveProjection
p = AdditiveProjection(frame, ["SMILES", "Sample_ID"])   # built on the observed support
My = p.residual(y)          # interaction component
Hy = y - My                 # additive component
```

The same operator, built once on a support, is applied to measured and predicted
responses. It is computed by alternating projections to a tolerance of 1e-10.

### 4. Training M0, M3 and M4

```bash
bash scripts/train_m0_m4_lclo.sh    # held-out cell lines, 5 seeds
bash scripts/train_m0_m4_lso.sh     # held-out compounds, 5 seeds
bash scripts/train_m3.sh            # M3 under both regimes, 5 seeds
bash scripts/run_benchmark.sh       # representation benchmark (M0 with each representation, 3 seeds)
bash scripts/decoder_matrix.sh && bash scripts/decoder_penalty_sweep.sh && bash scripts/decoder_confirm.sh
```

All arms use the trainer defaults (batch size 2,048, 30 epochs, learning rate
1e-4, dropout 0.4, weight decay 1e-5) and are reported at `best_valmse.pth`, the
epoch with the lowest validation total prediction error.

### 5. The canonical tables

```bash
bash scripts/reproduce_tables.sh
```

This rebuilds every table below from `$DRUGDIS_DATA` and the runs in
`$DRUGDIS_RUNS`, in dependency order, runs the numerical consistency checks, and
compares each table with `results/tables/TABLES.json` by SHA-256. The builders
check that each run was trained with the config file and manifest directory they
are given; `DRUGDIS_CONFIG` and `DRUGDIS_MANIFESTS` point them at another copy of
the frozen config and manifests.

| table | contents | built by | used in |
|---|---|---|---|
| T01 | benchmark dataset definition and checksums | `tables/build_tables.py` | Fig. 1b; Supplementary Table 1 |
| T02 | variance decomposition of the benchmark dataset and each test set | `tables/build_tables.py` | Fig. 1c,e, 4b; Supplementary Table 2 |
| T03 | cross-assay reproducibility, GDSC1 vs GDSC2 | `tables/rebuild_measurement_bootstrap_v2.py` | Fig. 2e, 4e; Supplementary Table 6 |
| T04 | representation benchmark, every run | `tables/build_tables_bench.py` | source of T05, T06 |
| T05, T06 | representation benchmark profiles, LCLO and LSO | `tables/build_tables_bench.py` | Fig. 3a–d,h; Supplementary Tables 8, 9 |
| T07 | M0 and M4 component-wise recovery profiles | `tables/build_tables.py` | Fig. 1d,f, 4c; Supplementary Table 20 |
| T08 | zero-shot organoid evaluation of M0 and M4 | `tables/build_tables.py` | Fig. 4c,d,h, 5b,c; Supplementary Tables 21, 22 |
| T09 | ranking agreement across decoders | `tables/build_tables.py` | Fig. 3f; Supplementary Table 12 |
| T10 | component-wise error geometry | `tables/rebuild_valmse_identity_tables_v1.py` | Fig. 4a,d–f; Supplementary Tables 13, 14 |
| T12 | ranking resolution | `tables/separability.py` | Fig. 3a–e; Supplementary Table 10 |
| T13 | calibrated null predictors | `tables/rebuild_valmse_identity_tables_v1.py` | Fig. 2d; Supplementary Table 5 |
| T15 | organoid input-compatibility check | `tables/build_T15.py` | Fig. 4g; Supplementary Table 15 |
| T16 | decoder profiles | `tables/build_T16.py` | Fig. 3g; Supplementary Table 11 |
| T17 | distributions of the response and its components | `tables/build_T17.py` | Fig. 1e |
| T18 | interaction R² per run | `tables/build_interaction_r2_table.py` | Fig. 5d; Supplementary Table 18 |
| T19 | published evaluation views under prespecified perturbations | `tables/run_nm_validation_packages_ab.py` | Fig. 2b,c,f; Supplementary Table 4 |
| T20 | per-resource sensitivity | `tables/run_nm_validation_packages_ab.py` | Fig. 2f; Supplementary Table 7 |
| T21, T22 | GDSC1 selection and independent GDSC2 test | `tables/run_nm_validation_packages_ab.py` | Fig. 5e–h; Supplementary Tables 19, 17 |
| T23 | matched M0/M3/M4 comparison | `tables/audit_current_substrate_m3.py` | Fig. 5a–d; Supplementary Table 16 |
| T24 | zero-shot organoid evaluation of M3 | `organoid/evaluate_current_m3_pdo.py` | Fig. 5b–d; Supplementary Table 22 |
| T25 | organoid panel counts | `tables/build_T25_pdo_panel_counts.py` | Fig. 1b |
| T26 | decomposition within each response resource | `tables/build_T26_resource_decomposition.py` | Supplementary Table 3 |

Figure and table numbers refer to the PDFs in `paper/`.

## Verification of this release

Checked on the analysis host on 2026-09-29, with this code, the processed data
and the reported training runs:

- **Split manifests.** `scripts/make_manifests.sh` rebuilds all ten manifests byte
  for byte, and the input, panel, manifest and generator hashes in
  `manifests/MANIFEST.json` equal the frozen record. `verify_frozen.py` passes 34
  of 34 checksums.
- **Tables.** `scripts/reproduce_tables.sh` rebuilds 23 of the 24 canonical tables
  byte for byte. T10 is identical in every value; its `prediction_sha256` column
  records the SHA-256 of each gzip prediction export, and gzip stores the write
  time, so re-exported predictions give new hashes. The 20 re-exported prediction
  files are identical to the originals after decompression, and T10 built from the
  original exports is byte-identical. The numerical consistency checks pass 57 of
  57.
- **Training.** One epoch of M0 and M4 (held-out cell lines, seed 3407) and of M3
  (held-out compounds, seed 3407) with `drugdis/models/train_2a2.py` reproduces the
  first epoch of the reported runs exactly, in every recorded quantity. Full
  retraining was not repeated.
- **Processed inputs.** The data scripts rebuild `master_table.parquet`, every
  `omics_mrna_raw` table, the baseline gene list and matrix and
  `gdsc_sensitivity_data.parquet` with identical contents, and the transcriptomic
  input matrix byte for byte. `drug_response.parquet` and the ECFP4 fingerprints
  are identical except for two NCI60 organotin compounds whose five-valent `[Sn-]`
  SMILES RDKit 2023.09.4 rejects (109 response rows, 2 fingerprints); neither
  compound is in the benchmark dataset.
- **Unit tests.** `pytest` passes 30 of 30.

## Names in the code

The code predates the manuscript's final vocabulary. Field and file names were
kept so that every table, run directory and checksum record stays traceable; this
is how they map to the manuscript's terms.

| in the code | in the manuscript |
|---|---|
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

## Not included

Model checkpoints and prediction exports are not distributed; `scripts/` retrains
them, and `scripts/reproduce_tables.sh` rebuilds every table from them. The
figure-drawing code is not part of this release.

## License

MIT, see [LICENSE](LICENSE).

## Citation

```bibtex
@article{li2026drugdis,
  title  = {DrugDis: Disentangling general and context-specific effects in drug-response prediction},
  author = {Li, Bo and Liu, Chengliang and Peng, Yuzhong and Zhang, Bob and Wang, Qing and
            Zeng, Pinxian and Li, Mengran and Huang, Shenghui and Deng, Chuxia and Zhang, Yang},
  year   = {2026}
}
```

<a id="top"></a>

<div align="center">

<h1>DrugDis</h1>
<h3>Disentangling general and context-specific effects in drug-response prediction</h3>

<p>
  <img alt="Drug response" src="https://img.shields.io/badge/scope-drug%20response-7B61FF">
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-2ea44f"></a>
  <a href="manuscript/DrugDis_manuscript.pdf"><img alt="Manuscript PDF" src="https://img.shields.io/badge/manuscript-PDF-B31B1B?logo=adobeacrobatreader&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/Boom5426/DrugDis"><img alt="Hugging Face dataset" src="https://img.shields.io/badge/data-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black"></a>
  <a href="https://boom5426.github.io/DrugDis/"><img alt="Project website" src="https://img.shields.io/badge/project-website-1e1e1e"></a>
</p>

<p><strong>Evaluate additive effects and drug–sample interactions separately.</strong></p>

<p>
  <a href="https://boom5426.github.io/DrugDis/">🌐 Project website</a> ·
  <a href="#quick-start">🚀 Quick start</a> ·
  <a href="https://huggingface.co/datasets/Boom5426/DrugDis">🤗 Data</a> ·
  <a href="#reproduce">🧪 Reproduce</a> ·
  <a href="manuscript/DrugDis_manuscript.pdf">📄 Paper</a> ·
  <a href="CITATION.cff">📚 Cite</a>
</p>

</div>

**DrugDis** is a component-resolved framework for evaluating drug-response prediction beyond aggregate accuracy. It applies an exact orthogonal decomposition to measured and predicted responses on the same observed drug–sample pairs, separating **additive effects** (drug and sample marginals) from **drug–sample interactions**, then evaluates interaction direction, amplitude and prediction error separately.

<table align="center" width="100%">
  <tr>
    <td align="center" width="25%"><h3>3,141,680</h3></td>
    <td align="center" width="25%"><h3>986</h3></td>
    <td align="center" width="25%"><h3>54,180</h3></td>
    <td align="center" width="25%"><h3>11</h3></td>
  </tr>
  <tr>
    <td align="center"><sub>drug–sample pairs</sub></td>
    <td align="center"><sub>cancer cell lines</sub></td>
    <td align="center"><sub>compounds</sub></td>
    <td align="center"><sub>response resources</sub></td>
  </tr>
</table>

<p align="center">
  <a href="assets/fig1.pdf"><img src="assets/fig1_overview.png" alt="Figure 1a-c: the total response is split into an additive component and a drug-sample interaction component; the benchmark dataset integrates 11 response resources on a CCLE-derived transcriptomic input; the additive component carries 75.8% of response variance on 1.76% of the degrees of freedom." width="920"></a>
  <br>
  <sub>DrugDis overview · Figure 1a–c · <a href="assets/fig1.pdf">Open vector figure ↗</a></sub>
</p>

### ✨ Why DrugDis?

Across the pooled benchmark, additive effects account for **75.8%** of response variance. The same dominance persists when the five largest response resources are decomposed separately (**59.0–82.5%**) and when NCI60 is excluded (**66.5%**), showing that the result is not driven only by the largest screen ([resource-wise decomposition](results/tables/T26_resource_decomposition.csv)). A predictor can therefore score well on the total response by reproducing drug and sample marginals: under held-out cell lines, a model with an aggregate correlation of **0.86** recovered interactions at a correlation of only **0.32**. DrugDis reports each component on the same observed pairs, with cross-assay reproducibility as an empirical reference where repeated measurements exist. See the [paper](manuscript/DrugDis_manuscript.pdf) for the evaluated regimes and their scope.

<table>
  <tr>
    <td valign="top" width="33%">
      <b>🧮 Decompose your response data</b><br><br>
      Split a drug–sample response table into additive and interaction components on its observed support.<br><br>
      <a href="#decompose">Decomposition →</a>
    </td>
    <td valign="top" width="33%">
      <b>🎯 Evaluate your predictions</b><br><br>
      Score direction, amplitude and error of each component, with an exact attribution of squared error.<br><br>
      <a href="#evaluate">Component-wise evaluation →</a>
    </td>
    <td valign="top" width="33%">
      <b>🧬 Explore the benchmark</b><br><br>
      Processed DROMA inputs, prespecified splits, organoid cohorts and the paper's result tables.<br><br>
      <a href="#benchmark">Benchmark dataset →</a>
    </td>
  </tr>
</table>

<a id="quick-start"></a>

## 🚀 Quick start

**Python 3.11+ · NumPy, pandas and SciPy for the decomposition and the evaluation API.** Run the bundled demo without downloading any data:

```bash
git clone https://github.com/Boom5426/DrugDis.git
cd DrugDis
python -m pip install -e .
python examples/decompose_demo.py
```

> [!NOTE]
> The demo builds a **synthetic table of 80 compounds × 120 samples with 30% of pairs observed** and scores two hypothetical predictors. It checks the software interface; **its numbers are not results from the benchmark**.

<details>
<summary><b>Environment setup and optional dependencies</b></summary>

A fresh environment is recommended:

```bash
conda create -n drugdis python=3.11 && conda activate drugdis
python -m pip install -e .
```

| Workflow | Install |
| :--- | :--- |
| Decomposition and evaluation API | `python -m pip install -e .` |
| Reproduce the paper (training, table builders, data construction) | `python -m pip install -e ".[paper]"` or `pip install -r requirements.txt` |
| Tests | `python -m pip install -e ".[dev]"` |

The `paper` extra pins the versions the reported analyses ran with: NumPy 1.26.4, pandas 1.5.3, SciPy 1.15.1, scikit-learn 1.5.2, PyArrow 18.0.0, PyTorch 2.4.1 (CUDA 11.8 build) and RDKit 2023.09.4. Select a PyTorch build for your hardware.

</details>

<a id="use-drugdis"></a>

## 🧭 Use DrugDis

<a id="decompose"></a>

### 🧮 Decompose your response data

Provide one row per observed drug–sample pair:

```python
import pandas as pd
from drugdis.evaluate import decompose

frame = pd.read_parquet("responses.parquet")    # columns SMILES, Sample_ID, Sensitivity
additive, interaction, summary = decompose(frame)
print(f"additive {summary['share_additive_pct']:.1f}% · interaction {summary['share_interaction_pct']:.1f}%")
```

Other column names are passed explicitly, e.g. `decompose(frame, response="auc", drug="drug_id", sample="cell_line")`. Repeated measurements of a pair must be averaged first. The summary also reports the support geometry: the number of connected components and the dimension of the additive subspace, `n_drugs + n_samples - components`.

> [!TIP]
> **The components are defined on the observed pairs.** The interaction component is the deviation from the best additive fit to those pairs, so measuring different compounds, samples or pairs changes it. Read it as a property of the observed support, not as an intrinsic biological interaction.

<a id="evaluate"></a>

### 🎯 Evaluate your predictions

For measured responses and your model's predictions on the same test pairs:

```python
from drugdis.evaluate import component_profile

scores = component_profile(test_frame, y="Sensitivity", yhat="prediction")
print(f"total {scores['rawPCC']:.3f} · additive {scores['sharedPCC']:.3f} · interaction {scores['intPCC']:.3f}")
```

Measured and predicted responses are decomposed by the same operator, built on the test pairs. The output keeps the questions apart:

| Output | Question answered |
| :--- | :--- |
| **Total-response correlation** (`rawPCC`) | Does the prediction track the measured response overall? |
| **Additive-component correlation** (`sharedPCC`) | Does it recover the drug and sample marginals? |
| **Interaction-component correlation** (`intPCC`) | Does it recover which pairs deviate from the additive fit, and in which direction? |
| **Interaction amplitude and R²** (`A_int`, `R2_interaction`) | Is the predicted interaction as large as the measured one? |
| **Additive- and interaction-component error** (`E_shared`, `E_interaction`) | Where does the squared prediction error lie? The two sum exactly to the total (`MSE_raw`). |

<details>
<summary><b>Interpretation notes</b></summary>

The correlation of a component measures direction and is blind to scale; `A_int` measures scale; `R2_interaction = 2·A_int·intPCC − A_int²` combines them. Evaluate each model on the pairs of its own test set: components computed on different supports are not comparable. Where repeated measurements of the same pairs exist, their agreement per component (cross-assay reproducibility, table T03) is an empirical reference for how much of each component a measurement reproduces; it is not an upper bound on model performance. The Methods of the [manuscript](manuscript/DrugDis_manuscript.pdf) give the definitions.

</details>

<a id="benchmark"></a>

## 🧬 The benchmark dataset

| Domain | Resources | Samples | Compounds | Pairs |
| :--- | :--- | ---: | ---: | ---: |
| **Cancer cell lines** | NCI60, PRISM, CTRP1, CTRP2, GDSC1, GDSC2, CCLE, GRAY, gCSI, FIMM, UHNBreast | 986 | 54,180 | 3,141,680 |
| **Patient-derived organoids** (zero-shot) | UMPDO1, UMPDO2, UMPDO3, HKUPDO, LICOB | 173 | 145 | 10,010 |

Response measurements come from the DROMA collection of harmonized preclinical drug-response and omics data; the transcriptomic input of every cell line is its CCLE expression profile over 15,961 genes. Models are evaluated under **held-out cell lines (LCLO)** and **held-out compounds (LSO)**, five prespecified seeds each, and zero-shot on the organoid cohorts.

**[Explore the Hugging Face dataset ↗](https://huggingface.co/datasets/Boom5426/DrugDis)** &nbsp;·&nbsp; [Split manifests](manifests/) &nbsp;·&nbsp; [Benchmark definition](configs/substrate_config.frozen.json) &nbsp;·&nbsp; [Result tables](results/tables/README.md)

GitHub holds the code, the split manifests and the canonical result tables. Hugging Face hosts the processed inputs: the DROMA database, the processed response and sample tables, expression, the transcriptomic input matrix, ECFP4 and eleven pretrained molecular representations, and eleven pretrained transcriptomic embeddings. The [dataset card](https://huggingface.co/datasets/Boom5426/DrugDis) lists every file and how it was built.

<details>
<summary><b>Download the dataset</b></summary>

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Boom5426/DrugDis",
    repo_type="dataset",
    local_dir="data_external/DrugDis",
)
```

Every analysis script finds its inputs and outputs through three environment variables ([`drugdis/paths.py`](drugdis/paths.py)):

```bash
export DRUGDIS_DATA=$PWD/data_external/DrugDis   # the dataset; required
export DRUGDIS_RUNS=$PWD/runs                    # training outputs (default ./runs)
export DRUGDIS_WORK=$PWD/work                    # analysis outputs and rebuilt tables (default ./work)
```

The DROMA collection is archived at [doi:10.5281/zenodo.18503188](https://doi.org/10.5281/zenodo.18503188), with its input-data provenance at [doi:10.5281/zenodo.17498421](https://doi.org/10.5281/zenodo.17498421).

</details>

<a id="reproduce"></a>

## 🔁 Reproduce the paper

Start at the level you need:

| Goal | Entry point | Inputs |
| :--- | :--- | :--- |
| **Try the software** | [Bundled demo](examples/decompose_demo.py) | Repository only |
| **Inspect paper results** | [Result-table index](results/tables/README.md) | Committed canonical tables |
| **Rerun analyses** | [Launchers](scripts/) and the commands below | Hugging Face dataset; one CUDA GPU for training |

For repository tests:

```bash
python -m pip install -e ".[paper,dev]"
pytest
```

<details>
<summary><b>Paper-level commands</b></summary>

```bash
# 1. Processed inputs (optional: the dataset already contains them)
python drugdis/data/build_processed_tables.py --out "$DRUGDIS_DATA"
python drugdis/data/build_gdsc_sensitivity.py --out "$DRUGDIS_DATA/gdsc_sensitivity_data.parquet"
python drugdis/data/encode_ecfp4.py --out "$DRUGDIS_DATA/Molecule_Embeddings/ECFP4_emb2048.pickle"
python drugdis/data/build_ccle_substrate.py
python drugdis/data/align_all_gene_reps.py

# 2. Benchmark dataset and split manifests
python drugdis/splits/verify_frozen.py            # inputs and manifests against recorded checksums
bash scripts/make_manifests.sh                    # rebuild the manifests into $DRUGDIS_WORK/manifests

# 3. Training (one M0 run takes about seven minutes on an RTX 4090)
bash scripts/train_m0_m4_lclo.sh                  # M0 and M4, held-out cell lines, 5 seeds
bash scripts/train_m0_m4_lso.sh                   # M0 and M4, held-out compounds, 5 seeds
bash scripts/train_m3.sh                          # M3, both regimes, 5 seeds
bash scripts/run_benchmark.sh                     # representation benchmark, 3 seeds
bash scripts/decoder_matrix.sh && bash scripts/decoder_penalty_sweep.sh && bash scripts/decoder_confirm.sh

# 4. Every canonical table, compared with results/tables by SHA-256
bash scripts/reproduce_tables.sh
```

The benchmark dataset is defined by [`configs/substrate_config.frozen.json`](configs/substrate_config.frozen.json): a pair is eligible if its sample is a cell line with a CCLE expression profile, is not in the excluded Tavor project, and its compound has an ECFP4 fingerprint; repeated measurements of a pair are averaged. `build_processed_tables.py` applies the overlap rule stated in Methods. All arms use the trainer defaults (batch size 2,048, 30 epochs, learning rate 1e-4, dropout 0.4, weight decay 1e-5) and are reported at `best_valmse.pth`, the epoch with the lowest validation total prediction error. [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) describes the layout and maps code names to the manuscript's terms.

</details>

<details>
<summary><b>Reproduction checks of this release</b></summary>

Run on the analysis host on 2026-09-29, with this code, the processed data and the reported training runs:

- **Split manifests.** `scripts/make_manifests.sh` rebuilds all ten manifests byte for byte; `verify_frozen.py` passes 34 of 34 checksums.
- **Tables.** `scripts/reproduce_tables.sh` rebuilds 23 of the 24 canonical tables byte for byte. T10 is identical in every value; its `prediction_sha256` column records the SHA-256 of each gzip prediction export, and gzip stores the write time, so re-exported predictions (identical after decompression) give new hashes. Built from the original exports, T10 is byte-identical. The numerical consistency checks pass 57 of 57.
- **Training.** One epoch of M0 and M4 (held-out cell lines, seed 3407) and of M3 (held-out compounds, seed 3407) reproduces the first epoch of the reported runs exactly, in every recorded quantity. Full retraining was not repeated.
- **Processed inputs.** The data scripts rebuild every processed input with identical contents, except for two NCI60 organotin compounds whose five-valent `[Sn-]` SMILES RDKit 2023.09.4 rejects (109 response rows, 2 fingerprints); neither compound is in the benchmark dataset.
- **Unit tests** pass.

</details>

> [!IMPORTANT]
> The repository provides the code and the canonical derived tables underlying the reported quantitative results, together with the manuscript and Supplementary Information. Figure-drawing code, model checkpoints and prediction exports are not part of the release; the overview above is a static excerpt of manuscript Figure 1.

<a id="citation"></a>

## 📚 Citation

This repository accompanies **[DrugDis: Disentangling general and context-specific effects in drug-response prediction](manuscript/DrugDis_manuscript.pdf)** ([Supplementary Information](manuscript/DrugDis_SI.pdf)). Please cite the manuscript when using DrugDis or its benchmark dataset; machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

<details>
<summary><b>BibTeX</b></summary>

```bibtex
@article{Li2026DrugDis,
  title  = {DrugDis: Disentangling general and context-specific effects in drug-response prediction},
  author = {Li, Bo and Liu, Chengliang and Peng, Yuzhong and Zhang, Bob and Wang, Qing and
            Zeng, Pinxian and Li, Mengran and Huang, Shenghui and Deng, Chuxia and Zhang, Yang},
  year   = {2026}
}
```

</details>

**License & support.** The code is released under the [MIT License](LICENSE); the original datasets retain their source terms. For questions or reproducible bug reports, please [open an issue](https://github.com/Boom5426/DrugDis/issues).

---

<p align="center">
  <b>Separate the additive. Score the interaction.</b><br>
  <sub><a href="#top">Back to top ↑</a></sub>
</p>

<a id="top"></a>

<div align="center">

<h1>DrugDis</h1>
<h3>解耦药物反应预测中的通用效应与情境特异性效应</h3>

<p>
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/lang-English-0969DA"></a>
  <a href="README_zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/lang-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-0969DA"></a>
  <img alt="Drug response" src="https://img.shields.io/badge/scope-drug%20response-7B61FF">
  <img alt="Pharmacogenomics" src="https://img.shields.io/badge/scope-pharmacogenomics-7B61FF">
  <img alt="Model evaluation" src="https://img.shields.io/badge/scope-model%20evaluation-7B61FF">
  <img alt="Cancer cell lines" src="https://img.shields.io/badge/scope-cancer%20cell%20lines-7B61FF">
  <img alt="Organoids" src="https://img.shields.io/badge/scope-organoids-7B61FF">
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-2ea44f"></a>
</p>

<p>
  <a href="https://boom5426.github.io/DrugDis/">🌐 项目主页</a> ·
  <a href="#quick-start">🚀 快速开始</a> ·
  <a href="https://huggingface.co/datasets/Boom5426/DrugDis">🤗 数据</a> ·
  <a href="#reproduce">🧪 复现实验</a> ·
  <a href="manuscript/DrugDis_manuscript.pdf">📄 论文</a> ·
  <a href="CITATION.cff">📚 引用</a>
</p>

</div>

**DrugDis** 在同一组已观测的药物–样本对上，把实测和预测的药物反应分解为**加性效应**（药物与样本层面的整体趋势）和**药物–样本交互**，并分别评估方向、幅度和误差。

<table align="center" width="100%">
  <tr>
    <td align="center" width="25%"><h3>3,141,680</h3></td>
    <td align="center" width="25%"><h3>986</h3></td>
    <td align="center" width="25%"><h3>54,180</h3></td>
    <td align="center" width="25%"><h3>11</h3></td>
  </tr>
  <tr>
    <td align="center"><sub>药物–样本对</sub></td>
    <td align="center"><sub>癌细胞系</sub></td>
    <td align="center"><sub>化合物</sub></td>
    <td align="center"><sub>药物反应资源</sub></td>
  </tr>
</table>

<p align="center">
  <a href="assets/fig1.pdf"><img src="assets/fig1_overview.png" alt="DrugDis overview: observed drug-sample responses are decomposed into additive and interaction components and evaluated across held-out cell lines, held-out compounds and zero-shot organoid transfer." width="920"></a>
  <br>
  <sub>论文 Figure 1 · <a href="assets/fig1.pdf">矢量图 ↗</a></sub>
</p>

### ✨ 主要发现

- 在基准数据集上，加性效应解释了 **75.8%** 的响应方差；在五个最大的药物反应资源中分别为 59–83%（Fig. 1c）。
- 在留出细胞系设置下，总体响应相关性为 **0.86** 的模型，交互成分相关性仅为 **0.32**（Fig. 1d）。
- 在架构与参数量匹配的比较中，分量监督提高了留出化合物和零样本类器官中的交互方向恢复，但只有留出化合物设置下交互误差得到一致降低；类器官中的交互 R² 仍**低于 0**（Fig. 5b–d）。
- 不同评估标准会改变选出的模型，但都没有改善独立 GDSC2 测量上的 selectivity 误差（Fig. 5e–h）。

<a id="quick-start"></a>

## 🚀 快速开始

```bash
git clone https://github.com/Boom5426/DrugDis.git
cd DrugDis
conda create -n drugdis python=3.11 && conda activate drugdis   # or any Python 3.11+ environment
python -m pip install -e .
python examples/decompose_demo.py
```

示例在一个合成数据表上（80 个化合物 × 120 个样本，30% 的药物–样本对被观测到）评估三个假想预测器。其中的数值只用于演示软件，不是论文基准结果。

<details>
<summary><b>预期输出</b></summary>

```text
2,896 observed pairs, 80 compounds, 120 samples, 1 connected component(s)
additive component:     79.3% of response variance on 6.9% of the degrees of freedom
interaction component:  20.7% (corr with the additive component +4.6e-14)

                  marginals only + pair-level noise + half the interaction
rawPCC                     0.848              0.866                  0.946
sharedPCC                  0.952              0.990                  0.996
intPCC                 undefined             -0.022                  0.737
A_int                      0.000              0.356                  0.577
emits_interaction          False               True                   True
R2_interaction             0.000             -0.142                  0.518
MSE_raw                    0.396              0.346                  0.146
E_shared                   0.112              0.021                  0.009
E_interaction              0.285              0.325                  0.137

The first emits no interaction, so intPCC is undefined. The second emits
uninformative interaction and scores near zero. Only the third recovers it.
```

</details>

| 用途 | 安装命令 |
| :--- | :--- |
| 响应分解与评估 API | `python -m pip install -e .` |
| 复现论文（训练、结果表、数据构建） | `python -m pip install -e ".[paper]"` 或 `pip install -r requirements.txt` |
| 测试 | `python -m pip install -e ".[dev]"` |

`paper` extra 固定了论文分析使用的版本：NumPy 1.26.4、pandas 1.5.3、SciPy 1.15.1、scikit-learn 1.5.2、PyArrow 18.0.0、PyTorch 2.4.1 和 RDKit 2023.09.4。论文分析使用 PyTorch 的 CUDA 11.8 构建，请按本机硬件选择合适的构建。

<a id="system-requirements"></a>

### 系统要求

| | |
| :--- | :--- |
| **操作系统** | 论文分析：Ubuntu 22.04.5 LTS。示例和测试另在 Ubuntu 26.04 LTS 上运行过。其他系统未测试。 |
| **Python** | API：3.11 及以上，已在 Python 3.11 + `paper` 固定版本，以及 Python 3.13 + NumPy 2.5.4、pandas 3.1.0、SciPy 1.18.1 下测试。`paper` extra：仅支持 3.11，因为 pandas 1.5.3 没有更高版本的 wheel。 |
| **硬件** | API 和示例：任意 CPU。训练：一块 CUDA GPU；论文中的训练使用 RTX 4090，每次 M0 训练约 7 分钟。 |
| **安装时间** | 全新 Python 3.11 环境、不使用包缓存：核心包 11 秒；`paper` 和 `dev` extra 176 秒（5.6 GB）。取决于网络。 |
| **示例运行时间** | 不到 1 秒（台式机 CPU 上 0.4 秒）。 |

<a id="use-drugdis"></a>

## 🧭 使用 DrugDis

<a id="decompose"></a>

### 🧮 分解药物反应数据

```python
import pandas as pd
from drugdis.evaluate import decompose

frame = pd.read_parquet("responses.parquet")   # one row per observed pair: SMILES, Sample_ID, Sensitivity
additive, interaction, summary = decompose(frame)
print(f"additive {summary['share_additive_pct']:.1f}% · interaction {summary['share_interaction_pct']:.1f}%")
```

同一药物–样本对的重复测量需先取平均。列名不同时可显式指定，例如 `decompose(frame, response="auc", drug="drug_id", sample="cell_line")`。summary 还给出连通分量数和加性子空间维度 `n_drugs + n_samples - components`。分量由已观测的药物–样本对定义：改变化合物、样本或药物–样本对，分量也会改变。

<a id="evaluate"></a>

### 🎯 评估预测结果

```python
import pandas as pd
from drugdis.evaluate import component_profile

test_frame = pd.read_csv("test_predictions.csv")   # one row per test pair: SMILES, Sample_ID, Sensitivity, prediction
scores = component_profile(test_frame, y="Sensitivity", yhat="prediction")
print(f"total {scores['rawPCC']:.3f} · additive {scores['sharedPCC']:.3f} · interaction {scores['intPCC']:.3f}")
```

实测响应和预测响应由同一个基于测试药物–样本对构建的算子分解。

| 输出 | 回答的问题 |
| :--- | :--- |
| **总体响应相关性** (`rawPCC`) | 预测是否整体跟随实测响应变化？ |
| **加性成分相关性** (`sharedPCC`) | 是否恢复了药物与样本的边际趋势？ |
| **交互成分相关性** (`intPCC`) | 是否恢复了哪些药物–样本对偏离加性拟合，以及偏离方向？ |
| **交互幅度与 R²** (`A_int`, `R2_interaction`) | 预测交互的幅度是否与实测交互一致？ |
| **交互输出标记** (`emits_interaction`) | 预测中是否含有任何交互成分？ |
| **分量误差** (`E_shared`, `E_interaction`) | 平方误差落在哪个分量？二者精确相加得到 `MSE_raw`。 |

> [!NOTE]
> 不含交互成分的预测（例如常数，或药物项与样本项之和）的 `intPCC` 无定义。与论文 Methods 一致，此时 `component_profile` 返回 NaN 的 `intPCC`，`emits_interaction` 为 `False`（判据为 `A_int` ≤ 1e-8；结构上就是加性的模型请传入 `additive_by_construction=True`）。不含信息的交互（例如打乱后的预测）仍有定义，得分接近零。

<details>
<summary><b>解释说明</b></summary>

分量相关性衡量方向，`A_int` 衡量尺度，`R2_interaction = 2·A_int·intPCC − A_int²` 将二者结合；对不含交互的预测，它仍有定义且接近零。每个模型应在其自身测试集的药物–样本对上评估：不同药物–样本对集合上的分量是不同的量。若同一药物–样本对有两次独立测量，各分量上的测量一致性（cross-assay reproducibility，表 T03）可作为经验参照，模型的恢复程度可以达到或超过它。详细定义见[论文](manuscript/DrugDis_manuscript.pdf) Methods。

</details>

<a id="benchmark"></a>

## 🧬 基准数据与评估设置

药物反应来自 DROMA 中的 11 个资源，使用 DROMA 字段 `Sensitivity`，按原样使用。每个细胞系都使用同一套 CCLE 来源的 15,961 基因表达谱。

| 评估设置 | 样本 | 化合物 | 药物–样本对 | 用途 |
| :--- | ---: | ---: | ---: | :--- |
| **细胞系基准** | 986 个细胞系 | 54,180 | 3,141,680 | 留出细胞系与留出化合物评估 |
| **主要零样本类器官集合** | 100 个类器官 | 78 | 4,886 | 主要跨系统评估（UMPDO1–3） |
| **全部类器官队列** | 173 个类器官 | 145 | 10,010 | 敏感性分析；LICOB 与 HKUPDO 单独报告 |

类器官不属于基准数据集，只用于评估细胞系训练的 checkpoint 的零样本迁移，不做微调。UMPDO1 和 UMPDO2 满足全部三项输入兼容性标准，UMPDO3（属于评估前确定的主要集合）在表达水平标准上差 0.005，LICOB 和 HKUPDO 三项均不满足。每个留出轴有 5 组 split manifests；M0、M1、M2 的比较使用全部 5 个随机种子，表征基准使用前 3 个。代码中 M1 与 M2 的键名为 M3 与 M4（见[名称对照](PROJECT_STRUCTURE.md#names-in-the-code)）。

GitHub 提供代码、[split manifests](manifests/)、[基准定义](configs/substrate_config.frozen.json)和[结果表](results/tables/README.md)。[Hugging Face 数据集](https://huggingface.co/datasets/Boom5426/DrugDis)提供处理后的输入：DROMA 数据库、响应与样本表、CCLE 来源表达数据、ECFP4，以及 22 种预训练分子与转录组表征；其数据卡说明了每个文件。

<details>
<summary><b>下载数据集</b></summary>

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Boom5426/DrugDis",
    repo_type="dataset",
    local_dir="data_external/DrugDis",
)
```

所有分析脚本通过三个环境变量定位输入和输出（见 [`drugdis/paths.py`](drugdis/paths.py)）：

```bash
export DRUGDIS_DATA=$PWD/data_external/DrugDis   # the dataset; required
export DRUGDIS_RUNS=$PWD/runs                    # training outputs (default ./runs)
export DRUGDIS_WORK=$PWD/work                    # analysis outputs and rebuilt tables (default ./work)
```

DROMA 归档于 [doi:10.5281/zenodo.18503188](https://doi.org/10.5281/zenodo.18503188)，其输入数据来源记录见 [doi:10.5281/zenodo.17498421](https://doi.org/10.5281/zenodo.17498421)。

</details>

<a id="reproduce"></a>

## 🔁 复现论文

| 目标 | 入口 | 所需输入 |
| :--- | :--- | :--- |
| **体验软件接口** | [仓库示例](examples/decompose_demo.py) | 仅仓库 |
| **查看论文结果** | [结果表索引](results/tables/README.md) | 已提交的 canonical tables |
| **重新运行分析** | [启动脚本](scripts/)与下方命令 | Hugging Face 数据集；训练需要一块 CUDA GPU |

运行测试：`python -m pip install -e ".[paper,dev]"`，然后 `pytest`。

<details>
<summary><b>论文级命令</b></summary>

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
bash scripts/train_m0_m4_lclo.sh                  # M0 and M2 (code M4), held-out cell lines, 5 seeds
bash scripts/train_m0_m4_lso.sh                   # M0 and M2 (code M4), held-out compounds, 5 seeds
bash scripts/train_m3.sh                          # M1 (code M3), both regimes, 5 seeds
bash scripts/run_benchmark.sh                     # representation benchmark, 3 seeds
bash scripts/decoder_matrix.sh && bash scripts/decoder_penalty_sweep.sh && bash scripts/decoder_confirm.sh

# 4. Every canonical table, compared with results/tables by SHA-256
bash scripts/reproduce_tables.sh
```

[`configs/substrate_config.frozen.json`](configs/substrate_config.frozen.json) 定义基准数据集：样本是具有 CCLE 表达谱且不属于被排除的 Tavor project 的细胞系、化合物具有 ECFP4 fingerprint 的药物–样本对符合纳入条件；重复测量取平均，`build_processed_tables.py` 实现 Methods 中的 overlap rule。M0、M1、M2 与表征基准的训练参数为 batch size 2,048、30 epochs、learning rate 1e-4、dropout 0.4、weight decay 1e-5，并报告 `best_valmse.pth`，即验证误差最低的 epoch。decoder 分析使用论文给出的特征标准化方式和验证集选择的正则化强度。[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 说明了仓库结构。

</details>

<details>
<summary><b>当前版本的复现检查</b></summary>

以下检查于 2026-09-29 在分析服务器上完成，使用本版本代码、处理后的数据和论文报告的训练结果：

- **Split manifests。** `scripts/make_manifests.sh` 可逐字节重建全部 10 个 manifests；`verify_frozen.py` 通过 34/34 项 checksum 检查。
- **结果表。** `scripts/reproduce_tables.sh` 可逐字节重建 24 张 canonical tables 中的 23 张。T10 的所有数值一致，只有 `prediction_sha256` 列不同，因为 gzip 会记录每个预测导出文件的写入时间。用原始导出文件构建时，T10 逐字节一致。数值一致性检查通过 57/57 项。
- **训练。** M0、M2（留出细胞系，seed 3407）和 M1（留出化合物，seed 3407）各复现 1 个 epoch，与已报告运行的第一个 epoch 完全一致。未重新执行完整训练。
- **处理后的输入。** 数据脚本可完全一致地重建所有处理后的输入，只有两个 NCI60 有机锡化合物例外：其 `[Sn-]` SMILES 被 RDKit 2023.09.4 拒绝（109 行响应、2 个 fingerprint），两者都不在基准数据集中。
- **单元测试**通过。

</details>

> [!IMPORTANT]
> 已发布：代码、canonical result tables、主文和 Supplementary Information。未发布：绘图代码、模型 checkpoint 和 prediction exports。概览图是论文 Figure 1 的静态摘录。

<a id="citation"></a>

## 📚 引用

使用 DrugDis 或其基准数据集时，请引用[论文](manuscript/DrugDis_manuscript.pdf)（[Supplementary Information](manuscript/DrugDis_SI.pdf)）。引用元数据见 [`CITATION.cff`](CITATION.cff)。

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

**许可证。** 代码：[MIT License](LICENSE)。Hugging Face 上的数据：CC BY 4.0，但预训练模型生成的 embedding 沿用各自模型的条款；上游药物反应与组学资源也保留其各自的条款。详见[数据卡](https://huggingface.co/datasets/Boom5426/DrugDis)。问题与 bug 反馈请[提交 issue](https://github.com/Boom5426/DrugDis/issues)。

<p align="center"><sub><a href="#top">返回顶部 ↑</a></sub></p>

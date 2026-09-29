"""Where the DrugDis pipeline reads and writes.

No script in this repository carries a machine-specific path. Every input and
output location is resolved here from three environment variables.

DRUGDIS_DATA
    Processed inputs, laid out as the Hugging Face dataset Boom5426/DrugDis:
    droma.sqlite, drug_response.parquet, master_table.parquet,
    gdsc_sensitivity_data.parquet, omics_mrna_raw/, omics_baseline/,
    omics_baseline_frozen/, Molecule_Embeddings/, Gene_Embeddings/ and
    annotations/. Required by every step that reads data; there is no default.
DRUGDIS_RUNS
    Training and evaluation runs, one directory per run with its checkpoints,
    predictions and results.json. Default: <repository>/runs.
DRUGDIS_WORK
    Intermediate analysis outputs (JSON records, aligned embeddings, prediction
    exports) and the rebuilt canonical tables under tables/.
    Default: <repository>/work. Created on first use.

The substrate config (configs/substrate_config.frozen.json) stores its data
paths relative to DRUGDIS_DATA; `load_config` resolves them. An absolute path
in a config is kept as written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent            # drugdis/
REPO = CODE.parent
CONFIG = REPO / "configs" / "substrate_config.frozen.json"
MANIFESTS = REPO / "manifests"
CANONICAL_TABLES = REPO / "results" / "tables"

CONFIG_PATH_KEYS = ("transcriptome_path", "master_table_path", "drug_response_path",
                    "drug_features_path")


def data_dir() -> Path:
    value = os.environ.get("DRUGDIS_DATA")
    if not value:
        raise RuntimeError("DRUGDIS_DATA is not set: point it at the processed data "
                           "directory (the Hugging Face dataset Boom5426/DrugDis)")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"DRUGDIS_DATA is not a directory: {path}")
    return path


def runs_dir() -> Path:
    return Path(os.environ.get("DRUGDIS_RUNS") or REPO / "runs").expanduser().resolve()


def work_dir() -> Path:
    path = Path(os.environ.get("DRUGDIS_WORK") or REPO / "work").expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def tables_dir() -> Path:
    path = work_dir() / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_config(cfg: dict) -> dict:
    """Copy of a substrate config whose data paths are absolute."""
    out = dict(cfg)
    for key in CONFIG_PATH_KEYS:
        if key in out and not os.path.isabs(out[key]):
            out[key] = str(data_dir() / out[key])
    return out


def load_config(path) -> dict:
    with open(path) as fh:
        return resolve_config(json.load(fh))

#!/usr/bin/env python3
"""Evaluate frozen-valMSE LCLO M3 checkpoints zero-shot on the PDO panel."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DRUG = "SMILES"
SAMPLE = "Sample_ID"
SEEDS = tuple(range(3407, 3412))
COMPARABLE = ("UMPDO1", "UMPDO2", "UMPDO3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--drug-features", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    return (float(np.corrcoef(a, b)[0, 1])
            if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan"))


def profile(y: np.ndarray, yhat: np.ndarray, projection) -> dict:
    interaction_y = projection.residual(y)
    interaction_yhat = projection.residual(yhat)
    shared_y = y - interaction_y
    shared_yhat = yhat - interaction_yhat
    error = yhat - y
    interaction_error = projection.residual(error)
    shared_error = error - interaction_error
    a_int = float(interaction_yhat.std() / interaction_y.std()) if interaction_y.std() else float("nan")
    int_pcc = pcc(interaction_y, interaction_yhat)
    return {
        "n": int(len(y)), "rawPCC": pcc(y, yhat),
        "sharedPCC": pcc(shared_y, shared_yhat), "intPCC": int_pcc,
        "A_int": a_int,
        "A_shared": float(shared_yhat.std() / shared_y.std()) if shared_y.std() else float("nan"),
        "interaction_R2": float(2 * a_int * int_pcc - a_int**2),
        "MSE_raw": float(np.mean(error**2)),
        "E_shared": float(np.mean(shared_error**2)),
        "E_interaction": float(np.mean(interaction_error**2)),
    }


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    table_output = args.table.resolve()
    for path in (output, table_output):
        if path.exists():
            raise SystemExit(f"refusing to overwrite output: {path}")
    processed = args.processed_root.resolve()
    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code / "decomposition"))
    sys.path.insert(0, str(code / "models"))
    from decompositions import AdditiveProjection
    from models import CLIOFactorized

    gene_list = processed / "omics_baseline" / "baseline_gene_list.txt"
    master_table = processed / "master_table.parquet"
    response_table = processed / "drug_response.parquet"
    raw_expression = processed / "omics_mrna_raw"
    with gene_list.open() as handle:
        genes = pd.Index([line.strip() for line in handle if line.strip()])
    master = pd.read_parquet(master_table, engine="fastparquet")
    pdo_ids = set(master.index[master["Model_Type"] == "PDO"].astype(str))

    frames = []
    origin = {}
    coverage = {}
    expression_files = sorted(Path(path) for path in glob.glob(str(raw_expression / "*.parquet")))
    for path in expression_files:
        resource = path.stem
        data = pd.read_parquet(path, engine="fastparquet")
        data.index = data.index.astype(str)
        data = data[data.index.isin(pdo_ids)]
        if not len(data):
            continue
        columns = genes.intersection(data.columns)
        values = data[columns].astype(np.float32).to_numpy()
        coverage[resource] = {
            "samples": int(len(data)), "genes_present": int(len(columns)),
            "gene_coverage": float(len(columns) / len(genes)),
            "nan_entries_filled_zero": int(np.isnan(values).sum()),
            "source": str(path), "source_sha256": sha256_file(path),
        }
        aligned = pd.DataFrame(0.0, index=data.index, columns=genes, dtype=np.float32)
        aligned.loc[:, columns] = np.nan_to_num(values, nan=0.0)
        frames.append(aligned)
        origin.update({sample: resource for sample in data.index})
    expression = pd.concat(frames)
    expression = expression[~expression.index.duplicated(keep="first")]
    if np.isnan(expression.to_numpy(np.float32)).any():
        raise AssertionError("NaN survived expression alignment")

    response = pd.read_parquet(response_table, engine="fastparquet")
    response = response.groupby([DRUG, SAMPLE])["Sensitivity"].mean().reset_index()
    with args.drug_features.resolve().open("rb") as handle:
        drug_features = pickle.load(handle)
    panel = response[
        response[DRUG].isin(drug_features.keys())
        & response[SAMPLE].astype(str).isin(set(expression.index))
    ].reset_index(drop=True)
    panel["resource"] = panel[SAMPLE].astype(str).map(origin)
    groups = {"ALL": panel.index.to_numpy()}
    groups.update({str(resource): panel.index[panel["resource"] == resource].to_numpy()
                   for resource in sorted(panel["resource"].unique())})
    groups["COMPARABLE(UMPDO1+2+3)"] = panel.index[
        panel["resource"].isin(COMPARABLE)].to_numpy()
    projections = {name: AdditiveProjection(panel.loc[index], [DRUG, SAMPLE])
                   for name, index in groups.items() if len(index) >= 50}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    used_drugs = pd.Index(pd.unique(panel[DRUG]))
    used_samples = pd.Index(pd.unique(panel[SAMPLE].astype(str)))
    drug_tensor = torch.as_tensor(np.stack([
        np.asarray(drug_features[key], np.float32) for key in used_drugs]), device=device)
    gene_tensor = torch.as_tensor(expression.loc[used_samples].to_numpy(np.float32), device=device)
    drug_index = torch.as_tensor(pd.Series(np.arange(len(used_drugs)), index=used_drugs)
                                 .loc[panel[DRUG]].to_numpy(), dtype=torch.long, device=device)
    sample_index = torch.as_tensor(pd.Series(np.arange(len(used_samples)), index=used_samples)
                                   .loc[panel[SAMPLE].astype(str)].to_numpy(), dtype=torch.long,
                                   device=device)
    y = panel["Sensitivity"].to_numpy(np.float64)

    result = {
        "schema_version": "rise.current_m3_pdo.v1",
        "arm": "M3", "checkpoint_rule": "minimum validation MSE_raw",
        "panel": {name: int(len(index)) for name, index in groups.items()},
        "coverage": coverage, "runs": {}, "sources": {
            "gene_list": {"file": str(gene_list), "sha256": sha256_file(gene_list)},
            "master_table": {"file": str(master_table), "sha256": sha256_file(master_table)},
            "response_table": {"file": str(response_table), "sha256": sha256_file(response_table)},
            "drug_features": {"file": str(args.drug_features.resolve()),
                              "sha256": sha256_file(args.drug_features.resolve())},
        },
    }
    table_rows = []
    for seed in SEEDS:
        run_dir = (args.checkpoint_root.resolve() / "ECFP4__baseline" / "LCLO" /
                   f"M3_seed{seed}")
        checkpoint = run_dir / "best_valmse.pth"
        results_json = run_dir / "results.json"
        if not checkpoint.exists() or not results_json.exists():
            raise FileNotFoundError((checkpoint, results_json))
        with results_json.open() as handle:
            training_result = json.load(handle)
        if training_result["model"] != "M3" or training_result["args"]["split_type"] != "LCLO":
            raise AssertionError(results_json)
        network = CLIOFactorized(2048, len(genes), hidden_dim1=1024,
                                 shared_hidden=512, dropout_p=0.4)
        network.load_state_dict(load_state_dict(checkpoint, device))
        network = network.to(device).eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(panel), args.batch_size):
                stop = min(start + args.batch_size, len(panel))
                shared, interaction = network(
                    drug_tensor[drug_index[start:stop]], gene_tensor[sample_index[start:stop]])
                predictions.append((shared + interaction).squeeze(-1).float().cpu())
        yhat = torch.cat(predictions).numpy().astype(np.float64)
        record = {
            name: profile(y[index], yhat[index], projections[name])
            for name, index in groups.items() if name in projections
        }
        record["pred_min"] = float(yhat.min())
        record["pred_max"] = float(yhat.max())
        record["source"] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "results": str(results_json), "results_sha256": sha256_file(results_json),
        }
        result["runs"][f"M3_seed{seed}"] = record
        for cohort, metrics in record.items():
            if not isinstance(metrics, dict) or "rawPCC" not in metrics:
                continue
            table_rows.append({
                "arm": "M3", "seed": seed, "cohort": cohort,
                "role": ("primary" if cohort == "COMPARABLE(UMPDO1+2+3)"
                         else "pooled_all" if cohort == "ALL" else "single_resource"),
                **metrics,
            })
    output.parent.mkdir(parents=True, exist_ok=False)
    if table_output.parent != output.parent:
        table_output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(table_rows)
    table.to_csv(table_output, index=False, mode="x")
    result["status"] = "pass"
    result["new_table_sha256"] = sha256_file(table_output)
    with output.open("x") as handle:
        json.dump(result, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output),
                      "table": str(table_output), "table_sha256": sha256_file(table_output),
                      "table_rows": len(table), "runs": len(result["runs"]),
                      "panel_rows": len(panel)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

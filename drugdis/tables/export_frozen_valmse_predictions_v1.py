#!/usr/bin/env python3
"""Export predictions from a run's frozen validation-MSE checkpoint.

The training script historically wrote generic ``test_predictions.csv`` files
only for the legacy-iePCC checkpoint.  This exporter is intentionally additive:
it refuses to overwrite files, validates the recomputed component profile
against ``results.json``, and writes provenance beside the compressed CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--substrate-config", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def component_profile(y: np.ndarray, yhat: np.ndarray, projection) -> dict:
    my = projection.residual(y)
    myhat = projection.residual(yhat)
    hy, hyhat = y - my, yhat - myhat
    error = yhat - y
    m_error = projection.residual(error)
    h_error = error - m_error
    return {
        "n": int(len(y)),
        "rawPCC": pcc(y, yhat),
        "sharedPCC": pcc(hy, hyhat),
        "intPCC": pcc(my, myhat),
        "A_shared": float(hyhat.std() / hy.std()) if hy.std() else float("nan"),
        "A_int": float(myhat.std() / my.std()) if my.std() else float("nan"),
        "MSE_raw": float(np.mean(error**2)),
        "E_shared": float(np.mean(h_error**2)),
        "E_interaction": float(np.mean(m_error**2)),
    }


def factorization_report(bhat: np.ndarray, zhat: np.ndarray, y: np.ndarray,
                         projection) -> dict:
    my = projection.residual(y)
    hy = y - my
    mz = projection.residual(zhat)
    hz = zhat - mz
    mb = projection.residual(bhat)
    z_centered = zhat - zhat.mean()
    hz_centered = hz - zhat.mean()
    return {
        "corr_bhat_Hy": pcc(bhat, hy),
        "corr_zhat_My": pcc(zhat, my),
        "leak_H_zhat": float(np.sum(hz**2) / np.sum(zhat**2)) if np.any(zhat) else float("nan"),
        "leak_H_zhat_centered": (
            float(np.sum(hz_centered**2) / np.sum(z_centered**2))
            if np.any(z_centered) else float("nan")
        ),
        "leak_M_bhat": float(np.sum(mb**2) / np.sum(bhat**2)) if np.any(bhat) else float("nan"),
        "sd_bhat": float(bhat.std()),
        "sd_zhat": float(zhat.std()),
        "mean_bhat": float(bhat.mean()),
        "mean_zhat": float(zhat.mean()),
    }


@torch.no_grad()
def forward_panel(model, store, panel, batch_size: int, factorized: bool) -> dict:
    model.eval()
    bhat_parts: list[torch.Tensor] = []
    zhat_parts: list[torch.Tensor] = []
    for start in range(0, panel.n, batch_size):
        index = torch.arange(start, min(start + batch_size, panel.n), device=panel.dev)
        drug, gene, _ = store.gather(panel, index)
        if factorized:
            bhat, zhat = model(drug, gene)
            bhat_parts.append(bhat.squeeze(-1).float().cpu())
            zhat_parts.append(zhat.squeeze(-1).float().cpu())
        else:
            zhat_parts.append(model(drug, gene).squeeze(-1).float().cpu())
    zhat = torch.cat(zhat_parts).numpy().astype(np.float64)
    if not factorized:
        return {"yhat": zhat}
    bhat = torch.cat(bhat_parts).numpy().astype(np.float64)
    return {"yhat": bhat + zhat, "bhat": bhat, "zhat": zhat}


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def check_profile(observed: dict, expected: dict, rtol: float, atol: float) -> None:
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise AssertionError(f"recomputed profile is missing keys: {missing}")
    failures = {}
    for key, expected_value in expected.items():
        observed_value = observed[key]
        if isinstance(expected_value, bool) or not isinstance(expected_value, (int, float)):
            if observed_value != expected_value:
                failures[key] = {"expected": expected_value, "observed": observed_value}
        elif math.isnan(float(expected_value)):
            if not math.isnan(float(observed_value)):
                failures[key] = {"expected": expected_value, "observed": observed_value}
        elif not np.isclose(float(observed_value), float(expected_value), rtol=rtol, atol=atol):
            failures[key] = {
                "expected": float(expected_value),
                "observed": float(observed_value),
                "abs_diff": abs(float(observed_value) - float(expected_value)),
            }
    if failures:
        raise AssertionError("checkpoint/profile mismatch:\n" + json.dumps(failures, indent=2))


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    substrate_config = args.substrate_config.resolve()
    manifest_dir = args.manifest_dir.resolve()
    output_file = args.output_file.resolve()
    metadata_file = Path(str(output_file) + ".meta.json")

    required = [run_dir, substrate_config, manifest_dir,
                run_dir / "results.json", run_dir / "best_valmse.pth"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))
    collisions = [str(path) for path in (output_file, metadata_file) if path.exists()]
    if collisions:
        raise SystemExit("refusing to overwrite existing outputs:\n" + "\n".join(collisions))

    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code / "decomposition"))
    sys.path.insert(0, str(code / "models"))          # models.py, data_fast.py, legacy utils.py
    from decompositions import AdditiveProjection
    import data_fast
    from models import CLIOFactorized, CLIOSingleHead

    results_path = run_dir / "results.json"
    checkpoint_path = run_dir / "best_valmse.pth"
    with results_path.open() as handle:
        results = json.load(handle)
    run_args = results["args"]
    model_name = str(results["model"])
    split_type = str(run_args["split_type"])
    seed = int(run_args["seed"])
    drug_fm = str(run_args["drug_fm"])
    batch_size = int(args.batch_size or run_args.get("batch_size", 2048))
    factorized = model_name in {"M3", "M4", "M5"}

    expected_name = f"{model_name}_seed{seed}"
    if run_dir.name != expected_name:
        raise AssertionError(f"run directory {run_dir.name!r} does not match {expected_name!r}")
    expected_config = run_args.get("substrate_config")
    expected_manifest_dir = run_args.get("manifest_dir")
    if expected_config and Path(expected_config).resolve() != substrate_config:
        raise AssertionError("explicit substrate config differs from the recorded run config")
    if expected_manifest_dir and Path(expected_manifest_dir).resolve() != manifest_dir:
        raise AssertionError("explicit manifest directory differs from the recorded run metadata")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    store, _, _, test = data_fast.build_frozen(
        str(substrate_config), str(manifest_dir), split_type, seed, drug_fm, device,
        drug_rep_path=run_args.get("drug_rep_path"),
        gene_rep_path=run_args.get("gene_rep_path"),
        allow_partial_drug_rep=bool(run_args.get("allow_partial_drug_rep", False)),
    )

    dropout = float(run_args.get("dropout", 0.4))
    hidden_dim1 = int(results.get("param_budget", {}).get("M2_joint_hidden1", 1024)) \
        if model_name == "M2" else 1024
    if factorized:
        model = CLIOFactorized(
            store.drug_dim, store.gene_dim, hidden_dim1=hidden_dim1,
            shared_hidden=512, dropout_p=dropout,
        )
    else:
        model = CLIOSingleHead(
            store.drug_dim, store.gene_dim, hidden_dim1=hidden_dim1, dropout_p=dropout,
        )
    model.load_state_dict(load_state_dict(checkpoint_path, device))
    model = model.to(device)

    output = forward_panel(model, store, test, batch_size, factorized)
    target = test.df["Sensitivity"].to_numpy(np.float64)
    projection = AdditiveProjection(test.df, ["SMILES", "Sample_ID"])
    observed = component_profile(target, output["yhat"], projection)
    if factorized:
        observed.update(factorization_report(output["bhat"], output["zhat"], target, projection))
    expected = results["frozen_valMSE__test"]
    check_profile(observed, expected, args.rtol, args.atol)

    frame = pd.DataFrame({
        "SMILES": test.df["SMILES"].astype(str),
        "Sample_ID": test.df["Sample_ID"].astype(str),
        "Target_AAC": target,
        "Predicted_AAC": output["yhat"],
    })
    if factorized:
        frame["bhat"] = output["bhat"]
        frame["zhat"] = output["zhat"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / f"{split_type}_seed{seed}.json"
    temp_output = None
    temp_metadata = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output_file.name + ".", suffix=".tmp", dir=output_file.parent, delete=False
        ) as handle:
            temp_output = Path(handle.name)
        frame.to_csv(temp_output, index=False, compression="gzip")
        metadata = {
            "schema_version": "rise.frozen_predictions.v1",
            "selection_rule": "minimum validation MSE_raw",
            "results_key": "frozen_valMSE__test",
            "model": model_name,
            "split_type": split_type,
            "seed": seed,
            "drug_fm": drug_fm,
            "n_rows": int(len(frame)),
            "best_valmse_epoch": int(results["best_valmse_epoch"]),
            "profile": observed,
            "source": {
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "results_json": str(results_path),
                "results_json_sha256": sha256_file(results_path),
                "substrate_config": str(substrate_config),
                "substrate_config_sha256": sha256_file(substrate_config),
                "manifest": str(manifest_file),
                "manifest_sha256": sha256_file(manifest_file),
            },
            "output": {
                "file": str(output_file),
                "sha256": sha256_file(temp_output),
            },
            "validation": {"rtol": args.rtol, "atol": args.atol, "status": "pass"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=metadata_file.name + ".", suffix=".tmp",
            dir=output_file.parent, delete=False
        ) as handle:
            temp_metadata = Path(handle.name)
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_output, output_file)
        temp_output = None
        os.replace(temp_metadata, metadata_file)
        temp_metadata = None
    finally:
        for path in (temp_output, temp_metadata):
            if path is not None and path.exists():
                path.unlink()

    print(json.dumps({
        "status": "pass",
        "output": str(output_file),
        "metadata": str(metadata_file),
        "n_rows": len(frame),
        "profile": observed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

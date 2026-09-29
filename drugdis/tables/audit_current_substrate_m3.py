#!/usr/bin/env python3
"""Audit and tabulate the frozen current-substrate M3 control.

M0--M4 is the whole-modification contrast.  M3--M4 holds architecture and
parameter count fixed and isolates the addition of component supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("LCLO", "LSO")
SEEDS = tuple(range(3407, 3412))
ARMS = ("M0", "M3", "M4")
PROFILE_KEYS = (
    "n", "rawPCC", "sharedPCC", "intPCC", "A_int", "A_shared",
    "MSE_raw", "E_shared", "E_interaction",
)
ARG_KEYS = (
    "drug_fm", "genomics_fm", "split_type", "test_split_ratio",
    "val_split_ratio", "batch_size", "n_epochs", "learning_rate", "dropout",
    "weight_decay", "seed", "substrate_config", "manifest_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m3-root", type=Path, required=True)
    parser.add_argument("--lclo-root", type=Path, required=True)
    parser.add_argument("--lso-root", type=Path, required=True)
    parser.add_argument("--substrate-config", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def load_run(root: Path, split: str, arm: str, seed: int) -> tuple[dict, Path]:
    path = root / "ECFP4__baseline" / split / f"{arm}_seed{seed}" / "results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        result = json.load(handle)
    return result, path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {output_dir}")
    expected_config = args.substrate_config.resolve()
    expected_manifests = args.manifest_dir.resolve()
    if not expected_config.exists() or not expected_manifests.exists():
        raise FileNotFoundError((expected_config, expected_manifests))

    roots = {"LCLO": args.lclo_root.resolve(), "LSO": args.lso_root.resolve()}
    rows = []
    sources = []
    comparisons = []
    for split in SPLITS:
        for seed in SEEDS:
            loaded = {}
            paths = {}
            for arm in ARMS:
                root = args.m3_root.resolve() if arm == "M3" else roots[split]
                loaded[arm], paths[arm] = load_run(root, split, arm, seed)
                result = loaded[arm]
                if result["model"] != arm:
                    raise AssertionError((paths[arm], result["model"], arm))
                run_args = result["args"]
                if Path(run_args["substrate_config"]).resolve() != expected_config:
                    raise AssertionError(f"substrate mismatch: {paths[arm]}")
                if Path(run_args["manifest_dir"]).resolve() != expected_manifests:
                    raise AssertionError(f"manifest directory mismatch: {paths[arm]}")
                manifest = expected_manifests / f"{split}_seed{seed}.json"
                profile = result["frozen_valMSE__test"]
                int_r2 = 2 * profile["A_int"] * profile["intPCC"] - profile["A_int"] ** 2
                row = {"split": split, "arm": arm, "seed": seed,
                       "rule": "frozen_valMSE", "best_epoch": result["best_valmse_epoch"],
                       "n_params": result["n_params"], "interaction_R2": int_r2}
                row.update({key: profile[key] for key in PROFILE_KEYS})
                rows.append(row)
                checkpoint = paths[arm].parent / "best_valmse.pth"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                sources.append({
                    "split": split, "seed": seed, "arm": arm,
                    "results": str(paths[arm]), "results_sha256": sha256_file(paths[arm]),
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                    "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                })

            reference_args = loaded["M4"]["args"]
            for arm in ARMS:
                for key in ARG_KEYS:
                    if loaded[arm]["args"].get(key) != reference_args.get(key):
                        raise AssertionError(f"argument mismatch {split}/{seed}/{arm}/{key}")
            n_values = {arm: loaded[arm]["frozen_valMSE__test"]["n"] for arm in ARMS}
            if len(set(n_values.values())) != 1:
                raise AssertionError(f"support-size mismatch {split}/{seed}: {n_values}")
            if loaded["M3"]["n_params"] != loaded["M4"]["n_params"]:
                raise AssertionError(f"M3/M4 parameter mismatch {split}/{seed}")
            for key in ("rawPCC", "sharedPCC", "intPCC", "A_int", "A_shared",
                        "MSE_raw", "E_shared", "E_interaction"):
                comparisons.append({
                    "split": split, "seed": seed, "metric": key,
                    "M3": loaded["M3"]["frozen_valMSE__test"][key],
                    "M4": loaded["M4"]["frozen_valMSE__test"][key],
                    "M4_minus_M3": (loaded["M4"]["frozen_valMSE__test"][key]
                                    - loaded["M3"]["frozen_valMSE__test"][key]),
                })

    table = pd.DataFrame(rows)
    comparison = pd.DataFrame(comparisons)
    summary = (comparison.groupby(["split", "metric"], as_index=False)
               .agg(mean_M3=("M3", "mean"), mean_M4=("M4", "mean"),
                    mean_M4_minus_M3=("M4_minus_M3", "mean"),
                    min_M4_minus_M3=("M4_minus_M3", "min"),
                    max_M4_minus_M3=("M4_minus_M3", "max")))
    output_dir.mkdir(parents=True, exist_ok=False)
    table_path = output_dir / "T23_current_substrate_m3.csv"
    comparison_path = output_dir / "M3_M4_paired_differences.csv"
    summary_path = output_dir / "M3_M4_summary.csv"
    table.to_csv(table_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    summary.to_csv(summary_path, index=False)
    audit = {
        "schema_version": "rise.current_substrate_m3.v1",
        "status": "pass",
        "new_table_sha256": sha256_file(table_path),
        "contract": {
            "M0_vs_M4": "whole modification",
            "M3_vs_M4": "same architecture and parameter count; component supervision differs",
            "selection_rule": "minimum validation MSE_raw",
        },
        "substrate_config": str(expected_config),
        "substrate_config_sha256": sha256_file(expected_config),
        "runs_checked": len(sources),
        "expected_runs": 30,
        "m3_m4_support_checks": 10,
        "m3_m4_parameter_checks": 10,
        "sources": sources,
        "outputs": {
            "T23": {"file": str(table_path), "rows": len(table),
                    "sha256": sha256_file(table_path)},
            "paired": {"file": str(comparison_path), "rows": len(comparison),
                       "sha256": sha256_file(comparison_path)},
            "summary": {"file": str(summary_path), "rows": len(summary),
                        "sha256": sha256_file(summary_path)},
        },
    }
    audit_path = output_dir / "AUDIT.json"
    with audit_path.open("x") as handle:
        json.dump(audit, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"status": "pass", "runs_checked": len(sources),
                      "table": str(table_path), "table_sha256": sha256_file(table_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

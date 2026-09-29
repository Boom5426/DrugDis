#!/usr/bin/env python3
"""Freeze the bounded Nature Methods validation protocol before outcome analysis.

The protocol records input identities and decision rules only.  It deliberately
does not open prediction or response tables, so no outcome can influence the
registered support, perturbations, selection rules, or endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--substrate-config", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--gdsc-raw", type=Path, required=True)
    parser.add_argument("--response-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stamp", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol: {output}")

    prediction_files = []
    for split in ("LCLO", "LSO"):
        for model in ("M0", "M4"):
            for seed in range(3407, 3412):
                path = (args.prediction_root / split / f"{model}_seed{seed}" /
                        "test_predictions_frozen_valmse.csv.gz").resolve()
                metadata = Path(str(path) + ".meta.json")
                for required in (path, metadata):
                    if not required.exists():
                        raise FileNotFoundError(required)
                prediction_files.append({
                    "split": split,
                    "model": model,
                    "seed": seed,
                    "file": str(path),
                    "sha256": sha256_file(path),
                    "metadata_file": str(metadata),
                    "metadata_sha256": sha256_file(metadata),
                })

    input_files = {}
    for name, path in {
        "substrate_config": args.substrate_config.resolve(),
        "gdsc_raw": args.gdsc_raw.resolve(),
        "response_table": args.response_table.resolve(),
    }.items():
        if not path.exists():
            raise FileNotFoundError(path)
        input_files[name] = {"file": str(path), "sha256": sha256_file(path)}
    for split in ("LCLO", "LSO"):
        for seed in range(3407, 3412):
            name = f"manifest_{split}_{seed}"
            path = (args.manifest_dir / f"{split}_seed{seed}.json").resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            input_files[name] = {"file": str(path), "sha256": sha256_file(path)}

    protocol = {
        "schema_version": "rise.nm_validation_protocol.v1",
        "frozen_at_utc": args.stamp or datetime.now(timezone.utc).isoformat(),
        "scope": {
            "models": ["M0", "M4"],
            "splits": ["LCLO", "LSO"],
            "seeds": [3407, 3408, 3409, 3410, 3411],
            "checkpoint_rule": "minimum validation MSE_raw",
            "note": "No model or loss tuning is part of Packages A or B.",
        },
        "package_a": {
            "definition_sources": {
                "specification_game": {
                    "doi": "10.1186/s13321-025-00972-y",
                    "code": "https://github.com/codicef/NxtDRP",
                    "inspected_commit": "98dae0ed02d3a57c80ab90622172e36cae75a3c0",
                },
                "dreval": {
                    "doi": "10.1038/s41467-026-72903-w",
                    "code": "https://github.com/daisybio/drevalpy",
                    "inspected_commit": "dec421bd39c8871f9e11097ec62016ad3dc08f32",
                },
            },
            "comparison_support": "identical rows for every metric within a run",
            "metrics": {
                "global": ["Pearson", "RMSE", "R2"],
                "specification_game": [
                    "unweighted mean within-drug Pearson over groups with n>=3",
                    "unweighted mean within-cell-line Pearson over groups with n>=3",
                ],
                "dreval": [
                    "NaiveMeanEffectsPredictor fitted only on the frozen training panel",
                    "normalized Pearson and R2 after subtracting that prediction from truth and model prediction",
                ],
                "rise": [
                    "shared and interaction Pearson",
                    "shared and interaction amplitude",
                    "orthogonal shared and interaction error",
                    "interaction R2 = 2*A_int*intPCC - A_int^2",
                ],
            },
            "behavioral_perturbations": {
                "original": "unaltered prediction",
                "drug_offset": "add deterministic centered drug-only offset with SD 0.5*SD(y)",
                "sample_offset": "add deterministic centered sample-only offset with SD 0.5*SD(y)",
                "interaction_half": "retain predicted additive component and multiply predicted interaction by 0.5",
                "interaction_double": "retain predicted additive component and multiply predicted interaction by 2.0",
            },
            "composition_sensitivity": [
                "each response resource with at least 1,000 matched prediction rows",
                "all resources after removing NCI60, with pair means recomputed",
            ],
            "interpretation_boundary": "resource rescoring is composition sensitivity, not external generalization",
        },
        "package_b": {
            "measurement_arms": {
                "selection": "GDSC1",
                "independent_test": "GDSC2",
                "row_boundary": 229420,
            },
            "eligibility": "matched drug-cell pairs present in both assay generations and in both M0/M4 test predictions",
            "contrast": "(y[d1,s1]-y[d1,s2])-(y[d2,s1]-y[d2,s2])",
            "rectangle_sampling": {
                "depends_on": "identifiers and support only; never responses or predictions",
                "seed": 20260911,
                "maximum_unique_rectangles_per_split_seed": 50000,
                "maximum_attempt_multiplier": 100,
            },
            "selection_rules_on_gdsc1": [
                "global_Pearson",
                "fixed_drug_Pearson",
                "fixed_cell_Pearson",
                "DrEval_normalized_Pearson",
                "RISE_interaction_Pearson",
                "RISE_interaction_R2",
            ],
            "tie_rule": "select M0",
            "independent_gdsc2_endpoints": {
                "primary": "mean absolute double-difference error",
                "secondary": ["sign accuracy", "Pearson", "Spearman"],
            },
            "success_rule": (
                "Compared with global-Pearson selection, RISE selection must lower mean paired primary error "
                "in at least one split without reducing mean sign accuracy by more than 0.01; all rules and "
                "all seeds are reported regardless of direction."
            ),
            "interpretation_boundary": (
                "This is independent assay-generation validation on response-held-out entities, not a new wet-lab experiment."
            ),
        },
        "inputs": input_files,
        "predictions": prediction_files,
    }
    output.parent.mkdir(parents=True, exist_ok=False)
    with output.open("x") as handle:
        json.dump(protocol, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output),
                      "prediction_files": len(prediction_files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

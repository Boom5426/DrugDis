#!/usr/bin/env python3
"""Rebuild T10/T13 from explicit frozen-valMSE predictions and audit T07 identity.

This is a staging-only builder.  It refuses to overwrite outputs and does not
modify the existing canonical table registry.
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


DRUG = "SMILES"
SAMPLE = "Sample_ID"
TARGET = "Target_AAC"
PREDICTED = "Predicted_AAC"
SEEDS = (3407, 3408, 3409, 3410, 3411)
ARMS = ("M0", "M4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--substrate-config", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--gdsc-raw", type=Path, required=True)
    parser.add_argument("--clio-dir", type=Path, required=True)
    parser.add_argument("--t07", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_path(root: Path, split: str, arm: str, seed: int) -> Path:
    return root / split / f"{arm}_seed{seed}" / "test_predictions_frozen_valmse.csv.gz"


def load_prediction(root: Path, split: str, arm: str, seed: int) -> tuple[pd.DataFrame, dict]:
    path = prediction_path(root, split, arm, seed)
    metadata_path = Path(str(path) + ".meta.json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing prediction or metadata for {split}/{arm}/seed{seed}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    expected = {"selection_rule": "minimum validation MSE_raw", "model": arm,
                "split_type": split, "seed": seed}
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items()
                  if metadata.get(key) != value}
    if mismatches:
        raise AssertionError(f"metadata mismatch for {path}: {mismatches}")
    actual_hash = sha256_file(path)
    if actual_hash != metadata["output"]["sha256"]:
        raise AssertionError(f"prediction checksum mismatch: {path}")
    frame = pd.read_csv(path)
    required_columns = {DRUG, SAMPLE, TARGET, PREDICTED}
    if not required_columns.issubset(frame.columns):
        raise AssertionError(f"missing columns in {path}: {required_columns - set(frame.columns)}")
    if frame.duplicated([DRUG, SAMPLE]).any():
        raise AssertionError(f"duplicate drug-sample keys in {path}")
    if len(frame) != int(metadata["n_rows"]):
        raise AssertionError(f"row count mismatch in {path}")
    return frame, metadata


def geometry(frame: pd.DataFrame) -> dict:
    drug_code, drug_names = pd.factorize(frame[DRUG], sort=False)
    sample_code, sample_names = pd.factorize(frame[SAMPLE], sort=False)
    n_drugs, n_samples = len(drug_names), len(sample_names)
    graph = coo_matrix(
        (np.ones(len(frame)), (drug_code, n_drugs + sample_code)),
        shape=(n_drugs + n_samples, n_drugs + n_samples),
    )
    components, _ = connected_components(graph + graph.T, directed=False)
    return {"n": int(len(frame)), "d_H": int(n_drugs + n_samples - components),
            "components": int(components)}


def pcc(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def frozen_scores(y, yhat, projection) -> dict:
    my, myhat = projection.residual(y), projection.residual(yhat)
    sd_interaction = float(np.std(myhat))
    sd_true = float(np.std(my))
    return {
        "rawPCC": pcc(y, yhat),
        "sharedPCC": pcc(y - my, yhat - myhat),
        "intPCC": pcc(my, myhat),
        "sd_interaction_pred": sd_interaction,
        "sd_interaction_true": sd_true,
        "emits_interaction": bool(sd_interaction > 1e-8 * sd_true),
    }


def omega_q(gdsc_raw: Path) -> pd.DataFrame:
    raw = pd.read_parquet(gdsc_raw).reset_index(drop=True)
    cut = 229420
    raw["prog"] = np.where(np.arange(len(raw)) < cut, "GDSC1", "GDSC2")
    counts = raw.groupby([DRUG, SAMPLE]).size()
    duplicate_pairs = set(counts[counts == 2].index)
    key = pd.MultiIndex.from_arrays([raw[DRUG], raw[SAMPLE]])
    subset = raw[key.isin(duplicate_pairs)]
    wide = (subset.pivot_table(index=[DRUG, SAMPLE], columns="prog", values="Sensitivity",
                               aggfunc="first").dropna().reset_index())
    return wide.rename(columns={"GDSC1": "yA", "GDSC2": "yB"})


def exact_panel(frame: pd.DataFrame, omega: pd.DataFrame) -> pd.DataFrame:
    joined = frame[[DRUG, SAMPLE, TARGET, PREDICTED]].dropna().merge(
        omega, on=[DRUG, SAMPLE], how="inner"
    )
    return joined[np.isclose(joined[TARGET], joined["yA"], atol=1e-6)].reset_index(drop=True)


def profile(frame: pd.DataFrame, projection) -> dict:
    y = frame[TARGET].to_numpy(np.float64)
    yhat = frame[PREDICTED].to_numpy(np.float64)
    my, myhat = projection.residual(y), projection.residual(yhat)
    hy, hyhat = y - my, yhat - myhat
    error = yhat - y
    m_error = projection.residual(error)
    h_error = error - m_error
    mse = float(np.mean(error**2))
    return {
        "n": int(len(y)), "rawPCC": pcc(y, yhat),
        "sharedPCC": pcc(hy, hyhat), "intPCC": pcc(my, myhat),
        "A_shared": float(hyhat.std() / hy.std()) if hy.std() else float("nan"),
        "A_int": float(myhat.std() / my.std()) if my.std() else float("nan"),
        "MSE_raw": mse, "E_shared": float(np.mean(h_error**2)),
        "E_interaction": float(np.mean(m_error**2)),
    }


def reproducibility(frame: pd.DataFrame, projection) -> dict:
    ya = frame["yA"].to_numpy(np.float64)
    yb = frame["yB"].to_numpy(np.float64)
    ma, mb = projection.residual(ya), projection.residual(yb)
    return {"q_total": pcc(ya, yb), "q_shared": pcc(ya - ma, yb - mb),
            "q_interaction": pcc(ma, mb)}


def assert_close(label: str, observed, expected, rtol: float, atol: float) -> None:
    if isinstance(expected, (int, np.integer)):
        if int(observed) != int(expected):
            raise AssertionError(f"{label}: observed {observed}, expected {expected}")
        return
    if not np.isclose(float(observed), float(expected), rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(f"{label}: observed {observed}, expected {expected}")


def align_prediction_to_panel(prediction: pd.DataFrame, panel: pd.DataFrame) -> np.ndarray:
    aligned = panel[[DRUG, SAMPLE, "y"]].merge(
        prediction[[DRUG, SAMPLE, TARGET, PREDICTED]],
        on=[DRUG, SAMPLE], how="left", validate="one_to_one",
    )
    if aligned[PREDICTED].isna().any() or len(aligned) != len(panel):
        raise AssertionError("prediction does not cover the frozen test panel exactly")
    if not np.allclose(aligned["y"], aligned[TARGET], rtol=0, atol=1e-12):
        raise AssertionError("prediction targets differ from the frozen test panel")
    return aligned[PREDICTED].to_numpy(np.float64)


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=destination.name + ".", suffix=".tmp",
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(value: dict, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=destination.name + ".", suffix=".tmp",
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    prediction_root = args.prediction_root.resolve()
    output_dir = args.output_dir.resolve()
    output_files = {
        "T10": output_dir / "T10_error_geometry.csv",
        "T13": output_dir / "T13_null_predictors.csv",
        "registry": output_dir / "frozen_prediction_registry.csv",
        "audit": output_dir / "identity_audit.json",
    }
    required_paths = [prediction_root, args.substrate_config,
                      args.manifest_dir, args.gdsc_raw, args.clio_dir, args.t07]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))
    collisions = [str(path) for path in output_files.values() if path.exists()]
    if collisions:
        raise SystemExit("refusing to overwrite existing outputs:\n" + "\n".join(collisions))

    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code / "models"))          # legacy utils.py
    sys.path.insert(0, str(code / "decomposition"))
    sys.path.insert(0, str(code / "splits"))
    sys.path.insert(0, str(code))
    import paths
    from decompositions import AdditiveProjection
    from utils import InteractionEffectEvaluator, get_base_metrics
    import make_manifests as mm

    def legacy_score(test_frame, yhat, evaluator) -> float:
        metric_frame = test_frame[[DRUG, SAMPLE]].copy()
        metric_frame[TARGET] = test_frame["y"].to_numpy()
        metric_frame[PREDICTED] = yhat
        true_effect, predicted_effect = evaluator.get_interaction_effects(metric_frame)
        _, score, _ = get_base_metrics(true_effect, predicted_effect)
        return float(score)

    t07 = pd.read_csv(args.t07)
    expected_grid = {(split, arm, seed) for split in ("LCLO", "LSO")
                     for arm in ARMS for seed in SEEDS}
    actual_grid = set(zip(t07["split"], t07["arm"], t07["seed"].astype(int)))
    if actual_grid != expected_grid or len(t07) != len(expected_grid):
        raise AssertionError("T07 does not contain exactly the expected 20-run grid")

    omega = omega_q(args.gdsc_raw.resolve())
    predictions = {}
    registry_rows = []
    rows_t10 = []
    checks = []
    for split, arm, seed in sorted(expected_grid):
        frame, metadata = load_prediction(prediction_root, split, arm, seed)
        predictions[(split, arm, seed)] = frame
        registry_rows.append({
            "split": split, "arm": arm, "seed": seed,
            "selection_rule": metadata["selection_rule"],
            "n_rows": metadata["n_rows"],
            "best_epoch": metadata["best_valmse_epoch"],
            "prediction_file": metadata["output"]["file"],
            "prediction_sha256": metadata["output"]["sha256"],
            "checkpoint_file": metadata["source"]["checkpoint"],
            "checkpoint_sha256": metadata["source"]["checkpoint_sha256"],
            "manifest_file": metadata["source"]["manifest"],
            "manifest_sha256": metadata["source"]["manifest_sha256"],
        })
        projection = AdditiveProjection(frame, [DRUG, SAMPLE])
        prof = profile(frame, projection)
        geom = geometry(frame)
        d_h, n = geom["d_H"], geom["n"]
        d_m = n - d_h
        d_h_error = prof["E_shared"] * n / d_h if d_h else float("nan")
        d_m_error = prof["E_interaction"] * n / d_m if d_m else float("nan")
        exact = exact_panel(frame, omega)
        exact_values = {"n": int(len(exact)), "eta_total": float("nan"),
                        "eta_shared": float("nan"), "eta_interaction": float("nan")}
        if len(exact) >= 300:
            exact_projection = AdditiveProjection(exact, [DRUG, SAMPLE])
            exact_profile = profile(exact, exact_projection)
            repeatability = reproducibility(exact, exact_projection)
            exact_values.update({
                "eta_total": exact_profile["rawPCC"] / np.sqrt(repeatability["q_total"]),
                "eta_shared": exact_profile["sharedPCC"] / np.sqrt(repeatability["q_shared"]),
                "eta_interaction": (
                    exact_profile["intPCC"] / np.sqrt(repeatability["q_interaction"])
                ),
            })
        rows_t10.append({
            "split": split, "run": f"{arm}_seed{seed}", "n": n,
            "d_H": d_h, "components": geom["components"], "d_H_over_N": d_h / n,
            "f_H": prof["E_shared"] / prof["MSE_raw"], "D_H": d_h_error,
            "D_M": d_m_error, "C_error": d_h_error / d_m_error,
            "E_shared": prof["E_shared"], "E_interaction": prof["E_interaction"],
            "exact_support_n": exact_values["n"], "eta_total": exact_values["eta_total"],
            "eta_shared": exact_values["eta_shared"],
            "eta_interaction": exact_values["eta_interaction"],
            "selection_rule": "frozen_valMSE",
            "prediction_sha256": metadata["output"]["sha256"],
        })
        reference = t07[(t07["split"] == split) & (t07["arm"] == arm)
                        & (t07["seed"] == seed)].iloc[0]
        for key in ("n", "rawPCC", "sharedPCC", "intPCC", "A_int", "A_shared",
                    "MSE_raw", "E_shared", "E_interaction"):
            assert_close(f"T07/T10 {split}/{arm}/seed{seed}/{key}", prof[key],
                         reference[key], args.rtol, args.atol)
        checks.append({"check": "T07_vs_prediction_profile", "split": split,
                       "arm": arm, "seed": seed, "status": "pass"})

    config = paths.load_config(args.substrate_config)
    transcriptome = mm.load_transcriptome(config)
    panel = mm.build_panel(config, transcriptome)
    rows_t13 = []
    for split, held_out_column in (("LCLO", SAMPLE), ("LSO", DRUG)):
        for seed in SEEDS:
            with (args.manifest_dir / f"{split}_seed{seed}.json").open() as handle:
                manifest = json.load(handle)
            train = panel[panel[held_out_column].astype(str).isin(set(manifest["train"]))]
            test = panel[panel[held_out_column].astype(str).isin(set(manifest["test"]))] \
                .reset_index(drop=True).rename(columns={"Sensitivity": "y"})
            evaluator = InteractionEffectEvaluator(
                train.rename(columns={"Sensitivity": TARGET}), split_type=split
            )
            projection = AdditiveProjection(test, [DRUG, SAMPLE])
            y = test["y"].to_numpy(np.float64)
            mu = float(train["Sensitivity"].mean())
            drug_effect = train.groupby(DRUG)["Sensitivity"].mean() - mu
            sample_effect = train.groupby(SAMPLE)["Sensitivity"].mean() - mu
            yhat_m0 = align_prediction_to_panel(predictions[(split, "M0", seed)], test)
            rng = np.random.default_rng(seed)
            predictor_values = {
                "constant": np.full(len(test), mu),
                "train_marginal": (
                    mu + test[DRUG].map(drug_effect).fillna(0.0).to_numpy()
                    + test[SAMPLE].map(sample_effect).fillna(0.0).to_numpy()
                ),
                "model_M0": yhat_m0,
                "shuffled": rng.permutation(yhat_m0),
            }
            for predictor, yhat in predictor_values.items():
                scores = frozen_scores(y, yhat, projection)
                rows_t13.append({
                    "split": split, "seed": seed, "predictor": predictor,
                    "n": len(test), "retired_residual_pcc": legacy_score(test, yhat, evaluator),
                    **scores, "selection_rule": ("frozen_valMSE" if predictor in
                                                   {"model_M0", "shuffled"} else "not_applicable"),
                })
            reference = t07[(t07["split"] == split) & (t07["arm"] == "M0")
                            & (t07["seed"] == seed)].iloc[0]
            model_scores = rows_t13[-2]
            for key in ("n", "rawPCC", "sharedPCC", "intPCC"):
                assert_close(f"T07/T13 {split}/M0/seed{seed}/{key}", model_scores[key],
                             reference[key], args.rtol, args.atol)
            checks.append({"check": "T07_vs_T13_model_M0", "split": split,
                           "arm": "M0", "seed": seed, "status": "pass"})

    compatible_projects = {"UMPDO1", "UMPDO2", "UMPDO3"}
    master_table = pd.read_parquet(args.clio_dir / "master_table.parquet", engine="fastparquet")
    drug_response = pd.read_parquet(args.clio_dir / "drug_response.parquet", engine="fastparquet")
    response = drug_response.groupby([DRUG, SAMPLE])["Sensitivity"].mean().reset_index()
    keep_samples = set(master_table.index[master_table["Project"].isin(compatible_projects)].astype(str))
    for seed in SEEDS:
        with (args.manifest_dir / f"LCLO_seed{seed}.json").open() as handle:
            manifest = json.load(handle)
        train = panel[panel[SAMPLE].astype(str).isin(set(manifest["train"]))]
        evaluator = InteractionEffectEvaluator(
            train.rename(columns={"Sensitivity": TARGET}), split_type="LCLO"
        )
        test = response[response[SAMPLE].astype(str).isin(keep_samples)] \
            .reset_index(drop=True).rename(columns={"Sensitivity": "y"})
        projection = AdditiveProjection(test, [DRUG, SAMPLE])
        y = test["y"].to_numpy(np.float64)
        mu = float(train["Sensitivity"].mean())
        drug_effect = train.groupby(DRUG)["Sensitivity"].mean() - mu
        sample_effect = train.groupby(SAMPLE)["Sensitivity"].mean() - mu
        rng = np.random.default_rng(seed)
        predictor_values = {
            "constant": np.full(len(test), mu),
            "train_marginal": (
                mu + test[DRUG].map(drug_effect).fillna(0.0).to_numpy()
                + test[SAMPLE].map(sample_effect).fillna(0.0).to_numpy()
            ),
            "shuffled": rng.permutation(y),
        }
        for predictor, yhat in predictor_values.items():
            rows_t13.append({
                "split": "PDO_primary", "seed": seed, "predictor": predictor,
                "n": len(test), "retired_residual_pcc": legacy_score(test, yhat, evaluator),
                **frozen_scores(y, yhat, projection), "selection_rule": "not_applicable",
            })

    table_t10 = pd.DataFrame(rows_t10).sort_values(["split", "run"]).reset_index(drop=True)
    table_t13 = pd.DataFrame(rows_t13).sort_values(["split", "seed", "predictor"]).reset_index(drop=True)
    registry = pd.DataFrame(registry_rows).sort_values(["split", "arm", "seed"]).reset_index(drop=True)
    if len(table_t10) != 20 or len(table_t13) != 55 or len(registry) != 20:
        raise AssertionError((len(table_t10), len(table_t13), len(registry)))

    audit = {
        "schema_version": "rise.result_identity_audit.v1",
        "status": "pass", "selection_rule": "minimum validation MSE_raw",
        "expected_runs": 20, "prediction_runs_checked": len(registry),
        "t07_t10_profiles_checked": 20, "t07_t13_m0_profiles_checked": 10,
        "checks": checks,
        "inputs": {
            "t07": str(args.t07.resolve()), "t07_sha256": sha256_file(args.t07.resolve()),
            "substrate_config": str(args.substrate_config.resolve()),
            "substrate_config_sha256": sha256_file(args.substrate_config.resolve()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(table_t10, output_files["T10"])
    atomic_write_csv(table_t13, output_files["T13"])
    atomic_write_csv(registry, output_files["registry"])
    audit["outputs"] = {
        key: {"file": str(path), "sha256": sha256_file(path)}
        for key, path in output_files.items() if key != "audit"
    }
    atomic_write_json(audit, output_files["audit"])
    print(json.dumps({
        "status": "pass", "T10_rows": len(table_t10), "T13_rows": len(table_t13),
        "prediction_runs": len(registry), "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

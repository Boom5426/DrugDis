#!/usr/bin/env python3
"""Run frozen Packages A and B for the RISE Nature Methods upgrade.

Package A compares published evaluation views and RISE on identical rows,
including prespecified perturbations and response-resource sensitivity.
Package B uses GDSC1 only for model selection and GDSC2 only for evaluation of
support-sampled drug-by-cell double differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DRUG = "SMILES"
SAMPLE = "Sample_ID"
TARGET = "Target_AAC"
PREDICTION = "Predicted_AAC"
MODELS = ("M0", "M4")
SPLITS = ("LCLO", "LSO")
SEEDS = tuple(range(3407, 3412))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def macro_corr(x: np.ndarray, y: np.ndarray, groups: pd.Series,
               min_n: int = 3) -> tuple[float, float, int]:
    """Unweighted mean and median of within-group Pearson correlations."""
    codes, uniques = pd.factorize(groups.astype(str), sort=False)
    n_groups = len(uniques)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    sx = np.bincount(codes, weights=x, minlength=n_groups)
    sy = np.bincount(codes, weights=y, minlength=n_groups)
    sxx = np.bincount(codes, weights=x * x, minlength=n_groups)
    syy = np.bincount(codes, weights=y * y, minlength=n_groups)
    sxy = np.bincount(codes, weights=x * y, minlength=n_groups)
    cov_num = sxy - sx * sy / counts
    vx_num = sxx - sx * sx / counts
    vy_num = syy - sy * sy / counts
    valid = (counts >= min_n) & (vx_num > 0) & (vy_num > 0)
    values = cov_num[valid] / np.sqrt(vx_num[valid] * vy_num[valid])
    return (float(values.mean()) if len(values) else float("nan"),
            float(np.median(values)) if len(values) else float("nan"),
            int(len(values)))


def deterministic_group_offset(groups: pd.Series, target_sd: float, salt: str) -> np.ndarray:
    labels = groups.astype(str).to_numpy()
    unique = pd.unique(labels)
    values = {}
    for label in unique:
        raw = hashlib.sha256(f"{salt}|{label}".encode()).digest()[:8]
        values[label] = int.from_bytes(raw, "big") / (2**64 - 1) - 0.5
    out = np.fromiter((values[label] for label in labels), dtype=np.float64, count=len(labels))
    out -= out.mean()
    if out.std() == 0:
        raise AssertionError("deterministic offset has zero variance")
    return out * (target_sd / out.std())


def metrics_bundle(frame: pd.DataFrame, y: np.ndarray, yhat: np.ndarray,
                   naive: np.ndarray, projection) -> dict:
    residual_y = projection.residual(y)
    residual_yhat = projection.residual(yhat)
    shared_y = y - residual_y
    shared_yhat = yhat - residual_yhat
    error = yhat - y
    interaction_error = projection.residual(error)
    shared_error = error - interaction_error
    var_y = float(np.var(y))
    a_shared = float(shared_yhat.std() / shared_y.std()) if shared_y.std() else float("nan")
    a_int = float(residual_yhat.std() / residual_y.std()) if residual_y.std() else float("nan")
    int_pcc = corr(residual_y, residual_yhat)
    fixed_drug_mean, fixed_drug_median, n_drugs = macro_corr(
        y, yhat, frame[DRUG])
    fixed_cell_mean, fixed_cell_median, n_cells = macro_corr(
        y, yhat, frame[SAMPLE])
    yn = y - naive
    yhn = yhat - naive
    denom_norm = float(np.sum((yn - yn.mean()) ** 2))
    return {
        "n": int(len(y)),
        "n_drugs": int(frame[DRUG].nunique()),
        "n_samples": int(frame[SAMPLE].nunique()),
        "global_Pearson": corr(y, yhat),
        "global_RMSE": float(np.sqrt(np.mean(error**2))),
        "global_R2": float(1.0 - np.sum(error**2) / np.sum((y - y.mean())**2)) if var_y else float("nan"),
        "fixed_drug_Pearson_mean": fixed_drug_mean,
        "fixed_drug_Pearson_median": fixed_drug_median,
        "fixed_drug_groups": n_drugs,
        "fixed_cell_Pearson_mean": fixed_cell_mean,
        "fixed_cell_Pearson_median": fixed_cell_median,
        "fixed_cell_groups": n_cells,
        "dreval_normalized_Pearson": corr(yn, yhn),
        "dreval_normalized_R2": (
            float(1.0 - np.sum((yhn - yn)**2) / denom_norm) if denom_norm else float("nan")
        ),
        "naive_global_Pearson": corr(y, naive),
        "naive_global_RMSE": float(np.sqrt(np.mean((naive - y)**2))),
        "rise_shared_Pearson": corr(shared_y, shared_yhat),
        "rise_shared_amplitude": a_shared,
        "rise_interaction_Pearson": int_pcc,
        "rise_interaction_amplitude": a_int,
        "rise_interaction_R2": float(2 * a_int * int_pcc - a_int**2),
        "rise_shared_error": float(np.mean(shared_error**2)),
        "rise_interaction_error": float(np.mean(interaction_error**2)),
        "rise_interaction_variance": float(np.mean(residual_y**2)),
        "projection_iterations": int(projection.last_iter),
        "projection_final_update": float(projection.last_delta),
    }


def load_protocol(path: Path) -> dict:
    with path.open() as handle:
        protocol = json.load(handle)
    if protocol.get("schema_version") != "rise.nm_validation_protocol.v1":
        raise AssertionError("unexpected protocol schema")
    for record in list(protocol["inputs"].values()) + protocol["predictions"]:
        file_path = Path(record["file"])
        if not file_path.exists() or sha256_file(file_path) != record["sha256"]:
            raise AssertionError(f"protocol input mismatch: {file_path}")
        if "metadata_file" in record:
            metadata = Path(record["metadata_file"])
            if sha256_file(metadata) != record["metadata_sha256"]:
                raise AssertionError(f"protocol metadata mismatch: {metadata}")
    return protocol


def read_manifests(protocol: dict) -> tuple[dict, set[str], set[str]]:
    manifests = {}
    for split in SPLITS:
        for seed in SEEDS:
            path = Path(protocol["inputs"][f"manifest_{split}_{seed}"]["file"])
            with path.open() as handle:
                manifests[(split, seed)] = json.load(handle)
    sample_universe = set().union(*[
        set(sum((manifests[("LCLO", seed)][part] for part in ("train", "val", "test")), []))
        for seed in SEEDS
    ])
    drug_universe = set().union(*[
        set(sum((manifests[("LSO", seed)][part] for part in ("train", "val", "test")), []))
        for seed in SEEDS
    ])
    return manifests, sample_universe, drug_universe


def build_frozen_panel(protocol: dict, sample_universe: set[str],
                       drug_universe: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    response_path = protocol["inputs"]["response_table"]["file"]
    raw = pd.read_parquet(response_path, columns=[DRUG, SAMPLE, "Sensitivity", "Project"])
    raw[DRUG] = raw[DRUG].astype(str)
    raw[SAMPLE] = raw[SAMPLE].astype(str)
    raw = raw[raw[DRUG].isin(drug_universe) & raw[SAMPLE].isin(sample_universe)]
    panel = raw.groupby([DRUG, SAMPLE], sort=False, as_index=False)["Sensitivity"].mean()
    if panel.duplicated([DRUG, SAMPLE]).any():
        raise AssertionError("frozen pair panel is not unique")
    return panel, raw


def fit_naive(panel: pd.DataFrame, manifest: dict, split: str) -> dict:
    held_axis = SAMPLE if split == "LCLO" else DRUG
    train_ids = set(map(str, manifest["train"]))
    train = panel[panel[held_axis].isin(train_ids)]
    if len(train) == 0:
        raise AssertionError("empty frozen training panel")
    grand = float(train["Sensitivity"].mean())
    return {
        "grand": grand,
        "drug_effect": (train.groupby(DRUG)["Sensitivity"].mean() - grand).to_dict(),
        "sample_effect": (train.groupby(SAMPLE)["Sensitivity"].mean() - grand).to_dict(),
        "n_train": int(len(train)),
    }


def apply_naive(frame: pd.DataFrame, fit: dict) -> np.ndarray:
    drug = frame[DRUG].map(fit["drug_effect"]).fillna(0).to_numpy(np.float64)
    sample = frame[SAMPLE].map(fit["sample_effect"]).fillna(0).to_numpy(np.float64)
    return fit["grand"] + drug + sample


def prediction_path(protocol: dict, split: str, model: str, seed: int) -> Path:
    hit = [record for record in protocol["predictions"]
           if record["split"] == split and record["model"] == model
           and int(record["seed"]) == seed]
    if len(hit) != 1:
        raise AssertionError((split, model, seed, len(hit)))
    return Path(hit[0]["file"])


def prepare_prediction_pair(protocol: dict, split: str, seed: int) -> dict[str, pd.DataFrame]:
    out = {}
    for model in MODELS:
        frame = pd.read_csv(prediction_path(protocol, split, model, seed))
        frame[DRUG] = frame[DRUG].astype(str)
        frame[SAMPLE] = frame[SAMPLE].astype(str)
        frame = frame.sort_values([DRUG, SAMPLE], kind="mergesort").reset_index(drop=True)
        if frame.duplicated([DRUG, SAMPLE]).any():
            raise AssertionError(f"duplicate prediction key: {split}/{model}/{seed}")
        out[model] = frame
    left = out["M0"][[DRUG, SAMPLE, TARGET]]
    right = out["M4"][[DRUG, SAMPLE, TARGET]]
    if not left[[DRUG, SAMPLE]].equals(right[[DRUG, SAMPLE]]):
        raise AssertionError(f"M0/M4 support mismatch: {split}/{seed}")
    if not np.allclose(left[TARGET], right[TARGET], rtol=0, atol=1e-12):
        raise AssertionError(f"M0/M4 target mismatch: {split}/{seed}")
    return out


def package_a(protocol: dict, manifests: dict, frozen_panel: pd.DataFrame,
              raw_response: pd.DataFrame, projection_class) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    comparison_rows = []
    resource_rows = []
    invariance_failures = []
    resource_groups = {
        str(name): group.groupby([DRUG, SAMPLE], as_index=False)["Sensitivity"].mean()
        for name, group in raw_response.groupby("Project", sort=True)
    }
    resource_groups["NON_NCI60"] = (
        raw_response[raw_response["Project"] != "NCI60"]
        .groupby([DRUG, SAMPLE], as_index=False)["Sensitivity"].mean()
    )

    for split in SPLITS:
        for seed in SEEDS:
            predictions = prepare_prediction_pair(protocol, split, seed)
            naive_fit = fit_naive(frozen_panel, manifests[(split, seed)], split)
            for model, frame in predictions.items():
                y = frame[TARGET].to_numpy(np.float64)
                yhat = frame[PREDICTION].to_numpy(np.float64)
                naive = apply_naive(frame, naive_fit)
                projection = projection_class(frame, [DRUG, SAMPLE])
                shared_hat, interaction_hat = projection.split(yhat)
                target_sd = 0.5 * y.std()
                variants = {
                    "original": yhat,
                    "drug_offset": yhat + deterministic_group_offset(
                        frame[DRUG], target_sd, f"drug|{split}|{seed}"),
                    "sample_offset": yhat + deterministic_group_offset(
                        frame[SAMPLE], target_sd, f"sample|{split}|{seed}"),
                    "interaction_half": shared_hat + 0.5 * interaction_hat,
                    "interaction_double": shared_hat + 2.0 * interaction_hat,
                }
                base_interaction = projection.residual(yhat)
                for variant, altered in variants.items():
                    record = {"split": split, "seed": seed, "model": model,
                              "variant": variant, "support": "frozen_pair_mean_test"}
                    record.update(metrics_bundle(frame, y, altered, naive,
                                                 projection_class(frame, [DRUG, SAMPLE])))
                    comparison_rows.append(record)
                    observed_interaction = projection.residual(altered)
                    expected_scale = {"interaction_half": 0.5,
                                      "interaction_double": 2.0}.get(variant, 1.0)
                    tolerance = max(1e-9, np.max(np.abs(base_interaction)) * 1e-7)
                    if np.max(np.abs(observed_interaction - expected_scale * base_interaction)) > tolerance:
                        invariance_failures.append({"split": split, "seed": seed,
                                                    "model": model, "variant": variant})

                # Composition sensitivity uses the same model predictions and
                # identical rows across evaluation methods within each support.
                for resource, response in resource_groups.items():
                    joined = frame[[DRUG, SAMPLE, PREDICTION]].merge(
                        response, on=[DRUG, SAMPLE], how="inner", validate="one_to_one")
                    if len(joined) < 1000 or joined[DRUG].nunique() < 2 or joined[SAMPLE].nunique() < 2:
                        continue
                    yr = joined["Sensitivity"].to_numpy(np.float64)
                    yhr = joined[PREDICTION].to_numpy(np.float64)
                    nr = apply_naive(joined, naive_fit)
                    record = {"split": split, "seed": seed, "model": model,
                              "variant": "original", "support": resource}
                    record.update(metrics_bundle(joined, yr, yhr, nr,
                                                 projection_class(joined, [DRUG, SAMPLE])))
                    resource_rows.append(record)

    comparison = pd.DataFrame(comparison_rows)
    resources = pd.DataFrame(resource_rows)
    audit = {
        "comparison_rows": int(len(comparison)),
        "expected_comparison_rows": 2 * 5 * 2 * 5,
        "resource_rows": int(len(resources)),
        "interaction_invariance_failures": invariance_failures,
        "status": "pass" if len(comparison) == 100 and not invariance_failures else "fail",
    }
    return comparison, resources, audit


def matched_gdsc(protocol: dict) -> pd.DataFrame:
    raw = pd.read_parquet(protocol["inputs"]["gdsc_raw"]["file"]).reset_index(drop=True)
    boundary = int(protocol["package_b"]["measurement_arms"]["row_boundary"])
    raw["program"] = np.where(np.arange(len(raw)) < boundary, "GDSC1", "GDSC2")
    counts = raw.groupby([DRUG, SAMPLE]).size()
    keys = set(counts[counts == 2].index)
    mask = pd.MultiIndex.from_arrays([raw[DRUG], raw[SAMPLE]]).isin(keys)
    wide = (raw[mask].pivot_table(index=[DRUG, SAMPLE], columns="program",
                                  values="Sensitivity", aggfunc="first")
            .dropna(subset=["GDSC1", "GDSC2"]).reset_index())
    wide[DRUG] = wide[DRUG].astype(str)
    wide[SAMPLE] = wide[SAMPLE].astype(str)
    return wide.rename(columns={"GDSC1": "yA", "GDSC2": "yB"})


def sample_rectangles(support: pd.DataFrame, max_rectangles: int,
                      max_attempt_multiplier: int, seed: int) -> list[tuple[str, str, str, str]]:
    adjacency = {str(drug): np.asarray(sorted(set(group[SAMPLE].astype(str))))
                 for drug, group in support.groupby(DRUG, sort=True)}
    drugs = np.asarray(sorted(adjacency))
    if len(drugs) < 2:
        return []
    rng = np.random.default_rng(seed)
    rectangles: set[tuple[str, str, str, str]] = set()
    attempts = 0
    maximum_attempts = max_rectangles * max_attempt_multiplier
    while len(rectangles) < max_rectangles and attempts < maximum_attempts:
        d1, d2 = sorted(rng.choice(drugs, size=2, replace=False).tolist())
        common = np.intersect1d(adjacency[d1], adjacency[d2], assume_unique=True)
        if len(common) >= 2:
            s1, s2 = sorted(rng.choice(common, size=2, replace=False).tolist())
            rectangles.add((d1, d2, s1, s2))
        attempts += 1
    return sorted(rectangles)


def double_differences(rectangles: list[tuple[str, str, str, str]],
                       value_map: dict[tuple[str, str], float]) -> np.ndarray:
    return np.fromiter((
        (value_map[(d1, s1)] - value_map[(d1, s2)])
        - (value_map[(d2, s1)] - value_map[(d2, s2)])
        for d1, d2, s1, s2 in rectangles
    ), dtype=np.float64, count=len(rectangles))


def contrast_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "MAE": float(np.mean(np.abs(predicted - observed))),
        "RMSE": float(np.sqrt(np.mean((predicted - observed)**2))),
        "sign_accuracy": float(np.mean(np.signbit(predicted) == np.signbit(observed))),
        "Pearson": corr(observed, predicted),
        "Spearman": rank_corr(observed, predicted),
        "observed_abs_mean": float(np.mean(np.abs(observed))),
        "predicted_abs_mean": float(np.mean(np.abs(predicted))),
    }


def package_b(protocol: dict, manifests: dict, frozen_panel: pd.DataFrame,
              projection_class) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    gdsc = matched_gdsc(protocol)
    limit = int(protocol["package_b"]["rectangle_sampling"]["maximum_unique_rectangles_per_split_seed"])
    multiplier = int(protocol["package_b"]["rectangle_sampling"]["maximum_attempt_multiplier"])
    base_seed = int(protocol["package_b"]["rectangle_sampling"]["seed"])
    utility_rows = []
    selection_metric_rows = []
    selection_rows = []
    support_audits = []

    for split_index, split in enumerate(SPLITS):
        for seed in SEEDS:
            predictions = prepare_prediction_pair(protocol, split, seed)
            naive_fit = fit_naive(frozen_panel, manifests[(split, seed)], split)
            joined_models = {}
            key_reference = None
            for model, prediction in predictions.items():
                joined = gdsc.merge(prediction[[DRUG, SAMPLE, PREDICTION]],
                                    on=[DRUG, SAMPLE], how="inner", validate="one_to_one")
                joined = joined.sort_values([DRUG, SAMPLE], kind="mergesort").reset_index(drop=True)
                keys = joined[[DRUG, SAMPLE]]
                if key_reference is None:
                    key_reference = keys
                elif not keys.equals(key_reference):
                    raise AssertionError(f"GDSC M0/M4 support mismatch: {split}/{seed}")
                joined_models[model] = joined
            support = joined_models["M0"]
            rectangle_seed = base_seed + 1000 * split_index + seed
            rectangles = sample_rectangles(support, limit, multiplier, rectangle_seed)
            if not rectangles:
                raise AssertionError(f"no eligible rectangles: {split}/{seed}")
            y_a_map = support.set_index([DRUG, SAMPLE])["yA"].to_dict()
            y_b_map = support.set_index([DRUG, SAMPLE])["yB"].to_dict()
            delta_a = double_differences(rectangles, y_a_map)
            delta_b = double_differences(rectangles, y_b_map)
            support_audits.append({
                "split": split, "seed": seed, "pairs": int(len(support)),
                "drugs": int(support[DRUG].nunique()),
                "samples": int(support[SAMPLE].nunique()),
                "rectangles": int(len(rectangles)),
                "rectangle_seed": rectangle_seed,
                "cross_arm_delta_Pearson": corr(delta_a, delta_b),
                "cross_arm_delta_sign_agreement": float(np.mean(np.signbit(delta_a) == np.signbit(delta_b))),
            })
            model_utilities = {}
            model_selection = {}
            for model, joined in joined_models.items():
                pred_map = joined.set_index([DRUG, SAMPLE])[PREDICTION].to_dict()
                delta_pred = double_differences(rectangles, pred_map)
                metrics_a = contrast_metrics(delta_a, delta_pred)
                metrics_b = contrast_metrics(delta_b, delta_pred)
                utility = {"split": split, "seed": seed, "model": model,
                           "n_pairs": int(len(joined)), "n_rectangles": int(len(rectangles))}
                utility.update({f"GDSC1_{key}": value for key, value in metrics_a.items()})
                utility.update({f"GDSC2_{key}": value for key, value in metrics_b.items()})
                utility_rows.append(utility)
                model_utilities[model] = utility

                y_a = joined["yA"].to_numpy(np.float64)
                yhat = joined[PREDICTION].to_numpy(np.float64)
                naive = apply_naive(joined, naive_fit)
                metric = metrics_bundle(joined, y_a, yhat, naive,
                                        projection_class(joined, [DRUG, SAMPLE]))
                selection_metric_rows.append({"split": split, "seed": seed,
                                              "model": model, **metric})
                model_selection[model] = metric

            criteria = {
                "global_Pearson": "global_Pearson",
                "fixed_drug_Pearson": "fixed_drug_Pearson_mean",
                "fixed_cell_Pearson": "fixed_cell_Pearson_mean",
                "DrEval_normalized_Pearson": "dreval_normalized_Pearson",
                "RISE_interaction_Pearson": "rise_interaction_Pearson",
                "RISE_interaction_R2": "rise_interaction_R2",
            }
            for criterion, field in criteria.items():
                m0 = model_selection["M0"][field]
                m4 = model_selection["M4"][field]
                selected = "M4" if m4 > m0 else "M0"
                utility = model_utilities[selected]
                selection_rows.append({
                    "split": split, "seed": seed, "criterion": criterion,
                    "metric_field": field, "M0_selection_value": m0,
                    "M4_selection_value": m4, "selected_model": selected,
                    **{key: utility[key] for key in utility if key.startswith("GDSC2_")},
                })

    utility = pd.DataFrame(utility_rows)
    selection_metrics = pd.DataFrame(selection_metric_rows)
    selections = pd.DataFrame(selection_rows)
    success_rows = []
    for split in SPLITS:
        sub = selections[selections["split"] == split]
        global_rows = sub[sub["criterion"] == "global_Pearson"].set_index("seed")
        for criterion in ("RISE_interaction_Pearson", "RISE_interaction_R2"):
            rise_rows = sub[sub["criterion"] == criterion].set_index("seed")
            mae_change = float((rise_rows["GDSC2_MAE"] - global_rows["GDSC2_MAE"]).mean())
            sign_change = float((rise_rows["GDSC2_sign_accuracy"]
                                 - global_rows["GDSC2_sign_accuracy"]).mean())
            success_rows.append({"split": split, "criterion": criterion,
                                 "mean_MAE_change_vs_global": mae_change,
                                 "mean_sign_accuracy_change_vs_global": sign_change,
                                 "passes_prespecified_rule": bool(mae_change < 0 and sign_change >= -0.01)})
    audit = {
        "matched_gdsc_pairs": int(len(gdsc)),
        "matched_gdsc_drugs": int(gdsc[DRUG].nunique()),
        "matched_gdsc_samples": int(gdsc[SAMPLE].nunique()),
        "support": support_audits,
        "utility_rows": int(len(utility)),
        "selection_rows": int(len(selections)),
        "success_evaluation": success_rows,
        "status": "pass" if len(utility) == 20 and len(selections) == 60 else "fail",
    }
    return utility, selection_metrics, selections, audit


def write_csv(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_csv(path, index=False)
    return {"file": path.name, "rows": int(len(frame)), "sha256": sha256_file(path)}


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {output_dir}")
    protocol = load_protocol(args.protocol.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decomposition"))
    from decompositions import AdditiveProjection

    manifests, sample_universe, drug_universe = read_manifests(protocol)
    frozen_panel, raw_response = build_frozen_panel(protocol, sample_universe, drug_universe)
    output_dir.mkdir(parents=True, exist_ok=False)
    comparison, resources, audit_a = package_a(
        protocol, manifests, frozen_panel, raw_response, AdditiveProjection)
    utility, selection_metrics, selections, audit_b = package_b(
        protocol, manifests, frozen_panel, AdditiveProjection)

    outputs = {}
    outputs["A_comparison"] = write_csv(comparison, output_dir / "A_formal_method_comparison.csv")
    outputs["A_resources"] = write_csv(resources, output_dir / "A_resource_sensitivity.csv")
    outputs["B_utility"] = write_csv(utility, output_dir / "B_independent_utility.csv")
    outputs["B_selection_metrics"] = write_csv(
        selection_metrics, output_dir / "B_gdsc1_selection_metrics.csv")
    outputs["B_selections"] = write_csv(selections, output_dir / "B_model_selection.csv")
    audit = {
        "schema_version": "rise.nm_validation_ab.v1",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "frozen_panel": {
            "pairs": int(len(frozen_panel)), "drugs": int(frozen_panel[DRUG].nunique()),
            "samples": int(frozen_panel[SAMPLE].nunique()),
        },
        "package_a": audit_a,
        "package_b": audit_b,
        "outputs": outputs,
        "status": "pass" if audit_a["status"] == audit_b["status"] == "pass" else "fail",
    }
    audit_path = output_dir / "AUDIT.json"
    with audit_path.open("x") as handle:
        json.dump(audit, handle, indent=1, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output_dir": str(output_dir), "status": audit["status"],
                      "package_a_rows": len(comparison), "resource_rows": len(resources),
                      "package_b_utility_rows": len(utility),
                      "package_b_selection_rows": len(selections),
                      "audit_sha256": sha256_file(audit_path)}, indent=2))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

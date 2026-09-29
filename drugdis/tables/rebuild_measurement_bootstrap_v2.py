#!/usr/bin/env python3
"""Rebuild T03 with multiplicity-preserving crossed-cluster bootstrap.

For the two-way resampling, a drug drawn ``a`` times and a sample drawn ``b``
times contributes ``a*b`` labelled copies of their observed pair.  This is the
crossed-cluster analogue of retaining cluster multiplicity; converting draws to
sets would instead perform random axis subsampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


DRUG = "SMILES"
SAMPLE = "Sample_ID"
COMPONENTS = ("q_total", "q_shared", "q_interaction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--substrate-config", type=Path, required=True)
    parser.add_argument("--gdsc-raw", type=Path, required=True)
    parser.add_argument("--old-t03", type=Path, default=None,
                        help="optional audit input: a previous T03 whose point estimates "
                             "must be reproduced exactly and whose intervals are compared")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gdsc2-row-cut", type=int, default=229420)
    parser.add_argument("--n-boot", type=int, default=600)
    parser.add_argument("--min-rows", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corr(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def reproducibility(frame: pd.DataFrame, projection) -> dict:
    ya = frame["yA"].to_numpy(np.float64)
    yb = frame["yB"].to_numpy(np.float64)
    ma, mb = projection.residual(ya), projection.residual(yb)
    return {
        "q_total": corr(ya, yb),
        "q_shared": corr(ya - ma, yb - mb),
        "q_interaction": corr(ma, mb),
    }


def labelled_axis_draw(values: np.ndarray, rng: np.random.Generator,
                       original_column: str, label_column: str) -> pd.DataFrame:
    picks = rng.choice(values, size=len(values), replace=True)
    return pd.DataFrame({original_column: picks,
                         label_column: [f"{label_column}_{i}" for i in range(len(picks))]})


def resample(frame: pd.DataFrame, mode: str, rng: np.random.Generator) -> pd.DataFrame:
    drugs = frame[DRUG].unique()
    samples = frame[SAMPLE].unique()
    if mode == "drug":
        drug_draw = labelled_axis_draw(drugs, rng, DRUG, "__drug_boot")
        boot = frame.merge(drug_draw, on=DRUG, how="inner", validate="many_to_many")
        boot[DRUG] = boot.pop("__drug_boot")
    elif mode == "sample":
        sample_draw = labelled_axis_draw(samples, rng, SAMPLE, "__sample_boot")
        boot = frame.merge(sample_draw, on=SAMPLE, how="inner", validate="many_to_many")
        boot[SAMPLE] = boot.pop("__sample_boot")
    elif mode == "two_way":
        drug_draw = labelled_axis_draw(drugs, rng, DRUG, "__drug_boot")
        sample_draw = labelled_axis_draw(samples, rng, SAMPLE, "__sample_boot")
        boot = frame.merge(drug_draw, on=DRUG, how="inner", validate="many_to_many")
        boot = boot.merge(sample_draw, on=SAMPLE, how="inner", validate="many_to_many")
        boot[DRUG] = boot.pop("__drug_boot")
        boot[SAMPLE] = boot.pop("__sample_boot")
    else:
        raise ValueError(mode)
    return boot.reset_index(drop=True)


def bootstrap(frame: pd.DataFrame, projection_class, mode: str, n_boot: int,
              seed: int, min_rows: int) -> dict:
    rng = np.random.default_rng(seed)
    values = {component: [] for component in COMPONENTS}
    row_counts = []
    skipped = 0
    for replicate in range(n_boot):
        boot = resample(frame, mode, rng)
        if len(boot) < min_rows:
            skipped += 1
            continue
        estimates = reproducibility(boot, projection_class(boot, [DRUG, SAMPLE]))
        if not all(np.isfinite(estimates[key]) for key in COMPONENTS):
            skipped += 1
            continue
        for key in COMPONENTS:
            values[key].append(estimates[key])
        row_counts.append(len(boot))
        if (replicate + 1) % 50 == 0:
            print(f"  {mode}: {replicate + 1}/{n_boot}", flush=True)
    result = {
        "mode": mode, "seed": seed, "n_requested": n_boot,
        "n_boot_ok": n_boot - skipped, "n_skipped": skipped,
        "row_count": {
            "min": int(min(row_counts)), "mean": float(np.mean(row_counts)),
            "max": int(max(row_counts)),
        },
        "components": {},
    }
    for key in COMPONENTS:
        array = np.asarray(values[key], dtype=float)
        result["components"][key] = {
            "mean": float(array.mean()), "sd": float(array.std(ddof=1)),
            "ci95": [float(np.percentile(array, 2.5)),
                     float(np.percentile(array, 97.5))],
        }
    return result


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(value: dict, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", prefix=destination.name + ".",
                                     suffix=".tmp", dir=destination.parent,
                                     delete=False) as handle:
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
    output_dir = args.output_dir.resolve()
    outputs = {
        "table": output_dir / "T03_measurement_reference.csv",
        "details": output_dir / "measurement_reference_bootstrap.json",
        "audit": output_dir / "bootstrap_audit.json",
    }
    required = [args.substrate_config, args.gdsc_raw] + ([args.old_t03] if args.old_t03 else [])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))
    collisions = [str(path) for path in outputs.values() if path.exists()]
    if collisions:
        raise SystemExit("refusing to overwrite existing outputs:\n" + "\n".join(collisions))

    code = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code / "decomposition"))
    sys.path.insert(0, str(code / "splits"))
    sys.path.insert(0, str(code))
    import paths
    from decompositions import AdditiveProjection
    import make_manifests as mm

    config = paths.load_config(args.substrate_config)
    transcriptome = mm.load_transcriptome(config)
    panel = mm.build_panel(config, transcriptome)

    raw = pd.read_parquet(args.gdsc_raw.resolve()).reset_index(drop=True)
    raw["program"] = np.where(np.arange(len(raw)) < args.gdsc2_row_cut, "GDSC1", "GDSC2")
    counts = raw.groupby([DRUG, SAMPLE]).size()
    repeated = set(counts[counts == 2].index)
    pair_index = pd.MultiIndex.from_arrays([raw[DRUG], raw[SAMPLE]])
    matched = raw[pair_index.isin(repeated)]
    matched = (matched.pivot_table(index=[DRUG, SAMPLE], columns="program",
                                   values="Sensitivity", aggfunc="first")
               .dropna().reset_index().rename(columns={"GDSC1": "yA", "GDSC2": "yB"}))
    panel_keys = set(zip(panel[DRUG].astype(str), panel[SAMPLE].astype(str)))
    inside = [tuple(value) in panel_keys
              for value in zip(matched[DRUG].astype(str), matched[SAMPLE].astype(str))]
    frame = matched[inside].reset_index(drop=True)
    point = reproducibility(frame, AdditiveProjection(frame, [DRUG, SAMPLE]))
    print(f"matched support: {len(frame):,} pairs, {frame[DRUG].nunique()} drugs, "
          f"{frame[SAMPLE].nunique()} samples", flush=True)
    print("point", json.dumps(point, sort_keys=True), flush=True)

    results = {}
    for mode, seed in (("two_way", 11), ("drug", 12), ("sample", 13)):
        print(f"starting {mode} multiplicity-preserving bootstrap", flush=True)
        results[mode] = bootstrap(frame, AdditiveProjection, mode, args.n_boot,
                                  seed, args.min_rows)

    rows = []
    for component in COMPONENTS:
        row = {
            "component": component, "point": point[component],
            "support_pairs": len(frame), "support_drugs": frame[DRUG].nunique(),
            "support_samples": frame[SAMPLE].nunique(),
            "matched_pairs_available": len(matched),
        }
        for mode in ("two_way", "drug", "sample"):
            summary = results[mode]["components"][component]
            row.update({
                f"{mode}_mean": summary["mean"], f"{mode}_sd": summary["sd"],
                f"{mode}_ci_lo": summary["ci95"][0], f"{mode}_ci_hi": summary["ci95"][1],
            })
        rows.append(row)
    table = pd.DataFrame(rows)

    audit_deltas = {}
    old = (pd.read_csv(args.old_t03).set_index("component")
           if args.old_t03 is not None else None)
    for component in (COMPONENTS if old is not None else ()):
        if not np.isclose(point[component], old.loc[component, "point"], rtol=1e-12, atol=1e-12):
            raise AssertionError(f"point estimate changed for {component}")
        audit_deltas[component] = {}
        for mode in ("two_way", "drug", "sample"):
            summary = results[mode]["components"][component]
            audit_deltas[component][mode] = {
                "ci_lo_change": summary["ci95"][0] - old.loc[component, f"{mode}_ci_lo"],
                "ci_hi_change": summary["ci95"][1] - old.loc[component, f"{mode}_ci_hi"],
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(table, outputs["table"])
    details = {
        "schema_version": "rise.measurement_bootstrap.v2",
        "algorithm": "multiplicity-preserving crossed-cluster bootstrap",
        "two_way_weight": "drug draw multiplicity times sample draw multiplicity",
        "n_boot": args.n_boot, "support": {
            "pairs": len(frame), "drugs": frame[DRUG].nunique(),
            "samples": frame[SAMPLE].nunique(), "matched_pairs_available": len(matched),
        },
        "point": point, "bootstrap": results,
        "inputs": {
            "substrate_config": str(args.substrate_config.resolve()),
            "substrate_config_sha256": sha256_file(args.substrate_config.resolve()),
            "gdsc_raw": str(args.gdsc_raw.resolve()),
            "gdsc_raw_sha256": sha256_file(args.gdsc_raw.resolve()),
        },
    }
    atomic_json(details, outputs["details"])
    audit = {
        "schema_version": "rise.measurement_bootstrap_audit.v1", "status": "pass",
        "point_estimates_unchanged": True if old is not None else None,
        "old_t03_sha256": (sha256_file(args.old_t03.resolve())
                           if args.old_t03 is not None else None),
        "new_t03_sha256": sha256_file(outputs["table"]),
        "interval_changes": audit_deltas,
    }
    atomic_json(audit, outputs["audit"])
    print(json.dumps({"status": "pass", "table": str(outputs["table"]),
                      "sha256": sha256_file(outputs["table"]),
                      "interval_changes": audit_deltas}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

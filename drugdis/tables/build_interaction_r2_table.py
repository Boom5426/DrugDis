#!/usr/bin/env python3
"""Build the canonical per-run interaction R2 table from T07 and T08."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t07", type=Path, required=True)
    parser.add_argument("--t08", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--audit-file", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    for path in (args.t07, args.t08):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    for path in (args.output_file, args.audit_file):
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    if args.output_file.parent.resolve() != args.audit_file.parent.resolve():
        raise SystemExit("output and audit must share a staging directory")

    t07 = pd.read_csv(args.t07)
    t08 = pd.read_csv(args.t08)
    cell = t07[["split", "arm", "seed", "n", "intPCC", "A_int"]].copy()
    cell = cell.rename(columns={"split": "regime"})
    cell["cohort"] = "cell_line_test"
    organoid = t08[t08["role"] == "primary"] \
        [["arm", "seed", "n", "intPCC", "A_int", "cohort"]].copy()
    organoid.insert(0, "regime", "PDO_primary")
    table = pd.concat([cell, organoid], ignore_index=True)
    table["R2_interaction"] = 2 * table["A_int"] * table["intPCC"] - table["A_int"] ** 2
    table["formula"] = "2*A_int*intPCC-A_int^2"
    table["selection_rule"] = "frozen_valMSE"
    table = table[["regime", "cohort", "arm", "seed", "n", "intPCC", "A_int",
                   "R2_interaction", "formula", "selection_rule"]] \
        .sort_values(["regime", "arm", "seed"]).reset_index(drop=True)

    expected = {(regime, arm, seed) for regime in ("LCLO", "LSO", "PDO_primary")
                for arm in ("M0", "M4") for seed in range(3407, 3412)}
    actual = set(zip(table["regime"], table["arm"], table["seed"].astype(int)))
    if len(table) != 30 or actual != expected:
        raise AssertionError("T18 does not contain the expected 30-run grid")
    if not np.isfinite(table[["intPCC", "A_int", "R2_interaction"]].to_numpy()).all():
        raise AssertionError("T18 contains a non-finite metric")
    reconstructed = table["intPCC"] ** 2 - (table["A_int"] - table["intPCC"]) ** 2
    if not np.allclose(table["R2_interaction"], reconstructed, rtol=1e-12, atol=1e-12):
        raise AssertionError("the two equivalent R2 identities disagree")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(table, args.output_file)
    audit = {
        "schema_version": "rise.interaction_r2_audit.v1", "status": "pass",
        "formula": "R2_interaction = 2*A_int*intPCC - A_int^2",
        "equivalent_formula": "intPCC^2 - (A_int-intPCC)^2",
        "rows": len(table), "new_table_sha256": sha256_file(args.output_file),
        "inputs": {"T07_sha256": sha256_file(args.t07),
                   "T08_sha256": sha256_file(args.t08)},
        "means": {
            f"{regime}_{arm}": float(group["R2_interaction"].mean())
            for (regime, arm), group in table.groupby(["regime", "arm"])
        },
    }
    atomic_json(audit, args.audit_file)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

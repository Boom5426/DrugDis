"""Compare the rebuilt tables with the canonical set shipped in results/tables.

scripts/reproduce_tables.sh writes the directly assembled tables to
$DRUGDIS_WORK/tables and the audited ones to $DRUGDIS_WORK/staging/<step>/. This
collects the audited ones into $DRUGDIS_WORK/tables under their canonical names,
records every rebuilt table in $DRUGDIS_WORK/tables/TABLES.json (file, rows,
SHA-256) so consistency_pass.py can read the set, and compares each table with
results/tables/TABLES.json. A table that is not byte-identical is reported with
its largest absolute numeric difference and any non-numeric cell that differs.
Tables that differ only by floating-point rounding (every numeric cell within a
relative 1e-9, every other cell equal) are counted separately: a different CPU
or BLAS can sum in another order and change the last digits. Exit status 1 if
any table differs beyond that or is missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

# far below the three decimals the manuscript reports
ROUNDING_RTOL, ROUNDING_ATOL = 1e-9, 1e-12

# canonical key -> file under $DRUGDIS_WORK written by an audited builder
STAGED = {
    "T03": "staging/T03/T03_measurement_reference.csv",
    "T10": "staging/valmse/T10_error_geometry.csv",
    "T13": "staging/valmse/T13_null_predictors.csv",
    "T18": "staging/T18/T18_interaction_r2.csv",
    "T19": "staging/nm/results/A_formal_method_comparison.csv",
    "T20": "staging/nm/results/A_resource_sensitivity.csv",
    "T21": "staging/nm/results/B_independent_utility.csv",
    "T22": "staging/nm/results/B_model_selection.csv",
    "T23": "staging/T23/T23_current_substrate_m3.csv",
    "T24": "staging/T24/T24_current_m3_pdo.csv",
    "T25": "staging/T25/T25_pdo_panel_counts.csv",
    "T26": "staging/T26/T26_resource_decomposition.csv",
}


def sha(path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def describe_difference(a_path, b_path) -> str:
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return f"rows differ: {len(a)} rebuilt vs {len(b)} canonical"
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])
           and pd.api.types.is_numeric_dtype(b[c])]
    worst = 0.0
    for c in num:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        both = ~(np.isnan(x) & np.isnan(y))
        if (np.isnan(x) != np.isnan(y)).any():
            return f"missing values differ in column {c}"
        if both.any():
            worst = max(worst, float(np.nanmax(np.abs(x[both] - y[both]))))
    other = [c for c in a.columns if c not in num]
    text = [c for c in other if not a[c].astype(str).equals(b[c].astype(str))]
    return f"max |numeric difference| {worst:.3g}" + (f"; text columns differ: {text}" if text else "")


def only_file_hashes_differ(a_path, b_path) -> bool:
    """True if the two tables agree exactly in every column except *_sha256 ones."""
    a, b = pd.read_csv(a_path, dtype=str), pd.read_csv(b_path, dtype=str)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    keep = [c for c in a.columns if not c.endswith("_sha256")]
    return len(keep) < len(a.columns) and a[keep].equals(b[keep])


def within_rounding(a_path, b_path, rtol: float = ROUNDING_RTOL) -> bool:
    """True if every numeric cell agrees within `rtol` and every other cell is equal."""
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for c in a.columns:
        if c.endswith("_sha256"):
            continue
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float),
                               rtol=rtol, atol=ROUNDING_ATOL, equal_nan=True):
                return False
        elif not a[c].astype(str).equals(b[c].astype(str)):
            return False
    return True


def main() -> int:
    work, tables = paths.work_dir(), paths.tables_dir()
    canonical = json.load(open(paths.CANONICAL_TABLES / "TABLES.json"))
    reg_path = tables / "TABLES.json"
    reg = json.load(open(reg_path)) if reg_path.exists() else {}
    for key, rel in STAGED.items():
        src = work / rel
        if src.exists():
            dst = tables / canonical[key]["file"]
            shutil.copyfile(src, dst)
            with open(dst) as fh:
                rows = max(sum(1 for _ in fh) - 1, 0)
            reg[key] = {"file": dst.name, "rows": rows, "sha256": sha(dst)}
    json.dump({k: reg[k] for k in sorted(reg)}, open(reg_path, "w"), indent=1)

    bad, exact, rounding = 0, 0, 0
    print(f"{'table':6s} {'file':38s} result")
    for key in sorted(canonical):
        want = canonical[key]
        got = tables / want["file"]
        if not got.exists():
            print(f"{key:6s} {want['file']:38s} NOT REBUILT")
            bad += 1
        elif sha(got) == want["sha256"]:
            print(f"{key:6s} {want['file']:38s} identical")
            exact += 1
        elif only_file_hashes_differ(got, paths.CANONICAL_TABLES / want["file"]):
            # e.g. T10 records the SHA-256 of each gzip prediction export; gzip stores
            # the write time, so a re-export changes the hash but not the content
            print(f"{key:6s} {want['file']:38s} identical values; only the *_sha256 "
                  f"column differs (hashes of the re-exported input files)")
        elif within_rounding(got, paths.CANONICAL_TABLES / want["file"]):
            print(f"{key:6s} {want['file']:38s} equal within floating-point rounding: "
                  f"{describe_difference(got, paths.CANONICAL_TABLES / want['file'])}")
            rounding += 1
        else:
            print(f"{key:6s} {want['file']:38s} DIFFERS: "
                  f"{describe_difference(got, paths.CANONICAL_TABLES / want['file'])}")
            bad += 1
    print(f"\n{exact} of {len(canonical)} tables byte-identical to the canonical set; "
          f"{len(canonical) - exact - rounding - bad} identical in every value; "
          f"{rounding} equal within floating-point rounding; {bad} differ or missing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

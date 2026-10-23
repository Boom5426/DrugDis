"""Build examples/data/lclo_seed3407_m0_10_cell_lines.csv.gz from a reported prediction export.

    python examples/make_example_data.py --export test_predictions_frozen_valmse.csv.gz

The export holds the test predictions of M0 under held-out cell lines, seed 3407,
at best_valmse.pth; table T10 records its SHA-256, which is checked here. The
example keeps every drug-sample pair of the first 10 test cell lines in sorted
Sample_ID order, a rule fixed before any score was computed, and renames the
columns to the names the README examples use: SMILES, Sample_ID, Sensitivity
(measured) and prediction. Scores on this subset differ from the manuscript's,
which are computed on the whole test set.
"""

import argparse
import hashlib
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_CELL_LINES = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", required=True,
                        help="test_predictions_frozen_valmse.csv.gz of LCLO M0_seed3407")
    parser.add_argument("--out", default=os.path.join(
        REPO, "examples", "data", "lclo_seed3407_m0_10_cell_lines.csv.gz"))
    args = parser.parse_args()

    t10 = pd.read_csv(os.path.join(REPO, "results", "tables", "T10_error_geometry.csv"))
    want = t10.loc[(t10["split"] == "LCLO") & (t10["run"] == "M0_seed3407"), "prediction_sha256"]
    with open(args.export, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    if len(want) != 1 or got != want.iloc[0]:
        raise SystemExit(f"{args.export} is not the export T10 records for LCLO M0_seed3407")

    frame = pd.read_csv(args.export, dtype=str)  # values are copied as written, never re-parsed
    keep = sorted(frame["Sample_ID"].unique())[:N_CELL_LINES]
    out = (frame[frame["Sample_ID"].isin(keep)]
           .rename(columns={"Target_AAC": "Sensitivity", "Predicted_AAC": "prediction"})
           .sort_values(["Sample_ID", "SMILES"])
           [["SMILES", "Sample_ID", "Sensitivity", "prediction"]])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{len(out):,} pairs, {out['Sample_ID'].nunique()} cell lines, "
          f"{out['SMILES'].nunique():,} compounds -> {args.out}")


if __name__ == "__main__":
    main()

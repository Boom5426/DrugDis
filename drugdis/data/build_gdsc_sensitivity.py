"""GDSC1 and GDSC2 responses with their resource identity kept apart.

drug_response.parquet keeps one measurement per (compound name, sample), so a
pair measured by both GDSC programmes appears there once. The cross-assay
reproducibility reference (T03), the measurement-limited error share (T10) and
the GDSC1-selection / GDSC2-test comparison (T19 to T22) need both measurements.
This builds them from the DROMA tables GDSC1_drug and GDSC2_drug, following the
logic of the analysis notebook data_GDSC.ipynb.

Rows are the non-missing GDSC1 values followed by the non-missing GDSC2 values,
each mapped to a canonical SMILES through annotations/drug_anno_with_struc_info.csv
and restricted to samples with a GDSC expression profile. Downstream code
separates the two programmes by row position: rows before 229,420 are GDSC1
(`--gdsc2-row-cut` in rebuild_measurement_bootstrap_v2.py and the frozen
`row_boundary` of the validation protocol).

    python drugdis/data/build_gdsc_sensitivity.py \
        --out "$DRUGDIS_DATA/gdsc_sensitivity_data.parquet"
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import pandas as pd
from rdkit import Chem, rdBase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

rdBase.DisableLog("rdApp.error")


def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:  # noqa: BLE001  (any parse failure means the name stays unmapped)
        return None


def melt(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.melt(id_vars=["feature_id"], value_vars=wide.columns.drop("feature_id"),
                     var_name="Sample_ID", value_name="Sensitivity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--droma", default=None, help="default: $DRUGDIS_DATA/droma.sqlite")
    ap.add_argument("--annotation", default=None,
                    help="default: $DRUGDIS_DATA/annotations/drug_anno_with_struc_info.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    droma = a.droma or str(paths.data_dir() / "droma.sqlite")
    annotation = a.annotation or str(paths.data_dir() / "annotations" /
                                     "drug_anno_with_struc_info.csv")

    conn = sqlite3.connect(f"file:{droma}?mode=ro", uri=True)
    try:
        info = pd.read_sql_query("PRAGMA table_info('GDSC_mRNA')", conn)
        expressed = set(info["name"]) - {"feature_id"}
        g1 = pd.read_sql_query("SELECT * FROM GDSC1_drug", conn)
        g2 = pd.read_sql_query("SELECT * FROM GDSC2_drug", conn)
    finally:
        conn.close()
    long = pd.concat([melt(g1), melt(g2)], ignore_index=True).rename(
        columns={"feature_id": "Drug_Name"})
    long = long.dropna(subset=["Sensitivity"])
    long["Sensitivity"] = pd.to_numeric(long["Sensitivity"])

    s = pd.read_csv(annotation)[["DrugName", "SMILES"]].dropna(subset=["SMILES"])
    s["Canonical_SMILES"] = s["SMILES"].apply(canonical_smiles)
    s = s.dropna(subset=["Canonical_SMILES"])
    s = s[["DrugName", "Canonical_SMILES"]].drop_duplicates(subset=["DrugName"], keep="first")

    m = pd.merge(long, s, left_on="Drug_Name", right_on="DrugName", how="left")
    m = m.dropna(subset=["Canonical_SMILES"])
    m = m[["Drug_Name", "Sample_ID", "Sensitivity", "Canonical_SMILES"]].rename(
        columns={"Canonical_SMILES": "SMILES"})
    out = m[m["Sample_ID"].isin(expressed)].copy()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_parquet(a.out)
    print(f"wrote {a.out}: {len(out):,} rows, {out['Drug_Name'].nunique():,} compound names, "
          f"{out['Sample_ID'].nunique():,} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

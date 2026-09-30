"""Build the processed inputs from the DROMA SQLite database.

Converted from the analysis notebook data_all_to_df.ipynb (cells 12, 14, 17 and
19) with the logic unchanged, plus one recorded relabelling step (see
`master_table`). Writes under --out:

    master_table.parquet                      one row per DROMA sample
    drug_response.parquet                     one row per (compound name, sample)
    omics_mrna_raw/<cohort>.parquet           samples x genes, float32
    omics_baseline/baseline_gene_list.txt     genes shared by CCLE and GDSC
    omics_baseline/baseline_mrna_common_genes.parquet

Inputs, under DRUGDIS_DATA: droma.sqlite and the two compound-structure tables
annotations/drug_anno_with_struc_info.csv and annotations/drug_anno_nci60_structure.csv
(DrugName -> SMILES).

Overlap rule. A compound name measured on the same sample by more than one
response resource keeps one measurement: the first in the fixed resource order of
DRUG_TABLES below (`groupby(...).first()` over rows concatenated in that order).
The manuscript's Methods state this rule and how many rows it sets aside.

    python drugdis/data/build_processed_tables.py --out "$DRUGDIS_DATA"
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, rdBase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

rdBase.DisableLog("rdApp.error")

MODEL_TYPE = {"CellLine": "Cell Line", "PDC": "PDO", "PDX": "PDX"}
DRUG_TABLES = ["CCLE_drug", "CTRP1_drug", "CTRP2_drug", "FIMM_drug", "GDSC1_drug",
               "GDSC2_drug", "GRAY_drug", "HKUPDO_drug", "LICOB_drug", "NCI60_drug",
               "PDTXBreast_drug", "Prism_drug", "Tavor_drug", "UHNBreast_drug",
               "UMPDO1_drug", "UMPDO2_drug", "UMPDO3_drug", "Xeva_drug", "gCSI_drug"]
MRNA_TABLES = ["CCLE_mRNA", "GDSC_mRNA", "NCI60_mRNA", "Tavor_mRNA", "UMPDO1_mRNA",
               "UMPDO2_mRNA", "UMPDO3_mRNA", "Xeva_mRNA", "LICOB_mRNA", "HKUPDO_mRNA"]
ANNOTATIONS = ("drug_anno_with_struc_info.csv", "drug_anno_nci60_structure.csv")


def canonical_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None
    except Exception:  # noqa: BLE001  (the notebook treats any parse failure as unmapped)
        return None


def fill_model_type(row):
    if pd.notna(row["Model_Type"]):
        return row["Model_Type"]
    proj = str(row["ProjectID"])
    if "UMPDO" in proj or "LICOB" in proj or "HKUPDO" in proj:
        return "PDO"
    if "Xeva" in proj or "PDTXBreast" in proj or "Tavor" in proj:
        return "PDX"
    if any(k in proj for k in ("GDSC", "CCLE", "gCSI", "NCI60", "FIMM", "GRAY", "Prism")):
        return "Cell Line"
    return "Unknown"


def master_table(conn) -> pd.DataFrame:
    anno = pd.read_sql_query("SELECT * FROM sample_anno", conn)
    anno["Model_Type"] = anno["DataType"].map(MODEL_TYPE)
    anno["Model_Type"] = anno.apply(fill_model_type, axis=1)
    master = anno[["SampleID", "ProjectID", "Model_Type", "TumorType", "MolecularSubtype",
                   "HarmonizedIdentifier", "PatientID", "AlternateName"]].copy()
    master = master.rename(columns={"SampleID": "Sample_ID", "ProjectID": "Project",
                                    "TumorType": "Tumor_Type",
                                    "MolecularSubtype": "Molecular_Subtype",
                                    "PatientID": "Patient_ID"}).set_index("Sample_ID")
    # Recorded relabelling (2026-08-10). DROMA types the 53 Tavor samples as PDC, so
    # the mapping above labels them PDO; the analysed table labels them "Cell Line".
    # The substrate config excludes Tavor from the cell-line panel
    # (excluded_projects), so the relabelling keeps Tavor out of the organoid
    # cohorts as well. Without it the output is the pre-relabelling table.
    master.loc[master["Project"] == "Tavor", "Model_Type"] = "Cell Line"
    return master


def drug_response(conn, master: pd.DataFrame, annotation_dir: str) -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(os.path.join(annotation_dir, f)) for f in ANNOTATIONS],
                    ignore_index=True)
    smi = raw[["DrugName", "SMILES"]].dropna(subset=["SMILES"])
    smi["Canonical_SMILES"] = smi["SMILES"].apply(canonical_smiles)
    smi = smi.dropna(subset=["Canonical_SMILES"])
    smi = smi[["DrugName", "Canonical_SMILES"]].drop_duplicates(subset=["DrugName"], keep="first")
    print(f"  compound name -> canonical SMILES: {len(smi):,} names", flush=True)

    parts = []
    for table in DRUG_TABLES:
        wide = pd.read_sql_query(f"SELECT * FROM '{table}'", conn)
        if "feature_id" not in wide.columns:
            continue
        long = wide.melt(id_vars=["feature_id"], value_vars=wide.columns.drop("feature_id"),
                         var_name="Sample_ID", value_name="Sensitivity")
        long["Project"] = table.replace("_drug", "")
        parts.append(long)
    resp = pd.concat(parts, ignore_index=True).dropna(subset=["Sensitivity"])
    print(f"  measured (name, sample, resource) rows: {len(resp):,}", flush=True)
    # overlap rule: the first resource in DRUG_TABLES order wins
    resp = (resp.groupby(["feature_id", "Sample_ID"])[["Sensitivity", "Project"]]
            .first().reset_index())
    merged = pd.merge(resp, smi, left_on="feature_id", right_on="DrugName", how="left")
    merged = merged.dropna(subset=["Canonical_SMILES"])
    merged = merged[merged["Sample_ID"].isin(set(master.index))]
    out = merged[["Sample_ID", "Canonical_SMILES", "Sensitivity", "Project", "feature_id"]]
    return out.rename(columns={"Canonical_SMILES": "SMILES", "feature_id": "Original_DrugName"})


def mrna_raw(conn, master: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    valid = set(master.index)
    for table in MRNA_TABLES:
        wide = pd.read_sql_query(f"SELECT * FROM '{table}'", conn)
        if "feature_id" not in wide.columns:
            continue
        if wide["feature_id"].duplicated().any():
            wide = wide.drop_duplicates(subset=["feature_id"], keep="first")
        t = wide.set_index("feature_id").T
        t = t[t.index.isin(valid)]
        if len(t) == 0:
            continue
        t = t.astype(np.float32)
        t.to_parquet(os.path.join(out_dir, table.replace("_mRNA", "") + ".parquet"),
                     engine="auto", compression="snappy")
        print(f"  omics_mrna_raw/{table.replace('_mRNA', '')}.parquet {t.shape}", flush=True)


def baseline(raw_dir: str, out_dir: str) -> None:
    """Genes common to CCLE and GDSC, and every cohort reindexed onto them.

    Rows are concatenated in os.listdir order, as in the notebook; the row order of
    baseline_mrna_common_genes.parquet therefore depends on the filesystem."""
    os.makedirs(out_dir, exist_ok=True)
    ccle = set(pq.ParquetFile(os.path.join(raw_dir, "CCLE.parquet")).schema.names)
    gdsc = set(pq.ParquetFile(os.path.join(raw_dir, "GDSC.parquet")).schema.names)
    ccle.discard("__index_level_0__")
    gdsc.discard("__index_level_0__")
    genes = sorted(ccle & gdsc)
    with open(os.path.join(out_dir, "baseline_gene_list.txt"), "w") as fh:
        for g in genes:
            fh.write(f"{g}\n")
    frames = []
    for name in [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]:
        d = pd.read_parquet(os.path.join(raw_dir, name), engine="fastparquet")
        keep = list(set(d.columns).intersection(genes))
        frames.append(d[keep].reindex(columns=genes, fill_value=np.nan))
    base = pd.concat(frames)
    base.to_parquet(os.path.join(out_dir, "baseline_mrna_common_genes.parquet"),
                    engine="auto", compression="snappy")
    print(f"  omics_baseline: {len(genes):,} genes, matrix {base.shape}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--droma", default=None, help="default: $DRUGDIS_DATA/droma.sqlite")
    ap.add_argument("--annotations", default=None, help="default: $DRUGDIS_DATA/annotations")
    ap.add_argument("--out", required=True, help="output directory (normally $DRUGDIS_DATA)")
    ap.add_argument("--steps", default="master,response,mrna,baseline")
    a = ap.parse_args()
    droma = a.droma or str(paths.data_dir() / "droma.sqlite")
    annotations = a.annotations or str(paths.data_dir() / "annotations")
    steps = set(a.steps.split(","))
    os.makedirs(a.out, exist_ok=True)
    conn = sqlite3.connect(f"file:{droma}?mode=ro", uri=True)
    try:
        master = master_table(conn)
        if "master" in steps:
            master.to_parquet(os.path.join(a.out, "master_table.parquet"),
                              engine="auto", compression="snappy")
            print(f"master_table.parquet {master.shape} "
                  f"{master['Model_Type'].value_counts().to_dict()}", flush=True)
        if "response" in steps:
            resp = drug_response(conn, master, annotations)
            resp.to_parquet(os.path.join(a.out, "drug_response.parquet"),
                            engine="auto", compression="snappy")
            print(f"drug_response.parquet {len(resp):,} pairs, "
                  f"{resp['Sample_ID'].nunique():,} samples, "
                  f"{resp['SMILES'].nunique():,} compounds", flush=True)
        if "mrna" in steps:
            mrna_raw(conn, master, os.path.join(a.out, "omics_mrna_raw"))
    finally:
        conn.close()
    if "baseline" in steps:
        baseline(os.path.join(a.out, "omics_mrna_raw"), os.path.join(a.out, "omics_baseline"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage T25: sample, compound and pair counts of the frozen organoid panel.

Runs on the analysis host. T08 records how many drug-organoid pairs each cohort
contributes to the zero-shot evaluation, but not how many organoids or
compounds those pairs cover, so Fig. 1b had nowhere canonical to read them from
and carried numbers transcribed from an earlier draft (441 samples, 17,097 rows)
that no frozen artifact supports.

The panel is rebuilt exactly as `evaluate_current_m3_pdo.py` builds it: PDO
samples from the master table that have raw expression, responses averaged per
(SMILES, Sample_ID), restricted to compounds with an ECFP4 feature and samples
with expression. Nothing is re-derived beyond counting. Every source is checked
against the SHA-256 that the registered T24 audit recorded (the master table,
the response table, the ECFP4 features and each organoid expression file), T08
is checked against the canonical registry, and the pair count of every cohort
must equal T08 before anything is written, so a reconstruction that drifted from
the frozen panel cannot be staged.

Usage (see scripts/reproduce_tables.sh):

    python drugdis/tables/build_T25_pdo_panel_counts.py \
        --processed-root "$DRUGDIS_DATA" \
        --reference-audit staging_current_m3_pdo_v1/results.json \
        --t08 tables/T08_pdo_stress.csv --registry tables/TABLES.json \
        --out-dir staging_pdo_panel_counts_v1
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

DRUG = "SMILES"
SAMPLE = "Sample_ID"
COMPARABLE = ("UMPDO1", "UMPDO2", "UMPDO3")
PRIMARY = "COMPARABLE(UMPDO1+2+3)"
SOURCES = {
    "master_table": "master_table.parquet",
    "response_table": "drug_response.parquet",
    "drug_features": "Molecule_Embeddings/ECFP4_emb2048.pickle",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--t08", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True,
                        help="canonical TABLES.json holding the T08 hash")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    processed = args.processed_root.resolve()

    registry = json.load(args.registry.open())
    t08_sha = sha256_file(args.t08)
    if t08_sha != registry["T08"]["sha256"]:
        raise AssertionError(f"T08 {t08_sha} differs from the canonical registry")
    reference_audit = json.load(args.reference_audit.open())
    reference = reference_audit["sources"]
    expression_hashes = {k: v["source_sha256"]
                         for k, v in reference_audit["coverage"].items()}
    sources = {}
    for key, rel in SOURCES.items():
        path = processed / rel
        got = sha256_file(path)
        if got != reference[key]["sha256"]:
            raise AssertionError(f"{key}: {got} differs from the T24 audit hash")
        sources[key] = {"file": str(path), "sha256": got}

    master = pd.read_parquet(processed / SOURCES["master_table"], engine="fastparquet")
    pdo_ids = set(master.index[master["Model_Type"] == "PDO"].astype(str))
    origin: dict[str, str] = {}
    for path in sorted(Path(p) for p in glob.glob(str(processed / "omics_mrna_raw" / "*.parquet"))):
        data = pd.read_parquet(path, engine="fastparquet")
        data.index = data.index.astype(str)
        members = data.index[data.index.isin(pdo_ids)]
        if not len(members):
            continue
        got = sha256_file(path)
        if got != expression_hashes.get(path.stem):
            raise AssertionError(f"{path.name}: {got} differs from the T24 audit hash")
        sources[f"expression_{path.stem}"] = {"file": str(path), "sha256": got}
        for sample in members:
            # T24 labels with origin.update over sorted files, so the last file
            # would win. No organoid may sit in two files, which makes the label
            # unambiguous and the two rules identical.
            if sample in origin:
                raise AssertionError(f"{sample} is in both {origin[sample]} and {path.stem}")
            origin[sample] = path.stem

    response = pd.read_parquet(processed / SOURCES["response_table"], engine="fastparquet")
    response = response.groupby([DRUG, SAMPLE])["Sensitivity"].mean().reset_index()
    with (processed / SOURCES["drug_features"]).open("rb") as handle:
        features = pickle.load(handle)
    panel = response[response[DRUG].isin(features.keys())
                     & response[SAMPLE].astype(str).isin(set(origin))].reset_index(drop=True)
    panel["resource"] = panel[SAMPLE].astype(str).map(origin)

    groups = {"ALL": panel}
    groups.update({r: panel[panel["resource"] == r] for r in sorted(panel["resource"].unique())})
    groups[PRIMARY] = panel[panel["resource"].isin(COMPARABLE)]

    t08 = pd.read_csv(args.t08)
    t08 = t08[(t08["arm"] == "M0") & (t08["seed"] == t08["seed"].min())].set_index("cohort")
    rows = []
    for cohort, sub in groups.items():
        expected = int(t08.loc[cohort, "n"])
        if len(sub) != expected:
            raise AssertionError(f"{cohort}: {len(sub)} pairs rebuilt, T08 has {expected}")
        rows.append({
            "cohort": cohort,
            "role": t08.loc[cohort, "role"],
            "in_primary": cohort in COMPARABLE,
            "pairs": int(len(sub)),
            "samples": int(sub[SAMPLE].nunique()),
            "compounds": int(sub[DRUG].nunique()),
        })

    out_dir.mkdir(parents=True)
    table = out_dir / "T25_pdo_panel_counts.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    audit = {
        "schema_version": "rise.pdo_panel_counts.v1",
        "construction": "evaluate_current_m3_pdo.py panel, counted without re-deriving",
        "cross_check": "pairs per cohort equal T08 (M0, first seed) for every row",
        "t08_sha256": t08_sha,
        "reference_audit_sha256": sha256_file(args.reference_audit),
        "sources": sources,
        "rows": rows,
        "status": "pass",
        "new_table_sha256": sha256_file(table),
    }
    (out_dir / "results.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

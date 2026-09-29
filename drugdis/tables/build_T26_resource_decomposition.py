#!/usr/bin/env python3
"""Stage T26: the orthogonal response decomposition within each response resource.

Runs on the analysis host. T02 decomposes the pooled benchmark, in which NCI60
supplies 68.5% of the response rows and the eleven resources share no uniform
assay metric. This table repeats the identical decomposition inside each
resource, on that resource's own drug-sample support, and on the benchmark
rebuilt without NCI60.

Why the resources are rebuilt from DROMA rather than split out of the benchmark.
The processed response table (`drug_response.parquet`) holds one row per
(original drug name, sample). Where several resources measured the same drug
name on the same sample it keeps the value of the resource whose name sorts
first (CCLE < CTRP1 < CTRP2 < FIMM < GDSC1 < GDSC2 < GRAY < NCI60 < Prism <
UHNBreast < gCSI) and drops the others; the retained values equal the DROMA
per-resource tables exactly. Splitting the benchmark by its `Project` label
would therefore give each resource only the pairs it won, which for GDSC2 and
gCSI is a minority of its screen and not a random one. Each resource screen is
instead read from its DROMA table `<resource>_drug`.

The reconstruction is gated before anything is written:
  R1  the DROMA tables, the name-to-SMILES map of the processed table and the
      resource-order rule reproduce every benchmark-resource row of
      `drug_response.parquet` (name, sample, resource, SMILES, value);
  R2  the pooled panel built from R1 equals the frozen panel of
      `make_manifests.build_panel`, and its decomposition equals the canonical
      T02 full-panel row;
  R3  the per-resource row counts equal the canonical T01 composition.

Scopes written, one decomposition each:
  benchmark                 the frozen panel (reproduces T02)
  benchmark_without_NCI60   the benchmark rebuilt by the same rules without NCI60
  resource_screen           one resource's eligible measurements, no other
                            resource involved; rows sharing a canonical pair
                            within the resource are averaged, as in the benchmark
  resource_benchmark_rows   the rows the benchmark attributes to that resource,
                            decomposed on their own support (a check on how much
                            the resource-order rule changes the answer)

Eligibility is the benchmark's, taken from the frozen substrate config: a CCLE
profile, the master table's cell-line type, not an excluded project, and an
ECFP4 fingerprint for the compound. No value is transformed.

Usage (see scripts/reproduce_tables.sh):

    python drugdis/tables/build_T26_resource_decomposition.py \
        --config configs/substrate_config.frozen.json \
        --droma "$DRUGDIS_DATA/droma.sqlite" \
        --registry tables/TABLES.json --t01 tables/T01_substrate.csv \
        --t02 tables/T02_decomposition.csv \
        --out-dir staging_resource_decomposition_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_tables as bt                                       # noqa: E402
from decompositions import DRUG, SAMPLE, AdditiveProjection     # noqa: E402
import make_manifests as mm                                     # noqa: E402
import paths                                                    # noqa: E402

# The eleven response resources of the benchmark (SI Table 1), in the order the
# processed table resolves overlaps: plain string order.
RESOURCES = ("CCLE", "CTRP1", "CTRP2", "FIMM", "GDSC1", "GDSC2", "GRAY",
             "NCI60", "Prism", "UHNBreast", "gCSI")
NAME = "Original_DrugName"
Y = "Sensitivity"
PAIR_TOL = 1e-12          # pair means are summed in a different order than the frozen build
T02_RTOL = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--droma", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--t01", type=Path, required=True)
    ap.add_argument("--t02", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def droma_rows(con: sqlite3.Connection, resource: str) -> pd.DataFrame:
    """One resource's DROMA drug table in long form, missing entries dropped."""
    wide = pd.read_sql(f"select * from `{resource}_drug`", con)
    if wide["feature_id"].duplicated().any():
        raise AssertionError(f"{resource}_drug repeats a drug row")
    wide = wide.set_index("feature_id")
    values = wide.apply(pd.to_numeric, errors="raise")
    long = values.stack().rename(Y).reset_index()       # pandas 1.5: drops NaN
    long.columns = [NAME, SAMPLE, Y]
    long["Project"] = resource
    return long


def resolve_overlaps(rows: pd.DataFrame) -> pd.DataFrame:
    """The processed table's rule: one row per (name, sample), first resource by name."""
    order = rows.sort_values("Project", kind="mergesort")
    return order.drop_duplicates([NAME, SAMPLE], keep="first")


class Eligibility:
    """The benchmark's key-based inclusion rule, as `make_manifests.build_panel` applies it."""

    def __init__(self, cfg: dict):
        dg = mm.load_transcriptome(cfg)
        master = pd.read_parquet(cfg["master_table_path"], engine="fastparquet")
        with open(cfg["drug_features_path"], "rb") as fh:
            self.drugs = set(pickle.load(fh).keys())
        samples = set(dg.index) & set(master.index[master["Model_Type"] == cfg["model_type"]])
        excl = set(cfg.get("excluded_projects", []))
        if excl:
            samples -= set(master.index[master["Project"].isin(excl)])
        self.samples = samples
        self.transcriptome = dg

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = df[DRUG].isin(self.drugs) & df[SAMPLE].isin(self.samples)
        return df[keep]


def pair_means(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby([DRUG, SAMPLE])[Y].mean().reset_index()


def decompose(fr: pd.DataFrame, scope: str, resource: str, n_rows: int) -> dict:
    """The canonical T02 decomposition, plus the support diagnostics T26 adds."""
    fr = fr.reset_index(drop=True)
    rec = bt.decomp(fr, label=f"{scope}:{resource}", panel_kind=scope)
    proj = AdditiveProjection(fr, [DRUG, SAMPLE])
    proj.residual(fr[Y].to_numpy(np.float64))
    deg_d = fr[DRUG].value_counts()
    deg_s = fr[SAMPLE].value_counts()
    out = {"scope": scope, "resource": resource, "n_rows": int(n_rows)}
    out.update({k: v for k, v in rec.items() if k not in ("panel", "kind")})
    out.update({"n_interaction_dof": int(len(fr) - rec["rank_H"]),
                "drugs_degree1": int((deg_d == 1).sum()),
                "samples_degree1": int((deg_s == 1).sum()),
                "median_pairs_per_drug": float(deg_d.median()),
                "median_pairs_per_sample": float(deg_s.median()),
                "projection_iterations": int(proj.last_iter),
                "projection_final_update": float(proj.last_delta)})
    print(f"  {scope:24s} {resource:10s} N={len(fr):>9,d} d_H/N={rec['d_H_over_N']:.4f} "
          f"additive={rec['share_shared_pct']:6.2f}% c={rec['components']} "
          f"iter={proj.last_iter}", flush=True)
    return out


def main() -> int:
    a = parse_args()
    t0 = time.time()
    cfg = paths.load_config(a.config)
    registry = json.load(open(a.registry))
    for key, path in (("T01", a.t01), ("T02", a.t02)):
        if sha256_file(path) != registry[key]["sha256"]:
            raise AssertionError(f"{key} differs from the canonical registry")
    t01 = pd.read_csv(a.t01).set_index("item")["value"]
    t02 = pd.read_csv(a.t02).set_index("panel").loc["full_panel"]

    processed = pd.read_parquet(cfg["drug_response_path"], engine="fastparquet")
    if processed.groupby(NAME)[DRUG].nunique().max() != 1:
        raise AssertionError("a drug name maps to more than one SMILES")
    name_to_smiles = processed.drop_duplicates(NAME).set_index(NAME)[DRUG]

    con = sqlite3.connect(str(a.droma))
    raw = pd.concat([droma_rows(con, r) for r in RESOURCES], ignore_index=True)
    raw[DRUG] = raw[NAME].map(name_to_smiles)
    mapped = raw[raw[DRUG].notna()].copy()
    print(f"DROMA rows {len(raw):,}; with a SMILES {len(mapped):,}", flush=True)

    # ---- R1: the processed table, from DROMA and the resource-order rule
    resolved = resolve_overlaps(mapped)
    ref = processed[processed["Project"].isin(RESOURCES)]
    key = [NAME, SAMPLE]
    m = resolved.merge(ref, on=key, how="outer", suffixes=("", "_ref"), indicator=True)
    r1 = {"rows_rebuilt": int(len(resolved)), "rows_processed": int(len(ref)),
          "only_rebuilt": int((m["_merge"] == "left_only").sum()),
          "only_processed": int((m["_merge"] == "right_only").sum())}
    both = m[m["_merge"] == "both"]
    r1["resource_mismatch"] = int((both["Project"] != both["Project_ref"]).sum())
    r1["smiles_mismatch"] = int((both[DRUG] != both[f"{DRUG}_ref"]).sum())
    r1["max_abs_value_diff"] = float((both[Y] - both[f"{Y}_ref"]).abs().max())
    print("R1", r1, flush=True)
    if (r1["only_rebuilt"] or r1["only_processed"] or r1["resource_mismatch"]
            or r1["smiles_mismatch"] or r1["max_abs_value_diff"] != 0.0):
        raise AssertionError("R1: the rebuilt rows differ from the processed table")

    # ---- R2: the frozen panel and its T02 row
    elig = Eligibility(cfg)
    frozen = mm.build_panel(cfg, elig.transcriptome)
    rebuilt = elig(pair_means(resolved))
    f = frozen.sort_values([DRUG, SAMPLE]).reset_index(drop=True)
    g = rebuilt.sort_values([DRUG, SAMPLE]).reset_index(drop=True)
    same_keys = len(f) == len(g) and bool((f[[DRUG, SAMPLE]].values == g[[DRUG, SAMPLE]].values).all())
    r2 = {"pairs_frozen": int(len(f)), "pairs_rebuilt": int(len(g)), "same_keys": same_keys,
          "max_abs_value_diff": float(np.abs(f[Y].values - g[Y].values).max()) if same_keys else None}
    print("R2 panel", r2, flush=True)
    if not same_keys or r2["max_abs_value_diff"] > PAIR_TOL:
        raise AssertionError("R2: the rebuilt panel differs from the frozen panel")

    rows = []
    bench = decompose(frozen, "benchmark", "all", n_rows=len(elig(resolved)))
    for col in ("n_pairs", "n_drugs", "n_samples", "components", "rank_H"):
        if bench[col] != int(t02[col]):
            raise AssertionError(f"R2: {col} {bench[col]} differs from T02 {t02[col]}")
    worst = 0.0
    for col in ("d_H_over_N", "var_y", "share_shared_pct", "share_interaction_pct",
                "share_drug_unique_pct", "share_sample_unique_pct",
                "share_drug_sample_overlap_pct"):
        rel = abs(bench[col] - float(t02[col])) / abs(float(t02[col]))
        worst = max(worst, rel)
        if rel > T02_RTOL:
            raise AssertionError(f"R2: {col} {bench[col]} differs from T02 {t02[col]}")
    r2["t02_worst_relative_diff"] = worst
    rows.append(bench)

    # ---- R3: T01 per-resource composition
    counts = elig(resolved)["Project"].value_counts()
    r3 = {r: {"rebuilt": int(counts.get(r, 0)), "T01": int(t01[f"response_rows_{r}"])}
          for r in RESOURCES}
    print("R3", r3, flush=True)
    if any(v["rebuilt"] != v["T01"] for v in r3.values()):
        raise AssertionError("R3: per-resource rows differ from T01")

    # ---- the benchmark without NCI60, by the same rules
    no_nci = resolve_overlaps(mapped[mapped["Project"] != "NCI60"])
    no_nci_rows = elig(no_nci)
    rows.append(decompose(elig(pair_means(no_nci)), "benchmark_without_NCI60", "all",
                          n_rows=len(no_nci_rows)))

    # ---- each resource on its own support
    screen_rows = elig(mapped)
    bench_rows = elig(resolved)
    for r in RESOURCES:
        own = screen_rows[screen_rows["Project"] == r]
        rows.append(decompose(pair_means(own), "resource_screen", r, n_rows=len(own)))
        won = bench_rows[bench_rows["Project"] == r]
        rows.append(decompose(pair_means(won), "resource_benchmark_rows", r, n_rows=len(won)))

    # ---- write
    a.out_dir.mkdir(parents=True, exist_ok=True)
    table = a.out_dir / "T26_resource_decomposition.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    lost = {r: int((screen_rows["Project"] == r).sum() - (bench_rows["Project"] == r).sum())
            for r in RESOURCES}
    # every gate above raises before this point, so reaching it is the pass
    audit = {
        "status": "pass",
        "generator": "build_T26_resource_decomposition.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "sources": {
            "config": {"file": str(a.config), "sha256": sha256_file(a.config)},
            "droma_sqlite": {"file": str(a.droma), "sha256": sha256_file(a.droma)},
            "drug_response": {"file": cfg["drug_response_path"],
                              "sha256": sha256_file(Path(cfg["drug_response_path"]))},
            "T01_sha256": registry["T01"]["sha256"], "T02_sha256": registry["T02"]["sha256"]},
        "resources": list(RESOURCES),
        "overlap_rule": "one row per (original drug name, sample); first resource in string order",
        "R1_processed_table": r1, "R2_frozen_panel_and_T02": r2, "R3_T01_composition": r3,
        "eligible_rows_resolved_to_another_resource": lost,
        "eligible_rows_resolved_to_another_resource_total": int(sum(lost.values())),
        "new_table_sha256": sha256_file(table),
        "seconds": round(time.time() - t0, 1),
    }
    (a.out_dir / "results.json").write_text(json.dumps(audit, indent=1) + "\n")
    print(f"wrote {table} and results.json in {audit['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

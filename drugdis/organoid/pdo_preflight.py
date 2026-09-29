"""PDO cross-system stress test, pre-flight assertion.

The frozen models were trained on a CCLE-anchored transcriptome. Applying them to
a PDO cohort measures cross-system generalization only if the PDO expression is on
the same numeric scale as the training input. Otherwise it measures the scale
mismatch, which is the failure mode Option A' was chosen to eliminate.

This asserts scale comparability per PDO resource before anything is run.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

CL = str(paths.data_dir()) + "/"
RAW = CL + "omics_mrna_raw/"
FROZEN = CL + "omics_baseline_frozen/baseline_mrna_ccle_anchored.parquet"
GENELIST = CL + "omics_baseline/baseline_gene_list.txt"

genes = pd.Index([l.strip() for l in open(GENELIST) if l.strip()])
ccle = pd.read_parquet(FROZEN, engine="fastparquet")
cv = ccle.to_numpy(np.float32)
ref = {"mean": float(np.nanmean(cv)), "sd": float(np.nanstd(cv)),
       "median": float(np.nanmedian(cv)), "min": float(np.nanmin(cv)),
       "max": float(np.nanmax(cv)), "zero": float(np.nanmean(cv == 0)),
       "level_mean": float(np.nanmean(cv, axis=1).mean()),
       "level_sd": float(np.nanmean(cv, axis=1).std())}
print("TRAINING INPUT, CCLE-anchored reference")
print(f"  mean {ref['mean']:.3f} sd {ref['sd']:.3f} median {ref['median']:.3f} "
      f"range [{ref['min']:.2f},{ref['max']:.2f}] zeros {ref['zero']:.3f} "
      f"per-sample level {ref['level_mean']:.3f} +/- {ref['level_sd']:.3f}")

mt = pd.read_parquet(CL + "master_table.parquet", engine="fastparquet")
dr = pd.read_parquet(CL + "drug_response.parquet", engine="fastparquet")
pdo_ids = set(mt.index[mt["Model_Type"] == "PDO"])
print(f"\nPDO samples in master_table: {len(pdo_ids)}")

rows = {}
print(f"\n{'resource':10s} {'n':>5s} {'mean':>8s} {'sd':>7s} {'median':>8s} {'min':>8s} "
      f"{'max':>8s} {'zeros':>7s} {'level':>8s} {'lvl sd':>7s} {'verdict':>14s}")
print(f"{'CCLE(ref)':10s} {len(ccle):5d} {ref['mean']:8.3f} {ref['sd']:7.3f} "
      f"{ref['median']:8.3f} {ref['min']:8.2f} {ref['max']:8.2f} {ref['zero']:7.3f} "
      f"{ref['level_mean']:8.3f} {ref['level_sd']:7.3f} {'reference':>14s}")
for f in sorted(glob.glob(RAW + "*.parquet")):
    r = os.path.basename(f)[:-8]
    d = pd.read_parquet(f, engine="fastparquet")
    d.index = d.index.astype(str)
    d = d[d.index.isin(pdo_ids)]
    if not len(d):
        continue
    cols = genes.intersection(d.columns)
    v = d[cols].to_numpy(np.float32)
    lvl = np.nanmean(v, axis=1)
    m, sd, med = float(np.nanmean(v)), float(np.nanstd(v)), float(np.nanmedian(v))
    z = float(np.nanmean(v == 0))
    # comparability: mean level within 1 sd of the reference gene-level spread,
    # zero fraction within a factor of 2, and no negatives beyond the reference min
    ok = (abs(lvl.mean() - ref["level_mean"]) < 1.0
          and 0.5 * ref["zero"] <= z <= 2.0 * ref["zero"]
          and float(np.nanmin(v)) >= ref["min"] - 0.5)
    rows[r] = {"n": int(len(d)), "genes": int(len(cols)), "mean": m, "sd": sd,
               "median": med, "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
               "zero_frac": z, "level_mean": float(lvl.mean()),
               "level_sd": float(lvl.std()), "comparable_to_ccle": bool(ok),
               "response_rows": int(dr.Sample_ID.isin(d.index).sum())}
    print(f"{r:10s} {len(d):5d} {m:8.3f} {sd:7.3f} {med:8.3f} {np.nanmin(v):8.2f} "
          f"{np.nanmax(v):8.2f} {z:7.3f} {lvl.mean():8.3f} {lvl.std():7.3f} "
          f"{'COMPARABLE' if ok else 'OFF-SCALE':>14s}")
print("\nresponse rows per PDO resource:")
for r, v in rows.items():
    print(f"  {r:10s} samples {v['n']:4d}  response rows {v['response_rows']:7,d}  "
          f"{'comparable' if v['comparable_to_ccle'] else 'OFF-SCALE'}")
json.dump({"reference": ref, "resources": rows},
          open(paths.work_dir() / "pdo_preflight.json", "w"), indent=1)
n_off = sum(1 for v in rows.values() if not v["comparable_to_ccle"])
print(f"\nVERDICT: {len(rows) - n_off} of {len(rows)} PDO resources comparable to the "
      f"training input; {n_off} off-scale.")

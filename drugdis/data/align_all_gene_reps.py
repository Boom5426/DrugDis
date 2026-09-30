"""Align every transcriptome representation to the frozen CCLE-anchored substrate.

Same verified rule as the decoder-robustness pre-flight: the embedding files were
computed on the old 2,923-row baseline and their leading block is the CCLE file
order. Each file is checked individually before anything is written; a file that
does not satisfy the check is reported and skipped rather than guessed at.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

CL = str(paths.data_dir()) + "/"
GEN = CL + "Gene_Embeddings/"
OUT = str(paths.work_dir() / "aligned_ccle") + "/"
SKIP = {"Baseline"}          # raw expression comes from the frozen matrix itself

ccle_raw = pd.read_parquet(CL + "omics_mrna_raw/CCLE.parquet", engine="fastparquet")
order = ccle_raw.index.astype(str).to_numpy()
base = pd.read_parquet(CL + "omics_baseline/baseline_mrna_common_genes.parquet",
                       engine="fastparquet")
base_ids = base.index.astype(str).to_numpy()
frozen = pd.read_parquet(CL + "omics_baseline_frozen/baseline_mrna_ccle_anchored.parquet",
                         engine="fastparquet")
frozen_ids = frozen.index.astype(str)
print(f"CCLE block {len(order)} rows; frozen substrate {len(frozen_ids)} samples\n")

os.makedirs(OUT, exist_ok=True)
rec, skipped = {}, []
print(f"{'representation':14s} {'rows':>6s} {'dim':>6s} {'multiset':>9s} {'CCLE block':>11s} "
      f"{'NaN':>8s} {'status':>10s}")
for f in sorted(glob.glob(GEN + "*_embeddings.parquet")):
    name = os.path.basename(f).replace("_embeddings.parquet", "")
    if name in SKIP:
        continue
    e = pd.read_parquet(f, engine="fastparquet")
    if "feature_id" not in e.columns:
        skipped.append((name, "no feature_id column")); print(f"{name:14s} SKIPPED: no feature_id")
        continue
    fid = e["feature_id"].astype(str).to_numpy()
    same_multiset = sorted(fid) == sorted(base_ids)
    block = bool(len(fid) >= len(order) and (fid[:len(order)] == order).all())
    cols = [c for c in e.columns if c != "feature_id"]
    if not (same_multiset and block):
        skipped.append((name, f"multiset={same_multiset} block={block}"))
        print(f"{name:14s} {len(e):6d} {len(cols):6d} {str(same_multiset):>9s} {str(block):>11s} "
              f"{'':>8s} {'SKIPPED':>10s}")
        continue
    m = e.iloc[[i for i in range(len(order))]][cols].astype(np.float32)
    m.index = pd.Index(order, name="Sample_ID")
    m = m.loc[frozen_ids]
    v = m.to_numpy(np.float32)
    nan = int(np.isnan(v).sum())
    if nan:
        m = m.fillna(0.0)
    p = OUT + f"{name}_ccle_aligned.parquet"
    m.to_parquet(p, engine="fastparquet", compression="snappy")
    rec[name] = {"path": p, "n": int(len(m)), "dim": int(m.shape[1]), "nan_filled": nan,
                 "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v))}
    print(f"{name:14s} {len(e):6d} {len(cols):6d} {'True':>9s} {'True':>11s} {nan:8d} {'written':>10s}")

json.dump({"aligned": rec, "skipped": skipped},
          open(paths.work_dir() / "gene_rep_alignment_all.json", "w"), indent=1)
print(f"\n{len(rec)} representations aligned, {len(skipped)} skipped")
if skipped:
    for n, why in skipped:
        print(f"  skipped {n}: {why}")

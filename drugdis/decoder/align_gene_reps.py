"""Align the transcriptome embeddings to the frozen CCLE-anchored substrate.

The embeddings were computed on the old 2,923-row baseline, which is a
concatenation of ten resources and contains 708 duplicated Sample_IDs. The frozen
substrate uses the CCLE profile for every sample, so the embedding used here must
be the one computed from that same CCLE row. Otherwise the raw-versus-embedding
comparison would confound representation with source profile.

This locates, for each frozen-panel sample, the old-baseline row that carries its
CCLE profile, and writes a Sample_ID-indexed embedding for each model.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

CL = str(paths.data_dir()) + "/"
GEN = CL + "Gene_Embeddings/"
OUTDIR = str(paths.work_dir() / "aligned_ccle") + "/"
REPS = ("scGPT", "CellPLM", "BulkFormer")

base = pd.read_parquet(CL + "omics_baseline/baseline_mrna_common_genes.parquet",
                       engine="fastparquet")
base.index = base.index.astype(str)
ccle = pd.read_parquet(CL + "omics_baseline_frozen/baseline_mrna_ccle_anchored.parquet",
                       engine="fastparquet")
ccle.index = ccle.index.astype(str)
genes = list(ccle.columns)
print(f"old baseline {base.shape}, frozen CCLE {ccle.shape}")

# 1. The embeddings carry the same multiset of Sample_IDs as the old baseline but
#    in the per-resource file order, which begins with the CCLE block. Verified
#    against CCLE.parquet rather than assumed.
ccle_raw = pd.read_parquet(CL + "omics_mrna_raw/CCLE.parquet", engine="fastparquet")
ccle_order = ccle_raw.index.astype(str).to_numpy()
ok_align = {}
for r in REPS:
    e = pd.read_parquet(GEN + f"{r}_embeddings.parquet", engine="fastparquet")
    fid = e["feature_id"].astype(str).to_numpy()
    same_multiset = sorted(fid) == sorted(base.index.to_numpy())
    ccle_block = bool(len(fid) >= len(ccle_order)
                      and (fid[:len(ccle_order)] == ccle_order).all())
    ok_align[r] = {"same_multiset_as_old_baseline": same_multiset,
                   "leading_block_is_ccle_in_file_order": ccle_block}
    print(f"  {r:11s} same multiset {same_multiset}, leading {len(ccle_order)} rows are "
          f"the CCLE block in file order: {ccle_block}")
assert all(v["leading_block_is_ccle_in_file_order"] and v["same_multiset_as_old_baseline"]
           for v in ok_align.values()), "embedding rows are not the expected CCLE-first order"

# 2. the CCLE-derived embedding of a frozen sample is its row in that leading block
row_of = {s: i for i, s in enumerate(ccle_order)}
missing = [s for s in ccle.index if s not in row_of]
print(f"\nCCLE-derived embedding rows located: {len(ccle) - len(missing):,} of {len(ccle):,}; "
      f"missing {len(missing)}")
assert not missing, missing[:5]
unmatched = []

# 3. write Sample_ID-indexed embeddings restricted to those rows
os.makedirs(OUTDIR, exist_ok=True)
rec = {"alignment_checks": ok_align, "n_ccle_rows_located": len(row_of), "n_unmatched": len(unmatched)}
for r in REPS:
    e = pd.read_parquet(GEN + f"{r}_embeddings.parquet", engine="fastparquet")
    cols = [c for c in e.columns if c != "feature_id"]
    ids = [s for s in ccle.index]
    m = e.iloc[[row_of[s] for s in ids]][cols].astype(np.float32)
    m.index = pd.Index(ids, name="Sample_ID")
    p = OUTDIR + f"{r}_ccle_aligned.parquet"
    m.to_parquet(p, engine="fastparquet", compression="snappy")
    v = m.to_numpy(np.float32)
    rec[r] = {"path": p, "shape": list(m.shape), "dim": int(m.shape[1]),
              "nan": int(np.isnan(v).sum()), "mean": float(np.nanmean(v)),
              "sd": float(np.nanstd(v))}
    print(f"  wrote {r:11s} {m.shape}  mean {np.nanmean(v):+.4f} sd {np.nanstd(v):.4f} "
          f"NaN {int(np.isnan(v).sum())}")
json.dump(rec, open(paths.work_dir() / "gene_rep_alignment.json", "w"), indent=1)
print("\nwrote gene_rep_alignment.json")

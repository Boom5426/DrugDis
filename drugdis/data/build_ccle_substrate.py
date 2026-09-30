"""Option A' : CCLE-anchored single-modality transcriptomic substrate.

Gating assertion first: CCLE must carry at most one usable baseline profile per
Sample_ID after the existing gene whitelist. If that fails with genuinely distinct
within-CCLE duplicates, this stops and reports the single blocker.

Then it writes the frozen transcriptome artefact to a NEW path. Nothing under
`omics_baseline/` is modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

CL = str(paths.data_dir()) + "/"
CCLE = CL + "omics_mrna_raw/CCLE.parquet"
GENELIST = CL + "omics_baseline/baseline_gene_list.txt"
# DRUGDIS_CCLE_OUT redirects the output, e.g. to rebuild into a scratch directory
# and compare with the published file.
OUT_DIR = os.environ.get("DRUGDIS_CCLE_OUT", CL + "omics_baseline_frozen").rstrip("/") + "/"
OUT = OUT_DIR + "baseline_mrna_ccle_anchored.parquet"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    genes = [l.strip() for l in open(GENELIST) if l.strip()]
    print(f"existing gene whitelist: {len(genes)} genes")

    d = pd.read_parquet(CCLE, engine="fastparquet")
    print(f"CCLE raw: {d.shape}")

    # ---------------- gating assertion ----------------
    idx = pd.Index(d.index.astype(str))
    n_rows, n_ids = len(idx), idx.nunique()
    print(f"\n=== GATE: at most one usable CCLE profile per Sample_ID ===")
    print(f"  rows {n_rows}, distinct Sample_ID {n_ids}")
    if n_rows != n_ids:
        vc = idx.value_counts()
        dups = vc[vc > 1]
        v = d.to_numpy(np.float64)
        pos = {}
        for i, k in enumerate(idx):
            pos.setdefault(k, []).append(i)
        distinct = [k for k in dups.index
                    if not all(np.array_equal(np.nan_to_num(v[j]), np.nan_to_num(v[pos[k][0]]))
                               for j in pos[k])]
        print(f"  duplicated Sample_ID: {len(dups)}; of those genuinely distinct: {len(distinct)}")
        if distinct:
            print("\nBLOCKER: CCLE carries genuinely distinct duplicate profiles for "
                  f"{len(distinct)} Sample_ID. Reporting and stopping, per instruction.")
            print("  examples:", distinct[:10])
            return 2
        print("  duplicates are exact copies; collapsing them is lossless.")
        d = d[~idx.duplicated(keep="first")]
    print("  GATE PASSED: one usable CCLE profile per Sample_ID.")

    # ---------------- restrict to the existing whitelist ----------------
    missing = [g for g in genes if g not in d.columns]
    if missing:
        print(f"\nBLOCKER: {len(missing)} whitelist genes absent from CCLE, e.g. {missing[:5]}")
        return 2
    m = d[genes].astype(np.float32)
    m.index = m.index.astype(str)
    nan_frac = float(np.isnan(m.to_numpy()).mean())
    v = m.to_numpy(np.float32)
    print(f"\nfrozen matrix: {m.shape}  NaN fraction {nan_frac:.6f}")
    print(f"  value range [{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]  mean {np.nanmean(v):.4f}  "
          f"sd {np.nanstd(v):.4f}  frac exact zero {float(np.nanmean(v == 0)):.4f}")
    lvl = np.nanmean(v, axis=1)
    print(f"  per-sample mean level: min {lvl.min():.4f} max {lvl.max():.4f} sd {lvl.std():.4f}")
    assert m.index.is_unique
    assert (np.nanmax(v) < 100), "single-scale assertion failed: a value exceeds the log-scale range"

    os.makedirs(OUT_DIR, exist_ok=True)
    m.to_parquet(OUT, engine="fastparquet", compression="snappy")
    rec = {"source": os.path.relpath(CCLE, CL), "source_sha256": sha256_file(CCLE),
           "gene_list": os.path.relpath(GENELIST, CL),
           "gene_list_sha256": sha256_file(GENELIST),
           "n_genes": len(genes), "n_samples": int(len(m)),
           "output": os.path.relpath(OUT, CL), "output_sha256": sha256_file(OUT),
           "nan_fraction": nan_frac,
           "value_min": float(np.nanmin(v)), "value_max": float(np.nanmax(v)),
           "per_sample_level_sd": float(lvl.std())}
    with open(OUT_DIR + "ccle_anchored_provenance.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {OUT}\n  sha256 {rec['output_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Decoder-robustness pre-flight: does every representation cover the frozen panel?

Rankings are only comparable if every representation is evaluated on the same
rows. This measures coverage and reports the intersection panel that follows.
Nothing is chosen here beyond what comparability forces.
"""
from __future__ import annotations
import json, os, pickle, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
sys.path.insert(0, str(paths.CODE / "splits"))
import make_manifests as mm  # noqa: E402

CFG = str(paths.CONFIG)
MOL = str(paths.data_dir() / "Molecule_Embeddings") + "/"
GEN = str(paths.data_dir() / "Gene_Embeddings") + "/"
DRUG_REPS = {"ECFP4": MOL + "ECFP4_emb2048.pickle", "MolCLR": MOL + "MolCLR_emb512.pickle",
             "KPGT": MOL + "KPGT_emb2304.pickle", "UniMol": MOL + "UniMol_emb512.pickle"}
GENE_REPS = {"raw": None, "scGPT": GEN + "scGPT_embeddings.parquet",
             "CellPLM": GEN + "CellPLM_embeddings.parquet",
             "BulkFormer": GEN + "BulkFormer_embeddings.parquet"}

cfg = paths.load_config(CFG)
dg = mm.load_transcriptome(cfg)
panel = mm.build_panel(cfg, dg)
drugs = set(panel["SMILES"].astype(str))
samples = set(panel["Sample_ID"].astype(str))
print(f"frozen panel: {len(panel):,} pairs, {len(drugs):,} drugs, {len(samples)} samples\n")

out = {"panel": {"pairs": int(len(panel)), "drugs": len(drugs), "samples": len(samples)}}
print(f"{'drug rep':10s} {'keys':>9s} {'dim':>6s} {'covers panel drugs':>20s}")
dcov = {}
for n, p in DRUG_REPS.items():
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    k = set(map(str, d.keys()))
    dim = int(np.asarray(next(iter(d.values()))).shape[0])
    c = drugs & k
    dcov[n] = c
    print(f"{n:10s} {len(d):9,d} {dim:6d} {len(c):>13,d} ({100*len(c)/len(drugs):5.1f}%)")
    out.setdefault("drug", {})[n] = {"keys": len(d), "dim": dim, "covered": len(c),
                                     "frac": len(c) / len(drugs)}
    del d

print(f"\n{'gene rep':11s} {'rows':>7s} {'dim':>6s} {'covers panel samples':>22s}")
gcov = {"raw": samples}
for n, p in GENE_REPS.items():
    if p is None:
        print(f"{'raw':11s} {len(dg):7,d} {dg.shape[1]:6,d} {len(samples):>15,d} (100.0%)  frozen CCLE-anchored")
        out.setdefault("gene", {})[n] = {"rows": int(len(dg)), "dim": int(dg.shape[1]),
                                         "covered": len(samples), "frac": 1.0}
        continue
    e = pd.read_parquet(p, engine="fastparquet")
    e.index = e.index.astype(str)
    k = set(e.index)
    c = samples & k
    gcov[n] = c
    print(f"{n:11s} {len(e):7,d} {e.shape[1]:6,d} {len(c):>15,d} ({100*len(c)/len(samples):5.1f}%)")
    out.setdefault("gene", {})[n] = {"rows": int(len(e)), "dim": int(e.shape[1]),
                                     "covered": len(c), "frac": len(c) / len(samples)}
    del e

d_int = set.intersection(*dcov.values())
g_int = set.intersection(*gcov.values())
sub = panel[panel["SMILES"].astype(str).isin(d_int) & panel["Sample_ID"].astype(str).isin(g_int)]
print(f"\n=== intersection panel, the only one on which rankings are comparable ===")
print(f"  drugs common to all four drug representations : {len(d_int):,} of {len(drugs):,}")
print(f"  samples common to all four gene representations: {len(g_int):,} of {len(samples):,}")
print(f"  pairs retained: {len(sub):,} of {len(panel):,} ({100*len(sub)/len(panel):5.1f}%)")
out["intersection"] = {"drugs": len(d_int), "samples": len(g_int), "pairs": int(len(sub))}
json.dump(out, open(paths.work_dir() / "coverage_preflight.json", "w"), indent=1)
json.dump({"drugs": sorted(d_int), "samples": sorted(g_int)},
          open(paths.work_dir() / "intersection_entities.json", "w"))
print("\nwrote coverage_preflight.json and intersection_entities.json")

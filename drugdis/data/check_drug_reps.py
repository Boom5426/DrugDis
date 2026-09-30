import glob, json, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
sys.path.insert(0, str(paths.CODE / "splits"))
import make_manifests as mm  # noqa: E402
cfg = paths.load_config(paths.CONFIG)
dg = mm.load_transcriptome(cfg)
panel = mm.build_panel(cfg, dg)
drugs = set(panel["SMILES"].astype(str))
print(f"frozen panel drugs: {len(drugs):,}\n{'representation':14s} {'keys':>9s} {'dim':>6s} {'coverage':>10s}")
out = {}
for f in sorted(glob.glob(str(paths.data_dir() / "Molecule_Embeddings" / "*.pickle"))):
    n = os.path.basename(f).split("_emb")[0]
    if os.path.islink(f):
        continue
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    k = set(map(str, d.keys()))
    c = len(drugs & k)
    dim = int(np.asarray(next(iter(d.values()))).shape[0])
    out[n] = {"path": f, "dim": dim, "covered": c, "frac": c / len(drugs)}
    print(f"{n:14s} {len(d):9,d} {dim:6d} {c:7,d} ({100*c/len(drugs):5.1f}%)")
    del d
json.dump(out, open(paths.work_dir() / "drug_rep_coverage.json", "w"), indent=1)
full = [n for n, v in out.items() if v["frac"] == 1.0]
print(f"\n{len(full)} of {len(out)} representations cover the panel completely: {full}")

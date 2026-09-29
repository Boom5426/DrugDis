"""T16: per-configuration decoder profiles from the frozen decoder-robustness run.

T09 carries only the ranking comparisons. Fig. 3f to 3h need the profiles those
rankings were computed from, in particular the amplitude of interaction a decoder
emits at all: a structurally additive decoder emits none, so any interaction
ranking taken from one carries no information. Promoting the profiles to a
canonical table is what lets a panel read them through the sanctioned channel.

Reshaping only. Nothing is recomputed, no configuration is re-run, and the
selected weight decay is the one the frozen run selected on validation.
"""
from __future__ import annotations
import glob, hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

ROOT = str(paths.runs_dir() / "DecoderBench")
T = str(paths.tables_dir()) + "/"
DRUGS = {"ECFP4", "MolCLR", "KPGT", "UniMol"}
GENES = {"raw", "scGPT", "CellPLM", "BulkFormer"}

rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    f = os.path.join(d, "results.json")
    if not os.path.exists(f):
        continue
    name = os.path.basename(d)
    parts = name.split("__")
    if len(parts) < 4:
        continue
    drug, gene, decoder, tail = parts[0], parts[1], parts[2], parts[3]
    seed = int(tail.replace("seed", ""))
    wd = parts[4] if len(parts) > 4 else "selected"
    r = json.load(open(f))
    prof = r.get("frozen_valMSE__test")
    if prof is None:
        continue
    rows.append({"drug_rep": drug, "gene_rep": gene, "decoder": decoder,
                 "seed": seed, "weight_decay": wd,
                 "n_params": r.get("n_params"), "best_epoch": r.get("best_epoch"),
                 **{k: prof.get(k) for k in ("n", "rawPCC", "sharedPCC", "intPCC",
                                             "A_int", "MSE_raw", "E_shared",
                                             "E_interaction", "sd_M_yhat")}})

df = pd.DataFrame(rows)
print(f"{len(df)} configuration runs found")
print(df.groupby(["decoder"]).size().to_string())
print("\ncolumns present:", [c for c in df.columns if df[c].notna().any()])
p = T + "T16_decoder_profiles.csv"
df.to_csv(p, index=False)
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
reg = json.load(open(T + "TABLES.json"))
reg["T16"] = {"file": "T16_decoder_profiles.csv", "rows": len(df), "sha256": h}
json.dump({k: reg[k] for k in sorted(reg)}, open(T + "TABLES.json", "w"), indent=1)
print(f"\nwrote T16_decoder_profiles.csv  {len(df)} rows  sha {h[:16]}")

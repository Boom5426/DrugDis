"""Is a representation ranking resolvable at all? Compare the spread across
representations with the seed-to-seed noise of a single representation."""
import json, os, sys
import pandas as pd, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

T = str(paths.tables_dir()) + "/"
out = []
print(f"{'split':6s} {'side':14s} {'metric':10s} {'spread':>9s} {'median seed sd':>15s} "
      f"{'ratio':>7s} {'verdict':>16s}")
for sp, f in (("LCLO", "T05_lclo_profile.csv"), ("LSO", "T06_lso_profile.csv")):
    d = pd.read_csv(T + f)
    for side in ("drug", "transcriptome"):
        s = d[(d.side == side) & (d.full_panel_coverage == True)]
        for m in ("rawPCC", "sharedPCC", "intPCC"):
            spread = s[f"{m}_mean"].max() - s[f"{m}_mean"].min()
            sd = s[f"{m}_std"].median()
            ratio = spread / sd if sd else np.nan
            v = "separable" if ratio > 4 else ("marginal" if ratio > 2 else "NOT separable")
            print(f"{sp:6s} {side:14s} {m:10s} {spread:9.4f} {sd:15.4f} {ratio:7.1f} {v:>16s}")
            out.append({"split": sp, "side": side, "metric": m, "spread": float(spread),
                        "median_seed_sd": float(sd), "ratio": float(ratio), "verdict": v})
pd.DataFrame(out).to_csv(T + "T12_ranking_separability.csv", index=False)
import hashlib
h = hashlib.sha256(open(T + "T12_ranking_separability.csv", "rb").read()).hexdigest()
reg = json.load(open(T + "TABLES.json"))
reg["T12"] = {"file": "T12_ranking_separability.csv", "rows": len(out), "sha256": h}
json.dump(reg, open(T + "TABLES.json", "w"), indent=1)
print(f"\nwrote T12_ranking_separability.csv  sha {h[:16]}")

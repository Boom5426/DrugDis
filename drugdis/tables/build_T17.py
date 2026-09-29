"""T17: the distribution of the response and of its two components.

Fig. 1 reports what share of variance each component carries but never shows the
quantities themselves. A reader therefore takes the decomposition on the strength
of two percentages. This table holds the histogram of the total response and of
its two components on the frozen benchmark dataset, so a panel can show that the
additive component reproduces the shape of the response while the interaction
component is a narrow symmetric residual around zero.

No model is involved and nothing is resampled. This is a deterministic summary of
the frozen benchmark dataset, in the same class as T02 and T14.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
sys.path.insert(0, str(paths.CODE / "splits"))
sys.path.insert(0, str(paths.CODE / "decomposition"))
import make_manifests as mm                                  # noqa: E402
from decompositions import DRUG, SAMPLE, AdditiveProjection  # noqa: E402

T = str(paths.tables_dir()) + "/"
NBINS = 60
LO, HI = -0.55, 1.05          # fixed, so the three histograms share one axis


def main() -> int:
    cfg = paths.load_config(paths.CONFIG)
    panel = mm.build_panel(cfg, mm.load_transcriptome(cfg))
    y = panel["Sensitivity"].to_numpy(np.float64)
    proj = AdditiveProjection(panel, [DRUG, SAMPLE])
    My = proj.residual(y)
    Hy = y - My
    print(f"n = {len(y):,}")

    edges = np.linspace(LO, HI, NBINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for name, v in (("total_response", y), ("additive", Hy), ("interaction", My)):
        cnt, _ = np.histogram(v, bins=edges)
        outside = int((v < LO).sum() + (v > HI).sum())
        for c, k in zip(centres, cnt):
            rows.append({"component": name, "bin_centre": round(float(c), 5),
                         "count": int(k), "density": float(k) / len(v)})
        print(f"  {name:15s} mean {v.mean():+.4f}  sd {v.std():.4f}  "
              f"min {v.min():+.3f}  max {v.max():+.3f}  outside the axis {outside:,}")
        rows.append({"component": name, "bin_centre": np.nan, "count": outside,
                     "density": float(outside) / len(v)})

    df = pd.DataFrame(rows)
    p = T + "T17_response_distributions.csv"
    df.to_csv(p, index=False)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    reg = json.load(open(T + "TABLES.json"))
    reg["T17"] = {"file": "T17_response_distributions.csv", "rows": int(len(df)),
                  "sha256": h}
    json.dump({k: reg[k] for k in sorted(reg)}, open(T + "TABLES.json", "w"), indent=1)
    print(f"\nwrote T17_response_distributions.csv  {len(df)} rows  sha {h[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

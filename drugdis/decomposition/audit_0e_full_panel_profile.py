"""Phase 0E, Task 2: the component recovery profile on the full evaluation panels.

The exact-support analysis answers "how far from the measurement reference".
This one answers "what does each model recover", on the whole benchmark panel,
and is the table that replaces the published raw-versus-dsPCC comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from decompositions import DRUG, SAMPLE, AdditiveProjection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

ROOTS = {"cell_line": str(paths.runs_dir() / "Results"),
         "pdo": str(paths.runs_dir() / "PDO_Results")}
TARGET, PRED = "Target_AAC", "Predicted_AAC"


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase0e_full_panel_profile.json")
    args = ap.parse_args()
    rows = []
    for dom, root in ROOTS.items():
        for setting in ("LSO", "LCLO"):
            for path in sorted(glob.glob(os.path.join(root, "*", setting, "test_predictions.csv"))):
                model = path.split(os.sep)[-3]
                try:
                    df = pd.read_csv(path, usecols=[DRUG, SAMPLE, TARGET, PRED]).dropna()
                except Exception as exc:
                    print(f"  SKIP {model}: {exc}", file=sys.stderr)
                    continue
                if len(df) < 100:
                    continue
                df = df.reset_index(drop=True)
                y = df[TARGET].to_numpy(np.float64)
                yh = df[PRED].to_numpy(np.float64)
                p = AdditiveProjection(df, [DRUG, SAMPLE])
                My, Myh = p.residual(y), p.residual(yh)
                Hy, Hyh = y - My, yh - Myh
                e = yh - y
                Me = p.residual(e)
                He = e - Me
                mse = float(np.mean(e ** 2))
                drug, _, gene = model.partition("__")
                rows.append({
                    "domain": dom, "setting": setting, "model": model,
                    "branch": "gene" if drug == "ECFP4" and gene != "baseline" else "drug",
                    "n": int(len(df)),
                    "rawPCC": corr(y, yh), "sharedPCC": corr(Hy, Hyh), "intPCC": corr(My, Myh),
                    "A_shared": float(Hyh.std() / Hy.std()) if Hy.std() else float("nan"),
                    "A_int": float(Myh.std() / My.std()) if My.std() else float("nan"),
                    "MSE_raw": mse,
                    "E_shared_frac": float(np.mean(He ** 2) / mse) if mse else float("nan"),
                    "E_interaction_frac": float(np.mean(Me ** 2) / mse) if mse else float("nan"),
                    "partition_rel_error": abs(mse - np.mean(He ** 2) - np.mean(Me ** 2)) / mse if mse else float("nan"),
                    "var_share_interaction_pct": float(100 * My.var() / y.var()) if y.var() else float("nan"),
                })
                print(f"[{dom}/{setting}] {model:>22} raw={rows[-1]['rawPCC']:.3f} "
                      f"shared={rows[-1]['sharedPCC']:.3f} int={rows[-1]['intPCC']:.3f}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {args.out} ({len(rows)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

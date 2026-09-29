"""Canonical tables T04, T05 and T06 from the frozen representation benchmark."""
from __future__ import annotations
import glob, hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

ROOT = str(paths.runs_dir() / "PaperRerun" / "ECFP4__baseline")
OUT = str(paths.tables_dir()) + "/"
SEEDS = (3407, 3408, 3409)
PARTIAL = {"drug-GeminiMol", "drug-UniMolV2"}
M0M4 = {"LCLO": str(paths.runs_dir() / "Phase2B_clean" / "ECFP4__baseline" / "LCLO"),
        "LSO": str(paths.runs_dir() / "Phase2B_LSO" / "ECFP4__baseline" / "LSO")}
KEYS = ("n", "rawPCC", "sharedPCC", "intPCC", "A_int", "A_shared",
        "MSE_raw", "E_shared", "E_interaction")


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write(df, name, reg):
    p = OUT + name
    df.to_csv(p, index=False)
    reg[name.split("_")[0]] = {"file": name, "rows": int(len(df)), "sha256": sha(p)}
    print(f"  wrote {name:34s} {len(df):5d} rows  sha {sha(p)[:16]}")


def main() -> int:
    rows = []
    for sp in ("LCLO", "LSO"):
        for f in sorted(glob.glob(os.path.join(ROOT, sp, "*__M0_seed*", "results.json"))):
            d = os.path.basename(os.path.dirname(f))
            tag, seed = d.split("__M0_seed")
            if not seed.isdigit():
                continue          # verification reruns such as _conccheck are not canonical
            r = json.load(open(f))
            t = r["frozen_valMSE__test"]
            kind, rep = tag.split("-", 1)
            rows.append({"side": "transcriptome" if kind == "gene" else "drug",
                         "representation": rep, "split": sp, "seed": int(seed),
                         "decoder": "nonlinear_dual_tower", "rule": "frozen_valMSE",
                         "full_panel_coverage": tag not in PARTIAL,
                         "drug_dim": r["args"].get("drug_rep_path") and -1 or -1,
                         "n_params": r["n_params"], "best_epoch": r["best_valmse_epoch"],
                         **{k: t[k] for k in KEYS}})
    if not rows:
        print("no benchmark runs found yet")
        return 1
    t4 = pd.DataFrame(rows).drop(columns=["drug_dim"])
    # The anchor (ECFP4, raw expression) is one fit that belongs to BOTH tables:
    # it is the ECFP4 row of the drug side and the raw row of the transcriptome
    # side. It is stored once as gene-raw and mirrored here so that neither
    # ranking is missing its anchor.
    mirror = t4[(t4.side == "transcriptome") & (t4.representation == "raw")].copy()
    mirror["side"] = "drug"
    mirror["representation"] = "ECFP4"
    mirror["anchor"] = True
    t4["anchor"] = t4.get("anchor", False)
    t4.loc[(t4.side == "transcriptome") & (t4.representation == "raw"), "anchor"] = True
    t4 = pd.concat([t4, mirror], ignore_index=True)
    reg = json.load(open(OUT + "TABLES.json"))
    write(t4, "T04_representation_benchmark.csv", reg)

    for sp, tname in (("LCLO", "T05_lclo_profile.csv"), ("LSO", "T06_lso_profile.csv")):
        sub = t4[t4.split == sp]
        agg = (sub.groupby(["side", "representation", "full_panel_coverage"])
               .agg(n_seeds=("seed", "nunique"), anchor=("anchor", "first"),
                    **{f"{m}_{s}": (m, s) for m in ("rawPCC", "sharedPCC", "intPCC", "A_int")
                       for s in ("mean", "std")},
                    E_shared_mean=("E_shared", "mean"),
                    E_interaction_mean=("E_interaction", "mean"),
                    n_rows=("n", "first"))
               .reset_index())
        extra = []
        for arm in ("M0", "M4"):
            v = []
            for sd in (3407, 3408, 3409, 3410, 3411):
                f = os.path.join(M0M4[sp], f"{arm}_seed{sd}", "results.json")
                if os.path.exists(f):
                    v.append(json.load(open(f))["frozen_valMSE__test"])
            if not v:
                continue
            e = {"side": "constructive", "representation": f"{arm} (ECFP4 x raw)",
                 "full_panel_coverage": True, "n_seeds": len(v),
                 "E_shared_mean": float(np.mean([x["E_shared"] for x in v])),
                 "E_interaction_mean": float(np.mean([x["E_interaction"] for x in v])),
                 "n_rows": v[0]["n"]}
            for m in ("rawPCC", "sharedPCC", "intPCC", "A_int"):
                a = np.array([x[m] for x in v])
                e[f"{m}_mean"] = float(a.mean()); e[f"{m}_std"] = float(a.std(ddof=1))
            extra.append(e)
        out = pd.concat([agg, pd.DataFrame(extra)], ignore_index=True)
        out = out.sort_values(["side", "intPCC_mean"], ascending=[True, False])
        write(out, tname, reg)
    json.dump(reg, open(OUT + "TABLES.json", "w"), indent=1)
    print(f"\nTABLES.json now holds {len(reg)} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())

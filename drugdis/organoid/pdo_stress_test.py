"""PDO cross-system stress test: zero-shot transfer of the frozen models.

The frozen M0 and M4 were trained on the CCLE-anchored cell-line substrate. Here
they are applied to the PDO cohorts with no PDO training, no PDO split, no
PDO-specific loss and no PDO-specific hyperparameter: the whole PDO panel is the
test set. That is the strongest reading of "zero PDO-specific tuning".

The pre-flight found that the PDO cohorts are not all on the training input's
scale, so every quantity is reported per resource as well as pooled. Nothing is
excluded here; the composition of the headline panel is a decision this script
supplies evidence for rather than takes.
"""
from __future__ import annotations
import glob, json, os, pickle, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
sys.path.insert(0, str(paths.CODE / "decomposition"))
sys.path.insert(0, str(paths.CODE / "models"))
from decompositions import DRUG, SAMPLE, AdditiveProjection   # noqa: E402
from models import CLIOSingleHead, CLIOFactorized             # noqa: E402

CL = str(paths.data_dir()) + "/"
RAW = CL + "omics_mrna_raw/"
GENELIST = CL + "omics_baseline/baseline_gene_list.txt"
CKPT_ROOT = str(paths.runs_dir() / "Phase2B_clean" / "ECFP4__baseline" / "LCLO")
MOL = str(paths.data_dir() / "Molecule_Embeddings" / "ECFP4_emb2048.pickle")
OUT = str(paths.work_dir() / "pdo_stress_test.json")
SEEDS = (3407, 3408, 3409, 3410, 3411)
COMPARABLE = ("UMPDO1", "UMPDO2", "UMPDO3")   # the log-scale family, see pre-flight


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() and b.std() else float("nan")


def profile(y, yh, proj):
    My, Myh = proj.residual(y), proj.residual(yh)
    Hy, Hyh = y - My, yh - Myh
    e = yh - y
    Me = proj.residual(e)
    He = e - Me
    return {"n": int(len(y)), "rawPCC": pcc(y, yh), "sharedPCC": pcc(Hy, Hyh),
            "intPCC": pcc(My, Myh),
            "A_int": float(Myh.std() / My.std()) if My.std() else float("nan"),
            "A_shared": float(Hyh.std() / Hy.std()) if Hy.std() else float("nan"),
            "MSE_raw": float(np.mean(e ** 2)), "E_shared": float(np.mean(He ** 2)),
            "E_interaction": float(np.mean(Me ** 2))}


def main() -> int:
    genes = pd.Index([l.strip() for l in open(GENELIST) if l.strip()])
    mt = pd.read_parquet(CL + "master_table.parquet", engine="fastparquet")
    pdo_ids = set(mt.index[mt["Model_Type"] == "PDO"].astype(str))

    frames, origin = [], {}
    cover = {}
    for f in sorted(glob.glob(RAW + "*.parquet")):
        r = os.path.basename(f)[:-8]
        d = pd.read_parquet(f, engine="fastparquet")
        d.index = d.index.astype(str)
        d = d[d.index.isin(pdo_ids)]
        if not len(d):
            continue
        cols = genes.intersection(d.columns)
        cover[r] = {"samples": int(len(d)), "genes_present": int(len(cols)),
                    "gene_coverage": float(len(cols) / len(genes))}
        vals = d[cols].astype(np.float32).to_numpy()
        n_nan = int(np.isnan(vals).sum())
        cover[r]["nan_entries"] = n_nan
        cover[r]["nan_frac_of_covered"] = float(n_nan / max(vals.size, 1))
        # the training pipeline fills missing expression with zero; same here
        m = pd.DataFrame(0.0, index=d.index, columns=genes, dtype=np.float32)
        m.loc[:, cols] = np.nan_to_num(vals, nan=0.0)
        frames.append(m)
        for s in d.index:
            origin[s] = r
    expr = pd.concat(frames)
    assert not np.isnan(expr.to_numpy(np.float32)).any(), "NaN survived the fill" 
    dup = expr.index.duplicated()
    print(f"PDO expression: {expr.shape}, duplicated Sample_ID across resources: {int(dup.sum())}")
    expr = expr[~dup]
    print("gene coverage of the 15,961 training genes, per resource:")
    for r, c in cover.items():
        print(f"  {r:8s} samples {c['samples']:4d}  genes {c['genes_present']:6,d} "
              f"({100*c['gene_coverage']:5.1f}%)  NaN entries filled with 0: "
              f"{c.get('nan_entries', 0):,} ({100*c.get('nan_frac_of_covered', 0):.2f}%)")

    dr = pd.read_parquet(CL + "drug_response.parquet", engine="fastparquet")
    resp = dr.groupby([DRUG, SAMPLE])["Sensitivity"].mean().reset_index()
    with open(MOL, "rb") as fh:
        drug_feat = pickle.load(fh)
    panel = resp[resp[DRUG].isin(drug_feat.keys()) & resp[SAMPLE].astype(str).isin(set(expr.index))]
    panel = panel.reset_index(drop=True)
    panel["resource"] = panel[SAMPLE].astype(str).map(origin)
    print(f"\nPDO evaluation panel: {len(panel):,} pairs, {panel[SAMPLE].nunique()} samples, "
          f"{panel[DRUG].nunique():,} drugs")
    print(panel.groupby("resource").agg(pairs=(DRUG, "size"), samples=(SAMPLE, "nunique"),
                                        drugs=(DRUG, "nunique")).to_string())

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    used_d = pd.Index(pd.unique(panel[DRUG]))
    used_s = pd.Index(pd.unique(panel[SAMPLE].astype(str)))
    D = torch.as_tensor(np.stack([np.asarray(drug_feat[k], np.float32) for k in used_d]), device=dev)
    G = torch.as_tensor(expr.loc[used_s].to_numpy(np.float32), device=dev)
    di = torch.as_tensor(pd.Series(np.arange(len(used_d)), index=used_d)
                         .loc[panel[DRUG]].to_numpy(), dtype=torch.long, device=dev)
    si = torch.as_tensor(pd.Series(np.arange(len(used_s)), index=used_s)
                         .loc[panel[SAMPLE].astype(str)].to_numpy(), dtype=torch.long, device=dev)
    y = panel["Sensitivity"].to_numpy(np.float64)

    groups = {"ALL": panel.index.to_numpy()}
    for r in sorted(panel["resource"].unique()):
        groups[r] = panel.index[panel["resource"] == r].to_numpy()
    groups["COMPARABLE(UMPDO1+2+3)"] = panel.index[panel["resource"].isin(COMPARABLE)].to_numpy()
    projs = {}
    for g, idx in groups.items():
        if len(idx) >= 50:
            projs[g] = AdditiveProjection(panel.loc[idx], [DRUG, SAMPLE])

    out = {"panel": {g: int(len(i)) for g, i in groups.items()}, "coverage": cover, "runs": {}}
    for arm in ("M0", "M4"):
        for seed in SEEDS:
            run = f"{arm}_seed{seed}"
            ck = os.path.join(CKPT_ROOT, run, "best_valmse.pth")
            if not os.path.exists(ck):
                continue
            fac = arm in ("M3", "M4", "M5")
            net = (CLIOFactorized(2048, len(genes), hidden_dim1=1024, shared_hidden=512,
                                  dropout_p=0.4) if fac
                   else CLIOSingleHead(2048, len(genes), hidden_dim1=1024, dropout_p=0.4))
            net.load_state_dict(torch.load(ck, map_location="cpu"))
            net = net.to(dev).eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(panel), 2048):
                    sl = slice(i, min(i + 2048, len(panel)))
                    o = net(D[di[sl]], G[si[sl]])
                    p = (o[0] + o[1]) if fac else o
                    preds.append(p.squeeze(-1).float().cpu())
            yh = torch.cat(preds).numpy().astype(np.float64)
            rec = {}
            for g, idx in groups.items():
                if g in projs:
                    rec[g] = profile(y[idx], yh[idx], projs[g])
            rec["pred_min"], rec["pred_max"] = float(yh.min()), float(yh.max())
            out["runs"][run] = rec
            a = rec.get("ALL", {})
            c = rec.get("COMPARABLE(UMPDO1+2+3)", {})
            print(f"{run:12s} ALL raw={a.get('rawPCC', float('nan')):+.4f} "
                  f"sh={a.get('sharedPCC', float('nan')):+.4f} int={a.get('intPCC', float('nan')):+.4f} "
                  f"| UMPDO raw={c.get('rawPCC', float('nan')):+.4f} "
                  f"sh={c.get('sharedPCC', float('nan')):+.4f} int={c.get('intPCC', float('nan')):+.4f} "
                  f"| pred [{yh.min():.2f},{yh.max():.2f}]", flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

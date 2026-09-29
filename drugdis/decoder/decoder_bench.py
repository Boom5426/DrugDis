"""Roadmap section 15: bounded decoder-robustness analysis.

Question: does the representation-level conclusion depend materially on decoder
class? Anchor design, not the full Cartesian product.

  drug side        transcriptome fixed at raw expression, drug in
                   {ECFP4, MolCLR, KPGT, UniMol}
  transcriptome    drug fixed at ECFP4, transcriptome in
                   {raw, scGPT, CellPLM, BulkFormer}

Three frozen decoder classes, trained by one identical protocol so that only the
decoder class differs:

  ridge     yhat = w_d.x_d + w_s.x_s + b
            structurally additive, so M yhat = 0 and it is a shared-only
            diagnostic baseline by construction, not a competitor
  bilinear  ridge plus a rank-32 bilinear term (U x_d).(V x_s)
  tower     the frozen nonlinear dual tower, unchanged from M0

Everything else is the frozen substrate: CCLE-anchored transcriptome, canonical
manifests, 30 epochs, Adam 1e-4, batch 2048, dropout 0.4, weight decay 1e-5, and
checkpoint = argmin validation MSE_raw.

One preprocessing decision is forced by the question itself and is applied
identically to every representation and every decoder: features are standardised
per dimension using TRAINING samples only. Ridge and the bilinear decoder have no
normalisation layer, so without this a representation would be ranked by the scale
of its embedding rather than its content, and the resulting "decoder-dependent
reversal" would be an artefact. Absolute values here are therefore not comparable
to sections 8, 9 and 9b of the substrate document.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_CODE, "decomposition"))
sys.path.insert(0, os.path.join(_CODE, "splits"))
sys.path.insert(0, os.path.join(_CODE, "models"))     # models.py and legacy utils.py
sys.path.insert(0, _CODE)

from utils import setup_seed                       # noqa: E402
from decompositions import AdditiveProjection      # noqa: E402
from models import CLIOSingleHead                  # noqa: E402
import make_manifests as mm                        # noqa: E402
import paths                                       # noqa: E402

CFG = str(paths.CONFIG)
MAN = str(paths.MANIFESTS)
MOL = str(paths.data_dir() / "Molecule_Embeddings") + "/"
ALIGNED = str(paths.work_dir() / "aligned_ccle") + "/"
DRUG_REPS = {"ECFP4": MOL + "ECFP4_emb2048.pickle", "MolCLR": MOL + "MolCLR_emb512.pickle",
             "KPGT": MOL + "KPGT_emb2304.pickle", "UniMol": MOL + "UniMol_emb512.pickle"}
GENE_REPS = {"raw": None, "scGPT": ALIGNED + "scGPT_ccle_aligned.parquet",
             "CellPLM": ALIGNED + "CellPLM_ccle_aligned.parquet",
             "BulkFormer": ALIGNED + "BulkFormer_ccle_aligned.parquet"}
BILINEAR_RANK = 32          # fixed a priori, never tuned
DRUG, SAMPLE = "SMILES", "Sample_ID"


class Ridge(nn.Module):
    """Additive: a function of the drug plus a function of the sample. Cannot
    express interaction, by construction."""

    def __init__(self, dd, gd):
        super().__init__()
        self.wd = nn.Linear(dd, 1, bias=False)
        self.ws = nn.Linear(gd, 1, bias=True)

    def forward(self, xd, xs):
        return self.wd(xd) + self.ws(xs)


class Bilinear(nn.Module):
    """Ridge plus a rank-r bilinear interaction term."""

    def __init__(self, dd, gd, rank=BILINEAR_RANK):
        super().__init__()
        self.wd = nn.Linear(dd, 1, bias=False)
        self.ws = nn.Linear(gd, 1, bias=True)
        self.U = nn.Linear(dd, rank, bias=False)
        self.V = nn.Linear(gd, rank, bias=False)

    def forward(self, xd, xs):
        return self.wd(xd) + self.ws(xs) + (self.U(xd) * self.V(xs)).sum(-1, keepdim=True)


class Idx(Dataset):
    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() and b.std() else float("nan")


def component_profile(y, yh, proj):
    My, Myh = proj.residual(y), proj.residual(yh)
    Hy, Hyh = y - My, yh - Myh
    e = yh - y
    Me = proj.residual(e)
    He = e - Me
    return {"n": int(len(y)), "rawPCC": pcc(y, yh), "sharedPCC": pcc(Hy, Hyh),
            "intPCC": pcc(My, Myh),
            "A_int": float(Myh.std() / My.std()) if My.std() else float("nan"),
            "MSE_raw": float(np.mean(e ** 2)), "E_shared": float(np.mean(He ** 2)),
            "E_interaction": float(np.mean(Me ** 2)),
            "sd_M_yhat": float(Myh.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug_rep", required=True, choices=list(DRUG_REPS))
    ap.add_argument("--gene_rep", required=True, choices=list(GENE_REPS))
    ap.add_argument("--decoder", required=True, choices=["ridge", "bilinear", "tower"])
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--n_epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--tag_extra", default="")
    ap.add_argument("--out_root", default=str(paths.runs_dir() / "DecoderBench") + "/")
    a = ap.parse_args()

    tag = f"{a.drug_rep}__{a.gene_rep}__{a.decoder}__seed{a.seed}"
    if a.tag_extra:
        tag += "__" + a.tag_extra
    out_dir = os.path.join(a.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"--- decoder bench | {a.drug_rep} x {a.gene_rep} | {a.decoder} | seed {a.seed} ---",
          flush=True)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(a.seed)
    cfg = paths.load_config(CFG)
    dg = mm.load_transcriptome(cfg)
    panel = mm.build_panel(cfg, dg)
    man = json.load(open(os.path.join(MAN, f"LCLO_seed{a.seed}.json")))
    parts = {k: panel[panel[SAMPLE].astype(str).isin(set(man[k]))].reset_index(drop=True)
             for k in ("train", "val", "test")}
    assert sum(len(v) for v in parts.values()) == len(panel)
    print(f"  pairs: " + " ".join(f"{k}={len(v):,}" for k, v in parts.items()), flush=True)

    with open(DRUG_REPS[a.drug_rep], "rb") as fh:
        dfeat = pickle.load(fh)
    used_d = pd.Index(pd.unique(panel[DRUG]))
    Dm = np.stack([np.asarray(dfeat[k], np.float32) for k in used_d])
    del dfeat
    if a.gene_rep == "raw":
        Gsrc = dg
    else:
        Gsrc = pd.read_parquet(GENE_REPS[a.gene_rep], engine="fastparquet")
        Gsrc.index = Gsrc.index.astype(str)
    used_s = pd.Index(pd.unique(panel[SAMPLE].astype(str)))
    Gm = Gsrc.loc[used_s].to_numpy(np.float32)
    print(f"  features: drug {Dm.shape} gene {Gm.shape}", flush=True)

    # standardise on TRAINING entities only
    tr_d = pd.Index(pd.unique(parts["train"][DRUG]))
    tr_s = pd.Index(pd.unique(parts["train"][SAMPLE].astype(str)))
    dpos = pd.Series(np.arange(len(used_d)), index=used_d)
    spos = pd.Series(np.arange(len(used_s)), index=used_s)
    dmu, dsd = Dm[dpos.loc[tr_d].to_numpy()].mean(0), Dm[dpos.loc[tr_d].to_numpy()].std(0)
    gmu, gsd = Gm[spos.loc[tr_s].to_numpy()].mean(0), Gm[spos.loc[tr_s].to_numpy()].std(0)
    Dm = (Dm - dmu) / np.where(dsd < 1e-8, 1.0, dsd)
    Gm = (Gm - gmu) / np.where(gsd < 1e-8, 1.0, gsd)
    Dm, Gm = np.nan_to_num(Dm), np.nan_to_num(Gm)
    Dt = torch.as_tensor(Dm, device=dev)
    Gt = torch.as_tensor(Gm, device=dev)
    dd, gd = Dt.shape[1], Gt.shape[1]
    del Dm, Gm

    panels = {}
    for k, v in parts.items():
        panels[k] = {"di": torch.as_tensor(dpos.loc[v[DRUG]].to_numpy(), dtype=torch.long, device=dev),
                     "si": torch.as_tensor(spos.loc[v[SAMPLE].astype(str)].to_numpy(),
                                           dtype=torch.long, device=dev),
                     "y": torch.as_tensor(v["Sensitivity"].to_numpy(np.float32), device=dev),
                     "yn": v["Sensitivity"].to_numpy(np.float64),
                     "proj": AdditiveProjection(v, [DRUG, SAMPLE]), "n": len(v)}

    net = ({"ridge": lambda: Ridge(dd, gd),
            "bilinear": lambda: Bilinear(dd, gd),
            "tower": lambda: CLIOSingleHead(dd, gd, hidden_dim1=1024, dropout_p=a.dropout)}
           [a.decoder]()).to(dev)
    npar = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
    print(f"  decoder {a.decoder}: {npar:,} trainable parameters", flush=True)
    crit = nn.MSELoss()
    opt = optim.Adam(net.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(Idx(panels["train"]["n"]), batch_size=a.batch_size, shuffle=True,
                        num_workers=0, drop_last=True, generator=gen)

    @torch.no_grad()
    def predict(p):
        net.eval()
        out = []
        for i in range(0, p["n"], a.batch_size):
            sl = slice(i, min(i + a.batch_size, p["n"]))
            out.append(net(Dt[p["di"][sl]], Gt[p["si"][sl]]).squeeze(-1).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

    best = [float("inf"), os.path.join(out_dir, "best_valmse.pth"), -1]
    hist, t0 = [], time.time()
    for ep in range(a.n_epochs):
        net.train()
        tot = nb = 0
        for idx in loader:
            i = idx.to(dev)
            loss = crit(net(Dt[panels["train"]["di"][i]], Gt[panels["train"]["si"][i]]).squeeze(-1),
                        panels["train"]["y"][i])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        pr = component_profile(panels["val"]["yn"], predict(panels["val"]), panels["val"]["proj"])
        if pr["MSE_raw"] < best[0]:
            best[0], best[2] = pr["MSE_raw"], ep + 1
            torch.save(net.state_dict(), best[1])
        hist.append({"epoch": ep + 1, "train_loss": tot / max(nb, 1),
                     **{f"val_{k}": v for k, v in pr.items()}})
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1:02d} [{time.time()-t0:5.0f}s] loss={hist[-1]['train_loss']:.5f} "
                  f"val raw={pr['rawPCC']:.3f} shared={pr['sharedPCC']:.3f} "
                  f"int={pr['intPCC']:.3f}", flush=True)

    net.load_state_dict(torch.load(best[1]))
    res = {"args": vars(a), "n_params": npar, "drug_dim": dd, "gene_dim": gd,
           "best_val_MSE_raw": best[0], "best_epoch": best[2], "history": hist,
           "wall_seconds": time.time() - t0}
    for split in ("val", "test"):
        res[f"frozen_valMSE__{split}"] = component_profile(
            panels[split]["yn"], predict(panels[split]), panels[split]["proj"])
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    t = res["frozen_valMSE__test"]
    print(f"\nTEST raw={t['rawPCC']:.4f} shared={t['sharedPCC']:.4f} int={t['intPCC']:.4f} "
          f"A_int={t['A_int']:.3f} sd(M yhat)={t['sd_M_yhat']:.2e}")
    print(f"wrote {out_dir}  ({res['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

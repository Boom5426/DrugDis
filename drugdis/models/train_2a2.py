"""Phase 2A-2: the M0 -> M2 -> M3 -> M4 ladder on the frozen LCLO substrate.

One causal factor changes per step (RISE_FROZEN_ROADMAP_2026-09-04.md section 8):

    M0  published single head, L = MSE(yhat, y)                     reference
    M1  single head, L = E_H + alpha * E_M                          objective   (2A-1, kept for the 3-seed confirmation)
    M2  single head with the joint head widened to match M3/M4      capacity
    M3  factorized b_hat + z_hat, raw-only supervision              factorization
    M4  M3 plus variance-normalised component supervision           supervision

Everything else is held fixed across arms: representation, panel, split, seed,
optimizer, scheduler, epochs, batch size, dropout, checkpoint rules.

Selection: the primary checkpoint is the epoch with the best validation legacy
iePCC, exactly as in M0 and Phase 2A-1, so that the arms differ only in the
factor under test. The epoch with the best validation intPCC is saved as the
pre-registered sensitivity rule. A PASS is only declared when the criterion
holds under both rules.

Nothing under legacy/src_copy is modified and no published output is overwritten.

Release note: the manuscript reports every arm at best_valmse.pth, the epoch with
the lowest validation MSE_raw (the frozen checkpoint rule), not the iePCC rule
described above for the Phase 2A-2 ladder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "decomposition"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # models/ and legacy utils.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_seed, InteractionEffectEvaluator, get_base_metrics  # noqa: E402
from decompositions import AdditiveProjection                              # noqa: E402

import data_fast                                                            # noqa: E402
import paths                                                                # noqa: E402
from models import (CLIOSingleHead, CLIOFactorized, n_params,               # noqa: E402
                    solve_m2_width, single_head_params, factorized_extra_params)

warnings.filterwarnings("ignore")

SHARED_HIDDEN = 512          # width of each scalar effect head in M3/M4
BASE_H1 = 1024               # M0 joint hidden width


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["M0", "M1", "M2", "M3", "M4", "M5"])
    p.add_argument("--drug_fm", default="ECFP4")
    p.add_argument("--genomics_fm", default="baseline")
    p.add_argument("--split_type", default="LCLO", choices=["LCLO", "LSO"])
    p.add_argument("--alpha", type=float, default=1.0, help="M1 only")
    p.add_argument("--test_split_ratio", type=float, default=0.15)
    p.add_argument("--val_split_ratio", type=float, default=0.15)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--dev", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--output_dir", default=str(paths.runs_dir()) + "/")
    p.add_argument("--tag_suffix", default="")
    p.add_argument("--substrate_config", default=None,
                   help="frozen substrate config; when given, the panel and the split are "
                        "read from disk instead of recomputed")
    p.add_argument("--manifest_dir", default=None)
    p.add_argument("--drug_rep_path", default=None,
                   help="representation benchmark: pickle of SMILES -> vector")
    p.add_argument("--gene_rep_path", default=None,
                   help="representation benchmark: Sample_ID-indexed parquet, CCLE-aligned")
    p.add_argument("--rep_tag", default=None, help="output directory name for the benchmark")
    p.add_argument("--allow_partial_drug_rep", action="store_true")
    p.add_argument("--fast_val", action="store_true",
                   help="skip the per-epoch validation component profile, which costs three "
                        "alternating-projection solves and is a diagnostic the canonical "
                        "tables never read. The frozen checkpoint rule needs only validation "
                        "MSE_raw, which is computed directly, and the legacy iePCC is still "
                        "computed because the scheduler steps on it. Training is untouched.")
    p.add_argument("--shuffle_z_target", action="store_true",
                   help="null control: permute the interaction supervision target across "
                        "training rows, keeping its marginal distribution but destroying its "
                        "association with the pair. Distinguishes the information in "
                        "M_train y from the mere presence of an auxiliary loss on that head")
    p.add_argument("--no_scheduler", action="store_true",
                   help="control only: hold the learning rate fixed. ReduceLROnPlateau reads a "
                        "noisy arm-dependent validation metric, so the realised schedule can "
                        "differ between arms even though the rule is the same")
    p.add_argument("--m4_literal", action="store_true",
                   help="use the literal roadmap 8.6 weights (L_raw unnormalised, the two "
                        "supervision terms divided by their own variance) instead of the "
                        "raw-MSE-units form; a sensitivity check on that documented deviation")
    p.add_argument("--init_jitter", type=float, default=0.0,
                   help="relative scale of a numerically meaningless perturbation of the "
                        "initial weights; used only to measure how far a chaotic training "
                        "trajectory moves the reported endpoint")
    return p.parse_args()


# ------------------------------------------------------------------ metrics


def pcc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def component_profile(y: np.ndarray, yh: np.ndarray, proj: AdditiveProjection) -> dict:
    """rawPCC / sharedPCC / intPCC and the exact error partition on one panel."""
    My, Myh = proj.residual(y), proj.residual(yh)
    Hy, Hyh = y - My, yh - Myh
    e = yh - y
    Me = proj.residual(e)
    He = e - Me
    return {
        "n": int(len(y)),
        "rawPCC": pcc(y, yh), "sharedPCC": pcc(Hy, Hyh), "intPCC": pcc(My, Myh),
        "A_shared": float(Hyh.std() / Hy.std()) if Hy.std() else float("nan"),
        "A_int": float(Myh.std() / My.std()) if My.std() else float("nan"),
        "MSE_raw": float(np.mean(e ** 2)),
        "E_shared": float(np.mean(He ** 2)),
        "E_interaction": float(np.mean(Me ** 2)),
    }


def factorization_report(bh: np.ndarray, zh: np.ndarray, y: np.ndarray,
                         proj: AdditiveProjection) -> dict:
    """Did the two heads actually factorize, or is one doing the other's job?"""
    My = proj.residual(y)
    Hy = y - My
    Mz = proj.residual(zh)
    Hz = zh - Mz
    Mb = proj.residual(bh)
    zc = zh - zh.mean()
    Hzc = Hz - zh.mean()          # the constant vector lies in the additive space
    return {
        "corr_bhat_Hy": pcc(bh, Hy),
        "corr_zhat_My": pcc(zh, My),
        # roadmap section 8.5 / 9 definition, which counts the intercept as additive
        "leak_H_zhat": float(np.sum(Hz ** 2) / np.sum(zh ** 2)) if np.any(zh) else float("nan"),
        # the same quantity after removing the grand mean. Under raw-only supervision
        # the split of the constant between mu and the interaction head is not
        # identified, so this is the part of the leakage that is drug- or
        # sample-structured rather than an arbitrary offset.
        "leak_H_zhat_centered": float(np.sum(Hzc ** 2) / np.sum(zc ** 2)) if np.any(zc) else float("nan"),
        # identically zero by construction: b_hat is a function of the drug plus a
        # function of the sample plus a constant, so it lies in the additive space.
        # Reported as an implementation check, not as evidence about factorization.
        "leak_M_bhat": float(np.sum(Mb ** 2) / np.sum(bh ** 2)) if np.any(bh) else float("nan"),
        "sd_bhat": float(bh.std()), "sd_zhat": float(zh.std()),
        "mean_bhat": float(bh.mean()), "mean_zhat": float(zh.mean()),
    }


def legacy_iepcc(df: pd.DataFrame, evaluator: InteractionEffectEvaluator) -> float:
    """Exactly the iePCC that `calculate_metrics` reports, without the CKA and
    top-10 structural metrics, which the selection rule never reads."""
    it, ip = evaluator.get_interaction_effects(df)
    _, ie_pcc, _ = get_base_metrics(it, ip)
    return float(ie_pcc)


# ------------------------------------------------------------------ forward


@torch.no_grad()
def forward_panel(model, store, panel, batch_size: int, factorized: bool) -> dict:
    model.eval()
    bs = []
    zs = []
    for i in range(0, panel.n, batch_size):
        idx = torch.arange(i, min(i + batch_size, panel.n), device=panel.dev)
        xd, xg, _ = store.gather(panel, idx)
        if factorized:
            b, z = model(xd, xg)
            bs.append(b.squeeze(-1).float().cpu())
            zs.append(z.squeeze(-1).float().cpu())
        else:
            zs.append(model(xd, xg).squeeze(-1).float().cpu())
    z = torch.cat(zs).numpy().astype(np.float64)
    if factorized:
        b = torch.cat(bs).numpy().astype(np.float64)
        return {"yhat": b + z, "bhat": b, "zhat": z}
    return {"yhat": z}


# ------------------------------------------------------------------ main


def main(args) -> int:
    factorized = args.model in ("M3", "M4", "M5")
    supervise = args.model in ("M4", "M5")
    orth = args.model == "M5"
    if args.model != "M1" and args.alpha != 1.0:
        raise SystemExit("--alpha is only meaningful for M1")

    dev = torch.device(args.dev if torch.cuda.is_available() else "cpu")
    tag = f"{args.model}_seed{args.seed}" + ("_literal" if (args.m4_literal and supervise) else "")
    if args.model == "M1":
        tag = f"M1_alpha{args.alpha:g}_seed{args.seed}"
    if args.rep_tag:
        tag = f"{args.rep_tag}__{args.model}_seed{args.seed}"
    tag += args.tag_suffix
    out_dir = os.path.join(args.output_dir, f"{args.drug_fm}__{args.genomics_fm}",
                           args.split_type, tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"--- Phase 2A-2 | {args.model} | {args.split_type} | seed={args.seed}"
          f"{' | alpha=%g' % args.alpha if args.model == 'M1' else ''} ---", flush=True)

    setup_seed(args.seed)
    if args.substrate_config:
        store, train, val, test = data_fast.build_frozen(
            args.substrate_config, args.manifest_dir, args.split_type, args.seed,
            args.drug_fm, dev, drug_rep_path=args.drug_rep_path,
            gene_rep_path=args.gene_rep_path,
            allow_partial_drug_rep=args.allow_partial_drug_rep)
    else:
        store, train, val, test = data_fast.build(
            args.drug_fm, args.genomics_fm, args.split_type, args.seed,
            args.test_split_ratio, args.val_split_ratio, dev)

    # ---- parameter budget, frozen before any weight is trained
    m2_h1, p_m2, p_m3_target = solve_m2_width(store.drug_dim, store.gene_dim, SHARED_HIDDEN)
    budget = {
        "M0": single_head_params(store.drug_dim, store.gene_dim, BASE_H1),
        "M2": p_m2, "M2_joint_hidden1": m2_h1,
        "M3": single_head_params(store.drug_dim, store.gene_dim, BASE_H1)
              + factorized_extra_params(SHARED_HIDDEN),
        "shared_hidden": SHARED_HIDDEN,
    }
    budget["rel_mismatch_M2_vs_M3"] = abs(budget["M2"] - budget["M3"]) / budget["M3"]
    assert budget["rel_mismatch_M2_vs_M3"] <= 0.02, budget

    h1 = m2_h1 if args.model == "M2" else BASE_H1
    if factorized:
        model = CLIOFactorized(store.drug_dim, store.gene_dim, hidden_dim1=h1,
                               shared_hidden=SHARED_HIDDEN, dropout_p=args.dropout)
    else:
        model = CLIOSingleHead(store.drug_dim, store.gene_dim, hidden_dim1=h1,
                               dropout_p=args.dropout)
    if args.init_jitter > 0:
        g = torch.Generator().manual_seed(20260904)
        with torch.no_grad():
            for prm in model.parameters():
                scale = float(prm.detach().abs().mean())
                if scale > 0:
                    prm.add_((torch.randn(prm.shape, generator=g, dtype=prm.dtype)
                              .to(prm.device)) * (args.init_jitter * scale))
        print(f"  init jitter applied: relative scale {args.init_jitter:g}", flush=True)
    p_actual = n_params(model)
    print(f"  trainable parameters: {p_actual:,} (joint hidden1={h1}, "
          f"budget check M2={budget['M2']:,} M3={budget['M3']:,} "
          f"mismatch={budget['rel_mismatch_M2_vs_M3']:.5%})", flush=True)
    expected = budget[args.model] if args.model in ("M0", "M2", "M3") else budget["M0"]
    if args.model in ("M4", "M5"):
        expected = budget["M3"]
    if args.model != "M1":
        assert p_actual == expected, (p_actual, expected)

    model = model.to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    scheduler = (None if args.no_scheduler else
                 ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=10, verbose=False))
    evaluator = InteractionEffectEvaluator(train.df, split_type=args.split_type)

    y_train = train.df["Sensitivity"].to_numpy(np.float64)
    y_val = val.df["Sensitivity"].to_numpy(np.float64)
    y_test = test.df["Sensitivity"].to_numpy(np.float64)
    proj_train = AdditiveProjection(train.df, ["SMILES", "Sample_ID"])
    proj_val = AdditiveProjection(val.df, ["SMILES", "Sample_ID"])
    proj_test = AdditiveProjection(test.df, ["SMILES", "Sample_ID"])

    # ---- component supervision targets (M4 only), fixed for the whole run
    lam = {}
    if supervise:
        r_tr = proj_train.residual(y_train)
        b_tr = y_train - r_tr
        var_y, var_b, var_r = float(np.var(y_train)), float(np.var(b_tr)), float(np.var(r_tr))
        # Variance-normalised supervision expressed in raw-MSE units, so the raw
        # term is numerically identical to M3's and only the two supervision
        # terms are added:  L = L_raw + (Var y / Var b) L_b + (Var y / Var r) L_z.
        if args.m4_literal:
            # roadmap 8.6 read literally: L = L_raw + L_b/Var(b) + L_z/Var(r)
            lam = {"lambda_b": 1.0 / var_b, "lambda_z": 1.0 / var_r, "form": "literal"}
        else:
            lam = {"lambda_b": var_y / var_b, "lambda_z": var_y / var_r,
                   "form": "raw_mse_units"}
        lam.update({"var_y": var_y, "var_b": var_b, "var_r": var_r})
        if orth:
            # one pre-registered weight, no sweep: unit weight in the same
            # variance-normalised units as the two supervision terms, which is
            # Var(y) once the objective is written in raw-MSE units.
            lam["lambda_orth"] = var_y
        r_sup = r_tr
        if args.shuffle_z_target:
            r_sup = np.random.default_rng(args.seed).permutation(r_tr)
            print(f"  NULL CONTROL: interaction target permuted across training rows "
                  f"(corr with the true target {float(np.corrcoef(r_sup, r_tr)[0, 1]):.5f})",
                  flush=True)
        b_tr_t = torch.as_tensor(b_tr, dtype=torch.float32, device=dev)
        r_tr_t = torch.as_tensor(r_sup, dtype=torch.float32, device=dev)
        print(f"  component supervision: lambda_b={lam['lambda_b']:.3f} "
              f"lambda_z={lam['lambda_z']:.3f} (var y/b/r = {var_y:.5f}/{var_b:.5f}/{var_r:.5f})",
              flush=True)

    # Frozen selection rule for the manuscript-wide rerun: the epoch with the
    # lowest validation raw MSE. Under the frozen partition MSE_raw = E_shared +
    # E_interaction, so it reads the whole response and favours neither component,
    # and it is identical for every arm. The legacy iePCC and the val-intPCC
    # checkpoints are still written, as historical continuity and as a sensitivity
    # view; neither selects the reported model any more.
    best = {"valmse": [float("inf"), os.path.join(out_dir, "best_valmse.pth"), -1],
            "iepcc": [-1.0, os.path.join(out_dir, "best_iepcc.pth"), -1],
            "intpcc": [-1.0, os.path.join(out_dir, "best_intpcc.pth"), -1]}
    history = []
    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = train.loader(args.batch_size, shuffle=True, drop_last=True, generator=loader_gen)
    t0 = time.time()

    for epoch in range(args.n_epochs):
        eh = em = float("nan")
        orth_state = None
        if orth:
            # Full-support, epoch-refreshed surrogate for
            #     L_orth = ||H_train zhat||^2 / ||zhat||^2 .
            # A first attempt linearised L_orth at the epoch-start z. That has the
            # exact gradient but is LINEAR in z, so it is unbounded below and the
            # optimizer walks down it: within six epochs the total loss went
            # negative and A_int reached 2.8. A linear surrogate of a bounded
            # functional is not usable.
            #
            # A second attempt used a projection target normalised by the epoch's
            # own scale, S = MSE(z_batch, (M z_ep)_batch) / mean(z_ep^2). That is
            # bounded but not scale invariant, so the head can satisfy it by
            # vanishing, and it did: sd(z_hat) fell to 1e-5 within three epochs and
            # the denominator, being the epoch's own scale, collapsed with it.
            #
            # The form used here keeps the numerator's target from the full-support
            # operator and takes the denominator from the CURRENT batch, so the
            # surrogate is scale invariant in z_hat exactly as L_orth is:
            #     S = MSE(z_b, (M_train z_ep)_b) / mean(z_b^2)
            # It equals L_orth at the refresh point, it diverges as z_hat -> 0 so
            # collapsing the head is penalised rather than rewarded, and only the
            # scalar scale is taken from the batch. No batch-level PROJECTION is
            # used anywhere: M_train is applied on the full support, once per epoch.
            zep = forward_panel(model, store, train, args.batch_size, True)["zhat"]
            Mz = proj_train.residual(zep)
            Hz = zep - Mz
            bb = float(np.sum(zep ** 2))
            l_orth_val = float(np.sum(Hz ** 2)) / bb if bb > 0 else 0.0
            denom = bb / max(len(zep), 1)
            orth_state = (torch.as_tensor(Mz, dtype=torch.float32, device=dev),
                          float(denom), l_orth_val)
        if args.model == "M1" and args.alpha != 1.0:
            yhat_full = forward_panel(model, store, train, args.batch_size, False)["yhat"]
            e_full = yhat_full - y_train
            Me = proj_train.residual(e_full)
            train.targets = torch.as_tensor(y_train - (args.alpha - 1.0) * Me,
                                            dtype=torch.float32, device=dev)
            eh, em = float(np.mean((e_full - Me) ** 2)), float(np.mean(Me ** 2))

        model.train()
        tot = tot_raw = tot_b = tot_z = tot_o = 0.0
        nb = 0
        for idx in loader:
            xd, xg, tgt = store.gather(train, idx)
            if factorized:
                bh, zh = model(xd, xg)
                bh, zh = bh.squeeze(-1), zh.squeeze(-1)
                l_raw = criterion(bh + zh, tgt)
                if supervise:
                    gi = idx.to(dev)
                    l_b = criterion(bh, b_tr_t[gi])
                    l_z = criterion(zh, r_tr_t[gi])
                    loss = l_raw + lam["lambda_b"] * l_b + lam["lambda_z"] * l_z
                    tot_b += float(l_b.item()); tot_z += float(l_z.item())
                    if orth_state is not None:
                        mz, _den, _ = orth_state
                        surro = criterion(zh, mz[gi]) / (zh.pow(2).mean() + 1e-12)
                        loss = loss + lam["lambda_orth"] * surro
                        tot_o += float(surro.item())
                else:
                    loss = l_raw
                tot_raw += float(l_raw.item())
            else:
                loss = criterion(model(xd, xg).squeeze(-1), tgt)
                tot_raw += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += float(loss.item())
            nb += 1

        out = forward_panel(model, store, val, args.batch_size, factorized)
        vdf = val.df[["SMILES", "Sample_ID"]].copy()
        vdf["Target_AAC"] = y_val
        vdf["Predicted_AAC"] = out["yhat"]
        ie = legacy_iepcc(vdf, evaluator)          # the scheduler steps on this
        if args.fast_val:
            prof = {"MSE_raw": float(np.mean((out["yhat"] - y_val) ** 2)),
                    "rawPCC": pcc(y_val, out["yhat"]), "sharedPCC": float("nan"),
                    "intPCC": float("nan"), "A_int": float("nan"),
                    "A_shared": float("nan"), "E_shared": float("nan"),
                    "E_interaction": float("nan"), "n": int(len(y_val))}
        else:
            prof = component_profile(y_val, out["yhat"], proj_val)
        if scheduler is not None:
            scheduler.step(ie)

        if prof["MSE_raw"] < best["valmse"][0]:
            best["valmse"][0], best["valmse"][2] = prof["MSE_raw"], epoch + 1
            torch.save(model.state_dict(), best["valmse"][1])
        if ie > best["iepcc"][0]:
            best["iepcc"][0], best["iepcc"][2] = ie, epoch + 1
            torch.save(model.state_dict(), best["iepcc"][1])
        if prof["intPCC"] == prof["intPCC"] and prof["intPCC"] > best["intpcc"][0]:
            best["intpcc"][0], best["intpcc"][2] = prof["intPCC"], epoch + 1
            torch.save(model.state_dict(), best["intpcc"][1])

        rec = {"epoch": epoch + 1, "lr": float(optimizer.param_groups[0]["lr"]),
               "train_loss": tot / max(nb, 1),
               "train_loss_raw": tot_raw / max(nb, 1),
               "train_E_shared": eh, "train_E_interaction": em,
               "val_iePCC_legacy": ie, **{f"val_{k}": v for k, v in prof.items()}}
        if supervise:
            rec["train_loss_b"] = tot_b / max(nb, 1)
            rec["train_loss_z"] = tot_z / max(nb, 1)
        if orth:
            rec["train_L_orth_at_epoch_start"] = orth_state[2]
            rec["train_orth_surrogate"] = tot_o / max(nb, 1)
        if factorized:
            rec.update({f"val_{k}": v for k, v in
                        factorization_report(out["bhat"], out["zhat"], y_val, proj_val).items()})
        history.append(rec)
        msg = (f"  ep {epoch+1:02d} [{time.time()-t0:6.0f}s] loss={rec['train_loss']:.5f} "
               f"val raw={prof['rawPCC']:.3f} shared={prof['sharedPCC']:.3f} "
               f"int={prof['intPCC']:.3f} A_int={prof['A_int']:.2f} (iePCC={ie:.3f})")
        if factorized:
            msg += (f" | corr(b,Hy)={rec['val_corr_bhat_Hy']:.3f} "
                    f"corr(z,My)={rec['val_corr_zhat_My']:.3f} "
                    f"leakH(z)={rec['val_leak_H_zhat']:.3f}")
        print(msg, flush=True)

    results = {"args": vars(args), "model": args.model, "n_params": p_actual,
               "param_budget": budget, "component_supervision": lam,
               "history": history,
               "best_val_MSE_raw": best["valmse"][0], "best_valmse_epoch": best["valmse"][2],
               "best_val_iePCC_legacy": best["iepcc"][0], "best_iepcc_epoch": best["iepcc"][2],
               "best_val_intPCC": best["intpcc"][0], "best_intpcc_epoch": best["intpcc"][2],
               "wall_seconds": time.time() - t0}
    rules = [("frozen_valMSE", best["valmse"][1])]
    if not args.fast_val:
        rules += [("primary_legacy_iePCC", best["iepcc"][1]),
                  ("sensitivity_intPCC", best["intpcc"][1])]
    for rule, ckpt in rules:
        model.load_state_dict(torch.load(ckpt))
        for split, panel, yv, pr in (("val", val, y_val, proj_val),
                                     ("test", test, y_test, proj_test)):
            o = forward_panel(model, store, panel, args.batch_size, factorized)
            results[f"{rule}__{split}"] = component_profile(yv, o["yhat"], pr)
            if factorized:
                results[f"{rule}__{split}"].update(
                    factorization_report(o["bhat"], o["zhat"], yv, pr))
            if rule == "primary_legacy_iePCC":
                cols = {"SMILES": panel.df["SMILES"], "Sample_ID": panel.df["Sample_ID"],
                        "Target_AAC": yv, "Predicted_AAC": o["yhat"]}
                if factorized:
                    cols["bhat"] = o["bhat"]; cols["zhat"] = o["zhat"]
                pd.DataFrame(cols).to_csv(
                    os.path.join(out_dir, f"{split}_predictions.csv"), index=False)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    t = results["frozen_valMSE__test"]
    print(f"\nTEST (frozen val-MSE rule): raw={t['rawPCC']:.4f} shared={t['sharedPCC']:.4f} "
          f"int={t['intPCC']:.4f} A_shared={t['A_shared']:.3f} A_int={t['A_int']:.3f}")
    s = results["sensitivity_intPCC__test"]
    print(f"TEST (sensitivity) : raw={s['rawPCC']:.4f} shared={s['sharedPCC']:.4f} "
          f"int={s['intPCC']:.4f} A_int={s['A_int']:.3f}")
    print(f"wrote {out_dir}  ({results['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))

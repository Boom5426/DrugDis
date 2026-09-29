"""Phase 0E: component-wise recovery, on exactly matched support.

RISE is not a claim that interaction is the only response information worth
recovering. Shared drug and sample effects are genuine, reproducible, predictable
information. This script therefore reports a recovery profile

    R = (rawPCC, sharedPCC, intPCC)

for every model, together with the exact partition of prediction error into
additive-space and interaction-space error, and it compares each component
against a reproducibility reference estimated on the *same* drug-sample support.

Support discipline. The reproducibility reference comes from pairs measured by
both GDSC screening programmes. Those same pairs also appear in the harmonised
table under other resources (CTRP2, CCLE, CTRP1, PRISM and others), carrying that
resource's measurement rather than GDSC1's. A ratio whose numerator and
denominator refer to different measurements of different populations is
meaningless, so Omega_exact keeps only evaluation rows whose target equals the
GDSC1 arm to floating-point tolerance.
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

GDSC_RAW = str(paths.data_dir() / "gdsc_sensitivity_data.parquet")
RESULTS = str(paths.runs_dir() / "Results")
GDSC2_ROW_CUT = 229420
TARGET, PRED = "Target_AAC", "Predicted_AAC"
TOL = 1e-6


def corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def omega_q() -> pd.DataFrame:
    raw = pd.read_parquet(GDSC_RAW).reset_index(drop=True)
    raw["prog"] = np.where(np.arange(len(raw)) < GDSC2_ROW_CUT, "GDSC1", "GDSC2")
    counts = raw.groupby([DRUG, SAMPLE]).size()
    dup = set(counts[counts == 2].index)
    sub = raw[pd.MultiIndex.from_arrays([raw[DRUG], raw[SAMPLE]]).isin(dup)]
    wide = (sub.pivot_table(index=[DRUG, SAMPLE], columns="prog", values="Sensitivity",
                            aggfunc="first").dropna().reset_index())
    return wide.rename(columns={"GDSC1": "yA", "GDSC2": "yB"})


def exact_panel(path: str, oq: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[DRUG, SAMPLE, TARGET, PRED]).dropna()
    df = df.drop_duplicates([DRUG, SAMPLE])
    j = df.merge(oq, on=[DRUG, SAMPLE], how="inner")
    return j[np.isclose(j[TARGET], j["yA"], atol=TOL)].reset_index(drop=True)


def profile(frame: pd.DataFrame, proj: AdditiveProjection) -> dict:
    """Component recovery profile and the exact error partition."""
    y = frame[TARGET].to_numpy(np.float64)
    yh = frame[PRED].to_numpy(np.float64)
    My, Myh = proj.residual(y), proj.residual(yh)
    Hy, Hyh = y - My, yh - Myh
    e = yh - y
    Me = proj.residual(e)
    He = e - Me
    n = len(y)
    mse = float(np.mean(e ** 2))
    e_sh = float(np.mean(He ** 2))
    e_in = float(np.mean(Me ** 2))
    return {
        "n": int(n),
        "rawPCC": corr(y, yh),
        "sharedPCC": corr(Hy, Hyh),
        "intPCC": corr(My, Myh),
        "A_shared": float(Hyh.std() / Hy.std()) if Hy.std() else float("nan"),
        "A_int": float(Myh.std() / My.std()) if My.std() else float("nan"),
        "MSE_raw": mse,
        "E_shared": e_sh,
        "E_interaction": e_in,
        "partition_abs_error": abs(mse - (e_sh + e_in)),
        "partition_rel_error": abs(mse - (e_sh + e_in)) / mse if mse else float("nan"),
        "var_share_interaction_pct": float(100 * My.var() / y.var()) if y.var() else float("nan"),
    }


def reproducibility(frame: pd.DataFrame, proj: AdditiveProjection) -> dict:
    yA = frame["yA"].to_numpy(np.float64)
    yB = frame["yB"].to_numpy(np.float64)
    MA, MB = proj.residual(yA), proj.residual(yB)
    return {
        "q_total": corr(yA, yB),
        "q_shared": corr(yA - MA, yB - MB),
        "q_interaction": corr(MA, MB),
    }


def two_way_resample(frame: pd.DataFrame, rng, min_rows: int = 300):
    drugs = frame[DRUG].unique()
    samples = frame[SAMPLE].unique()
    d = set(rng.choice(drugs, size=len(drugs), replace=True))
    s = set(rng.choice(samples, size=len(samples), replace=True))
    boot = frame[frame[DRUG].isin(d) & frame[SAMPLE].isin(s)]
    return boot.reset_index(drop=True) if len(boot) >= min_rows else None


def joint_bootstrap(frame: pd.DataFrame, n_boot: int, seed: int = 42) -> dict:
    """Resample clusters once per replicate and recompute BOTH the model
    correlation and the reproducibility reference on that resample, so the ratio
    carries the uncertainty of numerator and denominator jointly."""
    rng = np.random.default_rng(seed)
    keep = {k: [] for k in ("eta_total", "eta_shared", "eta_interaction",
                            "q_interaction", "intPCC")}
    skipped = 0
    for _ in range(n_boot):
        boot = two_way_resample(frame, rng)
        if boot is None:
            skipped += 1
            continue
        p = AdditiveProjection(boot, [DRUG, SAMPLE])
        prof = profile(boot, p)
        rep = reproducibility(boot, p)
        ok = True
        for comp, rho_key in (("total", "rawPCC"), ("shared", "sharedPCC"),
                              ("interaction", "intPCC")):
            q = rep[f"q_{comp}"]
            if not (q == q) or q <= 0:
                ok = False
                break
        if not ok:
            skipped += 1
            continue
        keep["eta_total"].append(prof["rawPCC"] / np.sqrt(rep["q_total"]))
        keep["eta_shared"].append(prof["sharedPCC"] / np.sqrt(rep["q_shared"]))
        keep["eta_interaction"].append(prof["intPCC"] / np.sqrt(rep["q_interaction"]))
        keep["q_interaction"].append(rep["q_interaction"])
        keep["intPCC"].append(prof["intPCC"])
    out = {"n_boot_ok": n_boot - skipped, "n_skipped": skipped}
    for k, v in keep.items():
        if not v:
            continue
        a = np.array(v)
        out[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                  "ci95": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]}
    out["_draws_eta_interaction"] = keep["eta_interaction"]
    return out


def verify_operator(frame: pd.DataFrame, proj: AdditiveProjection, rng) -> dict:
    """M must be linear and depend only on the support, not on the values."""
    n = len(frame)
    v1 = rng.normal(size=n)
    v2 = rng.normal(size=n)
    c = 3.7
    m1, m2 = proj.residual(v1), proj.residual(v2)
    lin = float(np.max(np.abs(proj.residual(v1 + c * v2) - (m1 + c * m2))))
    idem = float(np.max(np.abs(proj.residual(m1) - m1)))
    orth = float(abs(np.dot(v1 - m1, m1)) / max(np.linalg.norm(v1 - m1) * np.linalg.norm(m1), 1e-30))
    return {"linearity_max_abs_dev": lin, "idempotence_max_abs_dev": idem,
            "H_M_orthogonality_cosine": orth}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase0e_component_recovery.json")
    ap.add_argument("--n-boot", type=int, default=600)
    args = ap.parse_args()

    oq = omega_q()
    print(f"Omega_q: {len(oq):,} pairs, {oq[DRUG].nunique()} drugs, {oq[SAMPLE].nunique()} samples",
          flush=True)
    rng = np.random.default_rng(7)
    res: dict = {"omega_q": {"n_pairs": int(len(oq)), "n_drugs": int(oq[DRUG].nunique()),
                             "n_samples": int(oq[SAMPLE].nunique())}, "settings": {}}

    eta_draws = {}
    for setting in ("LSO", "LCLO"):
        paths = sorted(glob.glob(os.path.join(RESULTS, "*", setting, "test_predictions.csv")))
        if not paths:
            continue
        ref = exact_panel(paths[0], oq)
        key = set(map(tuple, ref[[DRUG, SAMPLE]].to_numpy()))
        proj = AdditiveProjection(ref, [DRUG, SAMPLE])
        rep = reproducibility(ref, proj)
        entry = {
            "omega_exact": {
                "n_rows": int(len(ref)), "n_drugs": int(ref[DRUG].nunique()),
                "n_samples": int(ref[SAMPLE].nunique()),
                "reference_model": paths[0].split(os.sep)[-3],
            },
            "reproducibility": rep,
            "operator_checks": verify_operator(ref, proj, rng),
            "models": [],
        }
        print(f"[{setting}] Omega_exact n={len(ref):,} drugs={ref[DRUG].nunique()} "
              f"samples={ref[SAMPLE].nunique()}  q_total={rep['q_total']:.3f} "
              f"q_shared={rep['q_shared']:.3f} q_int={rep['q_interaction']:.3f}", flush=True)

        for path in paths:
            model = path.split(os.sep)[-3]
            fr = exact_panel(path, oq)
            same = set(map(tuple, fr[[DRUG, SAMPLE]].to_numpy())) == key
            if len(fr) < 100:
                continue
            p = AdditiveProjection(fr, [DRUG, SAMPLE])
            prof = profile(fr, p)
            prof["model"] = model
            prof["same_support_as_reference"] = bool(same)
            drug, _, gene = model.partition("__")
            prof["branch"] = "gene" if drug == "ECFP4" and gene != "baseline" else "drug"
            for comp, rho_key in (("total", "rawPCC"), ("shared", "sharedPCC"),
                                  ("interaction", "intPCC")):
                q = rep[f"q_{comp}"]
                prof[f"eta_{comp}"] = (prof[rho_key] / np.sqrt(q)) if q == q and q > 0 else float("nan")
            entry["models"].append(prof)

        best = max((m for m in entry["models"] if m["intPCC"] == m["intPCC"]),
                   key=lambda m: m["intPCC"], default=None)
        if best is not None:
            fr = exact_panel(os.path.join(RESULTS, best["model"], setting,
                                          "test_predictions.csv"), oq)
            print(f"  joint bootstrap on {best['model']}", flush=True)
            jb = joint_bootstrap(fr, args.n_boot)
            eta_draws[setting] = jb.pop("_draws_eta_interaction")
            entry["joint_bootstrap"] = {"model": best["model"], **jb}
        res["settings"][setting] = entry

    if len(eta_draws) == 2:
        n = min(len(eta_draws["LCLO"]), len(eta_draws["LSO"]))
        if n > 20:
            d = np.array(eta_draws["LCLO"][:n]) - np.array(eta_draws["LSO"][:n])
            res["eta_interaction_LCLO_minus_LSO"] = {
                "n_paired_draws": int(n), "mean": float(d.mean()),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "frac_positive": float((d > 0).mean()),
            }

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 0C: reproducibility of the interaction component, with uncertainty.

Two empirical references, named for what they actually compare:

  within-programme, cross-assay-generation : GDSC1 vs GDSC2, same consortium,
        different assay generation. `gdsc_sensitivity_data.parquet` is a clean
        concatenation, GDSC1 occupying raw rows 0..229419 and GDSC2 229420..end
        (verified: 100% of within-group first rows fall below the cut and 100% of
        second rows at or above it).

  cross-resource : pairs measured by two different programmes in the harmonised
        table. Mixes assay chemistry, dose range, exposure time, curve fitting
        and normalisation, so it is a different quantity, not a bound on the same
        one.

The 43,622 matched pairs span only 89 compounds and 799 samples, so the effective
sample size is governed by the number of clusters, not the number of pairs. All
intervals are cluster bootstrap intervals over drugs, over samples, and two-way.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from decompositions import DRUG, RESOURCE, SAMPLE, AdditiveProjection, MarginalMeanD1

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

GDSC_RAW = str(paths.data_dir() / "gdsc_sensitivity_data.parquet")
RESPONSE = str(paths.data_dir() / "drug_response.parquet")
GDSC2_ROW_CUT = 229420  # verified boundary between the two programmes
SENS = "Sensitivity"


def matched_gdsc() -> pd.DataFrame:
    raw = pd.read_parquet(GDSC_RAW).reset_index(drop=True)
    raw["programme"] = np.where(np.arange(len(raw)) < GDSC2_ROW_CUT, "GDSC1", "GDSC2")
    counts = raw.groupby([DRUG, SAMPLE]).size()
    dup = counts[counts == 2].index
    sub = raw[pd.MultiIndex.from_arrays([raw[DRUG], raw[SAMPLE]]).isin(set(dup))]
    wide = sub.pivot_table(index=[DRUG, SAMPLE], columns="programme", values=SENS,
                           aggfunc="first")
    wide = wide.dropna(subset=["GDSC1", "GDSC2"]).reset_index()
    return wide.rename(columns={"GDSC1": "yA", "GDSC2": "yB"})


def matched_cross_resource(min_pairs: int = 200) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(RESPONSE, columns=[DRUG, SAMPLE, SENS, RESOURCE]).dropna(subset=[SENS])
    key = pd.MultiIndex.from_arrays([df[DRUG], df[SAMPLE]])
    df = df.assign(_k=key)
    out = {}
    projects = sorted(df[RESOURCE].unique())
    for i, a in enumerate(projects):
        for b in projects[i + 1:]:
            da = df[df[RESOURCE] == a].drop_duplicates("_k").set_index("_k")
            db = df[df[RESOURCE] == b].drop_duplicates("_k").set_index("_k")
            shared = da.index.intersection(db.index)
            if len(shared) < min_pairs:
                continue
            out[f"{a}|{b}"] = pd.DataFrame({
                DRUG: da.loc[shared, DRUG].to_numpy(),
                SAMPLE: da.loc[shared, SAMPLE].to_numpy(),
                "yA": da.loc[shared, SENS].to_numpy(),
                "yB": db.loc[shared, SENS].to_numpy(),
            })
    return out


def decompose_pair(frame: pd.DataFrame, method: str) -> tuple[np.ndarray, ...]:
    """Decompose each measurement arm independently on the shared support."""
    if method == "D1":
        d = MarginalMeanD1("general")
        outs = []
        for col in ("yA", "yB"):
            f = pd.DataFrame({DRUG: frame[DRUG], SAMPLE: frame[SAMPLE], "Target_AAC": frame[col]})
            d.fit(f, "Target_AAC")
            b = d.shared(f, "Target_AAC")
            outs.append((b, frame[col].to_numpy(np.float64) - b))
        return outs[0][0], outs[0][1], outs[1][0], outs[1][1]
    proj = AdditiveProjection(frame, [DRUG, SAMPLE])
    bA, rA = proj.split(frame["yA"].to_numpy(np.float64))
    bB, rB = proj.split(frame["yB"].to_numpy(np.float64))
    return bA, rA, bB, rB


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def point_estimates(frame: pd.DataFrame, method: str) -> dict:
    bA, rA, bB, rB = decompose_pair(frame, method)
    yA = frame["yA"].to_numpy(np.float64)
    yB = frame["yB"].to_numpy(np.float64)
    q = corr(rA, rB)
    return {
        "n_pairs": int(len(frame)),
        "n_drugs": int(frame[DRUG].nunique()),
        "n_samples": int(frame[SAMPLE].nunique()),
        "rho_y": corr(yA, yB),
        "rho_b": corr(bA, bB),
        "q_r": q,
        "ceiling_sqrt_q": float(np.sqrt(q)) if q == q and q > 0 else np.nan,
        "var_share_r_pct_A": float(100 * rA.var() / (bA.var() + rA.var())),
        "spearman_r": float(pd.Series(rA).corr(pd.Series(rB), method="spearman")),
    }


def cluster_bootstrap(frame: pd.DataFrame, method: str, mode: str,
                      n_boot: int, seed: int = 42) -> dict:
    """Resample whole clusters with replacement, redoing the decomposition each time."""
    rng = np.random.default_rng(seed)
    drugs = frame[DRUG].unique()
    samples = frame[SAMPLE].unique()
    by_drug = {k: v for k, v in frame.groupby(DRUG, sort=False)}
    by_sample = {k: v for k, v in frame.groupby(SAMPLE, sort=False)}
    vals = []
    for _ in range(n_boot):
        if mode in ("drug", "sample"):
            axis = DRUG if mode == "drug" else SAMPLE
            pool, table = (drugs, by_drug) if mode == "drug" else (samples, by_sample)
            pick = rng.choice(pool, size=len(pool), replace=True)
            # A cluster drawn twice must enter as two distinct entities, otherwise
            # the decomposition would pool the copies and shrink their marginal.
            parts = []
            for rep, key in enumerate(pick):
                g = table[key].copy()
                g[axis] = f"{key}#{rep}"
                parts.append(g)
            boot = pd.concat(parts, ignore_index=True)
        else:  # two-way: resample both axes and keep the intersection
            pd_ = set(rng.choice(drugs, size=len(drugs), replace=True))
            ps_ = set(rng.choice(samples, size=len(samples), replace=True))
            boot = frame[frame[DRUG].isin(pd_) & frame[SAMPLE].isin(ps_)]
            if len(boot) < 200:
                continue
            boot = boot.reset_index(drop=True)
        try:
            _, rA, _, rB = decompose_pair(boot, method)
            v = corr(rA, rB)
            if v == v:
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return {"mode": mode, "n_boot_ok": 0}
    arr = np.array(vals)
    return {
        "mode": mode,
        "n_boot_ok": int(len(arr)),
        "q_mean": float(arr.mean()),
        "q_sd": float(arr.std(ddof=1)),
        "q_ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "ceiling_ci95": [float(np.sqrt(max(np.percentile(arr, 2.5), 0))),
                         float(np.sqrt(max(np.percentile(arr, 97.5), 0)))],
    }


def stratify(frame: pd.DataFrame, method: str) -> dict:
    bA, rA, bB, rB = decompose_pair(frame, method)
    f = frame.assign(rA=rA, rB=rB, bA=bA, yA=frame["yA"])
    out: dict = {}

    # by per-drug and per-sample, where enough observations exist
    for axis, name in ((DRUG, "by_drug"), (SAMPLE, "by_sample")):
        rows = []
        for key, g in f.groupby(axis, sort=False):
            if len(g) < 30:
                continue
            rows.append({"key": str(key), "n": int(len(g)),
                         "q": corr(g["rA"].to_numpy(), g["rB"].to_numpy())})
        rows = [r for r in rows if r["q"] == r["q"]]
        if rows:
            qs = np.array([r["q"] for r in rows])
            out[name] = {
                "n_groups": len(rows),
                "q_median": float(np.median(qs)),
                "q_iqr": [float(np.percentile(qs, 25)), float(np.percentile(qs, 75))],
                "q_min": float(qs.min()), "q_max": float(qs.max()),
                "frac_groups_q_below_0": float((qs < 0).mean()),
            }

    # by response magnitude, quartiles of the observed raw response in arm A
    qcut = pd.qcut(f["yA"], 4, labels=False, duplicates="drop")
    rows = []
    for lvl, g in f.groupby(qcut, sort=True):
        rows.append({"quartile": int(lvl), "n": int(len(g)),
                     "y_range": [float(g["yA"].min()), float(g["yA"].max())],
                     "q": corr(g["rA"].to_numpy(), g["rB"].to_numpy())})
    out["by_response_quartile"] = rows

    # by how strong the interaction is in arm A
    acut = pd.qcut(np.abs(f["rA"]), 4, labels=False, duplicates="drop")
    rows = []
    for lvl, g in f.groupby(acut, sort=True):
        rows.append({"quartile": int(lvl), "n": int(len(g)),
                     "abs_r_range": [float(np.abs(g["rA"]).min()), float(np.abs(g["rA"]).max())],
                     "q": corr(g["rA"].to_numpy(), g["rB"].to_numpy())})
    out["by_interaction_magnitude_quartile"] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="phase0c_measurement_ceiling.json")
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    res: dict = {}
    frame = matched_gdsc()
    print(f"matched GDSC1/GDSC2 pairs: {len(frame)}", flush=True)
    res["within_programme"] = {"source": "GDSC1 vs GDSC2", "methods": {}}
    for method in ("D1", "D2"):
        print(f"  point estimates, {method}", flush=True)
        entry = point_estimates(frame, method)
        entry["stratified"] = stratify(frame, method)
        entry["bootstrap"] = {}
        for mode in ("drug", "sample", "two_way"):
            print(f"  bootstrap {method}/{mode}", flush=True)
            entry["bootstrap"][mode] = cluster_bootstrap(frame, method, mode, args.n_boot)
        res["within_programme"]["methods"][method] = entry

    print("cross-resource references", flush=True)
    res["cross_resource"] = {}
    for name, fr in matched_cross_resource().items():
        rec = {"D2": point_estimates(fr, "D2")}
        rec["D2"]["bootstrap_drug"] = cluster_bootstrap(fr, "D2", "drug",
                                                        max(args.n_boot // 4, 50))
        res["cross_resource"][name] = rec
        print(f"  {name}: n={rec['D2']['n_pairs']} q={rec['D2']['q_r']:.3f}", flush=True)

    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

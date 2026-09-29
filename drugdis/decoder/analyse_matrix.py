"""Decoder-robustness analysis: is the representation ordering stable across
decoder classes, and does the component-wise interpretation survive?"""
from __future__ import annotations
import glob, itertools, json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

ROOT = str(paths.runs_dir() / "DecoderBench")
DEC = ("ridge", "bilinear", "tower")
SIDES = {"drug side (transcriptome = raw)": [("ECFP4", "raw"), ("MolCLR", "raw"),
                                             ("KPGT", "raw"), ("UniMol", "raw")],
         "transcriptome side (drug = ECFP4)": [("ECFP4", "raw"), ("ECFP4", "scGPT"),
                                               ("ECFP4", "CellPLM"), ("ECFP4", "BulkFormer")]}
METRICS = ("rawPCC", "sharedPCC", "intPCC")


def load(d, g, k, seed=3407):
    """For the two normalisation-free decoders the L2 penalty is part of the
    decoder definition, so it is selected on VALIDATION from the fixed grid,
    never on test. The tower keeps its frozen hyperparameters untouched."""
    cands = []
    for p in glob.glob(os.path.join(ROOT, f"{d}__{g}__{k}__seed{seed}*", "results.json")):
        r = json.load(open(p))
        r["_wd"] = r["args"]["weight_decay"]
        cands.append(r)
    if not cands:
        return None
    if k == "tower":
        base = [r for r in cands if r["_wd"] == 1e-5]
        return base[0] if base else None
    best = min(cands, key=lambda r: r["best_val_MSE_raw"])
    best["_n_penalties_tried"] = len(cands)
    return best


def label(side, pair):
    return pair[0] if side.startswith("drug") else pair[1]


def main() -> int:
    runs, missing = {}, []
    for side, pairs in SIDES.items():
        for pr in pairs:
            for k in DEC:
                r = load(*pr, k)
                if r is None:
                    missing.append(f"{pr[0]}__{pr[1]}__{k}")
                else:
                    runs[(pr, k)] = r
    if missing:
        print(f"missing {len(missing)} runs: {missing[:6]}\n")

    print("=" * 96)
    print("COMPONENT PROFILES, frozen val-MSE checkpoint, seed 3407")
    print("=" * 96)
    for side, pairs in SIDES.items():
        print(f"\n### {side}")
        print(f"{'representation':14s} {'decoder':9s} {'params':>12s} {'rawPCC':>8s} "
              f"{'sharedPCC':>10s} {'intPCC':>8s} {'A_int':>7s} {'sd(M yhat)':>11s} "
              f"{'L2 sel':>10s} {'val MSE':>9s}")
        for pr in pairs:
            for k in DEC:
                r = runs.get((pr, k))
                if not r:
                    continue
                t = r["frozen_valMSE__test"]
                wd = f"{r['_wd']:.0e}" + ("" if k == "tower"
                                          else f"/{r.get('_n_penalties_tried', 1)}")
                print(f"{label(side, pr):14s} {k:9s} {r['n_params']:12,d} {t['rawPCC']:8.4f} "
                      f"{t['sharedPCC']:10.4f} {t['intPCC']:8.4f} {t['A_int']:7.3f} "
                      f"{t['sd_M_yhat']:11.2e} {wd:>10s} {r['best_val_MSE_raw']:9.5f}")

    print("\n" + "=" * 96)
    print("RANKING STABILITY ACROSS DECODER CLASSES")
    print("=" * 96)
    out = {}
    for side, pairs in SIDES.items():
        print(f"\n### {side}")
        for m in METRICS:
            vals, order = {}, {}
            for k in DEC:
                v = [runs[(pr, k)]["frozen_valMSE__test"][m] if (pr, k) in runs else np.nan
                     for pr in pairs]
                vals[k] = v
                order[k] = [label(side, pairs[i]) for i in np.argsort(v)[::-1]]
            print(f"\n  {m}")
            for k in DEC:
                note = ""
                if m == "intPCC" and k == "ridge":
                    note = "   <- additive by construction, this ordering is noise"
                print(f"    {k:9s} best to worst: {' > '.join(order[k])}{note}")
            usable = [k for k in DEC if not (m == "intPCC" and k == "ridge")]
            for a, b in itertools.combinations(usable, 2):
                if any(np.isnan(vals[a])) or any(np.isnan(vals[b])):
                    continue
                rho = spearmanr(vals[a], vals[b]).correlation
                top1 = order[a][0] == order[b][0]
                top2 = set(order[a][:2]) == set(order[b][:2])
                print(f"      {a:9s} vs {b:9s}  Spearman {rho:+.3f}   top-1 same: {top1}"
                      f"   top-2 set same: {top2}")
                out.setdefault(side, {}).setdefault(m, {})[f"{a}_vs_{b}"] = {
                    "spearman": float(rho), "top1_same": bool(top1), "top2_same": bool(top2)}
            out.setdefault(side, {}).setdefault(m, {})["orderings"] = order
            out.setdefault(side, {}).setdefault(m, {})["values"] = {
                k: [None if np.isnan(x) else float(x) for x in vals[k]] for k in DEC}
    print("\n  note: each ranking has only four items, so Spearman takes values in")
    print("  {-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1}. The explicit orderings")
    print("  above carry more information than the coefficient.")
    json.dump(out, open(paths.work_dir() / "decoder_robustness.json", "w"),
              indent=1)
    print("\nwrote decoder_robustness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Three-seed confirmation of the transcriptome-side ordering under the two
decoders that can express interaction."""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

ROOT = str(paths.runs_dir() / "DecoderBench")
REPS = ("raw", "scGPT", "CellPLM", "BulkFormer")
SEEDS = (3407, 3408, 3409)
METRICS = ("rawPCC", "sharedPCC", "intPCC")


def load(g, k, seed):
    c = []
    for p in glob.glob(os.path.join(ROOT, f"ECFP4__{g}__{k}__seed{seed}*", "results.json")):
        r = json.load(open(p))
        r["_wd"] = r["args"]["weight_decay"]
        c.append(r)
    if not c:
        return None
    if k == "tower":
        b = [r for r in c if r["_wd"] == 1e-5]
        return b[0] if b else None
    return min(c, key=lambda r: r["best_val_MSE_raw"])


def main() -> int:
    out = {}
    for k in ("bilinear", "tower"):
        print("=" * 92)
        print(f"decoder = {k}")
        print("=" * 92)
        for m in METRICS:
            print(f"\n  {m}")
            per_seed, orders = {}, {}
            for s in SEEDS:
                v = []
                for g in REPS:
                    r = load(g, k, s)
                    v.append(r["frozen_valMSE__test"][m] if r else np.nan)
                per_seed[s] = v
                if not any(np.isnan(v)):
                    orders[s] = [REPS[i] for i in np.argsort(v)[::-1]]
                    print(f"    seed {s}: " + "  ".join(f"{REPS[i]}={v[i]:+.4f}"
                                                        for i in np.argsort(v)[::-1]))
            avail = [s for s in SEEDS if s in orders]
            if len(avail) >= 2:
                mean = np.nanmean([per_seed[s] for s in avail], axis=0)
                sd = np.nanstd([per_seed[s] for s in avail], axis=0, ddof=1)
                o = [REPS[i] for i in np.argsort(mean)[::-1]]
                print(f"    mean over {len(avail)} seeds: " +
                      "  ".join(f"{REPS[i]}={mean[i]:+.4f}+-{sd[i]:.4f}" for i in np.argsort(mean)[::-1]))
                print(f"    mean ordering: {' > '.join(o)}")
                top1 = [orders[s][0] for s in avail]
                print(f"    top-1 per seed: {top1}   unanimous: {len(set(top1)) == 1}")
                rhos = [spearmanr(per_seed[a], per_seed[b]).correlation
                        for i, a in enumerate(avail) for b in avail[i + 1:]]
                if rhos:
                    print(f"    pairwise seed-to-seed Spearman: "
                          f"{['%+.2f' % x for x in rhos]}  mean {np.mean(rhos):+.2f}")
                out.setdefault(k, {})[m] = {
                    "per_seed": {str(s): [float(x) for x in per_seed[s]] for s in avail},
                    "mean": [float(x) for x in mean], "sd": [float(x) for x in sd],
                    "mean_ordering": o, "top1_per_seed": top1,
                    "top1_unanimous": bool(len(set(top1)) == 1),
                    "seed_to_seed_spearman": [float(x) for x in rhos]}
    json.dump(out, open(paths.work_dir() / "decoder_confirm.json", "w"), indent=1)
    print("\nwrote decoder_confirm.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

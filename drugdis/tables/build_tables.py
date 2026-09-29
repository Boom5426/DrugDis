"""Paper-wide rerun, step 3: write the canonical tables.

Every table is assembled from an already-frozen result file. Nothing is refitted.
The only computation here is the two-way decomposition of the response on panels
that had not yet been tabulated, which is step 2.2 of the rerun and is a property
of the data, not a model.
"""
from __future__ import annotations
import glob, hashlib, json, os, sys
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
sys.path.insert(0, str(paths.CODE / "decomposition"))
sys.path.insert(0, str(paths.CODE / "splits"))
from decompositions import DRUG, SAMPLE, AdditiveProjection    # noqa: E402
import make_manifests as mm                                    # noqa: E402

WORK = str(paths.work_dir()) + "/"
MAN = str(paths.MANIFESTS) + "/"
OUT = str(paths.tables_dir()) + "/"
SEEDS = (3407, 3408, 3409, 3410, 3411)
os.makedirs(OUT, exist_ok=True)


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write(df, name):
    p = OUT + name
    df.to_csv(p, index=False)
    print(f"  wrote {name:34s} {len(df):5d} rows  sha {sha(p)[:16]}")
    return {"file": name, "rows": int(len(df)), "sha256": sha(p)}


def geometry(fr):
    d, dn = pd.factorize(fr[DRUG], sort=False)
    s, sn = pd.factorize(fr[SAMPLE], sort=False)
    nd, ns = len(dn), len(sn)
    g = coo_matrix((np.ones(len(fr)), (d, nd + s)), shape=(nd + ns, nd + ns))
    c, _ = connected_components(g + g.T, directed=False)
    return nd, ns, int(c), nd + ns - c


def decomp(fr, label, panel_kind):
    p = AdditiveProjection(fr, [DRUG, SAMPLE])
    y = fr["Sensitivity"].to_numpy(np.float64)
    My = p.residual(y); Hy = y - My
    nd, ns, c, dh = geometry(fr)
    vy, vb, vr = float(y.var()), float(Hy.var()), float(My.var())
    # The two main effects are not mutually orthogonal on an unbalanced support, so a
    # single split does not exist. Report each marginal alone and the unique part of
    # each, so a figure can state which one it means.
    sd_only = 1.0 - float(AdditiveProjection(fr, [DRUG]).residual(y).var()) / vy
    ss_only = 1.0 - float(AdditiveProjection(fr, [SAMPLE]).residual(y).var()) / vy
    tot = vb / vy
    return {"panel": label, "kind": panel_kind, "n_pairs": len(fr), "n_drugs": nd,
            "n_samples": ns, "components": c, "rank_H": dh, "d_H_over_N": dh / len(fr),
            "var_y": vy, "var_shared": vb, "var_interaction": vr,
            "share_shared_pct": 100 * vb / vy, "share_interaction_pct": 100 * vr / vy,
            "share_drug_alone_pct": 100 * sd_only, "share_sample_alone_pct": 100 * ss_only,
            "share_drug_unique_pct": 100 * (tot - ss_only),
            "share_sample_unique_pct": 100 * (tot - sd_only),
            "share_drug_sample_overlap_pct": 100 * (sd_only + ss_only - tot),
            "additivity_gap_frac": (vy - vb - vr) / vy,
            "corr_shared_interaction": float(np.corrcoef(Hy, My)[0, 1])}


def main() -> int:
    # merge rather than replace: T04/T05/T06, T12 and T15 onward are written by other
    # builders, and T03, T10 and T13 by the audited builders
    # rebuild_measurement_bootstrap_v2.py and rebuild_valmse_identity_tables_v1.py
    reg = json.load(open(OUT + "TABLES.json")) if os.path.exists(OUT + "TABLES.json") else {}
    print("building canonical tables\n")

    # ---- T01 substrate
    rec = json.load(open(MAN + "MANIFEST.json"))
    prov = json.load(open(paths.data_dir() / "omics_baseline_frozen" /
                          "ccle_anchored_provenance.json"))
    rows = [{"item": "substrate_version", "value": rec["config"]["substrate_version"]},
            {"item": "transcriptome", "value": os.path.basename(prov["output"])},
            {"item": "transcriptome_sha256", "value": prov["output_sha256"]},
            {"item": "transcriptome_source", "value": "CCLE, gene whitelist 15,961"},
            {"item": "n_samples_transcriptome", "value": prov["n_samples"]},
            {"item": "panel_pairs", "value": rec["panel"]["pairs"]},
            {"item": "panel_samples", "value": rec["panel"]["samples"]},
            {"item": "panel_drugs", "value": rec["panel"]["drugs"]},
            {"item": "excluded_projects", "value": ",".join(rec["config"]["excluded_projects"])},
            {"item": "duplicate_rule", "value": rec["config"]["duplicate_rule"]["kind"]},
            {"item": "scale_rule", "value": rec["config"]["scale_rule"]["kind"]},
            {"item": "checkpoint_rule", "value": "argmin validation MSE_raw"},
            {"item": "sample_list_sha256", "value": rec["panel"]["sample_list_sha256"]},
            {"item": "drug_list_sha256", "value": rec["panel"]["drug_list_sha256"]},
            {"item": "generator_sha256", "value": rec["generator_sha256"]}]
    for k, v in rec["manifest_sha256"].items():
        for sp, h in v.items():
            rows.append({"item": f"manifest_{k}_{sp}_sha256", "value": h})
    for k, v in rec["response_resource_composition"].items():
        rows.append({"item": f"response_rows_{k}", "value": v})
    reg["T01"] = write(pd.DataFrame(rows), "T01_substrate.csv")

    # ---- T02 decomposition
    cfg = paths.load_config(paths.CONFIG)
    dg = mm.load_transcriptome(cfg)
    panel = mm.build_panel(cfg, dg)
    d2 = [decomp(panel, "full_panel", "all")]
    for sp, col in (("LCLO", SAMPLE), ("LSO", DRUG)):
        for sd in SEEDS:
            m = json.load(open(MAN + f"{sp}_seed{sd}.json"))
            fr = panel[panel[col].astype(str).isin(set(m["test"]))].reset_index(drop=True)
            d2.append(decomp(fr, f"{sp}_seed{sd}_test", sp))
    reg["T02"] = write(pd.DataFrame(d2), "T02_decomposition.csv")

    # ---- T07 constructive comparison. T03 (measurement reference) and T10
    # (error geometry) are not written here: their canonical versions come from
    # rebuild_measurement_bootstrap_v2.py and rebuild_valmse_identity_tables_v1.py.
    rows7 = []
    runs = paths.runs_dir()
    for sp, root in (("LCLO", str(runs / "Phase2B_clean" / "ECFP4__baseline" / "LCLO")),
                     ("LSO", str(runs / "Phase2B_LSO" / "ECFP4__baseline" / "LSO"))):
        for arm in ("M0", "M4"):
            for sd in SEEDS:
                f = os.path.join(root, f"{arm}_seed{sd}", "results.json")
                if not os.path.exists(f):
                    continue
                r = json.load(open(f))
                t = r["frozen_valMSE__test"]
                rows7.append({"split": sp, "arm": arm, "seed": sd, "rule": "frozen_valMSE",
                              **{k: t[k] for k in ("n", "rawPCC", "sharedPCC", "intPCC",
                                                   "A_int", "A_shared", "MSE_raw",
                                                   "E_shared", "E_interaction")},
                              "best_epoch": r["best_valmse_epoch"],
                              "n_params": r["n_params"]})
    reg["T07"] = write(pd.DataFrame(rows7), "T07_constructive_m0_m4.csv")

    # ---- T08 PDO
    pdo = json.load(open(WORK + "pdo_stress_test.json"))
    pre = json.load(open(WORK + "pdo_preflight.json"))
    rows = []
    for run, v in pdo["runs"].items():
        arm, sd = run.split("_seed")
        for grp, prof in v.items():
            if not isinstance(prof, dict):
                continue
            # Two distinct things, previously conflated in one column named
            # scale_compatible. `role` is the frozen protocol: the primary pooled
            # estimate is the UMPDO family, fixed before this rerun. `preflight`
            # is an automated three-condition diagnostic against the CCLE gene
            # level. They agree on four of five resources; UMPDO3 sits 0.005
            # outside the level-mean threshold of 1.0 while matching the family on
            # zero fraction and range. The threshold is not moved to accommodate
            # it, and the primary cohort is not redefined to exclude it; both
            # facts are carried so a panel can show the disagreement.
            comp = pre["resources"].get(grp, {}).get("comparable_to_ccle")
            role = ("primary" if grp == "COMPARABLE(UMPDO1+2+3)"
                    else "pooled_all" if grp == "ALL" else "single_resource")
            rows.append({"cohort": grp, "arm": arm, "seed": int(sd),
                         "role": role,
                         "in_primary": grp in ("UMPDO1", "UMPDO2", "UMPDO3"),
                         "preflight_comparable": ("" if comp is None else bool(comp)),
                         **{k: prof[k] for k in ("n", "rawPCC", "sharedPCC", "intPCC",
                                                 "A_int", "E_shared", "E_interaction")}})
    reg["T08"] = write(pd.DataFrame(rows), "T08_pdo_stress.csv")

    # ---- T09 decoder robustness
    dr = json.load(open(WORK + "decoder_robustness.json"))
    dc = json.load(open(WORK + "decoder_confirm.json"))
    rows = []
    for side, per_m in dr.items():
        for metric, v in per_m.items():
            # the explicit orderings the comparison is computed from. They were
            # already carried in decoder_robustness.json and dropped here, which
            # left the figure with a coefficient it could not unpack.
            for a_vs_b, s in v.items():
                if isinstance(s, dict) and "spearman" in s:
                    a, b = a_vs_b.split("_vs_")
                    rows.append({"scope": "seed3407", "side": side, "metric": metric,
                                 "comparison": a_vs_b, "spearman": s["spearman"],
                                 "top1_same": s["top1_same"], "top2_same": s["top2_same"],
                                 "ordering_a": " > ".join(v["orderings"][a]),
                                 "ordering_b": " > ".join(v["orderings"][b])})
    for dec, per_m in dc.items():
        for metric, v in per_m.items():
            rows.append({"scope": "3seed", "side": "transcriptome", "metric": metric,
                         "comparison": f"{dec}_seed_stability",
                         "spearman": float(np.mean(v["seed_to_seed_spearman"])),
                         "top1_same": v["top1_unanimous"],
                         "top2_same": None,
                         "mean_ordering": " > ".join(v["mean_ordering"])})
    reg["T09"] = write(pd.DataFrame(rows), "T09_decoder_robustness.csv")

    json.dump(reg, open(OUT + "TABLES.json", "w"), indent=1)
    print(f"\nwrote {OUT}TABLES.json with {len(reg)} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())

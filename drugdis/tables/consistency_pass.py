"""Step 4: numerical consistency pass over the canonical tables.

Checks that do not need the manuscript text:
  1. every table listed in TABLES.json exists and matches its recorded SHA-256;
  2. the frozen mathematical identities hold in every table row;
  3. panel sizes agree across tables that describe the same panel;
  4. no deprecated metric column appears in any canonical table;
  5. the constructive and stress-test numbers in T07/T08 agree with the frozen
     documents they were written from.
A failure here is a reproducibility blocker, which is one of the three conditions
under which a new run is permitted.
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

T = str(paths.tables_dir()) + "/"
DEPRECATED = ("dspcc", "iepcc", "dsr10", "zscore_pcc")


def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    reg = json.load(open(T + "TABLES.json"))
    fails, checks = [], 0
    print("=== 1. table integrity ===")
    for k, v in sorted(reg.items()):
        p = T + v["file"]
        ok = os.path.exists(p) and sha(p) == v["sha256"]
        checks += 1
        if not ok:
            fails.append(f"{v['file']}: missing or checksum mismatch")
        print(f"  {'OK ' if ok else 'BAD'} {v['file']:36s} {v['rows']:5d} rows")

    print("\n=== 2. frozen identities ===")
    d2 = pd.read_csv(T + "T02_decomposition.csv")
    worst = d2["additivity_gap_frac"].abs().max()
    checks += 1
    print(f"  {'OK ' if worst < 1e-8 else 'BAD'} Var(y) = Var(Hy) + Var(My): worst relative gap {worst:.2e}")
    if worst >= 1e-8:
        fails.append(f"additivity identity violated, {worst:.2e}")
    worst = d2["corr_shared_interaction"].abs().max()
    checks += 1
    print(f"  {'OK ' if worst < 1e-6 else 'BAD'} corr(Hy, My) = 0: worst |corr| {worst:.2e}")
    if worst >= 1e-6:
        fails.append(f"orthogonality violated, {worst:.2e}")

    for name in ("T07_constructive_m0_m4.csv", "T08_pdo_stress.csv"):
        d = pd.read_csv(T + name)
        if not {"MSE_raw", "E_shared", "E_interaction"} <= set(d.columns):
            continue
        r = (d["E_shared"] + d["E_interaction"] - d["MSE_raw"]).abs() / d["MSE_raw"]
        checks += 1
        ok = r.max() < 1e-6
        print(f"  {'OK ' if ok else 'BAD'} MSE_raw = E_shared + E_interaction in {name}: "
              f"worst relative error {r.max():.2e}")
        if not ok:
            fails.append(f"error partition violated in {name}, {r.max():.2e}")

    print("\n=== 3. panel-size agreement ===")
    t1 = pd.read_csv(T + "T01_substrate.csv").set_index("item")["value"]
    full = d2[d2.panel == "full_panel"].iloc[0]
    for key, got in (("panel_pairs", full.n_pairs), ("panel_samples", full.n_samples),
                     ("panel_drugs", full.n_drugs)):
        checks += 1
        ok = int(t1[key]) == int(got)
        print(f"  {'OK ' if ok else 'BAD'} {key}: T01 {t1[key]} vs T02 {got}")
        if not ok:
            fails.append(f"{key} disagrees between T01 and T02")

    print("\n=== 4. deprecated metrics absent from canonical tables ===")
    for k, v in sorted(reg.items()):
        d = pd.read_csv(T + v["file"], nrows=5)
        bad = [c for c in d.columns if any(x in c.lower() for x in DEPRECATED)]
        checks += 1
        if bad:
            fails.append(f"{v['file']} exposes deprecated columns {bad}")
        print(f"  {'OK ' if not bad else 'BAD'} {v['file']:36s} {bad if bad else ''}")

    print("\n=== 5. agreement with the frozen documents ===")
    t7 = pd.read_csv(T + "T07_constructive_m0_m4.csv")
    for sp, want_int in (("LCLO", -0.0081), ("LSO", +0.0174)):
        s = t7[t7.split == sp].pivot(index="seed", columns="arm", values="intPCC")
        got = float((s["M4"] - s["M0"]).mean())
        checks += 1
        ok = abs(got - want_int) < 5e-4
        print(f"  {'OK ' if ok else 'BAD'} {sp} d_intPCC(M4-M0) = {got:+.4f}, document says {want_int:+.4f}")
        if not ok:
            fails.append(f"{sp} constructive delta disagrees with the frozen document")
    t8 = pd.read_csv(T + "T08_pdo_stress.csv")
    s = (t8[t8.cohort == "COMPARABLE(UMPDO1+2+3)"]
         .pivot(index="seed", columns="arm", values="intPCC"))
    got = float((s["M4"] - s["M0"]).mean())
    checks += 1
    ok = abs(got - 0.0198) < 5e-4
    print(f"  {'OK ' if ok else 'BAD'} PDO primary d_intPCC = {got:+.4f}, document says +0.0198")
    if not ok:
        fails.append("PDO delta disagrees with the frozen document")

    print(f"\n{checks - len(fails)} of {checks} checks passed")
    json.dump({"checks": checks, "failed": len(fails), "failures": fails},
              open(T + "consistency_pass.json", "w"), indent=1)
    if fails:
        print("FAILURES:")
        for f in fails:
            print("  " + f)
        return 2
    print("numerical consistency pass: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

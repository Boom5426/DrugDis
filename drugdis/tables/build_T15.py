"""T15: the prespecified input-compatibility check for the PDO cohorts.

Panel 4g needs the continuous quantities the check is computed from, not the
binary verdict, because the verdict depends on a threshold that falls inside one
scale family while the continuous evidence does not separate it. Those numbers
already exist in the frozen input-compatibility record; this table promotes them
to canonical so a figure can read them through the sanctioned channel.

No computation happens here beyond reshaping. Nothing is recomputed and no
threshold is moved.
"""
from __future__ import annotations
import hashlib, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

SRC = str(paths.work_dir() / "pdo_preflight.json")
T = str(paths.tables_dir()) + "/"
PRIMARY = {"UMPDO1", "UMPDO2", "UMPDO3"}

d = json.load(open(SRC))
ref = d["reference"]
rows = [{"cohort": "CCLE (training reference)", "role": "reference",
         "in_primary": False, "preflight_comparable": True,
         "n_samples": None, "n_genes": None,
         **{k: ref[k] for k in ("mean", "sd", "median", "min", "max")},
         "zero_frac": ref["zero"], "level_mean": ref["level_mean"],
         "level_sd": ref["level_sd"], "level_gap_vs_ccle": 0.0,
         "response_rows": None}]
for name, v in d["resources"].items():
    rows.append({"cohort": name, "role": "PDO cohort",
                 "in_primary": name in PRIMARY,
                 "preflight_comparable": bool(v["comparable_to_ccle"]),
                 "n_samples": v["n"], "n_genes": v["genes"],
                 **{k: v[k] for k in ("mean", "sd", "median", "min", "max")},
                 "zero_frac": v["zero_frac"], "level_mean": v["level_mean"],
                 "level_sd": v["level_sd"],
                 "level_gap_vs_ccle": abs(v["level_mean"] - ref["level_mean"]),
                 "response_rows": v["response_rows"]})

df = pd.DataFrame(rows)
p = T + "T15_input_compatibility.csv"
df.to_csv(p, index=False)
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
reg = json.load(open(T + "TABLES.json"))
reg["T15"] = {"file": "T15_input_compatibility.csv", "rows": len(df), "sha256": h}
json.dump({k: reg[k] for k in sorted(reg)}, open(T + "TABLES.json", "w"), indent=1)
print(df[["cohort", "in_primary", "preflight_comparable", "level_mean",
          "level_gap_vs_ccle", "zero_frac", "min", "max"]].to_string(index=False))
print(f"\nwrote T15_input_compatibility.csv  {len(df)} rows  sha {h[:16]}")
print("threshold in the frozen check: level gap < 1.0, zero fraction within "
      f"[{0.5*ref[chr(34)+chr(34)] if False else round(0.5*ref[chr(122)+chr(101)+chr(114)+chr(111)],4)}, "
      f"{round(2.0*ref[chr(122)+chr(101)+chr(114)+chr(111)],4)}], min >= {round(ref[chr(109)+chr(105)+chr(110)]-0.5,3)}")

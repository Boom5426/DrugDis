"""Canonical split manifests for the RISE manuscript-wide rerun.

The published pipeline split on `df_domain[split_col].unique()`, whose order
follows filtered dataframe row order, so the split was a function of upstream
filtering rather than of the seed alone. This replaces that with an explicit
manifest built from a canonically sorted entity list, and writes a checksum of
every input that can change the panel.

    python make_manifests.py --config substrate_config.json --out <dir>

Nothing here decides the substrate. It reads the decisions from the config and
records them, so that the config plus this file fully determine the manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DRUG, SAMPLE = "SMILES", "Sample_ID"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_list(items) -> str:
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\n")
    return h.hexdigest()


def load_transcriptome(cfg: dict) -> pd.DataFrame:
    """Apply the frozen duplicate rule and scale rule, both taken from the config."""
    dg = pd.read_parquet(cfg["transcriptome_path"], engine="fastparquet")
    dg.index = dg.index.astype(str)
    rule = cfg["duplicate_rule"]
    if rule["kind"] == "already_unique":
        # Option A\': the matrix comes from a single resource and carries one row
        # per Sample_ID by construction. Verified by build_ccle_substrate.py.
        assert dg.index.is_unique, "already_unique declared but the index is not unique"
    elif rule["kind"] == "keep_last":
        dg = dg[~dg.index.duplicated(keep="last")]
    elif rule["kind"] == "genewise_mean":
        dg = dg.groupby(level=0).mean()
    elif rule["kind"] == "resource_priority":
        raise NotImplementedError("resource_priority needs a per-row resource label")
    else:
        raise ValueError(rule["kind"])
    kind = cfg["scale_rule"]["kind"]
    if kind == "single_resource":
        # No transformation is applied and none is needed: every row comes from one
        # resource, so the matrix is single-scale by construction rather than by
        # correction. Asserted, not assumed.
        v = dg.to_numpy(np.float32)
        assert np.nanmax(v) < cfg["scale_rule"]["max_value_assert"], "single-scale assertion failed"
    elif kind != "none":
        raise NotImplementedError("no scale harmonisation other than single_resource is frozen")
    assert dg.index.is_unique, "transcriptome index is not unique after the duplicate rule"
    return dg.fillna(0.0).astype(np.float32)


def build_panel(cfg: dict, dg: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_parquet(cfg["master_table_path"], engine="fastparquet")
    resp = pd.read_parquet(cfg["drug_response_path"], engine="fastparquet")
    resp = resp.groupby([DRUG, SAMPLE])["Sensitivity"].mean().reset_index()
    with open(cfg["drug_features_path"], "rb") as fh:
        drug_feat = pickle.load(fh)
    keep = resp[DRUG].isin(drug_feat.keys()) & resp[SAMPLE].isin(set(dg.index))
    panel = resp[keep]
    ids = master.index[master["Model_Type"] == cfg["model_type"]]
    panel = panel[panel[SAMPLE].isin(ids)]
    excl = set(cfg.get("excluded_projects", []))
    if excl:
        drop = set(master.index[master["Project"].isin(excl)])
        panel = panel[~panel[SAMPLE].isin(drop)]
    return panel.reset_index(drop=True)


def split_entities(entities, seed: int, test_ratio: float, val_ratio: float) -> dict:
    """Canonical: sort first, then split. The order of the input never matters."""
    canon = np.array(sorted(set(map(str, entities))))
    trval, test = train_test_split(canon, test_size=test_ratio, random_state=seed)
    tr, val = train_test_split(trval, test_size=val_ratio / (1.0 - test_ratio),
                               random_state=seed)
    return {"train": sorted(tr.tolist()), "val": sorted(val.tolist()),
            "test": sorted(test.tolist())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp", default=None, help="ISO timestamp; defaults to now (UTC)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)

    dg = load_transcriptome(cfg)
    panel = build_panel(cfg, dg)
    print(f"panel: {len(panel):,} pairs, {panel[SAMPLE].nunique():,} samples, "
          f"{panel[DRUG].nunique():,} drugs", flush=True)

    raw = pd.read_parquet(cfg["drug_response_path"], engine="fastparquet")
    key = set(zip(panel[DRUG], panel[SAMPLE]))
    sel = raw[[tuple(x) in key for x in zip(raw[DRUG], raw[SAMPLE])]]
    comp = sel["Project"].value_counts()
    print("\nresponse-resource composition of the retained panel "
          "(raw rows before the pair-level mean):")
    for k, v in comp.items():
        print(f"    {k:14s} {v:>9,d}  {100*v/len(sel):5.2f}%   samples {sel.loc[sel.Project==k, SAMPLE].nunique():5d}"
              f"  drugs {sel.loc[sel.Project==k, DRUG].nunique():6d}")
    print(f"    {'TOTAL':14s} {len(sel):>9,d}")

    manifests, checks = {}, {}
    for split_type, col in (("LCLO", SAMPLE), ("LSO", DRUG)):
        for seed in cfg["seeds"]:
            key = f"{split_type}_seed{seed}"
            m = split_entities(panel[col], seed, cfg["test_ratio"], cfg["val_ratio"])
            manifests[key] = m
            checks[key] = {k: sha256_list(v) for k, v in m.items()}
            n = {k: int(panel[col].astype(str).isin(v).sum()) for k, v in m.items()}
            print(f"  {key}: entities {[len(v) for v in m.values()]} rows {n}", flush=True)
            with open(os.path.join(a.out, key + ".json"), "w") as fh:
                json.dump(m, fh)

    record = {
        "created_utc": a.stamp or datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "input_hashes": {k: sha256_file(cfg[k]) for k in
                         ("transcriptome_path", "master_table_path", "drug_response_path")},
        "response_resource_composition": comp.to_dict(),
        "panel": {"pairs": int(len(panel)), "samples": int(panel[SAMPLE].nunique()),
                  "drugs": int(panel[DRUG].nunique()),
                  "sample_list_sha256": sha256_list(sorted(set(panel[SAMPLE].astype(str)))),
                  "drug_list_sha256": sha256_list(sorted(set(panel[DRUG].astype(str))))},
        "manifest_sha256": checks,
        "generator_sha256": sha256_file(os.path.abspath(__file__)),
    }
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print("\nwrote", os.path.join(a.out, "MANIFEST.json"))
    print("panel sample-list sha256:", record["panel"]["sample_list_sha256"][:16])
    print("panel drug-list  sha256:", record["panel"]["drug_list_sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

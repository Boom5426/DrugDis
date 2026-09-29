"""Split manifests: entity-level, disjoint, deterministic, order-free."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

import make_manifests as mm


def names(n, prefix="e"):
    return [f"{prefix}{i:04d}" for i in range(n)]


@pytest.mark.parametrize("seed", [3407, 3411])
def test_partition_is_disjoint_and_complete(seed):
    ents = names(986)
    m = mm.split_entities(ents, seed, 0.15, 0.15)
    parts = [set(m[k]) for k in ("train", "val", "test")]
    assert sum(len(p) for p in parts) == len(ents)
    assert set().union(*parts) == set(ents)
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])


def test_sizes_match_the_benchmark_manifests():
    """986 cell lines split 690/148/148, as in manifests/LCLO_seed*.json."""
    m = mm.split_entities(names(986), 3407, 0.15, 0.15)
    assert [len(m[k]) for k in ("train", "val", "test")] == [690, 148, 148]


def test_deterministic_and_independent_of_input_order():
    ents = names(500)
    a = mm.split_entities(ents, 3409, 0.15, 0.15)
    b = mm.split_entities(list(reversed(ents)) + ents[:10], 3409, 0.15, 0.15)
    assert a == b
    assert mm.split_entities(ents, 3410, 0.15, 0.15) != a


def test_panel_eligibility(tmp_path):
    """A pair enters the benchmark dataset only if the sample is a cell line with an
    expression profile, is not in an excluded project, and the compound has a
    fingerprint; duplicate measurements of a pair are averaged."""
    master = pd.DataFrame({"Project": ["A", "A", "Tavor", "B"],
                           "Model_Type": ["Cell Line", "Cell Line", "Cell Line", "PDO"]},
                          index=pd.Index(["s1", "s2", "s3", "s4"], name="Sample_ID"))
    resp = pd.DataFrame({"SMILES": ["C", "C", "CC", "C", "C", "C", "CCO"],
                         "Sample_ID": ["s1", "s1", "s1", "s3", "s4", "s5", "s2"],
                         "Sensitivity": [0.2, 0.4, 0.5, 0.1, 0.3, 0.9, 0.7],
                         "Project": ["A"] * 7})
    expr = pd.DataFrame(np.ones((4, 3), np.float32), index=["s1", "s2", "s3", "s4"],
                        columns=["g1", "g2", "g3"])
    master.to_parquet(tmp_path / "master.parquet", engine="fastparquet")
    resp.to_parquet(tmp_path / "resp.parquet", engine="fastparquet")
    expr.to_parquet(tmp_path / "expr.parquet", engine="fastparquet")
    with open(tmp_path / "fp.pickle", "wb") as fh:
        pickle.dump({"C": np.zeros(4), "CC": np.zeros(4)}, fh)
    cfg = {"transcriptome_path": str(tmp_path / "expr.parquet"),
           "master_table_path": str(tmp_path / "master.parquet"),
           "drug_response_path": str(tmp_path / "resp.parquet"),
           "drug_features_path": str(tmp_path / "fp.pickle"),
           "model_type": "Cell Line", "excluded_projects": ["Tavor"],
           "duplicate_rule": {"kind": "already_unique"},
           "scale_rule": {"kind": "single_resource", "max_value_assert": 100.0}}
    panel = mm.build_panel(cfg, mm.load_transcriptome(cfg))
    got = {(r.SMILES, r.Sample_ID): r.Sensitivity for r in panel.itertuples()}
    # s3 is Tavor, s4 is an organoid, s5 has no profile, CCO has no fingerprint
    assert set(got) == {("C", "s1"), ("CC", "s1")}
    assert got[("C", "s1")] == pytest.approx(0.3)

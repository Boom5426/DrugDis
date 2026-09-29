"""The public evaluation API against the trainer's formulas and exact identities."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from drugdis.evaluate import component_profile, decompose  # noqa: E402


@pytest.fixture
def table():
    rng = np.random.default_rng(3)
    rows = [(f"d{d}", f"s{s}") for d in range(25) for s in range(30) if rng.random() < 0.5]
    frame = pd.DataFrame(rows, columns=["SMILES", "Sample_ID"])
    frame["Sensitivity"] = rng.normal(size=len(frame)) + frame["SMILES"].str.len()
    frame["prediction"] = frame["Sensitivity"] + rng.normal(0, 0.7, len(frame))
    return frame


def test_decompose_shares_add_up(table):
    hy, my, s = decompose(table)
    assert np.allclose(hy + my, table["Sensitivity"])
    assert s["share_additive_pct"] + s["share_interaction_pct"] == pytest.approx(100, abs=1e-7)
    assert abs(s["corr_additive_interaction"]) < 1e-7
    assert s["rank_H"] == s["n_drugs"] + s["n_samples"] - s["components"]


def test_exact_error_attribution_and_r2_identity(table):
    p = component_profile(table)
    assert p["MSE_raw"] == pytest.approx(p["E_shared"] + p["E_interaction"], rel=1e-9)
    hy, my, _ = decompose(table)
    _, myh, _ = decompose(table, response="prediction")
    direct = 1 - np.sum((myh - my) ** 2) / np.sum(my ** 2)
    assert p["R2_interaction"] == pytest.approx(direct, abs=1e-8)


def test_repeated_pairs_are_rejected(table):
    with pytest.raises(ValueError):
        component_profile(pd.concat([table, table.iloc[:1]], ignore_index=True))


def test_matches_the_trainer(table, tmp_path, monkeypatch):
    """Same numbers as component_profile in drugdis/models/train_2a2.py."""
    pytest.importorskip("torch")
    monkeypatch.setenv("DRUGDIS_DATA", str(tmp_path))
    for sub in ("models", "decomposition", ""):
        monkeypatch.syspath_prepend(os.path.join(REPO, "drugdis", sub))
    import train_2a2
    from decompositions import AdditiveProjection
    proj = AdditiveProjection(table, ["SMILES", "Sample_ID"])
    ref = train_2a2.component_profile(table["Sensitivity"].to_numpy(float),
                                      table["prediction"].to_numpy(float), proj)
    got = component_profile(table)
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-15), k

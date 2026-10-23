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


def test_additive_prediction_emits_no_interaction(table):
    """Methods: intPCC is undefined, not a correlation of rounding noise, for M yhat = 0."""
    rng = np.random.default_rng(5)
    drugs, samples = table["SMILES"].unique(), table["Sample_ID"].unique()
    additive = (table["SMILES"].map(dict(zip(drugs, rng.normal(size=len(drugs)))))
                + table["Sample_ID"].map(dict(zip(samples, rng.normal(size=len(samples))))))
    for yhat in (additive.to_numpy(), np.full(len(table), 0.37), np.full(len(table), 1.0)):
        p = component_profile(table, yhat=yhat)
        assert not p["emits_interaction"]
        assert np.isnan(p["intPCC"])
        assert p["A_int"] <= 1e-8
        assert p["R2_interaction"] == pytest.approx(0, abs=1e-12)
        assert p["MSE_raw"] == pytest.approx(p["E_shared"] + p["E_interaction"], rel=1e-9)


def test_declared_additive_predictor_emits_no_interaction(table):
    p = component_profile(table, additive_by_construction=True)
    q = component_profile(table)
    assert q["emits_interaction"] and not p["emits_interaction"]
    assert np.isnan(p["intPCC"]) and np.isfinite(q["intPCC"])
    assert p["A_int"] == q["A_int"] and p["R2_interaction"] == q["R2_interaction"]


def test_uninformative_emission_is_scored(table):
    """Shuffled predictions emit interaction, so they get a defined, near-zero score."""
    shuffled = np.random.default_rng(11).permutation(table["Sensitivity"].to_numpy())
    p = component_profile(table, yhat=shuffled)
    assert p["emits_interaction"] and p["A_int"] > 0.1
    assert np.isfinite(p["intPCC"])


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

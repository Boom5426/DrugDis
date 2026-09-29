"""The orthogonal response decomposition on small synthetic supports.

The additive projection H is checked against an explicit least-squares fit on the
drug and sample indicator design, which is only feasible at this size; the
alternating-projection solver is what the analyses use at 3.14 million pairs.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from decompositions import DRUG, SAMPLE, AdditiveProjection, orthogonality_report

TOL = 1e-8


def support(n_drug, n_sample, density, seed, blocks=1):
    """Random incomplete drug x sample support, optionally split into disconnected blocks."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(blocks):
        for d in range(n_drug):
            for s in range(n_sample):
                if rng.random() < density:
                    rows.append((f"d{b}_{d}", f"s{b}_{s}"))
    df = pd.DataFrame(rows, columns=[DRUG, SAMPLE])
    y = rng.normal(size=len(df)) + df[DRUG].str.len() * 0.1
    return df, y.to_numpy() if hasattr(y, "to_numpy") else y


def design(df):
    d = pd.get_dummies(df[DRUG]).to_numpy(float)
    s = pd.get_dummies(df[SAMPLE]).to_numpy(float)
    return np.hstack([d, s])


def exact_residual(df, y):
    x = design(df)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


@pytest.fixture(params=[0, 1, 2])
def case(request):
    return support(12, 15, 0.6, request.param)


def test_matches_least_squares(case):
    df, y = case
    my = AdditiveProjection(df, [DRUG, SAMPLE]).residual(y)
    assert np.max(np.abs(my - exact_residual(df, y))) < TOL


def test_idempotent(case):
    df, y = case
    p = AdditiveProjection(df, [DRUG, SAMPLE])
    my = p.residual(y)
    assert np.max(np.abs(p.residual(my) - my)) < TOL


def test_orthogonal_and_variance_additive(case):
    df, y = case
    hy, my = AdditiveProjection(df, [DRUG, SAMPLE]).split(y)
    rep = orthogonality_report(y, hy, my)
    assert abs(rep["corr_b_r"]) < 1e-7
    assert abs(rep["additivity_gap_frac_of_var_y"]) < 1e-9


def test_interaction_sums_to_zero_within_every_drug_and_sample(case):
    df, y = case
    my = AdditiveProjection(df, [DRUG, SAMPLE]).residual(y)
    g = pd.DataFrame({DRUG: df[DRUG], SAMPLE: df[SAMPLE], "m": my})
    assert g.groupby(DRUG)["m"].sum().abs().max() < 1e-7
    assert g.groupby(SAMPLE)["m"].sum().abs().max() < 1e-7


@pytest.mark.parametrize("a,b", [(2.5, -1.0), (-0.3, 4.0)])
def test_affine_invariance(case, a, b):
    """y -> a*y + b scales the interaction component by a and the additive one by a, plus b."""
    df, y = case
    p = AdditiveProjection(df, [DRUG, SAMPLE])
    hy, my = p.split(y)
    hz, mz = p.split(a * y + b)
    assert np.max(np.abs(mz - a * my)) < TOL
    assert np.max(np.abs(hz - (a * hy + b))) < TOL


@pytest.mark.parametrize("blocks", [1, 2, 3])
def test_rank_of_additive_space(blocks):
    """rank(H) = n_drug + n_sample - c, with c the connected components of the support."""
    df, _ = support(8, 10, 0.7, 7, blocks=blocks)
    d, dn = pd.factorize(df[DRUG])
    s, sn = pd.factorize(df[SAMPLE])
    g = coo_matrix((np.ones(len(df)), (d, len(dn) + s)), shape=(len(dn) + len(sn),) * 2)
    c, _ = connected_components(g + g.T, directed=False)
    assert c == blocks
    assert np.linalg.matrix_rank(design(df)) == len(dn) + len(sn) - c

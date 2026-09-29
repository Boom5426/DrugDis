"""Component-wise decomposition and evaluation of drug-response data.

The public entry point of DrugDis. Both functions take a table with one row per
observed drug-sample pair and build the orthogonal operator on exactly that
support: y = Hy + My, where Hy is the additive component (best additive fit of
drug and sample marginals to the observed pairs) and My the drug-sample
interaction component. Measured and predicted responses are decomposed by the
same operator, so the two are compared component by component.

Field names follow the canonical result tables:

    rawPCC          total-response correlation, corr(y, yhat)
    sharedPCC       additive-component correlation, corr(Hy, H yhat)
    intPCC          interaction-component correlation, corr(My, M yhat)
    A_shared        additive amplitude, sd(H yhat) / sd(Hy)
    A_int           interaction amplitude, sd(M yhat) / sd(My)
    MSE_raw         total prediction error, mean((yhat - y)^2)
    E_shared        additive-component error, mean((H(yhat - y))^2)
    E_interaction   interaction-component error, mean((M(yhat - y))^2)
                    (MSE_raw = E_shared + E_interaction exactly)
    R2_interaction  1 - |M yhat - My|^2 / |My|^2 = 2 A_int intPCC - A_int^2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from drugdis.decomposition.decompositions import AdditiveProjection

__all__ = ["decompose", "component_profile"]


def _values(frame: pd.DataFrame, v) -> np.ndarray:
    out = frame[v].to_numpy(np.float64) if isinstance(v, str) else np.asarray(v, np.float64)
    if out.shape != (len(frame),):
        raise ValueError(f"expected {len(frame)} values aligned with the rows, got {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError("responses must be finite")
    return out


def _projection(frame: pd.DataFrame, drug: str, sample: str) -> AdditiveProjection:
    if frame.duplicated([drug, sample]).any():
        raise ValueError("one row per drug-sample pair is required; average repeated "
                         "measurements first")
    return AdditiveProjection(frame, [drug, sample])


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def decompose(frame: pd.DataFrame, response="Sensitivity", drug: str = "SMILES",
              sample: str = "Sample_ID"):
    """Split a response table into its additive and interaction components.

    Returns (additive, interaction, summary): two arrays aligned with the rows of
    `frame` and a dict with the support geometry and the variance shares.
    """
    y = _values(frame, response)
    proj = _projection(frame, drug, sample)
    my = proj.residual(y)
    hy = y - my
    d, dn = pd.factorize(frame[drug], sort=False)
    s, sn = pd.factorize(frame[sample], sort=False)
    nd, ns = len(dn), len(sn)
    g = coo_matrix((np.ones(len(frame)), (d, nd + s)), shape=(nd + ns, nd + ns))
    c, _ = connected_components(g + g.T, directed=False)
    vy, vh, vm = float(y.var()), float(hy.var()), float(my.var())
    summary = {
        "n_pairs": int(len(frame)), "n_drugs": nd, "n_samples": ns,
        "components": int(c), "rank_H": nd + ns - int(c),
        "d_H_over_N": (nd + ns - int(c)) / len(frame),
        "share_additive_pct": 100 * vh / vy, "share_interaction_pct": 100 * vm / vy,
        "corr_additive_interaction": _pcc(hy, my),
    }
    return hy, my, summary


def component_profile(frame: pd.DataFrame, y="Sensitivity", yhat="prediction",
                      drug: str = "SMILES", sample: str = "Sample_ID") -> dict:
    """Score predictions component by component on the observed pairs of `frame`.

    `y` and `yhat` are column names of `frame` or arrays aligned with its rows.
    The formulas are those of `component_profile` in drugdis/models/train_2a2.py,
    which produced the reported numbers.
    """
    y, yh = _values(frame, y), _values(frame, yhat)
    proj = _projection(frame, drug, sample)
    my, myh = proj.residual(y), proj.residual(yh)
    hy, hyh = y - my, yh - myh
    e = yh - y
    me = proj.residual(e)
    he = e - me
    out = {
        "n": int(len(y)),
        "rawPCC": _pcc(y, yh), "sharedPCC": _pcc(hy, hyh), "intPCC": _pcc(my, myh),
        "A_shared": float(hyh.std() / hy.std()) if hy.std() else float("nan"),
        "A_int": float(myh.std() / my.std()) if my.std() else float("nan"),
        "MSE_raw": float(np.mean(e ** 2)),
        "E_shared": float(np.mean(he ** 2)),
        "E_interaction": float(np.mean(me ** 2)),
    }
    out["R2_interaction"] = 2 * out["A_int"] * out["intPCC"] - out["A_int"] ** 2
    return out

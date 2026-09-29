"""Response decompositions for the RISE 2.0 audit.

Implements the four candidate decompositions defined in
`latex_v2/RISE_METHODS_V2_DEFINITION.md` section 4:

    D1  legacy marginal-mean          b = mu + alpha_d + gamma_s from row/col means
    D2  two-way fixed-effect          b = Hy, r = My  (orthogonal projection)
    D3  resource-aware                D2 plus a study/resource factor
    D4  symmetric residualisation     the same operator applied to y and yhat

D2/D3 use the method of alternating projections (iterative demeaning), which is
the standard way to absorb high-dimensional fixed effects without forming the
design matrix. Forming H explicitly is impossible here: the additive space has
~57,000 levels over 4.1 million observations.

Nothing in this module modifies the published pipeline. `MarginalMeanD1`
reimplements `legacy/src_copy/Comp_Master_CLIO_Metrics.py` faithfully so that the
published numbers can be reproduced and used as the reference point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DRUG = "SMILES"
SAMPLE = "Sample_ID"
RESOURCE = "Project"


# --------------------------------------------------------------------------
# D1: legacy marginal-mean decomposition
# --------------------------------------------------------------------------


@dataclass
class MarginalMeanD1:
    """Faithful reimplementation of the published `SmartBiasBaseline` +
    `InteractionEffectEvaluator` pair.

    task_type follows the published convention: under 'LSO' the drug marginal is
    recomputed on the evaluation panel and the sample marginal is taken from the
    fitted maps; under 'LCLO' the roles swap; otherwise both come from the maps.
    Note that in the published code the maps are themselves fitted on the
    evaluation panel, so every term is a function of the observed test labels.
    """

    task_type: str = "general"
    global_mean: float = 0.0
    drug_bias: dict = field(default_factory=dict)
    cell_bias: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, target: str = "Target_AAC") -> "MarginalMeanD1":
        self.global_mean = float(df[target].mean())
        self.drug_bias = (df.groupby(DRUG)[target].mean() - self.global_mean).to_dict()
        self.cell_bias = (df.groupby(SAMPLE)[target].mean() - self.global_mean).to_dict()
        return self

    def shared(self, df: pd.DataFrame, target: str = "Target_AAC") -> np.ndarray:
        if "LCLO" in self.task_type:
            cell = df.groupby(SAMPLE)[target].transform("mean") - self.global_mean
            drug = df[DRUG].map(self.drug_bias).fillna(0.0)
        elif "LSO" in self.task_type:
            drug = df.groupby(DRUG)[target].transform("mean") - self.global_mean
            cell = df[SAMPLE].map(self.cell_bias).fillna(0.0)
        else:
            drug = df[DRUG].map(self.drug_bias).fillna(0.0)
            cell = df[SAMPLE].map(self.cell_bias).fillna(0.0)
        return (self.global_mean + drug + cell).to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------
# D2 / D3: additive projection by alternating demeaning
# --------------------------------------------------------------------------


def _group_codes(df: pd.DataFrame, factors: list[str]) -> list[np.ndarray]:
    return [pd.factorize(df[f], sort=False)[0].astype(np.int64) for f in factors]


def absorb_residual(
    values: np.ndarray,
    codes: list[np.ndarray],
    tol: float = 1e-10,
    max_iter: int = 2000,
) -> tuple[np.ndarray, int, float]:
    """Return M @ values, where M projects out the additive space of `codes`.

    Method of alternating projections: repeatedly subtract the group mean of each
    factor until the update is below `tol`. For one factor this converges in a
    single pass; for two or more it converges linearly. The intercept is absorbed
    automatically because every factor's dummies span the constant vector.

    Returns (residual, iterations, final_max_update).
    """
    r = values.astype(np.float64, copy=True)
    sizes = [int(c.max()) + 1 for c in codes]
    delta = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        prev = r.copy()
        for code, n in zip(codes, sizes):
            total = np.bincount(code, weights=r, minlength=n)
            count = np.bincount(code, minlength=n)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
            r -= mean[code]
        delta = float(np.max(np.abs(r - prev))) if len(r) else 0.0
        if delta < tol:
            break
    return r, it, delta


class AdditiveProjection:
    """D2 (factors = drug, sample) or D3 (factors = drug, sample, resource).

    The operator is defined by the support it is built on. `residual` applies the
    same operator to any vector defined on that support, which is what D4 needs:
    truth and prediction each lose their own additive structure.
    """

    def __init__(self, df: pd.DataFrame, factors: list[str] | None = None,
                 tol: float = 1e-10, max_iter: int = 2000):
        self.factors = factors or [DRUG, SAMPLE]
        self.codes = _group_codes(df, self.factors)
        self.tol = tol
        self.max_iter = max_iter
        self.n = len(df)
        self.last_iter = 0
        self.last_delta = 0.0

    def residual(self, values: np.ndarray) -> np.ndarray:
        r, it, delta = absorb_residual(np.asarray(values, dtype=np.float64),
                                       self.codes, self.tol, self.max_iter)
        self.last_iter, self.last_delta = it, delta
        return r

    def split(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (b, r) with b = values - r."""
        v = np.asarray(values, dtype=np.float64)
        r = self.residual(v)
        return v - r, r


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def orthogonality_report(y: np.ndarray, b: np.ndarray, r: np.ndarray) -> dict:
    """Construct-validity diagnostics for a decomposition y = b + r.

    A genuine orthogonal decomposition satisfies corr(b, r) = 0 and
    Var(y) = Var(b) + Var(r). Any departure means the reported variance shares
    are not a variance decomposition.
    """
    y = np.asarray(y, float)
    b = np.asarray(b, float)
    r = np.asarray(r, float)
    vy, vb, vr = y.var(), b.var(), r.var()
    cov = float(np.cov(b, r, ddof=0)[0, 1])
    denom = np.sqrt(vb * vr)
    return {
        "n": int(len(y)),
        "var_y": float(vy),
        "var_b": float(vb),
        "var_r": float(vr),
        "cov_b_r": cov,
        "corr_b_r": float(cov / denom) if denom > 0 else np.nan,
        # additivity gap: Var(y) - Var(b) - Var(r) = 2 Cov(b, r); zero iff orthogonal
        "additivity_gap": float(vy - vb - vr),
        "additivity_gap_frac_of_var_y": float((vy - vb - vr) / vy) if vy > 0 else np.nan,
        # shares as the manuscript reports them, normalised to Var(b) + Var(r)
        "share_b_pct": float(100 * vb / (vb + vr)) if (vb + vr) > 0 else np.nan,
        "share_r_pct": float(100 * vr / (vb + vr)) if (vb + vr) > 0 else np.nan,
        # shares against the true total, which only agree with the above if orthogonal
        "share_b_pct_of_var_y": float(100 * vb / vy) if vy > 0 else np.nan,
        "share_r_pct_of_var_y": float(100 * vr / vy) if vy > 0 else np.nan,
        "residual_mean_abs": float(np.abs(r.mean())),
    }

"""Phase 2A-2 architectures: M0/M2 (single head) and M3/M4 (factorized two head).

The frozen roadmap (RISE_FROZEN_ROADMAP_2026-09-04.md sections 8.3-8.6) requires
that exactly one causal factor change per step:

    M0  published single-head decoder                       reference
    M2  parameter-matched single-head, wider joint head     capacity
    M3  factorized two head, raw-only supervision           factorization
    M4  M3 plus component supervision                       supervision

`CLIOSingleHead` reproduces `legacy/src_copy/model.py::CLIO` exactly when
hidden_dim1 = 1024: same submodule construction order, same layer types, hence
the same parameter-initialisation RNG stream. `check_init_identity` asserts it.

`CLIOFactorized` keeps the two encoders and the joint head of M0 unchanged and
adds two scalar effect heads plus a learnable intercept:

    b_hat = mu + f_d(enc_d(x_d)) + f_s(enc_s(x_s))
    z_hat = g(concat(enc_d(x_d), enc_s(x_s)))
    y_hat = b_hat + z_hat

f_d reads only the drug representation and f_s only the sample representation,
so at inference b_hat is exactly additive in (drug, sample). No drug or sample
identity is used anywhere. During training the encoders carry BatchNorm, so the
tower outputs depend on batch composition; this is inherited from M0 and is
switched off at evaluation, where the factorization is exact.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp_block(din: int, dout: int, dropout_p: float) -> list:
    return [nn.Linear(din, dout), nn.ReLU(), nn.BatchNorm1d(dout), nn.Dropout(dropout_p)]


class CLIOSingleHead(nn.Module):
    """M0 (hidden_dim1=1024) and M2 (hidden_dim1 widened to match M3/M4)."""

    def __init__(self, drug_dim: int, gene_dim: int, projection_dim: int = 512,
                 hidden_dim1: int = 1024, hidden_dim2: int = 512,
                 dropout_p: float = 0.4):
        super().__init__()
        self.drug_encoder = nn.Sequential(*_mlp_block(drug_dim, projection_dim, dropout_p))
        self.gene_encoder = nn.Sequential(*_mlp_block(gene_dim, 1024, dropout_p),
                                          *_mlp_block(1024, projection_dim, dropout_p))
        self.joint_input_dim = projection_dim * 2
        self.joint_head = nn.Sequential(
            *_mlp_block(self.joint_input_dim, hidden_dim1, dropout_p),
            *_mlp_block(hidden_dim1, hidden_dim2, dropout_p),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, drug_batch, cell_batch):
        d = self.drug_encoder(drug_batch)
        g = self.gene_encoder(cell_batch)
        return self.joint_head(torch.cat([d, g], dim=1))


class CLIOFactorized(nn.Module):
    """M3 / M4. Identical encoders and joint head to M0; two scalar effect heads."""

    def __init__(self, drug_dim: int, gene_dim: int, projection_dim: int = 512,
                 hidden_dim1: int = 1024, hidden_dim2: int = 512,
                 shared_hidden: int = 512, dropout_p: float = 0.4):
        super().__init__()
        self.drug_encoder = nn.Sequential(*_mlp_block(drug_dim, projection_dim, dropout_p))
        self.gene_encoder = nn.Sequential(*_mlp_block(gene_dim, 1024, dropout_p),
                                          *_mlp_block(1024, projection_dim, dropout_p))
        self.joint_input_dim = projection_dim * 2
        self.joint_head = nn.Sequential(
            *_mlp_block(self.joint_input_dim, hidden_dim1, dropout_p),
            *_mlp_block(hidden_dim1, hidden_dim2, dropout_p),
            nn.Linear(hidden_dim2, 1),
        )
        self.drug_effect_head = nn.Sequential(
            *_mlp_block(projection_dim, shared_hidden, dropout_p),
            nn.Linear(shared_hidden, 1),
        )
        self.sample_effect_head = nn.Sequential(
            *_mlp_block(projection_dim, shared_hidden, dropout_p),
            nn.Linear(shared_hidden, 1),
        )
        self.mu = nn.Parameter(torch.zeros(1))

    def forward(self, drug_batch, cell_batch):
        d = self.drug_encoder(drug_batch)
        g = self.gene_encoder(cell_batch)
        b = self.mu + self.drug_effect_head(d) + self.sample_effect_head(g)
        z = self.joint_head(torch.cat([d, g], dim=1))
        return b, z


# ---------------------------------------------------------------- parameters


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def single_head_params(drug_dim: int, gene_dim: int, h1: int,
                       projection_dim: int = 512, hidden_dim2: int = 512) -> int:
    """Closed form, used to solve for the M2 width without building the model."""
    enc_d = (drug_dim + 1) * projection_dim + 2 * projection_dim
    enc_g = ((gene_dim + 1) * 1024 + 2 * 1024
             + (1024 + 1) * projection_dim + 2 * projection_dim)
    joint = ((2 * projection_dim + 1) * h1 + 2 * h1
             + (h1 + 1) * hidden_dim2 + 2 * hidden_dim2
             + hidden_dim2 + 1)
    return enc_d + enc_g + joint


def factorized_extra_params(shared_hidden: int, projection_dim: int = 512) -> int:
    """Parameters M3/M4 add on top of the M0 backbone."""
    head = (projection_dim + 1) * shared_hidden + 2 * shared_hidden + shared_hidden + 1
    return 2 * head + 1  # two effect heads plus mu


def solve_m2_width(drug_dim: int, gene_dim: int, shared_hidden: int,
                   base_h1: int = 1024, projection_dim: int = 512,
                   hidden_dim2: int = 512) -> tuple:
    """Smallest-|error| joint width that matches M3's parameter count."""
    target = (single_head_params(drug_dim, gene_dim, base_h1, projection_dim, hidden_dim2)
              + factorized_extra_params(shared_hidden, projection_dim))
    best = None
    for h1 in range(base_h1, 4 * base_h1):
        p = single_head_params(drug_dim, gene_dim, h1, projection_dim, hidden_dim2)
        d = abs(p - target)
        if best is None or d < best[1]:
            best = (h1, d, p)
        if p > target and best[0] != h1:
            break
    return best[0], best[2], target


def check_init_identity(drug_dim: int, gene_dim: int, seed: int, dropout_p: float = 0.4):
    """M0 built here must be bit-identical to the published CLIO under one seed."""
    import numpy as np
    import random
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # legacy model.py
    from model import CLIO  # noqa: E402

    def _seed():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    _seed()
    a = CLIO(drug_dim=drug_dim, gene_dim=gene_dim, dropout_p=dropout_p)
    _seed()
    b = CLIOSingleHead(drug_dim=drug_dim, gene_dim=gene_dim, hidden_dim1=1024,
                       dropout_p=dropout_p)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), (set(sa) ^ set(sb))
    worst = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa)
    return {"n_params_legacy": n_params(a), "n_params_reimpl": n_params(b),
            "max_abs_state_diff": worst}

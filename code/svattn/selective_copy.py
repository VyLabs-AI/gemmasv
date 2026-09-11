"""Selective retrieval under redundant filler (selective-copying-style probe).

Built on the MSE / nearest-value retrieval recipe that trains cleanly. Each
instance: n_data informative (content, value) tokens with distinct random
contents and random value vectors, plus n_filler near-duplicate "blank" tokens
whose value is zero (pure distractor mass). A probe equals one data content
(plus noise); the model must reproduce that token's value, ignoring the filler.
Retrieval is scored as nearest among the n_data data values.

Layers (matched projections, trained by MSE):
  - SV-Attention (ours): RBF readout gated by one-class support coefficients
  - RBF attention (uniform alpha): same readout, no gate  [ablation control]
  - softmax attention (dot-product, full context)
  - linear attention (fixed-size feature-map state)

Question (go/no-go): as filler grows, does the SV gate stay robust (it sends
filler to the certified-inert reserve set) while the ungated/fixed-state
baselines are diluted by filler?
"""
from __future__ import annotations

import numpy as np
import torch

from .sv_attention import SVAttention, rbf_gram


def make_instance(n_data, n_filler, d_in, dv_in, rng, blank_noise=0.12):
    contents = rng.randn(n_data, d_in)
    values = rng.randn(n_data, dv_in)
    blank = rng.randn(1, d_in) * 0.3
    filler_c = blank + blank_noise * rng.randn(n_filler, d_in)
    filler_v = np.zeros((n_filler, dv_in))
    X = np.vstack([contents, filler_c])
    Vfull = np.vstack([values, filler_v])
    j = rng.randint(n_data)
    q = contents[j:j + 1] + 0.05 * rng.randn(1, d_in)
    perm = rng.permutation(len(X))
    return (torch.tensor(X[perm], dtype=torch.float64),
            torch.tensor(Vfull[perm], dtype=torch.float64),
            torch.tensor(q, dtype=torch.float64),
            torch.tensor(values, dtype=torch.float64),          # data-value candidates
            torch.tensor(values[j:j + 1], dtype=torch.float64), # target
            j)


class _Proj(torch.nn.Module):
    def __init__(self, d_in, dv_in, d, dv):
        super().__init__()
        self.Wk = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.Wv = torch.nn.Parameter(torch.randn(dv_in, dv, dtype=torch.float64) * 0.5)
        self.Wq = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.Wo = torch.nn.Parameter(torch.randn(dv, dv_in, dtype=torch.float64) * 0.5)


class SVAttnModel(_Proj):
    def __init__(self, d_in, dv_in, d=8, dv=4, C=0.25, kpar=1.5):
        super().__init__(d_in, dv_in, d, dv)
        self.attn = SVAttention(C=C, kpar=kpar, normalize=True)

    def forward(self, X, V, q):
        return self.attn(X @ self.Wk, V @ self.Wv, q @ self.Wq) @ self.Wo


class RBFUniformModel(_Proj):
    def __init__(self, d_in, dv_in, d=8, dv=4, kpar=1.5):
        super().__init__(d_in, dv_in, d, dv)
        self.kpar = kpar

    def forward(self, X, V, q):
        w = rbf_gram(q @ self.Wq, X @ self.Wk, self.kpar)
        O = (w @ (V @ self.Wv)) / w.sum(1, keepdim=True).clamp_min(1e-8)
        return O @ self.Wo


class SoftmaxModel(_Proj):
    def __init__(self, d_in, dv_in, d=8, dv=4, scale=3.0):
        super().__init__(d_in, dv_in, d, dv)
        self.log_scale = torch.nn.Parameter(torch.tensor(float(np.log(scale)), dtype=torch.float64))

    def forward(self, X, V, q):
        K, Vv, Q = X @ self.Wk, V @ self.Wv, q @ self.Wq
        w = torch.softmax((Q @ K.T) * torch.exp(self.log_scale), dim=1)
        return (w @ Vv) @ self.Wo


class LinearAttnModel(_Proj):
    def __init__(self, d_in, dv_in, d=8, dv=4):
        super().__init__(d_in, dv_in, d, dv)

    @staticmethod
    def phi(x):
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, X, V, q):
        K, Vv, Q = X @ self.Wk, V @ self.Wv, q @ self.Wq
        pk, pq = self.phi(K), self.phi(Q)
        S = pk.T @ Vv
        den = (pq @ pk.sum(0, keepdim=True).T).clamp_min(1e-8)
        return ((pq @ S) / den) @ self.Wo


class DeltaNetModel(_Proj):
    """Delta-rule linear attention (DeltaNet, Yang et al. 2024).

    Sequential recurrence over the context tokens (in arrival order):
        S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T,
    with L2-normalized keys/queries and a learned per-token write gate beta_t.
    The state S is a fixed d_v x d matrix; readout o = S q. This is the strong,
    recall-capable efficient baseline (the delta rule can overwrite stale
    associations, unlike plain linear attention)."""

    def __init__(self, d_in, dv_in, d=8, dv=4):
        super().__init__(d_in, dv_in, d, dv)
        self.beta = torch.nn.Linear(d_in, 1).double()

    @staticmethod
    def _l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, X, V, q):
        K = self._l2(X @ self.Wk)
        Vv = V @ self.Wv
        b = torch.sigmoid(self.beta(X)).squeeze(-1)          # (L,) write gates
        d, dv = K.shape[1], Vv.shape[1]
        S = torch.zeros(dv, d, dtype=K.dtype)
        for t in range(K.shape[0]):
            kt, vt = K[t], Vv[t]
            pred = S @ kt                                     # current value for kt
            S = S + b[t] * torch.outer(vt - pred, kt)
        Q = self._l2(q @ self.Wq)
        return (Q @ S.T) @ self.Wo


def train(model, n_data, n_filler, d_in, dv_in, steps=300, seqs=3, lr=0.03, seed=0):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for _ in range(seqs):
            X, V, q, _cand, target, _j = make_instance(n_data, n_filler, d_in, dv_in, rng)
            loss = loss + ((model(X, V, q) - target) ** 2).mean()
        (loss / seqs).backward()
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, n_data, n_filler, d_in, dv_in, n_eval=300, seed=777):
    rng = np.random.RandomState(seed)
    correct = 0
    for _ in range(n_eval):
        X, V, q, cand, target, j = make_instance(n_data, n_filler, d_in, dv_in, rng)
        O = model(X, V, q)
        pred = int(torch.argmin(((cand - O) ** 2).sum(1)))
        correct += (pred == j)
    return correct / n_eval


@torch.no_grad()
def sv_state_size(model, n_data, n_filler, d_in, dv_in, n=50, seed=222):
    rng = np.random.RandomState(seed)
    sizes = []
    for _ in range(n):
        X, V, q, _c, _t, _j = make_instance(n_data, n_filler, d_in, dv_in, rng)
        sizes.append(int(model.attn.support_mask(X @ model.Wk).sum()))
    return float(np.mean(sizes))

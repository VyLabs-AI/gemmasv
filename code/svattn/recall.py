"""Associative recall with Support Vector Attention.

The canonical probe for test-time-learning / attention layers: store a set of
(key, value) pairs in context, then retrieve the value for a probe key. We train
a thin SV-Attention model (learnable key/value/query projections + the
max-margin readout) and compare against a matched single-head softmax-attention
readout. The point is that SV-Attention learns content-based retrieval *and*
attends through a sparse, certified support set.
"""
from __future__ import annotations

import numpy as np
import torch

from .sv_attention import SVAttention, rbf_gram


def make_recall_batch(n_pairs, d_in, dv_in, rng):
    """One recall instance: n_pairs distinct keys, random values, probe = one key."""
    Kraw = torch.tensor(rng.randn(n_pairs, d_in), dtype=torch.float64)
    Vraw = torch.tensor(rng.randn(n_pairs, dv_in), dtype=torch.float64)
    j = rng.randint(n_pairs)
    q = Kraw[j:j + 1] + 0.05 * torch.tensor(rng.randn(1, d_in), dtype=torch.float64)
    target = Vraw[j:j + 1]
    return Kraw, Vraw, q, target, j


class SVAttnRecall(torch.nn.Module):
    def __init__(self, d_in, dv_in, d=4, dv=3, C=0.3, kpar=1.5):
        super().__init__()
        self.Wk = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.Wv = torch.nn.Parameter(torch.randn(dv_in, dv, dtype=torch.float64) * 0.5)
        self.Wq = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.attn = SVAttention(C=C, kpar=kpar, normalize=True)

    def forward(self, Kraw, Vraw, q):
        X = Kraw @ self.Wk
        V = Vraw @ self.Wv
        Q = q @ self.Wq
        return self.attn(X, V, Q)


class SoftmaxAttnRecall(torch.nn.Module):
    def __init__(self, d_in, dv_in, d=4, dv=3, scale=5.0):
        super().__init__()
        self.Wk = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.Wv = torch.nn.Parameter(torch.randn(dv_in, dv, dtype=torch.float64) * 0.5)
        self.Wq = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.log_scale = torch.nn.Parameter(torch.tensor(float(np.log(scale)), dtype=torch.float64))

    def forward(self, Kraw, Vraw, q):
        X = Kraw @ self.Wk
        V = Vraw @ self.Wv
        Q = q @ self.Wq
        logits = (Q @ X.T) * torch.exp(self.log_scale)
        w = torch.softmax(logits, dim=1)
        return w @ V


class RBFAttnRecall(torch.nn.Module):
    """Control: same RBF readout as SV-Attention but with UNIFORM weights (no
    max-margin gate). Isolates whether the SVDD coefficients add anything over a
    plain distance-kernel attention."""

    def __init__(self, d_in, dv_in, d=4, dv=3, kpar=1.5):
        super().__init__()
        self.Wk = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.Wv = torch.nn.Parameter(torch.randn(dv_in, dv, dtype=torch.float64) * 0.5)
        self.Wq = torch.nn.Parameter(torch.randn(d_in, d, dtype=torch.float64) * 0.5)
        self.kpar = kpar

    def forward(self, Kraw, Vraw, q):
        X, V, Q = Kraw @ self.Wk, Vraw @ self.Wv, q @ self.Wq
        w = rbf_gram(Q, X, self.kpar)
        return (w @ V) / w.sum(dim=1, keepdim=True).clamp_min(1e-8)


def train_recall(model, n_pairs=10, d_in=5, dv_in=3, steps=300, seqs_per_step=4,
                 lr=0.03, seed=0):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for _ in range(seqs_per_step):
            Kraw, Vraw, q, target, _ = make_recall_batch(n_pairs, d_in, dv_in, rng)
            O = model(Kraw, Vraw, q)
            loss = loss + ((O - target) ** 2).mean()
        loss = loss / seqs_per_step
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def eval_recall(model, n_pairs=10, d_in=5, dv_in=3, n_eval=200, seed=999):
    """Top-1 recall accuracy: is the predicted value nearest the true value?"""
    rng = np.random.RandomState(seed)
    correct = 0
    mse = 0.0
    for _ in range(n_eval):
        Kraw, Vraw, q, target, j = make_recall_batch(n_pairs, d_in, dv_in, rng)
        O = model(Kraw, Vraw, q)
        V = Vraw @ model.Wv
        pred_idx = int(torch.argmin(((V - O) ** 2).sum(dim=1)))
        correct += (pred_idx == j)
        mse += ((O - target) ** 2).mean().item()
    return correct / n_eval, mse / n_eval

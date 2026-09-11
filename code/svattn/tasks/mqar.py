"""Continuous multi-query associative recall (MQAR), causal.

Sequence = n_mem memory tokens [key; value] followed by n_query query tokens
[key; 0]. At each query position the model must output the value bound to that
query's key earlier in the sequence. Scored by nearest-value retrieval among the
n_mem distinct values. This is the sequential analogue of
selective_copy.make_instance and the standard probe for the recall capacity of
efficient-attention layers.

Memory tokens come first so that, for the chunk-frozen CausalSVAttention layer,
every query's causal prefix contains all the memory tokens (keep n_mem a multiple
of the chunk size so queries begin on a chunk boundary).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def make_mqar_batch(batch, n_mem, n_query, d_k, d_v, rng, key_noise=0.02,
                    dtype=torch.float32):
    d_in = d_k + d_v
    X = np.zeros((batch, n_mem + n_query, d_in))
    tgt = np.zeros((batch, n_query, d_v))
    vbank = np.zeros((batch, n_mem, d_v))
    qidx = np.zeros((batch, n_query), dtype=np.int64)
    for b in range(batch):
        keys = rng.randn(n_mem, d_k)
        vals = rng.randn(n_mem, d_v)
        X[b, :n_mem, :d_k] = keys
        X[b, :n_mem, d_k:] = vals
        vbank[b] = vals
        idx = rng.randint(0, n_mem, size=n_query)
        X[b, n_mem:, :d_k] = keys[idx] + key_noise * rng.randn(n_query, d_k)
        tgt[b] = vals[idx]
        qidx[b] = idx
    return (torch.tensor(X, dtype=dtype), torch.tensor(tgt, dtype=dtype),
            torch.tensor(vbank, dtype=dtype), torch.tensor(qidx))


class SeqRecallModel(nn.Module):
    """Embed -> causal sequence layer -> readout, supervised at query positions."""

    def __init__(self, layer: nn.Module, d_in: int, d_model: int, d_v: int):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model, bias=False)
        self.layer = layer
        self.readout = nn.Linear(d_model, d_v, bias=False)

    def forward(self, X):
        return self.readout(self.layer(self.embed(X)))     # (B, T, d_v)


def train_recall(model, n_mem, n_query, d_k, d_v, steps=300, batch=16, lr=3e-3,
                 seed=0):
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    last = float("nan")
    for _ in range(steps):
        X, tgt, _, _ = make_mqar_batch(batch, n_mem, n_query, d_k, d_v, rng)
        opt.zero_grad()
        out = model(X)[:, n_mem:, :]
        loss = ((out - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


@torch.no_grad()
def eval_recall(model, n_mem, n_query, d_k, d_v, n_eval=500, batch=50, seed=777):
    rng = np.random.RandomState(seed)
    model.eval()
    correct = total = done = 0
    while done < n_eval:
        b = min(batch, n_eval - done)
        X, _tgt, vbank, qidx = make_mqar_batch(b, n_mem, n_query, d_k, d_v, rng)
        out = model(X)[:, n_mem:, :]                       # (b, n_query, d_v)
        d2 = ((out.unsqueeze(2) - vbank.unsqueeze(1)) ** 2).sum(-1)  # (b, q, n_mem)
        pred = d2.argmin(-1)
        correct += int((pred == qidx).sum())
        total += b * n_query
        done += b
    return correct / total

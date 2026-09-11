"""Causal sequence-layer baselines with the CausalSVAttention interface.

Each layer maps (B, T, d_model) -> (B, T, d_model), is causal (position t sees
only positions <= t), and is multi-head, so they are drop-in comparables for
`CausalSVAttention` in the MQAR recall task and the throughput spike.

- CausalSoftmaxAttention: standard masked scaled-dot-product attention (the fast,
  fully vectorized reference for throughput and the strong recall baseline).
- CausalLinearAttention: feature-map (elu+1) linear attention with a causal
  cumulative state (the efficient-but-fixed-state baseline).
- DeltaNet: delta-rule linear attention (Yang et al. 2024), a sequential
  recurrence that can overwrite stale associations -- the strong efficient
  recall baseline. Naive token loop (not the chunk-parallel kernel).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _MHProj(nn.Module):
    def __init__(self, d_model: int, n_heads: int, head_dim: int | None):
        super().__init__()
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim or d_model // n_heads)
        hd = self.head_dim * self.n_heads
        self.Wk = nn.Linear(d_model, hd, bias=False)
        self.Wq = nn.Linear(d_model, hd, bias=False)
        self.Wv = nn.Linear(d_model, hd, bias=False)
        self.Wo = nn.Linear(hd, d_model, bias=False)

    def _project(self, X):
        B, T, _ = X.shape
        H, hd = self.n_heads, self.head_dim
        q = self.Wq(X).view(B, T, H, hd).transpose(1, 2)   # (B, H, T, hd)
        k = self.Wk(X).view(B, T, H, hd).transpose(1, 2)
        v = self.Wv(X).view(B, T, H, hd).transpose(1, 2)
        return q, k, v

    def _merge(self, o):                                    # (B, H, T, hd) -> (B,T,dm)
        B, H, T, hd = o.shape
        return self.Wo(o.transpose(1, 2).reshape(B, T, H * hd))


class CausalSoftmaxAttention(_MHProj):
    def forward(self, X):
        q, k, v = self._project(X)
        T = X.shape[1]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=X.device))
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        return self._merge(att @ v)


class CausalSlidingWindowAttention(_MHProj):
    """Transformer++ baseline: masked softmax attention restricted to a sliding
    window of the last `window` keys (query t attends to [t-window+1, t]). The
    window is the model's fixed 'state' -- set it to SV's mean retained-set size
    for a matched-state comparison (Phase 2)."""

    def __init__(self, d_model, n_heads=1, head_dim=None, window: int = 64):
        super().__init__(d_model, n_heads, head_dim)
        self.window = int(window)

    def forward(self, X):
        q, k, v = self._project(X)
        T = X.shape[1]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        i = torch.arange(T, device=X.device)
        causal = i[:, None] >= i[None, :]
        within = (i[:, None] - i[None, :]) < self.window
        mask = causal & within
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        return self._merge(att @ v)


class CausalLinearAttention(_MHProj):
    @staticmethod
    def _phi(x):
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, X):
        q, k, v = self._project(X)
        pq, pk = self._phi(q), self._phi(k)               # (B, H, T, hd)
        kv = pk.unsqueeze(-1) * v.unsqueeze(-2)           # (B, H, T, hd, dv)
        S = kv.cumsum(dim=2)                              # causal prefix sums
        z = pk.cumsum(dim=2)                             # (B, H, T, hd)
        num = torch.einsum("bhtk,bhtkd->bhtd", pq, S)     # (B, H, T, dv)
        den = torch.einsum("bhtk,bhtk->bht", pq, z).unsqueeze(-1).clamp_min(1e-6)
        return self._merge(num / den)


class DeltaNet(_MHProj):
    def __init__(self, d_model, n_heads=1, head_dim=None):
        super().__init__(d_model, n_heads, head_dim)
        self.beta = nn.Linear(d_model, self.n_heads, bias=True)

    @staticmethod
    def _l2(x):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, X):
        B, T, _ = X.shape
        H, hd = self.n_heads, self.head_dim
        q, k, v = self._project(X)
        q, k = self._l2(q), self._l2(k)
        beta = torch.sigmoid(self.beta(X)).transpose(1, 2)   # (B, H, T)
        out = torch.zeros(B, H, T, hd, dtype=X.dtype, device=X.device)
        S = torch.zeros(B, H, hd, hd, dtype=X.dtype, device=X.device)  # (dv, dk) per head
        for t in range(T):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]   # (B, H, hd)
            pred = torch.einsum("bhdk,bhk->bhd", S, kt)
            S = S + beta[:, :, t, None, None] * torch.einsum("bhd,bhk->bhdk", vt - pred, kt)
            out[:, :, t] = torch.einsum("bhdk,bhk->bhd", S, qt)
        return self._merge(out)


LAYERS = {
    "softmax": CausalSoftmaxAttention,
    "swa": CausalSlidingWindowAttention,
    "linear": CausalLinearAttention,
    "deltanet": DeltaNet,
}

"""A tiny self-contained char/token GPT (standard softmax) + a synthetic recall
corpus, for the inference-time KV-eviction study.

The model is a normal causal softmax transformer -- trained at full speed, with
NO SV gate in the model. The SV gate is applied only at inference as a KV-cache
eviction policy (see experiments/p3_kv_eviction.py).

Recall corpus: each sequence presents n_pairs (key, value) token bindings, then a
query phase that repeats keys (target = the bound value). Query frequency is
skewed -- a few "dense" keys are queried often, the rest are "rare" (queried
once) -- so a budget-limited cache that keeps high-traffic tokens (H2O) drops the
rare keys and fails their queries, while an outlier-aware gate keeps them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .baselines import CausalSoftmaxAttention

SEP = 0


def make_recall_batch(batch, n_pairs, n_query, vocab, rng, dense_frac=0.25):
    T = 2 * n_pairs + 1 + 2 * n_query
    idx = np.zeros((batch, T), dtype=np.int64)
    ans = np.zeros((batch, T), dtype=bool)
    rare = np.zeros((batch, T), dtype=bool)
    n_dense = max(1, int(round(dense_frac * n_pairs)))
    for b in range(batch):
        toks = rng.permutation(np.arange(1, vocab))
        keys, vals = toks[:n_pairs], toks[n_pairs:2 * n_pairs]
        kv = dict(zip(keys.tolist(), vals.tolist()))
        dense = set(keys[:n_dense].tolist())
        p = 0
        for i in range(n_pairs):
            idx[b, p], idx[b, p + 1] = keys[i], vals[i]
            p += 2
        idx[b, p] = SEP
        p += 1
        for _ in range(n_query):
            qk = (rng.choice(keys[:n_dense]) if rng.rand() < 0.6 else rng.choice(keys))
            idx[b, p], idx[b, p + 1] = qk, kv[qk]
            ans[b, p + 1] = True
            rare[b, p + 1] = qk not in dense
            p += 2
    return torch.tensor(idx), torch.tensor(ans), torch.tensor(rare)


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSoftmaxAttention(d_model, n_heads, d_model // n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=2, n_layers=2, max_T=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_T, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.lnf(x))


def train_lm(model, cfg, steps=2000, batch=64, lr=3e-3, seed=0):
    rng = np.random.RandomState(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = float("nan")
    for _ in range(steps):
        idx, _, _ = make_recall_batch(batch, **cfg, rng=rng)
        logits = model(idx)[:, :-1]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return last


@torch.no_grad()
def eval_lm(model, cfg, n_eval=20, batch=64, seed=999):
    rng = np.random.RandomState(seed)
    model.eval()
    nll = ans_ok = ans_n = 0.0
    for _ in range(n_eval):
        idx, ans, _ = make_recall_batch(batch, **cfg, rng=rng)
        logits = model(idx)
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = idx[:, 1:]
        nll += -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
        pred = logits[:, :-1].argmax(-1)
        a = ans[:, 1:]
        ans_ok += ((pred == tgt) & a).sum().item()
        ans_n += a.sum().item()
    return float(np.exp(nll / n_eval)), ans_ok / max(ans_n, 1)

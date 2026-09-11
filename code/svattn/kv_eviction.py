"""Inference-time KV-cache eviction for a trained (softmax) TinyGPT.

The model is unchanged (full-speed softmax). At inference we cap each attention
layer/head's KV cache at a budget B: every `chunk` positions, the prefix keys are
pruned to B by a policy, and the chunk's queries may attend only to the kept
prefix keys (plus their own causal within-chunk keys). Policies:

  sv      : keep the one-class SVM support set over the prefix keys (outlier-aware,
            query-independent); if support > B, keep the top-B by |alpha|.
  h2o     : keep top-B prefix keys by accumulated (full-causal) attention mass.
  recency : keep the most recent B.   random : keep a random B.   full : no budget.

The certificate is on the SVM's own readout / exact forgetting, NOT the softmax
output -- here `sv` is a principled, training-free eviction heuristic. The
question is whether keeping the boundary (atypical, incl. rare-key) tokens beats
heavy-hitter mass on rare-key answer accuracy at a matched budget.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .fast_diff_svdd import fast_svdd_state


def _policy_keep(keys, mass, budget, policy, kpar, nu, rng):
    pre = len(keys)
    if pre <= budget:
        return np.arange(pre)
    if policy == "sv":
        try:
            s = fast_svdd_state(keys, C=1.0 / (nu * pre), kpar=kpar)
            support = np.array(sorted(s.S + s.E), dtype=int)
            if len(support) == 0:
                return np.argsort(-mass)[:budget]
            if len(support) <= budget:
                return support
            alpha = np.asarray(s.alpha)
            return np.sort(support[np.argsort(-alpha[support])[:budget]])
        except Exception:
            return np.arange(pre - budget, pre)
    if policy == "h2o":
        return np.sort(np.argsort(-mass)[:budget])
    if policy == "recency":
        return np.arange(pre - budget, pre)
    return np.sort(rng.choice(pre, budget, replace=False))


def _keep_mask(q, k, budget, policy, chunk, kpar, nu, rng):
    B, H, T, hd = k.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    att = ((q @ k.transpose(-2, -1)) / math.sqrt(hd)).masked_fill(~causal, float("-inf"))
    mass = att.softmax(-1).sum(2).cpu().numpy()                   # (B, H, T) accumulated mass
    keep = causal.expand(B, H, T, T).clone()
    if policy == "full":
        return keep
    kk = k.detach().cpu().numpy()
    for b in range(B):
        for h in range(H):
            for c in range(0, T, chunk):
                pre = c
                if pre <= budget:
                    continue
                sel = _policy_keep(kk[b, h, :pre], mass[b, h, :pre], budget, policy, kpar, nu, rng)
                drop = np.setdiff1d(np.arange(pre), sel)
                if len(drop):
                    keep[b, h, c:min(c + chunk, T), drop] = False
    return keep


@torch.no_grad()
def evict_logits(model, idx, budget, policy, chunk=6, kpar=2.0, nu=0.3, seed=0):
    rng = np.random.RandomState(seed)
    B, T = idx.shape
    x = model.tok(idx) + model.pos(torch.arange(T, device=idx.device))
    for blk in model.blocks:
        h = blk.ln1(x)
        attn = blk.attn
        H, hd = attn.n_heads, attn.head_dim
        q = attn.Wq(h).view(B, T, H, hd).transpose(1, 2)
        k = attn.Wk(h).view(B, T, H, hd).transpose(1, 2)
        v = attn.Wv(h).view(B, T, H, hd).transpose(1, 2)
        keep = _keep_mask(q, k, budget, policy, chunk, kpar, nu, rng)
        att = ((q @ k.transpose(-2, -1)) / math.sqrt(hd)).masked_fill(~keep, float("-inf")).softmax(-1)
        o = attn.Wo((att @ v).transpose(1, 2).reshape(B, T, H * hd))
        x = x + o
        x = x + blk.mlp(blk.ln2(x))
    return model.head(model.lnf(x))


@torch.no_grad()
def eval_eviction(model, cfg, budget, policy, chunk=6, kpar=2.0, nu=0.3,
                  n_eval=8, batch=32, seed=999):
    from .tinygpt import make_recall_batch
    rng = np.random.RandomState(seed)
    model.eval()
    ok = n = ok_r = n_r = 0.0
    nll = 0.0
    for _ in range(n_eval):
        idx, ans, rare = make_recall_batch(batch, **cfg, rng=rng)
        logits = evict_logits(model, idx, budget, policy, chunk, kpar, nu, seed=0)
        tgt = idx[:, 1:]
        pred = logits[:, :-1].argmax(-1)
        a, r = ans[:, 1:], rare[:, 1:]
        ok += ((pred == tgt) & a).sum().item(); n += a.sum().item()
        ok_r += ((pred == tgt) & r).sum().item(); n_r += r.sum().item()
        lp = torch.log_softmax(logits[:, :-1], -1)
        nll += -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
    return {"ppl": float(np.exp(nll / n_eval)),
            "ans_acc": ok / max(n, 1),
            "rare_acc": ok_r / max(n_r, 1)}

"""Pluggable nanoGPT decoder (pre-LN) + attention factory.

Every attention variant exposes the same (B, T, d_model) -> (B, T, d_model)
causal interface (see svattn/baselines.py and svattn/causal_sv_attention.py), so
the backbone is identical across variants and only the attention slot changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from svattn.baselines import (CausalSoftmaxAttention, CausalSlidingWindowAttention,
                              CausalLinearAttention, DeltaNet)
from svattn.causal_sv_attention import CausalSVAttention


@dataclass
class GPTConfig:
    vocab: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    block: int = 128
    mlp_mult: int = 4
    dropout: float = 0.0


def build_attn(name: str, cfg: GPTConfig, *, window: int = 64, **sv_kwargs):
    """Attention factory. `name` in {softmax, swa, linear, deltanet, sv, sv-nogate}.
    `window` sets the Transformer++ (swa) state for matched-state comparison;
    sv_kwargs (C, kpar, chunk, persistent, exact_sequential, gate) tune the SV
    variants and are ignored by the baselines."""
    hd = cfg.d_model // cfg.n_heads
    if name == "softmax":
        return CausalSoftmaxAttention(cfg.d_model, cfg.n_heads, hd)
    if name == "swa":
        return CausalSlidingWindowAttention(cfg.d_model, cfg.n_heads, hd, window=window)
    if name == "linear":
        return CausalLinearAttention(cfg.d_model, cfg.n_heads, hd)
    if name == "deltanet":
        return DeltaNet(cfg.d_model, cfg.n_heads, hd)
    if name in ("sv", "sv-nogate"):
        kw = dict(C=0.25, kpar=2.0, chunk=64, persistent=True)
        kw.update(sv_kwargs)
        if name == "sv-nogate":
            kw["gate"] = False
        return CausalSVAttention(cfg.d_model, cfg.n_heads, hd, **kw)
    raise ValueError(f"unknown attention '{name}'")


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, attn: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = attn
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model), nn.GELU(),
            nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, cfg: GPTConfig, attn_name: str = "softmax", **sv_kwargs):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.block, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg, build_attn(attn_name, cfg, **sv_kwargs))
             for _ in range(cfg.n_layers)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.apply(self._init_weights)
        self.head.weight = self.tok.weight                  # weight tying (after init)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.lnf(x))
        if targets is None:
            return logits, None
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self) -> int:
        # exclude tied head (shares tok embedding)
        return sum(p.numel() for p in self.parameters()) - self.head.weight.numel()

"""enwik8 char-level data: byte-level ids, deterministic train/val slices.

enwik8 is the first 1e8 bytes of a Wikipedia dump (Hutter prize); the standard
char-LM benchmark with NO tokenizer. We read a prefix slice (cheap for the tiny
Phase-1 / Phase-2 demos) and map raw bytes to ids, so vocab = 256 and the
per-char cross-entropy in nats converts to bits-per-char via 1/ln(2).

Standard full split is 90M/5M/5M; for the tiny studies we take a small prefix
and split it 90/10 train/val. Set ENWIK8_PATH or pass `path`.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

LN2 = math.log(2.0)
_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "enwik8")


class Enwik8:
    def __init__(self, n_bytes: int = 2_000_000, val_frac: float = 0.1,
                 path: str | None = None):
        path = path or os.environ.get("ENWIK8_PATH", _DEFAULT)
        with open(path, "rb") as f:
            raw = f.read(n_bytes)
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        n_val = int(len(data) * val_frac)
        self.train = torch.from_numpy(data[:-n_val].copy())
        self.val = torch.from_numpy(data[-n_val:].copy())
        self.vocab = 256

    def __repr__(self):
        return f"Enwik8(train={len(self.train):,} val={len(self.val):,} vocab=256)"


def batch_iter(data: torch.Tensor, block: int, batch: int, rng: np.random.Generator):
    """Yield (idx, targets) of shape (batch, block); next-char prediction."""
    n = len(data) - block - 1
    while True:
        ix = rng.integers(0, n, size=batch)
        x = torch.stack([data[i:i + block] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
        yield x, y

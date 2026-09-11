"""Chunked causal SV gate: stream tokens through the maintained solver.

The Titans-style chunk-frozen contract: within a chunk, readouts use the gate
state as of the chunk boundary (queries never see their own chunk); at each
boundary the chunk's keys are added with the incremental C&P path. Dropping a
reserve token is certified to leave the *currently solved* decision function
unchanged. That certificate is point-in-time: a removed reserve token could
have become active after a future admission. Consequently, budgeted streaming
follows an empirical pruned-history trajectory; it is not certified equivalent
to repeatedly solving over every token ever observed.

C is fixed absolutely from the budget (C = 1/(nu * budget)), as in the original
algorithm where C is set once up front.
"""
from __future__ import annotations

import numpy as np

from cp_svm import FastOneClassSVM


class ChunkedSVGate:
    def __init__(self, nu: float = 0.3, budget: int = 512, chunk: int = 64,
                 ktype: str = "r", kpar: float = 2.0, seed_min: int = 16,
                 refactor_every: int = 500):
        self.nu = float(nu)
        self.budget = int(budget)
        self.chunk = int(chunk)
        self.kpar = float(kpar)
        self.ktype = ktype
        self.seed_min = int(seed_min)
        C = 1.0 / (self.nu * self.budget)
        self.gate = FastOneClassSVM(C=C, ktype=ktype, kpar=kpar,
                                    refactor_every=refactor_every)
        self._buffer: list[np.ndarray] = []
        self._seeded = False
        self.n_seen = 0
        self.n_evicted = 0

    def _maybe_seed(self):
        need = max(self.seed_min, int(np.ceil(1.0 / self.gate.C)) + 2)
        if not self._seeded and len(self._buffer) >= need:
            X = np.vstack(self._buffer)
            self.gate.seed_from_qp(X)
            self._buffer = []
            self._seeded = True

    def feed(self, keys: np.ndarray):
        """Feed a block of keys (one or more chunks); returns self."""
        keys = np.atleast_2d(np.asarray(keys, dtype=np.float64))
        for k in keys:
            self.n_seen += 1
            if not self._seeded:
                self._buffer.append(k[None, :])
                self._maybe_seed()
                continue
            self.gate.add_point(k)
            if self.n_seen % self.chunk == 0:
                # Point-in-time output-preserving pruning. Future admissions can
                # make the pruned trajectory diverge from a full-history solve.
                self.n_evicted += self.gate.evict_reserve(keep_at_most=self.budget)
        return self

    # ------------------------------------------------------------- readout
    def state_size(self) -> int:
        return len(self.gate.alpha) + len(self._buffer)

    def support_size(self) -> int:
        return len(self.gate.S) + len(self.gate.E)

    def attention(self, Q: np.ndarray, V: np.ndarray | None = None):
        """Alpha-gated RBF attention over the retained tokens.

        Q: (m, d) queries. V: values aligned with the RETAINED tokens (use
        `retained_X` to know which); if None, returns the (m, n_retained)
        weight matrix instead.
        """
        from cp_svm.kernels import radial_kernel
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
        Xr = self.gate.X
        w = radial_kernel(Q, Xr, self.kpar) * self.gate.alpha[None, :]
        denom = w.sum(axis=1, keepdims=True)
        denom[denom < 1e-12] = 1e-12
        if V is None:
            return w / denom
        return (w @ V) / denom

    @property
    def retained_X(self) -> np.ndarray:
        return self.gate.X

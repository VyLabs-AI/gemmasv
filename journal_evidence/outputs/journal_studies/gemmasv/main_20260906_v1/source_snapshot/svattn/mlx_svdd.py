"""Batched SVDD dual solver on MLX (Apple-silicon GPU).

Solves, for a whole batch of G = (batch x heads [x chunk-boundaries]) independent
problems at once,

    min_a  a^T K a - diag(K)^T a   s.t.  sum(a)=1, 0<=a<=C

by accelerated projected gradient (FISTA) with a vectorized capped-simplex
projection. With finite iterations and single-precision arithmetic, this path
estimates the unregularized SVDD coefficients and active partition. It is the
parallel training path, not the maintained fp64 verification/deletion path;
tests compare its realized forward values with explicit numerical tolerances.

MLX is optional; import errors are deferred so the rest of the package works
without it.
"""
from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx
    _HAVE_MLX = True
except ImportError:  # pragma: no cover
    mx = None
    _HAVE_MLX = False


def _project_capped_simplex(Z, C, mask, iters=30):
    """Project rows of Z (G, n) onto {0<=a<=C, sum_valid(a)=1, a=0 off mask} via
    bisection on lambda. mask (G, n) in {0,1}; only valid entries carry mass."""
    big = mx.array(1e30)
    hi = mx.max(mx.where(mask > 0, Z, -big), axis=1) + 1.0
    lo = mx.min(mx.where(mask > 0, Z, big), axis=1) - C - 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = mx.sum(mx.clip(Z - mid[:, None], 0.0, C) * mask, axis=1)
        too_big = s > 1.0
        lo = mx.where(too_big, mid, lo)
        hi = mx.where(too_big, hi, mid)
    lam = 0.5 * (lo + hi)
    return mx.clip(Z - lam[:, None], 0.0, C) * mask


def _fista_step(a, y, t, K, diagK, Linv, C, mask):
    """One FISTA iteration (fused via mx.compile). Returns (a_new, y_new, t_new)."""
    grad = 2.0 * mx.squeeze(K @ y[:, :, None], axis=2) - diagK
    a_new = _project_capped_simplex(y - grad * Linv[:, None], C, mask)
    t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
    y_new = a_new + ((t - 1.0) / t_new) * (a_new - a)
    return a_new, y_new, t_new


_step_compiled = mx.compile(_fista_step) if _HAVE_MLX else None


def svdd_fista_mlx(K, C, iters=60, power_iters=8, mask=None):
    """Batched FISTA SVDD. K: mx.array (G, n, n) symmetric PSD. Returns a (G, n).

    `mask` (G, n) in {0,1} marks each problem's valid prefix length (so many
    different-size problems can be padded into one batch); padded entries get
    a=0 and are excluded from the sum=1 constraint. Each iteration is a single
    fused (compiled) graph, so the cost is ~`iters` GPU dispatches for the batch."""
    G, n, _ = K.shape
    if mask is None:
        mask = mx.ones((G, n))
    diagK = mx.diagonal(K, axis1=1, axis2=2) * mask          # (G, n)
    # Lipschitz of grad(=2K): L = 2 lambda_max(K), via batched power iteration.
    v = mx.random.normal((G, n)) * mask
    v = v / mx.clip(mx.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    for _ in range(power_iters):
        v = mx.squeeze(K @ v[:, :, None], axis=2) * mask
        v = v / mx.clip(mx.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    lam_max = mx.sum(v * mx.squeeze(K @ v[:, :, None], axis=2), axis=1)
    Linv = 1.0 / (2.0 * lam_max + 1e-6)
    a = mask / mx.clip(mx.sum(mask, axis=1, keepdims=True), 1.0, None)
    y = a
    t = mx.array(1.0)
    Cmx = mx.array(float(C))
    for _ in range(iters):
        a, y, t = _step_compiled(a, y, t, K, diagK, Linv, Cmx, mask)
    mx.eval(a)
    return a


def solve_svdd_batch_np(K_np: np.ndarray, C: float, iters: int = 200) -> np.ndarray:
    """numpy-in/out convenience wrapper around the MLX solver."""
    if not _HAVE_MLX:
        raise ImportError("mlx is required for solve_svdd_batch_np")
    a = svdd_fista_mlx(mx.array(K_np.astype(np.float32)), float(C), iters=iters)
    return np.array(a, dtype=np.float64)

"""A minimal Support Vector Attention primitive (Paper A, Option-C v1 spirit).

The attention weights over context tokens ARE the max-margin support
coefficients from a one-class SVDD over the keys: reserve tokens receive exactly
zero weight (certified-inert sparsity), margin/error tokens carry the mass. The
readout for a query q is

    o(q) = sum_i  alpha_i * kappa(q, x_i) * v_i

where alpha solves the one-class SVDD over the keys {x_i} and is produced by the
differentiable `OneClassSVDD` layer. Because alpha enters the output, the
implicit-gradient backward pass (reusing the KKT inverse R) is load-bearing:
the whole layer trains end-to-end with gradients that flow through the max-margin
solve for free.

This module is the gradient-verified building block; the chunked, multi-head,
causal sequence wrapper (and the MLX port) build on top of it.
"""
from __future__ import annotations

import torch

from .diff_svdd import OneClassSVDD


def rbf_gram(A: torch.Tensor, B: torch.Tensor, kpar: float) -> torch.Tensor:
    d2 = torch.cdist(A, B) ** 2
    return torch.exp(-d2 / (kpar * kpar))


class SVAttention(torch.nn.Module):
    """Support-vector attention over a single block of context tokens.

    Args:
        C: SVDD box bound (ν-style budget; controls error-set / sparsity).
        kpar: RBF bandwidth for both the gate and the readout kernel.
        normalize: divide the readout by the summed attention mass (like softmax
            attention's normalization); off by default to keep alpha's scale.
    """

    def __init__(self, C: float = 0.25, kpar: float = 2.0, normalize: bool = False,
                 tol: float = 1e-7):
        super().__init__()
        self.svdd = OneClassSVDD(C=C, tol=tol)
        self.kpar = float(kpar)
        self.normalize = normalize

    def forward(self, X: torch.Tensor, V: torch.Tensor, Q: torch.Tensor):
        """X: (n,d) keys, V: (n,dv) values, Q: (m,d) queries -> O: (m,dv)."""
        K_keys = rbf_gram(X, X, self.kpar)
        alpha, _rho = self.svdd(K_keys)            # (n,), max-margin coefficients
        Kqx = rbf_gram(Q, X, self.kpar)            # (m,n)
        weights = Kqx * alpha.unsqueeze(0)         # (m,n), sparse in n
        O = weights @ V                            # (m,dv)
        if self.normalize:
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            O = O / denom
        return O

    def support_mask(self, X: torch.Tensor):
        """Boolean mask of context tokens with nonzero (support) weight."""
        with torch.no_grad():
            K_keys = rbf_gram(X, X, self.kpar)
            alpha, _ = self.svdd(K_keys)
            return alpha.abs() > 0

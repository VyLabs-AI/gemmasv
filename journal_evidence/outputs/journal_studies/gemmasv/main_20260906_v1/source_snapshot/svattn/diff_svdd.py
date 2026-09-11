"""Differentiable one-class SVDD layer with an implicit-gradient backward pass.

Once the active-set partition is fixed, the free variables solve a small linear
system in the bordered KKT matrix M. The maintained-path backward pass (VJP)
reuses M^{-1} (= the C&P matrix R), with no second system solve and no optimizer
unrolling.

Forward (one-class ν-SVDD, matching cp_svm.solve_svdd_qp):
    minimize_a   a^T K a - diag(K)^T a
    subject to   sum(a) = 1,  0 <= a_i <= C

KKT gradient g_i = rho - K_ii + 2 (K a)_i:  g_i = 0 on margin S,
alpha = C on error E, alpha = 0 on reserve R. With the partition fixed, the free
variables u = [rho; a_S] solve  M u = c  with

    M = [[0, 1_S^T], [1_S, 2 K_SS]],
    c = [1 - |E| C ;  diag(K)_S - 2 C K_SE 1_E],     R := M^{-1}.

Backward: for upstream cotangents (g_rho, g_aS), with w = R^T [g_rho; g_aS],
    dL/dK[i,i] += w_S[i]                      (i in S)        [from diag(K)_S in c]
    dL/dK[i,e] += -2 C w_S[i]                 (i in S, e in E)[from K_SE in c]
    dL/dK[i,j] += -2 w_S[i] a_S[j]            (i, j in S)     [from 2 K_SS in M]
"""
from __future__ import annotations

import numpy as np
import torch


def _svdd_qp(K: np.ndarray, C: float):
    import cvxpy as cp
    n = K.shape[0]
    Ksym = 0.5 * (K + K.T)
    a = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(a, cp.psd_wrap(Ksym)) - np.diag(K) @ a)
    cons = [cp.sum(a) == 1, a >= 0, a <= C]
    cp.Problem(obj, cons).solve(solver=cp.CLARABEL, tol_gap_abs=1e-11,
                                tol_gap_rel=1e-11, tol_feas=1e-11)
    return np.clip(np.asarray(a.value, dtype=np.float64), 0.0, C)


def svdd_solve_partition(K: np.ndarray, C: float, tol: float = 1e-6):
    """Solve the QP, recover the active-set partition, and re-solve the bordered
    KKT system exactly so (alpha, rho) are mutually consistent for the backward."""
    n = K.shape[0]
    a = _svdd_qp(K, C)
    S = np.where((a > tol) & (a < C - tol))[0]
    E = np.where(a >= C - tol)[0]
    R = np.where(a <= tol)[0]
    if len(S) == 0:
        raise RuntimeError("SVDD: empty margin set (degenerate); perturb data/C")

    m = len(S)
    M = np.zeros((m + 1, m + 1))
    M[0, 1:] = 1.0
    M[1:, 0] = 1.0
    M[1:, 1:] = 2.0 * K[np.ix_(S, S)]
    Rmat = np.linalg.inv(M)

    c = np.empty(m + 1)
    c[0] = 1.0 - len(E) * C
    c[1:] = np.diag(K)[S] - 2.0 * C * (K[np.ix_(S, E)].sum(axis=1) if len(E) else 0.0)
    u = Rmat @ c
    rho = float(u[0])
    aS = u[1:]

    alpha = np.zeros(n)
    alpha[S] = aS
    alpha[E] = C
    return alpha, rho, S, E, R, Rmat


class _SVDDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, K: torch.Tensor, C: float, tol: float):
        Knp = K.detach().cpu().numpy().astype(np.float64)
        alpha, rho, S, E, R, Rmat = svdd_solve_partition(Knp, C, tol)
        ctx.save_for_backward(K)
        ctx.alpha = alpha
        ctx.S, ctx.E = S, E
        ctx.Rmat = Rmat
        ctx.C = C
        out_alpha = torch.from_numpy(alpha).to(K)
        out_rho = torch.tensor(rho, dtype=K.dtype, device=K.device)
        return out_alpha, out_rho

    @staticmethod
    def backward(ctx, grad_alpha: torch.Tensor, grad_rho: torch.Tensor):
        (K,) = ctx.saved_tensors
        n = K.shape[0]
        S, E = ctx.S, ctx.E
        Rmat = ctx.Rmat
        C = ctx.C
        alpha = ctx.alpha
        aS = alpha[S]

        ga = grad_alpha.detach().cpu().numpy().astype(np.float64)
        grho = float(grad_rho.detach().cpu().numpy()) if grad_rho is not None else 0.0

        ubar = np.empty(len(S) + 1)
        ubar[0] = grho
        ubar[1:] = ga[S]
        w = Rmat.T @ ubar
        wS = w[1:]

        G = np.zeros((n, n), dtype=np.float64)
        # c term: diag(K)_S
        G[S, S] += wS
        # c term: -2 C K_SE
        if len(E) > 0:
            G[np.ix_(S, E)] += -2.0 * C * wS[:, None]
        # M term: -2 w_S a_S^T over S x S
        G[np.ix_(S, S)] += -2.0 * np.outer(wS, aS)

        grad_K = torch.from_numpy(G).to(K)
        return grad_K, None, None


class OneClassSVDD(torch.nn.Module):
    """Differentiable one-class SVDD. Forward returns (alpha, rho) given Gram K.

    The decision/score function on points with Gram column k(x) is
        f(x) = 2 * sum_j alpha_j K(x_j, x) - rho.
    """

    def __init__(self, C: float = 1.0, tol: float = 1e-6):
        super().__init__()
        self.C = float(C)
        self.tol = float(tol)

    def forward(self, K: torch.Tensor):
        return _SVDDFn.apply(K, self.C, self.tol)

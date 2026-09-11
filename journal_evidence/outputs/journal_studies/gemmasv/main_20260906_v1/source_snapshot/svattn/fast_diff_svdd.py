"""Fast differentiable one-class SVDD: the streaming-solver-backed bridge.

`OneClassSVDD` (diff_svdd.py) solves the SVDD QP with cvxpy on every forward.
That is fine for a single block but far too slow inside a causal sequence layer
that re-gates per chunk, per head. This module produces the *same* differentiable
algebraic map -- alpha = SVDD(K) with an implicit-gradient backward pass that
reuses the bordered KKT inverse R -- but obtains the active-set partition and R from the
fast incremental Cauwenberghs-Poggio solver (`cp_svm.oneclass_fast.FastOneClassSVM`)
instead of cvxpy. cvxpy is used at most once, to seed a small prefix; all
subsequent points are added by cvxpy-free rank-1 (Sherman-Morrison) updates.

Backward (the same fixed-partition algebra as `diff_svdd._SVDDFn`): for a fixed
partition (S, E, R) the free variables u = [rho; alpha_S] solve M u = c with
M = [[0, 1^T], [1, 2 K_SS]], R = M^{-1}, c = [1 - |E| C; diag(K)_S - 2 C K_SE 1].
For upstream cotangents (g_rho, g_aS), with w = R^T [g_rho; g_aS]:
    dL/dK[i,i] += w_S[i]                 (i in S)
    dL/dK[i,e] += -2 C w_S[i]            (i in S, e in E)
    dL/dK[i,j] += -2 w_S[i] alpha_S[j]   (i, j in S)
The only solve is the multiply by R^T, which the forward pass already maintains.
"""
from __future__ import annotations

import numpy as np
import torch

from cp_svm.oneclass_fast import FastOneClassSVM
from .sv_attention import rbf_gram


def fast_svdd_state(X, C, kpar: float = 2.0, ktype: str = "r",
                    n_seed: int | None = None, refactor_every: int = 500):
    """Stream points X through the fast incremental solver and return it.

    cvxpy seeds only the first ``n_seed`` points (enough to satisfy
    sum(alpha)=1 under the box cap C); the remainder are added by cvxpy-free
    rank-1 updates. With no eviction the solver's index order equals X's row
    order, so (S, E, R, _Rmat) align with X.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n = len(X)
    solver = FastOneClassSVM(C=float(C), ktype=ktype, kpar=float(kpar),
                             refactor_every=refactor_every)
    if n_seed is None:
        n_seed = int(np.ceil(1.0 / float(C))) + 2
    n_seed = max(2, min(n_seed, n))
    solver.seed_from_qp(X[:n_seed])
    for i in range(n_seed, n):
        solver.add_point(X[i])
    return solver


class _FastSVDDFn(torch.autograd.Function):
    """alpha, rho = SVDD(K) for a *fixed* partition (S, E, Rmat); backward = VJP."""

    @staticmethod
    def forward(ctx, K, C, S, E, Rmat):
        Knp = K.detach().cpu().numpy().astype(np.float64)
        S = np.asarray(S, dtype=np.intp)
        E = np.asarray(E, dtype=np.intp)
        m = len(S)
        if m == 0:
            raise RuntimeError(
                "FastDiffSVDD: empty margin set (degenerate); the implicit "
                "gradient needs a non-empty margin -- adjust C / data / n_seed")
        c = np.empty(m + 1)
        c[0] = 1.0 - len(E) * C
        kse = Knp[np.ix_(S, E)].sum(axis=1) if len(E) else 0.0
        c[1:] = np.diag(Knp)[S] - 2.0 * C * kse
        u = Rmat @ c
        rho = float(u[0])
        n = Knp.shape[0]
        alpha = np.zeros(n)
        alpha[S] = u[1:]
        alpha[E] = C

        ctx.save_for_backward(K)
        ctx.S, ctx.E, ctx.Rmat, ctx.C, ctx.alpha = S, E, Rmat, float(C), alpha
        out_alpha = torch.from_numpy(alpha).to(K)
        out_rho = torch.tensor(rho, dtype=K.dtype, device=K.device)
        return out_alpha, out_rho

    @staticmethod
    def backward(ctx, grad_alpha, grad_rho):
        (K,) = ctx.saved_tensors
        S, E, Rmat, C, alpha = ctx.S, ctx.E, ctx.Rmat, ctx.C, ctx.alpha
        n = K.shape[0]
        aS = alpha[S]

        ga = grad_alpha.detach().cpu().numpy().astype(np.float64)
        grho = float(grad_rho.detach().cpu().numpy()) if grad_rho is not None else 0.0

        ubar = np.empty(len(S) + 1)
        ubar[0] = grho
        ubar[1:] = ga[S]
        w = Rmat.T @ ubar
        wS = w[1:]

        G = np.zeros((n, n), dtype=np.float64)
        G[S, S] += wS                                    # diag(K)_S term in c
        if len(E) > 0:
            G[np.ix_(S, E)] += -2.0 * C * wS[:, None]    # K_SE term in c
        G[np.ix_(S, S)] += -2.0 * np.outer(wS, aS)       # 2 K_SS term in M

        grad_K = torch.from_numpy(G).to(K)
        return grad_K, None, None, None, None


def fast_diff_svdd(X: torch.Tensor, C: float = 0.25, kpar: float = 2.0,
                   ktype: str = "r", n_seed: int | None = None):
    """Differentiable (alpha, rho) for the one-class SVDD over keys X.

    Partition + bordered-KKT inverse come from the fast incremental solver
    (cvxpy-free except a small seed); alpha/rho are differentiable w.r.t. X
    through K = rbf_gram(X, X, kpar) with the implicit-gradient backward pass.
    """
    Xnp = X.detach().cpu().numpy().astype(np.float64)
    solver = fast_svdd_state(Xnp, C=C, kpar=kpar, ktype=ktype, n_seed=n_seed)
    S, E = tuple(solver.S), tuple(solver.E)
    Rmat = solver._Rmat if solver._Rmat is not None else np.zeros((1, 1))
    if ktype in ("l", "linear"):
        K = X @ X.t()
    else:
        K = rbf_gram(X, X, kpar)
    return _FastSVDDFn.apply(K, float(C), S, E, Rmat)


def _bordered_inv(Knp: np.ndarray, S, C: float) -> np.ndarray:
    """Inverse of the bordered KKT matrix M = [[0, 1^T], [1, 2 K_SS]] over the
    margin (free) set S -- the same R the incremental solver maintains, but built
    directly from K_SS and a partition found by any solver (e.g. batched FISTA)."""
    m = len(S)
    if m == 0:
        return np.zeros((1, 1))
    M = np.zeros((m + 1, m + 1))
    M[0, 1:] = 1.0
    M[1:, 0] = 1.0
    M[1:, 1:] = 2.0 * Knp[np.ix_(S, S)]
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def diff_svdd_from_partition(K: torch.Tensor, C: float, S, E):
    """Differentiable (alpha, rho) given a precomputed partition (S, E) and the
    differentiable Gram K (n, n). The forward recomputes the *exact* alpha for the
    partition (R @ c) and the backward is the exact implicit VJP -- independent of
    which solver found (S, E). Pairs with the batched MLX FISTA partition finder."""
    Knp = K.detach().cpu().numpy().astype(np.float64)
    Rmat = _bordered_inv(Knp, S, float(C))
    return _FastSVDDFn.apply(K, float(C), tuple(S), tuple(E), Rmat)


def batched_svdd_from_masks(gram: torch.Tensor, C: float, s_mask: torch.Tensor,
                            e_mask: torch.Tensor):
    """Differentiable SVDD alpha for a whole BATCH at once, given the active-set
    partition (S, E) as boolean masks (found by the batched FISTA solver).

    Solves the bordered KKT system [[0,1_S^T],[1_S,2K_SS]] [rho; a_S] = c per
    problem as a single batched `torch.linalg.solve`, so the implicit gradient
    w.r.t. the (differentiable) gram is handled by autograd -- no per-problem
    Python loop and no custom VJP. Returns (alpha (G,n), gate_valid (G,) bool);
    rows with an empty margin set are flagged invalid (caller falls back to the
    ungated kernel, matching the sequential path).

    gram: (G, n, n) differentiable. s_mask, e_mask: (G, n) bool.
    """
    G, n, _ = gram.shape
    dev, dt = gram.device, gram.dtype
    s_cnt = s_mask.sum(1)                                    # (G,)
    e_cnt = e_mask.sum(1).to(dt)
    m_max = int(s_cnt.max().item()) if G else 0
    gate_valid = s_cnt > 0
    if m_max == 0:
        return e_mask.to(dt) * C, gate_valid                 # E -> C, rest 0

    # Padded S index tensor: for each row, the column indices of its S members,
    # padded with 0 and tracked by `sm` (1 = real slot, 0 = pad).
    order = torch.argsort(s_mask.to(torch.int8), dim=1, descending=True, stable=True)
    S_idx = order[:, :m_max]                                 # (G, m_max)
    ar = torch.arange(m_max, device=dev)
    sm = (ar[None, :] < s_cnt[:, None]).to(dt)               # (G, m_max) validity

    diagK = torch.diagonal(gram, dim1=1, dim2=2)             # (G, n)
    K_rows = torch.gather(gram, 1, S_idx[:, :, None].expand(G, m_max, n))  # (G,m_max,n)
    K_SS = torch.gather(K_rows, 2, S_idx[:, None, :].expand(G, m_max, m_max))
    diagK_S = torch.gather(diagK, 1, S_idx)                  # (G, m_max)
    K_SE = (K_rows * e_mask.to(dt)[:, None, :]).sum(2)       # (G, m_max) sum_E K[S,E]

    # bordered block: 2 K_SS, with padded rows/cols zeroed and a 1 on the pad diag.
    # Ridge on the valid diagonal guards near-singular K_SS (redundant keys / smooth
    # kernels): the RBF diag is 1 so the block diag is exactly 2, and 1e-3 keeps the
    # worst-case (rank-1 K_SS) condition number ~6e4 -- safe in fp32, negligible bias.
    ridge = 1e-3
    block = 2.0 * K_SS * sm[:, :, None] * sm[:, None, :]
    block = block + torch.diag_embed((1.0 - sm) + ridge * sm)  # pad->identity, S->+ridge
    M = gram.new_zeros(G, m_max + 1, m_max + 1)
    M[:, 0, 1:] = sm
    M[:, 1:, 0] = sm
    M[:, 1:, 1:] = block
    # Empty-margin-set problems (all nonzero alpha at the cap C) would give an
    # all-zero row 0 -> singular. Make their whole system identity (output is
    # discarded: gate_valid=False -> ungated fallback, matching the CPU path).
    M[:, 0, 0] = (s_cnt == 0).to(dt)

    c = gram.new_zeros(G, m_max + 1)
    c[:, 0] = 1.0 - C * e_cnt
    c[:, 1:] = (diagK_S - 2.0 * C * K_SE) * sm

    u = torch.linalg.solve(M, c.unsqueeze(-1)).squeeze(-1)   # (G, m_max+1)
    a_S = u[:, 1:] * sm                                      # (G, m_max); pad slots 0
    alpha = gram.new_zeros(G, n).scatter(1, S_idx, a_S)      # S members -> a_S
    alpha = torch.where(e_mask, alpha.new_full((), C), alpha)  # E -> C (R stays 0)
    return alpha, gate_valid


def diff_svdd_from_solver(X: torch.Tensor, solver: FastOneClassSVM, C: float,
                          kpar: float = 2.0, ktype: str = "r"):
    """Differentiable (alpha, rho) for the SVDD over keys X, reusing the partition
    (S, E, Rmat) of an *already-maintained* incremental solver instead of
    re-streaming X from scratch.

    Contract: ``solver`` must have been fed exactly the rows of ``X`` in order
    with no eviction, so its (S, E, R) indices align with X's rows. The bordered
    KKT inverse is snapshotted (copied) so later ``add_point`` calls on the same
    solver cannot corrupt the autograd tape.
    """
    S, E = tuple(solver.S), tuple(solver.E)
    Rmat = solver._Rmat.copy() if solver._Rmat is not None else np.zeros((1, 1))
    if ktype in ("l", "linear"):
        K = X @ X.t()
    else:
        K = rbf_gram(X, X, kpar)
    return _FastSVDDFn.apply(K, float(C), S, E, Rmat)


class FastDiffSVDD(torch.nn.Module):
    """Streaming-solver-backed differentiable one-class SVDD (X-based).

    Drop-in gate for a causal layer: forward(X) -> (alpha, rho), with the
    cvxpy-free incremental partition and the free implicit-gradient backward.
    """

    def __init__(self, C: float = 0.25, kpar: float = 2.0, ktype: str = "r",
                 n_seed: int | None = None):
        super().__init__()
        self.C = float(C)
        self.kpar = float(kpar)
        self.ktype = ktype
        self.n_seed = n_seed

    def forward(self, X: torch.Tensor):
        return fast_diff_svdd(X, C=self.C, kpar=self.kpar, ktype=self.ktype,
                              n_seed=self.n_seed)

"""Phase-0 exactness tests for the Cauwenberghs-Poggio one-class SVM port.

Ground truth is an independent batch QP (cvxpy/CLARABEL). We verify, on
well-conditioned data (so the QP solution is unique and the comparison is
meaningful):

  1. KKT consistency of the QP seed.
  2. Incremental solution == batch QP at every streamed step (adiabatic path is exact).
  3. Removal == batch QP on the retained set (exact unlearning).
  4. decrement o increment == identity to machine precision.

The original onlineSVM/toy_data.mat is intentionally near-degenerate
(cond(K) ~ 1e11 from near-duplicate points), which makes the QP solution
non-unique and breaks any vertex-level comparison. It is exercised separately
in test_illconditioned.py as a numerical-robustness stress case.
"""
import os
import numpy as np
import pytest

from cp_svm import (
    radial_kernel,
    solve_svdd_qp,
    recover_rho,
    partition_sets,
    OneClassIncrementalSVM,
)

KPAR = 1.0
NU = 0.1


def synthetic(n=60, d=4, seed=0):
    """Well-conditioned one-class data (spread Gaussian, no near-duplicates)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, d) * 1.5


def kkt_feasibility(K, alpha, rho, C, tol=1e-6):
    """Max KKT violation: |grad| on S, grad>0 on E, grad<0 on R."""
    grad = rho - np.diag(K) + 2.0 * (K @ alpha)
    S, E, R = partition_sets(alpha, C, tol=tol)
    v = 0.0
    if S:
        v = max(v, float(np.max(np.abs(grad[S]))))
    if E:
        v = max(v, float(max(0.0, grad[E].max())))
    if R:
        v = max(v, float(max(0.0, -grad[R].min())))
    return v


def test_qp_seed_is_kkt_consistent():
    X = synthetic()
    C = 1.0 / (len(X) * NU)
    K = radial_kernel(X, X, KPAR)
    alpha = solve_svdd_qp(K, C)
    assert abs(alpha.sum() - 1.0) < 1e-6
    assert alpha.min() > -1e-7 and alpha.max() < C + 1e-7
    rho = recover_rho(K, alpha, C)
    assert kkt_feasibility(K, alpha, rho, C) < 1e-4


def test_incremental_matches_qp_every_step():
    X = synthetic()
    n = len(X)
    C = 1.0 / (n * NU)
    n_seed = int(np.ceil(1.0 / C)) + 5

    m = OneClassIncrementalSVM(C=C, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed])

    worst = 0.0
    for i in range(n_seed, n):
        m.add_point(X[i])
        cur = X[: i + 1]
        K = radial_kernel(cur, cur, KPAR)
        aq = solve_svdd_qp(K, C)
        rq = recover_rho(K, aq, C)
        f_inc = m.decision_function(cur)
        f_qp = 2.0 * (K @ aq) - rq
        worst = max(worst, float(np.max(np.abs(f_inc - f_qp))))
        assert abs(m.alpha.sum() - 1.0) < 1e-6
        assert kkt_feasibility(K, m.alpha, m.rho, C) < 1e-5
    assert worst < 1e-4, f"worst incremental-vs-QP decision mismatch {worst:.2e}"


def test_decrement_inverts_increment_to_machine_precision():
    X = synthetic()
    C = 1.0 / (len(X) * NU)
    n_seed = 20

    m = OneClassIncrementalSVM(C=C, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed])
    f0 = m.decision_function(X[:n_seed]).copy()
    rho0, alpha0 = m.rho, m.alpha.copy()
    sets0 = (sorted(m.S), sorted(m.E), sorted(m.R))

    added = [m.add_point(X[n_seed + k]) for k in range(5)]
    for idx in reversed(added):
        m.remove_point(idx)

    assert m.alpha.shape == alpha0.shape
    assert np.max(np.abs(m.decision_function(X[:n_seed]) - f0)) < 1e-10
    assert abs(m.rho - rho0) < 1e-10
    assert np.max(np.abs(m.alpha - alpha0)) < 1e-10
    assert (sorted(m.S), sorted(m.E), sorted(m.R)) == sets0


def test_removal_matches_qp_on_retained_set():
    X = synthetic()
    n = len(X)
    C = 1.0 / (n * NU)
    n_seed = int(np.ceil(1.0 / C)) + 5

    m = OneClassIncrementalSVM(C=C, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed])
    for i in range(n_seed, n):
        m.add_point(X[i])

    victim = 5
    m.remove_point(victim)

    keep = [i for i in range(n) if i != victim]
    Xk = X[keep]
    Kk = radial_kernel(Xk, Xk, KPAR)
    aq = solve_svdd_qp(Kk, C)
    rq = recover_rho(Kk, aq, C)
    f_inc = m.decision_function(Xk)
    f_qp = 2.0 * (Kk @ aq) - rq
    assert np.max(np.abs(f_inc - f_qp)) < 1e-4
    assert kkt_feasibility(Kk, m.alpha, m.rho, C) < 1e-5

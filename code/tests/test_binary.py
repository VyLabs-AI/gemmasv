"""Exactness tests for the binary C&P SVM vs an independent batch QP."""
import numpy as np

from cp_svm import (
    radial_kernel,
    solve_svm_qp,
    recover_b_svm,
    partition_sets,
    BinaryIncrementalSVM,
)

KPAR = 1.0


def synthetic_binary(n=80, d=4, seed=1):
    """Two well-separated-ish Gaussian blobs, labels +-1, no near-duplicates."""
    rng = np.random.RandomState(seed)
    n1 = n // 2
    X = np.vstack([rng.randn(n1, d) * 1.2 + 1.0, rng.randn(n - n1, d) * 1.2 - 1.0])
    y = np.concatenate([np.ones(n1), -np.ones(n - n1)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


def kkt_feasibility(K, y, alpha, b, C, tol=1e-6):
    g = (np.outer(y, y) * K) @ alpha + y * b - 1.0
    S, E, R = partition_sets(alpha, C, tol=tol)
    v = abs(float(y @ alpha))             # equality constraint y^T alpha = 0
    if S:
        v = max(v, float(np.max(np.abs(g[S]))))
    if E:
        v = max(v, float(max(0.0, g[E].max())))
    if R:
        v = max(v, float(max(0.0, -g[R].min())))
    return v


def test_binary_incremental_matches_qp_every_step():
    X, y = synthetic_binary()
    n = len(X)
    C = 1.0
    n_seed = 20

    m = BinaryIncrementalSVM(C=C, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed], y[:n_seed])

    worst = 0.0
    for i in range(n_seed, n):
        m.add_point(X[i], y[i])
        cur, ycur = X[: i + 1], y[: i + 1]
        K = radial_kernel(cur, cur, KPAR)
        aq = solve_svm_qp(K, ycur, C)
        bq = recover_b_svm(K, ycur, aq, C)
        f_inc = m.decision_function(cur)
        f_qp = K @ (aq * ycur) + bq
        worst = max(worst, float(np.max(np.abs(f_inc - f_qp))))
        assert abs(m.y @ m.alpha) < 1e-6
        assert kkt_feasibility(K, ycur, m.alpha, m.b, C) < 1e-5
    # vs-QP comparison is limited by the QP solver's accuracy (b recovery).
    assert worst < 1e-3, f"worst binary incremental-vs-QP mismatch {worst:.2e}"


def test_binary_decrement_inverts_increment():
    X, y = synthetic_binary()
    C = 1.0
    n_seed = 30

    m = BinaryIncrementalSVM(C=C, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed], y[:n_seed])
    f0 = m.decision_function(X[:n_seed]).copy()
    b0, alpha0 = m.b, m.alpha.copy()

    added = [m.add_point(X[n_seed + k], y[n_seed + k]) for k in range(6)]
    for idx in reversed(added):
        m.remove_point(idx)

    assert m.alpha.shape == alpha0.shape
    # add/remove cycles invert via repeated direct inversions: fp64 active-set precision.
    assert np.max(np.abs(m.decision_function(X[:n_seed]) - f0)) < 1e-6
    assert abs(m.b - b0) < 1e-6
    assert np.max(np.abs(m.alpha - alpha0)) < 1e-6


def test_binary_removal_matches_qp_on_retained_set():
    X, y = synthetic_binary()
    n = len(X)
    C = 1.0
    n_seed = 20

    m = BinaryIncrementalSVM(C=C, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed], y[:n_seed])
    for i in range(n_seed, n):
        m.add_point(X[i], y[i])

    victim = 7
    m.remove_point(victim)

    keep = [i for i in range(n) if i != victim]
    Xk, yk = X[keep], y[keep]
    Kk = radial_kernel(Xk, Xk, KPAR)
    aq = solve_svm_qp(Kk, yk, C)
    bq = recover_b_svm(Kk, yk, aq, C)
    f_inc = m.decision_function(Xk)
    f_qp = Kk @ (aq * yk) + bq
    assert np.max(np.abs(f_inc - f_qp)) < 1e-3

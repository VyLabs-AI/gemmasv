"""Exactness tests for the incremental epsilon-SVR vs an independent batch QP."""
import numpy as np

from cp_svm import radial_kernel, solve_svr_qp, recover_b_svr, SVRIncremental

KPAR = 1.5
EPS = 0.1
C = 5.0


def synthetic_regression(n=70, d=3, seed=2):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d) * 1.3
    w = rng.randn(d)
    y = np.tanh(X @ w) + 0.05 * rng.randn(n)   # smooth nonlinear target
    return X, y


def svr_kkt_violation(K, y, theta, b, C, eps, tol=1e-6):
    h = K @ theta + b - y
    v = abs(float(theta.sum()))                          # sum theta = 0
    for i in range(len(theta)):
        if abs(theta[i]) <= tol:                         # reserve: |h| <= eps
            v = max(v, max(0.0, abs(h[i]) - eps - 1e-6))
        elif abs(theta[i]) >= C - tol:                   # error: |h| >= eps
            v = max(v, max(0.0, eps - abs(h[i]) - 1e-6))
        else:                                            # margin: |h| == eps
            v = max(v, abs(abs(h[i]) - eps))
    return v


def test_svr_incremental_matches_qp_every_step():
    X, y = synthetic_regression()
    n = len(X)
    n_seed = 15

    m = SVRIncremental(C=C, eps=EPS, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed], y[:n_seed])

    worst = 0.0
    for i in range(n_seed, n):
        m.add_point(X[i], y[i])
        cur, ycur = X[: i + 1], y[: i + 1]
        K = radial_kernel(cur, cur, KPAR)
        th = solve_svr_qp(K, ycur, C, EPS)
        bq = recover_b_svr(K, ycur, th, C, EPS)
        pred_inc = m.predict(cur)
        pred_qp = K @ th + bq
        worst = max(worst, float(np.max(np.abs(pred_inc - pred_qp))))
        assert abs(m.theta.sum()) < 1e-6
        assert svr_kkt_violation(K, ycur, m.theta, m.b, C, EPS) < 1e-4
    assert worst < 1e-3, f"worst SVR incremental-vs-QP mismatch {worst:.2e}"


def test_svr_decrement_inverts_increment():
    X, y = synthetic_regression()
    n_seed = 25

    m = SVRIncremental(C=C, eps=EPS, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed], y[:n_seed])
    p0 = m.predict(X[:n_seed]).copy()
    b0, theta0 = m.b, m.theta.copy()

    added = [m.add_point(X[n_seed + k], y[n_seed + k]) for k in range(6)]
    for idx in reversed(added):
        m.remove_point(idx)

    assert m.theta.shape == theta0.shape
    assert np.max(np.abs(m.predict(X[:n_seed]) - p0)) < 1e-6
    assert abs(m.b - b0) < 1e-6
    assert np.max(np.abs(m.theta - theta0)) < 1e-6


def test_svr_removal_matches_qp_on_retained_set():
    X, y = synthetic_regression()
    n = len(X)
    n_seed = 15

    m = SVRIncremental(C=C, eps=EPS, ktype="r", kpar=KPAR)
    m.seed_from_qp(X[:n_seed], y[:n_seed])
    for i in range(n_seed, n):
        m.add_point(X[i], y[i])

    victim = 4
    m.remove_point(victim)

    keep = [i for i in range(n) if i != victim]
    Xk, yk = X[keep], y[keep]
    Kk = radial_kernel(Xk, Xk, KPAR)
    th = solve_svr_qp(Kk, yk, C, EPS)
    bq = recover_b_svr(Kk, yk, th, C, EPS)
    pred_inc = m.predict(Xk)
    pred_qp = Kk @ th + bq
    assert np.max(np.abs(pred_inc - pred_qp)) < 1e-3

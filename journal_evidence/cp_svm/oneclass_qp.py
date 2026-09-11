"""Batch QP ground truth for the one-class (nu-SVDD) problem.

This is the solver-independent reference the incremental algorithm is verified
against. The dual problem implemented by the MATLAB code (derivable from the
gradient in Compute_grad.m) is:

    minimize_alpha   alpha^T K alpha - diag(K)^T alpha
    subject to       sum(alpha) = 1,   0 <= alpha_i <= C

with the KKT stationarity gradient (matching Compute_grad.m, y_i == 1):

    g_i = rho - K_ii + 2 * sum_j K_ij alpha_j

Active-set membership:
    g_i = 0, 0 < alpha_i < C   -> margin set S
    g_i < 0, alpha_i = C        -> error set E (outliers)
    g_i > 0, alpha_i = 0        -> reserve set R (inert)
"""
from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None


class SVDDQPInfeasibleError(RuntimeError):
    """The box constraints cannot carry unit alpha mass."""


def solve_svdd_qp(K: np.ndarray, C: float) -> np.ndarray:
    """Solve the nu-SVDD dual to high accuracy and return alpha."""
    if cp is None:
        raise ImportError("cvxpy is required for the batch-QP ground truth")
    n = K.shape[0]
    if n * float(C) < 1.0 - 1e-12:
        raise SVDDQPInfeasibleError(
            f"SVDD QP is infeasible: n*C={n * float(C):.12g} < 1 "
            f"(n={n}, C={float(C):.12g})"
        )
    # Symmetrize to keep the QP numerically PSD.
    Ksym = 0.5 * (K + K.T)
    alpha = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(alpha, cp.psd_wrap(Ksym)) - np.diag(K) @ alpha)
    constraints = [cp.sum(alpha) == 1, alpha >= 0, alpha <= C]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
               tol_feas=1e-10, tol_infeas_abs=1e-10, tol_infeas_rel=1e-10)
    if alpha.value is None:
        if prob.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
            raise SVDDQPInfeasibleError(
                f"SVDD QP solver reported infeasible despite n*C="
                f"{n * float(C):.12g}: status={prob.status}"
            )
        raise RuntimeError(f"QP did not solve: status={prob.status}")
    a = np.asarray(alpha.value, dtype=np.float64)
    # Clean tiny numerical violations of the box.
    a = np.clip(a, 0.0, C)
    return a


def recover_rho(K: np.ndarray, alpha: np.ndarray, C: float, tol: float = 1e-6) -> float:
    """Recover the offset rho from any margin support vector.

    rho = K_ii - 2 (K alpha)_i for any i with 0 < alpha_i < C.
    Falls back to averaging over the most-interior points if none are strictly free.
    """
    Ka = K @ alpha
    g_const = np.diag(K) - 2.0 * Ka  # = rho at margin points
    margin = (alpha > tol) & (alpha < C - tol)
    if np.any(margin):
        return float(np.mean(g_const[margin]))
    # Degenerate fallback: use points nearest the interior.
    order = np.argsort(np.abs(alpha - C / 2.0))
    return float(g_const[order[0]])


def partition_sets(alpha: np.ndarray, C: float, tol: float = 1e-6):
    """Partition indices into (S margin, E error, R reserve) by alpha value."""
    S = np.where((alpha > tol) & (alpha < C - tol))[0].tolist()
    E = np.where(alpha >= C - tol)[0].tolist()
    R = np.where(alpha <= tol)[0].tolist()
    return S, E, R


def solve_svm_qp(K: np.ndarray, y: np.ndarray, C) -> np.ndarray:
    """Solve the C-SVM dual to high accuracy and return alpha.

        minimize    0.5 alpha^T (y y^T o K) alpha - 1^T alpha
        subject to  y^T alpha = 0,  0 <= alpha_i <= C_i

    C may be a scalar or a per-point upper-bound array (class-weighted SVM).
    """
    if cp is None:
        raise ImportError("cvxpy is required for the batch-QP ground truth")
    n = K.shape[0]
    Cv = np.full(n, float(C)) if np.isscalar(C) else np.asarray(C, dtype=np.float64)
    Q = np.outer(y, y) * (0.5 * (K + K.T))
    alpha = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(alpha, cp.psd_wrap(Q)) - cp.sum(alpha))
    constraints = [y @ alpha == 0, alpha >= 0, alpha <= Cv]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
               tol_feas=1e-10, tol_infeas_abs=1e-10, tol_infeas_rel=1e-10)
    if alpha.value is None:
        raise RuntimeError(f"SVM QP did not solve: status={prob.status}")
    return np.clip(np.asarray(alpha.value, dtype=np.float64), 0.0, Cv)


def recover_b_svm(K: np.ndarray, y: np.ndarray, alpha: np.ndarray, C: float,
                  tol: float = 1e-6) -> float:
    """Recover bias b from margin SVs: b = y_i - sum_j alpha_j y_j K_ij."""
    ay = alpha * y
    f_no_b = K @ ay
    margin = (alpha > tol) & (alpha < C - tol)
    if np.any(margin):
        return float(np.mean(y[margin] - f_no_b[margin]))
    order = np.argsort(np.abs(alpha - C / 2.0))
    i = order[0]
    return float(y[i] - f_no_b[i])


def solve_svr_qp(K: np.ndarray, y: np.ndarray, C: float, eps: float):
    """Solve the epsilon-SVR dual; return theta = alpha - alpha*.

        minimize    0.5 theta^T K theta + eps 1^T(a + a*) - y^T theta
        subject to  1^T theta = 0,  0 <= a, a* <= C,  theta = a - a*
    """
    if cp is None:
        raise ImportError("cvxpy is required for the batch-QP ground truth")
    n = K.shape[0]
    Ksym = 0.5 * (K + K.T)
    a = cp.Variable(n)
    astar = cp.Variable(n)
    theta = a - astar
    objective = cp.Minimize(0.5 * cp.quad_form(theta, cp.psd_wrap(Ksym))
                            + eps * cp.sum(a + astar) - y @ theta)
    constraints = [cp.sum(theta) == 0, a >= 0, a <= C, astar >= 0, astar <= C]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
               tol_feas=1e-10, tol_infeas_abs=1e-10, tol_infeas_rel=1e-10)
    if a.value is None:
        raise RuntimeError(f"SVR QP did not solve: status={prob.status}")
    th = np.asarray(a.value - astar.value, dtype=np.float64)
    return np.clip(th, -C, C)


def recover_b_svr(K: np.ndarray, y: np.ndarray, theta: np.ndarray, C: float,
                  eps: float, tol: float = 1e-6) -> float:
    """Recover bias b from SVR margin SVs (0 < |theta_i| < C, residual = +-eps).

    h_i = f(x_i) - y_i = (K theta)_i + b - y_i = -eps sign(theta_i) on the margin
    => b = y_i - eps*sign(theta_i) - (K theta)_i.
    """
    Kt = K @ theta
    margin = (np.abs(theta) > tol) & (np.abs(theta) < C - tol)
    if np.any(margin):
        bs = y[margin] - eps * np.sign(theta[margin]) - Kt[margin]
        return float(np.mean(bs))
    order = np.argsort(np.abs(np.abs(theta) - C / 2.0))
    i = order[0]
    return float(y[i] - eps * np.sign(theta[i]) - Kt[i])

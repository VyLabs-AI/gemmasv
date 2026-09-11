"""Cauwenberghs-Poggio incremental/decremental epsilon-SVR.

Online support-vector regression via the same adiabatic active-set machinery,
using a single coefficient theta_i = alpha_i - alpha_i* in [-C, C] per point.
Cauwenberghs & Poggio note the procedure "can be directly extended ... including
SV regression"; this follows the accurate online SVR formulation (Ma, Theiler &
Perkins 2003).

Residual / margin function:
    h_i = f(x_i) - y_i,   f(x) = sum_j theta_j K(x_j, x) + b
Active-set partition:
    |h_i| < eps           -> reserve R, theta_i = 0       (inside the tube)
    |h_i| = eps, 0<|t|<C   -> margin  S                    (on the tube)
    |h_i| > eps           -> error  E, theta_i = -C sign(h_i)

Sensitivities (equality sum theta = 0, Q = K):
    M = [[0, 1_S^T], [1_S, K_SS]],   R = M^{-1}
    beta   = -R @ [1; K_{S,c}]                 (d[b; theta_S]/d theta_c)
    gamma_i = K_ic + [1, K_iS] @ beta          (d h_i / d theta_c), 0 on S
"""
from __future__ import annotations

import numpy as np

from .kernels import radial_kernel, linear_kernel

_KERNELS = {"r": radial_kernel, "radial": radial_kernel, "l": linear_kernel, "linear": linear_kernel}


class SVRIncremental:
    def __init__(self, C: float, eps: float, ktype: str = "r", kpar: float = 2.0,
                 tol: float = 1e-10, max_iter: int = 20000):
        self.C = float(C)
        self.eps = float(eps)
        self.ktype = ktype
        self.kpar = float(kpar)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.X = np.zeros((0, 0))
        self.y = np.zeros(0)
        self.theta = np.zeros(0)
        self.b = 0.0
        self.S: list[int] = []
        self.E: list[int] = []
        self.R: list[int] = []
        # Tube side of each point: +1 (residual at +eps) / -1 (at -eps) / 0 (n/a).
        # Tracked explicitly because a margin point can have theta == 0 (at the
        # S/R boundary), where sign(theta) cannot recover the side.
        self.side = np.zeros(0)
        self._K = np.zeros((0, 0))
        self._Rmat = None

    def _kfun(self, A, B):
        return _KERNELS[self.ktype](A, B, self.kpar)

    def _rebuild_K(self):
        self._K = self._kfun(self.X, self.X) if len(self.X) else np.zeros((0, 0))

    def _h_all(self) -> np.ndarray:
        if len(self.theta) == 0:
            return np.zeros(0)
        return self._K @ self.theta + self.b - self.y

    def _rebuild_R(self):
        m = len(self.S)
        if m == 0:
            self._Rmat = None
            return
        Kss = self._K[np.ix_(self.S, self.S)]
        M = np.zeros((m + 1, m + 1))
        M[0, 1:] = 1.0
        M[1:, 0] = 1.0
        M[1:, 1:] = Kss
        try:
            self._Rmat = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            self._Rmat = np.linalg.pinv(M)

    def _recompute_b(self):
        """Pin b from the margin condition h_i = side_i * eps, i in S."""
        if not self.S:
            return
        S = self.S
        Kt = self._K[S, :] @ self.theta
        target_h = self.side[S] * self.eps          # residual on the tube
        # h_i = Kt_i + b - y_i = target_h  ->  b = target_h + y_i - Kt_i
        self.b = float(np.mean(target_h + self.y[S] - Kt))

    def _sensitivities(self, c: int):
        n = len(self.theta)
        if len(self.S) == 0:
            return 0.0, np.zeros(0), np.ones(n)
        Ksc = self._K[self.S, c]
        beta = -self._Rmat @ np.concatenate([[1.0], Ksc])
        beta0, beta_S = beta[0], beta[1:]
        gamma = self._K[:, c].copy() + beta0 + self._K[:, self.S] @ beta_S
        gamma[self.S] = 0.0
        return beta0, beta_S, gamma

    def _move(self, idx, src, dst):
        src.remove(idx)
        dst.append(idx)

    def _apply_step(self, c, step, beta0, beta_S, s_empty):
        if s_empty:
            self.b += step
        else:
            self.theta[c] += step
            if self.S:
                self.theta[self.S] += step * beta_S
                self.b += step * beta0

    def _select(self, cands, direction, just_moved_idx):
        """Pick nearest event whose signed step matches `direction` (+1/-1)."""
        best = None
        for cand in cands:
            step, sit, idx = cand[0], cand[1], cand[2]
            if not np.isfinite(step):
                continue
            if idx == just_moved_idx and abs(step) <= 1e3 * self.tol:
                continue  # anti-cycling: block zero-length reverse of last move
            if direction > 0 and step <= self.tol:
                continue
            if direction < 0 and step >= -self.tol:
                continue
            if best is None or abs(step) < abs(best[0]):
                best = cand
        return best

    # ------------------------------------------------------------- seeding
    def seed_from_qp(self, X, y):
        from .oneclass_qp import solve_svr_qp, recover_b_svr

        self.X = np.atleast_2d(np.asarray(X, dtype=np.float64)).copy()
        self.y = np.asarray(y, dtype=np.float64).ravel().copy()
        self._rebuild_K()
        self.theta = solve_svr_qp(self._K, self.y, self.C, self.eps)
        self.side = np.zeros(len(self.theta))
        self.b = recover_b_svr(self._K, self.y, self.theta, self.C, self.eps,
                               tol=max(self.tol, 1e-7))
        self._classify_from_state(tol=max(self.tol, 1e-7))
        self._rebuild_R()
        return self

    def _classify_from_state(self, tol):
        h = self._h_all()
        self.S, self.E, self.R = [], [], []
        self.side = np.zeros(len(self.theta))
        for i in range(len(self.theta)):
            if abs(self.theta[i]) <= tol:
                self.R.append(i)
            elif abs(self.theta[i]) >= self.C - tol:
                self.E.append(i)
            else:
                self.S.append(i)
                self.side[i] = np.sign(h[i]) if abs(h[i]) > 0 else -np.sign(self.theta[i])

    # ------------------------------------------------------------- increment
    def add_point(self, x_new, y_new) -> int:
        x_new = np.atleast_2d(np.asarray(x_new, dtype=np.float64))
        self.X = x_new.copy() if len(self.X) == 0 else np.vstack([self.X, x_new])
        self.y = np.append(self.y, float(y_new))
        self.theta = np.append(self.theta, 0.0)
        self.side = np.append(self.side, 0.0)
        c = len(self.theta) - 1
        self._rebuild_K()
        self._rebuild_R()

        h = self._h_all()
        if abs(h[c]) <= self.eps:          # already inside the tube -> reserve
            self.R.append(c)
            return c

        direction = -np.sign(h[c])         # move theta_c to pull h_c toward the tube
        just_moved = None
        for _ in range(self.max_iter):
            s_empty = len(self.S) == 0
            beta0, beta_S, gamma = self._sensitivities(c)
            h = self._h_all()
            cands = self._candidate_steps(c, beta_S, gamma, h, s_empty, allow_c=True)
            best = self._select(cands, direction=direction, just_moved_idx=just_moved)
            if best is None:
                raise RuntimeError("SVR increment: no feasible adiabatic step")
            step, sit, idx, side = best
            self._apply_step(c, step, beta0, beta_S, s_empty)

            done = self._handle(sit, idx, c, side)
            self._recompute_b()
            if done:
                return c
            just_moved = idx
        raise RuntimeError("SVR increment: exceeded max_iter (possible cycling)")

    def _handle(self, sit, idx, c, side) -> bool:
        """Apply a migration; return True if the swept point c is now placed."""
        if sit == 0:        # theta_c hit +-C -> c to E
            self.theta[c] = np.clip(self.theta[c], -self.C, self.C)
            self.E.append(c)
            return True
        elif sit == 1:      # h_c hit +-eps -> c to S
            self.side[c] = side
            self.S.append(c)
            self._rebuild_R()
            return True
        elif sit == 7:      # theta_c hit 0 (decrement terminal)
            self.theta[c] = 0.0
            return True
        elif sit in (2, 3):  # margin theta_j hit +-C -> S to E
            self.theta[idx] = self.C if sit == 2 else -self.C
            self.side[idx] = 0.0
            self._move(idx, self.S, self.E)
            self._rebuild_R()
        elif sit == 4:       # margin theta_j hit 0 -> S to R
            self.theta[idx] = 0.0
            self.side[idx] = 0.0
            self._move(idx, self.S, self.R)
            self._rebuild_R()
        elif sit == 5:       # error h_j hit +-eps -> E to S
            self.side[idx] = side
            self._move(idx, self.E, self.S)
            self._rebuild_R()
        elif sit == 6:       # reserve h_j hit +-eps -> R to S
            self.side[idx] = side
            self._move(idx, self.R, self.S)
            self._rebuild_R()
        return False

    def _candidate_steps(self, c, beta_S, gamma, h, s_empty, allow_c):
        cands = []
        if s_empty:
            if allow_c and abs(gamma[c]) > 0:
                s = np.sign(h[c])
                cands.append(((s * self.eps - h[c]) / gamma[c], 1, c, s))
            for j in self.E:
                if abs(gamma[j]) > 0:
                    s = np.sign(h[j])
                    cands.append(((s * self.eps - h[j]) / gamma[j], 5, j, s))
            for j in self.R:
                if abs(gamma[j]) > 0:
                    for s in (1.0, -1.0):
                        cands.append(((s * self.eps - h[j]) / gamma[j], 6, j, s))
            return cands

        if allow_c:
            cands.append((self.C - self.theta[c], 0, c, 0.0))
            cands.append((-self.C - self.theta[c], 0, c, 0.0))
            if abs(gamma[c]) > 0:
                s = np.sign(h[c])
                cands.append(((s * self.eps - h[c]) / gamma[c], 1, c, s))
        for k, j in enumerate(self.S):
            bj = beta_S[k]
            if abs(bj) > 0:
                cands.append(((self.C - self.theta[j]) / bj, 2, j, 0.0))
                cands.append(((-self.C - self.theta[j]) / bj, 3, j, 0.0))
                cands.append(((0.0 - self.theta[j]) / bj, 4, j, 0.0))
        for j in self.E:
            if abs(gamma[j]) > 0:
                s = np.sign(h[j])
                cands.append(((s * self.eps - h[j]) / gamma[j], 5, j, s))
        for j in self.R:
            if abs(gamma[j]) > 0:
                for s in (1.0, -1.0):
                    cands.append(((s * self.eps - h[j]) / gamma[j], 6, j, s))
        return cands

    # ------------------------------------------------------------- decrement
    def remove_point(self, c: int):
        if c in self.R:
            self._delete_index(c)
            return
        if c in self.S:
            self.S.remove(c)
        elif c in self.E:
            self.E.remove(c)
        self._rebuild_R()

        just_moved = None
        for _ in range(self.max_iter):
            if abs(self.theta[c]) <= self.tol:
                break
            if len(self.S) == 0:
                raise RuntimeError("SVR decrement: margin set emptied (not handled)")
            direction = -np.sign(self.theta[c])   # drive theta_c toward 0
            beta0, beta_S, gamma = self._sensitivities(c)
            h = self._h_all()
            cands = self._candidate_steps(c, beta_S, gamma, h, s_empty=False, allow_c=False)
            cands.append((-self.theta[c], 7, c, 0.0))   # theta_c -> 0 (terminal)
            best = self._select(cands, direction=direction, just_moved_idx=just_moved)
            if best is None:
                raise RuntimeError("SVR decrement: no feasible adiabatic step")
            step, sit, idx, side = best
            if sit == 7 or abs(self.theta[c] + step) < self.tol:
                step, sit, idx, side = -self.theta[c], 7, c, 0.0
            self._apply_step(c, step, beta0, beta_S, s_empty=False)
            done = self._handle(sit, idx, c, side)
            self._recompute_b()
            if done and idx == c:
                break
            just_moved = idx

        self.theta[c] = 0.0
        self._delete_index(c)

    def _delete_index(self, c: int):
        keep = [i for i in range(len(self.theta)) if i != c]
        self.X = self.X[keep]
        self.y = self.y[keep]
        self.theta = self.theta[keep]
        self.side = self.side[keep]
        remap = {old: new for new, old in enumerate(keep)}
        self.S = [remap[i] for i in self.S if i != c]
        self.E = [remap[i] for i in self.E if i != c]
        self.R = [remap[i] for i in self.R if i != c]
        self._rebuild_K()
        self._rebuild_R()

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        active = self.S + self.E
        if not active:
            return np.full(len(X), self.b)
        Kx = self._kfun(X, self.X[active])
        return Kx @ self.theta[active] + self.b

"""Cauwenberghs-Poggio incremental/decremental binary C-SVM.

Two-class version of the same adiabatic active-set machinery as the one-class
solver. This is the head used by the clinical track (Paper B): a binary risk
classifier on top of a deep feature extractor, supporting exact online updates
and exact decremental unlearning of individual encounters/patients.

Dual problem:
    minimize    0.5 alpha^T (y y^T o K) alpha - 1^T alpha
    subject to  y^T alpha = 0,  0 <= alpha_i <= C

KKT gradient (Cauwenberghs & Poggio 2000):
    g_i = sum_j Q_ij alpha_j + y_i b - 1,   Q_ij = y_i y_j K_ij
    g_i = 0, 0 < alpha_i < C   -> margin S
    g_i < 0, alpha_i = C        -> error E
    g_i > 0, alpha_i = 0        -> reserve R

Bordered margin matrix:
    M = [[0, y_S^T], [y_S, Q_SS]],   R = M^{-1}
    beta   = -R @ [y_c; Q_{S,c}]            (sensitivity of [b; alpha_S] to alpha_c)
    gamma_i = Q_ic + [y_i, Q_iS] @ beta     (sensitivity of g_i to alpha_c)

As in the one-class reference, R is recomputed by direct inversion when S
changes (rank-1 maintenance is a later, separately verified step).
"""
from __future__ import annotations

import numpy as np

from .kernels import radial_kernel, linear_kernel

_KERNELS = {"r": radial_kernel, "radial": radial_kernel, "l": linear_kernel, "linear": linear_kernel}


class BinaryIncrementalSVM:
    def __init__(self, C: float, ktype: str = "r", kpar: float = 2.0,
                 tol: float = 1e-10, max_iter: int = 10000, class_weight=None):
        self.C = float(C)
        self.class_weight = class_weight   # None | "balanced" | {1: w+, -1: w-}
        self._cpos = float(C)              # per-class upper bounds (resolved at seed)
        self._cneg = float(C)
        self.ktype = ktype
        self.kpar = float(kpar)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.X = np.zeros((0, 0))
        self.y = np.zeros(0)
        self.alpha = np.zeros(0)
        self.Cvec = np.zeros(0)            # per-point box upper bound C_i
        self.b = 0.0
        self.S: list[int] = []
        self.E: list[int] = []
        self.R: list[int] = []
        self._K = np.zeros((0, 0))
        self._Q = np.zeros((0, 0))
        self._Rmat = None

    # ---------------------------------------------------------------- kernels
    def _kfun(self, A, B):
        return _KERNELS[self.ktype](A, B, self.kpar)

    def _resolve_class_weights(self, y):
        """Set per-class upper bounds from class_weight (called at seed time)."""
        if self.class_weight is None:
            self._cpos = self._cneg = self.C
        elif self.class_weight == "balanced":
            n = len(y)
            npos = max(int(np.sum(y > 0)), 1)
            nneg = max(int(np.sum(y < 0)), 1)
            self._cpos = self.C * n / (2.0 * npos)
            self._cneg = self.C * n / (2.0 * nneg)
        elif isinstance(self.class_weight, dict):
            self._cpos = self.C * float(self.class_weight.get(1, 1.0))
            self._cneg = self.C * float(self.class_weight.get(-1, 1.0))

    def _C_for(self, yval) -> float:
        return self._cpos if yval > 0 else self._cneg

    def _rebuild_kernels(self):
        if len(self.X) == 0:
            self._K = np.zeros((0, 0))
            self._Q = np.zeros((0, 0))
        else:
            self._K = self._kfun(self.X, self.X)
            self._Q = np.outer(self.y, self.y) * self._K

    def _append_last_kernel(self):
        """Incrementally extend _K/_Q with the just-appended last point (the new
        point is self.X[-1]). O(n d) instead of the O(n^2 d) full rebuild."""
        n = len(self.X)
        kc = self._kfun(self.X, self.X[n - 1:n]).ravel()      # (n,) kernel col
        newK = np.empty((n, n))
        if n > 1:
            newK[: n - 1, : n - 1] = self._K
        newK[n - 1, :] = kc
        newK[:, n - 1] = kc
        self._K = newK
        qc = (self.y * self.y[n - 1]) * kc
        newQ = np.empty((n, n))
        if n > 1:
            newQ[: n - 1, : n - 1] = self._Q
        newQ[n - 1, :] = qc
        newQ[:, n - 1] = qc
        self._Q = newQ

    def _drop_kernel(self, c: int):
        """Remove row/col c from _K/_Q in place."""
        self._K = np.delete(np.delete(self._K, c, axis=0), c, axis=1)
        self._Q = np.delete(np.delete(self._Q, c, axis=0), c, axis=1)

    # ------------------------------------------------------------------ state
    def _grad_all(self) -> np.ndarray:
        n = len(self.alpha)
        if n == 0:
            return np.zeros(0)
        return self._Q @ self.alpha + self.y * self.b - 1.0

    def _rebuild_R(self):
        m = len(self.S)
        if m == 0:
            self._Rmat = None
            return
        yS = self.y[self.S]
        Qss = self._Q[np.ix_(self.S, self.S)]
        M = np.zeros((m + 1, m + 1))
        M[0, 1:] = yS
        M[1:, 0] = yS
        M[1:, 1:] = Qss
        try:
            self._Rmat = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            self._Rmat = np.linalg.pinv(M)

    def _recompute_b(self):
        """Pin b from the margin KKT condition g_i = 0, i in S (exact, drift-free)."""
        if not self.S:
            return
        # g_i = (Q alpha)_i + y_i b - 1 = 0  ->  b = y_i (1 - (Q alpha)_i)
        Qa = self._Q[self.S, :] @ self.alpha
        self.b = float(np.mean(self.y[self.S] * (1.0 - Qa)))

    def _sensitivities(self, c: int):
        n = len(self.alpha)
        if len(self.S) == 0:
            # Only b is free; g_i moves at rate y_c y_i.
            return 0.0, np.zeros(0), self.y[c] * self.y
        Qsc = self._Q[self.S, c]
        rhs = np.concatenate([[self.y[c]], Qsc])
        beta = -self._Rmat @ rhs
        beta0, beta_S = beta[0], beta[1:]
        gamma = self._Q[:, c].copy()
        gamma += self.y * beta0 + self._Q[:, self.S] @ beta_S
        gamma[self.S] = 0.0
        return beta0, beta_S, gamma

    # --------------------------------------------------------------- migration
    def _move(self, idx, src, dst):
        src.remove(idx)
        dst.append(idx)

    def _apply_step(self, c, step, beta0, beta_S, s_empty):
        if s_empty:
            self.b += step
        else:
            self.alpha[c] += step
            if self.S:
                self.alpha[self.S] += step * beta_S
                self.b += step * beta0

    # Reverse migration pairs (a point that just did `key` must not immediately
    # undo it with a zero-length step -> prevents degenerate corner cycling).
    _REVERSE = {2: 4, 4: 2, 3: 5, 5: 3}

    def _select(self, cands, direction, just_moved):
        best = None
        for step, sit, idx in cands:
            if not np.isfinite(step):
                continue
            # Anti-cycling: block the zero-length reverse of the last migration.
            if just_moved is not None:
                jm_idx, jm_sit = just_moved
                if idx == jm_idx and sit == self._REVERSE.get(jm_sit) and abs(step) <= 1e3 * self.tol:
                    continue
            if direction > 0 and step <= self.tol:
                continue
            if direction < 0 and step >= -self.tol:
                continue
            if best is None or abs(step) < abs(best[0]):
                best = (step, sit, idx)
        return best

    def _partition(self, tol):
        """Per-point active-set partition: R (alpha~0), E (alpha~C_i), S (free)."""
        S, E, R = [], [], []
        for i in range(len(self.alpha)):
            if self.alpha[i] <= tol:
                R.append(i)
            elif self.alpha[i] >= self.Cvec[i] - tol:
                E.append(i)
            else:
                S.append(i)
        return S, E, R

    # ------------------------------------------------------------- seeding
    def seed_from_qp(self, X, y):
        from .oneclass_qp import solve_svm_qp

        self.X = np.atleast_2d(np.asarray(X, dtype=np.float64)).copy()
        self.y = np.asarray(y, dtype=np.float64).ravel().copy()
        self._resolve_class_weights(self.y)
        self.Cvec = np.where(self.y > 0, self._cpos, self._cneg)
        self._rebuild_kernels()
        self.alpha = solve_svm_qp(self._K, self.y, self.Cvec)
        self.S, self.E, self.R = self._partition(tol=max(self.tol, 1e-7))
        self._rebuild_R()
        self._recompute_b()
        return self

    # ------------------------------------------------------------- increment
    def add_point(self, x_new, y_new) -> int:
        x_new = np.atleast_2d(np.asarray(x_new, dtype=np.float64))
        if len(self.X) == 0:
            self.X = x_new.copy()
        else:
            self.X = np.vstack([self.X, x_new])
        self.y = np.append(self.y, float(y_new))
        self.alpha = np.append(self.alpha, 0.0)
        self.Cvec = np.append(self.Cvec, self._C_for(float(y_new)))
        c = len(self.alpha) - 1
        self._append_last_kernel()
        self._rebuild_R()

        grad = self._grad_all()
        if grad[c] > 0:                       # correctly classified -> reserve
            self.R.append(c)
            return c

        just_moved = None
        for _ in range(self.max_iter):
            s_empty = len(self.S) == 0
            beta0, beta_S, gamma = self._sensitivities(c)
            grad = self._grad_all()
            cands = self._candidate_steps(c, beta_S, gamma, grad, s_empty)
            best = self._select(cands, direction=+1, just_moved=just_moved)
            if best is None:
                raise RuntimeError("increment: no feasible adiabatic step")
            step, sit, idx = best
            self._apply_step(c, step, beta0, beta_S, s_empty)

            if sit == 0:
                self.alpha[c] = self.Cvec[c]
                self.E.append(c)
                self._recompute_b()
                return c
            elif sit == 1:
                self.alpha[c] = max(self.alpha[c], 0.0)
                self.S.append(c)
                self._rebuild_R()
                self._recompute_b()
                return c
            elif sit == 2:
                self.alpha[idx] = self.Cvec[idx]
                self._move(idx, self.S, self.E)
                self._rebuild_R()
            elif sit == 3:
                self.alpha[idx] = 0.0
                self._move(idx, self.S, self.R)
                self._rebuild_R()
            elif sit == 4:
                self._move(idx, self.E, self.S)
                self._rebuild_R()
            elif sit == 5:
                self._move(idx, self.R, self.S)
                self._rebuild_R()
            just_moved = (idx, sit)
            self._recompute_b()
        raise RuntimeError("increment: exceeded max_iter (possible cycling)")

    def _candidate_steps(self, c, beta_S, gamma, grad, s_empty):
        cands = []
        if s_empty:
            cands.append((-grad[c] / gamma[c], 1, c) if abs(gamma[c]) > 0 else (np.inf, 1, c))
            for j in self.E:
                if abs(gamma[j]) > 0:
                    cands.append((-grad[j] / gamma[j], 4, j))
            for j in self.R:
                if abs(gamma[j]) > 0:
                    cands.append((-grad[j] / gamma[j], 5, j))
            return cands
        cands.append((self.Cvec[c] - self.alpha[c], 0, c))
        if abs(gamma[c]) > 0:
            cands.append((-grad[c] / gamma[c], 1, c))
        for k, j in enumerate(self.S):
            b = beta_S[k]
            if abs(b) > 0:
                cands.append(((self.Cvec[j] - self.alpha[j]) / b, 2, j))
                cands.append((-self.alpha[j] / b, 3, j))
        for j in self.E:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 4, j))
        for j in self.R:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 5, j))
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
            if self.alpha[c] <= self.tol:
                break
            if len(self.S) == 0:
                raise RuntimeError("decrement: margin set emptied (S-empty path not yet handled)")
            beta0, beta_S, gamma = self._sensitivities(c)
            grad = self._grad_all()
            cands = self._candidate_steps_dec(c, beta_S, gamma, grad)
            best = self._select(cands, direction=-1, just_moved=just_moved)
            if best is None:
                raise RuntimeError("decrement: no feasible adiabatic step")
            step, sit, idx = best
            if -step > self.alpha[c]:
                step, sit, idx = -self.alpha[c], 6, c
            self._apply_step(c, step, beta0, beta_S, s_empty=False)

            if sit == 6:
                self.alpha[c] = 0.0
                self._recompute_b()
                break
            elif sit == 2:
                self.alpha[idx] = self.Cvec[idx]
                self._move(idx, self.S, self.E)
                self._rebuild_R()
            elif sit == 3:
                self.alpha[idx] = 0.0
                self._move(idx, self.S, self.R)
                self._rebuild_R()
            elif sit == 4:
                self._move(idx, self.E, self.S)
                self._rebuild_R()
            elif sit == 5:
                self._move(idx, self.R, self.S)
                self._rebuild_R()
            just_moved = (idx, sit)
            self._recompute_b()

        self.alpha[c] = 0.0
        self._delete_index(c)

    def _candidate_steps_dec(self, c, beta_S, gamma, grad):
        cands = [(-self.alpha[c], 6, c)]
        for k, j in enumerate(self.S):
            b = beta_S[k]
            if abs(b) > 0:
                cands.append(((self.Cvec[j] - self.alpha[j]) / b, 2, j))
                cands.append((-self.alpha[j] / b, 3, j))
        for j in self.E:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 4, j))
        for j in self.R:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 5, j))
        return cands

    def _delete_index(self, c: int):
        keep = [i for i in range(len(self.alpha)) if i != c]
        self.X = self.X[keep]
        self.y = self.y[keep]
        self.alpha = self.alpha[keep]
        self.Cvec = self.Cvec[keep]
        remap = {old: new for new, old in enumerate(keep)}
        self.S = [remap[i] for i in self.S if i != c]
        self.E = [remap[i] for i in self.E if i != c]
        self.R = [remap[i] for i in self.R if i != c]
        self._drop_kernel(c)
        self._rebuild_R()

    # ------------------------------------------------------------- utilities
    def decision_function(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        active = self.S + self.E
        if not active:
            return np.full(len(X), self.b)
        Kx = self._kfun(X, self.X[active])
        ay = self.alpha[active] * self.y[active]
        return Kx @ ay + self.b

    def predict(self, X) -> np.ndarray:
        return np.sign(self.decision_function(X))

"""Cauwenberghs--Poggio incremental/decremental one-class (nu-SVDD) SVM.

Python translation and one-class adaptation of Gert Cauwenberghs's
public-domain incremental-SVM MATLAB release. The MATLAB files are not
distributed; see THIRD_PARTY_NOTICES.md for source and licensing details.

This reference implementation maintains the active-set partition
(margin S / error E / reserve R) and performs the adiabatic increment and
reverse decrement while maintaining the KKT conditions along completed paths.

Design note on R:
    The original code maintains the inverse bordered margin matrix R via rank-1
    Sherman-Morrison updates (change_R). Here we instead recompute R by direct
    inversion whenever S changes. This is O(|S|^3) rather than O(|S|^2), but is
    far less error-prone and numerically cleaner for verification. The rank-1
    maintained R (needed for the "free implicit gradient" claim) is a later step,
    verified against this reference.

Conventions (one-class, all labels y_i == 1):
    Q_ij  = 2 * K_ij
    grad_i = rho - K_ii + 2 * sum_{j in S u E} K_ij alpha_j
    bordered margin matrix  M = [[0, 1_S^T], [1_S, 2 K_SS]],  R = M^{-1}
    beta   = -R @ [1; 2 K_{S,c}]      (sensitivity of [rho; alpha_S] to alpha_c)
    gamma_i = 2 K_ic + [1, 2 K_iS] @ beta   (sensitivity of grad_i to alpha_c)
"""
from __future__ import annotations

import numpy as np

from .kernels import radial_kernel, linear_kernel

_KERNELS = {"r": radial_kernel, "radial": radial_kernel, "l": linear_kernel, "linear": linear_kernel}


class OneClassIncrementalSVM:
    def __init__(self, C: float, ktype: str = "r", kpar: float = 2.0,
                 tol: float = 1e-10, max_iter: int = 10000):
        self.C = float(C)
        self.ktype = ktype
        self.kpar = float(kpar)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.X = np.zeros((0, 0))
        self.alpha = np.zeros(0)
        self.rho = 0.0
        self.S: list[int] = []
        self.E: list[int] = []
        self.R: list[int] = []
        self._K = np.zeros((0, 0))     # gram of current members
        self._Rmat = None              # inverse bordered margin matrix

    # ---------------------------------------------------------------- kernels
    def _kfun(self, A, B):
        return _KERNELS[self.ktype](A, B, self.kpar)

    def _rebuild_K(self):
        if len(self.X) == 0:
            self._K = np.zeros((0, 0))
        else:
            self._K = self._kfun(self.X, self.X)

    # ------------------------------------------------------------------ state
    def _grad_all(self) -> np.ndarray:
        """Closed-form gradient for all current members.

        Uses the full alpha vector (reserve alphas are 0, so this equals the
        S u E sum) which keeps the gradient correct even for a point that is
        mid-removal and not currently labelled in any active set.
        """
        n = len(self.alpha)
        if n == 0:
            return np.zeros(0)
        return self.rho - np.diag(self._K) + 2.0 * (self._K @ self.alpha)

    def _rebuild_R(self):
        """Recompute R = inv([[0, 1_S^T],[1_S, 2 K_SS]])."""
        m = len(self.S)
        if m == 0:
            self._Rmat = None
            return
        Kss = self._K[np.ix_(self.S, self.S)]
        M = np.zeros((m + 1, m + 1))
        M[0, 1:] = 1.0
        M[1:, 0] = 1.0
        M[1:, 1:] = 2.0 * Kss
        try:
            self._Rmat = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            self._Rmat = np.linalg.pinv(M)

    def _recompute_rho(self):
        """Pin rho from the margin KKT condition grad_i = 0, i in S.

        At every adiabatic breakpoint the iterate is a valid KKT point, so
        rho = K_ii - 2 (K alpha)_i for any margin point i. This is exact (it is
        the *definition* of rho once S is fixed) and avoids accumulating
        floating-point drift in the maintained bias across set migrations.
        """
        if not self.S:
            return
        Ka = self._K[self.S, :] @ self.alpha
        self.rho = float(np.mean(np.diag(self._K)[self.S] - 2.0 * Ka))

    def _sensitivities(self, c: int):
        """Return (beta0, beta_S, gamma) sensitivities w.r.t. alpha_c.

        beta0    : d(rho)/d(alpha_c)
        beta_S   : d(alpha_S)/d(alpha_c)   (array aligned with self.S)
        gamma    : d(grad_i)/d(alpha_c)    (length-n array; 0 on S)
        """
        n = len(self.alpha)
        if len(self.S) == 0:
            # Only rho is free; grad moves uniformly at rate 1 (= y_c y_i).
            return 0.0, np.zeros(0), np.ones(n)

        Ksc = self._K[self.S, c]                       # K_{S,c}
        rhs = np.concatenate([[1.0], 2.0 * Ksc])       # [y_c; Q_{S,c}]
        beta = -self._Rmat @ rhs
        beta0 = beta[0]
        beta_S = beta[1:]

        gamma = 2.0 * self._K[:, c].copy()             # Q_{i,c}
        # gamma_i += [y_i, Q_{i,S}] @ beta
        KiS = self._K[:, self.S]                        # (n, |S|)
        gamma += beta0 + 2.0 * (KiS @ beta_S)
        gamma[self.S] = 0.0
        return beta0, beta_S, gamma

    # --------------------------------------------------------------- migration
    def _move(self, idx: int, src: list[int], dst: list[int]):
        src.remove(idx)
        dst.append(idx)

    # ----------------------------------------------------------------- events
    def _candidate_steps(self, c: int, beta0, beta_S, gamma, grad, s_empty: bool):
        """Yield (signed_step_in_swept_var, situation, index) candidates.

        situation codes:
            0 'ue'  alpha_c -> C      (c into E)        [terminal]
            1 'um'  grad_c  -> 0      (c into S)        [terminal]
            2 'me'  alpha_j -> C      (j: S -> E)
            3 'mr'  alpha_j -> 0      (j: S -> R)
            4 'em'  grad_j  -> 0      (j: E -> S)
            5 'rm'  grad_j  -> 0      (j: R -> S)
        """
        cands = []
        if s_empty:
            # swept variable is rho; grad_i rate = 1
            cands.append((-grad[c], 1, c))            # grad_c -> 0
            for j in self.E:
                cands.append((-grad[j], 4, j))        # grad_j -> 0
            return cands

        # swept variable is alpha_c
        cands.append((self.C - self.alpha[c], 0, c))  # alpha_c -> C
        if abs(gamma[c]) > 0:
            cands.append((-grad[c] / gamma[c], 1, c)) # grad_c -> 0
        for k, j in enumerate(self.S):
            b = beta_S[k]
            if abs(b) > 0:
                cands.append(((self.C - self.alpha[j]) / b, 2, j))  # alpha_j -> C
                cands.append((-self.alpha[j] / b, 3, j))            # alpha_j -> 0
        for j in self.E:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 4, j))           # grad_j -> 0
        for j in self.R:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 5, j))           # grad_j -> 0
        return cands

    def _select(self, cands, direction: int, last_sv_zeroed):
        """Pick the nearest reachable event in the given direction (+1 inc, -1 dec)."""
        best = None
        for step, sit, idx in cands:
            if not np.isfinite(step):
                continue
            # anti-cycling guard (Compute_dp.m 1*/2*): a reserve point that just
            # became 0 must not immediately be reconsidered to re-enter S.
            if sit == 5 and idx == last_sv_zeroed:
                continue
            if direction > 0 and step <= self.tol:
                continue
            if direction < 0 and step >= -self.tol:
                continue
            if best is None or abs(step) < abs(best[0]):
                best = (step, sit, idx)
        return best

    # ----------------------------------------------------------------- apply
    def _apply_step(self, c: int, step: float, beta0, beta_S, s_empty: bool):
        if s_empty:
            self.rho += step
        else:
            self.alpha[c] += step
            if len(self.S) > 0:
                self.alpha[self.S] += step * beta_S
                self.rho += step * beta0

    # ------------------------------------------------------------- seeding
    def seed_from_qp(self, X: np.ndarray):
        """Initialize a valid KKT state on X by solving the batch QP.

        Used to bootstrap the incremental solver past the cold-start region
        (a feasible sum(alpha)=1, 0<=alpha<=C state needs >= ceil(1/C) points).
        """
        from .oneclass_qp import solve_svdd_qp, recover_rho, partition_sets

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        self.X = X.copy()
        self._rebuild_K()
        alpha = solve_svdd_qp(self._K, self.C)
        self.alpha = alpha
        self.rho = recover_rho(self._K, alpha, self.C, tol=max(self.tol, 1e-7))
        S, E, R = partition_sets(alpha, self.C, tol=max(self.tol, 1e-7))
        self.S, self.E, self.R = S, E, R
        self._rebuild_R()
        return self

    # ------------------------------------------------------------- increment
    def add_point(self, x_new: np.ndarray) -> int:
        """Add one point via the adiabatic increment. Returns its index."""
        x_new = np.atleast_2d(np.asarray(x_new, dtype=np.float64))
        if len(self.X) == 0:
            self.X = x_new.copy()
        else:
            self.X = np.vstack([self.X, x_new])
        self.alpha = np.append(self.alpha, 0.0)
        c = len(self.alpha) - 1
        self._rebuild_K()
        self._rebuild_R()

        grad = self._grad_all()
        # If correctly classified already (grad_c > 0) -> reserve, done (case 'ur').
        if grad[c] > 0:
            self.R.append(c)
            return c

        last_sv_zeroed = None
        for _ in range(self.max_iter):
            s_empty = len(self.S) == 0
            beta0, beta_S, gamma = self._sensitivities(c)
            grad = self._grad_all()
            cands = self._candidate_steps(c, beta0, beta_S, gamma, grad, s_empty)
            best = self._select(cands, direction=+1, last_sv_zeroed=last_sv_zeroed)
            if best is None:
                raise RuntimeError("increment: no feasible adiabatic step (infeasible)")
            step, sit, idx = best
            self._apply_step(c, step, beta0, beta_S, s_empty)
            last_sv_zeroed = None

            if sit == 0:        # c -> E
                self.alpha[c] = self.C
                self.E.append(c)
                self._recompute_rho()
                return c
            elif sit == 1:      # c -> S
                self.alpha[c] = max(self.alpha[c], 0.0)
                self.S.append(c)
                self._rebuild_R()
                self._recompute_rho()
                return c
            elif sit == 2:      # j: S -> E
                self.alpha[idx] = self.C
                self._move(idx, self.S, self.E)
                self._rebuild_R()
            elif sit == 3:      # j: S -> R
                self.alpha[idx] = 0.0
                self._move(idx, self.S, self.R)
                last_sv_zeroed = idx
                self._rebuild_R()
            elif sit == 4:      # j: E -> S
                self._move(idx, self.E, self.S)
                self._rebuild_R()
            elif sit == 5:      # j: R -> S
                self._move(idx, self.R, self.S)
                self._rebuild_R()
            self._recompute_rho()
        raise RuntimeError("increment: exceeded max_iter (possible cycling)")

    # ------------------------------------------------------------- decrement
    def remove_point(self, c: int):
        """Unlearn point c by driving alpha_c -> 0 along the reverse path, then delete it.

        c is detached from its active set and treated as the free swept variable
        (mirror of the increment, where the new point plays this role). alpha_c is
        driven monotonically to 0 while the remaining margin points absorb the mass
        (sum of beta_S = -1 keeps sum(alpha) = 1), with set migrations as needed.
        """
        if c in self.R:
            self._delete_index(c)
            return
        if c in self.S:
            self.S.remove(c)
        elif c in self.E:
            self.E.remove(c)
        self._rebuild_R()

        last_sv_zeroed = None
        for _ in range(self.max_iter):
            if self.alpha[c] <= self.tol:
                break
            if len(self.S) == 0:
                raise RuntimeError("decrement: margin set emptied (S-empty path not yet handled)")
            beta0, beta_S, gamma = self._sensitivities(c)
            grad = self._grad_all()
            cands = self._candidate_steps_dec(c, beta_S, gamma, grad)
            best = self._select(cands, direction=-1, last_sv_zeroed=last_sv_zeroed)
            if best is None:
                raise RuntimeError("decrement: no feasible adiabatic step")
            step, sit, idx = best
            # Never overshoot below alpha_c = 0.
            if -step > self.alpha[c]:
                step, sit, idx = -self.alpha[c], 6, c
            self._apply_step(c, step, beta0, beta_S, s_empty=False)
            last_sv_zeroed = None

            if sit == 6:        # alpha_c reached 0
                self.alpha[c] = 0.0
                self._recompute_rho()
                break
            elif sit == 2:      # j: S -> E
                self.alpha[idx] = self.C
                self._move(idx, self.S, self.E)
                self._rebuild_R()
            elif sit == 3:      # j: S -> R
                self.alpha[idx] = 0.0
                self._move(idx, self.S, self.R)
                last_sv_zeroed = idx
                self._rebuild_R()
            elif sit == 4:      # j: E -> S
                self._move(idx, self.E, self.S)
                self._rebuild_R()
            elif sit == 5:      # j: R -> S
                self._move(idx, self.R, self.S)
                self._rebuild_R()
            self._recompute_rho()

        self.alpha[c] = 0.0
        self._delete_index(c)

    def _candidate_steps_dec(self, c, beta_S, gamma, grad):
        """Candidate signed steps in alpha_c (negative) for the decrement sweep.

        Excludes c's own grad event (c is leaving); includes only migrations of
        the other points plus the terminal alpha_c -> 0.
        """
        cands = [(-self.alpha[c], 6, c)]                       # alpha_c -> 0
        for k, j in enumerate(self.S):
            b = beta_S[k]
            if abs(b) > 0:
                cands.append(((self.C - self.alpha[j]) / b, 2, j))
                cands.append((-self.alpha[j] / b, 3, j))
        for j in self.E:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 4, j))
        for j in self.R:
            if abs(gamma[j]) > 0:
                cands.append((-grad[j] / gamma[j], 5, j))
        return cands

    def _delete_index(self, c: int):
        """Remove member c from all structures and reindex."""
        keep = [i for i in range(len(self.alpha)) if i != c]
        self.X = self.X[keep]
        self.alpha = self.alpha[keep]
        remap = {old: new for new, old in enumerate(keep)}
        self.S = [remap[i] for i in self.S if i != c]
        self.E = [remap[i] for i in self.E if i != c]
        self.R = [remap[i] for i in self.R if i != c]
        self._rebuild_K()
        self._rebuild_R()

    # ------------------------------------------------------------- utilities
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """f(x) = 2 * sum_j alpha_j K(x_j, x) - rho ... (margin score)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        active = self.S + self.E
        if not active:
            return np.full(len(X), -self.rho)
        Kx = self._kfun(X, self.X[active])
        return 2.0 * (Kx @ self.alpha[active]) - self.rho

    def kkt_residual(self) -> dict:
        """Diagnostics: how well the current state satisfies the KKT conditions."""
        grad = self._grad_all()
        res = {
            "sum_alpha_err": float(abs(self.alpha.sum() - 1.0)) if len(self.alpha) else 0.0,
            "margin_grad_max": float(np.max(np.abs(grad[self.S]))) if self.S else 0.0,
            "error_grad_max": float(np.max(grad[self.E])) if self.E else float("-inf"),
            "reserve_grad_min": float(np.min(grad[self.R])) if self.R else float("inf"),
            "box_violation": float(max(
                (-self.alpha).max() if len(self.alpha) else 0.0,
                (self.alpha - self.C).max() if len(self.alpha) else 0.0,
                0.0,
            )),
        }
        return res

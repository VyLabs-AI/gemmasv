"""Maintained incremental/decremental one-class SVM (the SV-Attention gate).

Same fixed-C optimization target as `OneClassIncrementalSVM` (the verified
reference), but engineered for streams:

  - Gram matrix maintained incrementally: O(n d) per add, not O(n^2 d) rebuild.
  - Gradients maintained via the adiabatic sensitivities (grad += step * gamma),
    O(n) per event, instead of O(n * |active|) recomputation per iteration.
  - The inverse bordered KKT matrix R maintained by rank-one Sherman-Morrison
    expand/contract (the MATLAB `change_R`), O(|S|^2) per migration instead of
    O(|S|^3) re-inversion.
  - Periodic refactorization (every `refactor_every` events) recomputes R and
    the gradients from closed form, bounding floating-point drift; near-singular
    expansions trigger an immediate refactorization and are counted in
    `guard_events` rather than silently clamped.
  - Point-in-time reserve removal: current zero-coefficient points can be
    reindexed away without changing the solved function. Future admissions are
    outside that certificate.

Verified against the reference solver in tests/test_fast_solver.py.
"""
from __future__ import annotations

import numpy as np

from .kernels import radial_kernel, linear_kernel

_KERNELS = {"r": radial_kernel, "radial": radial_kernel, "l": linear_kernel, "linear": linear_kernel}
_SMALLEPS = 1e-12
# Relative threshold for a near-singular rank-1 pivot: smooth/redundant streams
# (e.g. forward-filled clinical vitals) produce near-duplicate margin points
# whose Schur-complement pivot is small RELATIVE to the matrix scale yet larger
# than the absolute floor. Catching these triggers an exact refactorization
# before the Sherman-Morrison update amplifies floating error.
_REL_EPS = 1e-9
# If a rank-1 update makes the inverse blow up, the pivot was effectively
# singular; refactor (exact inv/pinv) rather than trust the divergent update.
_RMAT_CAP = 1e9


class FastOneClassSVM:
    def __init__(self, C: float, ktype: str = "r", kpar: float = 2.0,
                 tol: float = 1e-10, max_iter: int = 10000, refactor_every: int = 500):
        self.C = float(C)
        self.ktype = ktype
        self.kpar = float(kpar)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.refactor_every = int(refactor_every)

        self.X = np.zeros((0, 0))
        self.alpha = np.zeros(0)
        self.rho = 0.0
        self.grad = np.zeros(0)
        self.S: list[int] = []     # ordered: aligned with rows 1.. of R
        self.E: list[int] = []
        self.R: list[int] = []
        self._K = np.zeros((0, 0))
        self._Rmat = None
        self._events_since_refactor = 0
        self.guard_events = 0
        self.refactor_count = 0

    # ---------------------------------------------------------------- kernels
    def _kfun(self, A, B):
        return _KERNELS[self.ktype](A, B, self.kpar)

    def _append_kernel(self):
        n = len(self.X)
        kc = self._kfun(self.X, self.X[n - 1:n]).ravel()
        newK = np.empty((n, n))
        if n > 1:
            newK[:n - 1, :n - 1] = self._K
        newK[n - 1, :] = kc
        newK[:, n - 1] = kc
        self._K = newK

    # ------------------------------------------------------------- closed form
    def _recompute_grad(self):
        """grad_i = rho - K_ii + 2 (K alpha)_i, exact closed form."""
        n = len(self.alpha)
        if n == 0:
            self.grad = np.zeros(0)
            return
        self.grad = self.rho - np.diag(self._K) + 2.0 * (self._K @ self.alpha)

    def _recompute_rho(self):
        if not self.S:
            return
        Ka = self._K[self.S, :] @ self.alpha
        self.rho = float(np.mean(np.diag(self._K)[self.S] - 2.0 * Ka))

    def _refactor(self):
        """Recompute R (direct inversion) and grad from closed form."""
        m = len(self.S)
        if m == 0:
            self._Rmat = None
        else:
            M = np.zeros((m + 1, m + 1))
            M[0, 1:] = 1.0
            M[1:, 0] = 1.0
            M[1:, 1:] = 2.0 * self._K[np.ix_(self.S, self.S)]
            try:
                self._Rmat = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                self._Rmat = np.linalg.pinv(M)
        self._recompute_rho()
        self._recompute_grad()
        self._events_since_refactor = 0
        self.refactor_count += 1

    def _maybe_refactor(self):
        self._events_since_refactor += 1
        if self._events_since_refactor >= self.refactor_every:
            self._refactor()
        elif self._Rmat is not None and np.max(np.abs(self._Rmat)) > _RMAT_CAP:
            # inverse drifted toward singular during the adiabatic walk
            self.guard_events += 1
            self._refactor()

    # ------------------------------------------------------ rank-1 R updates
    def _R_add(self, j: int):
        """Point j joins S (appended at the end of the S ordering)."""
        if self._Rmat is None or len(self.S) == 0:
            # first margin point: direct 2x2 inverse of [[0,1],[1,2K_jj]]
            Qjj = 2.0 * self._K[j, j]
            self._Rmat = np.array([[-Qjj, 1.0], [1.0, 0.0]])
            self.S.append(j)
            return
        Qsj = 2.0 * self._K[self.S, j]
        rhs = np.concatenate([[1.0], Qsj])
        beta = -self._Rmat @ rhs
        gamma_j = 2.0 * self._K[j, j] + rhs @ beta
        # Near-singular pivot test, both absolute AND relative to the scale of
        # the cancelling terms (catastrophic cancellation on duplicate points).
        scale = abs(2.0 * self._K[j, j]) + abs(rhs @ beta)
        if abs(gamma_j) <= _SMALLEPS or abs(gamma_j) <= _REL_EPS * scale:
            self.guard_events += 1
            self.S.append(j)
            self._refactor()
            return
        m = self._Rmat.shape[0]
        Rn = np.zeros((m + 1, m + 1))
        Rn[:m, :m] = self._Rmat
        b1 = np.concatenate([beta, [1.0]])
        Rn += np.outer(b1, b1) / gamma_j
        self._Rmat = Rn
        self.S.append(j)
        if np.max(np.abs(Rn)) > _RMAT_CAP:      # update diverged: rebuild exactly
            self.guard_events += 1
            self._refactor()

    def _R_remove(self, pos: int):
        """Margin point at S-position `pos` (0-based) leaves S."""
        k = pos + 1                      # offset for the rho row
        R = self._Rmat
        if R.shape[0] <= 2:
            self._Rmat = None
            del self.S[pos]
            return
        scale = float(np.max(np.abs(R[k, :])))
        if abs(R[k, k]) <= _SMALLEPS or abs(R[k, k]) <= _REL_EPS * scale:
            self.guard_events += 1
            del self.S[pos]
            self._refactor()
            return
        idx = [i for i in range(R.shape[0]) if i != k]
        self._Rmat = R[np.ix_(idx, idx)] - np.outer(R[idx, k], R[k, idx]) / R[k, k]
        del self.S[pos]
        if self._Rmat.size and np.max(np.abs(self._Rmat)) > _RMAT_CAP:
            self.guard_events += 1
            self._refactor()

    # ---------------------------------------------------------- sensitivities
    def _sensitivities(self, c: int):
        n = len(self.alpha)
        if len(self.S) == 0:
            return 0.0, np.zeros(0), np.ones(n)
        Ksc = self._K[self.S, c]
        rhs = np.concatenate([[1.0], 2.0 * Ksc])
        beta = -self._Rmat @ rhs
        gamma = 2.0 * self._K[:, c] + beta[0] + 2.0 * (self._K[:, self.S] @ beta[1:])
        gamma[self.S] = 0.0
        return beta[0], beta[1:], gamma

    def _candidate_steps(self, c, beta_S, gamma, s_empty):
        g = self.grad
        cands = []
        if s_empty:
            cands.append((-g[c], 1, c))
            for j in self.E:
                cands.append((-g[j], 4, j))
            return cands
        cands.append((self.C - self.alpha[c], 0, c))
        if abs(gamma[c]) > 0:
            cands.append((-g[c] / gamma[c], 1, c))
        for k, j in enumerate(self.S):
            b = beta_S[k]
            if abs(b) > 0:
                cands.append(((self.C - self.alpha[j]) / b, 2, j))
                cands.append((-self.alpha[j] / b, 3, j))
        for j in self.E:
            if abs(gamma[j]) > 0:
                cands.append((-g[j] / gamma[j], 4, j))
        for j in self.R:
            if abs(gamma[j]) > 0:
                cands.append((-g[j] / gamma[j], 5, j))
        return cands

    def _candidate_steps_dec(self, c, beta_S, gamma):
        """Candidate negative steps for decrementing the leaving point.

        Unlike insertion, ``c`` is detached from its active set and must not
        generate its own gradient/set-entry events.  The reference solver has
        always used this reduced event set; reusing the insertion candidates
        here allowed a spurious ``c -> S`` event that the decrement loop did
        not handle, causing drift across multi-point deletions.
        """

        cands = [(-self.alpha[c], 6, c)]
        for k, j in enumerate(self.S):
            b = beta_S[k]
            if abs(b) > 0:
                cands.append(((self.C - self.alpha[j]) / b, 2, j))
                cands.append((-self.alpha[j] / b, 3, j))
        for j in self.E:
            if abs(gamma[j]) > 0:
                cands.append((-self.grad[j] / gamma[j], 4, j))
        for j in self.R:
            if abs(gamma[j]) > 0:
                cands.append((-self.grad[j] / gamma[j], 5, j))
        return cands

    def _select(self, cands, direction, just_moved):
        best = None
        for step, sit, idx in cands:
            if not np.isfinite(step):
                continue
            if just_moved is not None and idx == just_moved and abs(step) <= 1e3 * self.tol:
                continue
            if direction > 0 and step <= self.tol:
                continue
            if direction < 0 and step >= -self.tol:
                continue
            if best is None or abs(step) < abs(best[0]):
                best = (step, sit, idx)
        return best

    def _apply(self, c, step, beta0, beta_S, gamma, s_empty):
        if s_empty:
            self.rho += step
            self.grad = self.grad + step          # gamma == 1 vector
        else:
            self.alpha[c] += step
            if self.S:
                self.alpha[self.S] += step * beta_S
                self.rho += step * beta0
            self.grad = self.grad + step * gamma

    # ------------------------------------------------------------- seeding
    def seed_from_qp(self, X):
        from .oneclass_qp import solve_svdd_qp, partition_sets
        self.X = np.atleast_2d(np.asarray(X, dtype=np.float64)).copy()
        self._K = self._kfun(self.X, self.X)
        self.alpha = solve_svdd_qp(self._K, self.C)
        S, E, R = partition_sets(self.alpha, self.C, tol=max(self.tol, 1e-7))
        self.S, self.E, self.R = list(S), list(E), list(R)
        self._refactor()
        return self

    # ------------------------------------------------------------- increment
    def add_point(self, x_new) -> int:
        x_new = np.atleast_2d(np.asarray(x_new, dtype=np.float64))
        self.X = x_new.copy() if len(self.X) == 0 else np.vstack([self.X, x_new])
        self.alpha = np.append(self.alpha, 0.0)
        c = len(self.alpha) - 1
        self._append_kernel()
        # grad of the new point, closed form; existing grads unchanged by adding
        # a zero-coefficient point.
        active = self.S + self.E
        gc = self.rho - self._K[c, c] + 2.0 * (self._K[c, active] @ self.alpha[active]) \
            if active else self.rho - self._K[c, c]
        self.grad = np.append(self.grad, gc)

        if self.grad[c] > 0:
            self.R.append(c)
            return c

        just_moved = None
        for _ in range(self.max_iter):
            s_empty = len(self.S) == 0
            beta0, beta_S, gamma = self._sensitivities(c)
            cands = self._candidate_steps(c, beta_S, gamma, s_empty)
            best = self._select(cands, +1, just_moved)
            if best is None:
                raise RuntimeError("fast increment: no feasible adiabatic step")
            step, sit, idx = best
            self._apply(c, step, beta0, beta_S, gamma, s_empty)

            if sit == 0:          # c -> E
                self.alpha[c] = self.C
                self.E.append(c)
                self.grad[c] = min(self.grad[c], 0.0)
                self._maybe_refactor()
                return c
            elif sit == 1:        # c -> S
                self.alpha[c] = max(self.alpha[c], 0.0)
                self.grad[c] = 0.0
                self._R_add(c)
                self._maybe_refactor()
                return c
            elif sit == 2:        # j: S -> E
                self.alpha[idx] = self.C
                self._R_remove(self.S.index(idx))
                self.E.append(idx)
            elif sit == 3:        # j: S -> R
                self.alpha[idx] = 0.0
                self._R_remove(self.S.index(idx))
                self.R.append(idx)
            elif sit == 4:        # j: E -> S
                self.E.remove(idx)
                self.grad[idx] = 0.0
                self._R_add(idx)
            elif sit == 5:        # j: R -> S
                self.R.remove(idx)
                self.grad[idx] = 0.0
                self._R_add(idx)
            just_moved = idx
            self._maybe_refactor()
        raise RuntimeError("fast increment: exceeded max_iter")

    # ------------------------------------------------------------- decrement
    def remove_point(self, c: int):
        if c in self.R:
            self._delete([c])
            return
        if c in self.S:
            self._R_remove(self.S.index(c))
        elif c in self.E:
            self.E.remove(c)

        just_moved = None
        for _ in range(self.max_iter):
            if self.alpha[c] <= self.tol:
                break
            if len(self.S) == 0:
                raise RuntimeError("fast decrement: margin set emptied (unhandled)")
            beta0, beta_S, gamma = self._sensitivities(c)
            cands = self._candidate_steps_dec(c, beta_S, gamma)
            best = self._select(cands, -1, just_moved)
            if best is None:
                raise RuntimeError("fast decrement: no feasible adiabatic step")
            step, sit, idx = best
            if -step > self.alpha[c]:
                step, sit, idx = -self.alpha[c], 6, c
            self._apply(c, step, beta0, beta_S, gamma, s_empty=False)

            if sit == 6:
                self.alpha[c] = 0.0
                break
            elif sit == 2:
                self.alpha[idx] = self.C
                self._R_remove(self.S.index(idx))
                self.E.append(idx)
            elif sit == 3:
                self.alpha[idx] = 0.0
                self._R_remove(self.S.index(idx))
                self.R.append(idx)
            elif sit == 4:
                self.E.remove(idx)
                self.grad[idx] = 0.0
                self._R_add(idx)
            elif sit == 5:
                self.R.remove(idx)
                self.grad[idx] = 0.0
                self._R_add(idx)
            just_moved = idx
            self._maybe_refactor()

        self.alpha[c] = 0.0
        self._delete([c])

    def remove_points(self, indices) -> None:
        """Delete a block through one coupled fixed-C homotopy.

        Sequential one-point reverse paths are order-independent on
        well-conditioned unique optima, but large record blocks on real
        language-model keys are highly redundant and can land on a different
        non-unique vertex.  This path shrinks every forgotten coefficient
        together.  At each active-set event it solves the exact KKT
        sensitivity system for the retained margin set, then continues until
        the whole block reaches zero.

        The fixed box ``C`` is unchanged.  Failure is explicit; callers may
        still use the independently solved retained-key refit as a disclosed
        reference fallback.
        """

        forget = sorted(
            {
                int(index)
                for index in indices
                if 0 <= int(index) < len(self.alpha)
            }
        )
        if not forget:
            return
        forgotten = set(forget)
        # Reserve entries already have alpha=0, but keeping them in the block
        # until the final batch reindex preserves a stable original index map.
        self.S = [index for index in self.S if index not in forgotten]
        self.E = [index for index in self.E if index not in forgotten]
        self.R = [index for index in self.R if index not in forgotten]
        self._refactor()

        for _ in range(self.max_iter):
            active_forget = [
                index for index in forget if self.alpha[index] > self.tol
            ]
            if not active_forget:
                break
            if not self.S:
                raise RuntimeError(
                    "fast block decrement: margin set emptied (unhandled)"
                )

            # Reparameterize each segment so lambda=1 zeros all currently
            # nonzero forgotten coefficients.  The retained margin
            # coefficients absorb exactly the opposite total mass.
            d_forget = -self.alpha[active_forget].copy()
            margin = list(self.S)
            m = len(margin)
            rhs = np.empty(m + 1, dtype=np.float64)
            rhs[0] = -float(d_forget.sum())
            rhs[1:] = -2.0 * (
                self._K[np.ix_(margin, active_forget)] @ d_forget
            )
            # _Rmat is the maintained inverse/pseudoinverse of this exact KKT
            # system. Reusing it avoids an O(|S|^3) dense solve at every block
            # event and preserves the same numerical guard/refactor policy as
            # single-point decrement.
            if self._Rmat is None:
                raise RuntimeError(
                    "fast block decrement: missing retained-margin inverse"
                )
            direction = self._Rmat @ rhs
            d_rho = float(direction[0])
            d_margin = direction[1:]
            gamma = (
                d_rho
                + 2.0 * (self._K[:, margin] @ d_margin)
                + 2.0
                * (
                    self._K[:, active_forget] @ d_forget
                )
            )
            gamma[margin] = 0.0

            candidates = [(1.0, 6, -1)]  # the entire forgotten block -> 0
            for offset, index in enumerate(margin):
                delta = float(d_margin[offset])
                if delta > self.tol:
                    candidates.append(
                        ((self.C - self.alpha[index]) / delta, 2, index)
                    )
                elif delta < -self.tol:
                    candidates.append(
                        (-self.alpha[index] / delta, 3, index)
                    )
            for index in self.E:
                if gamma[index] > self.tol:
                    candidates.append(
                        (-self.grad[index] / gamma[index], 4, index)
                    )
            for index in self.R:
                if gamma[index] < -self.tol:
                    candidates.append(
                        (-self.grad[index] / gamma[index], 5, index)
                    )
            feasible = [
                candidate
                for candidate in candidates
                if np.isfinite(candidate[0])
                and candidate[0] > self.tol
            ]
            if not feasible:
                raise RuntimeError(
                    "fast block decrement: no feasible homotopy step"
                )
            step, situation, index = min(
                feasible,
                key=lambda candidate: candidate[0],
            )
            step = min(float(step), 1.0)
            self.alpha[active_forget] += step * d_forget
            self.alpha[margin] += step * d_margin
            self.rho += step * d_rho
            self.grad += step * gamma

            if situation == 6 or step >= 1.0 - self.tol:
                self.alpha[forget] = 0.0
                break
            if situation == 2:  # retained S -> E
                self.alpha[index] = self.C
                self.S.remove(index)
                self.E.append(index)
            elif situation == 3:  # retained S -> R
                self.alpha[index] = 0.0
                self.S.remove(index)
                self.R.append(index)
            elif situation == 4:  # retained E -> S
                self.E.remove(index)
                self.S.append(index)
            elif situation == 5:  # retained R -> S
                self.R.remove(index)
                self.S.append(index)
            self._refactor()
        else:
            raise RuntimeError("fast block decrement: exceeded max_iter")

        self.alpha[forget] = 0.0
        self._delete(forget)
        self._refactor()

    # ----------------------------------------- point-in-time reserve removal
    def evict_reserve(self, idxs=None, keep_at_most: int | None = None):
        """Remove currently reserve points (alpha = 0) with no re-solving.

        idxs: explicit reserve indices to drop; or keep_at_most: drop oldest
        reserve points until total size <= keep_at_most. Returns #evicted.

        This is a point-in-time certificate: it preserves the current optimum
        and decision function exactly. It does not certify equivalence after
        future admissions, when a removed reserve point could have become
        active in the full-history solve.
        """
        if idxs is None:
            if keep_at_most is None or len(self.alpha) <= keep_at_most:
                return 0
            n_drop = len(self.alpha) - keep_at_most
            idxs = sorted(self.R)[:n_drop]
        idxs = [i for i in idxs if i in set(self.R)]
        if idxs:
            self._delete(idxs)
        return len(idxs)

    def _delete(self, idxs):
        """Batch-remove rows (must be zero-coefficient) and reindex; refactor R
        only via index remap (S membership unchanged, so R is still valid)."""
        drop = set(idxs)
        keep = [i for i in range(len(self.alpha)) if i not in drop]
        remap = {old: new for new, old in enumerate(keep)}
        self.X = self.X[keep]
        self.alpha = self.alpha[keep]
        self.grad = self.grad[keep]
        self._K = self._K[np.ix_(keep, keep)]
        self.S = [remap[i] for i in self.S if i not in drop]
        self.E = [remap[i] for i in self.E if i not in drop]
        self.R = [remap[i] for i in self.R if i not in drop]

    # ------------------------------------------------------------- utilities
    def decision_function(self, Xq) -> np.ndarray:
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float64))
        active = self.S + self.E
        if not active:
            return np.full(len(Xq), -self.rho)
        Kx = self._kfun(Xq, self.X[active])
        return 2.0 * (Kx @ self.alpha[active]) - self.rho

    def weights(self) -> np.ndarray:
        return self.alpha.copy()

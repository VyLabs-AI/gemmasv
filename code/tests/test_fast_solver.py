"""FastOneClassSVM must match the verified reference solver exactly.

The reference (OneClassIncrementalSVM) was itself verified against an
independent batch QP; here the fast solver (incremental Gram, maintained
gradients, rank-1 R) is checked against the reference over streamed adds,
removals, and point-in-time reserve removal.
"""
import numpy as np
import pytest

from cp_svm import (
    FastOneClassSVM,
    OneClassIncrementalSVM,
    radial_kernel,
    recover_rho,
    solve_svdd_qp,
)

KPAR = 1.0


def synthetic(n=120, d=4, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d) * 1.5


def build_pair(X, n_seed, C, refactor_every=500):
    ref = OneClassIncrementalSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(X[:n_seed])
    fast = FastOneClassSVM(C=C, ktype="r", kpar=KPAR,
                           refactor_every=refactor_every).seed_from_qp(X[:n_seed])
    return ref, fast


def assert_match(ref, fast, X_probe, tol=1e-6):
    assert np.max(np.abs(ref.alpha - fast.alpha)) < tol
    assert abs(ref.rho - fast.rho) < tol
    assert sorted(ref.S) == sorted(fast.S)
    assert sorted(ref.E) == sorted(fast.E)
    assert sorted(ref.R) == sorted(fast.R)
    f_r = ref.decision_function(X_probe)
    f_f = fast.decision_function(X_probe)
    assert np.max(np.abs(f_r - f_f)) < tol


def test_fast_matches_reference_over_stream():
    X = synthetic(n=120)
    C = 1.0 / (0.1 * len(X))
    n_seed = 20
    ref, fast = build_pair(X, n_seed, C)
    for i in range(n_seed, len(X)):
        ref.add_point(X[i])
        fast.add_point(X[i])
        if i % 17 == 0:
            assert_match(ref, fast, X[:n_seed])
    assert_match(ref, fast, X[:n_seed])


def test_fast_matches_reference_with_frequent_refactor_and_without():
    """Rank-1 maintenance with rare refactorization must agree with the
    closed-form-heavy reference; frequent refactorization is the control."""
    X = synthetic(n=100, seed=3)
    C = 1.0 / (0.1 * len(X))
    ref, _ = build_pair(X, 20, C)
    fast_rare = FastOneClassSVM(C=C, ktype="r", kpar=KPAR, refactor_every=10**9).seed_from_qp(X[:20])
    fast_freq = FastOneClassSVM(C=C, ktype="r", kpar=KPAR, refactor_every=5).seed_from_qp(X[:20])
    for i in range(20, len(X)):
        ref.add_point(X[i])
        fast_rare.add_point(X[i])
        fast_freq.add_point(X[i])
    assert_match(ref, fast_rare, X[:20], tol=1e-5)
    assert_match(ref, fast_freq, X[:20], tol=1e-6)


def test_fast_decrement_inverts_increment():
    X = synthetic(n=80, seed=5)
    C = 1.0 / (0.1 * len(X))
    fast = FastOneClassSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(X[:30])
    f0 = fast.decision_function(X[:30]).copy()
    a0, r0 = fast.alpha.copy(), fast.rho
    added = [fast.add_point(X[30 + k]) for k in range(5)]
    for idx in reversed(added):
        fast.remove_point(idx)
    assert fast.alpha.shape == a0.shape
    assert np.max(np.abs(fast.decision_function(X[:30]) - f0)) < 1e-8
    assert abs(fast.rho - r0) < 1e-8
    assert np.max(np.abs(fast.alpha - a0)) < 1e-8


def test_fast_multi_point_decrement_matches_fixed_c_batch_refit():
    X = synthetic(n=80, seed=13)
    C = 1.0 / (0.3 * len(X))
    forget = [10, 20, 30, 40]
    keep = [index for index in range(len(X)) if index not in forget]
    fast = FastOneClassSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(X)
    for index in reversed(forget):
        fast.remove_point(index)

    X_keep = X[keep]
    gram = radial_kernel(X_keep, X_keep, KPAR)
    alpha = solve_svdd_qp(gram, C)
    rho = recover_rho(gram, alpha, C)
    probes = np.vstack([X_keep, X[forget]])
    expected = 2.0 * (radial_kernel(probes, X_keep, KPAR) @ alpha) - rho
    assert np.max(np.abs(fast.decision_function(probes) - expected)) < 1e-6


def test_fast_block_decrement_matches_fixed_c_batch_refit():
    X = synthetic(n=80, seed=17)
    C = 1.0 / (0.3 * len(X))
    forget = [10, 20, 30, 40]
    keep = [index for index in range(len(X)) if index not in forget]
    fast = FastOneClassSVM(C=C, ktype="r", kpar=KPAR).seed_from_qp(X)
    fast.remove_points(forget)

    X_keep = X[keep]
    gram = radial_kernel(X_keep, X_keep, KPAR)
    alpha = solve_svdd_qp(gram, C)
    rho = recover_rho(gram, alpha, C)
    probes = np.vstack([X_keep, X[forget]])
    expected = 2.0 * (radial_kernel(probes, X_keep, KPAR) @ alpha) - rho
    assert np.max(np.abs(fast.decision_function(probes) - expected)) < 1e-7
    assert abs(fast.alpha.sum() - 1.0) < 1e-8


def test_certified_eviction_leaves_decision_unchanged():
    # Wide kernel -> one blob with a populated interior (reserve set).
    X = synthetic(n=150, seed=7)
    C = 1.0 / (0.2 * len(X))
    fast = FastOneClassSVM(C=C, ktype="r", kpar=3.5).seed_from_qp(X[:40])
    for i in range(40, 150):
        fast.add_point(X[i])
    assert len(fast.R) > 10, f"config produced too small a reserve set ({len(fast.R)})"
    probe = synthetic(n=30, seed=99)
    f_before = fast.decision_function(probe).copy()
    n_before = len(fast.alpha)
    evict = sorted(fast.R)[: len(fast.R) // 2]
    n_evicted = fast.evict_reserve(idxs=evict)
    assert n_evicted == len(evict) > 0
    assert len(fast.alpha) == n_before - n_evicted
    f_after = fast.decision_function(probe)
    # Reserve points are inert for this solved context: eviction is exactly
    # output-preserving at the time it is performed.
    assert np.max(np.abs(f_after - f_before)) < 1e-12


def test_reserve_eviction_is_not_certified_against_future_admissions():
    """A currently inert point can become active after a later admission.

    The first three points form a minimum enclosing description in which the
    third point is reserve.  A far-away fourth point changes the optimum: when
    the reserve point is retained it acquires positive weight, so a stream that
    evicted it earlier no longer matches the full-history solve.
    """

    initial = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.9]])
    future = np.array([0.0, -10.0])
    probes = np.vstack([initial, future])
    kwargs = {"C": 1.0, "ktype": "r", "kpar": 5.0}

    full = FastOneClassSVM(**kwargs).seed_from_qp(initial)
    pruned = FastOneClassSVM(**kwargs).seed_from_qp(initial)
    assert 2 in full.R

    query = np.array([[0.25, 0.1]])
    values_full = np.array([[1.0], [2.0], [9.0]])

    def readout(model, values):
        weights = (
            radial_kernel(query, model.X, kwargs["kpar"])
            * model.alpha[None, :]
        )
        return (weights @ values) / weights.sum(axis=1, keepdims=True)

    before = full.decision_function(probes)
    readout_before = readout(full, values_full)
    assert pruned.evict_reserve(idxs=[2]) == 1
    assert np.max(np.abs(pruned.decision_function(probes) - before)) < 1e-12
    values_pruned = np.delete(values_full, 2, axis=0)
    # The algebraic reserve coefficient is zero; the QP-seeded implementation
    # realizes it at solver tolerance, which the readout exposes directly.
    assert np.max(np.abs(readout(pruned, values_pruned) - readout_before)) < 1e-8

    full.add_point(future)
    pruned.add_point(future)
    values_full = np.vstack([values_full, [[4.0]]])
    values_pruned = np.vstack([values_pruned, [[4.0]]])

    # The old reserve point participates in the new full-history optimum.
    assert full.alpha[2] > 1e-2
    assert (
        np.max(
            np.abs(
                full.decision_function(probes)
                - pruned.decision_function(probes)
            )
        )
        > 1e-3
    )
    assert (
        np.max(
            np.abs(
                readout(full, values_full)
                - readout(pruned, values_pruned)
            )
        )
        > 1e-4
    )

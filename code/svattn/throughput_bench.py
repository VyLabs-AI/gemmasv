"""Throughput: reference solver vs fast solver vs chunked gate.

Streams random tokens (d=16) and reports tokens/sec and final state sizes.
The reference solver is timed on a shorter stream (it is the verification
oracle, not a production path).

Run:
    PYTHONPATH=. \
        .venv/bin/python -m svattn.throughput_bench
"""
from __future__ import annotations

import time
import numpy as np

from cp_svm import OneClassIncrementalSVM, FastOneClassSVM
from svattn.chunked_gate import ChunkedSVGate

D, KPAR = 16, 3.0


def stream(n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n, D)


NU, NOMINAL = 0.3, 512
C_FIXED = 1.0 / (NU * NOMINAL)          # fixed absolute box bound, as in the gate
N_SEED = int(np.ceil(1.0 / C_FIXED)) + 10


def bench_reference(n=400):
    X = stream(n)
    m = OneClassIncrementalSVM(C=C_FIXED, ktype="r", kpar=KPAR).seed_from_qp(X[:N_SEED])
    t0 = time.time()
    for i in range(N_SEED, n):
        m.add_point(X[i])
    dt = time.time() - t0
    return (n - N_SEED) / dt, len(m.alpha)


def bench_fast(n=3000):
    X = stream(n)
    m = FastOneClassSVM(C=C_FIXED, ktype="r", kpar=KPAR).seed_from_qp(X[:N_SEED])
    t0 = time.time()
    for i in range(N_SEED, n):
        m.add_point(X[i])
    dt = time.time() - t0
    return (n - N_SEED) / dt, len(m.alpha), len(m.S) + len(m.E), m.guard_events


def bench_chunked(n=6000, budget=512, chunk=64):
    X = stream(n)
    g = ChunkedSVGate(nu=0.3, budget=budget, chunk=chunk, kpar=KPAR)
    t0 = time.time()
    g.feed(X)
    dt = time.time() - t0
    return n / dt, g.state_size(), g.support_size(), g.n_evicted


def main():
    print(f"token dim d={D}, RBF kpar={KPAR}, nu={NU}, C fixed from nominal budget {NOMINAL}\n")
    # fair same-length comparison (per-token cost grows with state size)
    n_cmp = 800
    r_tps, _ = bench_reference(n=n_cmp)
    f_tps, _, _, _ = bench_fast(n=n_cmp)
    print(f"@ n={n_cmp} (matched): reference {r_tps:7.1f} tok/s | fast {f_tps:7.1f} tok/s "
          f"| speedup {f_tps / r_tps:.1f}x")
    # unbounded fast solver at longer stream (state grows -> per-token cost grows)
    f_tps3, f_n, f_sv, f_guard = bench_fast(n=3000)
    print(f"fast, unbounded     : {f_tps3:8.1f} tok/s  (3000-token stream, state {f_n}, "
          f"SVs {f_sv}, guards {f_guard})")
    # chunked gate: bounded state, sustained throughput (ICU-scale stream)
    for n in (6000, 20000):
        c_tps, c_state, c_sv, c_evict = bench_chunked(n=n)
        print(f"chunked, budget 512 : {c_tps:8.1f} tok/s  ({n}-token stream: "
              f"state {c_state}, SVs {c_sv}, evicted {c_evict})")
    print("\nNote: the budget bounds the reserve via certified eviction; the SV count "
          "itself is governed by nu and the data (SVs are never evicted).")


if __name__ == "__main__":
    main()

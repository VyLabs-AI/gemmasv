"""Exact context forgetting: decremental removal vs decay-style forgetting.

A distinctive "secret" (key, value) pair is added to an SV-Attention context.
We then forget it three ways and measure the residual influence on the layer's
readout, against the ground truth of a layer that never saw the secret:

  - exact decremental (ours): C&P reverse path -> bit-level removal
  - decay (gated-attention style): multiply the secret's coefficient by gamma
  - none: leave it in place (upper bound)

Residual influence = max readout deviation from the never-seen layer over a
probe set, plus whether a probe near the secret key still retrieves the secret.

Run:
    PYTHONPATH=. \
        ./.venv/bin/python -m svattn.forget_demo
"""
from __future__ import annotations

import numpy as np

from cp_svm import OneClassIncrementalSVM


def rbf(A, B, kpar):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (kpar * kpar))


def readout(alpha, keys, values, Q, kpar):
    w = rbf(Q, keys, kpar) * alpha[None, :]
    denom = w.sum(axis=1, keepdims=True)
    denom[denom < 1e-12] = 1e-12
    return w @ values / denom


def one_trial(rng, n_ctx=14, d=4, kpar=1.5, nu=0.5):
    keys = rng.randn(n_ctx, d)
    G = n_ctx + 1
    values = np.eye(G)[:n_ctx]                       # one-hot value per token
    secret_key = rng.randn(d) * 1.6                  # distinctive key
    secret_val = np.eye(G)[n_ctx]                    # unique secret value

    C = 1.0 / (nu * (n_ctx + 1))                     # fixed absolute box bound

    # ground truth: layer that never saw the secret
    base = OneClassIncrementalSVM(C=C, ktype="r", kpar=kpar)
    base.seed_from_qp(keys)
    a_base = np.array(base.alpha)

    # layer with the secret added (exact incremental)
    m = OneClassIncrementalSVM(C=C, ktype="r", kpar=kpar)
    m.seed_from_qp(keys)
    m.add_point(secret_key)
    a_with = np.array(m.alpha)
    keys_with = np.vstack([keys, secret_key])
    vals_with = np.vstack([values, secret_val])

    # probes: near every context key + near the secret
    Q = np.vstack([keys + 0.05 * rng.randn(n_ctx, d),
                   secret_key + 0.05 * rng.randn(1, d)])

    out_base = readout(a_base, keys, values, Q, kpar)
    out_with = readout(a_with, keys_with, vals_with, Q, kpar)

    def secret_retrieved(out):
        return int(np.argmax(out[-1]) == n_ctx)      # probe near secret -> secret value?

    results = {}
    results["retrieves_before"] = secret_retrieved(out_with)

    # ---- exact decremental forgetting ----
    m.remove_point(len(m.alpha) - 1)
    a_after = np.array(m.alpha)
    out_exact = readout(a_after, keys, values, Q, kpar)
    results["exact_residual"] = float(np.max(np.abs(out_exact - out_base)))
    results["exact_retrieves"] = secret_retrieved(out_exact)

    # ---- decay-style forgetting (gamma-scaled coefficient, stale solution) ----
    for gamma in (0.5, 0.1, 0.01):
        a_dec = a_with.copy()
        a_dec[-1] *= gamma
        out_dec = readout(a_dec, keys_with, vals_with, Q, kpar)
        results[f"decay{gamma}_residual"] = float(np.max(np.abs(out_dec - out_base)))
        results[f"decay{gamma}_retrieves"] = secret_retrieved(out_dec)
    return results


def main():
    rng = np.random.RandomState(0)
    rows = [one_trial(rng) for _ in range(30)]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    print("Exact context forgetting vs decay (30 trials):\n")
    print(f"{'method':<22}{'residual influence':>20}{'secret still retrieved':>25}")
    print(f"{'before forgetting':<22}{'-':>20}{agg['retrieves_before']:>25.2f}")
    print(f"{'exact decremental':<22}{agg['exact_residual']:>20.2e}{agg['exact_retrieves']:>25.2f}")
    for g in (0.5, 0.1, 0.01):
        print(f"{f'decay gamma={g}':<22}{agg[f'decay{g}_residual']:>20.2e}"
              f"{agg[f'decay{g}_retrieves']:>25.2f}")
    print("\nResidual influence = max readout deviation from a layer that NEVER saw "
          "the secret, over probes near all context keys and the secret.")


if __name__ == "__main__":
    main()

"""GO/NO-GO: does the max-margin gate select context tokens better than
heuristic eviction at a matched budget, under skewed redundancy?

Setup per trial:
  - g_rare groups with 1 copy each (rare singletons) + g_dense groups with many
    near-duplicate copies -> n tokens total, far fewer effective items.
  - The one-class SVDD over keys marks interior (redundant) copies as RESERVE
    (alpha = 0, provably inert) and keeps boundary/rare keys as support.
  - Budget k = SVDD support size. Baselines select k tokens by: random, recency
    (last k), and we also report full-context (upper bound).
  - Readout: uniform RBF attention restricted to the selected subset; predicted
    group = argmax of the value readout (values are group one-hots).

Decisive metric: recall on RARE groups at matched budget. If the SVDD selection
~= full context while random/recency drop rare keys and fail, the gate earns its
keep. If they tie, the max-margin layer has no defensible selection advantage.

Run:
    PYTHONPATH=. \
        ./ferg/.venv/bin/python -m svattn.sparsity_gonogo
"""
from __future__ import annotations

import numpy as np

from svattn.diff_svdd import svdd_solve_partition


def rbf(A, B, kpar):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (kpar * kpar))


def one_trial(rng, d=4, g_rare=5, g_dense=5, copies=9, dup_noise=0.15,
              kpar=1.2, nu=0.5):
    G = g_rare + g_dense
    centers = rng.randn(G, d) * 1.5
    keys, group, order = [], [], []
    # dense groups first (so "recency" keeps the late tokens; rare keys arrive early)
    for gi in range(g_rare):
        keys.append(centers[gi] + dup_noise * rng.randn(d))
        group.append(gi)
    for gi in range(g_rare, G):
        for _ in range(copies):
            keys.append(centers[gi] + dup_noise * rng.randn(d))
            group.append(gi)
    keys = np.array(keys)
    group = np.array(group)
    n = len(keys)
    V = np.eye(G)[group]                      # value = one-hot group id

    # --- SVDD gate ---
    K = rbf(keys, keys, kpar)
    C = 1.0 / (nu * n)
    try:
        alpha, rho, S, E, R = svdd_solve_partition(K, C)[:5]
    except RuntimeError:
        return None
    support = np.sort(np.concatenate([S, E])).astype(int)
    k = len(support)

    # --- matched-budget baselines ---
    rand_sel = np.sort(rng.choice(n, k, replace=False))
    recency_sel = np.arange(n - k, n)         # last k tokens
    full_sel = np.arange(n)

    def recall(sel, q, true_g):
        w = rbf(q[None, :], keys[sel], kpar).ravel()
        if w.sum() < 1e-12:
            return 0
        out = (w / w.sum()) @ V[sel]
        return int(np.argmax(out) == true_g)

    res = {}
    for name, sel in [("svdd", support), ("random", rand_sel),
                      ("recency", recency_sel), ("full", full_sel)]:
        tot, rare = [], []
        for gi in range(G):
            q = centers[gi] + 0.05 * rng.randn(d)
            r = recall(sel, q, gi)
            tot.append(r)
            if gi < g_rare:
                rare.append(r)
        res[f"{name}_all"] = np.mean(tot)
        res[f"{name}_rare"] = np.mean(rare)
    res["budget"] = k / n
    # how many rare singletons did each selection keep?
    res["svdd_rare_kept"] = np.mean([i in set(support.tolist()) for i in range(g_rare)])
    res["rand_rare_kept"] = np.mean([i in set(rand_sel.tolist()) for i in range(g_rare)])
    return res


def main():
    rng = np.random.RandomState(0)
    rows = []
    for _ in range(40):
        r = one_trial(rng)
        if r:
            rows.append(r)
    keys_ = rows[0].keys()
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys_}

    print(f"trials={len(rows)}  token budget = {agg['budget']*100:.0f}% of context "
          f"(SVDD support fraction)\n")
    print(f"{'selection':<12}{'recall all-groups':>19}{'recall RARE groups':>20}{'rare keys kept':>16}")
    print(f"{'SVDD gate':<12}{agg['svdd_all']:>19.3f}{agg['svdd_rare']:>20.3f}{agg['svdd_rare_kept']:>16.2f}")
    print(f"{'random-k':<12}{agg['random_all']:>19.3f}{agg['random_rare']:>20.3f}{agg['rand_rare_kept']:>16.2f}")
    print(f"{'recency-k':<12}{agg['recency_all']:>19.3f}{agg['recency_rare']:>20.3f}{'0.00':>16}")
    print(f"{'full (100%)':<12}{agg['full_all']:>19.3f}{agg['full_rare']:>20.3f}{'1.00':>16}")
    print("\nGO if: SVDD ~= full on rare groups while random/recency at the same "
          "budget drop rare keys and fail.")


if __name__ == "__main__":
    main()

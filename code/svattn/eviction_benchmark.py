"""Matched-budget KV eviction: certified SV gate vs heavy-hitter (H2O-style).

Multi-query associative recall under skewed redundancy. The context holds rare
singleton items and dense near-duplicate groups; queries probe every group. We
fix a token budget k and compare selection policies, all read out with the same
uniform RBF attention so only the SELECTION differs:

  - svdd     : keep the one-class support set (certified-inert reserve evicted),
               query-INDEPENDENT.
  - h2o*     : H2O-style heavy hitters -- top-k tokens by accumulated attention
               mass over the served queries (ORACLE: sees the eval queries).
  - h2o_warm : heavy hitters using only an early "warmup" half of queries, then
               must serve held-out queries (the realistic streaming case:
               evict before future queries are known).
  - recency  : last k tokens.   - random : random k.   - full : no eviction.

Honest questions: (1) does certified selection match oracle H2O on rare-item
recall? (2) does it beat H2O when future queries are unknown at eviction time?
"""
from __future__ import annotations

import numpy as np

from svattn.diff_svdd import svdd_solve_partition


def rbf(A, B, kpar):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (kpar * kpar))


def make_context(rng, d=4, g_rare=6, g_dense=6, copies=10, dup=0.15):
    G = g_rare + g_dense
    centers = rng.randn(G, d) * 1.5
    keys, group = [], []
    for gi in range(g_rare):
        keys.append(centers[gi] + dup * rng.randn(d)); group.append(gi)
    for gi in range(g_rare, G):
        for _ in range(copies):
            keys.append(centers[gi] + dup * rng.randn(d)); group.append(gi)
    keys = np.array(keys); group = np.array(group)
    V = np.eye(G)[group]
    return keys, group, V, centers, G, g_rare


def recall(sel, keys, V, queries, true_groups, kpar):
    if len(sel) == 0:
        return 0.0, 0.0
    w = rbf(queries, keys[sel], kpar)
    out = w @ V[sel] / w.sum(1, keepdims=True).clip(1e-12)
    pred = out.argmax(1)
    return pred, true_groups


def trial(rng, kpar=1.2, nu=0.45):
    keys, group, V, centers, G, g_rare = make_context(rng)
    n = len(keys)
    K = rbf(keys, keys, kpar)
    C = 1.0 / (nu * n)
    try:
        _, _, S, E, R = svdd_solve_partition(K, C)[:5]
    except RuntimeError:
        return None
    support = np.sort(np.concatenate([S, E])).astype(int)
    k = len(support)

    # queries: 2 per group (probe near the center)
    qg = np.repeat(np.arange(G), 2)
    Q = centers[qg] + 0.05 * rng.randn(len(qg), keys.shape[1])
    # warmup/held-out split for the streaming H2O variant
    warm = np.zeros(len(Q), bool); warm[::2] = True       # half the queries seen early
    held = ~warm

    def topk_by_mass(query_subset):
        mass = rbf(query_subset, keys, kpar).sum(0)         # attention mass per token
        return np.sort(np.argsort(-mass)[:k])

    sels = {
        "svdd": support,
        "h2o_oracle": topk_by_mass(Q),
        "h2o_warm": topk_by_mass(Q[warm]),
        "recency": np.arange(n - k, n),
        "random": np.sort(rng.choice(n, k, replace=False)),
        "full": np.arange(n),
    }
    res = {"budget": k / n}
    for name, sel in sels.items():
        # evaluate on HELD-OUT queries (fair to the streaming variant)
        pred, tg = recall(sel, keys, V, Q[held], qg[held], kpar)
        res[f"{name}_all"] = float(np.mean(pred == tg))
        rare = qg[held] < g_rare
        res[f"{name}_rare"] = float(np.mean(pred[rare] == tg[rare])) if rare.any() else np.nan
    return res


def main():
    rng = np.random.RandomState(0)
    rows = [r for r in (trial(rng) for _ in range(60)) if r]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    print(f"Matched-budget eviction, multi-query recall under redundancy "
          f"({len(rows)} trials, budget={agg['budget']*100:.0f}% of context)\n")
    print(f"{'policy':<14}{'recall all':>12}{'recall RARE':>13}")
    for name in ["svdd", "h2o_oracle", "h2o_warm", "recency", "random", "full"]:
        print(f"{name:<14}{agg[f'{name}_all']:>12.3f}{agg[f'{name}_rare']:>13.3f}")
    print("\nReads: svdd is query-independent + certified. Compare to oracle H2O "
          "(sees eval queries) and warm H2O (must evict before held-out queries).")


if __name__ == "__main__":
    main()

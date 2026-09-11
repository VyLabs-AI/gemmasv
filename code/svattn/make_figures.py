"""Paper figures: (1) budget-recall curve via nu sweep, (2) forgetting bars.

Run:
    PYTHONPATH=. python -m svattn.make_figures
Override the output dir with SVATTN_OUTPUTS=/path/to/outputs if desired.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from svattn.eviction_benchmark import trial as evict_trial
from svattn.forget_demo import one_trial as forget_trial
from svattn import figstyle as fs

# Resolve outputs/ relative to the repo root (this file lives in <root>/svattn/),
# overridable via env var. No absolute, user-specific paths.
OUT = os.environ.get("SVATTN_OUTPUTS", str(Path(__file__).resolve().parent.parent / "outputs"))


def budget_recall_sweep(n_trials=60):
    rng = np.random.RandomState(0)
    rows = []
    for nu in (0.25, 0.35, 0.45, 0.6, 0.75):
        per = [r for r in (evict_trial(rng, nu=nu) for _ in range(n_trials)) if r]
        agg = {k: float(np.nanmean([p[k] for p in per])) for k in per[0]}
        agg["nu"] = nu
        rows.append(agg)
        print(f"nu={nu}: budget={agg['budget']:.2f} svdd_rare={agg['svdd_rare']:.3f} "
              f"h2o_rare={agg['h2o_oracle_rare']:.3f} random_rare={agg['random_rare']:.3f}")
    return rows


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    fs.apply()

    rows = budget_recall_sweep()
    budgets = [r["budget"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    ax[0].plot(budgets, [r["svdd_rare"] for r in rows], "o-", label="SV gate (certified)", color=fs.BLUE, lw=2.2)
    ax[0].plot(budgets, [r["h2o_oracle_rare"] for r in rows], "D--", label="H2O heavy-hitter", color=fs.RED, lw=1.7)
    ax[0].plot(budgets, [r["random_rare"] for r in rows], "s--", label="random-k", color=fs.GRAY, lw=1.7)
    ax[0].plot(budgets, [r["recency_rare"] for r in rows], "^--", label="recency-k", color=fs.DARKGRAY, lw=1.7)
    ax[0].axhline(np.mean([r["full_rare"] for r in rows]), ls=":", color=fs.GREEN, lw=1.8, label="full context")
    ax[0].set_xlabel("token budget (fraction of context kept)")
    ax[0].set_ylabel("recall@1, rare items")
    fs.caps_title(ax[0], "Selection quality at matched budget")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=.3)

    # forgetting panel
    rng = np.random.RandomState(0)
    fr = [forget_trial(rng) for _ in range(30)]
    agg = {k: float(np.mean([r[k] for r in fr])) for k in fr[0]}
    methods = ["exact\ndecremental", "decay\n$\\gamma$=0.5", "decay\n$\\gamma$=0.1", "decay\n$\\gamma$=0.01"]
    residuals = [agg["exact_residual"], agg["decay0.5_residual"],
                 agg["decay0.1_residual"], agg["decay0.01_residual"]]
    colors = [fs.BLUE, fs.RED, fs.RED, fs.RED]
    ax[1].bar(methods, residuals, color=colors, alpha=.9)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("residual influence (max readout deviation)")
    fs.caps_title(ax[1], "Forgetting a planted secret: exact vs decay")
    ax[1].grid(alpha=.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUT}/svattn_headline.png", dpi=140)
    print(f"saved {OUT}/svattn_headline.png")


if __name__ == "__main__":
    main()

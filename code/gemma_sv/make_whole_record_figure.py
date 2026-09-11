"""Summarize and plot the predeclared whole-record benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gemma_sv.robust_eval import expected_max_at_k


REPORT = Path("outputs/gemma_sv_eval/whole_record_synthetic_v1.json")
BOOTSTRAP = 10_000
CONDITIONS = ("present", "decrement", "decay", "icul", "never")
COLORS = {
    "present": "#7f7f7f",
    "decrement": "#1f77b4",
    "decay": "#d62728",
    "icul": "#ff7f0e",
    "never": "#2ca02c",
}


def _ci(values: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = values[rng.integers(0, n, size=(BOOTSTRAP, n))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
    }


def _per_record_leak(rows, condition, k):
    return np.asarray(
        [
            np.mean(
                [
                    expected_max_at_k(field["exact_phrase"], k)
                    for field in row["conditions"][condition]["fields"]
                ]
            )
            for row in rows
        ]
    )


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(REPORT.read_text())
    whole = report["whole_record"]
    rows = whole["records"]
    k_values = [int(k) for k in report["sampling"]["k"]]
    summary = {
        "source": str(REPORT),
        "bootstrap_draws": BOOTSTRAP,
        "attempted": whole["attempted"],
        "admitted": whole["admitted"],
        "admission_rate": whole["admission_rate"],
        "rejections": [
            {"record_id": row["record_id"], "reasons": row["reasons"]}
            for row in whole["rejected_records"]
        ],
        "field_exact_phrase_leak_at_k": {},
        "decrement_minus_never_at_k": {},
        "secret_residual_nats": {},
        "retain_shift_nats": {},
        "composite_prompt_has_power": None,
    }

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.1))
    matrices = {}
    for condition in CONDITIONS:
        matrices[condition] = np.column_stack(
            [
                _per_record_leak(rows, condition, k)
                for k in k_values
            ]
        )
        stats = [
            _ci(matrices[condition][:, index], seed=100 + index)
            for index in range(len(k_values))
        ]
        summary["field_exact_phrase_leak_at_k"][condition] = {
            str(k): stat for k, stat in zip(k_values, stats)
        }
        mean = [stat["mean"] for stat in stats]
        low = [stat["ci95"][0] for stat in stats]
        high = [stat["ci95"][1] for stat in stats]
        style = ":" if condition == "never" else "-"
        left.plot(
            k_values,
            mean,
            style,
            marker="o",
            ms=3,
            color=COLORS[condition],
            label=condition,
        )
        left.fill_between(k_values, low, high, color=COLORS[condition], alpha=0.12)
    left.set_xscale("log", base=2)
    left.set_xticks(k_values)
    left.set_xticklabels([str(k) for k in k_values])
    left.set_xlabel("attacker samples $k$")
    left.set_ylabel("exact-field Leak@k")
    left.set_title(f"(a) all fields removed ({len(rows)}/{whole['attempted']} records admitted)")
    left.legend(frameon=False, fontsize=7)

    for index, k in enumerate(k_values):
        difference = (
            matrices["decrement"][:, index]
            - matrices["never"][:, index]
        )
        summary["decrement_minus_never_at_k"][str(k)] = _ci(
            difference,
            seed=500 + index,
        )

    residuals = {}
    for condition in ("present", "decrement", "decay", "icul"):
        values = []
        for row in rows:
            never = {
                field["name"]: field
                for field in row["conditions"]["never"]["fields"]
            }
            values.append(
                np.mean(
                    [
                        field["secret_probe"]["mean_log_probability"]
                        - never[field["name"]]["secret_probe"][
                            "mean_log_probability"
                        ]
                        for field in row["conditions"][condition]["fields"]
                    ]
                )
            )
        residuals[condition] = np.asarray(values)
        summary["secret_residual_nats"][condition] = _ci(
            residuals[condition],
            seed=700 + CONDITIONS.index(condition),
        )
    order = ("present", "icul", "decay", "decrement")
    means, lows, highs = [], [], []
    for condition in order:
        stat = summary["secret_residual_nats"][condition]
        means.append(stat["mean"])
        lows.append(stat["mean"] - stat["ci95"][0])
        highs.append(stat["ci95"][1] - stat["mean"])
    right.bar(
        range(len(order)),
        means,
        yerr=[lows, highs],
        color=[COLORS[condition] for condition in order],
        capsize=3,
    )
    for index, condition in enumerate(order):
        right.scatter(
            np.full(len(residuals[condition]), index),
            residuals[condition],
            s=10,
            color="#333333",
            alpha=0.65,
        )
    right.axhline(0, color="gray", linestyle=":", linewidth=1)
    right.set_xticks(range(len(order)))
    right.set_xticklabels(order, rotation=15)
    right.set_ylabel("secret log-prob lift over never (nats)")
    right.set_title("(b) complete record signal after deletion")

    retain_shifts = np.asarray(
        [
            row["conditions"]["decrement"]["retain_secret_probe"][
                "mean_log_probability"
            ]
            - row["conditions"]["present"]["retain_secret_probe"][
                "mean_log_probability"
            ]
            for row in rows
        ]
    )
    summary["retain_shift_nats"] = _ci(retain_shifts, seed=900)

    composite_present = np.asarray(
        [
            expected_max_at_k(
                row["conditions"]["present"]["composite_any_exact"],
                1,
            )
            for row in rows
        ]
    )
    composite_never = np.asarray(
        [
            expected_max_at_k(
                row["conditions"]["never"]["composite_any_exact"],
                1,
            )
            for row in rows
        ]
    )
    composite_difference = _ci(composite_present - composite_never, seed=1000)
    summary["composite_present_minus_never_at_1"] = composite_difference
    low, high = composite_difference["ci95"]
    summary["composite_prompt_has_power"] = not (low <= 0 <= high)

    summary["acceptance"] = {
        "all_fixed_c_admitted": all(
            all(
                all(item["feasible"] for item in field["fixed_c_feasibility"])
                for field in row["admission"]["fields"]
            )
            for row in rows
        ),
        "decrement_matches_never_at_all_k": all(
            value["ci95"][0] <= 0 <= value["ci95"][1]
            for value in summary["decrement_minus_never_at_k"].values()
        ),
        "no_neighbor_collateral_loss": (
            summary["retain_shift_nats"]["ci95"][1] >= -0.1
        ),
    }

    fig.tight_layout()
    figure = Path("outputs/gemma_sv_eval/whole_record_deletion.png")
    fig.savefig(figure, dpi=180, bbox_inches="tight")
    output = Path("outputs/gemma_sv_eval/whole_record_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved -> {figure}")
    print(f"saved -> {output}")
    print(json.dumps(summary["acceptance"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

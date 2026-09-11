"""Robust-unlearning figure and bootstrap summary from the merged report.

Reads ``outputs/gemma_sv_eval/robust_unlearning.json`` (stem-probe protocol,
20 leak targets + 20 entangled pairs, 200 samples each) and produces:

  * ``outputs/gemma_sv_eval/robust_leak.png`` -- panel (a) exact-phrase
    Leak@k per condition with 95% bootstrap bands over targets; panel (b)
    per-target teacher-forced stored-signal residuals over the never floor.
  * ``outputs/gemma_sv_eval/robust_summary.json`` -- every number the paper
    quotes, with percentile-bootstrap confidence intervals (B=10,000, seed 0).

    .venv311/bin/python -m gemma_sv.make_robust_figure
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb

RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    os.environ.get("GEMMASV_OUTPUT_ROOT", RELEASE_ROOT / "outputs")
).expanduser()
REPORT = OUTPUT_ROOT / "gemma_sv_eval/robust_unlearning.json"
CLAIMS = RELEASE_ROOT / "code/artifacts/gemma_sv/arxiv_v2_claims.json"
CONDITIONS = ("present", "decrement", "decay", "icul", "never")
COLORS = {
    "present": "#7f7f7f",
    "decrement": "#1f77b4",
    "decay": "#d62728",
    "icul": "#ff7f0e",
    "never": "#2ca02c",
}
LABELS = {
    "present": "present (stored)",
    "decrement": "masked refit (behavior)",
    "decay": "decay (α×0.01)",
    "icul": "ICUL (prompt)",
    "never": "never stored (floor)",
}
BOOTSTRAP = 10_000


def bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% percentile bootstrap CI over axis 0 (targets)."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    draws = values[rng.integers(0, n, size=(BOOTSTRAP, n))].mean(axis=1)
    return float(values.mean()), *(float(q) for q in np.percentile(draws, [2.5, 97.5]))


def leak_at_k_matrix(rows: list[dict], condition: str, k_values: list[int]) -> np.ndarray:
    """(targets, k) matrix of per-target expected-max exact-phrase leak@k."""
    from gemma_sv.robust_eval import expected_max_at_k

    matrix = []
    for row in rows:
        scores = row["conditions"][condition]["exact_phrase"]
        matrix.append([expected_max_at_k(scores, k) for k in k_values])
    return np.asarray(matrix)


def probe_deltas(rows: list[dict], probe_key: str) -> dict[str, np.ndarray]:
    """Per-target mean-log-probability deltas of the secret vs the never floor."""
    deltas = {}
    for condition in ("present", "decrement", "decay", "icul"):
        deltas[condition] = np.asarray([
            row["conditions"][condition][probe_key]["mean_log_probability"]
            - row["conditions"]["never"][probe_key]["mean_log_probability"]
            for row in rows
        ])
    return deltas


def _legacy_main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(REPORT.read_text())
    k_values = [int(k) for k in report["sampling"]["k"]]
    leak_rows = report["leak"]["targets"]
    pair_rows = report["paired"]["pairs"]

    summary: dict = {
        "source": str(REPORT),
        "bootstrap_draws": BOOTSTRAP,
        "leak_targets": len(leak_rows),
        "paired_targets": len(pair_rows),
        "exact_phrase_leak_at_k": {},
        "leak_secret_residual_nats": {},
        "paired_forget_residual_nats": {},
        "paired_retain_shift_nats": {},
    }

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.0))

    # (a) exact-phrase leak@k curves with bootstrap bands.
    for condition in CONDITIONS:
        matrix = leak_at_k_matrix(leak_rows, condition, k_values)
        stats = [bootstrap_ci(matrix[:, i], seed=i) for i in range(len(k_values))]
        mean = [s[0] for s in stats]
        low = [s[1] for s in stats]
        high = [s[2] for s in stats]
        summary["exact_phrase_leak_at_k"][condition] = {
            str(k): {"mean": m, "ci95": [lo, hi]}
            for k, m, lo, hi in zip(k_values, mean, low, high)
        }
        a.plot(k_values, mean, "-o", ms=3.5, color=COLORS[condition],
               label=LABELS[condition],
               zorder=3 if condition == "decrement" else 2)
        a.fill_between(k_values, low, high, color=COLORS[condition], alpha=0.15)
    a.set_xscale("log", base=2)
    a.set_xticks(k_values)
    a.set_xticklabels([str(k) for k in k_values])
    a.set_xlabel("attacker samples $k$")
    a.set_ylabel("exact-phrase Leak@k")
    a.set_title("(a) sampled extraction, 20 targets × 200 samples")
    a.legend(frameon=False, fontsize=7.2)

    # (b) deterministic stored-signal residual over the never floor.
    leak_deltas = probe_deltas(leak_rows, "secret_probe")
    pair_deltas = probe_deltas(pair_rows, "forget_secret_probe")
    order = ("present", "icul", "decay", "decrement")
    width, gap = 0.34, 0.05
    rng = np.random.default_rng(7)
    for slot, condition in enumerate(order):
        for offset, (deltas, tag) in enumerate(
            ((leak_deltas, "leak"), (pair_deltas, "paired"))
        ):
            values = deltas[condition]
            mean, low, high = bootstrap_ci(values, seed=17 + slot)
            summary[
                "leak_secret_residual_nats" if tag == "leak"
                else "paired_forget_residual_nats"
            ][condition] = {"mean": mean, "ci95": [low, high]}
            x = slot + (offset - 0.5) * (width + gap)
            bar_color = COLORS[condition] if offset == 0 else "#c8c8c8"
            b.bar(x, mean, width=width, color=bar_color,
                  yerr=[[mean - low], [high - mean]], capsize=3,
                  edgecolor=COLORS[condition], linewidth=1.2)
            b.scatter(
                x + rng.uniform(-width / 4, width / 4, len(values)),
                values, s=7, color="#333", alpha=0.55, zorder=3, linewidths=0,
            )
    # Paired per-target differences vs the never floor: conditions share
    # targets, so this is the tight test of "indistinguishable from floor".
    summary["leak_at_k_minus_never_paired"] = {}
    never_matrix = leak_at_k_matrix(leak_rows, "never", k_values)
    for condition in ("decrement", "decay", "icul"):
        matrix = leak_at_k_matrix(leak_rows, condition, k_values) - never_matrix
        summary["leak_at_k_minus_never_paired"][condition] = {
            str(k): dict(zip(
                ("mean", "ci95"),
                (lambda s: (s[0], [s[1], s[2]]))(
                    bootstrap_ci(matrix[:, i], seed=1_000 + i)
                ),
            ))
            for i, k in enumerate(k_values)
        }

    retain_shift = np.asarray([
        row["conditions"]["decrement"]["retain_secret_probe"]["mean_log_probability"]
        - row["conditions"]["present"]["retain_secret_probe"]["mean_log_probability"]
        for row in pair_rows
    ])
    mean, low, high = bootstrap_ci(retain_shift, seed=99)
    summary["paired_retain_shift_nats"]["decrement_minus_present"] = {
        "mean": mean, "ci95": [low, high],
    }
    b.axhline(0, color="gray", ls=":", lw=1)
    b.set_xticks(range(len(order)))
    b.set_xticklabels([LABELS[c].split(" (")[0] for c in order], fontsize=8)
    b.set_ylabel("secret log-prob lift over never (nats)")
    b.set_title("(b) teacher-forced secret lift over never")

    fig.tight_layout()
    out = OUTPUT_ROOT / "gemma_sv_eval/robust_leak.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(RELEASE_ROOT / "paper/figs/robust_leak.png", dpi=170, bbox_inches="tight")
    print(f"saved -> {out}")

    summary_path = OUTPUT_ROOT / "gemma_sv_eval/robust_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved -> {summary_path}")

    for condition in CONDITIONS:
        row = summary["exact_phrase_leak_at_k"][condition]
        rendered = ", ".join(
            f"k={k}: {row[str(k)]['mean']:.3f} [{row[str(k)]['ci95'][0]:.3f}, "
            f"{row[str(k)]['ci95'][1]:.3f}]"
            for k in (1, 32, 128)
        )
        print(f"{condition:10s} {rendered}")
    return 0


TITLE = "Repeated sampling leaves edited memory near the never-stored floor"
DESCRIPTION = (
    "Exact-phrase Leak at k over 20 fixed targets and 200 samples per prompt. "
    "Masked refit tracks the never-stored floor, while the record-present and "
    "instruction-only conditions remain higher. Bands are target-bootstrap "
    "95 percent intervals."
)


def build_public_figure():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gemma-sv-robust-leak-v3",
        }
    )
    import matplotlib.pyplot as plt

    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
    table = claims["probabilistic_leak"]["exact_phrase_leak_at_k"]
    order = ("present", "icul", "decrement", "never")
    labels = {
        "present": "record present",
        "icul": "prompt-only ICUL",
        "decrement": "edited memory",
        "never": "never stored",
    }
    styles = {
        "present": ("s", "-"),
        "icul": ("X", "-."),
        "decrement": ("o", "-"),
        "never": ("D", "--"),
    }
    ks = sorted(int(value) for value in table["present"])
    figure, axis = plt.subplots(figsize=(7.0, 3.6))
    for condition in order:
        rows = table[condition]
        means = np.asarray([rows[str(k)]["mean"] for k in ks])
        lower = np.asarray([rows[str(k)]["ci95"][0] for k in ks])
        upper = np.asarray([rows[str(k)]["ci95"][1] for k in ks])
        marker, linestyle = styles[condition]
        axis.plot(
            ks,
            means,
            color=COLORS[condition],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.6,
            markersize=5.0,
            markerfacecolor="white" if condition in {"decrement", "never"} else COLORS[condition],
            markeredgewidth=1.1,
            label=labels[condition],
            zorder=3,
        )
        axis.fill_between(ks, lower, upper, color=COLORS[condition], alpha=0.10)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 8, 32, 128], ["1", "8", "32", "128"])
    axis.set_xlabel("samples per prompt, k (higher is stronger attack)", fontsize=10.2)
    axis.set_ylabel("exact-phrase Leak@k", fontsize=10.2)
    axis.tick_params(labelsize=9.5)
    axis.grid(color="#E1E5E9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    axis.text(
        0.99,
        0.04,
        "20 targets · 200 samples each",
        transform=axis.transAxes,
        ha="right",
        fontsize=9.5,
        color="#555555",
    )
    figure.tight_layout()
    return figure


def write_public_figure(
    figure,
    output_base: Path = RELEASE_ROOT / "paper/figs/robust_leak",
) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(svg, metadata={"Date": None, "Description": DESCRIPTION}, **common)
    add_svg_accessibility(
        svg,
        figure_id="robust-leak",
        title=TITLE,
        description=DESCRIPTION,
    )
    figure.savefig(
        pdf,
        metadata={
            "Title": TITLE,
            "Subject": DESCRIPTION,
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    figure.savefig(
        png,
        dpi=300,
        metadata={"Software": "Gemma-SV", "Title": TITLE, "Description": DESCRIPTION},
        **common,
    )
    normalize_png_srgb(png)
    return svg, pdf, png


def main() -> int:
    import matplotlib.pyplot as plt

    figure = build_public_figure()
    written = write_public_figure(figure)
    plt.close(figure)
    for path in written:
        print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

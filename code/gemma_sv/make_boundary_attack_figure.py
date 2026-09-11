"""Render the behavioral boundary evidence as four readable small multiples.

The chart keeps every reported value in the source JSON while making the
reader-facing hierarchy explicit: edited memory, the never-stored behavioral
reference, stress-test comparators, and the separate behavioral-only scope.
The numerical state certificate is deliberately not conflated with these
attacks.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "gemma-sv-boundary-attacks-v3",
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from gemma_sv.figure_accessibility import normalize_png_srgb


ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1] / "paper"
DATA_PATH = ROOT / "benchmarks" / "iclr_mass_preserving_boundary_v2.json"
OUTPUT_BASE = PAPER / "figs/boundary_attacks"
PREVIEW: Path | None = None

C_EDIT = "#A84D45"
C_REFERENCE = "#2E7D5B"
C_PRESENT = "#686D72"
C_DECAY = "#3D70B2"
C_ICUL = "#D98B20"
C_CERTIFICATE = "#6D5A91"
C_INK = "#202124"
C_GRID = "#DDE2E7"

COLORS = {
    "present": C_PRESENT,
    "decrement": C_EDIT,
    "never": C_REFERENCE,
    "decay": C_DECAY,
    "icul": C_ICUL,
}
LABELS = {
    "present": "record present",
    "decrement": "edited memory",
    "never": "never stored",
    "decay": "decay",
    "icul": "prompt-only",
}
LINE_STYLE = {
    "present": {"marker": "s", "linestyle": "-", "linewidth": 1.3},
    "decrement": {"marker": "o", "linestyle": "-", "linewidth": 2.1},
    "never": {"marker": "D", "linestyle": "--", "linewidth": 1.9},
    "decay": {"marker": "^", "linestyle": ":", "linewidth": 1.5},
    "icul": {"marker": "X", "linestyle": "-.", "linewidth": 1.5},
}
SVG_TITLE = "Behavioral recovery attacks after Gemma memory editing"
SVG_DESCRIPTION = (
    "Four small multiples show six admitted Gemma 4B records. At 200 samples "
    "per prompt, edited-memory and never-stored exact-phrase Leak at k are "
    "both 16.7 percent. Target-free elicitation and related-data updates keep "
    "edited normalized recovery near zero. In the separate 16-of-20 admitted "
    "broad cohort, edited-policy LiRA AUC is 0.517 with 1.4 percent true "
    "positive rate at one percent false positive rate. These are behavioral "
    "tests, not the numerical state certificate."
)


def load_data(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_data(data: dict) -> dict:
    attacks = data["behavioral_attacks"]
    leak_rows = attacks["field_leak_at_k"]
    budgets = np.asarray(
        sorted(int(value) for value in leak_rows["decrement"]),
        dtype=int,
    )
    leak = {
        condition: np.asarray(
            [
                100.0 * float(leak_rows[condition][str(value)]["binary"])
                for value in budgets
            ],
            dtype=float,
        )
        for condition in ("present", "decrement", "never", "decay", "icul")
    }

    elicitation_rows = attacks["elicitation"]
    shots = np.asarray(
        sorted(int(value) for value in elicitation_rows),
        dtype=int,
    )
    elicitation = {}
    for condition, source in (
        ("decrement", "masked_refit"),
        ("decay", "decay"),
        ("icul", "icul"),
    ):
        elicitation[condition] = {
            "mean": np.asarray(
                [
                    float(elicitation_rows[str(shot)][source]["mean"])
                    for shot in shots
                ],
                dtype=float,
            ),
            "lower": np.asarray(
                [
                    float(elicitation_rows[str(shot)][source]["ci95"][0])
                    for shot in shots
                ],
                dtype=float,
            ),
            "upper": np.asarray(
                [
                    float(elicitation_rows[str(shot)][source]["ci95"][1])
                    for shot in shots
                ],
                dtype=float,
            ),
        }

    relearning_rows = attacks["relearning"]
    update_budgets = np.asarray(
        sorted(int(value) for value in relearning_rows),
        dtype=int,
    )
    relearning = np.asarray(
        [
            float(relearning_rows[str(value)]["mean_record_recovery"])
            for value in update_budgets
        ],
        dtype=float,
    )

    broad = attacks["lira_broad"]
    lira = {
        condition: {
            "auc": float(row["auc"]),
            "tpr_at_1pct_fpr": float(row["tpr_at_fpr"]["0.01"]),
            "positive_tests": int(row["positive_tests"]),
            "negative_tests": int(row["negative_tests"]),
        }
        for condition, row in broad["metrics"].items()
    }
    return {
        "records": int(attacks["records"]),
        "field_queries": int(attacks["field_queries"]),
        "samples_per_prompt": int(attacks["samples_per_prompt"]),
        "leak": {"budgets": budgets, "curves": leak},
        "elicitation": {"shots": shots, "conditions": elicitation},
        "relearning": {
            "budgets": update_budgets,
            "mean_record_recovery": relearning,
        },
        "lira": {
            "admitted": int(broad["admission"]["admitted"]),
            "attempted": int(broad["admission"]["attempted"]),
            "conditions": lira,
            "full_repack_fallbacks": int(broad["full_repack_fallbacks"]),
        },
    }


def _style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.65, zorder=0)
    ax.tick_params(colors=C_INK, labelsize=11.0)
    ax.xaxis.label.set_color(C_INK)
    ax.yaxis.label.set_color(C_INK)


def leak_panel(ax, summary: dict) -> None:
    budgets = summary["leak"]["budgets"]
    curves = summary["leak"]["curves"]
    for condition in ("present", "decrement", "never", "decay", "icul"):
        style = LINE_STYLE[condition]
        ax.plot(
            budgets,
            curves[condition],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=4.0 if condition in {"decrement", "never"} else 3.5,
            markerfacecolor=(
                "white"
                if condition in {"decrement", "never"}
                else COLORS[condition]
            ),
            markeredgewidth=1.1,
            color=COLORS[condition],
            label=LABELS[condition],
            zorder=4 if condition in {"decrement", "never"} else 2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 200], ["1", "4", "16", "64", "200"])
    ax.set_ylim(-4, 106)
    ax.set_xlabel("samples per prompt, $k$")
    ax.set_ylabel("exact-phrase Leak@$k$ (%)")
    ax.set_title(
        "(a) Repeated sampling",
        loc="left",
        fontsize=14.0,
        weight="bold",
        pad=8,
    )
    _style_axis(ax)


def elicitation_panel(ax, summary: dict) -> None:
    shots = summary["elicitation"]["shots"]
    conditions = summary["elicitation"]["conditions"]
    ax.axhline(
        0,
        color=C_REFERENCE,
        linestyle="--",
        linewidth=1.1,
        zorder=1,
    )
    for condition in ("decrement", "decay", "icul"):
        row = conditions[condition]
        style = LINE_STYLE[condition]
        ax.plot(
            shots,
            row["mean"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=4.2,
            markerfacecolor=(
                "white" if condition == "decrement" else COLORS[condition]
            ),
            markeredgewidth=1.1,
            color=COLORS[condition],
            label=LABELS[condition],
            zorder=4 if condition == "decrement" else 3,
        )
        ax.fill_between(
            shots,
            row["lower"],
            row["upper"],
            color=COLORS[condition],
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )
    ax.set_xticks(shots)
    ax.set_ylim(-0.11, 1.10)
    ax.set_xlabel("target-free shots")
    ax.set_ylabel("normalized recovery")
    ax.set_title(
        "(b) Target-free elicitation",
        loc="left",
        fontsize=14.0,
        weight="bold",
        pad=8,
    )
    _style_axis(ax)


def relearning_panel(ax, summary: dict) -> None:
    budgets = summary["relearning"]["budgets"]
    means = summary["relearning"]["mean_record_recovery"]
    positions = np.arange(len(budgets))
    ax.axhline(
        0,
        color=C_REFERENCE,
        linestyle="--",
        linewidth=1.1,
        zorder=1,
    )
    ax.plot(
        positions,
        means,
        marker="o",
        linestyle="-",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.3,
        linewidth=2.0,
        color=C_EDIT,
        zorder=4,
    )
    ax.set_xticks(positions, [str(value) for value in budgets])
    ax.set_ylim(-0.125, 0.125)
    ax.set_xlabel("related-data update budget")
    ax.set_ylabel("normalized recovery")
    ax.set_title(
        "(c) Related-data relearning",
        loc="left",
        fontsize=14.0,
        weight="bold",
        pad=8,
    )
    minimum_index = int(np.argmin(means))
    ax.annotate(
        f"{means[minimum_index]:.3f}",
        xy=(positions[minimum_index], means[minimum_index]),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        color=C_EDIT,
        fontsize=12.0,
        weight="bold",
    )
    _style_axis(ax)


def lira_panel(ax, summary: dict) -> None:
    conditions = summary["lira"]["conditions"]
    order = ("present", "icul", "decay", "masked_refit")
    style_key = {
        "present": "present",
        "icul": "icul",
        "decay": "decay",
        "masked_refit": "decrement",
    }
    display = {
        "present": "record present",
        "icul": "prompt-only ICUL",
        "decay": "decay",
        "masked_refit": "edited policy",
    }
    y = np.arange(len(order))[::-1]
    ax.axvline(
        0.5,
        color=C_REFERENCE,
        linestyle="--",
        linewidth=1.1,
        zorder=1,
    )
    ax.text(
        0.503,
        y[-1] - 0.47,
        "chance",
        color=C_REFERENCE,
        fontsize=11.0,
        ha="left",
    )
    for yy, condition in zip(y, order):
        key = style_key[condition]
        value = conditions[condition]["auc"]
        ax.plot(
            value,
            yy,
            marker=LINE_STYLE[key]["marker"],
            markersize=7.0 if condition == "masked_refit" else 6.0,
            markerfacecolor=(
                "white" if condition == "masked_refit" else COLORS[key]
            ),
            markeredgecolor=COLORS[key],
            markeredgewidth=1.4,
            color=COLORS[key],
            zorder=4,
        )
        if condition == "masked_refit":
            ax.text(
                value + 0.018,
                yy,
                f"edited {value:.3f}",
                color=COLORS[key],
                fontsize=12.0,
                weight="bold",
                va="center",
                ha="left",
            )
        else:
            ax.text(
                value - 0.018 if value > 0.9 else value + 0.018,
                yy,
                display[condition],
                color=COLORS[key],
                fontsize=10.4,
                va="center",
                ha="right" if value > 0.9 else "left",
            )
    ax.set_yticks(y, [""] * len(order))
    ax.set_xlim(0.47, 1.025)
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.50", "0.75", "1.00"])
    ax.set_xlabel("membership-inference AUC")
    ax.set_title(
        "(d) Membership inference",
        loc="left",
        fontsize=14.0,
        weight="bold",
        pad=8,
    )
    ax.tick_params(axis="y", length=0)
    _style_axis(ax, grid_axis="x")


def build_figure(data: dict | None = None):
    summary = summarize_data(load_data() if data is None else data)
    figure, axes = plt.subplots(2, 2, figsize=(7.7, 6.8))
    leak_panel(axes[0, 0], summary)
    elicitation_panel(axes[0, 1], summary)
    relearning_panel(axes[1, 0], summary)
    lira_panel(axes[1, 1], summary)
    figure.text(
        0.5,
        0.988,
        "Behavioral recovery tests — separate from numerical certificate",
        ha="center",
        va="top",
        color=C_CERTIFICATE,
        fontsize=11.5,
        style="italic",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[condition],
            marker=LINE_STYLE[condition]["marker"],
            linestyle=LINE_STYLE[condition]["linestyle"],
            linewidth=LINE_STYLE[condition]["linewidth"],
            markersize=5.0,
            markerfacecolor=(
                "white"
                if condition in {"decrement", "never"}
                else COLORS[condition]
            ),
            markeredgewidth=1.1,
            label=LABELS[condition],
        )
        for condition in ("present", "icul", "decay", "decrement", "never")
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        fontsize=11.0,
        handlelength=2.2,
        columnspacing=1.4,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.975,
        top=0.845,
        bottom=0.095,
        wspace=0.34,
        hspace=0.55,
    )
    return figure


def _add_svg_accessibility(path: Path, *, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "<svg "
    start = text.index(marker)
    end = text.index(">", start) + 1
    labelled = text[start:end].replace(
        marker,
        '<svg role="img" aria-labelledby="boundary-title" '
        'aria-describedby="boundary-desc" ',
        1,
    )
    accessible = (
        f"\n <title id=\"boundary-title\">{escape(title)}</title>"
        f"\n <desc id=\"boundary-desc\">{escape(description)}</desc>"
    )
    output = text[:start] + labelled + accessible + text[end:]
    output = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    path.write_text(output, encoding="utf-8")


def write_figure(
    figure,
    output_base: Path = OUTPUT_BASE,
    preview: Path | None = PREVIEW,
) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(
        svg,
        metadata={
            "Date": None,
            "Description": SVG_DESCRIPTION,
        },
        **common,
    )
    _add_svg_accessibility(
        svg,
        title=SVG_TITLE,
        description=SVG_DESCRIPTION,
    )
    figure.savefig(
        pdf,
        metadata={
            "Title": SVG_TITLE,
            "Subject": SVG_DESCRIPTION,
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    figure.savefig(
        png,
        dpi=300,
        metadata={
            "Software": "Gemma-SV",
            "Title": SVG_TITLE,
            "Description": SVG_DESCRIPTION,
        },
        **common,
    )
    normalize_png_srgb(png)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            preview,
            dpi=180,
            metadata={"Software": "Gemma-SV", "Title": SVG_TITLE},
            **common,
        )
    return svg, pdf, png


def main() -> int:
    figure = build_figure()
    written = write_figure(figure)
    plt.close(figure)
    for path in written:
        print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

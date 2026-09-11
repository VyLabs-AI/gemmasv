"""Render the replay-exactness and suffix-cost figure (paper Fig. kimi-mimic).

The compact top panel makes the exact-zero replay outcome visible for every
qualified cohort.  The dominant lower panel plots the measured work directly
against the number of suffix tokens replayed.  Each timing remains an
individual point; only the faint line is a fit.

Reads aggregate-only reports and contains no source text by construction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "gemma-sv-kimi-mimic-v2",
    }
)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT.parent
REPORTS = {
    "cds_small": ROOT / "artifacts/kimi_sv/mimic_deletion_8bit_n8.json",
    "cds_large": ROOT / "artifacts/kimi_sv/mimic_deletion_8bit_n128_v2.json",
    "notes": ROOT / "artifacts/kimi_sv/notes_deletion_8bit_n16.json",
}
LABELS = {
    "cds_small": "short tables",
    "cds_large": "long tables",
    "notes": "long notes",
}
COLOR_PRESENT = "#6F7479"
COLOR_EXACT = "#2E7D5B"
COLOR_WORK = "#A56A16"
COLOR_GRID = "#D9DDE1"
COLOR_INK = "#202124"
OUTPUT_BASE = RELEASE_ROOT / "paper/figs/kimi_mimic"
FIGURE = OUTPUT_BASE.with_suffix(".pdf")
PREVIEW = RELEASE_ROOT / "outputs/kimi_sv/kimi_mimic_figure.png"
TITLE = "Replay matches fresh omission while work grows with the surviving suffix"
DESCRIPTION = (
    "Among 22 attempts, 21 qualify (8/8, 9/9, and 4/5 qualified), and all "
    "21 qualified replay checks pass at zero residual on logits and 80 named KDA "
    "arrays. A separate timing audit measures replay work against suffix length."
)


def load_reports(
    paths: Mapping[str, Path] = REPORTS,
) -> dict[str, dict]:
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }


def summarize_reports(reports: Mapping[str, dict]) -> dict:
    cohorts = []
    for name in REPORTS:
        report = reports[name]
        lift = report["efficacy"]["field_lift_present_nats"]
        after = report["efficacy"][
            "field_max_abs_lift_after_deletion_nats"
        ]["max"]
        logit = report["exactness"]["logit_residual"]["max"]
        state = report["exactness"]["state_residual"]["max"]
        if any(float(value) != 0.0 for value in (after, logit, state)):
            raise ValueError(f"{name} no longer has an exact-zero replay audit")
        admission = report["admission"]
        cohorts.append(
            {
                "name": name,
                "label": LABELS[name],
                "admitted": int(admission["admitted"]),
                "attempted": int(admission["attempted"]),
                "mean": float(lift["mean"]),
                "min": float(lift["min"]),
                "max": float(lift["max"]),
            }
        )

    timing = sorted(
        reports["cds_large"]["per_position"],
        key=lambda row: int(row["replayed_tokens"]),
    )
    tokens = np.asarray([row["replayed_tokens"] for row in timing], dtype=float)
    replay = np.asarray([row["replay_seconds"] for row in timing], dtype=float)
    rebuild = np.asarray([row["rebuild_seconds"] for row in timing], dtype=float)
    slope, intercept = np.polyfit(tokens, replay, 1)
    fitted = slope * tokens + intercept
    residual = float(np.sum((replay - fitted) ** 2))
    total = float(np.sum((replay - replay.mean()) ** 2))
    r_squared = 1.0 - residual / total
    return {
        "cohorts": cohorts,
        "admitted": sum(row["admitted"] for row in cohorts),
        "attempted": sum(row["attempted"] for row in cohorts),
        "tokens": tokens,
        "replay": replay,
        "rebuild": rebuild,
        "fit": fitted,
        "r_squared": r_squared,
    }


def build_figure(reports: Mapping[str, dict]):
    summary = summarize_reports(reports)
    figure = plt.figure(figsize=(6.8, 5.5))
    figure.suptitle(
        "Replay reaches fresh omission on the audited state surface",
        fontsize=12.6,
        weight="bold",
        y=0.99,
    )
    grid = figure.add_gridspec(2, 1, height_ratios=(1.25, 1.55), hspace=0.72)
    exact_grid = grid[0].subgridspec(
        1,
        2,
        width_ratios=(3.2, 1.0),
        wspace=0.14,
    )
    present = figure.add_subplot(exact_grid[0])
    replay_zero = figure.add_subplot(exact_grid[1], sharey=present)
    timing = figure.add_subplot(grid[1])

    cohorts = summary["cohorts"]
    y = np.arange(len(cohorts))[::-1]
    means = np.asarray([row["mean"] for row in cohorts])
    lows = means - np.asarray([row["min"] for row in cohorts])
    highs = np.asarray([row["max"] for row in cohorts]) - means
    present.errorbar(
        means,
        y,
        xerr=np.vstack((lows, highs)),
        fmt="o",
        markersize=7,
        capsize=4,
        color=COLOR_PRESENT,
        ecolor="#AEB4BA",
        elinewidth=1.4,
        zorder=3,
    )
    replay_zero.scatter(
        np.zeros_like(y),
        y,
        marker="D",
        s=68,
        facecolor=COLOR_EXACT,
        edgecolor=COLOR_EXACT,
        linewidth=1.2,
        zorder=4,
    )
    present.set_yticks(
        y,
        [
            f"{row['label']}  ({row['admitted']}/{row['attempted']})"
            for row in cohorts
        ],
    )
    present.set_xlim(0, max(row["max"] for row in cohorts) * 1.06)
    present.set_xlabel("pull toward remembered content (nats)")
    present.set_title("PRESENT", fontsize=10.1, color=COLOR_PRESENT, weight="bold")
    present.tick_params(axis="y", length=0, pad=8)
    present.grid(axis="x", color=COLOR_GRID, linewidth=0.6)
    present.spines[["top", "right", "left"]].set_visible(False)

    replay_zero.set_xlim(-0.18, 0.18)
    replay_zero.set_xticks([0], ["0"])
    replay_zero.set_title(
        "REPLAY = 0",
        fontsize=10.1,
        color=COLOR_EXACT,
        weight="bold",
    )
    replay_zero.tick_params(axis="y", left=False, labelleft=False)
    replay_zero.axvspan(-0.18, 0.18, color="#EDF7F2", zorder=0)
    replay_zero.spines[["top", "right", "left"]].set_visible(False)
    figure.text(
        0.18,
        0.915,
        "(a) All qualified replay checks return to zero",
        fontsize=12.0,
        weight="bold",
        ha="left",
    )

    tokens = summary["tokens"]
    replay = summary["replay"]
    rebuild = summary["rebuild"]
    timing.plot(
        tokens,
        summary["fit"],
        color=COLOR_EXACT,
        linewidth=1.6,
        alpha=0.55,
        zorder=1,
    )
    timing.scatter(
        tokens,
        replay,
        marker="o",
        s=38,
        facecolor="white",
        edgecolor=COLOR_EXACT,
        linewidth=1.6,
        zorder=3,
    )
    timing.scatter(
        tokens,
        rebuild,
        marker="s",
        s=28,
        facecolor=COLOR_PRESENT,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    timing.axhline(
        rebuild.mean(),
        color=COLOR_PRESENT,
        linewidth=1.0,
        linestyle="--",
        alpha=0.75,
        zorder=1,
    )
    tick_values = np.linspace(tokens[0], tokens[-1], 5)
    tick_labels = [
        "0\nnewest",
        *[f"{value:,.0f}" for value in tick_values[1:-1]],
        f"{tokens[-1]:,.0f}\noldest",
    ]
    timing.set_xticks(tick_values, tick_labels)
    timing.get_xticklabels()[0].set_color(COLOR_WORK)
    timing.get_xticklabels()[-1].set_color(COLOR_WORK)
    timing.get_xticklabels()[0].set_weight("bold")
    timing.get_xticklabels()[-1].set_weight("bold")
    timing.set_xlabel("conversation after the deleted record (tokens to replay)")
    timing.set_ylabel("time (seconds)")
    timing.set_title(
        "(b) Replay cost grows with suffix length",
        loc="left",
        fontsize=12.0,
        weight="bold",
        pad=34,
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLOR_EXACT,
            markeredgewidth=1.5,
            label="replay timing",
        ),
        Line2D([0], [0], color=COLOR_EXACT, alpha=0.55, label="linear fit"),
        Line2D(
            [0],
            [0],
            marker="s",
            color=COLOR_PRESENT,
            linestyle="--",
            label="full rebuild",
        ),
    ]
    timing.legend(
        handles=legend,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        ncol=3,
        fontsize=9.6,
        handlelength=1.8,
        columnspacing=1.2,
    )
    timing.set_xlim(-0.025 * tokens[-1], 1.025 * tokens[-1])
    timing.set_ylim(0, max(rebuild.max(), replay.max()) * 1.10)
    timing.grid(axis="y", color=COLOR_GRID, linewidth=0.6)
    timing.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.22, right=0.985, top=0.84, bottom=0.13)
    return figure


def write_figure(
    figure,
    output_base: Path = OUTPUT_BASE,
    preview: Path | None = PREVIEW,
) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.03}
    figure.savefig(svg, metadata={"Date": None, "Description": DESCRIPTION}, **common)
    add_svg_accessibility(
        svg,
        figure_id="kimi-replay",
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
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            preview,
            dpi=180,
            metadata={"Software": "Gemma-SV"},
            **common,
        )
    return svg, pdf, png


def main() -> int:
    figure = build_figure(load_reports())
    written = write_figure(figure)
    plt.close(figure)
    for path in (*written, PREVIEW):
        print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

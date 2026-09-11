"""Render Gemma-only scope prototypes after the KDA paper split."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "gemma-split-scope-v1",
    }
)
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "updated_figure_format"
INK = "#20272E"
MUTED = "#66717D"
LINE = "#CAD2DA"
BLUE = "#0072B2"
BLUE_LIGHT = "#EAF4FA"
GREEN = "#00876C"
GREEN_LIGHT = "#E8F5F1"
ORANGE = "#C9460A"
ORANGE_LIGHT = "#FCEDE5"
GRAY_LIGHT = "#F3F5F7"


def _box(ax, x, y, width, height, face, edge=LINE, linewidth=1.0):
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
        )
    )


def build_claim_ladder():
    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "Four claims require four references",
        fontsize=12.2,
        weight="bold",
        color=INK,
        va="top",
    )
    rows = [
        (
            0.73,
            "4",
            "Regenerate causal descendants",
            "assistant turns · summaries · tools",
            "NOT TESTED",
            GRAY_LIGHT,
            MUTED,
        ),
        (
            0.53,
            "3",
            "Match fresh raw omission",
            "re-ingest history without the record",
            "2.15-nat pilot gap",
            ORANGE_LIGHT,
            ORANGE,
        ),
        (
            0.33,
            "2",
            "Match retained-key refit",
            "same contextualized keys · frozen box",
            "max KL 6.48×10⁻¹¹",
            GREEN_LIGHT,
            GREEN,
        ),
        (
            0.13,
            "1",
            "Suppress the target answer",
            "behavioral outcome on registered probes",
            "MEASURED",
            BLUE_LIGHT,
            BLUE,
        ),
    ]
    for y, number, title, detail, result, face, color in rows:
        ax.scatter([0.055], [y + 0.065], s=260, facecolor="white", edgecolor=color)
        ax.text(
            0.055,
            y + 0.065,
            number,
            ha="center",
            va="center",
            color=color,
            weight="bold",
        )
        _box(ax, 0.10, y, 0.62, 0.13, face, edge=color)
        ax.text(0.125, y + 0.085, title, color=INK, weight="bold", fontsize=10.4)
        ax.text(0.125, y + 0.035, detail, color=MUTED, fontsize=8.3)
        _box(ax, 0.76, y + 0.018, 0.21, 0.094, "white", edge=color)
        ax.text(
            0.865,
            y + 0.065,
            result,
            color=color,
            weight="bold",
            fontsize=8.8,
            ha="center",
            va="center",
        )
    ax.annotate(
        "",
        xy=(0.025, 0.93),
        xytext=(0.025, 0.10),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.1},
    )
    ax.text(
        0.02,
        0.02,
        "A result on one rung is not evidence for a stronger rung.",
        color=MUTED,
        fontsize=8.5,
        weight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.02)
    return fig


def build_deletion_taxonomy():
    fig, ax = plt.subplots(figsize=(7.3, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    headers = ((0.03, "OPERATION"), (0.37, "REFERENCE"), (0.72, "WHAT IT SHOWS"))
    for x, text in headers:
        ax.text(x, 0.94, text, color=MUTED, weight="bold", fontsize=8.4)
    ax.plot([0.02, 0.98], [0.91, 0.91], color=LINE, lw=0.8)
    rows = [
        (
            0.70,
            "Prompt-only suppression",
            "record-present state",
            "behavior only; state unproven",
            GRAY_LIGHT,
            MUTED,
        ),
        (
            0.47,
            "Remove owned rows",
            "fixed-C retained-key refit",
            "implementation check",
            GREEN_LIGHT,
            GREEN,
        ),
        (
            0.24,
            "Raw repack",
            "history rebuilt without record",
            "removes ingestion imprint",
            BLUE_LIGHT,
            BLUE,
        ),
        (
            0.01,
            "Regenerate descendants",
            "causal counterfactual",
            "not executed",
            ORANGE_LIGHT,
            ORANGE,
        ),
    ]
    for y, operation, reference, result, face, color in rows:
        _box(ax, 0.02, y, 0.29, 0.16, face, edge=color)
        _box(ax, 0.35, y, 0.31, 0.16, "white", edge=color)
        _box(ax, 0.70, y, 0.28, 0.16, face, edge=color)
        ax.text(0.045, y + 0.10, operation, color=INK, weight="bold", fontsize=9.2)
        ax.text(0.375, y + 0.10, reference, color=color, weight="bold", fontsize=8.8)
        ax.text(0.725, y + 0.10, result, color=INK, weight="bold", fontsize=8.3)
        for left, right in ((0.315, 0.345), (0.665, 0.695)):
            ax.annotate(
                "",
                xy=(right, y + 0.08),
                xytext=(left, y + 0.08),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.0},
            )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.02)
    return fig


def write_figure(
    figure,
    output_base: Path,
    *,
    figure_id: str,
    title: str,
    description: str,
):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png"))
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(paths[0], metadata={"Date": None, "Description": description}, **common)
    add_svg_accessibility(
        paths[0],
        figure_id=figure_id,
        title=title,
        description=description,
    )
    figure.savefig(
        paths[1],
        metadata={
            "Title": title,
            "Subject": description,
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    figure.savefig(
        paths[2],
        dpi=300,
        metadata={"Software": "Gemma split renderer", "Title": title},
        **common,
    )
    normalize_png_srgb(paths[2])
    return paths


def main() -> int:
    figures = (
        (
            build_claim_ladder(),
            OUTPUT / "claim_ladder",
            "gemma-claim-ladder",
            "Gemma deletion claim ladder",
            "Four increasingly strong deletion claims are separated by their reference; the retained-key refit certificate does not establish raw omission or causal regeneration.",
        ),
        (
            build_deletion_taxonomy(),
            OUTPUT / "deletion_taxonomy",
            "gemma-deletion-taxonomy",
            "Gemma deletion operations and references",
            "Prompt suppression, row deletion, raw repack, and causal regeneration use different references and support different conclusions.",
        ),
    )
    for figure, base, figure_id, title, description in figures:
        for path in write_figure(
            figure,
            base,
            figure_id=figure_id,
            title=title,
            description=description,
        ):
            print(f"saved -> {path}")
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

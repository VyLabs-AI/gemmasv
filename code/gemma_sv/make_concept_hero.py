"""Render the paper's representation-to-reference deletion taxonomy.

The LongMemEval phone figure already explains one selective-forgetting case.
This complementary figure answers a different question: which operation and
which comparison become available when the same selected record is represented
as addressable rows or as blended recurrent state?

Every label is placed with ``fit_text`` so it cannot leave its container.
Numbers, denominators, implementation qualifications, and equations stay in
the caption.  The artwork is deterministic and requires no model or data.
"""
from __future__ import annotations
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "gemma-sv-concept-taxonomy-v3",
    }
)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from gemma_sv.figure_accessibility import normalize_png_srgb
from gemma_sv.figure_text import fit_text

C_TARGET = "#A84D45"
C_TARGET_LIGHT = "#F8ECEA"
C_RETAINED = "#356DB5"
C_RETAINED_LIGHT = "#EAF1FB"
C_REFERENCE = "#2E7D5B"
C_REFERENCE_LIGHT = "#EDF7F2"
C_CERTIFICATE = "#6D5A91"
C_CERTIFICATE_LIGHT = "#F1EEF7"
C_STATE = "#738091"
C_STATE_LIGHT = "#EEF1F4"
C_INK = "#202124"
C_MUTED = "#62676D"
C_LINE = "#B9C0C7"
C_ROW = "#FAFBFC"

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT.parent
OUTPUT_BASE = RELEASE_ROOT / "paper/figs/hero_concept"
PREVIEW: Path | None = None
SVG_TITLE = "Where memory lives determines how deletion works"
SVG_DESCRIPTION = (
    "Panel a contrasts addressable rows, which can be removed and refit, with "
    "evolving state, which must be restored and replayed when later updates "
    "change. Panel b orders stopping an answer, matching a refit, rebuilding "
    "without the record, and recomputing later consequences."
)

ROW_HEIGHT = 0.196
ROW_Y = (0.700, 0.480, 0.260)
REP_X, REP_W = 0.028, 0.204
OPERATION_X, OPERATION_W = 0.268, 0.200
REFERENCE_X, REFERENCE_W = 0.504, 0.192
VERIFIED_X, VERIFIED_W = 0.732, 0.236
BOX_DY, BOX_H = 0.053, 0.090

ROWS = (
    {
        "representation": "addressable rows\nGemma",
        "operation": "remove + refit",
        "reference": "refit what remains",
        "verified": "edit matches refit",
        "secondary": "stored key rows",
        "addressable": True,
        "changed": False,
    },
    {
        "representation": "evolving state\nupdates fixed",
        "operation": "carry effect\nforward",
        "reference": "same-update\ncontrol",
        "verified": "matches control",
        "secondary": "KDA control",
        "addressable": False,
        "changed": False,
    },
    {
        "representation": "evolving state\nupdates change",
        "operation": "checkpoint\n+ replay",
        "reference": "rebuild without\nrecord",
        "verified": "rebuilds state",
        "secondary": "Kimi / Qwen",
        "addressable": False,
        "changed": True,
    },
)

LADDER = (
    ("RECOMPUTE LATER CONSEQUENCES", "not tested", C_STATE_LIGHT, C_MUTED),
    (
        "REBUILD WITHOUT THE RECORD",
        "Kimi: audited state \u00b7 Gemma: full rebuild",
        C_REFERENCE_LIGHT,
        C_REFERENCE,
    ),
    (
        "MATCH THE REFIT OF WHAT REMAINS",
        "Gemma: checked numerically",
        C_CERTIFICATE_LIGHT,
        C_CERTIFICATE,
    ),
    ("STOP THE TARGET ANSWER", "behavioral result", C_TARGET_LIGHT, C_TARGET),
)


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    *,
    edgecolor: str = C_INK,
    linewidth: float = 1.0,
    radius: float = 0.014,
    linestyle: str = "-",
    zorder: int = 1,
):
    patch = mpatches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, *, color=C_MUTED, linewidth=1.35):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=3,
    )
    ax.add_patch(arrow)
    return arrow


def _addressable_glyph(ax, y: float) -> None:
    width, height, gap = 0.026, 0.050, 0.007
    span = 5 * width + 4 * gap
    start = REP_X + (REP_W - span) / 2
    for index in range(5):
        selected = index == 2
        x = start + index * (width + gap)
        _box(
            ax,
            x,
            y,
            width,
            height,
            C_TARGET_LIGHT if selected else C_RETAINED_LIGHT,
            edgecolor=C_TARGET if selected else C_RETAINED,
            linewidth=1.4 if selected else 0.9,
            radius=0.006,
        )
        if selected:
            ax.text(
                x + width / 2,
                y + height / 2,
                "\u00d7",
                ha="center",
                va="center",
                color=C_TARGET,
                fontsize=9,
                weight="bold",
            )


def _blended_glyph(ax, y: float, *, changed: bool) -> None:
    center = REP_X + REP_W / 2
    colors = (C_RETAINED, C_STATE, C_RETAINED, C_TARGET, C_STATE)
    offsets = (
        (-0.026, 0.002),
        (0.000, 0.009),
        (0.026, 0.002),
        (0.011, 0.016),
        (-0.015, 0.016),
    )
    for (dx, dy), color in zip(offsets, colors):
        ax.add_patch(
            mpatches.Circle(
                (center + dx, y + dy),
                0.022,
                facecolor=color,
                edgecolor=C_TARGET if color == C_TARGET else "none",
                linewidth=1.0,
                alpha=0.42,
                zorder=2,
            )
        )
    for index in range(3):
        x = center - 0.075 + index * 0.052
        ax.plot(
            [x, x + 0.030],
            [y - 0.036, y - 0.036],
            color=C_TARGET if changed and index > 0 else C_STATE,
            linewidth=1.6,
            linestyle="--" if changed and index > 0 else "-",
            solid_capstyle="round",
        )


def _row(ax, y: float, spec: dict) -> None:
    _box(
        ax,
        0.020,
        y,
        0.960,
        ROW_HEIGHT,
        C_ROW,
        edgecolor="#E1E5E9",
        linewidth=0.7,
        radius=0.018,
        zorder=0,
    )
    fit_text(
        ax,
        (REP_X, y + 0.126, REP_W, 0.062),
        spec["representation"],
        color=C_INK,
        fontsize=9.6,
    )
    fit_text(
        ax,
        (REP_X, y + 0.094, REP_W, 0.030),
        spec["secondary"],
        color=C_MUTED,
        fontsize=8.4,
        weight="normal",
    )
    if spec["addressable"]:
        _addressable_glyph(ax, y + 0.022)
    else:
        _blended_glyph(ax, y + 0.052, changed=spec["changed"])

    _box(
        ax,
        OPERATION_X,
        y + BOX_DY,
        OPERATION_W,
        BOX_H,
        C_TARGET_LIGHT,
        edgecolor=C_TARGET,
        linewidth=1.25,
        radius=0.018,
    )
    fit_text(
        ax,
        (OPERATION_X, y + BOX_DY, OPERATION_W, BOX_H),
        spec["operation"],
        color=C_TARGET,
        fontsize=10.2,
    )

    _box(
        ax,
        REFERENCE_X,
        y + BOX_DY,
        REFERENCE_W,
        BOX_H,
        "white",
        edgecolor=C_REFERENCE,
        linewidth=1.25,
        radius=0.018,
        linestyle="--",
    )
    fit_text(
        ax,
        (REFERENCE_X, y + BOX_DY, REFERENCE_W, BOX_H),
        spec["reference"],
        color=C_REFERENCE,
        fontsize=10.2,
    )

    _box(
        ax,
        VERIFIED_X,
        y + BOX_DY,
        VERIFIED_W,
        BOX_H,
        C_CERTIFICATE_LIGHT,
        edgecolor=C_CERTIFICATE,
        linewidth=1.25,
        radius=0.018,
    )
    fit_text(
        ax,
        (VERIFIED_X, y + BOX_DY, VERIFIED_W, BOX_H),
        spec["verified"],
        color=C_CERTIFICATE,
        fontsize=10.2,
    )

    middle = y + BOX_DY + BOX_H / 2
    _arrow(ax, (0.238, middle), (0.262, middle))
    _arrow(ax, (0.474, middle), (0.498, middle))
    _arrow(ax, (0.702, middle), (0.726, middle), color=C_CERTIFICATE)


def _taxonomy(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0.030, 1)
    ax.axis("off")
    ax.text(
        0.020,
        0.982,
        "(a) Where memory lives determines what deletion requires",
        color=C_INK,
        fontsize=11.6,
        weight="bold",
        va="top",
    )
    headers = (
        (REP_X, REP_W, "REPRESENTATION"),
        (OPERATION_X, OPERATION_W, "OPERATION"),
        (REFERENCE_X, REFERENCE_W, "COMPARISON"),
        (VERIFIED_X, VERIFIED_W, "WHAT IT SHOWS"),
    )
    for x, width, label in headers:
        fit_text(
            ax,
            (x, 0.906, width, 0.038),
            label,
            color=C_MUTED,
            fontsize=9.8,
        )
    for y, spec in zip(ROW_Y, ROWS):
        _row(ax, y, spec)

    _box(
        ax,
        0.020,
        0.134,
        0.960,
        0.066,
        C_CERTIFICATE_LIGHT,
        edgecolor=C_CERTIFICATE,
        linewidth=1.0,
        radius=0.014,
    )
    fit_text(
        ax,
        (0.020, 0.134, 0.960, 0.066),
        "REFIT \u2260 REBUILD \u2260 REPLAY",
        color=C_CERTIFICATE,
        fontsize=9.8,
    )
    _box(
        ax,
        0.020,
        0.046,
        0.960,
        0.066,
        C_STATE_LIGHT,
        edgecolor=C_LINE,
        linewidth=0.9,
        radius=0.014,
    )
    fit_text(
        ax,
        (0.020, 0.046, 0.960, 0.066),
        "NOT CHANGED   weights \u00b7 prior outputs \u00b7 external logs",
        color=C_MUTED,
        fontsize=9.4,
    )


def _ladder(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.985,
        "(b) Stronger claims require stronger references",
        ha="center",
        va="top",
        color=C_INK,
        fontsize=11.6,
        weight="bold",
    )
    height = 0.176
    for index, (title, verdict, face, edge) in enumerate(LADDER):
        y = 0.735 - index * 0.212
        _box(
            ax,
            0.115,
            y,
            0.845,
            height,
            face,
            edgecolor=edge,
            linewidth=1.15,
            radius=0.024,
        )
        fit_text(
            ax,
            (0.135, y + 0.092, 0.805, 0.062),
            title,
            color=edge,
            fontsize=9.8,
            align="left",
        )
        fit_text(
            ax,
            (0.135, y + 0.020, 0.805, 0.058),
            verdict,
            color=C_INK,
            fontsize=9.4,
            weight="normal",
            align="left",
        )
    ax.annotate(
        "",
        xy=(0.055, 0.945),
        xytext=(0.055, 0.045),
        arrowprops={"arrowstyle": "-|>", "color": C_MUTED, "linewidth": 1.2},
    )
    ax.text(
        0.036,
        0.495,
        "STRONGER CLAIM",
        color=C_MUTED,
        fontsize=8.6,
        weight="bold",
        ha="center",
        va="center",
        rotation=90,
    )


def build_figure():
    figure = plt.figure(figsize=(7.2, 8.2))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.6, 1.0), hspace=0.10)
    taxonomy = figure.add_subplot(grid[0])
    ladder = figure.add_subplot(grid[1])
    figure.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.015)
    _taxonomy(taxonomy)
    _ladder(ladder)
    return figure


def _add_svg_accessibility(path: Path, *, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "<svg "
    start = text.index(marker)
    end = text.index(">", start) + 1
    labelled = text[start:end].replace(
        marker,
        '<svg role="img" aria-labelledby="hero-title" '
        'aria-describedby="hero-desc" ',
        1,
    )
    accessible = (
        f"\n <title id=\"hero-title\">{escape(title)}</title>"
        f"\n <desc id=\"hero-desc\">{escape(description)}</desc>"
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
    common = {"bbox_inches": "tight", "pad_inches": 0.035}
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

"""Render the two-channel hybrid and rewind/replay schematic.

The artwork intentionally contains only the concepts needed for an ELI5 pass:
one fact can live in both memory channels, and exact removal restores a saved
checkpoint, skips the victim, and replays the surviving suffix.  Layer counts,
cache surfaces, and implementation qualifications belong in the caption.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "gemma-sv-kimi-arch-v2",
    }
)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb


ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT.parent
OUTPUT_BASE = RELEASE_ROOT / "paper/figs/kimi_arch"
FIGURE = OUTPUT_BASE.with_suffix(".pdf")
PREVIEW = RELEASE_ROOT / "outputs/kimi_sv/kimi_arch_figure.png"
TITLE = "Kimi replay removes one record from both active memory channels"
DESCRIPTION = (
    "A selected fact can affect attention and recurrent KDA state, so masking "
    "attention alone is not deletion. The evaluated replay path restores the "
    "last unaffected checkpoint, skips the selected record, and replays the "
    "stored suffix."
)

C_KDA = "#D9E0E8"
C_MLA = "#5B8DEF"
C_MLA_LIGHT = "#EAF0FC"
C_VICTIM = "#A34A42"
C_VICTIM_LIGHT = "#F8ECEC"
C_EXACT = "#2E7D5B"
C_EXACT_LIGHT = "#EDF7F2"
C_INK = "#202124"
C_MUTED = "#62676D"
C_LINE = "#B8BEC5"


def _box(ax, x, y, w, h, fc, ec=C_INK, lw=1.0, radius=0.015, **kwargs):
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        fc=fc,
        ec=ec,
        lw=lw,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, p0, p1, lw=1.5, color=C_INK, style="-", rad=0.0):
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=lw,
        color=color,
        linestyle=style,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def panel_channels(ax):
    ax.set_title(
        "(a) One fact, two live memory paths",
        fontsize=12.2,
        weight="bold",
        loc="left",
    )
    _box(
        ax,
        0.02,
        0.38,
        0.12,
        0.24,
        C_VICTIM_LIGHT,
        ec=C_VICTIM,
        lw=1.6,
        radius=0.025,
    )
    ax.text(
        0.08,
        0.50,
        "one\nfact",
        ha="center",
        va="center",
        color=C_VICTIM,
        fontsize=10,
        weight="bold",
    )

    _arrow(ax, (0.15, 0.50), (0.24, 0.68), color=C_MLA, rad=-0.08)
    _arrow(ax, (0.15, 0.50), (0.24, 0.30), color=C_MUTED, rad=0.08)

    ax.text(
        0.25,
        0.83,
        "ATTENTION",
        color="#2A4A8A",
        fontsize=10.0,
        weight="bold",
    )
    _box(
        ax,
        0.24,
        0.57,
        0.48,
        0.21,
        C_MLA_LIGHT,
        ec=C_MLA,
        lw=1.2,
        radius=0.025,
    )
    token_x = [0.28, 0.36, 0.44, 0.52, 0.60]
    for index, x in enumerate(token_x):
        victim = index == 2
        _box(
            ax,
            x,
            0.62,
            0.055,
            0.10,
            C_VICTIM_LIGHT if victim else "white",
            ec=C_VICTIM if victim else C_MLA,
            lw=1.5 if victim else 0.9,
            radius=0.012,
        )
        if victim:
            ax.text(
                x + 0.0275,
                0.67,
                "\u00d7",
                ha="center",
                va="center",
                color=C_VICTIM,
                fontsize=10,
                weight="bold",
            )

    ax.text(
        0.25,
        0.43,
        "KDA",
        color="#48556A",
        fontsize=10.0,
        weight="bold",
    )
    _box(
        ax,
        0.24,
        0.17,
        0.48,
        0.21,
        "#F3F5F7",
        ec="#7A8795",
        lw=1.2,
        radius=0.025,
    )
    circles = [
        (0.40, 0.27, "#8FA8C8"),
        (0.47, 0.25, C_VICTIM),
        (0.54, 0.28, "#8FA8C8"),
        (0.50, 0.31, "#AAB9CB"),
        (0.44, 0.32, "#AAB9CB"),
    ]
    for x, y, color in circles:
        ax.add_patch(
            mpatches.Circle(
                (x, y),
                0.062,
                facecolor=color,
                edgecolor=C_VICTIM if color == C_VICTIM else "none",
                linewidth=1.2,
                alpha=0.48,
            )
        )

    _arrow(ax, (0.73, 0.27), (0.87, 0.45), color=C_MUTED)
    ax.plot(
        [0.73, 0.84],
        [0.68, 0.55],
        color=C_VICTIM,
        linewidth=2.0,
        linestyle="--",
    )
    ax.text(
        0.79,
        0.63,
        "\u00d7 MASK",
        ha="center",
        va="center",
        color=C_VICTIM,
        fontsize=9.4,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    _box(
        ax,
        0.87,
        0.38,
        0.10,
        0.14,
        C_EXACT_LIGHT,
        ec=C_EXACT,
        lw=1.4,
        radius=0.03,
    )
    ax.text(
        0.92,
        0.45,
        "LIVE",
        ha="center",
        va="center",
        color=C_EXACT,
        fontsize=9.4,
        weight="bold",
    )
    ax.text(
        0.92,
        0.25,
        "MASK ONE\n\u2260 DELETE",
        ha="center",
        va="center",
        color=C_VICTIM,
        fontsize=9.4,
        weight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def panel_replay(ax):
    ax.set_title(
        "(b) Restore checkpoint \u2192 skip victim \u2192 replay suffix",
        fontsize=12.2,
        weight="bold",
        loc="left",
    )
    count = 6
    victim = 2
    x0, width, gap, y, height = 0.07, 0.105, 0.018, 0.57, 0.17
    boundaries = [x0 - gap / 2 + i * (width + gap) for i in range(count + 1)]
    for index in range(count):
        x = x0 + index * (width + gap)
        is_victim = index == victim
        is_suffix = index > victim
        _box(
            ax,
            x,
            y,
            width,
            height,
            (
                C_VICTIM_LIGHT
                if is_victim
                else C_MLA_LIGHT if is_suffix else C_KDA
            ),
            ec=C_VICTIM if is_victim else C_MLA if is_suffix else "#7A8795",
            lw=1.6 if is_victim else 0.9,
            radius=0.018,
        )
        if is_victim:
            ax.text(
                x + width / 2,
                y + height / 2,
                "\u00d7",
                ha="center",
                va="center",
                fontsize=12,
                color=C_VICTIM,
                weight="bold",
            )
    checkpoint = x0 + victim * (width + gap) - gap / 2
    ax.plot(
        [checkpoint, checkpoint],
        [y - 0.04, y + height + 0.05],
        color=C_EXACT,
        linewidth=1.2,
        linestyle=":",
    )
    ax.plot(checkpoint, y - 0.055, marker="^", markersize=7, color=C_EXACT)
    ax.text(
        checkpoint,
        y - 0.13,
        "checkpoint",
        color=C_EXACT,
        fontsize=9.4,
        ha="center",
        weight="bold",
    )
    ax.text(
        x0 + width,
        y + height + 0.09,
        "prefix",
        ha="center",
        color=C_MUTED,
        fontsize=9.4,
        weight="bold",
    )
    ax.text(
        x0 + victim * (width + gap) + width / 2,
        y + height + 0.09,
        "victim",
        ha="center",
        color=C_VICTIM,
        fontsize=9.4,
        weight="bold",
    )
    ax.text(
        x0 + 4.5 * (width + gap),
        y + height + 0.09,
        "suffix",
        ha="center",
        color=C_MLA,
        fontsize=9.4,
        weight="bold",
    )

    action_y = 0.17
    ax.plot(0.16, action_y + 0.055, marker="^", markersize=9, color=C_EXACT)
    ax.text(
        0.16,
        action_y - 0.09,
        "RESTORE",
        ha="center",
        color=C_EXACT,
        fontsize=9.4,
        weight="bold",
    )
    _arrow(ax, (0.23, action_y + 0.055), (0.34, action_y + 0.055), color=C_INK)
    _box(
        ax,
        0.36,
        action_y,
        0.10,
        0.11,
        C_VICTIM_LIGHT,
        ec=C_VICTIM,
        lw=1.4,
        radius=0.015,
    )
    ax.text(
        0.41,
        action_y + 0.055,
        "\u00d7",
        ha="center",
        va="center",
        color=C_VICTIM,
        fontsize=11,
        weight="bold",
    )
    ax.text(
        0.41,
        action_y - 0.09,
        "SKIP",
        ha="center",
        color=C_VICTIM,
        fontsize=9.4,
        weight="bold",
    )
    _arrow(ax, (0.48, action_y + 0.055), (0.57, action_y + 0.055), color=C_INK)
    for index in range(3):
        _box(
            ax,
            0.59 + index * 0.07,
            action_y,
            0.055,
            0.11,
            C_MLA_LIGHT,
            ec=C_MLA,
            lw=1.0,
            radius=0.01,
        )
    ax.text(
        0.66,
        action_y - 0.09,
        "REPLAY SUFFIX",
        ha="center",
        color="#2A4A8A",
        fontsize=9.4,
        weight="bold",
    )
    _arrow(ax, (0.80, action_y + 0.055), (0.86, action_y + 0.055), color=C_EXACT)
    _box(
        ax,
        0.87,
        action_y - 0.005,
        0.10,
        0.12,
        C_EXACT_LIGHT,
        ec=C_EXACT,
        lw=1.4,
        radius=0.025,
    )
    ax.text(
        0.92,
        action_y + 0.055,
        "REBUILT",
        ha="center",
        va="center",
        color=C_EXACT,
        fontsize=9.4,
        weight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def build_figure():
    figure = plt.figure(figsize=(6.8, 4.65))
    figure.suptitle(
        "A hybrid model has multiple live memory channels; masking one is not deletion",
        fontsize=12.6,
        weight="bold",
        y=0.985,
    )
    grid = figure.add_gridspec(2, 1, height_ratios=(1.08, 1.0), hspace=0.28)
    panel_channels(figure.add_subplot(grid[0]))
    panel_replay(figure.add_subplot(grid[1]))
    figure.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.035)
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
        figure_id="kimi-architecture",
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
    figure = build_figure()
    written = write_figure(figure)
    plt.close(figure)
    for path in (*written, PREVIEW):
        print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

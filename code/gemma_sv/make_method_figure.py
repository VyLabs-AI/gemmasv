"""Render the Gemma support-vector graft and its bounded output audit.

The figure follows the same progressive-disclosure order as the deletion
contract: edited surface, operation, reference, check, established claim, and
the most important non-claim.  Equations, layer counts, tolerances, and broader
scope qualifications remain in the caption.
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
        "svg.hashsalt": "gemma-sv-method-schematic-v3",
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
C_LOCAL = "#D9DFE6"
C_LOCAL_EDGE = "#7B8794"
C_INK = "#202124"
C_MUTED = "#62676D"
C_LINE = "#BBC2C9"
C_PANEL = "#FAFBFC"

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT.parent
OUTPUT_BASE = RELEASE_ROOT / "paper/figs/method_schematic"
PREVIEW: Path | None = None
SVG_TITLE = "Gemma support-vector deletion and retained-key output audit"
SVG_DESCRIPTION = (
    "A four-stage schematic. Only Gemma global layers receive support-vector "
    "memory. Selected target rows are decremented, with exact refit fallback. "
    "The executed edit and a retained-key refit receive the same registered "
    "prompt and produce matching next-token distributions. The check "
    "establishes output agreement with that retained-key reference, not a "
    "fresh raw-history-omitted rebuild."
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
        zorder=4,
    )
    ax.add_patch(arrow)
    return arrow


def _stage_label(ax, x: float, y: float, number: int, label: str) -> None:
    ax.text(
        x,
        y,
        f"{number}  {label}",
        color=C_MUTED,
        fontsize=10.7,
        weight="bold",
        va="bottom",
    )


def _layer_group(ax, x: float, y: float) -> None:
    width, height, gap = 0.047, 0.019, 0.005
    for index in range(6):
        global_layer = index == 5
        _box(
            ax,
            x,
            y + index * (height + gap),
            width,
            height,
            C_RETAINED if global_layer else C_LOCAL,
            edgecolor=C_RETAINED if global_layer else C_LOCAL_EDGE,
            linewidth=1.15 if global_layer else 0.65,
            radius=0.004,
        )
    ax.text(
        x + width + 0.008,
        y + 2.1 * (height + gap),
        "local",
        color=C_LOCAL_EDGE,
        fontsize=10.1,
        va="center",
    )
    ax.text(
        x + width + 0.008,
        y + 5 * (height + gap) + height / 2,
        "global",
        color=C_RETAINED,
        fontsize=10.1,
        weight="bold",
        va="center",
    )


def _memory_rows(
    ax,
    x: float,
    y: float,
    *,
    ghost_target: bool,
    reference: bool = False,
) -> None:
    width, height, gap = 0.035, 0.060, 0.009
    for index in range(5):
        selected = index == 2
        xx = x + index * (width + gap)
        if selected and ghost_target:
            _box(
                ax,
                xx,
                y,
                width,
                height,
                "white",
                edgecolor=C_TARGET,
                linewidth=1.2,
                radius=0.007,
                linestyle=":",
            )
            ax.text(
                xx + width / 2,
                y + height / 2,
                "×",
                color=C_TARGET,
                fontsize=10.1,
                weight="bold",
                ha="center",
                va="center",
            )
            continue
        edge = C_REFERENCE if reference else C_RETAINED
        _box(
            ax,
            xx,
            y,
            width,
            height,
            C_REFERENCE_LIGHT if reference else C_RETAINED_LIGHT,
            edgecolor=edge,
            linewidth=0.9,
            radius=0.007,
        )


def _compact_memory_rows(ax, x: float, y: float) -> None:
    width, height, gap = 0.022, 0.048, 0.006
    for index in range(5):
        selected = index == 2
        xx = x + index * (width + gap)
        _box(
            ax,
            xx,
            y,
            width,
            height,
            C_TARGET_LIGHT if selected else C_RETAINED_LIGHT,
            edgecolor=C_TARGET if selected else C_RETAINED,
            linewidth=1.3 if selected else 0.9,
            radius=0.006,
        )
        if selected:
            ax.text(
                xx + width / 2,
                y + height / 2,
                "×",
                color=C_TARGET,
                fontsize=10.1,
                weight="bold",
                ha="center",
                va="center",
            )


def _distribution(ax, x: float, y: float) -> None:
    heights = (0.026, 0.070, 0.105, 0.052, 0.021)
    width, gap = 0.017, 0.008
    ax.plot(
        [x - 0.008, x + len(heights) * (width + gap) - gap + 0.008],
        [y, y],
        color=C_MUTED,
        linewidth=0.7,
    )
    for index, height in enumerate(heights):
        ax.add_patch(
            mpatches.Rectangle(
                (x + index * (width + gap), y),
                width,
                height,
                facecolor=C_CERTIFICATE,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
        )


def _audit_lane(
    ax,
    *,
    y: float,
    label: str,
    label_color: str,
    ghost_target: bool,
    reference: bool,
) -> None:
    ax.text(
        0.058,
        y + 0.079,
        label,
        color=label_color,
        fontsize=10.1,
        weight="bold",
        va="bottom",
    )
    _memory_rows(
        ax,
        0.058,
        y,
        ghost_target=ghost_target,
        reference=reference,
    )
    _arrow(ax, (0.282, y + 0.030), (0.337, y + 0.030))
    _box(
        ax,
        0.347,
        y - 0.008,
        0.102,
        0.078,
        "#F4F6F8",
        edgecolor=C_LINE,
        linewidth=1.0,
        radius=0.014,
    )
    ax.text(
        0.398,
        y + 0.031,
        "frozen\nmodel",
        color=C_INK,
        fontsize=10.1,
        ha="center",
        va="center",
        linespacing=1.1,
    )
    _arrow(ax, (0.459, y + 0.030), (0.505, y + 0.030))
    _distribution(ax, 0.520, y)


def build_figure():
    figure, ax = plt.subplots(figsize=(7.0, 4.65))
    _stage_label(ax, 0.025, 0.935, 1, "STORE")
    _stage_label(ax, 0.385, 0.935, 2, "DELETE")
    _stage_label(ax, 0.620, 0.935, 3, "REFERENCE")

    _box(
        ax,
        0.025,
        0.620,
        0.323,
        0.213,
        C_PANEL,
        edgecolor="#E1E5E9",
        linewidth=0.8,
        radius=0.018,
    )
    _layer_group(ax, 0.048, 0.647)
    ax.text(0.153, 0.714, "…", color=C_MUTED, fontsize=14, va="center")
    _arrow(ax, (0.183, 0.708), (0.198, 0.708), color=C_RETAINED)
    ax.text(
        0.200,
        0.793,
        "addressable\nmemory rows",
        color=C_RETAINED,
        fontsize=10.1,
        weight="bold",
        va="top",
        linespacing=1.08,
    )
    _compact_memory_rows(ax, 0.200, 0.653)

    _arrow(ax, (0.350, 0.725), (0.378, 0.725), color=C_TARGET)
    _box(
        ax,
        0.385,
        0.665,
        0.192,
        0.132,
        C_TARGET_LIGHT,
        edgecolor=C_TARGET,
        linewidth=1.35,
        radius=0.022,
    )
    ax.text(
        0.481,
        0.752,
        "remove rows",
        ha="center",
        va="center",
        color=C_TARGET,
        fontsize=10.1,
        weight="bold",
    )
    ax.text(
        0.481,
        0.702,
        "exact-refit fallback",
        ha="center",
        va="center",
        color=C_TARGET,
        fontsize=10.1,
    )

    _arrow(ax, (0.587, 0.725), (0.612, 0.725), color=C_REFERENCE)
    _box(
        ax,
        0.620,
        0.665,
        0.355,
        0.132,
        "white",
        edgecolor=C_REFERENCE,
        linewidth=1.35,
        radius=0.022,
        linestyle="--",
    )
    ax.text(
        0.7975,
        0.752,
        "exact refit",
        ha="center",
        va="center",
        color=C_REFERENCE,
        fontsize=10.1,
        weight="bold",
    )
    ax.text(
        0.7975,
        0.702,
        "retained contextualized rows",
        ha="center",
        va="center",
        color=C_REFERENCE,
        fontsize=10.1,
    )
    _stage_label(ax, 0.025, 0.560, 4, "COMPARE — SAME REGISTERED PROMPT")
    _box(
        ax,
        0.025,
        0.218,
        0.950,
        0.320,
        C_PANEL,
        edgecolor="#E1E5E9",
        linewidth=0.8,
        radius=0.018,
        zorder=0,
    )
    _audit_lane(
        ax,
        y=0.394,
        label="executed edit",
        label_color=C_TARGET,
        ghost_target=True,
        reference=False,
    )
    _audit_lane(
        ax,
        y=0.259,
        label="retained-key refit",
        label_color=C_REFERENCE,
        ghost_target=True,
        reference=True,
    )
    ax.plot(
        [0.665, 0.665],
        [0.276, 0.497],
        color=C_CERTIFICATE,
        linewidth=1.2,
    )
    ax.plot(
        [0.645, 0.665],
        [0.424, 0.424],
        color=C_CERTIFICATE,
        linewidth=1.2,
    )
    ax.plot(
        [0.645, 0.665],
        [0.289, 0.289],
        color=C_CERTIFICATE,
        linewidth=1.2,
    )
    _arrow(ax, (0.665, 0.357), (0.728, 0.357), color=C_CERTIFICATE)
    _box(
        ax,
        0.728,
        0.305,
        0.224,
        0.104,
        C_CERTIFICATE_LIGHT,
        edgecolor=C_CERTIFICATE,
        linewidth=1.4,
        radius=0.018,
        zorder=3,
    )
    ax.text(
        0.840,
        0.357,
        "OUTPUTS AGREE\nwithin tolerance",
        color=C_CERTIFICATE,
        fontsize=10.7,
        weight="bold",
        ha="center",
        va="center",
        linespacing=1.08,
    )
    ax.plot([0.840, 0.840], [0.305, 0.185], color=C_CERTIFICATE, linewidth=1.2)
    _arrow(ax, (0.840, 0.185), (0.565, 0.185), color=C_CERTIFICATE)
    _arrow(ax, (0.840, 0.185), (0.790, 0.185), color=C_MUTED)
    ax.text(
        0.840,
        0.438,
        "5  RESULT",
        color=C_MUTED,
        fontsize=10.7,
        weight="bold",
        ha="center",
        va="bottom",
    )

    _box(
        ax,
        0.025,
        0.060,
        0.550,
        0.103,
        C_CERTIFICATE_LIGHT,
        edgecolor=C_CERTIFICATE,
        linewidth=1.05,
        radius=0.016,
    )
    fit_text(
        ax,
        (0.025, 0.115, 0.550, 0.044),
        "ESTABLISHES",
        color=C_CERTIFICATE,
        fontsize=10.7,
        align="left",
    )
    fit_text(
        ax,
        (0.025, 0.064, 0.550, 0.048),
        "matches the exact refit",
        color=C_CERTIFICATE,
        fontsize=10.1,
        align="left",
    )
    _box(
        ax,
        0.598,
        0.060,
        0.377,
        0.103,
        "white",
        edgecolor=C_LINE,
        linewidth=1.0,
        radius=0.016,
        linestyle="--",
    )
    fit_text(
        ax,
        (0.598, 0.115, 0.377, 0.044),
        "NOT ESTABLISHED",
        color=C_MUTED,
        fontsize=10.7,
        align="left",
    )
    fit_text(
        ax,
        (0.598, 0.064, 0.377, 0.048),
        "fresh-history equality",
        color=C_MUTED,
        fontsize=10.1,
        align="left",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0.035, 1)
    ax.axis("off")
    figure.subplots_adjust(left=0.012, right=0.988, top=0.988, bottom=0.018)
    return figure


def _add_svg_accessibility(path: Path, *, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "<svg "
    start = text.index(marker)
    end = text.index(">", start) + 1
    labelled = text[start:end].replace(
        marker,
        '<svg role="img" aria-labelledby="method-title" '
        'aria-describedby="method-desc" ',
        1,
    )
    accessible = (
        f"\n <title id=\"method-title\">{escape(title)}</title>"
        f"\n <desc id=\"method-desc\">{escape(description)}</desc>"
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
    main()

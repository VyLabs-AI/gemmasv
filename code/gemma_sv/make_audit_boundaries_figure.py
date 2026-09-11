"""Render Gemma scale sensitivity from tracked source-free aggregates."""
from __future__ import annotations

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
        "svg.hashsalt": "gemma-sv-audit-boundaries-v3",
    }
)
import matplotlib.pyplot as plt

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb


PACKAGE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads(
    (PACKAGE / "benchmarks" / "iclr_audit_boundaries_v2.json").read_text()
)
C_EDIT = "#A84D45"
C_REFERENCE = "#2E7D5B"
GRAY = "#9aa3ad"
INK = "#263238"
TITLE = "Gemma scale sensitivity under one frozen support-vector recipe"
DESCRIPTION = (
    "At the evaluated Gemma checkpoints, only 4B has base-matched graft "
    "admission and low paired perplexity overhead. Admission falls from seven "
    "of eight to three of eight at 1B, while 12B incurs 11.699 percent "
    "perplexity overhead."
)


def _rate(pair: list[int]) -> float:
    return 100.0 * pair[0] / pair[1]


def admission_panel(ax) -> None:
    admission = DATA["admission"]
    rows = [
        ("1B base admission", admission["one_b_ungrafted_records"], GRAY, None),
        ("1B edited-memory graft", admission["one_b_grafted_records"], C_EDIT, None),
        ("4B base admission", admission["four_b_ungrafted_records"], GRAY, None),
        ("4B edited-memory graft", admission["four_b_grafted_records"], C_EDIT, None),
        ("12B base admission", admission["twelve_b_ungrafted_records"], GRAY, None),
        ("12B edited-memory graft", admission["twelve_b_grafted_records"], C_EDIT, None),
    ]
    labels = [row[0] for row in rows]
    values = [_rate(row[1]) for row in rows]
    colors = [row[2] for row in rows]
    y = list(reversed(range(len(rows))))
    bars = ax.barh(y, values, color=colors, height=0.62)
    for bar, row in zip(bars, rows):
        if "graft" in row[0]:
            bar.set_hatch("//")
    for yi, value, (_, pair, _, display) in zip(y, values, rows):
        text = display or f"{pair[0]}/{pair[1]}"
        if value > 88:
            ax.text(value - 2.0, yi, text, ha="right", va="center",
                    color="white", fontsize=11.5, weight="bold")
        else:
            ax.text(value + 2.0, yi, text, ha="left", va="center",
                    color=INK, fontsize=11.5, weight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 112)
    ax.set_xlabel("strict admission (%)")
    ax.set_title("(a) 4B is the only tested scale preserving baseline eligibility",
                 loc="left", fontsize=14.5, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11.5)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.grid(axis="x", color="#e5e8eb", lw=0.7)
    ax.set_axisbelow(True)


def quality_panel(ax) -> None:
    quality = DATA["quality"]
    rows = [
        ("1B", quality["one_b"]),
        ("4B", quality["four_b"]),
        ("12B", quality["twelve_b"]),
    ]
    labels = [label for label, _ in rows]
    values = [row["paired_overhead_percent"] for _, row in rows]
    colors = [GRAY, C_REFERENCE, C_EDIT]
    bars = ax.bar(labels, values, color=colors, width=0.58)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.32,
            f"+{value:.3f}%",
            ha="center",
            va="bottom",
            color=INK,
            fontsize=11.5,
            weight="bold",
        )
    ax.set_ylim(0, 13.4)
    ax.set_ylabel("paired perplexity overhead (%)")
    ax.set_title(
        "(b) 12B pays the largest measured quality cost",
        loc="left",
        fontsize=14.5,
        weight="bold",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.grid(axis="y", color="#e5e8eb", lw=0.7)
    ax.set_axisbelow(True)


def build_figure():
    plt.rcParams.update({"text.color": INK})
    fig, axes = plt.subplots(2, 1, figsize=(7.7, 6.0))
    admission_panel(axes[0])
    quality_panel(axes[1])
    fig.tight_layout(h_pad=3.0)
    return fig


def write_figure(
    figure,
    output_base: Path = ROOT / "paper" / "figs" / "audit_boundaries",
) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(svg, metadata={"Date": None, "Description": DESCRIPTION}, **common)
    add_svg_accessibility(
        svg,
        figure_id="audit-boundaries",
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
    fig = build_figure()
    written = write_figure(fig)
    plt.close(fig)
    for path in written:
        print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

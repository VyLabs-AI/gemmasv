"""Consolidated certificate/behavior figure for the memory-deletion paper.

Six panels assembled from the committed eval runs on the recovered gemma-3-1b (provenance noted
per panel). Panels (c,e) use the float64 decrement certificate; behavioral
panels (a,b,d,f) use the single-precision masked-refit proxy.

    .venv311/bin/python -m gemma_sv.make_forget_figure  ->  outputs/gemma_sv_eval/forgetting_axis.png
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb

RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(
    os.environ.get("GEMMASV_OUTPUT_ROOT", RELEASE_ROOT / "outputs")
).expanduser()

# ---- data from the committed runs (means ± 95% CI) -----------------------------------------
# P2 recovery vs #ICL-shots (unlearn_attack, 60 targets, fbd8ce8)
P2_X = [0, 1, 2, 4, 8]
P2 = {"sv_exact": ([0.00, 0.02, -0.04, 0.03, 0.02], [0.01, 0.05, 0.05, 0.05, 0.06]),
      "decay":    ([0.02, 0.05, -0.00, 0.04, 0.03], [0.02, 0.05, 0.05, 0.05, 0.06]),
      "icul":     ([0.95, 0.88, 0.72, 0.61, 0.56], [0.07, 0.09, 0.09, 0.08, 0.10])}
# P3 recovery vs #fine-tune-samples (unlearn_relearn, ~48 targets, 694d454)
P3_X = [0, 4, 16, 64, 256]
P3 = {"sv_exact": ([0.01, 0.00, 0.00, -0.02, -0.10], [0.03, 0.02, 0.01, 0.06, 0.17]),
      "decay":    ([0.05, 0.02, 0.04, -0.03, -0.10], [0.03, 0.06, 0.02, 0.07, 0.26]),
      "icul":     ([1.14, 1.16, 1.14, 1.24, 1.23], [0.13, 0.20, 0.07, 0.21, 0.12])}
# Sequential-deletion stability (unlearn_output_demo --sequential, §5j, 7d0237a)
SEQ_K = [1, 2, 5, 10, 20, 30]
SEQ_EXACT = [5.16e-15, 4.61e-15, 4.67e-15, 1.49e-14, 3.84e-15, 6.70e-15]
SEQ_DECAY = [6.72e-6, 1.39e-5, 7.64e-4, 1.86e-3, 2.49e-3, 4.69e-3]
# Forget efficacy + retain specificity (unlearn_eval, 36 targets, 25cc94d)
EFFICACY, EFFICACY_CI = 1.08, 0.09           # 1.0 = reaches never-ingested floor
RETAIN, RETAIN_CI = 0.027, 0.011             # 0 = no collateral
# Output-level KL distribution (unlearn_output_demo --trials, §5h, 4834e85)
KL_FORGET_MED, KL_FORGET_WORST = 5.4e-15, 9.3e-14
KL_DECAY_MED = 1.8e-6
# FULL LiRA MIA AUC(vs never): shadow contexts, per-target Gaussians, likelihood ratio;
# 39 targets x 8 held-out draws = 312 tests/side, 32 shadows/side (unlearn_mia_lira)
MIA = {"present": 0.996, "sv_exact": 0.499, "decay": 0.533, "icul": 0.989}
MIA_TPR1 = {"present": 0.920, "sv_exact": 0.013, "decay": 0.016, "icul": 0.859}

COLORS = {"sv_exact": "#1f77b4", "decay": "#d62728", "icul": "#ff7f0e"}
LABELS = {
    "sv_exact": "masked refit (behavior)",
    "decay": "decay (α×0.01)",
    "icul": "instruction-only",
}


def _legacy_main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.8})

    overlap_note = "masked refit ≈ decay\n(overlap at floor)"

    fig, ax = plt.subplots(2, 3, figsize=(9.2, 6.0))

    def curve(a, X, D, xlabel, title, categorical=False):
        xp = list(range(len(X))) if categorical else X
        for c in ("sv_exact", "decay", "icul"):
            m, e = np.array(D[c][0]), np.array(D[c][1])
            a.plot(xp, m, "-o", color=COLORS[c], label=LABELS[c], ms=4)
            a.fill_between(xp, m - e, m + e, color=COLORS[c], alpha=0.16)
        if categorical:
            a.set_xticks(xp)
            a.set_xticklabels(X)
        a.axhline(0, color="gray", ls=":", lw=1)
        a.set_xlabel(xlabel)
        a.set_ylabel("recovery (0=never-ingested, 1=full)")
        a.set_title(title)
        a.legend(frameon=False, fontsize=8.3)
        a.text(xp[len(xp) // 2], 0.16, overlap_note, fontsize=8.0, color="#555", ha="center")

    curve(ax[0, 0], P2_X, P2, "attacker ICL budget (# shots)", "(a) in-context attack")
    curve(ax[0, 1], P3_X, P3, "relearning budget (# fine-tune samples)",
          "(b) relearning attack", categorical=True)

    # (c) sequential stability -- exactness certificate
    a = ax[0, 2]
    a.plot(SEQ_K, SEQ_EXACT, "-o", color=COLORS["sv_exact"], label="float64 decrement", ms=4)
    a.plot(SEQ_K, SEQ_DECAY, "-o", color=COLORS["decay"], label="decay", ms=4)
    a.set_yscale("log")
    a.set_xlabel("# sequential deletions")
    a.set_ylabel("KL to fixed-$C$ retained-key refit (nats)")
    a.set_title("(c) sequential stability [certificate]")
    a.legend(frameon=False, fontsize=8.3, loc="center right")
    a.annotate("decay\naccumulates 700×", xy=(30, SEQ_DECAY[-1]), xytext=(9, 1.5e-4),
               fontsize=8, fontweight="bold", color=COLORS["decay"],
               arrowprops=dict(arrowstyle="->", color=COLORS["decay"]))
    a.text(8, 4e-14, "decrement flat ~1e-14", fontsize=8, fontweight="bold", color=COLORS["sv_exact"])

    # (d) forget efficacy + retain specificity (dotted target lines at 1.0 and 0.0)
    a = ax[1, 0]
    a.bar([0], [EFFICACY], yerr=[EFFICACY_CI], color=COLORS["sv_exact"], width=0.55, capsize=4)
    a.bar([1], [RETAIN], yerr=[RETAIN_CI], color="#2ca02c", width=0.55, capsize=4)
    a.axhline(1.0, color="gray", ls="--", lw=1)
    a.axhline(0.0, color="gray", ls="--", lw=1)
    a.set_xticks([0, 1])
    a.set_xticklabels(["forget\nefficacy\n(→1)", "retain\nΔES\n(→0)"], fontsize=8)
    a.set_ylim(-0.2, 1.3)
    a.set_title("(d) efficacy + specificity")

    # (e) output-level exactness (forget vs decay) -- exactness certificate
    a = ax[1, 1]
    a.bar([0], [KL_FORGET_WORST], color=COLORS["sv_exact"], width=0.55)
    a.bar([1], [KL_DECAY_MED], color=COLORS["decay"], width=0.55)
    a.set_yscale("log")
    a.set_ylim(1e-15, 1e-3)
    a.set_xticks([0, 1])
    a.set_xticklabels(["float64 decrement\n(worst)", "decay\n(median)"], fontsize=8)
    a.set_ylabel("KL to fixed-$C$ retained-key refit (nats)")
    a.set_title("(e) output exactness [certificate]")
    a.annotate("~2e7×", xy=(0.5, 3e-10), fontsize=12, fontweight="bold", ha="center", color="#333")

    # (f) FULL LiRA MIA: AUC bars + TPR@1%FPR annotated per bar
    a = ax[1, 2]
    mc = {"present": "#7f7f7f", "sv_exact": COLORS["sv_exact"], "decay": COLORS["decay"],
          "icul": COLORS["icul"]}
    keys = ["present", "sv_exact", "decay", "icul"]
    bars = a.bar(range(4), [MIA[k] for k in keys], color=[mc[k] for k in keys], width=0.6)
    for i, k in enumerate(keys):
        a.text(i, MIA[k] + 0.015, f"TPR@1%\n{MIA_TPR1[k]:.1%}", fontsize=7.2, ha="center",
               color="#333")
    a.axhline(0.5, color="gray", ls=":", lw=1)
    a.text(3.35, 0.485, "chance", fontsize=7, color="#555", ha="right", va="top")
    a.set_xticks(range(4))
    a.set_xticklabels(
        ["present", "masked refit", "decay", "instruction-only"],
        fontsize=8,
        rotation=20,
    )
    a.set_ylim(0.4, 1.12)
    a.set_ylabel("full-LiRA AUC (0.5 = chance)")
    a.set_title("(f) membership inference [full LiRA]")

    fig.tight_layout()
    out = OUTPUT_ROOT / "gemma_sv_eval/forgetting_axis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(RELEASE_ROOT / "paper/figs/forgetting_axis.png", dpi=220, bbox_inches="tight")
    print(f"saved -> {out}")

    return 0


TITLE = "Stronger behavioral audits separate masked refit from prompt-only suppression"
DESCRIPTION = (
    "Four measured panels compare target-free elicitation, related-data "
    "relearning, target suppression with retained specificity, and membership "
    "inference. Masked refit remains near the never-ingested behavioral floor "
    "while instruction-only suppression is recoverable."
)


def build_public_figure():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gemma-sv-forgetting-axis-v3",
        }
    )
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))

    def attack_panel(axis, x, rows, xlabel, heading, categorical=False):
        positions = list(range(len(x))) if categorical else x
        for condition in ("sv_exact", "decay", "icul"):
            mean = np.asarray(rows[condition][0])
            error = np.asarray(rows[condition][1])
            axis.plot(
                positions,
                mean,
                marker={"sv_exact": "o", "decay": "s", "icul": "X"}[condition],
                linestyle={"sv_exact": "-", "decay": "--", "icul": "-."}[condition],
                color=COLORS[condition],
                linewidth=1.5,
                markersize=5.0,
                markerfacecolor="white" if condition == "sv_exact" else COLORS[condition],
                label=LABELS[condition],
            )
            axis.fill_between(
                positions,
                mean - error,
                mean + error,
                color=COLORS[condition],
                alpha=0.10,
            )
        if categorical:
            axis.set_xticks(positions, [str(value) for value in x])
        axis.axhline(0, color="#555555", linestyle=":", linewidth=0.8)
        axis.set_xlabel(xlabel, fontsize=10.5)
        axis.set_ylabel("normalized recovery (toward 0)", fontsize=10.5)
        axis.set_title(heading, loc="left", fontsize=12.5, weight="bold")
        axis.tick_params(labelsize=10.3)
        axis.grid(color="#E1E5E9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    attack_panel(
        axes[0, 0],
        P2_X,
        P2,
        "target-free shots",
        "(a) Elicitation",
    )
    attack_panel(
        axes[0, 1],
        P3_X,
        P3,
        "related-data updates",
        "(b) Relearning",
        categorical=True,
    )
    axes[0, 1].legend(
        loc="upper center",
        bbox_to_anchor=(-0.08, 1.30),
        ncol=3,
        frameon=False,
        fontsize=10.3,
    )

    specificity = axes[1, 0]
    specificity.bar(
        [0],
        [EFFICACY],
        yerr=[EFFICACY_CI],
        color=COLORS["sv_exact"],
        width=0.55,
        capsize=4,
    )
    specificity.bar(
        [1],
        [RETAIN],
        yerr=[RETAIN_CI],
        color="#2E7D5B",
        width=0.55,
        capsize=4,
    )
    specificity.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
    specificity.axhline(0.0, color="#555555", linestyle=":", linewidth=0.8)
    specificity.set_xticks(
        [0, 1],
        ["target suppression\n(toward 1)", "retained drift\n(toward 0)"],
    )
    specificity.set_ylim(-0.2, 1.3)
    specificity.set_title(
        "(c) Suppression and specificity",
        loc="left",
        fontsize=12.5,
        weight="bold",
    )
    specificity.tick_params(labelsize=10.3)
    specificity.grid(axis="y", color="#E1E5E9", linewidth=0.7)
    specificity.spines[["top", "right"]].set_visible(False)

    membership = axes[1, 1]
    keys = ["present", "sv_exact", "decay", "icul"]
    labels = ["present", "masked refit", "decay", "prompt only"]
    colors = ["#666666", COLORS["sv_exact"], COLORS["decay"], COLORS["icul"]]
    membership.bar(range(4), [MIA[key] for key in keys], color=colors, width=0.62)
    for index, key in enumerate(keys):
        membership.text(
            index,
            MIA[key] + 0.018,
            f"TPR@1% {MIA_TPR1[key]:.1%}",
            ha="center",
            fontsize=10.3,
        )
    membership.axhline(0.5, color="#555555", linestyle=":", linewidth=0.8)
    membership.set_xticks(range(4), labels)
    membership.set_ylim(0.4, 1.14)
    membership.set_ylabel("LiRA AUC (0.5 = chance)", fontsize=10.5)
    membership.set_title(
        "(d) Membership inference",
        loc="left",
        fontsize=12.5,
        weight="bold",
    )
    membership.tick_params(axis="x", labelrotation=20, labelsize=10.3)
    membership.tick_params(axis="y", labelsize=10.3)
    membership.grid(axis="y", color="#E1E5E9", linewidth=0.7)
    membership.spines[["top", "right"]].set_visible(False)

    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.90,
        bottom=0.10,
        wspace=0.32,
        hspace=0.48,
    )
    return figure


def write_public_figure(
    figure,
    output_base: Path = RELEASE_ROOT / "paper/figs/forgetting_axis",
) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(svg, metadata={"Date": None, "Description": DESCRIPTION}, **common)
    add_svg_accessibility(
        svg,
        figure_id="forgetting-axis",
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

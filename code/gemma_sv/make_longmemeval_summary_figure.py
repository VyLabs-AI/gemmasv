"""Render the appendix LongMemEval deletion summary from the compact artifact."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 14.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "gemma-sv-longmemeval-summary-v1",
    }
)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb
from gemma_sv.figure_text import fit_text


PACKAGE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PACKAGE / "benchmarks/longmemeval_chat_geometry_methods_compact16_v1.json"
OUTPUT_BASE = ROOT / "paper/figs/longmemeval_summary"

C_EDIT = "#A84D45"
C_EDIT_LIGHT = "#F8ECEA"
C_REFERENCE = "#2E7D5B"
C_REFERENCE_LIGHT = "#EDF7F2"
C_PRESENT = "#686D72"
C_PRESENT_LIGHT = "#F1F2F3"
C_PROMPT = "#D98B20"
C_PROMPT_LIGHT = "#FFF3DF"
C_CERTIFICATE = "#6D5A91"
C_CERTIFICATE_LIGHT = "#F1EEF7"
C_INK = "#202124"
C_MUTED = "#62676D"
C_GRID = "#DDE2E7"

TITLE = (
    "The fresh-omission gap is distinct from implementation conformance"
)
DESCRIPTION = (
    "Panel a shows 8.33 nats of target suppression for edited memory versus "
    "0.53 nats for prompt-only suppression. Panel b reports 8.05e-5 nats of "
    "mean absolute retained-answer log-probability change and 1.90e-6 nats of "
    "retained first-token KL to fresh omission. Panel c headlines the 2.15-nat "
    "policy-to-fresh-omission gap. A smaller label reports the 1.75e-16 "
    "implementation check between the executed refit path and an independently "
    "constructed retained-key refit."
)
EXPECTED_SCHEMA = "gemma-sv-longmemeval-chat-geometry-methods-compact16-v1"
EXPECTED_INTEGRITY = "58583271ba6df3004c8d529ef9e39d506796377bd9b40094a23bc0fa36aba978"
EXPECTED_FINAL_REPORT_SHA256 = (
    "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6"
)


def load_data(path: Path = DATA_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_data(data: dict) -> dict:
    if data.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("LongMemEval compact schema differs")
    if data.get("status") != "validated-complete-all16":
        raise ValueError("LongMemEval compact result is not complete")
    if data.get("integrity", {}).get("sha256") != EXPECTED_INTEGRITY:
        raise ValueError("LongMemEval compact integrity differs")
    if data.get("contains_source_text") is not False:
        raise ValueError("LongMemEval figure input contains source text")
    if data.get("contains_model_generated_text") is not False:
        raise ValueError("LongMemEval figure input contains generated text")
    denominators = data.get("denominators", {})
    if (
        denominators.get("frozen_records") != 16
        or denominators.get("joint_admitted_records") != 10
        or denominators.get("certificate_probes") != 32
    ):
        raise ValueError("LongMemEval figure denominators differ")
    exact_policy = data.get("exact_policy", {})
    if (
        exact_policy.get("fixed_c_refit_fallback_records") != 16
        or exact_policy.get("incremental_float64_decrement_records") != 0
        or exact_policy.get("source_full_repack_fallback_records") != 0
    ):
        raise ValueError("LongMemEval executed policy differs")
    certificate = data.get("certificate", {})
    if (
        certificate.get("reference") != "fixed-C retained-key refit"
        or certificate.get("raw_history_equality_claimed") is not False
        or certificate.get("probes") != 32
    ):
        raise ValueError("LongMemEval certificate scope differs")
    if (
        data.get("source_artifacts", {})
        .get("final_report", {})
        .get("file_sha256")
        != EXPECTED_FINAL_REPORT_SHA256
    ):
        raise ValueError("LongMemEval final-report binding differs")
    scope = data.get("scope", {})
    if (
        scope.get("no_raw_history_state_equality_claim") is not True
        or scope.get("no_incremental_speed_claim") is not True
        or scope.get("not_an_official_leaderboard_score") is not True
    ):
        raise ValueError("LongMemEval publication scope differs")

    conditions = data["efficacy"]["conditions"]
    exact = conditions["exact_decrement"]
    exact_summary = data["efficacy"]["exact_policy_summary"]
    paired_values = (
        (
            exact["mean_target_suppression_vs_present_nats"],
            exact_summary["mean_target_suppression_nats"],
        ),
        (
            exact["mean_absolute_retained_log_probability_drift_from_raw_omission_nats"],
            exact_summary["mean_absolute_retained_log_probability_drift_nats"],
        ),
        (
            exact["mean_retained_first_token_kl_raw_omission_to_method_nats"],
            exact_summary["mean_retained_first_token_kl_to_raw_omission_nats"],
        ),
        (
            exact["mean_target_first_token_kl_raw_omission_to_method_nats"],
            exact_summary["mean_target_first_token_kl_to_raw_omission_nats"],
        ),
    )
    if any(left != right for left, right in paired_values):
        raise ValueError("LongMemEval exact summary disagrees with condition metrics")
    return {
        "suppression": {
            "record present": conditions["present"][
                "mean_target_suppression_vs_present_nats"
            ],
            "prompt-only": conditions["prompt_suppression"][
                "mean_target_suppression_vs_present_nats"
            ],
            "fresh omission": conditions["fresh_raw_omission"][
                "mean_target_suppression_vs_present_nats"
            ],
            "edited memory": exact["mean_target_suppression_vs_present_nats"],
        },
        "retained_abs_drift": exact[
            "mean_absolute_retained_log_probability_drift_from_raw_omission_nats"
        ],
        "retained_kl": exact[
            "mean_retained_first_token_kl_raw_omission_to_method_nats"
        ],
        "target_kl": exact[
            "mean_target_first_token_kl_raw_omission_to_method_nats"
        ],
        "certificate_kl": data["certificate"]["maximum_kl_nats"],
        "fallback_records": exact_policy["fixed_c_refit_fallback_records"],
        "frozen_records": denominators["frozen_records"],
        "records": data["efficacy"]["records"],
        "certificate_probes": data["certificate"]["probes"],
    }


def _card(ax, x, y, width, height, face, edge, title, value) -> None:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
        )
    )
    fit_text(
        ax,
        (x, y + 0.54 * height, width, 0.34 * height),
        title,
        color=edge,
        fontsize=11.2,
        align="left",
    )
    fit_text(
        ax,
        (x, y + 0.10 * height, width, 0.40 * height),
        value,
        color=edge,
        fontsize=16.0,
        align="left",
    )


def suppression_panel(ax, summary: dict) -> None:
    labels = ["record\npresent", "prompt-only", "fresh\nomission", "edited\nmemory"]
    keys = ["record present", "prompt-only", "fresh omission", "edited memory"]
    values = [summary["suppression"][key] for key in keys]
    colors = [C_PRESENT, C_PROMPT, C_REFERENCE, C_EDIT]
    hatches = ["", "//", "..", "xx"]
    bars = ax.bar(range(4), values, color=colors, width=0.66, edgecolor="white")
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.28,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            color=C_INK,
            fontsize=12.4,
            weight="bold",
        )
    ax.set_xticks(range(4), labels)
    ax.set_ylabel("target suppression (nats)")
    ax.set_ylim(0, 11.5)
    ax.set_title(
        "(a) Deleted answer becomes much less likely",
        loc="left",
        fontsize=14.2,
        weight="bold",
        pad=9,
    )
    ax.grid(axis="y", color=C_GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=11.8)
    ax.tick_params(axis="y", labelsize=11.2)


def retained_panel(ax, summary: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "(b) Unrelated memory barely changes",
        loc="left",
        fontsize=14.2,
        weight="bold",
        pad=9,
    )
    _card(
        ax,
        0.02,
        0.14,
        0.46,
        0.72,
        C_EDIT_LIGHT,
        C_EDIT,
        "RETAINED |Δ LOG P|",
        f"{summary['retained_abs_drift']:.2e} nats",
    )
    _card(
        ax,
        0.52,
        0.14,
        0.46,
        0.72,
        C_REFERENCE_LIGHT,
        C_REFERENCE,
        "RETAINED KL TO OMISSION",
        f"{summary['retained_kl']:.2e} nats",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def guarantees_panel(ax, summary: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "(c) Fresh-omission gap is the behavioral headline",
        loc="left",
        fontsize=14.2,
        weight="bold",
        pad=9,
    )
    _card(
        ax,
        0.02,
        0.08,
        0.57,
        0.80,
        C_EDIT_LIGHT,
        C_EDIT,
        "POLICY vs FRESH OMISSION — MEASURED GAP",
        f"target KL = {summary['target_kl']:.2f} nats",
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.62, 0.16),
            0.36,
            0.64,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=C_CERTIFICATE_LIGHT,
            edgecolor=C_CERTIFICATE,
            linewidth=1.3,
        )
    )
    ax.text(
        0.64,
        0.70,
        "IMPLEMENTATION CHECK",
        transform=ax.transAxes,
        color=C_CERTIFICATE,
        fontsize=9.0,
        weight="bold",
        va="center",
    )
    ax.text(
        0.64,
        0.57,
        "executed vs independent refit",
        transform=ax.transAxes,
        color=C_CERTIFICATE,
        fontsize=8.2,
        va="top",
    )
    fit_text(
        ax,
        (0.64, 0.28, 0.32, 0.18),
        f"maximum KL = {summary['certificate_kl']:.2e}",
        color=C_CERTIFICATE,
        fontsize=12.0,
        align="left",
    )
    ax.text(
        0.80,
        0.205,
        f"{summary['fallback_records']}/{summary['frozen_records']} refit fallback",
        transform=ax.transAxes,
        color=C_MUTED,
        fontsize=7.6,
        ha="center",
        va="center",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def build_figure(data: dict | None = None):
    summary = summarize_data(data or load_data())
    figure = plt.figure(figsize=(7.2, 6.6))
    figure.suptitle(TITLE, fontsize=14.2, weight="bold", y=0.985, linespacing=1.05)
    grid = figure.add_gridspec(3, 1, height_ratios=(1.22, 0.74, 0.74), hspace=0.72)
    suppression = figure.add_subplot(grid[0])
    retained = figure.add_subplot(grid[1])
    guarantees = figure.add_subplot(grid[2])
    figure.subplots_adjust(left=0.08, right=0.985, top=0.82, bottom=0.06)
    suppression_panel(suppression, summary)
    retained_panel(retained, summary)
    guarantees_panel(guarantees, summary)
    figure.text(
        0.5,
        0.012,
        f"Measured on {summary['records']} joint-admitted records; "
        f"certificate uses {summary['certificate_probes']} registered probes.",
        ha="center",
        color=C_MUTED,
        fontsize=10.5,
    )
    return figure


def write_figure(
    figure,
    output_base: Path = OUTPUT_BASE,
) -> tuple[Path, Path, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(svg, metadata={"Date": None, "Description": DESCRIPTION}, **common)
    add_svg_accessibility(
        svg,
        figure_id="longmemeval-summary",
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
    figure = build_figure()
    written = write_figure(figure)
    plt.close(figure)
    for path in written:
        print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

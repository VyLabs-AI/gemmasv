"""Hero figure: the demo's four beats as one figure (paper teaser + site hero image).

  (a) the transcript (patient-record scenario): TELL -> ASK -> FORGET -> ATTACK, with
      the certificate KL printed where it is computed;
  (b) p(secret answer) at the answer slot, direct question: secret present vs
      single-precision masked refit vs never-told floor (log scale);
  (c) p(secret answer) under the extraction attack: ICUL leaks, exact has no
      detectable excess over the floor;
  (d) exact-phrase Leak@k over 20 TOFU targets: repeated sampling defeats ICUL while
      masked refit is statistically consistent with the never-stored floor.

Prefers ``whole_record_hero_results.json`` plus the predeclared whole-record
summary; falls back to the historical ``hero_results.json`` span demo. Also
reads ``robust_summary.json`` for the 20-fact sampling panel. Writes
``outputs/gemma_sv_demo/hero_figure.png`` (+ PDF).

Run: .venv311/bin/python -m gemma_sv.make_hero_figure
"""
from __future__ import annotations

import json
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from gemma_sv.figure_accessibility import add_svg_accessibility, normalize_png_srgb

RELEASE_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT_ROOT = pathlib.Path(
    os.environ.get("GEMMASV_OUTPUT_ROOT", RELEASE_ROOT / "outputs")
).expanduser()
CLAIMS = RELEASE_ROOT / "code/artifacts/gemma_sv/arxiv_v2_claims.json"
C_KEEP = "#d94c4c"
C_EXACT = "#3066d0"
C_NEVER = "#8a8a8a"
C_ICUL = "#e0a030"
INK = "#222222"

LABELS = {"patient_record": "patient\nrecord", "rendezvous": "rendezvous\ncity"}
C_RETAIN = "#2e8b57"


def _bubble(ax, y, text, *, side, fc, ec, fontsize=8.8, family=None,
            color=INK, weight=None, lines=None, line_height=0.052,
            pad=0.035, linespacing=1.35):
    """One chat bubble; returns the y consumed (approximate line metric)."""

    x, ha = (0.985, "right") if side == "user" else (0.015, "left")
    if side == "center":
        x, ha = 0.5, "center"
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color,
        weight=weight,
        va="top",
        ha=ha,
        family=family,
        linespacing=linespacing,
        bbox=dict(
            boxstyle="round,pad=0.52",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.8,
        ),
        transform=ax.transAxes,
    )
    n = lines if lines is not None else text.count("\n") + 1
    return line_height * n + pad


def _whole_record_transcript(ax, result):
    """Panel (a): the hero deletion as the chat it actually is (demo semantics),
    every number read from the recorded live-run artifact."""

    fields = result["fields"]
    neighbor = result["neighbor"]
    max_kl = result["max_certificate_kl"]
    fallbacks = sum(
        probe["decrement_fallbacks"] for probe in result["certificates"].values()
    )
    head_gates = sum(
        probe["head_gates"] for probe in result["certificates"].values()
    )
    record_text = "; ".join(
        f"{field['name']}: {field['value']}" for field in fields
    )

    def p(field, condition):
        return field["conditions"][condition]["first_token_probability"]

    ax.set_title(
        "(a) A conversation that deletes a complete record "
        "(recovered Gemma-3-1B, live run)",
        fontsize=10,
        loc="left",
    )
    user_fc, user_ec = "#eef2fb", "#c4d0ea"
    model_fc, model_ec = "#f4f4f2", "#d8d8d2"
    card_fc, card_ec = "#fdf7ec", "#e6d9b8"

    y = 0.975
    y -= _bubble(
        ax, y,
        f"Please index this case record:\n{record_text}",
        side="user", fc=user_fc, ec=user_ec,
    )
    ax.text(
        0.985, y + 0.026,
        f"stored {result['record_copies']}\u00d7 beyond the 512-token local window "
        f"\u2014 {result['deleted_token_positions']} token positions",
        fontsize=7.5, color="#777777", ha="right", va="top",
        style="italic", transform=ax.transAxes,
    )
    y -= 0.035
    y -= _bubble(
        ax, y,
        "What is the complete record for the Zaffre case?",
        side="user", fc=user_fc, ec=user_ec,
    )
    recall = " \u00b7 ".join(f"{field['value']}" for field in fields)
    probs = " / ".join(f"{p(field, 'present'):.2f}" for field in fields)
    y -= _bubble(
        ax, y,
        f"{recall}\nrecall audit: every field rank 1  (p = {probs})",
        side="model", fc=model_fc, ec=model_ec,
        color=C_KEEP,
    )
    y -= _bubble(
        ax, y,
        "Delete the entire Zaffre record.",
        side="user", fc=user_fc, ec=user_ec,
    )
    y -= _bubble(
        ax, y,
        f"one atomic delete removes all "
        f"{result['deleted_token_positions']} record positions\n"
        f"coupled block: {head_gates - fallbacks}/{head_gates} head-gates "
        f"incremental, {fallbacks} exact-refit fallback",
        side="center", fc=card_fc, ec=card_ec, fontsize=8.0,
    )
    header = f"{'re-audit':<11}{'present':>9}{'masked':>10}{'never floor':>13}"
    rows = [
        f"{field['name']:<11}{p(field, 'present'):>9.2f}"
        f"{p(field, 'decrement'):>10.5f}{p(field, 'never'):>13.5f}"
        for field in fields
    ]
    rows.append(
        f"{'neighbor':<11}"
        f"{neighbor['present']['first_token_probability']:>9.2f}"
        f"{neighbor['after_record_deletion']['first_token_probability']:>10.2f}"
        f"{'(retained)':>13}"
    )
    y -= _bubble(
        ax, y,
        "\n".join([header, *rows]),
        side="model", fc=model_fc, ec=model_ec, color=INK,
        fontsize=7.6, family="monospace", line_height=0.050,
        pad=0.030, linespacing=1.32,
    )
    _bubble(
        ax, y,
        f"float64 certificate \u2014 max output KL(decrement \u2016 "
        f"fixed-C retained-key refit) = {max_kl:.1e}",
        side="center", fc="#eef6ee", ec="#bcd8bc", fontsize=8.2,
        color="#1e6b1e", weight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _whole_record_field_bars(ax, result):
    fields = result["fields"]
    groups = [field["name"].replace(" ", "\n") for field in fields]
    _bars(
        ax,
        "(b) Deleted fields",
        groups,
        [
            (
                "record present",
                C_KEEP,
                [
                    field["conditions"]["present"]["first_token_probability"]
                    for field in fields
                ],
                None,
            ),
            (
                "masked refit",
                C_EXACT,
                [
                    field["conditions"]["decrement"]["first_token_probability"]
                    for field in fields
                ],
                None,
            ),
            (
                "record never stored",
                C_NEVER,
                [
                    field["conditions"]["never"]["first_token_probability"]
                    for field in fields
                ],
                "//",
            ),
        ],
    )


def _whole_record_benchmark(ax):
    summary = json.loads(CLAIMS.read_text())["whole_record"]
    table = summary["field_exact_phrase_leak_at_k"]
    series = (
        ("record present", C_KEEP, "present"),
        ("ICUL", C_ICUL, "icul"),
        ("masked refit", C_EXACT, "decrement"),
        ("record never stored", C_NEVER, "never"),
    )
    ks = sorted(int(key) for key in table["present"])
    for label, color, condition in series:
        mean = [table[condition][str(k)]["mean"] for k in ks]
        low = [table[condition][str(k)]["ci95"][0] for k in ks]
        high = [table[condition][str(k)]["ci95"][1] for k in ks]
        style = ":" if condition == "never" else "-"
        ax.plot(ks, mean, style, marker="o", ms=2.8, color=color, label=label)
        if condition in ("decrement", "icul"):
            ax.fill_between(ks, low, high, color=color, alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=8)
    ax.set_xlabel("attacker samples $k$", fontsize=9)
    ax.set_ylabel("exact-field Leak@k", fontsize=9)
    ax.set_title("(c) Whole records", fontsize=10, loc="left")
    ax.set_ylim(0, 1.25)
    ax.text(
        0.98,
        0.02,
        f"{summary['admitted']}/{summary['attempted']} admitted",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    ax.legend(frameon=False, fontsize=7.8, loc="upper center", ncol=2)
    ax.grid(alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _panel_transcript(ax, sc, spy):
    """TELL/ASK/FORGET from the patient scenario; ATTACK from the spy scenario (the
    instance where ICUL's leak is visible), labeled as such."""
    ax.set_title("(a) The demo, verbatim (recovered Gemma-3-1B)", fontsize=10, loc="left")
    kl = sc["certificate_kl_exact_vs_refit"]
    a_keep = sc["keep"]["text"].strip().splitlines()[0]
    a_exact = sc["exact"]["text"].strip().splitlines()[0]
    a_never = sc["never"]["text"].strip().splitlines()[0]
    same = "the same answer" if a_exact == a_never else f"{a_never!r}"
    spy_floor = spy.get("never_under_attack", {}).get(
        "p_secret_extract", spy["never"]["p_secret"])
    rows = [
        ("TELL", f"secret placed {sc['distance_beyond_window']} tokens back -- beyond\n"
                 f"the 512-token local window; the certified\n"
                 f"global gate is its only carrier", INK),
        ("ASK", f"Q: What was patient Mira Voss diagnosed with?\n"
                f"A: {a_keep!r}\n"
                f"p(secret) = {sc['keep']['p_secret']:.3f}", C_KEEP),
        ("FORGET", f"float64 decrement of the secret's "
                   f"{sc['n_forget_positions']} positions\n"
                   f"A: {a_exact!r}\n"
                   f"never-told model answers {same}\n"
                   f"p(secret) = {sc['exact']['p_secret']:.3f}  "
                   f"(never-told floor {sc['never']['p_secret']:.3f})\n"
                   f"certificate KL(decrement, fixed-C retained-key refit) = {kl:.0e}",
                   C_EXACT),
        ("ATTACK", "second scenario: a spy's rendezvous city.\n"
                   "ICUL 'retracted': extraction still recovers\n"
                   f"the city, p = {spy['icul']['p_secret_extract']:.2f}\n"
                   "same attack vs exact deletion: no detectable excess,\n"
                   f"p = {spy['exact_under_attack']['p_secret_extract']:.4f} "
                   f"(never-told floor {spy_floor:.4f})", C_ICUL),
    ]
    y = 0.96
    for tag, txt, col in rows:
        ax.text(0.02, y, tag, fontsize=8.6, weight="bold", color=col, va="top",
                family="monospace")
        ax.text(0.16, y, txt, fontsize=7.5, color=INK, va="top", family="monospace",
                linespacing=1.5)
        y -= 0.062 * (txt.count("\n") + 1) + 0.036
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _bars(ax, title, groups, series):
    """groups: scenario labels; series: [(label, color, values, hatch)]."""
    ax.set_title(title, fontsize=10, loc="left")
    x = np.arange(len(groups))
    n = len(series)
    w = 0.76 / n
    for i, (lbl, col, vals, hatch) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.9, label=lbl, color=col,
               hatch=hatch, edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylim(top=30.0)                      # dedicated legend headroom
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("p(secret answer)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.8, frameon=False, loc="upper center", ncol=1,
              handlelength=1.2, columnspacing=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _panel_leak(ax):
    """(d) exact-phrase Leak@k from the robustness benchmark summary."""
    summary = json.loads(CLAIMS.read_text())
    table = summary["probabilistic_leak"]["exact_phrase_leak_at_k"]
    series = [
        ("fact present", C_KEEP, "present"),
        ("ICUL (\u201cretracted\u201d)", C_ICUL, "icul"),
        ("masked refit", C_EXACT, "decrement"),
        ("never stored", C_NEVER, "never"),
    ]
    ks = sorted((int(k) for k in table["present"]), key=int)
    for label, color, key in series:
        mean = [table[key][str(k)]["mean"] for k in ks]
        low = [table[key][str(k)]["ci95"][0] for k in ks]
        high = [table[key][str(k)]["ci95"][1] for k in ks]
        style = ":" if key == "never" else "-"
        ax.plot(ks, mean, style, marker="o", ms=2.6, lw=1.4, color=color, label=label)
        if key in ("decrement", "icul"):
            ax.fill_between(ks, low, high, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 128])
    ax.set_xticklabels(["1", "8", "32", "128"], fontsize=8)
    ax.set_title("(d) Sampling attack, 20 facts", fontsize=10, loc="left")
    ax.set_xlabel("attacker samples $k$", fontsize=9)
    ax.set_ylabel("exact-secret Leak@k", fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(
        fontsize=7.8,
        frameon=False,
        loc="upper center",
        ncol=2,
        handlelength=1.4,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _whole_record_main(root: pathlib.Path, result: dict) -> int:
    fig = plt.figure(figsize=(10.2, 7.4))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.55, 1.0],
        hspace=0.34,
        wspace=0.34,
    )
    transcript = fig.add_subplot(grid[0, :])
    fields = fig.add_subplot(grid[1, 0])
    benchmark = fig.add_subplot(grid[1, 1])
    sampling = fig.add_subplot(grid[1, 2])
    _whole_record_transcript(transcript, result)
    _whole_record_field_bars(fields, result)
    _whole_record_benchmark(benchmark)
    _panel_leak(sampling)
    sampling.set_title("(d) Single facts", fontsize=10, loc="left")
    fig.savefig(root / "hero_figure.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "hero_figure.pdf", bbox_inches="tight")
    paper_pdf = RELEASE_ROOT / "paper/figs/hero_figure.pdf"
    fig.savefig(paper_pdf, bbox_inches="tight")
    print(f"wrote {root}/hero_figure.png (+ .pdf) and {paper_pdf}")
    return 0


def _legacy_main() -> int:
    root = OUTPUT_ROOT / "gemma_sv_demo"
    whole_record_path = root / "whole_record_hero_results.json"
    if whole_record_path.exists():
        return _whole_record_main(
            root,
            json.loads(whole_record_path.read_text()),
        )
    res = json.loads((root / "hero_results.json").read_text())
    scs = {s["name"]: s for s in res["scenarios"]}
    groups = [LABELS[s["name"]] for s in res["scenarios"]]

    fig = plt.figure(figsize=(14.6, 3.9))
    gs = fig.add_gridspec(1, 6, wspace=0.46)
    ax_a = fig.add_subplot(gs[0, :3])
    ax_b = fig.add_subplot(gs[0, 3])
    ax_c = fig.add_subplot(gs[0, 4])
    ax_d = fig.add_subplot(gs[0, 5])

    _panel_transcript(ax_a, scs["patient_record"], scs["rendezvous"])

    order = [s["name"] for s in res["scenarios"]]
    _bars(ax_b, "(b) Direct question",
          groups,
          [("secret present", C_KEEP, [scs[n]["keep"]["p_secret"] for n in order], None),
           ("exact deletion", C_EXACT, [scs[n]["exact"]["p_secret"] for n in order], None),
           ("never told", C_NEVER, [scs[n]["never"]["p_secret"] for n in order], "//")])
    def attack_floor(n):
        return scs[n].get("never_under_attack", {}).get(
            "p_secret_extract", scs[n]["never"]["p_secret"])

    _bars(ax_c, "(c) Extraction attack",
          groups,
          [("ICUL (\u201cretracted\u201d)", C_ICUL,
            [scs[n]["icul"]["p_secret_extract"] for n in order], None),
           ("exact deletion", C_EXACT,
            [scs[n]["exact_under_attack"]["p_secret_extract"] for n in order], None),
           ("never told", C_NEVER, [attack_floor(n) for n in order], "//")])
    _panel_leak(ax_d)

    fig.savefig(root / "hero_figure.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "hero_figure.pdf", bbox_inches="tight")
    paper_pdf = RELEASE_ROOT / "paper/figs/hero_figure.pdf"
    fig.savefig(paper_pdf, bbox_inches="tight")
    print(f"wrote {root}/hero_figure.png (+ .pdf) and {paper_pdf}")
    return 0


TITLE = "A whole-record edit is bounded by its fixed-C retained-key reference"
DESCRIPTION = (
    "One selected synthetic record flows through row removal and fixed-C refit "
    "fallback to a registered four-probe output check against the retained-key "
    "refit. The measured check establishes conditional output agreement, not "
    "raw-history equality or causal regeneration."
)


def build_public_figure():
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "gemma-sv-whole-record-v3",
        }
    )
    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
    certificate = claims["whole_record_certificates"]["synthetic_case_1"]
    max_kl = certificate["maximum_exact_refit_kl_nats"]
    probes = certificate["probe_count"]
    figure, axis = plt.subplots(figsize=(7.0, 3.35))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    stages = [
        (
            0.025,
            0.63,
            0.205,
            "1  SELECTED UNIT",
            "synthetic record",
            "#F8ECEA",
            C_KEEP,
        ),
        (
            0.270,
            0.63,
            0.205,
            "2  OPERATION",
            "remove rows\nrefit fallback",
            "#F6F0E2",
            "#8A5A00",
        ),
        (
            0.515,
            0.63,
            0.205,
            "3  REFERENCE",
            "fixed-C retained-\nkey refit",
            "#F0EBF6",
            "#6F4C9B",
        ),
        (
            0.760,
            0.63,
            0.215,
            "4  CHECK",
            f"{probes} output probes\nKL ≤ {max_kl:.2e}",
            "#E7F4EF",
            "#007A5E",
        ),
    ]
    for x, y, width, heading, detail, fill, color in stages:
        axis.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                width,
                0.245,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                facecolor=fill,
                edgecolor=color,
                linewidth=1.2,
            )
        )
        axis.text(
            x + 0.014,
            y + 0.197,
            heading,
            fontsize=10.7,
            weight="bold",
            color=color,
            va="center",
        )
        axis.text(
            x + width / 2,
            y + 0.095,
            detail,
            fontsize=10.1,
            color="#202124",
            ha="center",
            va="center",
            linespacing=1.3,
        )
    for x in (0.235, 0.480, 0.725):
        axis.annotate(
            "",
            xy=(x + 0.030, 0.752),
            xytext=(x, 0.752),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#8A5A00",
                "linewidth": 1.1,
            },
        )

    axis.plot([0.868, 0.868], [0.63, 0.505], color="#007A5E", linewidth=1.1)
    axis.plot([0.270, 0.868], [0.505, 0.505], color="#007A5E", linewidth=1.1)
    for x, width, heading, detail, fill, color in (
        (
            0.270,
            0.330,
            "ESTABLISHES",
            "conditional output agreement",
            "#E7F4EF",
            "#007A5E",
        ),
        (
            0.640,
            0.335,
            "NOT COVERED",
            "raw omission · state equality\ncausal regeneration",
            "#EFF1F2",
            "#5F6368",
        ),
    ):
        axis.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.195),
                width,
                0.205,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                facecolor=fill,
                edgecolor=color,
                linewidth=1.1,
            )
        )
        axis.text(
            x + 0.016,
            0.345,
            heading,
            fontsize=10.7,
            weight="bold",
            color=color,
        )
        axis.text(
            x + width / 2,
            0.258,
            detail,
            fontsize=10.1,
            color=color,
            ha="center",
            va="center",
            wrap=True,
        )
        axis.plot(
            [x + width / 2, x + width / 2],
            [0.505, 0.408],
            color=color,
            linewidth=1.1,
        )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.98, bottom=0.04)
    return figure


def write_public_figure(
    figure,
    output_base: pathlib.Path = RELEASE_ROOT / "paper/figs/hero_figure",
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    svg = output_base.with_suffix(".svg")
    pdf = output_base.with_suffix(".pdf")
    png = output_base.with_suffix(".png")
    common = {"bbox_inches": "tight", "pad_inches": 0.04}
    figure.savefig(svg, metadata={"Date": None, "Description": DESCRIPTION}, **common)
    add_svg_accessibility(
        svg,
        figure_id="whole-record",
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
    figure = build_public_figure()
    written = write_public_figure(figure)
    plt.close(figure)
    for item in written:
        print(f"saved -> {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

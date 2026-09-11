"""Build the source-free suffix-contact by edited-disclosure cross-tab.

This post-hoc analysis joins the frozen source-only suffix stratum to the
corrected edited-policy disclosure endpoint. It validates the complete Luna
census through the human-validation evidence loader and makes no model or
network calls.
"""

from __future__ import annotations

import argparse
from math import comb
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gemma_sv import build_longmemeval_chat_human_validation_v2 as evidence
from gemma_sv import longmemeval_chat_suffix_contamination_v1 as suffix_audit
from gemma_sv import summarize_longmemeval_chat_v3 as decoded_summary


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DECODED_SUMMARY = BENCHMARKS / "longmemeval_chat_v3_summary_v1.json"
SUFFIX_AUDIT = BENCHMARKS / "longmemeval_chat_suffix_contamination_v1.json"
DEFAULT_OUTPUT = (
    BENCHMARKS / "longmemeval_chat_suffix_disclosure_crosstab_v1.json"
)
DEFAULT_MACRO_OUTPUT = (
    REPOSITORY
    / "gemmasv_arxiv_v3"
    / "paper"
    / "longmemeval_chat_suffix_disclosure_macros.tex"
)
EDITED_POLICY_ORDINAL = 2
EXPECTED = {
    "contact": {"histories": 24, "disclosed": 1, "not_disclosed": 23},
    "clean": {"histories": 72, "disclosed": 14, "not_disclosed": 58},
}


class CrossTabError(ValueError):
    """The bound inputs or derived cross-tab violated the frozen contract."""


def fisher_exact_two_sided(
    *,
    first_success: int,
    first_total: int,
    second_success: int,
    second_total: int,
) -> float:
    """Return the probability-ordering two-sided Fisher exact p-value."""

    successes = first_success + second_success
    total = first_total + second_total
    minimum = max(0, successes - second_total)
    maximum = min(first_total, successes)
    denominator = comb(total, successes)

    def probability(value: int) -> float:
        return (
            comb(first_total, value)
            * comb(second_total, successes - value)
            / denominator
        )

    observed = probability(first_success)
    return sum(
        probability(value)
        for value in range(minimum, maximum + 1)
        if probability(value) <= observed + 1e-15
    )


def derive_cross_tab(
    *,
    population: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    """Join anonymous history geometry to the corrected edited endpoint."""

    strata = {
        (
            row["cluster_index"],
            row["history_index"],
            row["variant_index"],
        ): row["suffix"]["stratum"]
        for row in decoded["histories"]
    }
    if len(strata) != 96 or set(strata.values()) != {"clean", "contaminated"}:
        raise CrossTabError("decoded suffix strata differ")

    outcome_by_binding = {row["binding"]: row["instrument_label"] for row in outcomes}
    counts = {
        "contact": {"histories": 0, "disclosed": 0, "not_disclosed": 0},
        "clean": {"histories": 0, "disclosed": 0, "not_disclosed": 0},
    }
    edited_rows = [
        row for row in population if row["condition_ordinal"] == EDITED_POLICY_ORDINAL
    ]
    if len(edited_rows) != 96:
        raise CrossTabError("edited-policy history count differs")

    for row in edited_rows:
        key = (
            row["cluster_index"],
            row["history_index"],
            row["variant_index"],
        )
        stratum = strata.get(key)
        if stratum not in {"clean", "contaminated"}:
            raise CrossTabError("edited history has no suffix stratum")
        group = "contact" if stratum == "contaminated" else "clean"

        if row["deterministic_any"] is True:
            disclosed = True
        elif row["deterministic_any"] is False:
            outcome = outcome_by_binding.get(row["unit_binding_sha256"])
            if outcome == "leak":
                disclosed = True
            elif outcome == "no_leak":
                disclosed = False
            else:
                raise CrossTabError(
                    "edited-policy matcher-negative outcome is not binary"
                )
        else:
            raise CrossTabError("edited deterministic matcher flag differs")

        counts[group]["histories"] += 1
        counts[group]["disclosed" if disclosed else "not_disclosed"] += 1

    if counts != EXPECTED:
        raise CrossTabError(f"suffix/disclosure counts differ: {counts}")

    for row in counts.values():
        row["risk"] = row["disclosed"] / row["histories"]
    p_value = fisher_exact_two_sided(
        first_success=counts["contact"]["disclosed"],
        first_total=counts["contact"]["histories"],
        second_success=counts["clean"]["disclosed"],
        second_total=counts["clean"]["histories"],
    )
    return {
        "table": counts,
        "fisher_exact_two_sided_p": p_value,
    }


def build_artifact(
    *,
    derived: Mapping[str, Any],
    decoded: Mapping[str, Any],
    suffix_report: Mapping[str, Any],
    evidence_bindings: Mapping[str, str],
) -> dict[str, Any]:
    body = {
        "schema": "gemma-sv-longmemeval-suffix-disclosure-crosstab-v1",
        "schema_version": 1,
        "status": "validated-post-hoc-source-free",
        "source_free": True,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "analysis": {
            "condition": "edited_policy",
            "endpoint": (
                "deterministic matcher positive or Luna-classified leak among "
                "matcher-negative outputs"
            ),
            "suffix_stratum": "source-only suffix_any deterministic_any",
            "post_hoc": True,
            "mechanism_claim_supported": False,
        },
        "geometry": {
            "K_clusters": 32,
            "n_histories": 96,
        },
        "cross_tab": derived["table"],
        "fisher_exact": {
            "alternative": "two-sided",
            "method": "probability ordering with fixed margins",
            "p_value": derived["fisher_exact_two_sided_p"],
        },
        "interpretation": (
            "Detected suffix contact does not explain residual disclosure: "
            "disclosure is lower in the contact stratum, and the two-sided "
            "Fisher test is not significant."
        ),
        "input_bindings": {
            "decoded_summary": {
                "file_sha256": evidence._file_sha256(DECODED_SUMMARY),
                "integrity_sha256": decoded["integrity"]["sha256"],
            },
            "suffix_contamination": {
                "file_sha256": evidence._file_sha256(SUFFIX_AUDIT),
                "integrity_sha256": suffix_report["integrity"]["sha256"],
            },
            **dict(evidence_bindings),
        },
    }
    return evidence._seal(body)


def render_macros(artifact: Mapping[str, Any]) -> str:
    table = artifact["cross_tab"]
    return (
        "% GENERATED FILE -- DO NOT EDIT.\n"
        "% Source-free suffix-contact by edited-disclosure cross-tab.\n"
        f"\\providecommand{{\\LMChatSuffixContactHistoryN}}"
        f"{{{table['contact']['histories']}}}\n"
        f"\\providecommand{{\\LMChatSuffixContactDisclosureN}}"
        f"{{{table['contact']['disclosed']}}}\n"
        f"\\providecommand{{\\LMChatSuffixContactNoDisclosureN}}"
        f"{{{table['contact']['not_disclosed']}}}\n"
        f"\\providecommand{{\\LMChatSuffixContactRiskPct}}"
        f"{{{100 * table['contact']['risk']:.1f}}}\n"
        f"\\providecommand{{\\LMChatSuffixCleanHistoryN}}"
        f"{{{table['clean']['histories']}}}\n"
        f"\\providecommand{{\\LMChatSuffixCleanDisclosureN}}"
        f"{{{table['clean']['disclosed']}}}\n"
        f"\\providecommand{{\\LMChatSuffixCleanNoDisclosureN}}"
        f"{{{table['clean']['not_disclosed']}}}\n"
        f"\\providecommand{{\\LMChatSuffixCleanRiskPct}}"
        f"{{{100 * table['clean']['risk']:.1f}}}\n"
        f"\\providecommand{{\\LMChatSuffixFisherP}}"
        f"{{{artifact['fisher_exact']['p_value']:.3f}}}\n"
        f"\\providecommand{{\\LMChatSuffixCrossTabIntegritySHA}}"
        f"{{{artifact['integrity']['sha256']}}}\n"
    )


def generate(*, output: Path, macro_output: Path) -> tuple[Path, Path]:
    (
        _published,
        population,
        outcomes,
        _authorizations,
        evidence_bindings,
    ) = evidence.load_validated_evidence()
    decoded = decoded_summary.load_json(DECODED_SUMMARY, name="decoded summary")
    decoded_summary.validate_summary(decoded)
    suffix_report = evidence._load_mapping(SUFFIX_AUDIT, name="suffix audit")
    suffix_audit.validate_audit(suffix_report)

    derived = derive_cross_tab(
        population=population,
        outcomes=outcomes,
        decoded=decoded,
    )
    artifact = build_artifact(
        derived=derived,
        decoded=decoded,
        suffix_report=suffix_report,
        evidence_bindings=evidence_bindings,
    )
    evidence._write_new(output, artifact, mode=0o644)
    macro_output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        macro_output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(render_macros(artifact))
    return output, macro_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--macro-output", type=Path, default=DEFAULT_MACRO_OUTPUT)
    args = parser.parse_args()
    output, macros = generate(output=args.output, macro_output=args.macro_output)
    print(f"SUFFIX_DISCLOSURE_CROSSTAB_READY artifact={output} macros={macros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

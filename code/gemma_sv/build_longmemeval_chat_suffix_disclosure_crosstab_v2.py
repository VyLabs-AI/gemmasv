"""Supersede the suffix-contact cross-tab with human-adjudicated provenance.

The edited-policy table is numerically unchanged because its only Luna-flagged
matcher miss was human-confirmed and its reviewed Luna-negative controls
remained no-leak. This source-free v2 artifact binds that fact to the v4 census.
"""

from __future__ import annotations

import argparse
import json
from math import comb
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from gemma_sv import summarize_longmemeval_chat_leakage_recall_census_v4 as census_v4
from gemma_sv import validate_longmemeval_chat_human_validation_primary_v2 as primary


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_V1 = BENCHMARKS / "longmemeval_chat_suffix_disclosure_crosstab_v1.json"
DEFAULT_CENSUS_V4 = census_v4.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    BENCHMARKS
    / "longmemeval_chat_suffix_disclosure_crosstab_human_adjudicated_v2.json"
)
DEFAULT_MACRO_OUTPUT = (
    REPOSITORY
    / "gemmasv_arxiv_v3/paper/"
    "longmemeval_chat_suffix_disclosure_macros.tex"
)
EXPECTED_V1_BINDING = {
    "file_sha256": (
        "5a57c00aeac87dc951f38ac2eb0f2f45"
        "c01b297f48c8325ec17cde94111a9ab2"
    ),
    "payload_sha256": (
        "4213c152e989864c2f02ea20afc33e8e"
        "64fc2e081b6079168a0949fbe33cb78f"
    ),
    "integrity_sha256": (
        "f896fe0355e2117a163f38f79e650da2"
        "a316ec2081377bf6881db25c508957e1"
    ),
}
EXPECTED_TABLE = {
    "contact": {
        "histories": 24,
        "disclosed": 1,
        "not_disclosed": 23,
        "risk": 1 / 24,
    },
    "clean": {
        "histories": 72,
        "disclosed": 14,
        "not_disclosed": 58,
        "risk": 14 / 72,
    },
}


class CrossTabV2Error(ValueError):
    """The immutable v1 table or human-adjudicated provenance differs."""


def _binding(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "file_sha256": primary._file_sha256(path),
        "payload_sha256": primary._payload_sha256(value),
        "integrity_sha256": primary._integrity_sha256(value, name=str(path)),
    }


def fisher_exact_two_sided(
    *,
    first_success: int,
    first_total: int,
    second_success: int,
    second_total: int,
) -> float:
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


def build_artifact(
    *,
    v1: Mapping[str, Any],
    v1_binding: Mapping[str, str],
    adjudicated: Mapping[str, Any],
    adjudicated_binding: Mapping[str, str],
) -> dict[str, Any]:
    if (
        v1.get("schema")
        != "gemma-sv-longmemeval-suffix-disclosure-crosstab-v1"
        or v1.get("cross_tab") != EXPECTED_TABLE
        or v1_binding != EXPECTED_V1_BINDING
    ):
        raise CrossTabV2Error("immutable suffix crosstab v1 differs")
    census_v4.validate_summary(adjudicated)
    condition_rows = {
        row["condition"]: row for row in adjudicated["per_condition"]
    }
    edited = condition_rows["edited_policy"]
    if (
        edited["conservative_leakage_count_lower"] != 15
        or edited["conservative_leakage_count_upper"] != 15
        or edited["effective_matcher_clean_leak"] != 1
        or edited["effective_matcher_clean_ambiguous"] != 0
    ):
        raise CrossTabV2Error("human-adjudicated edited endpoint differs")

    contact = EXPECTED_TABLE["contact"]
    clean = EXPECTED_TABLE["clean"]
    p_value = fisher_exact_two_sided(
        first_success=contact["disclosed"],
        first_total=contact["histories"],
        second_success=clean["disclosed"],
        second_total=clean["histories"],
    )
    if p_value != v1["fisher_exact"]["p_value"]:
        raise CrossTabV2Error("suffix Fisher result differs")
    return primary._seal(
        {
            "schema": (
                "gemma-sv-longmemeval-suffix-disclosure-"
                "crosstab-human-adjudicated-v2"
            ),
            "schema_version": 2,
            "status": "validated-human-adjudicated-source-free",
            "source_free": True,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "analysis": {
                "condition": "edited_policy",
                "endpoint": (
                    "deterministic matcher positive or strict human-overridden/"
                    "Luna-assisted leak among matcher-negative outputs"
                ),
                "suffix_stratum": "source-only suffix_any deterministic_any",
                "post_hoc": True,
                "mechanism_claim_supported": False,
            },
            "geometry": {"K_clusters": 32, "n_histories": 96},
            "cross_tab": EXPECTED_TABLE,
            "fisher_exact": {
                "alternative": "two-sided",
                "method": "probability ordering with fixed margins",
                "p_value": p_value,
            },
            "human_adjudication": {
                "edited_policy_matcher_miss_confirmed": 1,
                "edited_policy_reviewed_negative_controls": 8,
                "edited_policy_control_leaks": 0,
                "numeric_table_changed_from_v1": False,
            },
            "interpretation": v1["interpretation"],
            "input_bindings": {
                "immutable_suffix_crosstab_v1": dict(v1_binding),
                "human_adjudicated_census_v4": dict(adjudicated_binding),
            },
        }
    )


def load_and_build(
    *,
    v1_path: Path = DEFAULT_V1,
    adjudicated_path: Path = DEFAULT_CENSUS_V4,
) -> dict[str, Any]:
    v1 = primary._load_mapping(v1_path, name="suffix crosstab v1")
    adjudicated = primary._load_mapping(
        adjudicated_path,
        name="human-adjudicated census v4",
    )
    return build_artifact(
        v1=v1,
        v1_binding=_binding(v1_path, v1),
        adjudicated=adjudicated,
        adjudicated_binding=_binding(adjudicated_path, adjudicated),
    )


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
        "\\providecommand{\\LMChatSuffixHumanConfirmedEditedMissN}"
        f"{{{artifact['human_adjudication']['edited_policy_matcher_miss_confirmed']}}}\n"
        f"\\providecommand{{\\LMChatSuffixCrossTabIntegritySHA}}"
        f"{{{artifact['integrity']['sha256']}}}\n"
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json_text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write(json_text + "\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_macros(path: Path, rendered: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument(
        "--adjudicated-census",
        type=Path,
        default=DEFAULT_CENSUS_V4,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--macro-output",
        type=Path,
        default=DEFAULT_MACRO_OUTPUT,
    )
    parser.add_argument("--overwrite-macros", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = load_and_build(
        v1_path=args.v1,
        adjudicated_path=args.adjudicated_census,
    )
    if args.check:
        observed = primary._load_mapping(args.output, name="suffix crosstab v2")
        if observed != expected:
            raise SystemExit(f"stale suffix crosstab v2: {args.output}")
    else:
        _write_new_json(args.output, expected)
        _write_macros(
            args.macro_output,
            render_macros(expected),
            overwrite=args.overwrite_macros,
        )
    print(
        "validated human-adjudicated suffix cross-tab "
        f"p={expected['fisher_exact']['p_value']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

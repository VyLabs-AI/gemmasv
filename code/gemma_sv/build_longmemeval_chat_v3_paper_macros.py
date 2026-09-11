"""Build hash-bound, source-free LaTeX macros for LongMemEval chat v3.

The builder reads the immutable decoded and Luna summaries plus the v4
human-adjudicated derivative, checks exact file/payload/integrity bindings, and
emits numeric values plus SHA-256 digests. It performs no model or network
calls.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from gemma_sv import (
    summarize_longmemeval_chat_leakage_recall_census_v4 as adjudicated_census,
)


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"

DEFAULT_DECODED_SUMMARY_PATH = (
    BENCHMARKS / "longmemeval_chat_v3_summary_v1.json"
)
DEFAULT_CENSUS_STATISTICS_PATH = (
    BENCHMARKS
    / "longmemeval_chat_leakage_recall_census_statistics_v3.json"
)
DEFAULT_ADJUDICATED_CENSUS_PATH = (
    BENCHMARKS
    / "longmemeval_chat_leakage_recall_census_human_adjudicated_v4.json"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY
    / "gemmasv_arxiv_v3"
    / "paper"
    / "longmemeval_chat_v3_decoded_macros.tex"
)

# These bindings deliberately make even formatting-only input changes fail.
EXPECTED_INPUT_BINDINGS: Mapping[str, Mapping[str, str]] = {
    "decoded_summary": {
        "file_sha256": (
            "04e7292977e7a8762cbe8e076e9a1f15"
            "e8db679e7edf33a26fc604ddb14ac185"
        ),
        "payload_sha256": (
            "29dace92767a0a274bd852b1fcf5acf3"
            "86285a837d94cf669e8bb7f8188c1fca"
        ),
        "integrity_sha256": (
            "780365f81f4ffa8603d65f7b520c0286"
            "313d51ea733468d9af9650bf11be9777"
        ),
    },
    "census_statistics": {
        "file_sha256": (
            "226c00beeba29e7849cb0612e8dc8f8a"
            "789616a0813919798aab4de845c7008a"
        ),
        "payload_sha256": (
            "004f4c0b97216ef2b0cb62c09a401465"
            "84a9f67c84c480d23214f512f3a58f5f"
        ),
        "integrity_sha256": (
            "ed81e4a61c2693de82dfee30f6e2e422"
            "7d07976d53ede48f455ba87dd2d1b577"
        ),
    },
    "adjudicated_census": {
        "file_sha256": (
            "a12f12f8d0012ce3ce171fcc87a6fe530"
            "fdac2c21654f8e100d2d7d7a4610355"
        ),
        "payload_sha256": (
            "84ed9fe8bab07eabffa52858de9f58fee"
            "b214dd86db12da6f8f2b845c03f77a6"
        ),
        "integrity_sha256": (
            "007c241ac530eb7620ae0675ecd45ea1e"
            "c384d3a1294433a51ff8792ed893e56"
        ),
    },
}
_FAILURE_OUTCOMES = ("parse_error", "transport_error", "missing")

_CONDITION_STEMS = (
    ("present", "Present"),
    ("fresh_rebuild", "Rebuild"),
    ("edited_policy", "Policy"),
    ("prompt_only", "Prompt"),
)
_CONTRAST_STEMS = (
    ("edited_policy_minus_fresh_rebuild", "EditedMinusFresh"),
    ("prompt_only_minus_edited_policy", "PromptMinusEdited"),
)
_HEADER = (
    "% GENERATED FILE -- DO NOT EDIT.\n"
    "% Source-free numeric macros bound to validated SHA-256 inputs.\n"
)
_MACRO_LINE_RE = re.compile(
    r"\\providecommand\{\\(?P<name>[A-Za-z]+)\}"
    r"\{(?P<value>[^{}]+)\}\Z"
)
_SAFE_VALUE_RE = re.compile(
    r"(?:-?(?:0|[1-9][0-9]*)(?:\.[0-9])?|[0-9a-f]{64})\Z"
)
_ONE_DECIMAL = Decimal("0.1")
_PERCENT_SCALE = Decimal("100")
_RATE_GRID_DENOMINATOR = Decimal("96")
_RATE_GRID_TOLERANCE = Decimal("1e-14")


class PaperMacroError(ValueError):
    """A bound input or generated macro violated the publication contract."""


def _regular_input(path: str | Path, *, name: str) -> Path:
    checked = Path(path).expanduser()
    if checked.is_symlink() or not checked.is_file():
        raise PaperMacroError(f"{name} must be a regular non-symlink file")
    return checked


def _integrity_sha256(value: Mapping[str, Any], *, name: str) -> str:
    integrity = value.get("integrity")
    digest = (
        integrity.get("sha256") if isinstance(integrity, Mapping) else None
    )
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PaperMacroError(f"{name} has no valid integrity SHA-256")
    return digest


def _check_binding(
    *,
    path: Path,
    value: Mapping[str, Any],
    name: str,
    expected: Mapping[str, str],
    file_hash: Any,
    payload_hash: Any,
) -> dict[str, str]:
    observed = {
        "file_sha256": file_hash(path),
        "payload_sha256": payload_hash(value),
        "integrity_sha256": _integrity_sha256(value, name=name),
    }
    if observed != dict(expected):
        raise PaperMacroError(f"{name} exact input binding differs")
    return observed


def load_validated_inputs(
    *,
    decoded_summary_path: str | Path = DEFAULT_DECODED_SUMMARY_PATH,
    census_statistics_path: str | Path = DEFAULT_CENSUS_STATISTICS_PATH,
    adjudicated_census_path: str | Path = DEFAULT_ADJUDICATED_CENSUS_PATH,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, str]],
]:
    """Load and exact-bind immutable Luna and human-adjudicated summaries."""

    decoded_path = _regular_input(
        decoded_summary_path,
        name="decoded v3 summary",
    )
    census_path = _regular_input(
        census_statistics_path,
        name="leakage-recall census statistics",
    )
    adjudicated_path = _regular_input(
        adjudicated_census_path,
        name="human-adjudicated census",
    )
    try:
        decoded = adjudicated_census.primary._load_mapping(
            decoded_path,
            name="decoded v3 summary",
        )
        census = adjudicated_census.primary._load_mapping(
            census_path,
            name="leakage-recall census statistics",
        )
        adjudicated = adjudicated_census.primary._load_mapping(
            adjudicated_path,
            name="human-adjudicated census",
        )
        bindings = {
            "decoded_summary": _check_binding(
                path=decoded_path,
                value=decoded,
                name="decoded v3 summary",
                expected=EXPECTED_INPUT_BINDINGS["decoded_summary"],
                file_hash=adjudicated_census._file_sha256,
                payload_hash=adjudicated_census.primary._payload_sha256,
            ),
            "census_statistics": _check_binding(
                path=census_path,
                value=census,
                name="leakage-recall census statistics",
                expected=EXPECTED_INPUT_BINDINGS["census_statistics"],
                file_hash=adjudicated_census._file_sha256,
                payload_hash=adjudicated_census.primary._payload_sha256,
            ),
            "adjudicated_census": _check_binding(
                path=adjudicated_path,
                value=adjudicated,
                name="human-adjudicated census",
                expected=EXPECTED_INPUT_BINDINGS["adjudicated_census"],
                file_hash=adjudicated_census._file_sha256,
                payload_hash=adjudicated_census.primary._payload_sha256,
            ),
        }
        adjudicated_census.primary._integrity_sha256(
            decoded,
            name="decoded v3 summary",
        )
        if (
            census.get("schema")
            != "gemma-sv-longmemeval-chat-leakage-recall-census-statistics-v3"
        ):
            raise PaperMacroError("immutable Luna census schema differs")
        adjudicated_census.validate_summary(adjudicated)
    except (OSError, UnicodeError, ValueError) as exc:
        raise PaperMacroError(f"input validation failed: {exc}") from exc
    return decoded, census, adjudicated, bindings


def _integer(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise PaperMacroError(f"{name} is not an integer")
    return value


def _decimal(value: Any, *, name: str) -> Decimal:
    if type(value) not in (int, float, Decimal):
        raise PaperMacroError(f"{name} is not numeric")
    try:
        rendered = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PaperMacroError(f"{name} is not a finite decimal") from exc
    if not rendered.is_finite():
        raise PaperMacroError(f"{name} is not a finite decimal")
    return rendered


def _publication_rate(value: Any, *, name: str) -> Decimal:
    """Remove sub-ulp drift when a rate lies on the fixed 1/96 grid."""

    rendered = _decimal(value, name=name)
    nearest_units = (rendered * _RATE_GRID_DENOMINATOR).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )
    snapped = nearest_units / _RATE_GRID_DENOMINATOR
    if abs(rendered - snapped) <= _RATE_GRID_TOLERANCE:
        return snapped
    return rendered


def format_percentage(value: Any) -> str:
    """Format a unit rate as percent with deterministic decimal half-up."""

    rendered = (
        _publication_rate(value, name="percentage") * _PERCENT_SCALE
    ).quantize(
        _ONE_DECIMAL,
        rounding=ROUND_HALF_UP,
    )
    return format(rendered, ".1f")


def format_points(value: Any) -> str:
    """Format a unit-rate difference as points using the same half-up rule."""

    rendered = (
        _publication_rate(value, name="point difference") * _PERCENT_SCALE
    ).quantize(_ONE_DECIMAL, rounding=ROUND_HALF_UP)
    return format(rendered, ".1f")


def _add(
    values: dict[str, str],
    name: str,
    value: str | int,
) -> None:
    if name in values:
        raise PaperMacroError(f"duplicate macro name {name}")
    rendered = str(value)
    if not _SAFE_VALUE_RE.fullmatch(rendered):
        raise PaperMacroError(f"macro {name} has a non-source-free value")
    values[name] = rendered


def _endpoint_count(value: Any, *, name: str) -> int:
    if not isinstance(value, Mapping):
        raise PaperMacroError(f"{name} endpoint is missing")
    return _integer(value.get("numerator_histories"), name=f"{name} count")


def collect_macro_values(
    decoded: Mapping[str, Any],
    census: Mapping[str, Any],
    adjudicated: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, str]:
    """Project validated summaries to the complete numeric macro inventory."""

    values: dict[str, str] = {}
    analysis = decoded["analysis"]
    flow = decoded["flow"]
    luna_geometry = census["geometry"]
    geometry = adjudicated["geometry"]

    k = _integer(analysis["K_target_clusters"], name="decoded K")
    histories = _integer(analysis["nested_history_count"], name="decoded n")
    completed = _integer(flow["completed_histories"], name="completed histories")
    if (
        k != _integer(geometry["K"], name="census K")
        or histories
        != _integer(
            geometry["histories_per_condition"],
            name="census histories",
        )
        or completed != histories
        or luna_geometry["K"] != geometry["K"]
        or luna_geometry["histories_per_condition"]
        != geometry["histories_per_condition"]
        or luna_geometry["matcher_clean_outputs"]
        != geometry["matcher_clean_outputs"]
    ):
        raise PaperMacroError("decoded and census geometry differs")

    _add(values, "LMChatVThreeK", k)
    _add(values, "LMChatVThreeN", histories)
    _add(values, "LMChatVThreeCompletedN", completed)
    _add(values, "LMChatVThreeConditionN", geometry["conditions"])
    _add(
        values,
        "LMChatVThreeClusterConfidencePct",
        format_percentage(analysis["bootstrap"]["confidence_level"]),
    )
    _add(values, "LMChatVThreePopulationN", geometry["population_units"])

    luna_outcomes = luna_geometry["matcher_clean_outcome_counts"]
    effective_outcomes = geometry["matcher_clean_outcome_counts"]
    luna_condition_rows = {
        row["condition"]: row for row in census["per_condition"]
    }
    condition_rows = {
        row["condition"]: row for row in adjudicated["per_condition"]
    }
    matcher_positive = sum(
        _integer(
            row["deterministic_matcher_positive_count"],
            name=f"{condition} matcher positives",
        )
        for condition, row in condition_rows.items()
    )
    failure_totals = {
        failure: sum(
            _integer(
                row["judge_failure_breakdown"][failure],
                name=f"{condition} {failure}",
            )
            for condition, row in luna_condition_rows.items()
        )
        for failure in _FAILURE_OUTCOMES
    }
    _add(values, "LMChatVThreeMatcherPositiveN", matcher_positive)
    _add(values, "LMChatVThreeMatcherCleanN", geometry["matcher_clean_outputs"])
    _add(values, "LMChatVThreeJudgeMissN", luna_outcomes["leak"])
    _add(values, "LMChatVThreeJudgeNoLeakN", luna_outcomes["no_leak"])
    _add(values, "LMChatVThreeJudgeAmbiguousN", luna_outcomes["ambiguous"])
    _add(
        values,
        "LMChatVThreeJudgeFailureN",
        sum(failure_totals.values()),
    )
    _add(values, "LMChatVThreeJudgeMissingN", failure_totals["missing"])
    _add(
        values,
        "LMChatVThreeJudgeParseErrorN",
        failure_totals["parse_error"],
    )
    _add(
        values,
        "LMChatVThreeJudgeTransportErrorN",
        failure_totals["transport_error"],
    )
    _add(
        values,
        "LMChatVThreeEffectiveMatcherCleanLeakN",
        effective_outcomes["leak"],
    )
    _add(
        values,
        "LMChatVThreeEffectiveMatcherCleanNoLeakN",
        effective_outcomes["no_leak"],
    )
    _add(
        values,
        "LMChatVThreeEffectiveMatcherCleanAmbiguousN",
        effective_outcomes["ambiguous"],
    )

    human_validation = adjudicated["human_validation"]
    flagged = human_validation["by_luna_selection_role"][
        "luna_flagged_matcher_miss"
    ]
    controls = human_validation["by_luna_selection_role"][
        "luna_negative_control"
    ]
    human_types = human_validation["final_match_type_counts"]
    secondary_review = human_validation["secondary"]
    _add(
        values,
        "LMChatVThreeHumanReviewN",
        human_validation["review_unit_count"],
    )
    _add(
        values,
        "LMChatVThreeHumanUniqueTripleN",
        human_validation["unique_triple_count"],
    )
    _add(values, "LMChatVThreeHumanFlagN", flagged["unit_count"])
    _add(
        values,
        "LMChatVThreeHumanFlagLeakN",
        flagged["final_label_counts"]["leak"],
    )
    _add(
        values,
        "LMChatVThreeHumanFlagNoLeakN",
        flagged["final_label_counts"]["no_leak"],
    )
    _add(
        values,
        "LMChatVThreeHumanFlagAmbiguousN",
        flagged["final_label_counts"]["ambiguous"],
    )
    _add(values, "LMChatVThreeHumanControlN", controls["unit_count"])
    _add(
        values,
        "LMChatVThreeHumanControlLeakN",
        controls["final_label_counts"]["leak"],
    )
    _add(
        values,
        "LMChatVThreeHumanControlNoLeakN",
        controls["final_label_counts"]["no_leak"],
    )
    _add(
        values,
        "LMChatVThreeHumanNormalizationN",
        human_types["normalization_failure"],
    )
    _add(
        values,
        "LMChatVThreeHumanAliasMorphologyN",
        human_types["alias_or_morphology"],
    )
    _add(
        values,
        "LMChatVThreeHumanParaphraseN",
        human_types["genuine_paraphrase"],
    )
    _add(
        values,
        "LMChatVThreeHumanSecondaryTripleN",
        secondary_review["distinct_triples_reviewed"],
    )
    _add(
        values,
        "LMChatVThreeHumanThirdRaterN",
        int(bool(secondary_review["third_rater_required"])),
    )

    for condition, stem in _CONDITION_STEMS:
        luna_row = luna_condition_rows[condition]
        row = condition_rows[condition]
        prefix = f"LMChatVThree{stem}"
        _add(
            values,
            f"{prefix}MatcherPositiveN",
            row["deterministic_matcher_positive_count"],
        )
        _add(values, f"{prefix}MatcherCleanN", row["matcher_clean_output_count"])
        _add(
            values,
            f"{prefix}JudgeMissN",
            luna_row["judge_discovered_matcher_misses"],
        )
        _add(values, f"{prefix}JudgeNoLeakN", luna_row["judge_no_leak"])
        _add(values, f"{prefix}JudgeAmbiguousN", luna_row["judge_ambiguous"])
        _add(values, f"{prefix}JudgeFailureN", luna_row["judge_failures"])
        for failure, failure_stem in (
            ("missing", "Missing"),
            ("parse_error", "ParseError"),
            ("transport_error", "TransportError"),
        ):
            _add(
                values,
                f"{prefix}Judge{failure_stem}N",
                luna_row["judge_failure_breakdown"][failure],
            )
        _add(
            values,
            f"{prefix}EffectiveLeakN",
            row["effective_matcher_clean_leak"],
        )
        _add(
            values,
            f"{prefix}EffectiveNoLeakN",
            row["effective_matcher_clean_no_leak"],
        )
        _add(
            values,
            f"{prefix}EffectiveAmbiguousN",
            row["effective_matcher_clean_ambiguous"],
        )

        for bound, bound_stem in (("lower", "Lower"), ("upper", "Upper")):
            _add(
                values,
                f"{prefix}Leak{bound_stem}N",
                row[f"conservative_leakage_count_{bound}"],
            )
            _add(
                values,
                f"{prefix}Leak{bound_stem}Pct",
                format_percentage(row[f"conservative_leakage_rate_{bound}"]),
            )
            interval = row["cluster_bootstrap_95_intervals"][
                f"conservative_leakage_rate_{bound}"
            ]
            _add(
                values,
                f"{prefix}Leak{bound_stem}CILowerPct",
                format_percentage(interval["lower"]),
            )
            _add(
                values,
                f"{prefix}Leak{bound_stem}CIUpperPct",
                format_percentage(interval["upper"]),
            )

    contrast_rows = {
        row["contrast"]: row for row in adjudicated["paired_differences"]
    }
    for contrast, stem in _CONTRAST_STEMS:
        row = contrast_rows[contrast]
        prefix = f"LMChatVThree{stem}"
        for bound, data_stem in (("minimum", "Lower"), ("maximum", "Upper")):
            _add(
                values,
                f"{prefix}{data_stem}N",
                row[f"conservative_difference_count_{bound}"],
            )
            _add(
                values,
                f"{prefix}{data_stem}Points",
                format_points(row[f"conservative_difference_rate_{bound}"]),
            )
            interval = row["cluster_bootstrap_95_intervals"][
                f"conservative_difference_rate_{bound}"
            ]
            _add(
                values,
                f"{prefix}{data_stem}CILowerPoints",
                format_points(interval["lower"]),
            )
            _add(
                values,
                f"{prefix}{data_stem}CIUpperPoints",
                format_points(interval["upper"]),
            )

    primary = decoded["primary_endpoints"]
    target_equality = primary["target"]["policy_vs_fresh_rebuild_equality"]
    target_exact = _endpoint_count(
        target_equality["trimmed_exact"],
        name="target policy/rebuild exact equality",
    )
    target_normalized = _endpoint_count(
        target_equality["deterministic_any"],
        name="target policy/rebuild normalized equality",
    )
    _add(values, "LMChatVThreeTargetPolicyRebuildExactN", target_exact)
    _add(
        values,
        "LMChatVThreeTargetPolicyRebuildExactPct",
        format_percentage(target_exact / histories),
    )
    _add(
        values,
        "LMChatVThreeTargetPolicyRebuildNormalizedN",
        target_normalized,
    )
    _add(
        values,
        "LMChatVThreeTargetPolicyRebuildNormalizedPct",
        format_percentage(target_normalized / histories),
    )

    retained = primary["retained"]
    retained_correctness = retained["intent_to_treat_directional_correctness"]
    for condition, stem in (
        ("present", "Present"),
        ("fresh_raw_omission", "Rebuild"),
        ("exact_decrement_or_refit_policy", "Policy"),
    ):
        count = _endpoint_count(
            retained_correctness[condition]["deterministic_any"],
            name=f"retained {condition} correctness",
        )
        _add(values, f"LMChatVThreeRetained{stem}CorrectN", count)
        _add(
            values,
            f"LMChatVThreeRetained{stem}CorrectPct",
            format_percentage(count / histories),
        )

    for comparison, stem in (
        ("policy_vs_present_equality", "PolicyPresent"),
        ("policy_vs_fresh_rebuild_equality", "PolicyRebuild"),
    ):
        equality = retained[comparison]
        exact = _endpoint_count(
            equality["trimmed_exact"],
            name=f"retained {comparison} exact equality",
        )
        normalized = _endpoint_count(
            equality["deterministic_any"],
            name=f"retained {comparison} normalized equality",
        )
        _add(values, f"LMChatVThreeRetained{stem}ExactN", exact)
        _add(
            values,
            f"LMChatVThreeRetained{stem}ExactPct",
            format_percentage(exact / histories),
        )
        _add(values, f"LMChatVThreeRetained{stem}NormalizedN", normalized)
        _add(
            values,
            f"LMChatVThreeRetained{stem}NormalizedPct",
            format_percentage(normalized / histories),
        )

    suffix = decoded["suffix_strata"]
    _add(
        values,
        "LMChatVThreeSuffixContaminatedHistoryN",
        suffix["contaminated_histories"],
    )
    _add(
        values,
        "LMChatVThreeSuffixCleanHistoryN",
        suffix["clean_histories"],
    )
    _add(
        values,
        "LMChatVThreeSuffixContaminatedClusterN",
        suffix["clusters_with_any_contaminated_history"],
    )

    for role, stem in (
        ("decoded_summary", "Summary"),
        ("census_statistics", "Census"),
        ("adjudicated_census", "AdjudicatedCensus"),
    ):
        binding = bindings[role]
        _add(values, f"LMChatVThree{stem}FileSHA", binding["file_sha256"])
        _add(
            values,
            f"LMChatVThree{stem}PayloadSHA",
            binding["payload_sha256"],
        )
        _add(
            values,
            f"LMChatVThree{stem}IntegritySHA",
            binding["integrity_sha256"],
        )
    return values


def assert_source_free_latex(rendered: str) -> None:
    """Require generated lines to contain only numeric or SHA macro values."""

    if not rendered.startswith(_HEADER) or not rendered.endswith("\n"):
        raise PaperMacroError("generated header or final newline differs")
    names: set[str] = set()
    for line in rendered[len(_HEADER) :].splitlines():
        match = _MACRO_LINE_RE.fullmatch(line)
        if match is None:
            raise PaperMacroError("generated output contains a non-macro line")
        name = match.group("name")
        value = match.group("value")
        if name in names or _SAFE_VALUE_RE.fullmatch(value) is None:
            raise PaperMacroError("generated output is not source-free")
        names.add(name)
    if not names:
        raise PaperMacroError("generated output contains no macros")


def render_macros(values: Mapping[str, str]) -> str:
    """Render macros in the validated insertion order."""

    lines = [
        f"\\providecommand{{\\{name}}}{{{value}}}"
        for name, value in values.items()
    ]
    rendered = _HEADER + "\n".join(lines) + "\n"
    assert_source_free_latex(rendered)
    return rendered


def build_paper_macros(
    *,
    decoded_summary_path: str | Path = DEFAULT_DECODED_SUMMARY_PATH,
    census_statistics_path: str | Path = DEFAULT_CENSUS_STATISTICS_PATH,
    adjudicated_census_path: str | Path = DEFAULT_ADJUDICATED_CENSUS_PATH,
) -> str:
    """Validate exact inputs and return deterministic source-free LaTeX."""

    decoded, census, adjudicated, bindings = load_validated_inputs(
        decoded_summary_path=decoded_summary_path,
        census_statistics_path=census_statistics_path,
        adjudicated_census_path=adjudicated_census_path,
    )
    return render_macros(
        collect_macro_values(decoded, census, adjudicated, bindings)
    )


def write_macros(
    path: str | Path,
    rendered: str,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically create output, replacing it only with explicit permission."""

    assert_source_free_latex(rendered)
    output = Path(path).expanduser()
    if output.is_symlink() or output.parent.is_symlink():
        raise PaperMacroError("output path must not use a symbolic link")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = rendered.encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor_open = False
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        if overwrite:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"{output} already exists; pass --overwrite to replace it"
                ) from exc
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoded-summary",
        type=Path,
        default=DEFAULT_DECODED_SUMMARY_PATH,
    )
    parser.add_argument(
        "--census-statistics",
        type=Path,
        default=DEFAULT_CENSUS_STATISTICS_PATH,
    )
    parser.add_argument(
        "--adjudicated-census",
        type=Path,
        default=DEFAULT_ADJUDICATED_CENSUS_PATH,
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and render in memory without writing output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing output file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.validate_only and args.overwrite:
        parser.error("--overwrite cannot be combined with --validate-only")
    try:
        rendered = build_paper_macros(
            decoded_summary_path=args.decoded_summary,
            census_statistics_path=args.census_statistics,
            adjudicated_census_path=args.adjudicated_census,
        )
        if args.validate_only:
            print("validated LongMemEval chat v3 paper macro inputs")
            return 0
        write_macros(args.output, rendered, overwrite=args.overwrite)
    except (OSError, PaperMacroError) as exc:
        raise SystemExit(f"paper macro build failed: {exc}") from exc
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

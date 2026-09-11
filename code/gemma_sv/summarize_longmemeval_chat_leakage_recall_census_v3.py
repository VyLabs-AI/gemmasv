"""Build source-free statistics from the completed Luna matcher-negative census.

This additive postprocessor only reads the immutable v1/v2 ledgers and run
evidence.  It performs no network or model calls and never alters run evidence.
When an output path is supplied, creation is atomic and overwrite is forbidden.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import (
    longmemeval_chat_leakage_recall_judge_protocol_v1 as judge_protocol,
)
from gemma_sv import longmemeval_chat_leakage_recall_openai_v1 as v1
from gemma_sv import longmemeval_chat_leakage_recall_openai_v2 as v2
from gemma_sv import summarize_longmemeval_chat_v3_utility as splitmix_stats


SUMMARY_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-census-statistics-v3"
)
SCHEMA_VERSION = 3
STATUS = "source-free-post-hoc-census-statistics"

EXPECTED_CLUSTERS = judge_protocol.EXPECTED_CLUSTERS
EXPECTED_HISTORIES = judge_protocol.EXPECTED_HISTORIES
HISTORIES_PER_CLUSTER = judge_protocol.HISTORIES_PER_CLUSTER
EXPECTED_POPULATION = v2.EXPECTED_POPULATION
EXPECTED_CLEAN = v2.EXPECTED_CLEAN
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = v2.BOOTSTRAP_SEED
CONFIDENCE_LEVEL = 0.95

CONDITIONS = (
    "present",
    "fresh_rebuild",
    "edited_policy",
    "prompt_only",
)
CONTRASTS = (
    ("edited_policy_minus_fresh_rebuild", 2, 1),
    ("prompt_only_minus_edited_policy", 3, 2),
    ("present_minus_edited_policy", 0, 2),
)
FAILURE_OUTCOMES = ("parse_error", "transport_error", "missing")
UNCERTAIN_OUTCOMES = ("ambiguous", *FAILURE_OUTCOMES)
ANALYSIS_OUTCOMES = tuple(judge_protocol.ANALYSIS_OUTCOMES)

_ARTIFACT_ROLES = (
    "v1_sample_lock",
    "v1_rubric",
    "v1_authorization",
    "v1_local_ledger",
    "v1_sample_summary",
    "v2_extension_lock",
    "v2_authorization",
    "v2_local_ledger",
    "v2_combined_summary",
)
_RUN_ROLES = ("v1", "v2")
_SHA256_CHARS = frozenset("0123456789abcdef")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "brief_reason",
        "candidate",
        "candidate_value",
        "cluster_id",
        "cluster_index",
        "evidence_span",
        "full_history",
        "history_id",
        "history_index",
        "parsed_response",
        "question",
        "question_value",
        "raw_output",
        "record_id",
        "reference",
        "reference_value",
        "response_id",
        "sample_index",
        "source_id",
        "source_text",
        "unit_binding_sha256",
        "variant_index",
    }
)

_SCOPE = {
    "judge_validates_matcher_recall_only": True,
    "matcher_positives_treated_as_leaks": True,
    "matcher_precision_validated": False,
    "statement": (
        "Matcher precision is unvalidated because deterministic matcher "
        "positives were not judged."
    ),
}
_INDICATOR_DEFINITION = {
    "deterministic_matcher_positive": {"lower": 1, "upper": 1},
    "matcher_clean_judged_leak": {"lower": 1, "upper": 1},
    "matcher_clean_judged_no_leak": {"lower": 0, "upper": 0},
    "matcher_clean_ambiguous_parse_transport_or_missing": {
        "lower": 0,
        "upper": 1,
    },
}
_HUMAN_SPOT_CHECK = {
    "status": "pending",
    "used_in_census_statistics": False,
}
_MISS_RATE_METHOD = {
    "estimand": "exact_census_proportion_among_matcher_clean_outputs",
    "sample_estimate": False,
    "confidence_interval_reported": False,
    "wilson_interval_used": False,
    "rule_of_three_used": False,
}


class CensusSummaryError(ValueError):
    """An immutable input, census invariant, or public projection differs."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_CHARS
    )


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": payload_sha256(result),
    }
    return result


def _validate_seal(value: Mapping[str, Any], *, name: str) -> None:
    body = copy.deepcopy(dict(value))
    integrity = body.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"algorithm", "scope", "sha256"}
        or integrity.get("algorithm") != "sha256"
        or integrity.get("scope")
        != "canonical JSON excluding this integrity object"
        or integrity.get("sha256") != payload_sha256(body)
    ):
        raise CensusSummaryError(f"{name} integrity differs")


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CensusSummaryError(f"{name} fields differ")
    return value


def _regular_input(path: str | Path, *, name: str) -> Path:
    resolved = Path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise CensusSummaryError(f"{name} must be a regular non-symlink file")
    return resolved


def _load_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    checked = _regular_input(path, name=name)
    try:
        return v1._load_mapping(checked, name=name)
    except (OSError, ValueError) as exc:
        raise CensusSummaryError(f"{name} is not strict JSON") from exc


def _file_binding(
    path: str | Path,
    value: Mapping[str, Any],
) -> dict[str, str]:
    checked = _regular_input(path, name="bound input")
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping) or not _is_sha256(
        integrity.get("sha256")
    ):
        raise CensusSummaryError("bound input has no valid integrity digest")
    return {
        "file_sha256": file_sha256(checked),
        "payload_sha256": payload_sha256(value),
        "integrity_sha256": str(integrity["sha256"]),
    }


def _corpus_binding(
    root: str | Path,
    phase: str,
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for index in sorted(records):
        path = Path(root) / phase / f"{index:03d}.json"
        rows.append(
            {
                "slot_ordinal": index,
                **_file_binding(path, records[index]),
            }
        )
    canonical = v2._evidence_corpus_binding(records)
    return {
        "record_count": len(rows),
        "canonical_record_bindings_sha256": canonical[
            "canonical_record_bindings_sha256"
        ],
        "canonical_file_bindings_sha256": payload_sha256(rows),
    }


def _build_input_bindings(
    *,
    artifact_paths: Mapping[str, str | Path],
    artifact_values: Mapping[str, Mapping[str, Any]],
    v1_run_root: str | Path,
    v1_manifest: Mapping[str, Any],
    v1_started: Mapping[int, Mapping[str, Any]],
    v1_terminals: Mapping[int, Mapping[str, Any]],
    v2_run_root: str | Path,
    v2_manifest: Mapping[str, Any],
    v2_started: Mapping[int, Mapping[str, Any]],
    v2_terminals: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(artifact_paths) != set(_ARTIFACT_ROLES) or set(
        artifact_values
    ) != set(_ARTIFACT_ROLES):
        raise CensusSummaryError("artifact binding roles differ")
    artifacts = {
        role: _file_binding(artifact_paths[role], artifact_values[role])
        for role in _ARTIFACT_ROLES
    }
    runs = {
        "v1": {
            "run_manifest": _file_binding(
                Path(v1_run_root) / "run.json",
                v1_manifest,
            ),
            "started_corpus": _corpus_binding(
                v1_run_root,
                "started",
                v1_started,
            ),
            "terminal_corpus": _corpus_binding(
                v1_run_root,
                "terminal",
                v1_terminals,
            ),
        },
        "v2": {
            "run_manifest": _file_binding(
                Path(v2_run_root) / "run.json",
                v2_manifest,
            ),
            "started_corpus": _corpus_binding(
                v2_run_root,
                "started",
                v2_started,
            ),
            "terminal_corpus": _corpus_binding(
                v2_run_root,
                "terminal",
                v2_terminals,
            ),
        },
    }
    bound = {"artifacts": artifacts, "run_evidence": runs}
    return {**bound, "binding_set_sha256": payload_sha256(bound)}


def _validate_input_bindings(value: Any) -> None:
    bindings = _require_exact_keys(
        value,
        {"artifacts", "run_evidence", "binding_set_sha256"},
        name="input bindings",
    )
    artifacts = _require_exact_keys(
        bindings["artifacts"],
        set(_ARTIFACT_ROLES),
        name="artifact bindings",
    )
    for role in _ARTIFACT_ROLES:
        descriptor = _require_exact_keys(
            artifacts[role],
            {"file_sha256", "payload_sha256", "integrity_sha256"},
            name=f"{role} binding",
        )
        if any(not _is_sha256(digest) for digest in descriptor.values()):
            raise CensusSummaryError(f"{role} binding digest differs")
    runs = _require_exact_keys(
        bindings["run_evidence"],
        set(_RUN_ROLES),
        name="run evidence bindings",
    )
    expected_counts = {"v1": v1.REQUEST_COUNT, "v2": v2.REQUEST_COUNT}
    for run_role in _RUN_ROLES:
        run = _require_exact_keys(
            runs[run_role],
            {"run_manifest", "started_corpus", "terminal_corpus"},
            name=f"{run_role} run binding",
        )
        manifest = _require_exact_keys(
            run["run_manifest"],
            {"file_sha256", "payload_sha256", "integrity_sha256"},
            name=f"{run_role} manifest binding",
        )
        if any(not _is_sha256(digest) for digest in manifest.values()):
            raise CensusSummaryError(
                f"{run_role} manifest binding digest differs"
            )
        for phase in ("started_corpus", "terminal_corpus"):
            corpus = _require_exact_keys(
                run[phase],
                {
                    "record_count",
                    "canonical_record_bindings_sha256",
                    "canonical_file_bindings_sha256",
                },
                name=f"{run_role} {phase} binding",
            )
            if (
                corpus.get("record_count") != expected_counts[run_role]
                or not _is_sha256(
                    corpus.get("canonical_record_bindings_sha256")
                )
                or not _is_sha256(
                    corpus.get("canonical_file_bindings_sha256")
                )
            ):
                raise CensusSummaryError(
                    f"{run_role} {phase} corpus binding differs"
                )
    bound = {
        "artifacts": bindings["artifacts"],
        "run_evidence": bindings["run_evidence"],
    }
    if bindings.get("binding_set_sha256") != payload_sha256(bound):
        raise CensusSummaryError("input binding-set digest differs")


def _normalize_population_geometry(
    population_units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    try:
        population = v2._normalize_population(population_units)
    except ValueError as exc:
        raise CensusSummaryError("population validation failed") from exc
    cells: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in population:
        ordinal = int(row["condition_ordinal"])
        history = int(row["history_index"])
        key = (ordinal, history)
        if (
            key in cells
            or row["cluster_index"] != history // HISTORIES_PER_CLUSTER
            or row["variant_index"] != history % HISTORIES_PER_CLUSTER
        ):
            raise CensusSummaryError(
                "population is not 32 clusters of three nested histories"
            )
        cells[key] = row
    expected_cells = {
        (ordinal, history)
        for ordinal in range(len(CONDITIONS))
        for history in range(EXPECTED_HISTORIES)
    }
    positive_counts = tuple(
        sum(
            row["condition_ordinal"] == ordinal
            and row["deterministic_any"]
            for row in population
        )
        for ordinal in range(len(CONDITIONS))
    )
    if (
        len(population) != EXPECTED_POPULATION
        or set(cells) != expected_cells
        or positive_counts != v2.EXPECTED_MATCHER_POSITIVE_BY_CONDITION
    ):
        raise CensusSummaryError("population census geometry differs")
    return population


def _normalize_outcomes(
    population: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, str],
) -> dict[str, str]:
    clean_bindings = {
        str(row["unit_binding_sha256"])
        for row in population
        if not row["deterministic_any"]
    }
    rendered = dict(outcomes)
    if (
        len(clean_bindings) != EXPECTED_CLEAN
        or set(rendered) != clean_bindings
        or any(value not in ANALYSIS_OUTCOMES for value in rendered.values())
    ):
        raise CensusSummaryError(
            "outcomes must cover all 253 matcher-clean outputs exactly once"
        )
    return rendered


@lru_cache(maxsize=128)
def _bootstrap_endpoints(
    values: tuple[float, ...],
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    try:
        interval = splitmix_stats.cluster_bootstrap_interval(
            values,
            resamples=resamples,
            seed=seed,
        )
    except ValueError as exc:
        raise CensusSummaryError("cluster bootstrap failed") from exc
    if (
        interval.get("K") != EXPECTED_CLUSTERS
        or interval.get("clusters_per_resample") != EXPECTED_CLUSTERS
        or interval.get("histories_resampled_within_cluster") is not False
        or interval.get("prng") != "splitmix64"
        or interval.get("percentile_interpolation") != "linear_type_7"
    ):
        raise CensusSummaryError("shared SplitMix64 bootstrap contract differs")
    return float(interval["lower"]), float(interval["upper"])


def cluster_bootstrap_interval(
    cluster_means: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = tuple(float(value) for value in cluster_means)
    if (
        len(values) != EXPECTED_CLUSTERS
        or any(not math.isfinite(value) for value in values)
        or type(resamples) is not int
        or resamples < 1
        or type(seed) is not int
    ):
        raise CensusSummaryError(
            "bootstrap requires 32 finite cluster means and positive settings"
        )
    lower, upper = _bootstrap_endpoints(values, resamples, seed)
    return {
        "method": "percentile_cluster_bootstrap",
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling_unit": "anonymous_target_cluster",
        "K": EXPECTED_CLUSTERS,
        "clusters_per_resample": EXPECTED_CLUSTERS,
        "histories_resampled_within_cluster": False,
        "resamples": resamples,
        "seed": seed,
        "prng": "splitmix64",
        "percentile_interpolation": "linear_type_7",
        "lower": lower,
        "upper": upper,
    }


def _exact_proportion(numerator: int, denominator: int) -> dict[str, Any]:
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or not 0 <= numerator <= denominator
        or denominator < 1
    ):
        raise CensusSummaryError("exact proportion counts differ")
    return {
        "numerator_judge_discovered_matcher_misses": numerator,
        "denominator_matcher_clean_outputs": denominator,
        "proportion": numerator / denominator,
    }


def _build_census_statistics(
    population_units: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    population = _normalize_population_geometry(population_units)
    clean_outcomes = _normalize_outcomes(population, outcomes)
    by_cell = {
        (int(row["condition_ordinal"]), int(row["history_index"])): row
        for row in population
    }

    indicators: dict[int, dict[int, tuple[int, int]]] = {
        ordinal: {} for ordinal in range(len(CONDITIONS))
    }
    outcome_counts: dict[int, Counter[str]] = {
        ordinal: Counter() for ordinal in range(len(CONDITIONS))
    }
    for ordinal in range(len(CONDITIONS)):
        for history in range(EXPECTED_HISTORIES):
            unit = by_cell[(ordinal, history)]
            if unit["deterministic_any"]:
                indicators[ordinal][history] = (1, 1)
                continue
            outcome = clean_outcomes[str(unit["unit_binding_sha256"])]
            outcome_counts[ordinal][outcome] += 1
            if outcome == "leak":
                indicators[ordinal][history] = (1, 1)
            elif outcome == "no_leak":
                indicators[ordinal][history] = (0, 0)
            else:
                indicators[ordinal][history] = (0, 1)

    cluster_lower: dict[int, list[float]] = {}
    cluster_upper: dict[int, list[float]] = {}
    per_condition = []
    anonymous_rows = []
    for ordinal, condition in enumerate(CONDITIONS):
        lower_means = []
        upper_means = []
        for cluster in range(EXPECTED_CLUSTERS):
            histories = range(
                cluster * HISTORIES_PER_CLUSTER,
                (cluster + 1) * HISTORIES_PER_CLUSTER,
            )
            lower_means.append(
                sum(indicators[ordinal][history][0] for history in histories)
                / HISTORIES_PER_CLUSTER
            )
            upper_means.append(
                sum(indicators[ordinal][history][1] for history in histories)
                / HISTORIES_PER_CLUSTER
            )
        cluster_lower[ordinal] = lower_means
        cluster_upper[ordinal] = upper_means
        counts = outcome_counts[ordinal]
        positive = v2.EXPECTED_MATCHER_POSITIVE_BY_CONDITION[ordinal]
        clean = v2.EXPECTED_CLEAN_BY_CONDITION[ordinal]
        failures = sum(counts[name] for name in FAILURE_OUTCOMES)
        lower_count = positive + counts["leak"]
        upper_count = lower_count + counts["ambiguous"] + failures
        per_condition.append(
            {
                "condition": condition,
                "condition_ordinal": ordinal,
                "histories": EXPECTED_HISTORIES,
                "deterministic_matcher_positive_count": positive,
                "matcher_clean_output_count": clean,
                "judge_discovered_matcher_misses": counts["leak"],
                "judge_no_leak": counts["no_leak"],
                "judge_ambiguous": counts["ambiguous"],
                "judge_failures": failures,
                "judge_failure_breakdown": {
                    name: counts[name] for name in FAILURE_OUTCOMES
                },
                "conservative_leakage_count_lower": lower_count,
                "conservative_leakage_count_upper": upper_count,
                "conservative_leakage_rate_lower": (
                    lower_count / EXPECTED_HISTORIES
                ),
                "conservative_leakage_rate_upper": (
                    upper_count / EXPECTED_HISTORIES
                ),
                "cluster_bootstrap_95_intervals": {
                    "conservative_leakage_rate_lower": (
                        cluster_bootstrap_interval(
                            lower_means,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                    "conservative_leakage_rate_upper": (
                        cluster_bootstrap_interval(
                            upper_means,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                },
                "matcher_miss_rate_among_clean_outputs": _exact_proportion(
                    counts["leak"],
                    clean,
                ),
            }
        )
        anonymous_rows.append(
            {
                "condition": condition,
                "leakage_indicator_mean_lower": lower_means,
                "leakage_indicator_mean_upper": upper_means,
            }
        )

    paired = []
    for contrast, minuend, subtrahend in CONTRASTS:
        minimum_means = [
            cluster_lower[minuend][index]
            - cluster_upper[subtrahend][index]
            for index in range(EXPECTED_CLUSTERS)
        ]
        maximum_means = [
            cluster_upper[minuend][index]
            - cluster_lower[subtrahend][index]
            for index in range(EXPECTED_CLUSTERS)
        ]
        minimum_count = (
            per_condition[minuend]["conservative_leakage_count_lower"]
            - per_condition[subtrahend]["conservative_leakage_count_upper"]
        )
        maximum_count = (
            per_condition[minuend]["conservative_leakage_count_upper"]
            - per_condition[subtrahend]["conservative_leakage_count_lower"]
        )
        paired.append(
            {
                "contrast": contrast,
                "minuend_condition": CONDITIONS[minuend],
                "subtrahend_condition": CONDITIONS[subtrahend],
                "conservative_difference_count_minimum": minimum_count,
                "conservative_difference_count_maximum": maximum_count,
                "conservative_difference_rate_minimum": (
                    minimum_count / EXPECTED_HISTORIES
                ),
                "conservative_difference_rate_maximum": (
                    maximum_count / EXPECTED_HISTORIES
                ),
                "anonymous_cluster_difference_mean_minimum": minimum_means,
                "anonymous_cluster_difference_mean_maximum": maximum_means,
                "cluster_bootstrap_95_intervals": {
                    "conservative_difference_rate_minimum": (
                        cluster_bootstrap_interval(
                            minimum_means,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                    "conservative_difference_rate_maximum": (
                        cluster_bootstrap_interval(
                            maximum_means,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                },
                "ambiguity_bounds_induced": True,
            }
        )

    combined_counts = Counter(clean_outcomes.values())
    miss_total = sum(
        row["judge_discovered_matcher_misses"] for row in per_condition
    )
    return {
        "geometry": {
            "conditions": len(CONDITIONS),
            "histories_per_condition": EXPECTED_HISTORIES,
            "K": EXPECTED_CLUSTERS,
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "population_units": EXPECTED_POPULATION,
            "matcher_clean_outputs": EXPECTED_CLEAN,
            "all_matcher_clean_outputs_censused": True,
            "nested_histories_complete": True,
            "matcher_clean_outcome_counts": {
                name: combined_counts[name] for name in ANALYSIS_OUTCOMES
            },
        },
        "per_condition": per_condition,
        "anonymous_cluster_means": {
            "ordered_by_anonymous_cluster_ordinal": True,
            "cluster_identifiers_included": False,
            "history_identifiers_included": False,
            "text_included": False,
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "per_condition": anonymous_rows,
        },
        "paired_differences": paired,
        "matcher_miss_rate_reporting": {
            **_MISS_RATE_METHOD,
            "overall": _exact_proportion(miss_total, EXPECTED_CLEAN),
        },
    }


def build_source_free_summary(
    *,
    population_units: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, str],
    input_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Project validated private values to a sealed aggregate-only summary."""

    _validate_input_bindings(input_bindings)
    statistics = _build_census_statistics(population_units, outcomes)
    summary = _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "source_free": True,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "contains_source_or_response_ids": False,
            "post_hoc": True,
            "postprocessor_only": True,
            "network_or_model_calls_made": 0,
            "scope": copy.deepcopy(_SCOPE),
            "indicator_definition": copy.deepcopy(_INDICATOR_DEFINITION),
            "input_bindings": copy.deepcopy(dict(input_bindings)),
            **statistics,
            "human_spot_check": copy.deepcopy(_HUMAN_SPOT_CHECK),
        }
    )
    validate_summary(summary)
    return summary


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk(child)


def assert_source_free(value: Mapping[str, Any]) -> None:
    for key, child in _walk(value):
        if key is not None and key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
            raise CensusSummaryError(
                f"source-free summary contains prohibited key {key!r}"
            )
        if isinstance(child, str) and child == v2.MODEL:
            raise CensusSummaryError(
                "source-free summary contains the provider model alias"
            )


def _close(left: Any, right: float) -> bool:
    return (
        type(left) in (int, float)
        and math.isfinite(float(left))
        and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1e-15)
    )


def _validate_interval(
    value: Any,
    *,
    cluster_means: Sequence[float],
    minimum: float,
    maximum: float,
) -> None:
    interval = _require_exact_keys(
        value,
        {
            "method",
            "confidence_level",
            "resampling_unit",
            "K",
            "clusters_per_resample",
            "histories_resampled_within_cluster",
            "resamples",
            "seed",
            "prng",
            "percentile_interpolation",
            "lower",
            "upper",
        },
        name="cluster bootstrap interval",
    )
    expected = cluster_bootstrap_interval(cluster_means)
    if dict(interval) != expected:
        raise CensusSummaryError(
            "cluster bootstrap interval is not the deterministic "
            "SplitMix64/type-7 result"
        )
    if not minimum <= float(interval["lower"]) <= float(
        interval["upper"]
    ) <= maximum:
        raise CensusSummaryError("cluster bootstrap endpoints are out of range")


def _validate_exact_proportion(
    value: Any,
    *,
    numerator: int,
    denominator: int,
) -> None:
    proportion = _require_exact_keys(
        value,
        {
            "numerator_judge_discovered_matcher_misses",
            "denominator_matcher_clean_outputs",
            "proportion",
        },
        name="matcher miss proportion",
    )
    if dict(proportion) != _exact_proportion(numerator, denominator):
        raise CensusSummaryError("matcher miss exact-census proportion differs")


def validate_summary(summary: Mapping[str, Any]) -> None:
    _validate_seal(summary, name="v3 census summary")
    _require_exact_keys(
        summary,
        {
            "schema",
            "schema_version",
            "status",
            "source_free",
            "contains_source_text",
            "contains_model_generated_text",
            "contains_source_or_response_ids",
            "post_hoc",
            "postprocessor_only",
            "network_or_model_calls_made",
            "scope",
            "indicator_definition",
            "input_bindings",
            "geometry",
            "per_condition",
            "anonymous_cluster_means",
            "paired_differences",
            "matcher_miss_rate_reporting",
            "human_spot_check",
            "integrity",
        },
        name="v3 census summary",
    )
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != STATUS
        or summary.get("source_free") is not True
        or summary.get("contains_source_text") is not False
        or summary.get("contains_model_generated_text") is not False
        or summary.get("contains_source_or_response_ids") is not False
        or summary.get("post_hoc") is not True
        or summary.get("postprocessor_only") is not True
        or summary.get("network_or_model_calls_made") != 0
        or summary.get("scope") != _SCOPE
        or summary.get("indicator_definition") != _INDICATOR_DEFINITION
        or summary.get("human_spot_check") != _HUMAN_SPOT_CHECK
    ):
        raise CensusSummaryError("v3 census summary contract differs")
    assert_source_free(summary)
    _validate_input_bindings(summary["input_bindings"])

    geometry = _require_exact_keys(
        summary["geometry"],
        {
            "conditions",
            "histories_per_condition",
            "K",
            "histories_per_cluster",
            "population_units",
            "matcher_clean_outputs",
            "all_matcher_clean_outputs_censused",
            "nested_histories_complete",
            "matcher_clean_outcome_counts",
        },
        name="census geometry",
    )
    outcomes = _require_exact_keys(
        geometry["matcher_clean_outcome_counts"],
        set(ANALYSIS_OUTCOMES),
        name="matcher-clean outcome counts",
    )
    if (
        geometry.get("conditions") != len(CONDITIONS)
        or geometry.get("histories_per_condition") != EXPECTED_HISTORIES
        or geometry.get("K") != EXPECTED_CLUSTERS
        or geometry.get("histories_per_cluster") != HISTORIES_PER_CLUSTER
        or geometry.get("population_units") != EXPECTED_POPULATION
        or geometry.get("matcher_clean_outputs") != EXPECTED_CLEAN
        or geometry.get("all_matcher_clean_outputs_censused") is not True
        or geometry.get("nested_histories_complete") is not True
        or any(type(count) is not int or count < 0 for count in outcomes.values())
        or sum(outcomes.values()) != EXPECTED_CLEAN
    ):
        raise CensusSummaryError("census geometry or coverage differs")

    anonymous = _require_exact_keys(
        summary["anonymous_cluster_means"],
        {
            "ordered_by_anonymous_cluster_ordinal",
            "cluster_identifiers_included",
            "history_identifiers_included",
            "text_included",
            "histories_per_cluster",
            "per_condition",
        },
        name="anonymous cluster means",
    )
    if (
        anonymous.get("ordered_by_anonymous_cluster_ordinal") is not True
        or anonymous.get("cluster_identifiers_included") is not False
        or anonymous.get("history_identifiers_included") is not False
        or anonymous.get("text_included") is not False
        or anonymous.get("histories_per_cluster") != HISTORIES_PER_CLUSTER
        or not isinstance(anonymous.get("per_condition"), list)
        or len(anonymous["per_condition"]) != len(CONDITIONS)
    ):
        raise CensusSummaryError("anonymous cluster projection differs")

    condition_rows = summary.get("per_condition")
    if not isinstance(condition_rows, list) or len(condition_rows) != len(
        CONDITIONS
    ):
        raise CensusSummaryError("per-condition census rows differ")
    lower_clusters: dict[int, list[float]] = {}
    upper_clusters: dict[int, list[float]] = {}
    aggregate_outcomes = Counter()
    total_misses = 0
    condition_keys = {
        "condition",
        "condition_ordinal",
        "histories",
        "deterministic_matcher_positive_count",
        "matcher_clean_output_count",
        "judge_discovered_matcher_misses",
        "judge_no_leak",
        "judge_ambiguous",
        "judge_failures",
        "judge_failure_breakdown",
        "conservative_leakage_count_lower",
        "conservative_leakage_count_upper",
        "conservative_leakage_rate_lower",
        "conservative_leakage_rate_upper",
        "cluster_bootstrap_95_intervals",
        "matcher_miss_rate_among_clean_outputs",
    }
    for ordinal, condition in enumerate(CONDITIONS):
        row = _require_exact_keys(
            condition_rows[ordinal],
            condition_keys,
            name=f"{condition} census row",
        )
        anonymous_row = _require_exact_keys(
            anonymous["per_condition"][ordinal],
            {
                "condition",
                "leakage_indicator_mean_lower",
                "leakage_indicator_mean_upper",
            },
            name=f"{condition} anonymous cluster means",
        )
        lower_means = anonymous_row["leakage_indicator_mean_lower"]
        upper_means = anonymous_row["leakage_indicator_mean_upper"]
        if (
            row.get("condition") != condition
            or row.get("condition_ordinal") != ordinal
            or anonymous_row.get("condition") != condition
            or row.get("histories") != EXPECTED_HISTORIES
            or row.get("deterministic_matcher_positive_count")
            != v2.EXPECTED_MATCHER_POSITIVE_BY_CONDITION[ordinal]
            or row.get("matcher_clean_output_count")
            != v2.EXPECTED_CLEAN_BY_CONDITION[ordinal]
            or not isinstance(lower_means, list)
            or not isinstance(upper_means, list)
            or len(lower_means) != EXPECTED_CLUSTERS
            or len(upper_means) != EXPECTED_CLUSTERS
        ):
            raise CensusSummaryError(f"{condition} geometry differs")
        for lower, upper in zip(lower_means, upper_means):
            if (
                type(lower) not in (int, float)
                or type(upper) not in (int, float)
                or not 0 <= float(lower) <= float(upper) <= 1
                or not _close(
                    float(lower) * HISTORIES_PER_CLUSTER,
                    round(float(lower) * HISTORIES_PER_CLUSTER),
                )
                or not _close(
                    float(upper) * HISTORIES_PER_CLUSTER,
                    round(float(upper) * HISTORIES_PER_CLUSTER),
                )
            ):
                raise CensusSummaryError(
                    f"{condition} cluster means are not three-history means"
                )
        lower_clusters[ordinal] = [float(value) for value in lower_means]
        upper_clusters[ordinal] = [float(value) for value in upper_means]

        failure_breakdown = _require_exact_keys(
            row["judge_failure_breakdown"],
            set(FAILURE_OUTCOMES),
            name=f"{condition} failure breakdown",
        )
        values_to_check = (
            row.get("judge_discovered_matcher_misses"),
            row.get("judge_no_leak"),
            row.get("judge_ambiguous"),
            row.get("judge_failures"),
            *failure_breakdown.values(),
        )
        if any(type(value) is not int or value < 0 for value in values_to_check):
            raise CensusSummaryError(f"{condition} judge counts differ")
        failures = sum(failure_breakdown.values())
        clean = v2.EXPECTED_CLEAN_BY_CONDITION[ordinal]
        positive = v2.EXPECTED_MATCHER_POSITIVE_BY_CONDITION[ordinal]
        misses = row["judge_discovered_matcher_misses"]
        lower_count = positive + misses
        upper_count = lower_count + row["judge_ambiguous"] + failures
        if (
            row["judge_failures"] != failures
            or misses
            + row["judge_no_leak"]
            + row["judge_ambiguous"]
            + failures
            != clean
            or row.get("conservative_leakage_count_lower") != lower_count
            or row.get("conservative_leakage_count_upper") != upper_count
            or not _close(
                row.get("conservative_leakage_rate_lower"),
                lower_count / EXPECTED_HISTORIES,
            )
            or not _close(
                row.get("conservative_leakage_rate_upper"),
                upper_count / EXPECTED_HISTORIES,
            )
            or not _close(
                math.fsum(lower_clusters[ordinal]) / EXPECTED_CLUSTERS,
                lower_count / EXPECTED_HISTORIES,
            )
            or not _close(
                math.fsum(upper_clusters[ordinal]) / EXPECTED_CLUSTERS,
                upper_count / EXPECTED_HISTORIES,
            )
        ):
            raise CensusSummaryError(f"{condition} census arithmetic differs")
        intervals = _require_exact_keys(
            row["cluster_bootstrap_95_intervals"],
            {
                "conservative_leakage_rate_lower",
                "conservative_leakage_rate_upper",
            },
            name=f"{condition} bootstrap intervals",
        )
        _validate_interval(
            intervals["conservative_leakage_rate_lower"],
            cluster_means=lower_clusters[ordinal],
            minimum=0.0,
            maximum=1.0,
        )
        _validate_interval(
            intervals["conservative_leakage_rate_upper"],
            cluster_means=upper_clusters[ordinal],
            minimum=0.0,
            maximum=1.0,
        )
        _validate_exact_proportion(
            row["matcher_miss_rate_among_clean_outputs"],
            numerator=misses,
            denominator=clean,
        )
        total_misses += misses
        aggregate_outcomes.update(
            {
                "leak": misses,
                "no_leak": row["judge_no_leak"],
                "ambiguous": row["judge_ambiguous"],
                **dict(failure_breakdown),
            }
        )
    if dict(outcomes) != {
        name: aggregate_outcomes[name] for name in ANALYSIS_OUTCOMES
    }:
        raise CensusSummaryError(
            "per-condition outcomes do not reconcile to census coverage"
        )

    paired = summary.get("paired_differences")
    if not isinstance(paired, list) or len(paired) != len(CONTRASTS):
        raise CensusSummaryError("paired contrast rows differ")
    paired_keys = {
        "contrast",
        "minuend_condition",
        "subtrahend_condition",
        "conservative_difference_count_minimum",
        "conservative_difference_count_maximum",
        "conservative_difference_rate_minimum",
        "conservative_difference_rate_maximum",
        "anonymous_cluster_difference_mean_minimum",
        "anonymous_cluster_difference_mean_maximum",
        "cluster_bootstrap_95_intervals",
        "ambiguity_bounds_induced",
    }
    for index, (contrast, minuend, subtrahend) in enumerate(CONTRASTS):
        row = _require_exact_keys(
            paired[index],
            paired_keys,
            name=f"{contrast} paired row",
        )
        expected_minimum = [
            lower_clusters[minuend][cluster]
            - upper_clusters[subtrahend][cluster]
            for cluster in range(EXPECTED_CLUSTERS)
        ]
        expected_maximum = [
            upper_clusters[minuend][cluster]
            - lower_clusters[subtrahend][cluster]
            for cluster in range(EXPECTED_CLUSTERS)
        ]
        minimum_count = (
            condition_rows[minuend]["conservative_leakage_count_lower"]
            - condition_rows[subtrahend]["conservative_leakage_count_upper"]
        )
        maximum_count = (
            condition_rows[minuend]["conservative_leakage_count_upper"]
            - condition_rows[subtrahend]["conservative_leakage_count_lower"]
        )
        if (
            row.get("contrast") != contrast
            or row.get("minuend_condition") != CONDITIONS[minuend]
            or row.get("subtrahend_condition") != CONDITIONS[subtrahend]
            or row.get("conservative_difference_count_minimum")
            != minimum_count
            or row.get("conservative_difference_count_maximum")
            != maximum_count
            or not _close(
                row.get("conservative_difference_rate_minimum"),
                minimum_count / EXPECTED_HISTORIES,
            )
            or not _close(
                row.get("conservative_difference_rate_maximum"),
                maximum_count / EXPECTED_HISTORIES,
            )
            or row.get("anonymous_cluster_difference_mean_minimum")
            != expected_minimum
            or row.get("anonymous_cluster_difference_mean_maximum")
            != expected_maximum
            or row.get("ambiguity_bounds_induced") is not True
        ):
            raise CensusSummaryError(f"{contrast} paired arithmetic differs")
        intervals = _require_exact_keys(
            row["cluster_bootstrap_95_intervals"],
            {
                "conservative_difference_rate_minimum",
                "conservative_difference_rate_maximum",
            },
            name=f"{contrast} intervals",
        )
        _validate_interval(
            intervals["conservative_difference_rate_minimum"],
            cluster_means=expected_minimum,
            minimum=-1.0,
            maximum=1.0,
        )
        _validate_interval(
            intervals["conservative_difference_rate_maximum"],
            cluster_means=expected_maximum,
            minimum=-1.0,
            maximum=1.0,
        )

    miss_reporting = _require_exact_keys(
        summary["matcher_miss_rate_reporting"],
        {*_MISS_RATE_METHOD, "overall"},
        name="matcher miss reporting",
    )
    if any(
        miss_reporting.get(key) != value
        for key, value in _MISS_RATE_METHOD.items()
    ):
        raise CensusSummaryError("matcher miss reporting method differs")
    _validate_exact_proportion(
        miss_reporting["overall"],
        numerator=total_misses,
        denominator=EXPECTED_CLEAN,
    )


def _load_v1_run(
    *,
    run_root: str | Path,
    expected_manifest: Mapping[str, Any],
    sample_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    try:
        root = v1._path_without_symlinks(run_root)
        v1._validate_mode(root, directory=True)
        v1._validate_mode(root / "run.json", directory=False)
        manifest = v1._load_mapping(
            root / "run.json",
            name="v1 run manifest",
        )
        v1.validate_run_manifest(manifest, expected=expected_manifest)
        started, terminals = v1._scan_evidence(
            root,
            manifest=manifest,
            sample_lock=sample_lock,
            local_ledger=local_ledger,
        )
    except (OSError, ValueError) as exc:
        raise CensusSummaryError("v1 run evidence validation failed") from exc
    expected = set(range(v1.REQUEST_COUNT))
    if set(started) != expected or set(terminals) != expected:
        raise CensusSummaryError(
            "completed v1 run must contain all started and terminal records"
        )
    return manifest, started, terminals


def _load_v2_run(
    *,
    run_root: str | Path,
    expected_manifest: Mapping[str, Any],
    extension_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    try:
        root = v2._path_without_symlinks(run_root)
        v2._validate_mode(root, directory=True)
        v2._validate_mode(root / "run.json", directory=False)
        manifest = v2._load_mapping(
            root / "run.json",
            name="v2 run manifest",
        )
        v2.validate_run_manifest(manifest, expected=expected_manifest)
        started, terminals = v2._scan_evidence(
            root,
            manifest=manifest,
            extension_lock=extension_lock,
            local_ledger=local_ledger,
        )
    except (OSError, ValueError) as exc:
        raise CensusSummaryError("v2 run evidence validation failed") from exc
    expected = set(range(v2.REQUEST_COUNT))
    if set(started) != expected or set(terminals) != expected:
        raise CensusSummaryError(
            "completed v2 run must contain all started and terminal records"
        )
    return manifest, started, terminals


def summarize_paths(
    *,
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_authorization_path: str | Path,
    v1_local_ledger_path: str | Path,
    v1_run_root: str | Path,
    v1_sample_summary_path: str | Path,
    v2_extension_lock_path: str | Path,
    v2_authorization_path: str | Path,
    v2_local_ledger_path: str | Path,
    v2_run_root: str | Path,
    v2_combined_summary_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every bound v1/v2 input and return the v3 public projection."""

    try:
        sample_lock, rubric = v1.load_public_protocol(
            sample_lock_path=v1_sample_lock_path,
            rubric_path=v1_rubric_path,
        )
        v1_ledger = v1.validate_local_ledger_file(
            v1_local_ledger_path,
            sample_lock=sample_lock,
            rubric=rubric,
        )
        v1_authorization = v1.validate_authorization_file(
            v1_authorization_path,
            sample_lock=sample_lock,
            rubric=rubric,
            local_ledger=v1_ledger,
            sample_lock_path=v1_sample_lock_path,
            rubric_path=v1_rubric_path,
        )
        v1_summary = _load_mapping(
            v1_sample_summary_path,
            name="v1 sample summary",
        )
        v1.validate_source_free_summary(v1_summary)

        extension_lock = _load_mapping(
            v2_extension_lock_path,
            name="v2 extension lock",
        )
        v2.validate_extension_lock(
            extension_lock,
            population_units=v1_ledger["population_units"],
            v1_sample_lock=sample_lock,
            v1_rubric=rubric,
        )
        v2_ledger = v2.validate_local_ledger_file(
            v2_local_ledger_path,
            extension_lock=extension_lock,
            v1_sample_lock=sample_lock,
            v1_rubric=rubric,
        )
        if v2._normalize_population(
            v1_ledger["population_units"]
        ) != v2._normalize_population(v2_ledger["population_units"]):
            raise CensusSummaryError("v1 and v2 population ledgers differ")

        v2_authorization = _load_mapping(
            v2_authorization_path,
            name="v2 authorization",
        )
        v2.validate_authorization(
            v2_authorization,
            v1_sample_lock=sample_lock,
            v1_rubric=rubric,
            v1_authorization=v1_authorization,
            v1_summary=v1_summary,
            extension_lock=extension_lock,
            local_ledger=v2_ledger,
            v1_sample_lock_path=v1_sample_lock_path,
            v1_rubric_path=v1_rubric_path,
            v1_authorization_path=v1_authorization_path,
            v1_summary_path=v1_sample_summary_path,
            extension_lock_path=v2_extension_lock_path,
        )

        expected_v1_manifest = v1.build_run_manifest(
            authorization_path=v1_authorization_path,
            authorization=v1_authorization,
            sample_lock_path=v1_sample_lock_path,
            sample_lock=sample_lock,
            rubric_path=v1_rubric_path,
            rubric=rubric,
            local_ledger_path=v1_local_ledger_path,
            local_ledger=v1_ledger,
        )
        v1_manifest, v1_started, v1_terminals = _load_v1_run(
            run_root=v1_run_root,
            expected_manifest=expected_v1_manifest,
            sample_lock=sample_lock,
            local_ledger=v1_ledger,
        )

        expected_v2_manifest = v2.build_run_manifest(
            authorization_path=v2_authorization_path,
            authorization=v2_authorization,
            extension_lock_path=v2_extension_lock_path,
            extension_lock=extension_lock,
            local_ledger_path=v2_local_ledger_path,
            local_ledger=v2_ledger,
        )
        v2_manifest, v2_started, v2_terminals = _load_v2_run(
            run_root=v2_run_root,
            expected_manifest=expected_v2_manifest,
            extension_lock=extension_lock,
            local_ledger=v2_ledger,
        )

        v2_summary = _load_mapping(
            v2_combined_summary_path,
            name="v2 combined summary",
        )
        v2.validate_source_free_summary(v2_summary)
        expected_v2_summary = v2.build_combined_source_free_summary(
            v1_sample_lock=sample_lock,
            v1_rubric=rubric,
            v1_local_ledger=v1_ledger,
            v1_summary=v1_summary,
            v1_run_manifest=v1_manifest,
            v1_started=v1_started,
            v1_terminals=v1_terminals,
            authorization=v2_authorization,
            extension_lock=extension_lock,
            local_ledger=v2_ledger,
            v2_run_manifest=v2_manifest,
            v2_started=v2_started,
            v2_terminals=v2_terminals,
        )
        if v2_summary != expected_v2_summary:
            raise CensusSummaryError(
                "v2 combined summary does not reproduce from bound evidence"
            )
        population = _normalize_population_geometry(
            v2_ledger["population_units"]
        )
        outcomes = v2._combined_outcomes(
            population=population,
            v1_sample_lock=sample_lock,
            v1_terminals=v1_terminals,
            extension_lock=extension_lock,
            v2_terminals=v2_terminals,
        )
    except CensusSummaryError:
        raise
    except (OSError, ValueError) as exc:
        raise CensusSummaryError("strict v1/v2 input validation failed") from exc

    artifact_paths = {
        "v1_sample_lock": v1_sample_lock_path,
        "v1_rubric": v1_rubric_path,
        "v1_authorization": v1_authorization_path,
        "v1_local_ledger": v1_local_ledger_path,
        "v1_sample_summary": v1_sample_summary_path,
        "v2_extension_lock": v2_extension_lock_path,
        "v2_authorization": v2_authorization_path,
        "v2_local_ledger": v2_local_ledger_path,
        "v2_combined_summary": v2_combined_summary_path,
    }
    artifact_values = {
        "v1_sample_lock": sample_lock,
        "v1_rubric": rubric,
        "v1_authorization": v1_authorization,
        "v1_local_ledger": v1_ledger,
        "v1_sample_summary": v1_summary,
        "v2_extension_lock": extension_lock,
        "v2_authorization": v2_authorization,
        "v2_local_ledger": v2_ledger,
        "v2_combined_summary": v2_summary,
    }
    input_bindings = _build_input_bindings(
        artifact_paths=artifact_paths,
        artifact_values=artifact_values,
        v1_run_root=v1_run_root,
        v1_manifest=v1_manifest,
        v1_started=v1_started,
        v1_terminals=v1_terminals,
        v2_run_root=v2_run_root,
        v2_manifest=v2_manifest,
        v2_started=v2_started,
        v2_terminals=v2_terminals,
    )
    summary = build_source_free_summary(
        population_units=population,
        outcomes=outcomes,
        input_bindings=input_bindings,
    )
    if output_path is not None:
        write_summary(output_path, summary)
    return summary


def deterministic_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def write_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    validate_summary(summary)
    v1._atomic_write_new(path, summary, local_only=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-sample-lock", type=Path, required=True)
    parser.add_argument("--v1-rubric", type=Path, required=True)
    parser.add_argument("--v1-authorization", type=Path, required=True)
    parser.add_argument("--v1-local-ledger", type=Path, required=True)
    parser.add_argument("--v1-run-root", type=Path, required=True)
    parser.add_argument("--v1-sample-summary", type=Path, required=True)
    parser.add_argument("--v2-extension-lock", type=Path, required=True)
    parser.add_argument("--v2-authorization", type=Path, required=True)
    parser.add_argument("--v2-local-ledger", type=Path, required=True)
    parser.add_argument("--v2-run-root", type=Path, required=True)
    parser.add_argument("--v2-combined-summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new output path; existing files are never overwritten.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = summarize_paths(
            v1_sample_lock_path=args.v1_sample_lock,
            v1_rubric_path=args.v1_rubric,
            v1_authorization_path=args.v1_authorization,
            v1_local_ledger_path=args.v1_local_ledger,
            v1_run_root=args.v1_run_root,
            v1_sample_summary_path=args.v1_sample_summary,
            v2_extension_lock_path=args.v2_extension_lock,
            v2_authorization_path=args.v2_authorization,
            v2_local_ledger_path=args.v2_local_ledger,
            v2_run_root=args.v2_run_root,
            v2_combined_summary_path=args.v2_combined_summary,
            output_path=args.output,
        )
    except (CensusSummaryError, FileExistsError) as exc:
        parser.error(str(exc))
    if args.output is None:
        print(deterministic_json(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

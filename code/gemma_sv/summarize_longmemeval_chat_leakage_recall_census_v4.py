"""Build the source-free human-adjudicated LongMemEval census v4.

The immutable Luna census v3 remains unchanged. This derivative applies final
human labels only to the 38 reviewed matcher-negative output occurrences,
retains Luna labels for unreviewed outputs, and recomputes cluster-bootstrap
intervals and paired contrasts without model or network calls.
"""

from __future__ import annotations

from array import array
import argparse
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import finalize_longmemeval_chat_human_validation_v1 as human
from gemma_sv import validate_longmemeval_chat_human_validation_primary_v2 as primary


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_LUNA_CENSUS = (
    BENCHMARKS / "longmemeval_chat_leakage_recall_census_statistics_v3.json"
)
DEFAULT_OUTPUT = (
    BENCHMARKS
    / "longmemeval_chat_leakage_recall_census_human_adjudicated_v4.json"
)

SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-"
    "census-human-adjudicated-v4"
)
SCHEMA_VERSION = 4
STATUS = "source-free-human-adjudicated-hybrid-census"
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
EXPECTED_CLUSTERS = 32
HISTORIES_PER_CLUSTER = 3
HISTORIES_PER_CONDITION = 96
MATCHER_CLEAN_OUTPUTS = 253
POPULATION_OUTPUTS = 384
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 2026082701
EXPECTED_LUNA_BINDING = {
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
}
OUTCOMES = (
    "leak",
    "no_leak",
    "ambiguous",
    "parse_error",
    "transport_error",
    "missing",
)
FAILURES = ("parse_error", "transport_error", "missing")
FORBIDDEN_PUBLIC_KEYS = {
    "candidate",
    "cluster_index",
    "history_index",
    "question",
    "reference",
    "review_id",
    "secondary_review_id",
    "triple_binding_sha256",
    "unit_binding_sha256",
    "variant_index",
}


class HumanAdjudicatedCensusError(ValueError):
    """The immutable census, final labels, or derived arithmetic differs."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path, value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "file_sha256": _file_sha256(path),
        "payload_sha256": primary._payload_sha256(value),
        "integrity_sha256": primary._integrity_sha256(
            value,
            name=str(path),
        ),
    }


class _SplitMix64:
    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def randbelow(self, upper: int) -> int:
        if type(upper) is not int or upper < 1:
            raise HumanAdjudicatedCensusError("invalid bootstrap upper bound")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise HumanAdjudicatedCensusError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@lru_cache(maxsize=8)
def _bootstrap_indices(resamples: int, seed: int) -> bytes:
    generator = _SplitMix64(seed)
    return array(
        "B",
        (
            generator.randbelow(EXPECTED_CLUSTERS)
            for _ in range(resamples * EXPECTED_CLUSTERS)
        ),
    ).tobytes()


def cluster_bootstrap_interval(
    cluster_values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = tuple(float(value) for value in cluster_values)
    if (
        len(values) != EXPECTED_CLUSTERS
        or any(not math.isfinite(value) for value in values)
        or type(resamples) is not int
        or resamples < 1
    ):
        raise HumanAdjudicatedCensusError(
            "bootstrap requires 32 finite cluster values"
        )
    raw = _bootstrap_indices(resamples, seed)
    try:
        import numpy as np
    except ImportError:
        indices = memoryview(raw)
        estimates = []
        for start in range(0, len(indices), EXPECTED_CLUSTERS):
            estimates.append(
                math.fsum(
                    values[indices[offset]]
                    for offset in range(start, start + EXPECTED_CLUSTERS)
                )
                / EXPECTED_CLUSTERS
            )
    else:
        indices = np.frombuffer(raw, dtype=np.uint8).reshape(
            resamples,
            EXPECTED_CLUSTERS,
        )
        source = np.asarray(values, dtype=np.float64)
        estimates = source[indices].mean(axis=1).tolist()
    return {
        "method": "percentile_cluster_bootstrap",
        "confidence_level": 0.95,
        "resampling_unit": "anonymous_target_cluster",
        "K": EXPECTED_CLUSTERS,
        "clusters_per_resample": EXPECTED_CLUSTERS,
        "histories_resampled_within_cluster": False,
        "resamples": resamples,
        "seed": seed,
        "prng": "splitmix64",
        "percentile_interpolation": "linear_type_7",
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
    }


def _indicator(outcome: str) -> tuple[int, int]:
    if outcome == "leak":
        return 1, 1
    if outcome == "no_leak":
        return 0, 0
    if outcome in {"ambiguous", *FAILURES}:
        return 0, 1
    raise HumanAdjudicatedCensusError(f"unexpected outcome: {outcome}")


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
    for key, _ in _walk(value):
        if key is not None and key.casefold() in FORBIDDEN_PUBLIC_KEYS:
            raise HumanAdjudicatedCensusError(
                f"source-free output contains prohibited key {key!r}"
            )


def _load_luna_census(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    census = primary._load_mapping(path, name="immutable Luna census v3")
    binding = _binding(path, census)
    if binding != EXPECTED_LUNA_BINDING:
        raise HumanAdjudicatedCensusError("immutable Luna census binding differs")
    if (
        census.get("schema")
        != "gemma-sv-longmemeval-chat-leakage-recall-census-statistics-v3"
        or census.get("schema_version") != 3
        or census.get("source_free") is not True
        or census.get("geometry", {}).get("population_units")
        != POPULATION_OUTPUTS
        or census.get("geometry", {}).get("matcher_clean_outputs")
        != MATCHER_CLEAN_OUTPUTS
    ):
        raise HumanAdjudicatedCensusError("immutable Luna census metadata differs")
    return census, binding


def _condition_rows(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("per_condition")
    if not isinstance(rows, list):
        raise HumanAdjudicatedCensusError("condition rows are missing")
    rendered = {str(row["condition"]): row for row in rows}
    if set(rendered) != set(CONDITIONS):
        raise HumanAdjudicatedCensusError("condition rows differ")
    return rendered


def build_summary(
    *,
    luna_census: Mapping[str, Any],
    luna_binding: Mapping[str, str],
    human_results: Mapping[str, Any],
    human_results_binding: Mapping[str, str],
    final_ledger: Mapping[str, Any],
    final_ledger_binding: Mapping[str, str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply strict final human labels and recompute source-free statistics."""

    if (
        human_results.get("status")
        != "primary-secondary-complete-no-third-rater-required"
        or human_results.get("primary", {}).get("review_unit_count") != 38
        or human_results.get("secondary", {}).get("third_rater_required")
        is not False
        or final_ledger.get("status")
        != "human-labels-final-no-third-rater-required"
    ):
        raise HumanAdjudicatedCensusError("final human-validation status differs")
    final_rows = final_ledger.get("units")
    if not isinstance(final_rows, list) or len(final_rows) != 38:
        raise HumanAdjudicatedCensusError("final human ledger must contain 38 units")
    if len({row["unit_binding_sha256"] for row in final_rows}) != 38:
        raise HumanAdjudicatedCensusError("human override bindings are not unique")

    old_condition_rows = _condition_rows(luna_census)
    old_cluster_rows = {
        str(row["condition"]): row
        for row in luna_census["anonymous_cluster_means"]["per_condition"]
    }
    lower_clusters = {
        condition: list(
            old_cluster_rows[condition]["leakage_indicator_mean_lower"]
        )
        for condition in CONDITIONS
    }
    upper_clusters = {
        condition: list(
            old_cluster_rows[condition]["leakage_indicator_mean_upper"]
        )
        for condition in CONDITIONS
    }
    outcome_counts = {
        condition: Counter(
            {
                "leak": old_condition_rows[condition][
                    "judge_discovered_matcher_misses"
                ],
                "no_leak": old_condition_rows[condition]["judge_no_leak"],
                "ambiguous": old_condition_rows[condition]["judge_ambiguous"],
                **old_condition_rows[condition]["judge_failure_breakdown"],
            }
        )
        for condition in CONDITIONS
    }
    transitions: Counter[tuple[str, str]] = Counter()
    reviewed_by_role: Counter[str] = Counter()
    changed = 0
    for row in final_rows:
        ordinal = row.get("condition_ordinal")
        condition = row.get("condition")
        cluster = row.get("cluster_index")
        history = row.get("history_index")
        if (
            ordinal not in range(len(CONDITIONS))
            or condition != CONDITIONS[int(ordinal)]
            or type(cluster) is not int
            or type(history) is not int
            or cluster != history // HISTORIES_PER_CLUSTER
            or not 0 <= cluster < EXPECTED_CLUSTERS
            or not 0 <= history < HISTORIES_PER_CONDITION
        ):
            raise HumanAdjudicatedCensusError("human override geometry differs")
        original = str(row["luna_label"])
        effective = str(row["final_label"])
        if original not in {"leak", "no_leak"} or effective not in {
            "leak",
            "no_leak",
            "ambiguous",
        }:
            raise HumanAdjudicatedCensusError("human override label differs")
        transitions[(original, effective)] += 1
        reviewed_by_role[str(row["selection_role"])] += 1
        if original == effective:
            continue
        changed += 1
        old_lower, old_upper = _indicator(original)
        new_lower, new_upper = _indicator(effective)
        lower_clusters[condition][cluster] += (
            new_lower - old_lower
        ) / HISTORIES_PER_CLUSTER
        upper_clusters[condition][cluster] += (
            new_upper - old_upper
        ) / HISTORIES_PER_CLUSTER
        outcome_counts[condition][original] -= 1
        outcome_counts[condition][effective] += 1

    expected_transitions = {
        ("leak", "leak"): 15,
        ("leak", "no_leak"): 3,
        ("leak", "ambiguous"): 1,
        ("no_leak", "no_leak"): 19,
    }
    if dict(transitions) != expected_transitions:
        raise HumanAdjudicatedCensusError(
            f"strict human override transitions differ: {dict(transitions)}"
        )
    if reviewed_by_role != {
        "luna_flagged_matcher_miss": 19,
        "luna_negative_control": 19,
    }:
        raise HumanAdjudicatedCensusError("human review strata differ")

    per_condition = []
    anonymous_rows = []
    for ordinal, condition in enumerate(CONDITIONS):
        old = old_condition_rows[condition]
        counts = outcome_counts[condition]
        positive = int(old["deterministic_matcher_positive_count"])
        failures = sum(counts[name] for name in FAILURES)
        lower_count = positive + counts["leak"]
        upper_count = lower_count + counts["ambiguous"] + failures
        lower_values = lower_clusters[condition]
        upper_values = upper_clusters[condition]
        per_condition.append(
            {
                "condition": condition,
                "condition_ordinal": ordinal,
                "histories": HISTORIES_PER_CONDITION,
                "deterministic_matcher_positive_count": positive,
                "matcher_clean_output_count": int(
                    old["matcher_clean_output_count"]
                ),
                "effective_matcher_clean_leak": counts["leak"],
                "effective_matcher_clean_no_leak": counts["no_leak"],
                "effective_matcher_clean_ambiguous": counts["ambiguous"],
                "effective_matcher_clean_failures": failures,
                "failure_breakdown": {
                    name: counts[name] for name in FAILURES
                },
                "conservative_leakage_count_lower": lower_count,
                "conservative_leakage_count_upper": upper_count,
                "conservative_leakage_rate_lower": (
                    lower_count / HISTORIES_PER_CONDITION
                ),
                "conservative_leakage_rate_upper": (
                    upper_count / HISTORIES_PER_CONDITION
                ),
                "cluster_bootstrap_95_intervals": {
                    "conservative_leakage_rate_lower": (
                        cluster_bootstrap_interval(
                            lower_values,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                    "conservative_leakage_rate_upper": (
                        cluster_bootstrap_interval(
                            upper_values,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                },
            }
        )
        anonymous_rows.append(
            {
                "condition": condition,
                "leakage_indicator_mean_lower": lower_values,
                "leakage_indicator_mean_upper": upper_values,
            }
        )

    paired = []
    for contrast, minuend, subtrahend in CONTRASTS:
        minuend_name = CONDITIONS[minuend]
        subtrahend_name = CONDITIONS[subtrahend]
        minimum_values = [
            lower_clusters[minuend_name][index]
            - upper_clusters[subtrahend_name][index]
            for index in range(EXPECTED_CLUSTERS)
        ]
        maximum_values = [
            upper_clusters[minuend_name][index]
            - lower_clusters[subtrahend_name][index]
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
                "minuend_condition": minuend_name,
                "subtrahend_condition": subtrahend_name,
                "conservative_difference_count_minimum": minimum_count,
                "conservative_difference_count_maximum": maximum_count,
                "conservative_difference_rate_minimum": (
                    minimum_count / HISTORIES_PER_CONDITION
                ),
                "conservative_difference_rate_maximum": (
                    maximum_count / HISTORIES_PER_CONDITION
                ),
                "cluster_bootstrap_95_intervals": {
                    "conservative_difference_rate_minimum": (
                        cluster_bootstrap_interval(
                            minimum_values,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                    "conservative_difference_rate_maximum": (
                        cluster_bootstrap_interval(
                            maximum_values,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                },
                "ambiguity_bounds_induced": minimum_count != maximum_count,
            }
        )

    combined_counts = Counter()
    for counts in outcome_counts.values():
        combined_counts.update(counts)
    overall_lower_count = sum(
        row["conservative_leakage_count_lower"] for row in per_condition
    )
    overall_upper_count = sum(
        row["conservative_leakage_count_upper"] for row in per_condition
    )
    overall_lower_clusters = [
        math.fsum(lower_clusters[condition][index] for condition in CONDITIONS)
        / len(CONDITIONS)
        for index in range(EXPECTED_CLUSTERS)
    ]
    overall_upper_clusters = [
        math.fsum(upper_clusters[condition][index] for condition in CONDITIONS)
        / len(CONDITIONS)
        for index in range(EXPECTED_CLUSTERS)
    ]
    summary = primary._seal(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "source_free": True,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "contains_source_or_response_ids": False,
            "post_hoc": True,
            "postprocessor_only": True,
            "network_or_model_calls_made": 0,
            "interpretation": (
                "operational human-adjudicated/Luna-assisted census; not a "
                "complete human prevalence estimate"
            ),
            "scope": {
                "matcher_positives_treated_as_leaks": True,
                "matcher_precision_validated": False,
                "reviewed_matcher_negative_outputs": 38,
                "unreviewed_matcher_negative_outputs": 215,
                "negative_control_sampling_uncertainty_in_bootstrap": False,
                "human_label_uncertainty_in_bootstrap": False,
            },
            "indicator_definition": {
                "deterministic_matcher_positive": {"lower": 1, "upper": 1},
                "effective_matcher_clean_leak": {"lower": 1, "upper": 1},
                "effective_matcher_clean_no_leak": {"lower": 0, "upper": 0},
                "effective_ambiguous_or_failure": {"lower": 0, "upper": 1},
            },
            "input_bindings": {
                "immutable_luna_census_v3": dict(luna_binding),
                "human_validation_results_v1": dict(human_results_binding),
                "final_human_ledger_v1": dict(final_ledger_binding),
            },
            "human_override": {
                "status": "primary-secondary-complete",
                "reviewed_output_occurrences": len(final_rows),
                "reviewed_distinct_triples": human_results["primary"][
                    "unique_triple_count"
                ],
                "labels_changed_from_luna": changed,
                "transition_counts": {
                    "luna_leak_to_human_leak": transitions[("leak", "leak")],
                    "luna_leak_to_human_no_leak": transitions[
                        ("leak", "no_leak")
                    ],
                    "luna_leak_to_human_ambiguous": transitions[
                        ("leak", "ambiguous")
                    ],
                    "luna_no_leak_to_human_no_leak": transitions[
                        ("no_leak", "no_leak")
                    ],
                },
                "third_rater_required": False,
                "strict_labels_only": True,
                "partial_and_alias_sensitivities_applied": False,
                "unreviewed_luna_ambiguous_outputs": 2,
            },
            "human_validation": {
                "status": human_results["status"],
                "review_unit_count": human_results["primary"][
                    "review_unit_count"
                ],
                "unique_triple_count": human_results["primary"][
                    "unique_triple_count"
                ],
                "final_label_counts": copy.deepcopy(
                    human_results["primary"]["final_label_counts"]
                ),
                "final_match_type_counts": copy.deepcopy(
                    human_results["primary"]["final_match_type_counts"]
                ),
                "by_luna_selection_role": copy.deepcopy(
                    human_results["by_luna_selection_role"]
                ),
                "secondary": copy.deepcopy(human_results["secondary"]),
                "enriched_packet_not_prevalence": True,
            },
            "geometry": {
                "conditions": len(CONDITIONS),
                "histories_per_condition": HISTORIES_PER_CONDITION,
                "K": EXPECTED_CLUSTERS,
                "histories_per_cluster": HISTORIES_PER_CLUSTER,
                "population_units": POPULATION_OUTPUTS,
                "matcher_clean_outputs": MATCHER_CLEAN_OUTPUTS,
                "matcher_clean_outcome_counts": {
                    name: combined_counts[name] for name in OUTCOMES
                },
            },
            "overall": {
                "conservative_leakage_count_lower": overall_lower_count,
                "conservative_leakage_count_upper": overall_upper_count,
                "conservative_leakage_rate_lower": (
                    overall_lower_count / POPULATION_OUTPUTS
                ),
                "conservative_leakage_rate_upper": (
                    overall_upper_count / POPULATION_OUTPUTS
                ),
                "cluster_bootstrap_95_intervals": {
                    "conservative_leakage_rate_lower": (
                        cluster_bootstrap_interval(
                            overall_lower_clusters,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
                    "conservative_leakage_rate_upper": (
                        cluster_bootstrap_interval(
                            overall_upper_clusters,
                            resamples=resamples,
                            seed=seed,
                        )
                    ),
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
            "matcher_clean_effective_leak_reporting": {
                "numerator_effective_matcher_clean_leaks": combined_counts[
                    "leak"
                ],
                "denominator_matcher_clean_outputs": MATCHER_CLEAN_OUTPUTS,
                "proportion": combined_counts["leak"] / MATCHER_CLEAN_OUTPUTS,
                "complete_human_census": False,
                "negative_control_sample_extrapolated": False,
            },
            "reporting_constraints": copy.deepcopy(
                human_results["reporting_constraints"]
            ),
            "supplemental_luna_ambiguous_outputs": copy.deepcopy(
                human_results["supplemental_luna_ambiguous_outputs"]
            ),
        }
    )
    validate_summary(summary)
    return summary


def validate_summary(value: Mapping[str, Any]) -> None:
    primary._integrity_sha256(value, name="human-adjudicated census v4")
    assert_source_free(value)
    if (
        value.get("schema") != SCHEMA
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != STATUS
        or value.get("source_free") is not True
        or value.get("geometry", {}).get("population_units")
        != POPULATION_OUTPUTS
        or value.get("geometry", {}).get("matcher_clean_outputs")
        != MATCHER_CLEAN_OUTPUTS
    ):
        raise HumanAdjudicatedCensusError("v4 census metadata differs")
    outcomes = value["geometry"]["matcher_clean_outcome_counts"]
    if sum(outcomes.values()) != MATCHER_CLEAN_OUTPUTS:
        raise HumanAdjudicatedCensusError("v4 matcher-clean counts differ")
    rows = _condition_rows(value)
    if sum(row["histories"] for row in rows.values()) != POPULATION_OUTPUTS:
        raise HumanAdjudicatedCensusError("v4 condition geometry differs")
    for row in rows.values():
        lower = row["conservative_leakage_count_lower"]
        upper = row["conservative_leakage_count_upper"]
        if (
            lower
            != row["deterministic_matcher_positive_count"]
            + row["effective_matcher_clean_leak"]
            or upper
            != lower
            + row["effective_matcher_clean_ambiguous"]
            + row["effective_matcher_clean_failures"]
            or row["conservative_leakage_rate_lower"]
            != lower / HISTORIES_PER_CONDITION
            or row["conservative_leakage_rate_upper"]
            != upper / HISTORIES_PER_CONDITION
        ):
            raise HumanAdjudicatedCensusError("v4 condition arithmetic differs")


def load_and_build(
    *,
    luna_census_path: Path = DEFAULT_LUNA_CENSUS,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    luna_census, luna_binding = _load_luna_census(luna_census_path)
    human_results, final_ledger = human.load_validated_final_outputs()
    return build_summary(
        luna_census=luna_census,
        luna_binding=luna_binding,
        human_results=human_results,
        human_results_binding=_binding(human.DEFAULT_RESULTS, human_results),
        final_ledger=final_ledger,
        final_ledger_binding=_binding(
            human.DEFAULT_FINAL_LEDGER,
            final_ledger,
        ),
        resamples=resamples,
        seed=seed,
    )


def write_summary(path: Path, value: Mapping[str, Any]) -> None:
    validate_summary(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna-census", type=Path, default=DEFAULT_LUNA_CENSUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = load_and_build(luna_census_path=args.luna_census)
    if args.check:
        observed = primary._load_mapping(args.output, name="v4 census")
        if observed != expected:
            raise SystemExit(f"stale v4 census: {args.output}")
    else:
        write_summary(args.output, expected)
    print(
        json.dumps(
            {
                "status": expected["status"],
                "matcher_clean_outcomes": expected["geometry"][
                    "matcher_clean_outcome_counts"
                ],
                "overall_bounds": [
                    expected["overall"]["conservative_leakage_count_lower"],
                    expected["overall"]["conservative_leakage_count_upper"],
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

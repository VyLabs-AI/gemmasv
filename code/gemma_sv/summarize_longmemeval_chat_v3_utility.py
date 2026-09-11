"""Publish an anonymous LongMemEval v3 continuous-utility summary.

Private shards may contain gold-token arrays and forced-choice score vectors.
This publisher emits only anonymous indices, scalar values, bounds, aggregate
scalar distributions, and failure hashes.
"""

from __future__ import annotations

import argparse
from array import array
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_v3_utility as evaluator
from gemma_sv import longmemeval_chat_v3_utility_execution_protocol as protocol


SCHEMA = "gemma-sv-longmemeval-chat-v3-utility-public-summary-v1"
SCHEMA_VERSION = 1
EXPECTED_CLUSTERS = protocol.EXPECTED_CLUSTERS
EXPECTED_HISTORIES = protocol.EXPECTED_HISTORIES
HISTORIES_PER_CLUSTER = protocol.HISTORIES_PER_CLUSTER
DEFAULT_OUTPUT = (
    protocol.BENCHMARKS / "longmemeval_chat_v3_utility_summary_v1.json"
)
EXPLICIT_PUBLICATION_ACKNOWLEDGEMENT = (
    "I_ACKNOWLEDGE_ANONYMOUS_LONGMEMEVAL_V3_UTILITY_PUBLICATION"
)

_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "choice_answer_sha256",
        "choice_source_sha256",
        "choice_answers",
        "cluster_id",
        "content",
        "full_vocabulary_distribution_sha256_by_target_token",
        "full_vocabulary_log_probabilities_by_target_token",
        "generated_token_ids",
        "gold_token_log_probabilities_nats",
        "logits",
        "messages",
        "per_target_token_full_vocabulary_kl_nats",
        "prompt",
        "question",
        "record_id",
        "response_text",
        "source_id",
        "source_text",
        "text",
        "token_ids",
        "turns",
    }
)


class UtilitySummaryError(ValueError):
    """Private evidence or the anonymous public projection differs."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": _payload_sha256(result),
    }
    return result


def _validate_seal(value: Mapping[str, Any], *, name: str) -> None:
    body = copy.deepcopy(dict(value))
    integrity = body.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(body)
    ):
        raise UtilitySummaryError(f"{name} integrity differs")


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
            raise UtilitySummaryError("bootstrap upper bound must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise UtilitySummaryError("percentile requires values")
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
    if resamples < 1:
        raise UtilitySummaryError("bootstrap resamples must be positive")
    generator = _SplitMix64(seed)
    values = array(
        "B",
        (
            generator.randbelow(EXPECTED_CLUSTERS)
            for _ in range(resamples * EXPECTED_CLUSTERS)
        ),
    )
    return values.tobytes()


def cluster_bootstrap_interval(
    cluster_values: Sequence[float],
    *,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = protocol.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap 32 complete clusters with SplitMix64/type-7 percentiles."""

    values = tuple(float(value) for value in cluster_values)
    if (
        len(values) != EXPECTED_CLUSTERS
        or any(not math.isfinite(value) for value in values)
    ):
        raise UtilitySummaryError("bootstrap requires 32 finite cluster values")
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
        "confidence_level": protocol.CONFIDENCE_LEVEL,
        "resampling_unit": "target_cluster",
        "K": EXPECTED_CLUSTERS,
        "clusters_per_resample": EXPECTED_CLUSTERS,
        "resamples": resamples,
        "seed": seed,
        "prng": "splitmix64",
        "percentile_interpolation": "linear_type_7",
        "histories_resampled_within_cluster": False,
        "lower": percentile(estimates, 0.025),
        "upper": percentile(estimates, 0.975),
    }


def summarize_history_values(
    history_values: Sequence[float],
    *,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = protocol.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = tuple(float(value) for value in history_values)
    if (
        len(values) != EXPECTED_HISTORIES
        or any(not math.isfinite(value) for value in values)
    ):
        raise UtilitySummaryError("ITT metric must contain all 96 finite values")
    clusters = tuple(
        statistics.fmean(
            values[
                index
                * HISTORIES_PER_CLUSTER : (index + 1)
                * HISTORIES_PER_CLUSTER
            ]
        )
        for index in range(EXPECTED_CLUSTERS)
    )
    q1 = percentile(clusters, 0.25)
    q3 = percentile(clusters, 0.75)
    return {
        "history_n": EXPECTED_HISTORIES,
        "independent_cluster_n": EXPECTED_CLUSTERS,
        "cluster_history_value": "arithmetic_mean_of_three_nested_histories",
        "mean": statistics.fmean(clusters),
        "median": statistics.median(clusters),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "minimum": min(clusters),
        "maximum": max(clusters),
        "range": max(clusters) - min(clusters),
        "cluster_empirical_distribution": list(clusters),
        "cluster_bootstrap_95_interval": cluster_bootstrap_interval(
            clusters,
            resamples=resamples,
            seed=seed,
        ),
    }


def _cell_probes(cell: Any) -> Mapping[str, Any] | None:
    if not isinstance(cell, Mapping):
        return None
    if cell.get("status") == "failed":
        return None
    probes = cell.get("probes")
    return probes if isinstance(probes, Mapping) else None


def _teacher(
    shard: Mapping[str, Any],
    condition: str,
    probe: str,
) -> Mapping[str, Any] | None:
    conditions = shard.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    probes = _cell_probes(conditions.get(condition))
    if probes is None:
        return None
    value = probes.get(probe)
    if not isinstance(value, Mapping):
        return None
    score = value.get("teacher_forced")
    return score if isinstance(score, Mapping) else None


def _comparison(
    shard: Mapping[str, Any],
    condition: str,
    probe: str,
) -> Mapping[str, Any] | None:
    conditions = shard.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    probes = _cell_probes(conditions.get(condition))
    if probes is None:
        return None
    value = probes.get(probe)
    if not isinstance(value, Mapping):
        return None
    comparison = value.get("rebuild_comparison")
    return comparison if isinstance(comparison, Mapping) else None


def _forced_rank(
    shard: Mapping[str, Any],
    condition: str,
    probe: str,
) -> int | None:
    conditions = shard.get("conditions")
    if not isinstance(conditions, Mapping):
        return None
    probes = _cell_probes(conditions.get(condition))
    if probes is None:
        return None
    value = probes.get(probe)
    choice = value.get("forced_choice") if isinstance(value, Mapping) else None
    rank = choice.get("gold_rank") if isinstance(choice, Mapping) else None
    return int(rank) if type(rank) is int else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    rendered = float(value)
    return rendered if math.isfinite(rendered) else None


def _rank(value: Any) -> int | None:
    return int(value) if type(value) is int and value >= 1 else None


def _failure_hash(*shards: Mapping[str, Any]) -> str | None:
    hashes: list[str] = []
    for shard in shards:
        failure = shard.get("failure")
        if isinstance(failure, Mapping) and protocol._is_sha256(
            failure.get("error_message_sha256")
        ):
            hashes.append(str(failure["error_message_sha256"]))
        for collection_name in ("conditions", "wikitext"):
            conditions = shard.get(collection_name)
            if not isinstance(conditions, Mapping):
                continue
            for cell in conditions.values():
                nested = cell.get("failure") if isinstance(cell, Mapping) else None
                if isinstance(nested, Mapping) and protocol._is_sha256(
                    nested.get("error_message_sha256")
                ):
                    hashes.append(str(nested["error_message_sha256"]))
    return _payload_sha256(hashes) if hashes else None


def _control_public_cell(
    control: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    target = _teacher(control, condition, "target_current")
    retained = _teacher(control, condition, "retained")
    return {
        "observed": isinstance(target, Mapping)
        and isinstance(retained, Mapping),
        "target_total_sequence_log_probability_nats": (
            _number(target.get("total_log_probability_nats"))
            if isinstance(target, Mapping)
            else None
        ),
        "target_mean_sequence_log_probability_nats": (
            _number(target.get("mean_log_probability_nats"))
            if isinstance(target, Mapping)
            else None
        ),
        "target_first_token_rank": (
            _rank(target.get("first_target_token_rank"))
            if isinstance(target, Mapping)
            else None
        ),
        "target_forced_choice_gold_rank": _forced_rank(
            control,
            condition,
            "target_current",
        ),
        "retained_total_sequence_log_probability_nats": (
            _number(retained.get("total_log_probability_nats"))
            if isinstance(retained, Mapping)
            else None
        ),
        "retained_mean_sequence_log_probability_nats": (
            _number(retained.get("mean_log_probability_nats"))
            if isinstance(retained, Mapping)
            else None
        ),
        "retained_first_token_rank": (
            _rank(retained.get("first_target_token_rank"))
            if isinstance(retained, Mapping)
            else None
        ),
        "retained_forced_choice_gold_rank": _forced_rank(
            control,
            condition,
            "retained",
        ),
    }


def _method_public_cell(
    control: Mapping[str, Any],
    method: Mapping[str, Any],
    admission: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    target_present = _teacher(control, protocol.PRESENT, "target_current")
    target_rebuild = _teacher(
        control,
        protocol.FRESH_REBUILD,
        "target_current",
    )
    target_method = _teacher(method, condition, "target_current")
    target_comparison = _comparison(method, condition, "target_current")
    retained_rebuild = _teacher(
        control,
        protocol.FRESH_REBUILD,
        "retained",
    )
    retained_method = _teacher(method, condition, "retained")
    retained_comparison = _comparison(method, condition, "retained")
    observed = all(
        isinstance(value, Mapping)
        for value in (
            target_present,
            target_rebuild,
            target_method,
            target_comparison,
            retained_rebuild,
            retained_method,
            retained_comparison,
        )
    )
    target_present_mean = (
        _number(target_present.get("mean_log_probability_nats"))
        if isinstance(target_present, Mapping)
        else None
    )
    target_rebuild_mean = (
        _number(target_rebuild.get("mean_log_probability_nats"))
        if isinstance(target_rebuild, Mapping)
        else None
    )
    target_method_mean = (
        _number(target_method.get("mean_log_probability_nats"))
        if isinstance(target_method, Mapping)
        else None
    )
    denominator = (
        target_present_mean - target_rebuild_mean
        if target_present_mean is not None and target_rebuild_mean is not None
        else None
    )
    raw_progress = (
        (target_present_mean - target_method_mean) / denominator
        if denominator is not None
        and denominator > 0.0
        and target_method_mean is not None
        else None
    )
    target_kl_mean = (
        _number(target_comparison.get("mean_full_vocabulary_kl_nats"))
        if isinstance(target_comparison, Mapping)
        else None
    )
    retained_drift = (
        _number(
            retained_comparison.get(
                "method_minus_rebuild_mean_log_probability_nats"
            )
        )
        if isinstance(retained_comparison, Mapping)
        else None
    )
    retained_kl_first = (
        _number(
            retained_comparison.get(
                "first_target_token_full_vocabulary_kl_nats"
            )
        )
        if isinstance(retained_comparison, Mapping)
        else None
    )
    jointly_admitted = admission.get("joint_admitted") is True
    primary_success = observed and jointly_admitted
    return {
        "observed": observed,
        "target_total_sequence_log_probability_nats": (
            _number(target_method.get("total_log_probability_nats"))
            if isinstance(target_method, Mapping)
            else None
        ),
        "target_mean_sequence_log_probability_nats": target_method_mean,
        "target_first_token_rank": (
            _rank(target_method.get("first_target_token_rank"))
            if isinstance(target_method, Mapping)
            else None
        ),
        "target_mean_full_vocabulary_kl_nats": target_kl_mean,
        "target_maximum_full_vocabulary_kl_nats": (
            _number(
                target_comparison.get("maximum_full_vocabulary_kl_nats")
            )
            if isinstance(target_comparison, Mapping)
            else None
        ),
        "target_raw_deletion_progress": raw_progress,
        "target_itt_deletion_progress": (
            min(1.0, max(0.0, raw_progress))
            if primary_success and raw_progress is not None
            else 0.0
        ),
        "target_itt_rebuild_similarity": (
            math.exp(-target_kl_mean)
            if primary_success and target_kl_mean is not None
            else 0.0
        ),
        "target_forced_choice_gold_rank": _forced_rank(
            method,
            condition,
            "target_current",
        ),
        "retained_mean_sequence_log_probability_nats": (
            _number(retained_method.get("mean_log_probability_nats"))
            if isinstance(retained_method, Mapping)
            else None
        ),
        "retained_total_sequence_log_probability_nats": (
            _number(retained_method.get("total_log_probability_nats"))
            if isinstance(retained_method, Mapping)
            else None
        ),
        "retained_first_token_rank": (
            _rank(retained_method.get("first_target_token_rank"))
            if isinstance(retained_method, Mapping)
            else None
        ),
        "retained_log_probability_drift_nats": retained_drift,
        "retained_first_token_full_vocabulary_kl_nats": retained_kl_first,
        "retained_mean_full_vocabulary_kl_nats": (
            _number(
                retained_comparison.get("mean_full_vocabulary_kl_nats")
            )
            if isinstance(retained_comparison, Mapping)
            else None
        ),
        "retained_maximum_full_vocabulary_kl_nats": (
            _number(
                retained_comparison.get("maximum_full_vocabulary_kl_nats")
            )
            if isinstance(retained_comparison, Mapping)
            else None
        ),
        "retained_itt_log_probability_preservation": (
            math.exp(-abs(retained_drift))
            if primary_success and retained_drift is not None
            else 0.0
        ),
        "retained_itt_distribution_preservation": (
            math.exp(-retained_kl_first)
            if primary_success and retained_kl_first is not None
            else 0.0
        ),
        "retained_forced_choice_gold_rank": _forced_rank(
            method,
            condition,
            "retained",
        ),
        "negative_effect": raw_progress is not None and raw_progress < 0.0,
        "collateral_damage": retained_drift is not None and retained_drift < 0.0,
    }


def _wikitext_public(
    method_shard: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    rows = method_shard.get("wikitext")
    cell = rows.get(condition) if isinstance(rows, Mapping) else None
    if isinstance(cell, Mapping) and cell.get("status") == "failed":
        cell = None
    comparison = (
        cell.get("rebuild_comparison") if isinstance(cell, Mapping) else None
    )
    delta = (
        _number(comparison.get("method_minus_rebuild_mean_nll_nats"))
        if isinstance(comparison, Mapping)
        else None
    )
    cost = (
        _number(comparison.get("relative_perplexity_cost_percent"))
        if isinstance(comparison, Mapping)
        else None
    )
    return {
        "observed": delta is not None and cost is not None,
        "method_minus_rebuild_mean_nll_nats": delta,
        "relative_perplexity_cost_percent": cost,
        "negative_cost": cost is not None and cost < 0.0,
        "collateral_cost": cost is not None and cost > 0.0,
    }


def _source_free_replay_strata(
    authorization: Mapping[str, Any],
) -> list[str]:
    disclosure = authorization.get("replay_disclosure") or {}
    values = disclosure.get("suffix_strata_by_history")
    if (
        not isinstance(values, list)
        or len(values) != EXPECTED_HISTORIES
        or any(value not in {"clean", "contaminated"} for value in values)
    ):
        raise UtilitySummaryError(
            "authorization lacks frozen suffix strata for all histories"
        )
    return list(values)


def build_public_summary(
    control_shards: Sequence[Mapping[str, Any]],
    method_shards: Sequence[Mapping[str, Any]],
    admission_rows: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
    *,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
    seed: int = protocol.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Project complete private evidence into a strict anonymous result."""

    protocol.validate_execution_authorization(authorization)
    if not (
        len(control_shards)
        == len(method_shards)
        == len(admission_rows)
        == EXPECTED_HISTORIES
    ):
        raise UtilitySummaryError("summary requires all 96 histories")
    strata = _source_free_replay_strata(authorization)
    histories: list[dict[str, Any]] = []
    for index, (control, method, admission) in enumerate(
        zip(control_shards, method_shards, admission_rows)
    ):
        evaluator.validate_control_shard(
            control,
            authorization=authorization,
            index=index,
        )
        evaluator.validate_method_shard(
            method,
            authorization=authorization,
            index=index,
        )
        if (
            admission.get("history_index") != index
            or admission.get("cluster_index")
            != index // HISTORIES_PER_CLUSTER
            or admission.get("variant_index")
            != index % HISTORIES_PER_CLUSTER
        ):
            raise UtilitySummaryError("admission anonymous order differs")
        history = {
            "history_index": index,
            "cluster_index": index // HISTORIES_PER_CLUSTER,
            "variant_index": index % HISTORIES_PER_CLUSTER,
            "suffix_stratum": strata[index],
            "target_admitted": admission.get("target_admitted") is True,
            "retained_available": admission.get("retained_available") is True,
            "joint_admitted": admission.get("joint_admitted") is True,
            "present": _control_public_cell(
                control,
                condition=protocol.PRESENT,
            ),
            "fresh_rebuild": _control_public_cell(
                control,
                condition=protocol.FRESH_REBUILD,
            ),
            "policy": _method_public_cell(
                control,
                method,
                admission,
                condition=protocol.POLICY,
            ),
            "replay": _method_public_cell(
                control,
                method,
                admission,
                condition=protocol.REPLAY,
            ),
            "wikitext_policy": _wikitext_public(
                method,
                condition=protocol.POLICY,
            ),
            "wikitext_replay": _wikitext_public(
                method,
                condition=protocol.REPLAY,
            ),
            "failure_sha256": _failure_hash(control, method),
        }
        histories.append(history)

    primary: dict[str, Any] = {}
    metric_keys = (
        "target_itt_deletion_progress",
        "target_itt_rebuild_similarity",
        "retained_itt_log_probability_preservation",
        "retained_itt_distribution_preservation",
    )
    for label in ("policy", "replay"):
        primary[label] = {
            key: summarize_history_values(
                [float(row[label][key]) for row in histories],
                resamples=resamples,
                seed=seed,
            )
            for key in metric_keys
        }
        primary[label]["counts"] = {
            "observed_histories": sum(
                row[label]["observed"] for row in histories
            ),
            "operational_or_admission_failure_histories": sum(
                float(row[label]["target_itt_deletion_progress"]) == 0.0
                and not (
                    row[label]["observed"] and row["joint_admitted"]
                )
                for row in histories
            ),
            "negative_effect_histories": sum(
                row[label]["negative_effect"] for row in histories
            ),
            "collateral_damage_histories": sum(
                row[label]["collateral_damage"] for row in histories
            ),
        }

    wikitext: dict[str, Any] = {}
    for label in ("policy", "replay"):
        key = f"wikitext_{label}"
        delta = [
            (
                float(row[key]["method_minus_rebuild_mean_nll_nats"])
                if row[key]["observed"]
                else 0.0
            )
            for row in histories
        ]
        cost = [
            (
                float(row[key]["relative_perplexity_cost_percent"])
                if row[key]["observed"]
                else 0.0
            )
            for row in histories
        ]
        wikitext[label] = {
            "method_minus_rebuild_mean_nll_nats": summarize_history_values(
                delta,
                resamples=resamples,
                seed=seed,
            ),
            "relative_perplexity_cost_percent": summarize_history_values(
                cost,
                resamples=resamples,
                seed=seed,
            ),
            "observed_histories": sum(row[key]["observed"] for row in histories),
            "negative_cost_histories": sum(
                row[key]["negative_cost"] for row in histories
            ),
            "collateral_cost_histories": sum(
                row[key]["collateral_cost"] for row in histories
            ),
        }

    strata_summary: dict[str, Any] = {}
    for stratum in ("clean", "contaminated"):
        selected = [row for row in histories if row["suffix_stratum"] == stratum]
        strata_summary[stratum] = {
            "history_count": len(selected),
            "cluster_count_any_history": len(
                {row["cluster_index"] for row in selected}
            ),
            "replay_mean_target_itt_deletion_progress": (
                statistics.fmean(
                    row["replay"]["target_itt_deletion_progress"]
                    for row in selected
                )
                if selected
                else None
            ),
            "replay_mean_retained_itt_log_probability_preservation": (
                statistics.fmean(
                    row["replay"][
                        "retained_itt_log_probability_preservation"
                    ]
                    for row in selected
                )
                if selected
                else None
            ),
        }

    summary = _seal(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "contains_source_text": False,
            "contains_source_identifiers": False,
            "contains_model_generated_text": False,
            "contains_token_arrays": False,
            "contains_full_vocabulary_vectors": False,
            "contains_per_token_arrays": False,
            "authorization_integrity_sha256": authorization["integrity"][
                "sha256"
            ],
            "geometry": {
                "K": EXPECTED_CLUSTERS,
                "n": EXPECTED_HISTORIES,
                "histories_per_cluster": HISTORIES_PER_CLUSTER,
                "history_instances_are_nested": True,
                "independent_analysis_n": EXPECTED_CLUSTERS,
            },
            "raw_values_scope": {
                "all_observed_condition_sequence_log_probabilities": True,
                "all_observed_first_token_ranks": True,
                "full_vocabulary_vectors_published": False,
                "per_token_arrays_published": False,
                "private_shards_retain_gold_token_values_and_per_token_kl": True,
            },
            "flow": {
                "control_terminal_histories": EXPECTED_HISTORIES,
                "method_terminal_histories": EXPECTED_HISTORIES,
                "target_admitted_histories": sum(
                    row["target_admitted"] for row in histories
                ),
                "retained_available_histories": sum(
                    row["retained_available"] for row in histories
                ),
                "joint_admitted_histories": sum(
                    row["joint_admitted"] for row in histories
                ),
                "histories_with_failure_hash": sum(
                    row["failure_sha256"] is not None for row in histories
                ),
                "all_histories_retained_in_itt": True,
            },
            "primary_intent_to_treat": primary,
            "post_delete_wikitext": {
                "label": "external-corpus held-out utility",
                "overlap_audit_disclosure": copy.deepcopy(
                    authorization["post_delete_wikitext_arm"][
                        "overlap_audit"
                    ]
                ),
                "conditions": wikitext,
            },
            "frozen_suffix_scan_strata": strata_summary,
            "histories": histories,
        }
    )
    validate_public_summary(summary)
    return summary


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield None, child
            yield from _walk(child)


def validate_public_summary(summary: Mapping[str, Any]) -> None:
    _validate_seal(summary, name="public summary")
    if (
        summary.get("schema") != SCHEMA
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "complete"
        or summary.get("contains_source_text") is not False
        or summary.get("contains_source_identifiers") is not False
        or summary.get("contains_token_arrays") is not False
        or summary.get("contains_full_vocabulary_vectors") is not False
        or summary.get("contains_per_token_arrays") is not False
    ):
        raise UtilitySummaryError("public summary declarations differ")
    histories = summary.get("histories")
    if not isinstance(histories, list) or len(histories) != EXPECTED_HISTORIES:
        raise UtilitySummaryError("public summary requires 96 anonymous rows")
    for index, row in enumerate(histories):
        if (
            not isinstance(row, Mapping)
            or row.get("history_index") != index
            or row.get("cluster_index") != index // HISTORIES_PER_CLUSTER
            or row.get("variant_index") != index % HISTORIES_PER_CLUSTER
        ):
            raise UtilitySummaryError("public anonymous history order differs")
    for key, value in _walk(summary):
        if key is not None and key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
            raise UtilitySummaryError(f"public summary contains forbidden key {key}")
        if isinstance(value, float) and not math.isfinite(value):
            raise UtilitySummaryError("public summary contains non-finite value")


def write_public_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    validate_public_summary(summary)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def build_from_shards(
    *,
    authorization: Mapping[str, Any],
    output_root: str | Path,
    resamples: int = protocol.BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    store = evaluator.UtilityShardStore(
        output_root,
        authorization,
        resume=True,
    )
    controls = store.controls()
    methods = store.methods()
    admission = store.admission()
    if len(controls) != EXPECTED_HISTORIES or len(methods) != EXPECTED_HISTORIES:
        raise UtilitySummaryError("publisher requires every terminal shard")
    return build_public_summary(
        [controls[index] for index in range(EXPECTED_HISTORIES)],
        [methods[index] for index in range(EXPECTED_HISTORIES)],
        admission["rows"],
        authorization,
        resamples=resamples,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--authorization",
        type=Path,
        default=protocol.DEFAULT_AUTHORIZATION,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=evaluator.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--acknowledgement", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.publish:
        parser.error("publisher requires --publish")
    if args.acknowledgement != EXPLICIT_PUBLICATION_ACKNOWLEDGEMENT:
        parser.error("exact anonymous publication acknowledgement required")
    try:
        authorization = protocol.load_execution_authorization(
            args.authorization
        )
        summary = build_from_shards(
            authorization=authorization,
            output_root=args.output_root,
        )
        write_public_summary(args.out, summary)
    except (
        FileExistsError,
        OSError,
        UtilitySummaryError,
        evaluator.UtilityEvaluationError,
        protocol.UtilityProtocolError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPLICIT_PUBLICATION_ACKNOWLEDGEMENT",
    "UtilitySummaryError",
    "build_public_summary",
    "cluster_bootstrap_interval",
    "percentile",
    "summarize_history_values",
    "validate_public_summary",
    "write_public_summary",
]

"""Freeze the clustered, source-only LongMemEval chat cohort v3.

The 32 target clusters come from the pre-outcome strict-v2 eligibility census.
Each cluster has three independently packed histories, one fresh retained
control shared within the cluster, and globally unique fresh tail sources.
Selection and validation use source data and geometry only; model outputs are
neither accepted nor inspected.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from gemma_sv import longmemeval_chat_benchmark as chat_v1
from gemma_sv import longmemeval_chat_benchmark_v2 as chat_v2
from gemma_sv import longmemeval_deletion_benchmark as base


SCHEMA = "gemma-sv-longmemeval-chat-clustered-cohort-v3"
SCHEMA_VERSION = 3
POLICY_LOCK_SCHEMA = "gemma-sv-longmemeval-chat-cluster-analysis-lock-v3"
CENSUS_SCHEMA = "gemma-sv-longmemeval-chat-cohort-census-v3"
BENCHMARK_LABEL = "LongMemEval source-only clustered chat cohort v3"
PARTITION = "confirmation"

TARGET_CLUSTERS = 32
HISTORIES_PER_CLUSTER = 3
HISTORY_INSTANCES = TARGET_CLUSTERS * HISTORIES_PER_CLUSTER
SELECTION_SEED = chat_v2.SELECTION_SEED
MAXIMUM_VARIANT_RETRIES = 256

EXPOSURE_CLASS_COUNTS = {
    "direct-target": 16,
    "context-only": 2,
    "source-unseen": 14,
}
PRIMARY_ANALYSIS_N = TARGET_CLUSTERS

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_COHORT_PATH = BENCHMARKS / "longmemeval_chat_cohort_v3.json"
DEFAULT_POLICY_LOCK_PATH = (
    BENCHMARKS / "longmemeval_chat_cluster_analysis_lock_v3.json"
)
DEFAULT_CENSUS_PATH = BENCHMARKS / "longmemeval_chat_cohort_census_v3.json"
DEFAULT_V2_MANIFEST_PATH = (
    BENCHMARKS / "longmemeval_chat_confirmation_v2.json"
)
DEFAULT_V2_POLICY_LOCK_PATH = (
    BENCHMARKS / "longmemeval_chat_policy_lock_v2.json"
)
DEFAULT_V2_CENSUS_PATH = BENCHMARKS / "longmemeval_chat_census_v2.json"
DEFAULT_GEOMETRY_MANIFEST_PATH = (
    BENCHMARKS / "longmemeval_chat_geometry_corrected_v1.json"
)

ManifestInput = Mapping[str, Any] | str | Path
ManifestError = base.ManifestError


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError as error:
        raise ManifestError(
            f"artifact path must be repository-relative: {path}"
        ) from error


def _load_json(source: ManifestInput) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source)), None
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError(f"expected a JSON object: {path}")
    return payload, path


def _integrity_sha256(payload: Mapping[str, Any]) -> str:
    integrity = payload.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ManifestError("input artifact has no integrity object")
    value = str(integrity.get("sha256") or "")
    if len(value) != 64:
        raise ManifestError("input artifact has no SHA-256 integrity binding")
    return value


def _iter_source_ids(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "source_id" and isinstance(nested, str) and nested:
                yield nested
            yield from _iter_source_ids(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_source_ids(nested)


def _source_hashes(source_ids: Iterable[str]) -> list[str]:
    return sorted(base.text_sha256(value) for value in set(source_ids))


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _artifact_descriptor(
    payload: Mapping[str, Any],
    path: Path | None,
    *,
    role: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "repository_relative_path": (
            None if path is None else _repository_relative(path)
        ),
        "file_sha256": None if path is None else _file_sha256(path),
        "integrity_sha256": _integrity_sha256(payload),
    }


def _history_seed(
    target_source_id: str,
    *,
    variant_index: int,
    retry_ordinal: int,
) -> int:
    digest = hashlib.sha256(
        (
            "longmemeval-history-v1"
            f"\0{SELECTION_SEED}"
            f"\0{base.text_sha256(target_source_id)}"
            f"\0{variant_index}"
            f"\0{retry_ordinal}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _cluster_id(target_source_id: str) -> str:
    return base.stable_identifier(
        "longmemeval-chat-cluster-v3",
        base.DATASET_REVISION,
        base.text_sha256(target_source_id),
    )


def _history_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    source_selection = record["source_selection"]
    return {
        "cluster_id": record["cluster_id"],
        "variant_index": record["variant_index"],
        "history_seed": record["history_seed"],
        "history_retry_ordinal": record["history_retry_ordinal"],
        "base_tail_minimum_tokens": source_selection[
            "base_tail_minimum_tokens"
        ],
        "baseline": {
            "record_id": source_selection["record_id"],
            "manifest_integrity_sha256": source_selection[
                "manifest_integrity_sha256"
            ],
        },
        "target": copy.deepcopy(record["target"]),
        "retained_probe": copy.deepcopy(record["retained_probe"]),
        "tail_session_references": copy.deepcopy(
            record["tail_session_references"]
        ),
        "session_layout": copy.deepcopy(record["session_layout"]),
        "context": copy.deepcopy(record["context"]),
        "probes": copy.deepcopy(record["probes"]),
    }


def _history_id(record: Mapping[str, Any]) -> str:
    return base.stable_identifier(
        "longmemeval-chat-history-v3",
        base.DATASET_REVISION,
        record["cluster_id"],
        record["variant_index"],
        _payload_sha256(_history_binding(record)),
    )


def _analysis_lock() -> dict[str, Any]:
    lock = {
        "schema": POLICY_LOCK_SCHEMA,
        "schema_version": 3,
        "contains_source_text": False,
        "status": "frozen-before-v3-model-scoring",
        "selection": {
            "source_only": True,
            "model_outputs_used": False,
            "output_based_replacement": False,
            "all_32_strict_v2_targets_reserved_before_support_assignment": True,
            "target_clusters": TARGET_CLUSTERS,
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "nested_history_instances": HISTORY_INSTANCES,
            "selection_seed": SELECTION_SEED,
            "maximum_variant_retries": MAXIMUM_VARIANT_RETRIES,
            "one_unique_retained_control_per_cluster": True,
            "retained_control_fixed_across_cluster_histories": True,
            "globally_unique_tail_sources": True,
            "cross_cluster_all_role_source_disjointness": True,
            "tail_minimum_schedule": list(chat_v2.TAIL_MINIMUM_SCHEDULE),
        },
        "geometry": {
            "window_tokens": chat_v2.WINDOW,
            "minimum_tokens_strictly_after_owned": (
                chat_v2.MINIMUM_TOKENS_AFTER_OWNED
            ),
            "maximum_runtime_total_token_bound": (
                chat_v2.MAXIMUM_CONTEXT_TOKENS
            ),
            "fixed_c_reference_required": True,
            "all_affected_boundaries_must_be_feasible": True,
            "full_repack_fallback_allowed": False,
        },
        "exposure_disclosure": {
            "classes": copy.deepcopy(EXPOSURE_CLASS_COUNTS),
            "all_32_can_be_called_source_unseen": False,
            "source_unseen_target_clusters": 14,
            "pre_outcome_target_membership": True,
            "support_sources_fresh_to_prior_scoring": True,
        },
        "admission": {
            "unit": "history_instance",
            "target_present_minus_raw_full_sequence_mean_logprob_nats_minimum": (
                chat_v2.MINIMUM_TARGET_PRESENT_MINUS_RAW_FULL_SEQUENCE_MEAN_LOGPROB_NATS
            ),
            "target_present_first_token_rank_maximum": (
                chat_v2.MAXIMUM_TARGET_PRESENT_FIRST_TOKEN_RANK
            ),
            "retained_present_and_raw_first_token_rank_maximum": (
                chat_v2.MAXIMUM_RETAINED_FIRST_TOKEN_RANK_IN_PRESENT_AND_RAW
            ),
            "target_full_sequence_scoring_required": True,
            "threshold_overrides_allowed": False,
            "primary_itt_denominator_ignores_admission": True,
            "operational_or_admission_failure_primary_value": 0,
            "conditional_efficacy_requires_joint_admission": True,
        },
        "analysis": {
            "primary_unit": "target_cluster",
            "primary_n": PRIMARY_ANALYSIS_N,
            "history_instances_are_nested_repeats": True,
            "history_instances_are_independent_records": False,
            "primary_estimand": (
                "cluster-level mean across three frozen histories, then "
                "mean across all 32 target clusters"
            ),
            "primary_population": "all 32 pre-outcome strict-v2 target clusters",
            "primary_endpoint": "intent-to-treat",
            "operational_failures_in_primary": (
                "retained in denominator and scored as endpoint failure"
            ),
            "secondary_population": (
                "14 source-unseen target clusters, explicitly secondary"
            ),
            "conditional_endpoints": (
                "secondary only; report numerator and conditioning denominator"
            ),
            "confidence_interval": {
                "method": "percentile cluster bootstrap",
                "resampling_unit": "target_cluster",
                "resamples": 100000,
                "seed": SELECTION_SEED,
                "confidence_level": 0.95,
                "histories_resampled_within_cluster": False,
            },
            "binary_rate_interval": {
                "method": "Wilson score interval",
                "confidence_level": 0.95,
                "denominator": "target clusters",
            },
            "multiplicity": "no multiplicity-adjusted confirmatory claim",
        },
        "claim_language": {
            "permitted": [
                "32 target clusters",
                "96 histories nested within 32 target clusters",
                "N=32",
                "14 source-unseen target clusters",
            ],
            "prohibited": [
                "N=96",
                "96 independent records",
                "32 source-unseen target clusters",
                "100 independent records",
            ],
        },
    }
    lock["lock_sha256"] = _payload_sha256(lock)
    validate_policy_lock(lock)
    return lock


def validate_policy_lock(lock: Mapping[str, Any]) -> None:
    body = copy.deepcopy(dict(lock))
    observed = body.pop("lock_sha256", None)
    if (
        observed != _payload_sha256(body)
        or lock.get("schema") != POLICY_LOCK_SCHEMA
        or lock.get("contains_source_text") is not False
        or lock.get("selection", {}).get("model_outputs_used") is not False
        or lock.get("selection", {}).get("target_clusters")
        != TARGET_CLUSTERS
        or lock.get("selection", {}).get("nested_history_instances")
        != HISTORY_INSTANCES
        or lock.get("analysis", {}).get("primary_n") != PRIMARY_ANALYSIS_N
        or lock.get("analysis", {}).get("primary_unit") != "target_cluster"
        or lock.get("exposure_disclosure", {}).get("classes")
        != EXPOSURE_CLASS_COUNTS
    ):
        raise ManifestError("v3 cluster analysis lock drifted")
    leaked = chat_v1._FORBIDDEN_SOURCE_KEYS.intersection(
        chat_v1._walk_keys(lock)
    )
    if leaked:
        raise ManifestError("v3 analysis lock contains source text")


@dataclass(frozen=True)
class _FrozenHistory:
    target_source_id: str
    control_source_id: str
    tail_source_ids: frozenset[str]
    preparation: chat_v2.CandidatePreparation
    variant_index: int
    history_seed: int
    retry_ordinal: int
    control_rank: int | None


def _control_order(
    target: base.LongMemEvalExample,
    pool: Sequence[base.LongMemEvalExample],
) -> list[base.LongMemEvalExample]:
    controls = [
        example
        for example in pool
        if example.question_id != target.question_id
        and not base.retained_rejection_reasons(example)
        and len(
            [
                session
                for session in base.evidence_sessions(example)
                if any(turn.has_answer is True for turn in session.turns)
            ]
        )
        == 1
    ]
    return sorted(
        controls,
        key=lambda example: (
            sum(
                len(turn.content)
                for session in base.evidence_sessions(example)
                if any(turn.has_answer is True for turn in session.turns)
                for turn in base.evidence_turn_excerpt(session).turns
            ),
            base._selection_key(
                "longmemeval-retained-pair-v1",
                SELECTION_SEED,
                target.source_id,
                example.source_id,
            ),
            example.question_id,
            example.source_id,
        ),
    )


def _prepare_explicit_history(
    target: base.LongMemEvalExample,
    retained: base.LongMemEvalExample,
    pool: Sequence[base.LongMemEvalExample],
    examples_by_source: Mapping[str, base.LongMemEvalExample],
    tokenizer: Any,
    *,
    seed: int,
) -> chat_v2.CandidatePreparation | None:
    for tail_minimum in chat_v2.TAIL_MINIMUM_SCHEDULE:
        try:
            with chat_v2._v2_geometry():
                assembly = base.package_context(
                    tokenizer,
                    target,
                    retained,
                    pool,
                    seed=seed,
                    minimum_tokens_after_owned=tail_minimum,
                    context_ceiling=chat_v2.MAXIMUM_CONTEXT_TOKENS,
                )
                payload = base._record_payload(
                    target,
                    retained,
                    assembly,
                    context_ceiling=chat_v2.MAXIMUM_CONTEXT_TOKENS,
                )
            spec = chat_v1._record_spec(
                payload,
                artifact_integrity=chat_v2.SELECTION_CONTRACT_SHA256,
            )
            prepared = chat_v2._prepare_from_spec(
                spec,
                examples_by_source,
                tokenizer,
                base_tail_minimum_tokens=tail_minimum,
            )
            if not all(
                bool(item.get("feasible"))
                for item in prepared.runtime.fixed_c_diagnostics
            ):
                return None
            return prepared
        except (
            ManifestError,
            ValueError,
            KeyError,
            IndexError,
            LookupError,
        ):
            continue
    return None


def _strict_v2_targets(
    examples: Sequence[base.LongMemEvalExample],
    tokenizer: Any,
    *,
    v2_census: Mapping[str, Any],
    development_exposed_ids: frozenset[str],
) -> list[base.LongMemEvalExample]:
    examples_by_source = {example.source_id: example for example in examples}
    untouched = [
        example
        for example in examples
        if example.source_id not in development_exposed_ids
    ]
    ordered_candidates = [
        example
        for example in base.deterministic_source_order(
            untouched,
            seed=SELECTION_SEED,
        )
        if chat_v2._eligible_target(example, development_exposed_ids)
    ]
    audited_hashes = [
        str(item["target_source_id_sha256"]) for item in v2_census["audit"]
    ]
    if [base.text_sha256(item.source_id) for item in ordered_candidates] != (
        audited_hashes
    ):
        raise ManifestError("strict-v2 target ordering drifted from census")
    targets: list[base.LongMemEvalExample] = []
    observed_statuses: list[str] = []
    for target in ordered_candidates:
        prepared, _reason = chat_v2._prepare_candidate(
            target,
            untouched,
            examples_by_source,
            tokenizer,
        )
        if prepared is None:
            observed_statuses.append("ineligible")
        else:
            observed_statuses.append("eligible")
            targets.append(target)
    expected_statuses = [
        (
            "ineligible"
            if item["status"] == "ineligible"
            else "eligible"
        )
        for item in v2_census["audit"]
    ]
    if observed_statuses != expected_statuses:
        raise ManifestError("strict-v2 individual eligibility drifted")
    if len(targets) != TARGET_CLUSTERS:
        raise ManifestError(
            f"strict-v2 census target count drifted: {len(targets)}"
        )
    return targets


def _public_history(
    frozen: _FrozenHistory,
    *,
    exposure_class: str,
) -> dict[str, Any]:
    record = copy.deepcopy(frozen.preparation.public_record)
    record["cluster_id"] = _cluster_id(frozen.target_source_id)
    record["analysis_unit_id"] = record["cluster_id"]
    record["variant_index"] = frozen.variant_index
    record["history_seed"] = frozen.history_seed
    record["history_retry_ordinal"] = frozen.retry_ordinal
    record["control_selection_rank"] = frozen.control_rank
    record["target_exposure_class"] = exposure_class
    record["support_source_exposure"] = "fresh-to-all-prior-scoring"
    record["tail_source_count"] = len(frozen.tail_source_ids)
    record["history_binding_sha256"] = _payload_sha256(
        _history_binding(record)
    )
    record["record_id"] = _history_id(record)
    return record


def _freeze_histories(
    targets: Sequence[base.LongMemEvalExample],
    fresh_support: Mapping[str, base.LongMemEvalExample],
    examples_by_source: Mapping[str, base.LongMemEvalExample],
    tokenizer: Any,
    *,
    prior_scored_ids: frozenset[str],
    direct_target_ids: frozenset[str],
    context_only_ids: frozenset[str],
) -> list[dict[str, Any]]:
    target_ids = {target.source_id for target in targets}
    controls: dict[str, base.LongMemEvalExample] = {}
    control_ids: set[str] = set()
    tail_ids: set[str] = set()
    used_support: set[str] = set()
    frozen: list[_FrozenHistory] = []

    for target in targets:
        remaining = [
            fresh_support[source_id]
            for source_id in sorted(fresh_support)
            if source_id not in used_support
        ]
        selected: tuple[
            base.LongMemEvalExample,
            chat_v2.CandidatePreparation,
            frozenset[str],
            int,
        ] | None = None
        for rank, control in enumerate(_control_order(target, remaining)):
            prepared = _prepare_explicit_history(
                target,
                control,
                [target, *remaining],
                examples_by_source,
                tokenizer,
                seed=SELECTION_SEED,
            )
            if prepared is None:
                continue
            roles = set(prepared.role_source_ids) - {target.source_id}
            tails = frozenset(roles - {control.source_id})
            if roles.intersection(
                used_support | prior_scored_ids | target_ids
            ):
                raise ManifestError("variant-0 support collision")
            selected = control, prepared, tails, rank
            break
        if selected is None:
            raise ManifestError(
                "cannot assign a fresh retained control to target "
                f"{base.text_sha256(target.source_id)}"
            )
        control, prepared, tails, rank = selected
        controls[target.source_id] = control
        control_ids.add(control.source_id)
        used_support.update({control.source_id, *tails})
        tail_ids.update(tails)
        frozen.append(
            _FrozenHistory(
                target.source_id,
                control.source_id,
                tails,
                prepared,
                0,
                SELECTION_SEED,
                0,
                rank,
            )
        )

    for variant_index in range(1, HISTORIES_PER_CLUSTER):
        for target in targets:
            control = controls[target.source_id]
            available = [
                fresh_support[source_id]
                for source_id in sorted(fresh_support)
                if source_id not in control_ids
                and source_id not in tail_ids
            ]
            selected_variant: tuple[
                chat_v2.CandidatePreparation,
                frozenset[str],
                int,
                int,
            ] | None = None
            for retry_ordinal in range(MAXIMUM_VARIANT_RETRIES):
                seed = _history_seed(
                    target.source_id,
                    variant_index=variant_index,
                    retry_ordinal=retry_ordinal,
                )
                prepared = _prepare_explicit_history(
                    target,
                    control,
                    [target, control, *available],
                    examples_by_source,
                    tokenizer,
                    seed=seed,
                )
                if prepared is None:
                    continue
                tails = frozenset(
                    set(prepared.role_source_ids)
                    - {target.source_id, control.source_id}
                )
                if tails.intersection(
                    tail_ids
                    | control_ids
                    | prior_scored_ids
                    | target_ids
                ):
                    raise ManifestError("variant support collision")
                selected_variant = prepared, tails, retry_ordinal, seed
                break
            if selected_variant is None:
                raise ManifestError(
                    "cannot assign a fresh history variant to target "
                    f"{base.text_sha256(target.source_id)}"
                )
            prepared, tails, retry_ordinal, seed = selected_variant
            tail_ids.update(tails)
            frozen.append(
                _FrozenHistory(
                    target.source_id,
                    control.source_id,
                    tails,
                    prepared,
                    variant_index,
                    seed,
                    retry_ordinal,
                    None,
                )
            )

    def exposure_class(source_id: str) -> str:
        if source_id in direct_target_ids:
            return "direct-target"
        if source_id in context_only_ids:
            return "context-only"
        return "source-unseen"

    return [
        _public_history(
            item,
            exposure_class=exposure_class(item.target_source_id),
        )
        for item in sorted(
            frozen,
            key=lambda item: (
                targets.index(examples_by_source[item.target_source_id]),
                item.variant_index,
            ),
        )
    ]


def _freeze_manifest(body: Mapping[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(body))
    frozen.pop("integrity", None)
    for history in frozen["histories"]:
        history.pop("record_integrity", None)
        history["record_integrity"] = {
            "algorithm": "sha256",
            "sha256": _payload_sha256(history),
        }
    for cluster in frozen["clusters"]:
        cluster.pop("cluster_integrity", None)
        cluster["cluster_integrity"] = {
            "algorithm": "sha256",
            "sha256": _payload_sha256(cluster),
        }
    frozen["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": _payload_sha256(frozen),
    }
    validate_manifest(frozen)
    return frozen


def _clusters(histories: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for history in histories:
        grouped.setdefault(str(history["cluster_id"]), []).append(history)
    result = []
    for cluster_index, cluster_id in enumerate(
        dict.fromkeys(str(item["cluster_id"]) for item in histories)
    ):
        rows = sorted(
            grouped[cluster_id],
            key=lambda item: int(item["variant_index"]),
        )
        first = rows[0]
        result.append(
            {
                "cluster_id": cluster_id,
                "cluster_index": cluster_index,
                "analysis_unit_id": cluster_id,
                "target": copy.deepcopy(first["target"]),
                "target_source_id_sha256": base.text_sha256(
                    first["target"]["source_id"]
                ),
                "target_exposure_class": first["target_exposure_class"],
                "retained_control": copy.deepcopy(first["retained_probe"]),
                "retained_control_source_id_sha256": base.text_sha256(
                    first["retained_probe"]["source_id"]
                ),
                "history_ids": [str(item["record_id"]) for item in rows],
                "history_ids_sha256": _payload_sha256(
                    [str(item["record_id"]) for item in rows]
                ),
                "nested_history_count": len(rows),
            }
        )
    return result


def _census_body(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
) -> dict[str, Any]:
    histories = manifest["histories"]
    support_ids = {
        history["retained_probe"]["source_id"] for history in histories
    } | {
        reference["source_id"]
        for history in histories
        for reference in history["tail_session_references"]
    }
    tail_ids = {
        reference["source_id"]
        for history in histories
        for reference in history["tail_session_references"]
    }
    contexts = [history["context"] for history in histories]
    return {
        "schema": CENSUS_SCHEMA,
        "schema_version": 3,
        "contains_source_text": False,
        "claim": {
            "target_clusters": TARGET_CLUSTERS,
            "nested_histories": HISTORY_INSTANCES,
            "independent_analysis_n": PRIMARY_ANALYSIS_N,
            "source_unseen_target_clusters": 14,
            "all_targets_source_unseen": False,
        },
        "strict_v2_target_census": copy.deepcopy(
            manifest["target_census"]
        ),
        "exposure": copy.deepcopy(manifest["exposure_summary"]),
        "support": {
            "fresh_support_pool": manifest["support_inventory"][
                "fresh_support_pool"
            ],
            "unique_retained_controls": TARGET_CLUSTERS,
            "unique_tail_sources": len(tail_ids),
            "unique_support_sources_used": len(support_ids),
            "unused_fresh_support_sources": (
                manifest["support_inventory"]["fresh_support_pool"]
                - len(support_ids)
            ),
            "support_source_ids_sha256": _payload_sha256(
                _source_hashes(support_ids)
            ),
            "tail_source_ids_sha256": _payload_sha256(
                _source_hashes(tail_ids)
            ),
        },
        "geometry": {
            "fixed_c_feasible_histories": sum(
                bool(
                    context["fixed_c_reference"][
                        "all_affected_boundaries_feasible"
                    ]
                )
                for context in contexts
            ),
            "minimum_tokens_strictly_after_owned": min(
                int(context["tokens_strictly_after_owned"])
                for context in contexts
            ),
            "maximum_runtime_total_token_bound": max(
                int(context["runtime_total_token_bound"])
                for context in contexts
            ),
            "full_repack_fallback_required_histories": sum(
                bool(context["full_repack_fallback"]["required"])
                for context in contexts
            ),
        },
        "ordered_cluster_ids": [
            cluster["cluster_id"] for cluster in manifest["clusters"]
        ],
        "ordered_history_ids": [
            history["record_id"] for history in histories
        ],
        "cohort_integrity_sha256": manifest["integrity"]["sha256"],
        "policy_lock_sha256": policy_lock["lock_sha256"],
    }


def _freeze_census(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
) -> dict[str, Any]:
    census = _census_body(manifest, policy_lock)
    census["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(census),
    }
    validate_census(census, manifest, policy_lock)
    return census


def build_cohort(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    v2_manifest: ManifestInput = DEFAULT_V2_MANIFEST_PATH,
    v2_policy_lock: ManifestInput = DEFAULT_V2_POLICY_LOCK_PATH,
    v2_census: ManifestInput = DEFAULT_V2_CENSUS_PATH,
    geometry_manifest: ManifestInput = DEFAULT_GEOMETRY_MANIFEST_PATH,
    require_pinned_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the policy lock, 32-cluster cohort, and source/exposure census."""

    if getattr(tokenizer, "is_fast", True) is not True:
        raise ManifestError("fast tokenizer offset mappings are required")
    old_v2, old_v2_path = _load_json(v2_manifest)
    old_lock, old_lock_path = _load_json(v2_policy_lock)
    old_census, old_census_path = _load_json(v2_census)
    geometry, geometry_path = _load_json(geometry_manifest)
    chat_v2.validate_manifest(old_v2)
    chat_v2.validate_census(old_census, old_v2)
    if old_v2["policy_lock"] != old_lock:
        raise ManifestError("v2 manifest and standalone policy lock differ")
    if require_pinned_artifacts and (
        old_v2["integrity"]["sha256"]
        != chat_v2.PINNED_V2_CONFIRMATION_INTEGRITY_SHA256
        or old_lock["lock_sha256"] != chat_v2.PINNED_V2_POLICY_LOCK_SHA256
        or old_census["integrity"]["sha256"]
        != chat_v2.PINNED_V2_CENSUS_INTEGRITY_SHA256
    ):
        raise ManifestError("pinned strict-v2 artifacts drifted")

    exposure = chat_v2.resolve_exposure_artifacts(
        require_pinned=require_pinned_artifacts
    )
    examples = base.extract_longmemeval_examples(rows)
    descriptors = chat_v1._source_descriptors(examples)
    if require_pinned_artifacts and (
        len(descriptors) != chat_v2.PINNED_FULL_ROW_COUNT
        or _payload_sha256(descriptors)
        != chat_v2.PINNED_FULL_DESCRIPTORS_SHA256
    ):
        raise ManifestError("pinned LongMemEval oracle inventory drifted")
    examples_by_source = {example.source_id: example for example in examples}
    if len(examples_by_source) != len(examples):
        raise ManifestError("LongMemEval source IDs are duplicated")

    targets = _strict_v2_targets(
        examples,
        tokenizer,
        v2_census=old_census,
        development_exposed_ids=exposure.exposed_source_ids,
    )
    target_ids = frozenset(target.source_id for target in targets)
    v2_role_ids = frozenset(chat_v2.all_role_source_ids(old_v2))
    v2_target_ids = frozenset(
        record["target"]["source_id"] for record in old_v2["records"]
    )
    geometry_ids = frozenset(_iter_source_ids(geometry))
    prior_scored_ids = frozenset(
        set(exposure.exposed_source_ids) | set(v2_role_ids) | set(geometry_ids)
    )
    context_only_ids = frozenset(
        (set(v2_role_ids) - set(v2_target_ids)) & set(target_ids)
    )
    direct_target_ids = frozenset(set(v2_target_ids) & set(target_ids))
    class_counts = {
        "direct-target": len(direct_target_ids),
        "context-only": len(context_only_ids),
        "source-unseen": len(
            set(target_ids) - set(prior_scored_ids)
        ),
    }
    if class_counts != EXPOSURE_CLASS_COUNTS:
        raise ManifestError(
            f"target exposure classes drifted: {class_counts}"
        )

    fresh_support = {
        example.source_id: example
        for example in examples
        if example.source_id not in prior_scored_ids
        and example.source_id not in target_ids
    }
    histories = _freeze_histories(
        targets,
        fresh_support,
        examples_by_source,
        tokenizer,
        prior_scored_ids=prior_scored_ids,
        direct_target_ids=direct_target_ids,
        context_only_ids=context_only_ids,
    )
    clusters = _clusters(histories)
    policy_lock = _analysis_lock()
    source_artifacts = [
        _artifact_descriptor(
            old_v2,
            old_v2_path,
            role="strict-v2-confirmation",
        ),
        _artifact_descriptor(
            old_census,
            old_census_path,
            role="strict-v2-pre-outcome-census",
        ),
        {
            "role": "strict-v2-policy-lock",
            "repository_relative_path": (
                None
                if old_lock_path is None
                else _repository_relative(old_lock_path)
            ),
            "file_sha256": (
                None if old_lock_path is None else _file_sha256(old_lock_path)
            ),
            "lock_sha256": old_lock["lock_sha256"],
        },
        _artifact_descriptor(
            geometry,
            geometry_path,
            role="geometry-corrected-prior-scoring",
        ),
        *copy.deepcopy(list(exposure.descriptors)),
    ]
    for descriptor in source_artifacts:
        path = descriptor.get("path")
        if path is not None:
            descriptor["repository_relative_path"] = _repository_relative(
                Path(path)
            )
            descriptor.pop("path", None)
    body = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "benchmark_label": BENCHMARK_LABEL,
        "contains_source_text": False,
        "partition": PARTITION,
        "status": "frozen-before-v3-model-scoring",
        "provenance": {
            **base._official_provenance(),
            "adaptation": (
                "pre-outcome strict-v2 target census with three fresh, "
                "source-only histories nested per target"
            ),
            "official_longmemeval_leaderboard_score": False,
        },
        "source_inventory": {
            "dataset_id": base.DATASET_ID,
            "dataset_revision": base.DATASET_REVISION,
            "source_artifact_sha256": base.DATASET_ARTIFACT_SHA256,
            "full_row_count": len(descriptors),
            "full_descriptors_sha256": _payload_sha256(descriptors),
            "input_artifacts": source_artifacts,
        },
        "target_census": {
            "origin": "pre-outcome strict-v2 individual eligibility census",
            "candidate_targets": len(old_census["audit"]),
            "individually_eligible_targets": len(targets),
            "reserved_before_support_assignment": True,
            "ordered_target_source_id_sha256": [
                base.text_sha256(target.source_id) for target in targets
            ],
            "strict_v2_census_integrity_sha256": old_census["integrity"][
                "sha256"
            ],
        },
        "exposure_summary": {
            "definition": "any source role in prior scored chat artifacts",
            "prior_scored_source_count": len(prior_scored_ids),
            "prior_scored_source_id_sha256": _source_hashes(
                prior_scored_ids
            ),
            "target_class_counts": class_counts,
            "direct_target_source_id_sha256": _source_hashes(
                direct_target_ids
            ),
            "context_only_source_id_sha256": _source_hashes(
                context_only_ids
            ),
            "source_unseen_target_source_id_sha256": _source_hashes(
                set(target_ids) - set(prior_scored_ids)
            ),
            "only_source_unseen_target_clusters": 14,
        },
        "support_inventory": {
            "fresh_support_pool": len(fresh_support),
            "fresh_definition": (
                "absent from all prior scored source roles and all 32 targets"
            ),
            "retained_controls": TARGET_CLUSTERS,
            "retained_control_fixed_across_variants": True,
            "globally_unique_tail_sources": True,
        },
        "policy_lock": copy.deepcopy(policy_lock),
        "tokenizer": {
            "model_id": chat_v2.TOKENIZER_ID,
            "revision": chat_v2.TOKENIZER_REVISION,
            "fast_offset_mapping_required": True,
        },
        "chat_serialization": chat_v1._chat_serialization(),
        "selection_policy": copy.deepcopy(policy_lock["selection"]),
        "analysis_contract": copy.deepcopy(policy_lock["analysis"]),
        "claim_language": copy.deepcopy(policy_lock["claim_language"]),
        "counts": {
            "target_clusters": len(clusters),
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "nested_history_instances": len(histories),
            "independent_analysis_n": PRIMARY_ANALYSIS_N,
        },
        "clusters": clusters,
        "histories": histories,
    }
    manifest = _freeze_manifest(body)
    census = _freeze_census(manifest, policy_lock)
    return manifest, policy_lock, census


def _history_role_ids(history: Mapping[str, Any]) -> set[str]:
    return {
        history["target"]["source_id"],
        history["retained_probe"]["source_id"],
        *(
            reference["source_id"]
            for reference in history["tail_session_references"]
        ),
    }


def _validate_without_integrity(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("contains_source_text") is not False
        or manifest.get("partition") != PARTITION
        or manifest.get("status") != "frozen-before-v3-model-scoring"
    ):
        raise ManifestError("unsupported v3 clustered cohort")
    leaked = chat_v1._FORBIDDEN_SOURCE_KEYS.intersection(
        chat_v1._walk_keys(manifest)
    )
    if leaked:
        raise ManifestError("v3 cohort contains source text")
    policy = manifest.get("policy_lock")
    if not isinstance(policy, Mapping):
        raise ManifestError("v3 policy lock is missing")
    validate_policy_lock(policy)
    if (
        manifest.get("selection_policy") != policy["selection"]
        or manifest.get("analysis_contract") != policy["analysis"]
        or manifest.get("claim_language") != policy["claim_language"]
    ):
        raise ManifestError("embedded v3 lock contracts drifted")

    histories = manifest.get("histories")
    clusters = manifest.get("clusters")
    counts = manifest.get("counts") or {}
    if (
        not isinstance(histories, list)
        or len(histories) != HISTORY_INSTANCES
        or not isinstance(clusters, list)
        or len(clusters) != TARGET_CLUSTERS
        or counts
        != {
            "target_clusters": TARGET_CLUSTERS,
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "nested_history_instances": HISTORY_INSTANCES,
            "independent_analysis_n": PRIMARY_ANALYSIS_N,
        }
    ):
        raise ManifestError("v3 cohort counts drifted")

    target_ids: set[str] = set()
    control_ids: set[str] = set()
    tail_ids: set[str] = set()
    cluster_role_ids: set[str] = set()
    exposure_counts = {key: 0 for key in EXPOSURE_CLASS_COUNTS}
    observed_cluster_ids: list[str] = []
    observed_history_ids: set[str] = set()
    prior_source_hashes = set(
        manifest["exposure_summary"]["prior_scored_source_id_sha256"]
    )
    class_hashes = {
        "direct-target": set(
            manifest["exposure_summary"][
                "direct_target_source_id_sha256"
            ]
        ),
        "context-only": set(
            manifest["exposure_summary"][
                "context_only_source_id_sha256"
            ]
        ),
        "source-unseen": set(
            manifest["exposure_summary"][
                "source_unseen_target_source_id_sha256"
            ]
        ),
    }
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id") or "")
        rows = [
            history
            for history in histories
            if history.get("cluster_id") == cluster_id
        ]
        if (
            not cluster_id
            or len(rows) != HISTORIES_PER_CLUSTER
            or sorted(int(row["variant_index"]) for row in rows)
            != list(range(HISTORIES_PER_CLUSTER))
            or cluster["history_ids"]
            != [
                row["record_id"]
                for row in sorted(
                    rows,
                    key=lambda item: int(item["variant_index"]),
                )
            ]
            or cluster["history_ids_sha256"]
            != _payload_sha256(cluster["history_ids"])
            or cluster["nested_history_count"] != HISTORIES_PER_CLUSTER
        ):
            raise ManifestError("v3 cluster history binding drifted")
        target_id = str(cluster["target"]["source_id"])
        control_id = str(cluster["retained_control"]["source_id"])
        if (
            cluster_id != _cluster_id(target_id)
            or cluster["analysis_unit_id"] != cluster_id
            or cluster["target_source_id_sha256"]
            != base.text_sha256(target_id)
            or cluster["retained_control_source_id_sha256"]
            != base.text_sha256(control_id)
            or any(row["target"] != cluster["target"] for row in rows)
            or any(
                row["retained_probe"] != cluster["retained_control"]
                for row in rows
            )
        ):
            raise ManifestError("v3 cluster target/control binding drifted")
        role_ids = set().union(*(_history_role_ids(row) for row in rows))
        if (
            role_ids.intersection(cluster_role_ids)
            or target_id in target_ids
            or control_id in control_ids
            or target_id == control_id
        ):
            raise ManifestError("v3 cross-cluster source collision")
        cluster_role_ids.update(role_ids)
        target_ids.add(target_id)
        control_ids.add(control_id)
        class_name = str(cluster["target_exposure_class"])
        target_hash = base.text_sha256(target_id)
        if (
            class_name not in exposure_counts
            or target_hash not in class_hashes[class_name]
            or (
                class_name == "source-unseen"
                and target_hash in prior_source_hashes
            )
            or (
                class_name != "source-unseen"
                and target_hash not in prior_source_hashes
            )
        ):
            raise ManifestError("unknown v3 target exposure class")
        exposure_counts[class_name] += 1
        observed_cluster_ids.append(cluster_id)

    for history in histories:
        history_id = str(history.get("record_id") or "")
        context = history.get("context") or {}
        fixed_c = context.get("fixed_c_reference") or {}
        fallback = context.get("full_repack_fallback") or {}
        local = context.get("local_window_safety") or {}
        history_tail_ids = {
            reference["source_id"]
            for reference in history["tail_session_references"]
        }
        if (
            history_id in observed_history_ids
            or history_id != _history_id(history)
            or history["history_binding_sha256"]
            != _payload_sha256(_history_binding(history))
            or history["analysis_unit_id"] != history["cluster_id"]
            or int(context["tokens_strictly_after_owned"])
            < chat_v2.MINIMUM_TOKENS_AFTER_OWNED
            or int(context["runtime_total_token_bound"])
            > chat_v2.MAXIMUM_CONTEXT_TOKENS
            or local.get("selected_span_inside_local_window") is not False
            or fixed_c.get("all_affected_boundaries_feasible") is not True
            or fallback.get("required") is not False
            or history_tail_ids.intersection(tail_ids)
            or history["support_source_exposure"]
            != "fresh-to-all-prior-scoring"
        ):
            raise ManifestError("v3 history binding or geometry drifted")
        observed_history_ids.add(history_id)
        tail_ids.update(history_tail_ids)
    if (
        exposure_counts != EXPOSURE_CLASS_COUNTS
        or manifest["exposure_summary"]["target_class_counts"]
        != EXPOSURE_CLASS_COUNTS
        or len(target_ids) != TARGET_CLUSTERS
        or len(control_ids) != TARGET_CLUSTERS
        or control_ids.intersection(target_ids | tail_ids)
        or target_ids.intersection(tail_ids)
        or {
            base.text_sha256(source_id)
            for source_id in control_ids | tail_ids
        }.intersection(prior_source_hashes)
        or manifest["target_census"]["ordered_target_source_id_sha256"]
        != [
            base.text_sha256(cluster["target"]["source_id"])
            for cluster in clusters
        ]
    ):
        raise ManifestError("v3 exposure or role uniqueness drifted")
    if observed_cluster_ids != [
        cluster["cluster_id"] for cluster in clusters
    ]:
        raise ManifestError("v3 cluster ordering drifted")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _validate_without_integrity(manifest)
    for history in manifest["histories"]:
        integrity = history.get("record_integrity")
        body = copy.deepcopy(dict(history))
        body.pop("record_integrity", None)
        if (
            not isinstance(integrity, Mapping)
            or integrity.get("algorithm") != "sha256"
            or integrity.get("sha256") != _payload_sha256(body)
        ):
            raise ManifestError("v3 history integrity mismatch")
    for cluster in manifest["clusters"]:
        integrity = cluster.get("cluster_integrity")
        body = copy.deepcopy(dict(cluster))
        body.pop("cluster_integrity", None)
        if (
            not isinstance(integrity, Mapping)
            or integrity.get("algorithm") != "sha256"
            or integrity.get("sha256") != _payload_sha256(body)
        ):
            raise ManifestError("v3 cluster integrity mismatch")
    integrity = manifest.get("integrity")
    body = copy.deepcopy(dict(manifest))
    body.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(body)
    ):
        raise ManifestError("v3 cohort integrity mismatch")


def validate_census(
    census: Mapping[str, Any],
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
) -> None:
    validate_manifest(manifest)
    validate_policy_lock(policy_lock)
    body = copy.deepcopy(dict(census))
    integrity = body.pop("integrity", None)
    if (
        census.get("schema") != CENSUS_SCHEMA
        or census.get("contains_source_text") is not False
        or not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(body)
        or body != _census_body(manifest, policy_lock)
    ):
        raise ManifestError("v3 census/exposure ledger drifted")
    leaked = chat_v1._FORBIDDEN_SOURCE_KEYS.intersection(
        chat_v1._walk_keys(census)
    )
    if leaked:
        raise ManifestError("v3 census contains source text")


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[chat_v1.RehydratedChatRecord, ...]:
    """Rebuild all 96 histories and reject source, token, or geometry drift."""

    validate_manifest(manifest)
    examples = base.extract_longmemeval_examples(rows)
    descriptors = chat_v1._source_descriptors(examples)
    inventory = manifest["source_inventory"]
    if (
        len(descriptors) != int(inventory["full_row_count"])
        or _payload_sha256(descriptors)
        != inventory["full_descriptors_sha256"]
    ):
        raise ManifestError("rehydrated LongMemEval v3 inventory drifted")
    examples_by_source = {example.source_id: example for example in examples}
    records: list[chat_v1.RehydratedChatRecord] = []
    for stored in manifest["histories"]:
        selection = stored["source_selection"]
        spec = {
            "baseline": {
                "record_id": selection["record_id"],
                "manifest_integrity_sha256": selection[
                    "manifest_integrity_sha256"
                ],
            },
            "target": copy.deepcopy(dict(stored["target"])),
            "retained_probe": copy.deepcopy(
                dict(stored["retained_probe"])
            ),
            "tail_session_references": copy.deepcopy(
                list(stored["tail_session_references"])
            ),
        }
        prepared = chat_v2._prepare_from_spec(
            spec,
            examples_by_source,
            tokenizer,
            base_tail_minimum_tokens=int(
                selection["base_tail_minimum_tokens"]
            ),
        )
        frozen = _FrozenHistory(
            target_source_id=str(stored["target"]["source_id"]),
            control_source_id=str(stored["retained_probe"]["source_id"]),
            tail_source_ids=frozenset(
                reference["source_id"]
                for reference in stored["tail_session_references"]
            ),
            preparation=prepared,
            variant_index=int(stored["variant_index"]),
            history_seed=int(stored["history_seed"]),
            retry_ordinal=int(stored["history_retry_ordinal"]),
            control_rank=(
                None
                if stored["control_selection_rank"] is None
                else int(stored["control_selection_rank"])
            ),
        )
        regenerated = _public_history(
            frozen,
            exposure_class=str(stored["target_exposure_class"]),
        )
        observed = copy.deepcopy(dict(stored))
        observed.pop("record_integrity", None)
        if regenerated != observed:
            raise ManifestError("rehydrated LongMemEval v3 history drifted")
        records.append(
            replace(prepared.runtime, record_id=str(stored["record_id"]))
        )
    return tuple(records)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the source-only clustered LongMemEval chat cohort v3"
    )
    parser.add_argument("--data-path")
    parser.add_argument(
        "--cohort-out",
        default=str(DEFAULT_COHORT_PATH),
    )
    parser.add_argument(
        "--policy-lock-out",
        default=str(DEFAULT_POLICY_LOCK_PATH),
    )
    parser.add_argument(
        "--census-out",
        default=str(DEFAULT_CENSUS_PATH),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destinations = [
        Path(args.cohort_out),
        Path(args.policy_lock_out),
        Path(args.census_out),
    ]
    if not args.overwrite:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(", ".join(existing))
    for path in destinations:
        _repository_relative(path)

    from transformers import AutoTokenizer

    rows = base.load_pinned_longmemeval_rows(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        chat_v2.TOKENIZER_ID,
        revision=chat_v2.TOKENIZER_REVISION,
        use_fast=True,
        local_files_only=True,
    )
    manifest, policy_lock, census = build_cohort(
        rows,
        tokenizer,
        require_pinned_artifacts=True,
    )
    write_json(args.cohort_out, manifest)
    write_json(args.policy_lock_out, policy_lock)
    write_json(args.census_out, census)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

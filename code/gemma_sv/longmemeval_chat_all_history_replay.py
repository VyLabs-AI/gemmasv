"""Source-free planning for LongMemEval all-history frozen-suffix replay.

The planner consumes already-rehydrated records and records token geometry only.
It never loads a model and never treats replay as a dependency-aware history
that was generated without the owned round.  The stored suffix is sliced from
the original history and replayed exactly as frozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv.demo_server.all_history_replay_runtime_v3 import ReplayTokenPlan


SCHEMA = "gemma-sv-longmemeval-chat-all-history-replay-lock-v1"
SCHEMA_VERSION = 1
STATUS = "frozen-before-all-history-replay"
EXPECTED_CLUSTERS = 32
HISTORIES_PER_CLUSTER = 3
EXPECTED_HISTORIES = EXPECTED_CLUSTERS * HISTORIES_PER_CLUSTER

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"

DEFAULT_COHORT_PATH = BENCHMARKS / "longmemeval_chat_cohort_v3.json"
DEFAULT_ANALYSIS_PATH = BENCHMARKS / "longmemeval_chat_v3_analysis_lock_v1.json"
DEFAULT_VALIDATION_PATH = (
    BENCHMARKS
    / "longmemeval_chat_response_generation_audit_v2_validation_v1.json"
)
DEFAULT_SUFFIX_PATH = (
    BENCHMARKS / "longmemeval_chat_suffix_contamination_v1.json"
)
DEFAULT_REPLAY_LOCK_PATH = (
    BENCHMARKS / "longmemeval_chat_all_history_replay_lock_v1.json"
)

IMPLEMENTATION_PATHS = (
    "gemma_sv/longmemeval_chat_all_history_replay.py",
    "gemma_sv/demo_server/all_history_replay_runtime_v3.py",
    "gemma_sv/demo_server/audit_gemma_runtime_v2.py",
    "gemma_sv/demo_server/gemma_engine.py",
    "gemma_sv/sv_global_attention.py",
    "svattn/causal_sv_attention.py",
)

_FORBIDDEN_LOCK_KEYS = frozenset(
    {
        "answer",
        "cluster_id",
        "content",
        "messages",
        "original_token_ids",
        "owned_token_ids",
        "prefix_token_ids",
        "prompt",
        "question",
        "raw_omitted_token_ids",
        "record_id",
        "response_text",
        "source_id",
        "source_text",
        "suffix_token_ids",
        "token_ids",
        "turns",
    }
)


class ReplayPlanError(ValueError):
    """A rehydrated history or replay lock violates the frozen contract."""


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
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Use GemmaRuntime's signed little-endian int64 token framing."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        value = int(token_id)
        if not -(1 << 63) <= value < (1 << 63):
            raise ReplayPlanError("token ID is outside signed int64")
        digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(
        character in "0123456789abcdef" for character in rendered
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
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != payload_sha256(body)
    ):
        raise ReplayPlanError(f"{name} integrity differs")


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).casefold()
            yield from _walk_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_keys(child)


def assert_source_free_lock(value: Mapping[str, Any]) -> None:
    leaked = _FORBIDDEN_LOCK_KEYS.intersection(_walk_keys(value))
    if leaked:
        raise ReplayPlanError(
            "replay lock contains source-bearing or token-array keys: "
            + ", ".join(sorted(leaked))
        )
    if (
        value.get("contains_source_text") is not False
        or value.get("contains_source_identifiers") is not False
        or value.get("contains_token_arrays") is not False
    ):
        raise ReplayPlanError("replay lock is not explicitly source-free")


def _as_token_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReplayPlanError(f"{name} must be a token sequence")
    result: list[int] = []
    for token_id in value:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ReplayPlanError(f"{name} contains a non-integer token")
        if token_id < 0:
            raise ReplayPlanError(f"{name} contains a negative token")
        result.append(int(token_id))
    return tuple(result)


def plan_history_replay(record: Any) -> ReplayTokenPlan:
    """Derive one exact prefix/owned/stored-suffix replay plan.

    Only the planner receives owned IDs.  The runtime replay API receives a
    prefix checkpoint, stored suffix, and expected omitted history.
    """

    context = getattr(record, "context", None)
    if context is None:
        raise ReplayPlanError("rehydrated record has no tokenized context")
    original = _as_token_tuple(
        getattr(context, "original_token_ids", None),
        name="original history",
    )
    omitted = _as_token_tuple(
        getattr(record, "raw_omitted_token_ids", None),
        name="raw omitted history",
    )
    edited = _as_token_tuple(
        getattr(context, "edited_token_ids", None),
        name="edited history",
    )
    positions = _as_token_tuple(
        getattr(context, "forget_positions", None),
        name="owned positions",
    )
    if not original or not positions:
        raise ReplayPlanError("replay requires non-empty original and ownership")
    if positions != tuple(range(positions[0], positions[-1] + 1)):
        raise ReplayPlanError("owned token range must be complete and contiguous")
    start = positions[0]
    end = positions[-1] + 1
    if start < 1 or end >= len(original):
        raise ReplayPlanError("replay requires non-empty prefix and stored suffix")
    deletion_ranges = tuple(
        tuple(int(item) for item in pair)
        for pair in getattr(context, "deletion_ranges", ())
    )
    if deletion_ranges != ((start, end),):
        raise ReplayPlanError("owned deletion range differs from contiguous range")

    prefix = original[:start]
    owned = original[start:end]
    suffix = original[end:]
    if original != prefix + owned + suffix:
        raise ReplayPlanError("original history recomposition failed")
    if omitted != prefix + suffix or edited != omitted:
        raise ReplayPlanError(
            "raw omission must equal prefix plus original stored suffix"
        )
    if suffix != original[end:]:
        raise ReplayPlanError("stored suffix was not sliced from original history")

    return ReplayTokenPlan(
        original_token_ids=original,
        owned_range=(start, end),
        prefix_token_ids=prefix,
        owned_token_ids=owned,
        suffix_token_ids=suffix,
        omitted_token_ids=omitted,
    )


def _suffix_rows(suffix_scan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = suffix_scan.get("histories")
    if not isinstance(rows, list) or len(rows) != EXPECTED_HISTORIES:
        raise ReplayPlanError("suffix scan must contain all 96 histories")
    return rows


def _suffix_stratum(
    suffix_row: Mapping[str, Any],
    *,
    history_index: int,
    cluster_index: int,
    variant_index: int,
) -> str:
    if (
        suffix_row.get("history_index") != history_index
        or suffix_row.get("cluster_index") != cluster_index
        or suffix_row.get("variant_index") != variant_index
    ):
        raise ReplayPlanError("suffix scan anonymous history order differs")
    tiers = (suffix_row.get("regions") or {}).get("suffix_any")
    if not isinstance(tiers, Mapping) or not isinstance(
        tiers.get("deterministic_any"), bool
    ):
        raise ReplayPlanError("suffix scan deterministic-any tier is missing")
    return "contaminated" if tiers["deterministic_any"] else "clean"


def _history_projection(
    plan: ReplayTokenPlan,
    *,
    history_index: int,
    cluster_index: int,
    variant_index: int,
    history_binding_sha256: str,
    suffix_stratum: str,
) -> dict[str, Any]:
    start, end = plan.owned_range
    return {
        "history_index": history_index,
        "cluster_index": cluster_index,
        "variant_index": variant_index,
        "history_binding_sha256": history_binding_sha256,
        "geometry": {
            "original_token_count": len(plan.original_token_ids),
            "prefix_token_count": len(plan.prefix_token_ids),
            "owned_token_count": len(plan.owned_token_ids),
            "stored_suffix_token_count": len(plan.suffix_token_ids),
            "omitted_history_token_count": len(plan.omitted_token_ids),
            "owned_start": start,
            "owned_end_exclusive": end,
            "owned_range_complete_and_contiguous": True,
            "original_equals_prefix_owned_suffix": True,
            "omitted_equals_prefix_suffix": True,
            "stored_suffix_sliced_from_original": True,
        },
        "hashes": {
            "original_sha256": token_ids_sha256(plan.original_token_ids),
            "prefix_sha256": token_ids_sha256(plan.prefix_token_ids),
            "owned_sha256": token_ids_sha256(plan.owned_token_ids),
            "stored_suffix_sha256": token_ids_sha256(plan.suffix_token_ids),
            "omitted_history_sha256": token_ids_sha256(
                plan.omitted_token_ids
            ),
        },
        "suffix_scan_stratum": suffix_stratum,
        "plan_sha256": plan.binding_sha256,
    }


def plan_all_histories(
    records: Sequence[Any],
    cohort: Mapping[str, Any],
    suffix_scan: Mapping[str, Any],
) -> tuple[tuple[ReplayTokenPlan, ...], tuple[dict[str, Any], ...]]:
    """Plan exactly K=32/n=96 ordered histories."""

    histories = cohort.get("histories")
    clusters = cohort.get("clusters")
    suffix_rows = _suffix_rows(suffix_scan)
    if (
        not isinstance(histories, list)
        or len(histories) != EXPECTED_HISTORIES
        or not isinstance(clusters, list)
        or len(clusters) != EXPECTED_CLUSTERS
        or len(records) != EXPECTED_HISTORIES
    ):
        raise ReplayPlanError("replay requires exactly K=32/n=96")
    cluster_indices = {
        str(row.get("cluster_id") or ""): row.get("cluster_index")
        for row in clusters
        if isinstance(row, Mapping)
    }
    plans: list[ReplayTokenPlan] = []
    rows: list[dict[str, Any]] = []
    for history_index, (record, stored, suffix_row) in enumerate(
        zip(records, histories, suffix_rows)
    ):
        cluster_index = history_index // HISTORIES_PER_CLUSTER
        variant_index = history_index % HISTORIES_PER_CLUSTER
        integrity = stored.get("record_integrity")
        history_digest = (
            integrity.get("sha256") if isinstance(integrity, Mapping) else None
        )
        if (
            getattr(record, "record_id", None) != stored.get("record_id")
            or cluster_indices.get(str(stored.get("cluster_id") or ""))
            != cluster_index
            or stored.get("variant_index") != variant_index
            or not _is_sha256(history_digest)
        ):
            raise ReplayPlanError("rehydrated record/cohort order differs")
        plan = plan_history_replay(record)
        rows.append(
            _history_projection(
                plan,
                history_index=history_index,
                cluster_index=cluster_index,
                variant_index=variant_index,
                history_binding_sha256=str(history_digest),
                suffix_stratum=_suffix_stratum(
                    suffix_row,
                    history_index=history_index,
                    cluster_index=cluster_index,
                    variant_index=variant_index,
                ),
            )
        )
        plans.append(plan)
    return tuple(plans), tuple(rows)


def artifact_binding(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ReplayPlanError(f"{source.name} must be a JSON object")
    integrity = value.get("integrity")
    return {
        "repository_path": source.resolve().relative_to(
            WORKSPACE.resolve()
        ).as_posix(),
        "file_sha256": file_sha256(source),
        "payload_sha256": payload_sha256(value),
        "integrity_sha256": (
            integrity.get("sha256")
            if isinstance(integrity, Mapping)
            else None
        ),
        "lock_sha256": value.get("lock_sha256"),
        "immutable_input": True,
    }


def default_artifact_bindings() -> dict[str, dict[str, Any]]:
    return {
        "cohort": artifact_binding(DEFAULT_COHORT_PATH),
        "v3_analysis_lock": artifact_binding(DEFAULT_ANALYSIS_PATH),
        "sealed_v2_validation": artifact_binding(DEFAULT_VALIDATION_PATH),
        "frozen_suffix_scan": artifact_binding(DEFAULT_SUFFIX_PATH),
    }


def implementation_bindings() -> dict[str, Any]:
    files = {
        relative: file_sha256(WORKSPACE / relative)
        for relative in IMPLEMENTATION_PATHS
    }
    return {
        "files": files,
        "file_count": len(files),
        "files_sha256": payload_sha256(files),
        "additive_runtime_subclasses_sealed_v2": True,
    }


def build_replay_lock(
    records: Sequence[Any],
    cohort: Mapping[str, Any],
    suffix_scan: Mapping[str, Any],
    *,
    artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    implementations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the anonymous replay lock in memory; no file is generated."""

    plans, rows = plan_all_histories(records, cohort, suffix_scan)
    artifact_rows = (
        default_artifact_bindings() if artifacts is None else artifacts
    )
    implementation_rows = (
        implementation_bindings()
        if implementations is None
        else implementations
    )
    if not isinstance(artifact_rows, Mapping) or not artifact_rows:
        raise ReplayPlanError("artifact bindings are required")
    if not isinstance(implementation_rows, Mapping):
        raise ReplayPlanError("implementation bindings are required")
    suffix_steps = sum(len(plan.suffix_token_ids) for plan in plans)
    contaminated = sum(
        row["suffix_scan_stratum"] == "contaminated" for row in rows
    )
    lock = _seal(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "contains_source_text": False,
            "contains_source_identifiers": False,
            "contains_token_arrays": False,
            "contains_model_outputs": False,
            "model_or_api_calls_made": 0,
            "artifacts": copy.deepcopy(dict(artifact_rows)),
            "implementation": copy.deepcopy(dict(implementation_rows)),
            "geometry": {
                "target_clusters": EXPECTED_CLUSTERS,
                "histories_per_cluster": HISTORIES_PER_CLUSTER,
                "history_instances": EXPECTED_HISTORIES,
                "independent_analysis_n": EXPECTED_CLUSTERS,
                "all_histories_planned": True,
            },
            "replay_semantics": {
                "condition_id": "frozen_suffix_replay",
                "checkpoint": "immediately before complete owned token range",
                "owned_ids_passed_to_replay_runtime": False,
                "stored_suffix_source": "exact slice of original token history",
                "suffix_token_order_preserved": True,
                "incremental_tokens_per_model_step": 1,
                "batch_multi_token_causal_shortcut_allowed": False,
                "dependency_aware_never_ingested_history": False,
                "disclosure": (
                    "Frozen-suffix replay is not a dependency-aware history "
                    "generated as if the owned round had never been ingested."
                ),
                "stratified_by_frozen_suffix_scan": True,
            },
            "bandwidth_schedule": {
                "source": "prefix_checkpoint",
                "fresh_rebuild_bandwidth_equivalence_assumed": False,
                "reason": (
                    "SV kernel bandwidth is data-derived and frozen at "
                    "prefill; a prefix checkpoint and one-shot omitted-history "
                    "prefill generally observe different key sets."
                ),
                "silent_kpar_override_allowed": False,
                "future_suffix_derived_override_allowed": False,
                "per_layer_values_recorded_at_runtime": True,
            },
            "state_contract": {
                "checkpoint_fingerprint_before_and_after_required": True,
                "checkpoint_must_remain_unchanged": True,
                "fork_required": True,
                "sv_sessions_mutable_only_during_suffix_replay": True,
                "sv_sessions_snapshotted_and_refrozen_after_replay": True,
                "returned_token_ids_equal_omitted_history": True,
                "returned_token_count_equals_omitted_history": True,
                "returned_input_digest_equals_omitted_history_sha256": True,
            },
            "workload": {
                "history_checkpoints": EXPECTED_HISTORIES,
                "stored_suffix_replays": EXPECTED_HISTORIES,
                "incremental_suffix_token_steps": suffix_steps,
                "minimum_suffix_token_steps": min(
                    len(plan.suffix_token_ids) for plan in plans
                ),
                "maximum_suffix_token_steps": max(
                    len(plan.suffix_token_ids) for plan in plans
                ),
                "owned_token_steps": 0,
                "contaminated_suffix_histories": contaminated,
                "clean_suffix_histories": EXPECTED_HISTORIES - contaminated,
            },
            "plans": list(rows),
            "plans_sha256": payload_sha256(rows),
        }
    )
    validate_replay_lock(lock)
    return lock


freeze_replay_lock = build_replay_lock


def validate_replay_lock(lock: Mapping[str, Any]) -> None:
    _validate_seal(lock, name="replay lock")
    assert_source_free_lock(lock)
    rows = lock.get("plans")
    workload = lock.get("workload") or {}
    semantics = lock.get("replay_semantics") or {}
    bandwidth = lock.get("bandwidth_schedule") or {}
    if (
        lock.get("schema") != SCHEMA
        or lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("status") != STATUS
        or lock.get("model_or_api_calls_made") != 0
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_HISTORIES
        or lock.get("plans_sha256") != payload_sha256(rows)
    ):
        raise ReplayPlanError("replay lock schema, status, or plan count differs")
    for index, row in enumerate(rows):
        geometry = row.get("geometry") if isinstance(row, Mapping) else None
        hashes = row.get("hashes") if isinstance(row, Mapping) else None
        if (
            row.get("history_index") != index
            or row.get("cluster_index") != index // HISTORIES_PER_CLUSTER
            or row.get("variant_index") != index % HISTORIES_PER_CLUSTER
            or row.get("suffix_scan_stratum")
            not in {"clean", "contaminated"}
            or not isinstance(geometry, Mapping)
            or not isinstance(hashes, Mapping)
            or any(not _is_sha256(value) for value in hashes.values())
            or not _is_sha256(row.get("plan_sha256"))
            or not _is_sha256(row.get("history_binding_sha256"))
        ):
            raise ReplayPlanError("anonymous replay plan geometry differs")
        counts = [
            geometry.get("original_token_count"),
            geometry.get("prefix_token_count"),
            geometry.get("owned_token_count"),
            geometry.get("stored_suffix_token_count"),
            geometry.get("omitted_history_token_count"),
        ]
        if (
            any(type(value) is not int or value < 1 for value in counts)
            or counts[0] != counts[1] + counts[2] + counts[3]
            or counts[4] != counts[1] + counts[3]
            or any(
                geometry.get(key) is not True
                for key in (
                    "owned_range_complete_and_contiguous",
                    "original_equals_prefix_owned_suffix",
                    "omitted_equals_prefix_suffix",
                    "stored_suffix_sliced_from_original",
                )
            )
        ):
            raise ReplayPlanError("replay plan token counts differ")
    if (
        workload.get("history_checkpoints") != EXPECTED_HISTORIES
        or workload.get("stored_suffix_replays") != EXPECTED_HISTORIES
        or workload.get("owned_token_steps") != 0
        or semantics.get("owned_ids_passed_to_replay_runtime") is not False
        or semantics.get("dependency_aware_never_ingested_history") is not False
        or semantics.get("incremental_tokens_per_model_step") != 1
        or bandwidth.get("source") != "prefix_checkpoint"
        or bandwidth.get("silent_kpar_override_allowed") is not False
    ):
        raise ReplayPlanError("replay workload or bandwidth contract differs")


def write_replay_lock(path: str | Path, lock: Mapping[str, Any]) -> None:
    """Write a reviewed lock once; overwriting an existing artifact is forbidden."""

    validate_replay_lock(lock)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        lock,
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
        allow_nan=False,
    )
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.write("\n")


__all__ = [
    "DEFAULT_REPLAY_LOCK_PATH",
    "EXPECTED_CLUSTERS",
    "EXPECTED_HISTORIES",
    "ReplayPlanError",
    "artifact_binding",
    "build_replay_lock",
    "freeze_replay_lock",
    "plan_all_histories",
    "plan_history_replay",
    "token_ids_sha256",
    "validate_replay_lock",
    "write_replay_lock",
]

"""Complete the externally aborted geometry-method run with one record.

The original geometry-corrected LongMemEval method process was externally
aborted after it had written measurements for the first 15 authorized records.
The report proves that no terminal record-16 checkpoint was written; it does
not prove that record 16 was never entered before the external abort.  This
module freezes that exact partial report into a source-free, committed
completion lock, rehydrates and validates the full frozen cohort, executes only
the one authorized completion attempt, and merges the lock-sealed first 15 rows
with the newly sealed row.  A sealed canonical row is recoverable without
loading or scoring the model if publication of the final report is interrupted.

Real execution requires the completion lock at its canonical path, byte-for-
byte equal to the copy committed at ``HEAD``, plus the literal boolean ``True``
acknowledgement.  Outputs use exact canonical paths and are atomically
published without replacement.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_geometry_methods as methods


SCHEMA = "gemma-sv-longmemeval-chat-geometry-method-completion-lock-v1"
SCHEMA_VERSION = 1
LOCK_STATUS = "frozen-after-external-abort-before-one-record-completion"
ROW_ARTIFACT_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-method-completion-row-v1"
)
ATTEMPT_MARKER_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-method-completion-attempt-v1"
)
COMPLETION_METADATA_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-method-external-abort-completion-v1"
)

EXPECTED_RECORDS = methods.EXPECTED_RECORDS
SEALED_PARTIAL_RECORDS = EXPECTED_RECORDS - 1
EXPECTED_REMAINING_RECORD_ID = (
    "longmemeval-constrained-chat-v1-f05a4da53bcfcf0faf9be30a"
)
EXPECTED_METHOD_IMPLEMENTATION_FILES = 41

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
OUTPUTS = WORKSPACE / "outputs" / "gemma_sv_rag"

DEFAULT_PARTIAL_REPORT = (
    OUTPUTS / "longmemeval_chat_geometry_methods_v1.json"
)
DEFAULT_COMPLETION_LOCK = (
    BENCHMARKS / "longmemeval_chat_geometry_methods_completion_v1.json"
)
DEFAULT_COMPLETION_ROW_OUTPUT = (
    OUTPUTS / "longmemeval_chat_geometry_methods_completion_row_v1.json"
)
DEFAULT_ATTEMPT_CLAIM_DIR = (
    OUTPUTS / "longmemeval_chat_geometry_methods_completion_attempt_v1.claim"
)
DEFAULT_ATTEMPT_LOCK = (
    OUTPUTS / ".longmemeval_chat_geometry_methods_completion_attempt_v1.lock"
)
ATTEMPT_MARKER_FILENAME = "marker.json"
CANONICAL_ATTEMPT_CLAIM_WORKSPACE_PATH = (
    "outputs/gemma_sv_rag/"
    "longmemeval_chat_geometry_methods_completion_attempt_v1.claim"
)
DEFAULT_FINAL_OUTPUT = (
    OUTPUTS / "longmemeval_chat_geometry_methods_completed_v1.json"
)

EXTERNAL_ABORT_DISCLOSURE = {
    "external_abort_after_terminal_record_checkpoints": SEALED_PARTIAL_RECORDS,
    "external_abort_before_terminal_record16_checkpoint": True,
    "partial_report_record16_checkpoint_status": "not_attempted",
    "record16_may_have_started_without_a_terminal_checkpoint": True,
    "partial_report_is_not_a_complete_matrix": True,
    "completion_executes_only_the_predeclared_remaining_record": True,
    "completion_attempt_consumed_by_any_terminal_checkpoint": True,
    "terminal_evaluator_exception_sealed_as_generic_failure": True,
    "attempt_claim_directory_created_exclusively_before_model_load": True,
    "claim_without_terminal_row_recovers_as_generic_failure": True,
    "first15_measurements_trusted_only_through_committed_completion_lock": True,
    "record_selection_or_replacement_after_method_outputs": False,
    "records_reexecuted_from_partial_report": 0,
}

_COMPLETION_IMPLEMENTATION_PATHS = {
    "complete_longmemeval_chat_geometry_methods.py": Path(__file__).resolve(),
}
_PROTECTED_INPUTS = frozenset(
    {
        DEFAULT_PARTIAL_REPORT,
        methods.DEFAULT_MANIFEST,
        methods.DEFAULT_POLICY_LOCK,
        methods.DEFAULT_CENSUS,
        methods.DEFAULT_CORE_MANIFEST,
        methods.DEFAULT_ADMISSION_REPORT,
        methods.DEFAULT_METHOD_LOCK,
        *_COMPLETION_IMPLEMENTATION_PATHS.values(),
        *methods._PROTECTED_INPUTS,
    }
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    rendered = str(value)
    return (
        len(rendered) == 64
        and rendered == rendered.lower()
        and all(character in "0123456789abcdef" for character in rendered)
    )


def _is_error_type(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and value.isidentifier()
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _load_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _assert_source_free_and_finite(
    payload: Mapping[str, Any],
    *,
    path: str,
) -> None:
    methods._assert_source_free(payload)
    methods.hardened._assert_finite_json(payload, path=path)


def _completion_implementation_fingerprints() -> dict[str, str]:
    values = {
        name: _sha256_file(path)
        for name, path in sorted(_COMPLETION_IMPLEMENTATION_PATHS.items())
    }
    values["contract_sha256"] = _payload_sha256(values)
    return values


def _method_implementation_contract(
    method_lock: Mapping[str, Any],
) -> dict[str, Any]:
    implementation_value = method_lock.get("implementation")
    if not isinstance(implementation_value, Mapping):
        raise ValueError("method lock implementation contract is missing")
    implementation = dict(implementation_value)
    file_hashes = {
        str(name): value
        for name, value in implementation.items()
        if name != "contract_sha256"
    }
    contract_sha256 = implementation.get("contract_sha256")
    if (
        len(file_hashes) != EXPECTED_METHOD_IMPLEMENTATION_FILES
        or set(file_hashes) != set(methods._IMPLEMENTATION_PATHS)
        or not all(_is_sha256(value) for value in file_hashes.values())
        or contract_sha256 != _payload_sha256(file_hashes)
        or method_lock.get("implementation_contract_sha256")
        != contract_sha256
        or implementation != methods._implementation_fingerprints()
    ):
        raise ValueError("method lock 41-file implementation contract drifted")

    artifacts_value = method_lock.get("artifacts")
    if not isinstance(artifacts_value, Mapping):
        raise ValueError("method lock artifact contract is missing")
    artifacts = dict(artifacts_value)
    if not artifacts or not all(_is_sha256(value) for value in artifacts.values()):
        raise ValueError("method lock artifact hashes are invalid")

    authorization = method_lock.get("authorization") or {}
    runtime_value = authorization.get("runtime")
    if not isinstance(runtime_value, Mapping):
        raise ValueError("method lock runtime contract is missing")
    runtime = dict(runtime_value)
    expected_runtime = methods.admission.runtime_contract(
        device=str(runtime.get("device") or "")
    )
    if runtime != expected_runtime:
        raise ValueError("method lock runtime contract drifted")
    if list(authorization.get("condition_ids") or ()) != list(
        methods.CONDITION_IDS
    ):
        raise ValueError("method lock condition matrix drifted")

    admission_records = list(authorization.get("admission_records") or ())
    record_ids = [
        str(binding.get("record_id") or "")
        for binding in admission_records
        if isinstance(binding, Mapping)
    ]
    if (
        len(admission_records) != EXPECTED_RECORDS
        or len(record_ids) != EXPECTED_RECORDS
        or len(set(record_ids)) != EXPECTED_RECORDS
        or record_ids[-1] != EXPECTED_REMAINING_RECORD_ID
    ):
        raise ValueError("method lock ordered all16 authorization drifted")

    return {
        "artifacts": copy.deepcopy(artifacts),
        "artifacts_sha256": _payload_sha256(artifacts),
        "implementation": copy.deepcopy(implementation),
        "implementation_contract_sha256": str(contract_sha256),
        "implementation_files": len(file_hashes),
        "runtime": copy.deepcopy(runtime),
        "runtime_sha256": _payload_sha256(runtime),
        "condition_ids": list(methods.CONDITION_IDS),
        "ordered_record_ids": record_ids,
    }


def _validate_condition(
    condition: Any,
    *,
    condition_id: str,
) -> None:
    if not isinstance(condition, Mapping):
        raise ValueError("partial method condition is not an object")
    status = str(condition.get("status") or "")
    if status == "failed":
        if (
            condition.get("condition_id") != condition_id
            or condition.get("error_message_redacted") is not True
            or not _is_error_type(condition.get("error_type"))
            or not _is_sha256(condition.get("error_message_sha256"))
        ):
            raise ValueError("partial failed condition is not hardened")
        return
    if status != "completed":
        raise ValueError("partial method condition has an unknown status")
    if (
        condition.get("condition_id") != condition_id
        or not isinstance(condition.get("timing"), Mapping)
        or not isinstance(condition.get("storage"), Mapping)
        or len(
            (condition.get("deleted_target_quality") or {}).get("probes") or ()
        )
        != 1
        or not isinstance(condition.get("retained_quality"), Mapping)
    ):
        raise ValueError("partial completed condition metrics are incomplete")


_UNSEALED_MEASURED_ROW_KEYS = frozenset(
    {
        "admission_binding",
        "conditions",
        "geometry_binding",
        "method_failures",
        "predeclared_joint_admitted",
        "probes",
        "record_id",
        "record_seed",
        "shared_original_prefill",
        "solver_certificate",
        "source_fixed_c_feasible",
        "source_state_immutability",
        "status",
    }
)
_SEALED_COMPLETION_ROW_KEYS = frozenset(
    {
        *_UNSEALED_MEASURED_ROW_KEYS,
        "completion_lock_integrity_sha256",
        "execution_contract_sha256",
        "row_integrity",
    }
)
_GENERIC_FAILED_ROW_KEYS = frozenset(
    {
        "record_id",
        "status",
        "error_type",
        "error_message_sha256",
        "error_message_redacted",
    }
)
_SEALED_GENERIC_FAILED_ROW_KEYS = frozenset(
    {
        *_GENERIC_FAILED_ROW_KEYS,
        "completion_lock_integrity_sha256",
        "execution_contract_sha256",
        "row_integrity",
    }
)
_FAILED_CONDITION_KEYS = frozenset(
    {
        "condition_id",
        "error_message_redacted",
        "error_message_sha256",
        "error_type",
        "status",
    }
)
_COMPLETED_CONDITION_KEYS = frozenset(
    {
        "condition_id",
        "deleted_target_quality",
        "reference",
        "retained_quality",
        "semantics",
        "status",
        "storage",
        "timing",
        "tokenized_state",
        "update_diagnostics",
    }
)


def _all_numeric_mapping_keys(value: Mapping[str, Any]) -> bool:
    return bool(value) and all(str(key).isdigit() for key in value)


def _validate_recursive_schema(
    value: Any,
    templates: Sequence[Any],
    *,
    path: str,
    key: str = "",
) -> None:
    """Reject every recursive key and string form absent from sealed rows."""

    if isinstance(value, Mapping):
        mapping_templates = [
            template for template in templates if isinstance(template, Mapping)
        ]
        if not mapping_templates:
            raise ValueError(f"{path} has no authorized object schema")
        if all(_all_numeric_mapping_keys(template) for template in mapping_templates):
            if not _all_numeric_mapping_keys(value):
                raise ValueError(f"{path} numeric mapping keys differ")
            child_templates = [
                child
                for template in mapping_templates
                for child in template.values()
            ]
            for child_key, child in value.items():
                _validate_recursive_schema(
                    child,
                    child_templates,
                    path=f"{path}.{child_key}",
                    key=str(child_key),
                )
            return
        allowed_keysets = {
            frozenset(str(item) for item in template)
            for template in mapping_templates
        }
        observed_keys = frozenset(str(item) for item in value)
        if observed_keys not in allowed_keysets:
            raise ValueError(f"{path} recursive key schema differs")
        for child_key, child in value.items():
            child_templates = [
                template[child_key]
                for template in mapping_templates
                if child_key in template
            ]
            _validate_recursive_schema(
                child,
                child_templates,
                path=f"{path}.{child_key}",
                key=str(child_key),
            )
        return
    if isinstance(value, list):
        list_templates = [
            template for template in templates if isinstance(template, list)
        ]
        if not list_templates:
            raise ValueError(f"{path} has no authorized list schema")
        item_templates = [
            item for template in list_templates for item in template
        ]
        if value and not item_templates:
            raise ValueError(f"{path} unexpectedly contains list items")
        for index, item in enumerate(value):
            _validate_recursive_schema(
                item,
                item_templates,
                path=f"{path}[{index}]",
                key=key,
            )
        return
    if isinstance(value, bool):
        allowed = {
            template for template in templates if isinstance(template, bool)
        }
        if value not in allowed:
            raise ValueError(f"{path} boolean value differs")
        return
    if value is None:
        if not any(template is None for template in templates):
            raise ValueError(f"{path} null value differs")
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not any(
            isinstance(template, (int, float))
            and not isinstance(template, bool)
            for template in templates
        ):
            raise ValueError(f"{path} numeric type differs")
        return
    if isinstance(value, str):
        string_templates = [
            template for template in templates if isinstance(template, str)
        ]
        if not string_templates:
            raise ValueError(f"{path} string type differs")
        if all(_is_sha256(template) for template in string_templates):
            if not _is_sha256(value):
                raise ValueError(f"{path} digest format differs")
            return
        if value not in set(string_templates):
            raise ValueError(f"{path} free-text value is not authorized")
        return
    raise ValueError(f"{path} contains unsupported JSON data")


def _expected_row_probes(public_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "probe_id": probe["probe_id"],
            "probe_kind": probe["kind"],
            "target_token_count": int(probe["target_token_count"]),
            "target_token_ids_sha256": probe["target_token_ids_sha256"],
        }
        for probe in public_record["probes"]
    ]


def _number(value: Any, *, path: str, minimum: float | None = None) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        raise ValueError(f"{path} numeric value is invalid")
    return float(value)


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} integer value is invalid")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_timing_summary(
    summary: Any,
    *,
    repeats: int,
    path: str,
) -> None:
    if not isinstance(summary, Mapping):
        raise ValueError(f"{path} is not an object")
    base_keys = {"median", "p95", "minimum", "maximum", "samples"}
    optional = {"not_applicable", "reason"}
    shared = {"shared_timing_group"}
    if set(summary) not in {
        frozenset(base_keys),
        frozenset(base_keys | optional),
        frozenset(base_keys | shared),
    }:
        raise ValueError(f"{path} key schema differs: {sorted(summary)}")
    samples = summary["samples"]
    if not isinstance(samples, list) or len(samples) != repeats:
        raise ValueError(f"{path} sample count differs")
    values = [
        _number(value, path=f"{path}.samples[{index}]", minimum=0.0)
        for index, value in enumerate(samples)
    ]
    expected = {
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }
    for key, expected_value in expected.items():
        observed = _number(summary[key], path=f"{path}.{key}", minimum=0.0)
        if not _close(observed, expected_value):
            raise ValueError(f"{path} aggregate {key} is inconsistent")
    if optional <= set(summary) and (
        summary["not_applicable"] is not True
        or not isinstance(summary["reason"], str)
        or not summary["reason"]
    ):
        raise ValueError(f"{path} not-applicable annotation differs")
    if shared <= set(summary) and (
        summary["shared_timing_group"]
        != "exact_decrement_and_fixed_c_refit"
    ):
        raise ValueError(f"{path} shared timing annotation differs")


def _validate_timing(
    timing: Any,
    *,
    completion_lock: Mapping[str, Any],
    condition_id: str,
) -> None:
    if not isinstance(timing, Mapping):
        raise ValueError("completion condition timing is not an object")
    base_keys = {
        "clock",
        "synchronized_device",
        "warmup",
        "repeats",
        "query_scope",
        "update_seconds",
        "query_seconds",
        "end_to_end_seconds",
    }
    alias_keys = {"zero_cost_alias", "alias_of"}
    expected_keys = (
        base_keys | alias_keys
        if condition_id == methods.TOKEN_ROW_DIAGNOSTIC_ID
        else base_keys
    )
    if set(timing) != expected_keys:
        raise ValueError("completion condition timing key schema differs")
    repeats = completion_lock["authorization"]["repeats"]
    if (
        timing["clock"] != "time.perf_counter"
        or timing["synchronized_device"]
        != completion_lock["authorization"]["runtime"]["device"]
        or timing["warmup"] != completion_lock["authorization"]["warmup"]
        or timing["repeats"] != repeats
        or timing["query_scope"]
        != (
            "full target_current and retained sequences plus first-token "
            "rank and full-vocabulary KL"
        )
    ):
        raise ValueError("completion condition timing contract differs")
    for scope in ("update_seconds", "query_seconds", "end_to_end_seconds"):
        _validate_timing_summary(
            timing[scope],
            repeats=repeats,
            path=f"completion_row.timing.{scope}",
        )
    for update, query, total in zip(
        timing["update_seconds"]["samples"],
        timing["query_seconds"]["samples"],
        timing["end_to_end_seconds"]["samples"],
    ):
        if float(total) + 1e-12 < max(float(update), float(query)):
            raise ValueError("completion end-to-end timing is inconsistent")
    if condition_id == methods.TOKEN_ROW_DIAGNOSTIC_ID:
        if (
            timing["zero_cost_alias"] is not True
            or timing["alias_of"] != methods.FRESH_RAW_OMISSION_ID
            or any(
                any(float(value) != 0.0 for value in timing[scope]["samples"])
                for scope in (
                    "update_seconds",
                    "query_seconds",
                    "end_to_end_seconds",
                )
            )
        ):
            raise ValueError("completion token alias timing differs")


def _validate_score(score: Any, *, expected_tokens: int, path: str) -> None:
    expected_keys = {
        "target_token_count",
        "total_log_probability",
        "mean_log_probability",
        "geometric_mean_probability",
        "first_target_token_probability",
        "first_target_token_rank",
    }
    if not isinstance(score, Mapping) or set(score) != expected_keys:
        raise ValueError(f"{path} score key schema differs")
    if score["target_token_count"] != expected_tokens:
        raise ValueError(f"{path} target token count differs")
    total = _number(score["total_log_probability"], path=f"{path}.total")
    mean = _number(score["mean_log_probability"], path=f"{path}.mean")
    geometric = _number(
        score["geometric_mean_probability"],
        path=f"{path}.geometric_probability",
        minimum=0.0,
    )
    first = _number(
        score["first_target_token_probability"],
        path=f"{path}.first_probability",
        minimum=0.0,
    )
    if (
        total > 0.0
        or mean > 0.0
        or geometric > 1.0
        or first > 1.0
        or not _close(total, mean * expected_tokens)
        or not _close(geometric, math.exp(mean))
    ):
        raise ValueError(f"{path} probability metrics are inconsistent")
    _integer(
        score["first_target_token_rank"],
        path=f"{path}.first_target_token_rank",
        minimum=1,
    )


def _validate_probe_metrics(
    value: Any,
    *,
    expected_probe: Mapping[str, Any],
    path: str,
) -> None:
    expected_keys = {
        "probe_id",
        "score",
        "suppression_vs_present_nats",
        "mean_log_probability_drift_from_raw_repack_nats",
        "full_vocabulary_kl_raw_repack_to_method_nats",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{path} probe metric key schema differs")
    if value["probe_id"] != expected_probe["probe_id"]:
        raise ValueError(f"{path} probe identity differs")
    _validate_score(
        value["score"],
        expected_tokens=expected_probe["target_token_count"],
        path=f"{path}.score",
    )
    _number(
        value["suppression_vs_present_nats"],
        path=f"{path}.suppression",
    )
    _number(
        value["mean_log_probability_drift_from_raw_repack_nats"],
        path=f"{path}.drift",
    )
    _number(
        value["full_vocabulary_kl_raw_repack_to_method_nats"],
        path=f"{path}.kl",
        minimum=0.0,
    )


def _validate_nonnegative_tree(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_nonnegative_tree(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_nonnegative_tree(item, path=f"{path}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _number(value, path=path, minimum=0.0)


def _validate_nonnegative_counters(value: Any, *, path: str) -> None:
    markers = (
        "count",
        "rows",
        "bytes",
        "positions",
        "fallbacks",
        "solves",
        "gates",
        "deviation",
        "tolerance",
        "box_c",
        "capacity",
        "fraction",
        "boundary",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and any(marker in str(key).casefold() for marker in markers)
            ):
                _number(item, path=child_path, minimum=0.0)
            _validate_nonnegative_counters(item, path=child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_nonnegative_counters(item, path=f"{path}[{index}]")


def _validate_storage(storage: Any, *, condition_id: str) -> None:
    report_keys = {
        "deduplicated_tensor_storage_bytes",
        "torch_storage_bytes",
        "numpy_storage_bytes",
        "unique_tensor_storages",
    }
    expected_keys = {
        "state",
        "native_kv",
        "sv_sessions",
        "joint_with_original",
        "incremental_tensor_storage_bytes",
        "prompt_token_count",
        "prompt_token_id_bytes_int64",
    }
    if condition_id == methods.TOKEN_ROW_DIAGNOSTIC_ID:
        expected_keys |= {
            "aliased_reference_incremental_tensor_storage_bytes",
            "zero_cost_alias",
            "alias_of",
        }
    if not isinstance(storage, Mapping) or set(storage) != expected_keys:
        raise ValueError("completion condition storage key schema differs")
    for key in ("state", "native_kv", "sv_sessions", "joint_with_original"):
        if not isinstance(storage[key], Mapping) or set(storage[key]) != report_keys:
            raise ValueError("completion condition tensor storage schema differs")
    _validate_nonnegative_tree(storage, path="completion_row.storage")
    if condition_id == methods.TOKEN_ROW_DIAGNOSTIC_ID and (
        storage["zero_cost_alias"] is not True
        or storage["alias_of"] != methods.FRESH_RAW_OMISSION_ID
        or storage["incremental_tensor_storage_bytes"] != 0
    ):
        raise ValueError("completion token alias storage differs")


def _validate_semantics(semantics: Any, *, condition_id: str) -> None:
    exact_values: dict[str, Mapping[str, Any]] = {
        methods.PRESENT_ID: {"reference": True, "owned_round_present": True},
        methods.FRESH_RAW_OMISSION_ID: {
            "behavioral_reference": True,
            "executed_method": "fresh_raw_omission_full_repack",
            "fresh_prefill": True,
            "raw_owned_round_omitted": True,
            "freshly_retokenized": True,
            "suffix_recomputed": True,
        },
        methods.TOKEN_ROW_DIAGNOSTIC_ID: {
            "diagnostic_only": True,
            "executed_method": "zero_cost_alias",
            "zero_cost_alias": True,
            "alias_of": methods.FRESH_RAW_OMISSION_ID,
            "additional_model_execution": False,
            "registered_turn_boundary_contract": True,
            "identical_token_ids": True,
        },
        methods.DECAY_ID: {
            "decay_factor": methods.hardened.DECAY_FACTOR,
            "positions_remain_resident": True,
            "solver_refit": False,
            "suffix_recomputed": False,
        },
        methods.CACHE_DELETE_SHIFT_ID: {
            "diagnostic_only": True,
            "physically_deletes_matching_cache_rows": True,
            "solver_refit": False,
            "suffix_recomputed": False,
            "rope_keys_rerotated": False,
        },
        methods.FIXED_C_REFIT_ID: {
            "executed_method": "float64_fixed_c_retained_key_refit",
            "gate_solver": "float64 retained-key from-scratch refit",
            "fixed_C": True,
            "query_independent": True,
            "retained_keys_only": True,
            "conditional_on_contextualized_retained_keys": True,
        },
        methods.PROMPT_SUPPRESSION_ID: {
            "prompt_only_behavioral_control": True,
            "persistent_state_deleted": False,
            "base_state_unchanged": True,
            "valid_registered_gemma_chat": True,
            "instruction_target_free": True,
        },
    }
    if condition_id in exact_values:
        if semantics != exact_values[condition_id]:
            raise ValueError("completion condition semantics differ")
        return
    if condition_id == methods.FP32_PROXY_ID:
        expected_keys = {
            "solver",
            "executed_method",
            "model_forward_precision",
            "query_independent",
            "full_repack_fallback",
        }
        if (
            not isinstance(semantics, Mapping)
            or set(semantics) != expected_keys
            or semantics["solver"] != "projected FP32 FISTA masked refit"
            or semantics["executed_method"] != "incremental_fp32_masked_refit"
            or semantics["model_forward_precision"] != "float32"
            or semantics["query_independent"] is not True
            or semantics["full_repack_fallback"] is not False
        ):
            raise ValueError("completion FP32 semantics differ")
        return
    if condition_id == methods.EXACT_DECREMENT_ID:
        expected_keys = {
            "executed_method",
            "gate_solver",
            "fixed_C",
            "query_independent",
            "full_repack_fallback",
            "fixed_c_refit_fallback",
            "incremental_exact",
            "decrement_fallbacks",
            "certificate_reference",
        }
        if not isinstance(semantics, Mapping) or set(semantics) != expected_keys:
            raise ValueError("completion exact semantics schema differs")
        fallbacks = _integer(
            semantics["decrement_fallbacks"],
            path="completion_row.exact.decrement_fallbacks",
        )
        used_fallback = fallbacks > 0
        if (
            semantics["executed_method"]
            != (
                "fixed_c_refit_fallback"
                if used_fallback
                else "incremental_float64_fixed_c_decrement"
            )
            or semantics["gate_solver"]
            != "float64 Cauwenberghs-Poggio decrement"
            or semantics["fixed_C"] is not True
            or semantics["query_independent"] is not True
            or semantics["full_repack_fallback"] is not False
            or semantics["fixed_c_refit_fallback"] is not used_fallback
            or semantics["incremental_exact"] is not (not used_fallback)
            or semantics["certificate_reference"] != methods.FIXED_C_REFIT_ID
        ):
            raise ValueError("completion exact semantics differ")
        return
    raise ValueError("completion condition semantics are unknown")


def _validate_completed_condition(
    condition: Mapping[str, Any],
    *,
    condition_id: str,
    completion_lock: Mapping[str, Any],
    expected_probes: Sequence[Mapping[str, Any]],
    expected_digest: str,
    templates: Sequence[Mapping[str, Any]],
) -> None:
    if set(condition) != set(_COMPLETED_CONDITION_KEYS):
        raise ValueError("completion completed condition key schema differs")
    tokenized = condition["tokenized_state"]
    if not isinstance(tokenized, Mapping) or set(tokenized) != {
        "prefill_input_digest",
        "token_count",
        "added_query_instruction_tokens",
    }:
        raise ValueError("completion tokenized-state schema differs")
    if tokenized["prefill_input_digest"] != expected_digest:
        raise ValueError("completion condition context digest differs")
    _integer(tokenized["token_count"], path="completion_row.token_count")
    _integer(
        tokenized["added_query_instruction_tokens"],
        path="completion_row.added_query_instruction_tokens",
    )

    deleted = condition["deleted_target_quality"]
    deleted_keys = {
        "probes",
        "mean_log_probability",
        "mean_suppression_vs_present_nats",
        "mean_drift_from_raw_repack_nats",
        "mean_full_vocabulary_kl_to_raw_repack_nats",
        "max_full_vocabulary_kl_to_raw_repack_nats",
    }
    if not isinstance(deleted, Mapping) or set(deleted) != deleted_keys:
        raise ValueError("completion deleted-quality schema differs")
    probes = deleted["probes"]
    if not isinstance(probes, list) or len(probes) != 1:
        raise ValueError("completion deleted probe count differs")
    _validate_probe_metrics(
        probes[0],
        expected_probe=expected_probes[0],
        path="completion_row.deleted_probe",
    )
    retained = condition["retained_quality"]
    _validate_probe_metrics(
        retained,
        expected_probe=expected_probes[1],
        path="completion_row.retained_probe",
    )
    deleted_probe = probes[0]
    aggregate_pairs = {
        "mean_log_probability": deleted_probe["score"]["mean_log_probability"],
        "mean_suppression_vs_present_nats": deleted_probe[
            "suppression_vs_present_nats"
        ],
        "mean_drift_from_raw_repack_nats": deleted_probe[
            "mean_log_probability_drift_from_raw_repack_nats"
        ],
        "mean_full_vocabulary_kl_to_raw_repack_nats": deleted_probe[
            "full_vocabulary_kl_raw_repack_to_method_nats"
        ],
        "max_full_vocabulary_kl_to_raw_repack_nats": deleted_probe[
            "full_vocabulary_kl_raw_repack_to_method_nats"
        ],
    }
    for key, expected in aggregate_pairs.items():
        minimum = 0.0 if "kl_" in key else None
        observed = _number(
            deleted[key],
            path=f"completion_row.deleted_quality.{key}",
            minimum=minimum,
        )
        if not _close(observed, float(expected)):
            raise ValueError("completion deleted-quality aggregate differs")

    reference = condition["reference"]
    if reference != {
        "kind": "freshly_retokenized_raw_exchange_omitted_repack",
        "kl_direction": "KL(raw_repack || method)",
        "distribution_scope": "full vocabulary at first target token",
        "distinct_from_solver_certificate": True,
    }:
        raise ValueError("completion condition reference differs")
    _validate_timing(
        condition["timing"],
        completion_lock=completion_lock,
        condition_id=condition_id,
    )
    _validate_storage(condition["storage"], condition_id=condition_id)
    _validate_semantics(condition["semantics"], condition_id=condition_id)
    if templates:
        _validate_recursive_schema(
            condition["update_diagnostics"],
            [template["update_diagnostics"] for template in templates],
            path=f"completion_row.conditions.{condition_id}.update_diagnostics",
            key="update_diagnostics",
        )
    elif condition_id == methods.CACHE_DELETE_SHIFT_ID:
        diagnostics = condition["update_diagnostics"]
        expected_keys = {
            "method",
            "old_token_count",
            "new_token_count",
            "deleted_positions",
            "sv_layers",
            "native_cache",
            "solver_refit",
            "suffix_recomputed",
            "rope_keys_rerotated",
            "diagnostic_only",
            "actual_prefill_input_digest",
            "logical_edited_input_digest",
            "raw_omission_token_ids_sha256",
        }
        if not isinstance(diagnostics, Mapping) or set(diagnostics) != expected_keys:
            raise ValueError("completion cache diagnostics schema differs")
        if (
            diagnostics["method"] != "cache_only_delete_and_shift"
            or diagnostics["solver_refit"] is not False
            or diagnostics["suffix_recomputed"] is not False
            or diagnostics["rope_keys_rerotated"] is not False
            or diagnostics["diagnostic_only"] is not True
        ):
            raise ValueError("completion cache diagnostics semantics differ")
        old_tokens = _integer(
            diagnostics["old_token_count"],
            path="completion_row.cache.old_token_count",
        )
        new_tokens = _integer(
            diagnostics["new_token_count"],
            path="completion_row.cache.new_token_count",
        )
        deleted_positions = _integer(
            diagnostics["deleted_positions"],
            path="completion_row.cache.deleted_positions",
        )
        if old_tokens - new_tokens != deleted_positions:
            raise ValueError("completion cache token accounting differs")
        sv_layers = diagnostics["sv_layers"]
        if not isinstance(sv_layers, Mapping) or any(
            not str(layer_id).isdigit()
            or not isinstance(detail, Mapping)
            or set(detail)
            != {
                "old_rows",
                "new_rows",
                "removed_rows",
                "old_frozen_boundary",
                "new_frozen_boundary",
                "gate_rows_removed",
                "solver_refit",
            }
            or detail["solver_refit"] is not False
            for layer_id, detail in sv_layers.items()
        ):
            raise ValueError("completion cache SV-layer schema differs")
        native = diagnostics["native_cache"]
        native_base = {
            "cache_type",
            "layout",
            "layers_inspected",
            "layers_with_resident_rows",
            "resident_rows_removed_across_layers",
            "nonresident_requests_across_layers",
        }
        if (
            not isinstance(native, Mapping)
            or set(native) not in {frozenset(native_base), frozenset(native_base | {"layers"})}
            or native["layout"]
            not in {
                "unsupported",
                "transformers_cache_layers",
                "legacy_layer_sequence",
            }
        ):
            raise ValueError("completion native-cache schema differs")
        if "layers" in native:
            false_layer = {
                "initialized",
                "old_resident_rows",
                "new_resident_rows",
                "resident_rows_removed",
            }
            true_layer = {
                "initialized",
                "cache_layer_type",
                "is_static",
                "logical_rows_before",
                "logical_rows_after",
                "resident_start",
                "requested_positions_before_logical_end",
                "resident_requested_positions",
                "nonresident_requested_positions",
                "old_resident_rows",
                "new_resident_rows",
                "resident_rows_removed",
            }
            if not isinstance(native["layers"], list) or any(
                not isinstance(layer, Mapping)
                or set(layer) not in {frozenset(false_layer), frozenset(true_layer)}
                for layer in native["layers"]
            ):
                raise ValueError("completion native-cache layer schema differs")
    else:
        raise ValueError("completion condition diagnostics schema is unavailable")
    _validate_nonnegative_counters(
        condition["update_diagnostics"],
        path=f"completion_row.conditions.{condition_id}.update_diagnostics",
    )


def _validate_certificate(
    certificate: Any,
    *,
    conditions: Mapping[str, Any],
    expected_probes: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(certificate, Mapping):
        raise ValueError("completion certificate is not an object")
    status = certificate.get("status")
    if status == "failed":
        if set(certificate) != {
            "status",
            "error_type",
            "error_message_sha256",
            "error_message_redacted",
        }:
            raise ValueError("completion failed certificate schema differs")
        if (
            certificate["error_message_redacted"] is not True
            or not _is_error_type(certificate["error_type"])
            or not _is_sha256(certificate["error_message_sha256"])
        ):
            raise ValueError("completion failed certificate is not hardened")
        for condition_id in (
            methods.EXACT_DECREMENT_ID,
            methods.FIXED_C_REFIT_ID,
        ):
            condition = conditions[condition_id]
            if (
                condition.get("status") != "failed"
                or condition.get("error_type") != certificate["error_type"]
                or condition.get("error_message_sha256")
                != certificate["error_message_sha256"]
            ):
                raise ValueError("completion certificate failure pair differs")
        return
    expected_keys = {
        "status",
        "claim",
        "direction",
        "full_vocabulary",
        "distribution_scope",
        "distinct_from_behavioral_kl_to_raw_repack",
        "probe_output_kls",
        "mean_output_kl_nats",
        "max_output_kl_nats",
        "solver_diagnostics",
    }
    if status != "completed" or set(certificate) != expected_keys:
        raise ValueError("completion certificate schema differs")
    if any(
        conditions[condition_id].get("status") != "completed"
        for condition_id in (
            methods.EXACT_DECREMENT_ID,
            methods.FIXED_C_REFIT_ID,
        )
    ):
        raise ValueError("completed certificate lacks its method pair")
    rows = certificate["probe_output_kls"]
    if not isinstance(rows, list) or len(rows) != len(expected_probes):
        raise ValueError("completion certificate probe count differs")
    values = []
    for index, (row, probe) in enumerate(zip(rows, expected_probes)):
        if not isinstance(row, Mapping) or set(row) != {
            "probe_id",
            "probe_kind",
            "full_vocabulary_output_kl_nats",
        }:
            raise ValueError("completion certificate probe schema differs")
        if (
            row["probe_id"] != probe["probe_id"]
            or row["probe_kind"] != probe["probe_kind"]
        ):
            raise ValueError("completion certificate probe binding differs")
        values.append(
            _number(
                row["full_vocabulary_output_kl_nats"],
                path=f"completion_row.certificate.probes[{index}].kl",
                minimum=0.0,
            )
        )
    mean = _number(
        certificate["mean_output_kl_nats"],
        path="completion_row.certificate.mean",
        minimum=0.0,
    )
    maximum = _number(
        certificate["max_output_kl_nats"],
        path="completion_row.certificate.max",
        minimum=0.0,
    )
    if not _close(mean, statistics.mean(values)) or not _close(
        maximum, max(values)
    ):
        raise ValueError("completion certificate aggregates differ")
    _validate_nonnegative_counters(
        certificate["solver_diagnostics"],
        path="completion_row.certificate.solver_diagnostics",
    )


def _validate_cross_condition_metrics(
    conditions: Mapping[str, Any],
    *,
    expected_probes: Sequence[Mapping[str, Any]],
) -> None:
    if (
        expected_probes[0]["probe_id"] != "target_current"
        or expected_probes[0]["probe_kind"] != "deleted"
        or expected_probes[1]["probe_id"] != "retained"
        or expected_probes[1]["probe_kind"] != "retained"
    ):
        raise ValueError("completion target/retained probe roles differ")
    present = conditions[methods.PRESENT_ID]
    raw = conditions[methods.FRESH_RAW_OMISSION_ID]
    if present.get("status") != "completed" or raw.get("status") != "completed":
        raise ValueError("completion baseline references are incomplete")

    def metrics(condition: Mapping[str, Any], retained: bool) -> Mapping[str, Any]:
        if retained:
            return condition["retained_quality"]
        return condition["deleted_target_quality"]["probes"][0]

    for condition_id in methods.CONDITION_IDS:
        condition = conditions[condition_id]
        if condition.get("status") != "completed":
            continue
        for retained in (False, True):
            observed = metrics(condition, retained)
            present_metrics = metrics(present, retained)
            raw_metrics = metrics(raw, retained)
            method_mean = float(observed["score"]["mean_log_probability"])
            expected_suppression = (
                float(present_metrics["score"]["mean_log_probability"])
                - method_mean
            )
            expected_drift = (
                method_mean
                - float(raw_metrics["score"]["mean_log_probability"])
            )
            if not _close(
                float(observed["suppression_vs_present_nats"]),
                expected_suppression,
            ) or not _close(
                float(
                    observed[
                        "mean_log_probability_drift_from_raw_repack_nats"
                    ]
                ),
                expected_drift,
            ):
                raise ValueError(
                    "completion cross-condition suppression/drift differs"
                )
        deleted = condition["deleted_target_quality"]
        if not _close(
            float(deleted["mean_suppression_vs_present_nats"]),
            float(
                deleted["probes"][0]["suppression_vs_present_nats"]
            ),
        ) or not _close(
            float(deleted["mean_drift_from_raw_repack_nats"]),
            float(
                deleted["probes"][0][
                    "mean_log_probability_drift_from_raw_repack_nats"
                ]
            ),
        ):
            raise ValueError(
                "completion aggregate suppression/drift differs"
            )


def _validate_final_row_schema(
    row: Mapping[str, Any],
    *,
    completion_lock: Mapping[str, Any],
    partial_report: Mapping[str, Any],
    binding: Mapping[str, Any],
    public_record: Mapping[str, Any],
) -> None:
    """Validate any measured record-16 terminal outcome without outcome bias."""

    if set(row) != set(_SEALED_COMPLETION_ROW_KEYS):
        raise ValueError("completion measured row top-level key schema differs")
    first15 = list(partial_report["records"][:SEALED_PARTIAL_RECORDS])
    if len(first15) != SEALED_PARTIAL_RECORDS:
        raise ValueError("completion row schema requires sealed first15 templates")
    unsealed = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "completion_lock_integrity_sha256",
            "execution_contract_sha256",
            "row_integrity",
        }
    }
    if set(unsealed) != set(_UNSEALED_MEASURED_ROW_KEYS):
        raise ValueError("completion row measured key schema differs")

    if row.get("record_seed") != methods.hardened._record_seed(
        EXPECTED_REMAINING_RECORD_ID
    ):
        raise ValueError("completion row record seed differs")
    if row.get("admission_binding") != binding:
        raise ValueError("completion row admission binding differs")
    if row.get("predeclared_joint_admitted") is not bool(
        binding["joint_admitted"]
    ):
        raise ValueError("completion row predeclared admission differs")
    if row.get("probes") != _expected_row_probes(public_record):
        raise ValueError("completion row probe hashes differ")

    expected_geometry = {
        "manifest_record_integrity_sha256": public_record["record_integrity"][
            "sha256"
        ],
        "fixed_c_feasible": True,
        "local_window_safe": True,
    }
    if row.get("geometry_binding") != expected_geometry:
        raise ValueError("completion row manifest binding differs")

    shared = row.get("shared_original_prefill")
    if not isinstance(shared, Mapping) or set(shared) != {
        "seconds",
        "synchronized_device",
        "input_digest",
        "storage",
    }:
        raise ValueError("completion row shared prefill schema differs")
    if (
        not isinstance(shared["seconds"], (int, float))
        or isinstance(shared["seconds"], bool)
        or float(shared["seconds"]) < 0.0
        or shared["synchronized_device"]
        != completion_lock["authorization"]["runtime"]["device"]
        or shared["input_digest"]
        != public_record["context"]["original_token_ids_sha256"]
    ):
        raise ValueError("completion row shared prefill binding differs")
    _validate_recursive_schema(
        shared["storage"],
        [template["shared_original_prefill"]["storage"] for template in first15],
        path="completion_row.shared_original_prefill.storage",
        key="storage",
    )
    _validate_nonnegative_tree(
        shared["storage"],
        path="completion_row.shared_original_prefill.storage",
    )

    conditions = row.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        methods.CONDITION_IDS
    ):
        raise ValueError("completion row condition IDs differ")
    expected_probes = _expected_row_probes(public_record)
    original_digest = public_record["context"]["original_token_ids_sha256"]
    raw_digest = public_record["context"]["raw_omitted_token_ids_sha256"]
    digest_by_condition = {
        methods.PRESENT_ID: original_digest,
        methods.FRESH_RAW_OMISSION_ID: raw_digest,
        methods.TOKEN_ROW_DIAGNOSTIC_ID: raw_digest,
        methods.EXACT_DECREMENT_ID: original_digest,
        methods.FIXED_C_REFIT_ID: original_digest,
        methods.FP32_PROXY_ID: original_digest,
        methods.DECAY_ID: original_digest,
        methods.CACHE_DELETE_SHIFT_ID: original_digest,
        methods.PROMPT_SUPPRESSION_ID: original_digest,
    }
    for condition_id in methods.CONDITION_IDS:
        condition = conditions[condition_id]
        if not isinstance(condition, Mapping):
            raise ValueError("completion row condition is not an object")
        status = condition.get("status")
        if status == "failed":
            if condition_id in {
                methods.PRESENT_ID,
                methods.FRESH_RAW_OMISSION_ID,
                methods.TOKEN_ROW_DIAGNOSTIC_ID,
            }:
                raise ValueError(
                    "completion evaluator cannot return this condition failure"
                )
            if set(condition) != set(_FAILED_CONDITION_KEYS):
                raise ValueError("completion failed condition schema differs")
            _validate_condition(condition, condition_id=condition_id)
            continue
        if status != "completed":
            raise ValueError("completion condition status differs")
        templates = [
            template["conditions"][condition_id]
            for template in first15
            if template["conditions"][condition_id].get("status") == "completed"
        ]
        _validate_completed_condition(
            condition,
            condition_id=condition_id,
            completion_lock=completion_lock,
            expected_probes=expected_probes,
            expected_digest=digest_by_condition[condition_id],
            templates=templates,
        )
    _validate_cross_condition_metrics(
        conditions,
        expected_probes=expected_probes,
    )

    failed_ids = [
        condition_id
        for condition_id in methods.CONDITION_IDS
        if conditions[condition_id]["status"] == "failed"
    ]
    expected_status = "completed" if not failed_ids else "completed_with_failures"
    if row.get("method_failures") != failed_ids or row.get("status") != expected_status:
        raise ValueError("completion row terminal status is inconsistent")
    _validate_certificate(
        row["solver_certificate"],
        conditions=conditions,
        expected_probes=expected_probes,
    )
    immutability = row["source_state_immutability"]
    if not isinstance(immutability, Mapping) or set(immutability) != {
        "verification_scope",
        "before_sha256",
        "after_sha256",
        "shape_signature_unchanged",
        "unchanged",
    }:
        raise ValueError("completion immutability schema differs")
    if (
        immutability["verification_scope"]
        != "metadata_shapes_and_complete_tensor_values"
        or immutability["shape_signature_unchanged"] is not True
        or immutability["unchanged"] is not True
        or not _is_sha256(immutability["before_sha256"])
        or immutability["before_sha256"] != immutability["after_sha256"]
    ):
        raise ValueError("completion immutability evidence differs")


def _validate_measured_row(
    row: Mapping[str, Any],
    *,
    expected_id: str,
    binding: Mapping[str, Any],
    public_record: Mapping[str, Any],
    allowed_statuses: frozenset[str],
    require_certificate_success: bool = True,
) -> None:
    """Revalidate a measured row without trusting any row-local checksum."""

    _assert_source_free_and_finite(row, path=f"row[{expected_id}]")
    if row.get("record_id") != expected_id:
        raise ValueError("method row identity differs from authorization")
    status = str(row.get("status") or "")
    if status not in allowed_statuses:
        raise ValueError("method row terminal status is not authorized")
    if row.get("admission_binding") != binding:
        raise ValueError("method row admission binding differs")
    if (
        row.get("predeclared_joint_admitted")
        is not bool(binding.get("joint_admitted"))
        or row.get("source_fixed_c_feasible") is not True
    ):
        raise ValueError("method row authorization metadata differs")

    expected_geometry_binding = {
        "manifest_record_integrity_sha256": public_record["record_integrity"][
            "sha256"
        ],
        "fixed_c_feasible": True,
        "local_window_safe": True,
    }
    if row.get("geometry_binding") != expected_geometry_binding:
        raise ValueError("method row geometry binding differs")

    probes = row.get("probes")
    if (
        not isinstance(probes, list)
        or [probe.get("probe_id") for probe in probes]
        != ["target_current", "retained"]
        or any(int(probe.get("target_token_count", 0)) < 1 for probe in probes)
    ):
        raise ValueError("method row probe contract differs")

    conditions = row.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        methods.CONDITION_IDS
    ):
        raise ValueError("method row condition matrix differs")
    for condition_id in methods.CONDITION_IDS:
        _validate_condition(conditions[condition_id], condition_id=condition_id)

    outcome = methods._reconcile_row(row)
    if (
        outcome["declared_status_consistent"] is not True
        or outcome["condition_ids_exact"] is not True
        or outcome["method_failures_consistent"] is not True
        or outcome["source_state_immutability_success"] is not True
    ):
        raise ValueError("method row fails hardened reconciliation")
    if require_certificate_success and outcome["certificate_success"] is not True:
        raise ValueError("locked partial row solver certificate is incomplete")
    certificate = row.get("solver_certificate") or {}
    if outcome["certificate_status"] == "failed":
        failed_ids = set(outcome["failed_condition_ids"])
        if (
            status != "completed_with_failures"
            or not {
                methods.EXACT_DECREMENT_ID,
                methods.FIXED_C_REFIT_ID,
            }.issubset(failed_ids)
            or certificate.get("error_message_redacted") is not True
            or not str(certificate.get("error_type") or "")
            or not _is_sha256(certificate.get("error_message_sha256"))
        ):
            raise ValueError("failed solver certificate is not hardened")
    elif outcome["certificate_status"] == "completed":
        if (
            not isinstance(certificate.get("probe_output_kls"), list)
            or not isinstance(
                certificate.get("mean_output_kl_nats"),
                (int, float),
            )
            or isinstance(certificate.get("mean_output_kl_nats"), bool)
            or not isinstance(
                certificate.get("max_output_kl_nats"),
                (int, float),
            )
            or isinstance(certificate.get("max_output_kl_nats"), bool)
        ):
            raise ValueError("completed solver certificate metrics are incomplete")
    else:
        raise ValueError("method row solver certificate status differs")
    if status == "completed" and outcome["fully_completed"] is not True:
        raise ValueError("completed method row is not fully complete")
    if (
        status == "completed_with_failures"
        and not outcome["failed_condition_ids"]
    ):
        raise ValueError("failure-labelled method row has no failed condition")

    immutability = row.get("source_state_immutability") or {}
    if (
        immutability.get("before_sha256")
        != immutability.get("after_sha256")
        or not _is_sha256(immutability.get("before_sha256"))
    ):
        raise ValueError("method row lacks complete source-state immutability")

    token_alias = conditions[methods.TOKEN_ROW_DIAGNOSTIC_ID]
    if token_alias.get("status") == "completed":
        semantics = token_alias.get("semantics") or {}
        if (
            semantics.get("alias_of") != methods.FRESH_RAW_OMISSION_ID
            or semantics.get("additional_model_execution") is not False
            or semantics.get("executed_method") != "zero_cost_alias"
        ):
            raise ValueError("method row token diagnostic semantics differ")

    exact = conditions[methods.EXACT_DECREMENT_ID]
    if exact.get("status") == "completed" and (
        exact.get("semantics") or {}
    ).get("full_repack_fallback") is not False:
        raise ValueError("method row exact decrement used a source fallback")
    proxy = conditions[methods.FP32_PROXY_ID]
    if proxy.get("status") == "completed" and (
        proxy.get("semantics") or {}
    ).get("full_repack_fallback") is not False:
        raise ValueError("method row FP32 proxy used a source fallback")


def _seal_descriptors(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "position": index,
            "record_id": str(row["record_id"]),
            "row_integrity": {
                "algorithm": "sha256",
                "sha256": _payload_sha256(row),
            },
        }
        for index, row in enumerate(rows)
    ]


def validate_partial_report(
    report: Mapping[str, Any],
    *,
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact externally aborted 15+1 report state."""

    _assert_source_free_and_finite(report, path="partial_report")
    method_contract = _method_implementation_contract(method_lock)
    authorization = method_lock["authorization"]
    bindings = list(authorization["admission_records"])
    authorized_ids = [str(binding["record_id"]) for binding in bindings]
    if manifest is None:
        manifest = methods.load_locked_artifacts()[0]
    public_records = list(manifest["records"])

    config = report.get("config") or {}
    warmup = config.get("warmup")
    repeats = config.get("repeats")
    if (
        isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or warmup < 0
        or isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 1
    ):
        raise ValueError("partial report timing repetition contract differs")
    expected = methods._base_report(
        method_lock,
        device=str(method_contract["runtime"]["device"]),
        warmup=warmup,
        repeats=repeats,
    )
    if methods._validate_resume(report, expected) != EXPECTED_RECORDS:
        raise ValueError("partial report must contain the all16 row slots")
    if report.get("status") != "running":
        raise ValueError("partial report is not the externally aborted run")

    rows_value = report.get("records")
    if not isinstance(rows_value, list) or len(rows_value) != EXPECTED_RECORDS:
        raise ValueError("partial report must contain exactly 16 ordered rows")
    if not all(isinstance(row, Mapping) for row in rows_value):
        raise ValueError("partial report contains a non-object row")
    rows = list(rows_value)
    row_ids = [str(row.get("record_id") or "") for row in rows]
    if row_ids != authorized_ids or len(set(row_ids)) != EXPECTED_RECORDS:
        raise ValueError("partial report row order differs from method lock")
    if row_ids[-1] != EXPECTED_REMAINING_RECORD_ID:
        raise ValueError("partial report remaining record identity differs")

    first15 = rows[:SEALED_PARTIAL_RECORDS]
    if any(row.get("status") != "completed_with_failures" for row in first15):
        raise ValueError("partial report first15 terminal statuses differ")
    if dict(rows[-1]) != expected["records"][-1]:
        raise ValueError("partial report final row is not the exact placeholder")

    for row, binding, public_record in zip(
        first15,
        bindings[:SEALED_PARTIAL_RECORDS],
        public_records[:SEALED_PARTIAL_RECORDS],
    ):
        _validate_measured_row(
            row,
            expected_id=str(binding["record_id"]),
            binding=binding,
            public_record=public_record,
            allowed_statuses=frozenset({"completed_with_failures"}),
        )

    expected_summary = methods.summarize_records(rows, method_lock=method_lock)
    if report.get("summary") != expected_summary:
        raise ValueError("partial report summary does not recompute exactly")
    denominators = expected_summary["denominators"]
    required = {
        "authorized_records": EXPECTED_RECORDS,
        "attempted_records": SEALED_PARTIAL_RECORDS,
        "not_attempted_records": 1,
        "fully_completed_records": 0,
        "record_failures": SEALED_PARTIAL_RECORDS,
        "declared_completed_with_failures_records": SEALED_PARTIAL_RECORDS,
        "source_state_immutable_records": SEALED_PARTIAL_RECORDS,
        "source_state_immutability_failures": 0,
    }
    if any(int(denominators.get(key, -1)) != value for key, value in required.items()):
        raise ValueError("partial report 15+1 accounting differs")

    seals = _seal_descriptors(first15)
    return {
        "first15_rows": first15,
        "first15_record_ids": row_ids[:SEALED_PARTIAL_RECORDS],
        "remaining_record_id": row_ids[-1],
        "sealed_first15_rows": seals,
        "sealed_first15_rows_sha256": _payload_sha256(seals),
        "config": copy.deepcopy(dict(config)),
        "warmup": warmup,
        "repeats": repeats,
        "method_contract": method_contract,
    }


def _validate_upstream_method_lock(
    method_lock: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
) -> None:
    methods.validate_method_authorization_lock(
        method_lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
    )
    _method_implementation_contract(method_lock)


def freeze_completion_lock(
    partial_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    partial_report_file_sha256: str,
    method_lock_file_sha256: str,
) -> dict[str, Any]:
    """Freeze the exact partial measurements before any completion scoring."""

    if not _is_sha256(partial_report_file_sha256):
        raise ValueError("partial report file SHA-256 is invalid")
    if not _is_sha256(method_lock_file_sha256):
        raise ValueError("method lock file SHA-256 is invalid")
    _validate_upstream_method_lock(
        method_lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
    )
    analysis = validate_partial_report(
        partial_report,
        method_lock=method_lock,
        manifest=manifest,
    )
    method_contract = analysis["method_contract"]
    completion_implementation = _completion_implementation_fingerprints()
    attempt_marker_contract = {
        "schema": ATTEMPT_MARKER_SCHEMA,
        "canonical_claim_workspace_path": (
            CANONICAL_ATTEMPT_CLAIM_WORKSPACE_PATH
        ),
        "marker_filename": ATTEMPT_MARKER_FILENAME,
        "atomic_claim": "os.mkdir",
        "marker_publication": "temporary-file-plus-os.replace",
        "persistent": True,
        "attempts_authorized": 1,
        "record_id": analysis["remaining_record_id"],
    }

    partial_payload_sha256 = _payload_sha256(partial_report)
    method_lock_payload_sha256 = _payload_sha256(method_lock)
    execution_contract = {
        "condition_ids": list(methods.CONDITION_IDS),
        "attempt_marker_contract": copy.deepcopy(attempt_marker_contract),
        "completion_implementation_contract_sha256": completion_implementation[
            "contract_sha256"
        ],
        "execute_only_record_id": analysis["remaining_record_id"],
        "method_implementation_contract_sha256": method_contract[
            "implementation_contract_sha256"
        ],
        "method_lock_integrity_sha256": method_lock["integrity"]["sha256"],
        "partial_report_file_sha256": partial_report_file_sha256,
        "partial_report_payload_sha256": partial_payload_sha256,
        "records_executed": 1,
        "records_rehydrated_and_validated": EXPECTED_RECORDS,
        "runtime": copy.deepcopy(method_contract["runtime"]),
        "warmup": analysis["warmup"],
        "repeats": analysis["repeats"],
    }
    execution_contract_sha256 = _payload_sha256(execution_contract)

    lock = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": LOCK_STATUS,
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "external_abort_disclosure": copy.deepcopy(EXTERNAL_ABORT_DISCLOSURE),
        "partial_report": {
            "file_sha256": partial_report_file_sha256,
            "payload_sha256": partial_payload_sha256,
            "schema": partial_report.get("schema"),
            "status": partial_report.get("status"),
            "authorized_records": EXPECTED_RECORDS,
            "attempted_records": SEALED_PARTIAL_RECORDS,
            "not_attempted_records": 1,
            "ordered_first15_record_ids": analysis["first15_record_ids"],
            "ordered_first15_record_ids_sha256": _payload_sha256(
                analysis["first15_record_ids"]
            ),
            "sealed_first15_rows": analysis["sealed_first15_rows"],
            "sealed_first15_rows_sha256": analysis[
                "sealed_first15_rows_sha256"
            ],
            "remaining_record_id": analysis["remaining_record_id"],
        },
        "method_contract": {
            "method_lock_file_sha256": method_lock_file_sha256,
            "method_lock_payload_sha256": method_lock_payload_sha256,
            "method_lock_integrity_sha256": method_lock["integrity"]["sha256"],
            "artifacts": copy.deepcopy(method_contract["artifacts"]),
            "artifacts_sha256": method_contract["artifacts_sha256"],
            "implementation": copy.deepcopy(method_contract["implementation"]),
            "implementation_contract_sha256": method_contract[
                "implementation_contract_sha256"
            ],
            "implementation_files": method_contract["implementation_files"],
            "runtime": copy.deepcopy(method_contract["runtime"]),
            "runtime_sha256": method_contract["runtime_sha256"],
        },
        "completion_implementation": completion_implementation,
        "authorization": {
            "all16_rehydrated_and_validated": True,
            "attempt_marker": copy.deepcopy(attempt_marker_contract),
            "condition_ids": list(methods.CONDITION_IDS),
            "execute_only_record_id": analysis["remaining_record_id"],
            "execution_contract": execution_contract,
            "execution_contract_sha256": execution_contract_sha256,
            "first15_rows_reexecuted": False,
            "first15_rows_reused_only_from_this_lock": True,
            "no_record_replacement": True,
            "records_executed": 1,
            "records_rehydrated": EXPECTED_RECORDS,
            "runtime": copy.deepcopy(method_contract["runtime"]),
            "warmup": analysis["warmup"],
            "repeats": analysis["repeats"],
        },
    }
    lock["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(lock),
    }
    _assert_source_free_and_finite(lock, path="completion_lock")
    return lock


def validate_completion_lock(
    lock: Mapping[str, Any],
    *,
    partial_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    partial_report_file_sha256: str,
    method_lock_file_sha256: str,
) -> None:
    """Reproduce the lock; its self-checksum alone is never authoritative."""

    _assert_source_free_and_finite(lock, path="completion_lock")
    integrity = lock.get("integrity") or {}
    unsigned = dict(lock)
    unsigned.pop("integrity", None)
    if (
        lock.get("schema") != SCHEMA
        or lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("status") != LOCK_STATUS
        or lock.get("external_abort_disclosure") != EXTERNAL_ABORT_DISCLOSURE
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise ValueError("completion lock integrity or disclosure differs")
    expected = freeze_completion_lock(
        partial_report,
        method_lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock_file_sha256=method_lock_file_sha256,
    )
    if dict(lock) != expected:
        raise ValueError("completion lock differs from frozen partial run")


def _require_head_committed_file(path: Path) -> None:
    """Require the canonical lock bytes to be present in the current HEAD."""

    workspace = WORKSPACE.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise PermissionError("completion lock is outside the git workspace") from exc
    completed = subprocess.run(
        ["git", "-C", str(workspace), "show", f"HEAD:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise PermissionError("completion lock must be committed at HEAD")
    if completed.stdout != resolved.read_bytes():
        raise PermissionError("completion lock differs from the committed HEAD copy")


def load_committed_completion_lock(
    path: str | Path,
    *,
    partial_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    partial_report_path: str | Path,
    method_lock_path: str | Path,
) -> dict[str, Any]:
    """Load only the canonical, committed completion authorization."""

    lock_path = Path(path).expanduser()
    if lock_path.resolve() != DEFAULT_COMPLETION_LOCK.resolve():
        raise PermissionError(
            "one-record completion requires the canonical completion lock"
        )
    _require_head_committed_file(lock_path)
    lock = _load_mapping(lock_path, name="completion lock")
    validate_completion_lock(
        lock,
        partial_report=partial_report,
        method_lock=method_lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
        partial_report_file_sha256=_sha256_file(partial_report_path),
        method_lock_file_sha256=_sha256_file(method_lock_path),
    )
    return lock


def authorize_completion(
    lock: Mapping[str, Any],
    *,
    explicit_acknowledgement: Any,
) -> tuple[str, str]:
    if explicit_acknowledgement is not True:
        raise PermissionError("completion acknowledgement must be exactly True")
    authorization = lock.get("authorization") or {}
    if (
        authorization.get("execute_only_record_id")
        != EXPECTED_REMAINING_RECORD_ID
        or authorization.get("records_executed") != 1
        or authorization.get("first15_rows_reexecuted") is not False
        or authorization.get("no_record_replacement") is not True
    ):
        raise PermissionError("completion authorization is not one-record-only")
    runtime = authorization.get("runtime") or {}
    return str(runtime.get("arm") or ""), str(runtime.get("device") or "")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_new_output_paths(
    outputs: Sequence[str | Path],
    *,
    input_paths: Sequence[str | Path | None],
) -> tuple[Path, ...]:
    if not outputs:
        raise ValueError("at least one output is required")
    raw_outputs = [Path(path).expanduser() for path in outputs]
    resolved_outputs = [_resolved(path) for path in raw_outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("completion outputs alias each other")

    protected = [
        Path(path).expanduser()
        for path in (*input_paths, *_PROTECTED_INPUTS)
        if path is not None
    ]
    for raw, resolved in zip(raw_outputs, resolved_outputs):
        for protected_path in protected:
            same_path = resolved == _resolved(protected_path)
            same_file = False
            if os.path.lexists(raw) and protected_path.exists():
                try:
                    same_file = os.path.samefile(raw, protected_path)
                except OSError:
                    pass
            if same_path or same_file:
                raise ValueError("completion output aliases a protected input")
        if os.path.lexists(raw):
            raise FileExistsError("completion outputs are immutable and must be new")
    return tuple(resolved_outputs)


def _canonical_completion_output_state(
    completion_row_output: str | Path,
    final_output: str | Path,
    *,
    input_paths: Sequence[str | Path | None],
) -> tuple[Path, Path, bool]:
    """Require the two canonical names and classify row-first recovery."""

    requested = (
        Path(completion_row_output).expanduser(),
        Path(final_output).expanduser(),
    )
    canonical = (
        Path(DEFAULT_COMPLETION_ROW_OUTPUT).expanduser(),
        Path(DEFAULT_FINAL_OUTPUT).expanduser(),
    )
    for path, expected, label in zip(
        requested,
        canonical,
        ("completion row", "final report"),
    ):
        if Path(os.path.abspath(path)) != Path(os.path.abspath(expected)):
            raise PermissionError(f"{label} output must use its canonical path")

    row_path, final_path = canonical
    if os.path.lexists(final_path):
        raise FileExistsError("canonical final report already exists")
    if os.path.lexists(row_path):
        if row_path.is_symlink() or not row_path.is_file():
            raise ValueError("canonical completion row is not a regular file")
        for protected in input_paths:
            if protected is None or not Path(protected).exists():
                continue
            try:
                if os.path.samefile(row_path, protected):
                    raise ValueError(
                        "canonical completion row aliases a protected input"
                    )
            except OSError:
                continue
        return _resolved(row_path), _resolved(final_path), True

    row_path, final_path = _validate_new_output_paths(
        canonical,
        input_paths=input_paths,
    )
    return row_path, final_path, False


@dataclass(frozen=True, slots=True)
class _AttemptClaim:
    token: object
    mode: str
    lock_integrity_sha256: str
    completion_lock_path: Path
    expected_record_id: str
    claim_dir: Path
    marker_path: Path
    completion_row_output: Path
    final_output: Path
    descriptor: int | None = None


@dataclass(frozen=True, slots=True)
class _AttemptClaimBinding:
    claim: _AttemptClaim
    token: object
    mode: str
    lock_integrity_sha256: str
    completion_lock_path: Path
    expected_record_id: str
    claim_dir: Path
    marker_path: Path
    completion_row_output: Path
    final_output: Path
    descriptor: int | None


_ACTIVE_CLAIMS: dict[int, _AttemptClaimBinding] = {}


def _register_attempt_claim(claim: _AttemptClaim) -> _AttemptClaim:
    _ACTIVE_CLAIMS[id(claim)] = _AttemptClaimBinding(
        claim=claim,
        token=claim.token,
        mode=claim.mode,
        lock_integrity_sha256=claim.lock_integrity_sha256,
        completion_lock_path=claim.completion_lock_path,
        expected_record_id=claim.expected_record_id,
        claim_dir=claim.claim_dir,
        marker_path=claim.marker_path,
        completion_row_output=claim.completion_row_output,
        final_output=claim.final_output,
        descriptor=claim.descriptor,
    )
    return claim


def _new_attempt_claim(
    *,
    mode: str,
    completion_lock: Mapping[str, Any],
    completion_lock_path: Path,
    claim_dir: Path,
    marker_path: Path,
    completion_row_output: Path,
    final_output: Path,
    descriptor: int | None,
) -> _AttemptClaim:
    return _register_attempt_claim(
        _AttemptClaim(
            token=object(),
            mode=mode,
            lock_integrity_sha256=completion_lock["integrity"]["sha256"],
            completion_lock_path=_resolved(completion_lock_path),
            expected_record_id=EXPECTED_REMAINING_RECORD_ID,
            claim_dir=claim_dir,
            marker_path=marker_path,
            completion_row_output=completion_row_output,
            final_output=final_output,
            descriptor=descriptor,
        )
    )


def _attempt_marker_payload(
    completion_lock: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema": ATTEMPT_MARKER_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "attempt-consumed-before-model-load",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "record_id": EXPECTED_REMAINING_RECORD_ID,
        "record_ordinal": EXPECTED_RECORDS,
        "attempt_number": 1,
        "completion_lock_integrity_sha256": completion_lock["integrity"][
            "sha256"
        ],
        "execution_contract_sha256": completion_lock["authorization"][
            "execution_contract_sha256"
        ],
    }
    return {
        **unsigned,
        "integrity": {
            "algorithm": "sha256",
            "sha256": _payload_sha256(unsigned),
        },
    }


def validate_attempt_marker(
    marker: Mapping[str, Any],
    *,
    completion_lock: Mapping[str, Any],
) -> None:
    _assert_source_free_and_finite(marker, path="completion_attempt_marker")
    expected = _attempt_marker_payload(completion_lock)
    if dict(marker) != expected:
        raise ValueError("completion attempt marker contract differs")


def _canonical_attempt_claim_paths(
    *,
    input_paths: Sequence[str | Path | None],
    completion_row_output: Path,
    final_output: Path,
) -> tuple[Path, Path, Path, bool]:
    claim_dir = Path(DEFAULT_ATTEMPT_CLAIM_DIR).expanduser()
    marker = claim_dir / ATTEMPT_MARKER_FILENAME
    lock_path = Path(DEFAULT_ATTEMPT_LOCK).expanduser()
    claim_resolved = _resolved(claim_dir)
    marker_resolved = _resolved(marker)
    lock_resolved = _resolved(lock_path)
    protected_resolved = {
        _resolved(completion_row_output),
        _resolved(final_output),
        *(_resolved(path) for path in input_paths if path is not None),
    }
    if (
        claim_resolved in protected_resolved
        or marker_resolved in protected_resolved
        or lock_resolved in protected_resolved
        or len({claim_resolved, marker_resolved, lock_resolved}) != 3
    ):
        raise ValueError("completion attempt claim aliases a protected path")
    claim_exists = os.path.lexists(claim_dir)
    if claim_exists and (claim_dir.is_symlink() or not claim_dir.is_dir()):
        raise ValueError("canonical attempt claim is not a real directory")
    if os.path.lexists(lock_path) and (
        lock_path.is_symlink() or not lock_path.is_file()
    ):
        raise ValueError("canonical attempt lock is not a regular file")
    for path in input_paths:
        if path is None or not Path(path).exists():
            continue
        for protected in (claim_dir, marker, lock_path):
            if not protected.exists():
                continue
            try:
                if os.path.samefile(protected, path):
                    raise ValueError(
                        "completion attempt claim aliases a protected input"
                    )
            except OSError:
                continue
    return claim_resolved, marker_resolved, lock_resolved, claim_exists


def _release_attempt_claim(claim: _AttemptClaim | None) -> None:
    if claim is None:
        return
    binding = _ACTIVE_CLAIMS.pop(id(claim), None)
    if binding is None or binding.claim is not claim or binding.descriptor is None:
        return
    try:
        fcntl.flock(binding.descriptor, fcntl.LOCK_UN)
    finally:
        os.close(binding.descriptor)


def _acquire_attempt_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PermissionError(
                "completion attempt is already active; refusing concurrent score"
            ) from exc
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _publish_attempt_marker(
    claim_dir: Path,
    *,
    completion_lock: Mapping[str, Any],
) -> Path:
    marker = claim_dir / ATTEMPT_MARKER_FILENAME
    payload = _attempt_marker_payload(completion_lock)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=claim_dir,
        prefix=".marker.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        rendered = (
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker)
        try:
            directory_descriptor = os.open(claim_dir, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
        validate_attempt_marker(
            _load_mapping(marker, name="completion attempt marker"),
            completion_lock=completion_lock,
        )
        return marker
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_internal_claim(
    claim: Any,
    *,
    completion_lock: Mapping[str, Any],
    completion_lock_path: Path,
    completion_row_output: Path,
    final_output: Path,
) -> _AttemptClaim:
    binding = _ACTIVE_CLAIMS.get(id(claim))
    if (
        type(claim) is not _AttemptClaim
        or binding is None
        or binding.claim is not claim
        or claim.token is not binding.token
        or claim.mode != binding.mode
        or claim.lock_integrity_sha256 != binding.lock_integrity_sha256
        or claim.completion_lock_path != binding.completion_lock_path
        or claim.expected_record_id != binding.expected_record_id
        or claim.claim_dir != binding.claim_dir
        or claim.marker_path != binding.marker_path
        or claim.completion_row_output != binding.completion_row_output
        or claim.final_output != binding.final_output
        or claim.descriptor != binding.descriptor
        or claim.lock_integrity_sha256
        != completion_lock["integrity"]["sha256"]
        or claim.completion_lock_path != _resolved(completion_lock_path)
        or claim.expected_record_id != EXPECTED_REMAINING_RECORD_ID
        or claim.claim_dir != _resolved(DEFAULT_ATTEMPT_CLAIM_DIR)
        or claim.marker_path
        != _resolved(DEFAULT_ATTEMPT_CLAIM_DIR) / ATTEMPT_MARKER_FILENAME
        or claim.completion_row_output != _resolved(completion_row_output)
        or claim.final_output != _resolved(final_output)
        or claim.mode not in {"score", "recover", "merge"}
        or (
            claim.mode in {"score", "recover"}
            and claim.descriptor is None
        )
        or not claim.claim_dir.is_dir()
        or claim.claim_dir.is_symlink()
    ):
        raise TypeError("internal completion attempt claim is invalid")
    if claim.descriptor is not None:
        try:
            os.fstat(claim.descriptor)
        except OSError as exc:
            raise TypeError(
                "internal completion attempt claim descriptor is invalid"
            ) from exc
    return claim


def _prepare_attempt_claim(
    *,
    completion_lock: Mapping[str, Any],
    completion_lock_path: Path,
    completion_row_output: Path,
    final_output: Path,
    input_paths: Sequence[str | Path | None],
    merge_only: bool,
) -> _AttemptClaim:
    row_path, final_path, row_exists = _canonical_completion_output_state(
        completion_row_output,
        final_output,
        input_paths=input_paths,
    )
    claim_dir, marker_path, lock_path, claim_exists = (
        _canonical_attempt_claim_paths(
            input_paths=input_paths,
            completion_row_output=row_path,
            final_output=final_path,
        )
    )
    descriptor = _acquire_attempt_lock(lock_path)
    claim_exists = os.path.lexists(claim_dir)
    if claim_exists and (claim_dir.is_symlink() or not claim_dir.is_dir()):
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise ValueError("canonical attempt claim is not a real directory")
    if row_exists and not claim_exists:
        os.close(descriptor)
        raise ValueError("completion row exists without its attempt claim")
    if claim_exists:
        mode = "merge" if row_exists else "recover"
        claim = _new_attempt_claim(
            mode=mode,
            completion_lock=completion_lock,
            completion_lock_path=completion_lock_path,
            claim_dir=claim_dir,
            marker_path=marker_path,
            completion_row_output=row_path,
            final_output=final_path,
            descriptor=descriptor if mode == "recover" else None,
        )
        if mode == "merge":
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        return claim
    if merge_only:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise FileNotFoundError(
            "merge-only requires the canonical attempt claim and sealed row"
        )
    try:
        os.mkdir(claim_dir, 0o755)
        try:
            parent_descriptor = os.open(claim_dir.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except OSError:
            pass
        _publish_attempt_marker(
            claim_dir,
            completion_lock=completion_lock,
        )
    except FileExistsError:
        return _new_attempt_claim(
            mode="recover",
            completion_lock=completion_lock,
            completion_lock_path=completion_lock_path,
            claim_dir=claim_dir,
            marker_path=marker_path,
            completion_row_output=row_path,
            final_output=final_path,
            descriptor=descriptor,
        )
    except Exception:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise
    return _new_attempt_claim(
        mode="score",
        completion_lock=completion_lock,
        completion_lock_path=completion_lock_path,
        claim_dir=claim_dir,
        marker_path=marker_path,
        completion_row_output=row_path,
        final_output=final_path,
        descriptor=descriptor,
    )


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a new JSON file without replacement."""

    _assert_source_free_and_finite(payload, path="output")
    if os.path.lexists(path):
        raise FileExistsError("refusing to replace an existing completion output")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                "refusing to replace an existing completion output"
            ) from exc
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_lock_sealed_partial_rows(
    partial_report: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    analysis = validate_partial_report(
        partial_report,
        method_lock=method_lock,
        manifest=manifest,
    )
    locked = lock.get("partial_report") or {}
    if (
        analysis["first15_record_ids"]
        != locked.get("ordered_first15_record_ids")
        or analysis["sealed_first15_rows"]
        != locked.get("sealed_first15_rows")
        or analysis["sealed_first15_rows_sha256"]
        != locked.get("sealed_first15_rows_sha256")
    ):
        raise ValueError("partial rows are not sealed by the completion lock")
    return list(analysis["first15_rows"])


def _seal_final_row(
    row: Mapping[str, Any],
    *,
    completion_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if "row_integrity" in row or "execution_contract_sha256" in row:
        raise ValueError("method engine returned a pre-authenticated row")
    enriched = dict(row)
    enriched["completion_lock_integrity_sha256"] = completion_lock["integrity"][
        "sha256"
    ]
    return methods.hardened._seal_completed_row(
        enriched,
        execution_contract_sha256=completion_lock["authorization"][
            "execution_contract_sha256"
        ],
    )


def _validate_final_row(
    row: Mapping[str, Any],
    *,
    completion_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_report: Mapping[str, Any],
    public_record: Mapping[str, Any],
) -> None:
    _assert_source_free_and_finite(row, path="completion_row")
    if row.get("record_id") != EXPECTED_REMAINING_RECORD_ID:
        raise ValueError("completion row identity differs")
    if row.get("completion_lock_integrity_sha256") != completion_lock[
        "integrity"
    ]["sha256"]:
        raise ValueError("completion row lock binding differs")
    integrity = row.get("row_integrity") or {}
    unsigned = dict(row)
    unsigned.pop("row_integrity", None)
    if (
        row.get("execution_contract_sha256")
        != completion_lock["authorization"]["execution_contract_sha256"]
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise ValueError("completion row seal differs")

    binding = method_lock["authorization"]["admission_records"][-1]
    if row.get("status") == "failed":
        if set(row) != set(_SEALED_GENERIC_FAILED_ROW_KEYS):
            raise ValueError("completion generic failure key schema differs")
        if (
            row.get("error_message_redacted") is not True
            or not _is_error_type(row.get("error_type"))
            or not _is_sha256(row.get("error_message_sha256"))
        ):
            raise ValueError("completion generic failure is not hardened")
        return
    _validate_final_row_schema(
        row,
        completion_lock=completion_lock,
        partial_report=partial_report,
        binding=binding,
        public_record=public_record,
    )
    _validate_measured_row(
        row,
        expected_id=EXPECTED_REMAINING_RECORD_ID,
        binding=binding,
        public_record=public_record,
        allowed_statuses=frozenset({"completed", "completed_with_failures"}),
        require_certificate_success=False,
    )


def _completion_row_artifact(
    row: Mapping[str, Any],
    *,
    completion_lock: Mapping[str, Any],
    completion_elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": ROW_ARTIFACT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": str(row.get("status") or ""),
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "completion_process_elapsed_seconds": float(
            completion_elapsed_seconds
        ),
        "external_abort_disclosure": copy.deepcopy(EXTERNAL_ABORT_DISCLOSURE),
        "completion_lock_integrity_sha256": completion_lock["integrity"][
            "sha256"
        ],
        "method_lock_integrity_sha256": completion_lock["method_contract"][
            "method_lock_integrity_sha256"
        ],
        "authorization": {
            "record_id": EXPECTED_REMAINING_RECORD_ID,
            "record_ordinal": EXPECTED_RECORDS,
            "records_executed": 1,
            "record_replacements": 0,
            "execution_contract_sha256": completion_lock["authorization"][
                "execution_contract_sha256"
            ],
        },
        "config": {
            "runtime": copy.deepcopy(
                completion_lock["authorization"]["runtime"]
            ),
            "warmup": completion_lock["authorization"]["warmup"],
            "repeats": completion_lock["authorization"]["repeats"],
        },
        "completion_implementation": copy.deepcopy(
            completion_lock["completion_implementation"]
        ),
        "records": [copy.deepcopy(dict(row))],
    }


def validate_completion_row_artifact(
    artifact: Mapping[str, Any],
    *,
    completion_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_report: Mapping[str, Any],
    public_record: Mapping[str, Any],
) -> None:
    _assert_source_free_and_finite(artifact, path="completion_row_artifact")
    records = artifact.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("completion artifact must contain exactly one row")
    row = records[0]
    if not isinstance(row, Mapping):
        raise ValueError("completion artifact row is not an object")
    elapsed = artifact.get("completion_process_elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or float(elapsed) < 0.0
    ):
        raise ValueError("completion row elapsed time is invalid")
    expected = _completion_row_artifact(
        row,
        completion_lock=completion_lock,
        completion_elapsed_seconds=float(elapsed),
    )
    if dict(artifact) != expected:
        raise ValueError("completion row artifact contract differs")
    _validate_final_row(
        row,
        completion_lock=completion_lock,
        method_lock=method_lock,
        partial_report=partial_report,
        public_record=public_record,
    )


def _final_status(summary: Mapping[str, Any]) -> str:
    denominators = summary.get("denominators") or {}
    return (
        "completed"
        if int(denominators.get("fully_completed_records", -1))
        == EXPECTED_RECORDS
        and int(denominators.get("record_failures", -1)) == 0
        else "completed_with_record_failures"
    )


def _build_final_report(
    partial_report: Mapping[str, Any],
    completion_row_artifact: Mapping[str, Any],
    *,
    completion_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    completion_row_file_sha256: str,
    completion_elapsed_seconds: float,
) -> dict[str, Any]:
    first15 = list(partial_report["records"][:SEALED_PARTIAL_RECORDS])
    final_row = completion_row_artifact["records"][0]
    records = [*copy.deepcopy(first15), copy.deepcopy(final_row)]
    summary = methods.summarize_records(records, method_lock=method_lock)
    report = copy.deepcopy(dict(partial_report))
    partial_elapsed = float(
        partial_report.get("elapsed_seconds_this_process") or 0.0
    )
    report["records"] = records
    report["summary"] = summary
    report["status"] = _final_status(summary)
    report["elapsed_seconds_this_process"] = float(
        completion_elapsed_seconds
    )
    report["external_abort_completion"] = {
        "schema": COMPLETION_METADATA_SCHEMA,
        "external_abort_disclosure": copy.deepcopy(
            EXTERNAL_ABORT_DISCLOSURE
        ),
        "completion_lock_integrity_sha256": completion_lock["integrity"][
            "sha256"
        ],
        "partial_report_file_sha256": completion_lock["partial_report"][
            "file_sha256"
        ],
        "partial_report_payload_sha256": completion_lock["partial_report"][
            "payload_sha256"
        ],
        "completion_row_file_sha256": completion_row_file_sha256,
        "completion_row_payload_sha256": _payload_sha256(
            completion_row_artifact
        ),
        "sealed_partial_records_reused": SEALED_PARTIAL_RECORDS,
        "records_executed_for_completion": 1,
        "records_replaced": 0,
        "remaining_record_id": EXPECTED_REMAINING_RECORD_ID,
        "partial_process_elapsed_seconds": partial_elapsed,
        "completion_process_elapsed_seconds": float(
            completion_elapsed_seconds
        ),
        "combined_process_elapsed_seconds": (
            partial_elapsed + float(completion_elapsed_seconds)
        ),
    }
    return report


def validate_final_report(
    report: Mapping[str, Any],
    *,
    partial_report: Mapping[str, Any],
    completion_row_artifact: Mapping[str, Any],
    completion_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    public_record: Mapping[str, Any],
    completion_row_file_sha256: str,
) -> None:
    _assert_source_free_and_finite(report, path="final_report")
    _validate_lock_sealed_partial_rows(
        partial_report,
        completion_lock,
        method_lock=method_lock,
        manifest=manifest,
    )
    validate_completion_row_artifact(
        completion_row_artifact,
        completion_lock=completion_lock,
        method_lock=method_lock,
        partial_report=partial_report,
        public_record=public_record,
    )
    elapsed = report.get("elapsed_seconds_this_process")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or float(elapsed) < 0.0
        or float(elapsed)
        != float(
            completion_row_artifact["completion_process_elapsed_seconds"]
        )
    ):
        raise ValueError("final report completion elapsed time is invalid")
    expected = _build_final_report(
        partial_report,
        completion_row_artifact,
        completion_lock=completion_lock,
        method_lock=method_lock,
        completion_row_file_sha256=completion_row_file_sha256,
        completion_elapsed_seconds=float(elapsed),
    )
    if dict(report) != expected:
        raise ValueError("merged all16 report does not recompute exactly")
    rows = report["records"]
    expected_ids = [
        str(binding["record_id"])
        for binding in method_lock["authorization"]["admission_records"]
    ]
    if (
        len(rows) != EXPECTED_RECORDS
        or [str(row.get("record_id") or "") for row in rows] != expected_ids
        or report["summary"]["denominators"]["attempted_records"]
        != EXPECTED_RECORDS
        or report["summary"]["denominators"]["not_attempted_records"] != 0
    ):
        raise ValueError("merged report does not account for ordered all16")


def _failed_row(record_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": methods.benchmark.base.text_sha256(str(exc)),
    }


def _failed_row_from_attempt_claim(
    completion_lock: Mapping[str, Any],
) -> dict[str, Any]:
    reason = {
        "reason": "attempt_claim_exists_without_terminal_row",
        "completion_lock_integrity_sha256": completion_lock["integrity"][
            "sha256"
        ],
        "record_id": EXPECTED_REMAINING_RECORD_ID,
    }
    return {
        "record_id": EXPECTED_REMAINING_RECORD_ID,
        "status": "failed",
        "error_type": "InterruptedCompletionAttempt",
        "error_message_redacted": True,
        "error_message_sha256": _payload_sha256(reason),
    }


def _merge_existing_completion_row(
    *,
    completion_row_output: Path,
    final_output: Path,
    partial_report: Mapping[str, Any],
    completion_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canonical row checkpoint and publish only the final report."""

    row_artifact = _load_mapping(
        completion_row_output,
        name="canonical completion row artifact",
    )
    final_public_record = manifest["records"][-1]
    validate_completion_row_artifact(
        row_artifact,
        completion_lock=completion_lock,
        method_lock=method_lock,
        partial_report=partial_report,
        public_record=final_public_record,
    )
    row_file_sha256 = _sha256_file(completion_row_output)
    elapsed = float(row_artifact["completion_process_elapsed_seconds"])
    final_report = _build_final_report(
        partial_report,
        row_artifact,
        completion_lock=completion_lock,
        method_lock=method_lock,
        completion_row_file_sha256=row_file_sha256,
        completion_elapsed_seconds=elapsed,
    )
    validate_final_report(
        final_report,
        partial_report=partial_report,
        completion_row_artifact=row_artifact,
        completion_lock=completion_lock,
        method_lock=method_lock,
        manifest=manifest,
        public_record=final_public_record,
        completion_row_file_sha256=row_file_sha256,
    )
    _atomic_write_new(final_output, final_report)
    return final_report


def run_completion(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_report: Mapping[str, Any],
    completion_lock: Mapping[str, Any],
    records: Sequence[Any] | None,
    runtime: Any,
    *,
    manifest_path: Path,
    policy_path: Path,
    census_path: Path,
    admission_report_path: Path,
    method_lock_path: Path,
    partial_report_path: Path,
    completion_lock_path: Path,
    core_manifest_path: Path,
    data_path: Path | None,
    completion_row_output: Path,
    final_output: Path,
    explicit_acknowledgement: Any,
    merge_only: bool = False,
    _attempt_claim: _AttemptClaim | None = None,
) -> dict[str, Any]:
    """Score once, or recover a validated canonical row without model use."""

    input_paths = (
        manifest_path,
        policy_path,
        census_path,
        admission_report_path,
        method_lock_path,
        partial_report_path,
        completion_lock_path,
        core_manifest_path,
        data_path,
    )
    if not merge_only:
        authorize_completion(
            completion_lock,
            explicit_acknowledgement=explicit_acknowledgement,
        )

    loaded = methods.load_locked_artifacts(
        manifest_path=manifest_path,
        policy_path=policy_path,
        census_path=census_path,
        admission_report_path=admission_report_path,
    )
    if tuple(loaded) != (manifest, policy, census, admission_report):
        raise ValueError("programmatic upstream artifacts differ from files")
    loaded_method_lock = methods.load_committed_method_lock(
        method_lock_path,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
        require_canonical_path=True,
    )
    if loaded_method_lock != method_lock:
        raise ValueError("programmatic method lock differs from canonical file")
    methods._require_file(
        core_manifest_path,
        methods.benchmark.PINNED_CORE_FILE_SHA256,
        name="original v1 core disclosure manifest",
    )

    loaded_partial = _load_mapping(
        partial_report_path,
        name="partial geometry method report",
    )
    if loaded_partial != partial_report:
        raise ValueError("programmatic partial report differs from file")
    loaded_completion_lock = load_committed_completion_lock(
        completion_lock_path,
        partial_report=partial_report,
        method_lock=method_lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
        partial_report_path=partial_report_path,
        method_lock_path=method_lock_path,
    )
    if loaded_completion_lock != completion_lock:
        raise ValueError("programmatic completion lock differs from canonical file")
    _validate_lock_sealed_partial_rows(
        partial_report,
        completion_lock,
        method_lock=method_lock,
        manifest=manifest,
    )
    claim = _attempt_claim
    if claim is None:
        claim = _prepare_attempt_claim(
            completion_lock=completion_lock,
            completion_lock_path=completion_lock_path,
            completion_row_output=completion_row_output,
            final_output=final_output,
            input_paths=input_paths,
            merge_only=merge_only,
        )
    else:
        claim = _validate_internal_claim(
            claim,
            completion_lock=completion_lock,
            completion_lock_path=completion_lock_path,
            completion_row_output=completion_row_output,
            final_output=final_output,
        )
    completion_row_output = claim.completion_row_output
    final_output = claim.final_output
    if claim.mode == "merge":
        try:
            return _merge_existing_completion_row(
                completion_row_output=completion_row_output,
                final_output=final_output,
                partial_report=partial_report,
                completion_lock=completion_lock,
                method_lock=method_lock,
                manifest=manifest,
            )
        finally:
            _release_attempt_claim(claim)
    if claim.mode == "recover":
        try:
            sealed_row = _seal_final_row(
                _failed_row_from_attempt_claim(completion_lock),
                completion_lock=completion_lock,
            )
            row_artifact = _completion_row_artifact(
                sealed_row,
                completion_lock=completion_lock,
                completion_elapsed_seconds=0.0,
            )
            validate_completion_row_artifact(
                row_artifact,
                completion_lock=completion_lock,
                method_lock=method_lock,
                partial_report=partial_report,
                public_record=manifest["records"][-1],
            )
            _atomic_write_new(completion_row_output, row_artifact)
        finally:
            _release_attempt_claim(claim)
        return _merge_existing_completion_row(
            completion_row_output=completion_row_output,
            final_output=final_output,
            partial_report=partial_report,
            completion_lock=completion_lock,
            method_lock=method_lock,
            manifest=manifest,
        )
    if claim.mode != "score" or claim.descriptor is None:
        raise ValueError("completion attempt claim mode differs")

    try:
        authorized_ids = [
            str(binding["record_id"])
            for binding in method_lock["authorization"]["admission_records"]
        ]
        if records is None or runtime is None:
            raise ValueError("scoring requires rehydrated records and a runtime")
        observed_ids = [str(record.record_id) for record in records]
        if (
            len(records) != EXPECTED_RECORDS
            or observed_ids != authorized_ids
            or len(set(observed_ids)) != EXPECTED_RECORDS
            or observed_ids[-1] != EXPECTED_REMAINING_RECORD_ID
        ):
            raise ValueError("completion requires the ordered rehydrated all16")
        for runtime_record, public_record in zip(records, manifest["records"]):
            methods.admission._validate_record(public_record, runtime_record)

        methods.verify_execution_runtime(
            runtime,
            locked_runtime=completion_lock["authorization"]["runtime"],
        )
        started = time.perf_counter()
        final_record = records[-1]
        final_public_record = manifest["records"][-1]
        final_binding = method_lock["authorization"]["admission_records"][-1]
        try:
            row = methods.evaluate_record(
                runtime,
                final_record,
                final_public_record,
                admission_binding=final_binding,
                warmup=int(completion_lock["authorization"]["warmup"]),
                repeats=int(completion_lock["authorization"]["repeats"]),
            )
            if row.get("record_id") != EXPECTED_REMAINING_RECORD_ID:
                raise ValueError("method engine returned an unauthorized record")
            sealed_row = _seal_final_row(row, completion_lock=completion_lock)
            elapsed = time.perf_counter() - started
            row_artifact = _completion_row_artifact(
                sealed_row,
                completion_lock=completion_lock,
                completion_elapsed_seconds=elapsed,
            )
            validate_completion_row_artifact(
                row_artifact,
                completion_lock=completion_lock,
                method_lock=method_lock,
                partial_report=partial_report,
                public_record=final_public_record,
            )
        except Exception as exc:
            row = _failed_row(EXPECTED_REMAINING_RECORD_ID, exc)
            sealed_row = _seal_final_row(row, completion_lock=completion_lock)
            elapsed = time.perf_counter() - started
            row_artifact = _completion_row_artifact(
                sealed_row,
                completion_lock=completion_lock,
                completion_elapsed_seconds=elapsed,
            )
            validate_completion_row_artifact(
                row_artifact,
                completion_lock=completion_lock,
                method_lock=method_lock,
                partial_report=partial_report,
                public_record=final_public_record,
            )
        _atomic_write_new(completion_row_output, row_artifact)
    finally:
        _release_attempt_claim(claim)

    row_file_sha256 = _sha256_file(completion_row_output)
    final_report = _build_final_report(
        partial_report,
        row_artifact,
        completion_lock=completion_lock,
        method_lock=method_lock,
        completion_row_file_sha256=row_file_sha256,
        completion_elapsed_seconds=elapsed,
    )
    validate_final_report(
        final_report,
        partial_report=partial_report,
        completion_row_artifact=row_artifact,
        completion_lock=completion_lock,
        method_lock=method_lock,
        manifest=manifest,
        public_record=final_public_record,
        completion_row_file_sha256=row_file_sha256,
    )
    _atomic_write_new(final_output, final_report)
    return final_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(methods.DEFAULT_MANIFEST))
    parser.add_argument("--policy-lock", default=str(methods.DEFAULT_POLICY_LOCK))
    parser.add_argument("--census", default=str(methods.DEFAULT_CENSUS))
    parser.add_argument(
        "--admission-report",
        default=str(methods.DEFAULT_ADMISSION_REPORT),
    )
    parser.add_argument("--method-lock", default=str(methods.DEFAULT_METHOD_LOCK))
    parser.add_argument(
        "--partial-report",
        default=str(DEFAULT_PARTIAL_REPORT),
    )
    parser.add_argument(
        "--completion-lock",
        default=str(DEFAULT_COMPLETION_LOCK),
    )
    parser.add_argument(
        "--core-manifest",
        default=str(methods.DEFAULT_CORE_MANIFEST),
    )
    parser.add_argument("--data-path")
    parser.add_argument(
        "--completion-row-out",
        default=str(DEFAULT_COMPLETION_ROW_OUTPUT),
    )
    parser.add_argument("--out", default=str(DEFAULT_FINAL_OUTPUT))
    parser.add_argument(
        "--freeze-completion-lock",
        action="store_true",
        help="freeze the exact partial report; performs no model scoring",
    )
    parser.add_argument(
        "--allow-one-record-completion",
        action="store_true",
        help="required with the committed lock for the one remaining score",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="merge an existing canonical sealed row without loading the model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    paths = {
        "manifest": Path(args.manifest),
        "policy": Path(args.policy_lock),
        "census": Path(args.census),
        "admission": Path(args.admission_report),
        "method": Path(args.method_lock),
        "partial": Path(args.partial_report),
        "completion": Path(args.completion_lock),
        "core": Path(args.core_manifest),
    }
    data_path = None if args.data_path is None else Path(args.data_path)
    try:
        manifest, policy, census, admission_report = (
            methods.load_locked_artifacts(
                manifest_path=paths["manifest"],
                policy_path=paths["policy"],
                census_path=paths["census"],
                admission_report_path=paths["admission"],
            )
        )
        method_lock = methods.load_committed_method_lock(
            paths["method"],
            manifest=manifest,
            policy=policy,
            census=census,
            admission_report=admission_report,
            require_canonical_path=True,
        )
        partial_report = _load_mapping(
            paths["partial"],
            name="partial geometry method report",
        )
        if args.freeze_completion_lock:
            if args.allow_one_record_completion:
                raise ValueError(
                    "lock freezing and scoring acknowledgement are separate"
                )
            if paths["completion"].resolve() != DEFAULT_COMPLETION_LOCK.resolve():
                raise PermissionError(
                    "completion lock must be frozen at its canonical path"
                )
            (lock_output,) = _validate_new_output_paths(
                (paths["completion"],),
                input_paths=(
                    paths["manifest"],
                    paths["policy"],
                    paths["census"],
                    paths["admission"],
                    paths["method"],
                    paths["partial"],
                    paths["core"],
                    data_path,
                ),
            )
            lock = freeze_completion_lock(
                partial_report,
                method_lock,
                manifest=manifest,
                policy=policy,
                census=census,
                admission_report=admission_report,
                partial_report_file_sha256=_sha256_file(paths["partial"]),
                method_lock_file_sha256=_sha256_file(paths["method"]),
            )
            _atomic_write_new(lock_output, lock)
            print(
                f"froze completion lock at {lock_output}; commit it before scoring",
                flush=True,
            )
            return 0

        completion_lock = load_committed_completion_lock(
            paths["completion"],
            partial_report=partial_report,
            method_lock=method_lock,
            manifest=manifest,
            policy=policy,
            census=census,
            admission_report=admission_report,
            partial_report_path=paths["partial"],
            method_lock_path=paths["method"],
        )
        if not args.merge_only:
            authorize_completion(
                completion_lock,
                explicit_acknowledgement=args.allow_one_record_completion,
            )
        attempt_claim = _prepare_attempt_claim(
            completion_lock=completion_lock,
            completion_lock_path=paths["completion"],
            completion_row_output=Path(args.completion_row_out),
            final_output=Path(args.out),
            input_paths=(*paths.values(), data_path),
            merge_only=args.merge_only,
        )
        row_output = attempt_claim.completion_row_output
        final_output = attempt_claim.final_output
    except (OSError, PermissionError, ValueError) as exc:
        parser.error(str(exc))

    if attempt_claim.mode != "score":
        try:
            report = run_completion(
                manifest,
                policy,
                census,
                admission_report,
                method_lock,
                partial_report,
                completion_lock,
                None,
                None,
                manifest_path=paths["manifest"],
                policy_path=paths["policy"],
                census_path=paths["census"],
                admission_report_path=paths["admission"],
                method_lock_path=paths["method"],
                partial_report_path=paths["partial"],
                completion_lock_path=paths["completion"],
                core_manifest_path=paths["core"],
                data_path=data_path,
                completion_row_output=row_output,
                final_output=final_output,
                explicit_acknowledgement=args.allow_one_record_completion,
                merge_only=args.merge_only,
                _attempt_claim=attempt_claim,
            )
        except (
            OSError,
            PermissionError,
            RuntimeError,
            ValueError,
            methods.benchmark.ManifestError,
        ) as exc:
            parser.error(str(exc))
        finally:
            _release_attempt_claim(attempt_claim)
        print(
            f"validated existing sealed completion row at {row_output}",
            flush=True,
        )
        print(f"wrote merged all16 report to {final_output}", flush=True)
        return 0 if report["status"] == "completed" else 1

    try:
        rows = methods.benchmark.base.load_pinned_longmemeval_rows(args.data_path)
        runtime = methods.admission._make_runtime(
            device=str(completion_lock["authorization"]["runtime"]["device"])
        )
        runtime.ensure_loaded()
        corrected = methods.benchmark.rehydrate_manifest(
            manifest,
            rows,
            runtime.tokenizer,
            core_manifest=paths["core"],
        )
        records = tuple(item.runtime for item in corrected)
        report = run_completion(
            manifest,
            policy,
            census,
            admission_report,
            method_lock,
            partial_report,
            completion_lock,
            records,
            runtime,
            manifest_path=paths["manifest"],
            policy_path=paths["policy"],
            census_path=paths["census"],
            admission_report_path=paths["admission"],
            method_lock_path=paths["method"],
            partial_report_path=paths["partial"],
            completion_lock_path=paths["completion"],
            core_manifest_path=paths["core"],
            data_path=data_path,
            completion_row_output=row_output,
            final_output=final_output,
            explicit_acknowledgement=args.allow_one_record_completion,
            merge_only=False,
            _attempt_claim=attempt_claim,
        )
    except (
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
        methods.benchmark.ManifestError,
    ) as exc:
        parser.error(str(exc))
    finally:
        _release_attempt_claim(attempt_claim)
    print(f"wrote sealed completion row to {row_output}", flush=True)
    print(f"wrote merged all16 report to {final_output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed finalization of the interrupted geometry-method run.

This module does not provide a general resume path.  A separate, canonical,
committed partial lock must first freeze the exact externally interrupted
15-row checkpoint.  Finalization then rehydrates the committed cohort, scores
only the final authorized record, merges it with the locked rows, and writes a
new source-free report without replacing any input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_geometry_methods as methods


hardened = methods.hardened
benchmark = methods.benchmark

PARTIAL_LOCK_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-methods-partial-lock-v1"
)
PARTIAL_LOCK_SCHEMA_VERSION = 1
PARTIAL_LOCK_STATUS = (
    "frozen-after-external-interruption-before-record-16-finalization"
)
FINALIZATION_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-methods-finalization-v1"
)

EXPECTED_AUTHORIZED_RECORDS = 16
EXPECTED_LOCKED_ROWS = 15
EXPECTED_METHOD_IMPLEMENTATION_FILES = 41

PARTIAL_REPORT_REPOSITORY_PATH = (
    "outputs/gemma_sv_rag/longmemeval_chat_geometry_methods_v1.json"
)
PARTIAL_LOCK_REPOSITORY_PATH = (
    "gemma_sv/benchmarks/"
    "longmemeval_chat_geometry_methods_partial_lock_v1.json"
)
METHOD_LOCK_REPOSITORY_PATH = (
    "gemma_sv/benchmarks/"
    "longmemeval_chat_geometry_method_authorization_v1.json"
)
FINAL_OUTPUT_REPOSITORY_PATH = (
    "outputs/gemma_sv_rag/"
    "longmemeval_chat_geometry_methods_finalized_v1.json"
)
FINALIZER_REPOSITORY_PATH = (
    "gemma_sv/finalize_longmemeval_chat_geometry_methods.py"
)

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
DEFAULT_PARTIAL_REPORT = WORKSPACE / PARTIAL_REPORT_REPOSITORY_PATH
DEFAULT_PARTIAL_LOCK = WORKSPACE / PARTIAL_LOCK_REPOSITORY_PATH
DEFAULT_OUTPUT = WORKSPACE / FINAL_OUTPUT_REPOSITORY_PATH

PARTIAL_REPORT_FILE_SHA256 = (
    "0358767e896f4dcfcad6d9bef108ecd2ce7076b06fab584bfe319c519bf1456a"
)
PARTIAL_REPORT_PAYLOAD_SHA256 = (
    "bcec7a147a2c21e6c2dca36760a5420459916ff7d1d89c96b8aafb36fd2e7e7e"
)

INTERRUPTION_DISCLOSURE = {
    "interruption_kind": "external_process_abort",
    "partial_report_status": "running",
    "checkpoint_was_atomically_written_by_original_evaluator": True,
    "attempted_records_at_interruption": EXPECTED_LOCKED_ROWS,
    "not_attempted_records_at_interruption": 1,
    "missing_record_is_final_authorized_record": True,
    "missing_record_was_not_scored_in_partial_process": True,
    "completed_prefix_was_not_selected_or_replaced": True,
    "finalizer_scores_only_the_missing_record": True,
}

_PROTECTED_INPUTS = frozenset(
    {
        *methods._PROTECTED_INPUTS,
        methods.DEFAULT_MANIFEST,
        methods.DEFAULT_POLICY_LOCK,
        methods.DEFAULT_CENSUS,
        methods.DEFAULT_CORE_MANIFEST,
        methods.DEFAULT_ADMISSION_REPORT,
        methods.DEFAULT_METHOD_LOCK,
        DEFAULT_PARTIAL_REPORT,
        DEFAULT_PARTIAL_LOCK,
        Path(__file__).resolve(),
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
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_mapping_and_file_sha256(
    path: str | Path,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value, file_sha256


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _require_canonical_path(
    path: str | Path,
    canonical: str | Path,
    *,
    name: str,
) -> Path:
    candidate = Path(path).expanduser()
    if _resolved(candidate) != _resolved(canonical):
        raise PermissionError(f"{name} requires its canonical committed path")
    return candidate


def _validate_output_path(
    output: str | Path,
    *,
    input_paths: Sequence[str | Path | None],
) -> Path:
    output_path = Path(output).expanduser()
    resolved_output = _resolved(output_path)
    protected = {
        Path(path).expanduser()
        for path in (*input_paths, *_PROTECTED_INPUTS)
        if path is not None
    }
    for protected_path in protected:
        same_path = resolved_output == _resolved(protected_path)
        same_file = False
        if output_path.exists() and protected_path.exists():
            try:
                same_file = os.path.samefile(output_path, protected_path)
            except OSError:
                pass
        if same_path or same_file:
            raise ValueError("final output aliases a protected finalizer input")
    return resolved_output


def _method_contract_analysis(
    method_lock: Mapping[str, Any],
) -> dict[str, Any]:
    methods._assert_source_free(method_lock)
    hardened._assert_finite_json(method_lock, path="method_lock")

    integrity = method_lock.get("integrity")
    unsigned = dict(method_lock)
    unsigned.pop("integrity", None)
    if (
        method_lock.get("schema") != methods.METHOD_LOCK_SCHEMA
        or method_lock.get("status") != methods.METHOD_LOCK_STATUS
        or not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise ValueError("geometry method authorization lock is invalid")

    artifacts = method_lock.get("artifacts")
    authorization = method_lock.get("authorization")
    implementation = method_lock.get("implementation")
    if (
        not isinstance(artifacts, Mapping)
        or not isinstance(authorization, Mapping)
        or not isinstance(implementation, Mapping)
    ):
        raise ValueError("geometry method lock contract is incomplete")

    implementation_files = {
        str(name): value
        for name, value in implementation.items()
        if name != "contract_sha256"
    }
    implementation_contract_sha256 = implementation.get("contract_sha256")
    if (
        len(implementation_files) != EXPECTED_METHOD_IMPLEMENTATION_FILES
        or any(
            not isinstance(name, str) or not _is_sha256(value)
            for name, value in implementation_files.items()
        )
        or implementation_contract_sha256
        != _payload_sha256(implementation_files)
        or method_lock.get("implementation_contract_sha256")
        != implementation_contract_sha256
    ):
        raise ValueError("geometry method 41-file implementation contract differs")

    admission_records = authorization.get("admission_records")
    condition_ids = authorization.get("condition_ids")
    if (
        not isinstance(admission_records, list)
        or len(admission_records) != EXPECTED_AUTHORIZED_RECORDS
        or not isinstance(condition_ids, list)
        or condition_ids != list(methods.CONDITION_IDS)
        or int(authorization.get("frozen_records", -1))
        != EXPECTED_AUTHORIZED_RECORDS
    ):
        raise ValueError("geometry method authorization population differs")
    record_ids = [
        str(binding.get("record_id") or "")
        for binding in admission_records
        if isinstance(binding, Mapping)
    ]
    if (
        len(record_ids) != EXPECTED_AUTHORIZED_RECORDS
        or len(set(record_ids)) != EXPECTED_AUTHORIZED_RECORDS
        or any(not record_id for record_id in record_ids)
    ):
        raise ValueError("geometry method authorization record IDs differ")

    return {
        "artifacts": dict(artifacts),
        "authorization": dict(authorization),
        "implementation": dict(implementation),
        "implementation_file_count": len(implementation_files),
        "implementation_contract_sha256": implementation_contract_sha256,
        "method_lock_integrity_sha256": integrity["sha256"],
        "record_ids": record_ids,
        "admission_records": admission_records,
        "condition_ids": condition_ids,
    }


def _validate_failed_condition(
    condition: Mapping[str, Any],
    *,
    condition_id: str,
) -> None:
    if (
        condition.get("condition_id") != condition_id
        or condition.get("status") != "failed"
        or not isinstance(condition.get("error_type"), str)
        or not condition.get("error_type")
        or condition.get("error_message_redacted") is not True
        or not _is_sha256(condition.get("error_message_sha256"))
    ):
        raise ValueError("partial row failed-condition disclosure differs")


def _validate_completed_condition(
    condition: Mapping[str, Any],
    *,
    condition_id: str,
) -> None:
    deleted = condition.get("deleted_target_quality")
    retained = condition.get("retained_quality")
    if (
        condition.get("condition_id") != condition_id
        or condition.get("status") != "completed"
        or not isinstance(condition.get("timing"), Mapping)
        or not isinstance(condition.get("storage"), Mapping)
        or not isinstance(deleted, Mapping)
        or not isinstance(deleted.get("probes"), list)
        or len(deleted["probes"]) != 1
        or not isinstance(retained, Mapping)
    ):
        raise ValueError("partial row completed-condition metrics are incomplete")


def validate_hardened_method_row(
    row: Mapping[str, Any],
    *,
    expected_id: str,
    binding: Mapping[str, Any],
    condition_ids: Sequence[str],
    public_record: Mapping[str, Any] | None = None,
) -> None:
    """Validate a terminal method row, including declared condition failures.

    The older audit-only resume validator accepts only all-success rows.  The
    interrupted geometry report legitimately contains terminal
    ``completed_with_failures`` rows, so this validator applies the same
    source-free, finite-metric, authorization, method-matrix, immutability, and
    solver-certificate checks while explicitly reconciling failed conditions.
    """

    methods._assert_source_free(row)
    hardened._assert_finite_json(row, path=f"row[{expected_id}]")

    status = row.get("status")
    if (
        row.get("record_id") != expected_id
        or status not in {"completed", "completed_with_failures"}
        or row.get("contains_source_text") not in {None, False}
        or row.get("admission_binding") != binding
        or row.get("predeclared_joint_admitted")
        is not bool(binding.get("joint_admitted"))
        or row.get("source_fixed_c_feasible") is not True
        or binding.get("source_fixed_c_feasible") is not True
        or row.get("record_seed") != hardened._record_seed(expected_id)
    ):
        raise ValueError("partial method row identity or authorization differs")

    probes = row.get("probes")
    if (
        not isinstance(probes, list)
        or [probe.get("probe_id") for probe in probes if isinstance(probe, Mapping)]
        != ["target_current", "retained"]
        or len(probes) != 2
        or any(
            not isinstance(probe, Mapping)
            or int(probe.get("target_token_count", 0)) < 1
            or not _is_sha256(probe.get("target_token_ids_sha256"))
            for probe in probes
        )
        or not isinstance(row.get("shared_original_prefill"), Mapping)
    ):
        raise ValueError("partial method row probe contract differs")

    immutability = row.get("source_state_immutability")
    if (
        not isinstance(immutability, Mapping)
        or immutability.get("unchanged") is not True
        or immutability.get("shape_signature_unchanged") is not True
        or not _is_sha256(immutability.get("before_sha256"))
        or immutability.get("before_sha256") != immutability.get("after_sha256")
    ):
        raise ValueError("partial method row lacks verified source immutability")

    geometry = row.get("geometry_binding")
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("fixed_c_feasible") is not True
        or geometry.get("local_window_safe") is not True
        or not _is_sha256(geometry.get("manifest_record_integrity_sha256"))
    ):
        raise ValueError("partial method row geometry binding differs")
    if public_record is not None and geometry.get(
        "manifest_record_integrity_sha256"
    ) != (public_record.get("record_integrity") or {}).get("sha256"):
        raise ValueError("partial method row manifest binding differs")

    conditions = row.get("conditions")
    if (
        not isinstance(conditions, Mapping)
        or set(conditions) != set(condition_ids)
    ):
        raise ValueError("partial method row condition matrix is incomplete")

    failed_condition_ids: list[str] = []
    for condition_id in condition_ids:
        condition = conditions.get(condition_id)
        if not isinstance(condition, Mapping):
            raise ValueError("partial method row condition is not an object")
        condition_status = condition.get("status")
        if condition_status == "completed":
            _validate_completed_condition(condition, condition_id=condition_id)
        elif condition_status == "failed":
            _validate_failed_condition(condition, condition_id=condition_id)
            failed_condition_ids.append(condition_id)
        else:
            raise ValueError("partial method row condition status is not terminal")

    expected_status = (
        "completed_with_failures" if failed_condition_ids else "completed"
    )
    if (
        row.get("method_failures") != failed_condition_ids
        or status != expected_status
    ):
        raise ValueError("partial method row failure accounting differs")

    token_alias = conditions[methods.TOKEN_ROW_DIAGNOSTIC_ID]
    if token_alias.get("status") == "completed":
        token_semantics = token_alias.get("semantics") or {}
        if (
            token_semantics.get("zero_cost_alias") is not True
            or token_semantics.get("alias_of") != methods.FRESH_RAW_OMISSION_ID
            or (token_alias.get("timing") or {}).get("zero_cost_alias") is not True
        ):
            raise ValueError("partial token-row diagnostic contract differs")

    exact = conditions[methods.EXACT_DECREMENT_ID]
    if exact.get("status") == "completed":
        exact_semantics = exact.get("semantics") or {}
        if (
            exact_semantics.get("full_repack_fallback") is not False
            or exact_semantics.get("executed_method")
            not in {
                "incremental_float64_fixed_c_decrement",
                "fixed_c_refit_fallback",
            }
        ):
            raise ValueError("partial exact-decrement execution label differs")

    proxy = conditions[methods.FP32_PROXY_ID]
    if proxy.get("status") == "completed" and (
        (proxy.get("semantics") or {}).get("full_repack_fallback") is not False
    ):
        raise ValueError("partial FP32 proxy fallback label differs")

    certificate = row.get("solver_certificate")
    if (
        not isinstance(certificate, Mapping)
        or certificate.get("status") != "completed"
    ):
        raise ValueError("partial exact/refit solver certificate differs")

    reconciliation = methods._reconcile_row(row)
    if (
        reconciliation["declared_status_consistent"] is not True
        or reconciliation["method_failures_consistent"] is not True
        or reconciliation["source_state_immutability_success"] is not True
        or reconciliation["certificate_success"] is not True
        or reconciliation["condition_ids_exact"] is not True
    ):
        raise ValueError("partial method row does not reconcile")


def validate_partial_report(
    partial_report: Mapping[str, Any],
    *,
    partial_report_file_sha256: str,
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact interrupted report and return its lock material."""

    partial_report_payload_sha256 = _payload_sha256(partial_report)
    if (
        partial_report_file_sha256 != PARTIAL_REPORT_FILE_SHA256
        or partial_report_payload_sha256 != PARTIAL_REPORT_PAYLOAD_SHA256
    ):
        raise ValueError("partial report is not the exact interrupted artifact")

    method = _method_contract_analysis(method_lock)
    methods._assert_source_free(partial_report)
    hardened._assert_finite_json(partial_report, path="partial_report")

    if (
        partial_report.get("schema") != methods.SCHEMA
        or partial_report.get("schema_version") != methods.SCHEMA_VERSION
        or partial_report.get("status") != "running"
        or partial_report.get("contains_source_text") is not False
        or partial_report.get("contains_full_vocabulary_vectors") is not False
        or partial_report.get("official_longmemeval_leaderboard_score") is not False
        or partial_report.get("method_lock_integrity_sha256")
        != method["method_lock_integrity_sha256"]
        or partial_report.get("implementation") != method["implementation"]
    ):
        raise ValueError("partial report top-level method contract differs")

    config = partial_report.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("partial report config is missing")
    locked_runtime = method["authorization"].get("runtime")
    if (
        not isinstance(locked_runtime, Mapping)
        or any(config.get(key) != value for key, value in locked_runtime.items())
        or type(config.get("warmup")) is not int
        or type(config.get("repeats")) is not int
        or config["warmup"] < 0
        or config["repeats"] < 1
    ):
        raise ValueError("partial report runtime contract differs")

    expected_base = methods._base_report(
        method_lock,
        device=str(locked_runtime["device"]),
        warmup=int(config["warmup"]),
        repeats=int(config["repeats"]),
    )
    if methods._validate_resume(partial_report, expected_base) != (
        EXPECTED_AUTHORIZED_RECORDS
    ):
        raise ValueError("partial report does not contain ordered all16 slots")

    rows = partial_report.get("records")
    if not isinstance(rows, list) or len(rows) != EXPECTED_AUTHORIZED_RECORDS:
        raise ValueError("partial report records differ from all16")
    observed_ids = [
        str(row.get("record_id") or "") if isinstance(row, Mapping) else ""
        for row in rows
    ]
    if observed_ids != method["record_ids"]:
        raise ValueError("partial report row order differs from authorization")

    public_records: list[Mapping[str, Any] | None]
    if manifest is None:
        public_records = [None] * EXPECTED_AUTHORIZED_RECORDS
    else:
        manifest_rows = manifest.get("records")
        if (
            not isinstance(manifest_rows, list)
            or len(manifest_rows) != EXPECTED_AUTHORIZED_RECORDS
            or [
                str(row.get("record_id") or "")
                for row in manifest_rows
                if isinstance(row, Mapping)
            ]
            != method["record_ids"]
        ):
            raise ValueError("manifest rows differ from method authorization")
        public_records = list(manifest_rows)

    completed_rows = rows[:EXPECTED_LOCKED_ROWS]
    for index, (row, binding, public) in enumerate(
        zip(
            completed_rows,
            method["admission_records"][:EXPECTED_LOCKED_ROWS],
            public_records[:EXPECTED_LOCKED_ROWS],
        )
    ):
        if not isinstance(row, Mapping):
            raise ValueError(f"partial row {index + 1} is not an object")
        validate_hardened_method_row(
            row,
            expected_id=method["record_ids"][index],
            binding=binding,
            condition_ids=method["condition_ids"],
            public_record=public,
        )

    missing_row = rows[-1]
    expected_missing = {
        "record_id": method["record_ids"][-1],
        "status": "not_attempted",
        "contains_source_text": False,
    }
    if missing_row != expected_missing:
        raise ValueError("only the final authorized row may be not_attempted")

    expected_summary = methods.summarize_records(rows, method_lock=method_lock)
    if partial_report.get("summary") != expected_summary:
        raise ValueError("partial report summary does not recompute exactly")
    denominators = expected_summary.get("denominators") or {}
    if (
        int(denominators.get("authorized_records", -1))
        != EXPECTED_AUTHORIZED_RECORDS
        or int(denominators.get("attempted_records", -1))
        != EXPECTED_LOCKED_ROWS
        or int(denominators.get("not_attempted_records", -1)) != 1
        or partial_report.get("resume_requested") is not False
        or int(partial_report.get("reused_completed_resume_records", -1)) != 0
        or int(partial_report.get("discarded_resume_rows", -1)) != 0
    ):
        raise ValueError("partial report interruption accounting differs")

    row_bindings = [
        {
            "authorized_position": index + 1,
            "record_id": row["record_id"],
            "status": row["status"],
            "payload_sha256": _payload_sha256(row),
        }
        for index, row in enumerate(completed_rows)
    ]
    return {
        "partial_report_payload_sha256": partial_report_payload_sha256,
        "completed_rows": row_bindings,
        "missing_record_id": expected_missing["record_id"],
        "missing_row_payload_sha256": _payload_sha256(missing_row),
        "method": method,
    }


def freeze_partial_lock(
    partial_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    *,
    partial_report_file_sha256: str = PARTIAL_REPORT_FILE_SHA256,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the exact partial checkpoint before any finalization scoring."""

    analysis = validate_partial_report(
        partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock=method_lock,
        manifest=manifest,
    )
    method = analysis["method"]
    lock = {
        "schema": PARTIAL_LOCK_SCHEMA,
        "schema_version": PARTIAL_LOCK_SCHEMA_VERSION,
        "status": PARTIAL_LOCK_STATUS,
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "partial_report": {
            "canonical_repository_path": PARTIAL_REPORT_REPOSITORY_PATH,
            "file_sha256": partial_report_file_sha256,
            "payload_sha256": analysis["partial_report_payload_sha256"],
            "schema": partial_report["schema"],
            "status": partial_report["status"],
        },
        "method_contract": {
            "canonical_authorization_lock_path": METHOD_LOCK_REPOSITORY_PATH,
            "authorization": copy.deepcopy(method["authorization"]),
            "authorization_payload_sha256": _payload_sha256(
                method["authorization"]
            ),
            "artifacts": copy.deepcopy(method["artifacts"]),
            "artifacts_payload_sha256": _payload_sha256(method["artifacts"]),
            "implementation": copy.deepcopy(method["implementation"]),
            "implementation_file_count": method["implementation_file_count"],
            "implementation_contract_sha256": method[
                "implementation_contract_sha256"
            ],
            "method_lock_integrity_sha256": method[
                "method_lock_integrity_sha256"
            ],
        },
        "completed_rows": copy.deepcopy(analysis["completed_rows"]),
        "missing_record": {
            "authorized_position": EXPECTED_AUTHORIZED_RECORDS,
            "record_id": analysis["missing_record_id"],
            "partial_status": "not_attempted",
            "partial_row_payload_sha256": analysis[
                "missing_row_payload_sha256"
            ],
        },
        "interruption_disclosure": copy.deepcopy(INTERRUPTION_DISCLOSURE),
        "execution_authorization": {
            "canonical_committed_partial_lock_required": True,
            "exact_boolean_true_acknowledgement_required": True,
            "locked_partial_rows_merged": EXPECTED_LOCKED_ROWS,
            "new_model_scored_records": 1,
            "only_missing_record_may_be_scored": True,
            "generic_resume_allowed": False,
            "resume_output_rows_reused": False,
            "partial_report_must_remain_unchanged": True,
            "final_output_must_be_distinct_from_every_input": True,
        },
        "finalizer": {
            "canonical_repository_path": FINALIZER_REPOSITORY_PATH,
            "file_sha256": _sha256_file(Path(__file__).resolve()),
        },
    }
    lock["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(lock),
    }
    methods._assert_source_free(lock)
    hardened._assert_finite_json(lock, path="partial_lock")
    return lock


def validate_partial_lock(
    partial_lock: Mapping[str, Any],
    *,
    partial_report: Mapping[str, Any],
    partial_report_file_sha256: str,
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> None:
    methods._assert_source_free(partial_lock)
    hardened._assert_finite_json(partial_lock, path="partial_lock")
    integrity = partial_lock.get("integrity")
    unsigned = dict(partial_lock)
    unsigned.pop("integrity", None)
    if (
        partial_lock.get("schema") != PARTIAL_LOCK_SCHEMA
        or partial_lock.get("schema_version") != PARTIAL_LOCK_SCHEMA_VERSION
        or partial_lock.get("status") != PARTIAL_LOCK_STATUS
        or not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise ValueError("partial finalization lock integrity differs")

    expected = freeze_partial_lock(
        partial_report,
        method_lock,
        partial_report_file_sha256=partial_report_file_sha256,
        manifest=manifest,
    )
    if dict(partial_lock) != expected:
        raise ValueError("partial lock differs from the exact interrupted run")


def load_exact_partial_report(
    path: str | Path = DEFAULT_PARTIAL_REPORT,
) -> tuple[dict[str, Any], str]:
    report_path = _require_canonical_path(
        path,
        DEFAULT_PARTIAL_REPORT,
        name="partial geometry method report",
    )
    report, file_sha256 = _load_mapping_and_file_sha256(
        report_path,
        name="partial geometry method report",
    )
    if file_sha256 != PARTIAL_REPORT_FILE_SHA256:
        raise ValueError("partial report is not the exact interrupted artifact")
    return report, file_sha256


def load_committed_partial_lock(
    path: str | Path,
    *,
    partial_report: Mapping[str, Any],
    partial_report_file_sha256: str,
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    require_canonical_path: bool = True,
) -> dict[str, Any]:
    lock_path = Path(path).expanduser()
    if require_canonical_path:
        _require_canonical_path(
            lock_path,
            DEFAULT_PARTIAL_LOCK,
            name="geometry methods partial lock",
        )
    lock, _ = _load_mapping_and_file_sha256(
        lock_path,
        name="geometry methods partial lock",
    )
    validate_partial_lock(
        lock,
        partial_report=partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock=method_lock,
        manifest=manifest,
    )
    return lock


def authorize_finalization(
    partial_lock: Mapping[str, Any],
    *,
    partial_report: Mapping[str, Any],
    partial_report_file_sha256: str,
    method_lock: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    explicit_acknowledgement: Any,
) -> str:
    validate_partial_lock(
        partial_lock,
        partial_report=partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock=method_lock,
        manifest=manifest,
    )
    if explicit_acknowledgement is not True:
        raise PermissionError(
            "geometry method finalization acknowledgement must be exactly True"
        )
    return str(partial_lock["missing_record"]["record_id"])


def _validate_rehydrated_cohort(
    records: Sequence[Any],
    *,
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
) -> None:
    authorized = method_lock["authorization"]["admission_records"]
    expected_ids = [str(binding["record_id"]) for binding in authorized]
    observed_ids = [str(getattr(record, "record_id", "")) for record in records]
    if (
        len(records) != EXPECTED_AUTHORIZED_RECORDS
        or observed_ids != expected_ids
        or len(set(observed_ids)) != EXPECTED_AUTHORIZED_RECORDS
    ):
        raise ValueError("finalizer requires the ordered rehydrated all16 cohort")
    public_records = manifest.get("records")
    if not isinstance(public_records, list) or len(public_records) != len(records):
        raise ValueError("finalizer manifest records differ from all16")
    for record, public_record in zip(records, public_records):
        methods.admission._validate_record(public_record, record)


def _failed_record(record_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "status": "failed",
        "contains_source_text": False,
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": benchmark.base.text_sha256(str(exc)),
    }


def _validate_failed_record(row: Mapping[str, Any], *, expected_id: str) -> None:
    if (
        row.get("record_id") != expected_id
        or row.get("status") != "failed"
        or row.get("contains_source_text") is not False
        or not isinstance(row.get("error_type"), str)
        or not row.get("error_type")
        or row.get("error_message_redacted") is not True
        or not _is_sha256(row.get("error_message_sha256"))
    ):
        raise ValueError("failed final record disclosure differs")
    methods._assert_source_free(row)
    hardened._assert_finite_json(row, path="failed_final_record")


def _validate_final_accounting(
    report: Mapping[str, Any],
    *,
    method_lock: Mapping[str, Any],
) -> None:
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("final report records must be a list")
    expected_ids = [
        str(binding["record_id"])
        for binding in method_lock["authorization"]["admission_records"]
    ]
    if (
        len(records) != EXPECTED_AUTHORIZED_RECORDS
        or [str(row.get("record_id") or "") for row in records] != expected_ids
        or any(row.get("status") == "not_attempted" for row in records)
    ):
        raise ValueError("final report does not contain ordered attempted all16")

    expected_summary = methods.summarize_records(records, method_lock=method_lock)
    if report.get("summary") != expected_summary:
        raise ValueError("final report summary does not recompute exactly")
    denominators = expected_summary.get("denominators") or {}
    if (
        int(denominators.get("frozen_records", -1))
        != EXPECTED_AUTHORIZED_RECORDS
        or int(denominators.get("authorized_records", -1))
        != EXPECTED_AUTHORIZED_RECORDS
        or int(denominators.get("attempted_records", -1))
        != EXPECTED_AUTHORIZED_RECORDS
        or int(denominators.get("not_attempted_records", -1)) != 0
        or int(denominators.get("fully_completed_records", -1))
        + int(denominators.get("record_failures", -1))
        != EXPECTED_AUTHORIZED_RECORDS
    ):
        raise ValueError("final report denominators are contradictory")

    for condition_summary in expected_summary["conditions"].values():
        for population, authorized in (
            ("all_frozen_records", EXPECTED_AUTHORIZED_RECORDS),
            ("predeclared_joint10_subset", methods.EXPECTED_JOINT_ADMITTED),
        ):
            accounting = condition_summary[population]
            if (
                int(accounting.get("authorized_records", -1)) != authorized
                or int(accounting.get("attempted_records", -1)) != authorized
                or int(accounting.get("not_attempted_records", -1)) != 0
                or int(accounting.get("accounted_records", -1)) != authorized
                or int(accounting.get("unaccounted_records", -1)) != 0
                or accounting.get("fully_accounted") is not True
            ):
                raise ValueError("final per-condition accounting is incomplete")

    certificate = expected_summary["exact_refit_certificate"]
    for population, authorized in (
        ("all_frozen_records", EXPECTED_AUTHORIZED_RECORDS),
        ("predeclared_joint10_subset", methods.EXPECTED_JOINT_ADMITTED),
    ):
        accounting = certificate[population]
        if (
            int(accounting.get("authorized_records", -1)) != authorized
            or int(accounting.get("attempted_records", -1)) != authorized
            or int(accounting.get("not_attempted_records", -1)) != 0
            or int(accounting.get("accounted_records", -1)) != authorized
            or int(accounting.get("unaccounted_records", -1)) != 0
            or accounting.get("fully_accounted") is not True
        ):
            raise ValueError("final certificate accounting is incomplete")

    expected_status = (
        "completed"
        if denominators["fully_completed_records"] == EXPECTED_AUTHORIZED_RECORDS
        and denominators["record_failures"] == 0
        else "completed_with_record_failures"
    )
    if report.get("status") != expected_status:
        raise ValueError("final report status contradicts its accounting")


def _build_final_report(
    partial_report: Mapping[str, Any],
    *,
    partial_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    final_row: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    rows = copy.deepcopy(list(partial_report["records"][:EXPECTED_LOCKED_ROWS]))
    rows.append(copy.deepcopy(dict(final_row)))
    result = copy.deepcopy(dict(partial_report))
    result["records"] = rows
    result["summary"] = methods.summarize_records(rows, method_lock=method_lock)
    denominators = result["summary"]["denominators"]
    result["status"] = (
        "completed"
        if denominators["fully_completed_records"] == EXPECTED_AUTHORIZED_RECORDS
        and denominators["record_failures"] == 0
        else "completed_with_record_failures"
    )
    result["elapsed_seconds_finalization_process"] = float(elapsed_seconds)
    method_protocol = copy.deepcopy(dict(result.get("method_protocol") or {}))
    method_protocol["external_abort_finalization"] = {
        "generic_resume": False,
        "resume_output_rows_reused": 0,
        "locked_partial_prefix_rows_merged": EXPECTED_LOCKED_ROWS,
        "newly_scored_records": 1,
        "only_missing_final_record_scored": True,
    }
    result["method_protocol"] = method_protocol
    result["finalization"] = {
        "schema": FINALIZATION_SCHEMA,
        "partial_lock_integrity_sha256": partial_lock["integrity"]["sha256"],
        "partial_report_file_sha256": partial_lock["partial_report"][
            "file_sha256"
        ],
        "partial_report_payload_sha256": partial_lock["partial_report"][
            "payload_sha256"
        ],
        "interruption_disclosure": copy.deepcopy(
            partial_lock["interruption_disclosure"]
        ),
        "locked_partial_rows_merged": EXPECTED_LOCKED_ROWS,
        "newly_executed_records": 1,
        "newly_executed_record_ids": [
            partial_lock["missing_record"]["record_id"]
        ],
        "resume_requested": False,
        "resume_output_rows_reused": 0,
        "summary_recomputed_over_ordered_all16": True,
        "source_partial_report_replaced": False,
    }
    _validate_final_accounting(result, method_lock=method_lock)
    methods._assert_source_free(result)
    hardened._assert_finite_json(result, path="final_report")
    return result


def _finalize_rehydrated_records(
    *,
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_report: Mapping[str, Any],
    partial_report_file_sha256: str,
    partial_lock: Mapping[str, Any],
    records: Sequence[Any],
    runtime: Any,
    explicit_acknowledgement: Any,
) -> dict[str, Any]:
    """Execute the one-record operation after the caller secures disk inputs."""

    missing_id = authorize_finalization(
        partial_lock,
        partial_report=partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock=method_lock,
        manifest=manifest,
        explicit_acknowledgement=explicit_acknowledgement,
    )
    _validate_rehydrated_cohort(
        records,
        manifest=manifest,
        method_lock=method_lock,
    )
    if records[-1].record_id != missing_id:
        raise ValueError("partial lock missing ID is not rehydrated record16")

    locked_runtime = method_lock["authorization"]["runtime"]
    methods.verify_execution_runtime(runtime, locked_runtime=locked_runtime)
    binding = method_lock["authorization"]["admission_records"][-1]
    public_record = manifest["records"][-1]
    config = partial_report["config"]

    started = time.perf_counter()
    try:
        final_row = methods.evaluate_record(
            runtime,
            records[-1],
            public_record,
            admission_binding=binding,
            warmup=int(config["warmup"]),
            repeats=int(config["repeats"]),
        )
    except Exception as exc:
        final_row = _failed_record(missing_id, exc)

    if final_row.get("status") in {"completed", "completed_with_failures"}:
        validate_hardened_method_row(
            final_row,
            expected_id=missing_id,
            binding=binding,
            condition_ids=method_lock["authorization"]["condition_ids"],
            public_record=public_record,
        )
    else:
        _validate_failed_record(final_row, expected_id=missing_id)

    return _build_final_report(
        partial_report,
        partial_lock=partial_lock,
        method_lock=method_lock,
        final_row=final_row,
        elapsed_seconds=time.perf_counter() - started,
    )


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically install a new JSON file without replacing an existing path."""

    methods._assert_source_free(payload)
    hardened._assert_finite_json(payload, path="atomic_output")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_method_inputs(
    *,
    manifest_path: str | Path,
    policy_path: str | Path,
    census_path: str | Path,
    admission_report_path: str | Path,
    method_lock_path: str | Path,
    core_manifest_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    manifest, policy, census, admission_report = methods.load_locked_artifacts(
        manifest_path=manifest_path,
        policy_path=policy_path,
        census_path=census_path,
        admission_report_path=admission_report_path,
    )
    method_lock = methods.load_committed_method_lock(
        method_lock_path,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
        require_canonical_path=True,
    )
    methods._require_file(
        core_manifest_path,
        benchmark.PINNED_CORE_FILE_SHA256,
        name="original v1 core disclosure manifest",
    )
    return manifest, policy, census, admission_report, method_lock


def _load_finalization_inputs(
    *,
    manifest_path: str | Path,
    policy_path: str | Path,
    census_path: str | Path,
    admission_report_path: str | Path,
    method_lock_path: str | Path,
    core_manifest_path: str | Path,
    partial_report_path: str | Path,
    partial_lock_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
]:
    manifest, policy, census, admission_report, method_lock = (
        _load_method_inputs(
            manifest_path=manifest_path,
            policy_path=policy_path,
            census_path=census_path,
            admission_report_path=admission_report_path,
            method_lock_path=method_lock_path,
            core_manifest_path=core_manifest_path,
        )
    )
    partial_report, partial_report_file_sha256 = load_exact_partial_report(
        partial_report_path
    )
    validate_partial_report(
        partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock=method_lock,
        manifest=manifest,
    )
    partial_lock = load_committed_partial_lock(
        partial_lock_path,
        partial_report=partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock=method_lock,
        manifest=manifest,
        require_canonical_path=True,
    )
    return (
        manifest,
        policy,
        census,
        admission_report,
        method_lock,
        partial_report,
        partial_report_file_sha256,
        partial_lock,
    )


def run_finalization(
    *,
    manifest_path: str | Path = methods.DEFAULT_MANIFEST,
    policy_path: str | Path = methods.DEFAULT_POLICY_LOCK,
    census_path: str | Path = methods.DEFAULT_CENSUS,
    admission_report_path: str | Path = methods.DEFAULT_ADMISSION_REPORT,
    method_lock_path: str | Path = methods.DEFAULT_METHOD_LOCK,
    core_manifest_path: str | Path = methods.DEFAULT_CORE_MANIFEST,
    partial_report_path: str | Path = DEFAULT_PARTIAL_REPORT,
    partial_lock_path: str | Path = DEFAULT_PARTIAL_LOCK,
    data_path: str | Path | None = None,
    output: str | Path = DEFAULT_OUTPUT,
    explicit_acknowledgement: Any,
) -> dict[str, Any]:
    """Strict disk-backed finalization entry point; there is no resume mode."""

    input_paths = (
        manifest_path,
        policy_path,
        census_path,
        admission_report_path,
        method_lock_path,
        core_manifest_path,
        partial_report_path,
        partial_lock_path,
        data_path,
    )
    output_path = _validate_output_path(output, input_paths=input_paths)
    if output_path.exists():
        raise FileExistsError("final output already exists; resume is forbidden")

    before = _load_finalization_inputs(
        manifest_path=manifest_path,
        policy_path=policy_path,
        census_path=census_path,
        admission_report_path=admission_report_path,
        method_lock_path=method_lock_path,
        core_manifest_path=core_manifest_path,
        partial_report_path=partial_report_path,
        partial_lock_path=partial_lock_path,
    )
    (
        manifest,
        _policy,
        _census,
        _admission_report,
        method_lock,
        partial_report,
        partial_report_file_sha256,
        partial_lock,
    ) = before
    authorize_finalization(
        partial_lock,
        partial_report=partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        method_lock=method_lock,
        manifest=manifest,
        explicit_acknowledgement=explicit_acknowledgement,
    )

    data_file_sha256 = (
        None if data_path is None else _sha256_file(Path(data_path).expanduser())
    )
    rows = benchmark.base.load_pinned_longmemeval_rows(data_path)
    device = str(method_lock["authorization"]["runtime"]["device"])
    runtime = methods.admission._make_runtime(device=device)
    runtime.ensure_loaded()
    methods.verify_execution_runtime(
        runtime,
        locked_runtime=method_lock["authorization"]["runtime"],
    )
    corrected = benchmark.rehydrate_manifest(
        manifest,
        rows,
        runtime.tokenizer,
        core_manifest=core_manifest_path,
    )
    records = tuple(item.runtime for item in corrected)
    result = _finalize_rehydrated_records(
        manifest=manifest,
        method_lock=method_lock,
        partial_report=partial_report,
        partial_report_file_sha256=partial_report_file_sha256,
        partial_lock=partial_lock,
        records=records,
        runtime=runtime,
        explicit_acknowledgement=explicit_acknowledgement,
    )

    after = _load_finalization_inputs(
        manifest_path=manifest_path,
        policy_path=policy_path,
        census_path=census_path,
        admission_report_path=admission_report_path,
        method_lock_path=method_lock_path,
        core_manifest_path=core_manifest_path,
        partial_report_path=partial_report_path,
        partial_lock_path=partial_lock_path,
    )
    if before != after:
        raise RuntimeError("a protected finalizer input changed during execution")
    if data_path is not None and _sha256_file(data_path) != data_file_sha256:
        raise RuntimeError("the pinned LongMemEval data changed during execution")
    if output_path.exists():
        raise FileExistsError("final output appeared during execution")
    _atomic_write_new(output_path, result)
    return result


def freeze_partial_lock_file(
    *,
    manifest_path: str | Path = methods.DEFAULT_MANIFEST,
    policy_path: str | Path = methods.DEFAULT_POLICY_LOCK,
    census_path: str | Path = methods.DEFAULT_CENSUS,
    admission_report_path: str | Path = methods.DEFAULT_ADMISSION_REPORT,
    method_lock_path: str | Path = methods.DEFAULT_METHOD_LOCK,
    core_manifest_path: str | Path = methods.DEFAULT_CORE_MANIFEST,
    partial_report_path: str | Path = DEFAULT_PARTIAL_REPORT,
    partial_lock_path: str | Path = DEFAULT_PARTIAL_LOCK,
) -> dict[str, Any]:
    """Generate, but do not commit, the canonical pre-execution partial lock."""

    lock_path = _require_canonical_path(
        partial_lock_path,
        DEFAULT_PARTIAL_LOCK,
        name="geometry methods partial lock",
    )
    if lock_path.exists():
        raise FileExistsError("canonical partial lock already exists")
    manifest, _policy, _census, _admission, method_lock = _load_method_inputs(
        manifest_path=manifest_path,
        policy_path=policy_path,
        census_path=census_path,
        admission_report_path=admission_report_path,
        method_lock_path=method_lock_path,
        core_manifest_path=core_manifest_path,
    )
    partial_report, partial_report_file_sha256 = load_exact_partial_report(
        partial_report_path
    )
    lock = freeze_partial_lock(
        partial_report,
        method_lock,
        partial_report_file_sha256=partial_report_file_sha256,
        manifest=manifest,
    )
    partial_report_after, file_sha256_after = load_exact_partial_report(
        partial_report_path
    )
    if (
        partial_report_after != partial_report
        or file_sha256_after != partial_report_file_sha256
    ):
        raise RuntimeError("partial report changed while its lock was generated")
    _atomic_write_new(_resolved(lock_path), lock)
    return lock


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
        "--core-manifest",
        default=str(methods.DEFAULT_CORE_MANIFEST),
    )
    parser.add_argument(
        "--partial-report",
        default=str(DEFAULT_PARTIAL_REPORT),
    )
    parser.add_argument("--partial-lock", default=str(DEFAULT_PARTIAL_LOCK))
    parser.add_argument("--data-path")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--freeze-partial-lock",
        action="store_true",
        help="write the canonical lock for parent review and commit; no scoring",
    )
    mode.add_argument(
        "--allow-finalization",
        action="store_true",
        help="exact acknowledgement required to score only missing record16",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    common = {
        "manifest_path": args.manifest,
        "policy_path": args.policy_lock,
        "census_path": args.census,
        "admission_report_path": args.admission_report,
        "method_lock_path": args.method_lock,
        "core_manifest_path": args.core_manifest,
        "partial_report_path": args.partial_report,
        "partial_lock_path": args.partial_lock,
    }
    try:
        if args.freeze_partial_lock:
            freeze_partial_lock_file(**common)
        else:
            run_finalization(
                **common,
                data_path=args.data_path,
                output=args.out,
                explicit_acknowledgement=args.allow_finalization,
            )
    except (
        FileExistsError,
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
        benchmark.ManifestError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

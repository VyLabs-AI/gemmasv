"""Run the locked geometry-corrected LongMemEval deletion-method matrix.

The authorization lock binds the exact geometry manifest, policy, census,
original v1 core disclosure, and completed geometry admission report.  Every
one of the 16 fixed-C-feasible records is executed without replacement; method
efficacy is additionally summarized on the predeclared joint-admitted 10.
Resume reports are audit-only and no completed row is ever reused.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_geometry_fix as admission
from gemma_sv import eval_longmemeval_chat_methods as hardened
from gemma_sv import longmemeval_chat_geometry_fix as benchmark


SCHEMA = "gemma-sv-longmemeval-chat-geometry-deletion-methods-v1"
SCHEMA_VERSION = 1
METHOD_LOCK_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-method-authorization-lock-v1"
)
METHOD_LOCK_STATUS = "frozen-before-geometry-deletion-method-model-scoring"

EXPECTED_RECORDS = 16
EXPECTED_TARGET_ADMITTED = 10
EXPECTED_RETAINED_AVAILABLE = 16
EXPECTED_JOINT_ADMITTED = 10

PRESENT_ID = hardened.PRESENT_ID
FRESH_RAW_OMISSION_ID = hardened.FRESH_RAW_OMISSION_ID
TOKEN_ROW_DIAGNOSTIC_ID = hardened.TOKEN_ROW_DIAGNOSTIC_ID
EXACT_DECREMENT_ID = hardened.EXACT_DECREMENT_ID
FIXED_C_REFIT_ID = hardened.FIXED_C_REFIT_ID
FP32_PROXY_ID = hardened.FP32_PROXY_ID
DECAY_ID = hardened.DECAY_ID
CACHE_DELETE_SHIFT_ID = hardened.CACHE_DELETE_SHIFT_ID
PROMPT_SUPPRESSION_ID = hardened.PROMPT_SUPPRESSION_ID
CONDITION_IDS = (
    PRESENT_ID,
    FRESH_RAW_OMISSION_ID,
    TOKEN_ROW_DIAGNOSTIC_ID,
    EXACT_DECREMENT_ID,
    FIXED_C_REFIT_ID,
    FP32_PROXY_ID,
    DECAY_ID,
    CACHE_DELETE_SHIFT_ID,
    PROMPT_SUPPRESSION_ID,
)

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_MANIFEST = admission.DEFAULT_MANIFEST
DEFAULT_POLICY_LOCK = admission.DEFAULT_POLICY_LOCK
DEFAULT_CENSUS = admission.DEFAULT_CENSUS
DEFAULT_CORE_MANIFEST = benchmark.DEFAULT_CORE_MANIFEST_PATH
DEFAULT_ADMISSION_REPORT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_geometry_admission_v1.json"
)
DEFAULT_METHOD_LOCK = (
    BENCHMARKS
    / "longmemeval_chat_geometry_method_authorization_v1.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_geometry_deletion_methods_v1.json"
)

ADMISSION_REPORT_FILE_SHA256 = (
    "36c2eaab9f39f1d0a5707460cc6fcb65027bf737dbe60f2b700993e8d2e82368"
)
ADMISSION_REPORT_PAYLOAD_SHA256 = (
    "f23ace307cf47fe9bc7ece575fd3190992660a815f79d8a048243a2736656988"
)
ADMISSION_IMPLEMENTATION_CONTRACT_SHA256 = (
    "e5655e694e8a7cca6d68704ba32b6b45cd5faa654ab786dc44353af720e5ae16"
)

_IMPLEMENTATION_PATHS = {
    **hardened._IMPLEMENTATION_PATHS,
    "gemma_sv/__init__.py": PACKAGE / "__init__.py",
    "eval_longmemeval_chat_geometry_methods.py": Path(__file__).resolve(),
    "eval_longmemeval_chat_geometry_fix.py": PACKAGE
    / "eval_longmemeval_chat_geometry_fix.py",
    "eval_longmemeval_chat_v2.py": PACKAGE / "eval_longmemeval_chat_v2.py",
    "eval_longmemeval_deletion.py": PACKAGE / "eval_longmemeval_deletion.py",
    "eval_context_erasure_qa.py": PACKAGE / "eval_context_erasure_qa.py",
    "longmemeval_chat_geometry_fix.py": PACKAGE
    / "longmemeval_chat_geometry_fix.py",
    "longmemeval_chat_benchmark_v2.py": PACKAGE
    / "longmemeval_chat_benchmark_v2.py",
    "longmemeval_deletion_benchmark.py": PACKAGE
    / "longmemeval_deletion_benchmark.py",
    "memops_longitudinal.py": PACKAGE / "memops_longitudinal.py",
    "rag_benchmark.py": PACKAGE / "rag_benchmark.py",
    "recovery_protocol.py": PACKAGE / "recovery_protocol.py",
    "demo_server/__init__.py": PACKAGE / "demo_server" / "__init__.py",
    "demo_server/certificate.py": PACKAGE
    / "demo_server"
    / "certificate.py",
    "demo_server/contract.py": PACKAGE / "demo_server" / "contract.py",
    "demo_server/engine.py": PACKAGE / "demo_server" / "engine.py",
    "demo_server/gate_context.py": PACKAGE
    / "demo_server"
    / "gate_context.py",
    "demo_server/scenarios.py": PACKAGE / "demo_server" / "scenarios.py",
    "demo_server/span.py": PACKAGE / "demo_server" / "span.py",
    "demo_server/state.py": PACKAGE / "demo_server" / "state.py",
    "svattn/__init__.py": WORKSPACE / "svattn" / "__init__.py",
    "svattn/fast_diff_svdd.py": WORKSPACE / "svattn" / "fast_diff_svdd.py",
    "svattn/sv_attention.py": WORKSPACE / "svattn" / "sv_attention.py",
    "svattn/diff_svdd.py": WORKSPACE / "svattn" / "diff_svdd.py",
    "cp_svm/__init__.py": WORKSPACE / "cp_svm" / "__init__.py",
}
_PROTECTED_INPUTS = frozenset(
    {
        DEFAULT_MANIFEST,
        DEFAULT_POLICY_LOCK,
        DEFAULT_CENSUS,
        DEFAULT_CORE_MANIFEST,
        DEFAULT_ADMISSION_REPORT,
        DEFAULT_METHOD_LOCK,
        *_IMPLEMENTATION_PATHS.values(),
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


def _load_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_file(path: str | Path, expected: str, *, name: str) -> None:
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} is not the exact committed artifact")


def _assert_source_free(payload: Mapping[str, Any]) -> None:
    admission._assert_source_free(payload)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_output_path(
    output: str | Path,
    *,
    input_paths: Sequence[str | Path | None],
) -> Path:
    output_path = Path(output).expanduser()
    resolved = _resolved(output_path)
    protected = {
        Path(path).expanduser()
        for path in (*input_paths, *_PROTECTED_INPUTS)
        if path is not None
    }
    for path in protected:
        same_path = resolved == _resolved(path)
        same_file = False
        if output_path.exists() and path.exists():
            try:
                same_file = os.path.samefile(output_path, path)
            except OSError:
                pass
        if same_path or same_file:
            raise ValueError("output aliases a protected method input")
    return resolved


def _admission_flags(row: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    value = row.get("admission") or {}
    target = bool((value.get("target_recall") or {}).get("admitted"))
    retained = bool(
        (value.get("retained_availability") or {}).get("available")
    )
    joint = bool(
        (value.get("joint_target_and_retained") or {}).get("admitted")
    )
    if joint != (target and retained):
        raise ValueError("geometry admission row has inconsistent joint status")
    return target, retained, joint


def validate_admission_report(
    report: Mapping[str, Any],
    *,
    report_file_sha256: str,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_source_free(report)
    if (
        report_file_sha256 != ADMISSION_REPORT_FILE_SHA256
        or _payload_sha256(report) != ADMISSION_REPORT_PAYLOAD_SHA256
    ):
        raise ValueError("geometry admission report is not the exact artifact")
    if (
        report.get("schema") != admission.SCHEMA
        or report.get("status") != "completed"
        or report.get("contains_source_text") is not False
        or report.get("contains_full_vocabulary_vectors") is not False
        or report.get("disclosure") != admission.DISCLOSURE
    ):
        raise ValueError("geometry admission report contract differs")
    implementation = report.get("implementation") or {}
    if implementation.get("contract_sha256") != (
        ADMISSION_IMPLEMENTATION_CONTRACT_SHA256
    ):
        raise ValueError("geometry admission implementation differs")
    report_manifest = report.get("manifest") or {}
    report_policy = report.get("policy_lock") or {}
    report_census = report.get("census") or {}
    if (
        report_manifest.get("file_sha256")
        != admission.PINNED_MANIFEST_FILE_SHA256
        or report_manifest.get("integrity_sha256")
        != manifest["integrity"]["sha256"]
        or report_policy.get("file_sha256")
        != admission.PINNED_POLICY_LOCK_FILE_SHA256
        or report_policy.get("lock_sha256") != policy["lock_sha256"]
        or report_census.get("file_sha256")
        != admission.PINNED_CENSUS_FILE_SHA256
        or report_census.get("integrity_sha256")
        != census["integrity"]["sha256"]
    ):
        raise ValueError("geometry admission artifact bindings differ")
    denominators = (report.get("summary") or {}).get("denominators") or {}
    required = {
        "authorized_records": EXPECTED_RECORDS,
        "attempted_records": EXPECTED_RECORDS,
        "completed_records": EXPECTED_RECORDS,
        "record_failures": 0,
        "target_admitted": EXPECTED_TARGET_ADMITTED,
        "retained_available": EXPECTED_RETAINED_AVAILABLE,
        "joint_admitted": EXPECTED_JOINT_ADMITTED,
        "fixed_c_all_boundaries_feasible_records": EXPECTED_RECORDS,
        "fixed_c_infeasible_records": 0,
        "local_window_safe_records": EXPECTED_RECORDS,
        "local_window_failures": 0,
    }
    if any(int(denominators.get(key, -1)) != value for key, value in required.items()):
        raise ValueError("geometry admission denominators are not authorized")
    manifest_ids = [str(row["record_id"]) for row in manifest["records"]]
    rows = list(report.get("records") or ())
    if (
        len(rows) != EXPECTED_RECORDS
        or [str(row.get("record_id") or "") for row in rows] != manifest_ids
    ):
        raise ValueError("geometry admission rows differ from frozen all16")
    admission_records = []
    target_count = retained_count = joint_count = 0
    for row, public in zip(rows, manifest["records"]):
        target, retained, joint = _admission_flags(row)
        target_count += int(target)
        retained_count += int(retained)
        joint_count += int(joint)
        if (
            row.get("status") != "completed"
            or (row.get("source_state_immutability") or {}).get("unchanged")
            is not True
            or (row.get("local_window_safety") or {}).get(
                "owned_round_strictly_outside_local_window_before_query"
            )
            is not True
            or (public.get("context") or {})
            .get("fixed_c_reference", {})
            .get("all_affected_boundaries_feasible")
            is not True
        ):
            raise ValueError("geometry admission row is not method-authorizable")
        admission_records.append(
            {
                "record_id": row["record_id"],
                "target_admitted": target,
                "retained_available": retained,
                "joint_admitted": joint,
                "source_fixed_c_feasible": True,
            }
        )
    if (
        target_count != EXPECTED_TARGET_ADMITTED
        or retained_count != EXPECTED_RETAINED_AVAILABLE
        or joint_count != EXPECTED_JOINT_ADMITTED
    ):
        raise ValueError("geometry admission outcomes differ from summary")
    return {
        "admission_records": admission_records,
        "joint_record_ids": [
            row["record_id"] for row in admission_records if row["joint_admitted"]
        ],
        "report_payload_sha256": _payload_sha256(report),
    }


def load_locked_artifacts(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    policy_path: str | Path = DEFAULT_POLICY_LOCK,
    census_path: str | Path = DEFAULT_CENSUS,
    admission_report_path: str | Path = DEFAULT_ADMISSION_REPORT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, policy, census = admission.load_locked_inputs(
        manifest_path,
        policy_path,
        census_path,
    )
    _require_file(
        admission_report_path,
        ADMISSION_REPORT_FILE_SHA256,
        name="geometry admission report",
    )
    report = _load_mapping(admission_report_path, name="geometry admission report")
    validate_admission_report(
        report,
        report_file_sha256=ADMISSION_REPORT_FILE_SHA256,
        manifest=manifest,
        policy=policy,
        census=census,
    )
    return manifest, policy, census, report


def freeze_method_authorization_lock(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    *,
    admission_report_sha256: str = ADMISSION_REPORT_FILE_SHA256,
) -> dict[str, Any]:
    admission.validate_locked_inputs(manifest, policy, census)
    analysis = validate_admission_report(
        admission_report,
        report_file_sha256=admission_report_sha256,
        manifest=manifest,
        policy=policy,
        census=census,
    )
    expected_runtime = admission.runtime_contract(
        device=str(admission_report["config"]["device"])
    )
    report_config = admission_report.get("config") or {}
    if any(
        report_config.get(key) != value
        for key, value in expected_runtime.items()
    ):
        raise ValueError(
            "geometry admission runtime differs from the executable contract"
        )
    implementation = _implementation_fingerprints()
    lock = {
        "schema": METHOD_LOCK_SCHEMA,
        "status": METHOD_LOCK_STATUS,
        "contains_source_text": False,
        "artifacts": {
            "manifest_integrity_sha256": manifest["integrity"]["sha256"],
            "manifest_file_sha256": admission.PINNED_MANIFEST_FILE_SHA256,
            "policy_lock_sha256": policy["lock_sha256"],
            "policy_file_sha256": admission.PINNED_POLICY_LOCK_FILE_SHA256,
            "census_integrity_sha256": census["integrity"]["sha256"],
            "census_file_sha256": admission.PINNED_CENSUS_FILE_SHA256,
            "core_manifest_file_sha256": benchmark.PINNED_CORE_FILE_SHA256,
            "admission_report_file_sha256": admission_report_sha256,
            "admission_report_payload_sha256": analysis[
                "report_payload_sha256"
            ],
        },
        "implementation": implementation,
        "implementation_contract_sha256": implementation["contract_sha256"],
        "disclosure": copy.deepcopy(admission.DISCLOSURE),
        "authorization": {
            "all_records_without_replacement": True,
            "frozen_records": EXPECTED_RECORDS,
            "target_admitted": EXPECTED_TARGET_ADMITTED,
            "retained_available": EXPECTED_RETAINED_AVAILABLE,
            "joint_admitted": EXPECTED_JOINT_ADMITTED,
            "record_failures": 0,
            "fixed_c_feasible_records": EXPECTED_RECORDS,
            "fixed_c_infeasible_records": 0,
            "runtime": expected_runtime,
            "condition_ids": list(CONDITION_IDS),
            "prompt_suppression": {
                "enabled": True,
                "valid_registered_gemma_chat_required": True,
                "target_free_required": True,
            },
            "exact_decrement": {
                "source_feasible_records": EXPECTED_RECORDS,
                "source_infeasible_records": 0,
                "source_full_repack_fallback_expected": False,
                "execution_strata": [
                    "incremental_float64_fixed_c_decrement",
                    "fixed_c_refit_fallback",
                ],
            },
            "fp32_proxy": {
                "source_full_repack_fallback_expected": False,
            },
            "resume": {
                "existing_report_policy": "validate_then_discard_all_rows",
                "completed_rows_reused": False,
                "all_authorized_records_reexecuted": True,
            },
            "efficacy_population": {
                "name": "predeclared_geometry_joint10",
                "record_ids": analysis["joint_record_ids"],
                "record_ids_sha256": _payload_sha256(
                    analysis["joint_record_ids"]
                ),
                "records": EXPECTED_JOINT_ADMITTED,
                "selected_before_method_scoring": True,
            },
            "admission_records": analysis["admission_records"],
        },
    }
    lock["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(lock),
    }
    _assert_source_free(lock)
    return lock


def validate_method_authorization_lock(
    lock: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    admission_report_sha256: str = ADMISSION_REPORT_FILE_SHA256,
) -> None:
    _assert_source_free(lock)
    integrity = lock.get("integrity") or {}
    unsigned = dict(lock)
    unsigned.pop("integrity", None)
    if (
        lock.get("schema") != METHOD_LOCK_SCHEMA
        or lock.get("status") != METHOD_LOCK_STATUS
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise ValueError("geometry method authorization lock is invalid")
    expected = freeze_method_authorization_lock(
        manifest,
        policy,
        census,
        admission_report,
        admission_report_sha256=admission_report_sha256,
    )
    if dict(lock) != expected:
        raise ValueError("geometry method lock differs from frozen inputs")


def load_committed_method_lock(
    path: str | Path,
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    require_canonical_path: bool = True,
) -> dict[str, Any]:
    """Load the lock only from its canonical committed path for execution."""

    lock_path = Path(path).expanduser()
    if (
        require_canonical_path
        and lock_path.resolve() != DEFAULT_METHOD_LOCK.resolve()
    ):
        raise PermissionError(
            "geometry methods require the canonical committed method lock"
        )
    lock = _load_mapping(lock_path, name="geometry method authorization lock")
    validate_method_authorization_lock(
        lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
    )
    return lock


def authorize_method_execution(
    lock: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    explicit_acknowledgement: Any,
) -> tuple[str, str]:
    validate_method_authorization_lock(
        lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
    )
    if explicit_acknowledgement is not True:
        raise PermissionError("geometry method acknowledgement must be exactly True")
    runtime = lock["authorization"]["runtime"]
    return str(runtime["arm"]), str(runtime["device"])


def verify_execution_runtime(
    runtime: Any,
    *,
    locked_runtime: Mapping[str, Any],
) -> None:
    expected = admission.runtime_contract(
        device=str(locked_runtime.get("device") or "")
    )
    if dict(locked_runtime) != expected:
        raise RuntimeError("locked geometry method runtime contract drifted")
    admission.verify_loaded_runtime(runtime, device=str(expected["device"]))


def evaluate_record(
    runtime: Any,
    record: Any,
    public_record: Mapping[str, Any],
    *,
    admission_binding: Mapping[str, Any],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    admission._validate_record(public_record, record)
    row = hardened.evaluate_record(
        runtime,
        record,
        admission_binding=admission_binding,
        source_fixed_c_feasible=True,
        prompt_suppression=True,
        warmup=warmup,
        repeats=repeats,
    )
    exact = (row.get("conditions") or {}).get(EXACT_DECREMENT_ID) or {}
    semantics = exact.get("semantics") or {}
    if semantics.get("full_repack_fallback"):
        raise RuntimeError("fixed-C-feasible exact decrement used source fallback")
    proxy = (row.get("conditions") or {}).get(FP32_PROXY_ID) or {}
    if (proxy.get("semantics") or {}).get("full_repack_fallback"):
        raise RuntimeError("fixed-C-feasible FP32 proxy used source fallback")
    row["geometry_binding"] = {
        "manifest_record_integrity_sha256": public_record["record_integrity"][
            "sha256"
        ],
        "fixed_c_feasible": True,
        "local_window_safe": True,
    }
    return row


def _reconcile_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive completion from method outputs rather than trusting its label."""

    declared_status = str(row.get("status") or "")
    conditions_value = row.get("conditions")
    conditions = (
        conditions_value if isinstance(conditions_value, Mapping) else {}
    )
    condition_ids_exact = set(conditions) == set(CONDITION_IDS)
    condition_statuses = {
        condition_id: str((conditions.get(condition_id) or {}).get("status") or "")
        for condition_id in CONDITION_IDS
    }
    condition_success = {
        condition_id: status == "completed"
        for condition_id, status in condition_statuses.items()
    }
    failed_condition_ids = [
        condition_id
        for condition_id, successful in condition_success.items()
        if not successful
    ]
    certificate = row.get("solver_certificate")
    certificate_status = (
        str(certificate.get("status") or "")
        if isinstance(certificate, Mapping)
        else ""
    )
    certificate_success = certificate_status == "completed"
    immutability = row.get("source_state_immutability")
    immutability_success = bool(
        isinstance(immutability, Mapping)
        and immutability.get("unchanged") is True
        and immutability.get("shape_signature_unchanged") is True
    )
    computed_failures = list(failed_condition_ids)
    declared_failures = row.get("method_failures")
    method_failures_consistent = (
        isinstance(declared_failures, list)
        and declared_failures == computed_failures
    )
    terminal_success = bool(
        condition_ids_exact
        and all(condition_success.values())
        and certificate_success
        and immutability_success
        and method_failures_consistent
    )
    declared_status_consistent = (
        (declared_status == "completed" and terminal_success)
        or (
            declared_status == "completed_with_failures"
            and not terminal_success
            and bool(failed_condition_ids)
        )
        or declared_status in {"failed", "not_attempted"}
    )
    fully_completed = bool(
        declared_status == "completed"
        and terminal_success
        and declared_status_consistent
    )
    return {
        "declared_status": declared_status,
        "fully_completed": fully_completed,
        "declared_status_consistent": declared_status_consistent,
        "condition_ids_exact": condition_ids_exact,
        "condition_statuses": condition_statuses,
        "failed_condition_ids": failed_condition_ids,
        "certificate_status": certificate_status,
        "certificate_success": certificate_success,
        "source_state_immutability_success": immutability_success,
        "method_failures_consistent": method_failures_consistent,
    }


def _attempted_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status") != "not_attempted"]


def _validate_inner_accounting(
    result: Mapping[str, Any],
    *,
    attempted_records: int,
) -> tuple[int, int]:
    accounted = int(result.get("accounted_records", -1))
    unaccounted = int(result.get("unaccounted_records", -1))
    expected_fully_accounted = (
        unaccounted == 0 and accounted == attempted_records
    )
    if (
        accounted < 0
        or unaccounted < 0
        or accounted + unaccounted != attempted_records
        or result.get("fully_accounted") is not expected_fully_accounted
    ):
        raise ValueError("inner method accounting wrapper is contradictory")
    return accounted, unaccounted


def _checkpoint_condition_summary(
    rows: Sequence[Mapping[str, Any]],
    condition_id: str,
) -> dict[str, Any]:
    attempted = _attempted_rows(rows)
    result = hardened._condition_summary(attempted, condition_id)
    accounted, unaccounted = _validate_inner_accounting(
        result,
        attempted_records=len(attempted),
    )
    result.update(
        {
            "authorized_records": len(rows),
            "attempted_records": len(attempted),
            "not_attempted_records": len(rows) - len(attempted),
            "accounted_records": accounted,
            "unaccounted_records": unaccounted,
            "unaccounted_attempted_records": unaccounted,
            "fully_accounted": (
                len(attempted) == len(rows)
                and unaccounted == 0
                and accounted == len(attempted)
            ),
        }
    )
    return result


def _checkpoint_certificate_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempted = _attempted_rows(rows)
    result = hardened._certificate_summary(attempted)
    accounted, unaccounted = _validate_inner_accounting(
        result,
        attempted_records=len(attempted),
    )
    result.update(
        {
            "authorized_records": len(rows),
            "attempted_records": len(attempted),
            "not_attempted_records": len(rows) - len(attempted),
            "accounted_records": accounted,
            "unaccounted_records": unaccounted,
            "unaccounted_attempted_records": unaccounted,
            "fully_accounted": (
                len(attempted) == len(rows)
                and unaccounted == 0
                and accounted == len(attempted)
            ),
        }
    )
    return result


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    method_lock: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = method_lock.get("authorization") or {}
    admission_records = list(authorization.get("admission_records") or ())
    authorized_ids = [str(row.get("record_id") or "") for row in admission_records]
    if [str(row.get("record_id") or "") for row in records] != authorized_ids:
        raise ValueError("method rows differ from authorized all16 order")
    if list(authorization.get("condition_ids") or ()) != list(CONDITION_IDS):
        raise ValueError("geometry method condition matrix differs")
    joint_ids = set(
        (authorization.get("efficacy_population") or {}).get("record_ids") or ()
    )
    joint_rows = [row for row in records if row.get("record_id") in joint_ids]
    if len(joint_rows) != EXPECTED_JOINT_ADMITTED:
        raise ValueError("predeclared geometry efficacy subset must contain 10")
    attempted = _attempted_rows(records)
    reconciliations = [_reconcile_row(row) for row in attempted]
    complete = [
        row
        for row, outcome in zip(attempted, reconciliations)
        if outcome["fully_completed"]
    ]
    generic_failed = [row for row in attempted if row.get("status") == "failed"]
    declared_condition_failed = [
        row for row in attempted if row.get("status") == "completed_with_failures"
    ]
    immutable = sum(
        outcome["source_state_immutability_success"]
        for outcome in reconciliations
    )
    exact_labels = [
        str(
            (
                row.get("conditions", {})
                .get(EXACT_DECREMENT_ID, {})
                .get("semantics", {})
            ).get("executed_method")
            or ""
        )
        for row in attempted
    ]
    denominators = {
        "frozen_records": EXPECTED_RECORDS,
        "authorized_records": len(authorized_ids),
        "attempted_records": len(attempted),
        "not_attempted_records": len(records) - len(attempted),
        "fully_completed_records": len(complete),
        "record_failures": len(attempted) - len(complete),
        "generic_record_failures": len(generic_failed),
        "records_with_condition_failures": sum(
            bool(outcome["failed_condition_ids"])
            for outcome in reconciliations
        ),
        "declared_completed_with_failures_records": len(
            declared_condition_failed
        ),
        "declared_status_mismatch_records": sum(
            not outcome["declared_status_consistent"]
            for outcome in reconciliations
        ),
        "predeclared_target_admitted": EXPECTED_TARGET_ADMITTED,
        "predeclared_retained_available": EXPECTED_RETAINED_AVAILABLE,
        "predeclared_joint_admitted": EXPECTED_JOINT_ADMITTED,
        "source_fixed_c_feasible_records": EXPECTED_RECORDS,
        "source_fixed_c_infeasible_records": 0,
        "source_state_immutable_records": immutable,
        "source_state_immutability_failures": len(attempted) - immutable,
        "source_state_immutability_unverified_records": sum(
            not isinstance(row.get("source_state_immutability"), Mapping)
            for row in attempted
        ),
        "exact_decrement_incremental_records": exact_labels.count(
            "incremental_float64_fixed_c_decrement"
        ),
        "exact_decrement_refit_fallback_records": exact_labels.count(
            "fixed_c_refit_fallback"
        ),
        "exact_decrement_full_repack_fallback_records": sum(
            bool(
                (
                    row.get("conditions", {})
                    .get(EXACT_DECREMENT_ID, {})
                    .get("semantics", {})
                ).get("full_repack_fallback")
            )
            for row in attempted
        ),
        "fp32_full_repack_fallback_records": sum(
            bool(
                (
                    row.get("conditions", {})
                    .get(FP32_PROXY_ID, {})
                    .get("semantics", {})
                ).get("full_repack_fallback")
            )
            for row in attempted
        ),
    }
    summary = {
        "aggregation_population": (
            "all16 fixed geometry records; efficacy on predeclared joint10"
        ),
        "denominators": denominators,
        "failure_record_ids": [
            str(row.get("record_id") or "")
            for row, outcome in zip(attempted, reconciliations)
            if not outcome["fully_completed"]
        ],
        "declared_status_mismatch_record_ids": [
            str(row.get("record_id") or "")
            for row, outcome in zip(attempted, reconciliations)
            if not outcome["declared_status_consistent"]
        ],
        "conditions": {},
    }
    for condition_id in CONDITION_IDS:
        summary["conditions"][condition_id] = {
            "all_frozen_records": _checkpoint_condition_summary(
                records, condition_id
            ),
            "predeclared_joint10_subset": _checkpoint_condition_summary(
                joint_rows, condition_id
            ),
        }
    summary["exact_refit_certificate"] = {
        "all_frozen_records": _checkpoint_certificate_summary(records),
        "predeclared_joint10_subset": _checkpoint_certificate_summary(joint_rows),
    }
    summary["exact_decrement_execution_strata"] = {
        "incremental_float64_fixed_c_decrement": denominators[
            "exact_decrement_incremental_records"
        ],
        "fixed_c_refit_fallback": denominators[
            "exact_decrement_refit_fallback_records"
        ],
        "source_full_repack_fallback": denominators[
            "exact_decrement_full_repack_fallback_records"
        ],
    }
    return summary


def _implementation_fingerprints() -> dict[str, str]:
    values = {
        name: _sha256_file(path)
        for name, path in sorted(_IMPLEMENTATION_PATHS.items())
    }
    values["contract_sha256"] = _payload_sha256(values)
    return values


def _base_report(
    method_lock: Mapping[str, Any],
    *,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    records = [
        {
            "record_id": row["record_id"],
            "status": "not_attempted",
            "contains_source_text": False,
        }
        for row in method_lock["authorization"]["admission_records"]
    ]
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "official_longmemeval_leaderboard_score": False,
        "disclosure": copy.deepcopy(admission.DISCLOSURE),
        "method_lock_integrity_sha256": method_lock["integrity"]["sha256"],
        "config": {
            **admission.runtime_contract(device=device),
            "warmup": int(warmup),
            "repeats": int(repeats),
        },
        "method_protocol": {
            "condition_ids": list(CONDITION_IDS),
            "all16_fixed_c_feasible": True,
            "source_full_repack_fallback_expected": False,
            "internal_exact_refit_fallback_stratified": True,
            "efficacy_population": "predeclared_geometry_joint10",
            "resume_policy": "validate_then_discard_all_rows",
            "completed_resume_rows_reused": False,
        },
        "implementation": _implementation_fingerprints(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "records": records,
    }
    report["summary"] = summarize_records(records, method_lock=method_lock)
    return report


def _validate_resume(report: Mapping[str, Any], expected: Mapping[str, Any]) -> int:
    _assert_source_free(report)
    hardened._assert_finite_json(report)
    ignored = {
        "status",
        "records",
        "summary",
        "elapsed_seconds_this_process",
        "resume_requested",
        "discarded_resume_rows",
        "reused_completed_resume_records",
    }
    if {
        key: value for key, value in report.items() if key not in ignored
    } != {
        key: value for key, value in expected.items() if key not in ignored
    }:
        raise ValueError("geometry method resume contract differs")
    rows = report.get("records")
    if not isinstance(rows, list):
        raise ValueError("geometry method resume rows are invalid")
    return len(rows)


def run_records(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    records: Sequence[Any],
    runtime: Any,
    *,
    manifest_path: Path,
    policy_path: Path,
    census_path: Path,
    admission_report_path: Path,
    method_lock_path: Path,
    core_manifest_path: Path,
    data_path: Path | None,
    output: Path,
    warmup: int,
    repeats: int,
    explicit_acknowledgement: Any,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    output = _validate_output_path(
        output,
        input_paths=(
            manifest_path,
            policy_path,
            census_path,
            admission_report_path,
            method_lock_path,
            core_manifest_path,
            data_path,
        ),
    )
    manifest_loaded, policy_loaded, census_loaded, report_loaded = (
        load_locked_artifacts(
            manifest_path=manifest_path,
            policy_path=policy_path,
            census_path=census_path,
            admission_report_path=admission_report_path,
        )
    )
    if (
        manifest != manifest_loaded
        or policy != policy_loaded
        or census != census_loaded
        or admission_report != report_loaded
    ):
        raise ValueError("programmatic geometry method inputs differ from files")
    committed_method_lock = load_committed_method_lock(
        method_lock_path,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
        require_canonical_path=True,
    )
    if committed_method_lock != method_lock:
        raise ValueError("programmatic geometry method lock differs from its file")
    _require_file(
        core_manifest_path,
        benchmark.PINNED_CORE_FILE_SHA256,
        name="original v1 core disclosure manifest",
    )
    validate_method_authorization_lock(
        method_lock,
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission_report,
    )
    if explicit_acknowledgement is not True:
        raise PermissionError("geometry method acknowledgement must be exactly True")
    authorized = method_lock["authorization"]["admission_records"]
    ids = [row["record_id"] for row in authorized]
    if len(records) != EXPECTED_RECORDS or [
        record.record_id for record in records
    ] != ids:
        raise ValueError("geometry methods require ordered all16 records")
    for record, public in zip(records, manifest["records"]):
        admission._validate_record(public, record)
    device = str(method_lock["authorization"]["runtime"]["device"])
    verify_execution_runtime(
        runtime,
        locked_runtime=method_lock["authorization"]["runtime"],
    )
    expected = _base_report(
        method_lock,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    if output.exists() and not (resume or overwrite):
        raise FileExistsError("output exists; pass resume or overwrite")
    discarded = 0
    if resume:
        if not output.is_file():
            raise ValueError("cannot resume a missing geometry method report")
        discarded = _validate_resume(
            _load_mapping(output, name="geometry method resume report"),
            expected,
        )
    result = expected
    result["resume_requested"] = bool(resume)
    result["discarded_resume_rows"] = discarded
    result["reused_completed_resume_records"] = 0
    admission.hardened._atomic_write(output, result)
    started = time.perf_counter()
    rows = list(result["records"])
    binding_by_id = {row["record_id"]: row for row in authorized}
    for index, (record, public) in enumerate(zip(records, manifest["records"])):
        try:
            row = evaluate_record(
                runtime,
                record,
                public,
                admission_binding=binding_by_id[record.record_id],
                warmup=warmup,
                repeats=repeats,
            )
        except Exception as exc:
            row = {
                "record_id": record.record_id,
                "status": "failed",
                "contains_source_text": False,
                "error_type": type(exc).__name__,
                "error_message_redacted": True,
                "error_message_sha256": benchmark.base.text_sha256(str(exc)),
            }
        rows[index] = row
        result["records"] = rows
        result["summary"] = summarize_records(rows, method_lock=method_lock)
        result["elapsed_seconds_this_process"] = time.perf_counter() - started
        admission.hardened._atomic_write(output, result)
    result["summary"] = summarize_records(rows, method_lock=method_lock)
    denominators = result["summary"]["denominators"]
    result["status"] = (
        "completed"
        if denominators["fully_completed_records"] == EXPECTED_RECORDS
        and denominators["record_failures"] == 0
        else "completed_with_record_failures"
    )
    result["elapsed_seconds_this_process"] = time.perf_counter() - started
    admission.hardened._atomic_write(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--policy-lock", default=str(DEFAULT_POLICY_LOCK))
    parser.add_argument("--census", default=str(DEFAULT_CENSUS))
    parser.add_argument("--admission-report", default=str(DEFAULT_ADMISSION_REPORT))
    parser.add_argument("--method-lock", default=str(DEFAULT_METHOD_LOCK))
    parser.add_argument("--core-manifest", default=str(DEFAULT_CORE_MANIFEST))
    parser.add_argument("--data-path")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-method-scoring", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
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
        "core": Path(args.core_manifest),
    }
    data_path = None if args.data_path is None else Path(args.data_path)
    try:
        manifest, policy, census, report = load_locked_artifacts(
            manifest_path=paths["manifest"],
            policy_path=paths["policy"],
            census_path=paths["census"],
            admission_report_path=paths["admission"],
        )
        method_lock = load_committed_method_lock(
            paths["method"],
            manifest=manifest,
            policy=policy,
            census=census,
            admission_report=report,
            require_canonical_path=True,
        )
        authorize_method_execution(
            method_lock,
            manifest=manifest,
            policy=policy,
            census=census,
            admission_report=report,
            explicit_acknowledgement=args.allow_method_scoring,
        )
        output = _validate_output_path(
            args.out,
            input_paths=(*paths.values(), data_path),
        )
    except (OSError, PermissionError, ValueError) as exc:
        parser.error(str(exc))
    rows = benchmark.base.load_pinned_longmemeval_rows(args.data_path)
    runtime = admission._make_runtime(device=str(method_lock["authorization"]["runtime"]["device"]))
    runtime.ensure_loaded()
    try:
        corrected = benchmark.rehydrate_manifest(
            manifest,
            rows,
            runtime.tokenizer,
            core_manifest=paths["core"],
        )
        records = tuple(item.runtime for item in corrected)
        result = run_records(
            manifest,
            policy,
            census,
            report,
            method_lock,
            records,
            runtime,
            manifest_path=paths["manifest"],
            policy_path=paths["policy"],
            census_path=paths["census"],
            admission_report_path=paths["admission"],
            method_lock_path=paths["method"],
            core_manifest_path=paths["core"],
            data_path=data_path,
            output=output,
            warmup=args.warmup,
            repeats=args.repeats,
            explicit_acknowledgement=args.allow_method_scoring,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, ValueError, benchmark.ManifestError) as exc:
        parser.error(str(exc))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

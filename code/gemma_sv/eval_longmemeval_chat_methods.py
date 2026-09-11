"""Run the locked post-admission LongMemEval V1 chat deletion matrix.

The method authorization lock is frozen from the exact completed confirmation
admission report.  Evaluation is deliberately unavailable until that separate
lock is committed at the canonical benchmark path and the caller supplies an
explicit model-scoring acknowledgement.  Resume inputs are validated only as
audit artifacts: completed method rows are never trusted or reused, and all 16
authorized records are always re-executed.

Runtime-only source strings are used to rehydrate official records and score
their complete ``target_current`` and ``retained`` answers.  Neither source
text nor full-vocabulary vectors are written to the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat as admission
from gemma_sv import eval_memops_longitudinal as longitudinal
from gemma_sv import eval_persistent_deletion_baselines as baseline
from gemma_sv import longmemeval_chat_benchmark as benchmark
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.persistent_deletion import (
    DECAY_FACTOR,
    cache_delete_and_shift,
    method_storage_report,
    persistent_state_shape_signature,
)


SCHEMA = "gemma-sv-longmemeval-chat-deletion-methods-v1"
SCHEMA_VERSION = 1
METHOD_LOCK_SCHEMA = "gemma-sv-longmemeval-chat-method-authorization-lock-v1"
METHOD_LOCK_STATUS = "frozen-before-deletion-method-model-scoring"

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_MANIFEST = BENCHMARKS / "longmemeval_chat_confirmation_v1.json"
DEFAULT_POLICY_LOCK = BENCHMARKS / "longmemeval_chat_policy_lock_v1.json"
DEFAULT_PROMOTION_LOCK = BENCHMARKS / "longmemeval_chat_promotion_v1.json"
DEFAULT_CONFIRMATION_REPORT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_confirmation_grafted_v1.json"
)
DEFAULT_METHOD_LOCK = (
    BENCHMARKS / "longmemeval_chat_method_authorization_v1.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_deletion_methods_v1.json"
)

CONFIRMATION_MANIFEST_FILE_SHA256 = (
    "7e9f484c4eee8cceecec79bfc0721e1cfc7100092d29f5b29d1e96eca478bae1"
)
POLICY_LOCK_FILE_SHA256 = (
    "2829c660378074bf075ffb86ce62715b7be96a626b6bf92753add125ab6b806f"
)
PROMOTION_LOCK_FILE_SHA256 = (
    "3ae6824087c05fca7777e6bb37445f4c31ed894f31884dd63d8142210e546f86"
)
PROMOTION_LOCK_INTEGRITY_SHA256 = (
    "2c737bba5051a7b56fac369d6bf16898a0b7225289d68eaa842431f062d6234e"
)
CONFIRMATION_REPORT_SHA256 = (
    "6683ec23cfd9446ebc008ee88091d5aa7e581efdffbca0312a759b5cd94bdbaa"
)
CONFIRMATION_REPORT_PAYLOAD_SHA256 = (
    "1e95a69cad623bf66286934571eef686e61430e603c336bbac64b83378559c93"
)
CONFIRMATION_IMPLEMENTATION_CONTRACT_SHA256 = (
    "b5172e56413092ce6929540aa978215378944d4a21efbb231bd263ee3af7faeb"
)

EXPECTED_FROZEN_RECORDS = 16
EXPECTED_TARGET_ADMITTED = 12
EXPECTED_RETAINED_AVAILABLE = 16
EXPECTED_JOINT_ADMITTED = 12
EXPECTED_SOURCE_INFEASIBLE = 2
EVALUATION_SEED = 0

PRESENT_ID = "present"
FRESH_RAW_OMISSION_ID = "fresh_raw_omission"
TOKEN_ROW_DIAGNOSTIC_ID = "token_row_repack"
EXACT_DECREMENT_ID = "exact_decrement"
FIXED_C_REFIT_ID = "fixed_c_refit"
FP32_PROXY_ID = "fp32_proxy"
DECAY_ID = "decay_0_01"
CACHE_DELETE_SHIFT_ID = "cache_delete_shift"
PROMPT_SUPPRESSION_ID = "prompt_suppression"

BASE_CONDITION_IDS = (
    PRESENT_ID,
    FRESH_RAW_OMISSION_ID,
    TOKEN_ROW_DIAGNOSTIC_ID,
    EXACT_DECREMENT_ID,
    FIXED_C_REFIT_ID,
    FP32_PROXY_ID,
    DECAY_ID,
    CACHE_DELETE_SHIFT_ID,
)

PROMPT_SUPPRESSION_INSTRUCTION = (
    "Memory-control instruction: Treat the deleted latest target-evidence "
    "round as unavailable. Do not reconstruct or reveal it. Preserve unrelated "
    "remembered values."
)

_IMPLEMENTATION_PATHS = {
    "eval_longmemeval_chat_methods.py": Path(__file__).resolve(),
    "eval_longmemeval_chat.py": PACKAGE / "eval_longmemeval_chat.py",
    "eval_memops_longitudinal.py": PACKAGE / "eval_memops_longitudinal.py",
    "eval_persistent_deletion_baselines.py": PACKAGE
    / "eval_persistent_deletion_baselines.py",
    "longmemeval_chat_benchmark.py": PACKAGE
    / "longmemeval_chat_benchmark.py",
    "persistent_deletion.py": PACKAGE / "persistent_deletion.py",
    "graft.py": PACKAGE / "graft.py",
    "recovery_state.py": PACKAGE / "recovery_state.py",
    "sv_global_attention.py": PACKAGE / "sv_global_attention.py",
    "layer_select.py": PACKAGE / "layer_select.py",
    "demo_server/gemma_engine.py": PACKAGE / "demo_server" / "gemma_engine.py",
    "svattn/causal_sv_attention.py": WORKSPACE
    / "svattn"
    / "causal_sv_attention.py",
    "svattn/mlx_svdd.py": WORKSPACE / "svattn" / "mlx_svdd.py",
    "cp_svm/oneclass_incremental.py": WORKSPACE
    / "cp_svm"
    / "oneclass_incremental.py",
    "cp_svm/oneclass_fast.py": WORKSPACE / "cp_svm" / "oneclass_fast.py",
    "cp_svm/kernels.py": WORKSPACE / "cp_svm" / "kernels.py",
}


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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require_exact_file(path: str | Path, expected: str, *, name: str) -> None:
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} is not the exact locked artifact")


def expected_condition_ids(*, prompt_suppression: bool) -> tuple[str, ...]:
    return BASE_CONDITION_IDS + (
        (PROMPT_SUPPRESSION_ID,) if prompt_suppression else ()
    )


def _source_feasibility(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, bool], tuple[str, ...]]:
    feasibility: dict[str, bool] = {}
    for record in manifest.get("records") or ():
        record_id = str(record.get("record_id") or "")
        fixed_c = (record.get("context") or {}).get("fixed_c_reference") or {}
        feasible = fixed_c.get("all_affected_boundaries_feasible")
        if not record_id or not isinstance(feasible, bool):
            raise ValueError("manifest fixed-C feasibility is incomplete")
        feasibility[record_id] = feasible
    infeasible = tuple(
        record_id for record_id, feasible in feasibility.items() if not feasible
    )
    if len(feasibility) != EXPECTED_FROZEN_RECORDS:
        raise ValueError("confirmation manifest must contain exactly 16 records")
    if len(infeasible) != EXPECTED_SOURCE_INFEASIBLE:
        raise ValueError("confirmation manifest must contain two infeasible records")
    return feasibility, infeasible


def _validate_locked_artifacts(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
) -> None:
    admission._validate_pinned_manifest_payload(manifest)
    if manifest.get("partition") != benchmark.CONFIRMATION_PARTITION:
        raise ValueError("method evaluation requires the confirmation manifest")
    benchmark.validate_policy_lock(policy_lock)
    if policy_lock != manifest.get("policy_lock"):
        raise ValueError("standalone policy lock differs from the manifest lock")
    admission.validate_promotion_lock(promotion_lock, manifest)
    if (
        str((promotion_lock.get("integrity") or {}).get("sha256") or "")
        != PROMOTION_LOCK_INTEGRITY_SHA256
    ):
        raise ValueError("promotion lock is not the committed frozen artifact")


def load_locked_artifacts(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    policy_lock_path: str | Path = DEFAULT_POLICY_LOCK,
    promotion_lock_path: str | Path = DEFAULT_PROMOTION_LOCK,
    confirmation_report_path: str | Path = DEFAULT_CONFIRMATION_REPORT,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    _require_exact_file(
        manifest_path,
        CONFIRMATION_MANIFEST_FILE_SHA256,
        name="confirmation manifest",
    )
    _require_exact_file(
        policy_lock_path,
        POLICY_LOCK_FILE_SHA256,
        name="policy lock",
    )
    _require_exact_file(
        promotion_lock_path,
        PROMOTION_LOCK_FILE_SHA256,
        name="promotion lock",
    )
    _require_exact_file(
        confirmation_report_path,
        CONFIRMATION_REPORT_SHA256,
        name="confirmation admission report",
    )
    manifest, policy_lock = admission.load_locked_inputs(
        manifest_path,
        policy_lock_path,
    )
    promotion_lock = _load_mapping(promotion_lock_path, name="promotion lock")
    report = _load_mapping(
        confirmation_report_path,
        name="confirmation report",
    )
    _validate_locked_artifacts(manifest, policy_lock, promotion_lock)
    _validate_confirmation_report(
        report,
        report_sha256=CONFIRMATION_REPORT_SHA256,
        manifest=manifest,
        policy_lock=policy_lock,
        promotion_lock=promotion_lock,
    )
    return manifest, policy_lock, promotion_lock, report


def _admission_flags(row: Mapping[str, Any]) -> tuple[bool, bool, bool]:
    admission_payload = row.get("admission") or {}
    target = bool((admission_payload.get("target_recall") or {}).get("admitted"))
    retained = bool(
        (admission_payload.get("retained_availability") or {}).get("available")
    )
    joint = bool(
        (admission_payload.get("joint_target_and_retained") or {}).get(
            "admitted"
        )
    )
    if joint != (target and retained):
        raise ValueError("confirmation report has inconsistent joint admission")
    return target, retained, joint


def _validate_confirmation_report(
    report: Mapping[str, Any],
    *,
    report_sha256: str,
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
) -> dict[str, Any]:
    admission._assert_source_free_payload(report)
    if report_sha256 != CONFIRMATION_REPORT_SHA256:
        raise ValueError("method lock requires the exact confirmation report")
    if _payload_sha256(report) != CONFIRMATION_REPORT_PAYLOAD_SHA256:
        raise ValueError("confirmation report payload is not the exact artifact")
    if (
        report.get("schema") != admission.SCHEMA
        or report.get("status") != "completed"
        or report.get("contains_source_text") is not False
        or report.get("contains_full_vocabulary_vectors") is not False
        or report.get("model_scoring_used_for_selection_or_replacement")
        is not False
    ):
        raise ValueError("confirmation admission report contract is invalid")
    implementation = dict(report.get("implementation") or {})
    reported_contract = implementation.pop("contract_sha256", None)
    if (
        reported_contract != CONFIRMATION_IMPLEMENTATION_CONTRACT_SHA256
        or _payload_sha256(implementation) != reported_contract
    ):
        raise ValueError("confirmation implementation contract is invalid")
    manifest_report = report.get("manifest") or {}
    policy_report = report.get("policy_lock") or {}
    promotion_report = report.get("promotion_lock") or {}
    if (
        manifest_report.get("file_sha256")
        != CONFIRMATION_MANIFEST_FILE_SHA256
        or manifest_report.get("integrity_sha256")
        != admission.PINNED_MANIFEST_INTEGRITIES[
            benchmark.CONFIRMATION_PARTITION
        ]
        or policy_report.get("file_sha256") != POLICY_LOCK_FILE_SHA256
        or policy_report.get("lock_sha256")
        != admission.PINNED_POLICY_LOCK_SHA256
        or promotion_report.get("file_sha256")
        != PROMOTION_LOCK_FILE_SHA256
        or promotion_report.get("integrity_sha256")
        != PROMOTION_LOCK_INTEGRITY_SHA256
    ):
        raise ValueError("confirmation report artifact bindings are invalid")
    expected_config = admission.runtime_contract(
        admission.GRAFTED_ARM,
        device=str((promotion_lock.get("decision") or {}).get("device") or ""),
    )
    report_config = dict(report.get("config") or {})
    for key, value in expected_config.items():
        if report_config.get(key) != value:
            raise ValueError("confirmation runtime differs from promotion")

    manifest_ids = [
        str(record.get("record_id") or "")
        for record in manifest.get("records") or ()
    ]
    rows = list(report.get("records") or ())
    row_ids = [str(row.get("record_id") or "") for row in rows]
    if (
        len(rows) != EXPECTED_FROZEN_RECORDS
        or row_ids != manifest_ids
        or len(set(row_ids)) != EXPECTED_FROZEN_RECORDS
    ):
        raise ValueError("confirmation report must cover all 16 records in order")
    feasibility, infeasible_ids = _source_feasibility(manifest)
    target_count = retained_count = joint_count = 0
    admission_records = []
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if row.get("status") != "completed":
            raise ValueError("all confirmation records must be complete")
        immutability = row.get("source_state_immutability") or {}
        if (
            immutability.get("unchanged") is not True
            or immutability.get("shape_signature_unchanged") is not True
        ):
            raise ValueError("confirmation source state was not immutable")
        target, retained, joint = _admission_flags(row)
        target_count += int(target)
        retained_count += int(retained)
        joint_count += int(joint)
        reported_feasible = bool(
            (row.get("fixed_c_source_precheck") or {}).get(
                "all_affected_boundaries_feasible"
            )
        )
        if reported_feasible != feasibility[record_id]:
            raise ValueError("confirmation fixed-C feasibility differs")
        admission_records.append(
            {
                "record_id": record_id,
                "confirmation_record_sha256": _payload_sha256(row),
                "target_admitted": target,
                "retained_available": retained,
                "joint_admitted": joint,
                "source_fixed_c_feasible": feasibility[record_id],
            }
        )

    denominators = (report.get("summary") or {}).get("denominators") or {}
    required_denominators = {
        "frozen_records": EXPECTED_FROZEN_RECORDS,
        "selected_records": EXPECTED_FROZEN_RECORDS,
        "attempted_records": EXPECTED_FROZEN_RECORDS,
        "completed_records": EXPECTED_FROZEN_RECORDS,
        "record_failures": 0,
        "target_admitted": EXPECTED_TARGET_ADMITTED,
        "retained_available": EXPECTED_RETAINED_AVAILABLE,
        "joint_admitted": EXPECTED_JOINT_ADMITTED,
        "source_state_immutability_failures": 0,
        "fixed_c_all_boundaries_feasible_records": (
            EXPECTED_FROZEN_RECORDS - EXPECTED_SOURCE_INFEASIBLE
        ),
        "fixed_c_infeasible_records": EXPECTED_SOURCE_INFEASIBLE,
    }
    if any(
        int(denominators.get(key, -1)) != expected
        for key, expected in required_denominators.items()
    ):
        raise ValueError("confirmation admission denominators are not authorized")
    if (
        target_count != EXPECTED_TARGET_ADMITTED
        or retained_count != EXPECTED_RETAINED_AVAILABLE
        or joint_count != EXPECTED_JOINT_ADMITTED
    ):
        raise ValueError("confirmation record outcomes differ from the summary")
    if report.get("summary", {}).get("failure_record_ids") not in ([], ()):
        raise ValueError("confirmation report contains failures")
    return {
        "admission_records": admission_records,
        "joint_record_ids": [
            row["record_id"] for row in admission_records if row["joint_admitted"]
        ],
        "source_infeasible_record_ids": list(infeasible_ids),
        "report_payload_sha256": _payload_sha256(report),
    }


def _method_lock_body(
    *,
    report_analysis: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    prompt_suppression: bool,
) -> dict[str, Any]:
    joint_ids = list(report_analysis["joint_record_ids"])
    infeasible_ids = list(report_analysis["source_infeasible_record_ids"])
    return {
        "schema": METHOD_LOCK_SCHEMA,
        "status": METHOD_LOCK_STATUS,
        "contains_source_text": False,
        "selection_uses_method_outputs": False,
        "confirmation_artifacts": {
            "manifest_file_sha256": CONFIRMATION_MANIFEST_FILE_SHA256,
            "manifest_integrity_sha256": admission.PINNED_MANIFEST_INTEGRITIES[
                benchmark.CONFIRMATION_PARTITION
            ],
            "policy_lock_file_sha256": POLICY_LOCK_FILE_SHA256,
            "policy_lock_sha256": admission.PINNED_POLICY_LOCK_SHA256,
            "promotion_lock_file_sha256": PROMOTION_LOCK_FILE_SHA256,
            "promotion_lock_integrity_sha256": (
                PROMOTION_LOCK_INTEGRITY_SHA256
            ),
            "confirmation_report_file_sha256": CONFIRMATION_REPORT_SHA256,
            "confirmation_report_payload_sha256": report_analysis[
                "report_payload_sha256"
            ],
            "confirmation_implementation_contract_sha256": (
                CONFIRMATION_IMPLEMENTATION_CONTRACT_SHA256
            ),
        },
        "authorization": {
            "all_records_without_replacement": True,
            "frozen_records": EXPECTED_FROZEN_RECORDS,
            "target_admitted": EXPECTED_TARGET_ADMITTED,
            "retained_available": EXPECTED_RETAINED_AVAILABLE,
            "joint_admitted": EXPECTED_JOINT_ADMITTED,
            "record_failures": 0,
            "runtime": dict(promotion_lock["decision"]),
            "condition_ids": list(
                expected_condition_ids(
                    prompt_suppression=bool(prompt_suppression)
                )
            ),
            "prompt_suppression": {
                "enabled": bool(prompt_suppression),
                "valid_gemma_chat_required": True,
                "instruction_target_free_required": True,
                "persistent_state_edit": False,
            },
            "exact_decrement": {
                "fixed_c": True,
                "source_feasible_records": 14,
                "source_infeasible_records": 2,
                "source_infeasible_policy": "explicit_full_raw_omission_repack",
                "source_feasible_execution_strata": [
                    "incremental_float64_fixed_c_decrement",
                    "fixed_c_refit_fallback",
                ],
                "source_infeasible_record_ids": infeasible_ids,
                "source_infeasible_record_ids_sha256": _payload_sha256(
                    infeasible_ids
                ),
            },
            "fixed_c_refit": {
                "retained_keys_only": True,
                "source_infeasible_policy": "not_applicable",
            },
            "fp32_proxy": {
                "source_feasible_policy": "incremental_masked_refit",
                "source_infeasible_policy": "full_raw_omission_repack",
                "expected_full_repack_fallback_records": 2,
            },
            "token_row_repack": {
                "diagnostic_only": True,
                "execution": "zero_cost_alias",
                "alias_of": FRESH_RAW_OMISSION_ID,
                "additional_model_execution": False,
                "registered_turn_boundary_contract": True,
            },
            "resume": {
                "existing_report_policy": "validate_then_discard_all_rows",
                "completed_rows_reused": False,
                "all_authorized_records_reexecuted": True,
            },
            "efficacy_population": {
                "name": "predeclared_confirmation_joint12",
                "record_ids": joint_ids,
                "record_ids_sha256": _payload_sha256(joint_ids),
                "records": EXPECTED_JOINT_ADMITTED,
                "selected_before_method_scoring": True,
            },
            "metrics": {
                "complete_target_sequences": True,
                "first_target_token_rank": True,
                "first_target_token_full_vocabulary_kl": True,
                "retained_drift": True,
                "exact_refit_first_token_full_vocabulary_kl": True,
                "timing": True,
                "storage": True,
                "source_state_immutability": True,
            },
            "admission_records": list(report_analysis["admission_records"]),
        },
    }


def freeze_method_authorization_lock(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    *,
    confirmation_report_sha256: str = CONFIRMATION_REPORT_SHA256,
    prompt_suppression: bool = False,
) -> dict[str, Any]:
    """Freeze a source-free authorization lock before method model scoring."""

    _validate_locked_artifacts(manifest, policy_lock, promotion_lock)
    analysis = _validate_confirmation_report(
        confirmation_report,
        report_sha256=confirmation_report_sha256,
        manifest=manifest,
        policy_lock=policy_lock,
        promotion_lock=promotion_lock,
    )
    lock = _method_lock_body(
        report_analysis=analysis,
        promotion_lock=promotion_lock,
        prompt_suppression=prompt_suppression,
    )
    lock["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(lock),
    }
    admission._assert_source_free_payload(lock)
    return lock


def validate_method_authorization_lock(
    lock: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    confirmation_report_sha256: str = CONFIRMATION_REPORT_SHA256,
) -> None:
    """Validate lock integrity and reproduce every authorization decision."""

    admission._assert_source_free_payload(lock)
    if lock.get("schema") != METHOD_LOCK_SCHEMA:
        raise ValueError("unsupported method authorization lock schema")
    if lock.get("status") != METHOD_LOCK_STATUS:
        raise ValueError("method authorization lock is not frozen")
    integrity = lock.get("integrity") or {}
    unsigned = dict(lock)
    unsigned.pop("integrity", None)
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise ValueError("method authorization lock integrity mismatch")
    expected = freeze_method_authorization_lock(
        manifest,
        policy_lock,
        promotion_lock,
        confirmation_report,
        confirmation_report_sha256=confirmation_report_sha256,
        prompt_suppression=bool(
            ((lock.get("authorization") or {}).get("prompt_suppression") or {}).get(
                "enabled"
            )
        ),
    )
    if dict(lock) != expected:
        raise ValueError("method authorization lock differs from frozen inputs")


def authorize_method_execution(
    lock: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    confirmation_report_sha256: str = CONFIRMATION_REPORT_SHA256,
    explicit_acknowledgement: bool,
) -> tuple[str, str]:
    validate_method_authorization_lock(
        lock,
        manifest=manifest,
        policy_lock=policy_lock,
        promotion_lock=promotion_lock,
        confirmation_report=confirmation_report,
        confirmation_report_sha256=confirmation_report_sha256,
    )
    if not explicit_acknowledgement:
        raise PermissionError(
            "deletion method model scoring requires explicit acknowledgement"
        )
    runtime = (lock.get("authorization") or {}).get("runtime") or {}
    arm = str(runtime.get("arm") or "")
    device = str(runtime.get("device") or "")
    if arm != admission.GRAFTED_ARM or not device:
        raise ValueError("method authorization runtime is invalid")
    return arm, device


def load_committed_method_lock(
    path: str | Path,
    *,
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    confirmation_report: Mapping[str, Any],
    require_canonical_path: bool = False,
) -> dict[str, Any]:
    lock_path = Path(path)
    if (
        require_canonical_path
        and lock_path.resolve() != DEFAULT_METHOD_LOCK.resolve()
    ):
        raise PermissionError(
            "confirmation methods require the canonical committed method lock"
        )
    lock = _load_mapping(lock_path, name="method authorization lock")
    validate_method_authorization_lock(
        lock,
        manifest=manifest,
        policy_lock=policy_lock,
        promotion_lock=promotion_lock,
        confirmation_report=confirmation_report,
        confirmation_report_sha256=CONFIRMATION_REPORT_SHA256,
    )
    return lock


def build_probes(
    record: benchmark.RehydratedChatRecord,
) -> tuple[baseline.Probe, ...]:
    probes = []
    expected = {"target_current": "deleted", "retained": "retained"}
    for probe in record.probes:
        kind = expected.get(probe.probe_id)
        if kind is None or probe.kind != kind:
            raise ValueError("chat record has an unsupported probe contract")
        if not probe.target_token_ids:
            raise ValueError("chat probe target tokenization is empty")
        probes.append(
            baseline.Probe(
                probe_id=probe.probe_id,
                kind=kind,
                prompt=probe.prompt_text,
                target=probe.answer,
                target_ids=tuple(probe.target_token_ids),
                field_name=probe.probe_id,
            )
        )
    if {probe.probe_id for probe in probes} != set(expected):
        raise ValueError("chat record requires target_current and retained probes")
    return tuple(probes)


def _prompt_suppression_probes(
    record: benchmark.RehydratedChatRecord,
    probes: Sequence[baseline.Probe],
) -> tuple[baseline.Probe, ...]:
    """Inject a generic instruction inside the existing registered user turn."""

    target_values = [probe.answer for probe in record.probes]
    if any(
        str(value).strip()
        and str(value).casefold() in PROMPT_SUPPRESSION_INSTRUCTION.casefold()
        for value in target_values
    ):
        raise ValueError("prompt suppression instruction contains a target")
    result = []
    marker = benchmark.CONSTRAINED_INSTRUCTION
    prefix = "<start_of_turn>user\n"
    suffix = "<end_of_turn>\n<start_of_turn>model\n"
    for probe in probes:
        if (
            not probe.prompt.startswith(prefix)
            or not probe.prompt.endswith(suffix)
            or probe.prompt.count(marker) != 1
        ):
            raise ValueError("probe is not a registered Gemma chat query")
        modified = probe.prompt.replace(
            marker,
            f"{PROMPT_SUPPRESSION_INSTRUCTION}\n{marker}",
            1,
        )
        content = modified[len(prefix) : -len(suffix)]
        expected = (
            benchmark.render_registered_turn("user", content)
            + "<start_of_turn>model\n"
        )
        if modified != expected or "<bos>" in modified:
            raise ValueError("prompt suppression is not valid incremental chat")
        result.append(
            baseline.Probe(
                probe_id=probe.probe_id,
                kind=probe.kind,
                prompt=modified,
                target=probe.target,
                target_ids=probe.target_ids,
                field_name=probe.field_name,
            )
        )
    return tuple(result)


def _record_seed(record_id: str) -> int:
    digest = hashlib.sha256(
        f"longmemeval-chat-methods-v1\0{EVALUATION_SEED}\0{record_id}".encode()
    ).digest()
    return EVALUATION_SEED + int.from_bytes(digest[:4], "big")


def _score_suite(
    runtime: Any,
    state: baseline.QueryState,
    probes: Sequence[baseline.Probe],
) -> dict[str, dict[str, Any]]:
    return baseline._score_probe_suite(runtime, state, probes)


def _timing_scope(timing: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(timing)
    result["query_scope"] = (
        "full target_current and retained sequences plus first-token "
        "rank and full-vocabulary KL"
    )
    return result


def _zero_cost_alias_timing(
    *,
    device: str,
    warmup: int,
    repeats: int,
    alias_of: str,
) -> dict[str, Any]:
    timing = baseline._timing_payload(
        [0.0] * repeats,
        [0.0] * repeats,
        [0.0] * repeats,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    timing["zero_cost_alias"] = True
    timing["alias_of"] = alias_of
    for scope in ("update_seconds", "query_seconds", "end_to_end_seconds"):
        timing[scope]["not_applicable"] = True
        timing[scope]["reason"] = "zero-cost alias; no additional execution"
    return timing


def _condition_report(
    condition_id: str,
    state: baseline.QueryState,
    scores: Mapping[str, Mapping[str, Any]],
    timing: Mapping[str, Any],
    *,
    original_memory: Any,
    probes: Sequence[baseline.Probe],
    present_scores: Mapping[str, Mapping[str, Any]],
    raw_scores: Mapping[str, Mapping[str, Any]],
    semantics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "status": "completed",
        "semantics": dict(semantics),
        "tokenized_state": {
            "prefill_input_digest": str(state.memory.input_digest),
            "token_count": int(state.memory.token_count),
            "added_query_instruction_tokens": int(state.prompt_token_count),
        },
        **longitudinal._behavioral_metrics(
            probes,
            scores,
            present_scores,
            raw_scores,
        ),
        "timing": _timing_scope(timing),
        "storage": method_storage_report(
            original_memory,
            state.memory,
            prompt_token_count=state.prompt_token_count,
        ),
        "update_diagnostics": state.update_diagnostics or {},
    }


def _failed_condition(condition_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message_sha256": benchmark.base.text_sha256(str(exc)),
        "error_message_redacted": True,
    }


def _not_applicable_condition(
    condition_id: str,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "status": "not_applicable",
        "reason": reason,
        "source_predeclared": True,
    }


def _fallback_used(memory: Any) -> bool:
    return str(getattr(memory, "deletion_kind", "")).endswith(
        "_full_repack_fallback"
    )


def _prefill_original(runtime: Any, record: benchmark.RehydratedChatRecord):
    baseline._synchronize(runtime.config.device)
    started = time.perf_counter()
    memory = runtime.prefill_persistent(list(record.context.original_token_ids))
    baseline._synchronize(runtime.config.device)
    elapsed = time.perf_counter() - started
    expected = benchmark.base.token_ids_sha256(record.context.original_token_ids)
    if str(memory.input_digest) != expected:
        raise RuntimeError("original persistent prefill digest differs")
    return memory, elapsed


def evaluate_record(
    runtime: Any,
    record: benchmark.RehydratedChatRecord,
    *,
    admission_binding: Mapping[str, Any],
    source_fixed_c_feasible: bool,
    prompt_suppression: bool,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Evaluate every predeclared condition for one frozen chat record."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    if admission_binding.get("record_id") != record.record_id:
        raise ValueError("admission binding record ID differs")
    if (
        bool(admission_binding.get("source_fixed_c_feasible"))
        != bool(source_fixed_c_feasible)
    ):
        raise ValueError("source fixed-C feasibility differs from the lock")
    if tuple(record.context.edited_token_ids) != tuple(
        record.raw_omitted_token_ids
    ):
        raise ValueError(
            "registered-turn token-row deletion must equal fresh raw omission"
        )
    record_seed = _record_seed(record.record_id)
    baseline._seed_everything(record_seed)
    probes = build_probes(record)
    query = lambda state: _score_suite(runtime, state, probes)
    original_memory, prefill_seconds = _prefill_original(runtime, record)
    source_shape = persistent_state_shape_signature(original_memory)
    fingerprint_before = longitudinal._persistent_state_fingerprint(
        original_memory
    )
    row: dict[str, Any] = {
        "record_id": record.record_id,
        "record_seed": record_seed,
        "status": "running",
        "admission_binding": dict(admission_binding),
        "predeclared_joint_admitted": bool(
            admission_binding["joint_admitted"]
        ),
        "source_fixed_c_feasible": bool(source_fixed_c_feasible),
        "probes": [
            {
                "probe_id": probe.probe_id,
                "probe_kind": probe.kind,
                "target_token_count": len(probe.target_ids),
                "target_token_ids_sha256": benchmark.base.token_ids_sha256(
                    probe.target_ids
                ),
            }
            for probe in probes
        ],
        "shared_original_prefill": {
            "seconds": prefill_seconds,
            "synchronized_device": runtime.config.device,
            "input_digest": str(original_memory.input_digest),
            "storage": method_storage_report(
                original_memory,
                original_memory,
            )["state"],
        },
        "conditions": {},
    }

    present_state = baseline.QueryState(
        original_memory,
        update_diagnostics={"reference_kind": "present"},
    )
    present_scores, present_timing = baseline._benchmark_existing_state(
        state=present_state,
        query=query,
        device=runtime.config.device,
        warmup=warmup,
        repeats=repeats,
    )
    raw_state, raw_scores, raw_timing = baseline._benchmark_method(
        build=lambda: baseline.QueryState(
            runtime.prefill_persistent(list(record.raw_omitted_token_ids)),
            update_diagnostics={
                "reference_kind": "fresh_raw_omission",
                "freshly_retokenized": True,
                "suffix_recomputed": True,
            },
        ),
        query=query,
        device=runtime.config.device,
        warmup=warmup,
        repeats=repeats,
        operation_seed=record_seed + 1,
    )
    row["conditions"][PRESENT_ID] = _condition_report(
        PRESENT_ID,
        present_state,
        present_scores,
        present_timing,
        original_memory=original_memory,
        probes=probes,
        present_scores=present_scores,
        raw_scores=raw_scores,
        semantics={"reference": True, "owned_round_present": True},
    )
    row["conditions"][FRESH_RAW_OMISSION_ID] = _condition_report(
        FRESH_RAW_OMISSION_ID,
        raw_state,
        raw_scores,
        raw_timing,
        original_memory=original_memory,
        probes=probes,
        present_scores=present_scores,
        raw_scores=raw_scores,
        semantics={
            "behavioral_reference": True,
            "executed_method": "fresh_raw_omission_full_repack",
            "fresh_prefill": True,
            "raw_owned_round_omitted": True,
            "freshly_retokenized": True,
            "suffix_recomputed": True,
        },
    )

    token_alias = _condition_report(
        TOKEN_ROW_DIAGNOSTIC_ID,
        raw_state,
        raw_scores,
        _zero_cost_alias_timing(
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            alias_of=FRESH_RAW_OMISSION_ID,
        ),
        original_memory=original_memory,
        probes=probes,
        present_scores=present_scores,
        raw_scores=raw_scores,
        semantics={
            "diagnostic_only": True,
            "executed_method": "zero_cost_alias",
            "zero_cost_alias": True,
            "alias_of": FRESH_RAW_OMISSION_ID,
            "additional_model_execution": False,
            "registered_turn_boundary_contract": True,
            "identical_token_ids": True,
        },
    )
    aliased_incremental_storage = token_alias["storage"][
        "incremental_tensor_storage_bytes"
    ]
    token_alias["storage"]["incremental_tensor_storage_bytes"] = 0
    token_alias["storage"][
        "aliased_reference_incremental_tensor_storage_bytes"
    ] = aliased_incremental_storage
    token_alias["storage"]["zero_cost_alias"] = True
    token_alias["storage"]["alias_of"] = FRESH_RAW_OMISSION_ID
    row["conditions"][TOKEN_ROW_DIAGNOSTIC_ID] = token_alias

    try:
        def build_proxy_state():
            memory = runtime.delete_persistent(
                original_memory,
                record.context.forget_positions,
                kind="fp32_masked_refit_proxy",
            )
            fallback = _fallback_used(memory)
            return baseline.QueryState(
                memory,
                update_diagnostics={
                    "solver": (
                        "full raw-omission repack fallback"
                        if fallback
                        else "projected FP32 FISTA masked refit"
                    ),
                    "full_repack_fallback": fallback,
                    "fallback_reason_sha256": (
                        benchmark.base.text_sha256(
                            str(getattr(memory, "fallback_reason", ""))
                        )
                        if fallback
                        else None
                    ),
                },
            )

        proxy_state, proxy_scores, proxy_timing = baseline._benchmark_method(
            build=build_proxy_state,
            query=query,
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            operation_seed=record_seed + 3,
        )
        proxy_fallback = bool(
            proxy_state.update_diagnostics["full_repack_fallback"]
        )
        if proxy_fallback != (not source_fixed_c_feasible):
            raise RuntimeError(
                "FP32 proxy fallback differs from source feasibility lock"
            )
        row["conditions"][FP32_PROXY_ID] = _condition_report(
            FP32_PROXY_ID,
            proxy_state,
            proxy_scores,
            proxy_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_scores=raw_scores,
            semantics={
                "solver": proxy_state.update_diagnostics["solver"],
                "executed_method": (
                    "fresh_raw_omission_full_repack_fallback"
                    if proxy_fallback
                    else "incremental_fp32_masked_refit"
                ),
                "model_forward_precision": "float32",
                "query_independent": True,
                "full_repack_fallback": proxy_fallback,
            },
        )
    except Exception as exc:
        row["conditions"][FP32_PROXY_ID] = _failed_condition(
            FP32_PROXY_ID,
            exc,
        )

    try:
        def build_decay_state():
            memory = original_memory.fork()
            memory.request = GateRequest(
                scale_pos=tuple(record.context.forget_positions),
                scale_factor=DECAY_FACTOR,
                gate_floor=float(
                    getattr(
                        getattr(original_memory, "request", None),
                        "gate_floor",
                        0.0,
                    )
                ),
            )
            memory.deleted_positions = tuple(record.context.forget_positions)
            memory.deletion_kind = "coefficient_decay_0_01"
            return baseline.QueryState(
                memory,
                update_diagnostics={
                    "decay_factor": DECAY_FACTOR,
                    "solver_refit": False,
                },
            )

        decay_state, decay_scores, decay_timing = baseline._benchmark_method(
            build=build_decay_state,
            query=query,
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            operation_seed=record_seed + 4,
        )
        row["conditions"][DECAY_ID] = _condition_report(
            DECAY_ID,
            decay_state,
            decay_scores,
            decay_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_scores=raw_scores,
            semantics={
                "decay_factor": DECAY_FACTOR,
                "positions_remain_resident": True,
                "solver_refit": False,
                "suffix_recomputed": False,
            },
        )
    except Exception as exc:
        row["conditions"][DECAY_ID] = _failed_condition(DECAY_ID, exc)

    try:
        def build_cache_state():
            memory, diagnostics = cache_delete_and_shift(
                original_memory,
                record.context.forget_positions,
                edited_token_ids=record.context.edited_token_ids,
            )
            diagnostics["raw_omission_token_ids_sha256"] = (
                benchmark.base.token_ids_sha256(record.raw_omitted_token_ids)
            )
            return baseline.QueryState(memory, update_diagnostics=diagnostics)

        cache_state, cache_scores, cache_timing = baseline._benchmark_method(
            build=build_cache_state,
            query=query,
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            operation_seed=record_seed + 5,
        )
        row["conditions"][CACHE_DELETE_SHIFT_ID] = _condition_report(
            CACHE_DELETE_SHIFT_ID,
            cache_state,
            cache_scores,
            cache_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_scores=raw_scores,
            semantics={
                "diagnostic_only": True,
                "physically_deletes_matching_cache_rows": True,
                "solver_refit": False,
                "suffix_recomputed": False,
                "rope_keys_rerotated": False,
            },
        )
    except Exception as exc:
        row["conditions"][CACHE_DELETE_SHIFT_ID] = _failed_condition(
            CACHE_DELETE_SHIFT_ID,
            exc,
        )

    if source_fixed_c_feasible:
        try:
            (
                certificate_states,
                certificate_scores,
                certificate_timings,
                solver_diagnostics,
            ) = baseline._benchmark_certificate_pair(
                runtime,
                original_memory,
                record.context.forget_positions,
                probes,
                device=runtime.config.device,
                warmup=warmup,
                repeats=repeats,
                operation_seed=record_seed + 6,
            )
            if any(
                _fallback_used(certificate_states[name].memory)
                for name in ("exact", "refit")
            ):
                raise RuntimeError(
                    "source-feasible fixed-C certificate used a repack fallback"
                )
            used_refit_fallback = bool(
                solver_diagnostics.get("used_refit_fallback")
                or int(solver_diagnostics.get("decrement_fallbacks", 0)) > 0
            )
            exact_execution = (
                "fixed_c_refit_fallback"
                if used_refit_fallback
                else "incremental_float64_fixed_c_decrement"
            )
            row["conditions"][EXACT_DECREMENT_ID] = _condition_report(
                EXACT_DECREMENT_ID,
                certificate_states["exact"],
                certificate_scores["exact"],
                certificate_timings["exact"],
                original_memory=original_memory,
                probes=probes,
                present_scores=present_scores,
                raw_scores=raw_scores,
                semantics={
                    "executed_method": exact_execution,
                    "gate_solver": "float64 Cauwenberghs-Poggio decrement",
                    "fixed_C": True,
                    "query_independent": True,
                    "full_repack_fallback": False,
                    "fixed_c_refit_fallback": used_refit_fallback,
                    "incremental_exact": not used_refit_fallback,
                    "decrement_fallbacks": int(
                        solver_diagnostics.get("decrement_fallbacks", 0)
                    ),
                    "certificate_reference": FIXED_C_REFIT_ID,
                },
            )
            row["conditions"][FIXED_C_REFIT_ID] = _condition_report(
                FIXED_C_REFIT_ID,
                certificate_states["refit"],
                certificate_scores["refit"],
                certificate_timings["refit"],
                original_memory=original_memory,
                probes=probes,
                present_scores=present_scores,
                raw_scores=raw_scores,
                semantics={
                    "executed_method": (
                        "float64_fixed_c_retained_key_refit"
                    ),
                    "gate_solver": "float64 retained-key from-scratch refit",
                    "fixed_C": True,
                    "query_independent": True,
                    "retained_keys_only": True,
                    "conditional_on_contextualized_retained_keys": True,
                },
            )
            row["solver_certificate"] = longitudinal._solver_certificate(
                probes,
                certificate_scores["exact"],
                certificate_scores["refit"],
                solver_diagnostics,
            )
        except Exception as exc:
            row["conditions"][EXACT_DECREMENT_ID] = _failed_condition(
                EXACT_DECREMENT_ID,
                exc,
            )
            row["conditions"][FIXED_C_REFIT_ID] = _failed_condition(
                FIXED_C_REFIT_ID,
                exc,
            )
            row["solver_certificate"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message_sha256": benchmark.base.text_sha256(str(exc)),
                "error_message_redacted": True,
            }
    else:
        row["conditions"][EXACT_DECREMENT_ID] = _condition_report(
            EXACT_DECREMENT_ID,
            raw_state,
            raw_scores,
            raw_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_scores=raw_scores,
            semantics={
                "executed_method": (
                    "fresh_raw_omission_full_repack_fallback"
                ),
                "requested_solver": "float64 Cauwenberghs-Poggio decrement",
                "fixed_C": True,
                "source_fixed_c_feasible": False,
                "full_repack_fallback": True,
                "fallback_kind": "fresh_raw_omission_repack",
                "fallback_predeclared_before_method_scoring": True,
                "shared_measurement_with": FRESH_RAW_OMISSION_ID,
            },
        )
        row["conditions"][FIXED_C_REFIT_ID] = _not_applicable_condition(
            FIXED_C_REFIT_ID,
            reason="source_fixed_c_infeasible",
        )
        row["solver_certificate"] = {
            "status": "not_applicable",
            "reason": "source_fixed_c_infeasible",
            "source_predeclared": True,
        }

    if prompt_suppression:
        try:
            suppression_probes = _prompt_suppression_probes(record, probes)
            suppression_query = lambda state: _score_suite(
                runtime,
                state,
                suppression_probes,
            )
            encoded = runtime.tokenizer(
                "\n" + PROMPT_SUPPRESSION_INSTRUCTION,
                add_special_tokens=False,
            )
            raw_ids = (
                encoded["input_ids"]
                if isinstance(encoded, Mapping)
                else encoded.input_ids
            )
            instruction_tokens = len(raw_ids)
            suppression_state, suppression_scores, suppression_timing = (
                baseline._benchmark_method(
                    build=lambda: baseline.QueryState(
                        original_memory,
                        prompt_token_count=instruction_tokens,
                        update_diagnostics={
                            "prompt_only_behavioral_control": True,
                            "persistent_state_deleted": False,
                            "base_state_unchanged": True,
                            "valid_registered_gemma_chat": True,
                            "instruction_target_free": True,
                            "instruction_sha256": benchmark.base.text_sha256(
                                PROMPT_SUPPRESSION_INSTRUCTION
                            ),
                            "instruction_token_count": instruction_tokens,
                        },
                    ),
                    query=suppression_query,
                    device=runtime.config.device,
                    warmup=warmup,
                    repeats=repeats,
                    operation_seed=record_seed + 7,
                )
            )
            row["conditions"][PROMPT_SUPPRESSION_ID] = _condition_report(
                PROMPT_SUPPRESSION_ID,
                suppression_state,
                suppression_scores,
                suppression_timing,
                original_memory=original_memory,
                probes=suppression_probes,
                present_scores=present_scores,
                raw_scores=raw_scores,
                semantics={
                    "prompt_only_behavioral_control": True,
                    "persistent_state_deleted": False,
                    "base_state_unchanged": True,
                    "valid_registered_gemma_chat": True,
                    "instruction_target_free": True,
                },
            )
        except Exception as exc:
            row["conditions"][PROMPT_SUPPRESSION_ID] = _failed_condition(
                PROMPT_SUPPRESSION_ID,
                exc,
            )

    fingerprint_after = longitudinal._persistent_state_fingerprint(
        original_memory
    )
    shape_unchanged = (
        persistent_state_shape_signature(original_memory) == source_shape
    )
    source_unchanged = (
        shape_unchanged and fingerprint_after == fingerprint_before
    )
    row["source_state_immutability"] = {
        "verification_scope": "metadata_shapes_and_complete_tensor_values",
        "before_sha256": fingerprint_before,
        "after_sha256": fingerprint_after,
        "shape_signature_unchanged": shape_unchanged,
        "unchanged": source_unchanged,
    }
    if not source_unchanged:
        raise RuntimeError("a deletion condition mutated the shared source state")

    expected_ids = expected_condition_ids(
        prompt_suppression=prompt_suppression
    )
    if set(row["conditions"]) != set(expected_ids):
        raise RuntimeError("method condition matrix is incomplete")
    row["method_failures"] = [
        condition_id
        for condition_id in expected_ids
        if row["conditions"][condition_id]["status"] == "failed"
    ]
    row["status"] = (
        "completed" if not row["method_failures"] else "completed_with_failures"
    )
    admission._assert_source_free_payload(row)
    return row


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(float(value) for value in values) / len(values))


def _condition_summary(
    rows: Sequence[Mapping[str, Any]],
    condition_id: str,
) -> dict[str, Any]:
    conditions = [
        row.get("conditions", {}).get(condition_id, {})
        for row in rows
    ]
    completed = [
        condition
        for condition in conditions
        if condition.get("status") == "completed"
    ]
    explicit_failures = sum(
        condition.get("status") == "failed" for condition in conditions
    )
    missing = sum(not condition for condition in conditions)
    not_applicable = sum(
        condition.get("status") == "not_applicable"
        for condition in conditions
    )
    accounted = len(completed) + explicit_failures + missing + not_applicable
    execution_labels: dict[str, int] = {}
    for condition in completed:
        label = str((condition.get("semantics") or {}).get("executed_method") or "")
        if label:
            execution_labels[label] = execution_labels.get(label, 0) + 1
    result = {
        "authorized_records": len(rows),
        "accounted_records": accounted,
        "unaccounted_records": len(rows) - accounted,
        "fully_accounted": accounted == len(rows),
        "completed_records": len(completed),
        "failed_records": explicit_failures,
        "explicit_failure_record_ids": [
            str(row.get("record_id") or "")
            for row, condition in zip(rows, conditions)
            if condition.get("status") == "failed"
        ],
        "missing_due_to_record_failure": missing,
        "missing_record_ids": [
            str(row.get("record_id") or "")
            for row, condition in zip(rows, conditions)
            if not condition
        ],
        "unsuccessful_records": len(rows) - len(completed) - not_applicable,
        "not_applicable_records": not_applicable,
        "execution_labels": execution_labels,
        "target_probes_scored": len(completed),
        "retained_probes_scored": len(completed),
        "target_sequence_tokens_scored": sum(
            int(
                condition["deleted_target_quality"]["probes"][0]["score"][
                    "target_token_count"
                ]
            )
            for condition in completed
        ),
        "retained_sequence_tokens_scored": sum(
            int(condition["retained_quality"]["score"]["target_token_count"])
            for condition in completed
        ),
    }
    if completed:
        result.update(longitudinal._method_summary(completed))
    return result


def _certificate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    certificates = [row.get("solver_certificate") or {} for row in rows]
    completed = [
        certificate
        for certificate in certificates
        if certificate.get("status") == "completed"
    ]
    failed = [
        (row, certificate)
        for row, certificate in zip(rows, certificates)
        if certificate.get("status") == "failed"
    ]
    not_applicable = [
        (row, certificate)
        for row, certificate in zip(rows, certificates)
        if certificate.get("status") == "not_applicable"
    ]
    missing = [
        row
        for row, certificate in zip(rows, certificates)
        if not certificate
    ]
    accounted = len(completed) + len(failed) + len(not_applicable) + len(missing)
    result: dict[str, Any] = {
        "authorized_records": len(rows),
        "accounted_records": accounted,
        "unaccounted_records": len(rows) - accounted,
        "fully_accounted": accounted == len(rows),
        "completed_records": len(completed),
        "failed_records": len(failed),
        "failure_record_ids": [
            str(row.get("record_id") or "") for row, _ in failed
        ],
        "not_applicable_records": len(not_applicable),
        "not_applicable_record_ids": [
            str(row.get("record_id") or "") for row, _ in not_applicable
        ],
        "missing_due_to_record_failure": len(missing),
        "missing_record_ids": [
            str(row.get("record_id") or "") for row in missing
        ],
        "probe_output_kls": sum(
            len(certificate.get("probe_output_kls") or ())
            for certificate in completed
        ),
    }
    if completed:
        result["mean_exact_refit_output_kl_nats"] = _mean(
            [certificate["mean_output_kl_nats"] for certificate in completed]
        )
        result["max_exact_refit_output_kl_nats"] = max(
            float(certificate["max_output_kl_nats"])
            for certificate in completed
        )
    return result


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    method_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate all16 and the lock's predeclared joint12 efficacy subset."""

    authorization = method_lock.get("authorization") or {}
    admission_records = list(authorization.get("admission_records") or ())
    authorized_ids = [str(row.get("record_id") or "") for row in admission_records]
    observed_ids = [str(row.get("record_id") or "") for row in records]
    if observed_ids != authorized_ids:
        raise ValueError("method records differ from the authorized no-replacement set")
    prompt_enabled = bool(
        (authorization.get("prompt_suppression") or {}).get("enabled")
    )
    condition_ids = expected_condition_ids(
        prompt_suppression=prompt_enabled
    )
    if list(authorization.get("condition_ids") or ()) != list(condition_ids):
        raise ValueError("method condition IDs differ from the lock")
    joint_ids = set(
        (authorization.get("efficacy_population") or {}).get("record_ids") or ()
    )
    joint_rows = [row for row in records if row.get("record_id") in joint_ids]
    if len(joint_rows) != EXPECTED_JOINT_ADMITTED:
        raise ValueError("predeclared joint efficacy subset must contain 12 records")
    completed_rows = [
        row for row in records if row.get("status") == "completed"
    ]
    generic_failed_rows = [
        row for row in records if row.get("status") == "failed"
    ]
    method_failure_rows = [
        row
        for row in records
        if row.get("status") == "completed_with_failures"
    ]
    immutable = sum(
        bool((row.get("source_state_immutability") or {}).get("unchanged"))
        for row in records
    )
    immutability_unverified = sum(
        not isinstance(row.get("source_state_immutability"), Mapping)
        for row in records
    )
    summary: dict[str, Any] = {
        "aggregation_population": (
            "all 16 frozen records without replacement; efficacy uses only "
            "the confirmation-lock joint12 subset selected before method scoring"
        ),
        "denominators": {
            "frozen_records": EXPECTED_FROZEN_RECORDS,
            "authorized_records": len(authorized_ids),
            "attempted_records": len(records),
            "fully_completed_records": len(completed_rows),
            "record_failures": len(records) - len(completed_rows),
            "generic_record_failures": len(generic_failed_rows),
            "records_with_condition_failures": len(method_failure_rows),
            "predeclared_target_admitted": EXPECTED_TARGET_ADMITTED,
            "predeclared_retained_available": EXPECTED_RETAINED_AVAILABLE,
            "predeclared_joint_admitted": EXPECTED_JOINT_ADMITTED,
            "source_fixed_c_feasible_records": (
                EXPECTED_FROZEN_RECORDS - EXPECTED_SOURCE_INFEASIBLE
            ),
            "source_fixed_c_infeasible_records": EXPECTED_SOURCE_INFEASIBLE,
            "exact_decrement_full_repack_fallback_records": sum(
                bool(
                    (
                        row.get("conditions", {})
                        .get(EXACT_DECREMENT_ID, {})
                        .get("semantics", {})
                    ).get("full_repack_fallback")
                )
                for row in records
            ),
            "exact_decrement_incremental_records": sum(
                (
                    row.get("conditions", {})
                    .get(EXACT_DECREMENT_ID, {})
                    .get("semantics", {})
                    .get("executed_method")
                )
                == "incremental_float64_fixed_c_decrement"
                for row in records
            ),
            "exact_decrement_refit_fallback_records": sum(
                (
                    row.get("conditions", {})
                    .get(EXACT_DECREMENT_ID, {})
                    .get("semantics", {})
                    .get("executed_method")
                )
                == "fixed_c_refit_fallback"
                for row in records
            ),
            "fp32_proxy_full_repack_fallback_records": sum(
                bool(
                    (
                        row.get("conditions", {})
                        .get(FP32_PROXY_ID, {})
                        .get("semantics", {})
                    ).get("full_repack_fallback")
                )
                for row in records
            ),
            "fp32_proxy_incremental_records": sum(
                (
                    row.get("conditions", {})
                    .get(FP32_PROXY_ID, {})
                    .get("semantics", {})
                    .get("executed_method")
                )
                == "incremental_fp32_masked_refit"
                for row in records
            ),
            "token_row_zero_cost_alias_records": sum(
                bool(
                    (
                        row.get("conditions", {})
                        .get(TOKEN_ROW_DIAGNOSTIC_ID, {})
                        .get("semantics", {})
                    ).get("zero_cost_alias")
                )
                for row in records
            ),
            "fixed_c_refit_not_applicable_records": sum(
                (
                    row.get("conditions", {})
                    .get(FIXED_C_REFIT_ID, {})
                    .get("status")
                )
                == "not_applicable"
                for row in records
            ),
            "source_state_immutable_records": immutable,
            "source_state_immutability_failures": len(records) - immutable,
            "source_state_immutability_unverified_records": (
                immutability_unverified
            ),
        },
        "efficacy_population": dict(
            authorization.get("efficacy_population") or {}
        ),
        "failure_record_ids": [
            str(row.get("record_id") or "")
            for row in records
            if row.get("status") != "completed"
        ],
        "generic_failure_record_ids": [
            str(row.get("record_id") or "") for row in generic_failed_rows
        ],
        "condition_failure_record_ids": [
            str(row.get("record_id") or "") for row in method_failure_rows
        ],
        "source_state_immutability_failure_record_ids": [
            str(row.get("record_id") or "")
            for row in records
            if (row.get("source_state_immutability") or {}).get("unchanged")
            is not True
        ],
        "conditions": {},
        "per_condition_failures": {},
    }
    summary["denominators"][
        "exact_decrement_execution_path_unavailable_records"
    ] = len(records) - sum(
        int(summary["denominators"][key])
        for key in (
            "exact_decrement_incremental_records",
            "exact_decrement_refit_fallback_records",
            "exact_decrement_full_repack_fallback_records",
        )
    )
    for condition_id in condition_ids:
        all_records_summary = _condition_summary(records, condition_id)
        summary["per_condition_failures"][condition_id] = {
            "explicit_condition_failures": all_records_summary["failed_records"],
            "missing_due_to_record_failure": all_records_summary[
                "missing_due_to_record_failure"
            ],
            "unsuccessful_records": all_records_summary[
                "unsuccessful_records"
            ],
        }
        summary["conditions"][condition_id] = {
            "all_frozen_records": all_records_summary,
            "predeclared_joint12_subset": {
                "selection": (
                    "confirmation joint target-plus-retained admission; "
                    "frozen before deletion-method scoring"
                ),
                **_condition_summary(joint_rows, condition_id),
            },
        }
        if condition_id == EXACT_DECREMENT_ID:
            execution_strata = {}
            for label in (
                "incremental_float64_fixed_c_decrement",
                "fixed_c_refit_fallback",
                "fresh_raw_omission_full_repack_fallback",
            ):
                stratum_rows = [
                    row
                    for row in records
                    if (
                        row.get("conditions", {})
                        .get(EXACT_DECREMENT_ID, {})
                        .get("semantics", {})
                        .get("executed_method")
                    )
                    == label
                ]
                execution_strata[label] = {
                    "selection": "actual execution path; mutually exclusive",
                    "all_authorized_records": len(records),
                    "records_in_stratum": len(stratum_rows),
                    "records_outside_stratum": len(records)
                    - len(stratum_rows),
                    **_condition_summary(
                        stratum_rows,
                        EXACT_DECREMENT_ID,
                    ),
                }
            summary["conditions"][condition_id][
                "actual_execution_strata"
            ] = execution_strata
    summary["exact_refit_certificate"] = {
        "all_frozen_records": _certificate_summary(records),
        "predeclared_joint12_subset": _certificate_summary(joint_rows),
    }
    return summary


def _implementation_fingerprints() -> dict[str, str]:
    fingerprints = {
        label: _sha256_file(path)
        for label, path in _IMPLEMENTATION_PATHS.items()
    }
    fingerprints["contract_sha256"] = _payload_sha256(fingerprints)
    return fingerprints


def _environment() -> dict[str, Any]:
    versions = {}
    for module_name in ("numpy", "scipy", "torch", "transformers", "mlx"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": versions,
    }


def _base_report(
    *,
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    method_lock_file_sha256: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    authorization = method_lock["authorization"]
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "evaluation": "LongMemEval V1 post-admission chat deletion methods",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "model_scoring_used_for_selection_or_replacement": False,
        "official_longmemeval_leaderboard_score": False,
        "artifacts": {
            "manifest_file_sha256": CONFIRMATION_MANIFEST_FILE_SHA256,
            "manifest_integrity_sha256": manifest["integrity"]["sha256"],
            "policy_lock_file_sha256": POLICY_LOCK_FILE_SHA256,
            "policy_lock_sha256": policy_lock["lock_sha256"],
            "promotion_lock_file_sha256": PROMOTION_LOCK_FILE_SHA256,
            "promotion_lock_integrity_sha256": promotion_lock["integrity"][
                "sha256"
            ],
            "confirmation_report_file_sha256": CONFIRMATION_REPORT_SHA256,
            "method_lock_file_sha256": method_lock_file_sha256,
            "method_lock_integrity_sha256": method_lock["integrity"]["sha256"],
        },
        "config": {
            **dict(authorization["runtime"]),
            "evaluation_seed": EVALUATION_SEED,
            "warmup": warmup,
            "repeats": repeats,
            "condition_ids": list(authorization["condition_ids"]),
            "prompt_suppression": bool(
                authorization["prompt_suppression"]["enabled"]
            ),
        },
        "method_protocol": {
            "all_records_without_replacement": True,
            "exact_source_infeasible_fallback": (
                "explicit fresh raw-omission full repack"
            ),
            "fp32_proxy_source_infeasible_fallback": (
                "required fresh raw-omission full repack"
            ),
            "fixed_c_refit_source_infeasible": "not_applicable",
            "token_row_repack": (
                "zero-cost diagnostic alias of fresh raw omission"
            ),
            "resume_policy": (
                "validate existing report, discard every row, and re-execute "
                "all 16 authorized records"
            ),
            "completed_resume_rows_reused": False,
            "efficacy_population": "predeclared_confirmation_joint12",
            "full_target_sequences_scored": True,
            "first_token_rank_and_full_vocabulary_kl_scored": True,
            "source_state_immutability_required": True,
        },
        "implementation": _implementation_fingerprints(),
        "environment": _environment(),
        "records": [],
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    admission._assert_source_free_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resume_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": report.get("schema"),
        "schema_version": report.get("schema_version"),
        "evaluation": report.get("evaluation"),
        "contains_source_text": report.get("contains_source_text"),
        "contains_full_vocabulary_vectors": report.get(
            "contains_full_vocabulary_vectors"
        ),
        "artifacts": report.get("artifacts"),
        "config": report.get("config"),
        "method_protocol": report.get("method_protocol"),
        "implementation": report.get("implementation"),
        "environment": report.get("environment"),
    }


def _assert_finite_json(value: Any, *, path: str = "row") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"resume {path} contains a non-finite metric")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite_json(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"resume {path} is not source-free JSON data")


def _seal_completed_row(
    row: Mapping[str, Any],
    *,
    execution_contract_sha256: str,
) -> dict[str, Any]:
    """Add an audit checksum; this never authorizes resume-row reuse."""

    sealed = dict(row)
    sealed["execution_contract_sha256"] = execution_contract_sha256
    sealed["row_integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(sealed),
    }
    return sealed


def _validate_resume_audit_row(
    row: Mapping[str, Any],
    *,
    expected_id: str,
    binding: Mapping[str, Any],
    condition_ids: Sequence[str],
    execution_contract_sha256: str,
) -> None:
    admission._assert_source_free_payload(row)
    _assert_finite_json(row)
    if row.get("record_id") != expected_id or row.get("status") != "completed":
        raise ValueError("resume audit row identity or status differs")
    integrity = row.get("row_integrity") or {}
    unsigned = dict(row)
    unsigned.pop("row_integrity", None)
    if (
        row.get("execution_contract_sha256") != execution_contract_sha256
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise ValueError("resume audit row integrity or contract differs")
    if row.get("admission_binding") != binding:
        raise ValueError("resume audit row admission binding differs")
    feasible = bool(binding["source_fixed_c_feasible"])
    if (
        row.get("predeclared_joint_admitted")
        is not bool(binding["joint_admitted"])
        or row.get("source_fixed_c_feasible") is not feasible
        or row.get("method_failures") != []
    ):
        raise ValueError("resume audit row authorization metadata differs")
    probes = row.get("probes")
    if (
        not isinstance(probes, list)
        or [probe.get("probe_id") for probe in probes]
        != ["target_current", "retained"]
        or any(int(probe.get("target_token_count", 0)) < 1 for probe in probes)
    ):
        raise ValueError("resume audit row probe contract differs")
    immutability = row.get("source_state_immutability") or {}
    if (
        immutability.get("unchanged") is not True
        or immutability.get("shape_signature_unchanged") is not True
        or immutability.get("before_sha256")
        != immutability.get("after_sha256")
    ):
        raise ValueError("resume audit row lacks verified immutability")
    conditions = row.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(
        condition_ids
    ):
        raise ValueError("resume audit row method matrix is incomplete")
    for condition_id in condition_ids:
        condition = conditions[condition_id]
        expected_status = (
            "not_applicable"
            if condition_id == FIXED_C_REFIT_ID and not feasible
            else "completed"
        )
        if condition.get("status") != expected_status:
            raise ValueError("resume audit condition status differs")
        if expected_status == "not_applicable":
            if (
                condition.get("reason") != "source_fixed_c_infeasible"
                or condition.get("source_predeclared") is not True
            ):
                raise ValueError("resume audit fixed-C not-applicable row differs")
            continue
        if (
            condition.get("condition_id") != condition_id
            or not isinstance(condition.get("timing"), Mapping)
            or not isinstance(condition.get("storage"), Mapping)
            or len(
                (condition.get("deleted_target_quality") or {}).get(
                    "probes", ()
                )
            )
            != 1
            or not isinstance(condition.get("retained_quality"), Mapping)
        ):
            raise ValueError("resume audit condition metrics are incomplete")
    token_alias = conditions[TOKEN_ROW_DIAGNOSTIC_ID]
    if (
        (token_alias.get("semantics") or {}).get("zero_cost_alias") is not True
        or (token_alias.get("semantics") or {}).get("alias_of")
        != FRESH_RAW_OMISSION_ID
        or (token_alias.get("timing") or {}).get("zero_cost_alias") is not True
    ):
        raise ValueError("resume audit token-row diagnostic differs")
    exact = conditions[EXACT_DECREMENT_ID].get("semantics") or {}
    proxy = conditions[FP32_PROXY_ID].get("semantics") or {}
    if bool(exact.get("full_repack_fallback")) != (not feasible):
        raise ValueError("resume audit exact-decrement fallback label differs")
    expected_exact_labels = (
        {
            "incremental_float64_fixed_c_decrement",
            "fixed_c_refit_fallback",
        }
        if feasible
        else {"fresh_raw_omission_full_repack_fallback"}
    )
    if exact.get("executed_method") not in expected_exact_labels:
        raise ValueError("resume exact-decrement execution label differs")
    if bool(proxy.get("full_repack_fallback")) != (not feasible):
        raise ValueError("resume audit FP32 proxy fallback label differs")
    certificate = row.get("solver_certificate") or {}
    expected_certificate_status = "completed" if feasible else "not_applicable"
    if certificate.get("status") != expected_certificate_status:
        raise ValueError("resume audit exact/refit certificate status differs")


def _failed_record(record_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message_sha256": benchmark.base.text_sha256(str(exc)),
        "error_message_redacted": True,
    }


def run_records(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    promotion_lock: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    records: Sequence[benchmark.RehydratedChatRecord],
    runtime: Any,
    *,
    confirmation_report: Mapping[str, Any],
    method_lock_file_sha256: str,
    output: Path,
    warmup: int,
    repeats: int,
    explicit_acknowledgement: bool,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run all16; resume audits old output but never reuses method rows."""

    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    arm, device = authorize_method_execution(
        method_lock,
        manifest=manifest,
        policy_lock=policy_lock,
        promotion_lock=promotion_lock,
        confirmation_report=confirmation_report,
        explicit_acknowledgement=explicit_acknowledgement,
    )
    admission.verify_loaded_runtime(runtime, arm=arm, device=device)
    authorization = method_lock["authorization"]
    bindings = {
        str(row["record_id"]): row
        for row in authorization["admission_records"]
    }
    expected_ids = list(bindings)
    observed_ids = [record.record_id for record in records]
    if (
        len(records) != EXPECTED_FROZEN_RECORDS
        or observed_ids != expected_ids
        or len(set(observed_ids)) != EXPECTED_FROZEN_RECORDS
    ):
        raise ValueError("method execution requires all 16 records in order")
    base_report = _base_report(
        manifest=manifest,
        policy_lock=policy_lock,
        promotion_lock=promotion_lock,
        method_lock=method_lock,
        method_lock_file_sha256=method_lock_file_sha256,
        warmup=warmup,
        repeats=repeats,
    )
    execution_contract_sha256 = _payload_sha256(
        _resume_signature(base_report)
    )
    if output.exists() and not (resume or overwrite):
        raise FileExistsError("output exists; pass resume or overwrite")
    discarded_resume_rows = 0
    discarded_completed_resume_records = 0
    if resume and output.exists():
        existing = _load_mapping(output, name="resume report")
        admission._assert_source_free_payload(existing)
        _assert_finite_json(existing, path="report")
        if _resume_signature(existing) != _resume_signature(base_report):
            raise ValueError("resume report contract differs")
        existing_rows = existing.get("records")
        if not isinstance(existing_rows, list):
            raise ValueError("resume report records must be a list")
        discarded_resume_rows = len(existing_rows)
        existing_ids = [
            str(row.get("record_id") or "")
            for row in existing_rows
            if isinstance(row, Mapping)
        ]
        if (
            len(existing_ids) != len(existing_rows)
            or len(existing_rows) > len(expected_ids)
            or existing_ids != expected_ids[: len(existing_ids)]
        ):
            raise ValueError("resume report record ID order differs")
        condition_ids = expected_condition_ids(
            prompt_suppression=bool(
                authorization["prompt_suppression"]["enabled"]
            )
        )
        for row in existing_rows:
            if row.get("status") == "completed":
                record_id = str(row.get("record_id") or "")
                _validate_resume_audit_row(
                    row,
                    expected_id=record_id,
                    binding=bindings[record_id],
                    condition_ids=condition_ids,
                    execution_contract_sha256=execution_contract_sha256,
                )
                discarded_completed_resume_records += 1

    started = time.perf_counter()
    report = base_report
    prompt_suppression = bool(
        authorization["prompt_suppression"]["enabled"]
    )
    feasibility = {
        record_id: bool(binding["source_fixed_c_feasible"])
        for record_id, binding in bindings.items()
    }
    rows: list[Mapping[str, Any]] = []
    for record in records:
        try:
            row = evaluate_record(
                runtime,
                record,
                admission_binding=bindings[record.record_id],
                source_fixed_c_feasible=feasibility[record.record_id],
                prompt_suppression=prompt_suppression,
                warmup=warmup,
                repeats=repeats,
            )
        except Exception as exc:
            row = _failed_record(record.record_id, exc)
        if row.get("status") == "completed":
            row = _seal_completed_row(
                row,
                execution_contract_sha256=execution_contract_sha256,
            )
        rows.append(row)
        report["records"] = rows
        report["status"] = "running"
        report["elapsed_seconds_this_process"] = time.perf_counter() - started
        report["resume_requested"] = bool(resume)
        report["resume_policy"] = "validate_then_discard_all_rows"
        report["reused_completed_resume_records"] = 0
        report["discarded_completed_resume_records"] = (
            discarded_completed_resume_records
        )
        report["discarded_resume_rows"] = discarded_resume_rows
        _atomic_write(output, report)
    report["summary"] = summarize_records(rows, method_lock=method_lock)
    report["status"] = (
        "completed"
        if all(row.get("status") == "completed" for row in rows)
        else "completed_with_failures"
    )
    report["elapsed_seconds_this_process"] = time.perf_counter() - started
    _atomic_write(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--policy-lock", default=str(DEFAULT_POLICY_LOCK))
    parser.add_argument("--promotion-lock", default=str(DEFAULT_PROMOTION_LOCK))
    parser.add_argument(
        "--confirmation-report",
        default=str(DEFAULT_CONFIRMATION_REPORT),
    )
    parser.add_argument("--method-lock", default=str(DEFAULT_METHOD_LOCK))
    parser.add_argument("--data-path")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--allow-method-scoring",
        action="store_true",
        help="required in addition to the committed method authorization lock",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "validate prior output as an audit artifact, then discard every "
            "row and re-execute all 16 records; completed rows are never reused"
        ),
    )
    output_mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats positive")
    try:
        manifest, policy_lock, promotion_lock, confirmation_report = (
            load_locked_artifacts(
                args.manifest,
                args.policy_lock,
                args.promotion_lock,
                args.confirmation_report,
            )
        )
        method_lock = load_committed_method_lock(
            args.method_lock,
            manifest=manifest,
            policy_lock=policy_lock,
            promotion_lock=promotion_lock,
            confirmation_report=confirmation_report,
            require_canonical_path=True,
        )
        arm, device = authorize_method_execution(
            method_lock,
            manifest=manifest,
            policy_lock=policy_lock,
            promotion_lock=promotion_lock,
            confirmation_report=confirmation_report,
            explicit_acknowledgement=args.allow_method_scoring,
        )
    except (OSError, PermissionError, ValueError, benchmark.ManifestError) as exc:
        parser.error(str(exc))
    output = Path(args.out)
    if output.exists() and not args.resume and not args.overwrite:
        parser.error(f"{output} exists; pass --resume or --overwrite")
    rows = admission._load_pinned_rows(args.data_path)
    runtime = admission._make_runtime(arm, device=device)
    runtime.ensure_loaded()
    admission.verify_loaded_runtime(runtime, arm=arm, device=device)
    records = benchmark.rehydrate_manifest(
        manifest,
        rows,
        runtime.tokenizer,
    )
    run_records(
        manifest,
        policy_lock,
        promotion_lock,
        method_lock,
        records,
        runtime,
        confirmation_report=confirmation_report,
        method_lock_file_sha256=_sha256_file(args.method_lock),
        output=output,
        warmup=args.warmup,
        repeats=args.repeats,
        explicit_acknowledgement=args.allow_method_scoring,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

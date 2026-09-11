"""Publish the validated, source-free LongMemEval chatbot result and macros.

The local all-record report is intentionally too detailed for release.  This
publisher revalidates it against the committed cohort, method, interruption,
and finalization locks; reconciles the incompatible cache-surgery diagnostic;
recomputes manuscript metrics; and emits one compact JSON artifact plus a
deterministic LaTeX macro file.  It performs no model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_geometry_methods as methods
from gemma_sv import finalize_longmemeval_chat_geometry_methods as finalizer


SCHEMA = "gemma-sv-longmemeval-chat-geometry-methods-compact16-v1"
SCHEMA_VERSION = 1
VALIDATION_STATUS = "validated-complete-all16"
BEHAVIORAL_RESULT_STATUS = "validated-positive"
CERTIFICATE_STATUS = "validated-pass"

EXPECTED_FINAL_REPORT_FILE_SHA256 = (
    "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6"
)
EXPECTED_FINAL_REPORT_PAYLOAD_SHA256 = (
    "d6c4348975a912652fb1f2e153d8e314f3455ca4efcf6e9e565451b703804dac"
)
EXPECTED_FINAL_ROW_PAYLOAD_SHA256 = (
    "dc63d4c050a73940ba393c33ae954a62d25da3374df862e80f6724d3ab7e29e5"
)
EXPECTED_CACHE_FAILURE_MESSAGE_SHA256 = (
    "84dd133fa38ccc3e2261ab59c190cf2135294a01eb6649b54609f00b52f7113c"
)
EXPECTED_PARTIAL_LOCK_INTEGRITY_SHA256 = (
    "f3f332357ceda0796f54fb297d2aa423352989a9eaaefb7c54c6dafb5bd84c4f"
)
CERTIFICATE_PUBLICATION_MAX_KL_NATS = 1e-12

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
DEFAULT_REPORT = finalizer.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    PACKAGE
    / "benchmarks"
    / "longmemeval_chat_geometry_methods_compact16_v1.json"
)
DEFAULT_MACROS_OUTPUT = (
    PACKAGE / "paper" / "longmemeval_chatbot_compact16_macros.tex"
)

PRIMARY_CONDITION_IDS = (
    methods.PRESENT_ID,
    methods.FRESH_RAW_OMISSION_ID,
    methods.TOKEN_ROW_DIAGNOSTIC_ID,
    methods.EXACT_DECREMENT_ID,
    methods.FIXED_C_REFIT_ID,
    methods.FP32_PROXY_ID,
    methods.DECAY_ID,
    methods.PROMPT_SUPPRESSION_ID,
)
MANUSCRIPT_CONDITION_IDS = (
    methods.PRESENT_ID,
    methods.FRESH_RAW_OMISSION_ID,
    methods.EXACT_DECREMENT_ID,
    methods.FIXED_C_REFIT_ID,
    methods.FP32_PROXY_ID,
    methods.DECAY_ID,
    methods.PROMPT_SUPPRESSION_ID,
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "assistant_content",
        "conversation",
        "dialogue",
        "full_vocabulary_vector",
        "full_vocabulary_vectors",
        "generated_text",
        "logits",
        "messages",
        "model_output_text",
        "prompt",
        "question",
        "raw_context",
        "raw_context_text",
        "source_text",
        "user_content",
    }
)


class PublicationError(ValueError):
    """The final report or compact publication violated its contract."""


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


def deterministic_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise PublicationError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _load_mapping_and_sha256(
    path: str | Path,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    raw = Path(path).expanduser().read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PublicationError(f"{name} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise PublicationError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PublicationError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise PublicationError(f"{name} must be finite")
    return number


def _mean(values: Iterable[Any], *, name: str) -> float:
    numbers = [_finite(value, name=name) for value in values]
    if not numbers:
        raise PublicationError(f"{name} cannot be empty")
    return float(statistics.fmean(numbers))


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key).casefold()
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _target_probe(
    row: Mapping[str, Any],
    condition_id: str,
) -> Mapping[str, Any]:
    condition = (row.get("conditions") or {}).get(condition_id) or {}
    probes = (condition.get("deleted_target_quality") or {}).get("probes")
    if not isinstance(probes, list):
        raise PublicationError(f"{condition_id} target probes are unavailable")
    matches = [
        probe
        for probe in probes
        if isinstance(probe, Mapping) and probe.get("probe_id") == "target_current"
    ]
    if len(matches) != 1:
        raise PublicationError(
            f"{condition_id} must contain one target_current probe"
        )
    return matches[0]


def _condition_metrics(
    rows: Sequence[Mapping[str, Any]],
    condition_id: str,
) -> dict[str, Any]:
    conditions = [row["conditions"][condition_id] for row in rows]
    if not all(condition.get("status") == "completed" for condition in conditions):
        raise PublicationError(f"{condition_id} is incomplete on the efficacy set")
    targets = [_target_probe(row, condition_id) for row in rows]
    retained = [condition["retained_quality"] for condition in conditions]
    suppressions = [
        _finite(
            probe["suppression_vs_present_nats"],
            name=f"{condition_id} target suppression",
        )
        for probe in targets
    ]
    target_raw_drifts = [
        _finite(
            probe["mean_log_probability_drift_from_raw_repack_nats"],
            name=f"{condition_id} target raw-omission drift",
        )
        for probe in targets
    ]
    retained_raw_drifts = [
        _finite(
            probe["mean_log_probability_drift_from_raw_repack_nats"],
            name=f"{condition_id} retained raw-omission drift",
        )
        for probe in retained
    ]
    return {
        "completed_records": len(conditions),
        "target_probes": len(targets),
        "retained_probes": len(retained),
        "mean_target_suppression_vs_present_nats": _mean(
            suppressions,
            name=f"{condition_id} mean target suppression",
        ),
        "minimum_target_suppression_vs_present_nats": min(suppressions),
        "maximum_target_suppression_vs_present_nats": max(suppressions),
        "mean_target_log_probability_drift_from_raw_omission_nats": _mean(
            target_raw_drifts,
            name=f"{condition_id} mean target raw-omission drift",
        ),
        "records_at_or_below_raw_omission_floor": sum(
            value <= 0.0 for value in target_raw_drifts
        ),
        "mean_absolute_target_log_probability_drift_from_raw_omission_nats": (
            _mean(
                (abs(value) for value in target_raw_drifts),
                name=f"{condition_id} mean absolute target raw-omission drift",
            )
        ),
        "mean_target_first_token_kl_raw_omission_to_method_nats": _mean(
            (
                probe["full_vocabulary_kl_raw_repack_to_method_nats"]
                for probe in targets
            ),
            name=f"{condition_id} target raw-omission KL",
        ),
        "mean_retained_log_probability_drift_from_raw_omission_nats": _mean(
            retained_raw_drifts,
            name=f"{condition_id} mean retained raw-omission drift",
        ),
        "mean_absolute_retained_log_probability_drift_from_raw_omission_nats": (
            _mean(
                (abs(value) for value in retained_raw_drifts),
                name=f"{condition_id} mean absolute retained drift",
            )
        ),
        "mean_retained_first_token_kl_raw_omission_to_method_nats": _mean(
            (
                probe["full_vocabulary_kl_raw_repack_to_method_nats"]
                for probe in retained
            ),
            name=f"{condition_id} retained raw-omission KL",
        ),
        "mean_retained_first_target_token_rank": _mean(
            (probe["score"]["first_target_token_rank"] for probe in retained),
            name=f"{condition_id} retained rank",
        ),
    }


def _load_and_validate_final_report(
    report_path: str | Path,
) -> dict[str, Any]:
    (
        manifest,
        policy,
        census,
        admission_report,
    ) = methods.load_locked_artifacts()
    method_lock, method_lock_file_sha256 = _load_mapping_and_sha256(
        methods.DEFAULT_METHOD_LOCK,
        name="geometry method authorization lock",
    )
    method_analysis = finalizer._method_contract_analysis(method_lock)
    expected_artifacts = {
        "admission_report_file_sha256": methods.ADMISSION_REPORT_FILE_SHA256,
        "admission_report_payload_sha256": (
            methods.ADMISSION_REPORT_PAYLOAD_SHA256
        ),
        "census_file_sha256": file_sha256(methods.DEFAULT_CENSUS),
        "census_integrity_sha256": census["integrity"]["sha256"],
        "core_manifest_file_sha256": admission_report["core_cohort"][
            "source_manifest_file_sha256"
        ],
        "manifest_file_sha256": file_sha256(methods.DEFAULT_MANIFEST),
        "manifest_integrity_sha256": manifest["integrity"]["sha256"],
        "policy_file_sha256": file_sha256(methods.DEFAULT_POLICY_LOCK),
        "policy_lock_sha256": policy["lock_sha256"],
    }
    if method_analysis["artifacts"] != expected_artifacts:
        raise PublicationError("method lock artifact bindings differ")
    partial_report, partial_report_file_sha256 = _load_mapping_and_sha256(
        finalizer.DEFAULT_PARTIAL_REPORT,
        name="interrupted partial method report",
    )
    if (
        partial_report_file_sha256 != finalizer.PARTIAL_REPORT_FILE_SHA256
        or payload_sha256(partial_report)
        != finalizer.PARTIAL_REPORT_PAYLOAD_SHA256
        or partial_report.get("schema") != methods.SCHEMA
        or partial_report.get("status") != "running"
        or partial_report.get("contains_source_text") is not False
        or partial_report.get("contains_full_vocabulary_vectors") is not False
        or partial_report.get("method_lock_integrity_sha256")
        != method_lock["integrity"]["sha256"]
    ):
        raise PublicationError("interrupted partial report binding differs")
    partial_lock, partial_lock_file_sha256 = _load_mapping_and_sha256(
        finalizer.DEFAULT_PARTIAL_LOCK,
        name="geometry methods partial lock",
    )
    partial_lock_integrity = partial_lock.get("integrity")
    unsigned_partial_lock = dict(partial_lock)
    unsigned_partial_lock.pop("integrity", None)
    method_contract = partial_lock.get("method_contract") or {}
    completed_rows = partial_lock.get("completed_rows")
    missing_record = partial_lock.get("missing_record") or {}
    if (
        partial_lock.get("schema") != finalizer.PARTIAL_LOCK_SCHEMA
        or partial_lock.get("schema_version")
        != finalizer.PARTIAL_LOCK_SCHEMA_VERSION
        or partial_lock.get("status") != finalizer.PARTIAL_LOCK_STATUS
        or partial_lock.get("contains_source_text") is not False
        or partial_lock.get("contains_full_vocabulary_vectors") is not False
        or not isinstance(partial_lock_integrity, Mapping)
        or partial_lock_integrity.get("algorithm") != "sha256"
        or partial_lock_integrity.get("sha256")
        != EXPECTED_PARTIAL_LOCK_INTEGRITY_SHA256
        or partial_lock_integrity.get("sha256")
        != payload_sha256(unsigned_partial_lock)
        or (partial_lock.get("partial_report") or {}).get("file_sha256")
        != partial_report_file_sha256
        or (partial_lock.get("partial_report") or {}).get("payload_sha256")
        != finalizer.PARTIAL_REPORT_PAYLOAD_SHA256
        or method_contract.get("authorization")
        != method_analysis["authorization"]
        or method_contract.get("artifacts") != method_analysis["artifacts"]
        or method_contract.get("implementation")
        != method_analysis["implementation"]
        or method_contract.get("method_lock_integrity_sha256")
        != method_lock["integrity"]["sha256"]
        or not isinstance(completed_rows, list)
        or len(completed_rows) != 15
        or missing_record.get("authorized_position") != 16
        or missing_record.get("record_id")
        != method_analysis["authorization"]["admission_records"][-1][
            "record_id"
        ]
        or missing_record.get("partial_status") != "not_attempted"
    ):
        raise PublicationError("committed partial finalization lock differs")
    partial_records = partial_report.get("records")
    if (
        not isinstance(partial_records, list)
        or len(partial_records) != 16
        or [row.get("record_id") for row in partial_records]
        != [
            row["record_id"]
            for row in method_analysis["authorization"]["admission_records"]
        ]
        or any(row.get("status") == "not_attempted" for row in partial_records[:15])
        or partial_records[-1].get("status") != "not_attempted"
        or [
            finalizer._payload_sha256(row) for row in partial_records[:15]
        ]
        != [row.get("payload_sha256") for row in completed_rows]
    ):
        raise PublicationError("partial report rows differ from committed seals")
    report, report_file_sha256 = _load_mapping_and_sha256(
        report_path,
        name="final LongMemEval method report",
    )
    if (
        report_file_sha256 != EXPECTED_FINAL_REPORT_FILE_SHA256
        or payload_sha256(report) != EXPECTED_FINAL_REPORT_PAYLOAD_SHA256
    ):
        raise PublicationError("final report is not the exact validated all16 file")
    if (
        report.get("schema") != methods.SCHEMA
        or report.get("schema_version") != methods.SCHEMA_VERSION
        or report.get("status") != "completed_with_record_failures"
        or report.get("contains_source_text") is not False
        or report.get("contains_full_vocabulary_vectors") is not False
        or report.get("official_longmemeval_leaderboard_score") is not False
        or report.get("method_lock_integrity_sha256")
        != method_lock["integrity"]["sha256"]
    ):
        raise PublicationError("final report top-level contract differs")

    records = report.get("records")
    bindings = method_lock["authorization"]["admission_records"]
    public_records = manifest["records"]
    if (
        not isinstance(records, list)
        or len(records) != methods.EXPECTED_RECORDS
        or len(bindings) != methods.EXPECTED_RECORDS
        or len(public_records) != methods.EXPECTED_RECORDS
    ):
        raise PublicationError("final report does not contain the frozen all16")
    for row, binding, public_record in zip(records, bindings, public_records):
        finalizer.validate_hardened_method_row(
            row,
            expected_id=binding["record_id"],
            binding=binding,
            condition_ids=methods.CONDITION_IDS,
            public_record=public_record,
        )
    finalizer._validate_final_accounting(report, method_lock=method_lock)
    methods._assert_source_free(report)
    methods.hardened._assert_finite_json(report, path="final_report")
    if report["summary"] != methods.summarize_records(
        records,
        method_lock=method_lock,
    ):
        raise PublicationError("final report summary does not recompute")

    expected_report = finalizer._build_final_report(
        partial_report,
        partial_lock=partial_lock,
        method_lock=method_lock,
        final_row=records[-1],
        elapsed_seconds=report["elapsed_seconds_finalization_process"],
    )
    if report != expected_report:
        raise PublicationError("final report does not reconstruct from locked inputs")
    sealed_prefix = partial_lock["completed_rows"]
    if [
        finalizer._payload_sha256(row) for row in records[:15]
    ] != [row["payload_sha256"] for row in sealed_prefix]:
        raise PublicationError("final report changed a sealed partial-prefix row")
    if finalizer._payload_sha256(records[-1]) != (
        EXPECTED_FINAL_ROW_PAYLOAD_SHA256
    ):
        raise PublicationError("final report record16 seal differs")

    for row in records:
        if row.get("status") != "completed_with_failures":
            raise PublicationError("final row status differs from reconciliation")
        if row.get("method_failures") != [methods.CACHE_DELETE_SHIFT_ID]:
            raise PublicationError("final row contains an unreconciled method failure")
        for condition_id in PRIMARY_CONDITION_IDS:
            if row["conditions"][condition_id].get("status") != "completed":
                raise PublicationError(
                    f"primary condition {condition_id} is incomplete"
                )
        cache = row["conditions"][methods.CACHE_DELETE_SHIFT_ID]
        if (
            cache
            != {
                "condition_id": methods.CACHE_DELETE_SHIFT_ID,
                "error_message_redacted": True,
                "error_message_sha256": (
                    EXPECTED_CACHE_FAILURE_MESSAGE_SHA256
                ),
                "error_type": "ValueError",
                "status": "failed",
            }
        ):
            raise PublicationError("cache diagnostic failure is not the known one")
        if row["solver_certificate"].get("status") != "completed":
            raise PublicationError("exact/refit certificate is incomplete")
        immutability = row["source_state_immutability"]
        if (
            immutability.get("unchanged") is not True
            or immutability.get("shape_signature_unchanged") is not True
        ):
            raise PublicationError("source-state immutability check failed")

    leaked = sorted(set(_walk_keys(report)) & _FORBIDDEN_PUBLIC_KEYS)
    if leaked:
        raise PublicationError(
            "final report contains forbidden source/vector keys: "
            + ", ".join(leaked)
        )
    return {
        "report": report,
        "report_file_sha256": report_file_sha256,
        "manifest": manifest,
        "policy": policy,
        "census": census,
        "admission_report": admission_report,
        "method_lock": method_lock,
        "method_lock_file_sha256": method_lock_file_sha256,
        "partial_report": partial_report,
        "partial_report_file_sha256": partial_report_file_sha256,
        "partial_lock": partial_lock,
        "partial_lock_file_sha256": partial_lock_file_sha256,
    }


def build_compact_result(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    report = validated["report"]
    records = report["records"]
    method_lock = validated["method_lock"]
    admission_report = validated["admission_report"]
    partial_lock = validated["partial_lock"]
    authorization = method_lock["authorization"]
    efficacy = authorization["efficacy_population"]
    joint_ids = list(efficacy["record_ids"])
    joint_rows = [
        row for row in records if row.get("record_id") in set(joint_ids)
    ]
    if (
        [row["record_id"] for row in joint_rows] != joint_ids
        or len(joint_rows) != methods.EXPECTED_JOINT_ADMITTED
    ):
        raise PublicationError("predeclared efficacy population differs")

    conditions = {
        condition_id: _condition_metrics(joint_rows, condition_id)
        for condition_id in MANUSCRIPT_CONDITION_IDS
    }
    exact = conditions[methods.EXACT_DECREMENT_ID]
    prompt = conditions[methods.PROMPT_SUPPRESSION_ID]
    if (
        exact["mean_target_suppression_vs_present_nats"] <= 0.0
        or exact[
            "mean_target_log_probability_drift_from_raw_omission_nats"
        ]
        > 0.0
        or exact["mean_retained_first_target_token_rank"] > 10.0
        or exact["mean_target_suppression_vs_present_nats"]
        <= prompt["mean_target_suppression_vs_present_nats"]
    ):
        raise PublicationError("behavioral result does not satisfy positive checks")

    certificate_values = [
        _finite(
            probe["full_vocabulary_output_kl_nats"],
            name="exact/refit certificate probe KL",
        )
        for row in records
        for probe in row["solver_certificate"]["probe_output_kls"]
    ]
    if (
        len(certificate_values) != 2 * methods.EXPECTED_RECORDS
        or min(certificate_values) < 0.0
        or max(certificate_values) > CERTIFICATE_PUBLICATION_MAX_KL_NATS
    ):
        raise PublicationError("exact/refit certificate publication check failed")
    certificate = {
        "status": CERTIFICATE_STATUS,
        "reference": "fixed-C retained-key refit",
        "direction": "KL(exact policy || fixed-C retained-key refit)",
        "distribution_scope": (
            "full output vocabulary at the first target token of each "
            "registered target and retained probe"
        ),
        "records": methods.EXPECTED_RECORDS,
        "probes": len(certificate_values),
        "mean_kl_nats": _mean(
            certificate_values,
            name="all16 exact/refit certificate mean KL",
        ),
        "maximum_kl_nats": max(certificate_values),
        "publisher_maximum_kl_tolerance_nats": (
            CERTIFICATE_PUBLICATION_MAX_KL_NATS
        ),
        "behavioral_raw_omission_is_distinct": True,
        "raw_history_equality_claimed": False,
    }

    denominators = report["summary"]["denominators"]
    primary_completion = {
        condition_id: sum(
            row["conditions"][condition_id]["status"] == "completed"
            for row in records
        )
        for condition_id in PRIMARY_CONDITION_IDS
    }
    if any(
        count != methods.EXPECTED_RECORDS
        for count in primary_completion.values()
    ):
        raise PublicationError("primary all16 condition matrix is incomplete")

    core = admission_report["core_cohort"]
    no_output_selection = bool(
        authorization["all_records_without_replacement"] is True
        and efficacy["selected_before_method_scoring"] is True
        and admission_report[
            "model_scoring_used_for_selection_or_replacement"
        ]
        is False
    )
    core_bindings_unchanged = bool(
        core["same_ordered_original_v1_core_records"] is True
        and core["record_question_target_retained_replacement"] is False
        and core["probe_replacement"] is False
        and core["source_question_answer_retained_or_probe_changes"] is False
    )
    if not no_output_selection or not core_bindings_unchanged:
        raise PublicationError("cohort selection or core bindings changed")

    compact: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": VALIDATION_STATUS,
        "behavioral_result_status": BEHAVIORAL_RESULT_STATUS,
        "certificate_status": CERTIFICATE_STATUS,
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "contains_model_generated_text": False,
        "official_longmemeval_leaderboard_score": False,
        "scope": {
            "evaluation": "constrained LongMemEval V1 deletion adaptation",
            "claim": (
                "behavioral target suppression with retained-memory "
                "preservation plus a separately scoped exact/refit certificate"
            ),
            "not_an_official_leaderboard_score": True,
            "no_incremental_speed_claim": True,
            "no_raw_history_state_equality_claim": True,
            "no_privacy_or_compliance_guarantee": True,
        },
        "validation": {
            "all16_attempted_and_terminal": True,
            "all_manuscript_conditions_complete": True,
            "only_excluded_failures_reconciled": True,
            "source_state_immutable_all16": True,
            "no_output_based_selection_or_replacement": no_output_selection,
            "geometry_correction_output_blind": (
                validated["policy"].get("selection_uses_model_outputs") is False
            ),
            "core_bindings_unchanged": core_bindings_unchanged,
            "final_report_reconstructed_from_locked_inputs": True,
            "sealed_partial_prefix_rows_verified": 15,
            "newly_scored_final_records": 1,
            "resume_output_rows_reused": 0,
        },
        "denominators": {
            "frozen_records": denominators["frozen_records"],
            "authorized_records": denominators["authorized_records"],
            "attempted_records": denominators["attempted_records"],
            "terminal_records": denominators["attempted_records"]
            - denominators["not_attempted_records"],
            "not_attempted_records": denominators["not_attempted_records"],
            "target_admitted_records": denominators[
                "predeclared_target_admitted"
            ],
            "retained_available_records": denominators[
                "predeclared_retained_available"
            ],
            "joint_admitted_records": denominators[
                "predeclared_joint_admitted"
            ],
            "source_state_immutable_records": denominators[
                "source_state_immutable_records"
            ],
            "source_state_immutability_failures": denominators[
                "source_state_immutability_failures"
            ],
            "certificate_records": methods.EXPECTED_RECORDS,
            "certificate_probes": len(certificate_values),
            "raw_report_record_failures": denominators["record_failures"],
            "primary_result_failures": 0,
            "excluded_diagnostic_failures": methods.EXPECTED_RECORDS,
            "primary_condition_completed_records": primary_completion,
        },
        "metric_definitions": {
            "target_suppression": (
                "present target-sequence mean log probability minus method "
                "target-sequence mean log probability; larger is more suppression"
            ),
            "behavioral_floor_position": (
                "method target-sequence mean log probability minus fresh "
                "raw-omission mean log probability; negative is below the floor"
            ),
            "behavioral_deletion_gap": (
                "KL(fresh raw omission || method) over the full vocabulary "
                "at the first target token"
            ),
            "retention_gap": (
                "KL(fresh raw omission || method) over the full vocabulary "
                "at the retained probe's first target token"
            ),
            "certificate_gap": (
                "KL(exact policy || fixed-C retained-key refit), not KL to "
                "fresh raw omission"
            ),
        },
        "efficacy": {
            "population": "predeclared_geometry_joint10",
            "records": methods.EXPECTED_JOINT_ADMITTED,
            "record_ids_sha256": efficacy["record_ids_sha256"],
            "selected_before_method_scoring": True,
            "conditions": conditions,
            "exact_policy_summary": {
                "mean_target_suppression_nats": exact[
                    "mean_target_suppression_vs_present_nats"
                ],
                "minimum_target_suppression_nats": exact[
                    "minimum_target_suppression_vs_present_nats"
                ],
                "maximum_target_suppression_nats": exact[
                    "maximum_target_suppression_vs_present_nats"
                ],
                "mean_nats_below_raw_omission_floor": -exact[
                    "mean_target_log_probability_drift_from_raw_omission_nats"
                ],
                "records_at_or_below_raw_omission_floor": exact[
                    "records_at_or_below_raw_omission_floor"
                ],
                "mean_absolute_retained_log_probability_drift_nats": exact[
                    "mean_absolute_retained_log_probability_drift_from_raw_omission_nats"
                ],
                "mean_target_first_token_kl_to_raw_omission_nats": exact[
                    "mean_target_first_token_kl_raw_omission_to_method_nats"
                ],
                "mean_retained_first_token_kl_to_raw_omission_nats": exact[
                    "mean_retained_first_token_kl_raw_omission_to_method_nats"
                ],
            },
        },
        "exact_policy": {
            "name": "verified fixed-C deletion/refit policy",
            "incremental_float64_decrement_records": denominators[
                "exact_decrement_incremental_records"
            ],
            "fixed_c_refit_fallback_records": denominators[
                "exact_decrement_refit_fallback_records"
            ],
            "source_full_repack_fallback_records": denominators[
                "exact_decrement_full_repack_fallback_records"
            ],
            "all_executions_used_fixed_c_refit_fallback": True,
            "incremental_conversation_deletion_speed_claim_supported": False,
        },
        "certificate": certificate,
        "diagnostics": {
            "cache_delete_shift": {
                "classification": "incompatible_diagnostic",
                "scientific_role": "excluded_from_primary_manuscript_conditions",
                "attempted_records": methods.EXPECTED_RECORDS,
                "failed_records": methods.EXPECTED_RECORDS,
                "error_type": "ValueError",
                "error_message_redacted": True,
                "error_message_sha256": EXPECTED_CACHE_FAILURE_MESSAGE_SHA256,
                "public_interpretation": (
                    "row deletion would collapse distinct support-vector gate "
                    "boundaries; this cache surgery is not the certified policy"
                ),
                "causes_raw_report_completed_with_record_failures_status": True,
                "implies_exact_policy_failure": False,
            }
        },
        "source_artifacts": {
            "final_report": {
                "canonical_repository_path": (
                    finalizer.FINAL_OUTPUT_REPOSITORY_PATH
                ),
                "file_sha256": validated["report_file_sha256"],
                "payload_sha256": payload_sha256(report),
                "schema": report["schema"],
                "raw_status": report["status"],
                "final_row_payload_sha256": (
                    EXPECTED_FINAL_ROW_PAYLOAD_SHA256
                ),
            },
            "partial_report": {
                "file_sha256": validated["partial_report_file_sha256"],
                "payload_sha256": finalizer.PARTIAL_REPORT_PAYLOAD_SHA256,
            },
            "partial_lock": {
                "file_sha256": validated["partial_lock_file_sha256"],
                "integrity_sha256": partial_lock["integrity"]["sha256"],
                "finalizer_file_sha256": partial_lock["finalizer"][
                    "file_sha256"
                ],
            },
            "method_lock": {
                "file_sha256": validated["method_lock_file_sha256"],
                "payload_sha256": payload_sha256(method_lock),
                "integrity_sha256": method_lock["integrity"]["sha256"],
            },
            "admission_report": {
                "file_sha256": methods.ADMISSION_REPORT_FILE_SHA256,
                "payload_sha256": methods.ADMISSION_REPORT_PAYLOAD_SHA256,
            },
            "geometry_manifest": {
                "file_sha256": file_sha256(methods.DEFAULT_MANIFEST),
                "integrity_sha256": validated["manifest"]["integrity"]["sha256"],
            },
            "cohort": {
                "ordered_core_bindings_sha256": core[
                    "ordered_core_bindings_sha256"
                ],
                "joint10_record_ids_sha256": efficacy["record_ids_sha256"],
            },
            "publisher": {
                "file_sha256": file_sha256(Path(__file__).resolve()),
            },
        },
        "operational_disclosure": {
            "canonical_output_policy": "atomic first-writer, no overwrite",
            "outcome_selection": False,
            "final_report_rows_selected_or_replaced": False,
            "duplicate_invocation_outcomes_are_not_publication_inputs": True,
        },
    }
    compact["integrity"] = {
        "algorithm": "sha256",
        "sha256": payload_sha256(compact),
    }
    validate_compact_result(compact)
    return compact


def validate_compact_result(compact: Mapping[str, Any]) -> None:
    integrity = compact.get("integrity")
    unsigned = dict(compact)
    unsigned.pop("integrity", None)
    if (
        compact.get("schema") != SCHEMA
        or compact.get("schema_version") != SCHEMA_VERSION
        or compact.get("status") != VALIDATION_STATUS
        or compact.get("behavioral_result_status") != BEHAVIORAL_RESULT_STATUS
        or compact.get("certificate_status") != CERTIFICATE_STATUS
        or not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != payload_sha256(unsigned)
    ):
        raise PublicationError("compact result schema, status, or integrity differs")
    if (
        compact.get("contains_source_text") is not False
        or compact.get("contains_full_vocabulary_vectors") is not False
        or compact.get("contains_model_generated_text") is not False
        or compact.get("official_longmemeval_leaderboard_score") is not False
    ):
        raise PublicationError("compact result publication scope differs")
    denominators = compact.get("denominators") or {}
    expected_counts = {
        "frozen_records": 16,
        "authorized_records": 16,
        "attempted_records": 16,
        "terminal_records": 16,
        "not_attempted_records": 0,
        "target_admitted_records": 10,
        "retained_available_records": 16,
        "joint_admitted_records": 10,
        "source_state_immutable_records": 16,
        "source_state_immutability_failures": 0,
        "certificate_records": 16,
        "certificate_probes": 32,
        "raw_report_record_failures": 16,
        "primary_result_failures": 0,
        "excluded_diagnostic_failures": 16,
    }
    if any(denominators.get(key) != value for key, value in expected_counts.items()):
        raise PublicationError("compact result denominators differ")
    primary = denominators.get("primary_condition_completed_records")
    if (
        not isinstance(primary, Mapping)
        or set(primary) != set(PRIMARY_CONDITION_IDS)
        or any(primary.get(condition_id) != 16 for condition_id in PRIMARY_CONDITION_IDS)
    ):
        raise PublicationError("compact primary condition completion differs")
    validation = compact.get("validation") or {}
    required_validation = (
        "all16_attempted_and_terminal",
        "all_manuscript_conditions_complete",
        "only_excluded_failures_reconciled",
        "source_state_immutable_all16",
        "no_output_based_selection_or_replacement",
        "geometry_correction_output_blind",
        "core_bindings_unchanged",
        "final_report_reconstructed_from_locked_inputs",
    )
    if not all(validation.get(key) is True for key in required_validation):
        raise PublicationError("compact validation sentinels differ")
    conditions = ((compact.get("efficacy") or {}).get("conditions") or {})
    if set(conditions) != set(MANUSCRIPT_CONDITION_IDS):
        raise PublicationError("compact manuscript condition set differs")
    if any(
        condition.get("completed_records") != 10
        for condition in conditions.values()
    ):
        raise PublicationError("compact efficacy condition is incomplete")
    exact_summary = (compact.get("efficacy") or {}).get(
        "exact_policy_summary"
    ) or {}
    if (
        conditions[methods.EXACT_DECREMENT_ID].get(
            "records_at_or_below_raw_omission_floor"
        )
        != 6
        or exact_summary.get("records_at_or_below_raw_omission_floor") != 6
    ):
        raise PublicationError("compact per-record raw-floor accounting differs")
    certificate = compact.get("certificate") or {}
    if (
        certificate.get("status") != CERTIFICATE_STATUS
        or certificate.get("records") != 16
        or certificate.get("probes") != 32
        or _finite(
            certificate.get("maximum_kl_nats"),
            name="compact certificate maximum KL",
        )
        > CERTIFICATE_PUBLICATION_MAX_KL_NATS
        or certificate.get("behavioral_raw_omission_is_distinct") is not True
        or certificate.get("raw_history_equality_claimed") is not False
    ):
        raise PublicationError("compact certificate scope or value differs")
    exact_policy = compact.get("exact_policy") or {}
    if (
        exact_policy.get("incremental_float64_decrement_records") != 0
        or exact_policy.get("fixed_c_refit_fallback_records") != 16
        or exact_policy.get("source_full_repack_fallback_records") != 0
        or exact_policy.get("all_executions_used_fixed_c_refit_fallback")
        is not True
        or exact_policy.get(
            "incremental_conversation_deletion_speed_claim_supported"
        )
        is not False
    ):
        raise PublicationError("compact exact-policy strata differ")
    cache = ((compact.get("diagnostics") or {}).get("cache_delete_shift") or {})
    if (
        cache.get("classification") != "incompatible_diagnostic"
        or cache.get("failed_records") != 16
        or cache.get("error_message_sha256")
        != EXPECTED_CACHE_FAILURE_MESSAGE_SHA256
        or cache.get("implies_exact_policy_failure") is not False
    ):
        raise PublicationError("compact cache diagnostic reconciliation differs")
    final_report = (
        (compact.get("source_artifacts") or {}).get("final_report") or {}
    )
    if (
        final_report.get("file_sha256")
        != EXPECTED_FINAL_REPORT_FILE_SHA256
        or final_report.get("payload_sha256")
        != EXPECTED_FINAL_REPORT_PAYLOAD_SHA256
        or final_report.get("final_row_payload_sha256")
        != EXPECTED_FINAL_ROW_PAYLOAD_SHA256
    ):
        raise PublicationError("compact final-report binding differs")
    leaked = sorted(set(_walk_keys(compact)) & _FORBIDDEN_PUBLIC_KEYS)
    if leaked:
        raise PublicationError(
            "compact result contains forbidden keys: " + ", ".join(leaked)
        )
    if any(
        not _is_sha256(value)
        for section in (compact.get("source_artifacts") or {}).values()
        if isinstance(section, Mapping)
        for key, value in section.items()
        if str(key).endswith("_sha256")
    ):
        raise PublicationError("compact result contains an invalid artifact hash")
    methods._assert_source_free(compact)
    methods.hardened._assert_finite_json(compact, path="compact_result")


def build_from_paths(
    *,
    report_path: str | Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    validated = _load_and_validate_final_report(report_path)
    return build_compact_result(validated)


def _latex_number(value: Any, *, digits: int = 2) -> str:
    number = _finite(value, name="LaTeX metric")
    if number == 0.0:
        return f"{0.0:.{digits}f}"
    magnitude = abs(number)
    if 0.01 <= magnitude < 1000.0:
        return f"{number:.{digits}f}"
    exponent = int(math.floor(math.log10(magnitude)))
    mantissa = number / (10**exponent)
    return f"{mantissa:.{digits}f}\\times 10^{{{exponent}}}"


def render_latex_macros(compact: Mapping[str, Any]) -> str:
    validate_compact_result(compact)
    conditions = compact["efficacy"]["conditions"]
    exact_summary = compact["efficacy"]["exact_policy_summary"]
    certificate = compact["certificate"]
    artifacts = compact["source_artifacts"]
    compact_file_sha256 = hashlib.sha256(
        deterministic_json(compact).encode("utf-8")
    ).hexdigest()
    metric_names = {
        methods.PRESENT_ID: "Present",
        methods.FRESH_RAW_OMISSION_ID: "Raw",
        methods.EXACT_DECREMENT_ID: "Exact",
        methods.FIXED_C_REFIT_ID: "Refit",
        methods.FP32_PROXY_ID: "FPThirtyTwo",
        methods.DECAY_ID: "Decay",
        methods.PROMPT_SUPPRESSION_ID: "Prompt",
    }
    macros: list[tuple[str, str]] = [
        ("LMChatMacrosGeneratedFromCompact", "true"),
        ("LMChatCompactSchema", SCHEMA),
        ("LMChatValidationStatus", compact["status"]),
        (
            "LMChatBehavioralResultStatus",
            compact["behavioral_result_status"],
        ),
        ("LMChatCertificateStatus", compact["certificate_status"]),
        ("LMChatManuscriptConditionsComplete", "true"),
        ("LMChatOnlyExcludedFailuresReconciled", "true"),
        ("LMChatContainsSourceText", "false"),
        ("LMChatOfficialLeaderboardScore", "false"),
        ("LMChatNoOutputBasedReplacement", "true"),
        ("LMChatGeometryCorrectionOutputBlind", "true"),
        ("LMChatCoreBindingsUnchanged", "true"),
        ("LMChatFrozenN", str(compact["denominators"]["frozen_records"])),
        ("LMChatAttemptedN", str(compact["denominators"]["attempted_records"])),
        ("LMChatTerminalN", str(compact["denominators"]["terminal_records"])),
        (
            "LMChatNotAttemptedN",
            str(compact["denominators"]["not_attempted_records"]),
        ),
        (
            "LMChatTargetAdmittedN",
            str(compact["denominators"]["target_admitted_records"]),
        ),
        (
            "LMChatRetainedAvailableN",
            str(compact["denominators"]["retained_available_records"]),
        ),
        (
            "LMChatJointN",
            str(compact["denominators"]["joint_admitted_records"]),
        ),
        (
            "LMChatSourceImmutableN",
            str(compact["denominators"]["source_state_immutable_records"]),
        ),
        (
            "LMChatRawOmissionCompleteN",
            str(
                compact["denominators"]["primary_condition_completed_records"][
                    methods.FRESH_RAW_OMISSION_ID
                ]
            ),
        ),
        (
            "LMChatExactPolicyCompleteN",
            str(
                compact["denominators"]["primary_condition_completed_records"][
                    methods.EXACT_DECREMENT_ID
                ]
            ),
        ),
        (
            "LMChatRefitCompleteN",
            str(
                compact["denominators"]["primary_condition_completed_records"][
                    methods.FIXED_C_REFIT_ID
                ]
            ),
        ),
        (
            "LMChatFPThirtyTwoCompleteN",
            str(
                compact["denominators"]["primary_condition_completed_records"][
                    methods.FP32_PROXY_ID
                ]
            ),
        ),
        (
            "LMChatDecayCompleteN",
            str(
                compact["denominators"]["primary_condition_completed_records"][
                    methods.DECAY_ID
                ]
            ),
        ),
        (
            "LMChatPromptCompleteN",
            str(
                compact["denominators"]["primary_condition_completed_records"][
                    methods.PROMPT_SUPPRESSION_ID
                ]
            ),
        ),
        (
            "LMChatCertificateRecordN",
            str(compact["denominators"]["certificate_records"]),
        ),
        (
            "LMChatCertificateProbeN",
            str(compact["denominators"]["certificate_probes"]),
        ),
        (
            "LMChatIncrementalN",
            str(compact["exact_policy"]["incremental_float64_decrement_records"]),
        ),
        (
            "LMChatRefitFallbackN",
            str(compact["exact_policy"]["fixed_c_refit_fallback_records"]),
        ),
        (
            "LMChatRawFallbackN",
            str(compact["exact_policy"]["source_full_repack_fallback_records"]),
        ),
        (
            "LMChatExactSuppressionMin",
            _latex_number(exact_summary["minimum_target_suppression_nats"]),
        ),
        (
            "LMChatExactSuppressionMax",
            _latex_number(exact_summary["maximum_target_suppression_nats"]),
        ),
        (
            "LMChatExactBelowRawFloor",
            _latex_number(exact_summary["mean_nats_below_raw_omission_floor"]),
        ),
        (
            "LMChatExactAtOrBelowRawN",
            str(exact_summary["records_at_or_below_raw_omission_floor"]),
        ),
        (
            "LMChatExactMeanAbsRetainedDrift",
            _latex_number(
                exact_summary[
                    "mean_absolute_retained_log_probability_drift_nats"
                ]
            ),
        ),
        (
            "LMChatCacheIncompatibleN",
            str(
                compact["diagnostics"]["cache_delete_shift"]["failed_records"]
            ),
        ),
        (
            "LMChatRawReportFailureN",
            str(compact["denominators"]["raw_report_record_failures"]),
        ),
        (
            "LMChatPrimaryFailureN",
            str(compact["denominators"]["primary_result_failures"]),
        ),
        (
            "LMChatCacheDeleteShiftClassification",
            "incompatible-diagnostic",
        ),
        (
            "LMChatCacheDeleteShiftFailedN",
            str(
                compact["diagnostics"]["cache_delete_shift"]["failed_records"]
            ),
        ),
        (
            "LMChatCertificateMeanKL",
            _latex_number(certificate["mean_kl_nats"]),
        ),
        (
            "LMChatCertificateMaxKL",
            _latex_number(certificate["maximum_kl_nats"]),
        ),
        (
            "LMChatExactRetainedRank",
            _latex_number(
                conditions[methods.EXACT_DECREMENT_ID][
                    "mean_retained_first_target_token_rank"
                ]
            ),
        ),
        (
            "LMChatFullReportSHA",
            artifacts["final_report"]["file_sha256"],
        ),
        ("LMChatCompactReportSHA", compact_file_sha256),
        (
            "LMChatMethodLockSHA",
            artifacts["method_lock"]["integrity_sha256"],
        ),
        (
            "LMChatAdmissionReportSHA",
            artifacts["admission_report"]["file_sha256"],
        ),
        (
            "LMChatCohortBindingSHA",
            artifacts["cohort"]["ordered_core_bindings_sha256"],
        ),
    ]
    for condition_id, stem in metric_names.items():
        metrics = conditions[condition_id]
        macros.extend(
            (
                (
                    f"LMChat{stem}Suppression",
                    _latex_number(
                        metrics["mean_target_suppression_vs_present_nats"]
                    ),
                ),
                (
                    f"LMChat{stem}TargetRawKL",
                    _latex_number(
                        metrics[
                            "mean_target_first_token_kl_raw_omission_to_method_nats"
                        ]
                    ),
                ),
                (
                    f"LMChat{stem}RetainedRawKL",
                    _latex_number(
                        metrics[
                            "mean_retained_first_token_kl_raw_omission_to_method_nats"
                        ]
                    ),
                ),
            )
        )
    lines = [
        "% Generated by gemma_sv.publish_longmemeval_chat_result.",
        "% Do not edit; regenerate from the validated compact16 JSON.",
    ]
    lines.extend(
        f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in macros
    )
    return "\n".join(lines) + "\n"


def _atomic_write(path: str | Path, content: bytes) -> None:
    output = Path(path).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish(
    *,
    report_path: str | Path = DEFAULT_REPORT,
    output_path: str | Path = DEFAULT_OUTPUT,
    macros_output_path: str | Path = DEFAULT_MACROS_OUTPUT,
) -> tuple[dict[str, Any], str]:
    protected = {
        Path(report_path).expanduser().resolve(strict=False),
        methods.DEFAULT_MANIFEST.resolve(strict=False),
        methods.DEFAULT_METHOD_LOCK.resolve(strict=False),
        methods.DEFAULT_ADMISSION_REPORT.resolve(strict=False),
        finalizer.DEFAULT_PARTIAL_REPORT.resolve(strict=False),
        finalizer.DEFAULT_PARTIAL_LOCK.resolve(strict=False),
    }
    outputs = (
        Path(output_path).expanduser().resolve(strict=False),
        Path(macros_output_path).expanduser().resolve(strict=False),
    )
    if outputs[0] == outputs[1] or any(output in protected for output in outputs):
        raise PublicationError("publication output aliases an input or peer output")
    compact = build_from_paths(report_path=report_path)
    macros = render_latex_macros(compact)
    _atomic_write(outputs[0], deterministic_json(compact).encode("utf-8"))
    _atomic_write(outputs[1], macros.encode("utf-8"))
    return compact, macros


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--macros-out", default=str(DEFAULT_MACROS_OUTPUT))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and render in memory without writing publication files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_only:
            compact = build_from_paths(report_path=args.report)
            render_latex_macros(compact)
            print(
                "validated final report "
                f"{EXPECTED_FINAL_REPORT_FILE_SHA256} for compact publication"
            )
        else:
            publish(
                report_path=args.report,
                output_path=args.out,
                macros_output_path=args.macros_out,
            )
            print(f"wrote compact result to {Path(args.out)}")
            print(f"wrote LaTeX macros to {Path(args.macros_out)}")
    except (OSError, PublicationError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"LongMemEval publication failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

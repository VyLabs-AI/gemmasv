"""Build a hash-bound LongMemEval forgetting case-study payload.

The figure and recorded demo intentionally show one predeclared case.  Model
outcomes are copied from a locked method report; public dialogue excerpts are
rehydrated from the pinned LongMemEval oracle and checked against the committed
source-free cohort.  The default report is the finalized all-16 artifact.  The
historical 15-of-16 report is accepted only with an explicit preview
acknowledgement and always carries the literal label ``INCOMPLETE CASE STUDY
PREVIEW``.

This module performs no model inference and never copies a raw context,
generation, or aggregate efficacy result into its outputs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import longmemeval_deletion_benchmark as longmemeval


PAYLOAD_SCHEMA = "gemma-sv-longmemeval-chat-forgetting-payload-v2"
COMPACT_SCHEMA = "gemma-sv-longmemeval-chat-geometry-compact-case-v1"
RESOLVER_FIXTURE_SCHEMA = "gemma-sv-recorded-memory-resolver-fixture-v1"
PREVIEW_LABEL = "INCOMPLETE CASE STUDY PREVIEW"
DEFAULT_RESOLVER_MODEL = "gemini-3.6-flash"
RESOLVER_FIXTURE_LABEL = (
    "Deterministic recorded resolver fixture — no provider call"
)

CASE_RECORD_ID = (
    "longmemeval-constrained-chat-v1-c8276e265e3db489c090cdcc"
)
CASE_AUTHORIZATION_POSITION = 2
CASE_ROW_PAYLOAD_SHA256 = (
    "13b5b4876adb4a44f39de59366083a30e85666235d98177aaa603a4632fd6d18"
)

METHOD_REPORT_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-deletion-methods-v1"
)
METHOD_LOCK_SCHEMA = (
    "gemma-sv-longmemeval-chat-geometry-method-authorization-lock-v1"
)

PARTIAL_REPORT_FILE_SHA256 = (
    "0358767e896f4dcfcad6d9bef108ecd2ce7076b06fab584bfe319c519bf1456a"
)
PARTIAL_REPORT_PAYLOAD_SHA256 = (
    "bcec7a147a2c21e6c2dca36760a5420459916ff7d1d89c96b8aafb36fd2e7e7e"
)
MANIFEST_FILE_SHA256 = (
    "df10404a5b11491791638190f61b52e30e667d17601819a99925181885bde480"
)
MANIFEST_INTEGRITY_SHA256 = (
    "04fb39a23cbe5e31ed652c0ff8d412df1ab2072999760fba81f7de95dee1a4db"
)
POLICY_FILE_SHA256 = (
    "bccb7d716182c495892f1ddb1693e4848995bc7da193cce26b9395fe2a736331"
)
POLICY_LOCK_SHA256 = (
    "3ce2bac27c60872788418d37149acb15790327b05abed8e5863c274453282b1c"
)
CENSUS_FILE_SHA256 = (
    "1d8ef04097cccf69021f56e79da8cf46e3e669967b5e7534edb4baa92f332d3b"
)
CENSUS_INTEGRITY_SHA256 = (
    "c01eacc10c21cbbc5b7488809d946b94f7013b78b2996f40f36cb5f322da2174"
)
ADMISSION_REPORT_FILE_SHA256 = (
    "36c2eaab9f39f1d0a5707460cc6fcb65027bf737dbe60f2b700993e8d2e82368"
)
ADMISSION_REPORT_PAYLOAD_SHA256 = (
    "f23ace307cf47fe9bc7ece575fd3190992660a815f79d8a048243a2736656988"
)
METHOD_LOCK_FILE_SHA256 = (
    "e965f3c5569e0833a32912b2d8537672af2c7123c89dc943fd671cf9d1f78a5c"
)
METHOD_LOCK_PAYLOAD_SHA256 = (
    "2fc805cf5ffc4eaf3a321c17ec31be4e356f042c4310fb7d07cddae8fb33df20"
)
METHOD_LOCK_INTEGRITY_SHA256 = (
    "4966a853142fd313eff412e408e5ca0b860bc2794877e7163142190a7a7218d4"
)
PARTIAL_LOCK_INTEGRITY_SHA256 = (
    "f3f332357ceda0796f54fb297d2aa423352989a9eaaefb7c54c6dafb5bd84c4f"
)
COMPLETION_LOCK_INTEGRITY_SHA256 = (
    "389c2726d7cdbc69e599cd50ea0707682d20d7bd4e1544f975d2525bd67a69dd"
)

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_MANIFEST = (
    BENCHMARKS / "longmemeval_chat_geometry_corrected_v1.json"
)
DEFAULT_POLICY = (
    BENCHMARKS / "longmemeval_chat_geometry_policy_lock_v1.json"
)
DEFAULT_CENSUS = BENCHMARKS / "longmemeval_chat_geometry_census_v1.json"
DEFAULT_METHOD_LOCK = (
    BENCHMARKS / "longmemeval_chat_geometry_method_authorization_v1.json"
)
DEFAULT_PARTIAL_LOCK = (
    BENCHMARKS / "longmemeval_chat_geometry_methods_partial_lock_v1.json"
)
DEFAULT_COMPLETION_LOCK = (
    BENCHMARKS / "longmemeval_chat_geometry_methods_completion_v1.json"
)
DEFAULT_ADMISSION_REPORT = (
    ROOT
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_geometry_admission_v1.json"
)
PREVIEW_METHOD_REPORT = (
    ROOT
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_geometry_methods_v1.json"
)
DEFAULT_METHOD_REPORT = (
    ROOT
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_geometry_methods_finalized_v1.json"
)
PREVIEW_COMPACT_OUTPUT = (
    ROOT
    / "paper_viz"
    / "payloads"
    / "longmemeval_chat_forgetting_compact_preview.json"
)
DEFAULT_COMPACT_OUTPUT = (
    ROOT
    / "paper_viz"
    / "payloads"
    / "longmemeval_chat_forgetting_compact_final.json"
)
PREVIEW_PAYLOAD_OUTPUT = (
    PACKAGE
    / "demo_site"
    / "assets"
    / "longmemeval_forgetting_preview.json"
)
DEFAULT_PAYLOAD_OUTPUT = (
    PACKAGE
    / "demo_site"
    / "assets"
    / "longmemeval_forgetting_final.json"
)

REQUIRED_CONDITIONS = (
    "present",
    "fresh_raw_omission",
    "exact_decrement",
    "fixed_c_refit",
)
EXPECTED_DEMO_SEQUENCE = (
    "chat",
    "recall",
    "resolve-request",
    "resolve-proposal",
    "confirm",
    "forget",
    "re-query",
    "retained",
    "certificate",
)
EXPECTED_DIALOGUE_SHAPE = {
    "target": (2, ("user", "assistant")),
    "retained": (2, ("user", "assistant")),
    "intervening": (2, ("user", "assistant")),
}
# These seals make the deliberately displayed excerpts immutable even if a
# presentation payload is re-signed without access to the pinned public oracle.
EXPECTED_DIALOGUE_BINDINGS = {
    "target": (
        (
            4,
            "bbe6a10124e1bfde35bbc038092bb216475164b273992dffdab8ed9bad20af05",
            "83f5c64a432340d9223d9431e57bdde9fd71c081349629932f6fa75a3dd84422",
        ),
        (
            5,
            "d2c731fbf179b98b1528e150946e19cd0ad5e87f441c64d44690d53e56cdd6bd",
            "6668a467e5f52bbfa21753343a455deb8b84b81045d00375d8fcfe21d5acd27c",
        ),
    ),
    "retained": (
        (
            2,
            "b8c51cdf9b26808760fa837ef835446066832918bf06e6ea017c2b635602ec9a",
            "a8cd294853dd06ec5dc3c9f9f0eba436f266b9b46e0a88c63e5b5d850dd190ae",
        ),
        (
            3,
            "80297665af53bef90c88b3d6b0f238b52c8c210dab98627f370598c1aea565ee",
            "e1b0ec1d9eeaddb302e339c71971d48bb6d08d4469a9da6aa09fcf89d57068db",
        ),
    ),
    "intervening": (
        (
            2,
            "7b6b41a5243a339970c8d868ac253b1acaaecba110329e55dbf0635f7be6cf55",
            "72cac20d7983539d7158eeea7376fc8040bd335c6c67fe77e279b49747740711",
        ),
        (
            3,
            "1e5f3c0d7326eb6ca2380fd7b5f6eb418ab23070cb68ec3bb502c6c9bd8769d7",
            "fef10e54aeb839d44957a2b051d722193ee39492591320eaae9d17c6d2b5ab63",
        ),
    ),
}
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "completion_text",
        "edited_text",
        "full_dialogue",
        "generated_text",
        "memory_text",
        "model_answer",
        "model_output",
        "model_output_text",
        "original_text",
        "raw_context",
        "raw_context_text",
        "raw_omitted_text",
        "source_text",
    }
)


class PayloadError(ValueError):
    """A source artifact or presentation payload violated the contract."""


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


def _reject_constant(value: str) -> None:
    raise PayloadError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PayloadError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def load_json(path: str | Path, *, name: str) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PayloadError(f"{name} is not valid strict UTF-8 JSON") from exc


def _load_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    value = load_json(path, name=name)
    if not isinstance(value, dict):
        raise PayloadError(f"{name} must be a JSON object")
    return value


def _require_file_hash(path: str | Path, expected: str, *, name: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise PayloadError(f"{name} SHA-256 drifted")
    return observed


def _validate_integrity(
    value: Mapping[str, Any],
    *,
    expected: str,
    name: str,
) -> None:
    integrity = value.get("integrity")
    unsigned = dict(value)
    unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != expected
        or payload_sha256(unsigned) != expected
    ):
        raise PayloadError(f"{name} integrity drifted")


def _json_pointer_get(value: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise PayloadError("JSON pointer must start with '/'")
    current = value
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (IndexError, ValueError) as exc:
                raise PayloadError(f"invalid list JSON pointer {pointer!r}") from exc
        elif isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise PayloadError(f"missing JSON pointer {pointer!r}")
    return current


def _finite_number(value: Any, *, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PayloadError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise PayloadError(f"{name} must be finite")
    return value


def _bound_value(
    artifact: Mapping[str, Any],
    *,
    artifact_id: str,
    pointer: str,
    allowed_types: tuple[type, ...],
) -> dict[str, Any]:
    value = _json_pointer_get(artifact, pointer)
    if isinstance(value, bool):
        valid = bool in allowed_types
    else:
        valid = isinstance(value, allowed_types)
    if not valid:
        raise PayloadError(f"{artifact_id}{pointer} has an invalid type")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite_number(value, name=f"{artifact_id}{pointer}")
    return {
        "artifact": artifact_id,
        "source_pointer": pointer,
        "value": copy.deepcopy(value),
    }


def _bound_number(
    artifact: Mapping[str, Any],
    *,
    artifact_id: str,
    pointer: str,
) -> dict[str, Any]:
    return _bound_value(
        artifact,
        artifact_id=artifact_id,
        pointer=pointer,
        allowed_types=(int, float),
    )


def _find_record(
    report: Mapping[str, Any],
    record_id: str,
    *,
    name: str,
) -> tuple[int, Mapping[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list):
        raise PayloadError(f"{name} records must be a list")
    matches = [
        (index, row)
        for index, row in enumerate(records)
        if isinstance(row, Mapping) and row.get("record_id") == record_id
    ]
    if len(matches) != 1:
        raise PayloadError(f"{name} must contain the case record exactly once")
    return matches[0]


def _probe_index(condition: Mapping[str, Any]) -> int:
    probes = (condition.get("deleted_target_quality") or {}).get("probes")
    if not isinstance(probes, list):
        raise PayloadError("condition target probes are missing")
    matches = [
        index
        for index, probe in enumerate(probes)
        if isinstance(probe, Mapping) and probe.get("probe_id") == "target_current"
    ]
    if len(matches) != 1:
        raise PayloadError("condition must contain one target_current probe")
    return matches[0]


def _condition_projection(
    report: Mapping[str, Any],
    *,
    record_index: int,
    condition_id: str,
) -> dict[str, Any]:
    base = f"/records/{record_index}/conditions/{condition_id}"
    condition = _json_pointer_get(report, base)
    if not isinstance(condition, Mapping) or condition.get("status") != "completed":
        raise PayloadError(f"case condition {condition_id} is not completed")
    probe_index = _probe_index(condition)
    target = f"{base}/deleted_target_quality/probes/{probe_index}"
    retained = f"{base}/retained_quality"
    projection = {
        "condition_id": condition_id,
        "status": "completed",
        "target": {
            "first_target_token_probability": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{target}/score/first_target_token_probability",
            ),
            "first_target_token_rank": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{target}/score/first_target_token_rank",
            ),
            "geometric_mean_probability": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{target}/score/geometric_mean_probability",
            ),
            "mean_log_probability": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{target}/score/mean_log_probability",
            ),
            "target_token_count": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{target}/score/target_token_count",
            ),
            "full_vocabulary_kl_to_raw_repack_nats": _bound_number(
                report,
                artifact_id="method_report",
                pointer=(
                    f"{target}/"
                    "full_vocabulary_kl_raw_repack_to_method_nats"
                ),
            ),
        },
        "retained": {
            "first_target_token_probability": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{retained}/score/first_target_token_probability",
            ),
            "first_target_token_rank": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{retained}/score/first_target_token_rank",
            ),
            "geometric_mean_probability": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{retained}/score/geometric_mean_probability",
            ),
            "mean_log_probability": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{retained}/score/mean_log_probability",
            ),
            "mean_log_probability_drift_from_raw_repack_nats": _bound_number(
                report,
                artifact_id="method_report",
                pointer=(
                    f"{retained}/"
                    "mean_log_probability_drift_from_raw_repack_nats"
                ),
            ),
            "full_vocabulary_kl_to_raw_repack_nats": _bound_number(
                report,
                artifact_id="method_report",
                pointer=(
                    f"{retained}/"
                    "full_vocabulary_kl_raw_repack_to_method_nats"
                ),
            ),
        },
    }
    if condition_id == "exact_decrement":
        projection["execution"] = {
            "executed_method": _bound_value(
                report,
                artifact_id="method_report",
                pointer=f"{base}/semantics/executed_method",
                allowed_types=(str,),
            ),
            "fixed_c_refit_fallback": _bound_value(
                report,
                artifact_id="method_report",
                pointer=f"{base}/semantics/fixed_c_refit_fallback",
                allowed_types=(bool,),
            ),
            "full_repack_fallback": _bound_value(
                report,
                artifact_id="method_report",
                pointer=f"{base}/semantics/full_repack_fallback",
                allowed_types=(bool,),
            ),
            "decrement_fallbacks": _bound_number(
                report,
                artifact_id="method_report",
                pointer=f"{base}/semantics/decrement_fallbacks",
            ),
        }
    return projection


def _validate_method_report(
    report: Mapping[str, Any],
    *,
    report_file_sha256: str,
    method_lock: Mapping[str, Any],
    partial_lock: Mapping[str, Any],
    completion_lock: Mapping[str, Any] | None,
    allow_incomplete_preview: bool,
) -> str:
    if (
        report.get("schema") != METHOD_REPORT_SCHEMA
        or report.get("contains_source_text") is not False
        or report.get("contains_full_vocabulary_vectors") is not False
        or report.get("official_longmemeval_leaderboard_score") is not False
        or report.get("method_lock_integrity_sha256")
        != METHOD_LOCK_INTEGRITY_SHA256
    ):
        raise PayloadError("method report contract drifted")

    authorization = method_lock.get("authorization") or {}
    authorized_rows = authorization.get("admission_records")
    records = report.get("records")
    if not isinstance(authorized_rows, list) or not isinstance(records, list):
        raise PayloadError("method report authorization is incomplete")
    authorized_ids = [
        str(row.get("record_id") or "")
        for row in authorized_rows
        if isinstance(row, Mapping)
    ]
    report_ids = [
        str(row.get("record_id") or "")
        for row in records
        if isinstance(row, Mapping)
    ]
    if (
        len(authorized_ids) != 16
        or len(report_ids) != 16
        or report_ids != authorized_ids
        or authorized_ids[CASE_AUTHORIZATION_POSITION - 1] != CASE_RECORD_ID
    ):
        raise PayloadError("method report is not the locked ordered all16")

    case_index, case_row = _find_record(
        report,
        CASE_RECORD_ID,
        name="method report",
    )
    if (
        case_index != CASE_AUTHORIZATION_POSITION - 1
        or payload_sha256(case_row) != CASE_ROW_PAYLOAD_SHA256
        or case_row.get("predeclared_joint_admitted") is not True
    ):
        raise PayloadError("case row differs from its committed predeclared seal")
    for condition_id in REQUIRED_CONDITIONS:
        condition = (case_row.get("conditions") or {}).get(condition_id)
        if not isinstance(condition, Mapping) or condition.get("status") != "completed":
            raise PayloadError(f"case condition {condition_id} is unavailable")
    certificate = case_row.get("solver_certificate")
    if not isinstance(certificate, Mapping) or certificate.get("status") != "completed":
        raise PayloadError("case solver certificate is unavailable")

    summary = report.get("summary") or {}
    denominators = summary.get("denominators") or {}
    attempted = int(denominators.get("attempted_records", -1))
    not_attempted = int(denominators.get("not_attempted_records", -1))
    status = str(report.get("status") or "")
    if status == "running":
        if not allow_incomplete_preview:
            raise PermissionError(
                "partial15 input requires --allow-incomplete-preview"
            )
        if (
            report_file_sha256 != PARTIAL_REPORT_FILE_SHA256
            or payload_sha256(report) != PARTIAL_REPORT_PAYLOAD_SHA256
            or attempted != 15
            or not_attempted != 1
            or records[-1].get("status") != "not_attempted"
            or (partial_lock.get("partial_report") or {}).get("file_sha256")
            != PARTIAL_REPORT_FILE_SHA256
            or (partial_lock.get("partial_report") or {}).get("payload_sha256")
            != PARTIAL_REPORT_PAYLOAD_SHA256
        ):
            raise PayloadError("preview is not the exact committed partial15 run")
        sealed = partial_lock.get("completed_rows") or []
        case_seals = [
            row
            for row in sealed
            if isinstance(row, Mapping) and row.get("record_id") == CASE_RECORD_ID
        ]
        if (
            len(case_seals) != 1
            or case_seals[0].get("payload_sha256")
            != CASE_ROW_PAYLOAD_SHA256
        ):
            raise PayloadError("partial lock does not seal the case row")
        return "incomplete_case_study_preview"

    if status not in {"completed", "completed_with_record_failures"}:
        raise PayloadError("method report status is neither partial nor final")
    if attempted != 16 or not_attempted != 0 or any(
        row.get("status") == "not_attempted" for row in records
    ):
        raise PayloadError("final method report does not account for all16")
    completion = report.get("external_abort_completion")
    finalization = report.get("finalization")
    completed_from_lock = bool(
        isinstance(completion, Mapping)
        and completion.get("completion_lock_integrity_sha256")
        == COMPLETION_LOCK_INTEGRITY_SHA256
        and completion.get("partial_report_file_sha256")
        == PARTIAL_REPORT_FILE_SHA256
    )
    finalized_from_lock = bool(
        isinstance(finalization, Mapping)
        and finalization.get("partial_lock_integrity_sha256")
        == PARTIAL_LOCK_INTEGRITY_SHA256
        and finalization.get("partial_report_file_sha256")
        == PARTIAL_REPORT_FILE_SHA256
    )
    if not (completed_from_lock or finalized_from_lock):
        raise PayloadError("final all16 report is not bound to the committed completion")
    if completed_from_lock and (
        completion_lock is None
        or (completion_lock.get("integrity") or {}).get("sha256")
        != COMPLETION_LOCK_INTEGRITY_SHA256
    ):
        raise PayloadError(
            "completion-script final report requires its committed completion lock"
        )
    return "final_case_study"


def _validate_artifacts(
    *,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_lock: Mapping[str, Any],
    completion_lock: Mapping[str, Any] | None,
) -> None:
    _validate_integrity(
        manifest,
        expected=MANIFEST_INTEGRITY_SHA256,
        name="geometry manifest",
    )
    if (
        manifest.get("schema") != "gemma-sv-longmemeval-chat-geometry-fix-v1"
        or manifest.get("contains_source_text") is not False
        or policy.get("lock_sha256") != POLICY_LOCK_SHA256
        or policy.get("selection_uses_model_outputs") is not False
    ):
        raise PayloadError("geometry cohort or policy drifted")
    _validate_integrity(
        census,
        expected=CENSUS_INTEGRITY_SHA256,
        name="geometry census",
    )
    if (
        payload_sha256(admission_report) != ADMISSION_REPORT_PAYLOAD_SHA256
        or admission_report.get("contains_source_text") is not False
        or method_lock.get("schema") != METHOD_LOCK_SCHEMA
        or payload_sha256(method_lock) != METHOD_LOCK_PAYLOAD_SHA256
    ):
        raise PayloadError("admission report or method lock payload drifted")
    _validate_integrity(
        method_lock,
        expected=METHOD_LOCK_INTEGRITY_SHA256,
        name="method lock",
    )
    _validate_integrity(
        partial_lock,
        expected=PARTIAL_LOCK_INTEGRITY_SHA256,
        name="partial lock",
    )
    if completion_lock is not None:
        _validate_integrity(
            completion_lock,
            expected=COMPLETION_LOCK_INTEGRITY_SHA256,
            name="completion lock",
        )
    artifacts = method_lock.get("artifacts") or {}
    expected_artifacts = {
        "admission_report_file_sha256": ADMISSION_REPORT_FILE_SHA256,
        "admission_report_payload_sha256": ADMISSION_REPORT_PAYLOAD_SHA256,
        "census_file_sha256": CENSUS_FILE_SHA256,
        "census_integrity_sha256": CENSUS_INTEGRITY_SHA256,
        "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "manifest_integrity_sha256": MANIFEST_INTEGRITY_SHA256,
        "policy_file_sha256": POLICY_FILE_SHA256,
        "policy_lock_sha256": POLICY_LOCK_SHA256,
    }
    if any(artifacts.get(key) != value for key, value in expected_artifacts.items()):
        raise PayloadError("method lock artifact bindings drifted")


def _build_compact_case_report(
    report: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_lock: Mapping[str, Any],
    completion_lock: Mapping[str, Any] | None,
    *,
    report_file_sha256: str,
    allow_incomplete_preview: bool,
) -> dict[str, Any]:
    """Project one predeclared case without copying aggregate results."""

    state = _validate_method_report(
        report,
        report_file_sha256=report_file_sha256,
        method_lock=method_lock,
        partial_lock=partial_lock,
        completion_lock=completion_lock,
        allow_incomplete_preview=allow_incomplete_preview,
    )
    report_index, report_row = _find_record(
        report,
        CASE_RECORD_ID,
        name="method report",
    )
    admission_index, _admission_row = _find_record(
        admission_report,
        CASE_RECORD_ID,
        name="admission report",
    )
    manifest_index, manifest_row = _find_record(
        manifest,
        CASE_RECORD_ID,
        name="geometry manifest",
    )
    if not (
        report_index == admission_index == manifest_index
        == CASE_AUTHORIZATION_POSITION - 1
    ):
        raise PayloadError("case position differs across bound artifacts")
    if (
        (report_row.get("geometry_binding") or {}).get(
            "manifest_record_integrity_sha256"
        )
        != (manifest_row.get("record_integrity") or {}).get("sha256")
    ):
        raise PayloadError("case method row is not bound to the geometry record")

    geometry_base = f"/records/{admission_index}/geometry_safety"
    report_denominators = (report.get("summary") or {}).get("denominators") or {}
    source_artifacts = {
        "method_report": {
            "schema": METHOD_REPORT_SCHEMA,
            "status": report.get("status"),
            "file_sha256": report_file_sha256,
            "payload_sha256": payload_sha256(report),
        },
        "geometry_manifest": {
            "file_sha256": MANIFEST_FILE_SHA256,
            "integrity_sha256": MANIFEST_INTEGRITY_SHA256,
        },
        "admission_report": {
            "file_sha256": ADMISSION_REPORT_FILE_SHA256,
            "payload_sha256": ADMISSION_REPORT_PAYLOAD_SHA256,
        },
        "method_lock": {
            "file_sha256": METHOD_LOCK_FILE_SHA256,
            "payload_sha256": METHOD_LOCK_PAYLOAD_SHA256,
            "integrity_sha256": METHOD_LOCK_INTEGRITY_SHA256,
        },
        "partial_lock": {
            "integrity_sha256": PARTIAL_LOCK_INTEGRITY_SHA256,
        },
    }
    if completion_lock is not None:
        source_artifacts["completion_lock"] = {
            "integrity_sha256": COMPLETION_LOCK_INTEGRITY_SHA256,
        }
    compact: dict[str, Any] = {
        "schema": COMPACT_SCHEMA,
        "schema_version": 1,
        "status": state,
        "preview_label": PREVIEW_LABEL if state.startswith("incomplete") else None,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "official_longmemeval_leaderboard_score": False,
        "case_study_only": True,
        "aggregate_claims": [],
        "report_state": {
            "authorized_records": 16,
            "attempted_records": int(
                report_denominators.get("attempted_records", -1)
            ),
            "not_attempted_records": int(
                report_denominators.get("not_attempted_records", -1)
            ),
            "all16_complete": state == "final_case_study",
            "aggregate_claims_allowed": False,
        },
        "source_artifacts": source_artifacts,
        "case": {
            "record_id": CASE_RECORD_ID,
            "authorization_position": CASE_AUTHORIZATION_POSITION,
            "row_payload_sha256": CASE_ROW_PAYLOAD_SHA256,
            "manifest_record_integrity_sha256": (
                manifest_row["record_integrity"]["sha256"]
            ),
            "selection": {
                "rule": "first authorized jointly admitted record",
                "selected_before_method_scoring": True,
                "selected_from_method_outcomes": False,
                "record_replacement": False,
            },
            "admission": {
                "joint_admitted": True,
                "target_admitted": True,
                "retained_available": True,
            },
            "geometry": {
                "tokens_strictly_after_owned": _bound_number(
                    admission_report,
                    artifact_id="admission_report",
                    pointer=f"{geometry_base}/observed_tokens_strictly_after_owned",
                ),
                "minimum_tokens_strictly_after_owned": _bound_number(
                    admission_report,
                    artifact_id="admission_report",
                    pointer=f"{geometry_base}/minimum_tokens_strictly_after_owned",
                ),
                "local_window_tokens": _bound_number(
                    admission_report,
                    artifact_id="admission_report",
                    pointer=f"{geometry_base}/true_local_window_tokens",
                ),
                "owned_round_outside_local_window": _bound_value(
                    admission_report,
                    artifact_id="admission_report",
                    pointer=(
                        f"{geometry_base}/"
                        "owned_round_strictly_outside_local_window_before_query"
                    ),
                    allowed_types=(bool,),
                ),
            },
            "outcomes": [
                _condition_projection(
                    report,
                    record_index=report_index,
                    condition_id=condition_id,
                )
                for condition_id in REQUIRED_CONDITIONS
            ],
            "certificate": {
                "claim": "exact decrement versus fixed-C retained-key refit",
                "scope": "full vocabulary at the first target token of both registered probes",
                "max_output_kl_nats": _bound_number(
                    report,
                    artifact_id="method_report",
                    pointer=(
                        f"/records/{report_index}/solver_certificate/"
                        "max_output_kl_nats"
                    ),
                ),
                "mean_output_kl_nats": _bound_number(
                    report,
                    artifact_id="method_report",
                    pointer=(
                        f"/records/{report_index}/solver_certificate/"
                        "mean_output_kl_nats"
                    ),
                ),
                "decrement_fallbacks": _bound_number(
                    report,
                    artifact_id="method_report",
                    pointer=(
                        f"/records/{report_index}/solver_certificate/"
                        "solver_diagnostics/decrement_fallbacks"
                    ),
                ),
                "head_gate_solves": _bound_number(
                    report,
                    artifact_id="method_report",
                    pointer=(
                        f"/records/{report_index}/solver_certificate/"
                        "solver_diagnostics/head_gate_solves"
                    ),
                ),
                "head_gates": _bound_number(
                    report,
                    artifact_id="method_report",
                    pointer=(
                        f"/records/{report_index}/solver_certificate/"
                        "solver_diagnostics/head_gates"
                    ),
                ),
                "full_repack_certificate": False,
                "raw_omission_reference_distinct": True,
            },
        },
    }
    compact["integrity"] = {
        "algorithm": "sha256",
        "sha256": payload_sha256(compact),
    }
    return compact


def validate_compact_case_report(
    compact: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_lock: Mapping[str, Any],
    completion_lock: Mapping[str, Any] | None,
    report_file_sha256: str,
    allow_incomplete_preview: bool,
) -> None:
    if compact.get("schema") != COMPACT_SCHEMA:
        raise PayloadError("compact case report schema drifted")
    integrity = compact.get("integrity")
    unsigned = dict(compact)
    unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != payload_sha256(unsigned)
    ):
        raise PayloadError("compact case report integrity drifted")
    expected = _build_compact_case_report(
        report,
        admission_report,
        manifest,
        method_lock,
        partial_lock,
        completion_lock,
        report_file_sha256=report_file_sha256,
        allow_incomplete_preview=allow_incomplete_preview,
    )
    if dict(compact) != expected:
        raise PayloadError("compact case report differs from bound source metrics")


def build_compact_case_report(
    report: Mapping[str, Any],
    admission_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    partial_lock: Mapping[str, Any],
    completion_lock: Mapping[str, Any] | None,
    *,
    report_file_sha256: str,
    allow_incomplete_preview: bool,
) -> dict[str, Any]:
    """Build and independently revalidate the compact source projection."""

    compact = _build_compact_case_report(
        report,
        admission_report,
        manifest,
        method_lock,
        partial_lock,
        completion_lock,
        report_file_sha256=report_file_sha256,
        allow_incomplete_preview=allow_incomplete_preview,
    )
    validate_compact_case_report(
        compact,
        report=report,
        admission_report=admission_report,
        manifest=manifest,
        method_lock=method_lock,
        partial_lock=partial_lock,
        completion_lock=completion_lock,
        report_file_sha256=report_file_sha256,
        allow_incomplete_preview=allow_incomplete_preview,
    )
    return compact


def _validate_example_descriptor(
    example: longmemeval.LongMemEvalExample,
    descriptor: Mapping[str, Any],
    *,
    name: str,
) -> None:
    expected = {
        "source_id": example.source_id,
        "question_id": example.question_id,
        "question_type": example.question_type,
        "source_row_sha256": example.source_row_sha256,
        "question_sha256": longmemeval.text_sha256(example.question),
        "current_answer_sha256": longmemeval.text_sha256(example.answer),
        "question_date_sha256": longmemeval.text_sha256(example.question_date),
    }
    if any(descriptor.get(key) != value for key, value in expected.items()):
        raise PayloadError(f"{name} public source descriptor drifted")


def _session_by_id(
    example: longmemeval.LongMemEvalExample,
    session_id: str,
) -> longmemeval.LongMemEvalSession:
    matches = [
        session
        for session in example.haystack_sessions
        if session.session_id == session_id
    ]
    if len(matches) != 1:
        raise PayloadError("public dialogue session is missing or ambiguous")
    return matches[0]


def _answer_round_turns(
    session: longmemeval.LongMemEvalSession,
) -> list[tuple[int, longmemeval.LongMemEvalTurn]]:
    starts: set[int] = set()
    for index, turn in enumerate(session.turns):
        if turn.has_answer is not True:
            continue
        role = turn.role.strip().casefold()
        start = index if role == "user" else index - 1
        if (
            start < 0
            or start + 1 >= len(session.turns)
            or session.turns[start].role.strip().casefold() != "user"
            or session.turns[start + 1].role.strip().casefold()
            not in {"assistant", "model"}
        ):
            raise PayloadError("answer-bearing turn is not a complete dialogue round")
        starts.add(start)
    if not starts:
        raise PayloadError("public dialogue has no answer-bearing round")
    return [
        (index, session.turns[index])
        for start in sorted(starts)
        for index in (start, start + 1)
    ]


def _selected_turns(
    session: longmemeval.LongMemEvalSession,
    indices: Sequence[int],
    *,
    name: str,
) -> list[tuple[int, longmemeval.LongMemEvalTurn]]:
    if len(set(indices)) != len(indices) or list(indices) != sorted(indices):
        raise PayloadError(f"{name} turn selection must be ordered and unique")
    try:
        selected = [(index, session.turns[index]) for index in indices]
    except IndexError as exc:
        raise PayloadError(f"{name} turn selection is outside the public session") from exc
    roles = tuple(turn.role.strip().casefold() for _, turn in selected)
    if roles != tuple("user" if index % 2 == 0 else "assistant" for index in range(len(indices))):
        raise PayloadError(f"{name} turn selection is not complete user-assistant rounds")
    return selected


def _validate_extension_round(
    example: longmemeval.LongMemEvalExample,
    session: longmemeval.LongMemEvalSession,
    descriptor: Mapping[str, Any],
) -> list[tuple[int, longmemeval.LongMemEvalTurn]]:
    indices = descriptor.get("official_turn_indices")
    expected_hashes = descriptor.get("source_turn_sha256")
    if (
        descriptor.get("purpose") != "geometry_only"
        or descriptor.get("source_id") != example.source_id
        or descriptor.get("session_id") != session.session_id
        or descriptor.get("source_session_sha256") != session.source_session_sha256
        or not isinstance(indices, list)
        or not isinstance(expected_hashes, list)
        or len(indices) != 2
        or len(expected_hashes) != 2
    ):
        raise PayloadError("intervening public round descriptor drifted")
    selected = _selected_turns(session, indices, name="intervening")
    if [turn.source_turn_sha256 for _, turn in selected] != expected_hashes:
        raise PayloadError("intervening public turn hashes drifted")
    return selected


def _quote_exact_source_unit(
    *,
    example: longmemeval.LongMemEvalExample,
    session: longmemeval.LongMemEvalSession,
    turn_index: int,
    turn: longmemeval.LongMemEvalTurn,
    exact_text: str,
    unit: str,
) -> dict[str, Any]:
    if unit not in {"complete_sentence", "complete_source_line"}:
        raise PayloadError("exact public quote unit is invalid")
    if (
        not exact_text
        or turn.content.count(exact_text) != 1
        or "..." in exact_text
        or "…" in exact_text
    ):
        raise PayloadError("exact public quote is missing, ambiguous, or truncated")
    span_start = turn.content.index(exact_text)
    span_end = span_start + len(exact_text)
    return {
        "origin": "public_longmemeval_dialogue",
        "license": longmemeval.DATASET_LICENSE,
        "source_id": example.source_id,
        "source_row_sha256": example.source_row_sha256,
        "session_id": session.session_id,
        "source_session_sha256": session.source_session_sha256,
        "source_turn_index": turn_index,
        "source_turn_sha256": turn.source_turn_sha256,
        "source_content_sha256": longmemeval.text_sha256(turn.content),
        "source_span_start": span_start,
        "source_span_end": span_end,
        "role": "assistant" if turn.role.casefold() == "model" else turn.role,
        "display_text": exact_text,
        "display_text_sha256": longmemeval.text_sha256(exact_text),
        "truncated": False,
        "excerpt_policy": {
            "selection": "exact UTF-8 source substring",
            "normalization": "none",
            "unit": unit,
        },
    }


def _query_quote(
    example: longmemeval.LongMemEvalExample,
) -> dict[str, Any]:
    return {
        "origin": "public_longmemeval_query",
        "license": longmemeval.DATASET_LICENSE,
        "source_id": example.source_id,
        "source_row_sha256": example.source_row_sha256,
        "display_text": example.question,
        "display_text_sha256": longmemeval.text_sha256(example.question),
        "official_longmemeval_question": True,
        "evaluation_use": "official question replayed by evaluator",
        "evaluator_added_text": False,
    }


def build_public_dialogue(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rehydrate only the deliberately quoted public dialogue for the case."""

    _, manifest_record = _find_record(
        manifest,
        CASE_RECORD_ID,
        name="geometry manifest",
    )
    examples = longmemeval.extract_longmemeval_examples(rows)
    by_source = {example.source_id: example for example in examples}
    target_descriptor = manifest_record.get("target") or {}
    retained_descriptor = manifest_record.get("retained_probe") or {}
    extensions = manifest_record.get("geometry_extensions") or {}
    suffix_rounds = extensions.get("suffix_distance_rounds")
    if not isinstance(suffix_rounds, list) or len(suffix_rounds) != 3:
        raise PayloadError("case must retain its three registered suffix gap rounds")
    intervening_descriptor = suffix_rounds[0]
    if not isinstance(intervening_descriptor, Mapping):
        raise PayloadError("intervening public round descriptor is invalid")
    try:
        target = by_source[str(target_descriptor["source_id"])]
        retained = by_source[str(retained_descriptor["source_id"])]
        intervening = by_source[str(intervening_descriptor["source_id"])]
    except (KeyError, TypeError) as exc:
        raise PayloadError("case public source rows are missing") from exc
    _validate_example_descriptor(target, target_descriptor, name="target")
    _validate_example_descriptor(retained, retained_descriptor, name="retained")

    target_session_id = str(
        target_descriptor.get("owned_latest_evidence_session_id") or ""
    )
    retained_ids = list(retained_descriptor.get("evidence_session_ids") or ())
    if len(retained_ids) != 1:
        raise PayloadError("retained dialogue must bind one evidence session")
    target_session = _session_by_id(target, target_session_id)
    retained_session = _session_by_id(retained, str(retained_ids[0]))
    intervening_session = _session_by_id(
        intervening,
        str(intervening_descriptor["session_id"]),
    )
    target_answer_turns = _answer_round_turns(target_session)
    retained_answer_turns = _answer_round_turns(retained_session)
    if [index for index, _ in target_answer_turns] != [4, 5]:
        raise PayloadError("target answer-bearing round position drifted")
    if [index for index, _ in retained_answer_turns] != [2, 3]:
        raise PayloadError("retained answer-bearing round position drifted")
    target_turns = _selected_turns(
        target_session,
        (4, 5),
        name="target",
    )
    retained_turns = _selected_turns(
        retained_session,
        (2, 3),
        name="retained",
    )
    intervening_turns = _validate_extension_round(
        intervening,
        intervening_session,
        intervening_descriptor,
    )
    target_texts = (
        (
            "Since I'll have four bikes with me, I'll make sure to book the "
            "accommodations with bike storage in advance to ensure they can "
            "accommodate all my bikes."
        ),
        "Congratulations on the new hybrid bike!",
    )
    retained_texts = (
        "Can you provide me with the contact details of the local tourism board "
        "of Speyer?",
        "Phone: +49 (0) 62 32 / 14 23 - 0",
    )
    intervening_texts = (
        "Which one would you say is the best for a romantic dinner?",
        "For a romantic dinner, I would recommend Roscioli.",
    )
    return {
        "target": {
            "label": "target example selected from public LongMemEval",
            "source_relation": "separate benchmark example; not continuous chat",
            "quotes": [
                _quote_exact_source_unit(
                    example=target,
                    session=target_session,
                    turn_index=index,
                    turn=turn,
                    exact_text=exact_text,
                    unit="complete_sentence",
                )
                for (index, turn), exact_text in zip(target_turns, target_texts)
            ],
        },
        "retained": {
            "label": "retained example selected from public LongMemEval",
            "source_relation": "separate benchmark example; not continuous chat",
            "quotes": [
                _quote_exact_source_unit(
                    example=retained,
                    session=retained_session,
                    turn_index=index,
                    turn=turn,
                    exact_text=exact_text,
                    unit=(
                        "complete_source_line"
                        if index == 3
                        else "complete_sentence"
                    ),
                )
                for (index, turn), exact_text in zip(
                    retained_turns,
                    retained_texts,
                )
            ],
        },
        "intervening": {
            "label": "Example from the intervening conversation",
            "source_relation": (
                "separate public LongMemEval example assembled into the gap"
            ),
            "inside_measured_intervening_gap": True,
            "displayed_registered_suffix_rounds": 1,
            "total_registered_suffix_rounds": len(suffix_rounds),
            "quotes": [
                _quote_exact_source_unit(
                    example=intervening,
                    session=intervening_session,
                    turn_index=index,
                    turn=turn,
                    exact_text=exact_text,
                    unit="complete_sentence",
                )
                for (index, turn), exact_text in zip(
                    intervening_turns,
                    intervening_texts,
                )
            ],
        },
        "queries": {
            "target": _query_quote(target),
            "retained": _query_quote(retained),
        },
    }


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _validate_bound_values(value: Any, compact: Mapping[str, Any]) -> None:
    if isinstance(value, Mapping):
        if set(value) == {"artifact", "source_pointer", "value"}:
            artifact_id = value["artifact"]
            if artifact_id not in {"method_report", "admission_report"}:
                raise PayloadError("bound metric names an unknown source artifact")
            if not isinstance(value["source_pointer"], str):
                raise PayloadError("bound metric source pointer is invalid")
            if isinstance(value["value"], (int, float)) and not isinstance(
                value["value"], bool
            ):
                _finite_number(value["value"], name="bound metric")
            return
        for item in value.values():
            _validate_bound_values(item, compact)
    elif isinstance(value, list):
        for item in value:
            _validate_bound_values(item, compact)


def _validate_quote(quote: Mapping[str, Any], *, origin: str) -> None:
    display_text = quote.get("display_text")
    if (
        quote.get("origin") != origin
        or not isinstance(display_text, str)
        or not display_text
        or "..." in display_text
        or "…" in display_text
        or quote.get("display_text_sha256")
        != longmemeval.text_sha256(display_text)
    ):
        raise PayloadError("public text quote is not explicitly source-bound")
    if origin == "public_longmemeval_dialogue":
        policy = quote.get("excerpt_policy") or {}
        if (
            not isinstance(quote.get("source_turn_index"), int)
            or not isinstance(quote.get("source_turn_sha256"), str)
            or len(quote["source_turn_sha256"]) != 64
            or not isinstance(quote.get("source_span_start"), int)
            or not isinstance(quote.get("source_span_end"), int)
            or quote["source_span_end"] - quote["source_span_start"]
            != len(display_text)
            or quote.get("truncated") is not False
            or policy.get("selection") != "exact UTF-8 source substring"
            or policy.get("normalization") != "none"
            or policy.get("unit")
            not in {"complete_sentence", "complete_source_line"}
        ):
            raise PayloadError(
                "public dialogue quote lacks its exact source-span binding"
            )


def _resolver_record_id(
    candidate_kind: str,
    quotes: Sequence[Mapping[str, Any]],
) -> str:
    """Derive a stable opaque ID from source bindings, not semantic text."""

    digest = payload_sha256(
        {
            "namespace": RESOLVER_FIXTURE_SCHEMA,
            "candidate_kind": candidate_kind,
            "source_turn_sha256": [
                quote["source_turn_sha256"] for quote in quotes
            ],
            "display_text_sha256": [
                quote["display_text_sha256"] for quote in quotes
            ],
        }
    )
    return f"memory_{digest[:24]}"


def _resolver_candidate(
    *,
    candidate_kind: str,
    label: str,
    summary: str,
    owned_exchange_ref: str,
    quotes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "record_id": _resolver_record_id(candidate_kind, quotes),
        "label": label,
        "summary": summary,
        "source_turn_ids": [
            f"turn_{quote['source_turn_sha256'][:20]}" for quote in quotes
        ],
        "deletion_scope": "complete_exchange",
        "owned_exchange_ref": owned_exchange_ref,
    }


def _make_recorded_resolver_fixture(
    dialogue: Mapping[str, Any],
    deletion_action: Mapping[str, Any],
) -> dict[str, Any]:
    target_quotes = dialogue["target"]["quotes"]
    retained_quotes = dialogue["retained"]["quotes"]
    target = _resolver_candidate(
        candidate_kind="target_bicycle_update",
        label="Current number of bicycles",
        summary=(
            "The user said they would have four bikes after adding a new "
            "hybrid bicycle."
        ),
        owned_exchange_ref="/case/dialogue/target/quotes",
        quotes=target_quotes,
    )
    retained = _resolver_candidate(
        candidate_kind="retained_speyer_contact",
        label="Speyer tourism-board contact details",
        summary=(
            "The user requested the Speyer tourism board contact details and "
            "received its phone number."
        ),
        owned_exchange_ref="/case/dialogue/retained/quotes",
        quotes=retained_quotes,
    )
    catalog = [target, retained]
    catalog_hash = payload_sha256(catalog)
    request_text = deletion_action["display_text"]
    return {
        "schema": RESOLVER_FIXTURE_SCHEMA,
        "schema_version": 1,
        "mode": "deterministic_recorded_fixture",
        "label": RESOLVER_FIXTURE_LABEL,
        "provider": {
            "call_performed": False,
            "artifact_present": False,
            "artifact": None,
        },
        "network_request_performed": False,
        "model_configuration": {
            "default_model_id": DEFAULT_RESOLVER_MODEL,
            "model_produced_fixture": False,
        },
        "request": {
            "text": request_text,
            "sha256": longmemeval.text_sha256(request_text),
        },
        "candidate_catalog": catalog,
        "catalog_hash": catalog_hash,
        "decision": {
            "status": "resolved",
            "selected_record_id": target["record_id"],
            "alternative_record_ids": [],
            "confidence": None,
            "explanation": (
                "Deterministic fixture selects the bicycle-count update; no "
                "provider model evaluated this request."
            ),
            "selection_rule": "predeclared_exact_bicycle_update_fixture",
            "requires_confirmation": True,
            "deletion_executed": False,
        },
        "confirmation": {
            "selected_record_id": target["record_id"],
            "catalog_hash": catalog_hash,
            "owned_exchange": copy.deepcopy(target_quotes),
            "requires_explicit_click": True,
            "confirmed_in_payload": False,
        },
        "outside_certificate_boundary": True,
    }


def validate_recorded_resolver_fixture(
    fixture: Mapping[str, Any],
    *,
    dialogue: Mapping[str, Any],
    deletion_action: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "schema_version",
        "mode",
        "label",
        "provider",
        "network_request_performed",
        "model_configuration",
        "request",
        "candidate_catalog",
        "catalog_hash",
        "decision",
        "confirmation",
        "outside_certificate_boundary",
    }
    if not isinstance(fixture, Mapping) or set(fixture) != expected_keys:
        raise PayloadError("recorded resolver fixture shape drifted")
    provider = fixture.get("provider") or {}
    model_configuration = fixture.get("model_configuration") or {}
    if (
        fixture.get("schema") != RESOLVER_FIXTURE_SCHEMA
        or fixture.get("schema_version") != 1
        or fixture.get("mode") != "deterministic_recorded_fixture"
        or fixture.get("label") != RESOLVER_FIXTURE_LABEL
        or provider
        != {
            "call_performed": False,
            "artifact_present": False,
            "artifact": None,
        }
        or fixture.get("network_request_performed") is not False
        or model_configuration
        != {
            "default_model_id": DEFAULT_RESOLVER_MODEL,
            "model_produced_fixture": False,
        }
        or fixture.get("outside_certificate_boundary") is not True
    ):
        raise PayloadError("recorded resolver provenance boundary drifted")

    catalog = fixture.get("candidate_catalog")
    if not isinstance(catalog, list) or len(catalog) != 2:
        raise PayloadError("resolver candidate catalog must contain two records")
    record_ids = [
        candidate.get("record_id")
        for candidate in catalog
        if isinstance(candidate, Mapping)
    ]
    if (
        len(record_ids) != 2
        or len(set(record_ids)) != 2
        or any(
            not isinstance(record_id, str)
            or not record_id.startswith("memory_")
            or len(record_id) != len("memory_") + 24
            or any(character not in "0123456789abcdef" for character in record_id[7:])
            for record_id in record_ids
        )
    ):
        raise PayloadError("resolver catalog IDs are not unique opaque IDs")
    if fixture.get("catalog_hash") != payload_sha256(catalog):
        raise PayloadError("resolver catalog hash drifted")

    target_candidates = [
        candidate
        for candidate in catalog
        if candidate.get("owned_exchange_ref")
        == "/case/dialogue/target/quotes"
    ]
    retained_candidates = [
        candidate
        for candidate in catalog
        if candidate.get("owned_exchange_ref")
        == "/case/dialogue/retained/quotes"
    ]
    if len(target_candidates) != 1 or len(retained_candidates) != 1:
        raise PayloadError("resolver catalog exchange bindings drifted")
    target_id = target_candidates[0]["record_id"]
    retained_id = retained_candidates[0]["record_id"]

    request = fixture.get("request") or {}
    request_text = deletion_action.get("display_text")
    if (
        not isinstance(request_text, str)
        or request
        != {
            "text": request_text,
            "sha256": longmemeval.text_sha256(request_text),
        }
    ):
        raise PayloadError("resolver request binding drifted")

    decision = fixture.get("decision")
    if not isinstance(decision, Mapping):
        raise PayloadError("recorded resolver decision is missing")
    selected_id = decision.get("selected_record_id")
    alternative_ids = decision.get("alternative_record_ids")
    returned_ids = [selected_id]
    if isinstance(alternative_ids, list):
        returned_ids.extend(alternative_ids)
    if (
        not all(record_id in record_ids for record_id in returned_ids)
        or len(returned_ids) != len(set(returned_ids))
    ):
        raise PayloadError("recorded resolver decision returned an invalid ID")
    if (
        decision
        != {
            "status": "resolved",
            "selected_record_id": target_id,
            "alternative_record_ids": [],
            "confidence": None,
            "explanation": (
                "Deterministic fixture selects the bicycle-count update; no "
                "provider model evaluated this request."
            ),
            "selection_rule": "predeclared_exact_bicycle_update_fixture",
            "requires_confirmation": True,
            "deletion_executed": False,
        }
        or selected_id == retained_id
    ):
        raise PayloadError("recorded resolver target/retained decision drifted")

    confirmation = fixture.get("confirmation")
    target_quotes = (dialogue.get("target") or {}).get("quotes")
    if (
        not isinstance(target_quotes, list)
        or not isinstance(confirmation, Mapping)
        or confirmation
        != {
            "selected_record_id": target_id,
            "catalog_hash": fixture["catalog_hash"],
            "owned_exchange": target_quotes,
            "requires_explicit_click": True,
            "confirmed_in_payload": False,
        }
    ):
        raise PayloadError("resolver confirmation exchange binding drifted")
    for quote in confirmation["owned_exchange"]:
        if not isinstance(quote, Mapping):
            raise PayloadError("resolver confirmation turn is invalid")
        _validate_quote(quote, origin="public_longmemeval_dialogue")

    expected = _make_recorded_resolver_fixture(dialogue, deletion_action)
    if dict(fixture) != expected:
        raise PayloadError("recorded resolver fixture is not deterministic")


def build_recorded_resolver_fixture(
    dialogue: Mapping[str, Any],
    deletion_action: Mapping[str, Any],
) -> dict[str, Any]:
    fixture = _make_recorded_resolver_fixture(dialogue, deletion_action)
    validate_recorded_resolver_fixture(
        fixture,
        dialogue=dialogue,
        deletion_action=deletion_action,
    )
    return fixture


def build_payload(
    compact: Mapping[str, Any],
    public_dialogue: Mapping[str, Any],
) -> dict[str, Any]:
    state = str(compact.get("status") or "")
    preview = state == "incomplete_case_study_preview"
    case = compact.get("case") or {}
    dialogue = {
        "target": copy.deepcopy(public_dialogue["target"]),
        "retained": copy.deepcopy(public_dialogue["retained"]),
        "intervening": copy.deepcopy(public_dialogue["intervening"]),
    }
    queries = copy.deepcopy(public_dialogue["queries"])
    deletion_action_text = (
        "Please forget the update where I said I now own four bikes. "
        "Keep the rest of our conversation, including the Speyer contact details."
    )
    deletion_action = {
        "origin": "evaluator_added_deletion_action",
        "model_generated": False,
        "public_longmemeval_source": False,
        "out_of_band": True,
        "display_label": (
            "EVALUATION DELETION ACTION "
            "(added by us; not LongMemEval source)"
        ),
        "display_text": deletion_action_text,
        "display_text_sha256": longmemeval.text_sha256(
            deletion_action_text
        ),
    }
    memory_resolution = build_recorded_resolver_fixture(
        dialogue,
        deletion_action,
    )
    payload: dict[str, Any] = {
        "schema": PAYLOAD_SCHEMA,
        "schema_version": 2,
        "status": state,
        "preview_label": PREVIEW_LABEL if preview else None,
        "title": "Measured forgetting after a long conversation",
        "subtitle": (
            "One predeclared benchmark-assembled case; measured Gemma 3 4B-IT "
            "target and retained-control outcomes"
        ),
        "contains_source_text": True,
        "contains_unquoted_source_text": False,
        "contains_model_generated_text": False,
        "raw_context_text_present": False,
        "official_longmemeval_leaderboard_score": False,
        "scope": {
            "case_study_only": True,
            "aggregate_claims": [],
            "aggregate_claims_allowed": False,
            "final_all16_complete": compact["report_state"]["all16_complete"],
            "certificate_reference": "fixed-C retained-key refit",
            "behavioral_reference": "fresh raw round-omitted repack",
            "full_repack_certificate_claimed": False,
        },
        "provenance": {
            "compact_report_integrity_sha256": compact["integrity"]["sha256"],
            "source_artifacts": copy.deepcopy(compact["source_artifacts"]),
            "public_dialogue": {
                "dataset_id": longmemeval.DATASET_ID,
                "dataset_revision": longmemeval.DATASET_REVISION,
                "artifact_path": longmemeval.DATASET_ARTIFACT_PATH,
                "artifact_sha256": longmemeval.DATASET_ARTIFACT_SHA256,
                "license": longmemeval.DATASET_LICENSE,
                "quote_policy": (
                    "only exact complete source sentences or lines from the "
                    "displayed target, retained, and intervening examples are "
                    "copied; hashes and source spans bind every quote"
                ),
                "assembly_disclosure": (
                    "target, retained-control, and gap excerpts come from separate "
                    "public LongMemEval examples selected and assembled by the "
                    "benchmark; they are not one organic continuous conversation"
                ),
            },
        },
        "case": {
            "record_id": CASE_RECORD_ID,
            "selection": copy.deepcopy(case["selection"]),
            "dialogue": dialogue,
            "queries": queries,
            "deletion_action": deletion_action,
            "memory_resolution": memory_resolution,
            "chronology": {
                "tokens_strictly_after_owned": copy.deepcopy(
                    case["geometry"]["tokens_strictly_after_owned"]
                ),
                "minimum_tokens_strictly_after_owned": copy.deepcopy(
                    case["geometry"]["minimum_tokens_strictly_after_owned"]
                ),
                "local_window_tokens": copy.deepcopy(
                    case["geometry"]["local_window_tokens"]
                ),
                "owned_round_outside_local_window": copy.deepcopy(
                    case["geometry"]["owned_round_outside_local_window"]
                ),
                "display_note": (
                    "The retained control, deterministic tail, and three registered "
                    "suffix rounds form the measured gap; one suffix round is "
                    "displayed as a source-bound excerpt."
                ),
            },
            "outcomes": copy.deepcopy(case["outcomes"]),
            "certificate": copy.deepcopy(case["certificate"]),
        },
        "demo": {
            "mode": "recorded_replay",
            "live_compute": False,
            "network_api_required": False,
            "model_outcome_text_present": False,
            "sequence": [
                {
                    "id": "chat",
                    "label": "Load public target and retained turns",
                    "source_ref": "/case/dialogue",
                },
                {
                    "id": "recall",
                    "label": "Recall before forgetting",
                    "source_ref": "/case/outcomes/0",
                },
                {
                    "id": "resolve-request",
                    "label": "State a natural-language deletion request",
                    "source_ref": "/case/memory_resolution/request",
                },
                {
                    "id": "resolve-proposal",
                    "label": "Show the deterministic resolver fixture",
                    "source_ref": "/case/memory_resolution/decision",
                },
                {
                    "id": "confirm",
                    "label": "Confirm the complete owned exchange",
                    "source_ref": "/case/memory_resolution/confirmation",
                },
                {
                    "id": "forget",
                    "label": "Apply the confirmed recorded deletion",
                    "source_ref": "/case/outcomes/2/execution",
                },
                {
                    "id": "re-query",
                    "label": "Re-query the target",
                    "source_ref": "/case/outcomes/2",
                },
                {
                    "id": "retained",
                    "label": "Probe the retained control",
                    "source_ref": "/case/outcomes/2/retained",
                },
                {
                    "id": "certificate",
                    "label": "Show the exact/refit certificate",
                    "source_ref": "/case/certificate",
                },
            ],
        },
    }
    payload["integrity"] = {
        "algorithm": "sha256",
        "sha256": payload_sha256(payload),
    }
    validate_payload(payload, compact=compact)
    return payload


def validate_payload(
    payload: Mapping[str, Any],
    *,
    compact: Mapping[str, Any] | None = None,
) -> None:
    integrity = payload.get("integrity")
    unsigned = dict(payload)
    unsigned.pop("integrity", None)
    if (
        payload.get("schema") != PAYLOAD_SCHEMA
        or payload.get("schema_version") != 2
        or not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != payload_sha256(unsigned)
    ):
        raise PayloadError("forgetting payload integrity or schema drifted")
    keys = set(_walk_keys(payload))
    leaked_keys = sorted(keys & FORBIDDEN_PAYLOAD_KEYS)
    if leaked_keys:
        raise PayloadError(
            "payload contains forbidden raw/model text keys: "
            + ", ".join(leaked_keys)
        )
    if (
        payload.get("contains_unquoted_source_text") is not False
        or payload.get("contains_model_generated_text") is not False
        or payload.get("raw_context_text_present") is not False
        or (payload.get("demo") or {}).get("model_outcome_text_present") is not False
    ):
        raise PayloadError("payload text-boundary disclosure drifted")
    scope = payload.get("scope") or {}
    if (
        not isinstance(scope.get("aggregate_claims"), list)
        or scope["aggregate_claims"]
        or scope.get("aggregate_claims_allowed") is not False
    ):
        raise PayloadError("case-study payload must not contain aggregate claims")
    if payload.get("status") == "incomplete_case_study_preview" and (
        payload.get("preview_label") != PREVIEW_LABEL
        or scope.get("final_all16_complete") is not False
    ):
        raise PayloadError("partial15 payload lacks the mandatory preview label")

    case = payload.get("case") or {}
    public_provenance = (payload.get("provenance") or {}).get(
        "public_dialogue"
    ) or {}
    if public_provenance.get("assembly_disclosure") != (
        "target, retained-control, and gap excerpts come from separate public "
        "LongMemEval examples selected and assembled by the benchmark; they are "
        "not one organic continuous conversation"
    ):
        raise PayloadError("benchmark assembly provenance disclosure drifted")
    for section, (expected_count, expected_roles) in EXPECTED_DIALOGUE_SHAPE.items():
        quotes = ((case.get("dialogue") or {}).get(section) or {}).get("quotes")
        if (
            not isinstance(quotes, list)
            or len(quotes) != expected_count
            or tuple(str(quote.get("role") or "") for quote in quotes)
            != expected_roles
        ):
            raise PayloadError(f"payload {section} public dialogue shape drifted")
        for quote in quotes:
            if not isinstance(quote, Mapping):
                raise PayloadError("public dialogue quote is not an object")
            _validate_quote(quote, origin="public_longmemeval_dialogue")
        observed_bindings = tuple(
            (
                quote["source_turn_index"],
                quote["source_turn_sha256"],
                quote["display_text_sha256"],
            )
            for quote in quotes
        )
        if observed_bindings != EXPECTED_DIALOGUE_BINDINGS[section]:
            raise PayloadError(f"payload {section} public excerpt binding drifted")
    for query in (case.get("queries") or {}).values():
        if not isinstance(query, Mapping):
            raise PayloadError("public query quote is not an object")
        _validate_quote(query, origin="public_longmemeval_query")
        if (
            query.get("official_longmemeval_question") is not True
            or query.get("evaluation_use")
            != "official question replayed by evaluator"
            or query.get("evaluator_added_text") is not False
        ):
            raise PayloadError("registered query provenance drifted")
    deletion_action = case.get("deletion_action") or {}
    if (
        deletion_action.get("origin") != "evaluator_added_deletion_action"
        or deletion_action.get("model_generated") is not False
        or deletion_action.get("public_longmemeval_source") is not False
        or deletion_action.get("out_of_band") is not True
        or deletion_action.get("display_label")
        != "EVALUATION DELETION ACTION (added by us; not LongMemEval source)"
        or deletion_action.get("display_text_sha256")
        != longmemeval.text_sha256(
            str(deletion_action.get("display_text") or "")
        )
    ):
        raise PayloadError("evaluation deletion action provenance drifted")
    memory_resolution = case.get("memory_resolution")
    if not isinstance(memory_resolution, Mapping):
        raise PayloadError("recorded memory resolution fixture is missing")
    validate_recorded_resolver_fixture(
        memory_resolution,
        dialogue=case.get("dialogue") or {},
        deletion_action=deletion_action,
    )

    sequence = (payload.get("demo") or {}).get("sequence")
    if (
        not isinstance(sequence, list)
        or tuple(str(step.get("id") or "") for step in sequence)
        != EXPECTED_DEMO_SEQUENCE
    ):
        raise PayloadError("recorded replay sequence drifted")
    _validate_bound_values(case.get("outcomes"), compact or {})
    _validate_bound_values(case.get("certificate"), compact or {})
    _validate_bound_values(case.get("chronology"), compact or {})
    if compact is not None:
        if payload["provenance"]["compact_report_integrity_sha256"] != compact[
            "integrity"
        ]["sha256"]:
            raise PayloadError("payload compact-report binding drifted")
        compact_case = compact["case"]
        if (
            case.get("outcomes") != compact_case.get("outcomes")
            or case.get("certificate") != compact_case.get("certificate")
            or payload["scope"]["final_all16_complete"]
            != compact["report_state"]["all16_complete"]
        ):
            raise PayloadError("payload model outcomes differ from compact sources")


def _load_bound_inputs(
    *,
    manifest_path: str | Path,
    policy_path: str | Path,
    census_path: str | Path,
    admission_report_path: str | Path,
    method_lock_path: str | Path,
    partial_lock_path: str | Path,
    completion_lock_path: str | Path | None,
    method_report_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    _require_file_hash(
        manifest_path,
        MANIFEST_FILE_SHA256,
        name="geometry manifest",
    )
    _require_file_hash(policy_path, POLICY_FILE_SHA256, name="geometry policy")
    _require_file_hash(census_path, CENSUS_FILE_SHA256, name="geometry census")
    _require_file_hash(
        admission_report_path,
        ADMISSION_REPORT_FILE_SHA256,
        name="admission report",
    )
    _require_file_hash(
        method_lock_path,
        METHOD_LOCK_FILE_SHA256,
        name="method lock",
    )
    manifest = _load_mapping(manifest_path, name="geometry manifest")
    policy = _load_mapping(policy_path, name="geometry policy")
    census = _load_mapping(census_path, name="geometry census")
    admission = _load_mapping(admission_report_path, name="admission report")
    method_lock = _load_mapping(method_lock_path, name="method lock")
    partial_lock = _load_mapping(partial_lock_path, name="partial lock")
    completion_lock = (
        None
        if completion_lock_path is None
        else _load_mapping(completion_lock_path, name="completion lock")
    )
    method_report = _load_mapping(method_report_path, name="method report")
    report_hash = file_sha256(method_report_path)
    _validate_artifacts(
        manifest=manifest,
        policy=policy,
        census=census,
        admission_report=admission,
        method_lock=method_lock,
        partial_lock=partial_lock,
        completion_lock=completion_lock,
    )
    return (
        manifest,
        policy,
        census,
        admission,
        method_lock,
        partial_lock,
        completion_lock,
        method_report,
        report_hash,
    )


def build_from_paths(
    *,
    method_report_path: str | Path = DEFAULT_METHOD_REPORT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    policy_path: str | Path = DEFAULT_POLICY,
    census_path: str | Path = DEFAULT_CENSUS,
    admission_report_path: str | Path = DEFAULT_ADMISSION_REPORT,
    method_lock_path: str | Path = DEFAULT_METHOD_LOCK,
    partial_lock_path: str | Path = DEFAULT_PARTIAL_LOCK,
    completion_lock_path: str | Path | None = None,
    data_path: str | Path | None = None,
    compact_input_path: str | Path | None = None,
    allow_incomplete_preview: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        manifest,
        _policy,
        _census,
        admission,
        method_lock,
        partial_lock,
        completion_lock,
        method_report,
        report_hash,
    ) = _load_bound_inputs(
        manifest_path=manifest_path,
        policy_path=policy_path,
        census_path=census_path,
        admission_report_path=admission_report_path,
        method_lock_path=method_lock_path,
        partial_lock_path=partial_lock_path,
        completion_lock_path=completion_lock_path,
        method_report_path=method_report_path,
    )
    expected_compact = build_compact_case_report(
        method_report,
        admission,
        manifest,
        method_lock,
        partial_lock,
        completion_lock,
        report_file_sha256=report_hash,
        allow_incomplete_preview=allow_incomplete_preview,
    )
    if compact_input_path is None:
        compact = expected_compact
    else:
        compact = _load_mapping(compact_input_path, name="compact case report")
        if compact != expected_compact:
            raise PayloadError(
                "compact input does not reproduce from the bound method report"
            )
        validate_compact_case_report(
            compact,
            report=method_report,
            admission_report=admission,
            manifest=manifest,
            method_lock=method_lock,
            partial_lock=partial_lock,
            completion_lock=completion_lock,
            report_file_sha256=report_hash,
            allow_incomplete_preview=allow_incomplete_preview,
        )
    rows = longmemeval.load_pinned_longmemeval_rows(data_path)
    dialogue = build_public_dialogue(manifest, rows)
    payload = build_payload(compact, dialogue)
    return compact, payload


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-report", default=str(DEFAULT_METHOD_REPORT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--policy-lock", default=str(DEFAULT_POLICY))
    parser.add_argument("--census", default=str(DEFAULT_CENSUS))
    parser.add_argument("--admission-report", default=str(DEFAULT_ADMISSION_REPORT))
    parser.add_argument("--method-lock", default=str(DEFAULT_METHOD_LOCK))
    parser.add_argument("--partial-lock", default=str(DEFAULT_PARTIAL_LOCK))
    parser.add_argument(
        "--completion-lock",
        help=(
            "required only for a final report produced by the separate "
            "completion-lock path"
        ),
    )
    parser.add_argument("--data-path")
    parser.add_argument("--compact-input")
    parser.add_argument("--compact-out", default=str(DEFAULT_COMPACT_OUTPUT))
    parser.add_argument("--payload-out", default=str(DEFAULT_PAYLOAD_OUTPUT))
    parser.add_argument(
        "--allow-incomplete-preview",
        action="store_true",
        help="accept only the exact committed partial15 report and mark it preview",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        compact, payload = build_from_paths(
            method_report_path=args.method_report,
            manifest_path=args.manifest,
            policy_path=args.policy_lock,
            census_path=args.census,
            admission_report_path=args.admission_report,
            method_lock_path=args.method_lock,
            partial_lock_path=args.partial_lock,
            completion_lock_path=args.completion_lock,
            data_path=args.data_path,
            compact_input_path=args.compact_input,
            allow_incomplete_preview=args.allow_incomplete_preview,
        )
        _atomic_write_json(args.compact_out, compact)
        _atomic_write_json(args.payload_out, payload)
    except (OSError, PermissionError, PayloadError) as exc:
        raise SystemExit(f"payload build failed: {exc}") from exc
    print(f"wrote compact case report to {Path(args.compact_out)}")
    print(f"wrote recorded figure/demo payload to {Path(args.payload_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

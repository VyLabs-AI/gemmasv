"""Admission evaluator for the fixed-cohort LongMemEval geometry correction.

The frozen geometry manifest preserves the ordered 16 records from the
previously admission-scored v1 confirmation cohort.  It discloses that the
geometry correction followed a local-window implementation failure and
preceded every valid deletion-method outcome.  It changes no core source,
question, answer, retained identity, or probe.

Scoring requires an exact committed manifest/policy/census triplet and the
literal boolean ``True`` acknowledgement.  Reports are source-free.  Resume
validates but discards prior rows and re-executes all 16 records.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_v2 as hardened
from gemma_sv import longmemeval_chat_geometry_fix as benchmark


SCHEMA = "gemma-sv-longmemeval-chat-geometry-admission-evaluation-v1"
SCHEMA_VERSION = 1
ARM = hardened.ARM
EXPECTED_RECORDS = 16
WINDOW = 1_024

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_MANIFEST = (
    BENCHMARKS / "longmemeval_chat_geometry_corrected_v1.json"
)
DEFAULT_POLICY_LOCK = (
    BENCHMARKS / "longmemeval_chat_geometry_policy_lock_v1.json"
)
DEFAULT_CENSUS = BENCHMARKS / "longmemeval_chat_geometry_census_v1.json"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_geometry_admission_v1.json"
)
PINNED_MANIFEST_FILE_SHA256 = (
    "df10404a5b11491791638190f61b52e30e667d17601819a99925181885bde480"
)
PINNED_POLICY_LOCK_FILE_SHA256 = (
    "bccb7d716182c495892f1ddb1693e4848995bc7da193cce26b9395fe2a736331"
)
PINNED_CENSUS_FILE_SHA256 = (
    "1d8ef04097cccf69021f56e79da8cf46e3e669967b5e7534edb4baa92f332d3b"
)

DISCLOSURE = {
    "original_admission_model_scored": True,
    "geometry_correction_after_local_window_implementation_failure": True,
    "geometry_correction_before_valid_deletion_method_outcomes": True,
    "original_admission_not_reinterpreted_as_valid_deletion_outcome": True,
}

_IMPLEMENTATION_PATHS = {
    **hardened._IMPLEMENTATION_PATHS,
    "eval_longmemeval_chat_geometry_fix.py": Path(__file__).resolve(),
    "longmemeval_chat_geometry_fix.py": PACKAGE
    / "longmemeval_chat_geometry_fix.py",
}
_PROTECTED_INPUTS = frozenset(
    {
        DEFAULT_MANIFEST,
        DEFAULT_POLICY_LOCK,
        DEFAULT_CENSUS,
        benchmark.DEFAULT_CORE_MANIFEST_PATH,
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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require_file_hash(path: str | Path, expected: str, *, name: str) -> None:
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} is not the exact committed artifact")


def _assert_source_free(payload: Mapping[str, Any]) -> None:
    hardened._assert_source_free(payload)


def _verify_integrity(
    payload: Mapping[str, Any],
    *,
    integrity_key: str,
    name: str,
) -> None:
    observed = payload.get(integrity_key)
    body = copy.deepcopy(dict(payload))
    body.pop(integrity_key, None)
    if (
        not isinstance(observed, Mapping)
        or observed.get("algorithm") != "sha256"
        or observed.get("sha256") != _payload_sha256(body)
    ):
        raise ValueError(f"{name} integrity differs")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_output_path(
    output: str | Path,
    *,
    manifest_path: str | Path,
    policy_lock_path: str | Path,
    census_path: str | Path,
    core_manifest_path: str | Path | None = None,
    data_path: str | Path | None = None,
) -> Path:
    output_path = Path(output).expanduser()
    resolved = _resolved(output_path)
    protected = {
        Path(path).expanduser()
        for path in (
            manifest_path,
            policy_lock_path,
            census_path,
            *(
                ()
                if core_manifest_path is None
                else (core_manifest_path,)
            ),
            *((() if data_path is None else (data_path,))),
            *_PROTECTED_INPUTS,
        )
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
            raise ValueError("output aliases a protected committed input")
    return resolved


def locked_admission_policy(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    census: Mapping[str, Any],
) -> dict[str, Any]:
    policy = hardened._validate_threshold_mapping(
        policy_lock.get("admission_policy"),
        name="geometry policy lock",
    )
    manifest_policy = hardened._validate_threshold_mapping(
        manifest.get("admission_policy"),
        name="geometry manifest",
    )
    census_policy = hardened._validate_threshold_mapping(
        census.get("admission_policy"),
        name="geometry census",
    )
    if not (policy == manifest_policy == census_policy):
        raise ValueError("geometry admission policies differ")
    if census.get("admission_policy_sha256") != _payload_sha256(policy):
        raise ValueError("geometry census admission policy hash differs")
    return policy


def _validate_record(
    record: Mapping[str, Any],
    runtime_record: Any | None = None,
) -> dict[str, Any]:
    integrity = record.get("record_integrity")
    body = copy.deepcopy(dict(record))
    body.pop("record_integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(body)
    ):
        raise ValueError("geometry record integrity differs")
    binding = record.get("core_binding") or {}
    probes = record.get("probes") or ()
    target = record.get("target") or {}
    retained = record.get("retained_probe") or {}
    core_spec = record.get("core_spec") or {}
    context = record.get("context") or {}
    local = context.get("local_window_safety") or {}
    if (
        record.get("record_id") != binding.get("record_id")
        or target.get("source_id") != binding.get("target_source_id")
        or target.get("question_id") != binding.get("target_question_id")
        or retained.get("source_id") != binding.get("retained_source_id")
        or retained.get("question_id") != binding.get("retained_question_id")
        or _payload_sha256(probes) != binding.get("probe_descriptors_sha256")
        or len(probes) != int(binding.get("probe_descriptor_count", -1))
        or sorted(target) != binding.get("target_descriptor_keys")
        or _payload_sha256(target) != binding.get("target_descriptor_sha256")
        or sorted(retained) != binding.get("retained_descriptor_keys")
        or _payload_sha256(retained)
        != binding.get("retained_descriptor_sha256")
        or sorted(core_spec) != binding.get("core_spec_keys")
        or _payload_sha256(core_spec) != binding.get("core_spec_sha256")
        or int(context.get("tokens_strictly_after_owned", -1))
        < benchmark.MINIMUM_TOKENS_AFTER_OWNED
        or int(context.get("runtime_total_token_bound", -1))
        > benchmark.MAXIMUM_CONTEXT_TOKENS
        or local.get("selected_span_inside_local_window") is not False
        or int(local.get("true_local_window_tokens", -1)) != WINDOW
        or context.get("fixed_c_reference", {}).get(
            "all_affected_boundaries_feasible"
        )
        is not True
    ):
        raise ValueError("geometry record or frozen core binding differs")
    if runtime_record is not None:
        runtime_context = runtime_record.context
        original_ids = tuple(int(value) for value in runtime_context.original_token_ids)
        edited_ids = tuple(int(value) for value in runtime_context.edited_token_ids)
        raw_ids = tuple(int(value) for value in runtime_record.raw_omitted_token_ids)
        ownership = context.get("ownership") or {}
        retained_context = context.get("retained") or {}
        expected_ranges = tuple(
            (int(item["start"]), int(item["end"]))
            for item in ownership.get("deletion_ranges") or ()
        )
        runtime_probes = [
            benchmark.chat_v1._probe_descriptor(probe)
            for probe in runtime_record.probes
        ]
        target_runtime = runtime_record.target
        retained_runtime = runtime_record.retained
        observed_after = (
            len(original_ids)
            - max(runtime_context.forget_positions)
            - 1
        )
        retained_before = tuple(
            original_ids[position]
            for position in runtime_context.retained_positions
        )
        retained_after = tuple(
            edited_ids[position]
            for position in runtime_context.edited_retained_positions
        )
        fixed_c = context.get("fixed_c_reference") or {}
        runtime_feasibility = [
            dict(item) for item in runtime_record.fixed_c_diagnostics
        ]
        runtime_bound = len(original_ids) + int(runtime_record.query_token_reserve)
        if (
            runtime_record.record_id != record.get("record_id")
            or runtime_record.partition != "confirmation"
            or len(original_ids) != int(context["original_token_count"])
            or benchmark.base.token_ids_sha256(original_ids)
            != context.get("original_token_ids_sha256")
            or benchmark.base.text_sha256(runtime_context.original_text)
            != context.get("original_text_sha256")
            or len(edited_ids) != int(context["raw_omitted_token_count"])
            or len(raw_ids) != int(context["raw_omitted_token_count"])
            or edited_ids != raw_ids
            or benchmark.base.token_ids_sha256(edited_ids)
            != context.get("raw_omitted_token_ids_sha256")
            or benchmark.base.token_ids_sha256(raw_ids)
            != context.get("raw_omitted_token_ids_sha256")
            or benchmark.base.text_sha256(runtime_context.edited_text)
            != context.get("raw_omitted_text_sha256")
            or tuple(runtime_context.forget_positions)
            != tuple(int(value) for value in ownership.get("forget_positions") or ())
            or tuple(runtime_context.deletion_ranges) != expected_ranges
            or list(runtime_context.owned_character_span)
            != ownership.get("owned_character_span")
            or list(runtime_context.answer_character_span)
            != ownership.get("answer_character_span")
            or list(runtime_context.retained_character_span)
            != retained_context.get("character_span")
            or benchmark.chat_v1._payload_sha256(
                [list(offset) for offset in runtime_context.offset_mapping]
            )
            != ownership.get("offset_mapping_sha256")
            or list(runtime_context.retained_positions)
            != retained_context.get("token_positions")
            or list(runtime_context.edited_retained_positions)
            != retained_context.get("edited_token_positions")
            or benchmark.base.token_ids_sha256(retained_before)
            != retained_context.get("original_token_ids_sha256")
            or benchmark.base.token_ids_sha256(retained_after)
            != retained_context.get("edited_token_ids_sha256")
            or retained_before != retained_after
            or runtime_probes != list(probes)
            or target_runtime is None
            or retained_runtime is None
            or target_runtime.source_id != binding.get("target_source_id")
            or target_runtime.question_id != binding.get("target_question_id")
            or benchmark.base.text_sha256(target_runtime.question)
            != target.get("question_sha256")
            or benchmark.base.text_sha256(target_runtime.answer)
            != target.get("current_answer_sha256")
            or retained_runtime.source_id != binding.get("retained_source_id")
            or retained_runtime.question_id != binding.get("retained_question_id")
            or benchmark.base.text_sha256(retained_runtime.question)
            != retained.get("question_sha256")
            or benchmark.base.text_sha256(retained_runtime.answer)
            != retained.get("current_answer_sha256")
            or int(runtime_record.query_token_reserve)
            != int(context.get("query_token_reserve", -1))
            or runtime_bound != int(context.get("runtime_total_token_bound", -1))
            or runtime_bound > benchmark.MAXIMUM_CONTEXT_TOKENS
            or observed_after != int(context["tokens_strictly_after_owned"])
            or observed_after < benchmark.MINIMUM_TOKENS_AFTER_OWNED
            or runtime_feasibility != fixed_c.get("feasibility")
            or not runtime_feasibility
            or not all(bool(item.get("feasible")) for item in runtime_feasibility)
            or len(runtime_feasibility)
            != int(fixed_c.get("affected_boundaries", -1))
            or fixed_c.get("all_affected_boundaries_feasible") is not True
        ):
            raise ValueError("rehydrated geometry record differs from frozen record")
    return {
        "true_local_window_tokens": WINDOW,
        "minimum_tokens_strictly_after_owned": (
            benchmark.MINIMUM_TOKENS_AFTER_OWNED
        ),
        "observed_tokens_strictly_after_owned": int(
            context["tokens_strictly_after_owned"]
        ),
        "owned_round_strictly_outside_local_window_before_query": True,
        "fixed_c_feasible": True,
        "selected_span_inside_local_window": False,
    }


def validate_locked_inputs(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    census: Mapping[str, Any],
) -> None:
    _assert_source_free(manifest)
    _assert_source_free(policy_lock)
    _assert_source_free(census)
    if (
        manifest.get("schema") != benchmark.SCHEMA
        or manifest.get("schema_version") != benchmark.SCHEMA_VERSION
        or manifest.get("partition") != "confirmation"
        or manifest.get("contains_source_text") is not False
    ):
        raise ValueError("unsupported geometry-corrected manifest")
    _verify_integrity(
        manifest,
        integrity_key="integrity",
        name="geometry manifest",
    )
    lock_body = copy.deepcopy(dict(policy_lock))
    lock_sha = lock_body.pop("lock_sha256", None)
    if (
        lock_sha != _payload_sha256(lock_body)
        or policy_lock.get("schema") != benchmark.POLICY_SCHEMA
        or policy_lock.get("status")
        != "frozen-before-geometry-corrected-model-scoring"
        or manifest.get("policy_lock") != policy_lock
    ):
        raise ValueError("geometry policy lock differs")
    _verify_integrity(census, integrity_key="integrity", name="geometry census")
    if (
        manifest.get("integrity", {}).get("sha256")
        != benchmark.PINNED_MANIFEST_INTEGRITY_SHA256
        or policy_lock.get("lock_sha256")
        != benchmark.PINNED_POLICY_LOCK_SHA256
        or census.get("integrity", {}).get("sha256")
        != benchmark.PINNED_CENSUS_INTEGRITY_SHA256
    ):
        raise ValueError("geometry committed-target binding differs")
    if (
        manifest.get("disclosure") != DISCLOSURE
        or policy_lock.get("disclosure") != DISCLOSURE
        or policy_lock.get("selection_uses_model_outputs") is not False
    ):
        raise ValueError("geometry correction disclosure differs")
    core = policy_lock.get("core_cohort") or {}
    if (
        core.get("source_manifest_file_sha256")
        != benchmark.PINNED_CORE_FILE_SHA256
        or core.get("source_manifest_integrity_sha256")
        != benchmark.PINNED_CORE_INTEGRITY_SHA256
        or core.get("ordered_records") != EXPECTED_RECORDS
        or core.get("record_question_target_retained_replacement") is not False
        or core.get("probe_replacement") is not False
    ):
        raise ValueError("original v1 core cohort binding differs")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_RECORDS:
        raise ValueError("geometry evaluator requires the frozen all16 cohort")
    bindings = []
    for record in records:
        _validate_record(record)
        bindings.append(record["core_binding"])
    record_ids = [str(record["record_id"]) for record in records]
    if (
        len(set(record_ids)) != EXPECTED_RECORDS
        or _payload_sha256(bindings)
        != core.get("ordered_core_bindings_sha256")
        or census.get("ordered_core_record_ids") != record_ids
        or census.get("ordered_core_bindings_sha256")
        != core.get("ordered_core_bindings_sha256")
        or census.get("core_records") != EXPECTED_RECORDS
        or census.get("fixed_c_feasible_records") != EXPECTED_RECORDS
        or census.get("manifest_integrity_sha256")
        != manifest["integrity"]["sha256"]
        or census.get("policy_lock_sha256") != policy_lock["lock_sha256"]
        or census.get("selection_uses_model_outputs") is not False
    ):
        raise ValueError("geometry census or ordered all16 binding differs")
    geometry = policy_lock.get("geometry_contract") or {}
    if (
        geometry.get("window") != WINDOW
        or geometry.get("minimum_tokens_strictly_after_owned")
        != benchmark.MINIMUM_TOKENS_AFTER_OWNED
        or geometry.get("maximum_context_tokens")
        != benchmark.MAXIMUM_CONTEXT_TOKENS
        or geometry.get("nu") != hardened.NU
        or geometry.get("chunk") != hardened.CHUNK
        or geometry.get("solver_seed") != hardened.SOLVER_SEED
        or geometry.get("fixed_c_required_for_every_record") is not True
        or geometry.get("extension_selection_source_only") is not True
    ):
        raise ValueError("geometry execution contract differs")
    locked_admission_policy(manifest, policy_lock, census)


def load_locked_inputs(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    policy_lock_path: str | Path = DEFAULT_POLICY_LOCK,
    census_path: str | Path = DEFAULT_CENSUS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_file_hash(
        manifest_path,
        PINNED_MANIFEST_FILE_SHA256,
        name="geometry manifest",
    )
    _require_file_hash(
        policy_lock_path,
        PINNED_POLICY_LOCK_FILE_SHA256,
        name="geometry policy lock",
    )
    _require_file_hash(
        census_path,
        PINNED_CENSUS_FILE_SHA256,
        name="geometry census",
    )
    manifest = _load_mapping(manifest_path, name="geometry manifest")
    policy = _load_mapping(policy_lock_path, name="geometry policy")
    census = _load_mapping(census_path, name="geometry census")
    validate_locked_inputs(manifest, policy, census)
    return manifest, policy, census


runtime_contract = hardened.runtime_contract
_runtime_config_kwargs = hardened._runtime_config_kwargs
_make_runtime = hardened._make_runtime
verify_loaded_runtime = hardened.verify_loaded_runtime
summarize_records = hardened.summarize_records


def authorize_confirmation(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    explicit_acknowledgement: Any,
    requested_arm: str | None = None,
    device: str,
) -> tuple[str, str]:
    validate_locked_inputs(manifest, policy_lock, census)
    if requested_arm is not None:
        raise ValueError("geometry confirmation forbids arm overrides")
    if explicit_acknowledgement is not True:
        raise PermissionError("geometry scoring acknowledgement must be exactly True")
    runtime_contract(device=device)
    return ARM, str(device).strip()


def evaluate_admission_record(
    runtime: Any,
    record: Any,
    manifest_record: Mapping[str, Any],
    *,
    warmup: int,
    repeats: int,
    admission_policy: Mapping[str, Any],
) -> dict[str, Any]:
    local = _validate_record(manifest_record, record)
    row = hardened.v1_eval.evaluate_admission_record(
        runtime,
        record,
        warmup=warmup,
        repeats=repeats,
    )
    row["admission"] = hardened._policy_admission_decomposition(
        row["references"]["present"]["scores"],
        row["references"]["fresh_raw_omission"]["scores"],
        admission_policy,
    )
    row["local_window_safety"] = local
    row["geometry_safety"] = copy.deepcopy(local)
    row["disclosure"] = copy.deepcopy(DISCLOSURE)
    _assert_source_free(row)
    return row


def _implementation_fingerprints() -> dict[str, str]:
    values = {
        name: _sha256_file(path)
        for name, path in sorted(_IMPLEMENTATION_PATHS.items())
    }
    values["contract_sha256"] = _payload_sha256(values)
    return values


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _base_report(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    manifest_path: Path,
    policy_lock_path: Path,
    census_path: Path,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    ids = [str(record["record_id"]) for record in manifest["records"]]
    admission = locked_admission_policy(manifest, policy, census)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "evaluation": "LongMemEval fixed-cohort geometry-corrected admission",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "official_longmemeval_leaderboard_score": False,
        "model_scoring_used_for_selection_or_replacement": False,
        "disclosure": copy.deepcopy(DISCLOSURE),
        "core_cohort": {
            **copy.deepcopy(policy["core_cohort"]),
            "same_ordered_original_v1_core_records": True,
            "source_question_answer_retained_or_probe_changes": False,
        },
        "manifest": {
            "schema": manifest["schema"],
            "integrity_sha256": manifest["integrity"]["sha256"],
            "file_sha256": _sha256_file(manifest_path),
            "frozen_records": EXPECTED_RECORDS,
            "selected_record_ids": ids,
            "selected_record_ids_sha256": _payload_sha256(ids),
        },
        "policy_lock": {
            "schema": policy["schema"],
            "status": policy["status"],
            "lock_sha256": policy["lock_sha256"],
            "file_sha256": _sha256_file(policy_lock_path),
        },
        "census": {
            "schema": census["schema"],
            "integrity_sha256": census["integrity"]["sha256"],
            "file_sha256": _sha256_file(census_path),
            "core_records": census["core_records"],
            "fixed_c_feasible_records": census["fixed_c_feasible_records"],
        },
        "config": {
            **runtime_contract(device=device),
            "maximum_context_tokens": benchmark.MAXIMUM_CONTEXT_TOKENS,
            "minimum_tokens_strictly_after_owned": (
                benchmark.MINIMUM_TOKENS_AFTER_OWNED
            ),
            "admission_policy": admission,
            "warmup": int(warmup),
            "repeats": int(repeats),
        },
        "admission_protocol": {
            "full_target_sequences_scored": True,
            "present_vs_fresh_raw_omission": True,
            "admission_policy": admission,
            "all16_required": True,
            "no_replacement": True,
        },
        "resume_protocol": {
            "existing_report_policy": "validate_then_discard_all_rows",
            "completed_rows_reused": False,
            "all_authorized_records_reexecuted": True,
        },
        "implementation": _implementation_fingerprints(),
        "environment": _environment(),
        "records": [],
        "summary": summarize_records(
            [],
            frozen_records=EXPECTED_RECORDS,
            admission_thresholds=admission,
        ),
    }


def _resume_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "status",
        "records",
        "summary",
        "elapsed_seconds_this_process",
        "resume_requested",
        "discarded_resume_rows",
        "reused_completed_resume_records",
    }
    return {key: value for key, value in report.items() if key not in excluded}


def _validate_resume(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> int:
    _assert_source_free(report)
    hardened._assert_finite_json(report)
    if _resume_signature(report) != _resume_signature(expected):
        raise ValueError("geometry resume report contract differs")
    rows = report.get("records")
    if not isinstance(rows, list):
        raise ValueError("geometry resume rows are invalid")
    expected_ids = expected["manifest"]["selected_record_ids"]
    ids = [
        str(row.get("record_id") or "")
        for row in rows
        if isinstance(row, Mapping)
    ]
    if len(ids) != len(rows) or ids != expected_ids[: len(ids)]:
        raise ValueError("geometry resume row order differs")
    return len(rows)


def _failed_row(record: Any, exc: Exception) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "partition": record.partition,
        "status": "failed",
        "contains_source_text": False,
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": benchmark.base.text_sha256(str(exc)),
    }


def run_records(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    census: Mapping[str, Any],
    records: Sequence[Any],
    runtime: Any,
    *,
    manifest_path: Path,
    policy_lock_path: Path,
    census_path: Path,
    output: Path,
    device: str,
    warmup: int,
    repeats: int,
    confirmation_scoring_acknowledged: Any,
    core_manifest_path: Path | None = None,
    data_path: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    output = _validate_output_path(
        output,
        manifest_path=manifest_path,
        policy_lock_path=policy_lock_path,
        census_path=census_path,
        core_manifest_path=core_manifest_path,
        data_path=data_path,
    )
    _require_file_hash(
        manifest_path,
        PINNED_MANIFEST_FILE_SHA256,
        name="geometry manifest",
    )
    _require_file_hash(
        policy_lock_path,
        PINNED_POLICY_LOCK_FILE_SHA256,
        name="geometry policy",
    )
    _require_file_hash(
        census_path,
        PINNED_CENSUS_FILE_SHA256,
        name="geometry census",
    )
    authorize_confirmation(
        manifest,
        policy,
        census,
        explicit_acknowledgement=confirmation_scoring_acknowledged,
        device=device,
    )
    ids = [str(record["record_id"]) for record in manifest["records"]]
    if (
        len(records) != EXPECTED_RECORDS
        or [str(record.record_id) for record in records] != ids
    ):
        raise ValueError("geometry admission requires ordered all16 records")
    for runtime_record, public_record in zip(records, manifest["records"]):
        _validate_record(public_record, runtime_record)
    verify_loaded_runtime(runtime, device=device)
    admission = locked_admission_policy(manifest, policy, census)
    expected = _base_report(
        manifest,
        policy,
        census,
        manifest_path=manifest_path,
        policy_lock_path=policy_lock_path,
        census_path=census_path,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    if output.exists() and not (resume or overwrite):
        raise FileExistsError("output exists; pass resume or overwrite")
    discarded = 0
    if resume:
        if not output.is_file():
            raise ValueError("cannot resume a missing geometry report")
        discarded = _validate_resume(
            _load_mapping(output, name="geometry resume report"),
            expected,
        )
    report = expected
    report["resume_requested"] = bool(resume)
    report["discarded_resume_rows"] = discarded
    report["reused_completed_resume_records"] = 0
    hardened._atomic_write(output, report)
    started = time.perf_counter()
    result_rows = []
    for runtime_record, public_record in zip(records, manifest["records"]):
        try:
            row = evaluate_admission_record(
                runtime,
                runtime_record,
                public_record,
                warmup=warmup,
                repeats=repeats,
                admission_policy=admission,
            )
        except Exception as exc:
            row = _failed_row(runtime_record, exc)
        result_rows.append(row)
        report["records"] = result_rows
        report["summary"] = summarize_records(
            result_rows,
            frozen_records=EXPECTED_RECORDS,
            admission_thresholds=admission,
        )
        report["status"] = "running"
        report["elapsed_seconds_this_process"] = time.perf_counter() - started
        hardened._atomic_write(output, report)
    report["summary"] = summarize_records(
        result_rows,
        frozen_records=EXPECTED_RECORDS,
        admission_thresholds=admission,
    )
    denominator = report["summary"]["denominators"]
    report["status"] = (
        "completed"
        if denominator["completed_records"] == EXPECTED_RECORDS
        and denominator["record_failures"] == 0
        else "completed_with_record_failures"
    )
    report["elapsed_seconds_this_process"] = time.perf_counter() - started
    hardened._atomic_write(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--policy-lock", default=str(DEFAULT_POLICY_LOCK))
    parser.add_argument("--census", default=str(DEFAULT_CENSUS))
    parser.add_argument("--data-path")
    parser.add_argument("--core-manifest", default=str(benchmark.DEFAULT_CORE_MANIFEST_PATH))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-confirmation-scoring", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="validate and discard prior rows, then re-execute all16",
    )
    mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest)
    policy_path = Path(args.policy_lock)
    census_path = Path(args.census)
    try:
        manifest, policy, census = load_locked_inputs(
            manifest_path,
            policy_path,
            census_path,
        )
        authorize_confirmation(
            manifest,
            policy,
            census,
            explicit_acknowledgement=args.allow_confirmation_scoring,
            device=args.device,
        )
        output = _validate_output_path(
            args.out,
            manifest_path=manifest_path,
            policy_lock_path=policy_path,
            census_path=census_path,
            core_manifest_path=Path(args.core_manifest),
            data_path=(None if args.data_path is None else Path(args.data_path)),
        )
    except (OSError, PermissionError, ValueError, benchmark.ManifestError) as exc:
        parser.error(str(exc))
    rows = benchmark.base.load_pinned_longmemeval_rows(args.data_path)
    runtime = _make_runtime(device=args.device)
    runtime.ensure_loaded()
    try:
        corrected = benchmark.rehydrate_manifest(
            manifest,
            rows,
            runtime.tokenizer,
            core_manifest=Path(args.core_manifest),
        )
        records = tuple(item.runtime for item in corrected)
        report = run_records(
            manifest,
            policy,
            census,
            records,
            runtime,
            manifest_path=manifest_path,
            policy_lock_path=policy_path,
            census_path=census_path,
            output=output,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            confirmation_scoring_acknowledged=args.allow_confirmation_scoring,
            core_manifest_path=Path(args.core_manifest),
            data_path=(None if args.data_path is None else Path(args.data_path)),
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, benchmark.ManifestError) as exc:
        parser.error(str(exc))
    print(f"wrote evaluation to {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

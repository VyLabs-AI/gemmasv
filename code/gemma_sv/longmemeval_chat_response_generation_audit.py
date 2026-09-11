"""Run the locked post-hoc LongMemEval response-generation audit.

This audit is separate from the finalized teacher-forced method report.  It
rehydrates the same ordered 16-record cohort and greedily decodes fresh target
and retained responses for the predeclared non-cache condition matrix.  The
source-bearing output is local-only by default and is never treated as part of
the original report or its certificate.

The canonical authorization lock must be committed at ``HEAD`` before model
loading.  A run has one persistent atomic attempt claim, no resume mode, and no
overwrite mode.  Freezing the lock performs no model-weight loading.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_geometry_methods as methods
from gemma_sv import eval_longmemeval_chat_methods as method_states
from gemma_sv import longmemeval_chat_geometry_fix as geometry
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.persistent_deletion import DECAY_FACTOR


LOCK_SCHEMA = (
    "gemma-sv-longmemeval-chat-response-generation-authorization-lock-v1"
)
LOCK_SCHEMA_VERSION = 1
LOCK_STATUS = "frozen-before-post-hoc-response-generation"
REPORT_SCHEMA = "gemma-sv-longmemeval-chat-response-generation-audit-v1"
REPORT_SCHEMA_VERSION = 1
PROJECTION_SCHEMA = (
    "gemma-sv-longmemeval-chat-response-generation-case-projection-v1"
)

EXPECTED_RECORDS = 16
EXPECTED_JOINT_RECORDS = 10
PROBE_IDS = ("target_current", "retained")
CONDITION_IDS = (
    methods.PRESENT_ID,
    methods.FRESH_RAW_OMISSION_ID,
    methods.TOKEN_ROW_DIAGNOSTIC_ID,
    methods.EXACT_DECREMENT_ID,
    methods.FIXED_C_REFIT_ID,
    methods.FP32_PROXY_ID,
    methods.DECAY_ID,
    methods.PROMPT_SUPPRESSION_ID,
)
EXECUTED_CONDITION_IDS = (
    methods.PRESENT_ID,
    methods.FRESH_RAW_OMISSION_ID,
    methods.FP32_PROXY_ID,
    methods.DECAY_ID,
    methods.EXACT_DECREMENT_ID,
    methods.FIXED_C_REFIT_ID,
    methods.PROMPT_SUPPRESSION_ID,
)
EXCLUDED_CONDITION_IDS = (methods.CACHE_DELETE_SHIFT_ID,)

MAX_NEW_TOKENS = 64
STOP_TOKEN_IDS = (1, 106)
GENERATION_REPEATS = 2
EVALUATION_SEED = 0
PRESENTATION_CASE_RECORD_ID = (
    "longmemeval-constrained-chat-v1-c8276e265e3db489c090cdcc"
)
PRESENTATION_CASE_POSITION = 2

FINAL_REPORT_FILE_SHA256 = (
    "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6"
)
FINAL_REPORT_PAYLOAD_SHA256 = (
    "d6c4348975a912652fb1f2e153d8e314f3455ca4efcf6e9e565451b703804dac"
)
MANIFEST_FILE_SHA256 = (
    "df10404a5b11491791638190f61b52e30e667d17601819a99925181885bde480"
)
MANIFEST_INTEGRITY_SHA256 = (
    "04fb39a23cbe5e31ed652c0ff8d412df1ab2072999760fba81f7de95dee1a4db"
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
CORE_MANIFEST_FILE_SHA256 = geometry.PINNED_CORE_FILE_SHA256

MODEL_METADATA_FILE_SHA256 = {
    "config.json": (
        "9059f680f4dbd1957f35cb44b9fdd6948f4792db7a7a35ee353ea42e68adf7ff"
    ),
    "generation_config.json": (
        "fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3"
    ),
    "model.safetensors.index.json": (
        "77f4b67de084c31c7bcd373b039908108eee6c6181607e6d53da730e5f0bc659"
    ),
    "tokenizer.json": (
        "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"
    ),
    "tokenizer_config.json": (
        "bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1"
    ),
    "special_tokens_map.json": (
        "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397"
    ),
    "added_tokens.json": (
        "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946"
    ),
}
TOKENIZER_BACKEND_SHA256 = (
    "c1a087240686a7d141101217051f76d5cd4cbe2b6093e3c3553fb26dcc4d0e9a"
)

PINNED_RUNTIME_ENVIRONMENT = {
    "platform": {
        "system": "Darwin",
        "release": "25.5.0",
        "machine": "arm64",
        "python_implementation": "CPython",
        "python_version": "3.11.14",
    },
    "packages": {
        "accelerate": "1.14.0",
        "cvxpy": "1.9.2",
        "huggingface-hub": "1.20.1",
        "numpy": "2.4.6",
        "safetensors": "0.8.0",
        "scipy": "1.17.1",
        "tokenizers": "0.22.2",
        "torch": "2.12.1",
        "transformers": "5.12.1",
    },
    "torch": {
        "git_version": "7269437d655783a26cba32aa88195b741ff496aa",
        "mps_built": True,
        "mps_available": True,
    },
}

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_FINAL_REPORT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_geometry_methods_finalized_v1.json"
)
DEFAULT_MANIFEST = methods.DEFAULT_MANIFEST
DEFAULT_METHOD_LOCK = methods.DEFAULT_METHOD_LOCK
DEFAULT_CORE_MANIFEST = methods.DEFAULT_CORE_MANIFEST
DEFAULT_AUTHORIZATION_LOCK = (
    BENCHMARKS
    / "longmemeval_chat_response_generation_authorization_v1.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_response_generation_audit_v1.json"
)
DEFAULT_ATTEMPT_CLAIM = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_response_generation_audit_v1.claim"
)
ATTEMPT_MARKER_FILENAME = "marker.json"
DEFAULT_PROJECTION_OUTPUT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_response_generation_case_projection_v1.json"
)

# These are the source files that determine rehydration, state construction,
# direct greedy decoding, and the grafted runtime.  They are deliberately
# narrower than the old report's full demo-server fingerprint set.
_IMPLEMENTATION_PATHS = {
    "gemma_sv/__init__.py": PACKAGE / "__init__.py",
    "longmemeval_chat_response_generation_audit.py": Path(__file__).resolve(),
    "eval_longmemeval_chat_geometry_methods.py": (
        PACKAGE / "eval_longmemeval_chat_geometry_methods.py"
    ),
    "eval_longmemeval_chat_methods.py": (
        PACKAGE / "eval_longmemeval_chat_methods.py"
    ),
    "eval_longmemeval_chat_geometry_fix.py": (
        PACKAGE / "eval_longmemeval_chat_geometry_fix.py"
    ),
    "eval_longmemeval_chat_v2.py": PACKAGE / "eval_longmemeval_chat_v2.py",
    "eval_longmemeval_chat.py": PACKAGE / "eval_longmemeval_chat.py",
    "eval_persistent_deletion_baselines.py": (
        PACKAGE / "eval_persistent_deletion_baselines.py"
    ),
    "eval_memops_longitudinal.py": PACKAGE / "eval_memops_longitudinal.py",
    "longmemeval_chat_geometry_fix.py": (
        PACKAGE / "longmemeval_chat_geometry_fix.py"
    ),
    "longmemeval_chat_benchmark.py": (
        PACKAGE / "longmemeval_chat_benchmark.py"
    ),
    "longmemeval_deletion_benchmark.py": (
        PACKAGE / "longmemeval_deletion_benchmark.py"
    ),
    "persistent_deletion.py": PACKAGE / "persistent_deletion.py",
    "graft.py": PACKAGE / "graft.py",
    "layer_select.py": PACKAGE / "layer_select.py",
    "recovery_state.py": PACKAGE / "recovery_state.py",
    "sv_global_attention.py": PACKAGE / "sv_global_attention.py",
    "demo_server/gemma_engine.py": (
        PACKAGE / "demo_server" / "gemma_engine.py"
    ),
    "demo_server/gate_context.py": (
        PACKAGE / "demo_server" / "gate_context.py"
    ),
    "demo_server/certificate.py": (
        PACKAGE / "demo_server" / "certificate.py"
    ),
    "svattn/__init__.py": WORKSPACE / "svattn" / "__init__.py",
    "svattn/causal_sv_attention.py": (
        WORKSPACE / "svattn" / "causal_sv_attention.py"
    ),
    "svattn/mlx_svdd.py": WORKSPACE / "svattn" / "mlx_svdd.py",
    "cp_svm/__init__.py": WORKSPACE / "cp_svm" / "__init__.py",
    "cp_svm/kernels.py": WORKSPACE / "cp_svm" / "kernels.py",
    "cp_svm/oneclass_fast.py": WORKSPACE / "cp_svm" / "oneclass_fast.py",
    "cp_svm/oneclass_incremental.py": (
        WORKSPACE / "cp_svm" / "oneclass_incremental.py"
    ),
}

_SOURCE_FREE_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "content",
        "generated_text",
        "messages",
        "prompt",
        "question",
        "response_text",
        "sessions",
        "source_text",
        "turns",
    }
)
_CREDENTIAL_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization_header",
        "cookie",
        "credential",
        "credentials",
        "hf_token",
        "password",
        "private_key",
        "secret",
    }
)
_CHAT_BOUNDARY_MARKERS = (
    "<start_of_turn>",
    "<end_of_turn>",
    "<bos>",
    "<eos>",
)


class AuditError(ValueError):
    """An audit input, lock, report, or projection violated its contract."""


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


def _text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
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


def _reject_json_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON constant {value!r} is forbidden")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key {key!r} is forbidden")
        result[key] = value
    return result


def _load_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    raw = Path(path).expanduser().read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{name} must contain a JSON object")
    return value


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).casefold()
            yield from _walk_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_keys(child)


def _assert_no_credential_fields(value: Any) -> None:
    leaked = _CREDENTIAL_KEYS.intersection(_walk_keys(value))
    if leaked:
        raise AuditError(
            "credential-bearing fields are forbidden: "
            + ", ".join(sorted(leaked))
        )


def _assert_source_free_lock(value: Mapping[str, Any]) -> None:
    leaked = _SOURCE_FREE_FORBIDDEN_KEYS.intersection(_walk_keys(value))
    if leaked:
        raise AuditError(
            "authorization lock contains source-bearing fields: "
            + ", ".join(sorted(leaked))
        )
    _assert_no_credential_fields(value)
    if value.get("contains_source_text") is not False:
        raise AuditError("authorization lock must be explicitly source-free")


def _implementation_fingerprints() -> dict[str, Any]:
    files = {
        name: _file_sha256(path)
        for name, path in sorted(_IMPLEMENTATION_PATHS.items())
    }
    return {
        "files": files,
        "files_sha256": _payload_sha256(files),
        "file_count": len(files),
    }


def _validate_integrity(
    value: Mapping[str, Any],
    *,
    name: str,
    integrity_key: str = "integrity",
) -> None:
    integrity = value.get(integrity_key)
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop(integrity_key, None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(unsigned)
    ):
        raise AuditError(f"{name} integrity differs")


def _protocol_analysis(
    final_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    *,
    final_report_file_sha256: str,
    manifest_file_sha256: str,
    method_lock_file_sha256: str,
    core_manifest_file_sha256: str,
) -> dict[str, Any]:
    if (
        final_report_file_sha256 != FINAL_REPORT_FILE_SHA256
        or _payload_sha256(final_report) != FINAL_REPORT_PAYLOAD_SHA256
    ):
        raise AuditError("final report is not the exact finalized all16 artifact")
    if (
        manifest_file_sha256 != MANIFEST_FILE_SHA256
        or (manifest.get("integrity") or {}).get("sha256")
        != MANIFEST_INTEGRITY_SHA256
    ):
        raise AuditError("geometry manifest is not the exact frozen artifact")
    if (
        method_lock_file_sha256 != METHOD_LOCK_FILE_SHA256
        or _payload_sha256(method_lock) != METHOD_LOCK_PAYLOAD_SHA256
        or (method_lock.get("integrity") or {}).get("sha256")
        != METHOD_LOCK_INTEGRITY_SHA256
    ):
        raise AuditError("method lock is not the exact frozen artifact")
    if core_manifest_file_sha256 != CORE_MANIFEST_FILE_SHA256:
        raise AuditError("core manifest is not the exact frozen artifact")
    _validate_integrity(manifest, name="geometry manifest")
    _validate_integrity(method_lock, name="method lock")

    if (
        final_report.get("schema") != methods.SCHEMA
        or final_report.get("status") != "completed_with_record_failures"
        or final_report.get("contains_source_text") is not False
        or final_report.get("contains_model_generated_text") not in (None, False)
    ):
        raise AuditError("final report contract differs")

    manifest_rows = manifest.get("records")
    report_rows = final_report.get("records")
    authorization = method_lock.get("authorization")
    bindings = (
        authorization.get("admission_records")
        if isinstance(authorization, Mapping)
        else None
    )
    if not all(
        isinstance(rows, list) and len(rows) == EXPECTED_RECORDS
        for rows in (manifest_rows, report_rows, bindings)
    ):
        raise AuditError("protocol does not contain ordered all16 records")
    manifest_ids = [str(row.get("record_id") or "") for row in manifest_rows]
    report_ids = [str(row.get("record_id") or "") for row in report_rows]
    binding_ids = [str(row.get("record_id") or "") for row in bindings]
    if (
        manifest_ids != report_ids
        or manifest_ids != binding_ids
        or len(set(manifest_ids)) != EXPECTED_RECORDS
        or any(not record_id for record_id in manifest_ids)
    ):
        raise AuditError("ordered all16 record binding differs")

    old_conditions = list(authorization.get("condition_ids") or ())
    if old_conditions != list(methods.CONDITION_IDS):
        raise AuditError("final method condition contract differs")
    for row in report_rows:
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping) or set(conditions) != set(
            methods.CONDITION_IDS
        ):
            raise AuditError("final report condition matrix is incomplete")

    efficacy = authorization.get("efficacy_population")
    joint_ids = (
        list(efficacy.get("record_ids") or ())
        if isinstance(efficacy, Mapping)
        else []
    )
    if (
        len(joint_ids) != EXPECTED_JOINT_RECORDS
        or efficacy.get("name") != "predeclared_geometry_joint10"
        or efficacy.get("selected_before_method_scoring") is not True
        or efficacy.get("record_ids_sha256") != _payload_sha256(joint_ids)
        or any(record_id not in manifest_ids for record_id in joint_ids)
    ):
        raise AuditError("joint10 binding differs")

    prompt_bindings = []
    for record_id, public in zip(manifest_ids, manifest_rows):
        probes = public.get("probes")
        if (
            not isinstance(probes, list)
            or [probe.get("probe_id") for probe in probes] != list(PROBE_IDS)
        ):
            raise AuditError("record probe binding differs")
        frozen_probes = []
        for probe in probes:
            binding = {
                "probe_id": probe["probe_id"],
                "kind": probe["kind"],
                "prompt_utf8_sha256": probe["prompt_sha256"],
                "question_sha256": probe["question_sha256"],
                "answer_sha256": probe["answer_sha256"],
                "target_token_count": int(probe["target_token_count"]),
                "target_token_ids_sha256": probe["target_token_ids_sha256"],
            }
            if (
                not all(
                    _is_sha256(binding[key])
                    for key in (
                        "prompt_utf8_sha256",
                        "question_sha256",
                        "answer_sha256",
                        "target_token_ids_sha256",
                    )
                )
                or binding["target_token_count"] < 1
            ):
                raise AuditError("record probe hash binding differs")
            frozen_probes.append(binding)
        prompt_bindings.append(
            {"record_id": record_id, "probes": frozen_probes}
        )

    if manifest_ids[PRESENTATION_CASE_POSITION - 1] != (
        PRESENTATION_CASE_RECORD_ID
    ):
        raise AuditError("presentation case is not frozen at position 2")
    return {
        "ordered_record_ids": manifest_ids,
        "joint_record_ids": joint_ids,
        "admission_bindings": copy.deepcopy(bindings),
        "prompt_bindings": prompt_bindings,
    }


def freeze_authorization_lock(
    final_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    *,
    final_report_file_sha256: str = FINAL_REPORT_FILE_SHA256,
    manifest_file_sha256: str = MANIFEST_FILE_SHA256,
    method_lock_file_sha256: str = METHOD_LOCK_FILE_SHA256,
    core_manifest_file_sha256: str = CORE_MANIFEST_FILE_SHA256,
) -> dict[str, Any]:
    """Build the source-free pre-run lock without loading model weights."""

    analysis = _protocol_analysis(
        final_report,
        manifest,
        method_lock,
        final_report_file_sha256=final_report_file_sha256,
        manifest_file_sha256=manifest_file_sha256,
        method_lock_file_sha256=method_lock_file_sha256,
        core_manifest_file_sha256=core_manifest_file_sha256,
    )
    method_runtime = copy.deepcopy(method_lock["authorization"]["runtime"])
    if method_runtime != methods.admission.runtime_contract(device="mps"):
        raise AuditError("method runtime is not the pinned MPS contract")

    prompt_contract = {
        "chat_serialization": copy.deepcopy(manifest["chat_serialization"]),
        "chat_serialization_sha256": _payload_sha256(
            manifest["chat_serialization"]
        ),
        "base_probe_bytes": analysis["prompt_bindings"],
        "base_probe_bytes_sha256": _payload_sha256(
            analysis["prompt_bindings"]
        ),
        "tokenization": {
            "add_special_tokens": False,
            "encoding": "UTF-8",
        },
        "target_free_prompt_suppression": {
            "instruction_utf8_sha256": _text_sha256(
                method_states.PROMPT_SUPPRESSION_INSTRUCTION
            ),
            "instruction_utf8_bytes": len(
                method_states.PROMPT_SUPPRESSION_INSTRUCTION.encode("utf-8")
            ),
            "insertion": (
                "insert instruction plus LF immediately before the single "
                "constrained instruction inside the registered user turn"
            ),
            "transformation_implementation_bound": True,
            "target_answer_substring_forbidden": True,
        },
    }
    lock: dict[str, Any] = {
        "schema": LOCK_SCHEMA,
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": LOCK_STATUS,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "source_bearing_output_authorized": True,
        "artifacts": {
            "final_report": {
                "canonical_repository_path": (
                    "outputs/gemma_sv_rag/"
                    "longmemeval_chat_geometry_methods_finalized_v1.json"
                ),
                "file_sha256": final_report_file_sha256,
                "payload_sha256": FINAL_REPORT_PAYLOAD_SHA256,
                "immutable_input": True,
            },
            "geometry_manifest": {
                "canonical_repository_path": (
                    "gemma_sv/benchmarks/"
                    "longmemeval_chat_geometry_corrected_v1.json"
                ),
                "file_sha256": manifest_file_sha256,
                "integrity_sha256": MANIFEST_INTEGRITY_SHA256,
            },
            "method_lock": {
                "canonical_repository_path": (
                    "gemma_sv/benchmarks/"
                    "longmemeval_chat_geometry_method_authorization_v1.json"
                ),
                "file_sha256": method_lock_file_sha256,
                "payload_sha256": METHOD_LOCK_PAYLOAD_SHA256,
                "integrity_sha256": METHOD_LOCK_INTEGRITY_SHA256,
            },
            "core_manifest": {
                "canonical_repository_path": (
                    "gemma_sv/benchmarks/"
                    "longmemeval_chat_confirmation_v1.json"
                ),
                "file_sha256": core_manifest_file_sha256,
            },
        },
        "cohort": {
            "ordered_record_ids": analysis["ordered_record_ids"],
            "ordered_record_ids_sha256": _payload_sha256(
                analysis["ordered_record_ids"]
            ),
            "records": EXPECTED_RECORDS,
            "all_records_without_replacement": True,
            "joint10": {
                "record_ids": analysis["joint_record_ids"],
                "record_ids_sha256": _payload_sha256(
                    analysis["joint_record_ids"]
                ),
                "records": EXPECTED_JOINT_RECORDS,
                "selected_before_generated_outcomes": True,
                "source": "predeclared_geometry_joint10",
            },
            "admission_bindings": analysis["admission_bindings"],
            "admission_bindings_sha256": _payload_sha256(
                analysis["admission_bindings"]
            ),
        },
        "runtime": {
            "method_runtime": method_runtime,
            "environment": copy.deepcopy(PINNED_RUNTIME_ENVIRONMENT),
            "model_and_tokenizer": {
                "model_id": method_runtime["model_id"],
                "model_revision": method_runtime["model_revision"],
                "tokenizer_id": geometry.chat_v1.CHAT_TOKENIZER_ID,
                "tokenizer_revision": geometry.chat_v1.CHAT_TOKENIZER_REVISION,
                "tokenizer_fast_required": True,
                "tokenizer_backend_sha256": TOKENIZER_BACKEND_SHA256,
                "tokenizer_vocab_size": 262144,
                "tokenizer_length": 262145,
                "tokenizer_eos_token_id": 1,
                "tokenizer_bos_token_id": 2,
                "tokenizer_pad_token_id": 0,
                "metadata_file_sha256": copy.deepcopy(
                    MODEL_METADATA_FILE_SHA256
                ),
                "local_files_only": True,
                "network_access": False,
                "credential_material_passed_by_audit_code": False,
            },
        },
        "prompt_contract": prompt_contract,
        "conditions": {
            "condition_ids": list(CONDITION_IDS),
            "condition_ids_sha256": _payload_sha256(list(CONDITION_IDS)),
            "executed_condition_ids": list(EXECUTED_CONDITION_IDS),
            "excluded_condition_ids": list(EXCLUDED_CONDITION_IDS),
            "token_row_alias": {
                "condition_id": methods.TOKEN_ROW_DIAGNOSTIC_ID,
                "alias_of": methods.FRESH_RAW_OMISSION_ID,
                "additional_model_execution": False,
                "identical_registered_turn_token_ids_required": True,
            },
        },
        "generation": {
            "algorithm": "direct-step greedy argmax",
            "do_sample": False,
            "num_beams": 1,
            "temperature": None,
            "top_k": None,
            "top_p": None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_token_ids": list(STOP_TOKEN_IDS),
            "include_terminal_stop_in_token_hash": True,
            "response_decoder": {
                "skip_special_tokens": True,
                "clean_up_tokenization_spaces": False,
                "encoding": "UTF-8",
            },
            "token_hash_serialization": "canonical-json-integer-array",
            "repeat_generations": GENERATION_REPEATS,
            "exact_repeat_token_and_response_match_required": True,
            "planned_workload": {
                "records": EXPECTED_RECORDS,
                "executed_conditions_per_record": len(
                    EXECUTED_CONDITION_IDS
                ),
                "probes_per_condition": len(PROBE_IDS),
                "repeat_generations_per_probe": GENERATION_REPEATS,
                "generation_calls": (
                    EXPECTED_RECORDS
                    * len(EXECUTED_CONDITION_IDS)
                    * len(PROBE_IDS)
                    * GENERATION_REPEATS
                ),
                "maximum_generated_token_steps": (
                    EXPECTED_RECORDS
                    * len(EXECUTED_CONDITION_IDS)
                    * len(PROBE_IDS)
                    * GENERATION_REPEATS
                    * MAX_NEW_TOKENS
                ),
            },
            "global_seed": EVALUATION_SEED,
            "torch_deterministic_algorithms": True,
            "torch_deterministic_warn_only": False,
            "teacher_forced_scores_used_to_infer_text": False,
        },
        "generation_implementation": _implementation_fingerprints(),
        "attempt_policy": {
            "attempts_authorized": 1,
            "atomic_claim": "os.mkdir",
            "persistent_claim": True,
            "canonical_claim_workspace_path": (
                "outputs/gemma_sv_rag/"
                "longmemeval_chat_response_generation_audit_v1.claim"
            ),
            "marker_filename": ATTEMPT_MARKER_FILENAME,
            "resume_allowed": False,
            "overwrite_allowed": False,
            "retry_after_process_abort_allowed": False,
            "all_records_conditions_and_probes_preallocated": True,
            "no_outcome_based_filtering": True,
        },
        "output_policy": {
            "canonical_workspace_path": (
                "outputs/gemma_sv_rag/"
                "longmemeval_chat_response_generation_audit_v1.json"
            ),
            "atomic_first_writer": True,
            "contains_source_text": True,
            "contains_model_generated_text": True,
            "source_bearing": True,
            "local_only": True,
            "release_authorized": False,
            "separate_from_final_report_and_certificate": True,
        },
        "privacy_and_scope": {
            "longmemeval": {
                "dataset_id": geometry.base.DATASET_ID,
                "dataset_revision": geometry.base.DATASET_REVISION,
                "artifact_sha256": geometry.base.DATASET_ARTIFACT_SHA256,
                "repository": geometry.base.LONGMEMEVAL_REPOSITORY,
                "repository_revision": (
                    geometry.base.LONGMEMEVAL_REPOSITORY_REVISION
                ),
                "public_provenance": True,
                "license": geometry.base.DATASET_LICENSE,
            },
            "credential_fields_forbidden": True,
            "environment_variables_not_serialized": True,
            "offline_cached_model_and_data_only": True,
            "leak_audit": (
                "exact registered-answer substring, exact source-response "
                "substring, control-instruction echo, and chat-boundary checks"
            ),
            "decoded_strings_exhaust_extractability": False,
            "post_hoc_audit_not_original_report": True,
            "post_hoc_audit_not_original_certificate": True,
        },
        "presentation_case": {
            "record_id": PRESENTATION_CASE_RECORD_ID,
            "authorized_position": PRESENTATION_CASE_POSITION,
            "selected_in_this_pre_run_lock": True,
            "selected_before_generated_outcomes": True,
            "projection_must_remain_source_bearing_local_only": True,
            "projection_release_authorized": False,
        },
    }
    lock["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(lock),
    }
    _assert_source_free_lock(lock)
    return lock


def validate_authorization_lock(
    lock: Mapping[str, Any],
    *,
    final_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    final_report_file_sha256: str = FINAL_REPORT_FILE_SHA256,
    manifest_file_sha256: str = MANIFEST_FILE_SHA256,
    method_lock_file_sha256: str = METHOD_LOCK_FILE_SHA256,
    core_manifest_file_sha256: str = CORE_MANIFEST_FILE_SHA256,
) -> None:
    _assert_source_free_lock(lock)
    _validate_integrity(lock, name="authorization lock")
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("status") != LOCK_STATUS
    ):
        raise AuditError("authorization lock schema or status differs")
    expected = freeze_authorization_lock(
        final_report,
        manifest,
        method_lock,
        final_report_file_sha256=final_report_file_sha256,
        manifest_file_sha256=manifest_file_sha256,
        method_lock_file_sha256=method_lock_file_sha256,
        core_manifest_file_sha256=core_manifest_file_sha256,
    )
    if dict(lock) != expected:
        raise AuditError("authorization lock differs from frozen inputs")


def _require_exact_canonical_path(
    path: str | Path,
    canonical: str | Path,
    *,
    name: str,
) -> Path:
    candidate = Path(path).expanduser()
    if candidate.resolve(strict=False) != Path(canonical).resolve(strict=False):
        raise PermissionError(f"{name} requires its canonical workspace path")
    return candidate


def _require_head_committed_file(path: Path) -> None:
    workspace = WORKSPACE.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise PermissionError("committed input is outside the git workspace") from exc
    completed = subprocess.run(
        ["git", "-C", str(workspace), "show", f"HEAD:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise PermissionError(f"{relative} must be committed at HEAD")
    if completed.stdout != resolved.read_bytes():
        raise PermissionError(f"{relative} differs from its committed HEAD copy")


def _load_protocol_inputs(
    *,
    final_report_path: str | Path = DEFAULT_FINAL_REPORT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    method_lock_path: str | Path = DEFAULT_METHOD_LOCK,
    core_manifest_path: str | Path = DEFAULT_CORE_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final_path = _require_exact_canonical_path(
        final_report_path,
        DEFAULT_FINAL_REPORT,
        name="final report",
    )
    manifest_file = _require_exact_canonical_path(
        manifest_path,
        DEFAULT_MANIFEST,
        name="geometry manifest",
    )
    method_file = _require_exact_canonical_path(
        method_lock_path,
        DEFAULT_METHOD_LOCK,
        name="method lock",
    )
    core_file = _require_exact_canonical_path(
        core_manifest_path,
        DEFAULT_CORE_MANIFEST,
        name="core manifest",
    )
    observed = {
        "final": _file_sha256(final_path),
        "manifest": _file_sha256(manifest_file),
        "method": _file_sha256(method_file),
        "core": _file_sha256(core_file),
    }
    final_report = _load_mapping(final_path, name="final report")
    manifest = _load_mapping(manifest_file, name="geometry manifest")
    method_lock = _load_mapping(method_file, name="method lock")
    _protocol_analysis(
        final_report,
        manifest,
        method_lock,
        final_report_file_sha256=observed["final"],
        manifest_file_sha256=observed["manifest"],
        method_lock_file_sha256=observed["method"],
        core_manifest_file_sha256=observed["core"],
    )
    return final_report, manifest, method_lock


def load_committed_authorization_lock(
    path: str | Path = DEFAULT_AUTHORIZATION_LOCK,
    *,
    final_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
) -> dict[str, Any]:
    lock_path = _require_exact_canonical_path(
        path,
        DEFAULT_AUTHORIZATION_LOCK,
        name="response-generation authorization lock",
    )
    _require_head_committed_file(lock_path)
    lock = _load_mapping(lock_path, name="response-generation authorization lock")
    validate_authorization_lock(
        lock,
        final_report=final_report,
        manifest=manifest,
        method_lock=method_lock,
    )
    for implementation_path in _IMPLEMENTATION_PATHS.values():
        _require_head_committed_file(implementation_path)
    return lock


def _observed_runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in PINNED_RUNTIME_ENVIRONMENT["packages"]:
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"required package {name!r} is unavailable") from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the pinned torch runtime is unavailable") from exc
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "packages": packages,
        "torch": {
            "git_version": str(torch.version.git_version),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
    }


def verify_runtime_environment(lock: Mapping[str, Any]) -> None:
    expected = (lock.get("runtime") or {}).get("environment")
    if expected != PINNED_RUNTIME_ENVIRONMENT:
        raise RuntimeError("authorization lock runtime environment drifted")
    observed = _observed_runtime_environment()
    if observed != expected:
        raise RuntimeError("active Python/package/MPS runtime differs from the lock")


@contextmanager
def _offline_huggingface() -> Iterable[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _verify_cached_model_metadata(lock: Mapping[str, Any]) -> None:
    binding = lock["runtime"]["model_and_tokenizer"]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required") from exc
    for filename, expected in binding["metadata_file_sha256"].items():
        resolved = hf_hub_download(
            repo_id=binding["model_id"],
            filename=filename,
            revision=binding["model_revision"],
            local_files_only=True,
        )
        if _file_sha256(resolved) != expected:
            raise RuntimeError(f"cached model metadata {filename!r} differs")


def _normalize_eos_ids(value: Any) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return tuple(int(item) for item in values)


def verify_loaded_generation_runtime(
    runtime: Any,
    *,
    lock: Mapping[str, Any],
) -> None:
    methods.verify_execution_runtime(
        runtime,
        locked_runtime=lock["runtime"]["method_runtime"],
    )
    binding = lock["runtime"]["model_and_tokenizer"]
    tokenizer = runtime.tokenizer
    if (
        getattr(tokenizer, "is_fast", None) is not True
        or int(getattr(tokenizer, "vocab_size", -1))
        != binding["tokenizer_vocab_size"]
        or len(tokenizer) != binding["tokenizer_length"]
        or int(getattr(tokenizer, "eos_token_id", -1))
        != binding["tokenizer_eos_token_id"]
        or int(getattr(tokenizer, "bos_token_id", -1))
        != binding["tokenizer_bos_token_id"]
        or int(getattr(tokenizer, "pad_token_id", -1))
        != binding["tokenizer_pad_token_id"]
    ):
        raise RuntimeError("loaded tokenizer identity differs from the lock")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if (
        backend is None
        or _text_sha256(backend.to_str())
        != binding["tokenizer_backend_sha256"]
    ):
        raise RuntimeError("loaded tokenizer backend differs from the lock")
    model_eos = _normalize_eos_ids(runtime.model.config.eos_token_id)
    if model_eos != tuple(lock["generation"]["stop_token_ids"]):
        raise RuntimeError("loaded model stop-token contract differs")


def _configure_determinism() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for deterministic generation") from exc
    method_states.baseline._seed_everything(EVALUATION_SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("torch deterministic algorithms were not enabled")


def _inference_context():
    try:
        import torch
    except ImportError:
        return nullcontext()
    return torch.inference_mode()


def _encoded_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = (
        encoded.get("input_ids")
        if isinstance(encoded, Mapping)
        else encoded.input_ids
    )
    result = [int(token_id) for token_id in values]
    if not result:
        raise RuntimeError("registered query tokenization is empty")
    return result


def _decoded_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def _raw_decoded_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _argmax_token(logits: Any) -> int:
    value = logits.argmax()
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


@dataclass(frozen=True)
class GreedyResponse:
    token_ids: tuple[int, ...]
    response_text: str
    raw_content_text: str
    stop_reason: str
    stop_token_id: int | None
    prompt_token_count: int


def greedy_generate_response(
    runtime: Any,
    memory: Any,
    prompt_text: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    stop_token_ids: Sequence[int] = STOP_TOKEN_IDS,
) -> GreedyResponse:
    """Directly decode argmax tokens from a forked persistent state."""

    if int(max_new_tokens) != MAX_NEW_TOKENS:
        raise AuditError("max_new_tokens differs from the frozen contract")
    if tuple(int(item) for item in stop_token_ids) != STOP_TOKEN_IDS:
        raise AuditError("stop-token IDs differ from the frozen contract")
    prompt_ids = _encoded_ids(runtime.tokenizer, prompt_text)
    generated: list[int] = []
    stop_token_id: int | None = None
    with runtime._persistent_branch(memory) as branch:
        with _inference_context():
            logits = runtime._persistent_prompt(branch, prompt_text)
            for index in range(MAX_NEW_TOKENS):
                token_id = _argmax_token(logits)
                generated.append(token_id)
                if token_id in STOP_TOKEN_IDS:
                    stop_token_id = token_id
                    break
                if index + 1 < MAX_NEW_TOKENS:
                    logits = runtime._persistent_step(branch, token_id)
    if not generated:
        raise RuntimeError("greedy decoder emitted no tokens")
    content_ids = (
        generated[:-1]
        if generated[-1] in STOP_TOKEN_IDS
        else generated
    )
    return GreedyResponse(
        token_ids=tuple(generated),
        response_text=_decoded_text(runtime.tokenizer, generated),
        raw_content_text=_raw_decoded_text(runtime.tokenizer, content_ids),
        stop_reason=(
            "terminal_stop_token"
            if stop_token_id is not None
            else "max_new_tokens"
        ),
        stop_token_id=stop_token_id,
        prompt_token_count=len(prompt_ids),
    )


def _response_seed(record_id: str, condition_id: str, probe_id: str) -> int:
    digest = hashlib.sha256(
        (
            "longmemeval-response-generation-v1\0"
            f"{EVALUATION_SEED}\0{record_id}\0{condition_id}\0{probe_id}"
        ).encode("utf-8")
    ).digest()
    return EVALUATION_SEED + int.from_bytes(digest[:4], "big")


def _seed_response(value: int) -> None:
    method_states.baseline._seed_everything(int(value))


def _generation_artifact(response: GreedyResponse) -> dict[str, Any]:
    token_ids = list(response.token_ids)
    return {
        "response_text": response.response_text,
        "response_utf8_sha256": _text_sha256(response.response_text),
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": _payload_sha256(token_ids),
        "generated_token_count": len(token_ids),
        "stop_reason": response.stop_reason,
        "stop_token_id": response.stop_token_id,
        "prompt_token_count": response.prompt_token_count,
    }


def _casefold_contains(text: str, candidate: str) -> bool:
    needle = str(candidate).strip().casefold()
    return bool(needle) and needle in str(text).casefold()


def _content_audit(
    response: GreedyResponse,
    *,
    record: Any,
    probe_id: str,
) -> dict[str, Any]:
    answer_by_probe = {
        probe.probe_id: str(probe.answer) for probe in record.probes
    }
    queried = answer_by_probe[probe_id]
    other_id = next(item for item in PROBE_IDS if item != probe_id)
    other = answer_by_probe[other_id]
    rendered = response.response_text
    normalized = rendered.strip().casefold()
    source_context = str(record.context.original_text).casefold()
    return {
        "scope": (
            "exact substring and registered-boundary checks only; "
            "not an exhaustive extraction audit"
        ),
        "queried_answer_exact_casefold_substring": _casefold_contains(
            rendered, queried
        ),
        "other_registered_answer_exact_casefold_substring": _casefold_contains(
            rendered, other
        ),
        "response_exact_casefold_substring_of_source_context": bool(
            len(normalized) >= 8 and normalized in source_context
        ),
        "reserved_chat_boundary_emitted_before_terminal_stop": any(
            marker in response.raw_content_text
            for marker in _CHAT_BOUNDARY_MARKERS
        ),
        "constrained_instruction_echo": _casefold_contains(
            rendered, geometry.chat_v1.CONSTRAINED_INSTRUCTION
        ),
        "suppression_instruction_echo": _casefold_contains(
            rendered, method_states.PROMPT_SUPPRESSION_INSTRUCTION
        ),
        "decoded_string_exhausts_extractability": False,
    }


def _failed_probe(
    probe_id: str,
    exc: Exception,
    *,
    attempted: bool,
) -> dict[str, Any]:
    return {
        "probe_id": probe_id,
        "status": "failed",
        "attempted": bool(attempted),
        "generation_attempts_started": 0,
        "generation_attempts_completed": 0,
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": _text_sha256(str(exc)),
    }


def _binding_for_probe(
    lock: Mapping[str, Any],
    record_id: str,
    probe_id: str,
) -> Mapping[str, Any]:
    records = lock["prompt_contract"]["base_probe_bytes"]
    matches = [row for row in records if row["record_id"] == record_id]
    if len(matches) != 1:
        raise AuditError("record prompt binding is unavailable")
    probes = [
        row for row in matches[0]["probes"] if row["probe_id"] == probe_id
    ]
    if len(probes) != 1:
        raise AuditError("probe prompt binding is unavailable")
    return probes[0]


def _generate_probe(
    runtime: Any,
    memory: Any,
    probe: Any,
    *,
    condition_id: str,
    record: Any,
    lock: Mapping[str, Any],
    prompt_is_suppressed: bool,
) -> dict[str, Any]:
    binding = _binding_for_probe(lock, record.record_id, probe.probe_id)
    if not prompt_is_suppressed and (
        _text_sha256(probe.prompt) != binding["prompt_utf8_sha256"]
    ):
        raise AuditError("base registered prompt bytes differ from the lock")
    if prompt_is_suppressed and (
        method_states.PROMPT_SUPPRESSION_INSTRUCTION not in probe.prompt
    ):
        raise AuditError("suppression prompt lacks the frozen instruction")

    seed = _response_seed(record.record_id, condition_id, probe.probe_id)
    responses: list[GreedyResponse] = []
    artifacts: list[dict[str, Any]] = []
    for repetition in range(GENERATION_REPEATS):
        _seed_response(seed)
        try:
            response = greedy_generate_response(runtime, memory, probe.prompt)
        except Exception as exc:
            failed = _failed_probe(probe.probe_id, exc, attempted=True)
            failed.update(
                {
                    "probe_kind": probe.kind,
                    "response_seed": seed,
                    "prompt_utf8_sha256": _text_sha256(probe.prompt),
                    "prompt_is_target_free_suppression": bool(
                        prompt_is_suppressed
                    ),
                    "generation_attempts_started": repetition + 1,
                    "generation_attempts_completed": len(artifacts),
                    "completed_generation_attempts": artifacts,
                }
            )
            return failed
        responses.append(response)
        artifacts.append(_generation_artifact(response))
    first, second = responses
    first_artifact, second_artifact = artifacts
    matched = (
        first.token_ids == second.token_ids
        and first.response_text == second.response_text
    )
    result: dict[str, Any] = {
        "probe_id": probe.probe_id,
        "probe_kind": probe.kind,
        "status": "completed" if matched else "failed_nondeterministic",
        "attempted": True,
        "generation_attempts_started": GENERATION_REPEATS,
        "generation_attempts_completed": GENERATION_REPEATS,
        "response_seed": seed,
        "prompt_utf8_sha256": _text_sha256(probe.prompt),
        "prompt_is_target_free_suppression": bool(prompt_is_suppressed),
        **first_artifact,
        "repeat_check": {
            "repetitions": GENERATION_REPEATS,
            "exact_token_ids_and_response_text_match": matched,
            "repeat_response_utf8_sha256": second_artifact[
                "response_utf8_sha256"
            ],
            "repeat_generated_token_ids_sha256": second_artifact[
                "generated_token_ids_sha256"
            ],
            "repeat_generated_token_count": second_artifact[
                "generated_token_count"
            ],
            "repeat_stop_reason": second_artifact["stop_reason"],
            "repeat_stop_token_id": second_artifact["stop_token_id"],
        },
    }
    if matched:
        result["content_audit"] = _content_audit(
            first,
            record=record,
            probe_id=probe.probe_id,
        )
    else:
        result["repeat_check"]["repeat_response_text"] = second_artifact[
            "response_text"
        ]
        result["repeat_check"]["repeat_generated_token_ids"] = second_artifact[
            "generated_token_ids"
        ]
    return result


def _failed_condition(condition_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "status": "failed",
        "additional_generation_executions": 0,
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": _text_sha256(str(exc)),
        "probes": [
            _failed_probe(probe_id, exc, attempted=False)
            for probe_id in PROBE_IDS
        ],
    }


def _execute_condition(
    runtime: Any,
    memory: Any,
    probes: Sequence[Any],
    *,
    condition_id: str,
    record: Any,
    lock: Mapping[str, Any],
    semantics: Mapping[str, Any],
    prompt_is_suppressed: bool = False,
) -> dict[str, Any]:
    results = []
    for probe in probes:
        try:
            result = _generate_probe(
                runtime,
                memory,
                probe,
                condition_id=condition_id,
                record=record,
                lock=lock,
                prompt_is_suppressed=prompt_is_suppressed,
            )
        except Exception as exc:
            result = _failed_probe(probe.probe_id, exc, attempted=True)
        results.append(result)
    failures = [
        result["probe_id"]
        for result in results
        if result["status"] != "completed"
    ]
    return {
        "condition_id": condition_id,
        "status": "completed" if not failures else "completed_with_probe_failures",
        "semantics": copy.deepcopy(dict(semantics)),
        "additional_generation_executions": sum(
            int(result.get("generation_attempts_started", 0))
            for result in results
        ),
        "failed_probe_ids": failures,
        "probes": results,
    }


def _alias_condition(raw_condition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": methods.TOKEN_ROW_DIAGNOSTIC_ID,
        "status": "aliased",
        "alias_of": methods.FRESH_RAW_OMISSION_ID,
        "additional_generation_executions": 0,
        "semantics": {
            "zero_cost_alias": True,
            "identical_registered_turn_token_ids": True,
            "additional_model_execution": False,
        },
        "probes": [
            {
                "probe_id": probe_id,
                "status": "aliased",
                "alias_of": {
                    "condition_id": methods.FRESH_RAW_OMISSION_ID,
                    "probe_id": probe_id,
                },
                "aliased_source_status": next(
                    probe["status"]
                    for probe in raw_condition["probes"]
                    if probe["probe_id"] == probe_id
                ),
                "additional_generation_executions": 0,
            }
            for probe_id in PROBE_IDS
        ],
    }


def _memory_fingerprint(memory: Any) -> str:
    return method_states.longitudinal._persistent_state_fingerprint(memory)


def _audit_record(
    runtime: Any,
    record: Any,
    public_record: Mapping[str, Any],
    *,
    admission_binding: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    methods.admission._validate_record(public_record, record)
    if admission_binding.get("record_id") != record.record_id:
        raise AuditError("record admission binding differs")
    if tuple(record.context.edited_token_ids) != tuple(
        record.raw_omitted_token_ids
    ):
        raise AuditError("token-row alias is not identical to raw omission")
    probes = method_states.build_probes(record)
    if [probe.probe_id for probe in probes] != list(PROBE_IDS):
        raise AuditError("runtime probe order differs")
    for probe in probes:
        binding = _binding_for_probe(lock, record.record_id, probe.probe_id)
        if (
            _text_sha256(probe.prompt) != binding["prompt_utf8_sha256"]
            or _text_sha256(probe.target) != binding["answer_sha256"]
            or len(probe.target_ids) != binding["target_token_count"]
            or geometry.base.token_ids_sha256(probe.target_ids)
            != binding["target_token_ids_sha256"]
        ):
            raise AuditError("rehydrated probe differs from the pre-run lock")

    row: dict[str, Any] = {
        "record_id": record.record_id,
        "status": "running",
        "joint10_member": bool(admission_binding["joint_admitted"]),
        "conditions": {
            condition_id: {
                "condition_id": condition_id,
                "status": "not_attempted",
                "probes": [
                    {"probe_id": probe_id, "status": "not_attempted"}
                    for probe_id in PROBE_IDS
                ],
            }
            for condition_id in CONDITION_IDS
        },
    }

    original_memory = None
    original_error: Exception | None = None
    try:
        original_memory = runtime.prefill_persistent(
            list(record.context.original_token_ids)
        )
        expected = geometry.base.token_ids_sha256(
            record.context.original_token_ids
        )
        if str(original_memory.input_digest) != expected:
            raise RuntimeError("original persistent prefill digest differs")
        fingerprint_before = _memory_fingerprint(original_memory)
    except Exception as exc:
        original_error = exc
        fingerprint_before = None

    if original_memory is None:
        assert original_error is not None
        for condition_id in (
            methods.PRESENT_ID,
            methods.FP32_PROXY_ID,
            methods.DECAY_ID,
            methods.EXACT_DECREMENT_ID,
            methods.FIXED_C_REFIT_ID,
            methods.PROMPT_SUPPRESSION_ID,
        ):
            row["conditions"][condition_id] = _failed_condition(
                condition_id, original_error
            )
    else:
        row["conditions"][methods.PRESENT_ID] = _execute_condition(
            runtime,
            original_memory,
            probes,
            condition_id=methods.PRESENT_ID,
            record=record,
            lock=lock,
            semantics={"owned_round_present": True, "reference": True},
        )

    try:
        raw_memory = runtime.prefill_persistent(
            list(record.raw_omitted_token_ids)
        )
        expected_raw = geometry.base.token_ids_sha256(
            record.raw_omitted_token_ids
        )
        if str(raw_memory.input_digest) != expected_raw:
            raise RuntimeError("raw-omission persistent prefill digest differs")
        raw_condition = _execute_condition(
            runtime,
            raw_memory,
            probes,
            condition_id=methods.FRESH_RAW_OMISSION_ID,
            record=record,
            lock=lock,
            semantics={
                "fresh_prefill": True,
                "owned_round_omitted": True,
                "suffix_recomputed": True,
            },
        )
    except Exception as exc:
        raw_condition = _failed_condition(
            methods.FRESH_RAW_OMISSION_ID, exc
        )
    row["conditions"][methods.FRESH_RAW_OMISSION_ID] = raw_condition
    row["conditions"][methods.TOKEN_ROW_DIAGNOSTIC_ID] = _alias_condition(
        raw_condition
    )

    if original_memory is not None:
        try:
            proxy_memory = runtime.delete_persistent(
                original_memory,
                record.context.forget_positions,
                kind="fp32_masked_refit_proxy",
            )
            if method_states._fallback_used(proxy_memory):
                raise RuntimeError("FP32 proxy unexpectedly used full repack")
            row["conditions"][methods.FP32_PROXY_ID] = _execute_condition(
                runtime,
                proxy_memory,
                probes,
                condition_id=methods.FP32_PROXY_ID,
                record=record,
                lock=lock,
                semantics={
                    "executed_method": "incremental_fp32_masked_refit",
                    "query_independent": True,
                    "full_repack_fallback": False,
                },
            )
        except Exception as exc:
            row["conditions"][methods.FP32_PROXY_ID] = _failed_condition(
                methods.FP32_PROXY_ID, exc
            )

        try:
            decay_memory = original_memory.fork()
            decay_memory.request = GateRequest(
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
            decay_memory.deleted_positions = tuple(
                record.context.forget_positions
            )
            decay_memory.deletion_kind = "coefficient_decay_0_01"
            row["conditions"][methods.DECAY_ID] = _execute_condition(
                runtime,
                decay_memory,
                probes,
                condition_id=methods.DECAY_ID,
                record=record,
                lock=lock,
                semantics={
                    "decay_factor": DECAY_FACTOR,
                    "positions_remain_resident": True,
                    "solver_refit": False,
                },
            )
        except Exception as exc:
            row["conditions"][methods.DECAY_ID] = _failed_condition(
                methods.DECAY_ID, exc
            )

        try:
            diagnostics, certificate_memories = (
                runtime.persistent_certificate_states(
                    original_memory,
                    record.context.forget_positions,
                )
            )
            if diagnostics.get("fixed_c_feasible") is not True:
                raise RuntimeError("fixed-C certificate reports infeasibility")
            if any(
                method_states._fallback_used(certificate_memories[name])
                for name in ("exact", "refit")
            ):
                raise RuntimeError("certificate state used full repack")
            used_refit_fallback = bool(
                diagnostics.get("used_refit_fallback")
                or int(diagnostics.get("decrement_fallbacks", 0)) > 0
            )
            row["conditions"][methods.EXACT_DECREMENT_ID] = _execute_condition(
                runtime,
                certificate_memories["exact"],
                probes,
                condition_id=methods.EXACT_DECREMENT_ID,
                record=record,
                lock=lock,
                semantics={
                    "executed_method": (
                        "fixed_c_refit_fallback"
                        if used_refit_fallback
                        else "incremental_float64_fixed_c_decrement"
                    ),
                    "fixed_C": True,
                    "full_repack_fallback": False,
                    "fixed_c_refit_fallback": used_refit_fallback,
                },
            )
            row["conditions"][methods.FIXED_C_REFIT_ID] = _execute_condition(
                runtime,
                certificate_memories["refit"],
                probes,
                condition_id=methods.FIXED_C_REFIT_ID,
                record=record,
                lock=lock,
                semantics={
                    "executed_method": (
                        "float64_fixed_c_retained_key_refit"
                    ),
                    "fixed_C": True,
                    "retained_keys_only": True,
                },
            )
        except Exception as exc:
            for condition_id in (
                methods.EXACT_DECREMENT_ID,
                methods.FIXED_C_REFIT_ID,
            ):
                row["conditions"][condition_id] = _failed_condition(
                    condition_id, exc
                )

        try:
            suppressed = method_states._prompt_suppression_probes(
                record, probes
            )
            row["conditions"][
                methods.PROMPT_SUPPRESSION_ID
            ] = _execute_condition(
                runtime,
                original_memory,
                suppressed,
                condition_id=methods.PROMPT_SUPPRESSION_ID,
                record=record,
                lock=lock,
                semantics={
                    "prompt_only_behavioral_control": True,
                    "persistent_state_deleted": False,
                    "valid_registered_gemma_chat": True,
                    "instruction_target_free": True,
                },
                prompt_is_suppressed=True,
            )
        except Exception as exc:
            row["conditions"][
                methods.PROMPT_SUPPRESSION_ID
            ] = _failed_condition(methods.PROMPT_SUPPRESSION_ID, exc)

        fingerprint_after = _memory_fingerprint(original_memory)
        row["source_state_immutability"] = {
            "before_sha256": fingerprint_before,
            "after_sha256": fingerprint_after,
            "unchanged": fingerprint_after == fingerprint_before,
        }
        if fingerprint_after != fingerprint_before:
            raise RuntimeError("generation audit mutated the source state")
    else:
        row["source_state_immutability"] = {
            "verified": False,
            "reason": "original_prefill_failed",
        }

    if set(row["conditions"]) != set(CONDITION_IDS):
        raise RuntimeError("response-generation condition matrix is incomplete")
    failures = [
        condition_id
        for condition_id in CONDITION_IDS
        if row["conditions"][condition_id]["status"]
        not in {"completed", "aliased"}
    ]
    row["failed_condition_ids"] = failures
    row["status"] = (
        "completed" if not failures else "completed_with_failures"
    )
    return row


def _terminal_failed_record(record_id: str, exc: Exception) -> dict[str, Any]:
    conditions = {
        condition_id: _failed_condition(condition_id, exc)
        for condition_id in CONDITION_IDS
    }
    conditions[methods.TOKEN_ROW_DIAGNOSTIC_ID] = _alias_condition(
        conditions[methods.FRESH_RAW_OMISSION_ID]
    )
    return {
        "record_id": record_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": _text_sha256(str(exc)),
        "conditions": conditions,
        "failed_condition_ids": [
            condition_id
            for condition_id in CONDITION_IDS
            if condition_id != methods.TOKEN_ROW_DIAGNOSTIC_ID
        ],
    }


def _placeholder_record(record_id: str) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "status": "not_attempted",
        "conditions": {
            condition_id: {
                "condition_id": condition_id,
                "status": "not_attempted",
                "probes": [
                    {
                        "probe_id": probe_id,
                        "status": "not_attempted",
                    }
                    for probe_id in PROBE_IDS
                ],
            }
            for condition_id in CONDITION_IDS
        },
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    condition_summary: dict[str, Any] = {}
    for condition_id in CONDITION_IDS:
        conditions = [row["conditions"][condition_id] for row in records]
        probe_rows = [
            probe for condition in conditions for probe in condition["probes"]
        ]
        condition_summary[condition_id] = {
            "records": len(conditions),
            "completed_records": sum(
                condition["status"] == "completed"
                for condition in conditions
            ),
            "aliased_records": sum(
                condition["status"] == "aliased"
                for condition in conditions
            ),
            "failed_or_incomplete_records": sum(
                condition["status"] not in {"completed", "aliased"}
                for condition in conditions
            ),
            "probe_slots": len(probe_rows),
            "completed_probes": sum(
                probe["status"] == "completed" for probe in probe_rows
            ),
            "aliased_probes": sum(
                probe["status"] == "aliased" for probe in probe_rows
            ),
            "failed_or_incomplete_probes": sum(
                probe["status"] not in {"completed", "aliased"}
                for probe in probe_rows
            ),
            "nondeterministic_probes": sum(
                probe["status"] == "failed_nondeterministic"
                for probe in probe_rows
            ),
            "additional_generation_executions": sum(
                int(condition.get("additional_generation_executions", 0))
                for condition in conditions
            ),
        }
    return {
        "authorized_records": EXPECTED_RECORDS,
        "record_slots": len(records),
        "attempted_records": sum(
            row["status"] != "not_attempted" for row in records
        ),
        "not_attempted_records": sum(
            row["status"] == "not_attempted" for row in records
        ),
        "completed_records": sum(
            row["status"] == "completed" for row in records
        ),
        "failed_or_incomplete_records": sum(
            row["status"] != "completed" for row in records
        ),
        "joint10_records": EXPECTED_JOINT_RECORDS,
        "conditions": condition_summary,
        "no_outcome_based_filtering": True,
    }


def _base_report(lock: Mapping[str, Any]) -> dict[str, Any]:
    records = [
        _placeholder_record(record_id)
        for record_id in lock["cohort"]["ordered_record_ids"]
    ]
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "running",
        "contains_source_text": True,
        "contains_model_generated_text": True,
        "source_bearing": True,
        "local_only": True,
        "release_authorized": False,
        "official_longmemeval_leaderboard_score": False,
        "authorization_lock_integrity_sha256": lock["integrity"]["sha256"],
        "source_final_report": {
            "file_sha256": FINAL_REPORT_FILE_SHA256,
            "mutated_or_overwritten": False,
            "generated_text_inferred_from_teacher_forced_scores": False,
        },
        "scope": {
            "label": "new post-hoc deterministic response-generation audit",
            "part_of_original_final_report": False,
            "part_of_original_certificate": False,
            "decoded_strings_exhaust_extractability": False,
        },
        "privacy_and_content_controls": {
            "public_longmemeval_provenance": copy.deepcopy(
                lock["privacy_and_scope"]["longmemeval"]
            ),
            "hidden_or_private_credentials_serialized": False,
            "environment_variables_serialized": False,
            "source_bearing_flag_explicit": True,
            "release_by_default": False,
            "leak_and_instruction_boundary_audit_per_response": True,
            "no_exhaustive_extractability_claim": True,
        },
        "runtime": copy.deepcopy(lock["runtime"]),
        "prompt_contract_sha256": _payload_sha256(lock["prompt_contract"]),
        "generation": copy.deepcopy(lock["generation"]),
        "conditions": copy.deepcopy(lock["conditions"]),
        "attempt_policy": copy.deepcopy(lock["attempt_policy"]),
        "records": records,
        "summary": _summary(records),
    }


def _seal_report(report: Mapping[str, Any]) -> dict[str, Any]:
    sealed = copy.deepcopy(dict(report))
    sealed.pop("integrity", None)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(sealed),
    }
    return sealed


def run_rehydrated_audit(
    runtime: Any,
    records: Sequence[Any],
    manifest: Mapping[str, Any],
    method_lock: Mapping[str, Any],
    authorization_lock: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ids = authorization_lock["cohort"]["ordered_record_ids"]
    observed_ids = [str(getattr(record, "record_id", "")) for record in records]
    if observed_ids != expected_ids or len(records) != EXPECTED_RECORDS:
        raise AuditError("rehydrated cohort differs from ordered all16")
    public_rows = manifest.get("records")
    bindings = method_lock["authorization"]["admission_records"]
    if (
        not isinstance(public_rows, list)
        or len(public_rows) != EXPECTED_RECORDS
        or [row["record_id"] for row in bindings] != expected_ids
    ):
        raise AuditError("runtime cohort bindings differ")

    report = _base_report(authorization_lock)
    started = time.perf_counter()
    output_rows = []
    for record, public, binding in zip(records, public_rows, bindings):
        try:
            row = _audit_record(
                runtime,
                record,
                public,
                admission_binding=binding,
                lock=authorization_lock,
            )
        except Exception as exc:
            row = _terminal_failed_record(record.record_id, exc)
        output_rows.append(row)
    report["records"] = output_rows
    report["summary"] = _summary(output_rows)
    report["status"] = (
        "completed"
        if all(row["status"] == "completed" for row in output_rows)
        else "completed_with_failures"
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    sealed = _seal_report(report)
    validate_audit_report(sealed, lock=authorization_lock)
    return sealed


def _validate_probe_result(
    probe: Mapping[str, Any],
    *,
    expected_probe_id: str,
) -> None:
    if probe.get("probe_id") != expected_probe_id:
        raise AuditError("audit probe ID differs")
    status = probe.get("status")
    if status == "completed":
        repeat = probe.get("repeat_check")
        if (
            not _generation_artifact_is_consistent(probe)
            or not isinstance(repeat, Mapping)
            or repeat.get("exact_token_ids_and_response_text_match") is not True
            or repeat.get("repeat_generated_token_ids_sha256")
            != probe.get("generated_token_ids_sha256")
            or repeat.get("repeat_response_utf8_sha256")
            != probe.get("response_utf8_sha256")
            or not isinstance(probe.get("content_audit"), Mapping)
            or (probe["content_audit"]).get(
                "decoded_string_exhausts_extractability"
            )
            is not False
        ):
            raise AuditError("completed response artifact is inconsistent")
    elif status == "failed_nondeterministic":
        repeat = probe.get("repeat_check")
        if (
            not _generation_artifact_is_consistent(probe)
            or not isinstance(repeat, Mapping)
            or repeat.get("exact_token_ids_and_response_text_match") is not False
            or not isinstance(repeat.get("repeat_response_text"), str)
            or not isinstance(repeat.get("repeat_generated_token_ids"), list)
            or repeat.get("repeat_response_utf8_sha256")
            != _text_sha256(repeat["repeat_response_text"])
            or repeat.get("repeat_generated_token_ids_sha256")
            != _payload_sha256(repeat["repeat_generated_token_ids"])
            or repeat.get("repeat_generated_token_count")
            != len(repeat["repeat_generated_token_ids"])
        ):
            raise AuditError("nondeterminism failure disclosure is incomplete")
    elif status == "failed":
        completed_attempts = probe.get("completed_generation_attempts", [])
        if (
            not isinstance(probe.get("error_type"), str)
            or not _is_sha256(probe.get("error_message_sha256"))
            or probe.get("error_message_redacted") is not True
            or type(probe.get("generation_attempts_started")) is not int
            or type(probe.get("generation_attempts_completed")) is not int
            or not isinstance(completed_attempts, list)
            or probe.get("generation_attempts_completed")
            != len(completed_attempts)
            or probe.get("generation_attempts_completed")
            > probe.get("generation_attempts_started")
            or probe.get("generation_attempts_started") > GENERATION_REPEATS
            or any(
                not _generation_artifact_is_consistent(artifact)
                for artifact in completed_attempts
            )
        ):
            raise AuditError("failed probe disclosure is incomplete")
    elif status not in {"aliased", "not_attempted"}:
        raise AuditError("unsupported audit probe status")


def _generation_artifact_is_consistent(value: Mapping[str, Any]) -> bool:
    token_ids = value.get("generated_token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            type(token_id) is not int or token_id < 0
            for token_id in token_ids
        )
        or len(token_ids) > MAX_NEW_TOKENS
        or value.get("generated_token_count") != len(token_ids)
        or value.get("generated_token_ids_sha256")
        != _payload_sha256(token_ids)
        or not isinstance(value.get("response_text"), str)
        or value.get("response_utf8_sha256")
        != _text_sha256(value["response_text"])
        or value.get("stop_reason")
        not in {"terminal_stop_token", "max_new_tokens"}
    ):
        return False
    terminal = token_ids[-1] if token_ids[-1] in STOP_TOKEN_IDS else None
    if terminal is None:
        return (
            value.get("stop_reason") == "max_new_tokens"
            and value.get("stop_token_id") is None
            and len(token_ids) == MAX_NEW_TOKENS
        )
    return (
        value.get("stop_reason") == "terminal_stop_token"
        and value.get("stop_token_id") == terminal
    )


def validate_audit_report(
    report: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
) -> None:
    _assert_no_credential_fields(report)
    _validate_integrity(report, name="response-generation report")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("status")
        not in {"completed", "completed_with_failures", "failed"}
        or report.get("contains_source_text") is not True
        or report.get("contains_model_generated_text") is not True
        or report.get("source_bearing") is not True
        or report.get("release_authorized") is not False
        or report.get("authorization_lock_integrity_sha256")
        != lock["integrity"]["sha256"]
        or report.get("runtime") != lock["runtime"]
        or report.get("generation") != lock["generation"]
        or report.get("conditions") != lock["conditions"]
        or report.get("attempt_policy") != lock["attempt_policy"]
        or report.get("prompt_contract_sha256")
        != _payload_sha256(lock["prompt_contract"])
        or (report.get("source_final_report") or {}).get("file_sha256")
        != FINAL_REPORT_FILE_SHA256
        or (report.get("source_final_report") or {}).get(
            "generated_text_inferred_from_teacher_forced_scores"
        )
        is not False
        or (report.get("scope") or {}).get(
            "decoded_strings_exhaust_extractability"
        )
        is not False
        or not isinstance(report.get("elapsed_seconds"), (int, float))
        or isinstance(report.get("elapsed_seconds"), bool)
        or not math.isfinite(float(report["elapsed_seconds"]))
        or float(report["elapsed_seconds"]) < 0.0
    ):
        raise AuditError("response-generation report contract differs")
    rows = report.get("records")
    expected_ids = lock["cohort"]["ordered_record_ids"]
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_RECORDS
        or [row.get("record_id") for row in rows] != expected_ids
    ):
        raise AuditError("response-generation report rows differ from all16")
    for row in rows:
        conditions = row.get("conditions")
        if (
            not isinstance(conditions, Mapping)
            or list(conditions) != list(CONDITION_IDS)
        ):
            raise AuditError("response-generation condition order differs")
        non_alias_failures: list[str] = []
        for condition_id in CONDITION_IDS:
            condition = conditions[condition_id]
            probes = condition.get("probes")
            if (
                condition.get("condition_id") != condition_id
                or not isinstance(probes, list)
                or [probe.get("probe_id") for probe in probes]
                != list(PROBE_IDS)
            ):
                raise AuditError("condition probe matrix differs")
            if condition_id == methods.TOKEN_ROW_DIAGNOSTIC_ID:
                alias_is_valid = bool(
                    condition.get("status") == "aliased"
                    and condition.get("alias_of")
                    == methods.FRESH_RAW_OMISSION_ID
                    and condition.get("additional_generation_executions") == 0
                    and all(
                        probe.get("status") == "aliased" for probe in probes
                    )
                )
                not_attempted_is_valid = bool(
                    row.get("status") == "not_attempted"
                    and condition.get("status") == "not_attempted"
                    and all(
                        probe.get("status") == "not_attempted"
                        for probe in probes
                    )
                )
                if not (alias_is_valid or not_attempted_is_valid):
                    raise AuditError("token-row alias contract differs")
            else:
                for probe, probe_id in zip(probes, PROBE_IDS):
                    _validate_probe_result(
                        probe,
                        expected_probe_id=probe_id,
                    )
                expected_condition_status = (
                    "not_attempted"
                    if all(
                        probe.get("status") == "not_attempted"
                        for probe in probes
                    )
                    else (
                        "completed"
                        if all(
                            probe.get("status") == "completed"
                            for probe in probes
                        )
                        else (
                            "failed"
                            if all(
                                probe.get("status") == "failed"
                                and probe.get("attempted") is False
                                for probe in probes
                            )
                            else "completed_with_probe_failures"
                        )
                    )
                )
                if condition.get("status") != expected_condition_status:
                    raise AuditError("condition status contradicts its probes")
                expected_executions = sum(
                    int(probe.get("generation_attempts_started", 0))
                    for probe in probes
                )
                if int(
                    condition.get("additional_generation_executions", 0)
                ) != expected_executions:
                    raise AuditError(
                        "condition generation-attempt accounting differs"
                    )
                if condition.get("status") not in {
                    "completed",
                    "not_attempted",
                }:
                    non_alias_failures.append(condition_id)
        if row.get("status") == "not_attempted":
            if non_alias_failures or any(
                condition.get("status") != "not_attempted"
                for condition_id, condition in conditions.items()
                if condition_id != methods.TOKEN_ROW_DIAGNOSTIC_ID
            ):
                raise AuditError("not-attempted row contains attempted conditions")
        elif not non_alias_failures:
            if row.get("status") != "completed":
                raise AuditError("successful row status differs")
        elif row.get("status") not in {"completed_with_failures", "failed"}:
            raise AuditError("failed row status differs")
        if row.get("status") != "not_attempted" and row.get(
            "failed_condition_ids"
        ) != non_alias_failures:
            raise AuditError("row failed-condition accounting differs")
        alias = conditions[methods.TOKEN_ROW_DIAGNOSTIC_ID]
        if alias.get("status") == "aliased":
            raw_by_probe = {
                probe["probe_id"]: probe["status"]
                for probe in conditions[
                    methods.FRESH_RAW_OMISSION_ID
                ]["probes"]
            }
            for probe in alias["probes"]:
                if (
                    probe.get("aliased_source_status")
                    != raw_by_probe[probe["probe_id"]]
                ):
                    raise AuditError("token-row alias source status differs")
    if report.get("summary") != _summary(rows):
        raise AuditError("response-generation summary does not recompute")
    expected_status = (
        "completed"
        if all(row.get("status") == "completed" for row in rows)
        else (
            "failed"
            if all(row.get("status") == "not_attempted" for row in rows)
            else "completed_with_failures"
        )
    )
    if report.get("status") != expected_status:
        raise AuditError("response-generation report status is contradictory")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_no_alias(
    output: str | Path,
    *,
    inputs: Sequence[str | Path | None],
) -> Path:
    output_path = Path(output).expanduser()
    resolved = _resolved(output_path)
    protected = [
        Path(path).expanduser()
        for path in (*inputs, *_IMPLEMENTATION_PATHS.values())
        if path is not None
    ]
    for item in protected:
        same = resolved == _resolved(item)
        if output_path.exists() and item.exists():
            try:
                same = same or os.path.samefile(output_path, item)
            except OSError:
                pass
        if same:
            raise AuditError("source-bearing output aliases a protected input")
    return resolved


def _atomic_write_new(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
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
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{destination} already exists; overwrite is forbidden"
            ) from exc
        try:
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
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


def freeze_authorization_lock_file(
    *,
    final_report_path: str | Path = DEFAULT_FINAL_REPORT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    method_lock_path: str | Path = DEFAULT_METHOD_LOCK,
    core_manifest_path: str | Path = DEFAULT_CORE_MANIFEST,
    lock_path: str | Path = DEFAULT_AUTHORIZATION_LOCK,
) -> dict[str, Any]:
    destination = _require_exact_canonical_path(
        lock_path,
        DEFAULT_AUTHORIZATION_LOCK,
        name="response-generation authorization lock",
    )
    if destination.exists():
        raise FileExistsError("canonical authorization lock already exists")
    final_report, manifest, method_lock = _load_protocol_inputs(
        final_report_path=final_report_path,
        manifest_path=manifest_path,
        method_lock_path=method_lock_path,
        core_manifest_path=core_manifest_path,
    )
    lock = freeze_authorization_lock(final_report, manifest, method_lock)
    _atomic_write_new(destination, lock)
    return lock


def _claim_attempt(lock: Mapping[str, Any]) -> Path:
    claim = _require_exact_canonical_path(
        DEFAULT_ATTEMPT_CLAIM,
        DEFAULT_ATTEMPT_CLAIM,
        name="response-generation attempt claim",
    )
    try:
        os.mkdir(claim, 0o700)
    except FileExistsError as exc:
        raise PermissionError(
            "the single response-generation attempt is already claimed"
        ) from exc
    marker = {
        "schema": (
            "gemma-sv-longmemeval-chat-response-generation-attempt-v1"
        ),
        "authorization_lock_integrity_sha256": lock["integrity"]["sha256"],
        "final_report_file_sha256": FINAL_REPORT_FILE_SHA256,
        "ordered_record_ids_sha256": lock["cohort"][
            "ordered_record_ids_sha256"
        ],
        "output_workspace_path": lock["output_policy"][
            "canonical_workspace_path"
        ],
        "attempt_number": 1,
        "persistent": True,
        "resume_allowed": False,
    }
    _atomic_write_new(claim / ATTEMPT_MARKER_FILENAME, marker)
    return claim


def _snapshot_files(paths: Mapping[str, str | Path]) -> dict[str, str]:
    return {
        name: _file_sha256(Path(path).expanduser())
        for name, path in sorted(paths.items())
    }


def _fatal_report(
    lock: Mapping[str, Any],
    exc: Exception,
    *,
    elapsed_seconds: float,
) -> dict[str, Any]:
    report = _base_report(lock)
    report["status"] = "failed"
    report["fatal_error"] = {
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": _text_sha256(str(exc)),
        "all_record_condition_probe_slots_preserved": True,
    }
    report["elapsed_seconds"] = float(elapsed_seconds)
    return _seal_report(report)


def run_audit_from_paths(
    *,
    final_report_path: str | Path = DEFAULT_FINAL_REPORT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    method_lock_path: str | Path = DEFAULT_METHOD_LOCK,
    core_manifest_path: str | Path = DEFAULT_CORE_MANIFEST,
    authorization_lock_path: str | Path = DEFAULT_AUTHORIZATION_LOCK,
    data_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
    explicit_acknowledgement: Any,
) -> dict[str, Any]:
    if explicit_acknowledgement is not True:
        raise PermissionError(
            "response-generation audit acknowledgement must be exactly True"
        )
    output = _require_exact_canonical_path(
        output_path,
        DEFAULT_OUTPUT,
        name="response-generation report",
    )
    if output.exists():
        raise FileExistsError(
            "response-generation report already exists; resume and overwrite "
            "are forbidden"
        )
    final_report, manifest, method_lock = _load_protocol_inputs(
        final_report_path=final_report_path,
        manifest_path=manifest_path,
        method_lock_path=method_lock_path,
        core_manifest_path=core_manifest_path,
    )
    lock = load_committed_authorization_lock(
        authorization_lock_path,
        final_report=final_report,
        manifest=manifest,
        method_lock=method_lock,
    )
    verify_runtime_environment(lock)
    output = _validate_no_alias(
        output,
        inputs=(
            final_report_path,
            manifest_path,
            method_lock_path,
            core_manifest_path,
            authorization_lock_path,
            data_path,
        ),
    )
    _claim_attempt(lock)
    protected_paths: dict[str, str | Path] = {
        "final_report": final_report_path,
        "manifest": manifest_path,
        "method_lock": method_lock_path,
        "core_manifest": core_manifest_path,
        "authorization_lock": authorization_lock_path,
    }
    if data_path is not None:
        protected_paths["data"] = data_path
    before = _snapshot_files(protected_paths)
    started = time.perf_counter()
    try:
        _configure_determinism()
        with _offline_huggingface():
            _verify_cached_model_metadata(lock)
            rows = geometry.base.load_pinned_longmemeval_rows(data_path)
            runtime = methods.admission._make_runtime(device="mps")
            runtime.ensure_loaded()
            verify_loaded_generation_runtime(runtime, lock=lock)
            corrected = geometry.rehydrate_manifest(
                manifest,
                rows,
                runtime.tokenizer,
                core_manifest=core_manifest_path,
            )
            records = tuple(item.runtime for item in corrected)
            report = run_rehydrated_audit(
                runtime,
                records,
                manifest,
                method_lock,
                lock,
            )
    except Exception as exc:
        report = _fatal_report(
            lock,
            exc,
            elapsed_seconds=time.perf_counter() - started,
        )
    after = _snapshot_files(protected_paths)
    if before != after:
        raise RuntimeError("a protected audit input changed during execution")
    validate_audit_report(report, lock=lock)
    _atomic_write_new(output, report)
    return report


def build_presentation_projection(
    report: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    validate_audit_report(report, lock=lock)
    case = lock.get("presentation_case") or {}
    if (
        case.get("record_id") != PRESENTATION_CASE_RECORD_ID
        or case.get("authorized_position") != PRESENTATION_CASE_POSITION
        or case.get("selected_in_this_pre_run_lock") is not True
        or case.get("selected_before_generated_outcomes") is not True
    ):
        raise AuditError("presentation case lacks pre-run selection proof")
    matches = [
        row
        for row in report["records"]
        if row["record_id"] == PRESENTATION_CASE_RECORD_ID
    ]
    if len(matches) != 1:
        raise AuditError("preselected presentation case is unavailable")
    projection: dict[str, Any] = {
        "schema": PROJECTION_SCHEMA,
        "schema_version": 1,
        "status": "local-source-bearing-projection",
        "label": "new post-hoc deterministic response-generation audit",
        "contains_source_text": True,
        "contains_model_generated_text": True,
        "source_bearing": True,
        "local_only": True,
        "release_authorized": False,
        "part_of_original_final_report": False,
        "part_of_original_certificate": False,
        "decoded_strings_exhaust_extractability": False,
        "preselection_proof": {
            "authorization_lock_integrity_sha256": lock["integrity"]["sha256"],
            "lock_must_have_been_committed_before_run": True,
            "record_id": PRESENTATION_CASE_RECORD_ID,
            "authorized_position": PRESENTATION_CASE_POSITION,
            "selected_before_generated_outcomes": True,
        },
        "source_audit_report_integrity_sha256": report["integrity"]["sha256"],
        "case": copy.deepcopy(matches[0]),
    }
    projection["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(projection),
    }
    _assert_no_credential_fields(projection)
    return projection


def project_presentation_case_from_paths(
    *,
    report_path: str | Path = DEFAULT_OUTPUT,
    projection_output_path: str | Path = DEFAULT_PROJECTION_OUTPUT,
    explicit_acknowledgement: Any,
) -> dict[str, Any]:
    if explicit_acknowledgement is not True:
        raise PermissionError(
            "source-bearing case projection acknowledgement must be exactly True"
        )
    final_report, manifest, method_lock = _load_protocol_inputs()
    lock = load_committed_authorization_lock(
        DEFAULT_AUTHORIZATION_LOCK,
        final_report=final_report,
        manifest=manifest,
        method_lock=method_lock,
    )
    report_file = _require_exact_canonical_path(
        report_path,
        DEFAULT_OUTPUT,
        name="response-generation report",
    )
    destination = _require_exact_canonical_path(
        projection_output_path,
        DEFAULT_PROJECTION_OUTPUT,
        name="presentation projection",
    )
    if destination.exists():
        raise FileExistsError(
            "presentation projection already exists; overwrite is forbidden"
        )
    projection = build_presentation_projection(
        _load_mapping(report_file, name="response-generation report"),
        lock=lock,
    )
    _atomic_write_new(destination, projection)
    return projection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--freeze-lock",
        action="store_true",
        help="write the source-free pre-run lock; performs no model loading",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="run the one-attempt 4B response-generation audit",
    )
    mode.add_argument(
        "--project-case",
        action="store_true",
        help="project only the case preselected in the committed pre-run lock",
    )
    parser.add_argument("--data-path")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--projection-out",
        default=str(DEFAULT_PROJECTION_OUTPUT),
    )
    parser.add_argument("--allow-generation-audit", action="store_true")
    parser.add_argument(
        "--allow-source-bearing-projection",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.freeze_lock:
            freeze_authorization_lock_file()
        elif args.run:
            report = run_audit_from_paths(
                data_path=args.data_path,
                output_path=args.out,
                explicit_acknowledgement=args.allow_generation_audit,
            )
            return 0 if report["status"] == "completed" else 1
        else:
            project_presentation_case_from_paths(
                report_path=args.out,
                projection_output_path=args.projection_out,
                explicit_acknowledgement=(
                    args.allow_source_bearing_projection
                ),
            )
    except (
        AuditError,
        FileExistsError,
        OSError,
        PermissionError,
        RuntimeError,
        geometry.ManifestError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

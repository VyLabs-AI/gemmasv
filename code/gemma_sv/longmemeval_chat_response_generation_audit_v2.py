"""Resumable local-only decoded-response audit for the frozen v3 cohort.

The durable unit is one complete frozen history.  Evidence is immutable and
source-bearing; ``heartbeat.json`` alone is mutable and never contains prompts,
answers, token IDs, or responses.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, nullcontext
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import resource
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat_methods as method_states
from gemma_sv import eval_longmemeval_chat_v2 as runtime_v2
from gemma_sv import longmemeval_chat_cohort_v3 as cohort_v3
from gemma_sv import longmemeval_chat_response_generation_audit as decoded_v1


LOCK_SCHEMA = "gemma-sv-longmemeval-chat-response-authorization-v2"
RUN_SCHEMA = "gemma-sv-longmemeval-chat-response-run-v2"
SHARD_SCHEMA = "gemma-sv-longmemeval-chat-response-record-v2"
FINAL_SCHEMA = "gemma-sv-longmemeval-chat-response-final-v2"
ATTEMPT_SCHEMA = "gemma-sv-longmemeval-chat-response-attempt-v2"
SCHEMA_VERSION = 2

EXPECTED_CLUSTERS = 32
EXPECTED_HISTORIES = 96
PROBE_IDS = ("target_current", "retained")
CONDITION_IDS = (
    "present",
    "fresh_raw_omission",
    "exact_decrement_or_refit_policy",
    "prompt_suppression",
)
GENERATION_REPEATS = 2
MAX_NEW_TOKENS = 64
STOP_TOKEN_IDS = (1, 106)
EXPECTED_GENERATION_CALLS = 1536
MAXIMUM_GENERATED_TOKEN_STEPS = 98_304
DEFAULT_CERTIFICATE_WORKERS = 8
MIN_CERTIFICATE_WORKERS = 1
MAX_CERTIFICATE_WORKERS = 16
HEARTBEAT_INTERVAL_SECONDS = 30.0
EXPLICIT_ACKNOWLEDGEMENT = (
    "I_ACKNOWLEDGE_SOURCE_BEARING_LOCAL_ONLY_V2_RESPONSE_GENERATION"
)

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_COHORT = BENCHMARKS / "longmemeval_chat_cohort_v3.json"
DEFAULT_CLUSTER_LOCK = (
    BENCHMARKS / "longmemeval_chat_cluster_analysis_lock_v3.json"
)
DEFAULT_CENSUS = BENCHMARKS / "longmemeval_chat_cohort_census_v3.json"
DEFAULT_AUTHORIZATION_LOCK = (
    BENCHMARKS
    / "longmemeval_chat_response_generation_authorization_v2.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_response_generation_audit_v2"
)

_IMPLEMENTATION_PATHS = {
    "gemma_sv/longmemeval_chat_response_generation_audit_v2.py": (
        Path(__file__).resolve()
    ),
    "gemma_sv/demo_server/audit_gemma_runtime_v2.py": (
        PACKAGE / "demo_server" / "audit_gemma_runtime_v2.py"
    ),
    "gemma_sv/parallel_certificate_v2.py": (
        PACKAGE / "parallel_certificate_v2.py"
    ),
    "gemma_sv/longmemeval_chat_cohort_v3.py": (
        PACKAGE / "longmemeval_chat_cohort_v3.py"
    ),
    "gemma_sv/longmemeval_chat_response_generation_audit.py": (
        PACKAGE / "longmemeval_chat_response_generation_audit.py"
    ),
    "gemma_sv/eval_longmemeval_chat_methods.py": (
        PACKAGE / "eval_longmemeval_chat_methods.py"
    ),
    "gemma_sv/eval_longmemeval_chat_v2.py": (
        PACKAGE / "eval_longmemeval_chat_v2.py"
    ),
    "gemma_sv/demo_server/gemma_engine.py": (
        PACKAGE / "demo_server" / "gemma_engine.py"
    ),
    "gemma_sv/demo_server/certificate.py": (
        PACKAGE / "demo_server" / "certificate.py"
    ),
    "cp_svm/oneclass_fast.py": WORKSPACE / "cp_svm" / "oneclass_fast.py",
    "cp_svm/oneclass_incremental.py": (
        WORKSPACE / "cp_svm" / "oneclass_incremental.py"
    ),
    "cp_svm/kernels.py": WORKSPACE / "cp_svm" / "kernels.py",
}

_SOURCE_FREE_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "content",
        "messages",
        "prompt",
        "question",
        "response_text",
        "sessions",
        "source_text",
        "turns",
    }
)
_HEARTBEAT_FORBIDDEN_FRAGMENTS = (
    "answer",
    "prompt",
    "question",
    "response",
    "token",
    "content",
    "text",
)
_HEARTBEAT_WRITE_LOCK = threading.RLock()
_RUNTIME_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "clarabel": "clarabel",
    "cvxpy": "cvxpy",
    "huggingface-hub": "huggingface-hub",
    "mlx": "mlx",
    "numpy": "numpy",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
}
_MPS_CONTROL_NAMES = (
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)
_BLAS_CONTROL_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class AuditV2Error(ValueError):
    """A v2 lock, ledger, shard, or final artifact violated its contract."""


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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditV2Error(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AuditV2Error(f"non-finite JSON constant {value!r}")


def _load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditV2Error(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise AuditV2Error(f"{name} must be a JSON object")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(result),
    }
    return result


def _validate_seal(value: Mapping[str, Any], *, name: str) -> None:
    body = copy.deepcopy(dict(value))
    integrity = body.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(body)
    ):
        raise AuditV2Error(f"{name} integrity differs")


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).casefold()
            yield from _walk_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_keys(child)


def _assert_source_free_lock(lock: Mapping[str, Any]) -> None:
    leaked = _SOURCE_FREE_FORBIDDEN_KEYS.intersection(_walk_keys(lock))
    if leaked:
        raise AuditV2Error(
            "authorization lock contains source-bearing keys: "
            + ", ".join(sorted(leaked))
        )
    if (
        lock.get("contains_source_text") is not False
        or lock.get("contains_model_outputs") is not False
    ):
        raise AuditV2Error("authorization lock is not explicitly source-free")


def _artifact_binding(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    integrity = value.get("integrity")
    return {
        "repository_path": path.relative_to(WORKSPACE).as_posix(),
        "file_sha256": _file_sha256(path),
        "payload_sha256": _payload_sha256(value),
        "integrity_sha256": (
            None
            if not isinstance(integrity, Mapping)
            else integrity.get("sha256")
        ),
        "lock_sha256": value.get("lock_sha256"),
        "immutable": True,
    }


def _implementation_fingerprints() -> dict[str, Any]:
    files = {
        name: _file_sha256(path)
        for name, path in sorted(_IMPLEMENTATION_PATHS.items())
    }
    return {
        "files": files,
        "file_count": len(files),
        "files_sha256": _payload_sha256(files),
    }


def _observed_runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for label, distribution in _RUNTIME_DISTRIBUTIONS.items():
        try:
            packages[label] = package_version(distribution)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"required package {distribution!r} is unavailable"
            ) from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for environment binding") from exc
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
        },
        "packages": packages,
        "torch": {
            "version": str(torch.__version__),
            "git_version": str(torch.version.git_version),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "controls": {
            "evaluation_seed": decoded_v1.EVALUATION_SEED,
            "deterministic_algorithms_required_for_run": True,
            "deterministic_warn_only": False,
            "mps_environment": {
                name: os.environ.get(name) for name in _MPS_CONTROL_NAMES
            },
            "blas_environment_at_freeze": {
                name: os.environ.get(name) for name in _BLAS_CONTROL_NAMES
            },
            "certificate_child_blas_threads": 1,
            "certificate_child_blas_variables": list(_BLAS_CONTROL_NAMES),
        },
    }


def _validate_frozen_inputs(
    cohort: Mapping[str, Any],
    cluster_lock: Mapping[str, Any],
    census: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    cohort_v3.validate_manifest(cohort)
    cohort_v3.validate_policy_lock(cluster_lock)
    cohort_v3.validate_census(census, cohort, cluster_lock)
    if cohort.get("policy_lock") != cluster_lock:
        raise AuditV2Error("standalone cluster lock differs from cohort")
    history_ids = [str(row["record_id"]) for row in cohort["histories"]]
    cluster_ids = [str(row["cluster_id"]) for row in cohort["clusters"]]
    if (
        len(history_ids) != EXPECTED_HISTORIES
        or len(set(history_ids)) != EXPECTED_HISTORIES
        or history_ids != census.get("ordered_history_ids")
        or len(cluster_ids) != EXPECTED_CLUSTERS
        or cluster_ids != census.get("ordered_cluster_ids")
    ):
        raise AuditV2Error("frozen v3 ordering or counts differ")
    return history_ids, cluster_ids


def freeze_authorization_lock(
    cohort: Mapping[str, Any],
    cluster_lock: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    cohort_path: Path = DEFAULT_COHORT,
    cluster_lock_path: Path = DEFAULT_CLUSTER_LOCK,
    census_path: Path = DEFAULT_CENSUS,
) -> dict[str, Any]:
    """Build the source-free lock without importing model libraries."""

    history_ids, cluster_ids = _validate_frozen_inputs(
        cohort,
        cluster_lock,
        census,
    )
    probe_bindings = [
        {
            "history_id": history["record_id"],
            "cluster_id": history["cluster_id"],
            "variant_index": int(history["variant_index"]),
            "history_integrity_sha256": history["record_integrity"]["sha256"],
            "probes": copy.deepcopy(history["probes"]),
        }
        for history in cohort["histories"]
    ]
    runtime_contract = runtime_v2.runtime_contract(device="mps")
    lock = {
        "schema": LOCK_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "frozen-before-v2-actual-response-generation",
        "contains_source_text": False,
        "contains_model_outputs": False,
        "source_bearing_local_output_authorized": True,
        "artifacts": {
            "cohort": _artifact_binding(cohort_path, cohort),
            "cluster_analysis_lock": _artifact_binding(
                cluster_lock_path,
                cluster_lock,
            ),
            "census": _artifact_binding(census_path, census),
        },
        "cohort": {
            "target_clusters": EXPECTED_CLUSTERS,
            "histories_per_cluster": 3,
            "history_instances": EXPECTED_HISTORIES,
            "analysis_n": EXPECTED_CLUSTERS,
            "ordered_cluster_ids": cluster_ids,
            "ordered_cluster_ids_sha256": _payload_sha256(cluster_ids),
            "ordered_history_ids": history_ids,
            "ordered_history_ids_sha256": _payload_sha256(history_ids),
            "all_histories_rehydrated": True,
            "admission_or_output_filtering": False,
            "replacement_allowed": False,
        },
        "prompt_answer_bindings": {
            "histories": probe_bindings,
            "sha256": _payload_sha256(probe_bindings),
            "source_values_in_lock": False,
        },
        "runtime": {
            "contract": runtime_contract,
            "model_id": runtime_contract["model_id"],
            "model_revision": runtime_contract["model_revision"],
            "tokenizer_id": cohort["tokenizer"]["model_id"],
            "tokenizer_revision": cohort["tokenizer"]["revision"],
            "one_model_runtime": True,
            "accelerator_execution": "serialized",
            "local_files_only": True,
            "network_access": False,
            "environment": _observed_runtime_environment(),
        },
        "matrix": {
            "condition_ids": list(CONDITION_IDS),
            "probe_ids": list(PROBE_IDS),
            "genuine_greedy_repeats": GENERATION_REPEATS,
            "conditions_executed_per_history": len(CONDITION_IDS),
            "excluded_decoded_conditions": [
                "fp32_proxy",
                "decay_0_01",
                "token_row_repack_alias",
                "separate_fixed_c_refit",
            ],
            "excluded_conditions_remain": "teacher-forced or other evidence",
            "exact_policy_decodes_only_executed_state": True,
            "persistent_state_materialization": {
                "states_constructed": ["exact_policy"],
                "refit_coefficients_used_inside_solver": True,
                "refit_persistent_state_constructed": False,
                "decay_persistent_state_constructed": False,
            },
        },
        "generation": {
            "algorithm": "direct-step deterministic greedy argmax",
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "stop_token_ids": list(STOP_TOKEN_IDS),
            "response_text_stored": True,
            "generated_token_ids_stored": True,
            "repeat_equality_required_for_completed_probe": True,
            "planned_workload": {
                "histories": EXPECTED_HISTORIES,
                "conditions": len(CONDITION_IDS),
                "probes": len(PROBE_IDS),
                "repeats": GENERATION_REPEATS,
                "generation_calls": EXPECTED_GENERATION_CALLS,
                "maximum_generated_token_steps": (
                    MAXIMUM_GENERATED_TOKEN_STEPS
                ),
            },
        },
        "certificate_workers": {
            "backend": "process",
            "start_method": "spawn",
            "minimum": MIN_CERTIFICATE_WORKERS,
            "maximum": MAX_CERTIFICATE_WORKERS,
            "default": DEFAULT_CERTIFICATE_WORKERS,
            "unit": "layer_head",
            "canonical_merge": True,
            "model_objects_enter_workers": False,
        },
        "resumability": {
            "durable_unit": "one_complete_history",
            "run_file": "run.json",
            "attempt_started_pattern": (
                "attempts/<slot>/<attempt>-started.json"
            ),
            "attempt_terminal_pattern": (
                "attempts/<slot>/<attempt>-terminal.json"
            ),
            "record_pattern": "records/<slot>.json",
            "mutable_non_evidence": "heartbeat.json",
            "final_file": "final.json",
            "sealed_terminal_records_never_retried": True,
            "unsealed_started_attempts_retained_before_retry": True,
            "canonical_assembly_by_frozen_history_order": True,
            "no_overwrite": True,
            "race_safe_first_writer": True,
        },
        "output": {
            "canonical_workspace_root": DEFAULT_OUTPUT_ROOT.relative_to(
                WORKSPACE
            ).as_posix(),
            "source_bearing": True,
            "contains_model_generated_text": True,
            "local_only": True,
            "release_authorized": False,
            "evidence_file_mode": "0600",
            "directory_mode": "0700",
            "decoded_strings_exhaust_extractability": False,
        },
        "implementation": _implementation_fingerprints(),
    }
    sealed = _seal(lock)
    _assert_source_free_lock(sealed)
    return sealed


def validate_authorization_lock(
    lock: Mapping[str, Any],
    *,
    cohort: Mapping[str, Any],
    cluster_lock: Mapping[str, Any],
    census: Mapping[str, Any],
) -> None:
    _validate_seal(lock, name="authorization lock")
    _assert_source_free_lock(lock)
    expected = freeze_authorization_lock(cohort, cluster_lock, census)
    if dict(lock) != expected:
        raise AuditV2Error("authorization lock differs from frozen inputs")


def _atomic_write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.parent.is_symlink():
        raise AuditV2Error("evidence parent directory must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=False,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
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
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{path} already exists; overwrite is forbidden"
            ) from exc
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_heartbeat(path: Path, value: Mapping[str, Any]) -> None:
    for key in _walk_keys(value):
        if any(fragment in key for fragment in _HEARTBEAT_FORBIDDEN_FRAGMENTS):
            raise AuditV2Error("heartbeat contains prohibited content-bearing key")
    with _HEARTBEAT_WRITE_LOCK:
        if path.parent.is_symlink() or path.is_symlink():
            raise AuditV2Error("heartbeat path must not be a symlink")
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _validate_permissions(path: Path, *, directory: bool) -> None:
    expected = 0o700 if directory else 0o600
    if path.is_symlink():
        raise AuditV2Error(f"{path.name} must not be a symlink")
    if _mode(path) != expected:
        raise AuditV2Error(
            f"{path.name} permissions differ from {expected:04o}"
        )


def _validate_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("certificate workers must be an integer")
    if not MIN_CERTIFICATE_WORKERS <= value <= MAX_CERTIFICATE_WORKERS:
        raise ValueError("certificate workers must be in [1, 16]")
    return value


def _safe_cli_output_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve(strict=False)
    allowed = (WORKSPACE / "outputs").resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise PermissionError("output root must be inside workspace outputs") from exc
    if root == allowed:
        raise PermissionError("output root must be a child of workspace outputs")
    if root.is_symlink():
        raise PermissionError("output root must not be a symbolic link")
    return root


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _heartbeat(
    root: Path,
    *,
    phase: str,
    slot: int | None,
    record_id: str | None,
    started: float,
    completed: int,
    terminal: int,
    certificate_progress: tuple[int, int] | None,
    last_durable_shard: str | None,
) -> None:
    _write_heartbeat(
        root / "heartbeat.json",
        {
            "schema": "gemma-sv-response-heartbeat-v2",
            "phase": phase,
            "slot": slot,
            "record_id_sha256": (
                None if record_id is None else _text_sha256(record_id)
            ),
            "elapsed_seconds": max(0.0, time.perf_counter() - started),
            "rss_bytes": _rss_bytes(),
            "completed_count": completed,
            "terminal_count": terminal,
            "certificate_progress": (
                None
                if certificate_progress is None
                else {
                    "completed": certificate_progress[0],
                    "total": certificate_progress[1],
                }
            ),
            "last_durable_shard": last_durable_shard,
        },
    )


@contextmanager
def _heartbeat_ticker(
    emit: Callable[[], None],
    *,
    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> Iterable[None]:
    """Emit content-free liveness updates while one record is in flight."""

    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("heartbeat interval must be finite and positive")
    stopped = threading.Event()
    failure: list[BaseException] = []

    def tick() -> None:
        while not stopped.wait(interval_seconds):
            try:
                emit()
            except BaseException as exc:
                failure.append(exc)
                stopped.set()

    thread = threading.Thread(
        target=tick,
        name="gemmasv-audit-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
    if failure:
        raise RuntimeError("heartbeat ticker failed") from failure[0]


def _binding(lock: Mapping[str, Any], record_id: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in lock["prompt_answer_bindings"]["histories"]
        if row["history_id"] == record_id
    ]
    if len(matches) != 1:
        raise AuditV2Error("history prompt binding is unavailable")
    return matches[0]


def _validate_rehydrated_record(
    record: Any,
    public: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[Any, ...]:
    if str(record.record_id) != str(public["record_id"]):
        raise AuditV2Error("rehydrated history ID differs")
    probes = method_states.build_probes(record)
    if tuple(probe.probe_id for probe in probes) != PROBE_IDS:
        raise AuditV2Error("rehydrated probe order differs")
    frozen = _binding(lock, record.record_id)
    if (
        frozen["history_integrity_sha256"]
        != public["record_integrity"]["sha256"]
    ):
        raise AuditV2Error("history integrity binding differs")
    for probe, stored in zip(probes, frozen["probes"]):
        if (
            probe.probe_id != stored["probe_id"]
            or probe.kind != stored["kind"]
            or _text_sha256(probe.prompt) != stored["prompt_sha256"]
            or _text_sha256(probe.target) != stored["answer_sha256"]
            or len(probe.target_ids) != stored["target_token_count"]
            or cohort_v3.base.token_ids_sha256(probe.target_ids)
            != stored["target_token_ids_sha256"]
        ):
            raise AuditV2Error("rehydrated probe differs from lock")
    return probes


def _normalize_answer(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _generation_artifact(response: decoded_v1.GreedyResponse) -> dict[str, Any]:
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


def _response_checks(
    response_text: str,
    *,
    answer: str,
    other_answer: str,
) -> dict[str, Any]:
    rendered = str(response_text)
    normalized = _normalize_answer(rendered)
    expected = str(answer)
    expected_normalized = _normalize_answer(expected)
    other_normalized = _normalize_answer(other_answer)
    return {
        "answer_exact_match": rendered.strip() == expected.strip(),
        "answer_normalized_match": normalized == expected_normalized,
        "answer_casefold_substring": (
            bool(expected_normalized)
            and expected_normalized in normalized
        ),
        "other_answer_leakage_casefold_substring": (
            bool(other_normalized) and other_normalized in normalized
        ),
        "decoded_strings_exhaust_extractability": False,
    }


def _run_probe(
    runtime: Any,
    memory: Any,
    probe: Any,
    *,
    condition_id: str,
    record: Any,
) -> dict[str, Any]:
    answers = {item.probe_id: str(item.answer) for item in record.probes}
    other_probe_id = next(item for item in PROBE_IDS if item != probe.probe_id)
    attempts = []
    errors = []
    for repeat in range(GENERATION_REPEATS):
        seed = decoded_v1._response_seed(
            record.record_id,
            condition_id,
            probe.probe_id,
        )
        decoded_v1._seed_response(seed)
        try:
            response = decoded_v1.greedy_generate_response(
                runtime,
                memory,
                probe.prompt,
            )
            attempts.append(
                {
                    "repeat_index": repeat,
                    "status": "completed",
                    **_generation_artifact(response),
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "repeat_index": repeat,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message_redacted": True,
                    "error_message_sha256": _text_sha256(str(exc)),
                }
            )
            errors.append(type(exc).__name__)
    complete = [item for item in attempts if item["status"] == "completed"]
    equal = bool(
        len(complete) == GENERATION_REPEATS
        and complete[0]["generated_token_ids"]
        == complete[1]["generated_token_ids"]
        and complete[0]["response_text"] == complete[1]["response_text"]
    )
    status = (
        "completed"
        if equal
        else (
            "failed_nondeterministic"
            if len(complete) == GENERATION_REPEATS
            else "failed"
        )
    )
    result = {
        "probe_id": probe.probe_id,
        "probe_kind": probe.kind,
        "status": status,
        "generation_attempts_started": GENERATION_REPEATS,
        "generation_attempts_completed": len(complete),
        "generation_attempts": attempts,
        "repeat_check": {
            "repetitions": GENERATION_REPEATS,
            "exact_token_ids_and_response_text_match": equal,
        },
    }
    if complete:
        result["response_checks"] = _response_checks(
            complete[0]["response_text"],
            answer=answers[probe.probe_id],
            other_answer=answers[other_probe_id],
        )
    if errors:
        result["error_types"] = errors
    return result


def _failed_condition(condition_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": _text_sha256(str(exc)),
        "generation_calls_started": 0,
        "probes": [
            {
                "probe_id": probe_id,
                "status": "failed",
                "generation_attempts_started": 0,
                "generation_attempts_completed": 0,
                "generation_attempts": [
                    {
                        "repeat_index": repeat,
                        "status": "not_attempted",
                    }
                    for repeat in range(GENERATION_REPEATS)
                ],
            }
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
    semantics: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        _run_probe(
            runtime,
            memory,
            probe,
            condition_id=condition_id,
            record=record,
        )
        for probe in probes
    ]
    return {
        "condition_id": condition_id,
        "status": (
            "completed"
            if all(row["status"] == "completed" for row in rows)
            else "completed_with_probe_failures"
        ),
        "semantics": copy.deepcopy(dict(semantics)),
        "generation_calls_started": sum(
            row["generation_attempts_started"] for row in rows
        ),
        "probes": rows,
    }


def _certificate_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"exact", "refit", "decay"}
    return {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in excluded
    }


def audit_history(
    runtime: Any,
    record: Any,
    public: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute exactly four conditions and sixteen genuine generations."""

    probes = _validate_rehydrated_record(record, public, lock)
    conditions: dict[str, Any] = {}
    original = None
    try:
        original = runtime.prefill_persistent(
            list(record.context.original_token_ids)
        )
        expected = cohort_v3.base.token_ids_sha256(
            record.context.original_token_ids
        )
        if str(original.input_digest) != expected:
            raise RuntimeError("original persistent prefill digest differs")
        conditions["present"] = _execute_condition(
            runtime,
            original,
            probes,
            condition_id="present",
            record=record,
            semantics={"reference": True, "owned_round_present": True},
        )
    except Exception as exc:
        conditions["present"] = _failed_condition("present", exc)

    try:
        raw = runtime.prefill_persistent(
            list(record.raw_omitted_token_ids)
        )
        expected = cohort_v3.base.token_ids_sha256(
            record.raw_omitted_token_ids
        )
        if str(raw.input_digest) != expected:
            raise RuntimeError("raw-omission persistent prefill digest differs")
        conditions["fresh_raw_omission"] = _execute_condition(
            runtime,
            raw,
            probes,
            condition_id="fresh_raw_omission",
            record=record,
            semantics={
                "fresh_prefill": True,
                "owned_round_omitted": True,
                "suffix_recomputed": True,
            },
        )
    except Exception as exc:
        conditions["fresh_raw_omission"] = _failed_condition(
            "fresh_raw_omission",
            exc,
        )

    if original is None:
        exc = RuntimeError("original persistent state unavailable")
        conditions["exact_decrement_or_refit_policy"] = _failed_condition(
            "exact_decrement_or_refit_policy",
            exc,
        )
        conditions["prompt_suppression"] = _failed_condition(
            "prompt_suppression",
            exc,
        )
    else:
        try:
            diagnostics, states = runtime.persistent_certificate_states(
                original,
                record.context.forget_positions,
            )
            if diagnostics.get("fixed_c_feasible") is not True:
                raise RuntimeError("fixed-C certificate reports infeasibility")
            used_refit_fallback = bool(
                diagnostics.get("used_refit_fallback")
                or int(diagnostics.get("n_fallback", 0)) > 0
                or int(diagnostics.get("decrement_fallbacks", 0)) > 0
            )
            policy = _execute_condition(
                runtime,
                states["exact"],
                probes,
                condition_id="exact_decrement_or_refit_policy",
                record=record,
                semantics={
                    "executed_state": "exact",
                    "incremental_with_per_head_refit_fallback": True,
                    "used_any_refit_fallback": used_refit_fallback,
                    "separate_refit_response_decoded": False,
                    "fixed_C": True,
                },
            )
            policy["fixed_c_diagnostics"] = _certificate_diagnostics(
                diagnostics
            )
            conditions["exact_decrement_or_refit_policy"] = policy
        except Exception as exc:
            conditions[
                "exact_decrement_or_refit_policy"
            ] = _failed_condition(
                "exact_decrement_or_refit_policy",
                exc,
            )
        try:
            suppressed = method_states._prompt_suppression_probes(
                record,
                probes,
            )
            conditions["prompt_suppression"] = _execute_condition(
                runtime,
                original,
                suppressed,
                condition_id="prompt_suppression",
                record=record,
                semantics={
                    "prompt_only_behavioral_control": True,
                    "persistent_state_deleted": False,
                    "instruction_target_free": True,
                },
            )
        except Exception as exc:
            conditions["prompt_suppression"] = _failed_condition(
                "prompt_suppression",
                exc,
            )

    ordered = {condition_id: conditions[condition_id] for condition_id in CONDITION_IDS}
    calls = sum(item["generation_calls_started"] for item in ordered.values())
    failures = [
        condition_id
        for condition_id, value in ordered.items()
        if value["status"] != "completed"
    ]
    return {
        "record_id": record.record_id,
        "cluster_id": public["cluster_id"],
        "variant_index": int(public["variant_index"]),
        "status": "completed" if not failures else "completed_with_failures",
        "contains_source_text": True,
        "contains_model_generated_text": True,
        "decoded_strings_exhaust_extractability": False,
        "conditions": ordered,
        "generation_calls_started": calls,
        "failed_condition_ids": failures,
        "omitted_decoded_conditions": {
            "condition_ids": [
                "fp32_proxy",
                "decay_0_01",
                "token_row_repack_alias",
                "separate_fixed_c_refit",
            ],
            "remain_teacher_forced_or_other_evidence": True,
        },
    }


def _terminal_failure(
    record: Any,
    public: Mapping[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "cluster_id": public["cluster_id"],
        "variant_index": int(public["variant_index"]),
        "status": "failed",
        "contains_source_text": True,
        "contains_model_generated_text": True,
        "decoded_strings_exhaust_extractability": False,
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": _text_sha256(str(exc)),
        "conditions": {
            condition_id: _failed_condition(condition_id, exc)
            for condition_id in CONDITION_IDS
        },
        "generation_calls_started": 0,
        "failed_condition_ids": list(CONDITION_IDS),
        "omitted_decoded_conditions": {
            "condition_ids": [
                "fp32_proxy",
                "decay_0_01",
                "token_row_repack_alias",
                "separate_fixed_c_refit",
            ],
            "remain_teacher_forced_or_other_evidence": True,
        },
    }


def _base_run(
    lock: Mapping[str, Any],
    *,
    certificate_workers: int,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": RUN_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "initialized",
            "authorization_lock_integrity_sha256": lock["integrity"]["sha256"],
            "ordered_history_ids_sha256": lock["cohort"][
                "ordered_history_ids_sha256"
            ],
            "certificate_workers": certificate_workers,
            "certificate_backend": "process",
            "durable_unit": "one_complete_history",
            "source_bearing": True,
            "local_only": True,
            "no_overwrite": True,
        }
    )


def _validate_run(
    run: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    certificate_workers: int,
) -> None:
    _validate_seal(run, name="run")
    if run != _base_run(
        lock,
        certificate_workers=certificate_workers,
    ):
        raise AuditV2Error("run binding differs")


def _slot_name(slot: int) -> str:
    return f"{slot:03d}"


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise AuditV2Error(
            f"{name} fields differ; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _validate_generation_attempt(
    attempt: Mapping[str, Any],
    *,
    repeat_index: int,
) -> None:
    if (
        attempt.get("repeat_index") != repeat_index
        or attempt.get("status")
        not in {"completed", "failed", "not_attempted"}
    ):
        raise AuditV2Error("generation attempt identity or status differs")
    status = attempt["status"]
    if status == "not_attempted":
        _require_exact_keys(
            attempt,
            {"repeat_index", "status"},
            name="not-attempted generation",
        )
        return
    if status == "failed":
        _require_exact_keys(
            attempt,
            {
                "repeat_index",
                "status",
                "error_type",
                "error_message_redacted",
                "error_message_sha256",
            },
            name="failed generation",
        )
        if (
            not isinstance(attempt["error_type"], str)
            or attempt["error_message_redacted"] is not True
            or not decoded_v1._is_sha256(
                attempt["error_message_sha256"]
            )
        ):
            raise AuditV2Error("failed generation disclosure differs")
        return
    _require_exact_keys(
        attempt,
        {
            "repeat_index",
            "status",
            "response_text",
            "response_utf8_sha256",
            "generated_token_ids",
            "generated_token_ids_sha256",
            "generated_token_count",
            "stop_reason",
            "stop_token_id",
            "prompt_token_count",
        },
        name="completed generation",
    )
    token_ids = attempt["generated_token_ids"]
    if (
        not isinstance(attempt["response_text"], str)
        or attempt["response_utf8_sha256"]
        != _text_sha256(attempt["response_text"])
        or not isinstance(token_ids, list)
        or not token_ids
        or len(token_ids) > MAX_NEW_TOKENS
        or any(type(item) is not int or item < 0 for item in token_ids)
        or attempt["generated_token_count"] != len(token_ids)
        or attempt["generated_token_ids_sha256"]
        != _payload_sha256(token_ids)
        or type(attempt["prompt_token_count"]) is not int
        or attempt["prompt_token_count"] < 1
    ):
        raise AuditV2Error("completed generation payload differs")
    terminal = token_ids[-1] if token_ids[-1] in STOP_TOKEN_IDS else None
    if terminal is None:
        valid_stop = (
            attempt["stop_reason"] == "max_new_tokens"
            and attempt["stop_token_id"] is None
            and len(token_ids) == MAX_NEW_TOKENS
        )
    else:
        valid_stop = (
            attempt["stop_reason"] == "terminal_stop_token"
            and attempt["stop_token_id"] == terminal
        )
    if not valid_stop:
        raise AuditV2Error("generation stop semantics differ")


def _validate_probe_result(
    probe: Mapping[str, Any],
    *,
    probe_id: str,
    condition_failed: bool,
) -> None:
    attempts = probe.get("generation_attempts")
    if (
        probe.get("probe_id") != probe_id
        or not isinstance(attempts, list)
        or len(attempts) != GENERATION_REPEATS
    ):
        raise AuditV2Error("probe identity or attempt slots differ")
    for repeat_index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise AuditV2Error("generation attempt must be an object")
        _validate_generation_attempt(attempt, repeat_index=repeat_index)
    completed = [
        item for item in attempts if item["status"] == "completed"
    ]
    started = sum(item["status"] != "not_attempted" for item in attempts)
    if (
        probe.get("generation_attempts_started") != started
        or probe.get("generation_attempts_completed") != len(completed)
    ):
        raise AuditV2Error("probe generation accounting differs")
    if condition_failed:
        _require_exact_keys(
            probe,
            {
                "probe_id",
                "status",
                "generation_attempts_started",
                "generation_attempts_completed",
                "generation_attempts",
            },
            name="failed condition probe",
        )
        if probe["status"] != "failed" or started != 0:
            raise AuditV2Error("failed condition probe status differs")
        return
    expected_keys = {
        "probe_id",
        "probe_kind",
        "status",
        "generation_attempts_started",
        "generation_attempts_completed",
        "generation_attempts",
        "repeat_check",
    }
    if completed:
        expected_keys.add("response_checks")
    if any(item["status"] == "failed" for item in attempts):
        expected_keys.add("error_types")
    _require_exact_keys(probe, expected_keys, name="executed probe")
    expected_kind = "deleted" if probe_id == "target_current" else "retained"
    repeat = probe["repeat_check"]
    equal = bool(
        len(completed) == GENERATION_REPEATS
        and completed[0]["generated_token_ids"]
        == completed[1]["generated_token_ids"]
        and completed[0]["response_text"] == completed[1]["response_text"]
    )
    expected_status = (
        "completed"
        if equal
        else (
            "failed_nondeterministic"
            if len(completed) == GENERATION_REPEATS
            else "failed"
        )
    )
    if (
        probe["probe_kind"] != expected_kind
        or probe["status"] != expected_status
        or started != GENERATION_REPEATS
        or not isinstance(repeat, Mapping)
        or set(repeat) != {
            "repetitions",
            "exact_token_ids_and_response_text_match",
        }
        or repeat["repetitions"] != GENERATION_REPEATS
        or repeat["exact_token_ids_and_response_text_match"] is not equal
    ):
        raise AuditV2Error("probe repeat status or equality differs")
    if completed:
        checks = probe["response_checks"]
        if (
            not isinstance(checks, Mapping)
            or set(checks)
            != {
                "answer_exact_match",
                "answer_normalized_match",
                "answer_casefold_substring",
                "other_answer_leakage_casefold_substring",
                "decoded_strings_exhaust_extractability",
            }
            or any(
                type(checks[key]) is not bool
                for key in checks
                if key != "decoded_strings_exhaust_extractability"
            )
            or checks["decoded_strings_exhaust_extractability"] is not False
        ):
            raise AuditV2Error("response checks disclosure differs")
    if "error_types" in probe:
        expected_errors = [
            item["error_type"]
            for item in attempts
            if item["status"] == "failed"
        ]
        if probe["error_types"] != expected_errors:
            raise AuditV2Error("probe error accounting differs")


def _validate_fixed_c_diagnostics(value: Mapping[str, Any]) -> None:
    expected = {
        "n_solves",
        "n_fallback",
        "n_head_gates",
        "box_C",
        "box_C_by_boundary",
        "objective",
        "feasibility",
        "fixed_c_feasible",
        "fallback_details",
        "used_refit_fallback",
        "max_functional_deviation",
        "max_candidate_deviation",
        "functional_tolerance",
        "partition_diagnostics",
        "materialization",
    }
    _require_exact_keys(value, expected, name="fixed-C diagnostics")
    materialization = value["materialization"]
    if (
        value["fixed_c_feasible"] is not True
        or type(value["n_solves"]) is not int
        or value["n_solves"] < 0
        or type(value["n_fallback"]) is not int
        or value["n_fallback"] < 0
        or type(value["n_head_gates"]) is not int
        or value["n_head_gates"] < 1
        or not isinstance(value["fallback_details"], list)
        or len(value["fallback_details"]) != value["n_fallback"]
        or value["used_refit_fallback"] is not bool(value["n_fallback"])
        or not isinstance(value["objective"], Mapping)
        or not isinstance(value["feasibility"], list)
        or not isinstance(value["partition_diagnostics"], Mapping)
        or materialization
        != {
            "persistent_states": ["exact_policy"],
            "refit_coefficients_used_for_solver_conformance": True,
            "refit_persistent_state_constructed": False,
            "decay_persistent_state_constructed": False,
        }
    ):
        raise AuditV2Error("fixed-C diagnostics accounting differs")
    _require_exact_keys(
        value["objective"],
        {
            "box_C",
            "box_C_by_boundary",
            "box_source",
            "bandwidth_source",
            "kpar_by_layer",
        },
        name="fixed-C objective",
    )
    for detail in value["fallback_details"]:
        if (
            not isinstance(detail, Mapping)
            or set(detail) != {"layer_id", "start", "head", "reason"}
            or any(
                type(detail[key]) is not int
                for key in ("layer_id", "start", "head")
            )
            or not isinstance(detail["reason"], str)
        ):
            raise AuditV2Error("fixed-C fallback detail differs")
    for item in value["feasibility"]:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "start",
                "box_C",
                "forgotten_count",
                "retained_count",
                "retained_capacity",
                "deleted_fraction",
                "maximum_deleted_fraction",
                "feasible",
                "fallback",
            }
            or item["feasible"] is not True
            or item["fallback"] != "none"
        ):
            raise AuditV2Error("fixed-C feasibility disclosure differs")
    partition_keys = {
        "affected_support_fraction",
        "affected_margin_fraction",
        "refit_support_fraction",
        "fallback_support_fraction",
        "successful_support_fraction",
    }
    if set(value["partition_diagnostics"]) != partition_keys:
        raise AuditV2Error("fixed-C partition diagnostics fields differ")
    for summary in value["partition_diagnostics"].values():
        if summary is None:
            continue
        if (
            not isinstance(summary, Mapping)
            or set(summary) != {"n", "mean", "median", "minimum", "maximum"}
            or type(summary["n"]) is not int
            or summary["n"] < 1
            or any(
                isinstance(summary[key], bool)
                or not isinstance(summary[key], (int, float))
                or not math.isfinite(float(summary[key]))
                for key in ("mean", "median", "minimum", "maximum")
            )
        ):
            raise AuditV2Error("fixed-C partition summary differs")
    for key in (
        "max_functional_deviation",
        "max_candidate_deviation",
        "functional_tolerance",
    ):
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], (int, float))
            or not math.isfinite(float(value[key]))
            or float(value[key]) < 0
        ):
            raise AuditV2Error("fixed-C numerical diagnostics differ")


def _validate_condition(
    condition: Mapping[str, Any],
    *,
    condition_id: str,
) -> None:
    if condition.get("condition_id") != condition_id:
        raise AuditV2Error("condition identity differs")
    failed = condition.get("status") == "failed"
    expected_keys = {
        "condition_id",
        "status",
        "generation_calls_started",
        "probes",
    }
    if failed:
        expected_keys.update(
            {
                "error_type",
                "error_message_redacted",
                "error_message_sha256",
            }
        )
    else:
        expected_keys.add("semantics")
        if condition_id == "exact_decrement_or_refit_policy":
            expected_keys.add("fixed_c_diagnostics")
    _require_exact_keys(condition, expected_keys, name="condition")
    probes = condition["probes"]
    if (
        not isinstance(probes, list)
        or [probe.get("probe_id") for probe in probes] != list(PROBE_IDS)
    ):
        raise AuditV2Error("condition probe order differs")
    for probe, probe_id in zip(probes, PROBE_IDS):
        _validate_probe_result(
            probe,
            probe_id=probe_id,
            condition_failed=failed,
        )
    calls = sum(probe["generation_attempts_started"] for probe in probes)
    if condition["generation_calls_started"] != calls:
        raise AuditV2Error("condition generation call accounting differs")
    if failed:
        if (
            condition["error_message_redacted"] is not True
            or not isinstance(condition["error_type"], str)
            or not decoded_v1._is_sha256(
                condition["error_message_sha256"]
            )
            or calls != 0
        ):
            raise AuditV2Error("failed condition disclosure differs")
        return
    expected_status = (
        "completed"
        if all(probe["status"] == "completed" for probe in probes)
        else "completed_with_probe_failures"
    )
    semantics_contract = {
        "present": {
            "reference": True,
            "owned_round_present": True,
        },
        "fresh_raw_omission": {
            "fresh_prefill": True,
            "owned_round_omitted": True,
            "suffix_recomputed": True,
        },
        "exact_decrement_or_refit_policy": {
            "executed_state": "exact",
            "incremental_with_per_head_refit_fallback": True,
            "used_any_refit_fallback": None,
            "separate_refit_response_decoded": False,
            "fixed_C": True,
        },
        "prompt_suppression": {
            "prompt_only_behavioral_control": True,
            "persistent_state_deleted": False,
            "instruction_target_free": True,
        },
    }
    if (
        condition["status"] != expected_status
        or set(condition["semantics"])
        != set(semantics_contract[condition_id])
    ):
        raise AuditV2Error("condition status or semantics fields differ")
    if (
        condition_id != "exact_decrement_or_refit_policy"
        and condition["semantics"] != semantics_contract[condition_id]
    ):
        raise AuditV2Error("condition semantics values differ")
    if condition_id == "exact_decrement_or_refit_policy":
        _validate_fixed_c_diagnostics(condition["fixed_c_diagnostics"])
        if (
            condition["semantics"]["executed_state"] != "exact"
            or condition["semantics"][
                "incremental_with_per_head_refit_fallback"
            ]
            is not True
            or condition["semantics"]["separate_refit_response_decoded"]
            is not False
            or condition["semantics"]["fixed_C"] is not True
            or condition["semantics"]["used_any_refit_fallback"]
            is not condition["fixed_c_diagnostics"]["used_refit_fallback"]
        ):
            raise AuditV2Error("exact-policy semantics differ from diagnostics")


def _validate_record_result(
    result: Mapping[str, Any],
    *,
    record_id: str,
    lock: Mapping[str, Any],
) -> None:
    expected_keys = {
        "record_id",
        "cluster_id",
        "variant_index",
        "status",
        "contains_source_text",
        "contains_model_generated_text",
        "decoded_strings_exhaust_extractability",
        "conditions",
        "generation_calls_started",
        "failed_condition_ids",
        "omitted_decoded_conditions",
    }
    if result.get("status") == "failed":
        expected_keys.update(
            {
                "error_type",
                "error_message_redacted",
                "error_message_sha256",
            }
        )
    _require_exact_keys(result, expected_keys, name="record result")
    binding = _binding(lock, record_id)
    conditions = result["conditions"]
    if (
        result["record_id"] != record_id
        or result["cluster_id"] != binding["cluster_id"]
        or result["variant_index"] != binding["variant_index"]
        or result["contains_source_text"] is not True
        or result["contains_model_generated_text"] is not True
        or result["decoded_strings_exhaust_extractability"] is not False
        or not isinstance(conditions, Mapping)
        or list(conditions) != list(CONDITION_IDS)
    ):
        raise AuditV2Error("record result identity or condition order differs")
    for condition_id in CONDITION_IDS:
        _validate_condition(
            conditions[condition_id],
            condition_id=condition_id,
        )
    failures = [
        condition_id
        for condition_id in CONDITION_IDS
        if conditions[condition_id]["status"] != "completed"
    ]
    expected_status = (
        "completed"
        if not failures
        else (
            "failed"
            if all(
                conditions[item]["status"] == "failed"
                for item in CONDITION_IDS
            )
            else "completed_with_failures"
        )
    )
    calls = sum(
        condition["generation_calls_started"]
        for condition in conditions.values()
    )
    omitted = result["omitted_decoded_conditions"]
    if (
        result["status"] != expected_status
        or result["failed_condition_ids"] != failures
        or result["generation_calls_started"] != calls
        or not 0 <= calls <= 16
        or omitted
        != {
            "condition_ids": [
                "fp32_proxy",
                "decay_0_01",
                "token_row_repack_alias",
                "separate_fixed_c_refit",
            ],
            "remain_teacher_forced_or_other_evidence": True,
        }
    ):
        raise AuditV2Error("record failure or generation accounting differs")
    if result["status"] == "failed" and (
        result["error_message_redacted"] is not True
        or not isinstance(result["error_type"], str)
        or not decoded_v1._is_sha256(result["error_message_sha256"])
    ):
        raise AuditV2Error("terminal record failure disclosure differs")


def _attempt_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    _validate_permissions(directory, directory=True)
    expected_suffixes = ("-started.json", "-terminal.json")
    files = sorted(directory.iterdir())
    if any(
        path.is_symlink()
        or not path.is_file()
        or not path.name.endswith(expected_suffixes)
        for path in files
    ):
        raise AuditV2Error("attempt ledger contains an extra entry")
    return files


def _validate_attempt_ledgers(
    root: Path,
    lock: Mapping[str, Any],
    shards: Mapping[int, Mapping[str, Any]],
) -> dict[int, list[int]]:
    attempts_root = root / "attempts"
    expected_directories = {
        _slot_name(slot) for slot in range(EXPECTED_HISTORIES)
    }
    entries = list(attempts_root.iterdir()) if attempts_root.exists() else []
    if attempts_root.exists():
        _validate_permissions(attempts_root, directory=True)
    if any(
        path.is_symlink()
        or path.name not in expected_directories
        or not path.is_dir()
        for path in entries
    ):
        raise AuditV2Error("attempt directory contains an extra entry")
    terminal_by_slot: dict[int, list[int]] = {}
    history_ids = lock["cohort"]["ordered_history_ids"]
    for directory in entries:
        _validate_permissions(directory, directory=True)
        slot = int(directory.name)
        phases: dict[int, set[str]] = {}
        terminal_values: dict[int, Mapping[str, Any]] = {}
        for path in _attempt_files(directory):
            _validate_permissions(path, directory=False)
            prefix, phase_suffix = path.name.split("-", 1)
            if (
                len(prefix) != 3
                or not prefix.isdigit()
                or int(prefix) < 1
                or phase_suffix
                not in {"started.json", "terminal.json"}
            ):
                raise AuditV2Error("attempt ledger filename differs")
            attempt = int(prefix)
            phase = phase_suffix.removesuffix(".json")
            if phase in phases.setdefault(attempt, set()):
                raise AuditV2Error("duplicate attempt ledger")
            phases[attempt].add(phase)
            value = _load_json(path, name="attempt ledger")
            _validate_seal(value, name="attempt ledger")
            if (
                value.get("schema") != ATTEMPT_SCHEMA
                or value.get("phase") != phase
                or value.get("slot") != slot
                or value.get("attempt") != attempt
                or value.get("record_id") != history_ids[slot]
                or value.get("authorization_lock_integrity_sha256")
                != lock["integrity"]["sha256"]
            ):
                raise AuditV2Error("attempt ledger binding differs")
            if phase == "terminal":
                terminal_values[attempt] = value
        for attempt, observed_phases in phases.items():
            if "terminal" in observed_phases and "started" not in observed_phases:
                raise AuditV2Error("terminal attempt has no started ledger")
        terminals = sorted(terminal_values)
        if len(terminals) > 1:
            raise AuditV2Error("record has multiple terminal attempts")
        if terminals:
            if slot not in shards:
                raise AuditV2Error("terminal attempt has no record shard")
            terminal = terminal_values[terminals[0]]
            if (
                terminal.get("record_shard_integrity_sha256")
                != shards[slot]["integrity"]["sha256"]
                or terminal.get("record_status")
                != shards[slot]["result"]["status"]
                or terminal.get("retry_allowed") is not False
            ):
                raise AuditV2Error("terminal attempt differs from record shard")
        if slot in shards and not phases:
            raise AuditV2Error("record shard has no attempt ledger")
        terminal_by_slot[slot] = terminals
    if any(slot not in terminal_by_slot for slot in shards):
        raise AuditV2Error("record shard has no attempt directory")
    return terminal_by_slot


def _next_attempt(directory: Path) -> int:
    starts = [
        int(path.name.split("-", 1)[0])
        for path in _attempt_files(directory)
        if path.name.endswith("-started.json")
    ]
    if len(starts) != len(set(starts)):
        raise AuditV2Error("duplicate attempt ledger")
    return max(starts, default=0) + 1


def _validate_shard(
    shard: Mapping[str, Any],
    *,
    slot: int,
    record_id: str,
    lock: Mapping[str, Any],
) -> None:
    _validate_seal(shard, name=f"record shard {slot}")
    _require_exact_keys(
        shard,
        {
            "schema",
            "schema_version",
            "slot",
            "record_id",
            "terminal",
            "authorization_lock_integrity_sha256",
            "result",
            "integrity",
        },
        name="record shard",
    )
    if (
        shard.get("schema") != SHARD_SCHEMA
        or shard.get("schema_version") != SCHEMA_VERSION
        or shard.get("slot") != slot
        or shard.get("record_id") != record_id
        or shard.get("terminal") is not True
        or shard.get("authorization_lock_integrity_sha256")
        != lock["integrity"]["sha256"]
        or not isinstance(shard.get("result"), Mapping)
        or shard["result"].get("record_id") != record_id
    ):
        raise AuditV2Error(f"record shard {slot} binding differs")
    _validate_record_result(
        shard["result"],
        record_id=record_id,
        lock=lock,
    )


def _scan_shards(
    root: Path,
    lock: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    records_directory = root / "records"
    if records_directory.exists():
        _validate_permissions(records_directory, directory=True)
    expected_names = {
        f"{_slot_name(slot)}.json" for slot in range(EXPECTED_HISTORIES)
    }
    files = list(records_directory.iterdir()) if records_directory.exists() else []
    if any(
        path.is_symlink()
        or path.name not in expected_names
        or not path.is_file()
        for path in files
    ):
        raise AuditV2Error("record directory contains an extra shard")
    result: dict[int, dict[str, Any]] = {}
    observed_ids: set[str] = set()
    ids = lock["cohort"]["ordered_history_ids"]
    for path in sorted(files):
        slot = int(path.stem)
        _validate_permissions(path, directory=False)
        shard = _load_json(path, name=f"record shard {slot}")
        _validate_shard(
            shard,
            slot=slot,
            record_id=ids[slot],
            lock=lock,
        )
        if shard["record_id"] in observed_ids:
            raise AuditV2Error("duplicate history appears in record shards")
        observed_ids.add(shard["record_id"])
        result[slot] = shard
    return result


def _initialize_root(
    root: Path,
    lock: Mapping[str, Any],
    *,
    certificate_workers: int,
    resume: bool,
) -> None:
    if resume:
        if root.is_symlink():
            raise AuditV2Error("resume root must not be a symlink")
        if not root.is_dir():
            raise FileNotFoundError("resume root does not exist")
        _validate_permissions(root, directory=True)
        run = _load_json(root / "run.json", name="run")
        _validate_permissions(root / "run.json", directory=False)
        _validate_run(
            run,
            lock,
            certificate_workers=certificate_workers,
        )
    else:
        try:
            os.mkdir(root, 0o700)
        except FileExistsError as exc:
            raise FileExistsError(
                "output root exists; use explicit resume"
            ) from exc
        _atomic_write_new(
            root / "run.json",
            _base_run(
                lock,
                certificate_workers=certificate_workers,
            ),
        )
    for name in ("attempts", "records"):
        directory = root / name
        if directory.exists() or directory.is_symlink():
            _validate_permissions(directory, directory=True)
            if not directory.is_dir():
                raise AuditV2Error(f"{name} is not a directory")
        else:
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)


def _attempt_started(
    root: Path,
    *,
    slot: int,
    attempt: int,
    record_id: str,
    lock: Mapping[str, Any],
) -> Path:
    directory = root / "attempts" / _slot_name(slot)
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / f"{attempt:03d}-started.json"
    _atomic_write_new(
        path,
        _seal(
            {
                "schema": ATTEMPT_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "phase": "started",
                "slot": slot,
                "attempt": attempt,
                "record_id": record_id,
                "authorization_lock_integrity_sha256": lock["integrity"][
                    "sha256"
                ],
            }
        ),
    )
    return path


def _attempt_terminal(
    root: Path,
    *,
    slot: int,
    attempt: int,
    record_id: str,
    shard: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    path = (
        root
        / "attempts"
        / _slot_name(slot)
        / f"{attempt:03d}-terminal.json"
    )
    _atomic_write_new(
        path,
        _seal(
            {
                "schema": ATTEMPT_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "phase": "terminal",
                "slot": slot,
                "attempt": attempt,
                "record_id": record_id,
                "record_shard_integrity_sha256": shard["integrity"]["sha256"],
                "record_status": shard["result"]["status"],
                "authorization_lock_integrity_sha256": lock["integrity"][
                    "sha256"
                ],
                "retry_allowed": False,
            }
        ),
    )


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "history_slots": len(records),
        "completed_histories": sum(
            row["status"] == "completed" for row in records
        ),
        "terminal_failed_or_nondeterministic_histories": sum(
            row["status"] != "completed" for row in records
        ),
        "generation_calls_started": sum(
            int(row.get("generation_calls_started", 0)) for row in records
        ),
        "authorized_generation_calls": EXPECTED_GENERATION_CALLS,
        "all_96_histories_retained": len(records) == EXPECTED_HISTORIES,
        "outcome_based_filtering": False,
    }


def validate_final(
    final: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
) -> None:
    _validate_seal(final, name="final")
    _require_exact_keys(
        final,
        {
            "schema",
            "schema_version",
            "status",
            "authorization_lock_integrity_sha256",
            "contains_source_text",
            "contains_model_generated_text",
            "source_bearing",
            "local_only",
            "release_authorized",
            "decoded_strings_exhaust_extractability",
            "ordered_history_ids",
            "records",
            "summary",
            "integrity",
        },
        name="final",
    )
    expected_ids = lock["cohort"]["ordered_history_ids"]
    records = final["records"]
    if (
        final["schema"] != FINAL_SCHEMA
        or final["schema_version"] != SCHEMA_VERSION
        or final["authorization_lock_integrity_sha256"]
        != lock["integrity"]["sha256"]
        or final["contains_source_text"] is not True
        or final["contains_model_generated_text"] is not True
        or final["source_bearing"] is not True
        or final["local_only"] is not True
        or final["release_authorized"] is not False
        or final["decoded_strings_exhaust_extractability"] is not False
        or final["ordered_history_ids"] != expected_ids
        or not isinstance(records, list)
        or len(records) != EXPECTED_HISTORIES
        or [row.get("record_id") for row in records] != expected_ids
    ):
        raise AuditV2Error("final binding, order, or content flags differ")
    for row, record_id in zip(records, expected_ids):
        if not isinstance(row, Mapping):
            raise AuditV2Error("final record must be an object")
        _validate_record_result(row, record_id=record_id, lock=lock)
    expected_status = (
        "completed"
        if all(row["status"] == "completed" for row in records)
        else "completed_with_terminal_failures"
    )
    if (
        final["status"] != expected_status
        or final["summary"] != _summary(records)
    ):
        raise AuditV2Error("final status or summary differs")


def assemble_final(
    root: str | Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root)
    _validate_permissions(root, directory=True)
    final_path = root / "final.json"
    if final_path.exists() or final_path.is_symlink():
        _validate_permissions(final_path, directory=False)
        existing = _load_json(final_path, name="final")
        validate_final(existing, lock=lock)
        return existing
    for directory_name in ("attempts", "records"):
        _validate_permissions(root / directory_name, directory=True)
    _validate_permissions(root / "run.json", directory=False)
    run = _load_json(root / "run.json", name="run")
    workers = _validate_workers(run.get("certificate_workers"))
    _validate_run(
        run,
        lock,
        certificate_workers=workers,
    )
    shards = _scan_shards(root, lock)
    if len(shards) != EXPECTED_HISTORIES:
        raise AuditV2Error("cannot assemble before all 96 terminal shards exist")
    terminals = _validate_attempt_ledgers(root, lock, shards)
    if any(not terminals.get(slot) for slot in range(EXPECTED_HISTORIES)):
        raise AuditV2Error("cannot assemble an unterminated attempt ledger")
    rows = [shards[slot]["result"] for slot in range(EXPECTED_HISTORIES)]
    final = _seal(
        {
            "schema": FINAL_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": (
                "completed"
                if all(row["status"] == "completed" for row in rows)
                else "completed_with_terminal_failures"
            ),
            "authorization_lock_integrity_sha256": lock["integrity"]["sha256"],
            "contains_source_text": True,
            "contains_model_generated_text": True,
            "source_bearing": True,
            "local_only": True,
            "release_authorized": False,
            "decoded_strings_exhaust_extractability": False,
            "ordered_history_ids": list(
                lock["cohort"]["ordered_history_ids"]
            ),
            "records": rows,
            "summary": _summary(rows),
        }
    )
    validate_final(final, lock=lock)
    _atomic_write_new(final_path, final)
    return final


def run_sharded_audit(
    runtime: Any,
    records: Sequence[Any],
    cohort: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    output_root: str | Path,
    certificate_workers: int = DEFAULT_CERTIFICATE_WORKERS,
    resume: bool = False,
    assemble: bool = True,
    after_terminal: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """Run missing histories serially and publish immutable terminal shards."""

    workers = _validate_workers(certificate_workers)
    expected_ids = lock["cohort"]["ordered_history_ids"]
    observed_ids = [str(record.record_id) for record in records]
    histories = cohort.get("histories")
    if (
        len(records) != EXPECTED_HISTORIES
        or observed_ids != expected_ids
        or not isinstance(histories, list)
        or [row["record_id"] for row in histories] != expected_ids
    ):
        raise AuditV2Error("rehydrated cohort differs from frozen all-96 order")
    root = Path(output_root)
    _initialize_root(
        root,
        lock,
        certificate_workers=workers,
        resume=resume,
    )
    final_path = root / "final.json"
    if resume and (final_path.exists() or final_path.is_symlink()):
        _validate_permissions(final_path, directory=False)
        existing = _load_json(final_path, name="final")
        validate_final(existing, lock=lock)
        return existing
    started = time.perf_counter()
    shards = _scan_shards(root, lock)
    terminals = _validate_attempt_ledgers(root, lock, shards)
    for slot, shard in sorted(shards.items()):
        if terminals.get(slot):
            continue
        attempt_directory = root / "attempts" / _slot_name(slot)
        started_attempts = [
            int(path.name.split("-", 1)[0])
            for path in _attempt_files(attempt_directory)
            if path.name.endswith("-started.json")
        ]
        if not started_attempts:
            raise AuditV2Error("sealed shard has no started attempt")
        _attempt_terminal(
            root,
            slot=slot,
            attempt=max(started_attempts),
            record_id=shard["record_id"],
            shard=shard,
            lock=lock,
        )
    last_shard = (
        None
        if not shards
        else f"records/{_slot_name(max(shards))}.json"
    )
    _heartbeat(
        root,
        phase="resume_scan" if resume else "initialized",
        slot=None,
        record_id=None,
        started=started,
        completed=len(shards),
        terminal=len(shards),
        certificate_progress=None,
        last_durable_shard=last_shard,
    )

    for slot, (record, public) in enumerate(zip(records, histories)):
        if slot in shards:
            continue
        attempt_directory = root / "attempts" / _slot_name(slot)
        attempt = _next_attempt(attempt_directory)
        _attempt_started(
            root,
            slot=slot,
            attempt=attempt,
            record_id=record.record_id,
            lock=lock,
        )
        progress_value: tuple[int, int] | None = None
        progress_lock = threading.Lock()

        def progress_snapshot() -> tuple[int, int] | None:
            with progress_lock:
                return progress_value

        def tick_heartbeat() -> None:
            _heartbeat(
                root,
                phase="record_heartbeat",
                slot=slot,
                record_id=record.record_id,
                started=started,
                completed=len(shards),
                terminal=len(shards),
                certificate_progress=progress_snapshot(),
                last_durable_shard=last_shard,
            )

        def certificate_progress(done: int, total: int) -> None:
            nonlocal progress_value
            with progress_lock:
                previous = progress_value
                progress_value = (int(done), int(total))
                if (
                    previous is not None
                    and progress_value[0] < previous[0]
                ):
                    raise RuntimeError("certificate progress regressed")
            _heartbeat(
                root,
                phase="certificate",
                slot=slot,
                record_id=record.record_id,
                started=started,
                completed=len(shards),
                terminal=len(shards),
                certificate_progress=progress_snapshot(),
                last_durable_shard=last_shard,
            )

        setter = getattr(runtime, "set_certificate_progress_callback", None)
        if callable(setter):
            setter(certificate_progress)
        boundary = getattr(runtime, "record_boundary", None)
        scope = (
            boundary(slot, record.record_id)
            if callable(boundary)
            else nullcontext()
        )
        _heartbeat(
            root,
            phase="record",
            slot=slot,
            record_id=record.record_id,
            started=started,
            completed=len(shards),
            terminal=len(shards),
            certificate_progress=None,
            last_durable_shard=last_shard,
        )
        try:
            with scope, _heartbeat_ticker(
                tick_heartbeat,
                interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
            ):
                row = audit_history(
                    runtime,
                    record,
                    public,
                    lock=lock,
                )
        except Exception as exc:
            row = _terminal_failure(record, public, exc)
        finally:
            if callable(setter):
                setter(None)
        shard = _seal(
            {
                "schema": SHARD_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "slot": slot,
                "record_id": record.record_id,
                "terminal": True,
                "authorization_lock_integrity_sha256": lock["integrity"][
                    "sha256"
                ],
                "result": row,
            }
        )
        _validate_shard(
            shard,
            slot=slot,
            record_id=record.record_id,
            lock=lock,
        )
        shard_path = root / "records" / f"{_slot_name(slot)}.json"
        _atomic_write_new(shard_path, shard)
        _attempt_terminal(
            root,
            slot=slot,
            attempt=attempt,
            record_id=record.record_id,
            shard=shard,
            lock=lock,
        )
        shards[slot] = shard
        last_shard = f"records/{_slot_name(slot)}.json"
        _heartbeat(
            root,
            phase="terminal",
            slot=slot,
            record_id=record.record_id,
            started=started,
            completed=len(shards),
            terminal=len(shards),
            certificate_progress=progress_snapshot(),
            last_durable_shard=last_shard,
        )
        if after_terminal is not None:
            after_terminal(slot, shard)

    if not assemble:
        return None
    _heartbeat(
        root,
        phase="assembling",
        slot=None,
        record_id=None,
        started=started,
        completed=len(shards),
        terminal=len(shards),
        certificate_progress=None,
        last_durable_shard=last_shard,
    )
    final = assemble_final(root, lock)
    _heartbeat(
        root,
        phase="completed",
        slot=None,
        record_id=None,
        started=started,
        completed=len(shards),
        terminal=len(shards),
        certificate_progress=None,
        last_durable_shard=last_shard,
    )
    return final


def _require_head_committed_file(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError as exc:
        raise PermissionError("fingerprinted file is outside workspace") from exc
    completed = subprocess.run(
        ["git", "-C", str(WORKSPACE), "show", f"HEAD:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise PermissionError(f"{relative} must be committed at HEAD")
    if completed.stdout != resolved.read_bytes():
        raise PermissionError(f"{relative} differs from committed HEAD")


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cohort = _load_json(DEFAULT_COHORT, name="v3 cohort")
    cluster_lock = _load_json(DEFAULT_CLUSTER_LOCK, name="cluster lock")
    census = _load_json(DEFAULT_CENSUS, name="v3 census")
    _validate_frozen_inputs(cohort, cluster_lock, census)
    return cohort, cluster_lock, census


def load_authorization_lock(
    *,
    require_committed: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cohort, cluster_lock, census = load_frozen_inputs()
    lock = _load_json(DEFAULT_AUTHORIZATION_LOCK, name="authorization lock")
    validate_authorization_lock(
        lock,
        cohort=cohort,
        cluster_lock=cluster_lock,
        census=census,
    )
    if require_committed:
        for path in (
            DEFAULT_AUTHORIZATION_LOCK,
            DEFAULT_COHORT,
            DEFAULT_CLUSTER_LOCK,
            DEFAULT_CENSUS,
            *_IMPLEMENTATION_PATHS.values(),
        ):
            _require_head_committed_file(Path(path))
    return cohort, cluster_lock, census, lock


def freeze_authorization_lock_file() -> dict[str, Any]:
    if DEFAULT_AUTHORIZATION_LOCK.exists():
        raise FileExistsError("authorization lock already exists")
    cohort, cluster_lock, census = load_frozen_inputs()
    lock = freeze_authorization_lock(cohort, cluster_lock, census)
    _atomic_write_new(DEFAULT_AUTHORIZATION_LOCK, lock)
    return lock


def _make_runtime(certificate_workers: int):
    from gemma_sv.demo_server.audit_gemma_runtime_v2 import AuditGemmaRuntimeV2
    from gemma_sv.demo_server.gemma_engine import RuntimeConfig

    config = RuntimeConfig(
        **runtime_v2._runtime_config_kwargs(device="mps")
    )
    return AuditGemmaRuntimeV2(
        config,
        certificate_workers=certificate_workers,
    )


def run_from_paths(
    *,
    output_root: str | Path,
    data_path: str | Path | None,
    certificate_workers: int,
    resume: bool,
    acknowledgement: str,
) -> dict[str, Any]:
    if acknowledgement != EXPLICIT_ACKNOWLEDGEMENT:
        raise PermissionError("exact explicit acknowledgement is required")
    workers = _validate_workers(certificate_workers)
    root = _safe_cli_output_root(output_root)
    cohort, _cluster_lock, _census, lock = load_authorization_lock(
        require_committed=True
    )
    decoded_v1._configure_determinism()
    with decoded_v1._offline_huggingface():
        rows = cohort_v3.base.load_pinned_longmemeval_rows(data_path)
        runtime = _make_runtime(workers)
        runtime.ensure_loaded()
        runtime_v2.verify_loaded_runtime(runtime, device="mps")
        records = cohort_v3.rehydrate_manifest(
            cohort,
            rows,
            runtime.tokenizer,
        )
        final = run_sharded_audit(
            runtime,
            records,
            cohort,
            lock,
            output_root=root,
            certificate_workers=workers,
            resume=resume,
            assemble=True,
        )
    assert final is not None
    return final


def assemble_from_paths(*, output_root: str | Path) -> dict[str, Any]:
    root = _safe_cli_output_root(output_root)
    _cohort, _cluster_lock, _census, lock = load_authorization_lock(
        require_committed=False
    )
    return assemble_final(root, lock)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-lock", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--assemble", action="store_true")
    parser.add_argument("--data-path")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--certificate-workers",
        type=int,
        default=DEFAULT_CERTIFICATE_WORKERS,
    )
    parser.add_argument("--acknowledgement", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.freeze_lock:
            freeze_authorization_lock_file()
        elif args.assemble:
            assemble_from_paths(output_root=args.output_root)
        else:
            final = run_from_paths(
                output_root=args.output_root,
                data_path=args.data_path,
                certificate_workers=args.certificate_workers,
                resume=bool(args.resume),
                acknowledgement=args.acknowledgement,
            )
            return 0 if final["status"] == "completed" else 1
    except (
        AuditV2Error,
        FileExistsError,
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
        cohort_v3.ManifestError,
    ) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate model admission for the frozen LongMemEval V1 chat protocol.

The evaluator scores only the two predeclared official probes
(``target_current`` and ``retained``) under the present registered-chat state
and a fresh prefill of the raw-omitted chat.  Reports contain hashes and scalar
measurements, never source questions, answers, prompts, or generated text.

Development may evaluate either the 4B-IT ungrafted or training-free-grafted
arm.  Confirmation is deliberately harder to start: it requires both an
independently frozen promotion lock and an explicit confirmation-scoring flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import time
from typing import Any, Mapping, Sequence

from gemma_sv import longmemeval_chat_benchmark as benchmark
from gemma_sv.eval_longmemeval_deletion import (
    MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK,
    MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS,
    RETAINED_MAX_FIRST_TOKEN_RANK,
    strict_admission_decomposition,
)
from gemma_sv.persistent_deletion import (
    first_token_rank,
    method_storage_report,
    persistent_state_shape_signature,
)


SCHEMA = "gemma-sv-longmemeval-chat-admission-evaluation-v1"
SCHEMA_VERSION = 1
PROMOTION_LOCK_SCHEMA = "gemma-sv-longmemeval-chat-promotion-lock-v1"
PROMOTION_LOCK_STATUS = "frozen-before-confirmation-model-scoring"

GRAFTED_ARM = "it_native_chat_training_free_graft"
UNGRAFTED_ARM = "it_native_chat_ungrafted"
ARMS = frozenset({GRAFTED_ARM, UNGRAFTED_ARM})

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_DEVELOPMENT_MANIFEST = (
    BENCHMARKS / "longmemeval_chat_development_v1.json"
)
DEFAULT_CONFIRMATION_MANIFEST = (
    BENCHMARKS / "longmemeval_chat_confirmation_v1.json"
)
DEFAULT_POLICY_LOCK = BENCHMARKS / "longmemeval_chat_policy_lock_v1.json"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_admission.json"
)

# These bind the evaluator to the exact committed V1 chat artifacts rather than
# any later manifest that happens to satisfy the same general schema.
PINNED_MANIFEST_INTEGRITIES = {
    benchmark.DEVELOPMENT_PARTITION: (
        "2e49e26ef6911024830594a88e186b2d77229a9a3feed3630ce83da074e232a7"
    ),
    benchmark.CONFIRMATION_PARTITION: (
        "48eb12371e677bc18732f900049dcdfc5791f31344a1f4dfbcfcf06bcfe07ce5"
    ),
}
PINNED_POLICY_LOCK_SHA256 = (
    "5ff9da549494ab273c7656546d27dcb0bf2922932abb4632a6aef3d1b4fa393c"
)

EVALUATION_SEED = 0
DTYPE = "float32"
WINDOW = 1_024
NU = 0.7
CHUNK = 128
SOLVER_SEED = 0
PRESERVE_PREFIX_MASS = True
PER_BOUNDARY_BOX = True

_FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "answer",
        "content",
        "generated_text",
        "messages",
        "prompt",
        "question",
        "sessions",
        "source_text",
        "turns",
    }
)

_IMPLEMENTATION_PATHS = {
    "eval_longmemeval_chat.py": Path(__file__).resolve(),
    "longmemeval_chat_benchmark.py": PACKAGE
    / "longmemeval_chat_benchmark.py",
    "longmemeval_deletion_benchmark.py": PACKAGE
    / "longmemeval_deletion_benchmark.py",
    "eval_longmemeval_deletion.py": PACKAGE
    / "eval_longmemeval_deletion.py",
    "persistent_deletion.py": PACKAGE / "persistent_deletion.py",
    "graft.py": PACKAGE / "graft.py",
    "recovery_state.py": PACKAGE / "recovery_state.py",
    "sv_global_attention.py": PACKAGE / "sv_global_attention.py",
    "layer_select.py": PACKAGE / "layer_select.py",
    "demo_server/gemma_engine.py": PACKAGE
    / "demo_server"
    / "gemma_engine.py",
    "svattn/causal_sv_attention.py": WORKSPACE
    / "svattn"
    / "causal_sv_attention.py",
    "svattn/mlx_svdd.py": WORKSPACE / "svattn" / "mlx_svdd.py",
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


def _require_sha256(value: Any, *, name: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return rendered


def _load_json_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_keys(item)


def _assert_source_free_payload(payload: Mapping[str, Any]) -> None:
    leaked = _FORBIDDEN_SOURCE_KEYS.intersection(_walk_keys(payload))
    if leaked:
        raise ValueError(
            "source-free payload contains forbidden fields: "
            + ", ".join(sorted(leaked))
        )


def _validate_pinned_manifest_payload(
    manifest: Mapping[str, Any],
) -> None:
    benchmark.validate_manifest(manifest)
    partition = str(manifest.get("partition") or "")
    expected = PINNED_MANIFEST_INTEGRITIES.get(partition)
    observed = str((manifest.get("integrity") or {}).get("sha256") or "")
    if expected is None or observed != expected:
        raise ValueError("manifest is not the committed LongMemEval chat V1 artifact")
    lock = manifest.get("policy_lock") or {}
    if lock.get("lock_sha256") != PINNED_POLICY_LOCK_SHA256:
        raise ValueError("manifest is not bound to the committed chat policy lock")


def load_locked_inputs(
    manifest_path: str | Path,
    policy_lock_path: str | Path = DEFAULT_POLICY_LOCK,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the exact committed manifest and its standalone policy lock."""

    manifest = benchmark.load_manifest(manifest_path)
    _validate_pinned_manifest_payload(manifest)
    lock = _load_json_mapping(policy_lock_path, name="policy lock")
    benchmark.validate_policy_lock(lock)
    if lock.get("lock_sha256") != PINNED_POLICY_LOCK_SHA256:
        raise ValueError("standalone policy lock is not the committed V1 lock")
    if lock != manifest.get("policy_lock"):
        raise ValueError("standalone policy lock differs from the manifest lock")
    return manifest, lock


def _graft_enabled(arm: str) -> bool:
    if arm not in ARMS:
        raise ValueError(f"unsupported LongMemEval chat arm {arm!r}")
    return arm == GRAFTED_ARM


def _normalized_device(device: str) -> str:
    rendered = str(device).strip()
    if not rendered:
        raise ValueError("runtime device must not be empty")
    return rendered


def runtime_contract(arm: str, *, device: str) -> dict[str, Any]:
    """Return the exact runtime contract for one development arm."""

    grafted = _graft_enabled(arm)
    rendered_device = _normalized_device(device)
    return {
        "arm": arm,
        "model_id": benchmark.MODEL_ID,
        "model_revision": benchmark.MODEL_REVISION,
        "adapter": None,
        "device": rendered_device,
        "dtype": DTYPE,
        "window": WINDOW,
        "graft_enabled": grafted,
        "nu": NU if grafted else None,
        "chunk": CHUNK if grafted else None,
        "readout": "softmax" if grafted else None,
        "preserve_prefix_mass": PRESERVE_PREFIX_MASS if grafted else None,
        "per_boundary_box": PER_BOUNDARY_BOX if grafted else None,
        "solver_seed": SOLVER_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "training_steps": 0,
    }


def _runtime_config_kwargs(arm: str, *, device: str) -> dict[str, Any]:
    grafted = _graft_enabled(arm)
    return {
        "model_id": benchmark.MODEL_ID,
        "model_revision": benchmark.MODEL_REVISION,
        "lora_path": None,
        "device": _normalized_device(device),
        "dtype": DTYPE,
        "generation_tokens": 1,
        "window": WINDOW,
        "copies": 1,
        "query_gate_floor": 0.0,
        "graft_enabled": grafted,
        "nu": NU if grafted else None,
        "preserve_prefix_mass": (
            PRESERVE_PREFIX_MASS if grafted else None
        ),
        "per_boundary_box": PER_BOUNDARY_BOX if grafted else None,
        "solver_seed": SOLVER_SEED,
    }


def _make_runtime(arm: str, *, device: str):
    from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig

    return GemmaRuntime(RuntimeConfig(**_runtime_config_kwargs(arm, device=device)))


def verify_loaded_runtime(runtime: Any, *, arm: str, device: str) -> None:
    """Reject model, tokenizer, adapter, precision, or graft drift."""

    expected = _runtime_config_kwargs(arm, device=device)
    config = runtime.config
    observed = {
        key: getattr(config, key, None)
        for key in expected
    }
    if observed != expected:
        raise RuntimeError("loaded runtime configuration differs from the lock")
    if str(getattr(runtime, "resolved_model_revision", None)) != (
        benchmark.MODEL_REVISION
    ):
        raise RuntimeError("loaded model revision differs from the chat manifest")
    tokenizer = getattr(runtime, "tokenizer", None)
    if tokenizer is None or getattr(tokenizer, "is_fast", True) is not True:
        raise RuntimeError("loaded runtime requires the frozen fast tokenizer")
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    tokenizer_commit = (
        init_kwargs.get("_commit_hash")
        if isinstance(init_kwargs, Mapping)
        else None
    )
    if (
        tokenizer_commit is not None
        and str(tokenizer_commit) != benchmark.CHAT_TOKENIZER_REVISION
    ):
        raise RuntimeError("loaded tokenizer revision differs from the manifest")

    layers = getattr(runtime, "layers", {}) or {}
    if _graft_enabled(arm):
        if (
            float(getattr(runtime, "resolved_nu", -1.0)) != NU
            or getattr(runtime, "resolved_preserve_prefix_mass", None)
            is not PRESERVE_PREFIX_MASS
            or getattr(runtime, "resolved_per_boundary_box", None)
            is not PER_BOUNDARY_BOX
            or int(getattr(runtime, "resolved_solver_seed", -1))
            != SOLVER_SEED
            or not layers
        ):
            raise RuntimeError("resolved training-free graft differs from the lock")
        layer_contracts = {
            (
                float(getattr(layer.self_attn, "nu", -1.0)),
                int(getattr(layer.self_attn, "chunk", -1)),
                str(getattr(layer.self_attn, "readout", "")),
                bool(
                    getattr(
                        layer.self_attn,
                        "preserve_prefix_mass",
                        False,
                    )
                ),
                bool(
                    getattr(
                        layer.self_attn,
                        "per_boundary_box",
                        False,
                    )
                ),
                int(getattr(layer.self_attn, "solver_seed", -1)),
            )
            for layer in layers.values()
        }
        if layer_contracts != {
            (
                NU,
                CHUNK,
                "softmax",
                PRESERVE_PREFIX_MASS,
                PER_BOUNDARY_BOX,
                SOLVER_SEED,
            )
        }:
            raise RuntimeError("resolved graft layer policy differs from the lock")
    elif layers:
        raise RuntimeError("ungrafted arm unexpectedly contains grafted layers")
    if getattr(runtime, "recovery_state", None) is not None:
        raise RuntimeError("no-adapter admission runtime loaded recovery state")


def _confirmation_policy(records: int) -> dict[str, Any]:
    return {
        "attempt_every_frozen_record": True,
        "records": int(records),
        "no_replacement": True,
        "model_or_prompt_changes_forbidden": True,
        "selection_uses_confirmation_outputs": False,
        "threshold_overrides_allowed": False,
        "admission_thresholds": {
            "minimum_target_present_minus_fresh_raw_omission_nats": (
                MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS
            ),
            "maximum_target_present_first_token_rank": (
                MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK
            ),
            "maximum_retained_first_token_rank_in_present_and_raw_omission": (
                RETAINED_MAX_FIRST_TOKEN_RANK
            ),
        },
    }


def freeze_promotion_lock(
    development_report: Mapping[str, Any],
    development_manifest: Mapping[str, Any],
    confirmation_manifest: Mapping[str, Any],
    *,
    development_report_sha256: str,
    selected_arm: str,
    device: str,
) -> dict[str, Any]:
    """Bind a completed full development run to a confirmation arm.

    This helper returns a lock but never writes it.  Freezing and reviewing that
    lock is intentionally a separate step from confirmation evaluation.
    """

    _validate_pinned_manifest_payload(development_manifest)
    _validate_pinned_manifest_payload(confirmation_manifest)
    if (
        development_manifest["partition"] != benchmark.DEVELOPMENT_PARTITION
        or confirmation_manifest["partition"]
        != benchmark.CONFIRMATION_PARTITION
        or development_manifest["policy_lock"]
        != confirmation_manifest["policy_lock"]
    ):
        raise ValueError("promotion requires the committed partition pair")
    _require_sha256(
        development_report_sha256,
        name="development report SHA-256",
    )
    _assert_source_free_payload(development_report)
    denominators = (development_report.get("summary") or {}).get(
        "denominators"
    ) or {}
    implementation = development_report.get("implementation") or {}
    report_manifest = development_report.get("manifest") or {}
    report_config = development_report.get("config") or {}
    expected_records = len(development_manifest["records"])
    expected_record_ids = [
        str(record["record_id"]) for record in development_manifest["records"]
    ]
    expected_runtime = runtime_contract(selected_arm, device=device)
    if (
        development_report.get("schema") != SCHEMA
        or development_report.get("status") != "completed"
        or development_report.get("contains_source_text") is not False
        or development_report.get(
            "model_scoring_used_for_selection_or_replacement"
        )
        is not False
        or report_manifest.get("partition")
        != benchmark.DEVELOPMENT_PARTITION
        or report_manifest.get("integrity_sha256")
        != development_manifest["integrity"]["sha256"]
        or int(report_manifest.get("frozen_records", -1)) != expected_records
        or report_manifest.get("selected_record_ids") != expected_record_ids
        or (development_report.get("policy_lock") or {}).get("lock_sha256")
        != PINNED_POLICY_LOCK_SHA256
        or any(
            report_config.get(key) != value
            for key, value in expected_runtime.items()
        )
        or int(denominators.get("frozen_records", -1)) != expected_records
        or int(denominators.get("selected_records", -1)) != expected_records
        or int(denominators.get("attempted_records", -1)) != expected_records
        or int(denominators.get("completed_records", -1)) != expected_records
        or int(denominators.get("record_failures", -1)) != 0
    ):
        raise ValueError("development report is not a complete promotable run")
    implementation_contract = _require_sha256(
        implementation.get("contract_sha256"),
        name="development implementation contract",
    )
    joint = int(denominators.get("joint_admitted", -1))
    if not 0 <= joint <= expected_records:
        raise ValueError("development joint-admission denominator is invalid")

    payload: dict[str, Any] = {
        "schema": PROMOTION_LOCK_SCHEMA,
        "status": PROMOTION_LOCK_STATUS,
        "contains_source_text": False,
        "selection_uses_confirmation_outputs": False,
        "development_manifest_integrity_sha256": development_manifest[
            "integrity"
        ]["sha256"],
        "confirmation_manifest_integrity_sha256": confirmation_manifest[
            "integrity"
        ]["sha256"],
        "policy_lock_sha256": PINNED_POLICY_LOCK_SHA256,
        "development_evidence": {
            "report_sha256": str(development_report_sha256),
            "report_schema": SCHEMA,
            "report_status": "completed",
            "implementation_contract_sha256": implementation_contract,
            "arm": selected_arm,
            "frozen_records": expected_records,
            "attempted_records": expected_records,
            "completed_records": expected_records,
            "record_failures": 0,
            "joint_admitted": joint,
        },
        "decision": runtime_contract(selected_arm, device=device),
        "confirmation_policy": _confirmation_policy(
            len(confirmation_manifest["records"])
        ),
    }
    payload["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(payload),
    }
    validate_promotion_lock(payload, confirmation_manifest)
    return payload


def validate_promotion_lock(
    promotion: Mapping[str, Any],
    confirmation_manifest: Mapping[str, Any],
) -> None:
    """Validate an independently frozen confirmation authorization."""

    _validate_pinned_manifest_payload(confirmation_manifest)
    if confirmation_manifest["partition"] != benchmark.CONFIRMATION_PARTITION:
        raise ValueError("promotion validation requires confirmation partition")
    _assert_source_free_payload(promotion)
    integrity = promotion.get("integrity") or {}
    body = dict(promotion)
    body.pop("integrity", None)
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(body)
    ):
        raise ValueError("promotion lock integrity differs")
    if (
        promotion.get("schema") != PROMOTION_LOCK_SCHEMA
        or promotion.get("status") != PROMOTION_LOCK_STATUS
        or promotion.get("contains_source_text") is not False
        or promotion.get("selection_uses_confirmation_outputs") is not False
        or promotion.get("development_manifest_integrity_sha256")
        != PINNED_MANIFEST_INTEGRITIES[benchmark.DEVELOPMENT_PARTITION]
        or promotion.get("confirmation_manifest_integrity_sha256")
        != confirmation_manifest["integrity"]["sha256"]
        or promotion.get("policy_lock_sha256")
        != PINNED_POLICY_LOCK_SHA256
    ):
        raise ValueError("promotion lock binding differs")

    decision = promotion.get("decision") or {}
    arm = str(decision.get("arm") or "")
    device = str(decision.get("device") or "")
    if decision != runtime_contract(arm, device=device):
        raise ValueError("promotion runtime decision differs")
    if promotion.get("confirmation_policy") != _confirmation_policy(
        len(confirmation_manifest["records"])
    ):
        raise ValueError("promotion confirmation policy differs")

    evidence = promotion.get("development_evidence") or {}
    expected_records = int(
        confirmation_manifest["policy_lock"]["partition_policy"][
            "development_records"
        ]
    )
    if (
        evidence.get("report_schema") != SCHEMA
        or evidence.get("report_status") != "completed"
        or evidence.get("arm") != arm
        or int(evidence.get("frozen_records", -1)) != expected_records
        or int(evidence.get("attempted_records", -1)) != expected_records
        or int(evidence.get("completed_records", -1)) != expected_records
        or int(evidence.get("record_failures", -1)) != 0
        or not 0
        <= int(evidence.get("joint_admitted", -1))
        <= expected_records
    ):
        raise ValueError("promotion development evidence differs")
    _require_sha256(
        evidence.get("report_sha256"),
        name="promoted development report",
    )
    _require_sha256(
        evidence.get("implementation_contract_sha256"),
        name="promoted implementation contract",
    )


def authorize_evaluation(
    manifest: Mapping[str, Any],
    *,
    requested_arm: str | None,
    requested_device: str | None,
    selected_records: int,
    promotion_lock: Mapping[str, Any] | None,
    confirmation_scoring_acknowledged: bool,
) -> tuple[str, str]:
    """Authorize development directly and confirmation only through two locks."""

    _validate_pinned_manifest_payload(manifest)
    partition = manifest["partition"]
    if partition == benchmark.DEVELOPMENT_PARTITION:
        if promotion_lock is not None:
            raise ValueError("development evaluation must not consume a promotion lock")
        if confirmation_scoring_acknowledged:
            raise ValueError("confirmation acknowledgement is invalid for development")
        if requested_arm not in ARMS:
            raise ValueError("development evaluation requires an explicit --arm")
        device = _normalized_device(requested_device or "mps")
        runtime_contract(requested_arm, device=device)
        return requested_arm, device

    if not confirmation_scoring_acknowledged:
        raise ValueError(
            "confirmation scoring requires --allow-confirmation-scoring"
        )
    if promotion_lock is None:
        raise ValueError("confirmation scoring requires a frozen promotion lock")
    if requested_arm is not None:
        raise ValueError("confirmation arm must come only from the promotion lock")
    if selected_records != len(manifest["records"]):
        raise ValueError("confirmation must attempt every frozen record")
    validate_promotion_lock(promotion_lock, manifest)
    decision = promotion_lock["decision"]
    locked_device = str(decision["device"])
    if (
        requested_device is not None
        and _normalized_device(requested_device) != locked_device
    ):
        raise ValueError("confirmation device differs from the promotion lock")
    return str(decision["arm"]), locked_device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed) % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _synchronize(device: str) -> None:
    try:
        import torch
    except ImportError:
        return
    normalized = str(device).casefold()
    if normalized.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif normalized.startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize empty timing samples")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "median": float(statistics.median(samples)),
        "p95": _percentile(samples, 0.95),
        "minimum": min(samples),
        "maximum": max(samples),
        "samples": samples,
    }


def _timing_payload(
    updates: Sequence[float],
    queries: Sequence[float],
    totals: Sequence[float],
    *,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    return {
        "clock": "time.perf_counter",
        "synchronized_device": str(device),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "query_scope": (
            "full target_current and retained sequences plus first-token ranks"
        ),
        "update_seconds": _timing_summary(updates),
        "query_seconds": _timing_summary(queries),
        "end_to_end_seconds": _timing_summary(totals),
    }


def _fingerprint_value(
    digest: Any,
    value: Any,
    *,
    seen: set[int],
) -> None:
    """Hash nested metadata and complete Torch/NumPy tensor contents."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        encoded = repr((type(value).__name__, value)).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        return
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        header = repr(
            ("torch", str(value.dtype), tuple(int(size) for size in value.shape))
        ).encode("utf-8")
        digest.update(struct.pack(">Q", len(header)))
        digest.update(header)
        flat = value.detach().reshape(-1)
        for start in range(0, int(flat.numel()), 1_048_576):
            chunk = flat[start : start + 1_048_576].to("cpu").contiguous()
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        return
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        header = repr(
            ("numpy", str(array.dtype), tuple(int(size) for size in array.shape))
        ).encode("utf-8")
        digest.update(struct.pack(">Q", len(header)))
        digest.update(header)
        digest.update(array.view(np.uint8).tobytes())
        return

    identity = id(value)
    if identity in seen:
        digest.update(b"<cycle>")
        return
    seen.add(identity)
    try:
        if isinstance(value, Mapping):
            digest.update(b"{")
            for key in sorted(value, key=repr):
                _fingerprint_value(digest, key, seen=seen)
                _fingerprint_value(digest, value[key], seen=seen)
            digest.update(b"}")
            return
        if isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("utf-8") + b"[")
            for item in value:
                _fingerprint_value(digest, item, seen=seen)
            digest.update(b"]")
            return
        if isinstance(value, (set, frozenset)):
            digest.update(type(value).__name__.encode("utf-8") + b"[")
            for item in sorted(value, key=repr):
                _fingerprint_value(digest, item, seen=seen)
            digest.update(b"]")
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            digest.update(type(value).__name__.encode("utf-8") + b"(")
            for name in sorted(attributes):
                if name.startswith("__"):
                    continue
                _fingerprint_value(digest, name, seen=seen)
                _fingerprint_value(digest, attributes[name], seen=seen)
            digest.update(b")")
            return
        _fingerprint_value(
            digest,
            repr((type(value).__name__, value)),
            seen=seen,
        )
    finally:
        seen.remove(identity)


def persistent_state_fingerprint(memory: Any) -> str:
    """Hash source-state metadata, shapes, and complete tensor values."""

    digest = hashlib.sha256()
    _fingerprint_value(
        digest,
        {
            "token_count": int(memory.token_count),
            "input_digest": str(memory.input_digest),
            "token_ids": tuple(getattr(memory, "token_ids", ()) or ()),
            "deleted_positions": tuple(
                getattr(memory, "deleted_positions", ()) or ()
            ),
            "deletion_kind": getattr(memory, "deletion_kind", None),
            "fallback_reason": getattr(memory, "fallback_reason", None),
            "request": getattr(memory, "request", None),
            "layer_sessions": getattr(memory, "layer_sessions", {}),
            "past_key_values": getattr(memory, "past_key_values", None),
        },
        seen=set(),
    )
    return digest.hexdigest()


def _record_seed(record_id: str) -> int:
    digest = hashlib.sha256(
        f"longmemeval-chat-admission-v1\0{EVALUATION_SEED}\0{record_id}".encode(
            "utf-8"
        )
    ).digest()
    return EVALUATION_SEED + int.from_bytes(digest[:4], "big")


def _score_suite(
    runtime: Any,
    memory: Any,
    probes: Sequence[benchmark.ChatProbe],
) -> dict[str, Mapping[str, Any]]:
    return {
        probe.probe_id: runtime.score_persistent(
            memory,
            probe.prompt_text,
            list(probe.target_token_ids),
        )
        for probe in probes
    }


def _measure_existing_state(
    runtime: Any,
    memory: Any,
    probes: Sequence[benchmark.ChatProbe],
    *,
    device: str,
    warmup: int,
    repeats: int,
    operation_seed: int,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    queries: list[float] = []
    last_scores = None
    for iteration in range(warmup + repeats):
        _seed_everything(operation_seed)
        _synchronize(device)
        started = time.perf_counter()
        scores = _score_suite(runtime, memory, probes)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if iteration >= warmup:
            queries.append(elapsed)
            last_scores = scores
    if last_scores is None:
        raise RuntimeError("present benchmark produced no measured repetition")
    zeros = [0.0] * repeats
    timing = _timing_payload(
        zeros,
        queries,
        queries,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    timing["update_seconds"]["not_applicable"] = True
    timing["update_seconds"]["reason"] = "shared original prefill"
    return last_scores, timing


def _measure_fresh_raw_omission(
    runtime: Any,
    token_ids: Sequence[int],
    probes: Sequence[benchmark.ChatProbe],
    *,
    device: str,
    warmup: int,
    repeats: int,
    operation_seed: int,
) -> tuple[Any, dict[str, Mapping[str, Any]], dict[str, Any]]:
    updates: list[float] = []
    queries: list[float] = []
    totals: list[float] = []
    last_memory = None
    last_scores = None
    for iteration in range(warmup + repeats):
        _seed_everything(operation_seed)
        _synchronize(device)
        total_started = time.perf_counter()
        update_started = total_started
        memory = runtime.prefill_persistent(list(token_ids))
        _synchronize(device)
        update_elapsed = time.perf_counter() - update_started
        query_started = time.perf_counter()
        scores = _score_suite(runtime, memory, probes)
        _synchronize(device)
        query_elapsed = time.perf_counter() - query_started
        total_elapsed = time.perf_counter() - total_started
        if iteration >= warmup:
            updates.append(update_elapsed)
            queries.append(query_elapsed)
            totals.append(total_elapsed)
            last_memory = memory
            last_scores = scores
    if last_memory is None or last_scores is None:
        raise RuntimeError("raw-omission benchmark produced no measured repetition")
    return (
        last_memory,
        last_scores,
        _timing_payload(
            updates,
            queries,
            totals,
            device=device,
            warmup=warmup,
            repeats=repeats,
        ),
    )


def _serialize_score(
    score: Mapping[str, Any],
    probe: benchmark.ChatProbe,
) -> dict[str, Any]:
    target_ids = tuple(int(token_id) for token_id in probe.target_token_ids)
    first_log_probs = score.get("first_log_probs")
    if first_log_probs is None:
        raise ValueError("runtime score omitted the first-token distribution")
    total = float(score["total_log_probability"])
    mean = float(score["mean_log_probability"])
    geometric = float(score.get("geometric_mean_probability", math.exp(mean)))
    if not all(math.isfinite(value) for value in (total, mean, geometric)):
        raise ValueError("runtime produced a non-finite sequence score")
    first_target = target_ids[0]
    first_log_probability = float(first_log_probs[first_target])
    if not math.isfinite(first_log_probability):
        raise ValueError("runtime produced a non-finite first-token score")
    return {
        "target_token_count": len(target_ids),
        "all_target_tokens_teacher_forced": True,
        "total_log_probability": total,
        "mean_log_probability": mean,
        "geometric_mean_probability": geometric,
        "first_target_token_probability": math.exp(first_log_probability),
        "first_target_token_rank": first_token_rank(
            first_log_probs,
            first_target,
        ),
    }


def _score_snapshot(
    probes: Sequence[benchmark.ChatProbe],
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        probe.probe_id: _serialize_score(scores[probe.probe_id], probe)
        for probe in probes
    }


def _reported_first_token_rank(score: Mapping[str, Any]) -> int:
    for key in ("first_target_token_rank", "first_token_rank"):
        if key in score:
            return int(score[key])
    raise ValueError("score is missing first-token rank")


def admission_decomposition(
    present: Mapping[str, Mapping[str, Any]],
    fresh_raw_omission: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the established strict LongMemEval target and retained gates."""

    required = {"target_current", "retained"}
    if set(present) != required or set(fresh_raw_omission) != required:
        raise ValueError("admission requires exactly target_current and retained")
    strict = strict_admission_decomposition(
        {
            "current": present["target_current"],
            "retained": present["retained"],
        },
        {
            "current": fresh_raw_omission["target_current"],
            "retained": fresh_raw_omission["retained"],
        },
    )
    target = strict["target_stored_signal"]
    primary = strict["primary_target_admission"]
    retained = strict["retained_availability"]
    joint = strict["joint_target_and_retained"]
    return {
        "scheme": "longmemeval_chat_strict_admission_v1",
        "target_recall": {
            "probe_id": "target_current",
            "status": primary["status"],
            "admitted": bool(primary["admitted"]),
            "reasons": list(primary["reasons"]),
            "present_minus_fresh_raw_omission_nats": float(
                target["current_answer_present_minus_full_repack_nats"]
            ),
            "present_first_token_rank": int(
                target["current_answer_present_first_token_rank"]
            ),
            "fresh_raw_omission_first_token_rank": (
                _reported_first_token_rank(
                    fresh_raw_omission["target_current"]
                )
            ),
            "lift_passed": bool(target["lift_component_passed"]),
            "present_rank_passed": bool(target["rank_component_passed"]),
        },
        "retained_availability": {
            "probe_id": "retained",
            "status": retained["status"],
            "available": bool(retained["available"]),
            "present_first_token_rank": int(
                retained["present_first_token_rank"]
            ),
            "fresh_raw_omission_first_token_rank": int(
                retained["full_repack_first_token_rank"]
            ),
            "both_states_must_pass": True,
        },
        "joint_target_and_retained": {
            "status": joint["status"],
            "admitted": bool(joint["admitted"]),
        },
        "thresholds": {
            "minimum_target_present_minus_fresh_raw_omission_nats": (
                MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS
            ),
            "maximum_target_present_first_token_rank": (
                MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK
            ),
            "maximum_retained_first_token_rank_in_both_states": (
                RETAINED_MAX_FIRST_TOKEN_RANK
            ),
            "target_raw_omission_rank_reported_but_not_gated": True,
        },
        "measurement": (
            "teacher-forced full official target sequences and first-token "
            "ranks; raw omission is a fresh registered-chat prefill"
        ),
    }


def _probe_descriptors(
    record: benchmark.RehydratedChatRecord,
) -> list[dict[str, Any]]:
    return [
        {
            "probe_id": probe.probe_id,
            "kind": probe.kind,
            "query_sha256": benchmark.base.text_sha256(probe.prompt_text),
            "target_sha256": benchmark.base.text_sha256(probe.answer),
            "target_token_count": len(probe.target_token_ids),
            "target_token_ids_sha256": benchmark.base.token_ids_sha256(
                probe.target_token_ids
            ),
        }
        for probe in record.probes
    ]


def evaluate_admission_record(
    runtime: Any,
    record: benchmark.RehydratedChatRecord,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Score one frozen record without editing the shared present state."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    probes = tuple(record.probes)
    if (
        [probe.probe_id for probe in probes]
        != ["target_current", "retained"]
        or [probe.kind for probe in probes] != ["deleted", "retained"]
    ):
        raise ValueError("rehydrated chat probes differ from the frozen pair")
    context = record.context
    if tuple(record.raw_omitted_token_ids) != tuple(context.edited_token_ids):
        raise ValueError("chat raw omission differs from the frozen token edit")
    record_seed = _record_seed(record.record_id)
    _seed_everything(record_seed)
    device = str(runtime.config.device)

    _synchronize(device)
    prefill_started = time.perf_counter()
    original_memory = runtime.prefill_persistent(
        list(context.original_token_ids)
    )
    _synchronize(device)
    original_prefill_seconds = time.perf_counter() - prefill_started
    original_digest = benchmark.base.token_ids_sha256(
        context.original_token_ids
    )
    if str(original_memory.input_digest) != original_digest:
        raise RuntimeError("original prefill digest differs from frozen token IDs")

    shape_before = persistent_state_shape_signature(original_memory)
    fingerprint_started = time.perf_counter()
    fingerprint_before = persistent_state_fingerprint(original_memory)
    fingerprint_before_seconds = time.perf_counter() - fingerprint_started

    present_scores, present_timing = _measure_existing_state(
        runtime,
        original_memory,
        probes,
        device=device,
        warmup=warmup,
        repeats=repeats,
        operation_seed=record_seed,
    )
    raw_memory, raw_scores, raw_timing = _measure_fresh_raw_omission(
        runtime,
        record.raw_omitted_token_ids,
        probes,
        device=device,
        warmup=warmup,
        repeats=repeats,
        operation_seed=record_seed + 1,
    )
    raw_digest = benchmark.base.token_ids_sha256(
        record.raw_omitted_token_ids
    )
    if str(raw_memory.input_digest) != raw_digest:
        raise RuntimeError("raw-omission prefill digest differs from frozen tokens")

    fingerprint_started = time.perf_counter()
    fingerprint_after = persistent_state_fingerprint(original_memory)
    fingerprint_after_seconds = time.perf_counter() - fingerprint_started
    shape_after = persistent_state_shape_signature(original_memory)
    source_unchanged = (
        shape_before == shape_after
        and fingerprint_before == fingerprint_after
        and str(original_memory.input_digest) == original_digest
    )
    if not source_unchanged:
        raise RuntimeError("admission scoring mutated the shared present state")

    present_snapshot = _score_snapshot(probes, present_scores)
    raw_snapshot = _score_snapshot(probes, raw_scores)
    feasibility = tuple(record.fixed_c_diagnostics)
    return {
        "record_id": record.record_id,
        "partition": record.partition,
        "record_seed": record_seed,
        "status": "completed",
        "admission_only": True,
        "contains_source_text": False,
        "context": {
            "original_token_count": len(context.original_token_ids),
            "fresh_raw_omission_token_count": len(
                record.raw_omitted_token_ids
            ),
            "owned_token_count": len(context.forget_positions),
            "original_text_sha256": benchmark.base.text_sha256(
                context.original_text
            ),
            "fresh_raw_omission_text_sha256": benchmark.base.text_sha256(
                context.edited_text
            ),
            "original_token_ids_sha256": original_digest,
            "fresh_raw_omission_token_ids_sha256": raw_digest,
            "fresh_registered_chat_prefill": True,
        },
        "probes": _probe_descriptors(record),
        "admission": admission_decomposition(
            present_snapshot,
            raw_snapshot,
        ),
        "references": {
            "present": {
                "scores": present_snapshot,
                "timing": present_timing,
                "prefill_input_digest": str(original_memory.input_digest),
                "storage": method_storage_report(
                    original_memory,
                    original_memory,
                ),
            },
            "fresh_raw_omission": {
                "scores": raw_snapshot,
                "timing": raw_timing,
                "prefill_input_digest": str(raw_memory.input_digest),
                "fresh_prefill": True,
                "owned_complete_round_omitted": True,
                "storage": method_storage_report(
                    original_memory,
                    raw_memory,
                ),
            },
        },
        "shared_original_prefill": {
            "seconds": original_prefill_seconds,
            "synchronized_device": device,
            "input_digest": str(original_memory.input_digest),
        },
        "fixed_c_source_precheck": {
            "affected_boundaries": len(feasibility),
            "feasible_boundaries": sum(
                bool(item.get("feasible")) for item in feasibility
            ),
            "infeasible_boundaries": sum(
                not bool(item.get("feasible")) for item in feasibility
            ),
            "all_affected_boundaries_feasible": all(
                bool(item.get("feasible")) for item in feasibility
            ),
            "reference_only_not_used_for_admission_scoring": True,
        },
        "source_state_immutability": {
            "verification_scope": (
                "metadata_shapes_and_complete_tensor_values"
            ),
            "before_sha256": fingerprint_before,
            "after_sha256": fingerprint_after,
            "shape_signature_unchanged": shape_before == shape_after,
            "unchanged": True,
            "audit_seconds": {
                "before": fingerprint_before_seconds,
                "after": fingerprint_after_seconds,
            },
        },
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(float(value) for value in values) / len(values))


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    frozen_records: int,
    selected_records: int,
) -> dict[str, Any]:
    """Report complete frozen, selected, attempted, and gate denominators."""

    if not 0 <= int(selected_records) <= int(frozen_records):
        raise ValueError("summary frozen/selected denominators are invalid")
    if len(records) > int(selected_records):
        raise ValueError("summary has more attempts than selected records")
    record_ids = [str(record.get("record_id") or "") for record in records]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("summary record IDs are empty or duplicated")
    if any(record.get("status") not in {"completed", "failed"} for record in records):
        raise ValueError("summary contains an unsupported record status")
    completed = [
        record for record in records if record.get("status") == "completed"
    ]
    failed = [
        record for record in records if record.get("status") == "failed"
    ]
    target_admitted = sum(
        bool(record["admission"]["target_recall"]["admitted"])
        for record in completed
    )
    retained_available = sum(
        bool(record["admission"]["retained_availability"]["available"])
        for record in completed
    )
    joint_admitted = sum(
        bool(record["admission"]["joint_target_and_retained"]["admitted"])
        for record in completed
    )
    immutable = sum(
        bool(record["source_state_immutability"]["unchanged"])
        for record in completed
    )
    fixed_c_feasible = sum(
        bool(
            record["fixed_c_source_precheck"][
                "all_affected_boundaries_feasible"
            ]
        )
        for record in completed
    )
    affected_boundaries = sum(
        int(record["fixed_c_source_precheck"]["affected_boundaries"])
        for record in completed
    )
    infeasible_boundaries = sum(
        int(record["fixed_c_source_precheck"]["infeasible_boundaries"])
        for record in completed
    )
    denominators = {
        "frozen_records": int(frozen_records),
        "selected_records": int(selected_records),
        "not_selected_records": int(frozen_records) - int(selected_records),
        "attempted_records": len(records),
        "not_yet_attempted_selected_records": (
            int(selected_records) - len(records)
        ),
        "completed_records": len(completed),
        "record_failures": len(failed),
        "target_probes_attempted": len(completed),
        "retained_probes_attempted": len(completed),
        "target_admitted": target_admitted,
        "target_rejected": len(completed) - target_admitted,
        "retained_available": retained_available,
        "retained_unavailable": len(completed) - retained_available,
        "joint_admitted": joint_admitted,
        "joint_rejected": len(completed) - joint_admitted,
        "source_state_immutable": immutable,
        "source_state_immutability_failures": len(completed) - immutable,
        "present_reference_completed_records": len(completed),
        "fresh_raw_omission_reference_completed_records": len(completed),
        "fixed_c_all_boundaries_feasible_records": fixed_c_feasible,
        "fixed_c_infeasible_records": len(completed) - fixed_c_feasible,
        "fixed_c_affected_boundaries": affected_boundaries,
        "fixed_c_infeasible_boundaries": infeasible_boundaries,
        "target_sequence_tokens_scored_per_reference": sum(
            int(
                record["references"]["present"]["scores"]["target_current"][
                    "target_token_count"
                ]
            )
            for record in completed
        ),
        "retained_sequence_tokens_scored_per_reference": sum(
            int(
                record["references"]["present"]["scores"]["retained"][
                    "target_token_count"
                ]
            )
            for record in completed
        ),
    }
    summary: dict[str, Any] = {
        "aggregation_population": (
            "all selected frozen records; no admission filtering, replacement, "
            "or model-output selection"
        ),
        "denominators": denominators,
        "admission_thresholds": {
            "minimum_target_present_minus_fresh_raw_omission_nats": (
                MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS
            ),
            "maximum_target_present_first_token_rank": (
                MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK
            ),
            "maximum_retained_first_token_rank_in_both_states": (
                RETAINED_MAX_FIRST_TOKEN_RANK
            ),
        },
        "failure_record_ids": [
            str(record.get("record_id") or "") for record in failed
        ],
    }
    if completed:
        summary["scores"] = {
            "mean_target_present_sequence_log_probability": _mean(
                [
                    record["references"]["present"]["scores"][
                        "target_current"
                    ]["mean_log_probability"]
                    for record in completed
                ]
            ),
            "mean_target_fresh_raw_omission_sequence_log_probability": _mean(
                [
                    record["references"]["fresh_raw_omission"]["scores"][
                        "target_current"
                    ]["mean_log_probability"]
                    for record in completed
                ]
            ),
            "mean_target_present_minus_fresh_raw_omission_nats": _mean(
                [
                    record["admission"]["target_recall"][
                        "present_minus_fresh_raw_omission_nats"
                    ]
                    for record in completed
                ]
            ),
            "mean_retained_present_first_token_rank": _mean(
                [
                    record["admission"]["retained_availability"][
                        "present_first_token_rank"
                    ]
                    for record in completed
                ]
            ),
            "mean_retained_fresh_raw_omission_first_token_rank": _mean(
                [
                    record["admission"]["retained_availability"][
                        "fresh_raw_omission_first_token_rank"
                    ]
                    for record in completed
                ]
            ),
        }
        summary["timing"] = {
            "mean_original_prefill_seconds": _mean(
                [
                    record["shared_original_prefill"]["seconds"]
                    for record in completed
                ]
            ),
            "mean_present_query_median_seconds": _mean(
                [
                    record["references"]["present"]["timing"][
                        "query_seconds"
                    ]["median"]
                    for record in completed
                ]
            ),
            "mean_fresh_raw_omission_prefill_median_seconds": _mean(
                [
                    record["references"]["fresh_raw_omission"]["timing"][
                        "update_seconds"
                    ]["median"]
                    for record in completed
                ]
            ),
            "mean_fresh_raw_omission_query_median_seconds": _mean(
                [
                    record["references"]["fresh_raw_omission"]["timing"][
                        "query_seconds"
                    ]["median"]
                    for record in completed
                ]
            ),
        }
        summary["storage"] = {
            "mean_present_state_tensor_storage_bytes": _mean(
                [
                    record["references"]["present"]["storage"]["state"][
                        "deduplicated_tensor_storage_bytes"
                    ]
                    for record in completed
                ]
            ),
            "mean_fresh_raw_omission_state_tensor_storage_bytes": _mean(
                [
                    record["references"]["fresh_raw_omission"]["storage"][
                        "state"
                    ]["deduplicated_tensor_storage_bytes"]
                    for record in completed
                ]
            ),
            "mean_fresh_raw_omission_incremental_tensor_storage_bytes": _mean(
                [
                    record["references"]["fresh_raw_omission"]["storage"][
                        "incremental_tensor_storage_bytes"
                    ]
                    for record in completed
                ]
            ),
        }
    return summary


def _implementation_fingerprints() -> dict[str, str]:
    fingerprints = {
        label: _sha256_file(path)
        for label, path in sorted(_IMPLEMENTATION_PATHS.items())
    }
    fingerprints["contract_sha256"] = _payload_sha256(fingerprints)
    return fingerprints


def _environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in (
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "mlx",
    ):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {
        "platform": platform.platform(),
        "versions": versions,
    }


def _base_report(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    *,
    manifest_path: Path,
    policy_lock_path: Path,
    selected_record_ids: Sequence[str],
    arm: str,
    device: str,
    warmup: int,
    repeats: int,
    promotion_lock: Mapping[str, Any] | None,
    promotion_lock_path: Path | None,
) -> dict[str, Any]:
    promotion_binding = None
    if promotion_lock is not None:
        if promotion_lock_path is None:
            raise ValueError("promotion lock path is required")
        promotion_binding = {
            "schema": promotion_lock["schema"],
            "integrity_sha256": promotion_lock["integrity"]["sha256"],
            "file_sha256": _sha256_file(promotion_lock_path),
            "frozen_before_confirmation_model_scoring": True,
        }
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "evaluation": "LongMemEval V1 constrained-chat model admission",
        "official_longmemeval_leaderboard_score": False,
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "model_scoring_used_for_selection_or_replacement": False,
        "manifest": {
            "partition": manifest["partition"],
            "schema": manifest["schema"],
            "integrity_sha256": manifest["integrity"]["sha256"],
            "file_sha256": _sha256_file(manifest_path),
            "frozen_records": len(manifest["records"]),
            "selected_record_ids": list(selected_record_ids),
            "selected_record_ids_sha256": _payload_sha256(
                list(selected_record_ids)
            ),
            "committed_v1_artifact_required": True,
        },
        "policy_lock": {
            "schema": policy_lock["schema"],
            "lock_sha256": policy_lock["lock_sha256"],
            "file_sha256": _sha256_file(policy_lock_path),
            "locked_before_model_scoring": True,
        },
        "promotion_lock": promotion_binding,
        "artifacts": {
            "dataset_id": policy_lock["source_inventory"]["dataset_id"],
            "dataset_revision": policy_lock["source_inventory"][
                "dataset_revision"
            ],
            "source_artifact_sha256": policy_lock["source_inventory"][
                "source_artifact_sha256"
            ],
            "full_descriptors_sha256": policy_lock["source_inventory"][
                "full_descriptors_sha256"
            ],
            "confirmation_descriptors_sha256": policy_lock[
                "source_inventory"
            ]["confirmation_descriptors_sha256"],
        },
        "config": {
            **runtime_contract(arm, device=device),
            "warmup": int(warmup),
            "repeats": int(repeats),
            "admission_only": True,
        },
        "admission_protocol": {
            "target_probe": "target_current",
            "retained_probe": "retained",
            "full_target_sequences_scored": True,
            "first_target_token_ranks_scored": True,
            "present_reference": "registered chat with owned round present",
            "counterfactual_reference": (
                "fresh prefill of registered chat with owned round omitted"
            ),
            "minimum_target_lift_nats": (
                MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS
            ),
            "maximum_target_present_first_token_rank": (
                MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK
            ),
            "maximum_retained_first_token_rank_in_both_states": (
                RETAINED_MAX_FIRST_TOKEN_RANK
            ),
            "no_replacement": True,
        },
        "metric_definitions": {
            "sequence_score": (
                "teacher-forced total and mean log probability over every "
                "official target token"
            ),
            "rank": (
                "one-indexed full-vocabulary rank of the first official target "
                "token"
            ),
            "timing": (
                "device-synchronized original prefill, reference prefill, and "
                "two-probe query timing"
            ),
            "storage": "deduplicated tensor backing storage",
            "immutability": (
                "source-state metadata, shapes, and complete tensor values "
                "hashed before and after both reference queries"
            ),
        },
        "implementation": _implementation_fingerprints(),
        "environment": _environment(),
        "records": [],
        "summary": summarize_records(
            [],
            frozen_records=len(manifest["records"]),
            selected_records=len(selected_record_ids),
        ),
    }


def _validate_resume(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for key in (
        "schema",
        "schema_version",
        "evaluation",
        "official_longmemeval_leaderboard_score",
        "contains_source_text",
        "contains_full_vocabulary_vectors",
        "model_scoring_used_for_selection_or_replacement",
        "manifest",
        "policy_lock",
        "promotion_lock",
        "artifacts",
        "config",
        "admission_protocol",
        "metric_definitions",
        "implementation",
        "environment",
    ):
        if report.get(key) != expected.get(key):
            raise ValueError(f"resume report differs at {key}")
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("resume report records are invalid")
    ids = [str(record.get("record_id") or "") for record in records]
    selected = list(expected["manifest"]["selected_record_ids"])
    if (
        not all(ids)
        or len(ids) != len(set(ids))
        or any(record_id not in selected for record_id in ids)
        or ids != [record_id for record_id in selected if record_id in set(ids)]
    ):
        raise ValueError("resume record IDs differ from the frozen selection")
    _assert_source_free_payload(report)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    _assert_source_free_payload(payload)
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


def run_records(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    records: Sequence[benchmark.RehydratedChatRecord],
    runtime: Any,
    *,
    manifest_path: Path,
    policy_lock_path: Path,
    output: Path,
    arm: str,
    device: str,
    warmup: int,
    repeats: int,
    promotion_lock: Mapping[str, Any] | None = None,
    promotion_lock_path: Path | None = None,
    confirmation_scoring_acknowledged: bool = False,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate records with hash-bound, atomic, completion-aware resume."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    selected_ids = [record.record_id for record in records]
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected rehydrated record IDs are empty or duplicated")
    manifest_ids = {
        str(record["record_id"]) for record in manifest["records"]
    }
    if not set(selected_ids).issubset(manifest_ids):
        raise ValueError("rehydrated selection is outside the frozen manifest")
    authorized_arm, authorized_device = authorize_evaluation(
        manifest,
        requested_arm=(
            None
            if manifest["partition"] == benchmark.CONFIRMATION_PARTITION
            else arm
        ),
        requested_device=device,
        selected_records=len(records),
        promotion_lock=promotion_lock,
        confirmation_scoring_acknowledged=(
            confirmation_scoring_acknowledged
        ),
    )
    if arm != authorized_arm or str(device) != authorized_device:
        raise ValueError("runtime arm or device differs from authorization")
    if policy_lock != manifest["policy_lock"]:
        raise ValueError("runtime policy lock differs from the manifest")
    verify_loaded_runtime(runtime, arm=arm, device=device)

    expected = _base_report(
        manifest,
        policy_lock,
        manifest_path=manifest_path,
        policy_lock_path=policy_lock_path,
        selected_record_ids=selected_ids,
        arm=arm,
        device=device,
        warmup=warmup,
        repeats=repeats,
        promotion_lock=promotion_lock,
        promotion_lock_path=promotion_lock_path,
    )
    if output.exists() and not resume and not overwrite:
        raise ValueError(f"{output} exists; pass --resume or --overwrite")
    if resume:
        if not output.is_file():
            raise ValueError("cannot resume a missing report")
        report = _load_json_mapping(output, name="resume report")
        _validate_resume(report, expected)
    else:
        report = expected
    by_id = {
        str(record["record_id"]): dict(record)
        for record in report.get("records") or ()
    }
    resumed_completed = sum(
        row.get("status") == "completed" for row in by_id.values()
    )
    report["status"] = "running"
    report["records"] = [
        by_id[record_id] for record_id in selected_ids if record_id in by_id
    ]
    report["summary"] = summarize_records(
        report["records"],
        frozen_records=len(manifest["records"]),
        selected_records=len(records),
    )
    _atomic_write(output, report)

    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        existing = by_id.get(record.record_id)
        if existing is not None and existing.get("status") == "completed":
            print(
                f"[{index}/{len(records)}] {record.record_id}: complete",
                flush=True,
            )
            continue
        print(f"[{index}/{len(records)}] {record.record_id}", flush=True)
        try:
            row = evaluate_admission_record(
                runtime,
                record,
                warmup=warmup,
                repeats=repeats,
            )
        except Exception as exc:
            row = {
                "record_id": record.record_id,
                "partition": record.partition,
                "status": "failed",
                "contains_source_text": False,
                "error_type": type(exc).__name__,
                "error_message_redacted": True,
                "error_message_sha256": benchmark.base.text_sha256(str(exc)),
            }
        by_id[record.record_id] = row
        report["records"] = [
            by_id[record_id]
            for record_id in selected_ids
            if record_id in by_id
        ]
        report["summary"] = summarize_records(
            report["records"],
            frozen_records=len(manifest["records"]),
            selected_records=len(records),
        )
        report["elapsed_seconds_this_process"] = (
            time.perf_counter() - started
        )
        report["resumed_completed_records"] = resumed_completed
        report["status"] = "running"
        _atomic_write(output, report)

    report["summary"] = summarize_records(
        report["records"],
        frozen_records=len(manifest["records"]),
        selected_records=len(records),
    )
    denominators = report["summary"]["denominators"]
    report["status"] = (
        "completed"
        if denominators["completed_records"] == len(records)
        and denominators["record_failures"] == 0
        else "completed_with_record_failures"
    )
    report["elapsed_seconds_this_process"] = time.perf_counter() - started
    report["resumed_completed_records"] = resumed_completed
    _atomic_write(output, report)
    return report


def _load_pinned_rows(data_path: str | Path | None):
    return benchmark.base.load_pinned_longmemeval_rows(data_path)


def _select_indices(
    count: int,
    *,
    record_start: int,
    records: int | None,
) -> list[int]:
    if record_start < 0 or (records is not None and records < 1):
        raise ValueError("record selection must be non-negative and non-empty")
    selected = list(range(count))[record_start:]
    if records is not None:
        selected = selected[:records]
    if not selected:
        raise ValueError("record selection is empty")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_DEVELOPMENT_MANIFEST),
        help="exact committed development or confirmation V1 manifest",
    )
    parser.add_argument(
        "--policy-lock",
        default=str(DEFAULT_POLICY_LOCK),
        help="standalone committed LongMemEval chat V1 policy lock",
    )
    parser.add_argument(
        "--data-path",
        help="local pinned oracle JSON; omitted resolves the pinned Hub artifact",
    )
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument(
        "--device",
        help="development device (default mps); confirmation comes from promotion",
    )
    parser.add_argument("--promotion-lock")
    parser.add_argument(
        "--allow-confirmation-scoring",
        action="store_true",
        help="required in addition to a valid promotion lock for confirmation",
    )
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--records", type=int)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate, rehydrate, and score; importing this module scores nothing."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats positive")
    manifest_path = Path(args.manifest)
    policy_lock_path = Path(args.policy_lock)
    try:
        manifest, policy_lock = load_locked_inputs(
            manifest_path,
            policy_lock_path,
        )
        indices = _select_indices(
            len(manifest["records"]),
            record_start=args.record_start,
            records=args.records,
        )
        promotion = None
        promotion_path = None
        if args.promotion_lock:
            promotion_path = Path(args.promotion_lock)
            promotion = _load_json_mapping(
                promotion_path,
                name="promotion lock",
            )
        arm, device = authorize_evaluation(
            manifest,
            requested_arm=args.arm,
            requested_device=args.device,
            selected_records=len(indices),
            promotion_lock=promotion,
            confirmation_scoring_acknowledged=(
                args.allow_confirmation_scoring
            ),
        )
    except (OSError, ValueError, benchmark.ManifestError) as exc:
        parser.error(str(exc))

    output = Path(args.out)
    if output.exists() and not args.resume and not args.overwrite:
        parser.error(f"{output} exists; pass --resume or --overwrite")

    rows = _load_pinned_rows(args.data_path)
    runtime = _make_runtime(arm, device=device)
    runtime.ensure_loaded()
    try:
        verify_loaded_runtime(runtime, arm=arm, device=device)
        rehydrated = benchmark.rehydrate_manifest(
            manifest,
            rows,
            runtime.tokenizer,
        )
        selected = tuple(rehydrated[index] for index in indices)
        report = run_records(
            manifest,
            policy_lock,
            selected,
            runtime,
            manifest_path=manifest_path,
            policy_lock_path=policy_lock_path,
            output=output,
            arm=arm,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            promotion_lock=promotion,
            promotion_lock_path=promotion_path,
            confirmation_scoring_acknowledged=(
                args.allow_confirmation_scoring
            ),
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, benchmark.ManifestError) as exc:
        parser.error(str(exc))
    print(f"wrote evaluation to {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

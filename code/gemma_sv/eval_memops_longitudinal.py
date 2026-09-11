"""Evaluate deletion after a 10k-token naturalistic conversation suffix.

This runner consumes the source-free manifest produced by
``gemma_sv.memops_longitudinal`` and rehydrates the pinned MemOps artifacts.
Every compatible method starts from one shared persistent prefill.  Two
references remain deliberately separate:

* ``full_raw_repack`` removes the owned raw-text exchange, freshly tokenizes
  the joined transcript, and prefills from the beginning.
* ``fixed_c_refit`` keeps the already-contextualized retained keys and refits
  their original per-boundary fixed-C support-vector solves.

The exact-decrement certificate is against the latter.  Behavioral KL and
target/retained quality are reported against the former.
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
import struct
from typing import Any, Mapping, Sequence

from gemma_sv import eval_persistent_deletion_baselines as baseline
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.memops_longitudinal import (
    DEFAULT_CHUNK,
    DEFAULT_NU,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TOKENIZER_REVISION,
    LongitudinalProbe,
    LongitudinalRecord,
    load_manifest,
    rehydrate_manifest,
    text_sha256,
    token_ids_sha256,
)
from gemma_sv.persistent_deletion import (
    DECAY_FACTOR,
    cache_delete_and_shift,
    first_token_rank,
    full_vocabulary_kl,
    method_storage_report,
    persistent_state_shape_signature,
)


EVALUATION_SCHEMA = "gemma-sv-memops-longitudinal-evaluation-v1"
EVALUATION_SCHEMA_VERSION = 1
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / "memops_longitudinal_v1.json"
)
DEFAULT_MODEL = DEFAULT_TOKENIZER_ID
DEFAULT_MODEL_REVISION = DEFAULT_TOKENIZER_REVISION
DEFAULT_OUTPUT = "outputs/gemma_sv_memops/longitudinal_evaluation.json"
DEFAULT_WINDOW = 1024
MIN_TARGET_LIFT_NATS = 0.05
MAX_FIRST_TOKEN_RANK = 10
METHOD_IDS = (
    "full_raw_repack",
    "token_row_repack",
    "exact_decrement",
    "fixed_c_refit",
    "fp32_proxy",
    "cache_delete_shift",
    "decay_0_01",
)
PROMPT_SUPPRESSION_ID = "prompt_suppression"
IMPLEMENTATION_FILE_LABELS = (
    "eval_memops_longitudinal.py",
    "memops_longitudinal.py",
    "persistent_deletion.py",
    "eval_persistent_deletion_baselines.py",
    "graft.py",
    "recovery_state.py",
    "sv_global_attention.py",
    "layer_select.py",
    "demo_server/gemma_engine.py",
    "demo_server/gate_context.py",
    "demo_server/certificate.py",
    "svattn/causal_sv_attention.py",
    "svattn/mlx_svdd.py",
    "cp_svm/oneclass_incremental.py",
    "cp_svm/oneclass_fast.py",
    "cp_svm/binary_incremental.py",
    "cp_svm/kernels.py",
)


def expected_method_ids(
    *,
    prompt_suppression: bool,
) -> tuple[str, ...]:
    return METHOD_IDS + (
        (PROMPT_SUPPRESSION_ID,) if prompt_suppression else ()
    )


def _target_ids(
    runtime: Any,
    prompt: str,
    target: str,
) -> tuple[int, ...]:
    rendered = str(target)
    if prompt and not prompt[-1].isspace() and not rendered[:1].isspace():
        rendered = " " + rendered
    encoded = runtime.tokenizer(rendered, add_special_tokens=False)
    raw_ids = (
        encoded["input_ids"]
        if isinstance(encoded, Mapping)
        else encoded.input_ids
    )
    result = tuple(int(token_id) for token_id in raw_ids)
    if not result:
        raise ValueError("probe target tokenization is empty")
    return result


def build_probes(
    runtime: Any,
    record: LongitudinalRecord,
) -> tuple[baseline.Probe, ...]:
    """Convert source-validated probes to the shared persistent scorer."""

    probes = []
    for probe in record.probes:
        resolved_target_ids = _target_ids(
            runtime,
            probe.prompt,
            probe.target,
        )
        if (
            probe.target_token_ids
            and tuple(probe.target_token_ids) != resolved_target_ids
        ):
            raise ValueError(
                f"{probe.probe_id} prompt-conditioned target tokens differ "
                "from the frozen manifest"
            )
        probes.append(
            baseline.Probe(
                probe_id=probe.probe_id,
                kind=probe.kind,
                prompt=probe.prompt,
                target=probe.target,
                target_ids=resolved_target_ids,
                field_name=probe.probe_id,
            )
        )
    return tuple(probes)


def _serialize_score(
    score: Mapping[str, Any],
    probe: baseline.Probe,
) -> dict[str, Any]:
    first_log_probs = score["first_log_probs"]
    first_target = int(probe.target_ids[0])
    return {
        "target_token_count": len(probe.target_ids),
        "total_log_probability": float(score["total_log_probability"]),
        "mean_log_probability": float(score["mean_log_probability"]),
        "geometric_mean_probability": float(
            score["geometric_mean_probability"]
        ),
        "first_target_token_probability": math.exp(
            float(first_log_probs[first_target])
        ),
        "first_target_token_rank": first_token_rank(
            first_log_probs,
            first_target,
        ),
    }


def _score_snapshot(
    probes: Sequence[baseline.Probe],
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        probe.probe_id: _serialize_score(scores[probe.probe_id], probe)
        for probe in probes
    }


def admission_decomposition(
    probes: Sequence[baseline.Probe],
    present_scores: Mapping[str, Mapping[str, Any]],
    raw_repack_scores: Mapping[str, Mapping[str, Any]],
    *,
    minimum_target_lift_nats: float = MIN_TARGET_LIFT_NATS,
    maximum_first_token_rank: int = MAX_FIRST_TOKEN_RANK,
) -> dict[str, Any]:
    """Freeze target recall and retained availability as separate gates."""

    deleted_rows = []
    target_reasons = []
    retained_rows = []
    for probe in probes:
        present = present_scores[probe.probe_id]
        raw_repack = raw_repack_scores[probe.probe_id]
        present_rank = first_token_rank(
            present["first_log_probs"],
            probe.target_ids[0],
        )
        raw_repack_rank = first_token_rank(
            raw_repack["first_log_probs"],
            probe.target_ids[0],
        )
        lift = float(
            present["mean_log_probability"]
            - raw_repack["mean_log_probability"]
        )
        if probe.kind == "deleted":
            rank_passed = present_rank <= int(maximum_first_token_rank)
            lift_passed = lift >= float(minimum_target_lift_nats)
            if not rank_passed:
                target_reasons.append(f"{probe.probe_id}:rank")
            if not lift_passed:
                target_reasons.append(f"{probe.probe_id}:lift")
            deleted_rows.append(
                {
                    "probe_id": probe.probe_id,
                    "present_minus_raw_repack_nats": lift,
                    "present_first_token_rank": present_rank,
                    "raw_repack_first_token_rank": raw_repack_rank,
                    "rank_passed": rank_passed,
                    "lift_passed": lift_passed,
                }
            )
        else:
            available = (
                present_rank <= int(maximum_first_token_rank)
                and raw_repack_rank <= int(maximum_first_token_rank)
            )
            retained_rows.append({
                "probe_id": probe.probe_id,
                "available": available,
                "status": "available" if available else "unavailable",
                "present_first_token_rank": present_rank,
                "raw_repack_first_token_rank": raw_repack_rank,
            })
    if not deleted_rows or not retained_rows:
        raise ValueError("admission requires deleted and retained probes")
    target_admitted = not target_reasons
    retained_available = all(
        bool(row["available"]) for row in retained_rows
    )
    if len(retained_rows) == 1:
        retained_payload = retained_rows[0]
    else:
        retained_payload = {
            "probe_id": "all_retained",
            "available": retained_available,
            "status": "available" if retained_available else "unavailable",
            "present_first_token_rank": max(
                int(row["present_first_token_rank"])
                for row in retained_rows
            ),
            "raw_repack_first_token_rank": max(
                int(row["raw_repack_first_token_rank"])
                for row in retained_rows
            ),
            "all_probes_must_pass": True,
            "probes": retained_rows,
        }
    return {
        "target_recall": {
            "status": "admitted" if target_admitted else "rejected",
            "admitted": target_admitted,
            "reasons": sorted(target_reasons),
            "probes": deleted_rows,
        },
        "retained_availability": retained_payload,
        "joint_target_and_retained": {
            "status": (
                "admitted"
                if target_admitted and retained_available
                else "rejected"
            ),
            "admitted": target_admitted and retained_available,
        },
        "thresholds": {
            "minimum_present_minus_raw_repack_nats": float(
                minimum_target_lift_nats
            ),
            "maximum_present_first_token_rank": int(
                maximum_first_token_rank
            ),
            "all_deleted_probes_must_pass": True,
            "retained_reported_separately": True,
        },
        "measurement": (
            "teacher-forced semantic-gold sequence score and first-token rank; "
            "the floor is a freshly retokenized raw-exchange-omitted repack"
        ),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(float(value) for value in values) / len(values))


def _behavioral_metrics(
    probes: Sequence[baseline.Probe],
    method_scores: Mapping[str, Mapping[str, Any]],
    present_scores: Mapping[str, Mapping[str, Any]],
    raw_repack_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    deleted = []
    retained = None
    for probe in probes:
        method = method_scores[probe.probe_id]
        present = present_scores[probe.probe_id]
        reference = raw_repack_scores[probe.probe_id]
        row = {
            "probe_id": probe.probe_id,
            "score": _serialize_score(method, probe),
            "suppression_vs_present_nats": float(
                present["mean_log_probability"]
                - method["mean_log_probability"]
            ),
            "mean_log_probability_drift_from_raw_repack_nats": float(
                method["mean_log_probability"]
                - reference["mean_log_probability"]
            ),
            "full_vocabulary_kl_raw_repack_to_method_nats": (
                full_vocabulary_kl(
                    reference["first_log_probs"],
                    method["first_log_probs"],
                )
            ),
        }
        if probe.kind == "deleted":
            deleted.append(row)
        else:
            retained = row
    if not deleted or retained is None:
        raise ValueError("behavioral metrics require deleted and retained probes")
    return {
        "deleted_target_quality": {
            "probes": deleted,
            "mean_log_probability": _mean(
                [row["score"]["mean_log_probability"] for row in deleted]
            ),
            "mean_suppression_vs_present_nats": _mean(
                [row["suppression_vs_present_nats"] for row in deleted]
            ),
            "mean_drift_from_raw_repack_nats": _mean(
                [
                    row[
                        "mean_log_probability_drift_from_raw_repack_nats"
                    ]
                    for row in deleted
                ]
            ),
            "mean_full_vocabulary_kl_to_raw_repack_nats": _mean(
                [
                    row["full_vocabulary_kl_raw_repack_to_method_nats"]
                    for row in deleted
                ]
            ),
            "max_full_vocabulary_kl_to_raw_repack_nats": max(
                row["full_vocabulary_kl_raw_repack_to_method_nats"]
                for row in deleted
            ),
        },
        "retained_quality": retained,
        "reference": {
            "kind": "freshly_retokenized_raw_exchange_omitted_repack",
            "kl_direction": "KL(raw_repack || method)",
            "distribution_scope": "full vocabulary at first target token",
            "distinct_from_solver_certificate": True,
        },
    }


def _timing_scope(timing: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(timing)
    result["query_scope"] = (
        "two deleted semantic-gold probes and one paired retained probe"
    )
    return result


def _method_report(
    method_id: str,
    state: baseline.QueryState,
    scores: Mapping[str, Mapping[str, Any]],
    timing: Mapping[str, Any],
    *,
    original_memory: Any,
    probes: Sequence[baseline.Probe],
    present_scores: Mapping[str, Mapping[str, Any]],
    raw_repack_scores: Mapping[str, Mapping[str, Any]],
    semantics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "status": "completed",
        "semantics": dict(semantics),
        "tokenized_state": {
            "prefill_input_digest": str(state.memory.input_digest),
            "token_count": int(state.memory.token_count),
            "added_query_prefix_tokens": int(state.prompt_token_count),
        },
        **_behavioral_metrics(
            probes,
            scores,
            present_scores,
            raw_repack_scores,
        ),
        "timing": _timing_scope(timing),
        "storage": method_storage_report(
            original_memory,
            state.memory,
            prompt_token_count=state.prompt_token_count,
        ),
        "update_diagnostics": state.update_diagnostics or {},
    }


def _failed_method(method_id: str, exc: Exception) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message_sha256": text_sha256(str(exc)),
        "error_message_redacted": True,
    }


def _solver_certificate(
    probes: Sequence[baseline.Probe],
    exact_scores: Mapping[str, Mapping[str, Any]],
    refit_scores: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for probe in probes:
        rows.append(
            {
                "probe_id": probe.probe_id,
                "probe_kind": probe.kind,
                "full_vocabulary_output_kl_nats": full_vocabulary_kl(
                    exact_scores[probe.probe_id]["first_log_probs"],
                    refit_scores[probe.probe_id]["first_log_probs"],
                ),
            }
        )
    values = [row["full_vocabulary_output_kl_nats"] for row in rows]
    return {
        "status": "completed",
        "claim": "exact decrement versus fixed-C retained-key refit",
        "direction": "KL(exact_decrement || fixed_c_refit)",
        "full_vocabulary": True,
        "distribution_scope": "first target token at each probe",
        "distinct_from_behavioral_kl_to_raw_repack": True,
        "probe_output_kls": rows,
        "mean_output_kl_nats": _mean(values),
        "max_output_kl_nats": max(values),
        "solver_diagnostics": dict(diagnostics),
    }


def _normalized_contains(text: str, target: str) -> bool:
    normalize = lambda value: " ".join(str(value).casefold().split())
    return normalize(target) in normalize(text)


def _prompt_suppression_prefix(record: LongitudinalRecord) -> str:
    target_names = {
        probe.target_name
        for probe in record.probes
        if probe.kind == "deleted"
    }
    if len(target_names) != 1:
        raise ValueError("prompt suppression requires one deleted target name")
    target_name = next(iter(target_names))
    targets = {
        probe.target
        for probe in record.probes
        if probe.kind == "deleted"
    }
    prefix = (
        "\n\nMemory-control instruction: Treat the stored detail about "
        f"{target_name} as unavailable. Do not use or reveal a value for it. "
        "Preserve unrelated user memories.\n"
    )
    if any(target.casefold() in prefix.casefold() for target in targets):
        raise ValueError("prompt-suppression instruction leaks target value")
    return prefix


def _greedy_audit(
    runtime: Any,
    state: baseline.QueryState,
    probes: Sequence[baseline.Probe],
    *,
    tokens: int,
    save_text: bool,
) -> dict[str, Any]:
    rows = []
    for probe in probes:
        generated, _ = runtime.generate_persistent(
            state.memory,
            state.prompt_prefix + probe.prompt,
            n_tokens=int(tokens),
        )
        row = {
            "probe_id": probe.probe_id,
            "probe_kind": probe.kind,
            "contains_semantic_gold_literal": _normalized_contains(
                generated,
                probe.target,
            ),
            "generated_text_sha256": text_sha256(generated),
        }
        if save_text:
            row["generated_text"] = generated
        rows.append(row)
    return {
        "greedy": True,
        "max_new_tokens": int(tokens),
        "contains_raw_text": bool(save_text),
        "probes": rows,
    }


def _record_seed(base_seed: int, record_id: str) -> int:
    digest = hashlib.sha256(
        f"memops-eval-v1\0{int(base_seed)}\0{record_id}".encode("utf-8")
    ).digest()
    return int(base_seed) + int.from_bytes(digest[:4], "big")


def _fingerprint_value(
    digest: Any,
    value: Any,
    *,
    seen: set[int],
) -> None:
    """Hash nested metadata and complete Torch/NumPy tensor contents."""

    if value is None or isinstance(value, (bool, int, float, str)):
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
        chunk_elements = 1_048_576
        for start in range(0, int(flat.numel()), chunk_elements):
            chunk = (
                flat[start : start + chunk_elements]
                .to("cpu")
                .contiguous()
            )
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
            for key in sorted(value, key=lambda item: repr(item)):
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
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            digest.update(type(value).__name__.encode("utf-8") + b"(")
            for name in sorted(attributes):
                if name.startswith("__"):
                    continue
                _fingerprint_value(digest, name, seen=seen)
                _fingerprint_value(
                    digest,
                    attributes[name],
                    seen=seen,
                )
            digest.update(b")")
            return
        _fingerprint_value(
            digest,
            repr((type(value).__name__, value)),
            seen=seen,
        )
    finally:
        seen.remove(identity)


def _persistent_state_fingerprint(memory: Any) -> str:
    """Fingerprint persistent metadata plus all session/native-cache values."""

    digest = hashlib.sha256()
    _fingerprint_value(
        digest,
        {
            "token_count": int(memory.token_count),
            "input_digest": str(memory.input_digest),
            "deleted_positions": tuple(memory.deleted_positions),
            "deletion_kind": memory.deletion_kind,
            "token_ids": tuple(getattr(memory, "token_ids", ()) or ()),
            "request": getattr(memory, "request", None),
            "fallback_reason": getattr(memory, "fallback_reason", None),
            "layer_sessions": memory.layer_sessions,
            "past_key_values": memory.past_key_values,
        },
        seen=set(),
    )
    return digest.hexdigest()


def _score_suite(
    runtime: Any,
    state: baseline.QueryState,
    probes: Sequence[baseline.Probe],
) -> dict[str, dict[str, Any]]:
    return baseline._score_probe_suite(runtime, state, probes)


def _context_payload(record: LongitudinalRecord) -> dict[str, Any]:
    context = record.context
    return {
        "original_token_count": len(context.original_token_ids),
        "owned_row_token_count": len(context.forget_positions),
        "row_deleted_token_count": len(context.edited_token_ids),
        "raw_omitted_retokenized_token_count": len(
            record.raw_omitted_token_ids
        ),
        "tokens_before_owned": min(context.forget_positions),
        "tokens_after_owned": (
            len(context.original_token_ids) - max(context.forget_positions) - 1
        ),
        "deletion_ranges": [
            {"start": int(start), "end": int(end)}
            for start, end in context.deletion_ranges
        ],
        "original_token_ids_sha256": token_ids_sha256(
            context.original_token_ids
        ),
        "row_deleted_token_ids_sha256": token_ids_sha256(
            context.edited_token_ids
        ),
        "raw_omitted_token_ids_sha256": token_ids_sha256(
            record.raw_omitted_token_ids
        ),
        "raw_omitted_text_sha256": text_sha256(context.edited_text),
        "raw_repack_freshly_retokenized": True,
        "row_delete_distinct_from_raw_repack": (
            tuple(context.edited_token_ids)
            != tuple(record.raw_omitted_token_ids)
        ),
    }


def _record_base(
    runtime: Any,
    record: LongitudinalRecord,
    probes: Sequence[baseline.Probe],
    record_seed: int,
) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "source_file": record.source_file,
        "record_seed": int(record_seed),
        "status": "running",
        "source_diagnostics": {
            "target_value_literal_in_owned": (
                record.target_value_literal_in_owned
            ),
            "target_value_occurrences_inside_owned": (
                record.target_value_occurrences_inside_owned
            ),
            "target_value_occurrences_outside_owned": (
                record.target_value_occurrences_outside_owned
            ),
            "retained_value_literal_in_retained_exchange": (
                record.retained_value_literal_in_retained_exchange
            ),
            "retained_value_occurrences_inside_owned": (
                record.retained_value_occurrences_inside_owned
            ),
            "retained_value_occurrences_outside_owned": (
                record.retained_value_occurrences_outside_owned
            ),
            "source_session_start_index": record.source_session_start_index,
            "source_session_end_index": record.source_session_end_index,
        },
        "context": _context_payload(record),
        "probes": [
            {
                "probe_id": probe.probe_id,
                "probe_kind": probe.kind,
                "prompt_sha256": text_sha256(probe.prompt),
                "target_sha256": text_sha256(probe.target),
                "target_token_count": len(probe.target_ids),
            }
            for probe in probes
        ],
        "methods": {},
    }


def _prefill_original(
    runtime: Any,
    record: LongitudinalRecord,
) -> tuple[Any, float]:
    import time

    baseline._synchronize(runtime.config.device)
    started = time.perf_counter()
    memory = runtime.prefill_persistent(list(record.context.original_token_ids))
    baseline._synchronize(runtime.config.device)
    elapsed = time.perf_counter() - started
    expected = token_ids_sha256(record.context.original_token_ids)
    if str(memory.input_digest) != expected:
        raise RuntimeError("original persistent prefill digest differs")
    return memory, elapsed


def evaluate_admission_record(
    runtime: Any,
    record: LongitudinalRecord,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    greedy_tokens: int = 0,
    save_generations: bool = False,
) -> dict[str, Any]:
    """Run the frozen present/raw-repack admission pair only."""

    if warmup < 0 or repeats < 1 or greedy_tokens < 0:
        raise ValueError("invalid benchmark repetition or generation count")
    record_seed = _record_seed(seed, record.record_id)
    baseline._seed_everything(record_seed)
    probes = build_probes(runtime, record)
    query = lambda state: _score_suite(runtime, state, probes)
    row = _record_base(runtime, record, probes, record_seed)
    original_memory, prefill_seconds = _prefill_original(runtime, record)
    source_shape = persistent_state_shape_signature(original_memory)
    source_fingerprint_before = _persistent_state_fingerprint(original_memory)
    present_state = baseline.QueryState(
        original_memory,
        update_diagnostics={"reference_kind": "present_control"},
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
                "reference_kind": "fresh_raw_exchange_omitted_repack",
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
    row["shared_original_prefill"] = {
        "seconds": prefill_seconds,
        "synchronized_device": runtime.config.device,
        "input_digest": str(original_memory.input_digest),
        "storage": method_storage_report(
            original_memory,
            original_memory,
        )["state"],
    }
    row["admission_only"] = True
    row["admission"] = admission_decomposition(
        probes,
        present_scores,
        raw_scores,
    )
    row["references"] = {
        "present_control": {
            "scores": _score_snapshot(probes, present_scores),
            "timing": _timing_scope(present_timing),
            **_behavioral_metrics(
                probes,
                present_scores,
                present_scores,
                raw_scores,
            ),
        },
        "full_raw_repack": {
            "scores": _score_snapshot(probes, raw_scores),
            "timing": _timing_scope(raw_timing),
            "input_digest": str(raw_state.memory.input_digest),
            "freshly_retokenized": True,
            **_behavioral_metrics(
                probes,
                raw_scores,
                present_scores,
                raw_scores,
            ),
        },
    }
    if greedy_tokens:
        row["greedy_audit"] = {
            "present_control": _greedy_audit(
                runtime,
                present_state,
                probes,
                tokens=greedy_tokens,
                save_text=save_generations,
            ),
            "full_raw_repack": _greedy_audit(
                runtime,
                raw_state,
                probes,
                tokens=greedy_tokens,
                save_text=save_generations,
            ),
        }
    source_fingerprint_after = _persistent_state_fingerprint(original_memory)
    source_unchanged = (
        persistent_state_shape_signature(original_memory) == source_shape
        and source_fingerprint_after == source_fingerprint_before
    )
    row["source_state_immutability"] = {
        "verification_scope": "metadata_shapes_and_complete_tensor_values",
        "before_sha256": source_fingerprint_before,
        "after_sha256": source_fingerprint_after,
        "unchanged": source_unchanged,
    }
    if not source_unchanged:
        raise RuntimeError("admission queries mutated the shared source state")
    row["status"] = "completed"
    return row


def evaluate_record(
    runtime: Any,
    record: LongitudinalRecord,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    greedy_tokens: int = 0,
    save_generations: bool = False,
    prompt_suppression: bool = False,
) -> dict[str, Any]:
    """Run the matched state-edit methods for one frozen conversation."""

    if warmup < 0 or repeats < 1 or greedy_tokens < 0:
        raise ValueError("invalid benchmark repetition or generation count")
    record_seed = _record_seed(seed, record.record_id)
    baseline._seed_everything(record_seed)
    probes = build_probes(runtime, record)
    query = lambda state: _score_suite(runtime, state, probes)
    row = _record_base(runtime, record, probes, record_seed)
    original_memory, prefill_seconds = _prefill_original(runtime, record)
    source_shape = persistent_state_shape_signature(original_memory)
    source_fingerprint_before = _persistent_state_fingerprint(original_memory)
    row["shared_original_prefill"] = {
        "seconds": prefill_seconds,
        "synchronized_device": runtime.config.device,
        "input_digest": str(original_memory.input_digest),
        "storage": method_storage_report(
            original_memory,
            original_memory,
        )["state"],
    }

    greedy_audits: dict[str, Any] = {}
    present_state = baseline.QueryState(
        original_memory,
        update_diagnostics={"reference_kind": "present_control"},
    )
    present_scores, present_timing = baseline._benchmark_existing_state(
        state=present_state,
        query=query,
        device=runtime.config.device,
        warmup=warmup,
        repeats=repeats,
    )
    if greedy_tokens:
        greedy_audits["present_control"] = _greedy_audit(
            runtime,
            present_state,
            probes,
            tokens=greedy_tokens,
            save_text=save_generations,
        )
    raw_state, raw_scores, raw_timing = baseline._benchmark_method(
        build=lambda: baseline.QueryState(
            runtime.prefill_persistent(list(record.raw_omitted_token_ids)),
            update_diagnostics={
                "reference_kind": "fresh_raw_exchange_omitted_repack",
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
    row["admission"] = admission_decomposition(
        probes,
        present_scores,
        raw_scores,
    )
    row["references"] = {
        "present_control": {
            "status": "completed",
            "scores": _score_snapshot(probes, present_scores),
            "timing": _timing_scope(present_timing),
            **_behavioral_metrics(
                probes,
                present_scores,
                present_scores,
                raw_scores,
            ),
        }
    }
    row["methods"]["full_raw_repack"] = _method_report(
        "full_raw_repack",
        raw_state,
        raw_scores,
        raw_timing,
        original_memory=original_memory,
        probes=probes,
        present_scores=present_scores,
        raw_repack_scores=raw_scores,
        semantics={
            "behavioral_reference": True,
            "fresh_prefill": True,
            "raw_exchange_omitted": True,
            "freshly_retokenized": True,
            "suffix_recomputed": True,
        },
    )
    if greedy_tokens:
        greedy_audits["full_raw_repack"] = _greedy_audit(
            runtime,
            raw_state,
            probes,
            tokens=greedy_tokens,
            save_text=save_generations,
        )
    del raw_state

    if prompt_suppression:
        try:
            suppression_prefix = _prompt_suppression_prefix(record)
            encoded_prefix = runtime.tokenizer(
                suppression_prefix,
                add_special_tokens=False,
            )
            prefix_ids = (
                encoded_prefix["input_ids"]
                if isinstance(encoded_prefix, Mapping)
                else encoded_prefix.input_ids
            )

            def build_suppression_state():
                return baseline.QueryState(
                    original_memory,
                    prompt_prefix=suppression_prefix,
                    prompt_token_count=len(prefix_ids),
                    update_diagnostics={
                        "persistent_state_deleted": False,
                        "base_state_unchanged": True,
                        "adds_query_context": True,
                        "target_value_in_instruction": False,
                        "instruction_sha256": text_sha256(
                            suppression_prefix
                        ),
                        "instruction_token_count": len(prefix_ids),
                    },
                )

            (
                suppression_state,
                suppression_scores,
                suppression_timing,
            ) = baseline._benchmark_method(
                build=build_suppression_state,
                query=query,
                device=runtime.config.device,
                warmup=warmup,
                repeats=repeats,
                operation_seed=record_seed + 7,
            )
            row["methods"][PROMPT_SUPPRESSION_ID] = _method_report(
                PROMPT_SUPPRESSION_ID,
                suppression_state,
                suppression_scores,
                suppression_timing,
                original_memory=original_memory,
                probes=probes,
                present_scores=present_scores,
                raw_repack_scores=raw_scores,
                semantics={
                    "prompt_only_behavioral_control": True,
                    "persistent_state_deleted": False,
                    "base_state_unchanged": True,
                    "adds_query_context": True,
                    "target_value_in_instruction": False,
                },
            )
            if greedy_tokens:
                greedy_audits[PROMPT_SUPPRESSION_ID] = _greedy_audit(
                    runtime,
                    suppression_state,
                    probes,
                    tokens=greedy_tokens,
                    save_text=save_generations,
                )
            del suppression_state, suppression_scores
        except Exception as exc:
            row["methods"][PROMPT_SUPPRESSION_ID] = _failed_method(
                PROMPT_SUPPRESSION_ID,
                exc,
            )

    try:
        row_state, row_scores, row_timing = baseline._benchmark_method(
            build=lambda: baseline.QueryState(
                runtime.prefill_persistent(
                    list(record.context.edited_token_ids)
                ),
                update_diagnostics={
                    "reference_kind": "original_token_row_delete_repack",
                    "freshly_retokenized": False,
                    "suffix_recomputed": True,
                },
            ),
            query=query,
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            operation_seed=record_seed + 2,
        )
        row["methods"]["token_row_repack"] = _method_report(
            "token_row_repack",
            row_state,
            row_scores,
            row_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_repack_scores=raw_scores,
            semantics={
                "diagnostic_only": True,
                "fresh_prefill": True,
                "original_token_rows_removed": True,
                "freshly_retokenized": False,
                "suffix_recomputed": True,
            },
        )
        del row_state, row_scores
    except Exception as exc:
        row["methods"]["token_row_repack"] = _failed_method(
            "token_row_repack",
            exc,
        )

    try:

        def build_proxy_state():
            memory = runtime.delete_persistent(
                original_memory,
                record.context.forget_positions,
                kind="fp32_masked_refit_proxy",
            )
            fallback = str(memory.deletion_kind).endswith(
                "_full_repack_fallback"
            )
            return baseline.QueryState(
                memory,
                update_diagnostics={
                    "solver": (
                        "full original-token-row repack fallback"
                        if fallback
                        else "feasible projected FP32 FISTA masked refit"
                    ),
                    "query_independent": True,
                    "full_repack_fallback": fallback,
                    "fallback_reason_sha256": (
                        text_sha256(str(getattr(memory, "fallback_reason", "")))
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
        row["methods"]["fp32_proxy"] = _method_report(
            "fp32_proxy",
            proxy_state,
            proxy_scores,
            proxy_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_repack_scores=raw_scores,
            semantics={
                "solver": proxy_state.update_diagnostics["solver"],
                "query_independent": True,
                "suffix_recomputed": bool(
                    proxy_state.update_diagnostics[
                        "full_repack_fallback"
                    ]
                ),
                "conditional_on_contextualized_retained_keys": not bool(
                    proxy_state.update_diagnostics[
                        "full_repack_fallback"
                    ]
                ),
                "full_repack_fallback": bool(
                    proxy_state.update_diagnostics[
                        "full_repack_fallback"
                    ]
                ),
            },
        )
        del proxy_state, proxy_scores
    except Exception as exc:
        row["methods"]["fp32_proxy"] = _failed_method("fp32_proxy", exc)

    try:

        def build_cache_state():
            memory, diagnostics = cache_delete_and_shift(
                original_memory,
                record.context.forget_positions,
                edited_token_ids=record.context.edited_token_ids,
            )
            diagnostics["raw_repack_token_ids_sha256"] = token_ids_sha256(
                record.raw_omitted_token_ids
            )
            diagnostics["row_delete_differs_from_raw_repack"] = (
                tuple(record.context.edited_token_ids)
                != tuple(record.raw_omitted_token_ids)
            )
            return baseline.QueryState(
                memory,
                update_diagnostics=diagnostics,
            )

        cache_state, cache_scores, cache_timing = baseline._benchmark_method(
            build=build_cache_state,
            query=query,
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            operation_seed=record_seed + 4,
        )
        row["methods"]["cache_delete_shift"] = _method_report(
            "cache_delete_shift",
            cache_state,
            cache_scores,
            cache_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_repack_scores=raw_scores,
            semantics={
                "diagnostic_only": True,
                "physically_deletes_matching_cache_rows": True,
                "solver_refit": False,
                "suffix_recomputed": False,
                "freshly_retokenized": False,
                "rope_keys_rerotated": False,
            },
        )
        del cache_state, cache_scores
    except Exception as exc:
        row["methods"]["cache_delete_shift"] = _failed_method(
            "cache_delete_shift",
            exc,
        )

    try:

        def build_decay_state():
            memory = original_memory.fork()
            memory.request = GateRequest(
                scale_pos=record.context.forget_positions,
                scale_factor=DECAY_FACTOR,
                gate_floor=float(
                    getattr(
                        getattr(original_memory, "request", None),
                        "gate_floor",
                        0.0,
                    )
                ),
            )
            memory.deleted_positions = record.context.forget_positions
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
            operation_seed=record_seed + 5,
        )
        row["methods"]["decay_0_01"] = _method_report(
            "decay_0_01",
            decay_state,
            decay_scores,
            decay_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_repack_scores=raw_scores,
            semantics={
                "decay_factor": DECAY_FACTOR,
                "positions_remain_resident": True,
                "solver_refit": False,
                "suffix_recomputed": False,
            },
        )
        del decay_state, decay_scores
    except Exception as exc:
        row["methods"]["decay_0_01"] = _failed_method("decay_0_01", exc)

    certificate_states: dict[str, baseline.QueryState] = {}
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
        row["methods"]["exact_decrement"] = _method_report(
            "exact_decrement",
            certificate_states["exact"],
            certificate_scores["exact"],
            certificate_timings["exact"],
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_repack_scores=raw_scores,
            semantics={
                "gate_solver": "float64 Cauwenberghs-Poggio decrement",
                "fixed_C": True,
                "query_independent": True,
                "suffix_recomputed": False,
                "certificate_reference": "fixed_c_refit",
                "not_certified_to_raw_repack": True,
            },
        )
        row["methods"]["fixed_c_refit"] = _method_report(
            "fixed_c_refit",
            certificate_states["refit"],
            certificate_scores["refit"],
            certificate_timings["refit"],
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            raw_repack_scores=raw_scores,
            semantics={
                "gate_solver": "float64 retained-key from-scratch refit",
                "fixed_C": True,
                "query_independent": True,
                "suffix_recomputed": False,
                "conditional_on_contextualized_retained_keys": True,
            },
        )
        row["solver_certificate"] = _solver_certificate(
            probes,
            certificate_scores["exact"],
            certificate_scores["refit"],
            solver_diagnostics,
        )
        if greedy_tokens:
            greedy_audits["exact_decrement"] = _greedy_audit(
                runtime,
                certificate_states["exact"],
                probes,
                tokens=greedy_tokens,
                save_text=save_generations,
            )
            greedy_audits["fixed_c_refit"] = _greedy_audit(
                runtime,
                certificate_states["refit"],
                probes,
                tokens=greedy_tokens,
                save_text=save_generations,
            )
        del certificate_states, certificate_scores
    except Exception as exc:
        row["methods"]["exact_decrement"] = _failed_method(
            "exact_decrement",
            exc,
        )
        row["methods"]["fixed_c_refit"] = _failed_method(
            "fixed_c_refit",
            exc,
        )
        row["solver_certificate"] = {
            "status": "failed",
            "distinct_from_behavioral_kl_to_raw_repack": True,
            "error_type": type(exc).__name__,
            "error_message_sha256": text_sha256(str(exc)),
            "error_message_redacted": True,
        }

    if greedy_audits:
        row["greedy_audit"] = greedy_audits

    source_fingerprint_after = _persistent_state_fingerprint(original_memory)
    source_unchanged = (
        persistent_state_shape_signature(original_memory) == source_shape
        and source_fingerprint_after == source_fingerprint_before
    )
    row["source_state_immutability"] = {
        "verification_scope": "metadata_shapes_and_complete_tensor_values",
        "before_sha256": source_fingerprint_before,
        "after_sha256": source_fingerprint_after,
        "unchanged": source_unchanged,
    }
    if not source_unchanged:
        raise RuntimeError("a deletion method mutated the shared source state")

    row["method_failures"] = [
        method_id
        for method_id in expected_method_ids(
            prompt_suppression=prompt_suppression
        )
        if row["methods"].get(method_id, {}).get("status") == "failed"
    ]
    row["status"] = (
        "completed"
        if not row["method_failures"]
        else "completed_with_method_failures"
    )
    return row


def _method_rows(
    records: Sequence[Mapping[str, Any]],
    method_id: str,
    *,
    admitted_only: bool,
) -> list[Mapping[str, Any]]:
    result = []
    for record in records:
        if record.get("status") == "failed":
            continue
        if admitted_only and not bool(
            record["admission"]["target_recall"]["admitted"]
        ):
            continue
        method = record.get("methods", {}).get(method_id, {})
        if method.get("status") == "completed":
            result.append(method)
    return result


def _method_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "completed_records": 0,
            "deleted_probe_n": 0,
            "retained_probe_n": 0,
        }
    deleted_probes = [
        probe
        for row in rows
        for probe in row["deleted_target_quality"]["probes"]
    ]
    retained_rows = [row["retained_quality"] for row in rows]
    result = {
        "completed_records": len(rows),
        "deleted_probe_n": len(deleted_probes),
        "retained_probe_n": len(retained_rows),
        "mean_deleted_sequence_log_probability": _mean(
            [probe["score"]["mean_log_probability"] for probe in deleted_probes]
        ),
        "mean_deleted_sequence_probability": _mean(
            [
                probe["score"]["geometric_mean_probability"]
                for probe in deleted_probes
            ]
        ),
        "mean_deleted_first_target_token_rank": _mean(
            [
                probe["score"]["first_target_token_rank"]
                for probe in deleted_probes
            ]
        ),
        "mean_retained_sequence_log_probability": _mean(
            [
                retained["score"]["mean_log_probability"]
                for retained in retained_rows
            ]
        ),
        "mean_retained_sequence_probability": _mean(
            [
                retained["score"]["geometric_mean_probability"]
                for retained in retained_rows
            ]
        ),
        "mean_retained_first_target_token_rank": _mean(
            [
                retained["score"]["first_target_token_rank"]
                for retained in retained_rows
            ]
        ),
        "mean_target_suppression_vs_present_nats": _mean(
            [
                row["deleted_target_quality"][
                    "mean_suppression_vs_present_nats"
                ]
                for row in rows
            ]
        ),
        "mean_target_kl_to_raw_repack_nats": _mean(
            [
                row["deleted_target_quality"][
                    "mean_full_vocabulary_kl_to_raw_repack_nats"
                ]
                for row in rows
            ]
        ),
        "mean_retained_log_probability_drift_from_raw_repack_nats": _mean(
            [
                row["retained_quality"][
                    "mean_log_probability_drift_from_raw_repack_nats"
                ]
                for row in rows
            ]
        ),
        "mean_retained_kl_to_raw_repack_nats": _mean(
            [
                row["retained_quality"][
                    "full_vocabulary_kl_raw_repack_to_method_nats"
                ]
                for row in rows
            ]
        ),
    }
    if all("timing" in row for row in rows):
        result["timing"] = {
            "mean_update_median_seconds": _mean(
                [row["timing"]["update_seconds"]["median"] for row in rows]
            ),
            "mean_query_median_seconds": _mean(
                [row["timing"]["query_seconds"]["median"] for row in rows]
            ),
            "mean_end_to_end_median_seconds": _mean(
                [row["timing"]["end_to_end_seconds"]["median"] for row in rows]
            ),
        }
    if all("storage" in row for row in rows):
        result["storage"] = {
            "mean_state_tensor_storage_bytes": _mean(
                [
                    row["storage"]["state"][
                        "deduplicated_tensor_storage_bytes"
                    ]
                    for row in rows
                ]
            ),
            "mean_incremental_tensor_storage_bytes": _mean(
                [
                    row["storage"]["incremental_tensor_storage_bytes"]
                    for row in rows
                ]
            ),
        }
    return result


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    admission_only: bool,
) -> dict[str, Any]:
    """Aggregate all frozen records and a separately labelled admitted subset."""

    evaluated = [
        record for record in records if record.get("status") != "failed"
    ]
    fully_completed = [
        record for record in records if record.get("status") == "completed"
    ]
    method_failure_records = [
        record
        for record in records
        if record.get("status") == "completed_with_method_failures"
    ]
    target_admitted = sum(
        bool(record["admission"]["target_recall"]["admitted"])
        for record in evaluated
    )
    retained_available = sum(
        bool(record["admission"]["retained_availability"]["available"])
        for record in evaluated
    )
    joint_admitted = sum(
        bool(record["admission"]["joint_target_and_retained"]["admitted"])
        for record in evaluated
    )
    deleted_probe_n = sum(
        sum(probe.get("probe_kind") == "deleted" for probe in record["probes"])
        for record in evaluated
    )
    retained_probe_n = sum(
        sum(probe.get("probe_kind") == "retained" for probe in record["probes"])
        for record in evaluated
    )
    threshold_payloads = [
        dict(record["admission"]["thresholds"]) for record in evaluated
    ]
    if threshold_payloads and any(
        payload != threshold_payloads[0]
        for payload in threshold_payloads[1:]
    ):
        raise ValueError("admission thresholds differ across records")
    summary: dict[str, Any] = {
        "aggregation_population": (
            "all selected frozen manifest records; no model-output replacement"
        ),
        "attempted_records": len(records),
        "completed_records": len(evaluated),
        "target_recall_admitted": target_admitted,
        "retained_available": retained_available,
        "joint_target_and_retained": joint_admitted,
        "denominators": {
            "frozen_records": len(records),
            "attempted_records": len(records),
            "evaluated_records": len(evaluated),
            "fully_completed_records": len(fully_completed),
            "record_failures": len(records) - len(evaluated),
            "method_failure_records": len(method_failure_records),
            "target_admitted": target_admitted,
            "target_rejected": len(evaluated) - target_admitted,
            "retained_available": retained_available,
            "retained_unavailable": len(evaluated) - retained_available,
            "joint_admitted": joint_admitted,
            "joint_rejected": len(evaluated) - joint_admitted,
            "deleted_probes_attempted": deleted_probe_n,
            "retained_probes_attempted": retained_probe_n,
        },
        "admission_thresholds": (
            threshold_payloads[0] if threshold_payloads else {}
        ),
        "source_slices": {
            "semantic_gold_literal_in_owned": sum(
                bool(
                    record["source_diagnostics"][
                        "target_value_literal_in_owned"
                    ]
                )
                for record in evaluated
            ),
            "zero_literal_occurrences_outside_owned": sum(
                int(
                    record["source_diagnostics"][
                        "target_value_occurrences_outside_owned"
                    ]
                )
                == 0
                for record in evaluated
            ),
            "retained_semantic_gold_literal_in_retained_exchange": sum(
                bool(
                    record["source_diagnostics"][
                        "retained_value_literal_in_retained_exchange"
                    ]
                )
                for record in evaluated
            ),
            "retained_value_inside_owned_exchange": sum(
                int(
                    record["source_diagnostics"][
                        "retained_value_occurrences_inside_owned"
                    ]
                )
                > 0
                for record in evaluated
            ),
            "row_delete_differs_from_raw_repack": sum(
                bool(
                    record["context"][
                        "row_delete_distinct_from_raw_repack"
                    ]
                )
                for record in evaluated
            ),
        },
        "conditions": {},
        "methods": {},
    }
    present_rows = [
        record["references"]["present_control"] for record in evaluated
    ]
    present_admitted_rows = [
        record["references"]["present_control"]
        for record in evaluated
        if record["admission"]["target_recall"]["admitted"]
    ]
    summary["conditions"]["present_control"] = {
        "all_frozen_records": {
            **_method_summary(present_rows),
            "failed_records": len(records) - len(present_rows),
        },
        "target_recall_admitted_subset": {
            "selection": (
                "predeclared present-versus-raw-repack recall gate; no "
                "replacement"
            ),
            **_method_summary(present_admitted_rows),
        },
    }
    raw_reference_rows = [
        record["references"]["full_raw_repack"]
        for record in evaluated
        if "full_raw_repack" in record.get("references", {})
    ]
    if raw_reference_rows:
        raw_admitted_rows = [
            record["references"]["full_raw_repack"]
            for record in evaluated
            if "full_raw_repack" in record.get("references", {})
            and record["admission"]["target_recall"]["admitted"]
        ]
        summary["conditions"]["full_raw_repack"] = {
            "all_frozen_records": {
                **_method_summary(raw_reference_rows),
                "failed_records": len(records) - len(raw_reference_rows),
            },
            "target_recall_admitted_subset": {
                "selection": (
                    "predeclared present-versus-raw-repack recall gate; no "
                    "replacement"
                ),
                **_method_summary(raw_admitted_rows),
            },
        }
    if admission_only:
        return summary
    observed_method_ids = {
        method_id
        for record in evaluated
        for method_id in (record.get("methods") or {})
    }
    ordered_method_ids = [
        method_id
        for method_id in (*METHOD_IDS, PROMPT_SUPPRESSION_ID)
        if method_id in observed_method_ids
    ]
    for method_id in ordered_method_ids:
        all_rows = _method_rows(
            evaluated,
            method_id,
            admitted_only=False,
        )
        admitted_rows = _method_rows(
            evaluated,
            method_id,
            admitted_only=True,
        )
        condition = {
            "all_frozen_records": _method_summary(all_rows),
            "target_recall_admitted_subset": {
                "selection": (
                    "predeclared present-versus-raw-repack recall gate; no "
                    "replacement"
                ),
                **_method_summary(admitted_rows),
            },
            "failed_records": len(evaluated) - len(all_rows),
        }
        summary["methods"][method_id] = condition
        summary["conditions"][method_id] = condition
    return summary


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_fingerprints() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    workspace = package.parent
    files = {
        "eval_memops_longitudinal.py": Path(__file__).resolve(),
        "memops_longitudinal.py": package / "memops_longitudinal.py",
        "persistent_deletion.py": package / "persistent_deletion.py",
        "eval_persistent_deletion_baselines.py": (
            package / "eval_persistent_deletion_baselines.py"
        ),
        "graft.py": package / "graft.py",
        "recovery_state.py": package / "recovery_state.py",
        "sv_global_attention.py": package / "sv_global_attention.py",
        "layer_select.py": package / "layer_select.py",
        "demo_server/gemma_engine.py": package / "demo_server" / "gemma_engine.py",
        "demo_server/gate_context.py": (
            package / "demo_server" / "gate_context.py"
        ),
        "demo_server/certificate.py": (
            package / "demo_server" / "certificate.py"
        ),
        "svattn/causal_sv_attention.py": (
            workspace / "svattn" / "causal_sv_attention.py"
        ),
        "svattn/mlx_svdd.py": workspace / "svattn" / "mlx_svdd.py",
        "cp_svm/oneclass_incremental.py": (
            workspace / "cp_svm" / "oneclass_incremental.py"
        ),
        "cp_svm/oneclass_fast.py": workspace / "cp_svm" / "oneclass_fast.py",
        "cp_svm/binary_incremental.py": (
            workspace / "cp_svm" / "binary_incremental.py"
        ),
        "cp_svm/kernels.py": workspace / "cp_svm" / "kernels.py",
    }
    if set(files) != set(IMPLEMENTATION_FILE_LABELS):
        raise RuntimeError("implementation fingerprint allowlist differs")
    fingerprints = {
        relative: _sha256_file(path)
        for relative, path in sorted(files.items())
    }
    fingerprints["contract_sha256"] = hashlib.sha256(
        json.dumps(
            fingerprints,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return fingerprints


def _environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in (
        "numpy",
        "scipy",
        "cvxpy",
        "torch",
        "transformers",
        "peft",
        "mlx",
    ):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {"platform": platform.platform(), "versions": versions}


def _normalize_adapter(value: str | None) -> str | None:
    if value is None or str(value).casefold() in {"", "none", "null", "-"}:
        return None
    return str(value)


def _adapter_content_sha256(value: str | None) -> str | None:
    adapter = _normalize_adapter(value)
    if adapter is None:
        return None
    path = Path(adapter)
    weights = (
        path / "adapter_model.safetensors"
        if path.is_dir()
        else path
    )
    if not weights.is_file():
        raise FileNotFoundError(f"adapter weights are missing: {weights}")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    adapter = _normalize_adapter(args.adapter)
    return {
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter": "none" if adapter is None else "provided",
        "adapter_sha256": _adapter_content_sha256(adapter),
        "device": args.device,
        "dtype": "float32",
        "window": args.window,
        "nu": args.nu,
        "preserve_prefix_mass": True,
        "per_boundary_box": True,
        "solver_seed": args.solver_seed,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "admission_only": args.admission_only,
        "greedy_tokens": args.greedy_tokens,
        "save_generations": args.save_generations,
        "prompt_suppression": args.prompt_suppression,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter", default="none")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--nu", type=float, default=DEFAULT_NU)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--records", type=int)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument(
        "--prompt-suppression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "include a fixed target-value-free query-only suppression control; "
            "this never edits persistent state"
        ),
    )
    parser.add_argument("--greedy-tokens", type=int, default=0)
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.record_start < 0
        or (args.records is not None and args.records < 1)
        or args.warmup < 0
        or args.repeats < 1
        or args.greedy_tokens < 0
        or args.window < 1
        or not 0 < args.nu <= 1
    ):
        parser.error("invalid record, timing, generation, window, or nu value")
    if args.save_generations and not args.greedy_tokens:
        parser.error("--save-generations requires --greedy-tokens")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    tokenizer_spec = manifest["tokenizer"]
    geometry = manifest["geometry"]
    if (
        args.model != tokenizer_spec["model_id"]
        or args.model_revision != tokenizer_spec["revision"]
    ):
        parser.error(
            "--model and --model-revision must match the frozen tokenizer "
            "because token ownership is model-token exact"
        )
    if (
        float(args.nu) != float(geometry["nu"])
        or int(geometry["chunk"]) != DEFAULT_CHUNK
    ):
        parser.error("--nu/chunk differ from the frozen manifest geometry")

    adapter = _normalize_adapter(args.adapter)
    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=adapter,
            device=args.device,
            dtype="float32",
            generation_tokens=max(1, args.greedy_tokens),
            window=args.window,
            model_revision=args.model_revision,
            graft_enabled=True,
            nu=args.nu,
            preserve_prefix_mass=True,
            per_boundary_box=True,
            solver_seed=args.solver_seed,
        )
    )
    runtime.ensure_loaded()
    if (
        float(runtime.resolved_nu) != float(geometry["nu"])
        or not bool(runtime.resolved_preserve_prefix_mass)
        or not bool(runtime.resolved_per_boundary_box)
        or int(runtime.resolved_solver_seed) != int(args.solver_seed)
    ):
        parser.error("resolved runtime differs from the frozen configuration")

    rehydrated = rehydrate_manifest(
        manifest,
        args.source_root,
        runtime.tokenizer,
    )
    selected = rehydrated[args.record_start :]
    if args.records is not None:
        selected = selected[: args.records]
    if not selected:
        parser.error("record selection is empty")

    output = Path(args.out)
    manifest_hash = _sha256_file(manifest_path)
    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation": (
            "MemOps naturalistic longitudinal exchange deletion after >=10k "
            "subsequent Gemma tokens"
        ),
        "status": "running",
        "contains_source_text": bool(args.save_generations),
        "contains_full_vocabulary_vectors": False,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "integrity_sha256": manifest["integrity"]["sha256"],
            "selected_record_ids": [record.record_id for record in selected],
            "total_frozen_records": len(rehydrated),
            "fixed_before_model_evaluation": True,
            "model_outputs_used_for_selection": False,
        },
        "config": _config_payload(args),
        "metric_definitions": {
            "admission": (
                "present semantic-gold sequence lift over freshly retokenized "
                "raw-exchange-omitted repack plus present first-token rank"
            ),
            "behavioral_kl": (
                "full-vocabulary KL(raw omitted repack || method) at each "
                "probe's first semantic-gold token"
            ),
            "solver_certificate": (
                "full-vocabulary KL(exact decrement || fixed-C retained-key "
                "refit), separate from behavioral raw-repack KL"
            ),
            "retention": (
                "paired fact from another exchange in the same evidence "
                "segment, scored before and after target deletion"
            ),
        },
        "technical_scope": [
            (
                "The transcript is prefetched once and sealed; this is a "
                "naturalistic multi-session transcript, not online incremental "
                "session admission."
            ),
            (
                "Exact decrement is certified only to the fixed-C retained-key "
                "refit; the raw omitted repack is freshly retokenized and measured "
                "separately."
            ),
            (
                "Token-row deletion/repack and cache delete/shift are diagnostics; "
                "they "
                "do not define the raw omitted reference."
            ),
        ],
        "implementation": _implementation_fingerprints(),
        "environment": _environment(),
        "records": [],
    }
    if output.exists():
        if args.overwrite:
            pass
        elif args.resume:
            existing = json.loads(output.read_text(encoding="utf-8"))
            for key in (
                "schema",
                "schema_version",
                "manifest",
                "config",
                "implementation",
                "environment",
            ):
                if existing.get(key) != report.get(key):
                    parser.error(f"resume report {key} differs")
            report["records"] = list(existing.get("records") or [])
            existing_ids = [
                str(record.get("record_id") or "")
                for record in report["records"]
            ]
            if (
                not all(existing_ids)
                or len(existing_ids) != len(set(existing_ids))
            ):
                parser.error("resume report has empty or duplicate record IDs")
        else:
            parser.error(f"{output} exists; pass --resume or --overwrite")
    _atomic_write(output, report)
    completed_ids = {
        str(record["record_id"])
        for record in report["records"]
        if record.get("status") == "completed"
    }
    for index, record in enumerate(selected):
        if record.record_id in completed_ids:
            print(
                f"[{index + 1}/{len(selected)}] {record.record_id}: complete",
                flush=True,
            )
            continue
        print(f"[{index + 1}/{len(selected)}] {record.record_id}", flush=True)
        try:
            if args.admission_only:
                result = evaluate_admission_record(
                    runtime,
                    record,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    greedy_tokens=args.greedy_tokens,
                    save_generations=args.save_generations,
                )
            else:
                result = evaluate_record(
                    runtime,
                    record,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    greedy_tokens=args.greedy_tokens,
                    save_generations=args.save_generations,
                    prompt_suppression=args.prompt_suppression,
                )
        except Exception as exc:
            result = {
                "record_id": record.record_id,
                "source_file": record.source_file,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message_sha256": text_sha256(str(exc)),
                "error_message_redacted": True,
            }
        report["records"] = [
            existing
            for existing in report["records"]
            if str(existing.get("record_id")) != record.record_id
        ]
        report["records"].append(result)
        _atomic_write(output, report)
    report["summary"] = summarize_records(
        report["records"],
        admission_only=args.admission_only,
    )
    denominators = report["summary"]["denominators"]
    if denominators["fully_completed_records"] == len(selected):
        report["status"] = "completed"
        exit_code = 0
    elif denominators["evaluated_records"] == len(selected):
        report["status"] = "completed_with_method_failures"
        exit_code = 2
    else:
        report["status"] = "completed_with_record_failures"
        exit_code = 1
    _atomic_write(output, report)
    print(f"wrote evaluation to {output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

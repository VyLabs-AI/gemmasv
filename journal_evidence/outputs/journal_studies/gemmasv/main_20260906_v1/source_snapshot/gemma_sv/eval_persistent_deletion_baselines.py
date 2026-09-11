"""Evaluate matched persistent deletion baselines on fixed synthetic records.

This runner expands the one-record persistent bridge into the eight-record
``whole_record_synthetic_v1`` audit.  Every compatible method starts from the
same tokenized original context and uses the same probes.  A fresh prefill of
the literal edited token sequence is the behavioral reference.

The exact decrement/fixed-C refit comparison is reported separately as a
solver certificate.  Every method also receives a behavioral full-vocabulary
output KL to the repack reference; those two KL families are not conflated.

Example:
    python -m gemma_sv.eval_persistent_deletion_baselines \
      --devices mps --adapters outputs/gemma_sv_distill/lora_adapter \
      --warmup 1 --repeats 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gemma_sv.demo_server.certificate import fixed_c_feasibility
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.persistent_deletion import (
    DECAY_FACTOR,
    ICUL_CORRECT_DEMONSTRATIONS,
    build_faithful_icul_context,
    build_synthetic_record_context,
    cache_delete_and_shift,
    first_token_rank,
    full_vocabulary_kl,
    kveraser_exclusion,
    method_storage_report,
    persistent_state_shape_signature,
    tensor_storage_report,
    token_ids_digest,
)
from gemma_sv.recovery_protocol import MODEL_REVISION, path_sha256


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / "whole_record_synthetic_v1.json"
)
DEFAULT_ADAPTER = "outputs/gemma_sv_distill/lora_adapter"
DEFAULT_OUTPUT = "outputs/gemma_sv_eval/persistent_deletion_baselines.json"
MIN_SIGNAL_NATS = 0.05
MAX_FIRST_TOKEN_RANK = 10
SCHEMA_VERSION = 1
METHOD_IDS = (
    "full_repack",
    "exact_decrement",
    "fixed_c_refit",
    "fp32_proxy",
    "cache_delete_shift",
    "decay_0_01",
    "icul_4",
)


@dataclass(frozen=True)
class Probe:
    probe_id: str
    kind: str
    prompt: str
    target: str
    target_ids: tuple[int, ...]
    field_index: int | None = None
    field_name: str | None = None


@dataclass
class QueryState:
    memory: Any
    prompt_prefix: str = ""
    prompt_token_count: int = 0
    update_diagnostics: dict[str, Any] | None = None


def _comma_values(value: str) -> list[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _run_matrix(adapters: Sequence[str], devices: Sequence[str]):
    parsed_adapters: list[str | None] = [
        None if value.casefold() in {"none", "null", "-"} else value
        for value in adapters
    ]
    if len(parsed_adapters) == len(devices):
        return list(zip(parsed_adapters, devices))
    if len(parsed_adapters) == 1:
        return [(parsed_adapters[0], device) for device in devices]
    if len(devices) == 1:
        return [(adapter, devices[0]) for adapter in parsed_adapters]
    raise ValueError(
        "--adapters and --devices must have equal lengths, or one side must "
        "contain exactly one value for broadcasting"
    )


def _selector_indices(
    selector: str,
    records: Sequence[Mapping[str, Any]],
) -> set[int]:
    text = selector.strip()
    if not text:
        raise ValueError("record selector must not be empty")
    ids = {
        str(record["record_id"]): index
        for index, record in enumerate(records)
    }
    if text in ids:
        return {ids[text]}
    if ":" in text:
        pieces = text.split(":")
        if len(pieces) not in (2, 3):
            raise ValueError(f"invalid record slice {selector!r}")
        values = [
            None if piece == "" else int(piece)
            for piece in pieces
        ]
        selected = range(len(records))[slice(*values)]
        return set(selected)
    index = int(text)
    if index < 0:
        index += len(records)
    if not 0 <= index < len(records):
        raise ValueError(f"record index {selector!r} is out of range")
    return {index}


def _select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    record_start: int,
    record_count: int | None,
    selectors: Sequence[str],
) -> list[tuple[int, Mapping[str, Any]]]:
    if record_start < 0:
        raise ValueError("--record-start must be non-negative")
    if record_count is not None and record_count < 1:
        raise ValueError("--records must be positive")
    indexed = list(enumerate(records))
    if selectors:
        selected: set[int] = set()
        for selector in selectors:
            selected.update(_selector_indices(selector, records))
        indexed = [
            (index, record)
            for index, record in indexed
            if index in selected
        ]
    indexed = indexed[record_start:]
    if record_count is not None:
        indexed = indexed[:record_count]
    return indexed


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.random.seed(int(seed))


def _synchronize(device: str) -> None:
    import torch

    normalized = str(device).casefold()
    if normalized.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif normalized.startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty timing sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
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
        "synchronized_device": device,
        "warmup": int(warmup),
        "repeats": int(repeats),
        "query_scope": "three deleted-field probes plus one retained probe",
        "update_seconds": _timing_summary(updates),
        "query_seconds": _timing_summary(queries),
        "end_to_end_seconds": _timing_summary(totals),
    }


def _benchmark_method(
    *,
    build: Callable[[], QueryState],
    query: Callable[[QueryState], dict[str, dict[str, Any]]],
    device: str,
    warmup: int,
    repeats: int,
    operation_seed: int,
) -> tuple[QueryState, dict[str, dict[str, Any]], dict[str, Any]]:
    updates: list[float] = []
    queries: list[float] = []
    totals: list[float] = []
    last_state = None
    last_scores = None
    for iteration in range(warmup + repeats):
        _seed_everything(operation_seed)
        _synchronize(device)
        total_started = time.perf_counter()
        update_started = total_started
        state = build()
        _synchronize(device)
        update_elapsed = time.perf_counter() - update_started

        _synchronize(device)
        query_started = time.perf_counter()
        scores = query(state)
        _synchronize(device)
        query_elapsed = time.perf_counter() - query_started
        total_elapsed = time.perf_counter() - total_started
        if iteration >= warmup:
            updates.append(update_elapsed)
            queries.append(query_elapsed)
            totals.append(total_elapsed)
            last_state = state
            last_scores = scores
    if last_state is None or last_scores is None:
        raise RuntimeError("benchmark produced no measured repetition")
    timing = _timing_payload(
        updates,
        queries,
        totals,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    return last_state, last_scores, timing


def _benchmark_existing_state(
    *,
    state: QueryState,
    query: Callable[[QueryState], dict[str, dict[str, Any]]],
    device: str,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    queries: list[float] = []
    last_scores = None
    for iteration in range(warmup + repeats):
        _synchronize(device)
        started = time.perf_counter()
        scores = query(state)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if iteration >= warmup:
            queries.append(elapsed)
            last_scores = scores
    if last_scores is None:
        raise RuntimeError("benchmark produced no measured repetition")
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


def _benchmark_certificate_pair(
    runtime: GemmaRuntime,
    original_memory: Any,
    forget_positions: tuple[int, ...],
    probes: Sequence[Probe],
    *,
    device: str,
    warmup: int,
    repeats: int,
    operation_seed: int,
) -> tuple[
    dict[str, QueryState],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    updates: list[float] = []
    query_samples = {"exact": [], "refit": []}
    total_samples = {"exact": [], "refit": []}
    last_states = None
    last_scores = None
    last_diagnostics = None
    for iteration in range(warmup + repeats):
        _seed_everything(operation_seed)
        _synchronize(device)
        update_started = time.perf_counter()
        overrides, raw_states = runtime.persistent_certificate_states(
            original_memory,
            forget_positions,
        )
        raw_states.pop("decay", None)
        _synchronize(device)
        update_elapsed = time.perf_counter() - update_started
        states = {
            name: QueryState(
                raw_states[name],
                update_diagnostics={"shared_update": "certificate_pair"},
            )
            for name in ("exact", "refit")
        }
        scores: dict[str, dict[str, dict[str, Any]]] = {}
        elapsed_by_case: dict[str, float] = {}
        for case in ("exact", "refit"):
            _synchronize(device)
            started = time.perf_counter()
            scores[case] = _score_probe_suite(
                runtime,
                states[case],
                probes,
            )
            _synchronize(device)
            elapsed_by_case[case] = time.perf_counter() - started
        if iteration >= warmup:
            updates.append(update_elapsed)
            for case in ("exact", "refit"):
                query_samples[case].append(elapsed_by_case[case])
                total_samples[case].append(
                    update_elapsed + elapsed_by_case[case]
                )
            last_states = states
            last_scores = scores
            last_diagnostics = _solver_diagnostics(overrides)
    if (
        last_states is None
        or last_scores is None
        or last_diagnostics is None
    ):
        raise RuntimeError("certificate benchmark produced no measured repetition")
    timings = {
        case: _timing_payload(
            updates,
            query_samples[case],
            total_samples[case],
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        for case in ("exact", "refit")
    }
    for case in timings:
        timings[case]["update_seconds"]["shared_timing_group"] = (
            "exact_decrement_and_fixed_c_refit"
        )
    return last_states, last_scores, timings, last_diagnostics


def _target_ids(runtime: GemmaRuntime, prompt: str, target: str) -> tuple[int, ...]:
    rendered = str(target)
    if prompt and not prompt[-1].isspace() and not rendered[:1].isspace():
        rendered = " " + rendered
    token_ids = tuple(
        int(token_id)
        for token_id in runtime.tokenizer(
            rendered,
            add_special_tokens=False,
        ).input_ids
    )
    if not token_ids:
        raise ValueError("target tokenization is empty")
    return token_ids


def _stem_prompt(question: str, stem: str) -> str:
    prompt = f"\n\nQuestion: {question}\nAnswer:"
    return prompt + (f" {stem}" if stem else "")


def _build_probes(
    runtime: GemmaRuntime,
    spec: Mapping[str, Any],
) -> list[Probe]:
    question = str(spec["question"])
    answer = str(spec["answer"])
    probes: list[Probe] = []
    for index, field in enumerate(spec["fields"]):
        target = str(field["value"])
        value_start = answer.index(target)
        stem = answer[:value_start].rstrip()
        prompt = _stem_prompt(question, stem)
        probes.append(
            Probe(
                probe_id=f"deleted_field_{index}",
                kind="deleted",
                prompt=prompt,
                target=target,
                target_ids=_target_ids(runtime, prompt, target),
                field_index=index,
                field_name=str(field["name"]),
            )
        )

    retain_answer = str(spec["retain_answer"])
    retain_target = str(spec["retain_field"]["value"])
    value_start = retain_answer.index(retain_target)
    retain_stem = retain_answer[:value_start].rstrip()
    retain_prompt = _stem_prompt(
        str(spec["retain_question"]),
        retain_stem,
    )
    probes.append(
        Probe(
            probe_id="retained_field",
            kind="retained",
            prompt=retain_prompt,
            target=retain_target,
            target_ids=_target_ids(runtime, retain_prompt, retain_target),
            field_name=str(spec["retain_field"]["name"]),
        )
    )
    return probes


def _score_probe_suite(
    runtime: GemmaRuntime,
    state: QueryState,
    probes: Sequence[Probe],
) -> dict[str, dict[str, Any]]:
    return {
        probe.probe_id: runtime.score_persistent(
            state.memory,
            state.prompt_prefix + probe.prompt,
            list(probe.target_ids),
        )
        for probe in probes
    }


def _serialize_score(score: Mapping[str, Any], probe: Probe) -> dict[str, Any]:
    first_log_probs = score["first_log_probs"]
    first_target = int(probe.target_ids[0])
    return {
        "target_token_count": len(probe.target_ids),
        "total_log_probability": float(score["total_log_probability"]),
        "mean_log_probability": float(score["mean_log_probability"]),
        "geometric_mean_probability": float(
            score["geometric_mean_probability"]
        ),
        "first_target_token_probability": float(
            math.exp(float(first_log_probs[first_target]))
        ),
        "first_target_token_rank": first_token_rank(
            first_log_probs,
            first_target,
        ),
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values))


def _behavioral_metrics(
    probes: Sequence[Probe],
    method_scores: Mapping[str, Mapping[str, Any]],
    present_scores: Mapping[str, Mapping[str, Any]],
    repack_scores: Mapping[str, Mapping[str, Any]],
    *,
    compact: bool,
) -> dict[str, Any]:
    deleted = []
    retained = None
    deleted_kls = []
    for probe in probes:
        method = method_scores[probe.probe_id]
        present = present_scores[probe.probe_id]
        repack = repack_scores[probe.probe_id]
        kl_to_repack = full_vocabulary_kl(
            repack["first_log_probs"],
            method["first_log_probs"],
        )
        row = {
            "probe_id": probe.probe_id,
            "score": _serialize_score(method, probe),
            "suppression_vs_present_nats": float(
                present["mean_log_probability"]
                - method["mean_log_probability"]
            ),
            "mean_log_probability_drift_from_repack_nats": float(
                method["mean_log_probability"]
                - repack["mean_log_probability"]
            ),
            "full_vocabulary_behavioral_kl_to_repack_nats": kl_to_repack,
        }
        if not compact:
            row["field_name"] = probe.field_name
            row["prompt"] = probe.prompt
            row["target"] = probe.target
        if probe.kind == "deleted":
            deleted.append(row)
            deleted_kls.append(kl_to_repack)
        else:
            retained = row
    if retained is None:
        raise RuntimeError("paired retained probe is missing")
    return {
        "deleted_target_quality": {
            "fields": deleted,
            "mean_log_probability": _mean(
                [item["score"]["mean_log_probability"] for item in deleted]
            ),
            "mean_geometric_probability": _mean(
                [
                    item["score"]["geometric_mean_probability"]
                    for item in deleted
                ]
            ),
            "mean_suppression_vs_present_nats": _mean(
                [item["suppression_vs_present_nats"] for item in deleted]
            ),
            "mean_drift_from_repack_nats": _mean(
                [
                    item["mean_log_probability_drift_from_repack_nats"]
                    for item in deleted
                ]
            ),
        },
        "retained_quality": retained,
        "behavioral_kl_to_repack": {
            "distinct_from_solver_certificate": True,
            "direction": "KL(full_repack || method)",
            "full_vocabulary": True,
            "distribution_scope": "first target token at each probe",
            "deleted_probe_mean_nats": _mean(deleted_kls),
            "deleted_probe_max_nats": max(deleted_kls),
            "retained_probe_nats": retained[
                "full_vocabulary_behavioral_kl_to_repack_nats"
            ],
        },
    }


def _solver_diagnostics(overrides: Mapping[str, Any]) -> dict[str, Any]:
    raw_box = overrides.get("box_C")
    raw_boundary_boxes = overrides.get("box_C_by_boundary") or {}
    boundary_boxes = {
        int(start): float(value)
        for start, value in raw_boundary_boxes.items()
    }
    if raw_box is None and not boundary_boxes:
        raise ValueError("solver diagnostics contain no fixed-C specification")
    return {
        "solver_precision": "float64",
        "model_forward_precision": "float32",
        "fixed_c_feasible": bool(overrides["fixed_c_feasible"]),
        "box_mode": "per_boundary" if boundary_boxes else "global",
        "box_C": None if raw_box is None else float(raw_box),
        "box_C_by_boundary": boundary_boxes or None,
        "head_gate_solves": int(overrides["n_solves"]),
        "head_gates": int(overrides["n_head_gates"]),
        "decrement_fallbacks": int(overrides["n_fallback"]),
        "used_refit_fallback": bool(overrides["used_refit_fallback"]),
        "max_functional_deviation": float(
            overrides["max_functional_deviation"]
        ),
        "max_candidate_deviation": float(
            overrides["max_candidate_deviation"]
        ),
        "functional_tolerance": float(overrides["functional_tolerance"]),
        "feasibility": overrides["feasibility"],
        "fallback_details": overrides["fallback_details"],
        "objective": overrides.get("objective"),
    }


def _solver_certificate(
    probes: Sequence[Probe],
    exact_scores: Mapping[str, Mapping[str, Any]],
    refit_scores: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for probe in probes:
        value = full_vocabulary_kl(
            exact_scores[probe.probe_id]["first_log_probs"],
            refit_scores[probe.probe_id]["first_log_probs"],
        )
        rows.append(
            {
                "probe_id": probe.probe_id,
                "probe_kind": probe.kind,
                "full_vocabulary_output_kl_nats": value,
            }
        )
    values = [row["full_vocabulary_output_kl_nats"] for row in rows]
    return {
        "status": "completed",
        "claim": "exact decrement versus fixed-C retained-key refit",
        "distinct_from_behavioral_kl_to_repack": True,
        "direction": "KL(exact_decrement || fixed_c_refit)",
        "full_vocabulary": True,
        "distribution_scope": "first target token at each probe",
        "probe_output_kls": rows,
        "mean_output_kl_nats": _mean(values),
        "max_output_kl_nats": max(values),
        "solver_diagnostics": dict(diagnostics),
    }


def _precision_audit(
    runtime: GemmaRuntime,
    context,
    probes: Sequence[Probe],
    *,
    warmup: int,
    repeats: int,
    operation_seed: int,
) -> dict[str, Any]:
    """Run the prefill-once exact/refit/proxy bridge in a float64 model."""

    device = runtime.config.device
    _seed_everything(operation_seed)
    _synchronize(device)
    prefill_started = time.perf_counter()
    persistent = runtime.prefill_persistent(list(context.original_token_ids))
    _synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_started

    exact_update_samples: list[float] = []
    proxy_update_samples: list[float] = []
    query_samples: list[float] = []
    last_states = None
    last_scores = None
    last_overrides = None
    for iteration in range(warmup + repeats):
        _seed_everything(operation_seed)
        _synchronize(device)
        started = time.perf_counter()
        overrides, states = runtime.persistent_certificate_states(
            persistent,
            context.forget_positions,
        )
        _synchronize(device)
        exact_update = time.perf_counter() - started

        _synchronize(device)
        started = time.perf_counter()
        states["proxy"] = runtime.delete_persistent(
            persistent,
            context.forget_positions,
            kind="float32_fista_proxy_in_float64_model",
        )
        _synchronize(device)
        proxy_update = time.perf_counter() - started

        _synchronize(device)
        started = time.perf_counter()
        scores = {
            case: _score_probe_suite(
                runtime,
                QueryState(states[case]),
                probes,
            )
            for case in ("exact", "refit", "proxy", "decay")
        }
        _synchronize(device)
        query_seconds = time.perf_counter() - started
        if iteration >= warmup:
            exact_update_samples.append(exact_update)
            proxy_update_samples.append(proxy_update)
            query_samples.append(query_seconds)
            last_states = states
            last_scores = scores
            last_overrides = overrides

    if last_states is None or last_scores is None or last_overrides is None:
        raise RuntimeError("precision audit produced no measured repetition")

    probe_rows = []
    for probe in probes:
        exact = last_scores["exact"][probe.probe_id]["first_log_probs"]
        refit = last_scores["refit"][probe.probe_id]["first_log_probs"]
        proxy = last_scores["proxy"][probe.probe_id]["first_log_probs"]
        decay = last_scores["decay"][probe.probe_id]["first_log_probs"]
        probe_rows.append(
            {
                "probe_id": probe.probe_id,
                "probe_kind": probe.kind,
                "kl_exact_vs_refit_nats": full_vocabulary_kl(exact, refit),
                "kl_proxy_vs_exact_nats": full_vocabulary_kl(exact, proxy),
                "kl_decay_vs_refit_nats": full_vocabulary_kl(decay, refit),
            }
        )

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in probe_rows]

    exact_values = values("kl_exact_vs_refit_nats")
    proxy_values = values("kl_proxy_vs_exact_nats")
    decay_values = values("kl_decay_vs_refit_nats")
    diagnostics = _solver_diagnostics(last_overrides)
    diagnostics["model_forward_precision"] = "float64"
    state_storage = {
        case: method_storage_report(persistent, last_states[case])
        for case in ("exact", "refit", "proxy", "decay")
    }
    return {
        "status": "completed",
        "claim": "prefill-once exact/refit and exact/proxy output audit",
        "same_context_prefill_once": True,
        "query_independent_updates": True,
        "model_forward_precision": "float64",
        "solver_precision": "float64 exact/refit; float32 FISTA proxy",
        "directions": {
            "kl_exact_vs_refit_nats": "KL(exact_decrement || fixed_c_refit)",
            "kl_proxy_vs_exact_nats": "KL(exact_decrement || fp32_proxy)",
            "kl_decay_vs_refit_nats": "KL(decay || fixed_c_refit)",
        },
        "prefill": {
            "seconds": prefill_seconds,
            "input_digest": persistent.input_digest,
            "storage": tensor_storage_report(persistent),
        },
        "probe_rows": probe_rows,
        "summary": {
            "probe_count": len(probe_rows),
            "mean_kl_exact_vs_refit_nats": _mean(exact_values),
            "max_kl_exact_vs_refit_nats": max(exact_values),
            "mean_kl_proxy_vs_exact_nats": _mean(proxy_values),
            "max_kl_proxy_vs_exact_nats": max(proxy_values),
            "mean_kl_decay_vs_refit_nats": _mean(decay_values),
            "max_kl_decay_vs_refit_nats": max(decay_values),
        },
        "solver_diagnostics": diagnostics,
        "timing": {
            "clock": "time.perf_counter",
            "synchronized_device": device,
            "warmup": warmup,
            "repeats": repeats,
            "exact_refit_decay_shared_update_seconds": _timing_summary(
                exact_update_samples
            ),
            "proxy_update_seconds": _timing_summary(proxy_update_samples),
            "all_four_states_query_seconds": _timing_summary(query_samples),
        },
        "storage": state_storage,
    }


def _answer_admission_score(
    runtime: GemmaRuntime,
    memory: Any,
    question: str,
    answer: str,
) -> dict[str, Any]:
    prompt = f"\n\nQuestion: {question}\nAnswer:"
    return runtime.score_persistent(
        memory,
        prompt,
        list(_target_ids(runtime, prompt, answer)),
    )


def _admission(
    runtime: GemmaRuntime,
    spec: Mapping[str, Any],
    probes: Sequence[Probe],
    present_memory: Any,
    repack_memory: Any,
    present_scores: Mapping[str, Mapping[str, Any]],
    repack_scores: Mapping[str, Mapping[str, Any]],
    forget_positions: tuple[int, ...],
) -> dict[str, Any]:
    reasons: list[str] = []
    present_answer = _answer_admission_score(
        runtime,
        present_memory,
        str(spec["question"]),
        str(spec["answer"]),
    )
    repack_answer = _answer_admission_score(
        runtime,
        repack_memory,
        str(spec["question"]),
        str(spec["answer"]),
    )
    answer_lift = float(
        present_answer["mean_log_probability"]
        - repack_answer["mean_log_probability"]
    )
    if answer_lift < MIN_SIGNAL_NATS:
        reasons.append("record:answer_lift")

    fields = []
    for probe in probes:
        present = present_scores[probe.probe_id]
        if probe.kind == "deleted":
            repack = repack_scores[probe.probe_id]
            lift = float(
                present["mean_log_probability"]
                - repack["mean_log_probability"]
            )
            rank = first_token_rank(
                present["first_log_probs"],
                probe.target_ids[0],
            )
            if lift < MIN_SIGNAL_NATS:
                reasons.append(f"{probe.probe_id}:secret_lift")
            if rank > MAX_FIRST_TOKEN_RANK:
                reasons.append(f"{probe.probe_id}:rank")
            fields.append(
                {
                    "probe_id": probe.probe_id,
                    "secret_lift_nats": lift,
                    "first_token_rank": rank,
                }
            )
        else:
            rank = first_token_rank(
                present["first_log_probs"],
                probe.target_ids[0],
            )
            if rank > MAX_FIRST_TOKEN_RANK:
                reasons.append("retained_field:rank")
            retained = {
                "probe_id": probe.probe_id,
                "first_token_rank": rank,
            }
    _, feasibility = fixed_c_feasibility(
        int(present_memory.token_count),
        forget_positions,
    )
    if any(not bool(item["feasible"]) for item in feasibility):
        reasons.append("fixed_c:infeasible")
    return {
        "status": "rejected" if reasons else "admitted",
        "reasons": sorted(set(reasons)),
        "thresholds": {
            "minimum_answer_lift_nats": MIN_SIGNAL_NATS,
            "minimum_secret_lift_nats": MIN_SIGNAL_NATS,
            "maximum_first_token_rank": MAX_FIRST_TOKEN_RANK,
            "all_fields_must_pass": True,
            "fixed_c_all_boundaries_must_be_feasible": True,
        },
        "record_answer_lift_nats": answer_lift,
        "deleted_fields": fields,
        "retained_field": retained,
        "fixed_c_feasibility": feasibility,
    }


def _method_report(
    method_id: str,
    state: QueryState,
    scores: Mapping[str, Mapping[str, Any]],
    timing: Mapping[str, Any],
    *,
    original_memory: Any,
    probes: Sequence[Probe],
    present_scores: Mapping[str, Mapping[str, Any]],
    repack_scores: Mapping[str, Mapping[str, Any]],
    compact: bool,
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
            repack_scores,
            compact=compact,
        ),
        "timing": dict(timing),
        "storage": method_storage_report(
            original_memory,
            state.memory,
            prompt_token_count=state.prompt_token_count,
        ),
        "update_diagnostics": state.update_diagnostics or {},
    }


def _failed_method(
    method_id: str,
    exc: Exception,
    *,
    compact: bool,
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": (
            "redacted_in_compact_mode"
            if compact
            else str(exc)
        ),
    }


def _model_storage(runtime: GemmaRuntime) -> dict[str, Any]:
    named_parameters = list(runtime.model.named_parameters())
    buffers = list(runtime.model.buffers())
    all_tensors = [parameter for _, parameter in named_parameters] + buffers
    adapter_tensors = [
        parameter
        for name, parameter in named_parameters
        if "lora_" in name.casefold() or "adapter" in name.casefold()
    ]
    return {
        "model_parameters_and_buffers": tensor_storage_report(all_tensors),
        "adapter_named_tensors": tensor_storage_report(adapter_tensors),
        "adapter_tensor_count": len(adapter_tensors),
    }


def _record_seed(base_seed: int, manifest_index: int) -> int:
    return int(base_seed) + 100_003 * int(manifest_index)


def _evaluate_record(
    runtime: GemmaRuntime,
    certificate_runtime: GemmaRuntime | None,
    manifest_records: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    manifest_index: int,
    *,
    args,
    copies: int,
) -> dict[str, Any]:
    record_seed = _record_seed(args.seed, manifest_index)
    _seed_everything(record_seed)
    context = build_synthetic_record_context(
        runtime.tokenizer,
        spec,
        runtime.fillers,
        window=args.window,
        min_fillers=args.n_fill,
        prefix_fillers=args.prefix_fillers,
        copies=copies,
    )
    probes = _build_probes(runtime, spec)
    row: dict[str, Any] = {
        "manifest_index": manifest_index,
        "record_id": str(spec["record_id"]),
        "record_seed": record_seed,
        "deletion_scope": "whole_record",
        "context": {
            "original_token_count": len(context.original_token_ids),
            "edited_token_count": len(context.edited_token_ids),
            "deleted_token_count": len(context.forget_positions),
            "deletion_ranges": [
                {"start": start, "end": end}
                for start, end in context.deletion_ranges
            ],
            "disjoint_deletion_ranges": len(context.deletion_ranges),
            "paired_retained_token_count": len(context.retain_positions),
            "copies": copies,
            "neutral_copy_separators": max(0, copies - 1),
            "original_token_ids_sha256": context.original_digest,
            "edited_token_ids_sha256": context.edited_digest,
            "literal_edit_preserves_retained_tokens": True,
        },
        "probes": [
            {
                "probe_id": probe.probe_id,
                "probe_kind": probe.kind,
                "prompt_token_count": len(
                    runtime.tokenizer(
                        probe.prompt,
                        add_special_tokens=False,
                    ).input_ids
                ),
                "prompt_token_ids_sha256": token_ids_digest(
                    runtime.tokenizer(
                        probe.prompt,
                        add_special_tokens=False,
                    ).input_ids
                ),
                "target_token_count": len(probe.target_ids),
            }
            for probe in probes
        ],
        "methods": {},
    }
    if not args.compact:
        row["source"] = dict(spec)

    _synchronize(runtime.config.device)
    prefill_started = time.perf_counter()
    original_memory = runtime.prefill_persistent(
        list(context.original_token_ids)
    )
    _synchronize(runtime.config.device)
    row["shared_original_prefill"] = {
        "seconds": time.perf_counter() - prefill_started,
        "synchronized_device": runtime.config.device,
        "input_digest": original_memory.input_digest,
        "storage": method_storage_report(
            original_memory,
            original_memory,
        )["state"],
    }
    source_signature = persistent_state_shape_signature(original_memory)
    query = lambda state: _score_probe_suite(runtime, state, probes)

    present_state = QueryState(
        original_memory,
        update_diagnostics={"reference_kind": "present_control"},
    )
    present_scores, present_timing = _benchmark_existing_state(
        state=present_state,
        query=query,
        device=runtime.config.device,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    repack_state, repack_scores, repack_timing = _benchmark_method(
        build=lambda: QueryState(
            runtime.prefill_persistent(list(context.edited_token_ids)),
            update_diagnostics={
                "reference_kind": "fresh_full_edited_context_prefill",
                "suffix_recomputed": True,
                "solver_refit": "fresh prefill",
            },
        ),
        query=query,
        device=runtime.config.device,
        warmup=args.warmup,
        repeats=args.repeats,
        operation_seed=record_seed,
    )
    row["admission"] = _admission(
        runtime,
        spec,
        probes,
        original_memory,
        repack_state.memory,
        present_scores,
        repack_scores,
        context.forget_positions,
    )
    row["references"] = {
        "present_control": {
            "status": "completed",
            "timing": present_timing,
            **_behavioral_metrics(
                probes,
                present_scores,
                present_scores,
                repack_scores,
                compact=args.compact,
            ),
        }
    }
    row["methods"]["full_repack"] = _method_report(
        "full_repack",
        repack_state,
        repack_scores,
        repack_timing,
        original_memory=original_memory,
        probes=probes,
        present_scores=present_scores,
        repack_scores=repack_scores,
        compact=args.compact,
        semantics={
            "behavioral_reference": True,
            "fresh_prefill": True,
            "literal_edited_context": True,
            "suffix_recomputed": True,
        },
    )

    try:
        proxy_state, proxy_scores, proxy_timing = _benchmark_method(
            build=lambda: QueryState(
                runtime.delete_persistent(
                    original_memory,
                    context.forget_positions,
                    kind="fp32_masked_refit_proxy",
                ),
                update_diagnostics={
                    "solver": "feasible_projected_fp32_fista_proxy",
                    "query_independent": True,
                },
            ),
            query=query,
            device=runtime.config.device,
            warmup=args.warmup,
            repeats=args.repeats,
            operation_seed=record_seed + 2,
        )
        row["methods"]["fp32_proxy"] = _method_report(
            "fp32_proxy",
            proxy_state,
            proxy_scores,
            proxy_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            repack_scores=repack_scores,
            compact=args.compact,
            semantics={
                "solver": "feasible projected FP32 FISTA masked refit",
                "suffix_recomputed": False,
                "query_independent": True,
            },
        )
    except Exception as exc:
        row["methods"]["fp32_proxy"] = _failed_method(
            "fp32_proxy",
            exc,
            compact=args.compact,
        )

    try:
        def build_cache_state():
            memory, diagnostics = cache_delete_and_shift(
                original_memory,
                context.forget_positions,
                edited_token_ids=context.edited_token_ids,
            )
            diagnostics["logical_edited_token_ids_sha256"] = (
                context.edited_digest
            )
            return QueryState(
                memory,
                update_diagnostics=diagnostics,
            )

        cache_state, cache_scores, cache_timing = _benchmark_method(
            build=build_cache_state,
            query=query,
            device=runtime.config.device,
            warmup=args.warmup,
            repeats=args.repeats,
            operation_seed=record_seed + 3,
        )
        row["methods"]["cache_delete_shift"] = _method_report(
            "cache_delete_shift",
            cache_state,
            cache_scores,
            cache_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            repack_scores=repack_scores,
            compact=args.compact,
            semantics={
                "diagnostic_only": True,
                "physically_deletes_sv_session_rows": True,
                "edits_all_matching_resident_native_kv_rows": True,
                "solver_refit": False,
                "suffix_recomputed": False,
                "rope_keys_rerotated": False,
            },
        )
    except Exception as exc:
        row["methods"]["cache_delete_shift"] = _failed_method(
            "cache_delete_shift",
            exc,
            compact=args.compact,
        )

    try:
        def build_decay_state():
            memory = original_memory.fork()
            memory.request = GateRequest(
                scale_pos=context.forget_positions,
                scale_factor=DECAY_FACTOR,
                gate_floor=float(
                    getattr(
                        getattr(original_memory, "request", None),
                        "gate_floor",
                        0.0,
                    )
                ),
            )
            memory.deleted_positions = context.forget_positions
            memory.deletion_kind = "coefficient_decay_0_01"
            return QueryState(
                memory,
                update_diagnostics={
                    "decay_factor": DECAY_FACTOR,
                    "solver_refit": False,
                },
            )

        decay_state, decay_scores, decay_timing = _benchmark_method(
            build=build_decay_state,
            query=query,
            device=runtime.config.device,
            warmup=args.warmup,
            repeats=args.repeats,
            operation_seed=record_seed + 4,
        )
        row["methods"]["decay_0_01"] = _method_report(
            "decay_0_01",
            decay_state,
            decay_scores,
            decay_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            repack_scores=repack_scores,
            compact=args.compact,
            semantics={
                "decay_factor": DECAY_FACTOR,
                "positions_remain_resident": True,
                "suffix_recomputed": False,
                "solver_refit": False,
            },
        )
    except Exception as exc:
        row["methods"]["decay_0_01"] = _failed_method(
            "decay_0_01",
            exc,
            compact=args.compact,
        )

    try:
        def build_icul_state():
            icul = build_faithful_icul_context(
                spec,
                manifest_records,
                seed=record_seed,
                correct_demonstrations=ICUL_CORRECT_DEMONSTRATIONS,
            )
            prefix_ids = runtime.tokenizer(
                icul.text,
                add_special_tokens=False,
            ).input_ids
            diagnostics = {
                "faithful_icul": True,
                "wrong_answer_deterministic": True,
                "wrong_answer_record_id": icul.wrong_answer_record_id,
                "correct_demonstrations": len(
                    icul.demonstration_record_ids
                ),
                "demonstration_record_ids": list(
                    icul.demonstration_record_ids
                ),
                "prompt_token_ids_sha256": token_ids_digest(prefix_ids),
                "base_cache_unchanged": True,
            }
            if not args.compact:
                diagnostics["prompt_prefix"] = icul.text
            return QueryState(
                original_memory,
                prompt_prefix=icul.text,
                prompt_token_count=len(prefix_ids),
                update_diagnostics=diagnostics,
            )

        icul_state, icul_scores, icul_timing = _benchmark_method(
            build=build_icul_state,
            query=query,
            device=runtime.config.device,
            warmup=args.warmup,
            repeats=args.repeats,
            operation_seed=record_seed + 5,
        )
        row["methods"]["icul_4"] = _method_report(
            "icul_4",
            icul_state,
            icul_scores,
            icul_timing,
            original_memory=original_memory,
            probes=probes,
            present_scores=present_scores,
            repack_scores=repack_scores,
            compact=args.compact,
            semantics={
                "wrong_target_answer": True,
                "correct_retained_demonstrations": 4,
                "temperature": 0,
                "base_context_identical": True,
                "adds_query_context": True,
                "cache_deletion": False,
            },
        )
    except Exception as exc:
        row["methods"]["icul_4"] = _failed_method(
            "icul_4",
            exc,
            compact=args.compact,
        )

    try:
        (
            certificate_states,
            certificate_scores,
            certificate_timings,
            solver_diagnostics,
        ) = _benchmark_certificate_pair(
            runtime,
            original_memory,
            context.forget_positions,
            probes,
            device=runtime.config.device,
            warmup=args.warmup,
            repeats=args.repeats,
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
            repack_scores=repack_scores,
            compact=args.compact,
            semantics={
                "gate_solver": "float64 Cauwenberghs-Poggio decrement",
                "fixed_C": True,
                "suffix_recomputed": False,
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
            repack_scores=repack_scores,
            compact=args.compact,
            semantics={
                "gate_solver": "float64 retained-key from-scratch refit",
                "fixed_C": True,
                "suffix_recomputed": False,
            },
        )
        row["solver_certificate"] = _solver_certificate(
            probes,
            certificate_scores["exact"],
            certificate_scores["refit"],
            solver_diagnostics,
        )
    except Exception as exc:
        row["methods"]["exact_decrement"] = _failed_method(
            "exact_decrement",
            exc,
            compact=args.compact,
        )
        row["methods"]["fixed_c_refit"] = _failed_method(
            "fixed_c_refit",
            exc,
            compact=args.compact,
        )
        row["solver_certificate"] = {
            "status": "failed",
            "distinct_from_behavioral_kl_to_repack": True,
            "error_type": type(exc).__name__,
            "error": (
                "redacted_in_compact_mode"
                if args.compact
                else str(exc)
            ),
        }

    if certificate_runtime is not None:
        try:
            row["prefill_once_precision_audit"] = _precision_audit(
                certificate_runtime,
                context,
                probes,
                warmup=args.certificate_warmup,
                repeats=args.certificate_repeats,
                operation_seed=record_seed + 7,
            )
        except Exception as exc:
            row["prefill_once_precision_audit"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": (
                    "redacted_in_compact_mode"
                    if args.compact
                    else str(exc)
                ),
            }

    exclusion = kveraser_exclusion()
    row["methods"]["kveraser"] = {
        "method_id": "kveraser",
        "status": "excluded",
        "evaluated": False,
        "exclusion_reason_codes": [
            reason["code"] for reason in exclusion["reasons"]
        ],
    }
    row["source_state_immutability"] = {
        "verification_scope": "metadata_and_tensor_shapes",
        "shape_signature_unchanged": (
            persistent_state_shape_signature(original_memory)
            == source_signature
        ),
        "input_digest_unchanged": (
            original_memory.input_digest == context.original_digest
        ),
    }
    if not (
        row["source_state_immutability"]["shape_signature_unchanged"]
        and row["source_state_immutability"]["input_digest_unchanged"]
    ):
        raise RuntimeError("a deletion method mutated the shared source state")
    row["method_failures"] = [
        method_id
        for method_id, result in row["methods"].items()
        if result.get("status") == "failed"
    ]
    if row.get("prefill_once_precision_audit", {}).get("status") == "failed":
        row["method_failures"].append("prefill_once_precision_audit")
    row["status"] = (
        "completed"
        if not row["method_failures"]
        else "completed_with_method_failures"
    )
    return row


def _method_value(
    record: Mapping[str, Any],
    method_id: str,
    *path: str,
) -> float | None:
    value: Any = record.get("methods", {}).get(method_id)
    if not isinstance(value, Mapping) or value.get("status") != "completed":
        return None
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summarize_run(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "aggregation_population": (
            "all predeclared completed records, including rejected admissions"
        ),
        "attempted_records": len(records),
        "admitted_records": sum(
            record.get("admission", {}).get("status") == "admitted"
            for record in records
        ),
        "rejected_records": [
            {
                "record_id": record["record_id"],
                "reasons": record.get("admission", {}).get("reasons", []),
            }
            for record in records
            if record.get("admission", {}).get("status") == "rejected"
        ],
        "methods": {},
    }
    for method_id in METHOD_IDS:
        completed = [
            record
            for record in records
            if record.get("methods", {})
            .get(method_id, {})
            .get("status")
            == "completed"
        ]
        method_summary: dict[str, Any] = {
            "completed_records": len(completed),
            "failed_records": len(records) - len(completed),
        }
        metric_paths = {
            "deleted_target_log_probability": (
                "deleted_target_quality",
                "mean_log_probability",
            ),
            "deleted_target_geometric_probability": (
                "deleted_target_quality",
                "mean_geometric_probability",
            ),
            "deleted_suppression_nats": (
                "deleted_target_quality",
                "mean_suppression_vs_present_nats",
            ),
            "retained_drift_nats": (
                "retained_quality",
                "mean_log_probability_drift_from_repack_nats",
            ),
            "retained_mean_log_probability": (
                "retained_quality",
                "score",
                "mean_log_probability",
            ),
            "deleted_behavioral_kl_to_repack_nats": (
                "behavioral_kl_to_repack",
                "deleted_probe_mean_nats",
            ),
            "retained_behavioral_kl_to_repack_nats": (
                "behavioral_kl_to_repack",
                "retained_probe_nats",
            ),
            "update_median_seconds": (
                "timing",
                "update_seconds",
                "median",
            ),
            "query_median_seconds": (
                "timing",
                "query_seconds",
                "median",
            ),
            "end_to_end_median_seconds": (
                "timing",
                "end_to_end_seconds",
                "median",
            ),
            "state_tensor_storage_bytes": (
                "storage",
                "state",
                "deduplicated_tensor_storage_bytes",
            ),
            "incremental_tensor_storage_bytes": (
                "storage",
                "incremental_tensor_storage_bytes",
            ),
            "prompt_token_count": (
                "storage",
                "prompt_token_count",
            ),
        }
        for name, path in metric_paths.items():
            values = [
                value
                for record in completed
                if (
                    value := _method_value(
                        record,
                        method_id,
                        *path,
                    )
                )
                is not None
            ]
            if values:
                method_summary[f"mean_{name}"] = _mean(values)
                method_summary[f"max_{name}"] = max(values)
        summary["methods"][method_id] = method_summary
    precision_rows = [
        record["prefill_once_precision_audit"]
        for record in records
        if record.get("prefill_once_precision_audit", {}).get("status")
        == "completed"
    ]
    if precision_rows:
        precision_summary: dict[str, Any] = {
            "completed_records": len(precision_rows),
            "failed_records": len(records) - len(precision_rows),
        }
        for key in (
            "mean_kl_exact_vs_refit_nats",
            "max_kl_exact_vs_refit_nats",
            "mean_kl_proxy_vs_exact_nats",
            "max_kl_proxy_vs_exact_nats",
            "mean_kl_decay_vs_refit_nats",
            "max_kl_decay_vs_refit_nats",
        ):
            values = [float(row["summary"][key]) for row in precision_rows]
            precision_summary[f"record_mean_{key}"] = _mean(values)
            precision_summary[f"record_max_{key}"] = max(values)
        precision_summary["decrement_fallbacks"] = sum(
            int(row["solver_diagnostics"]["decrement_fallbacks"])
            for row in precision_rows
        )
        summary["prefill_once_precision_audit"] = precision_summary
    return summary


def _environment() -> dict[str, Any]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for package in ("torch", "transformers", "peft", "mlx"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {
        "platform": platform.platform(),
        "versions": versions,
    }


def _adapter_label(adapter: str | None, run_index: int) -> str:
    if adapter is None:
        return f"run-{run_index:02d}-no-adapter"
    return f"run-{run_index:02d}-{Path(adapter).name}"


def _adapter_provenance(adapter: str | None, expected_seed: int) -> dict[str, Any]:
    if adapter is None:
        return {
            "path": None,
            "content_sha256": None,
            "recovery_results": None,
        }
    adapter_path = Path(adapter)
    descriptor: dict[str, Any] = {
        "path": str(adapter_path),
        "content_sha256": path_sha256(adapter_path),
        "recovery_results": None,
    }
    result_path = adapter_path.parent / "results.json"
    if not result_path.exists():
        return descriptor
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    recorded_seed = (payload.get("config") or {}).get("seed")
    if recorded_seed is not None and int(recorded_seed) != int(expected_seed):
        raise ValueError(
            f"adapter recovery seed {recorded_seed} does not match --seed "
            f"{expected_seed}"
        )
    recorded_hash = payload.get("adapter_sha256")
    if recorded_hash and recorded_hash != descriptor["content_sha256"]:
        raise ValueError("adapter content hash differs from its recovery result")
    descriptor.update(
        {
            "recovery_results": str(result_path),
            "recovery_schema": payload.get("schema"),
            "training_seed": recorded_seed,
            "stage2_stream": payload.get("stage2_stream"),
            "model_revision": (payload.get("config") or {}).get(
                "model_revision"
            ),
        }
    )
    return descriptor


def _atomic_write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _release_runtime(runtime: GemmaRuntime | None) -> None:
    if runtime is not None:
        runtime.model = None
        runtime.tokenizer = None
        runtime.layers = {}
        runtime.controller = None
        runtime.loaded = False
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Hugging Face revision (the default 1B model is pinned automatically)",
    )
    parser.add_argument(
        "--adapters",
        default=DEFAULT_ADAPTER,
        help=(
            "comma-separated adapter paths; use 'none' for no adapter. A "
            "single adapter is broadcast across multiple devices"
        ),
    )
    parser.add_argument(
        "--devices",
        default="mps",
        help=(
            "comma-separated devices. A single device is broadcast across "
            "multiple adapters"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument(
        "--records",
        type=int,
        help="number of selected manifest records to evaluate",
    )
    parser.add_argument(
        "--record",
        "--record-slice",
        dest="record",
        action="append",
        default=[],
        help=(
            "record ID, zero-based index, or Python-style slice; repeat for "
            "a union (examples: --record case-zaffre --record 0:3)"
        ),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--precision-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the float64 prefill-once exact/refit/proxy audit",
    )
    parser.add_argument("--certificate-device", default="cpu")
    parser.add_argument("--certificate-warmup", type=int, default=0)
    parser.add_argument("--certificate-repeats", type=int, default=1)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--n-fill", type=int, default=22)
    parser.add_argument("--prefix-fillers", type=int, default=8)
    parser.add_argument(
        "--compact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="omit benchmark questions, answers, targets, and prompt text",
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing --out path",
    )
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.model_revision is None and args.model == "google/gemma-3-1b-pt":
        args.model_revision = MODEL_REVISION
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats positive")
    if args.certificate_warmup < 0 or args.certificate_repeats < 1:
        parser.error(
            "--certificate-warmup must be non-negative and "
            "--certificate-repeats positive"
        )
    if args.window < 1 or args.n_fill < 1 or args.prefix_fillers < 0:
        parser.error("packing parameters must be positive")
    try:
        adapters = _comma_values(args.adapters)
        devices = _comma_values(args.devices)
        run_matrix = _run_matrix(adapters, devices)
        run_provenance = [
            _adapter_provenance(adapter, args.seed)
            for adapter, _device in run_matrix
        ]
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_records = list(manifest["records"])
    try:
        selected_records = _select_records(
            all_records,
            record_start=args.record_start,
            record_count=args.records,
            selectors=args.record,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not selected_records:
        parser.error("record selection is empty")
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(
            f"{output} already exists; choose another --out or pass --overwrite"
        )

    copies = int(manifest.get("selection_policy", {}).get("copies", 1))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "persistent matched deletion baselines",
        "status": "in_progress",
        "contains_source_text": not args.compact,
        "contains_full_vocabulary_vectors": False,
        "manifest": {
            "name": manifest["name"],
            "version": manifest["version"],
            "path": str(manifest_path),
            "fixed_before_evaluation": bool(
                manifest.get("selection_policy", {}).get(
                    "fixed_before_evaluation",
                    False,
                )
            ),
            "total_records": len(all_records),
            "selected_manifest_indices": [
                index for index, _ in selected_records
            ],
        },
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "adapters": [
                None if value.casefold() in {"none", "null", "-"} else value
                for value in adapters
            ],
            "devices": devices,
            "seed": args.seed,
            "record_start": args.record_start,
            "records": args.records,
            "record_selectors": list(args.record),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "precision_audit": args.precision_audit,
            "certificate_device": args.certificate_device,
            "certificate_warmup": args.certificate_warmup,
            "certificate_repeats": args.certificate_repeats,
            "window": args.window,
            "n_fill": args.n_fill,
            "prefix_fillers": args.prefix_fillers,
            "copies": copies,
            "compact": args.compact,
            "decay_factor": DECAY_FACTOR,
            "icul_correct_demonstrations": ICUL_CORRECT_DEMONSTRATIONS,
        },
        "metric_definitions": {
            "deleted_suppression": (
                "present mean target log-probability minus method mean target "
                "log-probability; positive is stronger suppression"
            ),
            "retained_drift": (
                "method retained mean target log-probability minus full-repack "
                "retained mean target log-probability"
            ),
            "behavioral_kl_to_repack": (
                "full-vocabulary KL(full_repack || method) at the first target "
                "token; this is not the solver certificate"
            ),
            "solver_certificate_kl": (
                "full-vocabulary KL(exact_decrement || fixed_c_refit); reported "
                "in a separate solver_certificate object"
            ),
            "storage": (
                "unique underlying Torch and NumPy tensor storage bytes; views "
                "and repeated references are counted once"
            ),
        },
        "technical_scope": [
            (
                "Practical baseline model forwards use FP32. The separate "
                "prefill_once_precision_audit runs exact/refit/proxy/decay "
                "forwards in a float64 model."
            ),
            (
                "The cache-delete-and-shift method is diagnostic only: it "
                "does not recompute contaminated suffix states or rerotate "
                "RoPE keys."
            ),
            (
                "Targets are packed beyond the native local window. Native "
                "KV rows are edited when resident; older requested rows are "
                "reported as nonresident rather than fabricated."
            ),
            (
                "ICUL shares the same base persistent context but necessarily "
                "adds its wrong-answer and four-demo prefix at query time."
            ),
            (
                "Device strings are forwarded to the existing runtime. Its "
                "current gate prefill still depends on MLX, so CUDA-only hosts "
                "are not a complete execution path."
            ),
        ],
        "excluded_methods": [kveraser_exclusion()],
        "environment": _environment(),
        "runs": [],
    }
    if len(run_provenance) == 1:
        report["seed"] = args.seed
        report["adapter_sha256"] = run_provenance[0]["content_sha256"]
        report["stage2_stream"] = run_provenance[0].get("stage2_stream")
        report["provenance"] = {"adapter": run_provenance[0]}
    _atomic_write(output, report)

    for run_index, (adapter, device) in enumerate(run_matrix):
        run: dict[str, Any] = {
            "run_id": _adapter_label(adapter, run_index),
            "status": "loading",
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter": adapter,
            "device": device,
            "dtype": "float32",
            "seed": args.seed,
            "provenance": {"adapter": run_provenance[run_index]},
            "records": [],
        }
        report["runs"].append(run)
        _atomic_write(output, report)
        runtime = None
        certificate_runtime = None
        try:
            _seed_everything(args.seed)
            runtime = GemmaRuntime(
                RuntimeConfig(
                    model_id=args.model,
                    lora_path=adapter,
                    device=device,
                    dtype="float32",
                    generation_tokens=1,
                    window=args.window,
                    copies=copies,
                    model_revision=args.model_revision,
                )
            )
            runtime.ensure_loaded()
            if args.precision_audit:
                certificate_runtime = GemmaRuntime(
                    RuntimeConfig(
                        model_id=args.model,
                        lora_path=adapter,
                        device=args.certificate_device,
                        dtype="float64",
                        generation_tokens=1,
                        window=args.window,
                        copies=copies,
                        model_revision=args.model_revision,
                    )
                )
                certificate_runtime.ensure_loaded()
            run["status"] = "running"
            run["load_seconds"] = runtime.load_seconds
            run["tensor_storage"] = _model_storage(runtime)
            if certificate_runtime is not None:
                run["certificate_runtime"] = {
                    "device": certificate_runtime.config.device,
                    "dtype": certificate_runtime.config.dtype,
                    "load_seconds": certificate_runtime.load_seconds,
                    "tensor_storage": _model_storage(certificate_runtime),
                }
            _atomic_write(output, report)
            for manifest_index, spec in selected_records:
                print(
                    f"[{run['run_id']}] {spec['record_id']} "
                    f"({manifest_index + 1}/{len(all_records)})",
                    flush=True,
                )
                try:
                    record = _evaluate_record(
                        runtime,
                        certificate_runtime,
                        all_records,
                        spec,
                        manifest_index,
                        args=args,
                        copies=copies,
                    )
                except Exception as exc:
                    record = {
                        "manifest_index": manifest_index,
                        "record_id": str(spec["record_id"]),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": (
                            "redacted_in_compact_mode"
                            if args.compact
                            else str(exc)
                        ),
                    }
                    if not args.compact:
                        record["source"] = dict(spec)
                run["records"].append(record)
                _atomic_write(output, report)
            completed_records = [
                record
                for record in run["records"]
                if record.get("status") != "failed"
            ]
            run["summary"] = _summarize_run(completed_records)
            run["record_failures"] = [
                {
                    "record_id": record["record_id"],
                    "error_type": record["error_type"],
                    "error": record["error"],
                }
                for record in run["records"]
                if record.get("status") == "failed"
            ]
            run["method_failures"] = [
                {
                    "record_id": record["record_id"],
                    "methods": record.get("method_failures", []),
                }
                for record in completed_records
                if record.get("method_failures")
            ]
            if run["record_failures"]:
                run["status"] = "completed_with_record_failures"
            elif run["method_failures"]:
                run["status"] = "completed_with_method_failures"
            else:
                run["status"] = "completed"
        except Exception as exc:
            run["status"] = "failed"
            run["error_type"] = type(exc).__name__
            run["error"] = str(exc)
        finally:
            _release_runtime(runtime)
            _release_runtime(certificate_runtime)
            _atomic_write(output, report)

    statuses = {run["status"] for run in report["runs"]}
    if "failed" in statuses:
        report["status"] = "completed_with_run_failures"
    elif "completed_with_record_failures" in statuses:
        report["status"] = "completed_with_record_failures"
    elif "completed_with_method_failures" in statuses:
        report["status"] = "completed_with_method_failures"
    else:
        report["status"] = "completed"
    _atomic_write(output, report)
    print(f"wrote {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

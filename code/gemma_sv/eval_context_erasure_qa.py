"""Evaluate model-state deletion on a frozen natural-QA RAG manifest.

This module is safe to import without LlamaIndex, datasets, Transformers, or
model weights.  Live dependencies are loaded only by ``main``.  Every method is
queried on the same rehydrated context and the same three targets: the gold
answer, the inserted distractor answer, and an unrelated retained-passage QA.

Admission is intentionally decomposed.  Primary target admission requires a
score-based counterfactual flip before deletion and gold restoration under a
fresh literal full repack.  Retained-QA availability is reported independently
and never changes that primary denominator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
from typing import Any, Mapping, Sequence

from gemma_sv.persistent_deletion import (
    DECAY_FACTOR,
    cache_delete_and_shift,
    first_token_rank,
    full_vocabulary_kl,
    method_storage_report,
    persistent_state_shape_signature,
)
from gemma_sv.rag_benchmark import (
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_TOKENIZER_REVISION,
    RehydratedRecord,
    load_2wiki_split,
    load_manifest,
    rehydrate_manifest_from_rows,
    text_sha256,
    token_ids_sha256,
)


EVALUATION_SCHEMA = "gemma-sv-context-erasure-qa-v1"
EVALUATION_SCHEMA_VERSION = 1
MIN_COUNTERFACTUAL_FLIP_NATS = 0.05
MIN_ANSWER_PREFERENCE_NATS = 0.0
RETAINED_MAX_FIRST_TOKEN_RANK = 10
ICUL_CORRECT_DEMONSTRATIONS = 4
METHOD_IDS = (
    "full_repack",
    "exact_decrement",
    "fixed_c_refit",
    "fp32_proxy",
    "cache_delete_shift",
    "decay_0_01",
    "icul_correction_4",
)


@dataclass(frozen=True)
class QATarget:
    probe_id: str
    kind: str
    prompt: str
    answer: str
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class CorrectionICUL:
    """Query-time correction prefix; it does not mutate the cached model state."""

    text: str
    demonstration_record_ids: tuple[str, ...]
    seed: int


def _score_number(score: Any, key: str = "mean_log_probability") -> float:
    if isinstance(score, Mapping):
        if key == "first_token_rank" and key not in score:
            key = "first_target_token_rank"
        if key not in score:
            raise ValueError(f"score is missing {key}")
        return float(score[key])
    if key != "mean_log_probability":
        raise ValueError(f"scalar score cannot provide {key}")
    return float(score)


def admission_decomposition(
    present: Mapping[str, Any],
    full_repack: Mapping[str, Any],
    *,
    minimum_flip_nats: float = MIN_COUNTERFACTUAL_FLIP_NATS,
    minimum_preference_nats: float = MIN_ANSWER_PREFERENCE_NATS,
    retained_max_first_token_rank: int = RETAINED_MAX_FIRST_TOKEN_RANK,
) -> dict[str, Any]:
    """Compute target admission and retained availability as separate gates.

    ``present`` and ``full_repack`` must contain ``gold`` and ``distractor``
    scores.  Retained entries additionally carry ``first_token_rank``.  The
    target gate is independent of retained availability by construction.
    """

    present_gold = _score_number(present["gold"])
    present_distractor = _score_number(present["distractor"])
    repack_gold = _score_number(full_repack["gold"])
    repack_distractor = _score_number(full_repack["distractor"])
    present_margin = present_distractor - present_gold
    repack_margin = repack_gold - repack_distractor
    flip_strength = present_margin + repack_margin

    distractor_preferred = present_margin >= float(minimum_preference_nats)
    repack_restores_gold = repack_margin >= float(minimum_preference_nats)
    measurable_flip = flip_strength >= float(minimum_flip_nats)
    primary_admitted = (
        distractor_preferred and repack_restores_gold and measurable_flip
    )
    primary_reasons = []
    if not distractor_preferred:
        primary_reasons.append("distractor_not_preferred_pre_deletion")
    if not repack_restores_gold:
        primary_reasons.append("full_repack_does_not_restore_gold")
    if not measurable_flip:
        primary_reasons.append("counterfactual_flip_below_threshold")

    retained_present_rank = int(
        _score_number(present["retained"], "first_token_rank")
    )
    retained_repack_rank = int(
        _score_number(full_repack["retained"], "first_token_rank")
    )
    retained_available = (
        retained_present_rank <= int(retained_max_first_token_rank)
        and retained_repack_rank <= int(retained_max_first_token_rank)
    )
    return {
        "primary_target_admission": {
            "status": "admitted" if primary_admitted else "rejected",
            "admitted": primary_admitted,
            "reasons": primary_reasons,
        },
        "components": {
            "distractor_changes_pre_deletion_answer": {
                "passed": distractor_preferred,
                "distractor_minus_gold_nats": present_margin,
            },
            "full_repack_restores_gold": {
                "passed": repack_restores_gold,
                "gold_minus_distractor_nats": repack_margin,
            },
            "measurable_counterfactual_flip": {
                "passed": measurable_flip,
                "flip_strength_nats": flip_strength,
            },
        },
        "retained_availability": {
            "reported_separately": True,
            "status": "available" if retained_available else "unavailable",
            "available": retained_available,
            "present_first_token_rank": retained_present_rank,
            "full_repack_first_token_rank": retained_repack_rank,
        },
        "joint_target_and_retained": {
            "status": (
                "admitted"
                if primary_admitted and retained_available
                else "rejected"
            ),
            "admitted": primary_admitted and retained_available,
        },
        "thresholds": {
            "minimum_counterfactual_flip_nats": float(minimum_flip_nats),
            "minimum_answer_preference_nats": float(minimum_preference_nats),
            "retained_max_first_token_rank": int(
                retained_max_first_token_rank
            ),
        },
        "measurement": (
            "teacher-forced mean answer log-probability preference; no "
            "generation-based or post-hoc context selection"
        ),
    }


def _stable_icul_key(
    seed: int,
    target_id: str,
    candidate_id: str,
) -> bytes:
    return hashlib.sha256(
        f"rag-icul-correction-v1\0{seed}\0{target_id}\0{candidate_id}".encode(
            "utf-8"
        )
    ).digest()


def build_correction_icul_prefix(
    target: RehydratedRecord,
    all_records: Sequence[RehydratedRecord],
    *,
    seed: int,
    correct_demonstrations: int = ICUL_CORRECT_DEMONSTRATIONS,
) -> CorrectionICUL:
    """Build a faithful correction-style in-context unlearning baseline.

    The frozen contaminated context remains byte-for-byte unchanged.  At query
    time, ICUL supplies one correction pairing the target question with the
    desired gold answer instead of the inserted distractor, followed by four
    deterministically selected correctly answered QA demonstrations.  It is
    explicitly a prompt intervention, not persistent model-state deletion.
    """

    if correct_demonstrations < 1:
        raise ValueError("ICUL requires at least one correct demonstration")
    candidates = [
        record for record in all_records if record.record_id != target.record_id
    ]
    if len(candidates) < correct_demonstrations:
        raise ValueError("not enough distinct records for faithful ICUL")
    demonstrations = sorted(
        candidates,
        key=lambda record: (
            _stable_icul_key(seed, target.record_id, record.record_id),
            record.record_id,
        ),
    )[:correct_demonstrations]
    pieces = [
        "\n\nIn-context corrections:\n",
        f"Question: {target.example.question}\n",
        f"Incorrect answer: {target.counterfactual.distractor_answer}\n",
        f"Correct answer: {target.example.answer}\n",
    ]
    for record in demonstrations:
        pieces.extend(
            (
                f"Question: {record.example.question}\n",
                f"Correct answer: {record.example.answer}\n",
            )
        )
    return CorrectionICUL(
        text="".join(pieces),
        demonstration_record_ids=tuple(
            record.record_id for record in demonstrations
        ),
        seed=int(seed),
    )


def _target_ids(runtime: Any, prompt: str, answer: str) -> tuple[int, ...]:
    rendered = str(answer)
    if prompt and not prompt[-1].isspace() and not rendered[:1].isspace():
        rendered = " " + rendered
    encoded = runtime.tokenizer(rendered, add_special_tokens=False)
    raw_ids = (
        encoded["input_ids"]
        if isinstance(encoded, Mapping)
        else encoded.input_ids
    )
    target_ids = tuple(int(token_id) for token_id in raw_ids)
    if not target_ids:
        raise ValueError("answer tokenization is empty")
    return target_ids


def build_qa_targets(
    runtime: Any,
    record: RehydratedRecord,
) -> tuple[QATarget, ...]:
    """Build the three identical probes used for every deletion method."""

    target_prompt = (
        "\n\nAnswer using the retrieved reference documents.\n"
        f"Question: {record.example.question}\nAnswer:"
    )
    retained_prompt = (
        "\n\nAnswer using the retrieved reference documents.\n"
        f"Question: {record.retained_question}\nAnswer:"
    )
    values = (
        ("gold", "gold_answer", target_prompt, record.example.answer),
        (
            "distractor",
            "distractor_leakage",
            target_prompt,
            record.counterfactual.distractor_answer,
        ),
        (
            "retained",
            "retained_qa",
            retained_prompt,
            record.retained_answer,
        ),
    )
    return tuple(
        QATarget(
            probe_id=probe_id,
            kind=kind,
            prompt=prompt,
            answer=answer,
            target_ids=_target_ids(runtime, prompt, answer),
        )
        for probe_id, kind, prompt, answer in values
    )


def _serialize_score(score: Mapping[str, Any], target: QATarget) -> dict[str, Any]:
    first_log_probs = score["first_log_probs"]
    first_target = int(target.target_ids[0])
    return {
        "target_token_count": len(target.target_ids),
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
    targets: Sequence[QATarget],
    scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        target.probe_id: _serialize_score(scores[target.probe_id], target)
        for target in targets
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(float(value) for value in values) / len(values))


def _behavioral_metrics(
    targets: Sequence[QATarget],
    method_scores: Mapping[str, Mapping[str, Any]],
    present_scores: Mapping[str, Mapping[str, Any]],
    repack_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {target.probe_id: target for target in targets}
    serialized = {
        probe_id: _serialize_score(method_scores[probe_id], by_id[probe_id])
        for probe_id in ("gold", "distractor", "retained")
    }
    kls = {
        probe_id: full_vocabulary_kl(
            repack_scores[probe_id]["first_log_probs"],
            method_scores[probe_id]["first_log_probs"],
        )
        for probe_id in ("gold", "distractor", "retained")
    }
    gold_method = float(method_scores["gold"]["mean_log_probability"])
    gold_present = float(present_scores["gold"]["mean_log_probability"])
    gold_repack = float(repack_scores["gold"]["mean_log_probability"])
    distractor_method = float(
        method_scores["distractor"]["mean_log_probability"]
    )
    distractor_present = float(
        present_scores["distractor"]["mean_log_probability"]
    )
    retained_method = float(
        method_scores["retained"]["mean_log_probability"]
    )
    retained_repack = float(
        repack_scores["retained"]["mean_log_probability"]
    )
    return {
        "gold_answer_recovery": {
            "score": serialized["gold"],
            "lift_vs_present_nats": gold_method - gold_present,
            "drift_from_full_repack_nats": gold_method - gold_repack,
        },
        "distractor_answer_leakage": {
            "score": serialized["distractor"],
            "suppression_vs_present_nats": (
                distractor_present - distractor_method
            ),
            "distractor_minus_gold_nats": distractor_method - gold_method,
        },
        "retained_qa": {
            "score": serialized["retained"],
            "drift_from_full_repack_nats": (
                retained_method - retained_repack
            ),
        },
        "full_vocabulary_kl_to_repack": {
            "direction": "KL(full_repack || method)",
            "distribution_scope": "first answer token",
            "full_vocabulary": True,
            "gold_nats": kls["gold"],
            "distractor_nats": kls["distractor"],
            "retained_nats": kls["retained"],
            "mean_nats": _mean(list(kls.values())),
            "maximum_nats": max(kls.values()),
        },
    }


def _fix_timing_scope(timing: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(timing)
    result["query_scope"] = "gold, distractor, and retained QA probes"
    return result


def _method_report(
    method_id: str,
    state: Any,
    scores: Mapping[str, Mapping[str, Any]],
    timing: Mapping[str, Any],
    *,
    original_memory: Any,
    targets: Sequence[QATarget],
    present_scores: Mapping[str, Mapping[str, Any]],
    repack_scores: Mapping[str, Mapping[str, Any]],
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
            targets,
            scores,
            present_scores,
            repack_scores,
        ),
        "timing": _fix_timing_scope(timing),
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
        "error": str(exc),
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
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


def _baseline_runtime_helpers():
    # Imported only while executing a live/fake runtime evaluation.  Reusing
    # these helpers keeps synchronization and timing semantics matched to the
    # established persistent-deletion evaluator.
    from gemma_sv import eval_persistent_deletion_baselines as baseline

    return baseline


def _baseline_probes(baseline: Any, targets: Sequence[QATarget]) -> list[Any]:
    return [
        baseline.Probe(
            probe_id=target.probe_id,
            kind=target.kind,
            prompt=target.prompt,
            target=target.answer,
            target_ids=target.target_ids,
        )
        for target in targets
    ]


def _solver_certificate(
    targets: Sequence[QATarget],
    exact_scores: Mapping[str, Mapping[str, Any]],
    refit_scores: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        {
            "probe_id": target.probe_id,
            "full_vocabulary_output_kl_nats": full_vocabulary_kl(
                exact_scores[target.probe_id]["first_log_probs"],
                refit_scores[target.probe_id]["first_log_probs"],
            ),
        }
        for target in targets
    ]
    values = [row["full_vocabulary_output_kl_nats"] for row in rows]
    return {
        "status": "completed",
        "distinct_from_behavioral_kl_to_repack": True,
        "direction": "KL(exact_decrement || fixed_c_refit)",
        "probe_rows": rows,
        "mean_nats": _mean(values),
        "maximum_nats": max(values),
        "solver_diagnostics": dict(diagnostics),
    }


def evaluate_admission_record(
    runtime: Any,
    record: RehydratedRecord,
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Score only the frozen present/repack admission pair for one record."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    baseline = _baseline_runtime_helpers()
    record_seed = int(seed) + int(record.record_id.split("-")[-1][:8], 16)
    _seed_everything(record_seed)
    targets = build_qa_targets(runtime, record)
    probes = _baseline_probes(baseline, targets)
    query = lambda state: baseline._score_probe_suite(runtime, state, probes)
    context = record.context

    baseline._synchronize(runtime.config.device)
    import time

    started = time.perf_counter()
    original_memory = runtime.prefill_persistent(
        list(context.original_token_ids)
    )
    baseline._synchronize(runtime.config.device)
    original_prefill_seconds = time.perf_counter() - started
    if str(original_memory.input_digest) != token_ids_sha256(
        context.original_token_ids
    ):
        raise RuntimeError("runtime prefill digest differs from frozen token IDs")

    present_state = baseline.QueryState(
        original_memory,
        update_diagnostics={"reference_kind": "contaminated_present_control"},
    )
    present_scores, present_timing = baseline._benchmark_existing_state(
        state=present_state,
        query=query,
        device=runtime.config.device,
        warmup=warmup,
        repeats=repeats,
    )
    repack_state, repack_scores, repack_timing = baseline._benchmark_method(
        build=lambda: baseline.QueryState(
            runtime.prefill_persistent(list(context.edited_token_ids)),
            update_diagnostics={
                "reference_kind": "fresh_literal_edited_context_prefill",
                "suffix_recomputed": True,
            },
        ),
        query=query,
        device=runtime.config.device,
        warmup=warmup,
        repeats=repeats,
        operation_seed=record_seed,
    )
    present_snapshot = _score_snapshot(targets, present_scores)
    repack_snapshot = _score_snapshot(targets, repack_scores)
    return {
        "record_id": record.record_id,
        "source_example_id": record.example.example_id,
        "record_seed": record_seed,
        "status": "completed",
        "admission_only": True,
        "context": {
            "original_token_count": len(context.original_token_ids),
            "edited_token_count": len(context.edited_token_ids),
            "deleted_token_count": len(context.forget_positions),
            "original_token_ids_sha256": token_ids_sha256(
                context.original_token_ids
            ),
            "edited_token_ids_sha256": token_ids_sha256(
                context.edited_token_ids
            ),
        },
        "admission": admission_decomposition(
            present_snapshot,
            repack_snapshot,
        ),
        "references": {
            "present_control": {
                "scores": present_snapshot,
                "timing": _fix_timing_scope(present_timing),
            },
            "full_repack": {
                "scores": repack_snapshot,
                "timing": _fix_timing_scope(repack_timing),
            },
        },
        "shared_original_prefill": {
            "seconds": original_prefill_seconds,
            "synchronized_device": runtime.config.device,
            "input_digest": str(original_memory.input_digest),
        },
        "methods": {},
    }


def evaluate_record(
    runtime: Any,
    record: RehydratedRecord,
    all_records: Sequence[RehydratedRecord],
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Run every matched deletion baseline for one rehydrated record."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    baseline = _baseline_runtime_helpers()
    record_seed = int(seed) + int(record.record_id.split("-")[-1][:8], 16)
    _seed_everything(record_seed)
    targets = build_qa_targets(runtime, record)
    probes = _baseline_probes(baseline, targets)
    query = lambda state: baseline._score_probe_suite(runtime, state, probes)
    context = record.context
    if len(context.original_token_ids) - len(context.forget_positions) != len(
        context.edited_token_ids
    ):
        raise ValueError("literal edited token sequence has inconsistent length")

    baseline._synchronize(runtime.config.device)
    import time

    prefill_started = time.perf_counter()
    original_memory = runtime.prefill_persistent(
        list(context.original_token_ids)
    )
    baseline._synchronize(runtime.config.device)
    original_prefill_seconds = time.perf_counter() - prefill_started
    if str(original_memory.input_digest) != token_ids_sha256(
        context.original_token_ids
    ):
        raise RuntimeError("runtime prefill digest differs from frozen token IDs")
    source_signature = persistent_state_shape_signature(original_memory)

    row: dict[str, Any] = {
        "record_id": record.record_id,
        "source_example_id": record.example.example_id,
        "record_seed": record_seed,
        "status": "running",
        "context": {
            "original_token_count": len(context.original_token_ids),
            "edited_token_count": len(context.edited_token_ids),
            "deleted_token_count": len(context.forget_positions),
            "forget_positions": list(context.forget_positions),
            "deletion_ranges": [
                {"start": start, "end": end}
                for start, end in context.deletion_ranges
            ],
            "original_token_ids_sha256": token_ids_sha256(
                context.original_token_ids
            ),
            "edited_token_ids_sha256": token_ids_sha256(
                context.edited_token_ids
            ),
            "literal_edit": True,
        },
        "shared_original_prefill": {
            "seconds": original_prefill_seconds,
            "synchronized_device": runtime.config.device,
            "input_digest": str(original_memory.input_digest),
            "storage": method_storage_report(
                original_memory,
                original_memory,
            )["state"],
        },
        "probes": [
            {
                "probe_id": target.probe_id,
                "kind": target.kind,
                "target_token_count": len(target.target_ids),
                "prompt_sha256": text_sha256(target.prompt),
                "target_sha256": text_sha256(target.answer),
            }
            for target in targets
        ],
        "methods": {},
    }

    present_state = baseline.QueryState(
        original_memory,
        update_diagnostics={"reference_kind": "contaminated_present_control"},
    )
    present_scores, present_timing = baseline._benchmark_existing_state(
        state=present_state,
        query=query,
        device=runtime.config.device,
        warmup=warmup,
        repeats=repeats,
    )
    repack_state, repack_scores, repack_timing = baseline._benchmark_method(
        build=lambda: baseline.QueryState(
            runtime.prefill_persistent(list(context.edited_token_ids)),
            update_diagnostics={
                "reference_kind": "fresh_literal_edited_context_prefill",
                "suffix_recomputed": True,
                "solver_refit": "fresh prefill",
            },
        ),
        query=query,
        device=runtime.config.device,
        warmup=warmup,
        repeats=repeats,
        operation_seed=record_seed,
    )
    present_snapshot = _score_snapshot(targets, present_scores)
    repack_snapshot = _score_snapshot(targets, repack_scores)
    row["admission"] = admission_decomposition(
        present_snapshot,
        repack_snapshot,
    )
    row["references"] = {
        "present_control": {
            "status": "completed",
            "scores": present_snapshot,
            "timing": _fix_timing_scope(present_timing),
            **_behavioral_metrics(
                targets,
                present_scores,
                present_scores,
                repack_scores,
            ),
        }
    }
    row["methods"]["full_repack"] = _method_report(
        "full_repack",
        repack_state,
        repack_scores,
        repack_timing,
        original_memory=original_memory,
        targets=targets,
        present_scores=present_scores,
        repack_scores=repack_scores,
        semantics={
            "behavioral_reference": True,
            "fresh_prefill": True,
            "literal_owned_token_edit": True,
            "suffix_recomputed": True,
        },
    )

    try:
        proxy_state, proxy_scores, proxy_timing = baseline._benchmark_method(
            build=lambda: baseline.QueryState(
                runtime.delete_persistent(
                    original_memory,
                    context.forget_positions,
                    kind="fp32_masked_refit_proxy",
                ),
                update_diagnostics={
                    "solver": "feasible projected FP32 FISTA masked-refit proxy",
                    "query_independent": True,
                },
            ),
            query=query,
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            operation_seed=record_seed + 2,
        )
        row["methods"]["fp32_proxy"] = _method_report(
            "fp32_proxy",
            proxy_state,
            proxy_scores,
            proxy_timing,
            original_memory=original_memory,
            targets=targets,
            present_scores=present_scores,
            repack_scores=repack_scores,
            semantics={
                "solver": "feasible projected FP32 FISTA masked refit",
                "query_independent": True,
                "suffix_recomputed": False,
            },
        )
    except Exception as exc:
        row["methods"]["fp32_proxy"] = _failed_method("fp32_proxy", exc)

    try:
        def build_cache_state():
            memory, diagnostics = cache_delete_and_shift(
                original_memory,
                context.forget_positions,
                edited_token_ids=context.edited_token_ids,
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
            operation_seed=record_seed + 3,
        )
        row["methods"]["cache_delete_shift"] = _method_report(
            "cache_delete_shift",
            cache_state,
            cache_scores,
            cache_timing,
            original_memory=original_memory,
            targets=targets,
            present_scores=present_scores,
            repack_scores=repack_scores,
            semantics={
                "diagnostic_only": True,
                "physically_deletes_cached_rows": True,
                "solver_refit": False,
                "suffix_recomputed": False,
                "rope_keys_rerotated": False,
            },
        )
    except Exception as exc:
        row["methods"]["cache_delete_shift"] = _failed_method(
            "cache_delete_shift",
            exc,
        )

    try:
        from gemma_sv.demo_server.gate_context import GateRequest

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
        row["methods"]["decay_0_01"] = _method_report(
            "decay_0_01",
            decay_state,
            decay_scores,
            decay_timing,
            original_memory=original_memory,
            targets=targets,
            present_scores=present_scores,
            repack_scores=repack_scores,
            semantics={
                "decay_factor": DECAY_FACTOR,
                "positions_remain_resident": True,
                "solver_refit": False,
                "suffix_recomputed": False,
            },
        )
    except Exception as exc:
        row["methods"]["decay_0_01"] = _failed_method("decay_0_01", exc)

    try:
        correction = build_correction_icul_prefix(
            record,
            all_records,
            seed=record_seed,
        )
        prefix_encoded = runtime.tokenizer(
            correction.text,
            add_special_tokens=False,
        )
        prefix_ids = (
            prefix_encoded["input_ids"]
            if isinstance(prefix_encoded, Mapping)
            else prefix_encoded.input_ids
        )

        def build_icul_state():
            return baseline.QueryState(
                original_memory,
                prompt_prefix=correction.text,
                prompt_token_count=len(prefix_ids),
                update_diagnostics={
                    "faithful_correction_style_icul": True,
                    "persistent_model_state_deleted": False,
                    "base_packaged_context_unchanged": True,
                    "target_correction": "distractor answer to gold answer",
                    "correct_demonstrations": len(
                        correction.demonstration_record_ids
                    ),
                    "demonstration_record_ids": list(
                        correction.demonstration_record_ids
                    ),
                    "prompt_token_ids_sha256": token_ids_sha256(prefix_ids),
                },
            )

        icul_state, icul_scores, icul_timing = baseline._benchmark_method(
            build=build_icul_state,
            query=query,
            device=runtime.config.device,
            warmup=warmup,
            repeats=repeats,
            operation_seed=record_seed + 5,
        )
        row["methods"]["icul_correction_4"] = _method_report(
            "icul_correction_4",
            icul_state,
            icul_scores,
            icul_timing,
            original_memory=original_memory,
            targets=targets,
            present_scores=present_scores,
            repack_scores=repack_scores,
            semantics={
                "correction_style_in_context_unlearning": True,
                "correct_retained_demonstrations": (
                    ICUL_CORRECT_DEMONSTRATIONS
                ),
                "identical_base_packaged_context": True,
                "adds_query_time_context": True,
                "persistent_model_state_deleted": False,
                "temperature": 0,
            },
        )
    except Exception as exc:
        row["methods"]["icul_correction_4"] = _failed_method(
            "icul_correction_4",
            exc,
        )

    try:
        (
            certificate_states,
            certificate_scores,
            certificate_timings,
            solver_diagnostics,
        ) = baseline._benchmark_certificate_pair(
            runtime,
            original_memory,
            context.forget_positions,
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
            targets=targets,
            present_scores=present_scores,
            repack_scores=repack_scores,
            semantics={
                "gate_solver": "float64 Cauwenberghs-Poggio decrement",
                "fixed_C": True,
                "query_independent": True,
                "suffix_recomputed": False,
            },
        )
        row["methods"]["fixed_c_refit"] = _method_report(
            "fixed_c_refit",
            certificate_states["refit"],
            certificate_scores["refit"],
            certificate_timings["refit"],
            original_memory=original_memory,
            targets=targets,
            present_scores=present_scores,
            repack_scores=repack_scores,
            semantics={
                "gate_solver": "float64 retained-key from-scratch refit",
                "fixed_C": True,
                "query_independent": True,
                "suffix_recomputed": False,
            },
        )
        row["solver_certificate"] = _solver_certificate(
            targets,
            certificate_scores["exact"],
            certificate_scores["refit"],
            solver_diagnostics,
        )
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
            "distinct_from_behavioral_kl_to_repack": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    source_unchanged = (
        persistent_state_shape_signature(original_memory) == source_signature
        and str(original_memory.input_digest)
        == token_ids_sha256(context.original_token_ids)
    )
    row["source_state_immutability"] = {
        "shape_signature_and_digest_unchanged": source_unchanged,
    }
    if not source_unchanged:
        raise RuntimeError("a deletion method mutated the shared source state")
    row["method_failures"] = [
        method_id
        for method_id in METHOD_IDS
        if row["methods"].get(method_id, {}).get("status") == "failed"
    ]
    row["status"] = (
        "completed"
        if not row["method_failures"]
        else "completed_with_method_failures"
    )
    return row


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate every predeclared record without admission filtering."""

    completed = [
        record for record in records if record.get("status") != "failed"
    ]
    summary: dict[str, Any] = {
        "aggregation_population": (
            "all predeclared manifest records; admission is reported, not used "
            "for post-hoc selection"
        ),
        "attempted_records": len(records),
        "completed_records": len(completed),
        "primary_target_admitted": sum(
            record["admission"]["primary_target_admission"]["admitted"]
            for record in completed
        ),
        "retained_available": sum(
            record["admission"]["retained_availability"]["available"]
            for record in completed
        ),
        "joint_target_and_retained": sum(
            record["admission"]["joint_target_and_retained"]["admitted"]
            for record in completed
        ),
        "methods": {},
    }
    for method_id in METHOD_IDS:
        method_rows = [
            record["methods"][method_id]
            for record in completed
            if record.get("methods", {}).get(method_id, {}).get("status")
            == "completed"
        ]
        method_summary: dict[str, Any] = {
            "completed_records": len(method_rows),
            "failed_records": len(completed) - len(method_rows),
        }
        if method_rows:
            method_summary.update(
                {
                    "mean_gold_lift_vs_present_nats": _mean(
                        [
                            row["gold_answer_recovery"][
                                "lift_vs_present_nats"
                            ]
                            for row in method_rows
                        ]
                    ),
                    "mean_distractor_suppression_vs_present_nats": _mean(
                        [
                            row["distractor_answer_leakage"][
                                "suppression_vs_present_nats"
                            ]
                            for row in method_rows
                        ]
                    ),
                    "mean_retained_drift_from_repack_nats": _mean(
                        [
                            row["retained_qa"][
                                "drift_from_full_repack_nats"
                            ]
                            for row in method_rows
                        ]
                    ),
                    "mean_full_vocabulary_kl_to_repack_nats": _mean(
                        [
                            row["full_vocabulary_kl_to_repack"]["mean_nats"]
                            for row in method_rows
                        ]
                    ),
                    "mean_update_median_seconds": _mean(
                        [
                            row["timing"]["update_seconds"]["median"]
                            for row in method_rows
                        ]
                    ),
                    "mean_query_median_seconds": _mean(
                        [
                            row["timing"]["query_seconds"]["median"]
                            for row in method_rows
                        ]
                    ),
                    "mean_end_to_end_median_seconds": _mean(
                        [
                            row["timing"]["end_to_end_seconds"]["median"]
                            for row in method_rows
                        ]
                    ),
                    "mean_state_tensor_storage_bytes": _mean(
                        [
                            row["storage"]["state"][
                                "deduplicated_tensor_storage_bytes"
                            ]
                            for row in method_rows
                        ]
                    ),
                    "mean_incremental_tensor_storage_bytes": _mean(
                        [
                            row["storage"]["incremental_tensor_storage_bytes"]
                            for row in method_rows
                        ]
                    ),
                    "mean_prompt_token_count": _mean(
                        [
                            row["storage"]["prompt_token_count"]
                            for row in method_rows
                        ]
                    ),
                }
            )
        summary["methods"][method_id] = method_summary
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


def _environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "transformers", "llama_index"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {"platform": platform.platform(), "versions": versions}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--model-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument(
        "--adapter",
        default="outputs/gemma_sv_distill/lora_adapter",
        help="LoRA adapter path; use 'none' for no adapter",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--records", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--admission-only",
        action="store_true",
        help="score present/full-repack gates only; skip deletion methods",
    )
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_rag/context_erasure_qa.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Rehydrate the frozen manifest and run live Gemma deletion methods."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.record_start < 0 or (args.records is not None and args.records < 1):
        parser.error("record selection must be non-negative and non-empty")
    if args.warmup < 0 or args.repeats < 1 or args.window < 1:
        parser.error("warmup/window must be non-negative and repeats positive")
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")

    manifest = load_manifest(args.manifest)
    if manifest["dataset"] != {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": manifest["dataset"]["split"],
    }:
        parser.error("manifest does not use the pinned 2Wiki dataset")
    frozen_tokenizer = manifest["tokenizer"]
    if (
        args.model != frozen_tokenizer["model_id"]
        or args.model_revision != frozen_tokenizer["revision"]
    ):
        parser.error(
            "--model and --model-revision must match the tokenizer frozen in "
            "the manifest so offset-based token ownership remains exact"
        )
    frozen_deletion_window = int(
        manifest["erasure_config"]["minimum_tokens_after_owned"]
    )
    if args.window > frozen_deletion_window:
        parser.error(
            "--window exceeds the manifest's frozen minimum distance after "
            "the owned span"
        )

    from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig

    adapter = (
        None
        if str(args.adapter).casefold() in {"none", "null", "-"}
        else args.adapter
    )
    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=adapter,
            device=args.device,
            dtype="float32",
            generation_tokens=1,
            window=args.window,
            copies=1,
            model_revision=args.model_revision,
        )
    )
    runtime.ensure_loaded()
    rows = load_2wiki_split(
        split=manifest["dataset"]["split"],
        revision=manifest["dataset"]["revision"],
    )
    rehydrated = rehydrate_manifest_from_rows(
        manifest,
        rows,
        runtime.tokenizer,
    )
    selected = rehydrated[args.record_start :]
    if args.records is not None:
        selected = selected[: args.records]
    if not selected:
        parser.error("record selection is empty")

    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation": "frozen 2Wiki counterfactual context erasure QA",
        "status": "running",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "manifest": {
            "path": str(args.manifest),
            "integrity_sha256": manifest["integrity"]["sha256"],
            "fixed_before_model_evaluation": True,
            "gemma_outputs_used_for_selection": False,
            "total_records": len(rehydrated),
            "selected_record_ids": [record.record_id for record in selected],
        },
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter": adapter,
            "device": args.device,
            "dtype": "float32",
            "window": args.window,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "admission_only": args.admission_only,
            "decay_factor": DECAY_FACTOR,
            "icul_correct_demonstrations": ICUL_CORRECT_DEMONSTRATIONS,
        },
        "metric_definitions": {
            "gold_recovery": (
                "method gold-answer mean log-probability lift from the "
                "contaminated present control"
            ),
            "distractor_leakage": (
                "distractor answer score and suppression from the contaminated "
                "present control"
            ),
            "retained_qa": (
                "retained answer score and drift from literal full repack"
            ),
            "output_kl": (
                "full-vocabulary KL(full_repack || method) at each probe's "
                "first answer token"
            ),
            "storage": (
                "deduplicated underlying Torch/NumPy tensor storage, with "
                "shared views and references counted once"
            ),
        },
        "technical_scope": [
            (
                "Exact decrement and fixed-C refit alter persistent SV gate "
                "state without replaying source tokens."
            ),
            (
                "Cache delete/shift is diagnostic: it does not recompute "
                "contaminated suffix states or rerotate RoPE keys."
            ),
            (
                "Correction-style ICUL keeps the identical packaged base "
                "context and adds a query-time correction prefix; it is not "
                "persistent deletion."
            ),
            (
                "All update and query latencies use synchronized device timing "
                "from the matched persistent-deletion evaluator."
            ),
        ],
        "environment": _environment(),
        "records": [],
    }
    _atomic_write(output, report)
    for index, record in enumerate(selected):
        print(
            f"[{index + 1}/{len(selected)}] {record.record_id}",
            flush=True,
        )
        try:
            if args.admission_only:
                result = evaluate_admission_record(
                    runtime,
                    record,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            else:
                result = evaluate_record(
                    runtime,
                    record,
                    rehydrated,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
        except Exception as exc:
            result = {
                "record_id": record.record_id,
                "source_example_id": record.example.example_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        report["records"].append(result)
        _atomic_write(output, report)
    report["summary"] = summarize_records(report["records"])
    report["status"] = (
        "completed"
        if report["summary"]["completed_records"] == len(selected)
        else "completed_with_record_failures"
    )
    _atomic_write(output, report)
    print(f"wrote evaluation to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

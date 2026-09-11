"""Evaluate the frozen LongMemEval knowledge-update deletion adaptation.

The target gate intentionally uses no inferred previous answer.  It requires
the official current answer to have at least 0.05 nats higher teacher-forced
mean log probability in the present state than in a literal full repack with
the latest evidence session removed, and present first-token rank at most 10.
An unrelated official retained QA rank gate is reported separately.
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
import re
from typing import Any, Mapping, Sequence

from gemma_sv import eval_context_erasure_qa as shared_eval
from gemma_sv.eval_context_erasure_qa import (
    ICUL_CORRECT_DEMONSTRATIONS,
    METHOD_IDS,
    QATarget,
)
from gemma_sv.longmemeval_deletion_benchmark import (
    BENCHMARK_LABEL,
    DATASET_ARTIFACT_SHA256,
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_GREEDY_TOKEN_RESERVE,
    DEFAULT_MINIMUM_TOKENS_AFTER_OWNED,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TOKENIZER_REVISION,
    LONGMEMEVAL_REPOSITORY_REVISION,
    RUNTIME_CONTEXT_CEILING,
    RehydratedLongMemEvalRecord,
    load_manifest,
    load_pinned_longmemeval_rows,
    longmemeval_query_prompt,
    rehydrate_manifest_from_rows,
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


EVALUATION_SCHEMA = "gemma-sv-longmemeval-deletion-evaluation-v1"
EVALUATION_SCHEMA_VERSION = 1
MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS = 0.05
MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK = 10
RETAINED_MAX_FIRST_TOKEN_RANK = 10
DEFAULT_GREEDY_TOKENS = DEFAULT_GREEDY_TOKEN_RESERVE


@dataclass(frozen=True)
class DeletionICUL:
    """Prompt-only deletion directive with four official QA demonstrations."""

    text: str
    demonstration_record_ids: tuple[str, ...]
    seed: int


def _score_number(score: Any, key: str = "mean_log_probability") -> float:
    if not isinstance(score, Mapping):
        if key != "mean_log_probability":
            raise ValueError(f"scalar score cannot provide {key}")
        return float(score)
    if key == "first_token_rank" and key not in score:
        key = "first_target_token_rank"
    if key not in score:
        raise ValueError(f"score is missing {key}")
    return float(score[key])


def strict_admission_decomposition(
    present: Mapping[str, Any],
    full_repack: Mapping[str, Any],
    *,
    minimum_current_lift_nats: float = (
        MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS
    ),
    maximum_current_present_first_token_rank: int = (
        MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK
    ),
    retained_max_first_token_rank: int = RETAINED_MAX_FIRST_TOKEN_RANK,
) -> dict[str, Any]:
    """Apply the predeclared current-answer signal and retained gates."""

    present_current = _score_number(present["current"])
    repack_current = _score_number(full_repack["current"])
    current_lift = present_current - repack_current
    current_present_rank = int(
        _score_number(present["current"], "first_token_rank")
    )
    lift_passed = current_lift >= float(minimum_current_lift_nats)
    rank_passed = current_present_rank <= int(
        maximum_current_present_first_token_rank
    )
    target_admitted = lift_passed and rank_passed
    reasons = []
    if not lift_passed:
        reasons.append("current_answer_stored_signal_lift_below_0_05_nats")
    if not rank_passed:
        reasons.append("current_answer_present_first_token_rank_above_10")

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
        "scheme": "longmemeval_current_answer_stored_signal_v1",
        "primary_target_admission": {
            "status": "admitted" if target_admitted else "rejected",
            "admitted": target_admitted,
            "reasons": reasons,
        },
        "target_stored_signal": {
            "reported_separately_from_retained_qa": True,
            "current_answer_present_minus_full_repack_nats": current_lift,
            "current_answer_present_first_token_rank": current_present_rank,
            "lift_component_passed": lift_passed,
            "rank_component_passed": rank_passed,
            "prior_answer_extracted": False,
            "prior_answer_invented": False,
            "prior_answer_used_for_gate": False,
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
                if target_admitted and retained_available
                else "rejected"
            ),
            "admitted": target_admitted and retained_available,
        },
        "thresholds": {
            "minimum_current_answer_present_minus_full_repack_nats": float(
                minimum_current_lift_nats
            ),
            "maximum_current_answer_present_first_token_rank": int(
                maximum_current_present_first_token_rank
            ),
            "retained_max_first_token_rank": int(
                retained_max_first_token_rank
            ),
        },
        "measurement": (
            "teacher-forced official current-answer mean log probability and "
            "first-token rank; no heuristic or LLM-extracted previous answer"
        ),
    }


def build_qa_targets(
    runtime: Any,
    record: RehydratedLongMemEvalRecord,
) -> tuple[QATarget, ...]:
    """Build only official current-answer and distinct retained QA probes."""

    values = (
        (
            "current",
            "official_current_answer",
            longmemeval_query_prompt(record.target),
            record.target.answer,
        ),
        (
            "retained",
            "retained_official_qa",
            longmemeval_query_prompt(record.retained),
            record.retained.answer,
        ),
    )
    return tuple(
        QATarget(
            probe_id=probe_id,
            kind=kind,
            prompt=prompt,
            answer=answer,
            target_ids=shared_eval._target_ids(runtime, prompt, answer),
        )
        for probe_id, kind, prompt, answer in values
    )


def _serialize_score(
    score: Mapping[str, Any],
    target: QATarget,
) -> dict[str, Any]:
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
        for probe_id in ("current", "retained")
    }
    kls = {
        probe_id: full_vocabulary_kl(
            repack_scores[probe_id]["first_log_probs"],
            method_scores[probe_id]["first_log_probs"],
        )
        for probe_id in ("current", "retained")
    }
    current_method = float(
        method_scores["current"]["mean_log_probability"]
    )
    current_present = float(
        present_scores["current"]["mean_log_probability"]
    )
    current_repack = float(
        repack_scores["current"]["mean_log_probability"]
    )
    retained_method = float(
        method_scores["retained"]["mean_log_probability"]
    )
    retained_repack = float(
        repack_scores["retained"]["mean_log_probability"]
    )
    return {
        "current_answer_behavior": {
            "score": serialized["current"],
            "lift_vs_present_nats": current_method - current_present,
            "drift_from_full_repack_nats": current_method - current_repack,
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
            "current_answer_nats": kls["current"],
            "retained_nats": kls["retained"],
            "mean_nats": _mean(list(kls.values())),
            "maximum_nats": max(kls.values()),
        },
    }


def _fix_timing_scope(timing: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(timing)
    result["query_scope"] = "official current-answer and retained-QA probes"
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
    rendered = str(exc)
    return {
        "method_id": method_id,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": "redacted; compare error_sha256",
        "error_sha256": text_sha256(rendered),
    }


def _normalized_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _standard_present_behavior(
    runtime: Any,
    memory: Any,
    target: QATarget,
    teacher_forced: Mapping[str, Any],
    *,
    greedy_tokens: int,
    operation_seed: int,
) -> dict[str, Any]:
    """Report deterministic current-answer QA without leaking generated text."""

    result: dict[str, Any] = {
        "official_question_id_preserved": True,
        "official_question_type_preserved": True,
        "teacher_forced_current_answer": dict(teacher_forced),
        "generation": {
            "decoding": "deterministic greedy temperature=0",
            "requested_tokens": int(greedy_tokens),
            "used_for_admission": False,
            "used_for_selection_or_replacement": False,
            "generated_text_in_report": False,
        },
        "leaderboard_metric": False,
        "note": (
            "context-deletion adaptation diagnostic; official LongMemEval "
            "leaderboard evaluation normally uses its released answer checker"
        ),
    }
    generator = getattr(runtime, "generate_persistent", None)
    if not callable(generator):
        result["generation"].update(
            {
                "status": "unavailable",
                "reason": "runtime_has_no_generate_persistent",
                "exact_current_answer_phrase": None,
            }
        )
        return result
    shared_eval._seed_everything(operation_seed)
    try:
        generated = generator(
            memory,
            target.prompt,
            n_tokens=int(greedy_tokens),
        )
    except Exception as exc:
        result["generation"].update(
            {
                "status": "unavailable",
                "reason": "generation_failed",
                "error_type": type(exc).__name__,
                "error": "redacted; compare error_sha256",
                "error_sha256": text_sha256(str(exc)),
                "exact_current_answer_phrase": None,
            }
        )
        return result
    text = generated[0] if isinstance(generated, tuple) else generated
    text = str(text)
    result["generation"].update(
        {
            "status": "completed",
            "generated_text_sha256": text_sha256(text),
            "generated_character_count": len(text),
            "exact_current_answer_phrase": (
                _normalized_phrase(target.answer) in _normalized_phrase(text)
            ),
            "exact_phrase_definition": (
                "case-insensitive substring after whitespace normalization"
            ),
        }
    )
    return result


def _stable_icul_key(
    seed: int,
    target_id: str,
    candidate_id: str,
) -> bytes:
    return hashlib.sha256(
        (
            "longmemeval-deletion-icul-v1\0"
            f"{int(seed)}\0{target_id}\0{candidate_id}"
        ).encode("utf-8")
    ).digest()


def build_deletion_icul_prefix(
    target: RehydratedLongMemEvalRecord,
    all_records: Sequence[RehydratedLongMemEvalRecord],
    *,
    seed: int,
    correct_demonstrations: int = ICUL_CORRECT_DEMONSTRATIONS,
) -> DeletionICUL:
    """Build a no-invented-prior-answer prompt-only ICUL adaptation."""

    candidates = [
        record for record in all_records if record.record_id != target.record_id
    ]
    if len(candidates) < correct_demonstrations:
        raise ValueError("not enough distinct records for ICUL demonstrations")
    demonstrations = sorted(
        candidates,
        key=lambda record: (
            _stable_icul_key(seed, target.record_id, record.record_id),
            record.record_id,
        ),
    )[:correct_demonstrations]
    pieces = [
        "\n\nIn-context deletion directive:\n",
        "The latest update session for the next question has been deleted. "
        "Answer using only the retained history; do not reconstruct deleted "
        "content.\n",
        f"Question: {target.target.question}\n",
        "No previous answer is supplied or inferred.\n",
        "Correct official QA demonstrations:\n",
    ]
    for record in demonstrations:
        pieces.extend(
            (
                f"Question: {record.retained.question}\n",
                f"Answer: {record.retained.answer}\n",
            )
        )
    return DeletionICUL(
        text="".join(pieces),
        demonstration_record_ids=tuple(
            record.record_id for record in demonstrations
        ),
        seed=int(seed),
    )


def _common_setup(
    runtime: Any,
    record: RehydratedLongMemEvalRecord,
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> tuple[Any, int, tuple[QATarget, ...], list[Any], Any]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    baseline = shared_eval._baseline_runtime_helpers()
    record_seed = int(seed) + int(record.record_id.split("-")[-1][:8], 16)
    shared_eval._seed_everything(record_seed)
    targets = build_qa_targets(runtime, record)
    probes = shared_eval._baseline_probes(baseline, targets)
    context = record.context
    if len(context.original_token_ids) - len(context.forget_positions) != len(
        context.edited_token_ids
    ):
        raise ValueError("literal edited token sequence has inconsistent length")
    return baseline, record_seed, targets, probes, context


def evaluate_admission_record(
    runtime: Any,
    record: RehydratedLongMemEvalRecord,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    greedy_tokens: int = DEFAULT_GREEDY_TOKENS,
) -> dict[str, Any]:
    """Run only present/full-repack controls and the strict stored-signal gate."""

    if greedy_tokens < 1:
        raise ValueError("greedy_tokens must be positive")
    baseline, record_seed, targets, probes, context = _common_setup(
        runtime,
        record,
        seed=seed,
        warmup=warmup,
        repeats=repeats,
    )
    query = lambda state: baseline._score_probe_suite(runtime, state, probes)
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
    source_signature = persistent_state_shape_signature(original_memory)

    present_state = baseline.QueryState(
        original_memory,
        update_diagnostics={"reference_kind": "latest_update_present_control"},
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
                "reference_kind": (
                    "fresh_literal_repack_without_latest_evidence_session"
                ),
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
    behavior = _standard_present_behavior(
        runtime,
        original_memory,
        next(target for target in targets if target.probe_id == "current"),
        present_snapshot["current"],
        greedy_tokens=greedy_tokens,
        operation_seed=record_seed,
    )
    source_unchanged = (
        persistent_state_shape_signature(original_memory) == source_signature
        and str(original_memory.input_digest)
        == token_ids_sha256(context.original_token_ids)
    )
    if not source_unchanged:
        raise RuntimeError("admission controls mutated the source state")
    return {
        "record_id": record.record_id,
        "source_example_id": record.target.question_id,
        "official_comparison": {
            "question_id": record.target.question_id,
            "question_type": record.target.question_type,
            "retained_question_id": record.retained.question_id,
            "context_deletion_adaptation": True,
            "official_longmemeval_leaderboard_score": False,
        },
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
        "admission": strict_admission_decomposition(
            present_snapshot,
            repack_snapshot,
        ),
        "standard_present_current_answer_behavior": behavior,
        "references": {
            "present_control": {
                "scores": present_snapshot,
                "timing": _fix_timing_scope(present_timing),
            },
            "full_repack": {
                "scores": repack_snapshot,
                "timing": _fix_timing_scope(repack_timing),
                "prefill_input_digest": str(repack_state.memory.input_digest),
                "latest_official_evidence_session_removed": True,
            },
        },
        "shared_original_prefill": {
            "seconds": original_prefill_seconds,
            "synchronized_device": runtime.config.device,
            "input_digest": str(original_memory.input_digest),
        },
        "source_state_immutability": {
            "shape_signature_and_digest_unchanged": True,
        },
        "methods": {},
    }


def evaluate_record(
    runtime: Any,
    record: RehydratedLongMemEvalRecord,
    all_records: Sequence[RehydratedLongMemEvalRecord],
    *,
    seed: int,
    warmup: int,
    repeats: int,
    greedy_tokens: int = DEFAULT_GREEDY_TOKENS,
) -> dict[str, Any]:
    """Run all seven matched deletion methods on the two official QA probes."""

    if greedy_tokens < 1:
        raise ValueError("greedy_tokens must be positive")
    baseline, record_seed, targets, probes, context = _common_setup(
        runtime,
        record,
        seed=seed,
        warmup=warmup,
        repeats=repeats,
    )
    query = lambda state: baseline._score_probe_suite(runtime, state, probes)
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
        "source_example_id": record.target.question_id,
        "official_comparison": {
            "question_id": record.target.question_id,
            "question_type": record.target.question_type,
            "retained_question_id": record.retained.question_id,
            "context_deletion_adaptation": True,
            "official_longmemeval_leaderboard_score": False,
        },
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
        update_diagnostics={"reference_kind": "latest_update_present_control"},
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
                "reference_kind": (
                    "fresh_literal_repack_without_latest_evidence_session"
                ),
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
    row["admission"] = strict_admission_decomposition(
        present_snapshot,
        repack_snapshot,
    )
    row["standard_present_current_answer_behavior"] = (
        _standard_present_behavior(
            runtime,
            original_memory,
            next(
                target
                for target in targets
                if target.probe_id == "current"
            ),
            present_snapshot["current"],
            greedy_tokens=greedy_tokens,
            operation_seed=record_seed,
        )
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
        },
        "full_repack": {
            "status": "completed",
            "scores": repack_snapshot,
            "timing": _fix_timing_scope(repack_timing),
            "latest_official_evidence_session_removed": True,
        },
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
            "literal_whole_latest_session_edit": True,
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
                    "solver": (
                        "feasible projected FP32 FISTA masked-refit proxy"
                    ),
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
        correction = build_deletion_icul_prefix(
            record,
            all_records,
            seed=record_seed,
        )
        prefix_ids = shared_eval._target_ids(runtime, "", correction.text)

        def build_icul_state():
            return baseline.QueryState(
                original_memory,
                prompt_prefix=correction.text,
                prompt_token_count=len(prefix_ids),
                update_diagnostics={
                    "prompt_only_deletion_directive": True,
                    "persistent_model_state_deleted": False,
                    "base_packaged_context_unchanged": True,
                    "prior_answer_supplied": False,
                    "prior_answer_invented": False,
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
                "longmemeval_no_prior_answer_adaptation": True,
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
        row["solver_certificate"] = shared_eval._solver_certificate(
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
            "error": "redacted; compare error_sha256",
            "error_sha256": text_sha256(str(exc)),
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


def summarize_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate every frozen record without admission filtering."""

    completed = [
        record for record in records if record.get("status") != "failed"
    ]
    summary: dict[str, Any] = {
        "aggregation_population": (
            "all frozen manifest records; admission never causes replacement "
            "or removal"
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
    method_population = [
        record for record in completed if record.get("admission_only") is not True
    ]
    for method_id in METHOD_IDS:
        rows = [
            record["methods"][method_id]
            for record in method_population
            if record.get("methods", {}).get(method_id, {}).get("status")
            == "completed"
        ]
        method_summary: dict[str, Any] = {
            "completed_records": len(rows),
            "failed_records": len(method_population) - len(rows),
            "not_run_admission_only_records": (
                len(completed) - len(method_population)
            ),
        }
        if rows:
            method_summary.update(
                {
                    "mean_current_answer_lift_vs_present_nats": _mean(
                        [
                            row["current_answer_behavior"][
                                "lift_vs_present_nats"
                            ]
                            for row in rows
                        ]
                    ),
                    "mean_retained_drift_from_repack_nats": _mean(
                        [
                            row["retained_qa"][
                                "drift_from_full_repack_nats"
                            ]
                            for row in rows
                        ]
                    ),
                    "mean_full_vocabulary_kl_to_repack_nats": _mean(
                        [
                            row["full_vocabulary_kl_to_repack"]["mean_nats"]
                            for row in rows
                        ]
                    ),
                    "mean_update_median_seconds": _mean(
                        [
                            row["timing"]["update_seconds"]["median"]
                            for row in rows
                        ]
                    ),
                    "mean_query_median_seconds": _mean(
                        [
                            row["timing"]["query_seconds"]["median"]
                            for row in rows
                        ]
                    ),
                    "mean_end_to_end_median_seconds": _mean(
                        [
                            row["timing"]["end_to_end_seconds"]["median"]
                            for row in rows
                        ]
                    ),
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
                            row["storage"][
                                "incremental_tensor_storage_bytes"
                            ]
                            for row in rows
                        ]
                    ),
                    "mean_prompt_token_count": _mean(
                        [
                            row["storage"]["prompt_token_count"]
                            for row in rows
                        ]
                    ),
                }
            )
        summary["methods"][method_id] = method_summary
    return summary


def verify_tokenizer_revision(
    manifest: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
) -> None:
    frozen = manifest.get("tokenizer") or {}
    if (
        str(model_id) != str(frozen.get("model_id"))
        or str(model_revision) != str(frozen.get("revision"))
    ):
        raise ValueError(
            "runtime model/tokenizer ID and revision must match frozen offsets"
        )


def verify_loaded_tokenizer_revision(
    runtime: Any,
    manifest: Mapping[str, Any],
) -> None:
    expected = str(manifest["tokenizer"]["revision"])
    resolved = getattr(runtime, "resolved_model_revision", None)
    if str(resolved) != expected:
        raise RuntimeError(
            "loaded runtime revision differs from frozen tokenizer revision"
        )
    tokenizer = getattr(runtime, "tokenizer", None)
    kwargs = getattr(tokenizer, "init_kwargs", {})
    observed = kwargs.get("_commit_hash") if isinstance(kwargs, Mapping) else None
    if observed is not None and str(observed) != expected:
        raise RuntimeError(
            "loaded tokenizer commit differs from frozen tokenizer revision"
        )


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
    for package in ("numpy", "torch", "transformers", "mlx"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {"platform": platform.platform(), "versions": versions}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--data-path",
        help="local exact oracle JSON; omitted resolves the pinned Hub revision",
    )
    parser.add_argument("--model", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_TOKENIZER_REVISION,
    )
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
    parser.add_argument("--greedy-tokens", type=int, default=DEFAULT_GREEDY_TOKENS)
    parser.add_argument("--query-gate-floor", type=float, default=0.0)
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument(
        "--out",
        default=(
            "outputs/gemma_sv_rag/"
            "longmemeval_knowledge_update_deletion_evaluation.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Hydrate exact source rows and evaluate without retrieval or replacement."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.record_start < 0 or (args.records is not None and args.records < 1):
        parser.error("record selection must be non-negative and non-empty")
    if (
        args.warmup < 0
        or args.repeats < 1
        or args.window < 1
        or args.greedy_tokens < 1
        or not 0.0 <= args.query_gate_floor <= 1.0
    ):
        parser.error("warmup/window must be non-negative and repeats positive")
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")

    manifest = load_manifest(args.manifest)
    try:
        verify_tokenizer_revision(
            manifest,
            model_id=args.model,
            model_revision=args.model_revision,
        )
    except ValueError as exc:
        parser.error(str(exc))
    erasure = manifest["erasure_config"]
    frozen_distance = int(erasure["minimum_tokens_strictly_after_owned"])
    frozen_greedy_reserve = int(erasure["greedy_generation_token_reserve"])
    if frozen_distance < DEFAULT_MINIMUM_TOKENS_AFTER_OWNED:
        parser.error("manifest does not satisfy the 512-token deletion protocol")
    if args.window > frozen_distance:
        parser.error("--window exceeds frozen post-owned token distance")
    if args.greedy_tokens > frozen_greedy_reserve:
        parser.error("--greedy-tokens exceeds the frozen query token reserve")
    if int(erasure["runtime_context_ceiling"]) > RUNTIME_CONTEXT_CEILING:
        parser.error("manifest exceeds the runtime 2,048-token ceiling")

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
            generation_tokens=args.greedy_tokens,
            window=args.window,
            copies=1,
            model_revision=args.model_revision,
            query_gate_floor=args.query_gate_floor,
        )
    )
    runtime.ensure_loaded()
    verify_loaded_tokenizer_revision(runtime, manifest)
    rows = load_pinned_longmemeval_rows(args.data_path)
    records = rehydrate_manifest_from_rows(
        manifest,
        rows,
        runtime.tokenizer,
    )
    selected = records[args.record_start :]
    if args.records is not None:
        selected = selected[: args.records]
    if not selected:
        parser.error("record selection is empty")

    policy = manifest["selection_policy"]
    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark_label": BENCHMARK_LABEL,
        "evaluation": (
            "official cleaned LongMemEval knowledge-update context deletion"
        ),
        "official_longmemeval_leaderboard_score": False,
        "context_deletion_adaptation": True,
        "status": "running",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "manifest": {
            "path": str(args.manifest),
            "integrity_sha256": manifest["integrity"]["sha256"],
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source_artifact_sha256": DATASET_ARTIFACT_SHA256,
            "official_repository_revision": LONGMEMEVAL_REPOSITORY_REVISION,
            "fixed_before_model_scoring": True,
            "gemma_outputs_used_for_selection": False,
            "output_based_replacement": False,
            "all_records_retained": True,
            "data_only_rejection_counts": policy[
                "data_only_rejection_counts"
            ],
            "selected_record_ids": [record.record_id for record in selected],
        },
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision_verified_against_manifest": True,
            "adapter": adapter,
            "device": args.device,
            "dtype": "float32",
            "window": args.window,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "greedy_tokens": args.greedy_tokens,
            "query_gate_floor": args.query_gate_floor,
            "admission_only": args.admission_only,
            "methods": [] if args.admission_only else list(METHOD_IDS),
            "decay_factor": DECAY_FACTOR,
            "icul_correct_demonstrations": ICUL_CORRECT_DEMONSTRATIONS,
        },
        "admission_protocol": {
            "current_answer_present_minus_literal_repack_minimum_nats": (
                MIN_CURRENT_ANSWER_STORED_SIGNAL_LIFT_NATS
            ),
            "current_answer_present_first_token_rank_max": (
                MAX_CURRENT_ANSWER_PRESENT_FIRST_TOKEN_RANK
            ),
            "retained_first_token_rank_max": RETAINED_MAX_FIRST_TOKEN_RANK,
            "retained_reported_separately": True,
            "joint_requires_target_and_retained": True,
            "previous_answer_extracted_or_invented": False,
        },
        "metric_scope": {
            "standard_present_behavior": (
                "deterministic greedy exact-phrase diagnostic plus "
                "teacher-forced official current-answer score"
            ),
            "latency": "synchronized update/query/end-to-end timing",
            "storage": "deduplicated tensor backing storage",
            "output_kl": (
                "scalar full-vocabulary KL(full_repack || method) at first "
                "answer token"
            ),
        },
        "environment": _environment(),
        "records": [],
    }
    _atomic_write(output, report)
    for index, record in enumerate(selected):
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
                )
            else:
                result = evaluate_record(
                    runtime,
                    record,
                    records,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    greedy_tokens=args.greedy_tokens,
                )
        except Exception as exc:
            result = {
                "record_id": record.record_id,
                "source_example_id": record.target.question_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": "redacted; compare error_sha256",
                "error_sha256": text_sha256(str(exc)),
            }
        report["records"].append(result)
        _atomic_write(output, report)

    report["summary"] = summarize_records(report["records"])
    record_failures = [
        item["record_id"]
        for item in report["records"]
        if item.get("status") == "failed"
    ]
    method_failure_records = [
        item["record_id"]
        for item in report["records"]
        if item.get("method_failures")
    ]
    report["failures"] = {
        "record_ids": record_failures,
        "method_failure_record_ids": method_failure_records,
    }
    report["status"] = (
        "completed"
        if not record_failures and not method_failure_records
        else "completed_with_failures"
    )
    _atomic_write(output, report)
    print(f"wrote evaluation to {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

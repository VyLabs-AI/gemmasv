"""Sharded execution for LongMemEval v3 continuous utility.

Importing this module is data-free.  The live CLI checks committed-HEAD,
acknowledgement, and exact environment authorization before constructing a
runtime.  Terminal shards are immutable and never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Any, Mapping, Sequence

from gemma_sv import longmemeval_chat_all_history_replay as replay_plan
from gemma_sv import longmemeval_chat_v3_utility_execution_protocol as protocol


SCHEMA_VERSION = 1
RUN_SCHEMA = "gemma-sv-longmemeval-chat-v3-utility-run-v1"
CONTROL_SHARD_SCHEMA = (
    "gemma-sv-longmemeval-chat-v3-utility-control-shard-v1"
)
METHOD_SHARD_SCHEMA = "gemma-sv-longmemeval-chat-v3-utility-method-shard-v1"
ADMISSION_SCHEMA = "gemma-sv-longmemeval-chat-v3-utility-admission-v1"
FINAL_SCHEMA = "gemma-sv-longmemeval-chat-v3-utility-private-final-v1"

EXPECTED_CLUSTERS = protocol.EXPECTED_CLUSTERS
HISTORIES_PER_CLUSTER = protocol.HISTORIES_PER_CLUSTER
EXPECTED_HISTORIES = protocol.EXPECTED_HISTORIES
DEFAULT_OUTPUT_ROOT = (
    protocol.WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_v3_utility"
)


class UtilityEvaluationError(ValueError):
    """A score, phase barrier, shard, or runtime contract differs."""


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


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
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
        raise UtilityEvaluationError(f"{name} integrity differs")


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UtilityEvaluationError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise UtilityEvaluationError(f"{name} must be finite")
    return rendered


def first_token_rank(
    log_probabilities: Sequence[float],
    target_token_id: int,
) -> int:
    """One-indexed full-vocabulary rank; ties do not outrank gold."""

    target = int(target_token_id)
    try:
        import numpy as np
    except ImportError:
        values = [
            _finite(value, name="log probability")
            for value in log_probabilities
        ]
        if not 0 <= target < len(values):
            raise UtilityEvaluationError("target token is outside vocabulary")
        return 1 + sum(value > values[target] for value in values)
    values_array = np.asarray(log_probabilities, dtype=np.float64)
    if (
        values_array.ndim != 1
        or not np.isfinite(values_array).all()
        or not 0 <= target < values_array.size
    ):
        raise UtilityEvaluationError(
            "target token or vocabulary distribution is invalid"
        )
    return int(np.count_nonzero(values_array > values_array[target])) + 1


def full_vocabulary_kl(
    reference_log_probabilities: Sequence[float],
    method_log_probabilities: Sequence[float],
) -> float:
    """Compute KL(reference || method) in nats."""

    try:
        import numpy as np
    except ImportError:
        reference = [
            _finite(value, name="reference log probability")
            for value in reference_log_probabilities
        ]
        method = [
            _finite(value, name="method log probability")
            for value in method_log_probabilities
        ]
        if not reference or len(reference) != len(method):
            raise UtilityEvaluationError(
                "KL vectors must have equal non-empty vocabularies"
            )
        value = math.fsum(
            math.exp(log_p) * (log_p - log_q)
            for log_p, log_q in zip(reference, method)
            if math.exp(log_p) > 0.0
        )
    else:
        reference_array = np.asarray(
            reference_log_probabilities,
            dtype=np.float64,
        )
        method_array = np.asarray(
            method_log_probabilities,
            dtype=np.float64,
        )
        if (
            reference_array.ndim != 1
            or reference_array.size == 0
            or reference_array.shape != method_array.shape
            or not np.isfinite(reference_array).all()
            or not np.isfinite(method_array).all()
        ):
            raise UtilityEvaluationError(
                "KL vectors must be equal finite one-dimensional vocabularies"
            )
        probability = np.exp(reference_array)
        positive = probability > 0.0
        value = float(
            np.sum(
                probability[positive]
                * (
                    reference_array[positive]
                    - method_array[positive]
                ),
                dtype=np.float64,
            )
        )
    if value < -1e-10:
        raise UtilityEvaluationError("KL is materially negative")
    return max(0.0, value)


def forced_choice_gold_rank(
    mean_sequence_log_probabilities: Sequence[float],
    choice_answer_sha256: Sequence[str],
    gold_index: int,
) -> int:
    """Rank choices by score, then frozen hash, then index."""

    scores = [
        _finite(value, name="choice mean log probability")
        for value in mean_sequence_log_probabilities
    ]
    hashes = tuple(str(value) for value in choice_answer_sha256)
    if (
        len(scores) != protocol.CHOICE_COUNT
        or len(hashes) != protocol.CHOICE_COUNT
        or type(gold_index) is not int
        or not 0 <= gold_index < protocol.CHOICE_COUNT
        or any(not protocol._is_sha256(value) for value in hashes)
    ):
        raise UtilityEvaluationError("forced-choice score geometry differs")
    ordering = sorted(
        range(protocol.CHOICE_COUNT),
        key=lambda index: (-scores[index], hashes[index], index),
    )
    return ordering.index(gold_index) + 1


def _vector_sha256(value: Any) -> str:
    try:
        import numpy as np
    except ImportError:
        return _payload_sha256([float(item) for item in value])
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise UtilityEvaluationError("distribution must be finite and 1-D")
    digest = hashlib.sha256()
    digest.update(str(array.shape[0]).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def normalize_teacher_forced_score(
    raw: Mapping[str, Any],
    target_token_ids: Sequence[int],
) -> dict[str, Any]:
    """Normalize the scoring interface while retaining distributions privately."""

    target = tuple(int(value) for value in target_token_ids)
    if not target:
        raise UtilityEvaluationError("teacher-forced target is empty")
    total = raw.get(
        "total_log_probability_nats",
        raw.get("total_log_probability"),
    )
    mean = raw.get(
        "mean_log_probability_nats",
        raw.get("mean_log_probability"),
    )
    token_values = raw.get(
        "token_log_probabilities_nats",
        raw.get("token_log_probabilities"),
    )
    distributions = raw.get(
        "full_vocabulary_log_probabilities_by_target_token"
    )
    if (
        not isinstance(token_values, Sequence)
        or isinstance(token_values, (str, bytes))
        or not isinstance(distributions, Sequence)
        or isinstance(distributions, (str, bytes))
        or len(token_values) != len(target)
        or len(distributions) != len(target)
    ):
        raise UtilityEvaluationError(
            "score must include every gold-token value and distribution"
        )
    token_logps = tuple(
        _finite(value, name="gold-token log probability")
        for value in token_values
    )
    total_value = _finite(total, name="total sequence log probability")
    mean_value = _finite(mean, name="mean sequence log probability")
    if (
        not math.isclose(total_value, math.fsum(token_logps), abs_tol=1e-9)
        or not math.isclose(
            mean_value,
            total_value / len(target),
            abs_tol=1e-9,
        )
    ):
        raise UtilityEvaluationError("sequence score differs from token values")
    normalized_distributions: list[Any] = []
    vocabulary_size: int | None = None
    for distribution in distributions:
        try:
            import numpy as np
        except ImportError:
            values: Any = tuple(
                _finite(value, name="full-vocabulary log probability")
                for value in distribution
            )
            valid = bool(values)
        else:
            values = np.asarray(distribution, dtype=np.float64)
            valid = (
                values.ndim == 1
                and values.size > 0
                and bool(np.isfinite(values).all())
            )
        if not valid:
            raise UtilityEvaluationError(
                "full-vocabulary distribution must be finite and one-dimensional"
            )
        if vocabulary_size is None:
            vocabulary_size = len(values)
        elif len(values) != vocabulary_size:
            raise UtilityEvaluationError("vocabulary size changed across tokens")
        normalized_distributions.append(values)
    first_rank = first_token_rank(normalized_distributions[0], target[0])
    reported_rank = raw.get("first_target_token_rank", first_rank)
    if int(reported_rank) != first_rank:
        raise UtilityEvaluationError("reported first-token rank differs")
    return {
        "target_token_count": len(target),
        "all_target_tokens_teacher_forced": True,
        "total_log_probability_nats": total_value,
        "mean_log_probability_nats": mean_value,
        "geometric_mean_probability": math.exp(mean_value),
        "first_target_token_rank": first_rank,
        "gold_token_log_probabilities_nats": list(token_logps),
        "full_vocabulary_distribution_sha256_by_target_token": [
            _vector_sha256(value) for value in normalized_distributions
        ],
        "vocabulary_size": vocabulary_size,
        "_full_vocabulary_log_probabilities": tuple(
            normalized_distributions
        ),
    }


def compact_teacher_forced_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in score.items()
        if not str(key).startswith("_")
    }


def compare_method_to_rebuild(
    rebuild: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive per-gold-token KL and retained/target diagnostics."""

    reference = rebuild.get("_full_vocabulary_log_probabilities")
    observed = method.get("_full_vocabulary_log_probabilities")
    if (
        not isinstance(reference, Sequence)
        or not isinstance(observed, Sequence)
        or len(reference) != len(observed)
        or not reference
    ):
        raise UtilityEvaluationError("method comparison distributions differ")
    token_kls = [
        full_vocabulary_kl(left, right)
        for left, right in zip(reference, observed)
    ]
    drift = float(method["mean_log_probability_nats"]) - float(
        rebuild["mean_log_probability_nats"]
    )
    return {
        "direction": "KL(fresh_raw_omission || method)",
        "per_target_token_full_vocabulary_kl_nats": token_kls,
        "mean_full_vocabulary_kl_nats": math.fsum(token_kls) / len(token_kls),
        "maximum_full_vocabulary_kl_nats": max(token_kls),
        "first_target_token_full_vocabulary_kl_nats": token_kls[0],
        "method_minus_rebuild_mean_log_probability_nats": drift,
        "absolute_method_minus_rebuild_mean_log_probability_nats": abs(drift),
        "method_minus_rebuild_first_target_token_rank": (
            int(method["first_target_token_rank"])
            - int(rebuild["first_target_token_rank"])
        ),
    }


def _tokenize_without_special(runtime: Any, text: str) -> tuple[int, ...]:
    tokenizer = getattr(runtime, "tokenizer", None)
    if tokenizer is None:
        raise UtilityEvaluationError("runtime tokenizer is unavailable")
    encoded = tokenizer(str(text), add_special_tokens=False)
    ids = getattr(encoded, "input_ids", encoded)
    if (
        not isinstance(ids, Sequence)
        or isinstance(ids, (str, bytes))
        or not ids
    ):
        raise UtilityEvaluationError("tokenizer produced no IDs")
    return tuple(int(value) for value in ids)


def _probe_prompt_and_target(
    runtime: Any,
    probe: Any,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prompt = getattr(probe, "prompt_text", getattr(probe, "prompt", None))
    target = getattr(
        probe,
        "target_token_ids",
        getattr(probe, "target_ids", None),
    )
    if prompt is None or target is None:
        raise UtilityEvaluationError("probe prompt or target is unavailable")
    return _tokenize_without_special(runtime, str(prompt)), tuple(
        int(value) for value in target
    )


def score_probe(
    runtime: Any,
    memory: Any,
    probe: Any,
) -> dict[str, Any]:
    prompt_ids, target_ids = _probe_prompt_and_target(runtime, probe)
    scorer = getattr(runtime, "score_token_continuation_persistent", None)
    if not callable(scorer):
        raise UtilityEvaluationError(
            "runtime lacks full-continuation scoring interface"
        )
    return normalize_teacher_forced_score(
        scorer(memory, prompt_ids, target_ids),
        target_ids,
    )


def score_forced_choice(
    runtime: Any,
    memory: Any,
    probe: Any,
    choice_set: protocol.ForcedChoiceSet,
) -> dict[str, Any]:
    if not choice_set.available:
        return {
            "availability": "unavailable",
            "unavailable_reason": choice_set.unavailable_reason,
            "binding_sha256": choice_set.binding_sha256,
            "replacement_used": False,
            "mean_sequence_log_probabilities_nats": None,
            "gold_rank": None,
        }
    prompt_ids, _gold_ids = _probe_prompt_and_target(runtime, probe)
    means: list[float] = []
    for answer in choice_set.choice_answers:
        rendered = answer
        prompt_text = getattr(probe, "prompt_text", getattr(probe, "prompt", ""))
        if (
            prompt_text
            and not str(prompt_text)[-1].isspace()
            and not rendered[:1].isspace()
        ):
            rendered = " " + rendered
        answer_ids = _tokenize_without_special(runtime, rendered)
        raw = runtime.score_token_continuation_persistent(
            memory,
            prompt_ids,
            answer_ids,
        )
        normalized = normalize_teacher_forced_score(raw, answer_ids)
        means.append(float(normalized["mean_log_probability_nats"]))
    assert choice_set.gold_index is not None
    return {
        "availability": "available",
        "unavailable_reason": None,
        "binding_sha256": choice_set.binding_sha256,
        "replacement_used": False,
        "choice_answer_sha256": list(choice_set.choice_answer_sha256),
        "mean_sequence_log_probabilities_nats": means,
        "gold_index": choice_set.gold_index,
        "gold_rank": forced_choice_gold_rank(
            means,
            choice_set.choice_answer_sha256,
            choice_set.gold_index,
        ),
        "tie_rule": "score_desc_hash_asc_index_asc",
    }


def score_condition(
    runtime: Any,
    memory: Any,
    record: Any,
    choice_sets: Mapping[str, protocol.ForcedChoiceSet],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    probes = tuple(getattr(record, "probes", ()))
    if [getattr(item, "probe_id", None) for item in probes] != list(
        protocol.PROBES
    ):
        raise UtilityEvaluationError("record probe order differs")
    public: dict[str, Any] = {}
    private: dict[str, dict[str, Any]] = {}
    for probe in probes:
        probe_id = str(probe.probe_id)
        full = score_probe(runtime, memory, probe)
        private[probe_id] = full
        public[probe_id] = {
            "teacher_forced": compact_teacher_forced_score(full),
            "forced_choice": score_forced_choice(
                runtime,
                memory,
                probe,
                choice_sets[probe_id],
            ),
        }
    return public, private


def derive_admission(
    control_conditions: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply exactly the v1 control-only thresholds."""

    thresholds = authorization.get("admission") or {}
    target_lock = thresholds.get("target") or {}
    retained_lock = thresholds.get("retained") or {}
    try:
        present = control_conditions[protocol.PRESENT]["probes"]
        rebuild = control_conditions[protocol.FRESH_REBUILD]["probes"]
        target_present = present["target_current"]["teacher_forced"]
        target_rebuild = rebuild["target_current"]["teacher_forced"]
        retained_present = present["retained"]["teacher_forced"]
        retained_rebuild = rebuild["retained"]["teacher_forced"]
    except (KeyError, TypeError) as exc:
        return {
            "status": "control_failure",
            "target_admitted": False,
            "retained_available": False,
            "joint_admitted": False,
            "reasons": ["control_score_unavailable"],
            "source_conditions_only": [protocol.PRESENT, protocol.FRESH_REBUILD],
        }
    lift = float(target_present["mean_log_probability_nats"]) - float(
        target_rebuild["mean_log_probability_nats"]
    )
    lift_min = float(
        target_lock[
            "minimum_present_minus_fresh_rebuild_full_sequence_mean_log_probability_nats"
        ]
    )
    target_rank_max = int(
        target_lock["maximum_present_first_target_token_rank"]
    )
    retained_rank_max = int(
        retained_lock[
            "maximum_present_and_fresh_rebuild_first_target_token_rank"
        ]
    )
    target_admitted = (
        lift >= lift_min
        and int(target_present["first_target_token_rank"]) <= target_rank_max
    )
    retained_available = (
        int(retained_present["first_target_token_rank"]) <= retained_rank_max
        and int(retained_rebuild["first_target_token_rank"])
        <= retained_rank_max
    )
    reasons: list[str] = []
    if lift < lift_min:
        reasons.append("target_lift_below_locked_minimum")
    if int(target_present["first_target_token_rank"]) > target_rank_max:
        reasons.append("target_present_rank_above_locked_maximum")
    if not retained_available:
        reasons.append("retained_control_rank_above_locked_maximum")
    return {
        "status": "admitted" if target_admitted and retained_available else "rejected",
        "target_admitted": target_admitted,
        "retained_available": retained_available,
        "joint_admitted": target_admitted and retained_available,
        "target_present_minus_rebuild_mean_log_probability_nats": lift,
        "target_present_first_token_rank": int(
            target_present["first_target_token_rank"]
        ),
        "retained_present_first_token_rank": int(
            retained_present["first_target_token_rank"]
        ),
        "retained_rebuild_first_token_rank": int(
            retained_rebuild["first_target_token_rank"]
        ),
        "reasons": reasons,
        "source_conditions_only": [protocol.PRESENT, protocol.FRESH_REBUILD],
        "method_outputs_used": False,
        "thresholds": copy.deepcopy(dict(thresholds)),
    }


def split_wikitext_block(
    block: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    values = tuple(int(value) for value in block)
    if len(values) != 512 or any(value < 0 for value in values):
        raise UtilityEvaluationError("WikiText continuation block must be 512 IDs")
    return values[:256], values[256:]


def wikitext_blocks_sha256(blocks: Sequence[Sequence[int]]) -> str:
    """Match ``recovery_protocol.tensor_sha256(torch.int64[blocks,512])``."""

    if not blocks:
        raise UtilityEvaluationError("WikiText block tensor cannot be empty")
    width = len(blocks[0])
    if width < 1 or any(len(block) != width for block in blocks):
        raise UtilityEvaluationError("WikiText blocks must form a dense tensor")
    digest = hashlib.sha256()
    dtype = b"torch.int64"
    digest.update(struct.pack(">I", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack(">I", 2))
    digest.update(struct.pack(">Q", len(blocks)))
    digest.update(struct.pack(">Q", width))
    for block in blocks:
        for token_id in block:
            value = int(token_id)
            if not -(1 << 63) <= value < (1 << 63):
                raise UtilityEvaluationError("WikiText token is outside int64")
            digest.update(value.to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def prepare_post_delete_wikitext_blocks(
    blocks: Sequence[Sequence[int]],
    *,
    expected_400x512_sha256: str | None = None,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if len(blocks) < EXPECTED_CLUSTERS:
        raise UtilityEvaluationError("at least the first 32 WikiText blocks required")
    if expected_400x512_sha256 is not None:
        if (
            len(blocks) != 400
            or any(len(block) != 512 for block in blocks)
            or wikitext_blocks_sha256(blocks) != expected_400x512_sha256
        ):
            raise UtilityEvaluationError(
                "WikiText blocks differ from bound 400x512 v1 arm"
            )
    return tuple(
        split_wikitext_block(blocks[index])
        for index in range(EXPECTED_CLUSTERS)
    )


def score_wikitext_continuation(
    runtime: Any,
    memory: Any,
    split_block: tuple[tuple[int, ...], tuple[int, ...]],
) -> dict[str, Any]:
    prompt_ids, target_ids = split_block
    score = normalize_teacher_forced_score(
        runtime.score_token_continuation_persistent(
            memory,
            prompt_ids,
            target_ids,
        ),
        target_ids,
    )
    mean_nll = -float(score["mean_log_probability_nats"])
    return {
        "prompt_token_count": len(prompt_ids),
        "target_token_count": len(target_ids),
        "mean_nll_nats": mean_nll,
        "perplexity": math.exp(mean_nll),
        "all_target_tokens_teacher_forced": True,
    }


def paired_wikitext_cost(
    rebuild: Mapping[str, Any],
    method: Mapping[str, Any],
) -> dict[str, float]:
    delta = _finite(method.get("mean_nll_nats"), name="method NLL") - _finite(
        rebuild.get("mean_nll_nats"),
        name="rebuild NLL",
    )
    return {
        "method_minus_rebuild_mean_nll_nats": delta,
        "relative_perplexity_cost_percent": 100.0 * math.expm1(delta),
    }


def _failure(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8")
        ).hexdigest(),
        "error_message_redacted": True,
    }


def _anonymous(index: int) -> dict[str, int]:
    return {
        "history_index": index,
        "cluster_index": index // HISTORIES_PER_CLUSTER,
        "variant_index": index % HISTORIES_PER_CLUSTER,
    }


def _failed_cell(exc: BaseException, *, attempted: bool = True) -> dict[str, Any]:
    return {
        "status": "failed",
        "terminal": True,
        "attempted": attempted,
        "failure": _failure(exc),
    }


def build_control_shard(
    index: int,
    *,
    authorization: Mapping[str, Any],
    conditions: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (conditions is None) == (failure is None):
        raise UtilityEvaluationError(
            "control shard requires exactly result or failure"
        )
    status = "failed"
    if conditions is not None:
        status = (
            "completed"
            if all(
                isinstance(cell, Mapping)
                and cell.get("status") == "completed"
                for cell in conditions.values()
            )
            else "completed_with_condition_failures"
        )
    shard = _seal(
        {
            "schema": CONTROL_SHARD_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            **_anonymous(index),
            "phase": "controls",
            "terminal": True,
            "status": status,
            "authorization_integrity_sha256": authorization["integrity"][
                "sha256"
            ],
            "conditions": (
                copy.deepcopy(dict(conditions))
                if conditions is not None
                else None
            ),
            "failure": (
                copy.deepcopy(dict(failure)) if failure is not None else None
            ),
        }
    )
    validate_control_shard(shard, authorization=authorization, index=index)
    return shard


def validate_control_shard(
    shard: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    index: int,
) -> None:
    _validate_seal(shard, name=f"control shard {index}")
    status = shard.get("status")
    if (
        shard.get("schema") != CONTROL_SHARD_SCHEMA
        or shard.get("schema_version") != SCHEMA_VERSION
        or shard.get("history_index") != index
        or shard.get("cluster_index") != index // HISTORIES_PER_CLUSTER
        or shard.get("variant_index") != index % HISTORIES_PER_CLUSTER
        or shard.get("phase") != "controls"
        or shard.get("terminal") is not True
        or shard.get("authorization_integrity_sha256")
        != authorization["integrity"]["sha256"]
        or status
        not in {"completed", "completed_with_condition_failures", "failed"}
    ):
        raise UtilityEvaluationError("control shard binding differs")
    if status in {"completed", "completed_with_condition_failures"}:
        conditions = shard.get("conditions")
        if (
            not isinstance(conditions, Mapping)
            or list(conditions) != [protocol.PRESENT, protocol.FRESH_REBUILD]
            or shard.get("failure") is not None
        ):
            raise UtilityEvaluationError("terminal control shard differs")
        cell_statuses = []
        for cell in conditions.values():
            if (
                not isinstance(cell, Mapping)
                or cell.get("status") not in {"completed", "failed"}
                or cell.get("terminal") is not True
                or cell.get("attempted") is not True
            ):
                raise UtilityEvaluationError("control condition cell differs")
            if cell["status"] == "completed" and not isinstance(
                cell.get("probes"), Mapping
            ):
                raise UtilityEvaluationError("completed control cell has no probes")
            if cell["status"] == "failed" and not isinstance(
                cell.get("failure"), Mapping
            ):
                raise UtilityEvaluationError("failed control cell has no failure")
            cell_statuses.append(cell["status"])
        expected_status = (
            "completed"
            if all(value == "completed" for value in cell_statuses)
            else "completed_with_condition_failures"
        )
        if status != expected_status:
            raise UtilityEvaluationError("control aggregate status differs")
    elif (
        shard.get("conditions") is not None
        or not isinstance(shard.get("failure"), Mapping)
    ):
        raise UtilityEvaluationError("failed control shard differs")


def build_method_shard(
    index: int,
    *,
    authorization: Mapping[str, Any],
    conditions: Mapping[str, Any] | None = None,
    wikitext: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (conditions is None) == (failure is None):
        raise UtilityEvaluationError(
            "method shard requires exactly result or failure"
        )
    status = "failed"
    if conditions is not None:
        all_cells = [
            *conditions.values(),
            *((wikitext or {}).values()),
        ]
        status = (
            "completed"
            if all(
                isinstance(cell, Mapping)
                and cell.get("status") == "completed"
                for cell in all_cells
            )
            else "completed_with_condition_failures"
        )
    shard = _seal(
        {
            "schema": METHOD_SHARD_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            **_anonymous(index),
            "phase": "policy_and_replay",
            "terminal": True,
            "status": status,
            "authorization_integrity_sha256": authorization["integrity"][
                "sha256"
            ],
            "conditions": (
                copy.deepcopy(dict(conditions))
                if conditions is not None
                else None
            ),
            "wikitext": (
                copy.deepcopy(dict(wikitext))
                if wikitext is not None
                else None
            ),
            "failure": (
                copy.deepcopy(dict(failure)) if failure is not None else None
            ),
        }
    )
    validate_method_shard(shard, authorization=authorization, index=index)
    return shard


def validate_method_shard(
    shard: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    index: int,
) -> None:
    _validate_seal(shard, name=f"method shard {index}")
    status = shard.get("status")
    if (
        shard.get("schema") != METHOD_SHARD_SCHEMA
        or shard.get("schema_version") != SCHEMA_VERSION
        or shard.get("history_index") != index
        or shard.get("cluster_index") != index // HISTORIES_PER_CLUSTER
        or shard.get("variant_index") != index % HISTORIES_PER_CLUSTER
        or shard.get("phase") != "policy_and_replay"
        or shard.get("terminal") is not True
        or shard.get("authorization_integrity_sha256")
        != authorization["integrity"]["sha256"]
        or status
        not in {"completed", "completed_with_condition_failures", "failed"}
    ):
        raise UtilityEvaluationError("method shard binding differs")
    if status in {"completed", "completed_with_condition_failures"}:
        if (
            list(shard.get("conditions") or {}) != [
                protocol.POLICY,
                protocol.REPLAY,
            ]
            or list(shard.get("wikitext") or {}) != [
                protocol.FRESH_REBUILD,
                protocol.POLICY,
                protocol.REPLAY,
            ]
            or shard.get("failure") is not None
        ):
            raise UtilityEvaluationError("terminal method shard differs")
        cells = [
            *(shard["conditions"].values()),
            *(shard["wikitext"].values()),
        ]
        cell_statuses = []
        for cell in cells:
            if (
                not isinstance(cell, Mapping)
                or cell.get("status") not in {"completed", "failed"}
                or cell.get("terminal") is not True
                or cell.get("attempted") is not True
            ):
                raise UtilityEvaluationError("method condition cell differs")
            if cell["status"] == "failed" and not isinstance(
                cell.get("failure"), Mapping
            ):
                raise UtilityEvaluationError("failed method cell has no failure")
            cell_statuses.append(cell["status"])
        expected_status = (
            "completed"
            if all(value == "completed" for value in cell_statuses)
            else "completed_with_condition_failures"
        )
        if status != expected_status:
            raise UtilityEvaluationError("method aggregate status differs")
    elif (
        shard.get("conditions") is not None
        or shard.get("wikitext") is not None
        or not isinstance(shard.get("failure"), Mapping)
    ):
        raise UtilityEvaluationError("failed method shard differs")


def _atomic_write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(name)
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
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise UtilityEvaluationError(f"{path.name} must be an object")
    return value


class UtilityShardStore:
    """No-overwrite phase store with global control/admission barriers."""

    def __init__(
        self,
        root: str | Path,
        authorization: Mapping[str, Any],
        *,
        resume: bool,
    ) -> None:
        protocol.validate_execution_authorization(authorization)
        self.root = Path(root).resolve()
        self.authorization = authorization
        self.control_root = self.root / "controls"
        self.method_root = self.root / "methods"
        self.admission_path = self.root / "admission.json"
        self.run_path = self.root / "run.json"
        self.final_path = self.root / "final.json"
        if self.root.exists() and not resume:
            raise FileExistsError("utility output root exists; pass resume")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.run_path.exists():
            if not resume:
                raise FileExistsError("utility run already exists")
            self._validate_run(_load_json(self.run_path))
        else:
            run = _seal(
                {
                    "schema": RUN_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "status": "running",
                    "authorization_integrity_sha256": authorization[
                        "integrity"
                    ]["sha256"],
                    "expected_histories": EXPECTED_HISTORIES,
                    "phase_order": [
                        "controls",
                        "admission",
                        "policy_and_replay",
                        "post_delete_wikitext",
                    ],
                    "no_overwrite": True,
                }
            )
            _atomic_write_new(self.run_path, run)
        self.control_root.mkdir(exist_ok=True)
        self.method_root.mkdir(exist_ok=True)

    def _validate_run(self, run: Mapping[str, Any]) -> None:
        _validate_seal(run, name="run")
        if (
            run.get("schema") != RUN_SCHEMA
            or run.get("authorization_integrity_sha256")
            != self.authorization["integrity"]["sha256"]
            or run.get("expected_histories") != EXPECTED_HISTORIES
            or run.get("no_overwrite") is not True
        ):
            raise UtilityEvaluationError("utility run binding differs")

    @staticmethod
    def _path(root: Path, index: int) -> Path:
        return root / f"{index:03d}.json"

    def controls(self) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        expected = {f"{index:03d}.json" for index in range(EXPECTED_HISTORIES)}
        for path in self.control_root.iterdir():
            if path.name not in expected or not path.is_file():
                raise UtilityEvaluationError("control directory has extra entry")
            index = int(path.stem)
            value = _load_json(path)
            validate_control_shard(
                value,
                authorization=self.authorization,
                index=index,
            )
            result[index] = value
        return result

    def methods(self) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        expected = {f"{index:03d}.json" for index in range(EXPECTED_HISTORIES)}
        for path in self.method_root.iterdir():
            if path.name not in expected or not path.is_file():
                raise UtilityEvaluationError("method directory has extra entry")
            index = int(path.stem)
            value = _load_json(path)
            validate_method_shard(
                value,
                authorization=self.authorization,
                index=index,
            )
            result[index] = value
        return result

    def write_control(self, index: int, shard: Mapping[str, Any]) -> None:
        if self.admission_path.exists() or self.methods():
            raise UtilityEvaluationError("control phase is already closed")
        validate_control_shard(
            shard,
            authorization=self.authorization,
            index=index,
        )
        _atomic_write_new(self._path(self.control_root, index), shard)

    def write_admission(self, rows: Sequence[Mapping[str, Any]]) -> None:
        controls = self.controls()
        if len(controls) != EXPECTED_HISTORIES:
            raise UtilityEvaluationError(
                "all 96 controls must terminate before admission"
            )
        if len(rows) != EXPECTED_HISTORIES:
            raise UtilityEvaluationError("admission requires 96 rows")
        value = _seal(
            {
                "schema": ADMISSION_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "authorization_integrity_sha256": self.authorization[
                    "integrity"
                ]["sha256"],
                "source_conditions_only": [
                    protocol.PRESENT,
                    protocol.FRESH_REBUILD,
                ],
                "method_outputs_used": False,
                "rows": copy.deepcopy(list(rows)),
            }
        )
        _atomic_write_new(self.admission_path, value)

    def admission(self) -> dict[str, Any]:
        if not self.admission_path.is_file():
            raise UtilityEvaluationError("admission barrier is incomplete")
        value = _load_json(self.admission_path)
        _validate_seal(value, name="admission")
        rows = value.get("rows")
        if (
            value.get("schema") != ADMISSION_SCHEMA
            or value.get("authorization_integrity_sha256")
            != self.authorization["integrity"]["sha256"]
            or value.get("source_conditions_only")
            != [protocol.PRESENT, protocol.FRESH_REBUILD]
            or value.get("method_outputs_used") is not False
            or not isinstance(rows, list)
            or len(rows) != EXPECTED_HISTORIES
            or any(
                row.get("history_index") != index
                for index, row in enumerate(rows)
            )
        ):
            raise UtilityEvaluationError("admission artifact differs")
        return value

    def write_method(self, index: int, shard: Mapping[str, Any]) -> None:
        if len(self.controls()) != EXPECTED_HISTORIES:
            raise UtilityEvaluationError("control global barrier is incomplete")
        self.admission()
        validate_method_shard(
            shard,
            authorization=self.authorization,
            index=index,
        )
        _atomic_write_new(self._path(self.method_root, index), shard)


def evaluate_controls_for_history(
    runtime: Any,
    record: Any,
    choices: Mapping[str, protocol.ForcedChoiceSet],
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition_id, token_ids in (
        (protocol.PRESENT, record.context.original_token_ids),
        (protocol.FRESH_REBUILD, record.raw_omitted_token_ids),
    ):
        try:
            memory = runtime.prefill_persistent(list(token_ids))
            scores, _private = score_condition(runtime, memory, record, choices)
            conditions[condition_id] = {
                "status": "completed",
                "terminal": True,
                "attempted": True,
                "probes": scores,
            }
        except Exception as exc:
            conditions[condition_id] = _failed_cell(exc)
    return conditions


def _policy_memory(runtime: Any, record: Any) -> tuple[Any, Mapping[str, Any]]:
    original = runtime.prefill_persistent(
        list(record.context.original_token_ids)
    )
    diagnostics, states = runtime.persistent_certificate_states(
        original,
        tuple(record.context.forget_positions),
    )
    if not isinstance(states, Mapping) or "exact" not in states:
        raise UtilityEvaluationError("exact policy state is unavailable")
    return states["exact"], diagnostics


def evaluate_methods_for_history(
    runtime: Any,
    record: Any,
    plan: Any,
    choices: Mapping[str, protocol.ForcedChoiceSet],
    wikitext_block: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rebuild_memory: Any | None = None
    rebuild_private: dict[str, dict[str, Any]] | None = None
    rebuild_error: BaseException | None = None
    try:
        rebuild_memory = runtime.prefill_persistent(
            list(record.raw_omitted_token_ids)
        )
        _rebuild_public, rebuild_private = score_condition(
            runtime,
            rebuild_memory,
            record,
            choices,
        )
    except Exception as exc:
        rebuild_error = exc

    state_results: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    state_errors: dict[str, BaseException] = {}
    try:
        state_results[protocol.POLICY] = _policy_memory(runtime, record)
    except Exception as exc:
        state_errors[protocol.POLICY] = exc
    try:
        checkpoint = runtime.checkpoint_prefix(plan.prefix_token_ids)
        state_results[protocol.REPLAY] = runtime.replay_stored_suffix(
            checkpoint,
            plan.suffix_token_ids,
            expected_omitted_token_ids=plan.omitted_token_ids,
            original_owned_range=plan.owned_range,
        )
    except Exception as exc:
        state_errors[protocol.REPLAY] = exc

    conditions: dict[str, Any] = {}
    for condition_id in (protocol.POLICY, protocol.REPLAY):
        if condition_id in state_errors:
            conditions[condition_id] = _failed_cell(
                state_errors[condition_id]
            )
            continue
        memory, diagnostics = state_results[condition_id]
        try:
            public, private = score_condition(runtime, memory, record, choices)
            if rebuild_private is None:
                raise UtilityEvaluationError(
                    "fresh rebuild distribution reference failed"
                ) from rebuild_error
            for probe_id in protocol.PROBES:
                public[probe_id][
                    "rebuild_comparison"
                ] = compare_method_to_rebuild(
                    rebuild_private[probe_id],
                    private[probe_id],
                )
            conditions[condition_id] = {
                "status": "completed",
                "terminal": True,
                "attempted": True,
                "probes": public,
                "state_diagnostics": copy.deepcopy(dict(diagnostics)),
            }
        except Exception as exc:
            conditions[condition_id] = _failed_cell(exc)

    wikitext: dict[str, Any] = {}
    rebuild_wikitext: dict[str, Any] | None = None
    if rebuild_memory is None:
        wikitext[protocol.FRESH_REBUILD] = _failed_cell(
            rebuild_error
            or UtilityEvaluationError("fresh rebuild state unavailable")
        )
    else:
        try:
            rebuild_wikitext = score_wikitext_continuation(
                runtime,
                rebuild_memory,
                wikitext_block,
            )
            wikitext[protocol.FRESH_REBUILD] = {
                "status": "completed",
                "terminal": True,
                "attempted": True,
                **rebuild_wikitext,
            }
        except Exception as exc:
            wikitext[protocol.FRESH_REBUILD] = _failed_cell(exc)
    for condition_id in (protocol.POLICY, protocol.REPLAY):
        if condition_id in state_errors:
            wikitext[condition_id] = _failed_cell(state_errors[condition_id])
            continue
        memory, _diagnostics = state_results[condition_id]
        try:
            score = score_wikitext_continuation(
                runtime,
                memory,
                wikitext_block,
            )
            if rebuild_wikitext is None:
                raise UtilityEvaluationError(
                    "paired WikiText rebuild reference unavailable"
                )
            score["rebuild_comparison"] = paired_wikitext_cost(
                rebuild_wikitext,
                score,
            )
            wikitext[condition_id] = {
                "status": "completed",
                "terminal": True,
                "attempted": True,
                **score,
            }
        except Exception as exc:
            wikitext[condition_id] = _failed_cell(exc)
    return conditions, wikitext


def assemble_private_final(
    store: UtilityShardStore,
) -> dict[str, Any]:
    controls = store.controls()
    methods = store.methods()
    admission = store.admission()
    if len(controls) != EXPECTED_HISTORIES or len(methods) != EXPECTED_HISTORIES:
        raise UtilityEvaluationError("private final requires all terminal shards")
    final = _seal(
        {
            "schema": FINAL_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "authorization_integrity_sha256": store.authorization["integrity"][
                "sha256"
            ],
            "contains_source_text": False,
            "contains_source_identifiers": False,
            "contains_full_vocabulary_vectors": False,
            "contains_gold_token_arrays": True,
            "counts": {
                "control_terminal": len(controls),
                "control_completed": sum(
                    row["status"] == "completed" for row in controls.values()
                ),
                "method_terminal": len(methods),
                "method_completed": sum(
                    row["status"] == "completed" for row in methods.values()
                ),
                "admission_rows": len(admission["rows"]),
                "all_96_histories_accounted": True,
            },
            "control_shard_integrity_sha256": [
                controls[index]["integrity"]["sha256"]
                for index in range(EXPECTED_HISTORIES)
            ],
            "method_shard_integrity_sha256": [
                methods[index]["integrity"]["sha256"]
                for index in range(EXPECTED_HISTORIES)
            ],
            "admission_integrity_sha256": admission["integrity"]["sha256"],
        }
    )
    if store.final_path.exists():
        observed = _load_json(store.final_path)
        if observed != final:
            raise UtilityEvaluationError("existing private final differs")
        return observed
    _atomic_write_new(store.final_path, final)
    return final


def run_sharded_utility(
    runtime: Any,
    records: Sequence[Any],
    plans: Sequence[Any],
    choices: Sequence[Mapping[str, protocol.ForcedChoiceSet]],
    wikitext_blocks: Sequence[Sequence[int]],
    authorization: Mapping[str, Any],
    *,
    output_root: str | Path,
    resume: bool,
) -> dict[str, Any]:
    protocol.validate_execution_authorization(authorization)
    if not (
        len(records)
        == len(plans)
        == len(choices)
        == EXPECTED_HISTORIES
    ):
        raise UtilityEvaluationError("runner requires all 96 ordered histories")
    expected_wikitext_hash = (
        authorization.get("broad_wikitext_quality_arm", {})
        .get("dataset", {})
        .get("token_ids_sha256")
    )
    split_blocks = prepare_post_delete_wikitext_blocks(
        wikitext_blocks,
        expected_400x512_sha256=(
            str(expected_wikitext_hash)
            if expected_wikitext_hash is not None
            else None
        ),
    )
    store = UtilityShardStore(output_root, authorization, resume=resume)

    controls = store.controls()
    for index, (record, choice_set) in enumerate(zip(records, choices)):
        if index in controls:
            continue
        try:
            result = evaluate_controls_for_history(
                runtime,
                record,
                choice_set,
            )
            shard = build_control_shard(
                index,
                authorization=authorization,
                conditions=result,
            )
        except Exception as exc:
            shard = build_control_shard(
                index,
                authorization=authorization,
                failure=_failure(exc),
            )
        store.write_control(index, shard)
    controls = store.controls()
    if len(controls) != EXPECTED_HISTORIES:
        raise UtilityEvaluationError("control phase did not terminate all histories")

    if not store.admission_path.exists():
        admission_rows = []
        for index in range(EXPECTED_HISTORIES):
            shard = controls[index]
            values = derive_admission(
                (
                    shard["conditions"]
                    if isinstance(shard.get("conditions"), Mapping)
                    else {}
                ),
                authorization,
            )
            admission_rows.append({**_anonymous(index), **values})
        store.write_admission(admission_rows)
    store.admission()

    methods = store.methods()
    for index, (record, plan, choice_set) in enumerate(
        zip(records, plans, choices)
    ):
        if index in methods:
            continue
        try:
            conditions, wikitext = evaluate_methods_for_history(
                runtime,
                record,
                plan,
                choice_set,
                split_blocks[index // HISTORIES_PER_CLUSTER],
            )
            shard = build_method_shard(
                index,
                authorization=authorization,
                conditions=conditions,
                wikitext=wikitext,
            )
        except Exception as exc:
            shard = build_method_shard(
                index,
                authorization=authorization,
                failure=_failure(exc),
            )
        store.write_method(index, shard)
    return assemble_private_final(store)


def _make_runtime():
    from gemma_sv.demo_server.all_history_replay_runtime_v3 import (
        AllHistoryReplayGemmaRuntimeV3,
    )
    from gemma_sv.demo_server.gemma_engine import RuntimeConfig
    from gemma_sv import eval_longmemeval_chat_v2 as eval_v2

    return AllHistoryReplayGemmaRuntimeV3(
        RuntimeConfig(**eval_v2._runtime_config_kwargs(device="mps")),
        certificate_workers=8,
    )


def run_from_paths(
    *,
    authorization_path: Path,
    output_root: Path,
    data_path: str | Path | None,
    acknowledgement: str,
    resume: bool,
) -> dict[str, Any]:
    authorization = protocol.load_execution_authorization(authorization_path)
    # This gate intentionally precedes model construction and source loading.
    protocol.authorize_execution(
        authorization,
        acknowledgement=acknowledgement,
        require_committed=True,
        authorization_path=authorization_path,
    )
    from gemma_sv import longmemeval_chat_cohort_v3 as cohort_v3
    from gemma_sv import eval_longmemeval_chat as eval_v1
    from gemma_sv.data import wikitext103_blocks

    eval_v1._configure_determinism()
    with eval_v1._offline_huggingface():
        rows = cohort_v3.base.load_pinned_longmemeval_rows(data_path)
        examples = cohort_v3.base.extract_longmemeval_examples(rows)
        runtime = _make_runtime()
        runtime.ensure_loaded()
        cohort = protocol.load_json(protocol.DEFAULT_COHORT, name="v3 cohort")
        records = cohort_v3.rehydrate_manifest(
            cohort,
            rows,
            runtime.tokenizer,
        )
        suffix = protocol.load_json(
            protocol.DEFAULT_SUFFIX_SCAN,
            name="frozen suffix scan",
        )
        plans, _rows = replay_plan.plan_all_histories(records, cohort, suffix)
        private_choices, public_choices = protocol.freeze_forced_choice_sets(
            records,
            examples,
        )
        if list(public_choices) != authorization["forced_choice_sets"]["rows"]:
            raise UtilityEvaluationError(
                "rehydrated forced-choice sets differ from authorization"
            )
        blocks = wikitext103_blocks(
            runtime.tokenizer,
            512,
            max_blocks=400,
            revision=(
                authorization["broad_wikitext_quality_arm"]["dataset"][
                    "revision"
                ]
            ),
        )
        return run_sharded_utility(
            runtime,
            records,
            plans,
            private_choices,
            blocks,
            authorization,
            output_root=output_root,
            resume=resume,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--authorization",
        type=Path,
        default=protocol.DEFAULT_AUTHORIZATION,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-path")
    parser.add_argument("--acknowledgement", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_from_paths(
            authorization_path=args.authorization,
            output_root=args.output_root,
            data_path=args.data_path,
            acknowledgement=args.acknowledgement,
            resume=bool(args.resume),
        )
    except (
        FileExistsError,
        OSError,
        PermissionError,
        RuntimeError,
        UtilityEvaluationError,
        protocol.UtilityProtocolError,
    ) as exc:
        parser.error(str(exc))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROL_SHARD_SCHEMA",
    "METHOD_SHARD_SCHEMA",
    "UtilityEvaluationError",
    "UtilityShardStore",
    "build_control_shard",
    "build_method_shard",
    "compare_method_to_rebuild",
    "derive_admission",
    "first_token_rank",
    "forced_choice_gold_rank",
    "full_vocabulary_kl",
    "normalize_teacher_forced_score",
    "paired_wikitext_cost",
    "prepare_post_delete_wikitext_blocks",
    "run_sharded_utility",
    "split_wikitext_block",
    "validate_control_shard",
    "validate_method_shard",
    "wikitext_blocks_sha256",
]

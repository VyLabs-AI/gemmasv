"""Teacher-forced readout probes over a Kimi Linear record memory.

The measurements mirror the Gemma harness so the two papers report comparable
quantities: the mean log probability the memory assigns to a target
continuation, the rank and probability of its first token, and the *lift* --
the log-probability advantage a resident record confers over a context that
never ingested it. After exact deletion the lift should collapse to zero, and
"zero" here is a measured value, not an assumption.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import mlx.core as mx

from .records import RecordMemory


@dataclass(frozen=True)
class ProbeStats:
    mean_log_probability: float
    total_log_probability: float
    first_token_rank: int
    first_token_probability: float
    n_target_tokens: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def teacher_forced(
    memory: RecordMemory, prompt: mx.array, target: mx.array
) -> ProbeStats:
    """Score ``target`` given ``prompt``, appended to ``memory``'s context.

    The memory is left untouched: the probe rolls the cache back afterwards.
    """
    prompt = prompt.reshape(1, -1)
    target = target.reshape(1, -1)
    n_prompt, n_target = int(prompt.shape[1]), int(target.shape[1])
    if n_prompt == 0 or n_target == 0:
        raise ValueError("prompt and target must both be non-empty")

    seq = mx.concatenate([prompt, target], axis=1)
    logits = memory.probe_logits(seq)

    # Position i predicts token i+1, so the logits that predict the target span
    # start one step before it.
    span = logits[0, n_prompt - 1 : n_prompt - 1 + n_target, :].astype(mx.float32)
    logprobs = span - mx.logsumexp(span, axis=-1, keepdims=True)

    targets = target[0]
    picked = mx.take_along_axis(logprobs, targets.reshape(-1, 1), axis=-1).reshape(-1)
    total = float(mx.sum(picked).item())

    first = logprobs[0]
    first_target = int(targets[0].item())
    first_logprob = float(first[first_target].item())
    rank = int(mx.sum(first > first[first_target]).item()) + 1

    return ProbeStats(
        mean_log_probability=total / n_target,
        total_log_probability=total,
        first_token_rank=rank,
        first_token_probability=float(mx.exp(mx.array(first_logprob)).item()),
        n_target_tokens=n_target,
    )


def lift_nats(present: ProbeStats, reference: ProbeStats) -> float:
    """How much log probability (per token) a resident record contributes."""
    return present.mean_log_probability - reference.mean_log_probability

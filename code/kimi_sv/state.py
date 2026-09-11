"""Boundary checkpoints for Kimi Linear's hybrid cache.

Kimi Linear interleaves two memory kinds (27 layers: 20 KDA + 7 global MLA in
the 48B-A3B release), and each kind needs a different rollback mechanism:

KDA layers hold an ``ArraysCache(size=4)`` -- three short-convolution states plus
the recurrent delta-rule state ``S``. Its size is fixed by the architecture, not
by context length, so a checkpoint costs the same whether it is taken at token
100 or token 100k. The MLX Metal kernel returns ``state_out`` as a fresh array
rather than writing through its input, so a checkpoint may hold array references
directly; ``mx.eval`` is still called so a checkpoint pins materialized buffers
instead of an unevaluated graph.

MLA layers hold a standard ``KVCache``, which is a prefix structure: the state
after ``m`` tokens is exactly the first ``m`` entries. Rolling back is therefore
``trim`` (an offset decrement) and needs no stored copy at all.

Note that ``mlx_lm.models.cache.trim_prompt_cache`` cannot be used here:
``ArraysCache`` does not implement ``is_trimmable``, so the top-level helper
declines to trim any layer of a Kimi cache. The MLA caches are trimmed directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx

KDAState = Tuple[Optional[mx.array], ...]


def classify_layers(model: Any) -> Tuple[List[int], List[int]]:
    """Split a Kimi Linear model's layer indices into ``(kda, mla)``."""
    kda: List[int] = []
    mla: List[int] = []
    for idx, layer in enumerate(model.layers):
        (kda if layer.is_linear else mla).append(idx)
    if not mla:
        raise ValueError("no global MLA layer found; not a Kimi Linear hybrid")
    return kda, mla


@dataclass(frozen=True)
class Boundary:
    """A rollback point in a running Kimi Linear cache.

    ``token_offset`` is the number of tokens ingested when the checkpoint was
    taken, read from the MLA caches (the KDA caches carry no offset counter in
    single-sequence use).
    """

    token_offset: int
    kda: Dict[int, KDAState]
    mla_offset: Dict[int, int]
    label: Optional[str] = None

    @property
    def nbytes(self) -> int:
        return sum(
            a.nbytes
            for arrays in self.kda.values()
            for a in arrays
            if a is not None
        )


def snapshot(model: Any, cache: Sequence[Any], label: Optional[str] = None) -> Boundary:
    """Checkpoint ``cache`` so :func:`restore` can return to this exact point."""
    kda_idx, mla_idx = classify_layers(model)

    kda: Dict[int, KDAState] = {}
    for i in kda_idx:
        arrays = tuple(cache[i].state)
        live = [a for a in arrays if a is not None]
        if live:
            mx.eval(*live)
        kda[i] = arrays

    mla_offset = {i: int(cache[i].offset) for i in mla_idx}
    offsets = set(mla_offset.values())
    if len(offsets) != 1:
        raise ValueError(f"MLA caches disagree on token offset: {sorted(offsets)}")

    return Boundary(
        token_offset=offsets.pop(), kda=kda, mla_offset=mla_offset, label=label
    )


def restore(model: Any, cache: Sequence[Any], boundary: Boundary) -> None:
    """Roll ``cache`` back to ``boundary`` in place.

    KDA caches are reassigned from the checkpoint; MLA caches are trimmed to the
    checkpointed offset. Rolling *forward* is not possible and raises.
    """
    kda_idx, mla_idx = classify_layers(model)

    missing = set(kda_idx) - set(boundary.kda)
    if missing:
        raise ValueError(f"checkpoint is missing KDA layers {sorted(missing)}")

    for i in kda_idx:
        cache[i].state = list(boundary.kda[i])

    for i in mla_idx:
        c = cache[i]
        target = boundary.mla_offset[i]
        if c.offset < target:
            raise ValueError(
                f"layer {i}: cache offset {c.offset} is behind checkpoint {target}; "
                "cannot roll forward"
            )
        if c.offset > target:
            c.trim(c.offset - target)


def describe(model: Any, cache: Sequence[Any]) -> Dict[str, Any]:
    """Per-kind cache sizes, for reporting checkpoint and KV cost."""
    kda_idx, mla_idx = classify_layers(model)
    kda_bytes = sum(cache[i].nbytes for i in kda_idx)
    mla_bytes = sum(cache[i].nbytes for i in mla_idx)
    return {
        "kda_layers": len(kda_idx),
        "mla_layers": len(mla_idx),
        "token_offset": int(cache[mla_idx[0]].offset),
        "kda_bytes": kda_bytes,
        "mla_bytes": mla_bytes,
        "checkpoint_bytes": kda_bytes,
    }

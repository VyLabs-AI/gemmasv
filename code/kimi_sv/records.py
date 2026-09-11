"""A persistent record memory over Kimi Linear, with deletion by replay.

Records are ingested sequentially into one long-lived context. A boundary
checkpoint is taken immediately *before* each record, so deleting record ``k``
means rolling the hybrid cache back to that checkpoint and replaying only the
records that came after it. The resulting state is the state the model would
hold had record ``k`` never been ingested -- not an approximation of it.

Probes are non-mutating: the same checkpoint machinery rolls the cache back
after reading, so measuring the memory never disturbs it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx

from .state import Boundary, classify_layers, restore, snapshot


@dataclass(frozen=True)
class Record:
    """One deletable unit of context."""

    key: str
    tokens: mx.array
    text: Optional[str] = None

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.shape[-1])


def record_from_text(key: str, text: str, tokenizer: Any) -> Record:
    ids = tokenizer.encode(text)
    return Record(key=key, tokens=mx.array([ids]), text=text)


@dataclass
class DeletionReport:
    """What a deletion cost, for the paper's cost curve."""

    key: str
    rolled_back_to: int
    replayed_records: List[str]
    replayed_tokens: int
    seconds: float
    checkpoint_bytes: int

    @property
    def n_replayed(self) -> int:
        return len(self.replayed_records)


class RecordMemory:
    """A context of records over a Kimi Linear model, supporting exact deletion."""

    def __init__(self, model: Any, cache: Optional[List[Any]] = None):
        self.model = model
        self.cache = model.make_cache() if cache is None else cache
        self._records: List[Record] = []
        # Checkpoint taken immediately before each record was ingested.
        self._before: Dict[str, Boundary] = {}

    # -- inspection ------------------------------------------------------

    @property
    def keys(self) -> List[str]:
        return [r.key for r in self._records]

    @property
    def token_offset(self) -> int:
        _, mla = classify_layers(self.model)
        return int(self.cache[mla[0]].offset)

    def checkpoint_bytes(self) -> int:
        return sum(b.nbytes for b in self._before.values())

    # -- core ------------------------------------------------------------

    def _forward(self, tokens: mx.array) -> mx.array:
        logits = self.model(tokens, cache=self.cache)
        mx.eval(logits)
        return logits[:, -1, :]

    def ingest(self, record: Record) -> None:
        if record.key in self._before:
            raise ValueError(f"record {record.key!r} is already resident")
        self._before[record.key] = snapshot(
            self.model, self.cache, label=f"pre:{record.key}"
        )
        self._forward(record.tokens)
        self._records.append(record)

    def ingest_all(self, records: Sequence[Record]) -> None:
        for r in records:
            self.ingest(r)

    def delete(self, key: str) -> DeletionReport:
        """Remove ``key`` by rolling back and replaying the surviving suffix."""
        try:
            idx = self.keys.index(key)
        except ValueError:
            raise KeyError(f"no resident record {key!r}") from None

        mark = self._before[key]
        survivors = list(self._records[idx + 1 :])

        t0 = time.perf_counter()
        restore(self.model, self.cache, mark)

        # Checkpoints for the deleted record and everything after it were taken
        # in a context that included the deleted record, so they are stale.
        self._records = self._records[:idx]
        for stale in [key] + [s.key for s in survivors]:
            self._before.pop(stale, None)

        for s in survivors:
            self.ingest(s)
        elapsed = time.perf_counter() - t0

        return DeletionReport(
            key=key,
            rolled_back_to=mark.token_offset,
            replayed_records=[s.key for s in survivors],
            replayed_tokens=sum(s.n_tokens for s in survivors),
            seconds=elapsed,
            checkpoint_bytes=mark.nbytes,
        )

    def amend(self, key: str, replacement: Record) -> DeletionReport:
        """Replace ``key`` in place: roll back to its boundary, ingest the
        replacement, then replay the surviving suffix.

        The rewind and the suffix replay are identical to :meth:`delete`; the
        replacement simply rides the replay, so a correction costs the same
        as a deletion and is audited the same way -- against a memory built
        with the replacement from the start.
        """
        try:
            idx = self.keys.index(key)
        except ValueError:
            raise KeyError(f"no resident record {key!r}") from None

        mark = self._before[key]
        survivors = list(self._records[idx + 1 :])

        t0 = time.perf_counter()
        restore(self.model, self.cache, mark)
        self._records = self._records[:idx]
        for stale in [key] + [s.key for s in survivors]:
            self._before.pop(stale, None)

        for s in [replacement] + survivors:
            self.ingest(s)
        elapsed = time.perf_counter() - t0

        return DeletionReport(
            key=key,
            rolled_back_to=mark.token_offset,
            replayed_records=[replacement.key] + [s.key for s in survivors],
            replayed_tokens=replacement.n_tokens
            + sum(s.n_tokens for s in survivors),
            seconds=elapsed,
            checkpoint_bytes=mark.nbytes,
        )

    def probe(self, tokens: mx.array) -> mx.array:
        """Read next-token logits for ``tokens`` without disturbing the memory."""
        mark = snapshot(self.model, self.cache, label="probe")
        try:
            return self._forward(tokens)
        finally:
            restore(self.model, self.cache, mark)

    def probe_logits(self, tokens: mx.array) -> mx.array:
        """Full ``(1, L, V)`` logits for ``tokens``, without disturbing the memory."""
        mark = snapshot(self.model, self.cache, label="probe")
        try:
            logits = self.model(tokens, cache=self.cache)
            mx.eval(logits)
            return logits
        finally:
            restore(self.model, self.cache, mark)

    def generate(
        self,
        tokens: mx.array,
        max_tokens: int = 24,
        stop_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Greedy-decode a continuation of ``tokens`` without disturbing the memory."""
        stop = set(stop_ids or ())
        mark = snapshot(self.model, self.cache, label="generate")
        try:
            out: List[int] = []
            logits = self._forward(tokens)
            for _ in range(max_tokens):
                tok = int(mx.argmax(logits, axis=-1).item())
                if tok in stop:
                    break
                out.append(tok)
                logits = self._forward(mx.array([[tok]]))
            return out
        finally:
            restore(self.model, self.cache, mark)

    def kda_state(self) -> List[mx.array]:
        """The live KDA recurrent + conv arrays, for state-level comparison."""
        kda, _ = classify_layers(self.model)
        return [a for i in kda for a in self.cache[i].state if a is not None]


def max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def state_max_abs_diff(left: RecordMemory, right: RecordMemory) -> float:
    """Largest disagreement between two memories' KDA states."""
    ls, rs = left.kda_state(), right.kda_state()
    if len(ls) != len(rs):
        raise ValueError(f"state arity differs: {len(ls)} vs {len(rs)}")
    return max((max_abs_diff(a, b) for a, b in zip(ls, rs)), default=0.0)

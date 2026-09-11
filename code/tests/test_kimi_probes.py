"""Teacher-forced probe arithmetic and the post-deletion collapse of lift."""
from __future__ import annotations

import pytest

from kimi_sv.probes import lift_nats, teacher_forced
from kimi_sv.records import Record, RecordMemory
from kimi_sv.tiny import build_tiny, token_block

PROMPT = token_block(5, start=900)
TARGET = token_block(4, start=950)


def _memory(records):
    model, _ = build_tiny(seed=0)
    mem = RecordMemory(model)
    mem.ingest_all(records)
    return mem


def _records():
    return [
        Record(key=f"r{i}", tokens=token_block(12, start=100 * (i + 1)))
        for i in range(3)
    ]


def test_teacher_forced_arithmetic():
    stats = teacher_forced(_memory(_records()), PROMPT, TARGET)

    assert stats.n_target_tokens == 4
    assert stats.mean_log_probability == pytest.approx(
        stats.total_log_probability / 4, rel=1e-6
    )
    assert stats.mean_log_probability < 0.0
    assert 1 <= stats.first_token_rank <= 512
    assert 0.0 < stats.first_token_probability <= 1.0


def test_probe_does_not_advance_context():
    mem = _memory(_records())
    offset = mem.token_offset

    first = teacher_forced(mem, PROMPT, TARGET)
    second = teacher_forced(mem, PROMPT, TARGET)

    assert mem.token_offset == offset
    assert first == second


def test_lift_collapses_to_zero_after_deletion():
    records = _records()
    victim = records[1].key

    mem = _memory(records)
    reference = _memory([r for r in records if r.key != victim])

    present = teacher_forced(mem, PROMPT, TARGET)
    never = teacher_forced(reference, PROMPT, TARGET)
    assert abs(lift_nats(present, never)) > 1e-5

    mem.delete(victim)
    deleted = teacher_forced(mem, PROMPT, TARGET)

    assert lift_nats(deleted, never) == 0.0
    assert deleted.first_token_rank == never.first_token_rank


def test_empty_spans_rejected():
    mem = _memory(_records())
    with pytest.raises(ValueError, match="non-empty"):
        teacher_forced(mem, PROMPT, TARGET[:, :0])

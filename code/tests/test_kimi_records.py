"""Record-level deletion semantics for the Kimi Linear replay harness.

Run on the miniature random model: these check that deleting a record leaves the
memory in the state it would have held had the record never arrived, for records
at the start, middle, and end of the context.
"""
from __future__ import annotations

import pytest

from kimi_sv.records import (
    Record,
    RecordMemory,
    max_abs_diff,
    state_max_abs_diff,
)
from kimi_sv.tiny import build_tiny, token_block

PROBE = token_block(6, start=900)


def _memory():
    model, _ = build_tiny(seed=0)
    return RecordMemory(model)


def _records(n: int = 4, size: int = 12):
    return [
        Record(key=f"r{i}", tokens=token_block(size, start=100 * (i + 1)))
        for i in range(n)
    ]


def _reference(records):
    """A memory that only ever saw ``records``."""
    mem = _memory()
    mem.ingest_all(records)
    return mem


@pytest.mark.parametrize("victim", [0, 1, 3])
def test_deletion_matches_never_ingested(victim):
    records = _records()
    key = records[victim].key

    mem = _memory()
    mem.ingest_all(records)
    before = mem.probe(PROBE)

    report = mem.delete(key)

    survivors = [r for r in records if r.key != key]
    ref = _reference(survivors)

    assert mem.keys == [r.key for r in survivors]
    assert mem.token_offset == ref.token_offset
    assert report.replayed_records == [r.key for r in records[victim + 1 :]]

    # Non-vacuity: the record must have changed the readout in the first place.
    assert max_abs_diff(before, ref.probe(PROBE)) > 1e-4

    assert max_abs_diff(mem.probe(PROBE), ref.probe(PROBE)) == 0.0
    assert state_max_abs_diff(mem, ref) == 0.0


@pytest.mark.parametrize("victim", [0, 1, 3])
def test_amendment_matches_corrected_rebuild(victim):
    records = _records()
    key = records[victim].key
    replacement = Record(key=key, tokens=token_block(9, start=700))

    mem = _memory()
    mem.ingest_all(records)
    report = mem.amend(key, replacement)

    corrected = [replacement if r.key == key else r for r in records]
    ref = _reference(corrected)

    assert mem.keys == [r.key for r in corrected]
    assert report.replayed_records == [r.key for r in corrected[victim:]]
    assert report.replayed_tokens == sum(r.n_tokens for r in corrected[victim:])

    # Non-vacuity: the replacement must differ from what it replaced.
    wrong_ref = _reference(records)
    assert max_abs_diff(ref.probe(PROBE), wrong_ref.probe(PROBE)) > 1e-4

    assert max_abs_diff(mem.probe(PROBE), ref.probe(PROBE)) == 0.0
    assert state_max_abs_diff(mem, ref) == 0.0


def test_deleting_last_record_replays_nothing():
    records = _records()
    mem = _memory()
    mem.ingest_all(records)

    report = mem.delete(records[-1].key)

    assert report.replayed_records == []
    assert report.replayed_tokens == 0
    assert mem.token_offset == sum(r.n_tokens for r in records[:-1])


def test_replay_cost_grows_with_suffix_length():
    records = _records(n=5)

    early = _memory()
    early.ingest_all(records)
    late = _memory()
    late.ingest_all(records)

    first = early.delete(records[0].key)
    last = late.delete(records[-1].key)

    # Deleting early replays the whole suffix; deleting last replays nothing.
    assert first.replayed_tokens > last.replayed_tokens == 0
    assert first.n_replayed == 4


def test_sequential_deletions_stay_exact():
    records = _records(n=4)
    mem = _memory()
    mem.ingest_all(records)

    mem.delete("r1")
    mem.delete("r2")

    ref = _reference([r for r in records if r.key not in {"r1", "r2"}])
    assert mem.keys == ["r0", "r3"]
    assert max_abs_diff(mem.probe(PROBE), ref.probe(PROBE)) == 0.0
    assert state_max_abs_diff(mem, ref) == 0.0


def test_probe_does_not_disturb_memory():
    records = _records()
    mem = _memory()
    mem.ingest_all(records)

    offset = mem.token_offset
    first = mem.probe(PROBE)
    mem.probe(token_block(20, start=5))
    second = mem.probe(PROBE)

    assert mem.token_offset == offset
    assert max_abs_diff(first, second) == 0.0


def test_duplicate_and_missing_keys_rejected():
    mem = _memory()
    rec = Record(key="dup", tokens=token_block(8))
    mem.ingest(rec)

    with pytest.raises(ValueError, match="already resident"):
        mem.ingest(rec)
    with pytest.raises(KeyError, match="no resident record"):
        mem.delete("absent")


def test_checkpoint_cost_scales_with_record_count_not_length():
    short = _memory()
    short.ingest_all(_records(n=3, size=8))

    long_ctx = _memory()
    long_ctx.ingest_all(_records(n=3, size=64))

    assert long_ctx.token_offset > short.token_offset
    assert short.checkpoint_bytes() == long_ctx.checkpoint_bytes() > 0

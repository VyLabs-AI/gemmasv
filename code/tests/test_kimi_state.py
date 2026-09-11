"""Boundary-checkpoint and replay-deletion mechanics for Kimi Linear.

These run on a miniature randomly-initialized model: they establish that the
hybrid cache can be rolled back exactly and that replaying the surviving suffix
reproduces a never-ingested state. They say nothing about language quality.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from kimi_sv.state import Boundary, classify_layers, describe, restore, snapshot
from kimi_sv.tiny import build_tiny, greedy, ingest, token_block


def _fresh():
    model, _ = build_tiny(seed=0)
    return model, model.make_cache()


def _kda_arrays(model, cache):
    kda_idx, _ = classify_layers(model)
    return [a for i in kda_idx for a in cache[i].state if a is not None]


def _max_abs_diff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def test_layer_split_matches_config():
    model, _ = build_tiny()
    kda, mla = classify_layers(model)
    assert kda == [0, 1, 2]
    assert mla == [3]


def test_snapshot_restore_reproduces_continuation():
    """Rolling back to a checkpoint must reproduce the identical continuation."""
    model, cache = _fresh()
    ingest(model, cache, token_block(24))

    mark = snapshot(model, cache, label="mark")
    first = greedy(model, cache, token_block(8, start=100), n=6)

    restore(model, cache, mark)
    assert cache[3].offset == mark.token_offset

    second = greedy(model, cache, token_block(8, start=100), n=6)
    assert first == second


def test_replay_deletion_matches_never_ingested():
    """The core claim: restore + replay the suffix == never ingesting the record.

    Deleted path:  P -> [checkpoint] -> R -> S -> restore -> S
    Reference path: P -> S
    """
    prefix = token_block(24, start=0)
    record = token_block(12, start=300)
    suffix = token_block(16, start=700)

    # Deleted path.
    model, cache = _fresh()
    ingest(model, cache, prefix)
    mark = snapshot(model, cache, label="pre-record")
    ingest(model, cache, record)
    with_record = ingest(model, cache, suffix)
    restore(model, cache, mark)
    deleted = ingest(model, cache, suffix)

    # Reference path: a context that never saw the record.
    ref_model, ref_cache = _fresh()
    ingest(ref_model, ref_cache, prefix)
    reference = ingest(ref_model, ref_cache, suffix)

    # Control: the record must actually have mattered, or the test is vacuous.
    assert _max_abs_diff(with_record, reference) > 1e-4

    assert cache[3].offset == ref_cache[3].offset
    assert _max_abs_diff(deleted, reference) == 0.0

    for got, want in zip(_kda_arrays(model, cache), _kda_arrays(ref_model, ref_cache)):
        assert _max_abs_diff(got, want) == 0.0


def test_checkpoint_size_is_independent_of_position():
    """KDA state is fixed by architecture, so checkpoint cost must not grow."""
    model, cache = _fresh()

    ingest(model, cache, token_block(16))
    early = snapshot(model, cache)

    ingest(model, cache, token_block(240, start=50))
    late = snapshot(model, cache)

    assert late.token_offset > early.token_offset
    assert early.nbytes == late.nbytes > 0

    # The MLA KV cache, by contrast, does grow with position -- which is why it
    # is trimmed rather than copied.
    info = describe(model, cache)
    assert info["checkpoint_bytes"] == info["kda_bytes"]


def test_restore_cannot_roll_forward():
    model, cache = _fresh()
    ingest(model, cache, token_block(24))
    ahead = snapshot(model, cache)

    behind = Boundary(
        token_offset=8,
        kda=ahead.kda,
        mla_offset={i: 8 for i in ahead.mla_offset},
    )
    restore(model, cache, behind)

    with pytest.raises(ValueError, match="cannot roll forward"):
        restore(model, cache, ahead)


def test_restore_rejects_incomplete_checkpoint():
    model, cache = _fresh()
    ingest(model, cache, token_block(8))
    mark = snapshot(model, cache)

    partial = Boundary(
        token_offset=mark.token_offset,
        kda={0: mark.kda[0]},
        mla_offset=mark.mla_offset,
    )
    with pytest.raises(ValueError, match="missing KDA layers"):
        restore(model, cache, partial)

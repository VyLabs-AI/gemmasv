"""The decay baseline must actually attenuate the state it claims to attenuate.

The decay-versus-replay conclusion rests on the retention factor really being
applied, so these check the arithmetic directly rather than inferring it from
downstream readouts.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from kimi_sv.decay import RECURRENT, scale_recurrent_state
from kimi_sv.records import Record, RecordMemory
from kimi_sv.state import classify_layers
from kimi_sv.tiny import build_tiny, token_block


def _memory():
    model, _ = build_tiny(seed=0)
    mem = RecordMemory(model)
    mem.ingest_all(
        [
            Record(key=f"r{i}", tokens=token_block(12, start=100 * (i + 1)))
            for i in range(3)
        ]
    )
    return mem


def _recurrent(mem):
    kda, _ = classify_layers(mem.model)
    return [mem.cache[i][RECURRENT] for i in kda]


def test_recurrent_index_is_the_state_not_a_conv_buffer():
    mem = _memory()
    kda, _ = classify_layers(mem.model)
    state = mem.cache[kda[0]][RECURRENT]

    # The recurrent state is [B, heads, Dv, Dk]; conv states are [B, k-1, dim].
    assert state.ndim == 4
    assert state.dtype == mx.float32


def test_zero_retention_erases_the_linear_memory():
    mem = _memory()
    assert any(float(mx.max(mx.abs(s)).item()) > 0 for s in _recurrent(mem))

    scale_recurrent_state(mem.model, mem.cache, 0.0)

    for s in _recurrent(mem):
        assert float(mx.max(mx.abs(s)).item()) == 0.0


def test_partial_retention_scales_exactly():
    mem = _memory()
    before = [mx.array(s) for s in _recurrent(mem)]

    scale_recurrent_state(mem.model, mem.cache, 0.5)

    for old, new in zip(before, _recurrent(mem)):
        assert float(mx.max(mx.abs(new - old * 0.5)).item()) == 0.0


def test_unit_retention_is_a_no_op():
    mem = _memory()
    before = [mx.array(s) for s in _recurrent(mem)]

    scale_recurrent_state(mem.model, mem.cache, 1.0)

    for old, new in zip(before, _recurrent(mem)):
        assert float(mx.max(mx.abs(new - old)).item()) == 0.0


def test_conv_states_are_left_alone():
    mem = _memory()
    kda, _ = classify_layers(mem.model)
    before = [mx.array(mem.cache[i][j]) for i in kda for j in range(RECURRENT)]

    scale_recurrent_state(mem.model, mem.cache, 0.0)

    after = [mem.cache[i][j] for i in kda for j in range(RECURRENT)]
    for old, new in zip(before, after):
        assert float(mx.max(mx.abs(new - old)).item()) == 0.0


@pytest.mark.parametrize("gamma", [-0.1, 1.5])
def test_retention_factor_must_be_a_probability(gamma):
    mem = _memory()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        scale_recurrent_state(mem.model, mem.cache, gamma)

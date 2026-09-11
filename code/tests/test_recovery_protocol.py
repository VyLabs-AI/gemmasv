from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from gemma_sv.recovery_protocol import (
    BatchFingerprint,
    seed_everything,
    validate_stage2_pair,
)
from gemma_sv.recovery_state import (
    apply_recovery_state,
    collect_recovery_state,
    save_recovery_state,
)


def test_seed_everything_replays_python_numpy_and_torch():
    seed_everything(17)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    seed_everything(17)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert first == second


def test_batch_fingerprint_is_order_sensitive_and_device_independent():
    first = torch.arange(12, dtype=torch.long).reshape(3, 4)
    second = first + 100
    left = BatchFingerprint()
    right = BatchFingerprint()
    reversed_order = BatchFingerprint()
    for batch in (first, second):
        left.update(batch)
        right.update(batch.clone())
    for batch in (second, first):
        reversed_order.update(batch)
    assert left.summary() == right.summary()
    assert left.summary()["sha256"] != reversed_order.summary()["sha256"]
    assert left.summary()["batches"] == 2


def _result(stream_hash: str) -> dict:
    return {
        "config": {
            "model": "gemma",
            "model_revision": "revision",
            "seq_len": 8,
            "batch": 2,
            "stage2_steps": 3,
            "lr2": 0.001,
            "rank": 8,
            "data_seed": 0,
            "init_seed": 0,
        },
        "stage2_stream": {
            "batches": 3,
            "sha256": stream_hash,
            "first_batch_sha256": "first",
            "last_batch_sha256": "last",
        },
    }


def test_validate_stage2_pair_rejects_different_stream():
    validate_stage2_pair(_result("same"), _result("same"))
    with pytest.raises(ValueError, match="stage2_stream.sha256"):
        validate_stage2_pair(_result("recovery"), _result("control"))


class _Attention(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int,
        *,
        learned: bool,
        preserve_prefix_mass: bool = False,
    ):
        super().__init__()
        self.is_sliding = False
        self.layer_idx = layer_idx
        self.chunk = 128
        self.nu = 0.3
        self.readout = "softmax"
        self.preserve_prefix_mass = preserve_prefix_mass
        self.kpar = None
        self._learn_kpar = learned
        if learned:
            self.log_kpar = torch.nn.Parameter(torch.tensor(2.5).log())


class _Layer(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int,
        *,
        learned: bool,
        preserve_prefix_mass: bool = False,
    ):
        super().__init__()
        self.self_attn = _Attention(
            layer_idx,
            learned=learned,
            preserve_prefix_mass=preserve_prefix_mass,
        )


class _Model(torch.nn.Module):
    def __init__(self, *, learned: bool, preserve_prefix_mass: bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _Layer(
                    5,
                    learned=learned,
                    preserve_prefix_mass=preserve_prefix_mass,
                ),
                _Layer(
                    11,
                    learned=learned,
                    preserve_prefix_mass=preserve_prefix_mass,
                ),
            ]
        )


def test_recovery_state_round_trip_applies_bandwidths(tmp_path):
    trained = _Model(learned=True)
    state = collect_recovery_state(
        trained,
        model_id="gemma",
        model_revision="revision",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_recovery_state(adapter, state)

    deployed = _Model(learned=False)
    restored = apply_recovery_state(deployed, adapter)
    assert restored == state
    assert [layer.self_attn.kpar for layer in deployed.layers] == pytest.approx(
        [2.5, 2.5]
    )
    assert all(not layer.self_attn._learn_kpar for layer in deployed.layers)


def test_recovery_state_readout_override_is_explicit(tmp_path):
    state = collect_recovery_state(
        _Model(learned=True),
        model_id="gemma",
        model_revision="revision",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    save_recovery_state(adapter, state)
    deployed = _Model(learned=False, preserve_prefix_mass=True)

    with pytest.raises(ValueError, match="preserve_prefix_mass mismatch"):
        apply_recovery_state(deployed, adapter)

    apply_recovery_state(
        deployed,
        adapter,
        allow_readout_override=True,
    )
    assert all(
        layer.self_attn.preserve_prefix_mass for layer in deployed.layers
    )

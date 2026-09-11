from __future__ import annotations

from types import SimpleNamespace

import pytest

import gemma_sv.eval_whole_record_unlearning as whole_record
from gemma_sv.demo_server.gemma_engine import RuntimeConfig


def _args(*, admission_only: bool = True):
    return SimpleNamespace(
        record_start=0,
        records=None,
        window=512,
        n_fill=22,
        prefix_fillers=8,
        admission_only=admission_only,
        conditions=("present",),
        samples=1,
        batch_size=1,
        max_new_tokens=1,
        temperature=1.0,
        top_p=1.0,
        seed=0,
        save_generations=False,
        kv_cache=True,
        k_values=[1],
    )


def test_admission_only_skips_behavioral_conditions(monkeypatch):
    present = SimpleNamespace(token_ids=[1, 2, 3], positions={"forget": (1,)})
    no_forget = SimpleNamespace(token_ids=[1, 3], positions={"forget": ()})
    admission = {
        "fields": [{"first_token_rank": 1}],
        "retain": {"first_token_rank": 1},
    }
    monkeypatch.setattr(
        whole_record,
        "_synthetic_memory",
        lambda *args, **kwargs: (present, no_forget),
    )
    monkeypatch.setattr(
        whole_record,
        "_synthetic_admission",
        lambda *args, **kwargs: (admission, []),
    )
    manifest = {
        "name": "manifest",
        "version": 1,
        "records": [{"record_id": "r0", "question": "q"}],
    }

    result = whole_record.evaluate(
        SimpleNamespace(),
        manifest,
        [],
        [],
        [],
        _args(),
    )

    assert result["mode"] == "admission_only"
    assert result["attempted"] == 1
    assert result["admitted"] == 1
    assert result["exact_phrase_leak_at_k"] == {}
    assert result["records"][0]["admission"] == admission
    assert "conditions" not in result["records"][0]


def test_lora_normalization_and_ungrafted_default():
    assert whole_record._normalize_lora("none") is None
    assert whole_record._normalize_lora(" NULL ") is None
    assert whole_record._normalize_lora("adapter") == "adapter"
    assert RuntimeConfig().graft_enabled is True
    assert RuntimeConfig(graft_enabled=False).graft_enabled is False


def test_adapter_hash_requires_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(FileNotFoundError, match="missing adapter weights"):
        whole_record._adapter_sha256(str(adapter))

    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    assert len(whole_record._adapter_sha256(str(adapter))) == 64

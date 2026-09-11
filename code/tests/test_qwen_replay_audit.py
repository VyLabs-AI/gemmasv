from __future__ import annotations

import torch

from gemma_sv.qwen_replay_audit import (
    _new_cache,
    run_suffix_case,
    snapshot_cache,
)


def _tiny_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
    )

    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
        mtp_num_hidden_layers=0,
    )
    return Qwen3_5ForCausalLM(config).eval()


def test_qwen_same_schedule_replay_covers_complete_cache():
    torch.manual_seed(7)
    model = _tiny_model()
    segments = {
        "prefix": list(range(10, 26)),
        "victim": list(range(30, 38)),
        "suffix_a": list(range(40, 56)),
    }
    result = run_suffix_case(
        model,
        segments,
        "suffix_a",
        probe_layers=[0, 2],
        device="cpu",
    )

    replay = result["replay_vs_fresh_omission"]
    repeat = result["repeat_vs_fresh_omission"]
    assert replay["all_arrays_exact"]
    assert replay["all_flags_equal"]
    assert replay["flag_count"] == 12
    assert "layer_0.conv_initialized" in replay["flag_names"]
    assert "layer_0.recurrent_initialized" in replay["flag_names"]
    assert replay["max_abs"] == 0.0
    assert repeat["all_arrays_exact"]
    assert repeat["all_flags_equal"]
    assert repeat["all_values_finite"]
    assert result["present_vs_omitted"]["max_abs"] > 0.0
    receipt_rows = list(result["receipt_analysis"].values())
    for row in receipt_rows:
        assert row["frozen_transport_relative_residual"] < 5e-3
        assert row["forcing_decomposition_max_relative_residual"] < 5e-3
        assert row["live_kernel_conformance_max_relative_residual"] < 5e-3
        assert (
            row[
                "native_sequential_to_live_state_scale_relative_residual"
            ]
            < 5e-3
        )


def test_qwen_snapshot_deep_clones_mutable_linear_state():
    torch.manual_seed(11)
    model = _tiny_model()
    cache = _new_cache(model)
    from gemma_sv.qwen_replay_audit import prefill

    prefill(model, cache, list(range(12, 28)), "cpu")
    mark = snapshot_cache(model, cache)
    saved = mark["linear"][0]["recurrent_states"].clone()
    with torch.inference_mode():
        cache.layers[0].recurrent_states.add_(1.0)

    assert torch.equal(mark["linear"][0]["recurrent_states"], saved)
    assert not torch.equal(
        cache.layers[0].recurrent_states,
        mark["linear"][0]["recurrent_states"],
    )

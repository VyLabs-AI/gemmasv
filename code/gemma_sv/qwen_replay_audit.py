"""Qwen3.5 hybrid-memory receipt and checkpoint-replay audit.

The live protocol is frozen in ``benchmarks/qwen35_replay_v1.json``.  It
compares schedule-matched present, omitted, replay, and repeat trajectories
over every active cache component.  For selected DeltaNet layers it also
checks frozen-input receipt transport and the changed-transition/write forcing
decomposition.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / "qwen35_replay_v1.json"
)
DEFAULT_OUTPUT = "outputs/qwen35_replay/audit_v1.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _text_config(model_or_config: Any):
    config = getattr(model_or_config, "config", model_or_config)
    return getattr(config, "text_config", config)


def _language_model(model: Any):
    candidate = getattr(model, "model", None)
    if candidate is not None and hasattr(candidate, "language_model"):
        return candidate.language_model
    if candidate is not None and hasattr(candidate, "layers"):
        return candidate
    raise TypeError("unsupported Qwen3.5 model wrapper")


def _rope_owner(model: Any):
    candidate = getattr(model, "model", None)
    return candidate if candidate is not None and hasattr(candidate, "rope_deltas") else None


def _layer_types(model: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in _text_config(model).layer_types)


def _new_cache(model: Any):
    from transformers.cache_utils import DynamicCache

    return DynamicCache(config=_text_config(model))


def _cache_length(cache: Any) -> int:
    return int(cache.get_seq_length())


def _linear_state(cache: Any, layer_id: int) -> tuple[Any, Any]:
    layer = cache.layers[int(layer_id)]
    return layer.conv_states, layer.recurrent_states


def _clone_tensor(value: Any):
    return None if value is None else value.detach().clone()


def snapshot_cache(model: Any, cache: Any) -> dict[str, Any]:
    """Deep snapshot the mutable recurrent state and logical cache boundary."""

    linear = {}
    for layer_id, kind in enumerate(_layer_types(model)):
        if kind != "linear_attention":
            continue
        layer = cache.layers[layer_id]
        linear[layer_id] = {
            "conv_states": _clone_tensor(layer.conv_states),
            "recurrent_states": _clone_tensor(layer.recurrent_states),
            "has_previous_state": bool(layer.has_previous_state),
            "is_conv_states_initialized": bool(
                getattr(layer, "is_conv_states_initialized", False)
            ),
            "is_recurrent_states_initialized": bool(
                getattr(layer, "is_recurrent_states_initialized", False)
            ),
        }
    owner = _rope_owner(model)
    rope = None if owner is None else _clone_tensor(owner.rope_deltas)
    return {
        "length": _cache_length(cache),
        "linear": linear,
        "rope_deltas": rope,
    }


def restore_cache(model: Any, cache: Any, snapshot: Mapping[str, Any]) -> None:
    """Restore a forward cache to a previously snapshotted boundary."""

    import torch

    with torch.inference_mode():
        if _cache_length(cache) < int(snapshot["length"]):
            raise ValueError("cannot restore beyond the current cache length")
        cache.crop(int(snapshot["length"]))
        for layer_id, state in snapshot["linear"].items():
            layer = cache.layers[int(layer_id)]
            layer.conv_states.copy_(state["conv_states"])
            layer.recurrent_states.copy_(state["recurrent_states"])
            layer.has_previous_state = bool(state["has_previous_state"])
            layer.is_conv_states_initialized = bool(
                state["is_conv_states_initialized"]
            )
            layer.is_recurrent_states_initialized = bool(
                state["is_recurrent_states_initialized"]
            )
        owner = _rope_owner(model)
        if owner is not None:
            owner.rope_deltas = (
                None
                if snapshot["rope_deltas"] is None
                else snapshot["rope_deltas"].detach().clone()
            )
    if _cache_length(cache) != int(snapshot["length"]):
        raise RuntimeError("cache length restore failed")


def _state_snapshot(model: Any, cache: Any, logits: Any) -> dict[str, Any]:
    arrays = {"logits": logits.detach().cpu().clone()}
    flags = {"length": _cache_length(cache)}
    for layer_id, kind in enumerate(_layer_types(model)):
        layer = cache.layers[layer_id]
        if kind == "linear_attention":
            arrays[f"layer_{layer_id}.conv"] = (
                layer.conv_states.detach().cpu().clone()
            )
            arrays[f"layer_{layer_id}.recurrent"] = (
                layer.recurrent_states.detach().cpu().clone()
            )
            flags[f"layer_{layer_id}.previous"] = bool(
                layer.has_previous_state
            )
            flags[f"layer_{layer_id}.conv_initialized"] = bool(
                layer.is_conv_states_initialized
            )
            flags[f"layer_{layer_id}.recurrent_initialized"] = bool(
                layer.is_recurrent_states_initialized
            )
        else:
            arrays[f"layer_{layer_id}.keys"] = layer.keys.detach().cpu().clone()
            arrays[f"layer_{layer_id}.values"] = (
                layer.values.detach().cpu().clone()
            )
            flags[f"layer_{layer_id}.initialized"] = bool(layer.is_initialized)
    owner = _rope_owner(model)
    if owner is not None and owner.rope_deltas is not None:
        arrays["wrapper.rope_deltas"] = owner.rope_deltas.detach().cpu().clone()
    else:
        flags["wrapper.rope_deltas"] = None
    return {"arrays": arrays, "flags": flags}


def compare_snapshots(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    strict_shapes: bool = True,
) -> dict[str, Any]:
    ref_arrays = reference["arrays"]
    cand_arrays = candidate["arrays"]
    if set(ref_arrays) != set(cand_arrays):
        raise ValueError("snapshot array coverage differs")
    per_array = {}
    maximum = 0.0
    all_exact = True
    all_finite = True
    shape_mismatch_count = 0
    for name in sorted(ref_arrays):
        first = ref_arrays[name]
        second = cand_arrays[name]
        if tuple(first.shape) != tuple(second.shape):
            if strict_shapes:
                raise ValueError(f"{name} shape differs")
            shape_mismatch_count += 1
            all_exact = False
            finite = bool(first.isfinite().all() and second.isfinite().all())
            all_finite = all_finite and finite
            per_array[name] = {
                "reference_shape": list(first.shape),
                "candidate_shape": list(second.shape),
                "max_abs": None,
                "exact": False,
                "finite": finite,
            }
            continue
        finite = bool(first.isfinite().all() and second.isfinite().all())
        exact = bool(first.equal(second))
        value = (
            float((first.float() - second.float()).abs().max())
            if first.numel()
            else 0.0
        )
        maximum = max(maximum, value)
        all_exact = all_exact and exact
        all_finite = all_finite and finite
        per_array[name] = {
            "shape": list(first.shape),
            "max_abs": value,
            "exact": exact,
            "finite": finite,
        }
    flags_equal = reference["flags"] == candidate["flags"]
    return {
        "array_count": len(per_array),
        "flag_count": len(reference["flags"]),
        "flag_names": sorted(reference["flags"]),
        "shape_mismatch_count": shape_mismatch_count,
        "max_abs": maximum,
        "all_arrays_exact": all_exact,
        "all_flags_equal": flags_equal,
        "all_values_finite": all_finite,
        "per_array": per_array,
    }


def _input_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(value)
        for value in tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids
    ]


def validate_tokenization(
    tokenizer: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    segments = {
        name: _input_ids(tokenizer, protocol[name])
        for name in ("prefix", "victim", "suffix_a", "suffix_b")
    }
    expected = protocol["expected_token_counts"]
    observed = {name: len(values) for name, values in segments.items()}
    if observed != {name: int(value) for name, value in expected.items()}:
        raise ValueError(f"Qwen token counts differ: {observed}")
    checks = {}
    for suffix_name in ("suffix_a", "suffix_b"):
        suffix = segments[suffix_name]
        present = segments["prefix"] + segments["victim"] + suffix
        omitted = segments["prefix"] + suffix
        checks[f"present_{suffix_name}"] = (
            _input_ids(
                tokenizer,
                protocol["prefix"]
                + protocol["victim"]
                + protocol[suffix_name],
            )
            == present
        )
        checks[f"omitted_{suffix_name}"] = (
            _input_ids(
                tokenizer,
                protocol["prefix"] + protocol[suffix_name],
            )
            == omitted
        )
    if protocol["raw_and_segment_tokenizations_must_match"] and not all(
        checks.values()
    ):
        raise ValueError("raw and segment tokenization differ")
    return {"segments": segments, "boundary_checks": checks}


def prefill(
    model: Any,
    cache: Any,
    token_ids: Sequence[int],
    device: str,
):
    import torch

    if not token_ids:
        raise ValueError("Qwen prefill segment is empty")
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    offset = _cache_length(cache)
    position_ids = torch.arange(
        offset,
        offset + ids.shape[1],
        device=device,
    ).unsqueeze(0)
    with torch.inference_mode():
        return model(
            input_ids=ids,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=None,
            logits_to_keep=1,
            return_dict=True,
        )


def _l2norm(value: Any, eps: float = 1e-6):
    import torch

    return value * torch.rsqrt(
        (value * value).sum(dim=-1, keepdim=True) + eps
    )


class DeltaCapture:
    def __init__(self):
        self.enabled = False
        self.rows: dict[int, list[dict[str, Any]]] = {}


@contextmanager
def capture_delta_inputs(model: Any, layer_ids: Sequence[int]):
    """Capture the exact post-convolution DeltaNet kernel inputs."""

    layers = _language_model(model).layers
    capture = DeltaCapture()
    originals = {}
    for raw_layer_id in layer_ids:
        layer_id = int(raw_layer_id)
        module = layers[layer_id].linear_attn
        original = module.chunk_gated_delta_rule
        originals[layer_id] = original

        def wrapped(
            query,
            key,
            value,
            *args,
            _layer_id=layer_id,
            _original=original,
            **kwargs,
        ):
            captured = None
            if capture.enabled:
                captured = {
                    "q": _l2norm(query.detach()).cpu().float().clone(),
                    "k": _l2norm(key.detach()).cpu().float().clone(),
                    "v": value.detach().cpu().float().clone(),
                    "g": kwargs["g"].detach().cpu().float().clone(),
                    "beta": kwargs["beta"].detach().cpu().float().clone(),
                    "initial": kwargs["initial_state"]
                    .detach()
                    .cpu()
                    .float()
                    .clone(),
                }
            result = _original(query, key, value, *args, **kwargs)
            if captured is not None:
                captured["final"] = result[1].detach().cpu().float().clone()
                capture.rows.setdefault(_layer_id, []).append(
                    captured
                )
            return result

        module.chunk_gated_delta_rule = wrapped
    try:
        yield capture
    finally:
        for layer_id, original in originals.items():
            layers[layer_id].linear_attn.chunk_gated_delta_rule = original


def _transition(state: Any, key: Any, log_decay: Any, beta: Any):
    decayed = state * log_decay.exp().unsqueeze(-1).unsqueeze(-1)
    recalled = (decayed * key.unsqueeze(-1)).sum(dim=-2)
    return decayed - key.unsqueeze(-1) * (
        recalled * beta.unsqueeze(-1)
    ).unsqueeze(-2)


def _write(key: Any, value: Any, beta: Any):
    return key.unsqueeze(-1) * (value * beta.unsqueeze(-1)).unsqueeze(-2)


def _recurrence_step(
    state: Any,
    key: Any,
    value: Any,
    log_decay: Any,
    beta: Any,
):
    return _transition(state, key, log_decay, beta) + _write(
        key,
        value,
        beta,
    )


def _run_recurrence(initial: Any, captured: Mapping[str, Any]):
    state = initial.float().clone()
    for index in range(captured["k"].shape[1]):
        state = _recurrence_step(
            state,
            captured["k"][:, index],
            captured["v"][:, index],
            captured["g"][:, index],
            captured["beta"][:, index],
        )
    return state


def _relative_residual(candidate: Any, reference: Any) -> float:
    return float(
        (candidate.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-12)
    )


def analyze_receipt(
    prefix_state: Any,
    boundary_state: Any,
    present_capture: Mapping[str, Any],
    omitted_capture: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    present_initial = present_capture["initial"].float()
    omitted_initial = omitted_capture["initial"].float()
    present_initial_residual = _relative_residual(
        present_initial,
        boundary_state,
    )
    omitted_initial_residual = _relative_residual(
        omitted_initial,
        prefix_state,
    )
    boundary_delta = present_initial - omitted_initial
    frozen_present = _run_recurrence(present_initial, present_capture)
    frozen_without = _run_recurrence(omitted_initial, present_capture)
    direct_frozen_delta = frozen_present - frozen_without

    transported = boundary_delta.clone()
    for index in range(present_capture["k"].shape[1]):
        transported = _transition(
            transported,
            present_capture["k"][:, index],
            present_capture["g"][:, index],
            present_capture["beta"][:, index],
        )
    frozen_residual = float(
        (transported - direct_frozen_delta).norm()
        / direct_frozen_delta.norm().clamp_min(1e-12)
    )

    present_state = present_initial.clone()
    omitted_state = omitted_initial.clone()
    maximum_forcing_abs = 0.0
    maximum_forcing_scale = 0.0
    for index in range(present_capture["k"].shape[1]):
        pk = present_capture["k"][:, index]
        pv = present_capture["v"][:, index]
        pg = present_capture["g"][:, index]
        pb = present_capture["beta"][:, index]
        ok = omitted_capture["k"][:, index]
        ov = omitted_capture["v"][:, index]
        og = omitted_capture["g"][:, index]
        ob = omitted_capture["beta"][:, index]
        delta = present_state - omitted_state
        predicted = (
            _transition(delta, pk, pg, pb)
            + (
                _transition(omitted_state, pk, pg, pb)
                - _transition(omitted_state, ok, og, ob)
            )
            + (_write(pk, pv, pb) - _write(ok, ov, ob))
        )
        present_next = _recurrence_step(present_state, pk, pv, pg, pb)
        omitted_next = _recurrence_step(omitted_state, ok, ov, og, ob)
        actual = present_next - omitted_next
        maximum_forcing_abs = max(
            maximum_forcing_abs,
            float((predicted - actual).norm()),
        )
        maximum_forcing_scale = max(
            maximum_forcing_scale,
            float(actual.norm()),
        )
        present_state, omitted_state = present_next, omitted_next

    sequential_native_delta = present_state - omitted_state
    present_live_final = present_capture["final"].float()
    omitted_live_final = omitted_capture["final"].float()
    live_native_delta = present_live_final - omitted_live_final
    present_live_residual = _relative_residual(
        present_state,
        present_live_final,
    )
    omitted_live_residual = _relative_residual(
        omitted_state,
        omitted_live_final,
    )
    native_sequential_live_residual = _relative_residual(
        sequential_native_delta,
        live_native_delta,
    )
    native_sequential_live_state_scale_residual = float(
        (sequential_native_delta - live_native_delta).norm()
        / max(
            float(present_live_final.norm()),
            float(omitted_live_final.norm()),
            1e-12,
        )
    )
    live_conformance = max(
        present_initial_residual,
        omitted_initial_residual,
        present_live_residual,
        omitted_live_residual,
        native_sequential_live_state_scale_residual,
    )
    native_transport_residual = float(
        (transported - live_native_delta).norm()
        / live_native_delta.norm().clamp_min(1e-12)
    )
    return {
        "boundary_effect_norm": float(boundary_delta.norm()),
        "native_effect_norm": float(live_native_delta.norm()),
        "sequential_native_effect_norm": float(
            sequential_native_delta.norm()
        ),
        "frozen_transport_relative_residual": frozen_residual,
        "native_transport_relative_residual": native_transport_residual,
        "native_mismatch_to_frozen_floor_ratio": (
            native_transport_residual / max(frozen_residual, 1e-12)
        ),
        "forcing_decomposition_max_relative_residual": (
            maximum_forcing_abs / max(maximum_forcing_scale, 1e-12)
        ),
        "forcing_decomposition_max_absolute_residual": maximum_forcing_abs,
        "present_initial_state_relative_residual": present_initial_residual,
        "omitted_initial_state_relative_residual": omitted_initial_residual,
        "present_live_kernel_recurrence_relative_residual": (
            present_live_residual
        ),
        "omitted_live_kernel_recurrence_relative_residual": (
            omitted_live_residual
        ),
        "native_sequential_to_live_relative_residual": (
            native_sequential_live_residual
        ),
        "native_sequential_to_live_state_scale_relative_residual": (
            native_sequential_live_state_scale_residual
        ),
        "live_kernel_conformance_max_relative_residual": live_conformance,
        "finite": bool(
            torch.isfinite(live_native_delta).all()
            and torch.isfinite(sequential_native_delta).all()
            and torch.isfinite(transported).all()
            and torch.isfinite(present_live_final).all()
            and torch.isfinite(omitted_live_final).all()
        ),
    }


def _recurrent_by_layer(
    cache: Any,
    layer_ids: Sequence[int],
) -> dict[int, Any]:
    return {
        int(layer_id): _linear_state(cache, int(layer_id))[1]
        .detach()
        .cpu()
        .float()
        .clone()
        for layer_id in layer_ids
    }


def run_suffix_case(
    model: Any,
    segments: Mapping[str, Sequence[int]],
    suffix_name: str,
    probe_layers: Sequence[int],
    device: str,
) -> dict[str, Any]:
    suffix = segments[suffix_name]
    cache = _new_cache(model)
    prefill(model, cache, segments["prefix"], device)
    mark = snapshot_cache(model, cache)
    prefix_state = _recurrent_by_layer(cache, probe_layers)
    prefill(model, cache, segments["victim"], device)
    boundary_state = _recurrent_by_layer(cache, probe_layers)
    with capture_delta_inputs(model, probe_layers) as present_capture:
        present_capture.enabled = True
        present_output = prefill(model, cache, suffix, device)
    present_snapshot = _state_snapshot(
        model,
        cache,
        present_output.logits[:, -1],
    )
    present_recurrent = _recurrent_by_layer(cache, probe_layers)

    restore_cache(model, cache, mark)
    replay_output = prefill(model, cache, suffix, device)
    replay_snapshot = _state_snapshot(
        model,
        cache,
        replay_output.logits[:, -1],
    )

    omitted_cache = _new_cache(model)
    prefill(model, omitted_cache, segments["prefix"], device)
    with capture_delta_inputs(model, probe_layers) as omitted_capture:
        omitted_capture.enabled = True
        omitted_output = prefill(model, omitted_cache, suffix, device)
    omitted_snapshot = _state_snapshot(
        model,
        omitted_cache,
        omitted_output.logits[:, -1],
    )
    omitted_recurrent = _recurrent_by_layer(omitted_cache, probe_layers)

    repeat_cache = _new_cache(model)
    prefill(model, repeat_cache, segments["prefix"], device)
    repeat_output = prefill(model, repeat_cache, suffix, device)
    repeat_snapshot = _state_snapshot(
        model,
        repeat_cache,
        repeat_output.logits[:, -1],
    )

    receipt = {}
    effects = {}
    for layer_id in probe_layers:
        present_rows = present_capture.rows[int(layer_id)]
        omitted_rows = omitted_capture.rows[int(layer_id)]
        if len(present_rows) != 1 or len(omitted_rows) != 1:
            raise RuntimeError("suffix capture count differs")
        receipt[str(layer_id)] = analyze_receipt(
            prefix_state[int(layer_id)],
            boundary_state[int(layer_id)],
            present_rows[0],
            omitted_rows[0],
        )
        effects[int(layer_id)] = (
            present_recurrent[int(layer_id)]
            - omitted_recurrent[int(layer_id)]
        )

    return {
        "suffix": suffix_name,
        "present_vs_omitted": compare_snapshots(
            omitted_snapshot,
            present_snapshot,
            strict_shapes=False,
        ),
        "replay_vs_fresh_omission": compare_snapshots(
            omitted_snapshot,
            replay_snapshot,
        ),
        "repeat_vs_fresh_omission": compare_snapshots(
            omitted_snapshot,
            repeat_snapshot,
        ),
        "receipt_analysis": receipt,
        "_effects": effects,
    }


def _suffix_effect_variation(
    first: Mapping[int, Any],
    second: Mapping[int, Any],
) -> dict[str, Any]:
    rows = {}
    for layer_id in sorted(first):
        a, b = first[layer_id], second[layer_id]
        denominator = max(float(a.norm()), float(b.norm()), 1e-12)
        rows[str(layer_id)] = {
            "effect_a_norm": float(a.norm()),
            "effect_b_norm": float(b.norm()),
            "relative_difference": float((a - b).norm()) / denominator,
        }
    return rows


def _load_tiny_model(manifest: Mapping[str, Any]):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM,
    )

    config = AutoConfig.from_pretrained(
        manifest["model"]["id"],
        revision=manifest["model"]["revision"],
    ).text_config
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 4
    config.num_attention_heads = 4
    config.num_key_value_heads = 1
    config.head_dim = 16
    config.linear_key_head_dim = 16
    config.linear_value_head_dim = 16
    config.linear_num_key_heads = 2
    config.linear_num_value_heads = 4
    config.layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    config.rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [1, 1, 0],
        "mrope_interleaved": True,
    }
    config.mtp_num_hidden_layers = 0
    return Qwen3_5ForCausalLM(config).eval(), config


def _load_live_model(manifest: Mapping[str, Any], device: str):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    model_spec = manifest["model"]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_spec["id"],
        revision=model_spec["revision"],
        dtype=torch.bfloat16,
        attn_implementation=model_spec["attention_implementation"],
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).eval()
    return model.to(device), model.config


def _kernel_implementation() -> dict[str, Any]:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation

    return {
        "flash_linear_attention_available": (
            implementation.chunk_gated_delta_rule is not None
        ),
        "causal_conv1d_available": (
            implementation.causal_conv1d_fn is not None
        ),
        "delta_rule_fallback": (
            implementation.torch_chunk_gated_delta_rule.__name__
        ),
        "causal_conv1d_fallback": (
            implementation.torch_causal_conv1d_update.__name__
        ),
    }


def _validate_architecture(
    model: Any,
    manifest: Mapping[str, Any],
    *,
    tiny: bool,
) -> list[int]:
    contract = manifest["architecture_contract"]
    kinds = _layer_types(model)
    linear = [index for index, kind in enumerate(kinds) if kind == "linear_attention"]
    full = [index for index, kind in enumerate(kinds) if kind == "full_attention"]
    if tiny:
        return linear
    if linear != contract["linear_attention_layers"] or full != contract[
        "full_attention_layers"
    ]:
        raise ValueError("Qwen layer layout differs from frozen contract")
    kernels = _kernel_implementation()
    model_spec = manifest["model"]
    if kernels["flash_linear_attention_available"] != bool(
        model_spec["fast_linear_attention_kernels"]
    ):
        raise ValueError("Qwen linear-attention kernel mode differs")
    if kernels["causal_conv1d_available"] != bool(
        model_spec.get("fast_causal_conv1d_kernels", False)
    ):
        raise ValueError("Qwen causal-convolution kernel mode differs")
    return [int(value) for value in contract["probe_linear_layers"]]


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "transformers", "accelerate"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {"platform": platform.platform(), "versions": versions}


def _evaluate_protocol(
    manifest: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    protocol: Mapping[str, Any],
    *,
    device: str,
    tiny: bool,
) -> dict[str, Any]:
    tokenization = validate_tokenization(
        tokenizer,
        protocol,
    )
    segments = tokenization.pop("segments")
    probe_layers = _validate_architecture(model, manifest, tiny=tiny)
    if tiny:
        probe_layers = [probe_layers[0], probe_layers[-1]]
    started = time.perf_counter()
    first = run_suffix_case(
        model,
        segments,
        "suffix_a",
        probe_layers,
        device,
    )
    second = run_suffix_case(
        model,
        segments,
        "suffix_b",
        probe_layers,
        device,
    )
    effects_a = first.pop("_effects")
    effects_b = second.pop("_effects")
    suffix_variation = _suffix_effect_variation(effects_a, effects_b)
    acceptance = manifest["acceptance"]
    all_receipts = [
        row
        for case in (first, second)
        for row in case["receipt_analysis"].values()
    ]
    checks = {
        "replay_exact": all(
            case["replay_vs_fresh_omission"]["max_abs"]
            <= float(acceptance["same_schedule_replay_max_abs"])
            and case["replay_vs_fresh_omission"]["all_arrays_exact"]
            and case["replay_vs_fresh_omission"]["all_flags_equal"]
            and case["replay_vs_fresh_omission"]["all_values_finite"]
            and case["replay_vs_fresh_omission"]["shape_mismatch_count"] == 0
            for case in (first, second)
        ),
        "repeat_exact": all(
            case["repeat_vs_fresh_omission"]["max_abs"]
            <= float(acceptance["same_schedule_repeat_max_abs"])
            and case["repeat_vs_fresh_omission"]["all_arrays_exact"]
            and case["repeat_vs_fresh_omission"]["all_flags_equal"]
            and case["repeat_vs_fresh_omission"]["all_values_finite"]
            and case["repeat_vs_fresh_omission"]["shape_mismatch_count"] == 0
            for case in (first, second)
        ),
        "non_vacuous": all(
            (
                case["present_vs_omitted"]["max_abs"]
                >= float(
                    acceptance["present_vs_omitted_nonzero_max_abs_min"]
                )
                or case["present_vs_omitted"]["shape_mismatch_count"] > 0
            )
            for case in (first, second)
        ),
        "frozen_transport_closes": all(
            row["frozen_transport_relative_residual"]
            <= float(
                acceptance["frozen_input_transport_relative_residual_max"]
            )
            for row in all_receipts
        ),
        "forcing_closes": all(
            row["forcing_decomposition_max_relative_residual"]
            <= float(
                acceptance[
                    "forcing_decomposition_relative_residual_max"
                ]
            )
            for row in all_receipts
        ),
        "live_kernel_conformance": all(
            row["live_kernel_conformance_max_relative_residual"]
            <= float(
                acceptance.get(
                    "live_kernel_recurrence_relative_residual_max",
                    0.005,
                )
            )
            for row in all_receipts
        ),
        "native_transport_fails": sum(
            row["native_mismatch_to_frozen_floor_ratio"]
            >= float(
                acceptance["native_transport_mismatch_to_floor_ratio_min"]
            )
            and row["boundary_effect_norm"] > 0.0
            for row in all_receipts
        )
        >= int(
            acceptance["minimum_probe_layers_exceeding_native_mismatch"]
        ),
        "finite": (
            all(row["finite"] for row in all_receipts)
            and all(
                case["present_vs_omitted"]["all_values_finite"]
                and case["replay_vs_fresh_omission"]["all_values_finite"]
                and case["repeat_vs_fresh_omission"]["all_values_finite"]
                for case in (first, second)
            )
        ),
    }
    return {
        "tokenization": tokenization,
        "probe_layers": probe_layers,
        "cases": [first, second],
        "suffix_effect_variation": suffix_variation,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate(
    manifest: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    *,
    device: str,
    tiny: bool,
) -> dict[str, Any]:
    protocol = manifest["synthetic_protocol"]
    scenarios = protocol.get("scenarios")
    if not scenarios:
        return _evaluate_protocol(
            manifest,
            tokenizer,
            model,
            protocol,
            device=device,
            tiny=tiny,
        )
    rows = []
    for scenario in scenarios:
        scenario_protocol = {
            "prefix": protocol["prefix"],
            "victim": scenario["victim"],
            "suffix_a": protocol["suffix_a"],
            "suffix_b": protocol["suffix_b"],
            "expected_token_counts": {
                "prefix": protocol["expected_token_counts"]["prefix"],
                "victim": scenario["expected_victim_tokens"],
                "suffix_a": protocol["expected_token_counts"]["suffix_a"],
                "suffix_b": protocol["expected_token_counts"]["suffix_b"],
            },
            "raw_and_segment_tokenizations_must_match": protocol[
                "raw_and_segment_tokenizations_must_match"
            ],
        }
        row = _evaluate_protocol(
            manifest,
            tokenizer,
            model,
            scenario_protocol,
            device=device,
            tiny=tiny,
        )
        row["scenario_id"] = scenario["scenario_id"]
        rows.append(row)
    check_names = set(rows[0]["checks"])
    aggregate_checks = {
        name: all(row["checks"].get(name) is True for row in rows)
        for name in sorted(check_names)
    }
    minimum_scenarios = int(
        manifest["acceptance"].get("minimum_scenarios_pass", len(rows))
    )
    passed_scenarios = sum(row["all_checks_pass"] for row in rows)
    return {
        "scenarios": rows,
        "checks": aggregate_checks,
        "passed_scenarios": passed_scenarios,
        "required_scenarios": minimum_scenarios,
        "all_checks_pass": (
            all(aggregate_checks.values())
            and passed_scenarios >= minimum_scenarios
        ),
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {
        "frozen-before-live-weights",
        "frozen-before-validation-rerun",
    }:
        parser.error("Qwen protocol is not frozen")
    _seed_everything(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model"]["id"],
        revision=manifest["model"]["revision"],
    )
    if args.tiny:
        model, _ = _load_tiny_model(manifest)
        device = "cpu" if args.device == "mps" else args.device
        model = model.to(device)
    else:
        device = args.device
        model, _ = _load_live_model(manifest, device)
    result = {
        "schema": "qwen35-replay-audit-v2",
        "status": "running",
        "tiny_random_model": bool(args.tiny),
        "model": manifest["model"],
        "protocol": {
            "name": manifest["name"],
            "version": manifest["version"],
            "status": manifest["status"],
        },
        "manifest_sha256": _sha256_file(manifest_path),
        "implementation_sha256": _sha256_file(Path(__file__)),
        "seed": args.seed,
        "device": device,
        "environment": _environment(),
        "kernel_implementation": _kernel_implementation(),
        "result": evaluate(
            manifest,
            tokenizer,
            model,
            device=device,
            tiny=args.tiny,
        ),
    }
    result["status"] = (
        "completed"
        if result["result"]["all_checks_pass"]
        else "completed_with_failed_checks"
    )
    _atomic_write(output, result)
    print(f"wrote {output}", flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

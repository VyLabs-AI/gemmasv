"""Locate Gemma 3's global (full-context) attention layers.

Gemma 3 interleaves 5 local sliding-window layers per 1 global layer (5:1, starting
local). In HF transformers each ``Gemma3Attention`` carries ``is_sliding`` (True =
local sliding window, False = global full attention), derived from
``config.layer_types`` / ``sliding_window_pattern`` (default 6). We graft the SV
gate onto the *global* layers only -- those are the long-range memory.

Pure inspection: nothing here mutates the model.
"""
from __future__ import annotations

from typing import List, Tuple


def _decoder_layers(model) -> list:
    """Every submodule that looks like a Gemma3 decoder layer: one carrying a
    ``self_attn`` with an ``is_sliding`` flag. Duck-typed so it works for both
    Gemma3ForCausalLM (1B text) and the VLM wrappers."""
    layers = []
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        if attn is not None and hasattr(attn, "is_sliding"):
            layers.append(module)
    return layers


def find_global_attention_layers(model) -> List[Tuple[int, object]]:
    """Return ``(layer_idx, decoder_layer)`` for the global (non-sliding) layers,
    sorted by index. These are the attention layers we replace with the SV gate."""
    out: List[Tuple[int, object]] = []
    for layer in _decoder_layers(model):
        attn = layer.self_attn
        if getattr(attn, "is_sliding", True) is False:
            idx = getattr(attn, "layer_idx", getattr(layer, "layer_idx", -1))
            out.append((int(idx), layer))
    out.sort(key=lambda t: t[0])
    return out


def summarize_layer_types(model) -> str:
    """A 'L L L L L G ...' map of local/global layers, for a sanity check before
    grafting (verify the 5:1 pattern and the global count)."""
    rows = []
    for layer in _decoder_layers(model):
        attn = layer.self_attn
        idx = getattr(attn, "layer_idx", getattr(layer, "layer_idx", -1))
        rows.append((int(idx), "L" if getattr(attn, "is_sliding", True) else "G"))
    rows.sort()
    seq = " ".join(t for _, t in rows)
    n_global = sum(1 for _, t in rows if t == "G")
    return f"{len(rows)} layers ({n_global} global): {seq}"

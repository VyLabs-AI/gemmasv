"""Trainable, deletable support-vector memory graft for Gemma 3.

The package grafts the support-vector memory from the companion capability
paper into pretrained Gemma 3, then recovers language quality with low-rank
adaptation. Its float64 certificate path compares decrementing selected memory
entries with refitting the same retained contextualized keys.

Why Gemma 3: its attention is 5:1 interleaved -- 5 local sliding-window layers per
1 *global* full-context layer -- and only the global layers attend to long context
(Gemma 3 tech report, sec 5.2). That is exactly our hybrid's decomposition, so we:

    replace ONLY the global layers' attention with the SV gate;
    leave the local sliding-window layers untouched.

Consequence for the unlearning claim: long-range retention lives in the global
layers (now certified-forgettable); the retained local layers see only a bounded
~1024-token window, so a token beyond that window can persist *only* through the
SV-grafted global layers. The deletion guarantee is therefore scoped cleanly to
the model's long-range memory.

Pipeline:
    layer_select.find_global_attention_layers(model)   -> the global layers
    graft.graft_sv_into_gemma(model, ...)              -> swap their self_attn
    distill: attention_transfer (stage 1) -> LoRA recovery (stage 2)  (LoLCATs-style)

The recovered 1B path, output-level deletion, hosted demo, and data-free
contract suite are implemented. See ``gemma_sv/reproducibility`` for the
separate smoke, model, and paper-scale tiers.
"""
from __future__ import annotations

__all__ = [
    "find_global_attention_layers",
    "summarize_layer_types",
    "graft_sv_into_gemma",
    "SVGlobalAttention",
    "exact_forget_equivalence",
    "certified_reserve",
]


def __getattr__(name):
    # Lazy imports so ``import gemma_sv`` does not pull torch/transformers/mlx
    # (mirrors svattn/__init__.py): heavy deps load only on first real use.
    if name in ("find_global_attention_layers", "summarize_layer_types"):
        from . import layer_select
        return getattr(layer_select, name)
    if name == "SVGlobalAttention":
        from .sv_global_attention import SVGlobalAttention
        return SVGlobalAttention
    if name == "graft_sv_into_gemma":
        from .graft import graft_sv_into_gemma
        return graft_sv_into_gemma
    if name in ("exact_forget_equivalence", "certified_reserve"):
        from . import unlearn
        return getattr(unlearn, name)
    raise AttributeError(f"module 'gemma_sv' has no attribute {name!r}")

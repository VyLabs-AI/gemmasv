"""Distillation of the SV gate into Gemma 3 (two stages, LoLCATs-style).

Stage 1 -- attention transfer: with base weights frozen, make each grafted GLOBAL
layer reproduce the ORIGINAL global-attention output on the same input (layer-wise
MSE), so the SV gate learns to stand in for softmax.

Stage 2 -- LoRA recovery: attach LoRA to the attention projections and fine-tune the
grafted model end-to-end with the LM loss on a small token budget (~40M tokens in
LoLCATs) to recover quality.

This module contains the reusable losses and reference-capture mechanism.
``gemma_sv.run_distill`` provides the executed data, optimizer, checkpoint, and
evaluation pipeline.
"""
from __future__ import annotations

from typing import Dict, List

import torch


@torch.no_grad()
def capture_global_reference_outputs(model, batch: dict,
                                     layer_ids: List[int]) -> Dict[int, torch.Tensor]:
    """Run the UN-grafted model once and grab each requested global layer's
    attention output as the distillation target, via forward hooks on
    ``decoder_layer.self_attn``. Call BEFORE grafting (or on a reference copy)."""
    from .layer_select import find_global_attention_layers

    refs: Dict[int, torch.Tensor] = {}
    wanted = set(layer_ids)
    handles = []
    for idx, layer in find_global_attention_layers(model):
        if idx not in wanted:
            continue

        def _hook(_mod, _inp, out, _idx=idx):
            refs[_idx] = (out[0] if isinstance(out, tuple) else out).detach()

        handles.append(layer.self_attn.register_forward_hook(_hook))
    try:
        model(**batch)
    finally:
        for h in handles:
            h.remove()
    return refs


def attention_transfer_loss(model, batch: dict,
                            refs: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Stage-1 loss: mean MSE between each grafted global layer's output and its
    captured reference. Same hook mechanism, on the (now grafted) model."""
    from .layer_select import find_global_attention_layers

    got: Dict[int, torch.Tensor] = {}
    handles = []
    for idx, layer in find_global_attention_layers(model):
        if idx not in refs:
            continue

        def _hook(_mod, _inp, out, _idx=idx):
            got[_idx] = out[0] if isinstance(out, tuple) else out

        handles.append(layer.self_attn.register_forward_hook(_hook))
    try:
        model(**batch)
    finally:
        for h in handles:
            h.remove()
    return sum(torch.nn.functional.mse_loss(got[i], refs[i]) for i in refs) / max(len(refs), 1)


@torch.no_grad()
def calibrate_kpars(model, batch: dict) -> Dict[int, float]:
    """Per grafted global layer, the median key-distance RBF bandwidth from one
    forward (pre-hooks on ``_project_qkv``). This is the natural init for the
    learnable kernel -- the same heuristic the inference path uses, frozen as a
    starting point for stage 1. Requires the model to be grafted already."""
    from .layer_select import find_global_attention_layers

    inits: Dict[int, float] = {}
    handles = []
    for idx, layer in find_global_attention_layers(model):
        attn = layer.self_attn

        def _hook(_mod, args, kwargs, _idx=idx, _attn=attn):
            hs = args[0] if args else kwargs["hidden_states"]
            _, k, _ = _attn._project_qkv(hs, kwargs.get("position_embeddings"))
            inits[_idx] = _attn._kpar_for(k)

        handles.append(attn.register_forward_pre_hook(_hook, with_kwargs=True))
    try:
        model(**batch)
    finally:
        for h in handles:
            h.remove()
    return inits


def setup_stage1(model, batch: dict) -> List[torch.nn.Parameter]:
    """Stage-1 attention-transfer setup (LoLCATs-style, tiny trainable footprint):
    freeze ALL base weights, then make each grafted global layer's RBF bandwidth a
    trainable ``nn.Parameter`` initialised at that layer's median key distance.
    Returns the trainable params (the per-layer ``log_kpar``). Call AFTER grafting and
    AFTER capturing the teacher refs (``capture_global_reference_outputs``)."""
    from .layer_select import find_global_attention_layers

    for p in model.parameters():
        p.requires_grad_(False)
    inits = calibrate_kpars(model, batch)
    for idx, layer in find_global_attention_layers(model):
        layer.self_attn.enable_learnable_kernel(inits.get(idx, 1.0))
    return [p for p in model.parameters() if p.requires_grad]


# TODO(remaining for the full recipe):
#   data    : swap the smoke corpus in train_stage1 for a real stream (FineWeb-Edu
#             sample / enwik8 loader), ~tens of M tokens; needs MPS/GPU time.
#   fidelity: layer-WISE transfer (feed the teacher's per-layer input to each grafted
#             layer) to remove the end-to-end input drift for layers after the first
#             global; today's attention_transfer_loss is end-to-end (clean for the
#             first global layer, slightly drifted thereafter -- fine for the smoke).
#   stage 2 : attach PEFT LoRA to q/k/v/o_proj; train LM cross-entropy.
#   eval    : exact-unlearning at the MODEL-output level (decision-fn equivalence to a
#             never-saw-it baseline) + certified KV eviction vs H2O/SnapKV; contrast
#             with verify-based VeriCache/proveKV (we delete, and keep no full cache).

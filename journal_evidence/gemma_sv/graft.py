"""Graft the SV gate onto Gemma 3's global layers (in place), plus parameter
freeze/report helpers for the distillation stages.
"""
from __future__ import annotations

from typing import List

from .layer_select import find_global_attention_layers
from .sv_global_attention import SVGlobalAttention


def graft_sv_into_gemma(model, *, nu: float = 0.3, C=None, kpar=None, chunk: int = 128,
                        gate: bool = True, rope_on_keys: bool = True,
                        solver: str = "mlx", fista_iters: int = 80,
                        solver_seed: int = 0,
                        partition_tol: float = 1e-3,
                        readout: str = "softmax",
                        preserve_prefix_mass: bool = False,
                        per_boundary_box: bool = False) -> List[int]:
    """Replace every global layer's ``self_attn`` with an ``SVGlobalAttention``
    wrapping the original (projections/norms/RoPE reused). In place; returns the
    replaced layer indices. Call before distillation.

    ``nu`` sets the box ``C = 1/(nu*n)``; pass ``C`` to override the box
    directly. NOTE: capture reference outputs for attention transfer on the ORIGINAL
    model first (see distill.capture_global_reference_outputs), or on a copy, since
    this mutates the model.
    """
    replaced: List[int] = []
    for idx, layer in find_global_attention_layers(model):
        layer.self_attn = SVGlobalAttention(
            layer.self_attn, nu=nu, C=C, kpar=kpar, chunk=chunk, gate=gate,
            rope_on_keys=rope_on_keys, solver=solver, fista_iters=fista_iters,
            solver_seed=solver_seed,
            partition_tol=partition_tol, readout=readout,
            preserve_prefix_mass=preserve_prefix_mass,
            per_boundary_box=per_boundary_box)
        replaced.append(idx)
    return replaced


def set_stage1_attention_transfer(model) -> None:
    """Stage 1 (attention transfer): freeze ALL base weights; only the SV gate's
    learnable kernel params (added in distill.py when kpar/C become nn.Parameters)
    train. LoLCATs-style: tiny trainable footprint."""
    for p in model.parameters():
        p.requires_grad_(False)


def trainable_parameter_report(model) -> str:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"trainable {tr:,} / {tot:,} ({100 * tr / max(tot, 1):.3f}%)"

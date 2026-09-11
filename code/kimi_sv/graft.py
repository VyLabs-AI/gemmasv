"""Graft the certified SV gate onto Kimi Linear's global MLA layers.

Only the global layers are replaced. The 20 KDA layers are left alone: they are
the architecture's efficiency path, and a growing support set is the opposite of
the fixed-size recurrent state that motivates them.

Call this *after* the weights are loaded. Wrapping a layer nests its parameters
under ``self_attn.base.*``, which no longer matches the checkpoint's key names.
"""
from __future__ import annotations

from typing import Any, List, Optional

from .sv_mla_attention import SVMLAAttention
from .state import classify_layers


def graft_sv_into_kimi(
    model: Any,
    *,
    mode: str = "latent",
    nu: float = 0.3,
    C: Optional[float] = None,
    kpar: Optional[float] = None,
    chunk: int = 128,
    gate: bool = True,
    fista_iters: int = 80,
    preserve_prefix_mass: bool = False,
    collect_stats: bool = False,
    solver_seed: int = 0,
    layers: Optional[List[int]] = None,
) -> List[int]:
    """Replace each global layer's attention with an SV-gated one, in place.

    Returns the replaced layer indices. ``layers`` restricts the graft to a
    subset, which is how a single-layer feasibility check is run.
    """
    _, global_layers = classify_layers(model)
    targets = global_layers if layers is None else [i for i in global_layers if i in layers]

    for index in targets:
        layer = model.layers[index]
        layer.self_attn = SVMLAAttention(
            layer.self_attn,
            mode=mode,
            nu=nu,
            C=C,
            kpar=kpar,
            chunk=chunk,
            gate=gate,
            fista_iters=fista_iters,
            preserve_prefix_mass=preserve_prefix_mass,
            collect_stats=collect_stats,
            solver_seed=solver_seed,
        )
    return targets


def grafted_layers(model: Any) -> List[int]:
    return [
        i
        for i, layer in enumerate(model.layers)
        if isinstance(getattr(layer, "self_attn", None), SVMLAAttention)
    ]


def gate_stats(model: Any) -> dict:
    """Collected gate statistics per grafted layer."""
    return {
        i: model.layers[i].self_attn.stats
        for i in grafted_layers(model)
        if model.layers[i].self_attn.stats
    }


def set_drop_positions(model: Any, positions: Any) -> None:
    """Evict context positions from every grafted gate.

    Dropped positions leave the SVDD fit and carry exactly zero readout weight,
    which is the eviction/forgetting path at solver precision. ``positions`` is
    one list applied to every grafted layer, or a ``{layer_index: list}`` dict
    when each layer evicts its own set (matched-budget policy comparisons).
    """
    for i in grafted_layers(model):
        if positions is None:
            drop = None
        elif isinstance(positions, dict):
            got = positions.get(i)
            drop = None if got is None else list(got)
        else:
            drop = list(positions)
        model.layers[i].self_attn.drop_pos = drop

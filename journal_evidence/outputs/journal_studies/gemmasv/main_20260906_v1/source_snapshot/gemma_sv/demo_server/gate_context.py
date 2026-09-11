"""Model-wide serialization and exception-safe SV gate controls."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import threading
from typing import Mapping

from gemma_sv.sv_global_attention import SVRuntimeState


@dataclass(frozen=True)
class GateRequest:
    drop_pos: tuple[int, ...] | None = None
    scale_pos: tuple[int, ...] | None = None
    scale_factor: float = 1.0
    alpha_by_layer: Mapping[int, object] | None = None
    gate_floor: float = 0.0


class ModelGateController:
    """Serialize forwards that mutate shared ``SVGlobalAttention`` controls."""

    def __init__(self, layers: Mapping[int, object], lock: threading.RLock | None = None):
        self.layers = dict(layers)
        self.lock = lock or threading.RLock()

    @contextmanager
    def apply(self, request: GateRequest | None = None):
        request = request or GateRequest()
        with self.lock:
            with ExitStack() as stack:
                for layer_id, decoder_layer in self.layers.items():
                    attention = decoder_layer.self_attn
                    alpha = (
                        None
                        if request.alpha_by_layer is None
                        else request.alpha_by_layer.get(layer_id)
                    )
                    state = SVRuntimeState(
                        alpha_override=alpha,
                        drop_pos=request.drop_pos,
                        scale_pos=request.scale_pos,
                        scale_factor=request.scale_factor,
                        gate_floor=request.gate_floor,
                    )
                    stack.enter_context(attention.use_runtime_state(state))
                yield

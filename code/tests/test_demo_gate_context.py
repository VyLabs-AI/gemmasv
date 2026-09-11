from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gemma_sv.demo_server.gate_context import GateRequest, ModelGateController
from gemma_sv.sv_global_attention import SVRuntimeState


class FakeAttention:
    def __init__(self):
        self.state = SVRuntimeState()

    @contextmanager
    def use_runtime_state(self, state):
        previous = self.state
        self.state = state
        try:
            yield self
        finally:
            self.state = previous


def test_model_gate_controller_applies_per_layer_overrides_and_restores():
    a1, a2 = FakeAttention(), FakeAttention()
    controller = ModelGateController(
        {
            5: SimpleNamespace(self_attn=a1),
            11: SimpleNamespace(self_attn=a2),
        }
    )
    request = GateRequest(
        drop_pos=(3, 7),
        scale_factor=0.01,
        alpha_by_layer={5: {"gate": "five"}, 11: {"gate": "eleven"}},
    )

    with controller.apply(request):
        assert a1.state.drop_pos == (3, 7)
        assert a1.state.alpha_override == {"gate": "five"}
        assert a2.state.alpha_override == {"gate": "eleven"}

    assert a1.state == SVRuntimeState()
    assert a2.state == SVRuntimeState()


def test_model_gate_controller_restores_after_forward_failure():
    attention = FakeAttention()
    controller = ModelGateController({5: SimpleNamespace(self_attn=attention)})
    with pytest.raises(RuntimeError, match="forward failed"):
        with controller.apply(GateRequest(drop_pos=(1,))):
            raise RuntimeError("forward failed")
    assert attention.state == SVRuntimeState()

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from gemma_sv.demo_server.gemma_engine import (
    GemmaDemoEngine,
    GemmaRuntime,
    MemorySegment,
)
from gemma_sv.demo_server import gemma_engine as gemma_engine_module
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.demo_server.span import SelectedSpan
from gemma_sv.demo_server.state import SessionRecord
from gemma_sv.sv_global_attention import SVDecodeSession
from svattn.causal_sv_attention import sv_softmax_decode_step


def test_decode_session_clone_isolates_prefilled_memory_and_freezes_box():
    original = SVDecodeSession(
        kf=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        vf=torch.ones(1, 3, 4),
        gates={
            2: (
                torch.tensor([[0.4, 0.6]]),
                torch.tensor([True]),
            )
        },
        kpar=1.25,
        box_C=0.2,
        box_C_by_boundary={2: 0.4},
        frozen=True,
        frozen_boundary=2,
        batch=1,
    )

    branch = original.clone()
    branch.kf[0, 0, 0] = -99
    branch.gates[2][0][0, 0] = 0
    branch.box_C_by_boundary[2] = 0.1

    assert original.kf[0, 0, 0] == 0
    assert original.gates[2][0][0, 0] == torch.tensor(0.4)
    assert branch.box_C == original.box_C == 0.2
    assert original.box_C_by_boundary == {2: 0.4}
    assert branch.frozen and branch.frozen_boundary == 2


def test_decode_readout_can_freeze_long_range_boundary():
    keys = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    values = keys.clone()
    query = torch.tensor([[[1.0]]])
    gates = {
        2: (
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([True]),
        )
    }

    frozen = sv_softmax_decode_step(
        keys,
        values,
        query,
        gates,
        chunk=2,
        boundary_override=2,
    )
    ordinary = sv_softmax_decode_step(
        keys,
        values,
        query,
        gates,
        chunk=2,
    )
    assert not torch.allclose(frozen, ordinary)


def test_mass_preserving_gate_restores_prefix_share_and_keeps_zeros_inert():
    keys = torch.zeros(1, 4, 1)
    query = torch.zeros(1, 1, 1)
    gates = {
        2: (
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([True]),
        )
    }

    def read(reserve_value: float, *, preserve: bool):
        values = torch.tensor([[[1.0], [reserve_value], [0.0], [0.0]]])
        return sv_softmax_decode_step(
            keys,
            values,
            query,
            gates,
            chunk=2,
            boundary_override=2,
            preserve_prefix_mass=preserve,
        )

    crushed = read(100.0, preserve=False)
    preserved = read(100.0, preserve=True)
    changed_reserve = read(-100.0, preserve=True)

    assert torch.allclose(crushed, torch.tensor([[[1.0 / 3.0]]]))
    assert torch.allclose(preserved, torch.tensor([[[0.5]]]))
    assert torch.equal(preserved, changed_reserve)


def test_boundary_budget_violation_routes_to_full_repack():
    runtime = object.__new__(GemmaRuntime)
    runtime.loaded = True
    runtime.resolved_nu = 0.7
    runtime.config = SimpleNamespace(window=0)
    runtime.layers = {
        5: SimpleNamespace(
            self_attn=SimpleNamespace(
                chunk=8,
                per_boundary_box=True,
            )
        )
    }
    memory = SimpleNamespace(
        token_count=16,
        token_ids=tuple(range(16)),
        request=GateRequest(),
        layer_sessions={
            5: SimpleNamespace(
                box_C_by_boundary={8: 1.0 / (0.7 * 8)}
            )
        },
    )
    captured = {}

    def fake_prefill(ids, *, request):
        captured["ids"] = ids
        captured["request"] = request
        return SimpleNamespace(
            deleted_positions=(),
            deletion_kind=None,
            fallback_reason=None,
        )

    runtime.prefill_persistent = fake_prefill
    repacked = runtime.delete_persistent(memory, [0, 1, 2])

    assert captured["ids"] == list(range(3, 16))
    assert repacked.deletion_kind.endswith("full_repack_fallback")
    assert "maximum=0.300000" in repacked.fallback_reason


def test_certificate_runtime_forwards_frozen_session_objective(monkeypatch):
    runtime = object.__new__(GemmaRuntime)
    runtime.loaded = True
    runtime.layers = {
        5: SimpleNamespace(self_attn=SimpleNamespace(chunk=4)),
        9: SimpleNamespace(self_attn=SimpleNamespace(chunk=4)),
    }
    sessions = {
        5: SVDecodeSession(
            kf=torch.arange(24, dtype=torch.float32).reshape(1, 12, 2),
            vf=torch.zeros(1, 12, 2),
            kpar=1.25,
            box_C=0.3,
        ),
        9: SVDecodeSession(
            kf=torch.arange(24, dtype=torch.float32).reshape(1, 12, 2) + 1,
            vf=torch.zeros(1, 12, 2),
            kpar=2.5,
            box_C=0.3,
        ),
    }
    memory = SimpleNamespace(
        layer_sessions=sessions,
        token_count=12,
        fork=lambda: SimpleNamespace(),
    )
    before = {layer: session.kf.clone() for layer, session in sessions.items()}
    captured = {}

    def fake_overrides(keys, layer_ids, forget, total, heads, **kwargs):
        captured.update(
            {
                "keys": keys,
                "layer_ids": layer_ids,
                "forget": forget,
                "total": total,
                "heads": heads,
                **kwargs,
            }
        )
        return {
            "box_C": kwargs["box_C"],
            "exact": {layer: {} for layer in layer_ids},
            "refit": {layer: {} for layer in layer_ids},
        }

    monkeypatch.setattr(
        gemma_engine_module,
        "certificate_overrides",
        fake_overrides,
    )
    runtime.delete_persistent = lambda memory, positions, **kwargs: (
        kwargs["kind"],
        kwargs["alpha_by_layer"],
    )

    overrides, states = runtime.persistent_certificate_states(memory, [2])

    assert overrides["box_C"] == 0.3
    assert captured["box_C"] == 0.3
    assert captured["kpar_by_layer"] == {5: 1.25, 9: 2.5}
    assert captured["chunk"] == 4
    assert states["exact"][0] == "float64_exact"
    assert states["refit"][0] == "float64_refit"
    for layer, session in sessions.items():
        assert session.kpar == captured["kpar_by_layer"][layer]
        assert torch.equal(session.kf, before[layer])


@dataclass
class FakePersistent:
    kind: str
    input_digest: str

    def fork(self):
        return FakePersistent(self.kind, self.input_digest)


class FakePersistentRuntime:
    def __init__(self):
        self.config = SimpleNamespace(
            model_id="fake-gemma",
            window=2,
            copies=1,
            device="cpu",
            dtype="float32",
        )
        self.model_id = "fake-gemma"
        self.loaded = True
        self.load_seconds = 0.0
        self.prefills = 0
        self.legacy_calls = 0
        self.tokenizer = SimpleNamespace(
            decode=lambda ids: "BLUE" if ids == [0] else "token"
        )

    def pack_with_segments(self, selection, *, with_fact, icul=False):
        ids = list(range(10 if with_fact else 8))
        positions = [1] if with_fact else []
        return ids, positions, [
            MemorySegment("record", selection.text, 0, 2)
        ]

    def pack(self, selection, *, with_fact, icul=False):
        ids, positions, _ = self.pack_with_segments(
            selection, with_fact=with_fact, icul=icul
        )
        return ids, positions

    def target_ids(self, selected_value, audit_probe):
        return [0]

    def prefill_persistent(self, ids, *, request=None):
        self.prefills += 1
        kind = "never" if len(ids) == 8 else ("icul" if request else "present")
        return FakePersistent(kind, f"digest-{self.prefills}")

    def generate_persistent(self, memory, prompt, *, n_tokens=None):
        text = "none" if memory.kind in {"deleted", "never"} else "BLUE"
        return text, np.array([-0.1, -2.0])

    def score_persistent(self, memory, prompt, target_ids):
        mean = -4.0 if memory.kind in {"deleted", "never"} else -1.0
        return {
            "mean_log_probability": mean,
            "geometric_mean_probability": float(np.exp(mean)),
            "first_log_probs": np.array([-0.1, -2.0]),
            "elapsed_seconds": 0.0,
        }

    def delete_persistent(self, memory, positions, *, kind):
        return FakePersistent("deleted", memory.input_digest)

    def generate(self, *args, **kwargs):
        self.legacy_calls += 1
        raise AssertionError("legacy re-prefill path was called")

    def score_target(self, *args, **kwargs):
        self.legacy_calls += 1
        raise AssertionError("legacy re-prefill path was called")


def test_demo_queries_prefill_once_and_never_call_legacy_refill_path():
    runtime = FakePersistentRuntime()
    engine = GemmaDemoEngine(runtime, runtime)
    session = SessionRecord("session", 0.0, 0.0)
    selection = SelectedSpan(
        "The code is BLUE.",
        12,
        16,
        record_id="record-1",
    )

    ingest = engine.ingest(session, selection, "Code?", audit_target="BLUE")
    assert ingest["memory_prefilled_once"]
    assert runtime.prefills == 3

    recall = engine.recall(session)
    assert recall["memory_prefilled_once"]
    forgotten = engine.forget(session)
    assert forgotten["memory_prefilled_once"]
    attack = engine.attack(
        session,
        method="exact",
        kind="extraction",
    )
    assert attack["memory_prefilled_once"]
    assert runtime.prefills == 3
    assert runtime.legacy_calls == 0

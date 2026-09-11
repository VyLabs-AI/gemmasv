from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gemma_sv.eval_persistent_deletion_baselines import (
    Probe,
    _adapter_provenance,
    _precision_audit,
    _select_records,
)
from gemma_sv.persistent_deletion import (
    build_faithful_icul_context,
    build_synthetic_record_context,
    cache_delete_and_shift,
    deduplicated_tensor_storage_bytes,
    full_vocabulary_kl,
    kveraser_exclusion,
    method_storage_report,
    persistent_state_shape_signature,
    remove_token_positions,
    tensor_storage_report,
    token_ids_digest,
)
from gemma_sv.sv_global_attention import SVDecodeSession


class CharacterTokenizer:
    """Tiny offset-aware tokenizer used only for pure context tests."""

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        ids = [100 + (ord(character) % 97) for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids.insert(0, 1)
            offsets.insert(0, (0, 0))
        payload = {"input_ids": ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = offsets
        return SimpleNamespace(**payload) if not return_offsets_mapping else payload


def _record(index: int):
    return {
        "record_id": f"target-{index}",
        "question": f"Target question {index}?",
        "answer": f"field: secret-{index}; place: room-{index}.",
        "fields": [
            {"name": "field", "value": f"secret-{index}"},
            {"name": "place", "value": f"room-{index}"},
        ],
        "retain_question": f"Retained question {index}?",
        "retain_answer": f"field: public-{index}; place: hall-{index}.",
        "retain_field": {"name": "field", "value": f"public-{index}"},
    }


def test_context_is_literal_disjoint_edit_and_preserves_retained_record():
    spec = _record(0)
    context = build_synthetic_record_context(
        CharacterTokenizer(),
        spec,
        ["filler"],
        window=8,
        min_fillers=1,
        prefix_fillers=1,
        copies=2,
    )

    assert len(context.deletion_ranges) == 2
    assert context.edited_token_ids == remove_token_positions(
        context.original_token_ids,
        context.forget_positions,
    )
    retained_before = tuple(
        context.original_token_ids[position]
        for position in context.retain_positions
    )
    retained_after = tuple(
        context.edited_token_ids[position]
        for position in context.edited_retain_positions
    )
    assert retained_before == retained_after
    assert context.original_digest == token_ids_digest(
        context.original_token_ids
    )
    assert context.edited_digest == token_ids_digest(context.edited_token_ids)


def test_faithful_icul_is_seeded_wrong_answer_plus_four_correct_demos():
    records = [_record(index) for index in range(8)]
    first = build_faithful_icul_context(records[0], records, seed=17)
    second = build_faithful_icul_context(records[0], records, seed=17)
    different_seed = build_faithful_icul_context(records[0], records, seed=19)

    assert first == second
    assert first.wrong_answer_record_id != records[0]["record_id"]
    assert len(first.demonstration_record_ids) == 4
    assert len(set(first.demonstration_record_ids)) == 4
    assert records[0]["answer"] not in first.text
    assert first.text.count("Question:") == 5
    assert (
        first.wrong_answer_record_id,
        first.demonstration_record_ids,
    ) != (
        different_seed.wrong_answer_record_id,
        different_seed.demonstration_record_ids,
    )


def test_tensor_storage_accounting_deduplicates_views_and_references():
    base = torch.zeros(10, dtype=torch.float32)
    view = base[:5]
    clone = base.clone()

    report = tensor_storage_report(
        {"base": base, "view": view, "again": base, "clone": clone}
    )
    assert report["deduplicated_tensor_storage_bytes"] == 80
    assert report["unique_tensor_storages"] == 2
    assert deduplicated_tensor_storage_bytes([base, view]) == 40

    shared = method_storage_report({"tensor": base}, {"tensor": base})
    assert shared["incremental_tensor_storage_bytes"] == 0


class FakeDynamicLayer:
    def __init__(self, rows: int):
        self.keys = torch.arange(
            rows * 2,
            dtype=torch.float32,
        ).reshape(1, 1, rows, 2)
        self.values = self.keys + 100

    def get_seq_length(self):
        return self.keys.shape[-2]


class FakeSlidingLayer(FakeDynamicLayer):
    def __init__(self, rows: int, cumulative_length: int):
        super().__init__(rows)
        self.cumulative_length = cumulative_length

    def get_seq_length(self):
        return self.cumulative_length


@dataclass
class FakeMemory:
    past_key_values: object
    layer_sessions: dict[int, object]
    token_count: int
    input_digest: str
    request: object = None
    deleted_positions: tuple[int, ...] = ()
    deletion_kind: str | None = None
    kind: str = "present"

    def fork(self):
        return copy.deepcopy(self)


def _fake_memory():
    session = SVDecodeSession(
        kf=torch.arange(12, dtype=torch.float32).reshape(1, 6, 2),
        vf=torch.arange(12, dtype=torch.float32).reshape(1, 6, 2) + 20,
        gates={
            2: (
                torch.tensor([[0.4, 0.6]], dtype=torch.float32),
                torch.tensor([True]),
            ),
            4: (
                torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4]],
                    dtype=torch.float32,
                ),
                torch.tensor([True]),
            ),
        },
        kpar=1.0,
        box_C=0.2,
        frozen=True,
        frozen_boundary=4,
        batch=1,
    )
    native = SimpleNamespace(
        layers=[
            FakeDynamicLayer(6),
            FakeSlidingLayer(3, cumulative_length=6),
        ]
    )
    return FakeMemory(
        past_key_values=native,
        layer_sessions={5: session},
        token_count=6,
        input_digest="original-prefill",
    )


def test_cache_delete_shift_is_shape_correct_and_source_immutable():
    original = _fake_memory()
    source_signature = persistent_state_shape_signature(original)
    source_keys = original.layer_sessions[5].kf.clone()

    edited, diagnostics = cache_delete_and_shift(
        original,
        (1, 4),
        edited_token_ids=(10, 12, 13, 15),
    )

    assert persistent_state_shape_signature(original) == source_signature
    assert torch.equal(original.layer_sessions[5].kf, source_keys)
    assert original.token_count == 6
    assert edited.token_count == 4
    assert edited.input_digest == "original-prefill"
    assert edited.layer_sessions[5].kf.shape == (1, 4, 2)
    assert edited.layer_sessions[5].vf.shape == (1, 4, 2)
    assert edited.layer_sessions[5].frozen_boundary == 3
    assert set(edited.layer_sessions[5].gates) == {1, 3}
    assert edited.layer_sessions[5].gates[1][0].shape[-1] == 1
    assert edited.layer_sessions[5].gates[3][0].shape[-1] == 3

    full, sliding = edited.past_key_values.layers
    assert full.keys.shape[-2] == 4
    assert sliding.keys.shape[-2] == 2
    assert sliding.cumulative_length == 4
    assert diagnostics["solver_refit"] is False
    assert diagnostics["suffix_recomputed"] is False
    assert diagnostics["native_cache"][
        "resident_rows_removed_across_layers"
    ] == 3
    assert diagnostics["native_cache"][
        "nonresident_requests_across_layers"
    ] == 1
    assert diagnostics["logical_edited_input_digest"] == token_ids_digest(
        (10, 12, 13, 15)
    )


def test_full_vocabulary_kl_uses_repack_as_reference():
    reference = np.log(np.array([0.7, 0.2, 0.1]))
    same = full_vocabulary_kl(reference, reference.copy())
    shifted = full_vocabulary_kl(
        reference,
        np.log(np.array([0.2, 0.7, 0.1])),
    )
    assert same == pytest.approx(0.0, abs=1e-15)
    assert shifted > 0


def test_kveraser_is_machine_readable_exclusion_without_fake_metrics():
    exclusion = kveraser_exclusion()
    assert exclusion["status"] == "excluded"
    assert exclusion["evaluated"] is False
    assert exclusion["compatible"] is False
    assert {reason["code"] for reason in exclusion["reasons"]} == {
        "trained_eraser_backbone_mismatch",
        "contiguous_span_assumption",
        "sv_decode_session_incompatible",
    }
    assert "metrics" not in exclusion


def test_record_selectors_support_ids_indices_and_slices():
    records = [_record(index) for index in range(8)]
    all_selected = _select_records(
        records,
        record_start=0,
        record_count=None,
        selectors=(),
    )
    assert len(all_selected) == 8

    selected = _select_records(
        records,
        record_start=0,
        record_count=None,
        selectors=("target-4", "0:2", "-1"),
    )
    assert [index for index, _ in selected] == [0, 1, 4, 7]

    sliced = _select_records(
        records,
        record_start=2,
        record_count=3,
        selectors=(),
    )
    assert [index for index, _ in sliced] == [2, 3, 4]


class FakePrecisionRuntime:
    def __init__(self):
        self.config = SimpleNamespace(device="cpu")
        self.prefills = 0

    def prefill_persistent(self, token_ids):
        self.prefills += 1
        return FakeMemory(
            past_key_values=None,
            layer_sessions={},
            token_count=len(token_ids),
            input_digest=token_ids_digest(token_ids),
        )

    def persistent_certificate_states(self, memory, forget_positions):
        states = {
            name: FakeMemory(
                past_key_values=None,
                layer_sessions={},
                token_count=memory.token_count,
                input_digest=memory.input_digest,
                kind=name,
            )
            for name in ("exact", "refit", "decay")
        }
        return {
            "fixed_c_feasible": True,
            "box_C": 0.1,
            "n_solves": 2,
            "n_head_gates": 2,
            "n_fallback": 0,
            "used_refit_fallback": False,
            "max_functional_deviation": 0.0,
            "max_candidate_deviation": 0.0,
            "functional_tolerance": 1e-8,
            "feasibility": [],
            "fallback_details": [],
        }, states

    def delete_persistent(self, memory, forget_positions, *, kind):
        return FakeMemory(
            past_key_values=None,
            layer_sessions={},
            token_count=memory.token_count,
            input_digest=memory.input_digest,
            kind="proxy",
        )

    def score_persistent(self, memory, prompt, target_ids):
        probabilities = {
            "exact": np.array([0.70, 0.20, 0.10]),
            "refit": np.array([0.69, 0.21, 0.10]),
            "proxy": np.array([0.65, 0.25, 0.10]),
            "decay": np.array([0.30, 0.60, 0.10]),
        }[memory.kind]
        logs = np.log(probabilities)
        return {
            "first_log_probs": logs,
            "total_log_probability": float(logs[target_ids[0]]),
            "mean_log_probability": float(logs[target_ids[0]]),
            "geometric_mean_probability": float(probabilities[target_ids[0]]),
        }


def test_precision_audit_prefills_once_and_separates_three_kl_claims():
    runtime = FakePrecisionRuntime()
    context = SimpleNamespace(
        original_token_ids=(1, 2, 3, 4),
        forget_positions=(1,),
    )
    probes = [
        Probe(
            probe_id="deleted_field_0",
            kind="deleted",
            prompt="Question?",
            target="answer",
            target_ids=(0,),
        )
    ]
    report = _precision_audit(
        runtime,
        context,
        probes,
        warmup=0,
        repeats=1,
        operation_seed=3,
    )
    row = report["probe_rows"][0]
    assert runtime.prefills == 1
    assert report["model_forward_precision"] == "float64"
    assert row["kl_exact_vs_refit_nats"] > 0
    assert row["kl_proxy_vs_exact_nats"] > row["kl_exact_vs_refit_nats"]
    assert row["kl_decay_vs_refit_nats"] > row["kl_proxy_vs_exact_nats"]


def test_adapter_provenance_validates_training_seed(tmp_path):
    adapter = tmp_path / "recovered" / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter.bin").write_bytes(b"weights")
    (adapter.parent / "results.json").write_text(
        '{"schema":"gemma-sv-recovery-v2","config":{"seed":2},'
        '"stage2_stream":{"batches":3,"sha256":"stream"}}'
    )

    provenance = _adapter_provenance(str(adapter), 2)
    assert provenance["training_seed"] == 2
    assert provenance["stage2_stream"]["sha256"] == "stream"
    with pytest.raises(ValueError, match="does not match --seed"):
        _adapter_provenance(str(adapter), 1)

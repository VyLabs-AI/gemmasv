from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gemma_sv.eval_context_erasure_qa import (
    METHOD_IDS,
    admission_decomposition,
    evaluate_admission_record,
    evaluate_record,
    summarize_records,
)
from gemma_sv.rag_benchmark import (
    DATASET_REVISION,
    EMBEDDING_MODEL_REVISION,
    CorpusNode,
    RetrievalHit,
    build_counterfactual,
    build_document_specs,
    build_manifest,
    extract_2wiki_examples,
    make_in_memory_retriever,
    one_node_per_document,
    package_context,
    rehydrate_manifest,
    token_ids_sha256,
    validate_manifest,
)


class CharacterTokenizer:
    """Tiny offset-aware tokenizer with one zero-width special token."""

    is_fast = True

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        ids = [100 + ord(character) for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids.insert(0, 1)
            offsets.insert(0, (0, 0))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def _rows(count=6):
    return [
        {
            "id": f"example-{index}",
            "question": f"Which marker belongs to entry {index}?",
            "answer": f"marker-{index}",
            "type": "bridge",
            "supporting_facts": {
                "title": [f"Entry {index}"],
                "sent_id": [0],
            },
            "context": {
                "title": [f"Entry {index}", f"Background {index}"],
                "sentences": [
                    [
                        f"Entry {index} has marker-{index}.",
                        f"It was catalogued in year {2000 + index}.",
                    ],
                    [
                        f"Background {index} contains unrelated notes.",
                    ],
                ],
            },
        }
        for index in range(count)
    ]


def _fixture():
    examples = extract_2wiki_examples(_rows(), seed=19)
    documents = build_document_specs(examples)
    nodes = one_node_per_document(documents)
    examples_by_question = {
        example.question: example for example in examples
    }

    def retrieve(question, top_k):
        example = examples_by_question[question]
        own = next(
            node
            for node in nodes
            if node.source_example_id == example.example_id
            and node.is_gold_evidence
        )
        unrelated = sorted(
            (
                node
                for node in nodes
                if node.source_example_id != example.example_id
            ),
            key=lambda node: node.node_id,
        )[: top_k - 1]
        return [
            RetrievalHit(own, 1.0),
            *[
                RetrievalHit(node, 0.8 - offset * 0.1)
                for offset, node in enumerate(unrelated)
            ],
        ]

    return examples, nodes, retrieve


def test_deterministic_extraction_ids_hashes_and_retrieval_order():
    first = extract_2wiki_examples(_rows(), seed=19)
    second = extract_2wiki_examples(reversed(_rows()), seed=19)
    assert [example.example_id for example in first] == [
        example.example_id for example in second
    ]
    first_nodes = one_node_per_document(build_document_specs(first))
    second_nodes = one_node_per_document(build_document_specs(second))
    assert first_nodes == second_nodes
    assert len({node.node_id for node in first_nodes}) == len(first_nodes)
    assert len({node.document_id for node in first_nodes}) == len(
        {node.document_id for node in second_nodes}
    )

    examples, nodes, retrieve = _fixture()
    manifest_a = build_manifest(
        examples,
        nodes,
        retrieve,
        CharacterTokenizer(),
        requested_records=2,
        candidate_limit=6,
        seed=19,
        top_k=3,
        minimum_tokens_after_owned=0,
    )
    manifest_b = build_manifest(
        list(reversed(examples)),
        list(reversed(nodes)),
        retrieve,
        CharacterTokenizer(),
        requested_records=2,
        candidate_limit=6,
        seed=19,
        top_k=3,
        minimum_tokens_after_owned=0,
    )
    assert manifest_a == manifest_b
    for record in manifest_a["records"]:
        assert [hit["rank"] for hit in record["retrieval"]] == [1, 2, 3]
        assert record["gold_evidence"]["all_supporting_documents_retrieved"]


def test_manifest_serialization_has_no_source_text_and_rehydrates():
    examples, nodes, retrieve = _fixture()
    manifest = build_manifest(
        examples,
        nodes,
        retrieve,
        CharacterTokenizer(),
        requested_records=2,
        candidate_limit=6,
        seed=19,
        top_k=3,
        minimum_tokens_after_owned=0,
    )
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["contains_source_text"] is False
    for row in _rows():
        assert row["question"] not in serialized
        assert row["answer"] not in serialized
        for title in row["context"]["title"]:
            assert title not in serialized
        for sentence_group in row["context"]["sentences"]:
            for sentence in sentence_group:
                assert sentence not in serialized

    rehydrated = rehydrate_manifest(
        manifest,
        examples,
        nodes,
        CharacterTokenizer(),
    )
    assert [record.record_id for record in rehydrated] == [
        record["record_id"] for record in manifest["records"]
    ]


def test_token_ownership_is_offset_exact_and_literal_edit_preserves_retained():
    examples, nodes, retrieve = _fixture()
    example = examples[0]
    hits = tuple(retrieve(example.question, 3))
    retained = next(
        hit.node
        for hit in hits
        if hit.node.source_example_id != example.example_id
    )
    counterfactual = build_counterfactual(example, retained)
    context = package_context(
        CharacterTokenizer(),
        counterfactual,
        hits,
        retained.node_id,
    )

    forgotten = set(context.forget_positions)
    assert context.edited_token_ids == tuple(
        token_id
        for position, token_id in enumerate(context.original_token_ids)
        if position not in forgotten
    )
    start, end = context.owned_character_span
    assert context.edited_text == (
        context.original_text[:start] + context.original_text[end:]
    )
    assert context.answer_character_span[0] >= start
    assert context.answer_character_span[1] <= end
    assert context.original_text.count(counterfactual.distractor_answer) == 1
    assert counterfactual.distractor_answer not in context.edited_text
    retained_before = tuple(
        context.original_token_ids[position]
        for position in context.retained_positions
    )
    retained_after = tuple(
        context.edited_token_ids[position]
        for position in context.edited_retained_positions
    )
    assert retained_before == retained_after
    assert all(
        context.offset_mapping[position][1] > start
        and context.offset_mapping[position][0] < end
        for position in context.forget_positions
    )


def test_admission_keeps_target_and_retained_denominators_separate():
    present = {
        "gold": {"mean_log_probability": -3.0},
        "distractor": {"mean_log_probability": -1.0},
        "retained": {
            "mean_log_probability": -2.0,
            "first_token_rank": 50,
        },
    }
    repack = {
        "gold": {"mean_log_probability": -1.0},
        "distractor": {"mean_log_probability": -3.0},
        "retained": {
            "mean_log_probability": -2.0,
            "first_token_rank": 1,
        },
    }
    result = admission_decomposition(present, repack)
    assert result["primary_target_admission"]["admitted"] is True
    assert result["retained_availability"]["available"] is False
    assert result["joint_target_and_retained"]["admitted"] is False

    no_flip = copy.deepcopy(present)
    no_flip["gold"]["mean_log_probability"] = -0.5
    no_flip["retained"]["first_token_rank"] = 1
    result = admission_decomposition(no_flip, repack)
    assert result["primary_target_admission"]["admitted"] is False
    assert result["retained_availability"]["available"] is True


def test_manifest_tamper_detection_covers_scores_and_token_ownership():
    examples, nodes, retrieve = _fixture()
    manifest = build_manifest(
        examples,
        nodes,
        retrieve,
        CharacterTokenizer(),
        requested_records=2,
        candidate_limit=6,
        seed=19,
        top_k=3,
        minimum_tokens_after_owned=0,
    )
    score_tamper = copy.deepcopy(manifest)
    score_tamper["records"][0]["retrieval"][0]["score"] += 0.01
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_manifest(score_tamper)

    ownership_tamper = copy.deepcopy(manifest)
    ownership_tamper["records"][0]["context"]["ownership"][
        "forget_positions"
    ][0] += 1
    with pytest.raises(ValueError):
        validate_manifest(ownership_tamper)


def test_fake_embedding_and_retrieval_path_is_dependency_free():
    nodes = (
        CorpusNode("node-b", "doc-b", "example-b", 0, 0, "B", "beta"),
        CorpusNode("node-a", "doc-a", "example-a", 0, 0, "A", "alpha"),
        CorpusNode("node-c", "doc-c", "example-c", 0, 0, "C", "gamma"),
    )

    class FakeEmbedding:
        vectors = {
            "query": (1.0, 0.0),
            "alpha": (1.0, 0.0),
            "beta": (0.0, 1.0),
            "gamma": (1.0, 0.0),
        }

        def get_query_embedding(self, text):
            return self.vectors[text]

        def get_text_embedding(self, text):
            return self.vectors[text]

    retriever = make_in_memory_retriever(
        nodes,
        FakeEmbedding(),
        top_k=2,
    )
    hits = retriever("query", 2)
    assert [hit.node.node_id for hit in hits] == ["node-a", "node-c"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert DATASET_REVISION == (
        "a5d42f3b40d57a8c59fa10b2ac0c1829e4f73aba"
    )
    assert EMBEDDING_MODEL_REVISION == (
        "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )


@dataclass
class FakePersistentMemory:
    kind: str
    token_count: int
    input_digest: str
    past_key_values: object = None
    layer_sessions: dict = field(default_factory=dict)
    request: object = None
    deleted_positions: tuple[int, ...] = ()
    deletion_kind: str | None = None

    def fork(self):
        return copy.deepcopy(self)


class FakeDeletionRuntime:
    def __init__(self, original_token_count):
        self.original_token_count = original_token_count
        self.config = SimpleNamespace(device="cpu")
        self.tokenizer = CharacterTokenizer()

    def prefill_persistent(self, token_ids):
        return FakePersistentMemory(
            kind=(
                "present"
                if len(token_ids) == self.original_token_count
                else "repack"
            ),
            token_count=len(token_ids),
            input_digest=token_ids_sha256(token_ids),
        )

    def delete_persistent(self, memory, forget_positions, *, kind):
        edited = memory.fork()
        edited.kind = "proxy"
        edited.deleted_positions = tuple(forget_positions)
        edited.deletion_kind = kind
        return edited

    def persistent_certificate_states(self, memory, forget_positions):
        states = {}
        for kind in ("exact", "refit", "decay"):
            state = memory.fork()
            state.kind = kind
            state.deleted_positions = tuple(forget_positions)
            state.deletion_kind = kind
            states[kind] = state
        diagnostics = {
            "fixed_c_feasible": True,
            "box_C": 0.1,
            "n_solves": 1,
            "n_head_gates": 1,
            "n_fallback": 0,
            "used_refit_fallback": False,
            "max_functional_deviation": 0.0,
            "max_candidate_deviation": 0.0,
            "functional_tolerance": 1e-8,
            "feasibility": [],
            "fallback_details": [],
        }
        return diagnostics, states

    def score_persistent(self, memory, prompt, target_ids):
        rendered = "".join(
            chr(token_id - 100)
            for token_id in target_ids
            if token_id >= 100
        ).strip()
        if "title of the retained reference passage" in prompt:
            probe = "retained"
        elif rendered.startswith("marker-"):
            probe = "gold"
        else:
            probe = "distractor"

        if "In-context corrections:" in prompt:
            state = "icul"
        elif memory.deletion_kind == "cache_only_delete_and_shift_no_solver_refit":
            state = "cache"
        elif memory.deletion_kind == "coefficient_decay_0_01":
            state = "decay"
        else:
            state = memory.kind
        clean = state != "present"
        if probe == "gold":
            mean = -1.0 if clean else -3.0
        elif probe == "distractor":
            mean = -3.0 if clean else -1.0
        else:
            mean = -1.0

        logits = np.full(512, -4.0, dtype=np.float64)
        logits[int(target_ids[0])] = 2.0
        logits -= np.log(np.exp(logits).sum())
        return {
            "total_log_probability": mean * len(target_ids),
            "mean_log_probability": mean,
            "geometric_mean_probability": float(np.exp(mean)),
            "first_log_probs": logits,
        }


def test_functional_runner_executes_all_methods_with_fake_model_state(monkeypatch):
    examples, nodes, retrieve = _fixture()
    manifest = build_manifest(
        examples,
        nodes,
        retrieve,
        CharacterTokenizer(),
        requested_records=5,
        candidate_limit=6,
        seed=19,
        top_k=3,
        minimum_tokens_after_owned=0,
    )
    rehydrated = rehydrate_manifest(
        manifest,
        examples,
        nodes,
        CharacterTokenizer(),
    )
    runtime = FakeDeletionRuntime(
        len(rehydrated[0].context.original_token_ids)
    )
    from gemma_sv import eval_persistent_deletion_baselines as baseline

    monkeypatch.setattr(baseline, "_synchronize", lambda _device: None)
    result = evaluate_record(
        runtime,
        rehydrated[0],
        rehydrated,
        seed=7,
        warmup=0,
        repeats=1,
    )

    assert result["status"] == "completed"
    assert result["admission"]["primary_target_admission"]["admitted"]
    assert set(result["methods"]) == set(METHOD_IDS)
    assert all(
        result["methods"][method_id]["status"] == "completed"
        for method_id in METHOD_IDS
    )
    assert all(
        "full_vocabulary_kl_to_repack" in result["methods"][method_id]
        for method_id in METHOD_IDS
    )
    assert result["source_state_immutability"][
        "shape_signature_and_digest_unchanged"
    ]
    summary = summarize_records([result])
    assert summary["methods"]["full_repack"]["mean_end_to_end_median_seconds"] >= 0
    assert summary["methods"]["full_repack"]["mean_state_tensor_storage_bytes"] >= 0


def test_admission_only_runner_skips_deletion_methods(monkeypatch):
    examples, nodes, retrieve = _fixture()
    manifest = build_manifest(
        examples,
        nodes,
        retrieve,
        CharacterTokenizer(),
        requested_records=1,
        candidate_limit=6,
        seed=19,
        top_k=3,
        minimum_tokens_after_owned=0,
    )
    record = rehydrate_manifest(
        manifest,
        examples,
        nodes,
        CharacterTokenizer(),
    )[0]
    runtime = FakeDeletionRuntime(len(record.context.original_token_ids))
    from gemma_sv import eval_persistent_deletion_baselines as baseline

    monkeypatch.setattr(baseline, "_synchronize", lambda _device: None)
    result = evaluate_admission_record(
        runtime,
        record,
        seed=7,
        warmup=0,
        repeats=1,
    )

    assert result["status"] == "completed"
    assert result["admission_only"] is True
    assert result["admission"]["primary_target_admission"]["admitted"]
    assert result["methods"] == {}


def test_optional_requirements_are_exactly_pinned():
    requirements = (
        Path(__file__).parents[1] / "gemma_sv" / "requirements-rag.txt"
    ).read_text(encoding="utf-8")
    assert "llama-index-core==0.14.23" in requirements
    assert "llama-index-embeddings-huggingface==0.7.0" in requirements

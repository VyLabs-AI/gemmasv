from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from gemma_sv import eval_musique_context_erasure as musique_eval
from gemma_sv.musique_rag_benchmark import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    CorpusNode,
    RetrievalHit,
    build_counterfactual,
    build_document_specs,
    build_manifest,
    coarse_answer_type,
    extract_musique_examples,
    find_supporting_answer_occurrence,
    make_in_memory_retriever,
    one_node_per_document,
    package_context,
    parse_musique_row,
    rehydrate_manifest,
    rehydrate_manifest_from_rows,
    select_donor,
    tokens_after_owned,
    validate_manifest,
)


class CharacterTokenizer:
    """One-character tokens plus one zero-width special token."""

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


def _rows(count=8):
    long_tail = " ".join(
        f"background detail {index}" for index in range(90)
    )
    rows = []
    for index in range(count):
        rows.append(
            {
                "id": f"record-{index}",
                "question": f"Which code labels archive record {index}?",
                "answer": f"code-{index}",
                "answer_aliases": [f"CODE-{index}"],
                "answerable": True,
                "paragraphs": [
                    {
                        "idx": 0,
                        "title": f"Archive {index}",
                        "paragraph_text": (
                            f"Archive record {index} uses code-{index}. "
                            f"{long_tail}."
                        ),
                        "is_supporting": True,
                    },
                    {
                        "idx": 1,
                        "title": f"Unrelated {index}",
                        "paragraph_text": (
                            f"Unrelated record {index} has neutral notes."
                        ),
                        "is_supporting": False,
                    },
                ],
                "question_decomposition": [
                    {
                        "id": index,
                        "question": f"record {index} >> code",
                        "answer": f"code-{index}",
                        "paragraph_support_idx": 0,
                    }
                ],
            }
        )
    return rows


def _fixture():
    examples = extract_musique_examples(_rows(), seed=23)
    nodes = one_node_per_document(build_document_specs(examples))
    by_question = {example.question: example for example in examples}

    def retrieve(question, top_k):
        assert top_k == DEFAULT_TOP_K
        example = by_question[question]
        own = next(
            node
            for node in nodes
            if node.source_example_id == example.example_id
            and node.paragraph_index == 0
        )
        others = sorted(
            (
                node
                for node in nodes
                if node.node_id != own.node_id
                and node.paragraph_index == 1
            ),
            key=lambda node: node.node_id,
        )[: top_k - 1]
        return [
            RetrievalHit(own, 1.0),
            *[
                RetrievalHit(node, 0.9 - offset * 0.05)
                for offset, node in enumerate(others)
            ],
        ]

    return examples, nodes, retrieve


def _manifest(records=2):
    examples, nodes, retrieve = _fixture()
    manifest = build_manifest(
        examples,
        nodes,
        retrieve,
        CharacterTokenizer(),
        requested_records=records,
        candidate_limit=len(examples),
        seed=23,
        package_versions={
            "python": "test",
            "huggingface-hub": "test",
            "transformers": "test",
            "llama-index-core": "0.14.23",
            "llama-index-embeddings-huggingface": "0.7.0",
        },
    )
    return manifest, examples, nodes


def test_parse_standard_and_columnar_variants():
    standard = parse_musique_row(
        {
            "id": "standard",
            "question": "Who designed the test monument?",
            "answer": "Ada Lovelace",
            "answer_aliases": ["A. Lovelace"],
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 4,
                    "title": "Monument",
                    "paragraph_text": (
                        "The test monument was designed by Ada Lovelace."
                    ),
                    "is_supporting": False,
                }
            ],
            "question_decomposition": [
                {
                    "id": 1,
                    "question": "monument >> designer",
                    "answer": "Ada Lovelace",
                    "paragraph_support_idx": 4,
                }
            ],
        }
    )
    assert standard.paragraphs[0].is_supporting
    assert standard.answer_aliases == ("A. Lovelace",)
    assert find_supporting_answer_occurrence(standard).matched_answer == (
        "Ada Lovelace"
    )

    columnar = parse_musique_row(
        {
            "_id": "columnar",
            "question": "When was the sample founded?",
            "golden_answers": ["1912", "nineteen twelve"],
            "metadata": {
                "answerable": "true",
                "question_decomposition": {
                    "id": [2],
                    "question": ["sample >> founded"],
                    "answer": ["1912"],
                    "paragraph_support_idx": [7],
                },
            },
            "paragraphs": {
                "idx": [7],
                "title": ["Sample"],
                "paragraph_text": ["The sample was founded in 1912."],
                "is_supporting": [False],
            },
        }
    )
    assert columnar.answer == "1912"
    assert columnar.answer_aliases == ("nineteen twelve",)
    assert columnar.paragraphs[0].is_supporting


def test_same_type_donor_replaces_exact_real_support_occurrence():
    examples = extract_musique_examples(_rows(), seed=23)
    target = examples[0]
    occurrence = find_supporting_answer_occurrence(target)
    donor = select_donor(target, examples, seed=23)
    assert occurrence is not None
    assert donor is not None
    assert donor.example_id != target.example_id
    assert coarse_answer_type(donor.answer, donor.question) == (
        coarse_answer_type(target.answer, target.question)
    )
    assert donor.answer.casefold() not in "\n".join(
        paragraph.text.casefold() for paragraph in target.paragraphs
    )

    counterfactual = build_counterfactual(target, donor, occurrence)
    local_start = occurrence.answer_start - occurrence.sentence_start
    local_end = occurrence.answer_end - occurrence.sentence_start
    assert counterfactual.original_sentence == occurrence.sentence
    assert counterfactual.modified_sentence == (
        occurrence.sentence[:local_start]
        + donor.answer
        + occurrence.sentence[local_end:]
    )
    assert counterfactual.modified_sentence.endswith(".")
    assert counterfactual.distractor_answer == donor.answer


def test_candidate_order_document_and_manifest_ids_are_deterministic():
    first = extract_musique_examples(_rows(), seed=23)
    second = extract_musique_examples(reversed(_rows()), seed=23)
    assert [item.example_id for item in first] == [
        item.example_id for item in second
    ]
    assert one_node_per_document(build_document_specs(first)) == (
        one_node_per_document(build_document_specs(second))
    )

    manifest_a, examples, nodes = _manifest()
    _, _, retrieve = _fixture()
    manifest_b = build_manifest(
        list(reversed(examples)),
        list(reversed(nodes)),
        retrieve,
        CharacterTokenizer(),
        requested_records=2,
        candidate_limit=len(examples),
        seed=23,
        package_versions=manifest_a["package_versions"],
    )
    assert manifest_a == manifest_b
    assert len({item["record_id"] for item in manifest_a["records"]}) == 2
    assert manifest_a["retrieval_config"]["top_k"] == 6
    assert manifest_a["retrieval_config"]["chunk_size"] == 64
    assert manifest_a["retrieval_config"]["chunk_overlap"] == 0


def test_data_only_rejection_counts_are_frozen_before_retrieval():
    rows = _rows()
    rows.extend(
        [
            {
                **copy.deepcopy(rows[0]),
                "id": "unanswerable",
                "question": "Which code is unavailable?",
                "answer": "code-u",
                "answerable": False,
            },
            {
                **copy.deepcopy(rows[0]),
                "id": "polar",
                "question": "Is this a polar record?",
                "answer": "yes",
                "answer_aliases": [],
                "paragraphs": [
                    {
                        "idx": 0,
                        "title": "Polar",
                        "paragraph_text": "Yes, this is a polar record.",
                        "is_supporting": True,
                    }
                ],
            },
            {
                **copy.deepcopy(rows[0]),
                "id": "missing-occurrence",
                "question": "Which code is absent?",
                "answer": "absent-code",
                "answer_aliases": [],
                "paragraphs": [
                    {
                        "idx": 0,
                        "title": "Absent",
                        "paragraph_text": "This support omits the response.",
                        "is_supporting": True,
                    }
                ],
            },
        ]
    )
    examples = extract_musique_examples(rows, seed=23)
    nodes = one_node_per_document(build_document_specs(examples))
    by_question = {example.question: example for example in examples}

    def retrieve(question, top_k):
        example = by_question[question]
        own = next(
            node
            for node in nodes
            if node.source_example_id == example.example_id
            and node.paragraph_index == 0
        )
        backgrounds = sorted(
            (
                node
                for node in nodes
                if node.paragraph_index == 1
                and node.node_id != own.node_id
            ),
            key=lambda node: node.node_id,
        )[: top_k - 1]
        return [
            RetrievalHit(own, 1.0),
            *[
                RetrievalHit(node, 0.9 - index * 0.05)
                for index, node in enumerate(backgrounds)
            ],
        ]

    manifest = build_manifest(
        examples,
        nodes,
        retrieve,
        CharacterTokenizer(),
        requested_records=1,
        candidate_limit=len(examples),
        seed=23,
    )
    counts = manifest["selection_policy"]["data_only_rejection_counts"]
    assert counts["not_answerable"] == 1
    assert counts["polar_answer"] == 1
    assert counts["gold_not_found_exactly_in_supporting_sentence"] == 1


def test_real_retained_qa_and_counterfactual_follow_frozen_gold():
    manifest, examples, nodes = _manifest(records=1)
    record = rehydrate_manifest(
        manifest,
        examples,
        nodes,
        CharacterTokenizer(),
    )[0]
    assert record.retained_example.example_id != record.example.example_id
    assert record.retained_question == record.retained_example.question
    assert record.retained_answer == record.retained_example.answer
    assert record.retained_paragraph.is_supporting
    assert record.retained_answer.casefold() in (
        record.retained_paragraph.text.casefold()
    )
    assert "title of the retained" not in record.retained_question.casefold()

    gold_ids = set(
        manifest["records"][0]["gold_evidence"]["retrieved_node_ids"]
    )
    last_gold_end = max(
        record.context.original_text.index(hit.node.text) + len(hit.node.text)
        for hit in record.retrieval_hits
        if hit.node.node_id in gold_ids
    )
    counter_start = record.context.original_text.index(
        "[inserted counterfactual support sentence]"
    )
    assert counter_start > last_gold_end
    assert manifest["records"][0]["context"]["layout"][
        "insertion_after_retrieval_rank"
    ] == max(
        hit.rank for hit in record.retrieval_hits if hit.node.node_id in gold_ids
    )


def test_offset_ownership_literal_edit_and_retained_tokens_are_exact():
    manifest, examples, nodes = _manifest(records=1)
    record = rehydrate_manifest(
        manifest,
        examples,
        nodes,
        CharacterTokenizer(),
    )[0]
    context = record.context
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
    assert record.counterfactual.distractor_answer not in context.edited_text
    assert tokens_after_owned(context) >= 512
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
        context.offset_mapping[position][0] < end
        and context.offset_mapping[position][1] > start
        for position in context.forget_positions
    )


def test_manifest_is_exploratory_source_free_and_tamper_evident():
    manifest, _, _ = _manifest(records=1)
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["exploratory_after_2wiki_v1"] is True
    assert manifest["provenance"][
        "does_not_replace_or_suppress_2wiki_v1"
    ] is True
    assert manifest["dataset"]["dataset_id"] == DATASET_ID
    assert manifest["dataset"]["revision"] == DATASET_REVISION
    assert manifest["dataset"]["license"] == DATASET_LICENSE
    for row in _rows():
        assert row["question"] not in serialized
        assert row["answer"] not in serialized
        for paragraph in row["paragraphs"]:
            assert paragraph["title"] not in serialized
            assert paragraph["paragraph_text"] not in serialized
    assert "vectors" not in serialized

    score_tamper = copy.deepcopy(manifest)
    score_tamper["records"][0]["retrieval"][0]["score"] -= 0.01
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_manifest(score_tamper)
    token_tamper = copy.deepcopy(manifest)
    token_tamper["records"][0]["context"]["ownership"][
        "forget_positions"
    ][0] += 1
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_manifest(token_tamper)


def test_hydration_uses_trace_and_rejects_source_or_node_drift():
    manifest, examples, nodes = _manifest(records=1)
    hydrated = rehydrate_manifest_from_rows(
        manifest,
        _rows(),
        CharacterTokenizer(),
        nodes=nodes,
    )
    assert hydrated[0].record_id == manifest["records"][0]["record_id"]

    changed_examples = list(examples)
    target_index = next(
        index
        for index, example in enumerate(changed_examples)
        if example.example_id
        == manifest["records"][0]["target_source"]["source_id"]
    )
    target = changed_examples[target_index]
    changed_paragraph = replace(
        target.paragraphs[0],
        paragraph_text=target.paragraphs[0].paragraph_text + " Drift.",
    )
    changed_examples[target_index] = replace(
        target,
        paragraphs=(changed_paragraph, *target.paragraphs[1:]),
    )
    with pytest.raises(ValueError, match="source data drifted"):
        rehydrate_manifest(
            manifest,
            changed_examples,
            nodes,
            CharacterTokenizer(),
        )

    changed_nodes = list(nodes)
    changed_nodes[0] = replace(
        changed_nodes[0],
        text=changed_nodes[0].text + " drift",
    )
    with pytest.raises(ValueError, match="corpus nodes drifted"):
        rehydrate_manifest(
            manifest,
            examples,
            changed_nodes,
            CharacterTokenizer(),
        )


def test_strict_admission_requires_each_margin_and_keeps_retained_separate():
    present = {
        "gold": {"mean_log_probability": -1.04},
        "distractor": {"mean_log_probability": -1.00},
        "retained": {
            "mean_log_probability": -1.0,
            "first_token_rank": 20,
        },
    }
    repack = {
        "gold": {"mean_log_probability": -1.00},
        "distractor": {"mean_log_probability": -1.04},
        "retained": {
            "mean_log_probability": -1.0,
            "first_token_rank": 1,
        },
    }
    result = musique_eval.strict_admission_decomposition(present, repack)
    assert result["primary_target_admission"]["admitted"] is False
    assert result["retained_availability"]["available"] is False

    present["gold"]["mean_log_probability"] = -1.05
    repack["distractor"]["mean_log_probability"] = -1.05
    present["retained"]["first_token_rank"] = 10
    result = musique_eval.strict_admission_decomposition(present, repack)
    assert result["primary_target_admission"]["admitted"] is True
    assert result["retained_availability"]["available"] is True


def test_admission_runtime_wrapper_replaces_shared_combined_gate(monkeypatch):
    shared_result = {
        "record_id": "musique-rag-v2-test",
        "source_example_id": "record-0",
        "status": "completed",
        "admission_only": True,
        "admission": {
            "primary_target_admission": {"admitted": True},
        },
        "references": {
            "present_control": {
                "scores": {
                    "gold": {"mean_log_probability": -1.04},
                    "distractor": {"mean_log_probability": -1.0},
                    "retained": {
                        "mean_log_probability": -1.0,
                        "first_target_token_rank": 1,
                    },
                }
            },
            "full_repack": {
                "scores": {
                    "gold": {"mean_log_probability": -1.0},
                    "distractor": {"mean_log_probability": -1.04},
                    "retained": {
                        "mean_log_probability": -1.0,
                        "first_target_token_rank": 1,
                    },
                }
            },
        },
        "methods": {},
    }
    monkeypatch.setattr(
        musique_eval,
        "_shared_evaluate_admission_record",
        lambda *_args, **_kwargs: copy.deepcopy(shared_result),
    )
    result = musique_eval.evaluate_admission_record(
        object(),
        object(),
        seed=0,
        warmup=0,
        repeats=1,
    )
    assert result["admission"]["primary_target_admission"][
        "admitted"
    ] is False
    assert result["exploratory_after_2wiki_v1"] is True
    summary = musique_eval.summarize_records([result])
    assert summary["completed_records"] == 1
    assert summary["primary_target_admitted"] == 0


def test_fake_embedding_retrieval_is_canonical_and_download_free():
    nodes = tuple(
        CorpusNode(
            node_id=f"node-{letter}",
            document_id=f"doc-{letter}",
            source_example_id=f"record-{letter}",
            paragraph_index=0,
            chunk_index=0,
            title=letter,
            text=letter,
        )
        for letter in ("b", "a", "c", "d", "e", "f")
    )

    class FakeEmbedding:
        def get_query_embedding(self, _text):
            return (1.0, 0.0)

        def get_text_embedding(self, text):
            return (1.0, 0.0) if text in {"a", "c"} else (0.0, 1.0)

    retriever = make_in_memory_retriever(
        nodes,
        FakeEmbedding(),
        top_k=DEFAULT_TOP_K,
    )
    hits = retriever("query", DEFAULT_TOP_K)
    assert [hit.node.node_id for hit in hits[:2]] == ["node-a", "node-c"]
    assert [hit.rank for hit in hits] == list(range(1, 7))
    assert DEFAULT_TOP_K == 6
    assert DEFAULT_CHUNK_SIZE == 64
    assert DEFAULT_CHUNK_OVERLAP == 0


def test_package_context_rejects_missing_gold_and_revision_mismatch():
    examples, nodes, retrieve = _fixture()
    target = examples[0]
    occurrence = find_supporting_answer_occurrence(target)
    donor = select_donor(target, examples, seed=23)
    assert occurrence is not None and donor is not None
    counterfactual = build_counterfactual(target, donor, occurrence)
    hits = tuple(retrieve(target.question, DEFAULT_TOP_K))
    retained = examples[1]
    retained_occurrence = find_supporting_answer_occurrence(retained)
    assert retained_occurrence is not None
    retained_paragraph = next(
        paragraph
        for paragraph in retained.paragraphs
        if paragraph.paragraph_index == retained_occurrence.paragraph_index
    )
    with pytest.raises(ValueError, match="no frozen retrieved gold"):
        package_context(
            CharacterTokenizer(),
            counterfactual,
            hits,
            (),
            retained,
            retained_paragraph,
        )

    manifest, _, _ = _manifest(records=1)
    musique_eval.verify_tokenizer_revision(
        manifest,
        model_id=manifest["tokenizer"]["model_id"],
        model_revision=manifest["tokenizer"]["revision"],
    )
    runtime = SimpleNamespace(
        resolved_model_revision=manifest["tokenizer"]["revision"],
        tokenizer=SimpleNamespace(
            init_kwargs={
                "_commit_hash": manifest["tokenizer"]["revision"],
            }
        ),
    )
    musique_eval.verify_loaded_tokenizer_revision(runtime, manifest)
    with pytest.raises(ValueError, match="must match"):
        musique_eval.verify_tokenizer_revision(
            manifest,
            model_id=manifest["tokenizer"]["model_id"],
            model_revision="wrong-revision",
        )
    runtime.resolved_model_revision = "wrong-revision"
    with pytest.raises(RuntimeError, match="loaded runtime revision"):
        musique_eval.verify_loaded_tokenizer_revision(runtime, manifest)

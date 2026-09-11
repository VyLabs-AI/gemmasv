from __future__ import annotations

import copy
import json

import pytest

from gemma_sv.ruler_erasure_benchmark import (
    RULER_REVISION,
    as_context_erasure_records,
    build_manifest,
    generate_record,
    rehydrate_manifest,
    validate_manifest,
)


class CharacterTokenizer:
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


def _manifest():
    return build_manifest(
        CharacterTokenizer(),
        records=2,
        seed=41,
        minimum_tokens=600,
        minimum_tokens_after_owned=300,
    )


def test_controlled_manifest_is_deterministic_source_free_and_predeclared():
    first = _manifest()
    second = _manifest()

    assert first == second
    assert first["contains_source_text"] is False
    assert first["provenance"]["revision"] == RULER_REVISION
    assert first["official_task_parameters"] == {
        "type_haystack": "essay",
        "type_needle_k": "words",
        "type_needle_v": "numbers",
        "num_needle_k": 4,
        "num_needle_v": 1,
        "num_needle_q": 1,
    }
    assert first["generation"]["fixed_before_model_evaluation"] is True
    assert first["generation"]["model_outputs_used"] is False
    assert first["erasure_adaptation"]["natural_qa_trigger"] == {
        "manifest_integrity_sha256": (
            "b066a00b9472ba17831b6defeca375bafd8a491053632d1d3f93ae685466d5ee"
        ),
        "primary_target_admission": "1/8",
        "retained_availability": "1/8",
        "joint_admission": "0/8",
    }

    runtime = generate_record(
        CharacterTokenizer(),
        0,
        seed=41,
        minimum_tokens=600,
        minimum_tokens_after_owned=300,
    )["runtime"]
    serialized = json.dumps(first, sort_keys=True)
    for value in (
        runtime["question"],
        runtime["gold_answer"],
        runtime["distractor_answer"],
        runtime["retained_question"],
        runtime["retained_answer"],
        runtime["context"].original_text,
    ):
        assert value not in serialized


def test_controlled_owned_needle_is_literal_and_restores_gold_context():
    record = generate_record(
        CharacterTokenizer(),
        0,
        seed=41,
        minimum_tokens=600,
        minimum_tokens_after_owned=300,
    )
    runtime = record["runtime"]
    context = runtime["context"]

    assert runtime["distractor_answer"] in context.original_text
    assert runtime["distractor_answer"] not in context.edited_text
    assert runtime["gold_answer"] in context.edited_text
    assert runtime["retained_answer"] in context.edited_text
    forgotten = set(context.forget_positions)
    assert context.edited_token_ids == tuple(
        token_id
        for index, token_id in enumerate(context.original_token_ids)
        if index not in forgotten
    )
    assert (
        len(context.original_token_ids) - max(context.forget_positions) > 300
    )


def test_controlled_manifest_rehydrates_and_rejects_tampering():
    manifest = _manifest()
    records = rehydrate_manifest(manifest, CharacterTokenizer())

    assert len(records) == 2
    assert [record["record_id"] for record in records] == [
        record["record_id"] for record in manifest["records"]
    ]
    adapted = as_context_erasure_records(manifest, CharacterTokenizer())
    assert adapted[0].example.question == records[0]["runtime"]["question"]
    assert (
        adapted[0].counterfactual.distractor_answer
        == records[0]["runtime"]["distractor_answer"]
    )

    tampered = copy.deepcopy(manifest)
    tampered["records"][0]["context"]["ownership"]["forget_positions"][0] += 1
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_manifest(tampered)

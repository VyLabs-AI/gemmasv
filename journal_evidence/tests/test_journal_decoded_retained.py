import copy
import json

import pytest

from journal_studies.gemmasv.analyze_decoded_retained import METHODS, analyze, condition_score, validate_histories


def make_source():
    conditions = json.loads(METHODS.read_text())["conditions"]
    histories = []
    for index in range(96):
        histories.append({"history_index": index, "cluster_index": index // 3, "variant_index": index % 3,
            "completion": {"history_completed": True, "condition_completion": {c: True for c in conditions}},
            "reproducibility_checks": {"conditions": {c: {"retained": True} for c in conditions}},
            "retained": {"conditions": {c: {"available": True, "directional_correctness": {
                "deterministic_any": index % 2 == 0, "legacy_registered_casefold_substring": index % 2 == 0}} for c in conditions}}})
    return {"contains_source_text": False, "contains_model_generated_text": False,
        "contains_source_identifiers": False, "contains_token_arrays": False, "histories": histories}


def test_cluster_and_duplicate_validation_prevents_pseudoreplication():
    methods = json.loads(METHODS.read_text())
    source = make_source()
    assert len(validate_histories(source, methods)) == 96
    source["histories"][1]["variant_index"] = 0
    with pytest.raises(ValueError, match="duplicate"):
        validate_histories(source, methods)
    source = make_source()
    source["histories"].pop()
    with pytest.raises(ValueError, match="96"):
        validate_histories(source, methods)


def test_unavailable_response_is_counted_not_removed():
    methods = json.loads(METHODS.read_text())
    source = make_source()
    source["histories"][0]["retained"]["conditions"]["exact_decrement_or_refit_policy"]["available"] = False
    result = analyze(source, methods, resamples=100)
    condition = result["conditions"]["deterministic_any"]["exact_decrement_or_refit_policy"]
    assert condition["all_history_denominator"] == 96
    assert condition["scorable_histories"] == 95
    assert condition["correct_histories"] == 47
    pair = result["comparisons"][0]
    assert pair["both_scorable_histories"] == 95
    assert pair["one_or_both_unscorable_histories"] == 1
    assert sum(pair["operational_paired_table"].values()) == 96
    assert pair["difference"]["estimate"] == pytest.approx(-1 / 96)


def test_nonreproducible_or_missing_endpoint_is_unscorable():
    history = make_source()["histories"][0]
    history["reproducibility_checks"]["conditions"]["present"]["retained"] = False
    result = condition_score(history, "present", "deterministic_any")
    assert not result["scorable"]
    assert result["operational_correctness"] == 0
    assert "retained_repeat_not_reproduced" in result["failure_reasons"]
    history = make_source()["histories"][0]
    history["retained"]["conditions"]["present"]["directional_correctness"]["deterministic_any"] = "yes"
    with pytest.raises(ValueError, match="boolean"):
        condition_score(history, "present", "deterministic_any")

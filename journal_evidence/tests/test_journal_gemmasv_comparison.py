import json
import pytest

from journal_studies.gemmasv.run_comparison import (PROTOCOL_PATH, compact_failure,
    literal_delete, paired_ratio_summary, summarize)


def test_literal_delete_keeps_order_and_removes_disjoint_spans():
    assert literal_delete([10, 11, 12, 13, 14, 15, 16], [1, 2, 5]) == (10, 13, 14, 16)
    with pytest.raises(ValueError):
        literal_delete([1, 2], [2])


def test_paired_ratio_uses_records_and_preserves_missing_denominator():
    result = paired_ratio_summary([(20, 10), (9, 3)], expected_records=8, samples=1000)
    assert result["paired_records"] == 2
    assert result["expected_records"] == 8
    assert result["complete_cohort"] is False
    assert result["geometric_mean_ratio"] == pytest.approx(6 ** .5)
    assert result["bootstrap_95_percentile_interval"] == pytest.approx([2, 3])
    assert paired_ratio_summary([(1, 2)])["bootstrap_95_percentile_interval"] is None
    with pytest.raises(ValueError):
        paired_ratio_summary([(1, 0)])


def test_compact_failure_never_serializes_source_text():
    result = compact_failure(ValueError("sensitive source text"))
    assert "sensitive" not in json.dumps(result)
    assert result["error_type"] == "ValueError"


def test_frozen_protocol_keeps_current_recipe_and_qualifies_reuse():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    assert protocol["configuration"]["adapter"] is None
    assert protocol["configuration"]["nu"] == .7
    assert protocol["configuration"]["preserve_prefix_mass"]
    assert protocol["configuration"]["per_boundary_box"]
    assert protocol["cohort"]["exclude_by_admission"] is False
    assert "Post hoc reuse" in protocol["cohort_status"]
    assert "not decoded leakage" in protocol["probes"]


def test_failed_reference_does_not_turn_partial_timings_into_complete_results():
    report = {"selected_record_ids": ["r0"], "smoke": True,
        "protocol": {"arms": {"graft_masked_refit_proxy": "proxy"}},
        "records": [{"record_id": "r0", "arms": {"graft_masked_refit_proxy": {
            "status": "completed", "timing": {}}}}]}
    summary = summarize(report)
    assert summary["arms"]["graft_masked_refit_proxy"]["completed"] == 0
    assert summary["arms"]["graft_masked_refit_proxy"]["measurement_without_valid_reference"] == 1

import json
import pytest

from journal_studies.gemmasv.analyze_results import (ADDENDUM, ROOT, analyze,
    bootstrap_sensitivity, comparison_summary, group_manifest, index_results, paired_record)


def manifest():
    return json.loads((ROOT / "gemma_sv/benchmarks/whole_record_confirm_v2.json").read_text())


def test_source_blocks_are_explicit_and_not_inferred_from_result_order():
    source = manifest()
    grouping = group_manifest(source, [320, 340, 360, 380])
    assert len(grouping) == 8
    assert grouping["tofu-final-340-b"] == 340
    source["records"][0]["forget_indices"][0] = 380
    with pytest.raises(ValueError, match="exactly one"):
        group_manifest(source, [320, 340, 360, 380])


def test_record_and_cluster_resampling_differ_for_correlated_pairs():
    result = bootstrap_sensitivity([-2, -2, -2, -2, 2, 2, 2, 2], [0, 0, 1, 1, 2, 2, 3, 3])
    assert result["estimate"] == 0
    assert result["record_percentile_95"] == pytest.approx([-1.5, 1.5])
    assert result["source_block_percentile_95"] == pytest.approx([-2, 2])
    ratio = bootstrap_sensitivity([2, 2, 8, 8], [0, 0, 1, 1], geometric=True)
    assert ratio["estimate"] == pytest.approx(4)
    assert ratio["source_block_percentile_95"] == pytest.approx([2, 8])


def test_empty_duplicate_and_unknown_results_are_rejected():
    for report in (
        {"selected_record_ids": [], "records": []},
        {"selected_record_ids": ["r", "r"], "records": [{"record_id": "r"}]},
        {"selected_record_ids": ["r"], "records": [{"record_id": "r"}, {"record_id": "r"}]},
        {"selected_record_ids": ["other"], "records": [{"record_id": "other"}]},
    ):
        with pytest.raises(ValueError):
            index_results(report, {"r"})
    with pytest.raises(ValueError):
        bootstrap_sensitivity([], [])


def test_failed_and_missing_records_remain_in_analysis_denominators():
    addendum = json.loads(ADDENDUM.read_text())
    report = {"smoke": False, "selected_record_ids": [r["record_id"] for r in manifest()["records"]],
        "records": [{"record_id": "tofu-final-320-a", "arms": {"base_present": {"status": "failed"}}}]}
    result = analyze(report, manifest(), addendum)
    first = result["comparisons"][0]
    assert len(first["records"]) == 8
    assert first["summary"]["expected_records"] == 8
    assert first["summary"]["complete_pairs"] == 0
    assert len(first["summary"]["unavailable_records"]) == 8
    assert all(metric["full_cohort"] is None for metric in first["summary"]["metrics"].values())


def arm(mean_value, tokens=2):
    def field(probe):
        return {"probe_id": probe, "score": {"target_token_count": tokens,
            "total_log_probability": tokens * mean_value, "mean_log_probability": mean_value}}
    return {"status": "completed", "metrics": {"deleted_target_quality": {
        "fields": [field(f"deleted_field_{i}") for i in range(3)]},
        "retained_quality": field("retained_field")},
        "timing": {key: {"median": 2} for key in ("update_seconds", "query_seconds", "end_to_end_seconds")}}


def test_paired_probe_deltas_and_incomplete_cohort_withhold_full_estimate():
    row = {"arms": {"left": arm(-1), "right": arm(-3)}}
    pair = paired_record(row, "r1", 320, "left", "right")
    assert pair["deleted_mean_token_log_probability_delta"] == 2
    assert pair["probes"][0]["total_log_probability"]["left_minus_right"] == 4
    missing = paired_record(None, "r2", 340, "left", "right")
    summary = comparison_summary([pair, missing], smoke=False)
    metric = summary["metrics"]["deleted_mean_token_log_probability_delta"]
    assert metric["full_cohort"] is None
    assert metric["partial_descriptive_mean_only"] == 2
    assert summary["complete_pairs"] == 1
    row["arms"]["right"] = arm(-3, tokens=3)
    with pytest.raises(ValueError, match="token counts"):
        paired_record(row, "r1", 320, "left", "right")

import math

import pytest

from gemma_sv.eval_training_free_quality import summarize_paired_blocks


def test_paired_quality_summary_uses_contiguous_groups():
    base = [math.log(10.0)] * 8
    graft = [
        math.log(10.0 * (1.0 + cost / 100.0))
        for cost in (1, 2, 3, 4, 5, 6, 7, 8)
    ]

    report = summarize_paired_blocks(base, graft, group_size=2)

    assert report["blocks"] == 8
    assert report["base_perplexity"] == pytest.approx(10.0)
    assert report["relative_cost_percent"] > 4
    assert report["contiguous_groups"]["n"] == 4
    assert report["contiguous_groups"]["group_size"] == 2
    assert report["independent_training_seeds"] == 0
    assert report["inferential_interval"] is None
    assert report["contiguous_groups"]["q1_cost_percent"] <= report[
        "contiguous_groups"
    ]["q3_cost_percent"]
    assert "not an iid confidence interval" in report["contiguous_groups"][
        "interpretation"
    ]
    assert [
        row["start_block"]
        for row in report["contiguous_groups"]["rows"]
    ] == [0, 2, 4, 6]


def test_paired_quality_summary_rejects_unmatched_or_partial_groups():
    with pytest.raises(ValueError):
        summarize_paired_blocks([1.0], [1.0, 2.0], group_size=1)
    with pytest.raises(ValueError):
        summarize_paired_blocks([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], group_size=2)

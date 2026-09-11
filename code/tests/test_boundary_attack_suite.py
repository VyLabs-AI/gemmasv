import json

import pytest

from gemma_sv.run_boundary_attack_suite import _load_report


def test_attack_shard_validation_binds_condition_and_sampling(tmp_path):
    path = tmp_path / "shard.json"
    path.write_text(
        json.dumps(
            {
                "sampling": {"samples": 2, "k": [1, 2]},
                "whole_record": {
                    "records": [{"conditions": {"present": {}}}]
                },
            }
        )
    )

    report = _load_report(
        path,
        "present",
        samples=2,
        k_values=[1, 2],
    )
    assert report["sampling"]["samples"] == 2

    with pytest.raises(ValueError):
        _load_report(path, "never", samples=2, k_values=[1, 2])
    with pytest.raises(ValueError):
        _load_report(path, "present", samples=200, k_values=[1, 2])

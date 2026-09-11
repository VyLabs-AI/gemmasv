from __future__ import annotations

import json
from pathlib import Path

import pytest

from gemma_sv.summarize_whole_record_multiseed import (
    WholeRecordSummaryError,
    summarize,
)


def _write_report(
    root: Path,
    seed: int,
    *,
    admitted: int,
    manifest_sha256: str = "m" * 64,
) -> Path:
    attempted = 8
    payload = {
        "evaluation": "whole-record in-context unlearning",
        "ungrafted": False,
        "provenance": {
            "manifest": {
                "name": "whole_record_synthetic_v1",
                "version": 1,
                "sha256": manifest_sha256,
            },
            "model": {
                "id": "google/gemma-3-4b-pt",
                "resolved_revision": "revision",
            },
            "adapter": {
                "path": f"seed-{seed}/adapter",
                "content_sha256": f"{seed + 1:064x}",
            },
            "dataset": {
                "id": "locuslab/TOFU",
                "forget_fingerprint": "forget",
                "retain_fingerprint": "retain",
            },
            "run_seed": seed,
            "geometry": {
                "window": 1024,
                "n_fill": 22,
                "prefix_fillers": 11,
                "record_start": 0,
                "records": None,
            },
            "admission_gates": {
                "minimum_answer_lift_nats": 0.05,
                "minimum_secret_lift_nats": 0.05,
                "maximum_first_token_rank": 10,
                "all_fields_must_pass": True,
                "fixed_c_all_boundaries_must_be_feasible": True,
            },
        },
        "whole_record": {
            "mode": "admission_only",
            "attempted": attempted,
            "admitted": admitted,
            "records": [
                {"record_id": f"admitted-{seed}-{index}"}
                for index in range(admitted)
            ],
            "rejected_records": [
                {
                    "record_id": f"rejected-{seed}-{index}",
                    "reasons": ["retain:rank"],
                }
                for index in range(attempted - admitted)
            ],
            "exact_phrase_leak_at_k": {},
            "composite_any_leak_at_k": {},
        },
    }
    path = root / f"seed-{seed}.json"
    path.write_text(json.dumps(payload))
    return path


def test_summary_aggregates_three_provenance_matched_reports(tmp_path: Path):
    paths = [
        _write_report(tmp_path, 0, admitted=4),
        _write_report(tmp_path, 1, admitted=5),
        _write_report(tmp_path, 2, admitted=3),
    ]

    result = summarize(paths, minimum_reports=3)

    assert result["n_reports"] == 3
    assert result["seeds"] == [0, 1, 2]
    assert result["admission"]["attempted"] == 24
    assert result["admission"]["admitted"] == 12
    assert result["admission"]["rate"] == 0.5
    assert result["admission"]["rejection_reason_counts"] == {"retain:rank": 12}
    assert result["contains_record_ids"] is False
    assert all("record_id" not in row for row in result["seed_rows"])


def test_summary_rejects_mismatched_manifest(tmp_path: Path):
    paths = [
        _write_report(tmp_path, 0, admitted=4),
        _write_report(tmp_path, 1, admitted=4),
        _write_report(tmp_path, 2, admitted=4, manifest_sha256="x" * 64),
    ]

    with pytest.raises(WholeRecordSummaryError, match="cohort provenance differs"):
        summarize(paths, minimum_reports=3)


def test_summary_rejects_duplicate_seed(tmp_path: Path):
    first = _write_report(tmp_path, 0, admitted=4)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(first.read_text())

    with pytest.raises(WholeRecordSummaryError, match="duplicate run seed"):
        summarize([first, duplicate], minimum_reports=2)

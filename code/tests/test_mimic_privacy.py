import csv
import json

from gemma_sv.eval_mimic_whole_record import (
    _aggregate_only,
    build_private_manifest,
)


def test_private_manifest_drops_source_identifiers_and_is_deterministic(tmp_path):
    source = tmp_path / "diagnosis.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stay_id",
                "chiefcomplaint",
                "arrival_transport",
                "disposition",
            ],
        )
        writer.writeheader()
        for index in range(4):
            writer.writerow(
                {
                    "stay_id": f"SENSITIVE-{index}",
                    "chiefcomplaint": f"complaint-{index}",
                    "arrival_transport": f"transport-{index}",
                    "disposition": f"disposition-{index}",
                }
            )
    manifest, stats = build_private_manifest(source, records=2, skip=0)
    rendered = json.dumps(manifest)
    assert "SENSITIVE" not in rendered
    assert [row["record_id"] for row in manifest["records"]] == [
        "mimic-local-000",
        "mimic-local-001",
    ]
    assert stats == {"rows_scanned": 4, "complete_rows_seen": 4}


def test_aggregate_report_strips_all_record_text():
    marker = "DO-NOT-PERSIST-CLINICAL-TEXT"
    raw = {
        "sampling": {
            "conditions": ["present", "decrement"],
            "samples": 1,
            "k": [1],
        },
        "whole_record": {
            "attempted": 1,
            "admitted": 1,
            "admission_rate": 1.0,
            "rejected_records": [],
            "exact_phrase_leak_at_k": {},
            "composite_any_leak_at_k": {},
            "records": [
                {
                    "admission": {"fields": [{"secret": marker}]},
                    "conditions": {
                        "present": {
                            "fields": [
                                {
                                    "secret": marker,
                                    "secret_probe": {
                                        "mean_log_probability": -1.0,
                                    },
                                }
                            ],
                            "retain_secret_probe": {
                                "mean_log_probability": -2.0,
                            },
                        },
                        "decrement": {
                            "fields": [
                                {
                                    "secret": marker,
                                    "secret_probe": {
                                        "mean_log_probability": -3.0,
                                    },
                                }
                            ],
                            "retain_secret_probe": {
                                "mean_log_probability": -1.9,
                            },
                        },
                    },
                }
            ],
        },
    }
    report = _aggregate_only(raw, {"rows_scanned": 2, "complete_rows_seen": 2})
    rendered = json.dumps(report)
    assert marker not in rendered
    assert report["contains_source_text"] is False
    assert report["contains_source_identifiers"] is False

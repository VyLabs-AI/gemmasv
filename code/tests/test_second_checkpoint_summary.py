import json

from gemma_sv.summarize_second_checkpoint import summarize


MODEL = {
    "id": "google/gemma-3-12b-pt",
    "revision": "revision",
}


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def _record(index, *, rejected=False):
    row = {
        "manifest_index": index,
        "admission": {
            "fields": [
                {
                    "fixed_c_feasibility": [
                        {"feasible": True},
                    ]
                }
            ],
            "retain": {},
        },
    }
    if rejected:
        row["reasons"] = ["field_0:rank"]
    return row


def _admission(*, ungrafted, admitted):
    rows = [_record(0), _record(1, rejected=admitted == 1)]
    return {
        "model": MODEL["id"],
        "model_revision": MODEL["revision"],
        "ungrafted": ungrafted,
        "nu": 0.7,
        "preserve_prefix_mass": not ungrafted,
        "per_boundary_box": True,
        "provenance": {
            "manifest": {
                "sha256": "manifest-sha",
            }
        },
        "whole_record": {
            "attempted": 2,
            "admitted": admitted,
            "records": rows[:admitted],
            "rejected_records": rows[admitted:],
        },
    }


def test_second_checkpoint_summary_is_source_free_and_paired(tmp_path):
    predeclaration = {
        "primary_model": MODEL,
        "configuration": {"nu": 0.7},
        "confirmation": {
            "manifest": "manifest.json",
            "sha256": "manifest-sha",
            "attempted_records": 2,
        },
        "quality": {
            "blocks": 400,
            "expected_token_sha256": "token-sha",
        },
        "reporting_policy": {
            "retain_all_primary_outcomes": True,
        },
    }
    quality = {
        "model": MODEL["id"],
        "model_revision": MODEL["revision"],
        "evaluation": {
            "blocks": 400,
            "token_ids_sha256": "token-sha",
        },
        "summary": {
            "relative_cost_percent": 1.0,
        },
    }
    result = summarize(
        _write(tmp_path / "predeclaration.json", predeclaration),
        _write(tmp_path / "ungrafted.json", _admission(ungrafted=True, admitted=2)),
        _write(tmp_path / "graft.json", _admission(ungrafted=False, admitted=1)),
        _write(tmp_path / "quality.json", quality),
    )

    assert result["confirmation"]["ungrafted"]["admitted"] == 2
    assert result["confirmation"]["graft"]["admitted"] == 1
    assert result["confirmation"]["graft"]["fixed_c_feasible"] == 2
    assert result["quality"]["token_hash_matches_predeclaration"] is True
    assert "question" not in json.dumps(result)

from gemma_sv.eval_boundary_attacks import _normalized, _selected_specs


def test_boundary_attack_selection_uses_frozen_manifest_indices():
    config = {"admitted_record_indices": [1, 3]}
    manifest = {"records": [{"id": index} for index in range(5)]}

    selected = _selected_specs(config, manifest)

    assert selected == [(1, {"id": 1}), (3, {"id": 3})]


def test_boundary_attack_selection_can_follow_fixed_admission_report():
    config = {}
    manifest = {
        "records": [
            {"record_id": "a"},
            {"record_id": "b"},
            {"record_id": "c"},
        ]
    }
    admission = {
        "whole_record": {
            "records": [{"record_id": "b"}, {"record_id": "c"}]
        }
    }

    selected = _selected_specs(config, manifest, admission)

    assert [index for index, _ in selected] == [1, 2]


def test_normalized_recovery_keeps_unmeasurable_fields():
    values = _normalized(
        intervention=[-2.0, -2.0],
        never=[-2.0, -2.0],
        present=[-1.0, -1.98],
    )

    assert values[0] == 0.0
    assert values[1] is None

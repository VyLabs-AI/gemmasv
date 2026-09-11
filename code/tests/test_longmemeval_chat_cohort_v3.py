from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gemma_sv import longmemeval_chat_benchmark as chat_v1
from gemma_sv import longmemeval_deletion_benchmark as base
from gemma_sv import longmemeval_chat_cohort_v3 as cohort_v3


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "gemma_sv" / "benchmarks"
COHORT_PATH = BENCHMARKS / "longmemeval_chat_cohort_v3.json"
POLICY_PATH = (
    BENCHMARKS / "longmemeval_chat_cluster_analysis_lock_v3.json"
)
CENSUS_PATH = BENCHMARKS / "longmemeval_chat_cohort_census_v3.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed():
    return _load(COHORT_PATH), _load(POLICY_PATH), _load(CENSUS_PATH)


@pytest.fixture(scope="module")
def rebuilt():
    from transformers import AutoTokenizer

    rows = base.load_pinned_longmemeval_rows()
    tokenizer = AutoTokenizer.from_pretrained(
        cohort_v3.chat_v2.TOKENIZER_ID,
        revision=cohort_v3.chat_v2.TOKENIZER_REVISION,
        use_fast=True,
        local_files_only=True,
    )
    manifest, policy, census = cohort_v3.build_cohort(
        rows,
        tokenizer,
        require_pinned_artifacts=True,
    )
    hydrated = cohort_v3.rehydrate_manifest(manifest, rows, tokenizer)
    return manifest, policy, census, hydrated


def test_committed_artifacts_validate_and_are_source_free(committed):
    manifest, policy, census = committed
    cohort_v3.validate_manifest(manifest)
    cohort_v3.validate_policy_lock(policy)
    cohort_v3.validate_census(census, manifest, policy)
    assert manifest["policy_lock"] == policy
    assert manifest["contains_source_text"] is False
    assert census["contains_source_text"] is False
    assert not chat_v1._FORBIDDEN_SOURCE_KEYS.intersection(
        chat_v1._walk_keys((manifest, policy, census))
    )
    for descriptor in manifest["source_inventory"]["input_artifacts"]:
        assert "path" not in descriptor
        path = descriptor.get("repository_relative_path")
        if path is not None:
            assert not Path(path).is_absolute()


def test_rebuild_is_value_and_byte_deterministic(
    committed,
    rebuilt,
    tmp_path,
):
    expected_manifest, expected_policy, expected_census = committed
    manifest, policy, census, _hydrated = rebuilt
    assert manifest == expected_manifest
    assert policy == expected_policy
    assert census == expected_census
    paths = [
        tmp_path / COHORT_PATH.name,
        tmp_path / POLICY_PATH.name,
        tmp_path / CENSUS_PATH.name,
    ]
    for path, payload, expected_path in zip(
        paths,
        (manifest, policy, census),
        (COHORT_PATH, POLICY_PATH, CENSUS_PATH),
        strict=True,
    ):
        cohort_v3.write_json(path, payload)
        assert path.read_bytes() == expected_path.read_bytes()


def test_exact_counts_exposure_classes_and_cluster_contract(committed):
    manifest, policy, census = committed
    assert manifest["counts"] == {
        "target_clusters": 32,
        "histories_per_cluster": 3,
        "nested_history_instances": 96,
        "independent_analysis_n": 32,
    }
    assert manifest["exposure_summary"]["target_class_counts"] == {
        "direct-target": 16,
        "context-only": 2,
        "source-unseen": 14,
    }
    assert census["claim"] == {
        "target_clusters": 32,
        "nested_histories": 96,
        "independent_analysis_n": 32,
        "source_unseen_target_clusters": 14,
        "all_targets_source_unseen": False,
    }
    analysis = policy["analysis"]
    assert analysis["primary_unit"] == "target_cluster"
    assert analysis["primary_n"] == 32
    assert analysis["history_instances_are_nested_repeats"] is True
    assert analysis["history_instances_are_independent_records"] is False
    assert analysis["primary_endpoint"] == "intent-to-treat"
    assert analysis["confidence_interval"]["resampling_unit"] == (
        "target_cluster"
    )
    admission = policy["admission"]
    assert admission["unit"] == "history_instance"
    assert admission["primary_itt_denominator_ignores_admission"] is True
    assert admission["operational_or_admission_failure_primary_value"] == 0
    assert admission["conditional_efficacy_requires_joint_admission"] is True
    assert "N=96" in policy["claim_language"]["prohibited"]
    assert "N=32" in policy["claim_language"]["permitted"]


def test_controls_tails_and_clusters_are_unique_and_disjoint(committed):
    manifest, _policy, census = committed
    histories = manifest["histories"]
    cluster_sources: list[set[str]] = []
    targets: set[str] = set()
    controls: set[str] = set()
    tails: set[str] = set()
    for cluster in manifest["clusters"]:
        rows = [
            row
            for row in histories
            if row["cluster_id"] == cluster["cluster_id"]
        ]
        assert len(rows) == 3
        assert {row["variant_index"] for row in rows} == {0, 1, 2}
        assert len({row["target"]["source_id"] for row in rows}) == 1
        assert len({row["retained_probe"]["source_id"] for row in rows}) == 1
        role_ids = set()
        for row in rows:
            target = row["target"]["source_id"]
            control = row["retained_probe"]["source_id"]
            row_tails = {
                reference["source_id"]
                for reference in row["tail_session_references"]
            }
            assert row_tails.isdisjoint(tails)
            targets.add(target)
            controls.add(control)
            tails.update(row_tails)
            role_ids.update({target, control, *row_tails})
        assert all(role_ids.isdisjoint(prior) for prior in cluster_sources)
        cluster_sources.append(role_ids)
    assert len(targets) == 32
    assert len(controls) == 32
    assert len(tails) == 175
    assert targets.isdisjoint(controls | tails)
    assert controls.isdisjoint(tails)
    assert census["support"]["unique_support_sources_used"] == 207


def test_every_history_is_geometry_safe_and_fixed_c(committed):
    manifest, _policy, census = committed
    for history in manifest["histories"]:
        context = history["context"]
        assert context["tokens_strictly_after_owned"] >= 1152
        assert context["runtime_total_token_bound"] <= 8192
        assert (
            context["local_window_safety"][
                "selected_span_inside_local_window"
            ]
            is False
        )
        assert (
            context["fixed_c_reference"][
                "all_affected_boundaries_feasible"
            ]
            is True
        )
        assert context["full_repack_fallback"]["required"] is False
    assert census["geometry"]["fixed_c_feasible_histories"] == 96
    assert census["geometry"]["full_repack_fallback_required_histories"] == 0
    assert census["geometry"]["minimum_tokens_strictly_after_owned"] == 1152
    assert census["geometry"]["maximum_runtime_total_token_bound"] == 5386


def test_history_identifier_binds_full_history(committed):
    manifest, _policy, _census = committed
    original = manifest["histories"][0]
    changed = copy.deepcopy(original)
    changed["tail_session_references"][0]["session_id"] += "-changed"
    assert cohort_v3._payload_sha256(
        cohort_v3._history_binding(changed)
    ) != original["history_binding_sha256"]
    assert cohort_v3._history_id(changed) != original["record_id"]


def test_rehydration_preserves_all_frozen_history_ids(rebuilt):
    manifest, _policy, _census, hydrated = rebuilt
    assert len(hydrated) == 96
    assert [record.record_id for record in hydrated] == [
        history["record_id"] for history in manifest["histories"]
    ]

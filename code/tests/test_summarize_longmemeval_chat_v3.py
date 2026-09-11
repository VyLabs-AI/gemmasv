from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gemma_sv import longmemeval_chat_matcher_v1 as matcher
from gemma_sv import longmemeval_deletion_benchmark as source
from gemma_sv import summarize_longmemeval_chat_v3 as summary


def _sha(character: str) -> str:
    return character * 64


def _artifact_bindings() -> dict[str, dict]:
    return {
        role: {
            "repository_path": (
                None if role == "pinned_oracle" else f"frozen/{role}.json"
            ),
            "file_sha256": _sha(hex(index + 1)[-1]),
            "payload_sha256": _sha(hex(index + 2)[-1]),
            "integrity_sha256": (
                None if role in {"pinned_oracle", "cluster_analysis_lock"}
                else _sha(hex(index + 3)[-1])
            ),
            "lock_sha256": (
                _sha("f") if role == "cluster_analysis_lock" else None
            ),
        }
        for index, role in enumerate(summary._ARTIFACT_ROLES)
    }


def _implementation_bindings() -> dict:
    files = {
        "test_implementation": {
            "repository_path": "gemma_sv/synthetic.py",
            "file_sha256": _sha("a"),
        }
    }
    return {
        "files": files,
        "file_count": 1,
        "files_sha256": summary.payload_sha256(files),
    }


def _anonymous_bindings() -> list[dict]:
    return [
        {
            "cluster_index": index // summary.HISTORIES_PER_CLUSTER,
            "history_index": index,
            "variant_index": index % summary.HISTORIES_PER_CLUSTER,
            "binding_sha256": summary.payload_sha256(
                {"anonymous_history_index": index}
            ),
        }
        for index in range(summary.EXPECTED_HISTORIES)
    ]


def _endpoint_rows(*, unavailable: set[int] = frozenset()) -> list[dict]:
    return [
        {
            "cluster_index": index // summary.HISTORIES_PER_CLUSTER,
            "history_index": index,
            "variant_index": index % summary.HISTORIES_PER_CLUSTER,
            "available": index not in unavailable,
            "success": index not in unavailable,
        }
        for index in range(summary.EXPECTED_HISTORIES)
    ]


def _attempt(repeat_index: int, text: str, tokens: list[int]) -> dict:
    return {
        "repeat_index": repeat_index,
        "status": "completed",
        "response_text": text,
        "response_utf8_sha256": summary.text_sha256(text),
        "generated_token_ids": tokens,
        "generated_token_ids_sha256": summary.payload_sha256(tokens),
        "generated_token_count": len(tokens),
    }


def test_directional_matcher_and_canonical_repeat_contract():
    observed = matcher.match_disclosure(
        "The remembered amount was exactly 2 weeks.",
        "two weeks",
    )
    assert observed["numeric_unit_token_span"] is True
    assert observed["deterministic_any"] is True

    probe = {
        "status": "completed",
        "generation_attempts": [
            _attempt(0, "two weeks", [1, 2, 3]),
            _attempt(1, "two weeks", [1, 2, 3]),
        ],
        "repeat_check": {
            "repetitions": 2,
            "exact_token_ids_and_response_text_match": True,
        },
    }
    canonical = summary.canonical_probe_value(probe)
    assert canonical.available is True
    assert canonical.reproducible is True
    assert canonical.value == "two weeks"

    nondeterministic = copy.deepcopy(probe)
    nondeterministic["status"] = "failed_nondeterministic"
    nondeterministic["generation_attempts"][1] = _attempt(
        1,
        "three weeks",
        [4, 5],
    )
    nondeterministic["repeat_check"][
        "exact_token_ids_and_response_text_match"
    ] = False
    failed = summary.canonical_probe_value(nondeterministic)
    assert failed.available is False
    assert failed.reproducible is False
    assert failed.value is None


def test_cluster_aggregation_is_k32_seeded_and_failure_preserving():
    rows = _endpoint_rows(unavailable={0})
    first = summary.aggregate_binary_endpoint(
        rows,
        success=lambda row: row["success"],
        available=lambda row: row["available"],
        resamples=1_000,
    )
    second = summary.aggregate_binary_endpoint(
        rows,
        success=lambda row: row["success"],
        available=lambda row: row["available"],
        resamples=1_000,
    )

    assert first == second
    assert first["K"] == 32
    assert first["nested_history_count"] == 96
    assert first["histories_per_cluster"] == 3
    assert first["denominator_histories"] == 96
    assert first["numerator_histories"] == 95
    assert first["failed_or_unavailable_histories"] == 1
    assert first["cluster_mean"] == 95 / 96
    interval = first["cluster_bootstrap_95_interval"]
    assert interval["clusters_per_resample"] == 32
    assert interval["histories_resampled_within_cluster"] is False
    assert interval["percentile_interpolation"] == "linear_type_7"

    assert summary.nonleak_success(available=False, disclosed=False) is False
    assert summary.nonleak_success(available=True, disclosed=False) is True
    with pytest.raises(summary.V3SummaryError, match="exactly 96"):
        summary.aggregate_binary_endpoint(
            rows[:-3],
            success=lambda row: row["success"],
        )


def test_analysis_lock_is_deterministic_strict_and_source_free():
    first = summary.build_analysis_lock(
        artifact_bindings=_artifact_bindings(),
        anonymous_histories=_anonymous_bindings(),
        implementation_bindings=_implementation_bindings(),
    )
    second = summary.build_analysis_lock(
        artifact_bindings=_artifact_bindings(),
        anonymous_histories=_anonymous_bindings(),
        implementation_bindings=_implementation_bindings(),
    )
    assert first == second
    summary.validate_analysis_lock(first)
    summary.assert_source_free(first)

    assert first["schema"] == summary.ANALYSIS_LOCK_SCHEMA
    assert first["status"] == summary.ANALYSIS_LOCK_STATUS
    assert first["post_hoc"] is True
    assert first["preregistered_before_response_generation"] is False
    assert first["geometry"]["K_target_clusters"] == 32
    assert first["geometry"]["nested_history_count"] == 96
    assert first["matrix"]["greedy_repeats_per_probe"] == 2
    assert first["analysis"]["bootstrap"]["resamples"] == 100_000
    assert first["analysis"]["bootstrap"]["seed"] == 20_260_823
    assert first["failures"]["missing_policy_output_counts_as_nonleak"] is False
    assert first["matching"]["target_success_direction_by_condition"] == {
        "present": "disclosure",
        "fresh_raw_omission": "non_disclosure",
        "exact_decrement_or_refit_policy": "non_disclosure",
        "prompt_suppression": "non_disclosure",
    }
    mechanism = first["decisions"]["mechanism"]
    assert mechanism["continuation_kl_threshold_nats"] == 1e-6
    assert mechanism["coefficient_residual_tolerance"] == 1e-6
    assert mechanism["decision_function_deviation_tolerance"] == 1e-5
    assert mechanism["kkt_tolerance"] == 1e-5
    assert "excess=max(0, edit vs canonical refit - floor)" in mechanism[
        "floor_relative_rule"
    ]
    assert (
        first["decisions"]["strong_deterministic_direct_removal"][
            "current_status"
        ]
        == "pending_continuous_utility"
    )

    tampered = copy.deepcopy(first)
    tampered["analysis"]["bootstrap"]["clusters_per_resample"] = 31
    tampered = summary._seal(tampered)
    with pytest.raises(summary.V3SummaryError, match="frozen contract"):
        summary.validate_analysis_lock(tampered)


def test_lock_writer_never_overwrites(tmp_path):
    lock = summary.build_analysis_lock(
        artifact_bindings=_artifact_bindings(),
        anonymous_histories=_anonymous_bindings(),
        implementation_bindings=_implementation_bindings(),
    )
    destination = tmp_path / "analysis-lock.json"
    summary.write_analysis_lock(destination, lock)
    original = destination.read_bytes()
    loaded = summary.load_json(destination, name="written analysis lock")
    summary.validate_analysis_lock(loaded)
    assert loaded == lock
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        summary.write_analysis_lock(destination, lock)
    assert destination.read_bytes() == original


def test_source_free_guard_rejects_content_ids_tokens_and_absolute_paths():
    prohibited = (
        {"response_text": "private"},
        {"safe": "longmemeval-chat-history-v3-private"},
        {"safe": [1, 2, 3]},
        {"safe": "/Users/private/oracle.json"},
    )
    for value in prohibited:
        with pytest.raises(summary.V3SummaryError):
            summary.assert_source_free(value)
    with pytest.raises(summary.V3SummaryError, match="source/generated"):
        summary.assert_source_free(
            {"safe": "a private generated sentence"},
            sensitive_strings=["a private generated sentence"],
        )


@pytest.fixture(scope="module")
def cached_source_path() -> Path:
    hub = pytest.importorskip("huggingface_hub")
    try:
        path = hub.hf_hub_download(
            repo_id=source.DATASET_ID,
            repo_type="dataset",
            filename=source.DATASET_ARTIFACT_PATH,
            revision=source.DATASET_REVISION,
            local_files_only=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"pinned local oracle unavailable: {type(exc).__name__}")
    return Path(path)


@pytest.fixture(scope="module")
def real_v3(cached_source_path):
    inputs = summary._load_exact_inputs(data_path=cached_source_path)
    lock = summary.build_analysis_lock(
        artifact_bindings=inputs.artifacts,
        anonymous_histories=summary._anonymous_bindings(inputs.cohort),
    )
    built = summary.build_summary(
        source_rows=inputs.source_rows,
        final=inputs.final,
        validation=inputs.validation,
        cohort=inputs.cohort,
        census=inputs.census,
        cluster_lock=inputs.cluster_lock,
        response_authorization=inputs.response_authorization,
        suffix_contamination=inputs.suffix_contamination,
        sample_lock=inputs.sample_lock,
        rubric=inputs.rubric,
        analysis_lock=lock,
    )
    return inputs, lock, built


def test_real_exact_96_row_summary_is_complete_and_source_free(real_v3):
    inputs, lock, built = real_v3
    summary.validate_analysis_lock(lock)
    summary.validate_summary(built)
    summary.assert_source_free(
        built,
        sensitive_strings=summary._sensitive_strings(
            inputs.source_rows,
            inputs.final,
        ),
    )

    assert built["schema"] == summary.SUMMARY_SCHEMA
    assert built["status"] == "complete-post-hoc-deterministic"
    assert len(built["histories"]) == 96
    assert all(
        row["history_index"] == index
        and row["cluster_index"] == index // 3
        and row["variant_index"] == index % 3
        and row["completion"]["terminal"] is True
        and row["completion"]["history_completed"] is True
        for index, row in enumerate(built["histories"])
    )
    assert built["flow"] == {
        "oracle_rows": 500,
        "candidate_targets": 36,
        "eligible_frozen_clusters": 32,
        "histories": 96,
        "attempted_histories": 96,
        "terminal_histories": 96,
        "completed_histories": 96,
        "target_admitted": None,
        "retained_available": None,
        "jointly_admitted": None,
        "admission_status": "pending_continuous_utility",
    }
    rendered = summary.deterministic_json(built)
    assert "longmemeval-chat-history-v3-" not in rendered
    assert "longmemeval-chat-cluster-v3-" not in rendered
    assert '"response_text"' not in rendered
    assert '"generated_token_ids"' not in rendered


def test_real_sample_strata_suffix_strata_and_all_endpoint_k(real_v3):
    inputs, _lock, built = real_v3
    published_strata = built["leakage_recall_instrument"]["strata"]
    frozen_strata = inputs.sample_lock["strata"]
    assert [
        (
            row["matcher_positive_histories"],
            row["matcher_clean_histories"],
        )
        for row in published_strata
    ] == [
        (row["matcher_positive_size"], row["matcher_clean_size"])
        for row in frozen_strata
    ]
    assert all(row["reconciled"] is True for row in published_strata)

    suffix_counts = built["suffix_strata"]
    assert suffix_counts["clean_histories"] == 72
    assert suffix_counts["contaminated_histories"] == 24
    assert suffix_counts["clusters_with_any_contaminated_history"] == 10

    endpoints = list(
        summary._walk_endpoint_objects(built["primary_endpoints"])
    )
    assert endpoints
    assert all(
        row["K"] == 32
        and row["nested_history_count"] == 96
        and row["denominator_histories"] == 96
        and row["cluster_bootstrap_95_interval"]["K"] == 32
        and row["cluster_bootstrap_95_interval"]["clusters_per_resample"] == 32
        for row in endpoints
    )
    policy = built["primary_endpoints"]["target"][
        "intent_to_treat_policy_nonleak"
    ]["deterministic_any"]
    assert policy["failed_or_unavailable_histories"] == 0


def test_real_outputs_are_deterministic_and_summary_writer_is_no_overwrite(
    real_v3,
    tmp_path,
):
    inputs, lock, built = real_v3
    rebuilt = summary.build_summary(
        source_rows=inputs.source_rows,
        final=inputs.final,
        validation=inputs.validation,
        cohort=inputs.cohort,
        census=inputs.census,
        cluster_lock=inputs.cluster_lock,
        response_authorization=inputs.response_authorization,
        suffix_contamination=inputs.suffix_contamination,
        sample_lock=inputs.sample_lock,
        rubric=inputs.rubric,
        analysis_lock=lock,
    )
    assert summary.deterministic_json(rebuilt) == summary.deterministic_json(
        built
    )

    destination = tmp_path / "summary.json"
    summary.write_summary(destination, built)
    original = destination.read_bytes()
    loaded = summary.load_json(destination, name="written summary")
    summary.validate_summary(loaded, analysis_lock=lock)
    assert loaded == built
    with pytest.raises(FileExistsError, match="overwrite is forbidden"):
        summary.write_summary(destination, built)
    assert destination.read_bytes() == original


def test_cli_requires_explicit_data_and_exactly_one_mode():
    parser = summary._parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--freeze-lock"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--data-path",
                "/local/oracle.json",
                "--freeze-lock",
                "--summarize",
            ]
        )

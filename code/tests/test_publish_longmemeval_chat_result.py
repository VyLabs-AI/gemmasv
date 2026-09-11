from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from gemma_sv import publish_longmemeval_chat_result as publisher


ROOT = Path(__file__).resolve().parents[1]
COMPACT_PATH = (
    ROOT
    / "gemma_sv"
    / "benchmarks"
    / "longmemeval_chat_geometry_methods_compact16_v1.json"
)
MACROS_PATH = (
    ROOT / "gemma_sv" / "paper" / "longmemeval_chatbot_compact16_macros.tex"
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _resign(value: dict) -> None:
    value.pop("integrity", None)
    value["integrity"] = {
        "algorithm": "sha256",
        "sha256": publisher.payload_sha256(value),
    }


@pytest.fixture(scope="module")
def compact() -> dict:
    value = _load(COMPACT_PATH)
    publisher.validate_compact_result(value)
    return value


def test_committed_publication_rebuilds_from_exact_final_report(compact):
    if not publisher.DEFAULT_REPORT.is_file():
        pytest.skip("local finalized all16 report is unavailable")
    rebuilt = publisher.build_from_paths()
    assert rebuilt == compact
    assert publisher.render_latex_macros(rebuilt) == MACROS_PATH.read_text(
        encoding="utf-8"
    )


def test_compact_result_recomputes_validated_headlines(compact):
    assert compact["status"] == "validated-complete-all16"
    assert compact["behavioral_result_status"] == "validated-positive"
    assert compact["certificate_status"] == "validated-pass"
    assert compact["source_artifacts"]["final_report"]["file_sha256"] == (
        publisher.EXPECTED_FINAL_REPORT_FILE_SHA256
    )
    assert compact["denominators"]["attempted_records"] == 16
    assert compact["denominators"]["joint_admitted_records"] == 10
    assert compact["denominators"]["retained_available_records"] == 16
    assert compact["denominators"]["source_state_immutable_records"] == 16
    assert compact["denominators"]["raw_report_record_failures"] == 16
    assert compact["denominators"]["primary_result_failures"] == 0
    assert compact["denominators"]["excluded_diagnostic_failures"] == 16

    exact = compact["efficacy"]["exact_policy_summary"]
    assert exact["mean_target_suppression_nats"] == pytest.approx(
        8.327004143363855
    )
    assert exact["minimum_target_suppression_nats"] == pytest.approx(
        1.7298680064801788
    )
    assert exact["maximum_target_suppression_nats"] == pytest.approx(
        19.89164355219483
    )
    assert exact["mean_nats_below_raw_omission_floor"] == pytest.approx(
        1.7528518655603365
    )
    assert exact["records_at_or_below_raw_omission_floor"] == 6
    assert exact[
        "mean_absolute_retained_log_probability_drift_nats"
    ] == pytest.approx(8.054104423996529e-05)
    assert exact[
        "mean_target_first_token_kl_to_raw_omission_nats"
    ] == pytest.approx(2.1517010661192484)

    certificate = compact["certificate"]
    assert certificate["records"] == 16
    assert certificate["probes"] == 32
    assert certificate["mean_kl_nats"] == pytest.approx(
        9.591298981982964e-18
    )
    assert certificate["maximum_kl_nats"] == pytest.approx(
        1.7528497686921997e-16
    )
    assert certificate["behavioral_raw_omission_is_distinct"] is True
    assert certificate["raw_history_equality_claimed"] is False


def test_exact_policy_and_cache_diagnostic_are_not_conflated(compact):
    policy = compact["exact_policy"]
    assert policy["incremental_float64_decrement_records"] == 0
    assert policy["fixed_c_refit_fallback_records"] == 16
    assert policy["source_full_repack_fallback_records"] == 0
    assert policy["incremental_conversation_deletion_speed_claim_supported"] is (
        False
    )
    cache = compact["diagnostics"]["cache_delete_shift"]
    assert cache["classification"] == "incompatible_diagnostic"
    assert cache["failed_records"] == 16
    assert cache["implies_exact_policy_failure"] is False
    assert cache["causes_raw_report_completed_with_record_failures_status"] is (
        True
    )


def test_prompt_only_forgetting_is_weaker_than_state_edit(compact):
    conditions = compact["efficacy"]["conditions"]
    exact = conditions["exact_decrement"]
    prompt = conditions["prompt_suppression"]
    assert exact["mean_target_suppression_vs_present_nats"] == pytest.approx(
        8.327004143363855
    )
    assert prompt["mean_target_suppression_vs_present_nats"] == pytest.approx(
        0.5342220649363307
    )
    assert prompt[
        "mean_target_log_probability_drift_from_raw_omission_nats"
    ] == pytest.approx(6.039930212867186)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("attempted", "denominators"),
        ("certificate", "certificate"),
        ("cache", "cache diagnostic"),
        ("report_hash", "final-report binding"),
    ],
)
def test_resigned_compact_tampering_fails_closed(compact, mutation, message):
    changed = copy.deepcopy(compact)
    if mutation == "attempted":
        changed["denominators"]["attempted_records"] = 15
    elif mutation == "certificate":
        changed["certificate"]["maximum_kl_nats"] = 1.0
    elif mutation == "cache":
        changed["diagnostics"]["cache_delete_shift"]["classification"] = (
            "primary_failure"
        )
    else:
        changed["source_artifacts"]["final_report"]["file_sha256"] = "0" * 64
    _resign(changed)
    with pytest.raises(publisher.PublicationError, match=message):
        publisher.validate_compact_result(changed)


def test_compact_and_macros_are_source_free_and_deterministic(compact):
    serialized = publisher.deterministic_json(compact)
    assert compact["contains_source_text"] is False
    assert compact["contains_full_vocabulary_vectors"] is False
    assert compact["contains_model_generated_text"] is False
    assert '"messages"' not in serialized
    assert '"source_text"' not in serialized
    assert '"logits"' not in serialized

    macros = publisher.render_latex_macros(compact)
    assert macros == publisher.render_latex_macros(copy.deepcopy(compact))
    assert "\\providecommand{\\LMChatExactSuppression}{8.33}" in macros
    assert (
        "\\providecommand{\\LMChatCertificateMeanKL}"
        "{9.59\\times 10^{-18}}" in macros
    )
    assert (
        "\\providecommand{\\LMChatCertificateMaxKL}"
        "{1.75\\times 10^{-16}}" in macros
    )
    assert "\\providecommand{\\LMChatRefitFallbackN}{16}" in macros
    assert "\\providecommand{\\LMChatIncrementalN}{0}" in macros
    assert "\\providecommand{\\LMChatRawReportFailureN}{16}" in macros
    assert "\\providecommand{\\LMChatPrimaryFailureN}{0}" in macros
    assert "\\providecommand{\\LMChatExactAtOrBelowRawN}{6}" in macros
    assert (
        "\\providecommand{\\LMChatCacheDeleteShiftClassification}"
        "{incompatible-diagnostic}" in macros
    )
    assert "INCOMPLETE CASE STUDY PREVIEW" not in macros

    compact_file_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert (
        f"\\providecommand{{\\LMChatCompactReportSHA}}"
        f"{{{compact_file_sha256}}}" in macros
    )


def test_strict_json_loader_rejects_duplicates_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"first","status":"second"}', encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="duplicate"):
        publisher._load_mapping_and_sha256(duplicate, name="duplicate fixture")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"metric":NaN}', encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="non-finite"):
        publisher._load_mapping_and_sha256(nonfinite, name="nonfinite fixture")


def test_partial_report_cannot_be_published_as_final():
    required = (
        publisher.finalizer.DEFAULT_PARTIAL_REPORT,
        publisher.methods.DEFAULT_ADMISSION_REPORT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("standalone release excludes bound local reports")
    with pytest.raises(publisher.PublicationError, match="exact validated all16"):
        publisher.build_from_paths(report_path=publisher.finalizer.DEFAULT_PARTIAL_REPORT)

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gemma_sv import build_longmemeval_chat_forgetting_payload as builder


ROOT = Path(__file__).resolve().parents[1]
COMPACT_PATH = (
    ROOT
    / "paper_viz"
    / "payloads"
    / "longmemeval_chat_forgetting_compact_preview.json"
)
PAYLOAD_PATH = (
    ROOT
    / "gemma_sv"
    / "demo_site"
    / "assets"
    / "longmemeval_forgetting_preview.json"
)
FINAL_COMPACT_PATH = builder.DEFAULT_COMPACT_OUTPUT
FINAL_PAYLOAD_PATH = builder.DEFAULT_PAYLOAD_OUTPUT
DEFAULT_ORACLE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--xiaowu0162--longmemeval-cleaned"
    / "snapshots"
    / builder.longmemeval.DATASET_REVISION
    / builder.longmemeval.DATASET_ARTIFACT_PATH
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _resign(value: dict) -> None:
    value.pop("integrity", None)
    value["integrity"] = {
        "algorithm": "sha256",
        "sha256": builder.payload_sha256(value),
    }


@pytest.fixture(scope="module")
def committed() -> tuple[dict, dict]:
    compact = _load(COMPACT_PATH)
    payload = _load(PAYLOAD_PATH)
    builder.validate_payload(payload, compact=compact)
    return compact, payload


@pytest.fixture(scope="module")
def bound_inputs():
    required = (
        builder.PREVIEW_METHOD_REPORT,
        builder.DEFAULT_ADMISSION_REPORT,
        builder.DEFAULT_MANIFEST,
        builder.DEFAULT_METHOD_LOCK,
        builder.DEFAULT_PARTIAL_LOCK,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("local hash-bound experiment reports are unavailable")
    (
        manifest,
        policy,
        census,
        admission,
        method_lock,
        partial_lock,
        completion_lock,
        report,
        report_hash,
    ) = builder._load_bound_inputs(
        manifest_path=builder.DEFAULT_MANIFEST,
        policy_path=builder.DEFAULT_POLICY,
        census_path=builder.DEFAULT_CENSUS,
        admission_report_path=builder.DEFAULT_ADMISSION_REPORT,
        method_lock_path=builder.DEFAULT_METHOD_LOCK,
        partial_lock_path=builder.DEFAULT_PARTIAL_LOCK,
        completion_lock_path=None,
        method_report_path=builder.PREVIEW_METHOD_REPORT,
    )
    return {
        "manifest": manifest,
        "policy": policy,
        "census": census,
        "admission": admission,
        "method_lock": method_lock,
        "partial_lock": partial_lock,
        "completion_lock": completion_lock,
        "report": report,
        "report_hash": report_hash,
    }


def test_committed_payload_is_explicitly_incomplete_case_only(committed):
    compact, payload = committed
    assert compact["preview_label"] == builder.PREVIEW_LABEL
    assert payload["preview_label"] == builder.PREVIEW_LABEL
    assert compact["report_state"] == {
        "aggregate_claims_allowed": False,
        "all16_complete": False,
        "attempted_records": 15,
        "authorized_records": 16,
        "not_attempted_records": 1,
    }
    assert compact["aggregate_claims"] == []
    assert payload["scope"]["aggregate_claims"] == []
    assert payload["case"]["record_id"] == builder.CASE_RECORD_ID
    assert payload["case"]["selection"]["selected_from_method_outcomes"] is False
    assert payload["official_longmemeval_leaderboard_score"] is False


def test_committed_final_payload_is_complete_but_case_only():
    compact = _load(FINAL_COMPACT_PATH)
    payload = _load(FINAL_PAYLOAD_PATH)
    builder.validate_payload(payload, compact=compact)
    assert compact["status"] == "final_case_study"
    assert payload["status"] == "final_case_study"
    assert compact["preview_label"] is None
    assert payload["preview_label"] is None
    assert compact["report_state"] == {
        "aggregate_claims_allowed": False,
        "all16_complete": True,
        "attempted_records": 16,
        "authorized_records": 16,
        "not_attempted_records": 0,
    }
    assert compact["source_artifacts"]["method_report"]["file_sha256"] == (
        "a7d0582d7d5aa6832320852f0b80e79799621d6520ebef5f7a44eeea0e327fb6"
    )
    assert payload["scope"]["aggregate_claims"] == []
    assert payload["case"]["certificate"]["decrement_fallbacks"]["value"] == 522
    assert payload["case"]["certificate"]["head_gate_solves"]["value"] == 560
    assert payload["case"]["certificate"]["head_gates"]["value"] == 960


def test_optional_completion_lock_constant_matches_committed_lock():
    lock = _load(builder.DEFAULT_COMPLETION_LOCK)
    assert lock["integrity"]["sha256"] == (
        builder.COMPLETION_LOCK_INTEGRITY_SHA256
    )
    builder._validate_integrity(
        lock,
        expected=builder.COMPLETION_LOCK_INTEGRITY_SHA256,
        name="completion lock",
    )


def test_every_model_outcome_has_a_report_pointer(committed):
    _, payload = committed
    for condition in payload["case"]["outcomes"]:
        for probe in ("target", "retained"):
            for metric in condition[probe].values():
                if not isinstance(metric, dict) or "source_pointer" not in metric:
                    continue
                assert metric["artifact"] == "method_report"
                assert metric["source_pointer"].startswith("/records/1/")
                if isinstance(metric["value"], (int, float)):
                    assert not isinstance(metric["value"], bool)
    assert payload["contains_model_generated_text"] is False
    assert payload["demo"]["model_outcome_text_present"] is False


def test_fabricated_model_value_fails_even_after_payload_is_resigned(committed):
    compact, payload = committed
    changed = copy.deepcopy(payload)
    changed["case"]["outcomes"][2]["target"][
        "geometric_mean_probability"
    ]["value"] = 0.75
    _resign(changed)
    with pytest.raises(builder.PayloadError, match="differ"):
        builder.validate_payload(changed, compact=compact)


def test_compact_value_must_reproduce_from_exact_method_report(
    committed,
    bound_inputs,
):
    compact, _ = committed
    changed = copy.deepcopy(compact)
    changed["case"]["outcomes"][0]["target"][
        "first_target_token_rank"
    ]["value"] = 1
    _resign(changed)
    with pytest.raises(builder.PayloadError, match="differs"):
        builder.validate_compact_case_report(
            changed,
            report=bound_inputs["report"],
            admission_report=bound_inputs["admission"],
            manifest=bound_inputs["manifest"],
            method_lock=bound_inputs["method_lock"],
            partial_lock=bound_inputs["partial_lock"],
            completion_lock=bound_inputs["completion_lock"],
            report_file_sha256=bound_inputs["report_hash"],
            allow_incomplete_preview=True,
        )


def test_public_dialogue_is_deliberately_quoted_and_hash_bound(committed):
    _, payload = committed
    expected_roles = {
        "target": ["user", "assistant"],
        "retained": ["user", "assistant"],
        "intervening": ["user", "assistant"],
    }
    for section, roles in expected_roles.items():
        quotes = payload["case"]["dialogue"][section]["quotes"]
        assert [quote["role"] for quote in quotes] == roles
        for quote in quotes:
            assert quote["origin"] == "public_longmemeval_dialogue"
            assert len(quote["source_turn_sha256"]) == 64
            assert quote["display_text_sha256"] == (
                builder.longmemeval.text_sha256(quote["display_text"])
            )
            assert quote["truncated"] is False
            assert quote["excerpt_policy"]["selection"] == (
                "exact UTF-8 source substring"
            )
            assert quote["excerpt_policy"]["normalization"] == "none"
    action = payload["case"]["deletion_action"]
    assert action["origin"] == "evaluator_added_deletion_action"
    assert action["public_longmemeval_source"] is False
    assert action["out_of_band"] is True


@pytest.mark.parametrize("payload_path", [PAYLOAD_PATH, FINAL_PAYLOAD_PATH])
def test_recorded_resolver_fixture_is_catalog_bound_and_provider_free(
    payload_path,
):
    payload = _load(payload_path)
    resolution = payload["case"]["memory_resolution"]
    builder.validate_recorded_resolver_fixture(
        resolution,
        dialogue=payload["case"]["dialogue"],
        deletion_action=payload["case"]["deletion_action"],
    )
    assert resolution["mode"] == "deterministic_recorded_fixture"
    assert resolution["label"] == builder.RESOLVER_FIXTURE_LABEL
    assert resolution["provider"] == {
        "artifact": None,
        "artifact_present": False,
        "call_performed": False,
    }
    assert resolution["network_request_performed"] is False
    assert resolution["model_configuration"] == {
        "default_model_id": builder.DEFAULT_RESOLVER_MODEL,
        "model_produced_fixture": False,
    }
    catalog = resolution["candidate_catalog"]
    assert len(catalog) == 2
    assert resolution["catalog_hash"] == builder.payload_sha256(catalog)
    assert len({candidate["record_id"] for candidate in catalog}) == 2
    assert all(
        candidate["record_id"].startswith("memory_")
        and len(candidate["record_id"]) == len("memory_") + 24
        for candidate in catalog
    )
    target = next(
        candidate
        for candidate in catalog
        if candidate["owned_exchange_ref"]
        == "/case/dialogue/target/quotes"
    )
    retained = next(
        candidate
        for candidate in catalog
        if candidate["owned_exchange_ref"]
        == "/case/dialogue/retained/quotes"
    )
    decision = resolution["decision"]
    assert decision["selected_record_id"] == target["record_id"]
    assert decision["selected_record_id"] != retained["record_id"]
    assert decision["alternative_record_ids"] == []
    assert decision["confidence"] is None
    assert decision["requires_confirmation"] is True
    assert decision["deletion_executed"] is False
    confirmation = resolution["confirmation"]
    assert confirmation["selected_record_id"] == target["record_id"]
    assert confirmation["catalog_hash"] == resolution["catalog_hash"]
    assert confirmation["requires_explicit_click"] is True
    assert confirmation["confirmed_in_payload"] is False
    assert confirmation["owned_exchange"] == (
        payload["case"]["dialogue"]["target"]["quotes"]
    )
    assert len(confirmation["owned_exchange"]) == 2


def test_recorded_resolver_unknown_or_retained_id_is_rejected(committed):
    compact, payload = committed
    unknown = copy.deepcopy(payload)
    unknown["case"]["memory_resolution"]["decision"][
        "selected_record_id"
    ] = "memory_" + "0" * 24
    _resign(unknown)
    with pytest.raises(builder.PayloadError, match="invalid ID"):
        builder.validate_payload(unknown, compact=compact)

    retained = copy.deepcopy(payload)
    resolution = retained["case"]["memory_resolution"]
    retained_candidate = next(
        candidate
        for candidate in resolution["candidate_catalog"]
        if candidate["owned_exchange_ref"]
        == "/case/dialogue/retained/quotes"
    )
    resolution["decision"]["selected_record_id"] = retained_candidate[
        "record_id"
    ]
    _resign(retained)
    with pytest.raises(builder.PayloadError, match="target/retained"):
        builder.validate_payload(retained, compact=compact)


def test_recorded_resolver_catalog_or_provider_claim_drift_is_rejected(
    committed,
):
    compact, payload = committed
    changed_catalog = copy.deepcopy(payload)
    changed_catalog["case"]["memory_resolution"]["candidate_catalog"][0][
        "label"
    ] = "fabricated label"
    _resign(changed_catalog)
    with pytest.raises(builder.PayloadError, match="catalog hash"):
        builder.validate_payload(changed_catalog, compact=compact)

    fake_provider = copy.deepcopy(payload)
    fake_provider["case"]["memory_resolution"]["provider"][
        "call_performed"
    ] = True
    _resign(fake_provider)
    with pytest.raises(builder.PayloadError, match="provenance boundary"):
        builder.validate_payload(fake_provider, compact=compact)


def test_resolver_stages_extend_the_six_scientific_stages(committed):
    _, payload = committed
    sequence = [step["id"] for step in payload["demo"]["sequence"]]
    assert sequence == [
        "chat",
        "recall",
        "resolve-request",
        "resolve-proposal",
        "confirm",
        "forget",
        "re-query",
        "retained",
        "certificate",
    ]
    assert [
        step
        for step in sequence
        if step
        in {
            "chat",
            "recall",
            "forget",
            "re-query",
            "retained",
            "certificate",
        }
    ] == list(builder.EXPECTED_DEMO_SEQUENCE[:2]) + list(
        builder.EXPECTED_DEMO_SEQUENCE[5:]
    )


def test_displayed_public_chat_forbids_all_ellipsis_markers(committed):
    _, payload = committed
    for section in ("target", "retained", "intervening"):
        for quote in payload["case"]["dialogue"][section]["quotes"]:
            assert "..." not in quote["display_text"]
            assert "…" not in quote["display_text"]


def test_source_quote_drift_fails_after_resigning(committed):
    compact, payload = committed
    changed = copy.deepcopy(payload)
    changed["case"]["dialogue"]["target"]["quotes"][0][
        "display_text"
    ] += " fabricated suffix"
    _resign(changed)
    with pytest.raises(builder.PayloadError, match="source-bound"):
        builder.validate_payload(changed, compact=compact)


def test_source_hash_drift_fails_even_after_payload_is_resigned(committed):
    compact, payload = committed
    changed = copy.deepcopy(payload)
    changed["case"]["dialogue"]["intervening"]["quotes"][0][
        "source_turn_sha256"
    ] = "0" * 64
    _resign(changed)
    with pytest.raises(builder.PayloadError, match="binding drifted"):
        builder.validate_payload(changed, compact=compact)


def test_excerpt_text_cannot_be_changed_and_self_rehashed(committed):
    compact, payload = committed
    changed = copy.deepcopy(payload)
    quote = changed["case"]["dialogue"]["retained"]["quotes"][1]
    quote["display_text"] += " fabricated"
    quote["display_text_sha256"] = builder.longmemeval.text_sha256(
        quote["display_text"]
    )
    _resign(changed)
    with pytest.raises(builder.PayloadError, match="source-span binding"):
        builder.validate_payload(changed, compact=compact)


@pytest.mark.parametrize(
    "forbidden_key",
    ["raw_context_text", "generated_text", "model_output_text"],
)
def test_raw_or_generated_text_leakage_is_rejected(
    committed,
    forbidden_key,
):
    compact, payload = committed
    changed = copy.deepcopy(payload)
    changed["case"][forbidden_key] = "SHOULD-NOT-LEAK"
    _resign(changed)
    with pytest.raises(builder.PayloadError, match="forbidden"):
        builder.validate_payload(changed, compact=compact)


def test_aggregate_claims_are_rejected_before_final16(committed):
    compact, payload = committed
    changed = copy.deepcopy(payload)
    changed["scope"]["aggregate_claims"] = ["all methods succeed"]
    _resign(changed)
    with pytest.raises(builder.PayloadError, match="aggregate"):
        builder.validate_payload(changed, compact=compact)


def test_partial_report_requires_explicit_preview_acknowledgement(bound_inputs):
    with pytest.raises(PermissionError, match="allow-incomplete-preview"):
        builder.build_compact_case_report(
            bound_inputs["report"],
            bound_inputs["admission"],
            bound_inputs["manifest"],
            bound_inputs["method_lock"],
            bound_inputs["partial_lock"],
            bound_inputs["completion_lock"],
            report_file_sha256=bound_inputs["report_hash"],
            allow_incomplete_preview=False,
        )


def test_geometry_manifest_file_hash_drift_fails_before_use(
    tmp_path,
    bound_inputs,
):
    changed_manifest = tmp_path / "geometry.json"
    changed_manifest.write_bytes(builder.DEFAULT_MANIFEST.read_bytes() + b"\n")
    with pytest.raises(builder.PayloadError, match="SHA-256 drifted"):
        builder._load_bound_inputs(
            manifest_path=changed_manifest,
            policy_path=builder.DEFAULT_POLICY,
            census_path=builder.DEFAULT_CENSUS,
            admission_report_path=builder.DEFAULT_ADMISSION_REPORT,
            method_lock_path=builder.DEFAULT_METHOD_LOCK,
            partial_lock_path=builder.DEFAULT_PARTIAL_LOCK,
            completion_lock_path=None,
            method_report_path=builder.DEFAULT_METHOD_REPORT,
        )


def test_current_preview_rebuilds_byte_for_value_when_oracle_is_available(
    committed,
):
    required = (
        DEFAULT_ORACLE,
        builder.PREVIEW_METHOD_REPORT,
        builder.DEFAULT_ADMISSION_REPORT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("pinned oracle or bound local reports are unavailable")
    expected_compact, expected_payload = committed
    compact, payload = builder.build_from_paths(
        method_report_path=builder.PREVIEW_METHOD_REPORT,
        data_path=DEFAULT_ORACLE,
        allow_incomplete_preview=True,
    )
    assert compact == expected_compact
    assert payload == expected_payload
    consumed_compact, consumed_payload = builder.build_from_paths(
        method_report_path=builder.PREVIEW_METHOD_REPORT,
        data_path=DEFAULT_ORACLE,
        compact_input_path=COMPACT_PATH,
        allow_incomplete_preview=True,
    )
    assert consumed_compact == expected_compact
    assert consumed_payload == expected_payload


def test_current_final_rebuilds_byte_for_value_when_oracle_is_available():
    required = (
        DEFAULT_ORACLE,
        builder.DEFAULT_METHOD_REPORT,
        builder.DEFAULT_ADMISSION_REPORT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("pinned oracle or bound local reports are unavailable")
    expected_compact = _load(FINAL_COMPACT_PATH)
    expected_payload = _load(FINAL_PAYLOAD_PATH)
    compact, payload = builder.build_from_paths(data_path=DEFAULT_ORACLE)
    assert compact == expected_compact
    assert payload == expected_payload

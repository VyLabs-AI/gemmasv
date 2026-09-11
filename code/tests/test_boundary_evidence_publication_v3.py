from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from gemma_sv import publish_boundary_evidence_v3 as publisher


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_PATH = (
    ROOT
    / "gemma_sv"
    / "benchmarks"
    / "iclr_mass_preserving_boundary_v3.json"
)
HISTORICAL_PATH = (
    ROOT
    / "gemma_sv"
    / "benchmarks"
    / "iclr_mass_preserving_boundary_v2.json"
)


@pytest.fixture(scope="module")
def publication() -> dict:
    value = json.loads(PUBLICATION_PATH.read_text(encoding="utf-8"))
    publisher.validate_publication(value)
    return value


def _resign(value: dict) -> dict:
    return publisher._seal(value)


def test_corrected_fallback_populations_and_one_decimal_rates(publication):
    one_b = publication["certificates"]["one_b"]
    four_b = publication["certificates"]["four_b"]

    assert one_b["decrement_fallbacks"] == 396
    assert one_b["affected_decrement_attempts"] == 640
    assert one_b["enumerated_head_gates"] == 1024
    assert one_b["fallback_fraction"] == 396 / 640
    assert one_b["fallback_percent_1dp"] == 61.9

    assert four_b["decrement_fallbacks"] == 2020
    assert four_b["affected_decrement_attempts"] == 3040
    assert four_b["enumerated_head_gates"] == 4320
    assert four_b["fallback_fraction"] == 2020 / 3040
    assert four_b["fallback_percent_1dp"] == 66.4

    for certificate in (one_b, four_b):
        assert certificate["fallback_denominator"] == (
            "affected_decrement_attempts"
        )
        assert "head_gates" not in certificate
        assert certificate["fallback_percent_1dp"] * 10 == round(
            certificate["fallback_percent_1dp"] * 10
        )


def test_exactness_and_other_certificate_metrics_match_v2(publication):
    historical = json.loads(HISTORICAL_PATH.read_text(encoding="utf-8"))
    correction = publication["correction"]
    assert correction["exactness_metrics_unaffected"] is True
    assert correction[
        "certificate_metrics_except_fallback_rate_unaffected"
    ] is True
    for scale in ("one_b", "four_b"):
        current = publication["certificates"][scale]
        old = historical["certificates"][scale]
        for field in publisher.UNAFFECTED_CERTIFICATE_FIELDS:
            assert current[field] == old[field]


def test_every_source_binds_file_payload_and_schema(publication):
    artifacts = publication["provenance"]["source_artifacts"]
    assert set(artifacts) == set(publisher.SOURCE_SPECS)
    assert publication["provenance"]["source_artifact_count"] == len(artifacts)
    assert publication["provenance"]["payload_hash_contract"] == (
        publisher.PAYLOAD_HASH_CONTRACT
    )
    for key, spec in publisher.SOURCE_SPECS.items():
        binding = artifacts[key]
        assert binding == {
            "path": spec.path,
            "file_sha256": spec.file_sha256,
            "payload_sha256": spec.payload_sha256,
            "schema": spec.schema,
            "schema_version": spec.schema_version,
        }
        assert len(binding["file_sha256"]) == 64
        assert len(binding["payload_sha256"]) == 64
        assert binding["schema_version"] >= 1

    implementation = publication["provenance"]["implementation"]
    assert implementation == publisher._implementation_fingerprint()
    assert set(implementation["files_sha256"]) == set(
        publisher.IMPLEMENTATION_FILES
    )


def test_committed_publication_is_a_deterministic_rebuild(publication):
    missing = [
        spec.path
        for spec in publisher.SOURCE_SPECS.values()
        if not (ROOT / spec.path).is_file()
    ]
    if missing:
        pytest.skip("local source reports are unavailable")
    assert publisher.build_publication() == publication
    assert PUBLICATION_PATH.read_text(encoding="utf-8") == (
        publisher.deterministic_json(publication)
    )


def test_historical_v2_and_audit_remain_hash_frozen():
    for key in ("historical_v2", "denominator_audit"):
        spec = publisher.SOURCE_SPECS[key]
        observed = hashlib.sha256((ROOT / spec.path).read_bytes()).hexdigest()
        assert observed == spec.file_sha256


@pytest.mark.parametrize("path", publisher.PROTECTED_HISTORICAL_OUTPUTS)
def test_publisher_refuses_to_overwrite_historical_aggregates(
    publication, path
):
    with pytest.raises(publisher.PublicationError, match="refusing to overwrite"):
        publisher._write_atomic(path, publication)


@pytest.mark.parametrize(
    ("scale", "stale_fraction"),
    [
        ("one_b", 396 / 1024),
        ("four_b", 2020 / 4320),
    ],
)
def test_mixed_enumerated_denominators_are_rejected(
    publication, scale, stale_fraction
):
    changed = copy.deepcopy(publication)
    changed["certificates"][scale]["fallback_fraction"] = stale_fraction
    changed = _resign(changed)
    with pytest.raises(publisher.PublicationError, match="stale or mixed"):
        publisher.validate_publication(changed)


@pytest.mark.parametrize(
    ("scale", "stale_percent"),
    [("one_b", 38.7), ("four_b", 46.8)],
)
def test_stale_percentages_are_rejected(publication, scale, stale_percent):
    changed = copy.deepcopy(publication)
    changed["certificates"][scale]["fallback_percent_1dp"] = stale_percent
    changed = _resign(changed)
    with pytest.raises(
        publisher.PublicationError, match="rounding drift|stale"
    ):
        publisher.validate_publication(changed)


def test_wrong_one_decimal_rounding_is_rejected(publication):
    changed = copy.deepcopy(publication)
    changed["certificates"]["four_b"]["fallback_percent_1dp"] = 66.5
    changed = _resign(changed)
    with pytest.raises(publisher.PublicationError, match="rounding drift"):
        publisher.validate_publication(changed)


@pytest.mark.parametrize("hash_field", ["file_sha256", "payload_sha256"])
def test_embedded_source_hash_drift_is_rejected(
    publication, hash_field
):
    changed = copy.deepcopy(publication)
    changed["provenance"]["source_artifacts"]["one_b_certificate"][
        hash_field
    ] = "0" * 64
    changed = _resign(changed)
    with pytest.raises(
        publisher.PublicationError, match="source/hash/schema binding drift"
    ):
        publisher.validate_publication(changed)


def test_source_file_drift_fails_before_derivation(tmp_path):
    changed = tmp_path / "one_b_certificate.json"
    changed.write_text('{"records":[]}\n', encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="source/hash drift"):
        publisher._load_source("one_b_certificate", changed)


def test_exactness_metric_tampering_is_rejected(publication):
    changed = copy.deepcopy(publication)
    changed["certificates"]["one_b"][
        "maximum_exact_refit_kl_nats"
    ] = 1e-4
    changed = _resign(changed)
    with pytest.raises(
        publisher.PublicationError, match="unaffected certificate metric drift"
    ):
        publisher.validate_publication(changed)


@pytest.mark.parametrize(
    "flag",
    ["contains_source_text", "contains_generations", "contains_record_ids"],
)
def test_source_bearing_publication_is_rejected(publication, flag):
    changed = copy.deepcopy(publication)
    changed[flag] = True
    changed = _resign(changed)
    with pytest.raises(publisher.PublicationError, match="source-bearing"):
        publisher.validate_publication(changed)


def test_embedded_source_payload_key_is_rejected(publication):
    changed = copy.deepcopy(publication)
    changed["record_id"] = "should-not-be-public"
    changed = _resign(changed)
    with pytest.raises(publisher.PublicationError, match="source-bearing"):
        publisher.validate_publication(changed)


def test_nonfinite_publication_is_rejected(publication):
    changed = copy.deepcopy(publication)
    changed["certificates"]["one_b"]["fallback_fraction"] = math.inf
    with pytest.raises(publisher.PublicationError, match="non-finite"):
        publisher.validate_publication(changed)


def test_stale_percentages_are_absent_from_publication(publication):
    serialized = publisher.deterministic_json(publication)
    assert "38.7" not in serialized
    assert "46.8" not in serialized

"""Publish the hash-bound Gemma boundary-evidence denominator correction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gemma_sv.summarize_boundary_evidence import _attacks, _certificate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "gemma_sv/benchmarks/iclr_mass_preserving_boundary_v3.json"
PROTECTED_HISTORICAL_OUTPUTS = (
    ROOT / "gemma_sv/benchmarks/iclr_mass_preserving_boundary_v1.json",
    ROOT / "gemma_sv/benchmarks/iclr_mass_preserving_boundary_v2.json",
)
SCHEMA = "gemma-sv-mass-preserving-boundary-evidence-v3"
SCHEMA_VERSION = 3
PAYLOAD_HASH_CONTRACT = {
    "algorithm": "sha256",
    "canonicalization": "sorted-key compact UTF-8 JSON",
}


class PublicationError(ValueError):
    """Raised when evidence cannot be published without weakening its bindings."""


@dataclass(frozen=True)
class SourceSpec:
    path: str
    file_sha256: str
    payload_sha256: str
    schema: str
    schema_version: int
    required_keys: tuple[str, ...]


SOURCE_SPECS: dict[str, SourceSpec] = {
    "historical_v2": SourceSpec(
        "gemma_sv/benchmarks/iclr_mass_preserving_boundary_v2.json",
        "083771d30a908fdb20c5c8b96ffb023a4c714044d35b4cb7272a3fcf6bea22a2",
        "e01d19bf7023dd4e7743e5e1c87b79ef61eb34549f59ed78dff85f714051bc32",
        "gemma-sv-mass-preserving-boundary-evidence",
        2,
        ("schema", "certificates", "sha256"),
    ),
    "denominator_audit": SourceSpec(
        "gemma_sv/benchmarks/fallback_denominator_audit_v1.json",
        "b7e65af12be4f16ac6f218a2dcd46b4ced28d6b2f1e046be90d9dc207f9498d0",
        "5dd51bdae23c51beb2c187ae668701d3b4881304d890353ce8e331b4b4e8cfd6",
        "gemma-sv-fallback-denominator-audit",
        1,
        ("name", "version", "denominator_semantics", "results"),
    ),
    "configuration": SourceSpec(
        "gemma_sv/benchmarks/mass_preserving_boundary_v1.json",
        "2c9f198c597ce05fb579fce026de7ea624c15d6a7473ea2e7fb7f62a998f315d",
        "eb1b9b0490f3b8b5f78b8718b61b487628068d248740a92e495ae8194610d3e6",
        "gemma-sv-mass-preserving-boundary-configuration",
        1,
        ("name", "version", "configuration"),
    ),
    "certificate_protocol": SourceSpec(
        "gemma_sv/benchmarks/boundary_certificate_sweep_v1.json",
        "b99065792a6ebb7c327d626da4a80b628298515f662040a7fa0052057f77fedb",
        "dfcb75d2886f7d2ac1873e4fa38f13506351ce0151bfd901869c7c77be1dde5d",
        "gemma-sv-boundary-certificate-sweep-protocol",
        1,
        ("name", "version", "configuration", "certificate"),
    ),
    "confirmation_manifest": SourceSpec(
        "gemma_sv/benchmarks/whole_record_confirm_v2.json",
        "40f25b73e23ea33e2f7b70924e703d9ab4c1cd0476b35023a830f28d1ccc8daa",
        "bcf9d50f769f770979275237f9e62cf7c1b252d2de97bf6be0be97961ba7ef9d",
        "gemma-sv-whole-record-confirmation-manifest",
        1,
        ("name", "version", "records"),
    ),
    "one_b_ungrafted_behavior": SourceSpec(
        "outputs/gemma_sv_boundary_v1_confirm/1b-ungrafted.json",
        "f84d164689c20642404dbac246003514087975f2d668d7febdb540a4aac27798",
        "8586859beec6911fb502087ada56b6037c1c545326bd6580dca035cbe42a3d06",
        "gemma-sv-whole-record-admission-report",
        1,
        ("evaluation", "provenance", "whole_record"),
    ),
    "four_b_ungrafted_behavior": SourceSpec(
        "outputs/gemma_sv_boundary_v1_confirm/4b-ungrafted.json",
        "7e887540f28c525cf30052a0328a410f1c63e7b6be38b4a154c4a474553b92a2",
        "737642a2296242b28780eb6f091590a8fd55df3efd5a6a4a49e0d7aa7f78f8af",
        "gemma-sv-whole-record-admission-report",
        1,
        ("evaluation", "provenance", "whole_record"),
    ),
    "one_b_behavior": SourceSpec(
        "outputs/gemma_sv_boundary_v1_confirm/1b-graft.json",
        "c127048297d02daffde19e93ccaeb401bb4252af5a9dd721c452e6db45e93254",
        "04df469ee29a7771e6ab11048bb5891cd1c09b1dc8a8d26e0accd89af83e1ecc",
        "gemma-sv-whole-record-admission-report",
        1,
        ("evaluation", "provenance", "whole_record"),
    ),
    "four_b_behavior": SourceSpec(
        "outputs/gemma_sv_boundary_v1_confirm/4b-graft.json",
        "3b162d75806e5509d3116f9fcf9a2574c034124a5362d1104c4510e49869ca00",
        "2f067ff766b0b73b17232be70705e6effbc4a689ce243df11db9a09ce55933d7",
        "gemma-sv-whole-record-admission-report",
        1,
        ("evaluation", "provenance", "whole_record"),
    ),
    "one_b_certificate": SourceSpec(
        "outputs/gemma_sv_boundary_certificate_sweep/1b.json",
        "0594b52faf2aded7aab7460197d3b2c90f52826e583f02d981fe65674ff37aff",
        "1896f5a3a83343626cf98ec3d7abc9f3729fb673f5bc73f3cb17a43c8edbbdc2",
        "gemma-sv-whole-record-float64-certificate-report",
        1,
        ("evaluation", "provenance", "records"),
    ),
    "four_b_certificate": SourceSpec(
        "outputs/gemma_sv_boundary_certificate_sweep/4b.json",
        "d41fe201d8c4beb5b311938ef59ad395995722fe3be615478feb8eb2e0feec5e",
        "e41d7fef3e4992aa3ae864c2bb1e410f818643661fb38423d225d1e56dc1d55c",
        "gemma-sv-whole-record-float64-certificate-report",
        1,
        ("evaluation", "provenance", "records"),
    ),
    "one_b_quality": SourceSpec(
        "outputs/gemma_sv_training_free_quality/1b.json",
        "2ffcb025f93c0212c197478ba492cd668c3773c3350bdadeeec5bdcf360fe655",
        "baa1871f70cdd9adc96b90ca799e5c21bbaca2c5185ffd5eef413d02486f2845",
        "gemma-sv-training-free-quality-report",
        1,
        ("base_block_nll", "graft_block_nll", "summary"),
    ),
    "four_b_quality": SourceSpec(
        "outputs/gemma_sv_training_free_quality/4b.json",
        "56ec43b10ab6026ec72aa7726a35793c7f799a04a62a3e4867e2c19bc2965df9",
        "3189cbbeb2b54d5bee507f705805280fb2bf80f2b6eb25402d0c148e2bc1d264",
        "gemma-sv-training-free-quality-report",
        1,
        ("base_block_nll", "graft_block_nll", "summary"),
    ),
    "whole_record_attacks": SourceSpec(
        "outputs/gemma_sv_boundary_attack_suite/whole_record.json",
        "e3cb872f5a29b58117ffad0b367fefe251cccf9037874041dc9dd9c91ad8dda3",
        "287aea6ae43eea4adf9523bc7849385479a399fb4b2f7ac8619782a1df269605",
        "gemma-sv-boundary-whole-record-attack-report",
        1,
        ("attack_suite", "sampling", "whole_record"),
    ),
    "elicitation": SourceSpec(
        "outputs/gemma_sv_boundary_attack_suite/elicitation.json",
        "0c46009dd5d7d67a12a7bd29810543cc1aef5f8a558ecdcc5ff75246ded4f6f2",
        "adac1f152fa46aae29a91d1243804acc4855f9eb53c86f41b0ef53faabd6d314",
        "gemma-sv-whole-record-elicitation-report",
        1,
        ("configuration", "records", "shots"),
    ),
    "lira": SourceSpec(
        "outputs/gemma_sv_boundary_attack_suite/lira.json",
        "bf08327e3250bacbcbee8bb3c3b3839d598faf03880dbac8fca35a958665f2c6",
        "9d8f7bcad1a4292f043f9fe2eb391108a63d2f2748aab2b8e6498e1c215dec78",
        "gemma-sv-whole-record-lira",
        1,
        ("configuration", "metrics", "records"),
    ),
    "broad_lira": SourceSpec(
        "outputs/gemma_sv_lira_broad/lira.json",
        "33870321668d30806a6d93ebf4a8f286b793b676089fcb3fe7f46c5b822061a2",
        "9e4ca3e067042984cfea61daf7cd00199a9f8b7c54a123c4b6d5edd975f59b46",
        "gemma-sv-whole-record-lira",
        1,
        ("admission_population", "configuration", "metrics", "records"),
    ),
    "broad_lira_admission": SourceSpec(
        "outputs/gemma_sv_lira_broad/admission.json",
        "967ee74ab6b6a11ceb7dd936ab3332dc74989a6d82a5619a534377f8a9703c09",
        "0e0de49a8aa28106dc16249e8fa469440c454987dfb1d2fde9729fcf634f1293",
        "gemma-sv-whole-record-admission-report",
        1,
        ("evaluation", "provenance", "whole_record"),
    ),
    "fallback_diagnostic": SourceSpec(
        "outputs/gemma_sv_fallback_density/report.json",
        "dcf384927fa2c768923e6397d23047843ed91d445d5e1a8db2a6ae36036f2015",
        "9f6e7f047f94760d44c59c769ef6a53f5730a7a7b899c41f536c10b0782289e4",
        "gemma-sv-fallback-density-diagnostic",
        1,
        ("config", "rows"),
    ),
    "relearning": SourceSpec(
        "outputs/gemma_sv_boundary_attack_suite/relearning.json",
        "87d2bc0d5abf07fc9b86c1eb8c60231751a7c289e53db8b03c7bce4833cd6a6f",
        "6fbd8fdf49e28fedce421657c34ef6255420bdce063e93855644bd909d970da3",
        "gemma-sv-whole-record-relearning",
        1,
        ("budget_rows", "configuration", "schema"),
    ),
}

V2_SOURCE_KEYS = (
    "one_b_behavior",
    "four_b_behavior",
    "one_b_certificate",
    "four_b_certificate",
    "one_b_quality",
    "four_b_quality",
    "whole_record_attacks",
    "elicitation",
    "lira",
    "broad_lira",
    "broad_lira_admission",
    "fallback_diagnostic",
    "relearning",
)

DENOMINATOR_DEFINITIONS = {
    "enumerated_head_gates": (
        "All enumerated layer/boundary/head cells, including cells at boundaries "
        "that precede every deleted position and therefore attempt no decrement."
    ),
    "affected_decrement_attempts": (
        "Head-boundary cells with at least one forgotten position, where the "
        "implementation actually attempts an incremental decrement."
    ),
    "decrement_fallbacks": (
        "RuntimeError fallbacks raised inside an affected decrement attempt; this "
        "numerator is a subset of affected_decrement_attempts."
    ),
    "fallback_rate": (
        "decrement_fallbacks / affected_decrement_attempts; enumerated_head_gates "
        "is reported separately and is not the fallback denominator."
    ),
    "raw_field_mapping": {
        "record.head_gates": "enumerated_head_gates",
        "probe.head_gates": "affected_decrement_attempts",
        "record.total_fallbacks": "decrement_fallbacks",
    },
}

UNAFFECTED_CERTIFICATE_FIELDS = (
    "records",
    "probes",
    "maximum_exact_refit_kl_nats",
    "median_exact_refit_kl_nats",
    "probes_above_1e_6",
    "median_decay_refit_kl_nats",
    "maximum_decay_refit_kl_nats",
    "decrement_fallbacks",
)

CORRECTION = {
    "scope": "fallback-denominator-only",
    "only_changed_metric": (
        "fallback_fraction now divides by affected decrement attempts; the "
        "enumerated head-gate count remains separately disclosed"
    ),
    "exactness_metrics_unaffected": True,
    "certificate_metrics_except_fallback_rate_unaffected": True,
    "unaffected_certificate_fields": list(UNAFFECTED_CERTIFICATE_FIELDS),
    "reason_exactness_is_unaffected": (
        "A failed incremental decrement routes to the same retained-key exact "
        "refit used by the certificate. Changing the disclosed fallback "
        "denominator changes no state, probe, KL value, or fallback numerator."
    ),
    "denominator_definitions": DENOMINATOR_DEFINITIONS,
}

IMPLEMENTATION_FILES = (
    "gemma_sv/publish_boundary_evidence_v3.py",
    "gemma_sv/summarize_boundary_evidence.py",
    "gemma_sv/robust_eval.py",
    "gemma_sv/demo_server/certificate.py",
    "gemma_sv/certify_whole_record.py",
)

FORBIDDEN_PUBLICATION_KEYS = {
    "record_id",
    "record_ids",
    "source_text",
    "messages",
    "generations",
    "top_tokens",
    "target_probabilities",
    "question",
    "answer",
    "secret",
}


def payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _reject_constant(token: str) -> None:
    raise PublicationError(f"non-finite JSON constant {token!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PublicationError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _ensure_finite(value: Any, *, path: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PublicationError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _ensure_finite(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _ensure_finite(item, path=f"{path}[{index}]")
        return
    raise PublicationError(f"unsupported value at {path}: {type(value).__name__}")


def _load_source(key: str, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = SOURCE_SPECS[key]
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PublicationError(f"{key} source is unavailable: {path}") from error
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != spec.file_sha256:
        raise PublicationError(
            f"{key} source/hash drift: expected {spec.file_sha256}, "
            f"observed {file_sha256}"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"{key} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise PublicationError(f"{key} must contain a JSON object")
    _ensure_finite(value)
    missing = [name for name in spec.required_keys if name not in value]
    if missing:
        raise PublicationError(
            f"{key} violates {spec.schema}/v{spec.schema_version}; missing "
            + ", ".join(missing)
        )
    observed_payload_sha256 = payload_sha256(value)
    if observed_payload_sha256 != spec.payload_sha256:
        raise PublicationError(
            f"{key} payload/hash drift: expected {spec.payload_sha256}, "
            f"observed {observed_payload_sha256}"
        )
    binding = {
        "path": spec.path,
        "file_sha256": file_sha256,
        "payload_sha256": observed_payload_sha256,
        "schema": spec.schema,
        "schema_version": spec.schema_version,
    }
    return value, binding


def load_sources(
    paths: Mapping[str, Path] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    resolved = {
        key: Path((paths or {}).get(key, ROOT / spec.path))
        for key, spec in SOURCE_SPECS.items()
    }
    reports: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for key, path in resolved.items():
        reports[key], bindings[key] = _load_source(key, path)
    return reports, bindings


def _implementation_fingerprint() -> dict[str, Any]:
    files = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in IMPLEMENTATION_FILES
    }
    contract = {
        "schema_version": 1,
        "payload_hash_contract": PAYLOAD_HASH_CONTRACT,
        "source_contracts": {
            key: asdict(spec) for key, spec in SOURCE_SPECS.items()
        },
        "denominator_definitions": DENOMINATOR_DEFINITIONS,
        "unaffected_certificate_fields": UNAFFECTED_CERTIFICATE_FIELDS,
    }
    return {
        "schema_version": 1,
        "files_sha256": files,
        "files_payload_sha256": payload_sha256(files),
        "publication_contract_sha256": payload_sha256(contract),
    }


def _fixed_c_feasible_records(report: Mapping[str, Any]) -> int:
    return sum(
        bool(record["probe_certificates"])
        and all(
            probe["fixed_c_feasible"]
            for probe in record["probe_certificates"].values()
        )
        for record in report["records"]
    )


def _validate_source_relationships(reports: Mapping[str, Mapping[str, Any]]) -> None:
    historical = reports["historical_v2"]
    expected_v2_hashes = {
        key: SOURCE_SPECS[key].file_sha256 for key in V2_SOURCE_KEYS
    }
    if historical["schema"] != "gemma-sv-mass-preserving-boundary-evidence-v2":
        raise PublicationError("historical_v2 schema drift")
    if historical["sha256"] != expected_v2_hashes:
        raise PublicationError("historical_v2 source bindings drifted")

    configuration = reports["configuration"]
    if (
        configuration["name"] != "mass_preserving_boundary_v1"
        or configuration["version"] != 1
    ):
        raise PublicationError("configuration schema/name drift")
    protocol = reports["certificate_protocol"]
    if (
        protocol["name"] != "boundary_certificate_sweep_v1"
        or protocol["version"] != 1
        or protocol["configuration"] != "mass_preserving_boundary_v1"
        or protocol["records"] != 8
        or protocol["probes_per_record"] != 4
    ):
        raise PublicationError("certificate protocol drift")
    manifest = reports["confirmation_manifest"]
    if (
        manifest["name"] != "whole_record_confirm_v2"
        or manifest["version"] != 1
        or len(manifest["records"]) != 8
    ):
        raise PublicationError("confirmation manifest drift")

    for key in (
        "one_b_ungrafted_behavior",
        "four_b_ungrafted_behavior",
        "one_b_behavior",
        "four_b_behavior",
    ):
        report = reports[key]
        manifest_binding = report["provenance"]["manifest"]
        whole = report["whole_record"]
        if (
            manifest_binding["sha256"]
            != SOURCE_SPECS["confirmation_manifest"].file_sha256
            or whole["manifest"] != "whole_record_confirm_v2"
            or whole["manifest_version"] != 1
            or whole["attempted"] != 8
        ):
            raise PublicationError(f"{key} manifest/population drift")

    for scale in ("one_b", "four_b"):
        certificate = reports[f"{scale}_certificate"]
        behavior_key = f"{scale}_behavior"
        if (
            certificate["provenance"]["behavior_report_sha256"]
            != SOURCE_SPECS[behavior_key].file_sha256
            or certificate["provenance"]["manifest_sha256"]
            != SOURCE_SPECS["confirmation_manifest"].file_sha256
            or len(certificate["records"]) != 8
        ):
            raise PublicationError(f"{scale} certificate provenance drift")

    audit = reports["denominator_audit"]
    if audit["name"] != "fallback_denominator_audit_v1" or audit["version"] != 1:
        raise PublicationError("denominator audit schema/name drift")
    for scale in ("one_b", "four_b"):
        if (
            audit["sources"][scale]["sha256"]
            != SOURCE_SPECS[f"{scale}_certificate"].file_sha256
            or audit["sources"][scale]["hash_matches_published_binding"] is not True
        ):
            raise PublicationError(f"{scale} denominator-audit binding drift")

    broad = reports["broad_lira"]["admission_population"]
    broad_admission = reports["broad_lira_admission"]["whole_record"]
    if (
        broad["attempted"] != broad_admission["attempted"]
        or broad["admitted"] != broad_admission["admitted"]
    ):
        raise PublicationError("broad LiRA admission population drift")


def _build_evidence(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    historical = reports["historical_v2"]
    one_behavior = reports["one_b_behavior"]["whole_record"]
    four_behavior = reports["four_b_behavior"]["whole_record"]
    one_ungrafted = reports["one_b_ungrafted_behavior"]["whole_record"]
    four_ungrafted = reports["four_b_ungrafted_behavior"]["whole_record"]
    attempted = {
        one_behavior["attempted"],
        four_behavior["attempted"],
        one_ungrafted["attempted"],
        four_ungrafted["attempted"],
    }
    if len(attempted) != 1:
        raise PublicationError("admission reports use mixed attempted populations")

    configuration_source = reports["configuration"]["configuration"]
    configuration = {
        key: configuration_source[key] for key in historical["configuration"]
    }
    summary = {
        "schema": SCHEMA,
        "configuration": configuration,
        "admission": {
            "attempted": attempted.pop(),
            "one_b": {
                "ungrafted": one_ungrafted["admitted"],
                "grafted": one_behavior["admitted"],
                "fixed_c_feasible": _fixed_c_feasible_records(
                    reports["one_b_certificate"]
                ),
            },
            "four_b": {
                "ungrafted": four_ungrafted["admitted"],
                "grafted": four_behavior["admitted"],
                "fixed_c_feasible": _fixed_c_feasible_records(
                    reports["four_b_certificate"]
                ),
            },
        },
        "quality": {
            "one_b": reports["one_b_quality"]["summary"],
            "four_b": reports["four_b_quality"]["summary"],
        },
        "certificates": {
            "one_b": _certificate(reports["one_b_certificate"]),
            "four_b": _certificate(reports["four_b_certificate"]),
        },
        "behavioral_attacks": _attacks(
            reports["whole_record_attacks"],
            reports["elicitation"],
            reports["lira"],
            reports["relearning"],
        ),
        "limitations": {
            "single_checkpoint_per_scale": True,
            "one_b_does_not_match_base_admission": True,
            "future_safe": False,
            "block_groups_are_descriptive_not_independent": True,
        },
        "sha256": {
            key: SOURCE_SPECS[key].file_sha256 for key in V2_SOURCE_KEYS
        },
        "contains_source_text": False,
        "contains_generations": False,
        "contains_record_ids": False,
    }

    broad = reports["broad_lira"]
    summary["behavioral_attacks"]["lira_broad"] = {
        "admission": {
            "attempted": broad["admission_population"]["attempted"],
            "admitted": broad["admission_population"]["admitted"],
        },
        "records": len(broad["records"]),
        "metrics": broad["metrics"],
        "full_repack_fallbacks": sum(
            record["full_repack_fallbacks"] for record in broad["records"]
        ),
    }
    diagnostic = reports["fallback_diagnostic"]
    summary["fallback_density_diagnostic"] = {
        "scope": "one 4B record; identical contextualized keys across nu",
        "rows": [
            {
                "nu": row["nu"],
                "fallback_fraction": row["fallback_fraction"],
                "support_fraction": row["partition_diagnostics"][
                    "affected_support_fraction"
                ],
                "margin_fraction": row["partition_diagnostics"][
                    "affected_margin_fraction"
                ],
            }
            for row in diagnostic["rows"]
        ],
    }
    return summary


def _validate_certificates(
    publication: Mapping[str, Any],
    historical: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    expected_new_fields = {
        "affected_decrement_attempts",
        "enumerated_head_gates",
        "fallback_denominator",
        "fallback_fraction",
        "fallback_percent_1dp",
    }
    for scale in ("one_b", "four_b"):
        current = publication["certificates"][scale]
        old = historical["certificates"][scale]
        result = audit["results"][scale]
        expected_keys = (
            set(UNAFFECTED_CERTIFICATE_FIELDS) | expected_new_fields
        )
        if set(current) != expected_keys:
            raise PublicationError(f"{scale} certificate field/schema drift")
        for field in UNAFFECTED_CERTIFICATE_FIELDS:
            if current[field] != old[field]:
                raise PublicationError(
                    f"{scale} unaffected certificate metric drift: {field}"
                )
        fallbacks = result["decrement_fallbacks"]
        affected = result["affected_head_decrements"]
        enumerated = result["enumerated_head_gates"]
        if (
            current["decrement_fallbacks"] != fallbacks
            or current["affected_decrement_attempts"] != affected
            or current["enumerated_head_gates"] != enumerated
            or current["fallback_denominator"] != "affected_decrement_attempts"
        ):
            raise PublicationError(f"{scale} mixed fallback denominator")
        expected_fraction = fallbacks / affected
        expected_percent = round(expected_fraction * 100.0, 1)
        if current["fallback_fraction"] != expected_fraction:
            raise PublicationError(f"{scale} fallback fraction is stale or mixed")
        if current["fallback_percent_1dp"] != expected_percent:
            raise PublicationError(f"{scale} fallback percentage rounding drift")
        if current["fallback_percent_1dp"] in {38.7, 46.8}:
            raise PublicationError(f"{scale} stale fallback percentage")
        if old["head_gates"] != enumerated:
            raise PublicationError(f"{scale} historical enumerated-gate drift")
        if old["fallback_fraction"] != fallbacks / enumerated:
            raise PublicationError(f"{scale} historical denominator premise drift")
        if round(expected_fraction, 4) != result["corrected_fraction"]:
            raise PublicationError(f"{scale} denominator-audit fraction drift")


def _validate_against_historical(
    publication: Mapping[str, Any],
    historical: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    additions = {
        "status",
        "schema_version",
        "correction",
        "provenance",
        "integrity",
    }
    if set(publication) != set(historical) | additions:
        raise PublicationError("v3 top-level schema is not a narrow v2 correction")
    for key, old_value in historical.items():
        if key == "schema":
            if publication[key] != SCHEMA:
                raise PublicationError("v3 schema drift")
        elif key == "certificates":
            continue
        elif publication[key] != old_value:
            raise PublicationError(f"non-certificate v2 evidence drift: {key}")
    _validate_certificates(publication, historical, audit)


def _walk_forbidden_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_PUBLICATION_KEYS:
                raise PublicationError(f"source-bearing publication key at {path}.{key}")
            _walk_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_forbidden_keys(item, path=f"{path}[{index}]")


def _expected_source_bindings() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "path": spec.path,
            "file_sha256": spec.file_sha256,
            "payload_sha256": spec.payload_sha256,
            "schema": spec.schema,
            "schema_version": spec.schema_version,
        }
        for key, spec in SOURCE_SPECS.items()
    }


def _seal(publication: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(publication)
    sealed.pop("integrity", None)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "sorted-key compact UTF-8 JSON; integrity omitted",
        "sha256": payload_sha256(sealed),
    }
    return sealed


def validate_publication(publication: Mapping[str, Any]) -> None:
    if not isinstance(publication, Mapping):
        raise PublicationError("publication must be a JSON object")
    _ensure_finite(publication)
    _walk_forbidden_keys(publication)
    if (
        publication.get("schema") != SCHEMA
        or publication.get("schema_version") != SCHEMA_VERSION
        or publication.get("status") != "validated-denominator-correction"
    ):
        raise PublicationError("publication schema/status drift")
    for flag in (
        "contains_source_text",
        "contains_generations",
        "contains_record_ids",
    ):
        if publication.get(flag) is not False:
            raise PublicationError(f"source-bearing publication: {flag} must be false")
    if publication.get("correction") != CORRECTION:
        raise PublicationError("correction or denominator definitions drift")
    provenance = publication.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PublicationError("missing provenance")
    if provenance.get("source_artifacts") != _expected_source_bindings():
        raise PublicationError("source/hash/schema binding drift")
    if provenance.get("source_artifact_count") != len(SOURCE_SPECS):
        raise PublicationError("source artifact denominator drift")
    if provenance.get("payload_hash_contract") != PAYLOAD_HASH_CONTRACT:
        raise PublicationError("source payload-hash contract drift")
    if provenance.get("implementation") != _implementation_fingerprint():
        raise PublicationError("implementation fingerprint drift")

    integrity = publication.get("integrity")
    unsigned = dict(publication)
    unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != payload_sha256(unsigned)
    ):
        raise PublicationError("publication integrity drift")

    historical, _ = _load_source(
        "historical_v2", ROOT / SOURCE_SPECS["historical_v2"].path
    )
    audit, _ = _load_source(
        "denominator_audit", ROOT / SOURCE_SPECS["denominator_audit"].path
    )
    _validate_against_historical(publication, historical, audit)


def build_publication(
    paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    reports, source_bindings = load_sources(paths)
    _validate_source_relationships(reports)
    publication = _build_evidence(reports)
    publication.update(
        {
            "status": "validated-denominator-correction",
            "schema_version": SCHEMA_VERSION,
            "correction": CORRECTION,
            "provenance": {
                "source_artifact_count": len(source_bindings),
                "source_artifacts": source_bindings,
                "payload_hash_contract": PAYLOAD_HASH_CONTRACT,
                "implementation": _implementation_fingerprint(),
            },
        }
    )
    publication = _seal(publication)
    validate_publication(publication)
    return publication


def _write_atomic(path: Path, publication: Mapping[str, Any]) -> None:
    protected = {
        (ROOT / spec.path).resolve() for spec in SOURCE_SPECS.values()
    } | {historical.resolve() for historical in PROTECTED_HISTORICAL_OUTPUTS}
    if path.resolve() in protected:
        raise PublicationError(f"refusing to overwrite source artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(deterministic_json(publication), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key, spec in SOURCE_SPECS.items():
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=Path,
            default=ROOT / spec.path,
        )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--public-out", type=Path)
    args = parser.parse_args(argv)
    paths = {key: getattr(args, key) for key in SOURCE_SPECS}
    try:
        publication = build_publication(paths)
        outputs = [args.out]
        if args.public_out is not None:
            outputs.append(args.public_out)
        for output in outputs:
            _write_atomic(output, publication)
            print(f"wrote {output}")
    except PublicationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

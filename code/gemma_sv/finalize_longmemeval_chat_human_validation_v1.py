"""Finalize concordant HV2 human labels and emit source-free results.

The primary lock, secondary protocol, completed secondary packet, and hidden
mappings are verified before final labels are constructed. If primary and
secondary humans disagree on a label or leak subtype, this module fails closed
and requires a separate third-rater adjudication workflow.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from gemma_sv import build_longmemeval_chat_human_validation_secondary_v1 as secondary
from gemma_sv import validate_longmemeval_chat_human_validation_primary_v2 as primary


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
REVIEW_ROOT = (
    REPOSITORY
    / "outputs/gemma_sv_rag/longmemeval_chat_human_validation_v2"
)

DEFAULT_PRIMARY_PACKET = primary.DEFAULT_PACKET
DEFAULT_PRIMARY_PROTOCOL = primary.DEFAULT_PROTOCOL
DEFAULT_ANALYSIS_PLAN = primary.DEFAULT_ANALYSIS_PLAN
DEFAULT_PRIMARY_LOCK = primary.DEFAULT_LOCK
DEFAULT_UNBLINDING = secondary.DEFAULT_UNBLINDING
DEFAULT_SECONDARY_PROTOCOL = secondary.DEFAULT_SECONDARY_PROTOCOL
DEFAULT_SECONDARY_PACKET = secondary.DEFAULT_SECONDARY_PACKET
DEFAULT_SECONDARY_MAPPING = secondary.DEFAULT_SECONDARY_MAPPING
DEFAULT_SECONDARY_LOCK = (
    BENCHMARKS / "longmemeval_chat_human_validation_secondary_lock_v1.json"
)
DEFAULT_RESULTS = (
    BENCHMARKS / "longmemeval_chat_human_validation_results_v1.json"
)
DEFAULT_FINAL_LEDGER = REVIEW_ROOT / "final_human_ledger.json"

SECONDARY_LOCK_SCHEMA = (
    "gemma-sv-longmemeval-human-validation-secondary-lock-v1"
)
RESULTS_SCHEMA = "gemma-sv-longmemeval-human-validation-results-v1"
FINAL_LEDGER_SCHEMA = (
    "gemma-sv-longmemeval-human-validation-final-local-ledger-v1"
)
CONDITION_NAMES = {
    0: "present",
    1: "fresh_rebuild",
    2: "edited_policy",
    3: "prompt_only",
}
PARTIAL_REFERENCE_REVIEW_IDS = {
    "HV2-220c2270421b",
    "HV2-360ffb110a98",
}
ALIAS_SENSITIVITY_REVIEW_ID = "HV2-6f0e93cae4e0"
PRIMARY_AMBIGUOUS_REVIEW_ID = "HV2-c06890edf0b9"


class FinalizationError(ValueError):
    """The completed human-label workflow is invalid or incomplete."""


class AdjudicationRequiredError(FinalizationError):
    """Primary and secondary humans disagree and require a third rater."""


def _write_new(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, mode)


def _counts(
    rows: list[Mapping[str, Any]],
    field: str,
    categories: tuple[str, ...],
) -> dict[str, int]:
    observed = Counter(str(row[field]) for row in rows)
    return {category: observed.get(category, 0) for category in categories}


def validate_completed_secondary(
    *,
    protocol: Mapping[str, Any],
    packet: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate completed secondary fields and bind them to hidden groups."""

    protocol_integrity = primary._integrity_sha256(
        protocol,
        name="secondary protocol",
    )
    primary._integrity_sha256(mapping, name="secondary hidden mapping")
    if (
        protocol.get("schema") != secondary.SECONDARY_PROTOCOL_SCHEMA
        or protocol.get("status") != "frozen-before-secondary-human-labels"
    ):
        raise FinalizationError("unexpected secondary protocol")
    if (
        packet.get("schema") != secondary.SECONDARY_PACKET_SCHEMA
        or packet.get("status") != "awaiting-secondary-human-labels"
        or packet.get("secondary_protocol_integrity_sha256")
        != protocol_integrity
    ):
        raise FinalizationError("unexpected completed secondary packet")
    if (
        mapping.get("schema") != secondary.SECONDARY_MAPPING_SCHEMA
        or mapping.get("status") != "sealed-before-secondary-labels"
        or mapping.get("secondary_protocol_integrity_sha256")
        != protocol_integrity
    ):
        raise FinalizationError("unexpected secondary hidden mapping")

    packet_units = packet.get("units")
    mapping_units = mapping.get("units")
    if not isinstance(packet_units, list) or not isinstance(mapping_units, list):
        raise FinalizationError("secondary unit arrays are missing")
    mapping_by_id = {
        str(unit["secondary_review_id"]): unit for unit in mapping_units
    }
    if len(mapping_by_id) != len(mapping_units):
        raise FinalizationError("duplicate secondary mapping IDs")

    groups: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for expected_ordinal, visible in enumerate(packet_units, start=1):
        if (
            not isinstance(visible, Mapping)
            or set(visible) != secondary.SECONDARY_UNIT_FIELDS
            or visible.get("secondary_ordinal") != expected_ordinal
        ):
            raise FinalizationError("completed secondary unit fields differ")
        secondary_id = visible.get("secondary_review_id")
        if not isinstance(secondary_id, str) or secondary_id in seen_ids:
            raise FinalizationError("completed secondary review IDs are invalid")
        seen_ids.add(secondary_id)
        hidden = mapping_by_id.get(secondary_id)
        if hidden is None:
            raise FinalizationError("secondary packet/mapping IDs differ")
        expected_binding = primary._payload_sha256(
            {
                "question": visible["question"],
                "reference": visible["reference"],
                "candidate": visible["candidate"],
            }
        )
        if hidden.get("triple_binding_sha256") != expected_binding:
            raise FinalizationError("completed secondary triple binding differs")

        label = visible.get("secondary_label")
        match_type = visible.get("secondary_match_type")
        if label not in primary.LABEL_TO_MATCH_TYPES:
            raise FinalizationError("secondary label is incomplete or invalid")
        if match_type not in primary.LABEL_TO_MATCH_TYPES[label]:
            raise FinalizationError("secondary label/match-type pairing is invalid")
        notes = visible.get("secondary_notes")
        if notes is not None and not isinstance(notes, str):
            raise FinalizationError("secondary notes must be text or null")
        if label == "ambiguous" and (
            not isinstance(notes, str) or not notes.strip()
        ):
            raise FinalizationError("secondary ambiguous label requires notes")

        occurrences = hidden.get("primary_occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise FinalizationError("secondary hidden occurrences are missing")
        primary_labels = {row["primary_label"] for row in occurrences}
        primary_match_types = {row["primary_match_type"] for row in occurrences}
        if len(primary_labels) != 1 or len(primary_match_types) != 1:
            raise FinalizationError("duplicate primary human decisions differ")
        primary_label = str(next(iter(primary_labels)))
        primary_match_type = str(next(iter(primary_match_types)))
        label_concordant = label == primary_label
        match_type_concordant = (
            label != "leak" or match_type == primary_match_type
        )
        groups.append(
            {
                "secondary_review_id": secondary_id,
                "triple_binding_sha256": expected_binding,
                "secondary_label": label,
                "secondary_match_type": match_type,
                "primary_label": primary_label,
                "primary_match_type": primary_match_type,
                "label_concordant": label_concordant,
                "match_type_concordant": match_type_concordant,
                "primary_occurrences": occurrences,
            }
        )
    if seen_ids != set(mapping_by_id):
        raise FinalizationError("secondary packet/mapping unit sets differ")
    expected_groups = protocol["selection"]["secondary_distinct_triples"]
    if len(groups) != expected_groups:
        raise FinalizationError("completed secondary group count differs")
    return groups


def build_final_outputs(
    *,
    primary_packet: Mapping[str, Any],
    primary_lock: Mapping[str, Any],
    unblinding: Mapping[str, Any],
    unblinding_file_sha256: str,
    secondary_protocol: Mapping[str, Any],
    secondary_packet: Mapping[str, Any],
    secondary_packet_file_sha256: str,
    secondary_mapping: Mapping[str, Any],
    secondary_mapping_file_sha256: str,
    secondary_packet_mode: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build secondary lock, local final ledger, and public aggregate."""

    groups = validate_completed_secondary(
        protocol=secondary_protocol,
        packet=secondary_packet,
        mapping=secondary_mapping,
    )
    needs_adjudication = [
        group
        for group in groups
        if not group["label_concordant"] or not group["match_type_concordant"]
    ]
    if needs_adjudication:
        raise AdjudicationRequiredError(
            f"{len(needs_adjudication)} distinct triples require adjudication"
        )

    protocol_integrity = primary._integrity_sha256(
        secondary_protocol,
        name="secondary protocol",
    )
    mapping_integrity = primary._integrity_sha256(
        secondary_mapping,
        name="secondary hidden mapping",
    )
    unblinding_by_id = secondary.validate_unblinding_ledger(
        unblinding,
        primary_units=primary_packet["units"],
        protocol_integrity_sha256=str(
            primary_packet["protocol_integrity_sha256"]
        ),
    )
    unblinding_integrity = unblinding_by_id.pop("_ledger_integrity")["sha256"]

    secondary_by_primary_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for occurrence in group["primary_occurrences"]:
            review_id = str(occurrence["review_id"])
            if review_id in secondary_by_primary_id:
                raise FinalizationError("primary occurrence maps to two secondary units")
            secondary_by_primary_id[review_id] = group

    primary_by_id = {
        str(unit["review_id"]): unit for unit in primary_packet["units"]
    }
    final_rows: list[dict[str, Any]] = []
    triple_labels: dict[str, tuple[str, str]] = {}
    for review_id, visible in primary_by_id.items():
        hidden = unblinding_by_id[review_id]
        reviewed = secondary_by_primary_id.get(review_id)
        if reviewed is None:
            final_label = str(visible["primary_label"])
            final_match_type = str(visible["primary_match_type"])
            review_stage = "primary_only"
            secondary_id = None
        else:
            final_label = str(reviewed["secondary_label"])
            final_match_type = str(reviewed["secondary_match_type"])
            review_stage = "primary_secondary_concordant"
            secondary_id = reviewed["secondary_review_id"]
        triple_binding = primary._payload_sha256(
            {
                "question": visible["question"],
                "reference": visible["reference"],
                "candidate": visible["candidate"],
            }
        )
        decision = (final_label, final_match_type)
        previous = triple_labels.setdefault(triple_binding, decision)
        if previous != decision:
            raise FinalizationError("final duplicate-triple decisions differ")
        final_rows.append(
            {
                "review_id": review_id,
                "triple_binding_sha256": triple_binding,
                "primary_label": visible["primary_label"],
                "primary_match_type": visible["primary_match_type"],
                "secondary_review_id": secondary_id,
                "secondary_label": (
                    None if reviewed is None else reviewed["secondary_label"]
                ),
                "secondary_match_type": (
                    None
                    if reviewed is None
                    else reviewed["secondary_match_type"]
                ),
                "final_label": final_label,
                "final_match_type": final_match_type,
                "human_review_stage": review_stage,
                "luna_label": hidden["luna_label"],
                "selection_role": hidden["selection_role"],
                "condition_ordinal": hidden["condition_ordinal"],
                "condition": CONDITION_NAMES[int(hidden["condition_ordinal"])],
                "cluster_index": hidden["cluster_index"],
                "history_index": hidden["history_index"],
                "variant_index": hidden["variant_index"],
                "unit_binding_sha256": hidden["unit_binding_sha256"],
            }
        )

    label_categories = ("leak", "no_leak", "ambiguous")
    match_categories = primary.MATCH_TYPES
    unique_rows = [
        {
            "final_label": decision[0],
            "final_match_type": decision[1],
        }
        for decision in triple_labels.values()
    ]
    by_role = {}
    for role in secondary.EXPECTED_ROLES:
        rows = [row for row in final_rows if row["selection_role"] == role]
        by_role[role] = {
            "unit_count": len(rows),
            "final_label_counts": _counts(rows, "final_label", label_categories),
            "final_match_type_counts": _counts(
                rows,
                "final_match_type",
                match_categories,
            ),
        }
    by_condition = {}
    for ordinal, name in CONDITION_NAMES.items():
        rows = [row for row in final_rows if row["condition_ordinal"] == ordinal]
        by_condition[name] = {
            "condition_ordinal": ordinal,
            "review_unit_count": len(rows),
            "selection_role_counts": dict(
                sorted(Counter(row["selection_role"] for row in rows).items())
            ),
            "final_label_counts": _counts(rows, "final_label", label_categories),
        }

    secondary_visible_rows = [
        {
            "secondary_label": group["secondary_label"],
            "secondary_match_type": group["secondary_match_type"],
        }
        for group in groups
    ]
    secondary_lock = primary._seal(
        {
            "schema": SECONDARY_LOCK_SCHEMA,
            "schema_version": 1,
            "status": "secondary-labels-frozen-before-final-aggregation",
            "source_free": True,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "secondary_protocol_integrity_sha256": protocol_integrity,
            "secondary_packet": {
                "path": primary._repository_path(DEFAULT_SECONDARY_PACKET),
                "file_sha256": secondary_packet_file_sha256,
                "mode": f"{secondary_packet_mode:04o}",
                "read_only": not bool(secondary_packet_mode & 0o222),
                "distinct_triple_count": len(groups),
                "label_counts": _counts(
                    secondary_visible_rows,
                    "secondary_label",
                    label_categories,
                ),
                "match_type_counts": _counts(
                    secondary_visible_rows,
                    "secondary_match_type",
                    match_categories,
                ),
            },
            "secondary_hidden_mapping": {
                "file_sha256": secondary_mapping_file_sha256,
                "integrity_sha256": mapping_integrity,
            },
            "rater_attestation": {
                "basis": "reported by the independent secondary rater",
                "different_researcher_from_primary": True,
                "primary_results_seen": False,
                "other_project_files_opened": False,
                "external_search_or_lookup_used": False,
                "only_secondary_human_fields_edited": True,
            },
        }
    )

    bindings = {
        "primary_lock_integrity_sha256": primary._integrity_sha256(
            primary_lock,
            name="primary lock",
        ),
        "unblinding_ledger_file_sha256": unblinding_file_sha256,
        "unblinding_ledger_integrity_sha256": unblinding_integrity,
        "secondary_protocol_integrity_sha256": protocol_integrity,
        "secondary_packet_file_sha256": secondary_packet_file_sha256,
        "secondary_mapping_file_sha256": secondary_mapping_file_sha256,
        "secondary_mapping_integrity_sha256": mapping_integrity,
    }
    final_ledger = primary._seal(
        {
            "schema": FINAL_LEDGER_SCHEMA,
            "schema_version": 1,
            "status": "human-labels-final-no-third-rater-required",
            "local_only": True,
            "source_bearing": False,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "bindings": bindings,
            "units": final_rows,
        }
    )

    strict_counts = _counts(final_rows, "final_label", label_categories)
    if not PARTIAL_REFERENCE_REVIEW_IDS <= set(primary_by_id):
        raise FinalizationError("partial-reference sensitivity IDs are missing")
    if ALIAS_SENSITIVITY_REVIEW_ID not in primary_by_id:
        raise FinalizationError("alias sensitivity ID is missing")
    if PRIMARY_AMBIGUOUS_REVIEW_ID not in primary_by_id:
        raise FinalizationError("primary ambiguous sensitivity ID is missing")
    if any(
        primary_by_id[review_id]["primary_label"] != "no_leak"
        for review_id in PARTIAL_REFERENCE_REVIEW_IDS
    ):
        raise FinalizationError("partial-reference strict labels differ")
    if (
        primary_by_id[ALIAS_SENSITIVITY_REVIEW_ID]["primary_label"]
        != "no_leak"
        or primary_by_id[PRIMARY_AMBIGUOUS_REVIEW_ID]["primary_label"]
        != "ambiguous"
    ):
        raise FinalizationError("predeclared sensitivity labels differ")

    results = primary._seal(
        {
            "schema": RESULTS_SCHEMA,
            "schema_version": 1,
            "status": "primary-secondary-complete-no-third-rater-required",
            "source_free": True,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "interpretation": (
                "instrument validation on an enriched packet; not a leakage "
                "prevalence or model-condition efficacy estimate"
            ),
            "bindings": bindings,
            "primary": {
                "review_unit_count": len(final_rows),
                "unique_triple_count": len(unique_rows),
                "final_label_counts": strict_counts,
                "final_match_type_counts": _counts(
                    final_rows,
                    "final_match_type",
                    match_categories,
                ),
                "unique_triple_label_counts": _counts(
                    unique_rows,
                    "final_label",
                    label_categories,
                ),
                "unique_triple_match_type_counts": _counts(
                    unique_rows,
                    "final_match_type",
                    match_categories,
                ),
            },
            "secondary": {
                "selected_primary_occurrences": sum(
                    len(group["primary_occurrences"]) for group in groups
                ),
                "distinct_triples_reviewed": len(groups),
                "duplicate_occurrences_collapsed": sum(
                    len(group["primary_occurrences"]) for group in groups
                )
                - len(groups),
                "label_counts": _counts(
                    secondary_visible_rows,
                    "secondary_label",
                    label_categories,
                ),
                "match_type_counts": _counts(
                    secondary_visible_rows,
                    "secondary_match_type",
                    match_categories,
                ),
                "primary_secondary_label_concordant_distinct_triples": sum(
                    bool(group["label_concordant"]) for group in groups
                ),
                "primary_secondary_label_discordant_distinct_triples": 0,
                "third_rater_required": False,
            },
            "by_luna_selection_role": by_role,
            "by_condition": by_condition,
            "predeclared_sensitivities": {
                "partial_reference": {
                    "strict_label": "no_leak",
                    "supplemental_label": "partial_disclosure",
                    "review_unit_count": len(PARTIAL_REFERENCE_REVIEW_IDS),
                    "distinct_triple_count": 1,
                    "atomic_fact_counts": {
                        "full_leak": strict_counts["leak"],
                        "partial_disclosure": len(
                            PARTIAL_REFERENCE_REVIEW_IDS
                        ),
                        "no_disclosure": strict_counts["no_leak"]
                        - len(PARTIAL_REFERENCE_REVIEW_IDS),
                        "ambiguous": strict_counts["ambiguous"],
                    },
                },
                "alias_sufficiency": {
                    "review_unit_count": 1,
                    "strict_label": "no_leak",
                    "alternative_label": "leak",
                    "alternative_full_reference_counts": {
                        "leak": strict_counts["leak"] + 1,
                        "no_leak": strict_counts["no_leak"] - 1,
                        "ambiguous": strict_counts["ambiguous"],
                    },
                },
                "pragmatic_yes_no": {
                    "review_unit_count": 1,
                    "final_label": "ambiguous",
                },
            },
            "supplemental_luna_ambiguous_outputs": {
                "count": 2,
                "included_in_primary_endpoint": False,
                "status": "separate-supplement-not-generated",
            },
            "reporting_constraints": {
                "packet_15_of_38_is_not_prevalence": True,
                "combined_unweighted_accuracy_prohibited": True,
                "selective_secondary_agreement_statistic_prohibited": True,
                "duplicate_aware_reporting_required": True,
                "post_hoc_normalizer_requires_fresh_validation": True,
            },
        }
    )
    return secondary_lock, final_ledger, results


def finalize(
    *,
    primary_packet_path: Path = DEFAULT_PRIMARY_PACKET,
    primary_protocol_path: Path = DEFAULT_PRIMARY_PROTOCOL,
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
    primary_lock_path: Path = DEFAULT_PRIMARY_LOCK,
    unblinding_path: Path = DEFAULT_UNBLINDING,
    secondary_protocol_path: Path = DEFAULT_SECONDARY_PROTOCOL,
    secondary_packet_path: Path = DEFAULT_SECONDARY_PACKET,
    secondary_mapping_path: Path = DEFAULT_SECONDARY_MAPPING,
    secondary_lock_path: Path = DEFAULT_SECONDARY_LOCK,
    results_path: Path = DEFAULT_RESULTS,
    final_ledger_path: Path = DEFAULT_FINAL_LEDGER,
) -> dict[str, Any]:
    """Validate, freeze, and finalize a concordant secondary pass."""

    expected_primary_lock = primary.build_lock(
        packet_path=primary_packet_path,
        protocol_path=primary_protocol_path,
        analysis_plan_path=analysis_plan_path,
    )
    observed_primary_lock = primary._load_mapping(
        primary_lock_path,
        name="primary lock",
    )
    if observed_primary_lock != expected_primary_lock:
        raise FinalizationError("primary lock is stale")

    primary_packet = primary._load_mapping(
        primary_packet_path,
        name="primary packet",
    )
    secondary_protocol = primary._load_mapping(
        secondary_protocol_path,
        name="secondary protocol",
    )
    secondary_packet = primary._load_mapping(
        secondary_packet_path,
        name="completed secondary packet",
    )
    secondary_mapping = primary._load_mapping(
        secondary_mapping_path,
        name="secondary hidden mapping",
    )
    validate_completed_secondary(
        protocol=secondary_protocol,
        packet=secondary_packet,
        mapping=secondary_mapping,
    )
    os.chmod(secondary_packet_path, 0o400)
    secondary_packet_mode = stat.S_IMODE(secondary_packet_path.stat().st_mode)

    unblinding = primary._load_mapping(
        unblinding_path,
        name="unblinding ledger",
    )
    outputs = build_final_outputs(
        primary_packet=primary_packet,
        primary_lock=observed_primary_lock,
        unblinding=unblinding,
        unblinding_file_sha256=primary._file_sha256(unblinding_path),
        secondary_protocol=secondary_protocol,
        secondary_packet=secondary_packet,
        secondary_packet_file_sha256=primary._file_sha256(
            secondary_packet_path
        ),
        secondary_mapping=secondary_mapping,
        secondary_mapping_file_sha256=primary._file_sha256(
            secondary_mapping_path
        ),
        secondary_packet_mode=secondary_packet_mode,
    )
    destinations = (
        (secondary_lock_path, outputs[0], 0o644),
        (final_ledger_path, outputs[1], 0o400),
        (results_path, outputs[2], 0o644),
    )
    occupied = [
        str(path)
        for path, _, _ in destinations
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise FinalizationError(f"final output paths already exist: {occupied}")
    created: list[Path] = []
    try:
        for path, payload, mode in destinations:
            _write_new(path, payload, mode=mode)
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    results = outputs[2]
    return {
        "status": results["status"],
        "review_unit_count": results["primary"]["review_unit_count"],
        "unique_triple_count": results["primary"]["unique_triple_count"],
        "final_label_counts": results["primary"]["final_label_counts"],
        "secondary_distinct_triples": results["secondary"][
            "distinct_triples_reviewed"
        ],
        "third_rater_required": results["secondary"]["third_rater_required"],
    }


def load_validated_final_outputs(
    *,
    primary_packet_path: Path = DEFAULT_PRIMARY_PACKET,
    primary_protocol_path: Path = DEFAULT_PRIMARY_PROTOCOL,
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
    primary_lock_path: Path = DEFAULT_PRIMARY_LOCK,
    unblinding_path: Path = DEFAULT_UNBLINDING,
    secondary_protocol_path: Path = DEFAULT_SECONDARY_PROTOCOL,
    secondary_packet_path: Path = DEFAULT_SECONDARY_PACKET,
    secondary_mapping_path: Path = DEFAULT_SECONDARY_MAPPING,
    secondary_lock_path: Path = DEFAULT_SECONDARY_LOCK,
    results_path: Path = DEFAULT_RESULTS,
    final_ledger_path: Path = DEFAULT_FINAL_LEDGER,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce final outputs exactly without chmod, writes, or network calls."""

    expected_primary_lock = primary.build_lock(
        packet_path=primary_packet_path,
        protocol_path=primary_protocol_path,
        analysis_plan_path=analysis_plan_path,
    )
    observed_primary_lock = primary._load_mapping(
        primary_lock_path,
        name="primary lock",
    )
    if observed_primary_lock != expected_primary_lock:
        raise FinalizationError("primary lock is stale")

    mode_expectations = (
        (primary_packet_path, 0o400),
        (secondary_packet_path, 0o400),
        (secondary_mapping_path, 0o400),
        (final_ledger_path, 0o400),
        (secondary_lock_path, 0o644),
        (results_path, 0o644),
    )
    for path, expected_mode in mode_expectations:
        observed_mode = stat.S_IMODE(path.stat().st_mode)
        if observed_mode != expected_mode:
            raise FinalizationError(
                f"unexpected final artifact mode for {path}: {observed_mode:04o}"
            )

    primary_packet = primary._load_mapping(
        primary_packet_path,
        name="primary packet",
    )
    unblinding = primary._load_mapping(
        unblinding_path,
        name="unblinding ledger",
    )
    secondary_protocol = primary._load_mapping(
        secondary_protocol_path,
        name="secondary protocol",
    )
    secondary_packet = primary._load_mapping(
        secondary_packet_path,
        name="secondary packet",
    )
    secondary_mapping = primary._load_mapping(
        secondary_mapping_path,
        name="secondary hidden mapping",
    )
    expected_secondary_lock, expected_final_ledger, expected_results = (
        build_final_outputs(
            primary_packet=primary_packet,
            primary_lock=observed_primary_lock,
            unblinding=unblinding,
            unblinding_file_sha256=primary._file_sha256(unblinding_path),
            secondary_protocol=secondary_protocol,
            secondary_packet=secondary_packet,
            secondary_packet_file_sha256=primary._file_sha256(
                secondary_packet_path
            ),
            secondary_mapping=secondary_mapping,
            secondary_mapping_file_sha256=primary._file_sha256(
                secondary_mapping_path
            ),
            secondary_packet_mode=stat.S_IMODE(
                secondary_packet_path.stat().st_mode
            ),
        )
    )
    observed_secondary_lock = primary._load_mapping(
        secondary_lock_path,
        name="secondary lock",
    )
    observed_final_ledger = primary._load_mapping(
        final_ledger_path,
        name="final human ledger",
    )
    observed_results = primary._load_mapping(
        results_path,
        name="human-validation results",
    )
    if observed_secondary_lock != expected_secondary_lock:
        raise FinalizationError("secondary lock is stale")
    if observed_final_ledger != expected_final_ledger:
        raise FinalizationError("final human ledger is stale")
    if observed_results != expected_results:
        raise FinalizationError("human-validation results are stale")
    return observed_results, observed_final_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--secondary-lock",
        type=Path,
        default=DEFAULT_SECONDARY_LOCK,
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--final-ledger",
        type=Path,
        default=DEFAULT_FINAL_LEDGER,
    )
    args = parser.parse_args()
    summary = finalize(
        secondary_lock_path=args.secondary_lock,
        results_path=args.results,
        final_ledger_path=args.final_ledger,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

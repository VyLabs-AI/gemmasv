"""Generate the blinded HV2 secondary packet under controlled unblinding.

The completed primary packet and pre-unblinding plan are verified against the
source-free primary lock before this module reads the unblinding ledger. The
secondary packet hides primary labels, Luna labels, conditions, selection
roles, review IDs, and duplicate multiplicities.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from gemma_sv import validate_longmemeval_chat_human_validation_primary_v2 as primary


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
REVIEW_ROOT = (
    REPOSITORY
    / "outputs/gemma_sv_rag/longmemeval_chat_human_validation_v2"
)

DEFAULT_PACKET = primary.DEFAULT_PACKET
DEFAULT_PROTOCOL = primary.DEFAULT_PROTOCOL
DEFAULT_ANALYSIS_PLAN = primary.DEFAULT_ANALYSIS_PLAN
DEFAULT_PRIMARY_LOCK = primary.DEFAULT_LOCK
DEFAULT_UNBLINDING = REVIEW_ROOT / "unblinding_ledger.json"
DEFAULT_SECONDARY_PROTOCOL = (
    BENCHMARKS / "longmemeval_chat_human_validation_secondary_protocol_v1.json"
)
DEFAULT_SECONDARY_PACKET = REVIEW_ROOT / "secondary_packet.json"
DEFAULT_SECONDARY_MAPPING = REVIEW_ROOT / "secondary_hidden_mapping.json"

SECONDARY_PROTOCOL_SCHEMA = (
    "gemma-sv-longmemeval-human-validation-secondary-protocol-v1"
)
SECONDARY_PACKET_SCHEMA = (
    "gemma-sv-longmemeval-human-validation-secondary-packet-v1"
)
SECONDARY_MAPPING_SCHEMA = (
    "gemma-sv-longmemeval-human-validation-secondary-hidden-mapping-v1"
)
UNBLINDING_SCHEMA = "gemma-sv-longmemeval-human-validation-unblinding-v2"
SECONDARY_REVIEW_ID_PATTERN = re.compile(r"^SHV2-[0-9a-f]{12}$")
SECONDARY_ID_NAMESPACE = "longmemeval-human-validation-v2-secondary-id"
SECONDARY_ORDER_NAMESPACE = "longmemeval-human-validation-v2-secondary-order"
EXPECTED_ROLES = {
    "luna_flagged_matcher_miss": 19,
    "luna_negative_control": 19,
}
LEDGER_UNIT_FIELDS = {
    "review_id",
    "unit_binding_sha256",
    "selection_role",
    "luna_label",
    "condition_ordinal",
    "cluster_index",
    "history_index",
    "variant_index",
    "source_run",
    "terminal_index",
    "terminal_integrity_sha256",
}
SECONDARY_UNIT_FIELDS = {
    "secondary_ordinal",
    "secondary_review_id",
    "question",
    "reference",
    "candidate",
    "secondary_label",
    "secondary_match_type",
    "secondary_notes",
}
MAPPING_UNIT_FIELDS = {
    "secondary_review_id",
    "triple_binding_sha256",
    "selection_reasons",
    "primary_occurrences",
}
MAPPING_OCCURRENCE_FIELDS = {
    "review_id",
    "primary_label",
    "primary_match_type",
    "luna_label",
    "selection_role",
    "condition_ordinal",
    "cluster_index",
    "history_index",
    "variant_index",
    "unit_binding_sha256",
}


class SecondaryPacketError(ValueError):
    """Controlled unblinding or secondary-packet construction failed."""


def _rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).hexdigest()


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


def validate_unblinding_ledger(
    ledger: Mapping[str, Any],
    *,
    primary_units: Sequence[Mapping[str, Any]],
    protocol_integrity_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate the hidden ledger without returning source-bearing text."""

    if (
        ledger.get("schema") != UNBLINDING_SCHEMA
        or ledger.get("schema_version") != 2
        or ledger.get("status") != "sealed-before-primary-labels"
        or ledger.get("local_only") is not True
        or ledger.get("source_bearing") is not False
    ):
        raise SecondaryPacketError("unexpected unblinding-ledger metadata")
    observed_integrity = primary._integrity_sha256(
        ledger,
        name="unblinding ledger",
    )
    if ledger.get("protocol_integrity_sha256") != protocol_integrity_sha256:
        raise SecondaryPacketError("unblinding ledger protocol binding differs")

    units = ledger.get("units")
    if not isinstance(units, list) or len(units) != primary.EXPECTED_UNITS:
        raise SecondaryPacketError("unblinding ledger must contain 38 units")
    expected_review_ids = {str(unit["review_id"]) for unit in primary_units}
    by_review_id: dict[str, dict[str, Any]] = {}
    roles: Counter[str] = Counter()
    for index, unit in enumerate(units, start=1):
        if not isinstance(unit, Mapping) or set(unit) != LEDGER_UNIT_FIELDS:
            raise SecondaryPacketError(
                f"unblinding-ledger unit {index} fields differ"
            )
        review_id = unit.get("review_id")
        if not isinstance(review_id, str) or review_id in by_review_id:
            raise SecondaryPacketError("unblinding review IDs are invalid")
        role = unit.get("selection_role")
        luna_label = unit.get("luna_label")
        expected_label = {
            "luna_flagged_matcher_miss": "leak",
            "luna_negative_control": "no_leak",
        }.get(str(role))
        if expected_label is None or luna_label != expected_label:
            raise SecondaryPacketError("Luna label and selection role differ")
        by_review_id[review_id] = dict(unit)
        roles[str(role)] += 1
    if set(by_review_id) != expected_review_ids:
        raise SecondaryPacketError("unblinding ledger review IDs differ")
    if dict(roles) != EXPECTED_ROLES:
        raise SecondaryPacketError(f"unblinding role counts differ: {dict(roles)}")
    by_review_id["_ledger_integrity"] = {
        "sha256": observed_integrity,
    }
    return by_review_id


def select_secondary_groups(
    *,
    primary_units: Sequence[Mapping[str, Any]],
    ledger_by_review_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select disagreements and abstentions, deduplicated by visible triple."""

    grouped: dict[
        tuple[str, str, str],
        list[tuple[dict[str, Any], dict[str, Any], str]],
    ] = defaultdict(list)
    for unit in primary_units:
        review_id = str(unit["review_id"])
        hidden = ledger_by_review_id.get(review_id)
        if hidden is None:
            raise SecondaryPacketError(f"missing hidden row for {review_id}")
        primary_label = str(unit["primary_label"])
        luna_label = str(hidden["luna_label"])
        if primary_label == "ambiguous":
            reason = "primary_ambiguous"
        elif primary_label != luna_label:
            reason = "primary_luna_disagreement"
        else:
            reason = "not_selected"
        triple = (
            str(unit["question"]),
            str(unit["reference"]),
            str(unit["candidate"]),
        )
        grouped[triple].append((dict(unit), dict(hidden), reason))

    selected: list[dict[str, Any]] = []
    for triple, rows in grouped.items():
        primary_decisions = {
            (row[0]["primary_label"], row[0]["primary_match_type"]) for row in rows
        }
        luna_labels = {row[1]["luna_label"] for row in rows}
        roles = {row[1]["selection_role"] for row in rows}
        reasons = {row[2] for row in rows}
        if len(primary_decisions) != 1:
            raise SecondaryPacketError("duplicate triple has inconsistent primary labels")
        if len(luna_labels) != 1 or len(roles) != 1:
            raise SecondaryPacketError("duplicate triple has inconsistent Luna strata")
        selected_reasons = reasons - {"not_selected"}
        if not selected_reasons:
            continue
        if "not_selected" in reasons:
            raise SecondaryPacketError(
                "duplicate triple is only partially selected for secondary review"
            )

        triple_binding = primary._payload_sha256(
            {
                "question": triple[0],
                "reference": triple[1],
                "candidate": triple[2],
            }
        )
        secondary_id = (
            f"SHV2-{_rank(SECONDARY_ID_NAMESPACE, triple_binding)[:12]}"
        )
        selected.append(
            {
                "secondary_review_id": secondary_id,
                "triple_binding_sha256": triple_binding,
                "question": triple[0],
                "reference": triple[1],
                "candidate": triple[2],
                "selection_reasons": sorted(selected_reasons),
                "rows": rows,
            }
        )
    selected.sort(
        key=lambda group: _rank(
            SECONDARY_ORDER_NAMESPACE,
            str(group["triple_binding_sha256"]),
        )
    )
    if len({group["secondary_review_id"] for group in selected}) != len(selected):
        raise SecondaryPacketError("secondary review ID collision")
    return selected


def build_outputs(
    *,
    packet: Mapping[str, Any],
    ledger: Mapping[str, Any],
    primary_lock: Mapping[str, Any],
    unblinding_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build source-free protocol, blinded packet, and hidden mapping."""

    primary_units = packet["units"]
    ledger_by_review_id = validate_unblinding_ledger(
        ledger,
        primary_units=primary_units,
        protocol_integrity_sha256=str(packet["protocol_integrity_sha256"]),
    )
    ledger_integrity = ledger_by_review_id.pop("_ledger_integrity")["sha256"]
    groups = select_secondary_groups(
        primary_units=primary_units,
        ledger_by_review_id=ledger_by_review_id,
    )

    selected_rows = [row for group in groups for row in group["rows"]]
    reasons = Counter(row[2] for row in selected_rows)
    selected_review_ids = sorted(str(row[0]["review_id"]) for row in selected_rows)
    group_bindings = sorted(str(group["triple_binding_sha256"]) for group in groups)
    protocol_body = {
        "schema": SECONDARY_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "status": "frozen-before-secondary-human-labels",
        "source_free": True,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "primary_lock_integrity_sha256": primary._integrity_sha256(
            primary_lock,
            name="primary lock",
        ),
        "controlled_unblinding": {
            "unblinding_ledger_file_sha256": unblinding_file_sha256,
            "unblinding_ledger_integrity_sha256": ledger_integrity,
        },
        "selection": {
            "primary_luna_disagreement_units": reasons.get(
                "primary_luna_disagreement",
                0,
            ),
            "primary_ambiguous_units": reasons.get("primary_ambiguous", 0),
            "selected_primary_units": len(selected_rows),
            "secondary_distinct_triples": len(groups),
            "duplicate_occurrences_collapsed": len(selected_rows) - len(groups),
            "selected_primary_review_ids_sha256": primary._payload_sha256(
                selected_review_ids
            ),
            "secondary_triple_bindings_sha256": primary._payload_sha256(
                group_bindings
            ),
        },
        "design": {
            "secondary_rater_blinded": True,
            "primary_labels_hidden": True,
            "luna_labels_hidden": True,
            "conditions_hidden": True,
            "selection_roles_hidden": True,
            "duplicate_multiplicities_hidden": True,
            "identical_triples_presented_once": True,
            "third_rater_for_human_disagreement": True,
            "agreement_statistic_prohibited": True,
        },
        "supplemental_luna_ambiguous_outputs": {
            "count": 2,
            "included_in_primary_endpoint": False,
            "included_in_secondary_packet": False,
            "status": "separate-supplement-not-generated",
        },
    }
    protocol = primary._seal(protocol_body)
    protocol_integrity = protocol["integrity"]["sha256"]

    instructions = packet["instructions"]
    secondary_packet = {
        "schema": SECONDARY_PACKET_SCHEMA,
        "schema_version": 1,
        "status": "awaiting-secondary-human-labels",
        "local_only": True,
        "source_bearing": True,
        "primary_labels_hidden": True,
        "luna_labels_hidden": True,
        "condition_identity_hidden": True,
        "selection_strata_hidden": True,
        "duplicate_multiplicity_hidden": True,
        "secondary_protocol_integrity_sha256": protocol_integrity,
        "instructions": {
            "leak": instructions["leak"],
            "no_leak": instructions["no_leak"],
            "ambiguous": instructions["ambiguous"],
            "match_type": instructions["match_type"],
            "independence": (
                "Label each distinct triple independently. Do not seek primary "
                "labels, Luna labels, model conditions, or duplicate counts."
            ),
        },
        "units": [
            {
                "secondary_ordinal": ordinal,
                "secondary_review_id": group["secondary_review_id"],
                "question": group["question"],
                "reference": group["reference"],
                "candidate": group["candidate"],
                "secondary_label": None,
                "secondary_match_type": None,
                "secondary_notes": None,
            }
            for ordinal, group in enumerate(groups, start=1)
        ],
    }

    hidden_mapping = primary._seal(
        {
            "schema": SECONDARY_MAPPING_SCHEMA,
            "schema_version": 1,
            "status": "sealed-before-secondary-labels",
            "local_only": True,
            "source_bearing": False,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "secondary_protocol_integrity_sha256": protocol_integrity,
            "units": [
                {
                    "secondary_review_id": group["secondary_review_id"],
                    "triple_binding_sha256": group["triple_binding_sha256"],
                    "selection_reasons": group["selection_reasons"],
                    "primary_occurrences": [
                        {
                            "review_id": visible["review_id"],
                            "primary_label": visible["primary_label"],
                            "primary_match_type": visible["primary_match_type"],
                            "luna_label": hidden["luna_label"],
                            "selection_role": hidden["selection_role"],
                            "condition_ordinal": hidden["condition_ordinal"],
                            "cluster_index": hidden["cluster_index"],
                            "history_index": hidden["history_index"],
                            "variant_index": hidden["variant_index"],
                            "unit_binding_sha256": hidden["unit_binding_sha256"],
                        }
                        for visible, hidden, _ in group["rows"]
                    ],
                }
                for group in groups
            ],
        }
    )
    return protocol, secondary_packet, hidden_mapping


def validate_generated_outputs(
    *,
    secondary_protocol_path: Path = DEFAULT_SECONDARY_PROTOCOL,
    secondary_packet_path: Path = DEFAULT_SECONDARY_PACKET,
    secondary_mapping_path: Path = DEFAULT_SECONDARY_MAPPING,
) -> dict[str, int]:
    """Validate generated artifacts while returning only source-free counts."""

    protocol = primary._load_mapping(
        secondary_protocol_path,
        name="secondary protocol",
    )
    packet = primary._load_mapping(
        secondary_packet_path,
        name="secondary packet",
    )
    mapping = primary._load_mapping(
        secondary_mapping_path,
        name="secondary hidden mapping",
    )
    protocol_integrity = primary._integrity_sha256(
        protocol,
        name="secondary protocol",
    )
    primary._integrity_sha256(mapping, name="secondary hidden mapping")
    if (
        protocol.get("schema") != SECONDARY_PROTOCOL_SCHEMA
        or protocol.get("schema_version") != 1
        or protocol.get("status") != "frozen-before-secondary-human-labels"
        or protocol.get("source_free") is not True
    ):
        raise SecondaryPacketError("unexpected secondary-protocol metadata")
    if (
        packet.get("schema") != SECONDARY_PACKET_SCHEMA
        or packet.get("schema_version") != 1
        or packet.get("status") != "awaiting-secondary-human-labels"
        or packet.get("local_only") is not True
        or packet.get("source_bearing") is not True
        or packet.get("primary_labels_hidden") is not True
        or packet.get("luna_labels_hidden") is not True
        or packet.get("condition_identity_hidden") is not True
        or packet.get("selection_strata_hidden") is not True
        or packet.get("duplicate_multiplicity_hidden") is not True
        or packet.get("secondary_protocol_integrity_sha256")
        != protocol_integrity
    ):
        raise SecondaryPacketError("unexpected secondary-packet metadata")
    if (
        mapping.get("schema") != SECONDARY_MAPPING_SCHEMA
        or mapping.get("schema_version") != 1
        or mapping.get("status") != "sealed-before-secondary-labels"
        or mapping.get("local_only") is not True
        or mapping.get("source_bearing") is not False
        or mapping.get("contains_source_text") is not False
        or mapping.get("contains_model_generated_text") is not False
        or mapping.get("secondary_protocol_integrity_sha256")
        != protocol_integrity
    ):
        raise SecondaryPacketError("unexpected secondary-mapping metadata")

    expected_modes = (
        (secondary_protocol_path, 0o644),
        (secondary_packet_path, 0o600),
        (secondary_mapping_path, 0o400),
    )
    for path, expected_mode in expected_modes:
        observed_mode = stat.S_IMODE(path.stat().st_mode)
        if observed_mode != expected_mode:
            raise SecondaryPacketError(
                f"unexpected mode for {path}: {observed_mode:04o}"
            )

    packet_units = packet.get("units")
    mapping_units = mapping.get("units")
    if not isinstance(packet_units, list) or not isinstance(mapping_units, list):
        raise SecondaryPacketError("secondary unit arrays are missing")
    packet_by_id: dict[str, Mapping[str, Any]] = {}
    for expected_ordinal, unit in enumerate(packet_units, start=1):
        if not isinstance(unit, Mapping) or set(unit) != SECONDARY_UNIT_FIELDS:
            raise SecondaryPacketError(
                f"secondary packet unit {expected_ordinal} fields differ"
            )
        if unit.get("secondary_ordinal") != expected_ordinal:
            raise SecondaryPacketError("secondary ordinals are not consecutive")
        review_id = unit.get("secondary_review_id")
        if (
            not isinstance(review_id, str)
            or SECONDARY_REVIEW_ID_PATTERN.fullmatch(review_id) is None
            or review_id in packet_by_id
        ):
            raise SecondaryPacketError("secondary review IDs are invalid")
        if any(
            unit.get(field) is not None
            for field in (
                "secondary_label",
                "secondary_match_type",
                "secondary_notes",
            )
        ):
            raise SecondaryPacketError(
                "secondary human-label fields must remain blank before review"
            )
        if any(
            not isinstance(unit.get(field), str)
            for field in ("question", "reference", "candidate")
        ):
            raise SecondaryPacketError("secondary visible review fields must be text")
        packet_by_id[review_id] = unit

    mapping_by_id: dict[str, Mapping[str, Any]] = {}
    selected_primary_review_ids: list[str] = []
    group_bindings: list[str] = []
    reason_counts: Counter[str] = Counter()
    selected_primary_units = 0
    for unit in mapping_units:
        if not isinstance(unit, Mapping) or set(unit) != MAPPING_UNIT_FIELDS:
            raise SecondaryPacketError("secondary mapping unit fields differ")
        review_id = unit.get("secondary_review_id")
        if not isinstance(review_id, str) or review_id in mapping_by_id:
            raise SecondaryPacketError("secondary mapping review IDs are invalid")
        visible = packet_by_id.get(review_id)
        if visible is None:
            raise SecondaryPacketError("secondary packet/mapping IDs differ")
        expected_binding = primary._payload_sha256(
            {
                "question": visible["question"],
                "reference": visible["reference"],
                "candidate": visible["candidate"],
            }
        )
        if unit.get("triple_binding_sha256") != expected_binding:
            raise SecondaryPacketError("secondary visible/hidden triple binding differs")
        reasons = unit.get("selection_reasons")
        if (
            not isinstance(reasons, list)
            or len(reasons) != 1
            or reasons[0]
            not in {"primary_luna_disagreement", "primary_ambiguous"}
        ):
            raise SecondaryPacketError("secondary selection reason is invalid")
        occurrences = unit.get("primary_occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise SecondaryPacketError("secondary mapping occurrences are missing")
        for occurrence in occurrences:
            if (
                not isinstance(occurrence, Mapping)
                or set(occurrence) != MAPPING_OCCURRENCE_FIELDS
            ):
                raise SecondaryPacketError(
                    "secondary mapping occurrence fields differ"
                )
            primary_label = occurrence.get("primary_label")
            luna_label = occurrence.get("luna_label")
            reason = reasons[0]
            if reason == "primary_ambiguous":
                if primary_label != "ambiguous":
                    raise SecondaryPacketError("ambiguous selection reason differs")
            elif (
                primary_label not in {"leak", "no_leak"}
                or luna_label not in {"leak", "no_leak"}
                or primary_label == luna_label
            ):
                raise SecondaryPacketError("disagreement selection reason differs")
            selected_primary_review_ids.append(str(occurrence["review_id"]))
            selected_primary_units += 1
            reason_counts[reason] += 1
        group_bindings.append(expected_binding)
        mapping_by_id[review_id] = unit

    if set(mapping_by_id) != set(packet_by_id):
        raise SecondaryPacketError("secondary packet/mapping unit sets differ")
    selection = protocol.get("selection")
    if not isinstance(selection, Mapping):
        raise SecondaryPacketError("secondary selection summary is missing")
    expected_counts = {
        "primary_luna_disagreement_units": reason_counts.get(
            "primary_luna_disagreement",
            0,
        ),
        "primary_ambiguous_units": reason_counts.get("primary_ambiguous", 0),
        "selected_primary_units": selected_primary_units,
        "secondary_distinct_triples": len(packet_units),
        "duplicate_occurrences_collapsed": selected_primary_units
        - len(packet_units),
    }
    for field, expected in expected_counts.items():
        if selection.get(field) != expected:
            raise SecondaryPacketError(
                f"secondary selection count differs for {field}"
            )
    if selection.get(
        "selected_primary_review_ids_sha256"
    ) != primary._payload_sha256(sorted(selected_primary_review_ids)):
        raise SecondaryPacketError("selected primary review binding differs")
    if selection.get(
        "secondary_triple_bindings_sha256"
    ) != primary._payload_sha256(sorted(group_bindings)):
        raise SecondaryPacketError("secondary triple binding aggregate differs")
    return expected_counts


def generate(
    *,
    packet_path: Path = DEFAULT_PACKET,
    protocol_path: Path = DEFAULT_PROTOCOL,
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
    primary_lock_path: Path = DEFAULT_PRIMARY_LOCK,
    unblinding_path: Path = DEFAULT_UNBLINDING,
    secondary_protocol_path: Path = DEFAULT_SECONDARY_PROTOCOL,
    secondary_packet_path: Path = DEFAULT_SECONDARY_PACKET,
    secondary_mapping_path: Path = DEFAULT_SECONDARY_MAPPING,
) -> dict[str, int]:
    """Verify the frozen primary state, then perform controlled unblinding."""

    expected_lock = primary.build_lock(
        packet_path=packet_path,
        protocol_path=protocol_path,
        analysis_plan_path=analysis_plan_path,
    )
    observed_lock = primary._load_mapping(primary_lock_path, name="primary lock")
    if observed_lock != expected_lock:
        raise SecondaryPacketError("primary packet or analysis-plan lock is stale")

    packet = primary._load_mapping(packet_path, name="primary packet")
    ledger = primary._load_mapping(unblinding_path, name="unblinding ledger")
    protocol, secondary_packet, hidden_mapping = build_outputs(
        packet=packet,
        ledger=ledger,
        primary_lock=observed_lock,
        unblinding_file_sha256=primary._file_sha256(unblinding_path),
    )
    outputs = (
        secondary_protocol_path,
        secondary_packet_path,
        secondary_mapping_path,
    )
    occupied = [str(path) for path in outputs if path.exists() or path.is_symlink()]
    if occupied:
        raise SecondaryPacketError(
            f"secondary output paths already exist: {occupied}"
        )
    created: list[Path] = []
    try:
        _write_new(secondary_protocol_path, protocol, mode=0o644)
        created.append(secondary_protocol_path)
        _write_new(secondary_packet_path, secondary_packet, mode=0o600)
        created.append(secondary_packet_path)
        _write_new(secondary_mapping_path, hidden_mapping, mode=0o400)
        created.append(secondary_mapping_path)
        validated = validate_generated_outputs(
            secondary_protocol_path=secondary_protocol_path,
            secondary_packet_path=secondary_packet_path,
            secondary_mapping_path=secondary_mapping_path,
        )
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=DEFAULT_ANALYSIS_PLAN,
    )
    parser.add_argument("--primary-lock", type=Path, default=DEFAULT_PRIMARY_LOCK)
    parser.add_argument("--unblinding", type=Path, default=DEFAULT_UNBLINDING)
    parser.add_argument(
        "--secondary-protocol",
        type=Path,
        default=DEFAULT_SECONDARY_PROTOCOL,
    )
    parser.add_argument(
        "--secondary-packet",
        type=Path,
        default=DEFAULT_SECONDARY_PACKET,
    )
    parser.add_argument(
        "--secondary-mapping",
        type=Path,
        default=DEFAULT_SECONDARY_MAPPING,
    )
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="validate generated outputs without reading the unblinding ledger",
    )
    args = parser.parse_args()
    if args.check_existing:
        summary = validate_generated_outputs(
            secondary_protocol_path=args.secondary_protocol,
            secondary_packet_path=args.secondary_packet,
            secondary_mapping_path=args.secondary_mapping,
        )
    else:
        summary = generate(
            packet_path=args.packet,
            protocol_path=args.protocol,
            analysis_plan_path=args.analysis_plan,
            primary_lock_path=args.primary_lock,
            unblinding_path=args.unblinding,
            secondary_protocol_path=args.secondary_protocol,
            secondary_packet_path=args.secondary_packet,
            secondary_mapping_path=args.secondary_mapping,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

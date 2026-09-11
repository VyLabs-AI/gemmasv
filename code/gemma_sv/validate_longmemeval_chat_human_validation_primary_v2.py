"""Validate and seal the completed HV2 primary packet without unblinding.

This module must not read the unblinding ledger. It validates only the
source-bearing primary packet and the source-free frozen protocol, then emits a
source-free lock that binds the packet bytes and pre-unblinding analysis plan.
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
from typing import Any, Mapping


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_PACKET = (
    REPOSITORY
    / "outputs/gemma_sv_rag/longmemeval_chat_human_validation_v2/"
    "primary_packet.json"
)
DEFAULT_PROTOCOL = (
    BENCHMARKS / "longmemeval_chat_human_validation_protocol_v2.json"
)
DEFAULT_ANALYSIS_PLAN = (
    BENCHMARKS / "LONGMEMEVAL_HUMAN_VALIDATION_ANALYSIS_PLAN_V1.md"
)
DEFAULT_LOCK = (
    BENCHMARKS / "longmemeval_chat_human_validation_primary_lock_v1.json"
)

EXPECTED_UNITS = 38
PACKET_SCHEMA = "gemma-sv-longmemeval-human-validation-primary-packet-v2"
PROTOCOL_SCHEMA = "gemma-sv-longmemeval-human-validation-protocol-v2"
LOCK_SCHEMA = "gemma-sv-longmemeval-human-validation-primary-lock-v1"
REVIEW_ID_PATTERN = re.compile(r"^HV2-[0-9a-f]{12}$")

PACKET_FIELDS = {
    "condition_identity_hidden",
    "instructions",
    "local_only",
    "machine_labels_hidden",
    "protocol_integrity_sha256",
    "schema",
    "schema_version",
    "selection_strata_hidden",
    "source_bearing",
    "status",
    "units",
}
UNIT_FIELDS = {
    "review_ordinal",
    "review_id",
    "question",
    "reference",
    "candidate",
    "primary_label",
    "primary_match_type",
    "primary_notes",
}
LABEL_TO_MATCH_TYPES = {
    "leak": {
        "normalization_failure",
        "alias_or_morphology",
        "genuine_paraphrase",
    },
    "no_leak": {"no_disclosure"},
    "ambiguous": {"ambiguous"},
}
MATCH_TYPES = (
    "normalization_failure",
    "alias_or_morphology",
    "genuine_paraphrase",
    "no_disclosure",
    "ambiguous",
)


class PrimaryValidationError(ValueError):
    """The primary packet or its source-free bindings are invalid."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PrimaryValidationError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrimaryValidationError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PrimaryValidationError(f"{name} must contain an object")
    return value


def _integrity_sha256(value: Mapping[str, Any], *, name: str) -> str:
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping):
        raise PrimaryValidationError(f"{name} lacks an integrity object")
    expected = integrity.get("sha256")
    unsigned = dict(value)
    unsigned.pop("integrity", None)
    observed = _payload_sha256(unsigned)
    if expected != observed:
        raise PrimaryValidationError(f"{name} integrity mismatch")
    return observed


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": _payload_sha256(value),
    }
    return sealed


def _repository_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError as exc:
        raise PrimaryValidationError(f"path is outside repository: {path}") from exc


def _count_with_zeros(
    values: Counter[str],
    categories: tuple[str, ...],
) -> dict[str, int]:
    return {category: values.get(category, 0) for category in categories}


def validate_primary_packet(
    packet_path: Path = DEFAULT_PACKET,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Return a source-free summary or fail closed on an invalid packet."""

    packet = _load_mapping(packet_path, name="primary packet")
    protocol = _load_mapping(protocol_path, name="human-validation protocol")
    protocol_integrity = _integrity_sha256(
        protocol,
        name="human-validation protocol",
    )
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 2:
        raise PrimaryValidationError("unexpected human-validation protocol schema")
    if set(packet) != PACKET_FIELDS:
        raise PrimaryValidationError("primary packet top-level fields differ")
    if packet.get("schema") != PACKET_SCHEMA or packet.get("schema_version") != 2:
        raise PrimaryValidationError("unexpected primary packet schema")
    expected_flags = {
        "local_only": True,
        "source_bearing": True,
        "machine_labels_hidden": True,
        "condition_identity_hidden": True,
        "selection_strata_hidden": True,
        "status": "awaiting-primary-human-labels",
    }
    for field, expected in expected_flags.items():
        if packet.get(field) != expected:
            raise PrimaryValidationError(
                f"primary packet {field} must remain {expected!r}"
            )
    if packet.get("protocol_integrity_sha256") != protocol_integrity:
        raise PrimaryValidationError("primary packet protocol binding differs")
    instructions = packet.get("instructions")
    if not isinstance(instructions, Mapping):
        raise PrimaryValidationError("primary packet instructions are missing")

    units = packet.get("units")
    if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
        raise PrimaryValidationError(
            f"primary packet must contain exactly {EXPECTED_UNITS} units"
        )

    labels: Counter[str] = Counter()
    match_types: Counter[str] = Counter()
    review_ids: set[str] = set()
    triple_decisions: dict[
        tuple[str, str, str],
        set[tuple[str, str]],
    ] = defaultdict(set)
    triple_sizes: Counter[tuple[str, str, str]] = Counter()
    notes_completed = 0

    for expected_ordinal, unit in enumerate(units, start=1):
        if not isinstance(unit, Mapping) or set(unit) != UNIT_FIELDS:
            raise PrimaryValidationError(
                f"unit {expected_ordinal} fields differ from the frozen schema"
            )
        if unit.get("review_ordinal") != expected_ordinal:
            raise PrimaryValidationError("review ordinals are not consecutive")
        review_id = unit.get("review_id")
        if not isinstance(review_id, str) or REVIEW_ID_PATTERN.fullmatch(review_id) is None:
            raise PrimaryValidationError(
                f"unit {expected_ordinal} has an invalid review ID"
            )
        if review_id in review_ids:
            raise PrimaryValidationError(f"duplicate review ID: {review_id}")
        review_ids.add(review_id)

        text_fields = ("question", "reference", "candidate")
        if any(not isinstance(unit.get(field), str) for field in text_fields):
            raise PrimaryValidationError(
                f"unit {expected_ordinal} contains a non-text review field"
            )
        label = unit.get("primary_label")
        match_type = unit.get("primary_match_type")
        if label not in LABEL_TO_MATCH_TYPES:
            raise PrimaryValidationError(
                f"unit {expected_ordinal} has an incomplete or invalid primary label"
            )
        if match_type not in LABEL_TO_MATCH_TYPES[label]:
            raise PrimaryValidationError(
                f"unit {expected_ordinal} label/match-type pairing is invalid"
            )
        notes = unit.get("primary_notes")
        if notes is not None and not isinstance(notes, str):
            raise PrimaryValidationError(
                f"unit {expected_ordinal} primary notes must be text or null"
            )
        if isinstance(notes, str) and notes.strip():
            notes_completed += 1

        triple = (
            str(unit["question"]),
            str(unit["reference"]),
            str(unit["candidate"]),
        )
        decision = (str(label), str(match_type))
        triple_decisions[triple].add(decision)
        triple_sizes[triple] += 1
        labels[str(label)] += 1
        match_types[str(match_type)] += 1

    inconsistent = [
        triple for triple, decisions in triple_decisions.items() if len(decisions) != 1
    ]
    if inconsistent:
        raise PrimaryValidationError(
            "identical question/reference/candidate triples have inconsistent labels"
        )

    repeated_sizes = [size for size in triple_sizes.values() if size > 1]
    mode = stat.S_IMODE(packet_path.stat().st_mode)
    return {
        "packet_file_sha256": _file_sha256(packet_path),
        "protocol_integrity_sha256": protocol_integrity,
        "unit_count": len(units),
        "unique_triple_count": len(triple_sizes),
        "duplicate_excess_count": len(units) - len(triple_sizes),
        "repeated_triple_group_count": len(repeated_sizes),
        "units_in_repeated_triples": sum(repeated_sizes),
        "primary_label_counts": _count_with_zeros(
            labels,
            ("leak", "no_leak", "ambiguous"),
        ),
        "primary_match_type_counts": _count_with_zeros(
            match_types,
            MATCH_TYPES,
        ),
        "units_with_primary_notes": notes_completed,
        "packet_mode": f"{mode:04o}",
        "packet_read_only": not bool(mode & 0o222),
    }


def build_lock(
    *,
    packet_path: Path = DEFAULT_PACKET,
    protocol_path: Path = DEFAULT_PROTOCOL,
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
) -> dict[str, Any]:
    """Build the source-free pre-unblinding lock payload."""

    if analysis_plan_path.is_symlink() or not analysis_plan_path.is_file():
        raise PrimaryValidationError("analysis plan must be a regular file")
    summary = validate_primary_packet(packet_path, protocol_path)
    body = {
        "schema": LOCK_SCHEMA,
        "schema_version": 1,
        "status": "primary-labels-frozen-before-controlled-unblinding",
        "source_free": True,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "packet": {
            "path": _repository_path(packet_path),
            "local_only": True,
            **summary,
        },
        "analysis_plan": {
            "path": _repository_path(analysis_plan_path),
            "file_sha256": _file_sha256(analysis_plan_path),
        },
        "rater_attestation": {
            "basis": "reported by the independent primary rater",
            "all_units_labeled": True,
            "condition_identity_hidden": True,
            "machine_labels_hidden": True,
            "selection_strata_hidden": True,
            "unblinding_ledger_opened_by_primary": False,
            "source_conversations_inspected": False,
            "external_service_used": False,
        },
    }
    return _seal(body)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=DEFAULT_ANALYSIS_PLAN,
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--write-lock", action="store_true")
    actions.add_argument("--check-lock", action="store_true")
    args = parser.parse_args()

    if args.write_lock or args.check_lock:
        expected = build_lock(
            packet_path=args.packet,
            protocol_path=args.protocol,
            analysis_plan_path=args.analysis_plan,
        )
        if args.write_lock:
            _write_new(args.lock, expected)
        else:
            observed = _load_mapping(args.lock, name="primary lock")
            if observed != expected:
                raise SystemExit(f"stale primary lock: {args.lock}")
        summary = expected["packet"]
    else:
        summary = validate_primary_packet(args.packet, args.protocol)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

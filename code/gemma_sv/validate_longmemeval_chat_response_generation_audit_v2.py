"""Read-only validation for the sealed LongMemEval response audit v2."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import longmemeval_chat_response_generation_audit_v2 as audit


VALIDATION_SCHEMA = (
    "gemma-sv-longmemeval-chat-response-generation-audit-validation-v1"
)
VALIDATION_SCHEMA_VERSION = 1
RECOVERY_SCHEMA = "gemma-sv-longmemeval-response-integrity-recovery-v1"
RECOVERY_FILENAME = "integrity_recovery_v1.json"
DEFAULT_ROOT = audit.DEFAULT_OUTPUT_ROOT

EXPECTED_STARTED_ATTEMPTS = 98
EXPECTED_TERMINAL_ATTEMPTS = 96
EXPECTED_INCOMPLETE_ATTEMPTS = 2
EXPECTED_ORDINARY_SEALS = 197
EXPECTED_COMPATIBILITY_SEALS = 96

_RECOVERY_REASON = (
    "The original in-memory seal sorted integer kpar_by_layer keys "
    "numerically; JSON reload converted those object keys to strings. "
    "Restoring only that known key type reproduces every recorded payload "
    "hash."
)
_KPAR_PATH = (
    "result.conditions.exact_decrement_or_refit_policy."
    "fixed_c_diagnostics.objective.kpar_by_layer"
)
_DECIMAL_KEY_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_ATTEMPT_FILE_RE = re.compile(
    r"(?P<attempt>[0-9]{3})-(?P<phase>started|terminal)\.json\Z"
)
_ATTEMPT_PATH_RE = re.compile(
    r"attempts/(?P<slot>[0-9]{3})/"
    r"(?P<attempt>[0-9]{3})-(?P<phase>started|terminal)\.json\Z"
)
_STABLE_IDENTIFIER_RE = re.compile(
    r"longmemeval(?:-chat)?-(?:history|cluster|source)(?:-v[0-9]+)?-",
    flags=re.IGNORECASE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_DECLARATION_KEYS = {
    "contains_source_text",
    "contains_source_identifiers",
    "contains_model_generated_text",
    "contains_token_arrays",
    "contains_raw_ledgers",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "answer",
    "generated_token_ids",
    "history_id",
    "messages",
    "prompt",
    "question",
    "raw_ledger",
    "record_id",
    "response_text",
    "sessions",
    "source_id",
    "source_text",
    "token_ids",
    "turns",
)
_INPUT_EVIDENCE = (
    (
        "gemma_sv/benchmarks/"
        "longmemeval_chat_response_generation_authorization_v2.json",
        "authorization-lock",
    ),
    (
        "gemma_sv/benchmarks/longmemeval_chat_cohort_v3.json",
        "cohort",
    ),
    (
        "gemma_sv/benchmarks/"
        "longmemeval_chat_cluster_analysis_lock_v3.json",
        "cluster-analysis-lock",
    ),
    (
        "gemma_sv/benchmarks/longmemeval_chat_cohort_census_v3.json",
        "census",
    ),
)


class AuditValidationError(audit.AuditV2Error):
    """The sealed audit or its source-free validation record is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditValidationError(message)


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    observed = set(value)
    _require(
        observed == expected,
        f"{name} fields differ; "
        f"missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{name} must be a canonical SHA-256",
    )
    return value


def _payload_integrity(
    value: Mapping[str, Any],
    *,
    name: str,
    allow_scope: bool = False,
) -> str:
    raw_integrity = value.get("integrity")
    expected_keys = {"algorithm", "sha256"}
    if (
        allow_scope
        and isinstance(raw_integrity, Mapping)
        and "scope" in raw_integrity
    ):
        expected_keys.add("scope")
    integrity = _require_exact_keys(
        raw_integrity,
        expected_keys,
        name=f"{name} integrity",
    )
    _require(
        integrity.get("algorithm") == "sha256",
        f"{name} integrity algorithm differs",
    )
    return _require_sha256(integrity.get("sha256"), name=f"{name} payload")


def _strict_root_json(path: Path, *, name: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{name} must not be a symlink")
    _require(path.is_file(), f"{name} must be a regular file")
    audit._validate_permissions(path, directory=False)
    return audit._load_json(path, name=name)


def _safe_relative_path(value: Any, *, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} path is empty")
    _require("\\" not in value, f"{name} path is not POSIX-relative")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and value != "."
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{name} path is not safely relative",
    )
    return value


def _descriptor(
    *,
    path: Path,
    relative_path: str,
    payload_sha256: str,
    validation: str,
) -> dict[str, str]:
    return {
        "path": _safe_relative_path(relative_path, name="evidence"),
        "file_sha256": audit._file_sha256(path),
        "payload_sha256": _require_sha256(
            payload_sha256,
            name="evidence payload",
        ),
        "validation": validation,
    }


def _root_descriptor(
    path: Path,
    root: Path,
    value: Mapping[str, Any],
    *,
    name: str,
    validation: str = "ordinary",
    payload_sha256: str | None = None,
) -> dict[str, str]:
    payload = (
        _payload_integrity(value, name=name)
        if payload_sha256 is None
        else _require_sha256(payload_sha256, name=f"{name} payload")
    )
    return _descriptor(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        payload_sha256=payload,
        validation=validation,
    )


def _validate_root_layout(root: Path) -> None:
    _require(not root.is_symlink(), "audit root must not be a symlink")
    _require(root.is_dir(), "audit root must be a directory")
    audit._validate_permissions(root, directory=True)
    required = {
        "attempts",
        "final.json",
        RECOVERY_FILENAME,
        "records",
        "run.json",
    }
    allowed = required | {"heartbeat.json"}
    entries = {entry.name: entry for entry in root.iterdir()}
    _require(
        required <= set(entries),
        f"audit root is missing entries: {sorted(required - set(entries))}",
    )
    _require(
        set(entries) <= allowed,
        f"audit root contains extra entries: {sorted(set(entries) - allowed)}",
    )
    for name in ("attempts", "records"):
        path = entries[name]
        _require(not path.is_symlink(), f"{name} must not be a symlink")
        _require(path.is_dir(), f"{name} must be a directory")
        audit._validate_permissions(path, directory=True)
    for name in ("run.json", "final.json", RECOVERY_FILENAME):
        path = entries[name]
        _require(not path.is_symlink(), f"{name} must not be a symlink")
        _require(path.is_file(), f"{name} must be a regular file")
    heartbeat = entries.get("heartbeat.json")
    if heartbeat is not None:
        _require(
            not heartbeat.is_symlink() and heartbeat.is_file(),
            "heartbeat must be a regular non-symlink when present",
        )
        audit._validate_permissions(heartbeat, directory=False)


def _validate_locked_inputs() -> tuple[dict[str, Any], list[dict[str, str]]]:
    cohort, cluster_lock, census, lock = audit.load_authorization_lock(
        require_committed=False
    )
    values = (
        (
            audit.DEFAULT_AUTHORIZATION_LOCK,
            lock,
            _payload_integrity(lock, name="authorization lock"),
        ),
        (
            audit.DEFAULT_COHORT,
            cohort,
            _payload_integrity(cohort, name="cohort", allow_scope=True),
        ),
        (
            audit.DEFAULT_CLUSTER_LOCK,
            cluster_lock,
            _require_sha256(
                cluster_lock.get("lock_sha256"),
                name="cluster analysis lock payload",
            ),
        ),
        (
            audit.DEFAULT_CENSUS,
            census,
            _payload_integrity(census, name="census"),
        ),
    )
    evidence: list[dict[str, str]] = []
    for (expected_path, validation), (path, value, payload) in zip(
        _INPUT_EVIDENCE,
        values,
    ):
        _require(
            not path.is_symlink() and path.is_file(),
            f"locked input {expected_path} must be a regular non-symlink",
        )
        relative = path.relative_to(audit.WORKSPACE).as_posix()
        _require(relative == expected_path, "locked input path differs")
        evidence.append(
            _descriptor(
                path=path,
                relative_path=relative,
                payload_sha256=payload,
                validation=validation,
            )
        )
    return lock, evidence


def _legacy_integer_key_body(
    shard: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return the one-path compatibility body and its recorded payload hash."""

    body = copy.deepcopy(dict(shard))
    integrity = _require_exact_keys(
        body.pop("integrity", None),
        {"algorithm", "sha256"},
        name="record shard integrity",
    )
    _require(
        integrity.get("algorithm") == "sha256",
        "record shard integrity algorithm differs",
    )
    recorded = _require_sha256(
        integrity.get("sha256"),
        name="record shard recorded payload",
    )
    try:
        objective = body["result"]["conditions"][
            "exact_decrement_or_refit_policy"
        ]["fixed_c_diagnostics"]["objective"]
        kpars = objective["kpar_by_layer"]
    except (KeyError, TypeError) as exc:
        raise AuditValidationError(
            f"legacy compatibility path {_KPAR_PATH} is missing"
        ) from exc
    _require(
        isinstance(kpars, Mapping) and bool(kpars),
        "legacy kpar_by_layer must be a nonempty object",
    )
    converted: dict[int, Any] = {}
    original_count = len(kpars)
    for key, item in kpars.items():
        _require(
            type(key) is str and _DECIMAL_KEY_RE.fullmatch(key) is not None,
            "legacy kpar_by_layer key is not a canonical decimal string",
        )
        integer_key = int(key, 10)
        _require(
            str(integer_key) == key,
            "legacy kpar_by_layer key conversion is not lossless",
        )
        _require(
            integer_key not in converted,
            "legacy kpar_by_layer integer conversion is not unique",
        )
        converted[integer_key] = item
    _require(
        len(converted) == original_count,
        "legacy kpar_by_layer key count changed during conversion",
    )
    objective["kpar_by_layer"] = converted
    return body, recorded


def validate_record_shard_compatibility_seal(
    shard: Mapping[str, Any],
) -> str:
    """Validate only the historical integer-key seal defect in one shard."""

    body, recorded = _legacy_integer_key_body(shard)
    _require(
        audit._payload_sha256(body) == recorded,
        "record shard compatibility payload SHA-256 differs",
    )
    return recorded


def _validate_shards(
    root: Path,
    lock: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, str]]]:
    directory = root / "records"
    expected_names = {
        f"{slot:03d}.json" for slot in range(audit.EXPECTED_HISTORIES)
    }
    entries = {path.name: path for path in directory.iterdir()}
    _require(
        set(entries) == expected_names,
        "records must contain exactly records/000.json through "
        "records/095.json",
    )
    expected_ids = lock["cohort"]["ordered_history_ids"]
    shards: dict[int, dict[str, Any]] = {}
    evidence: list[dict[str, str]] = []
    observed_ids: set[str] = set()
    for slot in range(audit.EXPECTED_HISTORIES):
        path = entries[f"{slot:03d}.json"]
        shard = _strict_root_json(path, name=f"record shard {slot}")
        recorded_payload = validate_record_shard_compatibility_seal(shard)
        audit._require_exact_keys(
            shard,
            {
                "schema",
                "schema_version",
                "slot",
                "record_id",
                "terminal",
                "authorization_lock_integrity_sha256",
                "result",
                "integrity",
            },
            name="record shard",
        )
        record_id = expected_ids[slot]
        _require(
            shard.get("schema") == audit.SHARD_SCHEMA
            and shard.get("schema_version") == audit.SCHEMA_VERSION
            and type(shard.get("slot")) is int
            and shard.get("slot") == slot
            and shard.get("record_id") == record_id
            and shard.get("terminal") is True
            and shard.get("authorization_lock_integrity_sha256")
            == lock["integrity"]["sha256"]
            and isinstance(shard.get("result"), Mapping)
            and shard["result"].get("record_id") == record_id,
            f"record shard {slot} binding differs",
        )
        audit._validate_record_result(
            shard["result"],
            record_id=record_id,
            lock=lock,
        )
        _require(
            record_id not in observed_ids,
            "duplicate record ID appears in record shards",
        )
        observed_ids.add(record_id)
        shards[slot] = shard
        evidence.append(
            _root_descriptor(
                path,
                root,
                shard,
                name=f"record shard {slot}",
                validation="compatibility",
                payload_sha256=recorded_payload,
            )
        )
    return shards, evidence


def _validate_attempt_ledgers(
    root: Path,
    lock: Mapping[str, Any],
    shards: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    attempts_root = root / "attempts"
    expected_directories = {
        f"{slot:03d}" for slot in range(audit.EXPECTED_HISTORIES)
    }
    directories = {path.name: path for path in attempts_root.iterdir()}
    _require(
        set(directories) == expected_directories,
        "attempts must contain exactly anonymous slots 000 through 095",
    )
    history_ids = lock["cohort"]["ordered_history_ids"]
    evidence: list[dict[str, str]] = []
    started_by_slot: dict[int, set[int]] = {}
    terminal_by_slot: dict[int, set[int]] = {}
    for slot in range(audit.EXPECTED_HISTORIES):
        directory = directories[f"{slot:03d}"]
        _require(
            not directory.is_symlink() and directory.is_dir(),
            f"attempt slot {slot} must be a regular directory",
        )
        audit._validate_permissions(directory, directory=True)
        files = sorted(directory.iterdir(), key=lambda path: path.name)
        _require(bool(files), f"attempt slot {slot} is empty")
        starts: set[int] = set()
        terminals: set[int] = set()
        seen_phases: set[tuple[int, str]] = set()
        for path in files:
            _require(
                not path.is_symlink() and path.is_file(),
                f"attempt slot {slot} contains a non-file entry",
            )
            match = _ATTEMPT_FILE_RE.fullmatch(path.name)
            _require(match is not None, "attempt ledger filename differs")
            attempt = int(match.group("attempt"))
            phase = match.group("phase")
            _require(attempt >= 1, "attempt number must be positive")
            _require(
                (attempt, phase) not in seen_phases,
                "duplicate attempt ledger phase",
            )
            seen_phases.add((attempt, phase))
            value = _strict_root_json(path, name="attempt ledger")
            audit._validate_seal(value, name="attempt ledger")
            expected_keys = {
                "schema",
                "schema_version",
                "phase",
                "slot",
                "attempt",
                "record_id",
                "authorization_lock_integrity_sha256",
                "integrity",
            }
            if phase == "terminal":
                expected_keys.update(
                    {
                        "record_shard_integrity_sha256",
                        "record_status",
                        "retry_allowed",
                    }
                )
            audit._require_exact_keys(
                value,
                expected_keys,
                name="attempt ledger",
            )
            _require(
                value.get("schema") == audit.ATTEMPT_SCHEMA
                and value.get("schema_version") == audit.SCHEMA_VERSION
                and value.get("phase") == phase
                and type(value.get("slot")) is int
                and value.get("slot") == slot
                and type(value.get("attempt")) is int
                and value.get("attempt") == attempt
                and value.get("record_id") == history_ids[slot]
                and value.get("authorization_lock_integrity_sha256")
                == lock["integrity"]["sha256"],
                "attempt ledger binding differs",
            )
            if phase == "started":
                starts.add(attempt)
            else:
                terminals.add(attempt)
                _require(
                    value.get("record_shard_integrity_sha256")
                    == shards[slot]["integrity"]["sha256"]
                    and value.get("record_status")
                    == shards[slot]["result"]["status"]
                    and value.get("retry_allowed") is False,
                    "terminal attempt differs from record shard",
                )
            evidence.append(
                _root_descriptor(
                    path,
                    root,
                    value,
                    name="attempt ledger",
                )
            )
        _require(
            starts == set(range(1, max(starts, default=0) + 1)),
            "started attempt numbers must be contiguous from one",
        )
        _require(
            terminals <= starts,
            "terminal attempt has no matching started ledger",
        )
        _require(
            len(terminals) == 1,
            "each terminal shard must have exactly one terminal attempt",
        )
        started_by_slot[slot] = starts
        terminal_by_slot[slot] = terminals
    started = sum(len(items) for items in started_by_slot.values())
    terminal = sum(len(items) for items in terminal_by_slot.values())
    incomplete = {
        slot: len(started_by_slot[slot] - terminal_by_slot[slot])
        for slot in range(audit.EXPECTED_HISTORIES)
        if started_by_slot[slot] - terminal_by_slot[slot]
    }
    _require(
        started == EXPECTED_STARTED_ATTEMPTS
        and terminal == EXPECTED_TERMINAL_ATTEMPTS
        and sum(incomplete.values()) == EXPECTED_INCOMPLETE_ATTEMPTS
        and incomplete == {95: 2},
        "attempt accounting differs from the sealed audit disclosure",
    )
    return evidence, {
        "started": started,
        "terminal": terminal,
        "incomplete_started": sum(incomplete.values()),
        "incomplete_by_anonymous_slot": [
            {"slot": slot, "count": count}
            for slot, count in sorted(incomplete.items())
        ],
    }


def _validate_final(
    root: Path,
    lock: Mapping[str, Any],
    shards: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    path = root / "final.json"
    final = _strict_root_json(path, name="final")
    audit.validate_final(final, lock=lock)
    _require(
        final.get("status") == "completed"
        and final.get("summary", {}).get("history_slots")
        == audit.EXPECTED_HISTORIES
        and final.get("summary", {}).get("generation_calls_started")
        == audit.EXPECTED_GENERATION_CALLS,
        "final expected completed counts differ",
    )
    _require(
        all(
            final["records"][slot] == shards[slot]["result"]
            for slot in range(audit.EXPECTED_HISTORIES)
        ),
        "final record results differ from record shards",
    )
    return final, _root_descriptor(path, root, final, name="final")


def _validate_recovery(
    root: Path,
    lock: Mapping[str, Any],
    shards: Mapping[int, Mapping[str, Any]],
    shard_evidence: Sequence[Mapping[str, str]],
    final: Mapping[str, Any],
    final_evidence: Mapping[str, str],
) -> dict[str, str]:
    path = root / RECOVERY_FILENAME
    recovery = _strict_root_json(path, name="integrity recovery")
    audit._validate_seal(recovery, name="integrity recovery")
    audit._require_exact_keys(
        recovery,
        {
            "schema",
            "status",
            "source_bearing",
            "reason",
            "authorization_lock_integrity_sha256",
            "record_shards",
            "record_shard_count",
            "all_recorded_payload_hashes_reproduced",
            "final",
            "integrity",
        },
        name="integrity recovery",
    )
    _require(
        recovery.get("schema") == RECOVERY_SCHEMA
        and recovery.get("status")
        == "validated-without-evidence-file-modification"
        and recovery.get("source_bearing") is False
        and recovery.get("reason") == _RECOVERY_REASON
        and recovery.get("authorization_lock_integrity_sha256")
        == lock["integrity"]["sha256"]
        and recovery.get("record_shard_count")
        == audit.EXPECTED_HISTORIES
        and recovery.get("all_recorded_payload_hashes_reproduced") is True,
        "integrity recovery contract differs",
    )
    rows = recovery.get("record_shards")
    _require(
        isinstance(rows, list) and len(rows) == audit.EXPECTED_HISTORIES,
        "integrity recovery shard list differs",
    )
    for slot, (row, evidence) in enumerate(zip(rows, shard_evidence)):
        audit._require_exact_keys(
            row,
            {
                "path",
                "file_sha256",
                "recorded_payload_sha256",
                "legacy_integer_key_canonicalization_valid",
                "file_modified",
            },
            name="integrity recovery shard",
        )
        _require(
            row.get("path") == f"records/{slot:03d}.json"
            and row.get("file_sha256") == evidence["file_sha256"]
            and row.get("recorded_payload_sha256")
            == shards[slot]["integrity"]["sha256"]
            and row.get("recorded_payload_sha256")
            == evidence["payload_sha256"]
            and row.get("legacy_integer_key_canonicalization_valid") is True
            and row.get("file_modified") is False,
            f"integrity recovery shard {slot} binding differs",
        )
    final_row = _require_exact_keys(
        recovery.get("final"),
        {
            "path",
            "file_sha256",
            "recorded_payload_sha256",
            "legacy_integer_key_canonicalization_valid",
            "file_modified",
        },
        name="integrity recovery final",
    )
    _require(
        final_row.get("path") == "final.json"
        and final_row.get("file_sha256") == final_evidence["file_sha256"]
        and final_row.get("recorded_payload_sha256")
        == final["integrity"]["sha256"]
        and final_row.get("recorded_payload_sha256")
        == final_evidence["payload_sha256"]
        and final_row.get("legacy_integer_key_canonicalization_valid") is True
        and final_row.get("file_modified") is False,
        "integrity recovery final binding differs",
    )
    return _root_descriptor(path, root, recovery, name="integrity recovery")


def _walk_validation(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_validation(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk_validation(child)


def assert_source_free_validation(value: Mapping[str, Any]) -> None:
    """Reject identifiers, source/model text, token arrays, and raw ledgers."""

    _require(isinstance(value, Mapping), "validation record must be an object")
    for key, child in _walk_validation(value):
        if key is not None:
            folded = key.casefold()
            if folded in _DECLARATION_KEYS:
                _require(
                    child is False,
                    f"source-free declaration {key} must be false",
                )
            elif folded.endswith(("_id", "_ids")) or any(
                fragment in folded for fragment in _FORBIDDEN_KEY_FRAGMENTS
            ):
                raise AuditValidationError(
                    f"validation record contains prohibited key {key!r}"
                )
        if isinstance(child, str):
            _require(
                _STABLE_IDENTIFIER_RE.search(child) is None,
                "validation record contains a source/history/cluster identifier",
            )
            _require(
                not any(
                    marker in child
                    for marker in (
                        "<start_of_turn>",
                        "<end_of_turn>",
                        "<bos>",
                        "<eos>",
                    )
                ),
                "validation record contains chat/source text",
            )
        if (
            isinstance(child, list)
            and bool(child)
            and all(type(item) is int for item in child)
        ):
            raise AuditValidationError(
                "validation record contains a possible token array"
            )


def _validate_evidence_descriptor(
    value: Any,
    *,
    name: str,
    allowed_validations: set[str],
) -> Mapping[str, Any]:
    row = _require_exact_keys(
        value,
        {"path", "file_sha256", "payload_sha256", "validation"},
        name=name,
    )
    _safe_relative_path(row.get("path"), name=name)
    _require_sha256(row.get("file_sha256"), name=f"{name} file")
    _require_sha256(row.get("payload_sha256"), name=f"{name} payload")
    _require(
        row.get("validation") in allowed_validations,
        f"{name} validation mode differs",
    )
    return row


def _validate_record_evidence(
    evidence: Mapping[str, Any],
    attempts: Mapping[str, Any],
    seals: Mapping[str, Any],
) -> None:
    _require_exact_keys(
        evidence,
        {
            "locked_inputs",
            "audit_root",
            "locked_input_count",
            "audit_root_file_count",
        },
        name="evidence",
    )
    locked = evidence.get("locked_inputs")
    _require(
        isinstance(locked, list) and len(locked) == len(_INPUT_EVIDENCE),
        "locked input evidence count differs",
    )
    for row, (path, validation) in zip(locked, _INPUT_EVIDENCE):
        checked = _validate_evidence_descriptor(
            row,
            name="locked input evidence",
            allowed_validations={validation},
        )
        _require(
            checked["path"] == path
            and checked["validation"] == validation,
            "locked input evidence binding differs",
        )
    _require(
        evidence.get("locked_input_count") == len(locked),
        "locked input evidence summary differs",
    )

    root_rows = evidence.get("audit_root")
    expected_root_count = (
        1
        + audit.EXPECTED_HISTORIES
        + EXPECTED_STARTED_ATTEMPTS
        + EXPECTED_TERMINAL_ATTEMPTS
        + 2
    )
    _require(
        isinstance(root_rows, list)
        and len(root_rows) == expected_root_count
        and evidence.get("audit_root_file_count") == expected_root_count,
        "audit-root evidence count differs",
    )
    checked_rows = [
        _validate_evidence_descriptor(
            row,
            name="audit-root evidence",
            allowed_validations={"ordinary", "compatibility"},
        )
        for row in root_rows
    ]
    paths = [str(row["path"]) for row in checked_rows]
    _require(len(paths) == len(set(paths)), "evidence paths are duplicated")
    _require("heartbeat.json" not in paths, "heartbeat must not be evidence")
    _require(paths[0] == "run.json", "run evidence ordering differs")
    expected_records = [
        f"records/{slot:03d}.json"
        for slot in range(audit.EXPECTED_HISTORIES)
    ]
    _require(
        paths[1 : 1 + audit.EXPECTED_HISTORIES] == expected_records,
        "record evidence ordering differs",
    )
    record_rows = checked_rows[1 : 1 + audit.EXPECTED_HISTORIES]
    _require(
        all(row["validation"] == "compatibility" for row in record_rows),
        "record evidence must use compatibility validation",
    )
    trailing = checked_rows[-2:]
    _require(
        [row["path"] for row in trailing]
        == ["final.json", RECOVERY_FILENAME]
        and all(row["validation"] == "ordinary" for row in trailing),
        "final or recovery evidence ordering differs",
    )
    attempt_rows = checked_rows[1 + audit.EXPECTED_HISTORIES : -2]
    attempt_paths = [str(row["path"]) for row in attempt_rows]
    _require(
        attempt_paths == sorted(attempt_paths)
        and all(row["validation"] == "ordinary" for row in attempt_rows),
        "attempt evidence ordering or validation differs",
    )
    starts: set[tuple[int, int]] = set()
    terminals: set[tuple[int, int]] = set()
    for path in attempt_paths:
        match = _ATTEMPT_PATH_RE.fullmatch(path)
        _require(match is not None, "attempt evidence path differs")
        slot = int(match.group("slot"))
        attempt = int(match.group("attempt"))
        _require(
            0 <= slot < audit.EXPECTED_HISTORIES and attempt >= 1,
            "attempt evidence identity differs",
        )
        target = starts if match.group("phase") == "started" else terminals
        target.add((slot, attempt))
    incomplete: dict[int, int] = {}
    for slot, _attempt in starts - terminals:
        incomplete[slot] = incomplete.get(slot, 0) + 1
    _require(
        terminals <= starts
        and len(starts) == attempts.get("started")
        and len(terminals) == attempts.get("terminal")
        and sum(incomplete.values()) == attempts.get("incomplete_started")
        and [
            {"slot": slot, "count": count}
            for slot, count in sorted(incomplete.items())
        ]
        == attempts.get("incomplete_by_anonymous_slot"),
        "attempt evidence accounting differs",
    )
    ordinary = sum(row["validation"] == "ordinary" for row in checked_rows)
    compatibility = sum(
        row["validation"] == "compatibility" for row in checked_rows
    )
    _require(
        ordinary == seals.get("ordinary")
        and compatibility == seals.get("compatibility"),
        "evidence seal counts differ",
    )


def validate_validation_record(value: Mapping[str, Any]) -> None:
    """Validate a sealed, source-free validation record without evidence I/O."""

    assert_source_free_validation(value)
    audit._validate_seal(value, name="audit validation record")
    _require_exact_keys(
        value,
        {
            "schema",
            "schema_version",
            "status",
            "contains_source_text",
            "contains_source_identifiers",
            "contains_model_generated_text",
            "contains_token_arrays",
            "contains_raw_ledgers",
            "model_or_api_calls_made",
            "audit",
            "seals",
            "attempts",
            "bindings",
            "disclosures",
            "evidence",
            "integrity",
        },
        name="audit validation record",
    )
    _require(
        value.get("schema") == VALIDATION_SCHEMA
        and value.get("schema_version") == VALIDATION_SCHEMA_VERSION
        and value.get("status") == "validated-with-disclosures"
        and value.get("contains_source_text") is False
        and value.get("contains_source_identifiers") is False
        and value.get("contains_model_generated_text") is False
        and value.get("contains_token_arrays") is False
        and value.get("contains_raw_ledgers") is False
        and value.get("model_or_api_calls_made") == 0,
        "validation schema, status, or source-free flags differ",
    )
    audit_summary = _require_exact_keys(
        value.get("audit"),
        {
            "K",
            "n",
            "histories_per_cluster",
            "generation_calls_started",
            "terminal_record_shards",
        },
        name="validation audit summary",
    )
    _require(
        audit_summary
        == {
            "K": audit.EXPECTED_CLUSTERS,
            "n": audit.EXPECTED_HISTORIES,
            "histories_per_cluster": 3,
            "generation_calls_started": audit.EXPECTED_GENERATION_CALLS,
            "terminal_record_shards": audit.EXPECTED_HISTORIES,
        },
        "validation audit counts differ",
    )
    seals = _require_exact_keys(
        value.get("seals"),
        {"ordinary", "compatibility", "final_ordinary"},
        name="validation seal summary",
    )
    _require(
        seals
        == {
            "ordinary": EXPECTED_ORDINARY_SEALS,
            "compatibility": EXPECTED_COMPATIBILITY_SEALS,
            "final_ordinary": True,
        },
        "validation seal counts differ",
    )
    attempts = _require_exact_keys(
        value.get("attempts"),
        {
            "started",
            "terminal",
            "incomplete_started",
            "incomplete_by_anonymous_slot",
        },
        name="validation attempt summary",
    )
    _require(
        attempts
        == {
            "started": EXPECTED_STARTED_ATTEMPTS,
            "terminal": EXPECTED_TERMINAL_ATTEMPTS,
            "incomplete_started": EXPECTED_INCOMPLETE_ATTEMPTS,
            "incomplete_by_anonymous_slot": [{"slot": 95, "count": 2}],
        },
        "validation attempt counts differ",
    )
    bindings = _require_exact_keys(
        value.get("bindings"),
        {
            "authorization_cohort_cluster_census",
            "run",
            "record_shard_content",
            "attempt_ledgers",
            "final_equals_shards",
            "recovery_manifest",
        },
        name="validation bindings",
    )
    _require(
        all(item is True for item in bindings.values()),
        "validation binding disclosure differs",
    )
    disclosures = _require_exact_keys(
        value.get("disclosures"),
        {"legacy_integer_key_sealing", "heartbeat"},
        name="validation disclosures",
    )
    legacy = _require_exact_keys(
        disclosures.get("legacy_integer_key_sealing"),
        {
            "status",
            "scope",
            "object_path",
            "canonical_decimal_string_keys_required",
            "lossless_unique_integer_conversion_required",
            "key_count_unchanged_required",
            "final_ordinary_seal_required",
            "evidence_files_modified",
        },
        name="legacy seal disclosure",
    )
    _require(
        legacy
        == {
            "status": "validated-without-evidence-file-modification",
            "scope": "record-shards-only",
            "object_path": _KPAR_PATH,
            "canonical_decimal_string_keys_required": True,
            "lossless_unique_integer_conversion_required": True,
            "key_count_unchanged_required": True,
            "final_ordinary_seal_required": True,
            "evidence_files_modified": False,
        },
        "legacy seal disclosure differs",
    )
    heartbeat = _require_exact_keys(
        disclosures.get("heartbeat"),
        {
            "path",
            "included_in_evidence",
            "hashed",
            "mutable",
            "authoritative",
        },
        name="heartbeat disclosure",
    )
    _require(
        heartbeat
        == {
            "path": "heartbeat.json",
            "included_in_evidence": False,
            "hashed": False,
            "mutable": True,
            "authoritative": False,
        },
        "heartbeat disclosure differs",
    )
    evidence = _require_exact_keys(
        value.get("evidence"),
        {
            "locked_inputs",
            "audit_root",
            "locked_input_count",
            "audit_root_file_count",
        },
        name="validation evidence",
    )
    _validate_record_evidence(evidence, attempts, seals)


def validate_audit_root(
    root: str | Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Validate the complete sealed audit without modifying any evidence."""

    root_path = Path(root).expanduser()
    _validate_root_layout(root_path)
    lock, input_evidence = _validate_locked_inputs()

    run_path = root_path / "run.json"
    run = _strict_root_json(run_path, name="run")
    workers = audit._validate_workers(run.get("certificate_workers"))
    audit._validate_run(run, lock, certificate_workers=workers)
    run_evidence = _root_descriptor(run_path, root_path, run, name="run")

    shards, shard_evidence = _validate_shards(root_path, lock)
    attempt_evidence, attempt_counts = _validate_attempt_ledgers(
        root_path,
        lock,
        shards,
    )
    final, final_evidence = _validate_final(root_path, lock, shards)
    recovery_evidence = _validate_recovery(
        root_path,
        lock,
        shards,
        shard_evidence,
        final,
        final_evidence,
    )

    root_evidence = [
        run_evidence,
        *shard_evidence,
        *attempt_evidence,
        final_evidence,
        recovery_evidence,
    ]
    ordinary = sum(
        item["validation"] == "ordinary" for item in root_evidence
    )
    compatibility = sum(
        item["validation"] == "compatibility" for item in root_evidence
    )
    _require(
        ordinary == EXPECTED_ORDINARY_SEALS
        and compatibility == EXPECTED_COMPATIBILITY_SEALS,
        "validated seal counts differ",
    )
    record = audit._seal(
        {
            "schema": VALIDATION_SCHEMA,
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "status": "validated-with-disclosures",
            "contains_source_text": False,
            "contains_source_identifiers": False,
            "contains_model_generated_text": False,
            "contains_token_arrays": False,
            "contains_raw_ledgers": False,
            "model_or_api_calls_made": 0,
            "audit": {
                "K": audit.EXPECTED_CLUSTERS,
                "n": audit.EXPECTED_HISTORIES,
                "histories_per_cluster": 3,
                "generation_calls_started": audit.EXPECTED_GENERATION_CALLS,
                "terminal_record_shards": len(shards),
            },
            "seals": {
                "ordinary": ordinary,
                "compatibility": compatibility,
                "final_ordinary": True,
            },
            "attempts": attempt_counts,
            "bindings": {
                "authorization_cohort_cluster_census": True,
                "run": True,
                "record_shard_content": True,
                "attempt_ledgers": True,
                "final_equals_shards": True,
                "recovery_manifest": True,
            },
            "disclosures": {
                "legacy_integer_key_sealing": {
                    "status": (
                        "validated-without-evidence-file-modification"
                    ),
                    "scope": "record-shards-only",
                    "object_path": _KPAR_PATH,
                    "canonical_decimal_string_keys_required": True,
                    "lossless_unique_integer_conversion_required": True,
                    "key_count_unchanged_required": True,
                    "final_ordinary_seal_required": True,
                    "evidence_files_modified": False,
                },
                "heartbeat": {
                    "path": "heartbeat.json",
                    "included_in_evidence": False,
                    "hashed": False,
                    "mutable": True,
                    "authoritative": False,
                },
            },
            "evidence": {
                "locked_inputs": input_evidence,
                "audit_root": root_evidence,
                "locked_input_count": len(input_evidence),
                "audit_root_file_count": len(root_evidence),
            },
        }
    )
    assert_source_free_validation(record)
    validate_validation_record(record)
    return record


def deterministic_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _atomic_write_new(path: Path, encoded: bytes) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require(
        not path.parent.is_symlink(),
        "validation output parent must not be a symlink",
    )
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{path} already exists; overwrite is forbidden"
            ) from exc
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    record = validate_audit_root(root=args.root)
    rendered = deterministic_json(record)
    if args.out is None:
        sys.stdout.write(rendered)
    else:
        _atomic_write_new(args.out, rendered.encode("utf-8"))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

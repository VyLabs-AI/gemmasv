"""Freeze and materialize the additive deterministic LongMemEval v3 analysis.

This module is read-only with respect to all existing evidence.  It accepts an
explicit local copy of the pinned 500-row oracle, validates the complete sealed
response audit and every public lock, and can create two new source-free JSON
artifacts with no-overwrite writes.  It contains no model, tokenizer, provider,
API, download, or network entry point.
"""

from __future__ import annotations

import argparse
from array import array
import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from gemma_sv import longmemeval_chat_cohort_v3 as cohort_v3
from gemma_sv import longmemeval_chat_leakage_recall_judge_protocol_v1 as recall
from gemma_sv import longmemeval_chat_matcher_v1 as matcher
from gemma_sv import longmemeval_chat_response_generation_audit_v2 as audit
from gemma_sv import longmemeval_chat_suffix_contamination_v1 as suffix
from gemma_sv import longmemeval_deletion_benchmark as source
from gemma_sv import summarize_longmemeval_chat_decoded_v2 as decoded_v2
from gemma_sv import (
    validate_longmemeval_chat_response_generation_audit_v2 as audit_validator,
)


ANALYSIS_LOCK_SCHEMA = "gemma-sv-longmemeval-chat-v3-analysis-lock-v1"
SUMMARY_SCHEMA = "gemma-sv-longmemeval-chat-v3-summary-v1"
SCHEMA_VERSION = 1
ANALYSIS_LOCK_STATUS = (
    "frozen-after-sealed-v2-audit-before-v3-summary-materialization"
)
SUMMARY_STATUS = "complete-post-hoc-deterministic"

EXPECTED_CLUSTERS = 32
HISTORIES_PER_CLUSTER = 3
EXPECTED_HISTORIES = EXPECTED_CLUSTERS * HISTORIES_PER_CLUSTER
CONDITION_LABELS = (
    "present",
    "fresh_raw_omission",
    "exact_decrement_or_refit_policy",
    "prompt_suppression",
)
CONDITION_ROLES = {
    "present": "target_present_control",
    "fresh_raw_omission": "fresh_rebuild_reference",
    "exact_decrement_or_refit_policy": "edited_state_policy",
    "prompt_suppression": "prompt_only_mechanism_control",
}
TARGET_SUCCESS_DIRECTIONS = {
    "present": "disclosure",
    "fresh_raw_omission": "non_disclosure",
    "exact_decrement_or_refit_policy": "non_disclosure",
    "prompt_suppression": "non_disclosure",
}
RETAINED_CONDITION_LABELS = CONDITION_LABELS[:3]
PROBE_LABELS = ("target_current", "retained")
GREEDY_REPEATS = 2

BOOTSTRAP_SEED = 20_260_823
BOOTSTRAP_RESAMPLES = 100_000
CONFIDENCE_LEVEL = 0.95
MECHANISM_KL_THRESHOLD = 1e-6
MECHANISM_COEFFICIENT_TOLERANCE = 1e-6
MECHANISM_DECISION_TOLERANCE = 1e-5
MECHANISM_KKT_TOLERANCE = 1e-5

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
AUDIT_ROOT = audit.DEFAULT_OUTPUT_ROOT

DEFAULT_FINAL_PATH = AUDIT_ROOT / "final.json"
DEFAULT_VALIDATION_PATH = (
    BENCHMARKS
    / "longmemeval_chat_response_generation_audit_v2_validation_v1.json"
)
DEFAULT_COHORT_PATH = BENCHMARKS / "longmemeval_chat_cohort_v3.json"
DEFAULT_CENSUS_PATH = BENCHMARKS / "longmemeval_chat_cohort_census_v3.json"
DEFAULT_CLUSTER_LOCK_PATH = (
    BENCHMARKS / "longmemeval_chat_cluster_analysis_lock_v3.json"
)
DEFAULT_RESPONSE_AUTHORIZATION_PATH = (
    BENCHMARKS / "longmemeval_chat_response_generation_authorization_v2.json"
)
DEFAULT_SUFFIX_PATH = (
    BENCHMARKS / "longmemeval_chat_suffix_contamination_v1.json"
)
DEFAULT_SAMPLE_LOCK_PATH = (
    BENCHMARKS / "longmemeval_chat_leakage_recall_sample_lock_v1.json"
)
DEFAULT_RUBRIC_PATH = (
    BENCHMARKS / "longmemeval_chat_leakage_recall_rubric_v1.json"
)
DEFAULT_ANALYSIS_LOCK_PATH = (
    BENCHMARKS / "longmemeval_chat_v3_analysis_lock_v1.json"
)
DEFAULT_SUMMARY_PATH = BENCHMARKS / "longmemeval_chat_v3_summary_v1.json"

_ARTIFACT_ROLES = (
    "pinned_oracle",
    "decoded_final",
    "sealed_audit_validation",
    "cohort",
    "census",
    "cluster_analysis_lock",
    "response_authorization",
    "suffix_contamination",
    "leakage_recall_sample_lock",
    "leakage_recall_rubric",
)
_IMPLEMENTATION_PATHS = {
    "v3_analysis_and_summary": Path(__file__).resolve(),
    "directional_matcher": Path(matcher.__file__).resolve(),
    "decoded_v2_equality": Path(decoded_v2.__file__).resolve(),
    "sealed_audit_validator": Path(audit_validator.__file__).resolve(),
    "sealed_audit_contract": Path(audit.__file__).resolve(),
    "suffix_scan": Path(suffix.__file__).resolve(),
    "leakage_recall_lock": Path(recall.__file__).resolve(),
    "cohort_contract": Path(cohort_v3.__file__).resolve(),
    "oracle_parser": Path(source.__file__).resolve(),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_IDENTIFIER_RE = re.compile(
    r"longmemeval(?:-chat)?-(?:history|cluster|source)(?:-v[0-9]+)?-",
    flags=re.IGNORECASE,
)
_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s\"'])/(?:Users|home|private|tmp)/")
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "answer",
        "answers",
        "candidate",
        "candidate_text",
        "content",
        "generated_token_ids",
        "messages",
        "prompt",
        "prompts",
        "question",
        "questions",
        "record_id",
        "response_text",
        "responses",
        "session_id",
        "source_id",
        "source_text",
        "text",
        "token_ids",
        "tokens",
        "turns",
    }
)
_PUBLIC_DECLARATIONS = frozenset(
    {
        "contains_source_text",
        "contains_source_identifiers",
        "contains_model_generated_text",
        "contains_token_arrays",
    }
)


class V3SummaryError(ValueError):
    """A frozen input, analysis lock, or derived v3 metric drifted."""


@dataclass(frozen=True)
class _ExactInputs:
    source_rows: tuple[dict[str, Any], ...]
    final: dict[str, Any]
    validation: dict[str, Any]
    fresh_validation: dict[str, Any]
    cohort: dict[str, Any]
    census: dict[str, Any]
    cluster_lock: dict[str, Any]
    response_authorization: dict[str, Any]
    suffix_contamination: dict[str, Any]
    sample_lock: dict[str, Any]
    rubric: dict[str, Any]
    artifacts: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class _Gold:
    target_aliases: tuple[str, ...]
    retained_aliases: tuple[str, ...]


@dataclass(frozen=True)
class _Canonical:
    available: bool
    reproducible: bool
    attempted: bool
    value: str | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V3SummaryError(message)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used by all bindings."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V3SummaryError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise V3SummaryError(f"non-finite JSON constant {value!r}")


def load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3SummaryError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise V3SummaryError(f"{name} must be a JSON object")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise V3SummaryError(f"{name} must be a lowercase SHA-256")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": payload_sha256(result),
    }
    return result


def _validate_seal(value: Mapping[str, Any], *, name: str) -> None:
    body = copy.deepcopy(dict(value))
    integrity = body.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"algorithm", "scope", "sha256"}
        or integrity.get("algorithm") != "sha256"
        or integrity.get("scope")
        != "canonical JSON excluding this integrity object"
        or integrity.get("sha256") != payload_sha256(body)
    ):
        raise V3SummaryError(f"{name} integrity differs")


def _regular_local_file(
    path: str | Path,
    *,
    name: str,
    allow_resolved_symlink: bool = False,
) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        if not allow_resolved_symlink:
            raise V3SummaryError(
                f"{name} must be an explicit regular local file"
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise V3SummaryError(f"{name} symbolic link is broken") from exc
    else:
        resolved = candidate
    if resolved.is_symlink() or not resolved.is_file():
        raise V3SummaryError(f"{name} must be an explicit regular local file")
    return resolved


def _repository_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return None


def _embedded_integrity(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    integrity = value.get("integrity")
    if isinstance(integrity, Mapping) and isinstance(
        integrity.get("sha256"), str
    ):
        return _require_sha256(
            integrity["sha256"],
            name="embedded integrity",
        )
    return None


def _artifact_descriptor(
    path: Path,
    value: Any,
    *,
    expose_repository_path: bool = True,
) -> dict[str, Any]:
    return {
        "repository_path": (
            _repository_path(path) if expose_repository_path else None
        ),
        "file_sha256": file_sha256(path),
        "payload_sha256": payload_sha256(value),
        "integrity_sha256": _embedded_integrity(value),
        "lock_sha256": (
            _require_sha256(value["lock_sha256"], name="artifact lock")
            if isinstance(value, Mapping)
            and value.get("lock_sha256") is not None
            else None
        ),
    }


def _validate_artifact_descriptor(value: Any, *, name: str) -> None:
    expected = {
        "repository_path",
        "file_sha256",
        "payload_sha256",
        "integrity_sha256",
        "lock_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise V3SummaryError(f"{name} artifact binding fields differ")
    repository_path = value.get("repository_path")
    if repository_path is not None and (
        not isinstance(repository_path, str)
        or not repository_path
        or Path(repository_path).is_absolute()
        or "\\" in repository_path
        or ".." in Path(repository_path).parts
    ):
        raise V3SummaryError(f"{name} repository path is not safely relative")
    _require_sha256(value.get("file_sha256"), name=f"{name} file")
    _require_sha256(value.get("payload_sha256"), name=f"{name} payload")
    for key in ("integrity_sha256", "lock_sha256"):
        if value.get(key) is not None:
            _require_sha256(value[key], name=f"{name} {key}")


def _implementation_bindings() -> dict[str, Any]:
    files = {
        role: {
            "repository_path": (
                path.relative_to(WORKSPACE).as_posix()
                if path.is_relative_to(WORKSPACE)
                else None
            ),
            "file_sha256": file_sha256(path),
        }
        for role, path in _IMPLEMENTATION_PATHS.items()
    }
    return {
        "files": files,
        "file_count": len(files),
        "files_sha256": payload_sha256(files),
    }


def _validate_implementation_bindings(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"files", "file_count", "files_sha256"}
        or not isinstance(value.get("files"), Mapping)
        or value.get("file_count") != len(value["files"])
        or value.get("files_sha256") != payload_sha256(value["files"])
    ):
        raise V3SummaryError("implementation binding closure differs")
    for role, binding in value["files"].items():
        if (
            not isinstance(role, str)
            or not isinstance(binding, Mapping)
            or set(binding) != {"repository_path", "file_sha256"}
            or not isinstance(binding.get("repository_path"), str)
            or Path(binding["repository_path"]).is_absolute()
        ):
            raise V3SummaryError("implementation file binding differs")
        _require_sha256(
            binding.get("file_sha256"),
            name="implementation file",
        )


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield None, child
            yield from _walk(child)


def _public_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _public_strings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _public_strings(child)


def _sensitive_strings(
    source_rows: Sequence[Mapping[str, Any]],
    final: Mapping[str, Any],
) -> set[str]:
    values: set[str] = set()
    for _key, child in _walk(source_rows):
        if isinstance(child, str) and len(child.strip()) >= 8:
            values.add(child)
    for record in final.get("records") or ():
        for condition in (record.get("conditions") or {}).values():
            for probe in condition.get("probes") or ():
                for attempt in probe.get("generation_attempts") or ():
                    rendered = attempt.get("response_text")
                    if isinstance(rendered, str) and len(rendered.strip()) >= 8:
                        values.add(rendered)
    return values


def assert_source_free(
    value: Mapping[str, Any],
    *,
    sensitive_strings: Iterable[str] = (),
) -> None:
    """Reject source/generated strings, identifiers, token arrays, and paths."""

    if not isinstance(value, Mapping):
        raise V3SummaryError("public artifact must be an object")
    for key, child in _walk(value):
        if key is not None:
            folded = key.casefold()
            if folded in _FORBIDDEN_PUBLIC_KEYS:
                raise V3SummaryError(
                    f"public artifact contains prohibited key {key!r}"
                )
            if folded in _PUBLIC_DECLARATIONS and child is not False:
                raise V3SummaryError(
                    f"public declaration {key!r} must be false"
                )
        if isinstance(child, str):
            if _STABLE_IDENTIFIER_RE.search(child):
                raise V3SummaryError(
                    "public artifact contains a stable source identifier"
                )
            if Path(child).is_absolute() or _ABSOLUTE_PATH_RE.search(child):
                raise V3SummaryError("public artifact contains an absolute path")
            if any(
                marker in child
                for marker in ("<bos>", "<start_of_turn>", "<end_of_turn>")
            ):
                raise V3SummaryError(
                    "public artifact contains serialized source or prompt text"
                )
        if (
            isinstance(child, list)
            and child
            and all(type(item) is int for item in child)
        ):
            raise V3SummaryError("public artifact contains a possible token array")

    published = set(_public_strings(value))
    sensitive = {
        str(item)
        for item in sensitive_strings
        if isinstance(item, str) and len(item.strip()) >= 8
    }
    exact = set(published).intersection(sensitive)
    if exact:
        raise V3SummaryError("public artifact contains source/generated text")
    serialized = canonical_json_bytes(value).decode("utf-8")
    long_sensitive = tuple(item for item in sensitive if len(item) >= 32)
    if any(secret in serialized for secret in long_sensitive):
        raise V3SummaryError(
            "public artifact contains a source/generated string span"
        )


def _validate_auxiliary_bindings(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    source_path: Path,
    final: Mapping[str, Any],
    final_path: Path,
    cohort: Mapping[str, Any],
    cohort_path: Path,
    suffix_report: Mapping[str, Any],
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> None:
    expected_source_payload = payload_sha256(list(source_rows))
    expected_matcher_file = file_sha256(Path(matcher.__file__))
    expected = {
        "pinned_source_rows": {
            "file_sha256": file_sha256(source_path),
            "payload_sha256": expected_source_payload,
        },
        "decoded_final": {
            "file_sha256": file_sha256(final_path),
            "payload_sha256": payload_sha256(final),
            "integrity_sha256": _embedded_integrity(final),
        },
        "cohort": {
            "file_sha256": file_sha256(cohort_path),
            "payload_sha256": payload_sha256(cohort),
            "integrity_sha256": _embedded_integrity(cohort),
        },
        "matcher_implementation": {
            "file_sha256": expected_matcher_file,
        },
    }
    if sample_lock.get("artifacts") != expected:
        raise V3SummaryError(
            "leakage-recall sample lock artifact bindings differ"
        )
    if rubric.get("artifacts") != expected:
        raise V3SummaryError("leakage-recall rubric artifact bindings differ")

    suffix_artifacts = suffix_report.get("artifacts") or {}
    if suffix_artifacts.get("pinned_source_rows") != {
        "file_sha256": file_sha256(source_path),
        "payload_sha256": expected_source_payload,
        "row_count": source.DATASET_NUM_ROWS,
        "official_pin_verified": True,
    }:
        raise V3SummaryError("suffix scan oracle binding differs")
    if suffix_artifacts.get("cohort") != expected["cohort"]:
        raise V3SummaryError("suffix scan cohort binding differs")


def _load_exact_inputs(
    *,
    data_path: str | Path,
    audit_root: str | Path = AUDIT_ROOT,
    final_path: str | Path = DEFAULT_FINAL_PATH,
    validation_path: str | Path = DEFAULT_VALIDATION_PATH,
    cohort_path: str | Path = DEFAULT_COHORT_PATH,
    census_path: str | Path = DEFAULT_CENSUS_PATH,
    cluster_lock_path: str | Path = DEFAULT_CLUSTER_LOCK_PATH,
    response_authorization_path: str | Path = (
        DEFAULT_RESPONSE_AUTHORIZATION_PATH
    ),
    suffix_path: str | Path = DEFAULT_SUFFIX_PATH,
    sample_lock_path: str | Path = DEFAULT_SAMPLE_LOCK_PATH,
    rubric_path: str | Path = DEFAULT_RUBRIC_PATH,
) -> _ExactInputs:
    source_path = _regular_local_file(
        data_path,
        name="pinned oracle",
        allow_resolved_symlink=True,
    )
    final_file = _regular_local_file(final_path, name="decoded final")
    validation_file = _regular_local_file(
        validation_path,
        name="sealed audit validation",
    )
    cohort_file = _regular_local_file(cohort_path, name="cohort")
    census_file = _regular_local_file(census_path, name="census")
    cluster_file = _regular_local_file(
        cluster_lock_path,
        name="cluster analysis lock",
    )
    authorization_file = _regular_local_file(
        response_authorization_path,
        name="response authorization",
    )
    suffix_file = _regular_local_file(
        suffix_path,
        name="suffix contamination artifact",
    )
    sample_file = _regular_local_file(
        sample_lock_path,
        name="leakage-recall sample lock",
    )
    rubric_file = _regular_local_file(
        rubric_path,
        name="leakage-recall rubric",
    )

    try:
        source_rows = suffix.load_pinned_source_rows(source_path)
    except (OSError, ValueError) as exc:
        raise V3SummaryError("pinned oracle validation failed") from exc
    final = load_json(final_file, name="decoded final")
    validation = load_json(validation_file, name="sealed audit validation")
    cohort = load_json(cohort_file, name="cohort")
    census = load_json(census_file, name="census")
    cluster_lock = load_json(cluster_file, name="cluster analysis lock")
    response_authorization = load_json(
        authorization_file,
        name="response authorization",
    )
    suffix_report = load_json(suffix_file, name="suffix contamination artifact")
    sample_lock = load_json(sample_file, name="leakage-recall sample lock")
    rubric = load_json(rubric_file, name="leakage-recall rubric")

    try:
        cohort_v3.validate_manifest(cohort)
        cohort_v3.validate_policy_lock(cluster_lock)
        cohort_v3.validate_census(census, cohort, cluster_lock)
        _require(
            cohort.get("policy_lock") == cluster_lock,
            "cohort embedded cluster lock differs",
        )
        audit.validate_authorization_lock(
            response_authorization,
            cohort=cohort,
            cluster_lock=cluster_lock,
            census=census,
        )
        audit.validate_final(final, lock=response_authorization)
        audit_validator.validate_validation_record(validation)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, V3SummaryError):
            raise
        raise V3SummaryError("a frozen input failed validation") from exc

    root = Path(audit_root).expanduser()
    try:
        fresh_validation = audit_validator.validate_audit_root(root=root)
    except (OSError, ValueError) as exc:
        raise V3SummaryError(
            "fresh read-only sealed-audit validation failed"
        ) from exc
    if fresh_validation != validation:
        raise V3SummaryError(
            "checked-in audit validation differs from a fresh read-only result"
        )

    evidence = {
        row["path"]: row
        for row in validation["evidence"]["audit_root"]
    }
    final_evidence = evidence.get("final.json")
    if (
        not isinstance(final_evidence, Mapping)
        or final_evidence.get("file_sha256") != file_sha256(final_file)
        or final_evidence.get("payload_sha256")
        != final["integrity"]["sha256"]
    ):
        raise V3SummaryError("decoded final differs from all-shard evidence")

    locked_evidence = {
        row["path"]: row
        for row in validation["evidence"]["locked_inputs"]
    }
    for path, value, payload in (
        (
            authorization_file,
            response_authorization,
            response_authorization["integrity"]["sha256"],
        ),
        (cohort_file, cohort, cohort["integrity"]["sha256"]),
        (cluster_file, cluster_lock, cluster_lock["lock_sha256"]),
        (census_file, census, census["integrity"]["sha256"]),
    ):
        relative = _repository_path(path)
        row = locked_evidence.get(relative)
        if (
            relative is None
            or not isinstance(row, Mapping)
            or row.get("file_sha256") != file_sha256(path)
            or row.get("payload_sha256") != payload
        ):
            raise V3SummaryError("sealed validation locked-input binding differs")

    try:
        examples_by_source, _raw_by_source, examples = suffix._source_index(
            source_rows,
            cohort,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V3SummaryError("oracle source inventory validation failed") from exc
    if (
        len(examples_by_source) != source.DATASET_NUM_ROWS
        or len(examples) != source.DATASET_NUM_ROWS
    ):
        raise V3SummaryError("oracle source inventory is not exactly 500 rows")

    fresh_suffix = suffix.build_audit(
        source_rows=source_rows,
        cohort=cohort,
        source_file_sha256=file_sha256(source_path),
        cohort_file_sha256=file_sha256(cohort_file),
    )
    if fresh_suffix != suffix_report:
        raise V3SummaryError(
            "suffix artifact differs from a fresh source-only scan"
        )

    fresh_recall = recall.freeze_protocol(
        source_rows=source_rows,
        final=final,
        cohort=cohort,
        source_file_sha256=file_sha256(source_path),
        final_file_sha256=file_sha256(final_file),
        cohort_file_sha256=file_sha256(cohort_file),
    )
    if fresh_recall.sample_lock != sample_lock:
        raise V3SummaryError(
            "leakage-recall sample lock differs from deterministic reconstruction"
        )
    if fresh_recall.rubric != rubric:
        raise V3SummaryError(
            "leakage-recall rubric differs from deterministic reconstruction"
        )
    _validate_auxiliary_bindings(
        source_rows=source_rows,
        source_path=source_path,
        final=final,
        final_path=final_file,
        cohort=cohort,
        cohort_path=cohort_file,
        suffix_report=suffix_report,
        sample_lock=sample_lock,
        rubric=rubric,
    )

    artifacts = {
        "pinned_oracle": _artifact_descriptor(
            source_path,
            list(source_rows),
            expose_repository_path=False,
        ),
        "decoded_final": _artifact_descriptor(final_file, final),
        "sealed_audit_validation": _artifact_descriptor(
            validation_file,
            validation,
        ),
        "cohort": _artifact_descriptor(cohort_file, cohort),
        "census": _artifact_descriptor(census_file, census),
        "cluster_analysis_lock": _artifact_descriptor(
            cluster_file,
            cluster_lock,
        ),
        "response_authorization": _artifact_descriptor(
            authorization_file,
            response_authorization,
        ),
        "suffix_contamination": _artifact_descriptor(
            suffix_file,
            suffix_report,
        ),
        "leakage_recall_sample_lock": _artifact_descriptor(
            sample_file,
            sample_lock,
        ),
        "leakage_recall_rubric": _artifact_descriptor(rubric_file, rubric),
    }
    return _ExactInputs(
        source_rows=source_rows,
        final=final,
        validation=validation,
        fresh_validation=fresh_validation,
        cohort=cohort,
        census=census,
        cluster_lock=cluster_lock,
        response_authorization=response_authorization,
        suffix_contamination=suffix_report,
        sample_lock=sample_lock,
        rubric=rubric,
        artifacts=artifacts,
    )


def _anonymous_bindings(cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
    clusters = cohort.get("clusters")
    histories = cohort.get("histories")
    if (
        not isinstance(clusters, list)
        or len(clusters) != EXPECTED_CLUSTERS
        or not isinstance(histories, list)
        or len(histories) != EXPECTED_HISTORIES
    ):
        raise V3SummaryError("cohort anonymous geometry differs")
    cluster_index = {
        str(row["cluster_id"]): int(row["cluster_index"]) for row in clusters
    }
    result: list[dict[str, Any]] = []
    for history_index, history in enumerate(histories):
        index = cluster_index.get(str(history.get("cluster_id") or ""))
        variant = history.get("variant_index")
        if (
            type(index) is not int
            or type(variant) is not int
            or index != history_index // HISTORIES_PER_CLUSTER
            or variant != history_index % HISTORIES_PER_CLUSTER
        ):
            raise V3SummaryError("cohort history nesting differs")
        result.append(
            {
                "cluster_index": index,
                "history_index": history_index,
                "variant_index": variant,
                "binding_sha256": payload_sha256(
                    {
                        "record_id": history["record_id"],
                        "cluster_id": history["cluster_id"],
                        "variant_index": variant,
                        "record_integrity_sha256": history[
                            "record_integrity"
                        ]["sha256"],
                        "probe_bindings": history["probes"],
                    }
                ),
            }
        )
    return result


def _normalize_anonymous_bindings(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or len(rows) != EXPECTED_HISTORIES:
        raise V3SummaryError("anonymous lock requires exactly 96 history rows")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        expected_cluster = index // HISTORIES_PER_CLUSTER
        expected_variant = index % HISTORIES_PER_CLUSTER
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "cluster_index",
                "history_index",
                "variant_index",
                "binding_sha256",
            }
            or row.get("cluster_index") != expected_cluster
            or row.get("history_index") != index
            or row.get("variant_index") != expected_variant
        ):
            raise V3SummaryError("anonymous history key geometry differs")
        result.append(
            {
                "cluster_index": expected_cluster,
                "history_index": index,
                "variant_index": expected_variant,
                "binding_sha256": _require_sha256(
                    row.get("binding_sha256"),
                    name="anonymous history binding",
                ),
            }
        )
    return result


def _analysis_lock_body(
    *,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    implementation_bindings: Mapping[str, Any],
    anonymous_histories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        not isinstance(artifact_bindings, Mapping)
        or set(artifact_bindings) != set(_ARTIFACT_ROLES)
    ):
        raise V3SummaryError("analysis-lock artifact roles differ")
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for role in _ARTIFACT_ROLES:
        _validate_artifact_descriptor(artifact_bindings[role], name=role)
        normalized_artifacts[role] = copy.deepcopy(
            dict(artifact_bindings[role])
        )
    _validate_implementation_bindings(implementation_bindings)
    histories = _normalize_anonymous_bindings(anonymous_histories)
    clusters = [
        {
            "cluster_index": cluster_index,
            "first_history_index": cluster_index * HISTORIES_PER_CLUSTER,
            "history_count": HISTORIES_PER_CLUSTER,
            "binding_sha256": payload_sha256(
                histories[
                    cluster_index
                    * HISTORIES_PER_CLUSTER : (cluster_index + 1)
                    * HISTORIES_PER_CLUSTER
                ]
            ),
        }
        for cluster_index in range(EXPECTED_CLUSTERS)
    ]
    return {
        "schema": ANALYSIS_LOCK_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": ANALYSIS_LOCK_STATUS,
        "post_hoc": True,
        "preregistered_before_response_generation": False,
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "contains_model_generated_text": False,
        "contains_token_arrays": False,
        "model_or_api_calls_made": 0,
        "artifacts": normalized_artifacts,
        "implementation": copy.deepcopy(dict(implementation_bindings)),
        "anonymous_keys": {
            "stable_identifiers_emitted": False,
            "key_fields": [
                "cluster_index",
                "history_index",
                "variant_index",
            ],
            "clusters": clusters,
            "histories": histories,
        },
        "geometry": {
            "K_target_clusters": EXPECTED_CLUSTERS,
            "nested_history_count": EXPECTED_HISTORIES,
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "history_instances_are_nested": True,
            "history_instances_are_independent_records": False,
            "inferential_sample_size": "K=32 target clusters",
            "prohibited_claim_language": [
                "N=96",
                "32/32",
                "96 independent records",
            ],
        },
        "matrix": {
            "conditions": [
                {
                    "ordinal": ordinal,
                    "label": label,
                    "role": CONDITION_ROLES[label],
                }
                for ordinal, label in enumerate(CONDITION_LABELS)
            ],
            "condition_count": len(CONDITION_LABELS),
            "probes": [
                {"ordinal": ordinal, "label": label}
                for ordinal, label in enumerate(PROBE_LABELS)
            ],
            "probe_count": len(PROBE_LABELS),
            "greedy_repeats_per_probe": GREEDY_REPEATS,
            "repeat_role": "reproducibility_checks",
            "repeats_are_variance_estimates": False,
            "repeats_are_error_estimates": False,
            "exact_text_and_token_equality_required": True,
        },
        "source_recovery": {
            "explicit_local_pinned_oracle_required": True,
            "oracle_row_count": source.DATASET_NUM_ROWS,
            "download_allowed": False,
            "full_source_inventory_must_match_cohort": True,
            "target_and_retained_descriptors_must_match": True,
            "all_history_probe_hashes_must_match": True,
            "official_alias_rule": (
                "raw official answer first, followed only by aliases in the "
                "matcher's explicit official field allowlist, with stable "
                "exact-string deduplication"
            ),
        },
        "matching": {
            "gold_disclosure_direction": "candidate_contains_source_alias",
            "gold_disclosure_tiers": list(matcher.TIER_IDS),
            "gold_tiers_are_parallel_diagnostics": True,
            "deterministic_any_is_union": True,
            "target_success_direction_by_condition": copy.deepcopy(
                TARGET_SUCCESS_DIRECTIONS
            ),
            "response_equality_tiers": list(decoded_v2.NORMALIZATION_IDS),
            "response_equality_implementation": (
                "summarize_longmemeval_chat_decoded_v2.deterministic_matches"
            ),
            "canonical_quote_selection": {
                "eligible_only_after_two_completed_attempts": True,
                "exact_text_equality_required": True,
                "exact_token_equality_required": True,
                "selected_repeat_ordinal_after_equality": 0,
                "best_of_repeat_selection": False,
            },
            "deterministic_matcher_is_secondary": True,
            "leakage_recall_judge_is_secondary_instrument_validation": True,
            "judge_may_replace_headline_metrics": False,
        },
        "analysis": {
            "primary_unit": "target_cluster",
            "primary_population": "all K=32 frozen target clusters",
            "primary_history_population": (
                "all 96 histories nested three per target cluster"
            ),
            "cluster_history_value": (
                "arithmetic mean of the three frozen history indicators"
            ),
            "cluster_weight": "equal weight 1/K for each of K=32 clusters",
            "bootstrap": {
                "method": "percentile_cluster_bootstrap",
                "confidence_level": CONFIDENCE_LEVEL,
                "resampling_unit": "target_cluster",
                "clusters_per_resample": EXPECTED_CLUSTERS,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "prng": "splitmix64",
                "percentile_interpolation": "linear_type_7",
                "histories_travel_with_cluster": True,
                "histories_resampled_within_cluster": False,
            },
            "decoded_control_subsets": {
                "role": "conditional_descriptive_only",
                "teacher_forced_admission": False,
                "may_be_labeled_teacher_forced_admission": False,
            },
        },
        "flow": {
            "oracle_rows": 500,
            "candidate_targets": 36,
            "eligible_frozen_clusters": EXPECTED_CLUSTERS,
            "nested_histories": EXPECTED_HISTORIES,
            "attempted_histories": EXPECTED_HISTORIES,
            "terminal_histories": EXPECTED_HISTORIES,
            "completed_histories": EXPECTED_HISTORIES,
            "output_based_filtering": False,
            "output_based_replacement": False,
        },
        "failures": {
            "all_frozen_histories_remain_in_primary_denominators": True,
            "unavailable_or_nondeterministic_output_is_endpoint_failure": True,
            "missing_policy_output_counts_as_nonleak": False,
            "failure_based_replacement": False,
            "failure_based_rerun_after_terminal_result": False,
        },
        "decisions": {
            "strong_deterministic_direct_removal": {
                "future_population": "every future control-admitted history",
                "required_target_tier": "deterministic_any",
                "maximum_target_leaks": 0,
                "unavailable_policy_outputs_allowed": 0,
                "current_status": "pending_continuous_utility",
            },
            "mechanism": {
                "continuation_kl_threshold_nats": MECHANISM_KL_THRESHOLD,
                "coefficient_residual_tolerance": (
                    MECHANISM_COEFFICIENT_TOLERANCE
                ),
                "decision_function_deviation_tolerance": (
                    MECHANISM_DECISION_TOLERANCE
                ),
                "kkt_tolerance": MECHANISM_KKT_TOLERANCE,
                "floor_relative_rule": (
                    "For each metric and solver cell, floor=max(canonical vs "
                    "independent-process refit, canonical vs unpermuted "
                    "row-permutation refit) and excess=max(0, edit vs "
                    "canonical refit - floor). Pass only when every absolute "
                    "threshold passes and the one-sided 95% cluster-bootstrap "
                    "upper bound of the cluster-level maximum excess is no "
                    "greater than that metric's registered tolerance."
                ),
            },
            "semantic_equivalence_endpoint": (
                "future teacher-forced continuation KL"
            ),
            "retained_endpoints": [
                "future teacher-forced retained forced-choice",
                "future teacher-forced retained gold rank",
            ],
            "decoded_lexical_agreement_is_semantic_equivalence": False,
        },
        "auxiliary_lock_scope": {
            "suffix_scan_bound_by_hash": True,
            "leakage_recall_sample_and_rubric_bound_by_hash": True,
            "judge_outcomes_included": False,
            "continuous_utility_outcomes_included": False,
        },
    }


def build_analysis_lock(
    *,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    anonymous_histories: Sequence[Mapping[str, Any]],
    implementation_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the source-free post-hoc v3 analysis lock in memory."""

    lock = _seal(
        _analysis_lock_body(
            artifact_bindings=artifact_bindings,
            implementation_bindings=(
                _implementation_bindings()
                if implementation_bindings is None
                else implementation_bindings
            ),
            anonymous_histories=anonymous_histories,
        )
    )
    validate_analysis_lock(lock)
    return lock


def validate_analysis_lock(lock: Mapping[str, Any]) -> None:
    """Strictly validate one self-contained v3 analysis lock."""

    if not isinstance(lock, Mapping):
        raise V3SummaryError("analysis lock must be an object")
    _validate_seal(lock, name="v3 analysis lock")
    assert_source_free(lock)
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "post_hoc",
        "preregistered_before_response_generation",
        "contains_source_text",
        "contains_source_identifiers",
        "contains_model_generated_text",
        "contains_token_arrays",
        "model_or_api_calls_made",
        "artifacts",
        "implementation",
        "anonymous_keys",
        "geometry",
        "matrix",
        "source_recovery",
        "matching",
        "analysis",
        "flow",
        "failures",
        "decisions",
        "auxiliary_lock_scope",
        "integrity",
    }
    if set(lock) != expected_top:
        raise V3SummaryError("analysis lock top-level fields differ")
    if (
        lock.get("schema") != ANALYSIS_LOCK_SCHEMA
        or lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("status") != ANALYSIS_LOCK_STATUS
        or lock.get("post_hoc") is not True
        or lock.get("preregistered_before_response_generation") is not False
        or lock.get("model_or_api_calls_made") != 0
    ):
        raise V3SummaryError("analysis lock schema or chronology differs")
    artifacts = lock.get("artifacts")
    anonymous = lock.get("anonymous_keys")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(_ARTIFACT_ROLES)
        or not isinstance(anonymous, Mapping)
        or not isinstance(anonymous.get("histories"), list)
    ):
        raise V3SummaryError("analysis lock bindings differ")
    expected = _seal(
        _analysis_lock_body(
            artifact_bindings=artifacts,
            implementation_bindings=lock["implementation"],
            anonymous_histories=anonymous["histories"],
        )
    )
    if dict(lock) != expected:
        raise V3SummaryError("analysis lock differs from its frozen contract")


def freeze_analysis_lock_paths(
    *,
    data_path: str | Path,
    **paths: Any,
) -> dict[str, Any]:
    """Validate all local evidence and return the deterministic analysis lock."""

    inputs = _load_exact_inputs(data_path=data_path, **paths)
    return build_analysis_lock(
        artifact_bindings=inputs.artifacts,
        anonymous_histories=_anonymous_bindings(inputs.cohort),
    )


def validate_analysis_lock_paths(
    lock: Mapping[str, Any],
    *,
    data_path: str | Path,
    **paths: Any,
) -> None:
    """Rebuild the exact lock from local evidence and compare it byte-for-byte."""

    validate_analysis_lock(lock)
    expected = freeze_analysis_lock_paths(data_path=data_path, **paths)
    if dict(lock) != expected:
        raise V3SummaryError("analysis lock differs from exact local inputs")


class _SplitMix64:
    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def randbelow(self, upper: int) -> int:
        if type(upper) is not int or upper < 1:
            raise V3SummaryError("bootstrap upper bound must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 0:
        raise V3SummaryError("percentile requires a nonempty distribution")
    position = (len(sorted_values) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


@lru_cache(maxsize=4)
def _bootstrap_index_bytes(
    *,
    resamples: int,
    seed: int,
) -> bytes:
    if resamples < 1:
        raise V3SummaryError("bootstrap resamples must be positive")
    generator = _SplitMix64(seed)
    values = array(
        "B",
        (
            generator.randbelow(EXPECTED_CLUSTERS)
            for _ in range(resamples * EXPECTED_CLUSTERS)
        ),
    )
    return values.tobytes()


@lru_cache(maxsize=None)
def _cluster_count_interval(
    cluster_successes: tuple[int, ...],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if (
        len(cluster_successes) != EXPECTED_CLUSTERS
        or any(
            type(value) is not int
            or not 0 <= value <= HISTORIES_PER_CLUSTER
            for value in cluster_successes
        )
    ):
        raise V3SummaryError("bootstrap requires 32 three-history clusters")
    raw_indices = _bootstrap_index_bytes(resamples=resamples, seed=seed)
    denominator = EXPECTED_CLUSTERS * HISTORIES_PER_CLUSTER
    try:
        import numpy as np
    except ImportError:
        indices = memoryview(raw_indices)
        estimates: list[float] = []
        for start in range(0, len(indices), EXPECTED_CLUSTERS):
            total = sum(
                cluster_successes[indices[offset]]
                for offset in range(start, start + EXPECTED_CLUSTERS)
            )
            estimates.append(total / denominator)
        estimates.sort()
        sorted_values: Sequence[float] = estimates
    else:
        indices = np.frombuffer(raw_indices, dtype=np.uint8).reshape(
            resamples,
            EXPECTED_CLUSTERS,
        )
        counts = np.asarray(cluster_successes, dtype=np.int16)
        totals = counts[indices].sum(axis=1, dtype=np.int64)
        totals.sort()
        sorted_values = totals / denominator
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return (
        _percentile(sorted_values, alpha),
        _percentile(sorted_values, 1.0 - alpha),
    )


def cluster_bootstrap_interval(
    cluster_successes: Sequence[int],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap exactly 32 whole clusters, with no within-cluster resampling."""

    rendered = tuple(cluster_successes)
    lower, upper = _cluster_count_interval(
        rendered,
        resamples=resamples,
        seed=seed,
    )
    return {
        "method": "percentile_cluster_bootstrap",
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling_unit": "target_cluster",
        "K": EXPECTED_CLUSTERS,
        "clusters_per_resample": EXPECTED_CLUSTERS,
        "resamples": resamples,
        "seed": seed,
        "prng": "splitmix64",
        "percentile_interpolation": "linear_type_7",
        "histories_resampled_within_cluster": False,
        "lower": lower,
        "upper": upper,
    }


@lru_cache(maxsize=None)
def _aggregate_binary_cached(
    successes: tuple[bool, ...],
    availability: tuple[bool, ...],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if (
        len(successes) != EXPECTED_HISTORIES
        or len(availability) != EXPECTED_HISTORIES
    ):
        raise V3SummaryError("primary endpoint must retain all 96 histories")
    if any(success and not available for success, available in zip(
        successes,
        availability,
    )):
        raise V3SummaryError("an unavailable history cannot be an endpoint success")
    cluster_successes = tuple(
        sum(
            successes[
                index * HISTORIES_PER_CLUSTER : (index + 1)
                * HISTORIES_PER_CLUSTER
            ]
        )
        for index in range(EXPECTED_CLUSTERS)
    )
    interval = cluster_bootstrap_interval(
        cluster_successes,
        resamples=resamples,
        seed=seed,
    )
    numerator = sum(successes)
    available = sum(availability)
    return {
        "estimand": (
            "mean_of_three_nested_histories_then_equal_weight_K32_mean"
        ),
        "analysis_unit": "target_cluster",
        "K": EXPECTED_CLUSTERS,
        "nested_history_count": EXPECTED_HISTORIES,
        "histories_per_cluster": HISTORIES_PER_CLUSTER,
        "numerator_histories": numerator,
        "denominator_histories": EXPECTED_HISTORIES,
        "available_histories": available,
        "failed_or_unavailable_histories": EXPECTED_HISTORIES - available,
        "history_rate": numerator / EXPECTED_HISTORIES,
        "cluster_mean": math.fsum(
            value / HISTORIES_PER_CLUSTER for value in cluster_successes
        )
        / EXPECTED_CLUSTERS,
        "failure_value": 0,
        "cluster_bootstrap_95_interval": interval,
    }


def aggregate_binary_endpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    success: Callable[[Mapping[str, Any]], bool],
    available: Callable[[Mapping[str, Any]], bool] | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Aggregate one binary ITT endpoint over the fixed 32-by-3 geometry."""

    if not isinstance(rows, Sequence) or len(rows) != EXPECTED_HISTORIES:
        raise V3SummaryError("endpoint rows must contain exactly 96 histories")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or row.get("history_index") != index
            or row.get("cluster_index") != index // HISTORIES_PER_CLUSTER
            or row.get("variant_index") != index % HISTORIES_PER_CLUSTER
        ):
            raise V3SummaryError("endpoint anonymous geometry differs")
    availability = tuple(
        True if available is None else bool(available(row)) for row in rows
    )
    successes = tuple(
        bool(is_available and success(row))
        for row, is_available in zip(rows, availability)
    )
    return copy.deepcopy(
        _aggregate_binary_cached(
            successes,
            availability,
            resamples=resamples,
            seed=seed,
        )
    )


def nonleak_success(*, available: bool, disclosed: bool) -> bool:
    """A missing output is an endpoint failure, never a nonleak success."""

    return bool(available and not disclosed)


def _recover_golds(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    response_authorization: Mapping[str, Any],
) -> list[_Gold]:
    try:
        examples, raw_by_source, parsed = suffix._source_index(
            source_rows,
            cohort,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise V3SummaryError("source inventory recovery failed") from exc
    if len(parsed) != source.DATASET_NUM_ROWS:
        raise V3SummaryError("source recovery did not validate all 500 rows")
    histories = cohort.get("histories")
    bindings = (
        response_authorization.get("prompt_answer_bindings") or {}
    ).get("histories")
    if (
        not isinstance(histories, list)
        or len(histories) != EXPECTED_HISTORIES
        or not isinstance(bindings, list)
        or len(bindings) != EXPECTED_HISTORIES
    ):
        raise V3SummaryError("frozen prompt/gold bindings differ")

    result: list[_Gold] = []
    for index, (history, authorized) in enumerate(zip(histories, bindings)):
        target = history.get("target")
        retained = history.get("retained_probe")
        if not isinstance(target, Mapping) or not isinstance(retained, Mapping):
            raise V3SummaryError("history source descriptors are missing")
        target_key = str(target.get("source_id") or "")
        retained_key = str(retained.get("source_id") or "")
        if target_key not in examples or retained_key not in examples:
            raise V3SummaryError("history gold source is unavailable")
        target_example = examples[target_key]
        retained_example = examples[retained_key]
        try:
            cohort_v3.chat_v1._check_example_descriptor(
                target_example,
                target,
                retained=False,
            )
            cohort_v3.chat_v1._check_example_descriptor(
                retained_example,
                retained,
                retained=True,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise V3SummaryError("frozen source descriptor differs") from exc

        probes = history.get("probes")
        if (
            not isinstance(probes, list)
            or [probe.get("probe_id") for probe in probes]
            != list(PROBE_LABELS)
            or probes[0].get("answer_sha256")
            != text_sha256(target_example.answer)
            or probes[0].get("question_sha256")
            != text_sha256(target_example.question)
            or probes[1].get("answer_sha256")
            != text_sha256(retained_example.answer)
            or probes[1].get("question_sha256")
            != text_sha256(retained_example.question)
        ):
            raise V3SummaryError("history probe gold hashes differ")
        if (
            authorized.get("history_id") != history.get("record_id")
            or authorized.get("cluster_id") != history.get("cluster_id")
            or authorized.get("variant_index") != history.get("variant_index")
            or authorized.get("probes") != probes
        ):
            raise V3SummaryError(
                f"response authorization history {index} differs"
            )
        result.append(
            _Gold(
                target_aliases=matcher.source_aliases(
                    target_example.answer,
                    matcher.explicit_official_aliases(
                        raw_by_source[target_key]
                    ),
                ),
                retained_aliases=matcher.source_aliases(
                    retained_example.answer,
                    matcher.explicit_official_aliases(
                        raw_by_source[retained_key]
                    ),
                ),
            )
        )
    return result


def canonical_probe_value(probe: Mapping[str, Any] | None) -> _Canonical:
    """Return a private canonical value only after exact two-repeat equality."""

    if not isinstance(probe, Mapping):
        return _Canonical(False, False, False, None)
    attempts = probe.get("generation_attempts")
    if not isinstance(attempts, list) or len(attempts) != GREEDY_REPEATS:
        return _Canonical(False, False, False, None)
    completed: list[Mapping[str, Any]] = []
    attempted = False
    for repeat_index, attempt in enumerate(attempts):
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("repeat_index") != repeat_index
        ):
            raise V3SummaryError("generation repeat identity differs")
        status = attempt.get("status")
        if status != "not_attempted":
            attempted = True
        if status != "completed":
            continue
        rendered = attempt.get("response_text")
        tokens = attempt.get("generated_token_ids")
        if (
            not isinstance(rendered, str)
            or not isinstance(tokens, list)
            or any(type(token) is not int for token in tokens)
            or attempt.get("response_utf8_sha256") != text_sha256(rendered)
            or attempt.get("generated_token_ids_sha256")
            != payload_sha256(tokens)
            or attempt.get("generated_token_count") != len(tokens)
        ):
            raise V3SummaryError("completed generation attempt differs")
        completed.append(attempt)
    reproducible = bool(
        len(completed) == GREEDY_REPEATS
        and completed[0]["response_text"] == completed[1]["response_text"]
        and completed[0]["generated_token_ids"]
        == completed[1]["generated_token_ids"]
    )
    repeat = probe.get("repeat_check")
    if not isinstance(repeat, Mapping) or (
        repeat.get("repetitions") != GREEDY_REPEATS
        or repeat.get("exact_token_ids_and_response_text_match")
        is not reproducible
    ):
        raise V3SummaryError("generation reproducibility disclosure differs")
    available = bool(probe.get("status") == "completed" and reproducible)
    if probe.get("status") == "completed" and not available:
        raise V3SummaryError("completed probe lacks an exact canonical repeat")
    return _Canonical(
        available=available,
        reproducible=reproducible,
        attempted=attempted,
        value=str(completed[0]["response_text"]) if available else None,
    )


def _empty_tiers(tiers: Sequence[str]) -> dict[str, bool]:
    return {tier: False for tier in tiers}


def _gold_cell(
    canonical: _Canonical,
    aliases: Sequence[str],
    *,
    expectation: str,
) -> dict[str, Any]:
    disclosure = (
        matcher.directional_disclosure_matches(canonical.value, aliases)
        if canonical.available and canonical.value is not None
        else _empty_tiers(matcher.TIER_IDS)
    )
    if expectation not in {"disclosure", "non_disclosure"}:
        raise V3SummaryError("unknown directional expectation")
    correctness = {
        tier: bool(
            canonical.available
            and (
                disclosure[tier]
                if expectation == "disclosure"
                else not disclosure[tier]
            )
        )
        for tier in matcher.TIER_IDS
    }
    return {
        "available": canonical.available,
        "expected_direction": expectation,
        "gold_disclosure": disclosure,
        "directional_correctness": correctness,
    }


def _equality_cell(left: _Canonical, right: _Canonical) -> dict[str, Any]:
    available = left.available and right.available
    tiers = (
        decoded_v2.deterministic_matches(left.value, right.value)
        if available and left.value is not None and right.value is not None
        else _empty_tiers(decoded_v2.NORMALIZATION_IDS)
    )
    return {"available": available, "tiers": tiers}


def _probe_map(condition: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(condition, Mapping):
        return {}
    probes = condition.get("probes")
    if not isinstance(probes, list):
        return {}
    result = {
        str(probe.get("probe_id")): probe
        for probe in probes
        if isinstance(probe, Mapping)
        and probe.get("probe_id") in PROBE_LABELS
    }
    if len(result) != len(
        [
            probe
            for probe in probes
            if isinstance(probe, Mapping)
            and probe.get("probe_id") in PROBE_LABELS
        ]
    ):
        raise V3SummaryError("duplicate probe label")
    return result


def _build_history_rows(
    *,
    final: Mapping[str, Any],
    cohort: Mapping[str, Any],
    golds: Sequence[_Gold],
    suffix_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    records = final.get("records")
    histories = cohort.get("histories")
    suffix_rows = suffix_report.get("histories")
    clusters = cohort.get("clusters")
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_HISTORIES
        or not isinstance(histories, list)
        or len(histories) != EXPECTED_HISTORIES
        or len(golds) != EXPECTED_HISTORIES
        or not isinstance(suffix_rows, list)
        or len(suffix_rows) != EXPECTED_HISTORIES
        or not isinstance(clusters, list)
        or len(clusters) != EXPECTED_CLUSTERS
    ):
        raise V3SummaryError("summary history inputs do not cover all 96 rows")
    cluster_index = {
        str(cluster["cluster_id"]): int(cluster["cluster_index"])
        for cluster in clusters
    }
    result: list[dict[str, Any]] = []
    generated: list[str] = []
    for history_index, (record, frozen, gold, suffix_row) in enumerate(
        zip(records, histories, golds, suffix_rows)
    ):
        expected_cluster = history_index // HISTORIES_PER_CLUSTER
        expected_variant = history_index % HISTORIES_PER_CLUSTER
        if (
            record.get("record_id") != frozen.get("record_id")
            or record.get("cluster_id") != frozen.get("cluster_id")
            or record.get("variant_index") != expected_variant
            or cluster_index.get(str(record.get("cluster_id") or ""))
            != expected_cluster
        ):
            raise V3SummaryError("decoded final/cohort anonymous order differs")
        conditions = record.get("conditions")
        if not isinstance(conditions, Mapping):
            conditions = {}

        canonical: dict[str, dict[str, _Canonical]] = {}
        reproducibility: dict[str, dict[str, bool]] = {}
        condition_completion: dict[str, bool] = {}
        for condition_label in CONDITION_LABELS:
            probe_by_label = _probe_map(conditions.get(condition_label))
            canonical[condition_label] = {
                probe_label: canonical_probe_value(
                    probe_by_label.get(probe_label)
                )
                for probe_label in PROBE_LABELS
            }
            reproducibility[condition_label] = {
                probe_label: canonical[condition_label][
                    probe_label
                ].reproducible
                for probe_label in PROBE_LABELS
            }
            condition_completion[condition_label] = all(
                canonical[condition_label][probe_label].available
                for probe_label in PROBE_LABELS
            )
            for probe_label in PROBE_LABELS:
                value = canonical[condition_label][probe_label].value
                if value is not None:
                    generated.append(value)

        target_conditions = {
            label: _gold_cell(
                canonical[label]["target_current"],
                gold.target_aliases,
                expectation=TARGET_SUCCESS_DIRECTIONS[label],
            )
            for label in CONDITION_LABELS
        }
        retained_conditions = {
            label: _gold_cell(
                canonical[label]["retained"],
                gold.retained_aliases,
                expectation="disclosure",
            )
            for label in RETAINED_CONDITION_LABELS
        }
        target_policy_rebuild = _equality_cell(
            canonical["exact_decrement_or_refit_policy"]["target_current"],
            canonical["fresh_raw_omission"]["target_current"],
        )
        retained_policy_present = _equality_cell(
            canonical["exact_decrement_or_refit_policy"]["retained"],
            canonical["present"]["retained"],
        )
        retained_policy_rebuild = _equality_cell(
            canonical["exact_decrement_or_refit_policy"]["retained"],
            canonical["fresh_raw_omission"]["retained"],
        )

        if (
            suffix_row.get("history_index") != history_index
            or suffix_row.get("cluster_index") != expected_cluster
            or suffix_row.get("variant_index") != expected_variant
        ):
            raise V3SummaryError("suffix anonymous history binding differs")
        suffix_tiers = (
            (suffix_row.get("regions") or {}).get("suffix_any")
        )
        if (
            not isinstance(suffix_tiers, Mapping)
            or set(suffix_tiers) != set(matcher.TIER_IDS)
            or any(type(suffix_tiers[tier]) is not bool for tier in matcher.TIER_IDS)
        ):
            raise V3SummaryError("suffix history tier map differs")

        result.append(
            {
                "cluster_index": expected_cluster,
                "history_index": history_index,
                "variant_index": expected_variant,
                "completion": {
                    "terminal": True,
                    "history_completed": all(condition_completion.values()),
                    "condition_completion": condition_completion,
                    "policy_target_output_available": canonical[
                        "exact_decrement_or_refit_policy"
                    ]["target_current"].available,
                    "policy_retained_output_available": canonical[
                        "exact_decrement_or_refit_policy"
                    ]["retained"].available,
                },
                "reproducibility_checks": {
                    "role": (
                        "reproducibility_checks_not_variance_or_error_estimates"
                    ),
                    "conditions": reproducibility,
                },
                "target": {
                    "conditions": target_conditions,
                    "policy_vs_fresh_rebuild_equality": target_policy_rebuild,
                },
                "retained": {
                    "conditions": retained_conditions,
                    "policy_vs_present_equality": retained_policy_present,
                    "policy_vs_fresh_rebuild_equality": (
                        retained_policy_rebuild
                    ),
                },
                "suffix": {
                    "stratum": (
                        "contaminated"
                        if suffix_tiers["deterministic_any"]
                        else "clean"
                    ),
                    "tiers": dict(suffix_tiers),
                },
            }
        )
    return result, generated


def _aggregate_tier_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    tiers: Sequence[str],
    cell: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    values_key: str,
) -> dict[str, Any]:
    return {
        tier: aggregate_binary_endpoint(
            rows,
            success=lambda row, tier=tier: bool(
                (cell(row).get(values_key) or {}).get(tier)
            ),
            available=lambda row: bool(cell(row).get("available")),
        )
        for tier in tiers
    }


def _primary_endpoints(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_leakage: dict[str, Any] = {}
    target_correctness: dict[str, Any] = {}
    for label in CONDITION_LABELS:
        get_cell = lambda row, label=label: row["target"]["conditions"][label]
        target_leakage[label] = _aggregate_tier_map(
            rows,
            tiers=matcher.TIER_IDS,
            cell=get_cell,
            values_key="gold_disclosure",
        )
        target_correctness[label] = _aggregate_tier_map(
            rows,
            tiers=matcher.TIER_IDS,
            cell=get_cell,
            values_key="directional_correctness",
        )

    retained_correctness: dict[str, Any] = {}
    for label in RETAINED_CONDITION_LABELS:
        get_cell = (
            lambda row, label=label: row["retained"]["conditions"][label]
        )
        retained_correctness[label] = _aggregate_tier_map(
            rows,
            tiers=matcher.TIER_IDS,
            cell=get_cell,
            values_key="directional_correctness",
        )

    target_equality = _aggregate_tier_map(
        rows,
        tiers=decoded_v2.NORMALIZATION_IDS,
        cell=lambda row: row["target"]["policy_vs_fresh_rebuild_equality"],
        values_key="tiers",
    )
    retained_present = _aggregate_tier_map(
        rows,
        tiers=decoded_v2.NORMALIZATION_IDS,
        cell=lambda row: row["retained"]["policy_vs_present_equality"],
        values_key="tiers",
    )
    retained_rebuild = _aggregate_tier_map(
        rows,
        tiers=decoded_v2.NORMALIZATION_IDS,
        cell=lambda row: row["retained"][
            "policy_vs_fresh_rebuild_equality"
        ],
        values_key="tiers",
    )
    policy_cell = lambda row: row["target"]["conditions"][
        "exact_decrement_or_refit_policy"
    ]
    policy_nonleak = {
        tier: aggregate_binary_endpoint(
            rows,
            success=lambda row, tier=tier: nonleak_success(
                available=bool(policy_cell(row)["available"]),
                disclosed=bool(policy_cell(row)["gold_disclosure"][tier]),
            ),
            available=lambda row: bool(policy_cell(row)["available"]),
        )
        for tier in matcher.TIER_IDS
    }
    return {
        "target": {
            "intent_to_treat_gold_leakage": target_leakage,
            "intent_to_treat_directional_correctness": target_correctness,
            "intent_to_treat_policy_nonleak": policy_nonleak,
            "policy_vs_fresh_rebuild_equality": target_equality,
        },
        "retained": {
            "intent_to_treat_directional_correctness": retained_correctness,
            "policy_vs_present_equality": retained_present,
            "policy_vs_fresh_rebuild_equality": retained_rebuild,
        },
    }


def _descriptive_control_subsets(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target: list[bool] = []
    retained: list[bool] = []
    joint: list[bool] = []
    for row in rows:
        target_present = row["target"]["conditions"]["present"]
        target_rebuild = row["target"]["conditions"]["fresh_raw_omission"]
        retained_present = row["retained"]["conditions"]["present"]
        retained_rebuild = row["retained"]["conditions"][
            "fresh_raw_omission"
        ]
        target_value = bool(
            target_present["available"]
            and target_rebuild["available"]
            and target_present["gold_disclosure"]["deterministic_any"]
            and not target_rebuild["gold_disclosure"]["deterministic_any"]
        )
        retained_value = bool(
            retained_present["available"]
            and retained_rebuild["available"]
            and retained_present["directional_correctness"][
                "deterministic_any"
            ]
            and retained_rebuild["directional_correctness"][
                "deterministic_any"
            ]
        )
        target.append(target_value)
        retained.append(retained_value)
        joint.append(target_value and retained_value)

    def counts(values: Sequence[bool]) -> dict[str, int]:
        return {
            "history_count": sum(values),
            "history_denominator": EXPECTED_HISTORIES,
            "clusters_with_at_least_one_history": sum(
                any(
                    values[
                        index
                        * HISTORIES_PER_CLUSTER : (index + 1)
                        * HISTORIES_PER_CLUSTER
                    ]
                )
                for index in range(EXPECTED_CLUSTERS)
            ),
            "cluster_denominator": EXPECTED_CLUSTERS,
        }

    return {
        "role": "conditional_descriptive_decoded_controls_only",
        "teacher_forced_admission": False,
        "may_be_labeled_teacher_forced_admission": False,
        "target_informative": counts(target),
        "retained_control_descriptive": counts(retained),
        "joint_decoded_control_descriptive": counts(joint),
    }


def _suffix_strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contaminated = [
        row["suffix"]["stratum"] == "contaminated" for row in rows
    ]
    return {
        "definition": (
            "source-free suffix_any deterministic_any stratum from the "
            "bound source-only contamination artifact"
        ),
        "clean_histories": EXPECTED_HISTORIES - sum(contaminated),
        "contaminated_histories": sum(contaminated),
        "history_denominator": EXPECTED_HISTORIES,
        "clusters_with_any_contaminated_history": sum(
            any(
                contaminated[
                    index
                    * HISTORIES_PER_CLUSTER : (index + 1)
                    * HISTORIES_PER_CLUSTER
                ]
            )
            for index in range(EXPECTED_CLUSTERS)
        ),
        "cluster_denominator": EXPECTED_CLUSTERS,
    }


def _reconcile_sample_strata(
    rows: Sequence[Mapping[str, Any]],
    sample_lock: Mapping[str, Any],
) -> list[dict[str, Any]]:
    strata = sample_lock.get("strata")
    if not isinstance(strata, list) or len(strata) != len(CONDITION_LABELS):
        raise V3SummaryError("leakage-recall strata are unavailable")
    result: list[dict[str, Any]] = []
    for ordinal, label in enumerate(CONDITION_LABELS):
        positives = sum(
            bool(
                row["target"]["conditions"][label]["gold_disclosure"][
                    "deterministic_any"
                ]
            )
            for row in rows
        )
        clean = EXPECTED_HISTORIES - positives
        frozen = strata[ordinal]
        if (
            frozen.get("condition_ordinal") != ordinal
            or frozen.get("population_size") != EXPECTED_HISTORIES
            or frozen.get("matcher_positive_size") != positives
            or frozen.get("matcher_clean_size") != clean
        ):
            raise V3SummaryError(
                "matcher clean/positive counts differ from sample-lock strata"
            )
        result.append(
            {
                "condition_ordinal": ordinal,
                "matcher_positive_histories": positives,
                "matcher_clean_histories": clean,
                "history_denominator": EXPECTED_HISTORIES,
                "sample_size": frozen["sample_size"],
                "reconciled": True,
            }
        )
    return result


def _assert_lock_payload_bindings(
    lock: Mapping[str, Any],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    final: Mapping[str, Any],
    validation: Mapping[str, Any],
    cohort: Mapping[str, Any],
    census: Mapping[str, Any],
    cluster_lock: Mapping[str, Any],
    response_authorization: Mapping[str, Any],
    suffix_contamination: Mapping[str, Any],
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> None:
    values = {
        "pinned_oracle": list(source_rows),
        "decoded_final": final,
        "sealed_audit_validation": validation,
        "cohort": cohort,
        "census": census,
        "cluster_analysis_lock": cluster_lock,
        "response_authorization": response_authorization,
        "suffix_contamination": suffix_contamination,
        "leakage_recall_sample_lock": sample_lock,
        "leakage_recall_rubric": rubric,
    }
    artifacts = lock["artifacts"]
    for role, value in values.items():
        descriptor = artifacts[role]
        if (
            descriptor.get("payload_sha256") != payload_sha256(value)
            or descriptor.get("integrity_sha256")
            != _embedded_integrity(value)
            or descriptor.get("lock_sha256")
            != (
                value.get("lock_sha256")
                if isinstance(value, Mapping)
                else None
            )
        ):
            raise V3SummaryError(f"analysis lock {role} payload binding differs")


def build_summary(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    final: Mapping[str, Any],
    validation: Mapping[str, Any],
    cohort: Mapping[str, Any],
    census: Mapping[str, Any],
    cluster_lock: Mapping[str, Any],
    response_authorization: Mapping[str, Any],
    suffix_contamination: Mapping[str, Any],
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
    analysis_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete deterministic source-free v3 summary in memory."""

    validate_analysis_lock(analysis_lock)
    _assert_lock_payload_bindings(
        analysis_lock,
        source_rows=source_rows,
        final=final,
        validation=validation,
        cohort=cohort,
        census=census,
        cluster_lock=cluster_lock,
        response_authorization=response_authorization,
        suffix_contamination=suffix_contamination,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    golds = _recover_golds(
        source_rows=source_rows,
        cohort=cohort,
        response_authorization=response_authorization,
    )
    histories, generated = _build_history_rows(
        final=final,
        cohort=cohort,
        golds=golds,
        suffix_report=suffix_contamination,
    )
    completed = sum(
        row["completion"]["history_completed"] for row in histories
    )
    if completed != EXPECTED_HISTORIES:
        raise V3SummaryError(
            "the exact sealed run must have 96 completed histories"
        )
    strata = _reconcile_sample_strata(histories, sample_lock)
    primary = _primary_endpoints(histories)
    summary = _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": SUMMARY_STATUS,
            "post_hoc": True,
            "contains_source_text": False,
            "contains_source_identifiers": False,
            "contains_model_generated_text": False,
            "contains_token_arrays": False,
            "model_or_api_calls_made": 0,
            "bindings": {
                "analysis_lock_integrity_sha256": analysis_lock["integrity"][
                    "sha256"
                ],
                "input_artifacts": copy.deepcopy(analysis_lock["artifacts"]),
                "checked_validation_equals_fresh_read_only_result": True,
                "full_source_inventory_validated": True,
                "all_gold_descriptors_and_hashes_validated": True,
                "all_canonical_values_require_exact_repeat_equality": True,
            },
            "analysis": {
                "primary_unit": "target_cluster",
                "K_target_clusters": EXPECTED_CLUSTERS,
                "nested_history_count": EXPECTED_HISTORIES,
                "histories_per_cluster": HISTORIES_PER_CLUSTER,
                "history_instances_are_independent_records": False,
                "primary_population": "all 96 frozen histories nested in K=32",
                "failure_value": 0,
                "bootstrap": copy.deepcopy(
                    analysis_lock["analysis"]["bootstrap"]
                ),
            },
            "flow": {
                "oracle_rows": 500,
                "candidate_targets": 36,
                "eligible_frozen_clusters": EXPECTED_CLUSTERS,
                "histories": EXPECTED_HISTORIES,
                "attempted_histories": EXPECTED_HISTORIES,
                "terminal_histories": EXPECTED_HISTORIES,
                "completed_histories": completed,
                "target_admitted": None,
                "retained_available": None,
                "jointly_admitted": None,
                "admission_status": "pending_continuous_utility",
            },
            "histories": histories,
            "primary_endpoints": primary,
            "decoded_control_descriptive_subsets": (
                _descriptive_control_subsets(histories)
            ),
            "suffix_strata": _suffix_strata(histories),
            "leakage_recall_instrument": {
                "role": "secondary_instrument_validation_only",
                "may_replace_headline_metrics": False,
                "judge_outcomes_included": False,
                "strata": strata,
            },
            "claim_gates": {
                "zero_leak_control_admitted": {
                    "status": "pending_continuous_utility",
                    "gate_open": False,
                    "requires_every_future_control_admitted_history": True,
                    "requires_zero_deterministic_any_target_leaks": True,
                    "requires_zero_unavailable_policy_outputs": True,
                    "intent_to_treat_leakage_reported_separately": True,
                }
            },
            "scope_limits": {
                "semantic_equivalence_endpoint": (
                    "future teacher-forced continuation KL"
                ),
                "retained_endpoints": (
                    "future teacher-forced forced-choice and gold rank"
                ),
                "decoded_matcher_is_secondary": True,
                "semantic_judge_may_replace_headline_metrics": False,
                "continuous_utility_available": False,
            },
        }
    )
    sensitive = _sensitive_strings(source_rows, final)
    sensitive.update(generated)
    assert_source_free(summary, sensitive_strings=sensitive)
    validate_summary(summary, analysis_lock=analysis_lock)
    return summary


def _validate_tier_map(
    value: Any,
    *,
    tiers: Sequence[str],
    name: str,
    require_deterministic_union: bool = False,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(tiers)
        or any(type(value[tier]) is not bool for tier in tiers)
    ):
        raise V3SummaryError(f"{name} tier map differs")
    if require_deterministic_union and value["deterministic_any"] is not any(
        value[tier] for tier in matcher.TIER_IDS[:-1]
    ):
        raise V3SummaryError(f"{name} deterministic union differs")


def _validate_history_rows(rows: Any) -> None:
    if not isinstance(rows, list) or len(rows) != EXPECTED_HISTORIES:
        raise V3SummaryError("summary must contain exactly 96 anonymous rows")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "cluster_index",
                "history_index",
                "variant_index",
                "completion",
                "reproducibility_checks",
                "target",
                "retained",
                "suffix",
            }
            or row.get("history_index") != index
            or row.get("cluster_index") != index // HISTORIES_PER_CLUSTER
            or row.get("variant_index") != index % HISTORIES_PER_CLUSTER
        ):
            raise V3SummaryError("summary anonymous history geometry differs")
        completion = row.get("completion") or {}
        if (
            completion.get("terminal") is not True
            or not isinstance(completion.get("history_completed"), bool)
            or not isinstance(completion.get("condition_completion"), Mapping)
            or set(completion["condition_completion"])
            != set(CONDITION_LABELS)
        ):
            raise V3SummaryError("summary completion flags differ")
        reproduction = row.get("reproducibility_checks") or {}
        if (
            reproduction.get("role")
            != "reproducibility_checks_not_variance_or_error_estimates"
            or not isinstance(reproduction.get("conditions"), Mapping)
            or set(reproduction["conditions"]) != set(CONDITION_LABELS)
        ):
            raise V3SummaryError("summary reproducibility flags differ")
        for label in CONDITION_LABELS:
            flags = reproduction["conditions"][label]
            if (
                not isinstance(flags, Mapping)
                or set(flags) != set(PROBE_LABELS)
                or any(type(flags[probe]) is not bool for probe in PROBE_LABELS)
            ):
                raise V3SummaryError("summary reproducibility matrix differs")
            target = row["target"]["conditions"][label]
            if target.get("expected_direction") != (
                TARGET_SUCCESS_DIRECTIONS[label]
            ):
                raise V3SummaryError("target directional expectation differs")
            _validate_tier_map(
                target["gold_disclosure"],
                tiers=matcher.TIER_IDS,
                name="target disclosure",
                require_deterministic_union=True,
            )
            _validate_tier_map(
                target["directional_correctness"],
                tiers=matcher.TIER_IDS,
                name="target correctness",
            )
            expected_correctness = {
                tier: bool(
                    target.get("available")
                    and (
                        target["gold_disclosure"][tier]
                        if TARGET_SUCCESS_DIRECTIONS[label] == "disclosure"
                        else not target["gold_disclosure"][tier]
                    )
                )
                for tier in matcher.TIER_IDS
            }
            if target["directional_correctness"] != expected_correctness:
                raise V3SummaryError("target directional correctness differs")
        for label in RETAINED_CONDITION_LABELS:
            retained = row["retained"]["conditions"][label]
            if retained.get("expected_direction") != "disclosure":
                raise V3SummaryError("retained directional expectation differs")
            _validate_tier_map(
                retained["gold_disclosure"],
                tiers=matcher.TIER_IDS,
                name="retained disclosure",
                require_deterministic_union=True,
            )
            _validate_tier_map(
                retained["directional_correctness"],
                tiers=matcher.TIER_IDS,
                name="retained correctness",
            )
            expected_correctness = {
                tier: bool(
                    retained.get("available")
                    and retained["gold_disclosure"][tier]
                )
                for tier in matcher.TIER_IDS
            }
            if retained["directional_correctness"] != expected_correctness:
                raise V3SummaryError("retained directional correctness differs")
        for cell in (
            row["target"]["policy_vs_fresh_rebuild_equality"],
            row["retained"]["policy_vs_present_equality"],
            row["retained"]["policy_vs_fresh_rebuild_equality"],
        ):
            _validate_tier_map(
                cell["tiers"],
                tiers=decoded_v2.NORMALIZATION_IDS,
                name="response equality",
            )
            if not cell.get("available") and any(cell["tiers"].values()):
                raise V3SummaryError(
                    "unavailable response equality cannot be successful"
                )
        suffix_cell = row.get("suffix") or {}
        if suffix_cell.get("stratum") not in {"clean", "contaminated"}:
            raise V3SummaryError("suffix anonymous stratum differs")
        _validate_tier_map(
            suffix_cell.get("tiers"),
            tiers=matcher.TIER_IDS,
            name="suffix",
            require_deterministic_union=True,
        )
        if (
            suffix_cell["stratum"] == "contaminated"
        ) is not suffix_cell["tiers"]["deterministic_any"]:
            raise V3SummaryError("suffix stratum/tier binding differs")
        for label in CONDITION_LABELS:
            target_available = bool(
                row["target"]["conditions"][label]["available"]
            )
            retained_available = (
                bool(row["retained"]["conditions"][label]["available"])
                if label in RETAINED_CONDITION_LABELS
                else bool(
                    reproduction["conditions"][label]["retained"]
                )
            )
            expected_complete = target_available and retained_available
            if (
                completion["condition_completion"][label]
                is not expected_complete
            ):
                raise V3SummaryError("condition completion binding differs")
        if completion["history_completed"] is not all(
            completion["condition_completion"].values()
        ):
            raise V3SummaryError("history completion binding differs")
        policy_target = row["target"]["conditions"][
            "exact_decrement_or_refit_policy"
        ]["available"]
        policy_retained = row["retained"]["conditions"][
            "exact_decrement_or_refit_policy"
        ]["available"]
        if (
            completion.get("policy_target_output_available")
            is not policy_target
            or completion.get("policy_retained_output_available")
            is not policy_retained
        ):
            raise V3SummaryError("policy availability binding differs")


def _walk_endpoint_objects(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if value.get("estimand") == (
            "mean_of_three_nested_histories_then_equal_weight_K32_mean"
        ):
            yield value
        for child in value.values():
            yield from _walk_endpoint_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_endpoint_objects(child)


def validate_summary(
    summary: Mapping[str, Any],
    *,
    analysis_lock: Mapping[str, Any] | None = None,
) -> None:
    """Validate source-free structure and recompute every published endpoint."""

    if not isinstance(summary, Mapping):
        raise V3SummaryError("v3 summary must be an object")
    _validate_seal(summary, name="v3 summary")
    assert_source_free(summary)
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "post_hoc",
        "contains_source_text",
        "contains_source_identifiers",
        "contains_model_generated_text",
        "contains_token_arrays",
        "model_or_api_calls_made",
        "bindings",
        "analysis",
        "flow",
        "histories",
        "primary_endpoints",
        "decoded_control_descriptive_subsets",
        "suffix_strata",
        "leakage_recall_instrument",
        "claim_gates",
        "scope_limits",
        "integrity",
    }
    if set(summary) != expected_top:
        raise V3SummaryError("v3 summary top-level fields differ")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != SUMMARY_STATUS
        or summary.get("post_hoc") is not True
        or summary.get("model_or_api_calls_made") != 0
    ):
        raise V3SummaryError("v3 summary status or scope differs")
    bindings = summary.get("bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings)
        != {
            "analysis_lock_integrity_sha256",
            "input_artifacts",
            "checked_validation_equals_fresh_read_only_result",
            "full_source_inventory_validated",
            "all_gold_descriptors_and_hashes_validated",
            "all_canonical_values_require_exact_repeat_equality",
        }
        or any(
            bindings.get(key) is not True
            for key in (
                "checked_validation_equals_fresh_read_only_result",
                "full_source_inventory_validated",
                "all_gold_descriptors_and_hashes_validated",
                "all_canonical_values_require_exact_repeat_equality",
            )
        )
        or not isinstance(bindings.get("input_artifacts"), Mapping)
        or set(bindings["input_artifacts"]) != set(_ARTIFACT_ROLES)
    ):
        raise V3SummaryError("v3 summary input bindings differ")
    _require_sha256(
        bindings.get("analysis_lock_integrity_sha256"),
        name="summary analysis-lock binding",
    )
    for role in _ARTIFACT_ROLES:
        _validate_artifact_descriptor(
            bindings["input_artifacts"][role],
            name=f"summary {role}",
        )
    if analysis_lock is not None:
        validate_analysis_lock(analysis_lock)
        if (
            bindings["analysis_lock_integrity_sha256"]
            != analysis_lock["integrity"]["sha256"]
            or bindings["input_artifacts"] != analysis_lock["artifacts"]
        ):
            raise V3SummaryError("summary differs from supplied analysis lock")
    expected_analysis = {
        "primary_unit": "target_cluster",
        "K_target_clusters": EXPECTED_CLUSTERS,
        "nested_history_count": EXPECTED_HISTORIES,
        "histories_per_cluster": HISTORIES_PER_CLUSTER,
        "history_instances_are_independent_records": False,
        "primary_population": "all 96 frozen histories nested in K=32",
        "failure_value": 0,
        "bootstrap": {
            "method": "percentile_cluster_bootstrap",
            "confidence_level": CONFIDENCE_LEVEL,
            "resampling_unit": "target_cluster",
            "clusters_per_resample": EXPECTED_CLUSTERS,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "prng": "splitmix64",
            "percentile_interpolation": "linear_type_7",
            "histories_travel_with_cluster": True,
            "histories_resampled_within_cluster": False,
        },
    }
    if summary.get("analysis") != expected_analysis:
        raise V3SummaryError("v3 summary analysis contract differs")
    histories = summary.get("histories")
    _validate_history_rows(histories)
    if sum(
        row["completion"]["history_completed"] for row in histories
    ) != EXPECTED_HISTORIES:
        raise V3SummaryError("v3 exact summary completion count differs")
    if summary.get("primary_endpoints") != _primary_endpoints(histories):
        raise V3SummaryError("v3 primary endpoint aggregation differs")
    if summary.get("decoded_control_descriptive_subsets") != (
        _descriptive_control_subsets(histories)
    ):
        raise V3SummaryError("decoded-control descriptive subsets differ")
    if summary.get("suffix_strata") != _suffix_strata(histories):
        raise V3SummaryError("suffix strata aggregation differs")
    expected_recall_strata = []
    for ordinal, label in enumerate(CONDITION_LABELS):
        positives = sum(
            row["target"]["conditions"][label]["gold_disclosure"][
                "deterministic_any"
            ]
            for row in histories
        )
        expected_recall_strata.append(
            {
                "condition_ordinal": ordinal,
                "matcher_positive_histories": positives,
                "matcher_clean_histories": EXPECTED_HISTORIES - positives,
                "history_denominator": EXPECTED_HISTORIES,
                "sample_size": recall.SAMPLE_PER_CONDITION,
                "reconciled": True,
            }
        )
    if summary.get("leakage_recall_instrument") != {
        "role": "secondary_instrument_validation_only",
        "may_replace_headline_metrics": False,
        "judge_outcomes_included": False,
        "strata": expected_recall_strata,
    }:
        raise V3SummaryError("leakage-recall stratum reconciliation differs")
    flow = summary.get("flow") or {}
    if flow != {
        "oracle_rows": 500,
        "candidate_targets": 36,
        "eligible_frozen_clusters": EXPECTED_CLUSTERS,
        "histories": EXPECTED_HISTORIES,
        "attempted_histories": EXPECTED_HISTORIES,
        "terminal_histories": EXPECTED_HISTORIES,
        "completed_histories": EXPECTED_HISTORIES,
        "target_admitted": None,
        "retained_available": None,
        "jointly_admitted": None,
        "admission_status": "pending_continuous_utility",
    }:
        raise V3SummaryError("v3 summary flow differs")
    gate = (summary.get("claim_gates") or {}).get(
        "zero_leak_control_admitted"
    ) or {}
    if (
        gate
        != {
            "status": "pending_continuous_utility",
            "gate_open": False,
            "requires_every_future_control_admitted_history": True,
            "requires_zero_deterministic_any_target_leaks": True,
            "requires_zero_unavailable_policy_outputs": True,
            "intent_to_treat_leakage_reported_separately": True,
        }
    ):
        raise V3SummaryError("zero-leak claim gate differs")
    if summary.get("scope_limits") != {
        "semantic_equivalence_endpoint": (
            "future teacher-forced continuation KL"
        ),
        "retained_endpoints": (
            "future teacher-forced forced-choice and gold rank"
        ),
        "decoded_matcher_is_secondary": True,
        "semantic_judge_may_replace_headline_metrics": False,
        "continuous_utility_available": False,
    }:
        raise V3SummaryError("v3 summary scope limits differ")
    endpoints = list(_walk_endpoint_objects(summary["primary_endpoints"]))
    if not endpoints or any(
        endpoint.get("K") != EXPECTED_CLUSTERS
        or endpoint.get("nested_history_count") != EXPECTED_HISTORIES
        or endpoint.get("histories_per_cluster") != HISTORIES_PER_CLUSTER
        or endpoint.get("denominator_histories") != EXPECTED_HISTORIES
        or (
            endpoint.get("cluster_bootstrap_95_interval") or {}
        ).get("clusters_per_resample")
        != EXPECTED_CLUSTERS
        for endpoint in endpoints
    ):
        raise V3SummaryError("not every primary endpoint uses K=32")


def summarize_paths(
    *,
    data_path: str | Path,
    analysis_lock: Mapping[str, Any] | None = None,
    analysis_lock_path: str | Path | None = None,
    **paths: Any,
) -> dict[str, Any]:
    """Validate exact local inputs and return the complete v3 summary."""

    if analysis_lock is not None and analysis_lock_path is not None:
        raise V3SummaryError(
            "provide an analysis lock value or path, not both"
        )
    inputs = _load_exact_inputs(data_path=data_path, **paths)
    if analysis_lock is None:
        lock_path = (
            DEFAULT_ANALYSIS_LOCK_PATH
            if analysis_lock_path is None
            else analysis_lock_path
        )
        analysis_lock = load_json(lock_path, name="v3 analysis lock")
    validate_analysis_lock(analysis_lock)
    expected_lock = build_analysis_lock(
        artifact_bindings=inputs.artifacts,
        anonymous_histories=_anonymous_bindings(inputs.cohort),
    )
    if dict(analysis_lock) != expected_lock:
        raise V3SummaryError("analysis lock differs from exact summary inputs")
    return build_summary(
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
        analysis_lock=analysis_lock,
    )


def deterministic_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _write_new(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser()
    if destination.is_symlink() or destination.parent.is_symlink():
        raise V3SummaryError("output path must not use a symbolic link")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    encoded = deterministic_json(value).encode("utf-8")
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
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{destination} already exists; overwrite is forbidden"
            ) from exc
        os.chmod(destination, 0o644)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_analysis_lock(
    path: str | Path,
    lock: Mapping[str, Any],
) -> None:
    validate_analysis_lock(lock)
    _write_new(path, lock)


def write_summary(
    path: str | Path,
    summary: Mapping[str, Any],
    *,
    analysis_lock: Mapping[str, Any] | None = None,
) -> None:
    validate_summary(summary, analysis_lock=analysis_lock)
    _write_new(path, summary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-lock", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="explicit local exact pinned 500-row oracle JSON",
    )
    parser.add_argument("--audit-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL_PATH)
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT_PATH)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS_PATH)
    parser.add_argument(
        "--cluster-lock",
        type=Path,
        default=DEFAULT_CLUSTER_LOCK_PATH,
    )
    parser.add_argument(
        "--response-authorization",
        type=Path,
        default=DEFAULT_RESPONSE_AUTHORIZATION_PATH,
    )
    parser.add_argument("--suffix", type=Path, default=DEFAULT_SUFFIX_PATH)
    parser.add_argument(
        "--sample-lock",
        type=Path,
        default=DEFAULT_SAMPLE_LOCK_PATH,
    )
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument(
        "--analysis-lock",
        type=Path,
        default=DEFAULT_ANALYSIS_LOCK_PATH,
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="new output path; defaults by mode and never overwrites",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    common = {
        "audit_root": args.audit_root,
        "final_path": args.final,
        "validation_path": args.validation,
        "cohort_path": args.cohort,
        "census_path": args.census,
        "cluster_lock_path": args.cluster_lock,
        "response_authorization_path": args.response_authorization,
        "suffix_path": args.suffix,
        "sample_lock_path": args.sample_lock,
        "rubric_path": args.rubric,
    }
    try:
        if args.freeze_lock:
            value = freeze_analysis_lock_paths(
                data_path=args.data_path,
                **common,
            )
            destination = args.out or DEFAULT_ANALYSIS_LOCK_PATH
            write_analysis_lock(destination, value)
        else:
            value = summarize_paths(
                data_path=args.data_path,
                analysis_lock_path=args.analysis_lock,
                **common,
            )
            destination = args.out or DEFAULT_SUMMARY_PATH
            write_summary(destination, value)
        print(f"wrote {destination}")
    except (FileExistsError, OSError, V3SummaryError) as exc:
        parser.error(str(exc))
    return 0


__all__ = [
    "ANALYSIS_LOCK_SCHEMA",
    "ANALYSIS_LOCK_STATUS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITION_LABELS",
    "DEFAULT_ANALYSIS_LOCK_PATH",
    "DEFAULT_SUMMARY_PATH",
    "EXPECTED_CLUSTERS",
    "EXPECTED_HISTORIES",
    "GREEDY_REPEATS",
    "HISTORIES_PER_CLUSTER",
    "SUMMARY_SCHEMA",
    "SUMMARY_STATUS",
    "V3SummaryError",
    "aggregate_binary_endpoint",
    "assert_source_free",
    "build_analysis_lock",
    "build_summary",
    "canonical_json_bytes",
    "canonical_probe_value",
    "cluster_bootstrap_interval",
    "deterministic_json",
    "file_sha256",
    "freeze_analysis_lock_paths",
    "load_json",
    "nonleak_success",
    "payload_sha256",
    "summarize_paths",
    "text_sha256",
    "validate_analysis_lock",
    "validate_analysis_lock_paths",
    "validate_summary",
    "write_analysis_lock",
    "write_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())

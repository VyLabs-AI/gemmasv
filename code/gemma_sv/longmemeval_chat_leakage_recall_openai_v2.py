"""Post-hoc GPT-5.6 Luna census of the v1 matcher-clean complement.

This additive runner leaves the completed v1 audit unchanged.  It freezes the
exact set difference between the 253 matcher-clean population bindings and the
128 v1 sampled bindings, then gives each of the remaining 125 units one
durable Responses API request slot.  Source-bearing material remains in a
single 0700/0600 local ledger.  A started slot is never retried.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from gemma_sv import (
    longmemeval_chat_leakage_recall_judge_protocol_v1 as judge_protocol,
)
from gemma_sv import longmemeval_chat_leakage_recall_openai_v1 as v1_runner


EXTENSION_LOCK_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-extension-lock-v2"
)
AUTHORIZATION_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-authorization-v2"
)
LOCAL_LEDGER_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-local-ledger-v2"
)
RUN_MANIFEST_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-run-v2"
)
STARTED_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-started-v2"
)
TERMINAL_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-terminal-v2"
)
SUMMARY_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-census-summary-v2"
)
SCHEMA_VERSION = 2
EXTENSION_STATUS = "post-hoc-census-extension-frozen"
AUTHORIZATION_STATUS = "frozen-before-first-openai-census-v2-request"

ENDPOINT = v1_runner.ENDPOINT
HTTP_METHOD = "POST"
MODEL = "gpt-5.6-luna"
# The response-format name remains v1 because the rubric and response schema
# are intentionally byte-for-byte the v1 instrument.
FORMAT_NAME = v1_runner.FORMAT_NAME
MAX_OUTPUT_TOKENS = 512
REQUEST_COUNT = 125
EXPECTED_POPULATION = 384
EXPECTED_CLEAN = 253
EXPECTED_V1_SAMPLE = 128
EXPECTED_EXTENSION_BY_CONDITION = (8, 51, 50, 16)
EXPECTED_CLEAN_BY_CONDITION = (40, 83, 82, 48)
EXPECTED_MATCHER_POSITIVE_BY_CONDITION = (56, 13, 14, 48)
CONDITION_LABELS = (
    "present",
    "fresh rebuild",
    "edited policy",
    "prompt-only",
)
POST_HOC_RATIONALE = (
    "eliminate sampling uncertainty by completing the matcher-negative census"
)
ACKNOWLEDGEMENT = (
    "I_ACKNOWLEDGE_OPENAI_SOURCE_BEARING_LEAKAGE_RECALL_CENSUS_V2"
)

INPUT_USD_PER_MILLION_TOKENS = 0.20
OUTPUT_USD_PER_MILLION_TOKENS = 1.20
PRICE_SOURCE = v1_runner.PRICE_SOURCE
BOOTSTRAP_RESAMPLES = v1_runner.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = 2026082701
CONFIDENCE_LEVEL = 0.95

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
RUNNER_PATH = Path(__file__).resolve()
V1_RUNNER_PATH = Path(v1_runner.__file__).resolve()
V1_JUDGE_PROTOCOL_PATH = Path(judge_protocol.__file__).resolve()

_POPULATION_FIELDS = (
    "condition_ordinal",
    "cluster_index",
    "history_index",
    "variant_index",
    "unit_binding_sha256",
    "question",
    "reference",
    "candidate",
    "deterministic_any",
)
_ANALYSIS_OUTCOMES = tuple(judge_protocol.ANALYSIS_OUTCOMES)
_TERMINAL_KINDS = (
    "completed",
    "parse_failure",
    "transport_failure",
    "permanently_indeterminate",
)
_SHA256_CHARS = frozenset("0123456789abcdef")

Transport = Callable[..., Any]


class OpenAILeakageRecallError(ValueError):
    """A v2 census lock, authorization, evidence, or analysis invariant moved."""


@dataclass(frozen=True)
class FrozenLeakageRecallExtension:
    """Public extension lock plus private complement and population values."""

    extension_lock: dict[str, Any]
    selected_units: tuple[dict[str, Any], ...]
    population_units: tuple[dict[str, Any], ...]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_CHARS
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OpenAILeakageRecallError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise OpenAILeakageRecallError(f"non-finite JSON constant {value!r}")


def _loads_json(value: str | bytes, *, name: str) -> Any:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAILeakageRecallError(
            f"{name} is not strict UTF-8 JSON"
        ) from exc


def _load_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    value = _loads_json(Path(path).read_bytes(), name=name)
    if not isinstance(value, dict):
        raise OpenAILeakageRecallError(f"{name} must be a JSON object")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": _payload_sha256(result),
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
        or integrity.get("sha256") != _payload_sha256(body)
    ):
        raise OpenAILeakageRecallError(f"{name} integrity differs")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    name: str,
) -> None:
    wanted = set(expected)
    observed = set(value)
    if observed != wanted:
        raise OpenAILeakageRecallError(
            f"{name} fields differ; "
            f"missing={sorted(wanted - observed)}, "
            f"extra={sorted(observed - wanted)}"
        )


def _walk(value: Any) -> Iterable[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            yield from _walk(child)


def _assert_secret_absent(value: Any, secret: str) -> None:
    if secret and any(
        isinstance(child, str) and secret in child for child in _walk(value)
    ):
        raise OpenAILeakageRecallError(
            "credential material is forbidden in durable evidence"
        )


def _path_without_symlinks(
    path: str | Path,
    *,
    include_leaf: bool = True,
) -> Path:
    candidate = Path(path).expanduser().absolute()
    checked = candidate if include_leaf else candidate.parent
    for component in reversed([checked, *checked.parents]):
        if component.is_symlink():
            raise OpenAILeakageRecallError(
                "artifact path must not traverse a symbolic link"
            )
    return candidate


def _mode(path: str | Path) -> int:
    return stat.S_IMODE(Path(path).stat().st_mode)


def _validate_mode(path: str | Path, *, directory: bool) -> None:
    artifact = _path_without_symlinks(path)
    expected = 0o700 if directory else 0o600
    if (
        (directory and not artifact.is_dir())
        or (not directory and not artifact.is_file())
        or _mode(artifact) != expected
    ):
        kind = "directory" if directory else "file"
        raise OpenAILeakageRecallError(
            f"local {kind} must have mode {expected:04o}"
        )


def deterministic_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _atomic_write_new(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    local_only: bool,
) -> None:
    destination = _path_without_symlinks(path)
    parent = _path_without_symlinks(destination.parent)
    if not parent.exists():
        parent.mkdir(
            parents=True,
            mode=0o700 if local_only else 0o755,
        )
    if not parent.is_dir() or parent.is_symlink():
        raise OpenAILeakageRecallError(
            "artifact parent must be a real directory"
        )
    if local_only:
        os.chmod(parent, 0o700)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600 if local_only else 0o644,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(deterministic_json(value).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{destination} already exists; overwrite is forbidden"
            ) from exc
        os.chmod(destination, 0o600 if local_only else 0o644)
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _repository_path(path: str | Path) -> str:
    artifact = _path_without_symlinks(path)
    try:
        return artifact.relative_to(WORKSPACE.absolute()).as_posix()
    except ValueError as exc:
        raise OpenAILeakageRecallError(
            "public bound artifacts must be inside the repository"
        ) from exc


def _value_binding(value: Mapping[str, Any]) -> dict[str, str]:
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping) or not _is_sha256(
        integrity.get("sha256")
    ):
        raise OpenAILeakageRecallError(
            "bound value has no valid integrity digest"
        )
    return {
        "payload_sha256": _payload_sha256(value),
        "integrity_sha256": str(integrity["sha256"]),
    }


def _artifact_binding(
    path: str | Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _path_without_symlinks(path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    observed = _load_mapping(artifact, name=artifact.name)
    if observed != dict(value):
        raise OpenAILeakageRecallError(
            f"{artifact.name} value differs from the bound file"
        )
    return {
        "repository_path": _repository_path(artifact),
        "file_sha256": _file_sha256(artifact),
        **_value_binding(value),
        "immutable": True,
        "committed_head_required_before_key_or_ledger_read": True,
    }


def _implementation_binding(path: str | Path) -> dict[str, Any]:
    artifact = _path_without_symlinks(path)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return {
        "repository_path": _repository_path(artifact),
        "file_sha256": _file_sha256(artifact),
        "committed_head_required_before_key_or_ledger_read": True,
    }


def _normalize_population(
    population_units: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    normalized = judge_protocol._normalize_units(population_units)
    if (
        len(normalized) != EXPECTED_POPULATION
        or any(set(row) != set(_POPULATION_FIELDS) for row in normalized)
    ):
        raise OpenAILeakageRecallError(
            "v1 local-ledger population projection differs"
        )
    return tuple(
        {
            field: copy.deepcopy(row[field])
            for field in _POPULATION_FIELDS
        }
        for row in normalized
    )


def select_extension_units(
    population_units: Sequence[Mapping[str, Any]],
    v1_sample_lock: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return the exhaustive matcher-clean set difference, with no ranking."""

    # This function deliberately has no outcome, summary, or terminal argument.
    judge_protocol.validate_sample_lock(v1_sample_lock)
    population = _normalize_population(population_units)
    by_binding = {
        row["unit_binding_sha256"]: row for row in population
    }
    if len(by_binding) != EXPECTED_POPULATION:
        raise OpenAILeakageRecallError("population bindings are not unique")

    clean = [row for row in population if not row["deterministic_any"]]
    clean_counts = tuple(
        sum(row["condition_ordinal"] == ordinal for row in clean)
        for ordinal in range(judge_protocol.CONDITION_COUNT)
    )
    if len(clean) != EXPECTED_CLEAN or clean_counts != (
        EXPECTED_CLEAN_BY_CONDITION
    ):
        raise OpenAILeakageRecallError(
            "matcher-clean census must be exactly 40/83/82/48 = 253"
        )

    samples = v1_sample_lock.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_V1_SAMPLE:
        raise OpenAILeakageRecallError(
            "v1 sample lock must contain exactly 128 bindings"
        )
    sampled_bindings: set[str] = set()
    sample_counts = [0] * judge_protocol.CONDITION_COUNT
    for sample in samples:
        binding = sample["unit_binding_sha256"]
        unit = by_binding.get(binding)
        ordinal = sample["condition_ordinal"]
        if (
            unit is None
            or unit["deterministic_any"] is not False
            or unit["condition_ordinal"] != ordinal
            or binding in sampled_bindings
        ):
            raise OpenAILeakageRecallError(
                "v1 sampled binding is not a unique matcher-clean population unit"
            )
        sampled_bindings.add(binding)
        sample_counts[ordinal] += 1
    if tuple(sample_counts) != (32, 32, 32, 32):
        raise OpenAILeakageRecallError(
            "v1 clean sample must contain 32 units per condition"
        )

    # Preserve canonical population order.  Ordering is not selection and no
    # rank, score, v1 label, or v1 outcome is consulted.
    selected: list[dict[str, Any]] = []
    for row in clean:
        if row["unit_binding_sha256"] in sampled_bindings:
            continue
        selected.append(
            {
                **copy.deepcopy(row),
                "extension_index": len(selected),
            }
        )
    extension_counts = tuple(
        sum(row["condition_ordinal"] == ordinal for row in selected)
        for ordinal in range(judge_protocol.CONDITION_COUNT)
    )
    if (
        len(selected) != REQUEST_COUNT
        or extension_counts != EXPECTED_EXTENSION_BY_CONDITION
        or len(sampled_bindings | {
            row["unit_binding_sha256"] for row in selected
        })
        != EXPECTED_CLEAN
    ):
        raise OpenAILeakageRecallError(
            "extension complement must be exactly 8/51/50/16 = 125 "
            "and complete all 253 clean units"
        )
    return tuple(selected)


def _evidence_corpus_binding(
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = []
    for index in sorted(records):
        value = records[index]
        rows.append(
            {
                "index": index,
                **_value_binding(value),
            }
        )
    return {
        "record_count": len(rows),
        "canonical_record_bindings_sha256": _payload_sha256(rows),
    }


def _validate_v1_completed_run(
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
    v1_local_ledger: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
    v1_run_manifest: Mapping[str, Any],
    v1_started: Mapping[int, Mapping[str, Any]],
    v1_terminals: Mapping[int, Mapping[str, Any]],
) -> None:
    v1_runner.validate_local_ledger(
        v1_local_ledger,
        sample_lock=v1_sample_lock,
        rubric=v1_rubric,
    )
    v1_runner.validate_source_free_summary(v1_summary)
    v1_runner.validate_run_manifest(v1_run_manifest)
    expected_indices = set(range(EXPECTED_V1_SAMPLE))
    if (
        set(v1_started) != expected_indices
        or set(v1_terminals) != expected_indices
    ):
        raise OpenAILeakageRecallError(
            "completed v1 evidence must cover all 128 request slots"
        )
    bindings = v1_run_manifest.get("bindings") or {}
    summary_bindings = v1_summary.get("bindings") or {}
    if (
        (bindings.get("sample_lock") or {}).get("integrity_sha256")
        != v1_sample_lock["integrity"]["sha256"]
        or (bindings.get("rubric") or {}).get("integrity_sha256")
        != v1_rubric["integrity"]["sha256"]
        or (bindings.get("local_ledger") or {}).get("integrity_sha256")
        != v1_local_ledger["integrity"]["sha256"]
        or (bindings.get("local_ledger") or {}).get("payload_sha256")
        != _payload_sha256(v1_local_ledger)
        or summary_bindings.get("sample_lock_integrity_sha256")
        != v1_sample_lock["integrity"]["sha256"]
        or summary_bindings.get("rubric_integrity_sha256")
        != v1_rubric["integrity"]["sha256"]
        or summary_bindings.get("local_ledger_integrity_sha256")
        != v1_local_ledger["integrity"]["sha256"]
        or summary_bindings.get("run_manifest_integrity_sha256")
        != v1_run_manifest["integrity"]["sha256"]
        or (v1_summary.get("coverage") or {}).get("sample_units")
        != EXPECTED_V1_SAMPLE
        or (v1_summary.get("coverage") or {}).get("terminal_units")
        != EXPECTED_V1_SAMPLE
        or (v1_summary.get("coverage") or {}).get("missing_unstarted_units")
        != 0
    ):
        raise OpenAILeakageRecallError(
            "v1 summary, run, ledger, sample, or rubric binding differs"
        )
    public = {
        row["sample_index"]: row for row in v1_sample_lock["samples"]
    }
    entries = v1_local_ledger["protocol_ledger"]["entries"]
    for index in range(EXPECTED_V1_SAMPLE):
        binding = public[index]["unit_binding_sha256"]
        v1_runner.validate_started_marker(
            v1_started[index],
            manifest=v1_run_manifest,
            sample_index=index,
            unit_binding_sha256=binding,
        )
        v1_runner.validate_terminal(
            v1_terminals[index],
            manifest=v1_run_manifest,
            started=v1_started[index],
            sample_index=index,
            unit_binding_sha256=binding,
            candidate=entries[index]["candidate_value"],
        )
    terminal_counts = Counter(
        terminal["analysis_outcome"] for terminal in v1_terminals.values()
    )
    summary_counts = (v1_summary.get("coverage") or {}).get(
        "outcome_counts"
    )
    expected_counts = {
        name: terminal_counts[name] for name in _ANALYSIS_OUTCOMES
    }
    usage = v1_runner._actual_usage(v1_terminals)
    if (
        summary_counts != expected_counts
        or v1_summary.get("actual_usage") != usage
        or v1_summary.get("actual_usage_cost") != v1_runner._actual_cost(usage)
    ):
        raise OpenAILeakageRecallError(
            "v1 source-free summary does not aggregate its bound terminals"
        )


def _v1_run_evidence_binding(
    manifest: Mapping[str, Any],
    started: Mapping[int, Mapping[str, Any]],
    terminals: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "run_manifest": _value_binding(manifest),
        "started": _evidence_corpus_binding(started),
        "terminal": _evidence_corpus_binding(terminals),
        "completed_request_slots": EXPECTED_V1_SAMPLE,
    }


def _validate_extension_v1_bindings(
    extension_lock: Mapping[str, Any],
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_local_ledger: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
    v1_run_manifest: Mapping[str, Any],
    v1_started: Mapping[int, Mapping[str, Any]],
    v1_terminals: Mapping[int, Mapping[str, Any]],
) -> None:
    expected = {
        "v1_sample_lock": _value_binding(v1_sample_lock),
        "v1_local_ledger": _value_binding(v1_local_ledger),
        "v1_source_free_summary": _value_binding(v1_summary),
        "v1_run_evidence": _v1_run_evidence_binding(
            v1_run_manifest,
            v1_started,
            v1_terminals,
        ),
    }
    if extension_lock.get("bindings") != expected:
        raise OpenAILeakageRecallError(
            "extension lock does not bind the supplied completed v1 audit"
        )


def build_extension_lock(
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
    v1_local_ledger: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
    v1_run_manifest: Mapping[str, Any],
    v1_started: Mapping[int, Mapping[str, Any]],
    v1_terminals: Mapping[int, Mapping[str, Any]],
) -> FrozenLeakageRecallExtension:
    """Freeze the post-hoc complement and bind, but never use, v1 outcomes."""

    population = _normalize_population(v1_local_ledger["population_units"])
    # Selection is completed before any summary or terminal field is inspected.
    selected = select_extension_units(population, v1_sample_lock)
    _validate_v1_completed_run(
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_local_ledger=v1_local_ledger,
        v1_summary=v1_summary,
        v1_run_manifest=v1_run_manifest,
        v1_started=v1_started,
        v1_terminals=v1_terminals,
    )
    units = [
        {
            "extension_index": row["extension_index"],
            "condition_ordinal": row["condition_ordinal"],
            "unit_binding_sha256": row["unit_binding_sha256"],
        }
        for row in selected
    ]
    strata = [
        {
            "condition_ordinal": ordinal,
            "population_histories": judge_protocol.EXPECTED_HISTORIES,
            "original_deterministic_matcher_positives": (
                EXPECTED_MATCHER_POSITIVE_BY_CONDITION[ordinal]
            ),
            "matcher_clean_population": EXPECTED_CLEAN_BY_CONDITION[ordinal],
            "v1_sampled_clean_units": judge_protocol.SAMPLE_PER_CONDITION,
            "extension_clean_units": EXPECTED_EXTENSION_BY_CONDITION[ordinal],
            "combined_clean_units": EXPECTED_CLEAN_BY_CONDITION[ordinal],
        }
        for ordinal in range(judge_protocol.CONDITION_COUNT)
    ]
    lock = _seal(
        {
            "schema": EXTENSION_LOCK_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": EXTENSION_STATUS,
            "source_free": True,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "contains_judge_outputs": False,
            "contains_source_identifiers": False,
            "instrument_validation_only": True,
            "headline_semantic_scoring": False,
            "post_hoc_after_v1_outcomes": True,
            "rationale": POST_HOC_RATIONALE,
            "selection": {
                "universe": (
                    "all matcher-clean population bindings minus all v1 "
                    "sampled bindings"
                ),
                "algorithm": "exhaustive_set_difference",
                "exhaustive": True,
                "ranking_used": False,
                "replacement": False,
                "v1_labels_inspected": False,
                "v1_outcomes_inspected": False,
                "population_units": EXPECTED_POPULATION,
                "matcher_clean_population": EXPECTED_CLEAN,
                "v1_sampled_clean_units": EXPECTED_V1_SAMPLE,
                "extension_units": REQUEST_COUNT,
                "combined_clean_units": EXPECTED_CLEAN,
            },
            "protocol": {
                "same_exact_v1_prompt_and_rubric": True,
                "rubric_integrity_sha256": v1_rubric["integrity"]["sha256"],
                "prompt_sha256": v1_rubric["prompt"]["prompt_sha256"],
                "response_schema_sha256": v1_rubric[
                    "response_schema_sha256"
                ],
            },
            "bindings": {
                "v1_sample_lock": _value_binding(v1_sample_lock),
                "v1_local_ledger": _value_binding(v1_local_ledger),
                "v1_source_free_summary": _value_binding(v1_summary),
                "v1_run_evidence": _v1_run_evidence_binding(
                    v1_run_manifest,
                    v1_started,
                    v1_terminals,
                ),
            },
            "strata": strata,
            "units": units,
        }
    )
    validate_extension_lock(
        lock,
        population_units=population,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    return FrozenLeakageRecallExtension(
        extension_lock=lock,
        selected_units=selected,
        population_units=population,
    )


def validate_extension_lock(
    extension_lock: Mapping[str, Any],
    *,
    population_units: Sequence[Mapping[str, Any]] | None = None,
    v1_sample_lock: Mapping[str, Any] | None = None,
    v1_rubric: Mapping[str, Any] | None = None,
) -> None:
    _validate_seal(extension_lock, name="extension lock")
    _require_exact_keys(
        extension_lock,
        {
            "schema",
            "schema_version",
            "status",
            "source_free",
            "contains_source_text",
            "contains_model_generated_text",
            "contains_judge_outputs",
            "contains_source_identifiers",
            "instrument_validation_only",
            "headline_semantic_scoring",
            "post_hoc_after_v1_outcomes",
            "rationale",
            "selection",
            "protocol",
            "bindings",
            "strata",
            "units",
            "integrity",
        },
        name="extension lock",
    )
    if (
        extension_lock.get("schema") != EXTENSION_LOCK_SCHEMA
        or extension_lock.get("schema_version") != SCHEMA_VERSION
        or extension_lock.get("status") != EXTENSION_STATUS
        or extension_lock.get("source_free") is not True
        or extension_lock.get("contains_source_text") is not False
        or extension_lock.get("contains_model_generated_text") is not False
        or extension_lock.get("contains_judge_outputs") is not False
        or extension_lock.get("contains_source_identifiers") is not False
        or extension_lock.get("instrument_validation_only") is not True
        or extension_lock.get("headline_semantic_scoring") is not False
        or extension_lock.get("post_hoc_after_v1_outcomes") is not True
        or extension_lock.get("rationale") != POST_HOC_RATIONALE
    ):
        raise OpenAILeakageRecallError(
            "extension lock status, disclosure, or post-hoc statement differs"
        )
    expected_selection = {
        "universe": (
            "all matcher-clean population bindings minus all v1 sampled bindings"
        ),
        "algorithm": "exhaustive_set_difference",
        "exhaustive": True,
        "ranking_used": False,
        "replacement": False,
        "v1_labels_inspected": False,
        "v1_outcomes_inspected": False,
        "population_units": EXPECTED_POPULATION,
        "matcher_clean_population": EXPECTED_CLEAN,
        "v1_sampled_clean_units": EXPECTED_V1_SAMPLE,
        "extension_units": REQUEST_COUNT,
        "combined_clean_units": EXPECTED_CLEAN,
    }
    if extension_lock.get("selection") != expected_selection:
        raise OpenAILeakageRecallError(
            "extension exhaustive-set-difference design differs"
        )
    protocol = extension_lock.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or set(protocol)
        != {
            "same_exact_v1_prompt_and_rubric",
            "rubric_integrity_sha256",
            "prompt_sha256",
            "response_schema_sha256",
        }
        or protocol.get("same_exact_v1_prompt_and_rubric") is not True
        or any(
            not _is_sha256(protocol.get(key))
            for key in (
                "rubric_integrity_sha256",
                "prompt_sha256",
                "response_schema_sha256",
            )
        )
    ):
        raise OpenAILeakageRecallError("extension v1 protocol binding differs")
    bindings = extension_lock.get("bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings)
        != {
            "v1_sample_lock",
            "v1_local_ledger",
            "v1_source_free_summary",
            "v1_run_evidence",
        }
    ):
        raise OpenAILeakageRecallError("extension artifact bindings differ")
    for role in (
        "v1_sample_lock",
        "v1_local_ledger",
        "v1_source_free_summary",
    ):
        binding = bindings[role]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"payload_sha256", "integrity_sha256"}
            or any(not _is_sha256(value) for value in binding.values())
        ):
            raise OpenAILeakageRecallError(
                f"extension {role} binding differs"
            )
    run_binding = bindings["v1_run_evidence"]
    if (
        not isinstance(run_binding, Mapping)
        or set(run_binding)
        != {
            "run_manifest",
            "started",
            "terminal",
            "completed_request_slots",
        }
        or run_binding.get("completed_request_slots") != EXPECTED_V1_SAMPLE
    ):
        raise OpenAILeakageRecallError("v1 run-evidence binding differs")
    for phase in ("started", "terminal"):
        binding = run_binding[phase]
        if (
            not isinstance(binding, Mapping)
            or binding.get("record_count") != EXPECTED_V1_SAMPLE
            or not _is_sha256(
                binding.get("canonical_record_bindings_sha256")
            )
        ):
            raise OpenAILeakageRecallError(
                f"v1 {phase} corpus binding differs"
            )
    manifest_binding = run_binding["run_manifest"]
    if (
        not isinstance(manifest_binding, Mapping)
        or set(manifest_binding)
        != {"payload_sha256", "integrity_sha256"}
        or any(not _is_sha256(value) for value in manifest_binding.values())
    ):
        raise OpenAILeakageRecallError(
            "v1 run-manifest binding differs"
        )
    strata = extension_lock.get("strata")
    units = extension_lock.get("units")
    if (
        not isinstance(strata, list)
        or len(strata) != judge_protocol.CONDITION_COUNT
        or not isinstance(units, list)
        or len(units) != REQUEST_COUNT
    ):
        raise OpenAILeakageRecallError(
            "extension strata or unit count differs"
        )
    for ordinal, row in enumerate(strata):
        expected = {
            "condition_ordinal": ordinal,
            "population_histories": judge_protocol.EXPECTED_HISTORIES,
            "original_deterministic_matcher_positives": (
                EXPECTED_MATCHER_POSITIVE_BY_CONDITION[ordinal]
            ),
            "matcher_clean_population": EXPECTED_CLEAN_BY_CONDITION[ordinal],
            "v1_sampled_clean_units": judge_protocol.SAMPLE_PER_CONDITION,
            "extension_clean_units": EXPECTED_EXTENSION_BY_CONDITION[ordinal],
            "combined_clean_units": EXPECTED_CLEAN_BY_CONDITION[ordinal],
        }
        if row != expected:
            raise OpenAILeakageRecallError(
                "extension condition accounting differs"
            )
    seen: set[str] = set()
    counts = [0] * judge_protocol.CONDITION_COUNT
    for index, row in enumerate(units):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "extension_index",
                "condition_ordinal",
                "unit_binding_sha256",
            }
            or row.get("extension_index") != index
            or type(row.get("condition_ordinal")) is not int
            or not 0 <= row["condition_ordinal"] < len(CONDITION_LABELS)
            or not _is_sha256(row.get("unit_binding_sha256"))
            or row["unit_binding_sha256"] in seen
        ):
            raise OpenAILeakageRecallError(
                "extension anonymous unit binding differs"
            )
        seen.add(row["unit_binding_sha256"])
        counts[row["condition_ordinal"]] += 1
    if tuple(counts) != EXPECTED_EXTENSION_BY_CONDITION:
        raise OpenAILeakageRecallError(
            "extension unit condition counts differ"
        )
    serialized = deterministic_json(extension_lock)
    for forbidden in (
        '"question"',
        '"reference"',
        '"candidate"',
        '"raw_output"',
        '"parsed_response"',
        '"response_id"',
    ):
        if forbidden in serialized:
            raise OpenAILeakageRecallError(
                "extension lock contains source-bearing or judge evidence"
            )
    supplied = (population_units, v1_sample_lock)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            raise OpenAILeakageRecallError(
                "population and v1 sample lock must be validated together"
            )
        assert population_units is not None
        assert v1_sample_lock is not None
        expected_units = [
            {
                "extension_index": row["extension_index"],
                "condition_ordinal": row["condition_ordinal"],
                "unit_binding_sha256": row["unit_binding_sha256"],
            }
            for row in select_extension_units(
                population_units,
                v1_sample_lock,
            )
        ]
        if units != expected_units:
            raise OpenAILeakageRecallError(
                "extension lock is not the exact clean complement"
            )
    if v1_rubric is not None:
        judge_protocol.validate_rubric(v1_rubric)
        if protocol != {
            "same_exact_v1_prompt_and_rubric": True,
            "rubric_integrity_sha256": v1_rubric["integrity"]["sha256"],
            "prompt_sha256": v1_rubric["prompt"]["prompt_sha256"],
            "response_schema_sha256": v1_rubric[
                "response_schema_sha256"
            ],
        }:
            raise OpenAILeakageRecallError(
                "extension lock does not bind the supplied v1 rubric"
            )


def write_extension_lock(
    path: str | Path,
    extension_lock: Mapping[str, Any],
) -> None:
    validate_extension_lock(extension_lock)
    _atomic_write_new(path, extension_lock, local_only=False)


def _load_v1_run_evidence(
    *,
    v1_run_root: str | Path,
    v1_sample_lock: Mapping[str, Any],
    v1_local_ledger: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, dict[str, Any]],
]:
    root = _path_without_symlinks(v1_run_root)
    _validate_mode(root, directory=True)
    _validate_mode(root / "run.json", directory=False)
    manifest = v1_runner._load_mapping(
        root / "run.json",
        name="v1 run manifest",
    )
    v1_runner.validate_run_manifest(manifest)
    started, terminals = v1_runner._scan_evidence(
        root,
        manifest=manifest,
        sample_lock=v1_sample_lock,
        local_ledger=v1_local_ledger,
    )
    return manifest, started, terminals


def freeze_extension_lock(
    *,
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_local_ledger_path: str | Path,
    v1_summary_path: str | Path,
    v1_run_root: str | Path,
    extension_lock_out: str | Path,
) -> FrozenLeakageRecallExtension:
    v1_sample_lock, v1_rubric = v1_runner.load_public_protocol(
        sample_lock_path=v1_sample_lock_path,
        rubric_path=v1_rubric_path,
    )
    v1_local_ledger = v1_runner.validate_local_ledger_file(
        v1_local_ledger_path,
        sample_lock=v1_sample_lock,
        rubric=v1_rubric,
    )
    # Establish the outcome-independent complement before loading result files.
    select_extension_units(
        v1_local_ledger["population_units"],
        v1_sample_lock,
    )
    v1_summary = _load_mapping(
        v1_summary_path,
        name="v1 source-free summary",
    )
    v1_manifest, v1_started, v1_terminals = _load_v1_run_evidence(
        v1_run_root=v1_run_root,
        v1_sample_lock=v1_sample_lock,
        v1_local_ledger=v1_local_ledger,
    )
    frozen = build_extension_lock(
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_local_ledger=v1_local_ledger,
        v1_summary=v1_summary,
        v1_run_manifest=v1_manifest,
        v1_started=v1_started,
        v1_terminals=v1_terminals,
    )
    write_extension_lock(extension_lock_out, frozen.extension_lock)
    return frozen


def build_local_ledger(
    frozen: FrozenLeakageRecallExtension,
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
) -> dict[str, Any]:
    validate_extension_lock(
        frozen.extension_lock,
        population_units=frozen.population_units,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    expected = select_extension_units(
        frozen.population_units,
        v1_sample_lock,
    )
    if tuple(frozen.selected_units) != expected:
        raise OpenAILeakageRecallError(
            "private extension units differ from the public complement"
        )
    entries = []
    for unit in frozen.selected_units:
        entries.append(
            {
                "extension_index": unit["extension_index"],
                "condition_ordinal": unit["condition_ordinal"],
                "cluster_index": unit["cluster_index"],
                "history_index": unit["history_index"],
                "variant_index": unit["variant_index"],
                "unit_binding_sha256": unit["unit_binding_sha256"],
                "question_value": unit["question"],
                "reference_value": unit["reference"],
                "candidate_value": unit["candidate"],
                "request": judge_protocol.render_judge_request(
                    v1_rubric,
                    question=unit["question"],
                    reference=unit["reference"],
                    candidate=unit["candidate"],
                ),
            }
        )
    ledger = _seal(
        {
            "schema": LOCAL_LEDGER_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "prepared-before-first-openai-census-v2-request",
            "local_only": True,
            "source_bearing": True,
            "contains_source_text": True,
            "contains_model_generated_text": True,
            "contains_provider_outputs": False,
            "directory_mode": "0700",
            "file_mode": "0600",
            "no_overwrite": True,
            "extension_lock_integrity_sha256": frozen.extension_lock[
                "integrity"
            ]["sha256"],
            "v1_sample_lock_integrity_sha256": v1_sample_lock["integrity"][
                "sha256"
            ],
            "v1_rubric_integrity_sha256": v1_rubric["integrity"]["sha256"],
            "entries": entries,
            "population_units": [
                {
                    field: copy.deepcopy(row[field])
                    for field in _POPULATION_FIELDS
                }
                for row in frozen.population_units
            ],
        }
    )
    validate_local_ledger(
        ledger,
        extension_lock=frozen.extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    return ledger


def validate_local_ledger(
    ledger: Mapping[str, Any],
    *,
    extension_lock: Mapping[str, Any],
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
) -> None:
    _validate_seal(ledger, name="extension local ledger")
    _require_exact_keys(
        ledger,
        {
            "schema",
            "schema_version",
            "status",
            "local_only",
            "source_bearing",
            "contains_source_text",
            "contains_model_generated_text",
            "contains_provider_outputs",
            "directory_mode",
            "file_mode",
            "no_overwrite",
            "extension_lock_integrity_sha256",
            "v1_sample_lock_integrity_sha256",
            "v1_rubric_integrity_sha256",
            "entries",
            "population_units",
            "integrity",
        },
        name="extension local ledger",
    )
    if (
        ledger.get("schema") != LOCAL_LEDGER_SCHEMA
        or ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("status")
        != "prepared-before-first-openai-census-v2-request"
        or ledger.get("local_only") is not True
        or ledger.get("source_bearing") is not True
        or ledger.get("contains_source_text") is not True
        or ledger.get("contains_model_generated_text") is not True
        or ledger.get("contains_provider_outputs") is not False
        or ledger.get("directory_mode") != "0700"
        or ledger.get("file_mode") != "0600"
        or ledger.get("no_overwrite") is not True
        or ledger.get("extension_lock_integrity_sha256")
        != extension_lock["integrity"]["sha256"]
        or ledger.get("v1_sample_lock_integrity_sha256")
        != v1_sample_lock["integrity"]["sha256"]
        or ledger.get("v1_rubric_integrity_sha256")
        != v1_rubric["integrity"]["sha256"]
    ):
        raise OpenAILeakageRecallError(
            "extension local-ledger contract differs"
        )
    population = ledger.get("population_units")
    entries = ledger.get("entries")
    if not isinstance(population, list) or not isinstance(entries, list):
        raise OpenAILeakageRecallError(
            "extension ledger population or entries are missing"
        )
    normalized = _normalize_population(population)
    validate_extension_lock(
        extension_lock,
        population_units=normalized,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    selected = select_extension_units(normalized, v1_sample_lock)
    if len(entries) != REQUEST_COUNT:
        raise OpenAILeakageRecallError(
            "extension ledger must contain exactly 125 requests"
        )
    for index, (entry, unit) in enumerate(zip(entries, selected)):
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "extension_index",
                "condition_ordinal",
                "cluster_index",
                "history_index",
                "variant_index",
                "unit_binding_sha256",
                "question_value",
                "reference_value",
                "candidate_value",
                "request",
            }
            or entry.get("extension_index") != index
            or entry.get("condition_ordinal") != unit["condition_ordinal"]
            or entry.get("cluster_index") != unit["cluster_index"]
            or entry.get("history_index") != unit["history_index"]
            or entry.get("variant_index") != unit["variant_index"]
            or entry.get("unit_binding_sha256")
            != unit["unit_binding_sha256"]
            or entry.get("question_value") != unit["question"]
            or entry.get("reference_value") != unit["reference"]
            or entry.get("candidate_value") != unit["candidate"]
            or entry.get("request")
            != judge_protocol.render_judge_request(
                v1_rubric,
                question=unit["question"],
                reference=unit["reference"],
                candidate=unit["candidate"],
            )
        ):
            raise OpenAILeakageRecallError(
                "extension local-ledger request differs"
            )


def write_local_ledger(
    path: str | Path,
    ledger: Mapping[str, Any],
    *,
    extension_lock: Mapping[str, Any],
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
) -> None:
    validate_local_ledger(
        ledger,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    _atomic_write_new(path, ledger, local_only=True)


def validate_local_ledger_file(
    path: str | Path,
    *,
    extension_lock: Mapping[str, Any],
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_path = _path_without_symlinks(path)
    _validate_mode(ledger_path.parent, directory=True)
    _validate_mode(ledger_path, directory=False)
    ledger = _load_mapping(ledger_path, name="extension local ledger")
    validate_local_ledger(
        ledger,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    return ledger


def prepare_local_ledger(
    *,
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_local_ledger_path: str | Path,
    extension_lock_path: str | Path,
    ledger_out: str | Path,
) -> dict[str, Any]:
    v1_sample_lock, v1_rubric = v1_runner.load_public_protocol(
        sample_lock_path=v1_sample_lock_path,
        rubric_path=v1_rubric_path,
    )
    v1_local_ledger = v1_runner.validate_local_ledger_file(
        v1_local_ledger_path,
        sample_lock=v1_sample_lock,
        rubric=v1_rubric,
    )
    extension_lock = _load_mapping(
        extension_lock_path,
        name="extension lock",
    )
    if (
        (extension_lock.get("bindings") or {}).get("v1_local_ledger")
        != _value_binding(v1_local_ledger)
    ):
        raise OpenAILeakageRecallError(
            "extension lock does not bind the supplied v1 local ledger"
        )
    population = _normalize_population(v1_local_ledger["population_units"])
    selected = select_extension_units(population, v1_sample_lock)
    frozen = FrozenLeakageRecallExtension(
        extension_lock=extension_lock,
        selected_units=selected,
        population_units=population,
    )
    ledger = build_local_ledger(
        frozen,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    write_local_ledger(
        ledger_out,
        ledger,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    return ledger


def build_responses_payload(
    request: Mapping[str, Any],
    *,
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    """Reuse the committed v1 payload contract exactly."""

    payload = v1_runner.build_responses_payload(request, rubric=rubric)
    if (
        payload.get("model") != MODEL
        or payload.get("reasoning") != {"mode": "standard", "effort": "low"}
        or payload.get("max_output_tokens") != MAX_OUTPUT_TOKENS
        or payload.get("store") is not False
        or "temperature" in payload
        or ((payload.get("text") or {}).get("format") or {}).get("strict")
        is not True
    ):
        raise OpenAILeakageRecallError("Luna request payload differs")
    return payload


def estimate_request_tokens(
    ledger: Mapping[str, Any],
    *,
    extension_lock: Mapping[str, Any],
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
) -> dict[str, Any]:
    validate_local_ledger(
        ledger,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    system_characters = 0
    user_characters = 0
    prompt_utf8_bytes = 0
    request_json_utf8_bytes = 0
    entries = ledger["entries"]
    for entry in entries:
        request = entry["request"]
        payload = build_responses_payload(request, rubric=v1_rubric)
        system = request["system_prompt"]
        user = request["user_prompt"]
        system_characters += len(system)
        user_characters += len(user)
        prompt_utf8_bytes += len(system.encode("utf-8"))
        prompt_utf8_bytes += len(user.encode("utf-8"))
        request_json_utf8_bytes += len(_canonical_json_bytes(payload))
    if len(entries) != REQUEST_COUNT:
        raise OpenAILeakageRecallError(
            "request estimate must cover exactly 125 units"
        )
    return {
        "request_count": REQUEST_COUNT,
        "rendered_prompt_character_counts": {
            "system_total": system_characters,
            "user_total": user_characters,
            "combined_total": system_characters + user_characters,
        },
        "rendered_prompt_utf8_bytes": prompt_utf8_bytes,
        "canonical_request_json_utf8_bytes": request_json_utf8_bytes,
        "approximation": (
            "Conservative upper approximation: count one input token per "
            "UTF-8 byte of each complete canonical request JSON. This is an "
            "estimate, not tokenizer output or provider usage."
        ),
        "estimated_input_tokens": request_json_utf8_bytes,
        "maximum_output_tokens": REQUEST_COUNT * MAX_OUTPUT_TOKENS,
        "actual_api_usage_available_only_after_run": True,
    }


def estimate_request_cost(
    token_estimate: Mapping[str, Any],
) -> dict[str, Any]:
    estimated_input = token_estimate.get("estimated_input_tokens")
    maximum_output = token_estimate.get("maximum_output_tokens")
    if (
        type(estimated_input) is not int
        or estimated_input < 0
        or type(maximum_output) is not int
        or maximum_output < 0
    ):
        raise OpenAILeakageRecallError("token estimate fields differ")
    input_cost = (
        estimated_input * INPUT_USD_PER_MILLION_TOKENS / 1_000_000
    )
    output_cost = (
        maximum_output * OUTPUT_USD_PER_MILLION_TOKENS / 1_000_000
    )
    return {
        "currency": "USD",
        "estimate_only": True,
        "input_usd_per_million_tokens": INPUT_USD_PER_MILLION_TOKENS,
        "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION_TOKENS,
        "price_source": PRICE_SOURCE,
        "estimated_input_cost_usd": input_cost,
        "maximum_output_cost_usd": output_cost,
        "maximum_estimated_total_cost_usd": input_cost + output_cost,
        "actual_usage_and_cost_reported_after_run": True,
    }


def _request_contract(rubric: Mapping[str, Any]) -> dict[str, Any]:
    probe_request = {
        "system_prompt": rubric["prompt"]["system"],
        "user_prompt": rubric["prompt"]["user_template"],
        "response_schema": rubric["response_schema"],
        "generation": judge_protocol.GENERATION_CONTROLS,
    }
    payload = build_responses_payload(probe_request, rubric=rubric)
    contract = {
        "method": HTTP_METHOD,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "top_level_fields": list(payload),
        "input": {
            "type": "message_array",
            "ordered_roles": ["system", "user"],
            "content_type": "string",
            "content_source": "frozen v1 render_judge_request output",
        },
        "reasoning": {"mode": "standard", "effort": "low"},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "text": copy.deepcopy(payload["text"]),
        "response_text_extraction": (
            "output[].content[].type == output_text"
        ),
    }
    contract["sha256"] = _payload_sha256(contract)
    return contract


def _validate_v1_public_chain(
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
    v1_authorization: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
) -> None:
    judge_protocol.validate_sample_lock(v1_sample_lock, rubric=v1_rubric)
    v1_runner.validate_authorization(v1_authorization)
    v1_runner.validate_source_free_summary(v1_summary)
    artifacts = v1_authorization.get("artifacts") or {}
    summary_bindings = v1_summary.get("bindings") or {}
    if (
        (artifacts.get("sample_lock") or {}).get("integrity_sha256")
        != v1_sample_lock["integrity"]["sha256"]
        or (artifacts.get("rubric") or {}).get("integrity_sha256")
        != v1_rubric["integrity"]["sha256"]
        or summary_bindings.get("authorization_integrity_sha256")
        != v1_authorization["integrity"]["sha256"]
        or summary_bindings.get("sample_lock_integrity_sha256")
        != v1_sample_lock["integrity"]["sha256"]
        or summary_bindings.get("rubric_integrity_sha256")
        != v1_rubric["integrity"]["sha256"]
    ):
        raise OpenAILeakageRecallError(
            "v1 public authorization/summary chain differs"
        )


def _validate_authorization_value_bindings(
    authorization: Mapping[str, Any],
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
    extension_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
) -> None:
    artifacts = authorization["artifacts"]
    values = {
        "v1_sample_lock": v1_sample_lock,
        "v1_rubric": v1_rubric,
        "v1_source_free_summary": v1_summary,
        "extension_lock": extension_lock,
    }
    for role, value in values.items():
        binding = artifacts[role]
        expected = _value_binding(value)
        if any(binding.get(key) != digest for key, digest in expected.items()):
            raise OpenAILeakageRecallError(
                f"v2 authorization no longer binds {role}"
            )
    if (
        artifacts["v1_authorization"]["integrity_sha256"]
        != v1_summary["bindings"]["authorization_integrity_sha256"]
        or artifacts["extension_local_ledger_integrity_sha256"]
        != local_ledger["integrity"]["sha256"]
        or artifacts["extension_local_ledger_payload_sha256"]
        != _payload_sha256(local_ledger)
    ):
        raise OpenAILeakageRecallError(
            "v2 authorization value-level artifact chain differs"
        )


def _authorization_body(
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
    v1_authorization: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
    extension_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_authorization_path: str | Path,
    v1_summary_path: str | Path,
    extension_lock_path: str | Path,
) -> dict[str, Any]:
    _validate_v1_public_chain(
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_authorization=v1_authorization,
        v1_summary=v1_summary,
    )
    validate_local_ledger(
        local_ledger,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    token_estimate = estimate_request_tokens(
        local_ledger,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": AUTHORIZATION_STATUS,
        "source_free": True,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "contains_source_identifiers": False,
        "contains_credentials": False,
        "instrument_validation_only": True,
        "headline_semantic_scoring": False,
        "post_hoc_after_v1_outcomes": True,
        "rationale": POST_HOC_RATIONALE,
        "artifacts": {
            "v1_sample_lock": _artifact_binding(
                v1_sample_lock_path,
                v1_sample_lock,
            ),
            "v1_rubric": _artifact_binding(
                v1_rubric_path,
                v1_rubric,
            ),
            "v1_authorization": _artifact_binding(
                v1_authorization_path,
                v1_authorization,
            ),
            "v1_source_free_summary": _artifact_binding(
                v1_summary_path,
                v1_summary,
            ),
            "v1_judge_protocol_implementation": _implementation_binding(
                V1_JUDGE_PROTOCOL_PATH
            ),
            "v1_provider_runner_implementation": _implementation_binding(
                V1_RUNNER_PATH
            ),
            "v2_census_implementation": _implementation_binding(
                RUNNER_PATH
            ),
            "extension_lock": _artifact_binding(
                extension_lock_path,
                extension_lock,
            ),
            "extension_local_ledger_integrity_sha256": local_ledger[
                "integrity"
            ]["sha256"],
            "extension_local_ledger_payload_sha256": _payload_sha256(
                local_ledger
            ),
        },
        "provider": {
            "name": "OpenAI",
            "api": "Responses API",
            "exact_requested_alias": MODEL,
            "model_reference_kind": "moving_alias",
            "dated_snapshot_available": False,
            "immutable_model_revision_claim": False,
            "terminal_provider_returned_model_recorded": True,
            "terminal_provider_returned_model_validated": True,
            "request_contract": _request_contract(v1_rubric),
        },
        "execution": {
            "extension_units": REQUEST_COUNT,
            "requests_per_unit": 1,
            "authorized_requests": REQUEST_COUNT,
            "passes": 1,
            "retries": 0,
            "batch_api_used": False,
            "same_exact_v1_prompt_and_rubric": True,
            "prompt_sha256": v1_rubric["prompt"]["prompt_sha256"],
            "generation_controls": copy.deepcopy(
                judge_protocol.GENERATION_CONTROLS
            ),
            "request_started_marker_precedes_transport": True,
            "started_without_terminal": "permanently_indeterminate",
            "started_without_terminal_may_be_retried": False,
            "terminal_slots_may_be_retried": False,
        },
        "selection": {
            "exhaustive": True,
            "ranking_used": False,
            "replacement": False,
            "v1_labels_or_outcomes_used": False,
            "extension_counts_by_condition": list(
                EXPECTED_EXTENSION_BY_CONDITION
            ),
            "combined_matcher_clean_units": EXPECTED_CLEAN,
        },
        "transfer": {
            "only": [
                "blinded question",
                "blinded reference",
                "blinded candidate",
                "frozen rubric",
                "frozen response schema",
            ],
            "prohibited": [
                "condition",
                "condition ID",
                "sample or unit ID",
                "source or record ID",
                "matcher result",
                "inclusion weight",
                "suffix stratum",
                "source metadata",
                "full history",
            ],
            "condition_identity_transferred": False,
            "identifiers_transferred": False,
            "matcher_result_transferred": False,
            "inclusion_weight_transferred": False,
            "suffix_stratum_transferred": False,
            "source_metadata_transferred": False,
            "full_history_transferred": False,
        },
        "credential_policy": {
            "environment_variable": "OPENAI_API_KEY",
            "environment_only": True,
            "dotenv_files_read": False,
            "credential_may_enter_logs_or_artifacts": False,
            "read_only_after_acknowledgement_and_committed_head_gate": True,
        },
        "retention": {
            "standard_api_implies_zero_retention": False,
            "store": False,
            "statement": (
                "Standard API use does not imply zero retention; store=false "
                "disables later Responses retrieval but does not promise zero "
                "abuse-monitoring retention."
            ),
        },
        "authorization_gate": {
            "exact_acknowledgement": ACKNOWLEDGEMENT,
            "authorization_and_all_bound_public_files_at_committed_head": True,
            "bound_v1_and_v2_implementations_at_committed_head": True,
            "gate_precedes_environment_key_read": True,
            "gate_precedes_local_ledger_read": True,
            "gate_precedes_requests": True,
        },
        "price_assumptions": {
            "source": PRICE_SOURCE,
            "official_gpt_5_6_luna_page": True,
            "estimate_only": True,
            "input_usd_per_million_tokens": (
                INPUT_USD_PER_MILLION_TOKENS
            ),
            "output_usd_per_million_tokens": (
                OUTPUT_USD_PER_MILLION_TOKENS
            ),
            "actual_billing_may_differ": True,
        },
        "estimate_request_tokens": token_estimate,
        "estimate_request_cost": estimate_request_cost(token_estimate),
        "analysis": {
            "design": "matcher-clean census",
            "horvitz_thompson_sampling_weights_used": False,
            "conservative_lower": (
                "original matcher positives plus clean judge leaks"
            ),
            "conservative_upper": (
                "lower plus clean ambiguous and judge failures"
            ),
            "cluster_bootstrap": {
                "method": "percentile_cluster_bootstrap",
                "K": judge_protocol.EXPECTED_CLUSTERS,
                "histories": judge_protocol.EXPECTED_HISTORIES,
                "histories_per_cluster": judge_protocol.HISTORIES_PER_CLUSTER,
                "histories_resampled_within_cluster": False,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "confidence_level": CONFIDENCE_LEVEL,
            },
            "judge_validates_matcher_recall_only": True,
            "assumes_matcher_positives_are_true_leaks": True,
            "validates_precision": False,
        },
        "human_validation": {
            "status": "pending",
            "remains_secondary": True,
            "not_used_in_census_estimate": True,
        },
    }


def build_authorization(
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
    v1_authorization: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
    extension_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_authorization_path: str | Path,
    v1_summary_path: str | Path,
    extension_lock_path: str | Path,
) -> dict[str, Any]:
    authorization = _seal(
        _authorization_body(
            v1_sample_lock=v1_sample_lock,
            v1_rubric=v1_rubric,
            v1_authorization=v1_authorization,
            v1_summary=v1_summary,
            extension_lock=extension_lock,
            local_ledger=local_ledger,
            v1_sample_lock_path=v1_sample_lock_path,
            v1_rubric_path=v1_rubric_path,
            v1_authorization_path=v1_authorization_path,
            v1_summary_path=v1_summary_path,
            extension_lock_path=extension_lock_path,
        )
    )
    validate_authorization(authorization)
    return authorization


def _validate_authorization_static(
    authorization: Mapping[str, Any],
) -> None:
    _validate_seal(authorization, name="v2 OpenAI authorization")
    required = {
        "schema",
        "schema_version",
        "status",
        "source_free",
        "contains_source_text",
        "contains_model_generated_text",
        "contains_source_identifiers",
        "contains_credentials",
        "instrument_validation_only",
        "headline_semantic_scoring",
        "post_hoc_after_v1_outcomes",
        "rationale",
        "artifacts",
        "provider",
        "execution",
        "selection",
        "transfer",
        "credential_policy",
        "retention",
        "authorization_gate",
        "price_assumptions",
        "estimate_request_tokens",
        "estimate_request_cost",
        "analysis",
        "human_validation",
        "integrity",
    }
    _require_exact_keys(
        authorization,
        required,
        name="v2 OpenAI authorization",
    )
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("schema_version") != SCHEMA_VERSION
        or authorization.get("status") != AUTHORIZATION_STATUS
        or authorization.get("source_free") is not True
        or authorization.get("contains_source_text") is not False
        or authorization.get("contains_model_generated_text") is not False
        or authorization.get("contains_source_identifiers") is not False
        or authorization.get("contains_credentials") is not False
        or authorization.get("instrument_validation_only") is not True
        or authorization.get("headline_semantic_scoring") is not False
        or authorization.get("post_hoc_after_v1_outcomes") is not True
        or authorization.get("rationale") != POST_HOC_RATIONALE
    ):
        raise OpenAILeakageRecallError(
            "v2 authorization status or disclosure differs"
        )
    provider = authorization.get("provider") or {}
    request = provider.get("request_contract") or {}
    if (
        provider.get("name") != "OpenAI"
        or provider.get("api") != "Responses API"
        or provider.get("exact_requested_alias") != MODEL
        or provider.get("model_reference_kind") != "moving_alias"
        or provider.get("dated_snapshot_available") is not False
        or provider.get("immutable_model_revision_claim") is not False
        or request.get("method") != HTTP_METHOD
        or request.get("endpoint") != ENDPOINT
        or request.get("model") != MODEL
        or "temperature" in request
        or request.get("reasoning") != {"mode": "standard", "effort": "low"}
        or request.get("max_output_tokens") != MAX_OUTPUT_TOKENS
        or request.get("store") is not False
        or request.get("sha256")
        != _payload_sha256(
            {
                key: copy.deepcopy(value)
                for key, value in request.items()
                if key != "sha256"
            }
        )
        or ((request.get("text") or {}).get("format") or {}).get("name")
        != FORMAT_NAME
        or ((request.get("text") or {}).get("format") or {}).get("type")
        != "json_schema"
        or ((request.get("text") or {}).get("format") or {}).get("strict")
        is not True
        or ((request.get("text") or {}).get("format") or {}).get("schema")
        != judge_protocol.RESPONSE_SCHEMA
        or provider.get("terminal_provider_returned_model_recorded")
        is not True
        or provider.get("terminal_provider_returned_model_validated")
        is not True
    ):
        raise OpenAILeakageRecallError(
            "v2 provider request contract differs"
        )
    execution = authorization.get("execution") or {}
    selection = authorization.get("selection") or {}
    if (
        execution.get("extension_units") != REQUEST_COUNT
        or execution.get("authorized_requests") != REQUEST_COUNT
        or execution.get("requests_per_unit") != 1
        or execution.get("passes") != 1
        or execution.get("retries") != 0
        or execution.get("batch_api_used") is not False
        or execution.get("same_exact_v1_prompt_and_rubric") is not True
        or execution.get("generation_controls")
        != judge_protocol.GENERATION_CONTROLS
        or execution.get("request_started_marker_precedes_transport")
        is not True
        or execution.get("started_without_terminal")
        != "permanently_indeterminate"
        or execution.get("started_without_terminal_may_be_retried")
        is not False
        or execution.get("terminal_slots_may_be_retried") is not False
        or selection.get("exhaustive") is not True
        or selection.get("ranking_used") is not False
        or selection.get("replacement") is not False
        or selection.get("v1_labels_or_outcomes_used") is not False
        or selection.get("extension_counts_by_condition")
        != list(EXPECTED_EXTENSION_BY_CONDITION)
        or selection.get("combined_matcher_clean_units") != EXPECTED_CLEAN
    ):
        raise OpenAILeakageRecallError(
            "v2 one-pass or exhaustive selection contract differs"
        )
    transfer = authorization.get("transfer") or {}
    credentials = authorization.get("credential_policy") or {}
    retention = authorization.get("retention") or {}
    gate = authorization.get("authorization_gate") or {}
    if (
        transfer.get("only")
        != [
            "blinded question",
            "blinded reference",
            "blinded candidate",
            "frozen rubric",
            "frozen response schema",
        ]
        or transfer.get("prohibited")
        != [
            "condition",
            "condition ID",
            "sample or unit ID",
            "source or record ID",
            "matcher result",
            "inclusion weight",
            "suffix stratum",
            "source metadata",
            "full history",
        ]
        or any(
            transfer.get(key) is not False
            for key in (
                "condition_identity_transferred",
                "identifiers_transferred",
                "matcher_result_transferred",
                "inclusion_weight_transferred",
                "suffix_stratum_transferred",
                "source_metadata_transferred",
                "full_history_transferred",
            )
        )
        or credentials.get("environment_variable") != "OPENAI_API_KEY"
        or credentials.get("environment_only") is not True
        or credentials.get("dotenv_files_read") is not False
        or credentials.get("credential_may_enter_logs_or_artifacts")
        is not False
        or retention.get("standard_api_implies_zero_retention") is not False
        or retention.get("store") is not False
        or gate.get("exact_acknowledgement") != ACKNOWLEDGEMENT
        or any(
            gate.get(key) is not True
            for key in (
                "authorization_and_all_bound_public_files_at_committed_head",
                "bound_v1_and_v2_implementations_at_committed_head",
                "gate_precedes_environment_key_read",
                "gate_precedes_local_ledger_read",
                "gate_precedes_requests",
            )
        )
    ):
        raise OpenAILeakageRecallError(
            "v2 transfer, credential, retention, or gate differs"
        )
    prices = authorization.get("price_assumptions") or {}
    analysis = authorization.get("analysis") or {}
    bootstrap = analysis.get("cluster_bootstrap") or {}
    human = authorization.get("human_validation") or {}
    if (
        prices.get("input_usd_per_million_tokens")
        != INPUT_USD_PER_MILLION_TOKENS
        or prices.get("output_usd_per_million_tokens")
        != OUTPUT_USD_PER_MILLION_TOKENS
        or bootstrap.get("K") != judge_protocol.EXPECTED_CLUSTERS
        or bootstrap.get("histories") != judge_protocol.EXPECTED_HISTORIES
        or analysis.get("horvitz_thompson_sampling_weights_used") is not False
        or analysis.get("judge_validates_matcher_recall_only") is not True
        or analysis.get("assumes_matcher_positives_are_true_leaks") is not True
        or analysis.get("validates_precision") is not False
        or human
        != {
            "status": "pending",
            "remains_secondary": True,
            "not_used_in_census_estimate": True,
        }
        or authorization.get("estimate_request_cost")
        != estimate_request_cost(authorization["estimate_request_tokens"])
    ):
        raise OpenAILeakageRecallError(
            "v2 price, analysis, or human-validation contract differs"
        )
    artifacts = authorization.get("artifacts")
    artifact_roles = {
        "v1_sample_lock",
        "v1_rubric",
        "v1_authorization",
        "v1_source_free_summary",
        "v1_judge_protocol_implementation",
        "v1_provider_runner_implementation",
        "v2_census_implementation",
        "extension_lock",
        "extension_local_ledger_integrity_sha256",
        "extension_local_ledger_payload_sha256",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_roles:
        raise OpenAILeakageRecallError(
            "v2 authorization artifact roles differ"
        )
    for role in (
        "v1_sample_lock",
        "v1_rubric",
        "v1_authorization",
        "v1_source_free_summary",
        "extension_lock",
    ):
        binding = artifacts[role]
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "repository_path",
                "file_sha256",
                "payload_sha256",
                "integrity_sha256",
                "immutable",
                "committed_head_required_before_key_or_ledger_read",
            }
            or binding.get("immutable") is not True
            or binding.get(
                "committed_head_required_before_key_or_ledger_read"
            )
            is not True
            or not isinstance(binding.get("repository_path"), str)
            or any(
                not _is_sha256(binding.get(key))
                for key in (
                    "file_sha256",
                    "payload_sha256",
                    "integrity_sha256",
                )
            )
        ):
            raise OpenAILeakageRecallError(
                f"v2 {role} artifact binding differs"
            )
    for role in (
        "v1_judge_protocol_implementation",
        "v1_provider_runner_implementation",
        "v2_census_implementation",
    ):
        binding = artifacts[role]
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "repository_path",
                "file_sha256",
                "committed_head_required_before_key_or_ledger_read",
            }
            or not isinstance(binding.get("repository_path"), str)
            or not _is_sha256(binding.get("file_sha256"))
            or binding.get(
                "committed_head_required_before_key_or_ledger_read"
            )
            is not True
        ):
            raise OpenAILeakageRecallError(
                f"v2 {role} implementation binding differs"
            )
    if (
        not _is_sha256(
            artifacts.get("extension_local_ledger_integrity_sha256")
        )
        or not _is_sha256(
            artifacts.get("extension_local_ledger_payload_sha256")
        )
    ):
        raise OpenAILeakageRecallError(
            "v2 local-ledger hashes differ"
        )


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    v1_sample_lock: Mapping[str, Any] | None = None,
    v1_rubric: Mapping[str, Any] | None = None,
    v1_authorization: Mapping[str, Any] | None = None,
    v1_summary: Mapping[str, Any] | None = None,
    extension_lock: Mapping[str, Any] | None = None,
    local_ledger: Mapping[str, Any] | None = None,
    v1_sample_lock_path: str | Path | None = None,
    v1_rubric_path: str | Path | None = None,
    v1_authorization_path: str | Path | None = None,
    v1_summary_path: str | Path | None = None,
    extension_lock_path: str | Path | None = None,
) -> None:
    _validate_authorization_static(authorization)
    supplied = (
        v1_sample_lock,
        v1_rubric,
        v1_authorization,
        v1_summary,
        extension_lock,
        local_ledger,
        v1_sample_lock_path,
        v1_rubric_path,
        v1_authorization_path,
        v1_summary_path,
        extension_lock_path,
    )
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            raise OpenAILeakageRecallError(
                "full v2 authorization validation inputs are required"
            )
        expected = _seal(
            _authorization_body(
                v1_sample_lock=v1_sample_lock,  # type: ignore[arg-type]
                v1_rubric=v1_rubric,  # type: ignore[arg-type]
                v1_authorization=v1_authorization,  # type: ignore[arg-type]
                v1_summary=v1_summary,  # type: ignore[arg-type]
                extension_lock=extension_lock,  # type: ignore[arg-type]
                local_ledger=local_ledger,  # type: ignore[arg-type]
                v1_sample_lock_path=v1_sample_lock_path,  # type: ignore[arg-type]
                v1_rubric_path=v1_rubric_path,  # type: ignore[arg-type]
                v1_authorization_path=v1_authorization_path,  # type: ignore[arg-type]
                v1_summary_path=v1_summary_path,  # type: ignore[arg-type]
                extension_lock_path=extension_lock_path,  # type: ignore[arg-type]
            )
        )
        if dict(authorization) != expected:
            raise OpenAILeakageRecallError(
                "v2 authorization differs from bound artifacts"
            )


def write_authorization(
    path: str | Path,
    authorization: Mapping[str, Any],
) -> None:
    validate_authorization(authorization)
    _atomic_write_new(path, authorization, local_only=False)


def freeze_authorization(
    *,
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_authorization_path: str | Path,
    v1_summary_path: str | Path,
    extension_lock_path: str | Path,
    local_ledger_path: str | Path,
    authorization_out: str | Path,
) -> dict[str, Any]:
    v1_sample_lock, v1_rubric = v1_runner.load_public_protocol(
        sample_lock_path=v1_sample_lock_path,
        rubric_path=v1_rubric_path,
    )
    v1_authorization = _load_mapping(
        v1_authorization_path,
        name="v1 authorization",
    )
    v1_summary = _load_mapping(
        v1_summary_path,
        name="v1 source-free summary",
    )
    extension_lock = _load_mapping(
        extension_lock_path,
        name="extension lock",
    )
    local_ledger = validate_local_ledger_file(
        local_ledger_path,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    authorization = build_authorization(
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_authorization=v1_authorization,
        v1_summary=v1_summary,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
        v1_sample_lock_path=v1_sample_lock_path,
        v1_rubric_path=v1_rubric_path,
        v1_authorization_path=v1_authorization_path,
        v1_summary_path=v1_summary_path,
        extension_lock_path=extension_lock_path,
    )
    write_authorization(authorization_out, authorization)
    return authorization


def _require_head_committed_file(path: str | Path) -> None:
    artifact = _path_without_symlinks(path)
    try:
        relative = artifact.relative_to(WORKSPACE.absolute()).as_posix()
    except ValueError as exc:
        raise PermissionError(
            "authorization-bound file is outside the repository"
        ) from exc
    completed = subprocess.run(
        ["git", "-C", str(WORKSPACE), "show", f"HEAD:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise PermissionError(f"{relative} must be committed at HEAD")
    if not artifact.is_file() or completed.stdout != artifact.read_bytes():
        raise PermissionError(f"{relative} differs from committed HEAD")


def _check_bound_committed_file(
    authorization: Mapping[str, Any],
    *,
    role: str,
    path: str | Path,
) -> Path:
    binding = authorization["artifacts"][role]
    artifact = _path_without_symlinks(path)
    if _repository_path(artifact) != binding["repository_path"]:
        raise PermissionError(f"{role} path differs from authorization")
    _require_head_committed_file(artifact)
    if (
        not artifact.is_file()
        or _mode(artifact) != 0o644
        or _file_sha256(artifact) != binding["file_sha256"]
    ):
        raise PermissionError(f"{role} differs from authorization hash")
    return artifact


def _authorize_before_private_read(
    *,
    acknowledgement: str,
    authorization_path: str | Path,
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_authorization_path: str | Path,
    v1_summary_path: str | Path,
    extension_lock_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Path],
]:
    if acknowledgement != ACKNOWLEDGEMENT:
        raise PermissionError(
            "exact OpenAI source-bearing census-v2 acknowledgement is required"
        )
    authorization_file = _path_without_symlinks(authorization_path)
    # This is intentionally the first file access after acknowledgement.
    _require_head_committed_file(authorization_file)
    if (
        not authorization_file.is_file()
        or _mode(authorization_file) != 0o644
    ):
        raise PermissionError(
            "committed v2 authorization must be a regular 0644 file"
        )
    authorization = _load_mapping(
        authorization_file,
        name="v2 OpenAI authorization",
    )
    _validate_authorization_static(authorization)
    provided = {
        "v1_sample_lock": v1_sample_lock_path,
        "v1_rubric": v1_rubric_path,
        "v1_authorization": v1_authorization_path,
        "v1_source_free_summary": v1_summary_path,
        "extension_lock": extension_lock_path,
    }
    checked = {
        role: _check_bound_committed_file(
            authorization,
            role=role,
            path=path,
        )
        for role, path in provided.items()
    }
    implementations = {
        "v1_judge_protocol_implementation": V1_JUDGE_PROTOCOL_PATH,
        "v1_provider_runner_implementation": V1_RUNNER_PATH,
        "v2_census_implementation": RUNNER_PATH,
    }
    for role, path in implementations.items():
        binding = authorization["artifacts"][role]
        artifact = _path_without_symlinks(path)
        if _repository_path(artifact) != binding["repository_path"]:
            raise PermissionError(
                f"{role} path differs from authorization"
            )
        _require_head_committed_file(artifact)
        if _file_sha256(artifact) != binding["file_sha256"]:
            raise PermissionError(
                f"{role} differs from authorization hash"
            )
    return authorization, checked


def _run_manifest_body(
    *,
    authorization_path: str | Path,
    authorization: Mapping[str, Any],
    extension_lock_path: str | Path,
    extension_lock: Mapping[str, Any],
    local_ledger_path: str | Path,
    local_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "one-pass-census-v2-run-initialized",
        "local_only": True,
        "source_bearing_evidence_may_follow": True,
        "directory_mode": "0700",
        "file_mode": "0600",
        "request_slots": REQUEST_COUNT,
        "passes": 1,
        "retries": 0,
        "batch_api_used": False,
        "separate_from_v1_run": True,
        "bindings": {
            "authorization": {
                "file_sha256": _file_sha256(authorization_path),
                "integrity_sha256": authorization["integrity"]["sha256"],
            },
            "extension_lock": {
                "file_sha256": _file_sha256(extension_lock_path),
                "integrity_sha256": extension_lock["integrity"]["sha256"],
            },
            "local_ledger": {
                "file_sha256": _file_sha256(local_ledger_path),
                "payload_sha256": _payload_sha256(local_ledger),
                "integrity_sha256": local_ledger["integrity"]["sha256"],
            },
        },
    }


def build_run_manifest(**kwargs: Any) -> dict[str, Any]:
    manifest = _seal(_run_manifest_body(**kwargs))
    validate_run_manifest(manifest, expected=manifest)
    return manifest


def validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> None:
    _validate_seal(manifest, name="v2 run manifest")
    _require_exact_keys(
        manifest,
        {
            "schema",
            "schema_version",
            "status",
            "local_only",
            "source_bearing_evidence_may_follow",
            "directory_mode",
            "file_mode",
            "request_slots",
            "passes",
            "retries",
            "batch_api_used",
            "separate_from_v1_run",
            "bindings",
            "integrity",
        },
        name="v2 run manifest",
    )
    if (
        manifest.get("schema") != RUN_MANIFEST_SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status")
        != "one-pass-census-v2-run-initialized"
        or manifest.get("local_only") is not True
        or manifest.get("source_bearing_evidence_may_follow") is not True
        or manifest.get("directory_mode") != "0700"
        or manifest.get("file_mode") != "0600"
        or manifest.get("request_slots") != REQUEST_COUNT
        or manifest.get("passes") != 1
        or manifest.get("retries") != 0
        or manifest.get("batch_api_used") is not False
        or manifest.get("separate_from_v1_run") is not True
    ):
        raise OpenAILeakageRecallError("v2 run-manifest contract differs")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "authorization",
        "extension_lock",
        "local_ledger",
    }:
        raise OpenAILeakageRecallError(
            "v2 run-manifest bindings differ"
        )
    if any(
        not isinstance(binding, Mapping)
        or any(not _is_sha256(value) for value in binding.values())
        for binding in bindings.values()
    ):
        raise OpenAILeakageRecallError(
            "v2 run-manifest binding hash differs"
        )
    if expected is not None and dict(manifest) != dict(expected):
        raise OpenAILeakageRecallError(
            "v2 run manifest differs from bound artifacts"
        )


def _initialize_or_validate_run_root(
    root: str | Path,
    *,
    expected_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output_root = _path_without_symlinks(root)
    if output_root.exists():
        _validate_mode(output_root, directory=True)
        _validate_mode(output_root / "run.json", directory=False)
        manifest = _load_mapping(
            output_root / "run.json",
            name="v2 run manifest",
        )
        validate_run_manifest(manifest, expected=expected_manifest)
    else:
        output_root.mkdir(mode=0o700)
        os.chmod(output_root, 0o700)
        manifest = copy.deepcopy(dict(expected_manifest))
        _atomic_write_new(
            output_root / "run.json",
            manifest,
            local_only=True,
        )
    for name in ("started", "terminal"):
        directory = output_root / name
        if directory.exists():
            _validate_mode(directory, directory=True)
        else:
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
    if any(
        path.name not in {"run.json", "started", "terminal"}
        for path in output_root.iterdir()
    ):
        raise OpenAILeakageRecallError(
            "v2 run root contains an unrecognized entry"
        )
    return output_root, manifest


def _marker_bindings(
    *,
    manifest: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "extension_index": extension_index,
        "unit_binding_sha256": unit_binding_sha256,
        "run_manifest_integrity_sha256": manifest["integrity"]["sha256"],
        "authorization_integrity_sha256": manifest["bindings"][
            "authorization"
        ]["integrity_sha256"],
        "extension_lock_integrity_sha256": manifest["bindings"][
            "extension_lock"
        ]["integrity_sha256"],
        "local_ledger_integrity_sha256": manifest["bindings"][
            "local_ledger"
        ]["integrity_sha256"],
    }


def build_started_marker(
    *,
    manifest: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": STARTED_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "phase": "started",
            **_marker_bindings(
                manifest=manifest,
                extension_index=extension_index,
                unit_binding_sha256=unit_binding_sha256,
            ),
            "request_slot_consumed": True,
            "retry_allowed": False,
        }
    )


def validate_started_marker(
    marker: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
) -> None:
    _validate_seal(marker, name="v2 started marker")
    expected = build_started_marker(
        manifest=manifest,
        extension_index=extension_index,
        unit_binding_sha256=unit_binding_sha256,
    )
    if dict(marker) != expected:
        raise OpenAILeakageRecallError(
            "v2 started-marker binding differs"
        )


def _terminal_base(
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "phase": "terminal",
        **_marker_bindings(
            manifest=manifest,
            extension_index=extension_index,
            unit_binding_sha256=unit_binding_sha256,
        ),
        "started_integrity_sha256": started["integrity"]["sha256"],
        "request_slot_consumed": True,
        "retry_allowed": False,
    }


def build_response_terminal(
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
    response: Mapping[str, Any],
    candidate: str,
) -> dict[str, Any]:
    projection, parsed, outcome, failure = v1_runner._evaluate_response(
        response,
        candidate=candidate,
    )
    return _seal(
        {
            **_terminal_base(
                manifest=manifest,
                started=started,
                extension_index=extension_index,
                unit_binding_sha256=unit_binding_sha256,
            ),
            "terminal_kind": (
                "completed" if parsed is not None else "parse_failure"
            ),
            "response": projection,
            "parsed_response": parsed,
            "analysis_outcome": outcome,
            "failure": failure,
        }
    )


def build_transport_failure_terminal(
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    return _seal(
        {
            **_terminal_base(
                manifest=manifest,
                started=started,
                extension_index=extension_index,
                unit_binding_sha256=unit_binding_sha256,
            ),
            "terminal_kind": "transport_failure",
            "response": None,
            "parsed_response": None,
            "analysis_outcome": "transport_error",
            "failure": v1_runner._error_record("transport", error),
        }
    )


def build_indeterminate_terminal(
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    error = RuntimeError(
        "a prior census-v2 started request has no durable terminal record"
    )
    return _seal(
        {
            **_terminal_base(
                manifest=manifest,
                started=started,
                extension_index=extension_index,
                unit_binding_sha256=unit_binding_sha256,
            ),
            "terminal_kind": "permanently_indeterminate",
            "response": None,
            "parsed_response": None,
            "analysis_outcome": "transport_error",
            "failure": v1_runner._error_record(
                "interrupted_after_start",
                error,
            ),
        }
    )


def validate_terminal(
    terminal: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
    candidate: str,
) -> None:
    _validate_seal(terminal, name="v2 terminal")
    expected_keys = {
        "schema",
        "schema_version",
        "phase",
        "extension_index",
        "unit_binding_sha256",
        "run_manifest_integrity_sha256",
        "authorization_integrity_sha256",
        "extension_lock_integrity_sha256",
        "local_ledger_integrity_sha256",
        "started_integrity_sha256",
        "request_slot_consumed",
        "retry_allowed",
        "terminal_kind",
        "response",
        "parsed_response",
        "analysis_outcome",
        "failure",
        "integrity",
    }
    _require_exact_keys(terminal, expected_keys, name="v2 terminal")
    for key, value in _terminal_base(
        manifest=manifest,
        started=started,
        extension_index=extension_index,
        unit_binding_sha256=unit_binding_sha256,
    ).items():
        if terminal.get(key) != value:
            raise OpenAILeakageRecallError("v2 terminal binding differs")
    kind = terminal.get("terminal_kind")
    outcome = terminal.get("analysis_outcome")
    response = terminal.get("response")
    parsed = terminal.get("parsed_response")
    failure = terminal.get("failure")
    if kind not in _TERMINAL_KINDS or outcome not in _ANALYSIS_OUTCOMES:
        raise OpenAILeakageRecallError(
            "v2 terminal kind or outcome differs"
        )
    if kind in {"transport_failure", "permanently_indeterminate"}:
        expected_failure = (
            "transport"
            if kind == "transport_failure"
            else "interrupted_after_start"
        )
        if (
            response is not None
            or parsed is not None
            or outcome != "transport_error"
            or not isinstance(failure, Mapping)
            or failure.get("kind") != expected_failure
            or not _is_sha256(failure.get("error_sha256"))
            or failure.get("message_redacted") is not True
        ):
            raise OpenAILeakageRecallError(
                "v2 non-response terminal differs"
            )
        return
    if not isinstance(response, Mapping) or set(response) != {
        "id",
        "model",
        "status",
        "usage",
        "raw_output",
    }:
        raise OpenAILeakageRecallError(
            "v2 response terminal lacks strict metadata"
        )
    valid_metadata = (
        isinstance(response.get("id"), str)
        and bool(response["id"])
        and response.get("model") == MODEL
        and response.get("status") == "completed"
        and v1_runner._normalize_usage(response.get("usage"))
        == response.get("usage")
        and isinstance(response.get("raw_output"), str)
    )
    if kind == "completed":
        if not valid_metadata or not isinstance(parsed, Mapping) or failure is not None:
            raise OpenAILeakageRecallError(
                "v2 completed terminal metadata differs"
            )
        expected_parsed = judge_protocol.validate_judge_response(
            response["raw_output"],
            candidate=candidate,
        )
        if dict(parsed) != expected_parsed or outcome != expected_parsed["label"]:
            raise OpenAILeakageRecallError(
                "v2 completed terminal response differs"
            )
        return
    parse_failed = not valid_metadata
    if not parse_failed:
        try:
            judge_protocol.validate_judge_response(
                response["raw_output"],
                candidate=candidate,
            )
        except judge_protocol.LeakageRecallProtocolError:
            parse_failed = True
    if (
        not parse_failed
        or parsed is not None
        or outcome != "parse_error"
        or not isinstance(failure, Mapping)
        or failure.get("kind") != "parse"
        or not _is_sha256(failure.get("error_sha256"))
        or failure.get("message_redacted") is not True
    ):
        raise OpenAILeakageRecallError(
            "v2 parse-failure terminal differs"
        )


def _evidence_file(root: Path, phase: str, index: int) -> Path:
    return root / phase / f"{index:03d}.json"


def _scan_evidence(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    extension_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    _validate_mode(root, directory=True)
    allowed = {"run.json", "started", "terminal"}
    if any(
        path.is_symlink()
        or path.name not in allowed
        or (
            path.name == "run.json"
            and (not path.is_file() or _mode(path) != 0o600)
        )
        or (
            path.name in {"started", "terminal"}
            and (not path.is_dir() or _mode(path) != 0o700)
        )
        for path in root.iterdir()
    ):
        raise OpenAILeakageRecallError(
            "v2 run-root entry, type, or permissions differ"
        )
    public = {
        row["extension_index"]: row for row in extension_lock["units"]
    }
    entries = local_ledger["entries"]
    expected_names = {f"{index:03d}.json" for index in range(REQUEST_COUNT)}
    found: dict[str, dict[int, dict[str, Any]]] = {
        "started": {},
        "terminal": {},
    }
    for phase in ("started", "terminal"):
        directory = root / phase
        _validate_mode(directory, directory=True)
        paths = list(directory.iterdir())
        if any(
            path.is_symlink()
            or not path.is_file()
            or path.name not in expected_names
            for path in paths
        ):
            raise OpenAILeakageRecallError(
                f"v2 {phase} evidence has an extra entry"
            )
        for path in paths:
            _validate_mode(path, directory=False)
            index = int(path.stem)
            value = _load_mapping(path, name=f"v2 {phase} marker")
            binding = public[index]["unit_binding_sha256"]
            if phase == "started":
                validate_started_marker(
                    value,
                    manifest=manifest,
                    extension_index=index,
                    unit_binding_sha256=binding,
                )
            else:
                started = found["started"].get(index)
                if started is None:
                    raise OpenAILeakageRecallError(
                        "v2 terminal exists without a started marker"
                    )
                validate_terminal(
                    value,
                    manifest=manifest,
                    started=started,
                    extension_index=index,
                    unit_binding_sha256=binding,
                    candidate=entries[index]["candidate_value"],
                )
            found[phase][index] = value
    if any(index not in found["started"] for index in found["terminal"]):
        raise OpenAILeakageRecallError(
            "v2 terminal evidence has no started marker"
        )
    return found["started"], found["terminal"]


# The network implementation is exactly the committed v1 urllib transport.
urllib_responses_transport = v1_runner.urllib_responses_transport


def _read_openai_key() -> str:
    value = os.environ.get("OPENAI_API_KEY")
    if not isinstance(value, str) or not value.strip():
        raise PermissionError(
            "OPENAI_API_KEY must be present in the process environment"
        )
    return value


def _write_started(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    extension_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    marker = build_started_marker(
        manifest=manifest,
        extension_index=extension_index,
        unit_binding_sha256=unit_binding_sha256,
    )
    _atomic_write_new(
        _evidence_file(root, "started", extension_index),
        marker,
        local_only=True,
    )
    return marker


def _write_terminal(
    root: Path,
    terminal: Mapping[str, Any],
    *,
    secret: str = "",
) -> None:
    _assert_secret_absent(terminal, secret)
    _atomic_write_new(
        _evidence_file(
            root,
            "terminal",
            terminal["extension_index"],
        ),
        terminal,
        local_only=True,
    )


def _finalize_incomplete_starts(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    extension_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    started, terminals = _scan_evidence(
        root,
        manifest=manifest,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
    )
    public = {
        row["extension_index"]: row for row in extension_lock["units"]
    }
    for index in sorted(set(started) - set(terminals)):
        terminal = build_indeterminate_terminal(
            manifest=manifest,
            started=started[index],
            extension_index=index,
            unit_binding_sha256=public[index]["unit_binding_sha256"],
        )
        _write_terminal(root, terminal)
        terminals[index] = terminal
    return started, terminals


def run_one_pass(
    *,
    authorization_path: str | Path,
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_authorization_path: str | Path,
    v1_summary_path: str | Path,
    extension_lock_path: str | Path,
    local_ledger_path: str | Path,
    run_root: str | Path,
    acknowledgement: str,
    transport: Transport | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Consume each of the 125 extension request slots at most once."""

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    authorization, checked = _authorize_before_private_read(
        acknowledgement=acknowledgement,
        authorization_path=authorization_path,
        v1_sample_lock_path=v1_sample_lock_path,
        v1_rubric_path=v1_rubric_path,
        v1_authorization_path=v1_authorization_path,
        v1_summary_path=v1_summary_path,
        extension_lock_path=extension_lock_path,
    )
    # No local source-bearing ledger access occurs before the full commit gate.
    v1_sample_lock, v1_rubric = v1_runner.load_public_protocol(
        sample_lock_path=checked["v1_sample_lock"],
        rubric_path=checked["v1_rubric"],
    )
    v1_authorization = _load_mapping(
        checked["v1_authorization"],
        name="v1 authorization",
    )
    v1_summary = _load_mapping(
        checked["v1_source_free_summary"],
        name="v1 source-free summary",
    )
    extension_lock = _load_mapping(
        checked["extension_lock"],
        name="extension lock",
    )
    local_ledger = validate_local_ledger_file(
        local_ledger_path,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    validate_authorization(
        authorization,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_authorization=v1_authorization,
        v1_summary=v1_summary,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
        v1_sample_lock_path=checked["v1_sample_lock"],
        v1_rubric_path=checked["v1_rubric"],
        v1_authorization_path=checked["v1_authorization"],
        v1_summary_path=checked["v1_source_free_summary"],
        extension_lock_path=checked["extension_lock"],
    )
    manifest = build_run_manifest(
        authorization_path=authorization_path,
        authorization=authorization,
        extension_lock_path=checked["extension_lock"],
        extension_lock=extension_lock,
        local_ledger_path=local_ledger_path,
        local_ledger=local_ledger,
    )
    root, manifest = _initialize_or_validate_run_root(
        run_root,
        expected_manifest=manifest,
    )
    started, terminals = _finalize_incomplete_starts(
        root,
        manifest=manifest,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
    )
    pending = [
        index for index in range(REQUEST_COUNT) if index not in terminals
    ]
    key = _read_openai_key() if pending else None
    selected_transport = transport or urllib_responses_transport
    entries = local_ledger["entries"]
    public = {
        row["extension_index"]: row for row in extension_lock["units"]
    }
    for index in range(REQUEST_COUNT):
        if index in terminals:
            continue
        if index in started:
            raise OpenAILeakageRecallError(
                "incomplete v2 started slot was not made indeterminate"
            )
        binding = public[index]["unit_binding_sha256"]
        marker = _write_started(
            root,
            manifest=manifest,
            extension_index=index,
            unit_binding_sha256=binding,
        )
        started[index] = marker
        assert key is not None
        payload = build_responses_payload(
            entries[index]["request"],
            rubric=v1_rubric,
        )
        try:
            raw_response = selected_transport(
                url=ENDPOINT,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                body=_canonical_json_bytes(payload),
                timeout=float(timeout),
            )
            response = v1_runner._decode_transport_result(raw_response)
            terminal = build_response_terminal(
                manifest=manifest,
                started=marker,
                extension_index=index,
                unit_binding_sha256=binding,
                response=response,
                candidate=entries[index]["candidate_value"],
            )
            _assert_secret_absent(terminal, key)
        except Exception as exc:
            terminal = build_transport_failure_terminal(
                manifest=manifest,
                started=marker,
                extension_index=index,
                unit_binding_sha256=binding,
                error=exc,
            )
        _write_terminal(root, terminal, secret=key)
        terminals[index] = terminal
    started, terminals = _scan_evidence(
        root,
        manifest=manifest,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
    )
    if len(started) != REQUEST_COUNT or len(terminals) != REQUEST_COUNT:
        raise OpenAILeakageRecallError(
            "v2 one-pass run does not cover all 125 request slots"
        )
    return {
        "request_slots": REQUEST_COUNT,
        "started_slots": len(started),
        "terminal_slots": len(terminals),
        "completed_labels": sum(
            row["terminal_kind"] == "completed"
            for row in terminals.values()
        ),
        "parse_failures": sum(
            row["terminal_kind"] == "parse_failure"
            for row in terminals.values()
        ),
        "transport_failures": sum(
            row["terminal_kind"] == "transport_failure"
            for row in terminals.values()
        ),
        "permanently_indeterminate": sum(
            row["terminal_kind"] == "permanently_indeterminate"
            for row in terminals.values()
        ),
        "retries": 0,
    }


def cluster_bootstrap_interval(
    cluster_contributions: Sequence[float],
    *,
    denominator: int,
    resamples: int | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    return v1_runner.cluster_bootstrap_interval(
        cluster_contributions,
        denominator=denominator,
        resamples=BOOTSTRAP_RESAMPLES if resamples is None else resamples,
        seed=seed,
    )


def _actual_usage(
    terminals: Mapping[int, Mapping[str, Any]],
) -> dict[str, int]:
    usage_rows = [
        terminal["response"]["usage"]
        for terminal in terminals.values()
        if isinstance(terminal.get("response"), Mapping)
        and isinstance(terminal["response"].get("usage"), Mapping)
    ]
    return {
        "responses_with_usage": len(usage_rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in usage_rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in usage_rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in usage_rows),
    }


def _actual_cost(usage: Mapping[str, Any]) -> dict[str, Any]:
    input_cost = (
        int(usage["input_tokens"])
        * INPUT_USD_PER_MILLION_TOKENS
        / 1_000_000
    )
    output_cost = (
        int(usage["output_tokens"])
        * OUTPUT_USD_PER_MILLION_TOKENS
        / 1_000_000
    )
    return {
        "currency": "USD",
        "computed_from_actual_api_usage": True,
        "billing_charge_claimed": False,
        "input_usd_per_million_tokens": INPUT_USD_PER_MILLION_TOKENS,
        "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION_TOKENS,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_estimate_usd": input_cost + output_cost,
        "actual_billing_may_differ": True,
    }


def _combined_outcomes(
    *,
    population: Sequence[Mapping[str, Any]],
    v1_sample_lock: Mapping[str, Any],
    v1_terminals: Mapping[int, Mapping[str, Any]],
    extension_lock: Mapping[str, Any],
    v2_terminals: Mapping[int, Mapping[str, Any]],
) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for row in v1_sample_lock["samples"]:
        binding = row["unit_binding_sha256"]
        terminal = v1_terminals[row["sample_index"]]
        outcomes[binding] = terminal["analysis_outcome"]
    for row in extension_lock["units"]:
        binding = row["unit_binding_sha256"]
        if binding in outcomes:
            raise OpenAILeakageRecallError(
                "v1 and v2 clean outcome bindings overlap"
            )
        outcomes[binding] = v2_terminals[row["extension_index"]][
            "analysis_outcome"
        ]
    clean_bindings = {
        row["unit_binding_sha256"]
        for row in population
        if not row["deterministic_any"]
    }
    if (
        set(outcomes) != clean_bindings
        or len(outcomes) != EXPECTED_CLEAN
        or any(outcome not in _ANALYSIS_OUTCOMES for outcome in outcomes.values())
    ):
        raise OpenAILeakageRecallError(
            "combined v1/v2 outcomes do not census all 253 clean units"
        )
    return outcomes


def _condition_census_rows(
    population: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for ordinal, label in enumerate(CONDITION_LABELS):
        condition = [
            row for row in population if row["condition_ordinal"] == ordinal
        ]
        clean = [row for row in condition if not row["deterministic_any"]]
        positive = len(condition) - len(clean)
        counts = Counter(
            outcomes[row["unit_binding_sha256"]] for row in clean
        )
        failures = sum(
            counts[name] for name in ("parse_error", "transport_error", "missing")
        )
        lower = positive + counts["leak"]
        upper = lower + counts["ambiguous"] + failures
        row = {
            "condition_ordinal": ordinal,
            "condition": label,
            "histories": judge_protocol.EXPECTED_HISTORIES,
            "original_deterministic_matcher_positives": positive,
            "matcher_clean_population": len(clean),
            "judged_clean_count": sum(counts.values()),
            "judge_leaks": counts["leak"],
            "judge_no_leak": counts["no_leak"],
            "judge_ambiguous": counts["ambiguous"],
            "judge_failures": failures,
            "judge_failure_breakdown": {
                "parse_error": counts["parse_error"],
                "transport_error": counts["transport_error"],
                "missing": counts["missing"],
            },
            "conservative_corrected_leakage_count_lower": lower,
            "conservative_corrected_leakage_count_upper": upper,
            "conservative_corrected_leakage_rate_lower": (
                lower / judge_protocol.EXPECTED_HISTORIES
            ),
            "conservative_corrected_leakage_rate_upper": (
                upper / judge_protocol.EXPECTED_HISTORIES
            ),
        }
        if (
            positive != EXPECTED_MATCHER_POSITIVE_BY_CONDITION[ordinal]
            or len(clean) != EXPECTED_CLEAN_BY_CONDITION[ordinal]
            or row["judged_clean_count"] != len(clean)
        ):
            raise OpenAILeakageRecallError(
                "combined condition census arithmetic differs"
            )
        rows.append(row)
    return rows


def _cluster_contributions(
    population: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, str],
) -> tuple[list[float], list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    seen_histories: set[int] = set()
    for cluster_index in range(judge_protocol.EXPECTED_CLUSTERS):
        cell = [
            row
            for row in population
            if row["cluster_index"] == cluster_index
        ]
        if len(cell) != (
            judge_protocol.HISTORIES_PER_CLUSTER
            * judge_protocol.CONDITION_COUNT
        ):
            raise OpenAILeakageRecallError(
                "combined cluster geometry differs"
            )
        seen_histories.update(int(row["history_index"]) for row in cell)
        lower_value = 0
        upper_value = 0
        for unit in cell:
            if unit["deterministic_any"]:
                lower_value += 1
                upper_value += 1
                continue
            outcome = outcomes[unit["unit_binding_sha256"]]
            if outcome == "leak":
                lower_value += 1
            if outcome != "no_leak":
                upper_value += 1
        lower.append(float(lower_value))
        upper.append(float(upper_value))
    if seen_histories != set(range(judge_protocol.EXPECTED_HISTORIES)):
        raise OpenAILeakageRecallError(
            "cluster bootstrap does not cover all 96 histories"
        )
    return lower, upper


def build_combined_source_free_summary(
    *,
    v1_sample_lock: Mapping[str, Any],
    v1_rubric: Mapping[str, Any],
    v1_local_ledger: Mapping[str, Any],
    v1_summary: Mapping[str, Any],
    v1_run_manifest: Mapping[str, Any],
    v1_started: Mapping[int, Mapping[str, Any]],
    v1_terminals: Mapping[int, Mapping[str, Any]],
    authorization: Mapping[str, Any],
    extension_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
    v2_run_manifest: Mapping[str, Any],
    v2_started: Mapping[int, Mapping[str, Any]],
    v2_terminals: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one aggregate-only census summary bound to both immutable runs."""

    _validate_v1_completed_run(
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_local_ledger=v1_local_ledger,
        v1_summary=v1_summary,
        v1_run_manifest=v1_run_manifest,
        v1_started=v1_started,
        v1_terminals=v1_terminals,
    )
    _validate_extension_v1_bindings(
        extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_local_ledger=v1_local_ledger,
        v1_summary=v1_summary,
        v1_run_manifest=v1_run_manifest,
        v1_started=v1_started,
        v1_terminals=v1_terminals,
    )
    _validate_authorization_static(authorization)
    _validate_authorization_value_bindings(
        authorization,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_summary=v1_summary,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
    )
    validate_local_ledger(
        local_ledger,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    validate_run_manifest(v2_run_manifest)
    v2_bindings = v2_run_manifest.get("bindings") or {}
    if (
        (v2_bindings.get("authorization") or {}).get(
            "integrity_sha256"
        )
        != authorization["integrity"]["sha256"]
        or (v2_bindings.get("extension_lock") or {}).get(
            "integrity_sha256"
        )
        != extension_lock["integrity"]["sha256"]
        or (v2_bindings.get("local_ledger") or {}).get(
            "integrity_sha256"
        )
        != local_ledger["integrity"]["sha256"]
        or (v2_bindings.get("local_ledger") or {}).get("payload_sha256")
        != _payload_sha256(local_ledger)
    ):
        raise OpenAILeakageRecallError(
            "v2 run manifest does not bind its authorization, lock, and ledger"
        )
    expected_v2 = set(range(REQUEST_COUNT))
    if set(v2_started) != expected_v2 or set(v2_terminals) != expected_v2:
        raise OpenAILeakageRecallError(
            "v2 census summary requires all 125 immutable request slots"
        )
    public_v2 = {
        row["extension_index"]: row for row in extension_lock["units"]
    }
    for index in range(REQUEST_COUNT):
        binding = public_v2[index]["unit_binding_sha256"]
        validate_started_marker(
            v2_started[index],
            manifest=v2_run_manifest,
            extension_index=index,
            unit_binding_sha256=binding,
        )
        validate_terminal(
            v2_terminals[index],
            manifest=v2_run_manifest,
            started=v2_started[index],
            extension_index=index,
            unit_binding_sha256=binding,
            candidate=local_ledger["entries"][index]["candidate_value"],
        )
    population = _normalize_population(local_ledger["population_units"])
    outcomes = _combined_outcomes(
        population=population,
        v1_sample_lock=v1_sample_lock,
        v1_terminals=v1_terminals,
        extension_lock=extension_lock,
        v2_terminals=v2_terminals,
    )
    per_condition = _condition_census_rows(population, outcomes)
    lower_count = sum(
        row["conservative_corrected_leakage_count_lower"]
        for row in per_condition
    )
    upper_count = sum(
        row["conservative_corrected_leakage_count_upper"]
        for row in per_condition
    )
    cluster_lower, cluster_upper = _cluster_contributions(
        population,
        outcomes,
    )
    v1_usage = _actual_usage(v1_terminals)
    v2_usage = _actual_usage(v2_terminals)
    combined_usage = {
        key: v1_usage[key] + v2_usage[key]
        for key in (
            "responses_with_usage",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
    }
    outcome_counts = Counter(outcomes.values())
    summary = _seal(
        {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "source-free-combined-census-summary",
            "source_free": True,
            "contains_source_text": False,
            "contains_model_generated_text": False,
            "contains_source_or_response_ids": False,
            "instrument_validation_only": True,
            "headline_semantic_scoring": False,
            "post_hoc_after_v1_outcomes": True,
            "rationale": POST_HOC_RATIONALE,
            "scope": {
                "judge_validates_matcher_recall_only": True,
                "assumes_matcher_positives_are_true_leaks": True,
                "validates_precision": False,
                "statement": (
                    "The judge validates deterministic-matcher recall on "
                    "matcher-clean outputs only; matcher positives are assumed "
                    "true leaks, so this census does not validate precision."
                ),
            },
            "bindings": {
                "v1_source_free_summary_integrity_sha256": v1_summary[
                    "integrity"
                ]["sha256"],
                "v1_run_manifest_integrity_sha256": v1_run_manifest[
                    "integrity"
                ]["sha256"],
                "v1_started_corpus_sha256": _evidence_corpus_binding(
                    v1_started
                )["canonical_record_bindings_sha256"],
                "v1_terminal_corpus_sha256": _evidence_corpus_binding(
                    v1_terminals
                )["canonical_record_bindings_sha256"],
                "v2_authorization_integrity_sha256": authorization[
                    "integrity"
                ]["sha256"],
                "v2_extension_lock_integrity_sha256": extension_lock[
                    "integrity"
                ]["sha256"],
                "v2_local_ledger_integrity_sha256": local_ledger[
                    "integrity"
                ]["sha256"],
                "v2_run_manifest_integrity_sha256": v2_run_manifest[
                    "integrity"
                ]["sha256"],
                "v2_started_corpus_sha256": _evidence_corpus_binding(
                    v2_started
                )["canonical_record_bindings_sha256"],
                "v2_terminal_corpus_sha256": _evidence_corpus_binding(
                    v2_terminals
                )["canonical_record_bindings_sha256"],
            },
            "coverage": {
                "clusters": judge_protocol.EXPECTED_CLUSTERS,
                "histories": judge_protocol.EXPECTED_HISTORIES,
                "conditions": judge_protocol.CONDITION_COUNT,
                "population_units": EXPECTED_POPULATION,
                "matcher_clean_population": EXPECTED_CLEAN,
                "v1_judged_clean_units": EXPECTED_V1_SAMPLE,
                "v2_judged_clean_units": REQUEST_COUNT,
                "combined_judged_clean_units": EXPECTED_CLEAN,
                "all_matcher_clean_units_covered": True,
                "outcome_counts": {
                    name: outcome_counts[name]
                    for name in _ANALYSIS_OUTCOMES
                },
                "passes_per_unit": 1,
                "retries": 0,
            },
            "census_estimate": {
                "design": "complete matcher-clean census",
                "horvitz_thompson_sampling_weights_used": False,
                "bounds": {
                    "lower": (
                        "original deterministic matcher positives plus "
                        "clean-unit judge leaks"
                    ),
                    "upper": (
                        "lower plus clean-unit ambiguous, parse, transport, "
                        "and missing outcomes"
                    ),
                },
                "per_condition": per_condition,
                "overall": {
                    "population_units": EXPECTED_POPULATION,
                    "original_deterministic_matcher_positives": sum(
                        EXPECTED_MATCHER_POSITIVE_BY_CONDITION
                    ),
                    "matcher_clean_population": EXPECTED_CLEAN,
                    "judged_clean_count": EXPECTED_CLEAN,
                    "judge_leaks": outcome_counts["leak"],
                    "judge_no_leak": outcome_counts["no_leak"],
                    "judge_ambiguous": outcome_counts["ambiguous"],
                    "judge_failures": sum(
                        outcome_counts[name]
                        for name in (
                            "parse_error",
                            "transport_error",
                            "missing",
                        )
                    ),
                    "conservative_corrected_leakage_count_lower": lower_count,
                    "conservative_corrected_leakage_count_upper": upper_count,
                    "conservative_corrected_leakage_rate_lower": (
                        lower_count / EXPECTED_POPULATION
                    ),
                    "conservative_corrected_leakage_rate_upper": (
                        upper_count / EXPECTED_POPULATION
                    ),
                },
                "cluster_bootstrap_95_intervals": {
                    "corrected_leakage_rate_lower": (
                        cluster_bootstrap_interval(
                            cluster_lower,
                            denominator=EXPECTED_POPULATION,
                        )
                    ),
                    "corrected_leakage_rate_upper": (
                        cluster_bootstrap_interval(
                            cluster_upper,
                            denominator=EXPECTED_POPULATION,
                        )
                    ),
                    "all_96_histories_covered": True,
                },
            },
            "actual_usage": {
                "v1": v1_usage,
                "v2_extension": v2_usage,
                "combined": combined_usage,
            },
            "actual_usage_cost": _actual_cost(combined_usage),
            "human_spot_check": {
                "status": "pending",
                "remains_secondary": True,
                "not_used_in_census_estimate": True,
            },
        }
    )
    validate_source_free_summary(summary)
    return summary


# A concise compatibility name for callers that already use the v1 convention.
build_source_free_summary = build_combined_source_free_summary


def validate_source_free_summary(summary: Mapping[str, Any]) -> None:
    _validate_seal(summary, name="combined source-free census summary")
    required = {
        "schema",
        "schema_version",
        "status",
        "source_free",
        "contains_source_text",
        "contains_model_generated_text",
        "contains_source_or_response_ids",
        "instrument_validation_only",
        "headline_semantic_scoring",
        "post_hoc_after_v1_outcomes",
        "rationale",
        "scope",
        "bindings",
        "coverage",
        "census_estimate",
        "actual_usage",
        "actual_usage_cost",
        "human_spot_check",
        "integrity",
    }
    _require_exact_keys(
        summary,
        required,
        name="combined source-free census summary",
    )
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "source-free-combined-census-summary"
        or summary.get("source_free") is not True
        or summary.get("contains_source_text") is not False
        or summary.get("contains_model_generated_text") is not False
        or summary.get("contains_source_or_response_ids") is not False
        or summary.get("instrument_validation_only") is not True
        or summary.get("headline_semantic_scoring") is not False
        or summary.get("post_hoc_after_v1_outcomes") is not True
        or summary.get("rationale") != POST_HOC_RATIONALE
    ):
        raise OpenAILeakageRecallError(
            "combined summary status or disclosure differs"
        )
    serialized = deterministic_json(summary)
    for forbidden in (
        '"raw_output"',
        '"parsed_response"',
        '"response_id"',
        '"unit_binding_sha256"',
        '"question"',
        '"reference"',
        '"candidate"',
        '"full_history"',
        MODEL,
    ):
        if forbidden in serialized:
            raise OpenAILeakageRecallError(
                "combined summary contains prohibited detailed evidence"
            )
    scope = summary.get("scope") or {}
    if (
        scope.get("judge_validates_matcher_recall_only") is not True
        or scope.get("assumes_matcher_positives_are_true_leaks") is not True
        or scope.get("validates_precision") is not False
        or "does not validate precision" not in str(scope.get("statement"))
    ):
        raise OpenAILeakageRecallError(
            "combined summary judge-scope statement differs"
        )
    bindings = summary.get("bindings")
    expected_binding_keys = {
        "v1_source_free_summary_integrity_sha256",
        "v1_run_manifest_integrity_sha256",
        "v1_started_corpus_sha256",
        "v1_terminal_corpus_sha256",
        "v2_authorization_integrity_sha256",
        "v2_extension_lock_integrity_sha256",
        "v2_local_ledger_integrity_sha256",
        "v2_run_manifest_integrity_sha256",
        "v2_started_corpus_sha256",
        "v2_terminal_corpus_sha256",
    }
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != expected_binding_keys
        or any(not _is_sha256(value) for value in bindings.values())
    ):
        raise OpenAILeakageRecallError(
            "combined summary run bindings differ"
        )
    coverage = summary.get("coverage") or {}
    outcomes = coverage.get("outcome_counts") or {}
    if (
        coverage.get("clusters") != judge_protocol.EXPECTED_CLUSTERS
        or coverage.get("histories") != judge_protocol.EXPECTED_HISTORIES
        or coverage.get("conditions") != judge_protocol.CONDITION_COUNT
        or coverage.get("population_units") != EXPECTED_POPULATION
        or coverage.get("matcher_clean_population") != EXPECTED_CLEAN
        or coverage.get("v1_judged_clean_units") != EXPECTED_V1_SAMPLE
        or coverage.get("v2_judged_clean_units") != REQUEST_COUNT
        or coverage.get("combined_judged_clean_units") != EXPECTED_CLEAN
        or coverage.get("all_matcher_clean_units_covered") is not True
        or coverage.get("passes_per_unit") != 1
        or coverage.get("retries") != 0
        or set(outcomes) != set(_ANALYSIS_OUTCOMES)
        or any(type(value) is not int or value < 0 for value in outcomes.values())
        or sum(outcomes.values()) != EXPECTED_CLEAN
    ):
        raise OpenAILeakageRecallError(
            "combined summary census coverage differs"
        )
    estimate = summary.get("census_estimate") or {}
    if (
        estimate.get("design") != "complete matcher-clean census"
        or estimate.get("horvitz_thompson_sampling_weights_used") is not False
    ):
        raise OpenAILeakageRecallError(
            "combined summary census estimator differs"
        )
    per_condition = estimate.get("per_condition")
    if not isinstance(per_condition, list) or len(per_condition) != 4:
        raise OpenAILeakageRecallError(
            "combined per-condition rows differ"
        )
    for ordinal, row in enumerate(per_condition):
        failures = row.get("judge_failure_breakdown") or {}
        lower = row.get("conservative_corrected_leakage_count_lower")
        upper = row.get("conservative_corrected_leakage_count_upper")
        positive = EXPECTED_MATCHER_POSITIVE_BY_CONDITION[ordinal]
        clean = EXPECTED_CLEAN_BY_CONDITION[ordinal]
        if (
            row.get("condition_ordinal") != ordinal
            or row.get("condition") != CONDITION_LABELS[ordinal]
            or row.get("histories") != judge_protocol.EXPECTED_HISTORIES
            or row.get("original_deterministic_matcher_positives") != positive
            or row.get("matcher_clean_population") != clean
            or row.get("judged_clean_count") != clean
            or set(failures) != {"parse_error", "transport_error", "missing"}
            or row.get("judge_failures") != sum(failures.values())
            or sum(
                (
                    row.get("judge_leaks", -1),
                    row.get("judge_no_leak", -1),
                    row.get("judge_ambiguous", -1),
                    row.get("judge_failures", -1),
                )
            )
            != clean
            or lower != positive + row.get("judge_leaks", -1)
            or upper
            != lower + row.get("judge_ambiguous", -1) + row.get(
                "judge_failures",
                -1,
            )
            or row.get("conservative_corrected_leakage_rate_lower")
            != lower / judge_protocol.EXPECTED_HISTORIES
            or row.get("conservative_corrected_leakage_rate_upper")
            != upper / judge_protocol.EXPECTED_HISTORIES
        ):
            raise OpenAILeakageRecallError(
                "combined per-condition census arithmetic differs"
            )
    overall = estimate.get("overall") or {}
    expected_lower = sum(
        row["conservative_corrected_leakage_count_lower"]
        for row in per_condition
    )
    expected_upper = sum(
        row["conservative_corrected_leakage_count_upper"]
        for row in per_condition
    )
    if (
        overall.get("population_units") != EXPECTED_POPULATION
        or overall.get("original_deterministic_matcher_positives")
        != sum(EXPECTED_MATCHER_POSITIVE_BY_CONDITION)
        or overall.get("matcher_clean_population") != EXPECTED_CLEAN
        or overall.get("judged_clean_count") != EXPECTED_CLEAN
        or overall.get("judge_leaks") != outcomes["leak"]
        or overall.get("judge_no_leak") != outcomes["no_leak"]
        or overall.get("judge_ambiguous") != outcomes["ambiguous"]
        or overall.get("judge_failures")
        != outcomes["parse_error"] + outcomes["transport_error"] + outcomes["missing"]
        or overall.get("conservative_corrected_leakage_count_lower")
        != expected_lower
        or overall.get("conservative_corrected_leakage_count_upper")
        != expected_upper
        or overall.get("conservative_corrected_leakage_rate_lower")
        != expected_lower / EXPECTED_POPULATION
        or overall.get("conservative_corrected_leakage_rate_upper")
        != expected_upper / EXPECTED_POPULATION
    ):
        raise OpenAILeakageRecallError(
            "combined overall census arithmetic differs"
        )
    intervals = estimate.get("cluster_bootstrap_95_intervals") or {}
    if (
        intervals.get("all_96_histories_covered") is not True
        or any(
            (intervals.get(name) or {}).get("K")
            != judge_protocol.EXPECTED_CLUSTERS
            for name in (
                "corrected_leakage_rate_lower",
                "corrected_leakage_rate_upper",
            )
        )
    ):
        raise OpenAILeakageRecallError(
            "combined 32-cluster bootstrap differs"
        )
    usage = summary.get("actual_usage") or {}
    combined_usage = usage.get("combined") or {}
    if (
        set(usage) != {"v1", "v2_extension", "combined"}
        or any(
            type(combined_usage.get(key)) is not int
            or combined_usage[key] < 0
            for key in (
                "responses_with_usage",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        )
        or combined_usage.get("total_tokens")
        != combined_usage.get("input_tokens") + combined_usage.get("output_tokens")
        or summary.get("actual_usage_cost") != _actual_cost(combined_usage)
        or summary.get("human_spot_check")
        != {
            "status": "pending",
            "remains_secondary": True,
            "not_used_in_census_estimate": True,
        }
    ):
        raise OpenAILeakageRecallError(
            "combined usage, cost, or human status differs"
        )


def summarize_run(
    *,
    v1_sample_lock_path: str | Path,
    v1_rubric_path: str | Path,
    v1_local_ledger_path: str | Path,
    v1_summary_path: str | Path,
    v1_run_root: str | Path,
    authorization_path: str | Path,
    extension_lock_path: str | Path,
    local_ledger_path: str | Path,
    run_root: str | Path,
    summary_out: str | Path,
) -> dict[str, Any]:
    v1_sample_lock, v1_rubric = v1_runner.load_public_protocol(
        sample_lock_path=v1_sample_lock_path,
        rubric_path=v1_rubric_path,
    )
    v1_local_ledger = v1_runner.validate_local_ledger_file(
        v1_local_ledger_path,
        sample_lock=v1_sample_lock,
        rubric=v1_rubric,
    )
    v1_summary = _load_mapping(
        v1_summary_path,
        name="v1 source-free summary",
    )
    v1_manifest, v1_started, v1_terminals = _load_v1_run_evidence(
        v1_run_root=v1_run_root,
        v1_sample_lock=v1_sample_lock,
        v1_local_ledger=v1_local_ledger,
    )
    authorization = _load_mapping(
        authorization_path,
        name="v2 authorization",
    )
    extension_lock = _load_mapping(
        extension_lock_path,
        name="extension lock",
    )
    local_ledger = validate_local_ledger_file(
        local_ledger_path,
        extension_lock=extension_lock,
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
    )
    expected_manifest = build_run_manifest(
        authorization_path=authorization_path,
        authorization=authorization,
        extension_lock_path=extension_lock_path,
        extension_lock=extension_lock,
        local_ledger_path=local_ledger_path,
        local_ledger=local_ledger,
    )
    root = _path_without_symlinks(run_root)
    _validate_mode(root, directory=True)
    _validate_mode(root / "run.json", directory=False)
    manifest = _load_mapping(root / "run.json", name="v2 run manifest")
    validate_run_manifest(manifest, expected=expected_manifest)
    v2_started, v2_terminals = _finalize_incomplete_starts(
        root,
        manifest=manifest,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
    )
    summary = build_combined_source_free_summary(
        v1_sample_lock=v1_sample_lock,
        v1_rubric=v1_rubric,
        v1_local_ledger=v1_local_ledger,
        v1_summary=v1_summary,
        v1_run_manifest=v1_manifest,
        v1_started=v1_started,
        v1_terminals=v1_terminals,
        authorization=authorization,
        extension_lock=extension_lock,
        local_ledger=local_ledger,
        v2_run_manifest=manifest,
        v2_started=v2_started,
        v2_terminals=v2_terminals,
    )
    _atomic_write_new(summary_out, summary, local_only=False)
    return summary


def _require_arguments(
    args: argparse.Namespace,
    names: Sequence[str],
    *,
    mode: str,
) -> None:
    missing = [
        name.replace("_", "-")
        for name in names
        if getattr(args, name) is None
    ]
    if missing:
        raise OpenAILeakageRecallError(
            f"{mode} requires explicit --" + ", --".join(missing)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-extension-lock", action="store_true")
    mode.add_argument("--prepare-local-ledger", action="store_true")
    mode.add_argument("--freeze-authorization", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--v1-sample-lock", type=Path)
    parser.add_argument("--v1-rubric", type=Path)
    parser.add_argument("--v1-local-ledger", type=Path)
    parser.add_argument("--v1-authorization", type=Path)
    parser.add_argument("--v1-summary", type=Path)
    parser.add_argument("--v1-run-root", type=Path)
    parser.add_argument("--extension-lock", type=Path)
    parser.add_argument("--extension-lock-out", type=Path)
    parser.add_argument("--local-ledger", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-out", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--acknowledgement", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        common_v1 = ("v1_sample_lock", "v1_rubric")
        if args.freeze_extension_lock:
            _require_arguments(
                args,
                (
                    *common_v1,
                    "v1_local_ledger",
                    "v1_summary",
                    "v1_run_root",
                    "extension_lock_out",
                ),
                mode="--freeze-extension-lock",
            )
            freeze_extension_lock(
                v1_sample_lock_path=args.v1_sample_lock,
                v1_rubric_path=args.v1_rubric,
                v1_local_ledger_path=args.v1_local_ledger,
                v1_summary_path=args.v1_summary,
                v1_run_root=args.v1_run_root,
                extension_lock_out=args.extension_lock_out,
            )
        elif args.prepare_local_ledger:
            _require_arguments(
                args,
                (
                    *common_v1,
                    "v1_local_ledger",
                    "extension_lock",
                    "local_ledger",
                ),
                mode="--prepare-local-ledger",
            )
            prepare_local_ledger(
                v1_sample_lock_path=args.v1_sample_lock,
                v1_rubric_path=args.v1_rubric,
                v1_local_ledger_path=args.v1_local_ledger,
                extension_lock_path=args.extension_lock,
                ledger_out=args.local_ledger,
            )
        elif args.freeze_authorization:
            _require_arguments(
                args,
                (
                    *common_v1,
                    "v1_authorization",
                    "v1_summary",
                    "extension_lock",
                    "local_ledger",
                    "authorization_out",
                ),
                mode="--freeze-authorization",
            )
            freeze_authorization(
                v1_sample_lock_path=args.v1_sample_lock,
                v1_rubric_path=args.v1_rubric,
                v1_authorization_path=args.v1_authorization,
                v1_summary_path=args.v1_summary,
                extension_lock_path=args.extension_lock,
                local_ledger_path=args.local_ledger,
                authorization_out=args.authorization_out,
            )
        elif args.run:
            _require_arguments(
                args,
                (
                    *common_v1,
                    "v1_authorization",
                    "v1_summary",
                    "extension_lock",
                    "local_ledger",
                    "authorization",
                    "run_root",
                ),
                mode="--run",
            )
            result = run_one_pass(
                authorization_path=args.authorization,
                v1_sample_lock_path=args.v1_sample_lock,
                v1_rubric_path=args.v1_rubric,
                v1_authorization_path=args.v1_authorization,
                v1_summary_path=args.v1_summary,
                extension_lock_path=args.extension_lock,
                local_ledger_path=args.local_ledger,
                run_root=args.run_root,
                acknowledgement=args.acknowledgement,
                timeout=args.timeout,
            )
            return 0 if result["terminal_slots"] == REQUEST_COUNT else 1
        else:
            _require_arguments(
                args,
                (
                    *common_v1,
                    "v1_local_ledger",
                    "v1_summary",
                    "v1_run_root",
                    "extension_lock",
                    "local_ledger",
                    "authorization",
                    "run_root",
                    "summary_out",
                ),
                mode="--summarize",
            )
            summarize_run(
                v1_sample_lock_path=args.v1_sample_lock,
                v1_rubric_path=args.v1_rubric,
                v1_local_ledger_path=args.v1_local_ledger,
                v1_summary_path=args.v1_summary,
                v1_run_root=args.v1_run_root,
                authorization_path=args.authorization,
                extension_lock_path=args.extension_lock,
                local_ledger_path=args.local_ledger,
                run_root=args.run_root,
                summary_out=args.summary_out,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        OpenAILeakageRecallError,
        PermissionError,
        ValueError,
        judge_protocol.LeakageRecallProtocolError,
        v1_runner.OpenAILeakageRecallError,
    ) as exc:
        parser.error(str(exc))
    return 0


__all__ = [
    "ACKNOWLEDGEMENT",
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_STATUS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITION_LABELS",
    "EXPECTED_CLEAN",
    "EXPECTED_CLEAN_BY_CONDITION",
    "EXPECTED_EXTENSION_BY_CONDITION",
    "EXTENSION_LOCK_SCHEMA",
    "FORMAT_NAME",
    "FrozenLeakageRecallExtension",
    "INPUT_USD_PER_MILLION_TOKENS",
    "LOCAL_LEDGER_SCHEMA",
    "MAX_OUTPUT_TOKENS",
    "MODEL",
    "OUTPUT_USD_PER_MILLION_TOKENS",
    "OpenAILeakageRecallError",
    "POST_HOC_RATIONALE",
    "REQUEST_COUNT",
    "RUN_MANIFEST_SCHEMA",
    "SUMMARY_SCHEMA",
    "build_authorization",
    "build_combined_source_free_summary",
    "build_indeterminate_terminal",
    "build_local_ledger",
    "build_responses_payload",
    "build_response_terminal",
    "build_run_manifest",
    "build_source_free_summary",
    "build_started_marker",
    "build_transport_failure_terminal",
    "cluster_bootstrap_interval",
    "estimate_request_cost",
    "estimate_request_tokens",
    "freeze_authorization",
    "freeze_extension_lock",
    "prepare_local_ledger",
    "run_one_pass",
    "select_extension_units",
    "summarize_run",
    "urllib_responses_transport",
    "validate_authorization",
    "validate_extension_lock",
    "validate_local_ledger",
    "validate_local_ledger_file",
    "validate_run_manifest",
    "validate_source_free_summary",
    "validate_started_marker",
    "validate_terminal",
    "write_authorization",
    "write_extension_lock",
    "write_local_ledger",
]


if __name__ == "__main__":
    raise SystemExit(main())

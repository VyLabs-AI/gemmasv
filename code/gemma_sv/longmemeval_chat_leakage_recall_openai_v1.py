"""Authorize and run the frozen leakage-recall sample with GPT-5.6 Luna.

The public authorization is additive: it does not alter the frozen sample or
rubric.  Source-bearing values stay in a private 0700/0600 ledger and enter a
single Responses API request only after an exact acknowledgement and committed
HEAD checks.  Each sample has one durable request slot.  A started slot without
a terminal record is permanently indeterminate and is never retried.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.request

from gemma_sv import (
    longmemeval_chat_leakage_recall_judge_protocol_v1 as judge_protocol,
)


AUTHORIZATION_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-authorization-v1"
)
LOCAL_LEDGER_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-local-ledger-v1"
)
RUN_MANIFEST_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-run-v1"
)
STARTED_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-started-v1"
)
TERMINAL_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-terminal-v1"
)
SUMMARY_SCHEMA = (
    "gemma-sv-longmemeval-chat-leakage-recall-openai-summary-v1"
)
SCHEMA_VERSION = 1
AUTHORIZATION_STATUS = "frozen-before-first-openai-request"

ENDPOINT = "https://api.openai.com/v1/responses"
HTTP_METHOD = "POST"
MODEL = "gpt-5.6-luna"
FORMAT_NAME = "leakage_recall_v1"
MAX_OUTPUT_TOKENS = 512
REQUEST_COUNT = judge_protocol.EXPECTED_SAMPLE
ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_OPENAI_SOURCE_BEARING_LEAKAGE_RECALL_V1"

INPUT_USD_PER_MILLION_TOKENS = 0.20
OUTPUT_USD_PER_MILLION_TOKENS = 1.20
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 2026082610
CONFIDENCE_LEVEL = 0.95

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
RUNNER_PATH = Path(__file__).resolve()
JUDGE_PROTOCOL_PATH = Path(judge_protocol.__file__).resolve()

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
_TERMINAL_KINDS = (
    "completed",
    "parse_failure",
    "transport_failure",
    "permanently_indeterminate",
)
_ANALYSIS_OUTCOMES = frozenset(judge_protocol.ANALYSIS_OUTCOMES)
_SHA256_CHARS = frozenset("0123456789abcdef")

Transport = Callable[..., Any]


class OpenAILeakageRecallError(ValueError):
    """An authorization, ledger, request, or evidence invariant drifted."""


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


def _assert_json_value(value: Any, *, name: str) -> None:
    try:
        _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise OpenAILeakageRecallError(
            f"{name} is not finite JSON data"
        ) from exc


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
    lineage = [checked, *checked.parents]
    for component in reversed(lineage):
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
    encoded = deterministic_json(value).encode("utf-8")
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
            handle.write(encoded)
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
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping) or not _is_sha256(
        integrity.get("sha256")
    ):
        raise OpenAILeakageRecallError(
            f"{artifact.name} has no valid integrity binding"
        )
    return {
        "repository_path": _repository_path(artifact),
        "file_sha256": _file_sha256(artifact),
        "payload_sha256": _payload_sha256(value),
        "integrity_sha256": integrity["sha256"],
        "immutable": True,
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


def load_public_protocol(
    *,
    sample_lock_path: str | Path,
    rubric_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_lock = _load_mapping(sample_lock_path, name="sample lock")
    rubric = _load_mapping(rubric_path, name="rubric")
    # The frozen protocol's artifact validator also freezes insertion order,
    # while its canonical JSON writer sorts object keys.  Restore that declared
    # order after strict loading so a canonical on-disk lock remains valid.
    artifact_roles = tuple(judge_protocol._ARTIFACT_ROLES)
    for value in (sample_lock, rubric):
        artifacts = value.get("artifacts")
        if isinstance(artifacts, Mapping) and set(artifacts) == set(
            artifact_roles
        ):
            value["artifacts"] = {
                role: artifacts[role] for role in artifact_roles
            }
    judge_protocol.validate_sample_lock(sample_lock, rubric=rubric)
    return sample_lock, rubric


def _population_projection(
    population_units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = judge_protocol._normalize_units(population_units)
    return [
        {field: copy.deepcopy(row[field]) for field in _POPULATION_FIELDS}
        for row in normalized
    ]


def build_local_ledger(
    frozen: judge_protocol.FrozenLeakageRecallProtocol,
) -> dict[str, Any]:
    """Build the private immutable provider ledger without network activity."""

    judge_protocol.validate_sample_lock(
        frozen.sample_lock,
        rubric=frozen.rubric,
    )
    protocol_ledger = judge_protocol.build_local_ledger(frozen)
    value = _seal(
        {
            "schema": LOCAL_LEDGER_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "prepared-before-first-openai-request",
            "local_only": True,
            "source_bearing": True,
            "contains_source_text": True,
            "contains_model_generated_text": True,
            "contains_provider_outputs": False,
            "directory_mode": "0700",
            "file_mode": "0600",
            "no_overwrite": True,
            "sample_lock_integrity_sha256": frozen.sample_lock["integrity"][
                "sha256"
            ],
            "rubric_integrity_sha256": frozen.rubric["integrity"]["sha256"],
            "protocol_ledger": protocol_ledger,
            "population_units": _population_projection(
                frozen.population_units
            ),
        }
    )
    validate_local_ledger(
        value,
        sample_lock=frozen.sample_lock,
        rubric=frozen.rubric,
    )
    return value


def validate_local_ledger(
    ledger: Mapping[str, Any],
    *,
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> None:
    """Validate source values, all 128 requests, and bootstrap population."""

    judge_protocol.validate_sample_lock(sample_lock, rubric=rubric)
    _validate_seal(ledger, name="OpenAI local ledger")
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
            "sample_lock_integrity_sha256",
            "rubric_integrity_sha256",
            "protocol_ledger",
            "population_units",
            "integrity",
        },
        name="OpenAI local ledger",
    )
    if (
        ledger.get("schema") != LOCAL_LEDGER_SCHEMA
        or ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("status")
        != "prepared-before-first-openai-request"
        or ledger.get("local_only") is not True
        or ledger.get("source_bearing") is not True
        or ledger.get("contains_source_text") is not True
        or ledger.get("contains_model_generated_text") is not True
        or ledger.get("contains_provider_outputs") is not False
        or ledger.get("directory_mode") != "0700"
        or ledger.get("file_mode") != "0600"
        or ledger.get("no_overwrite") is not True
        or ledger.get("sample_lock_integrity_sha256")
        != sample_lock["integrity"]["sha256"]
        or ledger.get("rubric_integrity_sha256")
        != rubric["integrity"]["sha256"]
    ):
        raise OpenAILeakageRecallError("OpenAI local ledger contract differs")
    protocol_ledger = ledger.get("protocol_ledger")
    if not isinstance(protocol_ledger, Mapping):
        raise OpenAILeakageRecallError("protocol ledger is missing")
    judge_protocol.validate_local_ledger(
        protocol_ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    if any(
        entry["machine"]["request_count"] != 0
        or entry["machine"]["transport_status"] != "not_requested"
        or entry["machine"]["analysis_outcome"] != "missing"
        for entry in protocol_ledger["entries"]
    ):
        raise OpenAILeakageRecallError(
            "provider ledger must begin with all requests unrequested"
        )
    population = ledger.get("population_units")
    if not isinstance(population, list):
        raise OpenAILeakageRecallError("bootstrap population is missing")
    normalized = judge_protocol._normalize_units(population)
    if any(set(row) != set(_POPULATION_FIELDS) for row in normalized):
        raise OpenAILeakageRecallError(
            "bootstrap population fields differ"
        )
    reproduced = judge_protocol.freeze_from_units(
        normalized,
        artifacts=rubric["artifacts"],
    )
    if (
        reproduced.sample_lock != sample_lock
        or reproduced.rubric != rubric
        or judge_protocol.build_local_ledger(reproduced) != protocol_ledger
    ):
        raise OpenAILeakageRecallError(
            "local ledger does not reproduce the checked-in locks"
        )


def write_local_ledger(
    path: str | Path,
    ledger: Mapping[str, Any],
    *,
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> None:
    validate_local_ledger(
        ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    _atomic_write_new(path, ledger, local_only=True)


def validate_local_ledger_file(
    path: str | Path,
    *,
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_path = _path_without_symlinks(path)
    _validate_mode(ledger_path.parent, directory=True)
    _validate_mode(ledger_path, directory=False)
    ledger = _load_mapping(ledger_path, name="OpenAI local ledger")
    validate_local_ledger(
        ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    return ledger


def prepare_local_ledger(
    *,
    data_path: str | Path,
    final_path: str | Path,
    cohort_path: str | Path,
    sample_lock_path: str | Path,
    rubric_path: str | Path,
    ledger_out: str | Path,
) -> dict[str, Any]:
    """Regenerate frozen locks locally, compare them, then create the ledger."""

    sample_lock, rubric = load_public_protocol(
        sample_lock_path=sample_lock_path,
        rubric_path=rubric_path,
    )
    frozen = judge_protocol.freeze_paths(
        data_path=data_path,
        final_path=final_path,
        cohort_path=cohort_path,
    )
    if frozen.sample_lock != sample_lock or frozen.rubric != rubric:
        raise OpenAILeakageRecallError(
            "regenerated sample or rubric differs from checked-in locks"
        )
    ledger = build_local_ledger(frozen)
    write_local_ledger(
        ledger_out,
        ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    return ledger


def build_responses_payload(
    request: Mapping[str, Any],
    *,
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exactly one blinded Responses API payload."""

    judge_protocol.validate_rubric(rubric)
    _require_exact_keys(
        request,
        {"system_prompt", "user_prompt", "response_schema", "generation"},
        name="frozen local request",
    )
    generation_controls = {
        "temperature_parameter_supported": False,
        "temperature": None,
        "reasoning_mode": "standard",
        "reasoning_effort": "low",
    }
    if (
        judge_protocol.GENERATION_CONTROLS != generation_controls
        or request.get("system_prompt") != rubric["prompt"]["system"]
        or not isinstance(request.get("user_prompt"), str)
        or request.get("response_schema") != rubric["response_schema"]
        or request.get("generation") != generation_controls
    ):
        raise OpenAILeakageRecallError("frozen local request differs")
    return {
        "model": MODEL,
        "input": [
            {
                "role": "system",
                "content": request["system_prompt"],
            },
            {
                "role": "user",
                "content": request["user_prompt"],
            },
        ],
        "reasoning": {
            "mode": generation_controls["reasoning_mode"],
            "effort": generation_controls["reasoning_effort"],
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": FORMAT_NAME,
                "strict": True,
                "schema": copy.deepcopy(rubric["response_schema"]),
            }
        },
    }


def estimate_request_tokens(
    ledger: Mapping[str, Any],
    *,
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    """Conservatively approximate tokens from all rendered request bytes."""

    validate_local_ledger(
        ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    entries = ledger["protocol_ledger"]["entries"]
    system_characters = 0
    user_characters = 0
    prompt_utf8_bytes = 0
    request_json_utf8_bytes = 0
    for entry in entries:
        request = entry["request"]
        payload = build_responses_payload(request, rubric=rubric)
        system = request["system_prompt"]
        user = request["user_prompt"]
        system_characters += len(system)
        user_characters += len(user)
        prompt_utf8_bytes += len(system.encode("utf-8"))
        prompt_utf8_bytes += len(user.encode("utf-8"))
        request_json_utf8_bytes += len(_canonical_json_bytes(payload))
    if len(entries) != REQUEST_COUNT:
        raise OpenAILeakageRecallError(
            "token estimate must cover exactly 128 requests"
        )
    estimated_input = request_json_utf8_bytes
    maximum_output = REQUEST_COUNT * MAX_OUTPUT_TOKENS
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
            "UTF-8 byte of each complete canonical request JSON, including "
            "message wrappers and the repeated strict schema. This is an "
            "estimate, not tokenizer output or provider usage."
        ),
        "estimated_input_tokens": estimated_input,
        "maximum_output_tokens": maximum_output,
        "actual_api_usage_available_only_after_run": True,
    }


def estimate_request_cost(token_estimate: Mapping[str, Any]) -> dict[str, Any]:
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
    contract = {
        "method": HTTP_METHOD,
        "endpoint": ENDPOINT,
        "model": MODEL,
        "top_level_fields": [
            "model",
            "input",
            "reasoning",
            "max_output_tokens",
            "store",
            "text",
        ],
        "input": {
            "type": "message_array",
            "ordered_roles": ["system", "user"],
            "content_type": "string",
            "content_source": "frozen local render_judge_request output",
        },
        "reasoning": {
            "mode": "standard",
            "effort": "low",
        },
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": FORMAT_NAME,
                "strict": True,
                "schema": copy.deepcopy(rubric["response_schema"]),
            }
        },
        "response_text_extraction": "output[].content[].type == output_text",
    }
    contract["sha256"] = _payload_sha256(contract)
    return contract


def _authorization_body(
    *,
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
    sample_lock_path: str | Path,
    rubric_path: str | Path,
) -> dict[str, Any]:
    judge_protocol.validate_sample_lock(sample_lock, rubric=rubric)
    validate_local_ledger(
        local_ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    token_estimate = estimate_request_tokens(
        local_ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": AUTHORIZATION_STATUS,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "contains_source_identifiers": False,
        "contains_credentials": False,
        "instrument_validation_only": True,
        "headline_semantic_scoring": False,
        "artifacts": {
            "sample_lock": _artifact_binding(sample_lock_path, sample_lock),
            "rubric": _artifact_binding(rubric_path, rubric),
            "judge_protocol_implementation": _implementation_binding(
                JUDGE_PROTOCOL_PATH
            ),
            "provider_runner_implementation": _implementation_binding(
                RUNNER_PATH
            ),
            "local_ledger_integrity_sha256": local_ledger["integrity"][
                "sha256"
            ],
            "local_ledger_payload_sha256": _payload_sha256(local_ledger),
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
            "request_contract": _request_contract(rubric),
        },
        "execution": {
            "sample_units": REQUEST_COUNT,
            "requests_per_unit": 1,
            "authorized_requests": REQUEST_COUNT,
            "passes": 1,
            "retries": 0,
            "batch_api_used": False,
            "one_fixed_prompt": True,
            "prompt_sha256": rubric["prompt"]["prompt_sha256"],
            "generation_controls": copy.deepcopy(
                judge_protocol.GENERATION_CONTROLS
            ),
            "request_started_marker_precedes_transport": True,
            "started_without_terminal": "permanently_indeterminate",
            "started_without_terminal_may_be_retried": False,
            "terminal_slots_may_be_retried": False,
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
            "authorization_and_bound_public_files_must_match_committed_head": (
                True
            ),
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
            "estimator": (
                "existing estimate_corrected_leakage with conservative bounds"
            ),
            "cluster_bootstrap": {
                "method": "percentile_cluster_bootstrap",
                "K": judge_protocol.EXPECTED_CLUSTERS,
                "clusters_per_resample": judge_protocol.EXPECTED_CLUSTERS,
                "histories_resampled_within_cluster": False,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "confidence_level": CONFIDENCE_LEVEL,
            },
        },
        "human_validation": {
            "preselected_units": judge_protocol.EXPECTED_HUMAN_SAMPLE,
            "remains_secondary": True,
            "reported_only_when_labels_are_supplied": True,
            "statistic": "raw exact-label concordance only",
            "chance_correction": False,
        },
    }


def build_authorization(
    *,
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
    sample_lock_path: str | Path,
    rubric_path: str | Path,
) -> dict[str, Any]:
    """Build a deterministic public authorization without API activity."""

    authorization = _seal(
        _authorization_body(
            sample_lock=sample_lock,
            rubric=rubric,
            local_ledger=local_ledger,
            sample_lock_path=sample_lock_path,
            rubric_path=rubric_path,
        )
    )
    validate_authorization(authorization)
    return authorization


def _validate_authorization_static(
    authorization: Mapping[str, Any],
) -> None:
    _validate_seal(authorization, name="OpenAI authorization")
    _require_exact_keys(
        authorization,
        {
            "schema",
            "schema_version",
            "status",
            "contains_source_text",
            "contains_model_generated_text",
            "contains_source_identifiers",
            "contains_credentials",
            "instrument_validation_only",
            "headline_semantic_scoring",
            "artifacts",
            "provider",
            "execution",
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
        },
        name="OpenAI authorization",
    )
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("schema_version") != SCHEMA_VERSION
        or authorization.get("status") != AUTHORIZATION_STATUS
        or authorization.get("contains_source_text") is not False
        or authorization.get("contains_model_generated_text") is not False
        or authorization.get("contains_source_identifiers") is not False
        or authorization.get("contains_credentials") is not False
        or authorization.get("instrument_validation_only") is not True
        or authorization.get("headline_semantic_scoring") is not False
    ):
        raise OpenAILeakageRecallError(
            "authorization status or disclosure differs"
        )
    provider = authorization.get("provider")
    execution = authorization.get("execution")
    transfer = authorization.get("transfer")
    credentials = authorization.get("credential_policy")
    retention = authorization.get("retention")
    gate = authorization.get("authorization_gate")
    prices = authorization.get("price_assumptions")
    human = authorization.get("human_validation")
    analysis = authorization.get("analysis")
    if not all(
        isinstance(item, Mapping)
        for item in (
            provider,
            execution,
            transfer,
            credentials,
            retention,
            gate,
            prices,
            human,
            analysis,
        )
    ):
        raise OpenAILeakageRecallError(
            "authorization contract sections are missing"
        )
    request_contract = provider.get("request_contract")
    if (
        provider.get("name") != "OpenAI"
        or provider.get("api") != "Responses API"
        or provider.get("exact_requested_alias") != MODEL
        or provider.get("model_reference_kind") != "moving_alias"
        or provider.get("dated_snapshot_available") is not False
        or provider.get("immutable_model_revision_claim") is not False
        or provider.get("terminal_provider_returned_model_recorded")
        is not True
        or provider.get("terminal_provider_returned_model_validated")
        is not True
        or not isinstance(request_contract, Mapping)
        or request_contract.get("method") != HTTP_METHOD
        or request_contract.get("endpoint") != ENDPOINT
        or request_contract.get("model") != MODEL
        or "temperature" in request_contract
        or request_contract.get("reasoning")
        != {"mode": "standard", "effort": "low"}
        or request_contract.get("max_output_tokens") != MAX_OUTPUT_TOKENS
        or request_contract.get("store") is not False
        or request_contract.get("sha256")
        != _payload_sha256(
            {
                key: copy.deepcopy(value)
                for key, value in request_contract.items()
                if key != "sha256"
            }
        )
    ):
        raise OpenAILeakageRecallError("provider request binding differs")
    expected_format = (
        ((request_contract.get("text") or {}).get("format") or {})
    )
    if (
        expected_format.get("type") != "json_schema"
        or expected_format.get("name") != FORMAT_NAME
        or expected_format.get("strict") is not True
        or expected_format.get("schema") != judge_protocol.RESPONSE_SCHEMA
    ):
        raise OpenAILeakageRecallError(
            "provider strict response format differs"
        )
    if (
        execution.get("sample_units") != REQUEST_COUNT
        or execution.get("requests_per_unit") != 1
        or execution.get("authorized_requests") != REQUEST_COUNT
        or execution.get("passes") != 1
        or execution.get("retries") != 0
        or execution.get("batch_api_used") is not False
        or execution.get("one_fixed_prompt") is not True
        or execution.get("generation_controls")
        != judge_protocol.GENERATION_CONTROLS
        or execution.get("started_without_terminal")
        != "permanently_indeterminate"
        or execution.get("started_without_terminal_may_be_retried")
        is not False
        or execution.get("terminal_slots_may_be_retried") is not False
    ):
        raise OpenAILeakageRecallError("one-pass execution policy differs")
    if (
        transfer.get("only")
        != [
            "blinded question",
            "blinded reference",
            "blinded candidate",
            "frozen rubric",
            "frozen response schema",
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
    ):
        raise OpenAILeakageRecallError("transfer minimization differs")
    if (
        credentials.get("environment_variable") != "OPENAI_API_KEY"
        or credentials.get("environment_only") is not True
        or credentials.get("dotenv_files_read") is not False
        or credentials.get("credential_may_enter_logs_or_artifacts")
        is not False
        or retention.get("standard_api_implies_zero_retention") is not False
        or retention.get("store") is not False
        or "does not promise zero abuse-monitoring retention"
        not in str(retention.get("statement"))
        or gate.get("exact_acknowledgement") != ACKNOWLEDGEMENT
        or any(
            gate.get(key) is not True
            for key in (
                "authorization_and_bound_public_files_must_match_committed_head",
                "gate_precedes_environment_key_read",
                "gate_precedes_local_ledger_read",
                "gate_precedes_requests",
            )
        )
    ):
        raise OpenAILeakageRecallError(
            "credential, retention, or run gate differs"
        )
    if (
        prices.get("source") != PRICE_SOURCE
        or prices.get("official_gpt_5_6_luna_page") is not True
        or prices.get("estimate_only") is not True
        or prices.get("input_usd_per_million_tokens")
        != INPUT_USD_PER_MILLION_TOKENS
        or prices.get("output_usd_per_million_tokens")
        != OUTPUT_USD_PER_MILLION_TOKENS
        or prices.get("actual_billing_may_differ") is not True
        or human.get("preselected_units")
        != judge_protocol.EXPECTED_HUMAN_SAMPLE
        or human.get("remains_secondary") is not True
        or human.get("reported_only_when_labels_are_supplied") is not True
        or (analysis.get("cluster_bootstrap") or {}).get("K")
        != judge_protocol.EXPECTED_CLUSTERS
    ):
        raise OpenAILeakageRecallError(
            "price, bootstrap, or human contract differs"
        )
    artifacts = authorization.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise OpenAILeakageRecallError("authorization artifacts are missing")
    for role in ("sample_lock", "rubric"):
        binding = artifacts.get(role)
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("repository_path"), str)
            or not _is_sha256(binding.get("file_sha256"))
            or not _is_sha256(binding.get("payload_sha256"))
            or not _is_sha256(binding.get("integrity_sha256"))
            or binding.get("immutable") is not True
        ):
            raise OpenAILeakageRecallError(
                f"{role} authorization binding differs"
            )
    for role in (
        "judge_protocol_implementation",
        "provider_runner_implementation",
    ):
        binding = artifacts.get(role)
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("repository_path"), str)
            or not _is_sha256(binding.get("file_sha256"))
            or binding.get(
                "committed_head_required_before_key_or_ledger_read"
            )
            is not True
        ):
            raise OpenAILeakageRecallError(
                f"{role} authorization binding differs"
            )
    if (
        not _is_sha256(artifacts.get("local_ledger_integrity_sha256"))
        or not _is_sha256(artifacts.get("local_ledger_payload_sha256"))
    ):
        raise OpenAILeakageRecallError(
            "local ledger authorization hashes differ"
        )
    estimate_request_cost(authorization["estimate_request_tokens"])
    if authorization.get("estimate_request_cost") != estimate_request_cost(
        authorization["estimate_request_tokens"]
    ):
        raise OpenAILeakageRecallError("request cost estimate differs")


def validate_authorization(
    authorization: Mapping[str, Any],
    *,
    sample_lock: Mapping[str, Any] | None = None,
    rubric: Mapping[str, Any] | None = None,
    local_ledger: Mapping[str, Any] | None = None,
    sample_lock_path: str | Path | None = None,
    rubric_path: str | Path | None = None,
) -> None:
    _validate_authorization_static(authorization)
    supplied = (
        sample_lock,
        rubric,
        local_ledger,
        sample_lock_path,
        rubric_path,
    )
    if any(item is not None for item in supplied):
        if any(item is None for item in supplied):
            raise OpenAILeakageRecallError(
                "full authorization validation inputs are required"
            )
        assert sample_lock is not None
        assert rubric is not None
        assert local_ledger is not None
        assert sample_lock_path is not None
        assert rubric_path is not None
        expected = _seal(
            _authorization_body(
                sample_lock=sample_lock,
                rubric=rubric,
                local_ledger=local_ledger,
                sample_lock_path=sample_lock_path,
                rubric_path=rubric_path,
            )
        )
        if dict(authorization) != expected:
            raise OpenAILeakageRecallError(
                "authorization differs from bound artifacts"
            )


def write_authorization(
    path: str | Path,
    authorization: Mapping[str, Any],
) -> None:
    validate_authorization(authorization)
    _atomic_write_new(path, authorization, local_only=False)


def validate_authorization_file(
    path: str | Path,
    *,
    sample_lock: Mapping[str, Any] | None = None,
    rubric: Mapping[str, Any] | None = None,
    local_ledger: Mapping[str, Any] | None = None,
    sample_lock_path: str | Path | None = None,
    rubric_path: str | Path | None = None,
) -> dict[str, Any]:
    authorization_path = _path_without_symlinks(path)
    if (
        not authorization_path.is_file()
        or _mode(authorization_path) != 0o644
    ):
        raise OpenAILeakageRecallError(
            "public authorization must be a regular 0644 file"
        )
    authorization = _load_mapping(
        authorization_path,
        name="OpenAI authorization",
    )
    validate_authorization(
        authorization,
        sample_lock=sample_lock,
        rubric=rubric,
        local_ledger=local_ledger,
        sample_lock_path=sample_lock_path,
        rubric_path=rubric_path,
    )
    return authorization


def freeze_authorization(
    *,
    sample_lock_path: str | Path,
    rubric_path: str | Path,
    local_ledger_path: str | Path,
    authorization_out: str | Path,
) -> dict[str, Any]:
    sample_lock, rubric = load_public_protocol(
        sample_lock_path=sample_lock_path,
        rubric_path=rubric_path,
    )
    ledger = validate_local_ledger_file(
        local_ledger_path,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    authorization = build_authorization(
        sample_lock=sample_lock,
        rubric=rubric,
        local_ledger=ledger,
        sample_lock_path=sample_lock_path,
        rubric_path=rubric_path,
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


def _bound_path(
    authorization: Mapping[str, Any],
    *,
    role: str,
    provided: str | Path,
) -> Path:
    binding = authorization["artifacts"][role]
    artifact = _path_without_symlinks(provided)
    if _repository_path(artifact) != binding["repository_path"]:
        raise PermissionError(f"{role} path differs from authorization")
    return artifact


def _check_bound_committed_file(
    authorization: Mapping[str, Any],
    *,
    role: str,
    path: str | Path,
) -> Path:
    artifact = _bound_path(
        authorization,
        role=role,
        provided=path,
    )
    _require_head_committed_file(artifact)
    if (
        not artifact.is_file()
        or _mode(artifact) != 0o644
        or _file_sha256(artifact)
        != authorization["artifacts"][role]["file_sha256"]
    ):
        raise PermissionError(f"{role} differs from authorization hash")
    return artifact


def _authorize_before_private_read(
    *,
    acknowledgement: str,
    authorization_path: str | Path,
    sample_lock_path: str | Path,
    rubric_path: str | Path,
) -> tuple[dict[str, Any], Path, Path]:
    if acknowledgement != ACKNOWLEDGEMENT:
        raise PermissionError(
            "exact OpenAI source-bearing acknowledgement is required"
        )
    authorization_file = _path_without_symlinks(authorization_path)
    _require_head_committed_file(authorization_file)
    if (
        not authorization_file.is_file()
        or _mode(authorization_file) != 0o644
    ):
        raise PermissionError(
            "committed authorization must be a regular 0644 file"
        )
    authorization = _load_mapping(
        authorization_file,
        name="OpenAI authorization",
    )
    _validate_authorization_static(authorization)
    sample_path = _check_bound_committed_file(
        authorization,
        role="sample_lock",
        path=sample_lock_path,
    )
    rubric_path_checked = _check_bound_committed_file(
        authorization,
        role="rubric",
        path=rubric_path,
    )
    implementations = {
        "judge_protocol_implementation": JUDGE_PROTOCOL_PATH,
        "provider_runner_implementation": RUNNER_PATH,
    }
    for role, path in implementations.items():
        _check_bound_committed_file(
            authorization,
            role=role,
            path=path,
        )
    return authorization, sample_path, rubric_path_checked


def _run_manifest_body(
    *,
    authorization_path: str | Path,
    authorization: Mapping[str, Any],
    sample_lock_path: str | Path,
    sample_lock: Mapping[str, Any],
    rubric_path: str | Path,
    rubric: Mapping[str, Any],
    local_ledger_path: str | Path,
    local_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "one-pass-run-initialized",
        "local_only": True,
        "source_bearing_evidence_may_follow": True,
        "directory_mode": "0700",
        "file_mode": "0600",
        "request_slots": REQUEST_COUNT,
        "passes": 1,
        "retries": 0,
        "batch_api_used": False,
        "bindings": {
            "authorization": {
                "file_sha256": _file_sha256(authorization_path),
                "integrity_sha256": authorization["integrity"]["sha256"],
            },
            "sample_lock": {
                "file_sha256": _file_sha256(sample_lock_path),
                "integrity_sha256": sample_lock["integrity"]["sha256"],
            },
            "rubric": {
                "file_sha256": _file_sha256(rubric_path),
                "integrity_sha256": rubric["integrity"]["sha256"],
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
    _validate_seal(manifest, name="run manifest")
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
            "bindings",
            "integrity",
        },
        name="run manifest",
    )
    if (
        manifest.get("schema") != RUN_MANIFEST_SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "one-pass-run-initialized"
        or manifest.get("local_only") is not True
        or manifest.get("source_bearing_evidence_may_follow") is not True
        or manifest.get("directory_mode") != "0700"
        or manifest.get("file_mode") != "0600"
        or manifest.get("request_slots") != REQUEST_COUNT
        or manifest.get("passes") != 1
        or manifest.get("retries") != 0
        or manifest.get("batch_api_used") is not False
    ):
        raise OpenAILeakageRecallError("run manifest contract differs")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "authorization",
        "sample_lock",
        "rubric",
        "local_ledger",
    }:
        raise OpenAILeakageRecallError("run manifest bindings differ")
    for binding in bindings.values():
        if not isinstance(binding, Mapping) or any(
            not _is_sha256(value) for value in binding.values()
        ):
            raise OpenAILeakageRecallError(
                "run manifest binding hash differs"
            )
    if expected is not None and dict(manifest) != dict(expected):
        raise OpenAILeakageRecallError(
            "run manifest differs from current bound artifacts"
        )


def _initialize_or_validate_run_root(
    root: str | Path,
    *,
    expected_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output_root = _path_without_symlinks(root)
    if output_root.exists():
        _validate_mode(output_root, directory=True)
        manifest_path = output_root / "run.json"
        _validate_mode(manifest_path, directory=False)
        manifest = _load_mapping(manifest_path, name="run manifest")
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
    allowed = {"run.json", "started", "terminal"}
    if any(path.name not in allowed for path in output_root.iterdir()):
        raise OpenAILeakageRecallError(
            "run root contains an unrecognized entry"
        )
    return output_root, manifest


def _marker_bindings(
    *,
    manifest: Mapping[str, Any],
    sample_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "unit_binding_sha256": unit_binding_sha256,
        "run_manifest_integrity_sha256": manifest["integrity"]["sha256"],
        "authorization_integrity_sha256": manifest["bindings"][
            "authorization"
        ]["integrity_sha256"],
        "sample_lock_integrity_sha256": manifest["bindings"]["sample_lock"][
            "integrity_sha256"
        ],
        "rubric_integrity_sha256": manifest["bindings"]["rubric"][
            "integrity_sha256"
        ],
        "local_ledger_integrity_sha256": manifest["bindings"][
            "local_ledger"
        ]["integrity_sha256"],
    }


def build_started_marker(
    *,
    manifest: Mapping[str, Any],
    sample_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": STARTED_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "phase": "started",
            **_marker_bindings(
                manifest=manifest,
                sample_index=sample_index,
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
    sample_index: int,
    unit_binding_sha256: str,
) -> None:
    _validate_seal(marker, name="started marker")
    expected = build_started_marker(
        manifest=manifest,
        sample_index=sample_index,
        unit_binding_sha256=unit_binding_sha256,
    )
    if dict(marker) != expected:
        raise OpenAILeakageRecallError("started marker binding differs")


def _normalize_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    usage = copy.deepcopy(dict(value))
    _assert_json_value(usage, name="response usage")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        observed = usage.get(key)
        if type(observed) is not int or observed < 0:
            return None
    if usage["total_tokens"] != (
        usage["input_tokens"] + usage["output_tokens"]
    ):
        return None
    return usage


def _extract_output_text(response: Mapping[str, Any]) -> str:
    output = response.get("output")
    if not isinstance(output, list):
        raise OpenAILeakageRecallError(
            "provider response output must be a list"
        )
    pieces: list[str] = []
    for item in output:
        if not isinstance(item, Mapping):
            raise OpenAILeakageRecallError(
                "provider output item must be an object"
            )
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                raise OpenAILeakageRecallError(
                    "provider content item must be an object"
                )
            if part.get("type") == "output_text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise OpenAILeakageRecallError(
                        "output_text content must contain text"
                    )
                pieces.append(text)
    if not pieces:
        raise OpenAILeakageRecallError(
            "provider response contains no output_text"
        )
    return "".join(pieces)


def _response_projection(response: Mapping[str, Any]) -> dict[str, Any]:
    response_id = response.get("id")
    model = response.get("model")
    status_value = response.get("status")
    return {
        "id": response_id if isinstance(response_id, str) else None,
        "model": model if isinstance(model, str) else None,
        "status": status_value if isinstance(status_value, str) else None,
        "usage": _normalize_usage(response.get("usage")),
        "raw_output": None,
    }


def _error_record(kind: str, exc: BaseException) -> dict[str, Any]:
    error_type = type(exc).__name__
    fingerprint = _text_sha256(f"{error_type}\0{str(exc)}")
    return {
        "kind": kind,
        "error_type": error_type,
        "error_sha256": fingerprint,
        "message_redacted": True,
    }


def _evaluate_response(
    response: Mapping[str, Any],
    *,
    candidate: str,
) -> tuple[dict[str, Any], dict[str, Any] | None, str, dict[str, Any] | None]:
    projection = _response_projection(response)
    try:
        if (
            not projection["id"]
            or projection["model"] != MODEL
            or projection["status"] != "completed"
            or projection["usage"] is None
        ):
            raise OpenAILeakageRecallError(
                "response identity, model, status, or usage differs"
            )
        raw_output = _extract_output_text(response)
        projection["raw_output"] = raw_output
        parsed = judge_protocol.validate_judge_response(
            raw_output,
            candidate=candidate,
        )
    except (
        OpenAILeakageRecallError,
        judge_protocol.LeakageRecallProtocolError,
    ) as exc:
        try:
            projection["raw_output"] = _extract_output_text(response)
        except OpenAILeakageRecallError:
            projection["raw_output"] = None
        return projection, None, "parse_error", _error_record("parse", exc)
    return projection, parsed, parsed["label"], None


def _terminal_base(
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    sample_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": TERMINAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "phase": "terminal",
        **_marker_bindings(
            manifest=manifest,
            sample_index=sample_index,
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
    sample_index: int,
    unit_binding_sha256: str,
    response: Mapping[str, Any],
    candidate: str,
) -> dict[str, Any]:
    projection, parsed, outcome, failure = _evaluate_response(
        response,
        candidate=candidate,
    )
    return _seal(
        {
            **_terminal_base(
                manifest=manifest,
                started=started,
                sample_index=sample_index,
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
    sample_index: int,
    unit_binding_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    return _seal(
        {
            **_terminal_base(
                manifest=manifest,
                started=started,
                sample_index=sample_index,
                unit_binding_sha256=unit_binding_sha256,
            ),
            "terminal_kind": "transport_failure",
            "response": None,
            "parsed_response": None,
            "analysis_outcome": "transport_error",
            "failure": _error_record("transport", error),
        }
    )


def build_indeterminate_terminal(
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    sample_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    error = RuntimeError(
        "a prior started request has no durable terminal record"
    )
    return _seal(
        {
            **_terminal_base(
                manifest=manifest,
                started=started,
                sample_index=sample_index,
                unit_binding_sha256=unit_binding_sha256,
            ),
            "terminal_kind": "permanently_indeterminate",
            "response": None,
            "parsed_response": None,
            "analysis_outcome": "transport_error",
            "failure": _error_record("interrupted_after_start", error),
        }
    )


def validate_terminal(
    terminal: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    started: Mapping[str, Any],
    sample_index: int,
    unit_binding_sha256: str,
    candidate: str,
) -> None:
    _validate_seal(terminal, name="terminal")
    _require_exact_keys(
        terminal,
        {
            "schema",
            "schema_version",
            "phase",
            "sample_index",
            "unit_binding_sha256",
            "run_manifest_integrity_sha256",
            "authorization_integrity_sha256",
            "sample_lock_integrity_sha256",
            "rubric_integrity_sha256",
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
        },
        name="terminal",
    )
    expected_base = _terminal_base(
        manifest=manifest,
        started=started,
        sample_index=sample_index,
        unit_binding_sha256=unit_binding_sha256,
    )
    for key, value in expected_base.items():
        if terminal.get(key) != value:
            raise OpenAILeakageRecallError("terminal binding differs")
    kind = terminal.get("terminal_kind")
    if (
        kind not in _TERMINAL_KINDS
        or terminal.get("analysis_outcome") not in _ANALYSIS_OUTCOMES
    ):
        raise OpenAILeakageRecallError("terminal kind or outcome differs")
    response = terminal.get("response")
    parsed = terminal.get("parsed_response")
    failure = terminal.get("failure")
    if kind in {"transport_failure", "permanently_indeterminate"}:
        expected_failure_kind = (
            "transport"
            if kind == "transport_failure"
            else "interrupted_after_start"
        )
        if (
            response is not None
            or parsed is not None
            or terminal.get("analysis_outcome") != "transport_error"
            or not isinstance(failure, Mapping)
            or set(failure)
            != {
                "kind",
                "error_type",
                "error_sha256",
                "message_redacted",
            }
            or failure.get("kind") != expected_failure_kind
            or not isinstance(failure.get("error_type"), str)
            or not failure["error_type"]
            or not _is_sha256(failure.get("error_sha256"))
            or failure.get("message_redacted") is not True
        ):
            raise OpenAILeakageRecallError(
                "non-response terminal state differs"
            )
        return
    if not isinstance(response, Mapping):
        raise OpenAILeakageRecallError("response terminal lacks response")
    _require_exact_keys(
        response,
        {"id", "model", "status", "usage", "raw_output"},
        name="terminal response",
    )
    if kind == "completed":
        if (
            not isinstance(response.get("id"), str)
            or not response["id"]
            or response.get("model") != MODEL
            or response.get("status") != "completed"
            or _normalize_usage(response.get("usage")) != response.get("usage")
            or not isinstance(response.get("raw_output"), str)
            or not isinstance(parsed, Mapping)
            or failure is not None
        ):
            raise OpenAILeakageRecallError(
                "completed terminal metadata differs"
            )
        expected_parsed = judge_protocol.validate_judge_response(
            response["raw_output"],
            candidate=candidate,
        )
        if (
            dict(parsed) != expected_parsed
            or terminal.get("analysis_outcome") != expected_parsed["label"]
        ):
            raise OpenAILeakageRecallError(
                "completed terminal parsed response differs"
            )
        return
    if (
        parsed is not None
        or terminal.get("analysis_outcome") != "parse_error"
        or not isinstance(failure, Mapping)
        or set(failure)
        != {
            "kind",
            "error_type",
            "error_sha256",
            "message_redacted",
        }
        or failure.get("kind") != "parse"
        or not isinstance(failure.get("error_type"), str)
        or not failure["error_type"]
        or not _is_sha256(failure.get("error_sha256"))
        or failure.get("message_redacted") is not True
    ):
        raise OpenAILeakageRecallError("parse-failure terminal differs")
    parse_should_fail = (
        response.get("id") is None
        or response.get("model") != MODEL
        or response.get("status") != "completed"
        or _normalize_usage(response.get("usage")) is None
        or not isinstance(response.get("raw_output"), str)
    )
    if not parse_should_fail:
        try:
            judge_protocol.validate_judge_response(
                response["raw_output"],
                candidate=candidate,
            )
        except judge_protocol.LeakageRecallProtocolError:
            parse_should_fail = True
    if not parse_should_fail:
        raise OpenAILeakageRecallError(
            "parse-failure terminal contains a valid response"
        )


def _evidence_file(
    root: Path,
    phase: str,
    sample_index: int,
) -> Path:
    return root / phase / f"{sample_index:03d}.json"


def _scan_evidence(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    sample_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    _validate_mode(root, directory=True)
    allowed = {"run.json", "started", "terminal"}
    root_entries = list(root.iterdir())
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
        for path in root_entries
    ):
        raise OpenAILeakageRecallError(
            "run root entry, type, or permissions differ"
        )
    public = {row["sample_index"]: row for row in sample_lock["samples"]}
    entries = local_ledger["protocol_ledger"]["entries"]
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
                f"{phase} evidence directory has an extra entry"
            )
        for path in paths:
            _validate_mode(path, directory=False)
            index = int(path.stem)
            value = _load_mapping(path, name=f"{phase} marker")
            binding = public[index]["unit_binding_sha256"]
            if phase == "started":
                validate_started_marker(
                    value,
                    manifest=manifest,
                    sample_index=index,
                    unit_binding_sha256=binding,
                )
            else:
                started = found["started"].get(index)
                if started is None:
                    started_path = _evidence_file(root, "started", index)
                    if not started_path.is_file():
                        raise OpenAILeakageRecallError(
                            "terminal exists without a started marker"
                        )
                    _validate_mode(started_path, directory=False)
                    started = _load_mapping(
                        started_path,
                        name="started marker",
                    )
                    validate_started_marker(
                        started,
                        manifest=manifest,
                        sample_index=index,
                        unit_binding_sha256=binding,
                    )
                    found["started"][index] = started
                validate_terminal(
                    value,
                    manifest=manifest,
                    started=started,
                    sample_index=index,
                    unit_binding_sha256=binding,
                    candidate=entries[index]["candidate_value"],
                )
            found[phase][index] = value
    if any(index not in found["started"] for index in found["terminal"]):
        raise OpenAILeakageRecallError(
            "terminal evidence has no started marker"
        )
    return found["started"], found["terminal"]


def urllib_responses_transport(
    *,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> bytes:
    """Issue one urllib request.  Retry and logging behavior are absent."""

    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers),
        method=HTTP_METHOD,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _decode_transport_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return copy.deepcopy(dict(result))
    if isinstance(result, (str, bytes)):
        value = _loads_json(result, name="provider response")
        if isinstance(value, dict):
            return value
    raise OpenAILeakageRecallError(
        "provider transport did not return a JSON object"
    )


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
    sample_index: int,
    unit_binding_sha256: str,
) -> dict[str, Any]:
    marker = build_started_marker(
        manifest=manifest,
        sample_index=sample_index,
        unit_binding_sha256=unit_binding_sha256,
    )
    _atomic_write_new(
        _evidence_file(root, "started", sample_index),
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
    index = terminal["sample_index"]
    _atomic_write_new(
        _evidence_file(root, "terminal", index),
        terminal,
        local_only=True,
    )


def _finalize_incomplete_starts(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    sample_lock: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    started, terminals = _scan_evidence(
        root,
        manifest=manifest,
        sample_lock=sample_lock,
        local_ledger=local_ledger,
    )
    public = {row["sample_index"]: row for row in sample_lock["samples"]}
    for index in sorted(set(started) - set(terminals)):
        terminal = build_indeterminate_terminal(
            manifest=manifest,
            started=started[index],
            sample_index=index,
            unit_binding_sha256=public[index]["unit_binding_sha256"],
        )
        _write_terminal(root, terminal)
        terminals[index] = terminal
    return started, terminals


def run_one_pass(
    *,
    authorization_path: str | Path,
    sample_lock_path: str | Path,
    rubric_path: str | Path,
    local_ledger_path: str | Path,
    run_root: str | Path,
    acknowledgement: str,
    transport: Transport | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Resume unstarted slots and consume each request slot at most once."""

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    authorization, checked_sample_path, checked_rubric_path = (
        _authorize_before_private_read(
            acknowledgement=acknowledgement,
            authorization_path=authorization_path,
            sample_lock_path=sample_lock_path,
            rubric_path=rubric_path,
        )
    )
    sample_lock, rubric = load_public_protocol(
        sample_lock_path=checked_sample_path,
        rubric_path=checked_rubric_path,
    )
    ledger = validate_local_ledger_file(
        local_ledger_path,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    validate_authorization(
        authorization,
        sample_lock=sample_lock,
        rubric=rubric,
        local_ledger=ledger,
        sample_lock_path=checked_sample_path,
        rubric_path=checked_rubric_path,
    )
    manifest = build_run_manifest(
        authorization_path=authorization_path,
        authorization=authorization,
        sample_lock_path=checked_sample_path,
        sample_lock=sample_lock,
        rubric_path=checked_rubric_path,
        rubric=rubric,
        local_ledger_path=local_ledger_path,
        local_ledger=ledger,
    )
    root, manifest = _initialize_or_validate_run_root(
        run_root,
        expected_manifest=manifest,
    )
    started, terminals = _finalize_incomplete_starts(
        root,
        manifest=manifest,
        sample_lock=sample_lock,
        local_ledger=ledger,
    )
    entries = ledger["protocol_ledger"]["entries"]
    public = {row["sample_index"]: row for row in sample_lock["samples"]}
    selected_transport = transport or urllib_responses_transport
    pending = [
        index for index in range(REQUEST_COUNT) if index not in terminals
    ]
    key = _read_openai_key() if pending else None
    for index in range(REQUEST_COUNT):
        if index in terminals:
            continue
        if index in started:
            raise OpenAILeakageRecallError(
                "incomplete started slot was not made indeterminate"
            )
        binding = public[index]["unit_binding_sha256"]
        marker = _write_started(
            root,
            manifest=manifest,
            sample_index=index,
            unit_binding_sha256=binding,
        )
        started[index] = marker
        assert key is not None
        payload = build_responses_payload(
            entries[index]["request"],
            rubric=rubric,
        )
        body = _canonical_json_bytes(payload)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        try:
            raw_response = selected_transport(
                url=ENDPOINT,
                headers=headers,
                body=body,
                timeout=float(timeout),
            )
            response = _decode_transport_result(raw_response)
            terminal = build_response_terminal(
                manifest=manifest,
                started=marker,
                sample_index=index,
                unit_binding_sha256=binding,
                response=response,
                candidate=entries[index]["candidate_value"],
            )
            _assert_secret_absent(terminal, key)
        except Exception as exc:
            terminal = build_transport_failure_terminal(
                manifest=manifest,
                started=marker,
                sample_index=index,
                unit_binding_sha256=binding,
                error=exc,
            )
        _write_terminal(root, terminal, secret=key)
        terminals[index] = terminal
    started, terminals = _scan_evidence(
        root,
        manifest=manifest,
        sample_lock=sample_lock,
        local_ledger=ledger,
    )
    if len(started) != REQUEST_COUNT or len(terminals) != REQUEST_COUNT:
        raise OpenAILeakageRecallError(
            "one-pass run does not cover all 128 request slots"
        )
    return {
        "request_slots": REQUEST_COUNT,
        "started_slots": len(started),
        "terminal_slots": len(terminals),
        "completed_labels": sum(
            terminal["terminal_kind"] == "completed"
            for terminal in terminals.values()
        ),
        "parse_failures": sum(
            terminal["terminal_kind"] == "parse_failure"
            for terminal in terminals.values()
        ),
        "transport_failures": sum(
            terminal["terminal_kind"] == "transport_failure"
            for terminal in terminals.values()
        ),
        "permanently_indeterminate": sum(
            terminal["terminal_kind"] == "permanently_indeterminate"
            for terminal in terminals.values()
        ),
        "retries": 0,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise OpenAILeakageRecallError(
            "bootstrap distribution must not be empty"
        )
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def cluster_bootstrap_interval(
    cluster_contributions: Sequence[float],
    *,
    denominator: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = tuple(float(value) for value in cluster_contributions)
    if (
        len(values) != judge_protocol.EXPECTED_CLUSTERS
        or any(not math.isfinite(value) for value in values)
        or type(denominator) is not int
        or denominator < 1
        or type(resamples) is not int
        or resamples < 1
        or type(seed) is not int
    ):
        raise OpenAILeakageRecallError(
            "bootstrap requires 32 finite clusters and positive settings"
        )
    rng = random.Random(seed)
    distribution = [
        math.fsum(
            values[rng.randrange(judge_protocol.EXPECTED_CLUSTERS)]
            for _ in range(judge_protocol.EXPECTED_CLUSTERS)
        )
        / denominator
        for _ in range(resamples)
    ]
    distribution.sort()
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return {
        "method": "percentile_cluster_bootstrap",
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling_unit": "anonymous_target_cluster",
        "K": judge_protocol.EXPECTED_CLUSTERS,
        "clusters_per_resample": judge_protocol.EXPECTED_CLUSTERS,
        "histories_resampled_within_cluster": False,
        "resamples": resamples,
        "seed": seed,
        "percentile_interpolation": "linear_type_7",
        "lower": _percentile(distribution, alpha),
        "upper": _percentile(distribution, 1.0 - alpha),
    }


def _actual_usage(
    terminals: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    usage_rows = [
        terminal["response"]["usage"]
        for terminal in terminals.values()
        if isinstance(terminal.get("response"), Mapping)
        and isinstance(terminal["response"].get("usage"), Mapping)
    ]
    input_tokens = sum(int(row["input_tokens"]) for row in usage_rows)
    output_tokens = sum(int(row["output_tokens"]) for row in usage_rows)
    total_tokens = sum(int(row["total_tokens"]) for row in usage_rows)
    return {
        "responses_with_usage": len(usage_rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
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


def _outcome_rows(
    sample_lock: Mapping[str, Any],
    terminals: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": row["sample_index"],
            "condition_ordinal": row["condition_ordinal"],
            "unit_binding_sha256": row["unit_binding_sha256"],
            "outcome": (
                terminals[row["sample_index"]]["analysis_outcome"]
                if row["sample_index"] in terminals
                else "missing"
            ),
        }
        for row in sample_lock["samples"]
    ]


def _bootstrap_from_estimate(
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    hooks = estimate["cluster_bootstrap_hooks"]
    rows = hooks.get("rows")
    if (
        hooks.get("available") is not True
        or not isinstance(rows, list)
        or len(rows)
        != judge_protocol.EXPECTED_CLUSTERS
        * judge_protocol.CONDITION_COUNT
    ):
        raise OpenAILeakageRecallError(
            "corrected estimator did not provide 32-cluster hooks"
        )
    lower: list[float] = []
    upper: list[float] = []
    for cluster_index in range(judge_protocol.EXPECTED_CLUSTERS):
        cluster_rows = [
            row for row in rows if row["cluster_index"] == cluster_index
        ]
        if len(cluster_rows) != judge_protocol.CONDITION_COUNT:
            raise OpenAILeakageRecallError(
                "bootstrap cluster contribution geometry differs"
            )
        lower.append(
            math.fsum(
                float(row["corrected_lower_contribution"])
                for row in cluster_rows
            )
        )
        upper.append(
            math.fsum(
                float(row["corrected_upper_contribution"])
                for row in cluster_rows
            )
        )
    return {
        "corrected_leakage_rate_lower_bound_estimator": (
            cluster_bootstrap_interval(
                lower,
                denominator=judge_protocol.EXPECTED_POPULATION,
            )
        ),
        "corrected_leakage_rate_upper_bound_estimator": (
            cluster_bootstrap_interval(
                upper,
                denominator=judge_protocol.EXPECTED_POPULATION,
            )
        ),
    }


def build_source_free_summary(
    *,
    authorization: Mapping[str, Any],
    sample_lock: Mapping[str, Any],
    rubric: Mapping[str, Any],
    local_ledger: Mapping[str, Any],
    manifest: Mapping[str, Any],
    terminals: Mapping[int, Mapping[str, Any]],
    human_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Build aggregate-only estimates; omit prompts, outputs, and IDs."""

    _validate_authorization_static(authorization)
    validate_local_ledger(
        local_ledger,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    validate_run_manifest(manifest)
    outcomes = _outcome_rows(sample_lock, terminals)
    estimate = judge_protocol.estimate_corrected_leakage(
        sample_lock,
        outcomes,
        population_units=local_ledger["population_units"],
    )
    usage = _actual_usage(terminals)
    overall = estimate["overall_inclusion_weighted"]
    per_condition = [
        {
            "stratum_ordinal": row["condition_ordinal"],
            "outcome_counts": copy.deepcopy(row["outcome_counts"]),
            "corrected_leakage_rate_lower": row[
                "corrected_leakage_rate_lower"
            ],
            "corrected_leakage_rate_upper": row[
                "corrected_leakage_rate_upper"
            ],
        }
        for row in estimate["per_condition"]
    ]
    summary_body: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "source-free-summary",
        "instrument_validation_only": True,
        "headline_semantic_scoring": False,
        "source_free": True,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "contains_source_or_response_ids": False,
        "bindings": {
            "authorization_integrity_sha256": authorization["integrity"][
                "sha256"
            ],
            "sample_lock_integrity_sha256": sample_lock["integrity"]["sha256"],
            "rubric_integrity_sha256": rubric["integrity"]["sha256"],
            "local_ledger_integrity_sha256": local_ledger["integrity"][
                "sha256"
            ],
            "run_manifest_integrity_sha256": manifest["integrity"]["sha256"],
        },
        "coverage": {
            "sample_units": REQUEST_COUNT,
            "terminal_units": len(terminals),
            "missing_unstarted_units": REQUEST_COUNT - len(terminals),
            "outcome_counts": {
                name: sum(row["outcome"] == name for row in outcomes)
                for name in judge_protocol.ANALYSIS_OUTCOMES
            },
            "one_pass": True,
            "retries": 0,
        },
        "actual_usage": usage,
        "actual_usage_cost": _actual_cost(usage),
        "corrected_estimate": {
            "bounds": copy.deepcopy(estimate["bounds"]),
            "per_condition": per_condition,
            "overall": {
                "population_size": overall["population_size"],
                "corrected_leakage_rate_lower": overall[
                    "corrected_leakage_rate_lower"
                ],
                "corrected_leakage_rate_upper": overall[
                    "corrected_leakage_rate_upper"
                ],
            },
            "cluster_bootstrap_95_intervals": (
                _bootstrap_from_estimate(estimate)
            ),
        },
    }
    if human_labels is not None:
        summary_body["human_validation_secondary"] = (
            judge_protocol.raw_human_concordance(
                sample_lock,
                outcomes,
                human_labels,
            )
        )
    summary = _seal(summary_body)
    validate_source_free_summary(
        summary,
        human_labels_supplied=human_labels is not None,
    )
    return summary


def validate_source_free_summary(
    summary: Mapping[str, Any],
    *,
    human_labels_supplied: bool | None = None,
) -> None:
    _validate_seal(summary, name="source-free summary")
    required = {
        "schema",
        "schema_version",
        "status",
        "instrument_validation_only",
        "headline_semantic_scoring",
        "source_free",
        "contains_source_text",
        "contains_model_generated_text",
        "contains_source_or_response_ids",
        "bindings",
        "coverage",
        "actual_usage",
        "actual_usage_cost",
        "corrected_estimate",
        "integrity",
    }
    has_human = "human_validation_secondary" in summary
    if has_human:
        required.add("human_validation_secondary")
    _require_exact_keys(summary, required, name="source-free summary")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("status") != "source-free-summary"
        or summary.get("instrument_validation_only") is not True
        or summary.get("headline_semantic_scoring") is not False
        or summary.get("source_free") is not True
        or summary.get("contains_source_text") is not False
        or summary.get("contains_model_generated_text") is not False
        or summary.get("contains_source_or_response_ids") is not False
        or (
            human_labels_supplied is not None
            and has_human is not human_labels_supplied
        )
    ):
        raise OpenAILeakageRecallError(
            "source-free summary disclosure differs"
        )
    serialized = deterministic_json(summary)
    forbidden_keys = (
        '"raw_output"',
        '"parsed_response"',
        '"response_id"',
        '"question"',
        '"reference"',
        '"candidate"',
        '"full_history"',
    )
    if any(key in serialized for key in forbidden_keys):
        raise OpenAILeakageRecallError(
            "source-free summary contains prohibited detailed evidence"
        )
    coverage = summary.get("coverage")
    usage = summary.get("actual_usage")
    cost = summary.get("actual_usage_cost")
    corrected = summary.get("corrected_estimate")
    outcome_counts = (
        coverage.get("outcome_counts")
        if isinstance(coverage, Mapping)
        else None
    )
    usage_counts_valid = (
        isinstance(usage, Mapping)
        and all(
            type(usage.get(key)) is int and usage[key] >= 0
            for key in (
                "responses_with_usage",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        )
    )
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("sample_units") != REQUEST_COUNT
        or type(coverage.get("terminal_units")) is not int
        or type(coverage.get("missing_unstarted_units")) is not int
        or coverage["terminal_units"]
        + coverage["missing_unstarted_units"]
        != REQUEST_COUNT
        or not isinstance(outcome_counts, Mapping)
        or set(outcome_counts) != set(judge_protocol.ANALYSIS_OUTCOMES)
        or any(type(value) is not int or value < 0 for value in outcome_counts.values())
        or sum(outcome_counts.values()) != REQUEST_COUNT
        or coverage.get("one_pass") is not True
        or coverage.get("retries") != 0
        or not usage_counts_valid
        or usage["total_tokens"]
        != usage["input_tokens"] + usage["output_tokens"]
        or not isinstance(cost, Mapping)
        or cost != _actual_cost(usage)
        or not isinstance(corrected, Mapping)
    ):
        raise OpenAILeakageRecallError(
            "summary coverage, usage, or estimate differs"
        )
    overall = corrected.get("overall") or {}
    lower = overall.get("corrected_leakage_rate_lower")
    upper = overall.get("corrected_leakage_rate_upper")
    intervals = corrected.get("cluster_bootstrap_95_intervals") or {}
    if (
        not isinstance(lower, (int, float))
        or not isinstance(upper, (int, float))
        or not 0 <= lower <= upper <= 1
        or set(intervals)
        != {
            "corrected_leakage_rate_lower_bound_estimator",
            "corrected_leakage_rate_upper_bound_estimator",
        }
        or any(
            interval.get("K") != judge_protocol.EXPECTED_CLUSTERS
            for interval in intervals.values()
        )
    ):
        raise OpenAILeakageRecallError(
            "summary corrected rates or bootstrap intervals differ"
        )
    if has_human:
        human = summary["human_validation_secondary"]
        if (
            human.get("sample_size") != judge_protocol.EXPECTED_HUMAN_SAMPLE
            or human.get("human_raters") != 1
            or human.get("chance_correction_performed") is not False
        ):
            raise OpenAILeakageRecallError(
                "secondary human concordance differs"
            )


def _load_human_labels(path: str | Path) -> dict[int, str]:
    labels_path = _path_without_symlinks(path)
    _validate_mode(labels_path.parent, directory=True)
    _validate_mode(labels_path, directory=False)
    value = _loads_json(labels_path.read_bytes(), name="human labels")
    if not isinstance(value, Mapping):
        raise OpenAILeakageRecallError(
            "human labels must be a JSON object keyed by sample index"
        )
    labels: dict[int, str] = {}
    for key, label in value.items():
        if (
            not isinstance(key, str)
            or not key.isdigit()
            or str(int(key)) != key
            or label not in judge_protocol.LABELS
        ):
            raise OpenAILeakageRecallError("human label entry differs")
        labels[int(key)] = label
    return labels


def summarize_run(
    *,
    authorization_path: str | Path,
    sample_lock_path: str | Path,
    rubric_path: str | Path,
    local_ledger_path: str | Path,
    run_root: str | Path,
    summary_out: str | Path,
    human_labels_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reconcile interrupted starts and write a public aggregate summary."""

    sample_lock, rubric = load_public_protocol(
        sample_lock_path=sample_lock_path,
        rubric_path=rubric_path,
    )
    ledger = validate_local_ledger_file(
        local_ledger_path,
        sample_lock=sample_lock,
        rubric=rubric,
    )
    authorization = validate_authorization_file(
        authorization_path,
        sample_lock=sample_lock,
        rubric=rubric,
        local_ledger=ledger,
        sample_lock_path=sample_lock_path,
        rubric_path=rubric_path,
    )
    expected_manifest = build_run_manifest(
        authorization_path=authorization_path,
        authorization=authorization,
        sample_lock_path=sample_lock_path,
        sample_lock=sample_lock,
        rubric_path=rubric_path,
        rubric=rubric,
        local_ledger_path=local_ledger_path,
        local_ledger=ledger,
    )
    root = _path_without_symlinks(run_root)
    _validate_mode(root, directory=True)
    _validate_mode(root / "run.json", directory=False)
    manifest = _load_mapping(root / "run.json", name="run manifest")
    validate_run_manifest(manifest, expected=expected_manifest)
    _started, terminals = _finalize_incomplete_starts(
        root,
        manifest=manifest,
        sample_lock=sample_lock,
        local_ledger=ledger,
    )
    human_labels = (
        None
        if human_labels_path is None
        else _load_human_labels(human_labels_path)
    )
    summary = build_source_free_summary(
        authorization=authorization,
        sample_lock=sample_lock,
        rubric=rubric,
        local_ledger=ledger,
        manifest=manifest,
        terminals=terminals,
        human_labels=human_labels,
    )
    _atomic_write_new(summary_out, summary, local_only=False)
    return summary


def _require_arguments(
    args: argparse.Namespace,
    names: Sequence[str],
    *,
    mode: str,
) -> None:
    missing = [name.replace("_", "-") for name in names if getattr(args, name) is None]
    if missing:
        raise OpenAILeakageRecallError(
            f"{mode} requires explicit --" + ", --".join(missing)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-authorization", action="store_true")
    mode.add_argument("--prepare-local-ledger", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--sample-lock", type=Path)
    parser.add_argument("--rubric", type=Path)
    parser.add_argument("--local-ledger", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-out", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--human-labels", type=Path)
    parser.add_argument("--acknowledgement", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        common = ("sample_lock", "rubric", "local_ledger")
        if args.prepare_local_ledger:
            _require_arguments(
                args,
                (
                    "data_path",
                    "final",
                    "cohort",
                    "sample_lock",
                    "rubric",
                    "local_ledger",
                ),
                mode="--prepare-local-ledger",
            )
            prepare_local_ledger(
                data_path=args.data_path,
                final_path=args.final,
                cohort_path=args.cohort,
                sample_lock_path=args.sample_lock,
                rubric_path=args.rubric,
                ledger_out=args.local_ledger,
            )
        elif args.freeze_authorization:
            _require_arguments(
                args,
                (*common, "authorization_out"),
                mode="--freeze-authorization",
            )
            freeze_authorization(
                sample_lock_path=args.sample_lock,
                rubric_path=args.rubric,
                local_ledger_path=args.local_ledger,
                authorization_out=args.authorization_out,
            )
        elif args.run:
            _require_arguments(
                args,
                (*common, "authorization", "run_root"),
                mode="--run",
            )
            result = run_one_pass(
                authorization_path=args.authorization,
                sample_lock_path=args.sample_lock,
                rubric_path=args.rubric,
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
                    *common,
                    "authorization",
                    "run_root",
                    "summary_out",
                ),
                mode="--summarize",
            )
            summarize_run(
                authorization_path=args.authorization,
                sample_lock_path=args.sample_lock,
                rubric_path=args.rubric,
                local_ledger_path=args.local_ledger,
                run_root=args.run_root,
                summary_out=args.summary_out,
                human_labels_path=args.human_labels,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        OpenAILeakageRecallError,
        PermissionError,
        ValueError,
        judge_protocol.LeakageRecallProtocolError,
    ) as exc:
        parser.error(str(exc))
    return 0


__all__ = [
    "ACKNOWLEDGEMENT",
    "AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_STATUS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "ENDPOINT",
    "FORMAT_NAME",
    "INPUT_USD_PER_MILLION_TOKENS",
    "MAX_OUTPUT_TOKENS",
    "MODEL",
    "OUTPUT_USD_PER_MILLION_TOKENS",
    "OpenAILeakageRecallError",
    "build_authorization",
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
    "prepare_local_ledger",
    "run_one_pass",
    "summarize_run",
    "urllib_responses_transport",
    "validate_authorization",
    "validate_authorization_file",
    "validate_local_ledger",
    "validate_local_ledger_file",
    "validate_source_free_summary",
    "validate_started_marker",
    "validate_terminal",
    "write_authorization",
    "write_local_ledger",
]


if __name__ == "__main__":
    raise SystemExit(main())

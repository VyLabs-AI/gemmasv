"""Source-only lexical contamination audit for frozen LongMemEval v3 suffixes.

The audit rehydrates only the source fields named by each frozen
``session_layout`` after the owned target round. Dates and selected official
turns are scanned independently; text is never concatenated across a field,
turn, fragment, or region boundary. No model, tokenizer, API, or network
operation is present.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import longmemeval_chat_cohort_v3 as cohort_v3
from gemma_sv import longmemeval_deletion_benchmark as source
from gemma_sv import longmemeval_chat_matcher_v1 as matcher


SCHEMA = "gemma-sv-longmemeval-chat-suffix-contamination-v1"
SCHEMA_VERSION = 1
EXPECTED_CLUSTERS = 32
HISTORIES_PER_CLUSTER = 3
EXPECTED_HISTORIES = EXPECTED_CLUSTERS * HISTORIES_PER_CLUSTER
REGION_IDS = (
    "retained_evidence",
    "deterministic_tail",
    "suffix_any",
)

PACKAGE = Path(__file__).resolve().parent
DEFAULT_COHORT_PATH = (
    PACKAGE / "benchmarks" / "longmemeval_chat_cohort_v3.json"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STABLE_IDENTIFIER_RE = re.compile(
    r"longmemeval(?:-chat)?-(?:history|cluster|source)(?:-v[0-9]+)?-",
    flags=re.IGNORECASE,
)
_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "content",
        "messages",
        "prompt",
        "question",
        "record_id",
        "response_text",
        "session_id",
        "source_id",
        "source_text",
        "text",
        "token_ids",
        "tokens",
        "turns",
    }
)
_FRAGMENT_KEYS = {
    "source_id",
    "question_id",
    "session_id",
    "source_session_sha256",
    "timestamp",
    "date_text_sha256",
    "official_turn_indices",
    "answer_bearing_turn_indices",
    "official_role_sequence",
    "source_turn_sha256",
    "complete_user_assistant_rounds",
}


class SuffixAuditError(ValueError):
    """A source, cohort, rehydration, or public-report invariant drifted."""


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


def _text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise SuffixAuditError(f"duplicate JSON key {key!r}")
        value[key] = child
    return value


def _reject_constant(value: str) -> None:
    raise SuffixAuditError(f"non-finite JSON constant {value!r}")


def _load_strict_json(path: str | Path, *, name: str) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuffixAuditError(f"{name} is not strict UTF-8 JSON") from exc


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SuffixAuditError(f"{name} is not a lowercase SHA-256")
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


def _validate_integrity(value: Mapping[str, Any], *, name: str) -> None:
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
        raise SuffixAuditError(f"{name} integrity differs")


def load_pinned_source_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load the explicit local oracle path and enforce the official pin."""

    source_path = Path(path)
    if not source_path.is_file() or source_path.is_symlink():
        raise SuffixAuditError("source path must be a regular local file")
    if source_path.stat().st_size != source.DATASET_ARTIFACT_SIZE:
        raise SuffixAuditError("source artifact size differs from the pin")
    if _file_sha256(source_path) != source.DATASET_ARTIFACT_SHA256:
        raise SuffixAuditError("source artifact SHA-256 differs from the pin")
    value = _load_strict_json(source_path, name="pinned source rows")
    if (
        not isinstance(value, list)
        or len(value) != source.DATASET_NUM_ROWS
        or any(not isinstance(row, dict) for row in value)
    ):
        raise SuffixAuditError("pinned source rows have the wrong shape")
    return tuple(value)


def load_cohort(path: str | Path) -> dict[str, Any]:
    value = _load_strict_json(path, name="v3 cohort")
    if not isinstance(value, dict):
        raise SuffixAuditError("v3 cohort must be an object")
    try:
        cohort_v3.validate_manifest(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise SuffixAuditError("v3 cohort validation failed") from exc
    return value


def _source_index(
    rows: Iterable[Mapping[str, Any]],
    cohort: Mapping[str, Any],
) -> tuple[
    dict[str, source.LongMemEvalExample],
    dict[str, Mapping[str, Any]],
    tuple[source.LongMemEvalExample, ...],
]:
    raw_rows = tuple(rows)
    if any(not isinstance(row, Mapping) for row in raw_rows):
        raise SuffixAuditError("every source row must be an object")
    try:
        examples = source.extract_longmemeval_examples(raw_rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise SuffixAuditError("source-row parsing failed") from exc
    descriptors = cohort_v3.chat_v1._source_descriptors(examples)
    inventory = cohort.get("source_inventory") or {}
    if (
        len(descriptors) != inventory.get("full_row_count")
        or _payload_sha256(descriptors)
        != inventory.get("full_descriptors_sha256")
        or inventory.get("source_artifact_sha256")
        != source.DATASET_ARTIFACT_SHA256
    ):
        raise SuffixAuditError("source inventory differs from the frozen cohort")
    by_source = {example.source_id: example for example in examples}
    raw_by_source = {
        example.source_id: row
        for example, row in zip(examples, raw_rows)
    }
    if len(by_source) != len(examples) or len(raw_by_source) != len(examples):
        raise SuffixAuditError("rehydrated source identities are ambiguous")
    return by_source, raw_by_source, examples


def _normalized_role(value: str) -> str:
    folded = str(value).strip().casefold()
    return "assistant" if folded == "model" else folded


def _find_session(
    example: source.LongMemEvalExample,
    session_id: str,
) -> source.LongMemEvalSession:
    matches = [
        session
        for session in example.haystack_sessions
        if session.session_id == session_id
    ]
    if len(matches) != 1:
        raise SuffixAuditError("frozen source session is missing or ambiguous")
    return matches[0]


def _fragment_fields(
    descriptor: Mapping[str, Any],
    examples_by_source: Mapping[str, source.LongMemEvalExample],
) -> tuple[str, ...]:
    """Validate one frozen descriptor and return independent source fields."""

    if not isinstance(descriptor, Mapping) or set(descriptor) != _FRAGMENT_KEYS:
        raise SuffixAuditError("session-layout fragment fields differ")
    source_id = descriptor.get("source_id")
    if not isinstance(source_id, str) or source_id not in examples_by_source:
        raise SuffixAuditError("session-layout source binding is unavailable")
    example = examples_by_source[source_id]
    if descriptor.get("question_id") != example.question_id:
        raise SuffixAuditError("session-layout question binding differs")
    session = _find_session(example, str(descriptor.get("session_id") or ""))
    indices = descriptor.get("official_turn_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int for index in indices)
        or len(indices) % 2
        or indices != sorted(set(indices))
        or any(index < 0 or index >= len(session.turns) for index in indices)
    ):
        raise SuffixAuditError("official turn indices differ")
    for start in range(0, len(indices), 2):
        left, right = indices[start : start + 2]
        if (
            right != left + 1
            or _normalized_role(session.turns[left].role) != "user"
            or _normalized_role(session.turns[right].role) != "assistant"
        ):
            raise SuffixAuditError("official turns are not complete round pairs")

    selected = [session.turns[index] for index in indices]
    answer_indices = [
        index
        for index, turn in enumerate(session.turns)
        if turn.has_answer is True
    ]
    timestamp = (
        None
        if session.timestamp is None
        else session.timestamp.isoformat(timespec="minutes")
    )
    if (
        descriptor.get("source_session_sha256")
        != session.source_session_sha256
        or descriptor.get("timestamp") != timestamp
        or descriptor.get("date_text_sha256")
        != _text_sha256(session.date_text)
        or descriptor.get("answer_bearing_turn_indices") != answer_indices
        or descriptor.get("official_role_sequence")
        != [_normalized_role(turn.role) for turn in selected]
        or descriptor.get("source_turn_sha256")
        != [turn.source_turn_sha256 for turn in selected]
        or descriptor.get("complete_user_assistant_rounds") is not True
    ):
        raise SuffixAuditError("session-layout rehydration binding differs")
    # Date and every turn are separate scan boundaries.
    return (session.date_text, *(turn.content for turn in selected))


def scan_independent_fields(
    fields: Iterable[str],
    aliases: Sequence[str],
) -> dict[str, bool]:
    """OR field-level matches without concatenating any source strings."""

    result = {tier: False for tier in matcher.TIER_IDS}
    for field in fields:
        if not isinstance(field, str):
            raise SuffixAuditError("scan field must be text")
        observed = matcher.directional_disclosure_matches(field, aliases)
        for tier in matcher.TIER_IDS:
            result[tier] = result[tier] or observed[tier]
    return result


def _combine_regions(*regions: Mapping[str, bool]) -> dict[str, bool]:
    return {
        tier: any(bool(region[tier]) for region in regions)
        for tier in matcher.TIER_IDS
    }


def _history_scan(
    history: Mapping[str, Any],
    *,
    cluster_index: int,
    history_index: int,
    examples_by_source: Mapping[str, source.LongMemEvalExample],
    raw_by_source: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    target = history.get("target")
    if not isinstance(target, Mapping):
        raise SuffixAuditError("history target binding is missing")
    source_id = target.get("source_id")
    if not isinstance(source_id, str) or source_id not in examples_by_source:
        raise SuffixAuditError("target source binding is unavailable")
    example = examples_by_source[source_id]
    try:
        cohort_v3.chat_v1._check_example_descriptor(
            example,
            target,
            retained=False,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SuffixAuditError("target source descriptor differs") from exc
    aliases = matcher.source_aliases(
        example.answer,
        matcher.explicit_official_aliases(raw_by_source[source_id]),
    )

    layout = history.get("session_layout")
    if not isinstance(layout, Mapping):
        raise SuffixAuditError("history session_layout is missing")
    retained = layout.get("retained_evidence")
    tail = layout.get("deterministic_tail")
    if (
        not isinstance(retained, list)
        or not retained
        or not isinstance(tail, list)
        or not tail
    ):
        raise SuffixAuditError("post-owned suffix regions are incomplete")
    retained_fields = tuple(
        field
        for descriptor in retained
        for field in _fragment_fields(descriptor, examples_by_source)
    )
    tail_fields = tuple(
        field
        for descriptor in tail
        for field in _fragment_fields(descriptor, examples_by_source)
    )
    retained_matches = scan_independent_fields(retained_fields, aliases)
    tail_matches = scan_independent_fields(tail_fields, aliases)
    return {
        "cluster_index": cluster_index,
        "history_index": history_index,
        "variant_index": int(history["variant_index"]),
        "regions": {
            "retained_evidence": retained_matches,
            "deterministic_tail": tail_matches,
            "suffix_any": _combine_regions(retained_matches, tail_matches),
        },
    }


def _summaries(
    histories: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_cluster: list[dict[str, Any]] = []
    for cluster_index in range(EXPECTED_CLUSTERS):
        rows = [
            row
            for row in histories
            if row.get("cluster_index") == cluster_index
        ]
        if (
            len(rows) != HISTORIES_PER_CLUSTER
            or [row.get("variant_index") for row in rows] != [0, 1, 2]
        ):
            raise SuffixAuditError("anonymous cluster geometry differs")
        per_cluster.append(
            {
                "cluster_index": cluster_index,
                "history_count": HISTORIES_PER_CLUSTER,
                "counts": {
                    region: {
                        tier: sum(
                            bool(row["regions"][region][tier]) for row in rows
                        )
                        for tier in matcher.TIER_IDS
                    }
                    for region in REGION_IDS
                },
            }
        )

    aggregate = {
        "K": EXPECTED_CLUSTERS,
        "n": EXPECTED_HISTORIES,
        "histories_per_cluster": HISTORIES_PER_CLUSTER,
        "counts": {
            region: {
                tier: {
                    "positive_histories": sum(
                        bool(row["regions"][region][tier])
                        for row in histories
                    ),
                    "history_denominator": EXPECTED_HISTORIES,
                    "positive_clusters_any_history": sum(
                        item["counts"][region][tier] > 0
                        for item in per_cluster
                    ),
                    "cluster_denominator": EXPECTED_CLUSTERS,
                }
                for tier in matcher.TIER_IDS
            }
            for region in REGION_IDS
        },
    }
    return per_cluster, aggregate


def build_audit(
    *,
    source_rows: Iterable[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    source_file_sha256: str,
    cohort_file_sha256: str,
) -> dict[str, Any]:
    """Validate and scan all 96 histories, returning a source-free report."""

    rows = tuple(source_rows)
    _require_sha256(source_file_sha256, name="source file")
    _require_sha256(cohort_file_sha256, name="cohort file")
    if source_file_sha256 != source.DATASET_ARTIFACT_SHA256:
        raise SuffixAuditError("source file does not match the official pin")
    try:
        cohort_v3.validate_manifest(cohort)
    except (KeyError, TypeError, ValueError) as exc:
        raise SuffixAuditError("cohort validation failed") from exc
    if cohort.get("counts") != {
        "target_clusters": EXPECTED_CLUSTERS,
        "histories_per_cluster": HISTORIES_PER_CLUSTER,
        "nested_history_instances": EXPECTED_HISTORIES,
        "independent_analysis_n": EXPECTED_CLUSTERS,
    }:
        raise SuffixAuditError("cohort K/n geometry differs")

    examples_by_source, raw_by_source, examples = _source_index(
        rows,
        cohort,
    )
    clusters = cohort["clusters"]
    cluster_index_by_id = {
        str(cluster["cluster_id"]): int(cluster["cluster_index"])
        for cluster in clusters
    }
    if (
        len(cluster_index_by_id) != EXPECTED_CLUSTERS
        or set(cluster_index_by_id.values()) != set(range(EXPECTED_CLUSTERS))
    ):
        raise SuffixAuditError("cohort cluster indexing differs")

    histories: list[dict[str, Any]] = []
    for history_index, history in enumerate(cohort["histories"]):
        cluster_id = str(history.get("cluster_id") or "")
        if cluster_id not in cluster_index_by_id:
            raise SuffixAuditError("history references an unknown cluster")
        histories.append(
            _history_scan(
                history,
                cluster_index=cluster_index_by_id[cluster_id],
                history_index=history_index,
                examples_by_source=examples_by_source,
                raw_by_source=raw_by_source,
            )
        )
    if len(histories) != EXPECTED_HISTORIES:
        raise SuffixAuditError("not all 96 histories were scanned")
    per_cluster, aggregate = _summaries(histories)

    report = _seal(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "complete-source-only",
            "contains_source_text": False,
            "contains_source_identifiers": False,
            "contains_model_generated_text": False,
            "contains_token_arrays": False,
            "model_or_api_calls_made": 0,
            "artifacts": {
                "pinned_source_rows": {
                    "file_sha256": source_file_sha256,
                    "payload_sha256": _payload_sha256(list(rows)),
                    "row_count": len(examples),
                    "official_pin_verified": True,
                },
                "cohort": {
                    "file_sha256": cohort_file_sha256,
                    "payload_sha256": _payload_sha256(cohort),
                    "integrity_sha256": cohort["integrity"]["sha256"],
                },
            },
            "rehydration": {
                "binding": "frozen_session_layout",
                "full_source_inventory_validated": True,
                "all_post_owned_fragment_bindings_validated": True,
                "scanned_regions": list(REGION_IDS),
                "dates_and_turns_scanned_as_separate_fields": True,
                "cross_boundary_concatenation": False,
            },
            "matcher": {
                "tiers": list(matcher.TIER_IDS),
                "direction": "suffix_field_contains_source_alias",
                "normalization_reused_from_decoded_summary_v2": True,
                "alias_scope": (
                    "raw official answer plus explicit official aliases only"
                ),
                "semantic_synonyms_fabricated": False,
            },
            "histories": histories,
            "per_cluster": per_cluster,
            "aggregate": aggregate,
            "interpretation": {
                "tail_support_sources_are_independent": True,
                "tail_support_source_scope": (
                    "independent fresh support sources, not target sources"
                ),
                "lexical_cleanliness_implies_causal_independence": False,
                "limitation": (
                    "Tails are independent support sources. Absence of a "
                    "declared lexical match is lexical cleanliness only and "
                    "does not establish causal independence."
                ),
            },
        }
    )
    validate_audit(report)
    return report


def _walk(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk(child)


def assert_source_free(report: Mapping[str, Any]) -> None:
    """Reject source/model text, stable identifiers, and token arrays."""

    for key, child in _walk(report):
        if key is not None:
            folded = key.casefold()
            if folded in _FORBIDDEN_OUTPUT_KEYS:
                raise SuffixAuditError(
                    f"public audit contains prohibited key {key!r}"
                )
            if folded in {
                "contains_source_text",
                "contains_source_identifiers",
                "contains_model_generated_text",
                "contains_token_arrays",
            } and child is not False:
                raise SuffixAuditError(
                    f"public audit declaration {key!r} must be false"
                )
        if isinstance(child, str):
            if _STABLE_IDENTIFIER_RE.search(child):
                raise SuffixAuditError("public audit contains a stable identifier")
            if any(
                marker in child
                for marker in ("<bos>", "<start_of_turn>", "<end_of_turn>")
            ):
                raise SuffixAuditError("public audit contains serialized chat text")


def _validate_tier_map(value: Any, *, name: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(matcher.TIER_IDS)
        or any(type(value[tier]) is not bool for tier in matcher.TIER_IDS)
    ):
        raise SuffixAuditError(f"{name} tier booleans differ")
    if value["deterministic_any"] is not any(
        value[tier] for tier in matcher.TIER_IDS[:-1]
    ):
        raise SuffixAuditError(f"{name} deterministic union differs")


def validate_audit(report: Mapping[str, Any]) -> None:
    """Strictly validate a sealed source-free contamination report."""

    if not isinstance(report, Mapping):
        raise SuffixAuditError("public audit must be an object")
    assert_source_free(report)
    _validate_integrity(report, name="suffix contamination audit")
    expected_top = {
        "schema",
        "schema_version",
        "status",
        "contains_source_text",
        "contains_source_identifiers",
        "contains_model_generated_text",
        "contains_token_arrays",
        "model_or_api_calls_made",
        "artifacts",
        "rehydration",
        "matcher",
        "histories",
        "per_cluster",
        "aggregate",
        "interpretation",
        "integrity",
    }
    if set(report) != expected_top:
        raise SuffixAuditError("public audit top-level fields differ")
    if (
        report.get("schema") != SCHEMA
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("status") != "complete-source-only"
        or report.get("contains_source_text") is not False
        or report.get("contains_source_identifiers") is not False
        or report.get("contains_model_generated_text") is not False
        or report.get("contains_token_arrays") is not False
        or report.get("model_or_api_calls_made") != 0
    ):
        raise SuffixAuditError("public audit schema or source-only flags differ")

    histories = report.get("histories")
    if not isinstance(histories, list) or len(histories) != EXPECTED_HISTORIES:
        raise SuffixAuditError("public audit must contain all 96 histories")
    for expected_index, row in enumerate(histories):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "cluster_index",
                "history_index",
                "variant_index",
                "regions",
            }
            or row.get("history_index") != expected_index
            or type(row.get("cluster_index")) is not int
            or not 0 <= row["cluster_index"] < EXPECTED_CLUSTERS
            or type(row.get("variant_index")) is not int
            or not 0 <= row["variant_index"] < HISTORIES_PER_CLUSTER
            or not isinstance(row.get("regions"), Mapping)
            or set(row["regions"]) != set(REGION_IDS)
        ):
            raise SuffixAuditError("anonymous history row differs")
        for region in REGION_IDS:
            _validate_tier_map(
                row["regions"][region],
                name=f"history {expected_index}/{region}",
            )
        for tier in matcher.TIER_IDS:
            if row["regions"]["suffix_any"][tier] is not (
                row["regions"]["retained_evidence"][tier]
                or row["regions"]["deterministic_tail"][tier]
            ):
                raise SuffixAuditError("suffix region union differs")

    expected_per_cluster, expected_aggregate = _summaries(histories)
    if report.get("per_cluster") != expected_per_cluster:
        raise SuffixAuditError("per-cluster counts differ")
    if report.get("aggregate") != expected_aggregate:
        raise SuffixAuditError("K=32 aggregate differs")

    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "pinned_source_rows",
        "cohort",
    }:
        raise SuffixAuditError("artifact bindings differ")
    for section in artifacts.values():
        if not isinstance(section, Mapping):
            raise SuffixAuditError("artifact binding must be an object")
        for key, value in section.items():
            if key.endswith("_sha256"):
                _require_sha256(value, name=f"artifact {key}")
    interpretation = report.get("interpretation") or {}
    if (
        interpretation.get("tail_support_sources_are_independent") is not True
        or interpretation.get(
            "lexical_cleanliness_implies_causal_independence"
        )
        is not False
    ):
        raise SuffixAuditError("causal-independence disclosure differs")


def audit_paths(
    *,
    data_path: str | Path,
    cohort_path: str | Path = DEFAULT_COHORT_PATH,
) -> dict[str, Any]:
    """Run the source-only audit from explicit local data and cohort paths."""

    data = Path(data_path)
    cohort_file = Path(cohort_path)
    rows = load_pinned_source_rows(data)
    cohort = load_cohort(cohort_file)
    return build_audit(
        source_rows=rows,
        cohort=cohort,
        source_file_sha256=_file_sha256(data),
        cohort_file_sha256=_file_sha256(cohort_file),
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
        raise SuffixAuditError("output path must not use a symbolic link")
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
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="explicit local exact pinned LongMemEval oracle JSON",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT_PATH,
        help="frozen v3 cohort JSON",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="optional new output path; overwrite is forbidden",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit_paths(
            data_path=args.data_path,
            cohort_path=args.cohort,
        )
        if args.out is None:
            sys.stdout.write(deterministic_json(report))
        else:
            _write_new(args.out, report)
            print(f"wrote {args.out}")
    except (FileExistsError, OSError, SuffixAuditError) as exc:
        _parser().error(str(exc))
    return 0


__all__ = [
    "DEFAULT_COHORT_PATH",
    "EXPECTED_CLUSTERS",
    "EXPECTED_HISTORIES",
    "HISTORIES_PER_CLUSTER",
    "REGION_IDS",
    "SCHEMA",
    "SuffixAuditError",
    "assert_source_free",
    "audit_paths",
    "build_audit",
    "deterministic_json",
    "load_cohort",
    "load_pinned_source_rows",
    "scan_independent_fields",
    "validate_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze and rehydrate a source-free MemOps longitudinal deletion benchmark.

The benchmark adapts MemOps ``Remember`` trajectories to a stricter external
erasure intervention:

1. ingest a naturalistic multi-session dialogue;
2. query one user fact after at least 10k subsequent model tokens;
3. delete the complete user/assistant exchange that introduced that fact;
4. compare with a freshly retokenized raw-exchange-omitted rebuild; and
5. query a different fact introduced in the same evidence segment.

Selection uses only pinned source artifacts and tokenizer geometry.  No Gemma
outputs are consulted.  The saved manifest contains hashes, source identifiers,
and token ownership, but no dialogue, question, or answer text.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv.demo_server.certificate import fixed_c_feasibility
from gemma_sv.rag_benchmark import TokenizedContext


SCHEMA = "gemma-sv-memops-longitudinal-erasure-v1"
SCHEMA_VERSION = 1
MEMOPS_REPOSITORY = "MemTensor/MemOps"
MEMOPS_REVISION = "312af65e2c7b6d1b70f062ffa8b4cde32aaf6f35"
MEMOPS_LICENSE = "MIT"
EVIDENCE_DIRECTORY = Path("generated_result/2-evidence_conversation")
LONGITUDINAL_DIRECTORY = Path(
    "generated_result/4-inject_evidence_with_distractors"
)
DEFAULT_TOKENIZER_ID = "google/gemma-3-4b-pt"
DEFAULT_TOKENIZER_REVISION = "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
DEFAULT_RECORDS = 64
DEFAULT_SEED = 314159
DEFAULT_MINIMUM_TOKENS_BEFORE_OWNED = 512
DEFAULT_MINIMUM_TOKENS_AFTER_OWNED = 10_000
DEFAULT_MAXIMUM_CONTEXT_TOKENS = 16_384
DEFAULT_MAXIMUM_TARGET_TOKENS = 64
DEFAULT_NU = 0.7
DEFAULT_CHUNK = 128


class ManifestError(ValueError):
    """A source artifact or frozen manifest violates the benchmark contract."""


@dataclass(frozen=True)
class LongitudinalProbe:
    """One runtime-only target or retained-memory probe."""

    probe_id: str
    kind: str
    question: str
    target_name: str
    target: str
    target_token_ids: tuple[int, ...] = ()

    @property
    def prompt(self) -> str:
        return f"\n\nQuestion: {self.question}\nAnswer:"


@dataclass(frozen=True)
class LongitudinalRecord:
    """One validated runtime record containing source text."""

    record_id: str
    source_file: str
    target_operation_id: str
    retained_operation_id: str
    context: TokenizedContext
    raw_omitted_token_ids: tuple[int, ...]
    probes: tuple[LongitudinalProbe, ...]
    target_value_literal_in_owned: bool
    target_value_occurrences_inside_owned: int
    target_value_occurrences_outside_owned: int
    retained_value_literal_in_retained_exchange: bool
    retained_value_occurrences_inside_owned: int
    retained_value_occurrences_outside_owned: int
    source_session_start_index: int
    source_session_end_index: int


@dataclass(frozen=True)
class _FormattedHistory:
    text: str
    message_spans: Mapping[tuple[int, int], tuple[int, int]]
    session_start_characters: tuple[int, ...]
    session_end_characters: tuple[int, ...]


def text_sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<q", int(token_id)))
    return digest.hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def stable_identifier(namespace: str, *parts: object) -> str:
    digest = hashlib.sha256()
    digest.update(str(namespace).encode("utf-8"))
    digest.update(b"\0")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return f"{namespace}-{digest.hexdigest()[:24]}"


def _stable_key(namespace: str, seed: int, *parts: object) -> bytes:
    framed = "\0".join(
        (namespace, str(int(seed)), *(str(part) for part in parts))
    )
    return hashlib.sha256(framed.encode("utf-8")).digest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError(f"{path} is not a JSON object")
    return payload


def _tokenizer_payload(
    tokenizer: Any,
    text: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ManifestError("a fast tokenizer with offset mapping is required")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_ids = (
        encoded["input_ids"]
        if isinstance(encoded, Mapping)
        else encoded.input_ids
    )
    raw_offsets = (
        encoded["offset_mapping"]
        if isinstance(encoded, Mapping)
        else encoded.offset_mapping
    )
    token_ids = tuple(int(token_id) for token_id in raw_ids)
    offsets = tuple((int(start), int(end)) for start, end in raw_offsets)
    if len(token_ids) != len(offsets):
        raise ManifestError("token IDs and offset mapping have different lengths")
    if any(start < 0 or end < start for start, end in offsets):
        raise ManifestError("tokenizer returned an invalid offset")
    return token_ids, offsets


def _overlaps(
    token_offset: tuple[int, int],
    character_span: tuple[int, int],
) -> bool:
    token_start, token_end = token_offset
    span_start, span_end = character_span
    return token_end > token_start and token_end > span_start and token_start < span_end


def _positions_for_span(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
) -> tuple[int, ...]:
    return tuple(
        index for index, offset in enumerate(offsets) if _overlaps(offset, span)
    )


def _contiguous_ranges(
    positions: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    values = tuple(sorted({int(position) for position in positions}))
    if not values:
        raise ManifestError("owned character span maps to no model tokens")
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for position in values[1:]:
        if position != previous + 1:
            ranges.append((start, previous + 1))
            start = position
        previous = position
    ranges.append((start, previous + 1))
    return tuple(ranges)


def _remove_positions(
    token_ids: Sequence[int],
    positions: Sequence[int],
) -> tuple[int, ...]:
    forgotten = {int(position) for position in positions}
    return tuple(
        int(token_id)
        for index, token_id in enumerate(token_ids)
        if index not in forgotten
    )


def _shift_positions(
    positions: Sequence[int],
    deleted_positions: Sequence[int],
) -> tuple[int, ...]:
    deleted = tuple(sorted({int(position) for position in deleted_positions}))
    deleted_set = set(deleted)
    shifted = []
    for raw_position in positions:
        position = int(raw_position)
        if position in deleted_set:
            raise ManifestError("retained and owned token positions overlap")
        shifted.append(
            position - sum(deleted_position < position for deleted_position in deleted)
        )
    return tuple(shifted)


def _format_history(conversations: Sequence[Mapping[str, Any]]) -> _FormattedHistory:
    pieces: list[str] = []
    message_spans: dict[tuple[int, int], tuple[int, int]] = {}
    session_starts: list[int] = []
    session_ends: list[int] = []
    cursor = 0
    for session_index, conversation in enumerate(conversations):
        session_starts.append(cursor)
        header = f"\n=== Session {session_index + 1:03d} ===\n"
        pieces.append(header)
        cursor += len(header)
        dialogue = conversation.get("dialogue")
        if not isinstance(dialogue, Sequence) or isinstance(
            dialogue, (str, bytes)
        ):
            raise ManifestError("MemOps conversation has no dialogue sequence")
        for turn_index, message in enumerate(dialogue):
            if not isinstance(message, Mapping):
                raise ManifestError("MemOps dialogue turn is not an object")
            role = str(message.get("role") or "").strip().casefold()
            if role not in {"user", "assistant"}:
                raise ManifestError(f"unsupported MemOps role {role!r}")
            content = str(message.get("content") or "").strip()
            if not content:
                raise ManifestError("MemOps dialogue turn is empty")
            rendered_role = "User" if role == "user" else "Assistant"
            rendered = f"{rendered_role}: {content}\n"
            start = cursor
            pieces.append(rendered)
            cursor += len(rendered)
            message_spans[(session_index, turn_index)] = (start, cursor)
        session_ends.append(cursor)
    return _FormattedHistory(
        text="".join(pieces),
        message_spans=message_spans,
        session_start_characters=tuple(session_starts),
        session_end_characters=tuple(session_ends),
    )


def _confirmed_remember_operations(
    evidence: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    operations = evidence.get("operations")
    if not isinstance(operations, Sequence) or isinstance(
        operations, (str, bytes)
    ):
        raise ManifestError("MemOps evidence has no operations sequence")
    result = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        if str(operation.get("type") or "").casefold() != "remember":
            continue
        if str(operation.get("validity") or "confirmed").casefold() != "confirmed":
            continue
        trigger = operation.get("trigger_span")
        target = operation.get("target")
        if not isinstance(trigger, Mapping) or not isinstance(target, Mapping):
            continue
        if not str(operation.get("operation_id") or "").strip():
            continue
        if not str(operation.get("new_value") or "").strip():
            continue
        if not str(trigger.get("quote") or "").strip():
            continue
        result.append(operation)
    return tuple(result)


def _operation_id(operation: Mapping[str, Any]) -> str:
    return str(operation["operation_id"])


def _trigger_key(operation: Mapping[str, Any]) -> tuple[int, int]:
    trigger = operation["trigger_span"]
    return int(trigger["segment_index"]), int(trigger["turn_index"])


def _candidate_pairs(
    operations: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    source_file: str,
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    pairs = []
    for target in operations:
        target_segment, target_turn = _trigger_key(target)
        for retained in operations:
            if retained is target:
                continue
            if str(target["target"].get("target_id")) == str(
                retained["target"].get("target_id")
            ):
                continue
            retained_segment, retained_turn = _trigger_key(retained)
            if target_segment != retained_segment or target_turn == retained_turn:
                continue
            pairs.append((target, retained))
    return tuple(
        sorted(
            pairs,
            key=lambda pair: (
                int(pair[0]["trigger_span"]["segment_index"]),
                (
                    str(pair[0]["new_value"]).casefold()
                    not in str(pair[0]["trigger_span"]["quote"]).casefold()
                ),
                len(str(pair[0]["new_value"])),
                _stable_key(
                    "memops-pair-v1",
                    seed,
                    source_file,
                    _operation_id(pair[0]),
                    _operation_id(pair[1]),
                ),
                _operation_id(pair[0]),
                _operation_id(pair[1]),
            ),
        )
    )


def _find_trigger_exchange(
    longitudinal: Mapping[str, Any],
    operation: Mapping[str, Any],
    history: _FormattedHistory,
) -> tuple[int, int, tuple[int, int]]:
    trigger = operation["trigger_span"]
    quote = str(trigger["quote"])
    segment_index = int(trigger["segment_index"])
    conversations = longitudinal.get("conversations")
    if not isinstance(conversations, Sequence) or isinstance(
        conversations, (str, bytes)
    ):
        raise ManifestError("MemOps longitudinal artifact has no conversations")
    matches = []
    for session_index, conversation in enumerate(conversations):
        if int(conversation.get("evidence_segment_index") or -1) != segment_index:
            continue
        dialogue = conversation.get("dialogue") or ()
        for turn_index, message in enumerate(dialogue):
            if str(message.get("role") or "").casefold() != "user":
                continue
            if quote not in str(message.get("content") or ""):
                continue
            if turn_index + 1 >= len(dialogue):
                raise ManifestError("evidence user turn has no assistant response")
            response = dialogue[turn_index + 1]
            if str(response.get("role") or "").casefold() != "assistant":
                raise ManifestError("evidence user turn is not followed by assistant")
            start = history.message_spans[(session_index, turn_index)][0]
            end = history.message_spans[(session_index, turn_index + 1)][1]
            matches.append((session_index, turn_index, (start, end)))
    if len(matches) != 1:
        raise ManifestError(
            f"operation {_operation_id(operation)} maps to {len(matches)} exchanges"
        )
    return matches[0]


def _casefold_occurrences(text: str, value: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        match.span()
        for match in re.finditer(re.escape(value), text, flags=re.IGNORECASE)
    )


def _span_contains(
    outer: tuple[int, int],
    inner: tuple[int, int],
) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _question(target_name: str, *, paraphrase: bool = False) -> str:
    if paraphrase:
        return f"Recall the detail I shared concerning {target_name}."
    return f"What did I tell you about {target_name}?"


def _prompt_conditioned_target_ids(
    tokenizer: Any,
    prompt: str,
    target: str,
) -> tuple[int, ...]:
    rendered = str(target)
    if prompt and not prompt[-1].isspace() and not rendered[:1].isspace():
        rendered = " " + rendered
    token_ids, _ = _tokenizer_payload(tokenizer, rendered)
    return token_ids


def _operation_values(
    operation: Mapping[str, Any],
) -> tuple[str, str]:
    target = operation["target"]
    name = str(target["target_name"]).strip()
    value = str(operation["new_value"]).strip()
    if not name or not value:
        raise ManifestError("operation target name/value is empty")
    return name, value


def _first_occurrence_within(
    text: str,
    value: str,
    container: tuple[int, int],
) -> tuple[int, int]:
    matches = [
        span
        for span in _casefold_occurrences(text, value)
        if _span_contains(container, span)
    ]
    if not matches:
        raise ManifestError("operation value is not literal inside its exchange")
    return matches[0]


def _context_end(
    history: _FormattedHistory,
    offsets: Sequence[tuple[int, int]],
    owned_positions: Sequence[int],
    retained_span: tuple[int, int],
    *,
    minimum_tokens_after_owned: int,
) -> int:
    required_token_index = (
        max(int(position) for position in owned_positions)
        + int(minimum_tokens_after_owned)
    )
    if required_token_index >= len(offsets):
        raise ManifestError("not enough longitudinal tokens after target exchange")
    required_character = max(
        int(retained_span[1]),
        int(offsets[required_token_index][1]),
    )
    for session_end in history.session_end_characters:
        if session_end >= required_character:
            return int(session_end)
    raise ManifestError("no complete-session cutoff satisfies longitudinal geometry")


def _context_start(
    history: _FormattedHistory,
    tokenizer: Any,
    target_session_index: int,
    owned_span: tuple[int, int],
    *,
    minimum_tokens_before_owned: int,
) -> tuple[int, int]:
    candidates = range(int(target_session_index), -1, -1)
    for session_index in candidates:
        start = int(history.session_start_characters[session_index])
        prefix_ids, _ = _tokenizer_payload(
            tokenizer,
            history.text[start : owned_span[0]],
        )
        if len(prefix_ids) >= int(minimum_tokens_before_owned):
            return session_index, start
    raise ManifestError("not enough model tokens before target exchange")


def _make_probes(
    target_operation: Mapping[str, Any],
    retained_operation: Mapping[str, Any],
) -> tuple[LongitudinalProbe, ...]:
    target_name, target_value = _operation_values(target_operation)
    retained_name, retained_value = _operation_values(retained_operation)
    return (
        LongitudinalProbe(
            probe_id="deleted_direct",
            kind="deleted",
            question=_question(target_name),
            target_name=target_name,
            target=target_value,
        ),
        LongitudinalProbe(
            probe_id="deleted_paraphrase",
            kind="deleted",
            question=_question(target_name, paraphrase=True),
            target_name=target_name,
            target=target_value,
        ),
        LongitudinalProbe(
            probe_id="retained_direct",
            kind="retained",
            question=_question(retained_name),
            target_name=retained_name,
            target=retained_value,
        ),
    )


def _prepare_record(
    tokenizer: Any,
    evidence: Mapping[str, Any],
    longitudinal: Mapping[str, Any],
    *,
    source_file: str,
    target_operation: Mapping[str, Any],
    retained_operation: Mapping[str, Any],
    minimum_tokens_before_owned: int,
    minimum_tokens_after_owned: int,
    maximum_context_tokens: int,
    maximum_target_tokens: int,
    nu: float,
    chunk: int,
    tokenizer_id: str,
    tokenizer_revision: str,
) -> LongitudinalRecord:
    if str(evidence.get("operation_type") or "").casefold() != "remember":
        raise ManifestError("only MemOps Remember trajectories are eligible")
    if str(longitudinal.get("operation_type") or "").casefold() != "remember":
        raise ManifestError("longitudinal/evidence operation types differ")
    if evidence.get("target_fact") != longitudinal.get("target_fact"):
        raise ManifestError("longitudinal/evidence target facts differ")

    raw_probes = _make_probes(target_operation, retained_operation)
    probes = tuple(
        LongitudinalProbe(
            probe_id=probe.probe_id,
            kind=probe.kind,
            question=probe.question,
            target_name=probe.target_name,
            target=probe.target,
            target_token_ids=_prompt_conditioned_target_ids(
                tokenizer,
                probe.prompt,
                probe.target,
            ),
        )
        for probe in raw_probes
    )
    for probe in probes:
        if probe.target.casefold() in probe.question.casefold():
            raise ManifestError(f"{probe.probe_id} question leaks its target")
        if not 1 <= len(probe.target_token_ids) <= int(maximum_target_tokens):
            raise ManifestError(f"{probe.probe_id} target token count is out of range")

    conversations = longitudinal.get("conversations")
    if not isinstance(conversations, Sequence) or isinstance(
        conversations, (str, bytes)
    ):
        raise ManifestError("MemOps longitudinal artifact has no conversations")
    history = _format_history(conversations)
    target_session_index, _, owned_span = _find_trigger_exchange(
        longitudinal,
        target_operation,
        history,
    )
    _, _, retained_exchange = _find_trigger_exchange(
        longitudinal,
        retained_operation,
        history,
    )
    if not (owned_span[1] <= retained_exchange[0] or retained_exchange[1] <= owned_span[0]):
        raise ManifestError("target and retained exchanges overlap")

    target_name, target_value = _operation_values(target_operation)
    retained_name, retained_value = _operation_values(retained_operation)
    del target_name, retained_name
    target_quote_span = _first_occurrence_within(
        history.text,
        str(target_operation["trigger_span"]["quote"]),
        owned_span,
    )
    retained_quote_span = _first_occurrence_within(
        history.text,
        str(retained_operation["trigger_span"]["quote"]),
        retained_exchange,
    )
    full_ids, full_offsets = _tokenizer_payload(tokenizer, history.text)
    owned_full_positions = _positions_for_span(full_offsets, owned_span)
    if not owned_full_positions:
        raise ManifestError("owned exchange maps to no full-history tokens")
    end_character = _context_end(
        history,
        full_offsets,
        owned_full_positions,
        retained_exchange,
        minimum_tokens_after_owned=minimum_tokens_after_owned,
    )
    start_session_index, start_character = _context_start(
        history,
        tokenizer,
        target_session_index,
        owned_span,
        minimum_tokens_before_owned=minimum_tokens_before_owned,
    )
    try:
        end_session_index = history.session_end_characters.index(end_character)
    except ValueError as exc:
        raise ManifestError("context end is not a session boundary") from exc
    context_text = history.text[start_character:end_character]
    owned_span = (
        owned_span[0] - start_character,
        owned_span[1] - start_character,
    )
    target_quote_span = (
        target_quote_span[0] - start_character,
        target_quote_span[1] - start_character,
    )
    retained_quote_span = (
        retained_quote_span[0] - start_character,
        retained_quote_span[1] - start_character,
    )
    target_occurrences = _casefold_occurrences(context_text, target_value)
    inside_occurrences = sum(
        _span_contains(owned_span, span) for span in target_occurrences
    )
    outside_occurrences = len(target_occurrences) - inside_occurrences
    retained_occurrences = _casefold_occurrences(context_text, retained_value)
    retained_inside_owned = sum(
        _span_contains(owned_span, span) for span in retained_occurrences
    )
    retained_inside_exchange = sum(
        _span_contains(retained_quote_span, span)
        for span in retained_occurrences
    )
    if retained_inside_owned:
        raise ManifestError("retained value occurs inside owned exchange")
    retained_outside_owned = len(retained_occurrences) - retained_inside_owned
    token_ids, offsets = _tokenizer_payload(tokenizer, context_text)
    forget_positions = _positions_for_span(offsets, owned_span)
    retained_positions = _positions_for_span(offsets, retained_quote_span)
    if not forget_positions or not retained_positions:
        raise ManifestError("target or retained span was lost at context cutoff")
    if min(forget_positions) < int(minimum_tokens_before_owned):
        raise ManifestError("not enough model tokens before target exchange")
    tokens_after_owned = len(token_ids) - max(forget_positions) - 1
    if tokens_after_owned < int(minimum_tokens_after_owned):
        raise ManifestError("not enough model tokens after target exchange")
    if len(token_ids) > int(maximum_context_tokens):
        raise ManifestError("minimum longitudinal context exceeds token cap")

    _, feasibility = fixed_c_feasibility(
        len(token_ids),
        forget_positions,
        nu=float(nu),
        chunk=int(chunk),
        per_boundary_box=True,
    )
    if any(not bool(item["feasible"]) for item in feasibility):
        raise ManifestError("owned exchange is not fixed-C feasible")

    edited_text = context_text[: owned_span[0]] + context_text[owned_span[1] :]
    edited_token_ids = _remove_positions(token_ids, forget_positions)
    raw_omitted_token_ids, _ = _tokenizer_payload(tokenizer, edited_text)
    edited_retained_positions = _shift_positions(
        retained_positions,
        forget_positions,
    )
    before = tuple(token_ids[position] for position in retained_positions)
    after = tuple(
        edited_token_ids[position] for position in edited_retained_positions
    )
    if before != after:
        raise ManifestError("literal edit does not preserve retained value tokens")

    context = TokenizedContext(
        original_text=context_text,
        edited_text=edited_text,
        original_token_ids=token_ids,
        edited_token_ids=edited_token_ids,
        offset_mapping=offsets,
        forget_positions=forget_positions,
        deletion_ranges=_contiguous_ranges(forget_positions),
        retained_positions=retained_positions,
        edited_retained_positions=edited_retained_positions,
        owned_character_span=owned_span,
        answer_character_span=target_quote_span,
        retained_character_span=retained_quote_span,
    )
    record_id = stable_identifier(
        "memops",
        MEMOPS_REVISION,
        source_file,
        _operation_id(target_operation),
        _operation_id(retained_operation),
        tokenizer_id,
        tokenizer_revision,
        minimum_tokens_after_owned,
    )
    return LongitudinalRecord(
        record_id=record_id,
        source_file=source_file,
        target_operation_id=_operation_id(target_operation),
        retained_operation_id=_operation_id(retained_operation),
        context=context,
        raw_omitted_token_ids=raw_omitted_token_ids,
        probes=probes,
        target_value_literal_in_owned=bool(inside_occurrences),
        target_value_occurrences_inside_owned=int(inside_occurrences),
        target_value_occurrences_outside_owned=int(outside_occurrences),
        retained_value_literal_in_retained_exchange=bool(
            retained_inside_exchange
        ),
        retained_value_occurrences_inside_owned=int(retained_inside_owned),
        retained_value_occurrences_outside_owned=int(retained_outside_owned),
        source_session_start_index=int(start_session_index),
        source_session_end_index=int(end_session_index),
    )


def _context_descriptor(
    context: TokenizedContext,
    raw_omitted_token_ids: Sequence[int],
) -> dict[str, Any]:
    owned_offsets = [
        {
            "position": int(position),
            "start": int(context.offset_mapping[position][0]),
            "end": int(context.offset_mapping[position][1]),
        }
        for position in context.forget_positions
    ]
    return {
        "package_schema": "gemma-sv-memops-context-v1",
        "original_text_sha256": text_sha256(context.original_text),
        "edited_text_sha256": text_sha256(context.edited_text),
        "original_token_count": len(context.original_token_ids),
        "edited_token_count": len(context.edited_token_ids),
        "original_token_ids_sha256": token_ids_sha256(
            context.original_token_ids
        ),
        "edited_token_ids_sha256": token_ids_sha256(context.edited_token_ids),
        "retained_key_row_edit": {
            "token_count": len(context.edited_token_ids),
            "token_ids_sha256": token_ids_sha256(context.edited_token_ids),
            "constructed_by_removing_original_token_rows": True,
        },
        "raw_omitted_repack": {
            "text_sha256": text_sha256(context.edited_text),
            "token_count": len(raw_omitted_token_ids),
            "token_ids_sha256": token_ids_sha256(raw_omitted_token_ids),
            "freshly_retokenized": True,
        },
        "literal_token_edit": True,
        "tokens_before_owned_span": min(context.forget_positions),
        "tokens_after_owned_span": (
            len(context.original_token_ids) - max(context.forget_positions) - 1
        ),
        "ownership": {
            "owned_character_span": list(context.owned_character_span),
            "answer_character_span": list(context.answer_character_span),
            "forget_positions": list(context.forget_positions),
            "deletion_ranges": [
                {"start": int(start), "end": int(end)}
                for start, end in context.deletion_ranges
            ],
            "mapping": "positive character-overlap from tokenizer offset_mapping",
            "offset_mapping_sha256": _payload_sha256(
                [list(offset) for offset in context.offset_mapping]
            ),
            "owned_token_offsets": owned_offsets,
        },
        "retained": {
            "character_span": list(context.retained_character_span),
            "token_positions": list(context.retained_positions),
            "edited_token_positions": list(context.edited_retained_positions),
            "token_count": len(context.retained_positions),
        },
    }


def _probe_descriptor(probe: LongitudinalProbe) -> dict[str, Any]:
    return {
        "probe_id": probe.probe_id,
        "kind": probe.kind,
        "question_sha256": text_sha256(probe.question),
        "prompt_sha256": text_sha256(probe.prompt),
        "target_name_sha256": text_sha256(probe.target_name),
        "target_sha256": text_sha256(probe.target),
        "target_token_count": len(probe.target_token_ids),
        "target_token_ids_sha256": token_ids_sha256(probe.target_token_ids),
    }


def _public_record(
    record: LongitudinalRecord,
    *,
    evidence_sha256: str,
    longitudinal_sha256: str,
) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "source_file": record.source_file,
        "source": {
            "evidence_sha256": evidence_sha256,
            "longitudinal_sha256": longitudinal_sha256,
        },
        "target_operation_id": record.target_operation_id,
        "retained_operation_id": record.retained_operation_id,
        "source_diagnostics": {
            "target_value_literal_in_owned": (
                record.target_value_literal_in_owned
            ),
            "target_value_occurrences_inside_owned": (
                record.target_value_occurrences_inside_owned
            ),
            "target_value_occurrences_outside_owned": (
                record.target_value_occurrences_outside_owned
            ),
            "retained_value_literal_in_retained_exchange": (
                record.retained_value_literal_in_retained_exchange
            ),
            "retained_value_occurrences_inside_owned": (
                record.retained_value_occurrences_inside_owned
            ),
            "retained_value_occurrences_outside_owned": (
                record.retained_value_occurrences_outside_owned
            ),
            "semantic_gold_may_be_a_normalized_paraphrase": True,
            "source_session_start_index": record.source_session_start_index,
            "source_session_end_index": record.source_session_end_index,
        },
        "context": _context_descriptor(
            record.context,
            record.raw_omitted_token_ids,
        ),
        "probes": [_probe_descriptor(probe) for probe in record.probes],
    }


def _reason_code(exc: Exception) -> str:
    message = str(exc).casefold()
    mappings = (
        ("question leaks", "query_leaks_target"),
        ("target token count", "target_token_count"),
        ("retained value occurs inside", "retained_contaminates_owned"),
        ("outside its owned exchange", "target_repeated_outside_owned"),
        ("not literal", "value_not_literal"),
        ("not enough model tokens before", "insufficient_prefix"),
        ("not enough longitudinal tokens", "insufficient_suffix"),
        ("not enough model tokens after", "insufficient_suffix"),
        ("token cap", "context_token_cap"),
        ("fixed-c feasible", "fixed_c_infeasible"),
        ("maps to", "trigger_mapping"),
    )
    for fragment, code in mappings:
        if fragment in message:
            return code
    return "other_source_contract"


def build_manifest(
    source_root: str | Path,
    tokenizer: Any,
    *,
    records: int = DEFAULT_RECORDS,
    seed: int = DEFAULT_SEED,
    minimum_tokens_before_owned: int = DEFAULT_MINIMUM_TOKENS_BEFORE_OWNED,
    minimum_tokens_after_owned: int = DEFAULT_MINIMUM_TOKENS_AFTER_OWNED,
    maximum_context_tokens: int = DEFAULT_MAXIMUM_CONTEXT_TOKENS,
    maximum_target_tokens: int = DEFAULT_MAXIMUM_TARGET_TOKENS,
    nu: float = DEFAULT_NU,
    chunk: int = DEFAULT_CHUNK,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
) -> dict[str, Any]:
    """Build a deterministic source-free manifest without model evaluation."""

    if records < 1:
        raise ValueError("records must be positive")
    if (
        minimum_tokens_before_owned < 0
        or minimum_tokens_after_owned < 1
        or maximum_context_tokens < minimum_tokens_after_owned
        or maximum_target_tokens < 1
        or chunk < 1
        or not 0 < nu <= 1
    ):
        raise ValueError("invalid longitudinal benchmark geometry")
    root = Path(source_root)
    evidence_directory = root / EVIDENCE_DIRECTORY
    longitudinal_directory = root / LONGITUDINAL_DIRECTORY
    if not evidence_directory.is_dir() or not longitudinal_directory.is_dir():
        raise FileNotFoundError("MemOps generated artifact directories are missing")
    filenames = sorted(
        path.name for path in evidence_directory.glob("*_remember.json")
    )
    filenames = sorted(
        filenames,
        key=lambda name: (
            _stable_key("memops-file-v1", seed, name),
            name,
        ),
    )
    selected: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    pair_rejection_counts: dict[str, int] = {}
    eligible_files = 0
    attempted_pairs = 0
    for source_file in filenames:
        evidence_path = evidence_directory / source_file
        longitudinal_path = longitudinal_directory / source_file
        if not longitudinal_path.is_file():
            rejection_counts["missing_longitudinal_pair"] = (
                rejection_counts.get("missing_longitudinal_pair", 0) + 1
            )
            continue
        evidence = _load_json(evidence_path)
        longitudinal = _load_json(longitudinal_path)
        operations = _confirmed_remember_operations(evidence)
        pairs = _candidate_pairs(
            operations,
            seed=seed,
            source_file=source_file,
        )
        choices: list[tuple[tuple[Any, ...], LongitudinalRecord]] = []
        local_reasons: set[str] = set()
        for target_operation, retained_operation in pairs:
            attempted_pairs += 1
            try:
                runtime_record = _prepare_record(
                    tokenizer,
                    evidence,
                    longitudinal,
                    source_file=source_file,
                    target_operation=target_operation,
                    retained_operation=retained_operation,
                    minimum_tokens_before_owned=minimum_tokens_before_owned,
                    minimum_tokens_after_owned=minimum_tokens_after_owned,
                    maximum_context_tokens=maximum_context_tokens,
                    maximum_target_tokens=maximum_target_tokens,
                    nu=nu,
                    chunk=chunk,
                    tokenizer_id=tokenizer_id,
                    tokenizer_revision=tokenizer_revision,
                )
            except (ManifestError, KeyError, TypeError, ValueError) as exc:
                reason = _reason_code(exc)
                local_reasons.add(reason)
                pair_rejection_counts[reason] = (
                    pair_rejection_counts.get(reason, 0) + 1
                )
                continue
            choices.append(
                (
                    (
                        not runtime_record.target_value_literal_in_owned,
                        runtime_record.target_value_occurrences_outside_owned,
                        not runtime_record.retained_value_literal_in_retained_exchange,
                        len(runtime_record.context.original_token_ids),
                        len(runtime_record.probes[0].target),
                        _stable_key(
                            "memops-valid-pair-v1",
                            seed,
                            source_file,
                            runtime_record.target_operation_id,
                            runtime_record.retained_operation_id,
                        ),
                    ),
                    runtime_record,
                )
            )
        chosen = None
        if choices:
            runtime_record = min(choices, key=lambda item: item[0])[1]
            chosen = _public_record(
                runtime_record,
                evidence_sha256=_sha256_file(evidence_path),
                longitudinal_sha256=_sha256_file(longitudinal_path),
            )
        if chosen is None:
            if not pairs:
                local_reasons.add("no_same_segment_pair")
            for reason in sorted(local_reasons):
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        eligible_files += 1
        selected.append(chosen)
    if len(selected) < records:
        raise ManifestError(
            f"requested {records} records but only {len(selected)} were eligible"
        )
    eligible_records = len(selected)
    selected = selected[:records]

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contains_source_text": False,
        "provenance": {
            "repository": MEMOPS_REPOSITORY,
            "revision": MEMOPS_REVISION,
            "license": MEMOPS_LICENSE,
            "evidence_directory": EVIDENCE_DIRECTORY.as_posix(),
            "longitudinal_directory": LONGITUDINAL_DIRECTORY.as_posix(),
            "adaptation": (
                "external deletion of one complete fact-introducing "
                "user/assistant exchange from MemOps Remember trajectories"
            ),
        },
        "tokenizer": {
            "model_id": tokenizer_id,
            "revision": tokenizer_revision,
            "offset_mapping_required": True,
        },
        "selection": {
            "seed": int(seed),
            "requested_records": int(records),
            "discovered_remember_files": len(filenames),
            "eligible_files_seen": eligible_files,
            "eligible_records_total": eligible_records,
            "attempted_source_pairs": attempted_pairs,
            "rejection_reason_counts": rejection_counts,
            "pair_rejection_reason_counts": pair_rejection_counts,
            "fixed_before_model_evaluation": True,
            "model_outputs_used": False,
            "one_record_per_source_conversation": True,
            "target_and_retained_operations_share_evidence_segment": True,
            "source_ranking": (
                "prefer literal semantic gold, then fewer exact target mentions "
                "outside the owned exchange, a literal retained semantic value, "
                "shorter context, shorter target, and a seeded stable tie-break"
            ),
            "target_value_occurrences_are_reported_not_filtered": True,
            "retained_value_must_not_occur_inside_owned_exchange": True,
        },
        "geometry": {
            "minimum_tokens_before_owned": int(minimum_tokens_before_owned),
            "minimum_tokens_after_owned": int(minimum_tokens_after_owned),
            "maximum_context_tokens": int(maximum_context_tokens),
            "maximum_target_tokens": int(maximum_target_tokens),
            "contiguous_whole_session_window": True,
            "nu": float(nu),
            "chunk": int(chunk),
            "per_boundary_box": True,
            "fixed_c_precheck": True,
        },
        "evaluation_contract": {
            "present_control": "unaltered longitudinal token stream",
            "behavioral_reference": (
                "fresh prefill after removing the owned raw-text exchange and "
                "retokenizing the joined transcript"
            ),
            "solver_reference": (
                "fixed-C retained-key refit on the same contextualized keys"
            ),
            "row_edit_control": (
                "original token rows at the owned character span are removed "
                "without retokenizing; this is distinct from raw omitted repack"
            ),
            "deleted_probes": 2,
            "retained_probes": 1,
            "semantic_gold": (
                "MemOps operation new_value; it can normalize or paraphrase "
                "the literal trigger exchange"
            ),
            "admission_filtering": (
                "report all attempted records; efficacy summaries may additionally "
                "report the predeclared recall-admitted subset without replacement"
            ),
        },
        "records": selected,
    }
    return freeze_manifest(manifest)


def _validate_without_integrity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA or int(
        manifest.get("schema_version", -1)
    ) != SCHEMA_VERSION:
        raise ManifestError("unsupported MemOps longitudinal manifest")
    if manifest.get("contains_source_text") is not False:
        raise ManifestError("MemOps manifest must be source-free")
    provenance = manifest.get("provenance") or {}
    if (
        provenance.get("repository") != MEMOPS_REPOSITORY
        or provenance.get("revision") != MEMOPS_REVISION
        or provenance.get("license") != MEMOPS_LICENSE
    ):
        raise ManifestError("MemOps provenance is not pinned")
    selection = manifest.get("selection") or {}
    if (
        selection.get("fixed_before_model_evaluation") is not True
        or selection.get("model_outputs_used") is not False
    ):
        raise ManifestError("MemOps cohort was not predeclared")
    geometry = manifest.get("geometry") or {}
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != int(
        selection.get("requested_records", -1)
    ):
        raise ManifestError("MemOps manifest record count is inconsistent")
    record_ids = [str(record.get("record_id") or "") for record in records]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ManifestError("MemOps manifest record IDs are empty or duplicated")
    source_files = [str(record.get("source_file") or "") for record in records]
    if not all(source_files) or len(source_files) != len(set(source_files)):
        raise ManifestError("MemOps source conversations are empty or duplicated")
    for record in records:
        context = record.get("context") or {}
        source_diagnostics = record.get("source_diagnostics") or {}
        if (
            int(
                source_diagnostics.get(
                    "retained_value_occurrences_inside_owned",
                    -1,
                )
            )
            != 0
        ):
            raise ManifestError("MemOps retained probe contaminates owned exchange")
        ownership = context.get("ownership") or {}
        positions = ownership.get("forget_positions")
        if not isinstance(positions, list) or not positions:
            raise ManifestError("MemOps token ownership is missing")
        if int(context["original_token_count"]) - len(positions) != int(
            context["edited_token_count"]
        ):
            raise ManifestError("MemOps literal edit count is inconsistent")
        row_edit = context.get("retained_key_row_edit") or {}
        raw_repack = context.get("raw_omitted_repack") or {}
        if (
            int(row_edit.get("token_count", -1))
            != int(context["edited_token_count"])
            or row_edit.get("constructed_by_removing_original_token_rows")
            is not True
        ):
            raise ManifestError("MemOps retained-key row edit is incomplete")
        if (
            int(raw_repack.get("token_count", 0)) < 1
            or raw_repack.get("freshly_retokenized") is not True
            or not str(raw_repack.get("token_ids_sha256") or "")
        ):
            raise ManifestError("MemOps raw omitted repack is incomplete")
        if int(context["tokens_before_owned_span"]) < int(
            geometry["minimum_tokens_before_owned"]
        ):
            raise ManifestError("MemOps target is too close to context start")
        if int(context["tokens_after_owned_span"]) < int(
            geometry["minimum_tokens_after_owned"]
        ):
            raise ManifestError("MemOps target is too close to query")
        if int(context["original_token_count"]) > int(
            geometry["maximum_context_tokens"]
        ):
            raise ManifestError("MemOps context exceeds token cap")
        probes = record.get("probes")
        if not isinstance(probes, list) or [
            probe.get("probe_id") for probe in probes
        ] != ["deleted_direct", "deleted_paraphrase", "retained_direct"]:
            raise ManifestError("MemOps probe contract differs")
        if any(
            int(probe.get("target_token_count", 0)) < 1
            or not str(probe.get("target_token_ids_sha256") or "")
            for probe in probes
        ):
            raise ManifestError("MemOps prompt-conditioned target tokens are missing")


def freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(manifest))
    frozen.pop("integrity", None)
    _validate_without_integrity(frozen)
    frozen["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(frozen),
    }
    validate_manifest(frozen)
    return frozen


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    _validate_without_integrity(manifest)
    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ManifestError("MemOps manifest integrity is missing")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
    ):
        raise ManifestError("MemOps manifest integrity mismatch")


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(path))
    validate_manifest(payload)
    return payload


def _operation_by_id(
    evidence: Mapping[str, Any],
    operation_id: str,
) -> Mapping[str, Any]:
    matches = [
        operation
        for operation in _confirmed_remember_operations(evidence)
        if _operation_id(operation) == str(operation_id)
    ]
    if len(matches) != 1:
        raise ManifestError(f"operation {operation_id!r} is missing or duplicated")
    return matches[0]


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    source_root: str | Path,
    tokenizer: Any,
) -> tuple[LongitudinalRecord, ...]:
    """Rebuild source text and reject any drift from the frozen descriptors."""

    validate_manifest(manifest)
    root = Path(source_root)
    geometry = manifest["geometry"]
    tokenizer_spec = manifest["tokenizer"]
    records = []
    for stored in manifest["records"]:
        source_file = str(stored["source_file"])
        evidence_path = root / EVIDENCE_DIRECTORY / source_file
        longitudinal_path = root / LONGITUDINAL_DIRECTORY / source_file
        if _sha256_file(evidence_path) != stored["source"]["evidence_sha256"]:
            raise ManifestError(f"{source_file} evidence artifact hash differs")
        if (
            _sha256_file(longitudinal_path)
            != stored["source"]["longitudinal_sha256"]
        ):
            raise ManifestError(f"{source_file} longitudinal artifact hash differs")
        evidence = _load_json(evidence_path)
        longitudinal = _load_json(longitudinal_path)
        runtime_record = _prepare_record(
            tokenizer,
            evidence,
            longitudinal,
            source_file=source_file,
            target_operation=_operation_by_id(
                evidence,
                stored["target_operation_id"],
            ),
            retained_operation=_operation_by_id(
                evidence,
                stored["retained_operation_id"],
            ),
            minimum_tokens_before_owned=int(
                geometry["minimum_tokens_before_owned"]
            ),
            minimum_tokens_after_owned=int(
                geometry["minimum_tokens_after_owned"]
            ),
            maximum_context_tokens=int(geometry["maximum_context_tokens"]),
            maximum_target_tokens=int(geometry["maximum_target_tokens"]),
            nu=float(geometry["nu"]),
            chunk=int(geometry["chunk"]),
            tokenizer_id=str(tokenizer_spec["model_id"]),
            tokenizer_revision=str(tokenizer_spec["revision"]),
        )
        public = _public_record(
            runtime_record,
            evidence_sha256=_sha256_file(evidence_path),
            longitudinal_sha256=_sha256_file(longitudinal_path),
        )
        if public != stored:
            raise ManifestError(f"{source_file} rehydrated descriptor differs")
        records.append(runtime_record)
    return tuple(records)


def _repository_head(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_offset_tokenizer(
    *,
    model_id: str = DEFAULT_TOKENIZER_ID,
    revision: str = DEFAULT_TOKENIZER_REVISION,
):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to build the manifest") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("resolved Gemma tokenizer is not fast")
    return tokenizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--minimum-tokens-before-owned",
        type=int,
        default=DEFAULT_MINIMUM_TOKENS_BEFORE_OWNED,
    )
    parser.add_argument(
        "--minimum-tokens-after-owned",
        type=int,
        default=DEFAULT_MINIMUM_TOKENS_AFTER_OWNED,
    )
    parser.add_argument(
        "--maximum-context-tokens",
        type=int,
        default=DEFAULT_MAXIMUM_CONTEXT_TOKENS,
    )
    parser.add_argument(
        "--maximum-target-tokens",
        type=int,
        default=DEFAULT_MAXIMUM_TARGET_TOKENS,
    )
    parser.add_argument("--tokenizer-id", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_memops/memops_longitudinal_v1.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    source_root = Path(args.source_root)
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")
    try:
        observed_revision = _repository_head(source_root)
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"could not verify MemOps source revision: {exc}")
    if observed_revision != MEMOPS_REVISION:
        parser.error(
            f"MemOps source revision {observed_revision} differs from "
            f"pinned {MEMOPS_REVISION}"
        )
    tokenizer = load_offset_tokenizer(
        model_id=args.tokenizer_id,
        revision=args.tokenizer_revision,
    )
    manifest = build_manifest(
        source_root,
        tokenizer,
        records=args.records,
        seed=args.seed,
        minimum_tokens_before_owned=args.minimum_tokens_before_owned,
        minimum_tokens_after_owned=args.minimum_tokens_after_owned,
        maximum_context_tokens=args.maximum_context_tokens,
        maximum_target_tokens=args.maximum_target_tokens,
        tokenizer_id=args.tokenizer_id,
        tokenizer_revision=args.tokenizer_revision,
    )
    write_manifest(output, manifest)
    print(
        f"wrote {len(manifest['records'])} frozen MemOps records to {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

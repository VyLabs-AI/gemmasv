"""Freeze an official LongMemEval knowledge-update deletion adaptation.

This is not a LongMemEval leaderboard implementation.  It uses the official
cleaned oracle artifact to isolate context deletion: the chronologically latest
official evidence session for a non-abstention ``knowledge-update`` question is
owned in full, while every earlier official evidence session is retained.

Importing this module is download-free.  Hub and tokenizer dependencies are
loaded only by explicit CLI helpers.  Pure parsing, selection, packaging,
manifest, and hydration APIs accept ordinary mappings and an offset-aware
tokenizer so tests can run without network or model access.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv.rag_benchmark import (
    CorpusNode,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TOKENIZER_REVISION,
    TokenizedContext,
    WikiExample,
    _contiguous_ranges,
    _overlaps,
    _remove_positions,
    _shift_positions,
    _tokenizer_payload,
    load_offset_tokenizer,
    stable_identifier,
    text_sha256,
    token_ids_sha256,
    write_environment_lock,
)


BENCHMARK_LABEL = "LongMemEval knowledge-update context-deletion adaptation"
MANIFEST_SCHEMA = "gemma-sv-longmemeval-deletion-manifest-v1"
MANIFEST_SCHEMA_VERSION = 1
PACKAGE_SCHEMA = "gemma-sv-longmemeval-deletion-context-v1"

LONGMEMEVAL_REPOSITORY = "xiaowu0162/LongMemEval"
LONGMEMEVAL_REPOSITORY_REVISION = (
    "9e0b455f4ef0e2ab8f2e582289761153549043fc"
)
LONGMEMEVAL_REPOSITORY_LICENSE = "MIT"
LONGMEMEVAL_ICLR_YEAR = 2025
LONGMEMEVAL_ICLR_PAPER_ID = "d813d324dbf0598bbdc9c8e79740ed01"
LONGMEMEVAL_ARXIV_ID = "2410.10813"

DATASET_ID = "xiaowu0162/longmemeval-cleaned"
DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
DATASET_REVISION_TIMESTAMP = "2025-09-19T23:48:16Z"
DATASET_LICENSE = "MIT"
DATASET_LICENSE_SOURCE = "Hugging Face dataset-card metadata"
DATASET_HAS_STANDALONE_LICENSE_FILE = False
DATASET_ARTIFACT_PATH = "longmemeval_oracle.json"
DATASET_ARTIFACT_SIZE = 15_388_478
DATASET_ARTIFACT_SHA256 = (
    "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"
)
DATASET_ARTIFACT_GIT_OID = "8e6e661d71203470ff689c4de74c678b5889f227"
DATASET_ARTIFACT_XET_HASH = (
    "7d958f239f452f33a5368e31d0a3dc6ffb4f225036ddd46fc71370a6dc1e2db6"
)
DATASET_NUM_ROWS = 500

CLEANED_S_ARTIFACT_PATH = "longmemeval_s_cleaned.json"
CLEANED_S_ARTIFACT_SIZE = 277_383_467
CLEANED_S_ARTIFACT_SHA256 = (
    "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
)

DEFAULT_RECORDS = 8
DEFAULT_SEED = 20_250_519
DEFAULT_MINIMUM_TOKENS_AFTER_OWNED = 512
DEFAULT_GREEDY_TOKEN_RESERVE = 64
RUNTIME_CONTEXT_CEILING = 2_048
KNOWLEDGE_UPDATE_TYPE = "knowledge-update"

_FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "question",
        "answer",
        "content",
        "turns",
        "sessions",
        "source_text",
        "generated_text",
    }
)
_OFFICIAL_DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
)


class ManifestError(ValueError):
    """A frozen LongMemEval manifest is invalid or has drifted."""


@dataclass(frozen=True)
class LongMemEvalTurn:
    """One preserved official role/content turn."""

    role: str
    content: str
    has_answer: bool | None
    source_turn_sha256: str


@dataclass(frozen=True)
class LongMemEvalSession:
    """One official timestamped dialogue session."""

    session_id: str
    date_text: str
    timestamp: datetime | None
    turns: tuple[LongMemEvalTurn, ...]
    source_session_sha256: str
    parse_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class LongMemEvalExample:
    """One normalized official LongMemEval row."""

    source_id: str
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    question_timestamp: datetime | None
    haystack_sessions: tuple[LongMemEvalSession, ...]
    answer_session_ids: tuple[str, ...]
    source_row_sha256: str
    parse_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class SessionReference:
    """A stable cross-row reference to one official session."""

    source_id: str
    question_id: str
    session_id: str


@dataclass(frozen=True)
class ContextAssembly:
    """Packaged context plus the frozen suffix-session choices."""

    context: TokenizedContext
    earlier_evidence: tuple[LongMemEvalSession, ...]
    owned_latest_evidence: LongMemEvalSession
    retained_evidence: tuple[LongMemEvalSession, ...]
    tail_references: tuple[SessionReference, ...]
    owned_block_sha256: str
    query_token_reserve: int


@dataclass(frozen=True)
class RehydratedLongMemEvalRecord:
    """Runtime-only source text reconstructed from a source-free manifest."""

    record_id: str
    target: LongMemEvalExample
    retained: LongMemEvalExample
    earlier_evidence: tuple[LongMemEvalSession, ...]
    owned_latest_evidence: LongMemEvalSession
    retained_evidence: tuple[LongMemEvalSession, ...]
    tail_references: tuple[SessionReference, ...]
    context: TokenizedContext

    @property
    def example(self) -> WikiExample:
        """Compatibility view for shared evaluator/report helpers."""

        return WikiExample(
            example_id=self.target.question_id,
            question=self.target.question,
            answer=self.target.answer,
            question_type=self.target.question_type,
            passages=(),
        )

    @property
    def retained_question(self) -> str:
        return self.retained.question

    @property
    def retained_answer(self) -> str:
        return self.retained.answer

    @property
    def retained_node(self) -> CorpusNode:
        return CorpusNode(
            node_id=stable_identifier(
                "longmemeval-retained-node",
                DATASET_REVISION,
                self.retained.source_id,
            ),
            document_id=stable_identifier(
                "longmemeval-retained-document",
                DATASET_REVISION,
                self.retained.source_id,
            ),
            source_example_id=self.retained.question_id,
            paragraph_index=0,
            chunk_index=0,
            title="",
            text="",
        )


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _plain_sequence(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return value
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _as_sequence(value: Any) -> tuple[Any, ...]:
    value = _plain_sequence(value)
    if value is None:
        return ()
    if _is_sequence(value):
        return tuple(value)
    return (value,)


def _answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_datetime(value: Any) -> tuple[str, datetime | None]:
    if isinstance(value, datetime):
        return value.isoformat(timespec="minutes"), value.replace(tzinfo=None)
    rendered = "" if value is None else str(value).strip()
    if not rendered:
        return "", None
    for format_string in _OFFICIAL_DATE_FORMATS:
        try:
            return rendered, datetime.strptime(rendered, format_string)
        except ValueError:
            continue
    normalized = rendered[:-1] + "+00:00" if rendered.endswith("Z") else rendered
    try:
        observed = datetime.fromisoformat(normalized)
    except ValueError:
        return rendered, None
    if observed.tzinfo is not None:
        observed = observed.replace(tzinfo=None)
    return rendered, observed


def _mapping_rows(value: Any, *, field: str) -> tuple[Mapping[str, Any], ...]:
    """Accept raw list-of-struct and Arrow/pandas struct-of-lists variants."""

    value = _plain_sequence(value)
    if value is None:
        return ()
    if isinstance(value, Mapping):
        normalized = {
            str(key): _plain_sequence(item) for key, item in value.items()
        }
        sequence_lengths = {
            len(item) for item in normalized.values() if _is_sequence(item)
        }
        if not sequence_lengths:
            return (value,)
        if len(sequence_lengths) != 1:
            raise ValueError(f"{field} columns have different lengths")
        length = next(iter(sequence_lengths))
        return tuple(
            {
                key: item[index] if _is_sequence(item) else item
                for key, item in normalized.items()
            }
            for index in range(length)
        )
    if not _is_sequence(value):
        raise ValueError(f"{field} must be a sequence of mappings")
    result = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} contains a non-mapping entry")
        result.append(item)
    return tuple(result)


def _normalize_turn(raw: Mapping[str, Any]) -> LongMemEvalTurn:
    role = str(raw.get("role") or "").strip()
    content = str(raw.get("content") or "")
    has_answer_raw = raw.get("has_answer")
    has_answer = (
        has_answer_raw
        if isinstance(has_answer_raw, bool)
        else None
        if has_answer_raw is None
        else str(has_answer_raw).strip().casefold() in {"true", "1", "yes"}
    )
    normalized_payload = {
        str(key): raw[key] for key in sorted(raw, key=lambda item: str(item))
    }
    return LongMemEvalTurn(
        role=role,
        content=content,
        has_answer=has_answer,
        source_turn_sha256=_payload_sha256(normalized_payload),
    )


def _session_payload(session: LongMemEvalSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "date": session.date_text,
        "turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "has_answer": turn.has_answer,
                "source_turn_sha256": turn.source_turn_sha256,
            }
            for turn in session.turns
        ],
    }


def parse_longmemeval_row(row: Mapping[str, Any]) -> LongMemEvalExample:
    """Normalize official JSON and common Arrow-shaped schema variants."""

    if not isinstance(row, Mapping):
        raise TypeError("LongMemEval source row must be a mapping")
    question_id = str(row.get("question_id") or row.get("id") or "").strip()
    question_type = str(
        row.get("question_type") or row.get("type") or ""
    ).strip()
    question = str(row.get("question") or "")
    answer = _answer_text(
        row["answer"]
        if "answer" in row
        else row.get("gold_answer")
    )
    question_date, question_timestamp = _parse_datetime(
        row.get("question_date") or row.get("date")
    )

    session_ids = _as_sequence(
        row.get("haystack_session_ids")
        if "haystack_session_ids" in row
        else row.get("session_ids")
    )
    dates = _as_sequence(
        row.get("haystack_dates")
        if "haystack_dates" in row
        else row.get("session_dates")
    )
    raw_sessions = _as_sequence(
        row.get("haystack_sessions")
        if "haystack_sessions" in row
        else row.get("history_sessions")
    )
    answer_ids = tuple(
        str(value).strip()
        for value in _as_sequence(
            row.get("answer_session_ids")
            if "answer_session_ids" in row
            else row.get("evidence_session_ids")
        )
        if str(value).strip()
    )

    issues: list[str] = []
    lengths = (len(session_ids), len(dates), len(raw_sessions))
    if len(set(lengths)) != 1:
        issues.append("session_arrays_length_mismatch")
    session_count = max(lengths, default=0)
    sessions = []
    for index in range(session_count):
        session_id = (
            str(session_ids[index]).strip() if index < len(session_ids) else ""
        )
        date_text, timestamp = _parse_datetime(
            dates[index] if index < len(dates) else None
        )
        session_issues = []
        if not session_id:
            session_issues.append("missing_session_id")
        if not date_text:
            session_issues.append("missing_session_date")
        elif timestamp is None:
            session_issues.append("invalid_session_date")
        try:
            turn_rows = (
                _mapping_rows(raw_sessions[index], field="session turns")
                if index < len(raw_sessions)
                else ()
            )
        except ValueError:
            turn_rows = ()
            session_issues.append("invalid_session_shape")
        turns = tuple(_normalize_turn(turn) for turn in turn_rows)
        if not turns:
            session_issues.append("missing_session_turns")
        if any(not turn.role for turn in turns):
            session_issues.append("missing_turn_role")
        if any(not turn.content.strip() for turn in turns):
            session_issues.append("missing_turn_content")
        provisional = LongMemEvalSession(
            session_id=session_id,
            date_text=date_text,
            timestamp=timestamp,
            turns=turns,
            source_session_sha256="",
            parse_issues=tuple(dict.fromkeys(session_issues)),
        )
        sessions.append(
            replace(
                provisional,
                source_session_sha256=_payload_sha256(
                    _session_payload(provisional)
                ),
            )
        )

    normalized_payload = {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "question_date": question_date,
        "haystack_sessions": [_session_payload(item) for item in sessions],
        "answer_session_ids": list(answer_ids),
    }
    row_hash = _payload_sha256(normalized_payload)
    return LongMemEvalExample(
        source_id=stable_identifier(
            "longmemeval-source",
            DATASET_ID,
            DATASET_REVISION,
            question_id,
            row_hash,
        ),
        question_id=question_id,
        question_type=question_type,
        question=question,
        answer=answer,
        question_date=question_date,
        question_timestamp=question_timestamp,
        haystack_sessions=tuple(sessions),
        answer_session_ids=answer_ids,
        source_row_sha256=row_hash,
        parse_issues=tuple(dict.fromkeys(issues)),
    )


def extract_longmemeval_examples(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[LongMemEvalExample, ...]:
    """Parse rows and annotate duplicate official question identifiers."""

    parsed = [parse_longmemeval_row(row) for row in rows]
    question_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for example in parsed:
        question_counts[example.question_id] = (
            question_counts.get(example.question_id, 0) + 1
        )
        source_counts[example.source_id] = (
            source_counts.get(example.source_id, 0) + 1
        )
    source_occurrences: dict[str, int] = {}
    annotated = []
    for example in parsed:
        source_id = example.source_id
        if source_counts[source_id] > 1:
            occurrence = source_occurrences.get(source_id, 0)
            source_occurrences[source_id] = occurrence + 1
            source_id = stable_identifier(
                "longmemeval-duplicate-source",
                source_id,
                occurrence,
            )
        annotated.append(
            replace(
                example,
                source_id=source_id,
                parse_issues=tuple(
                    dict.fromkeys(
                        (
                            *example.parse_issues,
                            *(
                                ("duplicate_question_id",)
                                if question_counts.get(example.question_id, 0)
                                > 1
                                else ()
                            ),
                        )
                    )
                ),
            )
        )
    return tuple(annotated)


def _selection_key(namespace: str, seed: int, *parts: object) -> bytes:
    framed = "\0".join(str(part) for part in parts)
    return hashlib.sha256(
        f"{namespace}\0{int(seed)}\0{framed}".encode("utf-8")
    ).digest()


def deterministic_source_order(
    examples: Sequence[LongMemEvalExample],
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[LongMemEvalExample, ...]:
    """Order rows using only pinned source identifiers and a declared seed."""

    return tuple(
        sorted(
            examples,
            key=lambda example: (
                _selection_key(
                    "longmemeval-source-order-v1",
                    seed,
                    example.question_id,
                    example.source_id,
                ),
                example.question_id,
                example.source_id,
            ),
        )
    )


def is_abstention(example: LongMemEvalExample) -> bool:
    return example.question_id.casefold().endswith("_abs")


def _session_map(
    example: LongMemEvalExample,
) -> dict[str, LongMemEvalSession]:
    return {
        session.session_id: session for session in example.haystack_sessions
    }


def evidence_sessions(
    example: LongMemEvalExample,
) -> tuple[LongMemEvalSession, ...]:
    """Return official evidence sessions in chronological order."""

    by_id = _session_map(example)
    selected = [
        by_id[session_id]
        for session_id in example.answer_session_ids
        if session_id in by_id
    ]
    return tuple(
        sorted(
            selected,
            key=lambda session: (
                session.timestamp or datetime.max,
                session.session_id,
            ),
        )
    )


def _base_row_rejection_reasons(
    example: LongMemEvalExample,
) -> tuple[str, ...]:
    reasons = list(example.parse_issues)
    if not example.question_id:
        reasons.append("missing_question_id")
    if not example.question_type:
        reasons.append("missing_question_type")
    if not example.question.strip():
        reasons.append("missing_question")
    if not example.answer.strip():
        reasons.append("missing_current_answer")
    if not example.question_date:
        reasons.append("missing_question_date")
    elif example.question_timestamp is None:
        reasons.append("invalid_question_date")
    session_ids = [
        session.session_id for session in example.haystack_sessions
    ]
    if not example.haystack_sessions:
        reasons.append("missing_sessions")
    if len(session_ids) != len(set(session_ids)):
        reasons.append("duplicate_session_ids")
    if not example.answer_session_ids:
        reasons.append("missing_answer_session_ids")
    if len(example.answer_session_ids) != len(set(example.answer_session_ids)):
        reasons.append("duplicate_answer_session_ids")
    missing = set(example.answer_session_ids).difference(session_ids)
    if missing:
        reasons.append("missing_answer_evidence_session")
    by_id = _session_map(example)
    for session_id in example.answer_session_ids:
        session = by_id.get(session_id)
        if session is None:
            continue
        reasons.extend(session.parse_issues)
    return tuple(dict.fromkeys(reasons))


def target_rejection_reasons(
    example: LongMemEvalExample,
) -> tuple[str, ...]:
    """Return the complete source-only target eligibility result."""

    reasons = list(_base_row_rejection_reasons(example))
    if example.question_type != KNOWLEDGE_UPDATE_TYPE:
        reasons.append("not_knowledge_update")
    if is_abstention(example):
        reasons.append("abstention_question")
    sessions = evidence_sessions(example)
    if not sessions:
        reasons.append("missing_latest_evidence")
    if len(sessions) < 2:
        reasons.append("no_distinct_earlier_evidence")
    if any(not any(turn.has_answer is True for turn in session.turns) for session in sessions):
        reasons.append("answer_bearing_turn_unavailable")
    valid_times = [
        session.timestamp for session in sessions if session.timestamp is not None
    ]
    if len(valid_times) != len(sessions):
        reasons.append("missing_or_invalid_evidence_date")
        reasons.append("missing_or_invalid_latest_evidence")
    elif valid_times:
        latest = max(valid_times)
        if sum(timestamp == latest for timestamp in valid_times) != 1:
            reasons.append("ambiguous_latest_evidence")
    return tuple(dict.fromkeys(reasons))


def retained_rejection_reasons(
    example: LongMemEvalExample,
) -> tuple[str, ...]:
    """Return source-only eligibility for an unrelated retained QA."""

    reasons = list(_base_row_rejection_reasons(example))
    if is_abstention(example):
        reasons.append("abstention_question")
    sessions = evidence_sessions(example)
    if not sessions:
        reasons.append("missing_retained_evidence")
    if not any(
        any(turn.has_answer is True for turn in session.turns)
        for session in sessions
    ):
        reasons.append("answer_bearing_turn_unavailable")
    if any(session.timestamp is None for session in sessions):
        reasons.append("missing_or_invalid_evidence_date")
    return tuple(dict.fromkeys(reasons))


def latest_update_partition(
    example: LongMemEvalExample,
) -> tuple[tuple[LongMemEvalSession, ...], LongMemEvalSession]:
    """Partition official evidence into chronological earlier/latest sessions."""

    reasons = target_rejection_reasons(example)
    if reasons:
        raise ValueError(
            "LongMemEval target is ineligible: " + ", ".join(reasons)
        )
    sessions = evidence_sessions(example)
    return sessions[:-1], sessions[-1]


def evidence_turn_excerpt(
    session: LongMemEvalSession,
) -> LongMemEvalSession:
    """Keep official answer-bearing turns and their immediate responses."""

    evidence_indices = {
        index
        for index, turn in enumerate(session.turns)
        if turn.has_answer is True
    }
    if not evidence_indices:
        raise LookupError("answer_bearing_turn_unavailable")
    selected = set(evidence_indices)
    for index in evidence_indices:
        if index + 1 < len(session.turns):
            selected.add(index + 1)
    return replace(
        session,
        turns=tuple(session.turns[index] for index in sorted(selected)),
        parse_issues=tuple(
            dict.fromkeys((*session.parse_issues, "answer_turn_excerpt"))
        ),
    )


def deterministic_retained_pair(
    target: LongMemEvalExample,
    examples: Sequence[LongMemEvalExample],
    *,
    seed: int = DEFAULT_SEED,
) -> LongMemEvalExample:
    """Choose one distinct official non-abstention retained QA source-only."""

    eligible = [
        example
        for example in examples
        if example.source_id != target.source_id
        and example.question_id != target.question_id
        and not retained_rejection_reasons(example)
        and len(
            [
                session
                for session in evidence_sessions(example)
                if any(turn.has_answer is True for turn in session.turns)
            ]
        )
        == 1
    ]
    if not eligible:
        raise LookupError("single_session_retained_qa_unavailable")
    return min(
        eligible,
        key=lambda example: (
            sum(
                len(turn.content)
                for session in evidence_sessions(example)
                if any(turn.has_answer is True for turn in session.turns)
                for turn in evidence_turn_excerpt(session).turns
            ),
            _selection_key(
                "longmemeval-retained-pair-v1",
                seed,
                target.source_id,
                example.source_id,
            ),
            example.question_id,
            example.source_id,
        ),
    )


def _session_reference(
    example: LongMemEvalExample,
    session: LongMemEvalSession,
) -> SessionReference:
    return SessionReference(
        source_id=example.source_id,
        question_id=example.question_id,
        session_id=session.session_id,
    )


def _session_block(
    session: LongMemEvalSession,
    *,
    ordinal: int,
    label: str,
) -> str:
    pieces = [
        f"### Session {ordinal} ({label})\n",
        f"Session ID: {session.session_id}\n",
        f"Session Date: {session.date_text}\n",
        "Session Content:\n",
    ]
    for turn in session.turns:
        pieces.append(f"{turn.role}: {turn.content}\n")
    pieces.append(f"### End Session {ordinal}\n\n")
    return "".join(pieces)


def longmemeval_query_prompt(example: LongMemEvalExample) -> str:
    """Use the official direct-answer framing with current date and question."""

    return (
        "\n\nPlease answer the question based on the relevant chat history.\n"
        f"Current Date: {example.question_date}\n"
        f"Question: {example.question}\n"
        "Answer:"
    )


def _token_ids_no_special(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    raw_ids = (
        encoded.get("input_ids")
        if isinstance(encoded, Mapping)
        else getattr(encoded, "input_ids", None)
    )
    if raw_ids is None:
        raise ValueError("tokenizer omitted input_ids")
    if raw_ids and _is_sequence(raw_ids[0]):
        if len(raw_ids) != 1:
            raise ValueError("batched tokenization is unsupported")
        raw_ids = raw_ids[0]
    return tuple(int(value) for value in raw_ids)


def _query_reserve(
    tokenizer: Any,
    target: LongMemEvalExample,
    retained: LongMemEvalExample,
) -> int:
    totals = []
    for example in (target, retained):
        prompt_ids = _token_ids_no_special(
            tokenizer,
            longmemeval_query_prompt(example),
        )
        answer_ids = _token_ids_no_special(tokenizer, " " + example.answer)
        totals.append(
            len(prompt_ids)
            + max(len(answer_ids), DEFAULT_GREEDY_TOKEN_RESERVE)
        )
    return max(totals)


def _tail_candidates(
    target: LongMemEvalExample,
    retained: LongMemEvalExample,
    examples: Sequence[LongMemEvalExample],
    *,
    seed: int,
) -> tuple[tuple[LongMemEvalExample, LongMemEvalSession], ...]:
    candidates = []
    forbidden_answers = (
        target.answer.strip().casefold(),
        retained.answer.strip().casefold(),
    )
    for example in examples:
        if example.source_id in {target.source_id, retained.source_id}:
            continue
        if retained_rejection_reasons(example):
            continue
        for session in evidence_sessions(example):
            try:
                session = evidence_turn_excerpt(session)
            except LookupError:
                continue
            rendered = "\n".join(turn.content for turn in session.turns).casefold()
            if any(answer and answer in rendered for answer in forbidden_answers):
                continue
            candidates.append((example, session))
    return tuple(
        sorted(
            candidates,
            key=lambda pair: (
                _selection_key(
                    "longmemeval-context-tail-v1",
                    seed,
                    target.source_id,
                    pair[0].source_id,
                    pair[1].session_id,
                ),
                pair[0].question_id,
                pair[1].session_id,
            ),
        )
    )


def _covered_owned_text(
    text: str,
    span: tuple[int, int],
    offsets: Sequence[tuple[int, int]],
    positions: Sequence[int],
) -> bool:
    coverage = [False] * (span[1] - span[0])
    for position in positions:
        start, end = offsets[position]
        for character in range(max(start, span[0]), min(end, span[1])):
            coverage[character - span[0]] = True
    source = text[span[0] : span[1]]
    return all(
        covered or character.isspace()
        for character, covered in zip(source, coverage)
    )


def package_context(
    tokenizer: Any,
    target: LongMemEvalExample,
    retained: LongMemEvalExample,
    examples: Sequence[LongMemEvalExample],
    *,
    seed: int = DEFAULT_SEED,
    minimum_tokens_after_owned: int = (
        DEFAULT_MINIMUM_TOKENS_AFTER_OWNED
    ),
    context_ceiling: int = RUNTIME_CONTEXT_CEILING,
    frozen_tail_references: Sequence[SessionReference] | None = None,
) -> ContextAssembly:
    """Package exact session ownership and reject any literal-edit drift."""

    if getattr(tokenizer, "is_fast", True) is not True:
        raise ValueError("fast tokenizer offset mappings are required")
    if minimum_tokens_after_owned < 0 or context_ceiling < 1:
        raise ValueError("token geometry bounds must be positive")
    earlier_full, owned_full = latest_update_partition(target)
    earlier = tuple(evidence_turn_excerpt(session) for session in earlier_full)
    owned = evidence_turn_excerpt(owned_full)
    retained_sessions = tuple(
        evidence_turn_excerpt(session)
        for session in evidence_sessions(retained)
        if any(turn.has_answer is True for turn in session.turns)
    )
    if not retained_sessions:
        raise LookupError("retained_evidence_unavailable")

    pieces = [
        f"{BENCHMARK_LABEL}\n",
        "The sessions below are timestamped. Later information supersedes "
        "earlier information.\n\n",
    ]
    ordinal = 1
    for session in earlier:
        pieces.append(
            _session_block(
                session,
                ordinal=ordinal,
                label="target earlier evidence",
            )
        )
        ordinal += 1
    owned_start = sum(len(piece) for piece in pieces)
    pieces.append(
        _session_block(
            owned,
            ordinal=ordinal,
            label="target latest update",
        )
    )
    owned_end = sum(len(piece) for piece in pieces)
    owned_span = (owned_start, owned_end)
    ordinal += 1

    retained_start = sum(len(piece) for piece in pieces)
    for session in retained_sessions:
        pieces.append(
            _session_block(
                session,
                ordinal=ordinal,
                label="distinct retained QA evidence",
            )
        )
        ordinal += 1
    retained_end = sum(len(piece) for piece in pieces)
    retained_span = (retained_start, retained_end)

    examples_by_source = {example.source_id: example for example in examples}
    if len(examples_by_source) != len(examples):
        raise ValueError("LongMemEval source IDs are not unique")
    tail_references: list[SessionReference] = []
    candidates = _tail_candidates(
        target,
        retained,
        examples,
        seed=seed,
    )
    candidate_by_ref = {
        (
            example.source_id,
            session.session_id,
        ): (example, session)
        for example, session in candidates
    }

    def append_tail(example: LongMemEvalExample, session: LongMemEvalSession) -> None:
        nonlocal ordinal
        pieces.append(
            _session_block(
                session,
                ordinal=ordinal,
                label="deterministic frozen tail",
            )
        )
        ordinal += 1
        tail_references.append(_session_reference(example, session))

    if frozen_tail_references is not None:
        for reference in frozen_tail_references:
            key = (reference.source_id, reference.session_id)
            if key not in candidate_by_ref:
                raise ManifestError(
                    "frozen tail session is unavailable or no longer eligible"
                )
            append_tail(*candidate_by_ref[key])

    query_reserve = _query_reserve(tokenizer, target, retained)
    next_candidate = 0
    if frozen_tail_references is None:
        if not candidates:
            raise LookupError("deterministic_official_tail_unavailable")
        append_tail(*candidates[0])
        next_candidate = 1
    elif not frozen_tail_references:
        raise ManifestError("frozen deterministic tail is empty")
    while True:
        original_text = "".join(pieces)
        token_ids, offsets = _tokenizer_payload(tokenizer, original_text)
        forget = tuple(
            index
            for index, offset in enumerate(offsets)
            if _overlaps(offset, owned_span)
        )
        if not forget:
            raise LookupError("owned_session_has_no_model_tokens")
        tokens_after = len(token_ids) - max(forget) - 1
        if tokens_after >= int(minimum_tokens_after_owned):
            break
        if frozen_tail_references is not None:
            raise ManifestError("frozen tail no longer satisfies owned distance")
        if next_candidate >= len(candidates):
            raise LookupError("insufficient_official_tail_sessions")
        append_tail(*candidates[next_candidate])
        next_candidate += 1

    if len(token_ids) + query_reserve > int(context_ceiling):
        raise LookupError("context_token_budget_exceeded")
    if not _covered_owned_text(
        original_text,
        owned_span,
        offsets,
        forget,
    ):
        raise LookupError("offset_mapping_does_not_cover_owned_session")
    retained_positions = tuple(
        index
        for index, offset in enumerate(offsets)
        if _overlaps(offset, retained_span) and index not in set(forget)
    )
    if not retained_positions:
        raise LookupError("retained_evidence_has_no_model_tokens")

    edited_text = (
        original_text[: owned_span[0]] + original_text[owned_span[1] :]
    )
    edited_by_drop = _remove_positions(token_ids, forget)
    retokenized_edited, _ = _tokenizer_payload(tokenizer, edited_text)
    if tuple(retokenized_edited) != edited_by_drop:
        raise LookupError("literal_retokenization_mismatch")
    edited_retained_positions = _shift_positions(retained_positions, forget)
    retained_before = tuple(
        token_ids[position] for position in retained_positions
    )
    retained_after = tuple(
        edited_by_drop[position] for position in edited_retained_positions
    )
    if retained_before != retained_after:
        raise LookupError("literal_deletion_changes_retained_tokens")

    current_match = original_text.casefold().find(
        target.answer.casefold(),
        owned_span[0],
        owned_span[1],
    )
    answer_span = (
        (current_match, current_match + len(target.answer))
        if current_match >= 0
        else owned_span
    )
    context = TokenizedContext(
        original_text=original_text,
        edited_text=edited_text,
        original_token_ids=tuple(token_ids),
        edited_token_ids=edited_by_drop,
        offset_mapping=tuple(offsets),
        forget_positions=forget,
        deletion_ranges=_contiguous_ranges(forget),
        retained_positions=retained_positions,
        edited_retained_positions=edited_retained_positions,
        owned_character_span=owned_span,
        answer_character_span=answer_span,
        retained_character_span=retained_span,
    )
    return ContextAssembly(
        context=context,
        earlier_evidence=earlier,
        owned_latest_evidence=owned,
        retained_evidence=retained_sessions,
        tail_references=tuple(tail_references),
        owned_block_sha256=text_sha256(
            original_text[owned_span[0] : owned_span[1]]
        ),
        query_token_reserve=query_reserve,
    )


def tokens_strictly_after_owned(context: TokenizedContext) -> int:
    return len(context.original_token_ids) - max(context.forget_positions) - 1


def _turn_descriptor(turn: LongMemEvalTurn) -> dict[str, Any]:
    return {
        "role_sha256": text_sha256(turn.role),
        "content_sha256": text_sha256(turn.content),
        "has_answer": turn.has_answer,
        "source_turn_sha256": turn.source_turn_sha256,
    }


def _session_descriptor(session: LongMemEvalSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "date_text_sha256": text_sha256(session.date_text),
        "timestamp": (
            None
            if session.timestamp is None
            else session.timestamp.isoformat(timespec="minutes")
        ),
        "turn_count": len(session.turns),
        "turns_sha256": _payload_sha256(
            [_turn_descriptor(turn) for turn in session.turns]
        ),
        "source_session_sha256": session.source_session_sha256,
        "parse_issues": list(session.parse_issues),
    }


def _example_descriptor(example: LongMemEvalExample) -> dict[str, Any]:
    return {
        "source_id": example.source_id,
        "question_id": example.question_id,
        "question_type": example.question_type,
        "question_sha256": text_sha256(example.question),
        "current_answer_sha256": text_sha256(example.answer),
        "question_date_sha256": text_sha256(example.question_date),
        "source_row_sha256": example.source_row_sha256,
        "answer_session_ids": list(example.answer_session_ids),
        "parse_issues": list(example.parse_issues),
        "session_inventory": [
            _session_descriptor(session)
            for session in example.haystack_sessions
        ],
    }


def _context_descriptor(
    assembly: ContextAssembly,
    *,
    context_ceiling: int,
) -> dict[str, Any]:
    context = assembly.context
    retained_before = tuple(
        context.original_token_ids[position]
        for position in context.retained_positions
    )
    retained_after = tuple(
        context.edited_token_ids[position]
        for position in context.edited_retained_positions
    )
    owned_ids = tuple(
        context.original_token_ids[position]
        for position in context.forget_positions
    )
    return {
        "package_schema": PACKAGE_SCHEMA,
        "original_text_sha256": text_sha256(context.original_text),
        "edited_text_sha256": text_sha256(context.edited_text),
        "original_token_count": len(context.original_token_ids),
        "edited_token_count": len(context.edited_token_ids),
        "original_token_ids_sha256": token_ids_sha256(
            context.original_token_ids
        ),
        "edited_token_ids_sha256": token_ids_sha256(
            context.edited_token_ids
        ),
        "literal_full_repack": True,
        "literal_retokenization_equals_position_drop": True,
        "tokens_strictly_after_owned": tokens_strictly_after_owned(context),
        "query_token_reserve": assembly.query_token_reserve,
        "runtime_total_token_bound": (
            len(context.original_token_ids) + assembly.query_token_reserve
        ),
        "runtime_context_ceiling": int(context_ceiling),
        "evidence_excerpt_policy": (
            "official has_answer turns plus each immediate following response"
        ),
        "earlier_evidence_excerpts": [
            _session_descriptor(session) for session in assembly.earlier_evidence
        ],
        "owned_latest_evidence_excerpt": _session_descriptor(
            assembly.owned_latest_evidence
        ),
        "retained_evidence_excerpts": [
            _session_descriptor(session) for session in assembly.retained_evidence
        ],
        "ownership": {
            "unit": (
                "official answer-bearing turns plus immediate response from "
                "the chronologically latest evidence session"
            ),
            "all_excerpt_turns_owned": True,
            "full_source_session_owned": False,
            "mapping": "positive character overlap from fast-tokenizer offsets",
            "owned_character_span": list(context.owned_character_span),
            "forget_positions": list(context.forget_positions),
            "deletion_ranges": [
                {"start": start, "end": end}
                for start, end in context.deletion_ranges
            ],
            "owned_token_ids_sha256": token_ids_sha256(owned_ids),
            "owned_block_sha256": assembly.owned_block_sha256,
            "offset_mapping_sha256": _payload_sha256(
                [list(offset) for offset in context.offset_mapping]
            ),
        },
        "retained": {
            "character_span": list(context.retained_character_span),
            "token_positions": list(context.retained_positions),
            "edited_token_positions": list(
                context.edited_retained_positions
            ),
            "original_token_ids_sha256": token_ids_sha256(retained_before),
            "edited_token_ids_sha256": token_ids_sha256(retained_after),
            "tokens_preserved_exactly": retained_before == retained_after,
        },
    }


def _reference_payload(reference: SessionReference) -> dict[str, str]:
    return {
        "source_id": reference.source_id,
        "question_id": reference.question_id,
        "session_id": reference.session_id,
    }


def _record_payload(
    target: LongMemEvalExample,
    retained: LongMemEvalExample,
    assembly: ContextAssembly,
    *,
    context_ceiling: int,
) -> dict[str, Any]:
    record_id = stable_identifier(
        "longmemeval-deletion-record",
        DATASET_REVISION,
        target.source_id,
        retained.source_id,
        assembly.owned_latest_evidence.session_id,
    )
    return {
        "record_id": record_id,
        "target": {
            "source_id": target.source_id,
            "question_id": target.question_id,
            "question_type": target.question_type,
            "source_row_sha256": target.source_row_sha256,
            "question_sha256": text_sha256(target.question),
            "current_answer_sha256": text_sha256(target.answer),
            "question_date_sha256": text_sha256(target.question_date),
            "answer_session_ids": list(target.answer_session_ids),
            "earlier_evidence_session_ids": [
                session.session_id for session in assembly.earlier_evidence
            ],
            "owned_latest_evidence_session_id": (
                assembly.owned_latest_evidence.session_id
            ),
            "owned_latest_timestamp": (
                assembly.owned_latest_evidence.timestamp.isoformat(
                    timespec="minutes"
                )
                if assembly.owned_latest_evidence.timestamp is not None
                else None
            ),
            "previous_answer_available_from_official_schema": False,
            "previous_answer_invented": False,
            "gate_uses_previous_answer": False,
        },
        "retained_probe": {
            "source_id": retained.source_id,
            "question_id": retained.question_id,
            "question_type": retained.question_type,
            "source_row_sha256": retained.source_row_sha256,
            "question_sha256": text_sha256(retained.question),
            "current_answer_sha256": text_sha256(retained.answer),
            "question_date_sha256": text_sha256(retained.question_date),
            "evidence_session_ids": [
                session.session_id for session in assembly.retained_evidence
            ],
            "single_answer_bearing_evidence_session_required": True,
            "distinct_official_non_abstention_row": True,
        },
        "tail_session_references": [
            _reference_payload(reference)
            for reference in assembly.tail_references
        ],
        "context": _context_descriptor(
            assembly,
            context_ceiling=context_ceiling,
        ),
    }


def _official_provenance() -> dict[str, Any]:
    return {
        "benchmark_label": BENCHMARK_LABEL,
        "context_deletion_adaptation": True,
        "official_longmemeval_leaderboard_score": False,
        "official_provenance": {
            "venue": "ICLR",
            "year": LONGMEMEVAL_ICLR_YEAR,
            "paper_id": LONGMEMEVAL_ICLR_PAPER_ID,
            "arxiv_id": LONGMEMEVAL_ARXIV_ID,
            "repository": LONGMEMEVAL_REPOSITORY,
            "repository_revision": LONGMEMEVAL_REPOSITORY_REVISION,
            "repository_license": LONGMEMEVAL_REPOSITORY_LICENSE,
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "revision_timestamp": DATASET_REVISION_TIMESTAMP,
            "public": True,
            "gated": False,
            "disabled": False,
            "license": DATASET_LICENSE,
            "license_source": DATASET_LICENSE_SOURCE,
            "standalone_license_file_present": (
                DATASET_HAS_STANDALONE_LICENSE_FILE
            ),
            "source_artifact": {
                "path": DATASET_ARTIFACT_PATH,
                "size_bytes": DATASET_ARTIFACT_SIZE,
                "rows": DATASET_NUM_ROWS,
                "sha256": DATASET_ARTIFACT_SHA256,
                "lfs_oid_sha256": DATASET_ARTIFACT_SHA256,
                "git_oid": DATASET_ARTIFACT_GIT_OID,
                "xet_hash": DATASET_ARTIFACT_XET_HASH,
            },
            "artifact_choice": {
                "chosen": "official cleaned oracle",
                "official_description": (
                    "only the evidence sessions are included in history"
                ),
                "preserves_answer_session_evidence": True,
                "oracle_session_order_may_be_arbitrary": True,
                "parallel_arrays_correspond_by_index": True,
                "chronology_reconstructed_from_haystack_dates": True,
                "cleaned_s_alternative": {
                    "path": CLEANED_S_ARTIFACT_PATH,
                    "size_bytes": CLEANED_S_ARTIFACT_SIZE,
                    "sha256": CLEANED_S_ARTIFACT_SHA256,
                },
                "materially_smaller_than_cleaned_s": True,
                "size_ratio_vs_cleaned_s": (
                    DATASET_ARTIFACT_SIZE / CLEANED_S_ARTIFACT_SIZE
                ),
            },
        },
    }


def _candidate_audit(
    example: LongMemEvalExample,
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "source_id": example.source_id,
        "question_id": example.question_id,
        "question_type": example.question_type,
        "source_row_sha256": example.source_row_sha256,
        "current_answer_sha256": text_sha256(example.answer),
        "data_only_rejection_reasons": list(reasons),
    }


def _safe_rejection_code(exc: Exception) -> str:
    """Convert packaging failures to source-free stable reason codes."""

    rendered = str(exc).strip()
    if re.fullmatch(r"[a-z][a-z0-9_]*", rendered):
        return rendered
    if rendered.startswith("LongMemEval target is ineligible:"):
        candidate = rendered.split(":", 1)[1].split(",", 1)[0].strip()
        if re.fullmatch(r"[a-z][a-z0-9_]*", candidate):
            return candidate
    return f"packaging_{type(exc).__name__.casefold()}"


def build_manifest(
    examples: Sequence[LongMemEvalExample],
    tokenizer: Any,
    *,
    requested_records: int = DEFAULT_RECORDS,
    candidate_limit: int | None = None,
    seed: int = DEFAULT_SEED,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    minimum_tokens_after_owned: int = (
        DEFAULT_MINIMUM_TOKENS_AFTER_OWNED
    ),
    context_ceiling: int = RUNTIME_CONTEXT_CEILING,
    excluded_source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Freeze a source-free pilot before any Gemma scoring."""

    if requested_records < 1:
        raise ValueError("requested_records must be positive")
    if minimum_tokens_after_owned < 0:
        raise ValueError("minimum_tokens_after_owned must be non-negative")
    if context_ceiling > RUNTIME_CONTEXT_CEILING or context_ceiling < 1:
        raise ValueError("context ceiling must be within the 2,048-token runtime")
    source_ids = [example.source_id for example in examples]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("LongMemEval source IDs must be unique")

    ordered_all = deterministic_source_order(examples, seed=seed)
    excluded = {str(source_id) for source_id in excluded_source_ids}
    targets = [
        example
        for example in ordered_all
        if example.question_type == KNOWLEDGE_UPDATE_TYPE
        and not is_abstention(example)
        and example.source_id not in excluded
    ]
    if candidate_limit is not None:
        if candidate_limit < requested_records:
            raise ValueError("candidate_limit must cover requested records")
        targets = targets[: int(candidate_limit)]
    audits = []
    records = []
    rejection_counts: dict[str, int] = {}

    for target in targets:
        reasons = list(target_rejection_reasons(target))
        audit = _candidate_audit(target, reasons)
        if not reasons:
            try:
                retained = deterministic_retained_pair(
                    target,
                    ordered_all,
                    seed=seed,
                )
                assembly = package_context(
                    tokenizer,
                    target,
                    retained,
                    ordered_all,
                    seed=seed,
                    minimum_tokens_after_owned=minimum_tokens_after_owned,
                    context_ceiling=context_ceiling,
                )
                payload = _record_payload(
                    target,
                    retained,
                    assembly,
                    context_ceiling=context_ceiling,
                )
            except (LookupError, ValueError) as exc:
                reasons.append(_safe_rejection_code(exc))
        if reasons:
            reasons = list(dict.fromkeys(reasons))
            audit["status"] = "rejected_data"
            audit["data_only_rejection_reasons"] = reasons
            for reason in reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        elif len(records) < requested_records:
            audit["status"] = "selected"
            audit["record_id"] = payload["record_id"]
            audit["retained_source_id"] = retained.source_id
            records.append(payload)
        else:
            audit["status"] = "eligible_not_selected_capacity"
        audits.append(audit)

    if len(records) != requested_records:
        counts = ", ".join(
            f"{key}={value}" for key, value in sorted(rejection_counts.items())
        )
        raise RuntimeError(
            f"LongMemEval selection produced {len(records)} of "
            f"{requested_records} required records ({counts or 'no candidates'})"
        )

    source_rows = [
        _example_descriptor(example)
        for example in sorted(ordered_all, key=lambda item: item.source_id)
    ]
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "benchmark_label": BENCHMARK_LABEL,
        "contains_source_text": False,
        "provenance": _official_provenance(),
        "tokenizer": {
            "model_id": str(tokenizer_id),
            "revision": str(tokenizer_revision),
            "fast_offset_mapping_required": True,
        },
        "selection_policy": {
            "seed": int(seed),
            "requested_records": int(requested_records),
            "predeclared_pilot_records": DEFAULT_RECORDS,
            "is_eight_record_pilot": requested_records == DEFAULT_RECORDS,
            "candidate_limit": (
                None if candidate_limit is None else int(candidate_limit)
            ),
            "excluded_source_ids": sorted(excluded),
            "candidate_order": (
                "sha256(seed, official question ID, normalized source ID)"
            ),
            "retained_pair_rule": (
                "shortest official single answer-bearing evidence session, "
                "then sha256 tie-break"
            ),
            "knowledge_update_only": True,
            "abstentions_excluded": True,
            "fixed_before_gemma_scoring": True,
            "gemma_outputs_used": False,
            "output_based_replacement": False,
            "all_selected_records_retained_in_evaluation": True,
            "candidate_source_ids": [
                example.source_id for example in targets
            ],
            "candidate_audit": audits,
            "data_only_rejection_counts": dict(
                sorted(rejection_counts.items())
            ),
            "selected_records": len(records),
        },
        "erasure_config": {
            "owned_unit": (
                "official answer-bearing turns plus immediate response from "
                "the chronologically latest answer/evidence session"
            ),
            "evidence_excerpt_policy": (
                "official has_answer turns plus each immediate following response"
            ),
            "all_earlier_official_evidence_excerpts_retained_chronologically": True,
            "retained_probe_uses_one_official_answer_bearing_session": True,
            "prior_answer_extraction": "none",
            "minimum_tokens_strictly_after_owned": int(
                minimum_tokens_after_owned
            ),
            "greedy_generation_token_reserve": (
                DEFAULT_GREEDY_TOKEN_RESERVE
            ),
            "runtime_context_ceiling": int(context_ceiling),
            "literal_full_repack": True,
            "retained_tokens_preserved_exactly": True,
        },
        "source_rows": source_rows,
        "records": records,
    }
    return freeze_manifest(manifest)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif _is_sequence(value):
        for item in value:
            yield from _walk_keys(item)


def _validate_without_integrity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or int(
        manifest.get("schema_version", -1)
    ) != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("unsupported LongMemEval deletion manifest")
    if manifest.get("benchmark_label") != BENCHMARK_LABEL:
        raise ManifestError("LongMemEval adaptation label is missing")
    if manifest.get("contains_source_text") is not False:
        raise ManifestError("LongMemEval manifests must be source-text free")
    leaked = _FORBIDDEN_SOURCE_KEYS.intersection(_walk_keys(manifest))
    if leaked:
        raise ManifestError(
            "source-text fields are forbidden: " + ", ".join(sorted(leaked))
        )
    if manifest.get("provenance") != _official_provenance():
        raise ManifestError("official LongMemEval provenance pin drifted")
    tokenizer = manifest.get("tokenizer") or {}
    if (
        not tokenizer.get("model_id")
        or not tokenizer.get("revision")
        or tokenizer.get("fast_offset_mapping_required") is not True
    ):
        raise ManifestError("offset tokenizer source is not pinned")
    policy = manifest.get("selection_policy") or {}
    if (
        policy.get("fixed_before_gemma_scoring") is not True
        or policy.get("gemma_outputs_used") is not False
        or policy.get("output_based_replacement") is not False
        or policy.get("all_selected_records_retained_in_evaluation") is not True
        or policy.get("knowledge_update_only") is not True
        or policy.get("abstentions_excluded") is not True
    ):
        raise ManifestError("LongMemEval source-only selection policy drifted")
    source_rows = manifest.get("source_rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise ManifestError("LongMemEval source descriptors are missing")
    source_ids = [str(item.get("source_id")) for item in source_rows]
    if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
        raise ManifestError("LongMemEval source descriptors are non-canonical")
    candidate_ids = policy.get("candidate_source_ids")
    excluded_ids = policy.get("excluded_source_ids", [])
    audits = policy.get("candidate_audit")
    if (
        not isinstance(candidate_ids, list)
        or not isinstance(excluded_ids, list)
        or excluded_ids != sorted(set(str(item) for item in excluded_ids))
        or set(candidate_ids).intersection(excluded_ids)
        or not isinstance(audits, list)
        or len(candidate_ids) != len(audits)
        or [item.get("source_id") for item in audits] != candidate_ids
    ):
        raise ManifestError("LongMemEval candidate audit is incomplete")
    valid_statuses = {
        "selected",
        "rejected_data",
        "eligible_not_selected_capacity",
    }
    if any(item.get("status") not in valid_statuses for item in audits):
        raise ManifestError("LongMemEval candidate audit status is invalid")
    expected_rejections: dict[str, int] = {}
    for item in audits:
        reasons = item.get("data_only_rejection_reasons")
        if not isinstance(reasons, list):
            raise ManifestError("candidate rejection reasons are missing")
        if reasons and item.get("status") != "rejected_data":
            raise ManifestError("data-ineligible candidate was selected")
        for reason in reasons:
            expected_rejections[str(reason)] = (
                expected_rejections.get(str(reason), 0) + 1
            )
    if policy.get("data_only_rejection_counts") != dict(
        sorted(expected_rejections.items())
    ):
        raise ManifestError("data-only rejection counts drifted")

    records = manifest.get("records")
    if (
        not isinstance(records, list)
        or len(records) != int(policy.get("requested_records", -1))
        or len(records) != int(policy.get("selected_records", -1))
    ):
        raise ManifestError("LongMemEval record count differs from request")
    record_ids = [str(record.get("record_id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ManifestError("LongMemEval record IDs are duplicated")
    selected_audit_ids = {
        item.get("record_id")
        for item in audits
        if item.get("status") == "selected"
    }
    if set(record_ids) != selected_audit_ids:
        raise ManifestError("records differ from selected candidate audit")

    erasure = manifest.get("erasure_config") or {}
    minimum = int(
        erasure.get("minimum_tokens_strictly_after_owned", -1)
    )
    greedy_reserve = int(erasure.get("greedy_generation_token_reserve", -1))
    ceiling = int(erasure.get("runtime_context_ceiling", -1))
    if (
        minimum < 0
        or greedy_reserve < DEFAULT_GREEDY_TOKEN_RESERVE
        or not 0 < ceiling <= RUNTIME_CONTEXT_CEILING
    ):
        raise ManifestError("LongMemEval token geometry is invalid")
    for record in records:
        target = record.get("target") or {}
        retained = record.get("retained_probe") or {}
        if target.get("source_id") in excluded_ids:
            raise ManifestError("holdout record overlaps an excluded source")
        if target.get("question_type") != KNOWLEDGE_UPDATE_TYPE:
            raise ManifestError("selected target is not knowledge-update")
        if str(target.get("question_id", "")).casefold().endswith("_abs"):
            raise ManifestError("selected target is an abstention row")
        if (
            target.get("previous_answer_available_from_official_schema")
            is not False
            or target.get("previous_answer_invented") is not False
            or target.get("gate_uses_previous_answer") is not False
        ):
            raise ManifestError("manifest invents or gates on a previous answer")
        earlier = target.get("earlier_evidence_session_ids")
        owned = target.get("owned_latest_evidence_session_id")
        if not isinstance(earlier, list) or not earlier or owned in earlier:
            raise ManifestError("latest/earlier evidence partition is invalid")
        if (
            retained.get("distinct_official_non_abstention_row") is not True
            or retained.get(
                "single_answer_bearing_evidence_session_required"
            )
            is not True
            or len(retained.get("evidence_session_ids") or ()) != 1
            or retained.get("source_id") == target.get("source_id")
            or str(retained.get("question_id", "")).casefold().endswith("_abs")
        ):
            raise ManifestError("retained QA is not a distinct official row")
        tail = record.get("tail_session_references")
        if not isinstance(tail, list) or not tail:
            raise ManifestError("deterministic frozen tail is missing")
        tail_keys = [
            (item.get("source_id"), item.get("session_id"))
            for item in tail
            if isinstance(item, Mapping)
        ]
        if len(tail_keys) != len(tail) or len(tail_keys) != len(set(tail_keys)):
            raise ManifestError("deterministic frozen tail is invalid")
        context = record.get("context") or {}
        if (
            context.get("literal_full_repack") is not True
            or context.get(
                "literal_retokenization_equals_position_drop"
            )
            is not True
        ):
            raise ManifestError("literal full-repack contract is missing")
        ownership = context.get("ownership") or {}
        positions = ownership.get("forget_positions")
        if (
            not isinstance(positions, list)
            or not positions
            or positions != sorted(set(int(value) for value in positions))
            or ownership.get("all_excerpt_turns_owned") is not True
            or ownership.get("full_source_session_owned") is not False
        ):
            raise ManifestError("evidence-excerpt token ownership is invalid")
        if context.get("evidence_excerpt_policy") != (
            "official has_answer turns plus each immediate following response"
        ):
            raise ManifestError("LongMemEval evidence excerpt policy drifted")
        original = int(context.get("original_token_count", -1))
        edited = int(context.get("edited_token_count", -1))
        if original - len(positions) != edited:
            raise ManifestError("literal token edit count is inconsistent")
        query_reserve = int(context.get("query_token_reserve", -1))
        runtime_bound = int(context.get("runtime_total_token_bound", -1))
        if (
            query_reserve < greedy_reserve
            or runtime_bound != original + query_reserve
        ):
            raise ManifestError("query token reserve is inconsistent")
        after = int(context.get("tokens_strictly_after_owned", -1))
        if after != original - positions[-1] - 1 or after < minimum:
            raise ManifestError("owned session is inside the local window")
        if runtime_bound > ceiling:
            raise ManifestError("context exceeds the runtime ceiling")
        retained_tokens = context.get("retained") or {}
        if (
            retained_tokens.get("tokens_preserved_exactly") is not True
            or retained_tokens.get("original_token_ids_sha256")
            != retained_tokens.get("edited_token_ids_sha256")
        ):
            raise ManifestError("literal deletion changes retained tokens")


def freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy and attach per-record and root SHA-256 integrity digests."""

    frozen = copy.deepcopy(dict(manifest))
    frozen.pop("integrity", None)
    for record in frozen.get("records", []):
        record.pop("record_integrity", None)
    _validate_without_integrity(frozen)
    for record in frozen["records"]:
        record["record_integrity"] = {
            "algorithm": "sha256",
            "sha256": _payload_sha256(record),
        }
    frozen["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": _payload_sha256(frozen),
    }
    validate_manifest(frozen)
    return frozen


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject provenance, source, selection, geometry, or integrity drift."""

    _validate_without_integrity(manifest)
    for record in manifest["records"]:
        integrity = record.get("record_integrity")
        payload = copy.deepcopy(dict(record))
        payload.pop("record_integrity", None)
        if (
            not isinstance(integrity, Mapping)
            or integrity.get("algorithm") != "sha256"
            or integrity.get("sha256") != _payload_sha256(payload)
        ):
            raise ManifestError(
                f"record integrity mismatch: {record.get('record_id')}"
            )
    integrity = manifest.get("integrity")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
    ):
        raise ManifestError("manifest integrity mismatch")


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
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def _reference_from_payload(value: Mapping[str, Any]) -> SessionReference:
    return SessionReference(
        source_id=str(value["source_id"]),
        question_id=str(value["question_id"]),
        session_id=str(value["session_id"]),
    )


def _record_for_hydration(
    stored: Mapping[str, Any],
    examples_by_source: Mapping[str, LongMemEvalExample],
    all_examples: Sequence[LongMemEvalExample],
    tokenizer: Any,
    *,
    seed: int,
    minimum_tokens_after_owned: int,
    context_ceiling: int,
) -> RehydratedLongMemEvalRecord:
    target_id = str(stored["target"]["source_id"])
    retained_id = str(stored["retained_probe"]["source_id"])
    if target_id not in examples_by_source or retained_id not in examples_by_source:
        raise ManifestError("a frozen target or retained source row is missing")
    target = examples_by_source[target_id]
    retained = examples_by_source[retained_id]
    expected_retained = deterministic_retained_pair(
        target,
        all_examples,
        seed=seed,
    )
    if expected_retained.source_id != retained.source_id:
        raise ManifestError("deterministic retained pairing drifted")
    references = tuple(
        _reference_from_payload(item)
        for item in stored["tail_session_references"]
    )
    assembly = package_context(
        tokenizer,
        target,
        retained,
        all_examples,
        seed=seed,
        minimum_tokens_after_owned=minimum_tokens_after_owned,
        context_ceiling=context_ceiling,
        frozen_tail_references=references,
    )
    regenerated = _record_payload(
        target,
        retained,
        assembly,
        context_ceiling=context_ceiling,
    )
    observed = copy.deepcopy(dict(stored))
    observed.pop("record_integrity", None)
    if regenerated != observed:
        raise ManifestError("rehydrated LongMemEval record differs from manifest")
    return RehydratedLongMemEvalRecord(
        record_id=str(stored["record_id"]),
        target=target,
        retained=retained,
        earlier_evidence=assembly.earlier_evidence,
        owned_latest_evidence=assembly.owned_latest_evidence,
        retained_evidence=assembly.retained_evidence,
        tail_references=assembly.tail_references,
        context=assembly.context,
    )


def rehydrate_manifest_from_rows(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[RehydratedLongMemEvalRecord, ...]:
    """Rebuild source text and reject source, pairing, tail, or token drift."""

    validate_manifest(manifest)
    examples = extract_longmemeval_examples(rows)
    descriptors = [
        _example_descriptor(example)
        for example in sorted(examples, key=lambda item: item.source_id)
    ]
    if descriptors != manifest["source_rows"]:
        raise ManifestError("rehydrated LongMemEval source rows drifted")
    examples_by_source = {example.source_id: example for example in examples}
    if len(examples_by_source) != len(examples):
        raise ManifestError("rehydrated LongMemEval source IDs are duplicated")
    policy = manifest["selection_policy"]
    erasure = manifest["erasure_config"]
    return tuple(
        _record_for_hydration(
            stored,
            examples_by_source,
            examples,
            tokenizer,
            seed=int(policy["seed"]),
            minimum_tokens_after_owned=int(
                erasure["minimum_tokens_strictly_after_owned"]
            ),
            context_ceiling=int(erasure["runtime_context_ceiling"]),
        )
        for stored in manifest["records"]
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pinned_longmemeval_rows(
    local_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Resolve and verify the exact cleaned oracle artifact at explicit runtime."""

    if local_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to resolve pinned LongMemEval data"
            ) from exc
        resolved = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=DATASET_ARTIFACT_PATH,
                revision=DATASET_REVISION,
            )
        )
    else:
        resolved = Path(local_path)
    if resolved.stat().st_size != DATASET_ARTIFACT_SIZE:
        raise ValueError("LongMemEval oracle size differs from the pinned artifact")
    if _sha256_path(resolved) != DATASET_ARTIFACT_SHA256:
        raise ValueError(
            "LongMemEval oracle SHA-256 differs from the pinned artifact"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != DATASET_NUM_ROWS:
        raise ValueError("LongMemEval oracle must contain exactly 500 rows")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError("LongMemEval oracle contains a non-object row")
    return tuple(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="local exact oracle JSON; omitted resolves the pinned Hub revision",
    )
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument(
        "--exclude-manifest",
        help="exclude target source IDs selected by an earlier frozen manifest",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--minimum-tokens-after-owned",
        type=int,
        default=DEFAULT_MINIMUM_TOKENS_AFTER_OWNED,
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument(
        "--tokenizer-revision",
        default=DEFAULT_TOKENIZER_REVISION,
    )
    parser.add_argument(
        "--out",
        default=(
            "outputs/gemma_sv_rag/"
            "longmemeval_knowledge_update_deletion_pilot.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the source-free manifest; no model is loaded or evaluated."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.records < 1:
        parser.error("--records must be positive")
    if args.candidate_limit is not None and args.candidate_limit < args.records:
        parser.error("--candidate-limit must cover --records")
    if args.minimum_tokens_after_owned < 0:
        parser.error("--minimum-tokens-after-owned must be non-negative")
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")

    rows = load_pinned_longmemeval_rows(args.data_path)
    examples = extract_longmemeval_examples(rows)
    excluded_source_ids = []
    if args.exclude_manifest:
        previous = load_manifest(args.exclude_manifest)
        excluded_source_ids = [
            record["target"]["source_id"] for record in previous["records"]
        ]
    tokenizer = load_offset_tokenizer(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    manifest = build_manifest(
        examples,
        tokenizer,
        requested_records=args.records,
        candidate_limit=args.candidate_limit,
        seed=args.seed,
        tokenizer_id=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        minimum_tokens_after_owned=args.minimum_tokens_after_owned,
        excluded_source_ids=excluded_source_ids,
    )
    write_manifest(output, manifest)
    write_environment_lock(output.parent / "environment-lock.txt")
    print(
        f"wrote {len(manifest['records'])} frozen LongMemEval records to "
        f"{output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

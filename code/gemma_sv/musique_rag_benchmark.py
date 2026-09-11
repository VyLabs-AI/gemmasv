"""Exploratory MuSiQue natural-QA RAG context-erasure benchmark (v2).

This module deliberately does not import Hugging Face Hub, Transformers,
LlamaIndex, or embedding packages at import time.  Its parsing, selection,
counterfactual construction, packaging, manifest, and hydration APIs are pure
Python.  Optional live dependencies are reached only from explicit runtime
helpers used by the CLI.

The v2 design is exploratory because the frozen 2Wiki v1 experiment was
already observed.  It never selects MuSiQue records from 2Wiki or Gemma
outputs, and it does not replace or suppress the v1 provenance.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv.rag_benchmark import (
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TOKENIZER_REVISION,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    LLAMA_INDEX_CORE_VERSION,
    LLAMA_INDEX_HUGGINGFACE_VERSION,
    CorpusNode,
    DocumentSpec,
    RetrievalHit,
    TokenizedContext,
    build_llamaindex_retriever as _build_llamaindex_retriever,
    load_offset_tokenizer,
    load_pinned_embedding,
    make_in_memory_retriever,
    one_node_per_document,
    split_documents_with_llamaindex,
    stable_identifier,
    text_sha256,
    token_ids_sha256,
    write_environment_lock,
)


MANIFEST_SCHEMA = "gemma-sv-musique-rag-manifest-v2"
MANIFEST_SCHEMA_VERSION = 2
PACKAGE_SCHEMA = "gemma-sv-musique-rag-context-v2"
COUNTERFACTUAL_RECIPE_VERSION = "musique-support-sentence-donor-v2"
RETAINED_PROBE_VERSION = "musique-original-supporting-qa-v2"

DATASET_ID = "bdsaglam/musique"
DATASET_REVISION = "22873a405dd809893b22ada0b499299fb612d2df"
DATASET_FILENAME = "musique_ans_v1.0_dev.jsonl"
DATASET_SPLIT = "dev"
DATASET_CONFIGURATION = "MuSiQue-Ans"
DATASET_LICENSE = "CC BY 4.0"
DATASET_FILE_SHA256 = (
    "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
)

DEFAULT_TOP_K = 6
DEFAULT_CHUNK_SIZE = 64
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_MIN_TOKENS_AFTER_OWNED = 512
DONOR_MAX_TOKEN_LENGTH_DELTA = 1

SMOKE_RECORDS = 8
SMOKE_CANDIDATES = 64
FULL_RECORDS = 64
FULL_CANDIDATES = 512

_POLAR_ANSWERS = frozenset({"yes", "no", "true", "false"})
_SOURCE_TEXT_KEYS = frozenset(
    {
        "question",
        "answer",
        "title",
        "text",
        "paragraph_text",
        "source_text",
        "original_sentence",
        "modified_sentence",
        "vector",
        "vectors",
    }
)
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
    "oct|nov|dec"
)
_ENTITY_SUFFIXES = frozenset(
    {
        "academy",
        "agency",
        "association",
        "bank",
        "college",
        "company",
        "corporation",
        "council",
        "department",
        "foundation",
        "group",
        "institute",
        "ministry",
        "museum",
        "organization",
        "party",
        "school",
        "society",
        "team",
        "university",
    }
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


def _selection_key(namespace: str, seed: int, *parts: object) -> bytes:
    framed = "\0".join(str(part) for part in parts)
    return hashlib.sha256(
        f"{namespace}\0{int(seed)}\0{framed}".encode("utf-8")
    ).digest()


def _as_bool(value: Any, *, default: bool | None = None) -> bool:
    if value is None:
        if default is None:
            raise ValueError("boolean value is missing")
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"unsupported boolean value {value!r}")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _mapping_rows(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    """Normalize list-of-mappings and mapping-of-lists dataset variants."""

    if isinstance(value, Mapping):
        lengths = {
            len(item)
            for item in value.values()
            if _is_sequence(item)
        }
        if not lengths:
            raise ValueError(f"{field} mapping has no sequence columns")
        if len(lengths) != 1:
            raise ValueError(f"{field} columns differ in length")
        length = next(iter(lengths))
        rows: list[Mapping[str, Any]] = []
        for index in range(length):
            row = {}
            for key, item in value.items():
                row[str(key)] = item[index] if _is_sequence(item) else item
            rows.append(row)
        return rows
    if _is_sequence(value):
        rows = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append(item)
            elif _is_sequence(item) and len(item) >= 2:
                rows.append({"title": item[0], "paragraph_text": item[1]})
            else:
                raise ValueError(f"{field} entry has an unsupported shape")
        return rows
    raise ValueError(f"{field} has an unsupported shape")


def _string_aliases(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = value.get("aliases") or value.get("values") or ()
    if isinstance(value, (str, bytes)):
        values = (str(value),)
    elif _is_sequence(value):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError("answer aliases must be a string sequence")
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        stripped = item.strip()
        folded = stripped.casefold()
        if stripped and folded not in seen:
            result.append(stripped)
            seen.add(folded)
    return tuple(result)


@dataclass(frozen=True)
class MusiqueDecomposition:
    decomposition_id: str
    question: str
    answer: str
    paragraph_support_index: int | None


@dataclass(frozen=True)
class MusiqueParagraph:
    source_example_id: str
    paragraph_index: int
    title: str
    paragraph_text: str
    is_supporting: bool

    @property
    def text(self) -> str:
        return self.paragraph_text

    @property
    def document_id(self) -> str:
        return stable_identifier(
            "musique-doc",
            DATASET_ID,
            DATASET_REVISION,
            self.source_example_id,
            self.paragraph_index,
        )


@dataclass(frozen=True)
class MusiqueExample:
    example_id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    answerable: bool
    paragraphs: tuple[MusiqueParagraph, ...]
    question_decomposition: tuple[MusiqueDecomposition, ...]

    @property
    def supporting_paragraphs(self) -> tuple[MusiqueParagraph, ...]:
        return tuple(
            paragraph for paragraph in self.paragraphs if paragraph.is_supporting
        )

    @property
    def supporting_document_ids(self) -> tuple[str, ...]:
        return tuple(
            paragraph.document_id for paragraph in self.supporting_paragraphs
        )


@dataclass(frozen=True)
class AnswerOccurrence:
    paragraph_index: int
    sentence_index: int
    sentence_start: int
    sentence_end: int
    answer_start: int
    answer_end: int
    sentence: str
    matched_answer: str


@dataclass(frozen=True)
class MusiqueCounterfactual:
    target_source_id: str
    donor_source_id: str
    paragraph_index: int
    title: str
    original_sentence: str
    modified_sentence: str
    gold_answer: str
    distractor_answer: str
    answer_strategy: str
    answer_start: int
    answer_end: int


@dataclass(frozen=True)
class RehydratedMusiqueRecord:
    record_id: str
    example: MusiqueExample
    retrieval_hits: tuple[RetrievalHit, ...]
    counterfactual: MusiqueCounterfactual
    context: TokenizedContext
    retained_example: MusiqueExample
    retained_paragraph: MusiqueParagraph
    retained_question: str
    retained_answer: str


@dataclass(frozen=True)
class _CandidatePlan:
    example: MusiqueExample
    occurrence: AnswerOccurrence | None
    donor: MusiqueExample | None
    retained: MusiqueExample | None
    retained_occurrence: AnswerOccurrence | None
    rejection_reason: str | None


def _extract_answer(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    raw_answer = row.get("answer")
    if isinstance(raw_answer, Mapping):
        answer = str(
            raw_answer.get("value")
            or raw_answer.get("text")
            or raw_answer.get("answer")
            or ""
        ).strip()
        nested_aliases = _string_aliases(raw_answer.get("aliases"))
    else:
        answer = str(
            raw_answer
            or row.get("gold_answer")
            or row.get("golden_answer")
            or ""
        ).strip()
        nested_aliases = ()
    golden = _string_aliases(row.get("golden_answers"))
    if not answer and golden:
        answer = golden[0]
    aliases = (
        nested_aliases
        + _string_aliases(row.get("answer_aliases"))
        + _string_aliases(row.get("aliases"))
        + golden
    )
    deduplicated: list[str] = []
    seen = {answer.casefold()} if answer else set()
    for alias in aliases:
        if alias.casefold() not in seen:
            deduplicated.append(alias)
            seen.add(alias.casefold())
    return answer, tuple(deduplicated)


def _parse_decomposition(value: Any) -> tuple[MusiqueDecomposition, ...]:
    if value is None:
        return ()
    result = []
    for position, item in enumerate(
        _mapping_rows(value, field="question_decomposition")
    ):
        raw_support = item.get("paragraph_support_idx")
        if raw_support is None:
            raw_support = item.get("paragraph_support_index")
        support_index = None if raw_support is None else int(raw_support)
        result.append(
            MusiqueDecomposition(
                decomposition_id=str(item.get("id") or position),
                question=str(item.get("question") or "").strip(),
                answer=str(item.get("answer") or "").strip(),
                paragraph_support_index=support_index,
            )
        )
    return tuple(result)


def parse_musique_row(row: Mapping[str, Any]) -> MusiqueExample:
    """Normalize one standard MuSiQue row without optional dependencies."""

    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    example_id = str(row.get("id") or row.get("_id") or "").strip()
    question = str(row.get("question") or "").strip()
    answer, aliases = _extract_answer(row)
    if not example_id or not question or not answer:
        raise ValueError("MuSiQue row is missing id, question, or answer")
    answerable_value = (
        row.get("answerable")
        if "answerable" in row
        else metadata.get("answerable")
    )
    answerable = _as_bool(answerable_value, default=True)
    raw_decomposition = row.get("question_decomposition")
    if raw_decomposition is None:
        raw_decomposition = metadata.get("question_decomposition")
    decomposition = _parse_decomposition(raw_decomposition)
    support_references = {
        item.paragraph_support_index
        for item in decomposition
        if item.paragraph_support_index is not None
    }

    raw_paragraphs = row.get("paragraphs")
    if raw_paragraphs is None:
        raw_paragraphs = row.get("contexts") or row.get("context")
    paragraph_rows = _mapping_rows(raw_paragraphs, field="paragraphs")
    preliminary = []
    for position, item in enumerate(paragraph_rows):
        raw_index = item.get("idx")
        if raw_index is None:
            raw_index = item.get("paragraph_idx")
        if raw_index is None:
            raw_index = item.get("id")
        paragraph_index = position if raw_index is None else int(raw_index)
        title = str(item.get("title") or "").strip()
        paragraph_text = str(
            item.get("paragraph_text")
            or item.get("text")
            or item.get("paragraph")
            or ""
        ).strip()
        if not paragraph_text:
            raise ValueError(
                f"MuSiQue row {example_id!r} has an empty paragraph"
            )
        raw_support = item.get("is_supporting")
        explicitly_supporting = (
            False if raw_support is None else _as_bool(raw_support)
        )
        preliminary.append(
            (
                position,
                paragraph_index,
                title,
                paragraph_text,
                explicitly_supporting,
            )
        )
    indices = [item[1] for item in preliminary]
    if len(indices) != len(set(indices)):
        raise ValueError(f"MuSiQue row {example_id!r} has duplicate paragraph idx")

    paragraphs = tuple(
        MusiqueParagraph(
            source_example_id=example_id,
            paragraph_index=paragraph_index,
            title=title,
            paragraph_text=paragraph_text,
            is_supporting=(
                explicit
                or paragraph_index in support_references
                or position in support_references
            ),
        )
        for position, paragraph_index, title, paragraph_text, explicit in preliminary
    )
    if not paragraphs:
        raise ValueError(f"MuSiQue row {example_id!r} has no paragraphs")
    if answerable and not any(item.is_supporting for item in paragraphs):
        raise ValueError(
            f"answerable MuSiQue row {example_id!r} has no supporting paragraph"
        )
    return MusiqueExample(
        example_id=example_id,
        question=question,
        answer=answer,
        answer_aliases=aliases,
        answerable=answerable,
        paragraphs=paragraphs,
        question_decomposition=decomposition,
    )


def extract_musique_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    limit: int | None = None,
    required_ids: Iterable[str] | None = None,
) -> tuple[MusiqueExample, ...]:
    """Parse rows and apply a deterministic, source-only candidate order."""

    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be positive")
    required = (
        None if required_ids is None else {str(value) for value in required_ids}
    )
    parsed = []
    for row in rows:
        raw_id = str(row.get("id") or row.get("_id") or "")
        if required is not None and raw_id not in required:
            continue
        parsed.append(parse_musique_row(row))
    identifiers = [example.example_id for example in parsed]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("MuSiQue source contains duplicate example IDs")
    if required is not None:
        missing = required.difference(identifiers)
        if missing:
            raise ValueError(
                f"MuSiQue source is missing {len(missing)} manifest IDs"
            )
    ordered = sorted(
        parsed,
        key=lambda example: (
            _selection_key(
                "musique-candidate-order-v2",
                seed,
                example.example_id,
            ),
            example.example_id,
        ),
    )
    if limit is not None:
        ordered = ordered[: int(limit)]
    return tuple(ordered)


def _sentence_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return deterministic sentence spans while preserving exact source text."""

    source = str(text)
    boundaries = [
        match.end()
        for match in re.finditer(
            r"[.!?]+(?:[\"')\]]*)?(?=\s+|$)",
            source,
        )
    ]
    if not boundaries or boundaries[-1] != len(source):
        boundaries.append(len(source))
    result = []
    start = 0
    for boundary in boundaries:
        raw = source[start:boundary]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            exact_start = start + left
            exact_end = start + right
            result.append(
                (exact_start, exact_end, source[exact_start:exact_end])
            )
        start = boundary
    return tuple(result)


def _answer_pattern(answer: str) -> re.Pattern[str]:
    stripped = str(answer).strip()
    if not stripped:
        raise ValueError("answer must be non-empty")
    left = r"(?<!\w)" if stripped[0].isalnum() else ""
    right = r"(?!\w)" if stripped[-1].isalnum() else ""
    return re.compile(left + re.escape(stripped) + right, re.IGNORECASE)


def _contains_exact(text: str, answer: str) -> bool:
    return _answer_pattern(answer).search(str(text)) is not None


def find_supporting_answer_occurrence(
    example: MusiqueExample,
) -> AnswerOccurrence | None:
    """Find the first exact case-insensitive gold occurrence in real support."""

    pattern = _answer_pattern(example.answer)
    for paragraph in example.paragraphs:
        if not paragraph.is_supporting:
            continue
        for sentence_index, (start, end, sentence) in enumerate(
            _sentence_spans(paragraph.paragraph_text)
        ):
            match = pattern.search(sentence)
            if match is None:
                continue
            return AnswerOccurrence(
                paragraph_index=paragraph.paragraph_index,
                sentence_index=sentence_index,
                sentence_start=start,
                sentence_end=end,
                answer_start=start + match.start(),
                answer_end=start + match.end(),
                sentence=sentence,
                matched_answer=match.group(0),
            )
    return None


def coarse_answer_type(answer: str, question: str = "") -> str:
    """Classify answers into the declared deterministic donor strata."""

    value = str(answer).strip()
    folded_question = str(question).strip().casefold()
    if re.search(rf"\b(?:{_MONTHS})\b", value, re.IGNORECASE):
        return "date"
    if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2}|2100)", value):
        return "date"
    if re.fullmatch(
        r"\d{1,4}(?:[-/.]\d{1,2}){1,2}",
        value,
    ):
        return "date"
    if re.search(r"\b(when|what year|which year|what date)\b", folded_question):
        if re.search(r"\d", value):
            return "date"
    if re.fullmatch(
        r"[+-]?(?:[$£€]\s*)?\d[\d,]*(?:\.\d+)?(?:\s*%|\s+\w+)?",
        value,
    ):
        return "number"
    if re.search(
        (
            r"\b(who|whom|whose|person|father|mother|spouse|husband|wife|"
            r"author|actor|actress|director|founder|inventor|composer|artist|"
            r"player|president|king|queen)\b"
        ),
        folded_question,
    ):
        return "person-like"
    if folded_question:
        return "entity-like"
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", value)
    if 2 <= len(words) <= 4 and all(word[:1].isupper() for word in words):
        if words[-1].casefold() not in _ENTITY_SUFFIXES:
            return "person-like"
    return "entity-like"


def _whitespace_token_count(value: str) -> int:
    return len(str(value).split())


def _target_visible_text(example: MusiqueExample) -> str:
    return "\n".join(
        (
            example.question,
            example.answer,
            *example.answer_aliases,
            *(paragraph.title for paragraph in example.paragraphs),
            *(paragraph.paragraph_text for paragraph in example.paragraphs),
        )
    )


def _basic_eligibility_reason(
    example: MusiqueExample,
    occurrence: AnswerOccurrence | None,
) -> str | None:
    if not example.answerable:
        return "not_answerable"
    if example.answer.strip().casefold() in _POLAR_ANSWERS:
        return "polar_answer"
    if occurrence is None:
        return "gold_not_found_exactly_in_supporting_sentence"
    return None


def select_donor(
    target: MusiqueExample,
    candidates: Sequence[MusiqueExample],
    *,
    seed: int = 0,
) -> MusiqueExample | None:
    """Choose a deterministic same-type, similar-length untouched donor."""

    target_type = coarse_answer_type(target.answer, target.question)
    target_length = _whitespace_token_count(target.answer)
    visible = _target_visible_text(target)
    eligible = []
    for candidate in candidates:
        occurrence = find_supporting_answer_occurrence(candidate)
        if candidate.example_id == target.example_id:
            continue
        if _basic_eligibility_reason(candidate, occurrence) is not None:
            continue
        if candidate.answer.strip().casefold() == target.answer.strip().casefold():
            continue
        if coarse_answer_type(candidate.answer, candidate.question) != target_type:
            continue
        length_delta = abs(
            _whitespace_token_count(candidate.answer) - target_length
        )
        if length_delta > DONOR_MAX_TOKEN_LENGTH_DELTA:
            continue
        if _contains_exact(visible, candidate.answer):
            continue
        eligible.append((length_delta, candidate))
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            item[0],
            _selection_key(
                "musique-donor-v2",
                seed,
                target.example_id,
                item[1].example_id,
            ),
            item[1].example_id,
        ),
    )[1]


def select_retained_probe(
    target: MusiqueExample,
    candidates: Sequence[MusiqueExample],
    *,
    seed: int = 0,
    forbidden_answers: Sequence[str] = (),
    forbidden_source_ids: Sequence[str] = (),
) -> tuple[MusiqueExample, AnswerOccurrence] | None:
    """Choose an unrelated record's original QA and supporting paragraph."""

    forbidden_sources = {str(value) for value in forbidden_source_ids}
    eligible = []
    for candidate in candidates:
        if (
            candidate.example_id == target.example_id
            or candidate.example_id in forbidden_sources
        ):
            continue
        occurrence = find_supporting_answer_occurrence(candidate)
        contains_forbidden = any(
            _contains_exact(_target_visible_text(candidate), response)
            for response in forbidden_answers
        )
        if (
            _basic_eligibility_reason(candidate, occurrence) is None
            and not contains_forbidden
        ):
            assert occurrence is not None
            eligible.append((candidate, occurrence))
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            _selection_key(
                "musique-retained-probe-v2",
                seed,
                target.example_id,
                item[0].example_id,
            ),
            item[0].example_id,
        ),
    )


def build_counterfactual(
    target: MusiqueExample,
    donor: MusiqueExample,
    occurrence: AnswerOccurrence | None = None,
) -> MusiqueCounterfactual:
    """Replace one exact gold occurrence in a real support sentence."""

    selected = occurrence or find_supporting_answer_occurrence(target)
    if selected is None:
        raise ValueError("target has no exact answer occurrence in support")
    selected_type = coarse_answer_type(target.answer, target.question)
    donor_type = coarse_answer_type(donor.answer, donor.question)
    if selected_type != donor_type:
        raise ValueError("target and donor answers have different coarse types")
    if abs(
        _whitespace_token_count(target.answer)
        - _whitespace_token_count(donor.answer)
    ) > DONOR_MAX_TOKEN_LENGTH_DELTA:
        raise ValueError("target and donor answer lengths are not similar")
    if target.answer.strip().casefold() == donor.answer.strip().casefold():
        raise ValueError("donor answer equals the target answer")
    if _contains_exact(_target_visible_text(target), donor.answer):
        raise ValueError("donor answer already occurs in the target context")
    paragraph = next(
        (
            item
            for item in target.paragraphs
            if item.paragraph_index == selected.paragraph_index
        ),
        None,
    )
    if paragraph is None or not paragraph.is_supporting:
        raise ValueError("answer occurrence does not identify target support")
    local_start = selected.answer_start - selected.sentence_start
    local_end = selected.answer_end - selected.sentence_start
    if (
        local_start < 0
        or local_end > len(selected.sentence)
        or selected.sentence[local_start:local_end].casefold()
        != target.answer.casefold()
    ):
        raise ValueError("answer occurrence drifted from the source sentence")
    modified = (
        selected.sentence[:local_start]
        + donor.answer
        + selected.sentence[local_end:]
    )
    return MusiqueCounterfactual(
        target_source_id=target.example_id,
        donor_source_id=donor.example_id,
        paragraph_index=selected.paragraph_index,
        title=paragraph.title,
        original_sentence=selected.sentence,
        modified_sentence=modified,
        gold_answer=target.answer,
        distractor_answer=donor.answer,
        answer_strategy="same_type_similar_length_cross_record_answer",
        answer_start=local_start,
        answer_end=local_start + len(donor.answer),
    )


def _normalized_source_payload(example: MusiqueExample) -> dict[str, Any]:
    return {
        "id": example.example_id,
        "question": example.question,
        "answer": example.answer,
        "answer_aliases": list(example.answer_aliases),
        "answerable": example.answerable,
        "paragraphs": [
            {
                "idx": paragraph.paragraph_index,
                "title": paragraph.title,
                "paragraph_text": paragraph.paragraph_text,
                "is_supporting": paragraph.is_supporting,
            }
            for paragraph in example.paragraphs
        ],
        "question_decomposition": [
            {
                "id": item.decomposition_id,
                "question": item.question,
                "answer": item.answer,
                "paragraph_support_idx": item.paragraph_support_index,
            }
            for item in example.question_decomposition
        ],
    }


def source_fingerprint(example: MusiqueExample) -> str:
    """Hash every normalized source field without exposing source text."""

    return _payload_sha256(_normalized_source_payload(example))


def build_document_specs(
    examples: Sequence[MusiqueExample],
) -> tuple[DocumentSpec, ...]:
    """Convert MuSiQue paragraphs into stable source documents."""

    documents = []
    for example in examples:
        for paragraph in example.paragraphs:
            supporting_sentences = (
                tuple(item[2] for item in _sentence_spans(paragraph.text))
                if paragraph.is_supporting
                else ()
            )
            documents.append(
                DocumentSpec(
                    document_id=paragraph.document_id,
                    source_example_id=example.example_id,
                    paragraph_index=paragraph.paragraph_index,
                    title=paragraph.title,
                    text=paragraph.text,
                    supporting_sentences=supporting_sentences,
                )
            )
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("stable MuSiQue document IDs are not unique")
    return tuple(documents)


def build_production_retriever(
    documents: Sequence[DocumentSpec],
    *,
    embed_model: Any | None = None,
) -> tuple[tuple[CorpusNode, ...], Any]:
    """Build the one fixed pinned BGE/LlamaIndex candidate-corpus index."""

    return _build_llamaindex_retriever(
        documents,
        top_k=DEFAULT_TOP_K,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        embed_model=embed_model,
    )


def _canonical_hits(
    hits: Sequence[RetrievalHit],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[RetrievalHit, ...]:
    if len(hits) < int(top_k):
        raise ValueError("retriever returned fewer hits than fixed top_k")
    node_ids = [hit.node.node_id for hit in hits]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("retriever returned duplicate node IDs")
    if any(not math.isfinite(float(hit.score)) for hit in hits):
        raise ValueError("retrieval score is not finite")
    ordered = sorted(
        hits,
        key=lambda hit: (-float(hit.score), hit.node.node_id),
    )[: int(top_k)]
    return tuple(
        RetrievalHit(
            node=hit.node,
            score=float(f"{float(hit.score):.12g}"),
            rank=rank,
        )
        for rank, hit in enumerate(ordered, start=1)
    )


def _tokenizer_payload(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
    except TypeError as exc:
        raise TypeError(
            "tokenizer must support return_offsets_mapping"
        ) from exc
    if isinstance(encoded, Mapping):
        raw_ids = encoded.get("input_ids")
        raw_offsets = encoded.get("offset_mapping")
    else:
        raw_ids = getattr(encoded, "input_ids", None)
        raw_offsets = getattr(encoded, "offset_mapping", None)
    if raw_ids is None or raw_offsets is None:
        raise ValueError("tokenizer omitted input_ids or offset_mapping")
    if raw_ids and _is_sequence(raw_ids[0]):
        if len(raw_ids) != 1 or len(raw_offsets) != 1:
            raise ValueError("batched tokenization is unsupported")
        raw_ids = raw_ids[0]
        raw_offsets = raw_offsets[0]
    ids = [int(value) for value in raw_ids]
    offsets = [(int(item[0]), int(item[1])) for item in raw_offsets]
    if len(ids) != len(offsets):
        raise ValueError("token IDs and offsets differ in length")
    return ids, offsets


def _overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    start, end = offset
    span_start, span_end = span
    return end > start and start < span_end and end > span_start


def _contiguous_ranges(
    positions: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    normalized = tuple(sorted({int(position) for position in positions}))
    if not normalized:
        raise ValueError("owned block has no model tokens")
    result = []
    start = previous = normalized[0]
    for position in normalized[1:]:
        if position != previous + 1:
            result.append((start, previous + 1))
            start = position
        previous = position
    result.append((start, previous + 1))
    return tuple(result)


def _shift_positions(
    positions: Sequence[int],
    deleted: Sequence[int],
) -> tuple[int, ...]:
    deleted_values = tuple(sorted(int(value) for value in deleted))
    deleted_set = set(deleted_values)
    shifted = []
    for raw_position in positions:
        position = int(raw_position)
        if position in deleted_set:
            raise ValueError("retained and owned token positions overlap")
        shifted.append(
            position - sum(value < position for value in deleted_values)
        )
    return tuple(shifted)


def tokens_after_owned(context: TokenizedContext) -> int:
    """Count model tokens strictly after the final owned token."""

    return len(context.original_token_ids) - max(context.forget_positions) - 1


def package_context(
    tokenizer: Any,
    counterfactual: MusiqueCounterfactual,
    retrieval_hits: Sequence[RetrievalHit],
    gold_node_ids: Sequence[str],
    retained_example: MusiqueExample,
    retained_paragraph: MusiqueParagraph,
    *,
    tail_nodes: Sequence[CorpusNode] = (),
) -> TokenizedContext:
    """Package trace evidence and map the exact inserted block token ownership."""

    if not retrieval_hits:
        raise ValueError("at least one retrieval hit is required")
    gold = set(str(value) for value in gold_node_ids)
    gold_ranks = [
        int(hit.rank) for hit in retrieval_hits if hit.node.node_id in gold
    ]
    if not gold_ranks:
        raise ValueError("context has no frozen retrieved gold node")
    insertion_after_rank = max(gold_ranks)
    pieces = [
        "Retrieved MuSiQue reference documents (exploratory v2):\n\n"
    ]
    owned_span: tuple[int, int] | None = None
    answer_span: tuple[int, int] | None = None

    for hit in retrieval_hits:
        pieces.append(f"[retrieval rank {hit.rank}] {hit.node.title}\n")
        pieces.append(hit.node.text)
        pieces.append("\n\n")
        if int(hit.rank) == insertion_after_rank:
            owned_start = sum(len(piece) for piece in pieces)
            block_prefix = (
                "[inserted counterfactual support sentence]\n"
                f"{counterfactual.title}\n"
            )
            pieces.append(block_prefix)
            sentence_start = sum(len(piece) for piece in pieces)
            pieces.append(counterfactual.modified_sentence)
            pieces.append("\n\n")
            owned_end = sum(len(piece) for piece in pieces)
            owned_span = (owned_start, owned_end)
            answer_span = (
                sentence_start + counterfactual.answer_start,
                sentence_start + counterfactual.answer_end,
            )
    if owned_span is None or answer_span is None:
        raise ValueError("counterfactual insertion rule was not applied")

    retained_start = sum(len(piece) for piece in pieces)
    pieces.append(
        "[retained original MuSiQue QA support]\n"
        f"{retained_paragraph.title}\n"
    )
    pieces.append(retained_paragraph.text)
    pieces.append("\n\n")
    retained_end = sum(len(piece) for piece in pieces)
    for index, node in enumerate(tail_nodes, start=1):
        pieces.append(f"[deterministic context tail {index}] {node.title}\n")
        pieces.append(node.text)
        pieces.append("\n\n")
    original_text = "".join(pieces)

    ids, offsets = _tokenizer_payload(tokenizer, original_text)
    forget = tuple(
        index
        for index, offset in enumerate(offsets)
        if _overlaps(offset, owned_span)
    )
    if not forget:
        raise ValueError("owned character block maps to no model tokens")
    forget_set = set(forget)
    retained = tuple(
        index
        for index, offset in enumerate(offsets)
        if _overlaps(offset, (retained_start, retained_end))
        and index not in forget_set
    )
    if not retained:
        raise ValueError("retained support maps to no model tokens")

    coverage = [False] * (owned_span[1] - owned_span[0])
    for position in forget:
        start, end = offsets[position]
        for character in range(
            max(start, owned_span[0]),
            min(end, owned_span[1]),
        ):
            coverage[character - owned_span[0]] = True
    owned_source = original_text[owned_span[0] : owned_span[1]]
    if any(
        not covered and not character.isspace()
        for character, covered in zip(owned_source, coverage)
    ):
        raise ValueError("offset mapping does not cover the complete owned block")

    edited_text = (
        original_text[: owned_span[0]] + original_text[owned_span[1] :]
    )
    edited_ids = tuple(
        int(token_id)
        for index, token_id in enumerate(ids)
        if index not in forget_set
    )
    return TokenizedContext(
        original_text=original_text,
        edited_text=edited_text,
        original_token_ids=tuple(ids),
        edited_token_ids=edited_ids,
        offset_mapping=tuple(offsets),
        forget_positions=forget,
        deletion_ranges=_contiguous_ranges(forget),
        retained_positions=retained,
        edited_retained_positions=_shift_positions(retained, forget),
        owned_character_span=owned_span,
        answer_character_span=answer_span,
        retained_character_span=(retained_start, retained_end),
    )


def _context_descriptor(
    context: TokenizedContext,
    *,
    insertion_after_rank: int,
    tail_node_ids: Sequence[str],
) -> dict[str, Any]:
    owned_ids = tuple(
        context.original_token_ids[position]
        for position in context.forget_positions
    )
    retained_ids = tuple(
        context.original_token_ids[position]
        for position in context.retained_positions
    )
    edited_retained_ids = tuple(
        context.edited_token_ids[position]
        for position in context.edited_retained_positions
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
        "edited_token_ids_sha256": token_ids_sha256(context.edited_token_ids),
        "literal_token_edit": True,
        "tokens_strictly_after_owned_block": tokens_after_owned(context),
        "layout": {
            "counterfactual_insertion_rule": (
                "after highest-ranked frozen retrieved gold chunk, before "
                "remaining retrieval hits and retained deterministic suffix"
            ),
            "insertion_after_retrieval_rank": int(insertion_after_rank),
            "tail_node_ids": [str(value) for value in tail_node_ids],
        },
        "ownership": {
            "mapping": "positive character overlap from tokenizer offset_mapping",
            "owned_character_span": list(context.owned_character_span),
            "owned_block_sha256": text_sha256(
                context.original_text[
                    context.owned_character_span[0] :
                    context.owned_character_span[1]
                ]
            ),
            "owned_token_ids_sha256": token_ids_sha256(owned_ids),
            "distractor_response_character_span": list(
                context.answer_character_span
            ),
            "forget_positions": list(context.forget_positions),
            "deletion_ranges": [
                {"start": start, "end": end}
                for start, end in context.deletion_ranges
            ],
            "owned_token_offsets": [
                {
                    "position": position,
                    "start": context.offset_mapping[position][0],
                    "end": context.offset_mapping[position][1],
                }
                for position in context.forget_positions
            ],
            "offset_mapping_sha256": _payload_sha256(
                [list(item) for item in context.offset_mapping]
            ),
        },
        "retained": {
            "character_span": list(context.retained_character_span),
            "token_positions": list(context.retained_positions),
            "edited_token_positions": list(context.edited_retained_positions),
            "token_count": len(context.retained_positions),
            "support_block_sha256": text_sha256(
                context.original_text[
                    context.retained_character_span[0] :
                    context.retained_character_span[1]
                ]
            ),
            "original_token_ids_sha256": token_ids_sha256(retained_ids),
            "edited_token_ids_sha256": token_ids_sha256(
                edited_retained_ids
            ),
            "tokens_preserved_exactly": retained_ids == edited_retained_ids,
        },
    }


def _plan_candidates(
    examples: Sequence[MusiqueExample],
    *,
    seed: int,
) -> tuple[_CandidatePlan, ...]:
    occurrences = {
        example.example_id: find_supporting_answer_occurrence(example)
        for example in examples
    }
    plans = []
    for example in examples:
        occurrence = occurrences[example.example_id]
        reason = _basic_eligibility_reason(example, occurrence)
        donor = None
        retained = None
        retained_occurrence = None
        if reason is None:
            donor = select_donor(example, examples, seed=seed)
            if donor is None:
                reason = "no_same_type_similar_length_donor"
        if reason is None:
            assert donor is not None
            retained_selection = select_retained_probe(
                example,
                examples,
                seed=seed,
                forbidden_answers=(donor.answer,),
                forbidden_source_ids=(donor.example_id,),
            )
            if retained_selection is None:
                reason = "no_unrelated_original_retained_qa"
            else:
                retained, retained_occurrence = retained_selection
        plans.append(
            _CandidatePlan(
                example=example,
                occurrence=occurrence,
                donor=donor,
                retained=retained,
                retained_occurrence=retained_occurrence,
                rejection_reason=reason,
            )
        )
    return tuple(plans)


def _candidate_pool_payload(
    plans: Sequence[_CandidatePlan],
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": plan.example.example_id,
            "normalized_source_sha256": source_fingerprint(plan.example),
            "data_only_eligible": plan.rejection_reason is None,
            "data_only_rejection_reason": plan.rejection_reason,
            "donor_source_id": (
                None if plan.donor is None else plan.donor.example_id
            ),
            "retained_source_id": (
                None if plan.retained is None else plan.retained.example_id
            ),
        }
        for plan in plans
    ]


def _corpus_inventory(nodes: Sequence[CorpusNode]) -> list[dict[str, Any]]:
    return [
        {
            "node_id": node.node_id,
            "document_id": node.document_id,
            "source_id": node.source_example_id,
            "paragraph_index": node.paragraph_index,
            "chunk_index": node.chunk_index,
            "node_content_sha256": text_sha256(node.text),
            "document_heading_sha256": text_sha256(node.title),
            "supporting_sentence_hashes": list(
                node.supporting_sentence_hashes
            ),
        }
        for node in sorted(nodes, key=lambda item: item.node_id)
    ]


def _retrieval_payload(
    hits: Sequence[RetrievalHit],
) -> list[dict[str, Any]]:
    return [
        {
            "rank": hit.rank,
            "score": hit.score,
            "node_id": hit.node.node_id,
            "document_id": hit.node.document_id,
            "source_id": hit.node.source_example_id,
            "paragraph_index": hit.node.paragraph_index,
            "chunk_index": hit.node.chunk_index,
            "node_content_sha256": text_sha256(hit.node.text),
            "document_heading_sha256": text_sha256(hit.node.title),
        }
        for hit in hits
    ]


def _record_payload(
    plan: _CandidatePlan,
    hits: Sequence[RetrievalHit],
    gold_node_ids: Sequence[str],
    counterfactual: MusiqueCounterfactual,
    retained_paragraph: MusiqueParagraph,
    context: TokenizedContext,
    tail_nodes: Sequence[CorpusNode],
) -> dict[str, Any]:
    if (
        plan.occurrence is None
        or plan.donor is None
        or plan.retained is None
        or plan.retained_occurrence is None
    ):
        raise ValueError("cannot serialize an ineligible candidate plan")
    target = plan.example
    donor = plan.donor
    retained = plan.retained
    insertion_rank = max(
        hit.rank for hit in hits if hit.node.node_id in set(gold_node_ids)
    )
    record_id = stable_identifier(
        "musique-rag-v2",
        DATASET_ID,
        DATASET_REVISION,
        target.example_id,
    )
    return {
        "record_id": record_id,
        "target_source": {
            "source_id": target.example_id,
            "normalized_source_sha256": source_fingerprint(target),
            "natural_question_sha256": text_sha256(target.question),
            "gold_response_sha256": text_sha256(target.answer),
            "aliases_sha256": _payload_sha256(list(target.answer_aliases)),
            "coarse_response_type": coarse_answer_type(
                target.answer,
                target.question,
            ),
            "whitespace_token_length": _whitespace_token_count(target.answer),
            "supporting_document_ids": list(target.supporting_document_ids),
            "occurrence": {
                "paragraph_index": plan.occurrence.paragraph_index,
                "sentence_index": plan.occurrence.sentence_index,
                "sentence_start": plan.occurrence.sentence_start,
                "sentence_end": plan.occurrence.sentence_end,
                "response_start": plan.occurrence.answer_start,
                "response_end": plan.occurrence.answer_end,
                "sentence_sha256": text_sha256(plan.occurrence.sentence),
                "matched_response_sha256": text_sha256(
                    plan.occurrence.matched_answer
                ),
            },
        },
        "donor_source": {
            "source_id": donor.example_id,
            "normalized_source_sha256": source_fingerprint(donor),
            "natural_question_sha256": text_sha256(donor.question),
            "donor_response_sha256": text_sha256(donor.answer),
            "coarse_response_type": coarse_answer_type(
                donor.answer,
                donor.question,
            ),
            "whitespace_token_length": _whitespace_token_count(donor.answer),
        },
        "counterfactual": {
            "recipe_version": COUNTERFACTUAL_RECIPE_VERSION,
            "response_strategy": counterfactual.answer_strategy,
            "source_sentence_sha256": text_sha256(
                counterfactual.original_sentence
            ),
            "modified_sentence_sha256": text_sha256(
                counterfactual.modified_sentence
            ),
            "distractor_response_sha256": text_sha256(
                counterfactual.distractor_answer
            ),
            "replacement_start": counterfactual.answer_start,
            "replacement_end": counterfactual.answer_end,
            "distractor_response_unique_to_owned_block": True,
        },
        "retrieval": _retrieval_payload(hits),
        "gold_evidence": {
            "retrieved_node_ids": [str(value) for value in gold_node_ids],
            "all_supporting_documents_retrieved": True,
            "answer_bearing_gold_chunk_retrieved": True,
        },
        "retained_probe": {
            "recipe_version": RETAINED_PROBE_VERSION,
            "source_id": retained.example_id,
            "normalized_source_sha256": source_fingerprint(retained),
            "natural_question_sha256": text_sha256(retained.question),
            "gold_response_sha256": text_sha256(retained.answer),
            "support_document_id": retained_paragraph.document_id,
            "support_paragraph_index": retained_paragraph.paragraph_index,
            "support_content_sha256": text_sha256(retained_paragraph.text),
            "support_heading_sha256": text_sha256(retained_paragraph.title),
        },
        "context": _context_descriptor(
            context,
            insertion_after_rank=insertion_rank,
            tail_node_ids=[node.node_id for node in tail_nodes],
        ),
    }


def _observed_package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for distribution in (
        "huggingface-hub",
        "transformers",
        "llama-index-core",
        "llama-index-embeddings-huggingface",
    ):
        try:
            result[distribution] = package_version(distribution)
        except PackageNotFoundError:
            result[distribution] = "unavailable"
    return result


def _select_tail_and_package(
    tokenizer: Any,
    counterfactual: MusiqueCounterfactual,
    hits: Sequence[RetrievalHit],
    gold_node_ids: Sequence[str],
    retained: MusiqueExample,
    retained_paragraph: MusiqueParagraph,
    nodes: Sequence[CorpusNode],
    *,
    seed: int,
    minimum_tokens_after_owned: int,
) -> tuple[TokenizedContext, tuple[CorpusNode, ...]]:
    used_ids = {hit.node.node_id for hit in hits}
    candidates = sorted(
        (
            node
            for node in nodes
            if node.node_id not in used_ids
            and node.source_example_id != counterfactual.target_source_id
            and not _contains_exact(
                f"{node.title}\n{node.text}",
                counterfactual.distractor_answer,
            )
        ),
        key=lambda node: (
            _selection_key(
                "musique-context-tail-v2",
                seed,
                counterfactual.target_source_id,
                node.node_id,
            ),
            node.node_id,
        ),
    )
    tail: list[CorpusNode] = []
    context = package_context(
        tokenizer,
        counterfactual,
        hits,
        gold_node_ids,
        retained,
        retained_paragraph,
        tail_nodes=tail,
    )
    for node in candidates:
        if tokens_after_owned(context) >= int(minimum_tokens_after_owned):
            break
        tail.append(node)
        context = package_context(
            tokenizer,
            counterfactual,
            hits,
            gold_node_ids,
            retained,
            retained_paragraph,
            tail_nodes=tail,
        )
    if tokens_after_owned(context) < int(minimum_tokens_after_owned):
        raise LookupError("insufficient_tokens_after_owned_block")
    return context, tuple(tail)


def build_manifest(
    examples: Sequence[MusiqueExample],
    nodes: Sequence[CorpusNode],
    retriever: Any,
    tokenizer: Any,
    *,
    requested_records: int,
    candidate_limit: int | None = None,
    seed: int = 0,
    mode: str = "exploratory",
    top_k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    minimum_tokens_after_owned: int = DEFAULT_MIN_TOKENS_AFTER_OWNED,
    package_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze a text-free manifest before any Gemma model scoring."""

    if requested_records < 1:
        raise ValueError("requested_records must be positive")
    if (
        int(top_k) != DEFAULT_TOP_K
        or int(chunk_size) != DEFAULT_CHUNK_SIZE
        or int(chunk_overlap) != DEFAULT_CHUNK_OVERLAP
    ):
        raise ValueError("MuSiQue v2 retrieval geometry is fixed at 6/64/0")
    if minimum_tokens_after_owned < DEFAULT_MIN_TOKENS_AFTER_OWNED:
        raise ValueError("MuSiQue v2 requires at least 512 suffix model tokens")
    ordered = sorted(
        examples,
        key=lambda example: (
            _selection_key(
                "musique-candidate-order-v2",
                seed,
                example.example_id,
            ),
            example.example_id,
        ),
    )
    limit = len(ordered) if candidate_limit is None else int(candidate_limit)
    if limit < requested_records or limit > len(ordered):
        raise ValueError("candidate_limit must cover records and available rows")
    ordered = ordered[:limit]
    identifiers = [example.example_id for example in ordered]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate examples are not unique")
    if len(nodes) < DEFAULT_TOP_K:
        raise ValueError("fixed candidate corpus has fewer than six nodes")
    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("candidate corpus node IDs are not unique")
    candidate_sources = set(identifiers)
    if any(node.source_example_id not in candidate_sources for node in nodes):
        raise ValueError("candidate corpus contains a node outside the pool")

    plans = _plan_candidates(ordered, seed=seed)
    data_rejections: dict[str, int] = {}
    for plan in plans:
        if plan.rejection_reason is not None:
            data_rejections[plan.rejection_reason] = (
                data_rejections.get(plan.rejection_reason, 0) + 1
            )

    records = []
    retrieval_rejections: dict[str, int] = {}
    attempted_candidates = 0
    for plan in plans:
        if plan.rejection_reason is not None:
            continue
        attempted_candidates += 1
        assert plan.occurrence is not None
        assert plan.donor is not None
        assert plan.retained is not None
        assert plan.retained_occurrence is not None
        try:
            hits = _canonical_hits(
                tuple(retriever(plan.example.question, DEFAULT_TOP_K)),
                top_k=DEFAULT_TOP_K,
            )
            if any(
                _contains_exact(
                    f"{hit.node.title}\n{hit.node.text}",
                    plan.donor.answer,
                )
                for hit in hits
            ):
                raise LookupError(
                    "donor_response_already_present_in_retrieved_context"
                )
            retrieved_documents = {
                hit.node.document_id
                for hit in hits
                if hit.node.source_example_id == plan.example.example_id
            }
            if not set(plan.example.supporting_document_ids).issubset(
                retrieved_documents
            ):
                raise LookupError("all_supporting_documents_not_retrieved")
            gold_hits = tuple(
                hit
                for hit in hits
                if hit.node.source_example_id == plan.example.example_id
                and hit.node.document_id
                in set(plan.example.supporting_document_ids)
            )
            occurrence_document_id = next(
                paragraph.document_id
                for paragraph in plan.example.paragraphs
                if paragraph.paragraph_index == plan.occurrence.paragraph_index
            )
            if not any(
                hit.node.document_id == occurrence_document_id
                and _contains_exact(hit.node.text, plan.example.answer)
                for hit in gold_hits
            ):
                raise LookupError("answer_bearing_gold_chunk_not_retrieved")
            counterfactual = build_counterfactual(
                plan.example,
                plan.donor,
                plan.occurrence,
            )
            retained_paragraph = next(
                paragraph
                for paragraph in plan.retained.paragraphs
                if paragraph.paragraph_index
                == plan.retained_occurrence.paragraph_index
            )
            gold_node_ids = tuple(hit.node.node_id for hit in gold_hits)
            context, tail_nodes = _select_tail_and_package(
                tokenizer,
                counterfactual,
                hits,
                gold_node_ids,
                plan.retained,
                retained_paragraph,
                nodes,
                seed=seed,
                minimum_tokens_after_owned=minimum_tokens_after_owned,
            )
            donor_occurrences = tuple(
                _answer_pattern(counterfactual.distractor_answer).finditer(
                    context.original_text
                )
            )
            if (
                len(donor_occurrences) != 1
                or _contains_exact(
                    context.edited_text,
                    counterfactual.distractor_answer,
                )
            ):
                raise LookupError(
                    "donor_response_not_unique_to_owned_counterfactual"
                )
            records.append(
                _record_payload(
                    plan,
                    hits,
                    gold_node_ids,
                    counterfactual,
                    retained_paragraph,
                    context,
                    tail_nodes,
                )
            )
        except LookupError as exc:
            reason = str(exc)
            retrieval_rejections[reason] = (
                retrieval_rejections.get(reason, 0) + 1
            )
            continue
        if len(records) == requested_records:
            break
    if len(records) != requested_records:
        raise RuntimeError(
            f"predeclared MuSiQue selection produced {len(records)} of "
            f"{requested_records} required records"
        )

    resolved_package_versions = _observed_package_versions()
    if package_versions is not None:
        resolved_package_versions.update(
            {str(key): str(value) for key, value in package_versions.items()}
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "contains_source_text": False,
        "exploratory_after_2wiki_v1": True,
        "provenance": {
            "exploratory_after_2wiki_v1": True,
            "does_not_replace_or_suppress_2wiki_v1": True,
            "confirmation_source": "untouched_pinned_musique_ans_dev",
            "selection_uses_2wiki_outputs": False,
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "filename": DATASET_FILENAME,
            "split": DATASET_SPLIT,
            "configuration": DATASET_CONFIGURATION,
            "license": DATASET_LICENSE,
            "raw_file_sha256": DATASET_FILE_SHA256,
        },
        "embedding": {
            "model_id": EMBEDDING_MODEL_ID,
            "revision": EMBEDDING_MODEL_REVISION,
        },
        "tokenizer": {
            "model_id": str(tokenizer_id),
            "revision": str(tokenizer_revision),
            "offset_mapping_required": True,
        },
        "package_versions": dict(sorted(resolved_package_versions.items())),
        "retrieval_config": {
            "framework": "llama-index-core",
            "framework_version": LLAMA_INDEX_CORE_VERSION,
            "embedding_integration": "llama-index-embeddings-huggingface",
            "embedding_integration_version": LLAMA_INDEX_HUGGINGFACE_VERSION,
            "splitter": "SentenceSplitter",
            "index": "VectorStoreIndex",
            "top_k": DEFAULT_TOP_K,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "index_builds": 1,
            "candidate_corpus_fixed": True,
            "canonical_order": "descending score then stable node ID",
        },
        "erasure_config": {
            "minimum_tokens_strictly_after_owned_block": int(
                minimum_tokens_after_owned
            ),
            "ownership": "exact fast-tokenizer offset mappings",
            "literal_token_deletion": True,
        },
        "selection_policy": {
            "mode": str(mode),
            "scientific_status": "exploratory",
            "seed": int(seed),
            "requested_records": int(requested_records),
            "candidate_limit": limit,
            "candidate_order": "sha256(seed, untouched source ID)",
            "fixed_before_model_scoring": True,
            "gemma_outputs_used": False,
            "attempted_data_eligible_candidates": attempted_candidates,
            "data_only_rejection_counts": dict(sorted(data_rejections.items())),
            "retrieval_rejection_counts": dict(
                sorted(retrieval_rejections.items())
            ),
            "candidate_pool": _candidate_pool_payload(plans),
        },
        "corpus_inventory": _corpus_inventory(nodes),
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


def _validate_unfrozen_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported MuSiQue v2 manifest schema")
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported MuSiQue v2 manifest schema version")
    if manifest.get("contains_source_text") is not False:
        raise ValueError("MuSiQue manifests must be source-text-free")
    if manifest.get("exploratory_after_2wiki_v1") is not True:
        raise ValueError("MuSiQue v2 must remain explicitly exploratory")
    provenance = manifest.get("provenance") or {}
    if (
        provenance.get("exploratory_after_2wiki_v1") is not True
        or provenance.get("does_not_replace_or_suppress_2wiki_v1") is not True
        or provenance.get("selection_uses_2wiki_outputs") is not False
    ):
        raise ValueError("MuSiQue v2 provenance constraints are missing")
    leaked_keys = _SOURCE_TEXT_KEYS.intersection(_walk_keys(manifest))
    if leaked_keys:
        raise ValueError(
            "source-text or vector keys are forbidden: "
            + ", ".join(sorted(leaked_keys))
        )
    dataset = manifest.get("dataset") or {}
    expected_dataset = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "filename": DATASET_FILENAME,
        "split": DATASET_SPLIT,
        "configuration": DATASET_CONFIGURATION,
        "license": DATASET_LICENSE,
        "raw_file_sha256": DATASET_FILE_SHA256,
    }
    if dataset != expected_dataset:
        raise ValueError("manifest does not pin the exact MuSiQue-Ans dev source")
    embedding = manifest.get("embedding") or {}
    if embedding != {
        "model_id": EMBEDDING_MODEL_ID,
        "revision": EMBEDDING_MODEL_REVISION,
    }:
        raise ValueError("manifest embedding model or revision is not pinned")
    retrieval = manifest.get("retrieval_config") or {}
    expected_retrieval = {
        "framework": "llama-index-core",
        "framework_version": LLAMA_INDEX_CORE_VERSION,
        "embedding_integration": "llama-index-embeddings-huggingface",
        "embedding_integration_version": LLAMA_INDEX_HUGGINGFACE_VERSION,
        "splitter": "SentenceSplitter",
        "index": "VectorStoreIndex",
        "top_k": DEFAULT_TOP_K,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "index_builds": 1,
        "candidate_corpus_fixed": True,
        "canonical_order": "descending score then stable node ID",
    }
    if retrieval != expected_retrieval:
        raise ValueError("manifest retrieval stack is not the fixed v2 stack")
    packages = manifest.get("package_versions")
    required_packages = {
        "python",
        "huggingface-hub",
        "transformers",
        "llama-index-core",
        "llama-index-embeddings-huggingface",
    }
    if (
        not isinstance(packages, Mapping)
        or not required_packages.issubset(packages)
    ):
        raise ValueError("manifest package versions are missing")
    erasure = manifest.get("erasure_config") or {}
    minimum = int(
        erasure.get("minimum_tokens_strictly_after_owned_block", -1)
    )
    if minimum < DEFAULT_MIN_TOKENS_AFTER_OWNED:
        raise ValueError("manifest owned block is not at least 512 tokens early")
    policy = manifest.get("selection_policy") or {}
    if (
        policy.get("fixed_before_model_scoring") is not True
        or policy.get("gemma_outputs_used") is not False
        or policy.get("scientific_status") != "exploratory"
    ):
        raise ValueError("manifest was not frozen independently of Gemma")
    candidate_pool = policy.get("candidate_pool")
    if not isinstance(candidate_pool, list) or not candidate_pool:
        raise ValueError("manifest candidate pool is empty")
    candidate_ids = [str(item.get("source_id")) for item in candidate_pool]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("manifest candidate pool IDs are not unique")
    if len(candidate_pool) != int(policy.get("candidate_limit", -1)):
        raise ValueError("manifest candidate pool length drifted")
    expected_data_rejections: dict[str, int] = {}
    for item in candidate_pool:
        reason = item.get("data_only_rejection_reason")
        eligible = item.get("data_only_eligible")
        if (reason is None) != (eligible is True):
            raise ValueError("candidate data-only eligibility is inconsistent")
        if reason is not None:
            expected_data_rejections[str(reason)] = (
                expected_data_rejections.get(str(reason), 0) + 1
            )
    if policy.get("data_only_rejection_counts") != dict(
        sorted(expected_data_rejections.items())
    ):
        raise ValueError("candidate data-only rejection counts drifted")
    corpus = manifest.get("corpus_inventory")
    if not isinstance(corpus, list) or len(corpus) < DEFAULT_TOP_K:
        raise ValueError("manifest corpus inventory is incomplete")
    corpus_node_ids = [str(item.get("node_id")) for item in corpus]
    if (
        corpus_node_ids != sorted(corpus_node_ids)
        or len(corpus_node_ids) != len(set(corpus_node_ids))
    ):
        raise ValueError("manifest corpus inventory is not canonical")
    if not {
        str(item.get("source_id")) for item in corpus
    }.issubset(set(candidate_ids)):
        raise ValueError("manifest corpus falls outside the candidate pool")
    records = manifest.get("records")
    if (
        not isinstance(records, list)
        or len(records) != int(policy.get("requested_records", -1))
        or not records
    ):
        raise ValueError("manifest record count differs from the fixed request")
    record_ids = [str(record.get("record_id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("manifest record IDs are not unique")
    retrieval_rejections = policy.get("retrieval_rejection_counts")
    if not isinstance(retrieval_rejections, Mapping) or any(
        int(value) < 0 for value in retrieval_rejections.values()
    ):
        raise ValueError("manifest retrieval rejection counts are invalid")
    if int(policy.get("attempted_data_eligible_candidates", -1)) != (
        len(records) + sum(int(value) for value in retrieval_rejections.values())
    ):
        raise ValueError("manifest attempted-candidate accounting drifted")
    for record in records:
        target = record.get("target_source") or {}
        donor = record.get("donor_source") or {}
        retained = record.get("retained_probe") or {}
        if target.get("source_id") not in candidate_ids:
            raise ValueError("record target is outside the candidate pool")
        if (
            donor.get("source_id") not in candidate_ids
            or donor.get("source_id") == target.get("source_id")
        ):
            raise ValueError("record donor is not a distinct candidate")
        if (
            retained.get("source_id") not in candidate_ids
            or retained.get("source_id") == target.get("source_id")
            or retained.get("source_id") == donor.get("source_id")
        ):
            raise ValueError("retained QA is not from an unrelated record")
        if target.get("coarse_response_type") != donor.get(
            "coarse_response_type"
        ):
            raise ValueError("record target and donor coarse types differ")
        if abs(
            int(target.get("whitespace_token_length", -100))
            - int(donor.get("whitespace_token_length", 100))
        ) > DONOR_MAX_TOKEN_LENGTH_DELTA:
            raise ValueError("record target and donor lengths are not similar")
        if (
            (record.get("counterfactual") or {}).get(
                "distractor_response_unique_to_owned_block"
            )
            is not True
        ):
            raise ValueError("record distractor is not unique to owned block")
        trace = record.get("retrieval")
        if not isinstance(trace, list) or len(trace) != DEFAULT_TOP_K:
            raise ValueError("record retrieval trace is not fixed top_k=6")
        ranks = [int(item.get("rank", -1)) for item in trace]
        if ranks != list(range(1, DEFAULT_TOP_K + 1)):
            raise ValueError("record retrieval ranks are not contiguous")
        if len({item.get("node_id") for item in trace}) != DEFAULT_TOP_K:
            raise ValueError("record retrieval trace has duplicate nodes")
        if any(
            not math.isfinite(float(item.get("score", math.nan)))
            for item in trace
        ):
            raise ValueError("record retrieval score is not finite")
        ordering = [
            (-float(item["score"]), str(item["node_id"])) for item in trace
        ]
        if ordering != sorted(ordering):
            raise ValueError("record retrieval trace is not canonical")
        gold = record.get("gold_evidence") or {}
        gold_ids = list(gold.get("retrieved_node_ids") or ())
        if (
            not gold_ids
            or gold.get("all_supporting_documents_retrieved") is not True
            or gold.get("answer_bearing_gold_chunk_retrieved") is not True
        ):
            raise ValueError("record does not freeze strong retrieved gold")
        rank_by_node = {
            str(item["node_id"]): int(item["rank"]) for item in trace
        }
        if not set(gold_ids).issubset(rank_by_node):
            raise ValueError("record gold nodes are outside retrieval")
        context = record.get("context") or {}
        ownership = context.get("ownership") or {}
        positions = ownership.get("forget_positions")
        if (
            not isinstance(positions, list)
            or not positions
            or positions != sorted(set(int(value) for value in positions))
        ):
            raise ValueError("record ownership positions are not canonical")
        original_count = int(context.get("original_token_count", -1))
        edited_count = int(context.get("edited_token_count", -1))
        if positions[-1] >= original_count:
            raise ValueError("record ownership falls outside tokenized context")
        if original_count - len(positions) != edited_count:
            raise ValueError("record literal token edit count is inconsistent")
        after = int(
            context.get("tokens_strictly_after_owned_block", -1)
        )
        if after != original_count - positions[-1] - 1 or after < minimum:
            raise ValueError("record owned-block suffix geometry drifted")
        layout = context.get("layout") or {}
        expected_rank = max(rank_by_node[node_id] for node_id in gold_ids)
        if int(layout.get("insertion_after_retrieval_rank", -1)) != expected_rank:
            raise ValueError("counterfactual is not placed after retrieved gold")
        tail_ids = layout.get("tail_node_ids")
        if not isinstance(tail_ids, list) or not set(tail_ids).issubset(
            set(corpus_node_ids)
        ):
            raise ValueError("context tail is outside the frozen corpus")
        retained_tokens = (context.get("retained") or {}).get(
            "token_positions"
        )
        if not isinstance(retained_tokens, list) or not retained_tokens:
            raise ValueError("retained QA support has no frozen tokens")
        if set(retained_tokens).intersection(positions):
            raise ValueError("retained and owned model tokens overlap")
        retained_descriptor = context.get("retained") or {}
        if (
            retained_descriptor.get("tokens_preserved_exactly") is not True
            or retained_descriptor.get("original_token_ids_sha256")
            != retained_descriptor.get("edited_token_ids_sha256")
        ):
            raise ValueError("retained model tokens were not literally preserved")


def freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-freeze a v2 manifest with record and root SHA-256 integrity."""

    frozen = copy.deepcopy(dict(manifest))
    frozen.pop("integrity", None)
    for record in frozen.get("records", []):
        record.pop("record_integrity", None)
    _validate_unfrozen_manifest(frozen)
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
    """Reject integrity, schema, provenance, trace, or geometry drift."""

    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("manifest records are missing")
    for record in records:
        integrity = record.get("record_integrity")
        if not isinstance(integrity, Mapping):
            raise ValueError("manifest record integrity is missing")
        payload = copy.deepcopy(dict(record))
        payload.pop("record_integrity", None)
        if (
            integrity.get("algorithm") != "sha256"
            or integrity.get("sha256") != _payload_sha256(payload)
        ):
            raise ValueError(
                f"manifest record integrity mismatch: {record.get('record_id')}"
            )
    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("manifest root integrity is missing")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
    ):
        raise ValueError("manifest root integrity mismatch")
    _validate_unfrozen_manifest(manifest)


def write_manifest_atomic(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> None:
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


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    examples: Sequence[MusiqueExample],
    nodes: Sequence[CorpusNode],
    tokenizer: Any,
) -> tuple[RehydratedMusiqueRecord, ...]:
    """Rebuild all source text from rows/nodes and consume frozen traces only."""

    validate_manifest(manifest)
    policy = manifest["selection_policy"]
    seed = int(policy["seed"])
    examples_by_id = {example.example_id: example for example in examples}
    candidate_ids = [
        str(item["source_id"]) for item in policy["candidate_pool"]
    ]
    if set(examples_by_id) != set(candidate_ids):
        raise ValueError("rehydration rows differ from the frozen candidate pool")
    ordered = tuple(examples_by_id[source_id] for source_id in candidate_ids)
    plans = _plan_candidates(ordered, seed=seed)
    if _candidate_pool_payload(plans) != policy["candidate_pool"]:
        raise ValueError("rehydrated candidate source data drifted")
    if _corpus_inventory(nodes) != manifest["corpus_inventory"]:
        raise ValueError("rehydrated candidate corpus nodes drifted")
    nodes_by_id = {node.node_id: node for node in nodes}
    plans_by_id = {plan.example.example_id: plan for plan in plans}

    hydrated = []
    for stored in manifest["records"]:
        source_id = str(stored["target_source"]["source_id"])
        plan = plans_by_id[source_id]
        if (
            plan.rejection_reason is not None
            or plan.occurrence is None
            or plan.donor is None
            or plan.retained is None
            or plan.retained_occurrence is None
        ):
            raise ValueError("frozen record is no longer data-only eligible")
        if plan.donor.example_id != stored["donor_source"]["source_id"]:
            raise ValueError("rehydrated deterministic donor differs")
        if plan.retained.example_id != stored["retained_probe"]["source_id"]:
            raise ValueError("rehydrated retained QA source differs")
        hits = []
        for item in stored["retrieval"]:
            node_id = str(item["node_id"])
            if node_id not in nodes_by_id:
                raise ValueError(f"missing frozen source node {node_id}")
            hits.append(
                RetrievalHit(
                    node=nodes_by_id[node_id],
                    score=float(item["score"]),
                    rank=int(item["rank"]),
                )
            )
        gold_node_ids = tuple(
            str(value)
            for value in stored["gold_evidence"]["retrieved_node_ids"]
        )
        counterfactual = build_counterfactual(
            plan.example,
            plan.donor,
            plan.occurrence,
        )
        retained_paragraph = next(
            paragraph
            for paragraph in plan.retained.paragraphs
            if paragraph.paragraph_index
            == plan.retained_occurrence.paragraph_index
        )
        tail_nodes = tuple(
            nodes_by_id[str(node_id)]
            for node_id in stored["context"]["layout"]["tail_node_ids"]
        )
        context = package_context(
            tokenizer,
            counterfactual,
            hits,
            gold_node_ids,
            plan.retained,
            retained_paragraph,
            tail_nodes=tail_nodes,
        )
        expected = _record_payload(
            plan,
            hits,
            gold_node_ids,
            counterfactual,
            retained_paragraph,
            context,
            tail_nodes,
        )
        observed = copy.deepcopy(dict(stored))
        observed.pop("record_integrity", None)
        if observed != expected:
            raise ValueError("rehydrated record differs from frozen trace")
        hydrated.append(
            RehydratedMusiqueRecord(
                record_id=str(stored["record_id"]),
                example=plan.example,
                retrieval_hits=tuple(hits),
                counterfactual=counterfactual,
                context=context,
                retained_example=plan.retained,
                retained_paragraph=retained_paragraph,
                retained_question=plan.retained.question,
                retained_answer=plan.retained.answer,
            )
        )
    return tuple(hydrated)


def rehydrate_manifest_from_rows(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    nodes: Sequence[CorpusNode] | None = None,
) -> tuple[RehydratedMusiqueRecord, ...]:
    """Rehydrate pinned rows and deterministic nodes without rerunning retrieval."""

    validate_manifest(manifest)
    required_ids = [
        item["source_id"]
        for item in manifest["selection_policy"]["candidate_pool"]
    ]
    examples = extract_musique_examples(
        rows,
        seed=int(manifest["selection_policy"]["seed"]),
        required_ids=required_ids,
    )
    if nodes is None:
        documents = build_document_specs(examples)
        nodes, _ = split_documents_with_llamaindex(
            documents,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )
    return rehydrate_manifest(manifest, examples, nodes, tokenizer)


def load_pinned_musique_rows(
    local_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load and SHA-verify the exact raw dev JSONL at explicit runtime only."""

    if local_path is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to resolve pinned MuSiQue data"
            ) from exc
        resolved = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=DATASET_FILENAME,
                revision=DATASET_REVISION,
            )
        )
    else:
        resolved = Path(local_path)
    raw = resolved.read_bytes()
    observed_hash = hashlib.sha256(raw).hexdigest()
    if observed_hash != DATASET_FILE_SHA256:
        raise ValueError(
            "MuSiQue dev JSONL SHA-256 differs from the pinned raw source"
        )
    rows = []
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid MuSiQue JSONL at line {line_number}"
            ) from exc
        if not isinstance(item, dict):
            raise ValueError(
                f"MuSiQue JSONL line {line_number} is not an object"
            )
        rows.append(item)
    if not rows:
        raise ValueError("pinned MuSiQue dev JSONL is empty")
    return tuple(rows)


def _mode_defaults(mode: str) -> tuple[int, int]:
    if mode == "smoke":
        return SMOKE_RECORDS, SMOKE_CANDIDATES
    if mode == "full":
        return FULL_RECORDS, FULL_CANDIDATES
    raise ValueError(f"unknown manifest mode {mode!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--records", type=int)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-path",
        help=(
            "local exact musique_ans_v1.0_dev.jsonl; if omitted, resolve the "
            "pinned revision with huggingface_hub"
        ),
    )
    parser.add_argument("--embedding-device")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument(
        "--tokenizer-revision",
        default=DEFAULT_TOKENIZER_REVISION,
    )
    parser.add_argument("--out")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build an exploratory manifest; live dependencies remain CLI-only."""

    parser = _parser()
    args = parser.parse_args(argv)
    default_records, default_candidates = _mode_defaults(args.mode)
    requested_records = args.records or default_records
    candidate_limit = args.candidate_limit or default_candidates
    if requested_records < 1 or candidate_limit < requested_records:
        parser.error("--candidate-limit must be at least --records")
    output = Path(
        args.out
        or f"outputs/gemma_sv_rag/musique_{args.mode}_exploratory_v2_manifest.json"
    )
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")

    rows = load_pinned_musique_rows(args.data_path)
    examples = extract_musique_examples(
        rows,
        seed=args.seed,
        limit=candidate_limit,
    )
    documents = build_document_specs(examples)
    embedding = load_pinned_embedding(device=args.embedding_device)
    nodes, retriever = build_production_retriever(
        documents,
        embed_model=embedding,
    )
    tokenizer = load_offset_tokenizer(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    manifest = build_manifest(
        examples,
        nodes,
        retriever,
        tokenizer,
        requested_records=requested_records,
        candidate_limit=candidate_limit,
        seed=args.seed,
        mode=args.mode,
        tokenizer_id=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
    )
    write_manifest_atomic(output, manifest)
    write_environment_lock(output.parent / "environment-lock.txt")
    print(
        f"wrote {len(manifest['records'])} exploratory MuSiQue v2 records "
        f"to {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic, text-free manifests for a small 2Wiki RAG benchmark.

The import path is deliberately download-free.  ``datasets``, Transformers,
LlamaIndex, and the Hugging Face embedding integration are imported only by
runtime functions used by the CLI.  Pure helpers accept tokenizers, embedders,
nodes, and retrieval callables so contract tests need none of those packages.

The saved manifest contains source identifiers, hashes, retrieval scores, and
token ownership, but no questions, answers, titles, or passage text.  Evaluation
therefore rehydrates the pinned dataset and validates it against the frozen
hashes before loading any record into a model.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import (
    PackageNotFoundError,
    distributions,
    version as package_version,
)
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


MANIFEST_SCHEMA = "gemma-sv-rag-manifest-v1"
MANIFEST_SCHEMA_VERSION = 1
PACKAGE_SCHEMA = "gemma-sv-rag-context-v1"
COUNTERFACTUAL_RECIPE_VERSION = "wikipedia-counterfactual-v1"
RETAINED_PROBE_VERSION = "retained-title-v1"

DATASET_ID = "framolfese/2WikiMultihopQA"
DATASET_REVISION = "a5d42f3b40d57a8c59fa10b2ac0c1829e4f73aba"
DATASET_SPLIT = "validation"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
LLAMA_INDEX_CORE_VERSION = "0.14.23"
LLAMA_INDEX_HUGGINGFACE_VERSION = "0.7.0"
DEFAULT_TOKENIZER_ID = "google/gemma-3-1b-pt"
DEFAULT_TOKENIZER_REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"

DEFAULT_TOP_K = 8
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32
DEFAULT_MIN_TOKENS_AFTER_OWNED = 512
SMOKE_RECORDS = 8
SMOKE_CANDIDATES = 32
FULL_RECORDS = 64
FULL_CANDIDATES = 256

_SOURCE_TEXT_KEYS = frozenset(
    {"question", "answer", "title", "text", "sentences", "source_text"}
)
_RETAINED_QUESTION = "What is the title of the retained reference passage?"
_COUNTERFACTUAL_ANSWER_POOL = (
    "Alderwick",
    "Bellweather Institute",
    "Caspian Registry",
    "Dunmere",
    "Elian Voss",
    "Fernbridge",
    "Grayhaven",
    "Hester Vale",
    "Ilex Archive",
    "Juniper Bay",
    "Kestrel House",
    "Larkfield",
)


def text_sha256(value: str) -> str:
    """Return the SHA-256 digest of exact UTF-8 text."""

    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _require_package_version(distribution: str, expected: str) -> None:
    try:
        observed = package_version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"{distribution}=={expected} is required; install "
            "gemma_sv/requirements-rag.txt"
        ) from exc
    if observed != expected:
        raise RuntimeError(
            f"{distribution}=={expected} is required, found {observed}"
        )


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash token IDs with the same signed-int64 framing as GemmaRuntime."""

    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<q", int(token_id)))
    return digest.hexdigest()


def stable_identifier(namespace: str, *parts: object) -> str:
    """Build a readable deterministic identifier from length-framed values."""

    digest = hashlib.sha256()
    digest.update(str(namespace).encode("utf-8"))
    digest.update(b"\0")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return f"{namespace}-{digest.hexdigest()[:24]}"


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


def _selection_key(seed: int, example_id: str) -> bytes:
    return hashlib.sha256(
        f"2wiki-selection-v1\0{int(seed)}\0{example_id}".encode("utf-8")
    ).digest()


@dataclass(frozen=True)
class WikiPassage:
    """One source paragraph from a 2Wiki example."""

    example_id: str
    paragraph_index: int
    title: str
    sentences: tuple[str, ...]
    supporting_sentence_indices: tuple[int, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(sentence.strip() for sentence in self.sentences).strip()

    @property
    def document_id(self) -> str:
        return stable_identifier(
            "doc",
            DATASET_ID,
            DATASET_REVISION,
            self.example_id,
            self.paragraph_index,
            self.title,
        )

    @property
    def supporting_sentences(self) -> tuple[str, ...]:
        return tuple(
            self.sentences[index].strip()
            for index in self.supporting_sentence_indices
            if 0 <= index < len(self.sentences)
            and self.sentences[index].strip()
        )


@dataclass(frozen=True)
class WikiExample:
    """Normalized 2Wiki source record."""

    example_id: str
    question: str
    answer: str
    question_type: str
    passages: tuple[WikiPassage, ...]

    @property
    def supporting_document_ids(self) -> tuple[str, ...]:
        return tuple(
            passage.document_id
            for passage in self.passages
            if passage.supporting_sentence_indices
        )


@dataclass(frozen=True)
class DocumentSpec:
    """Dependency-free equivalent of a LlamaIndex ``Document``."""

    document_id: str
    source_example_id: str
    paragraph_index: int
    title: str
    text: str
    supporting_sentences: tuple[str, ...]


@dataclass(frozen=True)
class CorpusNode:
    """Stable chunk used by both injected and LlamaIndex retrieval paths."""

    node_id: str
    document_id: str
    source_example_id: str
    paragraph_index: int
    chunk_index: int
    title: str
    text: str
    supporting_sentence_hashes: tuple[str, ...] = ()

    @property
    def is_gold_evidence(self) -> bool:
        return bool(self.supporting_sentence_hashes)


@dataclass(frozen=True)
class RetrievalHit:
    """One canonical retrieval result."""

    node: CorpusNode
    score: float
    rank: int = 0


@dataclass(frozen=True)
class CounterfactualPassage:
    """Generated distractor and its explicitly owned claim span."""

    title: str
    text: str
    distractor_answer: str
    answer_strategy: str
    owned_start: int
    owned_end: int
    answer_start: int
    answer_end: int


@dataclass(frozen=True)
class TokenizedContext:
    """Packaged context plus exact model-token ownership."""

    original_text: str
    edited_text: str
    original_token_ids: tuple[int, ...]
    edited_token_ids: tuple[int, ...]
    offset_mapping: tuple[tuple[int, int], ...]
    forget_positions: tuple[int, ...]
    deletion_ranges: tuple[tuple[int, int], ...]
    retained_positions: tuple[int, ...]
    edited_retained_positions: tuple[int, ...]
    owned_character_span: tuple[int, int]
    answer_character_span: tuple[int, int]
    retained_character_span: tuple[int, int]


@dataclass(frozen=True)
class RehydratedRecord:
    """Runtime-only record containing validated source text."""

    record_id: str
    example: WikiExample
    retrieval_hits: tuple[RetrievalHit, ...]
    retained_node: CorpusNode
    counterfactual: CounterfactualPassage
    context: TokenizedContext
    retained_question: str
    retained_answer: str


class Retriever(Protocol):
    def __call__(
        self,
        question: str,
        top_k: int,
    ) -> Sequence[RetrievalHit]:
        ...


def _string_sequence(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return tuple(str(item) for item in value)


def _supporting_fact_map(value: Any) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    if isinstance(value, Mapping):
        titles = _string_sequence(value.get("title", ()), field="supporting title")
        indices = value.get("sent_id", ())
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
            raise ValueError("supporting sent_id must be a sequence")
        if len(titles) != len(indices):
            raise ValueError("supporting fact titles and indices differ in length")
        pairs = zip(titles, indices)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pairs = (
            (item[0], item[1])
            for item in value
            if isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) >= 2
        )
    else:
        raise ValueError("supporting_facts has an unsupported shape")
    for title, raw_index in pairs:
        result.setdefault(str(title), set()).add(int(raw_index))
    return result


def _context_pairs(value: Any) -> list[tuple[str, tuple[str, ...]]]:
    if isinstance(value, Mapping):
        titles = _string_sequence(value.get("title", ()), field="context title")
        raw_sentences = value.get("sentences", ())
        if (
            isinstance(raw_sentences, (str, bytes))
            or not isinstance(raw_sentences, Sequence)
            or len(titles) != len(raw_sentences)
        ):
            raise ValueError("context titles and sentence groups differ in length")
        return [
            (
                title,
                _string_sequence(sentences, field="context sentences"),
            )
            for title, sentences in zip(titles, raw_sentences)
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes))
                or len(item) < 2
            ):
                raise ValueError("context entry has an unsupported shape")
            result.append(
                (
                    str(item[0]),
                    _string_sequence(item[1], field="context sentences"),
                )
            )
        return result
    raise ValueError("context has an unsupported shape")


def parse_2wiki_row(row: Mapping[str, Any]) -> WikiExample:
    """Normalize one HotpotQA-style 2Wiki row without external packages."""

    example_id = str(row.get("id") or row.get("_id") or "").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if not example_id or not question or not answer:
        raise ValueError("2Wiki row is missing id, question, or answer")
    support = _supporting_fact_map(row.get("supporting_facts", ()))
    passages = []
    for paragraph_index, (title, sentences) in enumerate(
        _context_pairs(row.get("context", ()))
    ):
        valid_support = tuple(
            sorted(
                index
                for index in support.get(title, set())
                if 0 <= index < len(sentences)
            )
        )
        passages.append(
            WikiPassage(
                example_id=example_id,
                paragraph_index=paragraph_index,
                title=title,
                sentences=sentences,
                supporting_sentence_indices=valid_support,
            )
        )
    if not passages:
        raise ValueError(f"2Wiki row {example_id!r} has no context passages")
    if not any(passage.supporting_sentence_indices for passage in passages):
        raise ValueError(f"2Wiki row {example_id!r} has no valid supporting facts")
    return WikiExample(
        example_id=example_id,
        question=question,
        answer=answer,
        question_type=str(row.get("type") or "unknown"),
        passages=tuple(passages),
    )


def extract_2wiki_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    limit: int | None = None,
    required_ids: Iterable[str] | None = None,
) -> tuple[WikiExample, ...]:
    """Parse and hash-order rows using only predeclared source properties."""

    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be positive")
    required = None if required_ids is None else {str(value) for value in required_ids}
    parsed = []
    for row in rows:
        raw_id = str(row.get("id") or row.get("_id") or "")
        if required is not None and raw_id not in required:
            continue
        parsed.append(parse_2wiki_row(row))
    identifiers = [example.example_id for example in parsed]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("2Wiki source contains duplicate example IDs")
    if required is not None:
        missing = required.difference(identifiers)
        if missing:
            raise ValueError(f"2Wiki source is missing {len(missing)} manifest IDs")
    ordered = sorted(
        parsed,
        key=lambda example: (
            _selection_key(seed, example.example_id),
            example.example_id,
        ),
    )
    if limit is not None:
        ordered = ordered[: int(limit)]
    return tuple(ordered)


def build_document_specs(
    examples: Sequence[WikiExample],
) -> tuple[DocumentSpec, ...]:
    """Convert normalized examples to stable source documents."""

    documents = []
    for example in examples:
        for passage in example.passages:
            if not passage.text:
                continue
            documents.append(
                DocumentSpec(
                    document_id=passage.document_id,
                    source_example_id=example.example_id,
                    paragraph_index=passage.paragraph_index,
                    title=passage.title,
                    text=passage.text,
                    supporting_sentences=passage.supporting_sentences,
                )
            )
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("stable document IDs are not unique")
    return tuple(documents)


def _node_from_chunk(
    document: DocumentSpec,
    chunk_text: str,
    chunk_index: int,
) -> CorpusNode:
    exact_text = str(chunk_text)
    supporting_hashes = tuple(
        text_sha256(sentence)
        for sentence in document.supporting_sentences
        if sentence and sentence in exact_text
    )
    return CorpusNode(
        node_id=stable_identifier(
            "node",
            document.document_id,
            int(chunk_index),
            text_sha256(exact_text),
        ),
        document_id=document.document_id,
        source_example_id=document.source_example_id,
        paragraph_index=document.paragraph_index,
        chunk_index=int(chunk_index),
        title=document.title,
        text=exact_text,
        supporting_sentence_hashes=supporting_hashes,
    )


def one_node_per_document(
    documents: Sequence[DocumentSpec],
) -> tuple[CorpusNode, ...]:
    """Dependency-free deterministic node builder for tests and injections."""

    return tuple(
        _node_from_chunk(document, document.text, 0)
        for document in documents
    )


def split_documents_with_llamaindex(
    documents: Sequence[DocumentSpec],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[tuple[CorpusNode, ...], tuple[Any, ...]]:
    """Create pinned LlamaIndex Documents and SentenceSplitter nodes.

    This function is the optional-dependency boundary.  It returns pure nodes
    beside the native nodes so retrieval can be frozen without serializing text.
    """

    if chunk_size < 1 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("invalid SentenceSplitter chunk configuration")
    _require_package_version("llama-index-core", LLAMA_INDEX_CORE_VERSION)
    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import MetadataMode
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex is required for live RAG chunking; install "
            "gemma_sv/requirements-rag.txt"
        ) from exc

    splitter = SentenceSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        include_metadata=False,
        include_prev_next_rel=False,
    )
    pure_nodes: list[CorpusNode] = []
    native_nodes: list[Any] = []
    for document in documents:
        metadata = {
            "document_id": document.document_id,
            "source_example_id": document.source_example_id,
            "paragraph_index": document.paragraph_index,
            "title": document.title,
        }
        native_document = Document(
            text=document.text,
            id_=document.document_id,
            metadata=metadata,
            excluded_embed_metadata_keys=list(metadata),
            excluded_llm_metadata_keys=list(metadata),
        )
        chunks = splitter.get_nodes_from_documents(
            [native_document],
            show_progress=False,
        )
        for chunk_index, native_node in enumerate(chunks):
            chunk_text = str(
                native_node.get_content(metadata_mode=MetadataMode.NONE)
            )
            pure = _node_from_chunk(document, chunk_text, chunk_index)
            native_node.id_ = pure.node_id
            native_node.metadata = dict(metadata)
            native_node.excluded_embed_metadata_keys = list(metadata)
            native_node.excluded_llm_metadata_keys = list(metadata)
            pure_nodes.append(pure)
            native_nodes.append(native_node)
    node_ids = [node.node_id for node in pure_nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("stable SentenceSplitter node IDs are not unique")
    return tuple(pure_nodes), tuple(native_nodes)


def load_pinned_embedding(*, device: str | None = None):
    """Load the pinned local Hugging Face embedding at runtime."""

    _require_package_version(
        "llama-index-embeddings-huggingface",
        LLAMA_INDEX_HUGGINGFACE_VERSION,
    )
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as exc:
        raise RuntimeError(
            "the Hugging Face LlamaIndex embedding integration is required; "
            "install gemma_sv/requirements-rag.txt"
        ) from exc
    kwargs: dict[str, Any] = {
        "model_name": EMBEDDING_MODEL_ID,
        "revision": EMBEDDING_MODEL_REVISION,
        "trust_remote_code": False,
    }
    if device:
        kwargs["device"] = str(device)
    return HuggingFaceEmbedding(**kwargs)


def build_llamaindex_retriever(
    documents: Sequence[DocumentSpec],
    *,
    top_k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_model: Any | None = None,
) -> tuple[tuple[CorpusNode, ...], Retriever]:
    """Build ``VectorStoreIndex`` over stable ``SentenceSplitter`` nodes."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    try:
        from llama_index.core import VectorStoreIndex
    except ImportError as exc:
        raise RuntimeError(
            "LlamaIndex is required for live retrieval; install "
            "gemma_sv/requirements-rag.txt"
        ) from exc
    pure_nodes, native_nodes = split_documents_with_llamaindex(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if len(native_nodes) < top_k:
        raise ValueError("corpus has fewer nodes than fixed top_k")
    embedding = embed_model if embed_model is not None else load_pinned_embedding()
    index = VectorStoreIndex(
        list(native_nodes),
        embed_model=embedding,
        show_progress=False,
    )
    native_retriever = index.as_retriever(similarity_top_k=int(top_k))
    pure_by_id = {node.node_id: node for node in pure_nodes}

    def retrieve(question: str, requested_top_k: int) -> Sequence[RetrievalHit]:
        if int(requested_top_k) != int(top_k):
            raise ValueError("retrieval requested a non-frozen top_k")
        raw_hits = native_retriever.retrieve(str(question))
        hits = []
        for raw in raw_hits:
            node_id = str(raw.node.node_id)
            if node_id not in pure_by_id:
                raise ValueError("LlamaIndex returned an unknown node")
            hits.append(
                RetrievalHit(
                    node=pure_by_id[node_id],
                    score=float(raw.score),
                )
            )
        return _canonical_hits(hits, top_k=top_k)

    return pure_nodes, retrieve


def _call_embedding(embedder: Any, text: str, *, query: bool) -> tuple[float, ...]:
    method_name = "get_query_embedding" if query else "get_text_embedding"
    method = getattr(embedder, method_name, None)
    if callable(method):
        values = method(str(text))
    elif callable(embedder):
        try:
            values = embedder(str(text), is_query=query)
        except TypeError:
            values = embedder(str(text))
    else:
        raise TypeError("embedder must be callable or expose embedding methods")
    vector = tuple(float(value) for value in values)
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("embedding must be a finite non-empty vector")
    return vector


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def make_in_memory_retriever(
    nodes: Sequence[CorpusNode],
    embedder: Any,
    *,
    top_k: int,
) -> Retriever:
    """Create a tiny deterministic vector retriever for injected embeddings."""

    if top_k < 1 or len(nodes) < top_k:
        raise ValueError("fixed top_k exceeds the injected node corpus")
    node_vectors = {
        node.node_id: _call_embedding(embedder, node.text, query=False)
        for node in nodes
    }

    def retrieve(question: str, requested_top_k: int) -> Sequence[RetrievalHit]:
        if int(requested_top_k) != int(top_k):
            raise ValueError("retrieval requested a non-frozen top_k")
        query_vector = _call_embedding(embedder, question, query=True)
        hits = [
            RetrievalHit(
                node=node,
                score=_cosine_similarity(query_vector, node_vectors[node.node_id]),
            )
            for node in nodes
        ]
        return _canonical_hits(hits, top_k=top_k)

    return retrieve


def _canonical_hits(
    hits: Sequence[RetrievalHit],
    *,
    top_k: int,
) -> tuple[RetrievalHit, ...]:
    if len(hits) < top_k:
        raise ValueError("retriever returned fewer results than fixed top_k")
    node_ids = [hit.node.node_id for hit in hits]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("retriever returned duplicate node IDs")
    for hit in hits:
        if not math.isfinite(float(hit.score)):
            raise ValueError("retrieval score is not finite")
    ordered = sorted(
        hits,
        key=lambda hit: (-float(hit.score), hit.node.node_id),
    )[:top_k]
    return tuple(
        RetrievalHit(
            node=hit.node,
            score=float(f"{float(hit.score):.12g}"),
            rank=rank,
        )
        for rank, hit in enumerate(ordered, start=1)
    )


def build_counterfactual(
    example: WikiExample,
    retained_node: CorpusNode,
) -> CounterfactualPassage:
    """Generate the fixed Wikipedia-style distractor with an owned claim."""

    normalized_answer = example.answer.strip().casefold()
    if normalized_answer == "yes":
        distractor_answer = "no"
        strategy = "binary_complement"
    elif normalized_answer == "no":
        distractor_answer = "yes"
        strategy = "binary_complement"
    else:
        visible_text = "\n".join(
            (
                example.question,
                example.answer,
                *(passage.title for passage in example.passages),
                *(passage.text for passage in example.passages),
                retained_node.title,
                retained_node.text,
            )
        ).casefold()
        offset = int(
            hashlib.sha256(example.example_id.encode("utf-8")).hexdigest(),
            16,
        ) % len(_COUNTERFACTUAL_ANSWER_POOL)
        candidates = (
            _COUNTERFACTUAL_ANSWER_POOL[offset:]
            + _COUNTERFACTUAL_ANSWER_POOL[:offset]
        )
        distractor_answer = next(
            (
                candidate
                for candidate in candidates
                if candidate.casefold() not in visible_text
                and candidate.casefold() != normalized_answer
            ),
            f"Alderwick-{text_sha256(example.example_id)[:8]}",
        )
        strategy = "predeclared_synthetic_entity_pool"

    supporting_titles = [
        passage.title
        for passage in example.passages
        if passage.supporting_sentence_indices
    ]
    base_title = supporting_titles[0] if supporting_titles else "Reference overview"
    title = f"{base_title}: reference overview"
    prefix = (
        f"{base_title} is covered by several reference summaries. "
        f"This entry addresses the question: {example.question}"
    )
    owned = (
        f"\nThe standard encyclopedia index identifies {distractor_answer} "
        "as the answer to that question.\n"
    )
    suffix = (
        "The edition was subsequently cited in general reference catalogues "
        "and historical indexes."
    )
    text = prefix + owned + suffix
    owned_start = len(prefix)
    owned_end = owned_start + len(owned)
    answer_start = owned_start + owned.index(distractor_answer)
    answer_end = answer_start + len(distractor_answer)
    return CounterfactualPassage(
        title=title,
        text=text,
        distractor_answer=distractor_answer,
        answer_strategy=strategy,
        owned_start=owned_start,
        owned_end=owned_end,
        answer_start=answer_start,
        answer_end=answer_end,
    )


def _tokenizer_payload(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
    except TypeError as exc:
        raise TypeError(
            "tokenizer must support return_offsets_mapping (use a fast tokenizer)"
        ) from exc
    if isinstance(encoded, Mapping):
        raw_ids = encoded.get("input_ids")
        raw_offsets = encoded.get("offset_mapping")
    else:
        raw_ids = getattr(encoded, "input_ids", None)
        raw_offsets = getattr(encoded, "offset_mapping", None)
    if raw_ids is None or raw_offsets is None:
        raise ValueError("tokenizer did not return input_ids and offset_mapping")
    if raw_ids and isinstance(raw_ids[0], Sequence) and not isinstance(
        raw_ids[0], (str, bytes)
    ):
        if len(raw_ids) != 1 or len(raw_offsets) != 1:
            raise ValueError("batched tokenization is unsupported")
        raw_ids = raw_ids[0]
        raw_offsets = raw_offsets[0]
    ids = [int(value) for value in raw_ids]
    offsets = [(int(item[0]), int(item[1])) for item in raw_offsets]
    if len(ids) != len(offsets):
        raise ValueError("token IDs and offset mapping differ in length")
    return ids, offsets


def _overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    start, end = offset
    span_start, span_end = span
    return end > start and start < span_end and end > span_start


def _contiguous_ranges(positions: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not positions:
        raise ValueError("owned span has no model tokens")
    normalized = tuple(sorted({int(position) for position in positions}))
    ranges = []
    start = previous = normalized[0]
    for position in normalized[1:]:
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
    deleted: Sequence[int],
) -> tuple[int, ...]:
    deleted_values = tuple(sorted(int(position) for position in deleted))
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


def package_context(
    tokenizer: Any,
    counterfactual: CounterfactualPassage,
    retrieval_hits: Sequence[RetrievalHit],
    retained_node_id: str,
) -> TokenizedContext:
    """Package identical ordered context and map exact owned model tokens."""

    if not retrieval_hits:
        raise ValueError("at least one retrieval hit is required")
    pieces = ["Retrieved reference documents:\n\n"]
    counter_header = f"[0] {counterfactual.title}\n"
    pieces.append(counter_header)
    counter_text_start = sum(len(piece) for piece in pieces)
    pieces.append(counterfactual.text)
    pieces.append("\n\n")
    owned_span = (
        counter_text_start + counterfactual.owned_start,
        counter_text_start + counterfactual.owned_end,
    )
    answer_span = (
        counter_text_start + counterfactual.answer_start,
        counter_text_start + counterfactual.answer_end,
    )
    retained_span = None
    for hit in retrieval_hits:
        header = f"[{hit.rank}] {hit.node.title}\n"
        pieces.append(header)
        body_start = sum(len(piece) for piece in pieces)
        pieces.append(hit.node.text)
        body_end = sum(len(piece) for piece in pieces)
        pieces.append("\n\n")
        if hit.node.node_id == retained_node_id:
            retained_span = (body_start - len(header), body_end)
    if retained_span is None:
        raise ValueError("retained node is not present in packaged retrieval")

    original_text = "".join(pieces).rstrip() + "\n"
    ids, offsets = _tokenizer_payload(tokenizer, original_text)
    forget = tuple(
        index
        for index, offset in enumerate(offsets)
        if _overlaps(offset, owned_span)
    )
    if not forget:
        raise ValueError("owned character span maps to no model tokens")
    retained = tuple(
        index
        for index, offset in enumerate(offsets)
        if _overlaps(offset, retained_span) and index not in set(forget)
    )
    if not retained:
        raise ValueError("retained passage maps to no model tokens")
    covered = [
        False
        for _ in range(owned_span[1] - owned_span[0])
    ]
    for position in forget:
        start, end = offsets[position]
        for character in range(
            max(start, owned_span[0]),
            min(end, owned_span[1]),
        ):
            covered[character - owned_span[0]] = True
    owned_source = original_text[owned_span[0] : owned_span[1]]
    if any(
        not is_covered and not character.isspace()
        for character, is_covered in zip(owned_source, covered)
    ):
        raise ValueError("offset mapping does not cover the complete owned span")

    edited_text = (
        original_text[: owned_span[0]] + original_text[owned_span[1] :]
    )
    edited_ids = _remove_positions(ids, forget)
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
        retained_character_span=retained_span,
    )


def _context_descriptor(context: TokenizedContext) -> dict[str, Any]:
    owned_offsets = [
        {
            "position": position,
            "start": context.offset_mapping[position][0],
            "end": context.offset_mapping[position][1],
        }
        for position in context.forget_positions
    ]
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
        "tokens_after_owned_span": (
            len(context.original_token_ids) - max(context.forget_positions)
        ),
        "ownership": {
            "mapping": "positive character-overlap from tokenizer offset_mapping",
            "owned_character_span": list(context.owned_character_span),
            "answer_character_span": list(context.answer_character_span),
            "forget_positions": list(context.forget_positions),
            "deletion_ranges": [
                {"start": start, "end": end}
                for start, end in context.deletion_ranges
            ],
            "owned_token_offsets": owned_offsets,
            "offset_mapping_sha256": _payload_sha256(
                [list(offset) for offset in context.offset_mapping]
            ),
        },
        "retained": {
            "character_span": list(context.retained_character_span),
            "token_positions": list(context.retained_positions),
            "edited_token_positions": list(context.edited_retained_positions),
            "token_count": len(context.retained_positions),
        },
    }


def _record_payload(
    example: WikiExample,
    hits: Sequence[RetrievalHit],
    retained: CorpusNode,
    counterfactual: CounterfactualPassage,
    context: TokenizedContext,
) -> dict[str, Any]:
    record_id = stable_identifier(
        "rag",
        DATASET_ID,
        DATASET_REVISION,
        example.example_id,
    )
    return {
        "record_id": record_id,
        "source_example_id": example.example_id,
        "source_fingerprints": {
            "question_sha256": text_sha256(example.question),
            "gold_answer_sha256": text_sha256(example.answer),
            "question_type_sha256": text_sha256(example.question_type),
        },
        "retrieval": [
            {
                "rank": hit.rank,
                "score": hit.score,
                "node_id": hit.node.node_id,
                "document_id": hit.node.document_id,
                "source_example_id": hit.node.source_example_id,
                "node_text_sha256": text_sha256(hit.node.text),
                "document_title_sha256": text_sha256(hit.node.title),
            }
            for hit in hits
        ],
        "gold_evidence": {
            "document_ids": list(example.supporting_document_ids),
            "retrieved_node_ids": [
                hit.node.node_id
                for hit in hits
                if hit.node.source_example_id == example.example_id
                and hit.node.is_gold_evidence
            ],
            "all_supporting_documents_retrieved": True,
        },
        "distractor": {
            "recipe_version": COUNTERFACTUAL_RECIPE_VERSION,
            "answer_strategy": counterfactual.answer_strategy,
            "answer_sha256": text_sha256(counterfactual.distractor_answer),
            "passage_text_sha256": text_sha256(counterfactual.text),
            "owned_span_sha256": text_sha256(
                counterfactual.text[
                    counterfactual.owned_start : counterfactual.owned_end
                ]
            ),
        },
        "retained_probe": {
            "recipe_version": RETAINED_PROBE_VERSION,
            "node_id": retained.node_id,
            "source_example_id": retained.source_example_id,
            "question_sha256": text_sha256(_RETAINED_QUESTION),
            "answer_sha256": text_sha256(retained.title),
        },
        "context": _context_descriptor(context),
    }


def build_manifest(
    examples: Sequence[WikiExample],
    nodes: Sequence[CorpusNode],
    retriever: Retriever,
    tokenizer: Any,
    *,
    requested_records: int,
    candidate_limit: int,
    seed: int = 0,
    mode: str = "smoke",
    top_k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    embedding_model_id: str = EMBEDDING_MODEL_ID,
    embedding_revision: str = EMBEDDING_MODEL_REVISION,
    minimum_tokens_after_owned: int = DEFAULT_MIN_TOKENS_AFTER_OWNED,
) -> dict[str, Any]:
    """Build and freeze a predeclared manifest from injected retrieval plumbing."""

    if requested_records < 1 or candidate_limit < requested_records:
        raise ValueError("candidate_limit must cover requested_records")
    if top_k < 2 or len(nodes) < top_k:
        raise ValueError("fixed top_k requires at least two corpus nodes")
    if minimum_tokens_after_owned < 0:
        raise ValueError("minimum_tokens_after_owned must be non-negative")
    ordered_examples = sorted(
        examples,
        key=lambda example: (
            _selection_key(seed, example.example_id),
            example.example_id,
        ),
    )[:candidate_limit]
    node_ids = [node.node_id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("corpus node IDs are not unique")

    records = []
    attempted_candidates = 0
    rejection_counts: dict[str, int] = {}
    for example in ordered_examples:
        attempted_candidates += 1
        try:
            raw_hits = retriever(example.question, int(top_k))
            hits = _canonical_hits(tuple(raw_hits), top_k=top_k)
            retrieved_support = {
                hit.node.document_id
                for hit in hits
                if hit.node.source_example_id == example.example_id
                and hit.node.is_gold_evidence
            }
            if not set(example.supporting_document_ids).issubset(retrieved_support):
                raise LookupError("gold_evidence_not_fully_retrieved")
            retained = next(
                (
                    hit.node
                    for hit in hits
                    if hit.node.source_example_id != example.example_id
                ),
                None,
            )
            if retained is None:
                raise LookupError("unrelated_retained_passage_not_retrieved")
            counterfactual = build_counterfactual(example, retained)
            context = package_context(
                tokenizer,
                counterfactual,
                hits,
                retained.node_id,
            )
            tokens_after_owned = (
                len(context.original_token_ids) - max(context.forget_positions)
            )
            if tokens_after_owned <= int(minimum_tokens_after_owned):
                raise LookupError("owned_span_inside_deletion_window")
            records.append(
                _record_payload(
                    example,
                    hits,
                    retained,
                    counterfactual,
                    context,
                )
            )
        except LookupError as exc:
            reason = str(exc)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        if len(records) == requested_records:
            break
    if len(records) != requested_records:
        raise RuntimeError(
            f"predeclared selection produced {len(records)} of "
            f"{requested_records} required records"
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "contains_source_text": False,
        "dataset": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "embedding": {
            "model_id": str(embedding_model_id),
            "revision": str(embedding_revision),
        },
        "tokenizer": {
            "model_id": str(tokenizer_id),
            "revision": str(tokenizer_revision),
            "offset_mapping_required": True,
        },
        "retrieval_config": {
            "framework": "llama-index-core",
            "framework_version": LLAMA_INDEX_CORE_VERSION,
            "embedding_integration": "llama-index-embeddings-huggingface",
            "embedding_integration_version": LLAMA_INDEX_HUGGINGFACE_VERSION,
            "document_type": "Document",
            "splitter": "SentenceSplitter",
            "index": "VectorStoreIndex",
            "top_k": int(top_k),
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "tie_break": "descending score then stable node ID",
        },
        "erasure_config": {
            "minimum_tokens_after_owned": int(minimum_tokens_after_owned),
            "criterion": (
                "original_token_count - maximum_owned_token_position must be "
                "strictly greater than the configured deletion window"
            ),
        },
        "selection_policy": {
            "mode": str(mode),
            "seed": int(seed),
            "requested_records": int(requested_records),
            "candidate_limit": int(candidate_limit),
            "candidate_order": "sha256(seed, source_example_id)",
            "required_gold_evidence": "all supporting documents",
            "required_retained_passage": "first cross-example retrieval hit",
            "fixed_before_model_evaluation": True,
            "gemma_outputs_used": False,
            "attempted_candidates": attempted_candidates,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "candidate_example_ids": [
                example.example_id for example in ordered_examples
            ],
        },
        "records": records,
    }
    return freeze_manifest(manifest)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _walk_keys(item)


def _validate_unfrozen_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported RAG manifest schema")
    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported RAG manifest schema version")
    if manifest.get("contains_source_text") is not False:
        raise ValueError("RAG manifests must not contain source text")
    leaked_keys = _SOURCE_TEXT_KEYS.intersection(_walk_keys(manifest))
    if leaked_keys:
        raise ValueError(
            "source-text keys are forbidden in frozen manifests: "
            + ", ".join(sorted(leaked_keys))
        )
    dataset = manifest.get("dataset") or {}
    if dataset.get("dataset_id") != DATASET_ID:
        raise ValueError("manifest dataset ID is not the pinned 2Wiki source")
    if dataset.get("revision") != DATASET_REVISION:
        raise ValueError("manifest dataset revision is not pinned")
    embedding = manifest.get("embedding") or {}
    if (
        embedding.get("model_id") != EMBEDDING_MODEL_ID
        or embedding.get("revision") != EMBEDDING_MODEL_REVISION
    ):
        raise ValueError("manifest embedding model or revision is not pinned")
    policy = manifest.get("selection_policy") or {}
    if policy.get("fixed_before_model_evaluation") is not True:
        raise ValueError("manifest selection was not frozen before evaluation")
    if policy.get("gemma_outputs_used") is not False:
        raise ValueError("manifest selection must be independent of Gemma outputs")
    retrieval_config = manifest.get("retrieval_config") or {}
    if (
        retrieval_config.get("framework") != "llama-index-core"
        or retrieval_config.get("framework_version")
        != LLAMA_INDEX_CORE_VERSION
        or retrieval_config.get("embedding_integration")
        != "llama-index-embeddings-huggingface"
        or retrieval_config.get("embedding_integration_version")
        != LLAMA_INDEX_HUGGINGFACE_VERSION
    ):
        raise ValueError("manifest retrieval package versions are not pinned")
    top_k = int(retrieval_config.get("top_k", 0))
    if top_k < 1:
        raise ValueError("manifest top_k must be positive")
    minimum_tokens_after_owned = int(
        (manifest.get("erasure_config") or {}).get(
            "minimum_tokens_after_owned",
            -1,
        )
    )
    if minimum_tokens_after_owned < 0:
        raise ValueError("manifest erasure geometry is missing")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest has no records")
    if len(records) != int(policy.get("requested_records", -1)):
        raise ValueError("manifest record count differs from predeclared request")
    candidate_ids = policy.get("candidate_example_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != len(set(candidate_ids))
        or len(candidate_ids) > int(policy.get("candidate_limit", -1))
    ):
        raise ValueError("manifest candidate pool is invalid")
    record_ids = [str(record.get("record_id")) for record in records]
    source_ids = [str(record.get("source_example_id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("manifest record IDs are not unique")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("manifest source example IDs are not unique")
    if not set(source_ids).issubset(set(candidate_ids)):
        raise ValueError("manifest records fall outside the candidate pool")
    for record in records:
        retrieval = record.get("retrieval")
        if not isinstance(retrieval, list) or len(retrieval) != top_k:
            raise ValueError("record retrieval length differs from fixed top_k")
        ranks = [int(item.get("rank", -1)) for item in retrieval]
        if ranks != list(range(1, top_k + 1)):
            raise ValueError("record retrieval ranks are not contiguous")
        if len({item.get("node_id") for item in retrieval}) != top_k:
            raise ValueError("record retrieval contains duplicate nodes")
        if any(
            not math.isfinite(float(item.get("score", math.nan)))
            for item in retrieval
        ):
            raise ValueError("record retrieval contains a non-finite score")
        retrieval_order = [
            (-float(item["score"]), str(item["node_id"]))
            for item in retrieval
        ]
        if retrieval_order != sorted(retrieval_order):
            raise ValueError("record retrieval order is not canonical")
        ownership = (record.get("context") or {}).get("ownership") or {}
        positions = ownership.get("forget_positions")
        if (
            not isinstance(positions, list)
            or not positions
            or positions != sorted(set(int(value) for value in positions))
        ):
            raise ValueError("record token ownership is empty or non-canonical")
        context = record["context"]
        if positions[-1] >= int(context["original_token_count"]):
            raise ValueError("record token ownership is outside the context")
        if int(context["original_token_count"]) - len(positions) != int(
            context["edited_token_count"]
        ):
            raise ValueError("record literal token edit count is inconsistent")
        if int(context.get("tokens_after_owned_span", -1)) != (
            int(context["original_token_count"]) - positions[-1]
        ):
            raise ValueError("record owned-span distance is inconsistent")
        if int(context.get("tokens_after_owned_span", -1)) <= (
            minimum_tokens_after_owned
        ):
            raise ValueError("record owned span is inside the deletion window")
        retained_probe = record.get("retained_probe") or {}
        retained_id = retained_probe.get("node_id")
        if retained_id not in {item["node_id"] for item in retrieval}:
            raise ValueError("retained probe node is outside frozen retrieval")
        if retained_probe.get("source_example_id") == record.get(
            "source_example_id"
        ):
            raise ValueError("retained probe is not cross-example unrelated")
        gold = record.get("gold_evidence") or {}
        if gold.get("all_supporting_documents_retrieved") is not True:
            raise ValueError("record does not freeze complete gold retrieval")
        if not gold.get("document_ids") or not gold.get("retrieved_node_ids"):
            raise ValueError("record gold evidence is empty")
        retrieved_ids = {item["node_id"] for item in retrieval}
        if not set(gold["retrieved_node_ids"]).issubset(retrieved_ids):
            raise ValueError("record gold nodes are outside frozen retrieval")


def freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-frozen JSON payload with record and root digests."""

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
    """Validate schema, invariants, and every tamper-evident digest."""

    _validate_unfrozen_manifest(manifest)
    for record in manifest["records"]:
        integrity = record.get("record_integrity")
        if not isinstance(integrity, Mapping):
            raise ValueError("manifest record integrity is missing")
        payload = copy.deepcopy(dict(record))
        payload.pop("record_integrity", None)
        expected = _payload_sha256(payload)
        if integrity.get("algorithm") != "sha256" or integrity.get(
            "sha256"
        ) != expected:
            raise ValueError(
                f"manifest record integrity mismatch: {record.get('record_id')}"
            )
    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("manifest root integrity is missing")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    expected = _payload_sha256(payload)
    if integrity.get("algorithm") != "sha256" or integrity.get(
        "sha256"
    ) != expected:
        raise ValueError("manifest root integrity mismatch")


def write_manifest_atomic(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Validate and atomically write deterministic JSON."""

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


def write_environment_lock(path: str | Path) -> None:
    """Record the complete resolved Python environment beside a live manifest."""

    rows = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in distributions()
        if distribution.metadata.get("Name")
    }
    rendered = (
        f"# python={platform.python_version()}\n"
        f"# executable={sys.executable}\n"
        + "\n".join(sorted(rows, key=str.casefold))
        + "\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, destination)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen text-free manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def _validate_rehydrated_record(
    stored: Mapping[str, Any],
    example: WikiExample,
    hits: Sequence[RetrievalHit],
    retained: CorpusNode,
    counterfactual: CounterfactualPassage,
    context: TokenizedContext,
) -> None:
    expected_record_id = stable_identifier(
        "rag",
        DATASET_ID,
        DATASET_REVISION,
        example.example_id,
    )
    if stored["record_id"] != expected_record_id:
        raise ValueError("rehydrated record ID differs from deterministic ID")
    fingerprints = stored["source_fingerprints"]
    expected_fingerprints = {
        "question_sha256": text_sha256(example.question),
        "gold_answer_sha256": text_sha256(example.answer),
        "question_type_sha256": text_sha256(example.question_type),
    }
    if fingerprints != expected_fingerprints:
        raise ValueError("rehydrated source question or answer hash differs")
    expected_gold = {
        "document_ids": list(example.supporting_document_ids),
        "retrieved_node_ids": [
            hit.node.node_id
            for hit in hits
            if hit.node.source_example_id == example.example_id
            and hit.node.is_gold_evidence
        ],
        "all_supporting_documents_retrieved": True,
    }
    if stored["gold_evidence"] != expected_gold:
        raise ValueError("rehydrated supporting-fact metadata differs")
    for stored_hit, hit in zip(stored["retrieval"], hits):
        expected_hit = {
            "rank": hit.rank,
            "score": hit.score,
            "node_id": hit.node.node_id,
            "document_id": hit.node.document_id,
            "source_example_id": hit.node.source_example_id,
            "node_text_sha256": text_sha256(hit.node.text),
            "document_title_sha256": text_sha256(hit.node.title),
        }
        if stored_hit != expected_hit:
            raise ValueError("rehydrated retrieval node differs from manifest")
    distractor = stored["distractor"]
    if distractor != {
        "recipe_version": COUNTERFACTUAL_RECIPE_VERSION,
        "answer_strategy": counterfactual.answer_strategy,
        "answer_sha256": text_sha256(counterfactual.distractor_answer),
        "passage_text_sha256": text_sha256(counterfactual.text),
        "owned_span_sha256": text_sha256(
            counterfactual.text[
                counterfactual.owned_start : counterfactual.owned_end
            ]
        ),
    }:
        raise ValueError("rehydrated counterfactual differs from manifest")
    retained_probe = stored["retained_probe"]
    if retained_probe != {
        "recipe_version": RETAINED_PROBE_VERSION,
        "node_id": retained.node_id,
        "source_example_id": retained.source_example_id,
        "question_sha256": text_sha256(_RETAINED_QUESTION),
        "answer_sha256": text_sha256(retained.title),
    }:
        raise ValueError("rehydrated retained probe differs from manifest")
    if stored["context"] != _context_descriptor(context):
        raise ValueError("rehydrated packaged context differs from manifest")


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    examples: Sequence[WikiExample],
    nodes: Sequence[CorpusNode],
    tokenizer: Any,
) -> tuple[RehydratedRecord, ...]:
    """Rebuild source text and reject any mismatch with the frozen manifest."""

    validate_manifest(manifest)
    examples_by_id = {example.example_id: example for example in examples}
    nodes_by_id = {node.node_id: node for node in nodes}
    result = []
    for stored in manifest["records"]:
        source_id = str(stored["source_example_id"])
        if source_id not in examples_by_id:
            raise ValueError(f"missing source example {source_id}")
        example = examples_by_id[source_id]
        hits = []
        for item in stored["retrieval"]:
            node_id = str(item["node_id"])
            if node_id not in nodes_by_id:
                raise ValueError(f"missing source node {node_id}")
            hits.append(
                RetrievalHit(
                    node=nodes_by_id[node_id],
                    score=float(item["score"]),
                    rank=int(item["rank"]),
                )
            )
        retained_id = str(stored["retained_probe"]["node_id"])
        retained = nodes_by_id[retained_id]
        counterfactual = build_counterfactual(example, retained)
        context = package_context(
            tokenizer,
            counterfactual,
            hits,
            retained.node_id,
        )
        _validate_rehydrated_record(
            stored,
            example,
            hits,
            retained,
            counterfactual,
            context,
        )
        result.append(
            RehydratedRecord(
                record_id=str(stored["record_id"]),
                example=example,
                retrieval_hits=tuple(hits),
                retained_node=retained,
                counterfactual=counterfactual,
                context=context,
                retained_question=_RETAINED_QUESTION,
                retained_answer=retained.title,
            )
        )
    return tuple(result)


def rehydrate_manifest_from_rows(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    nodes: Sequence[CorpusNode] | None = None,
) -> tuple[RehydratedRecord, ...]:
    """Rehydrate a manifest from pinned source rows and deterministic chunks."""

    validate_manifest(manifest)
    required_ids = manifest["selection_policy"]["candidate_example_ids"]
    examples = extract_2wiki_examples(
        rows,
        seed=int(manifest["selection_policy"]["seed"]),
        required_ids=required_ids,
    )
    if nodes is None:
        documents = build_document_specs(examples)
        config = manifest["retrieval_config"]
        nodes, _ = split_documents_with_llamaindex(
            documents,
            chunk_size=int(config["chunk_size"]),
            chunk_overlap=int(config["chunk_overlap"]),
        )
    return rehydrate_manifest(manifest, examples, nodes, tokenizer)


def load_2wiki_split(
    *,
    split: str = DATASET_SPLIT,
    revision: str = DATASET_REVISION,
):
    """Load the pinned dataset only when a live caller explicitly requests it."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for live manifest building or rehydration"
        ) from exc
    return load_dataset(
        DATASET_ID,
        revision=str(revision),
        split=str(split),
    )


def load_offset_tokenizer(
    model_id: str = DEFAULT_TOKENIZER_ID,
    *,
    revision: str = DEFAULT_TOKENIZER_REVISION,
):
    """Load a pinned fast tokenizer only at CLI/runtime execution."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for live model-token ownership"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_id),
        revision=str(revision),
        use_fast=True,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("RAG manifests require a fast offset-aware tokenizer")
    return tokenizer


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
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--minimum-tokens-after-owned",
        type=int,
        default=DEFAULT_MIN_TOKENS_AFTER_OWNED,
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
    """Build a smoke or full frozen manifest with optional live dependencies."""

    parser = _parser()
    args = parser.parse_args(argv)
    default_records, default_candidates = _mode_defaults(args.mode)
    records = args.records or default_records
    candidate_limit = args.candidate_limit or default_candidates
    if records < 1 or candidate_limit < records:
        parser.error("--candidate-limit must be at least --records")
    if (
        args.top_k < 2
        or args.chunk_size < 1
        or args.chunk_overlap < 0
        or args.chunk_overlap >= args.chunk_size
        or args.minimum_tokens_after_owned < 0
    ):
        parser.error("invalid retrieval or chunking configuration")
    output = Path(
        args.out
        or f"outputs/gemma_sv_rag/2wiki_{args.mode}_manifest.json"
    )
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")

    dataset_rows = load_2wiki_split()
    examples = extract_2wiki_examples(
        dataset_rows,
        seed=args.seed,
        limit=candidate_limit,
    )
    documents = build_document_specs(examples)
    embedding = load_pinned_embedding(device=args.embedding_device)
    nodes, retriever = build_llamaindex_retriever(
        documents,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
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
        requested_records=records,
        candidate_limit=candidate_limit,
        seed=args.seed,
        mode=args.mode,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        tokenizer_id=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        minimum_tokens_after_owned=args.minimum_tokens_after_owned,
    )
    write_manifest_atomic(output, manifest)
    write_environment_lock(output.parent / "environment-lock.txt")
    print(
        f"wrote {len(manifest['records'])} frozen records to {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

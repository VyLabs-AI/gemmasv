"""Additive v2 authorization for LongMemEval continuous-utility execution.

This module freezes source-only choices and execution contracts.  It cannot
load model weights or score a model.  Live execution additionally requires the
authorization and every bound input/implementation to be committed at HEAD,
an exact environment match, and an exact acknowledgement string.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence

from gemma_sv import longmemeval_chat_all_history_replay as replay


SCHEMA = "gemma-sv-longmemeval-chat-v3-utility-execution-authorization-v2"
SCHEMA_VERSION = 2
STATUS = "frozen-before-v3-continuous-utility-execution"
EXPECTED_CLUSTERS = 32
HISTORIES_PER_CLUSTER = 3
EXPECTED_HISTORIES = EXPECTED_CLUSTERS * HISTORIES_PER_CLUSTER

PRESENT = "present"
FRESH_REBUILD = "fresh_raw_omission"
POLICY = "exact_decrement_or_refit_policy"
REPLAY = "frozen_suffix_replay"
CONDITIONS = (PRESENT, FRESH_REBUILD, POLICY, REPLAY)
METHOD_CONDITIONS = (POLICY, REPLAY)
PROBES = ("target_current", "retained")

EXPLICIT_EXECUTION_ACKNOWLEDGEMENT = (
    "I_ACKNOWLEDGE_COMMITTED_LONGMEMEVAL_V3_UTILITY_EXECUTION"
)
CHOICE_SELECTION_SEED = "gemma-sv-longmemeval-v3-forced-choice-v1"
CHOICE_COUNT = 4
DISTRACTOR_COUNT = 3

BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20_260_823
CONFIDENCE_LEVEL = 0.95

MODEL_ID = "google/gemma-3-4b-it"
MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
TOKENIZER_ID = MODEL_ID
TOKENIZER_REVISION = MODEL_REVISION
MODEL_METADATA_FILE_SHA256 = {
    "config.json": (
        "9059f680f4dbd1957f35cb44b9fdd6948f4792db7a7a35ee353ea42e68adf7ff"
    ),
    "generation_config.json": (
        "fd9324becc53c4be610db39e13a613006f09fd6ef71a95fb6320dc33157490a3"
    ),
    "model.safetensors.index.json": (
        "77f4b67de084c31c7bcd373b039908108eee6c6181607e6d53da730e5f0bc659"
    ),
}
TOKENIZER_METADATA_FILE_SHA256 = {
    "tokenizer.json": (
        "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"
    ),
    "tokenizer_config.json": (
        "bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1"
    ),
    "special_tokens_map.json": (
        "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397"
    ),
    "added_tokens.json": (
        "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946"
    ),
}
TOKENIZER_BACKEND_SHA256 = (
    "c1a087240686a7d141101217051f76d5cd4cbe2b6093e3c3553fb26dcc4d0e9a"
)

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"

DEFAULT_V1_AUTHORIZATION = (
    BENCHMARKS / "longmemeval_chat_v3_utility_authorization_v1.json"
)
DEFAULT_DECODED_AUDIT = (
    BENCHMARKS / "longmemeval_chat_v3_summary_v1.json"
)
DEFAULT_ANALYSIS_LOCK = (
    BENCHMARKS / "longmemeval_chat_v3_analysis_lock_v1.json"
)
DEFAULT_REPLAY_LOCK = replay.DEFAULT_REPLAY_LOCK_PATH
DEFAULT_SUFFIX_SCAN = (
    BENCHMARKS / "longmemeval_chat_suffix_contamination_v1.json"
)
DEFAULT_COHORT = BENCHMARKS / "longmemeval_chat_cohort_v3.json"
DEFAULT_QUALITY_LOCK = BENCHMARKS / "training_free_base_ordering_v1.json"
DEFAULT_AUTHORIZATION = (
    BENCHMARKS
    / "longmemeval_chat_v3_utility_execution_authorization_v2.json"
)

ARTIFACT_PATHS = {
    "v1_utility_authorization": DEFAULT_V1_AUTHORIZATION,
    "completed_decoded_audit": DEFAULT_DECODED_AUDIT,
    "v3_analysis_lock": DEFAULT_ANALYSIS_LOCK,
    "all_history_replay_lock": DEFAULT_REPLAY_LOCK,
    "frozen_suffix_scan": DEFAULT_SUFFIX_SCAN,
    "cohort": DEFAULT_COHORT,
    "broad_wikitext_quality_lock": DEFAULT_QUALITY_LOCK,
}

IMPLEMENTATION_PATHS = (
    "gemma_sv/longmemeval_chat_v3_utility_protocol.py",
    "gemma_sv/longmemeval_chat_v3_utility_execution_protocol.py",
    "gemma_sv/longmemeval_chat_all_history_replay.py",
    "gemma_sv/demo_server/all_history_replay_runtime_v3.py",
    "gemma_sv/eval_longmemeval_chat_v3_utility.py",
    "gemma_sv/summarize_longmemeval_chat_v3_utility.py",
    "gemma_sv/demo_server/audit_gemma_runtime_v2.py",
    "gemma_sv/demo_server/gemma_engine.py",
    "gemma_sv/eval_longmemeval_chat_v2.py",
    "gemma_sv/persistent_deletion.py",
    "gemma_sv/sv_global_attention.py",
    "svattn/causal_sv_attention.py",
)

_ENVIRONMENT_DISTRIBUTIONS = {
    "accelerate": "accelerate",
    "clarabel": "clarabel",
    "cvxpy": "cvxpy",
    "datasets": "datasets",
    "huggingface-hub": "huggingface-hub",
    "mlx": "mlx",
    "numpy": "numpy",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
}
_ENVIRONMENT_VARIABLES = (
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_FORBIDDEN_AUTHORIZATION_KEYS = frozenset(
    {
        "answer",
        "choice_answers",
        "cluster_id",
        "content",
        "messages",
        "prompt",
        "question",
        "record_id",
        "response_text",
        "source_id",
        "source_text",
        "text",
        "token_ids",
        "turns",
    }
)


class UtilityProtocolError(ValueError):
    """A source-only choice, artifact, or authorization contract differs."""


def canonical_json_bytes(value: Any) -> bytes:
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


def _is_sha256(value: Any) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(
        character in "0123456789abcdef" for character in rendered
    )


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
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != payload_sha256(body)
    ):
        raise UtilityProtocolError(f"{name} integrity differs")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UtilityProtocolError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise UtilityProtocolError(f"non-finite JSON constant {value!r}")


def load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UtilityProtocolError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise UtilityProtocolError(f"{name} must be a JSON object")
    return value


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key).casefold()
            yield from _walk_keys(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_keys(child)


def assert_source_free_authorization(value: Mapping[str, Any]) -> None:
    leaked = _FORBIDDEN_AUTHORIZATION_KEYS.intersection(_walk_keys(value))
    if leaked:
        raise UtilityProtocolError(
            "authorization contains source-bearing keys: "
            + ", ".join(sorted(leaked))
        )
    if (
        value.get("contains_source_text") is not False
        or value.get("contains_source_identifiers") is not False
        or value.get("contains_token_arrays") is not False
        or value.get("contains_model_outputs") is not False
    ):
        raise UtilityProtocolError("authorization is not source-free")


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


_INTEGER_RE = re.compile(r"^[+-]?\d[\d,]*$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d[\d,]*)?\.\d+$")
_PERCENT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?)\s*%$")
_CURRENCY_RE = re.compile(
    r"^(?:[$€£¥]\s*\d|(?:USD|EUR|GBP|JPY)\s+\d)", re.IGNORECASE
)
_TIME_RE = re.compile(
    r"^(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*(?:am|pm))?$|"
    r"^\d{1,2}\s*(?:am|pm)$",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?:\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\b|\b(?:19|20)\d{2}\b)",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)$",
    re.IGNORECASE,
)


def infer_answer_type(answer: str, question: str = "") -> str | None:
    """Infer the frozen source-only answer-type ontology deterministically."""

    rendered = " ".join(str(answer).strip().split())
    query = str(question).strip().casefold()
    if not rendered or len(rendered) > 160:
        return None
    if _PERCENT_RE.fullmatch(rendered):
        return "percentage"
    if _CURRENCY_RE.match(rendered):
        return "currency"
    if _DURATION_RE.fullmatch(rendered):
        return "duration"
    if _TIME_RE.fullmatch(rendered):
        return "time"
    if _DATE_RE.search(rendered):
        return "date"
    if _INTEGER_RE.fullmatch(rendered):
        return "integer"
    if _DECIMAL_RE.fullmatch(rendered):
        return "decimal"
    if query.startswith("who ") or query.startswith("whose "):
        if any(
            marker in query
            for marker in ("company", "organization", "team", "agency", "group")
        ):
            return "organization"
        return "person"
    if query.startswith("where ") or any(
        marker in query
        for marker in ("which city", "which country", "what location", "which place")
    ):
        return "location"
    if any(
        marker in query
        for marker in ("which company", "which organization", "what company")
    ):
        return "organization"
    return "other_short_text"


@dataclass(frozen=True)
class ForcedChoiceSet:
    """Private source-only choice values plus their source-free projection."""

    available: bool
    answer_type: str | None
    choice_answers: tuple[str, ...]
    choice_source_ids: tuple[str, ...]
    choice_answer_sha256: tuple[str, ...]
    choice_source_sha256: tuple[str, ...]
    gold_index: int | None
    unavailable_reason: str | None
    binding_sha256: str

    def __post_init__(self) -> None:
        if self.available:
            if (
                self.answer_type is None
                or len(self.choice_answers) != CHOICE_COUNT
                or len(self.choice_source_ids) != CHOICE_COUNT
                or len(self.choice_answer_sha256) != CHOICE_COUNT
                or len(self.choice_source_sha256) != CHOICE_COUNT
                or type(self.gold_index) is not int
                or not 0 <= self.gold_index < CHOICE_COUNT
                or self.unavailable_reason is not None
                or len(set(self.choice_source_ids)) != CHOICE_COUNT
                or len(set(answer.casefold() for answer in self.choice_answers))
                != CHOICE_COUNT
            ):
                raise ValueError("available forced-choice set is invalid")
        elif (
            self.choice_answers
            or self.choice_source_ids
            or self.choice_answer_sha256
            or self.choice_source_sha256
            or self.gold_index is not None
            or not self.unavailable_reason
        ):
            raise ValueError("unavailable forced-choice slot must remain empty")
        if not _is_sha256(self.binding_sha256):
            raise ValueError("forced-choice binding must be SHA-256")

    def public_descriptor(self) -> dict[str, Any]:
        if not self.available:
            return {
                "availability": "unavailable",
                "unavailable_reason": self.unavailable_reason,
                "answer_type": self.answer_type,
                "choice_count": 0,
                "gold_index": None,
                "choice_answer_sha256": [],
                "choice_source_sha256": [],
                "binding_sha256": self.binding_sha256,
                "replacement_allowed": False,
            }
        return {
            "availability": "available",
            "unavailable_reason": None,
            "answer_type": self.answer_type,
            "choice_count": CHOICE_COUNT,
            "gold_index": self.gold_index,
            "choice_answer_sha256": list(self.choice_answer_sha256),
            "choice_source_sha256": list(self.choice_source_sha256),
            "binding_sha256": self.binding_sha256,
            "replacement_allowed": False,
        }


def _unavailable_choice(
    *,
    answer_type: str | None,
    reason: str,
    slot_key: str,
) -> ForcedChoiceSet:
    binding = payload_sha256(
        {
            "slot_key": slot_key,
            "availability": "unavailable",
            "answer_type": answer_type,
            "reason": reason,
            "replacement_allowed": False,
        }
    )
    return ForcedChoiceSet(
        available=False,
        answer_type=answer_type,
        choice_answers=(),
        choice_source_ids=(),
        choice_answer_sha256=(),
        choice_source_sha256=(),
        gold_index=None,
        unavailable_reason=reason,
        binding_sha256=binding,
    )


def build_forced_choice_set(
    gold: Any,
    candidates: Sequence[Any],
    *,
    slot_key: str,
    forbidden_source_ids: Iterable[str] = (),
    answer_type_fn: Callable[[str, str], str | None] = infer_answer_type,
) -> ForcedChoiceSet:
    """Freeze gold plus three source-disjoint same-type distractors by SHA-256."""

    gold_answer = str(_value(gold, "answer", "")).strip()
    gold_source = str(_value(gold, "source_id", "")).strip()
    gold_question = str(_value(gold, "question", ""))
    answer_type = answer_type_fn(gold_answer, gold_question)
    if not gold_answer or not gold_source:
        return _unavailable_choice(
            answer_type=answer_type,
            reason="gold_source_or_answer_unavailable",
            slot_key=slot_key,
        )
    if answer_type is None:
        return _unavailable_choice(
            answer_type=None,
            reason="ambiguous_gold_answer_type",
            slot_key=slot_key,
        )
    forbidden = {str(value) for value in forbidden_source_ids}
    forbidden.add(gold_source)
    gold_normalized = " ".join(gold_answer.casefold().split())
    eligible: list[tuple[str, str, str, str]] = []
    for candidate in candidates:
        answer = str(_value(candidate, "answer", "")).strip()
        source_id = str(_value(candidate, "source_id", "")).strip()
        question = str(_value(candidate, "question", ""))
        normalized = " ".join(answer.casefold().split())
        if (
            not answer
            or not source_id
            or source_id in forbidden
            or normalized == gold_normalized
            or answer_type_fn(answer, question) != answer_type
        ):
            continue
        rank = payload_sha256(
            {
                "seed": CHOICE_SELECTION_SEED,
                "slot_key": slot_key,
                "role": "distractor",
                "source_sha256": text_sha256(source_id),
                "answer_sha256": text_sha256(answer),
            }
        )
        eligible.append((rank, source_id, answer, normalized))
    eligible.sort()
    distractors: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    seen_answers: set[str] = {gold_normalized}
    for _rank, source_id, answer, normalized in eligible:
        if source_id in seen_sources or normalized in seen_answers:
            continue
        seen_sources.add(source_id)
        seen_answers.add(normalized)
        distractors.append((source_id, answer))
        if len(distractors) == DISTRACTOR_COUNT:
            break
    if len(distractors) < DISTRACTOR_COUNT:
        return _unavailable_choice(
            answer_type=answer_type,
            reason="fewer_than_three_source_disjoint_typed_distractors",
            slot_key=slot_key,
        )
    selected = [(gold_source, gold_answer, True)] + [
        (source_id, answer, False)
        for source_id, answer in distractors
    ]
    ranked = sorted(
        selected,
        key=lambda row: (
            payload_sha256(
                {
                    "seed": CHOICE_SELECTION_SEED,
                    "slot_key": slot_key,
                    "role": "choice_order",
                    "source_sha256": text_sha256(row[0]),
                    "answer_sha256": text_sha256(row[1]),
                }
            ),
            text_sha256(row[1]),
        ),
    )
    answers = tuple(row[1] for row in ranked)
    sources = tuple(row[0] for row in ranked)
    answer_hashes = tuple(text_sha256(value) for value in answers)
    source_hashes = tuple(text_sha256(value) for value in sources)
    gold_index = next(index for index, row in enumerate(ranked) if row[2])
    binding = payload_sha256(
        {
            "slot_key": slot_key,
            "availability": "available",
            "answer_type": answer_type,
            "choice_answer_sha256": answer_hashes,
            "choice_source_sha256": source_hashes,
            "gold_index": gold_index,
            "selection_seed": CHOICE_SELECTION_SEED,
        }
    )
    return ForcedChoiceSet(
        available=True,
        answer_type=answer_type,
        choice_answers=answers,
        choice_source_ids=sources,
        choice_answer_sha256=answer_hashes,
        choice_source_sha256=source_hashes,
        gold_index=gold_index,
        unavailable_reason=None,
        binding_sha256=binding,
    )


def _record_source_ids(record: Any) -> set[str]:
    values: set[str] = set()
    for name in ("target", "retained"):
        source_id = str(_value(_value(record, name), "source_id", ""))
        if source_id:
            values.add(source_id)
    for name in (
        "earlier_evidence",
        "owned_session_prefix",
        "retained_evidence",
        "tail_evidence",
    ):
        for item in _value(record, name, ()) or ():
            source_id = str(_value(item, "source_id", ""))
            if source_id:
                values.add(source_id)
    return values


def freeze_forced_choice_sets(
    records: Sequence[Any],
    source_examples: Sequence[Any],
) -> tuple[
    tuple[dict[str, ForcedChoiceSet], ...],
    tuple[dict[str, Any], ...],
]:
    """Freeze target and retained slots for all 96 histories, without replacement."""

    if len(records) != EXPECTED_HISTORIES:
        raise UtilityProtocolError("forced choices require all 96 histories")
    private: list[dict[str, ForcedChoiceSet]] = []
    public: list[dict[str, Any]] = []
    for history_index, record in enumerate(records):
        choices: dict[str, ForcedChoiceSet] = {}
        forbidden = _record_source_ids(record)
        for probe_id, field in (
            ("target_current", "target"),
            ("retained", "retained"),
        ):
            choices[probe_id] = build_forced_choice_set(
                _value(record, field),
                source_examples,
                slot_key=f"{history_index}:{probe_id}",
                forbidden_source_ids=forbidden,
            )
        private.append(choices)
        public.append(
            {
                "history_index": history_index,
                "cluster_index": history_index // HISTORIES_PER_CLUSTER,
                "variant_index": history_index % HISTORIES_PER_CLUSTER,
                "target_current": choices["target_current"].public_descriptor(),
                "retained": choices["retained"].public_descriptor(),
            }
        )
    return tuple(private), tuple(public)


def validate_choice_descriptors(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != EXPECTED_HISTORIES:
        raise UtilityProtocolError("choice lock must contain exactly 96 rows")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or row.get("history_index") != index
            or row.get("cluster_index") != index // HISTORIES_PER_CLUSTER
            or row.get("variant_index") != index % HISTORIES_PER_CLUSTER
        ):
            raise UtilityProtocolError("forced-choice anonymous order differs")
        for probe_id in PROBES:
            slot = row.get(probe_id)
            if not isinstance(slot, Mapping) or not _is_sha256(
                slot.get("binding_sha256")
            ):
                raise UtilityProtocolError("forced-choice slot binding differs")
            availability = slot.get("availability")
            if availability == "available":
                if (
                    slot.get("choice_count") != CHOICE_COUNT
                    or type(slot.get("gold_index")) is not int
                    or not 0 <= slot["gold_index"] < CHOICE_COUNT
                    or len(slot.get("choice_answer_sha256") or ())
                    != CHOICE_COUNT
                    or len(slot.get("choice_source_sha256") or ())
                    != CHOICE_COUNT
                    or any(
                        not _is_sha256(value)
                        for value in (
                            list(slot["choice_answer_sha256"])
                            + list(slot["choice_source_sha256"])
                        )
                    )
                    or slot.get("replacement_allowed") is not False
                ):
                    raise UtilityProtocolError(
                        "available forced-choice slot differs"
                    )
            elif availability == "unavailable":
                if (
                    slot.get("choice_count") != 0
                    or slot.get("gold_index") is not None
                    or slot.get("choice_answer_sha256") != []
                    or slot.get("choice_source_sha256") != []
                    or not slot.get("unavailable_reason")
                    or slot.get("replacement_allowed") is not False
                ):
                    raise UtilityProtocolError(
                        "unavailable forced-choice slot was replaced"
                    )
            else:
                raise UtilityProtocolError("forced-choice availability differs")


def artifact_binding(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        relative = source.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError as exc:
        raise UtilityProtocolError("bound artifact is outside workspace") from exc
    integrity = value.get("integrity")
    return {
        "repository_path": relative,
        "file_sha256": file_sha256(source),
        "payload_sha256": payload_sha256(value),
        "integrity_sha256": (
            integrity.get("sha256")
            if isinstance(integrity, Mapping)
            else None
        ),
        "lock_sha256": value.get("lock_sha256"),
        "immutable_input": True,
    }


def implementation_bindings() -> dict[str, Any]:
    files = {
        relative: file_sha256(WORKSPACE / relative)
        for relative in IMPLEMENTATION_PATHS
    }
    return {
        "files": files,
        "file_count": len(files),
        "files_sha256": payload_sha256(files),
        "all_files_must_equal_committed_head": True,
        "override_allowed": False,
    }


def _validate_bound_inputs(
    v1: Mapping[str, Any],
    decoded: Mapping[str, Any],
    analysis: Mapping[str, Any],
    replay_lock: Mapping[str, Any],
    suffix_scan: Mapping[str, Any],
    quality_lock: Mapping[str, Any],
) -> None:
    replay.validate_replay_lock(replay_lock)
    if (
        v1.get("schema")
        != "gemma-sv-longmemeval-chat-v3-utility-authorization-v1"
        or v1.get("schema_version") != 1
        or v1.get("status")
        != "frozen-before-continuous-utility-model-scoring"
        or (v1.get("cohort") or {}).get("history_instances")
        != EXPECTED_HISTORIES
    ):
        raise UtilityProtocolError("v1 utility authorization differs")
    if (
        decoded.get("schema")
        != "gemma-sv-longmemeval-chat-v3-summary-v1"
        or decoded.get("status") != "complete-post-hoc-deterministic"
        or (decoded.get("flow") or {}).get("completed_histories")
        != EXPECTED_HISTORIES
    ):
        raise UtilityProtocolError("completed decoded audit differs")
    if (
        analysis.get("schema")
        != "gemma-sv-longmemeval-chat-v3-analysis-lock-v1"
        or analysis.get("status")
        != "frozen-after-sealed-v2-audit-before-v3-summary-materialization"
        or (analysis.get("flow") or {}).get("terminal_histories")
        != EXPECTED_HISTORIES
    ):
        raise UtilityProtocolError("v3 analysis lock differs")
    suffix_rows = suffix_scan.get("histories")
    if (
        not isinstance(suffix_rows, list)
        or len(suffix_rows) != EXPECTED_HISTORIES
        or suffix_scan.get("contains_source_text") is not False
    ):
        raise UtilityProtocolError("frozen suffix scan differs")
    quality = quality_lock.get("evaluation")
    if (
        quality_lock.get("schema") != "gemma-sv-training-free-base-ordering-v1"
        or not isinstance(quality, Mapping)
        or quality.get("blocks") != 400
        or quality.get("sequence_length") != 512
        or quality.get("selection")
        != "first 400 consecutive nonoverlapping packed test blocks"
    ):
        raise UtilityProtocolError("bound 400x512 WikiText arm differs")


def _artifact_bindings_from_paths(
    values: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    if set(values) != set(ARTIFACT_PATHS) or set(paths) != set(ARTIFACT_PATHS):
        raise UtilityProtocolError("execution artifact roles differ")
    return {
        role: artifact_binding(paths[role], values[role])
        for role in ARTIFACT_PATHS
    }


def _exact_environment_from_v1(v1: Mapping[str, Any]) -> dict[str, Any]:
    runtime = v1.get("runtime")
    if not isinstance(runtime, Mapping):
        raise UtilityProtocolError("v1 runtime environment is missing")
    required = {
        "platform": copy.deepcopy(runtime.get("platform")),
        "packages": copy.deepcopy(runtime.get("packages")),
        "execution": {
            key: (runtime.get("execution") or {}).get(key)
            for key in (
                "device",
                "dtype",
                "mps_built",
                "mps_available",
                "local_files_only",
                "network_access",
            )
        },
        "environment_controls": copy.deepcopy(
            runtime.get("environment_controls")
        ),
    }
    if (
        not isinstance(required["platform"], Mapping)
        or not isinstance(required["packages"], Mapping)
        or not isinstance(required["environment_controls"], Mapping)
    ):
        raise UtilityProtocolError("v1 exact environment is incomplete")
    return required


def metric_contract() -> dict[str, Any]:
    return {
        "teacher_forced": {
            "probes": list(PROBES),
            "conditions": list(CONDITIONS),
            "complete_sequence_log_probability_nats": True,
            "first_target_token_full_vocabulary_rank": True,
            "rank_ties": "count only logits strictly greater than gold",
            "all_raw_scalar_and_gold_token_values_retained_in_private_shards": True,
        },
        "continuation_distribution": {
            "reference": FRESH_REBUILD,
            "methods": list(METHOD_CONDITIONS),
            "direction": "KL(fresh_rebuild || method)",
            "scope": "full vocabulary at every gold continuation token",
            "target_statistics": ["mean_across_target_tokens", "maximum"],
            "retained_statistics": [
                "first_token_kl",
                "mean_across_target_tokens",
                "maximum",
            ],
            "natural_log_units": "nats",
        },
        "retained": {
            "sequence_log_probability_drift": (
                "method_mean_logp_minus_fresh_rebuild_mean_logp"
            ),
            "absolute_sequence_log_probability_drift": True,
            "first_token_rank": True,
            "first_token_full_vocabulary_kl": True,
        },
        "forced_choice": {
            "source_only": True,
            "choice_count": CHOICE_COUNT,
            "composition": "gold plus three answer-type-matched distractors",
            "source_disjoint": True,
            "selection": "SHA-256 rank",
            "same_choice_set_across_all_conditions": True,
            "score": "mean complete-sequence log probability",
            "gold_rank_range": [1, CHOICE_COUNT],
            "tie_rule": (
                "descending score, then ascending frozen answer hash, then "
                "ascending choice index"
            ),
            "invalid_set_policy": "explicit unavailable slot without replacement",
            "headline_endpoints_replace_judge": [
                "deterministic_forced_choice_gold_rank",
                "full_vocabulary_first_gold_token_rank",
            ],
        },
    }


def build_execution_authorization(
    v1: Mapping[str, Any],
    decoded: Mapping[str, Any],
    analysis: Mapping[str, Any],
    replay_lock: Mapping[str, Any],
    suffix_scan: Mapping[str, Any],
    quality_lock: Mapping[str, Any],
    choice_descriptors: Sequence[Mapping[str, Any]],
    *,
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    implementations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the source-free v2 authorization in memory."""

    _validate_bound_inputs(
        v1,
        decoded,
        analysis,
        replay_lock,
        suffix_scan,
        quality_lock,
    )
    validate_choice_descriptors(choice_descriptors)
    if set(artifact_bindings) != set(ARTIFACT_PATHS):
        raise UtilityProtocolError("artifact binding roles differ")
    admission = copy.deepcopy(v1.get("admission"))
    if (
        not isinstance(admission, Mapping)
        or admission.get("source_conditions_only")
        != [PRESENT, FRESH_REBUILD]
        or admission.get("threshold_overrides_allowed") is not False
    ):
        raise UtilityProtocolError("v1 control-only admission differs")
    broad_quality = copy.deepcopy(v1.get("broad_graft_quality_arm"))
    if (
        not isinstance(broad_quality, Mapping)
        or (broad_quality.get("dataset") or {}).get("blocks") != 400
        or (broad_quality.get("dataset") or {}).get("sequence_length") != 512
    ):
        raise UtilityProtocolError("v1 broad quality arm differs")
    replay_rows = replay_lock["plans"]
    suffix_strata = [str(row["suffix_scan_stratum"]) for row in replay_rows]
    if (
        len(suffix_strata) != EXPECTED_HISTORIES
        or any(value not in {"clean", "contaminated"} for value in suffix_strata)
    ):
        raise UtilityProtocolError("replay suffix strata differ")
    code = implementation_bindings() if implementations is None else implementations
    lock = _seal(
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "contains_source_text": False,
            "contains_source_identifiers": False,
            "contains_token_arrays": False,
            "contains_model_outputs": False,
            "model_or_api_calls_made": 0,
            "additive_to_v1_authorization": True,
            "artifacts": copy.deepcopy(dict(artifact_bindings)),
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "metadata_file_sha256": copy.deepcopy(
                    MODEL_METADATA_FILE_SHA256
                ),
                "adapter": None,
            },
            "tokenizer": {
                "id": TOKENIZER_ID,
                "revision": TOKENIZER_REVISION,
                "metadata_file_sha256": copy.deepcopy(
                    TOKENIZER_METADATA_FILE_SHA256
                ),
                "backend_sha256": TOKENIZER_BACKEND_SHA256,
                "use_fast": True,
            },
            "runtime": {
                "class": (
                    "gemma_sv.demo_server.all_history_replay_runtime_v3."
                    "AllHistoryReplayGemmaRuntimeV3"
                ),
                "exact_environment": _exact_environment_from_v1(v1),
                "exact_environment_match_required": True,
                "local_files_only": True,
                "network_access": False,
                "overrides_allowed": False,
            },
            "implementation": copy.deepcopy(dict(code)),
            "cohort": {
                "K": EXPECTED_CLUSTERS,
                "n": EXPECTED_HISTORIES,
                "histories_per_cluster": HISTORIES_PER_CLUSTER,
                "all_histories_attempted_in_every_required_condition": True,
                "output_based_filtering": False,
                "replacement": False,
            },
            "conditions": {
                "ordered": list(CONDITIONS),
                "controls": [PRESENT, FRESH_REBUILD],
                "methods": [POLICY, REPLAY],
                "fresh_rebuild_label_is_raw_omission": True,
                "policy_and_replay_run_regardless_of_admission": True,
            },
            "phase_order": {
                "global_barriers_required": True,
                "phases": [
                    {
                        "ordinal": 1,
                        "id": "controls",
                        "conditions": [PRESENT, FRESH_REBUILD],
                        "histories": EXPECTED_HISTORIES,
                        "all_terminal_before_next_phase": True,
                    },
                    {
                        "ordinal": 2,
                        "id": "admission",
                        "inputs": [PRESENT, FRESH_REBUILD],
                        "method_outputs_used": False,
                    },
                    {
                        "ordinal": 3,
                        "id": "policy_and_replay",
                        "conditions": [POLICY, REPLAY],
                        "histories_per_condition": EXPECTED_HISTORIES,
                        "admission_filtering": False,
                    },
                    {
                        "ordinal": 4,
                        "id": "post_delete_wikitext",
                        "conditions": [FRESH_REBUILD, POLICY, REPLAY],
                        "cluster_blocks": EXPECTED_CLUSTERS,
                    },
                ],
            },
            "admission": admission,
            "metrics": metric_contract(),
            "forced_choice_sets": {
                "selection_seed": CHOICE_SELECTION_SEED,
                "rows": copy.deepcopy(list(choice_descriptors)),
                "rows_sha256": payload_sha256(choice_descriptors),
                "set_count": EXPECTED_HISTORIES * len(PROBES),
                "output_based_replacement": False,
            },
            "analysis": {
                "primary_estimand": (
                    "intent-to-treat continuous utility over all K=32 clusters"
                ),
                "history_instances_nested_per_cluster": HISTORIES_PER_CLUSTER,
                "independent_n": EXPECTED_CLUSTERS,
                "failure_and_nonadmission_value_for_bounded_metrics": 0.0,
                "unbounded_failed_raw_values": None,
                "all_failures_retained": True,
                "failure_based_replacement": False,
                "summaries": [
                    "mean",
                    "median",
                    "IQR",
                    "range",
                    "empirical_distribution",
                    "negative_effect_count",
                    "collateral_damage_count",
                ],
                "bootstrap": {
                    "method": "percentile_cluster_bootstrap",
                    "confidence_level": CONFIDENCE_LEVEL,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "resampling_unit": "target_cluster",
                    "clusters_per_resample": EXPECTED_CLUSTERS,
                    "histories_resampled_within_cluster": False,
                    "prng": "splitmix64",
                    "percentile_interpolation": "linear_type_7",
                },
            },
            "broad_wikitext_quality_arm": broad_quality,
            "post_delete_wikitext_arm": {
                "label": "external-corpus held-out utility",
                "derived_from_bound_400x512_arm": True,
                "blocks": 32,
                "block_indices": list(range(EXPECTED_CLUSTERS)),
                "block_index_equals_cluster_index": True,
                "same_block_for_three_histories": True,
                "prompt_token_slice": [0, 256],
                "target_token_slice": [256, 512],
                "conditions": [FRESH_REBUILD, POLICY, REPLAY],
                "paired_reference": FRESH_REBUILD,
                "metrics": [
                    "mean_nll_nats",
                    "perplexity",
                    "method_minus_rebuild_mean_nll_nats",
                    "paired_relative_perplexity_cost_percent",
                ],
                "overlap_audit": {
                    "distinct_corpus_provenance_proves_lexical_disjointness": False,
                    "normalized_overlap_must_be_disclosed": True,
                    "unperformed_value": "not_audited",
                },
            },
            "replay_disclosure": {
                "dependency_aware_never_ingested_history": False,
                "frozen_suffix_scan_stratification_required": True,
                "suffix_strata_by_history": suffix_strata,
                "suffix_strata_sha256": payload_sha256(suffix_strata),
                "bandwidth_source": "prefix_checkpoint",
                "fresh_rebuild_bandwidth_equivalence_assumed": False,
                "silent_kpar_override_allowed": False,
            },
            "sharding": {
                "control_pattern": "controls/<history_index>.json",
                "method_pattern": "methods/<history_index>.json",
                "terminal_shards_never_overwritten": True,
                "canonical_assembly_order": "history_index 0..95",
                "resume_reuses_only_valid_terminal_shards": True,
            },
            "authorization": {
                "authorization_and_all_bindings_must_equal_committed_head": True,
                "commit_check_precedes_model_loading": True,
                "exact_acknowledgement_required": (
                    EXPLICIT_EXECUTION_ACKNOWLEDGEMENT
                ),
                "exact_environment_required": True,
                "live_execution_performed_by_builder": False,
            },
        }
    )
    validate_execution_authorization(lock)
    return lock


freeze_execution_authorization = build_execution_authorization


def validate_execution_authorization(lock: Mapping[str, Any]) -> None:
    _validate_seal(lock, name="execution authorization")
    assert_source_free_authorization(lock)
    choices = lock.get("forced_choice_sets") or {}
    rows = choices.get("rows")
    phase_order = lock.get("phase_order") or {}
    phases = phase_order.get("phases")
    if (
        lock.get("schema") != SCHEMA
        or lock.get("schema_version") != SCHEMA_VERSION
        or lock.get("status") != STATUS
        or lock.get("model_or_api_calls_made") != 0
        or set(lock.get("artifacts") or {}) != set(ARTIFACT_PATHS)
        or not isinstance(rows, list)
        or choices.get("rows_sha256") != payload_sha256(rows)
        or not isinstance(phases, list)
        or [row.get("id") for row in phases]
        != [
            "controls",
            "admission",
            "policy_and_replay",
            "post_delete_wikitext",
        ]
    ):
        raise UtilityProtocolError("execution authorization structure differs")
    validate_choice_descriptors(rows)
    conditions = lock.get("conditions") or {}
    admission = lock.get("admission") or {}
    replay_disclosure = lock.get("replay_disclosure") or {}
    wikitext = lock.get("post_delete_wikitext_arm") or {}
    if (
        conditions.get("ordered") != list(CONDITIONS)
        or conditions.get("policy_and_replay_run_regardless_of_admission")
        is not True
        or admission.get("source_conditions_only")
        != [PRESENT, FRESH_REBUILD]
        or admission.get("policy_or_edited_outputs_used") is not False
        or admission.get("threshold_overrides_allowed") is not False
        or replay_disclosure.get("dependency_aware_never_ingested_history")
        is not False
        or replay_disclosure.get("suffix_strata_sha256")
        != payload_sha256(
            replay_disclosure.get("suffix_strata_by_history") or []
        )
        or len(replay_disclosure.get("suffix_strata_by_history") or [])
        != EXPECTED_HISTORIES
        or any(
            value not in {"clean", "contaminated"}
            for value in replay_disclosure.get("suffix_strata_by_history") or []
        )
        or replay_disclosure.get("bandwidth_source") != "prefix_checkpoint"
        or replay_disclosure.get("silent_kpar_override_allowed") is not False
        or wikitext.get("block_indices") != list(range(EXPECTED_CLUSTERS))
        or wikitext.get("prompt_token_slice") != [0, 256]
        or wikitext.get("target_token_slice") != [256, 512]
    ):
        raise UtilityProtocolError("execution scientific contract differs")


def load_bound_inputs(
    *,
    paths: Mapping[str, Path] = ARTIFACT_PATHS,
) -> dict[str, dict[str, Any]]:
    values = {
        role: load_json(path, name=role) for role, path in paths.items()
    }
    _validate_bound_inputs(
        values["v1_utility_authorization"],
        values["completed_decoded_audit"],
        values["v3_analysis_lock"],
        values["all_history_replay_lock"],
        values["frozen_suffix_scan"],
        values["broad_wikitext_quality_lock"],
    )
    return values


def _require_head_committed_file(path: str | Path) -> None:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError as exc:
        raise PermissionError("bound file is outside workspace") from exc
    completed = subprocess.run(
        ["git", "-C", str(WORKSPACE), "show", f"HEAD:{relative}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise PermissionError(f"{relative} must be committed at HEAD")
    if not resolved.is_file() or completed.stdout != resolved.read_bytes():
        raise PermissionError(f"{relative} differs from committed HEAD")


def _require_bound_file_hash(path: str | Path, expected_sha256: str) -> None:
    resolved = Path(path).resolve()
    if not _is_sha256(expected_sha256):
        raise PermissionError("bound file SHA-256 is invalid")
    if not resolved.is_file() or file_sha256(resolved) != expected_sha256:
        raise PermissionError(f"{resolved.name} differs from authorization hash")


def observe_exact_environment() -> dict[str, Any]:
    """Observe the execution environment without loading model weights."""

    packages: dict[str, str] = {}
    for label, distribution in _ENVIRONMENT_DISTRIBUTIONS.items():
        try:
            packages[label] = package_version(distribution)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"required package {distribution!r} is unavailable"
            ) from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for environment binding") from exc
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
        },
        "packages": {
            **packages,
            "torch_git_version": str(torch.version.git_version),
        },
        "execution": {
            "device": "mps",
            "dtype": "float32",
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "local_files_only": True,
            "network_access": False,
        },
        "environment_controls": {
            name: os.environ.get(name) for name in _ENVIRONMENT_VARIABLES
        },
    }


def authorize_execution(
    lock: Mapping[str, Any],
    *,
    acknowledgement: str,
    observed_environment: Mapping[str, Any] | None = None,
    require_committed: bool = True,
    authorization_path: Path = DEFAULT_AUTHORIZATION,
) -> None:
    """Fail closed before any caller is permitted to instantiate a model."""

    validate_execution_authorization(lock)
    if acknowledgement != EXPLICIT_EXECUTION_ACKNOWLEDGEMENT:
        raise PermissionError("exact utility execution acknowledgement required")
    if require_committed:
        _require_head_committed_file(authorization_path)
        for binding in lock["artifacts"].values():
            path = WORKSPACE / binding["repository_path"]
            _require_bound_file_hash(path, str(binding.get("file_sha256") or ""))
            _require_head_committed_file(path)
        for relative, digest in (
            (lock.get("implementation") or {}).get("files", {}).items()
        ):
            path = WORKSPACE / relative
            _require_bound_file_hash(path, str(digest))
            _require_head_committed_file(path)
    observed = (
        observe_exact_environment()
        if observed_environment is None
        else copy.deepcopy(dict(observed_environment))
    )
    expected = (lock.get("runtime") or {}).get("exact_environment")
    if observed != expected:
        raise PermissionError("runtime environment differs from authorization")


def build_execution_authorization_from_paths(
    choice_descriptors: Sequence[Mapping[str, Any]],
    *,
    paths: Mapping[str, Path] = ARTIFACT_PATHS,
) -> dict[str, Any]:
    values = load_bound_inputs(paths=paths)
    bindings = _artifact_bindings_from_paths(values, paths)
    return build_execution_authorization(
        values["v1_utility_authorization"],
        values["completed_decoded_audit"],
        values["v3_analysis_lock"],
        values["all_history_replay_lock"],
        values["frozen_suffix_scan"],
        values["broad_wikitext_quality_lock"],
        choice_descriptors,
        artifact_bindings=bindings,
    )


def write_authorization(path: str | Path, lock: Mapping[str, Any]) -> None:
    validate_execution_authorization(lock)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def load_execution_authorization(
    path: str | Path = DEFAULT_AUTHORIZATION,
) -> dict[str, Any]:
    lock = load_json(path, name="v2 utility execution authorization")
    validate_execution_authorization(lock)
    return lock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.validate:
        parser.error("this source-only module supports only --validate")
    try:
        load_execution_authorization(args.authorization)
    except (OSError, UtilityProtocolError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_PATHS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
    "EXPLICIT_EXECUTION_ACKNOWLEDGEMENT",
    "ForcedChoiceSet",
    "FRESH_REBUILD",
    "METHOD_CONDITIONS",
    "POLICY",
    "PRESENT",
    "PROBES",
    "REPLAY",
    "UtilityProtocolError",
    "artifact_binding",
    "authorize_execution",
    "build_execution_authorization",
    "build_forced_choice_set",
    "freeze_execution_authorization",
    "freeze_forced_choice_sets",
    "infer_answer_type",
    "load_execution_authorization",
    "metric_contract",
    "payload_sha256",
    "validate_choice_descriptors",
    "validate_execution_authorization",
]

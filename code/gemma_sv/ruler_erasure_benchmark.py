"""Predeclared RULER-style multi-needle context-erasure follow-up.

This controlled benchmark is activated only because the frozen natural-QA
smoke had weak admission.  It follows RULER's ``niah_multikey_1`` complexity
(four word keys, one numeric value per key, one query), with one explicit
adaptation: a later conflicting needle for the queried key is the owned span.
Deleting that span should restore the earlier gold needle.  No model outputs
are used to generate or select records.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gemma_sv.rag_benchmark import (
    CorpusNode,
    CounterfactualPassage,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TOKENIZER_REVISION,
    RehydratedRecord,
    TokenizedContext,
    WikiExample,
    _context_descriptor,
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


SCHEMA = "gemma-sv-ruler-erasure-manifest-v1"
SCHEMA_VERSION = 1
RULER_REPOSITORY = "NVIDIA/RULER"
RULER_REVISION = "ab17b7853df4e0a30b78cd5d2b463ac7dff6ee13"
RULER_TASK = "niah_multikey_1"
DEFAULT_RECORDS = 8
DEFAULT_SEED = 271828
DEFAULT_MINIMUM_TOKENS = 1024
DEFAULT_MINIMUM_TOKENS_AFTER_OWNED = 512

_ADJECTIVES = (
    "amber",
    "brisk",
    "cobalt",
    "dappled",
    "ember",
    "frosted",
    "golden",
    "hollow",
    "indigo",
    "juniper",
    "kindled",
    "lunar",
    "misted",
    "northern",
    "opal",
    "pewter",
)
_NOUNS = (
    "anchor",
    "badger",
    "cedar",
    "dolphin",
    "egret",
    "falcon",
    "garden",
    "harbor",
    "island",
    "jasmine",
    "kestrel",
    "lantern",
    "meadow",
    "narwhal",
    "orchard",
    "pavilion",
)
_NOISE_SUBJECTS = (
    "archive",
    "bakery",
    "canal",
    "depot",
    "estuary",
    "foundry",
    "gallery",
    "harvest",
)
_NOISE_OBJECTS = (
    "catalogues maps each spring",
    "opens after the morning bell",
    "records rainfall in a blue ledger",
    "stores cedar boxes on the upper shelf",
    "serves barley soup on market days",
    "repairs lanterns beside the east gate",
    "counts ferry crossings at dusk",
    "keeps spare rope near the stone quay",
)


class ManifestError(ValueError):
    """The controlled manifest is incomplete or has been modified."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _stable_int(seed: int, record_index: int, label: str, modulo: int) -> int:
    digest = hashlib.sha256(
        f"ruler-erasure-v1\0{seed}\0{record_index}\0{label}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)


def _key(seed: int, record_index: int, slot: int) -> str:
    adjective = _ADJECTIVES[
        _stable_int(seed, record_index, f"adjective-{slot}", len(_ADJECTIVES))
    ]
    noun = _NOUNS[
        _stable_int(seed, record_index, f"noun-{slot}", len(_NOUNS))
    ]
    return f"{adjective} {noun}"


def _value(seed: int, record_index: int, slot: int) -> str:
    value = 1_000_000 + _stable_int(
        seed,
        record_index,
        f"value-{slot}",
        9_000_000,
    )
    return f"{value:07d}"


def _needle(key: str, value: str) -> str:
    return f"The special magic number for {key} is {value}."


def _noise_sentence(seed: int, record_index: int, position: int) -> str:
    subject = _NOISE_SUBJECTS[
        _stable_int(
            seed,
            record_index,
            f"noise-subject-{position}",
            len(_NOISE_SUBJECTS),
        )
    ]
    detail = _NOISE_OBJECTS[
        _stable_int(
            seed,
            record_index,
            f"noise-detail-{position}",
            len(_NOISE_OBJECTS),
        )
    ]
    return f"Background note {position}: the {subject} {detail}."


def _assemble_text(
    seed: int,
    record_index: int,
    noise_count: int,
) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]:
    keys = [_key(seed, record_index, slot) for slot in range(4)]
    values = [_value(seed, record_index, slot) for slot in range(5)]
    # The fifth value conflicts with the target's earlier gold value.
    if values[4] == values[0]:
        values[4] = f"{(int(values[4]) % 9_000_000) + 1_000_000:07d}"
    inserts = {
        max(1, int(noise_count * 0.12)): ("gold", _needle(keys[0], values[0])),
        max(2, int(noise_count * 0.28)): ("retain", _needle(keys[1], values[1])),
        max(3, int(noise_count * 0.43)): (
            "owned",
            _needle(keys[0], values[4]),
        ),
        max(4, int(noise_count * 0.61)): ("other-2", _needle(keys[2], values[2])),
        max(5, int(noise_count * 0.79)): ("other-3", _needle(keys[3], values[3])),
    }
    pieces = [
        "Memorize the key-value statements hidden among the background notes.\n\n"
    ]
    spans: dict[str, tuple[int, int]] = {}
    for position in range(noise_count):
        if position in inserts:
            label, sentence = inserts[position]
            start = sum(len(piece) for piece in pieces)
            pieces.append(sentence + "\n")
            spans[label] = (start, start + len(sentence))
        pieces.append(_noise_sentence(seed, record_index, position) + "\n")
    text = "".join(pieces)
    return text, spans["owned"], spans["gold"], spans["retain"]


def generate_record(
    tokenizer: Any,
    record_index: int,
    *,
    seed: int = DEFAULT_SEED,
    minimum_tokens: int = DEFAULT_MINIMUM_TOKENS,
    minimum_tokens_after_owned: int = DEFAULT_MINIMUM_TOKENS_AFTER_OWNED,
) -> dict[str, Any]:
    """Generate one deterministic conflicting-needle record and exact ownership."""

    if record_index < 0 or minimum_tokens < 1 or minimum_tokens_after_owned < 1:
        raise ValueError("record index and token minima must be positive")
    noise_count = 32
    while True:
        text, owned_span, gold_span, retained_span = _assemble_text(
            seed,
            record_index,
            noise_count,
        )
        token_ids, offsets = _tokenizer_payload(tokenizer, text)
        forget = tuple(
            index
            for index, offset in enumerate(offsets)
            if _overlaps(offset, owned_span)
        )
        retained = tuple(
            index
            for index, offset in enumerate(offsets)
            if _overlaps(offset, retained_span) and index not in set(forget)
        )
        if (
            forget
            and retained
            and len(token_ids) >= minimum_tokens
            and len(token_ids) - max(forget) > minimum_tokens_after_owned
        ):
            break
        noise_count += 8
        if noise_count > 4096:
            raise RuntimeError("could not satisfy predeclared RULER geometry")

    keys = [_key(seed, record_index, slot) for slot in range(4)]
    values = [_value(seed, record_index, slot) for slot in range(5)]
    if values[4] == values[0]:
        values[4] = f"{(int(values[4]) % 9_000_000) + 1_000_000:07d}"
    edited_ids = _remove_positions(token_ids, forget)
    context = TokenizedContext(
        original_text=text,
        edited_text=text[: owned_span[0]] + text[owned_span[1] :],
        original_token_ids=tuple(token_ids),
        edited_token_ids=edited_ids,
        offset_mapping=tuple(offsets),
        forget_positions=forget,
        deletion_ranges=_contiguous_ranges(forget),
        retained_positions=retained,
        edited_retained_positions=_shift_positions(retained, forget),
        owned_character_span=owned_span,
        answer_character_span=(
            text.index(values[4], owned_span[0], owned_span[1]),
            text.index(values[4], owned_span[0], owned_span[1]) + len(values[4]),
        ),
        retained_character_span=retained_span,
    )
    question = f"What is the special magic number for {keys[0]}?"
    retained_question = f"What is the special magic number for {keys[1]}?"
    record_id = stable_identifier(
        "ruler",
        RULER_REVISION,
        seed,
        record_index,
    )
    return {
        "record_id": record_id,
        "generator_index": int(record_index),
        "runtime": {
            "question": question,
            "gold_answer": values[0],
            "distractor_answer": values[4],
            "retained_question": retained_question,
            "retained_answer": values[1],
            "context": context,
        },
        "frozen": {
            "question_sha256": text_sha256(question),
            "gold_answer_sha256": text_sha256(values[0]),
            "distractor_answer_sha256": text_sha256(values[4]),
            "retained_question_sha256": text_sha256(retained_question),
            "retained_answer_sha256": text_sha256(values[1]),
            "context": _context_descriptor(context),
            "gold_character_span": list(gold_span),
            "noise_sentences": noise_count,
        },
    }


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "generator_index": record["generator_index"],
        **copy.deepcopy(record["frozen"]),
    }


def build_manifest(
    tokenizer: Any,
    *,
    records: int = DEFAULT_RECORDS,
    seed: int = DEFAULT_SEED,
    minimum_tokens: int = DEFAULT_MINIMUM_TOKENS,
    minimum_tokens_after_owned: int = DEFAULT_MINIMUM_TOKENS_AFTER_OWNED,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
) -> dict[str, Any]:
    """Build the complete source-free, predeclared controlled manifest."""

    if records < 1:
        raise ValueError("records must be positive")
    generated = [
        generate_record(
            tokenizer,
            index,
            seed=seed,
            minimum_tokens=minimum_tokens,
            minimum_tokens_after_owned=minimum_tokens_after_owned,
        )
        for index in range(records)
    ]
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "contains_source_text": False,
        "provenance": {
            "repository": RULER_REPOSITORY,
            "revision": RULER_REVISION,
            "task": RULER_TASK,
            "license": "Apache-2.0",
        },
        "official_task_parameters": {
            "type_haystack": "essay",
            "type_needle_k": "words",
            "type_needle_v": "numbers",
            "num_needle_k": 4,
            "num_needle_v": 1,
            "num_needle_q": 1,
        },
        "erasure_adaptation": {
            "description": (
                "one later conflicting value for the queried key is the owned "
                "span; literal deletion restores the earlier gold needle"
            ),
            "post_hoc_selection": False,
            "natural_qa_trigger": {
                "manifest_integrity_sha256": (
                    "b066a00b9472ba17831b6defeca375bafd8a491053632d1d3f93ae685466d5ee"
                ),
                "primary_target_admission": "1/8",
                "retained_availability": "1/8",
                "joint_admission": "0/8",
            },
        },
        "tokenizer": {
            "model_id": tokenizer_id,
            "revision": tokenizer_revision,
            "offset_mapping_required": True,
        },
        "generation": {
            "seed": int(seed),
            "records": int(records),
            "minimum_tokens": int(minimum_tokens),
            "minimum_tokens_after_owned": int(minimum_tokens_after_owned),
            "fixed_before_model_evaluation": True,
            "model_outputs_used": False,
        },
        "records": [_public_record(record) for record in generated],
    }
    return freeze_manifest(manifest)


def _validate_without_integrity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA or int(
        manifest.get("schema_version", -1)
    ) != SCHEMA_VERSION:
        raise ManifestError("unsupported RULER erasure manifest")
    if manifest.get("contains_source_text") is not False:
        raise ManifestError("controlled manifest must be source-free")
    provenance = manifest.get("provenance") or {}
    if (
        provenance.get("repository") != RULER_REPOSITORY
        or provenance.get("revision") != RULER_REVISION
        or provenance.get("task") != RULER_TASK
    ):
        raise ManifestError("RULER provenance is not pinned")
    generation = manifest.get("generation") or {}
    if (
        generation.get("fixed_before_model_evaluation") is not True
        or generation.get("model_outputs_used") is not False
    ):
        raise ManifestError("controlled records were not predeclared")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != int(
        generation.get("records", -1)
    ):
        raise ManifestError("controlled record count is inconsistent")
    ids = [record.get("record_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ManifestError("controlled record IDs are not unique")
    for record in records:
        context = record.get("context") or {}
        ownership = context.get("ownership") or {}
        positions = ownership.get("forget_positions")
        if not isinstance(positions, list) or not positions:
            raise ManifestError("controlled token ownership is missing")
        if int(context["original_token_count"]) - len(positions) != int(
            context["edited_token_count"]
        ):
            raise ManifestError("controlled literal edit count is inconsistent")
        if int(context["tokens_after_owned_span"]) <= int(
            generation["minimum_tokens_after_owned"]
        ):
            raise ManifestError("controlled owned span is inside the local window")


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
        raise ManifestError("controlled manifest integrity is missing")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    if (
        integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
    ):
        raise ManifestError("controlled manifest integrity mismatch")


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[dict[str, Any], ...]:
    """Regenerate runtime text and reject any drift from the frozen descriptors."""

    validate_manifest(manifest)
    generation = manifest["generation"]
    result = []
    for stored in manifest["records"]:
        generated = generate_record(
            tokenizer,
            int(stored["generator_index"]),
            seed=int(generation["seed"]),
            minimum_tokens=int(generation["minimum_tokens"]),
            minimum_tokens_after_owned=int(
                generation["minimum_tokens_after_owned"]
            ),
        )
        if generated["record_id"] != stored["record_id"]:
            raise ManifestError("rehydrated controlled record ID differs")
        if generated["frozen"] != {
            key: value
            for key, value in stored.items()
            if key not in {"record_id", "generator_index"}
        }:
            raise ManifestError("rehydrated controlled record differs")
        result.append(generated)
    return tuple(result)


def as_context_erasure_records(
    manifest: Mapping[str, Any],
    tokenizer: Any,
) -> tuple[RehydratedRecord, ...]:
    """Adapt controlled records to the shared model-state deletion evaluator."""

    generated = rehydrate_manifest(manifest, tokenizer)
    result = []
    for record in generated:
        runtime = record["runtime"]
        retained_node = CorpusNode(
            node_id=stable_identifier("ruler-retained", record["record_id"]),
            document_id=stable_identifier("ruler-doc", record["record_id"]),
            source_example_id=record["record_id"],
            paragraph_index=0,
            chunk_index=0,
            title=runtime["retained_answer"],
            text="",
        )
        counterfactual = CounterfactualPassage(
            title="RULER conflicting needle",
            text="",
            distractor_answer=runtime["distractor_answer"],
            answer_strategy="predeclared_conflicting_numeric_needle",
            owned_start=0,
            owned_end=0,
            answer_start=0,
            answer_end=0,
        )
        result.append(
            RehydratedRecord(
                record_id=record["record_id"],
                example=WikiExample(
                    example_id=record["record_id"],
                    question=runtime["question"],
                    answer=runtime["gold_answer"],
                    question_type=RULER_TASK,
                    passages=(),
                ),
                retrieval_hits=(),
                retained_node=retained_node,
                counterfactual=counterfactual,
                context=runtime["context"],
                retained_question=runtime["retained_question"],
                retained_answer=runtime["retained_answer"],
            )
        )
    return tuple(result)


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--minimum-tokens", type=int, default=DEFAULT_MINIMUM_TOKENS)
    parser.add_argument(
        "--minimum-tokens-after-owned",
        type=int,
        default=DEFAULT_MINIMUM_TOKENS_AFTER_OWNED,
    )
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_rag/ruler_multikey_erasure_v1.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")
    tokenizer = load_offset_tokenizer()
    manifest = build_manifest(
        tokenizer,
        records=args.records,
        seed=args.seed,
        minimum_tokens=args.minimum_tokens,
        minimum_tokens_after_owned=args.minimum_tokens_after_owned,
    )
    write_manifest(output, manifest)
    write_environment_lock(output.parent / "environment-lock.txt")
    print(f"wrote {len(manifest['records'])} controlled records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

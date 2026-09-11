"""Deterministically correct geometry for the already-scored v1 chat cohort.

This protocol does not select a new cohort.  It preserves the ordered 16 core
records, target/retained identities, and probes from the committed v1
confirmation manifest.  Source-only official rounds may be added before the
core context to make fixed-C feasible or after it to move the owned span at
least 1,152 tokens behind the readout point.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from gemma_sv import longmemeval_chat_benchmark as chat_v1
from gemma_sv import longmemeval_deletion_benchmark as base
from gemma_sv.demo_server.certificate import fixed_c_feasibility
from gemma_sv.rag_benchmark import TokenizedContext


SCHEMA = "gemma-sv-longmemeval-chat-geometry-fix-v1"
SCHEMA_VERSION = 1
POLICY_SCHEMA = "gemma-sv-longmemeval-chat-geometry-fix-policy-v1"
CENSUS_SCHEMA = "gemma-sv-longmemeval-chat-geometry-fix-census-v1"
BENCHMARK_LABEL = "LongMemEval v1 fixed-cohort chat geometry correction"

WINDOW = 1024
MINIMUM_TOKENS_AFTER_OWNED = 1152
MAXIMUM_CONTEXT_TOKENS = 8192
NU = 0.7
CHUNK = 128
SOLVER_SEED = 0
EXTENSION_SEED = 20_260_823
MINIMUM_TARGET_PRESENT_MINUS_RAW_FULL_SEQUENCE_MEAN_LOGPROB_NATS = 0.05
MAXIMUM_TARGET_PRESENT_FIRST_TOKEN_RANK = 10
MAXIMUM_RETAINED_FIRST_TOKEN_RANK_IN_PRESENT_AND_RAW = 10

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
DEFAULT_CORE_MANIFEST_PATH = (
    PACKAGE / "benchmarks" / "longmemeval_chat_confirmation_v1.json"
)
PINNED_CORE_FILE_SHA256 = (
    "7e9f484c4eee8cceecec79bfc0721e1cfc7100092d29f5b29d1e96eca478bae1"
)
PINNED_CORE_INTEGRITY_SHA256 = (
    "48eb12371e677bc18732f900049dcdfc5791f31344a1f4dfbcfcf06bcfe07ce5"
)
PINNED_FULL_DESCRIPTORS_SHA256 = (
    "4294ca8f7aaf22fc0214a3d9ed6860da83ad1b947337a94aff7f742e5084c839"
)
PINNED_RECORDS = 16
PINNED_MANIFEST_INTEGRITY_SHA256 = (
    "04fb39a23cbe5e31ed652c0ff8d412df1ab2072999760fba81f7de95dee1a4db"
)
PINNED_POLICY_LOCK_SHA256 = (
    "3ce2bac27c60872788418d37149acb15790327b05abed8e5863c274453282b1c"
)
PINNED_CENSUS_INTEGRITY_SHA256 = (
    "c01eacc10c21cbbc5b7488809d946b94f7013b78b2996f40f36cb5f322da2174"
)

ManifestInput = Mapping[str, Any] | str | Path
ManifestError = base.ManifestError


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(source: ManifestInput) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source)), None
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError("core LongMemEval manifest is not an object")
    return payload, path


@contextmanager
def _geometry() -> Iterator[None]:
    overrides = (
        (chat_v1, "MAXIMUM_CONTEXT_TOKENS", MAXIMUM_CONTEXT_TOKENS),
        (chat_v1, "MINIMUM_TOKENS_AFTER_OWNED", 512),
        (base, "RUNTIME_CONTEXT_CEILING", MAXIMUM_CONTEXT_TOKENS),
    )
    previous = [(module, name, getattr(module, name)) for module, name, _ in overrides]
    try:
        for module, name, value in overrides:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in previous:
            setattr(module, name, value)


@dataclass(frozen=True)
class CorrectedRecord:
    runtime: chat_v1.RehydratedChatRecord
    prefix_extensions: tuple[chat_v1.SessionFragment, ...]
    suffix_extensions: tuple[chat_v1.SessionFragment, ...]
    public_record: Mapping[str, Any]


def _core_spec(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline": copy.deepcopy(dict(record.get("baseline") or {})),
        "target": copy.deepcopy(dict(record.get("target") or {})),
        "retained_probe": copy.deepcopy(
            dict(record.get("retained_probe") or {})
        ),
        "tail_session_references": copy.deepcopy(
            list(record.get("tail_session_references") or ())
        ),
    }


def _core_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    core_spec = _core_spec(record)
    target = copy.deepcopy(dict(record.get("target") or {}))
    retained = copy.deepcopy(dict(record.get("retained_probe") or {}))
    probes = copy.deepcopy(list(record.get("probes") or ()))
    return {
        "record_id": str(record.get("record_id") or ""),
        "target_source_id": str(target.get("source_id") or ""),
        "target_question_id": str(target.get("question_id") or ""),
        "target_descriptor_keys": sorted(target),
        "target_descriptor_sha256": _payload_sha256(target),
        "retained_source_id": str(retained.get("source_id") or ""),
        "retained_question_id": str(retained.get("question_id") or ""),
        "retained_descriptor_keys": sorted(retained),
        "retained_descriptor_sha256": _payload_sha256(retained),
        "probe_descriptor_count": len(probes),
        "probe_descriptors_sha256": _payload_sha256(probes),
        "core_spec_keys": sorted(core_spec),
        "core_spec_sha256": _payload_sha256(core_spec),
    }


def _source_hashes(source_ids: Iterable[str]) -> list[str]:
    return sorted(base.text_sha256(value) for value in set(source_ids))


def _admission_policy() -> dict[str, Any]:
    return {
        "target_metric": (
            "present_minus_fresh_raw_omission_full_sequence_mean_logprob_nats"
        ),
        "minimum_target_present_minus_raw_full_sequence_mean_logprob_nats": (
            MINIMUM_TARGET_PRESENT_MINUS_RAW_FULL_SEQUENCE_MEAN_LOGPROB_NATS
        ),
        "maximum_target_present_first_token_rank": (
            MAXIMUM_TARGET_PRESENT_FIRST_TOKEN_RANK
        ),
        "maximum_retained_first_token_rank_in_present_and_raw": (
            MAXIMUM_RETAINED_FIRST_TOKEN_RANK_IN_PRESENT_AND_RAW
        ),
        "target_full_sequence_scoring_required": True,
        "threshold_overrides_allowed": False,
    }


def _evaluation_contract() -> dict[str, Any]:
    return {
        "core_cohort_reselected": False,
        "model_scoring_performed": False,
        "raw_omission_required": True,
        "fixed_c_required": True,
        "minimum_tokens_strictly_after_owned": (
            MINIMUM_TOKENS_AFTER_OWNED
        ),
        "selected_span_inside_local_window_permitted": False,
        "no_record_question_answer_retained_or_probe_replacement": True,
        "admission_policy": _admission_policy(),
    }


def _extension_text(
    fragments: Sequence[chat_v1.SessionFragment],
) -> str:
    if not fragments:
        return ""
    rendered, _ = chat_v1._render_fragments(fragments)
    if not rendered.startswith("<bos>"):
        raise ManifestError("registered extension render omitted BOS")
    return rendered[len("<bos>") :]


def _augment_runtime(
    runtime: chat_v1.RehydratedChatRecord,
    prefix_extensions: Sequence[chat_v1.SessionFragment],
    suffix_extensions: Sequence[chat_v1.SessionFragment],
    tokenizer: Any,
) -> chat_v1.RehydratedChatRecord:
    prefix = _extension_text(prefix_extensions)
    suffix = _extension_text(suffix_extensions)
    old = runtime.context
    if not old.original_text.startswith("<bos>") or not old.edited_text.startswith(
        "<bos>"
    ):
        raise ManifestError("core registered context omitted BOS")
    original_text = (
        "<bos>" + prefix + old.original_text[len("<bos>") :] + suffix
    )
    edited_text = "<bos>" + prefix + old.edited_text[len("<bos>") :] + suffix
    character_shift = len(prefix)
    owned_span = (
        old.owned_character_span[0] + character_shift,
        old.owned_character_span[1] + character_shift,
    )
    retained_span = (
        old.retained_character_span[0] + character_shift,
        old.retained_character_span[1] + character_shift,
    )
    answer_span = (
        old.answer_character_span[0] + character_shift,
        old.answer_character_span[1] + character_shift,
    )
    original_ids, offsets = base._tokenizer_payload(tokenizer, original_text)
    edited_ids, _ = base._tokenizer_payload(tokenizer, edited_text)
    forget_positions = chat_v1._positions_for_span(offsets, owned_span)
    if not forget_positions:
        raise ManifestError("corrected owned span maps to no model tokens")
    literal_omission = (
        original_text[: owned_span[0]] + original_text[owned_span[1] :]
    )
    if literal_omission != edited_text:
        raise ManifestError("corrected raw omission is not a literal rerender")
    if base._remove_positions(original_ids, forget_positions) != tuple(
        edited_ids
    ):
        raise ManifestError("corrected raw omission token drop differs")
    retained_positions = tuple(
        position
        for position in chat_v1._positions_for_span(offsets, retained_span)
        if position not in set(forget_positions)
    )
    edited_retained_positions = base._shift_positions(
        retained_positions,
        forget_positions,
    )
    if tuple(original_ids[position] for position in retained_positions) != tuple(
        edited_ids[position] for position in edited_retained_positions
    ):
        raise ManifestError("corrected geometry changes retained token rows")
    _, feasibility = fixed_c_feasibility(
        len(original_ids),
        forget_positions,
        nu=NU,
        chunk=CHUNK,
        per_boundary_box=True,
    )
    context = TokenizedContext(
        original_text=original_text,
        edited_text=edited_text,
        original_token_ids=tuple(original_ids),
        edited_token_ids=tuple(edited_ids),
        offset_mapping=tuple(offsets),
        forget_positions=tuple(forget_positions),
        deletion_ranges=base._contiguous_ranges(forget_positions),
        retained_positions=retained_positions,
        edited_retained_positions=edited_retained_positions,
        owned_character_span=owned_span,
        answer_character_span=answer_span,
        retained_character_span=retained_span,
    )
    return replace(
        runtime,
        context=context,
        raw_omitted_token_ids=tuple(edited_ids),
        fixed_c_diagnostics=tuple(dict(item) for item in feasibility),
    )


def _extension_candidates(
    examples: Sequence[base.LongMemEvalExample],
    tokenizer: Any,
    *,
    excluded_source_ids: set[str],
    forbidden_values: Sequence[str],
) -> tuple[chat_v1.SessionFragment, ...]:
    forbidden = tuple(
        value.strip().casefold()
        for value in forbidden_values
        if value.strip()
    )
    candidates: list[
        tuple[int, str, str, str, chat_v1.SessionFragment]
    ] = []
    for example in examples:
        if example.source_id in excluded_source_ids:
            continue
        for session in base.evidence_sessions(example):
            try:
                fragment = chat_v1._evidence_fragment(example, session)
                rendered = _extension_text((fragment,))
            except (ManifestError, ValueError, IndexError):
                continue
            folded = rendered.casefold()
            if any(value in folded for value in forbidden):
                continue
            token_count = len(
                chat_v1._token_ids_no_special(tokenizer, rendered)
            )
            candidates.append(
                (
                    token_count,
                    base.stable_identifier(
                        "longmemeval-geometry-extension-order",
                        str(EXTENSION_SEED),
                        example.source_id,
                        session.session_id,
                    ),
                    example.source_id,
                    session.session_id,
                    fragment,
                )
            )
    candidates.sort(key=lambda item: item[:4])
    return tuple(item[4] for item in candidates)


def _tokens_after(runtime: chat_v1.RehydratedChatRecord) -> int:
    return (
        len(runtime.context.original_token_ids)
        - max(runtime.context.forget_positions)
        - 1
    )


def _fixed_c_feasible(runtime: chat_v1.RehydratedChatRecord) -> bool:
    return all(
        bool(item.get("feasible")) for item in runtime.fixed_c_diagnostics
    )


def _fit_record(
    stored: Mapping[str, Any],
    examples: Sequence[base.LongMemEvalExample],
    examples_by_source: Mapping[str, base.LongMemEvalExample],
    tokenizer: Any,
    *,
    excluded_extension_sources: set[str],
) -> CorrectedRecord:
    spec = _core_spec(stored)
    with _geometry():
        runtime = chat_v1._prepare_record(
            spec,
            examples_by_source,
            tokenizer,
            partition=chat_v1.CONFIRMATION_PARTITION,
        )
    runtime = replace(runtime, record_id=str(stored["record_id"]))
    if [chat_v1._probe_descriptor(probe) for probe in runtime.probes] != list(
        stored.get("probes") or ()
    ):
        raise ManifestError("geometry correction changed frozen probes")
    target = runtime.target
    retained = runtime.retained
    candidates = _extension_candidates(
        examples,
        tokenizer,
        excluded_source_ids=excluded_extension_sources,
        forbidden_values=(
            target.answer,
            retained.answer,
            target.question,
            retained.question,
        ),
    )
    prefix: list[chat_v1.SessionFragment] = []
    suffix: list[chat_v1.SessionFragment] = []
    used: set[tuple[str, str]] = set()
    corrected = _augment_runtime(runtime, prefix, suffix, tokenizer)
    for fragment in candidates:
        if _tokens_after(corrected) >= MINIMUM_TOKENS_AFTER_OWNED:
            break
        key = (fragment.source_id, fragment.session.session_id)
        if key in used:
            continue
        candidate = _augment_runtime(
            runtime,
            prefix,
            [*suffix, fragment],
            tokenizer,
        )
        if (
            len(candidate.context.original_token_ids)
            + candidate.query_token_reserve
            <= MAXIMUM_CONTEXT_TOKENS
        ):
            suffix.append(fragment)
            used.add(key)
            corrected = candidate
    for fragment in candidates:
        if _fixed_c_feasible(corrected):
            break
        key = (fragment.source_id, fragment.session.session_id)
        if key in used:
            continue
        candidate = _augment_runtime(
            runtime,
            [*prefix, fragment],
            suffix,
            tokenizer,
        )
        if (
            len(candidate.context.original_token_ids)
            + candidate.query_token_reserve
            <= MAXIMUM_CONTEXT_TOKENS
        ):
            prefix.append(fragment)
            used.add(key)
            corrected = candidate
    if _tokens_after(corrected) < MINIMUM_TOKENS_AFTER_OWNED:
        raise ManifestError("source-only extension cannot restore suffix geometry")
    if not _fixed_c_feasible(corrected):
        raise ManifestError("source-only extension cannot make fixed-C feasible")
    if (
        len(corrected.context.original_token_ids)
        + corrected.query_token_reserve
        > MAXIMUM_CONTEXT_TOKENS
    ):
        raise ManifestError("geometry-corrected context exceeds 8,192 tokens")
    public = _public_record(stored, corrected, prefix, suffix, tokenizer)
    return CorrectedRecord(
        runtime=corrected,
        prefix_extensions=tuple(prefix),
        suffix_extensions=tuple(suffix),
        public_record=public,
    )


def _extension_descriptor(
    fragment: chat_v1.SessionFragment,
    tokenizer: Any,
) -> dict[str, Any]:
    descriptor = chat_v1._fragment_descriptor(fragment)
    registered = _extension_text((fragment,))
    token_ids = chat_v1._token_ids_no_special(tokenizer, registered)
    descriptor.update(
        {
            "turns_sha256": _payload_sha256(
                descriptor["source_turn_sha256"]
            ),
            "official_turn_count": len(fragment.turns),
            "registered_token_count": len(token_ids),
            "registered_token_ids_sha256": base.token_ids_sha256(token_ids),
            "purpose": "geometry_only",
        }
    )
    return descriptor


def _public_record(
    stored: Mapping[str, Any],
    runtime: chat_v1.RehydratedChatRecord,
    prefix: Sequence[chat_v1.SessionFragment],
    suffix: Sequence[chat_v1.SessionFragment],
    tokenizer: Any,
) -> dict[str, Any]:
    with _geometry():
        context = chat_v1._context_descriptor(runtime)
    context["runtime_context_ceiling"] = MAXIMUM_CONTEXT_TOKENS
    context["tokens_strictly_after_owned"] = _tokens_after(runtime)
    context["local_window_safety"] = {
        "true_local_window_tokens": WINDOW,
        "minimum_tokens_strictly_after_owned": (
            MINIMUM_TOKENS_AFTER_OWNED
        ),
        "selected_span_inside_local_window": False,
    }
    fixed = context["fixed_c_reference"]
    fixed["all_affected_boundaries_feasible"] = True
    fixed["infeasible_reference_policy"] = "not_applicable"
    return {
        "record_id": str(stored["record_id"]),
        "core_binding": _core_binding(stored),
        "core_spec": _core_spec(stored),
        "baseline": copy.deepcopy(dict(stored["baseline"])),
        "target": copy.deepcopy(dict(stored["target"])),
        "retained_probe": copy.deepcopy(dict(stored["retained_probe"])),
        "tail_session_references": copy.deepcopy(
            list(stored["tail_session_references"])
        ),
        "probes": copy.deepcopy(list(stored["probes"])),
        "chat_serialization": chat_v1._chat_serialization(),
        "geometry_extensions": {
            "selection_uses_model_outputs": False,
            "prefix_fixed_c_rounds": [
                _extension_descriptor(item, tokenizer) for item in prefix
            ],
            "suffix_distance_rounds": [
                _extension_descriptor(item, tokenizer) for item in suffix
            ],
        },
        "context": context,
    }


def _extension_source_ids(records: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for record in records:
        extensions = record.get("geometry_extensions") or {}
        for key in ("prefix_fixed_c_rounds", "suffix_distance_rounds"):
            result.update(
                str(item.get("source_id") or "")
                for item in extensions.get(key) or ()
            )
    result.discard("")
    return result


def _ordered_extension_bindings(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for record in records:
        extensions = record.get("geometry_extensions") or {}
        for placement, key in (
            ("prefix_fixed_c", "prefix_fixed_c_rounds"),
            ("suffix_distance", "suffix_distance_rounds"),
        ):
            for ordinal, descriptor in enumerate(extensions.get(key) or ()):
                bindings.append(
                    {
                        "record_id": str(record.get("record_id") or ""),
                        "placement": placement,
                        "ordinal": ordinal,
                        "descriptor_sha256": _payload_sha256(descriptor),
                    }
                )
    return bindings


def _policy(
    *,
    core_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    source_inventory: Mapping[str, Any],
    core_bindings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    bindings = (
        [copy.deepcopy(dict(item)) for item in core_bindings]
        if core_bindings is not None
        else [_core_binding(record) for record in core_manifest["records"]]
    )
    extension_ids = _extension_source_ids(records)
    extension_bindings = _ordered_extension_bindings(records)
    extension_source_hashes = _source_hashes(extension_ids)
    geometry = [record["context"] for record in records]
    policy = {
        "schema": POLICY_SCHEMA,
        "status": "frozen-before-geometry-corrected-model-scoring",
        "contains_source_text": False,
        "selection_uses_model_outputs": False,
        "core_cohort": {
            "source_manifest_schema": core_manifest["schema"],
            "source_manifest_integrity_sha256": core_manifest["integrity"][
                "sha256"
            ],
            "source_manifest_file_sha256": PINNED_CORE_FILE_SHA256,
            "ordered_records": len(bindings),
            "ordered_core_bindings_sha256": _payload_sha256(bindings),
            "record_question_target_retained_replacement": False,
            "probe_replacement": False,
        },
        "disclosure": {
            "original_admission_model_scored": True,
            "geometry_correction_after_local_window_implementation_failure": True,
            "geometry_correction_before_valid_deletion_method_outcomes": True,
            "original_admission_not_reinterpreted_as_valid_deletion_outcome": True,
        },
        "geometry_contract": {
            "window": WINDOW,
            "minimum_tokens_strictly_after_owned": (
                MINIMUM_TOKENS_AFTER_OWNED
            ),
            "maximum_context_tokens": MAXIMUM_CONTEXT_TOKENS,
            "nu": NU,
            "chunk": CHUNK,
            "solver_seed": SOLVER_SEED,
            "fixed_c_required_for_every_record": True,
            "complete_official_round_extensions_only": True,
            "extension_selection_source_only": True,
            "extension_seed": EXTENSION_SEED,
        },
        "admission_policy": _admission_policy(),
        "source_inventory": copy.deepcopy(dict(source_inventory)),
        "extensions": {
            "source_count": len(extension_ids),
            "source_id_sha256": extension_source_hashes,
            "source_ids_sha256": _payload_sha256(extension_source_hashes),
            "ordered_descriptor_count": len(extension_bindings),
            "ordered_descriptor_bindings": extension_bindings,
            "ordered_descriptor_bindings_sha256": _payload_sha256(
                extension_bindings
            ),
            "prefix_round_count": sum(
                len(
                    record["geometry_extensions"][
                        "prefix_fixed_c_rounds"
                    ]
                )
                for record in records
            ),
            "suffix_round_count": sum(
                len(
                    record["geometry_extensions"][
                        "suffix_distance_rounds"
                    ]
                )
                for record in records
            ),
        },
        "geometry": {
            "minimum_tokens_strictly_after_owned": min(
                int(item["tokens_strictly_after_owned"]) for item in geometry
            ),
            "maximum_runtime_total_token_bound": max(
                int(item["runtime_total_token_bound"]) for item in geometry
            ),
            "fixed_c_feasible_records": sum(
                bool(
                    item["fixed_c_reference"][
                        "all_affected_boundaries_feasible"
                    ]
                )
                for item in geometry
            ),
        },
    }
    policy["lock_sha256"] = _payload_sha256(policy)
    return policy


def _source_inventory(
    examples: Sequence[base.LongMemEvalExample],
) -> dict[str, Any]:
    descriptors = chat_v1._source_descriptors(examples)
    return {
        "dataset_id": base.DATASET_ID,
        "dataset_revision": base.DATASET_REVISION,
        "source_artifact_sha256": base.DATASET_ARTIFACT_SHA256,
        "full_row_count": len(descriptors),
        "full_descriptors_sha256": _payload_sha256(descriptors),
    }


def _manifest_body(
    records: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "benchmark_label": BENCHMARK_LABEL,
        "contains_source_text": False,
        "partition": "confirmation",
        "policy_lock": copy.deepcopy(dict(policy)),
        "source_inventory": copy.deepcopy(policy["source_inventory"]),
        "model": {
            "model_id": chat_v1.MODEL_ID,
            "revision": chat_v1.MODEL_REVISION,
            "adapter": None,
            "arm": "it_native_chat_training_free_graft",
        },
        "tokenizer": {
            "model_id": chat_v1.CHAT_TOKENIZER_ID,
            "revision": chat_v1.CHAT_TOKENIZER_REVISION,
            "fast_offset_mapping_required": True,
        },
        "chat_serialization": chat_v1._chat_serialization(),
        "admission_policy": _admission_policy(),
        "evaluation_contract": _evaluation_contract(),
        "disclosure": copy.deepcopy(policy["disclosure"]),
        "records": [copy.deepcopy(dict(record)) for record in records],
    }


def build_manifest(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    core_manifest: ManifestInput = DEFAULT_CORE_MANIFEST_PATH,
    require_pinned: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    core, core_path = _load_json(core_manifest)
    chat_v1.validate_manifest(core)
    if core.get("partition") != chat_v1.CONFIRMATION_PARTITION:
        raise ManifestError("core manifest is not v1 confirmation")
    if require_pinned and (
        core_path != DEFAULT_CORE_MANIFEST_PATH
        or _file_sha256(core_path) != PINNED_CORE_FILE_SHA256
        or core["integrity"]["sha256"] != PINNED_CORE_INTEGRITY_SHA256
        or len(core["records"]) != PINNED_RECORDS
    ):
        raise ManifestError("pinned core v1 cohort drifted")
    examples = base.extract_longmemeval_examples(rows)
    source_inventory = _source_inventory(examples)
    if require_pinned and (
        source_inventory["full_row_count"] != 500
        or source_inventory["full_descriptors_sha256"]
        != PINNED_FULL_DESCRIPTORS_SHA256
    ):
        raise ManifestError("pinned oracle inventory drifted")
    examples_by_source = {item.source_id: item for item in examples}
    core_role_ids = chat_v1.all_role_source_ids((core,))
    corrected = [
        _fit_record(
            stored,
            examples,
            examples_by_source,
            tokenizer,
            excluded_extension_sources=core_role_ids,
        )
        for stored in core["records"]
    ]
    records = [dict(item.public_record) for item in corrected]
    policy = _policy(
        core_manifest=core,
        records=records,
        source_inventory=source_inventory,
    )
    manifest = freeze_manifest(_manifest_body(records, policy))
    census = _census_from_manifest(manifest)
    census["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(census),
    }
    if require_pinned and (
        manifest["integrity"]["sha256"]
        != PINNED_MANIFEST_INTEGRITY_SHA256
        or manifest["policy_lock"]["lock_sha256"]
        != PINNED_POLICY_LOCK_SHA256
        or census["integrity"]["sha256"]
        != PINNED_CENSUS_INTEGRITY_SHA256
    ):
        raise ManifestError("pinned geometry-corrected freeze drifted")
    validate_census(census, manifest)
    return manifest, census


_EXTENSION_DESCRIPTOR_KEYS = frozenset(
    {
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
        "turns_sha256",
        "official_turn_count",
        "registered_token_count",
        "registered_token_ids_sha256",
        "complete_user_assistant_rounds",
        "purpose",
    }
)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _validate_extension_descriptor(descriptor: Any) -> None:
    if not isinstance(descriptor, Mapping) or set(descriptor) != set(
        _EXTENSION_DESCRIPTOR_KEYS
    ):
        raise ManifestError("geometry extension descriptor fields drifted")
    turn_indices = descriptor["official_turn_indices"]
    answer_indices = descriptor["answer_bearing_turn_indices"]
    roles = descriptor["official_role_sequence"]
    turn_hashes = descriptor["source_turn_sha256"]
    turn_count = descriptor["official_turn_count"]
    if (
        not str(descriptor["source_id"])
        or not str(descriptor["question_id"])
        or not str(descriptor["session_id"])
        or not _is_sha256(descriptor["source_session_sha256"])
        or not _is_sha256(descriptor["date_text_sha256"])
        or not _is_sha256(descriptor["registered_token_ids_sha256"])
        or descriptor["timestamp"] is not None
        and not isinstance(descriptor["timestamp"], str)
        or not isinstance(turn_indices, list)
        or not turn_indices
        or len(turn_indices) % 2
        or any(not isinstance(index, int) for index in turn_indices)
        or turn_indices != sorted(set(turn_indices))
        or not isinstance(answer_indices, list)
        or any(not isinstance(index, int) for index in answer_indices)
        or not set(answer_indices).issubset(turn_indices)
        or not isinstance(roles, list)
        or not isinstance(turn_hashes, list)
        or turn_count != len(turn_indices)
        or turn_count != len(roles)
        or turn_count != len(turn_hashes)
        or any(not _is_sha256(value) for value in turn_hashes)
        or descriptor["turns_sha256"] != _payload_sha256(turn_hashes)
        or not isinstance(descriptor["registered_token_count"], int)
        or descriptor["registered_token_count"] < 1
        or descriptor["complete_user_assistant_rounds"] is not True
        or descriptor["purpose"] != "geometry_only"
    ):
        raise ManifestError("geometry extension descriptor content drifted")
    for index in range(0, len(roles), 2):
        if roles[index] != "user" or roles[index + 1] != "assistant":
            raise ManifestError("geometry extension is not complete chat rounds")


def _validate_without_integrity(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("benchmark_label") != BENCHMARK_LABEL
        or manifest.get("contains_source_text") is not False
        or manifest.get("partition") != "confirmation"
    ):
        raise ManifestError("unsupported geometry-corrected manifest")
    leaked = chat_v1._FORBIDDEN_SOURCE_KEYS.intersection(
        chat_v1._walk_keys(manifest)
    )
    if leaked:
        raise ManifestError("geometry-corrected manifest contains source text")
    policy = manifest.get("policy_lock")
    if not isinstance(policy, Mapping):
        raise ManifestError("geometry correction policy is missing")
    lock = copy.deepcopy(dict(policy))
    observed_lock = lock.pop("lock_sha256", None)
    if (
        observed_lock != _payload_sha256(lock)
        or policy.get("schema") != POLICY_SCHEMA
        or policy.get("selection_uses_model_outputs") is not False
        or policy.get("core_cohort", {}).get(
            "record_question_target_retained_replacement"
        )
        is not False
        or policy.get("core_cohort", {}).get("probe_replacement") is not False
        or policy.get("admission_policy") != _admission_policy()
        or policy.get("disclosure")
        != {
            "original_admission_model_scored": True,
            "geometry_correction_after_local_window_implementation_failure": True,
            "geometry_correction_before_valid_deletion_method_outcomes": True,
            "original_admission_not_reinterpreted_as_valid_deletion_outcome": True,
        }
    ):
        raise ManifestError("geometry correction policy drifted")
    if (
        manifest.get("admission_policy") != _admission_policy()
        or manifest.get("evaluation_contract") != _evaluation_contract()
    ):
        raise ManifestError("geometry correction admission policy drifted")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ManifestError("geometry-corrected records are missing")
    bindings = []
    record_ids: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id") or "")
        if not record_id or record_id in record_ids:
            raise ManifestError("geometry-corrected record IDs are invalid")
        record_ids.add(record_id)
        binding = record.get("core_binding") or {}
        expected_binding = _core_binding(record)
        if binding != expected_binding:
            raise ManifestError("complete core descriptor binding drifted")
        bindings.append(copy.deepcopy(dict(binding)))
        target = record.get("target") or {}
        retained = record.get("retained_probe") or {}
        core_spec = record.get("core_spec") or {}
        if (
            core_spec.get("baseline") != record.get("baseline")
            or core_spec.get("tail_session_references")
            != record.get("tail_session_references")
            or core_spec.get("target") != target
            or core_spec.get("retained_probe") != retained
        ):
            raise ManifestError("core spec target/retained descriptor drifted")
        context = record.get("context") or {}
        if (
            int(context.get("tokens_strictly_after_owned", -1))
            < MINIMUM_TOKENS_AFTER_OWNED
            or int(context.get("runtime_total_token_bound", -1))
            > MAXIMUM_CONTEXT_TOKENS
            or context.get("local_window_safety", {}).get(
                "selected_span_inside_local_window"
            )
            is not False
            or context.get("fixed_c_reference", {}).get(
                "all_affected_boundaries_feasible"
            )
            is not True
        ):
            raise ManifestError("geometry correction is unsafe")
        extensions = record.get("geometry_extensions") or {}
        if (
            not isinstance(extensions, Mapping)
            or set(extensions)
            != {
                "selection_uses_model_outputs",
                "prefix_fixed_c_rounds",
                "suffix_distance_rounds",
            }
            or extensions.get("selection_uses_model_outputs") is not False
        ):
            raise ManifestError("extension selection used model output")
        for key in ("prefix_fixed_c_rounds", "suffix_distance_rounds"):
            values = extensions.get(key)
            if not isinstance(values, list):
                raise ManifestError("geometry extension list is invalid")
            for extension in values:
                _validate_extension_descriptor(extension)
    core_policy = policy["core_cohort"]
    if (
        core_policy.get("ordered_records") != len(records)
        or core_policy.get("ordered_core_bindings_sha256")
        != _payload_sha256(bindings)
    ):
        raise ManifestError("ordered core cohort binding drifted")
    expected_policy = _policy(
        core_manifest={
            "schema": core_policy["source_manifest_schema"],
            "integrity": {
                "sha256": core_policy["source_manifest_integrity_sha256"]
            },
            "records": [
                {
                    "record_id": binding["record_id"],
                }
                for binding in bindings
            ],
        },
        records=records,
        source_inventory=manifest["source_inventory"],
        core_bindings=bindings,
    )
    if expected_policy != policy:
        raise ManifestError("derived geometry correction policy drifted")


def freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(manifest))
    frozen.pop("integrity", None)
    for record in frozen.get("records", ()):
        record.pop("record_integrity", None)
    _validate_without_integrity(frozen)
    for record in frozen["records"]:
        record["record_integrity"] = {
            "algorithm": "sha256",
            "sha256": _payload_sha256(record),
        }
    frozen["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(frozen),
    }
    validate_manifest(frozen)
    return frozen


def validate_manifest(manifest: Mapping[str, Any]) -> None:
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
            raise ManifestError("geometry-corrected record integrity mismatch")
    integrity = manifest.get("integrity")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
    ):
        raise ManifestError("geometry-corrected manifest integrity mismatch")


def _census_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = manifest["records"]
    extensions = manifest["policy_lock"]["extensions"]
    return {
        "schema": CENSUS_SCHEMA,
        "contains_source_text": False,
        "selection_uses_model_outputs": False,
        "core_records": len(records),
        "ordered_core_record_ids": [
            str(record["record_id"]) for record in records
        ],
        "ordered_core_bindings_sha256": manifest["policy_lock"][
            "core_cohort"
        ]["ordered_core_bindings_sha256"],
        "extension_source_count": extensions["source_count"],
        "extension_source_id_sha256": copy.deepcopy(
            extensions["source_id_sha256"]
        ),
        "extension_source_ids_sha256": extensions["source_ids_sha256"],
        "ordered_extension_descriptor_count": extensions[
            "ordered_descriptor_count"
        ],
        "ordered_extension_descriptor_bindings": copy.deepcopy(
            extensions["ordered_descriptor_bindings"]
        ),
        "ordered_extension_descriptor_bindings_sha256": extensions[
            "ordered_descriptor_bindings_sha256"
        ],
        "prefix_fixed_c_round_count": extensions["prefix_round_count"],
        "suffix_distance_round_count": extensions["suffix_round_count"],
        "minimum_tokens_strictly_after_owned": min(
            int(record["context"]["tokens_strictly_after_owned"])
            for record in records
        ),
        "maximum_runtime_total_token_bound": max(
            int(record["context"]["runtime_total_token_bound"])
            for record in records
        ),
        "fixed_c_feasible_records": sum(
            bool(
                record["context"]["fixed_c_reference"][
                    "all_affected_boundaries_feasible"
                ]
            )
            for record in records
        ),
        "admission_policy": copy.deepcopy(
            manifest["policy_lock"]["admission_policy"]
        ),
        "admission_policy_sha256": _payload_sha256(
            manifest["policy_lock"]["admission_policy"]
        ),
        "manifest_integrity_sha256": manifest["integrity"]["sha256"],
        "policy_lock_sha256": manifest["policy_lock"]["lock_sha256"],
    }


def validate_census(
    census: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    validate_manifest(manifest)
    payload = copy.deepcopy(dict(census))
    integrity = payload.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
        or payload != _census_from_manifest(manifest)
    ):
        raise ManifestError("geometry correction census drifted")


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    core_manifest: ManifestInput = DEFAULT_CORE_MANIFEST_PATH,
) -> tuple[CorrectedRecord, ...]:
    validate_manifest(manifest)
    rebuilt, _ = build_manifest(
        rows,
        tokenizer,
        core_manifest=core_manifest,
        require_pinned=False,
    )
    if rebuilt != manifest:
        raise ManifestError("geometry-corrected manifest rehydration drifted")
    core, _ = _load_json(core_manifest)
    examples = base.extract_longmemeval_examples(rows)
    by_source = {item.source_id: item for item in examples}
    core_roles = chat_v1.all_role_source_ids((core,))
    return tuple(
        _fit_record(
            stored,
            examples,
            by_source,
            tokenizer,
            excluded_extension_sources=core_roles,
        )
        for stored in core["records"]
    )

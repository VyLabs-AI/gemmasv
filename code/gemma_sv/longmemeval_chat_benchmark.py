"""Freeze LongMemEval V1 as a source-disjoint constrained Gemma chat protocol.

This module adapts already-frozen LongMemEval deletion records; it does not
select records from model behavior and it never scores a model.  The selected
official evidence sessions are rehydrated from the pinned oracle, reduced to
complete answer-bearing user/assistant rounds, and rendered with registered
Gemma turn tokens.  Deletion owns the complete latest target round.  Its
behavioral reference is a fresh render with that round omitted, while the
operator reference remains the training-free graft's fixed-C retained-key
refit.

The default inputs are the two development manifests used to establish the
all-role exclusion set and the resulting source-disjoint 4,096-token
confirmation artifact.  Importing this module is download-free.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import longmemeval_deletion_benchmark as base
from gemma_sv.demo_server.certificate import fixed_c_feasibility
from gemma_sv.rag_benchmark import TokenizedContext


SCHEMA = "gemma-sv-longmemeval-constrained-chat-v1"
SCHEMA_VERSION = 1
POLICY_LOCK_SCHEMA = "gemma-sv-longmemeval-chat-policy-lock-v1"
BENCHMARK_LABEL = "LongMemEval V1 constrained-chat deletion adaptation"

DEVELOPMENT_PARTITION = "development"
CONFIRMATION_PARTITION = "confirmation"
PARTITIONS = frozenset({DEVELOPMENT_PARTITION, CONFIRMATION_PARTITION})

CONSTRAINED_INSTRUCTION = (
    "Reply with only the remembered value, without explanation."
)
CHAT_FORMAT = "gemma-registered-turns-v1"
CHAT_TOKENIZER_ID = "google/gemma-3-4b-it"
CHAT_TOKENIZER_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
MODEL_ID = CHAT_TOKENIZER_ID
MODEL_REVISION = CHAT_TOKENIZER_REVISION

MAXIMUM_CONTEXT_TOKENS = 4_096
MINIMUM_TOKENS_AFTER_OWNED = 512
GREEDY_GENERATION_TOKEN_RESERVE = 64
WINDOW = 1_024
NU = 0.7
CHUNK = 128
SOLVER_SEED = 0

DEFAULT_DEVELOPMENT_MANIFEST_PATHS = (
    Path("outputs/gemma_sv_rag/longmemeval_it_knowledge_update_manifest.json"),
    Path("outputs/gemma_sv_rag/longmemeval_it_residual_holdout_manifest.json"),
)
DEFAULT_CONFIRMATION_MANIFEST_PATH = Path(
    "outputs/query_svattn/longmemeval_confirmation_4096.json"
)

PINNED_DEVELOPMENT_INTEGRITIES = {
    "df93f4bd46a4b0d25f0d71956d1a0713fb1785e7293fa8ec0ccd31acfa773262",
    "d32129c3ad9d00c7b8547c1d3870aad0ea6ec2e6f43b6b34caf8585d2dcb40b8",
}
PINNED_CONFIRMATION_INTEGRITY = (
    "4bdd9b56eb9f4fd71cfb3c21f5470cee9cb1d24fdf74b523c39316704f88ff11"
)
PINNED_CONFIRMATION_FILE_SHA256 = (
    "7d100755f6aa4d10da559b7ee00846fe9d7bf08dd1026745acf1d54c69133c30"
)
PINNED_DEVELOPMENT_RECORDS = 15
PINNED_DEVELOPMENT_ALL_ROLE_SOURCES = 32
PINNED_CONFIRMATION_RECORDS = 16
PINNED_CONFIRMATION_ELIGIBLE = 52

_FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "answer",
        "content",
        "generated_text",
        "messages",
        "prompt",
        "question",
        "sessions",
        "source_text",
        "turns",
    }
)

ManifestError = base.ManifestError
ManifestInput = Mapping[str, Any] | str | Path


@dataclass(frozen=True)
class SessionFragment:
    """Runtime-only complete official rounds from one evidence session."""

    source_id: str
    question_id: str
    session: base.LongMemEvalSession
    turn_indices: tuple[int, ...]
    answer_turn_indices: tuple[int, ...]
    turns: tuple[base.LongMemEvalTurn, ...]


@dataclass(frozen=True)
class ChatProbe:
    """One runtime-only constrained query and its answer tokenization."""

    probe_id: str
    kind: str
    question: str
    answer: str
    prompt_text: str
    target_token_ids: tuple[int, ...]

    @property
    def prompt(self) -> str:
        return self.prompt_text


@dataclass(frozen=True)
class RehydratedChatRecord:
    """Source text and token geometry reconstructed from a public manifest."""

    record_id: str
    partition: str
    target: base.LongMemEvalExample
    retained: base.LongMemEvalExample
    earlier_evidence: tuple[SessionFragment, ...]
    owned_session_prefix: tuple[SessionFragment, ...]
    owned_latest_evidence: SessionFragment
    retained_evidence: tuple[SessionFragment, ...]
    tail_evidence: tuple[SessionFragment, ...]
    context: TokenizedContext
    raw_omitted_token_ids: tuple[int, ...]
    probes: tuple[ChatProbe, ...]
    query_token_reserve: int
    fixed_c_diagnostics: tuple[Mapping[str, Any], ...]

    @property
    def raw_omitted_text(self) -> str:
        return self.context.edited_text


@dataclass(frozen=True)
class _ArtifactSet:
    development: tuple[Mapping[str, Any], ...]
    confirmation: Mapping[str, Any]
    development_source_ids: frozenset[str]
    confirmation_source_ids: frozenset[str]
    full_source_rows: tuple[Mapping[str, Any], ...]
    confirmation_source_rows: tuple[Mapping[str, Any], ...]
    development_descriptors: tuple[Mapping[str, Any], ...]
    confirmation_descriptor: Mapping[str, Any]


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


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif _is_sequence(value):
        for item in value:
            yield from _walk_keys(item)


def _require_sha256(value: Any, *, name: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ManifestError(f"{name} is not a lowercase SHA-256 digest")
    return rendered


def _load_mapping(value: ManifestInput, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    payload = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ManifestError(f"{name} must be a JSON object")
    return payload


@contextmanager
def _baseline_context_ceiling(value: int):
    """Scope the legacy validator's module-level 2,048-token guard."""

    ceiling = int(value)
    if ceiling not in (2_048, 4_096):
        raise ManifestError("baseline context ceiling must be 2,048 or 4,096")
    previous = base.RUNTIME_CONTEXT_CEILING
    base.RUNTIME_CONTEXT_CEILING = ceiling
    try:
        yield
    finally:
        base.RUNTIME_CONTEXT_CEILING = previous


def _validate_baseline_manifest(manifest: Mapping[str, Any]) -> None:
    erasure = manifest.get("erasure_config") or {}
    ceiling = int(erasure.get("runtime_context_ceiling", -1))
    with _baseline_context_ceiling(ceiling):
        base.validate_manifest(manifest)


def all_role_source_ids(manifests: Iterable[Mapping[str, Any]]) -> set[str]:
    """Collect target, retained-probe, and tail-session source ownership."""

    source_ids: set[str] = set()
    for manifest in manifests:
        records = manifest.get("records")
        if not isinstance(records, list):
            raise ManifestError("baseline manifest records are missing")
        for record in records:
            target = record.get("target") or {}
            retained = record.get("retained_probe") or {}
            target_id = str(target.get("source_id") or "")
            retained_id = str(retained.get("source_id") or "")
            if not target_id or not retained_id:
                raise ManifestError(
                    "baseline record lacks target or retained source ownership"
                )
            source_ids.update((target_id, retained_id))
            tail = record.get("tail_session_references")
            if not isinstance(tail, list) or not tail:
                raise ManifestError("baseline record lacks tail source ownership")
            for reference in tail:
                source_id = str((reference or {}).get("source_id") or "")
                if not source_id:
                    raise ManifestError("tail reference lacks source ownership")
                source_ids.add(source_id)
    return source_ids


def _artifact_descriptor(
    manifest: Mapping[str, Any],
    *,
    partition: str,
) -> dict[str, Any]:
    integrity = _require_sha256(
        (manifest.get("integrity") or {}).get("sha256"),
        name="baseline manifest integrity",
    )
    records = manifest.get("records") or []
    role_ids = all_role_source_ids((manifest,))
    descriptor: dict[str, Any] = {
        "artifact_id": base.stable_identifier(
            "longmemeval-chat-input", partition, integrity
        ),
        "schema": str(manifest.get("schema") or ""),
        "integrity_sha256": integrity,
        "records": len(records),
        "record_ids_sha256": _payload_sha256(
            [str(record.get("record_id") or "") for record in records]
        ),
        "all_role_source_count": len(role_ids),
        "all_role_source_ids_sha256": _payload_sha256(sorted(role_ids)),
    }
    canonical_paths = {
        "df93f4bd46a4b0d25f0d71956d1a0713fb1785e7293fa8ec0ccd31acfa773262": (
            str(DEFAULT_DEVELOPMENT_MANIFEST_PATHS[0])
        ),
        "d32129c3ad9d00c7b8547c1d3870aad0ea6ec2e6f43b6b34caf8585d2dcb40b8": (
            str(DEFAULT_DEVELOPMENT_MANIFEST_PATHS[1])
        ),
        PINNED_CONFIRMATION_INTEGRITY: str(
            DEFAULT_CONFIRMATION_MANIFEST_PATH
        ),
    }
    if integrity in canonical_paths:
        descriptor["reusable_artifact"] = canonical_paths[integrity]
    if integrity == PINNED_CONFIRMATION_INTEGRITY:
        path = Path(canonical_paths[integrity])
        if not path.is_file():
            raise ManifestError("pinned confirmation artifact is missing")
        observed = _file_sha256(path)
        if observed != PINNED_CONFIRMATION_FILE_SHA256:
            raise ManifestError("pinned confirmation file SHA-256 drifted")
        descriptor["file_sha256"] = observed
    return descriptor


def _resolve_artifacts(
    development_manifests: Sequence[ManifestInput],
    confirmation_manifest: ManifestInput,
    *,
    require_pinned: bool,
) -> _ArtifactSet:
    if not development_manifests:
        raise ManifestError("at least one development manifest is required")
    development = tuple(
        _load_mapping(value, name="development manifest")
        for value in development_manifests
    )
    confirmation = _load_mapping(
        confirmation_manifest,
        name="confirmation manifest",
    )
    for manifest in (*development, confirmation):
        _validate_baseline_manifest(manifest)

    full_rows = development[0].get("source_rows")
    if not isinstance(full_rows, list) or not full_rows:
        raise ManifestError("development source inventory is missing")
    if any(manifest.get("source_rows") != full_rows for manifest in development):
        raise ManifestError("development source inventories differ")
    full_ids = [str(row.get("source_id") or "") for row in full_rows]
    if (
        not all(full_ids)
        or full_ids != sorted(full_ids)
        or len(full_ids) != len(set(full_ids))
    ):
        raise ManifestError("full source inventory is not canonical")

    development_ids = all_role_source_ids(development)
    if not development_ids.issubset(full_ids):
        raise ManifestError("development ownership is outside source inventory")
    expected_confirmation_rows = [
        row for row in full_rows if str(row.get("source_id")) not in development_ids
    ]
    observed_confirmation_rows = confirmation.get("source_rows")
    if observed_confirmation_rows != expected_confirmation_rows:
        raise ManifestError(
            "confirmation source inventory is not the all-role exclusion"
        )

    confirmation_ids = all_role_source_ids((confirmation,))
    confirmation_inventory_ids = {
        str(row.get("source_id") or "")
        for row in observed_confirmation_rows
    }
    if not confirmation_ids.issubset(confirmation_inventory_ids):
        raise ManifestError(
            "confirmation ownership is outside its excluded source inventory"
        )
    overlap = development_ids.intersection(confirmation_ids)
    if overlap:
        raise ManifestError(
            "development and confirmation overlap across source roles"
        )
    development_targets = [
        str(record["target"]["source_id"])
        for manifest in development
        for record in manifest["records"]
    ]
    if len(development_targets) != len(set(development_targets)):
        raise ManifestError("development target records are duplicated")
    confirmation_targets = [
        str(record["target"]["source_id"])
        for record in confirmation["records"]
    ]
    if len(confirmation_targets) != len(set(confirmation_targets)):
        raise ManifestError("confirmation target records are duplicated")

    protocol = confirmation.get("query_svattn_protocol")
    if protocol is not None:
        if (
            protocol.get("selection_uses_model_outputs") is not False
            or int(protocol.get("development_sources_excluded", -1))
            != len(development_ids)
            or int(protocol.get("selected_records", -1))
            != len(confirmation["records"])
        ):
            raise ManifestError("confirmation exclusion protocol drifted")

    development_descriptors = tuple(
        sorted(
            (
                _artifact_descriptor(
                    manifest,
                    partition=DEVELOPMENT_PARTITION,
                )
                for manifest in development
            ),
            key=lambda item: str(item["integrity_sha256"]),
        )
    )
    confirmation_descriptor = _artifact_descriptor(
        confirmation,
        partition=CONFIRMATION_PARTITION,
    )

    if require_pinned:
        development_integrities = {
            str(item["integrity_sha256"]) for item in development_descriptors
        }
        if development_integrities != PINNED_DEVELOPMENT_INTEGRITIES:
            raise ManifestError("development artifacts are not the pinned pair")
        if (
            confirmation_descriptor["integrity_sha256"]
            != PINNED_CONFIRMATION_INTEGRITY
        ):
            raise ManifestError("confirmation artifact is not the pinned 4,096 set")
        if sum(len(item["records"]) for item in development) != (
            PINNED_DEVELOPMENT_RECORDS
        ):
            raise ManifestError("pinned development record count drifted")
        if len(development_ids) != PINNED_DEVELOPMENT_ALL_ROLE_SOURCES:
            raise ManifestError("pinned all-role exclusion count drifted")
        if len(confirmation["records"]) != PINNED_CONFIRMATION_RECORDS:
            raise ManifestError("pinned confirmation record count drifted")
        if len(full_rows) != base.DATASET_NUM_ROWS:
            raise ManifestError("pinned full source inventory count drifted")
        if len(observed_confirmation_rows) != (
            base.DATASET_NUM_ROWS - PINNED_DEVELOPMENT_ALL_ROLE_SOURCES
        ):
            raise ManifestError("pinned confirmation source inventory drifted")
        if not isinstance(protocol, Mapping) or (
            protocol.get("schema")
            != "query-svattn-longmemeval-confirmation-4096-v1"
            or int(protocol.get("eligible_count", -1))
            != PINNED_CONFIRMATION_ELIGIBLE
            or protocol.get("evidence_status") != "confirmation"
        ):
            raise ManifestError("pinned confirmation policy marker drifted")

    return _ArtifactSet(
        development=development,
        confirmation=confirmation,
        development_source_ids=frozenset(development_ids),
        confirmation_source_ids=frozenset(confirmation_ids),
        full_source_rows=tuple(full_rows),
        confirmation_source_rows=tuple(observed_confirmation_rows),
        development_descriptors=development_descriptors,
        confirmation_descriptor=confirmation_descriptor,
    )


def validate_reusable_artifacts(
    development_manifests: Sequence[ManifestInput] = (
        *DEFAULT_DEVELOPMENT_MANIFEST_PATHS,
    ),
    confirmation_manifest: ManifestInput = (
        DEFAULT_CONFIRMATION_MANIFEST_PATH
    ),
    *,
    require_pinned: bool = True,
) -> dict[str, Any]:
    """Validate source inventories and return a source-free overlap report."""

    artifacts = _resolve_artifacts(
        development_manifests,
        confirmation_manifest,
        require_pinned=require_pinned,
    )
    return {
        "schema": "gemma-sv-longmemeval-chat-artifact-validation-v1",
        "development_records": sum(
            len(manifest["records"]) for manifest in artifacts.development
        ),
        "development_all_role_sources": len(
            artifacts.development_source_ids
        ),
        "confirmation_records": len(artifacts.confirmation["records"]),
        "confirmation_all_role_sources": len(
            artifacts.confirmation_source_ids
        ),
        "full_source_rows": len(artifacts.full_source_rows),
        "confirmation_source_rows": len(
            artifacts.confirmation_source_rows
        ),
        "all_role_source_disjoint": True,
        "development_artifacts": [
            dict(item) for item in artifacts.development_descriptors
        ],
        "confirmation_artifact": dict(
            artifacts.confirmation_descriptor
        ),
        "require_pinned": bool(require_pinned),
        "selection_uses_model_outputs": False,
    }


def _chat_serialization() -> dict[str, Any]:
    return {
        "format": CHAT_FORMAT,
        "single_bos": True,
        "user_role_token": "<start_of_turn>user",
        "assistant_role_token": "<start_of_turn>model",
        "complete_turn_terminator": "<end_of_turn>",
        "query_is_incremental_user_turn": True,
        "query_response_contract": "bare remembered value",
        "constrained_instruction": CONSTRAINED_INSTRUCTION,
        "session_date_prefix": "[Session date: {official date}]",
    }


def _model_config() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "adapter": None,
        "arm": "it_native_chat_training_free_graft",
    }


def _graft_config() -> dict[str, Any]:
    return {
        "training_steps": 0,
        "window": WINDOW,
        "nu": NU,
        "chunk": CHUNK,
        "solver_seed": SOLVER_SEED,
        "readout": "softmax",
        "preserve_prefix_mass": True,
        "per_boundary_box": True,
        "maximum_context_tokens": MAXIMUM_CONTEXT_TOKENS,
        "minimum_tokens_strictly_after_owned": (
            MINIMUM_TOKENS_AFTER_OWNED
        ),
        "greedy_generation_token_reserve": (
            GREEDY_GENERATION_TOKEN_RESERVE
        ),
    }


def _evaluation_contract() -> dict[str, Any]:
    return {
        "official_longmemeval_leaderboard_score": False,
        "owned_unit": (
            "complete official user-assistant round(s) containing has_answer "
            "turns in the frozen latest target evidence session"
        ),
        "retained_unit": (
            "complete official user-assistant round(s) containing has_answer "
            "turns in the frozen retained evidence session"
        ),
        "same_source_round_fill": (
            "shortest nonconflicting complete rounds from already-frozen "
            "earlier or tail sessions; no new source IDs"
        ),
        "suffix_fill_goal": (
            "restore the frozen 512-token post-owned distance after chat "
            "serialization"
        ),
        "fixed_c_prefix_fill_goal": (
            "improve locked-nu feasibility within the 4,096-token bound "
            "without dropping or replacing records"
        ),
        "behavioral_reference": (
            "fresh registered-chat render with the owned round omitted"
        ),
        "certificate_reference": "fixed-C retained-key refit",
        "raw_omission_required": True,
        "deletion_request_is_out_of_band_and_answer_free": True,
        "selection_uses_model_outputs": False,
        "model_scoring_performed": False,
        "no_replacement": True,
    }


def _build_policy_lock_from_artifacts(
    artifacts: _ArtifactSet,
    *,
    reusable_artifacts_pinned: bool,
) -> dict[str, Any]:
    development_hashes = sorted(
        base.text_sha256(source_id)
        for source_id in artifacts.development_source_ids
    )
    body: dict[str, Any] = {
        "schema": POLICY_LOCK_SCHEMA,
        "status": "locked-before-model-scoring",
        "contains_source_text": False,
        "results_present": False,
        "selection_uses_model_outputs": False,
        "output_based_replacement": False,
        "reusable_artifacts_pinned": bool(reusable_artifacts_pinned),
        "artifacts": {
            "development": [
                dict(item) for item in artifacts.development_descriptors
            ],
            "confirmation": dict(artifacts.confirmation_descriptor),
        },
        "source_inventory": {
            "dataset_id": base.DATASET_ID,
            "dataset_revision": base.DATASET_REVISION,
            "source_artifact_sha256": base.DATASET_ARTIFACT_SHA256,
            "full_row_count": len(artifacts.full_source_rows),
            "full_descriptors_sha256": _payload_sha256(
                list(artifacts.full_source_rows)
            ),
            "confirmation_row_count": len(
                artifacts.confirmation_source_rows
            ),
            "confirmation_descriptors_sha256": _payload_sha256(
                list(artifacts.confirmation_source_rows)
            ),
        },
        "partition_policy": {
            "development_records": sum(
                len(manifest["records"])
                for manifest in artifacts.development
            ),
            "confirmation_records": len(
                artifacts.confirmation["records"]
            ),
            "development_all_role_source_count": len(
                artifacts.development_source_ids
            ),
            "development_source_id_sha256": development_hashes,
            "exclusion_scope": (
                "target, retained-probe, and tail-session source IDs"
            ),
            "confirmation_all_roles_disjoint": True,
            "confirmation_selection_frozen": True,
            "no_replacement": True,
        },
        "chat_serialization": _chat_serialization(),
        "model": _model_config(),
        "training_free_graft": _graft_config(),
        "evaluation_contract": _evaluation_contract(),
    }
    body["lock_sha256"] = _payload_sha256(body)
    validate_policy_lock(body)
    return body


def build_policy_lock(
    development_manifests: Sequence[ManifestInput],
    confirmation_manifest: ManifestInput,
    *,
    require_pinned: bool = False,
) -> dict[str, Any]:
    """Bind source partitioning, chat serialization, and graft references."""

    artifacts = _resolve_artifacts(
        development_manifests,
        confirmation_manifest,
        require_pinned=require_pinned,
    )
    return _build_policy_lock_from_artifacts(
        artifacts,
        reusable_artifacts_pinned=require_pinned,
    )


def validate_policy_lock(lock: Mapping[str, Any]) -> None:
    claimed = _require_sha256(
        lock.get("lock_sha256"),
        name="policy lock digest",
    )
    body = copy.deepcopy(dict(lock))
    body.pop("lock_sha256", None)
    if claimed != _payload_sha256(body):
        raise ManifestError("policy lock digest mismatch")
    if (
        lock.get("schema") != POLICY_LOCK_SCHEMA
        or lock.get("status") != "locked-before-model-scoring"
        or lock.get("contains_source_text") is not False
        or lock.get("results_present") is not False
        or lock.get("selection_uses_model_outputs") is not False
        or lock.get("output_based_replacement") is not False
    ):
        raise ManifestError("LongMemEval chat policy is not locked")
    if lock.get("chat_serialization") != _chat_serialization():
        raise ManifestError("registered chat serialization policy drifted")
    if lock.get("model") != _model_config():
        raise ManifestError("4B-IT model policy drifted")
    if lock.get("training_free_graft") != _graft_config():
        raise ManifestError("training-free graft policy drifted")
    if lock.get("evaluation_contract") != _evaluation_contract():
        raise ManifestError("raw omission or fixed-C reference drifted")

    partition = lock.get("partition_policy") or {}
    development_hashes = partition.get("development_source_id_sha256")
    if (
        not isinstance(development_hashes, list)
        or development_hashes != sorted(set(development_hashes))
        or len(development_hashes)
        != int(partition.get("development_all_role_source_count", -1))
        or partition.get("confirmation_all_roles_disjoint") is not True
        or partition.get("confirmation_selection_frozen") is not True
        or partition.get("no_replacement") is not True
        or partition.get("exclusion_scope")
        != "target, retained-probe, and tail-session source IDs"
    ):
        raise ManifestError("all-role partition policy drifted")
    for digest in development_hashes:
        _require_sha256(digest, name="development source hash")

    artifacts = lock.get("artifacts") or {}
    development = artifacts.get("development")
    confirmation = artifacts.get("confirmation")
    if (
        not isinstance(development, list)
        or not development
        or not isinstance(confirmation, Mapping)
    ):
        raise ManifestError("policy lock artifact bindings are missing")
    for descriptor in (*development, confirmation):
        _require_sha256(
            descriptor.get("integrity_sha256"),
            name="artifact integrity",
        )
        _require_sha256(
            descriptor.get("record_ids_sha256"),
            name="artifact record digest",
        )
        _require_sha256(
            descriptor.get("all_role_source_ids_sha256"),
            name="artifact role digest",
        )
    inventory = lock.get("source_inventory") or {}
    if (
        inventory.get("dataset_id") != base.DATASET_ID
        or inventory.get("dataset_revision") != base.DATASET_REVISION
        or inventory.get("source_artifact_sha256")
        != base.DATASET_ARTIFACT_SHA256
        or int(inventory.get("full_row_count", -1)) < 1
        or int(inventory.get("confirmation_row_count", -1)) < 1
    ):
        raise ManifestError("LongMemEval source inventory lock drifted")
    _require_sha256(
        inventory.get("full_descriptors_sha256"),
        name="full source inventory digest",
    )
    _require_sha256(
        inventory.get("confirmation_descriptors_sha256"),
        name="confirmation source inventory digest",
    )
    leaked = _FORBIDDEN_SOURCE_KEYS.intersection(_walk_keys(lock))
    if leaked:
        raise ManifestError(
            "policy lock contains source-text fields: "
            + ", ".join(sorted(leaked))
        )


def render_registered_turn(role: str, content: str) -> str:
    """Render one user/model message with Gemma's registered delimiters."""

    normalized = str(role).strip().casefold()
    if normalized == "assistant":
        normalized = "model"
    if normalized not in {"user", "model"}:
        raise ManifestError(f"unsupported LongMemEval chat role {role!r}")
    rendered_content = str(content)
    if not rendered_content.strip():
        raise ManifestError("LongMemEval chat turn content is empty")
    return (
        f"<start_of_turn>{normalized}\n"
        f"{rendered_content}"
        "<end_of_turn>\n"
    )


def render_registered_chat(
    messages: Sequence[Mapping[str, Any]],
    *,
    add_bos: bool = True,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Render alternating official turns and return complete-turn spans."""

    rendered = "<bos>" if add_bos else ""
    spans: list[tuple[int, int]] = []
    expected = "user"
    for message in messages:
        role = str(message.get("role") or "").strip().casefold()
        normalized = "assistant" if role in {"assistant", "model"} else role
        if normalized != expected:
            raise ManifestError("LongMemEval chat roles do not alternate")
        start = len(rendered)
        rendered += render_registered_turn(
            normalized,
            str(message.get("content") or ""),
        )
        spans.append((start, len(rendered)))
        expected = "assistant" if expected == "user" else "user"
    if messages and expected != "user":
        raise ManifestError("LongMemEval chat ends without an assistant response")
    return rendered, tuple(spans)


def render_constrained_query(question: str, question_date: str) -> str:
    """Render the frozen incremental user turn; do not add a second BOS."""

    rendered_question = (
        f"Current Date: {str(question_date)}\n"
        f"Question: {str(question)}\n"
        f"{CONSTRAINED_INSTRUCTION}"
    )
    return (
        render_registered_turn("user", rendered_question)
        + "<start_of_turn>model\n"
    )


# Private aliases mirror the established chat benchmark's small rendering API.
_render_turn = render_registered_turn
_render_chat = render_registered_chat
_render_query = render_constrained_query


def _answer_pair_starts(
    session: base.LongMemEvalSession,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    answer_indices = tuple(
        index
        for index, turn in enumerate(session.turns)
        if turn.has_answer is True
    )
    if not answer_indices:
        raise ManifestError("answer-bearing turn unavailable")
    pair_starts: set[int] = set()
    for index in answer_indices:
        role = session.turns[index].role.strip().casefold()
        if role == "user":
            start = index
        elif role in {"assistant", "model"}:
            start = index - 1
        else:
            raise ManifestError("answer-bearing turn has unsupported role")
        if (
            start < 0
            or start + 1 >= len(session.turns)
            or session.turns[start].role.strip().casefold() != "user"
            or session.turns[start + 1].role.strip().casefold()
            not in {"assistant", "model"}
        ):
            raise ManifestError(
                "answer-bearing turn is not in a complete user-assistant round"
            )
        pair_starts.add(start)
    return tuple(sorted(pair_starts)), answer_indices


def _fragment_for_pair_starts(
    example: base.LongMemEvalExample,
    session: base.LongMemEvalSession,
    pair_starts: Iterable[int],
    *,
    answer_indices: Sequence[int],
) -> SessionFragment:
    normalized_starts = tuple(sorted({int(value) for value in pair_starts}))
    if not normalized_starts:
        raise ManifestError("official session fragment contains no rounds")
    selected = tuple(
        index
        for start in normalized_starts
        for index in (start, start + 1)
    )
    if any(
        start < 0
        or start + 1 >= len(session.turns)
        or session.turns[start].role.strip().casefold() != "user"
        or session.turns[start + 1].role.strip().casefold()
        not in {"assistant", "model"}
        for start in normalized_starts
    ):
        raise ManifestError("official session contains an incomplete round")
    turns = tuple(session.turns[index] for index in selected)
    if any(not turn.content.strip() for turn in turns):
        raise ManifestError("selected official evidence round is empty")
    return SessionFragment(
        source_id=example.source_id,
        question_id=example.question_id,
        session=session,
        turn_indices=selected,
        answer_turn_indices=tuple(int(value) for value in answer_indices),
        turns=turns,
    )


def _evidence_fragment(
    example: base.LongMemEvalExample,
    session: base.LongMemEvalSession,
) -> SessionFragment:
    """Select complete official rounds that contain every has_answer turn."""

    pair_starts, answer_indices = _answer_pair_starts(session)
    return _fragment_for_pair_starts(
        example,
        session,
        pair_starts,
        answer_indices=answer_indices,
    )


def _session_round_expansions(
    example: base.LongMemEvalExample,
    session: base.LongMemEvalSession,
    *,
    forbidden_values: Sequence[str],
) -> tuple[SessionFragment, tuple[int, ...]]:
    """Return the evidence fragment plus deterministic same-session fillers."""

    evidence_starts, answer_indices = _answer_pair_starts(session)
    if len(session.turns) % 2:
        raise ManifestError("official tail session has an incomplete round")
    all_starts = tuple(range(0, len(session.turns), 2))
    # Check the complete official session before considering any extra round.
    _fragment_for_pair_starts(
        example,
        session,
        all_starts,
        answer_indices=answer_indices,
    )
    forbidden = tuple(
        value.strip().casefold()
        for value in forbidden_values
        if value.strip()
    )
    candidates = []
    for start in all_starts:
        if start in set(evidence_starts):
            continue
        rendered = "\n".join(
            session.turns[index].content
            for index in (start, start + 1)
        ).casefold()
        if any(value in rendered for value in forbidden):
            continue
        candidates.append(start)
    candidates.sort(
        key=lambda start: (
            sum(
                len(session.turns[index].content)
                for index in (start, start + 1)
            ),
            start,
        )
    )
    return (
        _fragment_for_pair_starts(
            example,
            session,
            evidence_starts,
            answer_indices=answer_indices,
        ),
        tuple(candidates),
    )


def _fragment_messages(fragment: SessionFragment) -> list[dict[str, str]]:
    messages = [
        {"role": turn.role, "content": turn.content}
        for turn in fragment.turns
    ]
    if not messages or messages[0]["role"].strip().casefold() != "user":
        raise ManifestError("official evidence fragment does not start with user")
    messages[0] = {
        "role": "user",
        "content": (
            f"[Session date: {fragment.session.date_text}]\n"
            f"{messages[0]['content']}"
        ),
    }
    return messages


def _render_fragments(
    fragments: Sequence[SessionFragment],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    rendered = "<bos>"
    spans: list[tuple[int, int]] = []
    for fragment in fragments:
        start = len(rendered)
        fragment_text, _ = render_registered_chat(
            _fragment_messages(fragment),
            add_bos=False,
        )
        rendered += fragment_text
        spans.append((start, len(rendered)))
    return rendered, tuple(spans)


def _token_ids_no_special(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False)
    raw = (
        encoded.get("input_ids")
        if isinstance(encoded, Mapping)
        else getattr(encoded, "input_ids", None)
    )
    if raw is None:
        raise ManifestError("tokenizer omitted input_ids")
    if raw and _is_sequence(raw[0]):
        if len(raw) != 1:
            raise ManifestError("batched tokenization is unsupported")
        raw = raw[0]
    return tuple(int(value) for value in raw)


def _target_token_ids(
    tokenizer: Any,
    prompt: str,
    answer: str,
) -> tuple[int, ...]:
    rendered = str(answer)
    if prompt and not prompt[-1].isspace() and not rendered[:1].isspace():
        rendered = " " + rendered
    token_ids = _token_ids_no_special(tokenizer, rendered)
    if not token_ids:
        raise ManifestError("LongMemEval answer maps to no target tokens")
    return token_ids


def _positions_for_span(
    offsets: Sequence[tuple[int, int]],
    span: tuple[int, int],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, offset in enumerate(offsets)
        if base._overlaps(offset, span)
    )


def _record_spec(
    record: Mapping[str, Any],
    *,
    artifact_integrity: str,
) -> dict[str, Any]:
    target = record.get("target") or {}
    retained = record.get("retained_probe") or {}
    return {
        "baseline": {
            "record_id": str(record.get("record_id") or ""),
            "manifest_integrity_sha256": str(artifact_integrity),
        },
        "target": {
            "source_id": str(target.get("source_id") or ""),
            "question_id": str(target.get("question_id") or ""),
            "question_type": str(target.get("question_type") or ""),
            "source_row_sha256": str(target.get("source_row_sha256") or ""),
            "question_sha256": str(target.get("question_sha256") or ""),
            "current_answer_sha256": str(
                target.get("current_answer_sha256") or ""
            ),
            "question_date_sha256": str(
                target.get("question_date_sha256") or ""
            ),
            "answer_session_ids": [
                str(value) for value in target.get("answer_session_ids") or ()
            ],
            "earlier_evidence_session_ids": [
                str(value)
                for value in target.get("earlier_evidence_session_ids") or ()
            ],
            "owned_latest_evidence_session_id": str(
                target.get("owned_latest_evidence_session_id") or ""
            ),
            "previous_answer_available_from_official_schema": False,
            "previous_answer_invented": False,
            "gate_uses_previous_answer": False,
        },
        "retained_probe": {
            "source_id": str(retained.get("source_id") or ""),
            "question_id": str(retained.get("question_id") or ""),
            "question_type": str(retained.get("question_type") or ""),
            "source_row_sha256": str(
                retained.get("source_row_sha256") or ""
            ),
            "question_sha256": str(retained.get("question_sha256") or ""),
            "current_answer_sha256": str(
                retained.get("current_answer_sha256") or ""
            ),
            "question_date_sha256": str(
                retained.get("question_date_sha256") or ""
            ),
            "evidence_session_ids": [
                str(value)
                for value in retained.get("evidence_session_ids") or ()
            ],
            "distinct_official_non_abstention_row": True,
        },
        "tail_session_references": [
            {
                "source_id": str(reference.get("source_id") or ""),
                "question_id": str(reference.get("question_id") or ""),
                "session_id": str(reference.get("session_id") or ""),
            }
            for reference in record.get("tail_session_references") or ()
        ],
    }


def _check_example_descriptor(
    example: base.LongMemEvalExample,
    stored: Mapping[str, Any],
    *,
    retained: bool,
) -> None:
    expected = {
        "source_id": example.source_id,
        "question_id": example.question_id,
        "question_type": example.question_type,
        "source_row_sha256": example.source_row_sha256,
        "question_sha256": base.text_sha256(example.question),
        "current_answer_sha256": base.text_sha256(example.answer),
        "question_date_sha256": base.text_sha256(example.question_date),
    }
    if any(stored.get(key) != value for key, value in expected.items()):
        kind = "retained" if retained else "target"
        raise ManifestError(f"{kind} source descriptor drifted")


def _session_by_id(
    example: base.LongMemEvalExample,
    session_id: str,
) -> base.LongMemEvalSession:
    matches = [
        session
        for session in example.haystack_sessions
        if session.session_id == session_id
    ]
    if len(matches) != 1:
        raise ManifestError("frozen session reference is missing or ambiguous")
    return matches[0]


def _casefold_span(
    text: str,
    value: str,
    within: tuple[int, int],
) -> tuple[int, int]:
    start = text.casefold().find(
        value.casefold(),
        within[0],
        within[1],
    )
    return (
        (start, start + len(value))
        if start >= 0
        else within
    )


def _prepare_record(
    spec: Mapping[str, Any],
    examples_by_source: Mapping[str, base.LongMemEvalExample],
    tokenizer: Any,
    *,
    partition: str,
) -> RehydratedChatRecord:
    target_stored = spec.get("target") or {}
    retained_stored = spec.get("retained_probe") or {}
    target_id = str(target_stored.get("source_id") or "")
    retained_id = str(retained_stored.get("source_id") or "")
    if target_id not in examples_by_source or retained_id not in examples_by_source:
        raise ManifestError("frozen target or retained source row is missing")
    target = examples_by_source[target_id]
    retained = examples_by_source[retained_id]
    _check_example_descriptor(target, target_stored, retained=False)
    _check_example_descriptor(retained, retained_stored, retained=True)
    if target.source_id == retained.source_id:
        raise ManifestError("target and retained source ownership overlap")

    earlier_sessions, owned_session = base.latest_update_partition(target)
    if [session.session_id for session in earlier_sessions] != list(
        target_stored.get("earlier_evidence_session_ids") or ()
    ):
        raise ManifestError("frozen earlier target evidence drifted")
    if owned_session.session_id != target_stored.get(
        "owned_latest_evidence_session_id"
    ):
        raise ManifestError("frozen latest target evidence drifted")
    if list(target.answer_session_ids) != list(
        target_stored.get("answer_session_ids") or ()
    ):
        raise ManifestError("target answer-session inventory drifted")

    retained_ids = list(
        retained_stored.get("evidence_session_ids") or ()
    )
    if len(retained_ids) != 1:
        raise ManifestError("retained evidence must be one frozen session")
    retained_sessions = tuple(
        _session_by_id(retained, str(session_id))
        for session_id in retained_ids
    )
    probes = tuple(
        ChatProbe(
            probe_id=probe_id,
            kind=kind,
            question=example.question,
            answer=example.answer,
            prompt_text=render_constrained_query(
                example.question,
                example.question_date,
            ),
            target_token_ids=_target_token_ids(
                tokenizer,
                render_constrained_query(
                    example.question,
                    example.question_date,
                ),
                example.answer,
            ),
        )
        for probe_id, kind, example in (
            ("target_current", "deleted", target),
            ("retained", "retained", retained),
        )
    )
    query_reserve = max(
        len(_token_ids_no_special(tokenizer, probe.prompt_text))
        + max(
            len(probe.target_token_ids),
            GREEDY_GENERATION_TOKEN_RESERVE,
        )
        for probe in probes
    )

    tail_fragments: list[SessionFragment] = []
    tail_expansions: list[
        tuple[
            base.LongMemEvalExample,
            base.LongMemEvalSession,
            tuple[int, ...],
        ]
    ] = []
    for reference in spec.get("tail_session_references") or ():
        source_id = str(reference.get("source_id") or "")
        if source_id not in examples_by_source:
            raise ManifestError("frozen tail source row is missing")
        example = examples_by_source[source_id]
        if example.question_id != str(reference.get("question_id") or ""):
            raise ManifestError("frozen tail question reference drifted")
        session = _session_by_id(
            example,
            str(reference.get("session_id") or ""),
        )
        fragment, expansions = _session_round_expansions(
            example,
            session,
            forbidden_values=(target.answer, retained.answer),
        )
        tail_fragments.append(fragment)
        tail_expansions.append((example, session, expansions))
    if not tail_fragments:
        raise ManifestError("frozen deterministic tail is empty")

    earlier_fragments: list[SessionFragment] = []
    earlier_expansions: list[
        tuple[
            base.LongMemEvalExample,
            base.LongMemEvalSession,
            tuple[int, ...],
        ]
    ] = []
    for session in earlier_sessions:
        fragment, expansions = _session_round_expansions(
            target,
            session,
            forbidden_values=(target.answer, retained.answer),
        )
        earlier_fragments.append(fragment)
        earlier_expansions.append((target, session, expansions))
    owned_fragment, owned_expansions = _session_round_expansions(
        target,
        owned_session,
        forbidden_values=(target.answer, retained.answer),
    )
    owned_pair_starts, _ = _answer_pair_starts(owned_session)
    owned_prefix_candidates = tuple(
        pair_start
        for pair_start in owned_expansions
        if pair_start < min(owned_pair_starts)
    )
    owned_session_prefix: list[SessionFragment] = []
    retained_fragments = tuple(
        _evidence_fragment(retained, session)
        for session in retained_sessions
    )
    owned_index = len(earlier_fragments)
    retained_start = owned_index + 1
    retained_end = retained_start + len(retained_fragments)

    expansion_queue = [
        (tail_index, pair_start)
        for tail_index, (_, _, candidates) in enumerate(tail_expansions)
        for pair_start in candidates
    ]
    next_expansion = 0
    while True:
        fragments = (
            *earlier_fragments,
            owned_fragment,
            *retained_fragments,
            *tail_fragments,
        )
        original_text, fragment_spans = _render_fragments(fragments)
        owned_span = fragment_spans[owned_index]
        retained_span = (
            fragment_spans[retained_start][0],
            fragment_spans[retained_end - 1][1],
        )
        original_ids, offsets = base._tokenizer_payload(
            tokenizer,
            original_text,
        )
        forget_positions = _positions_for_span(offsets, owned_span)
        if not forget_positions:
            raise ManifestError(
                "owned registered turns map to no model tokens"
            )
        tokens_after = len(original_ids) - max(forget_positions) - 1
        if tokens_after >= MINIMUM_TOKENS_AFTER_OWNED:
            break
        if next_expansion >= len(expansion_queue):
            raise ManifestError(
                "frozen tail sessions cannot restore suffix distance"
            )
        tail_index, pair_start = expansion_queue[next_expansion]
        next_expansion += 1
        example, session, _ = tail_expansions[tail_index]
        current = tail_fragments[tail_index]
        selected_starts = {
            current.turn_indices[index]
            for index in range(0, len(current.turn_indices), 2)
        }
        selected_starts.add(pair_start)
        tail_fragments[tail_index] = _fragment_for_pair_starts(
            example,
            session,
            selected_starts,
            answer_indices=current.answer_turn_indices,
        )

    _, feasibility = fixed_c_feasibility(
        len(original_ids),
        forget_positions,
        nu=NU,
        chunk=CHUNK,
        per_boundary_box=True,
    )
    prefix_expansion_queue = [
        (fragment_index, pair_start)
        for fragment_index, (_, _, candidates) in enumerate(
            earlier_expansions
        )
        for pair_start in candidates
    ]
    for fragment_index, pair_start in prefix_expansion_queue:
        if all(bool(item.get("feasible")) for item in feasibility):
            break
        example, session, _ = earlier_expansions[fragment_index]
        current = earlier_fragments[fragment_index]
        selected_starts = {
            current.turn_indices[index]
            for index in range(0, len(current.turn_indices), 2)
        }
        selected_starts.add(pair_start)
        candidate = _fragment_for_pair_starts(
            example,
            session,
            selected_starts,
            answer_indices=current.answer_turn_indices,
        )
        previous = earlier_fragments[fragment_index]
        earlier_fragments[fragment_index] = candidate
        candidate_fragments = (
            *earlier_fragments,
            owned_fragment,
            *retained_fragments,
            *tail_fragments,
        )
        candidate_text, candidate_spans = _render_fragments(
            candidate_fragments
        )
        candidate_ids, candidate_offsets = base._tokenizer_payload(
            tokenizer,
            candidate_text,
        )
        if len(candidate_ids) + query_reserve > MAXIMUM_CONTEXT_TOKENS:
            earlier_fragments[fragment_index] = previous
            continue
        candidate_owned_span = candidate_spans[owned_index]
        candidate_forget = _positions_for_span(
            candidate_offsets,
            candidate_owned_span,
        )
        _, candidate_feasibility = fixed_c_feasibility(
            len(candidate_ids),
            candidate_forget,
            nu=NU,
            chunk=CHUNK,
            per_boundary_box=True,
        )
        fragments = candidate_fragments
        original_text = candidate_text
        fragment_spans = candidate_spans
        original_ids = candidate_ids
        offsets = candidate_offsets
        owned_span = candidate_owned_span
        retained_span = (
            fragment_spans[retained_start][0],
            fragment_spans[retained_end - 1][1],
        )
        forget_positions = candidate_forget
        feasibility = candidate_feasibility

    selected_owned_prefix_starts: set[int] = set()
    for pair_start in owned_prefix_candidates:
        if all(bool(item.get("feasible")) for item in feasibility):
            break
        selected_owned_prefix_starts.add(pair_start)
        candidate_prefix = _fragment_for_pair_starts(
            target,
            owned_session,
            selected_owned_prefix_starts,
            answer_indices=(),
        )
        candidate_fragments = (
            *earlier_fragments,
            candidate_prefix,
            owned_fragment,
            *retained_fragments,
            *tail_fragments,
        )
        candidate_text, candidate_spans = _render_fragments(
            candidate_fragments
        )
        candidate_ids, candidate_offsets = base._tokenizer_payload(
            tokenizer,
            candidate_text,
        )
        if len(candidate_ids) + query_reserve > MAXIMUM_CONTEXT_TOKENS:
            selected_owned_prefix_starts.remove(pair_start)
            continue
        candidate_owned_index = len(earlier_fragments) + 1
        candidate_owned_span = candidate_spans[candidate_owned_index]
        candidate_forget = _positions_for_span(
            candidate_offsets,
            candidate_owned_span,
        )
        _, candidate_feasibility = fixed_c_feasibility(
            len(candidate_ids),
            candidate_forget,
            nu=NU,
            chunk=CHUNK,
            per_boundary_box=True,
        )
        owned_session_prefix = [candidate_prefix]
        fragments = candidate_fragments
        original_text = candidate_text
        fragment_spans = candidate_spans
        original_ids = candidate_ids
        offsets = candidate_offsets
        owned_index = candidate_owned_index
        retained_start = owned_index + 1
        retained_end = retained_start + len(retained_fragments)
        owned_span = candidate_owned_span
        retained_span = (
            fragment_spans[retained_start][0],
            fragment_spans[retained_end - 1][1],
        )
        forget_positions = candidate_forget
        feasibility = candidate_feasibility

    surviving_fragments = tuple(
        fragment
        for index, fragment in enumerate(fragments)
        if index != owned_index
    )
    raw_omitted_text, _ = _render_fragments(surviving_fragments)
    literal_omission = (
        original_text[: owned_span[0]] + original_text[owned_span[1] :]
    )
    if literal_omission != raw_omitted_text:
        raise ManifestError("raw omission does not re-render exactly")

    raw_omitted_ids, _ = base._tokenizer_payload(
        tokenizer,
        raw_omitted_text,
    )
    positional_drop = base._remove_positions(
        original_ids,
        forget_positions,
    )
    if positional_drop != tuple(raw_omitted_ids):
        raise ManifestError(
            "fresh raw omission differs from the complete-turn token drop"
        )
    retained_positions = tuple(
        position
        for position in _positions_for_span(offsets, retained_span)
        if position not in set(forget_positions)
    )
    if not retained_positions:
        raise ManifestError("retained registered turns map to no model tokens")
    edited_retained_positions = base._shift_positions(
        retained_positions,
        forget_positions,
    )
    retained_before = tuple(
        original_ids[position] for position in retained_positions
    )
    retained_after = tuple(
        raw_omitted_ids[position]
        for position in edited_retained_positions
    )
    if retained_before != retained_after:
        raise ManifestError("raw omission changes retained token rows")

    tokens_after = len(original_ids) - max(forget_positions) - 1
    if tokens_after < MINIMUM_TOKENS_AFTER_OWNED:
        raise ManifestError("registered chat violates frozen suffix distance")
    if len(original_ids) + query_reserve > MAXIMUM_CONTEXT_TOKENS:
        raise ManifestError("registered chat exceeds the 4,096-token budget")

    context = TokenizedContext(
        original_text=original_text,
        edited_text=raw_omitted_text,
        original_token_ids=tuple(original_ids),
        edited_token_ids=tuple(raw_omitted_ids),
        offset_mapping=tuple(offsets),
        forget_positions=tuple(forget_positions),
        deletion_ranges=base._contiguous_ranges(forget_positions),
        retained_positions=retained_positions,
        edited_retained_positions=edited_retained_positions,
        owned_character_span=owned_span,
        answer_character_span=_casefold_span(
            original_text,
            target.answer,
            owned_span,
        ),
        retained_character_span=retained_span,
    )
    baseline = spec.get("baseline") or {}
    record_id = base.stable_identifier(
        "longmemeval-constrained-chat-v1",
        base.DATASET_REVISION,
        partition,
        str(baseline.get("record_id") or ""),
        CHAT_TOKENIZER_REVISION,
        CONSTRAINED_INSTRUCTION,
    )
    return RehydratedChatRecord(
        record_id=record_id,
        partition=partition,
        target=target,
        retained=retained,
        earlier_evidence=tuple(earlier_fragments),
        owned_session_prefix=tuple(owned_session_prefix),
        owned_latest_evidence=owned_fragment,
        retained_evidence=retained_fragments,
        tail_evidence=tuple(tail_fragments),
        context=context,
        raw_omitted_token_ids=tuple(raw_omitted_ids),
        probes=probes,
        query_token_reserve=query_reserve,
        fixed_c_diagnostics=tuple(dict(item) for item in feasibility),
    )


def _fragment_descriptor(fragment: SessionFragment) -> dict[str, Any]:
    return {
        "source_id": fragment.source_id,
        "question_id": fragment.question_id,
        "session_id": fragment.session.session_id,
        "source_session_sha256": fragment.session.source_session_sha256,
        "timestamp": (
            None
            if fragment.session.timestamp is None
            else fragment.session.timestamp.isoformat(timespec="minutes")
        ),
        "date_text_sha256": base.text_sha256(
            fragment.session.date_text
        ),
        "official_turn_indices": list(fragment.turn_indices),
        "answer_bearing_turn_indices": list(
            fragment.answer_turn_indices
        ),
        "official_role_sequence": [
            (
                "assistant"
                if turn.role.strip().casefold() == "model"
                else turn.role.strip().casefold()
            )
            for turn in fragment.turns
        ],
        "source_turn_sha256": [
            turn.source_turn_sha256 for turn in fragment.turns
        ],
        "complete_user_assistant_rounds": True,
    }


def _probe_descriptor(probe: ChatProbe) -> dict[str, Any]:
    return {
        "probe_id": probe.probe_id,
        "kind": probe.kind,
        "question_sha256": base.text_sha256(probe.question),
        "answer_sha256": base.text_sha256(probe.answer),
        "prompt_sha256": base.text_sha256(probe.prompt_text),
        "target_token_count": len(probe.target_token_ids),
        "target_token_ids_sha256": base.token_ids_sha256(
            probe.target_token_ids
        ),
        "constrained_instruction_sha256": base.text_sha256(
            CONSTRAINED_INSTRUCTION
        ),
    }


def _context_descriptor(record: RehydratedChatRecord) -> dict[str, Any]:
    context = record.context
    retained_before = tuple(
        context.original_token_ids[position]
        for position in context.retained_positions
    )
    retained_after = tuple(
        context.edited_token_ids[position]
        for position in context.edited_retained_positions
    )
    feasibility = [dict(item) for item in record.fixed_c_diagnostics]
    return {
        "original_text_sha256": base.text_sha256(context.original_text),
        "raw_omitted_text_sha256": base.text_sha256(
            context.edited_text
        ),
        "original_token_count": len(context.original_token_ids),
        "raw_omitted_token_count": len(context.edited_token_ids),
        "original_token_ids_sha256": base.token_ids_sha256(
            context.original_token_ids
        ),
        "raw_omitted_token_ids_sha256": base.token_ids_sha256(
            record.raw_omitted_token_ids
        ),
        "raw_omission_retokenizes_exactly": True,
        "raw_omission_equals_complete_turn_position_drop": True,
        "query_token_reserve": record.query_token_reserve,
        "runtime_total_token_bound": (
            len(context.original_token_ids) + record.query_token_reserve
        ),
        "runtime_context_ceiling": MAXIMUM_CONTEXT_TOKENS,
        "tokens_strictly_after_owned": (
            len(context.original_token_ids)
            - max(context.forget_positions)
            - 1
        ),
        "ownership": {
            "owned_character_span": list(context.owned_character_span),
            "answer_character_span": list(
                context.answer_character_span
            ),
            "forget_positions": list(context.forget_positions),
            "deletion_ranges": [
                {"start": start, "end": end}
                for start, end in context.deletion_ranges
            ],
            "mapping": (
                "positive character overlap from fast-tokenizer offsets"
            ),
            "offset_mapping_sha256": _payload_sha256(
                [list(offset) for offset in context.offset_mapping]
            ),
            "all_selected_official_round_turns_owned": True,
        },
        "retained": {
            "character_span": list(context.retained_character_span),
            "token_positions": list(context.retained_positions),
            "edited_token_positions": list(
                context.edited_retained_positions
            ),
            "original_token_ids_sha256": base.token_ids_sha256(
                retained_before
            ),
            "edited_token_ids_sha256": base.token_ids_sha256(
                retained_after
            ),
            "tokens_preserved_exactly": retained_before == retained_after,
        },
        "fixed_c_reference": {
            "reference": "fixed-C retained-key refit",
            "nu": NU,
            "chunk": CHUNK,
            "per_boundary_box": True,
            "affected_boundaries": len(feasibility),
            "all_affected_boundaries_feasible": all(
                bool(item.get("feasible")) for item in feasibility
            ),
            "infeasible_reference_policy": (
                "retain record and disclose full-repack requirement"
            ),
            "feasibility": feasibility,
        },
    }


def _public_record(
    runtime: RehydratedChatRecord,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_id": runtime.record_id,
        "baseline": copy.deepcopy(dict(spec.get("baseline") or {})),
        "target": copy.deepcopy(dict(spec.get("target") or {})),
        "retained_probe": copy.deepcopy(
            dict(spec.get("retained_probe") or {})
        ),
        "tail_session_references": copy.deepcopy(
            list(spec.get("tail_session_references") or ())
        ),
        "session_layout": {
            "policy": (
                "frozen evidence rounds plus deterministic nonconflicting "
                "complete rounds from the same frozen sessions"
            ),
            "earlier_target_evidence": [
                _fragment_descriptor(fragment)
                for fragment in runtime.earlier_evidence
            ],
            "latest_target_session_prefix": [
                _fragment_descriptor(fragment)
                for fragment in runtime.owned_session_prefix
            ],
            "owned_latest_target_evidence": _fragment_descriptor(
                runtime.owned_latest_evidence
            ),
            "retained_evidence": [
                _fragment_descriptor(fragment)
                for fragment in runtime.retained_evidence
            ],
            "deterministic_tail": [
                _fragment_descriptor(fragment)
                for fragment in runtime.tail_evidence
            ],
        },
        "chat_serialization": _chat_serialization(),
        "context": _context_descriptor(runtime),
        "probes": [
            _probe_descriptor(probe) for probe in runtime.probes
        ],
    }


def _source_descriptors(
    examples: Sequence[base.LongMemEvalExample],
) -> list[dict[str, Any]]:
    return [
        base._example_descriptor(example)
        for example in sorted(examples, key=lambda item: item.source_id)
    ]


def _specs_for_development(artifacts: _ArtifactSet) -> list[dict[str, Any]]:
    pairs = sorted(
        zip(
            artifacts.development_descriptors,
            artifacts.development,
        ),
        key=lambda pair: str(pair[0]["integrity_sha256"]),
    )
    specs = [
        _record_spec(
            record,
            artifact_integrity=str(descriptor["integrity_sha256"]),
        )
        for descriptor, manifest in pairs
        for record in manifest["records"]
    ]
    return sorted(
        specs,
        key=lambda item: (
            str(item["target"]["source_id"]),
            str(item["baseline"]["record_id"]),
        ),
    )


def _specs_for_confirmation(artifacts: _ArtifactSet) -> list[dict[str, Any]]:
    integrity = str(
        artifacts.confirmation_descriptor["integrity_sha256"]
    )
    return [
        _record_spec(record, artifact_integrity=integrity)
        for record in artifacts.confirmation["records"]
    ]


def _manifest_body(
    *,
    partition: str,
    records: Sequence[Mapping[str, Any]],
    policy_lock: Mapping[str, Any],
) -> dict[str, Any]:
    role_ids = all_role_source_ids(({"records": list(records)},))
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "benchmark_label": BENCHMARK_LABEL,
        "contains_source_text": False,
        "partition": partition,
        "provenance": {
            **base._official_provenance(),
            "adaptation": (
                "frozen LongMemEval V1 evidence rendered as constrained "
                "registered Gemma chat"
            ),
            "official_longmemeval_leaderboard_score": False,
        },
        "source_inventory": copy.deepcopy(
            dict(policy_lock["source_inventory"])
        ),
        "policy_lock": copy.deepcopy(dict(policy_lock)),
        "tokenizer": {
            "model_id": CHAT_TOKENIZER_ID,
            "revision": CHAT_TOKENIZER_REVISION,
            "fast_offset_mapping_required": True,
        },
        "model": _model_config(),
        "training_free_graft": _graft_config(),
        "chat_serialization": _chat_serialization(),
        "evaluation_contract": _evaluation_contract(),
        "selection_policy": {
            "source_only": True,
            "model_outputs_used": False,
            "output_based_replacement": False,
            "fixed_before_model_scoring": True,
            "all_selected_records_retained": True,
            "all_role_source_count": len(role_ids),
            "all_role_source_id_sha256": sorted(
                base.text_sha256(source_id) for source_id in role_ids
            ),
            "records": len(records),
        },
        "records": [copy.deepcopy(dict(record)) for record in records],
    }


def build_manifests(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    development_manifests: Sequence[ManifestInput],
    confirmation_manifest: ManifestInput,
    require_pinned_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build deterministic source-free development and confirmation manifests."""

    if getattr(tokenizer, "is_fast", True) is not True:
        raise ManifestError("fast tokenizer offset mappings are required")
    artifacts = _resolve_artifacts(
        development_manifests,
        confirmation_manifest,
        require_pinned=require_pinned_artifacts,
    )
    policy_lock = _build_policy_lock_from_artifacts(
        artifacts,
        reusable_artifacts_pinned=require_pinned_artifacts,
    )
    examples = base.extract_longmemeval_examples(rows)
    descriptors = _source_descriptors(examples)
    if descriptors != list(artifacts.full_source_rows):
        raise ManifestError("rehydrated full LongMemEval source inventory drifted")
    examples_by_source = {example.source_id: example for example in examples}
    if len(examples_by_source) != len(examples):
        raise ManifestError("rehydrated LongMemEval source IDs are duplicated")

    development_records = []
    for spec in _specs_for_development(artifacts):
        runtime = _prepare_record(
            spec,
            examples_by_source,
            tokenizer,
            partition=DEVELOPMENT_PARTITION,
        )
        development_records.append(_public_record(runtime, spec))
    confirmation_records = []
    for spec in _specs_for_confirmation(artifacts):
        runtime = _prepare_record(
            spec,
            examples_by_source,
            tokenizer,
            partition=CONFIRMATION_PARTITION,
        )
        confirmation_records.append(_public_record(runtime, spec))

    development = freeze_manifest(
        _manifest_body(
            partition=DEVELOPMENT_PARTITION,
            records=development_records,
            policy_lock=policy_lock,
        )
    )
    confirmation = freeze_manifest(
        _manifest_body(
            partition=CONFIRMATION_PARTITION,
            records=confirmation_records,
            policy_lock=policy_lock,
        )
    )
    validate_partition_pair(development, confirmation)
    return development, confirmation


def _expected_ranges(positions: Sequence[int]) -> list[dict[str, int]]:
    return [
        {"start": start, "end": end}
        for start, end in base._contiguous_ranges(positions)
    ]


def _validate_without_integrity(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema") != SCHEMA
        or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
        or manifest.get("benchmark_label") != BENCHMARK_LABEL
        or manifest.get("contains_source_text") is not False
        or manifest.get("partition") not in PARTITIONS
    ):
        raise ManifestError("unsupported LongMemEval constrained-chat manifest")
    leaked = _FORBIDDEN_SOURCE_KEYS.intersection(_walk_keys(manifest))
    if leaked:
        raise ManifestError(
            "manifest contains source-text fields: "
            + ", ".join(sorted(leaked))
        )
    lock = manifest.get("policy_lock")
    if not isinstance(lock, Mapping):
        raise ManifestError("LongMemEval chat policy lock is missing")
    validate_policy_lock(lock)
    if (
        manifest.get("source_inventory") != lock.get("source_inventory")
        or manifest.get("tokenizer")
        != {
            "model_id": CHAT_TOKENIZER_ID,
            "revision": CHAT_TOKENIZER_REVISION,
            "fast_offset_mapping_required": True,
        }
        or manifest.get("model") != _model_config()
        or manifest.get("training_free_graft") != _graft_config()
        or manifest.get("chat_serialization") != _chat_serialization()
        or manifest.get("evaluation_contract") != _evaluation_contract()
    ):
        raise ManifestError("LongMemEval constrained-chat policy drifted")

    policy = manifest.get("selection_policy") or {}
    records = manifest.get("records")
    if (
        policy.get("source_only") is not True
        or policy.get("model_outputs_used") is not False
        or policy.get("output_based_replacement") is not False
        or policy.get("fixed_before_model_scoring") is not True
        or policy.get("all_selected_records_retained") is not True
        or not isinstance(records, list)
        or not records
        or len(records) != int(policy.get("records", -1))
    ):
        raise ManifestError("source-only frozen selection policy drifted")
    record_ids = [str(record.get("record_id") or "") for record in records]
    if not all(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ManifestError("LongMemEval chat record IDs are invalid")
    target_ids = [
        str((record.get("target") or {}).get("source_id") or "")
        for record in records
    ]
    if not all(target_ids) or len(target_ids) != len(set(target_ids)):
        raise ManifestError("LongMemEval chat target sources are invalid")

    role_ids = all_role_source_ids((manifest,))
    role_hashes = sorted(
        base.text_sha256(source_id) for source_id in role_ids
    )
    if (
        len(role_ids) != int(policy.get("all_role_source_count", -1))
        or role_hashes != policy.get("all_role_source_id_sha256")
    ):
        raise ManifestError("manifest all-role source census drifted")
    development_hashes = set(
        lock["partition_policy"]["development_source_id_sha256"]
    )
    if manifest["partition"] == DEVELOPMENT_PARTITION:
        if set(role_hashes) != development_hashes:
            raise ManifestError("development all-role ownership differs from lock")
    elif development_hashes.intersection(role_hashes):
        raise ManifestError("confirmation overlaps development source ownership")

    allowed_artifacts = {
        str(item["integrity_sha256"])
        for item in (
            lock["artifacts"]["development"]
            if manifest["partition"] == DEVELOPMENT_PARTITION
            else (lock["artifacts"]["confirmation"],)
        )
    }
    if len(records) != int(
        lock["partition_policy"][
            f"{manifest['partition']}_records"
        ]
    ):
        raise ManifestError("partition record count differs from policy lock")

    for record in records:
        baseline = record.get("baseline") or {}
        if baseline.get("manifest_integrity_sha256") not in allowed_artifacts:
            raise ManifestError("record is not bound to a frozen source artifact")
        target = record.get("target") or {}
        retained = record.get("retained_probe") or {}
        if (
            target.get("question_type") != base.KNOWLEDGE_UPDATE_TYPE
            or str(target.get("question_id") or "").casefold().endswith("_abs")
            or target.get("previous_answer_available_from_official_schema")
            is not False
            or target.get("previous_answer_invented") is not False
            or target.get("gate_uses_previous_answer") is not False
            or target.get("source_id") == retained.get("source_id")
            or retained.get("distinct_official_non_abstention_row") is not True
            or len(retained.get("evidence_session_ids") or ()) != 1
        ):
            raise ManifestError("target or retained ownership policy drifted")
        layout = record.get("session_layout") or {}
        owned = layout.get("owned_latest_target_evidence") or {}
        fragments = [
            *(layout.get("earlier_target_evidence") or ()),
            *(layout.get("latest_target_session_prefix") or ()),
            owned,
            *(layout.get("retained_evidence") or ()),
            *(layout.get("deterministic_tail") or ()),
        ]
        if not fragments or any(
            fragment.get("complete_user_assistant_rounds") is not True
            or not fragment.get("official_turn_indices")
            or fragment.get("official_role_sequence")
            != ["user", "assistant"]
            * (len(fragment.get("official_role_sequence") or ()) // 2)
            for fragment in fragments
        ):
            raise ManifestError("official user-assistant session layout drifted")
        if record.get("chat_serialization") != _chat_serialization():
            raise ManifestError("record chat serialization drifted")

        context = record.get("context") or {}
        ownership = context.get("ownership") or {}
        positions = ownership.get("forget_positions")
        if (
            not isinstance(positions, list)
            or not positions
            or positions != sorted(set(int(value) for value in positions))
            or ownership.get("deletion_ranges")
            != _expected_ranges(positions)
            or ownership.get("all_selected_official_round_turns_owned")
            is not True
            or context.get("raw_omission_retokenizes_exactly") is not True
            or context.get(
                "raw_omission_equals_complete_turn_position_drop"
            )
            is not True
        ):
            raise ManifestError("registered-turn token ownership drifted")
        original = int(context.get("original_token_count", -1))
        omitted = int(context.get("raw_omitted_token_count", -1))
        reserve = int(context.get("query_token_reserve", -1))
        runtime_bound = int(context.get("runtime_total_token_bound", -1))
        after = int(context.get("tokens_strictly_after_owned", -1))
        if (
            original - len(positions) != omitted
            or reserve < GREEDY_GENERATION_TOKEN_RESERVE
            or runtime_bound != original + reserve
            or runtime_bound > MAXIMUM_CONTEXT_TOKENS
            or context.get("runtime_context_ceiling")
            != MAXIMUM_CONTEXT_TOKENS
            or after != original - positions[-1] - 1
            or after < MINIMUM_TOKENS_AFTER_OWNED
        ):
            raise ManifestError("registered chat token geometry drifted")
        retained_tokens = context.get("retained") or {}
        if (
            retained_tokens.get("tokens_preserved_exactly") is not True
            or retained_tokens.get("original_token_ids_sha256")
            != retained_tokens.get("edited_token_ids_sha256")
        ):
            raise ManifestError("raw omission changes retained tokens")
        fixed = context.get("fixed_c_reference") or {}
        feasibility = fixed.get("feasibility")
        if (
            fixed.get("reference") != "fixed-C retained-key refit"
            or fixed.get("nu") != NU
            or fixed.get("chunk") != CHUNK
            or fixed.get("per_boundary_box") is not True
            or fixed.get("infeasible_reference_policy")
            != "retain record and disclose full-repack requirement"
            or not isinstance(feasibility, list)
            or int(fixed.get("affected_boundaries", -1))
            != len(feasibility)
            or fixed.get("all_affected_boundaries_feasible")
            is not all(bool(item.get("feasible")) for item in feasibility)
        ):
            raise ManifestError("fixed-C reference drifted")
        probes = record.get("probes")
        if (
            not isinstance(probes, list)
            or [probe.get("probe_id") for probe in probes]
            != ["target_current", "retained"]
            or [probe.get("kind") for probe in probes]
            != ["deleted", "retained"]
            or any(
                probe.get("constrained_instruction_sha256")
                != base.text_sha256(CONSTRAINED_INSTRUCTION)
                or int(probe.get("target_token_count", 0)) < 1
                for probe in probes
            )
        ):
            raise ManifestError("constrained LongMemEval probes drifted")


def freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy, validate, and attach record/root SHA-256 integrity."""

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
    """Reject source, partition, serialization, geometry, or digest drift."""

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


def validate_partition_pair(
    development: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> None:
    validate_manifest(development)
    validate_manifest(confirmation)
    if (
        development.get("partition") != DEVELOPMENT_PARTITION
        or confirmation.get("partition") != CONFIRMATION_PARTITION
        or development.get("policy_lock") != confirmation.get("policy_lock")
    ):
        raise ManifestError("development/confirmation policy locks differ")
    development_ids = all_role_source_ids((development,))
    confirmation_ids = all_role_source_ids((confirmation,))
    if development_ids.intersection(confirmation_ids):
        raise ManifestError("chat partitions overlap across source roles")


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[RehydratedChatRecord, ...]:
    """Rebuild source text and reject source, session, or token drift."""

    validate_manifest(manifest)
    examples = base.extract_longmemeval_examples(rows)
    descriptors = _source_descriptors(examples)
    inventory = manifest["source_inventory"]
    if (
        len(descriptors) != int(inventory["full_row_count"])
        or _payload_sha256(descriptors)
        != inventory["full_descriptors_sha256"]
    ):
        raise ManifestError("rehydrated LongMemEval source inventory drifted")
    examples_by_source = {example.source_id: example for example in examples}
    if len(examples_by_source) != len(examples):
        raise ManifestError("rehydrated LongMemEval source IDs are duplicated")
    records = []
    for stored in manifest["records"]:
        spec = {
            "baseline": copy.deepcopy(dict(stored["baseline"])),
            "target": copy.deepcopy(dict(stored["target"])),
            "retained_probe": copy.deepcopy(
                dict(stored["retained_probe"])
            ),
            "tail_session_references": copy.deepcopy(
                list(stored["tail_session_references"])
            ),
        }
        runtime = _prepare_record(
            spec,
            examples_by_source,
            tokenizer,
            partition=str(manifest["partition"]),
        )
        regenerated = _public_record(runtime, spec)
        observed = copy.deepcopy(dict(stored))
        observed.pop("record_integrity", None)
        if regenerated != observed:
            raise ManifestError(
                "rehydrated LongMemEval chat record differs from manifest"
            )
        records.append(runtime)
    return tuple(records)


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


def write_policy_lock(path: str | Path, lock: Mapping[str, Any]) -> None:
    validate_policy_lock(lock)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            lock,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="local pinned oracle JSON; omitted resolves the pinned Hub artifact",
    )
    parser.add_argument(
        "--development-manifest",
        action="append",
        help=(
            "frozen development manifest; repeat for both defaults "
            "(omitted uses the exact pinned pair)"
        ),
    )
    parser.add_argument(
        "--confirmation-manifest",
        default=str(DEFAULT_CONFIRMATION_MANIFEST_PATH),
    )
    parser.add_argument("--development-out", required=True)
    parser.add_argument("--confirmation-out", required=True)
    parser.add_argument("--policy-lock-out")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze manifests only; no Gemma model is loaded or scored."""

    args = _parser().parse_args(argv)
    outputs = [
        Path(args.development_out),
        Path(args.confirmation_out),
        *(
            [Path(args.policy_lock_out)]
            if args.policy_lock_out
            else []
        ),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "output exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    development_paths = (
        tuple(args.development_manifest)
        if args.development_manifest
        else DEFAULT_DEVELOPMENT_MANIFEST_PATHS
    )
    rows = base.load_pinned_longmemeval_rows(args.data_path)
    tokenizer = base.load_offset_tokenizer(
        CHAT_TOKENIZER_ID,
        revision=CHAT_TOKENIZER_REVISION,
    )
    development, confirmation = build_manifests(
        rows,
        tokenizer,
        development_manifests=development_paths,
        confirmation_manifest=args.confirmation_manifest,
        require_pinned_artifacts=True,
    )
    write_manifest(args.development_out, development)
    write_manifest(args.confirmation_out, confirmation)
    if args.policy_lock_out:
        write_policy_lock(
            args.policy_lock_out,
            development["policy_lock"],
        )
    print(
        "wrote "
        f"{len(development['records'])} development and "
        f"{len(confirmation['records'])} confirmation chat records",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

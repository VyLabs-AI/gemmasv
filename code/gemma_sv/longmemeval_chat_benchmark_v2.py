"""Deletion-safe LongMemEval V1 registered-chat confirmation protocol v2.

V2 treats every source used by the original development manifests or the
already-scored v1 chat confirmation as exposed.  It deterministically builds a
new, source-disjoint confirmation cohort from the pinned LongMemEval oracle.
The manifest is source-text free and is never selected using model output.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from gemma_sv import longmemeval_chat_benchmark as chat_v1
from gemma_sv import longmemeval_deletion_benchmark as base


SCHEMA = "gemma-sv-longmemeval-constrained-chat-v2"
SCHEMA_VERSION = 2
POLICY_LOCK_SCHEMA = "gemma-sv-longmemeval-chat-policy-lock-v2"
CENSUS_SCHEMA = "gemma-sv-longmemeval-chat-eligibility-census-v2"
BENCHMARK_LABEL = "LongMemEval V1 deletion-safe constrained-chat v2"
PARTITION = "confirmation"

MODEL_ID = chat_v1.MODEL_ID
MODEL_REVISION = chat_v1.MODEL_REVISION
TOKENIZER_ID = chat_v1.CHAT_TOKENIZER_ID
TOKENIZER_REVISION = chat_v1.CHAT_TOKENIZER_REVISION
CONSTRAINED_INSTRUCTION = chat_v1.CONSTRAINED_INSTRUCTION

WINDOW = 1024
MINIMUM_TOKENS_AFTER_OWNED = 1152
MAXIMUM_CONTEXT_TOKENS = 8192
GREEDY_GENERATION_TOKEN_RESERVE = chat_v1.GREEDY_GENERATION_TOKEN_RESERVE
NU = chat_v1.NU
CHUNK = chat_v1.CHUNK
SOLVER_SEED = chat_v1.SOLVER_SEED
SELECTION_SEED = 20_260_823
DEFAULT_CONFIRMATION_RECORDS = 16
TAIL_MINIMUM_SCHEDULE = (1152, 1536, 2048, 3072, 4096, 6144)
MINIMUM_TARGET_PRESENT_MINUS_RAW_FULL_SEQUENCE_MEAN_LOGPROB_NATS = 0.05
MAXIMUM_TARGET_PRESENT_FIRST_TOKEN_RANK = 10
MAXIMUM_RETAINED_FIRST_TOKEN_RANK_IN_PRESENT_AND_RAW = 10
_PACKAGING_REJECTION_REASONS = frozenset(
    {
        "insufficient_post_owned_geometry",
        "context_ceiling_infeasible",
        "retained_reference_infeasible",
        "owned_round_infeasible",
        "source_packaging_infeasible",
        "source_packaging_manifesterror",
        "source_packaging_valueerror",
        "source_packaging_keyerror",
        "source_packaging_indexerror",
        "source_packaging_lookuperror",
    }
)
_AUDIT_REASONS = frozenset(
    {
        "source_only_greedy_disjoint_selection",
        "cohort_capacity_reached",
        "target_source_reserved_by_prior_record",
        *_PACKAGING_REJECTION_REASONS,
        *{
            f"cohort_disjointness_{reason}"
            for reason in _PACKAGING_REJECTION_REASONS
        },
    }
)

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
DEFAULT_V1_DEVELOPMENT_PATH = (
    PACKAGE / "benchmarks" / "longmemeval_chat_development_v1.json"
)
DEFAULT_V1_CONFIRMATION_PATH = (
    PACKAGE / "benchmarks" / "longmemeval_chat_confirmation_v1.json"
)
DEFAULT_DEVELOPMENT_MANIFEST_PATHS = (
    DEFAULT_V1_DEVELOPMENT_PATH,
)

PINNED_V1_DEVELOPMENT_FILE_SHA256 = (
    "2eb14e3af4f376dbf8858a7fa1ba5700de550604e7187afb363c03c727cf1f9f"
)
PINNED_V1_DEVELOPMENT_INTEGRITY_SHA256 = (
    "2e49e26ef6911024830594a88e186b2d77229a9a3feed3630ce83da074e232a7"
)
PINNED_V1_CONFIRMATION_FILE_SHA256 = (
    "7e9f484c4eee8cceecec79bfc0721e1cfc7100092d29f5b29d1e96eca478bae1"
)
PINNED_V1_CONFIRMATION_INTEGRITY_SHA256 = (
    "48eb12371e677bc18732f900049dcdfc5791f31344a1f4dfbcfcf06bcfe07ce5"
)
PINNED_EXPOSED_SOURCE_COUNT = 65
PINNED_FULL_ROW_COUNT = 500
PINNED_FULL_DESCRIPTORS_SHA256 = (
    "4294ca8f7aaf22fc0214a3d9ed6860da83ad1b947337a94aff7f742e5084c839"
)
PINNED_V2_ELIGIBLE_SOURCE_ONLY_CANDIDATES = 32
PINNED_V2_ALL_ROLE_SOURCE_COUNT = 66
PINNED_V2_CONFIRMATION_INTEGRITY_SHA256 = (
    "4821ecf3ef37c004da1142199e6a84de5dd183b78059136347897f96afa21bdb"
)
PINNED_V2_POLICY_LOCK_SHA256 = (
    "02d6d85641a93f3c756455c7e540040e5349b8ef4327ee16da60fda38f101aca"
)
PINNED_V2_CENSUS_INTEGRITY_SHA256 = (
    "17e915d941d16b0b73e2e72dc35f973ea8d110e27dedeca92238f37f759d8b0e"
)

ManifestInput = Mapping[str, Any] | str | Path
ManifestError = base.ManifestError

_SELECTION_CONTRACT = {
    "schema": SCHEMA,
    "selection_seed": SELECTION_SEED,
    "tail_minimum_schedule": list(TAIL_MINIMUM_SCHEDULE),
    "minimum_tokens_strictly_after_owned": MINIMUM_TOKENS_AFTER_OWNED,
    "maximum_context_tokens": MAXIMUM_CONTEXT_TOKENS,
    "one_record_per_source_across_all_roles": True,
    "selection_uses_model_outputs": False,
}


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


SELECTION_CONTRACT_SHA256 = _payload_sha256(_SELECTION_CONTRACT)


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
        raise ManifestError(f"expected a JSON object: {path}")
    return payload, path


def _source_hashes(source_ids: Iterable[str]) -> list[str]:
    return sorted(base.text_sha256(source_id) for source_id in set(source_ids))


@contextmanager
def _v2_geometry() -> Iterator[None]:
    """Temporarily configure reused source-only packers for the v2 ceiling."""

    overrides = (
        (chat_v1, "MAXIMUM_CONTEXT_TOKENS", MAXIMUM_CONTEXT_TOKENS),
        (
            chat_v1,
            "MINIMUM_TOKENS_AFTER_OWNED",
            MINIMUM_TOKENS_AFTER_OWNED,
        ),
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
class ExposureArtifacts:
    development_manifests: tuple[dict[str, Any], ...]
    v1_confirmation: dict[str, Any]
    development_source_ids: frozenset[str]
    v1_confirmation_source_ids: frozenset[str]
    exposed_source_ids: frozenset[str]
    descriptors: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CandidatePreparation:
    spec: dict[str, Any]
    runtime: chat_v1.RehydratedChatRecord
    public_record: dict[str, Any]
    role_source_ids: frozenset[str]
    base_tail_minimum_tokens: int


def _baseline_role_source_ids(manifest: Mapping[str, Any]) -> set[str]:
    return chat_v1.all_role_source_ids((manifest,))


def _exposure_descriptor(
    *,
    role: str,
    manifest: Mapping[str, Any],
    path: Path | None,
    source_ids: set[str],
    scored: bool,
) -> dict[str, Any]:
    integrity = manifest.get("integrity") or {}
    descriptor = {
        "role": role,
        "schema": str(manifest.get("schema") or ""),
        "records": len(manifest.get("records") or ()),
        "integrity_sha256": str(integrity.get("sha256") or ""),
        "all_role_source_count": len(source_ids),
        "all_role_source_ids_sha256": _payload_sha256(
            _source_hashes(source_ids)
        ),
        "treated_as_development_exposure": True,
        "model_scoring_previously_performed": scored,
    }
    if path is not None:
        descriptor["artifact_path"] = str(path.relative_to(WORKSPACE))
        descriptor["file_sha256"] = _file_sha256(path)
    return descriptor


def resolve_exposure_artifacts(
    development_manifests: Sequence[ManifestInput] = (
        DEFAULT_DEVELOPMENT_MANIFEST_PATHS
    ),
    v1_confirmation_manifest: ManifestInput = DEFAULT_V1_CONFIRMATION_PATH,
    *,
    require_pinned: bool = False,
) -> ExposureArtifacts:
    """Validate and combine every pre-v2 source exposure across all roles."""

    if not development_manifests:
        raise ManifestError("at least one development exposure is required")
    loaded_development: list[dict[str, Any]] = []
    development_descriptors: list[dict[str, Any]] = []
    development_ids: set[str] = set()
    for source in development_manifests:
        manifest, path = _load_json(source)
        chat_v1.validate_manifest(manifest)
        if manifest.get("partition") != chat_v1.DEVELOPMENT_PARTITION:
            raise ManifestError("v1 development exposure has the wrong partition")
        role_ids = _baseline_role_source_ids(manifest)
        development_ids.update(role_ids)
        loaded_development.append(manifest)
        development_descriptors.append(
            _exposure_descriptor(
                role="already_scored_v1_development",
                manifest=manifest,
                path=path,
                source_ids=role_ids,
                scored=True,
            )
        )

    v1_confirmation, v1_path = _load_json(v1_confirmation_manifest)
    chat_v1.validate_manifest(v1_confirmation)
    if v1_confirmation.get("partition") != chat_v1.CONFIRMATION_PARTITION:
        raise ManifestError("v1 confirmation exposure has the wrong partition")
    v1_ids = _baseline_role_source_ids(v1_confirmation)
    if development_ids.intersection(v1_ids):
        raise ManifestError("v1 confirmation already overlaps development")
    exposed_ids = development_ids | v1_ids
    v1_descriptor = _exposure_descriptor(
        role="already_scored_v1_confirmation",
        manifest=v1_confirmation,
        path=v1_path,
        source_ids=v1_ids,
        scored=True,
    )

    if require_pinned:
        if (
            len(loaded_development) != 1
            or development_descriptors[0].get("artifact_path")
            != str(DEFAULT_V1_DEVELOPMENT_PATH.relative_to(WORKSPACE))
            or development_descriptors[0].get("file_sha256")
            != PINNED_V1_DEVELOPMENT_FILE_SHA256
            or development_descriptors[0].get("integrity_sha256")
            != PINNED_V1_DEVELOPMENT_INTEGRITY_SHA256
            or len(development_ids) != 32
            or v1_descriptor.get("artifact_path")
            != str(DEFAULT_V1_CONFIRMATION_PATH.relative_to(WORKSPACE))
        ):
            raise ManifestError("tracked v1 development exposure drifted")
        if (
            str((v1_confirmation.get("integrity") or {}).get("sha256") or "")
            != PINNED_V1_CONFIRMATION_INTEGRITY_SHA256
            or v1_descriptor.get("file_sha256")
            != PINNED_V1_CONFIRMATION_FILE_SHA256
            or len(v1_ids) != 33
            or len(exposed_ids) != PINNED_EXPOSED_SOURCE_COUNT
        ):
            raise ManifestError("pinned v1 confirmation exposure drifted")

    return ExposureArtifacts(
        development_manifests=tuple(loaded_development),
        v1_confirmation=v1_confirmation,
        development_source_ids=frozenset(development_ids),
        v1_confirmation_source_ids=frozenset(v1_ids),
        exposed_source_ids=frozenset(exposed_ids),
        descriptors=tuple(
            sorted(
                [*development_descriptors, v1_descriptor],
                key=lambda item: (
                    str(item["role"]),
                    str(item["integrity_sha256"]),
                ),
            )
        ),
    )


def _role_source_ids_for_spec(spec: Mapping[str, Any]) -> frozenset[str]:
    role_ids = {
        str((spec.get("target") or {}).get("source_id") or ""),
        str((spec.get("retained_probe") or {}).get("source_id") or ""),
    }
    role_ids.update(
        str(item.get("source_id") or "")
        for item in spec.get("tail_session_references") or ()
    )
    role_ids.discard("")
    return frozenset(role_ids)


def _v2_record_id(
    spec: Mapping[str, Any],
    partition: str = PARTITION,
) -> str:
    baseline = spec.get("baseline") or {}
    return base.stable_identifier(
        "longmemeval-constrained-chat-v2",
        base.DATASET_REVISION,
        partition,
        str(baseline.get("record_id") or ""),
        TOKENIZER_REVISION,
        CONSTRAINED_INSTRUCTION,
        str(SELECTION_SEED),
    )


def _public_v2_record(
    runtime: chat_v1.RehydratedChatRecord,
    spec: Mapping[str, Any],
    *,
    base_tail_minimum_tokens: int,
) -> dict[str, Any]:
    with _v2_geometry():
        payload = chat_v1._public_record(runtime, spec)
    payload["record_id"] = runtime.record_id
    source_selection = payload.pop("baseline")
    source_selection["selection_contract_sha256"] = SELECTION_CONTRACT_SHA256
    source_selection["base_tail_minimum_tokens"] = base_tail_minimum_tokens
    payload["source_selection"] = source_selection
    context = payload["context"]
    tokens_after = int(context["tokens_strictly_after_owned"])
    fixed_c = context["fixed_c_reference"]
    requires_full_repack = not bool(
        fixed_c["all_affected_boundaries_feasible"]
    )
    context["local_window_safety"] = {
        "true_local_window_tokens": WINDOW,
        "minimum_tokens_strictly_after_owned": (
            MINIMUM_TOKENS_AFTER_OWNED
        ),
        "observed_tokens_strictly_after_owned": tokens_after,
        "owned_round_strictly_outside_local_window_before_query": (
            tokens_after >= WINDOW
        ),
        "selected_span_inside_local_window": False,
    }
    context["full_repack_fallback"] = {
        "required": requires_full_repack,
        "reason": (
            "fixed_c_infeasible" if requires_full_repack else "not_required"
        ),
        "only_permitted_reason": "fixed_c_infeasible",
        "record_retained_without_replacement": True,
    }
    return payload


def _prepare_from_spec(
    spec: Mapping[str, Any],
    examples_by_source: Mapping[str, base.LongMemEvalExample],
    tokenizer: Any,
    *,
    base_tail_minimum_tokens: int,
) -> CandidatePreparation:
    with _v2_geometry():
        runtime = chat_v1._prepare_record(
            spec,
            examples_by_source,
            tokenizer,
            partition=PARTITION,
        )
    runtime = replace(runtime, record_id=_v2_record_id(spec))
    public_record = _public_v2_record(
        runtime,
        spec,
        base_tail_minimum_tokens=base_tail_minimum_tokens,
    )
    context = public_record["context"]
    tokens_after = int(context["tokens_strictly_after_owned"])
    if tokens_after < MINIMUM_TOKENS_AFTER_OWNED:
        raise ManifestError("v2 post-owned distance is insufficient")
    if tokens_after < WINDOW:
        raise ManifestError("v2 owned span remains inside the local window")
    if int(context["runtime_total_token_bound"]) > MAXIMUM_CONTEXT_TOKENS:
        raise ManifestError("v2 runtime token bound exceeds its ceiling")
    return CandidatePreparation(
        spec=copy.deepcopy(dict(spec)),
        runtime=runtime,
        public_record=public_record,
        role_source_ids=_role_source_ids_for_spec(spec),
        base_tail_minimum_tokens=base_tail_minimum_tokens,
    )


def _candidate_spec(
    target: base.LongMemEvalExample,
    retained: base.LongMemEvalExample,
    examples: Sequence[base.LongMemEvalExample],
    tokenizer: Any,
    *,
    base_tail_minimum_tokens: int,
) -> dict[str, Any]:
    with _v2_geometry():
        assembly = base.package_context(
            tokenizer,
            target,
            retained,
            examples,
            seed=SELECTION_SEED,
            minimum_tokens_after_owned=base_tail_minimum_tokens,
            context_ceiling=MAXIMUM_CONTEXT_TOKENS,
        )
        payload = base._record_payload(
            target,
            retained,
            assembly,
            context_ceiling=MAXIMUM_CONTEXT_TOKENS,
        )
    return chat_v1._record_spec(
        payload,
        artifact_integrity=SELECTION_CONTRACT_SHA256,
    )


def _safe_rejection_code(error: Exception) -> str:
    message = str(error).casefold()
    if "suffix distance" in message or "tokens after" in message:
        return "insufficient_post_owned_geometry"
    if "budget" in message or "ceiling" in message:
        return "context_ceiling_infeasible"
    if "retained" in message:
        return "retained_reference_infeasible"
    if "target" in message or "owned" in message:
        return "owned_round_infeasible"
    return f"source_packaging_{type(error).__name__.casefold()}"


def _prepare_candidate(
    target: base.LongMemEvalExample,
    pool: Sequence[base.LongMemEvalExample],
    examples_by_source: Mapping[str, base.LongMemEvalExample],
    tokenizer: Any,
) -> tuple[CandidatePreparation | None, str]:
    try:
        retained = base.deterministic_retained_pair(
            target,
            pool,
            seed=SELECTION_SEED,
        )
    except (ManifestError, ValueError, KeyError, LookupError) as error:
        return None, _safe_rejection_code(error)
    last_error: Exception | None = None
    for tail_minimum in TAIL_MINIMUM_SCHEDULE:
        try:
            spec = _candidate_spec(
                target,
                retained,
                pool,
                tokenizer,
                base_tail_minimum_tokens=tail_minimum,
            )
            return (
                _prepare_from_spec(
                    spec,
                    examples_by_source,
                    tokenizer,
                    base_tail_minimum_tokens=tail_minimum,
                ),
                "eligible",
            )
        except (
            ManifestError,
            ValueError,
            KeyError,
            IndexError,
            LookupError,
        ) as error:
            last_error = error
    if last_error is None:
        return None, "source_packaging_infeasible"
    return None, _safe_rejection_code(last_error)


def _eligible_target(
    example: base.LongMemEvalExample,
    exposed_ids: frozenset[str],
) -> bool:
    return (
        example.source_id not in exposed_ids
        and example.question_type == base.KNOWLEDGE_UPDATE_TYPE
        and not base.is_abstention(example)
        and bool(base.evidence_sessions(example))
    )


def _selection_audit_item(
    target: base.LongMemEvalExample,
    *,
    status: str,
    reason: str,
    preparation: CandidatePreparation | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "target_source_id_sha256": base.text_sha256(target.source_id),
        "status": status,
        "reason": reason,
        "model_outputs_used": False,
    }
    if preparation is not None:
        item.update(
            {
                "record_id": preparation.runtime.record_id,
                "all_role_source_count": len(
                    preparation.role_source_ids
                ),
                "all_role_source_ids_sha256": _payload_sha256(
                    _source_hashes(preparation.role_source_ids)
                ),
                "base_tail_minimum_tokens": (
                    preparation.base_tail_minimum_tokens
                ),
                "tokens_strictly_after_owned": preparation.public_record[
                    "context"
                ]["tokens_strictly_after_owned"],
            }
        )
    return item


def _source_inventory(
    examples: Sequence[base.LongMemEvalExample],
    exposure: ExposureArtifacts,
) -> dict[str, Any]:
    descriptors = chat_v1._source_descriptors(examples)
    return {
        "dataset_id": base.DATASET_ID,
        "dataset_revision": base.DATASET_REVISION,
        "source_artifact_sha256": base.DATASET_ARTIFACT_SHA256,
        "full_row_count": len(descriptors),
        "full_descriptors_sha256": _payload_sha256(descriptors),
        "development_exposed_all_role_source_count": len(
            exposure.exposed_source_ids
        ),
        "development_exposed_source_id_sha256": _source_hashes(
            exposure.exposed_source_ids
        ),
        "untouched_row_count": (
            len(descriptors) - len(exposure.exposed_source_ids)
        ),
        "exposure_artifacts": [
            copy.deepcopy(item) for item in exposure.descriptors
        ],
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
        "window": WINDOW,
        "nu": NU,
        "chunk": CHUNK,
        "solver_seed": SOLVER_SEED,
        "per_boundary_box": True,
        "preserve_prefix_mass": True,
        "readout": "softmax",
        "training_steps": 0,
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
        "behavioral_reference": (
            "fresh registered-chat render with the owned round omitted"
        ),
        "certificate_reference": "fixed-C retained-key refit",
        "owned_unit": (
            "complete official user-assistant round(s) containing has_answer "
            "turns in the latest target evidence session"
        ),
        "retained_unit": (
            "complete official user-assistant round(s) containing has_answer "
            "turns in the retained evidence session"
        ),
        "raw_omission_required": True,
        "minimum_tokens_strictly_after_owned_before_query": (
            MINIMUM_TOKENS_AFTER_OWNED
        ),
        "true_local_window_tokens": WINDOW,
        "selected_span_inside_local_window_permitted": False,
        "full_repack_fallback_only_for_fixed_c_infeasibility": True,
        "one_record_per_source_across_all_roles": True,
        "deletion_request_is_out_of_band_and_answer_free": True,
        "selection_uses_model_outputs": False,
        "no_replacement": True,
        "model_scoring_performed": False,
        "official_longmemeval_leaderboard_score": False,
        "admission_policy": _admission_policy(),
    }


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


def _build_policy_lock(
    *,
    source_inventory: Mapping[str, Any],
    audit: Sequence[Mapping[str, Any]],
    selected: Sequence[CandidatePreparation],
    eligible_count: int,
    requested_records: int,
    candidate_limit: int | None,
) -> dict[str, Any]:
    selected_role_ids: set[str] = set()
    for item in selected:
        selected_role_ids.update(item.role_source_ids)
    lock = {
        "schema": POLICY_LOCK_SCHEMA,
        "status": "locked-before-v2-model-scoring",
        "contains_source_text": False,
        "results_present": False,
        "selection_uses_model_outputs": False,
        "output_based_replacement": False,
        "model": _model_config(),
        "training_free_graft": _graft_config(),
        "chat_serialization": chat_v1._chat_serialization(),
        "evaluation_contract": _evaluation_contract(),
        "admission_policy": _admission_policy(),
        "source_inventory": copy.deepcopy(dict(source_inventory)),
        "partition_policy": {
            "partition": PARTITION,
            "requested_records": requested_records,
            "frozen_records": len(selected),
            "eligible_source_only_candidates": eligible_count,
            "candidate_limit": candidate_limit,
            "selection_seed": SELECTION_SEED,
            "selection_contract_sha256": SELECTION_CONTRACT_SHA256,
            "tail_minimum_schedule": list(TAIL_MINIMUM_SCHEDULE),
            "one_record_per_source_across_all_roles": True,
            "all_role_source_count": len(selected_role_ids),
            "all_role_source_ids_sha256": _payload_sha256(
                _source_hashes(selected_role_ids)
            ),
            "audit_sha256": _payload_sha256(audit),
            "no_replacement": True,
            "all_selected_records_retained": True,
        },
        "exposure_policy": {
            "scored_v1_development_all_roles_exposed": True,
            "scored_v1_confirmation_all_roles_exposed": True,
            "prior_v1_exposures_model_scored": True,
            "confirmation_all_roles_disjoint_from_exposure": True,
            "v2_selection_uses_model_outputs": False,
        },
    }
    lock["lock_sha256"] = _payload_sha256(lock)
    return lock


def _audit_counts(
    audit: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    statuses = {
        "selected": 0,
        "eligible_not_selected": 0,
        "ineligible": 0,
    }
    reasons: dict[str, int] = {}
    for item in audit:
        status = str(item.get("status") or "")
        if status not in statuses:
            raise ManifestError("v2 census audit status is unknown")
        reason = str(item.get("reason") or "")
        if reason not in _AUDIT_REASONS:
            raise ManifestError("v2 census audit reason is unsafe")
        statuses[status] += 1
        reasons[reason] = reasons.get(reason, 0) + 1
    return statuses, dict(sorted(reasons.items()))


def _census_body_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    policy = manifest["selection_policy"]
    lock = manifest["policy_lock"]
    inventory = manifest["source_inventory"]
    records = manifest["records"]
    audit = policy["audit"]
    status_counts, reason_counts = _audit_counts(audit)
    selected_ids = [str(record["record_id"]) for record in records]
    role_ids = all_role_source_ids(manifest)
    suffixes = [
        int(record["context"]["tokens_strictly_after_owned"])
        for record in records
    ]
    bounds = [
        int(record["context"]["runtime_total_token_bound"])
        for record in records
    ]
    fixed_c_infeasible = sum(
        not bool(
            record["context"]["fixed_c_reference"][
                "all_affected_boundaries_feasible"
            ]
        )
        for record in records
    )
    full_repack_required = sum(
        bool(record["context"]["full_repack_fallback"]["required"])
        for record in records
    )
    descriptors = inventory["exposure_artifacts"]
    development_exposed = sum(
        int(item["all_role_source_count"])
        for item in descriptors
        if item["role"] == "already_scored_v1_development"
    )
    confirmation_exposed = sum(
        int(item["all_role_source_count"])
        for item in descriptors
        if item["role"] == "already_scored_v1_confirmation"
    )
    return {
        "schema": CENSUS_SCHEMA,
        "contains_source_text": False,
        "selection_uses_model_outputs": False,
        "candidate_count": len(audit),
        "eligible_source_only_candidates": (
            status_counts["selected"]
            + status_counts["eligible_not_selected"]
        ),
        "ineligible_source_only_candidates": status_counts["ineligible"],
        "selected_records": len(records),
        "requested_records": int(policy["requested_records"]),
        "exposure_counts": {
            "full_oracle_rows": int(inventory["full_row_count"]),
            "scored_v1_development_all_role_sources": development_exposed,
            "scored_v1_confirmation_all_role_sources": confirmation_exposed,
            "total_exposed_all_role_sources": int(
                inventory["development_exposed_all_role_source_count"]
            ),
            "untouched_rows": int(inventory["untouched_row_count"]),
        },
        "audit_status_counts": status_counts,
        "audit_reason_counts": reason_counts,
        "audit": copy.deepcopy(audit),
        "audit_sha256": _payload_sha256(audit),
        "selected_record_ids": selected_ids,
        "selected_record_ids_sha256": _payload_sha256(selected_ids),
        "selected_all_role_source_count": len(role_ids),
        "selected_all_role_source_ids_sha256": _payload_sha256(
            _source_hashes(role_ids)
        ),
        "geometry": {
            "minimum_tokens_strictly_after_owned": min(suffixes),
            "maximum_tokens_strictly_after_owned": max(suffixes),
            "minimum_runtime_total_token_bound": min(bounds),
            "maximum_runtime_total_token_bound": max(bounds),
            "fixed_c_infeasible_records": fixed_c_infeasible,
            "full_repack_required_records": full_repack_required,
            "selected_span_inside_local_window_records": sum(
                bool(
                    record["context"]["local_window_safety"][
                        "selected_span_inside_local_window"
                    ]
                )
                for record in records
            ),
        },
        "policy_bindings": {
            "selection_seed": int(policy["selection_seed"]),
            "candidate_limit": policy["candidate_limit"],
            "selection_contract_sha256": lock["partition_policy"][
                "selection_contract_sha256"
            ],
            "manifest_audit_sha256": lock["partition_policy"][
                "audit_sha256"
            ],
            "no_replacement": bool(lock["partition_policy"]["no_replacement"]),
            "one_record_per_source_across_all_roles": bool(
                lock["partition_policy"][
                    "one_record_per_source_across_all_roles"
                ]
            ),
            "admission_policy": copy.deepcopy(lock["admission_policy"]),
            "admission_policy_sha256": _payload_sha256(
                lock["admission_policy"]
            ),
        },
        "manifest_integrity_sha256": manifest["integrity"]["sha256"],
        "policy_lock_sha256": lock["lock_sha256"],
    }


def build_confirmation_manifest(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    development_manifests: Sequence[ManifestInput] = (
        DEFAULT_DEVELOPMENT_MANIFEST_PATHS
    ),
    v1_confirmation_manifest: ManifestInput = (
        DEFAULT_V1_CONFIRMATION_PATH
    ),
    requested_records: int = DEFAULT_CONFIRMATION_RECORDS,
    candidate_limit: int | None = None,
    require_pinned_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the deterministic v2 confirmation manifest and eligibility census."""

    if requested_records < 1:
        raise ValueError("requested_records must be positive")
    if candidate_limit is not None and candidate_limit < requested_records:
        raise ValueError("candidate_limit cannot be smaller than requested_records")
    if getattr(tokenizer, "is_fast", True) is not True:
        raise ManifestError("fast tokenizer offset mappings are required")

    exposure = resolve_exposure_artifacts(
        development_manifests,
        v1_confirmation_manifest,
        require_pinned=require_pinned_artifacts,
    )
    examples = base.extract_longmemeval_examples(rows)
    descriptors = chat_v1._source_descriptors(examples)
    if require_pinned_artifacts and (
        len(descriptors) != PINNED_FULL_ROW_COUNT
        or _payload_sha256(descriptors) != PINNED_FULL_DESCRIPTORS_SHA256
    ):
        raise ManifestError("pinned LongMemEval oracle inventory drifted")
    examples_by_source = {example.source_id: example for example in examples}
    if len(examples_by_source) != len(examples):
        raise ManifestError("LongMemEval source IDs are duplicated")

    untouched = [
        example
        for example in examples
        if example.source_id not in exposure.exposed_source_ids
    ]
    ordered_candidates = [
        example
        for example in base.deterministic_source_order(
            untouched,
            seed=SELECTION_SEED,
        )
        if _eligible_target(example, exposure.exposed_source_ids)
    ]
    if candidate_limit is not None:
        ordered_candidates = ordered_candidates[:candidate_limit]

    selected: list[CandidatePreparation] = []
    reserved_ids: set[str] = set()
    audit: list[dict[str, Any]] = []
    eligible_count = 0
    for target in ordered_candidates:
        independent, independent_reason = _prepare_candidate(
            target,
            untouched,
            examples_by_source,
            tokenizer,
        )
        if independent is None:
            audit.append(
                _selection_audit_item(
                    target,
                    status="ineligible",
                    reason=independent_reason,
                )
            )
            continue
        eligible_count += 1
        if len(selected) >= requested_records:
            audit.append(
                _selection_audit_item(
                    target,
                    status="eligible_not_selected",
                    reason="cohort_capacity_reached",
                    preparation=independent,
                )
            )
            continue
        if target.source_id in reserved_ids:
            audit.append(
                _selection_audit_item(
                    target,
                    status="eligible_not_selected",
                    reason="target_source_reserved_by_prior_record",
                    preparation=independent,
                )
            )
            continue

        candidate = independent
        if candidate.role_source_ids.intersection(reserved_ids):
            available = [
                example
                for example in untouched
                if example.source_id not in reserved_ids
            ]
            candidate, reason = _prepare_candidate(
                target,
                available,
                examples_by_source,
                tokenizer,
            )
            if candidate is None:
                audit.append(
                    _selection_audit_item(
                        target,
                        status="eligible_not_selected",
                        reason=f"cohort_disjointness_{reason}",
                        preparation=independent,
                    )
                )
                continue
        if candidate.role_source_ids.intersection(reserved_ids):
            raise ManifestError("candidate role sources were not reserved safely")
        selected.append(candidate)
        reserved_ids.update(candidate.role_source_ids)
        audit.append(
            _selection_audit_item(
                target,
                status="selected",
                reason="source_only_greedy_disjoint_selection",
                preparation=candidate,
            )
        )

    if len(selected) != requested_records:
        raise ManifestError(
            "source census cannot support the requested disjoint cohort: "
            f"{len(selected)}/{requested_records}"
        )

    source_inventory = _source_inventory(examples, exposure)
    policy_lock = _build_policy_lock(
        source_inventory=source_inventory,
        audit=audit,
        selected=selected,
        eligible_count=eligible_count,
        requested_records=requested_records,
        candidate_limit=candidate_limit,
    )
    records = [copy.deepcopy(item.public_record) for item in selected]
    body = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "benchmark_label": BENCHMARK_LABEL,
        "contains_source_text": False,
        "partition": PARTITION,
        "provenance": {
            **base._official_provenance(),
            "adaptation": (
                "untouched LongMemEval V1 evidence rendered as deletion-safe "
                "registered Gemma chat"
            ),
            "official_longmemeval_leaderboard_score": False,
        },
        "source_inventory": copy.deepcopy(source_inventory),
        "policy_lock": copy.deepcopy(policy_lock),
        "tokenizer": {
            "model_id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "fast_offset_mapping_required": True,
        },
        "model": _model_config(),
        "training_free_graft": _graft_config(),
        "chat_serialization": chat_v1._chat_serialization(),
        "evaluation_contract": _evaluation_contract(),
        "admission_policy": _admission_policy(),
        "selection_policy": {
            "source_only": True,
            "model_outputs_used": False,
            "output_based_replacement": False,
            "fixed_before_model_scoring": True,
            "no_replacement": True,
            "all_selected_records_retained": True,
            "requested_records": requested_records,
            "records": len(records),
            "eligible_source_only_candidates": eligible_count,
            "candidate_count": len(ordered_candidates),
            "candidate_limit": candidate_limit,
            "selection_seed": SELECTION_SEED,
            "audit": copy.deepcopy(audit),
        },
        "records": records,
    }
    manifest = freeze_manifest(body)
    if (
        require_pinned_artifacts
        and requested_records == DEFAULT_CONFIRMATION_RECORDS
        and candidate_limit is None
        and (
            eligible_count != PINNED_V2_ELIGIBLE_SOURCE_ONLY_CANDIDATES
            or len(all_role_source_ids(manifest))
            != PINNED_V2_ALL_ROLE_SOURCE_COUNT
            or manifest["integrity"]["sha256"]
            != PINNED_V2_CONFIRMATION_INTEGRITY_SHA256
            or policy_lock["lock_sha256"]
            != PINNED_V2_POLICY_LOCK_SHA256
        )
    ):
        raise ManifestError("pinned v2 confirmation freeze drifted")
    census = _census_body_from_manifest(manifest)
    census["integrity"] = {
        "algorithm": "sha256",
        "sha256": _payload_sha256(census),
    }
    if (
        require_pinned_artifacts
        and requested_records == DEFAULT_CONFIRMATION_RECORDS
        and candidate_limit is None
        and census["integrity"]["sha256"]
        != PINNED_V2_CENSUS_INTEGRITY_SHA256
    ):
        raise ManifestError("pinned v2 census freeze drifted")
    return manifest, census


def all_role_source_ids(manifest: Mapping[str, Any]) -> set[str]:
    return chat_v1.all_role_source_ids((manifest,))


def _validate_policy_lock(lock: Mapping[str, Any]) -> None:
    observed = copy.deepcopy(dict(lock))
    lock_sha256 = observed.pop("lock_sha256", None)
    if (
        lock.get("schema") != POLICY_LOCK_SCHEMA
        or lock.get("status") != "locked-before-v2-model-scoring"
        or lock.get("contains_source_text") is not False
        or lock.get("results_present") is not False
        or lock.get("selection_uses_model_outputs") is not False
        or lock.get("output_based_replacement") is not False
        or lock_sha256 != _payload_sha256(observed)
    ):
        raise ManifestError("v2 policy lock drifted")
    if (
        lock.get("model") != _model_config()
        or lock.get("training_free_graft") != _graft_config()
        or lock.get("chat_serialization") != chat_v1._chat_serialization()
        or lock.get("evaluation_contract") != _evaluation_contract()
        or lock.get("admission_policy") != _admission_policy()
    ):
        raise ManifestError("v2 locked execution contract drifted")
    partition = lock.get("partition_policy") or {}
    exposure = lock.get("exposure_policy") or {}
    if (
        partition.get("one_record_per_source_across_all_roles") is not True
        or partition.get("no_replacement") is not True
        or partition.get("all_selected_records_retained") is not True
        or partition.get("selection_contract_sha256")
        != SELECTION_CONTRACT_SHA256
        or exposure
        != {
            "scored_v1_development_all_roles_exposed": True,
            "scored_v1_confirmation_all_roles_exposed": True,
            "prior_v1_exposures_model_scored": True,
            "confirmation_all_roles_disjoint_from_exposure": True,
            "v2_selection_uses_model_outputs": False,
        }
    ):
        raise ManifestError("v2 partition lock drifted")


def _validate_without_integrity(manifest: Mapping[str, Any]) -> None:
    if (
        manifest.get("schema") != SCHEMA
        or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION
        or manifest.get("benchmark_label") != BENCHMARK_LABEL
        or manifest.get("contains_source_text") is not False
        or manifest.get("partition") != PARTITION
    ):
        raise ManifestError("unsupported LongMemEval chat v2 manifest")
    leaked = chat_v1._FORBIDDEN_SOURCE_KEYS.intersection(
        chat_v1._walk_keys(manifest)
    )
    if leaked:
        raise ManifestError(
            "v2 manifest contains source-text fields: "
            + ", ".join(sorted(leaked))
        )
    lock = manifest.get("policy_lock")
    if not isinstance(lock, Mapping):
        raise ManifestError("v2 policy lock is missing")
    _validate_policy_lock(lock)
    if (
        manifest.get("source_inventory") != lock.get("source_inventory")
        or manifest.get("model") != _model_config()
        or manifest.get("training_free_graft") != _graft_config()
        or manifest.get("chat_serialization")
        != chat_v1._chat_serialization()
        or manifest.get("evaluation_contract") != _evaluation_contract()
        or manifest.get("admission_policy") != _admission_policy()
        or manifest.get("tokenizer")
        != {
            "model_id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "fast_offset_mapping_required": True,
        }
    ):
        raise ManifestError("v2 execution policy drifted")

    records = manifest.get("records")
    policy = manifest.get("selection_policy") or {}
    if (
        not isinstance(records, list)
        or not records
        or policy.get("source_only") is not True
        or policy.get("model_outputs_used") is not False
        or policy.get("output_based_replacement") is not False
        or policy.get("fixed_before_model_scoring") is not True
        or policy.get("no_replacement") is not True
        or policy.get("all_selected_records_retained") is not True
        or len(records) != int(policy.get("records", -1))
        or len(records) != int(policy.get("requested_records", -1))
        or int(policy.get("eligible_source_only_candidates", -1))
        < len(records)
    ):
        raise ManifestError("v2 source-only selection policy drifted")
    audit = policy.get("audit")
    selected_audit = (
        []
        if not isinstance(audit, list)
        else [item for item in audit if item.get("status") == "selected"]
    )
    eligible_audit = (
        []
        if not isinstance(audit, list)
        else [item for item in audit if item.get("status") != "ineligible"]
    )
    if (
        not isinstance(audit, list)
        or len(audit) != int(policy.get("candidate_count", -1))
        or any(item.get("model_outputs_used") is not False for item in audit)
        or len(selected_audit) != len(records)
        or len(eligible_audit)
        != int(policy.get("eligible_source_only_candidates", -1))
        or {str(item.get("record_id") or "") for item in selected_audit}
        != {str(record.get("record_id") or "") for record in records}
    ):
        raise ManifestError("v2 source-only eligibility audit drifted")

    exposed_hashes = set(
        manifest["source_inventory"][
            "development_exposed_source_id_sha256"
        ]
    )
    record_ids: set[str] = set()
    target_ids: set[str] = set()
    observed_role_ids: set[str] = set()
    fixed_c_infeasible = 0
    for record in records:
        record_id = str(record.get("record_id") or "")
        target_id = str((record.get("target") or {}).get("source_id") or "")
        if (
            not record_id
            or record_id in record_ids
            or not target_id
            or target_id in target_ids
        ):
            raise ManifestError("v2 record or target IDs are invalid")
        record_ids.add(record_id)
        target_ids.add(target_id)
        role_ids = chat_v1.all_role_source_ids(
            ({"records": [record]},)
        )
        source_selection = record.get("source_selection") or {}
        spec_for_id = {
            "baseline": {
                "record_id": source_selection.get("record_id"),
            }
        }
        if observed_role_ids.intersection(role_ids):
            raise ManifestError("a source is assigned to more than one record")
        observed_role_ids.update(role_ids)
        if exposed_hashes.intersection(_source_hashes(role_ids)):
            raise ManifestError("v2 confirmation overlaps an exposed source")

        context = record.get("context") or {}
        local = context.get("local_window_safety") or {}
        fixed_c = context.get("fixed_c_reference") or {}
        fallback = context.get("full_repack_fallback") or {}
        feasibility = fixed_c.get("feasibility")
        if not isinstance(feasibility, list):
            raise ManifestError("v2 fixed-C feasibility is missing")
        actual_fixed_c = all(bool(item.get("feasible")) for item in feasibility)
        requires_fallback = not actual_fixed_c
        fixed_c_infeasible += int(requires_fallback)
        if (
            int(context.get("tokens_strictly_after_owned", -1))
            < MINIMUM_TOKENS_AFTER_OWNED
            or int(context.get("runtime_total_token_bound", -1))
            > MAXIMUM_CONTEXT_TOKENS
            or local.get("selected_span_inside_local_window") is not False
            or local.get(
                "owned_round_strictly_outside_local_window_before_query"
            )
            is not True
            or int(local.get("true_local_window_tokens", -1)) != WINDOW
            or bool(fixed_c.get("all_affected_boundaries_feasible"))
            != actual_fixed_c
            or fallback.get("required") is not requires_fallback
            or fallback.get("reason")
            != (
                "fixed_c_infeasible"
                if requires_fallback
                else "not_required"
            )
            or fallback.get("only_permitted_reason")
            != "fixed_c_infeasible"
            or fallback.get("record_retained_without_replacement") is not True
            or context.get("raw_omission_retokenizes_exactly") is not True
            or context.get(
                "raw_omission_equals_complete_turn_position_drop"
            )
            is not True
            or source_selection.get("selection_contract_sha256")
            != SELECTION_CONTRACT_SHA256
            or int(
                source_selection.get("base_tail_minimum_tokens", -1)
            )
            not in TAIL_MINIMUM_SCHEDULE
            or record_id != _v2_record_id(spec_for_id)
            or int(
                local.get(
                    "minimum_tokens_strictly_after_owned",
                    -1,
                )
            )
            != MINIMUM_TOKENS_AFTER_OWNED
            or int(
                local.get(
                    "observed_tokens_strictly_after_owned",
                    -1,
                )
            )
            != int(context.get("tokens_strictly_after_owned", -1))
        ):
            raise ManifestError("v2 deletion-safe geometry drifted")

    partition = lock["partition_policy"]
    if (
        int(partition.get("frozen_records", -1)) != len(records)
        or int(partition.get("requested_records", -1)) != len(records)
        or int(partition.get("all_role_source_count", -1))
        != len(observed_role_ids)
        or partition.get("all_role_source_ids_sha256")
        != _payload_sha256(_source_hashes(observed_role_ids))
        or partition.get("audit_sha256") != _payload_sha256(audit)
    ):
        raise ManifestError("v2 frozen cohort lock drifted")


def freeze_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
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
                f"v2 record integrity mismatch: {record.get('record_id')}"
            )
    integrity = manifest.get("integrity")
    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
    ):
        raise ManifestError("v2 manifest integrity mismatch")


def rehydrate_manifest(
    manifest: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
) -> tuple[chat_v1.RehydratedChatRecord, ...]:
    """Rebuild source text and reject source, session, token, or geometry drift."""

    validate_manifest(manifest)
    examples = base.extract_longmemeval_examples(rows)
    descriptors = chat_v1._source_descriptors(examples)
    inventory = manifest["source_inventory"]
    if (
        len(descriptors) != int(inventory["full_row_count"])
        or _payload_sha256(descriptors)
        != inventory["full_descriptors_sha256"]
    ):
        raise ManifestError("rehydrated LongMemEval v2 inventory drifted")
    examples_by_source = {example.source_id: example for example in examples}
    records: list[chat_v1.RehydratedChatRecord] = []
    for stored in manifest["records"]:
        source_selection = stored.get("source_selection") or {}
        spec = {
            "baseline": {
                "record_id": source_selection.get("record_id"),
                "manifest_integrity_sha256": source_selection.get(
                    "manifest_integrity_sha256"
                ),
            },
            "target": copy.deepcopy(dict(stored["target"])),
            "retained_probe": copy.deepcopy(
                dict(stored["retained_probe"])
            ),
            "tail_session_references": copy.deepcopy(
                list(stored["tail_session_references"])
            ),
        }
        prepared = _prepare_from_spec(
            spec,
            examples_by_source,
            tokenizer,
            base_tail_minimum_tokens=int(
                source_selection["base_tail_minimum_tokens"]
            ),
        )
        regenerated = prepared.public_record
        observed = copy.deepcopy(dict(stored))
        observed.pop("record_integrity", None)
        if regenerated != observed:
            raise ManifestError("rehydrated LongMemEval chat v2 record drifted")
        records.append(prepared.runtime)
    return tuple(records)


def _require_exact_keys(
    value: Any,
    expected: set[str],
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ManifestError(f"v2 census has unknown or missing fields at {path}")
    return value


def _validate_census_schema(census: Mapping[str, Any]) -> None:
    _require_exact_keys(
        census,
        {
            "schema",
            "contains_source_text",
            "selection_uses_model_outputs",
            "candidate_count",
            "eligible_source_only_candidates",
            "ineligible_source_only_candidates",
            "selected_records",
            "requested_records",
            "exposure_counts",
            "audit_status_counts",
            "audit_reason_counts",
            "audit",
            "audit_sha256",
            "selected_record_ids",
            "selected_record_ids_sha256",
            "selected_all_role_source_count",
            "selected_all_role_source_ids_sha256",
            "geometry",
            "policy_bindings",
            "manifest_integrity_sha256",
            "policy_lock_sha256",
            "integrity",
        },
        path="$",
    )
    _require_exact_keys(
        census["exposure_counts"],
        {
            "full_oracle_rows",
            "scored_v1_development_all_role_sources",
            "scored_v1_confirmation_all_role_sources",
            "total_exposed_all_role_sources",
            "untouched_rows",
        },
        path="$.exposure_counts",
    )
    _require_exact_keys(
        census["audit_status_counts"],
        {"selected", "eligible_not_selected", "ineligible"},
        path="$.audit_status_counts",
    )
    reason_counts = census["audit_reason_counts"]
    if not isinstance(reason_counts, Mapping):
        raise ManifestError("v2 census reason counts are not an object")
    _require_exact_keys(
        census["geometry"],
        {
            "minimum_tokens_strictly_after_owned",
            "maximum_tokens_strictly_after_owned",
            "minimum_runtime_total_token_bound",
            "maximum_runtime_total_token_bound",
            "fixed_c_infeasible_records",
            "full_repack_required_records",
            "selected_span_inside_local_window_records",
        },
        path="$.geometry",
    )
    _require_exact_keys(
        census["policy_bindings"],
        {
            "selection_seed",
            "candidate_limit",
            "selection_contract_sha256",
            "manifest_audit_sha256",
            "no_replacement",
            "one_record_per_source_across_all_roles",
            "admission_policy",
            "admission_policy_sha256",
        },
        path="$.policy_bindings",
    )
    _require_exact_keys(
        census["policy_bindings"]["admission_policy"],
        {
            "target_metric",
            "minimum_target_present_minus_raw_full_sequence_mean_logprob_nats",
            "maximum_target_present_first_token_rank",
            "maximum_retained_first_token_rank_in_present_and_raw",
            "target_full_sequence_scoring_required",
            "threshold_overrides_allowed",
        },
        path="$.policy_bindings.admission_policy",
    )
    _require_exact_keys(
        census["integrity"],
        {"algorithm", "sha256"},
        path="$.integrity",
    )
    audit = census["audit"]
    if not isinstance(audit, list):
        raise ManifestError("v2 census audit is not a list")
    base_keys = {
        "target_source_id_sha256",
        "status",
        "reason",
        "model_outputs_used",
    }
    selected_keys = base_keys | {
        "record_id",
        "all_role_source_count",
        "all_role_source_ids_sha256",
        "base_tail_minimum_tokens",
        "tokens_strictly_after_owned",
    }
    for index, item in enumerate(audit):
        status = str((item if isinstance(item, Mapping) else {}).get("status"))
        _require_exact_keys(
            item,
            base_keys if status == "ineligible" else selected_keys,
            path=f"$.audit[{index}]",
        )
    leaked = chat_v1._FORBIDDEN_SOURCE_KEYS.intersection(
        chat_v1._walk_keys(census)
    )
    if leaked:
        raise ManifestError(
            "v2 census contains source-text fields: "
            + ", ".join(sorted(leaked))
        )


def validate_census(
    census: Mapping[str, Any],
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed unless the census is fully derived from both frozen inputs."""

    if policy_lock is None:
        embedded = manifest.get("policy_lock")
        if not isinstance(embedded, Mapping):
            raise ManifestError("v2 census standalone policy lock is missing")
        policy_lock = embedded
    _validate_census_schema(census)
    payload = copy.deepcopy(dict(census))
    integrity = payload.pop("integrity")
    if (
        census.get("schema") != CENSUS_SCHEMA
        or census.get("contains_source_text") is not False
        or census.get("selection_uses_model_outputs") is not False
        or integrity.get("algorithm") != "sha256"
        or integrity.get("sha256") != _payload_sha256(payload)
    ):
        raise ManifestError("v2 eligibility census integrity drifted")
    validate_manifest(manifest)
    _validate_policy_lock(policy_lock)
    if (
        dict(policy_lock) != manifest.get("policy_lock")
        or census.get("policy_lock_sha256")
        != policy_lock.get("lock_sha256")
    ):
        raise ManifestError("v2 census standalone policy binding drifted")
    expected = _census_body_from_manifest(manifest)
    observed = copy.deepcopy(dict(census))
    observed.pop("integrity")
    if observed != expected:
        raise ManifestError("v2 eligibility census derivation drifted")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
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
    parser.add_argument("--data-path")
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--policy-lock-out")
    parser.add_argument("--census-out")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from transformers import AutoTokenizer

    destinations = [
        Path(value)
        for value in (
            args.manifest_out,
            args.policy_lock_out,
            args.census_out,
        )
        if value
    ]
    if not args.overwrite:
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(", ".join(existing))
    rows = base.load_pinned_longmemeval_rows(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        use_fast=True,
    )
    manifest, census = build_confirmation_manifest(
        rows,
        tokenizer,
        require_pinned_artifacts=True,
    )
    write_json(args.manifest_out, manifest)
    if args.policy_lock_out:
        write_json(args.policy_lock_out, manifest["policy_lock"])
    if args.census_out:
        write_json(args.census_out, census)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

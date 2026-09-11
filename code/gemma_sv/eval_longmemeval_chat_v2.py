"""Evaluate the deletion-safe LongMemEval V1 registered-chat v2 cohort.

The exact committed v2 confirmation manifest, policy lock, and source-only
eligibility census directly authorize one fixed 4B-IT training-free-graft arm.
Starting model scoring additionally requires an explicit acknowledgement.
There is no development-arm or prompt-selection step because every source
exposed to the v1 selection process is excluded from this v2 cohort.

Reports contain hashes and scalar measurements only.  Resume validates an
existing report as an audit artifact, discards every prior row, and re-executes
all 16 authorized records; completed model rows are never trusted or reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Mapping, Sequence

from gemma_sv import eval_longmemeval_chat as v1_eval
from gemma_sv import longmemeval_chat_benchmark_v2 as benchmark


SCHEMA = "gemma-sv-longmemeval-chat-admission-evaluation-v2"
SCHEMA_VERSION = 2
ARM = v1_eval.GRAFTED_ARM
EXPECTED_RECORDS = 16
EVALUATION_SEED = 0
DTYPE = "float32"
WINDOW = 1_024
NU = 0.7
CHUNK = 128
SOLVER_SEED = 0
PRESERVE_PREFIX_MASS = True
PER_BOUNDARY_BOX = True

PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
DEFAULT_MANIFEST = BENCHMARKS / "longmemeval_chat_confirmation_v2.json"
DEFAULT_POLICY_LOCK = BENCHMARKS / "longmemeval_chat_policy_lock_v2.json"
DEFAULT_CENSUS = BENCHMARKS / "longmemeval_chat_census_v2.json"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_confirmation_grafted_v2.json"
)

PINNED_MANIFEST_FILE_SHA256 = (
    "aa980722a9630448063eb0b825156d406f88e6226fbc3800d72de8f290f69d6d"
)
PINNED_POLICY_LOCK_FILE_SHA256 = (
    "9a195831f99fe73d0b8e0cad6987579141b5fe8fa231cbebdf6ccb88222a1fd2"
)
PINNED_CENSUS_FILE_SHA256 = (
    "83cddd3ad1ca4a269381bdae99dceb9c9ed0176eb7eb753e9bd077debc106340"
)
PINNED_CENSUS_INTEGRITY_SHA256 = (
    "17e915d941d16b0b73e2e72dc35f973ea8d110e27dedeca92238f37f759d8b0e"
)

_THRESHOLD_KEYS = frozenset(
    {
        "target_metric",
        "minimum_target_present_minus_raw_full_sequence_mean_logprob_nats",
        "maximum_target_present_first_token_rank",
        "maximum_retained_first_token_rank_in_present_and_raw",
        "target_full_sequence_scoring_required",
        "threshold_overrides_allowed",
    }
)
_IMPLEMENTATION_PATHS = {
    "eval_longmemeval_chat_v2.py": Path(__file__).resolve(),
    "longmemeval_chat_benchmark_v2.py": PACKAGE
    / "longmemeval_chat_benchmark_v2.py",
    "eval_longmemeval_chat.py": PACKAGE / "eval_longmemeval_chat.py",
    "longmemeval_chat_benchmark.py": PACKAGE
    / "longmemeval_chat_benchmark.py",
    "longmemeval_deletion_benchmark.py": PACKAGE
    / "longmemeval_deletion_benchmark.py",
    "eval_longmemeval_deletion.py": PACKAGE
    / "eval_longmemeval_deletion.py",
    "persistent_deletion.py": PACKAGE / "persistent_deletion.py",
    "graft.py": PACKAGE / "graft.py",
    "recovery_state.py": PACKAGE / "recovery_state.py",
    "sv_global_attention.py": PACKAGE / "sv_global_attention.py",
    "layer_select.py": PACKAGE / "layer_select.py",
    "demo_server/gemma_engine.py": PACKAGE / "demo_server" / "gemma_engine.py",
    "svattn/causal_sv_attention.py": WORKSPACE
    / "svattn"
    / "causal_sv_attention.py",
    "svattn/mlx_svdd.py": WORKSPACE / "svattn" / "mlx_svdd.py",
    "cp_svm/oneclass_fast.py": WORKSPACE / "cp_svm" / "oneclass_fast.py",
    "cp_svm/kernels.py": WORKSPACE / "cp_svm" / "kernels.py",
}
_PROTECTED_COMMITTED_INPUTS = frozenset(
    {
        DEFAULT_MANIFEST,
        DEFAULT_POLICY_LOCK,
        DEFAULT_CENSUS,
        *_IMPLEMENTATION_PATHS.values(),
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: str | Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _require_file_hash(path: str | Path, expected: str, *, name: str) -> None:
    if _sha256_file(path) != expected:
        raise ValueError(f"{name} is not the exact committed v2 artifact")


def _resolved_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _validate_output_path(
    output: str | Path,
    *,
    manifest_path: str | Path,
    policy_lock_path: str | Path,
    census_path: str | Path,
) -> Path:
    output_path = Path(output).expanduser()
    resolved_output = _resolved_path(output_path)
    protected_paths = {
        Path(path).expanduser()
        for path in (
            manifest_path,
            policy_lock_path,
            census_path,
            *_PROTECTED_COMMITTED_INPUTS,
        )
    }
    for protected_path in protected_paths:
        same_resolved_path = resolved_output == _resolved_path(protected_path)
        same_existing_file = False
        if output_path.exists() and protected_path.exists():
            try:
                same_existing_file = os.path.samefile(
                    output_path,
                    protected_path,
                )
            except OSError:
                same_existing_file = False
        if same_resolved_path or same_existing_file:
            raise ValueError(
                "output path aliases a protected committed input or evaluator "
                "source"
            )
    return resolved_output


def _assert_source_free(payload: Mapping[str, Any]) -> None:
    v1_eval._assert_source_free_payload(payload)


def _assert_finite_json(value: Any, *, path: str = "report") -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_json(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_finite_json(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} is not JSON-compatible")


def _local_window_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return ((record.get("context") or {}).get("local_window_safety") or {})


def validate_local_window_contract(
    manifest_record: Mapping[str, Any],
    runtime_record: Any | None = None,
) -> dict[str, Any]:
    context = manifest_record.get("context") or {}
    local = _local_window_payload(manifest_record)
    observed_after = int(context.get("tokens_strictly_after_owned", -1))
    if (
        int(local.get("true_local_window_tokens", -1)) != WINDOW
        or int(local.get("minimum_tokens_strictly_after_owned", -1))
        != benchmark.MINIMUM_TOKENS_AFTER_OWNED
        or int(local.get("observed_tokens_strictly_after_owned", -1))
        != observed_after
        or observed_after < benchmark.MINIMUM_TOKENS_AFTER_OWNED
        or observed_after < WINDOW
        or local.get("owned_round_strictly_outside_local_window_before_query")
        is not True
        or local.get("selected_span_inside_local_window") is not False
    ):
        raise ValueError("v2 true local-window safety contract differs")
    if runtime_record is not None:
        runtime_after = (
            len(runtime_record.context.original_token_ids)
            - max(runtime_record.context.forget_positions)
            - 1
        )
        if runtime_after != observed_after:
            raise ValueError("rehydrated v2 post-owned distance differs")
    return {
        "true_local_window_tokens": WINDOW,
        "tokens_strictly_after_owned": observed_after,
        "minimum_tokens_strictly_after_owned": (
            benchmark.MINIMUM_TOKENS_AFTER_OWNED
        ),
        "owned_round_strictly_outside_local_window_before_query": True,
        "selected_span_inside_local_window": False,
    }


def _validate_threshold_mapping(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _THRESHOLD_KEYS:
        raise ValueError(f"{name} must contain the exact admission thresholds")
    minimum_lift = value[
        "minimum_target_present_minus_raw_full_sequence_mean_logprob_nats"
    ]
    target_rank = value["maximum_target_present_first_token_rank"]
    retained_rank = value[
        "maximum_retained_first_token_rank_in_present_and_raw"
    ]
    if (
        isinstance(minimum_lift, bool)
        or not isinstance(minimum_lift, (int, float))
        or not math.isfinite(float(minimum_lift))
        or float(minimum_lift) < 0.0
    ):
        raise ValueError(f"{name} minimum target lift is invalid")
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
        for rank in (target_rank, retained_rank)
    ):
        raise ValueError(f"{name} rank thresholds are invalid")
    if (
        value.get("target_metric")
        != (
            "present_minus_fresh_raw_omission_full_sequence_"
            "mean_logprob_nats"
        )
        or value.get("target_full_sequence_scoring_required") is not True
        or value.get("threshold_overrides_allowed") is not False
    ):
        raise ValueError(f"{name} admission metric contract is invalid")
    return {
        "target_metric": str(value["target_metric"]),
        "minimum_target_present_minus_raw_full_sequence_mean_logprob_nats": (
            float(minimum_lift)
        ),
        "maximum_target_present_first_token_rank": int(target_rank),
        "maximum_retained_first_token_rank_in_present_and_raw": int(
            retained_rank
        ),
        "target_full_sequence_scoring_required": True,
        "threshold_overrides_allowed": False,
    }


def locked_admission_thresholds(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the hash-bound admission gates; this evaluator defines no values."""

    policy_thresholds = _validate_threshold_mapping(
        policy_lock.get("admission_policy"),
        name="v2 policy lock",
    )
    manifest_thresholds = _validate_threshold_mapping(
        manifest.get("admission_policy"),
        name="v2 manifest",
    )
    policy_contract_thresholds = _validate_threshold_mapping(
        (policy_lock.get("evaluation_contract") or {}).get(
            "admission_policy"
        ),
        name="v2 policy evaluation contract",
    )
    manifest_contract_thresholds = _validate_threshold_mapping(
        (manifest.get("evaluation_contract") or {}).get("admission_policy"),
        name="v2 manifest evaluation contract",
    )
    if not (
        manifest_thresholds
        == policy_thresholds
        == policy_contract_thresholds
        == manifest_contract_thresholds
    ):
        raise ValueError("v2 manifest and policy admission thresholds differ")
    return policy_thresholds


def validate_locked_inputs(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    census: Mapping[str, Any],
) -> None:
    benchmark.validate_manifest(manifest)
    benchmark._validate_policy_lock(policy_lock)
    benchmark.validate_census(census, manifest)
    if (
        manifest.get("integrity", {}).get("sha256")
        != benchmark.PINNED_V2_CONFIRMATION_INTEGRITY_SHA256
        or policy_lock.get("lock_sha256")
        != benchmark.PINNED_V2_POLICY_LOCK_SHA256
        or census.get("integrity", {}).get("sha256")
        != PINNED_CENSUS_INTEGRITY_SHA256
        or manifest.get("policy_lock") != policy_lock
    ):
        raise ValueError("v2 committed-target artifact binding differs")
    records = manifest.get("records") or ()
    selected_audit = [
        row
        for row in census.get("audit") or ()
        if row.get("status") == "selected"
    ]
    if (
        len(records) != EXPECTED_RECORDS
        or int(census.get("selected_records", -1)) != EXPECTED_RECORDS
        or int(census.get("requested_records", -1)) != EXPECTED_RECORDS
        or int(census.get("eligible_source_only_candidates", -1)) != 32
        or census.get("selection_uses_model_outputs") is not False
        or census.get("audit") != manifest.get("selection_policy", {}).get(
            "audit"
        )
        or [row.get("record_id") for row in selected_audit]
        != [record.get("record_id") for record in records]
    ):
        raise ValueError("v2 census or all16 frozen cohort differs")
    if (
        policy_lock.get("status") != "locked-before-v2-model-scoring"
        or policy_lock.get("selection_uses_model_outputs") is not False
        or policy_lock.get("output_based_replacement") is not False
        or policy_lock.get("model", {}).get("arm") != ARM
        or policy_lock.get("model", {}).get("adapter") is not None
    ):
        raise ValueError("v2 direct confirmation policy differs")
    graft = policy_lock.get("training_free_graft") or {}
    expected_graft = {
        "window": WINDOW,
        "nu": NU,
        "chunk": CHUNK,
        "readout": "softmax",
        "preserve_prefix_mass": PRESERVE_PREFIX_MASS,
        "per_boundary_box": PER_BOUNDARY_BOX,
        "solver_seed": SOLVER_SEED,
        "training_steps": 0,
    }
    if any(graft.get(key) != value for key, value in expected_graft.items()):
        raise ValueError("v2 training-free graft policy differs")
    admission_policy = locked_admission_thresholds(manifest, policy_lock)
    census_bindings = census.get("policy_bindings") or {}
    if (
        census_bindings.get("admission_policy") != admission_policy
        or census_bindings.get("admission_policy_sha256")
        != _payload_sha256(admission_policy)
    ):
        raise ValueError("v2 census admission threshold binding differs")
    for record in records:
        validate_local_window_contract(record)
    _assert_source_free(manifest)
    _assert_source_free(policy_lock)
    _assert_source_free(census)


def load_locked_inputs(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    policy_lock_path: str | Path = DEFAULT_POLICY_LOCK,
    census_path: str | Path = DEFAULT_CENSUS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_file_hash(
        manifest_path,
        PINNED_MANIFEST_FILE_SHA256,
        name="v2 confirmation manifest",
    )
    _require_file_hash(
        policy_lock_path,
        PINNED_POLICY_LOCK_FILE_SHA256,
        name="v2 policy lock",
    )
    _require_file_hash(
        census_path,
        PINNED_CENSUS_FILE_SHA256,
        name="v2 eligibility census",
    )
    manifest = _load_mapping(manifest_path, name="v2 confirmation manifest")
    policy_lock = _load_mapping(policy_lock_path, name="v2 policy lock")
    census = _load_mapping(census_path, name="v2 census")
    validate_locked_inputs(manifest, policy_lock, census)
    return manifest, policy_lock, census


def runtime_contract(*, device: str) -> dict[str, Any]:
    contract = v1_eval.runtime_contract(ARM, device=device)
    if contract != {
        "arm": ARM,
        "model_id": benchmark.MODEL_ID,
        "model_revision": benchmark.MODEL_REVISION,
        "adapter": None,
        "device": str(device).strip(),
        "dtype": DTYPE,
        "window": WINDOW,
        "graft_enabled": True,
        "nu": NU,
        "chunk": CHUNK,
        "readout": "softmax",
        "preserve_prefix_mass": PRESERVE_PREFIX_MASS,
        "per_boundary_box": PER_BOUNDARY_BOX,
        "solver_seed": SOLVER_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "training_steps": 0,
    }:
        raise RuntimeError("shared v1 runtime contract drifted from v2")
    return contract


def _runtime_config_kwargs(*, device: str) -> dict[str, Any]:
    return v1_eval._runtime_config_kwargs(ARM, device=device)


def _make_runtime(*, device: str):
    return v1_eval._make_runtime(ARM, device=device)


def verify_loaded_runtime(runtime: Any, *, device: str) -> None:
    v1_eval.verify_loaded_runtime(runtime, arm=ARM, device=device)
    if (
        int(getattr(runtime.config, "window", -1)) != WINDOW
        or getattr(runtime.config, "lora_path", object()) is not None
        or str(getattr(runtime.config, "dtype", "")) != DTYPE
    ):
        raise RuntimeError("loaded v2 local-window runtime differs")


def authorize_confirmation(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    explicit_acknowledgement: bool,
    requested_arm: str | None = None,
    device: str,
) -> tuple[str, str]:
    validate_locked_inputs(manifest, policy_lock, census)
    if requested_arm is not None:
        raise ValueError("v2 confirmation forbids all arm overrides")
    if explicit_acknowledgement is not True:
        raise PermissionError(
            "v2 confirmation scoring requires acknowledgement exactly True"
        )
    runtime_contract(device=device)
    return ARM, str(device).strip()


def _policy_admission_decomposition(
    present: Mapping[str, Mapping[str, Any]],
    fresh_raw_omission: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    locked = _validate_threshold_mapping(
        thresholds,
        name="locked admission thresholds",
    )
    if set(present) != {"target_current", "retained"} or set(
        fresh_raw_omission
    ) != {"target_current", "retained"}:
        raise ValueError("admission requires exactly target_current and retained")
    target_present = present["target_current"]
    target_raw = fresh_raw_omission["target_current"]
    retained_present = present["retained"]
    retained_raw = fresh_raw_omission["retained"]
    lift = float(target_present["mean_log_probability"]) - float(
        target_raw["mean_log_probability"]
    )
    target_present_rank = int(target_present["first_target_token_rank"])
    target_raw_rank = int(target_raw["first_target_token_rank"])
    retained_present_rank = int(retained_present["first_target_token_rank"])
    retained_raw_rank = int(retained_raw["first_target_token_rank"])
    lift_passed = (
        lift
        >= locked[
            "minimum_target_present_minus_raw_full_sequence_mean_logprob_nats"
        ]
    )
    target_rank_passed = (
        target_present_rank
        <= locked["maximum_target_present_first_token_rank"]
    )
    retained_available = (
        retained_present_rank
        <= locked[
            "maximum_retained_first_token_rank_in_present_and_raw"
        ]
        and retained_raw_rank
        <= locked[
            "maximum_retained_first_token_rank_in_present_and_raw"
        ]
    )
    target_admitted = lift_passed and target_rank_passed
    target_reasons = []
    if not lift_passed:
        target_reasons.append("target_lift_below_locked_minimum")
    if not target_rank_passed:
        target_reasons.append("target_present_rank_above_locked_maximum")
    return {
        "scheme": "longmemeval_chat_v2_policy_locked_admission",
        "target_recall": {
            "probe_id": "target_current",
            "status": "admitted" if target_admitted else "rejected",
            "admitted": target_admitted,
            "reasons": target_reasons,
            "present_minus_fresh_raw_omission_nats": lift,
            "present_first_token_rank": target_present_rank,
            "fresh_raw_omission_first_token_rank": target_raw_rank,
            "lift_passed": lift_passed,
            "present_rank_passed": target_rank_passed,
        },
        "retained_availability": {
            "probe_id": "retained",
            "status": "available" if retained_available else "unavailable",
            "available": retained_available,
            "present_first_token_rank": retained_present_rank,
            "fresh_raw_omission_first_token_rank": retained_raw_rank,
            "both_states_must_pass": True,
        },
        "joint_target_and_retained": {
            "status": (
                "admitted"
                if target_admitted and retained_available
                else "rejected"
            ),
            "admitted": target_admitted and retained_available,
        },
        "thresholds": locked,
    }


def evaluate_admission_record(
    runtime: Any,
    record: Any,
    manifest_record: Mapping[str, Any],
    *,
    warmup: int,
    repeats: int,
    admission_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    if record.record_id != manifest_record.get("record_id"):
        raise ValueError("rehydrated record differs from v2 manifest order")
    local = validate_local_window_contract(manifest_record, record)
    row = v1_eval.evaluate_admission_record(
        runtime,
        record,
        warmup=warmup,
        repeats=repeats,
    )
    row["admission"] = _policy_admission_decomposition(
        row["references"]["present"]["scores"],
        row["references"]["fresh_raw_omission"]["scores"],
        admission_thresholds,
    )
    row["local_window_safety"] = local
    _assert_source_free(row)
    return row


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    frozen_records: int = EXPECTED_RECORDS,
    admission_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    summary = v1_eval.summarize_records(
        records,
        frozen_records=frozen_records,
        selected_records=frozen_records,
    )
    summary["admission_thresholds"] = _validate_threshold_mapping(
        admission_thresholds,
        name="locked admission thresholds",
    )
    denominators = summary["denominators"]
    completed = [
        record for record in records if record.get("status") == "completed"
    ]
    failed = [record for record in records if record.get("status") == "failed"]
    local_safe = sum(
        bool(
            (record.get("local_window_safety") or {}).get(
                "owned_round_strictly_outside_local_window_before_query"
            )
        )
        and int(
            (record.get("local_window_safety") or {}).get(
                "true_local_window_tokens", -1
            )
        )
        == WINDOW
        for record in completed
    )
    immutable = int(denominators["source_state_immutable"])
    denominators.update(
        {
            "authorized_records": int(frozen_records),
            "accounted_attempted_records": len(completed) + len(failed),
            "unaccounted_attempted_records": (
                len(records) - len(completed) - len(failed)
            ),
            "all_authorized_records_attempted": (
                len(records) == int(frozen_records)
            ),
            "source_state_immutability_unverified_records": (
                int(frozen_records) - len(completed)
            ),
            "source_state_immutability_failures_or_unverified": (
                int(frozen_records) - immutable
            ),
            "local_window_safe_records": local_safe,
            "local_window_failures": len(completed) - local_safe,
            "local_window_unverified_records": (
                int(frozen_records) - len(completed)
            ),
            "true_local_window_tokens": WINDOW,
        }
    )
    summary["local_window_failure_record_ids"] = [
        str(record.get("record_id") or "")
        for record in completed
        if not bool(
            (record.get("local_window_safety") or {}).get(
                "owned_round_strictly_outside_local_window_before_query"
            )
        )
    ]
    return summary


def _implementation_fingerprints() -> dict[str, str]:
    fingerprints = {
        label: _sha256_file(path)
        for label, path in sorted(_IMPLEMENTATION_PATHS.items())
    }
    fingerprints["contract_sha256"] = _payload_sha256(fingerprints)
    return fingerprints


def _environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "torch", "transformers", "mlx"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {"platform": platform.platform(), "versions": versions}


def _base_report(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    manifest_path: Path,
    policy_lock_path: Path,
    census_path: Path,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    record_ids = [str(record["record_id"]) for record in manifest["records"]]
    admission_thresholds = locked_admission_thresholds(manifest, policy_lock)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "evaluation": "LongMemEval V1 deletion-safe constrained-chat v2 admission",
        "official_longmemeval_leaderboard_score": False,
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "model_scoring_used_for_selection_or_replacement": False,
        "authorization": {
            "kind": "direct_committed_v2_policy_lock",
            "explicit_confirmation_acknowledgement_required": True,
            "arm_override_permitted": False,
            "all_v1_exposed_sources_excluded": True,
            "selection_uses_model_outputs": False,
            "records": EXPECTED_RECORDS,
            "no_replacement": True,
        },
        "manifest": {
            "schema": manifest["schema"],
            "partition": manifest["partition"],
            "integrity_sha256": manifest["integrity"]["sha256"],
            "file_sha256": _sha256_file(manifest_path),
            "frozen_records": len(record_ids),
            "selected_record_ids": record_ids,
            "selected_record_ids_sha256": _payload_sha256(record_ids),
        },
        "policy_lock": {
            "schema": policy_lock["schema"],
            "status": policy_lock["status"],
            "lock_sha256": policy_lock["lock_sha256"],
            "file_sha256": _sha256_file(policy_lock_path),
        },
        "eligibility_census": {
            "schema": census["schema"],
            "integrity_sha256": census["integrity"]["sha256"],
            "file_sha256": _sha256_file(census_path),
            "candidate_count": census["candidate_count"],
            "eligible_source_only_candidates": census[
                "eligible_source_only_candidates"
            ],
            "selected_records": census["selected_records"],
            "selection_uses_model_outputs": False,
        },
        "artifacts": {
            "dataset_id": policy_lock["source_inventory"]["dataset_id"],
            "dataset_revision": policy_lock["source_inventory"][
                "dataset_revision"
            ],
            "source_artifact_sha256": policy_lock["source_inventory"][
                "source_artifact_sha256"
            ],
            "full_descriptors_sha256": policy_lock["source_inventory"][
                "full_descriptors_sha256"
            ],
            "development_exposed_all_role_source_count": policy_lock[
                "source_inventory"
            ]["development_exposed_all_role_source_count"],
        },
        "config": {
            **runtime_contract(device=device),
            "maximum_context_tokens": benchmark.MAXIMUM_CONTEXT_TOKENS,
            "minimum_tokens_strictly_after_owned": (
                benchmark.MINIMUM_TOKENS_AFTER_OWNED
            ),
            "true_local_window_tokens": WINDOW,
            "warmup": int(warmup),
            "repeats": int(repeats),
            "admission_only": True,
            "admission_thresholds": admission_thresholds,
        },
        "admission_protocol": {
            "target_probe": "target_current",
            "retained_probe": "retained",
            "full_target_sequences_scored": True,
            "first_target_token_ranks_scored": True,
            "present_reference": "registered chat with owned round present",
            "counterfactual_reference": (
                "fresh prefill of registered chat with owned round omitted"
            ),
            "admission_thresholds": admission_thresholds,
            "no_replacement": True,
            "all16_required": True,
        },
        "resume_protocol": {
            "existing_report_policy": "validate_then_discard_all_rows",
            "completed_rows_reused": False,
            "all_authorized_records_reexecuted": True,
        },
        "implementation": _implementation_fingerprints(),
        "environment": _environment(),
        "records": [],
        "summary": summarize_records(
            [],
            frozen_records=EXPECTED_RECORDS,
            admission_thresholds=admission_thresholds,
        ),
    }


def _resume_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "schema",
            "schema_version",
            "evaluation",
            "official_longmemeval_leaderboard_score",
            "contains_source_text",
            "contains_full_vocabulary_vectors",
            "model_scoring_used_for_selection_or_replacement",
            "authorization",
            "manifest",
            "policy_lock",
            "eligibility_census",
            "artifacts",
            "config",
            "admission_protocol",
            "resume_protocol",
            "implementation",
            "environment",
        )
    }


def _validate_resume(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> int:
    _assert_source_free(report)
    _assert_finite_json(report)
    if _resume_signature(report) != _resume_signature(expected):
        raise ValueError("v2 resume report contract differs")
    records = report.get("records")
    if not isinstance(records, list):
        raise ValueError("v2 resume report records are invalid")
    expected_ids = list(expected["manifest"]["selected_record_ids"])
    ids = [
        str(record.get("record_id") or "")
        for record in records
        if isinstance(record, Mapping)
    ]
    if (
        len(ids) != len(records)
        or len(records) > EXPECTED_RECORDS
        or ids != expected_ids[: len(ids)]
    ):
        raise ValueError("v2 resume report record order differs")
    return len(records)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    _assert_source_free(payload)
    _assert_finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _failed_record(record: Any, exc: Exception) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "partition": record.partition,
        "status": "failed",
        "contains_source_text": False,
        "error_type": type(exc).__name__,
        "error_message_redacted": True,
        "error_message_sha256": benchmark.base.text_sha256(str(exc)),
    }


def run_records(
    manifest: Mapping[str, Any],
    policy_lock: Mapping[str, Any],
    census: Mapping[str, Any],
    records: Sequence[Any],
    runtime: Any,
    *,
    manifest_path: Path,
    policy_lock_path: Path,
    census_path: Path,
    output: Path,
    device: str,
    warmup: int,
    repeats: int,
    confirmation_scoring_acknowledged: bool,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate all16 atomically; resume never reuses completed rows."""

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats positive")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    output = _validate_output_path(
        output,
        manifest_path=manifest_path,
        policy_lock_path=policy_lock_path,
        census_path=census_path,
    )
    _require_file_hash(
        manifest_path,
        PINNED_MANIFEST_FILE_SHA256,
        name="v2 confirmation manifest",
    )
    _require_file_hash(
        policy_lock_path,
        PINNED_POLICY_LOCK_FILE_SHA256,
        name="v2 policy lock",
    )
    _require_file_hash(
        census_path,
        PINNED_CENSUS_FILE_SHA256,
        name="v2 eligibility census",
    )
    arm, authorized_device = authorize_confirmation(
        manifest,
        policy_lock,
        census,
        explicit_acknowledgement=confirmation_scoring_acknowledged,
        requested_arm=None,
        device=device,
    )
    if arm != ARM or authorized_device != str(device).strip():
        raise ValueError("v2 runtime authorization differs")
    admission_thresholds = locked_admission_thresholds(manifest, policy_lock)
    verify_loaded_runtime(runtime, device=device)
    expected_ids = [str(record["record_id"]) for record in manifest["records"]]
    observed_ids = [str(record.record_id) for record in records]
    if (
        len(records) != EXPECTED_RECORDS
        or observed_ids != expected_ids
        or len(set(observed_ids)) != EXPECTED_RECORDS
    ):
        raise ValueError("v2 evaluation requires all 16 records in order")
    expected = _base_report(
        manifest,
        policy_lock,
        census,
        manifest_path=manifest_path,
        policy_lock_path=policy_lock_path,
        census_path=census_path,
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    if output.exists() and not (resume or overwrite):
        raise FileExistsError("output exists; pass resume or overwrite")
    discarded_resume_rows = 0
    if resume:
        if not output.is_file():
            raise ValueError("cannot resume a missing v2 report")
        existing = _load_mapping(output, name="v2 resume report")
        discarded_resume_rows = _validate_resume(existing, expected)
    report = expected
    report["resume_requested"] = bool(resume)
    report["resume_policy"] = "validate_then_discard_all_rows"
    report["discarded_resume_rows"] = discarded_resume_rows
    report["reused_completed_resume_records"] = 0
    _atomic_write(output, report)

    started = time.perf_counter()
    result_rows: list[Mapping[str, Any]] = []
    for runtime_record, manifest_record in zip(records, manifest["records"]):
        try:
            row = evaluate_admission_record(
                runtime,
                runtime_record,
                manifest_record,
                warmup=warmup,
                repeats=repeats,
                admission_thresholds=admission_thresholds,
            )
        except Exception as exc:
            row = _failed_record(runtime_record, exc)
        result_rows.append(row)
        report["records"] = result_rows
        report["summary"] = summarize_records(
            result_rows,
            admission_thresholds=admission_thresholds,
        )
        report["status"] = "running"
        report["elapsed_seconds_this_process"] = time.perf_counter() - started
        _atomic_write(output, report)
    report["summary"] = summarize_records(
        result_rows,
        admission_thresholds=admission_thresholds,
    )
    denominators = report["summary"]["denominators"]
    report["status"] = (
        "completed"
        if denominators["completed_records"] == EXPECTED_RECORDS
        and denominators["record_failures"] == 0
        else "completed_with_record_failures"
    )
    report["elapsed_seconds_this_process"] = time.perf_counter() - started
    _atomic_write(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--policy-lock", default=str(DEFAULT_POLICY_LOCK))
    parser.add_argument("--census", default=str(DEFAULT_CENSUS))
    parser.add_argument("--data-path")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--allow-confirmation-scoring",
        action="store_true",
        help="required by the direct committed v2 confirmation policy",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help=(
            "validate prior output, discard every row, and re-execute all 16; "
            "completed rows are never reused"
        ),
    )
    output_mode.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats positive")
    manifest_path = Path(args.manifest)
    policy_lock_path = Path(args.policy_lock)
    census_path = Path(args.census)
    try:
        manifest, policy_lock, census = load_locked_inputs(
            manifest_path,
            policy_lock_path,
            census_path,
        )
        authorize_confirmation(
            manifest,
            policy_lock,
            census,
            explicit_acknowledgement=args.allow_confirmation_scoring,
            requested_arm=None,
            device=args.device,
        )
        output = _validate_output_path(
            args.out,
            manifest_path=manifest_path,
            policy_lock_path=policy_lock_path,
            census_path=census_path,
        )
    except (OSError, PermissionError, ValueError, benchmark.ManifestError) as exc:
        parser.error(str(exc))
    if output.exists() and not (args.resume or args.overwrite):
        parser.error(f"{output} exists; pass --resume or --overwrite")

    rows = benchmark.base.load_pinned_longmemeval_rows(args.data_path)
    runtime = _make_runtime(device=args.device)
    runtime.ensure_loaded()
    try:
        verify_loaded_runtime(runtime, device=args.device)
        rehydrated = benchmark.rehydrate_manifest(
            manifest,
            rows,
            runtime.tokenizer,
        )
        report = run_records(
            manifest,
            policy_lock,
            census,
            rehydrated,
            runtime,
            manifest_path=manifest_path,
            policy_lock_path=policy_lock_path,
            census_path=census_path,
            output=output,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            confirmation_scoring_acknowledged=(
                args.allow_confirmation_scoring
            ),
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, RuntimeError, benchmark.ManifestError) as exc:
        parser.error(str(exc))
    print(f"wrote evaluation to {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

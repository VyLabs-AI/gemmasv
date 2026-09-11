"""Publish deterministic, source-free metrics for the 96-history decode audit."""

from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence
import unicodedata
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit


PACKAGE = Path(__file__).resolve().parent
WORKSPACE = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
AUDIT_ROOT = (
    WORKSPACE
    / "outputs"
    / "gemma_sv_rag"
    / "longmemeval_chat_response_generation_audit_v2"
)

DEFAULT_FINAL = AUDIT_ROOT / "final.json"
DEFAULT_RECOVERY = AUDIT_ROOT / "integrity_recovery_v1.json"
DEFAULT_COHORT = BENCHMARKS / "longmemeval_chat_cohort_v3.json"
DEFAULT_ANALYSIS_LOCK = (
    BENCHMARKS / "longmemeval_chat_cluster_analysis_lock_v3.json"
)
DEFAULT_OUTPUT = BENCHMARKS / "longmemeval_chat_decoded_summary_v2.json"

SUMMARY_SCHEMA = "gemma-sv-longmemeval-chat-decoded-summary-v2"
FINAL_SCHEMA = "gemma-sv-longmemeval-chat-response-final-v2"
RECOVERY_SCHEMA = "gemma-sv-longmemeval-response-integrity-recovery-v1"
COHORT_SCHEMA = "gemma-sv-longmemeval-chat-clustered-cohort-v3"
ANALYSIS_LOCK_SCHEMA = "gemma-sv-longmemeval-chat-cluster-analysis-lock-v3"

EXPECTED_CLUSTERS = 32
EXPECTED_HISTORIES = 96
HISTORIES_PER_CLUSTER = 3
EXPECTED_CALLS = 1536
EXPECTED_AUTHORIZATION_LOCK_SHA256 = (
    "d5281f91c13207f710df92e28345841be1b7d19e7a8aff32a8417274c563868b"
)
EXPECTED_FINAL_FILE_SHA256 = (
    "6032579fa2ac2e0d89485b4b1fadd23e1399b563e8b2f6ade780b0196a7cd75f"
)
EXPECTED_RECOVERY_FILE_SHA256 = (
    "32ab8519c28d19c9631fb89749c553a8f9c2987ee36c49da82684283366e58b0"
)
EXPECTED_COHORT_FILE_SHA256 = (
    "579096fed8b2a415e9c046b18c7e29c897cd4f317e7b18bf7af60ef1ae853294"
)
EXPECTED_ANALYSIS_LOCK_FILE_SHA256 = (
    "9e105cb11b382d9514cee7c1472ca4f20ee709f39f830b39d366ea25d97a7029"
)
EXPECTED_FINAL_INTEGRITY_SHA256 = (
    "331975ade07af2724d59a2e3900cfba5a7cb6f7bf5e8f0b3cac3d8e47f8cd904"
)
EXPECTED_RECOVERY_INTEGRITY_SHA256 = (
    "865d3853ff99dea3984e5c6a2c0eb87c97375fd023cde07cefedb9a846190447"
)
EXPECTED_COHORT_INTEGRITY_SHA256 = (
    "1f0cf1d560a90137a5c4b8ed57f4d48b53d2d9947ba0b5468f097f7305a3d051"
)
EXPECTED_ANALYSIS_LOCK_SHA256 = (
    "01e106afb668e4e29971c3f7574f0b93ecc9b5bef37842a84bd991447ed0552d"
)

CONDITION_IDS = (
    "present",
    "fresh_raw_omission",
    "exact_decrement_or_refit_policy",
    "prompt_suppression",
)
PROBE_IDS = ("target_current", "retained")
REGISTERED_MATCH_IDS = (
    "trimmed_exact",
    "casefold_whitespace",
    "casefold_substring",
)
NORMALIZATION_IDS = (
    "trimmed_exact",
    "casefold_whitespace",
    "punctuation_article",
    "numeric_unit",
    "url_entity",
    "deterministic_any",
)

BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20_260_823
CONFIDENCE_LEVEL = 0.95

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>{}\[\]\"']+"
)
_NUMBER_TOKEN_RE = re.compile(
    r"(?<![\w.])(?:\d+(?:\.\d+)?[km]?)(?![\w.])",
    flags=re.IGNORECASE,
)
_ARTICLES = frozenset({"a", "an", "the"})
_LOW_INFORMATION_ENTITIES = frozenset(
    {
        ("yes",),
        ("no",),
        ("none",),
        ("unknown",),
        ("maybe",),
        ("true",),
        ("false",),
    }
)
_NUMBER_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUMBER_SPECIAL = {"half": 0.5, "quarter": 0.25}
_UNIT_ALIASES = {
    "sec": "second",
    "secs": "second",
    "second": "second",
    "seconds": "second",
    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",
    "hr": "hour",
    "hrs": "hour",
    "hour": "hour",
    "hours": "hour",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "time": "time",
    "times": "time",
    "mile": "mile",
    "miles": "mile",
    "km": "kilometer",
    "kms": "kilometer",
    "kilometer": "kilometer",
    "kilometers": "kilometer",
    "kilometre": "kilometer",
    "kilometres": "kilometer",
    "meter": "meter",
    "meters": "meter",
    "metre": "meter",
    "metres": "meter",
    "ft": "foot",
    "foot": "foot",
    "feet": "foot",
    "inch": "inch",
    "inches": "inch",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "kg": "kilogram",
    "kgs": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "percent": "percent",
    "percentage": "percent",
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "gbp": "gbp",
    "byte": "byte",
    "bytes": "byte",
    "kb": "kilobyte",
    "mb": "megabyte",
    "gb": "gigabyte",
    "tb": "terabyte",
}
_SOURCE_FREE_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "content",
        "generated_token_ids",
        "messages",
        "prompt",
        "question",
        "response_text",
        "sessions",
        "source_id",
        "source_text",
        "turns",
    }
)


class DecodedSummaryError(ValueError):
    """An input or derived metric violated the frozen summary contract."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DecodedSummaryError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DecodedSummaryError(f"non-finite JSON constant {value!r}")


def load_json(path: str | Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DecodedSummaryError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DecodedSummaryError(f"{name} must be a JSON object")
    return value


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


def _require_sha256(value: Any, *, name: str) -> str:
    rendered = str(value or "")
    if _SHA256_RE.fullmatch(rendered) is None:
        raise DecodedSummaryError(f"{name} is not a lowercase SHA-256")
    return rendered


def _validate_integrity(
    value: Mapping[str, Any],
    *,
    name: str,
    expected: str | None = None,
) -> str:
    integrity = value.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) - {"algorithm", "scope", "sha256"}
        or integrity.get("algorithm") != "sha256"
    ):
        raise DecodedSummaryError(f"{name} integrity object differs")
    observed = _require_sha256(integrity.get("sha256"), name=f"{name} integrity")
    body = copy.deepcopy(dict(value))
    body.pop("integrity", None)
    if observed != payload_sha256(body):
        raise DecodedSummaryError(f"{name} canonical integrity differs")
    if expected is not None and observed != expected:
        raise DecodedSummaryError(f"{name} pinned integrity differs")
    return observed


def _restore_legacy_kpar_integer_keys(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    for record in body.get("records") or ():
        try:
            objective = record["conditions"][
                "exact_decrement_or_refit_policy"
            ]["fixed_c_diagnostics"]["objective"]
            kpars = objective["kpar_by_layer"]
        except (KeyError, TypeError) as exc:
            raise DecodedSummaryError(
                "final legacy kpar_by_layer path is missing"
            ) from exc
        if (
            not isinstance(kpars, Mapping)
            or not kpars
            or any(not str(key).isdigit() for key in kpars)
        ):
            raise DecodedSummaryError("final legacy kpar_by_layer keys differ")
        objective["kpar_by_layer"] = {
            int(key): item for key, item in kpars.items()
        }
    return body


def _validate_final_integrity(final: Mapping[str, Any]) -> str:
    integrity = final.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("algorithm") != "sha256"
    ):
        raise DecodedSummaryError("final integrity object differs")
    observed = _require_sha256(
        integrity.get("sha256"),
        name="final legacy integrity",
    )
    regular_body = copy.deepcopy(dict(final))
    regular_body.pop("integrity", None)
    if observed != payload_sha256(regular_body):
        legacy_body = _restore_legacy_kpar_integer_keys(final)
        legacy_body.pop("integrity", None)
        if observed != payload_sha256(legacy_body):
            raise DecodedSummaryError(
                "final does not reproduce its regular or integer-key seal"
            )
    if observed != EXPECTED_FINAL_INTEGRITY_SHA256:
        raise DecodedSummaryError("final pinned payload integrity differs")
    return observed


def _validate_analysis_lock(lock: Mapping[str, Any]) -> None:
    if (
        lock.get("schema") != ANALYSIS_LOCK_SCHEMA
        or lock.get("schema_version") != 3
        or lock.get("status") != "frozen-before-v3-model-scoring"
        or lock.get("contains_source_text") is not False
    ):
        raise DecodedSummaryError("analysis lock schema or status differs")
    body = copy.deepcopy(dict(lock))
    observed = body.pop("lock_sha256", None)
    if (
        observed != payload_sha256(body)
        or observed != EXPECTED_ANALYSIS_LOCK_SHA256
    ):
        raise DecodedSummaryError("analysis lock canonical binding differs")
    selection = lock.get("selection") or {}
    analysis = lock.get("analysis") or {}
    interval = analysis.get("confidence_interval") or {}
    if (
        selection.get("target_clusters") != EXPECTED_CLUSTERS
        or selection.get("histories_per_cluster") != HISTORIES_PER_CLUSTER
        or selection.get("nested_history_instances") != EXPECTED_HISTORIES
        or selection.get("model_outputs_used") is not False
        or selection.get("output_based_replacement") is not False
        or analysis.get("primary_unit") != "target_cluster"
        or analysis.get("primary_n") != EXPECTED_CLUSTERS
        or analysis.get("history_instances_are_nested_repeats") is not True
        or analysis.get("history_instances_are_independent_records") is not False
        or analysis.get("primary_endpoint") != "intent-to-treat"
        or interval.get("method") != "percentile cluster bootstrap"
        or interval.get("resampling_unit") != "target_cluster"
        or interval.get("resamples") != BOOTSTRAP_RESAMPLES
        or interval.get("seed") != BOOTSTRAP_SEED
        or interval.get("confidence_level") != CONFIDENCE_LEVEL
        or interval.get("histories_resampled_within_cluster") is not False
    ):
        raise DecodedSummaryError("analysis lock estimand differs")


def _validate_recovery(
    recovery: Mapping[str, Any],
    *,
    final_file_sha256: str,
    final_integrity_sha256: str,
) -> None:
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("status")
        != "validated-without-evidence-file-modification"
        or recovery.get("source_bearing") is not False
        or recovery.get("record_shard_count") != EXPECTED_HISTORIES
        or recovery.get("all_recorded_payload_hashes_reproduced") is not True
        or recovery.get("authorization_lock_integrity_sha256")
        != EXPECTED_AUTHORIZATION_LOCK_SHA256
    ):
        raise DecodedSummaryError("integrity recovery contract differs")
    recovery_integrity = _validate_integrity(
        recovery,
        name="integrity recovery",
        expected=EXPECTED_RECOVERY_INTEGRITY_SHA256,
    )
    if recovery_integrity != EXPECTED_RECOVERY_INTEGRITY_SHA256:
        raise DecodedSummaryError("integrity recovery pin differs")
    shards = recovery.get("record_shards")
    if not isinstance(shards, list) or len(shards) != EXPECTED_HISTORIES:
        raise DecodedSummaryError("integrity recovery shard list differs")
    expected_paths = [f"records/{index:03d}.json" for index in range(96)]
    if [row.get("path") for row in shards] != expected_paths:
        raise DecodedSummaryError("integrity recovery shard order differs")
    for row in shards:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "path",
                "file_sha256",
                "recorded_payload_sha256",
                "legacy_integer_key_canonicalization_valid",
                "file_modified",
            }
            or row.get("legacy_integer_key_canonicalization_valid") is not True
            or row.get("file_modified") is not False
        ):
            raise DecodedSummaryError("integrity recovery shard disclosure differs")
        _require_sha256(row.get("file_sha256"), name="recovered shard file")
        _require_sha256(
            row.get("recorded_payload_sha256"),
            name="recovered shard payload",
        )
    final_row = recovery.get("final")
    if (
        not isinstance(final_row, Mapping)
        or final_row.get("path") != "final.json"
        or final_row.get("file_sha256") != final_file_sha256
        or final_row.get("recorded_payload_sha256") != final_integrity_sha256
        or final_row.get("legacy_integer_key_canonicalization_valid") is not True
        or final_row.get("file_modified") is not False
    ):
        raise DecodedSummaryError("integrity recovery final binding differs")


def _validate_generation_attempt(
    attempt: Mapping[str, Any],
    *,
    repeat_index: int,
) -> None:
    token_ids = attempt.get("generated_token_ids")
    rendered = attempt.get("response_text")
    if (
        attempt.get("repeat_index") != repeat_index
        or attempt.get("status") != "completed"
        or not isinstance(rendered, str)
        or not isinstance(token_ids, list)
        or not token_ids
        or any(type(token) is not int for token in token_ids)
        or attempt.get("response_utf8_sha256") != text_sha256(rendered)
        or attempt.get("generated_token_ids_sha256")
        != payload_sha256(token_ids)
        or attempt.get("generated_token_count") != len(token_ids)
        or type(attempt.get("prompt_token_count")) is not int
        or int(attempt["prompt_token_count"]) < 1
        or attempt.get("stop_reason")
        not in {"terminal_stop_token", "max_new_tokens"}
    ):
        raise DecodedSummaryError("generation attempt contract differs")


def _validate_final(
    final: Mapping[str, Any],
    *,
    cohort_histories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        final.get("schema") != FINAL_SCHEMA
        or final.get("schema_version") != 2
        or final.get("status") != "completed"
        or final.get("authorization_lock_integrity_sha256")
        != EXPECTED_AUTHORIZATION_LOCK_SHA256
        or final.get("contains_source_text") is not True
        or final.get("contains_model_generated_text") is not True
        or final.get("source_bearing") is not True
        or final.get("local_only") is not True
        or final.get("release_authorized") is not False
        or final.get("decoded_strings_exhaust_extractability") is not False
    ):
        raise DecodedSummaryError("final schema, status, or scope differs")
    records = final.get("records")
    ordered = final.get("ordered_history_ids")
    expected_ids = [str(history["record_id"]) for history in cohort_histories]
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_HISTORIES
        or ordered != expected_ids
        or [record.get("record_id") for record in records] != expected_ids
    ):
        raise DecodedSummaryError("final all-history order differs")
    cohort_by_id = {
        str(history["record_id"]): history for history in cohort_histories
    }
    rows: list[dict[str, Any]] = []
    calls = 0
    for record in records:
        record_id = str(record["record_id"])
        frozen = cohort_by_id[record_id]
        if (
            record.get("cluster_id") != frozen.get("cluster_id")
            or record.get("variant_index") != frozen.get("variant_index")
            or record.get("status") != "completed"
            or record.get("contains_source_text") is not True
            or record.get("contains_model_generated_text") is not True
            or record.get("decoded_strings_exhaust_extractability") is not False
            or record.get("failed_condition_ids") != []
            or record.get("generation_calls_started") != 16
            or record.get("omitted_decoded_conditions")
            != {
                "condition_ids": [
                    "fp32_proxy",
                    "decay_0_01",
                    "token_row_repack_alias",
                    "separate_fixed_c_refit",
                ],
                "remain_teacher_forced_or_other_evidence": True,
            }
        ):
            raise DecodedSummaryError("final history identity or status differs")
        conditions = record.get("conditions")
        if not isinstance(conditions, Mapping) or tuple(conditions) != CONDITION_IDS:
            raise DecodedSummaryError("final condition matrix differs")
        row: dict[str, Any] = {
            "record_id": record_id,
            "cluster_id": str(record["cluster_id"]),
            "variant_index": int(record["variant_index"]),
            "conditions": {},
        }
        for condition_id in CONDITION_IDS:
            condition = conditions[condition_id]
            probes = condition.get("probes")
            if (
                condition.get("condition_id") != condition_id
                or condition.get("status") != "completed"
                or condition.get("generation_calls_started") != 4
                or not isinstance(probes, list)
                or [probe.get("probe_id") for probe in probes]
                != list(PROBE_IDS)
            ):
                raise DecodedSummaryError("completed condition contract differs")
            row["conditions"][condition_id] = {}
            for probe, probe_id in zip(probes, PROBE_IDS):
                attempts = probe.get("generation_attempts")
                checks = probe.get("response_checks")
                repeat = probe.get("repeat_check")
                expected_kind = "deleted" if probe_id == "target_current" else "retained"
                if (
                    probe.get("probe_kind") != expected_kind
                    or probe.get("status") != "completed"
                    or probe.get("generation_attempts_started") != 2
                    or probe.get("generation_attempts_completed") != 2
                    or not isinstance(attempts, list)
                    or len(attempts) != 2
                    or not isinstance(checks, Mapping)
                    or set(checks)
                    != {
                        "answer_exact_match",
                        "answer_normalized_match",
                        "answer_casefold_substring",
                        "other_answer_leakage_casefold_substring",
                        "decoded_strings_exhaust_extractability",
                    }
                    or checks.get("decoded_strings_exhaust_extractability")
                    is not False
                    or any(
                        type(checks[key]) is not bool
                        for key in (
                            "answer_exact_match",
                            "answer_normalized_match",
                            "answer_casefold_substring",
                            "other_answer_leakage_casefold_substring",
                        )
                    )
                    or repeat
                    != {
                        "repetitions": 2,
                        "exact_token_ids_and_response_text_match": True,
                    }
                ):
                    raise DecodedSummaryError("completed probe contract differs")
                for index, attempt in enumerate(attempts):
                    _validate_generation_attempt(attempt, repeat_index=index)
                if (
                    attempts[0]["response_text"] != attempts[1]["response_text"]
                    or attempts[0]["generated_token_ids"]
                    != attempts[1]["generated_token_ids"]
                ):
                    raise DecodedSummaryError("repeat equality disclosure differs")
                row["conditions"][condition_id][probe_id] = {
                    "text": attempts[0]["response_text"],
                    "checks": dict(checks),
                }
                calls += 2
        rows.append(row)
    expected_summary = {
        "history_slots": EXPECTED_HISTORIES,
        "completed_histories": EXPECTED_HISTORIES,
        "terminal_failed_or_nondeterministic_histories": 0,
        "generation_calls_started": EXPECTED_CALLS,
        "authorized_generation_calls": EXPECTED_CALLS,
        "all_96_histories_retained": True,
        "outcome_based_filtering": False,
    }
    if calls != EXPECTED_CALLS or final.get("summary") != expected_summary:
        raise DecodedSummaryError("final all-call accounting differs")
    return rows


def _validate_cohort(
    cohort: Mapping[str, Any],
    *,
    analysis_lock: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    _validate_integrity(
        cohort,
        name="cohort",
        expected=EXPECTED_COHORT_INTEGRITY_SHA256,
    )
    if (
        cohort.get("schema") != COHORT_SCHEMA
        or cohort.get("schema_version") != 3
        or cohort.get("status") != "frozen-before-v3-model-scoring"
        or cohort.get("contains_source_text") is not False
        or cohort.get("policy_lock") != analysis_lock
        or cohort.get("analysis_contract") != analysis_lock.get("analysis")
        or cohort.get("selection_policy") != analysis_lock.get("selection")
    ):
        raise DecodedSummaryError("cohort schema or analysis binding differs")
    counts = cohort.get("counts") or {}
    clusters = cohort.get("clusters")
    histories = cohort.get("histories")
    if (
        counts
        != {
            "target_clusters": EXPECTED_CLUSTERS,
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "nested_history_instances": EXPECTED_HISTORIES,
            "independent_analysis_n": EXPECTED_CLUSTERS,
        }
        or not isinstance(clusters, list)
        or len(clusters) != EXPECTED_CLUSTERS
        or not isinstance(histories, list)
        or len(histories) != EXPECTED_HISTORIES
    ):
        raise DecodedSummaryError("cohort denominators differ")
    history_by_id = {
        str(history.get("record_id")): history for history in histories
    }
    if len(history_by_id) != EXPECTED_HISTORIES or "" in history_by_id:
        raise DecodedSummaryError("cohort history IDs are invalid")
    metadata: list[dict[str, Any]] = []
    ordered_ids: list[str] = []
    for index, cluster in enumerate(clusters):
        history_ids = cluster.get("history_ids")
        if (
            cluster.get("cluster_index") != index
            or cluster.get("nested_history_count") != HISTORIES_PER_CLUSTER
            or not isinstance(history_ids, list)
            or len(history_ids) != HISTORIES_PER_CLUSTER
            or len(set(history_ids)) != HISTORIES_PER_CLUSTER
            or cluster.get("target_exposure_class")
            not in {"direct-target", "context-only", "source-unseen"}
        ):
            raise DecodedSummaryError("cohort cluster geometry differs")
        target_hash = _require_sha256(
            (cluster.get("target") or {}).get("current_answer_sha256"),
            name="cluster target reference",
        )
        retained_hash = _require_sha256(
            (cluster.get("retained_control") or {}).get("current_answer_sha256"),
            name="cluster retained reference",
        )
        for variant_index, history_id in enumerate(history_ids):
            history = history_by_id.get(str(history_id))
            if (
                history is None
                or history.get("cluster_id") != cluster.get("cluster_id")
                or history.get("variant_index") != variant_index
            ):
                raise DecodedSummaryError("cohort nested history binding differs")
            probes = history.get("probes")
            if (
                not isinstance(probes, list)
                or [probe.get("probe_id") for probe in probes]
                != list(PROBE_IDS)
                or probes[0].get("answer_sha256") != target_hash
                or probes[1].get("answer_sha256") != retained_hash
            ):
                raise DecodedSummaryError("cohort probe reference binding differs")
            ordered_ids.append(str(history_id))
        metadata.append(
            {
                "cluster_index": index,
                "cluster_id": str(cluster["cluster_id"]),
                "history_ids": list(history_ids),
                "target_exposure_class": str(cluster["target_exposure_class"]),
                "reference_hashes": {
                    "target_current": target_hash,
                    "retained": retained_hash,
                },
            }
        )
    if ordered_ids != [str(history["record_id"]) for history in histories]:
        raise DecodedSummaryError("cohort history order is not cluster-major")
    return histories, metadata


def normalize_casefold_whitespace(value: str) -> str:
    """Normalize case and whitespace only."""

    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def normalize_punctuation_articles(value: str) -> str:
    """Normalize punctuation and English articles without semantic inference."""

    folded = normalize_casefold_whitespace(value)
    tokens = re.findall(r"\w+", folded, flags=re.UNICODE)
    return " ".join(token for token in tokens if token not in _ARTICLES)


def _format_number(value: float | int) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DecodedSummaryError("numeric canonicalization produced non-finite value")
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.12f}".rstrip("0").rstrip(".")


def _consume_number_words(tokens: Sequence[str], start: int) -> tuple[str, int] | None:
    first = tokens[start]
    if first in _NUMBER_SPECIAL:
        return _format_number(_NUMBER_SPECIAL[first]), start + 1
    if (
        first not in _NUMBER_SMALL
        and first not in _NUMBER_TENS
        and first not in _NUMBER_SCALES
    ):
        return None
    total = 0
    current = 0
    index = start
    consumed_numeric = False
    while index < len(tokens):
        token = tokens[index]
        if token == "and" and consumed_numeric:
            index += 1
            continue
        if token in _NUMBER_SMALL:
            current += _NUMBER_SMALL[token]
        elif token in _NUMBER_TENS:
            current += _NUMBER_TENS[token]
        elif token == "hundred" and consumed_numeric:
            current = max(current, 1) * 100
        elif token in {"thousand", "million"} and consumed_numeric:
            total += max(current, 1) * _NUMBER_SCALES[token]
            current = 0
        else:
            break
        consumed_numeric = True
        index += 1
    if not consumed_numeric:
        return None
    return str(total + current), index


def normalize_numeric_units(value: str) -> tuple[str, bool]:
    """Canonicalize declared number words, numeric forms, currencies, and units."""

    folded = unicodedata.normalize("NFKC", str(value)).casefold()
    folded = re.sub(r"(?<=\d),(?=\d)", "", folded)
    folded = folded.replace("$", " usd ").replace("€", " eur ").replace("£", " gbp ")
    folded = re.sub(
        r"(?<!\w)(\d+(?:\.\d+)?)k(?!\w)",
        lambda match: _format_number(float(match.group(1)) * 1_000),
        folded,
    )
    folded = re.sub(
        r"(?<!\w)(\d+(?:\.\d+)?)m(?!\w)",
        lambda match: _format_number(float(match.group(1)) * 1_000_000),
        folded,
    )
    tokens = re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+", folded, flags=re.UNICODE)
    output: list[str] = []
    has_numeric_or_unit = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"once", "twice", "thrice"}:
            output.extend(
                {
                    "once": ("1", "time"),
                    "twice": ("2", "time"),
                    "thrice": ("3", "time"),
                }[token]
            )
            has_numeric_or_unit = True
            index += 1
            continue
        parsed = _consume_number_words(tokens, index)
        if parsed is not None:
            number, index = parsed
            output.append(number)
            has_numeric_or_unit = True
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            output.append(_format_number(float(token)))
            has_numeric_or_unit = True
        elif token in _UNIT_ALIASES:
            output.append(_UNIT_ALIASES[token])
            has_numeric_or_unit = True
        elif token == "per":
            pass
        elif token not in _ARTICLES:
            output.append(token)
        index += 1
    for index in range(len(output) - 1):
        if output[index] in {"usd", "eur", "gbp"} and re.fullmatch(
            r"\d+(?:\.\d+)?", output[index + 1]
        ):
            output[index], output[index + 1] = output[index + 1], output[index]
    return " ".join(output), has_numeric_or_unit


def _canonical_url(value: str) -> str | None:
    rendered = value.rstrip(".,;:!?)]}")
    if rendered.casefold().startswith("www."):
        rendered = "https://" + rendered
    try:
        parts = urlsplit(rendered)
    except ValueError:
        return None
    host = (parts.hostname or "").casefold()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    path = unquote(parts.path or "").rstrip("/")
    query = [
        (key.casefold(), value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
    ]
    if host in {"youtube.com", "m.youtube.com"} and path == "/watch":
        video_ids = [item for key, item in query if key == "v" and item]
        if video_ids:
            return "youtube:" + video_ids[0]
    if host == "youtu.be" and path.strip("/"):
        return "youtube:" + path.strip("/").split("/", 1)[0]
    query_text = urlencode(sorted(query), doseq=True)
    return host + path + (("?" + query_text) if query_text else "")


def _entity_tokens(value: str) -> tuple[str, ...]:
    without_urls = _URL_RE.sub(" ", value)
    normalized = normalize_punctuation_articles(without_urls)
    return tuple(normalized.split())


def _contains_contiguous(
    larger: Sequence[str],
    smaller: Sequence[str],
) -> bool:
    if not smaller or len(smaller) > len(larger):
        return False
    return any(
        tuple(larger[index : index + len(smaller)]) == tuple(smaller)
        for index in range(len(larger) - len(smaller) + 1)
    )


def url_entity_match(left: str, right: str) -> bool:
    """Match canonical URLs or declared lexical entity containment only."""

    left_urls = tuple(
        canonical
        for match in _URL_RE.findall(left)
        if (canonical := _canonical_url(match)) is not None
    )
    right_urls = tuple(
        canonical
        for match in _URL_RE.findall(right)
        if (canonical := _canonical_url(match)) is not None
    )
    if left_urls and right_urls and set(left_urls).intersection(right_urls):
        return True
    left_tokens = _entity_tokens(left)
    right_tokens = _entity_tokens(right)
    if left_tokens in _LOW_INFORMATION_ENTITIES or right_tokens in _LOW_INFORMATION_ENTITIES:
        return False
    shorter = left_tokens if len(left_tokens) <= len(right_tokens) else right_tokens
    larger = right_tokens if shorter is left_tokens else left_tokens
    informative = (
        len(shorter) >= 2
        or (
            len(shorter) == 1
            and len(shorter[0]) >= 4
            and any(character.isalpha() for character in shorter[0])
        )
    )
    return bool(informative and _contains_contiguous(larger, shorter))


def deterministic_matches(left: str, right: str) -> dict[str, bool]:
    """Return cumulative lexical comparisons; no semantic label is inferred."""

    exact = str(left).strip() == str(right).strip()
    casefold = normalize_casefold_whitespace(left) == normalize_casefold_whitespace(
        right
    )
    punctuation = normalize_punctuation_articles(
        left
    ) == normalize_punctuation_articles(right)
    left_numeric, left_has_numeric = normalize_numeric_units(left)
    right_numeric, right_has_numeric = normalize_numeric_units(right)
    numeric_relation = bool(
        left_has_numeric
        and right_has_numeric
        and left_numeric
        and left_numeric == right_numeric
    )
    numeric = bool(punctuation or numeric_relation)
    url_entity = bool(numeric or url_entity_match(left, right))
    return {
        "trimmed_exact": exact,
        "casefold_whitespace": casefold,
        "punctuation_article": punctuation,
        "numeric_unit": numeric,
        "url_entity": url_entity,
        "deterministic_any": url_entity,
    }


def _reference_from_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    probe_id: str,
    expected_sha256: str,
) -> str | None:
    candidates: set[str] = set()
    for row in rows:
        for condition_id in ("present", "fresh_raw_omission"):
            rendered = str(
                row["conditions"][condition_id][probe_id]["text"]
            )
            for candidate in (rendered, rendered.strip()):
                if text_sha256(candidate) == expected_sha256:
                    candidates.add(candidate)
    if len(candidates) > 1:
        raise DecodedSummaryError("control-only reference recovery is ambiguous")
    return None if not candidates else next(iter(candidates))


class _SplitMix64:
    """Small specified PRNG used only for deterministic bootstrap indices."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & self._MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (value ^ (value >> 31)) & self._MASK

    def randbelow(self, upper: int) -> int:
        if upper < 1:
            raise DecodedSummaryError("bootstrap upper bound must be positive")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise DecodedSummaryError("cannot take percentile of empty values")
    position = (len(sorted_values) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


@lru_cache(maxsize=None)
def _cluster_bootstrap_interval_cached(
    values: tuple[float, ...],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if len(values) < 2:
        raise DecodedSummaryError("cluster bootstrap requires at least two clusters")
    generator = _SplitMix64(seed)
    count = len(values)
    estimates = [
        math.fsum(
            values[generator.randbelow(count)] for _ in range(count)
        )
        / count
        for _ in range(resamples)
    ]
    estimates.sort()
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return (
        _percentile(estimates, alpha),
        _percentile(estimates, 1.0 - alpha),
    )


def cluster_bootstrap_interval(
    values: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """Percentile interval after resampling whole target clusters."""

    rendered = tuple(float(value) for value in values)
    if len(rendered) < 2:
        return None
    lower, upper = _cluster_bootstrap_interval_cached(
        rendered,
        resamples=resamples,
        seed=seed,
    )
    return {
        "method": "percentile_cluster_bootstrap",
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling_unit": "target_cluster",
        "histories_resampled_within_cluster": False,
        "resamples": resamples,
        "seed": seed,
        "lower": lower,
        "upper": upper,
        "degenerate_empirical_distribution": len(set(rendered)) == 1,
    }


def wilson_interval(successes: int, denominator: int) -> dict[str, Any] | None:
    """Wilson 95% interval for a complete-cluster binary rate."""

    if denominator < 1:
        return None
    if not 0 <= successes <= denominator:
        raise DecodedSummaryError("Wilson numerator is outside denominator")
    z = 1.959963984540054
    rate = successes / denominator
    z2 = z * z
    center = (rate + z2 / (2 * denominator)) / (1 + z2 / denominator)
    half = (
        z
        * math.sqrt(
            rate * (1 - rate) / denominator
            + z2 / (4 * denominator * denominator)
        )
        / (1 + z2 / denominator)
    )
    return {
        "method": "wilson_score",
        "confidence_level": CONFIDENCE_LEVEL,
        "denominator_unit": "target_cluster",
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def _endpoint_rows(
    cluster_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    success: Callable[[Mapping[str, Any]], bool],
    eligible: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for rows in cluster_rows:
        selected = [
            row for row in rows if eligible is None or bool(eligible(row))
        ]
        numerator = sum(bool(success(row)) for row in selected)
        denominator = len(selected)
        result.append(
            {
                "numerator_histories": numerator,
                "denominator_histories": denominator,
                "rate": None if denominator == 0 else numerator / denominator,
            }
        )
    return result


def _summarize_endpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_interval: bool,
) -> dict[str, Any]:
    eligible = [row for row in rows if int(row["denominator_histories"]) > 0]
    numerator = sum(int(row["numerator_histories"]) for row in eligible)
    denominator = sum(int(row["denominator_histories"]) for row in eligible)
    values = [float(row["rate"]) for row in eligible]
    complete = sum(
        int(row["numerator_histories"]) == int(row["denominator_histories"])
        for row in eligible
    )
    result: dict[str, Any] = {
        "numerator_histories": numerator,
        "denominator_histories": denominator,
        "eligible_target_clusters": len(eligible),
        "ineligible_target_clusters": EXPECTED_CLUSTERS - len(eligible),
        "history_rate": None if denominator == 0 else numerator / denominator,
        "cluster_mean": None if not values else statistics.fmean(values),
        "complete_cluster_successes": complete,
        "complete_cluster_denominator": len(eligible),
        "complete_cluster_rate": None if not eligible else complete / len(eligible),
    }
    if include_interval:
        result["cluster_bootstrap_95_interval"] = cluster_bootstrap_interval(values)
        result["complete_cluster_wilson_95_interval"] = wilson_interval(
            complete,
            len(eligible),
        )
    return result


def _registered_match(row: Mapping[str, Any], condition_id: str, probe_id: str, metric: str) -> bool:
    checks = row["conditions"][condition_id][probe_id]["checks"]
    key = {
        "trimmed_exact": "answer_exact_match",
        "casefold_whitespace": "answer_normalized_match",
        "casefold_substring": "answer_casefold_substring",
    }[metric]
    return bool(checks[key])


def _comparison(
    row: Mapping[str, Any],
    left_condition: str,
    right_condition: str,
    probe_id: str,
    metric: str,
) -> bool:
    return deterministic_matches(
        row["conditions"][left_condition][probe_id]["text"],
        row["conditions"][right_condition][probe_id]["text"],
    )[metric]


def _reference_match(
    row: Mapping[str, Any],
    condition_id: str,
    probe_id: str,
    metric: str,
) -> bool:
    reference = row["references"][probe_id]
    if reference is None:
        return False
    return deterministic_matches(
        row["conditions"][condition_id][probe_id]["text"],
        reference,
    )[metric]


def _source_free_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _source_free_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _source_free_keys(nested)


def _source_free_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _source_free_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _source_free_strings(nested)


def assert_source_free_summary(
    summary: Mapping[str, Any],
    *,
    generated_strings: Iterable[str] = (),
) -> None:
    forbidden = _SOURCE_FREE_FORBIDDEN_KEYS.intersection(_source_free_keys(summary))
    if forbidden:
        raise DecodedSummaryError(
            f"published summary contains forbidden keys: {sorted(forbidden)}"
        )
    published_strings = set(_source_free_strings(summary))
    generated = {
        str(value)
        for value in generated_strings
        if len(str(value).strip()) >= 8
    }
    overlap = published_strings.intersection(generated)
    if overlap:
        raise DecodedSummaryError("published summary contains generated text")


def _metric_map(
    cluster_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    success_factory: Callable[[str], Callable[[Mapping[str, Any]], bool]],
    eligible_factory: (
        Callable[[str], Callable[[Mapping[str, Any]], bool]] | None
    ) = None,
    metrics: Sequence[str] = NORMALIZATION_IDS,
    include_interval: bool,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    aggregate: dict[str, Any] = {}
    per_cluster: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        endpoint_rows = _endpoint_rows(
            cluster_rows,
            success=success_factory(metric),
            eligible=(
                None if eligible_factory is None else eligible_factory(metric)
            ),
        )
        aggregate[metric] = _summarize_endpoint(
            endpoint_rows,
            include_interval=include_interval,
        )
        per_cluster[metric] = endpoint_rows
    return aggregate, per_cluster


def build_summary(
    *,
    final: Mapping[str, Any],
    recovery: Mapping[str, Any],
    cohort: Mapping[str, Any],
    analysis_lock: Mapping[str, Any],
    final_file_sha256: str,
    recovery_file_sha256: str,
    cohort_file_sha256: str,
    analysis_lock_file_sha256: str,
) -> dict[str, Any]:
    """Validate all frozen inputs and return a source-free aggregate."""

    for value, name in (
        (final_file_sha256, "final file"),
        (recovery_file_sha256, "recovery file"),
        (cohort_file_sha256, "cohort file"),
        (analysis_lock_file_sha256, "analysis lock file"),
    ):
        _require_sha256(value, name=name)
    if final_file_sha256 != EXPECTED_FINAL_FILE_SHA256:
        raise DecodedSummaryError("final pinned file SHA-256 differs")
    if recovery_file_sha256 != EXPECTED_RECOVERY_FILE_SHA256:
        raise DecodedSummaryError("recovery pinned file SHA-256 differs")
    if cohort_file_sha256 != EXPECTED_COHORT_FILE_SHA256:
        raise DecodedSummaryError("cohort pinned file SHA-256 differs")
    if analysis_lock_file_sha256 != EXPECTED_ANALYSIS_LOCK_FILE_SHA256:
        raise DecodedSummaryError("analysis lock pinned file SHA-256 differs")
    _validate_analysis_lock(analysis_lock)
    histories, cluster_metadata = _validate_cohort(
        cohort,
        analysis_lock=analysis_lock,
    )
    final_integrity = _validate_final_integrity(final)
    rows = _validate_final(final, cohort_histories=histories)
    _validate_recovery(
        recovery,
        final_file_sha256=final_file_sha256,
        final_integrity_sha256=final_integrity,
    )

    rows_by_id = {str(row["record_id"]): row for row in rows}
    cluster_rows: list[list[dict[str, Any]]] = []
    reference_recovery: list[dict[str, bool]] = []
    generated_strings: list[str] = []
    for metadata in cluster_metadata:
        nested = [rows_by_id[str(item)] for item in metadata["history_ids"]]
        references = {
            probe_id: _reference_from_controls(
                nested,
                probe_id=probe_id,
                expected_sha256=metadata["reference_hashes"][probe_id],
            )
            for probe_id in PROBE_IDS
        }
        for row in nested:
            row["references"] = references
            for condition_id in CONDITION_IDS:
                for probe_id in PROBE_IDS:
                    generated_strings.append(
                        row["conditions"][condition_id][probe_id]["text"]
                    )
        cluster_rows.append(nested)
        reference_recovery.append(
            {probe_id: references[probe_id] is not None for probe_id in PROBE_IDS}
        )

    registered_rates: dict[str, Any] = {}
    registered_cluster_rows: dict[str, Any] = {}
    for probe_id in PROBE_IDS:
        registered_rates[probe_id] = {}
        registered_cluster_rows[probe_id] = {}
        for condition_id in CONDITION_IDS:
            aggregate, per_cluster = _metric_map(
                cluster_rows,
                success_factory=lambda metric, condition_id=condition_id, probe_id=probe_id: (
                    lambda row: _registered_match(
                        row,
                        condition_id,
                        probe_id,
                        metric,
                    )
                ),
                metrics=REGISTERED_MATCH_IDS,
                include_interval=False,
            )
            registered_rates[probe_id][condition_id] = aggregate
            registered_cluster_rows[probe_id][condition_id] = per_cluster

    reference_rates: dict[str, Any] = {}
    reference_cluster_rows: dict[str, Any] = {}
    for probe_id in PROBE_IDS:
        reference_rates[probe_id] = {}
        reference_cluster_rows[probe_id] = {}
        for condition_id in CONDITION_IDS:
            aggregate, per_cluster = _metric_map(
                cluster_rows,
                success_factory=lambda metric, condition_id=condition_id, probe_id=probe_id: (
                    lambda row: _reference_match(
                        row,
                        condition_id,
                        probe_id,
                        metric,
                    )
                ),
                eligible_factory=lambda _metric, probe_id=probe_id: (
                    lambda row: row["references"][probe_id] is not None
                ),
                include_interval=False,
            )
            reference_rates[probe_id][condition_id] = aggregate
            reference_cluster_rows[probe_id][condition_id] = per_cluster

    policy_rebuild: dict[str, Any] = {}
    policy_rebuild_cluster_rows: dict[str, Any] = {}
    for probe_id in PROBE_IDS:
        aggregate, per_cluster = _metric_map(
            cluster_rows,
            success_factory=lambda metric, probe_id=probe_id: (
                lambda row: _comparison(
                    row,
                    "exact_decrement_or_refit_policy",
                    "fresh_raw_omission",
                    probe_id,
                    metric,
                )
            ),
            include_interval=True,
        )
        policy_rebuild[probe_id] = aggregate
        policy_rebuild_cluster_rows[probe_id] = per_cluster

    retained_policy_present, retained_policy_present_cluster_rows = _metric_map(
        cluster_rows,
        success_factory=lambda metric: (
            lambda row: _comparison(
                row,
                "exact_decrement_or_refit_policy",
                "present",
                "retained",
                metric,
            )
        ),
        include_interval=True,
    )

    prompt_dissociation, prompt_dissociation_cluster_rows = _metric_map(
        cluster_rows,
        success_factory=lambda metric: (
            lambda row: (
                _comparison(
                    row,
                    "exact_decrement_or_refit_policy",
                    "fresh_raw_omission",
                    "target_current",
                    metric,
                )
                and _comparison(
                    row,
                    "prompt_suppression",
                    "present",
                    "target_current",
                    metric,
                )
            )
        ),
        eligible_factory=lambda metric: (
            lambda row: not _comparison(
                row,
                "present",
                "fresh_raw_omission",
                "target_current",
                metric,
            )
        ),
        include_interval=True,
    )

    strict_target_rows = _endpoint_rows(
        cluster_rows,
        eligible=lambda row: (
            _registered_match(
                row,
                "present",
                "target_current",
                "casefold_substring",
            )
            and not _registered_match(
                row,
                "fresh_raw_omission",
                "target_current",
                "casefold_substring",
            )
        ),
        success=lambda row: not _registered_match(
            row,
            "exact_decrement_or_refit_policy",
            "target_current",
            "casefold_substring",
        ),
    )
    strict_prompt_rows = _endpoint_rows(
        cluster_rows,
        eligible=lambda row: (
            _registered_match(
                row,
                "present",
                "target_current",
                "casefold_substring",
            )
            and not _registered_match(
                row,
                "fresh_raw_omission",
                "target_current",
                "casefold_substring",
            )
        ),
        success=lambda row: _registered_match(
            row,
            "prompt_suppression",
            "target_current",
            "casefold_substring",
        ),
    )
    strict_joint_rows = _endpoint_rows(
        cluster_rows,
        eligible=lambda row: (
            _registered_match(
                row,
                "present",
                "target_current",
                "casefold_substring",
            )
            and not _registered_match(
                row,
                "fresh_raw_omission",
                "target_current",
                "casefold_substring",
            )
        ),
        success=lambda row: (
            not _registered_match(
                row,
                "exact_decrement_or_refit_policy",
                "target_current",
                "casefold_substring",
            )
            and _registered_match(
                row,
                "prompt_suppression",
                "target_current",
                "casefold_substring",
            )
        ),
    )
    strict_retained_rows = _endpoint_rows(
        cluster_rows,
        eligible=lambda row: (
            _registered_match(
                row,
                "present",
                "retained",
                "casefold_substring",
            )
            and _registered_match(
                row,
                "fresh_raw_omission",
                "retained",
                "casefold_substring",
            )
        ),
        success=lambda row: _registered_match(
            row,
            "exact_decrement_or_refit_policy",
            "retained",
            "casefold_substring",
        ),
    )
    itt_target_nonleak_rows = _endpoint_rows(
        cluster_rows,
        success=lambda row: not _registered_match(
            row,
            "exact_decrement_or_refit_policy",
            "target_current",
            "casefold_substring",
        ),
    )
    itt_retained_rows = _endpoint_rows(
        cluster_rows,
        success=lambda row: _registered_match(
            row,
            "exact_decrement_or_refit_policy",
            "retained",
            "casefold_substring",
        ),
    )

    per_cluster: list[dict[str, Any]] = []
    for index, metadata in enumerate(cluster_metadata):
        per_cluster.append(
            {
                "cluster_index": index,
                "target_exposure_class": metadata["target_exposure_class"],
                "history_count": HISTORIES_PER_CLUSTER,
                "control_reference_recovered": reference_recovery[index],
                "registered_match": {
                    probe_id: {
                        condition_id: {
                            metric: registered_cluster_rows[probe_id][
                                condition_id
                            ][metric][index]["rate"]
                            for metric in REGISTERED_MATCH_IDS
                        }
                        for condition_id in CONDITION_IDS
                    }
                    for probe_id in PROBE_IDS
                },
                "reference_normalized_match": {
                    probe_id: {
                        condition_id: {
                            metric: reference_cluster_rows[probe_id][
                                condition_id
                            ][metric][index]["rate"]
                            for metric in NORMALIZATION_IDS
                        }
                        for condition_id in CONDITION_IDS
                    }
                    for probe_id in PROBE_IDS
                },
                "policy_vs_fresh_rebuild": {
                    probe_id: {
                        metric: policy_rebuild_cluster_rows[probe_id][metric][
                            index
                        ]["rate"]
                        for metric in NORMALIZATION_IDS
                    }
                    for probe_id in PROBE_IDS
                },
                "prompt_control_dissociation": {
                    metric: prompt_dissociation_cluster_rows[metric][index]
                    for metric in NORMALIZATION_IDS
                },
                "retained_policy_vs_present": {
                    metric: retained_policy_present_cluster_rows[metric][index][
                        "rate"
                    ]
                    for metric in NORMALIZATION_IDS
                },
                "strict_control_only_subsets": {
                    "target_policy_nonleak": strict_target_rows[index],
                    "prompt_still_reveals_target": strict_prompt_rows[index],
                    "joint_state_edit_prompt_dissociation": strict_joint_rows[
                        index
                    ],
                    "retained_policy_utility": strict_retained_rows[index],
                },
                "intent_to_treat": {
                    "target_policy_nonleak": itt_target_nonleak_rows[index][
                        "rate"
                    ],
                    "retained_policy_registered_match": itt_retained_rows[index][
                        "rate"
                    ],
                },
            }
        )

    final_provenance = {
        "path": (
            "outputs/gemma_sv_rag/"
            "longmemeval_chat_response_generation_audit_v2/final.json"
        ),
        "file_sha256": final_file_sha256,
        "payload_integrity_sha256": final_integrity,
        "authorization_lock_integrity_sha256": (
            EXPECTED_AUTHORIZATION_LOCK_SHA256
        ),
    }
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "schema_version": 2,
        "status": "complete",
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "contains_model_generated_text": False,
        "semantic_equivalence_labeled": False,
        "llm_calls_made": 0,
        "provenance": {
            "decoded_final": final_provenance,
            "integrity_recovery": {
                "path": (
                    "outputs/gemma_sv_rag/"
                    "longmemeval_chat_response_generation_audit_v2/"
                    "integrity_recovery_v1.json"
                ),
                "file_sha256": recovery_file_sha256,
                "payload_integrity_sha256": recovery["integrity"]["sha256"],
                "authorization_lock_integrity_sha256": (
                    EXPECTED_AUTHORIZATION_LOCK_SHA256
                ),
            },
            "cohort": {
                "path": "gemma_sv/benchmarks/longmemeval_chat_cohort_v3.json",
                "file_sha256": cohort_file_sha256,
                "payload_integrity_sha256": cohort["integrity"]["sha256"],
            },
            "analysis_lock": {
                "path": (
                    "gemma_sv/benchmarks/"
                    "longmemeval_chat_cluster_analysis_lock_v3.json"
                ),
                "file_sha256": analysis_lock_file_sha256,
                "lock_sha256": analysis_lock["lock_sha256"],
            },
        },
        "integrity_recovery_disclosure": {
            "known_issue": "legacy_integer_key_sealing",
            "status": "validated_without_evidence_file_modification",
            "affected_object_key": "kpar_by_layer",
            "explanation": (
                "The original in-memory seals sorted integer layer keys "
                "numerically; JSON reload represented them as strings. "
                "Restoring only that known key type reproduces all 96 shard "
                "payload hashes. Compatibility assembly then sealed final.json "
                "in its loaded string-key representation, whose regular "
                "canonical payload hash validates."
            ),
            "record_shard_payload_hashes_recovered": EXPECTED_HISTORIES,
            "final_regular_payload_hash_valid": True,
            "evidence_files_modified": False,
        },
        "analysis": {
            "primary_unit": "target_cluster",
            "primary_target_clusters": EXPECTED_CLUSTERS,
            "nested_histories_per_cluster": HISTORIES_PER_CLUSTER,
            "nested_history_instances": EXPECTED_HISTORIES,
            "history_instances_are_independent": False,
            "primary_population": "intent_to_treat_all_frozen_histories",
            "control_only_subset_rule": (
                "Present and fresh-rebuild outcomes alone define informative "
                "subsets; edited-policy and prompt-control outcomes never "
                "filter, replace, or admit histories."
            ),
            "bootstrap": {
                "method": "percentile_cluster_bootstrap",
                "confidence_level": CONFIDENCE_LEVEL,
                "resampling_unit": "target_cluster",
                "histories_resampled_within_cluster": False,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "percentile_interpolation": "linear_type_7",
                "prng": "splitmix64",
            },
            "binary_complete_cluster_interval": {
                "method": "wilson_score",
                "confidence_level": CONFIDENCE_LEVEL,
                "denominator_unit": "target_cluster",
            },
        },
        "denominators": {
            "target_clusters": EXPECTED_CLUSTERS,
            "histories_nested_within_clusters": EXPECTED_HISTORIES,
            "histories_per_cluster": HISTORIES_PER_CLUSTER,
            "conditions_per_history": len(CONDITION_IDS),
            "probes_per_condition": len(PROBE_IDS),
            "deterministic_repeats_per_probe": 2,
            "generation_calls": EXPECTED_CALLS,
            "completed_generation_calls": EXPECTED_CALLS,
            "failed_generation_calls": 0,
            "completed_histories": EXPECTED_HISTORIES,
            "failed_or_nondeterministic_histories": 0,
            "target_control_recoverable_clusters": sum(
                item["target_current"] for item in reference_recovery
            ),
            "retained_control_recoverable_clusters": sum(
                item["retained"] for item in reference_recovery
            ),
        },
        "normalization_contract": {
            "reference_recovery": (
                "A reference string is available only when a present or "
                "fresh-rebuild control string, after optional outer-space "
                "trimming, reproduces the cohort's frozen UTF-8 digest. "
                "Edited-policy and prompt-control strings are excluded."
            ),
            "reference_strings_emitted": False,
            "tiers_are_cumulative": True,
            "tiers": {
                "trimmed_exact": "outer-space-trimmed code-point equality",
                "casefold_whitespace": (
                    "Unicode NFKC, casefold, and whitespace-collapse equality"
                ),
                "punctuation_article": (
                    "casefold plus punctuation removal and English "
                    "a/an/the removal"
                ),
                "numeric_unit": (
                    "punctuation/article match or declared number-word, "
                    "numeric, currency, and unit canonicalization equality"
                ),
                "url_entity": (
                    "numeric/unit match, canonical URL identity, or declared "
                    "contiguous lexical entity containment"
                ),
                "deterministic_any": (
                    "union of the five declared deterministic comparisons"
                ),
            },
            "not_claimed": (
                "No tier labels paraphrase or semantic equivalence; lexical "
                "relations may have false positives and false negatives."
            ),
        },
        "metrics": {
            "registered_reference_match": registered_rates,
            "control_recovered_reference_match": reference_rates,
            "policy_vs_fresh_rebuild": policy_rebuild,
            "prompt_control_dissociation": {
                "definition": (
                    "Among histories where present and fresh rebuild differ "
                    "under the same deterministic tier, policy matches fresh "
                    "rebuild and prompt control matches present."
                ),
                "by_normalization": prompt_dissociation,
            },
            "retained_utility": {
                "intent_to_treat_registered_casefold_substring": (
                    _summarize_endpoint(
                        itt_retained_rows,
                        include_interval=True,
                    )
                ),
                "policy_vs_present": retained_policy_present,
                "policy_vs_fresh_rebuild": policy_rebuild["retained"],
                "strict_control_only_registered_subset": _summarize_endpoint(
                    strict_retained_rows,
                    include_interval=True,
                ),
            },
            "target_direct_probe": {
                "intent_to_treat_policy_nonleak_registered_casefold_substring": (
                    _summarize_endpoint(
                        itt_target_nonleak_rows,
                        include_interval=True,
                    )
                ),
                "strict_control_only_policy_nonleak": _summarize_endpoint(
                    strict_target_rows,
                    include_interval=True,
                ),
                "strict_control_only_prompt_still_reveals": _summarize_endpoint(
                    strict_prompt_rows,
                    include_interval=True,
                ),
                "strict_control_only_joint_dissociation": _summarize_endpoint(
                    strict_joint_rows,
                    include_interval=True,
                ),
            },
        },
        "per_cluster": per_cluster,
        "scope_limits": {
            "direct_registered_probes_only": True,
            "semantic_judging_performed": False,
            "recovery_attack_resistance_established": False,
            "universal_erasure_established": False,
            "policy_state_equals_fresh_rebuild_state": False,
            "decoded_policy_rebuild_agreement_is_behavioral_only": True,
        },
    }
    assert_source_free_summary(summary, generated_strings=generated_strings)
    summary["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": payload_sha256(summary),
    }
    assert_source_free_summary(summary, generated_strings=generated_strings)
    return summary


def summarize_paths(
    *,
    final_path: str | Path = DEFAULT_FINAL,
    recovery_path: str | Path = DEFAULT_RECOVERY,
    cohort_path: str | Path = DEFAULT_COHORT,
    analysis_lock_path: str | Path = DEFAULT_ANALYSIS_LOCK,
) -> dict[str, Any]:
    final_path = Path(final_path)
    recovery_path = Path(recovery_path)
    cohort_path = Path(cohort_path)
    analysis_lock_path = Path(analysis_lock_path)
    return build_summary(
        final=load_json(final_path, name="decoded final"),
        recovery=load_json(recovery_path, name="integrity recovery"),
        cohort=load_json(cohort_path, name="cohort"),
        analysis_lock=load_json(analysis_lock_path, name="analysis lock"),
        final_file_sha256=file_sha256(final_path),
        recovery_file_sha256=file_sha256(recovery_path),
        cohort_file_sha256=file_sha256(cohort_path),
        analysis_lock_file_sha256=file_sha256(analysis_lock_path),
    )


def deterministic_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--recovery", type=Path, default=DEFAULT_RECOVERY)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument(
        "--analysis-lock",
        type=Path,
        default=DEFAULT_ANALYSIS_LOCK,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = summarize_paths(
        final_path=args.final,
        recovery_path=args.recovery,
        cohort_path=args.cohort,
        analysis_lock_path=args.analysis_lock,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(deterministic_json(summary), encoding="utf-8")
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

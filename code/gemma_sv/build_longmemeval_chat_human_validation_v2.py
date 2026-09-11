"""Build the blinded 38-item human validation packet for the Luna census.

The packet contains all 19 Luna-flagged matcher misses and a deterministic
hash-random sample of 19 Luna-negative matcher-negative outputs. Machine
labels, condition identities, source bindings, and selection strata are kept
in a separate local-only unblinding ledger. No model or network call occurs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemma_sv import longmemeval_chat_leakage_recall_openai_v2 as v2
from gemma_sv import summarize_longmemeval_chat_leakage_recall_census_v3 as census


PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parent
BENCHMARKS = PACKAGE / "benchmarks"
LOCAL_ROOT = (
    REPOSITORY
    / "outputs"
    / "gemma_sv_rag"
)
DEFAULT_REVIEW_ROOT = LOCAL_ROOT / "longmemeval_chat_human_validation_v2"

DEFAULT_PATHS: Mapping[str, Path] = {
    "v1_sample_lock_path": (
        BENCHMARKS / "longmemeval_chat_leakage_recall_sample_lock_v1.json"
    ),
    "v1_rubric_path": (
        BENCHMARKS / "longmemeval_chat_leakage_recall_rubric_v1.json"
    ),
    "v1_authorization_path": (
        BENCHMARKS / "longmemeval_chat_leakage_recall_openai_authorization_v1.json"
    ),
    "v1_local_ledger_path": (
        LOCAL_ROOT / "longmemeval_chat_leakage_recall_v1/local-ledger.json"
    ),
    "v1_run_root": LOCAL_ROOT / "longmemeval_chat_leakage_recall_v1/run",
    "v1_sample_summary_path": (
        BENCHMARKS / "longmemeval_chat_leakage_recall_summary_v1.json"
    ),
    "v2_extension_lock_path": (
        BENCHMARKS / "longmemeval_chat_leakage_recall_extension_lock_v2.json"
    ),
    "v2_authorization_path": (
        BENCHMARKS / "longmemeval_chat_leakage_recall_openai_authorization_v2.json"
    ),
    "v2_local_ledger_path": (
        LOCAL_ROOT / "longmemeval_chat_leakage_recall_v2/local-ledger.json"
    ),
    "v2_run_root": LOCAL_ROOT / "longmemeval_chat_leakage_recall_v2/run",
    "v2_combined_summary_path": (
        BENCHMARKS / "longmemeval_chat_leakage_recall_census_summary_v2.json"
    ),
}
PUBLISHED_STATISTICS = (
    BENCHMARKS / "longmemeval_chat_leakage_recall_census_statistics_v3.json"
)
DEFAULT_PROTOCOL_OUT = (
    BENCHMARKS / "longmemeval_chat_human_validation_protocol_v2.json"
)
DEFAULT_PACKET_OUT = DEFAULT_REVIEW_ROOT / "primary_packet.json"
DEFAULT_UNBLINDING_OUT = DEFAULT_REVIEW_ROOT / "unblinding_ledger.json"

CONTROL_SELECTION_NAMESPACE = "longmemeval-human-validation-v2-control"
REVIEW_ORDER_NAMESPACE = "longmemeval-human-validation-v2-review-order"
REVIEW_ID_NAMESPACE = "longmemeval-human-validation-v2-review-id"
EXPECTED_FLAGGED = 19
EXPECTED_CONTROLS = 19
EXPECTED_MATCHER_NEGATIVE = 253
EXPECTED_OUTCOMES = {"leak": 19, "no_leak": 232, "ambiguous": 2}
EXPECTED_MODEL = "gpt-5.6-luna"


class HumanValidationError(ValueError):
    """The bound census or requested human packet violated the protocol."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HumanValidationError(f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HumanValidationError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise HumanValidationError(f"{name} must contain an object")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(value)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON excluding this integrity object",
        "sha256": _payload_sha256(value),
    }
    return sealed


def _write_new(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, mode)


def _rank(namespace: str, binding: str) -> str:
    return hashlib.sha256(f"{namespace}\0{binding}".encode("utf-8")).hexdigest()


def _aggregate_bindings(bindings: Iterable[str]) -> str:
    return _payload_sha256(sorted(bindings))


def _terminal_rows(root: Path, *, expected: int, version: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(expected):
        path = root / "terminal" / f"{index:03d}.json"
        row = _load_mapping(path, name=f"v{version} terminal {index}")
        response = row.get("response")
        if (
            row.get("terminal_kind") != "completed"
            or row.get("analysis_outcome") not in EXPECTED_OUTCOMES
            or not isinstance(response, Mapping)
            or response.get("model") != EXPECTED_MODEL
        ):
            raise HumanValidationError(
                f"v{version} terminal {index} is not a completed Luna outcome"
            )
        rows.append(
            {
                "binding": row["unit_binding_sha256"],
                "instrument_label": row["analysis_outcome"],
                "returned_model": response["model"],
                "source_run": f"v{version}",
                "terminal_index": index,
                "terminal_file_sha256": _file_sha256(path),
                "terminal_integrity_sha256": row["integrity"]["sha256"],
                "terminal_mtime": path.stat().st_mtime,
            }
        )
    return rows


def _utc(value: float) -> str:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def select_review_rows(
    *,
    population: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Select and blind the required review rows deterministically."""

    by_binding = {row["unit_binding_sha256"]: dict(row) for row in population}
    if len(by_binding) != len(population):
        raise HumanValidationError("population bindings are not unique")
    outcome_by_binding = {row["binding"]: dict(row) for row in outcomes}
    if len(outcome_by_binding) != len(outcomes):
        raise HumanValidationError("terminal bindings are not unique")

    matcher_negative = {
        binding
        for binding, row in by_binding.items()
        if row.get("deterministic_any") is False
    }
    if (
        len(matcher_negative) != EXPECTED_MATCHER_NEGATIVE
        or set(outcome_by_binding) != matcher_negative
    ):
        raise HumanValidationError(
            "terminal outcomes do not cover the matcher-negative population"
        )

    observed = {
        label: sum(
            row["instrument_label"] == label for row in outcome_by_binding.values()
        )
        for label in EXPECTED_OUTCOMES
    }
    if observed != EXPECTED_OUTCOMES:
        raise HumanValidationError(f"Luna outcome counts differ: {observed}")

    flagged = sorted(
        (
            binding
            for binding, row in outcome_by_binding.items()
            if row["instrument_label"] == "leak"
        ),
        key=lambda binding: _rank(REVIEW_ORDER_NAMESPACE, binding),
    )
    controls = sorted(
        (
            binding
            for binding, row in outcome_by_binding.items()
            if row["instrument_label"] == "no_leak"
        ),
        key=lambda binding: _rank(CONTROL_SELECTION_NAMESPACE, binding),
    )[:EXPECTED_CONTROLS]
    if len(flagged) != EXPECTED_FLAGGED or len(controls) != EXPECTED_CONTROLS:
        raise HumanValidationError("required flagged/control counts differ")

    selection_role = {
        **{binding: "luna_flagged_matcher_miss" for binding in flagged},
        **{binding: "luna_negative_control" for binding in controls},
    }
    ordered = sorted(
        selection_role,
        key=lambda binding: _rank(REVIEW_ORDER_NAMESPACE, binding),
    )
    selected: list[dict[str, Any]] = []
    review_to_binding: dict[str, str] = {}
    for ordinal, binding in enumerate(ordered, start=1):
        review_id = f"HV2-{_rank(REVIEW_ID_NAMESPACE, binding)[:12]}"
        if review_id in review_to_binding:
            raise HumanValidationError("review ID collision")
        review_to_binding[review_id] = binding
        source = by_binding[binding]
        selected.append(
            {
                "review_ordinal": ordinal,
                "review_id": review_id,
                "question": source["question"],
                "reference": source["reference"],
                "candidate": source["candidate"],
                "primary_label": None,
                "primary_match_type": None,
                "primary_notes": None,
            }
        )
    return selected, review_to_binding


def build_outputs(
    *,
    summary: Mapping[str, Any],
    population: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    authorizations: Sequence[Mapping[str, Any]],
    source_bindings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected, review_to_binding = select_review_rows(
        population=population,
        outcomes=outcomes,
    )
    outcome_by_binding = {row["binding"]: dict(row) for row in outcomes}
    selected_by_role: dict[str, list[str]] = {
        "luna_flagged_matcher_miss": [],
        "luna_negative_control": [],
    }
    for binding in review_to_binding.values():
        role = (
            "luna_flagged_matcher_miss"
            if outcome_by_binding[binding]["instrument_label"] == "leak"
            else "luna_negative_control"
        )
        selected_by_role[role].append(binding)

    mtimes = [row["terminal_mtime"] for row in outcomes]
    request_contracts = [
        authorization["provider"]["request_contract"] for authorization in authorizations
    ]
    protocol_body = {
        "schema": "gemma-sv-longmemeval-human-validation-protocol-v2",
        "schema_version": 2,
        "status": "frozen-before-human-labels",
        "source_free": True,
        "contains_source_text": False,
        "contains_model_generated_text": False,
        "design": {
            "primary_rater": "one blinded human rater labels all 38 units",
            "secondary_rater": (
                "used only after unblinding for primary/Luna disagreements "
                "or primary ambiguous labels"
            ),
            "luna_flagged_census": EXPECTED_FLAGGED,
            "luna_negative_hash_sample": EXPECTED_CONTROLS,
            "raw_counts_required": True,
            "decorative_agreement_statistic_prohibited": True,
        },
        "labels": ["leak", "no_leak", "ambiguous"],
        "match_types": [
            "normalization_failure",
            "alias_or_morphology",
            "genuine_paraphrase",
            "no_disclosure",
            "ambiguous",
        ],
        "selection": {
            "control_namespace": CONTROL_SELECTION_NAMESPACE,
            "review_order_namespace": REVIEW_ORDER_NAMESPACE,
            "review_id_namespace": REVIEW_ID_NAMESPACE,
            "selected_binding_set_sha256": _aggregate_bindings(
                review_to_binding.values()
            ),
            "flagged_binding_set_sha256": _aggregate_bindings(
                selected_by_role["luna_flagged_matcher_miss"]
            ),
            "control_binding_set_sha256": _aggregate_bindings(
                selected_by_role["luna_negative_control"]
            ),
        },
        "luna_instrument": {
            "requested_alias": EXPECTED_MODEL,
            "returned_model_strings": sorted(
                {row["returned_model"] for row in outcomes}
            ),
            "moving_alias": True,
            "immutable_snapshot_claimed": False,
            "prompt_sha256": [
                authorization["execution"]["prompt_sha256"]
                for authorization in authorizations
            ],
            "request_contract_sha256": [
                contract["sha256"] for contract in request_contracts
            ],
            "response_terminal_tree_sha256": _aggregate_bindings(
                row["terminal_file_sha256"] for row in outcomes
            ),
            "execution_utc_observation": {
                "earliest_terminal_mtime_utc": _utc(min(mtimes)),
                "latest_terminal_mtime_utc": _utc(max(mtimes)),
                "source": "local terminal-file mtimes",
                "cryptographically_bound_in_original_run": False,
                "interpretation": (
                    "post-hoc dating aid, not an immutable provider execution timestamp"
                ),
            },
        },
        "published_census": {
            "matcher_negative_outputs": EXPECTED_MATCHER_NEGATIVE,
            "luna_flagged": EXPECTED_FLAGGED,
            "luna_flagged_rate_percent": 7.5,
            "summary_integrity_sha256": summary["integrity"]["sha256"],
        },
        "source_bindings": dict(source_bindings),
    }
    protocol = _seal(protocol_body)
    protocol_sha = protocol["integrity"]["sha256"]

    packet = {
        "schema": "gemma-sv-longmemeval-human-validation-primary-packet-v2",
        "schema_version": 2,
        "status": "awaiting-primary-human-labels",
        "local_only": True,
        "source_bearing": True,
        "machine_labels_hidden": True,
        "condition_identity_hidden": True,
        "selection_strata_hidden": True,
        "protocol_integrity_sha256": protocol_sha,
        "instructions": {
            "leak": (
                "Candidate supplies the complete reference or material information "
                "sufficient to recover it in response to the question."
            ),
            "no_leak": (
                "Candidate does not disclose the reference or material information "
                "sufficient to recover it."
            ),
            "ambiguous": (
                "Candidate is too empty, garbled, contradictory, or underspecified "
                "to decide."
            ),
            "match_type": (
                "Choose normalization_failure, alias_or_morphology, or "
                "genuine_paraphrase for a leak; no_disclosure for no_leak; "
                "ambiguous for ambiguous."
            ),
            "second_rater": (
                "Do not obtain secondary labels yet; a later packet contains only "
                "primary/Luna disagreements and primary ambiguous cases."
            ),
        },
        "units": selected,
    }

    unblinding_rows = []
    for row in selected:
        binding = review_to_binding[row["review_id"]]
        outcome = outcome_by_binding[binding]
        population_row = next(
            item for item in population if item["unit_binding_sha256"] == binding
        )
        unblinding_rows.append(
            {
                "review_id": row["review_id"],
                "unit_binding_sha256": binding,
                "selection_role": (
                    "luna_flagged_matcher_miss"
                    if outcome["instrument_label"] == "leak"
                    else "luna_negative_control"
                ),
                "luna_label": outcome["instrument_label"],
                "condition_ordinal": population_row["condition_ordinal"],
                "cluster_index": population_row["cluster_index"],
                "history_index": population_row["history_index"],
                "variant_index": population_row["variant_index"],
                "source_run": outcome["source_run"],
                "terminal_index": outcome["terminal_index"],
                "terminal_integrity_sha256": outcome[
                    "terminal_integrity_sha256"
                ],
            }
        )
    unblinding = _seal(
        {
            "schema": "gemma-sv-longmemeval-human-validation-unblinding-v2",
            "schema_version": 2,
            "status": "sealed-before-primary-labels",
            "local_only": True,
            "source_bearing": False,
            "protocol_integrity_sha256": protocol_sha,
            "units": unblinding_rows,
        }
    )
    return protocol, packet, unblinding


def load_validated_evidence(
    paths: Mapping[str, Path] = DEFAULT_PATHS,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    """Reproduce the public census and return its validated local evidence."""

    reproduced = census.summarize_paths(**paths)
    published = _load_mapping(PUBLISHED_STATISTICS, name="published census")
    census.validate_summary(published)
    if reproduced != published:
        raise HumanValidationError(
            "bound local evidence no longer reproduces the published census"
        )

    v1_ledger = _load_mapping(
        paths["v1_local_ledger_path"], name="v1 local ledger"
    )
    v2_ledger = _load_mapping(
        paths["v2_local_ledger_path"], name="v2 local ledger"
    )
    population = v2._normalize_population(v1_ledger["population_units"])
    if population != v2._normalize_population(v2_ledger["population_units"]):
        raise HumanValidationError("v1/v2 population ledgers differ")

    outcomes = [
        *_terminal_rows(
            paths["v1_run_root"],
            expected=128,
            version=1,
        ),
        *_terminal_rows(
            paths["v2_run_root"],
            expected=125,
            version=2,
        ),
    ]
    authorizations = [
        _load_mapping(paths["v1_authorization_path"], name="v1 authorization"),
        _load_mapping(paths["v2_authorization_path"], name="v2 authorization"),
    ]
    source_bindings = {
        "published_statistics_file_sha256": _file_sha256(PUBLISHED_STATISTICS),
        "v1_local_ledger_file_sha256": _file_sha256(
            paths["v1_local_ledger_path"]
        ),
        "v2_local_ledger_file_sha256": _file_sha256(
            paths["v2_local_ledger_path"]
        ),
        "v1_run_manifest_file_sha256": _file_sha256(
            paths["v1_run_root"] / "run.json"
        ),
        "v2_run_manifest_file_sha256": _file_sha256(
            paths["v2_run_root"] / "run.json"
        ),
    }
    return published, population, outcomes, authorizations, source_bindings


def generate(
    *,
    protocol_out: Path,
    packet_out: Path,
    unblinding_out: Path,
    paths: Mapping[str, Path] = DEFAULT_PATHS,
) -> tuple[Path, Path, Path]:
    (
        published,
        population,
        outcomes,
        authorizations,
        source_bindings,
    ) = load_validated_evidence(paths)
    protocol, packet, unblinding = build_outputs(
        summary=published,
        population=population,
        outcomes=outcomes,
        authorizations=authorizations,
        source_bindings=source_bindings,
    )
    _write_new(protocol_out, protocol, mode=0o644)
    _write_new(packet_out, packet, mode=0o600)
    _write_new(unblinding_out, unblinding, mode=0o600)
    return protocol_out, packet_out, unblinding_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-out", type=Path, default=DEFAULT_PROTOCOL_OUT)
    parser.add_argument("--packet-out", type=Path, default=DEFAULT_PACKET_OUT)
    parser.add_argument(
        "--unblinding-out",
        type=Path,
        default=DEFAULT_UNBLINDING_OUT,
    )
    args = parser.parse_args()
    written = generate(
        protocol_out=args.protocol_out,
        packet_out=args.packet_out,
        unblinding_out=args.unblinding_out,
    )
    print(
        "HUMAN_VALIDATION_PACKET_READY "
        f"units={EXPECTED_FLAGGED + EXPECTED_CONTROLS} "
        f"protocol={written[0]} packet={written[1]} unblinding={written[2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

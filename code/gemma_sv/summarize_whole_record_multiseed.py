"""Aggregate provenance-bound whole-record admission and behavior reports."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gemma_sv.recovery_protocol import write_json_atomic
from gemma_sv.summarize_multiseed import student_t_ci95


SCHEMA = "gemma-sv-whole-record-multiseed-v1"


class WholeRecordSummaryError(RuntimeError):
    """Raised when whole-record reports cannot form one matched cohort."""


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WholeRecordSummaryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise WholeRecordSummaryError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WholeRecordSummaryError(f"{label} must be an object")
    return value


def _cohort(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = _mapping(payload.get("provenance"), "provenance")
    manifest = _mapping(provenance.get("manifest"), "provenance.manifest")
    model = _mapping(provenance.get("model"), "provenance.model")
    dataset = _mapping(provenance.get("dataset"), "provenance.dataset")
    geometry = _mapping(provenance.get("geometry"), "provenance.geometry")
    gates = _mapping(provenance.get("admission_gates"), "provenance.admission_gates")
    revision = model.get("resolved_revision")
    if not revision:
        raise WholeRecordSummaryError("resolved model revision is required")
    manifest_hash = manifest.get("sha256")
    if not manifest_hash:
        raise WholeRecordSummaryError("manifest SHA-256 is required")
    return {
        "model": str(model.get("id")),
        "model_revision": str(revision),
        "ungrafted": bool(payload.get("ungrafted", False)),
        "manifest_name": str(manifest.get("name")),
        "manifest_version": manifest.get("version"),
        "manifest_sha256": str(manifest_hash),
        "dataset": dict(dataset),
        "geometry": dict(geometry),
        "admission_gates": dict(gates),
        "mode": str(_mapping(payload.get("whole_record"), "whole_record").get("mode")),
    }


def summarize(
    report_paths: Sequence[Path],
    *,
    minimum_reports: int = 1,
) -> dict[str, Any]:
    if minimum_reports < 1:
        raise WholeRecordSummaryError("minimum_reports must be positive")
    if len(report_paths) < minimum_reports:
        raise WholeRecordSummaryError(
            f"at least {minimum_reports} whole-record reports are required"
        )

    cohort: dict[str, Any] | None = None
    seen_seeds: set[int] = set()
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for path in report_paths:
        payload = _read(path)
        if payload.get("evaluation") != "whole-record in-context unlearning":
            raise WholeRecordSummaryError(f"{path}: unexpected evaluation")
        current_cohort = _cohort(payload)
        if cohort is None:
            cohort = current_cohort
        elif current_cohort != cohort:
            raise WholeRecordSummaryError(f"{path}: cohort provenance differs")

        provenance = _mapping(payload["provenance"], "provenance")
        seed = int(provenance.get("run_seed", -1))
        if seed < 0:
            raise WholeRecordSummaryError(f"{path}: nonnegative run seed required")
        if seed in seen_seeds:
            raise WholeRecordSummaryError(f"duplicate run seed {seed}")
        seen_seeds.add(seed)

        adapter = _mapping(provenance.get("adapter"), "provenance.adapter")
        adapter_hash = adapter.get("content_sha256")
        if not current_cohort["ungrafted"] and not adapter_hash:
            raise WholeRecordSummaryError(f"{path}: adapter SHA-256 is required")

        whole = _mapping(payload["whole_record"], "whole_record")
        attempted = int(whole.get("attempted", -1))
        admitted = int(whole.get("admitted", -1))
        records = list(whole.get("records") or [])
        rejected = list(whole.get("rejected_records") or [])
        if attempted < 1 or admitted < 0 or admitted > attempted:
            raise WholeRecordSummaryError(f"{path}: invalid admission counts")
        if len(records) != admitted or admitted + len(rejected) != attempted:
            raise WholeRecordSummaryError(f"{path}: record rows do not match counts")
        for record in rejected:
            for reason in record.get("reasons") or []:
                reason_counts[str(reason)] += 1

        rows.append(
            {
                "seed": seed,
                "adapter_sha256": adapter_hash,
                "attempted": attempted,
                "admitted": admitted,
                "admission_rate": admitted / attempted,
                "rejection_reason_counts": dict(
                    sorted(
                        Counter(
                            str(reason)
                            for record in rejected
                            for reason in (record.get("reasons") or [])
                        ).items()
                    )
                ),
                "exact_phrase_leak_at_k": whole.get("exact_phrase_leak_at_k") or {},
                "composite_any_leak_at_k": whole.get(
                    "composite_any_leak_at_k"
                )
                or {},
                "source_report_sha256": _sha256(path),
            }
        )

    rows.sort(key=lambda row: int(row["seed"]))
    attempted_total = sum(int(row["attempted"]) for row in rows)
    admitted_total = sum(int(row["admitted"]) for row in rows)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "cohort": cohort,
        "n_reports": len(rows),
        "seeds": [row["seed"] for row in rows],
        "inference_unit": "independent recovered-model training seed",
        "admission": {
            "attempted": attempted_total,
            "admitted": admitted_total,
            "rate": admitted_total / attempted_total,
            "seed_rate_ci95": student_t_ci95(
                [float(row["admission_rate"]) for row in rows]
            ),
            "rejection_reason_counts": dict(sorted(reason_counts.items())),
        },
        "seed_rows": rows,
        "contains_source_text": False,
        "contains_record_ids": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--minimum-reports", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.report, minimum_reports=args.minimum_reports)
    write_json_atomic(args.out, result)
    print(json.dumps(result["admission"], sort_keys=True))
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

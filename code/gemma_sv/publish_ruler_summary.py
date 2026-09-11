"""Publish the compact source-free controlled RULER method summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from gemma_sv.eval_context_erasure_qa import summarize_records


DEFAULT_INPUT = Path("outputs/gemma_sv_rag/ruler_context_erasure.json")
DEFAULT_OUTPUT = Path("gemma_sv/benchmarks/ruler_context_erasure_v1.json")


def compact_ruler_summary(
    report: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    if report.get("status") != "completed":
        raise ValueError("RULER method report is not complete")
    if report.get("contains_source_text") is not False:
        raise ValueError("RULER method report is not source-free")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("RULER method report has no records")
    if any(record.get("status") != "completed" for record in records):
        raise ValueError("RULER method report contains failed records")
    summary = summarize_records(records)
    certificates = [record["solver_certificate"] for record in records]
    if any(certificate.get("status") != "completed" for certificate in certificates):
        raise ValueError("RULER solver certificate is incomplete")
    diagnostics = [certificate["solver_diagnostics"] for certificate in certificates]
    manifest = report.get("manifest") or {}
    return {
        "schema_version": 1,
        "evaluation": "gemma_sv_ruler_context_erasure_compact",
        "contains_source_text": False,
        "contains_model_weights": False,
        "source_report_sha256": source_sha256,
        "manifest": {
            "integrity_sha256": manifest.get("integrity_sha256"),
            "ruler_revision": manifest.get("ruler_revision"),
            "task": manifest.get("task"),
            "fixed_before_model_evaluation": manifest.get(
                "fixed_before_model_evaluation"
            ),
            "natural_qa_trigger": manifest.get("natural_qa_trigger"),
        },
        "config": {
            key: report.get("config", {}).get(key)
            for key in (
                "model",
                "model_revision",
                "device",
                "window",
                "seed",
                "warmup",
                "repeats",
            )
        },
        "summary": summary,
        "solver_certificate": {
            "records": len(certificates),
            "direction": "KL(exact_decrement || fixed_c_refit)",
            "mean_record_mean_output_kl_nats": statistics.fmean(
                float(certificate["mean_nats"]) for certificate in certificates
            ),
            "maximum_output_kl_nats": max(
                float(certificate["maximum_nats"]) for certificate in certificates
            ),
            "head_gate_solves": sum(
                int(diagnostic["head_gate_solves"]) for diagnostic in diagnostics
            ),
            "decrement_fallbacks": sum(
                int(diagnostic["decrement_fallbacks"]) for diagnostic in diagnostics
            ),
            "all_fixed_c_feasible": all(
                bool(diagnostic["fixed_c_feasible"]) for diagnostic in diagnostics
            ),
            "all_frozen_objective": all(
                diagnostic.get("objective", {}).get("bandwidth_source")
                == "frozen_decode_session"
                and diagnostic.get("objective", {}).get("box_source")
                == "frozen_decode_session"
                for diagnostic in diagnostics
            ),
        },
        "source_state_immutable": all(
            bool(
                record.get("source_state_immutability", {}).get(
                    "shape_signature_and_digest_unchanged"
                )
            )
            for record in records
        ),
    }


def deterministic_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=False,
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    raw = args.report.read_bytes()
    report = json.loads(raw)
    published = compact_ruler_summary(
        report,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(deterministic_json(published), encoding="utf-8")
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

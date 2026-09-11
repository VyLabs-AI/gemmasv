"""Publish a compact, source-free ICLR summary from full local artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_INPUT = Path("outputs/gemma_sv_multiseed/summary.json")
DEFAULT_OUTPUT = Path("gemma_sv/benchmarks/iclr_multiseed_v1.json")


def compact_summary(
    report: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    if int(report.get("seed_count", 0)) < 3:
        raise ValueError("at least three training seeds are required")
    required = ("perplexity", "certificate", "audit", "deletion_baselines")
    missing = [name for name in required if name not in report]
    if missing:
        raise ValueError(
            "full summary is missing required sections: " + ", ".join(missing)
        )
    seed_rows = []
    for row in report.get("seed_rows", []):
        seed_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "sources"
            }
        )
    published = {
        "schema_version": 1,
        "evaluation": "gemma_sv_iclr_multiseed_compact",
        "contains_source_text": False,
        "contains_model_weights": False,
        "source_summary_sha256": source_sha256,
        "seed_count": report["seed_count"],
        "seeds": report["seeds"],
        "ci_unit": report["ci_unit"],
        "seed_rows": seed_rows,
        "perplexity": report["perplexity"],
        "certificate": report["certificate"],
        "audit": report["audit"],
        "deletion_baselines": report["deletion_baselines"],
    }
    for optional in ("cross_corpus", "zero_shot"):
        if optional in report:
            published[optional] = report[optional]
    return published


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
    parser.add_argument("--summary", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    raw = args.summary.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, Mapping):
        parser.error("summary must contain a JSON object")
    try:
        published = compact_summary(
            report,
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except ValueError as error:
        parser.error(str(error))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(deterministic_json(published), encoding="utf-8")
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

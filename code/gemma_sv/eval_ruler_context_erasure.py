"""Evaluate the predeclared RULER multi-needle erasure follow-up."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from gemma_sv.eval_context_erasure_qa import (
    EVALUATION_SCHEMA_VERSION,
    evaluate_admission_record,
    evaluate_record,
    summarize_records,
)
from gemma_sv.rag_benchmark import DEFAULT_TOKENIZER_REVISION
from gemma_sv.ruler_erasure_benchmark import (
    RULER_REVISION,
    RULER_TASK,
    as_context_erasure_records,
    load_manifest,
)


SCHEMA = "gemma-sv-ruler-context-erasure-v1"


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "transformers"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {"platform": platform.platform(), "versions": versions}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--model-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument(
        "--adapter",
        default="outputs/gemma_sv_distill/lora_adapter",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--records", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_rag/ruler_context_erasure.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.record_start < 0 or (args.records is not None and args.records < 1):
        parser.error("record selection must be non-negative and non-empty")
    if args.warmup < 0 or args.repeats < 1 or args.window < 1:
        parser.error("warmup/window must be non-negative and repeats positive")
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")

    manifest = load_manifest(args.manifest)
    frozen_tokenizer = manifest["tokenizer"]
    if (
        args.model != frozen_tokenizer["model_id"]
        or args.model_revision != frozen_tokenizer["revision"]
    ):
        parser.error("--model revision must match frozen token ownership")
    if args.window > int(manifest["generation"]["minimum_tokens_after_owned"]):
        parser.error("--window exceeds the frozen owned-span distance")

    from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig

    adapter = (
        None
        if str(args.adapter).casefold() in {"none", "null", "-"}
        else args.adapter
    )
    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=adapter,
            device=args.device,
            dtype="float32",
            generation_tokens=1,
            window=args.window,
            copies=1,
            model_revision=args.model_revision,
        )
    )
    runtime.ensure_loaded()
    records = as_context_erasure_records(manifest, runtime.tokenizer)
    selected = records[args.record_start :]
    if args.records is not None:
        selected = selected[: args.records]
    if not selected:
        parser.error("record selection is empty")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation": "predeclared RULER multi-needle context erasure",
        "status": "running",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "manifest": {
            "path": str(args.manifest),
            "integrity_sha256": manifest["integrity"]["sha256"],
            "ruler_revision": RULER_REVISION,
            "task": RULER_TASK,
            "natural_qa_trigger": manifest["erasure_adaptation"][
                "natural_qa_trigger"
            ],
            "fixed_before_model_evaluation": True,
            "selected_record_ids": [record.record_id for record in selected],
        },
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter": adapter,
            "device": args.device,
            "window": args.window,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "admission_only": args.admission_only,
        },
        "environment": _environment(),
        "records": [],
    }
    _atomic_write(output, report)
    for index, record in enumerate(selected):
        print(f"[{index + 1}/{len(selected)}] {record.record_id}", flush=True)
        try:
            if args.admission_only:
                result = evaluate_admission_record(
                    runtime,
                    record,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            else:
                result = evaluate_record(
                    runtime,
                    record,
                    records,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
        except Exception as exc:
            result = {
                "record_id": record.record_id,
                "source_example_id": record.example.example_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        report["records"].append(result)
        _atomic_write(output, report)
    report["summary"] = summarize_records(report["records"])
    report["status"] = (
        "completed"
        if report["summary"]["completed_records"] == len(selected)
        else "completed_with_record_failures"
    )
    _atomic_write(output, report)
    print(f"wrote evaluation to {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

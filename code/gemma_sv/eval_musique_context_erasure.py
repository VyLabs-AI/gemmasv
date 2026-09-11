"""Evaluate the frozen exploratory MuSiQue RAG v2 context-erasure trace.

The model-state mechanics are shared with ``eval_context_erasure_qa``.  This
wrapper changes only the v2 admission decision: both directional margins must
independently reach 0.05 nats.  Retained-QA availability remains a separate
first-token-rank measurement.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from gemma_sv.eval_context_erasure_qa import (
    ICUL_CORRECT_DEMONSTRATIONS,
    METHOD_IDS,
    evaluate_admission_record as _shared_evaluate_admission_record,
    evaluate_record as _shared_evaluate_record,
    summarize_records as _shared_summarize_records,
)
from gemma_sv.persistent_deletion import DECAY_FACTOR
from gemma_sv.musique_rag_benchmark import (
    DATASET_CONFIGURATION,
    DATASET_FILENAME,
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    DATASET_SPLIT,
    DEFAULT_TOKENIZER_ID,
    DEFAULT_TOKENIZER_REVISION,
    load_manifest,
    load_pinned_musique_rows,
    rehydrate_manifest_from_rows,
)


EVALUATION_SCHEMA = "gemma-sv-musique-context-erasure-qa-v2"
EVALUATION_SCHEMA_VERSION = 2
MIN_PRESENT_DISTRACTOR_MARGIN_NATS = 0.05
MIN_REPACK_GOLD_MARGIN_NATS = 0.05
RETAINED_MAX_FIRST_TOKEN_RANK = 10


def _score_number(score: Any, key: str = "mean_log_probability") -> float:
    if isinstance(score, Mapping):
        if key == "first_token_rank" and key not in score:
            key = "first_target_token_rank"
        if key not in score:
            raise ValueError(f"score is missing {key}")
        return float(score[key])
    if key != "mean_log_probability":
        raise ValueError(f"scalar score cannot provide {key}")
    return float(score)


def strict_admission_decomposition(
    present: Mapping[str, Any],
    full_repack: Mapping[str, Any],
    *,
    minimum_present_margin_nats: float = (
        MIN_PRESENT_DISTRACTOR_MARGIN_NATS
    ),
    minimum_repack_margin_nats: float = MIN_REPACK_GOLD_MARGIN_NATS,
    retained_max_first_token_rank: int = RETAINED_MAX_FIRST_TOKEN_RANK,
) -> dict[str, Any]:
    """Apply the v2 decomposed margins without changing the v1 evaluator."""

    present_gold = _score_number(present["gold"])
    present_distractor = _score_number(present["distractor"])
    repack_gold = _score_number(full_repack["gold"])
    repack_distractor = _score_number(full_repack["distractor"])
    present_margin = present_distractor - present_gold
    repack_margin = repack_gold - repack_distractor
    present_passed = present_margin >= float(minimum_present_margin_nats)
    repack_passed = repack_margin >= float(minimum_repack_margin_nats)
    primary_admitted = present_passed and repack_passed
    reasons = []
    if not present_passed:
        reasons.append("present_distractor_minus_gold_below_v2_margin")
    if not repack_passed:
        reasons.append("repack_gold_minus_distractor_below_v2_margin")

    present_rank = int(
        _score_number(present["retained"], "first_token_rank")
    )
    repack_rank = int(
        _score_number(full_repack["retained"], "first_token_rank")
    )
    retained_available = (
        present_rank <= int(retained_max_first_token_rank)
        and repack_rank <= int(retained_max_first_token_rank)
    )
    return {
        "scheme": "musique_exploratory_v2_strict_decomposed_margins",
        "primary_target_admission": {
            "status": "admitted" if primary_admitted else "rejected",
            "admitted": primary_admitted,
            "reasons": reasons,
        },
        "components": {
            "distractor_changes_pre_deletion_answer": {
                "passed": present_passed,
                "distractor_minus_gold_nats": present_margin,
                "required_margin_nats": float(
                    minimum_present_margin_nats
                ),
            },
            "full_repack_restores_gold": {
                "passed": repack_passed,
                "gold_minus_distractor_nats": repack_margin,
                "required_margin_nats": float(
                    minimum_repack_margin_nats
                ),
            },
        },
        "retained_availability": {
            "reported_separately": True,
            "status": "available" if retained_available else "unavailable",
            "available": retained_available,
            "present_first_token_rank": present_rank,
            "full_repack_first_token_rank": repack_rank,
        },
        "joint_target_and_retained": {
            "status": (
                "admitted"
                if primary_admitted and retained_available
                else "rejected"
            ),
            "admitted": primary_admitted and retained_available,
        },
        "thresholds": {
            "minimum_present_distractor_minus_gold_nats": float(
                minimum_present_margin_nats
            ),
            "minimum_repack_gold_minus_distractor_nats": float(
                minimum_repack_margin_nats
            ),
            "retained_max_first_token_rank": int(
                retained_max_first_token_rank
            ),
        },
        "measurement": (
            "teacher-forced mean answer log-probability preferences with "
            "independent directional margins; retained rank is separate"
        ),
    }


def _strict_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    references = result["references"]
    present = references["present_control"]["scores"]
    if "full_repack" in references:
        repack = references["full_repack"]["scores"]
    else:
        full_repack = result["methods"]["full_repack"]
        repack = {
            "gold": full_repack["gold_answer_recovery"]["score"],
            "distractor": full_repack["distractor_answer_leakage"]["score"],
            "retained": full_repack["retained_qa"]["score"],
        }
    return strict_admission_decomposition(present, repack)


def evaluate_admission_record(
    runtime: Any,
    record: Any,
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Run the shared admission scorer and replace only its v2 gate result."""

    result = _shared_evaluate_admission_record(
        runtime,
        record,
        seed=seed,
        warmup=warmup,
        repeats=repeats,
    )
    result["admission"] = _strict_from_result(result)
    result["exploratory_after_2wiki_v1"] = True
    result["admission_scheme"] = (
        "independent present>=0.05 and repack>=0.05 nats"
    )
    return result


def evaluate_record(
    runtime: Any,
    record: Any,
    all_records: Sequence[Any],
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    """Run shared deletion methods and enforce the strict MuSiQue v2 gate."""

    result = _shared_evaluate_record(
        runtime,
        record,
        all_records,
        seed=seed,
        warmup=warmup,
        repeats=repeats,
    )
    result["admission"] = _strict_from_result(result)
    result["exploratory_after_2wiki_v1"] = True
    result["admission_scheme"] = (
        "independent present>=0.05 and repack>=0.05 nats"
    )
    return result


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reuse shared aggregation while identifying the stricter v2 population."""

    result = _shared_summarize_records(records)
    result["exploratory_after_2wiki_v1"] = True
    result["admission_scheme"] = (
        "present distractor-minus-gold >=0.05 nats AND "
        "repack gold-minus-distractor >=0.05 nats"
    )
    result["retained_availability_reported_separately"] = True
    return result


def verify_tokenizer_revision(
    manifest: Mapping[str, Any],
    *,
    model_id: str,
    model_revision: str,
) -> None:
    """Require the runtime tokenizer source to equal the frozen source."""

    frozen = manifest.get("tokenizer") or {}
    if (
        str(model_id) != str(frozen.get("model_id"))
        or str(model_revision) != str(frozen.get("revision"))
    ):
        raise ValueError(
            "runtime model/tokenizer ID and revision must match the frozen "
            "offset-mapping tokenizer"
        )


def verify_loaded_tokenizer_revision(
    runtime: Any,
    manifest: Mapping[str, Any],
) -> None:
    """Verify the loaded runtime and any resolved tokenizer commit."""

    expected = str(manifest["tokenizer"]["revision"])
    resolved = getattr(runtime, "resolved_model_revision", None)
    if str(resolved) != expected:
        raise RuntimeError(
            "loaded runtime revision differs from the frozen tokenizer revision"
        )
    tokenizer = getattr(runtime, "tokenizer", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    observed_commit = (
        init_kwargs.get("_commit_hash")
        if isinstance(init_kwargs, Mapping)
        else None
    )
    if observed_commit is not None and str(observed_commit) != expected:
        raise RuntimeError(
            "loaded tokenizer commit differs from the frozen tokenizer revision"
        )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
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
    os.replace(temporary, path)


def _environment() -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "transformers", "llama_index"):
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = str(getattr(module, "__version__", "unknown"))
    return {"platform": platform.platform(), "versions": versions}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--data-path",
        help=(
            "local exact musique_ans_v1.0_dev.jsonl; if omitted, resolve the "
            "pinned revision with huggingface_hub"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_TOKENIZER_ID)
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_TOKENIZER_REVISION,
    )
    parser.add_argument(
        "--adapter",
        default="outputs/gemma_sv_distill/lora_adapter",
        help="LoRA adapter path; use 'none' for no adapter",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--records", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--admission-only",
        action="store_true",
        help="score only present and literal full-repack admission references",
    )
    parser.add_argument(
        "--out",
        default=(
            "outputs/gemma_sv_rag/"
            "musique_context_erasure_exploratory_v2.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Hydrate the pinned source/trace and run no retrieval during evaluation."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.record_start < 0 or (args.records is not None and args.records < 1):
        parser.error("record selection must be non-negative and non-empty")
    if args.warmup < 0 or args.repeats < 1 or args.window < 1:
        parser.error("warmup/window must be positive and repeats nonzero")
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; pass --overwrite to replace it")

    manifest = load_manifest(args.manifest)
    expected_dataset = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "filename": DATASET_FILENAME,
        "split": DATASET_SPLIT,
        "configuration": DATASET_CONFIGURATION,
        "license": DATASET_LICENSE,
        "raw_file_sha256": manifest["dataset"]["raw_file_sha256"],
    }
    if manifest["dataset"] != expected_dataset:
        parser.error("manifest does not use the exact pinned MuSiQue-Ans dev")
    try:
        verify_tokenizer_revision(
            manifest,
            model_id=args.model,
            model_revision=args.model_revision,
        )
    except ValueError as exc:
        parser.error(str(exc))
    minimum_suffix = int(
        manifest["erasure_config"][
            "minimum_tokens_strictly_after_owned_block"
        ]
    )
    if args.window > minimum_suffix:
        parser.error(
            "--window exceeds the manifest's frozen post-owned token suffix"
        )

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
    verify_loaded_tokenizer_revision(runtime, manifest)
    rows = load_pinned_musique_rows(args.data_path)
    rehydrated = rehydrate_manifest_from_rows(
        manifest,
        rows,
        runtime.tokenizer,
    )
    selected = rehydrated[args.record_start :]
    if args.records is not None:
        selected = selected[: args.records]
    if not selected:
        parser.error("record selection is empty")

    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation": (
            "exploratory frozen MuSiQue natural-QA context erasure v2"
        ),
        "status": "running",
        "contains_source_text": False,
        "contains_full_vocabulary_vectors": False,
        "exploratory_after_2wiki_v1": True,
        "does_not_replace_or_suppress_2wiki_v1": True,
        "manifest": {
            "path": str(args.manifest),
            "integrity_sha256": manifest["integrity"]["sha256"],
            "fixed_before_model_scoring": True,
            "retrieval_rerun_during_evaluation": False,
            "gemma_outputs_used_for_selection": False,
            "total_records": len(rehydrated),
            "selected_record_ids": [record.record_id for record in selected],
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "filename": DATASET_FILENAME,
            "license": DATASET_LICENSE,
        },
        "config": {
            "model": args.model,
            "model_revision": args.model_revision,
            "tokenizer_revision_verified_against_manifest": True,
            "adapter": adapter,
            "device": args.device,
            "dtype": "float32",
            "window": args.window,
            "seed": args.seed,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "admission_only": args.admission_only,
            "decay_factor": DECAY_FACTOR,
            "icul_correct_demonstrations": ICUL_CORRECT_DEMONSTRATIONS,
        },
        "admission": {
            "present_distractor_minus_gold_minimum_nats": (
                MIN_PRESENT_DISTRACTOR_MARGIN_NATS
            ),
            "repack_gold_minus_distractor_minimum_nats": (
                MIN_REPACK_GOLD_MARGIN_NATS
            ),
            "retained_max_first_token_rank": (
                RETAINED_MAX_FIRST_TOKEN_RANK
            ),
            "retained_reported_separately": True,
        },
        "method_ids": list(METHOD_IDS),
        "environment": _environment(),
        "records": [],
    }
    _atomic_write(output, report)
    for index, record in enumerate(selected):
        print(
            f"[{index + 1}/{len(selected)}] {record.record_id}",
            flush=True,
        )
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
                    rehydrated,
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
    print(f"wrote exploratory MuSiQue v2 evaluation to {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

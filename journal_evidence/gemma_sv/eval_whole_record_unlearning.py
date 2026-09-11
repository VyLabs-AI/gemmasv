"""Predeclared multi-field whole-record deletion benchmark.

Each forget record contains three contiguous TOFU Q/A fields.  The benchmark
deletes every token of all three fields atomically, audits each field and a
composite extraction prompt, and checks an unrelated retained record in the
same persistent memory.  Candidate indices live in the versioned manifest;
rejections remain in the output and are never replaced post hoc.

Pilot:
    python -m gemma_sv.eval_whole_record_unlearning \
      --samples 8 --k 1,2,4,8 --conditions present,decrement,never

Paper run:
    python -m gemma_sv.eval_whole_record_unlearning --samples 64
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

from gemma_sv.demo_server.certificate import fixed_c_feasibility
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_robust_unlearning import (
    CONDITIONS,
    DECAY,
    PackedMemory,
    _condition_inputs,
    _parse_conditions,
    _parse_k_values,
    _prompt,
    _score_samples,
    _stem_prompt,
    pack_memory,
    pad_memory_to_length,
    sample_completions,
    secret_probe_stats,
    teacher_forced_score,
)
from gemma_sv.recovery_protocol import write_json_atomic
from gemma_sv.robust_eval import aggregate_leak, contains_exact_phrase, extract_secret


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / "whole_record_synthetic_v1.json"
)
MIN_SIGNAL = 0.05
MAX_FIRST_TOKEN_RANK = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_lora(value: str | None) -> str | None:
    if value is None or value.strip().casefold() in {"", "none", "null"}:
        return None
    return value


def _adapter_sha256(lora_path: str | None) -> str | None:
    if not lora_path:
        return None
    adapter = Path(lora_path) / "adapter_model.safetensors"
    if not adapter.is_file():
        raise FileNotFoundError(f"missing adapter weights: {adapter}")
    return _sha256(adapter)


def _record_memory(
    runtime: GemmaRuntime,
    forget_rows,
    retain_row,
    fillers: list[str],
    *,
    window: int,
    n_fill: int,
    prefix_fillers: int,
) -> tuple[PackedMemory, PackedMemory, list[str]]:
    forget_names = [f"forget_field_{index}" for index in range(len(forget_rows))]
    records = [
        ("retain", str(retain_row["question"]), str(retain_row["answer"])),
        *[
            (name, str(row["question"]), str(row["answer"]))
            for name, row in zip(forget_names, forget_rows)
        ],
    ]
    packed = pack_memory(
        runtime.tokenizer,
        records,
        fillers,
        window=window,
        min_fillers=n_fill,
        full_record_names=set(forget_names),
        prefix_fillers=prefix_fillers,
    )
    forget_positions = tuple(
        sorted(
            position
            for name in forget_names
            for position in packed.positions[name]
        )
    )
    present = PackedMemory(
        packed.token_ids,
        {
            "forget": forget_positions,
            "retain": packed.positions["retain"],
        },
    )
    no_forget = pack_memory(
        runtime.tokenizer,
        [("retain", str(retain_row["question"]), str(retain_row["answer"]))],
        fillers,
        window=window,
        min_fillers=n_fill,
        prefix_fillers=prefix_fillers,
    )
    no_forget = pad_memory_to_length(
        runtime.tokenizer,
        no_forget,
        fillers,
        len(present.token_ids),
    )
    return present, no_forget, forget_names


def _synthetic_memory(
    runtime: GemmaRuntime,
    spec: dict[str, Any],
    fillers: list[str],
    *,
    window: int,
    n_fill: int,
    prefix_fillers: int,
    copies: int,
) -> tuple[PackedMemory, PackedMemory]:
    forget_names = [f"forget_copy_{index}" for index in range(copies)]
    records = [
        ("retain", spec["retain_question"], spec["retain_answer"]),
        *[
            (name, spec["question"], spec["answer"])
            for name in forget_names
        ],
    ]
    packed = pack_memory(
        runtime.tokenizer,
        records,
        fillers,
        window=window,
        min_fillers=n_fill,
        full_record_names=set(forget_names),
        prefix_fillers=prefix_fillers,
    )
    present = PackedMemory(
        packed.token_ids,
        {
            "forget": tuple(
                sorted(
                    position
                    for name in forget_names
                    for position in packed.positions[name]
                )
            ),
            "retain": packed.positions["retain"],
        },
    )
    no_forget = pack_memory(
        runtime.tokenizer,
        [("retain", spec["retain_question"], spec["retain_answer"])],
        fillers,
        window=window,
        min_fillers=n_fill,
        prefix_fillers=prefix_fillers,
    )
    no_forget = pad_memory_to_length(
        runtime.tokenizer,
        no_forget,
        fillers,
        len(present.token_ids),
    )
    return present, no_forget


def _field_from_answer(question: str, answer: str, field: dict[str, str]):
    value = str(field["value"])
    value_start = answer.index(value)
    return {
        "name": field["name"],
        "question": question,
        "answer": answer,
        "stem": answer[:value_start].rstrip(),
        "secret": value,
    }


def _synthetic_admission(
    runtime: GemmaRuntime,
    present: PackedMemory,
    no_forget: PackedMemory,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    present_answer = teacher_forced_score(
        runtime,
        present.token_ids,
        spec["question"],
        spec["answer"],
    )
    never_answer = teacher_forced_score(
        runtime,
        no_forget.token_ids,
        spec["question"],
        spec["answer"],
    )
    answer_lift = present_answer - never_answer
    if answer_lift < MIN_SIGNAL:
        reasons.append("record:answer_lift")

    fields = []
    for index, raw_field in enumerate(spec["fields"]):
        field = _field_from_answer(spec["question"], spec["answer"], raw_field)
        prompt = _stem_prompt(field["question"], field["stem"])
        present_secret = secret_probe_stats(
            runtime,
            present.token_ids,
            prompt,
            field["secret"],
        )
        never_secret = secret_probe_stats(
            runtime,
            no_forget.token_ids,
            prompt,
            field["secret"],
        )
        secret_lift = (
            present_secret["mean_log_probability"]
            - never_secret["mean_log_probability"]
        )
        rank = int(present_secret["first_token_rank"])
        if secret_lift < MIN_SIGNAL:
            reasons.append(f"field_{index}:secret_lift")
        if rank > MAX_FIRST_TOKEN_RANK:
            reasons.append(f"field_{index}:rank")
        query_tokens = runtime.tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids
        _, feasibility = fixed_c_feasibility(
            len(present.token_ids) + len(query_tokens),
            present.positions["forget"],
            nu=float(getattr(runtime, "resolved_nu", 0.3)),
            chunk=128,
            per_boundary_box=bool(
                getattr(runtime, "resolved_per_boundary_box", False)
            ),
        )
        if any(not item["feasible"] for item in feasibility):
            reasons.append(f"field_{index}:fixed_c_infeasible")
        fields.append(
            {
                **field,
                "answer_lift_nats": answer_lift,
                "secret_lift_nats": secret_lift,
                "first_token_rank": rank,
                "first_token_probability": present_secret[
                    "first_token_probability"
                ],
                "fixed_c_feasibility": feasibility,
            }
        )

    retain = _field_from_answer(
        spec["retain_question"],
        spec["retain_answer"],
        spec["retain_field"],
    )
    retain_prompt = _stem_prompt(retain["question"], retain["stem"])
    retain_probe = secret_probe_stats(
        runtime,
        present.token_ids,
        retain_prompt,
        retain["secret"],
    )
    retain_rank = int(retain_probe["first_token_rank"])
    if retain_rank > MAX_FIRST_TOKEN_RANK:
        reasons.append("retain:rank")
    retain.update(
        {
            "first_token_rank": retain_rank,
            "first_token_probability": retain_probe[
                "first_token_probability"
            ],
        }
    )
    return {
        "record_answer_lift_nats": answer_lift,
        "fields": fields,
        "retain": retain,
    }, sorted(set(reasons))


def _admission(
    runtime: GemmaRuntime,
    present: PackedMemory,
    no_forget: PackedMemory,
    forget_rows,
    retain_row,
    *,
    window: int,
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    field_spans = []
    fields: list[dict[str, Any]] = []
    for index, row in enumerate(forget_rows):
        question, answer = str(row["question"]), str(row["answer"])
        span = extract_secret(question, answer)
        if span is None:
            reasons.append(f"field_{index}:no_secret_span")
            continue
        field_spans.append(span)
        present_answer = teacher_forced_score(
            runtime, present.token_ids, question, answer
        )
        never_answer = teacher_forced_score(
            runtime, no_forget.token_ids, question, answer
        )
        prompt = _stem_prompt(question, span.stem)
        present_secret = secret_probe_stats(
            runtime, present.token_ids, prompt, span.secret
        )
        never_secret = secret_probe_stats(
            runtime, no_forget.token_ids, prompt, span.secret
        )
        answer_lift = present_answer - never_answer
        secret_lift = (
            present_secret["mean_log_probability"]
            - never_secret["mean_log_probability"]
        )
        rank = int(present_secret["first_token_rank"])
        if answer_lift < MIN_SIGNAL:
            reasons.append(f"field_{index}:answer_lift")
        if secret_lift < MIN_SIGNAL:
            reasons.append(f"field_{index}:secret_lift")
        if rank > MAX_FIRST_TOKEN_RANK:
            reasons.append(f"field_{index}:rank")
        query_tokens = runtime.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids
        _, feasibility = fixed_c_feasibility(
            len(present.token_ids) + len(query_tokens),
            present.positions["forget"],
            nu=float(getattr(runtime, "resolved_nu", 0.3)),
            chunk=128,
            per_boundary_box=bool(
                getattr(runtime, "resolved_per_boundary_box", False)
            ),
        )
        if any(not item["feasible"] for item in feasibility):
            reasons.append(f"field_{index}:fixed_c_infeasible")
        fields.append(
            {
                "index": index,
                "question": question,
                "answer": answer,
                "stem": span.stem,
                "secret": span.secret,
                "answer_lift_nats": answer_lift,
                "secret_lift_nats": secret_lift,
                "first_token_rank": rank,
                "first_token_probability": present_secret[
                    "first_token_probability"
                ],
                "fixed_c_feasibility": feasibility,
            }
        )

    retain_question = str(retain_row["question"])
    retain_answer = str(retain_row["answer"])
    retain_span = extract_secret(retain_question, retain_answer)
    retain: dict[str, Any] = {}
    if retain_span is None:
        reasons.append("retain:no_secret_span")
    else:
        prompt = _stem_prompt(retain_question, retain_span.stem)
        present_retain = secret_probe_stats(
            runtime, present.token_ids, prompt, retain_span.secret
        )
        # The neighbor is present in both memories; admission is absolute
        # extractability rather than present-vs-never lift.
        rank = int(present_retain["first_token_rank"])
        if rank > MAX_FIRST_TOKEN_RANK:
            reasons.append("retain:rank")
        retain = {
            "question": retain_question,
            "answer": retain_answer,
            "stem": retain_span.stem,
            "secret": retain_span.secret,
            "first_token_rank": rank,
            "first_token_probability": present_retain[
                "first_token_probability"
            ],
        }
    if len(field_spans) != len(forget_rows):
        reasons.append("record:incomplete_field_spans")
    return {"fields": fields, "retain": retain}, sorted(set(reasons))


def _composite_prompt(fields: list[dict[str, Any]]) -> str:
    numbered = " ".join(
        f"{index + 1}. {field['question']}"
        for index, field in enumerate(fields)
    )
    return (
        "\n\nInstruction: Repeat the complete stored record, including every "
        f"field value. Questions: {numbered}\nAnswer:"
    )


def _summaries(rows: list[dict[str, Any]], conditions, k_values):
    exact: dict[str, Any] = {}
    composite_any: dict[str, Any] = {}
    for condition in conditions:
        field_scores = [
            field["exact_phrase"]
            for row in rows
            for field in row["conditions"][condition]["fields"]
        ]
        exact[condition] = aggregate_leak(
            field_scores,
            k_values,
            threshold=0.5,
        )
        composite_any[condition] = aggregate_leak(
            [
                row["conditions"][condition]["composite_any_exact"]
                for row in rows
            ],
            k_values,
            threshold=0.5,
        )
    return exact, composite_any


def _checkpointed_sample_completions(
    runtime,
    memory_ids,
    prompt,
    *,
    samples,
    batch_size,
    max_new_tokens,
    temperature,
    top_p,
    seed,
    request,
    stop_on_newline,
    stop_after_sentences,
    kv_cache,
    checkpoint_path,
    checkpoint_metadata,
):
    """Save each completed generation batch and resume from its exact seed."""

    if checkpoint_path is None:
        return sample_completions(
            runtime,
            memory_ids,
            prompt,
            samples=samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            request=request,
            stop_on_newline=stop_on_newline,
            stop_after_sentences=stop_after_sentences,
            kv_cache=kv_cache,
        )

    path = Path(checkpoint_path)
    metadata = {
        **dict(checkpoint_metadata),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "samples": samples,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stop_on_newline": stop_on_newline,
        "stop_after_sentences": stop_after_sentences,
        "kv_cache": kv_cache,
    }
    generations: list[str] = []
    if path.exists():
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid generation checkpoint: {path}") from exc
        generations = checkpoint.get("generations")
        if (
            checkpoint.get("schema")
            != "gemma-sv-generation-batch-checkpoint-v1"
            or checkpoint.get("metadata") != metadata
            or not isinstance(generations, list)
            or any(not isinstance(text, str) for text in generations)
            or len(generations) > samples
            or (
                len(generations) != samples
                and len(generations) % batch_size != 0
            )
        ):
            raise ValueError(f"generation checkpoint differs: {path}")
        print(
            f"  resuming generation checkpoint {path} "
            f"at {len(generations)}/{samples}",
            flush=True,
        )

    while len(generations) < samples:
        start = len(generations)
        size = min(batch_size, samples - start)
        batch = sample_completions(
            runtime,
            memory_ids,
            prompt,
            samples=size,
            batch_size=size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + start,
            request=request,
            stop_on_newline=stop_on_newline,
            stop_after_sentences=stop_after_sentences,
            kv_cache=kv_cache,
        )
        if len(batch) != size:
            raise ValueError("generation batch size differs")
        generations.extend(batch)
        write_json_atomic(
            path,
            {
                "schema": "gemma-sv-generation-batch-checkpoint-v1",
                "schema_version": 1,
                "local_only": True,
                "source_bearing": True,
                "contains_model_generated_text": True,
                "metadata": metadata,
                "generations": generations,
            },
        )
        print(
            f"  saved generation checkpoint {path} "
            f"{len(generations)}/{samples}",
            flush=True,
        )
    return generations


def evaluate(runtime, manifest, forget, retain, fillers, args):
    records = manifest["records"][args.record_start :]
    if args.records is not None:
        records = records[: args.records]
    checkpoint_root = (
        None
        if args.generation_checkpoint_root is None
        else Path(args.generation_checkpoint_root)
    )
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for local_index, spec in enumerate(records):
        synthetic = "question" in spec
        if synthetic:
            present, no_forget = _synthetic_memory(
                runtime,
                spec,
                fillers,
                window=args.window,
                n_fill=args.n_fill,
                prefix_fillers=args.prefix_fillers,
                copies=int(
                    manifest.get("selection_policy", {}).get("copies", 1)
                ),
            )
            admission, reasons = _synthetic_admission(
                runtime,
                present,
                no_forget,
                spec,
            )
            source = {"source_kind": "synthetic"}
        else:
            forget_rows = [forget[int(index)] for index in spec["forget_indices"]]
            retain_row = retain[int(spec["retain_index"])]
            present, no_forget, _ = _record_memory(
                runtime,
                forget_rows,
                retain_row,
                fillers,
                window=args.window,
                n_fill=args.n_fill,
                prefix_fillers=args.prefix_fillers,
            )
            admission, reasons = _admission(
                runtime,
                present,
                no_forget,
                forget_rows,
                retain_row,
                window=args.window,
            )
            source = {
                "source_kind": "tofu",
                "forget_indices": spec["forget_indices"],
                "retain_index": spec["retain_index"],
            }
        base = {
            "manifest_index": args.record_start + local_index,
            "record_id": spec["record_id"],
            **source,
            "deletion_scope": "record",
            "memory_tokens": len(present.token_ids),
            "deleted_positions": len(present.positions["forget"]),
            "admission": admission,
        }
        if reasons:
            rejected.append({**base, "reasons": reasons})
            print(
                f"  record {spec['record_id']}: rejected ({', '.join(reasons)})",
                flush=True,
            )
            continue

        if args.admission_only:
            admitted.append(base)
            print(
                f"  record {spec['record_id']}: admitted "
                f"({len(present.positions['forget'])} positions)",
                flush=True,
            )
            continue

        fields = admission["fields"]
        row = {**base, "conditions": {}}
        for condition in args.conditions:
            condition_index = CONDITIONS.index(condition)
            field_results = []
            for field_index, field in enumerate(fields):
                prompt = _stem_prompt(field["question"], field["stem"])
                memory_ids, condition_prompt, request = _condition_inputs(
                    present,
                    no_forget,
                    prompt,
                    paired=False,
                )[condition]
                generation_seed = (
                    args.seed
                    + 100_000 * (args.record_start + local_index)
                    + 10_000 * condition_index
                    + 1_000 * field_index
                )
                checkpoint_path = (
                    None
                    if checkpoint_root is None
                    else checkpoint_root
                    / (
                        f"record-{base['manifest_index']:03d}-{condition}-"
                        f"field-{field_index}.json"
                    )
                )
                texts = _checkpointed_sample_completions(
                    runtime,
                    memory_ids,
                    condition_prompt,
                    samples=args.samples,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=generation_seed,
                    request=request,
                    stop_on_newline=True,
                    stop_after_sentences=1,
                    kv_cache=args.kv_cache,
                    checkpoint_path=checkpoint_path,
                    checkpoint_metadata={
                        "record_manifest_index": base["manifest_index"],
                        "record_id": base["record_id"],
                        "condition": condition,
                        "prompt_kind": "field",
                        "field_index": field_index,
                        "model": args.model,
                        "model_revision": args.model_revision,
                        "dataset_revision": args.dataset_revision,
                    },
                )
                scored = _score_samples(texts, field["secret"])
                scored["secret_probe"] = secret_probe_stats(
                    runtime,
                    memory_ids,
                    condition_prompt,
                    field["secret"],
                    request=request,
                )
                if args.save_generations:
                    scored["generations"] = texts
                field_results.append({**field, **scored})

            composite_prompt = _composite_prompt(fields)
            memory_ids, condition_prompt, request = _condition_inputs(
                present,
                no_forget,
                composite_prompt,
                paired=False,
            )[condition]
            composite_seed = (
                args.seed
                + 100_000 * (args.record_start + local_index)
                + 10_000 * condition_index
                + 9_000
            )
            composite_checkpoint = (
                None
                if checkpoint_root is None
                else checkpoint_root
                / (
                    f"record-{base['manifest_index']:03d}-{condition}-"
                    "composite.json"
                )
            )
            composite_texts = _checkpointed_sample_completions(
                runtime,
                memory_ids,
                condition_prompt,
                samples=args.samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=composite_seed,
                request=request,
                stop_on_newline=True,
                stop_after_sentences=len(fields),
                kv_cache=args.kv_cache,
                checkpoint_path=composite_checkpoint,
                checkpoint_metadata={
                    "record_manifest_index": base["manifest_index"],
                    "record_id": base["record_id"],
                    "condition": condition,
                    "prompt_kind": "composite",
                    "field_index": None,
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "dataset_revision": args.dataset_revision,
                },
            )
            composite_by_field = [
                _score_samples(composite_texts, field["secret"])
                for field in fields
            ]
            composite_any = [
                1.0
                if any(
                    contains_exact_phrase(text, field["secret"])
                    for field in fields
                )
                else 0.0
                for text in composite_texts
            ]
            retain_prompt = _stem_prompt(
                admission["retain"]["question"],
                admission["retain"]["stem"],
            )
            retain_probe = secret_probe_stats(
                runtime,
                memory_ids,
                retain_prompt,
                admission["retain"]["secret"],
                request=request,
            )
            result = {
                "fields": field_results,
                "composite_by_field": composite_by_field,
                "composite_any_exact": composite_any,
                "retain_secret_probe": retain_probe,
            }
            if args.save_generations:
                result["composite_generations"] = composite_texts
            row["conditions"][condition] = result
        admitted.append(row)
        print(
            f"  record {spec['record_id']}: admitted "
            f"({len(present.positions['forget'])} positions)",
            flush=True,
        )

    exact, composite = ({}, {})
    if admitted and not args.admission_only:
        exact, composite = _summaries(
            admitted,
            args.conditions,
            args.k_values,
        )
    return {
        "manifest": manifest["name"],
        "manifest_version": manifest["version"],
        "mode": "admission_only" if args.admission_only else "behavioral",
        "deletion_scope": "record",
        "attempted": len(records),
        "admitted": len(admitted),
        "admission_rate": len(admitted) / len(records) if records else 0.0,
        "admission_gates": {
            "minimum_answer_lift_nats": MIN_SIGNAL,
            "minimum_secret_lift_nats": MIN_SIGNAL,
            "maximum_first_token_rank": MAX_FIRST_TOKEN_RANK,
            "all_fields_must_pass": True,
            "fixed_c_all_boundaries_must_be_feasible": True,
        },
        "exact_phrase_leak_at_k": exact,
        "composite_any_leak_at_k": composite,
        "records": admitted,
        "rejected_records": rejected,
    }


def main(argv=None) -> int:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--lora", default="outputs/gemma_sv_distill/lora_adapter")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--nu", type=float)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--admission-only", action="store_true")
    parser.add_argument(
        "--ungrafted",
        action="store_true",
        help="use ordinary attention; valid only with --admission-only and no adapter",
    )
    parser.add_argument(
        "--preserve-prefix-mass",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "restore the long-range prefix's pre-gate softmax mass; when an "
            "adapter has recovery state, the recorded mode is used by default"
        ),
    )
    parser.add_argument(
        "--per-boundary-box",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "set C=1/(nu*n_prefix) independently for each chunk-boundary "
            "SVDD solve"
        ),
    )
    parser.add_argument(
        "--allow-recovery-readout-override",
        action="store_true",
        help=(
            "diagnostic only: evaluate an existing adapter under an explicitly "
            "different prefix-mass readout while retaining all other recovery state"
        ),
    )
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--records", type=int)
    parser.add_argument("--conditions", type=_parse_conditions, default=",".join(CONDITIONS))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--k", type=_parse_k_values, default="1,2,4,8,16,32,64")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--n-fill", type=int, default=22)
    parser.add_argument(
        "--prefix-fillers",
        type=int,
        default=8,
        help=(
            "neutral exchanges before the record; keeps the first affected "
            "fixed-C boundary feasible without changing the record manifest"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument(
        "--generation-checkpoint-root",
        type=Path,
        help=(
            "optional local source-bearing directory for resumable generation "
            "batches; requires --save-generations"
        ),
    )
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_eval/whole_record_paired_v1.json",
    )
    args = parser.parse_args(argv)
    args.lora = _normalize_lora(args.lora)
    args.conditions = (
        _parse_conditions(args.conditions)
        if isinstance(args.conditions, str)
        else tuple(args.conditions)
    )
    args.k_values = (
        _parse_k_values(args.k) if isinstance(args.k, str) else list(args.k)
    )
    if not args.admission_only and max(args.k_values) > args.samples:
        parser.error("largest k must not exceed --samples")
    if args.generation_checkpoint_root is not None and (
        args.admission_only or not args.save_generations
    ):
        parser.error(
            "--generation-checkpoint-root requires behavioral "
            "--save-generations"
        )
    if args.ungrafted and not args.admission_only:
        parser.error("--ungrafted is valid only with --admission-only")
    if args.ungrafted and args.lora:
        parser.error("--ungrafted cannot load --lora")
    if args.ungrafted and not args.model_revision:
        parser.error("--ungrafted requires --model-revision")
    if args.ungrafted and args.preserve_prefix_mass:
        parser.error("--ungrafted cannot use --preserve-prefix-mass")
    if args.nu is not None and not 0.0 < args.nu <= 1.0:
        parser.error("--nu must be in (0, 1]")
    if args.allow_recovery_readout_override and (
        not args.lora or args.preserve_prefix_mass is None
    ):
        parser.error(
            "--allow-recovery-readout-override requires an adapter and an "
            "explicit prefix-mass mode"
        )
    if (
        args.record_start < 0
        or args.prefix_fillers < 0
        or (args.records is not None and args.records < 1)
    ):
        parser.error("record offsets/counts must be positive")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    adapter_hash = _adapter_sha256(args.lora)
    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora,
            device=args.device,
            dtype="float32",
            generation_tokens=args.max_new_tokens,
            window=args.window,
            model_revision=args.model_revision,
            graft_enabled=not args.ungrafted,
            nu=args.nu,
            preserve_prefix_mass=args.preserve_prefix_mass,
            per_boundary_box=args.per_boundary_box,
            solver_seed=args.solver_seed,
            allow_recovery_readout_override=(
                args.allow_recovery_readout_override
            ),
        )
    )
    runtime.ensure_loaded()
    forget_dataset = load_dataset(
        "locuslab/TOFU",
        "forget10",
        split="train",
        revision=args.dataset_revision,
    )
    retain_dataset = load_dataset(
        "locuslab/TOFU",
        "retain90",
        split="train",
        revision=args.dataset_revision,
    )
    forget = list(forget_dataset)
    retain = list(retain_dataset)
    fillers = [str(row["answer"]) for row in retain[: max(args.n_fill + 8, 32)]]
    result = {
        "evaluation": "whole-record in-context unlearning",
        "behavioral_scope": (
            "admission only; no deletion intervention"
            if args.admission_only
            else "fp32 sampled generation; float64 certificate separate"
        ),
        "model": args.model,
        "model_revision": runtime.resolved_model_revision,
        "lora": args.lora,
        "ungrafted": bool(args.ungrafted),
        "nu": runtime.resolved_nu,
        "preserve_prefix_mass": runtime.resolved_preserve_prefix_mass,
        "per_boundary_box": runtime.resolved_per_boundary_box,
        "solver_seed": runtime.resolved_solver_seed,
        "recovery_readout_override": bool(
            args.allow_recovery_readout_override
        ),
        "device": args.device,
        "platform": platform.platform(),
        "provenance": {
            "manifest": {
                "path": manifest_path.as_posix(),
                "name": manifest["name"],
                "version": manifest["version"],
                "sha256": _sha256(manifest_path),
            },
            "model": {
                "id": args.model,
                "requested_revision": args.model_revision,
                "resolved_revision": runtime.resolved_model_revision,
            },
            "adapter": {
                "path": args.lora,
                "content_sha256": adapter_hash,
            },
            "dataset": {
                "id": "locuslab/TOFU",
                "requested_revision": args.dataset_revision,
                "forget_split": "forget10",
                "forget_fingerprint": getattr(forget_dataset, "_fingerprint", None),
                "retain_split": "retain90",
                "retain_fingerprint": getattr(retain_dataset, "_fingerprint", None),
            },
            "run_seed": args.seed,
            "geometry": {
                "window": args.window,
                "n_fill": args.n_fill,
                "prefix_fillers": args.prefix_fillers,
                "record_start": args.record_start,
                "records": args.records,
                "nu": runtime.resolved_nu,
                "per_boundary_box": runtime.resolved_per_boundary_box,
                "solver_seed": runtime.resolved_solver_seed,
                "preserve_prefix_mass": (
                    runtime.resolved_preserve_prefix_mass
                ),
            },
            "admission_gates": {
                "minimum_answer_lift_nats": MIN_SIGNAL,
                "minimum_secret_lift_nats": MIN_SIGNAL,
                "maximum_first_token_rank": MAX_FIRST_TOKEN_RANK,
                "all_fields_must_pass": True,
                "fixed_c_all_boundaries_must_be_feasible": True,
            },
        },
        "sampling": {
            "samples": args.samples,
            "k": args.k_values,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "conditions": [] if args.admission_only else list(args.conditions),
            "kv_cache": bool(args.kv_cache),
            "generation_batch_checkpoints": (
                args.generation_checkpoint_root is not None
            ),
        },
        "whole_record": evaluate(
            runtime,
            manifest,
            forget,
            retain,
            fillers,
            args,
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

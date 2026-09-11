"""Ingestion-time imprint versus packing: standard versus window-disjoint.

The certificate ties the executed decrement to a fixed-C retained-key refit;
the remaining gap to a fully repacked, never-ingested memory is the
ingestion-time imprint on retained keys. The imprint's local channel is
causal and window-local: only retained tokens within one sliding window
*after* a record attended to it through the untouched local layers. This
experiment packs a deletable shadow of >= window padding tokens directly
after the record (deleted together with it), so no retained token lies
within the local window after any record token, and measures the
deletion-versus-repack output gap under both packings.

Both sides of every comparison use the same fixed-C float64 QP alphas:
the deleted state is the refit-without-record on the present memory, and
the repack state is the full solve on a budget-matched memory that never
contained the record (and never contained the shadow). Any residual gap
under shadow packing bounds the non-local (global-layer) imprint channel
plus solver numerics.

Run:
    .venv311/bin/python -m gemma_sv.eval_imprint_packing \
      --record-ids case-zaffre,incident-helios
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from gemma_sv.demo_server.certificate import certificate_overrides
from gemma_sv.demo_server.gate_context import GateRequest
from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_robust_unlearning import (
    PackedMemory,
    pack_memory,
    pad_memory_to_length,
)
from gemma_sv.eval_whole_record_unlearning import (
    DEFAULT_MANIFEST,
    _composite_prompt,
    _field_from_answer,
)

SHADOW_SENTENCE = (
    "This reserved padding line separates records and carries no information."
)


def _kl(log_p: np.ndarray, log_q: np.ndarray) -> float:
    probability = np.exp(log_p)
    return max(0.0, float(np.sum(probability * (log_p - log_q))))


def _shadow_text(tokenizer, window: int) -> str:
    """Neutral deletable padding of at least ``window + 16`` tokens."""

    sentence = f" {SHADOW_SENTENCE}"
    text = SHADOW_SENTENCE
    while len(tokenizer(f"Question:  Answer: {text}\n").input_ids) < window + 16:
        text += sentence
    return text


def _neutral_span(tokenizer, token_count: int) -> list[int]:
    """Neutral padding tokens of an exact length, never resembling a record."""

    ids: list[int] = []
    while len(ids) < token_count:
        ids.extend(
            tokenizer(
                f"Background note: {SHADOW_SENTENCE}\n",
                add_special_tokens=False,
            ).input_ids
        )
    return [int(token_id) for token_id in ids[:token_count]]


def _build_memories(
    runtime: GemmaRuntime,
    spec: dict[str, Any],
    fillers: list[str],
    *,
    window: int,
    n_fill: int,
    prefix_fillers: int,
    copies: int,
    shadow: bool,
    repack_style: str,
) -> tuple[PackedMemory, PackedMemory, int]:
    """Present memory (record [+ shadow]) and a never-ingested repack.

    ``budget_matched`` mirrors the benchmark's behavioral reference: the
    record is absent and trailing fillers restore the token budget, so
    retained fillers shift position. ``position_matched`` substitutes the
    contiguous record span with neutral padding of identical token count:
    every retained token keeps its exact position and content, so the
    deleted-versus-repack gap isolates what the record's presence during
    ingestion did to retained keys, with no packing confound.
    """

    forget_names = [f"forget_copy_{index}" for index in range(copies)]
    records = [
        ("retain", spec["retain_question"], spec["retain_answer"]),
        *[(name, spec["question"], spec["answer"]) for name in forget_names],
    ]
    full_names = set(forget_names)
    if shadow:
        records.append(("shadow", "", _shadow_text(runtime.tokenizer, window)))
        full_names.add("shadow")
    packed = pack_memory(
        runtime.tokenizer,
        records,
        fillers,
        window=window,
        min_fillers=n_fill,
        full_record_names=full_names,
        prefix_fillers=prefix_fillers,
    )
    forget_positions = sorted(
        position
        for name in (*forget_names, *(["shadow"] if shadow else []))
        for position in packed.positions[name]
    )
    present = PackedMemory(
        packed.token_ids,
        {
            "forget": tuple(forget_positions),
            "retain": packed.positions["retain"],
        },
    )
    shadow_tokens = len(packed.positions["shadow"]) if shadow else 0
    if repack_style == "position_matched":
        span_start, span_end = min(forget_positions), max(forget_positions) + 1
        if forget_positions != list(range(span_start, span_end)):
            raise RuntimeError("record span is not contiguous; cannot substitute")
        ids = list(packed.token_ids)
        ids[span_start:span_end] = _neutral_span(
            runtime.tokenizer, span_end - span_start
        )
        repack = PackedMemory(
            ids,
            {
                "forget": tuple(forget_positions),
                "retain": packed.positions["retain"],
            },
        )
    else:
        repack = pack_memory(
            runtime.tokenizer,
            [("retain", spec["retain_question"], spec["retain_answer"])],
            fillers,
            window=window,
            min_fillers=n_fill,
            prefix_fillers=prefix_fillers,
        )
        repack = pad_memory_to_length(
            runtime.tokenizer,
            repack,
            fillers,
            len(present.token_ids),
        )
    return present, repack, shadow_tokens


def _fixed_c_overrides(
    runtime: GemmaRuntime,
    memory: PackedMemory,
    prompt: str,
    forget_positions: tuple[int, ...],
) -> dict:
    query_ids = list(
        runtime.tokenizer(prompt, add_special_tokens=False).input_ids
    )
    full_ids = list(memory.token_ids) + query_ids
    keys = runtime.capture_keys(full_ids)
    layer_ids = sorted(keys)
    n_heads = keys[layer_ids[0]].shape[0]
    return certificate_overrides(
        keys,
        layer_ids,
        forget_positions,
        len(full_ids),
        n_heads,
        skip_incremental=True,
    )


def measure_record(
    runtime: GemmaRuntime,
    spec: dict[str, Any],
    fillers: list[str],
    *,
    window: int,
    n_fill: int,
    prefix_fillers: int,
    copies: int,
    shadow: bool,
    repack_style: str,
) -> dict[str, Any]:
    present, repack, shadow_tokens = _build_memories(
        runtime,
        spec,
        fillers,
        window=window,
        n_fill=n_fill,
        prefix_fillers=prefix_fillers,
        copies=copies,
        shadow=shadow,
        repack_style=repack_style,
    )
    fields = [
        _field_from_answer(spec["question"], spec["answer"], field)
        for field in spec["fields"]
    ]
    probes = [
        (
            field["name"],
            f"\n\nQuestion: {field['question']}\nAnswer: {field['stem']}",
            field["secret"],
        )
        for field in fields
    ]
    probes.append(("composite", _composite_prompt(fields), fields[0]["secret"]))

    results: dict[str, Any] = {}
    for name, prompt, target in probes:
        started = time.perf_counter()
        deleted_overrides = _fixed_c_overrides(
            runtime, present, prompt, present.positions["forget"]
        )
        repack_overrides = _fixed_c_overrides(
            runtime, repack, prompt, repack.positions.get("forget", ())
        )
        target_ids = runtime.target_ids(target, prompt)
        deleted = runtime.score_target(
            list(present.token_ids),
            prompt,
            target_ids,
            request=GateRequest(alpha_by_layer=deleted_overrides["refit"]),
        )
        repacked = runtime.score_target(
            list(repack.token_ids),
            prompt,
            target_ids,
            request=GateRequest(alpha_by_layer=repack_overrides["refit"]),
        )
        results[name] = {
            "imprint_kl_deleted_vs_repack_nats": _kl(
                deleted["first_log_probs"], repacked["first_log_probs"]
            ),
            "target_probability_deleted": deleted["geometric_mean_probability"],
            "target_probability_repack": repacked["geometric_mean_probability"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(
            f"    {spec['record_id']} / {'shadow' if shadow else 'standard'} / "
            f"{name}: KL {results[name]['imprint_kl_deleted_vs_repack_nats']:.3e}",
            flush=True,
        )
    return {
        "packing": "window_disjoint_shadow" if shadow else "standard",
        "repack_style": repack_style,
        "memory_tokens": len(present.token_ids),
        "deleted_positions": len(present.positions["forget"]),
        "shadow_tokens": shadow_tokens,
        "probes": results,
        "max_imprint_kl_nats": max(
            result["imprint_kl_deleted_vs_repack_nats"]
            for result in results.values()
        ),
    }


def main(argv=None) -> int:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--record-ids", default="case-zaffre,incident-helios")
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--lora", default="outputs/gemma_sv_distill/lora_adapter")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--n-fill", type=int, default=22)
    # More retained prefix than the benchmark's 8 keeps every affected
    # fixed-C boundary feasible once the shadow positions are also deleted
    # (retained prefix must exceed 0.3x the total length); both packings
    # share the value so the contrast stays clean, and 16 fillers is the
    # largest prefix that also fits the shadow inside the token budget.
    parser.add_argument("--prefix-fillers", type=int, default=16)
    parser.add_argument(
        "--repack-style",
        choices=("position_matched", "budget_matched"),
        default="position_matched",
        help=(
            "position_matched substitutes the record span with equal-length "
            "neutral padding (isolates the ingestion imprint); budget_matched "
            "mirrors the benchmark's trailing-filler never memory"
        ),
    )
    parser.add_argument(
        "--out", default="outputs/gemma_sv_eval/imprint_packing_v1.json"
    )
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text())
    wanted = [item.strip() for item in args.record_ids.split(",") if item.strip()]
    specs = {
        spec["record_id"]: spec
        for spec in manifest["records"]
        if spec["record_id"] in wanted
    }
    missing = [record_id for record_id in wanted if record_id not in specs]
    if missing:
        parser.error(f"records not in manifest: {missing}")
    copies = int(manifest.get("selection_policy", {}).get("copies", 1))

    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora or None,
            device=args.device,
            dtype="float64",
            generation_tokens=1,
            window=args.window,
        )
    )
    runtime.ensure_loaded()
    retain_rows = list(load_dataset("locuslab/TOFU", "retain90", split="train"))
    fillers = [str(row["answer"]) for row in retain_rows[:32]]

    records = []
    for record_id in wanted:
        spec = specs[record_id]
        row: dict[str, Any] = {"record_id": record_id, "conditions": {}}
        for shadow in (False, True):
            outcome = measure_record(
                runtime,
                spec,
                fillers,
                window=args.window,
                n_fill=args.n_fill,
                prefix_fillers=args.prefix_fillers,
                copies=copies,
                shadow=shadow,
                repack_style=args.repack_style,
            )
            row["conditions"][outcome["packing"]] = outcome
        standard = row["conditions"]["standard"]["max_imprint_kl_nats"]
        disjoint = row["conditions"]["window_disjoint_shadow"][
            "max_imprint_kl_nats"
        ]
        row["imprint_reduction_factor"] = (
            standard / disjoint if disjoint > 0 else float("inf")
        )
        records.append(row)
        print(
            f"  {record_id}: standard max KL {standard:.3e} -> "
            f"window-disjoint {disjoint:.3e}",
            flush=True,
        )

    report = {
        "evaluation": "ingestion-imprint versus packing (float64 output gap)",
        "model": args.model,
        "manifest": manifest["name"],
        "repack_style": args.repack_style,
        "protocol": (
            "deleted state = fixed-C retained-key refit on the present memory; "
            "repack state = the same fixed-C refit on a memory whose record "
            "span was neutral padding during ingestion (position_matched) or "
            "a budget-matched trailing-filler memory (budget_matched); gap = "
            "KL(deleted || repack) at the registered probe's next-token "
            "distribution"
        ),
        "window": args.window,
        "prefix_fillers": args.prefix_fillers,
        "records": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

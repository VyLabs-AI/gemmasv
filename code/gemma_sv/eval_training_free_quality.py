"""Paired block-level quality for the frozen training-free Gemma graft."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics

import torch
import torch.nn.functional as F

from gemma_sv.recovery_protocol import (
    MODEL_REVISION,
    WIKITEXT_REVISION,
    seed_everything,
    tensor_sha256,
    write_json_atomic,
)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_paired_blocks(
    base_nll: list[float],
    graft_nll: list[float],
    *,
    group_size: int,
) -> dict:
    if len(base_nll) != len(graft_nll) or not base_nll:
        raise ValueError("paired non-empty block losses are required")
    if group_size < 1 or len(base_nll) % group_size:
        raise ValueError("group_size must evenly divide the block count")
    block_cost = [
        100.0 * math.expm1(graft - base)
        for base, graft in zip(base_nll, graft_nll)
    ]
    group_rows = []
    for start in range(0, len(base_nll), group_size):
        base = statistics.mean(base_nll[start : start + group_size])
        graft = statistics.mean(graft_nll[start : start + group_size])
        group_rows.append(
            {
                "start_block": start,
                "end_block_exclusive": start + group_size,
                "base_nll": base,
                "graft_nll": graft,
                "relative_cost_percent": 100.0 * math.expm1(graft - base),
            }
        )
    group_cost = [row["relative_cost_percent"] for row in group_rows]
    base_mean = statistics.mean(base_nll)
    graft_mean = statistics.mean(graft_nll)
    return {
        "blocks": len(base_nll),
        "checkpoint_count": 1,
        "independent_training_seeds": 0,
        "blocks_are_independent_replicates": False,
        "inferential_interval": None,
        "base_perplexity": math.exp(base_mean),
        "graft_perplexity": math.exp(graft_mean),
        "relative_cost_percent": 100.0 * math.expm1(graft_mean - base_mean),
        "block_cost_percent": {
            "median": statistics.median(block_cost),
            "p2_5": _percentile(block_cost, 0.025),
            "p97_5": _percentile(block_cost, 0.975),
        },
        "contiguous_groups": {
            "group_size": group_size,
            "n": len(group_rows),
            "interpretation": (
                "descriptive dispersion across fixed contiguous block groups; "
                "not an iid confidence interval"
            ),
            "median_cost_percent": statistics.median(group_cost),
            "q1_cost_percent": _percentile(group_cost, 0.25),
            "q3_cost_percent": _percentile(group_cost, 0.75),
            "minimum_cost_percent": min(group_cost),
            "maximum_cost_percent": max(group_cost),
            "p2_5_cost_percent": _percentile(group_cost, 0.025),
            "p97_5_cost_percent": _percentile(group_cost, 0.975),
            "fraction_positive": sum(value > 0 for value in group_cost)
            / len(group_cost),
            "sample_sd_cost_percent": (
                statistics.stdev(group_cost) if len(group_cost) > 1 else 0.0
            ),
            "rows": group_rows,
        },
    }


@torch.no_grad()
def _block_nll(model, blocks, *, device: str, batch: int) -> list[float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for start in range(0, len(blocks), batch):
        ids = torch.tensor(
            blocks[start : start + batch],
            dtype=torch.long,
            device=device,
        )
        logits = model(input_ids=ids).logits[:, :-1].float()
        labels = ids[:, 1:]
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        losses.extend(float(value) for value in token_loss.mean(1).cpu())
    if was_training:
        model.train()
    return losses


def _load(model_id: str, revision: str, device: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
    ).to(device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--wikitext-revision", default=WIKITEXT_REVISION)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=400)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--nu", type=float, default=0.7)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    from gemma_sv import graft_sv_into_gemma
    from gemma_sv.data import wikitext103_blocks

    seed_everything(args.solver_seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
    )
    blocks = wikitext103_blocks(
        tokenizer,
        args.seq_len,
        max_blocks=args.blocks,
        revision=args.wikitext_revision,
    )
    token_hash = tensor_sha256(torch.tensor(blocks, dtype=torch.long))

    base = _load(args.model, args.model_revision, args.device)
    base_nll = _block_nll(base, blocks, device=args.device, batch=args.batch)
    del base
    if args.device == "mps":
        torch.mps.empty_cache()

    graft = _load(args.model, args.model_revision, args.device)
    graft_sv_into_gemma(
        graft,
        nu=args.nu,
        chunk=args.chunk,
        readout="softmax",
        preserve_prefix_mass=True,
        per_boundary_box=True,
        solver_seed=args.solver_seed,
    )
    graft_nll = _block_nll(
        graft,
        blocks,
        device=args.device,
        batch=args.batch,
    )

    report = {
        "schema": "gemma-sv-training-free-quality-v1",
        "model": args.model,
        "model_revision": args.model_revision,
        "platform": platform.platform(),
        "device": args.device,
        "configuration": {
            "nu": args.nu,
            "chunk": args.chunk,
            "solver_seed": args.solver_seed,
            "preserve_prefix_mass": True,
            "per_boundary_box": True,
            "training_steps": 0,
        },
        "evaluation": {
            "dataset": "Salesforce/wikitext/wikitext-103-raw-v1",
            "revision": args.wikitext_revision,
            "split": "test",
            "blocks": len(blocks),
            "seq_len": args.seq_len,
            "token_ids_sha256": token_hash,
        },
        "summary": summarize_paired_blocks(
            base_nll,
            graft_nll,
            group_size=args.group_size,
        ),
        "base_block_nll": base_nll,
        "graft_block_nll": graft_nll,
    }
    write_json_atomic(Path(args.out), report)
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

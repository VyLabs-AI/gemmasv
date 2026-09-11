"""Training-free cost of grafting the SV gate onto Kimi's global MLA layers.

Replacing softmax with a certified gate changes the function each grafted layer
computes, and the Gemma work needed attention transfer plus LoRA recovery before
quality returned. The first question for Kimi is therefore how much is lost with
no training at all, using only the median-bandwidth heuristic -- and whether the
shared-latent gate (one problem per layer) differs from the per-head gate
(one per head) in either quality or selectivity.

Reports language-model loss on held-out text against the ungrafted model, plus
the gate's own selection statistics: how many context tokens are support vectors
and how many are certified inert.

Usage::

    HF_HUB_CACHE=/path/to/huggingface/cache \\
      .venv311/bin/python -m kimi_sv.eval_graft_degradation --tokens 1024
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .graft import gate_stats, graft_sv_into_kimi, grafted_layers
from .protocol import DEFAULT_MODEL, encode, load_model

DEFAULT_TEXT = "data/tinystories_valid.txt"


def load_tokens(tokenizer: Any, path: str, n_tokens: int) -> mx.array:
    """Tokenize the head of a local text file to exactly ``n_tokens`` tokens."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    # Take generously more characters than tokens, then trim in token space.
    ids = encode(tokenizer, raw[: n_tokens * 8], special=True)
    if ids.shape[1] < n_tokens:
        raise ValueError(f"{path} yielded only {ids.shape[1]} tokens; need {n_tokens}")
    return ids[:, :n_tokens]


def language_model_loss(model: Any, tokens: mx.array) -> float:
    """Mean next-token negative log likelihood, in nats."""
    logits = model(tokens[:, :-1])
    logits = logits.astype(mx.float32)
    targets = tokens[:, 1:]
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logprobs, targets[..., None], axis=-1)
    loss = -float(mx.mean(picked).item())
    return loss


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--fista-iters", type=int, default=80)
    ap.add_argument("--modes", nargs="+", default=["latent", "per_head"])
    ap.add_argument(
        "--preserve-mass",
        action="store_true",
        help="rescale the gated prefix to its pre-gate mass share so the gate "
        "redistributes within the prefix instead of crushing it",
    )
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="restrict the graft to these layer indices (default: all global layers)",
    )
    ap.add_argument("--out", default="outputs/kimi_sv/graft_degradation.json")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)

    rows: List[Dict[str, Any]] = []

    # The graft mutates the model in place, so each configuration gets a freshly
    # loaded copy rather than an unwound one.
    def fresh():
        return load_model(args.model, tiny=args.tiny)

    model, tokenizer, label = fresh()
    tokens = load_tokens(tokenizer, args.text, args.tokens)
    print(f"loaded {label}; {int(tokens.shape[1])} tokens of {args.text}", flush=True)

    t0 = time.perf_counter()
    baseline = language_model_loss(model, tokens)
    baseline_seconds = time.perf_counter() - t0
    print(
        f"  ungrafted     loss {baseline:.4f} nats"
        f" (ppl {math.exp(baseline):.2f}) in {baseline_seconds:.1f}s",
        flush=True,
    )
    del model

    for mode in args.modes:
        model, _, _ = fresh()
        replaced = graft_sv_into_kimi(
            model,
            mode=mode,
            nu=args.nu,
            chunk=args.chunk,
            fista_iters=args.fista_iters,
            preserve_prefix_mass=args.preserve_mass,
            collect_stats=True,
            layers=args.layers,
        )
        t0 = time.perf_counter()
        loss = language_model_loss(model, tokens)
        seconds = time.perf_counter() - t0
        stats = gate_stats(model)

        support = [s["support_mean"] for s in stats.values()]
        inert = [s["inert_fraction"] for s in stats.values()]
        rows.append(
            {
                "mode": mode,
                "grafted_layers": replaced,
                "problems_per_layer": next(iter(stats.values()))["problems"]
                if stats
                else None,
                "loss_nats": loss,
                "perplexity": math.exp(loss),
                "loss_delta_vs_ungrafted": loss - baseline,
                "seconds": seconds,
                "slowdown_vs_ungrafted": seconds / max(baseline_seconds, 1e-9),
                "support_mean": sum(support) / len(support) if support else None,
                "inert_fraction_mean": sum(inert) / len(inert) if inert else None,
                "per_layer_stats": stats,
            }
        )
        print(
            f"  {mode:13s} loss {loss:.4f} nats (ppl {math.exp(loss):.2f});"
            f" delta {loss - baseline:+.4f};"
            f" inert {rows[-1]['inert_fraction_mean']:.1%};"
            f" support {rows[-1]['support_mean']:.0f};"
            f" {seconds:.1f}s ({seconds / max(baseline_seconds, 1e-9):.1f}x)",
            flush=True,
        )
        del model

    result = {
        "model": label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "text": args.text,
        "tokens": int(tokens.shape[1]),
        "gate": {
            "nu": args.nu,
            "chunk": args.chunk,
            "fista_iters": args.fista_iters,
            "preserve_prefix_mass": args.preserve_mass,
            "bandwidth": "median pairwise key distance, frozen at first forward",
        },
        "ungrafted": {
            "loss_nats": baseline,
            "perplexity": math.exp(baseline),
            "seconds": baseline_seconds,
        },
        "grafted": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

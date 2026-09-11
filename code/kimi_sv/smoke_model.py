"""Load Kimi lazily and run one minimal forward pass.

Use this before any paper-scale Kimi experiment. It avoids mlx-lm's eager
``mx.eval(model.parameters())`` load, clears stale allocator buffers, and
reports Metal memory after a one-token prefill.
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx

from .protocol import DEFAULT_MODEL, encode, load_model


def _gib(value: int | float) -> float:
    return float(value) / (1024 ** 3)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Test")
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args(argv)

    mx.clear_cache()
    mx.set_cache_limit(int(args.cache_limit_gb * 1024 ** 3))
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, tokenizer, label = load_model(
        args.model,
        tiny=args.tiny,
        lazy=not args.eager,
    )
    print(
        f"loader returned for {label} in {time.perf_counter() - started:.2f}s "
        f"(lazy={not args.eager})",
        flush=True,
    )

    tokens = encode(tokenizer, args.prompt, special=True)[:, :1]
    cache = model.make_cache()
    forward_started = time.perf_counter()
    logits = model(tokens, cache=cache)
    mx.eval(logits)
    print(
        f"one-token smoke completed in "
        f"{time.perf_counter() - forward_started:.2f}s; "
        f"logits={tuple(logits.shape)}; "
        f"active={_gib(mx.get_active_memory()):.2f} GiB; "
        f"peak={_gib(mx.get_peak_memory()):.2f} GiB; "
        f"cache={_gib(mx.get_cache_memory()):.2f} GiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

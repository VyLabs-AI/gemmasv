"""Benchmark the exact operations exposed by the hosted demo API.

This uses only fictional presets and writes raw timings under the gitignored
``outputs/`` tree.  It is intentionally separate from the public replay profile.

Run:
    .venv311/bin/python -m gemma_sv.benchmark_hosted_demo \
      --scenario halcyon-crimson-sparrow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

from gemma_sv.demo_server.gemma_engine import GemmaDemoEngine
from gemma_sv.demo_server.scenarios import PRESETS
from gemma_sv.demo_server.span import SelectedSpan
from gemma_sv.demo_server.state import SessionStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=sorted(PRESETS),
        default="halcyon-crimson-sparrow",
    )
    parser.add_argument(
        "--skip-certificate",
        action="store_true",
        help="measure only the fp32/MPS interactive path",
    )
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_demo/hosted_benchmark.json",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.scenario]
    engine = GemmaDemoEngine.from_environment()
    session = SessionStore(ttl_seconds=3_600).create(domain=preset.domain)
    selection = SelectedSpan(
        preset.memory_text,
        preset.secret_start,
        preset.secret_end,
    )
    results = {
        "scenario": preset.slug,
        "platform": platform.platform(),
        "runtime": engine.runtime_info(),
        "steps": {},
    }

    def timed(name, fn, *fn_args, **fn_kwargs):
        started = time.perf_counter()
        value = fn(*fn_args, **fn_kwargs)
        results["steps"][name] = {
            "wall_seconds": time.perf_counter() - started,
            "result": value,
        }
        print(f"{name:>18}: {results['steps'][name]['wall_seconds']:.2f} s", flush=True)
        return value

    timed(
        "ingest",
        engine.ingest,
        session,
        selection,
        preset.audit_probe,
        preset.target_value,
    )
    timed("recall", engine.recall, session)
    timed("forget", engine.forget, session)
    timed(
        "attack_icul",
        engine.attack,
        session,
        method="icul",
        kind="extraction",
    )
    timed(
        "attack_exact",
        engine.attack,
        session,
        method="exact",
        kind="extraction",
    )
    if not args.skip_certificate:
        timed(
            "certificate",
            engine.certify,
            session,
            progress=lambda value: print(
                f"\rcertificate progress {100 * value:5.1f}%", end="", flush=True
            ),
        )
        print()
        timed("twin", engine.twin_start, session)
        # Labeled exact/refit distributions for the committed replay profile.
        results["twin_labeled"] = session.engine_state.get("twin_distributions")

    results["runtime"] = engine.runtime_info()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=_json_default) + "\n")
    print(f"wrote {output}")
    return 0


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())

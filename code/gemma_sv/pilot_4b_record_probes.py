"""Sweep completion-probe formats for the recovered 4B record scenarios.

This is calibration only. It loads the model once, keeps the packed conversation
fixed, and varies only the registered probe. Results are written under
``outputs/`` and must not be treated as benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gemma_sv.demo_server.gemma_engine import (
    GemmaDemoEngine,
    GemmaRuntime,
    RuntimeConfig,
)
from gemma_sv.demo_server.scenarios import PRESETS
from gemma_sv.demo_server.span import SelectedSpan
from gemma_sv.demo_server.state import SessionStore


def probes(preset, target_value: str | None = None) -> dict[str, str]:
    target = target_value or preset.target_value
    stem = preset.fact[: preset.fact.index(target)].rstrip()
    key = (
        "working_attribution"
        if preset.domain == "cybersecurity"
        else "working_diagnosis"
    )
    return {
        "record-lookup": f"\n\nRecord lookup:\n{stem}",
        "bare-prefix": f"\n\n{stem}",
        "complete-only": (
            "\n\nComplete only the stored JSON string value; do not explain.\n"
            f"{stem}"
        ),
        "field-arrow": f"\n\nPersistent field completion:\n{key} =>",
        "qa-plus-field": (
            f"\n\nQuestion: {preset.question}\n"
            f"Return only the exact stored value.\n{key} =>"
        ),
        "one-shot": (
            '\n\nExample field completion:\nexample_value => "Sunhaven"\n'
            f"Stored field completion:\n{key} =>"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument(
        "--lora", default="outputs/gemma_sv_distill_4b/lora_adapter"
    )
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_demo/record_probe_pilot_4b.json",
    )
    parser.add_argument("--scenario", choices=tuple(PRESETS), default=None)
    parser.add_argument("--probe", default=None)
    parser.add_argument(
        "--audit-target",
        action="append",
        default=[],
        help="candidate target already present in the record; repeatable",
    )
    args = parser.parse_args(argv)

    fast = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora or None,
            device=args.device,
            dtype="float32",
            generation_tokens=10,
            window=1024,
            copies=args.copies,
        )
    )
    certificate = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora or None,
            device="cpu",
            dtype="float64",
            generation_tokens=1,
            window=1024,
            copies=args.copies,
        )
    )
    engine = GemmaDemoEngine(fast, certificate)
    rows = []
    for preset in PRESETS.values():
        if args.scenario and preset.slug != args.scenario:
            continue
        selection = SelectedSpan(
            preset.memory_text,
            preset.secret_start,
            preset.secret_end,
        )
        targets = args.audit_target or [preset.target_value]
        for target in targets:
            if target not in preset.fact:
                raise ValueError(f"target {target!r} is not in {preset.slug}")
            for name, probe in probes(preset, target).items():
                if args.probe and name != args.probe:
                    continue
                session = SessionStore(ttl_seconds=3_600).create(domain=preset.domain)
                started = time.perf_counter()
                try:
                    ingest = engine.ingest(
                        session,
                        selection,
                        probe,
                        target,
                    )
                    recall = engine.recall(session)
                    row = {
                        "scenario": preset.slug,
                        "audit_target": target,
                        "probe": name,
                        "probe_text": probe,
                        "ingest": ingest,
                        "recall": recall,
                        "wall_seconds": time.perf_counter() - started,
                    }
                    admission = recall["admission"]
                    print(
                        f"{preset.domain:>13} {target[:22]:<22} {name:<14} "
                        f"{admission['status']:<16} "
                        f"lift={admission['probability_ratio']:8.1f}x "
                        f"rank={recall['first_target_token_rank']:>6} "
                        f"greedy={admission['greedy_match']} "
                        f"gen={recall['generated_text'][:38]!r}",
                        flush=True,
                    )
                except Exception as error:
                    row = {
                        "scenario": preset.slug,
                        "audit_target": target,
                        "probe": name,
                        "error": f"{type(error).__name__}: {error}",
                        "wall_seconds": time.perf_counter() - started,
                    }
                    print(
                        f"{preset.domain:>13} {target[:22]:<22} {name:<14} "
                        f"ERROR {error}",
                        flush=True,
                    )
                rows.append(row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": rows}, indent=2, default=str) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


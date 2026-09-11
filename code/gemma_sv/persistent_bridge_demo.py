"""Run one recovered-Gemma prefill-once deletion and exact/proxy bridge."""
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
from gemma_sv.demo_server.span import CharacterRange, SelectedSpan
from gemma_sv.demo_server.state import SessionRecord


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument(
        "--lora",
        default="outputs/gemma_sv_distill/lora_adapter",
    )
    parser.add_argument("--preset", default="whole-case-zaffre")
    parser.add_argument("--fast-device", default="mps")
    parser.add_argument("--certificate-device", default="cpu")
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_demo/persistent_state_bridge_v2.json",
    )
    args = parser.parse_args(argv)

    preset = PRESETS[args.preset]
    ranges = tuple(
        CharacterRange(int(item["start"]), int(item["end"]))
        for item in preset.deletion_ranges
    )
    selection = SelectedSpan(
        preset.memory_text,
        ranges[0].start,
        ranges[0].end,
        deletion_ranges=ranges,
        deletion_scope=preset.delete_scope,
        record_id=preset.slug,
    )
    fast = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora or None,
            device=args.fast_device,
            dtype="float32",
            generation_tokens=4,
            window=512,
            copies=1,
        )
    )
    certificate = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora or None,
            device=args.certificate_device,
            dtype="float64",
            generation_tokens=1,
            window=512,
            copies=1,
        )
    )
    engine = GemmaDemoEngine(fast, certificate)
    now = time.monotonic()
    session = SessionRecord("persistent-bridge", now, now)
    ingest = engine.ingest(
        session,
        selection,
        preset.audit_probe,
        audit_target=preset.target_value,
    )
    recall = engine.recall(session)
    forgotten = engine.forget(session)
    certificate_result = engine.certify(session)
    report = {
        "evaluation": "recovered Gemma prefill-once deletion bridge",
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "model": args.model,
        "preset": args.preset,
        "memory_prefilled_once": ingest["memory_prefilled_once"],
        "memory_input_digest": ingest["memory_input_digest"],
        "recall": {
            "target_probability": recall["target_probability"],
            "floor_probability": recall["floor_probability"],
        },
        "forget": {
            "target_probability": forgotten["after_probability"],
            "floor_probability": forgotten["floor_probability"],
            "deletion_ms": forgotten["deletion_ms"],
        },
        "certificate": certificate_result,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")
    print(
        "KL decrement/refit="
        f"{certificate_result['kl_nats']:.3e}; "
        "KL decrement/proxy="
        f"{certificate_result['exact_proxy_kl_nats']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

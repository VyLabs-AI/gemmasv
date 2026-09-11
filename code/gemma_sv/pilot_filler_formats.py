"""Compare natural-language padding formats for the persistent conversation log.

Variants:
  qa    — fillers as "Question: … Answer: …" (same format as the record)
  ua    — fillers as "User: … Assistant: …" dialogue (natural chat, distinct format)
  qa3   — qa fillers but the record is written three times instead of two
  prose — fillers as plain declarative sentences

Each variant runs both shipped presets through ingest → recall on the live fast
runtime and reports the admission verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from gemma_sv.demo_server.gemma_engine import (
    _QA_FILLERS,
    GemmaDemoEngine,
    GemmaRuntime,
    RuntimeConfig,
)
from gemma_sv.demo_server.scenarios import PRESETS
from gemma_sv.demo_server.span import SelectedSpan
from gemma_sv.demo_server.state import SessionStore


def to_user_assistant(filler: str) -> str:
    return filler.replace("Question: ", "User: ").replace("Answer: ", "Assistant: ")


def to_prose(filler: str) -> str:
    return re.sub(r"^Question: .*? Answer: ", "", filler)


def strip_fictional(filler: str) -> str:
    """Remove the word that collides with the record and the probe question."""

    cleaned = re.sub(r"\b[Tt]he fictional\b", "the", filler)
    cleaned = re.sub(r"\b[Ff]ictional\b ", "", cleaned)
    return cleaned


VARIANTS = {
    "qa": (list(_QA_FILLERS), 2),
    "ua": ([to_user_assistant(filler) for filler in _QA_FILLERS], 2),
    "qa3": (list(_QA_FILLERS), 3),
    "prose": ([to_prose(filler) for filler in _QA_FILLERS], 2),
    "qa_nofic": ([strip_fictional(filler) for filler in _QA_FILLERS], 2),
    "ua_nofic": (
        [strip_fictional(to_user_assistant(filler)) for filler in _QA_FILLERS],
        2,
    ),
    "prose_nofic": (
        [strip_fictional(to_prose(filler)) for filler in _QA_FILLERS],
        2,
    ),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument(
        "--out", default="outputs/gemma_sv_demo/filler_format_pilot.json"
    )
    args = parser.parse_args(argv)

    fast = GemmaRuntime(
        RuntimeConfig(
            lora_path="outputs/gemma_sv_distill/lora_adapter",
            device="mps",
            dtype="float32",
            generation_tokens=8,
        )
    )
    fast.ensure_loaded()
    certificate = GemmaRuntime(
        RuntimeConfig(device="cpu", dtype="float64", generation_tokens=1)
    )
    engine = GemmaDemoEngine(fast, certificate)

    results = []
    for variant in args.variants:
        fillers, copies = VARIANTS[variant]
        fast.fillers = fillers
        fast.config = RuntimeConfig(
            lora_path=fast.config.lora_path,
            device=fast.config.device,
            dtype=fast.config.dtype,
            generation_tokens=fast.config.generation_tokens,
            copies=copies,
        )
        for preset in PRESETS.values():
            selection = SelectedSpan(
                preset.memory_text, preset.secret_start, preset.secret_end
            )
            session = SessionStore(ttl_seconds=3_600).create(domain=preset.domain)
            started = time.perf_counter()
            row = {"variant": variant, "preset": preset.slug}
            try:
                engine.ingest(
                    session,
                    selection,
                    preset.audit_probe,
                    preset.target_value,
                )
                recall = engine.recall(session)
                row["recall"] = {
                    "generated_text": recall["generated_text"],
                    "target_probability": recall["target_probability"],
                    "floor_probability": recall["floor_probability"],
                    "admission": recall["admission"],
                }
                admission = recall["admission"]
                print(
                    f"{variant:>6} {preset.slug:<26} {admission['status']:<16} "
                    f"lift {admission['probability_ratio']:9.1f}x "
                    f"greedy={admission['greedy_match']} gen={recall['generated_text'][:38]!r}",
                    flush=True,
                )
            except Exception as exc:  # report and continue
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{variant:>6} {preset.slug:<26} ERROR {exc}", flush=True)
            row["wall_seconds"] = time.perf_counter() - started
            results.append(row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": results}, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

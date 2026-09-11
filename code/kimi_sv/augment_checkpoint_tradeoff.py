"""Add sparse-checkpoint trade-offs to an existing aggregate replay report.

This command loads only the tokenizer and credentialed source rows; it does not
load Kimi weights or generate text.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mimic import assert_no_source_text
from mimic import cds as cds_source
from mimic import notes as notes_source

from .eval_mimic_deletion import checkpoint_tradeoff
from .protocol import DEFAULT_MODEL, case_records


SOURCES = {"cds": cds_source, "notes": notes_source}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="cds")
    parser.add_argument("--data-dir")
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument(
        "--body-chars",
        type=int,
        default=notes_source.DEFAULT_BODY_CHARS,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    loader = SOURCES[args.source]
    env_var = (
        "MIMIC_EXT_CDS_DIR" if args.source == "cds" else "MIMIC_NOTE_DIR"
    )
    data_dir = args.data_dir or os.getenv(env_var)
    if not data_dir:
        parser.error(f"--data-dir or {env_var} is required")
    if args.source == "notes":
        cases, _ = loader.load_cases(
            data_dir,
            args.records,
            skip=args.skip,
            body_chars=args.body_chars,
        )
    else:
        cases, _ = loader.load_cases(
            data_dir,
            args.records,
            skip=args.skip,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=os.getenv("HF_HUB_CACHE"),
    )
    records = case_records(tokenizer, cases)
    report = json.loads(Path(args.report).read_text())
    per_position = report.get("per_position", [])
    checkpoint_bytes = int(
        report.get("cost", {}).get("checkpoint_bytes_per_boundary", 0)
    )
    if not per_position or checkpoint_bytes <= 0:
        parser.error("report lacks per-position timings or checkpoint size")
    report["checkpoint_tradeoff"] = checkpoint_tradeoff(
        [record.n_tokens for record in records],
        per_position,
        checkpoint_bytes,
    )
    report["checkpoint_tradeoff"]["timing_status"] = (
        "estimated from measured dense replay timings; token counts are exact"
    )
    payload = json.dumps(report, indent=2)
    assert_no_source_text(payload, cases)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload + "\n")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

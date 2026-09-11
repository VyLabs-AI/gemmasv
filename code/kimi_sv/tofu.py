"""TOFU fictitious-author facts as Kimi deletion cases.

The Gemma act evaluates on TOFU (locuslab/TOFU, ``forget10`` split), packed
into persistent context as ``Question: ... Answer: ...`` records. This loader
serves the same facts, in the same record format and dataset order, to the
hybrid pipeline, so one table can carry both architectures on one substrate.

TOFU is public and fictitious: no disclosure discipline applies, and greedy
readbacks are allowed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

PREAMBLE = "Persistent memory:\n"


def load_cases(
    n: int, *, skip: int = 0
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    from datasets import load_dataset

    rows = list(load_dataset("locuslab/TOFU", "forget10", split="train"))
    if skip + n > len(rows):
        raise SystemExit(f"asked for {n} rows after skip {skip}, have {len(rows)}")

    cases: List[Dict[str, Any]] = []
    for i, row in enumerate(rows[skip : skip + n]):
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        cases.append(
            {
                "key": f"tofu-{skip + i:03d}",
                # The Gemma packer's record text, verbatim.
                "text": f"Question: {question} Answer: {answer}\n",
                "question": f"Question: {question}\nAnswer:",
                "secret": f" {answer}",
            }
        )
    stats = {
        "dataset": "locuslab/TOFU:forget10",
        "rows_available": len(rows),
        "skip": skip,
        "taken": n,
    }
    return cases, stats

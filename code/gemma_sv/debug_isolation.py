"""Distinguish session-state bleed from phrasing noise for the isolation case.

Runs the same fact through the live API three ways and compares recall:
  1. solo — fresh session, ingest, recall immediately;
  2. after-foreign-ingest — ingest A, ingest B, recall A;
  3. after-foreign-recall — ingest A, ingest+recall B, then recall A.

If (1) recalls but (3) does not, inference on session B contaminates session A,
i.e. a real isolation bug rather than fact-dependent weakness.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from gemma_sv.demo_server.scenarios import (
    normalize_question,
    scaffold_memory,
    scaffold_offset,
    stem_probe,
)

FACT_A = (
    "Access to the fictional archive lift was assigned to the operator Redpoll."
)
VALUE_A = "Redpoll"
QUESTION_A = "Who was assigned access to the fictional archive lift?"

FACT_B = "Fictional botanist Ila Chen grows winter saffron in greenhouse four."
VALUE_B = "winter saffron"
QUESTION_B = "What does fictional botanist Ila Chen grow in greenhouse four?"


def call(base: str, method: str, path: str, body: dict | None = None):
    request = urllib.request.Request(
        f"{base}{path}",
        method=method,
        headers={"Content-Type": "application/json"},
        data=None if body is None else json.dumps(body).encode(),
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


def ingest(base: str, fact: str, value: str, question: str) -> str:
    question = normalize_question(question)
    session = call(base, "POST", "/api/v1/sessions", {"domain": "custom"})[
        "session_id"
    ]
    span_start = fact.index(value)
    call(
        base,
        "POST",
        f"/api/v1/sessions/{session}/ingest",
        {
            "memory_text": scaffold_memory(question, fact),
            "secret_start": scaffold_offset(question) + span_start,
            "secret_end": scaffold_offset(question) + span_start + len(value),
            "audit_probe": stem_probe(question, fact, span_start),
        },
    )
    return session


def recall(base: str, session: str) -> dict:
    return call(base, "POST", f"/api/v1/sessions/{session}/recall")


def purge(base: str, session: str) -> None:
    call(base, "DELETE", f"/api/v1/sessions/{session}")


def report(tag: str, result: dict) -> None:
    admission = result["admission"]
    print(
        f"{tag:>22}: {admission['status']:<16} p={result['target_probability']:.4g} "
        f"lift={admission['probability_ratio']:.1f}x greedy={admission['greedy_match']} "
        f"gen={result['generated_text'][:40]!r}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8002")
    args = parser.parse_args(argv)
    base = args.api.rstrip("/")

    session = ingest(base, FACT_A, VALUE_A, QUESTION_A)
    report("solo", recall(base, session))
    purge(base, session)

    session_a = ingest(base, FACT_A, VALUE_A, QUESTION_A)
    session_b = ingest(base, FACT_B, VALUE_B, QUESTION_B)
    report("after-foreign-ingest", recall(base, session_a))
    purge(base, session_a)
    purge(base, session_b)

    session_a = ingest(base, FACT_A, VALUE_A, QUESTION_A)
    session_b = ingest(base, FACT_B, VALUE_B, QUESTION_B)
    report("foreign-recall-b", recall(base, session_b))
    report("after-foreign-recall", recall(base, session_a))
    purge(base, session_a)
    purge(base, session_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

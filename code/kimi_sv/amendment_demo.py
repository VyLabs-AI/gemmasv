"""The paper's motivating scenario, run end to end: amend a scribe's memory.

A fictional intake conversation is ingested statement by statement. One
statement is wrong -- the family history records ``mother had breast
cancer`` where imaging later shows a benign lump. The demo runs both of the
paper's mechanisms on it:

  subtraction   the statement's positions leave every grafted gate (instant);
                the residual the recurrent state keeps is measured.
  amendment     rewind to the statement's boundary, ingest the corrected
                statement, replay the rest -- the same operation as deletion,
                audited bitwise against a memory that heard the corrected
                statement from the start.

Every quoted string is invented, so the full transcript is printable. The
report captures greedy answers, teacher-forced targets, costs, and the
bitwise audit. Run ``--tiny`` for a plumbing check on random weights.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .graft import graft_sv_into_kimi, set_drop_positions
from .probes import teacher_forced
from .protocol import DEFAULT_MODEL, answer_text, encode, load_model
from .records import Record, RecordMemory, max_abs_diff, state_max_abs_diff

PREAMBLE = (
    "Ambient scribe memory. Each entry is one statement from today's visit. "
    "Answer questions using only these entries.\n\n"
)

STATEMENTS: List[Tuple[str, str]] = [
    (
        "hpi",
        "Statement 1 (patient): I've had this cough for about three weeks, "
        "and it is worse at night.\n\n",
    ),
    (
        "meds",
        "Statement 2 (patient): I take lisinopril 10 mg daily and a "
        "multivitamin.\n\n",
    ),
    (
        "famhx",
        "Statement 3 (patient): My mother had breast cancer.\n\n",
    ),
    (
        "allergy",
        "Statement 4 (patient): I'm allergic to penicillin; it gives me "
        "hives.\n\n",
    ),
    (
        "social",
        "Statement 5 (patient): I quit smoking six years ago.\n\n",
    ),
]

CORRECTED_FAMHX = (
    "Statement 3 (patient, corrected): My mother had a breast lump that "
    "imaging showed was benign, not cancer.\n\n"
)

QUESTION = (
    "Question: According to the statements, what condition did the "
    "patient's mother have?\nAnswer:"
)
WRONG_TARGET = " Breast cancer."
CORRECTED_TARGET = " A benign breast lump."
AUDIT_PROBE = "\nAudit complete."


def build(
    model: Any, tokenizer: Any, famhx_text: str
) -> Tuple[RecordMemory, Tuple[int, int]]:
    """A scribe memory over the five statements; returns the famhx span."""
    records = [Record(key="__preamble__", tokens=encode(tokenizer, PREAMBLE, special=True))]
    for key, text in STATEMENTS:
        body = famhx_text if key == "famhx" else text
        records.append(Record(key=key, tokens=encode(tokenizer, body), text=body))

    memory = RecordMemory(model)
    span: Optional[Tuple[int, int]] = None
    offset = 0
    for r in records:
        if r.key == "famhx":
            span = (offset, offset + r.n_tokens)
        memory.ingest(r)
        offset += r.n_tokens
    assert span is not None
    return memory, span


def scores(memory: RecordMemory, tokenizer: Any) -> Dict[str, Any]:
    q = encode(tokenizer, "\n" + QUESTION)
    wrong = teacher_forced(memory, q, encode(tokenizer, WRONG_TARGET))
    corrected = teacher_forced(memory, q, encode(tokenizer, CORRECTED_TARGET))
    return {
        "wrong_target_mean_logprob": wrong.mean_log_probability,
        "wrong_target_first_token_rank": wrong.first_token_rank,
        "corrected_target_mean_logprob": corrected.mean_log_probability,
        "corrected_target_first_token_rank": corrected.first_token_rank,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--nu", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--greedy-tokens", type=int, default=16)
    ap.add_argument("--out", default="outputs/kimi_sv/amendment_demo.json")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    model, tokenizer, label = load_model(args.model, tiny=args.tiny)
    print(f"loaded {label} in {(time.perf_counter() - t0)/60:.1f} min", flush=True)

    grafted = graft_sv_into_kimi(
        model,
        mode="latent",
        nu=args.nu,
        chunk=args.chunk,
        preserve_prefix_mass=True,
    )
    print(f"grafted SV gate onto layers {grafted}", flush=True)

    def greedy(memory: RecordMemory) -> str:
        return answer_text(memory, tokenizer, QUESTION, max_tokens=args.greedy_tokens)

    # The scribe's memory as heard, and the reference that heard the
    # corrected statement from the start.
    heard, span = build(model, tokenizer, STATEMENTS[2][1])
    reference, _ = build(model, tokenizer, CORRECTED_FAMHX)

    out: Dict[str, Any] = {"conditions": {}}
    out["conditions"]["as_heard"] = {
        **scores(heard, tokenizer),
        "greedy_answer": greedy(heard),
    }
    out["conditions"]["corrected_from_start"] = {
        **scores(reference, tokenizer),
        "greedy_answer": greedy(reference),
    }
    print(f"as heard          : {out['conditions']['as_heard']['greedy_answer']!r}")
    print(
        "corrected-from-start reference: "
        f"{out['conditions']['corrected_from_start']['greedy_answer']!r}",
        flush=True,
    )

    # -- subtraction: instant, gates only ---------------------------------
    t0 = time.perf_counter()
    set_drop_positions(model, list(range(span[0], span[1])))
    subtraction_seconds = time.perf_counter() - t0
    out["conditions"]["after_subtraction"] = {
        **scores(heard, tokenizer),
        "greedy_answer": greedy(heard),
        "seconds": subtraction_seconds,
    }
    set_drop_positions(model, None)
    print(
        f"after subtraction : "
        f"{out['conditions']['after_subtraction']['greedy_answer']!r} "
        f"({subtraction_seconds*1000:.1f} ms)",
        flush=True,
    )

    # -- amendment: rewind, ingest the correction, replay the rest --------
    corrected_record = Record(
        key="famhx", tokens=encode(tokenizer, CORRECTED_FAMHX), text=CORRECTED_FAMHX
    )
    report = heard.amend("famhx", corrected_record)
    audit = encode(tokenizer, AUDIT_PROBE)
    logit_delta = max_abs_diff(heard.probe(audit), reference.probe(audit))
    state_delta = state_max_abs_diff(heard, reference)
    out["conditions"]["after_amendment"] = {
        **scores(heard, tokenizer),
        "greedy_answer": greedy(heard),
        "seconds": report.seconds,
        "replayed_tokens": report.replayed_tokens,
        "replayed_records": report.replayed_records,
        "checkpoint_bytes": report.checkpoint_bytes,
    }
    out["audit"] = {
        "max_abs_logit_delta_vs_corrected_from_start": logit_delta,
        "max_abs_kda_state_delta_vs_corrected_from_start": state_delta,
        "greedy_answer_matches_reference": (
            out["conditions"]["after_amendment"]["greedy_answer"]
            == out["conditions"]["corrected_from_start"]["greedy_answer"]
        ),
    }
    print(
        f"after amendment   : "
        f"{out['conditions']['after_amendment']['greedy_answer']!r} "
        f"({report.seconds:.2f} s / {report.replayed_tokens} tokens replayed)",
        flush=True,
    )
    print(
        f"audit: logit delta {logit_delta:.3e}, state delta {state_delta:.3e}, "
        f"greedy matches reference: {out['audit']['greedy_answer_matches_reference']}",
        flush=True,
    )

    out.update(
        {
            "model": label,
            "host": {"platform": platform.platform(), "machine": platform.machine()},
            "protocol": {
                "nu": args.nu,
                "chunk": args.chunk,
                "preserve_prefix_mass": True,
                "greedy_tokens": args.greedy_tokens,
                "statements": [text for _, text in STATEMENTS],
                "corrected_statement": CORRECTED_FAMHX,
                "question": QUESTION,
                "wrong_target": WRONG_TARGET,
                "corrected_target": CORRECTED_TARGET,
            },
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

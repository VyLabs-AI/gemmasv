"""Can KDA's own forgetting remove a record? Decay versus exact replay.

Three ways of trying to make a resident record stop mattering are compared
against one reference: a context that never ingested it.

1. *Wait*  -- keep ingesting unrelated content and let the learned per-channel
   retention gates attenuate the record naturally.
2. *Decay* -- attenuate the recurrent state directly by a retention factor,
   the most favourable form of the mechanism KDA already implements.
3. *Replay* -- roll the hybrid cache back to the record's boundary and replay
   the survivors.

Only the third can separate one record from the rest, because the state's
channels are shared by every token ever written. The point of the sweep is to
show what the first two cost: attenuation strong enough to suppress the target
also suppresses the records that were meant to be kept.

Usage::

    HF_HUB_CACHE=/path/to/huggingface/cache \\
      .venv311/bin/python -m kimi_sv.eval_decay_contrast
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .decay import scale_recurrent_state
from .probes import lift_nats
from .protocol import (
    CASES,
    DEFAULT_MODEL,
    answer_text,
    case_records,
    encode,
    filler_record,
    load_model,
    preamble_record,
    score_all,
)
from .records import Record, RecordMemory, max_abs_diff, state_max_abs_diff


def measure(
    arm: str,
    memory: RecordMemory,
    reference: RecordMemory,
    tokenizer: Any,
    cases: List[Dict[str, str]],
    victim_key: str,
    detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score one intervention against the never-ingested reference."""
    stats = score_all(memory, tokenizer, cases)
    ref_stats = score_all(reference, tokenizer, cases)

    victim = stats[victim_key]
    retained = [v for k, v in stats.items() if k != victim_key]
    retained_ref = [v for k, v in ref_stats.items() if k != victim_key]

    victim_question = next(c["question"] for c in cases if c["key"] == victim_key)
    probe = encode(tokenizer, "\n" + victim_question)

    return {
        "arm": arm,
        **(detail or {}),
        "victim": {
            "first_token_rank": victim.first_token_rank,
            "first_token_probability": victim.first_token_probability,
            "mean_log_probability": victim.mean_log_probability,
            "nats_vs_reference": lift_nats(victim, ref_stats[victim_key]),
        },
        "retained": {
            "mean_first_token_rank": statistics.mean(
                s.first_token_rank for s in retained
            ),
            "worst_first_token_rank": max(s.first_token_rank for s in retained),
            "mean_first_token_probability": statistics.mean(
                s.first_token_probability for s in retained
            ),
            "mean_nats_vs_reference": statistics.mean(
                lift_nats(s, r) for s, r in zip(retained, retained_ref)
            ),
        },
        "residual": {
            "max_abs_logit_delta": max_abs_diff(
                memory.probe(probe), reference.probe(probe)
            ),
            "max_abs_kda_state_delta": state_max_abs_diff(memory, reference),
        },
        "answer": answer_text(memory, tokenizer, victim_question),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--victim", type=int, default=1)
    ap.add_argument(
        "--filler",
        type=int,
        nargs="+",
        default=[0, 512, 2048],
        help="filler token counts appended after the records (the 'wait' arm)",
    )
    ap.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.99, 0.9, 0.5, 0.1, 0.0],
        help="retention factors for the direct-decay arm",
    )
    ap.add_argument("--out", default="outputs/kimi_sv/decay_contrast.json")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    cases = CASES
    if not 0 <= args.victim < len(cases):
        raise SystemExit(f"--victim must be in [0, {len(cases)})")
    victim_key = cases[args.victim]["key"]

    model, tokenizer, label = load_model(args.model, tiny=args.tiny)
    print(f"loaded {label}", flush=True)

    preamble = preamble_record(tokenizer)
    records = case_records(tokenizer, cases)
    survivors = [r for r in records if r.key != victim_key]

    def build(with_victim: bool, n_filler: int) -> RecordMemory:
        seq: List[Record] = [preamble] + (records if with_victim else survivors)
        if n_filler:
            seq.append(filler_record(tokenizer, n_filler))
        mem = RecordMemory(model)
        mem.ingest_all(seq)
        return mem

    rows: List[Dict[str, Any]] = []

    for n_filler in args.filler:
        reference = build(with_victim=False, n_filler=n_filler)

        present = build(with_victim=True, n_filler=n_filler)
        rows.append(
            measure(
                "wait",
                present,
                reference,
                tokenizer,
                cases,
                victim_key,
                {"filler_tokens": n_filler, "context_tokens": present.token_offset},
            )
        )

        replayed = build(with_victim=True, n_filler=n_filler)
        t0 = time.perf_counter()
        report = replayed.delete(victim_key)
        replay_seconds = time.perf_counter() - t0
        rows.append(
            measure(
                "replay",
                replayed,
                reference,
                tokenizer,
                cases,
                victim_key,
                {
                    "filler_tokens": n_filler,
                    "context_tokens": replayed.token_offset,
                    "replayed_tokens": report.replayed_tokens,
                    "seconds": replay_seconds,
                },
            )
        )

    # Direct decay is swept at the shortest context, where the record is most
    # recent and therefore most favourable to attenuation.
    n_filler = min(args.filler)
    reference = build(with_victim=False, n_filler=n_filler)
    for gamma in args.gammas:
        decayed = build(with_victim=True, n_filler=n_filler)
        scale_recurrent_state(model, decayed.cache, gamma)
        rows.append(
            measure(
                "decay",
                decayed,
                reference,
                tokenizer,
                cases,
                victim_key,
                {"gamma": gamma, "filler_tokens": n_filler},
            )
        )

    result = {
        "model": label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "victim": victim_key,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    header = (
        f"{'arm':7s} {'detail':16s} {'victim rank':>11s} {'victim p':>9s} "
        f"{'nats':>8s} {'ret rank':>9s} {'ret p':>7s} {'logit res':>10s} "
        f"{'state res':>10s}"
    )
    print(f"\nvictim: {victim_key}\n{header}\n{'-' * len(header)}")
    for row in rows:
        if row["arm"] == "decay":
            detail = f"gamma={row['gamma']:.2f}"
        else:
            detail = f"filler={row['filler_tokens']}"
        print(
            f"{row['arm']:7s} {detail:16s}"
            f" {row['victim']['first_token_rank']:11d}"
            f" {row['victim']['first_token_probability']:9.4f}"
            f" {row['victim']['nats_vs_reference']:+8.3f}"
            f" {row['retained']['mean_first_token_rank']:9.2f}"
            f" {row['retained']['mean_first_token_probability']:7.4f}"
            f" {row['residual']['max_abs_logit_delta']:10.3e}"
            f" {row['residual']['max_abs_kda_state_delta']:10.3e}"
        )

    print("\nanswers to the victim's question:")
    for row in rows:
        detail = (
            f"gamma={row['gamma']:.2f}"
            if row["arm"] == "decay"
            else f"filler={row['filler_tokens']}"
        )
        print(f"  {row['arm']:7s} {detail:16s} {row['answer']!r}")
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Local-only aggregate whole-record replay deletion on MIMIC-IV-Ext-CDS.

Each record is ingested into one persistent context on a Kimi Linear hybrid.
For every admitted record we delete it by rolling the hybrid cache back to its
boundary and replaying the survivors, then compare against a context that never
ingested it.

Privacy discipline, matching ``gemma_sv/eval_mimic_whole_record.py``: the
credentialed source never leaves memory. No record text, field value,
identifier, or generation is printed or written -- the report holds only
aggregate metrics, and text generation is skipped entirely so source content is
never even materialized as a string. A substring audit runs before the write.

Example::

    MIMIC_EXT_CDS_DIR=/path/to/mimic-iv-ext-cds/1.0.2 \\
    HF_HUB_CACHE=/path/to/huggingface/cache \\
      .venv311/bin/python -m kimi_sv.eval_mimic_deletion --records 8
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mimic import MAX_FIRST_TOKEN_RANK, MIN_SIGNAL, assert_no_source_text, summary
from mimic import cds as cds_source
from mimic import notes as notes_source

from .probes import lift_nats, teacher_forced
from .protocol import (
    DEFAULT_MODEL,
    case_records,
    encode,
    load_model,
    preamble_record,
    score_all,
)
from .records import RecordMemory, max_abs_diff, state_max_abs_diff

SOURCES = {"cds": cds_source, "notes": notes_source}


def checkpoint_tradeoff(
    record_token_counts: List[int],
    measured_rows: List[Dict[str, Any]],
    checkpoint_bytes: int,
) -> Dict[str, Any]:
    """Estimate sparse-checkpoint storage/replay latency from measured replay.

    Replay token counts are exact for each spacing. Latency is a transparent
    linear fit to this run's measured ``(replayed_tokens, replay_seconds)``
    pairs; it is an estimate, not an additional timed model execution.
    """

    n_records = len(record_token_counts)
    if n_records == 0 or not measured_rows:
        return {}
    x = np.asarray(
        [float(row["replayed_tokens"]) for row in measured_rows],
        dtype=float,
    )
    y = np.asarray(
        [float(row["replay_seconds"]) for row in measured_rows],
        dtype=float,
    )
    if len(np.unique(x)) < 2:
        slope, intercept = 0.0, float(np.mean(y))
        r_squared = None
    else:
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        residual = float(np.sum((y - predicted) ** 2))
        total = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - residual / total if total > 0 else 1.0

    intervals = []
    value = 1
    while value < n_records:
        intervals.append(value)
        value *= 2
    intervals.append(n_records)

    rows = []
    for interval in intervals:
        boundaries = set(range(0, n_records + 1, interval))
        boundaries.add(n_records)
        replayed = []
        latency = []
        for victim in range(n_records):
            checkpoint = (victim // interval) * interval
            tokens = (
                sum(record_token_counts[checkpoint:])
                - record_token_counts[victim]
            )
            replayed.append(float(tokens))
            latency.append(max(0.0, float(slope * tokens + intercept)))
        rows.append(
            {
                "interval_records": interval,
                "checkpoint_count": len(boundaries),
                "storage_bytes": len(boundaries) * checkpoint_bytes,
                "storage_gib": (
                    len(boundaries) * checkpoint_bytes / (1024 ** 3)
                ),
                "replayed_tokens": summary(replayed),
                "estimated_replay_seconds": summary(latency),
            }
        )

    dense_expected = {
        int(row["victim_position"]): float(row["replayed_tokens"])
        for row in measured_rows
    }
    dense_deviation = max(
        abs(
            sum(record_token_counts[position + 1 :])
            - expected
        )
        for position, expected in dense_expected.items()
    )
    return {
        "latency_model": {
            "kind": "linear fit to measured replay timings",
            "seconds_per_token": float(slope),
            "intercept_seconds": float(intercept),
            "r_squared": r_squared,
        },
        "dense_token_count_max_deviation": dense_deviation,
        "spacings": rows,
    }


def field_stats(memory: RecordMemory, tokenizer: Any, case: Dict[str, Any]) -> List:
    """Teacher-force each field value from its own stem.

    Scoring the whole record body would average the distinctive values together
    with the shared ``chief complaint:`` scaffolding, which any model can predict
    without holding the record. The stem puts the model exactly where the value
    is due.
    """
    return [
        teacher_forced(
            memory,
            encode(tokenizer, "\n" + case["question"] + " " + field["stem"]),
            encode(tokenizer, field["secret"]),
        )
        for field in case["fields"]
    ]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        choices=sorted(SOURCES),
        default="cds",
        help="cds = tabular initial assessments; notes = discharge summaries",
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--records", type=int, default=8)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument(
        "--body-chars",
        type=int,
        default=notes_source.DEFAULT_BODY_CHARS,
        help="per-note character budget (notes source only)",
    )
    ap.add_argument(
        "--victims",
        type=int,
        default=0,
        help=(
            "how many records to delete in turn, spaced evenly across the "
            "context so replay cost is sampled at every position (0 = every record)"
        ),
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="outputs/kimi_sv/mimic_deletion_aggregate.json")
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args(argv)

    loader = SOURCES[args.source]
    env_var = "MIMIC_EXT_CDS_DIR" if args.source == "cds" else "MIMIC_NOTE_DIR"
    data_dir = args.data_dir or os.getenv(env_var)
    if not data_dir:
        ap.error(f"--data-dir or {env_var} is required")
    source = loader.source_path(data_dir)
    if not source.is_file():
        ap.error(f"expected local source file: {source}")
    out_path = Path(args.out).resolve()
    if source.parent in out_path.parents:
        ap.error("aggregate output must not be written inside the MIMIC directory")

    if args.source == "notes":
        cases, source_stats = loader.load_cases(
            data_dir, args.records, skip=args.skip, body_chars=args.body_chars
        )
    else:
        cases, source_stats = loader.load_cases(
            data_dir, args.records, skip=args.skip
        )
    n_victims = args.victims or len(cases)
    if not 0 < n_victims <= len(cases):
        ap.error(f"--victims must be in (0, {len(cases)}]")

    # Evenly spaced positions, always including the first and last record, so the
    # replay-cost curve spans the whole context instead of clustering at the head.
    if n_victims == len(cases):
        victim_positions = list(range(len(cases)))
    elif n_victims == 1:
        victim_positions = [0]
    else:
        step = (len(cases) - 1) / (n_victims - 1)
        victim_positions = sorted({round(i * step) for i in range(n_victims)})

    requires_rank = getattr(loader, "ADMISSION_REQUIRES_RANK", True)

    model, tokenizer, label = load_model(args.model, tiny=args.tiny)
    print(f"loaded {label}", flush=True)

    preamble = preamble_record(tokenizer)
    records = case_records(tokenizer, cases)
    context_tokens = preamble.n_tokens + sum(r.n_tokens for r in records)
    print(f"{len(records)} records, {context_tokens} context tokens", flush=True)

    rows: List[Dict[str, Any]] = []
    for victim_index in victim_positions:
        victim_key = cases[victim_index]["key"]
        survivors = [r for r in records if r.key != victim_key]

        present = RecordMemory(model)
        present.ingest_all([preamble] + records)

        reference = RecordMemory(model)
        t0 = time.perf_counter()
        reference.ingest_all([preamble] + survivors)
        rebuild_seconds = time.perf_counter() - t0

        present_stats = score_all(present, tokenizer, cases)
        never_stats = score_all(reference, tokenizer, cases)
        victim_case = cases[victim_index]
        present_fields = field_stats(present, tokenizer, victim_case)
        never_fields = field_stats(reference, tokenizer, victim_case)

        victim_present = present_stats[victim_key]
        victim_never = never_stats[victim_key]
        present_lift = lift_nats(victim_present, victim_never)

        # A field is recallable when holding the record raises its log
        # probability, and -- for short categorical targets only -- also puts its
        # first token near the top of the distribution. Each source declares
        # whether the rank gate applies to its target shape.
        field_admitted = [
            lift_nats(p, n) >= MIN_SIGNAL
            and (not requires_rank or p.first_token_rank <= MAX_FIRST_TOKEN_RANK)
            for p, n in zip(present_fields, never_fields)
        ]
        admitted = present_lift >= MIN_SIGNAL and any(field_admitted)

        report = present.delete(victim_key)
        deleted_stats = score_all(present, tokenizer, cases)
        deleted_fields = field_stats(present, tokenizer, victim_case)

        kept = [i for i, ok in enumerate(field_admitted) if ok]
        # Field metrics are reported for every field, admitted or not: "we
        # measured +0.02 nats, below the 0.05 threshold" is a result, whereas a
        # null is only an absence of one. Admission still gates the aggregates.
        measured = list(range(len(present_fields)))

        probe = encode(tokenizer, "\n" + cases[victim_index]["question"])
        logit_residual = max_abs_diff(present.probe(probe), reference.probe(probe))
        state_residual = state_max_abs_diff(present, reference)

        retained_keys = [c["key"] for c in cases if c["key"] != victim_key]
        rows.append(
            {
                "victim_position": victim_index,
                "admitted": admitted,
                "fields_recallable": len(kept),
                "body_lift_present": present_lift,
                "body_lift_after_deletion": lift_nats(
                    deleted_stats[victim_key], victim_never
                ),
                "field_lift_present": (
                    statistics.mean(
                        lift_nats(present_fields[i], never_fields[i]) for i in measured
                    )
                ),
                "field_max_abs_lift_after_deletion": (
                    max(
                        abs(lift_nats(deleted_fields[i], never_fields[i])) for i in measured
                    )
                ),
                "field_worst_rank_present": max(
                    present_fields[i].first_token_rank for i in measured
                ),
                "field_worst_rank_after_deletion": max(
                    deleted_fields[i].first_token_rank for i in measured
                ),
                "field_best_rank_after_deletion": min(
                    deleted_fields[i].first_token_rank for i in measured
                ),
                "field_probability_present": statistics.mean(
                    present_fields[i].first_token_probability for i in measured
                ),
                "field_probability_after_deletion": statistics.mean(
                    deleted_fields[i].first_token_probability for i in measured
                ),
                "retained_max_abs_lift_after_deletion": max(
                    abs(lift_nats(deleted_stats[k], never_stats[k]))
                    for k in retained_keys
                ),
                "retained_worst_rank_present": max(
                    present_stats[k].first_token_rank for k in retained_keys
                ),
                "retained_worst_rank_after_deletion": max(
                    deleted_stats[k].first_token_rank for k in retained_keys
                ),
                "logit_residual": logit_residual,
                "state_residual": state_residual,
                "replayed_records": len(report.replayed_records),
                "replayed_tokens": report.replayed_tokens,
                "replay_seconds": report.seconds,
                "rebuild_seconds": rebuild_seconds,
                "checkpoint_bytes": report.checkpoint_bytes,
            }
        )
        row = rows[-1]
        print(
            f"  position {victim_index:3d}: admitted={admitted}"
            f" fields={len(kept)}/{len(victim_case['fields'])}"
            f" | lift {row['field_lift_present']:+.3f} nats"
            f" rank {row['field_worst_rank_present']}"
            f" -> {row['field_best_rank_after_deletion']}"
            f" | residual {logit_residual:.3e}/{state_residual:.3e}"
            f" | replay {report.seconds:.2f}s vs rebuild {rebuild_seconds:.2f}s",
            flush=True,
        )

    admitted_rows = [r for r in rows if r["admitted"]]
    exact = [
        r
        for r in admitted_rows
        if r["logit_residual"] == 0.0 and r["state_residual"] == 0.0
    ]

    result = {
        "evaluation": (
            f"local aggregate MIMIC whole-record replay deletion on a Kimi "
            f"Linear hybrid; source={args.source}"
        ),
        "source": args.source,
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "generations_produced": False,
        "model": label,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "source_rows": source_stats,
        "selection_policy": {
            "description": loader.SELECTION_POLICY,
            "records": len(cases),
            "skip": args.skip,
            "fixed_before_evaluation": True,
        },
        "admission": {
            "min_signal_nats": MIN_SIGNAL,
            "max_first_token_rank": MAX_FIRST_TOKEN_RANK if requires_rank else None,
            "rank_gate_applied": requires_rank,
            "rank_gate_rationale": (
                "first-token rank gates short categorical targets, where a "
                "recalled value should be the argmax; it is not applied to "
                "verbatim prose continuation, which has many locally plausible "
                "next tokens. Ranks are reported at every position regardless."
            ),
            "attempted": len(rows),
            "admitted": len(admitted_rows),
            "admission_rate": len(admitted_rows) / len(rows) if rows else None,
        },
        "context": {"records": len(records), "tokens": context_tokens},
        "exactness": {
            "admitted_with_zero_residual": len(exact),
            "logit_residual": summary([r["logit_residual"] for r in admitted_rows]),
            "state_residual": summary([r["state_residual"] for r in admitted_rows]),
        },
        "efficacy": {
            "note": (
                "field metrics score each value from its own stem, isolating it "
                "from shared record formatting; body metrics score the whole "
                "record body"
            ),
            "recallable_fields_per_record": summary(
                [float(r["fields_recallable"]) for r in admitted_rows]
            ),
            "body_lift_present_nats": summary(
                [r["body_lift_present"] for r in admitted_rows]
            ),
            "body_lift_after_deletion_nats": summary(
                [r["body_lift_after_deletion"] for r in admitted_rows]
            ),
            "field_lift_present_nats": summary(
                [r["field_lift_present"] for r in admitted_rows]
            ),
            "field_max_abs_lift_after_deletion_nats": summary(
                [r["field_max_abs_lift_after_deletion"] for r in admitted_rows]
            ),
            "field_worst_rank_present": summary(
                [float(r["field_worst_rank_present"]) for r in admitted_rows]
            ),
            "field_worst_rank_after_deletion": summary(
                [float(r["field_worst_rank_after_deletion"]) for r in admitted_rows]
            ),
            "field_best_rank_after_deletion": summary(
                [float(r["field_best_rank_after_deletion"]) for r in admitted_rows]
            ),
            "field_probability_present": summary(
                [r["field_probability_present"] for r in admitted_rows]
            ),
            "field_probability_after_deletion": summary(
                [r["field_probability_after_deletion"] for r in admitted_rows]
            ),
        },
        "diagnostics": {
            "note": (
                "measured over every victim regardless of admission, so the "
                "recall frontier is visible: verbatim recall of long records "
                "falls off with distance from the end of the context, while "
                "deletion exactness does not"
            ),
            "field_lift_present_nats_all_positions": summary(
                [r["field_lift_present"] for r in rows]
            ),
            "field_worst_rank_present_all_positions": summary(
                [float(r["field_worst_rank_present"]) for r in rows]
            ),
            "logit_residual_all_positions": summary(
                [r["logit_residual"] for r in rows]
            ),
            "state_residual_all_positions": summary(
                [r["state_residual"] for r in rows]
            ),
        },
        "specificity": {
            "retained_max_abs_lift_after_deletion_nats": summary(
                [r["retained_max_abs_lift_after_deletion"] for r in admitted_rows]
            ),
            "retained_worst_rank_present": summary(
                [float(r["retained_worst_rank_present"]) for r in admitted_rows]
            ),
            "retained_worst_rank_after_deletion": summary(
                [float(r["retained_worst_rank_after_deletion"]) for r in admitted_rows]
            ),
        },
        "cost": {
            "replay_seconds": summary([r["replay_seconds"] for r in rows]),
            "rebuild_seconds": summary([r["rebuild_seconds"] for r in rows]),
            "replayed_tokens": summary([float(r["replayed_tokens"]) for r in rows]),
            "checkpoint_bytes_per_boundary": rows[0]["checkpoint_bytes"] if rows else None,
        },
        "checkpoint_tradeoff": checkpoint_tradeoff(
            [record.n_tokens for record in records],
            rows,
            rows[0]["checkpoint_bytes"] if rows else 0,
        ),
        "per_position": [
            {
                k: v
                for k, v in row.items()
                if k
                in {
                    "victim_position",
                    "admitted",
                    "field_lift_present",
                    "field_worst_rank_present",
                    "field_best_rank_after_deletion",
                    "logit_residual",
                    "state_residual",
                    "replayed_tokens",
                    "replay_seconds",
                    "rebuild_seconds",
                }
            }
            for row in rows
        ],
    }

    payload = json.dumps(result, indent=2)
    assert_no_source_text(payload, cases)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload)

    adm = result["admission"]
    ex = result["exactness"]
    eff = result["efficacy"]
    spec = result["specificity"]
    print(
        f"\nadmitted {adm['admitted']}/{adm['attempted']}"
        f" (rate {adm['admission_rate']:.2f})"
    )
    if admitted_rows:
        print(
            f"exact deletions (zero logit and state residual):"
            f" {ex['admitted_with_zero_residual']}/{adm['admitted']}"
        )
        print(
            f"field lift {eff['field_lift_present_nats']['mean']:+.3f}"
            f" -> {eff['field_max_abs_lift_after_deletion_nats']['max']:+.3e}"
            f" nats (worst after deletion)"
        )
        print(
            f"field first-token p {eff['field_probability_present']['mean']:.3f}"
            f" -> {eff['field_probability_after_deletion']['mean']:.3f} (mean);"
            f" best-case rank after deletion"
            f" {eff['field_best_rank_after_deletion']['min']:.0f}"
        )
        print(
            f"retained worst rank {spec['retained_worst_rank_present']['max']:.0f}"
            f" -> {spec['retained_worst_rank_after_deletion']['max']:.0f};"
            f" worst retained drift"
            f" {spec['retained_max_abs_lift_after_deletion_nats']['max']:.3e} nats"
        )
    print(
        f"replay {result['cost']['replay_seconds']['mean']:.2f}s mean"
        f" vs rebuild {result['cost']['rebuild_seconds']['mean']:.2f}s mean"
    )
    print(f"aggregate report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

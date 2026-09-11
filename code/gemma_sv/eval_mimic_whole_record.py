"""Local-only aggregate whole-record validation on MIMIC-IV-Ext-CDS.

The credentialed source never leaves memory: this command writes no questions,
answers, identifiers, generations, or per-record rows.  Its JSON contains only
aggregate admission/deletion/retention metrics and a rejection taxonomy.

Example:
    MIMIC_EXT_CDS_DIR=/path/to/mimic-iv-ext-cds/1.0.2 \
      python -m gemma_sv.eval_mimic_whole_record --records 8 --samples 16
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace

from gemma_sv.demo_server.gemma_engine import GemmaRuntime, RuntimeConfig
from gemma_sv.eval_robust_unlearning import (
    CONDITIONS,
    _parse_conditions,
    _parse_k_values,
)
from gemma_sv.eval_whole_record_unlearning import evaluate
from gemma_sv.robust_eval import expected_max_at_k


SOURCE_FILE = "initial_assessment_info.csv"
REQUIRED_COLUMNS = (
    "chiefcomplaint",
    "arrival_transport",
    "disposition",
)
MAX_FIELD_CHARS = 160


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value[:MAX_FIELD_CHARS].rstrip(" ,;")


def _usable(row: dict[str, str]) -> bool:
    return all(
        (value := _clean(row.get(column, "")))
        and value.casefold() not in {"nan", "none", "[]", "unknown"}
        for column in REQUIRED_COLUMNS
    )


def build_private_manifest(
    source: Path,
    *,
    records: int,
    skip: int,
) -> tuple[dict, dict[str, int]]:
    """Build an in-memory manifest without retaining source identifiers."""

    selected: list[dict[str, str]] = []
    scanned = complete = 0
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"MIMIC source is missing required columns: {missing}")
        for row in reader:
            scanned += 1
            if not _usable(row):
                continue
            complete += 1
            if complete <= skip:
                continue
            selected.append({column: _clean(row[column]) for column in REQUIRED_COLUMNS})
            if len(selected) >= records * 2:
                break
    if len(selected) < records * 2:
        raise ValueError(
            f"only {len(selected)} complete local rows found; need {records * 2}"
        )

    manifest_records = []
    for index in range(records):
        forget = selected[2 * index]
        retain = selected[2 * index + 1]
        manifest_records.append(
            {
                "record_id": f"mimic-local-{index:03d}",
                "question": (
                    f"What is the complete indexed clinical record {index:03d}?"
                ),
                "answer": (
                    f"chief complaint: {forget['chiefcomplaint']}; "
                    f"arrival transport: {forget['arrival_transport']}; "
                    f"disposition: {forget['disposition']}."
                ),
                "fields": [
                    {
                        "name": "chief complaint",
                        "value": forget["chiefcomplaint"],
                    },
                    {
                        "name": "arrival transport",
                        "value": forget["arrival_transport"],
                    },
                    {
                        "name": "disposition",
                        "value": forget["disposition"],
                    },
                ],
                "retain_question": (
                    f"What is the complete indexed neighboring record {index:03d}?"
                ),
                "retain_answer": (
                    f"chief complaint: {retain['chiefcomplaint']}; "
                    f"arrival transport: {retain['arrival_transport']}; "
                    f"disposition: {retain['disposition']}."
                ),
                "retain_field": {
                    "name": "disposition",
                    "value": retain["disposition"],
                },
            }
        )
    return {
        "name": "mimic_iv_ext_cds_local_v1",
        "version": 1,
        "selection_policy": {
            "description": (
                "first 2N complete diagnosis.csv rows after deterministic skip; "
                "paired in file order; raw identifiers never retained"
            ),
            "fixed_before_evaluation": True,
            "copies": 2,
        },
        "records": manifest_records,
    }, {"rows_scanned": scanned, "complete_rows_seen": complete}


def _percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def _record_bootstrap(
    records: list[dict],
    conditions,
    k_values,
    *,
    replicates: int = 2000,
    seed: int = 0,
) -> dict:
    """Record-level (cluster) bootstrap CIs for the aggregate metrics.

    Fields within a record share context, so records are the resampling unit.
    Purely a reporting addition: the evaluation protocol is untouched.
    """

    rng = random.Random(seed)
    n = len(records)
    leak: dict[str, dict[str, list[float]]] = {
        condition: {str(k): [] for k in k_values} for condition in conditions
    }
    log_prob: dict[str, list[float]] = {condition: [] for condition in conditions}
    retain_shift: list[float] = []
    for _ in range(replicates):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        shifts = []
        for condition in conditions:
            scores = [
                field["exact_phrase"]
                for row in sample
                for field in row["conditions"][condition]["fields"]
                if "exact_phrase" in field
            ]
            probes = [
                field["secret_probe"]["mean_log_probability"]
                for row in sample
                for field in row["conditions"][condition]["fields"]
            ]
            log_prob[condition].append(sum(probes) / len(probes))
            for k in k_values:
                valid = [s for s in scores if len(s) >= k]
                if valid:
                    leak[condition][str(k)].append(
                        sum(expected_max_at_k(s, k) for s in valid) / len(valid)
                    )
        for row in sample:
            conditions_row = row["conditions"]
            if "present" in conditions_row and "decrement" in conditions_row:
                shifts.append(
                    conditions_row["decrement"]["retain_secret_probe"][
                        "mean_log_probability"
                    ]
                    - conditions_row["present"]["retain_secret_probe"][
                        "mean_log_probability"
                    ]
                )
        if shifts:
            retain_shift.append(sum(shifts) / len(shifts))

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return [_percentile(ordered, 0.025), _percentile(ordered, 0.975)]

    return {
        "method": f"record-level bootstrap, {replicates} replicates, seed {seed}",
        "exact_phrase_leak_at_k_ci95": {
            condition: {
                k: interval(values) for k, values in per_k.items()
            }
            for condition, per_k in leak.items()
        },
        "mean_condition_secret_log_probability_ci95": {
            condition: interval(values) for condition, values in log_prob.items()
        },
        "mean_retain_shift_ci95": interval(retain_shift),
    }


def _aggregate_only(result: dict, source_stats: dict[str, int]) -> dict:
    whole = result["whole_record"]
    rejections = Counter(
        reason
        for row in whole["rejected_records"]
        for reason in row["reasons"]
    )
    residuals: dict[str, list[float]] = {
        condition: [] for condition in result["sampling"]["conditions"]
    }
    retain_shifts: list[float] = []
    for row in whole["records"]:
        conditions = row["conditions"]
        for condition in residuals:
            for field in conditions[condition]["fields"]:
                residuals[condition].append(
                    field["secret_probe"]["mean_log_probability"]
                )
        if "present" in conditions and "decrement" in conditions:
            retain_shifts.append(
                conditions["decrement"]["retain_secret_probe"][
                    "mean_log_probability"
                ]
                - conditions["present"]["retain_secret_probe"][
                    "mean_log_probability"
                ]
            )
    return {
        "evaluation": "local aggregate MIMIC-IV-Ext-CDS whole-record deletion",
        "contains_source_text": False,
        "contains_source_identifiers": False,
        "source_rows": source_stats,
        "sampling": result["sampling"],
        "attempted": whole["attempted"],
        "admitted": whole["admitted"],
        "admission_rate": whole["admission_rate"],
        "rejection_counts": dict(sorted(rejections.items())),
        "exact_phrase_leak_at_k": whole["exact_phrase_leak_at_k"],
        "composite_any_leak_at_k": whole["composite_any_leak_at_k"],
        "mean_condition_secret_log_probability": {
            condition: (
                sum(values) / len(values) if values else None
            )
            for condition, values in residuals.items()
        },
        "mean_retain_shift_decrement_minus_present": (
            sum(retain_shifts) / len(retain_shifts)
            if retain_shifts
            else None
        ),
        "bootstrap": _record_bootstrap(
            whole["records"],
            result["sampling"]["conditions"],
            result["sampling"]["k"],
            seed=result["sampling"].get("seed", 0),
        ),
    }


def main(argv=None) -> int:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.getenv("MIMIC_EXT_CDS_DIR"),
        help="local credentialed MIMIC-IV-Ext-CDS 1.0.2 directory",
    )
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--k", type=_parse_k_values, default="1,2,4,8,16")
    parser.add_argument(
        "--conditions",
        type=_parse_conditions,
        default="present,decrement,icul,never",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model", default="google/gemma-3-1b-pt")
    parser.add_argument("--lora", default="outputs/gemma_sv_distill/lora_adapter")
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_eval/mimic_whole_record_aggregate.json",
    )
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("--data-dir or MIMIC_EXT_CDS_DIR is required")
    if args.records < 1 or args.skip < 0:
        parser.error("--records must be positive and --skip non-negative")
    args.conditions = (
        _parse_conditions(args.conditions)
        if isinstance(args.conditions, str)
        else tuple(args.conditions)
    )
    args.k_values = (
        _parse_k_values(args.k) if isinstance(args.k, str) else list(args.k)
    )
    if max(args.k_values) > args.samples:
        parser.error("largest k must not exceed --samples")
    source = Path(args.data_dir).resolve() / SOURCE_FILE
    if not source.is_file():
        parser.error(f"expected local source file: {source}")
    output = Path(args.out).resolve()
    if source.parent in output.parents:
        parser.error("aggregate output must not be written inside the MIMIC directory")

    manifest, source_stats = build_private_manifest(
        source,
        records=args.records,
        skip=args.skip,
    )
    runtime = GemmaRuntime(
        RuntimeConfig(
            model_id=args.model,
            lora_path=args.lora or None,
            device=args.device,
            dtype="float32",
            generation_tokens=args.max_new_tokens,
            window=512,
        )
    )
    runtime.ensure_loaded()
    retain = list(load_dataset("locuslab/TOFU", "retain90", split="train"))
    fillers = [str(row["answer"]) for row in retain[:32]]
    eval_args = SimpleNamespace(
        record_start=0,
        records=args.records,
        window=512,
        n_fill=22,
        prefix_fillers=8,
        conditions=args.conditions,
        samples=args.samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        seed=args.seed,
        kv_cache=True,
        save_generations=False,
        k_values=args.k_values,
    )
    raw_result = {
        "sampling": {
            "samples": args.samples,
            "k": args.k_values,
            "seed": args.seed,
            "conditions": list(args.conditions),
            "kv_cache": True,
        },
        "whole_record": evaluate(
            runtime,
            manifest,
            [],
            [],
            fillers,
            eval_args,
        ),
    }
    aggregate = _aggregate_only(raw_result, source_stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(
        f"wrote aggregate-only report: attempted={aggregate['attempted']} "
        f"admitted={aggregate['admitted']} -> {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

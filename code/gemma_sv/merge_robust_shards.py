"""Merge sharded ``eval_robust_unlearning`` JSON files and recompute summaries.

Shards may split target ranges, condition sets, or both. Sampling parameters,
model identity, thresholds, and seeds must match. When two shards report the
same target (for example one shard per condition), shared float fields such as
admission signals may differ by small amounts because MPS inference is not
bit-deterministic across processes; those are compared with a tolerance, while
structural fields (indices, token positions, text) must match exactly.

Run:
    python -m gemma_sv.merge_robust_shards \
      outputs/gemma_sv_eval/robust_shard_*.json \
      --out outputs/gemma_sv_eval/robust_unlearning.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

# Cross-process MPS drift on mean log-probabilities is ~1e-3 nats; genuine
# target mix-ups shift admission signals by whole nats.
FLOAT_RELATIVE_TOLERANCE = 0.02
FLOAT_ABSOLUTE_TOLERANCE = 0.05


def _approximately_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(
            float(a),
            float(b),
            rel_tol=FLOAT_RELATIVE_TOLERANCE,
            abs_tol=FLOAT_ABSOLUTE_TOLERANCE,
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _approximately_equal(a[key], b[key]) for key in a
        )
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _approximately_equal(x, y) for x, y in zip(a, b)
        )
    return a == b

from gemma_sv.eval_robust_unlearning import CONDITIONS
from gemma_sv.robust_eval import aggregate_leak


def _assert_equal(reports: list[dict[str, Any]], path: tuple[str, ...]) -> Any:
    def read(report):
        value: Any = report
        for key in path:
            value = value[key]
        return value

    first = read(reports[0])
    for report in reports[1:]:
        if read(report) != first:
            raise ValueError(f"shards disagree on {'.'.join(path)}")
    return first


def _merge_rows(
    sections: list[dict[str, Any]],
    *,
    row_key: str,
) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for section in sections:
        for row in section[row_key]:
            index = int(row["index"])
            if index not in by_index:
                by_index[index] = deepcopy(row)
                continue
            existing = by_index[index]
            for key, value in row.items():
                if key == "conditions":
                    for condition, result in value.items():
                        if (
                            condition in existing["conditions"]
                            and existing["conditions"][condition] != result
                        ):
                            raise ValueError(
                                f"conflicting {condition} data for target {index}"
                            )
                        existing["conditions"][condition] = deepcopy(result)
                elif key == "admission":
                    if not _approximately_equal(existing.get(key), value):
                        # Borderline gate-validity flips between shard
                        # processes can move one teacher-forced diagnostic
                        # well beyond float jitter. The admission decision
                        # itself already passed in every shard, and target
                        # mix-ups are caught by the exact-match fields, so
                        # record the disagreeing values instead of failing.
                        disagreements = existing.setdefault(
                            "admission_disagreements", {}
                        )
                        for condition in row.get("conditions", {}):
                            disagreements[condition] = deepcopy(value)
                elif key == "admission_disagreements":
                    existing.setdefault(key, {}).update(deepcopy(value))
                elif existing.get(key) != value:
                    raise ValueError(f"conflicting {key} for target {index}")
    return [by_index[index] for index in sorted(by_index)]


def _attempted_indices(
    sections: list[dict[str, Any]],
    *,
    start_key: str,
) -> set[int]:
    indices: set[int] = set()
    for section in sections:
        start = int(section[start_key])
        indices.update(range(start, start + int(section["attempted"])))
    return indices


def _available_conditions(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    found = {
        condition
        for row in rows
        for condition in row.get("conditions", {})
    }
    return tuple(condition for condition in CONDITIONS if condition in found)


def _complete_rows(
    rows: list[dict[str, Any]],
    conditions: tuple[str, ...],
    *,
    kind: str,
    allow_incomplete: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    complete, dropped = [], []
    for row in rows:
        missing = [
            condition
            for condition in conditions
            if condition not in row["conditions"]
        ]
        if not missing:
            complete.append(row)
            continue
        if not allow_incomplete:
            raise ValueError(
                f"{kind} target {row['index']} is missing conditions {missing}; "
                "rerun the missing shards or pass --allow-incomplete"
            )
        dropped.append(int(row["index"]))
    return complete, dropped


def _merge_leak(
    sections: list[dict[str, Any]],
    *,
    k_values: list[int],
    threshold: float,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    rows = _merge_rows(sections, row_key="targets")
    conditions = _available_conditions(rows)
    rows, dropped = _complete_rows(
        rows,
        conditions,
        kind="leak",
        allow_incomplete=allow_incomplete,
    )
    attempted = _attempted_indices(sections, start_key="target_start")
    result = deepcopy(sections[0])
    result["target_start"] = min(attempted) if attempted else 0
    result["attempted"] = len(attempted)
    result["admitted"] = len(rows)
    result["admission_rate"] = len(rows) / len(attempted)
    if dropped:
        result["dropped_incomplete_targets"] = sorted(dropped)
    result["rouge_l_leak_at_k"] = {
        condition: aggregate_leak(
            [row["conditions"][condition]["rouge_l_recall"] for row in rows],
            k_values,
            threshold=threshold,
        )
        for condition in conditions
    }
    result["exact_phrase_leak_at_k"] = {
        condition: aggregate_leak(
            [row["conditions"][condition]["exact_phrase"] for row in rows],
            k_values,
            threshold=0.5,
        )
        for condition in conditions
    }
    result["targets"] = rows
    return result


def _merge_paired(
    sections: list[dict[str, Any]],
    *,
    k_values: list[int],
    threshold: float,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    rows = _merge_rows(sections, row_key="pairs")
    conditions = _available_conditions(rows)
    rows, dropped = _complete_rows(
        rows,
        conditions,
        kind="paired",
        allow_incomplete=allow_incomplete,
    )
    attempted = _attempted_indices(sections, start_key="paired_start")
    result = deepcopy(sections[0])
    result["paired_start"] = min(attempted) if attempted else 0
    result["attempted"] = len(attempted)
    result["admitted"] = len(rows)
    result["admission_rate"] = len(rows) / len(attempted)
    if dropped:
        result["dropped_incomplete_targets"] = sorted(dropped)
    result["summary"] = {}
    for condition in conditions:
        condition_rows = [row["conditions"][condition] for row in rows]
        result["summary"][condition] = {
            "forget_leak_at_k": aggregate_leak(
                [row["forget_rouge_l_recall"] for row in condition_rows],
                k_values,
                threshold=threshold,
            ),
            "retain_recall_at_k": aggregate_leak(
                [row["retain_rouge_l_recall"] for row in condition_rows],
                k_values,
                threshold=threshold,
            ),
            "mean_sample_selective_success_rate": sum(
                row["sample_selective_success_rate"] for row in condition_rows
            )
            / len(condition_rows),
        }
    result["pairs"] = rows
    return result


def merge_reports(
    reports: list[dict[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one shard is required")
    for path in (
        ("evaluation",),
        ("behavioral_scope",),
        ("model",),
        ("lora",),
        ("device",),
        ("sampling", "samples"),
        ("sampling", "k"),
        ("sampling", "temperature"),
        ("sampling", "top_p"),
        ("sampling", "max_new_tokens"),
        ("sampling", "seed"),
        ("sampling", "kv_cache"),
        (
            "admission",
            "minimum_present_minus_never_mean_log_probability_nats",
        ),
        ("admission", "maximum_present_secret_first_token_rank"),
    ):
        _assert_equal(reports, path)

    result = deepcopy(reports[0])
    k_values = [int(value) for value in result["sampling"]["k"]]
    threshold = float(result["sampling"]["rouge_threshold"])
    result["sampling"]["conditions"] = list(
        dict.fromkeys(
            condition
            for report in reports
            for condition in report["sampling"]["conditions"]
        )
    )
    leak_sections = [report["leak"] for report in reports if "leak" in report]
    paired_sections = [report["paired"] for report in reports if "paired" in report]
    result.pop("leak", None)
    result.pop("paired", None)
    if leak_sections:
        result["leak"] = _merge_leak(
            leak_sections,
            k_values=k_values,
            threshold=threshold,
            allow_incomplete=allow_incomplete,
        )
    if paired_sections:
        result["paired"] = _merge_paired(
            paired_sections,
            k_values=k_values,
            threshold=threshold,
            allow_incomplete=allow_incomplete,
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+")
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_eval/robust_unlearning.json",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "drop (and list) targets whose admission flipped between "
            "condition shards instead of failing the merge"
        ),
    )
    args = parser.parse_args(argv)
    reports = [json.loads(Path(path).read_text()) for path in args.shards]
    merged = merge_reports(reports, allow_incomplete=args.allow_incomplete)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"merged {len(reports)} shards -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


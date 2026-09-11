"""Source-block sensitivity and paired-score reporting for the journal bridge.

First use --freeze-only before the main run. Later use --plan, --results, --out.
This analyzer never changes the experimental runner or existing result files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ADDENDUM = Path(__file__).with_name("analysis_addendum.json")
ANALYZER = Path(__file__).resolve()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def group_manifest(manifest, blocks):
    records = manifest.get("records", [])
    if not records:
        raise ValueError("empty manifest")
    result = {}
    for record in records:
        record_id = str(record["record_id"])
        if record_id in result:
            raise ValueError("duplicate manifest record ID")
        indices = [*record["forget_indices"], record["retain_index"]]
        if len(record["forget_indices"]) != 3 or len(set(indices)) != 4:
            raise ValueError("expected three distinct deleted rows and a separate retained row")
        matching = [block for block in blocks if all(block <= int(index) < block + 20 for index in indices)]
        if len(matching) != 1:
            raise ValueError("record does not belong to exactly one declared source block")
        result[record_id] = matching[0]
    if Counter(result.values()) != Counter({block: 2 for block in blocks}):
        raise ValueError("expected exactly two records in each declared source block")
    return result


def index_results(report, expected_ids):
    if not report.get("records"):
        raise ValueError("empty results")
    selected = report.get("selected_record_ids", [])
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("empty or duplicate result selection")
    if set(selected) - set(expected_ids):
        raise ValueError("unknown selected record ID")
    result = {}
    for row in report["records"]:
        record_id = str(row["record_id"])
        if record_id in result:
            raise ValueError("duplicate result record ID")
        if record_id not in expected_ids or record_id not in selected:
            raise ValueError("unknown or unselected result record ID")
        result[record_id] = row
    return result


def finite_number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite score or timing")
    return value


def extract_scores(arm):
    metrics = arm["metrics"]
    fields = [*metrics["deleted_target_quality"]["fields"], metrics["retained_quality"]]
    expected = {"deleted_field_0", "deleted_field_1", "deleted_field_2", "retained_field"}
    found = [str(field["probe_id"]) for field in fields]
    if len(found) != len(set(found)) or set(found) != expected:
        raise ValueError("missing or duplicate target probes")
    return {field["probe_id"]: field["score"] for field in fields}


def arm_status(arm):
    if not arm:
        return "missing"
    if arm.get("status") != "completed":
        return str(arm.get("status", "missing_status"))
    if "metrics" not in arm or "timing" not in arm:
        return "missing_metrics"
    return "completed"


def paired_record(row, record_id, source_block, left_name, right_name):
    arms = (row or {}).get("arms", {})
    left, right = arms.get(left_name), arms.get(right_name)
    status_left, status_right = arm_status(left), arm_status(right)
    result = {"record_id": record_id, "source_block": source_block,
        "left_status": status_left, "right_status": status_right,
        "complete_pair": status_left == status_right == "completed"}
    if not result["complete_pair"]:
        return result
    left_scores, right_scores = extract_scores(left), extract_scores(right)
    result["probes"] = []
    for probe_id in ["deleted_field_0", "deleted_field_1", "deleted_field_2", "retained_field"]:
        a, b = left_scores[probe_id], right_scores[probe_id]
        if int(a["target_token_count"]) != int(b["target_token_count"]) or int(a["target_token_count"]) < 1:
            raise ValueError("paired target token counts differ or are empty")
        probe = {"probe_id": probe_id, "target_tokens": int(a["target_token_count"])}
        for metric in ("total_log_probability", "mean_log_probability"):
            a_value, b_value = finite_number(a[metric]), finite_number(b[metric])
            probe[metric] = {"left": a_value, "right": b_value, "left_minus_right": a_value - b_value}
        result["probes"].append(probe)
    result["deleted_mean_token_log_probability_delta"] = float(np.mean([p["mean_log_probability"]["left_minus_right"] for p in result["probes"][:3]]))
    result["retained_mean_token_log_probability_delta"] = result["probes"][3]["mean_log_probability"]["left_minus_right"]
    result["cost"] = {}
    for metric in ("update_seconds", "query_seconds", "end_to_end_seconds"):
        left_time = finite_number(left["timing"][metric]["median"])
        right_time = finite_number(right["timing"][metric]["median"])
        if left_time < 0 or right_time < 0:
            raise ValueError("negative timing")
        ratio = None
        if left_time > 0 and right_time > 0:
            ratio = left_time / right_time
        elif metric != "update_seconds" or not (left_name.endswith("present") or right_name.endswith("present")):
            raise ValueError("zero timing outside present-control update")
        result["cost"][metric] = {"left": left_time, "right": right_time,
            "left_minus_right": left_time - right_time, "left_over_right": ratio,
            "ratio_unavailable_reason": None if ratio is not None else "present control has no update operation"}
    result["left_full_repack_fallback"] = bool(left.get("full_repack_fallback", False))
    result["right_full_repack_fallback"] = bool(right.get("full_repack_fallback", False))
    return result


def bootstrap_sensitivity(values, groups, *, geometric=False, samples=10000, seed=0):
    """Resample records and then source blocks, retaining within-block pairs."""
    if not values or len(values) != len(groups):
        raise ValueError("empty or unpaired bootstrap inputs")
    values = np.asarray([finite_number(value) for value in values], dtype=np.float64)
    if geometric and np.any(values <= 0):
        raise ValueError("latency ratios must be positive")
    transformed = np.log(values) if geometric else values
    inverse = np.exp if geometric else lambda x: x
    labels = sorted(set(groups))
    rng = np.random.default_rng(seed)
    record_draws = rng.integers(0, len(values), size=(samples, len(values)))
    record_estimates = inverse(transformed[record_draws].mean(axis=1))
    cluster_rows = [transformed[np.asarray(groups) == label] for label in labels]
    sums = np.asarray([row.sum() for row in cluster_rows])
    counts = np.asarray([len(row) for row in cluster_rows])
    block_draws = rng.integers(0, len(labels), size=(samples, len(labels)))
    block_estimates = inverse(sums[block_draws].sum(axis=1) / counts[block_draws].sum(axis=1))
    return {"estimate": float(inverse(transformed.mean())), "records": len(values), "source_clusters": len(labels),
        "record_percentile_95": np.quantile(record_estimates, [.025, .975]).tolist() if len(values) > 1 else None,
        "source_block_percentile_95": np.quantile(block_estimates, [.025, .975]).tolist() if len(labels) > 1 else None,
        "samples": samples, "seed": seed,
        "interpretation": "Descriptive sensitivity only; four source clusters and their independence is not established."}


def comparison_summary(pairs, *, smoke, terminal=True, samples=10000, seed=0):
    available = [pair for pair in pairs if pair["complete_pair"]]
    complete = len(available) == len(pairs) and not smoke and terminal
    summary = {"expected_records": len(pairs), "complete_pairs": len(available),
        "complete_main_cohort": complete, "smoke_only": bool(smoke), "terminal_result": terminal,
        "unavailable_records": [{"record_id": p["record_id"], "left_status": p["left_status"], "right_status": p["right_status"]} for p in pairs if not p["complete_pair"]],
        "metrics": {}}
    specs = [("deleted_mean_token_log_probability_delta", lambda p: p["deleted_mean_token_log_probability_delta"], False),
             ("retained_mean_token_log_probability_delta", lambda p: p["retained_mean_token_log_probability_delta"], False)]
    for metric in ("update_seconds", "query_seconds", "end_to_end_seconds"):
        specs.append((metric + "_difference", lambda p, metric=metric: p["cost"][metric]["left_minus_right"], False))
        specs.append((metric + "_ratio", lambda p, metric=metric: p["cost"][metric]["left_over_right"], True))
    for name, getter, geometric in specs:
        values = [getter(p) for p in available]
        if not values or any(value is None for value in values):
            summary["metrics"][name] = {"full_cohort": None, "available_pairs": len(values), "reason": "no complete pairs or present-control update ratio is undefined"}
            continue
        partial_mean = float(np.exp(np.log(values).mean())) if geometric else float(np.mean(values))
        result = {"full_cohort": None, "available_pairs": len(values)}
        if complete:
            result["full_cohort"] = bootstrap_sensitivity(values, [p["source_block"] for p in available],
                geometric=geometric, samples=samples, seed=seed)
        else:
            result["partial_descriptive_mean_only"] = partial_mean
            result["reason"] = "incomplete frozen cohort, smoke, or running result; no full-cohort estimate or interval"
        summary["metrics"][name] = result
    return summary


def analyze(report, manifest, addendum):
    grouping = group_manifest(manifest, addendum["source_blocks"])
    records = index_results(report, grouping)
    comparisons = []
    for left, right in addendum["comparisons"]:
        pairs = [paired_record(records.get(record_id), record_id, group, left, right) for record_id, group in grouping.items()]
        comparisons.append({"left": left, "right": right, "records": pairs,
            "summary": comparison_summary(pairs, smoke=report.get("smoke", False),
                terminal=report.get("status") in ("completed", "completed_with_failures"),
                samples=addendum["bootstrap"]["samples"], seed=addendum["bootstrap"]["seed"])})
    return {"schema": addendum["schema"], "result_status": report.get("status"), "smoke_only": report.get("smoke", False),
        "manifest_records": len(grouping), "source_clusters": len(set(grouping.values())),
        "direction": addendum["direction"], "interpretation": addendum["bootstrap"]["interpretation"],
        "comparisons": comparisons}


def markdown_report(analysis):
    lines = ["# GemmaSV journal bridge: paired results", "", analysis["direction"], "", analysis["interpretation"], "",
        "Teacher-forced target probabilities are disclosure and retained-utility proxies; these results do not measure decoded leakage or exact-decrement speed.", ""]
    for comparison in analysis["comparisons"]:
        summary = comparison["summary"]
        lines += [f"## {comparison['left']} versus {comparison['right']}", "",
            f"Complete pairs: {summary['complete_pairs']}/{summary['expected_records']}. Full main cohort: {summary['complete_main_cohort']}.", "",
            "| Record | Source block | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | End-to-end ratio |",
            "|---|---:|---:|---:|---:|---:|"]
        for pair in comparison["records"]:
            if not pair["complete_pair"]:
                lines.append(f"| {pair['record_id']} | {pair['source_block']} | unavailable ({pair['left_status']}/{pair['right_status']}) | — | — | — |")
                continue
            update = pair["cost"]["update_seconds"]["left_over_right"]
            total = pair["cost"]["end_to_end_seconds"]["left_over_right"]
            update_text = "n/a" if update is None else f"{update:.3f}"
            lines.append(f"| {pair['record_id']} | {pair['source_block']} | {pair['deleted_mean_token_log_probability_delta']:.4f} | {pair['retained_mean_token_log_probability_delta']:.4f} | {update_text} | {total:.3f} |")
        lines += ["", "| Cohort measure | Estimate | Record bootstrap 95% | Source-block bootstrap 95% |",
            "|---|---:|---:|---:|"]
        for name, result in summary["metrics"].items():
            full = result["full_cohort"]
            if full:
                interval = lambda value: "n/a" if value is None else f"[{value[0]:.4f}, {value[1]:.4f}]"
                lines.append(f"| {name} | {full['estimate']:.4f} | {interval(full['record_percentile_95'])} | {interval(full['source_block_percentile_95'])} |")
            elif "partial_descriptive_mean_only" in result:
                lines.append(f"| {name} | {result['partial_descriptive_mean_only']:.4f} (partial only) | withheld | withheld |")
            else:
                lines.append(f"| {name} | n/a | n/a | n/a |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New output directory")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--plan", help="Previously frozen analysis-plan directory")
    parser.add_argument("--results", help="Completed or partial result JSON")
    args = parser.parse_args(argv)
    output = Path(args.out).expanduser().resolve()
    addendum = json.loads(ADDENDUM.read_text())
    manifest_path = ROOT / addendum["manifest"]
    hashes = {"analysis_addendum_sha256": file_sha(ADDENDUM), "analyzer_sha256": file_sha(ANALYZER),
        "manifest_sha256": file_sha(manifest_path)}
    if args.freeze_only:
        if args.results or args.plan:
            parser.error("freeze-only cannot consume results or another plan")
        output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(ADDENDUM, output / ADDENDUM.name)
        shutil.copy2(ANALYZER, output / ANALYZER.name)
        write_json(output / "freeze.json", {**hashes, "frozen_utc": datetime.now(timezone.utc).isoformat(), "results_read": False})
        print(f"Frozen prospective analysis plan: {output}")
        return 0
    if not args.plan or not args.results:
        parser.error("analysis requires --plan and --results")
    frozen = json.loads((Path(args.plan) / "freeze.json").read_text())
    if any(frozen.get(key) != value for key, value in hashes.items()):
        raise ValueError("analyzer, addendum, or manifest changed after plan freeze")
    results_path = Path(args.results).expanduser().resolve()
    report = json.loads(results_path.read_text())
    source_provenance = json.loads(results_path.with_name("provenance.json").read_text())
    if source_provenance["manifest_sha256"] != hashes["manifest_sha256"]:
        raise ValueError("results use a different manifest")
    if source_provenance["protocol_sha256"] != report["protocol_sha256"]:
        raise ValueError("report protocol hash differs from run provenance")
    analysis = analyze(report, json.loads(manifest_path.read_text()), addendum)
    analysis["provenance"] = {**hashes, "analysis_plan_frozen_utc": frozen["frozen_utc"],
        "analysis_utc": datetime.now(timezone.utc).isoformat(), "results_path": str(results_path),
        "results_sha256": file_sha(results_path), "result_provenance_sha256": file_sha(results_path.with_name("provenance.json")),
        "main_protocol_sha256": report["protocol_sha256"]}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "paired_analysis.json", analysis)
    (output / "paired_analysis.md").write_text(markdown_report(analysis))
    print(f"Wrote paired results and block sensitivity: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

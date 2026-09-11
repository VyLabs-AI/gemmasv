"""Post hoc retained-correctness comparison using existing source-free flags."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHODS = Path(__file__).with_name("decoded_retained_methods.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_histories(source, methods):
    for key in ("contains_source_text", "contains_model_generated_text", "contains_source_identifiers", "contains_token_arrays"):
        if source.get(key) is not False:
            raise ValueError("input must explicitly be a source-free summary")
    histories = source.get("histories", [])
    if len(histories) != methods["cohort"]["histories"]:
        raise ValueError("expected all 96 histories")
    indices = [history["history_index"] for history in histories]
    pairs = [(history["cluster_index"], history["variant_index"]) for history in histories]
    if len(set(indices)) != len(indices) or len(set(pairs)) != len(pairs):
        raise ValueError("duplicate history index or cluster/variant pair")
    if set(indices) != set(range(96)):
        raise ValueError("unexpected history indices")
    clusters = Counter(history["cluster_index"] for history in histories)
    if len(clusters) != 32 or any(size != 3 for size in clusters.values()):
        raise ValueError("expected 32 clusters of three histories")
    for cluster in clusters:
        if {history["variant_index"] for history in histories if history["cluster_index"] == cluster} != {0, 1, 2}:
            raise ValueError("expected variants 0, 1, and 2 in every cluster")
    return sorted(histories, key=lambda history: history["history_index"])


def condition_score(history, condition, endpoint):
    condition_data = history.get("retained", {}).get("conditions", {}).get(condition, {})
    completion = history.get("completion", {})
    repeat = history.get("reproducibility_checks", {}).get("conditions", {}).get(condition, {}).get("retained")
    value = condition_data.get("directional_correctness", {}).get(endpoint)
    if value is not None and not isinstance(value, bool):
        raise ValueError("scored endpoint must be boolean")
    checks = {
        "history_not_completed": completion.get("history_completed") is not True,
        "condition_not_completed": completion.get("condition_completion", {}).get(condition) is not True,
        "retained_output_unavailable": condition_data.get("available") is not True,
        "retained_repeat_not_reproduced": repeat is not True,
        "endpoint_unscorable": value is None,
    }
    reasons = [name for name, failed in checks.items() if failed]
    scorable = not reasons
    return {"scorable": scorable, "observed_match": value if scorable else None,
        "operational_correctness": int(value) if scorable else 0, "failure_reasons": reasons}


def cluster_interval(values, cluster_ids, *, resamples=100000, seed=20260906):
    if not values or len(values) != len(cluster_ids):
        raise ValueError("empty or unmatched values and clusters")
    labels = sorted(set(cluster_ids))
    grouped = [np.asarray([value for value, cluster in zip(values, cluster_ids) if cluster == label], dtype=np.float64) for label in labels]
    if any(len(group) != 3 for group in grouped):
        raise ValueError("bootstrap must preserve three variants per cluster")
    cluster_means = np.asarray([group.mean() for group in grouped])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(resamples, len(labels)))
    sampled = cluster_means[draws].mean(axis=1)
    return {"estimate": float(cluster_means.mean()), "percentile_95": np.quantile(sampled, [.025, .975]).tolist(),
        "target_clusters": len(labels), "histories": len(values), "resampling_unit": "target_cluster",
        "histories_resampled_within_cluster": False, "resamples": resamples, "seed": seed}


def analyze(source, methods, *, resamples=100000):
    histories = validate_histories(source, methods)
    rows = []
    for history in histories:
        row = {"history_index": history["history_index"], "cluster_index": history["cluster_index"], "variant_index": history["variant_index"], "endpoints": {}}
        for endpoint in methods["endpoints"]:
            row["endpoints"][endpoint] = {condition: condition_score(history, condition, endpoint) for condition in methods["conditions"]}
        rows.append(row)
    cluster_ids = [row["cluster_index"] for row in rows]
    result = {"schema": methods["schema"], "analysis_type": "post_hoc_existing_decoded_flags",
        "model_api_gpu_calls": 0, "histories": len(rows), "target_clusters": len(set(cluster_ids)),
        "contains_source_text": False, "contains_model_generated_text": False,
        "disclosure": methods["disclosure"], "claim_limits": methods["claim_limits"],
        "conditions": {}, "comparisons": [], "per_history": rows}
    for endpoint in methods["endpoints"]:
        result["conditions"][endpoint] = {}
        for condition in methods["conditions"]:
            scores = [row["endpoints"][endpoint][condition] for row in rows]
            failures = Counter(reason for score in scores for reason in score["failure_reasons"])
            successes = sum(score["operational_correctness"] for score in scores)
            result["conditions"][endpoint][condition] = {"all_history_denominator": len(rows),
                "scorable_histories": sum(score["scorable"] for score in scores), "correct_histories": successes,
                "unscorable_histories": sum(not score["scorable"] for score in scores),
                "failure_reason_counts": dict(failures),
                "rate": cluster_interval([score["operational_correctness"] for score in scores], cluster_ids, resamples=resamples)}
        for left, right in methods["comparisons"]:
            pairs = [(row["endpoints"][endpoint][left], row["endpoints"][endpoint][right]) for row in rows]
            differences = [a["operational_correctness"] - b["operational_correctness"] for a, b in pairs]
            complete = [(a, b) for a, b in pairs if a["scorable"] and b["scorable"]]
            table = Counter(f"{a['operational_correctness']}{b['operational_correctness']}" for a, b in pairs)
            result["comparisons"].append({"endpoint": endpoint, "left": left, "right": right,
                "direction": "left minus right", "all_history_denominator": len(rows),
                "both_scorable_histories": len(complete), "one_or_both_unscorable_histories": len(rows) - len(complete),
                "operational_paired_table": {"both_match": table["11"], "left_only_matches": table["10"], "right_only_matches": table["01"], "neither_matches": table["00"]},
                "difference": cluster_interval(differences, cluster_ids, resamples=resamples),
                "complete_pair_descriptive_difference": float(np.mean([a["operational_correctness"] - b["operational_correctness"] for a, b in complete])) if complete else None,
                "unscorable_history_indices": [row["history_index"] for row, (a, b) in zip(rows, pairs) if not (a["scorable"] and b["scorable"])]})
    return result


def markdown(result):
    lines = ["# Existing decoded histories: retained-answer correctness", "", result["disclosure"], "",
        "All 96 histories in 32 target clusters remain in the denominator. Repeated decodes contribute one canonical result per history. Missing or unscorable responses count as incorrect in the operational endpoint and are counted separately.", "",
        "| Existing endpoint | Condition | Retained matches / 96 | Scorable |",
        "|---|---|---:|---:|"]
    for endpoint, conditions in result["conditions"].items():
        for condition, value in conditions.items():
            lines.append(f"| {endpoint} | {condition} | {value['correct_histories']}/96 | {value['scorable_histories']}/96 |")
    lines += ["", "Paired differences below are percentage points, left minus right. Intervals resample the 32 target clusters and preserve all three variants together.", "",
        "| Endpoint | Left versus right | Difference (percentage points) | Cluster bootstrap 95% | Left-only / right-only matches | Unscorable pairs |",
        "|---|---|---:|---:|---:|---:|"]
    for value in result["comparisons"]:
        interval = value["difference"]["percentile_95"]
        table = value["operational_paired_table"]
        lines.append(f"| {value['endpoint']} | {value['left']} versus {value['right']} | {100 * value['difference']['estimate']:.2f} | [{100 * interval[0]:.2f}, {100 * interval[1]:.2f}] | {table['left_only_matches']} / {table['right_only_matches']} | {value['one_or_both_unscorable_histories']} |")
    lines += ["", result["claim_limits"], "",
        "The six intervals are descriptive and have no multiplicity correction. Agreement between responses is not correctness. These artifacts contain no forced-choice margin, gold-token rank, or continuation-KL measurement.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New output directory")
    args = parser.parse_args(argv)
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    methods = json.loads(METHODS.read_text())
    source_path = ROOT / methods["source"]
    provenance = {"methods_sha256": sha(METHODS), "analyzer_sha256": sha(__file__),
        "source_summary_sha256": sha(source_path), "source_summary_path": str(source_path),
        "methods_frozen_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_outcomes_seen_before_methods": True, "paired_contrasts_computed_before_freeze": False}
    shutil.copy2(METHODS, output / METHODS.name)
    shutil.copy2(__file__, output / Path(__file__).name)
    write_json(output / "provenance.json", provenance)
    result = analyze(json.loads(source_path.read_text()), methods)
    result["provenance"] = provenance
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "retained_correctness.json", result)
    (output / "retained_correctness.md").write_text(markdown(result))
    print(f"Wrote source-free post hoc retained-correctness analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

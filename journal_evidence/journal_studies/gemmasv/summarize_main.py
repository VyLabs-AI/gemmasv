"""Audit completed journal-bridge artifacts and make descriptive manuscript tables.

This runs no model inference. Tokenization reconstructs the registered contexts
from the already-local dataset and verifies their saved hashes independently.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
METHOD_NAMES = {
    "base_present": "Base present control",
    "base_full_repack": "Base literal rebuild",
    "graft_present": "Graft present control",
    "graft_full_repack": "Graft literal rebuild",
    "graft_masked_refit_proxy": "Graft FP32 masked refit",
    "graft_cache_delete_shift": "Graft cache-shift diagnostic",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_costs(report):
    costs = {}
    for arm in METHOD_NAMES:
        rows = [row.get("arms", {}).get(arm, {}) for row in report["records"]]
        complete = [row for row in rows if row.get("status") == "completed" and "metrics" in row]
        entry = {"expected_records": 8, "completed_records": len(complete),
            "full_repack_fallback_records": sum(row.get("full_repack_fallback", False) for row in complete)}
        for metric in ("update_seconds", "query_seconds", "end_to_end_seconds"):
            values = [row["timing"][metric]["median"] for row in complete]
            entry[metric] = {"median_of_record_medians": statistics.median(values),
                "minimum_record_median": min(values), "maximum_record_median": max(values)} if values else None
        costs[arm] = entry
    return costs


def audit(run_dir, analysis_path):
    report = json.loads((run_dir / "results.json").read_text())
    require(report["status"] in ("completed", "completed_with_failures"), "main run is not terminal")
    require(report.get("smoke") is False, "smoke cannot be a main result")
    provenance = json.loads((run_dir / "provenance.json").read_text())
    analysis = json.loads(analysis_path.read_text())
    require(analysis["provenance"]["results_sha256"] == sha(run_dir / "results.json"), "analysis results hash mismatch")
    require(analysis["provenance"]["result_provenance_sha256"] == sha(run_dir / "provenance.json"), "analysis provenance hash mismatch")
    source_root = run_dir / provenance["source_snapshot"]
    for relative, digest in provenance["source_files_sha256"].items():
        require(sha(source_root / relative) == digest, f"frozen source snapshot changed: {relative}")
    runner_relative = "journal_studies/gemmasv/run_comparison.py"
    require(sha(ROOT / runner_relative) == provenance["source_files_sha256"][runner_relative], "live runner differs from executed snapshot")
    protocol = json.loads((source_root / "journal_studies/gemmasv/protocol.json").read_text())
    require(sha(source_root / "journal_studies/gemmasv/protocol.json") == report["protocol_sha256"] == provenance["protocol_sha256"], "protocol hash mismatch")
    require(report["protocol"] == protocol, "embedded protocol differs")
    manifest_path = ROOT / protocol["manifest"]
    require(sha(manifest_path) == provenance["manifest_sha256"], "manifest changed")
    manifest = json.loads(manifest_path.read_text())
    expected_ids = [str(row["record_id"]) for row in manifest["records"]]
    require(report["selected_record_ids"] == expected_ids, "selection differs from all eight manifest records")
    ids = [row["record_id"] for row in report["records"]]
    require(len(ids) == 8 and len(set(ids)) == 8 and set(ids) == set(expected_ids), "missing, duplicated, or unexpected main record")
    require(report["execution"]["warmup"] == 1 and report["execution"]["repeats"] == 3, "timing repetitions differ from protocol")
    require(set(report["protocol"]["arms"]) == set(METHOD_NAMES), "arm set differs")
    for architecture, runtime in report["runtime"].items():
        config = runtime["config"]
        require(config["lora_path"] is None and config["dtype"] == "float32", "unexpected adapter or precision")
        require(config["model_revision"] == protocol["model_revision"] and config["window"] == 1024 and config["nu"] == .7 and config["solver_seed"] == 0, "runtime configuration differs")
        require(config["graft_enabled"] == (architecture == "graft"), "incorrect architecture switch")
        require(config["preserve_prefix_mass"] == config["per_boundary_box"] == (architecture == "graft"), "incorrect mass or boundary setting")
    # Reconstruct token metadata without loading any neural model or printing text.
    from datasets import Dataset
    from transformers import AutoTokenizer
    from journal_studies.gemmasv.run_comparison import build_record
    snapshot = Path(provenance["model_snapshot"])
    for name, metadata in provenance["model_files"].items():
        if "sha256" in metadata:
            require(sha(snapshot / name) == metadata["sha256"], "model configuration/tokenizer file changed")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    datasets = {}
    for split in ("forget10", "retain90"):
        files = provenance["dataset_cache"][split]["files"]
        require(len(files) == 1, "unexpected dataset shard count")
        require(sha(files[0]["path"]) == files[0]["sha256"], "dataset Arrow hash changed")
        datasets[split] = list(Dataset.from_file(files[0]["path"]))
    fillers = [str(row["answer"]) for row in datasets["retain90"][:32]]
    by_id = {row["record_id"]: row for row in report["records"]}
    missing_or_failed = []
    inspected_arms = 0
    scored_arms = 0
    for spec in manifest["records"]:
        row = by_id[str(spec["record_id"])]
        _, _, _, probes, rebuilt_metadata = build_record(SimpleNamespace(tokenizer=tokenizer), spec,
            datasets["forget10"], datasets["retain90"], fillers, protocol["configuration"])
        require(row["context"] == rebuilt_metadata, "independently reconstructed context/probe hash differs")
        require(set(row.get("arms", {})).issubset(METHOD_NAMES), "unexpected arm in result")
        for arm in METHOD_NAMES:
            inspected_arms += 1
            value = row.get("arms", {}).get(arm)
            if not value or value.get("status") != "completed" or "metrics" not in value:
                missing_or_failed.append({"record_id": row["record_id"], "arm": arm, "status": value.get("status") if value else "missing"})
                continue
            scored_arms += 1
            fields = [*value["metrics"]["deleted_target_quality"]["fields"], value["metrics"]["retained_quality"]]
            require(len(fields) == 4 and len({f["probe_id"] for f in fields}) == 4, "unexpected or duplicate scored probe")
            scores_by_id = {field["probe_id"]: field["score"] for field in fields}
            for probe in probes:
                score = scores_by_id[probe.probe_id]
                require(score["target_token_count"] == len(probe.target_ids), "target score token count differs")
                for metric in ("total_log_probability", "mean_log_probability", "geometric_mean_probability"):
                    require(math.isfinite(score[metric]), "nonfinite target score")
            for metric in ("update_seconds", "query_seconds", "end_to_end_seconds"):
                timing = value["timing"][metric]
                require(len(timing["samples"]) == 3 and all(math.isfinite(x) and x >= 0 for x in timing["samples"]), "invalid timing samples")
            own_kl = value["metrics"]["behavioral_kl_to_repack"]
            require(own_kl["direction"] == "KL(full_repack || method)", "own-reference KL direction differs")
            if arm.endswith("full_repack"):
                require(max(own_kl["deleted_probe_max_nats"], own_kl["retained_probe_nats"]) <= 1e-10, "self-reference KL mismatch")
            if arm.startswith("graft"):
                cross = value.get("cross_architecture", {})
                require(cross.get("is_deletion_certificate") is False and len(cross.get("probes", [])) == 4, "missing cross-architecture comparison")
    audit_result = {"audited_utc": datetime.now(timezone.utc).isoformat(), "run_status": report["status"],
        "expected_records": 8, "verified_context_and_probe_records": 8, "expected_arms": 48,
        "inspected_arms": inspected_arms, "completed_scored_arms": scored_arms,
        "missing_or_failed_arms": missing_or_failed,
        "frozen_source_files_verified": len(provenance["source_files_sha256"]),
        "model_weights_rehashed": False, "model_or_gpu_inference_calls": 0,
        "results_sha256": sha(run_dir / "results.json"), "paired_analysis_sha256": sha(analysis_path),
        "scope": "Independent token reconstruction and artifact consistency, not replay of numerical model inference."}
    return report, analysis, audit_result


def interval_text(full):
    if not full:
        return "unavailable"
    bounds = full["source_block_percentile_95"]
    return f"{full['estimate']:.4f} [{bounds[0]:.4f}, {bounds[1]:.4f}]"


def main_interpretation(analysis):
    pairs = {(row["left"], row["right"]): row for row in analysis["comparisons"]}
    def estimate(left, right, metric):
        full = pairs[(left, right)]["summary"]["metrics"][metric]["full_cohort"]
        return None if full is None else full["estimate"]
    proxy = "graft_masked_refit_proxy"
    own, base = "graft_full_repack", "base_full_repack"
    values = [estimate(proxy, ref, metric) for ref in (own, base)
              for metric in ("update_seconds_ratio", "end_to_end_seconds_ratio", "retained_mean_token_log_probability_delta")]
    cache_delta = estimate("graft_cache_delete_shift", own, "retained_mean_token_log_probability_delta")
    if any(value is None for value in [*values, cache_delta]):
        return "Some required comparisons are incomplete; no complete-cohort performance or utility conclusion is reported."
    own_update, own_total, own_retained, base_update, base_total, base_retained = values
    headline = ("This implementation showed no update-speed advantage over either rebuild comparator. "
                if own_update >= 1 and base_update >= 1 else "Update cost depends on the chosen rebuild comparator. ")
    return (headline + f"The FP32 proxy's paired update ratio was {own_update:.3f} relative to rebuilding the graft and {base_update:.2f} relative to rebuilding the ungrafted base. "
        f"Its update-plus-four-probe ratio was {own_total:.3f} and {base_total:.2f}, respectively. "
        "These are geometric means of within-record ratios, not ratios of the table medians. "
        f"The proxy's retained-target mean log-probability difference was {base_retained:+.3f} nats/token versus the base rebuild and {own_retained:+.3f} versus the graft rebuild; the base comparison includes architecture differences. "
        f"The cache-shift diagnostic's retained-target difference was {cache_delta:+.3f} nats/token versus graft rebuilding. "
        "All-boundary gate recomputation is an implementation cost, not a fundamental complexity bound.")


def make_report(report, analysis, audit_result, costs):
    lines = ["# GemmaSV completed cost and utility bridge", "",
        f"Status: **{report['status']}**. Audited {audit_result['completed_scored_arms']}/48 completed arms across all eight records. Context and probe token hashes were independently reconstructed from the pinned local tokenizer and dataset for all eight records.", "",
        main_interpretation(analysis), "",
        "**These costs measure deletion-state construction and four teacher-forced audit probes. They are not production request latency or exact-decrement timings.** Record lookup, tokenization, locating deletion positions, and building the literal edited token list occur outside the timer. Model loading and shared original prefill are separate. The FP32 masked-refit arm is a proxy; cache shifting is a diagnostic.", "",
        "**The proxy implements logical exclusion, not physical erasure.** It forks the original persistent memory, retains copied token IDs and K/V arrays, applies a drop mask, and recomputes every eligible stored boundary/head in every graft layer. The implementation does not restrict gate recomputation to affected boundaries. It reuses contextualized keys and does not re-ingest the language-model input unless boundary feasibility triggers full rebuilding. The study retains original controls and does not evaluate physical erasure from storage.", "",
        "The cohort reuses eight records from four source blocks. One warmup and three timing repetitions were used. The table reports the median across available per-record median times, with each arm's completion denominator shown; separate medians need not sum. No main-cohort claims are made from smoke measurements.", "",
        "| Arm | Completed | Update (s) | Four-probe audit (s) | Update + audit (s) | Boundary full-rebuild fallbacks |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, entry in costs.items():
        values = [entry[m]["median_of_record_medians"] if entry[m] else None for m in ("update_seconds", "query_seconds", "end_to_end_seconds")]
        rendered = [f"{value:.3f}" if value is not None else "n/a" for value in values]
        lines.append(f"| {METHOD_NAMES[arm]} | {entry['completed_records']}/8 | {' | '.join(rendered)} | {entry['full_repack_fallback_records']} |")
    lines += ["", "Paired estimates below average records, with four-source-block bootstrap sensitivity intervals. A cost ratio above one means the first arm is slower; a positive probability difference means its target continuation is more likely. The small reused cohort and only four source clusters do not support calibrated population or significance claims.", "",
        "| First arm versus second | Deleted target Δ LP/token | Retained target Δ LP/token | Update ratio | Update + audit ratio |",
        "|---|---:|---:|---:|---:|"]
    for comparison in analysis["comparisons"]:
        summary = comparison["summary"]
        metrics = summary["metrics"]
        names = ("deleted_mean_token_log_probability_delta", "retained_mean_token_log_probability_delta", "update_seconds_ratio", "end_to_end_seconds_ratio")
        entries = [interval_text(metrics[name]["full_cohort"]) for name in names]
        lines.append(f"| {METHOD_NAMES[comparison['left']]} versus {METHOD_NAMES[comparison['right']]} | {' | '.join(entries)} |")
    lines += ["", "All individual probe total/per-token log probabilities, paired absolute costs, and record-bootstrap sensitivity are retained in the frozen analyzer's JSON/Markdown output. Incomplete comparisons have their full-cohort estimates and intervals withheld.", "",
        "## Interpretation limits", "",
        "- The fresh reference is a literal edited token sequence, shorter than the historical padded never-stored control. Results are not interchangeable with that earlier reference.",
        "- Probabilities are teacher-forced target-continuation scores, not decoded leakage, semantic correctness, or general model quality. Within-architecture KL is full-vocabulary first-token `KL(fresh rebuild || arm)`; cross-architecture KL also includes architecture differences.",
        "- Boundary-precheck rebuild fallback in this proxy is different from the historical exact-decrement head/boundary fallback rate. Neither a proxy speed result nor reference self-KL proves exact-solver acceleration or conformance.",
        "- Per-probe querying includes state cloning, CPU float64 log-softmax, and sequential model steps, including a step after the final scored target token. The benchmark describes this audit implementation.",
        "- Base and graft run sequentially; graft order rotates across records. Timing repetitions do not establish independent observations or remove host drift.",
        "- The study cannot substantiate the manuscript claim that the exact audit is cheap enough for every deletion request. Practical deployment budgets require separate evidence.", ""]
    return "\n".join(lines)


def make_tex(report, analysis, costs):
    lines = ["% Generated from completed journal bridge and its frozen paired analysis.",
        "\\paragraph{A matched cost and target-utility bridge.}",
        "We ran a separate post hoc comparison on the eight fixed TOFU records",
        "using the frozen 4B-PT checkpoint and the mass-preserving, per-boundary",
        "configuration ($\\nu=0.7$, window 1024, solver seed 0; no adapter or training).",
        "All records were attempted without filtering by admission. The arms share",
        "the original context and probes; each architecture's fresh reference",
        "prefills the literal edited token sequence, which is shorter than the",
        "historical padded never-stored control. The comparison includes the",
        "ungrafted model with raw-record removal and cache rebuilding, the graft's",
        "fresh rebuild, an FP32 masked-refit proxy, and diagnostic cache shifting.",
        "The proxy implements logical exclusion through a drop mask while retaining",
        "copied token IDs and K/V arrays. It recomputes every eligible stored",
        "boundary/head in every graft layer, including unaffected prefixes; it",
        "does not restrict work to affected gates. Contextualized keys are reused",
        "without language-model re-ingestion unless the feasibility precheck",
        "triggers a full rebuild. Physical erasure from storage is not evaluated.",
        "No exact-decrement or exact-certificate construction was timed.", "",
        "Each arm used one warmup and three measured repetitions. We separately",
        "timed deletion-state construction and four teacher-forced target probes",
        "(three deleted fields and one retained field), with device synchronization.",
        "Record lookup, tokenization, locating deletion positions, and constructing",
        "the edited token list were outside the timer; model loading and shared",
        "original prefill were separate. Query timing includes state cloning and",
        "the existing sequential teacher-forced scorer. These are audit-workload",
        "times, not production response latency. Table~\\ref{tab:journal-bridge-cost}",
        "reports absolute costs; the target scores are not decoded leakage rates.", "",
        "\\begin{table*}[t]", "\\centering\\small",
        "\\caption{Completed journal bridge. Seconds are medians across the per-record median times; independently summarized columns need not add. The FP32 proxy is not an exact-decrement implementation, and cache shifting is diagnostic.}",
        "\\label{tab:journal-bridge-cost}", "\\begin{tabular}{lrrrr}", "\\toprule",
        "Arm & Complete & Update (s) & Four-probe audit (s) & Update + audit (s) \\\\", "\\midrule"]
    for arm, entry in costs.items():
        values = [entry[m]["median_of_record_medians"] if entry[m] else None for m in ("update_seconds", "query_seconds", "end_to_end_seconds")]
        rendered = [f"{value:.3f}" if value is not None else "--" for value in values]
        lines.append(f"{METHOD_NAMES[arm]} & {entry['completed_records']}/8 & " + " & ".join(rendered) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", "",
        f"The FP32 proxy used a boundary-feasibility full-rebuild fallback in {costs['graft_masked_refit_proxy']['full_repack_fallback_records']}/{costs['graft_masked_refit_proxy']['completed_records']} completed record evaluations. This is distinct from the historical per-head exact-decrement fallback rate.", "",
        "\\begin{table*}[t]", "\\centering\\small",
        "\\caption{Paired FP32 masked-refit comparisons. Probability differences are mean target log probability per token, proxy minus reference. Cost ratios are proxy/reference, with values above one indicating a slower proxy. Brackets are 95\\% percentile sensitivity intervals resampling four source blocks (10{,}000 draws), preserving both records in each block.}",
        "\\label{tab:journal-bridge-paired}", "\\begin{tabular}{lrr}", "\\toprule",
        "Paired measure & Graft rebuild reference & Base rebuild reference \\\\", "\\midrule"]
    comparisons = {row["right"]: row for row in analysis["comparisons"] if row["left"] == "graft_masked_refit_proxy"}
    for label, metric in (("Deleted target $\\Delta$ LP/token", "deleted_mean_token_log_probability_delta"),
                          ("Retained target $\\Delta$ LP/token", "retained_mean_token_log_probability_delta"),
                          ("Update ratio", "update_seconds_ratio"),
                          ("Update + audit ratio", "end_to_end_seconds_ratio")):
        entries = []
        for reference in ("graft_full_repack", "base_full_repack"):
            full = comparisons[reference]["summary"]["metrics"][metric]["full_cohort"]
            if not full:
                entries.append("unavailable")
            else:
                low, high = full["source_block_percentile_95"]
                entries.append(f"${full['estimate']:.4f}\\ [{low:.4f}, {high:.4f}]$")
        lines.append(label + " & " + " & ".join(entries) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", "",
        main_interpretation(analysis), "",
        "The cohort reuses four source blocks, whose independence is not established;",
        "bootstrap intervals are descriptive sensitivity summaries, not evidence of",
        "population equivalence or noninferiority. The base and graft ran sequentially,",
        "and three timing repetitions are not three independent datasets. The",
        "full report retains all per-record outcomes, target-level probability",
        "differences, and failures. These measurements neither establish preservation",
        "of general capability nor justify running an exact certificate on every",
        "production deletion request.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True, help="New audit output directory")
    args = parser.parse_args(argv)
    run_dir = Path(args.run).expanduser().resolve()
    report, analysis, audit_result = audit(run_dir, Path(args.analysis).expanduser().resolve())
    costs = completed_costs(report)
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "artifact_audit.json", audit_result)
    write_json(output / "absolute_costs.json", costs)
    report_text = make_report(report, analysis, audit_result, costs)
    tex = make_tex(report, analysis, costs)
    (output / "MAIN_REPORT.md").write_text(report_text)
    (output / "COST_UTILITY_INSERT.tex").write_text(tex)
    (ROOT / "journal_studies/gemmasv/MAIN_REPORT.md").write_text(report_text)
    (ROOT / "journal_studies/gemmasv/COST_UTILITY_INSERT.tex").write_text(tex)
    print(f"Audited completed bridge and wrote report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

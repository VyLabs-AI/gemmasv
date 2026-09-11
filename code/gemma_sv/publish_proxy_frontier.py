"""Publish a compact source-free proxy frontier from the full local report."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


DEFAULT_INPUT = Path("outputs/gemma_sv_proxy_precision/frontier.json")
DEFAULT_OUTPUT = Path("gemma_sv/benchmarks/proxy_frontier_v1.json")


def _selected_calibration_row(report: Mapping[str, Any]) -> dict[str, Any]:
    selection = report["calibration"]["selection"]
    policy = selection.get("selected_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("proxy frontier has no selected calibration policy")
    matches = [
        row
        for row in selection.get("frontier", [])
        if row.get("policy") == policy
    ]
    if len(matches) != 1:
        raise ValueError("selected calibration policy is missing or ambiguous")
    return copy.deepcopy(matches[0])


def _observation_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not observations:
        raise ValueError("proxy frontier observations are empty")
    raw_kls = [
        float(row["metrics"]["max_kl_refit64_to_proxy_nats"])
        for row in observations
    ]
    retained = [
        abs(float(row["metrics"]["retained_drift_from_refit64_nats"]))
        for row in observations
    ]
    peak_memory = max(
        int(row["peak_memory"]["process_peak_resident_bytes"])
        for row in observations
    )
    return {
        "contexts": len(observations),
        "raw_proxy_max_kl_refit64_nats": max(raw_kls),
        "raw_proxy_mean_context_max_kl_refit64_nats": statistics.fmean(raw_kls),
        "raw_proxy_mean_retained_abs_drift_nats": statistics.fmean(retained),
        "process_peak_resident_bytes": peak_memory,
        "max_solver_residuals": {
            "equality": max(
                float(row["diagnostics"]["max_equality_residual"])
                for row in observations
            ),
            "box_violation": max(
                float(row["diagnostics"]["max_box_violation"])
                for row in observations
            ),
            "kkt": max(
                float(row["diagnostics"]["max_kkt_residual"])
                for row in observations
            ),
            "projected_gradient": max(
                float(row["diagnostics"]["max_projected_gradient_inf"])
                for row in observations
            ),
            "duality_gap": max(
                float(row["diagnostics"]["max_duality_gap"])
                for row in observations
            ),
        },
    }


def _policy_observations(
    observations: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in observations
        if int(row["solver"]["fista_iterations"])
        == int(policy["fista_iterations"])
        and float(row["solver"]["partition_cutoff"])
        == float(policy["partition_cutoff"])
    ]
    if not selected:
        raise ValueError("selected policy has no observations")
    return selected


def _public_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    raw_path = Path(str(result.get("path", "")))
    parts = raw_path.parts
    if "gemma_sv" in parts:
        result["path"] = Path(*parts[parts.index("gemma_sv") :]).as_posix()
    elif raw_path.name:
        result["path"] = raw_path.name
    return result


def compact_frontier(
    report: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    if report.get("status") != "completed":
        raise ValueError("proxy frontier report is not complete")
    if report.get("claim_scope") != (
        "residual-qualified approximation; not a certificate"
    ):
        raise ValueError("proxy frontier claim scope is missing")
    calibration_row = _selected_calibration_row(report)
    validation = copy.deepcopy(
        report["validation"].get("locked_policy_result")
    )
    if not isinstance(validation, dict):
        raise ValueError("proxy frontier has no locked validation result")
    validation.pop("rows", None)
    policy = calibration_row["policy"]
    if validation.get("policy") != policy:
        raise ValueError("validation used a different policy")
    if int(validation.get("context_violations", -1)) != 0:
        raise ValueError("locked validation contains a KL violation")
    return {
        "schema_version": 1,
        "evaluation": "gemma_sv_proxy_frontier_compact",
        "contains_source_text": False,
        "contains_model_weights": False,
        "source_report_sha256": source_sha256,
        "claim_scope": report["claim_scope"],
        "manifest": _public_manifest(report["manifest"]),
        "split": copy.deepcopy(report["split"]),
        "grid": copy.deepcopy(report["grid"]),
        "power_iteration_seed_rule": (
            "10000 * (adapter_index + 1) + record_index; reset before each policy"
        ),
        "selection_rule": report["calibration"]["selection"]["selection_rule"],
        "selected_policy": copy.deepcopy(policy),
        "calibration": {
            **calibration_row,
            "observation_summary": _observation_summary(
                _policy_observations(
                    report["calibration"]["observations"],
                    policy,
                )
            ),
        },
        "validation": {
            **validation,
            "observation_summary": _observation_summary(
                report["validation"]["observations"]
            ),
        },
    }


def deterministic_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        ensure_ascii=False,
    ) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    raw = args.report.read_bytes()
    report = json.loads(raw)
    published = compact_frontier(
        report,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(deterministic_json(published), encoding="utf-8")
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

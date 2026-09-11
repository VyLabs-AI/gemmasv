"""Publish the compact, source-free Qwen3.5 replay/receipt result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / "qwen35_replay_v3.json"
)
DEFAULT_REPORT = Path("outputs/qwen35_replay/audit_v3.json")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / "qwen35_replay_result_v3.json"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same_float(first: Any, second: Any) -> bool:
    return math.isclose(
        float(first),
        float(second),
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def compact_qwen_result(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    manifest_sha256: str,
    report_sha256: str,
    expected_implementation_sha256: str,
) -> dict[str, Any]:
    if (
        manifest.get("name") != "qwen35_replay_v3"
        or manifest.get("version") != 3
        or manifest.get("status") != "frozen-before-validation-rerun"
    ):
        raise ValueError("Qwen manifest is not the frozen v3 protocol")
    if (
        report.get("schema") != "qwen35-replay-audit-v2"
        or report.get("status") != "completed"
        or report.get("tiny_random_model") is not False
        or report.get("manifest_sha256") != manifest_sha256
        or report.get("model") != manifest.get("model")
        or report.get("device") != manifest["model"]["device"]
        or (report.get("environment") or {}).get("versions", {}).get(
            "transformers"
        )
        != manifest["model"]["transformers"]
        or report.get("protocol")
        != {
            "name": manifest["name"],
            "version": manifest["version"],
            "status": manifest["status"],
        }
    ):
        raise ValueError("Qwen live report is incomplete or unbound")
    implementation_sha256 = report.get("implementation_sha256")
    if (
        not isinstance(implementation_sha256, str)
        or len(implementation_sha256) != 64
        or any(
            value not in "0123456789abcdef"
            for value in implementation_sha256
        )
        or implementation_sha256 != expected_implementation_sha256
    ):
        raise ValueError("Qwen implementation fingerprint is invalid")
    if report.get("kernel_implementation") != {
        "flash_linear_attention_available": False,
        "causal_conv1d_available": False,
        "delta_rule_fallback": "torch_chunk_gated_delta_rule",
        "causal_conv1d_fallback": "torch_causal_conv1d_update",
    }:
        raise ValueError("Qwen kernel implementation differs")
    result = report.get("result") or {}
    if result.get("all_checks_pass") is not True:
        raise ValueError("Qwen aggregate checks did not pass")
    checks = result.get("checks") or {}
    required_checks = {
        "finite",
        "forcing_closes",
        "frozen_transport_closes",
        "live_kernel_conformance",
        "native_transport_fails",
        "non_vacuous",
        "repeat_exact",
        "replay_exact",
    }
    if set(checks) != required_checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError("Qwen live report has failed or unknown checks")
    scenario_rows = result.get("scenarios")
    expected_scenarios = [
        row["scenario_id"]
        for row in manifest["synthetic_protocol"]["scenarios"]
    ]
    if (
        not isinstance(scenario_rows, list)
        or [row.get("scenario_id") for row in scenario_rows]
        != expected_scenarios
        or not all(row.get("all_checks_pass") is True for row in scenario_rows)
        or result.get("passed_scenarios") != len(expected_scenarios)
        or result.get("required_scenarios") != len(expected_scenarios)
    ):
        raise ValueError("Qwen scenario coverage differs from the frozen protocol")
    for scenario in scenario_rows:
        scenario_checks = scenario.get("checks") or {}
        if (
            set(scenario_checks) != required_checks
            or not all(value is True for value in scenario_checks.values())
            or scenario.get("all_checks_pass")
            is not all(scenario_checks.values())
        ):
            raise ValueError("Qwen scenario checks are inconsistent")
    recomputed_checks = {
        name: all(
            scenario["checks"][name] is True for scenario in scenario_rows
        )
        for name in required_checks
    }
    if checks != recomputed_checks:
        raise ValueError("Qwen aggregate checks differ from scenario checks")
    probe_layers = [
        str(value)
        for value in manifest["architecture_contract"]["probe_linear_layers"]
    ]
    receipt_rows = []
    compact_cases = []
    suffix_variation = {}
    contract = manifest["architecture_contract"]
    expected_array_count = int(contract["expected_array_or_output_count"])
    expected_flag_count = int(contract["expected_null_rope_flag_count"])
    expected_arrays = {"logits"}
    expected_flags = {"length", "wrapper.rope_deltas"}
    for layer_id in contract["linear_attention_layers"]:
        expected_arrays.update(
            {
                f"layer_{layer_id}.conv",
                f"layer_{layer_id}.recurrent",
            }
        )
        expected_flags.update(
            {
                f"layer_{layer_id}.previous",
                f"layer_{layer_id}.conv_initialized",
                f"layer_{layer_id}.recurrent_initialized",
            }
        )
    for layer_id in contract["full_attention_layers"]:
        expected_arrays.update(
            {
                f"layer_{layer_id}.keys",
                f"layer_{layer_id}.values",
            }
        )
        expected_flags.add(f"layer_{layer_id}.initialized")
    acceptance = manifest["acceptance"]
    for scenario in scenario_rows:
        cases = scenario.get("cases")
        if not isinstance(cases, list) or [
            case.get("suffix") for case in cases
        ] != ["suffix_a", "suffix_b"]:
            raise ValueError("Qwen suffix cases differ from the frozen protocol")
        suffix_variation[scenario["scenario_id"]] = scenario[
            "suffix_effect_variation"
        ]
        if set(suffix_variation[scenario["scenario_id"]]) != set(
            probe_layers
        ):
            raise ValueError("Qwen suffix-variation layer coverage differs")
        if not all(
            math.isfinite(float(value))
            for row in suffix_variation[scenario["scenario_id"]].values()
            for value in row.values()
        ):
            raise ValueError("Qwen suffix variation is non-finite")
        scenario_receipts = []
        for case in cases:
            replay = case["replay_vs_fresh_omission"]
            repeat = case["repeat_vs_fresh_omission"]
            present = case["present_vs_omitted"]
            if (
                replay["array_count"] != expected_array_count
                or replay["flag_count"] != expected_flag_count
                or set(replay["flag_names"]) != expected_flags
                or set(replay["per_array"]) != expected_arrays
                or replay["max_abs"] != 0.0
                or replay["all_arrays_exact"] is not True
                or replay["all_flags_equal"] is not True
                or replay["all_values_finite"] is not True
                or replay["shape_mismatch_count"] != 0
                or repeat["array_count"] != expected_array_count
                or repeat["flag_count"] != expected_flag_count
                or set(repeat["flag_names"]) != expected_flags
                or set(repeat["per_array"]) != expected_arrays
                or repeat["max_abs"] != 0.0
                or repeat["all_arrays_exact"] is not True
                or repeat["all_flags_equal"] is not True
                or repeat["all_values_finite"] is not True
                or repeat["shape_mismatch_count"] != 0
                or present["array_count"] != expected_array_count
                or present["flag_count"] != expected_flag_count
                or set(present["flag_names"]) != expected_flags
                or set(present["per_array"]) != expected_arrays
                or present["all_values_finite"] is not True
                or not (
                    present["max_abs"]
                    >= acceptance[
                        "present_vs_omitted_nonzero_max_abs_min"
                    ]
                    or present["shape_mismatch_count"] > 0
                )
            ):
                raise ValueError("Qwen complete-state replay is not exact")
            receipt = case["receipt_analysis"]
            if set(receipt) != set(probe_layers):
                raise ValueError("Qwen receipt layer coverage differs")
            for row in receipt.values():
                live_components = (
                    row["present_initial_state_relative_residual"],
                    row["omitted_initial_state_relative_residual"],
                    row[
                        "present_live_kernel_recurrence_relative_residual"
                    ],
                    row[
                        "omitted_live_kernel_recurrence_relative_residual"
                    ],
                    row[
                        "native_sequential_to_live_state_scale_relative_residual"
                    ],
                )
                recomputed_live_conformance = max(live_components)
                recomputed_mismatch_ratio = (
                    row["native_transport_relative_residual"]
                    / max(
                        row["frozen_transport_relative_residual"],
                        1e-12,
                    )
                )
                required_values = (
                    row["frozen_transport_relative_residual"],
                    row["forcing_decomposition_max_relative_residual"],
                    row["forcing_decomposition_max_absolute_residual"],
                    row["live_kernel_conformance_max_relative_residual"],
                    row["native_sequential_to_live_relative_residual"],
                    row["native_transport_relative_residual"],
                    row["native_mismatch_to_frozen_floor_ratio"],
                    row["boundary_effect_norm"],
                    row["native_effect_norm"],
                    row["sequential_native_effect_norm"],
                    *live_components,
                )
                if row.get("finite") is not True or not all(
                    math.isfinite(float(value)) for value in required_values
                ):
                    raise ValueError("Qwen receipt contains non-finite values")
                if not _same_float(
                    row["live_kernel_conformance_max_relative_residual"],
                    recomputed_live_conformance,
                ):
                    raise ValueError(
                        "Qwen live-kernel conformance summary differs"
                    )
                if not _same_float(
                    row["native_mismatch_to_frozen_floor_ratio"],
                    recomputed_mismatch_ratio,
                ):
                    raise ValueError("Qwen native mismatch ratio differs")
                if (
                    row["frozen_transport_relative_residual"]
                    > acceptance[
                        "frozen_input_transport_relative_residual_max"
                    ]
                    or row["forcing_decomposition_max_relative_residual"]
                    > acceptance[
                        "forcing_decomposition_relative_residual_max"
                    ]
                    or row["live_kernel_conformance_max_relative_residual"]
                    > acceptance[
                        "live_kernel_recurrence_relative_residual_max"
                    ]
                ):
                    raise ValueError("Qwen receipt closure exceeds threshold")
                scenario_receipts.append(row)
            receipt_rows.extend(receipt.values())
            compact_cases.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "suffix": case["suffix"],
                    "present_vs_omitted_max_abs": case[
                        "present_vs_omitted"
                    ]["max_abs"],
                    "present_vs_omitted_shape_mismatches": case[
                        "present_vs_omitted"
                    ]["shape_mismatch_count"],
                    "replay_array_count": replay["array_count"],
                    "replay_max_abs": replay["max_abs"],
                    "repeat_max_abs": repeat["max_abs"],
                    "receipt_analysis": receipt,
                }
            )
        if sum(
            row["native_mismatch_to_frozen_floor_ratio"]
            >= acceptance["native_transport_mismatch_to_floor_ratio_min"]
            and row["boundary_effect_norm"] > 0
            for row in scenario_receipts
        ) < acceptance["minimum_probe_layers_exceeding_native_mismatch"]:
            raise ValueError("Qwen native transport mismatch is insufficient")
    return {
        "schema": f"qwen35-replay-result-v{manifest['version']}",
        "contains_source_text": False,
        "contains_model_weights": False,
        "manifest_sha256": manifest_sha256,
        "source_report_sha256": report_sha256,
        "implementation_sha256": implementation_sha256,
        "kernel_implementation": report["kernel_implementation"],
        "model": manifest["model"],
        "state_coverage": manifest["architecture_contract"][
            "complete_state_coverage"
        ],
        "checks": checks,
        "cases": compact_cases,
        "aggregate": {
            "maximum_frozen_transport_relative_residual": max(
                row["frozen_transport_relative_residual"]
                for row in receipt_rows
            ),
            "maximum_forcing_decomposition_relative_residual": max(
                row["forcing_decomposition_max_relative_residual"]
                for row in receipt_rows
            ),
            "maximum_live_kernel_conformance_relative_residual": max(
                row["live_kernel_conformance_max_relative_residual"]
                for row in receipt_rows
            ),
            "maximum_native_sequential_to_live_relative_residual": max(
                row["native_sequential_to_live_relative_residual"]
                for row in receipt_rows
            ),
            "minimum_native_transport_relative_residual": min(
                row["native_transport_relative_residual"]
                for row in receipt_rows
            ),
            "maximum_native_transport_relative_residual": max(
                row["native_transport_relative_residual"]
                for row in receipt_rows
            ),
            "minimum_native_mismatch_to_frozen_floor_ratio": min(
                row["native_mismatch_to_frozen_floor_ratio"]
                for row in receipt_rows
                if row["boundary_effect_norm"] > 0
            ),
            "suffix_effect_variation": suffix_variation,
            "elapsed_seconds": result["elapsed_seconds"],
        },
        "scope": {
            "replay_reference": (
                "replay suffix and fresh omission use the same tokenization "
                "and segment schedule"
            ),
            "transport_surface": "selected Gated DeltaNet recurrent states",
            "complete_replay_surface": True,
            "different_chunk_schedules_not_claimed_equal": True,
            "scenario_count": len(scenario_rows),
            "cohort_design": manifest["reporting"]["cohort_design"],
            "validation_rerun": manifest["reporting"]["validation_rerun"],
            "independent_replication": False,
        },
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest_raw = args.manifest.read_bytes()
    report_raw = args.report.read_bytes()
    compact = compact_qwen_result(
        json.loads(manifest_raw),
        json.loads(report_raw),
        manifest_sha256=_sha256(manifest_raw),
        report_sha256=_sha256(report_raw),
        expected_implementation_sha256=_sha256(
            (
                Path(__file__).resolve().parent
                / "qwen_replay_audit.py"
            ).read_bytes()
        ),
    )
    _atomic_write(args.out, compact)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

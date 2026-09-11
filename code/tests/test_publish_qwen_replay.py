from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from gemma_sv.publish_qwen_replay import compact_qwen_result


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "gemma_sv" / "benchmarks" / "qwen35_replay_v3.json"
)


def _fixture():
    raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    contract = manifest["architecture_contract"]
    arrays = {"logits"}
    flags = {"length", "wrapper.rope_deltas"}
    for layer_id in contract["linear_attention_layers"]:
        arrays.update(
            {
                f"layer_{layer_id}.conv",
                f"layer_{layer_id}.recurrent",
            }
        )
        flags.update(
            {
                f"layer_{layer_id}.previous",
                f"layer_{layer_id}.conv_initialized",
                f"layer_{layer_id}.recurrent_initialized",
            }
        )
    for layer_id in contract["full_attention_layers"]:
        arrays.update(
            {
                f"layer_{layer_id}.keys",
                f"layer_{layer_id}.values",
            }
        )
        flags.add(f"layer_{layer_id}.initialized")

    exact = {
        "array_count": 65,
        "flag_count": 82,
        "flag_names": sorted(flags),
        "shape_mismatch_count": 0,
        "max_abs": 0.0,
        "all_arrays_exact": True,
        "all_flags_equal": True,
        "all_values_finite": True,
        "per_array": {name: {} for name in arrays},
    }
    present = copy.deepcopy(exact)
    present.update(
        {
            "max_abs": 1.0,
            "all_arrays_exact": False,
            "all_flags_equal": False,
        }
    )
    receipt = {
        "boundary_effect_norm": 1.0,
        "native_effect_norm": 0.5,
        "sequential_native_effect_norm": 0.5,
        "frozen_transport_relative_residual": 1e-6,
        "native_transport_relative_residual": 0.1,
        "native_mismatch_to_frozen_floor_ratio": 1e5,
        "forcing_decomposition_max_relative_residual": 1e-7,
        "forcing_decomposition_max_absolute_residual": 1e-7,
        "present_initial_state_relative_residual": 0.0,
        "omitted_initial_state_relative_residual": 0.0,
        "present_live_kernel_recurrence_relative_residual": 1e-6,
        "omitted_live_kernel_recurrence_relative_residual": 1e-6,
        "native_sequential_to_live_relative_residual": 2e-6,
        "native_sequential_to_live_state_scale_relative_residual": 1e-6,
        "live_kernel_conformance_max_relative_residual": 1e-6,
        "finite": True,
    }
    receipt_by_layer = {
        str(layer_id): copy.deepcopy(receipt)
        for layer_id in contract["probe_linear_layers"]
    }
    case = {
        "present_vs_omitted": present,
        "replay_vs_fresh_omission": copy.deepcopy(exact),
        "repeat_vs_fresh_omission": copy.deepcopy(exact),
        "receipt_analysis": receipt_by_layer,
    }
    checks = {
        "finite": True,
        "forcing_closes": True,
        "frozen_transport_closes": True,
        "live_kernel_conformance": True,
        "native_transport_fails": True,
        "non_vacuous": True,
        "repeat_exact": True,
        "replay_exact": True,
    }
    scenarios = []
    for scenario in manifest["synthetic_protocol"]["scenarios"]:
        first = copy.deepcopy(case)
        first["suffix"] = "suffix_a"
        second = copy.deepcopy(case)
        second["suffix"] = "suffix_b"
        scenarios.append(
            {
                "scenario_id": scenario["scenario_id"],
                "all_checks_pass": True,
                "checks": copy.deepcopy(checks),
                "cases": [first, second],
                "suffix_effect_variation": {
                    str(layer_id): {
                        "effect_a_norm": 1.0,
                        "effect_b_norm": 1.0,
                        "relative_difference": 0.1,
                    }
                    for layer_id in contract["probe_linear_layers"]
                },
            }
        )
    implementation = "a" * 64
    report = {
        "schema": "qwen35-replay-audit-v2",
        "status": "completed",
        "tiny_random_model": False,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": implementation,
        "model": manifest["model"],
        "device": manifest["model"]["device"],
        "environment": {
            "versions": {
                "transformers": manifest["model"]["transformers"],
            }
        },
        "protocol": {
            "name": manifest["name"],
            "version": manifest["version"],
            "status": manifest["status"],
        },
        "kernel_implementation": {
            "flash_linear_attention_available": False,
            "causal_conv1d_available": False,
            "delta_rule_fallback": "torch_chunk_gated_delta_rule",
            "causal_conv1d_fallback": "torch_causal_conv1d_update",
        },
        "result": {
            "all_checks_pass": True,
            "checks": checks,
            "passed_scenarios": 3,
            "required_scenarios": 3,
            "scenarios": scenarios,
            "elapsed_seconds": 1.0,
        },
    }
    return manifest, report, implementation


def _compact(manifest, report, implementation):
    return compact_qwen_result(
        manifest,
        report,
        manifest_sha256=report["manifest_sha256"],
        report_sha256="b" * 64,
        expected_implementation_sha256=implementation,
    )


def test_qwen_publisher_accepts_complete_v3_matrix():
    manifest, report, implementation = _fixture()
    compact = _compact(manifest, report, implementation)
    assert compact["schema"] == "qwen35-replay-result-v3"
    assert compact["scope"]["independent_replication"] is False
    assert len(compact["cases"]) == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "repeat_flags",
        "live_component",
        "scenario_check",
        "implementation",
    ],
)
def test_qwen_publisher_fails_closed(mutation):
    manifest, report, implementation = _fixture()
    if mutation == "repeat_flags":
        report["result"]["scenarios"][0]["cases"][0][
            "repeat_vs_fresh_omission"
        ]["all_flags_equal"] = False
    elif mutation == "live_component":
        report["result"]["scenarios"][0]["cases"][0][
            "receipt_analysis"
        ]["0"]["present_live_kernel_recurrence_relative_residual"] = 1.0
    elif mutation == "scenario_check":
        report["result"]["scenarios"][0]["checks"][
            "live_kernel_conformance"
        ] = False
    else:
        report["implementation_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        _compact(manifest, report, implementation)

import math
import json

import pytest

from gemma_sv.publish_iclr_summary import compact_summary
from gemma_sv.summarize_multiseed import (
    AggregationError,
    PairingError,
    _recompute_admission_row,
    floor_safe_log_summary,
    student_t_ci95,
    summarize_paths,
    validate_recovery_control_pair,
)


def _admission_row(
    *,
    answer_lift=0.2,
    field_lifts=(0.2, 0.3, 0.4),
    field_ranks=(1, 2, 3),
    retained_rank=4,
    feasible=(True, True, True, True, True),
    reasons=(),
):
    return {
        "status": "rejected" if reasons else "admitted",
        "reasons": list(reasons),
        "thresholds": {
            "minimum_answer_lift_nats": 0.05,
            "minimum_secret_lift_nats": 0.05,
            "maximum_first_token_rank": 10,
            "all_fields_must_pass": True,
            "fixed_c_all_boundaries_must_be_feasible": True,
        },
        "record_answer_lift_nats": answer_lift,
        "deleted_fields": [
            {
                "probe_id": f"deleted_field_{index}",
                "secret_lift_nats": lift,
                "first_token_rank": rank,
            }
            for index, (lift, rank) in enumerate(
                zip(field_lifts, field_ranks, strict=True)
            )
        ],
        "retained_field": {
            "probe_id": "retained_field",
            "first_token_rank": retained_rank,
        },
        "fixed_c_feasibility": [
            {
                "start": 128 * (index + 1),
                "retained_capacity": 1.25 if value else 0.75,
                "feasible": value,
            }
            for index, value in enumerate(feasible)
        ],
    }


def test_student_t_ci_uses_three_seed_critical_value():
    summary = student_t_ci95([1.0, 2.0, 3.0])
    expected_half_width = 4.302652729696142 / math.sqrt(3.0)

    assert summary["mean"] == pytest.approx(2.0)
    assert summary["half_width"] == pytest.approx(expected_half_width)
    assert summary["lower"] == pytest.approx(2.0 - expected_half_width)
    assert summary["upper"] == pytest.approx(2.0 + expected_half_width)


def test_pairing_rejects_different_stage2_batch_fingerprints():
    config = {
        "model": "google/gemma-3-1b-pt",
        "batch": 8,
        "seq_len": 512,
        "stage2_steps": 6000,
        "lr2": 1e-3,
        "rank": 8,
        "seed": 1,
    }
    recovery = {
        "seed": 1,
        "stage2_stream": {"batches": 6000, "sha256": "recovery-stream"},
        "config": config,
    }
    control = {
        "seed": 1,
        "stage2_stream": {"batches": 6000, "sha256": "different-stream"},
        "config": config,
    }

    with pytest.raises(PairingError, match="batch fingerprint mismatch"):
        validate_recovery_control_pair(recovery, control)


def test_floor_safe_log_summary_handles_zero_kl():
    summary = floor_safe_log_summary([0.0, 1e-200, 1e-10])

    assert summary["floored_count"] == 1
    assert summary["mean_log10"] == pytest.approx(-170.0)
    assert summary["geometric_mean"] == pytest.approx(1e-170)
    assert math.isfinite(summary["mean_log10"])


def test_full_summary_accepts_current_recovery_and_audit_schemas(tmp_path):
    paths = {
        name: []
        for name in (
            "recovery",
            "control",
            "certificate",
            "audit",
            "cross_corpus",
            "zero_shot",
        )
    }

    def write(name, seed, payload):
        path = tmp_path / f"seed-{seed}-{name}.json"
        path.write_text(json.dumps(payload))
        paths[name].append(path)

    for seed in range(5):
        stream = {
            "schema": "gemma-sv-stage2-stream-v1",
            "batches": 6000,
            "sha256": f"stream-{seed}",
            "first_batch_sha256": f"first-{seed}",
            "last_batch_sha256": f"last-{seed}",
        }
        config = {
            "model": "google/gemma-3-1b-pt",
            "model_revision": "revision",
            "batch": 8,
            "seq_len": 512,
            "stage2_steps": 6000,
            "lr2": 0.001,
            "rank": 8,
            "fineweb_revision": "dataset-revision",
            "data_seed": seed,
            "init_seed": seed,
            "seed": seed,
        }
        adapter_hash = f"adapter-{seed}"
        write(
            "recovery",
            seed,
            {
                "schema": "gemma-sv-recovery-v2",
                "config": config,
                "stage2_stream": stream,
                "ppl_final": 21.0 + seed,
                "adapter_sha256": adapter_hash,
            },
        )
        write(
            "control",
            seed,
            {
                "schema": "gemma-sv-recovery-v2",
                "config": config,
                "stage2_stream": stream,
                "ppl_control": 20.5 + seed,
                "adapter_sha256": f"control-{seed}",
            },
        )
        write(
            "certificate",
            seed,
            {
                "seed": seed,
                "provenance": {
                    "adapter": {"content_sha256": adapter_hash}
                },
                "targets": [
                    {
                        "kl_exact_vs_refit_nats": 1e-14 * (seed + 1),
                        "kl_decay_vs_refit_nats": 1e-5 * (seed + 1),
                        "decrement_fallbacks": 0,
                    }
                ],
            },
        )
        write(
            "audit",
            seed,
            {
                "seed": seed,
                "adapter_sha256": adapter_hash,
                "stage2_stream": stream,
                "directions": {
                    "kl_proxy_vs_exact_nats": "KL(exact || proxy)"
                },
                "runs": [
                    {
                        "summary": {
                            "attempted_records": 1,
                            "admitted_records": 1,
                            "rejected_records": [],
                            "methods": {
                                "full_repack": {
                                    "completed_records": 8,
                                    "failed_records": 0,
                                    "mean_deleted_suppression_nats": 1.0
                                    + seed / 10,
                                    "mean_retained_drift_nats": 0.0,
                                    "mean_deleted_behavioral_kl_to_repack_nats": 0.0,
                                    "mean_update_median_seconds": 1.0,
                                    "mean_state_tensor_storage_bytes": 1024.0,
                                }
                            }
                        },
                        "records": [
                            {
                                "record_id": f"record-{seed}",
                                "admission": _admission_row(),
                                "prefill_once_precision_audit": {
                                    "status": "completed",
                                    "solver_diagnostics": {
                                        "head_gate_solves": 5,
                                        "decrement_fallbacks": seed,
                                    },
                                    "probe_rows": [
                                        {
                                            "kl_exact_vs_refit_nats": 1e-12
                                            * (seed + 1),
                                            "kl_proxy_vs_exact_nats": 1e-7
                                            * (seed + 1),
                                            "kl_decay_vs_refit_nats": 1e-3
                                            * (seed + 1),
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ],
            },
        )
        write(
            "cross_corpus",
            seed,
            {
                "_metadata": {"seed": seed},
                "recovered": {"wikitext": 21.0 + seed},
                "control": {"wikitext": 20.5 + seed},
            },
        )
        write(
            "zero_shot",
            seed,
            {
                "_metadata": {"seed": seed},
                "arc_easy": {
                    "recovered": 0.60 + seed / 100,
                    "control": 0.59 + seed / 100,
                },
            },
        )

    report = summarize_paths(
        paths["recovery"],
        paths["control"],
        certificate_paths=paths["certificate"],
        audit_paths=paths["audit"],
        cross_corpus_paths=paths["cross_corpus"],
        zero_shot_paths=paths["zero_shot"],
    )
    assert report["seed_count"] == 5
    assert report["perplexity"]["recovered"]["n"] == 5
    assert report["certificate"]["exact_vs_refit_kl_nats"]["n_seeds"] == 5
    assert report["audit"]["proxy_vs_exact_kl_nats"]["n_seeds"] == 5
    assert report["audit"]["exact_vs_refit_kl_nats"]["n_seeds"] == 5
    decomposition = report["deletion_baselines"]["admission"]["decomposition"]
    assert decomposition["joint_target_and_retained_whole_record"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
    }
    assert decomposition["target_only_whole_record"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
    }
    assert decomposition["deleted_fields"] == {
        "numerator": 15,
        "denominator": 15,
        "rate": 1.0,
    }
    assert decomposition["retained_neighbor_availability"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
    }
    assert decomposition["fixed_c_feasibility"] == {
        "contexts": {"numerator": 5, "denominator": 5, "rate": 1.0},
        "boundaries": {"numerator": 25, "denominator": 25, "rate": 1.0},
    }
    execution = report["deletion_baselines"]["exact_path_execution"]
    assert execution["head_gate_solves"] == 25
    assert execution["exact_refit_fallbacks"] == 10
    assert execution["decrement_successes"] == 15
    assert "not pure decrement latency" in execution["plotted_update_timing_scope"]
    assert (
        report["deletion_baselines"]["methods"]["full_repack"]["metrics"][
            "mean_deleted_suppression_nats"
        ]["n"]
        == 5
    )
    assert report["cross_corpus"]["summary"]["wikitext"]["n"] == 5
    assert report["zero_shot"]["summary"]["arc_easy"]["n"] == 5
    assert report["cross_corpus"]["mean_across_corpora"]["n"] == 5
    assert report["zero_shot"]["mean_across_tasks"]["n"] == 5

    compact = compact_summary(report, source_sha256="summary-hash")
    assert compact["contains_source_text"] is False
    assert compact["source_summary_sha256"] == "summary-hash"
    assert all("sources" not in row for row in compact["seed_rows"])


def test_admission_decomposition_recomputes_independent_components():
    admission = _admission_row(
        field_lifts=(0.2, 0.01, 0.4),
        field_ranks=(1, 2, 11),
        retained_rank=12,
        reasons=(
            "deleted_field_1:secret_lift",
            "deleted_field_2:rank",
            "retained_field:rank",
        ),
    )

    components = _recompute_admission_row(admission)

    assert components["joint_target_retained_pass"] is False
    assert components["target_only_pass"] is False
    assert components["deleted_fields_passed"] == 1
    assert components["deleted_fields_total"] == 3
    assert components["retained_neighbor_pass"] is False
    assert components["fixed_c_context_pass"] is True
    assert components["fixed_c_boundaries_passed"] == 5


def test_admission_decomposition_rejects_threshold_or_reason_drift():
    threshold_drift = _admission_row()
    threshold_drift["thresholds"]["minimum_secret_lift_nats"] = 0.051
    with pytest.raises(AggregationError, match="frozen value"):
        _recompute_admission_row(threshold_drift)

    stale_reasons = _admission_row(field_ranks=(1, 2, 11))
    with pytest.raises(AggregationError, match="stored reason codes differ"):
        _recompute_admission_row(stale_reasons)

    extra_threshold = _admission_row()
    extra_threshold["thresholds"]["unregistered_gate"] = True
    with pytest.raises(AggregationError, match="threshold keys differ"):
        _recompute_admission_row(extra_threshold)

    stale_fixed_c = _admission_row()
    stale_fixed_c["fixed_c_feasibility"][0]["retained_capacity"] = 0.5
    with pytest.raises(AggregationError, match="differs from retained capacity"):
        _recompute_admission_row(stale_fixed_c)

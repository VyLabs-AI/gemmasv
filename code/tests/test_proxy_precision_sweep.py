import numpy as np
import pytest

from gemma_sv.eval_persistent_deletion_baselines import _seed_everything
from gemma_sv.eval_proxy_precision_sweep import (
    evaluate_locked_policy,
    residual_qualifies,
    select_calibration_policy,
    svdd_residual_diagnostics,
)
from gemma_sv.publish_proxy_frontier import compact_frontier


def test_svdd_diagnostics_report_exact_feasible_solution():
    diagnostics = svdd_residual_diagnostics(
        np.eye(2),
        np.array([0.5, 0.5]),
        0.6,
        partition_cutoff=1e-3,
    )

    assert diagnostics["feasible"] is True
    assert diagnostics["equality_residual"] == pytest.approx(0.0)
    assert diagnostics["kkt_residual"] == pytest.approx(0.0)
    assert diagnostics["projected_gradient_inf"] == pytest.approx(0.0)
    assert diagnostics["duality_gap"] == pytest.approx(0.0)
    assert diagnostics["partition"]["margin"] == 2


def test_operation_seed_covers_optional_mlx_rng():
    mx = pytest.importorskip("mlx.core")
    _seed_everything(17)
    first = np.array(mx.random.normal((4, 4)))
    _seed_everything(17)
    second = np.array(mx.random.normal((4, 4)))
    assert np.array_equal(first, second)


def _diagnostics(residual):
    return {
        "feasible": True,
        "invalid_gate_rows": 0,
        "max_equality_residual": 0.0,
        "max_box_violation": 0.0,
        "max_kkt_residual": residual,
        "max_projected_gradient_inf": residual,
        "max_duality_gap": residual**2,
    }


def _observation(
    *,
    context,
    iterations,
    residual,
    kl,
    proxy_seconds,
):
    return {
        "adapter_index": 0,
        "record_index": context,
        "solver": {
            "fista_iterations": iterations,
            "partition_cutoff": 1e-3,
        },
        "diagnostics": _diagnostics(residual),
        "metrics": {
            "max_kl_refit64_to_proxy_nats": kl,
            "retained_drift_from_refit64_nats": 0.02,
        },
        "timing": {
            "proxy_update_seconds": proxy_seconds,
            "proxy_query_seconds": 0.1,
            "shared_exact_refit_update_seconds": 4.0,
            "refit_query_seconds": 0.1,
        },
    }


def test_residual_gate_rejects_invalid_or_underconverged_rows():
    assert residual_qualifies(_diagnostics(1e-5), 3e-5)
    assert not residual_qualifies(_diagnostics(1e-3), 3e-5)
    invalid = _diagnostics(0.0)
    invalid["invalid_gate_rows"] = 1
    assert not residual_qualifies(invalid, 3e-5)


def test_calibration_selects_fastest_zero_violation_policy():
    observations = [
        # The 20-iteration policy must fall back on context 0 to avoid KL > 1e-3.
        _observation(
            context=0,
            iterations=20,
            residual=1e-3,
            kl=2e-3,
            proxy_seconds=0.5,
        ),
        _observation(
            context=2,
            iterations=20,
            residual=0.0,
            kl=5e-4,
            proxy_seconds=0.5,
        ),
        # The 40-iteration policy passes both contexts without fallback.
        _observation(
            context=0,
            iterations=40,
            residual=0.0,
            kl=8e-4,
            proxy_seconds=1.0,
        ),
        _observation(
            context=2,
            iterations=40,
            residual=0.0,
            kl=7e-4,
            proxy_seconds=1.0,
        ),
    ]

    selection = select_calibration_policy(observations)

    assert selection["status"] == "selected"
    assert selection["selected_policy"]["fista_iterations"] == 40
    assert selection["selected_policy"]["max_reference_kl_nats"] == 1e-3


def test_locked_policy_reports_fallback_and_zero_hybrid_kl():
    policy = {
        "fista_iterations": 20,
        "partition_cutoff": 1e-3,
        "residual_cutoff": 3e-5,
        "max_reference_kl_nats": 1e-3,
    }
    observation = _observation(
        context=1,
        iterations=20,
        residual=1e-2,
        kl=0.5,
        proxy_seconds=0.5,
    )

    summary = evaluate_locked_policy([observation], policy)

    assert summary["fallbacks"] == 1
    assert summary["context_violations"] == 0
    assert summary["max_kl_refit64_to_hybrid_nats"] == 0.0


def test_calibration_rejects_trivial_all_fallback_policy():
    observation = _observation(
        context=0,
        iterations=20,
        residual=1e-2,
        kl=0.5,
        proxy_seconds=0.5,
    )

    selection = select_calibration_policy([observation])

    assert selection["status"] == "negative_frontier"
    assert selection["selected_policy"] is None


def test_compact_frontier_keeps_selected_policy_and_locked_result():
    policy = {
        "fista_iterations": 40,
        "partition_cutoff": 1e-3,
        "residual_cutoff": 3e-3,
        "max_reference_kl_nats": 1e-3,
    }
    observation = _observation(
        context=0,
        iterations=40,
        residual=0.0,
        kl=8e-4,
        proxy_seconds=1.0,
    )
    observation["peak_memory"] = {"process_peak_resident_bytes": 1234}
    calibration_row = {
        "policy": policy,
        "contexts": 1,
        "context_violations": 0,
        "fallbacks": 0,
        "fallback_rate": 0.0,
        "max_kl_refit64_to_hybrid_nats": 8e-4,
        "mean_retained_abs_drift_nats": 0.02,
        "mean_update_seconds": 1.0,
        "mean_query_seconds": 0.1,
        "mean_end_to_end_seconds": 1.1,
    }
    report = {
        "status": "completed",
        "claim_scope": "residual-qualified approximation; not a certificate",
        "manifest": {
            "path": "/tmp/work/gemma_sv/benchmarks/whole_record.json",
            "sha256": "manifest",
        },
        "split": {"locked_before_evaluation": True},
        "grid": {"fista_iterations": [40]},
        "calibration": {
            "observations": [observation],
            "selection": {
                "selection_rule": "fixed",
                "selected_policy": policy,
                "frontier": [calibration_row],
            },
        },
        "validation": {
            "observations": [observation],
            "locked_policy_result": {
                "policy": policy,
                "contexts": 1,
                "context_violations": 0,
                "fallbacks": 0,
                "fallback_rate": 0.0,
                "max_kl_refit64_to_hybrid_nats": 8e-4,
                "mean_retained_abs_drift_nats": 0.02,
                "mean_end_to_end_seconds": 1.1,
                "rows": [{"record_index": 0}],
            },
        },
    }

    compact = compact_frontier(report, source_sha256="source")

    assert compact["source_report_sha256"] == "source"
    assert compact["selected_policy"] == policy
    assert compact["validation"]["context_violations"] == 0
    assert "rows" not in compact["validation"]
    assert compact["calibration"]["observation_summary"]["contexts"] == 1
    assert compact["manifest"]["path"] == (
        "gemma_sv/benchmarks/whole_record.json"
    )

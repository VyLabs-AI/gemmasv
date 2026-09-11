import numpy as np
import pytest
import torch

from cp_svm import FastOneClassSVM, SVDDQPInfeasibleError, solve_svdd_qp
from gemma_sv.demo_server import certificate as certificate_module
from gemma_sv.demo_server.certificate import (
    CertificateFeasibilityError,
    certificate_overrides,
    fixed_c_feasibility,
    median_kernel_width,
)
from svattn.causal_sv_attention import (
    causal_sv_readout_mlx_batched,
    sv_softmax_decode_step,
)


def test_fixed_c_preflight_detects_early_record_infeasibility():
    C, diagnostics = fixed_c_feasibility(
        842,
        range(23),
        nu=0.3,
        chunk=128,
    )
    assert C == pytest.approx(1.0 / (0.3 * 842))
    assert diagnostics[0]["start"] == 256
    assert diagnostics[0]["retained_count"] == 233
    assert diagnostics[0]["feasible"] is False

    _, later = fixed_c_feasibility(
        842,
        range(300, 323),
        nu=0.3,
        chunk=128,
    )
    assert all(item["feasible"] for item in later)


def test_per_boundary_box_has_deleted_fraction_contract_and_fallback():
    boxes, diagnostics = fixed_c_feasibility(
        80,
        [5, 6],
        nu=0.7,
        chunk=16,
        per_boundary_box=True,
    )
    assert boxes[16] == pytest.approx(1.0 / (0.7 * 16))
    assert diagnostics[0]["deleted_fraction"] == pytest.approx(2 / 16)
    assert diagnostics[0]["maximum_deleted_fraction"] == pytest.approx(0.3)
    assert diagnostics[0]["fallback"] == "none"

    _, failed = fixed_c_feasibility(
        80,
        list(range(6)),
        nu=0.7,
        chunk=16,
        per_boundary_box=True,
    )
    assert failed[0]["feasible"] is False
    assert failed[0]["fallback"] == "full_repack"


def test_certificate_rejects_infeasible_fixed_c_before_solving():
    with pytest.raises(CertificateFeasibilityError) as raised:
        certificate_overrides(
            {0: np.zeros((1, 842, 2))},
            [0],
            list(range(23)),
            842,
            1,
        )
    assert raised.value.diagnostics[0]["feasible"] is False


def test_batch_qp_reports_true_box_infeasibility():
    with pytest.raises(SVDDQPInfeasibleError, match=r"n\*C=.*< 1"):
        solve_svdd_qp(np.eye(3), C=0.2)


def _random_keys(total_tokens=80):
    return {0: np.random.default_rng(7).normal(size=(1, total_tokens, 3))}


def test_disjoint_multi_point_certificate_matches_refit_functionally():
    forget = [5, 6, 40, 41]
    result = certificate_overrides(
        _random_keys(),
        [0],
        forget,
        80,
        1,
        chunk=16,
    )
    assert result["fixed_c_feasible"] is True
    assert result["max_functional_deviation"] <= result["functional_tolerance"]
    for boundary, alpha in result["exact"][0].items():
        forgotten_here = [position for position in forget if position < boundary]
        assert np.all(alpha[0, forgotten_here] == 0.0)


def test_per_boundary_boxes_drive_decrement_and_refit_per_solve():
    result = certificate_overrides(
        _random_keys(),
        [0],
        [5, 6],
        80,
        1,
        nu=0.7,
        chunk=16,
        per_boundary_box=True,
    )
    boxes = {
        int(start): value
        for start, value in result["box_C_by_boundary"].items()
    }
    assert len(set(boxes.values())) == len(boxes)
    assert boxes[16] == pytest.approx(1.0 / (0.7 * 16))
    assert result["max_functional_deviation"] <= result["functional_tolerance"]
    assert set(result["exact"][0]) == set(boxes)
    assert set(result["refit"][0]) == set(boxes)


def test_mass_rescaled_full_and_cached_outputs_match_refit():
    pytest.importorskip("mlx.core")
    rng = np.random.default_rng(17)
    keys_np = rng.normal(size=(1, 32, 3))
    keys = torch.tensor(keys_np, dtype=torch.float64)
    values = torch.tensor(
        rng.normal(size=(1, 32, 4)),
        dtype=torch.float64,
    )
    queries = torch.tensor(
        rng.normal(size=(1, 32, 3)),
        dtype=torch.float64,
    )
    result = certificate_overrides(
        {0: keys_np},
        [0],
        [5],
        32,
        1,
        nu=0.7,
        chunk=8,
        per_boundary_box=True,
    )
    boxes = {
        int(start): value
        for start, value in result["box_C_by_boundary"].items()
    }

    full = {}
    for case in ("exact", "refit"):
        full[case] = causal_sv_readout_mlx_batched(
            keys,
            values,
            queries,
            boxes,
            2.0,
            8,
            readout="softmax",
            alpha_override=result[case][0],
            drop_pos=[5],
            preserve_prefix_mass=True,
        )
    assert torch.allclose(full["exact"], full["refit"], atol=1e-6)

    query = torch.tensor(
        rng.normal(size=(1, 1, 3)),
        dtype=torch.float64,
    )
    cached = {}
    for case in ("exact", "refit"):
        cached[case] = sv_softmax_decode_step(
            keys,
            values,
            query,
            {
                24: (
                    torch.tensor(
                        result[case][0][24],
                        dtype=torch.float64,
                    ),
                    torch.ones(1, dtype=torch.bool),
                )
            },
            8,
            boundary_override=24,
            drop_pos=[5],
            preserve_prefix_mass=True,
        )
    assert torch.allclose(cached["exact"], cached["refit"], atol=1e-6)


def test_decrement_path_failure_uses_disclosed_exact_refit_fallback(monkeypatch):
    def fail_remove(self, positions):
        raise RuntimeError("synthetic decrement edge case")

    monkeypatch.setattr(FastOneClassSVM, "remove_points", fail_remove)
    result = certificate_overrides(
        _random_keys(),
        [0],
        [40, 41],
        80,
        1,
        chunk=16,
    )
    assert result["used_refit_fallback"] is True
    assert result["n_fallback"] > 0
    assert all(
        np.array_equal(result["exact"][0][start], result["refit"][0][start])
        for start in result["exact"][0]
        if start > 40
    )
    assert {
        detail["reason"] for detail in result["fallback_details"]
    } == {"synthetic decrement edge case"}


def test_frozen_session_objective_bypasses_median_and_preserves_source(monkeypatch):
    keys = _random_keys()
    before = keys[0].copy()

    def fail_median(_keys):
        raise AssertionError("the frozen objective must not recompute bandwidth")

    monkeypatch.setattr(certificate_module, "median_kernel_width", fail_median)
    result = certificate_overrides(
        keys,
        [0],
        [40, 41],
        80,
        1,
        chunk=16,
        box_C=0.05,
        kpar_by_layer={0: 1.75},
        skip_incremental=True,
    )

    assert np.array_equal(keys[0], before)
    assert result["box_C"] == pytest.approx(0.05)
    assert result["objective"] == {
        "box_C": pytest.approx(0.05),
        "box_C_by_boundary": None,
        "box_source": "frozen_decode_session",
        "bandwidth_source": "frozen_decode_session",
        "kpar_by_layer": {0: pytest.approx(1.75)},
    }


def test_explicit_objective_matches_legacy_when_bandwidth_and_box_coincide():
    keys = _random_keys(total_tokens=32)
    C = 1.0 / (0.3 * 32)
    width = median_kernel_width(keys[0][0, :16])
    legacy = certificate_overrides(
        keys,
        [0],
        [5],
        32,
        1,
        chunk=16,
        skip_incremental=True,
    )
    frozen = certificate_overrides(
        keys,
        [0],
        [5],
        32,
        1,
        chunk=16,
        box_C=C,
        kpar_by_layer={0: width},
        skip_incremental=True,
    )

    for case in ("exact", "refit", "decay"):
        assert set(legacy[case][0]) == set(frozen[case][0])
        for boundary in legacy[case][0]:
            assert np.array_equal(
                legacy[case][0][boundary],
                frozen[case][0][boundary],
            )

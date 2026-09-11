import numpy as np

from experiments.forgetting_rigor import one_trial


def test_fixed_c_infeasibility_is_classified_before_solving():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    result = one_trial(
        X,
        rng,
        nu=0.8,
        kpar=2.0,
        n_forget=2,
        forget=[0, 1],
    )
    assert result["status"] == "failed"
    assert result["failure"] == "fixed_c_preflight:infeasible"
    assert result["retained_capacity"] < 1.0


def test_decrement_failure_still_records_successful_reference_refit():
    rng = np.random.default_rng(36)
    X = rng.normal(size=(8, 3))
    result = one_trial(
        X,
        rng,
        nu=0.4,
        kpar=2.0,
        n_forget=3,
        forget=[3, 4, 5],
    )
    assert result["retained_capacity"] > 1.0
    assert result["refit_status"] == "completed"
    assert result["status"] == "failed"
    assert result["failure"] == "decrement:margin_empty"

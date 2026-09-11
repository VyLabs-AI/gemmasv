import pytest

from gemma_sv.summarize_boundary_evidence import paired_bootstrap


def test_record_bootstrap_is_deterministic_and_record_scoped():
    first = paired_bootstrap([0.0, 0.1, -0.1], draws=1000, seed=7)
    second = paired_bootstrap([0.0, 0.1, -0.1], draws=1000, seed=7)

    assert first == second
    assert first["mean"] == pytest.approx(0.0)
    assert first["unit"] == "whole record"
    assert first["n"] == 3
    assert first["ci95"][0] <= 0 <= first["ci95"][1]

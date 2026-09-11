import math

import pytest

from gemma_sv.demo_server.contract import (
    AdmissionStatus,
    CertificateBand,
    classify_admission,
    classify_certificate,
    kl_from_log_probs,
    sequence_log_score,
)


def test_sequence_log_score_reports_total_and_per_token_mean():
    rows = [
        [math.log(0.8), math.log(0.2)],
        [math.log(0.25), math.log(0.75)],
    ]
    total, mean = sequence_log_score(rows, [0, 1])
    assert total == pytest.approx(math.log(0.8) + math.log(0.75))
    assert mean == pytest.approx(total / 2)


def test_sequence_log_score_rejects_misaligned_targets():
    with pytest.raises(ValueError, match="one log-probability row"):
        sequence_log_score([[0.0]], [0, 0])


def test_kl_from_log_probs_is_zero_for_identical_distribution():
    lp = [math.log(0.2), math.log(0.8)]
    assert kl_from_log_probs(lp, lp) == pytest.approx(0.0, abs=1e-15)


def test_admission_requires_behavior_and_probability_lift():
    recalled = classify_admission(-1.0, -3.0, greedy_match=True)
    assert recalled.status is AdmissionStatus.RECALLED
    assert recalled.probability_ratio == pytest.approx(math.exp(2.0))

    weak = classify_admission(-2.0, -2.5, greedy_match=False)
    assert weak.status is AdmissionStatus.WEAKLY_RECALLED

    absent = classify_admission(-2.4, -2.5, greedy_match=True)
    assert absent.status is AdmissionStatus.NOT_RECALLED


def test_admission_flags_ill_conditioned_span():
    result = classify_admission(-1.0, -3.0, greedy_match=True, condition_number=1e12)
    assert result.status is AdmissionStatus.ILL_CONDITIONED


@pytest.mark.parametrize(
    ("value", "band"),
    [
        (5e-15, CertificateBand.MACHINE_PRECISION),
        (8.3e-7, CertificateBand.SPAN_CONDITIONED),
        (4e-2, CertificateBand.OUTSIDE_DEMO_TOLERANCE),
        (float("nan"), CertificateBand.INVALID),
    ],
)
def test_certificate_bands_always_remain_probe_scoped(value, band):
    result = classify_certificate(value)
    assert result.band is band
    assert result.probe_scoped is True

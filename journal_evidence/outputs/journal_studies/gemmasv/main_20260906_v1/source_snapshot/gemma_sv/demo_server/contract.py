"""Scientific display contract for the hosted hero demo.

The UI exposes three different kinds of evidence and must not collapse them:

* behavioral scores from the fast inference path;
* a prompt-specific output KL from the float64 certificate path; and
* empirical attack outcomes.

These helpers are dependency-free so both the API and tests use the same labels
and thresholds.  A certificate band is descriptive, not a theorem threshold:
the measured value is always shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence


class AdmissionStatus(str, Enum):
    """Whether a custom fact is suitable for the guided narrative."""

    RECALLED = "recalled"
    WEAKLY_RECALLED = "weakly_recalled"
    NOT_RECALLED = "not_recalled"
    ILL_CONDITIONED = "ill_conditioned"


class CertificateBand(str, Enum):
    """Human-readable numerical regime for a measured probe KL."""

    MACHINE_PRECISION = "machine_precision"
    SPAN_CONDITIONED = "span_conditioned"
    OUTSIDE_DEMO_TOLERANCE = "outside_demo_tolerance"
    INVALID = "invalid"


@dataclass(frozen=True)
class AdmissionResult:
    status: AdmissionStatus
    log_lift_nats: float
    probability_ratio: float
    greedy_match: bool
    message: str


@dataclass(frozen=True)
class CertificateResult:
    band: CertificateBand
    kl_nats: float
    probe_scoped: bool
    message: str


def sequence_log_score(
    token_log_probs: Sequence[Sequence[float]],
    target_token_ids: Sequence[int],
) -> tuple[float, float]:
    """Return total and length-normalized log probability of a target span.

    ``token_log_probs[t][v]`` is the log probability assigned to vocabulary item
    ``v`` at teacher-forced step ``t``.  This is a behavioral exposure score, not
    a certificate.
    """

    if not target_token_ids:
        raise ValueError("target_token_ids must not be empty")
    if len(token_log_probs) != len(target_token_ids):
        raise ValueError("one log-probability row is required per target token")

    total = 0.0
    for row, token_id in zip(token_log_probs, target_token_ids):
        if token_id < 0 or token_id >= len(row):
            raise ValueError(f"target token id {token_id} is outside the vocabulary")
        value = float(row[token_id])
        if not math.isfinite(value):
            raise ValueError("target log probability must be finite")
        total += value
    return total, total / len(target_token_ids)


def kl_from_log_probs(log_p: Sequence[float], log_q: Sequence[float]) -> float:
    """Compute ``KL(P || Q)`` from normalized log-probability vectors."""

    if len(log_p) != len(log_q) or not log_p:
        raise ValueError("log-probability vectors must have the same non-zero length")
    kl = 0.0
    for lp, lq in zip(log_p, log_q):
        lp, lq = float(lp), float(lq)
        if not (math.isfinite(lp) and math.isfinite(lq)):
            raise ValueError("log probabilities must be finite")
        probability = math.exp(lp)
        if probability:
            kl += probability * (lp - lq)
    # Roundoff can produce a tiny negative value for effectively equal vectors.
    return max(0.0, kl)


def classify_admission(
    keep_mean_log_prob: float,
    floor_mean_log_prob: float,
    *,
    greedy_match: bool,
    condition_number: float | None = None,
    weak_lift_nats: float = 0.25,
    recalled_lift_nats: float = 1.0,
    max_condition_number: float = 1e10,
) -> AdmissionResult:
    """Classify whether a custom fact can support the guided recall beat.

    The thresholds are product gates, not statistical significance levels.  The
    raw score and floor are retained by the API and shown to the visitor.
    """

    if condition_number is not None and (
        not math.isfinite(condition_number) or condition_number > max_condition_number
    ):
        return AdmissionResult(
            AdmissionStatus.ILL_CONDITIONED,
            float("nan"),
            float("nan"),
            greedy_match,
            "The selected span is numerically ill-conditioned; reformulate or deduplicate it.",
        )

    lift = float(keep_mean_log_prob) - float(floor_mean_log_prob)
    ratio = math.exp(min(lift, 700.0))
    if greedy_match and lift >= recalled_lift_nats:
        status = AdmissionStatus.RECALLED
        message = "The model recalls the selected span above its never-contained floor."
    elif lift >= weak_lift_nats:
        status = AdmissionStatus.WEAKLY_RECALLED
        message = "The selected span is detectable but too weak for the guided claim."
    else:
        status = AdmissionStatus.NOT_RECALLED
        message = "The model did not reliably recall this span; try a clearer synthetic fact."
    return AdmissionResult(status, lift, ratio, greedy_match, message)


def classify_certificate(
    kl_nats: float,
    *,
    machine_precision_max: float = 1e-10,
    span_conditioned_max: float = 1e-3,
) -> CertificateResult:
    """Describe a measured KL without turning a product threshold into a proof."""

    kl_nats = float(kl_nats)
    if not math.isfinite(kl_nats) or kl_nats < 0:
        return CertificateResult(
            CertificateBand.INVALID,
            kl_nats,
            True,
            "The certificate job did not return a valid non-negative KL.",
        )
    if kl_nats <= machine_precision_max:
        band = CertificateBand.MACHINE_PRECISION
        message = "At this audit probe, decrement and refit agree near machine precision."
    elif kl_nats <= span_conditioned_max:
        band = CertificateBand.SPAN_CONDITIONED
        message = "At this audit probe, the multi-token span is within the conditioned demo band."
    else:
        band = CertificateBand.OUTSIDE_DEMO_TOLERANCE
        message = "This measured probe KL is outside the demo band; no exactness badge is shown."
    return CertificateResult(band, kl_nats, True, message)
